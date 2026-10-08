import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, ModuleDict, LayerNorm, Dropout
from torch_geometric.nn import global_mean_pool, to_hetero
from stage1.gnns import load_gnn_model, DeepResidualBlock
from torch_geometric.utils import negative_sampling

class HeteroCodeGraphEncoder(torch.nn.Module):
    def __init__(self, metadata, in_channels, hidden_channels, out_channels, gnn_type='gat',
                 num_layers=2, dropout=0.1, num_heads=8):
        super(HeteroCodeGraphEncoder, self).__init__()
        self.metadata = metadata
        self.in_channels = in_channels

        # 1. 初始化基础 GNN
        base_gnn = load_gnn_model[gnn_type](
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads
        )
        self.hetero_gnn = to_hetero(base_gnn, metadata, aggr='sum')
        # 定义节点类别映射（确保与 Trainer 中的逻辑一致）
        self.logic_types = ['expression']  # 逻辑节点
        self.semantic_types = ['variable'] # 语义节点

        # 2. 深度非对称节点特征重构器
        decoders = {}
        for node_type in metadata[0]:
            if node_type in self.logic_types or node_type in ['structure', 'statement']: # 💡 结构节点也需要深度重构
                layers = [Linear(out_channels, hidden_channels), ReLU()]
                for _ in range(6): # 加深
                    layers.append(DeepResidualBlock(hidden_channels, dropout=dropout))
                layers.append(Linear(hidden_channels, in_channels))
                decoders[node_type] = Sequential(*layers)
            else:
                # 针对其他节点（如语义）：保持轻量（2-3层），防止过拟合
                layers = [Linear(out_channels, hidden_channels), ReLU()]
                for _ in range(2): 
                    layers.append(DeepResidualBlock(hidden_channels, dropout=dropout))
                layers.append(Linear(hidden_channels, in_channels))
                decoders[node_type] = Sequential(*layers)
        
        self.feature_decoders = ModuleDict(decoders)

        # 3. 修改链路预测分类器：增加一个“逻辑错误”类别
        self.SIMPLE_REL_MAP = {
            'ast': 1,
            'cfg': 2,
            'dfg': 3
        }
        self.num_edge_classes = 5
        self.LOGIC_ERR_IDX = 4

        # 唯一的分类器，现在只输出 5 个维度
        self.edge_classifier = Sequential(
            Linear(out_channels * 2, hidden_channels),
            LayerNorm(hidden_channels),
            ReLU(),
            Linear(hidden_channels, self.num_edge_classes)
        )

        # 4. 图级投射层
        self.graph_projector = Sequential(
            Linear(out_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, out_channels)
        )

        self.path_predictor = Sequential(
            Linear(out_channels * 2, hidden_channels),
            LayerNorm(hidden_channels),
            ReLU(),
            Linear(hidden_channels, 1)
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, batch_dict=None, num_graphs=None):
        device = next(iter(x_dict.values())).device
        fixed_x_dict = {nt: x_dict.get(nt, torch.zeros((0, self.in_channels), device=device)) for nt in
                        self.metadata[0]}
        fixed_edge_index_dict = {et: edge_index_dict.get(et, torch.empty((2, 0), dtype=torch.long, device=device)) for
                                 et in self.metadata[1]}

        # 获取节点 Embedding
        node_emb_dict = self.hetero_gnn(fixed_x_dict, fixed_edge_index_dict)

        # 计算图级指纹
        total_graph_emb = None
        if batch_dict is not None:
            out_dim = next(iter(node_emb_dict.values())).size(-1)
            bs = num_graphs if num_graphs else (
                        max([b.max().item() for b in batch_dict.values() if b is not None and b.numel() > 0]) + 1)
            combined_emb = torch.zeros((bs, out_dim), device=device)
            counts = torch.zeros((bs, 1), device=device)
            for nt, emb in node_emb_dict.items():
                if nt in batch_dict and batch_dict[nt] is not None and emb.size(0) > 0:
                    combined_emb += global_mean_pool(emb, batch_dict[nt], size=bs)
                    counts[batch_dict[nt].unique()] += 1.0
            total_graph_emb = self.graph_projector(combined_emb / torch.clamp(counts, min=1.0))

        return node_emb_dict, total_graph_emb, {}  # Aux 逻辑移至 Trainer 实现

    def predict_edge_types_unified(self, node_emb_dict, edge_index_dict, cfg_key=None, max_samples_per_type=256):
        all_logits, all_targets = [], []
        device = next(iter(node_emb_dict.values())).device

        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.numel() == 0: continue

            src_t, rel, dst_t = edge_type
            rel_lower = rel.lower()

            # 找到当前边属于哪一个简化的核心类 (AST/CFG/DFG)
            target_label = 0
            for key, label in self.SIMPLE_REL_MAP.items():
                if key in rel_lower:
                    target_label = label
                    break

            # 如果没匹配上或者是 AST，根据你的需求决定是否跳过
            if target_label == 0 or target_label == 1: continue

            num_samples = min(edge_index.size(1), max_samples_per_type)
            perm = torch.randperm(edge_index.size(1), device=device)[:num_samples]
            pos_idx = edge_index[:, perm]

            # 正样本预测
            z_src = node_emb_dict[src_t][pos_idx[0]]
            z_dst = node_emb_dict[dst_t][pos_idx[1]]
            logits = self.edge_classifier(torch.cat([z_src, z_dst], dim=1))

            all_logits.append(logits)
            all_targets.append(torch.full((logits.size(0),), target_label, device=device))

            # 负采样 (Label 0)
            neg_edge_index = negative_sampling(
                edge_index=edge_index,
                num_nodes=(node_emb_dict[src_t].size(0), node_emb_dict[dst_t].size(0)),
                num_neg_samples=num_samples,
                method='sparse'
            )
            if neg_edge_index.numel() > 0:
                zn_src = node_emb_dict[src_t][neg_edge_index[0]]
                zn_dst = node_emb_dict[dst_t][neg_edge_index[1]]
                all_logits.append(self.edge_classifier(torch.cat([zn_src, zn_dst], dim=1)))
                all_targets.append(torch.zeros(neg_edge_index.size(1), dtype=torch.long, device=device))

            # 逻辑错误采样 (Label 4)
            if target_label == 2:  # 仅对 CFG 进行反向采样
                rev_samples = max(1, num_samples // 4)
                z_rev_src = node_emb_dict[dst_t][pos_idx[1][:rev_samples]]
                z_rev_dst = node_emb_dict[src_t][pos_idx[0][:rev_samples]]
                all_logits.append(self.edge_classifier(torch.cat([z_rev_src, z_rev_dst], dim=1)))
                all_targets.append(torch.full((rev_samples,), self.LOGIC_ERR_IDX, device=device))

        if len(all_logits) == 0:
            return (torch.empty((0, self.num_edge_classes), device=device),
                    torch.empty((0,), dtype=torch.long, device=device))

        return torch.cat(all_logits) / 0.5, torch.cat(all_targets)

    def predict_logic_consistency(self, z_src, z_dst):
        """预测 z_src -> z_dst 是否为正确的逻辑顺序"""
        return self.logic_consistency_predictor(torch.cat([z_src, z_dst], dim=-1))

    def reconstruct_node_features(self, node_emb_dict):
        recons = {}
        for nt, emb in node_emb_dict.items():
            if emb.size(0) > 0:
                pred = self.feature_decoders[nt](emb)
                # 💡 建议：对预测值进行初步归一化，使其分布更接近预训练模型的 Embedding
                recons[nt] = pred 
        return recons

    def predict_edge_types(self, node_emb_dict, edge_index_dict, neg_edge_index_dict=None):
        all_logits, all_targets = [], []
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.numel() == 0: continue
            src_t, rel, dst_t = edge_type
            key = f"{src_t}_{rel}_{dst_t}"
            logits = self.edge_classifier(
                torch.cat([node_emb_dict[src_t][edge_index[0]], node_emb_dict[dst_t][edge_index[1]]], dim=1))
            all_logits.append(logits)
            all_targets.append(torch.full((logits.size(0),), self.rel_to_idx[key], device=logits.device))

        if neg_edge_index_dict is not None:
            for edge_type, neg_idx in neg_edge_index_dict.items():
                if neg_idx.numel() == 0: continue
                src_t, _, dst_t = edge_type
                logits = self.edge_classifier(
                    torch.cat([node_emb_dict[src_t][neg_idx[0]], node_emb_dict[dst_t][neg_idx[1]]], dim=1))
                all_logits.append(logits);
                all_targets.append(torch.zeros(logits.size(0), dtype=torch.long, device=logits.device))
        return torch.cat(all_logits), torch.cat(all_targets)