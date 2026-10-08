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

        base_gnn = load_gnn_model[gnn_type](
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads
        )
        self.hetero_gnn = to_hetero(base_gnn, metadata, aggr='sum')
        self.logic_types = ['expression']  
        self.semantic_types = ['variable'] 

        decoders = {}
        for node_type in metadata[0]:
            if node_type in self.logic_types or node_type in ['structure', 'statement']:
                layers = [Linear(out_channels, hidden_channels), ReLU()]
                for _ in range(4): # 工业落地建议调至 4 层，降低深度以防过拟合
                    layers.append(DeepResidualBlock(hidden_channels, dropout=0.05))
                layers.append(Linear(hidden_channels, in_channels))
                decoders[node_type] = Sequential(*layers)
            else:
                decoders[node_type] = Sequential(
                    Linear(out_channels, hidden_channels // 2),
                    ReLU(),
                    Linear(hidden_channels // 2, in_channels)
                )
        self.feature_decoders = ModuleDict(decoders)

        self.SIMPLE_REL_MAP = {'ast': 1, 'cfg': 2, 'dfg': 3}
        self.num_edge_classes = 4  # 0: Neg, 1: AST, 2: CFG, 3: DFG

        self.edge_classifier = Sequential(
            Linear(out_channels * 2, hidden_channels),
            LayerNorm(hidden_channels),
            ReLU(),
            Linear(hidden_channels, self.num_edge_classes)
        )

        self.direction_classifier = Sequential(
            Linear(out_channels * 2, hidden_channels // 2),
            LayerNorm(hidden_channels // 2),
            ReLU(),
            Linear(hidden_channels // 2, 2) 
        )

        self.graph_projector = Sequential(
            Linear(out_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, out_channels)
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, batch_dict=None, num_graphs=None):
        device = next(iter(x_dict.values())).device
        fixed_x_dict = {nt: x_dict.get(nt, torch.zeros((0, self.in_channels), device=device)) for nt in self.metadata[0]}
        fixed_edge_index_dict = {et: edge_index_dict.get(et, torch.empty((2, 0), dtype=torch.long, device=device)) for et in self.metadata[1]}

        node_emb_dict = self.hetero_gnn(fixed_x_dict, fixed_edge_index_dict)

        total_graph_emb = None
        if batch_dict is not None:
            out_dim = next(iter(node_emb_dict.values())).size(-1)
            bs = num_graphs if num_graphs else (max([b.max().item() for b in batch_dict.values() if b is not None and b.numel() > 0]) + 1)
            combined_emb = torch.zeros((bs, out_dim), device=device)
            counts = torch.zeros((bs, 1), device=device)
            for nt, emb in node_emb_dict.items():
                if nt in batch_dict and batch_dict[nt] is not None and emb.size(0) > 0:
                    combined_emb += global_mean_pool(emb, batch_dict[nt], size=bs)
                    counts[batch_dict[nt].unique()] += 1.0
            total_graph_emb = self.graph_projector(combined_emb / torch.clamp(counts, min=1.0))

        return node_emb_dict, total_graph_emb, {}

    def predict_edge_types_unified(self, node_emb_dict, edge_index_dict, cfg_key=None, batch_dict=None):
        all_type_logits, all_type_targets = [], []
        all_dir_logits, all_dir_targets = [], []
        device = next(iter(node_emb_dict.values())).device
    
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.numel() == 0: continue
            src_t, rel, dst_t = edge_type
            
            target_label = 0
            for key, label in self.SIMPLE_REL_MAP.items():
                if key in rel.lower():
                    target_label = label
                    break
            if target_label == 0: continue 
    
            # 正样本随机采样以实现动态平衡
            num_samples = min(edge_index.size(1), 64 if target_label == 1 else 256)
            perm = torch.randperm(edge_index.size(1), device=device)[:num_samples]
            pos_idx = edge_index[:, perm]
    
            z_src = node_emb_dict[src_t][pos_idx[0]]
            z_dst = node_emb_dict[dst_t][pos_idx[1]]
            feats = torch.cat([z_src, z_dst], dim=1)
    
            all_type_logits.append(self.edge_classifier(feats))
            all_type_targets.append(torch.full((feats.size(0),), target_label, device=device))
            
            # 【修复点】：使负采样数量与当前筛选后的正样本数量严格保持 1:1 绝对平衡
            neg_edge_index = negative_sampling(
                edge_index=edge_index,
                num_nodes=(node_emb_dict[src_t].size(0), node_emb_dict[dst_t].size(0)),
                num_neg_samples=pos_idx.size(1),
                method='sparse'
            )
            if neg_edge_index.numel() > 0:
                zn_src = node_emb_dict[src_t][neg_edge_index[0]]
                zn_dst = node_emb_dict[dst_t][neg_edge_index[1]]
                all_type_logits.append(self.edge_classifier(torch.cat([zn_src, zn_dst], dim=1)))
                all_type_targets.append(torch.zeros(neg_edge_index.size(1), dtype=torch.long, device=device))
    
            # 任务 B: 方向/逻辑一致性预测 (CFG)
            if target_label == 2:
                all_dir_logits.append(self.direction_classifier(feats))
                all_dir_targets.append(torch.zeros(feats.size(0), dtype=torch.long, device=device))
    
                logic_err_feats = []
                # 策略 A: 1-hop 逆转
                logic_err_feats.append(torch.cat([z_dst, z_src], dim=1))
    
                # 策略 B: 同图不连贯跳跃
                if batch_dict is not None and pos_idx.size(1) > 1:
                    src_batch = batch_dict[src_t][pos_idx[0]]
                    dst_batch = batch_dict[dst_t][pos_idx[1]]
                    shift_idx = torch.roll(torch.arange(pos_idx.size(1), device=device), shifts=1)
                    same_graph_mask = (dst_batch == src_batch[shift_idx])
                    if same_graph_mask.any():
                        z_path_src = node_emb_dict[dst_t][pos_idx[1][same_graph_mask]]
                        z_path_dst = node_emb_dict[src_t][pos_idx[0][shift_idx][same_graph_mask]]
                        logic_err_feats.append(torch.cat([z_path_src, z_path_dst], dim=1))
    
                if logic_err_feats:
                    combined_err = torch.cat(logic_err_feats)
                    all_dir_logits.append(self.direction_classifier(combined_err))
                    all_dir_targets.append(torch.ones(combined_err.size(0), dtype=torch.long, device=device))
    
        res_type = (torch.cat(all_type_logits), torch.cat(all_type_targets)) if all_type_logits else (None, None)
        res_dir = (torch.cat(all_dir_logits), torch.cat(all_dir_targets)) if all_dir_logits else (None, None)
        return res_type, res_dir

    def reconstruct_node_features(self, node_emb_dict):
        recons = {}
        for nt, emb in node_emb_dict.items():
            if emb.size(0) > 0:
                recons[nt] = self.feature_decoders[nt](emb)
        return recons