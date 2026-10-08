import os
import torch
import pandas as pd
import gc
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer

from stage1.HeteroCodeGraphEncoder import HeteroCodeGraphEncoder
from python_code_graph import get_model_metadata

class Stage2DataGenerator:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_path'])

        # 2. 初始化编码器结构
        node_types, edge_types = get_model_metadata()
        self.encoder = HeteroCodeGraphEncoder(
            metadata=(node_types, edge_types),
            in_channels=768,
            hidden_channels=config['hidden_channels'],
            out_channels=config['out_channels'],
            gnn_type='gat',
            num_layers=config.get('num_layers', 4),
            num_heads=config.get('num_heads', 8)
        ).to(self.device)

        # 手动触发延迟初始化
        self._trigger_lazy_init(node_types, edge_types)

        # 3. 加载权重
        print(f"Loading checkpoint from {config['stage1_checkpoint']}...")
        checkpoint = torch.load(config['stage1_checkpoint'], map_location=self.device)
        self.encoder.load_state_dict(checkpoint)
        self.encoder.eval()

        self.TYPE_MAPPING = {
            'structural': ['structure', 'statement'],
            'logical': ['expression'],
            'semantic': ['variable']
        }

    def _trigger_lazy_init(self, node_types, edge_types):
        dummy_x = {nt: torch.randn(2, 768).to(self.device) for nt in node_types}
        dummy_edge = {et: torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(self.device) for et in edge_types}
        dummy_attr = {et: torch.randn(2, 768).to(self.device) for et in edge_types}
        with torch.no_grad():
            try:
                _ = self.encoder(dummy_x, dummy_edge, edge_attr_dict=dummy_attr)
            except Exception:
                pass

    def get_node_token_alignment(self, code, node_positions):
        """
        优化后的对齐逻辑：采用交集判定，防止因边界微差导致的漏匹配
        """
        if not node_positions or len(node_positions) == 0:
            return []

        inputs = self.tokenizer(code, return_offsets_mapping=True, padding='max_length', truncation=True,
                                max_length=512)
        offsets = inputs['offset_mapping']
        alignment = []

        for n_idx, pos in enumerate(node_positions):
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                n_start, n_end = pos
                for t_idx, (t_start, t_end) in enumerate(offsets):
                    if t_start == t_end: continue  # 跳过特殊 token

                    # 交集判定：Token 与节点范围有重叠
                    if t_start < n_end and t_end > n_start:
                        alignment.append([n_idx, t_idx])
                        break
        return alignment

    def process(self, graph_dataset_path, raw_code_df, output_path, chunk_size=200, max_nodes=800):
        print(f"Loading graph dataset...")
        graph_data_obj = torch.load(graph_dataset_path, map_location='cpu', weights_only=False)
        graph_data = graph_data_obj['graphs'] if isinstance(graph_data_obj, dict) else graph_data_obj

        output_dir = os.path.dirname(output_path)
        base_filename = os.path.basename(output_path).replace('.pt', '')
        os.makedirs(output_dir, exist_ok=True)

        results = []
        shard_count = 0
        loader = DataLoader(graph_data, batch_size=1, shuffle=False)

        for i, data in enumerate(tqdm(loader, desc="Stage 2 Processing")):
            # --- 1. 数据合法性检查 (关键修复) ---
            # 检查是否包含任何边，如果没有边，edge_index_dict 会报错
            has_edges = False
            for store in data.edge_stores:
                if 'edge_index' in store and store.edge_index.numel() > 0:
                    has_edges = True
                    break

            if not has_edges:
                continue  # 跳过没有边的孤立图

            total_nodes = sum([data[nt].num_nodes for nt in data.node_types])
            if total_nodes > max_nodes or total_nodes == 0:
                continue

            # --- 2. 准备数据 ---
            data = data.to(self.device)
            actual_idx = data.idx.item() if torch.is_tensor(data.idx) else data.idx

            if actual_idx >= len(raw_code_df):
                continue

                # 使用 actual_idx 而不是 i
            code = raw_code_df.iloc[actual_idx]['cleaned_code']

            # --- 3. 推理与特征提取 ---
            with torch.no_grad():
                try:
                    # --- 修改后 (在后面多加一个逗号和下划线) ---
                    node_emb_dict, graph_emb, _ = self.encoder(
                        data.x_dict,
                        data.edge_index_dict,
                        edge_attr_dict=getattr(data, 'edge_attr_dict', None),
                        batch_dict={nt: data[nt].batch for nt in data.node_types}
                    )
                except Exception as e:
                    print(f"\nError processing sample {i}: {e}")
                    continue

                ast_list, flow_list, sem_list = [], [], []
                for nt in self.encoder.metadata[0]:
                    if nt in node_emb_dict:
                        emb = node_emb_dict[nt].cpu()

                        # 路由逻辑：根据节点类型分配到不同的特征池
                        if nt in ['structure', 'statement']:
                            ast_list.append(emb)
                        elif nt in ['expression']:
                            flow_list.append(emb)
                        elif nt in ['variable']:
                            sem_list.append(emb)
                        else:
                            # 其他节点（other_node）可以根据需要归类
                            ast_list.append(emb)

                    # 拼接每一类特征
                    # 注意：如果某一类为空，需要补一个占位符 Tensor 防止后续报错
                ast_embs = torch.cat(ast_list, dim=0) if ast_list else torch.zeros((1, 256))
                flow_embs = torch.cat(flow_list, dim=0) if flow_list else torch.zeros((1, 256))
                sem_embs = torch.cat(sem_list, dim=0) if sem_list else torch.zeros((1, 256))

            # --- 4. 结果整理 (关键修复部分) ---
            motif_indices = {cat: [] for cat in self.TYPE_MAPPING}
            current_idx = 0
            ordered_embs = []
            all_node_positions = []  # 新增：用于收集所有类型的节点位置

            # 严格按照 metadata 的顺序循环，确保与 Embedding 拼接一致
            for nt in self.encoder.metadata[0]:
                if nt in node_emb_dict:
                    emb = node_emb_dict[nt].cpu()
                    ordered_embs.append(emb)

                    # 修复位置信息提取：从子对象 [nt] 中获取 node_positions
                    if hasattr(data[nt], 'node_positions'):
                        # data[nt].node_positions 是 [num_nodes, 2] 的 tensor
                        all_node_positions.extend(data[nt].node_positions.tolist())
                    else:
                        # 如果该类型没有位置信息，用空占位，保持索引对齐
                        all_node_positions.extend([[0, 0]] * emb.size(0))

                    # 计算所属分类 (这部分逻辑保持不变)
                    target_cat = 'semantic'
                    for cat, types in self.TYPE_MAPPING.items():
                        if nt in types:
                            target_cat = cat
                            break
                    motif_indices[target_cat].extend(range(current_idx, current_idx + emb.size(0)))
                    current_idx += emb.size(0)

            sample = {
                'idx': actual_idx,  # 【关键修改】使用 actual_idx 替代 i，保证与路径数据集对齐
                'node_embs': torch.cat(ordered_embs, dim=0).tolist(),
                'code': code,
                'ast_embs': ast_embs.tolist(),  # 对应 m_struct
                'flow_embs': flow_embs.tolist(),  # 对应 m_logic
                'sem_embs': sem_embs.tolist(),  # 对应 m_semantic
                'meta': {
                    'motif_indices': motif_indices,
                    'node_token_pairs': self.get_node_token_alignment(code, all_node_positions)
                }
            }
            results.append(sample)

            # --- 5. 清理显存与分片保存 ---
            del node_emb_dict, ordered_embs, data
            if i % 10 == 0:
                torch.cuda.empty_cache()

            if len(results) >= chunk_size:
                shard_count += 1
                save_path = os.path.join(output_dir, f"{base_filename}_shard_{shard_count}.pt")
                torch.save(results, save_path)
                results = []
                gc.collect()

        if results:
            shard_count += 1
            torch.save(results, os.path.join(output_dir, f"{base_filename}_shard_{shard_count}.pt"))

        print(f"\nProcessing Complete. Saved {shard_count} shards to {output_dir}")

if __name__ == "__main__":
    config = {
        'stage1_checkpoint': '/root/autodl-tmp/first_paper/output/python/ablation_stage1_model/best_hetero_model.pt',
        'model_path': '/root/autodl-tmp/unixcoder-base',
        'hidden_channels': 512,
        'out_channels': 256,
        'num_layers': 4,
        'num_heads': 8
    }

    df = pd.read_pickle('/root/autodl-tmp/first_paper/code_dataset/python/python_data.pkl')
    generator = Stage2DataGenerator(config)
    generator.process(
        graph_dataset_path='/root/autodl-tmp/first_paper/graph_dataset/python/processed_dataset.pt',
        raw_code_df=df,
        output_path='/root/autodl-tmp/first_paper/graph_dataset/ablation_dataset_stage1/stage2_dataset/processed_dataset_stage2.pt',
        chunk_size=1000,
        max_nodes=800
    )