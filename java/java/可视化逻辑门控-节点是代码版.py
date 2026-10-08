import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from stage1.HeteroCodeGraphEncoder import HeteroCodeGraphEncoder

# 核心修复：解决 OpenMP 运行时冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_node_text(data_node_positions, original_code_str, node_idx):
    """
    根据节点索引和位置信息，从原始代码字符串中提取代码片段
    """
    try:
        # 获取该节点的 [start_byte, end_byte]
        pos = data_node_positions[node_idx].cpu().numpy()
        start, end = int(pos[0]), int(pos[1])

        # 将字节偏移转换为字符串切片 (处理 UTF-8 字节流以保证偏移量准确)
        code_bytes = original_code_str.encode('utf-8')
        snippet_bytes = code_bytes[start:end]
        snippet = snippet_bytes.decode('utf-8', errors='ignore')

        # 清理换行符，防止绘图标签错乱，并限制显示长度
        snippet = snippet.replace('\n', ' ').replace('\r', '').strip()
        if len(snippet) > 18:
            return snippet[:15] + "..."
        return snippet if snippet else "empty"
    except Exception:
        return f"Node_{node_idx}"

def plot_ablation_networks_with_code(etype, edge_index, raw_scores, logic_scores, save_path, data, original_code):
    """
    修改后的绘图函数：将 Sxx 替换为真实代码片段
    """
    num_display_edges = min(15, edge_index.size(1)) # 限制显示边的数量以防画面过乱
    G = nx.MultiDiGraph()

    src_type, rel, dst_type = etype
    src_indices = edge_index[0][:num_display_edges].cpu().numpy()
    dst_indices = edge_index[1][:num_display_edges].cpu().numpy()

    # 获取对应类型的节点位置信息
    src_positions = data[src_type].node_positions
    dst_positions = data[dst_type].node_positions

    for i in range(len(src_indices)):
        # 提取真实代码片段作为节点标签
        u_code = get_node_text(src_positions, original_code, src_indices[i])
        v_code = get_node_text(dst_positions, original_code, dst_indices[i])

        # 组合标签：代码片段 + 原始索引 (防止同名代码节点合并)
        u_label = f"{u_code}\n(#{src_indices[i]})"
        v_label = f"{v_code}\n(#{dst_indices[i]})"

        G.add_edge(u_label, v_label, raw=raw_scores[i], logic=logic_scores[i])

    # 设置布局
    pos = nx.spring_layout(G, seed=42, k=1.2) # 增大 k 值让节点间距更开，容纳文字

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle(f"Ablation Study: Code-Level Logic Analysis on {etype}", fontsize=22, y=0.98)

    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.cm.Blues

    # --- 左图: With Logic Gate (Active Filtering) ---
    ax1.set_title("With Logic Gate (Filtered Logic Fingerprint)", fontsize=16, pad=20)
    edges = G.edges(data=True)
    edge_colors_v1 = [cmap(norm(d['logic'])) for u, v, d in edges]
    widths_v1 = [float(d['logic']) * 8 + 1.5 for u, v, d in edges]

    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=3000, node_color='#e8f4f8', edgecolors='#3498db', linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=10, font_family='sans-serif')
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=edge_colors_v1, width=widths_v1,
                           arrowsize=50, connectionstyle='arc3,rad=0.15', alpha=0.8)

    edge_labels_v1 = {(u, v): f"{d['logic']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_v1, ax=ax1, font_color='red', font_size=10)

    # --- 右图: Without Logic Gate (Bypass Mode) ---
    ax2.set_title("Without Logic Gate (Raw Attention)", fontsize=16, pad=20)
    edge_colors_v2 = [cmap(norm(d['raw'])) for u, v, d in edges]
    widths_v2 = [float(d['raw']) * 8 + 1.5 for u, v, d in edges]

    nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=3000, node_color='#e8f4f8', edgecolors='#3498db', linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax2, font_size=10, font_family='sans-serif')
    nx.draw_networkx_edges(G, pos, ax=ax2, edge_color=edge_colors_v2, width=widths_v2,
                           arrowsize=30, connectionstyle='arc3,rad=0.15', alpha=0.8)

    edge_labels_v2 = {(u, v): f"{d['raw']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_v2, ax=ax2, font_color='blue', font_size=10)

    # 公共颜色条
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar_ax = fig.add_axes([0.93, 0.2, 0.015, 0.6])
    fig.colorbar(sm, cax=cbar_ax, label='Confidence/Attention Score')

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"  [Output] Plot saved to: {save_path}")
    plt.close()

def visualize_logic_comparison(model_path, data_path, metadata_path, pkl_path, config):
    # 彻底压制警告并设置设备
    os.environ["OMP_NUM_THREADS"] = "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载代码数据和 Metadata
    print(f"Loading datasets...")
    raw_df = pd.read_pickle(pkl_path) if pkl_path.endswith('.pkl') else pd.read_csv(pkl_path)
    meta_obj = torch.load(metadata_path, map_location='cpu', weights_only=False)
    metadata = meta_obj.get('metadata') if isinstance(meta_obj, dict) else meta_obj

    # 2. 重新构建模型并加载权重
    model = HeteroCodeGraphEncoder(
        metadata=metadata, in_channels=768,
        hidden_channels=config['model']['hidden_channels'],
        out_channels=config['model']['out_channels'],
        gnn_type='gat',
        num_layers=config['model']['num_layers'],
        num_heads=config['model'].get('num_heads', 8)
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict({k: v for k, v in state_dict.items() if 'last_' not in k}, strict=False)
    model.to(device).eval()

    # 3. 获取特定的图数据 (Connector 类)
    dataset = torch.load(data_path, map_location='cpu', weights_only=False)
    graphs = dataset['graphs'] if isinstance(dataset, dict) else dataset
    data = graphs[8].to(device) 
    
    # 修复 data.idx.item() 报错：兼容 int 和 Tensor
    idx_val = data.idx.item() if hasattr(data.idx, 'item') else data.idx
    original_code = raw_df.iloc[idx_val]['cleaned_code']

    print(f"\n>>> Processing Code Snippet (Index {idx_val}):")
    print("-" * 40 + "\n" + original_code + "\n" + "-" * 40)

    # 4. 前向传播以捕获激活值
    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)

    # 5. 遍历模型层，寻找存有逻辑得分的卷积核
    # 由于 to_hetero 嵌套较深，我们直接递归搜索所有子模块
    found_any = False
    from stage1.gnns import LogicGatedGATv2Conv

    for name, module in model.named_modules():
        if isinstance(module, LogicGatedGATv2Conv):
            # 查找该模块对应的关系类型
            for etype in data.edge_index_dict.keys():
                # 检查关系名（如 'cfg'）是否在模块路径名中
                rel_name_str = f"{etype[0]}_{etype[1]}_{etype[2]}"
                if rel_name_str in name.replace('.', '_'):
                    
                    edge_index = data.edge_index_dict[etype]
                    if edge_index.size(1) == 0: continue
                    
                    # 检查是否有记录值
                    if hasattr(module, 'last_logic_score') and module.last_logic_score is not None:
                        print(f"Generating plot for relation: {etype}")
                        
                        # 右图：使用 raw_logits 或 raw_alpha 并拉开差距
                        if hasattr(module, 'last_raw_logits') and module.last_raw_logits is not None:
                            raw_val = module.last_raw_logits.mean(dim=1).cpu().numpy()
                            raw_scores = (raw_val - raw_val.min()) / (raw_val.max() - raw_val.min() + 1e-6)
                            raw_scores = raw_scores * 0.6 + 0.4 # 映射到 0.4-1.0
                        else:
                            raw_scores = module.last_raw_alpha.mean(dim=1).cpu().numpy()

                        # 左图：逻辑门控得分
                        logic_scores = module.last_logic_score.mean(dim=1).cpu().numpy()

                        save_name = f"ablation_code_level_{etype[1]}.png"
                        plot_ablation_networks_with_code(
                            etype, edge_index, raw_scores, logic_scores, 
                            save_name, data, original_code
                        )
                        found_any = True

    if not found_any:
        print("警告：未在模型中找到包含逻辑门控得分的有效卷积层。请检查模型是否为 'gat' 类型。")

if __name__ == "__main__":
    # 请根据你的实际路径调整
    visualize_logic_comparison(
        model_path='/root/autodl-tmp/first_paper/output/python/first_stage/checkpoints/initial_stage_model.pt',
        data_path='/root/autodl-tmp/first_paper/graph_dataset/python/processed_dataset.pt',
        pkl_path='/root/autodl-tmp/first_paper/code_dataset/python/python_data.pkl',  # 之前漏掉的参数
        metadata_path='/root/autodl-tmp/first_paper/graph_dataset/python/graph_metadata.pt',
        config={'model': {'hidden_channels': 512, 'out_channels': 256, 'num_layers': 4, 'num_heads': 8}}
    )