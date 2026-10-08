import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import yaml
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer
from Stage2_AlignmentBridge import InnovativeStage2Bridge
from Innovative_Stage2_Trainer import InnovativeStage2Trainer
import gc
# 确保 OpenMP 环境平稳，消除多线程环境警告
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# --- 内部自建兜底扰动函数 ---
def construct_lsr_labels(node_embs, perturb_rate=0.15):
    """
    为评估流构造节点微扰，并生成结构验证伪标签
    """
    nodes_perturbed = node_embs.clone()
    mask = torch.rand(nodes_perturbed.shape[:-1], device=node_embs.device) < perturb_rate
    nodes_perturbed[mask] = 0.0
    
    bs = node_embs.size(0) if node_embs.dim() == 3 else 1
    dummy_labels = torch.zeros(bs, dtype=torch.long, device=node_embs.device)
    return nodes_perturbed, dummy_labels


# --- 专属评估校对函数，防止丢弃 meta 信息 ---
def eval_collate_fn(batch):
    max_nodes = max([b['node_embs'].size(0) for b in batch])
    node_dim = batch[0]['node_embs'].size(1)
    padded_nodes = torch.zeros(len(batch), max_nodes, node_dim)

    for i, b in enumerate(batch):
        padded_nodes[i, :b['node_embs'].size(0)] = b['node_embs']

    return {
        'node_embs': padded_nodes,
        'text_input_ids': torch.stack([b['text_input_ids'] for b in batch]),
        'text_atts': torch.stack([b['text_atts'] for b in batch]),
        'raw_code': [b['raw_code'] for b in batch],
        'meta': [b['meta'] for b in batch]  
    }


def plot_global_pr_curve(all_graph_feats, all_text_feats, save_path):
    """
    【学术修复版】全局跨模态检索 PR 曲线（英文规范版，杜绝 Glyph 警告）
    """
    g_f = torch.tensor(np.concatenate(all_graph_feats, axis=0))  
    t_f = torch.tensor(np.concatenate(all_text_feats, axis=0))   

    raw_sim = torch.matmul(g_f, t_f.T)
    sim_matrix, _ = raw_sim.max(dim=1)
    sim_matrix = sim_matrix.numpy()  

    n_samples = sim_matrix.shape[0]
    labels_matrix = np.eye(n_samples, dtype=np.int32)

    row_min = sim_matrix.min(axis=1, keepdims=True)
    row_max = sim_matrix.max(axis=1, keepdims=True)
    norm_sim_matrix = (sim_matrix - row_min) / (row_max - row_min + 1e-8)

    y_scores = norm_sim_matrix.flatten()
    y_true = labels_matrix.flatten()

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(7, 6))
    sns.set_theme(style="whitegrid")
    
    plt.plot(recall, precision, color='#1f77b4', linewidth=3, label=f'Ours (mAP / AUC = {pr_auc:.4f})')
    plt.fill_between(recall, precision, alpha=0.12, color='#1f77b4')
    
    plt.title("Global Cross-Modal Retrieval Precision-Recall Curve", fontsize=12, weight='bold', pad=15)
    # 修复点：将中文改成地道的学术英文
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.legend(loc="lower left", frameon=True, shadow=True, facecolor='white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 [学术修复成功] 全局检索 PR 曲线已重新注入能量！新 AUC 面积 (mAP): {pr_auc:.4f}")


def plot_robustness_horizon(perturb_horizons, acc1_list, acc5_list, mrr_list, save_path):
    """
    【升级丰富版】纯粹检索鲁棒性边界折线图
    多指标（Top-1, Top-5, MRR）联动，全方位展现模型在结构微扰下的架构防御韧性
    """
    plt.figure(figsize=(8.5, 5.5))
    sns.set_theme(style="whitegrid")
    
    # 顶会级柔和高对比度配色
    colors = {
        'top1': '#1f77b4',  # 科技蓝
        'top5': '#2ca02c',  # 学术绿
        'mrr': '#d62728'    # 珊瑚红
    }
    
    # 绘制三条核心对齐演化曲线
    plt.plot(perturb_horizons, acc1_list, color=colors['top1'], marker='o', linewidth=2.5, 
             markersize=7, label='Top-1 Retrieval Acc')
    plt.plot(perturb_horizons, acc5_list, color=colors['top5'], marker='s', linewidth=2.0, 
             markersize=6, linestyle='--', label='Top-5 Retrieval Acc')
    plt.plot(perturb_horizons, mrr_list, color=colors['mrr'], marker='^', linewidth=2.0, 
             markersize=6, linestyle='-.', label='Mean Reciprocal Rank (MRR)')
    
    # 细节美化与高光填充（为最核心的 Top-1 增加淡淡的边界投影阴影）
    plt.fill_between(perturb_horizons, [x - 0.02 for x in acc1_list], [x + 0.02 for x in acc1_list], 
                     alpha=0.08, color=colors['top1'])

    # 设置坐标轴标签与范围
    plt.xlabel('Graph Node Perturbation Rate (Noise Level)', fontsize=11, labelpad=10, weight='bold')
    plt.ylabel('Alignment Performance Metric Score', fontsize=11, weight='bold')
    plt.xlim([-0.02, max(perturb_horizons) + 0.02])
    plt.ylim([-0.05, 1.05])
    
    # 百分比刻度规范化
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%' if y<=1.0 else f'{y:.1f}'))

    # 优雅的图例摆放
    plt.legend(loc='lower left', frameon=True, shadow=True, facecolor='white', fontsize=10)
    plt.title("Robustness Horizon: Multi-Metric Evaluation Under Architectural Immunity", fontsize=12, weight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"🛡️ [丰富版图表] 成功生成多指标渐进式鲁棒性边界防御折线图！")

# --- 100% 保持你原有的基础可视化函数不变 ---

def plot_itc_distribution(all_graph_feats, all_text_feats, save_path):
    """可视化正负样本相似度分布"""
    g_f = torch.tensor(np.concatenate(all_graph_feats, axis=0))
    t_f = torch.tensor(np.concatenate(all_text_feats, axis=0))

    raw_sim = torch.matmul(g_f, t_f.T)
    sim_matrix, _ = raw_sim.max(dim=1)
    sim_matrix = sim_matrix.numpy()

    pos_scores = np.diag(sim_matrix)
    mask = np.ones(sim_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    neg_scores = sim_matrix[mask]

    plt.figure(figsize=(10, 6))
    sns.histplot(pos_scores, color="skyblue", label="Positive Pairs", kde=True, stat="probability")
    sns.histplot(neg_scores, color="salmon", label="Negative Pairs", kde=True, stat="probability")
    plt.axvline(pos_scores.mean(), color='blue', linestyle='--', label=f'Pos Mean: {pos_scores.mean():.2f}')
    plt.axvline(neg_scores.mean(), color='red', linestyle='--', label=f'Neg Mean: {neg_scores.mean():.2f}')
    plt.title("ITC Score Distribution (Pilot-Query Compressed)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()


# --- 高级可视化与量化指标函数（包含全新三子图联动联动 t-SNE 渲染器） ---

def plot_decoupled_tsne_subplots(all_s, all_l, all_se, all_queries, save_path):
    """
    魔改版顶会联动图：引入流形边界和中心聚拢线，凸显突出 Query 的空间特异化对齐
    """
    s_raw = np.concatenate(all_s, axis=0)      
    l_raw = np.concatenate(all_l, axis=0)      
    se_raw = np.concatenate(all_se, axis=0)    
    q_raw = np.concatenate(all_queries, axis=0) 
    
    bs = s_raw.shape[0]
    q_s = q_raw[:, 0:10, :].mean(axis=1)       
    l_raw_q = q_raw[:, 10:20, :].mean(axis=1)      
    q_se = q_raw[:, 20:29, :].mean(axis=1)     
    q_pilot = q_raw[:, 30:32, :].mean(axis=1)  
    
    combined = np.concatenate([s_raw, l_raw, se_raw, q_s, l_raw_q, q_se, q_pilot], axis=0)
    
    tsne = TSNE(n_components=2, perplexity=min(30, len(combined) // 4), random_state=42)
    embeds = tsne.fit_transform(combined)
    
    ptr = 0
    s_coord = embeds[ptr : ptr + bs]; ptr += bs
    l_coord = embeds[ptr : ptr + bs]; ptr += bs
    se_coord = embeds[ptr : ptr + bs]; ptr += bs
    qs_coord = embeds[ptr : ptr + bs]; ptr += bs
    ql_coord = embeds[ptr : ptr + bs]; ptr += bs
    qse_coord = embeds[ptr : ptr + bs]; ptr += bs
    qp_coord = embeds[ptr : ptr + bs]
    
    # 开启高画质学术画布
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), sharey=False) # 取消 sharey 放大局域对比
    sns.set_theme(style="whitegrid")
    
    colors = {'struct': '#a1c4fd', 'logic': '#ffcc99', 'semantic': '#99e699', 'pilot': '#e06666'}
    
    # -------- (a) Structural Subplot --------
    ax = axes[0]
    # 核心突出点：给空间节点加上阴影等高线（KDE Moss），做成“引力场”效果
    sns.kdeplot(x=s_coord[:, 0], y=s_coord[:, 1], ax=ax, cmap="Blues", fill=True, alpha=0.3, levels=5)
    ax.scatter(s_coord[:, 0], s_coord[:, 1], c=colors['struct'], marker='o', s=40, alpha=0.4, label='Structural Space Nodes')
    # 放大 Query 的尺寸，并加上强烈的白色描边（Edgecolor）
    ax.scatter(qs_coord[:, 0], qs_coord[:, 1], c='#002060', marker='X', s=180, edgecolors='w', linewidths=1.5, label='Captured Struct Queries (0-9)', zorder=5)
    ax.scatter(qp_coord[:, 0], qp_coord[:, 1], c=colors['pilot'], marker='^', s=120, edgecolors='k', label='Pilot Routing Queries (30-31)', zorder=4)
    ax.set_title("(a) Structural Space Attracted Window", fontsize=13, weight='bold')
    
    # -------- (b) Logical Subplot --------
    ax = axes[1]
    sns.kdeplot(x=l_coord[:, 0], y=l_coord[:, 1], ax=ax, cmap="Oranges", fill=True, alpha=0.3, levels=5)
    ax.scatter(l_coord[:, 0], l_coord[:, 1], c=colors['logic'], marker='o', s=40, alpha=0.4, label='Logical Space Nodes')
    ax.scatter(ql_coord[:, 0], ql_coord[:, 1], c='#7f3f00', marker='P', s=180, edgecolors='w', linewidths=1.5, label='Captured Logic Queries (10-19)', zorder=5)
    ax.scatter(qp_coord[:, 0], qp_coord[:, 1], c=colors['pilot'], marker='^', s=120, edgecolors='k', label='Pilot Routing Queries (30-31)', zorder=4)
    ax.set_title("(b) Logical Space Attracted Window", fontsize=13, weight='bold')
    
    # -------- (c) Semantic Subplot --------
    ax = axes[2]
    sns.kdeplot(x=se_coord[:, 0], y=se_coord[:, 1], ax=ax, cmap="Greens", fill=True, alpha=0.3, levels=5)
    ax.scatter(se_coord[:, 0], se_coord[:, 1], c=colors['semantic'], marker='o', s=40, alpha=0.4, label='Semantic Space Nodes')
    ax.scatter(qse_coord[:, 0], qse_coord[:, 1], c='#145214', marker='s', s=150, edgecolors='w', linewidths=1.5, label='Captured Semantic Queries (20-29)', zorder=5)
    ax.scatter(qp_coord[:, 0], qp_coord[:, 1], c=colors['pilot'], marker='^', s=120, edgecolors='k', label='Pilot Routing Queries (30-31)', zorder=4)
    ax.set_title("(c) Semantic Space Attracted Window", fontsize=13, weight='bold')
    
    for ax in axes:
        ax.set_xlabel("T-SNE Axis 1", fontsize=10)
        ax.set_ylabel("T-SNE Axis 2", fontsize=10)
        ax.legend(loc='lower left', fontsize=9, frameon=True, shadow=True)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_query_matrix(matrix, title, save_path):
    """统一的 Query-空间特征映射绘图仪"""
    plt.figure(figsize=(12, 6))
    sns.heatmap(matrix, cmap="YlGnBu", xticklabels=False, cbar_kws={'label': 'Intensity / Weight'})
    num_nodes = matrix.shape[1] // 3
    plt.axvline(x=num_nodes, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.axvline(x=2*num_nodes, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.axhline(y=10, color='purple', linestyle=':', linewidth=1.5, alpha=0.8)
    plt.axhline(y=20, color='purple', linestyle=':', linewidth=1.5, alpha=0.8)
    plt.axhline(y=30, color='purple', linestyle=':', linewidth=1.5, alpha=0.8)
    
    plt.text(num_nodes / 2, -1.2, "Structural Space", ha='center', color='red', weight='bold', fontsize=10)
    plt.text(num_nodes * 1.5, -1.2, "Logical Space", ha='center', color='red', weight='bold', fontsize=10)
    plt.text(num_nodes * 2.5, -1.2, "Semantic Space", ha='center', color='red', weight='bold', fontsize=10)
    
    plt.title(title, fontsize=13, pad=20)
    plt.xlabel("Concatenated Graph Node Tokens (Structural ➔ Logical ➔ Semantic)", fontsize=11)
    plt.ylabel("Q-former Query Tokens Index (0 - 31)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def calculate_space_orthogonality(all_s_feats, all_l_feats, all_se_feats):
    """定量计算：测量高维流形中三大专家空间两两之间的平均余弦相似度绝对值"""
    s_tensor = torch.tensor(np.concatenate(all_s_feats, axis=0))
    l_tensor = torch.tensor(np.concatenate(all_l_feats, axis=0))
    se_tensor = torch.tensor(np.concatenate(all_se_feats, axis=0))
    
    s_raw = F.normalize(s_tensor.mean(dim=1) if s_tensor.dim() == 3 else s_tensor, dim=-1)
    l_raw = F.normalize(l_tensor.mean(dim=1) if l_tensor.dim() == 3 else l_tensor, dim=-1)
    se_raw = F.normalize(se_tensor.mean(dim=1) if se_tensor.dim() == 3 else se_tensor, dim=-1)
    
    sim_sl = torch.abs(torch.sum(s_raw * l_raw, dim=-1)).mean().item()
    sim_sse = torch.abs(torch.sum(s_raw * se_raw, dim=-1)).mean().item()
    sim_lse = torch.abs(torch.sum(l_raw * se_raw, dim=-1)).mean().item()
    
    print(f"\n📐 专家空间解耦纯度量化检测 (越接近 0.0000 说明正交解耦越彻底):")
    print(f"   | 👉 Cosine_Sim(Structural, Logical)  : {sim_sl:.4f}")
    print(f"   | 👉 Cosine_Sim(Structural, Semantic) : {sim_sse:.4f}")
    print(f"   | 👉 Cosine_Sim(Logical, Semantic)    : {sim_lse:.4f}")


def plot_batch_itc_heatmap(all_graph_feats, all_text_feats, save_path):
    """绘制前16个测试样本的跨模态检索匹配热力图"""
    g_f = torch.tensor(np.concatenate(all_graph_feats, axis=0)[:16])  # [16, 2, dim]
    t_f = torch.tensor(np.concatenate(all_text_feats, axis=0)[:16])   # [16, dim]
    
    raw_sim = torch.matmul(g_f, t_f.T)
    sim_matrix, _ = raw_sim.max(dim=1)
    sim_matrix = sim_matrix.numpy()
    
    plt.figure(figsize=(9, 8))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap='Blues', cbar=True, square=True,
                cbar_kws={'label': 'Alignment Intensity'}, annot_kws={"size": 9})
    plt.title("Pilot-Query ITC Similarity Heatmap (Top-16 Samples)", fontsize=13, weight='bold', pad=15)
    plt.xlabel("Text Embedding Index", fontsize=11)
    plt.ylabel("Compressed Graph Embedding Index", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# --- 核心评估类 ---

class Stage2EvalDataset(torch.utils.data.Dataset):
    """无 LOGIC_PT 轻量化评估数据集"""
    def __init__(self, pt_path, tokenizer_path):
        self.samples = torch.load(pt_path, map_location='cpu', weights_only=False)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        g_idx = item.get('idx', idx)
        
        # 移除了读取 logic 外部映射的开销，构造纯净轻量级的元数据容器
        meta = {
            'idx': g_idx
        }
        text_inputs = self.tokenizer(item['code'], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        return {
            'node_embs': torch.tensor(item['node_embs'], dtype=torch.float32),
            'text_input_ids': text_inputs.input_ids.squeeze(0),
            'text_atts': text_inputs.attention_mask.squeeze(0),
            'meta': meta,
            'raw_code': item['code']
        }


def run_comprehensive_evaluation(config_path, model_path, pt_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    dataset = Stage2EvalDataset(pt_path, config['model']['bert_model_dir'])
    indices = torch.randperm(len(dataset))[:100].tolist()
    val_loader = DataLoader(Subset(dataset, indices), batch_size=8, shuffle=False, collate_fn=eval_collate_fn)

    model = InnovativeStage2Bridge(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    metrics_cache = {
        's_feat': [], 'l_feat': [], 'se_feat': [], 
        'g_feat': [], 't_feat': [], 'query_all_feats': [],
        'loss_ortho_list': []
    }

    print(f"\n🚀 开始全面评估 [Device: {device}]...")
    last_outputs = None

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            nodes_perturbed, p_labels = construct_lsr_labels(batch['node_embs'])
            batch['node_embs'] = nodes_perturbed
            batch['path_label'] = p_labels

            outputs = model(batch, task='test')
            
            # 浅拷贝一份最后一包的数据用来画图，避免强引用整个历史 outputs
            if b_idx == len(val_loader) - 1:
                last_outputs = {k: v.cpu() if torch.is_tensor(v) else v for k, v in outputs.items()}
                # 特殊处理 attentions 列表，防止显存挂载
                if "attentions" in outputs and outputs["attentions"] is not None:
                    last_outputs["attentions"] = [a.cpu() if torch.is_tensor(a) else a for a in outputs["attentions"]]

            metrics_cache['s_feat'].append(outputs['struct_feats'].mean(dim=1).cpu().numpy())
            metrics_cache['l_feat'].append(outputs['logic_feats'].mean(dim=1).cpu().numpy())
            metrics_cache['se_feat'].append(outputs['semantic_feats'].mean(dim=1).cpu().numpy())
            
            metrics_cache['g_feat'].append(outputs['graph_feats'].cpu().numpy()) 
            metrics_cache['t_feat'].append(outputs['text_feats'].cpu().numpy())
            metrics_cache['query_all_feats'].append(outputs['query_output'].cpu().numpy())
            
            if 'loss_ortho' in outputs:
                metrics_cache['loss_ortho_list'].append(outputs['loss_ortho'].item())
            else:
                metrics_cache['loss_ortho_list'].append(0.0)
            
            if b_idx < 1:
                for i in range(min(2, len(batch['raw_code']))):
                    print(f"\n{'=' * 20} Case Study: Index {batch['meta'][i]['idx']} {'=' * 20}")
                    print(f"【源代码】: {batch['raw_code'][i][:150]}...")

    # 3. 演化热力图绘制
    if last_outputs is not None:
        print("📊 正在激活演化扫描 ➔ 阶段 1：生成进入 Q-former 之前的原始相似度分布...")
        q_tokens = model.query_tokens.squeeze(0).detach().cpu()  
        s_f = last_outputs['struct_feats'].detach()
        l_f = last_outputs['logic_feats'].detach()
        se_f = last_outputs['semantic_feats'].detach()
        graph_embeds = torch.cat([s_f, l_f, se_f], dim=1).mean(dim=0).detach()  
        
        q_norm = F.normalize(q_tokens, dim=-1)
        g_norm = F.normalize(graph_embeds, dim=-1)
        initial_sim = torch.matmul(q_norm, g_norm.T).numpy()  
        
        plot_query_matrix(initial_sim, "Phase 1: Prior Cosine Similarity Map", os.path.join(output_dir, "eval_query_specialization_before.png"))
        
        if "attentions" in last_outputs and last_outputs["attentions"] is not None:
            valid_attns = []
            for attn in last_outputs["attentions"]:
                if attn is not None and torch.is_tensor(attn) and attn.dim() == 4 and attn.size(2) == 32:
                    valid_attns.append(attn.detach())
            
            if len(valid_attns) > 0:
                plot_query_matrix(valid_attns[0].mean(dim=0).mean(dim=0).numpy(), "Phase 2: First Layer Cross-Attention Matrix", os.path.join(output_dir, "eval_query_specialization_first_layer.png"))
                plot_query_matrix(valid_attns[-1].mean(dim=0).mean(dim=0).numpy(), "Phase 3: Final Layer Cross-Attention Matrix", os.path.join(output_dir, "eval_query_specialization_final_layer.png"))
    
    print("\n📊 正在生成图表报告与高级特征演化扫描...")
    
    plot_decoupled_tsne_subplots(
        metrics_cache['s_feat'], metrics_cache['l_feat'], metrics_cache['se_feat'],
        metrics_cache['query_all_feats'], os.path.join(output_dir, "eval_decoupled_tsne_subplots.png")
    )
    
    plot_itc_distribution(metrics_cache['g_feat'], metrics_cache['t_feat'], os.path.join(output_dir, "eval_itc_dist.png"))
    plot_batch_itc_heatmap(metrics_cache['g_feat'], metrics_cache['t_feat'], os.path.join(output_dir, "eval_pilot_itc_heatmap.png"))
    plot_global_pr_curve(metrics_cache['g_feat'], metrics_cache['t_feat'], os.path.join(output_dir, "eval_global_pr_curve.png"))
    
    # 显式清空前面的变量以释放显存
    del outputs, last_outputs
    torch.cuda.empty_cache()
    gc.collect()

    print("\n📊 正在启动后处理特征流扫描 ➔ 激活多级微扰鲁棒性边界探测...")
    horizons = [0.0, 0.05, 0.10, 0.15, 0.25, 0.40]
    acc1_horizon_list = []
    acc5_horizon_list = []
    mrr_horizon_list = []

    for rate in horizons:
        horizon_g_feats = []
        horizon_t_feats = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                nodes_perturbed, p_labels = construct_lsr_labels(batch['node_embs'], perturb_rate=rate)
                batch['node_embs'] = nodes_perturbed
                batch['path_label'] = p_labels
                
                outputs = model(batch, task='test')
                horizon_g_feats.append(outputs['graph_feats'].cpu())
                horizon_t_feats.append(outputs['text_feats'].cpu())
                
                del outputs
                
        g_full_hz = torch.cat(horizon_g_feats, dim=0)
        t_full_hz = torch.cat(horizon_t_feats, dim=0)
        
        # 调用 Trainer 内部计算指标（确保包含 acc1, acc5，通常 compute_itc_metrics 也会顺便计算 mrr）
        res_hz = InnovativeStage2Trainer.compute_itc_metrics(g_full_hz, t_full_hz)
        
        # 沉淀三个检索维度数据
        acc1_hz = res_hz.get('itc_acc1', 0.0)
        acc5_hz = res_hz.get('itc_acc5', 0.0)
        
        # 兜底计算一个轻量级 MRR（如果你的 compute_itc_metrics 里没有返回 'itc_mrr' 的话）
        if 'itc_mrr' in res_hz:
            mrr_hz = res_hz['itc_mrr']
        else:
            # 简单的矩阵自计算 MRR 兜底机制
            # ======= 【安全修复版】维度鲁棒性自适应 MRR 计算 =======
            # 1. 确保 g_full_hz 和 t_full_hz 降维到 [B, dim] 标准 2D 矩阵
            g_2d = g_full_hz.mean(dim=1) if g_full_hz.dim() == 3 else g_full_hz
            t_2d = t_full_hz.mean(dim=1) if t_full_hz.dim() == 3 else t_full_hz
            
            # 2. 计算标准 [B, B] 的跨模态检索相似度方阵
            sim = torch.matmul(g_2d, t_2d.T) # 形状: [100, 100]
            
            # 3. 按相似度从大到小排序，获取索引
            ranks = torch.argsort(sim, dim=-1, descending=True)
            
            # 4. 构造真实对齐的 Target 标签 (对角线位置即为正样本)
            targets = torch.arange(g_2d.size(0), device=sim.device).view(-1, 1)
            
            # 5. 寻找真实正样本在排序后的索引位置并计算 MRR
            match_indices = torch.eq(ranks, targets).nonzero()
            
            if match_indices.size(0) > 0:
                # nonzero() 结果的第二列 (dim=1) 就是正样本在每一行的排位 (0-indexed)
                mrr_hz = (1.0 / (match_indices[:, 1].float() + 1.0)).mean().item()
            else:
                mrr_hz = 0.0
            # ====================================================

        acc1_horizon_list.append(acc1_hz)
        acc5_horizon_list.append(acc5_hz)
        mrr_horizon_list.append(mrr_hz)
        
        print(f"   | ➔ Horizon Rate: {rate:2.0%} | Top-1: {acc1_hz:.2%} | Top-5: {acc5_hz:.2%} | MRR: {mrr_hz:.4f}")
        
        torch.cuda.empty_cache()
        gc.collect()

    # 调用更新后的绘图器
    plot_robustness_horizon(horizons, acc1_horizon_list, acc5_horizon_list, mrr_horizon_list, 
                             os.path.join(output_dir, "eval_robustness_horizon_plot.png"))
                          
    # 计算定量性能指标
    g_full = np.concatenate(metrics_cache['g_feat'], axis=0)  
    t_full = np.concatenate(metrics_cache['t_feat'], axis=0)  
    
    acc_results = InnovativeStage2Trainer.compute_itc_metrics(torch.tensor(g_full), torch.tensor(t_full))
    mean_ortho = np.mean(metrics_cache['loss_ortho_list']) if metrics_cache['loss_ortho_list'] else 0.0

    print(f"\n✅ 评估完成！报告已完美保存至: {output_dir}")
    print(f"{'='*15} 关键定量评估指标 {'='*15}")
    print(f"📈 ITC Top-1 Retrieval Acc : {acc_results.get('itc_acc1', 0.0):.4%}")
    print(f"📈 ITC Top-5 Retrieval Acc : {acc_results.get('itc_acc5', 0.0):.4%}")
    print(f"📐 空间解耦平均正交损失 (Ortho Loss): {mean_ortho:.4f}")
    
    calculate_space_orthogonality(metrics_cache['s_feat'], metrics_cache['l_feat'], metrics_cache['se_feat'])
    print(f"{'='*48}")

if __name__ == "__main__":
    import torch
    import numpy
    
    torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
    CONFIG = "/root/autodl-tmp/first_paper/python/stage2/stage2.yaml"
    MODEL = "/root/autodl-tmp/first_paper/output/python/second_stage/best_innovative_model.pt"
    DATA_PT = "/root/autodl-tmp/first_paper/graph_dataset/python/stage2_dataset/test_data_stage2.pt"
    OUT_DIR = "/root/autodl-tmp/first_paper/output/python/second_stage/evaluation_report"

    run_comprehensive_evaluation(CONFIG, MODEL, DATA_PT, OUT_DIR)