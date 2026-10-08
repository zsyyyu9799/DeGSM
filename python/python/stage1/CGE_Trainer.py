"""
Stage 1: Heterogeneous Code Structure & Attribute Learning (Optimization Enhanced)
1. 保持原有的双视角对比学习与逻辑门控逻辑。
2. 集成动态掩码率（Curriculum Learning）和动态 Loss 权重分配。
3. 增加辅助任务：节点类型分类，增强对逻辑/结构节点的辨识度。
4. 保持原有的日志、可视化（2x3 布局）和历史记录逻辑。
"""
import os
import sys, yaml
import gc
import argparse
import logging
from pathlib import Path
import shutil

# 核心修复
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling, dropout_edge
from tqdm.auto import tqdm
from accelerate import Accelerator
from stage1.HeteroCodeGraphEncoder import HeteroCodeGraphEncoder

try:
    from utils import outputs_dir, load_config
except ImportError:
    def outputs_dir(): return Path("./output")

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class HeteroCodeGNNTrainer:
    def __init__(self, model, optimizer, scheduler, train_loader, valid_loader, accelerator,
                 save_dir='checkpoints', lambda_recon=1.0, lambda_edge=0.5, lambda_cl=0.5,
                 mask_prob=0.2, temperature=0.2, config=None, edge_stats=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.accelerator = accelerator
        self.config = config

        self.save_dir = save_dir
        # 从 YAML 读取权重参数
        t_conf = self.config.get('trainer', {})
        self.lambda_recon = t_conf.get('lambda_recon', 1.0)
        self.lambda_edge_logic = t_conf.get('lambda_edge', 0.6)
        self.lambda_cl = t_conf.get('lambda_cl', 0.8)
        self.temperature = t_conf.get('cl_temperature', 0.07)

        # 掩码概率配置 (重点修改 1: 统一掩码策略)
        self.target_probs = {
            'structural': t_conf.get('mask_prob_struct', 0.15),
            'logical': t_conf.get('mask_prob_logic', 0.3),
            'semantic': t_conf.get('mask_prob_semantic', 0.3)
        }

        self.TYPE_MAPPING = {
            'structural': ['structure', 'statement'],
            'logical': ['expression'],
            'semantic': ['variable']
        }

        self.current_epoch = 0
        self.warmup_epochs = 30

        # 存储历史记录（保持原有结构，增加辅助任务统计）
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_recon': [], 'val_recon': [],
            'train_edge': [], 'val_edge': [],  # 此时 edge 代表统一后的逻辑边损失
            'train_cl': [], 'val_cl': [],
            'train_path': [], 'val_path': [],
            'train_recon_structural': [], 'val_recon_structural': [],
            'train_recon_logical': [], 'val_recon_logical': [],
            'train_recon_semantic': [], 'val_recon_semantic': [],
            'train_sim_structural': [], 'val_sim_structural': [],
            'train_sim_logical': [], 'val_sim_logical': [],
            'train_sim_semantic': [], 'val_sim_semantic': []
        }

        if self.accelerator.is_main_process:
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            log_filename = f'hetero_structure_learning.log'
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(message)s',
                handlers=[logging.FileHandler(os.path.join(save_dir, log_filename)), logging.StreamHandler()]
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())

        # 计算类别权重
        num_simple_classes = 5  # 0:Neg, 1:AST, 2:CFG, 3:DFG, 4:LogicErr
        simple_counts = torch.zeros(num_simple_classes)

        if edge_stats:
            for et_str, count in edge_stats.items():
                et_lower = et_str.lower()
                if 'cfg' in et_lower:
                    simple_counts[2] += count
                elif 'dfg' in et_lower:
                    simple_counts[3] += count
                elif 'ast' in et_lower:
                    simple_counts[1] += count

            # 填充负样本和逻辑错误样本的虚拟统计（用于计算权重）
            simple_counts[0] = simple_counts[2:4].mean() * 1.2
            simple_counts[4] = simple_counts[2:4].min() * 0.8
            simple_counts = torch.clamp(simple_counts, min=1.0)

            weights = 1.0 / torch.log(simple_counts + 2.0)
            self.edge_class_weights = (weights / weights.sum() * num_simple_classes).to(self.accelerator.device)
        else:
            self.edge_class_weights = None

    def _get_curriculum_factor(self):
        """课程学习因子：0.5 -> 1.0，用于控制掩码难度和 Loss 权重。"""
        # if self.current_epoch < self.warmup_epochs:
        #     return 1.0
        return 1.0 #+ 0.5 * (self.current_epoch / self.warmup_epochs)

    def _mask_hetero_nodes_targeted(self, x_dict):
        """统一掩码策略：根据 YAML 定义的节点类别比例进行针对性遮蔽。"""
        factor = self._get_curriculum_factor()
        x_masked_dict, mask_dict = {}, {}

        # 建立 node_type 到 target_prob 的映射
        nt_to_prob = {}
        for category, nts in self.TYPE_MAPPING.items():
            for nt in nts:
                nt_to_prob[nt] = self.target_probs[category] * factor

        for node_type, x in x_dict.items():
            device = x.device
            prob = nt_to_prob.get(node_type, 0.15 * factor)  # 默认比例
            mask = torch.rand(x.size(0), device=device) < prob
            x_masked = x.clone()
            x_masked[mask] = 0.0
            x_masked_dict[node_type], mask_dict[node_type] = x_masked, mask
        return x_masked_dict, mask_dict

    def _augment_graph(self, data):
        """
        对比学习增强：形状安全版本 (Safe Node Masking + Edge Filtering)
        保持 Tensor 维度不变，防止 batch_dict 越界导致的崩溃。
        """
        p_drop_node = self.config['trainer'].get('node_drop_prob', 0.1)
        p_drop_edge = self.config['trainer'].get('edge_dropout', 0.3)

        x_aug_dict = {}
        edge_index_aug_dict = {}
        edge_attr_aug_dict = {}
        node_keep_masks = {}

        # 1. 节点层面：生成保留掩码，但不改变 Tensor 形状
        for nt, x in data.x_dict.items():
            # 生成掩码：True 为保留，False 为“掉落”
            mask = torch.rand(x.size(0), device=x.device) > p_drop_node
            node_keep_masks[nt] = mask

            x_aug = x.clone()
            x_aug[~mask] = 0.0  # 将“掉落”的节点特征清零
            x_aug_dict[nt] = x_aug

        # 2. 边层面：过滤掉与“掉落节点”相连的边 + 随机 Edge Dropout
        for et, edge_index in data.edge_index_dict.items():
            src_t, rel, dst_t = et
            if edge_index.numel() == 0:
                edge_index_aug_dict[et] = edge_index
                continue

            # 仅保留两端节点都处于 mask 状态的边
            src_mask = node_keep_masks[src_t][edge_index[0]]
            dst_mask = node_keep_masks[dst_t][edge_index[1]]
            combined_mask = src_mask & dst_mask

            filtered_edges = edge_index[:, combined_mask]

            # 随机丢弃剩下的边
            if filtered_edges.numel() > 0:
                final_edges, drop_mask = dropout_edge(filtered_edges, p=p_drop_edge, training=True)
                edge_index_aug_dict[et] = final_edges
                if et in data.edge_attr_dict:
                    # 同步过滤边特征
                    edge_attr_aug_dict[et] = data.edge_attr_dict[et][combined_mask][drop_mask]
            else:
                edge_index_aug_dict[et] = filtered_edges

        return x_aug_dict, edge_index_aug_dict, edge_attr_aug_dict, None

    def _mask_hetero_nodes_random(self, x_dict, mask_prob):
        x_masked_dict, mask_dict = {}, {}
        for node_type, x in x_dict.items():
            device = x.device
            mask = torch.rand(x.size(0), device=device) < mask_prob
            x_masked = x.clone()
            x_masked[mask] = 0.0
            x_masked_dict[node_type], mask_dict[node_type] = x_masked, mask
        return x_masked_dict, mask_dict

    def _sample_path_pairs(self, edge_index, num_src_nodes, num_dst_nodes, num_samples=1024):
        """
        优化后的采样：增加采样数，并确保正负样本平衡
        """
        if edge_index.numel() == 0 or num_src_nodes == 0 or num_dst_nodes == 0:
            return None, None, None

        device = edge_index.device
        # 增加采样数量到 1024 以平滑梯度
        actual_pos_samples = min(num_samples // 2, edge_index.size(1))

        # 1. 正样本：从现有 CFG 边中随机采样
        perm = torch.randperm(edge_index.size(1))[:actual_pos_samples]
        pos_src = edge_index[0][perm]
        pos_dst = edge_index[1][perm]

        # 2. 负样本：采用更严谨的随机采样
        # 避免只采样到编号靠前的节点，使用与正样本同等规模的负样本
        neg_src = torch.randint(0, num_src_nodes, (actual_pos_samples,), device=device)
        neg_dst = torch.randint(0, num_dst_nodes, (actual_pos_samples,), device=device)

        # 3. 合并
        indices_src = torch.cat([pos_src, neg_src])
        indices_dst = torch.cat([pos_dst, neg_dst])

        # 构造标签：正样本为 1，负样本为 0
        labels = torch.cat([
            torch.ones(pos_src.size(0), device=device),
            torch.zeros(neg_src.size(0), device=device)
        ])

        return indices_src, indices_dst, labels

    def _compute_cosine_similarity(self, pred, target):
        """计算预测值与真实值的余弦相似度作为准确率参考。"""
        if pred.size(0) == 0: return torch.tensor(0.0, device=pred.device)
        return F.cosine_similarity(pred.float(), target.float(), dim=-1).mean()

    def _contrastive_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss

    def _save_loss_plot(self):
        if not self.accelerator.is_main_process or len(self.history['train_loss']) == 0:
            return
        try:
            epochs = range(1, len(self.history['train_loss']) + 1)
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1); plt.plot(epochs, self.history['train_loss'], 'b-', label='Train'); plt.plot(epochs, self.history['val_loss'], 'r-', label='Val'); plt.title('Total Loss'); plt.legend(); plt.grid(True)
            plt.subplot(2, 2, 2); plt.plot(epochs, self.history['train_recon'], 'b-', label='Train'); plt.plot(epochs, self.history['val_recon'], 'r-', label='Val'); plt.title('Recon Loss'); plt.legend(); plt.grid(True)
            plt.subplot(2, 2, 3); plt.plot(epochs, self.history['train_edge'], 'b-', label='Train'); plt.plot(epochs, self.history['val_edge'], 'r-', label='Val'); plt.title('Edge Loss'); plt.legend(); plt.grid(True)
            plt.subplot(2, 2, 4); plt.plot(epochs, self.history['train_cl'], 'b-', label='Train'); plt.plot(epochs, self.history['val_cl'], 'r-', label='Val'); plt.title('CL Loss'); plt.legend(); plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'training_metrics.png'))
            plt.close('all')
        except Exception as e: self.logger.warning(f"Failed to save plot: {e}")
    def _save_category_recon_plot(self):
        """生成分维度的 Loss 和 Similarity (准确率) 对比图。"""
        if not self.accelerator.is_main_process or len(self.history['train_recon_structural']) == 0:
            return
        try:
            epochs = range(1, len(self.history['train_recon_structural']) + 1)
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            # --- 第一行：Loss (蓝/红) ---
            axes[0, 0].plot(epochs, self.history['train_recon_structural'], 'b-', label='Train (Loss)')
            axes[0, 0].plot(epochs, self.history['val_recon_structural'], 'r-', label='Val (Loss)')
            axes[0, 0].set_title('Struct Loss');
            axes[0, 0].grid(True)

            axes[0, 1].plot(epochs, self.history['train_recon_logical'], 'b-', label='Train (Loss)')
            axes[0, 1].plot(epochs, self.history['val_recon_logical'], 'r-', label='Val (Loss)')
            axes[0, 1].set_title('Logic Loss');
            axes[0, 1].grid(True)

            axes[0, 2].plot(epochs, self.history['train_recon_semantic'], 'b-', label='Train (Loss)')
            axes[0, 2].plot(epochs, self.history['val_recon_semantic'], 'r-', label='Val (Loss)')
            axes[0, 2].set_title('Semantic Loss');
            axes[0, 2].grid(True)

            # --- 第二行：Similarity (绿/紫) ---
            axes[1, 0].plot(epochs, self.history['train_sim_structural'], 'g-', label='Train (Sim)')
            axes[1, 0].plot(epochs, self.history['val_sim_structural'], 'm-', label='Val (Sim)')
            axes[1, 0].set_title('Struct Sim (Acc)');
            axes[1, 0].set_ylim(0.0, 0.8);
            axes[1, 0].grid(True)

            axes[1, 1].plot(epochs, self.history['train_sim_logical'], 'g-', label='Train (Sim)')
            axes[1, 1].plot(epochs, self.history['val_sim_logical'], 'm-', label='Val (Sim)')
            axes[1, 1].set_title('Logic Sim (Acc)');
            axes[1, 1].set_ylim(0.0, 0.8);
            axes[1, 1].grid(True)

            axes[1, 2].plot(epochs, self.history['train_sim_semantic'], 'g-', label='Train (Sim)')
            axes[1, 2].plot(epochs, self.history['val_sim_semantic'], 'm-', label='Val (Sim)')
            axes[1, 2].set_title('Semantic Sim (Acc)');
            axes[1, 2].set_ylim(0.0, 1.0);
            axes[1, 2].grid(True)

            # --- 关键修改：右侧分行显示图例 ---

            # 获取第一行的句柄和标签
            h0, l0 = axes[0, 0].get_legend_handles_labels()
            # 获取第二行的句柄和标签
            h1, l1 = axes[1, 0].get_legend_handles_labels()

            # 创建第一个全局图例（对应第一行），放在右侧上方
            leg1 = fig.legend(h0, l0, loc='center left', bbox_to_anchor=(0.91, 0.75), title="Loss Metrics")

            # 创建第二个全局图例（对应第二行），放在右侧下方
            # 注意：在同一个 fig 上添加第二个 legend 需要用到 add_artist
            leg2 = fig.legend(h1, l1, loc='center left', bbox_to_anchor=(0.91, 0.25), title="Sim Metrics")

            # 调整布局：rect=[左, 下, 右, 上]，这里 0.9 是为了给右侧留出 10% 的宽度放图例
            plt.tight_layout(rect=[0, 0, 0.9, 1])

            plt.savefig(os.path.join(self.save_dir, 'category_recon_metrics.png'))
            plt.close('all')
        except Exception as e: self.logger.warning(f"Failed to save category plot: {e}")

    def _forward_pass(self, data):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        batch_dict = {nt: data[nt].batch for nt in data.node_types}
        device = self.accelerator.device
        factor = self._get_curriculum_factor()

        stage1_threshold = 20

        if self.current_epoch < stage1_threshold:
            # 第一阶段：专注局部结构 (Recon + Edge)
            current_lambda_cl = 0.0
            current_lambda_recon = self.lambda_recon
            current_lambda_edge = self.lambda_edge_logic
        else:
            # 第二阶段：引入全局对比学习 (CL)
            current_lambda_cl = self.lambda_cl
            current_lambda_recon = self.lambda_recon
            current_lambda_edge = self.lambda_edge_logic

        with self.accelerator.autocast():
            # 1. 视角 1 (用于 Recon 和 Edge 任务)
            x_v1_masked, mask_v1_dict = self._mask_hetero_nodes_targeted(data.x_dict)
            node_emb_v1, graph_emb_v1, _ = self.model(x_v1_masked, data.edge_index_dict,
                                                      edge_attr_dict=data.edge_attr_dict, batch_dict=batch_dict)

            # 2. 视角 2 (仅在开启 CL 时进行增强和计算)
            cl_loss = torch.tensor(0.0, device=device)
            if current_lambda_cl > 0:
                x_v2, edge_v2, attr_v2, _ = self._augment_graph(data)
                _, graph_emb_v2, _ = self.model(x_v2, edge_v2, edge_attr_dict=attr_v2, batch_dict=batch_dict)
                cl_loss = self._contrastive_loss(graph_emb_v1, graph_emb_v2)

            # 2. 修改重构损失计算循环
            category_weights = {
                    'structural': 2.0,
                    'logical': 4,    # 显著提高逻辑节点的关注度
                    'semantic': 0.3     # 语义节点已达 0.9+，可以适当降低权重
                }
            recon_dict = unwrapped_model.reconstruct_node_features(node_emb_v1)
            recon_loss = torch.tensor(0.0, device=device)
            dim_metrics = {k: {'loss': [], 'sim': []} for k in self.TYPE_MAPPING.keys()}
            total_weighted_count = 0.0  # 用于加权平均

            for nt, x_recon in recon_dict.items():
                if nt in mask_v1_dict:
                    m = mask_v1_dict[nt]
                    if m.any():
                        target_x, pred_x = data.x_dict[nt][m], x_recon[m]
                        # 计算原始 Smooth L1 Loss
                        type_l = F.smooth_l1_loss(pred_x.float(), target_x.float(), beta=1.0)

                        # --- 核心修改：查找类别并应用权重 ---
                        current_weight = 1.0
                        for cat_name, node_types in self.TYPE_MAPPING.items():
                            if nt in node_types:
                                current_weight = category_weights[cat_name]
                                break

                        # 应用加权损失
                        recon_loss += type_l * current_weight
                        total_weighted_count += current_weight # 或者使用 num_node_types，取决于你想要的缩放尺度

                        # 记录指标（保持原样）
                        sim = self._compute_cosine_similarity(pred_x, target_x)
                        for k, v in self.TYPE_MAPPING.items():
                            if nt in v:
                                dim_metrics[k]['loss'].append(type_l)
                                dim_metrics[k]['sim'].append(sim)

            # 归一化处理
            num_node_types = len([nt for nt in recon_dict if nt in mask_v1_dict and mask_v1_dict[nt].any()])
            if num_node_types > 0:
                recon_loss /= num_node_types # 建议按节点类型数量平均，或者按权重总和平均

            # 3. 统一边逻辑任务 (类型分类 + CFG 方向辨识)
            cfg_key = next((et for et in data.edge_types if 'cfg' in et[1].lower()), None)
            logits_e, targets_e = unwrapped_model.predict_edge_types_unified(node_emb_v1, data.edge_index_dict,
                                                                             cfg_key=cfg_key)

            unified_edge_loss = torch.tensor(0.0, device=device)
            if logits_e.numel() > 0:
                unified_edge_loss = F.cross_entropy(logits_e, targets_e, weight=self.edge_class_weights)

            # 5. 总损失：使用动态调整后的权重
            total_loss = (current_lambda_cl * cl_loss +
                          current_lambda_recon * recon_loss +
                          current_lambda_edge * unified_edge_loss)

        cat_results = {}
        for k in dim_metrics:
            # 同时保存损失值和相似度
            cat_results[f'{k}_loss'] = torch.stack(dim_metrics[k]['loss']).mean().item() if dim_metrics[k][
                'loss'] else 0.0
            cat_results[f'{k}_sim'] = torch.stack(dim_metrics[k]['sim']).mean().item() if dim_metrics[k]['sim'] else 0.0

        return total_loss, recon_loss, unified_edge_loss, cl_loss, cat_results

    def analyze_edge_predictions(self, loader, num_batches=5):
        """
        深入分析 Edge 任务：计算各类型边的准确率分布
        """
        self.model.eval()
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # 类别索引映射
        idx_to_name = {0: "Negative", 1: "AST", 2: "CFG", 3: "DFG", 4: "LogicErr"}
        confusion_stats = {i: {"correct": 0, "total": 0} for i in range(5)}

        with torch.no_grad():
            for i, data in enumerate(loader):
                if i >= num_batches: break

                # 前向传播获取 Embedding
                batch_dict = {nt: data[nt].batch for nt in data.node_types}
                node_emb, _, _ = self.model(data.x_dict, data.edge_index_dict,
                                            edge_attr_dict=data.edge_attr_dict, batch_dict=batch_dict)

                # 获取预测结果
                cfg_key = next((et for et in data.edge_types if 'cfg' in et[1].lower()), None)
                logits_e, targets_e = unwrapped_model.predict_edge_types_unified(node_emb, data.edge_index_dict,
                                                                                 cfg_key=cfg_key)

                if logits_e.numel() > 0:
                    preds = logits_e.argmax(dim=-1)
                    for t in range(5):
                        mask = (targets_e == t)
                        confusion_stats[t]["total"] += mask.sum().item()
                        confusion_stats[t]["correct"] += (preds[mask] == t).sum().item()

        # 打印分析结果
        self.logger.info("\n" + "=" * 30 + " EDGE TASK ANALYSIS " + "=" * 30)
        for i, name in idx_to_name.items():
            stats = confusion_stats[i]
            acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0.0
            self.logger.info(f" -> Category [{name:10}]: Acc {acc:6.2f}% | Count: {int(stats['total'])}")
        self.logger.info("=" * 80)

    def train_epoch(self):
        self.model.train()
        total_m = {k: 0.0 for k in ['loss', 'recon', 'edge', 'cl', 'structural_loss', 'logical_loss', 'semantic_loss',
                                   'structural_sim', 'logical_sim', 'semantic_sim']}
        for data in tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} Train", disable=not self.accelerator.is_main_process):
            self.optimizer.zero_grad()
            loss, rl, el, cl, cats = self._forward_pass(data)
            self.accelerator.backward(loss)
            self.optimizer.step()
            total_m['loss'] += loss.item(); total_m['recon'] += rl.item(); total_m['edge'] += el.item()
            total_m['cl'] += cl.item()
            for k, v in cats.items(): total_m[k] += v
        n = len(self.train_loader)
        return total_m['loss']/n, {k: v/n for k, v in total_m.items()}

    def eval_epoch(self):
        self.model.eval()
        total_m = {k: 0.0 for k in ['loss', 'recon', 'edge', 'cl', 'structural_loss', 'logical_loss', 'semantic_loss',
                                   'structural_sim', 'logical_sim', 'semantic_sim']}
        with torch.no_grad():
            for data in tqdm(self.valid_loader, desc=f"Epoch {self.current_epoch+1} Eval", disable=not self.accelerator.is_main_process):
                loss, rl, el, cl, cats = self._forward_pass(data)
                total_m['loss'] += loss.item(); total_m['recon'] += rl.item(); total_m['edge'] += el.item()
                total_m['cl'] += cl.item()
                for k, v in cats.items(): total_m[k] += v
        n = len(self.valid_loader)
        return total_m['loss']/n, {k: v/n for k, v in total_m.items()}

    def train(self, num_epochs=100, patience=10, save_path=None):
        early_stopping = EarlyStopping(patience=patience, mode='min', logger=self.logger, save_dir=self.save_dir) if self.accelerator.is_main_process else None
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            train_loss, tm = self.train_epoch()
            val_loss, vm = self.eval_epoch()
            
            self.analyze_edge_predictions(self.valid_loader)

            if self.accelerator.is_main_process and epoch == 0:
                torch.save(self.accelerator.unwrap_model(self.model).state_dict(), os.path.join(self.save_dir, "initial_stage_model.pt"))

            # 历史记录逻辑
            self.history['train_loss'].append(train_loss); self.history['val_loss'].append(val_loss)
            self.history['train_recon'].append(tm['recon']); self.history['val_recon'].append(vm['recon'])
            self.history['train_edge'].append(tm['edge']); self.history['val_edge'].append(vm['edge'])
            self.history['train_cl'].append(tm['cl']); self.history['val_cl'].append(vm['cl'])
            self.history['train_recon_structural'].append(tm['structural_loss']); self.history['val_recon_structural'].append(vm['structural_loss'])
            self.history['train_recon_logical'].append(tm['logical_loss']); self.history['val_recon_logical'].append(vm['logical_loss'])
            self.history['train_recon_semantic'].append(tm['semantic_loss']); self.history['val_recon_semantic'].append(vm['semantic_loss'])
            self.history['train_sim_structural'].append(tm['structural_sim']); self.history['val_sim_structural'].append(vm['structural_sim'])
            self.history['train_sim_logical'].append(tm['logical_sim']); self.history['val_sim_logical'].append(vm['logical_sim'])
            self.history['train_sim_semantic'].append(tm['semantic_sim']); self.history['val_sim_semantic'].append(vm['semantic_sim'])

            self.scheduler.step(val_loss)
            if self.accelerator.is_main_process:
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
                self.logger.info(f"  Train: Loss {train_loss:.4f} | Recon {tm['recon']:.4f} | Edge {tm['edge']:.4f} | CL {tm['cl']:.4f}")
                self.logger.info(f"  Train_Recon: Struct_Recon {tm['structural_loss']:.4f} | Logic_Recon {tm['logical_loss']:.4f} | Semantic_Recon {tm['semantic_loss']:.4f}")
                self.logger.info(f"  Train_Cat: Struct {tm['structural_sim']:.4f} | Logic {tm['logical_sim']:.4f} | Semantic {tm['semantic_sim']:.4f}")
                self.logger.info(f"  Val:   Loss {val_loss:.4f} | Recon {vm['recon']:.4f} | Edge {vm['edge']:.4f} | CL {vm['cl']:.4f}")
                self.logger.info(f"  Val_Recon: Struct_Recon {vm['structural_loss']:.4f} | Logic_Recon {vm['logical_loss']:.4f} | Semantic_Recon {vm['semantic_loss']:.4f}")
                self.logger.info(f"  Val_Cat:   Struct {vm['structural_sim']:.4f} | Logic {vm['logical_sim']:.4f} | Semantic {vm['semantic_sim']:.4f}")
                self._save_loss_plot()
                self._save_category_recon_plot()

                early_stopping(val_loss, self.accelerator.unwrap_model(self.model))
                if self.accelerator.is_main_process:
                    if self.current_epoch == 20: 
                        self.logger.info("--- Switching to Stage 2: Resetting EarlyStopping ---")
                        early_stopping.best_score = None
                        early_stopping.counter = 0
                
                    early_stopping(val_loss, self.accelerator.unwrap_model(self.model))

                self.accelerator.wait_for_everyone()
                
            gc.collect();
            torch.cuda.empty_cache()

        # === 核心修改部分：训练结束后保存最优模型 ===
        if self.accelerator.is_main_process and save_path:
            best_model_path = os.path.join(self.save_dir, "best_hetero_model.pt")
            final_path = Path(save_path) / "final_hetero_cge_model.pt"

            if os.path.exists(best_model_path):
                # 如果训练中产生了最佳模型（由 EarlyStopping 保存），则将其重命名/复制为最终模型
                shutil.copyfile(best_model_path, final_path)
                self.logger.info(f"🏆 已将最优模型 (Best Val Loss) 保存至: {final_path}")
            else:
                # 万一没进过 EarlyStopping 的保存逻辑（比如只有 1 个 epoch），则保存当前权重
                torch.save(self.accelerator.unwrap_model(self.model).state_dict(), final_path)
                self.logger.info(f"💾 已将最后一轮模型保存至: {final_path}")
            self._save_loss_plot();
            self._save_category_recon_plot()

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='min', logger=None, save_dir=None):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.logger = logger
        self.save_dir = save_dir

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
        elif (score < self.best_score - self.min_delta) if self.mode == 'min' else (score > self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
            if self.save_dir:
                torch.save(model.state_dict(), os.path.join(self.save_dir, "best_hetero_model.pt"))
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def load_hetero_datasets(train_path, val_path):
    t_load = torch.load(train_path, map_location='cpu', weights_only=False)
    v_load = torch.load(val_path, map_location='cpu', weights_only=False)
    t_g = t_load['graphs'] if isinstance(t_load, dict) else t_load
    v_g = v_load['graphs'] if isinstance(v_load, dict) else v_load

    # --- 修复后的过滤逻辑 ---
    def is_valid_graph(g):
        # 1. 检查是否有任何 edge_store
        if not g.edge_stores:
            return False
        # 2. 检查所有边存储中是否有实际的边索引
        for store in g.edge_stores:
            if 'edge_index' in store and store.edge_index.numel() > 0:
                return True
        return False

    train_graphs = [g for g in t_g if is_valid_graph(g)]
    valid_graphs = [g for g in v_g if is_valid_graph(g)]

    # --- 新增：统计边类型频率 ---
    edge_stats = {}
    for g in train_graphs:
        for et, idx in g.edge_index_dict.items():
            et_str = f"{et[0]}_{et[1]}_{et[2]}"
            edge_stats[et_str] = edge_stats.get(et_str, 0) + idx.size(1)

    return train_graphs, valid_graphs, edge_stats

if __name__ == "__main__":
    accelerator = Accelerator()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/root/autodl-tmp/first_paper/python/stage1/stage1.yaml')
    parser.add_argument('--metadata_path', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    # 获取统计信息
    train_dataset, val_dataset, edge_stats = load_hetero_datasets(
        config['data']['train_path'],
        config['data']['valid_path']
    )

    metadata = None
    m_path = args.metadata_path or config['data'].get('metadata_path')
    if m_path and os.path.exists(m_path):
        metadata = torch.load(m_path, map_location='cpu', weights_only=False).get('metadata')
    if metadata is None:
        # 不要只取第一张图 [0]
        # 建议扫描整个数据集以获取完整的边类型并集
        all_edge_types = set()
        all_node_types = set()
        for g in train_dataset:
            all_node_types.update(g.node_types)
            all_edge_types.update(g.edge_types)
        metadata = (sorted(list(all_node_types)), sorted(list(all_edge_types)))

    save_dir = Path(config['output']['save_dir'])
    if accelerator.is_main_process: os.makedirs(save_dir, exist_ok=True)

    # 初始化 HeteroCodeGraphEncoder，它内部会根据 gnn_type 加载新型门控算子
    model = HeteroCodeGraphEncoder(
        metadata=metadata, in_channels=768,
        hidden_channels=config['model']['hidden_channels'],
        out_channels=config['model']['out_channels'],
        gnn_type=config['model'].get('gnn_type', 'gat'), # 建议用 gat 以触发 EdgeLogicGate
        num_layers=config['model'].get('num_layers', 4),
        dropout=config['model'].get('dropout', 0.1),
        num_heads=config['model'].get('num_heads', 6)
    )

    train_loader = DataLoader(train_dataset, batch_size=config['trainer']['batch_size'], shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=config['trainer']['batch_size'], shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['trainer']['lr'], weight_decay=config['trainer']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )

    trainer = HeteroCodeGNNTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        train_loader=train_loader, valid_loader=valid_loader,
        accelerator=accelerator, save_dir=save_dir,
        lambda_recon=config['trainer'].get('lambda_recon', 1.0),
        lambda_edge=config['trainer'].get('lambda_edge', 0.8),
        lambda_cl=config['trainer'].get('lambda_cl', 0.6),
        mask_prob=config['trainer'].get('mask_prob', 0.2),
        edge_stats=edge_stats, # 传入统计信息
        config=config
    )

    # ================= 核心添加部分 =================
    if accelerator.is_main_process:
        # 将本次运行的完整 config 保存为 yaml 文件到 checkpoints 目录
        import yaml
        final_config_path = os.path.join(save_dir, 'run_config_best_snapshot.yaml')
        with open(final_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        print(f"[*] 训练参数已快照至: {final_config_path}")
    # ===============================================
    trainer.train(num_epochs=config['trainer']['num_epochs'], patience=config['trainer']['patience'], save_path=save_dir)