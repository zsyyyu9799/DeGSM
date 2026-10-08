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

        # 掩码概率配置
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

        # 完整保留你原本的历史记录字典（包括多维度重构损耗与相似度统计）
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_recon': [], 'val_recon': [],
            'train_edge': [], 'val_edge': [],  
            'train_cl': [], 'val_cl': [],
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
        num_simple_classes = 4  # 0:Neg, 1:AST, 2:CFG, 3:DFG
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
        
            simple_counts[0] = simple_counts[1:4].mean() * 1.0
            simple_counts = torch.clamp(simple_counts, min=1.0)
    
            weights = 1.0 / torch.log(simple_counts + 2.0)
            self.edge_class_weights = (weights / weights.sum() * num_simple_classes).to(self.accelerator.device)

    def _get_curriculum_factor(self):
        """课程学习因子"""
        return 1.0

    def _mask_hetero_nodes_targeted(self, x_dict):
        """统一掩码策略"""
        factor = self._get_curriculum_factor()
        x_masked_dict, mask_dict = {}, {}

        nt_to_prob = {}
        for category, nts in self.TYPE_MAPPING.items():
            for nt in nts:
                nt_to_prob[nt] = self.target_probs[category] * factor

        for node_type, x in x_dict.items():
            device = x.device
            prob = nt_to_prob.get(node_type, 0.15 * factor)  
            mask = torch.rand(x.size(0), device=device) < prob
            x_masked = x.clone()
            x_masked[mask] = 0.0
            x_masked_dict[node_type], mask_dict[node_type] = x_masked, mask
        return x_masked_dict, mask_dict

    def _augment_graph(self, data):
        p_drop_node = self.config['trainer'].get('node_drop_prob', 0.1)
        p_drop_edge = self.config['trainer'].get('edge_dropout', 0.3)
    
        x_aug_dict = {}
        edge_index_aug_dict = {}
        node_keep_masks = {}
    
        # 修复点：加入安全防御，防止单类节点特征全掉落引发均值计算返回 NaN 击穿梯度
        for nt, x in data.x_dict.items():
            mask = torch.rand(x.size(0), device=x.device) > p_drop_node
            node_keep_masks[nt] = mask
            x_aug = x.clone()
            
            if not mask.all(): 
                dropped_indices = ~mask
                mean_feat = x[mask].mean(dim=0, keepdim=True) if mask.any() else torch.zeros((1, x.size(1)), device=x.device)
                noise = 0.01 * torch.randn(dropped_indices.sum(), x.size(1), device=x.device)
                x_aug[dropped_indices] = mean_feat + noise
            
            x_aug_dict[nt] = x_aug
    
        for et, edge_index in data.edge_index_dict.items():
            src_t, rel, dst_t = et
            if edge_index.numel() == 0:
                edge_index_aug_dict[et] = edge_index
                continue
    
            src_mask = node_keep_masks[src_t][edge_index[0]]
            dst_mask = node_keep_masks[dst_t][edge_index[1]]
            combined_mask = src_mask & dst_mask
            
            filtered_edges = edge_index[:, combined_mask]
    
            if filtered_edges.numel() > 0:
                final_edges, _ = dropout_edge(filtered_edges, p=p_drop_edge, training=True)
                edge_index_aug_dict[et] = final_edges
            else:
                edge_index_aug_dict[et] = filtered_edges
    
        return x_aug_dict, edge_index_aug_dict, data.edge_attr_dict, None

    def _sample_path_pairs(self, edge_index, num_src_nodes, num_dst_nodes, num_samples=1024):
        """
        保持你原本代码所带的图路径连接采样支路
        """
        if edge_index.numel() == 0 or num_src_nodes == 0 or num_dst_nodes == 0:
            return None, None, None

        device = edge_index.device
        actual_pos_samples = min(num_samples // 2, edge_index.size(1))

        perm = torch.randperm(edge_index.size(1))[:actual_pos_samples]
        pos_src = edge_index[0][perm]
        pos_dst = edge_index[1][perm]

        neg_src = torch.randint(0, num_src_nodes, (actual_pos_samples,), device=device)
        neg_dst = torch.randint(0, num_dst_nodes, (actual_pos_samples,), device=device)

        indices_src = torch.cat([pos_src, neg_src])
        indices_dst = torch.cat([pos_dst, neg_dst])

        labels = torch.cat([
            torch.ones(pos_src.size(0), device=device),
            torch.zeros(neg_src.size(0), device=device)
        ])

        return indices_src, indices_dst, labels

    def _compute_cosine_similarity(self, pred, target):
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
        """完整保留：2x3 分维度 Loss 与 Similarity (准确率) 可视化画布布局"""
        if not self.accelerator.is_main_process or len(self.history['train_recon_structural']) == 0:
            return
        try:
            epochs = range(1, len(self.history['train_recon_structural']) + 1)
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            axes[0, 0].plot(epochs, self.history['train_recon_structural'], 'b-', label='Train (Loss)')
            axes[0, 0].plot(epochs, self.history['val_recon_structural'], 'r-', label='Val (Loss)')
            axes[0, 0].set_title('Struct Loss')
            axes[0, 0].grid(True)

            axes[0, 1].plot(epochs, self.history['train_recon_logical'], 'b-', label='Train (Loss)')
            axes[0, 1].plot(epochs, self.history['val_recon_logical'], 'r-', label='Val (Loss)')
            axes[0, 1].set_title('Logic Loss')
            axes[0, 1].grid(True)

            axes[0, 2].plot(epochs, self.history['train_recon_semantic'], 'b-', label='Train (Loss)')
            axes[0, 2].plot(epochs, self.history['val_recon_semantic'], 'r-', label='Val (Loss)')
            axes[0, 2].set_title('Semantic Loss')
            axes[0, 2].grid(True)

            axes[1, 0].plot(epochs, self.history['train_sim_structural'], 'g-', label='Train (Sim)')
            axes[1, 0].plot(epochs, self.history['val_sim_structural'], 'm-', label='Val (Sim)')
            axes[1, 0].set_title('Struct Sim (Acc)')
            axes[1, 0].set_ylim(0.0, 0.8)
            axes[1, 0].grid(True)

            axes[1, 1].plot(epochs, self.history['train_sim_logical'], 'g-', label='Train (Sim)')
            axes[1, 1].plot(epochs, self.history['val_sim_logical'], 'm-', label='Val (Sim)')
            axes[1, 1].set_title('Logic Sim (Acc)')
            axes[1, 1].set_ylim(0.0, 0.8)
            axes[1, 1].grid(True)

            axes[1, 2].plot(epochs, self.history['train_sim_semantic'], 'g-', label='Train (Sim)')
            axes[1, 2].plot(epochs, self.history['val_sim_semantic'], 'm-', label='Val (Sim)')
            axes[1, 2].set_title('Semantic Sim (Acc)')
            axes[1, 2].set_ylim(0.0, 1.0)
            axes[1, 2].grid(True)

            h0, l0 = axes[0, 0].get_legend_handles_labels()
            h1, l1 = axes[1, 0].get_legend_handles_labels()

            fig.legend(h0, l0, loc='center left', bbox_to_anchor=(0.91, 0.75), title="Loss Metrics")
            fig.legend(h1, l1, loc='center left', bbox_to_anchor=(0.91, 0.25), title="Sim Metrics")

            plt.tight_layout(rect=[0, 0, 0.9, 1])
            plt.savefig(os.path.join(self.save_dir, 'category_recon_metrics.png'))
            plt.close('all')
        except Exception as e: self.logger.warning(f"Failed to save category plot: {e}")

    def _forward_pass(self, data):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        batch_dict = {nt: data[nt].batch for nt in data.node_types}
        device = self.accelerator.device

        with self.accelerator.autocast():
            # 1. 视角 1 掩码重构
            x_v1_masked, mask_v1_dict = self._mask_hetero_nodes_targeted(data.x_dict)
            node_emb_v1, graph_emb_v1, _ = self.model(x_v1_masked, data.edge_index_dict,
                                                      edge_attr_dict=data.edge_attr_dict, batch_dict=batch_dict)

            # 2. 视角 2 图增强对比学习
            x_v2, edge_v2, attr_v2, _ = self._augment_graph(data)
            _, graph_emb_v2, _ = self.model(x_v2, edge_v2, edge_attr_dict=attr_v2, batch_dict=batch_dict)
            cl_loss = self._contrastive_loss(graph_emb_v1, graph_emb_v2)

            # 原重构任务加权映射循环
            category_weights = {
                'structural': 5.0,
                'logical': 10.0,    
                'semantic': 0.2     
            }
            recon_dict = unwrapped_model.reconstruct_node_features(node_emb_v1)
            recon_loss = torch.tensor(0.0, device=device)
            dim_metrics = {k: {'loss': [], 'sim': []} for k in self.TYPE_MAPPING.keys()}

            for nt, x_recon in recon_dict.items():
                if nt in mask_v1_dict and mask_v1_dict[nt].any():
                    m = mask_v1_dict[nt]
                    target_x, pred_x = data.x_dict[nt][m], x_recon[m]
                    type_l = F.smooth_l1_loss(pred_x.float(), target_x.float(), beta=1.0)

                    current_weight = 1.0
                    for cat_name, node_types in self.TYPE_MAPPING.items():
                        if nt in node_types:
                            current_weight = category_weights[cat_name]
                            break

                    recon_loss += type_l * current_weight

                    sim = self._compute_cosine_similarity(pred_x, target_x)
                    for k, v in self.TYPE_MAPPING.items():
                        if nt in v:
                            dim_metrics[k]['loss'].append(type_l)
                            dim_metrics[k]['sim'].append(sim)

            num_node_types = len([nt for nt in recon_dict if nt in mask_v1_dict and mask_v1_dict[nt].any()])
            if num_node_types > 0:
                recon_loss /= num_node_types

            # 3. 统一边连接任务（主任务：类型分类 | 辅助任务：CFG 逆转方向纠偏）
            cfg_key = next((et for et in data.edge_types if 'cfg' in et[1].lower()), None)
            res_type, res_dir = unwrapped_model.predict_edge_types_unified(node_emb_v1, data.edge_index_dict, cfg_key=cfg_key, batch_dict=batch_dict)
            
            total_edge_loss = torch.tensor(0.0, device=device)
            if res_type[0] is not None:
                total_edge_loss += F.cross_entropy(res_type[0], res_type[1], weight=self.edge_class_weights)
            if res_dir[0] is not None:
                total_edge_loss += 0.5 * F.cross_entropy(res_dir[0], res_dir[1])

            # 融合三大原有主力任务
            total_loss = (self.lambda_cl * cl_loss +
                          self.lambda_recon * recon_loss +
                          self.lambda_edge_logic * total_edge_loss)

        cat_results = {}
        for k in dim_metrics:
            cat_results[f'{k}_loss'] = torch.stack(dim_metrics[k]['loss']).mean().item() if dim_metrics[k]['loss'] else 0.0
            cat_results[f'{k}_sim'] = torch.stack(dim_metrics[k]['sim']).mean().item() if dim_metrics[k]['sim'] else 0.0

        return total_loss, recon_loss, total_edge_loss, cl_loss, cat_results

    def analyze_edge_predictions(self, loader, num_batches=5):
        """完整保留：分析报告中混淆矩阵式多视角性能分析分析"""
        self.model.eval()
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        type_names = {0: "Negative", 1: "AST", 2: "CFG", 3: "DFG"}
        type_stats = {i: {"correct": 0, "total": 0} for i in range(4)}
        
        dir_names = {0: "Correct", 1: "LogicErr"}
        dir_stats = {i: {"correct": 0, "total": 0} for i in range(2)}

        with torch.no_grad():
            for i, data in enumerate(loader):
                if i >= num_batches: break

                batch_dict = {nt: data[nt].batch for nt in data.node_types}
                node_emb, _, _ = self.model(data.x_dict, data.edge_index_dict,
                                            edge_attr_dict=data.edge_attr_dict, batch_dict=batch_dict)

                res_type, res_dir = unwrapped_model.predict_edge_types_unified(
                    node_emb, data.edge_index_dict, batch_dict=batch_dict
                )
                
                logits_t, targets_t = res_type
                if logits_t is not None and logits_t.numel() > 0:
                    preds_t = logits_t.argmax(dim=-1)
                    for t in type_names.keys():
                        mask = (targets_t == t)
                        type_stats[t]["total"] += mask.sum().item()
                        type_stats[t]["correct"] += (preds_t[mask] == t).sum().item()

                logits_d, targets_d = res_dir
                if logits_d is not None and logits_d.numel() > 0:
                    preds_d = logits_d.argmax(dim=-1)
                    for d in dir_names.keys():
                        mask = (targets_d == d)
                        dir_stats[d]["total"] += mask.sum().item()
                        dir_stats[d]["correct"] += (preds_d[mask] == d).sum().item()

        self.logger.info("\n" + "=" * 25 + " [TASK A: EDGE TYPE] " + "=" * 25)
        for i, name in type_names.items():
            s = type_stats[i]
            acc = (s["correct"] / s["total"]) * 100 if s["total"] > 0 else 0.0
            self.logger.info(f" -> {name:10}: Acc {acc:6.2f}% | Count: {int(s['total'])}")

        if any(s["total"] > 0 for s in dir_stats.values()):
            self.logger.info("-" * 25 + " [TASK B: DIRECTION] " + "-" * 25)
            for i, name in dir_names.items():
                s = dir_stats[i]
                acc = (s["correct"] / s["total"]) * 100 if s["total"] > 0 else 0.0
                self.logger.info(f" -> {name:10}: Acc {acc:6.2f}% | Count: {int(s['total'])}")
        self.logger.info("=" * 70)

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
            
            # 每一轮评估都输出混淆矩阵分析
            self.analyze_edge_predictions(self.valid_loader)

            if self.accelerator.is_main_process and epoch == 0:
                torch.save(self.accelerator.unwrap_model(self.model).state_dict(), os.path.join(self.save_dir, "initial_stage_model.pt"))

            # 完整归档历史快照
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

            if self.accelerator.is_main_process:
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
                self.logger.info(f"  Train: Loss {train_loss:.4f} | Recon {tm['recon']:.4f} | Edge {tm['edge']:.4f} | CL {tm['cl']:.4f}")
                self.logger.info(f"  Train_Recon: Struct_Recon {tm['structural_loss']:.4f} | Logic_Recon {tm['logical_loss']:.4f} | Semantic_Recon {tm['semantic_loss']:.4f}")
                self.logger.info(f"  Val:   Loss {val_loss:.4f} | Recon {vm['recon']:.4f} | Edge {vm['edge']:.4f} | CL {vm['cl']:.4f}")
                self.logger.info(f"  Val_Recon: Struct_Recon {vm['structural_loss']:.4f} | Logic_Recon {vm['logical_loss']:.4f} | Semantic_Recon {vm['semantic_loss']:.4f}")
                self.logger.info(f"  Val_Cat:   Struct {vm['structural_sim']:.4f} | Logic {vm['logical_sim']:.4f} | Semantic {vm['semantic_sim']:.4f}")
                self._save_loss_plot()
                self._save_category_recon_plot()
                
                early_stopping(val_loss, self.accelerator.unwrap_model(self.model))
                stop_signal = early_stopping.early_stop
            else:
                stop_signal = False
                
            stop_tensor = torch.tensor(1.0 if stop_signal else 0.0).to(self.accelerator.device)
            merged_stop = self.accelerator.reduce(stop_tensor, reduction='sum')
            
            if merged_stop > 0:
                if self.accelerator.is_main_process:
                    self.logger.info("Early stopping triggered.")
                break

            self.scheduler.step(val_loss)
            self.accelerator.wait_for_everyone()
            gc.collect()
            torch.cuda.empty_cache() 

        if self.accelerator.is_main_process and save_path:
            best_model_path = os.path.join(self.save_dir, "best_hetero_model.pt")
            final_path = Path(save_path) / "final_hetero_cge_model.pt"
            if os.path.exists(best_model_path):
                shutil.copyfile(best_model_path, final_path)
                self.logger.info(f"🏆 已将最优模型 (Best Val Loss) 保存至: {final_path}")
            else:
                torch.save(self.accelerator.unwrap_model(self.model).state_dict(), final_path)
                self.logger.info(f"💾 已将最后一轮模型保存至: {final_path}")

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
            self.save_checkpoint(model)
        elif (score < self.best_score - self.min_delta) if self.mode == 'min' else (score > self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.logger:
                self.logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        if self.save_dir:
            save_path = os.path.join(self.save_dir, "best_hetero_model.pt")
            torch.save(model.state_dict(), save_path)
            if self.logger:
                self.logger.info(f"🏆 Improved! Saved best model to {save_path}")

def load_hetero_datasets(train_path, val_path):
    t_load = torch.load(train_path, map_location='cpu', weights_only=False)
    v_load = torch.load(val_path, map_location='cpu', weights_only=False)
    t_g = t_load['graphs'] if isinstance(t_load, dict) else t_load
    v_g = v_load['graphs'] if isinstance(v_load, dict) else v_load

    def is_valid_graph(g):
        if not g.edge_stores: return False
        for store in g.edge_stores:
            if 'edge_index' in store and store.edge_index.numel() > 0: return True
        return False

    train_graphs = [g for g in t_g if is_valid_graph(g)]
    valid_graphs = [g for g in v_g if is_valid_graph(g)]

    edge_stats = {}
    for g in train_graphs:
        for et, idx in g.edge_index_dict.items():
            et_str = f"{et[0]}_{et[1]}_{et[2]}"
            edge_stats[et_str] = edge_stats.get(et_str, 0) + idx.size(1)

    return train_graphs, valid_graphs, edge_stats

if __name__ == "__main__":
    accelerator = Accelerator()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/root/autodl-tmp/first_paper/cpp/stage1/stage1.yaml')
    parser.add_argument('--metadata_path', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train_dataset, val_dataset, edge_stats = load_hetero_datasets(
        config['data']['train_path'],
        config['data']['valid_path']
    )

    metadata = None
    m_path = args.metadata_path or config['data'].get('metadata_path')
    if m_path and os.path.exists(m_path):
        metadata = torch.load(m_path, map_location='cpu', weights_only=False).get('metadata')
    if metadata is None:
        all_edge_types = set()
        all_node_types = set()
        for g in train_dataset:
            all_node_types.update(g.node_types)
            all_edge_types.update(g.edge_types)
        metadata = (sorted(list(all_node_types)), sorted(list(all_edge_types)))

    save_dir = Path(config['output']['save_dir'])
    if accelerator.is_main_process: os.makedirs(save_dir, exist_ok=True)

    model = HeteroCodeGraphEncoder(
        metadata=metadata, in_channels=768,
        hidden_channels=config['model']['hidden_channels'],
        out_channels=config['model']['out_channels'],
        gnn_type=config['model'].get('gnn_type', 'gat'), 
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
        edge_stats=edge_stats, 
        config=config
    )

    if accelerator.is_main_process:
        final_config_path = os.path.join(save_dir, 'run_config_best_snapshot.yaml')
        with open(final_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        print(f"[*] 训练参数已快照至: {final_config_path}")

    trainer.train(num_epochs=config['trainer']['num_epochs'], patience=config['trainer']['patience'], save_path=save_dir)