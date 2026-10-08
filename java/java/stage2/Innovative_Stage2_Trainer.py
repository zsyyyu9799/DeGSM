import os

os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import warnings
import logging
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from pathlib import Path
import copy
import yaml
import random
import io
from stage2.Stage2_AlignmentBridge import InnovativeStage2Bridge
from transformers import logging as transformers_logging

if sys.platform.startswith('win'):
    # 强制标准输出使用 utf-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 屏蔽警告
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

# 确保路径包含项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class SinglePtDataset(Dataset):
    """适配加载真实数据列表的 Dataset"""

    def __init__(self, pt_path, tokenizer_path, max_txt_len=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_txt_len = max_txt_len
        # 直接加载由分割脚本生成的完整列表
        self.samples = torch.load(pt_path, map_location='cpu', weights_only=True)
        print(f"成功加载数据集 {pt_path}, 样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text_inputs = self.tokenizer(
            item['code'],
            padding='max_length',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt"
        )
        return {
            'node_embs': torch.as_tensor(item['node_embs'], dtype=torch.float32),
            'text_input_ids': text_inputs.input_ids.squeeze(0),
            'text_atts': text_inputs.attention_mask.squeeze(0),
            'raw_code': item['code']
        }


def collate_fn(batch):
    # 处理节点嵌入的 Padding
    max_nodes = max([b['node_embs'].size(0) for b in batch])
    node_dim = batch[0]['node_embs'].size(1)
    padded_nodes = torch.zeros(len(batch), max_nodes, node_dim)

    for i, b in enumerate(batch):
        padded_nodes[i, :b['node_embs'].size(0)] = b['node_embs']

    return {
        'node_embs': padded_nodes,
        'text_input_ids': torch.stack([b['text_input_ids'] for b in batch]),
        'text_atts': torch.stack([b['text_atts'] for b in batch]),
        'raw_code': [b['raw_code'] for b in batch]
    }


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
        elif (score < self.best_score - self.min_delta) if self.mode == 'min' else (
                score > self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
            if self.save_dir:
                torch.save(model.state_dict(), os.path.join(self.save_dir, "best_innovative_model.pt"))
                if self.logger: self.logger.info(f"Saved Best Model to {self.save_dir}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.logger.info(f"提前停止")


class InnovativeStage2Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, valid_loader, accelerator, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.accelerator = accelerator
        self.config = config

        # 定义最终目标权重 (Target 阶段)
        self.target_lambdas = config['trainer'].get('lambdas', {
            'itc': 1.0,
            'gtg': 1.2,
            'itm': 1.2, 'ortho': 0.5
        })

        self.save_dir = Path(config['output']['save_dir'])
        self.source_printed = False

        # 统一指标键值
        self.metric_keys = ['itc', 'gtg', 'itm', 'ortho', 'itc_acc1', 'itc_acc5']
        self.history = {f"train_{k}": [] for k in self.metric_keys}
        self.history.update({f"val_{k}": [] for k in self.metric_keys})
        self.history.update({f"val_{k}": [] for k in ['itc_acc1', 'itc_acc5']})
        self.history.update({'train_loss': [], 'val_loss': []})

        if self.accelerator.is_main_process:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self._setup_logging()

    def _update_lambdas(self, epoch, total_epochs):
        if epoch < 25:
            # 阶段0：纯对齐预热
            lambdas = {'itc': 0.5, 'itm': 0.5, 'gtg': 2.5, 'ortho': 0.05}
            phase_name = "Syntax Startup"

        elif epoch < 60:
            # 阶段1：逐步加入生成，但 ITC 仍为主
            lambdas = {
                'itc': 1.5,
                'itm': 0.8,
                'gtg': 1.2, 'ortho': 0.2
            }
            phase_name = "Alignment Warmup"

        elif epoch < 100:
            # 阶段2：平衡三个阶段
            lambdas = {'itc': 1.2, 'itm': 1.5, 'gtg': 1.2, 'ortho': 0.3}
            phase_name = "stage2 Task Sprint"

        elif epoch < 150:
            # 阶段3：最终微调，可稍微提高生成权重
            lambdas = {'itc': 1.0, 'itm': 1.5, 'gtg': 1.5, 'ortho': 0.4}
            phase_name = "Final Refinement"
        else:
            lambdas = {'itc': 1.0, 'itm': 1.3, 'gtg': 2.0, 'ortho': 0.5}  # 大幅提高生成权重
            phase_name = "Final Refinement"

        self.lambdas = lambdas

        if self.accelerator.is_main_process:
            lambda_str = ", ".join([f"{k}: {v:.3f}" for k, v in self.lambdas.items()])
            self.logger.info(f"Epoch {epoch:03d}|{phase_name:<18}|Lambdas: {lambda_str}")

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                # 显式指定编码为 utf-8
                logging.FileHandler(self.save_dir / "training_innovative.log", encoding='utf-8'),
                logging.StreamHandler(sys.stdout)  # 确保流指向已经设置为 utf-8 的 stdout
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _save_all_plots(self):
        """整合后的绘图函数，不再绘制 Motif"""
        if not self.accelerator.is_main_process or len(self.history['train_loss']) == 0:
            return
        try:
            epochs = range(1, len(self.history['train_loss']) + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, self.history['train_loss'], 'b-', label='Train Total Loss')
            plt.plot(epochs, self.history['val_loss'], 'r-', label='Val Total Loss')
            plt.title('Stage 2 Alignment Total Loss')
            plt.legend();
            plt.grid(True)
            plt.savefig(self.save_dir / "loss_plot.png")
            plt.close()

            # 2. 修改中间部分：使用 plt 独立绘制 ITC Accuracy 曲线
            if len(self.history.get('val_itc_acc1', [])) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, self.history['val_itc_acc1'], 'g-o', label='Val Top-1 Acc')
                plt.plot(epochs, self.history['val_itc_acc5'], 'b-s', label='Val Top-5 Acc')
                plt.title('ITC Retrieval Accuracy (Val Only)')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy (%)')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend()
                plt.savefig(self.save_dir / "ITC_acc_vs.png")
                plt.close()

            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            metrics = ['itc', 'gtg', 'itm', 'ortho']
            titles = ['ITC Loss', 'GTG Loss', 'ITM Loss', 'Ortho Loss']
            for i, (m, t) in enumerate(zip(metrics, titles)):
                ax = axes[i // 2, i % 2]
                ax.plot(epochs, self.history[f'train_{m}'], 'b-', label='Train')
                ax.plot(epochs, self.history[f'val_{m}'], 'r-', label='Val')
                ax.set_title(t)
                ax.legend();
                ax.grid(True)
            plt.tight_layout()
            plt.savefig(self.save_dir / 'training_metrics.png')
            plt.close('all')
        except Exception as e:
            self.logger.warning(f"Failed to save plots: {e}")

    def _create_monitor_snapshot(self, epoch, samples, metrics):
        """
        在验证阶段创建监控快照：保存相似度热力图并分析正负样本得分
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        import os
        import torch

        # 1. 路径防御性处理
        base_dir = self.config['trainer'].get('save_dir',
                                              self.config['trainer'].get('output_dir', './output'))
        save_path = os.path.join(base_dir, 'snapshots')
        os.makedirs(save_path, exist_ok=True)

        # 4. GTG 生成预览
        if epoch % 5 == 0:  # 没必要每轮都跑 generate，比较耗时
            self.model.eval()
            with torch.no_grad():
                # 获取当前 batch 的生成结果
                # 注意：从 dataset 获取 tokenizer
                tokenizer = self.valid_loader.dataset.tokenizer if hasattr(self.valid_loader.dataset,
                                                                           'tokenizer') else self.valid_loader.dataset.dataset.tokenizer

                # 调用新写的 generate
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                generated_codes = unwrapped_model.generate(
                    samples,
                    tokenizer=tokenizer,
                    num_beams=3,
                    max_length=512
                )

                self.logger.info(f"\n[ Epoch {epoch} Real Generation Preview ]:")
                self.logger.info(f"Target: {samples['raw_code'][0][:150]}...")
                self.logger.info(f"Generated: {generated_codes[0][:150]}...")

    def _plot_itc_histogram(self, epoch, graph_feat, text_feat):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os

        # 1. 计算所有成对的相似度 [BS, BS]
        scores = torch.matmul(graph_feat, text_feat.T).detach().cpu().float()
        bs = scores.size(0)

        # 2. 提取正样本和负样本得分
        pos_scores = torch.diagonal(scores).numpy()
        mask = torch.eye(bs).bool()
        neg_scores = scores[~mask].numpy()

        # 3. 绘图
        plt.figure(figsize=(10, 6))
        sns.histplot(pos_scores, color="skyblue", label="Positive Pairs", kde=True, stat="probability")
        sns.histplot(neg_scores, color="salmon", label="Negative Pairs", kde=True, stat="probability")

        plt.axvline(pos_scores.mean(), color='blue', linestyle='--', label=f'Pos Mean: {pos_scores.mean():.2f}')
        plt.axvline(neg_scores.mean(), color='red', linestyle='--', label=f'Neg Mean: {neg_scores.mean():.2f}')

        plt.title(f"ITC Score Distribution (Epoch {epoch})")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Probability Density")
        plt.legend()

        save_path = os.path.join(self.config['trainer']['output_dir'], 'snapshots')

        img_name = f"itc_hist_e{epoch}.png"
        plt.savefig(os.path.join(save_path, img_name))
        plt.close()
        self.logger.info(f"ITC saved to {os.path.join(save_path, img_name)}")

    @staticmethod
    def compute_itc_metrics(graph_feats, text_feats, topk=(1, 5)):
        """
        修复版：支持 [N, num_q, D] 维度的图特征
        graph_feats: [N, 32, D]
        text_feats: [N, D]
        """
        with torch.no_grad():
            # 1. 计算原始相似度矩阵 [N_graph, num_q, N_text]
            # graph_feats: [N, 32, D], text_feats.T: [D, N]
            sim_q2t = torch.matmul(graph_feats, text_feats.T)

            # 2. 压缩 Query 维度 (逻辑同 Bridge 中的 compute_itc)
            # Graph -> Text 相似度 [N, N]
            sim_g2t, _ = sim_q2t.max(dim=1)
            # Text -> Graph 相似度 [N, N]
            # 先转置为 [N_text, N_graph, num_q] 再取 Max
            sim_t2g, _ = sim_q2t.permute(2, 0, 1).max(dim=-1)

            n = sim_g2t.size(0)
            labels = torch.arange(n).to(graph_feats.device)

            results = {}
            for k in topk:
                if k > n: continue
                # Graph -> Text Accuracy
                _, pred_g = sim_g2t.topk(k, dim=1)
                # 此时 pred_g 是 [N, k]，labels.view(-1, 1) 是 [N, 1]，expand 成功
                acc_g = pred_g.eq(labels.view(-1, 1).expand_as(pred_g)).any(dim=1).float().mean().item()

                # Text -> Graph Accuracy
                _, pred_t = sim_t2g.topk(k, dim=1)
                acc_t = pred_t.eq(labels.view(-1, 1).expand_as(pred_t)).any(dim=1).float().mean().item()

                results[f'itc_acc{k}'] = (acc_g + acc_t) / 2

            return results

    # def debug_alignment(self, input_ids, labels, n_samples=1):
    #     """
    #     打印输入和标签的 Token 对齐情况
    #     """
    #     # 从 DataLoader 的 Dataset 中获取 tokenizer
    #     tokenizer = self.train_loader.dataset.tokenizer if hasattr(self.train_loader.dataset, 'tokenizer') else self.train_loader.dataset.dataset.tokenizer

    #     # 仅查看 batch 中的前 n 个样本
    #     for i in range(min(n_samples, input_ids.size(0))):
    #         # 1. 还原输入序列
    #         raw_input = input_ids[i].cpu().tolist()
    #         input_tokens = tokenizer.convert_ids_to_tokens(raw_input)

    #         # 2. 还原标签序列（处理 -100 屏蔽位）
    #         raw_labels = labels[i].cpu().tolist()
    #         label_tokens = [
    #             tokenizer.convert_ids_to_tokens([tid])[0] if tid != -100 else "[MASKED]"
    #             for tid in raw_labels
    #         ]

    #         self.logger.info(f"\n=== Alignment Check (Sample {i}) ===")
    #         # 打印前 20 个 token 观察对齐情况
    #         for idx in range(20):
    #             in_t = input_tokens[idx] if idx < len(input_tokens) else "OUT_OF_RANGE"
    #             la_t = label_tokens[idx] if idx < len(label_tokens) else "OUT_OF_RANGE"
    #             self.logger.info(f"Pos {idx:02d}: Input -> {in_t:<15} | Label -> {la_t}")

    def _forward_pass(self, samples, epoch):
        device = self.accelerator.device
        original_text_ids = samples['text_input_ids'].to(device)
        samples['text_input_ids'] = original_text_ids

        # 模型前向
        outputs = self.model(samples, task='train')

        loss_itc = outputs['loss_itc']
        loss_itm = outputs['loss_itm']
        loss_gtg = outputs['loss_gtg']
        loss_ortho = outputs['loss_ortho']  # 接住正交约束 Loss

        # 动态组装总损失（这里我们引入一个由 yaml 灵活配置的 lambda_ortho，默认给 0.5 ~ 1.0）
        lambda_ortho = self.config['trainer'].get('lambdas', {}).get('ortho', 1.0)

        total_loss = (
                self.lambdas['itc'] * loss_itc +
                self.lambdas['itm'] * loss_itm +
                self.lambdas['gtg'] * loss_gtg +
                lambda_ortho * loss_ortho
        )

        metrics = {
            'itc': loss_itc,
            'itm': loss_itm,
            'gtg': loss_gtg,
            'ortho': loss_ortho,  # 传出供 tqdm 栏目实时滚动
            'graph_feat': outputs['graph_feats'],
            'text_feat': outputs['text_feats'],
            'logits': outputs.get('logits')
        }

        return total_loss, metrics

    def train_epoch(self, epoch):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.model.train()
        total_stats = {k: 0.0 for k in ['loss', 'itc', 'gtg', 'itm', 'ortho']}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}",
                    disable=not self.accelerator.is_main_process,
                    mininterval=0.5)  # 减少更新频率

        for data in pbar:
            # 使用 accumulate 包装梯度累积
            with self.accelerator.accumulate(self.model):
                # 1. 关键修改：开启自动混合精度上下文
                with self.accelerator.autocast():
                    data = {k: v.to(self.accelerator.device) if torch.is_tensor(v) else v
                            for k, v in data.items()}

                    self.optimizer.zero_grad()
                    loss, metrics = self._forward_pass(data, epoch)

                # 2. 反向传播（accelerator 会在后台自动处理梯度缩放 Scaler）
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

                # ===== 在参数更新后，再更新 ITC queue =====

            total_stats['loss'] += loss.item()
            for k in ['itc', 'gtg', 'itm', 'ortho']:
                total_stats[k] += metrics[k].item()
            pbar.set_postfix(L=f"{loss.item():.3f}", G=f"{metrics['gtg']:.3f}", O=f"{metrics['ortho']:.3f}")

        n = len(self.train_loader)
        return total_stats['loss'] / n, {k: v / n for k, v in total_stats.items() if k != 'loss'}

    def eval_epoch(self, epoch):
        self.model.eval()
        total_stats = {k: 0.0 for k in ['itc', 'gtg', 'itm', 'ortho']}
        total_loss = 0.0

        # 频率控制开关
        do_snapshot = (epoch % 5 == 0)  # 每 10 轮画一次热力图

        all_graph_feats = []
        all_text_feats = []

        for i, data in enumerate(self.valid_loader):
            with torch.no_grad():
                with self.accelerator.autocast():
                    # 1. 正常前向计算 Loss
                    loss, metrics = self._forward_pass(data, epoch)

                # 2. 收集特征用于计算全局准确率
                # 仅收集对齐后的投影特征，内存占用小
                g_f = self.accelerator.gather(metrics['graph_feat'].detach())
                t_f = self.accelerator.gather(metrics['text_feat'].detach())
                all_graph_feats.append(g_f.cpu())
                all_text_feats.append(t_f.cpu())

                # 3. 仅在第一批数据且满足频率时执行快照
                if i == 0 and self.accelerator.is_main_process:
                    if do_snapshot:
                        self._create_monitor_snapshot(epoch, data, metrics)

                total_loss += loss.item()
                for k in total_stats.keys():
                    if k in metrics: total_stats[k] += metrics[k].item()

        # 4. 计算全局 Top-K
        n = len(self.valid_loader)
        avg_metrics = {k: v / n for k, v in total_stats.items()}

        # 将所有 batch 的特征合并
        full_g = torch.cat(all_graph_feats, dim=0)
        full_t = torch.cat(all_text_feats, dim=0)

        # 仅对主进程计算准确率，避免多卡重复计算
        if self.accelerator.is_main_process:
            # 限制计算矩阵大小（例如前 1000 个），防止 OOM
            eval_size = min(len(full_g), 1000)
            itc_acc = self.compute_itc_metrics(full_g[:eval_size], full_t[:eval_size])
            avg_metrics.update(itc_acc)

        return total_loss / n, avg_metrics

    def train(self, num_epochs=50, patience=5):
        early_stopping = EarlyStopping(patience=patience, mode='min',
                                       logger=self.logger if self.accelerator.is_main_process else None,
                                       save_dir=self.save_dir)

        for epoch in range(num_epochs):
            self._update_lambdas(epoch, num_epochs)

            train_loss, train_m = self.train_epoch(epoch)
            val_loss, val_m = self.eval_epoch(epoch)

            # 更新历史记录（去除了 motif 参数）
            self._update_history(train_loss, train_m, val_loss, val_m)

            if self.accelerator.is_main_process:
                if (epoch + 1) % 5 == 0:
                    self._save_all_plots()
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs}:")
                self.logger.info(
                    f" Train:Loss {train_loss:.4f}|ITC {train_m['itc']:.4f} | GTG {train_m['gtg']:.4f} | ITM {train_m['itm']:.4f} | Ortho {train_m['ortho']:.4f}")
                self.logger.info(
                    f" Val:Loss {val_loss:.4f}|ITC {val_m['itc']:.4f} | GTG {val_m['gtg']:.4f} | ITM {val_m['itm']:.4f}  | Ortho {val_m['ortho']:.4f}")

                early_stopping(val_m['itc'] + val_m['gtg'] + val_m['itm'] + val_m['ortho'],
                               self.accelerator.unwrap_model(self.model))
                if early_stopping.early_stop: break
            self.accelerator.wait_for_everyone()

    def _update_history(self, t_l, t_m, v_l, v_m):
        self.history['train_loss'].append(t_l)
        self.history['val_loss'].append(v_l)
        for k in ['itc', 'gtg', 'itm', 'ortho']:
            self.history[f'train_{k}'].append(t_m[k])
            self.history[f'val_{k}'].append(v_m[k])
        for k in ['itc_acc1', 'itc_acc5']:
            self.history[f'val_{k}'].append(v_m.get(k, 0.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/root/autodl-tmp/first_paper/java/stage2/stage2.yaml')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    accelerator = Accelerator(
        gradient_accumulation_steps=config['trainer'].get('accumulation_steps', 1),
        mixed_precision="bf16"
    )
    model = InnovativeStage2Bridge(config)

    shard_dir = config['data']['input_path']
    train_path = os.path.join(shard_dir, "train_data_stage2.pt")
    val_path = os.path.join(shard_dir, "val_data_stage2.pt")
    test_path = os.path.join(shard_dir, "test_data_stage2.pt")
    train_set = SinglePtDataset(
        train_path,
        config['model']['bert_model_dir'],
        max_txt_len=config['data'].get('max_length', 512)
    )
    val_set = SinglePtDataset(
        val_path,
        config['model']['bert_model_dir'],
        max_txt_len=config['data'].get('max_length', 512)
    )
    test_set = SinglePtDataset(
        test_path,
        config['model']['bert_model_dir'],
        max_txt_len=config['data'].get('max_length', 512)
    )

    # 定义测试集加载器（如果需要最后评估）
    test_loader = DataLoader(test_set, batch_size=config['data']['batch_size'], collate_fn=collate_fn)

    train_loader = DataLoader(
        train_set,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
        prefetch_factor=2
    )

    valid_loader = DataLoader(
        val_set,
        batch_size=config['data']['batch_size'],
        shuffle=False,  # 验证集通常不洗牌
        collate_fn=collate_fn,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    # 方案 D: 分层学习率实现
    # --- 优化后的参数分组逻辑 ---
    high_lr_keywords = [
        "query_tokens",  # 基础 Query 和 领航 Query
        "gen_query_tokens",  # 如果你定义了专门的生成 Query
        "proj",  # graph_proj, text_proj 等投影层
        "adapter",  # gtg_adapter 等适配层
        "crossattention",  # Q-former 中的交叉注意力层（对齐核心）
        "itm_head",  # ITM 分类头
        "log_temp"  # 温度系数
    ]

    # 2. 定义无需权重衰减的参数类型
    no_decay = ["bias", "LayerNorm.weight"]

    base_lr = float(config['trainer'].get('lr', 5e-6))
    new_layer_lr = base_lr * 10.0  # 针对新层使用 10 倍学习率

    params_high_lr_decay = []  # 高学习率 + 权重衰减 (如线性层权重)
    params_high_lr_no_decay = []  # 高学习率 + 无权重衰减 (如偏置、LN)
    params_base_lr_decay = []  # 基础学习率 + 权重衰减 (预训练层)
    params_base_lr_no_decay = []  # 基础学习率 + 无权重衰减

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # 判断是否属于高学习率模块
        is_high_lr = any(key in n for key in high_lr_keywords)
        # 判断是否属于排除衰减的类型
        is_no_decay = any(nd in n for nd in no_decay)

        if is_high_lr:
            if is_no_decay:
                params_high_lr_no_decay.append(p)
            else:
                params_high_lr_decay.append(p)
        else:
            if is_no_decay:
                params_base_lr_no_decay.append(p)
            else:
                params_base_lr_decay.append(p)

    # 3. 构造优化器分组
    optimizer_grouped_parameters = [
        {
            "params": params_high_lr_decay,
            "lr": new_layer_lr,
            "weight_decay": float(config['trainer'].get('weight_decay', 0.05))
        },
        {
            "params": params_high_lr_no_decay,
            "lr": new_layer_lr,
            "weight_decay": 0.0
        },
        {
            "params": params_base_lr_decay,
            "lr": base_lr,
            "weight_decay": float(config['trainer'].get('weight_decay', 0.05))
        },
        {
            "params": params_base_lr_no_decay,
            "lr": base_lr,
            "weight_decay": 0.0
        }
    ]

    # 4. 防御性检查
    unique_params = set(id(p) for g in optimizer_grouped_parameters for p in g["params"])
    total_trainable = len([p for p in model.parameters() if p.requires_grad])
    if len(unique_params) != total_trainable:
        print(f"警告：参数分组覆盖不完整！唯一参数数 {len(unique_params)} vs 总可训练数 {total_trainable}")
    # --- 逻辑修改结束 ---

    # 4. 验证是否有参数重复（防御性检查）
    total_params = sum(len(g["params"]) for g in optimizer_grouped_parameters)
    unique_params = len(set(id(p) for g in optimizer_grouped_parameters for p in g["params"]))
    if total_params != unique_params:
        raise ValueError(f"检测到参数冲突！总数 {total_params} vs 唯一数 {unique_params}")

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    num_train_steps = len(train_loader) * config['trainer'].get('num_epochs', 100)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['trainer'].get('warmup_steps', 500),
        num_training_steps=num_train_steps
    )

    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(model, optimizer, train_loader,
                                                                                  valid_loader, scheduler)

    trainer = InnovativeStage2Trainer(model, optimizer, scheduler, train_loader, valid_loader, accelerator, config)
    trainer.train(num_epochs=config['trainer'].get('num_epochs', 50), patience=config['trainer'].get('patience', 5))

    # 2. 评估测试集
    print("\n" + "=" * 20 + " Final Test Evaluation " + "=" * 20)
    # 加载训练过程中保存的最佳模型权重（可选）
    best_model_path = os.path.join(config['output']['save_dir'], "best_innovative_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")

    # 使用 eval_epoch 逻辑跑一遍测试集
    test_loss, test_m = trainer.eval_epoch(epoch=999)  # 传入一个特殊标记的 epoch 序号

    print(
        f"Test Result: Loss {test_loss:.4f} | ITC {test_m['itc']:.4f} | GTG {test_m['gtg']:.4f} | ITM {test_m['itm']:.4f}")


if __name__ == "__main__":
    main()