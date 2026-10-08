import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch.optim import AdamW
from torch.utils.data import DataLoader
from CGBridge_Stage3 import CGBridgeStage3
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader, random_split, Dataset
from transformers import get_cosine_schedule_with_warmup

class HumanEvalOcamlDataset(Dataset):
    def __init__(self, jsonl_path):
        self.df = pd.read_json(jsonl_path, lines=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        raw_emb = item['graph_emb']
        graph_emb = torch.tensor(raw_emb if not isinstance(raw_emb, str) else json.loads(raw_emb))
        
        return {
            'idx': idx,
            'graph_emb': graph_emb,
            'java_code': item['java_source'],  
            'target_text': item['ocaml_target']
        }

def train_human_eval(config_path, data_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    save_dir = config['output']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据准备与分割
    full_dataset = HumanEvalOcamlDataset(data_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=config['traindata']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['traindata']['batch_size'], shuffle=False)

    # 1. 初始化模型
    model = CGBridgeStage3(config).to(device)

    # ================== 🛡️ 多层线性投影层安全隔离与初始化 ==================
    if isinstance(model.llm_proj, nn.Sequential):
        print("🔄 [架构升级] 正在激活多层感知机(MLP)投影映射管线...")
        torch_dtype = torch.bfloat16 if config["model"].get("llm_dtype") == "bfloat16" else torch.float16
        model.llm_proj.to(device=device, dtype=torch_dtype)
        print("✅ 升级成功：当前 llm_proj 拥有深层非线性拟合能力。")

    # ================== 🔒 [方案 B] 全盘锁死大模型主干与词表 ==================
    print("🔒 正在执行 [方案 B]：全面锁死大模型主干、Embedding 与 LM Head...")
    for param in model.parameters():
        param.requires_grad = False

    # 200条极小数据下彻底移除 LoRA。大模型完全作为纯粹的固定特征提取器与条件生成器。
    if hasattr(model.llm_model, "gradient_checkpointing_enable"):
        model.llm_model.gradient_checkpointing_enable()
        print("🔒 大模型核心已开启 Gradient Checkpointing 模式")

    # ================== 🔓 [方案 B] 定点释放桥接层与投影层 ==================
    # ================== 🔓 [方案 B - 深度瘦身版] 定点精确解冻 ==================
    bridge_params = []
    
    # 1. 100% 释放：多层线性投影层 (MLP) —— 这必须全力学习
    for param in model.llm_proj.parameters():
        param.requires_grad = True
        bridge_params.append(param)

    # 2. 🎯 定点精确解冻：只解冻 Q-Former 内部的 Cross-Attention (交叉注意力)
    # 这样既保留了 Stage 2 已经学到的图拓扑提取能力，又强迫模型用极小的参数量去对齐 Stage 3
    qformer_unlocked_count = 0
    qformer_total_count = 0
    
    for name, param in model.stage2_bridge.named_parameters():
        qformer_total_count += 1
        # 核心拦截：只放开交叉注意力层 (crossattention) 的权重与偏置
        if "crossattention" in name.lower() or "cross_attn" in name.lower():
            param.requires_grad = True
            bridge_params.append(param)
            qformer_unlocked_count += 1
            
    print(f"🔄 [Q-Former 瘦身] 识别到 Q-Former 共 {qformer_total_count} 个张量，"
          f"已定点唤醒 {qformer_unlocked_count} 个 Cross-Attention 关键张量，其余全部锁死！")

    # 3. 100% 释放：动态四组权重参数
    if hasattr(model, "group_weights") and model.group_weights is not None:
        model.group_weights.requires_grad = True
        bridge_params.append(model.group_weights)
            
    print(f"🔓 [方案 B 激活] 大模型与 Q-Former 基座已完全冻结，进入全轻量化对齐模式。")
    # ========================================================================
    # ========================================================================

    # 💡 针对小数据集微调的优化器设置：降低权重衰减以防阻碍欠拟合的桥接层收敛
    base_lr = float(config['trainer']['lr'])
    wd = config['trainer'].get('weight_decay', 0.01)
    optimizer = AdamW([
        {'params': bridge_params, 'lr': base_lr * 2, 'weight_decay': wd}
    ])

    num_epochs = config['trainer']['num_epochs']
    accumulation_steps = config['traindata'].get('accumulation_steps', 1)
    total_training_steps = (len(train_loader) // accumulation_steps) * num_epochs
    num_warmup_steps = int(total_training_steps * 0.1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps
    )

    # 🎯 强力抗过拟合：引入 0.15 的高标签平滑度，防止网络死记硬背
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.15)
    
    best_val_loss = float('inf')
    patience = 10  
    patience_counter = 0

    print(f"🚀 开始训练。可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            target_dtype = model.llm_model.dtype
            query_feats = batch['graph_emb'].to(device=device, dtype=target_dtype)
    
            mock_nl_desc = [""] * len(batch['idx'])
            inputs_embeds, attention_mask = model.prepare_lm_input(
                gtokens=query_feats, 
                text_input=batch['java_code'], 
                nl_desc=mock_nl_desc, 
                use_context=True
            )
            
            model.llm_tokenizer.padding_side = "right"
            targets_with_eos = [t + model.llm_tokenizer.eos_token for t in batch['target_text']]
            target_batch = model.llm_tokenizer(
                targets_with_eos, 
                padding=True, 
                return_tensors="pt"
            ).to(device)
            
            target_embeds = model.llm_model.get_input_embeddings()(target_batch.input_ids)
            full_inputs_embeds = torch.cat([inputs_embeds, target_embeds], dim=1)
            full_attention_mask = torch.cat([attention_mask, target_batch.attention_mask], dim=1)
            
            model_outputs = model.llm_model(
                inputs_embeds=full_inputs_embeds,
                attention_mask=full_attention_mask,
                return_dict=True
            )
            
            prompt_len = inputs_embeds.size(1)
            target_logits = model_outputs['logits'][:, prompt_len:, :]
            
            shift_logits = target_logits[..., :-1, :].contiguous()
            shift_labels = target_batch.input_ids[..., 1:].contiguous()
            
            target_mask = target_batch.attention_mask[..., 1:].contiguous()
            shift_labels[target_mask == 0] = -100
            
            generation_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # 🎯 方案 B 核心纯化：彻底移除了极易导致小样本偏离的对比学习损失，全部梯度集中于自回归生成
            loss = generation_loss / accumulation_steps  
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            pbar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Eval]"):
                query_feats = batch['graph_emb'].to(device=device, dtype=model.llm_model.dtype)
                
                mock_nl_desc = [""] * len(batch['idx'])
                inputs_embeds, attention_mask = model.prepare_lm_input(
                    gtokens=query_feats, 
                    text_input=batch['java_code'], 
                    nl_desc=mock_nl_desc, 
                    use_context=True
                )
                
                model.llm_tokenizer.padding_side = "right"
                targets_with_eos = [t + model.llm_tokenizer.eos_token for t in batch['target_text']]
                target_batch = model.llm_tokenizer(targets_with_eos, padding=True, return_tensors="pt").to(device)
                
                target_embeds = model.llm_model.get_input_embeddings()(target_batch.input_ids)
                full_inputs_embeds = torch.cat([inputs_embeds, target_embeds], dim=1)
                full_attention_mask = torch.cat([attention_mask, target_batch.attention_mask], dim=1)
                
                target_outputs = model.llm_model(
                    inputs_embeds=full_inputs_embeds,
                    attention_mask=full_attention_mask,
                    return_dict=True
                )
                
                prompt_len = inputs_embeds.size(1)
                target_logits = target_outputs['logits'][:, prompt_len:, :]
                
                shift_logits = target_logits[..., :-1, :].contiguous()
                shift_labels = target_batch.input_ids[..., 1:].contiguous()
                target_mask = target_batch.attention_mask[..., 1:].contiguous()
                shift_labels[target_mask == 0] = -100
                
                v_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                val_loss += v_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"📊 Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_stage3_model.pt"))
            print(f"🌟 已保存当前最佳模型权重")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⚠️ 验证集表现未提升，连续 {patience_counter}/{patience} 轮未优化")
            if patience_counter >= patience:
                print("🛑 [Early Stopping] 检测到平台期震荡/轻微反弹，触发安全早停！")
                break

def main():
    CONFIG_PATH = "/root/autodl-tmp/first_paper/java/stage3/stage3.yaml"
    DATA_PATH = "/root/autodl-tmp/first_paper/human_stage3_llm/java/ocaml/java_to_ocaml_tasks.jsonl"
    
    if os.path.exists(CONFIG_PATH) and os.path.exists(DATA_PATH):
        train_human_eval(CONFIG_PATH, DATA_PATH)
    else:
        print("❌ 请检查配置文件或数据路径是否存在")

if __name__ == "__main__":
    main()