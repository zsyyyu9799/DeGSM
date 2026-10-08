import os
import sys
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel

# 将父目录加入到系统路径，防止跨文件夹导包失败
sys.path.append("/root/autodl-tmp/first_paper/python")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入 Stage 3 顶层架构
from stage3.CGBridge_Stage3 import CGBridgeStage3


class Stage3FineTuneDataset(Dataset):
    """
    升级版数据集：支持将一条 Python 源代码与多条通过编译/测试的目标语言候选代码分别配对，
    实现多候选平铺（Data Augmentation），最大化利用小样本数据。
    """
    def __init__(self, jsonl_path, original_stage3_jsonl):
        self.samples = []
        
        # 1. 建立快速索引，用于从原始数据集中提取图特征 (graph_emb)
        print("🔄 正在预加载原始图特征索引...")
        graph_emb_lookup = {}
        with open(original_stage3_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if 'idx' in item and 'graph_emb' in item:
                    graph_emb_lookup[int(item['idx'])] = item['graph_emb']

        # 2. 读取通过了自动化评测清洗后的黄金对并平铺
        print("扫 正在解析洗白数据集并激活多候选数据扩增...")
        total_unique_python = 0
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                idx = int(item['idx'])
                
                if idx not in graph_emb_lookup:
                    continue  
                    
                raw_emb = graph_emb_lookup[idx]
                if isinstance(raw_emb, str):
                    graph_emb = torch.tensor(json.loads(raw_emb), dtype=torch.float)
                else:
                    graph_emb = torch.tensor(raw_emb, dtype=torch.float)
                
                # 兼容不同语言清洗出来的字段名，优先读取 passed_r_candidates
                passed_candidates = item.get('passed_r_candidates', item.get('passed_candidates', []))
                if not passed_candidates:
                    continue
                
                total_unique_python += 1
                
                for cand_idx, tgt_code in enumerate(passed_candidates):
                    self.samples.append({
                        'idx': idx,
                        'sub_idx': cand_idx, 
                        'graph_emb': graph_emb,
                        'src_code': str(item.get('python_code', "")),
                        'nl_desc': str(item.get('python_test_cases', [{}])[0].get('nl_description', "Translate python to target language")), 
                        'tgt_code': str(tgt_code)
                    })
                    
        print(f"==================================================")
        print(f"📊 模态对齐数据扩增报告:")
        print(f"  - 原始有效 Python 核心题目数: {total_unique_python} 条")
        print(f"  - 平铺扩增后总微调样本数: {len(self.samples)} 条 🚀")
        print(f"==================================================")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    idxs = [item['idx'] for item in batch]
    graph_embs = torch.stack([item['graph_emb'] for item in batch])
    src_codes = [item['src_code'] for item in batch]
    nl_descs = [item['nl_desc'] for item in batch]
    tgt_codes = [item['tgt_code'] for item in batch]
    return idxs, graph_embs, src_codes, nl_descs, tgt_codes


def run_finetune(config_path, train_jsonl, original_jsonl):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_type = config["model"].get("llm_type", "qwen").lower()
    
    print(f"🚀 正在初始化基座类型为 [{llm_type.upper()}] 的 CGBridgeStage3 架构...")
    model = CGBridgeStage3(config)
    
    # ========================================================================
    # 🎯 【核心前置防御】多模型自适应词表空间二次显式校验与重缩放
    # ========================================================================
    current_vocab_size = len(model.llm_tokenizer)
    if hasattr(model, "llm_model") and model.llm_model is not None:
        try:
            if hasattr(model.llm_model, "get_input_embeddings"):
                input_embeddings = model.llm_model.get_input_embeddings()
            elif hasattr(model.llm_model, "base_model") and hasattr(model.llm_model.base_model, "get_input_embeddings"):
                input_embeddings = model.llm_model.base_model.get_input_embeddings()
            else:
                input_embeddings = getattr(model.llm_model, "embed_tokens", getattr(model.llm_model, "wte", None))
            
            if input_embeddings is not None and hasattr(input_embeddings, "weight"):
                model_vocab_size = input_embeddings.weight.size(0)
                if model_vocab_size != current_vocab_size:
                    print(f"⚠️ [微调前置防御] 检测到模型词表层大小 ({model_vocab_size}) 与分词器 ({current_vocab_size}) 不一致，正在强制重缩放...")
                    if hasattr(model.llm_model, "resize_token_embeddings"):
                        model.llm_model.resize_token_embeddings(current_vocab_size)
            else:
                print("⚠️ [微调前置防御] 无法自动获取模型 Embedding 层权重，跳过词表大小校验。")
        except Exception as e:
            print(f"⚠️ [微调前置防御] 探测模型词表大小时发生异常 ({e})，安全跳过此防御步，继续执行。")
            
    # ========================================================================
    # 🛠️ 【LoRA 注入】为大模型注入 LoRA，使其参数协同参与翻译微调
    # ========================================================================
    if hasattr(model, "llm_model") and not isinstance(model.llm_model, PeftModel):
        print("🎨 正在动态挂载大模型 LoRA 适配器结构...")
        
        if "deepseek" in llm_type:
            print("💡 路由选择：检测到 DeepSeek 系列经典架构，挂载全注意力与全多层感知机旁路矩阵...")
            target_mods = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "starcoder" in llm_type:
            print("💡 路由选择：检测到 StarCoder 家族，同时挂载 1代与2代 的特异性靶向层...")
            target_mods = ["c_attn", "c_proj", "c_fc", "q_proj", "kv_proj", "o_proj"]
        elif "llamabase" in llm_type or "llama" in llm_type:
            print("💡 路由选择：检测到标准 Llama 架构，挂载标准注意力靶向层...")
            target_mods = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            target_mods = ["q_proj", "v_proj", "o_proj"]
            
        if "starcoder" in llm_type:
            peft_config = LoraConfig(
                r=16,                           
                lora_alpha=32,                  
                target_modules=target_mods,  
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                modules_to_save=["lm_head"]     # 解除对输出头的锁定，加速特定语言分布的对齐
            )
        else:
            peft_config = LoraConfig(
                r=8,                           
                lora_alpha=16,                 
                target_modules=target_mods,  
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
        model.llm_model = get_peft_model(model.llm_model, peft_config)
        model.llm_model.print_trainable_parameters()
    
    # ========================================================================
    # 🛠️ 【梯度控锁】精准控梯度状态（闭锁二阶段 Q-Former，放行三阶段翻译层）
    # ========================================================================
    print("\n🔒 开始配置 Stage 3 参数防御墙 (锁定 Q-Former, 激活 llm_proj + LoRA)...")
    for param in model.parameters():
        param.requires_grad = False
        
    if hasattr(model, "stage2_bridge"):
        model.stage2_bridge.eval() 
        for param in model.stage2_bridge.parameters():
            param.requires_grad = False
            
    if hasattr(model, "llm_proj"):
        model.llm_proj.train() 
        for param in model.llm_proj.parameters():
            param.requires_grad = True
            
    if hasattr(model, "group_weights") and model.group_weights is not None:
        model.group_weights.requires_grad = True

    if hasattr(model, "llm_model"):
        model.llm_model.train() 
        for name, param in model.llm_model.named_parameters():
            if "lora_" in name or "modules_to_save" in name:
                param.requires_grad = True
        
        # 激活大模型梯度检查点，阻断显存溢出 (OOM)
        print(f"DEBUG: 激活大模型 ({llm_type.upper()}) 梯度检查点 Gradient Checkpointing...")
        model.llm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 二次严格检验真正的优化器入参
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\n🔥 [Stage 3 真正参与微调的可训练参数块数]: {len(trainable_params)}")
    if len(trainable_params) == 0:
        raise ValueError("❌ 错误：检测到可训练参数量为 0！请检查组件命名与 requires_grad 绑定逻辑。")
            
    if hasattr(model, "stage2_bridge") and model.stage2_bridge is not None:
        model.stage2_bridge.to(device)
    if hasattr(model, "llm_proj"):
        model.llm_proj.to(device)
    if hasattr(model, "group_weights") and model.group_weights is not None:
        model.group_weights = nn.Parameter(model.group_weights.to(device))
    
    # 3. 划分训练集与验证集
    full_dataset = Stage3FineTuneDataset(train_jsonl, original_jsonl)
    val_size = int(0.3 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    print(f"📊 验证集划分完成: 训练集数 = {len(train_dataset)}, 验证集数 = {len(val_dataset)}")
    
    current_batch_size = config.get('finetune', {}).get('batch_size', 4)
    num_workers = config.get('finetune', {}).get('num_workers', 2)
    
    train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    # 4. 优化器、调度器与早停设置
    lr = float(config.get('finetune', {}).get('lr', 2e-5))
    weight_decay = float(config.get('finetune', {}).get('weight_decay', 0.05))
    epochs = config.get('finetune', {}).get('num_epochs', 5)
    
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    
    os.makedirs(os.path.dirname(config['finetune']['finetune_model']), exist_ok=True)
    best_val_loss = float('inf')
    early_stop_patience = 10
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # ----------------------------------------------------
        # 【A 阶段：训练循环】
        # ----------------------------------------------------
        model.train()
        if hasattr(model, "stage2_bridge"):
            model.stage2_bridge.eval()
            
        epoch_train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            idxs, graph_embs, src_codes, nl_descs, tgt_codes = batch
            graph_embs = graph_embs.to(device)
            
            # 1. 提取图模态对齐特征 (保持 eval 状态，不计算梯度)
            with torch.no_grad():
                s2_samples = {
                    'node_embs': graph_embs,
                    'text_input_ids': torch.zeros((graph_embs.size(0), 1), dtype=torch.long).to(device),
                    'text_atts': torch.zeros((graph_embs.size(0), 1), dtype=torch.long).to(device),
                    'meta': [{'node_type_map': []} for _ in range(graph_embs.size(0))]
                }
                s2_outputs = model.stage2_bridge(s2_samples, task='test')
                query_feats = s2_outputs['query_output'].to(dtype=model.llm_model.dtype)
            
            # 2. 构造与推理完全同构的纯文本前置 Prompt (use_context=True 内部会插入占位符)
            # 注意：此处 prepare_lm_input 内部只需要返回前置 Prompt 对应的 inputs_embeds 结构
            inputs_embeds, attention_mask = model.prepare_lm_input(
                gtokens=query_feats, text_input=src_codes, nl_desc=nl_descs, use_context=True
            )
            
            # 3. 动态识别当前基座的终止符 (EOS)
            if "starcoder" in llm_type:
                eot_token = "<|endoftext|>" if "<|endoftext|>" in model.llm_tokenizer.get_vocab() else model.llm_tokenizer.eos_token
            elif "llama" in llm_type:
                eot_token = "<|eot_id|>" if "<|eot_id|>" in model.llm_tokenizer.get_vocab() else model.llm_tokenizer.eos_token
            elif "qwen" in llm_type or "deepseek" in llm_type:
                eot_token = "<|im_end|>" if "<|im_end|>" in model.llm_tokenizer.get_vocab() else model.llm_tokenizer.eos_token
            else:
                eot_token = model.llm_tokenizer.eos_token
                
            # 4. 核心重构：将 R 答案与 </s> 终止符编码为右填充张量
            model.llm_tokenizer.padding_side = "right" 
            targets_text = [f"{code}{eot_token}" for code in tgt_codes]
            target_batch = model.llm_tokenizer(
                targets_text, 
                padding=True, 
                return_tensors="pt", 
                truncation=True, 
                max_length=768,
                add_special_tokens=False # 杜绝系统自动在答案头部加 <s> 导致二次错位
            ).to(device)
            
            # 5. 精确提取 Target 的词嵌入向量 (保持 requires_grad，使其随 LoRA 协同微调)
            raw_llm = model.llm_model.base_model.model if hasattr(model.llm_model, "base_model") else model.llm_model
            target_embeds = raw_llm.get_input_embeddings()(target_batch.input_ids)
            
            # 6. 张量水平拼接：[Prompt 嵌入 (含图特征)] + [Target 答案嵌入]
            full_inputs_embeds = torch.cat([inputs_embeds, target_embeds], dim=1)
            full_attention_mask = torch.cat([attention_mask, target_batch.attention_mask], dim=1)
            
            # 7. 精确构建 Label 掩码墙：前置 Prompt 部分全部赋为 -100 (不参与 Loss 计算)
            prompt_labels = torch.full((inputs_embeds.size(0), inputs_embeds.size(1)), -100, dtype=torch.long, device=device)
            
            target_ids = target_batch.input_ids.clone()
            target_ids[target_ids == model.llm_tokenizer.pad_token_id] = -100 # 忽略 Padding 位的 Loss
            
            full_labels = torch.cat([prompt_labels, target_ids], dim=1)
            
            # 8. 前向传播：传入全量拼接 Embeds，将错位控锁交给 Transformers 内部的自回归移位机制
            outputs = model.llm_model(
                inputs_embeds=full_inputs_embeds, 
                attention_mask=full_attention_mask, 
                return_dict=True
            )
            
            # 9. 严格执行标准 Causal LM 的逻辑对齐切片：
            # Logits 往前挪一位，Labels 往后挪一位，完美对齐因果预测时序
            shift_logits = outputs.logits[:, inputs_embeds.size(1) - 1 : -1, :].contiguous()
            shift_labels = full_labels[:, inputs_embeds.size(1) :].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # 反向传播与阶梯裁剪
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_train_loss += loss.item()
            progress_bar.set_postfix({"Train_Loss": f"{loss.item():.4f}"})
            del full_inputs_embeds, full_attention_mask, full_labels, outputs, loss
            
        avg_train_loss = epoch_train_loss / len(train_loader)
        torch.cuda.empty_cache()

        # ----------------------------------------------------
        # 【B 阶段：验证评估循环】（同步镜像重构）
        # ----------------------------------------------------
        model.eval() 
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Validation]")
            for batch in val_progress:
                idxs, graph_embs, src_codes, nl_descs, tgt_codes = batch
                graph_embs = graph_embs.to(device)
                
                s2_samples = {
                    'node_embs': graph_embs,
                    'text_input_ids': torch.zeros((graph_embs.size(0), 1), dtype=torch.long).to(device),
                    'text_atts': torch.zeros((graph_embs.size(0), 1), dtype=torch.long).to(device),
                    'meta': [{'node_type_map': []} for _ in range(graph_embs.size(0))]
                }
                s2_outputs = model.stage2_bridge(s2_samples, task='test')
                query_feats = s2_outputs['query_output'].to(dtype=model.llm_model.dtype)
                
                inputs_embeds, attention_mask = model.prepare_lm_input(
                    gtokens=query_feats, text_input=src_codes, nl_desc=nl_descs, use_context=True
                )
                
                model.llm_tokenizer.padding_side = "right"
                targets_text = [f"{c}{eot_token}" for c in tgt_codes]
                target_batch = model.llm_tokenizer(
                    targets_text, 
                    padding=True, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=768,
                    add_special_tokens=False
                ).to(device)
                
                raw_llm = model.llm_model.base_model.model if hasattr(model.llm_model, "base_model") else model.llm_model
                target_embeds = raw_llm.get_input_embeddings()(target_batch.input_ids)
                
                full_inputs_embeds = torch.cat([inputs_embeds, target_embeds], dim=1)
                full_attention_mask = torch.cat([attention_mask, target_batch.attention_mask], dim=1)
                
                prompt_labels = torch.full((inputs_embeds.size(0), inputs_embeds.size(1)), -100, dtype=torch.long, device=device)
                target_ids = target_batch.input_ids.clone()
                target_ids[target_ids == model.llm_tokenizer.pad_token_id] = -100
                full_labels = torch.cat([prompt_labels, target_ids], dim=1)
                
                outputs = model.llm_model(inputs_embeds=full_inputs_embeds, attention_mask=full_attention_mask, return_dict=True)
                
                shift_logits = outputs.logits[:, inputs_embeds.size(1) - 1 : -1, :].contiguous()
                shift_labels = full_labels[:, inputs_embeds.size(1) :].contiguous()
                
                loss_fct = nn.CrossEntropyLoss()
                val_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                epoch_val_loss += val_loss.item()
                
        avg_val_loss = epoch_val_loss / len(val_loader)
        print(f"📊 Epoch {epoch} 总结: Train Loss = {avg_train_loss:.4f} || Validation Loss = {avg_val_loss:.4f}")
        
        # ----------------------------------------------------
        # 【C 阶段：基于核心验证损失的早停判定与权重精细保存】
        # ----------------------------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  
            save_path = config['finetune']['finetune_model']
            print(f"🏆 验证集 Loss 刷新历史最低! 正在安全保存增量融合权重至: {save_path}")
            
            # 只提取真正打上解冻烙印的可训练参数保存
            trainable_param_names = {name for name, param in model.named_parameters() if param.requires_grad}
            saved_state_dict = {k: v for k, v in model.state_dict().items() if k in trainable_param_names}
            torch.save(saved_state_dict, save_path)
        else:
            patience_counter += 1
            print(f"⚠️ 验证集 Loss 未能下降。早停触发器计数: {patience_counter} / {early_stop_patience}")
            if patience_counter >= early_stop_patience:
                print(f"🛑 [早停机制触发] 模型已连续 {early_stop_patience} 个 Epoch 未能好转，强制终止训练。")
                break
                
    print("🎉 【微调流程完美结束】多模型通用对齐翻译模型已成功保存！")


if __name__ == "__main__":
    CONFIG_PATH = "/root/autodl-tmp/first_paper/python/stage3/stage3.yaml"
    PASSED_CLEAN_JSONL = "/root/autodl-tmp/first_paper/python/testcase_dataset/R_pass/R_passed_clean.jsonl"
    ORIGINAL_STAGE3_JSONL = "/root/autodl-tmp/first_paper/python/translation_dataset/nl_code_stage3_compiled_clean.jsonl"
    
    run_finetune(CONFIG_PATH, PASSED_CLEAN_JSONL, ORIGINAL_STAGE3_JSONL)