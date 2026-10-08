import os
import sys
# 将 stage3-r 的父目录加入到系统路径
sys.path.append("/root/autodl-tmp/first_paper/python")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
# 确保能找到 Stage 3 和 Stage 2 的文件
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入你前面定义的 Stage 3 顶层架构
from stage3.CGBridge_Stage3 import CGBridgeStage3

class Stage3FineTuneDataset(Dataset):
    """
    升级版数据集：支持将一条 Python 源代码与多条通过编译/测试的 Rust 语言候选代码分别配对，
    实现多候选平铺（Data Augmentation），最大化利用 200 条小样本数据。
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

        # 2. 读取通过了自动化评测清洗后的黄金对
        print("🧹 正在解析 Rust_passed_clean 并激活多候选数据扩增...")
        total_unique_python = 0
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                idx = int(item['idx'])
                
                if idx not in graph_emb_lookup:
                    continue  # 容错：如果找不到对应的原始图特征则跳过
                    
                # 提取图特征
                raw_emb = graph_emb_lookup[idx]
                if isinstance(raw_emb, str):
                    graph_emb = torch.tensor(json.loads(raw_emb), dtype=torch.float)
                else:
                    graph_emb = torch.tensor(raw_emb, dtype=torch.float)
                
                # 获取所有成功跑通单元测试的 R 语言候选代码列表
                passed_candidates = item.get('passed_rust_candidates', [])
                if not passed_candidates:
                    continue
                
                total_unique_python += 1
                
                # 🎯 核心改变：遍历所有通过的 R 代码，与当前的 Python 图特征分别绑定
                for cand_idx, tgt_rust_code in enumerate(passed_candidates):
                    self.samples.append({
                        'idx': idx,
                        'sub_idx': cand_idx, # 记录是第几个候选，方便调试
                        'graph_emb': graph_emb,
                        'src_code': str(item.get('python_code', "")),
                        # 提取第一条测试用例的语义描述作为 Prompt 输入
                        'nl_desc': str(item.get('python_test_cases', [{}])[0].get('nl_description', "Translate python to R")), 
                        'tgt_code': str(tgt_rust_code)
                    })
                    
        print(f"==================================================")
        print(f"📊 模态对齐数据扩增报告:")
        print(f"  - 原始有效 Python 题目数: {total_unique_python} 条")
        print(f"  - 平铺扩增后总微调样本数: {len(self.samples)} 条 🚀")
        print(f"==================================================")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    # 自定义 Batch 组装
    idxs = [item['idx'] for item in batch]
    graph_embs = torch.stack([item['graph_emb'] for item in batch])
    src_codes = [item['src_code'] for item in batch]
    nl_descs = [item['nl_desc'] for item in batch]
    tgt_codes = [item['tgt_code'] for item in batch]
    return idxs, graph_embs, src_codes, nl_descs, tgt_codes

def run_finetune(config_path, train_jsonl, original_jsonl):
    # 1. 加载基本配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 实例化基本模型
    print("🚀 正在初始化 CGBridgeStage3 架构...")
    model = CGBridgeStage3(config)
    
    # ========================================================================
    # 🛠️ 核心改造一：为大模型注入 LoRA，使其参数参与 Stage 3 翻译微调
    # ========================================================================
    if hasattr(model, "llm_model"):
        # 如果模型已经是 PeftModel，则无需重复注入
        if not isinstance(model.llm_model, PeftModel):
            print("🎨 正在为大模型基座注入 LoRA 适配器（使其参数参与微调）...")
            # 🎯 完美适配各大家族独有的注意力靶向层
            llm_type = config["model"].get("llm_type", "qwen").lower()
            if "deepseek" in llm_type:
                print("💡 路由选择：检测到 DeepSeek 系列，挂载 MLA 专属低秩注意力旁路矩阵...")
                target_mods = ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
            elif "starcoder" in llm_type:
                print("💡 路由选择：检测到 StarCoder 家族，挂载其特异性注意力投影层...")
                target_mods = ["c_attn", "c_proj"]  # StarCoder 经典的合并注意力机制
            elif "qwen" in llm_type or "llama" in llm_type:
                print("💡 路由选择：检测到标准 MHA 架构 (Qwen/Llama)，挂载常规注意力靶向层...")
                target_mods = ["q_proj", "v_proj"]
            else:
                # 兜底通用标准层
                target_mods = ["q_proj", "v_proj", "o_proj"]

            peft_config = LoraConfig(
                r=8,                           
                lora_alpha=16,                 
                target_modules=target_mods,  # 👈 动态替换为适配的靶向层
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            # 用 LoRA 结构包裹原有大模型
            model.llm_model = get_peft_model(model.llm_model, peft_config)
            model.llm_model.print_trainable_parameters()
    
    # ========================================================================
    # 🛠️ 核心改造二：精准控梯度状态（闭锁二阶段 Q-Former，放行三阶段翻译层）
    # ========================================================================
    print("\n🔒 开始配置 Stage 3 参数防御墙 (锁定 Q-Former, 激活 llm_proj + LoRA)...")
    
    # 【第一步】默认将整个大模型架构的所有组件完全锁定
    for param in model.parameters():
        param.requires_grad = False
        
    # 【第二步】强制锁定二阶段预训练好的 Q-Former (使其只充当稳定的图特征提取器)
    if hasattr(model, "stage2_bridge"):
        model.stage2_bridge.eval() # 显式切到 eval 模式
        for param in model.stage2_bridge.parameters():
            param.requires_grad = False
            
    # 【第三步】解冻跨模态翻译的核心桥梁层 (llm_proj)
    if hasattr(model, "llm_proj"):
        model.llm_proj.train() 
        for param in model.llm_proj.parameters():
            param.requires_grad = True
            
    # 【第四步】解冻多模态专家门控权重（如果有）
    if hasattr(model, "group_weights") and model.group_weights is not None:
        model.group_weights.requires_grad = True

    # 【第五步】解冻大模型中的 LoRA 参数，并激活梯度检查点防 OOM
    if hasattr(model, "llm_model"):
        model.llm_model.train() 
        # 显式放行带有 "lora_" 关键字的训练层
        for name, param in model.llm_model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        
        # 🚀 激活大模型梯度检查点，这是 31G 显存能够同时训练大模型参数的绝对关键！
        print(f"DEBUG: 激活大模型 ({llm_type.upper()}) 梯度检查点 Gradient Checkpointing...")
        if "starcoder" in llm_type:
            model.llm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.llm_model.gradient_checkpointing_enable()

    # 严格二次校验：收集真正标记为 requires_grad = True 的层送入优化器
    trainable_params = []
    print("\n==================================================")
    print("🔥 [Stage 3 真正参与微调的可训练参数报告]:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            print(f"  - {name} -> 形状: {param.shape}")
    print(f"总计可训练参数块数: {len(trainable_params)}")
    print("==================================================\n")
    
    if len(trainable_params) == 0:
        raise ValueError("❌ 错误：检测到可训练参数量为 0！请检查组件命名。")
            
    # 🎯 核心修复：二阶段桥接模块（包含 Q-Former 和 Bottleneck）必须完整送上 GPU 0，与数据对齐
    if hasattr(model, "stage2_bridge") and model.stage2_bridge is not None:
        print("🚚 正在将二阶段桥接模块整体搬移至目标显卡...")
        model.stage2_bridge.to(device)

    # 将三阶段增量可训练的组件靶向送上显卡
    if hasattr(model, "llm_proj"):
        model.llm_proj.to(device)
    if hasattr(model, "group_weights") and model.group_weights is not None:
        # 直接使用组件的 .to(device) 转换，不要点 .data
        model.group_weights = nn.Parameter(model.group_weights.to(device))
    
    # 3. 准备微调数据
    full_dataset = Stage3FineTuneDataset(train_jsonl, original_jsonl)
    
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    # 使用固定随机种子以确保每次划分的一致性，方便对比实验
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    print(f"📊 验证集划分完成: 训练样本数 = {len(train_dataset)}, 验证样本数 = {len(val_dataset)}")
    
    current_batch_size = config.get('finetune', {}).get('batch_size', 4)
    num_workers = config.get('finetune', {}).get('num_workers', 2)
    
    train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    # 4. 配置优化器与学习率调度器
    lr = float(config.get('finetune', {}).get('lr', 2e-5))
    weight_decay = float(config.get('finetune', {}).get('weight_decay', 0.05))
    epochs = config.get('finetune', {}).get('num_epochs', 5)
    
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=100, # 固定 100 步爬坡即可
        num_training_steps=total_steps
    )
    
    print(f"🏋️ 启动协同微调流。总 Epochs: {epochs}, 基础学习率: {lr}, 训练集总步数: {total_steps}")
    
    # 5. 训练与早停控制初始化
    os.makedirs(os.path.dirname(config['finetune']['finetune_model']), exist_ok=True)
    best_val_loss = float('inf')
    
    # 早停耐受度配置
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
            
            with torch.no_grad():
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
            
            targets_text = [f"{code}{model.llm_tokenizer.eos_token}" for code in tgt_codes]
            model.llm_tokenizer.padding_side = "right" 
            target_batch = model.llm_tokenizer(targets_text, padding=True, return_tensors="pt", truncation=True, max_length=768).to(device)
            
            with torch.no_grad():
                # 1. 优先尝试解开 Peft 包装拿到底层原生大模型
                if hasattr(model.llm_model, "base_model"):
                    raw_llm = model.llm_model.base_model.model
                else:
                    raw_llm = model.llm_model
                
                # 2. 直接安全提取 Input Embeddings 张量
                target_embeds = raw_llm.get_input_embeddings()(target_batch.input_ids)
            
            target_ids = target_batch.input_ids.clone()
            target_ids[target_ids == model.llm_tokenizer.pad_token_id] = -100
            
            full_inputs_embeds = torch.cat([inputs_embeds, target_embeds], dim=1)
            if full_inputs_embeds.requires_grad == False:
                full_inputs_embeds.requires_grad_() # 强制激活叶子节点梯度，保护 DeepSeek/Llama 推理流
            full_attention_mask = torch.cat([attention_mask, target_batch.attention_mask], dim=1)
            prompt_labels = torch.full((inputs_embeds.size(0), inputs_embeds.size(1)), -100, dtype=torch.long, device=device)
            full_labels = torch.cat([prompt_labels, target_ids], dim=1)
            
            outputs = model.llm_model(inputs_embeds=full_inputs_embeds, attention_mask=full_attention_mask, return_dict=True)
            
            logits = outputs.logits[:, inputs_embeds.size(1)-1 : -1, :].contiguous()
            loss_labels = full_labels[:, inputs_embeds.size(1):].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), loss_labels.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_train_loss += loss.item()
            progress_bar.set_postfix({"Train_Loss": f"{loss.item():.4f}"})
            del full_inputs_embeds, full_attention_mask, outputs, loss
            
        avg_train_loss = epoch_train_loss / len(train_loader)
        torch.cuda.empty_cache()
        # ----------------------------------------------------
        # 【B 阶段：验证评估循环 (Validation)】
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
                
                targets_text = [f"{code}{model.llm_tokenizer.eos_token}" for code in tgt_codes]
                model.llm_tokenizer.padding_side = "right"
                target_batch = model.llm_tokenizer(targets_text, padding=True, return_tensors="pt", truncation=True, max_length=768).to(device)
                
                # 🎯 核心修复：干净安全地剥离 PEFT 壳，拿到原生底座的 Embeddings 层
                if hasattr(model.llm_model, "base_model"):
                    raw_llm = model.llm_model.base_model.model
                else:
                    raw_llm = model.llm_model
                
                target_embeds = raw_llm.get_input_embeddings()(target_batch.input_ids)
                
                target_ids = target_batch.input_ids.clone()
                target_ids[target_ids == model.llm_tokenizer.pad_token_id] = -100
                
                full_inputs_embeds = torch.cat([inputs_embeds, target_embeds], dim=1)
                full_attention_mask = torch.cat([attention_mask, target_batch.attention_mask], dim=1)
                prompt_labels = torch.full((inputs_embeds.size(0), inputs_embeds.size(1)), -100, dtype=torch.long, device=device)
                full_labels = torch.cat([prompt_labels, target_ids], dim=1)
                
                # 🎯 强力同步：加上评估期的梯度检查点强力防御
                full_inputs_embeds = full_inputs_embeds.to(device)
                full_attention_mask = full_attention_mask.to(device)
                
                outputs = model.llm_model(inputs_embeds=full_inputs_embeds, attention_mask=full_attention_mask, return_dict=True)
                
                logits = outputs.logits[:, inputs_embeds.size(1)-1 : -1, :].contiguous()
                loss_labels = full_labels[:, inputs_embeds.size(1):].contiguous()
                
                loss_fct = nn.CrossEntropyLoss()
                val_loss = loss_fct(logits.view(-1, logits.size(-1)), loss_labels.view(-1))
                epoch_val_loss += val_loss.item()
                
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        print(f"📊 Epoch {epoch} 总结: Train Loss = {avg_train_loss:.4f} || Validation Loss = {avg_val_loss:.4f}")
        
        # ----------------------------------------------------
        # 【C 阶段：基于核心验证损失 (Validation Loss) 的早停判定】
        # ----------------------------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # 刷新耐受计数器
            save_path = config['finetune']['finetune_model']
            print(f"🏆 验证集 Loss 刷新历史最低! 正在安全保存增量融合权重至: {save_path}")
            
            # 优雅保存 requires_grad=True 参数
            trainable_param_names = {name for name, param in model.named_parameters() if param.requires_grad}
            saved_state_dict = {k: v for k, v in model.state_dict().items() if k in trainable_param_names}
            torch.save(saved_state_dict, save_path)
        else:
            patience_counter += 1
            print(f"⚠️ 验证集 Loss 未能下降。早停触发器计数: {patience_counter} / {early_stop_patience}")
            if patience_counter >= early_stop_patience:
                print(f"🛑 [早停机制触发] 检测到模型在验证集上已连续 {early_stop_patience} 个 Epoch 未能好转，正在强制终止训练以防止过拟合！")
                break
                
    print("🎉 【微调流程完美谢幕】最佳的对齐翻译模型已经安全锁在你的固态硬盘中了！")

if __name__ == "__main__":
    CONFIG_PATH = "/root/autodl-tmp/first_paper/python/stage3/stage3.yaml"
    # 二阶段自动化单元测试洗白流产生的数据集
    PASSED_CLEAN_JSONL = "/root/autodl-tmp/first_paper/python/testcase_dataset/Rust_pass/Rust_passed_clean.jsonl"
    # 一阶段输入所需的带有图特征的原始全量数据集
    ORIGINAL_STAGE3_JSONL = "/root/autodl-tmp/first_paper/python/translation_dataset/nl_code_stage3_compiled_clean.jsonl"
    
    run_finetune(CONFIG_PATH, PASSED_CLEAN_JSONL, ORIGINAL_STAGE3_JSONL)