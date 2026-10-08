import os, re
import json
import yaml
import torch
import torch.nn as nn
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import torch.nn.functional as F
import types
import pandas as pd
# 动态添加项目根目录，确保能找到 stage2 文件夹
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../stage3
project_root = os.path.dirname(current_dir)             # .../cpp (如果 stage2 在这里)
# 如果 cpp 文件夹外面还有一层 root，请确保 project_root 指向包含 'stage2' 文件夹的那一层
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 统一使用相对于 project_root 的路径
from stage2.Stage2_AlignmentBridge import InnovativeStage2Bridge

class CGBridgeStage3(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 配置参数
        self.llm_dir = self.config["model"]["llm_name_or_path"]
        self.max_txt_len = self.config["model"].get("max_txt_len", 512)

        # 2. 初始化 Stage 2 模块 (包含 Q-Former)
        self.stage2_bridge = InnovativeStage2Bridge(self.config)
        self._load_stage2_weights()

        # 3. 初始化 LLM
        self.llm_tokenizer, self.llm_model = self._init_llm()

        # 4. 定义投影层 (Linear 层默认 requires_grad=True)
        # 确保维度对齐：Q-Former 输出 (768) -> LLM 输入维度
        d_graph = self.config["model"].get("graph_width", 768)
        d_llm = self.llm_model.config.hidden_size
        self.llm_proj = nn.Sequential(
            nn.Linear(d_graph, 512),
            nn.GELU(),
            nn.Dropout(0.1),  # 💡 新增：扰动微调空间，防止 200 条数据导致神经元协同过拟合
            nn.LayerNorm(512),
            nn.Linear(512, d_llm),
            nn.LayerNorm(d_llm)
        )
        torch_dtype = torch.bfloat16 if self.config["model"].get("llm_dtype") == "bfloat16" else torch.float16
        self.llm_proj.to(torch_dtype)
        self.group_weights = nn.Parameter(torch.ones(4))

        # --- 核心冻结逻辑：和你要求的一样 ---
        # 只冻结 LLM 模型的参数
        for param in self.llm_model.parameters():
            param.requires_grad = False

    # 在 CGBridge_Stage3 顶层新增一个安全加载函数，专供推理脚本调用
    def load_tuned_stage3_weights(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            print(f"🔄 [绝对融合] 正在加载 Stage 3 完整微调字典...")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            
            llm_type = self.config["model"].get("llm_type", "qwen").lower()
            found_embed_key = None

            if "starcoder" in llm_type:
                # StarCoder 通常为 transformer.wte.weight 或带有 peft 外壳的 base_model.model.transformer.wte.weight
                possible_keys = ["llm_model.transformer.wte.weight", "transformer.wte.weight", "base_model.model.llm_model.transformer.wte.weight"]
                found_embed_key = next((k for k in possible_keys if k in state_dict), None)
                
            elif "deepseek" in llm_type:
                # DeepSeek 系列遵循类似 Llama 命名
                possible_keys = ["llm_model.model.embed_tokens.weight", "llm_model.embed_tokens.weight", "model.embed_tokens.weight"]
                found_embed_key = next((k for k in possible_keys if k in state_dict), None)
                
            elif "codellama" in llm_type:
                # CodeLlama 遵循标准 Llama 命名
                possible_keys = ["llm_model.model.embed_tokens.weight", "llm_model.embed_tokens.weight", "model.embed_tokens.weight"]
                found_embed_key = next((k for k in possible_keys if k in state_dict), None)
            else:
                # 泛用性兜底扫描
                for k in state_dict.keys():
                    if "embed_tokens.weight" in k or "wte.weight" in k:
                        found_embed_key = k
                        break

            # 执行防御性词表重缩放
            if found_embed_key:
                saved_vocab_size = state_dict[found_embed_key].size(0)
                if saved_vocab_size != len(self.llm_tokenizer):
                    print(f"⚠️ 检测到目标基座 [{llm_type.upper()}] 词表膨胀，正在动态对齐空间: {saved_vocab_size}")
                    self.llm_model.resize_token_embeddings(saved_vocab_size)
                    
            # 强力加载
            res = self.load_state_dict(state_dict, strict=False)
            print(f"✅ 加载完成。关键漏项检查 (应当为空): {[k for k in res.missing_keys if 'llm_proj' in k]}")
            return res

    def _load_stage2_weights(self):
        s2_ckpt = self.config["model"].get("stage2_model_dir", "")
        if os.path.exists(s2_ckpt):
            # 1. 加载保存的 .pt 文件
            state_dict = torch.load(s2_ckpt, map_location="cpu", weights_only=False)

            load_result = self.stage2_bridge.load_state_dict(state_dict, strict=False)
            # 3. 打印详细信息确保关键层已加载
            print(f"--- Stage 2 Weight Loading Report ---")
            print(f"Missing Keys: {len(load_result.missing_keys)} (这些层没找到权重，将保持随机初始化)")
            print(f"Unexpected Keys: {len(load_result.unexpected_keys)} (权重文件里有多余的参数)")

            # 验证 Cross-Attention 是否在 Missing Keys 里
            ca_missing = [k for k in load_result.missing_keys if "crossattention" in k]
            if ca_missing:
                print(f"❌ 严重警告: 发现 Cross-Attention 权重未加载成功！请检查 Key 是否匹配。")
            else:
                print(f"✅ 确认: Cross-Attention 权重已成功从 Checkpoint 覆盖。")

    def _init_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(self.llm_dir, trust_remote_code=True)
        special_tokens = [
            "<|graph|>",
            "<|struct_start|>", "<|struct_end|>",
            "<|logic_start|>", "<|logic_end|>",
            "<|semantic_start|>", "<|semantic_end|>",
            "<|global_start|>", "<|global_end|>"  # 🎯 确保这两个词在列表里！
        ]
        new_tokens = [t for t in special_tokens if t not in tokenizer.get_vocab()]
        if new_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        if "<|graph|>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": ["<|graph|>"]})
            
        llm_type = self.config["model"].get("llm_type", "qwen").lower()
        
        if "starcoder" in llm_type:
            # StarCoder 默认没有 pad_token，指定为其标准的 eos_token (<|endoftext|>)
            tokenizer.pad_token = tokenizer.eos_token
        elif "deepseek" in llm_type:
            # DeepSeek ChatML 格式若没有专门指定，建议赋予其内置的特殊 token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        elif "llama" in llm_type or "llamabase" in llm_type:
            # CodeLlama 默认不带 pad_token，严格指定为 <s> 或 </s>
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token

        torch_dtype = torch.bfloat16 if self.config["model"].get("llm_dtype") == "bfloat16" else torch.float16
        
        config_json_path = os.path.join(self.llm_dir, "config.json")
        if "llama" in llm_type and os.path.exists(config_json_path):
            print("⚙️ [Llama3 物理适配] 正在直接对磁盘 config.json 文件进行脱敏改写...")
            try:
                with open(config_json_path, 'r', encoding='utf-8') as f:
                    disk_config = json.load(f)
                
                # 如果检测到含有高版本不兼容的 rope_scaling 字段，直接从根源抹除并回写磁盘
                if "rope_scaling" in disk_config:
                    disk_config.pop("rope_scaling", None)
                    with open(config_json_path, 'w', encoding='utf-8') as f:
                        json.dump(disk_config, f, indent=2, ensure_ascii=False)
                    print("✅ 磁盘 config.json 违规 RoPE 字段已成功剔除，环境已被强力净化！")
            except Exception as e:
                print(f"⚠️ 物理改写磁盘 config.json 失败: {e}，将尝试通过内存兜底。")
    
        # 1. 加载目标大模型 (Target Model)
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu" # 🎯 强制锁定在同一张卡上
        )
    
        model.resize_token_embeddings(len(tokenizer))
        model.eval()
        return tokenizer, model

    def prepare_lm_input(self, gtokens, text_input, nl_desc, use_context=True):
        bsz = gtokens.size(0)
        full_texts = []
        
        llm_type = self.config["model"].get("llm_type", "qwen").lower()
        
        # 1. 批量构造 Prompt 字符串
        for i in range(bsz):
            # 🎯 核心精髓：
            # 如果是正常模式 (use_context=True)，使用标准的完整交叉模态 Prompt 结构
            if use_context:
                struct_str   = f"<|struct_start|>{'<|graph|>' * 10}<|struct_end|>"
                logic_str    = f"<|logic_start|>{'<|graph|>' * 10}<|logic_end|>"
                semantic_str = f"<|semantic_start|>{'<|graph|>' * 10}<|semantic_end|>"
                global_str   = f"<|global_start|>{'<|graph|>' * 2}<|global_end|>"
                
                user_p = self.config["prompts"]["translation"]["user_prompt"].format(
                    struct_tokens=struct_str,
                    logic_tokens=logic_str,
                    semantic_tokens=semantic_str,
                    global_tokens=global_str, 
                    nl_description=nl_desc[i] or "None",
                    cpp_code=text_input[i]
                )
            else:
                tgt_lang_name = self.config.get("tgt_language", "R")
                user_p = (
                            f"# C++ Implementation:\n"
                            f"# ```cpp\n"
                            f"# {text_input[i].replace('\n', '\n# ')}\n"
                            f"# ```\n"
                            f"# Intent: {nl_desc[i] or f'Translate cpp to native {tgt_lang_name} code'}\n"
                            f"# Idiomatic {tgt_lang_name} Translation:\n"
                        )
            
            # ==================== 多模型分流适配 ====================
            if "starcoder" in llm_type:
                # 替换为 R 语言的标准注释符 #
                sys_comment = f"(*\n{self.config['prompts']['translation']['system_prompt']}\n*)"
                prompt = (
                    f"## SYSTEM GUIDELINES:\n## {sys_comment}\n\n"
                    f"{user_p}\n\n"
                    f"### R Target Implementation\n```r\n"
                )
            elif "llama" in llm_type:
                prompt = (
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    f"{self.config['prompts']['translation']['system_prompt']}<|eot_id|>"
                    f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"{user_p}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            elif "llamabase" in llm_type:
                # 💡 Base 模型无视 Instruct 标签，直接改用标准的“前置上下文 + 强暗示”续写结构
                prompt = (
                    f"{user_p}\n"
                )
            else:
                # Qwen / DeepSeek 的标准 ChatML 封装
                prompt = (
                    f"<|im_start|>system\n{self.config['prompts']['translation']['system_prompt']}<|im_end|>\n"
                    f"<|im_start|>user\n{user_p}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
            full_texts.append(prompt)
    
        # 2. 文本转为 Embeddings
        self.llm_tokenizer.padding_side = "left" 
        batch = self.llm_tokenizer(full_texts, padding=True, return_tensors="pt").to(gtokens.device)
        inputs_embeds = self.llm_model.get_input_embeddings()(batch.input_ids)
        
        # 3. --- 张量化注入逻辑 ---
        if use_context:
            proj_dtype = next(self.llm_proj.parameters()).dtype
            projected_all = self.llm_proj(gtokens.to(dtype=proj_dtype)) 
            
            s_reps  = projected_all[:, 0:10, :]   
            l_reps  = projected_all[:, 10:20, :]  
            se_reps = projected_all[:, 20:30, :]  
            p_reps  = projected_all[:, 30:32, :]  
            
            group_w = F.softmax(self.group_weights, dim=0)  
            s_reps = s_reps * group_w[0]
            l_reps = l_reps * group_w[1]
            se_reps = se_reps * group_w[2]
            p_reps = p_reps * group_w[3]
            ordered_reps = torch.cat([s_reps, l_reps, se_reps, p_reps], dim=1)
            
            # 🎯 新增防暴走机制：自适应将图特征的范数缩放到与文本 Embeddings 相同的量级
            with torch.no_grad():
                text_scale = inputs_embeds.abs().mean()
                graph_scale = ordered_reps.abs().mean()
                if graph_scale > 0:
                    scale_factor = text_scale / graph_scale
                else:
                    scale_factor = 1.0
            
            # 如果缩放因子过大或过小（证明不平衡），激活安全动态平滑
            ordered_reps = ordered_reps * scale_factor
            
            graph_token_id = self.llm_tokenizer.convert_tokens_to_ids("<|graph|>")
            pos_mask = (batch.input_ids == graph_token_id)
            
            inputs_embeds[pos_mask] = ordered_reps.view(-1, ordered_reps.size(-1)).to(inputs_embeds.dtype)
                
        return inputs_embeds, batch.attention_mask

    def forward(self, batch_data):
        """
        🎯 修复：使 forward 行为与训练（finetune_human_eval.py）完全同步。
        避免直接在前部拼接 gtokens 导致的位置编码空间混乱。
        """
        # 解包 Dataset 返回的 4 元组
        idx, graph_embs, codes, p_descs = batch_data
        device = graph_embs.device

        # 1. 调用 Stage 2 桥接提取融合特征
        samples = {
            'node_embs': graph_embs.to(device),
            'text_input_ids': torch.zeros((graph_embs.size(0), 1), dtype=torch.long).to(device),
            'text_atts': torch.zeros((graph_embs.size(0), 1), dtype=torch.long).to(device),
            'meta': [{'node_type_map': []} for _ in range(graph_embs.size(0))]
        }
        s2_outputs = self.stage2_bridge(samples, task='test')
        query_feats = s2_outputs['query_output'].to(dtype=self.llm_model.dtype)

        # 2. 统一走定点插值注入管线
        inputs_embeds, attention_mask = self.prepare_lm_input(
            gtokens=query_feats, 
            text_input=codes, 
            nl_desc=p_descs, 
            use_context=True
        )

        # 3. 送入大模型 Backbone
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs

class CodeTranslateDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path):
        self.df = pd.read_json(jsonl_path, lines=True)
        # 确保关键字段不为空
        self.df = self.df.dropna(subset=['graph_emb'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        # 处理 graph_emb，JSONL 中通常直接是 list，无需 json.loads
        raw_emb = item['graph_emb']
        if isinstance(raw_emb, str):
            graph_emb = torch.tensor(json.loads(raw_emb), dtype=torch.float)
        else:
            graph_emb = torch.tensor(raw_emb, dtype=torch.float)

        return (
            item.get('idx', idx),  # 容错处理
            graph_emb,
            str(item.get('src_code', item.get('cpp_code', ""))),  # 兼容不同键名
            str(item.get('nl_description', ""))
        )