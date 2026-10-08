import os
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "FALSE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import math
import sys
import torch
import re
import csv
import yaml
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from peft import LoraConfig, get_peft_model
from openai import OpenAI
import shutil
# 🎯 强力定位真实的 cpp 核心项目目录，防止导包死锁
cpp_project_dir = "/root/autodl-tmp/first_paper/cpp"
if cpp_project_dir not in sys.path:
    sys.path.insert(0, cpp_project_dir)

from stage3.CGBridge_Stage3 import CGBridgeStage3, CodeTranslateDataset

def is_logic_valid(code: str) -> (bool, str):
    if not code:
        return False, "FIX: Empty code."
        
    # 1. Lisp/Racket 致命伤：前置括号平衡检查
    if code.count('(') != code.count(')'):
        return False, f"FIX: Unbalanced parenthesis. ( Left: {code.count('(')}, Right: {code.count(')')} )"

    # 2. 基础占位符检查
    if "error" in code and "unimplemented" in code:
        return False, "FIX: Do not use error placeholders."
    
    # 3. 增强版：面向 C++ 到 Racket 的泄漏防御机制 (拦截 :: 和 . 符号)
    if "cout" in code or "std::" in code or ";" in code:
        return False, "FIX: C++ syntax leakage detected (semicolons or std::)."
    
    # 🎯 新增拦截：防止大模型生成诸如 Audio::Master 或 obj.method 的伪代码
    if "::" in code:
        return False, "FIX: C++ scope resolution operator '::' detected in Racket code!"
    if re.search(r'\b[a-zA-Z_]\w*\.[a-zA-Z_]\w*\b', code) and not "make-hash" in code:
        if not re.search(r'#\d+\.\d+', code): # 排除浮点数如 0.5
            return False, "FIX: C++ dot-notation object syntax detected. Use Racket accessors."

    if re.search(r'\b(std::vector|std::map|std::set|std::string)\b', code):
        return False, "FIX: C++ Standard Library containers leakage."
    if re.search(r'\b(include|using namespace|#define)\b', code):
        return False, "FIX: C++ preprocessor macros or headers detected."

    # 4. 防御 Lua/Rust 泄露
    if re.search(r'\b(local\s+function|local\s+\w+=|end\b|fn\b|impl\b|->)', code):
        return False, "FIX: Lua or Rust syntax leakage."
        
    has_idiomatic_functional = bool(
        "map" in code or "filter" in code or "foldl" in code or "foldr" in code or
        "define" in code or "let" in code or "vector" in code or "hash" in code
    )
    if not has_idiomatic_functional:
        return False, "FIX: The implementation looks empty or non-functional."

    return True, ""

def execute_racket_unit_test_reinforced(code: str, test_cases: list, timeout: int = 5) -> (bool, bool):
    """
    针对 Racket 的强力多级校验机（MultiPL-E / MBPP 适配版）：
    同时返回 (is_syntax_ok, is_unit_test_passed)，自动对齐 candidate 别名与测试环境
    """
    if not code.strip():
        return False, False

    # 1. 第一层：利用 raco make 运行基本的 Racket 语法与词法预校验
    is_syntax_ok, stderr_msg, _ = verify_racket_syntax_with_log(code, timeout=timeout)
    is_logic_clean, _ = is_logic_valid(code)
    final_syntax_compile_ok = is_syntax_ok and is_logic_clean

    if not final_syntax_compile_ok:
        return False, False 

    # 2. 第二层：处理测试用例，完美注入别名
    base_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
    
    # 🎯 【核心重构 1】从模型生成的 Racket 代码中反解真实的函数名，绑定 candidate 别名
    # 匹配模式如：(define (func_name ...) 或 (define func_name ...
    alias_str = ""
    func_match = re.search(r'\(\s*define\s*\(\s*([a-zA-Z0-9_\-\?\!]+)', code)
    if not func_match:
        func_match = re.search(r'\(\s*define\s+([a-zA-Z0-9_\-\?\!]+)', code)
        
    if func_match:
        model_func_name = func_match.group(1)
        if model_func_name != "candidate":
            alias_str += f"\n(define candidate {model_func_name})\n"
    else:
        model_func_name = None

    cleaned_tests = []
    raw_test_list = test_cases if isinstance(test_cases, list) else [str(test_cases)]
    
    # 判断是否需要引入 Racket 官方单测库
    need_rackunit = False

    for t in raw_test_list:
        t_str = str(t).strip()
        if not t_str:
            continue
            
        # 🎯 【核心重构 2】如果是标准多行或已经包含断言的 Racket 单测脚本，直接放行
        if "\n" in t_str or "check-equal?" in t_str or "test-begin" in t_str:
            if "check-" in t_str:
                need_rackunit = True
            
            # 如果单测预期的名字和模型定义的不一样，动态让预期的名字也指向大模型生成的函数
            expected_match = re.search(r'candidate\s*=\s*([a-zA-Z0-9_\-\?\!]+)', t_str)
            if expected_match and model_func_name:
                expected_func_name = expected_match.group(1)
                if expected_func_name != model_func_name:
                    alias_str += f"(define {expected_func_name} {model_func_name})\n"
            
            cleaned_tests.append(t_str)
            continue
            
        # 🎯 【核心重构 3】处理可能残留下来的 Python 风格断言或极简断言，转换为 Racket 的 rackunit 形式
        if t_str.startswith("assert "):
            need_rackunit = True
            inner = t_str[7:].strip()
            # 简单将等号或标准结构进行基础对齐转义
            if "==" in inner:
                parts = inner.split("==")
                cleaned_tests.append(f"(check-equal? {parts[0].strip()} {parts[1].strip()})")
            else:
                cleaned_tests.append(f"(check-true {inner})")
        else:
            # 默认如果是包裹在括号里的原生 Racket 测试表达式 (check-equal? ...) 
            if "check-" in t_str:
                need_rackunit = True
            cleaned_tests.append(t_str)

    # 3. 终极组装：Racket 必须保证 #lang racket 在最顶行
    # 剥离原代码中可能重复的 #lang racket
    pure_code = re.sub(r'#lang racket\n?', '', code).strip()
    
    rackunit_header = "(require rackunit)\n" if need_rackunit else ""
    test_script_str = "\n".join(cleaned_tests)
    
    full_code_with_assertions = (
        f"#lang racket\n"
        f"{rackunit_header}"
        f"{pure_code}\n\n"
        f";; --- ALIAS MAPPING ---\n{alias_str}\n"
        f";; --- EXPLICIT CRITICAL ASSERT_TRACKER ---\n{test_script_str}"
    )
    
    with tempfile.NamedTemporaryFile(suffix=".rkt", dir=base_dir, mode="w", encoding="utf-8", delete=False) as tf:
        tf.write(full_code_with_assertions)
        temp_file_path = tf.name

    try:
        # 使用 racket 命令行工具执行完整的含有单测的代码
        res = subprocess.run(
            ["racket", temp_file_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout, encoding='utf-8', errors='ignore'
        )
        
        # Racket 在 rackunit 断言失败或运行崩溃时，会向 stderr 输出错误并返回非 0 状态码
        # 检查 stderr 中是否包含 FAILURE（rackunit 报错关键字）或 Exception
        is_unit_test_ok = (
            res.returncode == 0 and 
            "failure" not in res.stderr.lower() and 
            "exception" not in res.stderr.lower()
        )
        return final_syntax_compile_ok, is_unit_test_ok
    except:
        return final_syntax_compile_ok, False
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def verify_racket_syntax_with_log(code: str, timeout: int = 5):
    """
    使用 Racket 官方命令行工具进行编译与语法检查
    """
    base_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
    
    # Racket 脚本通常以 .rkt 结尾
    with tempfile.NamedTemporaryFile(suffix=".rkt", dir=base_dir, mode="w", encoding="utf-8", delete=False) as tf:
        tf.write(code)
        temp_file_path = tf.name

    try:
        # 使用 racket 命令行执行代码检查。
        # Racket 可以直接载入并编译该文件，如果有未闭合括号或未定义变量会直接报错
        check_res = subprocess.run(
            ["racket", temp_file_path], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='ignore'
        )

        if check_res.returncode == 0:
            return True, "Success", []

        stderr = check_res.stderr.strip()
        clean_error = stderr.replace(temp_file_path, "line")
        
        # 错题本分类细化（Racket 常见错误分类）
        error_tags = []
        if "read" in stderr and "parenthesis" in stderr:
            error_tags.append("Racket_UnbalancedParenthesis") # 括号未闭合（Lisp致命伤）
        elif "unbound" in stderr:
            error_tags.append("Racket_UnboundIdentifier")    # 变量/函数未定义
        else:
            error_tags.append("Racket_CompileOrRuntimeError")

        return False, clean_error, error_tags

    except subprocess.TimeoutExpired:
        return False, "Parsing timeout", ["TIMEOUT"]
    except FileNotFoundError:
        return False, "Racket CLI not found. Please install it via 'sudo apt-get install racket'", ["ENV_ERROR"]
    except Exception as e:
        return False, f"Execution failed: {str(e)}", ["EXEC_ERROR"]
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
def clean_generated_code(code: str, is_repair_mode: bool = False) -> str:
    if not code:
        return ""

    # 🎯 1. 优先提取 ```racket ... ``` 块内部的代码
    racket_blocks = re.findall(r'```racket\s*\n?(.*?)\n?```', code, re.DOTALL)
    if not racket_blocks:
        # 🎯 2. 次优先提取通用的 ``` ... ``` 块
        racket_blocks = re.findall(r'```\s*\n?(.*?)\n?```', code, re.DOTALL)
        
    final_code = racket_blocks[-1] if racket_blocks else code

    # 清理大模型可能夹带的 Qwen 标签
    final_code = re.sub(r'<\|im_.*?\|>', '', final_code)
    final_code = final_code.strip()
    
    # 🎯 3. 强力防御：确保生成的代码以 #lang racket 开头，如果大模型漏写了，帮它补上
    if final_code and not final_code.startswith("#lang"):
        final_code = "#lang racket\n\n" + final_code
            
    return final_code

def calculate_pass_at_k(n, c, k):
    if c - n < k:
        return 1.0
    return 1.0 - math.comb(c - n, k) / math.comb(c, k)
    
def evarackette_pass_at_k_hybrid_job(candidates: list, test_cases: list):
    syntax_results = []
    unittest_results = []
    
    for cand in candidates:
        syn_ok, ut_ok = execute_racket_unit_test_reinforced(cand, test_cases)
        syntax_results.append(syn_ok)
        unittest_results.append(ut_ok)
        
    compile_pass = 1 if any(syntax_results) else 0
    total_cands = len(candidates)              
    passed_cands = sum(1 for x in unittest_results if x) 
    
    p1 = calculate_pass_at_k(passed_cands, total_cands, k=1)
    p2 = calculate_pass_at_k(passed_cands, total_cands, k=2)
    p5 = calculate_pass_at_k(passed_cands, total_cands, k=5)
    
    passed_code = ""
    for idx, ut_passed in enumerate(unittest_results):
        if ut_passed:
            passed_code = candidates[idx]
            break
            
    return compile_pass, p1, p2, p5, passed_code

def batch_smart_generate(model, graph_embs, codes, nl_descs, config, device):
    use_graph_info = config['generation'].get('use_graph_info', True)
    num_candidates = config['generation'].get('num_candidates', 5)
    max_new_tokens = config['generation'].get('max_new_tokens', 1024)
    
    # 1. 保存当前 padding_side，并强制设定为 left 用于批推理，防止位置崩塌
    old_side = model.llm_tokenizer.padding_side
    model.llm_tokenizer.padding_side = "left"

    if use_graph_info:
        query_feats = graph_embs.to(device).to(dtype=model.llm_model.dtype)
        with torch.no_grad():
            inputs_embeds, attn_mask = model.prepare_lm_input(
                gtokens=query_feats, text_input=codes, nl_desc=nl_descs, use_context=True
            )
    else:
        fake_gtokens = torch.zeros((graph_embs.size(0), 32, 768), device=device)
        with torch.no_grad():
            inputs_embeds, attn_mask = model.prepare_lm_input(
                gtokens=fake_gtokens, text_input=codes, nl_desc=nl_descs, use_context=False
            )

    if num_candidates > 1:
        inputs_embeds = inputs_embeds.repeat_interleave(num_candidates, dim=0)
        attn_mask = attn_mask.repeat_interleave(num_candidates, dim=0)

    llm_type = config["model"].get("llm_type", "qwen").lower()
    custom_eos_token = None

    if "llama" in llm_type:
        # Llama 3 Chat/Instruct 的标准结束符是 <|eot_id|>
        custom_eos_token = "<|eot_id|>"
    elif "llamabase" in llm_type:
        custom_eos_token = None  # 走下面的默认实际实际基座实际 actual_eos_id
    elif "qwen" in llm_type:
        # Qwen 2/2.5 ChatML 格式的结束符是 <|im_end|>
        custom_eos_token = "<|im_end|>"
    elif "deepseek" in llm_type:
        # DeepSeek-V2 / DeepSeek-Coder-Instruct 标准结束符也是 <|im_end|> 或者是 <｜end of sentence｜>
        # 优先探测标准的 ChatML 结束符
        custom_eos_token = "<|im_end|>" if "<|im_end|>" in model.llm_tokenizer.get_vocab() else None
    elif "starcoder" in llm_type:
        vocab = model.llm_tokenizer.get_vocab()
        custom_eos_token = "<|endoftext|>"
        
    # 动态转换 Token ID 并设置安全防线
    actual_eos_ids = [model.llm_tokenizer.eos_token_id] # 默认包含基座自带的 eos_token_id
    
    if custom_eos_token:
        token_id = model.llm_tokenizer.convert_tokens_to_ids(custom_eos_token)
        if token_id is not None and token_id != model.llm_tokenizer.unk_token_id:
            if token_id not in actual_eos_ids:
                actual_eos_ids.append(token_id)
                
    # 强行塞入咱们在微调和特征对齐时辛辛苦苦扩充的特异性控制符作为终极物理墙
    for extra_token in ["<|global_end|>", "<|im_end|>", "<|eot_id|>", "</s>"]:
        t_id = model.llm_tokenizer.convert_tokens_to_ids(extra_token)
        if t_id is not None and t_id != model.llm_tokenizer.unk_token_id:
            if t_id not in actual_eos_ids:
                actual_eos_ids.append(t_id)

    # 根据模型微调状态自适应调整重复惩罚：针对小样本微调后的 Llama 稍微拉高防止套娃
    current_rep_penalty = 1.1 if "llama" in llm_type else config['generation'].get('repetition_penalty', 1.05)
    # ==============================================================================

    with torch.no_grad():
        generate_ids = model.llm_model.generate(
            inputs_embeds=inputs_embeds, 
            attention_mask=attn_mask, 
            max_new_tokens=max_new_tokens,
            temperature=config['generation'].get('temperature', 0.9), 
            top_p=config['generation'].get('top_p', 0.95),
            repetition_penalty=current_rep_penalty, 
            use_cache=True,
            do_sample=True if num_candidates > 1 else False, 
            pad_token_id=model.llm_tokenizer.pad_token_id,
            eos_token_id=actual_eos_ids  # 🎯 动态锁死当前模型真正的结束符
        )
    all_raw_texts = model.llm_tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
    # print("🚨 [DEBUG Llama 3 真实吐字快照]：")
    # print(f"--- 候选 1 ---\n{all_raw_texts[0]}")
    # print("───────────────────────────────────")
    batch_results = []
    origin_bsz = graph_embs.size(0)
    for i in range(origin_bsz):
        start_idx = i * num_candidates
        end_idx = (i + 1) * num_candidates
        sample_candidates = [clean_generated_code(t) for t in all_raw_texts[start_idx:end_idx]]
        batch_results.append(sample_candidates)
        
    model.llm_tokenizer.padding_side = old_side
    return batch_results

def custom_collate_fn(batch):
    """
    自定义 DataLoader 打包函数。
    拦截 PyTorch 的默认打包行为，防止其因为测试用例长度不一而报错或折叠列表。
    """
    # batch 是一个列表，里面包含了当前 Batch 大小的元组：
    # [(idx, graph_emb, cpp_code, nl_desc, unit_tests), ...]
    
    idxs = [item[0] for item in batch]
    # 图特征是 Tensor，我们需要手动把它 stack 起来拼成 Batch 维度 [B, 32, 768]
    graph_embs = torch.stack([item[1] for item in batch])
    cpp_codes = [item[2] for item in batch]
    nl_descs = [item[3] for item in batch]
    
    # 测试用例保持原汁原味的 Python 嵌套列表，不强行转换
    unit_tests = [item[4] for item in batch]
    
    return idxs, graph_embs, cpp_codes, nl_descs, unit_tests

class PassKEvarackettionDataset(CodeTranslateDataset):
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        
        raw_emb = item.get('graph_emb')
        graph_emb = torch.tensor(raw_emb, dtype=torch.float) if not isinstance(raw_emb, str) else torch.tensor(json.loads(raw_emb), dtype=torch.float)
        
        # 1. 提取测试用例
        unit_tests = item.get('tests_racket') or item.get('tests') or item.get('test_cases')
        
        import pandas as pd
        if isinstance(unit_tests, float) and pd.isna(unit_tests):
            unit_tests = []
        elif isinstance(unit_tests, str):
            try:
                unit_tests = json.loads(unit_tests)
            except:
                unit_tests = []
        
        if not unit_tests:
            unit_tests = []
            
        # 2. 提取文本
        cpp_code_str = str(item.get('cpp_source', item.get('cpp_code', "")))
        nl_desc_str = str(item.get('racket_prompt', item.get('nl_description', "")))
            
        return (
            item.get('idx', idx),
            graph_emb,
            cpp_code_str,
            nl_desc_str,
            unit_tests  # 🌟 直接返回原生 Python 列表，没有任何包袱
        )

def run_reinforced_inference(config_path):
    with open(config_path, 'r', encoding='utf-8') as f: 
        config = yaml.safe_load(f)
    
    tgt_language = config.get('tgt_language', 'Lua')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    use_api_mode = config["model"].get("use_api_mode", False)
    llm_type = config["model"].get("llm_type", "qwen").lower()
    use_graph = config.get("generation", {}).get("use_graph_info", False) # 💡 动态读取 YAML 配置中的消融开关
    
    print(f"\n================ ⚙️ 启动完全体【消融实验：纯代码/纯文本基座推理管线】================")
    print(f"当前运行模式: {'【图增强模式】' if use_graph else '【纯代码消融模式】'} | 目标基座: {llm_type.upper()}")
    
    import gc
    gc.collect()
    torch.cuda.empty_cache() # 强制排空显存残留
    
    # 1. 实例化基本模型架构
    model = CGBridgeStage3(config)
    
    # ========================================================================
    # 🛠️ 消融改造一：根据 use_graph_info 状态自适应控锁二阶段状态
    # ========================================================================
    if use_graph:
        print("🚚 正在激活图增强评估流，锁死 Q-Former 推理期状态...")
        if hasattr(model, "stage2_bridge") and model.stage2_bridge is not None:
            model.stage2_bridge.eval()
            if hasattr(model.stage2_bridge, "Qformer"):
                model.stage2_bridge.Qformer.eval()
                model.stage2_bridge.Qformer.bert.config.gradient_checkpointing = False
    else:
        print("⚠️ [消融警报开启]：图增强流已被物理切断，Q-Former 特征提取在前向传播中将被跳过！")

    # ========================================================================
    # 🛠️ 消融改造二：绕过 LoRA 结构挂载与微调字典加载，直接让原生大模型基座登舰
    # ========================================================================
    if not use_api_mode:
        if use_graph:
            # 正常实验模式：需要同步挂载 LoRA 结构并安全融合微调权重
            print("🎨 正在动态同步大模型 LoRA 适配器结构（与训练完全看齐）...")
            if "deepseek" in llm_type:
                print("💡 路由选择：检测到 DeepSeek 系列，挂载 MLA 专属低秩注意力旁路矩阵...")
                target_mods = ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
            elif "starcoder" in llm_type:
                print("💡 路由选择：检测到 StarCoder 家族，挂载其特异性注意力投影层...")
                target_mods = ["c_attn", "c_proj", "c_fc", "q_proj", "kv_proj", "o_proj"]  # StarCoder 经典的合并注意力机制
            elif "qwen" in llm_type:
                print("💡 路由选择：检测到标准 MHA 架构 (Qwen/Llama)，挂载常规注意力靶向层...")
                target_mods = ["q_proj", "v_proj"]
            elif "llama" in llm_type or "llamabase" in llm_type:
                target_mods = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            else:
                # 兜底通用标准层
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
            
            stage3_ckpt = config.get('finetune', {}).get('finetune_model', '')
            if os.path.exists(stage3_ckpt):
                print(f"🔥 [精准靶向融合] 检测到微调模型，正在调用安全融合机制: {stage3_ckpt}...")
                res = model.load_tuned_stage3_weights(stage3_ckpt)
                if res is not None:
                    print(f"✅ 增量参数同步报告 ── 缺失项总数: {len(res.missing_keys)}")
            else:
                print(f"❌ 严重错误: 找不到微调模型路径 {stage3_ckpt}！")
                return
        else:
            # 纯文本消融模式：拒绝任何微调参数污染，直接以 100% 纯原生底座启动加载
            print("🎨 正在以 [纯原生大模型基座] 模式启动硬件加载...")
            
        model.to(device)
        model.eval()

    # ========================================================================
    # 📦 3. 准备独立多语言测试集并加载
    # ========================================================================
    print(f"📖 正在加载独立多语言测试集: {config['data']['test_path']} ...")
    dataset = PassKEvarackettionDataset(config['data']['test_path'])
    
    # 🎯 核心修改：在 DataLoader 中增加 collate_fn=custom_collate_fn
    dataloader = DataLoader(
        dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=False, 
        num_workers=config['data']['num_workers'],
        collate_fn=custom_collate_fn  # <--- 加上这一行
    )

    output_csv = config['output']['multilang_evaluation_report']
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    total_samples = len(dataset)
    compile_success_total = 0
    pass_1_total = 0
    pass_2_total = 0
    pass_5_total = 0
    processed_samples_count = 0
    
    max_workers = min(os.cpu_count(), 16)
    print(f"🚀 双轨单元测试评测开启！目标语言: {tgt_language} | 题目总数: {total_samples} | CPU 并行度: {max_workers}")

    with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'cpp_code', 'nl_desc', 'all_candidates', 'passed_code', 'compile_ok', 'pass_1', 'pass_2', 'pass_5'])

        # 全局锁死不计算梯度
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating & Dual Evaluating"):
                idxs, gtokens, cpp_codes, nl_descs, batch_unit_tests = batch
                
                # 1. 大模型推理生成候选 (显存占用高峰)
                batch_candidates = batch_smart_generate(
                    model, gtokens, cpp_codes, nl_descs, config, device)
                
                # 2. 局部开启多进程池，处理完当前 Batch 立即释放，彻底阻断异步积压
                batch_futures = []
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    for i in range(len(idxs)):
                        raw_idx = idxs[i].item() if isinstance(idxs[i], torch.Tensor) else str(idxs[i])
                        future = executor.submit(evarackette_pass_at_k_hybrid_job, batch_candidates[i], batch_unit_tests[i])
                        batch_futures.append({
                            'idx': raw_idx, 'cpp_code': cpp_codes[i], 'nl_desc': nl_descs[i],
                            'candidates': batch_candidates[i], 'future': future
                        })

                    # 3. 阻塞式等待当前 Batch 所有的 CPU 评测任务结束
                    for item in batch_futures:
                        try:
                            comp_ok, p1, p2, p5, passed_code = item['future'].result()
                            compile_success_total += comp_ok
                            pass_1_total += p1
                            pass_2_total += p2
                            pass_5_total += p5
                            
                            writer.writerow([
                                item['idx'], item['cpp_code'], item['nl_desc'],
                                json.dumps(item['candidates'], ensure_ascii=False),
                                passed_code, comp_ok, p1, p2, p5
                            ])
                            
                            processed_samples_count += 1
                            
                            if processed_samples_count % 10 == 0:
                                tqdm.write(f"\n📢 [中间进度报告] ── 累计已处理数: {processed_samples_count}/{total_samples}")
                                tqdm.write(f"    ├─  当前累计编译通过率: {compile_success_total / processed_samples_count * 100:.2f}%")
                                tqdm.write(f"    ├─ 真实单元测试 Pass@1: {pass_1_total / processed_samples_count * 100:.2f}%")
                                tqdm.write(f"    ├─ 真实单元测试 Pass@2: {pass_2_total / processed_samples_count * 100:.2f}%")
                                tqdm.write(f"    └─ 真实单元测试 Pass@5: {pass_5_total / processed_samples_count * 100:.2f}%\n")
                                
                        except Exception as e:
                            print(f"\n[Error] 样本 {item['idx']} 评测异常: {e}")
                
                f.flush()
                
                # 4. 彻底斩断引用链，强制释放显存
                del batch_candidates, batch_futures, idxs, gtokens
                import gc
                gc.collect()
                if not use_api_mode:
                    torch.cuda.empty_cache()

    # --- 终极学术战绩报告汇报 ---
    print(f"\n🏆 ======================== CGBridge 终极 {tgt_language} 多语言翻译评测报告 ========================")
    print(f"📊 评估题目总计: {total_samples} 道")
    print(f"🛠️ 基础语法/编译通过率 (Compilation Pass Rate): {compile_success_total / total_samples * 100:.2f}%")
    print(f"📈 真实自动化单元测试 Pass@1 成功率: {pass_1_total / total_samples * 100:.2f}%")
    print(f"📈 真实自动化单元测试 Pass@2 成功率: {pass_2_total / total_samples * 100:.2f}%")
    print(f"📈 真实自动化单元测试 Pass@5 成功率: {pass_5_total / total_samples * 100:.2f}%")
    print(f"==================================================================================\n")

if __name__ == "__main__":
    CONFIG_FILE = "/root/autodl-tmp/first_paper/cpp/stage3/stage3.yaml"
    run_reinforced_inference(CONFIG_FILE)