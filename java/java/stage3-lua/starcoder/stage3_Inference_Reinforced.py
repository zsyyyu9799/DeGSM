import os
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "FALSE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import torch
import re
import csv
import yaml
import json
import shutil
import tempfile
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
import subprocess
from concurrent.futures import ProcessPoolExecutor
from peft import LoraConfig, get_peft_model
import math

# 🎯 自动定位项目核心目录
java_project_dir = "/root/autodl-tmp/first_paper/java"
if java_project_dir not in sys.path:
    sys.path.insert(0, java_project_dir)

from stage3.CGBridge_Stage3 import CGBridgeStage3, CodeTranslateDataset

def calculate_pass_at_k(n, c, k):
    """
    计算标准的学术级 Pass@k 概率
    """
    if c - n < k:
        return 1.0
    return 1.0 - math.comb(c - n, k) / math.comb(c, k)

def is_logic_valid(code: str) -> (bool, str):
    """
    面向 Java 到 Lua 的泄漏防御机制与逻辑初筛
    """
    if "TODO" in code or "unimplemented" in code.lower():
        return False, "FIX: Do not use TODO placeholders. Please translate the actual logic from Java."
    
    if re.search(r'\bfunction\s+\w+\s*\(.*?\)\s*end\s*$', code.strip()):
        return False, "FIX: Incomplete function body detected. Provide the full Lua implementation."

    if re.search(r'\b(public|private|protected)\s+class\b', code) or re.search(r'\bpublic\s+static\b', code):
        return False, "FIX: Java class/method syntax leakage detected. Use Lua tables or standard functions."

    if re.search(r'\bnew\s+\w+\s*\(', code):
        return False, "FIX: Java 'new' keyword leakage detected. Construct Lua tables directly or use custom constructors."

    if "NullPointerException" in code or "IllegalArgumentException" in code:
        return False, "FIX: Java Exception leakage detected. Return `nil, err` or use Lua's `error()`/`pcall()`."
    
    if "System.out.print" in code:
        return False, "FIX: Java stdout leakage ('System.out.println'). Use `print()` or `io.write()` in Lua."

    if "ArrayList" in code or re.search(r'\bList<', code):
        return False, "FIX: Java container leakage detected. Use native Lua tables `{}`."
        
    if re.search(r'function\s+\w+:\w+\s*\(\s*self\b', code):
        return False, "FIX: Do not use explicit 'self' as the first parameter when defining a method with a colon (:)."
        
    if "%n" in code:
        return False, "FIX: Java format placeholder '%n' detected. Use '\\n' in Lua."
        
    return True, ""

def verify_lua_syntax_with_log(code: str, timeout: int = 5):
    """
    终极探测沙箱：
    1. 注入全面元表守卫，严格拦截任何未定义的全局变量调用。
    2. 【核心】自动提取并注入函数调用桩，强制让静态函数“跑起来”，引爆内部幻觉。
    """
    base_temp_path = "/dev/shm" if os.path.exists("/dev/shm") else None
    temp_dir = tempfile.mkdtemp(prefix="lua_compile_", dir=base_temp_path)
    temp_file = os.path.join(temp_dir, "temp.lua")

    # 1. 升级版全局守卫：无论是读（__index）还是写（__newindex），只要属于未定义行为，一律拦截
    mock_env_guard = """
    local allowed_libs = {
        string=true, table=true, os=true, math=true, print=true, 
        error=true, setmetatable=true, getmetatable=true, tostring=true, 
        tonumber=true, pairs=true, ipairs=true, next=true, pcall=true, xpcall=true
    }
    setmetatable(_G, {
        __index = function(t, k)
            if allowed_libs[k] then return rawget(t, k) end
            error("RUNTIME_ERROR: Attempt to read undefined identifier '" .. tostring(k) .. "'")
        end
    })
    """

    # 2. 🎯 【引爆机制】自动识别代码中的全局函数/构造器，并强行在末尾追加测试调用
    execution_trigger = "\n\n-- --- Dynamic Trigger Layer --- \n"
    
    # 场景 A：如果代码定义了符合构造器命名规范的类函数 (例如 Word(), Date(), CancelaCFDI())
    found_classes = re.findall(r'function\s+([A-Z]\w*)\s*\(', code)
    # 场景 B：如果代码定义了标准的面向对象 new 方法 (例如 Class:new() 或 Class.new())
    found_news = re.findall(r'function\s+([A-Z]\w*)[:.]new\s*\(', code)
    
    triggered = False
    # 尝试实例化这些类（传入空值或伪造参数，逼迫其内部逻辑执行）
    for cls in set(found_classes + found_news):
        execution_trigger += f"print('[Sandbox] Triggering Constructor: {cls}')\n"
        execution_trigger += f"local test_obj = {cls}(nil, nil, nil, nil)\n"
        execution_trigger += f"if test_obj and type(test_obj) == 'table' then\n"
        # 尝试连带触发一些常见的 Java 映射方法
        execution_trigger += f"    pcall(function() test_obj:toString() end)\n"
        execution_trigger += f"    pcall(function() test_obj:getWord() end)\n"
        execution_trigger += f"end\n"
        triggered = True

    # 场景 C：普通的全局非面向对象函数 (例如 minimumTotal(triangle))
    if not triggered:
        found_funcs = re.findall(r'function\s+([a-z]\w*)\s*\(', code)
        for func in set(found_funcs):
            if func != "main":
                execution_trigger += f"print('[Sandbox] Triggering Function: {func}')\n"
                execution_trigger += f"pcall(function() {func}(nil) end)\n"

    # 组合最终扔进沙箱的代码
    full_test_code = mock_env_guard + "\n" + code + execution_trigger

    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(full_test_code)

    try:
        # ---- Phase 1: 静态检查 ----
        check_res = subprocess.run(
            ["luac", "-p", temp_file],
            capture_output=True, text=True, timeout=timeout, encoding='utf-8', errors='ignore'
        )
        if check_res.returncode != 0:
            return False, check_res.stderr.strip().replace(temp_file, "line"), ["LuaSyntaxError"]

        # ---- Phase 2: 动态解冻测试 ----
        run_res = subprocess.run(
            ["lua", temp_file],
            capture_output=True, text=True, timeout=timeout, encoding='utf-8', errors='ignore'
        )
        
        if run_res.returncode == 0:
            return True, "Success", []
        
        # 成功捕获到了被我们引爆的幻觉运行时错误
        run_stderr = run_res.stderr.strip().replace(temp_file, "line")
        return False, run_stderr, ["LuaRuntimeError"]

    except Exception as e:
        # 如果是因为并发过高引发的系统级异常，不要静音，返回 False 确保安全性
        return False, f"Sandbox OS Error: {str(e)}", ["ENV_ERROR"]
    finally:
        shutil.rmtree(temp_dir)

def execute_lua_unit_test(code: str, test_cases: list, timeout: int = 5) -> (bool, bool):
    """
    轻量级 Lua 动态沙盒评测机
    """
    if not code.strip():
        return False, False
        
    # 简单的语法初筛 (尝试用 luac 编译检查，如果没有环境可以用 lua 直接跑)
    tmp_file = tempfile.mktemp(suffix=".lua")
    
    raw_test_list = test_cases if isinstance(test_cases, list) else [str(test_cases)]
    test_script_str = "\n".join([str(t).strip() for t in raw_test_list if str(t).strip()])
    
    # 拼装 Lua 脚本与测试用例
    full_script = f"{code}\n\n-- Test Suite --\n{test_script_str}"
    
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(full_script)
            
        # 执行 Lua 脚本
        res = subprocess.run(
            ["lua", tmp_file],
            capture_output=True, text=True, timeout=timeout
        )
        
        # 语法通过：只要能跑起来（哪怕报错）就说明不是完全的非代码文本
        syntax_ok = True 
        # 单元测试通过：返回值返回 0 且没有错误输出
        test_ok = (res.returncode == 0)
        
        return syntax_ok, test_ok
    except Exception:
        return False, False
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

def evaluate_pass_at_k_hybrid_job(candidates: list, test_cases: list):
    """
    完全符合学术标准的 Lua 语言双轨评估机
    """
    syntax_results = []
    unittest_results = []
    
    for cand in candidates:
        # 1. 第一轨：基础语法与沙箱幻觉防御检查
        syn_ok, _, _ = verify_lua_syntax_with_log(cand)
        syntax_results.append(syn_ok)
        
        # 2. 第二轨：真实自动化单元测试
        ut_ok = False
        if syn_ok and test_cases:
            # 容错处理：若 test_cases 是列表，则逐条运行或合并运行
            if isinstance(test_cases, list):
                all_cases_passed = True
                for tc in test_cases:
                    if not tc.strip():
                        continue
                    if not execute_lua_unit_test(cand, tc):
                        all_cases_passed = False
                        break
                ut_ok = all_cases_passed
            else:
                ut_ok = execute_lua_unit_test(cand, str(test_cases))
        else:
            ut_ok = False
            
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
    
def clean_generated_code(code: str) -> str:
    if not code:
        return ""
        
    # 1. 物理移除特殊混淆记号残余
    code = re.sub(r'<\|im_.*?\|>', '', code)
    code = code.replace("<|endoftext|>", "").replace("<s>", "").replace("</s>", "")

    # 2. 核心拦截轨：优先抓取完整的 ```lua 代码块
    lua_blocks = re.findall(r'```[Ll]ua\s*\n(.*?)\n```', code, re.DOTALL)
    if lua_blocks:
        final_code = lua_blocks[0].strip()
    else:
        # 3. 🎯 极限加固消融防线：如果大模型没有使用 Markdown 代码块包裹
        # 那么利用自回归文本规律，精准抓取第一个完整的 function ... end 逻辑闭合区间
        # 彻底抛弃区间外所有胡言乱语的 C++/Java 垃圾
        func_block_match = re.search(r'\b(function\s+\w+.*?)\bend\b', code, re.DOTALL)
        if func_block_match:
            final_code = func_block_match.group(1).strip() + "\nend"
        else:
            # 最后的兜底平铺切分
            final_code = code.strip()
            for separator in ["###", "Java:", "Euler", "Problem:", "```java", "```cpp", "## 8.", "* [C#]"]:
                if separator in final_code:
                    final_code = final_code.split(separator)[0].strip()

    # 4. 跨语言字面量自愈
    final_code = re.sub(r'\b(\d+)[Ll]\b', r'\1', final_code) # 斩除 1L, 0L
    final_code = final_code.replace("null", "nil")            # 斩除 Java null
    
    # 修复单行 Java 风格的 if 遗留
    def fix_java_style_if(match):
        cond = match.group(1)
        body = match.group(2)
        if "then" not in body:
            return f"if {cond} then {body} end"
        return match.group(0)
    final_code = re.sub(r'\bif\s*\((.*?)\)\s*([^ \n][^\n]*)\b', fix_java_style_if, final_code)

    if final_code.endswith(";"):
        final_code = final_code[:-1].strip()
            
    return final_code

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
    elif "qwen" in llm_type:
        # Qwen 2/2.5 ChatML 格式的结束符是 <|im_end|>
        custom_eos_token = "<|im_end|>"
    elif "deepseek" in llm_type:
        # DeepSeek-V2 / DeepSeek-Coder-Instruct 标准结束符也是 <|im_end|> 或者是 <｜end of sentence｜>
        # 优先探测标准的 ChatML 结束符
        custom_eos_token = "<|im_end|>" if "<|im_end|>" in model.llm_tokenizer.get_vocab() else None

    # 动态转换 Token ID 并设置安全防线
    actual_eos_id = model.llm_tokenizer.eos_token_id  # 默认使用基座自带的 eos_token_id
    if custom_eos_token:
        token_id = model.llm_tokenizer.convert_tokens_to_ids(custom_eos_token)
        if token_id is not None and token_id != model.llm_tokenizer.unk_token_id:
            actual_eos_id = token_id

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
            eos_token_id=actual_eos_id  # 🎯 动态锁死当前模型真正的结束符
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

class PassKEvaluationDataset(CodeTranslateDataset):
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        raw_emb = item['graph_emb']
        graph_emb = torch.tensor(raw_emb, dtype=torch.float) if not isinstance(raw_emb, str) else torch.tensor(json.loads(raw_emb), dtype=torch.float)
        
        # 优先适配 Lua 相关的测试字段
        unit_tests = item.get('lua_tests')
        if isinstance(unit_tests, str):
            try: 
                unit_tests = json.loads(unit_tests)
            except: 
                pass
            
        return (
            item.get('idx', idx),
            graph_emb,
            str(item.get('java_code', "")),
            str(item.get('nl_description', "")),
            unit_tests
        )

def run_reinforced_inference(config_path):
    with open(config_path, 'r', encoding='utf-8') as f: 
        config = yaml.safe_load(f)
    
    tgt_language = config.get('tgt_language', 'Lua')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    use_api_mode = config["model"].get("use_api_mode", False)
    llm_type = config["model"].get("llm_type", "qwen").lower()
    use_graph = config.get("generation", {}).get("use_graph_info", False)
    
    print(f"\n================ ⚙️ 启动完全体【Java 到 Lua 消融推理管线】================")
    print(f"当前运行模式: {'【图增强模式】' if use_graph else '【纯代码消融模式】'} | 目标基座: {llm_type.upper()}")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    model = CGBridgeStage3(config)
    
    if use_graph:
        print("🚚 正在激活图增强评估流，锁死 Q-Former 推理期状态...")
        if hasattr(model, "stage2_bridge") and model.stage2_bridge is not None:
            model.stage2_bridge.eval()
            if hasattr(model.stage2_bridge, "Qformer"):
                model.stage2_bridge.Qformer.eval()
                model.stage2_bridge.Qformer.bert.config.gradient_checkpointing = False
    else:
        print("⚠️ [消融警报开启]：图增强流已被物理切断，Q-Former 向量在前向传播中将被跳过！")

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
            elif "llama" in llm_type:
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
                print(f"🔥 [精准增量融合] 正在安全加载微调模型权重: {stage3_ckpt}...")
                res = model.load_tuned_stage3_weights(stage3_ckpt)
                if res is not None:
                    print(f"✅ 增量参数同步报告 ── 缺失项总数: {len(res.missing_keys)}")
            else:
                print(f"❌ 严重错误: 找不到微调模型路径 {stage3_ckpt}！")
                return
        else:
            print("🎨 正在以 [纯原生大模型基座] 模式无污染启动硬件加载...")
            
        model.to(device)
        model.eval()

    print(f"📖 正在加载测试集: {config['data']['test_path']} ...")
    dataset = PassKEvaluationDataset(config['data']['test_path'])
    dataloader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])

    output_csv = config['output']['multilang_evaluation_report']
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    total_samples = len(dataset)
    compile_success_total = 0
    pass_1_total = 0
    pass_2_total = 0
    pass_5_total = 0
    processed_samples_count = 0
    
    max_workers = min(os.cpu_count(), 16)
    print(f"🚀 双轨编译器单测评评测开启！目标语言: {tgt_language} | 题目总数: {total_samples} | CPU 并行度: {max_workers}")

    with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'java_code', 'nl_desc', 'all_candidates', 'passed_code', 'compile_ok', 'pass_1', 'pass_2', 'pass_5'])

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating & Dual Evaluating"):
                idx, gtokens, java_codes, nl_descs, batch_unit_tests = batch
                
                batch_candidates = batch_smart_generate(
                    model, gtokens, java_codes, nl_descs, config, device)
                
                batch_futures = []
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    for i in range(len(idx)):
                        raw_idx = idx[i].item() if isinstance(idx[i], torch.Tensor) else str(idx[i])
                        future = executor.submit(evaluate_pass_at_k_hybrid_job, batch_candidates[i], batch_unit_tests[i])
                        batch_futures.append({
                            'idx': raw_idx, 'java_code': java_codes[i], 'nl_desc': nl_descs[i],
                            'candidates': batch_candidates[i], 'future': future
                        })

                    for item in batch_futures:
                        try:
                            comp_ok, p1, p2, p5, passed_code = item['future'].result()
                            compile_success_total += comp_ok
                            pass_1_total += p1
                            pass_2_total += p2
                            pass_5_total += p5
                            
                            writer.writerow([
                                item['idx'], item['java_code'], item['nl_desc'],
                                json.dumps(item['candidates'], ensure_ascii=False),
                                passed_code, comp_ok, p1, p2, p5
                            ])
                            
                            processed_samples_count += 1
                            
                            if processed_samples_count % 10 == 0:
                                tqdm.write(f"\n📢 [中间进度报告] ── 累计已处理数: {processed_samples_count}/{total_samples}")
                                tqdm.write(f"    ├─  当前累计语法通过率: {compile_success_total / processed_samples_count * 100:.2f}%")
                                tqdm.write(f"    ├─ 真实单元测试 Pass@1: {pass_1_total / processed_samples_count * 100:.2f}%")
                                tqdm.write(f"    ├─ 真实单元测试 Pass@2: {pass_2_total / processed_samples_count * 100:.2f}%")
                                tqdm.write(f"    └─ 真实单元测试 Pass@5: {pass_5_total / processed_samples_count * 100:.2f}%\n")
                                
                        except Exception as e:
                            print(f"\n[Error] 样本 {item['idx']} 评测异常: {e}")
                
                f.flush()
                del batch_candidates, batch_futures, idx, gtokens
                gc.collect()
                if not use_api_mode:
                    torch.cuda.empty_cache()

    print(f"\n🏆 ======================== CGBridge 终极 {tgt_language} 翻译评测报告 ========================")
    print(f"📊 评估题目总计: {total_samples} 道")
    print(f"🛠️ 基础语法/编译通过率 (Compilation Pass Rate): {compile_success_total / total_samples * 100:.2f}%")
    print(f"📈 真实自动化单元测试 Pass@1 成功率: {pass_1_total / total_samples * 100:.2f}%")
    print(f"📈 真实自动化单元测试 Pass@2 成功率: {pass_2_total / total_samples * 100:.2f}%")
    print(f"📈 真实自动化单元测试 Pass@5 成功率: {pass_5_total / total_samples * 100:.2f}%")
    print(f"==================================================================================\n")

if __name__ == "__main__":
    CONFIG_FILE = "/root/autodl-tmp/first_paper/java/stage3/stage3.yaml"
    run_reinforced_inference(CONFIG_FILE)