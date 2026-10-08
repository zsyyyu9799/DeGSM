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
from stage3_Inference import verify_lua_syntax_with_log

def is_logic_valid(code: str) -> (bool, str):
    # 基类拦截
    if "error(" in code and "unimplemented" in code:
        return False, "FIX: Do not use error placeholders. Please translate the actual logic from C++."
    if re.search(r'function\s+\w+\s*\(.*?\)\s*end', code.strip()):
        return False, "FIX: Empty function body detected. Provide the full Lua implementation."

    # C++ 语法泄漏防御机制
    if "cout" in code or "std::" in code:
        return False, "FIX: C++ namespace/stdout leakage detected. Use `print()` or `io.write()` in Lua."
    if re.search(r'\b(std::vector|std::map|std::set|std::string|std::pair)\b', code):
        return False, "FIX: C++ Standard Library containers leakage detected. Use native Lua tables."
    if re.search(r'\b(include|using namespace|#define|#ifdef)\b', code):
        return False, "FIX: C++ preprocessor macros or headers detected. Write pure Lua syntax."
    if re.search(r'\b(class|struct|public:|private:|protected:)\b', code):
        return False, "FIX: C++ OOP object/class syntax leakage detected. Use Lua tables or metatables."
    if re.search(r'\b(nullptr|NULL)\b', code):
        return False, "FIX: C++ pointer leakage detected. Use Lua `nil`."

    # 🎯 新增：拦截 Lua 不支持的自增自减操作符（防止逃过静态检查引发诡异报错）
    if re.search(r'\w+\+\+', code) or re.search(r'\+\+\w+', code) or re.search(r'\w+--', code):
        return False, "FIX: Lua does not support '++' or '--' operators. Use 'i = i + 1' explicitly."

    # 防御历史遗留残留
    if re.search(r'\b(let rec|match|with|fn|impl|->)\b', code):
        return False, "FIX: OCaml/Rust syntax leakage detected. Use Lua `local function` or standard loop syntax."

    return True, ""

def execute_lua_unit_test_reinforced(code: str, test_cases: list, timeout: int = 5) -> (bool, bool):
    if not code.strip():
        return False, False

    # 1. 语法检查与清理
    code = re.sub(r'\blocal\s+function\b', 'function', code)
    code = re.sub(r'\blocal\s+([a-zA-Z0-9_]+)\s*=\s*function', r'\1 = function', code)

    is_syntax_ok, _, _ = verify_lua_syntax_with_log(code)
    is_logic_clean, _ = is_logic_valid(code)
    final_syntax_compile_ok = is_syntax_ok and is_logic_clean

    if not final_syntax_compile_ok:
        return False, False 

    # 2. 提取模型生成的函数名
    code_no_comments = re.sub(r'--.*', '', code)
    func_matches = re.findall(r'(?:function\s+([a-zA-Z0-9_]+)\s*\()|(?:([a-zA-Z0-9_]+)\s*=\s*function\s*\()', code_no_comments)
    func_names = [m[0] or m[1] for m in func_matches if m[0] or m[1]]
    model_func_name = func_names[-1] if func_names else ""

    # 3. 提取并清理测试用例源码
    raw_test_list = test_cases if isinstance(test_cases, list) else [str(test_cases)]
    test_script_str = "\n".join([str(t) for t in raw_test_list if str(t).strip()])

    # 4. 🔥 构造针对 LuaUnit 测试用例的零碰撞沙箱环境
    FULL_LUA_SANDBOX = f"""
-- 1. 深度对比与标准 Assert 垫片
if not table.unpack then table.unpack = unpack or _G.unpack end
if not unpack then unpack = table.unpack end

local function deep_compare(t1, t2)
    if type(t1) ~= type(t2) then return false end
    if type(t1) ~= 'table' then 
        if type(t1) == 'number' and type(t2) == 'number' then
            return math.abs(t1 - t2) < 1e-4
        end
        return t1 == t2 
    end
    for k, v in pairs(t1) do if not deep_compare(v, t2[k]) then return false end end
    for k, v in pairs(t2) do if t1[k] == nil then return false end end
    return true
end

-- 2. Mock 真正的 LuaUnit 框架 (杜绝依赖缺失)
package.loaded['luaunit'] = {{}}
local lu = package.loaded['luaunit']
lu.assertEquals = function(a, b)
    if not deep_compare(a, b) then
        print("LUAUNIT_ASSERT_FAILED")
        os.exit(1)
    end
end
lu.assertTrue = function(v) if not v then print("LUAUNIT_ASSERT_FAILED"); os.exit(1) end end
lu.assertFalse = function(v) if v then print("LUAUNIT_ASSERT_FAILED"); os.exit(1) end end
lu.assertIsNil = function(v) if v ~= nil then print("LUAUNIT_ASSERT_FAILED"); os.exit(1) end end
lu.LuaUnit = {{ 
    run = function() return 0 end 
}}

-- 3. 拦截 os.exit，替换为安全退出
local original_os_exit = os.exit
os.exit = function(code)
    if code == 0 or code == nil then
        print("LUAUNIT_SUCCESS")
        original_os_exit(0)
    else
        print("LUAUNIT_FAILED")
        original_os_exit(1)
    end
end

-- 4. 载入模型代码
{code}

-- 5. 🔥 全局变量别名智能路由 (防止测试用例里的 candidate = check_integer 找不到函数)
local target_func_name = "{model_func_name}"
if target_func_name ~= "" and _G[target_func_name] then
    local real_func = _G[target_func_name]
    setmetatable(_G, {{
        __index = function(t, k)
            -- 无论是 check_integer, candidate, 还是大小写差异，只要找不到就自动路由到模型生成的函数
            return real_func
        end
    }})
end

-- 6. 执行测试脚本
{test_script_str}

-- 7. 兜底驱动所有的 test_ 函数
for k, v in pairs(_G) do
    if type(v) == 'function' and type(k) == 'string' and k:match('^test_') then
        local ok, err = pcall(v)
        if not ok then 
            print("LUAUNIT_EXEC_ERROR: " .. tostring(err))
            original_os_exit(1) 
        end
    end
end
print("LUAUNIT_SUCCESS")
"""

    base_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
    with tempfile.NamedTemporaryFile(suffix=".lua", dir=base_dir, mode="w", encoding="utf-8", delete=False) as tf:
        tf.write(FULL_LUA_SANDBOX)
        temp_file_path = tf.name

    try:
        res = subprocess.run(
            ["lua", temp_file_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout, encoding='utf-8', errors='ignore'
        )
        output = res.stdout + "\n" + res.stderr
        
        # 只要没有判定失败，且输出了 LUAUNIT_SUCCESS，说明单元测试 100% 跑通！
        is_unit_test_ok = (res.returncode == 0 and "LUAUNIT_ASSERT_FAILED" not in output and "LUAUNIT_EXEC_ERROR" not in output)
        return final_syntax_compile_ok, is_unit_test_ok
    except Exception:
        return final_syntax_compile_ok, False
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def verify_lua_syntax_with_log(code: str, timeout: int = 5):
    """
    使用官方 luac -p (Parse only) 严格验证 Lua 语法合法性。
    """
    base_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
    with tempfile.NamedTemporaryFile(suffix=".lua", dir=base_dir, mode="w", encoding="utf-8", delete=False) as tf:
        tf.write(code)
        temp_file_path = tf.name

    try:
        check_res = subprocess.run(
            ["luac", "-p", temp_file_path],
            capture_output=True, text=True, timeout=timeout, encoding='utf-8', errors='ignore'
        )

        if check_res.returncode == 0:
            return True, "Success", []

        stderr = check_res.stderr.strip()
        clean_error = stderr.replace(temp_file_path, "line")
        return False, clean_error, ["LuaSyntaxError"]
    except Exception as e:
        return False, f"Execution failed: {str(e)}", ["EXEC_ERROR"]
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def clean_generated_code(code: str, is_repair_mode: bool = False) -> str:
    if not code:
        return ""

    # 1. 提取 ```lua ... ``` 代码块中的核心部分
    lua_blocks = re.findall(r'```lua\s*\n?(.*?)\n?```', code, re.DOTALL)
    if not lua_blocks:
        lua_blocks = re.findall(r'```\s*\n?(.*?)\n?```', code, re.DOTALL)
    final_code = lua_blocks[-1] if lua_blocks else code

    final_code = re.sub(r'<\|im_.*?\|>', '', final_code)
    
    # 🎯 核心修复 1：把 'local function xxx' 暴力替换为全局 'function xxx'，确保单元测试能在沙箱中直接调用！
    final_code = re.sub(r'\blocal\s+function\s+', 'function ', final_code)
    final_code = re.sub(r'\blocal\s+([a-zA-Z0-9_]+)\s*=\s*function', r'\1 = function', final_code)

    # 🎯 核心修复 2：修复 Lua 字符串下标越界幻觉 s[i] -> string.sub(s, i, i)
    # (解决 Lua 无法直接用 str[i] 获取字符的问题)

    final_code = final_code.strip()
    if final_code.endswith(";"):
        final_code = final_code[:-1].strip()
            
    return final_code

def calculate_pass_at_k(n, c, k):
    if c - n < k:
        return 1.0
    return 1.0 - math.comb(c - n, k) / math.comb(c, k)
    
def evaluate_pass_at_k_hybrid_job(candidates: list, test_cases: list):
    syntax_results = []
    unittest_results = []
    
    for cand in candidates:
        syn_ok, ut_ok = execute_lua_unit_test_reinforced(cand, test_cases)
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

class PassKEvaluationDataset(CodeTranslateDataset):
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        
        raw_emb = item.get('graph_emb')
        graph_emb = torch.tensor(raw_emb, dtype=torch.float) if not isinstance(raw_emb, str) else torch.tensor(json.loads(raw_emb), dtype=torch.float)
        
        # 1. 提取测试用例
        unit_tests = item.get('tests_lua') or item.get('tests') or item.get('test_cases')
        
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
        nl_desc_str = str(item.get('lua_prompt', item.get('nl_description', "")))
            
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
    dataset = PassKEvaluationDataset(config['data']['test_path'])
    
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
                        future = executor.submit(evaluate_pass_at_k_hybrid_job, batch_candidates[i], batch_unit_tests[i])
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