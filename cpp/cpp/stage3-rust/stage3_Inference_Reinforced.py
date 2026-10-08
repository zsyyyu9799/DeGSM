import os
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "FALSE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import pandas as pd
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
cpp_project_dir = "/root/autodl-tmp/first_paper/cpp"
if cpp_project_dir not in sys.path:
    sys.path.insert(0, cpp_project_dir)

from stage3.CGBridge_Stage3 import CGBridgeStage3, CodeTranslateDataset


def create_cargo_project(temp_dir, code):
    """
    为生成的 Rust 代码动态构建标准的虚拟 Cargo 编译工程环境
    """
    # 1. 检查代码中是否包含底层或无标准库操作，自动配置 no_std
    is_no_std = "core::" in code or "#![no_std]" in code

    # 2. 写入 Cargo.toml
    cargo_toml = os.path.join(temp_dir, "Cargo.toml")
    with open(cargo_toml, 'w', encoding='utf-8') as f:
        f.write('[package]\nname = "verify_pkg"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\n')
        # 如果您的 C++ 翻译涉及特定的高频库（如 bitflags 或 libc），可在此处静默注入

    # 3. 创建 src 目录并写入 lib.rs（作为库编译，避免无 main 函数报错）
    src_dir = os.path.join(temp_dir, "src")
    os.makedirs(src_dir, exist_ok=True)
    lib_rs = os.path.join(src_dir, "lib.rs")
    
    with open(lib_rs, 'w', encoding='utf-8') as f:
        if is_no_std and "#![no_std]" not in code:
            f.write("#![no_std]\n")
        f.write(code)
        
def is_logic_valid(code: str) -> (bool, str):
    """
    防御性规则拦截器：防止大模型将 C++ 的独有指针或老旧宏语法直接带入 Rust 中
    """
    if not code:
        return False, "FIX: Code is empty."

    # 1. 基础占位符检查
    if "unimplemented!" in code or "todo!" in code:
        return False, "FIX: Do not use error placeholders. Please translate the actual logic from C++."
    
    # 2. 空实现拦截
    if re.search(r'fn\s+\w+\s*\(.*?\)\s*->\s*.*?\s*\{\s*\}', code.strip()):
         return False, "FIX: Empty function body detected. Provide the full Rust implementation."

    # 3. 面向 C++ 到 Rust 的泄漏防御机制
    if "std::" in code and "std::vec" not in code and "std::collections" not in code:
        if re.search(r'\b(std::cout|std::cin|endl|std::string)\b', code):
            return False, "FIX: C++ syntax leakage detected (e.g., std::cout, std::string). Use Rust equivalent."

    if re.search(r'\b(include|using namespace|#define)\b', code):
        return False, "FIX: C++ preprocessor macros or headers detected. Use Rust 'use' syntax."

    if re.search(r'\b(class|public:|private:)\b', code):
        return False, "FIX: C++ OOP class syntax leakage detected. Use Rust 'struct' and 'impl' blocks."

    if re.search(r'\b(nullptr|NULL)\b', code):
        return False, "FIX: C++ pointer leakage detected. Use Rust 'Option::None' or safe references."

    return True, ""
    
def calculate_pass_at_k(n, c, k):
    """
    计算标准的学术级 Pass@k 概率
    """
    if c - n < k:
        return 1.0
    return 1.0 - math.comb(c - n, k) / math.comb(c, k)

def verify_rust_syntax_with_log(code: str, timeout: int = 20):
    """使用 JSON 格式精准解析编译器反馈"""
    base_temp_path = "/dev/shm" if os.path.exists("/dev/shm") else None
    temp_dir = tempfile.mkdtemp(prefix="rust_verify_", dir=base_temp_path)
    src_dir = os.path.join(temp_dir, "src")
    os.makedirs(src_dir)

    main_rs = os.path.join(src_dir, "lib.rs")  # 以库形式验证
    with open(os.path.join(src_dir, "lib.rs"), 'w', encoding='utf-8') as f:
        f.write(code)
    with open(os.path.join(temp_dir, "Cargo.toml"), 'w', encoding='utf-8') as f:
        f.write('[package]\nname = "verify_pkg"\nversion = "0.1.0"\nedition = "2021"\n')

    try:
        check_res = subprocess.run(
            ["/root/.cargo/bin/cargo", "check", "--message-format=json", "--offline"], 
            cwd=temp_dir, capture_output=True, text=True, timeout=timeout, encoding='utf-8', errors='ignore'
        )

        if check_res.returncode == 0:
            return True, "Success", []

        errors = []
        error_codes = []
        # 用于去重的计数器：key 是错误码，value 是出现的次数
        error_counts = {}

        for line in check_res.stdout.splitlines():
            try:
                msg = json.loads(line)
                if msg.get("reason") == "compiler-message":
                    compiler_msg = msg.get("message", {})
                    
                    # 策略 1：只保留 level 为 error 的消息，跳过 warning
                    if compiler_msg.get("level") != "error":
                        continue

                    # 提取错误代码 (如 E0308)，若无错误码则标记为 SyntaxError
                    code_info = compiler_msg.get("code")
                    err_code = code_info.get("code") if code_info else "SyntaxError"
                    
                    # --- 策略 2：去重逻辑 ---
                    # 如果同一种错误已经出现了 2 次，则跳过后续相同的错误
                    current_count = error_counts.get(err_code, 0)
                    if current_count >= 2:
                        continue
                    
                    # 提取纯文本描述 (rendered)
                    rendered = compiler_msg.get("rendered", "")
                    if rendered:
                        # 清理 ANSI 颜色转义符
                        clean_rendered = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', rendered)
                        
                        errors.append(f"[{err_code}] {clean_rendered.strip()}")
                        error_codes.append(err_code)
                        # 更新该错误码的计数
                        error_counts[err_code] = current_count + 1

            except json.JSONDecodeError:
                continue

        # 兜底：如果 JSON 没解析出 error 但进程确实报错了
        if not errors:
            return False, f"Compiler execution failed:\n{check_res.stderr[-500:]}", ["E_UNKNOWN"]

        # 限制返回的总错误条数，避免 Prompt 过长（通常 3-5 条 error 足够了）
        return False, "\n---\n".join(errors[:4]), error_codes

    finally:
        shutil.rmtree(temp_dir)

def execute_rust_unit_test(code: str, test_cases: list, timeout: int = 15) -> (bool, bool):
    if not code.strip():
        return False, False

    base_dir = "/root/autodl-tmp/rust_eval_sandboxes"
    os.makedirs(base_dir, exist_ok=True)
    tmp_project_dir = tempfile.mkdtemp(prefix="rust_eval_", dir=base_dir)
    
    try:
        cargo_toml_content = """[package]
name = "rust_eval_sandbox"
version = "0.1.0"
edition = "2021"

[dependencies]
"""
        with open(os.path.join(tmp_project_dir, "Cargo.toml"), "w", encoding="utf-8") as f:
            f.write(cargo_toml_content)

        src_dir = os.path.join(tmp_project_dir, "src")
        os.makedirs(src_dir, exist_ok=True)
        lib_rs_path = os.path.join(src_dir, "lib.rs")

        # ================= 🚀 第一步：纯净代码语法检查 =================
        # 只写入大模型生成的代码，绝不带任何外来测试用例
        with open(lib_rs_path, "w", encoding="utf-8") as f:
            f.write(code)

        compile_res = subprocess.run(
            ["/root/.cargo/bin/cargo", "check", "--offline"],
            cwd=tmp_project_dir,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout
        )

        if compile_res.returncode != 0:
            # 如果连纯代码本身都存在语法错误，才判决 False
            return False, False

        # ================= 🚀 第二步：注入测试用例并执行 =================
        func_match = re.search(r'pub\s+fn\s+([a-zA-Z0-9_]+)', code)
        if not func_match:
            func_match = re.search(r'fn\s+([a-zA-Z0-9_]+)', code)
        model_fn = func_match.group(1) if func_match else None

        source_lines = [code, "\n// ==================== TESTS ====================\n"]
        raw_test_list = test_cases if isinstance(test_cases, list) else [str(test_cases)]
        
        for idx, t in enumerate(raw_test_list):
            t_str = str(t).strip()
            if t_str.startswith("}"):
                t_str = t_str.lstrip("}").strip()
            if not t_str: continue
            
            if model_fn and model_fn != "candidate":
                alias_line = f"    use super::{model_fn} as candidate;"
            else:
                alias_line = "    use super::*;"

            if "fn main()" in t_str:
                t_str = re.sub(r'let\s+candidate\s*=\s*[a-zA-Z0-9_]+;', '', t_str)
                source_lines.append("#[cfg(test)]")
                source_lines.append(f"mod academic_test_space_{idx} {{")
                source_lines.append(alias_line)
                source_lines.append("    use super::*;")
                for line in t_str.splitlines():
                    source_lines.append(f"    {line}")
                source_lines.append("    #[test]")
                source_lines.append("    fn run_sandbox_test() {")
                source_lines.append("        main();")
                source_lines.append("    }")
                source_lines.append("}")
            else:
                source_lines.append("#[cfg(test)]")
                source_lines.append(f"mod academic_test_fragment_{idx} {{")
                source_lines.append("    use super::*;")
                source_lines.append("    #[test]")
                source_lines.append(f"    fn test_fragment_case() {{")
                if model_fn and model_fn != "candidate" and "candidate" in t_str:
                    source_lines.append(f"        let candidate = super::{model_fn};")
                source_lines.append(f"        {t_str if t_str.endswith(';') else t_str + ';'}")
                source_lines.append("    }")
                source_lines.append("}")

        # 将包含单测的代码完整回写
        with open(lib_rs_path, "w", encoding="utf-8") as f:
            f.write("\n".join(source_lines))

        # 执行单测
        test_res = subprocess.run(
            ["/root/.cargo/bin/cargo", "test", "--offline", "--", "--nocapture"],
            cwd=tmp_project_dir,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout
        )
        
        # 此时已经确认语法通过 (True)，只看测试结果
        return True, (test_res.returncode == 0)

    except Exception:
        return False, False
    finally:
        if os.path.exists(tmp_project_dir):
            shutil.rmtree(tmp_project_dir)
            
def evaluate_pass_at_k_hybrid_job(candidates: list, test_cases: list):
    """
    完全符合学术标准的双轨评估机
    """
    syntax_results = []
    unittest_results = []
    
    for cand in candidates:
        syn_ok, ut_ok = execute_rust_unit_test(cand, test_cases)
        syntax_results.append(syn_ok)
        unittest_results.append(ut_ok)
        
    compile_pass = 1 if any(syntax_results) else 0
    
    # 1. 统计这道题总共生成了多少个候选，其中有多少个完美通过了单测
    total_cands = len(candidates)              # 比如 5
    passed_cands = sum(1 for x in unittest_results if x) # 比如 1 或 2
    
    # 2. 严格调用学术期望公式计算概率
    p1 = calculate_pass_at_k(passed_cands, total_cands, k=1)
    p2 = calculate_pass_at_k(passed_cands, total_cands, k=2)
    p5 = calculate_pass_at_k(passed_cands, total_cands, k=5)
    
    # 挑出一个通过的代码作为存盘参考
    passed_code = ""
    for idx, ut_passed in enumerate(unittest_results):
        if ut_passed:
            passed_code = candidates[idx]
            break
            
    return compile_pass, p1, p2, p5, passed_code
    
def clean_generated_code(code: str, is_repair_mode: bool = False) -> str:
    if not code:
        return ""

    # 1. 优先尝试提取成对的代码块
    rust_blocks = re.findall(r'```(?:rust)?\s*\n?(.*?)\n?```', code, re.DOTALL | re.IGNORECASE)
    final_code = rust_blocks[-1] if rust_blocks else code

    # 2. 清理大模型特殊 Token
    final_code = re.sub(r'<\|.*?\|>', '', final_code)
    
    # 3. 拦截多余的测试函数
    final_code = re.split(r'\n\s*fn\s+main\s*\(\s*\)', final_code)[0]
    final_code = re.split(r'\n\s*#\s*\[\s*cfg\(\s*test\s*\)\s*\]', final_code)[0]
    final_code = re.split(r'\n\s*#\s*\[\s*test\s*\]', final_code)[0]
    final_code = re.split(r'//\s*[Tt]est cases', final_code)[0]

    # 4. 🚀 致命修复：清除因为 Prefill 遗留的单边反引号，防止污染 lib.rs 源码
    final_code = final_code.replace("```rust", "").replace("```", "")
    
    return final_code.strip()

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
    
    # 🎯【核心修复 1】：精准收拢各种模型的 EOS / Stop Tokens，彻底防止吐字越界
    actual_eos_ids = [model.llm_tokenizer.eos_token_id]

    # 针对 Llama 架构，增加专属系统的 eot_id
    if "llama" in llm_type:
        eot_id = model.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != model.llm_tokenizer.unk_token_id:
            actual_eos_ids.append(eot_id)
    
    # 仅保留绝不可能在代码开头出现的结构性终止符，绝不能放 "```" !
    stop_words = ["<|endoftext|>", "<|im_end|>", "<|global_end|>"]
    for word in stop_words:
        word_ids = model.llm_tokenizer.encode(word, add_special_tokens=False)
        if word_ids and word_ids[0] not in actual_eos_ids:
            actual_eos_ids.append(word_ids[0])

    actual_eos_ids = list(set(actual_eos_ids))
    current_rep_penalty = config['generation'].get('repetition_penalty', 1.1)

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

class PassKEvaluationDataset(CodeTranslateDataset):
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        raw_emb = item['graph_emb']
        graph_emb = torch.tensor(raw_emb, dtype=torch.float) if not isinstance(raw_emb, str) else torch.tensor(json.loads(raw_emb), dtype=torch.float)
        
        # 1. 稳健检索测试字段，避开 pandas 的 NaN 陷阱
        unit_tests = None
        for key in ['rust_tests', 'ocaml_tests', 'test', 'tests']:
            if key in item and pd.notna(item[key]):
                unit_tests = item[key]
                break
        
        if unit_tests is None:
            unit_tests = []
            
        if isinstance(unit_tests, str):
            try: 
                unit_tests = json.loads(unit_tests)
            except: 
                unit_tests = [unit_tests]
                
        # 2. 强转为 JSON 字符串，防止 PyTorch 把其转化为 Tensor(NaN)
        return (
            item.get('idx', idx),
            graph_emb,
            str(item.get('cpp_code', "")),
            str(item.get('nl_description', "")),
            json.dumps(unit_tests)
        )

def run_reinforced_inference(config_path):
    with open(config_path, 'r', encoding='utf-8') as f: 
        config = yaml.safe_load(f)
    
    tgt_language = config.get('tgt_language', 'Rust')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    use_api_mode = config["model"].get("use_api_mode", False)
    llm_type = config["model"].get("llm_type", "qwen").lower()
    use_graph = config.get("generation", {}).get("use_graph_info", False)
    
    print(f"\n================ ⚙️ 启动完全体【Cpp 到 Rust 消融推理管线】================")
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
                target_mods = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "starcoder" in llm_type:
                print("💡 路由选择：检测到 StarCoder 家族，挂载其特异性注意力投影层...")
                target_mods = ["c_attn", "c_proj", "c_fc", "q_proj", "kv_proj", "o_proj"]
            elif "qwen" in llm_type or "llama" in llm_type:
                print("💡 路由选择：检测到标准 MHA 架构 (Qwen/Llama)，挂载常规注意力靶向层...")
                target_mods = ["q_proj", "v_proj"]
            else:
                # 兜底通用标准层
                target_mods = ["q_proj", "v_proj", "o_proj"]
                
            peft_config = LoraConfig(
                r=16 if "starcoder" in llm_type else 8,                           
                lora_alpha=32 if "starcoder" in llm_type else 16,                 
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
        writer.writerow(['idx', 'cpp_code', 'nl_desc', 'all_candidates', 'passed_code', 'compile_ok', 'pass_1', 'pass_2', 'pass_5'])

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating & Dual Evaluating"):
                idx, gtokens, cpp_codes, nl_descs, batch_unit_tests = batch
                
                # 💡 新增这一行：将序列化的 JSON 重新转回 list
                batch_unit_tests = [json.loads(t) for t in batch_unit_tests]
                
                batch_candidates = batch_smart_generate(
                    model, gtokens, cpp_codes, nl_descs, config, device)
                
                batch_futures = []
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    for i in range(len(idx)):
                        raw_idx = idx[i].item() if isinstance(idx[i], torch.Tensor) else str(idx[i])
                        future = executor.submit(evaluate_pass_at_k_hybrid_job, batch_candidates[i], batch_unit_tests[i])
                        batch_futures.append({
                            'idx': raw_idx, 'cpp_code': cpp_codes[i], 'nl_desc': nl_descs[i],
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
                                item['idx'], item['cpp_code'], item['nl_desc'],
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
    CONFIG_FILE = "/root/autodl-tmp/first_paper/cpp/stage3/stage3.yaml"
    run_reinforced_inference(CONFIG_FILE)