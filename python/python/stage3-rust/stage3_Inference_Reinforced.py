import os
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "FALSE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import shutil
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
import math
# 🎯 强力定位真实的 python 核心项目目录，防止导包死锁
python_project_dir = "/root/autodl-tmp/first_paper/python"
if python_project_dir not in sys.path:
    sys.path.insert(0, python_project_dir)

from stage3.CGBridge_Stage3 import CGBridgeStage3, CodeTranslateDataset

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
            ["cargo", "check", "--message-format=json", "--offline"],
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

def clean_generated_code(code: str) -> str:
    """
    针对 Llama 3.1 Base 模型的鲁棒后处理与多重物理切片机。
    通过动态字符串拼接生成三反引号，完美防止网页解析引擎中断。
    """
    if not code:
        return ""
        
    # 💡 动态构造三反引号字符串，防止 Canvas 渲染引擎提前截断
    bt = '`' * 3
        
    # 1. 物理移除 Base 模型特有的各类连体标签残骸 (例如：}#lang rustusestd -> }usestd)
    code = re.sub(r'#\s*lang\s*(?:rust|python|html|javascript|cpp|c|ruby)?', '\n', code, flags=re.IGNORECASE)

    # 2. 提取标准的 Markdown Rust 代码块（若存在）
    rust_blocks = re.findall(rf'{bt}(?:[Rr]ust|[Rr]sthon)\s*\n(.*?)\n{bt}', code, re.DOTALL)
    if rust_blocks:
        code = rust_blocks[0].strip()
    else:
        # 兼容处理未闭合的代码块
        unclosed = re.search(rf'{bt}(?:[Rr]ust|[Rr]sthon)\s*\n(.*)', code, re.DOTALL)
        if unclosed:
            code = unclosed.group(1).strip()
        
    # 3. 剥离前置非代码废话，准确定位 Rust 首个合法关键字作为起点
    # 包括常见的：fn, pub, use, struct, enum, impl, const, static 或注释
    start_match = re.search(r'(?m)^(?:pub\s+)?(?:fn|use|struct|enum|impl|const|static)\s+|^(?:\/\/\/|\/\/|\/\*)', code)
    if start_match:
        code = code[start_match.start():]

    # 4. 逐行精确扫描与逻辑熔断
    lines = code.splitlines()
    clean_lines = []
    
    # 熔断关键字集合：当 Base 模型在正文中开始输出这些非代码标记时，立即停止向下加载
    stop_patterns = [
        r'^\s*###',
        r'^\s*Question',
        r'^\s*Explanation',
        r'^\s*respond',
        r'^\s*Instruct',
        r'^\s*Solution',
        r'^\s*Title\s*[:>]',
        r'^\s*Answer\s*:',
        r'^\s*Example\s+usage',
        r'^\s*Output\s*:',
        r'^\s*Input\s*:',
        r'^\s*response\s+type',
        r'mathjax',
        r'html',
        rf'{bt}',
        r'^\s*def\s+',
        r'^\s*class\s+',
        r'^\s*import\s+(sys|math|ast|io)',
        r'^\s*function\s+',
        r'^\s*assert_eq!',
        r'^\s*println!',
        r'^\s*fn\s+main\s*\(\)' # 如果在主体函数后开始生成用于演示的 fn main() 则直接在此处截断
    ]
    
    for line in lines:
        stripped = line.strip()
        if not clean_lines and not stripped:
            continue
            
        # 发现任何熔断标志，物理停机
        triggered = False
        for pattern in stop_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                triggered = True
                break
                
        if triggered:
            break
        clean_lines.append(line)
        
    final_code_temp = "\n".join(clean_lines).strip()
    # 5. 基于大括号配对的终极闭包裁剪，只保留合法的 Rust 作用域，抛弃其后所有文字
    final_lines = []
    brace_count = 0
    has_opened = False
    
    for line in final_code_temp.splitlines():
        final_lines.append(line)
        for char in line:
            if char == '{':
                brace_count += 1
                has_opened = True
            elif char == '}':
                brace_count -= 1
        
        # 当大括号曾经打开过，且当前完全配对归零，意味着合法的 Rust 逻辑块完整结束，其后废话全部切除
        if has_opened and brace_count <= 0:
            break

    final_code = "\n".join(final_lines).strip()
    if brace_count > 0:
        final_code += "\n" + "}" * brace_count
    
    # 6. 字面量格式兼容与洗白
    final_code = final_code.replace("null", "None")
    return final_code.strip()

def calculate_pass_at_k(n, c, k):
    if c - n < k:
        return 1.0
    return 1.0 - math.comb(c - n, k) / math.comb(c, k)

def is_logic_valid(code: str) -> (bool, str):
    # 1. 基础占位符检查（保持不变，防止模型偷懒）
    if code.count("unimplemented!") > 2 or "todo!" in code:
        return False, "FIX: Do not use 'unimplemented!' or 'todo!'. Please translate the actual logic from Python."
    
    # 2. 检查空函数体
    if re.search(r'fn\s+\w+\s*\(.*?\)\s*(->\s*.*?)?\{\s*\}', code):
        return False, "FIX: Empty function body detected. Provide the implementation based on Python source."

    # 检查是否误用了 Python 的定义关键字 'def'
    if re.search(r'\bdef\s+\w+', code):
        return False, "FIX: Python syntax leakage detected ('def'). Use 'fn' for function definitions in Rust."

    # 检查是否误用了 Python 的布尔或空值关键字
    if "None" in code and "Option::None" not in code:
        return False, "FIX: Python leakage ('None'). Use 'None' (if Option) or '()' in Rust."
    
    if "True" in code or "False" in code:
        # 注意：Rust 是全小写的 true / false
        return False, "FIX: Python boolean leakage ('True/False'). Use lowercase 'true/false' in Rust."

    # 检查是否误用了 Python 的逻辑关键字
    if "elif" in code:
        return False, "FIX: Python syntax leakage ('elif'). Use 'else if' in Rust."
    
    # 4. 类型转换检查
    if re.search(r'\bint\(', code) or re.search(r'\bstr\(', code):
        return False, "FIX: Python-style type casting detected. Use '.parse()', 'as', or 'to_string()' in Rust."

    return True, ""

def execute_rust_unit_test(code: str, test_cases: list, timeout: int = 15) -> (bool, bool):
    """
    针对 HumanEval 特化的物理隔离 Rust 评测机
    """
    if not code.strip():
        return False, False

    # 1. 强力剥离 Markdown 标记
    code = code.replace("```rust", "").replace("```", "").strip()

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

        # 🎯 提取大模型生成的函数名
        func_match = re.search(r'pub\s+fn\s+([a-zA-Z0-9_]+)', code)
        if not func_match:
            func_match = re.search(r'fn\s+([a-zA-Z0-9_]+)', code)
        model_fn = func_match.group(1) if func_match else None

        source_lines = [code, "\n// ==================== TESTS ====================\n"]
        raw_test_list = test_cases if isinstance(test_cases, list) else [str(test_cases)]
        
        for idx, t in enumerate(raw_test_list):
            t_str = str(t).strip()
            
            # 🌟【降维打击】彻底消灭测试集开头的“内鬼”右花括号
            if t_str.startswith("}"):
                t_str = t_str.lstrip("}").strip()
                
            if not t_str:
                continue
            
            # 确定别名导入
            if model_fn and model_fn != "candidate":
                alias_line = f"    use super::{model_fn} as candidate;"
            else:
                alias_line = "    use super::*;"

            if "fn main()" in t_str:
                # 移除单测自带的重复 candidate 绑定声明，防止 Rust 报冲突警告
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
                # 碎片化单行断言
                source_lines.append("#[test]")
                source_lines.append(f"fn test_fragment_case_{idx}() {{")
                source_lines.append("    use super::*;")
                if model_fn and model_fn != "candidate" and "candidate" in t_str:
                    source_lines.append(f"    let candidate = super::{model_fn};")
                source_lines.append(f"    {t_str if t_str.endswith(';') else t_str + ';'}")
                source_lines.append("}")

        with open(os.path.join(src_dir, "lib.rs"), "w", encoding="utf-8") as f:
            f.write("\n".join(source_lines))

        # 1. 静态编译检查
        compile_res = subprocess.run(
            ["cargo", "check", "--offline"],
            cwd=tmp_project_dir,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout
        )

        if compile_res.returncode != 0:
            return False, False

        # 2. 动态单元测试执行
        test_res = subprocess.run(
            ["cargo", "test", "--offline", "--", "--nocapture"],
            cwd=tmp_project_dir,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout
        )
        
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

class ExtendedCodeTranslateDataset(CodeTranslateDataset):
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        raw_emb = item['graph_emb']
        graph_emb = torch.tensor(raw_emb, dtype=torch.float) if not isinstance(raw_emb, str) else torch.tensor(json.loads(raw_emb), dtype=torch.float)
        
        unit_tests = item.get('racket_tests') or item.get('r_tests') or item.get('rust_tests') or item.get('ocaml_tests') or item.get('lua_tests')
        if isinstance(unit_tests, str):
            try:
                unit_tests = json.loads(unit_tests)
            except:
                pass
            
        return (
            item.get('idx', idx),
            graph_emb,
            str(item.get('src_code', item.get('python_code', ""))),
            str(item.get('nl_description', "")),
            unit_tests
        )

def batch_smart_generate(model, graph_embs, codes, nl_descs, config, device):
    use_graph_info = config['generation'].get('use_graph_info', True)
    num_candidates = config['generation'].get('num_candidates', 5)
    max_new_tokens = config['generation'].get('max_new_tokens', 1536)

    model.llm_tokenizer.padding_side = "left" 

    if use_graph_info:
        query_feats = graph_embs.to(device).to(dtype=model.llm_model.dtype)
        with torch.no_grad():
            inputs_embeds, attn_mask = model.prepare_lm_input(
                gtokens=query_feats, text_input=codes, nl_desc=nl_descs, use_context=True
            )
    else:
        # 纯代码消融模式
        full_texts = []
        for i in range(graph_embs.size(0)):
            user_p = config["prompts"]["translation"]["user_prompt"].format(
                struct_tokens="<|struct_start|><|struct_end|>",
                logic_tokens="<|logic_start|><|logic_end|>",
                semantic_tokens="<|semantic_start|><|semantic_end|>",
                global_tokens="<|global_start|><|global_end|>",
                nl_description=nl_descs[i] or "None",
                python_code=codes[i]
            )
            prompt = f"<|im_start|>system\n{config['prompts']['translation']['system_prompt']}<|im_end|>\n<|im_start|>user\n{user_p}<|im_end|>\n<|im_start|>assistant\n"
            full_texts.append(prompt)
        
        batch = model.llm_tokenizer(full_texts, padding=True, return_tensors="pt").to(device)
        inputs_embeds = model.llm_model.get_input_embeddings()(batch.input_ids)
        attn_mask = batch.attention_mask

    if num_candidates > 1:
        inputs_embeds = inputs_embeds.repeat_interleave(num_candidates, dim=0)
        attn_mask = attn_mask.repeat_interleave(num_candidates, dim=0)

    with torch.no_grad():
        generate_ids = model.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            temperature=config['generation'].get('temperature', 0.9),
            top_p=config['generation'].get('top_p', 0.95),
            repetition_penalty=config['generation'].get('repetition_penalty', 1.05),
            use_cache=True,
            do_sample=True if num_candidates > 1 else False,
            pad_token_id=model.llm_tokenizer.pad_token_id,
            eos_token_id=model.llm_tokenizer.eos_token_id
        )

    all_raw_texts = model.llm_tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
    
    batch_results = []
    origin_bsz = graph_embs.size(0)
    for i in range(origin_bsz):
        start_idx = i * num_candidates
        end_idx = (i + 1) * num_candidates
        sample_candidates = [clean_generated_code(t) for t in all_raw_texts[start_idx:end_idx]]
        batch_results.append(sample_candidates)
        
    return batch_results

def run_inference(config_path):
    # 1. 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    use_graph = config.get("generation", {}).get("use_graph_info", False)
    tgt_language = config.get('tgt_language', 'Rust')
    batch_size = config.get('data', {}).get('batch_size', 4)
    num_workers = config.get('data', {}).get('num_workers', 2)

    print(f"\n================ ⚙️ 启动【CGBridge Stage3 推理与全维度评估】================")
    print(f"当前运行模式: {'【图增强模式】' if use_graph else '【纯代码消融模式】'}")
    print(f"检测到 Batch Size: {batch_size} | 目标语言: {tgt_language}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CGBridgeStage3(config).to(device)

    # 保留 stage3_Inference 原有的 stage3_model_dir 加载机制
    stage3_ckpt = config['model'].get('stage3_model_dir', '')
    if os.path.exists(stage3_ckpt):
        print(f"🔥 [权重融合] 检测到微调权重，正在从 {stage3_ckpt} 加载最新投影层与桥接层...")
        load_result = model.load_tuned_stage3_weights(stage3_ckpt) 
        important_missing = [k for k in load_result.missing_keys if "llm_proj" in k or "stage2_bridge" in k]
        if important_missing:
            print(f"❌ 严重警告: Stage 3 关键训练层在 Checkpoint 中缺失: {important_missing[:5]}")
        else:
            print(f"✅ 确认: Stage 3 投影层 (llm_proj) 与桥接层 (Q-Former) 权重已完美融合！")
    else:
        print("⚠️ [警告] 未找到 Stage 3 微调权重！将使用原始未经微调的基座进行推理！")

    model.to(device)
    model.eval()

    # 2. 准备数据集与 DataLoader
    print(f"📖 正在加载测试集: {config['data']['test_path']} ...")
    dataset = ExtendedCodeTranslateDataset(config['data']['test_path'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    output_csv = config['output'].get('multilang_evaluation_report', './output/stage3_inference_result.csv')
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # 3. 初始化 Reinforced 格式的评测计数器
    total_samples = len(dataset)
    compile_success_total = 0
    pass_1_total = 0
    pass_2_total = 0
    pass_5_total = 0
    processed_samples_count = 0

    max_workers = min(os.cpu_count(), 16)
    print(f"🚀 双轨单元测试评测开启！题目总数: {total_samples} | CPU 并行度: {max_workers}")

    # 4. 执行推理与并行评估
    with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'python_code', 'nl_desc', 'all_candidates', 'passed_code', 'compile_ok', 'pass_1', 'pass_2', 'pass_5'])

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating & Dual Evaluating"):
                idxs, gtokens, python_codes, nl_descs, batch_unit_tests = batch

                # GPU 批量生成候选代码
                batch_candidates = batch_smart_generate(model, gtokens.to(device), python_codes, nl_descs, config, device)

                # CPU 并行处理单元测试与 Pass@K 评估
                batch_futures = []
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    for i in range(len(idxs)):
                        raw_idx = idxs[i].item() if isinstance(idxs[i], torch.Tensor) else str(idxs[i])
                        future = executor.submit(evaluate_pass_at_k_hybrid_job, batch_candidates[i], batch_unit_tests[i])
                        batch_futures.append({
                            'idx': raw_idx,
                            'python_code': python_codes[i],
                            'nl_desc': nl_descs[i],
                            'candidates': batch_candidates[i],
                            'future': future
                        })

                    # 收集并打印当前 Batch 的评估结果
                    for item in batch_futures:
                        try:
                            comp_ok, p1, p2, p5, passed_code = item['future'].result()
                            compile_success_total += comp_ok
                            pass_1_total += p1
                            pass_2_total += p2
                            pass_5_total += p5

                            writer.writerow([
                                item['idx'],
                                item['python_code'],
                                item['nl_desc'],
                                json.dumps(item['candidates'], ensure_ascii=False),
                                passed_code,
                                comp_ok, p1, p2, p5
                            ])

                            processed_samples_count += 1

                            # 实时迁移 Reinforced 的中间进度报告打印
                            if processed_samples_count % 10 == 0:
                                tqdm.write(f"\n📢 [中间进度报告] ── 累计已处理数: {processed_samples_count}/{total_samples}")
                                tqdm.write(f"    ├─  当前累计编译通过率: {compile_success_total / processed_samples_count * 100:.2f}%")
                                tqdm.write(f"    ├─ 真实单元测试 Pass@1: {pass_1_total / processed_samples_count * 100:.2f}%")
                                tqdm.write(f"    ├─ 真实单元测试 Pass@2: {pass_2_total / processed_samples_count * 100:.2f}%")
                                tqdm.write(f"    └─ 真实单元测试 Pass@5: {pass_5_total / processed_samples_count * 100:.2f}%\n")

                        except Exception as e:
                            print(f"\n[Error] 样本 {item['idx']} 评测异常: {e}")

                f.flush()
                torch.cuda.empty_cache()

    # 5. 迁移 Reinforced 的终极学术报告汇报
    print(f"\n🏆 ======================== CGBridge 终极 {tgt_language} 多语言翻译评测报告 ========================")
    print(f"📊 评估题目总计: {total_samples} 道")
    print(f"🛠️ 基础语法/编译通过率 (Compilation Pass Rate): {compile_success_total / total_samples * 100:.2f}%")
    print(f"📈 真实自动化单元测试 Pass@1 成功率: {pass_1_total / total_samples * 100:.2f}%")
    print(f"📈 真实自动化单元测试 Pass@2 成功率: {pass_2_total / total_samples * 100:.2f}%")
    print(f"📈 真实自动化单元测试 Pass@5 成功率: {pass_5_total / total_samples * 100:.2f}%")
    print(f"==================================================================================\n")

if __name__ == "__main__":
    CONFIG_FILE = "/root/autodl-tmp/first_paper/python/stage3/stage3.yaml"
    run_inference(CONFIG_FILE)