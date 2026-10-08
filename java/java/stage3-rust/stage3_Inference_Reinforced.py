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
    # 1. 静态语法空壳拦截（保持原样）
    if re.search(r'fn\s+\w+\s*\(.*?\)\s*(->\s*.*?)?\{\s*\}', code):
        return False, "FIX: Empty function body detected."
        
    # 2. 严格的 Java 关键字与语法泄漏拦截
    if re.search(r'\b(public|private|protected)\s+class\b', code) or re.search(r'\bpublic\s+static\b', code):
        return False, "FIX: Java class/method syntax leakage detected."
    if re.search(r'\bnew\s+\w+\s*\(', code):
        return False, "FIX: Java 'new' keyword leakage detected."
    if "NullPointerException" in code or "IllegalArgumentException" in code:
        return False, "FIX: Java Exception leakage detected."
    if "System.out.print" in code:
        return False, "FIX: Java stdout leakage detected."
        
    # 3. 优化后的容器泄漏拦截：只拦截带有 Java 特性的 new 实例化，不拦截正常的 HashMap<K, V>
    if "ArrayList" in code or "new ArrayList" in code:
        return False, "FIX: Java container leakage ('ArrayList')."
    if "new HashMap" in code:
        return False, "FIX: Java container leakage ('new HashMap')."
        
    return True, ""

def verify_rust_syntax_with_log(code: str, timeout: int = 20):
    """来自 stage3_Inference.py 的 Cargo 沙盒精准语法编译器检查"""
    base_temp_path = "/dev/shm" if os.path.exists("/dev/shm") else None
    temp_dir = tempfile.mkdtemp(prefix="rust_verify_", dir=base_temp_path)
    src_dir = os.path.join(temp_dir, "src")
    os.makedirs(src_dir, exist_ok=True)

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
        for line in check_res.stdout.splitlines():
            try:
                msg = json.loads(line)
                if msg.get("reason") == "compiler-message" and msg.get("message", {}).get("level") == "error":
                    compiler_msg = msg.get("message", {})
                    code_info = compiler_msg.get("code")
                    err_code = code_info.get("code") if code_info else "SyntaxError"
                    rendered = compiler_msg.get("rendered", "")
                    if rendered:
                        clean_rendered = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', rendered)
                        errors.append(f"[{err_code}] {clean_rendered.strip()}")
                        error_codes.append(err_code)
            except:
                continue
        return False, "\n---\n".join(errors[:4]), error_codes
    except:
        return False, "Timeout/Execution Error", ["E_UNKNOWN"]
    finally:
        shutil.rmtree(temp_dir)

def execute_rust_unit_test(code: str, test_cases: list, timeout: int = 20) -> (bool, bool):
    """
    特化特强版 Rust 评测机：
    1. 强力清除测试集开头的“内鬼”右花括号 `}`
    2. 支持完美的真名还原机制与别名重映射
    """
    if not code.strip():
        return False, False

    # 逻辑初筛 (Java 泄漏防御)
    is_logic_clean, _ = is_logic_valid(code)
    if not is_logic_clean:
        return False, False

    # 静态语法初筛
    is_syntax_ok, _, _ = verify_rust_syntax_with_log(code, timeout=timeout)
    if not is_syntax_ok:
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

        # 🎯 A. 提取大模型生成的真实函数名
        func_match = re.search(r'\bfn\s+([a-zA-Z0-9_]+)\s*\(', code)
        model_fn = func_match.group(1) if func_match else None

        raw_test_list = test_cases if isinstance(test_cases, list) else [str(test_cases)]
        test_script_str = "\n".join([str(t).strip() for t in raw_test_list if str(t).strip()])

        # 🌟🌟🌟【降维打击】彻底消灭测试数据开头的“内鬼”右花括号
        test_script_str = test_script_str.lstrip("}").strip()

        # 🎯 B. 从测试用例提取它期望赋值给 candidate 的函数名
        expected_func_name = None
        expected_match = re.search(r'let\s+candidate\s*=\s*([a-zA-Z0-9_]+);', test_script_str)
        if expected_match:
            expected_func_name = expected_match.group(1)

        # 🎯 C. 生成动态重命名绑定映射
        alias_str = ""
        if model_fn and expected_func_name and model_fn != expected_func_name:
            alias_str = f"\n    use super::{model_fn} as {expected_func_name};\n"
        elif model_fn and model_fn != "candidate":
            alias_str = f"\n    use super::{model_fn} as candidate;\n"

        # 🎯 D. 转换测试用例中的 fn main() 为标准的 cargo test 格式
        test_script_str = test_script_str.replace("fn main()", "fn test_pipeline()")
        if "#[test]" not in test_script_str:
            test_script_str = "#[test]\n" + test_script_str

        # 完美拉平拓扑组装
        source_lines = [
            code,
            "\n#[cfg(test)]",
            "mod tests {",
            "    use super::*;",
            f"    {alias_str}",
            f"    {test_script_str}",
            "}"
        ]

        with open(os.path.join(src_dir, "lib.rs"), "w", encoding="utf-8") as f:
            f.write("\n".join(source_lines))

        # 1. 静态深度编译检查
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
            ["cargo", "test", "--offline"],
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
    """提取大模型输出中的 Markdown 代码块"""
    if not code: return ""
    rust_blocks = re.findall(r'```rust\s*\n?(.*?)\n?```', code, re.DOTALL)
    if not rust_blocks:
        rust_blocks = re.findall(r'```\s*\n?(.*?)\n?```', code, re.DOTALL)
    final_code = rust_blocks[-1] if rust_blocks else code
    final_code = re.sub(r'<\|im_.*?\|>', '', final_code)
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

class PassKEvaluationDataset(CodeTranslateDataset):
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        raw_emb = item['graph_emb']
        graph_emb = torch.tensor(raw_emb, dtype=torch.float) if not isinstance(raw_emb, str) else torch.tensor(json.loads(raw_emb), dtype=torch.float)
        
        # 终极多字段适配：防止跨数据集导致键名污染
        unit_tests = item.get('rust_tests') or item.get('racket_tests') or item.get('rs_tests') or item.get('test_cases')
        if isinstance(unit_tests, str):
            try: unit_tests = json.loads(unit_tests)
            except: pass
            
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
    
    tgt_language = config.get('tgt_language', 'Rust')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    use_api_mode = config["model"].get("use_api_mode", False)
    llm_type = config["model"].get("llm_type", "qwen").lower()
    use_graph = config.get("generation", {}).get("use_graph_info", False)
    
    print(f"\n================ ⚙️ 启动完全体【Java 到 Rust 消融推理管线】================")
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
                target_mods = ["c_attn", "c_proj"]  # StarCoder 经典的合并注意力机制
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

    # 准备独立多语言测试集并加载
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
                                tqdm.write(f"    ├─  当前累计编译通过率: {compile_success_total / processed_samples_count * 100:.2f}%")
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