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
    # 1. 基础占位符检查（防止模型用 stop 或类似挂空挡）
    if code.count("stop(") > 2 and "not implemented" in code.lower():
        return False, "FIX: Do not use 'stop()' placeholders. Please translate the actual logic from Java."
    
    # 2. 检查悬空函数定义
    if re.search(r'<- \s*function\s*\(.*?\)\s*$', code.strip()):
        return False, "FIX: Incomplete function body detected. Provide the full R functional implementation."

    # 3. 🎯 面向 Java 到 R 的泄漏防御机制
    if re.search(r'\b(public|private|protected)\s+class\b', code) or re.search(r'\bpublic\s+static\b', code):
        return False, "FIX: Java class/method syntax leakage detected. Use plain R functions (`<- function`)."

    if re.search(r'\bnew\s+\w+\s*\(', code):
        return False, "FIX: Java 'new' keyword leakage detected. Create R lists, vectors or data.frames directly."

    if "NullPointerException" in code or "IllegalArgumentException" in code:
        return False, "FIX: Java Exception leakage detected. Use R's standard `tryCatch` or custom error handling."
    
    if "System.out.print" in code:
        return False, "FIX: Java stdout leakage. Use `print()`, `cat()`, or `message()` in R."

    if "ArrayList" in code or re.search(r'\bList<', code):
        return False, "FIX: Java container leakage detected. Use native R vectors `c()`, `list()`, or `data.frame()`."

    # 4. 🎯 防御 OCaml/Rust 历史遗留残留（防止模型串词）
    if "fn " in code or "let " in code or "match " in code or "failwith " in code:
        return False, "FIX: Historical language syntax leakage (Rust/OCaml) detected. Write pure R syntax using `<-`."

    return True, ""

def verify_r_syntax_with_log(code: str, timeout: int = 5):
    base_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
    
    with tempfile.NamedTemporaryFile(suffix=".R", dir=base_dir, mode="w", encoding="utf-8", delete=False) as tf:
        tf.write(code)
        temp_file_path = tf.name

    try:
        # 【修改点】不要用 parse()，直接用 Rscript 运行它！
        # 如果只是定义类，运行它不会花太多时间，但能暴露出所有的运行时错误
        check_res = subprocess.run(
            ["Rscript", temp_file_path], # 直接执行整段脚本
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
        
        # 错题本分类细化
        error_tags = []
        if "unused argument" in stderr or "fields" in stderr:
            error_tags.append("R_OO_InvalidArgumentError") # 面向对象参数错误
        elif "unexpected" in stderr:
            error_tags.append("R_UnexpectedTokenError")
        else:
            error_tags.append("R_RuntimeError")

        return False, clean_error, error_tags

    except subprocess.TimeoutExpired:
        return False, "Parsing timeout", ["TIMEOUT"]
    except FileNotFoundError:
        return False, "Rscript not found. Run 'sudo apt-get install r-base'", ["ENV_ERROR"]
    except Exception as e:
        return False, f"Execution failed: {str(e)}", ["EXEC_ERROR"]
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def execute_r_unit_test(code: str, test_cases: list, timeout: int = 20) -> (bool, bool):
    """
    特化强化版 R 语言单元测试评测机：
    1. 动态合并 R 翻译代码与对应单元测试集（如 testthat 断言或 stopifnot 表达式）
    2. 基于原生 Rscript 运行时进行双轨安全评测
    """
    if not code.strip():
        return False, False

    # 1. 逻辑初筛 (Java 泄漏防御)
    is_logic_clean, _ = is_logic_valid(code)
    if not is_logic_clean:
        return False, False

    # 2. 静态/运行时语法初筛
    is_syntax_ok, _, _ = verify_r_syntax_with_log(code, timeout=timeout)
    if not is_syntax_ok:
        return False, False

    base_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
    
    raw_test_list = test_cases if isinstance(test_cases, list) else [str(test_cases)]
    test_script_str = "\n".join([str(t).strip() for t in raw_test_list if str(t).strip()])

    func_match = re.search(r'([a-zA-Z0-9_.]+)\s*(<-|=)\s*function', code)
    alias_str = ""
    if func_match:
        model_func_name = func_match.group(1).strip()
        for t in raw_test_list:
            t_str = str(t)
    full_code_with_tests = code + "\n\n" + alias_str + "\n\n" + test_script_str

    with tempfile.NamedTemporaryFile(suffix=".R", dir=base_dir, mode="w", encoding="utf-8", delete=False) as tf:
        tf.write(full_code_with_tests)
        temp_file_path = tf.name

    try:
        # 🎯 动态拉起 Rscript 解释器运行单元测试
        test_res = subprocess.run(
            ["Rscript", temp_file_path],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True, 
            timeout=timeout,
            encoding='utf-8',
            errors='ignore'
        )
        
        # 🎯 R 语言测试通过判定标准：
        # Rscript 报错时 returncode 不为 0。
        # 此外，针对常见的测试框架（如 testthat），捕获 stderr/stdout 中的 "Error"、"failed" 或 "FAILURE" 关键字。
        output_and_err = (test_res.stdout + "\n" + test_res.stderr).lower()
        
        is_test_ok = (test_res.returncode == 0) and (
            "error" not in output_and_err and 
            "failed" not in output_and_err and 
            "failure" not in output_and_err
        )
        
        return True, is_test_ok

    except Exception:
        return True, False
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def evaluate_pass_at_k_hybrid_job(candidates: list, test_cases: list):
    """
    完全符合学术标准的 R 语言双轨评估机
    """
    syntax_results = []
    unittest_results = []
    
    for cand in candidates:
        # 调用上方重构后的 R 语言测试机
        syn_ok, ut_ok = execute_r_unit_test(cand, test_cases)
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
    
def clean_generated_code(code: str, is_repair_mode: bool = False) -> str:
    if not code:
        return ""

    # 🎯 强力策略：优先寻找第一个完整的 ```R ... ``` 块，一旦闭合，后面的幻觉文本全部抛弃
    match = re.search(r'```[rR]\s*\n?(.*?)\n?```', code, re.DOTALL)
    if match:
        final_code = match.group(1)
    else:
        # 如果模型连 ```R 都没打，只打了匿名的 ```
        match_anon = re.search(r'```\s*\n?(.*?)\n?```', code, re.DOTALL)
        if match_anon:
            final_code = match_anon.group(1)
        else:
            # 兜底：如果模型完全平铺，且后面开始疯狂吐 Java 代码包名，直接截断到 Java 关键字前
            final_code = code
            if "public class" in final_code:
                final_code = final_code.split("public class")[0]
            if "package " in final_code:
                final_code = final_code.split("package ")[0]

    # 清洗特殊标签
    final_code = re.sub(r'<\|im_.*?\|>', '', final_code)
    final_code = final_code.strip()
    
    # 彻底擦除由于 Java 习惯带过来的尾部无用分号
    # 逐行清洗分号，防止行内或行尾分号导致 R 语言解析怪异
    lines = final_code.split("\n")
    cleaned_lines = []
    for line in lines:
        l = line.strip()
        if l.endswith(";") and not l.endswith(";;"):
            l = l[:-1].strip()
        cleaned_lines.append(l)
    final_code = "\n".join(cleaned_lines)
            
    return final_code

def batch_smart_generate(model, graph_embs, codes, nl_descs, config, device):
    use_graph_info = config['generation'].get('use_graph_info', True)
    num_candidates = config['generation'].get('num_candidates', 5)
    max_new_tokens = config['generation'].get('max_new_tokens', 1024)
    model.llm_tokenizer.padding_side = "left"

    # 🎯 无论是否开启图增强，一律走统一的 prepare_lm_input 接口，让多模型 Prompt 路由生效
    if use_graph_info:
        query_feats = graph_embs.to(device).to(dtype=model.llm_model.dtype)
        with torch.no_grad():
            inputs_embeds, attn_mask = model.prepare_lm_input(
                gtokens=query_feats, text_input=codes, nl_desc=nl_descs, use_context=True
            )
    else:
        # 消融模式：传入空的或者全零的特征占位，仅为了借用 prepare_lm_input 内部完美的模板分流
        fake_gtokens = torch.zeros((graph_embs.size(0), 32, 768), device=device)
        with torch.no_grad():
            inputs_embeds, attn_mask = model.prepare_lm_input(
                gtokens=fake_gtokens, text_input=codes, nl_desc=nl_descs, use_context=False
            )

    if num_candidates > 1:
        inputs_embeds = inputs_embeds.repeat_interleave(num_candidates, dim=0)
        attn_mask = attn_mask.repeat_interleave(num_candidates, dim=0)

    with torch.no_grad():
        generate_ids = model.llm_model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attn_mask, max_new_tokens=max_new_tokens,
            temperature=config['generation'].get('temperature', 0.2), # 💡 建议在代码中加入安全温度卡点
            top_p=config['generation'].get('top_p', 0.95),
            repetition_penalty=config['generation'].get('repetition_penalty', 1.05), use_cache=True,
            do_sample=True if num_candidates > 1 else False, pad_token_id=model.llm_tokenizer.pad_token_id,
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

class PassKEvaluationDataset(CodeTranslateDataset):
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        raw_emb = item['graph_emb']
        graph_emb = torch.tensor(raw_emb, dtype=torch.float) if not isinstance(raw_emb, str) else torch.tensor(json.loads(raw_emb), dtype=torch.float)
        
        # 优先适配 r 相关的测试字段
        unit_tests = item.get('r_tests') or item.get('test_cases') or item.get('r_tests') or item.get('rs_tests')
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
    
    tgt_language = config.get('tgt_language', 'R')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    use_api_mode = config["model"].get("use_api_mode", False)
    llm_type = config["model"].get("llm_type", "qwen").lower()
    use_graph = config.get("generation", {}).get("use_graph_info", False)
    
    print(f"\n================ ⚙️ 启动完全体【Java 到 R 消融推理管线】================")
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
                
            peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=target_mods, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
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