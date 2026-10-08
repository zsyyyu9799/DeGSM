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
    验证生成的 OCaml 代码是否包含严重的逻辑占位符或 Java 语法泄漏。
    """
    # 1. 基础占位符检查（防止模型用 failwith 等占位挂空挡）
    if "failwith" in code.lower() and "not implemented" in code.lower():
        return False, "FIX: Do not use 'failwith \"not implemented\"' placeholders. Please translate the actual logic from Java."
    
    # 2. 检查悬空函数定义
    if code.strip().endswith("let") or code.strip().endswith("="):
        return False, "FIX: Incomplete OCaml function body detected. Provide the full implementation."

    # 3. 🎯 面向 Java 到 OCaml 的泄漏防御机制
    if re.search(r'\b(public|private|protected)\s+class\b', code) or re.search(r'\bpublic\s+static\b', code):
        return False, "FIX: Java class/method syntax leakage detected. Use plain OCaml definitions ('let')."

    if re.search(r'\bnew\s+\w+\s*\(', code):
        return False, "FIX: Java 'new' keyword leakage detected. Use record types or tuples directly in OCaml."

    if "NullPointerException" in code or "IllegalArgumentException" in code:
        return False, "FIX: Java Exception leakage detected. Use OCaml's standard Option/Result types or custom exceptions."
    
    if "System.out.print" in code:
        return False, "FIX: Java stdout leakage. Use OCaml's 'print_string' or 'Printf.printf'."

    if "ArrayList" in code or re.search(r'\bList<', code):
        return False, "FIX: Java container leakage detected. Use native OCaml 'list' or the 'List' module."

    # 4. OCaml 必要元素验证：确保代码具有核心声明标志
    if "let " not in code and "rec " not in code:
        return False, "FIX: Missing standard OCaml declaration bindings (such as 'let' or 'let rec')."

    return True, ""

def verify_ocaml_syntax_with_log(code: str, timeout: int = 5):
    """
    使用 ocamlc 编译器（或降级到 ocaml 解释器）对生成的 OCaml 代码进行严格的静态语法及类型检查。
    """
    base_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
    
    with tempfile.NamedTemporaryFile(suffix=".ml", dir=base_dir, mode="w", encoding="utf-8", delete=False) as tf:
        tf.write(code)
        temp_file_path = tf.name

    try:
        # 1. 优先使用 ocamlc -c (仅执行编译和类型推断检查，不产生二进制，非常快)
        check_res = subprocess.run(
            ["ocamlc", "-c", temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='ignore'
        )

        # 编译成功后清理可能产生的 .cmo 或 .cmi 中间文件
        cmo_file = temp_file_path.replace(".ml", ".cmo")
        cmi_file = temp_file_path.replace(".ml", ".cmi")
        if os.path.exists(cmo_file):
            os.remove(cmo_file)
        if os.path.exists(cmi_file):
            os.remove(cmi_file)

        if check_res.returncode == 0:
            return True, "Success", []

        stderr = check_res.stderr.strip()
        clean_error = stderr.replace(temp_file_path, "line")
        
        error_tags = []
        if "syntax error" in stderr.lower():
            error_tags.append("OCaml_SyntaxError")
        elif "type" in stderr.lower() or "expected" in stderr.lower():
            error_tags.append("OCaml_TypeError")
        else:
            error_tags.append("OCaml_CompileError")

        return False, clean_error, error_tags

    except FileNotFoundError:
        # 2. 如果环境中没有 ocamlc，无缝退回到普通的 ocaml 解释器诊断
        try:
            check_res = subprocess.run(
                ["ocaml", temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='ignore'
            )
            if check_res.returncode == 0:
                return True, "Success (Interpreter)", []
            stderr = check_res.stderr.strip()
            clean_error = stderr.replace(temp_file_path, "line")
            return False, clean_error, ["OCaml_InterpreterError"]
        except Exception:
            return False, "ocaml/ocamlc command not found. Run 'sudo apt-get install ocaml'", ["ENV_ERROR"]
    except subprocess.TimeoutExpired:
        return False, "Parsing timeout", ["TIMEOUT"]
    except Exception as e:
        return False, f"Execution failed: {str(e)}", ["EXEC_ERROR"]
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def execute_ocaml_unit_test(code: str, test_cases: list, timeout: int = 15) -> (bool, bool):
    """
    特化强化版 OCaml 语言单元测试评测机：
    1. 动态合并 OCaml 翻译代码与对应单测用例（例如：assert 断言）
    2. 基于原生 ocaml 解释器执行双轨安全评测，不漏掉任何 assert_failure。
    """
    if not code.strip():
        return False, False

    # 1. 逻辑与泄漏防御初筛
    is_logic_clean, _ = is_logic_valid(code)
    if not is_logic_clean:
        return False, False

    # 2. 静态语法/类型初筛
    is_syntax_ok, _, _ = verify_ocaml_syntax_with_log(code, timeout=timeout)
    if not is_syntax_ok:
        return False, False

    base_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
    
    raw_test_list = test_cases if isinstance(test_cases, list) else [str(test_cases)]
    test_script_str = "\n".join([str(t).strip() for t in raw_test_list if str(t).strip()])

    # 将生成的实现代码和测试代码（通常是 let () = assert ...）拼接成一个可运行的 .ml 文件
    full_code_with_tests = code + "\n\n(* --- Academic Unit Tests --- *)\n" + test_script_str

    with tempfile.NamedTemporaryFile(suffix=".ml", dir=base_dir, mode="w", encoding="utf-8", delete=False) as tf:
        tf.write(full_code_with_tests)
        temp_file_path = tf.name

    try:
        # 🎯 动态拉起 ocaml 解释器运行脚本
        test_res = subprocess.run(
            ["ocaml", temp_file_path],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True, 
            timeout=timeout,
            encoding='utf-8',
            errors='ignore'
        )
        
        output_and_err = (test_res.stdout + "\n" + test_res.stderr).lower()
        
        # OCaml 测试通过严格判定标准：
        # 1. 运行返回码必须为 0。
        # 2. 输出与标准错误中绝对不能出现 exception、failure 或 assert_failure 等崩溃信息。
        is_test_ok = (test_res.returncode == 0) and (
            "exception" not in output_and_err and 
            "failed" not in output_and_err and 
            "failure" not in output_and_err and
            "assert_failure" not in output_and_err and
            "error:" not in output_and_err
        )
        
        return True, is_test_ok

    except Exception:
        return True, False
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def evaluate_pass_at_k_hybrid_job(candidates: list, test_cases: list):
    """
    完全符合学术标准的 OCaml 语言双轨评估机
    """
    syntax_results = []
    unittest_results = []
    
    for cand in candidates:
        # 调用上方重构后的 OCaml 语言单元测试机
        syn_ok, ut_ok = execute_ocaml_unit_test(cand, test_cases)
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
    """
    融合了单分号精准防御、多标签兼容与【半闭合区间完美切除】的高级 OCaml 清洗函数。
    提示：通过 16 进制字符转义表示反引号，绕过 Markdown 解析边界问题。
    """
    if not code:
        return ""

    # 1. 在函数头部最优先定义并提取 cleaned
    cleaned = code.strip()
    
    # 💡 使用 \x60 精确指代反引号 `, 这样定义三连反引号，绝对不会触发 IDE 或文本解析器崩溃
    triple_backtick = '\x60\x60\x60'

    # 2. 匹配 markdown 中的 OCaml、ocaml 或 ml 代码块（不区分大小写）
    ocaml_blocks = re.findall(r'\x60\x60\x60(?:ocaml|ml|OCaml)?\s*\n?(.*?)\n?\x60\x60\x60', cleaned, re.DOTALL | re.IGNORECASE)
    if not ocaml_blocks:
        # 兼容无语言标签的普通闭合代码块
        ocaml_blocks = re.findall(r'\x60\x60\x60\s*\n?(.*?)\n?\x60\x60\x60', cleaned, re.DOTALL)

    if ocaml_blocks:
        # 取最后一个闭合块（通常是最新或完整的实现）
        final_code = ocaml_blocks[-1]
    else:
        # 3. 🎯【半闭合区间截断】
        # 如果代码开头没有 ```，但是中间包含 ```（大模型在 prompt 后面续写，然后自行打了个 ``` 闭合）
        if triple_backtick in cleaned and not cleaned.startswith(triple_backtick):
            final_code = cleaned.split(triple_backtick)[0].strip()
        else:
            # 4. 完全没有 ``` 标记的代码平铺模式
            final_code = cleaned

    # 5. 物理截断残留的无用 Markdown 标题和 Java 泄漏
    if "###" in final_code:
        final_code = final_code.split("###")[0]
    if "##" in final_code:
        final_code = final_code.split("##")[0]
    if "public class" in final_code:
        final_code = final_code.split("public class")[0]
    if "package " in final_code:
        final_code = final_code.split("package ")[0]

    # 6. 擦除特殊控制 Token (如 ChatML 标签)
    final_code = re.sub(r'<\|im_.*?\|>', '', final_code)
    final_code = final_code.strip()

    # 7. 🎯 绝妙的单分号强力防御逻辑：擦除大模型由于 Java/C 习惯在末尾带出来的非法单分号
    if final_code.endswith(";;"):
        final_code = final_code[:-2].strip()
    elif final_code.endswith(";"):
        if len(final_code) > 1 and final_code[-2] not in [')', '}', ']']:
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
        
        # 优先适配 OCaml 相关的测试字段
        unit_tests = item.get('ocaml_tests')
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
    
    tgt_language = config.get('tgt_language', 'OCaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    use_api_mode = config["model"].get("use_api_mode", False)
    llm_type = config["model"].get("llm_type", "qwen").lower()
    use_graph = config.get("generation", {}).get("use_graph_info", False)
    
    print(f"\n================ ⚙️ 启动完全体【Java 到 OCaml 消融推理管线】================")
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