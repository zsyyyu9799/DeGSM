import os
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "FALSE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import re
import csv
import yaml
from tqdm import tqdm
from CGBridge_Stage3 import CGBridgeStage3, CodeTranslateDataset
from torch.utils.data import DataLoader
import subprocess
import tempfile
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

def is_logic_valid(code: str) -> (bool, str):
    if not code.strip():
        return False, "FIX: Code is completely empty."

    # ================= 1. 拦截 Python 关键字与结构泄漏 =================
    if re.search(r'\bdef\s+\w+', code):
        return False, "FIX: Python 'def' detected. Use (define ...) instead."
    if re.search(r'\belif\b', code):
        return False, "FIX: Python 'elif' detected. Use 'cond' or 'if' in Racket."
    if "True" in code or "False" in code:
        return False, "FIX: Python booleans 'True/False' detected. Racket uses #t/#f."
    if re.search(r'(if|else|for|while).+:\s*$', code, re.MULTILINE):
        return False, "FIX: Python-style trailing colon ':' detected."
    
    # 🎯 Racket 专属修复：更严谨地拦截 Python 式独占行注释（# text），但放行 Racket 自身的字符/哈希/向量标相（#t, #hash）
    if re.search(r'(?m)^\s*#\s+[^&%laft\s\[\(]', code):
        return False, "FIX: Python-style comment '#' detected. Racket uses ';' for comments."

    # ================= 2. Racket 强约束指标校验 =================
    # 约束 A: 必须包含 Racket 的语言声明头部
    if "#lang racket" not in code:
        return False, "FIX: Missing '#lang racket' header at the beginning of the script."

    # 约束 B: 极简的括号对称性预检
    if code.count('(') != code.count(')'):
        return False, f"FIX: Unbalanced parentheses! '(' count is {code.count('(')}, ')' count is {code.count(')')}."
    if code.count('[') != code.count(']'):
        return False, f"FIX: Unbalanced square brackets! '[' count is {code.count('[')}, ']' count is {code.count(']')}"

    return True, ""

def batch_smart_generate(model, graph_embs, codes, nl_descs, config, device):
    use_graph_info = config['generation'].get('use_graph_info', True)
    num_candidates = config['generation'].get('num_candidates', 5)
    max_new_tokens = config['generation'].get('max_new_tokens', 1536)

    model.llm_tokenizer.padding_side = "left" 

    if use_graph_info:
        # =================【图增强模式】=================
        s2_samples = {
            'node_embs': graph_embs.to(device),
            'text_input_ids': torch.zeros((graph_embs.size(0), 1), dtype=torch.long).to(device),
            'text_atts': torch.zeros((graph_embs.size(0), 1), dtype=torch.long).to(device),
            'meta': [{'node_type_map': []} for _ in range(graph_embs.size(0))]
        }
        with torch.no_grad():
            s2_outputs = model.stage2_bridge(s2_samples, task='test')
            query_feats = s2_outputs['query_output'].to(dtype=model.llm_model.dtype)

            inputs_embeds, attn_mask = model.prepare_lm_input(
                gtokens=query_feats, 
                text_input=codes, 
                nl_desc=nl_descs, 
                use_context=True
            )
    else:
        # =================【纯代码消融模式】=================
        # 🎯 修复：保留结构边界特殊标记，仅抽离图向量占位符，保证 RoPE 位置编码分布一致
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

    # 🎯 修复：严谨处理多候选生成的张量复制
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

def verify_racket_syntax_with_log(code: str, timeout: int = 4):
    """
    使用 Racket 官方命令行工具进行纯语法语义检查（不实际执行代码逻辑）
    """
    base_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
    
    with tempfile.NamedTemporaryFile(suffix=".rkt", dir=base_dir, mode="w", encoding="utf-8", delete=False) as tf:
        tf.write(code)
        temp_file_path = tf.name

    try:
        # 🎯 Racket 专属修复：绝不能直接运行 `racket file.rkt`！否则死循环代码会导致进程永远卡死。
        # 解决方案：使用 Racket 内置展开检查，只做编译和词法语法验证 (syntax-only)。
        check_res = subprocess.run(
            ["raco", "make", "--no-deps", temp_file_path], 
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
        
        error_tags = []
        if "read" in stderr and "parenthesis" in stderr:
            error_tags.append("Racket_UnbalancedParenthesis") 
        elif "unbound" in stderr or "undefined" in stderr:
            error_tags.append("Racket_UnboundIdentifier")    
        else:
            error_tags.append("Racket_CompileOrRuntimeError")

        return False, clean_error, error_tags

    except subprocess.TimeoutExpired:
        return False, "Parsing timeout (Likely infinite macro expansion or compilation stall)", ["TIMEOUT"]
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

    # 匹配 ```racket ... ``` 或者 ```lisp ... ``` 或者 ``` ... ```
    racket_blocks = re.findall(r'```[Rr]acket\s*\n?(.*?)\n?```', code, re.DOTALL)
    if not racket_blocks:
        racket_blocks = re.findall(r'```\s*\n(.*?)\n```', code, re.DOTALL)
        
    final_code = racket_blocks[-1] if racket_blocks else code

    final_code = re.sub(r'<\|im_.*?\|>', '', final_code)
    final_code = final_code.strip()
    
    # 如果大模型错把 #lang 弄丢了，强制补齐
    if not final_code.startswith("#lang "):
        final_code = "#lang racket\n" + final_code
            
    return final_code

def select_best_candidate_on_cpu(candidates, python_src):
    """
    CPU 并行校验：优先挑出完美通过 Racket 语言语法与运行审查的候选。
    """
    first_candidate_info = None 
    
    for i, cand in enumerate(candidates):
        is_syntax_ok, stderr, found_errors = verify_racket_syntax_with_log(cand)
        is_logic_ok, logic_msg = is_logic_valid(cand)
        current_is_passed = is_syntax_ok and is_logic_ok
        
        if i == 0:
            first_candidate_info = (cand, is_syntax_ok, is_logic_ok, found_errors)
        
        if current_is_passed:
            return cand, is_syntax_ok, is_logic_ok, found_errors, "Passed"

    cand, s_ok, l_ok, errs = first_candidate_info
    return cand, s_ok, l_ok, errs, "Failed"

def run_inference(config_path):
    # 1. 加载配置与模型
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    use_graph = config.get("generation", {}).get("use_graph_info", False)
    print(f"--- 启动推理 ---")
    batch_size = config.get('data', {}).get('batch_size', 4)
    num_workers = config.get('data', {}).get('num_workers', 2)

    print(f"当前运行模式: {'【图增强模式】' if use_graph else '【纯代码消融模式】'}")
    print(f"检测到 Batch Size: {batch_size}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CGBridgeStage3(config).to(device)
    # ======= ⭐ 新增：加载你微调训练好的 Stage 3 权重（修复版） =======
    stage3_ckpt = config['model'].get('stage3_model_dir', '')
    if os.path.exists(stage3_ckpt):
        print(f"🔥 [权重融合] 检测到微调权重，正在从 {stage3_ckpt} 加载最新投影层与桥接层...")
        
        # 🎯 修复点：用 load_result 接住安全放行通道返回的结果
        load_result = model.load_tuned_stage3_weights(stage3_ckpt) 
        
        # 过滤掉大模型（LLM）那几十万个被冻结的 Missing Keys，只看我们在乎的层
        important_missing = [k for k in load_result.missing_keys if "llm_proj" in k or "stage2_bridge" in k]
        
        if important_missing:
            print(f"❌ 严重警告: Stage 3 关键训练层在 Checkpoint 中缺失: {important_missing[:5]}")
        else:
            print(f"✅ 确认: Stage 3 投影层 (llm_proj) 与桥接层 (Q-Former) 权重已完美融合！")
    else:
        print("⚠️ [警告] 未找到 Stage 3 微调权重！将使用原始未经微调的基座进行推理！")
    # ===============================================================

    model.to(device)
    model.eval()

    # 2. 准备数据集
    dataset = CodeTranslateDataset(config['data']['test_path'])

    # dataset.df = dataset.df.iloc[39:].reset_index(drop=True) #第40条数据开始
    dataloader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=False)

    output_csv = config['output']['inference_csv']
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    output_jsonl = os.path.join(os.path.dirname(output_csv), "Racket_inference_results_original.jsonl")
    
    # 3. 初始化统计与并行池
    max_workers = os.cpu_count()
    success_count = 0
    total_samples = len(dataset)
    
    print(f"--- 启动实时流式推理 ---")
    print(f"总样本数: {total_samples}, 并行校验度: {max_workers}")

    # 以追加模式打开文件，防止程序崩溃丢失数据
    with open(output_csv, mode='w', encoding='utf-8', newline='') as f_csv, \
         open(output_jsonl, mode='w', encoding='utf-8') as f_jsonl:
        writer = csv.writer(f_csv)
        writer.writerow(['idx', 'python_code', 'nl_desc', 'all_candidates', 'passed_code', 'status'])
             
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for batch in tqdm(dataloader, desc="Generating & Verifying"):
                idxs, gtokens, python_codes, nl_descs = batch
                
                # --- Step A: GPU 批量生成 ---
                with torch.no_grad():
                    # batch_candidates: List[List[str]]
                    batch_candidates = batch_smart_generate(model, gtokens.to(device), python_codes, nl_descs, config, device)
                
                # --- Step B: 提交当前 Batch 到 CPU 并行校验 ---
                batch_futures = []
                for i in range(len(idxs)):
                    future = executor.submit(select_best_candidate_on_cpu, batch_candidates[i], python_codes[i])
                    batch_futures.append({
                        'idx': idxs[i].item(),
                        'python_code': python_codes[i],
                        'nl_desc': nl_descs[i],
                        'candidates': batch_candidates[i],
                        'future': future
                    })

                # --- Step C: 等待当前 Batch 校验完成并立即写入文件 ---
                for item in batch_futures:
                    try:
                        # res_text 是选出的最好代码, status 是 "Passed" 或 "Failed"
                        res_text, is_comp_ok, is_logic_ok, found_errors, status = item['future'].result()
                        
                        passed_code = ""
                        final_status = "Failed"
                        
                        if status == "Passed":
                            passed_code = res_text
                            final_status = "Passed"
                            success_count += 1
                            print(f"样本 {item['idx']}通过编译，共通过编译{success_count}")
                        
                        # 写入数据：all_candidates 保存为 JSON 字符串
                        writer.writerow([
                            item['idx'],
                            item['python_code'],
                            item['nl_desc'],
                            json.dumps(item['candidates'], ensure_ascii=False),
                            passed_code,
                            final_status
                        ])
                        jsonl_item = {
                            "idx": item['idx'],
                            "python_code": item['python_code'],
                            "nl_desc": item['nl_desc'],
                            "all_candidates": item['candidates'],
                            "passed_code": passed_code,
                            "status": final_status
                        }
                        f_jsonl.write(json.dumps(jsonl_item, ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"\n[Error] 样本 {item['idx']} 校验异常: {e}")
                
                # 显式刷新缓冲区，确保每一 Batch 都写到硬盘
                f_csv.flush()
                f_jsonl.flush()
                torch.cuda.empty_cache()

    print(f"\n推理完成!")
    print(f"当前运行模式: {'【图增强模式】' if use_graph else '【纯代码消融模式】'}")
    print(f"总样本: {total_samples}, 编译通过: {success_count}")
    print(f"最终成功率: {success_count/total_samples*100:.2f}%")

if __name__ == "__main__":
    # 现在只需要指定一个配置文件路径即可
    CONFIG_FILE = "/root/autodl-tmp/first_paper/python/stage3/stage3.yaml"
    run_inference(CONFIG_FILE)