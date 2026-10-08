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
    # 1. 基础占位符检查（防止模型偷懒使用未实现标记）
    if "error" in code and "unimplemented" in code.lower():
        # Racket 常用 (error 'name "unimplemented")
        return False, "FIX: Do not use error placeholders. Please translate the actual logic from Java."
    
    # 2. 检查空实现或悬空定义
    if re.search(r'\(define\s+\([a-zA-Z0-9_-]+\)\s*\)$', code.strip()):
        return False, "FIX: Incomplete function body detected. Provide the full Racket functional implementation."

    # 3. 🎯 面向 Java 到 Racket 的泄漏防御机制
    if re.search(r'\b(public|private|protected)\s+class\b', code) or re.search(r'\bpublic\s+static\b', code):
        return False, "FIX: Java class/method syntax leakage detected. Use Racket functions `(define (...) ...)`."

    if "{" in code or "}" in code:
        return False, "FIX: C-style curly braces leakage detected. Racket uses parentheses `(...)` or square brackets `[...]`."

    if re.search(r';$', code.strip()):
        # Racket 的注释是分号，末尾出现孤立的分号通常是 C 语法的分号泄漏
        # 排除掉只有注释的代码行
        lines = [l.strip() for l in code.split('\n') if l.strip()]
        if any(l.endswith(';') and not l.startswith(';') for l in lines):
            return False, "FIX: C-style trailing semicolon ';' detected. Racket does not use semicolons to terminate statements."

    if "System.out.print" in code:
        return False, "FIX: Java stdout leakage. Use `display`, `write`, or `printf` in Racket."

    if "ArrayList" in code or "HashMap" in code:
        return False, "FIX: Java container leakage detected. Use Racket native lists, vectors, or hash maps."

    # 防御 Java 的赋值语句操作符泄漏（Racket 应该用 set! 或 define）
    if re.search(r'\b[a-zA-Z0-9_-]+\s*=\s*[^=]', code) and not "define" in code:
        # 排除掉类似注释或 #lang 后的干扰，识别标准的 `x = 1;` 行为
        if ";" in code or "for" in code:
            return False, "FIX: C-style assignment '=' detected. Use (define ...) or (set! ...)."
    
    # 防御 Java 的中缀数学运算 (如 x + y, a < b)
    if re.search(r'\b[a-zA-Z0-9_-]+\s*([\+\-\*/\<>]=?)\s*[a-zA-Z0-9_-]+', code):
        # 排除 Racket 的类似合同或符号定义
        if not "(" in code.split('\n')[1]: # 简单过滤
            return False, "FIX: Infix operators detected. Racket uses prefix notation, e.g., (+ x y)."

    # 4. 🎯 防御 OCaml/Rust 历史遗留残留（防止大模型在历史任务中串词）
    if "fn " in code or "let " in code or "match " in code or "->" in code:
        if not re.search(r'\(let\b', code) and not re.search(r'\(match\b', code): # 排除 Racket 自身的 (let ...)
            return False, "FIX: ML-family syntax leakage detected (e.g., 'fn', 'let x ='). Write pure Racket S-expression syntax."

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
                java_code=codes[i]
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

    # 提取 racket 代码块块
    racket_blocks = re.findall(r'```racket\s*\n?(.*?)\n?```', code, re.DOTALL)
    if not racket_blocks:
        racket_blocks = re.findall(r'```\s*\n?(.*?)\n?```', code, re.DOTALL)
    final_code = racket_blocks[-1] if racket_blocks else code

    final_code = re.sub(r'<\|im_.*?\|>', '', final_code)
    final_code = final_code.strip()
    
    # 强力防御：大模型生成多行代码时可能会漏掉 #lang racket，或者在块内又写了一遍
    if not final_code.startswith("#lang racket"):
        # 如果代码内部有，则移到最上方；如果没有，则强行补齐
        if "#lang racket" in final_code:
            final_code = final_code.replace("#lang racket", "").strip()
        final_code = "#lang racket\n" + final_code
            
    return final_code

def select_best_candidate_on_cpu(candidates, java_src):
    """
    在子进程中运行。负责从多个候选里选出最好的，并返回检查详情。
    移除了 AST 强制过滤，改为以编译和逻辑校验为主。
    """
    first_candidate_info = None # 留底，如果全失败了返回第一个的结果
    
    for i, cand in enumerate(candidates):
        # 1. 语法检查 (调用外部 cargo check)
        is_syntax_ok, stderr, found_errors = verify_racket_syntax_with_log(cand)
        
        # 2. 逻辑检查 (检测 C 语法泄漏)
        is_logic_ok, logic_msg = is_logic_valid(cand)
        
        # 3. 综合判断
        current_is_passed = is_syntax_ok and is_logic_ok
        
        # 记录第一个候选的情况，用于全军覆没时的兜底返回
        if i == 0:
            first_candidate_info = (cand, is_syntax_ok, is_logic_ok, found_errors)
        
        # 如果当前候选完美通过，直接返回
        if current_is_passed:
            # 这里的 "Passed" 状态会被主进程识别并计入 success_count
            return cand, is_syntax_ok, is_logic_ok, found_errors, "Passed"

    # 如果所有候选都没通过，返回第一个候选的信息，标记为 Failed
    # 这样主进程会把这个样本放入 to_be_repaired 列表进行修复
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
    # dataset.df = dataset.df.head(10)
    # dataset.df = dataset.df.iloc[21:28].reset_index(drop=True)
    dataloader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=False)

    output_csv = config['output']['inference_csv']
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # 3. 初始化统计与并行池
    max_workers = os.cpu_count()
    success_count = 0
    total_samples = len(dataset)
    
    print(f"--- 启动实时流式推理 ---")
    print(f"总样本数: {total_samples}, 并行校验度: {max_workers}")

    # 以追加模式打开文件，防止程序崩溃丢失数据
    with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 表头：包含全部候选和最终通过的代码
        writer.writerow(['idx', 'java_code', 'nl_desc', 'all_candidates', 'passed_code', 'status'])

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for batch in tqdm(dataloader, desc="Generating & Verifying"):
                idxs, gtokens, java_codes, nl_descs = batch
                
                # --- Step A: GPU 批量生成 ---
                with torch.no_grad():
                    # batch_candidates: List[List[str]]
                    batch_candidates = batch_smart_generate(model, gtokens.to(device), java_codes, nl_descs, config, device)
                
                # --- Step B: 提交当前 Batch 到 CPU 并行校验 ---
                batch_futures = []
                for i in range(len(idxs)):
                    future = executor.submit(select_best_candidate_on_cpu, batch_candidates[i], java_codes[i])
                    batch_futures.append({
                        'idx': idxs[i].item(),
                        'java_code': java_codes[i],
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
                            item['java_code'],
                            item['nl_desc'],
                            json.dumps(item['candidates'], ensure_ascii=False),
                            passed_code,
                            final_status
                        ])
                    except Exception as e:
                        print(f"\n[Error] 样本 {item['idx']} 校验异常: {e}")
                
                # 显式刷新缓冲区，确保每一 Batch 都写到硬盘
                f.flush()
                torch.cuda.empty_cache()

    print(f"\n推理完成!")
    print(f"总样本: {total_samples}, 编译通过: {success_count}")
    print(f"最终成功率: {success_count/total_samples*100:.2f}%")

if __name__ == "__main__":
    # 现在只需要指定一个配置文件路径即可
    CONFIG_FILE = "/root/autodl-tmp/first_paper/java/stage3/stage3.yaml"
    run_inference(CONFIG_FILE)