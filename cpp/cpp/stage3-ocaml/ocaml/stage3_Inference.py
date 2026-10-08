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
    """
    进化版 OCaml 静态边界拦截器。
    """
    if not code or not code.strip():
        return False, "FIX: Generated code is empty."
    # ================= 1. 拦截摆烂占位符 =================
    if code.count("failwith") > 2 or "raise" in code and "Unimplemented" in code:
        return False, "FIX: Do not use 'failwith' placeholders. Please translate the actual logic from C++."
    
    if re.search(r'let\s+\w+\s*.*?\s*=\s*$', code.strip()):
        return False, "FIX: Incomplete function body detected. Provide the full OCaml functional implementation."

    for match in re.finditer(r'([^=\s<>!+\-*/%&|^]+)\s*<-\s*', code):
        left_side = match.group(1).strip()
        if not (left_side.endswith('.') or left_side.endswith('.(') or left_side.endswith('.[')):
            # 排除可能出现的末尾可能带有的特殊符号干扰
            if not any(char in left_side for char in [')', ']', '"', "'"]):
                return False, f"FIX: Illegal assignment to immutable variable ({left_side} <- ...). Use 'ref' with ':=' or functional 'let-in' pattern."

    if "class " in code and "object" in code:
        # 如果代码里包含明显的 C++ 痕迹如 public: 或 private:，直接判断为 OOP 语法污染
        if re.search(r'\b(public|private|protected)\s*:', code):
            return False, "FIX: C++ access specifier leaked inside OCaml class context."

    if "open Core" in code or "Core.Std" in code or "open Base" in code:
        return False, "FIX: Core/Base library dependency detected. Restrict code to standard OCaml modules."

    if "cout" in code or "printf" in code and "%" not in code:
        return False, "FIX: C++ stdout leakage detected."

    if re.search(r'\b(std::vector|std::map|std::set|std::string|thread_local)\b', code):
        return False, "FIX: C++ Standard Library / keyword leakage detected."

    if re.search(r'\b(include|using namespace|#define)\b', code):
        return False, "FIX: C++ preprocessor macros or headers detected."

    if re.search(r'\b(nullptr|NULL)\b', code):
        return False, "FIX: C++ pointer leakage detected. Use Option (Some/None)."

    if "fn " in code or "impl " in code or "::new()" in code or "-> " in code and "val " not in code:
        return False, "FIX: Rust syntax leakage detected."

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
                cpp_code=codes[i]
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
    
def verify_ocaml_syntax_with_log(code: str, timeout: int = 5):
    """
    使用官方 ocamlc -c 严格验证 OCaml 语法与强类型系统合法性。
    修复了在大缓冲区输出下潜在的进程死锁隐患。
    """
    # 极致性能优化：优先使用 Linux 内存文件系统，减少多线程磁盘 I/O 冲突
    base_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
    
    with tempfile.NamedTemporaryFile(suffix=".ml", dir=base_dir, mode="w", encoding="utf-8", delete=False) as tf:
        tf.write(code)
        temp_file_path = tf.name

    try:
        # -w +A: 开启全量警告追踪
        # stdout/stderr 显式接入 PIPE，防止因缓冲区溢出导致多进程无声死锁
        check_res = subprocess.run(
            ["ocamlc", "-c", "-w", "+A", temp_file_path],
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
        
        # 建立严谨的跨语言错题本分类标签（用于后续消融实验分析）
        error_tags = []
        if "This expression has type" in stderr:
            error_tags.append("OCamlTypeError")
        if "Syntax error" in stderr:
            error_tags.append("OCamlSyntaxError")
        if "not exhaustive" in stderr:
            error_tags.append("PatternMatchingNotExhaustive")
        if "Unbound value" in stderr:
            error_tags.append("UnboundValueOrMissingRec")
        if not error_tags:
            error_tags.append("OCamlCompileError")

        return False, clean_error, error_tags

    except subprocess.TimeoutExpired:
        return False, "Compilation timeout (Potential infinite type loop)", ["TIMEOUT"]
    except FileNotFoundError:
        return False, "OCaml compiler (ocamlc) not found. Run 'sudo apt-get install ocaml-nox'", ["ENV_ERROR"]
    except Exception as e:
        return False, f"Execution failed: {str(e)}", ["EXEC_ERROR"]
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            # 清理 ocamlc 产生的中间产物 (.cmi, .cmo) 保证工作区干净
            cmi_path = temp_file_path.replace(".ml", ".cmi")
            cmo_path = temp_file_path.replace(".ml", ".cmo")
            if os.path.exists(cmi_path): os.remove(cmi_path)
            if os.path.exists(cmo_path): os.remove(cmo_path)

def clean_generated_code(code: str, is_repair_mode: bool = False) -> str:
    if not code:
        return ""

    ocaml_blocks = re.findall(r'```ocaml\s*\n?(.*?)\n?```', code, re.DOTALL)
    if not ocaml_blocks:
        ocaml_blocks = re.findall(r'```\s*\n?(.*?)\n?```', code, re.DOTALL)
    final_code = ocaml_blocks[-1] if ocaml_blocks else code

    final_code = re.sub(r'<\|im_.*?\|>', '', final_code)
    final_code = final_code.strip()
    
    # 强力防御：擦除大模型习惯性带出来的末尾非法单分号
    if final_code.endswith(";"):
        # 如果不是 ;; 结尾，单分号极易改变函数返回类型为 unit 导致编译失败
        if not final_code.endswith(";;"):
            final_code = final_code[:-1].strip()
            
    return final_code

def select_best_candidate_on_cpu(candidates, java_src):
    """
    在子进程中运行。负责从多个候选里选出最好的，并返回检查详情。
    移除了 AST 强制过滤，改为以编译和逻辑校验为主。
    """
    first_candidate_info = None # 留底，如果全失败了返回第一个的结果
    
    for i, cand in enumerate(candidates):
        # 1. 语法检查 (调用外部 cargo check)
        is_syntax_ok, stderr, found_errors = verify_ocaml_syntax_with_log(cand)
        
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
        writer.writerow(['idx', 'cpp_code', 'nl_desc', 'all_candidates', 'passed_code', 'status'])

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for batch in tqdm(dataloader, desc="Generating & Verifying"):
                idxs, gtokens, cpp_codes, nl_descs = batch
                
                # --- Step A: GPU 批量生成 ---
                with torch.no_grad():
                    # batch_candidates: List[List[str]]
                    batch_candidates = batch_smart_generate(model, gtokens.to(device), cpp_codes, nl_descs, config, device)
                
                # --- Step B: 提交当前 Batch 到 CPU 并行校验 ---
                batch_futures = []
                for i in range(len(idxs)):
                    future = executor.submit(select_best_candidate_on_cpu, batch_candidates[i], cpp_codes[i])
                    batch_futures.append({
                        'idx': idxs[i].item(),
                        'cpp_code': cpp_codes[i],
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
                            item['cpp_code'],
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
    CONFIG_FILE = "/root/autodl-tmp/first_paper/cpp/stage3/stage3.yaml"
    run_inference(CONFIG_FILE)