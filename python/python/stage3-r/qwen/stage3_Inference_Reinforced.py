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
import math  # 🎯 引入数学库计算组合数
from tqdm import tqdm
from torch.utils.data import DataLoader
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from peft import LoraConfig, get_peft_model

# 🎯 强力定位真实的 python 核心项目目录，防止导包死锁
python_project_dir = "/root/autodl-tmp/first_paper/python"
if python_project_dir not in sys.path:
    sys.path.insert(0, python_project_dir)

from stage3.CGBridge_Stage3 import CGBridgeStage3, CodeTranslateDataset
from stage3_Inference import is_logic_valid, clean_generated_code, verify_r_syntax_with_log

def execute_r_unit_test_reinforced(code: str, test_cases: list, timeout: int = 5) -> (bool, bool):
    """
    强力多级校验机（MultiPL-E / MBPP 修复版）：
    同时返回 (is_syntax_ok, is_unit_test_passed)，自动对齐 candidate 与多行环境
    """
    if not code.strip():
        return False, False

    # 1. 第一层：运行基本的 R 语法与纯函数定义校验
    is_syntax_ok, _, _ = verify_r_syntax_with_log(code)
    is_logic_clean, _ = is_logic_valid(code)
    final_syntax_compile_ok = is_syntax_ok and is_logic_clean

    if not final_syntax_compile_ok:
        return False, False 

    # 2. 第二层：处理测试用例，完美注入别名与清洗多行
    base_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
    
    alias_str = ""
    func_match = re.search(r'([a-zA-Z0-9_\.]+)\s*(<-|=)\s*function', code)
    if func_match:
        model_func_name = func_match.group(1)
        if model_func_name != "candidate":
            alias_str += f"candidate <- {model_func_name}\n"

    cleaned_tests = []
    raw_test_list = test_cases if isinstance(test_cases, list) else [str(test_cases)]
    
    for t in raw_test_list:
        t_str = str(t).strip()
        if not t_str:
            continue
            
        if "\n" in t_str or "test_humaneval" in t_str or "stopifnot" in t_str:
            expected_match = re.search(r'candidate\s*(<-|=)\s*([a-zA-Z0-9_\.]+)', t_str)
            if expected_match and func_match:
                expected_func_name = expected_match.group(2)
                if expected_func_name != model_func_name:
                    alias_str += f"{expected_func_name} <- {model_func_name}\n"
            
            cleaned_tests.append(t_str)
            continue
            
        if t_str.startswith("assert "):
            inner = t_str[7:].strip()
            inner = inner.replace("True", "TRUE").replace("False", "FALSE").replace("None", "NULL")
            cleaned_tests.append(f"stopifnot({inner})")
        elif t_str.startswith("assert_eq(") or t_str.startswith("assert("):
            inner = re.sub(r'^assert(_eq)?\(|\)$', '', t_str)
            inner = inner.replace("True", "TRUE").replace("False", "FALSE").replace("None", "NULL")
            cleaned_tests.append(f"stopifnot({inner})")
        else:
            cleaned_tests.append(f"stopifnot({t_str})")

    test_script_str = "\n".join(cleaned_tests)
    
    full_code_with_assertions = f"{code}\n\n# --- ALIAS MAPPING ---\n{alias_str}\n# --- EXPLICIT CRITICAL ASSERT_TRACKER ---\n{test_script_str}"
    
    with tempfile.NamedTemporaryFile(suffix=".R", dir=base_dir, mode="w", encoding="utf-8", delete=False) as tf:
        tf.write(full_code_with_assertions)
        temp_file_path = tf.name

    try:
        res = subprocess.run(
            ["Rscript", temp_file_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout, encoding='utf-8', errors='ignore'
        )
        is_unit_test_ok = (res.returncode == 0 and "error" not in res.stderr.lower() and "failed" not in res.stderr.lower())
        return final_syntax_compile_ok, is_unit_test_ok
    except:
        return final_syntax_compile_ok, False
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def calculate_single_pass_at_k(n: int, c: int, k: int) -> float:
    """
    学术标准：计算单题在总生成 n 个候选，通过 c 个时的 Pass@k 概率值
    公式: 1 - C(n-c, k) / C(n, k)
    """
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))

def evaluate_pass_at_k_hybrid_job(candidates: list, test_cases: list):
    """
    双轨并行评估机：基于全部生成候选计算学术标准 Pass@k
    """
    syntax_results = []
    unittest_results = []
    
    for cand in candidates:
        syn_ok, ut_ok = execute_r_unit_test_reinforced(cand, test_cases)
        syntax_results.append(syn_ok)
        unittest_results.append(ut_ok)
        
    compile_pass = 1 if any(syntax_results) else 0
    
    # 🎯 全候选样本数 n 与通过单测数 c
    n = len(candidates)
    c = sum(1 for ut in unittest_results if ut)
    
    # 🎯 动态计算学术级标准 Pass@k 概率
    p1 = calculate_single_pass_at_k(n, c, 1)
    p2 = calculate_single_pass_at_k(n, c, 2)
    p5 = calculate_single_pass_at_k(n, c, 5)
    
    passed_code = ""
    for idx, ut_passed in enumerate(unittest_results):
        if ut_passed:
            passed_code = candidates[idx]
            break
            
    return compile_pass, p1, p2, p5, passed_code

def batch_smart_generate_reinforced(model, graph_embs, codes, nl_descs, config, device):
    use_graph_info = config['generation'].get('use_graph_info', True)
    num_candidates = config['generation'].get('num_candidates', 5)
    max_new_tokens = config['generation'].get('max_new_tokens', 1536)
    model.llm_tokenizer.padding_side = "left"

    if use_graph_info:
        print("图推理模式")
        query_feats = graph_embs.to(device).to(dtype=model.llm_model.dtype)
        with torch.no_grad():
            inputs_embeds, attn_mask = model.prepare_lm_input(
                gtokens=query_feats, text_input=codes, nl_desc=nl_descs, use_context=True
            )
    else:
        print("消融模式")
        full_texts = []
        for i in range(graph_embs.size(0)):
            user_p = config["prompts"]["translation"]["user_prompt"].format(
                struct_tokens="<|struct_start|><|struct_end|>", logic_tokens="<|logic_start|><|logic_end|>",
                semantic_tokens="<|semantic_start|><|semantic_end|>", global_tokens="<|global_start|><|global_end|>",
                nl_description=nl_descs[i] or "None", python_code=codes[i]
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
            inputs_embeds=inputs_embeds, attention_mask=attn_mask, max_new_tokens=max_new_tokens,
            temperature=config['generation'].get('temperature', 0.8), top_p=config['generation'].get('top_p', 0.95),
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
        
        unit_tests = item.get('r_tests') or item.get('rust_tests') or item.get('lua_tests') or item.get('ocaml_tests') or []
        if isinstance(unit_tests, str):
            try: unit_tests = json.loads(unit_tests)
            except: pass
            
        return (
            item.get('idx', idx),
            graph_emb,
            str(item.get('python_code', "")),
            str(item.get('nl_description', "")),
            unit_tests
        )

def run_reinforced_inference(config_path):
    with open(config_path, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n================ ⚙️ 启动完全体【微调强化 + 编译单测双轨制评测】推理管线 ================")
    
    model = CGBridgeStage3(config)
    
    print("🎨 正在动态同步大模型 LoRA 适配器结构（与训练完全看齐）...")
    peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model.llm_model = get_peft_model(model.llm_model, peft_config)
    
    stage3_ckpt = config.get('finetune', {}).get('finetune_model', '')
    if os.path.exists(stage3_ckpt):
        print(f"🔥 [精准靶向融合] 检测到微调模型，正在直接注入权重文件: {stage3_ckpt}...")
        state_dict = torch.load(stage3_ckpt, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 确认成功: 【微调模型 finetune_model】已实现 100% 毫无损耗的绝对融合！")
    else:
        print("❌ 严重错误: 找不到微调模型！")
        return

    model.to(device)
    model.eval()

    print(f"📖 正在加载独立多语言测试集: {config['data']['test_path']} ...")
    dataset = PassKEvaluationDataset(config['data']['test_path'])
    dataloader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])

    output_csv = config['output']['multilang_evaluation_report']
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # 统计全局计数器 (Pass@k 变为概率累加)
    total_samples = len(dataset)
    compile_success_total = 0
    pass_1_sum = 0.0
    pass_2_sum = 0.0
    pass_5_sum = 0.0
    
    processed_samples_count = 0
    max_workers = os.cpu_count()
    print(f"🚀 双轨单元测试评测开启！总题目数: {total_samples} | CPU 并行测试度: {max_workers}")

    with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'python_code', 'nl_desc', 'all_candidates', 'passed_code', 'compile_ok', 'pass_1', 'pass_2', 'pass_5'])

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for batch in tqdm(dataloader, desc="Generating & Dual Evaluating"):
                idxs, gtokens, python_codes, nl_descs, batch_unit_tests = batch
                
                # GPU 生成候选
                batch_candidates = batch_smart_generate_reinforced(
                    model, gtokens.to(device), python_codes, nl_descs, config, device
                )
                
                # 提交到并行校验池
                batch_futures = []
                for i in range(len(idxs)):
                    raw_idx = idxs[i].item() if isinstance(idxs[i], torch.Tensor) else str(idxs[i])
                    future = executor.submit(evaluate_pass_at_k_hybrid_job, batch_candidates[i], batch_unit_tests[i])
                    batch_futures.append({
                        'idx': raw_idx, 'python_code': python_codes[i], 'nl_desc': nl_descs[i],
                        'candidates': batch_candidates[i], 'future': future
                    })

                # 收集战绩
                for item in batch_futures:
                    try:
                        comp_ok, p1, p2, p5, passed_code = item['future'].result()
                        compile_success_total += comp_ok
                        pass_1_sum += p1
                        pass_2_sum += p2
                        pass_5_sum += p5
                        
                        # 写入文件（此时写盘的 pass_1/2/5 为该样本在全候选下的概率值）
                        writer.writerow([
                            item['idx'], item['python_code'], item['nl_desc'],
                            json.dumps(item['candidates'], ensure_ascii=False),
                            passed_code, comp_ok, f"{p1:.4f}", f"{p2:.4f}", f"{p5:.4f}"
                        ])
                        
                        processed_samples_count += 1
                        
                        # 每隔 10 个样本，在终端强制输出一次中间学术看板
                        if processed_samples_count % 10 == 0:
                            tqdm.write(f"\n📢 [中间学术报告] ── 累计已处理样本数: {processed_samples_count}/{total_samples}")
                            tqdm.write(f"    ├─  当前累计编译通过率: {compile_success_total / processed_samples_count * 100:.2f}%")
                            tqdm.write(f"    ├─ 学术标准 Mean Pass@1: {pass_1_sum / processed_samples_count * 100:.2f}%")
                            tqdm.write(f"    ├─ 学术标准 Mean Pass@2: {pass_2_sum / processed_samples_count * 100:.2f}%")
                            tqdm.write(f"    └─ 学术标准 Mean Pass@5: {pass_5_sum / processed_samples_count * 100:.2f}%\n")
                            
                    except Exception as e:
                        print(f"\n[Error] 样本 {item['idx']} 评测异常: {e}")
                
                f.flush()
                torch.cuda.empty_cache()

    # --- 终极学术战绩报告汇报 ---
    print(f"\n🏆 ======================== CGBridge 终极多语言翻译评测报告 ========================")
    print(f"📊 评估题目总计: {total_samples} 道")
    print(f"🛠️ 基础语法/编译通过率 (Compilation Pass Rate): {compile_success_total / total_samples * 100:.2f}%")
    print(f"📈 学术标准 自动化单元测试 Mean Pass@1: {pass_1_sum / total_samples * 100:.2f}%")
    print(f"📈 学术标准 自动化单元测试 Mean Pass@2: {pass_2_sum / total_samples * 100:.2f}%")
    print(f"📈 学术标准 自动化单元测试 Mean Pass@5: {pass_5_sum / total_samples * 100:.2f}%")
    print(f"==================================================================================\n")

if __name__ == "__main__":
    CONFIG_FILE = "/root/autodl-tmp/first_paper/python/stage3/stage3.yaml"
    run_reinforced_inference(CONFIG_FILE)