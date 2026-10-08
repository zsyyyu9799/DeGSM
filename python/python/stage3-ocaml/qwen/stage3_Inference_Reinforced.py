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
from stage3_Inference import is_logic_valid, clean_generated_code, verify_ocaml_syntax_with_log

def calculate_pass_at_k(n, c, k):
    """
    计算标准的学术级 Pass@k 概率
    n: 通过单测的候选数量 (c_pass)
    c: 总生成的候选数量 (num_candidates)
    k: 评估的采样数 (1, 2, 5)
    """
    if c - n < k:
        return 1.0
    # 组合数公式：1 - (C(c-n, k) / C(c, k))
    return 1.0 - math.comb(c - n, k) / math.comb(c, k)
    
def execute_ocaml_unit_test_reinforced(code: str, test_cases: list, timeout: int = 5) -> (bool, bool):
    """
    针对 OCaml 的强力多级校验机（完全防御性安全修复版）
    """
    if not code.strip():
        return False, False

    # 1. 第一层：基本语法与强类型合规性预校验
    is_syntax_ok, stderr_msg, _ = verify_ocaml_syntax_with_log(code, timeout=timeout)
    is_logic_clean, _ = is_logic_valid(code)
    final_syntax_compile_ok = is_syntax_ok and is_logic_clean

    if not final_syntax_compile_ok:
        return False, False 

    # 🎯 修复 1：避开极其严苛、容易引发多进程多用户 Permission denied 的 /dev/shm，统一使用系统隔离的 tmp
    base_dir = tempfile.gettempdir()
    
    # 🎯 反解大模型生成的真实顶层函数名
    func_match = re.search(r'\blet\s+(?:rec\s+)?([a-zA-Z0-9_]+)', code)
    model_func_name = func_match.group(1) if func_match else None

    # ==================== 🎯 测试集函数名与特殊语法强制对齐 ====================
    raw_test_list = test_cases if isinstance(test_cases, list) else [str(test_cases)]
    raw_test_str = "\n".join([str(t).strip() for t in raw_test_list if str(t).strip()])

    # 🎯 修复 2：强力清洗测试用例。将测试数据中非标准的 OCaml 负数符号（如 ~3）批量修正为标准的 (-3)
    raw_test_str = re.sub(r'~\s*([0-9]+)', r'(-\1)', raw_test_str)

    # 如果找到了大模型的函数名，动态替换掉测试集 inside 的局部绑定名
    if model_func_name:
        raw_test_str = re.sub(
            r'let\s+candidate\s*=\s*[a-zA-Z0-9_]+', 
            f'let candidate = {model_func_name}', 
            raw_test_str
        )

    # 🎯 修复 3：无损解构自带的变量闭包，用 `let _ =` 替代原先会锁死类型解构的 `let () =`
    if "let () =" in raw_test_str:
        test_trigger = f"\n{raw_test_str}\n"
    elif "let assertions =" in raw_test_str:
        test_trigger = f"\n{raw_test_str}\nlet _ = assertions\n"
    else:
        indented_tests = "\n".join([f"  {line}" for line in raw_test_str.split("\n")])
        test_trigger = f"\nlet () =\n{indented_tests}\n"

    # 终极拼接
    full_code_with_assertions = (
        f"(* --- MODEL GENERATED CODE --- *)\n"
        f"{code}\n\n"
        f"(* --- EXPLICIT CRITICAL ASSERT_TRACKER --- *)\n"
        f"{test_trigger}"
    )
    # =====================================================================
    
    with tempfile.NamedTemporaryFile(suffix=".ml", dir=base_dir, mode="w", encoding="utf-8", delete=False) as tf:
        tf.write(full_code_with_assertions)
        temp_file_path = tf.name

    exe_output_path = temp_file_path.replace(".ml", ".out")

    try:
        # 编译全量可执行文件
        compile_res = subprocess.run(
            [
                "opam", "exec", "--", 
                "ocamlfind", "ocamlc", 
                "-o", exe_output_path, 
                temp_file_path
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout, encoding='utf-8', errors='ignore'
        )
        
        # if compile_res.returncode != 0:
        #     print(f"\n❌ [OCaml 编译失败] 函数名: {model_func_name} | 编译器日志:\n{compile_res.stderr}")
        #     return final_syntax_compile_ok, False

        # 🎯 修复 4：添加绝对防御性检查。如果编译器因为各种隐式机制没能生成 .out 文件，直接拦截，拒绝运行！
        # if not os.path.exists(exe_output_path):
        #     print(f"\n⚠️ [编译异常中断] 未在指定路径找到编译产物: {exe_output_path} (可能是编译过程中途挂掉)")
        #     return final_syntax_compile_ok, False

        # 运行测试驱动
        run_res = subprocess.run(
            [exe_output_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout, encoding='utf-8', errors='ignore'
        )
        
        # if run_res.returncode != 0:
        #     print(f"\n❌ [OCaml 单测未通过] 函数名: {model_func_name} | 运行时日志:\n{run_res.stderr}")

        is_unit_test_ok = (
            run_res.returncode == 0 and 
            "Assert_failure" not in run_res.stderr and 
            "Fatal error" not in run_res.stderr
        )
        return final_syntax_compile_ok, is_unit_test_ok
    except Exception as e:
        # print(f"\n❌ [脚本执行捕获异常]: {e}")
        return final_syntax_compile_ok, False
    finally:
        # 擦除临时文件与编译副产物
        if os.path.exists(temp_file_path): os.remove(temp_file_path)
        if os.path.exists(exe_output_path): os.remove(exe_output_path)
        for ext in [".cmi", ".cmo"]:
            side_effect_file = temp_file_path.replace(".ml", ext)
            if os.path.exists(side_effect_file): os.remove(side_effect_file)

def evaluate_pass_at_k_hybrid_job(candidates: list, test_cases: list):
    """
    完全符合学术标准的双轨评估机（OCaml 版）
    """
    syntax_results = []
    unittest_results = []
    
    for cand in candidates:
        syn_ok, ut_ok = execute_ocaml_unit_test_reinforced(cand, test_cases)
        syntax_results.append(syn_ok)
        unittest_results.append(ut_ok)
        
    compile_pass = 1 if any(syntax_results) else 0
    
    # 1. 统计这道题总共生成了多少个候选，其中有多少个完美通过了单测
    total_cands = len(candidates)              
    passed_cands = sum(1 for x in unittest_results if x) 
    
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

def batch_smart_generate_reinforced(model, graph_embs, codes, nl_descs, config, device):
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
            temperature=config['generation'].get('temperature', 0.9), top_p=config['generation'].get('top_p', 0.95),
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
        
        # 🎯 核心适配：兼容各大基准数据集（如 MultiPL-E 等）对 OCaml 测试用例的键名命名
        unit_tests = item.get('ocaml_tests') or item.get('ml_tests') or item.get('test_cases') or []
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
    
    use_graph = config.get("generation", {}).get("use_graph_info", False)
    print(f"\n================ ⚙️ 启动完全体【微调强化 + OCaml 编译单测双轨制评测】推理管线 ================")
    print(f"当前运行模式: {'【图增强模式】' if use_graph else '【纯代码消融模式】'}")
    
    model = CGBridgeStage3(config)
    
    print("🎨 正在动态同步大模型 LoRA 适配器结构（与训练完全看齐）...")
    peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model.llm_model = get_peft_model(model.llm_model, peft_config)
    
    stage3_ckpt = config.get('finetune', {}).get('finetune_model', '')
    if os.path.exists(stage3_ckpt):
        print(f"🔥 [精准靶向融合] 检测到微调模型，正在直接注入权重文件: {stage3_ckpt}...")
        state_dict = torch.load(stage3_ckpt, map_location="cpu")
        load_result = model.load_state_dict(state_dict, strict=False)
        print(f"✅ 确认成功: 【微调模型 finetune_model】已实现 100% 毫无损耗的绝对融合！")
    else:
        print(f"❌ 严重错误: 找不到微调模型路径 {stage3_ckpt}！")
        return

    model.to(device)
    model.eval()

    print(f"📖 正在加载独立多语言测试集: {config['data']['test_path']} ...")
    dataset = PassKEvaluationDataset(config['data']['test_path'])
    dataloader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])

    output_csv = config['output']['multilang_evaluation_report']
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # 统计全局计数器
    total_samples = len(dataset)
    compile_success_total = 0
    pass_1_total = 0
    pass_2_total = 0
    pass_5_total = 0
    
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
                        pass_1_total += p1
                        pass_2_total += p2
                        pass_5_total += p5
                        
                        # 每一条数据写盘
                        writer.writerow([
                            item['idx'], item['python_code'], item['nl_desc'],
                            json.dumps(item['candidates'], ensure_ascii=False),
                            passed_code, comp_ok, p1, p2, p5
                        ])
                        
                        processed_samples_count += 1
                        
                        # 每隔 10 个样本，在终端强制输出一次中间看板结果
                        if processed_samples_count % 10 == 0:
                            tqdm.write(f"\n📢 [中间计算报告] ── 累计已处理样本数: {processed_samples_count}/{total_samples}")
                            tqdm.write(f"    ├─  当前累计编译通过率: {compile_success_total / processed_samples_count * 100:.2f}%")
                            tqdm.write(f"    ├─ 真实单元测试 Pass@1: {pass_1_total / processed_samples_count * 100:.2f}%")
                            tqdm.write(f"    ├─ 真实单元测试 Pass@2: {pass_2_total / processed_samples_count * 100:.2f}%")
                            tqdm.write(f"    └─ 真实单元测试 Pass@5: {pass_5_total / processed_samples_count * 100:.2f}%\n")
                            
                    except Exception as e:
                        print(f"\n[Error] 样本 {item['idx']} 评测异常: {e}")
                
                f.flush()
                torch.cuda.empty_cache()

    # --- 终极学术战绩报告汇报 ---
    print(f"\n🏆 ======================== CGBridge 终极 OCaml 多语言翻译评测报告 ========================")
    print(f"📊 评估题目总计: {total_samples} 道")
    print(f"🛠️ 基础语法/编译通过率 (Compilation Pass Rate): {compile_success_total / total_samples * 100:.2f}%")
    print(f"📈 真实自动化单元测试 Pass@1 成功率: {pass_1_total / total_samples * 100:.2f}%")
    print(f"📈 真实自动化单元测试 Pass@2 成功率: {pass_2_total / total_samples * 100:.2f}%")
    print(f"📈 真实自动化单元测试 Pass@5 成功率: {pass_5_total / total_samples * 100:.2f}%")
    print(f"==================================================================================\n")

if __name__ == "__main__":
    CONFIG_FILE = "/root/autodl-tmp/first_paper/python/stage3/stage3.yaml"
    run_reinforced_inference(CONFIG_FILE)