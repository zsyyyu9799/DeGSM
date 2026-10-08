import os
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "FALSE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import math
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
from openai import OpenAI

# 🎯 强力定位真实的 python 核心项目目录，防止导包死锁
python_project_dir = "/root/autodl-tmp/first_paper/python"
if python_project_dir not in sys.path:
    sys.path.insert(0, python_project_dir)

from stage3.CGBridge_Stage3 import CGBridgeStage3, CodeTranslateDataset
from stage3_Inference import is_logic_valid, clean_generated_code, verify_lua_syntax_with_log

def execute_lua_unit_test_reinforced(code: str, test_cases: list, timeout: int = 5) -> (bool, bool):
    """
    针对 Lua 的强力多级校验机：同时返回 (is_syntax_ok, is_unit_test_passed)
    """
    if not code.strip():
        return False, False

    # 1. 第一层：运行基本的 Lua 语法与纯函数定义校验
    is_syntax_ok, _, _ = verify_lua_syntax_with_log(code)
    is_logic_clean, _ = is_logic_valid(code)
    final_syntax_compile_ok = is_syntax_ok and is_logic_clean

    if not final_syntax_compile_ok:
        return False, False 

    # 2. 第二层：处理测试用例，完美注入别名与清洗多行
    base_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
    
    alias_str = ""
    func_match = re.search(r'(?:local\s+)?function\s+([a-zA-Z0-9_]+)', code)
    if not func_match:
        func_match = re.search(r'([a-zA-Z0-9_]+)\s*=\s*function', code)
        
    if func_match:
        model_func_name = func_match.group(1)
        if model_func_name != "candidate":
            alias_str += f"candidate = {model_func_name}\n"
    else:
        model_func_name = None

    cleaned_tests = []
    raw_test_list = test_cases if isinstance(test_cases, list) else [str(test_cases)]
    
    for t in raw_test_list:
        t_str = str(t).strip()
        if not t_str:
            continue
            
        if "\n" in t_str or "test_humaneval" in t_str or "assert(" in t_str:
            expected_match = re.search(r'candidate\s*=\s*([a-zA-Z0-9_]+)', t_str)
            if expected_match and model_func_name:
                expected_func_name = expected_match.group(1)
                if expected_func_name != model_func_name:
                    alias_str += f"{expected_func_name} = {model_func_name}\n"
            cleaned_tests.append(t_str)
            continue
            
        if t_str.startswith("assert "):
            inner = t_str[7:].strip()
            inner = inner.replace("True", "true").replace("False", "false").replace("None", "nil").replace("!=", "~=")
            cleaned_tests.append(f"assert({inner})")
        elif t_str.startswith("assert_eq("):
            inner = re.sub(r'^assert_eq\(|\)$', '', t_str)
            inner = inner.replace("True", "true").replace("False", "false").replace("None", "nil").replace("!=", "~=")
            if "," in inner:
                parts = inner.split(",", 1)
                cleaned_tests.append(f"assert(({parts[0].strip()}) == ({parts[1].strip()}))")
            else:
                cleaned_tests.append(f"assert({inner})")
        else:
            cleaned_tests.append(f"assert({t_str})")

    test_script_str = "\n".join(cleaned_tests)
    full_code_with_assertions = f"{code}\n\n-- --- ALIAS MAPPING ---\n{alias_str}\n-- --- EXPLICIT CRITICAL ASSERT_TRACKER ---\n{test_script_str}"
    
    with tempfile.NamedTemporaryFile(suffix=".lua", dir=base_dir, mode="w", encoding="utf-8", delete=False) as tf:
        tf.write(full_code_with_assertions)
        temp_file_path = tf.name

    try:
        res = subprocess.run(
            ["lua", temp_file_path],
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
            
def calculate_pass_at_k(n, c, k):
    if c - n < k:
        return 1.0
    return 1.0 - math.comb(c - n, k) / math.comb(c, k)
    
def evaluate_pass_at_k_hybrid_job(candidates: list, test_cases: list):
    syntax_results = []
    unittest_results = []
    
    for cand in candidates:
        syn_ok, ut_ok = execute_lua_unit_test_reinforced(cand, test_cases)
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

def batch_smart_generate_reinforced(model, graph_embs, codes, nl_descs, config, device):
    use_graph_info = config['generation'].get('use_graph_info', True)
    num_candidates = config['generation'].get('num_candidates', 5)
    max_new_tokens = config['generation'].get('max_new_tokens', 1024)
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
        
        unit_tests = item.get('lua_tests') or item.get('r_tests') or item.get('rust_tests') or item.get('ocaml_tests') or []
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
    with open(config_path, 'r', encoding='utf-8') as f: 
        config = yaml.safe_load(f)
    
    tgt_language = config.get('tgt_language', 'Lua')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    use_api_mode = config["model"].get("use_api_mode", False)
    llm_type = config["model"].get("llm_type", "qwen").lower()
    use_graph = config.get("generation", {}).get("use_graph_info", False) # 💡 动态读取 YAML 配置中的消融开关
    
    print(f"\n================ ⚙️ 启动完全体【消融实验：纯代码/纯文本基座推理管线】================")
    print(f"当前运行模式: {'【图增强模式】' if use_graph else '【纯代码消融模式】'} | 目标基座: {llm_type.upper()}")
    
    import gc
    gc.collect()
    torch.cuda.empty_cache() # 强制排空显存残留
    
    # 1. 实例化基本模型架构
    model = CGBridgeStage3(config)
    
    # ========================================================================
    # 🛠️ 消融改造一：根据 use_graph_info 状态自适应控锁二阶段状态
    # ========================================================================
    if use_graph:
        print("🚚 正在激活图增强评估流，锁死 Q-Former 推理期状态...")
        if hasattr(model, "stage2_bridge") and model.stage2_bridge is not None:
            model.stage2_bridge.eval()
            if hasattr(model.stage2_bridge, "Qformer"):
                model.stage2_bridge.Qformer.eval()
                model.stage2_bridge.Qformer.bert.config.gradient_checkpointing = False
    else:
        print("⚠️ [消融警报开启]：图增强流已被物理切断，Q-Former 特征提取在前向传播中将被跳过！")

    # ========================================================================
    # 🛠️ 消融改造二：绕过 LoRA 结构挂载与微调字典加载，直接让原生大模型基座登舰
    # ========================================================================
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
            if "starcoder" in llm_type:
                peft_config = LoraConfig(
                    r=16,                           
                    lora_alpha=32,                  
                    target_modules=target_mods,  
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    modules_to_save=["lm_head"]     # 解除对输出头的锁定，加速特定语言分布的对齐
                )
            else:
                peft_config = LoraConfig(
                    r=8,                           
                    lora_alpha=16,                 
                    target_modules=target_mods,  
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
            model.llm_model = get_peft_model(model.llm_model, peft_config)
            
            stage3_ckpt = config.get('finetune', {}).get('finetune_model', '')
            if os.path.exists(stage3_ckpt):
                print(f"🔥 [精准靶向融合] 检测到微调模型，正在调用安全融合机制: {stage3_ckpt}...")
                res = model.load_tuned_stage3_weights(stage3_ckpt)
                if res is not None:
                    print(f"✅ 增量参数同步报告 ── 缺失项总数: {len(res.missing_keys)}")
            else:
                print(f"❌ 严重错误: 找不到微调模型路径 {stage3_ckpt}！")
                return
        else:
            # 纯文本消融模式：拒绝任何微调参数污染，直接以 100% 纯原生底座启动加载
            print("🎨 正在以 [纯原生大模型基座] 模式启动硬件加载...")
            
        model.to(device)
        model.eval()

    # ========================================================================
    # 📦 3. 准备独立多语言测试集并加载
    # ========================================================================
    print(f"📖 正在加载独立多语言测试集: {config['data']['test_path']} ...")
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
    print(f"🚀 双轨单元测试评测开启！目标语言: {tgt_language} | 题目总数: {total_samples} | CPU 并行度: {max_workers}")

    with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'python_code', 'nl_desc', 'all_candidates', 'passed_code', 'compile_ok', 'pass_1', 'pass_2', 'pass_5'])

        # 全局锁死不计算梯度
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating & Dual Evaluating"):
                idxs, gtokens, python_codes, nl_descs, batch_unit_tests = batch
                
                # 1. 大模型推理生成候选 (显存占用高峰)
                batch_candidates = batch_smart_generate_reinforced(
                    model, gtokens, python_codes, nl_descs, config, device)
                
                # 2. 局部开启多进程池，处理完当前 Batch 立即释放，彻底阻断异步积压
                batch_futures = []
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    for i in range(len(idxs)):
                        raw_idx = idxs[i].item() if isinstance(idxs[i], torch.Tensor) else str(idxs[i])
                        future = executor.submit(evaluate_pass_at_k_hybrid_job, batch_candidates[i], batch_unit_tests[i])
                        batch_futures.append({
                            'idx': raw_idx, 'python_code': python_codes[i], 'nl_desc': nl_descs[i],
                            'candidates': batch_candidates[i], 'future': future
                        })

                    # 3. 阻塞式等待当前 Batch 所有的 CPU 评测任务结束
                    for item in batch_futures:
                        try:
                            comp_ok, p1, p2, p5, passed_code = item['future'].result()
                            compile_success_total += comp_ok
                            pass_1_total += p1
                            pass_2_total += p2
                            pass_5_total += p5
                            
                            writer.writerow([
                                item['idx'], item['python_code'], item['nl_desc'],
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
                
                # 4. 彻底斩断引用链，强制释放显存
                del batch_candidates, batch_futures, idxs, gtokens
                import gc
                gc.collect()
                if not use_api_mode:
                    torch.cuda.empty_cache()

    # --- 终极学术战绩报告汇报 ---
    print(f"\n🏆 ======================== CGBridge 终极 {tgt_language} 多语言翻译评测报告 ========================")
    print(f"📊 评估题目总计: {total_samples} 道")
    print(f"🛠️ 基础语法/编译通过率 (Compilation Pass Rate): {compile_success_total / total_samples * 100:.2f}%")
    print(f"📈 真实自动化单元测试 Pass@1 成功率: {pass_1_total / total_samples * 100:.2f}%")
    print(f"📈 真实自动化单元测试 Pass@2 成功率: {pass_2_total / total_samples * 100:.2f}%")
    print(f"📈 真实自动化单元测试 Pass@5 成功率: {pass_5_total / total_samples * 100:.2f}%")
    print(f"==================================================================================\n")

if __name__ == "__main__":
    CONFIG_FILE = "/root/autodl-tmp/first_paper/python/stage3/stage3.yaml"
    run_reinforced_inference(CONFIG_FILE)