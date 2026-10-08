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
from stage3_Inference import is_logic_valid, verify_racket_syntax_with_log

def clean_generated_code(code: str, is_repair_mode: bool = False) -> str:
    if not code:
        return ""

    # =========================================================================
    # 1. 物理强力截断：Base 模型套娃与退化垃圾清洗
    # =========================================================================
    degeneration_tokens = [
        "yaltyassistant", "assistant", "userlabiliracket", "@nate", 
        "vivastreet", "구글상위", "出品者", "Question:", "Output and Comments",
        "###", "руковод", "料無料", "駅徒歩", "yolc", "user"
    ]
    for token in degeneration_tokens:
        if token in code:
            code = code.split(token)[0]
            
    if "```" in code and code.count("```") % 2 != 0:
        code += "\n```"

    # 依次探测带语言标记和不带语言标记的 Markdown 块
    racket_blocks = re.findall(r'```[Rr]acket\s*\n?(.*?)\n?```', code, re.DOTALL)
    if not racket_blocks:
        racket_blocks = re.findall(r'```[Ll]isp\s*\n?(.*?)\n?```', code, re.DOTALL)
    if not racket_blocks:
        racket_blocks = re.findall(r'```\s*\n(.*?)\n```', code, re.DOTALL)
        
    if racket_blocks:
        final_code = racket_blocks[-1]
    else:
        final_code = code

    final_code = re.sub(r'<\|im_.*?\|>', '', final_code)
    final_code = re.sub(r'<\|.*?\|>', '', final_code) # 顺便洗掉类似 <|global_end|> 的残留
    
    # 清洗掉多余的换行与空格
    final_code = final_code.strip()

    final_code = re.sub(r'#lang\s+\w+\s*\n?', '', final_code)
    final_code = final_code.strip()
    
    # 重新在头部注入标准唯一的 #lang racket
    final_code = "#lang racket\n" + final_code
            
    return final_code


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
    
def execute_racket_unit_test_reinforced(code: str, test_cases: list, timeout: int = 5) -> (bool, bool):
    """
    针对 Racket 的强力多级校验机（MultiPL-E / MBPP 适配版）：
    同时返回 (is_syntax_ok, is_unit_test_passed)，自动对齐 candidate 别名与测试环境
    """
    if not code.strip():
        return False, False

    # 1. 第一层：利用 raco make 运行基本的 Racket 语法与词法预校验
    is_syntax_ok, stderr_msg, _ = verify_racket_syntax_with_log(code, timeout=timeout)
    is_logic_clean, _ = is_logic_valid(code)
    final_syntax_compile_ok = is_syntax_ok and is_logic_clean

    if not final_syntax_compile_ok:
        return False, False 

    # 2. 第二层：处理测试用例，完美注入别名
    base_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
    
    # 🎯 【核心重构 1】从模型生成的 Racket 代码中反解真实的函数名，绑定 candidate 别名
    # 匹配模式如：(define (func_name ...) 或 (define func_name ...
    alias_str = ""
    func_match = re.search(r'\(\s*define\s*\(\s*([a-zA-Z0-9_\-\?\!]+)', code)
    if not func_match:
        func_match = re.search(r'\(\s*define\s+([a-zA-Z0-9_\-\?\!]+)', code)
        
    if func_match:
        model_func_name = func_match.group(1)
        if model_func_name != "candidate":
            alias_str += f"\n(define candidate {model_func_name})\n"
    else:
        model_func_name = None

    cleaned_tests = []
    raw_test_list = test_cases if isinstance(test_cases, list) else [str(test_cases)]
    
    # 判断是否需要引入 Racket 官方单测库
    need_rackunit = False

    for t in raw_test_list:
        t_str = str(t).strip()
        if not t_str:
            continue
            
        # 🎯 【核心重构 2】如果是标准多行或已经包含断言的 Racket 单测脚本，直接放行
        if "\n" in t_str or "check-equal?" in t_str or "test-begin" in t_str:
            if "check-" in t_str:
                need_rackunit = True
            
            # 如果单测预期的名字和模型定义的不一样，动态让预期的名字也指向大模型生成的函数
            expected_match = re.search(r'candidate\s*=\s*([a-zA-Z0-9_\-\?\!]+)', t_str)
            if expected_match and model_func_name:
                expected_func_name = expected_match.group(1)
                if expected_func_name != model_func_name:
                    alias_str += f"(define {expected_func_name} {model_func_name})\n"
            
            cleaned_tests.append(t_str)
            continue
            
        # 🎯 【核心重构 3】处理可能残留下来的 Python 风格断言或极简断言，转换为 Racket 的 rackunit 形式
        if t_str.startswith("assert "):
            need_rackunit = True
            inner = t_str[7:].strip()
            # 简单将等号或标准结构进行基础对齐转义
            if "==" in inner:
                parts = inner.split("==")
                cleaned_tests.append(f"(check-equal? {parts[0].strip()} {parts[1].strip()})")
            else:
                cleaned_tests.append(f"(check-true {inner})")
        else:
            # 默认如果是包裹在括号里的原生 Racket 测试表达式 (check-equal? ...) 
            if "check-" in t_str:
                need_rackunit = True
            cleaned_tests.append(t_str)

    # 3. 终极组装：Racket 必须保证 #lang racket 在最顶行
    # 剥离原代码中可能重复的 #lang racket
    pure_code = re.sub(r'#lang racket\n?', '', code).strip()
    
    rackunit_header = "(require rackunit)\n" if need_rackunit else ""
    test_script_str = "\n".join(cleaned_tests)
    
    full_code_with_assertions = (
        f"#lang racket\n"
        f"{rackunit_header}"
        f"{pure_code}\n\n"
        f";; --- ALIAS MAPPING ---\n{alias_str}\n"
        f";; --- EXPLICIT CRITICAL ASSERT_TRACKER ---\n{test_script_str}"
    )
    
    with tempfile.NamedTemporaryFile(suffix=".rkt", dir=base_dir, mode="w", encoding="utf-8", delete=False) as tf:
        tf.write(full_code_with_assertions)
        temp_file_path = tf.name

    try:
        # 使用 racket 命令行工具执行完整的含有单测的代码
        res = subprocess.run(
            ["racket", temp_file_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout, encoding='utf-8', errors='ignore'
        )
        
        # Racket 在 rackunit 断言失败或运行崩溃时，会向 stderr 输出错误并返回非 0 状态码
        # 检查 stderr 中是否包含 FAILURE（rackunit 报错关键字）或 Exception
        is_unit_test_ok = (
            res.returncode == 0 and 
            "failure" not in res.stderr.lower() and 
            "exception" not in res.stderr.lower()
        )
        return final_syntax_compile_ok, is_unit_test_ok
    except:
        return final_syntax_compile_ok, False
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def evaluate_pass_at_k_hybrid_job(candidates: list, test_cases: list):
    """
    完全符合学术标准的双轨评估机
    """
    syntax_results = []
    unittest_results = []
    
    for cand in candidates:
        syn_ok, ut_ok = execute_racket_unit_test_reinforced(cand, test_cases)
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
        
        # 兼容各大基准数据集（如 MultiPL-E 等）对 Racket 测试用例的键名命名
        unit_tests = item.get('racket_tests') or item.get('rkt_tests') or item.get('test_cases') or []
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
    
    tgt_language = config.get('tgt_language', 'Racket')
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
                target_mods = ["c_attn", "c_proj", "c_fc", "q_proj", "kv_proj", "o_proj"]  # StarCoder 经典的合并注意力机制
            elif "qwen" in llm_type:
                print("💡 路由选择：检测到标准 MHA 架构 (Qwen/Llama)，挂载常规注意力靶向层...")
                target_mods = ["q_proj", "v_proj"]
            elif "llama" in llm_type:
                target_mods = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
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
                batch_candidates = batch_smart_generate(
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