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
import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from peft import LoraConfig, get_peft_model
from openai import OpenAI

# 🎯 定位真实的 python 核心项目目录
python_project_dir = "/root/autodl-tmp/first_paper/python"
if python_project_dir not in sys.path:
    sys.path.insert(0, python_project_dir)

from stage3.CGBridge_Stage3 import CGBridgeStage3, CodeTranslateDataset
from stage3_Inference import is_logic_valid

def clean_generated_code(code: str) -> str:
    """
    针对 Llama 3.1 Base 模型的鲁棒后处理与多重物理切片机。
    通过动态字符串拼接生成三反引号，完美防止网页解析引擎中断。
    """
    if not code:
        return ""
        
    # 💡 动态构造三反引号字符串，防止 Canvas 渲染引擎提前截断
    bt = '`' * 3
        
    # 1. 物理移除 Base 模型特有的各类连体标签残骸 (例如：}#lang rustusestd -> }usestd)
    code = re.sub(r'#\s*lang\s*(?:rust|python|html|javascript|cpp|c|ruby)?', '\n', code, flags=re.IGNORECASE)

    # 2. 提取标准的 Markdown Rust 代码块（若存在）
    rust_blocks = re.findall(rf'{bt}(?:[Rr]ust|[Rr]sthon)\s*\n(.*?)\n{bt}', code, re.DOTALL)
    if rust_blocks:
        code = rust_blocks[0].strip()
    else:
        # 兼容处理未闭合的代码块
        unclosed = re.search(rf'{bt}(?:[Rr]ust|[Rr]sthon)\s*\n(.*)', code, re.DOTALL)
        if unclosed:
            code = unclosed.group(1).strip()
        
    # 3. 剥离前置非代码废话，准确定位 Rust 首个合法关键字作为起点
    # 包括常见的：fn, pub, use, struct, enum, impl, const, static 或注释
    start_match = re.search(r'(?m)^(?:pub\s+)?(?:fn|use|struct|enum|impl|const|static)\s+|^(?:\/\/\/|\/\/|\/\*)', code)
    if start_match:
        code = code[start_match.start():]

    # 4. 逐行精确扫描与逻辑熔断
    lines = code.splitlines()
    clean_lines = []
    
    # 熔断关键字集合：当 Base 模型在正文中开始输出这些非代码标记时，立即停止向下加载
    stop_patterns = [
        r'^\s*###',
        r'^\s*Question',
        r'^\s*Explanation',
        r'^\s*respond',
        r'^\s*Instruct',
        r'^\s*Solution',
        r'^\s*Title\s*[:>]',
        r'^\s*Answer\s*:',
        r'^\s*Example\s+usage',
        r'^\s*Output\s*:',
        r'^\s*Input\s*:',
        r'^\s*response\s+type',
        r'mathjax',
        r'html',
        rf'{bt}',
        r'^\s*def\s+',
        r'^\s*class\s+',
        r'^\s*import\s+(sys|math|ast|io)',
        r'^\s*function\s+',
        r'^\s*assert_eq!',
        r'^\s*println!',
        r'^\s*fn\s+main\s*\(\)' # 如果在主体函数后开始生成用于演示的 fn main() 则直接在此处截断
    ]
    
    for line in lines:
        stripped = line.strip()
        if not clean_lines and not stripped:
            continue
            
        # 发现任何熔断标志，物理停机
        triggered = False
        for pattern in stop_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                triggered = True
                break
                
        if triggered:
            break
        clean_lines.append(line)
        
    final_code_temp = "\n".join(clean_lines).strip()
    # 5. 基于大括号配对的终极闭包裁剪，只保留合法的 Rust 作用域，抛弃其后所有文字
    final_lines = []
    brace_count = 0
    has_opened = False
    
    for line in final_code_temp.splitlines():
        final_lines.append(line)
        for char in line:
            if char == '{':
                brace_count += 1
                has_opened = True
            elif char == '}':
                brace_count -= 1
        
        # 当大括号曾经打开过，且当前完全配对归零，意味着合法的 Rust 逻辑块完整结束，其后废话全部切除
        if has_opened and brace_count <= 0:
            break

    final_code = "\n".join(final_lines).strip()
    if brace_count > 0:
        final_code += "\n" + "}" * brace_count
    
    # 6. 字面量格式兼容与洗白
    final_code = final_code.replace("null", "None")
    return final_code.strip()
    
# ==================== 🦀 RUST 学术级沙盒编译与测试机 ====================
def execute_rust_unit_test(code: str, test_cases: list, timeout: int = 15) -> (bool, bool):
    """
    针对 HumanEval 特化的物理隔离 Rust 评测机
    """
    if not code.strip():
        return False, False

    # 1. 强力剥离 Markdown 标记
    code = code.replace("```rust", "").replace("```", "").strip()

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

        # 🎯 提取大模型生成的函数名
        func_match = re.search(r'pub\s+fn\s+([a-zA-Z0-9_]+)', code)
        if not func_match:
            func_match = re.search(r'fn\s+([a-zA-Z0-9_]+)', code)
        model_fn = func_match.group(1) if func_match else None

        source_lines = [code, "\n// ==================== TESTS ====================\n"]
        raw_test_list = test_cases if isinstance(test_cases, list) else [str(test_cases)]
        
        for idx, t in enumerate(raw_test_list):
            t_str = str(t).strip()
            
            # 🌟【降维打击】彻底消灭测试集开头的“内鬼”右花括号
            if t_str.startswith("}"):
                t_str = t_str.lstrip("}").strip()
                
            if not t_str:
                continue
            
            # 确定别名导入
            if model_fn and model_fn != "candidate":
                alias_line = f"    use super::{model_fn} as candidate;"
            else:
                alias_line = "    use super::*;"

            if "fn main()" in t_str:
                # 移除单测自带的重复 candidate 绑定声明，防止 Rust 报冲突警告
                t_str = re.sub(r'let\s+candidate\s*=\s*[a-zA-Z0-9_]+;', '', t_str)
                
                source_lines.append("#[cfg(test)]")
                source_lines.append(f"mod academic_test_space_{idx} {{")
                source_lines.append(alias_line)
                source_lines.append("    use super::*;")
                
                for line in t_str.splitlines():
                    source_lines.append(f"    {line}")
                
                source_lines.append("    #[test]")
                source_lines.append("    fn run_sandbox_test() {")
                source_lines.append("        main();")
                source_lines.append("    }")
                source_lines.append("}")
            else:
                # 碎片化单行断言
                source_lines.append("#[test]")
                source_lines.append(f"fn test_fragment_case_{idx}() {{")
                source_lines.append("    use super::*;")
                if model_fn and model_fn != "candidate" and "candidate" in t_str:
                    source_lines.append(f"    let candidate = super::{model_fn};")
                source_lines.append(f"    {t_str if t_str.endswith(';') else t_str + ';'}")
                source_lines.append("}")

        with open(os.path.join(src_dir, "lib.rs"), "w", encoding="utf-8") as f:
            f.write("\n".join(source_lines))

        # 1. 静态编译检查
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
            ["cargo", "test", "--offline", "--", "--nocapture"],
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


# ==================== 学术通用评估机 ====================
def calculate_pass_at_k(n, c, k):
    if c - n < k:
        return 1.0
    return 1.0 - math.comb(c - n, k) / math.comb(c, k)
    
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
        
        # 兼容读取不同的单测字段
        unit_tests = item.get('rust_tests') or item.get('python_tests') or item.get('lua_tests') or item.get('test') or []
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
    
    tgt_language = config.get('tgt_language', 'Rust')
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
                target_mods = ["c_attn", "c_proj", "c_fc", "q_proj", "kv_proj", "o_proj"]
            elif "qwen" in llm_type:
                print("💡 路由选择：检测到标准 MHA 架构 (Qwen/Llama)，挂载常规注意力靶向层...")
                target_mods = ["q_proj", "v_proj"]
            elif "llama" in llm_type or "llamabase" in llm_type:
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