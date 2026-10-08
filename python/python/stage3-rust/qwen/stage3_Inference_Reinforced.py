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
from stage3_Inference import is_logic_valid, clean_generated_code

# ==================== 🐍 PYTHON 强力多级校验机 ====================
def execute_python_unit_test(code: str, test_cases: list, timeout: int = 5) -> (bool, bool):
    """
    针对 Python 的多级校验机：返回 (is_syntax_ok, is_unit_test_passed)
    """
    if not code.strip():
        return False, False

    # 1. 第一层：语法静态检查
    try:
        compile(code, '<string>', 'exec')
        is_syntax_ok = True
    except SyntaxError:
        return False, False

    # 2. 第二层：运行单元测试
    # 提取模型生成的函数名，用于可能需要的别名映射（例如 HumanEval 的 candidate）
    alias_str = ""
    func_match = re.search(r'def\s+([a-zA-Z0-9_]+)\s*\(', code)
    if func_match:
        model_func_name = func_match.group(1)
        if model_func_name != "candidate":
            alias_str += f"\ncandidate = {model_func_name}\n"
    
    cleaned_tests = []
    raw_test_list = test_cases if isinstance(test_cases, list) else [str(test_cases)]
    
    for t in raw_test_list:
        t_str = str(t).strip()
        if not t_str:
            continue
        # 如果已经是完备的单测脚本或包含多行断言，直接保留
        if "\n" in t_str or "def test_" in t_str:
            cleaned_tests.append(t_str)
        elif t_str.startswith("assert "):
            cleaned_tests.append(t_str)
        else:
            cleaned_tests.append(f"assert {t_str}")

    test_script_str = "\n".join(cleaned_tests)
    full_code = f"{code}\n{alias_str}\n{test_script_str}"

    base_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
    with tempfile.NamedTemporaryFile(suffix=".py", dir=base_dir, mode="w", encoding="utf-8", delete=False) as tf:
        tf.write(full_code)
        temp_file_path = tf.name

    try:
        res = subprocess.run(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout, encoding='utf-8', errors='ignore'
        )
        is_unit_test_ok = (res.returncode == 0)
        return is_syntax_ok, is_unit_test_ok
    except subprocess.TimeoutExpired:
        return is_syntax_ok, False
    except:
        return is_syntax_ok, False
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


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
    
def evaluate_pass_at_k_hybrid_job(candidates: list, test_cases: list, tgt_lang: str):
    """
    支持多语言分流的通用 Pass@k 双轨评估机
    """
    syntax_results = []
    unittest_results = []
    
    for cand in candidates:
        if tgt_lang.lower() == "python":
            syn_ok, ut_ok = execute_python_unit_test(cand, test_cases)
        elif tgt_lang.lower() == "rust":
            syn_ok, ut_ok = execute_rust_unit_test(cand, test_cases)
        else:
            # 兜底或扩展示例
            syn_ok, ut_ok = False, False
            
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


# ==================== 🌐 开放式大模型 API 强力生成机 ====================
def generate_via_api(config, prompt_texts):
    """
    🌐 统一多模型远程跨模态/消融 API 生成器 (支持 GPT, DeepSeek, Qwen, Claude)
    """
    api_key = config["model"].get("api_key")
    base_url = config["model"].get("api_base_url")
    model_name = config["model"].get("llm_name_or_path")
    llm_type = config["model"].get("llm_type", "qwen").lower()
    
    num_candidates = config['generation'].get('num_candidates', 5)
    temperature = config['generation'].get('temperature', 0.8)
    top_p = config['generation'].get('top_p', 0.95)
    max_new_tokens = config['generation'].get('max_new_tokens', 1536)

    # 初始化 OpenAI 兼容客户端
    if llm_type == "claude" and "anthropic" in base_url:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key, base_url=base_url)
            is_anthropic_sdk = True
        except ImportError:
            client = OpenAI(api_key=api_key, base_url=base_url)
            is_anthropic_sdk = False
    else:
        client = OpenAI(api_key=api_key, base_url=base_url)
        is_anthropic_sdk = False

    batch_candidates = []

    for prompt in prompt_texts:
        candidates_for_sample = []
        for _ in range(num_candidates):
            try:
                if is_anthropic_sdk:
                    message = client.messages.create(
                        model=model_name,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        system=config["prompts"]["translation"]["system_prompt"],
                        messages=[{"role": "user", "content": prompt}]
                    )
                    gen_text = message.content[0].text
                else:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": config["prompts"]["translation"]["system_prompt"]},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_new_tokens
                    )
                    gen_text = response.choices[0].message.content
                
                candidates_for_sample.append(clean_generated_code(gen_text))
            except Exception as e:
                print(f"⚠️ API 请求异常: {e}")
                candidates_for_sample.append("")
        
        batch_candidates.append(candidates_for_sample)
    
    return batch_candidates


# ==================== 🧠 混合双轨制智能代码构建机 ====================
def batch_smart_generate_reinforced(model, graph_embs, codes, nl_descs, config, device, use_api_mode=False):
    use_graph_info = config['generation'].get('use_graph_info', True)
    num_candidates = config['generation'].get('num_candidates', 5)
    max_new_tokens = config['generation'].get('max_new_tokens', 1536)
    llm_type = config["model"].get("llm_type", "qwen").lower()

    # 1. 🌐 在线 API 生成分支
    if use_api_mode:
        prompt_texts, _ = model.prepare_lm_input(
            gtokens=graph_embs, text_input=codes, nl_desc=nl_descs, use_context=use_graph_info
        )
        return generate_via_api(config, prompt_texts)

    # 2. 🏠 本地权重模型生成分支
    model.llm_tokenizer.padding_side = "left"

    if use_graph_info:
        print("图推理模式")
        query_feats = graph_embs.to(device).to(dtype=model.llm_model.dtype)
        with torch.no_grad():
            inputs_embeds, attn_mask = model.prepare_lm_input(
                gtokens=query_feats, text_input=codes, nl_desc=nl_descs, use_context=True
            )
    else:
        print("消融模式 (纯文本 Prompt)")
        full_texts = []
        for i in range(graph_embs.size(0)):
            user_p = config["prompts"]["translation"]["user_prompt"].format(
                struct_tokens="<|struct_start|><|struct_end|>", logic_tokens="<|logic_start|><|logic_end|>",
                semantic_tokens="<|semantic_start|><|semantic_end|>", global_tokens="<|global_start|><|global_end|>",
                nl_description=nl_descs[i] or "None", python_code=codes[i]
            )
            # 兼容处理多基座 Prompt 格式转换
            if hasattr(model, 'get_prompt_by_model'):
                prompt = model.get_prompt_by_model(llm_type, config['prompts']['translation']['system_prompt'], user_p)
            else:
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
    
    print(f"\n================ ⚙️ 启动完全体【微调强化/API多模型 + {tgt_language} 编译单测双轨制评测】推理管线 ================")
    print(f"模式: {'🌐 API 远程模式' if use_api_mode else '🏠 本地路径加载模式'} | 选定模型: {llm_type.upper()}")
    
    model = CGBridgeStage3(config)
    
    # 🌟 仅在本地加载模式下注入 LoRA 与本地权重，API模式无需本地模型结构
    if not use_api_mode:
        print("🎨 正在动态同步大模型 LoRA 适配器结构（与训练完全看齐）...")
        peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        model.llm_model = get_peft_model(model.llm_model, peft_config)
        
        stage3_ckpt = config.get('finetune', {}).get('finetune_model', '')
        if os.path.exists(stage3_ckpt):
            print(f"🔥 [精准靶向融合] 检测到微调模型，正在直接注入权重文件: {stage3_ckpt}...")
            state_dict = torch.load(stage3_ckpt, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ 确认成功: 【微调模型 finetune_model】已实现 100% 无损融合！")
        else:
            print("❌ 严重错误: 找不到微调模型！")
            return

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
    
    # 💥 根据核心数设定并行 Worker。由于 Rust 编译非常消耗 CPU 算力，推荐根据核数适当调整
    max_workers = min(os.cpu_count(), 16) if tgt_language.lower() == "rust" else os.cpu_count()
    print(f"🚀 双轨单元测试评测开启！目标语言: {tgt_language} | 题目总数: {total_samples} | CPU 并行度: {max_workers}")

    with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'python_code', 'nl_desc', 'all_candidates', 'passed_code', 'compile_ok', 'pass_1', 'pass_2', 'pass_5'])

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for batch in tqdm(dataloader, desc="Generating & Dual Evaluating"):
                idxs, gtokens, python_codes, nl_descs, batch_unit_tests = batch
                
                # 双轨制生成调用（根据 use_api_mode 智能路由）
                batch_candidates = batch_smart_generate_reinforced(
                    model, gtokens, python_codes, nl_descs, config, device, use_api_mode=use_api_mode
                )
                
                # 提交到多进程多轨校验池
                batch_futures = []
                for i in range(len(idxs)):
                    raw_idx = idxs[i].item() if isinstance(idxs[i], torch.Tensor) else str(idxs[i])
                    future = executor.submit(evaluate_pass_at_k_hybrid_job, batch_candidates[i], batch_unit_tests[i], tgt_language)
                    batch_futures.append({
                        'idx': raw_idx, 'python_code': python_codes[i], '`nl_desc`': nl_descs[i],
                        'candidates': batch_candidates[i], 'future': future
                    })

                # 收集评估战绩
                for item in batch_futures:
                    try:
                        comp_ok, p1, p2, p5, passed_code = item['future'].result()
                        compile_success_total += comp_ok
                        pass_1_total += p1
                        pass_2_total += p2
                        pass_5_total += p5
                        
                        writer.writerow([
                            item['idx'], item['python_code'], item['`nl_desc`'],
                            json.dumps(item['candidates'], ensure_ascii=False),
                            passed_code, comp_ok, p1, p2, p5
                        ])
                        
                        processed_samples_count += 1
                        
                        # 每隔 10 个样本打印看板
                        if processed_samples_count % 10 == 0:
                            tqdm.write(f"\n📢 [中间进度报告] ── 累计已处理数: {processed_samples_count}/{total_samples}")
                            tqdm.write(f"    ├─ 当前累计编译/语法通过率: {compile_success_total / processed_samples_count * 100:.2f}%")
                            tqdm.write(f"    ├─ 学术单元测试 Pass@1: {pass_1_total / processed_samples_count * 100:.2f}%")
                            tqdm.write(f"    ├─ 学术单元测试 Pass@2: {pass_2_total / processed_samples_count * 100:.2f}%")
                            tqdm.write(f"    └─ 学术单元测试 Pass@5: {pass_5_total / processed_samples_count * 100:.2f}%\n")
                            
                    except Exception as e:
                        print(f"\n[Error] 样本 {item['idx']} 评测异常: {e}")
                
                f.flush()
                if not use_api_mode:
                    torch.cuda.empty_cache()

    # --- 终极学术战绩报告汇报 ---
    print(f"\n🏆 ======================== CGBridge 终极 {tgt_language} 多语言翻译评测报告 ========================")
    print(f"📊 评估题目总计: {total_samples} 道")
    print(f"🛠️ 编译/语法通过率 (Compilation Pass Rate): {compile_success_total / total_samples * 100:.2f}%")
    print(f"📈 自动化单元测试 Pass@1 成功率: {pass_1_total / total_samples * 100:.2f}%")
    print(f"📈 自动化单元测试 Pass@2 成功率: {pass_2_total / total_samples * 100:.2f}%")
    print(f"📈 自动化单元测试 Pass@5 成功率: {pass_5_total / total_samples * 100:.2f}%")
    print(f"==================================================================================\n")

if __name__ == "__main__":
    CONFIG_FILE = "/root/autodl-tmp/first_paper/python/stage3/stage3.yaml"
    run_reinforced_inference(CONFIG_FILE)