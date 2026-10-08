import os
import csv
import json
import re
import random
import subprocess
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ==========================================
# 1. 动态输入生成器 (根据你的函数特征可自行扩展)
# ==========================================
def generate_test_inputs(python_code: str, num_tests: int = 5) -> list:
    """
    自适应探测 Python 代码并生成测试输入。
    论文标准：通常需要多组不同的输入来防止大模型通过 trick 硬编码硬凑正确率。
    """
    inputs = []
    
    # 简单的静态分析：通过正则探测 Python 函数定义中的参数个数
    match = re.search(r'def\s+\w+\s*\(([^)]*)\)', python_code)
    num_args = 1
    if match:
        args_str = match.group(1).strip()
        if args_str:
            num_args = len([a for a in args_str.split(',') if a.strip() and 'self' not in a])

    # 简单策略：根据参数个数，生成多组随机基础类型组合（整数、浮点、简单数组）
    for _ in range(num_tests):
        test_case = []
        for _ in range(num_args):
            choice = random.choice(['int', 'float', 'list'])
            if choice == 'int':
                test_case.append(random.randint(-50, 100))
            elif choice == 'float':
                test_case.append(round(random.uniform(-10.0, 50.0), 2))
            else:
                test_case.append([random.randint(1, 20) for _ in range(random.randint(2, 5))])
        
        # 如果只有一个参数，解包，多个参数保持为 list
        inputs.append(test_case[0] if num_args == 1 else test_case)
        
    return inputs

# ==========================================
# 2. 核心：单样本双端差分测试隔离沙箱
# ==========================================
def evaluate_single_sample(idx, python_code, candidates_json, test_inputs):
    """
    在独立的子进程中运行：执行 Python 源码获取真值 -> 组装 Lua 运行单测 -> 评测 5 个候选
    """
    try:
        candidates = json.loads(candidates_json)
    except Exception:
        return idx, [False] * 5, "JSON_ERROR"

    # --- Step A: 提取 Python 的主函数名并执行获取 Ground Truth ---
    py_func_match = re.search(r'def\s+(\w+)\s*\(', python_code)
    if not py_func_match:
        return idx, [False] * len(candidates), "NO_PYTHON_FUNCTION"
    py_func_name = py_func_match.group(1)

    # 构造临时的 Python 脚本来跑标答
    ground_truth = []
    py_success = True
    
    for inp in test_inputs:
        # 动态组装一段包装代码
        args_str = json.dumps(inp) if isinstance(inp, list) else str(inp)
        # 如果是多参数，去掉外面 json 的中括号包裹
        if isinstance(inp, list) and len(inp) > 1:
            args_str = args_str[1:-1]
            
        py_runner = f"{python_code}\n\nimport json\nprint(json.dumps({py_func_name}({args_str})))"
        
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(py_runner)
            py_path = f.name
        try:
            res = subprocess.run(["python3", py_path], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                ground_truth.append(res.stdout.strip())
            else:
                py_success = False
                break
        except Exception:
            py_success = False
            break
        finally:
            if os.path.exists(py_path): os.remove(py_path)

    if not py_success or len(ground_truth) != len(test_inputs):
        return idx, [False] * len(candidates), "PYTHON_ORACLE_CRASH"

    # --- Step B: 逐个测试大模型生成的 Lua 候选代码 ---
    candidate_results = []
    
    for cand in candidates:
        if not cand.strip():
            candidate_results.append(False)
            continue
            
        # 提取 Lua 的函数名（通常模型会保持与 Python 相同的函数名）
        # 兜底：如果模型自己改了名字或者没写 function，直接判错
        lua_func_match = re.search(r'function\s+(\w+)\s*\(', cand)
        if not lua_func_match:
            candidate_results.append(False)
            continue
        lua_func_name = lua_func_match.group(1)

        # 构建内嵌断言的 Lua 脚本
        lua_script = cand + "\n\n-- --- AUTOMATED DIFFERENTIAL TESTING ---\n"
        lua_script += "local json = require('json')\n"  # 提示：环境中需要安装 lua-cjson 或类似库，如果没有可用原生拼接
        
        cand_passed = True
        for inp, py_expected in zip(test_inputs, ground_truth):
            # 将 Python 传入的数据格式化为 Lua 的字面量（处理 list 映射为 table 等问题）
            if isinstance(inp, list):
                # 简单映射：[1, 2] -> {1, 2}
                lua_args = "{" + ", ".join([str(x) for x in inp]) + "}"
            else:
                lua_args = str(inp)
                
            # 处理多参数展开
            if isinstance(inp, list) and len(inp) > 1:
                lua_args = ", ".join([str(x) for x in inp])

            # 动态注入运行与断言逻辑（比对函数返回值的标准输出或序列化后的结果）
            # 这里简单起见，假设返回值是基础数据，直接进行等值比较
            # 复杂的结构化对比可以将其转为 string 比较
            lua_script += f"assert({lua_func_name}({lua_args}) == {py_expected}, 'Assertion failed')\n"

        # 写入临时文件，使用真正的 lua 解释器执行
        with tempfile.NamedTemporaryFile(suffix=".lua", mode="w", delete=False) as f:
            f.write(lua_script)
            lua_path = f.name
            
        try:
            lua_res = subprocess.run(["lua", lua_path], capture_output=True, text=True, timeout=2)
            if lua_res.returncode == 0:
                candidate_results.append(True)
            else:
                candidate_results.append(False)
        except subprocess.TimeoutExpired:
            candidate_results.append(False) # 拦截死循环
        finally:
            if os.path.exists(lua_path): os.remove(lua_path)

    return idx, candidate_results, "Success"

# ==========================================
# 3. 统计无偏估计 Pass@k 的数学公式
# ==========================================
def calculate_pass_at_k(n, c, k):
    """
    标准的 Pass@k 无偏估计计算公式 (来自 OpenAI HumanEval 论文)
    n: 通过单测的候选数量 (0 <= n <= c)
    c: 总生成的候选数量 (这里是 5)
    k: 计算评估的 k 值 (如 1 或 5)
    """
    if c - n < k:
        return 1.0
    
    # 组合数公式 C(c-n, k) / C(c, k)
    import math
    def comb(d, r):
        return math.comb(d, r) if d >= r else 0
        
    return 1.0 - (comb(c - n, k) / comb(c, k))

# ==========================================
# 4. 主控评测流水线
# ==========================================
def run_pass_at_k_evaluation(csv_path):
    print(f"🎬 开始载入推理结果进行 Pass@k 差分评测: {csv_path}")
    
    samples = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
            
    total_samples = len(samples)
    max_workers = min(os.cpu_count(), 32) # 控制并发数，防止内存爆掉
    
    print(f"📊 检测到总样本数: {total_samples}，启动 {max_workers} 核并行评测沙箱...")
    
    pass_1_sum = 0.0
    pass_5_sum = 0.0
    valid_eval_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for row in samples:
            # 为每个样本独立随机生成 5 组测试点
            test_inputs = generate_test_inputs(row['python_code'], num_tests=5)
            
            futures.append(executor.submit(
                evaluate_single_sample, 
                row['idx'], 
                row['python_code'], 
                row['all_candidates'], 
                test_inputs
            ))
            
        for future in tqdm(as_completed(futures), total=len(futures), desc="Executing Test Cases"):
            idx, res_list, status = future.result()
            
            if status == "Success":
                valid_eval_count += 1
                c = len(res_list)             # 总生成的候选数量，这里为 5
                n = sum(1 for x in res_list if x) # 这 5 个候选里真正跑通全部单测的数量
                
                # 累加公式计算出的概率值
                pass_1_sum += calculate_pass_at_k(n, c, k=1)
                pass_5_sum += calculate_pass_at_k(n, c, k=5)

    if valid_eval_count > 0:
        final_pass_1 = (pass_1_sum / valid_eval_count) * 100
        final_pass_5 = (pass_5_sum / valid_eval_count) * 100
        print(f"\n======================================")
        print(f"🏆 【CGBridge Stage 3 真实代码能力评测结果】")
        print(f"======================================")
        print(f"🔹 有效评估样本数: {valid_eval_count} / {total_samples}")
        print(f"🔥 真实无偏 Pass@1: {final_pass_1:.2f}%")
        print(f"🔥 真实无偏 Pass@5: {final_pass_5:.2f}%")
        print(f"======================================")
    else:
        print("❌ 未能成功评估任何样本，请检查 Python 源码执行环境或环境依赖。")

if __name__ == "__main__":
    # 指向你的推理生成的 CSV 文件路径
    INFERENCE_CSV = "/root/autodl-tmp/first_paper/python/translation_dataset/Lua_inference_results_original.csv"
    run_pass_at_k_evaluation(INFERENCE_CSV)