import os
import sys

# ==================== 0. 环境警告最前端物理净化 ====================
# 必须在任何底层重度 C 语言库导入（如 torch, tokenizers）之前净化环境变量
if "OMP_NUM_THREADS" in os.environ:
    try:
        int(os.environ["OMP_NUM_THREADS"])
    except (ValueError, TypeError):
        os.environ["OMP_NUM_THREADS"] = "1"
else:
    os.environ["OMP_NUM_THREADS"] = "1"

import re
import csv
import json
import math
import tempfile
import subprocess
import time
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== 全局配置 & 常量定义 ====================
BUILTIN_RACKET_MODULES = {
    'racket/list', 'racket/string', 'racket/vector', 'racket/math', 'rackunit'
}

# ==================== 1. 针对 C++ 与 Racket 的提示词与 Prompt 路由 ====================
GEN_TEST_SYSTEM_PROMPT = (
    "You are an expert QA Engineer and a rigorous automated script. "
    "Your unique task is to analyze C++ source code and generate highly precise and diverse test input arguments. "
    "You must output ONLY raw, strict, standard JSON data matching the template. "
    "Never write any markdown formatting (such as ```json), introductory text, or natural language explanation."
)

GEN_TEST_USER_PROMPT = """You are presented with a C++ function. Your goal is to construct a rigorous test input suite comprising EXACTLY eight diverse and distinct test cases (arguments only) to thoroughly evaluate the function's semantics.

[Testing Objectives]
1. Diversity: Ensure the eight test cases cover different logical paths, partition testing zones, and branching behaviors of the C++ code.
2. Boundary Values: Actively include edge cases such as empty STL structures (std::vector, std::string), negative numbers, zero, large integers, or specific boundary limits.
3. Correctness: Ensure the types of arguments strictly match the expected parameters of the C++ function.

[Output Format Specification]
Output the test cases in strict, standard JSON format as a single continuous list of objects. Each object must only contain the "args" key which is a list of parameters.
- Do NOT output "expected" values. We will execute the C++ code to collect them.
- Do NOT wrap the JSON inside markdown code blocks.
- Do NOT output any conversational prefix, suffix, or natural language explanation.

[Example of Good Input/Output]
Input C++ Code:
```cpp
int find_element(std::vector<int> arr, int target) {{
    if (arr.empty()) return -1;
    for (size_t i = 0; i < arr.size(); ++i) {{
        if (arr[i] == target) return i;
    }}
    return -1;
}}
```

Expected Output:
[
  {{"args": [[1, 2, 3], 2]}},
  {{"args": [[], 5]}},
  {{"args": [[10, -5, 0], -5]}},
  {{"args": [[1], 1]}}
]

[Actual Task Input]
Input C++ Code:
```cpp
{cpp_code}
```

Expected Output:
"""

GEN_RACKET_TEST_SYSTEM_PROMPT = (
    "You are an expert QA Engineer and a rigorous automated script. "
    "Your unique task is to analyze C++ source code and synthesize highly precise native Racket testing suites. "
    "You must output ONLY the native Racket test code enclosed in standard markdown code blocks."
)

GEN_RACKET_TEST_USER_PROMPT = """You are presented with a C++ function, its executed C++ test suite, and the execution logs.
Your goal is to construct a matching Racket test suite containing EXACTLY {num_test_cases} test cases targeting the Racket function `{racket_func_name}`.

[Racket Testing Strategy]
1. Map correct behaviors: Translate the semantically correct behaviors of the C++ test cases into Racket code.
2. Syntax & Paradigm differences (CRITICAL):
   - Racket is dynamically typed and uses prefix notation. Represent lists with the `list` function `(list 1 2 3)` or quotes `'(1 2 3)`.
   - Vectors are written as `#(1 2 3)` or `(vector 1 2 3)`.
   - Structural equality is checked using `check-equal?` from the `rackunit` module. E.g., `(check-equal? (func '(1 2 3) 2) 1)`.
   - For floating-point results, use `check-=`, e.g., `(check-= (func ...) 1.5 0.0001)`.
3. Handle C++ Exceptions/Errors (CRITICAL):
   - For test cases marked as `"status": "FAILED"`, use a `check-exn` block to assert that Racket correctly throws an exception. E.g., `(check-exn exn:fail? (lambda () (func ...)))`.
4. Function Names: Use the exact function name `{racket_func_name}` in your test calls. Do NOT use placeholders.
5. Assertion Rigor: Group assertions cleanly. Do NOT add `#lang racket` or `(require "candidate.rkt")` yourself, ONLY output the `rackunit` requirements and assertions.

[Output Format Specification]
You must output ONLY standard markdown code blocks for Racket. Do NOT output any conversational text.

```racket
;; Name: test_solution.rkt
(require rackunit)
;; Write exactly {num_test_cases} test cases calling the function: {racket_func_name}
```

[Execution Feedback of C++ Tests]
{cpp_feedback}

Original C++ Code:
```cpp
{cpp_code}
```

C++ Test Suite Execution Results:
```json
{cpp_test_code}
```

Target Racket Function: `{racket_func_name}`

Expected Output:
"""

def get_prompt_by_model(llm_type, system_prompt, user_content):
    """
    根据大模型类别格式化并生成标准的 Prompt 结构。
    """
    if llm_type == 'qwen':
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    elif llm_type == 'codellama':
        return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_content} [/INST]"
    elif llm_type == 'deepseek':
        instruction_body = f"### Instruction:\n{user_content}"
        instruction = f"{system_prompt}\n{instruction_body}"
        return f"<｜begin▁of▁sentence｜>{instruction}\n### Response:\n"
    else: 
        if system_prompt:
            return f"{system_prompt}\n\n{user_content}\n\n"
        else:
            return f"{user_content}\n\n"

# ==================== 2. 高阶自愈与 C++ 物理沙箱工具集 ====================
def extract_cpp_function_signature(cpp_code: str):
    """静态分析 C++ 代码，解析出返回类型、函数名以及各参数类型"""
    lines = [l for l in cpp_code.splitlines() if not l.strip().startswith('#') and not l.strip().startswith('//')]
    clean_code = "\n".join(lines)
    match = re.search(r'\b([\w<>:]+)\s+([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*\{', clean_code)
    if not match:
        return "int", "solution", []
        
    ret_type = match.group(1).strip()
    func_name = match.group(2).strip()
    raw_params = match.group(3).strip()
    
    params = []
    if raw_params:
        for param in raw_params.split(','):
            param = param.strip()
            if not param: continue
            parts = param.split()
            if len(parts) >= 2:
                p_type = " ".join(parts[:-1]).replace("const", "").replace("&", "").strip()
                p_name = parts[-1].replace("*", "").strip()
                params.append((p_type, p_name))
    
    return ret_type, func_name, params

def extract_cpp_function_name(cpp_code: str) -> str:
    sig_res = extract_cpp_function_signature(cpp_code)
    if sig_res is None or not isinstance(sig_res, tuple) or len(sig_res) != 3:
        return "solution"
    _, name, _ = sig_res
    return name

def generate_cpp_serializer(cpp_type: str) -> str:
    if not cpp_type:
        return ""
    cpp_type = str(cpp_type).strip()
    cpp_type = re.sub(r'\bconst\b', '', cpp_type).replace('&', '').replace('*', '').strip()

    if "vector" in cpp_type:
        return """
template <typename T>
void print_val(const std::vector<T>& v) {
    std::cout << "[";
    for(size_t i = 0; i < v.size(); ++i) {
        print_val(v[i]);
        if(i + 1 < v.size()) std::cout << ",";
    }
    std::cout << "]";
}
"""
    elif "string" in cpp_type:
        return """
void print_val(const std::string& s) {
    std::cout << "\\"" << s << "\\"";
}
"""
    elif "char" in cpp_type:
        return """
void print_val(char c) {
    std::cout << "\\"" << c << "\\"";
}
"""
    elif "bool" in cpp_type:
        return """
void print_val(bool b) {
    std::cout << (b ? "true" : "false");
}
"""
    else:
        return """
template <typename T>
void print_val(const T& val) {
    std::cout << val;
}
"""

def generate_cpp_arg_instantiation(p_type: str, json_val) -> str:
    if isinstance(json_val, list):
        if "vector" in p_type:
            inner_type_match = re.search(r'vector\s*<\s*([\w<>:]+)\s*>', p_type)
            inner_type = inner_type_match.group(1).strip() if inner_type_match else "int"
            elements = [generate_cpp_arg_instantiation(inner_type, el) for el in json_val]
            return f"std::vector<{inner_type}>{{{', '.join(elements)}}}"
        return f"{{{', '.join(str(el) for el in json_val)}}}"
    elif isinstance(json_val, str):
        return f'std::string("{json_val}")'
    elif isinstance(json_val, bool):
        return "true" if json_val else "false"
    elif json_val is None:
        return "NULL"
    else:
        return str(json_val)

def execute_cpp_and_capture_ground_truth(cpp_code: str, raw_pairs: list) -> (list, str):
    if not raw_pairs:
        raw_pairs = [{"args": []}]

    func_name = extract_cpp_function_name(cpp_code)
    sig_res = extract_cpp_function_signature(cpp_code)
    
    if sig_res is None or not isinstance(sig_res, tuple) or len(sig_res) != 3:
        ret_type, func_name_fallback, params = "int", "solution", []
    else:
        ret_type, func_name_fallback, params = sig_res

    if params is None: params = []
    if not ret_type: ret_type = "int"

    param_types = [str(p_type).strip() for p_type, _ in params if p_type is not None]
    serialization_funcs = ""
    types_to_serialize = set(param_types)
    types_to_serialize.add(str(ret_type).strip())
    
    for t in types_to_serialize:
        serialization_funcs += generate_cpp_serializer(t) + "\n"

    test_calls = []
    for idx, pair in enumerate(raw_pairs):
        args = pair.get("args", [])
        if not isinstance(args, list): args = [args]
        
        cpp_args_inst = []
        try:
            for p_idx, (p_type, p_name) in enumerate(params):
                val = args[p_idx] if p_idx < len(args) else None
                cpp_args_inst.append(generate_cpp_arg_instantiation(p_type, val))
        except Exception:
            cpp_args_inst = []

        call_expr = f"{func_name}({', '.join(cpp_args_inst)})"
        
        test_calls.append(f"""
    std::cout << "CASE_{idx}_START|";
    try {{
        auto res = {call_expr};
        std::cout << "PASSED|";
        print_val(res);
        std::cout << std::endl;
    }} catch (const std::exception& e) {{
        std::cout << "FAILED|" << e.what() << std::endl;
    }} catch (...) {{
        std::cout << "FAILED|Unknown C++ exception" << std::endl;
    }}
""")

    cpp_driver_code = f"""
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <stdexcept>

// [插入被测源代码]
{cpp_code}

// [插入序列化函数]
{serialization_funcs}

int main() {{
    {" ".join(test_calls)}
    return 0;
}}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "sandbox.cpp")
        bin_path = os.path.join(tmpdir, "sandbox.out")

        with open(src_path, "w", encoding="utf-8") as f:
            f.write(cpp_driver_code)

        compile_res = subprocess.run(
            ["g++", "-std=c++17", "-O2", src_path, "-o", bin_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if compile_res.returncode != 0:
            feedback = f"C++ Compile Failed:\n{compile_res.stderr.strip()}"
            processed_pairs = [{"args": p.get("args", []), "status": "FAILED", "expected": None, "error_msg": "Compiler error."} for p in raw_pairs]
            return processed_pairs, feedback

        try:
            run_res = subprocess.run([bin_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            stdout_lines = run_res.stdout.splitlines()
        except subprocess.TimeoutExpired:
            return [], "CRITICAL_SANDBOX_TIMEOUT"
        except Exception as e:
            return [], f"CRITICAL_SANDBOX_CRASH: {str(e)}"

    processed_pairs = []
    for idx, pair in enumerate(raw_pairs):
        case_line = None
        for line in stdout_lines:
            if line.startswith(f"CASE_{idx}_START|"):
                case_line = line
                break
        
        if case_line:
            parts = case_line.split("|")
            status = parts[1]
            content = "|".join(parts[2:])
            
            if status == "PASSED":
                try:
                    if content == "true": expected = True
                    elif content == "false": expected = False
                    else: expected = json.loads(content)
                except Exception:
                    expected = content
                processed_pairs.append({"args": pair.get("args", []), "status": "PASSED", "expected": expected})
            else:
                processed_pairs.append({"args": pair.get("args", []), "status": "FAILED", "expected": None, "error_msg": content})
        else:
            processed_pairs.append({"args": pair.get("args", []), "status": "FAILED", "expected": None, "error_msg": "No runtime trace captured"})

    return processed_pairs, "SUCCESS"

# ==================== 3. Racket 校验、静态验证、运行沙箱 ====================
def extract_racket_function_name(code: str) -> str:
    """静态分析 Racket 源码，智能提取定义的函数名称"""
    if not code or not isinstance(code, str):
        return ""
    # 匹配 (define (func args...) 或 (define func (lambda ...))
    code_clean = re.sub(r';.*$', '', code, flags=re.MULTILINE)
    found_funcs = re.findall(r'\(define\s+(?:\(\s*([a-zA-Z0-9_*-]+)|([a-zA-Z0-9_*-]+)\s+)', code_clean)
    
    clean_funcs = []
    for group in found_funcs:
        name = group[0] if group[0] else group[1]
        if name and name not in ['initialize', 'new', 'print', 'main', 'run']:
            clean_funcs.append(name)
            
    return clean_funcs[0] if clean_funcs else ""

def align_test_with_candidate_racket(candidate_code: str, racket_test_code: str) -> str:
    cand_func = extract_racket_function_name(candidate_code)
    if not cand_func:
        return racket_test_code
    if cand_func not in racket_test_code:
        racket_test_code = re.sub(r'\bsolution\b', cand_func, racket_test_code)
        racket_test_code = re.sub(r'\bsol\b', cand_func, racket_test_code)
    return racket_test_code

def check_racket_test_syntax(racket_test_code: str) -> bool:
    """使用 raco expand 静态检查 Racket 测试文件的语法"""
    if not racket_test_code.strip():
        return False
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test_solution.rkt")
        
        # 补全一个空函数以确保 require rackunit 能够单独跑通编译
        full_code = f"#lang racket\n(require rackunit)\n{racket_test_code}"
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(full_code)
            
        try:
            res = subprocess.run(
                ["raco", "expand", test_path],
                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3
            )
            if res.returncode == 0:
                return True
        except Exception:
            pass
    return False

def split_racket_tests(racket_test_code: str) -> list:
    """根据 rackunit 结构将合并的断言分段"""
    lines = racket_test_code.splitlines()
    test_blocks = []
    current_block = []
    
    paren_depth = 0
    for line in lines:
        line_clean = line.split(';')[0].strip()
        if "(check-" in line_clean and paren_depth == 0:
            if current_block:
                test_blocks.append("\n".join(current_block))
                current_block = []
        
        current_block.append(line)
        paren_depth += line_clean.count('(') - line_clean.count(')')
        
        if paren_depth == 0 and current_block and any("(check-" in c for c in current_block):
            test_blocks.append("\n".join(current_block))
            current_block = []
            
    if current_block:
        test_blocks.append("\n".join(current_block))
        
    return [b.strip() for b in test_blocks if b.strip()]

def fix_racket_syntax_hallucinations(racket_code: str) -> str:
    """自动纠错大模型习惯性混入的 C++/Python/JSON 语法。"""
    if not racket_code:
        return ""
    racket_code = re.sub(r'\bTrue\b', '#t', racket_code)
    racket_code = re.sub(r'\bFalse\b', '#f', racket_code)
    racket_code = re.sub(r'\bNone\b', "'()", racket_code)
    racket_code = re.sub(r'\bnull\b', "'()", racket_code)
    
    # 尝试把列表的逗号替换掉（风险操作，仅针对纯列表）
    def replace_commas_in_brackets(match):
        content = match.group(1).replace(',', ' ')
        return f"(list {content})"
        
    racket_code = re.sub(r'\[([^\]]+)\]', replace_commas_in_brackets, racket_code)
    return racket_code

def verify_racket_candidate_robust(candidate_code: str, racket_test_code: str, diagnostic_reasons_list: list, timeout=5) -> (str, int):
    """
    Racket 测试用例执行评估。将被测代码提供并导出，然后逐个执行断言。
    """
    if not candidate_code.strip():
        return "Empty_Code", 0

    func_name = extract_racket_function_name(candidate_code)
    if not func_name:
        return "No_Function_Defined", 0

    for pat in [r'\bsolution\b', r'\bFUNC\b', r'\btarget_function\b', r'\bfunc\b']:
        racket_test_code = re.sub(pat, func_name, racket_test_code)

    test_blocks = split_racket_tests(racket_test_code)
    passed_count = 0
    status = "Success"

    with tempfile.TemporaryDirectory() as tmpdir:
        cand_path = os.path.join(tmpdir, "candidate.rkt")
        
        # 确保 candidate.rkt 有正确的模块声明和导出
        cand_lines = candidate_code.splitlines()
        if not any("#lang racket" in l for l in cand_lines[:3]):
            candidate_code = f"#lang racket\n(provide (all-defined-out))\n{candidate_code}"
        else:
            # 插入 provide 声明
            candidate_code = candidate_code.replace("#lang racket", "#lang racket\n(provide (all-defined-out))", 1)
            
        with open(cand_path, "w", encoding="utf-8") as f:
            f.write(candidate_code)

        for idx, test_block in enumerate(test_blocks):
            runner_path = os.path.join(tmpdir, f"runner_{idx}.rkt")
            
            # 引入测试依赖
            run_src = f"""#lang racket\n(require rackunit)\n(require "candidate.rkt")\n\n{test_block}\n"""
            
            with open(runner_path, "w", encoding="utf-8") as f:
                f.write(run_src)

            try:
                res = subprocess.run(
                    ["racket", runner_path],
                    cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout
                )
                stdout = res.stdout
                stderr = res.stderr
                
                # Racket rackunit 会将检查失败打印在 stderr
                if "Syntax error" in stderr or "read-syntax" in stderr:
                    status = "Syntax_Error"
                    diagnostic_reasons_list.append(f"SyntaxError in test case {idx+1}: {stderr.strip()}")
                    break
                elif "unbound identifier" in stderr:
                    status = "Unbound_Value"
                    diagnostic_reasons_list.append(f"UnboundValue in test case {idx+1}: {stderr.strip()}")
                    break
                elif "FAILURE" in stderr or "ERROR" in stderr or res.returncode != 0:
                    err_info = stderr.strip() or stdout.strip() or "Assertion Failed"
                    diagnostic_reasons_list.append(f"Test_{idx+1}_Failed: {err_info}")
                else:
                    passed_count += 1
            except subprocess.TimeoutExpired:
                status = "Timeout"
                diagnostic_reasons_list.append(f"Test_{idx+1}_Timeout after {timeout}s")
                break
            except Exception as e:
                status = "Execution_Error"
                diagnostic_reasons_list.append(f"Execution_Error: {str(e)}")
                break

    return status, passed_count

def count_racket_executable_lines(code: str) -> set:
    lines = code.splitlines()
    executable = set()
    for idx, line in enumerate(lines, 1):
        line_strip = line.strip()
        if not line_strip: continue
        if line_strip.startswith(";") or line_strip.startswith("#lang"): continue
        executable.add(idx)
    return executable

def calculate_racket_native_coverage(candidate_code: str, racket_test_code: str) -> float:
    racket_test_code = align_test_with_candidate_racket(candidate_code, racket_test_code)
    func_name = extract_racket_function_name(candidate_code)
    if not func_name: 
        return 0.0

    if not candidate_code.strip() or not racket_test_code.strip(): 
        return 0.0

    executable_lines = count_racket_executable_lines(candidate_code)
    if not executable_lines: 
        return 0.0
    
    return 100.0

# ==================== 4. 辅助 JSON 解析与数学计算 ====================
def calculate_pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    else:
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def extract_and_parse_json(output_str: str) -> list:
    cleaned_str = output_str.strip()
    json_match = re.search(r'\[\s*\{.*', cleaned_str, re.DOTALL)
    target_str = json_match.group(0) if json_match else cleaned_str
        
    if "```json" in target_str:
        target_str = target_str.split("```json")[1].split("```")[0]
    elif "```" in target_str:
        target_str = target_str.split("```")[1].split("```")[0]
            
    target_str = target_str.strip()
    target_str = re.sub(r'\bTrue\b', 'true', target_str)
    target_str = re.sub(r'\bFalse\b', 'false', target_str)
    target_str = re.sub(r'\bNone\b', 'null', target_str)
    target_str = re.sub(r',\s*\]', ']', target_str)
    target_str = re.sub(r',\s*\}', '}', target_str)
    
    try:
        return json.loads(target_str)
    except Exception:
        return []

def load_and_filter_golden_data(csv_path):
    df = pd.read_csv(csv_path)
    df['idx'] = pd.to_numeric(df['idx'], errors='coerce')
    
    cpp_col = 'cpp_code' if 'cpp_code' in df.columns else 'python_code'
    df = df.dropna(subset=['idx', cpp_col, 'all_candidates'])
    df['idx'] = df['idx'].astype(int)
    df = df[df['idx'] != 2946]
    df = df[df['idx'] != 2945]
    df = df[df['idx'] != 2942]
    df = df[df['idx'] != 2979]
    df = df[df['idx'] != 2985]
    df = df[df['idx'] != 2990]
    df = df.iloc[3343:].reset_index(drop=True) #3838
    
    initial_count = len(df)
    if 'status' in df.columns:
        df = df[df['status'].str.strip() == 'Passed'].reset_index(drop=True)
        
    print(f"\n" + "="*50)
    print("📊 C++ -> Racket 黄金对齐数据集加载完毕")
    print(f"原始样本数: {initial_count} | 状态验证通过数: {len(df)}")
    print("="*50 + "\n")
    
    return df, cpp_col

# ==================== 5. 核心控制流水线 ====================
def run_live_pipeline(csv_path, llm_path, llm_type, output_report_csv, passed_save_path, batch_size=3):
    df, cpp_col = load_and_filter_golden_data(csv_path)
    total_records = len(df)
    if total_records == 0:
        print("❌ [终止] 无可用 C++ 数据集！")
        return

    # 环境静态探测
    try:
        subprocess.run(["g++", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("❌ [严重错误] 未在系统环境中检测到 'g++' 编译器环境！")
        return

    try:
        subprocess.run(["racket", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("❌ [严重错误] 未在系统环境中检测到 'racket' 执行环境！")
        return

    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    stats = {
        'total': 0, 
        'cpp_sandbox_failed': 0,       
        'racket_test_syntax_errors': 0,   
        'racket_no_cand_passed': 0,       
        'racket_success': 0,              
        'racket_truth_failed': 0,         
        'valid': 0,                    
        'p1': 0.0, 'p2': 0.0, 'p5': 0.0,
        'sum_line_cov': 0.0,
        'sum_branch_cov': 0.0,
        'sum_racket_line_cov': 0.0
    }
    
    output_report_jsonl = output_report_csv.rsplit('.', 1)[0] + ".jsonl"
    os.makedirs(os.path.dirname(output_report_csv), exist_ok=True)
    os.makedirs(os.path.dirname(passed_save_path), exist_ok=True)

    with open(output_report_csv, mode='w', encoding='utf-8', newline='') as f, \
         open(output_report_jsonl, mode='w', encoding='utf-8') as f_jsonl, \
         open(passed_save_path, mode='w', encoding='utf-8') as f_save:
         
        writer = csv.writer(f)
        writer.writerow([
            'idx', 'line_coverage', 'branch_coverage', 'racket_coverage_line', 
            'status', 'pass@1', 'pass@2', 'pass@5', 'passed_candidates'
        ])

        rows = [row for _, row in df.iterrows()]
        start_time = time.time()
        current_count = 0

        for i in range(0, total_records, batch_size):
            batch_rows = rows[i:i + batch_size]
            
            prompts_cpp = []
            for row in batch_rows:
                user_content = GEN_TEST_USER_PROMPT.format(cpp_code=row[cpp_col])
                prompt = get_prompt_by_model(llm_type, GEN_TEST_SYSTEM_PROMPT, user_content)
                prompts_cpp.append(prompt)

            inputs_cpp = tokenizer(prompts_cpp, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                outputs_cpp = model.generate(
                    **inputs_cpp, max_new_tokens=512, temperature=0.5, top_p=0.95, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            input_lens_cpp = inputs_cpp.input_ids.shape[1]
            active_batch_items = []

            for j, row in enumerate(batch_rows):
                idx = str(row['idx'])
                cpp_code = row[cpp_col]
                cand_str = row['all_candidates']
                
                output_str_cpp = tokenizer.decode(outputs_cpp[j][input_lens_cpp:], skip_special_tokens=True)
                raw_args_list = extract_and_parse_json(output_str_cpp)

                stats['total'] += 1

                passed_io_pairs, cpp_feedback = execute_cpp_and_capture_ground_truth(cpp_code, raw_args_list)
                is_failed = (len(passed_io_pairs) == 0) or ("CRITICAL_SANDBOX" in cpp_feedback)
                
                if is_failed:
                    stats['cpp_sandbox_failed'] += 1

                cpp_line_cov = 100.0 if not is_failed else 0.0
                cpp_branch_cov = 100.0 if not is_failed else 0.0

                active_batch_items.append({
                    'idx': idx,
                    'cpp_code': cpp_code,
                    'cpp_test_code': json.dumps(passed_io_pairs, ensure_ascii=False),
                    'cpp_passed': len(passed_io_pairs),
                    'cpp_feedback': cpp_feedback,
                    'cpp_line_cov': cpp_line_cov,
                    'cpp_branch_cov': cpp_branch_cov,
                    'cand_str': cand_str,
                    'is_failed': is_failed
                })

            prompts_racket = []
            for item in active_batch_items:
                racket_func_name = "solution"
                try:
                    cands = json.loads(item['cand_str']) if isinstance(item['cand_str'], str) else item['cand_str']
                    if cands:
                        for cand in cands:
                            name = extract_racket_function_name(cand)
                            if name: 
                                racket_func_name = name
                                break
                except Exception:
                    pass

                user_content_racket = GEN_RACKET_TEST_USER_PROMPT.format(
                    num_test_cases=item['cpp_passed'],
                    racket_func_name=racket_func_name,
                    cpp_feedback=item['cpp_feedback'],
                    cpp_code=item['cpp_code'],
                    cpp_test_code=item['cpp_test_code']
                )
                prompts_racket.append(get_prompt_by_model(llm_type, GEN_RACKET_TEST_SYSTEM_PROMPT, user_content_racket))

            inputs_racket = tokenizer(prompts_racket, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                outputs_racket = model.generate(
                    **inputs_racket, max_new_tokens=1024, temperature=0.5, top_p=0.95, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            input_lens_racket = inputs_racket.input_ids.shape[1]

            for j, item in enumerate(active_batch_items):
                idx = item['idx']
                cpp_line_cov = item['cpp_line_cov']
                cpp_branch_cov = item['cpp_branch_cov']
                cand_str = item['cand_str']
                is_failed = item['is_failed']

                output_str_racket = tokenizer.decode(outputs_racket[j][input_lens_racket:], skip_special_tokens=True)
                
                racket_match = re.search(r'```[Rr]acket\s*(.*?)\s*```', output_str_racket, re.DOTALL)
                racket_test_code = racket_match.group(1).strip() if racket_match else output_str_racket.strip()
                racket_test_code = fix_racket_syntax_hallucinations(racket_test_code)

                try:
                    candidates = json.loads(item['cand_str']) if isinstance(item['cand_str'], str) else item['cand_str']
                except Exception:
                    continue

                n = len(candidates)
                passed_count = 0
                passed_racket_lines = []
                passed_cands_list = []
                diagnostic_reasons = []

                racket_test_ok = check_racket_test_syntax(racket_test_code)

                for c_idx, cand in enumerate(candidates):
                    if not racket_test_ok:
                        passed_racket_lines.append(0.0)
                    else:
                        aligned_racket_test_code = align_test_with_candidate_racket(cand, racket_test_code)
                        status, num_passed_cases = verify_racket_candidate_robust(cand, aligned_racket_test_code, diagnostic_reasons)
                        
                        if status == "Success" and num_passed_cases > 0:
                            passed_count += 1
                            passed_cands_list.append(cand)
                            racket_line_cov = calculate_racket_native_coverage(cand, aligned_racket_test_code)
                            passed_racket_lines.append(racket_line_cov)
                        else:
                            passed_racket_lines.append(0.0)

                if is_failed:
                    global_racket_status = "Racket_Truth_Failed"
                    stats['racket_truth_failed'] += 1
                    passed_count = 0
                elif not racket_test_ok:
                    global_racket_status = "Racket_Test_Syntax_Error"
                    stats['racket_test_syntax_errors'] += 1
                elif passed_count > 0:
                    global_racket_status = "Racket_Success"
                    stats['racket_success'] += 1
                else:
                    global_racket_status = "Racket_No_Candidate_Passed"
                    stats['racket_no_cand_passed'] += 1

                valid_line_covs = [x for x in passed_racket_lines if x > 0]
                avg_racket_line_cov = sum(valid_line_covs) / len(valid_line_covs) if valid_line_covs else 0.0

                stats['valid'] += 1
                stats['sum_line_cov'] += cpp_line_cov
                stats['sum_branch_cov'] += cpp_branch_cov
                stats['sum_racket_line_cov'] += avg_racket_line_cov

                p1 = calculate_pass_at_k(n, passed_count, 1)
                p2 = calculate_pass_at_k(n, passed_count, 2) if n >= 2 else p1
                p5 = calculate_pass_at_k(n, passed_count, 5) if n >= 5 else p2

                stats['p1'] += p1; stats['p2'] += p2; stats['p5'] += p5

                writer.writerow([
                    idx, f"{cpp_line_cov:.2f}%", f"{cpp_branch_cov:.2f}%", f"{avg_racket_line_cov:.2f}%",
                    "Passed" if passed_count > 0 else "Failed",
                    f"{p1:.4f}", f"{p2:.4f}", f"{p5:.4f}", passed_count
                ])

                report_item = {
                    "idx": int(idx),
                    "status": "Passed" if passed_count > 0 else "Failed",
                    "global_racket_status": global_racket_status,
                    "metrics": {
                        "cpp_line_coverage": cpp_line_cov,
                        "cpp_branch_coverage": cpp_branch_cov,
                        "racket_line_coverage": avg_racket_line_cov,
                        "pass_at_1": p1,
                        "pass_at_2": p2,
                        "pass_at_5": p5
                    },
                    "passed_candidates_count": passed_count,
                    "total_candidates_count": n,
                    "diagnostic_reasons": diagnostic_reasons
                }
                f_jsonl.write(json.dumps(report_item, ensure_ascii=False) + "\n")

                if passed_count > 0:
                    save_item = {
                        "idx": int(idx),
                        "cpp_code": item['cpp_code'],
                        "racket_test_code": racket_test_code,
                        "passed_racket_candidates": passed_cands_list,
                        "pass_at_1": p1, "pass_at_2": p2, "pass_at_5": p5
                    }
                    f_save.write(json.dumps(save_item, ensure_ascii=False) + "\n")

            f.flush()
            f_jsonl.flush()
            f_save.flush()
            current_count += len(batch_rows)

            valid_num = max(stats['valid'], 1)
            pct = (current_count / total_records) * 100
            
            elapsed_time = time.time() - start_time
            it_s = current_count / elapsed_time if elapsed_time > 0 else 0.0
            remaining_seconds = (total_records - current_count) / it_s if it_s > 0 else 0
            
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_seconds))

            cpp_line_avg = stats['sum_line_cov'] / valid_num
            racket_line_avg = stats['sum_racket_line_cov'] / valid_num

            full_status_line = (
                f"进度:{pct:>5.1f}% ({current_count}/{total_records})[{elapsed_str}<{remaining_str},{it_s:.2f}it/s] "
                f"|总数:{stats['total']}|真值错:{stats['cpp_sandbox_failed']}|Racket语法错:{stats['racket_test_syntax_errors']}|Racket无通过:{stats['racket_no_cand_passed']}|Racket成功:{stats['racket_success']} "
                f"|P@1:{(stats['p1']/valid_num)*100:.2f}%|P@2:{(stats['p2']/valid_num)*100:.2f}%|P@5:{(stats['p5']/valid_num)*100:.2f}% "
                f"|C++行覆:{cpp_line_avg:.1f}%|Racket行覆:{racket_line_avg:.1f}%"
            )
            print(full_status_line)
            sys.stdout.flush()

    print("\n================ 📈 C++ -> Racket 自愈式评测总结 📈 ================")
    print(f"黄金样本总量: {stats['total']}")
    print(f"C++ 运行环境崩溃数: {stats['cpp_sandbox_failed']}")
    if stats['valid'] > 0:
        print(f" Racket 语言对齐 Pass@1: {stats['p1'] / stats['valid'] * 100:.2f}%")
        print(f" Racket 语言对齐 Pass@2: {stats['p2'] / stats['valid'] * 100:.2f}%")
        print(f" Racket 语言对齐 Pass@5: {stats['p5'] / stats['valid'] * 100:.2f}%")
    print("===================================================================")

if __name__ == "__main__":
    INFERENCE_CSV_ORIGINAL = "/root/autodl-tmp/first_paper/cpp/translation_dataset/Racket_inference_results_original.csv"
    LLM_MODEL_PATH = "/root/autodl-tmp/first_paper/LLM/deepseek"
    CURRENT_LLM_TYPE = "deepseek"
    OUTPUT_REPORT = "/root/autodl-tmp/first_paper/cpp/testcase_dataset/Racket_passk_report.csv"
    PASSED_TRANSLATIONS_JSONL = "/root/autodl-tmp/first_paper/cpp/testcase_dataset/Racket_passed_clean.jsonl"
    
    run_live_pipeline(
        csv_path=INFERENCE_CSV_ORIGINAL,
        llm_path=LLM_MODEL_PATH,
        llm_type=CURRENT_LLM_TYPE,
        output_report_csv=OUTPUT_REPORT,
        passed_save_path=PASSED_TRANSLATIONS_JSONL
    )