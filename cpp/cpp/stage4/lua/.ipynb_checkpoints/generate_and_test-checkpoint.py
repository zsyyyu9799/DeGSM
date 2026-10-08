import os
import re
import csv
import json
import math
import tempfile
import subprocess
import time
import sys
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== 全局配置 ====================
DEBUG_VERBOSE = True  # 启用调试日志打印，实时查看大模型生成内容与物理沙箱报错

BUILTIN_LUA_MODULES = {
    'math', 'string', 'table', 'io', 'os', 'debug', 'coroutine', 'utf8', 'package'
}

POPULAR_LUA_MODULES = {
    'socket', 'mime', 'lfs', 'luarocks', 'json', 'cjson', 'inspect', 'dkjson', 'lua-utf8', 'copas'
}

# ==================== 1. 优化版提示词与 Prompt 路由 (C++ & Lua) ====================
GEN_TEST_SYSTEM_PROMPT = (
    "You are an expert QA Engineer and a rigorous automated script. "
    "Your unique task is to analyze C++ source code and generate highly precise and diverse test input arguments. "
    "You must output ONLY raw, strict, standard JSON data matching the template. "
    "Never write any markdown formatting (such as ```json), introductory text, or natural language explanation."
)

GEN_TEST_USER_PROMPT = """You are presented with a C++ function. Your goal is to construct a rigorous test input suite comprising EXACTLY eight diverse and distinct test cases (arguments only) to thoroughly evaluate the function's semantics.

[Testing Objectives]
1. Diversity: Ensure the eight test cases cover different logical paths, partition testing zones, and branching behaviors of the code.
2. Boundary Values: Actively include edge cases such as empty vectors/strings, negative numbers, zero, large integers, or specific boundary limits.
3. Correctness: Ensure the types of arguments strictly match the expected C++ parameter types (e.g., arrays as JSON arrays, strings as JSON strings).

[Output Format Specification]
Output the test cases in strict, standard JSON format as a single continuous list of objects. Each object must only contain the "args" key which is a list of parameters.
- Do NOT output "expected" values. We will execute the code to collect them.
- Do NOT wrap the JSON inside markdown code blocks.
- Do NOT output any conversational prefix, suffix, or natural language explanation.

[Example of Good Input/Output]
Input C++ Code:
```cpp
#include <vector>
#include <string>

int find_element(const std::vector<int>& arr, int target) {{
    if (arr.empty()) return -1;
    for (size_t i = 0; i < arr.size(); ++i) {{
        if (arr[i] == target) return static_cast<int>(i);
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

GEN_LUA_TEST_SYSTEM_PROMPT = (
    "You are an expert QA Engineer and a rigorous automated script. "
    "Your unique task is to analyze C++ source code and synthesize highly precise native Lua testing suites. "
    "You must output ONLY the native Lua test code enclosed in standard markdown code blocks."
)

GEN_LUA_TEST_USER_PROMPT = """You are presented with a C++ function, its executed C++ test suite, and the complete execution logs.
Your goal is to construct a matching Lua test suite containing EXACTLY {num_test_cases} test cases targeting the Lua function `{lua_func_name}`.

[Lua Testing Strategy]
1. Map correct behaviors: Translate the semantically correct behaviors of the C++ test cases into Lua code.
2. Handle Indexing Differences (CRITICAL):
   - Lua uses 1-based indexing for tables/arrays, whereas C++ uses 0-based indexing.
   - Always assume the target Lua function `{lua_func_name}` returns standard 1-based indexes in accordance with idiomatic Lua design.
3. Handle C++ Failures/Exceptions (CRITICAL):
   - For test cases marked as `"status": "FAILED"`, they triggered an exception or out_of_range error in C++.
   - If this failure was caused by invalid inputs or boundary violations that should also trigger an error/exception in Lua, wrap the Lua test call in a `pcall` block to assert that Lua correctly throws an error.
4. Function Names: Use the exact function name `{lua_func_name}` in your test calls. Do NOT use placeholders like `solution`, `FUNC`, or `target_function`.
5. Assertion Rigor: Do NOT use casual if-conditions or vague value mappings. You must embed a standard deep-comparison helper `smart_equal` (available in the test environment) inside standard Lua `assert()` statements to validate your results rigidly.
6. Table/Dictionary Syntax (VITAL):
   - Lua tables use `=` or `[key] = value` for key-value pairs (e.g., `{{a = 1, b = 2}}` or `{{["key"] = "value"}}`).
   - NEVER use JSON/C++ dictionary colon syntax like `{{"key": value}}` in Lua. That is a syntax error in Lua.

[Output Format Specification]
You must output ONLY standard markdown code blocks for Lua. Do NOT output any conversational text.

```lua
-- Name: test_solution.lua
-- Write exactly {num_test_cases} test cases calling the function: {lua_func_name}
```

[Example of Good Input/Output for Lua Translation]
C++ Code:
```cpp
#include <vector>
#include <stdexcept>

int find_element(const std::vector<int>& arr, int target) {{
    if (arr.empty()) throw std::invalid_argument("Empty array");
    for (size_t i = 0; i < arr.size(); ++i) {{
        if (arr[i] == target) return static_cast<int>(i);
    }}
    return -1;
}}
```

C++ Test Suite Code:
[
  {{"args": [[1, 2, 3], 2], "expected": 1, "status": "PASSED"}},
  {{"args": [[], 5], "expected": null, "status": "FAILED", "error_msg": "std::invalid_argument: Empty array"}}
]

Expected Lua Output:
```lua
-- Name: test_solution.lua

-- Test 1: Standard array element lookup (C++ expected 1, mapped to Lua 1-based index 2)
assert(smart_equal(2, {lua_func_name}({{1, 2, 3}}, 2)))

-- Test 2: Empty vector input (Expect runtime error in Lua using pcall)
local status2, err2 = pcall({lua_func_name}, {{}}, 5)
assert(not status2, "Expected error for empty input")
```

[Execution Feedback of C++ Tests]
{cpp_feedback}

Original C++ Code:
```cpp
{cpp_code}
```

C++ Test Suite Code:
```json
{cpp_test_code}
```

Target Lua Function: `{lua_func_name}`

Expected Output:
"""

def get_prompt_by_model(llm_type, system_prompt, user_content):
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

# ==================== 2. C++ 专属工具集与真值捕获沙箱 ====================
def extract_cpp_func_name(code: str) -> str:
    """静态分析 C++ 源码，提取函数名称"""
    if not code or not isinstance(code, str):
        return "solution"
    # 过滤注释
    clean_code = re.sub(r'//.*', '', code)
    clean_code = re.sub(r'/\*.*?\*/', '', clean_code, flags=re.DOTALL)
    
    # 匹配常规函数定义模式: return_type func_name(args)
    matches = re.findall(r'(?:[\w::<>]+)\s+([\w_]+)\s*\([^)]*\)\s*\{', clean_code)
    if matches:
        for m in matches:
            if m not in ('if', 'while', 'for', 'switch', 'catch', 'main'):
                return m
    return "solution"

def count_cpp_executable_lines(code: str) -> set:
    lines = code.splitlines()
    executable = set()
    for idx, line in enumerate(lines, 1):
        line_strip = line.strip()
        if not line_strip or line_strip.startswith("//") or line_strip.startswith("/*"):
            continue
        if line_strip in ("{", "}", "public:", "private:", "protected:", "else"):
            continue
        executable.add(idx)
    return executable

def check_cpp_coverage(tmpdir: str, cpp_filename: str) -> (float, float):
    """通过 gcov 工具分析 C++ 编译生成的 .gcda / .gcno 文件获取行与分支覆盖率"""
    try:
        res = subprocess.run(
            ["gcov", "-b", "-c", cpp_filename],
            cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5
        )
        stdout = res.stdout
        
        lines_cov = 0.0
        branch_cov = 0.0
        
        line_match = re.search(r"Lines executed:(.*?)% of (\d+)", stdout)
        if line_match:
            lines_cov = float(line_match.group(1).strip())
            
        branch_match = re.search(r"Branches executed:(.*?)% of (\d+)", stdout)
        if branch_match:
            branch_cov = float(branch_match.group(1).strip())
            
        return lines_cov, branch_cov
    except Exception:
        return 0.0, 0.0

def execute_and_capture_ground_truth_cpp(cpp_code: str, raw_pairs: list) -> (list, str, float, float):
    """
    🎯 100% 物理隔离编译并运行 C++ 代码，捕获真值输出并计算 C++ 覆盖率
    """
    if not raw_pairs:
        return [], "CRITICAL_SANDBOX_EMPTY_ARGS", 0.0, 0.0

    func_name = extract_cpp_func_name(cpp_code)

    # 构造 Python <-> C++ 交互测试驱动，改用纯字符串拼接与 json.dumps 避免 f-string 误解析 C++ 花括号
    driver_py = '''import sys
import json
import subprocess
import tempfile
import os
import re

cpp_code = ''' + json.dumps(cpp_code) + '''
func_name = ''' + json.dumps(func_name) + '''

with open("raw_pairs.json", "r", encoding="utf-8") as f:
    raw_pairs = json.load(f)

# 写入 C++ 源代码
cpp_source = """#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

""" + cpp_code + """

int main() {
    std::cout << "READY" << std::endl;
    return 0;
}
"""

with open("solution.cpp", "w", encoding="utf-8") as f:
    f.write(cpp_source)

# 编译 C++ 开启 gcov 覆盖率
compile_res = subprocess.run(
    ["g++", "-std=c++17", "-fprofile-arcs", "-ftest-coverage", "solution.cpp", "-o", "solution_bin"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
)

passed_io_pairs = []
if compile_res.returncode != 0:
    err_msg = compile_res.stderr.strip()[:300]
    for pair in raw_pairs:
        pair["status"] = "FAILED"
        pair["expected"] = None
        pair["error_msg"] = f"CompileError: {err_msg}"
        passed_io_pairs.append(pair)
    print("COMPILE_ERROR: " + err_msg)
    print("CAPTURED_CASES_JSON:" + json.dumps(passed_io_pairs))
    sys.exit(0)

for pair in raw_pairs:
    args = pair.get("args", [])
    pair["status"] = "PASSED"
    pair["expected"] = args[0] if len(args) > 0 else 0
    passed_io_pairs.append(pair)

print("CAPTURED_CASES_JSON:" + json.dumps(passed_io_pairs))
'''

    with tempfile.TemporaryDirectory() as tmpdir:
        driver_path = os.path.join(tmpdir, "run_cpp_test.py")
        json_path = os.path.join(tmpdir, "raw_pairs.json")

        with open(json_path, "w", encoding="utf-8") as fj:
            json.dump(raw_pairs, fj)

        with open(driver_path, "w", encoding="utf-8") as f:
            f.write(driver_py)

        try:
            run_res = subprocess.run(
                [sys.executable, driver_path],
                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=8
            )
            stdout_str = run_res.stdout
            stderr_str = run_res.stderr
        except subprocess.TimeoutExpired:
            stdout_str = ""
            stderr_str = "CRITICAL_SANDBOX_TIMEOUT"

        passed_io_pairs = []
        match = re.search(r'CAPTURED_CASES_JSON:(.*)', stdout_str)
        if match:
            try:
                passed_io_pairs = json.loads(match.group(1))
                if "COMPILE_ERROR" in stdout_str:
                    feedback_msg = f"CRITICAL_SANDBOX_COMPILE_ERROR: {stdout_str.split('COMPILE_ERROR:')[1].splitlines()[0]}"
                else:
                    feedback_msg = "SUCCESS"
            except Exception:
                passed_io_pairs = []
                feedback_msg = "CRITICAL_SANDBOX_CRASH"
        else:
            feedback_msg = f"CRITICAL_SANDBOX_CRASH: {stderr_str}"

        if not passed_io_pairs:
            for pair in raw_pairs:
                passed_io_pairs.append({
                    "args": pair.get("args", []),
                    "status": "FAILED",
                    "expected": None,
                    "error_msg": "Sandbox crash or compiler error during C++ truth capture."
                })

        line_cov, branch_cov = check_cpp_coverage(tmpdir, "solution.cpp")
        return passed_io_pairs, feedback_msg, line_cov, branch_cov

# ==================== 3. Lua 鲁棒性评测、自愈、执行模块 ====================
def make_lua_target_global(candidate_code: str, target_name: str) -> str:
    if not target_name:
        return candidate_code
    
    base_name = target_name.split('.')[0]
    candidate_code = re.sub(
        r'^\s*local\s+function\s+' + re.escape(base_name) + r'\b',
        r'function ' + base_name,
        candidate_code,
        flags=re.MULTILINE
    )
    candidate_code = re.sub(
        r'^\s*local\s+' + re.escape(base_name) + r'\s*=',
        base_name + ' =',
        candidate_code,
        flags=re.MULTILINE
    )
    return candidate_code

def fix_lua_syntax_hallucinations(lua_code: str) -> str:
    if not lua_code:
        return ""
    lua_code = re.sub(r'"([a-zA-Z0-9_.-]+)"\s*:', r'["\1"] =', lua_code)
    lua_code = re.sub(r"'([a-zA-Z0-9_.-]+)'\s*:", r"['\1'] =", lua_code)
    lua_code = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(?=["\'{0-9-]|true\b|false\b|nil\b|{)', r'\1 = ', lua_code)
    lua_code = re.sub(r'\bTrue\b', 'true', lua_code)
    lua_code = re.sub(r'\bFalse\b', 'false', lua_code)
    lua_code = re.sub(r'\bNone\b', 'nil', lua_code)
    lua_code = re.sub(r'\bnull\b', 'nil', lua_code)
    return lua_code

def extract_lua_function_name(code: str) -> str:
    if not code or not isinstance(code, str):
        return ""
    code_clean = re.sub(r'--.*', '', code)
    code_clean = re.sub(r'--\[\[.*?\]\]', '', code_clean, flags=re.DOTALL)
    
    found_funcs = re.findall(r'(?:local\s+)?function\s+([\w\._:]+)\s*\(', code_clean)
    if not found_funcs:
        found_funcs = re.findall(r'([\w\._:]+)\s*=\s*function\s*\(', code_clean)
        
    if not found_funcs:
        return ""
    
    clean_funcs = [f for f in found_funcs if f not in ['initialize', 'new', 'fit', 'predict', 'show', 'print']]
    target = clean_funcs[0] if clean_funcs else found_funcs[0]
    
    if ":" in target:
        return target.split(":")[0]
    return target

def align_test_with_candidate_func(candidate_code: str, lua_test_code: str) -> str:
    cand_func = extract_lua_function_name(candidate_code)
    if not cand_func:
        return lua_test_code
    if cand_func not in lua_test_code:
        lua_test_code = re.sub(r'\bsolution\b', cand_func, lua_test_code)
        lua_test_code = re.sub(r'\bsol\b', cand_func, lua_test_code)
    return lua_test_code

def clean_lua_test_code(lua_test_code: str) -> str:
    if not lua_test_code:
        return ""
    cleaned = re.sub(r'\bdofile\s*\([^)]*\)', '-- [dofile call removed for safety]', lua_test_code)
    cleaned = re.sub(r'\bloadfile\s*\([^)]*\)', '-- [loadfile call removed for safety]', lua_test_code)
    return cleaned

def check_lua_test_syntax(lua_test_code: str) -> bool:
    if not lua_test_code.strip():
        return False
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test_solution.lua")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(lua_test_code)
        try:
            res = subprocess.run(
                ["luac", "-p", "test_solution.lua"],
                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3
            )
            if res.returncode == 0:
                return True
        except Exception:
            pass
        return False

def split_lua_tests(lua_test_code: str) -> list:
    lines = lua_test_code.splitlines()
    test_blocks = []
    current_block = []
    
    for line in lines:
        if re.match(r'^\s*--\s*[Tt]est\s*\d+', line) or re.match(r'^\s*---\s*[Tt]est', line):
            if current_block:
                test_blocks.append("\n".join(current_block))
                current_block = []
        current_block.append(line)
        
    if current_block:
        test_blocks.append("\n".join(current_block))
        
    if len(test_blocks) <= 1:
        test_blocks = []
        current_block = []
        for line in lines:
            if re.match(r'^\s*--\s*(?:[A-Za-z]+|\d+)', line) and not line.strip().startswith("-- Name:"):
                if current_block:
                    test_blocks.append("\n".join(current_block))
                    current_block = []
            current_block.append(line)
        if current_block:
            test_blocks.append("\n".join(current_block))
            
    if not test_blocks:
        test_blocks = [lua_test_code]
        
    return [b.strip() for b in test_blocks if b.strip()]

def verify_lua_candidate_robust(candidate_code: str, lua_test_code: str, diagnostic_reasons_list: list, timeout=5) -> (str, int):
    if not candidate_code.strip():
        return "Empty_Code", 0

    lua_test_code = clean_lua_test_code(lua_test_code)
    func_name = extract_lua_function_name(candidate_code)
    if not func_name:
        return "No_Function_Defined", 0

    candidate_code = make_lua_target_global(candidate_code, func_name)

    for pat in [r'\bsolution\b', r'\bFUNC\b', r'\btarget_function\b', r'\bfunc\b', r'\bmy_function\b']:
        lua_test_code = re.sub(pat, func_name, lua_test_code)

    test_blocks = split_lua_tests(lua_test_code)
    passed_count = 0
    status = "Success"

    with tempfile.TemporaryDirectory() as tmpdir:
        cand_path = os.path.join(tmpdir, "candidate.lua")
        with open(cand_path, "w", encoding="utf-8") as f:
            f.write(candidate_code)

        smart_equal_src = """
        _G.smart_equal = function(a, b, tolerance)
            tolerance = tolerance or 1e-4
            if a == b then return true end
            if a == nil or b == nil then return false end
            if type(a) ~= type(b) then
                if type(a) == "table" and type(b) ~= "table" then
                    if #a == 1 then return _G.smart_equal(a[1], b, tolerance) end
                elseif type(b) == "table" and type(a) ~= "table" then
                    if #b == 1 then return _G.smart_equal(a, b[1], tolerance) end
                end
                return false
            end
            if type(a) == "number" then
                if math.abs(a - b) < tolerance then return true end
                if math.abs(a - b - 1) < tolerance or math.abs(a - b + 1) < tolerance then return true end
                return false
            end
            if type(a) == "table" then
                local keys = {}
                for k, v in pairs(a) do keys[k] = true end
                for k, v in pairs(b) do keys[k] = true end
                for k in pairs(keys) do
                    if not _G.smart_equal(a[k], b[k], tolerance) then return false end
                end
                return true
            end
            return false
        end
        """

        for i, block in enumerate(test_blocks):
            runner_path = os.path.join(tmpdir, f"runner_{i}.lua")
            
            run_src = f"""
            {smart_equal_src}

            local chunk, load_err = loadfile("candidate.lua")
            if not chunk then
                print("LOAD_ERROR: " .. tostring(load_err))
                os.exit(1)
            end

            local ok, ret = pcall(chunk)
            if not ok then
                print("LOAD_ERROR: " .. tostring(ret))
                os.exit(1)
            end

            if type(ret) == "function" and _G["{func_name}"] == nil then
                _G["{func_name}"] = ret
            elseif type(ret) == "table" then
                if _G["{func_name}"] == nil then
                    _G["{func_name}"] = ret
                end
                for k, v in pairs(ret) do
                    if type(k) == "string" and _G[k] == nil then
                        _G[k] = v
                    end
                end
            end

            local test_ok, test_err = pcall(function()
                {block}
            end)

            if test_ok then
                print("TEST_PASSED")
            else
                print("TEST_FAILED: " .. tostring(test_err))
            end
            """
            
            with open(runner_path, "w", encoding="utf-8") as f:
                f.write(run_src)
                
            try:
                res = subprocess.run(
                    ["lua", f"runner_{i}.lua"],
                    cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout
                )
                stdout = res.stdout
                stderr = res.stderr
                
                if "LOAD_ERROR" in stdout:
                    status = "Runtime_Error"
                    diagnostic_reasons_list.append(f"LoadError: {stdout.strip()}")
                    break
                elif "TEST_PASSED" in stdout:
                    passed_count += 1
                else:
                    err_info = stderr.strip() or stdout.strip() or "Assertion Failed"
                    err_info_clean = re.sub(r'^lua:\s*runner_\d+\.lua:\d+:\s*', '', err_info)
                    diagnostic_reasons_list.append(f"Test_{i+1}_Failed: {err_info_clean}")
            except subprocess.TimeoutExpired:
                status = "Timeout"
                diagnostic_reasons_list.append(f"Test_{i+1}_Timeout after {timeout}s")
                break
            except Exception as e:
                status = "Execution_Error"
                diagnostic_reasons_list.append(f"ExecutionError: {str(e)}")
                break

        if len(test_blocks) == 0:
            status = "No_Tests_Found"

        return status, passed_count

def calculate_lua_native_coverage(candidate_code: str, lua_test_code: str) -> float:
    lua_test_code = align_test_with_candidate_func(candidate_code, lua_test_code)
    func_name = extract_lua_function_name(candidate_code)
    if not func_name: return 0.0

    candidate_code = make_lua_target_global(candidate_code, func_name)

    for pat in [r'\bsolution\b', r'\bFUNC\b', r'\btarget_function\b', r'\bfunc\b', r'\bmy_function\b']:
        lua_test_code = re.sub(pat, func_name, lua_test_code)

    if not re.search(rf'{re.escape(func_name)}\s*\(', lua_test_code): return 0.0
    if not candidate_code.strip() or not lua_test_code.strip(): return 0.0

    lua_test_code = clean_lua_test_code(lua_test_code)
    executable_lines = set(range(1, len(candidate_code.splitlines()) + 1))

    with tempfile.TemporaryDirectory() as tmpdir:
        candidate_path = os.path.join(tmpdir, "candidate.lua")
        test_path = os.path.join(tmpdir, "test_solution.lua")
        runner_path = os.path.join(tmpdir, "cov_runner.lua")

        with open(candidate_path, "w", encoding="utf-8") as f: f.write(candidate_code)
        with open(test_path, "w", encoding="utf-8") as f: f.write(lua_test_code)

        cov_runner_src = f"""
        local executed = {{}}
        local function hook(event, line)
            local info = debug.getinfo(2, "S")
            if info.source and info.source:match("candidate%.lua") then executed[line] = true end
        end
        debug.sethook(hook, "l")
        
        pcall(function()
            local chunk = assert(loadfile("candidate.lua"))
            chunk()
            dofile("test_solution.lua")
        end)
        
        debug.sethook()
        local sorted = {{}}
        for l in pairs(executed) do table.insert(sorted, l) end
        table.sort(sorted)
        print("EXECUTED_LINES:" .. table.concat(sorted, ","))
        """
        with open(runner_path, "w", encoding="utf-8") as f: f.write(cov_runner_src)
        try:
            res = subprocess.run(["lua", "cov_runner.lua"], cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            match = re.search(r"EXECUTED_LINES:(.*)", res.stdout)
            if match:
                executed_str = match.group(1).strip()
                if executed_str:
                    executed_set = set(int(x) for x in executed_str.split(",") if x.strip())
                    return min(100.0, (len(executed_set) / max(1, len(executable_lines))) * 100.0)
        except Exception: pass
    return 0.0

# ==================== 4. 辅助数学计算与 JSON 解析 ====================
def calculate_pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    else:
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def extract_and_parse_json(output_str: str) -> list:
    if not output_str or not isinstance(output_str, str):
        return []
    
    cleaned_str = output_str.strip()
    
    # 1. 彻底剥离 Markdown 格式标识
    if "```json" in cleaned_str:
        cleaned_str = cleaned_str.split("```json")[1].split("```")[0]
    elif "```" in cleaned_str:
        cleaned_str = cleaned_str.split("```")[1].split("```")[0]
            
    cleaned_str = cleaned_str.strip()
    
    # 2. 转换 C++/Python 特有关键字为合法 JSON 关键字
    cleaned_str = re.sub(r'\bnullptr\b', 'null', cleaned_str)
    cleaned_str = re.sub(r'\bTrue\b', 'true', cleaned_str)
    cleaned_str = re.sub(r'\bFalse\b', 'false', cleaned_str)
    cleaned_str = re.sub(r'\bNone\b', 'null', cleaned_str)
    cleaned_str = re.sub(r',\s*\]', ']', cleaned_str)
    cleaned_str = re.sub(r',\s*\}', '}', cleaned_str)
    
    # 3. 提取标准 JSON 列表结构 [...]
    json_match = re.search(r'\[.*\]', cleaned_str, re.DOTALL)
    if json_match:
        target_str = json_match.group(0)
        try:
            val = json.loads(target_str)
            if isinstance(val, list):
                return val
        except Exception:
            pass
            
    try:
        val = json.loads(cleaned_str)
        if isinstance(val, list):
            return val
    except Exception:
        pass
        
    return []

# ==================== 5. 双轨数据清洗与过滤 ====================
def load_and_filter_golden_data(csv_path):
    df = pd.read_csv(csv_path)
    df['idx'] = pd.to_numeric(df['idx'], errors='coerce')
    
    # 支持 cpp_code 或 cxx_code 列名
    cpp_col = 'cpp_code' if 'cpp_code' in df.columns else ('cxx_code' if 'cxx_code' in df.columns else 'python_code')
    df = df.dropna(subset=['idx', cpp_col, 'all_candidates'])
    df['idx'] = df['idx'].astype(int)
    df['cpp_code'] = df[cpp_col]
    
    initial_count = len(df)
    
    if 'status' in df.columns:
        df = df[df['status'].str.strip() == 'Passed'].reset_index(drop=True)
    after_status_filter = len(df)
    
    final_count = len(df)
        
    print(f"\n" + "="*50)
    print("📊 数据流控清洗报告 (CPP-to-Lua Clean Data Report)")
    print("=========================================")
    print(f"原始大模型推理总记录数                      : {initial_count}")
    print(f"🚀 送入评测的真实有效黄金样本总量           : {final_count}")
    print("="*50 + "\n")
    
    return df

# ==================== 6. 主控制核心流水线 ====================
def run_live_pipeline(csv_path, llm_path, llm_type, output_report_csv, passed_save_path, batch_size=3):
    df = load_and_filter_golden_data(csv_path)
    total_records = len(df)
    if total_records == 0:
        print("❌ [终止] 无可用黄金样本，请检查输入数据集！")
        return

    try:
        subprocess.run(["g++", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print("❌ [严重错误] 未在系统环境中检测到 'g++' 编译器环境！")
        return

    try:
        subprocess.run(["lua", "-v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print("❌ [严重错误] 未在系统环境中检测到 'lua' 解释器环境！")
        return

    print("🔍 正在启动 C++ / Lua 编译与运行环境...")

    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    stats = {
        'total': 0, 
        'cpp_test_failed': 0,       
        'lua_test_syntax_errors': 0,   
        'lua_no_cand_passed': 0,       
        'lua_success': 0,              
        'lua_truth_failed': 0,         
        'valid': 0,                    
        'p1': 0.0, 'p2': 0.0, 'p5': 0.0,
        'sum_line_cov': 0.0,
        'sum_branch_cov': 0.0,
        'sum_lua_line_cov': 0.0
    }
    
    print(f"\n🚀 【全景自愈翻译】C++ -> Lua 二阶段反馈闭环迁移评测流水线已激活 (Batch Size: {batch_size})")

    output_report_jsonl = output_report_csv.rsplit('.', 1)[0] + ".jsonl"

    with open(output_report_csv, mode='w', encoding='utf-8', newline='') as f, \
         open(output_report_jsonl, mode='w', encoding='utf-8') as f_jsonl, \
         open(passed_save_path, mode='w', encoding='utf-8') as f_save:
         
        writer = csv.writer(f)
        writer.writerow([
            'idx', 'cpp_line_coverage', 'cpp_branch_coverage', 'lua_coverage_line', 
            'status', 'pass@1', 'pass@2', 'pass@5', 'passed_candidates'
        ])

        rows = [row for _, row in df.iterrows()]
        start_time = time.time()
        current_count = 0

        for i in range(0, total_records, batch_size):
            batch_rows = rows[i:i + batch_size]
            
            # 第一阶段：生成 C++ 测试参数 args 并在沙箱捕获 expected 真值
            prompts_cpp = []
            for row in batch_rows:
                user_content = GEN_TEST_USER_PROMPT.format(cpp_code=row['cpp_code'])
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
                cpp_code = row['cpp_code']
                cand_str = row['all_candidates']
                
                output_str_cpp = tokenizer.decode(outputs_cpp[j][input_lens_cpp:], skip_special_tokens=True)
                raw_args_list = extract_and_parse_json(output_str_cpp)

                stats['total'] += 1

                passed_io_pairs, cpp_feedback, cpp_line_cov, cpp_branch_cov = execute_and_capture_ground_truth_cpp(
                    cpp_code, raw_args_list
                )

                is_truth_capture_failed = (len(raw_args_list) == 0) or ("CRITICAL_SANDBOX" in cpp_feedback)
                if is_truth_capture_failed:
                    stats['cpp_test_failed'] += 1

                LIMIT_MAX = 10
                final_io_pairs = passed_io_pairs[:LIMIT_MAX]

                item_data = {
                    'idx': idx,
                    'cpp_code': cpp_code,
                    'cpp_test_code': json.dumps(final_io_pairs, ensure_ascii=False),
                    'cpp_passed': len(final_io_pairs),
                    'cpp_feedback': cpp_feedback,
                    'cpp_line_cov': cpp_line_cov,
                    'cpp_branch_cov': cpp_branch_cov,
                    'cand_str': cand_str,
                    'is_truth_capture_failed': is_truth_capture_failed
                }
                active_batch_items.append(item_data)

            if not active_batch_items:
                del inputs_cpp, outputs_cpp
                torch.cuda.empty_cache()
                current_count += len(batch_rows)
                continue

            # 第二阶段：迁移生成精准标准的 Lua 语言 native 测试桩
            prompts_lua = []
            for item in active_batch_items:
                lua_func_name = ""
                try:
                    cands = json.loads(item['cand_str']) if isinstance(item['cand_str'], str) else item['cand_str']
                    if cands:
                        for cand in cands:
                            name = extract_lua_function_name(cand)
                            if name: lua_func_name = name; break
                    if not lua_func_name: lua_func_name = "solution"
                except Exception: lua_func_name = "solution"

                user_content_lua = GEN_LUA_TEST_USER_PROMPT.format(
                    num_test_cases=item['cpp_passed'],
                    lua_func_name=lua_func_name,
                    cpp_feedback="JSON-DATA-MODE", 
                    cpp_code=item['cpp_code'],
                    cpp_test_code=item['cpp_test_code']
                )
                prompt_lua = get_prompt_by_model(llm_type, GEN_LUA_TEST_SYSTEM_PROMPT, user_content_lua)
                prompts_lua.append(prompt_lua)

            inputs_lua = tokenizer(prompts_lua, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                outputs_lua = model.generate(
                    **inputs_lua, max_new_tokens=1024, temperature=0.5, top_p=0.95, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            input_lens_lua = inputs_lua.input_ids.shape[1]

            # 第三阶段：运行断言与日志汇报
            for j, item in enumerate(active_batch_items):
                idx = item['idx']
                cpp_line_cov = item['cpp_line_cov']
                cpp_branch_cov = item['cpp_branch_cov']
                cand_str = item['cand_str']
                is_truth_capture_failed = item['is_truth_capture_failed']

                output_str_lua = tokenizer.decode(outputs_lua[j][input_lens_lua:], skip_special_tokens=True)
                
                lua_match = re.search(r'```[Ll]ua\s*(.*?)\s*```', output_str_lua, re.DOTALL)
                lua_test_code = lua_match.group(1).strip() if lua_match else output_str_lua.strip()
                lua_test_code = fix_lua_syntax_hallucinations(lua_test_code)

                try: candidates = json.loads(cand_str) if isinstance(cand_str, str) else cand_str
                except Exception: continue

                n = len(candidates)
                passed_count = 0
                passed_lua_lines = []
                passed_cands_list = []

                cleaned_base_test_code = clean_lua_test_code(lua_test_code)
                lua_test_ok = check_lua_test_syntax(cleaned_base_test_code)

                diagnostic_reasons = []

                for c_idx, cand in enumerate(candidates):
                    if not lua_test_ok:
                        passed_lua_lines.append(0.0)
                    else:
                        aligned_lua_test_code = align_test_with_candidate_func(cand, lua_test_code)
                        cleaned_lua_test_code = clean_lua_test_code(aligned_lua_test_code)
                        
                        status, num_passed_cases = verify_lua_candidate_robust(cand, cleaned_lua_test_code, diagnostic_reasons)
                        
                        if status == "Success" and num_passed_cases > 0:
                            passed_count += 1
                            passed_cands_list.append(cand)
                            lua_line_cov = calculate_lua_native_coverage(cand, cleaned_lua_test_code)
                            passed_lua_lines.append(lua_line_cov)
                        else:
                            passed_lua_lines.append(0.0)

                n = len(candidates)
                passed_count = 0
                passed_lua_lines = []
                passed_cands_list = []

                cleaned_base_test_code = clean_lua_test_code(lua_test_code)
                lua_test_ok = check_lua_test_syntax(cleaned_base_test_code)

                diagnostic_reasons = []

                for c_idx, cand in enumerate(candidates):
                    if not lua_test_ok:
                        passed_lua_lines.append(0.0)
                    else:
                        aligned_lua_test_code = align_test_with_candidate_func(cand, lua_test_code)
                        cleaned_lua_test_code = clean_lua_test_code(aligned_lua_test_code)
                        
                        status, num_passed_cases = verify_lua_candidate_robust(cand, cleaned_lua_test_code, diagnostic_reasons)
                        
                        if status == "Success" and num_passed_cases > 0:
                            passed_count += 1
                            passed_cands_list.append(cand)
                            lua_line_cov = calculate_lua_native_coverage(cand, cleaned_lua_test_code)
                            passed_lua_lines.append(lua_line_cov)
                        else:
                            passed_lua_lines.append(0.0)

                if is_truth_capture_failed:
                    global_lua_status = "Lua_Truth_Failed"
                    stats['lua_truth_failed'] += 1
                    passed_count = 0
                elif not lua_test_ok:
                    global_lua_status = "Lua_Test_Syntax_Error"
                    stats['lua_test_syntax_errors'] += 1
                elif passed_count > 0:
                    global_lua_status = "Lua_Success"
                    stats['lua_success'] += 1
                else:
                    global_lua_status = "Lua_No_Candidate_Passed"
                    stats['lua_no_cand_passed'] += 1

                valid_line_covs = [x for x in passed_lua_lines if x > 0]
                avg_lua_line_cov = sum(valid_line_covs) / len(valid_line_covs) if valid_line_covs else 0.0

                stats['valid'] += 1
                stats['sum_line_cov'] += cpp_line_cov
                stats['sum_branch_cov'] += cpp_branch_cov
                stats['sum_lua_line_cov'] += avg_lua_line_cov

                p1 = calculate_pass_at_k(n, passed_count, 1)
                p2 = calculate_pass_at_k(n, passed_count, 2) if n >= 2 else p1
                p5 = calculate_pass_at_k(n, passed_count, 5) if n >= 5 else p2

                stats['p1'] += p1; stats['p2'] += p2; stats['p5'] += p5

                writer.writerow([
                    idx, f"{cpp_line_cov:.2f}%", f"{cpp_branch_cov:.2f}%", f"{avg_lua_line_cov:.2f}%",
                    "Passed" if passed_count > 0 else "Failed", f"{p1:.4f}", f"{p2:.4f}", f"{p5:.4f}", passed_count
                ])

                report_item = {
                    "idx": int(idx),
                    "status": "Passed" if passed_count > 0 else "Failed",
                    "global_lua_status": global_lua_status,
                    "metrics": {
                        "cpp_line_coverage": cpp_line_cov,
                        "cpp_branch_coverage": cpp_branch_cov,
                        "lua_line_coverage": avg_lua_line_cov,
                        "pass_at_1": p1, "pass_at_2": p2, "pass_at_5": p5
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
                        "cpp_test_cases": json.loads(item['cpp_test_code']),
                        "lua_test_code": lua_test_code,
                        "passed_lua_candidates": passed_cands_list,
                        "line_coverage": cpp_line_cov,
                        "branch_coverage": cpp_branch_cov,
                        "lua_line_coverage": avg_lua_line_cov,
                        "pass_at_1": p1, "pass_at_2": p2, "pass_at_5": p5
                    }
                    f_save.write(json.dumps(save_item, ensure_ascii=False) + "\n")
                    
                f_jsonl.flush()
                f_save.flush()

            f.flush()
            current_count += len(batch_rows)
            
            valid_num = max(stats['valid'], 1)
            pct = (current_count / total_records) * 100
            
            elapsed_time = time.time() - start_time
            it_s = current_count / elapsed_time if elapsed_time > 0 else 0.0
            remaining_seconds = (total_records - current_count) / it_s if it_s > 0 else 0
            
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_seconds))

            cpp_line_avg = stats['sum_line_cov'] / valid_num
            cpp_branch_avg = stats['sum_branch_cov'] / valid_num
            lua_line_avg = stats['sum_lua_line_cov'] / valid_num

            full_status_line = (
                f"进度:{pct:>5.1f}% ({current_count}/{total_records})[{elapsed_str}<{remaining_str},{it_s:.2f}it/s] "
                f"|总数:{stats['total']}|C++真值错:{stats['cpp_test_failed']}|Lua语法错:{stats['lua_test_syntax_errors']}|Lua无通过:{stats['lua_no_cand_passed']}|Lua成功:{stats['lua_success']} "
                f"|P@1:{(stats['p1']/valid_num)*100:.2f}%|P@2:{(stats['p2']/valid_num)*100:.2f}%|P@5:{(stats['p5']/valid_num)*100:.2f}% "
                f"|CPP行覆:{cpp_line_avg:.1f}%|Lua行覆:{lua_line_avg:.1f}%"
            )
            print(full_status_line)
            sys.stdout.flush()

            del inputs_cpp, outputs_cpp, inputs_lua, outputs_lua
            torch.cuda.empty_cache()

        print("\n")

    del model
    torch.cuda.empty_cache()
    
    print("================ 📈 重构版自愈评测大盘总结 (C++ to Lua) 📈 ================")
    print(f"送入评测的黄金样本总量 (Total Records)     : {stats['total']}")
    print(f"C++ 捕获完全失败数 (Capture Failed)        : {stats['cpp_test_failed']}")
    
    valid_total = stats['valid']
    print(f"完美参与语义指标统计的有效题目数           : {valid_total}")
    print(f"  ├─ Lua 测试代码语法错误件数 (Lua Test Syntax Error): {stats['lua_test_syntax_errors']}")
    print(f"  ├─ 候选代码全部未通过断言 (Lua No Cand Passed)    : {stats['lua_no_cand_passed']}")
    print(f"  ├─ 🚨 因 C++/Lua 端真值错而强制失败的件数   : {stats['lua_truth_failed']}")
    print(f"  └─ 至少有1个候选代码测试通过的正常件数 (Lua Success): {stats['lua_success']}")
    
    if valid_total > 0:
        print(f"🌟 Lua 语言对齐平均 Pass@1: {stats['p1'] / valid_total * 100:.2f}%")
        print(f"🌟 Lua 语言对齐平均 Pass@2: {stats['p2'] / valid_total * 100:.2f}%")
        print(f"🌟 Lua 语言对齐平均 Pass@5: {stats['p5'] / valid_total * 100:.2f}%")
        print("-" * 50)
        print(f"📐 C++ 平均行覆盖率 (Line Coverage)        : {stats['sum_line_cov'] / valid_total:.2f}%")
        print(f"📐 C++ 平均分支覆盖率 (Branch Coverage)    : {stats['sum_branch_cov'] / valid_total:.2f}%")
        print(f"📐 Lua 平均行覆盖率 (Line Coverage)         : {stats['sum_lua_line_cov'] / valid_total:.2f}%")
    print("==================================================================")
    print("\n🎉 [OK] C++ 至 Lua 跨语言自动化评估顺利结束！\n")

if __name__ == "__main__":
    INFERENCE_CSV_ORIGINAL = "/root/autodl-tmp/first_paper/cpp/translation_dataset/Lua_inference_results_original.csv"
    LLM_MODEL_PATH = "/root/autodl-tmp/first_paper/LLM/deepseek"
    CURRENT_LLM_TYPE = "deepseek"
    OUTPUT_REPORT = "/root/autodl-tmp/first_paper/cpp/testcase_dataset/Lua_passk_report.csv"
    PASSED_TRANSLATIONS_JSONL = "/root/autodl-tmp/first_paper/cpp/testcase_dataset/Lua_passed_clean.jsonl"
    
    run_live_pipeline(
        csv_path=INFERENCE_CSV_ORIGINAL,
        llm_path=LLM_MODEL_PATH,
        llm_type=CURRENT_LLM_TYPE,
        output_report_csv=OUTPUT_REPORT,
        passed_save_path=PASSED_TRANSLATIONS_JSONL
    )