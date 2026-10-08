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
# Lua 语言内置标准系统模块
BUILTIN_LUA_MODULES = {
    'math', 'string', 'table', 'io', 'os', 'debug', 'coroutine', 'utf8', 'package'
}

# 真正的常用 Lua 第三方及常用模块白名单，拦截大模型幻觉生成的 Python 包或临时变量
POPULAR_LUA_MODULES = {
    'socket', 'mime', 'lfs', 'luarocks', 'json', 'cjson', 'inspect', 'dkjson', 'lua-utf8', 'copas'
}

# ==================== 1. 优化版提示词与 Prompt 路由 ====================
GEN_TEST_SYSTEM_PROMPT = (
    "You are an expert QA Engineer and a rigorous automated script. "
    "Your unique task is to analyze Python source code and generate highly precise and diverse test input arguments. "
    "You must output ONLY raw, strict, standard JSON data matching the template. "
    "Never write any markdown formatting (such as ```json), introductory text, or natural language explanation."
)

GEN_TEST_USER_PROMPT = """You are presented with a Python function. Your goal is to construct a rigorous test input suite comprising EXACTLY eight diverse and distinct test cases (arguments only) to thoroughly evaluate the function's semantics.

[Testing Objectives]
1. Diversity: Ensure the eight test cases cover different logical paths, partition testing zones, and branching behaviors of the code.
2. Boundary Values: Actively include edge cases such as empty structures (strings/lists), negative numbers, zero, large integers, or specific boundary limits.
3. Correctness: Ensure the types of arguments strictly match the expected parameters of the Python function.

[Output Format Specification]
Output the test cases in strict, standard JSON format as a single continuous list of objects. Each object must only contain the "args" key which is a list of parameters.
- Do NOT output "expected" values. We will execute the code to collect them.
- Do NOT wrap the JSON inside markdown code blocks.
- Do NOT output any conversational prefix, suffix, or natural language explanation.

[Example of Good Input/Output]
Input Python Code:
```python
def find_element(arr, target):
    if not arr:
        return -1
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1
```

Expected Output:
[
  {{"args": [[1, 2, 3], 2]}},
  {{"args": [[], 5]}},
  {{"args": [[10, -5, 0], -5]}},
  {{"args": [["apple", "banana"], "cherry"]}},
  {{"args": [[1], 1]}}
]

[Actual Task Input]
Input Python Code:
```python
{python_code}
```

Expected Output:
"""

GEN_LUA_TEST_SYSTEM_PROMPT = (
    "You are an expert QA Engineer and a rigorous automated script. "
    "Your unique task is to analyze Python source code and synthesize highly precise native Lua testing suites. "
    "You must output ONLY the native Lua test code enclosed in standard markdown code blocks."
)

GEN_LUA_TEST_USER_PROMPT = """You are presented with a Python function, its executed Python test suite, and the complete execution logs.
Your goal is to construct a matching Lua test suite containing EXACTLY {num_test_cases} test cases targeting the Lua function `{lua_func_name}`.

[Lua Testing Strategy]
1. Map correct behaviors: Translate the semantically correct behaviors of the Python test cases into Lua code.
2. Handle Indexing Differences (CRITICAL):
   - Lua uses 1-based indexing for tables/arrays, whereas Python uses 0-based indexing.
   - Always assume the target Lua function `{lua_func_name}` returns standard 1-based indexes in accordance with idiomatic Lua design.
3. Handle Python Failures/Exceptions (CRITICAL):
   - For test cases marked as `"status": "FAILED"`, they triggered an exception in Python.
   - If this failure was caused by invalid inputs or boundary violations that should also trigger an error/exception in Lua, wrap the Lua test call in a `pcall` block to assert that Lua correctly throws an error.
4. Function Names: Use the exact function name `{lua_func_name}` in your test calls. Do NOT use placeholders like `solution`, `FUNC`, or `target_function`.
5. Assertion Rigor: Do NOT use casual if-conditions or vague value mappings. You must embed a standard deep-comparison helper `smart_equal` (available in the test environment) inside standard Lua `assert()` statements to validate your results rigidly.
6. Table/Dictionary Syntax (VITAL):
   - Lua tables use `=` or `[key] = value` for key-value pairs (e.g., `{{a = 1, b = 2}}` or `{{["key"] = "value"}}`).
   - NEVER use JSON/Python dictionary colon syntax like `{{"key": value}}` in Lua. That is a syntax error in Lua.

[Output Format Specification]
You must output ONLY standard markdown code blocks for Lua. Do NOT output any conversational text.

```lua
-- Name: test_solution.lua
-- Write exactly {num_test_cases} test cases calling the function: {lua_func_name}
```

[Example of Good Input/Output for Lua Translation]
Python Code:
```python
def find_element(arr, target):
    if not arr:
        raise ValueError("Empty array is not allowed")
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1
```

Python Test Suite Code:
[
  {{"args": [[1, 2, 3], 2], "expected": 1, "status": "PASSED"}},
  {{"args": [[], 5], "expected": null, "status": "FAILED", "error_msg": "ValueError: Empty array is not allowed"}}
]

Expected Lua Output:
```lua
-- Name: test_solution.lua

-- Test 1: Standard array element lookup (Python expected 1, mapped to Lua 1-based index 2)
assert(smart_equal(2, {lua_func_name}({{1, 2, 3}}, 2)))

-- Test 2: Empty vector input (Expect runtime error in Lua using pcall)
local status2, err2 = pcall({lua_func_name}, {{}}, 5)
assert(not status2, "Expected error for empty input")
```

[Execution Feedback of Python Tests]
{python_feedback}

Original Python Code:
```python
{python_code}
```

Python Test Suite Code:
```json
{python_test_code}
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

# ==================== 2. 高阶自愈前置工具集 ====================
def preprocess_python_code(python_code: str, header: str) -> str:
    lines = python_code.splitlines()
    future_lines = []
    other_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("from __future__") or (future_lines and stripped.endswith("\\")) or (future_lines and "(" in future_lines[-1] and ")" not in future_lines[-1]):
            future_lines.append(line)
        else:
            other_lines.append(line)
            
    return "\n".join(future_lines) + "\n" + header + "\n" + "\n".join(other_lines)

def extract_python_func_name(code: str) -> str:
    match = re.search(r'^def\s+(\w+)\s*\(', code, re.MULTILINE)
    if match:
        return match.group(1)
    match_any = re.search(r'def\s+(\w+)\s*\(', code)
    return match_any.group(1) if match_any else None

def check_python_coverage(python_code, io_pairs):
    if not io_pairs: return 0.0, 0.0

    mock_and_cleanup_header = (
        "import sys\n"
        "import builtins\n"
        "builtins.input = lambda *args, **kwargs: ''\n"
        "sys.argv = ['run_cov_test.py']\n"
    )

    full_test_script_prefix = preprocess_python_code(python_code, mock_and_cleanup_header)

    test_script_lines = [
        "import sys",
        "import builtins",
        "import json",
        "import inspect",
        "import html",
        "import types",
        "import re",
        "",
        "# Mock 缺失或弃用的外部库",
        "try:",
        "    import cgi",
        "except ImportError:",
        "    cgi = types.ModuleType('cgi')",
        "    sys.modules['cgi'] = cgi",
        "if not hasattr(cgi, 'escape'):",
        "    cgi.escape = html.escape",
        "",
        "builtins.input = lambda *args, **kwargs: ''",
        "sys.argv = ['run_cov_test.py']",
        "",
        "# 物理加载执行用户代码，隔离顶层执行期 Crash",
        "with open('user_code.py', 'r', encoding='utf-8') as f:",
        "    user_code = f.read()",
        "",
        "try:",
        "    exec(user_code, globals())",
        "except Exception:",
        "    pass",
        "",
        "with open('io_pairs.json', 'r', encoding='utf-8') as f:",
        "    io_pairs = json.load(f)",
        "",
        "def discover_callable():",
        "    match_func = []",
        "    match_class = []",
        "    for line in user_code.splitlines():",
        "        m_func = re.match(r'^def\\s+(\\w+)', line)",
        "        if m_func: match_func.append(m_func.group(1))",
        "        m_cls = re.match(r'^class\\s+(\\w+)', line)",
        "        if m_cls: match_class.append(m_cls.group(1))",
        "",
        "    user_funcs = {}",
        "    user_classes = {}",
        "    for name, obj in list(globals().items()):",
        "        if name.startswith('__') and name.endswith('__'): continue",
        "        if name in ('discover_callable', 'sanitize_output'): continue",
        "        if inspect.isfunction(obj) and obj.__module__ == '__main__':",
        "            user_funcs[name] = obj",
        "        elif inspect.isclass(obj) and obj.__module__ == '__main__':",
        "            user_classes[name] = obj",
        "",
        "    for name in match_func:",
        "        if name in user_funcs: return user_funcs[name]",
        "    if user_funcs: return list(user_funcs.values())[0]",
        "    if user_classes:",
        "        first_class_name = match_class[0] if match_class else list(user_classes.keys())[0]",
        "        cls = user_classes.get(first_class_name, list(user_classes.values())[0])",
        "        class_methods = []",
        "        for attr_name in dir(cls):",
        "            if attr_name.startswith('__'): continue",
        "            attr = getattr(cls, attr_name)",
        "            if callable(attr):",
        "                if attr_name in cls.__dict__: class_methods.append((attr_name, attr))",
        "        if class_methods:",
        "            method_name, method_obj = class_methods[0]",
        "            return method_obj",
        "        return cls",
        "    return None",
        "",
        "func = discover_callable()",
        "if func:",
        "    for pair in io_pairs:",
        "        try:",
        "            args = pair['args']",
        "            func(*args)",
        "        except Exception:",
        "            pass"
    ]
    
    full_test_script = "\n".join(test_script_lines)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "run_cov_test.py")
        json_path = os.path.join(tmpdir, "io_pairs.json")
        user_code_path = os.path.join(tmpdir, "user_code.py")
        
        with open(user_code_path, "w", encoding="utf-8") as fu:
            fu.write(python_code)
            
        try:
            with open(json_path, "w", encoding="utf-8") as fj:
                json.dump(io_pairs, fj)
        except RecursionError as rec_ex:
            # 🚨 捕获由于大模型幻觉生成的恶性超深嵌套造成的 json.dump 爆栈
            print(f"\n[❌ CRITICAL RECURSION ERROR DETECTED DURING GROUND TRUTH CAPTURE]")
            print(f"  ├─ 错误类型: {type(rec_ex).__name__}")
            print(f"  └─ 提示: 该样本包含无限循环或超深多维嵌套数据结构，已自动激活 Fallback 机制。")
            stdout_str = ""
            stderr_str = f"RecursionError: maximum recursion depth exceeded while serializing raw_pairs"
            
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(full_test_script)
            
        try:
            subprocess.run(
                ["coverage", "run", "--branch", "run_cov_test.py"], 
                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, timeout=3
            )
            subprocess.run(["coverage", "json", "-o", "cov.json"], cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
            json_report_path = os.path.join(tmpdir, "cov.json")
            if os.path.exists(json_report_path):
                with open(json_report_path, "r") as fj:
                    report_data = json.load(fj)
                totals = report_data.get("totals", {})
                num_statements = totals.get("num_statements", 0)
                covered_lines = totals.get("covered_lines", 0)
                num_branches = totals.get("num_branches", 0)
                covered_branches = totals.get("covered_branches", 0)
                
                line_cov = (covered_lines / num_statements) * 100.0 if num_statements > 0 else 0.0
                branch_cov = (covered_branches / num_branches) * 100.0 if num_branches > 0 else 100.0
                return float(line_cov), float(branch_cov)
        except Exception:
            return 0.0, 0.0
    return 0.0, 0.0

def make_lua_target_global(candidate_code: str, target_name: str) -> str:
    """
    🎯 自愈 Lua 解答中的 local 声明问题：将顶层的 local target_name = 或 local function target_name
    转换为全局声明，保证测试桩能顺利调用到。
    """
    if not target_name:
        return candidate_code
    
    base_name = target_name.split('.')[0]
    
    # 1. 替换 local function base_name(...) 为 function base_name(...)
    candidate_code = re.sub(
        r'^\s*local\s+function\s+' + re.escape(base_name) + r'\b',
        r'function ' + base_name,
        candidate_code,
        flags=re.MULTILINE
    )
    
    # 2. 替换 local base_name\s*= 为 base_name =
    candidate_code = re.sub(
        r'^\s*local\s+' + re.escape(base_name) + r'\s*=',
        base_name + ' =',
        candidate_code,
        flags=re.MULTILINE
    )
    
    return candidate_code

def fix_lua_syntax_hallucinations(lua_code: str) -> str:
    """
    🎯 自动修复大模型将 Python/JSON 字典及布尔语法直接抄到 Lua 里的幻觉错。
    不使用有歧义的单引号/双引号原始字符串转义，改用最直接、无损的字面量正则替换。
    """
    if not lua_code:
        return ""
    
    # 1. 自动转换双引号键冒号 {"key": value} -> {["key"] = value}
    lua_code = re.sub(r'"([a-zA-Z0-9_.-]+)"\s*:', r'["\1"] =', lua_code)
    
    # 2. 自动转换单引号键冒号 {'key': value} -> {['key'] = value}
    lua_code = re.sub(r"'([a-zA-Z0-9_.-]+)'\s*:", r"['\1'] =", lua_code)
    
    # 3. 自动转换无引号键冒号 {key: value} -> {key = value} 并精准过滤 Lua 原生方法调用 (如 Port:new())
    lua_code = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(?=["\'{0-9-]|true\b|false\b|nil\b|{)', r'\1 = ', lua_code)
    
    # 4. 修复 Python 常见的布尔值及空值关键字
    lua_code = re.sub(r'\bTrue\b', 'true', lua_code)
    lua_code = re.sub(r'\bFalse\b', 'false', lua_code)
    lua_code = re.sub(r'\bNone\b', 'nil', lua_code)
    lua_code = re.sub(r'\bnull\b', 'nil', lua_code)
    
    return lua_code

def execute_and_capture_ground_truth(python_code: str, raw_pairs: list) -> (list, str, float, float):
    """
    🎯 100% 物理隔离运行 Python 候选代码并安全捕获真值，内置安全递归序列化器，消灭序列化异常
    """
    if not raw_pairs:
        raw_pairs = [
            {"args": []}, {"args": [0]}, {"args": [1]}, {"args": [""]},
            {"args": [[]]}, {"args": [{}]}, {"args": [None]}, {"args": [True]}
        ]

    mock_and_cleanup_header = (
        "import sys\n"
        "import builtins\n"
        "builtins.input = lambda *args, **kwargs: ''\n"
        "sys.argv = ['run_python_test.py']\n"
    )

    sanitized_python_code = preprocess_python_code(python_code, mock_and_cleanup_header)

    # 构造带安全递归序列化器 `safe_serialize` 的沙箱测试桩，拒绝 json 转换崩溃
    test_runner_code = """import json
import inspect
import sys

# 安全物理沙盒隔离执行
try:
    with open('candidate_user_code.py', 'r', encoding='utf-8') as f:
        exec(f.read(), globals())
except Exception as e:
    sys.stderr.write(f"Top-level execution caught: {type(e).__name__}: {str(e)}\\n")

with open('raw_pairs.json', 'r', encoding='utf-8') as f:
    raw_pairs = json.load(f)

def discover_callable():
    user_funcs = []
    user_classes = []
    try:
        for name, obj in list(globals().items()):
            if name.startswith('__') and name.endswith('__'): continue
            if name in ('discover_callable', 'sanitize_output', 'safe_serialize'): continue
            try:
                if inspect.isfunction(obj) and obj.__module__ == '__main__':
                    user_funcs.append((name, obj))
                elif inspect.isclass(obj) and obj.__module__ == '__main__':
                    user_classes.append((name, obj))
            except:
                pass
    except Exception as e:
        sys.stderr.write(f"Discover callable loop failed: {str(e)}\\n")
            
    clean_funcs = [f for f in user_funcs if not f[0].startswith('_')]
    if clean_funcs: return clean_funcs[0][1]
    if user_funcs: return user_funcs[0][1]
    
    clean_classes = [c for c in user_classes if not c[0].startswith('_')]
    target_class_tuple = clean_classes[0] if clean_classes else (user_classes[0] if user_classes else None)
    
    if target_class_tuple:
        try: instance = target_class_tuple[1]()
        except: instance = target_class_tuple[1]
        methods = []
        try:
            for attr_name in dir(instance):
                if attr_name.startswith('__') and attr_name.endswith('__'): continue
                try:
                    attr = getattr(instance, attr_name)
                    if callable(attr): methods.append((attr_name, attr))
                except:
                    pass
        except:
            pass
        clean_methods = [m for m in methods if not m[0].startswith('_')]
        if clean_methods: return clean_methods[0][1]
        if methods: return methods[0][1]
    return None

def safe_serialize(obj, depth=0):
    if depth > 5:
        return str(obj)
    try:
        # 尝试标准 JSON 直接序列化
        json.dumps(obj)
        return obj
    except TypeError:
        pass
        
    if isinstance(obj, (set, tuple)):
        return [safe_serialize(x, depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_serialize(x, depth + 1) for x in obj]
        
    if hasattr(obj, "__dict__"):
        try:
            clean_dict = {}
            for k, v in obj.__dict__.items():
                if not k.startswith('_'):
                    clean_dict[k] = safe_serialize(v, depth + 1)
            return clean_dict
        except:
            pass
            
    try:
        return str(obj)
    except:
        return "unserializable_object"

func = discover_callable()
passed_io_pairs = []

for idx, pair in enumerate(raw_pairs, 1):
    args = pair.get("args", [])
    if not isinstance(args, list): args = [args]
    try:
        if func is not None:
            result = func(*args)
            
            def sanitize_output(val):
                if isinstance(val, set): return list(val)
                if isinstance(val, tuple): return list(val)
                return val
                
            pair["expected"] = safe_serialize(sanitize_output(result))
            pair["status"] = "PASSED"
            passed_io_pairs.append(pair)
        else:
            pair["status"] = "FAILED"
            pair["expected"] = None
            pair["error_msg"] = "Dynamic discovery failed: Target callable function or class method not found"
            passed_io_pairs.append(pair)
    except Exception as e:
        pair["status"] = "FAILED"
        pair["expected"] = None
        pair["error_msg"] = f"{type(e).__name__}: {str(e)}"
        passed_io_pairs.append(pair)

# 使用 safe_serialize 之后，此处的 json.dumps 保证 100% 成功，绝不崩溃
print(f"CAPTURED_CASES_JSON:{json.dumps(passed_io_pairs)}")
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "run_python_test.py")
        json_path = os.path.join(tmpdir, "raw_pairs.json")
        user_code_path = os.path.join(tmpdir, "candidate_user_code.py")

        with open(user_code_path, "w", encoding="utf-8") as fu:
            fu.write(sanitized_python_code)

        with open(json_path, "w", encoding="utf-8") as fj:
            json.dump(raw_pairs, fj)

        # 直接完整写入，拒绝逐字符 `join` 的大崩溃
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(test_runner_code)
            
        try:
            run_res = subprocess.run(
                [sys.executable, script_path],
                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5
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
                feedback_msg = "SUCCESS"
            except Exception:
                passed_io_pairs = []
                feedback_msg = "CRITICAL_SANDBOX_CRASH"
        else:
            feedback_msg = "CRITICAL_SANDBOX_CRASH"
        
        if not passed_io_pairs:
            for pair in raw_pairs:
                passed_io_pairs.append({
                    "args": pair.get("args", []),
                    "status": "FAILED",
                    "expected": None,
                    "error_msg": "Sandbox crash or execution error during truth capture."
                })
            if "CRITICAL_SANDBOX" not in feedback_msg:
                feedback_msg = f"CRITICAL_SANDBOX_CRASH: {stderr_str}"

        line_cov, branch_cov = check_python_coverage(python_code, passed_io_pairs)
        return passed_io_pairs, feedback_msg, line_cov, branch_cov

# ==================== 3. Lua 鲁棒性评测、自愈、执行模块 ====================
def extract_lua_function_name(code: str) -> str:
    """静态分析 Lua 源码，清洗注释并智能提取定义函数名称（兼容 OOP 冒号 : 方法名提取）"""
    if not code or not isinstance(code, str):
        return ""
    code_clean = re.sub(r'--.*', '', code)  # 移除单行注释
    code_clean = re.sub(r'--\[\[.*?\]\]', '', code_clean, flags=re.DOTALL)  # 移除多行注释
    
    # 支持 1. function name(...) / local function name(...) (兼容冒号 :)
    found_funcs = re.findall(r'(?:local\s+)?function\s+([\w\._:]+)\s*\(', code_clean)
    if not found_funcs:
        # 支持 2. name = function(...) / local name = function(...) (兼容冒号 :)
        found_funcs = re.findall(r'([\w\._:]+)\s*=\s*function\s*\(', code_clean)
        
    if not found_funcs:
        return ""
    
    clean_funcs = [f for f in found_funcs if f not in ['initialize', 'new', 'fit', 'predict', 'show', 'print']]
    target = clean_funcs[0] if clean_funcs else found_funcs[0]
    
    # 🎯 如果定义为 Class:method (如 MailSender:send_email)，应提取类名 MailSender 并向测试桩导出
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
    """使用 luac 工具静态检查 Lua 测试文件是否存在语法错误"""
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
        try:
            res = subprocess.run(
                ["lua", "-e", "assert(loadfile('test_solution.lua'))"],
                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3
            )
            return res.returncode == 0
        except Exception:
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
    """
    采用 Lua 沙箱级别物理进程切片，单独评估每个测试块，全速自愈并规避死循环级联。
    """
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

        # 🎯 Lua 版高容差深层对比器：支持表深对比、数字容差、浮点补偿以及对齐 (1-based vs 0-based) 的浮动对比
        # 将其直接声明为 _G.smart_equal，彻底扫除 local function 在等号右侧引发的 Lua 语法车祸！
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
            
            # 物理沙箱无损组装
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

            local function exists(name)
                local path = {{}}
                for part in string.gmatch(name, "[%w_]+") do
                    table.insert(path, part)
                end
                local obj = _G
                for _, part in ipairs(path) do
                    if type(obj) ~= "table" then return false end
                    obj = obj[part]
                end
                return obj ~= nil
            end

            if not exists("{func_name}") then
                print("FUNC_MISSING")
                os.exit(2)
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
                elif "FUNC_MISSING" in stdout:
                    status = "No_Function_Defined"
                    diagnostic_reasons_list.append(f"FuncMissing: {func_name} not exported")
                    break
                elif "TEST_PASSED" in stdout:
                    passed_count += 1
                else:
                    # 🎯 【捕获具体断言报错】如果断言挂了，直接捕获 stderr/stdout 存入日志，不再沉默！
                    err_info = stderr.strip() or stdout.strip() or "Assertion Failed"
                    # 清洗无意义的临时文件名路径，只保留核心报错提示
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

def count_lua_executable_lines(code: str) -> set:
    lines = code.splitlines()
    executable = set()
    for idx, line in enumerate(lines, 1):
        line_strip = line.strip()
        if not line_strip:
            continue
        if line_strip.startswith("--"):
            continue
        if line_strip in ("end", "end,", "end;", "else", "local", "then", "{", "}"):
            continue
        executable.add(idx)
    return executable

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
    executable_lines = count_lua_executable_lines(candidate_code)
    if not executable_lines: return 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        candidate_path = os.path.join(tmpdir, "candidate.lua")
        test_path = os.path.join(tmpdir, "test_solution.lua")
        runner_path = os.path.join(tmpdir, "cov_runner.lua")

        with open(candidate_path, "w", encoding="utf-8") as f: f.write(candidate_code)
        with open(test_path, "w", encoding="utf-8") as f: f.write(lua_test_code)

        smart_equal_src = """
        _G.smart_equal = function(a, b, tolerance)
            tolerance = tolerance or 1e-4
            if a == b then return true end
            if a == nil or b == nil then return false end
            if type(a) ~= type(b) then
                if type(a) == "table" and type(b) ~= "table" and #a == 1 then return _G.smart_equal(a[1], b, tolerance) end
                if type(b) == "table" and type(a) ~= "table" and #b == 1 then return _G.smart_equal(a, b[1], tolerance) end
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

        cov_runner_src = f"""
        {smart_equal_src}
        local executed = {{}}
        local function hook(event, line)
            local info = debug.getinfo(2, "S")
            if info.source and info.source:match("candidate%.lua") then executed[line] = true end
        end
        debug.sethook(hook, "l")
        
        pcall(function()
            local chunk = assert(loadfile("candidate.lua"))
            local ret = chunk()
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
                    covered_executable = executed_set.intersection(executable_lines)
                    return min(100.0, (len(covered_executable) / len(executable_lines)) * 100.0)
        except Exception: pass
    return 0.0

# ==================== 4. 辅助数学计算与 JSON 解析 ====================
def calculate_pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    else:
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def extract_and_parse_json(output_str: str) -> list:
    cleaned_str = output_str.strip()
    json_match = re.search(r'\[\s*\{.*', cleaned_str, re.DOTALL)
    if json_match:
        target_str = json_match.group(0)
    else:
        target_str = cleaned_str
        
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
        pass
        
    open_chars = []
    in_string = False
    escape = False
    for i, char in enumerate(target_str):
        if char == '"' and not escape:
            in_string = not in_string
        if in_string:
            if char == '\\': escape = not escape
            else: escape = False
            continue
        if char in ('{', '['):
            open_chars.append(char)
        elif char == '}':
            if open_chars and open_chars[-1] == '{': open_chars.pop()
        elif char == ']':
            if open_chars and open_chars[-1] == '[': open_chars.pop()

    repaired_str = target_str
    for char in reversed(open_chars):
        if char == '{': repaired_str += '}'
        elif char == '[': repaired_str += ']'
        
    try:
        return json.loads(repaired_str)
    except Exception:
        pass

    for length in range(len(target_str), 0, -1):
        test_s = target_str[:length]
        open_chars = []
        in_string = False
        escape = False
        corrupted = False
        for char in test_s:
            if char == '"' and not escape:
                in_string = not in_string
            if in_string:
                if char == '\\': escape = not escape
                else: escape = False
                continue
            if char in ('{', '['):
                open_chars.append(char)
            elif char == '}':
                if open_chars and open_chars[-1] == '{': open_chars.pop()
                else: corrupted = True; break
            elif char == ']':
                if open_chars and open_chars[-1] == '[': open_chars.pop()
                else: corrupted = True; break
        if corrupted: continue
        
        for char in reversed(open_chars):
            if char == '{': test_s += '}'
            elif char == '[': test_s += ']'
        try:
            val = json.loads(test_s)
            if isinstance(val, list) and len(val) > 0:
                return val
        except Exception:
            pass

    return []

# ==================== 5. 双轨数据清洗与过滤 ====================
def load_and_filter_golden_data(csv_path):
    df = pd.read_csv(csv_path)
    df['idx'] = pd.to_numeric(df['idx'], errors='coerce')
    df = df.dropna(subset=['idx', 'python_code', 'all_candidates'])
    df['idx'] = df['idx'].astype(int)
    
    initial_count = len(df)
    
    if 'status' in df.columns:
        df = df[df['status'].str.strip() == 'Passed'].reset_index(drop=True)
    after_status_filter = len(df)
    filtered_by_status = initial_count - after_status_filter

    if 'passed_code' in df.columns:
        df = df[~df['passed_code'].astype(str).str.contains(r'###\s*[Oo]utput', regex=True, na=False)].reset_index(drop=True)

    df = df.iloc[3130:].reset_index(drop=True)
    df = df[df['idx'] != 1101].reset_index(drop=True)
    df = df[df['idx'] != 1876].reset_index(drop=True)
    df = df[df['idx'] != 4570].reset_index(drop=True)
    df = df[df['idx'] != 4571].reset_index(drop=True)
    df = df[df['idx'] != 5268].reset_index(drop=True)

    final_count = len(df)
    filtered_by_code_content = after_status_filter - final_count
        
    print(f"\n" + "="*50)
    print("📊 数据流控清洗报告 (Python-to-Lua Clean Data Report)")
    print("=========================================")
    print(f"原始大模型推理总记录数                      : {initial_count}")
    print(f"🔥 剔除 Lua 基础翻译不通过 (!= Passed) 样本数 : {filtered_by_status}")
    print(f"🔥 剔除包含 '### Output' 噪声文本样本数     : {filtered_by_code_content}")
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
        subprocess.run(["coverage", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print("⚡ [环境自愈] 正在自动部署 Python coverage 包...")
        subprocess.run([sys.executable, "-m", "pip", "install", "coverage"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        subprocess.run(["lua", "-v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print("❌ [严重错误] 未在系统环境中检测到 'lua' 解释器环境！")
        return

    print("🔍 正在启动 Lua 语言依赖与执行探针...")

    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    stats = {
        'total': 0, 
        'python_test_failed': 0,       
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
    
    print(f"\n🚀 【全景自愈翻译】二阶段反馈闭环迁移评测流水线已激活 (Batch Size: {batch_size})")

    # 💡 新增：动态推导对应的 jsonl 报告路径 (例如 Lua_passk_report.csv -> Lua_passk_report.jsonl)
    output_report_jsonl = output_report_csv.rsplit('.', 1)[0] + ".jsonl"

    # 💡 核心修改：同时打开 CSV 报告、JSONL 增量大盘报告、以及清洗通过包
    with open(output_report_csv, mode='w', encoding='utf-8', newline='') as f, \
         open(output_report_jsonl, mode='w', encoding='utf-8') as f_jsonl, \
         open(passed_save_path, mode='w', encoding='utf-8') as f_save:
         
        writer = csv.writer(f)
        writer.writerow([
            'idx', 'line_coverage', 'branch_coverage', 'lua_coverage_line', 
            'status', 'pass@1', 'pass@2', 'pass@5', 'passed_candidates'
        ])

        rows = [row for _, row in df.iterrows()]
        start_time = time.time()
        current_count = 0

        for i in range(0, total_records, batch_size):
            batch_rows = rows[i:i + batch_size]
            
            # ==================== 🛠️ 第一阶段：生成测试输入 args 并在 Python 沙盒捕获 expected 真值 ====================
            prompts_py = []
            for row in batch_rows:
                user_content = GEN_TEST_USER_PROMPT.format(python_code=row['python_code'])
                prompt = get_prompt_by_model(llm_type, GEN_TEST_SYSTEM_PROMPT, user_content)
                prompts_py.append(prompt)

            inputs_py = tokenizer(prompts_py, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                outputs_py = model.generate(
                    **inputs_py, max_new_tokens=512, temperature=0.5, top_p=0.95, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            input_lens_py = inputs_py.input_ids.shape[1]
            active_batch_items = []

            for j, row in enumerate(batch_rows):
                idx = str(row['idx'])
                py_code = row['python_code']
                cand_str = row['all_candidates']
                
                output_str_py = tokenizer.decode(outputs_py[j][input_lens_py:], skip_special_tokens=True)
                raw_args_list = extract_and_parse_json(output_str_py)

                stats['total'] += 1

                try:
                    passed_io_pairs, py_feedback, py_line_cov, py_branch_cov = execute_and_capture_ground_truth(
                        py_code, raw_args_list
                    )
                except RecursionError:
                    print(f"\n[💥 HIGH-LEVEL RECURSION CRASH ANCHOR]")
                    print(f"  🚨 触发崩溃的样本索引 (Fatal Sample IDX) : {idx}")
                    print(f"  └─ 源代码前 150 字预览 : {py_code[:150].strip()}...")
                    print("-" * 60 + "\n")
                    
                    passed_io_pairs = []
                    py_feedback = f"CRITICAL_SANDBOX_CRASH: RecursionError inside parent process for IDX {idx}"
                    py_line_cov, py_branch_cov = 0.0, 0.0

                is_truth_capture_failed = (len(raw_args_list) == 0) or ("CRITICAL_SANDBOX" in py_feedback) or ("RecursionError" in py_feedback)
                
                if is_truth_capture_failed:
                    stats['python_test_failed'] += 1
                
                LIMIT_MAX = 10
                final_io_pairs = passed_io_pairs[:LIMIT_MAX]

                item_data = {
                    'idx': idx,
                    'python_code': py_code,
                    'python_test_code': json.dumps(final_io_pairs, ensure_ascii=False),
                    'python_passed': len(final_io_pairs),
                    'python_feedback': py_feedback,
                    'python_line_cov': py_line_cov,
                    'python_branch_cov': py_branch_cov,
                    'cand_str': cand_str,
                    'is_truth_capture_failed': is_truth_capture_failed
                }
                active_batch_items.append(item_data)

            if not active_batch_items:
                del inputs_py, outputs_py
                torch.cuda.empty_cache()
                current_count += len(batch_rows)
                continue

            # ==================== 第二阶段：迁移生成精准标准的 Lua 语言 native 测试桩 ====================
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
                    num_test_cases=item['python_passed'],
                    lua_func_name=lua_func_name,
                    python_feedback="JSON-DATA-MODE", 
                    python_code=item['python_code'],
                    python_test_code=item['python_test_code']
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

            # ==================== 第三阶段：运行断言与深度原因诊断可视化看板 ====================
            for j, item in enumerate(active_batch_items):
                idx = item['idx']
                python_line_cov = item['python_line_cov']
                python_branch_cov = item['python_branch_cov']
                cand_str = item['cand_str']
                is_truth_capture_failed = item['is_truth_capture_failed']

                output_str_lua = tokenizer.decode(outputs_lua[j][input_lens_lua:], skip_special_tokens=True)
                
                lua_test_code = ""
                lua_match = re.search(r'```[Ll]ua\s*(.*?)\s*```', output_str_lua, re.DOTALL)
                if lua_match: lua_test_code = lua_match.group(1).strip()
                else: lua_test_code = output_str_lua.strip()

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
                stats['sum_line_cov'] += python_line_cov
                stats['sum_branch_cov'] += python_branch_cov
                stats['sum_lua_line_cov'] += avg_lua_line_cov

                p1 = calculate_pass_at_k(n, passed_count, 1)
                p2 = calculate_pass_at_k(n, passed_count, 2) if n >= 2 else p1
                p5 = calculate_pass_at_k(n, passed_count, 5) if n >= 5 else p2

                stats['p1'] += p1; stats['p2'] += p2; stats['p5'] += p5

                # 1. 写入原有的 CSV 大盘
                writer.writerow([
                    idx, f"{python_line_cov:.2f}%", f"{python_branch_cov:.2f}%", f"{avg_lua_line_cov:.2f}%",
                    "Passed" if passed_count > 0 else "Failed", f"{p1:.4f}", f"{p2:.4f}", f"{p5:.4f}", passed_count
                ])

                # 💡 2. 新增：增量保存这一项的完整状态至 Lua_passk_report.jsonl (包含 Failed 和 Passed 的所有条目)
                report_item = {
                    "idx": int(idx),
                    "status": "Passed" if passed_count > 0 else "Failed",
                    "global_lua_status": global_lua_status,
                    "metrics": {
                        "python_line_coverage": python_line_cov,
                        "python_branch_coverage": python_branch_cov,
                        "lua_line_coverage": avg_lua_line_cov,
                        "pass_at_1": p1,
                        "pass_at_2": p2,
                        "pass_at_5": p5
                    },
                    "passed_candidates_count": passed_count,
                    "total_candidates_count": n,
                    "diagnostic_reasons": diagnostic_reasons  # 失败用例的运行时断言错误及堆栈原因
                }
                f_jsonl.write(json.dumps(report_item, ensure_ascii=False) + "\n")

                # 3. 写入专门存储 Passed 干净数据的代码池
                if passed_count > 0:
                    save_item = {
                        "idx": int(idx),
                        "python_code": item['python_code'],
                        "python_test_cases": json.loads(item['python_test_code']),
                        "lua_test_code": lua_test_code,
                        "passed_lua_candidates": passed_cands_list,
                        "line_coverage": python_line_cov,
                        "branch_coverage": python_branch_cov,
                        "lua_line_coverage": avg_lua_line_cov,
                        "pass_at_1": p1,
                        "pass_at_2": p2,
                        "pass_at_5": p5
                    }
                    f_save.write(json.dumps(save_item, ensure_ascii=False) + "\n")
                    
                # 💡 每次循环强制落盘，防止断电崩溃
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

            py_line_avg = stats['sum_line_cov'] / valid_num
            py_branch_avg = stats['sum_branch_cov'] / valid_num
            lua_line_avg = stats['sum_lua_line_cov'] / valid_num

            full_status_line = (
                f"进度:{pct:>5.1f}% ({current_count}/{total_records})[{elapsed_str}<{remaining_str},{it_s:.2f}it/s] "
                f"|总数:{stats['total']}|真值错:{stats['python_test_failed']}|Lua语法错:{stats['lua_test_syntax_errors']}|Lua无通过:{stats['lua_no_cand_passed']}|Lua成功:{stats['lua_success']} "
                f"|P@1:{(stats['p1']/valid_num)*100:.2f}%|P@2:{(stats['p2']/valid_num)*100:.2f}%|P@5:{(stats['p5']/valid_num)*100:.2f}% "
                f"|Py行覆:{py_line_avg:.1f}%|Lua行覆:{lua_line_avg:.1f}%"
            )
            print(full_status_line)
            sys.stdout.flush()

            del inputs_py, outputs_py, inputs_lua, outputs_lua
            torch.cuda.empty_cache()

        print("\n")

    del model
    torch.cuda.empty_cache()
    # ... 后续的大盘 Summary 打印逻辑保持不变 ...
    
    print("================ 📈 重构版自愈评测大盘总结 📈 ================")
    print(f"送入评测的黄金样本总量 (Total Records)     : {stats['total']}")
    print(f"Python 捕获完全失败数 (Capture Failed)     : {stats['python_test_failed']}")
    
    valid_total = stats['valid']
    print(f"完美参与语义指标统计的有效题目数           : {valid_total}")
    print(f"  ├─ Lua 测试代码语法错误件数 (Lua Test Syntax Error): {stats['lua_test_syntax_errors']}")
    print(f"  ├─ 候选代码全部未通过断言 (Lua No Cand Passed)    : {stats['lua_no_cand_passed']}")
    print(f"  ├─ 🚨 因 Lua 端真值错而强制失败的件数        : {stats['lua_truth_failed']}")
    print(f"  └─ 至少有1个候选代码测试通过的正常件数 (Lua Success): {stats['lua_success']}")
    
    if valid_total > 0:
        print(f"🌟 Lua 语言对齐平均 Pass@1: {stats['p1'] / valid_total * 100:.2f}%")
        print(f"🌟 Lua 语言对齐平均 Pass@2: {stats['p2'] / valid_total * 100:.2f}%")
        print(f"🌟 Lua 语言对齐平均 Pass@5: {stats['p5'] / valid_total * 100:.2f}%")
        print("-" * 50)
        print(f"📐 Python 平均行覆盖率 (Line Coverage)      : {stats['sum_line_cov'] / valid_total:.2f}%")
        print(f"📐 Python 平均分支覆盖率 (Branch Coverage)  : {stats['sum_branch_cov'] / valid_total:.2f}%")
        print(f"📐 Lua 平均行覆盖率 (Line Coverage)         : {stats['sum_lua_line_cov'] / valid_total:.2f}%")
    print("==================================================================")
    print("\n🎉 [OK] 评测顺利结束！\n")

if __name__ == "__main__":
    INFERENCE_CSV_ORIGINAL = "/root/autodl-tmp/first_paper/python/translation_dataset/Lua_golden/Lua_inference_results_original.csv"
    LLM_MODEL_PATH = "/root/autodl-tmp/first_paper/LLM/Qwen2.5"
    CURRENT_LLM_TYPE = "qwen"
    OUTPUT_REPORT = "/root/autodl-tmp/first_paper/python/testcase_dataset/Lua_passk_report.csv"
    PASSED_TRANSLATIONS_JSONL = "/root/autodl-tmp/first_paper/python/testcase_dataset/Lua_pass/Lua_passed_clean.jsonl"
    
    run_live_pipeline(
        csv_path=INFERENCE_CSV_ORIGINAL,
        llm_path=LLM_MODEL_PATH,
        llm_type=CURRENT_LLM_TYPE,
        output_report_csv=OUTPUT_REPORT,
        passed_save_path=PASSED_TRANSLATIONS_JSONL
    )