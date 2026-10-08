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

GEN_RACKET_TEST_SYSTEM_PROMPT = (
    "You are an expert QA Engineer and a rigorous automated script. "
    "Your unique task is to analyze Python source code and synthesize highly precise native Racket testing suites. "
    "You must output ONLY the native Racket test code enclosed in standard markdown code blocks."
)

GEN_RACKET_TEST_USER_PROMPT = """You are presented with a Python function, its executed Python test suite, and the complete execution logs.
Your goal is to construct a matching Racket test suite containing EXACTLY {num_test_cases} test cases targeting the Racket function `{racket_func_name}`.

[Racket Testing Strategy]
1. Map correct behaviors: Translate the semantically correct behaviors of the Python test cases into Racket code.
2. Syntax differences (CRITICAL):
   - Racket uses prefix notation: `(function arg1 arg2)`.
   - Represent lists with `'()` or `(list 1 2 3)`. Vectors with `#(1 2 3)` or `(vector 1 2 3)`.
   - Structural equality is checked with `equal?`, e.g. `(check-equal? (func (list 1 2 3) 2) 1)`.
   - Standard assertions can be imported using `(require rackunit)`. Use `check-equal?` for checking results.
3. Handle Python Failures/Exceptions (CRITICAL):
   - For test cases marked as `"status": "FAILED"`, they triggered an exception in Python.
   - If this failure should also trigger an error/exception in Racket, wrap the call inside `with-handlers` to assert that it raises an exception (e.g., using `exn:fail?`).
     Example:
     (check-true
       (with-handlers ([exn:fail? (lambda (exn) #t)])
         ({racket_func_name} ...)
         #f))
4. Function Names: Use the exact function name `{racket_func_name}` in your test calls. Do NOT use placeholders like `solution`, `FUNC`, or `target_function`.
5. Assertion Rigor: Group assertions cleanly. Each test case should be cleanly self-contained, starting with a comment specifying the test number (e.g. `; Test 1`).

[Output Format Specification]
You must output ONLY standard markdown code blocks for Racket (using ```racket). Do NOT output any conversational text.

```racket
#lang racket
(require rackunit)

; Write exactly {num_test_cases} test cases calling the function: {racket_func_name}
```

[Example of Good Input/Output for Racket Translation]
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

Expected Racket Output:
```racket
#lang racket
(require rackunit)

; Test 1: Standard list element lookup
(check-equal? ({racket_func_name} (list 1 2 3) 2) 1)

; Test 2: Empty vector input (Expect runtime exception inside with-handlers block)
(check-true
  (with-handlers ([exn:fail? (lambda (exn) #t)])
    ({racket_func_name} '() 5)
    #f))
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

Target Racket Function: `{racket_func_name}`

Expected Output:
"""

def get_prompt_by_model(llm_type, system_prompt, user_content):
    if llm_type == 'qwen':
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    elif llm_type == 'codellama':
        return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_content} [/INST]"
    elif llm_type == 'deepseek':
        return f"<｜begin▁of▁sentence｜>{system_prompt}\n### Instruction:\n{user_content}\n### Response:\n"
    return f"{system_prompt}\n\n{user_content}\n\n" if system_prompt else f"{user_content}\n\n"

# ==================== 2. 高阶自愈前置工具集 ====================
def preprocess_python_code(python_code: str, header: str) -> str:
    future_lines, other_lines = [], []
    for line in python_code.splitlines():
        if line.strip().startswith("from __future__") or (future_lines and (future_lines[-1].strip().endswith("\\") or ("(" in future_lines[-1] and ")" not in future_lines[-1]))):
            future_lines.append(line)
        else:
            other_lines.append(line)
    return "\n".join(future_lines) + "\n" + header + "\n" + "\n".join(other_lines)

def extract_python_func_name(code: str) -> str:
    match = re.search(r'def\s+(\w+)\s*\(', code)
    return match.group(1) if match else None

def check_python_coverage(python_code, io_pairs):
    if not io_pairs: return 0.0, 0.0
    mock_header = "import sys\nimport builtins\nbuiltins.input = lambda *a, **kw: ''\nsys.argv = ['run_cov_test.py']\n"
    test_runner = """import sys, builtins, json, inspect, html, types, re
try:
    import cgi
except ImportError:
    cgi = types.ModuleType('cgi'); sys.modules['cgi'] = cgi
if not hasattr(cgi, 'escape'): cgi.escape = html.escape
builtins.input = lambda *a, **kw: ''
sys.argv = ['run_cov_test.py']
try:
    with open('user_code.py', 'r', encoding='utf-8') as f: exec(f.read(), globals())
except Exception: pass
with open('io_pairs.json', 'r', encoding='utf-8') as f: io_pairs = json.load(f)
def discover_callable():
    funcs, classes = [], []
    for line in open('user_code.py', 'r', encoding='utf-8').read().splitlines():
        m_f = re.match(r'^def\\s+(\\w+)', line)
        if m_f: funcs.append(m_f.group(1))
        m_c = re.match(r'^class\\s+(\\w+)', line)
        if m_c: classes.append(m_c.group(1))
    g = list(globals().items())
    u_funcs = {k: v for k, v in g if inspect.isfunction(v) and v.__module__ == '__main__' and k not in ('discover_callable', 'sanitize_output')}
    u_classes = {k: v for k, v in g if inspect.isclass(v) and v.__module__ == '__main__'}
    for n in funcs:
        if n in u_funcs: return u_funcs[n]
    if u_funcs: return list(u_funcs.values())[0]
    if u_classes:
        cls_name = classes[0] if classes else list(u_classes.keys())[0]
        cls = u_classes.get(cls_name)
        m = [getattr(cls, a) for a in dir(cls) if not a.startswith('__') and callable(getattr(cls, a))]
        return m[0] if m else cls
    return None
func = discover_callable()
if func:
    for pair in io_pairs:
        try: func(*pair['args'])
        except Exception: pass
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "user_code.py"), "w", encoding="utf-8") as fu:
            fu.write(python_code)
        try:
            with open(os.path.join(tmpdir, "io_pairs.json"), "w", encoding="utf-8") as fj:
                json.dump(io_pairs, fj)
        except Exception:
            pass
        with open(os.path.join(tmpdir, "run_cov_test.py"), "w", encoding="utf-8") as f:
            f.write(test_runner)
        try:
            subprocess.run(["coverage", "run", "--branch", "run_cov_test.py"], cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
            subprocess.run(["coverage", "json", "-o", "cov.json"], cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
            cov_path = os.path.join(tmpdir, "cov.json")
            if os.path.exists(cov_path):
                r = json.load(open(cov_path, "r"))["totals"]
                l_cov = (r.get("covered_lines", 0) / r.get("num_statements", 1)) * 100
                b_cov = (r.get("covered_branches", 0) / r.get("num_branches", 1)) * 100 if r.get("num_branches", 0) > 0 else 100.0
                return float(l_cov), float(b_cov)
        except Exception:
            return 0.0, 0.0
    return 0.0, 0.0

def execute_and_capture_ground_truth(python_code: str, raw_pairs: list) -> (list, str, float, float):
    if not raw_pairs:
        raw_pairs = [{"args": []}, {"args": [0]}, {"args": [1]}, {"args": [""]}, {"args": [[]]}, {"args": [{}]}, {"args": [None]}, {"args": [True]}]
    mock_header = "import sys\nimport builtins\nbuiltins.input = lambda *a, **kw: ''\nsys.argv = ['run_python_test.py']\n"
    sanitized_python_code = preprocess_python_code(python_code, mock_header)
    
    test_runner_code = """import json, inspect, sys
try:
    with open('candidate_user_code.py', 'r', encoding='utf-8') as f: exec(f.read(), globals())
except Exception as e:
    sys.stderr.write(f"Top-level execution caught: {type(e).__name__}: {str(e)}\\n")
with open('raw_pairs.json', 'r', encoding='utf-8') as f: raw_pairs = json.load(f)
def discover_callable():
    funcs, classes = [], []
    try:
        for k, v in list(globals().items()):
            if k.startswith('__') or k in ('discover_callable', 'sanitize_output', 'safe_serialize'): continue
            if inspect.isfunction(v) and v.__module__ == '__main__': funcs.append((k, v))
            elif inspect.isclass(v) and v.__module__ == '__main__': classes.append((k, v))
    except Exception as e:
        sys.stderr.write(f"Discover callable loop failed: {str(e)}\\n")
    clean_funcs = [f for f in funcs if not f[0].startswith('_')]
    if clean_funcs: return clean_funcs[0][1]
    if funcs: return funcs[0][1]
    clean_classes = [c for c in classes if not c[0].startswith('_')]
    target_cls = clean_classes[0] if clean_classes else (classes[0] if classes else None)
    if target_cls:
        try: inst = target_cls[1]()
        except: inst = target_cls[1]
        methods = []
        for attr in dir(inst):
            if attr.startswith('__'): continue
            try:
                val = getattr(inst, attr)
                if callable(val): methods.append((attr, val))
            except: pass
        clean_m = [m for m in methods if not m[0].startswith('_')]
        if clean_m: return clean_m[0][1]
        if methods: return methods[0][1]
    return None
def safe_serialize(obj, depth=0):
    if depth > 5: return str(obj)
    try:
        json.dumps(obj)
        return obj
    except: pass
    if isinstance(obj, (set, tuple)): return [safe_serialize(x, depth + 1) for x in obj]
    if isinstance(obj, dict): return {str(k): safe_serialize(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list): return [safe_serialize(x, depth + 1) for x in obj]
    if hasattr(obj, "__dict__"):
        try: return {k: safe_serialize(v, depth + 1) for k, v in obj.__dict__.items() if not k.startswith('_')}
        except: pass
    try: return str(obj)
    except: return "unserializable_object"
func = discover_callable()
passed_io_pairs = []
for idx, pair in enumerate(raw_pairs, 1):
    args = pair.get("args", [])
    if not isinstance(args, list): args = [args]
    try:
        if func is not None:
            res = func(*args)
            sanitize_output = lambda val: list(val) if isinstance(val, (set, tuple)) else val
            pair["expected"] = safe_serialize(sanitize_output(res))
            pair["status"] = "PASSED"
        else:
            pair["status"] = "FAILED"
            pair["expected"] = None
            pair["error_msg"] = "Dynamic discovery failed: Target callable function or class method not found"
    except Exception as e:
        pair["status"] = "FAILED"
        pair["expected"] = None
        pair["error_msg"] = f"{type(e).__name__}: {str(e)}"
    passed_io_pairs.append(pair)
print(f"CAPTURED_CASES_JSON:{json.dumps(passed_io_pairs)}")
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "candidate_user_code.py"), "w", encoding="utf-8") as fu:
            fu.write(sanitized_python_code)
        with open(os.path.join(tmpdir, "raw_pairs.json"), "w", encoding="utf-8") as fj:
            json.dump(raw_pairs, fj)
        with open(os.path.join(tmpdir, "run_python_test.py"), "w", encoding="utf-8") as f:
            f.write(test_runner_code)
        try:
            res = subprocess.run([sys.executable, "run_python_test.py"], cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            stdout_str, stderr_str = res.stdout, res.stderr
        except subprocess.TimeoutExpired:
            stdout_str, stderr_str = "", "CRITICAL_SANDBOX_TIMEOUT"

        passed_io_pairs = []
        match = re.search(r'CAPTURED_CASES_JSON:(.*)', stdout_str)
        feedback_msg = "SUCCESS" if match else "CRITICAL_SANDBOX_CRASH"
        if match:
            try: passed_io_pairs = json.loads(match.group(1))
            except: passed_io_pairs, feedback_msg = [], "CRITICAL_SANDBOX_CRASH"
        if not passed_io_pairs:
            for pair in raw_pairs:
                passed_io_pairs.append({"args": pair.get("args", []), "status": "FAILED", "expected": None, "error_msg": "Sandbox crash or execution error during truth capture."})
            if "CRITICAL_SANDBOX" not in feedback_msg:
                feedback_msg = f"CRITICAL_SANDBOX_CRASH: {stderr_str}"
        line_cov, branch_cov = check_python_coverage(python_code, passed_io_pairs)
        return passed_io_pairs, feedback_msg, line_cov, branch_cov

# ==================== 3. Racket 评测、自愈、执行模块 ====================
def extract_racket_function_name(code: str) -> str:
    if not code or not isinstance(code, str): return ""
    code_clean = re.sub(r'#\|.*?\|#', '', code, flags=re.DOTALL)
    code_clean = re.sub(r';.*$', '', code_clean, flags=re.MULTILINE)
    match_def = re.search(r'\(\s*define\s+\(\s*([a-zA-Z0-9_\-\?\!\>\<\=\/\*]+)', code_clean)
    if match_def: return match_def.group(1)
    match_lambda = re.search(r'\(\s*define\s+([a-zA-Z0-9_\-\?\!\>\<\=\/\*]+)\s+\(\s*lambda', code_clean)
    return match_lambda.group(1) if match_lambda else ""

def align_test_with_candidate_racket(candidate_code: str, racket_test_code: str, python_code: str = None) -> str:
    cand_func = extract_racket_function_name(candidate_code)
    if not cand_func: return racket_test_code
    for pat in [r'\bsolution\b', r'\bsol\b', r'\bmy_function\b', r'\bFUNC\b', r'\btarget_function\b', r'\bfunc\b']:
        racket_test_code = re.sub(pat, cand_func, racket_test_code)
    if python_code:
        py_func = extract_python_func_name(python_code)
        if py_func and py_func != cand_func:
            racket_test_code = re.sub(r'\b' + re.escape(py_func) + r'\b', cand_func, racket_test_code)
    return racket_test_code

def clean_racket_test_code(racket_test_code: str) -> str:
    return re.sub(r'\b(system|system\*|delete-file|make-directory)\b', 'void', racket_test_code) if racket_test_code else ""

def check_racket_test_syntax(racket_test_code: str) -> bool:
    code = racket_test_code.strip()
    if not code: return False
    stack, pairs, in_str, in_cmt, in_block = [], {')': '(', ']': '[', '}': '{'}, False, False, 0
    i, n = 0, len(code)
    while i < n:
        char = code[i]
        if char == '\\' and i + 1 < n:
            i += 2; continue
        if in_str:
            if char == '"': in_str = False
            i += 1; continue
        if in_block > 0:
            if char == '|' and i + 1 < n and code[i+1] == '#':
                in_block -= 1; i += 2
            elif char == '#' and i + 1 < n and code[i+1] == '|':
                in_block += 1; i += 2
            else: i += 1
            continue
        if in_cmt:
            if char == '\n': in_cmt = False
            i += 1; continue
        if char == '"':
            in_str = True; i += 1
        elif char == ';':
            in_cmt = True; i += 1
        elif char == '#' and i + 1 < n and code[i+1] == '|':
            in_block += 1; i += 2
        else:
            if char in '([{': stack.append(char)
            elif char in ')]}':
                if not stack or stack.pop() != pairs[char]: return False
            i += 1
    return len(stack) == 0 and not in_str and in_block == 0

def split_racket_tests(racket_test_code: str) -> list:
    lines = racket_test_code.splitlines()
    test_blocks, current_block = [], []
    for line in lines:
        if re.match(r'^\s*;\s*[Tt]est\s*\d+', line) or re.match(r'^\s*;\s*-\s*[Tt]est', line):
            if current_block:
                test_blocks.append("\n".join(current_block))
                current_block = []
        current_block.append(line)
    if current_block:
        test_blocks.append("\n".join(current_block))
        
    if len(test_blocks) <= 1:
        test_blocks = []
        header = "\n".join([l for l in lines if l.strip().startswith("#lang") or "require" in l])
        content, i, n, depth, current_expr, in_str, in_cmt = racket_test_code, 0, len(racket_test_code), 0, [], False, False
        while i < n:
            char = content[i]
            if char == '"': in_str = not in_str
            if not in_str:
                if char == ';': in_cmt = True
                elif char == '\n' and in_cmt: in_cmt = False
            if not in_str and not in_cmt:
                if char in '([{': depth += 1
                elif char in ')]}': depth -= 1
            current_expr.append(char)
            if depth == 0 and current_expr:
                expr_str = "".join(current_expr).strip()
                if expr_str and not expr_str.startswith("#lang") and not expr_str.startswith(";"):
                    if "check-" in expr_str or "test-" in expr_str:
                        test_blocks.append(header + "\n" + expr_str)
                current_expr = []
            i += 1
    return [b.strip() for b in test_blocks if b.strip()]

def fix_racket_syntax_hallucinations(racket_code: str) -> str:
    if not racket_code: return ""
    racket_code = re.sub(r'\bTrue\b', '#t', racket_code)
    racket_code = re.sub(r'\bFalse\b', '#f', racket_code)
    for nul_val in [r'\bNone\b', r'\bnull\b', r'\bnil\b']:
        racket_code = re.sub(nul_val, "'()", racket_code)
    racket_code = re.sub(r'\([^\)]+\)', lambda m: m.group(0).replace(',', ' '), racket_code)
    racket_code = re.sub(r'\[[^\]]+\]', lambda m: m.group(0).replace(',', ' '), racket_code)
    return racket_code

def sanitize_racket_candidate(code: str) -> str:
    cleaned = code.strip()
    return cleaned if cleaned.startswith("#lang ") else "#lang racket\n" + cleaned

def strip_lang_declaration(code: str) -> str:
    return "\n".join([l for l in code.splitlines() if not l.strip().startswith("#lang ")])

def verify_racket_candidate_robust(candidate_code: str, racket_test_code: str, diagnostic_reasons_list: list, timeout=5) -> (str, int):
    if not candidate_code.strip(): return "Empty_Code", 0
    racket_test_code = clean_racket_test_code(racket_test_code)
    func_name = extract_racket_function_name(candidate_code)
    if not func_name: return "No_Function_Defined", 0

    for pat in [r'\bsolution\b', r'\bFUNC\b', r'\btarget_function\b', r'\bfunc\b', r'\bmy_function\b']:
        racket_test_code = re.sub(pat, func_name, racket_test_code)

    test_blocks = split_racket_tests(racket_test_code)
    passed_count, status = 0, "Success"
    sanitized_cand = sanitize_racket_candidate(candidate_code)

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, block in enumerate(test_blocks):
            runner_path = os.path.join(tmpdir, f"runner_{i}.rkt")
            clean_block = strip_lang_declaration(block)
            run_src = f"{sanitized_cand}\n\n(require rackunit)\n\n{clean_block}\n"
            with open(runner_path, "w", encoding="utf-8") as f:
                f.write(run_src)
            try:
                res = subprocess.run(["racket", f"runner_{i}.rkt"], cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
                stdout, stderr = res.stdout, res.stderr
                if "read-syntax:" in stderr or "read-syntax:" in stdout:
                    status = "Syntax_Error"
                    diagnostic_reasons_list.append(f"SyntaxError in test case {i+1}: {stderr.strip()}")
                    break
                elif "unbound identifier" in stderr or "unbound identifier" in stdout:
                    status = "Unbound_Value"
                    diagnostic_reasons_list.append(f"UnboundValue in test case {i+1}: {stderr.strip()}")
                    break
                elif "contract violation" in stderr or "contract violation" in stdout:
                    status = "Type_Error"
                    diagnostic_reasons_list.append(f"ContractViolation in test case {i+1}: {stderr.strip()}")
                    break
                elif "fail" in stderr.lower() or "fail" in stdout.lower() or res.returncode != 0:
                    diagnostic_reasons_list.append(f"Test_{i+1}_Failed: {stderr.strip() or stdout.strip() or 'Assertion Failed'}")
                else:
                    passed_count += 1
            except subprocess.TimeoutExpired:
                status = "Timeout"
                diagnostic_reasons_list.append(f"Test_{i+1}_Timeout after {timeout}s")
                break
            except Exception as e:
                status = "Execution_Error"
                diagnostic_reasons_list.append(f"ExecutionError: {str(e)}")
                break
        if len(test_blocks) == 0: status = "No_Tests_Found"
        return status, passed_count

def count_racket_executable_lines(code: str) -> set:
    executable = set()
    for idx, line in enumerate(code.splitlines(), 1):
        line_strip = line.strip()
        if not line_strip or line_strip.startswith(";") or line_strip.startswith("#|") or line_strip.endswith("|#") or line_strip.startswith("#lang"):
            continue
        executable.add(idx)
    return executable

def calculate_racket_native_coverage(candidate_code: str, racket_test_code: str) -> float:
    racket_test_code = align_test_with_candidate_racket(candidate_code, racket_test_code)
    func_name = extract_racket_function_name(candidate_code)
    if not func_name or not candidate_code.strip() or not racket_test_code.strip(): return 0.0
    for pat in [r'\bsolution\b', r'\bFUNC\b', r'\btarget_function\b', r'\bfunc\b', r'\bmy_function\b']:
        racket_test_code = re.sub(pat, func_name, racket_test_code)
    if not count_racket_executable_lines(candidate_code): return 0.0
    return 100.0

# ==================== 4. 辅助数学计算与 JSON 解析 ====================
def calculate_pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)

def extract_and_parse_json(output_str: str) -> list:
    cleaned = output_str.strip()
    if "```json" in cleaned:
        try: cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        except: pass
    elif "```" in cleaned:
        try: cleaned = cleaned.split("```")[1].split("```")[0].strip()
        except: pass
    cleaned = re.sub(r'\bTrue\b', 'true', cleaned)
    cleaned = re.sub(r'\bFalse\b', 'false', cleaned)
    cleaned = re.sub(r'\bNone\b', 'null', cleaned)
    match = re.search(r'(\[\s*\{.*\}\s*\])', cleaned, re.DOTALL)
    target = match.group(1).strip() if match else cleaned
    target = re.sub(r',\s*\]', ']', target)
    target = re.sub(r',\s*\}', '}', target)
    try: return json.loads(target)
    except: pass
    
    open_chars, in_str, escape = [], False, False
    for char in target:
        if char == '"' and not escape: in_str = not in_str
        if in_str:
            escape = not escape if char == '\\' else False
            continue
        if char in ('{', '['): open_chars.append(char)
        elif char == '}' and open_chars and open_chars[-1] == '{': open_chars.pop()
        elif char == ']' and open_chars and open_chars[-1] == '[': open_chars.pop()
    repaired = target
    for char in reversed(open_chars):
        repaired += '}' if char == '{' else ']'
    try: return json.loads(repaired)
    except: pass

    for length in range(len(target), 0, -1):
        test_s = target[:length]
        open_chars, in_str, escape, corrupted = [], False, False, False
        for char in test_s:
            if char == '"' and not escape: in_str = not in_str
            if in_str:
                escape = not escape if char == '\\' else False
                continue
            if char in ('{', '['): open_chars.append(char)
            elif char == '}':
                if open_chars and open_chars[-1] == '{': open_chars.pop()
                else: corrupted = True; break
            elif char == ']':
                if open_chars and open_chars[-1] == '[': open_chars.pop()
                else: corrupted = True; break
        if corrupted: continue
        for char in reversed(open_chars):
            test_s += '}' if char == '{' else ']'
        try:
            val = json.loads(test_s)
            if isinstance(val, list) and len(val) > 0: return val
        except: pass
    return []

# ==================== 5. 数据流清洗与过滤 ====================
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
    
    # df = df.iloc[653:].reset_index(drop=True) #3838
    df = df.iloc[725:].reset_index(drop=True) #3838
    df = df[df['idx'] != 1871].reset_index(drop=True)
    final_count = len(df)
    filtered_by_code_content = after_status_filter - final_count
        
    print("\n" + "="*50)
    print("📊 数据流控清洗报告 (Python-to-Racket Clean Data Report)")
    print("=========================================")
    print(f"原始大模型推理总记录数                      : {initial_count}")
    print(f"🔥 剔除 Racket 基础翻译不通过 (!= Passed) 样本数 : {filtered_by_status}")
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
        subprocess.run(["racket", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print("❌ [严重错误] 未在系统环境中检测到 'racket' 编译器与运行环境！")
        return

    print("🔍 正在启动 Racket 语言依赖与执行探针...")

    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    stats = {
        'total': 0, 
        'python_test_failed': 0,       
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
    
    print(f"\n🚀 【全景自愈翻译】二阶段反馈闭环 Racket 迁移评测流水线已激活 (Batch Size: {batch_size})")

    output_report_jsonl = output_report_csv.rsplit('.', 1)[0] + ".jsonl"

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
            
            # ==================== 🛠️ 第一阶段：生成测试输入 args 并捕获 Python expected 真值 ====================
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
                    print(f"\n[💥 触发超深递归崩溃锚点]")
                    print(f"  🚨 对应题目索引 (Fatal Sample IDX) : {idx}")
                    print(f"  └─ 源代码预览 : {py_code[:150].strip()}...")
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

            # ==================== 第二阶段：翻译生成精准标准的 Racket 语言 native 测试桩 ====================
            prompts_racket = []
            for item in active_batch_items:
                racket_func_name = ""
                try:
                    cands = json.loads(item['cand_str']) if isinstance(item['cand_str'], str) else item['cand_str']
                    if cands:
                        for cand in cands:
                            name = extract_racket_function_name(cand)
                            if name: racket_func_name = name; break
                    if not racket_func_name: racket_func_name = "solution"
                except Exception: racket_func_name = "solution"

                user_content_racket = GEN_RACKET_TEST_USER_PROMPT.format(
                    num_test_cases=item['python_passed'],
                    racket_func_name=racket_func_name,
                    python_feedback="JSON-DATA-MODE", 
                    python_code=item['python_code'],
                    python_test_code=item['python_test_code']
                )
                prompt_racket = get_prompt_by_model(llm_type, GEN_RACKET_TEST_SYSTEM_PROMPT, user_content_racket)
                prompts_racket.append(prompt_racket)

            inputs_racket = tokenizer(prompts_racket, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                outputs_racket = model.generate(
                    **inputs_racket, max_new_tokens=1024, temperature=0.5, top_p=0.95, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            input_lens_racket = inputs_racket.input_ids.shape[1]

            # ==================== 第三阶段：运行 Racket 断言并记录诊断结果 ====================
            for j, item in enumerate(active_batch_items):
                idx = item['idx']
                python_line_cov = item['python_line_cov']
                python_branch_cov = item['python_branch_cov']
                cand_str = item['cand_str']
                is_truth_capture_failed = item['is_truth_capture_failed']

                output_str_racket = tokenizer.decode(outputs_racket[j][input_lens_racket:], skip_special_tokens=True)
                
                racket_test_code = ""
                racket_match = re.search(r'```[Rr]acket\s*(.*?)\s*```', output_str_racket, re.DOTALL)
                if racket_match: racket_test_code = racket_match.group(1).strip()
                else: racket_test_code = output_str_racket.strip()

                racket_test_code = fix_racket_syntax_hallucinations(racket_test_code)

                try: candidates = json.loads(cand_str) if isinstance(cand_str, str) else cand_str
                except Exception: continue

                n = len(candidates)
                passed_count = 0
                passed_racket_lines = []
                passed_cands_list = []

                cleaned_base_test_code = clean_racket_test_code(racket_test_code)
                racket_test_ok = check_racket_test_syntax(cleaned_base_test_code)

                diagnostic_reasons = []

                for c_idx, cand in enumerate(candidates):
                    if not racket_test_ok:
                        passed_racket_lines.append(0.0)
                    else:
                        aligned_racket_test_code = align_test_with_candidate_racket(cand, racket_test_code, item['python_code'])
                        cleaned_racket_test_code = clean_racket_test_code(aligned_racket_test_code)
                        
                        status, num_passed_cases = verify_racket_candidate_robust(cand, cleaned_racket_test_code, diagnostic_reasons)
                        
                        if status == "Success" and num_passed_cases > 0:
                            passed_count += 1
                            passed_cands_list.append(cand)
                            racket_line_cov = calculate_racket_native_coverage(cand, cleaned_racket_test_code)
                            passed_racket_lines.append(racket_line_cov)
                        else:
                            passed_racket_lines.append(0.0)

                if is_truth_capture_failed:
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
                stats['sum_line_cov'] += python_line_cov
                stats['sum_branch_cov'] += python_branch_cov
                stats['sum_racket_line_cov'] += avg_racket_line_cov

                p1 = calculate_pass_at_k(n, passed_count, 1)
                p2 = calculate_pass_at_k(n, passed_count, 2) if n >= 2 else p1
                p5 = calculate_pass_at_k(n, passed_count, 5) if n >= 5 else p2

                stats['p1'] += p1; stats['p2'] += p2; stats['p5'] += p5

                # 1. 写入 CSV 大盘文件
                writer.writerow([
                    idx, f"{python_line_cov:.2f}%", f"{python_branch_cov:.2f}%", f"{avg_racket_line_cov:.2f}%",
                    "Passed" if passed_count > 0 else "Failed", f"{p1:.4f}", f"{p2:.4f}", f"{p5:.4f}", passed_count
                ])

                # 2. 增量保存完整 JSONL 数据流记录
                report_item = {
                    "idx": int(idx),
                    "status": "Passed" if passed_count > 0 else "Failed",
                    "global_racket_status": global_racket_status,
                    "metrics": {
                        "python_line_coverage": python_line_cov,
                        "python_branch_coverage": python_branch_cov,
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

                # 3. 归档 Passed 干净翻译的数据集代码池
                if passed_count > 0:
                    save_item = {
                        "idx": int(idx),
                        "python_code": item['python_code'],
                        "python_test_cases": json.loads(item['python_test_code']),
                        "racket_test_code": racket_test_code,
                        "passed_racket_candidates": passed_cands_list,
                        "line_coverage": python_line_cov,
                        "branch_coverage": python_branch_cov,
                        "racket_line_coverage": avg_racket_line_cov,
                        "pass_at_1": p1,
                        "pass_at_2": p2,
                        "pass_at_5": p5
                    }
                    f_save.write(json.dumps(save_item, ensure_ascii=False) + "\n")
                    
                f_jsonl.flush()
                f_save.flush()

            f.flush()
            current_count += len(batch_rows)
            
            valid_num = max(stats['valid'], 1)
            total_num = max(stats['total'], 1)
            pct = (current_count / total_records) * 100
            
            elapsed_time = time.time() - start_time
            it_s = current_count / elapsed_time if elapsed_time > 0 else 0.0
            remaining_seconds = (total_records - current_count) / it_s if it_s > 0 else 0
            
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_seconds))

            py_line_avg = stats['sum_line_cov'] / valid_num
            py_branch_avg = stats['sum_branch_cov'] / valid_num
            racket_line_avg = stats['sum_racket_line_cov'] / valid_num

            full_status_line = (
                f"进度:{pct:>5.1f}% ({current_count}/{total_records})[{elapsed_str}<{remaining_str},{it_s:.2f}it/s] "
                f"|总数:{stats['total']}|真值错:{stats['python_test_failed']}|Racket语法错:{stats['racket_test_syntax_errors']}|Racket无通过:{stats['racket_no_cand_passed']}|Racket成功:{stats['racket_success']} "
                f"|P@1:{(stats['p1']/total_num)*100:.2f}%|P@2:{(stats['p2']/total_num)*100:.2f}%|P@5:{(stats['p5']/total_num)*100:.2f}% "
                f"|Py行覆:{py_line_avg:.1f}%|Racket行覆:{racket_line_avg:.1f}%"
            )
            print(full_status_line)
            sys.stdout.flush()

            del prompts_racket, inputs_py, outputs_py, inputs_racket, outputs_racket
            torch.cuda.empty_cache()

        print("\n")

    del model
    torch.cuda.empty_cache()
    
    print("================ 📈 重构版自愈评测大盘总结 (Racket) 📈 ================")
    print(f"送入评测的黄金样本总量 (Total Records)     : {stats['total']}")
    print(f"Python 捕获完全失败数 (Capture Failed)     : {stats['python_test_failed']}")
    
    valid_total = stats['valid']
    grand_total = stats['total']
    print(f"完美参与语义指标统计的有效题目数           : {valid_total}")
    print(f"  ├─ Racket 测试代码语法错误件数 (Racket Test Syntax Error): {stats['racket_test_syntax_errors']}")
    print(f"  ├─ 候选代码全部未通过断言 (Racket No Cand Passed)    : {stats['racket_no_cand_passed']}")
    print(f"  ├─ 🚨 因 Racket 端真值错而强制失败的件数        : {stats['racket_truth_failed']}")
    print(f"  └─ 至少有1个候选代码测试通过的正常件数 (Racket Success): {stats['racket_success']}")
    
    if grand_total > 0:
        print(f"🌟 Racket 语言对齐平均 Pass@1: {stats['p1'] / grand_total * 100:.2f}%")
        print(f"🌟 Racket 语言对齐平均 Pass@2: {stats['p2'] / grand_total * 100:.2f}%")
        print(f"🌟 Racket 语言对齐平均 Pass@5: {stats['p5'] / grand_total * 100:.2f}%")
        print("-" * 50)
    if valid_total > 0:
        print(f"📐 Python 平均行覆盖率 (Line Coverage)      : {stats['sum_line_cov'] / valid_total:.2f}%")
        print(f"📐 Python 平均分支覆盖率 (Branch Coverage)  : {stats['sum_branch_cov'] / valid_total:.2f}%")
        print(f"📐 Racket 平均行覆盖率 (Line Coverage)       : {stats['sum_racket_line_cov'] / valid_total:.2f}%")
    print("==================================================================")
    print("\n🎉 [OK] 评测顺利结束！\n")

if __name__ == "__main__":
    INFERENCE_CSV_ORIGINAL = "/root/autodl-tmp/first_paper/python/translation_dataset/Racket_golden/Racket_inference_results_original.csv"
    LLM_MODEL_PATH = "/root/autodl-tmp/first_paper/LLM/Qwen2.5"
    CURRENT_LLM_TYPE = "qwen"
    OUTPUT_REPORT = "/root/autodl-tmp/first_paper/python/testcase_dataset/Racket_passk_report.csv"
    PASSED_TRANSLATIONS_JSONL = "/root/autodl-tmp/first_paper/python/testcase_dataset/Racket_pass/Racket_passed_clean.jsonl"
    
    run_live_pipeline(
        csv_path=INFERENCE_CSV_ORIGINAL,
        llm_path=LLM_MODEL_PATH,
        llm_type=CURRENT_LLM_TYPE,
        output_report_csv=OUTPUT_REPORT,
        passed_save_path=PASSED_TRANSLATIONS_JSONL
    )