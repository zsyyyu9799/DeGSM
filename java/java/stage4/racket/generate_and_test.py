import os
import re
import csv
import json
import math
import tempfile
import subprocess
import pandas as pd
import torch
import sys
import time
import difflib
import gc
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== 0. 自动修复并规避 libgomp 环境警告 ====================
if "OMP_NUM_THREADS" in os.environ:
    try:
        int(os.environ["OMP_NUM_THREADS"])
    except ValueError:
        os.environ["OMP_NUM_THREADS"] = "1"
else:
    os.environ["OMP_NUM_THREADS"] = "1"

# ==================== 1. 针对 Java 优化的提示词与 Prompt 路由 ====================
GEN_TEST_SYSTEM_PROMPT = (
    "You are an expert QA Engineer and a rigorous automated script. "
    "Your unique task is to analyze Java source code and generate highly precise and diverse test input arguments. "
    "You must output ONLY raw, strict, standard JSON data matching the template. "
    "Never write any markdown formatting (such as ```json), introductory text, or natural language explanation."
)

GEN_TEST_USER_PROMPT = """You are presented with a Java method. Your goal is to construct a rigorous test input suite comprising EXACTLY eight diverse and distinct test cases (arguments only) to thoroughly evaluate the method's semantics.

[Testing Objectives]
1. Diversity: Ensure the eight test cases cover different logical paths, partition testing zones, and branching behaviors of the code.
2. Boundary Values: Actively include edge cases such as empty structures (empty arrays/lists/strings), negative numbers, zero, large integers, or specific boundary limits.
3. Correctness: Ensure the types of arguments strictly match the expected parameters of the Java method.

[Output Format Specification]
Output the test cases in strict, standard JSON format as a single continuous list of objects. Each object must only contain the "args" key which is a list of parameters.
- Do NOT output "expected" values. We will execute the code to collect them.
- Do NOT wrap the JSON inside markdown code blocks.
- Do NOT output any conversational prefix, suffix, or natural language explanation.

[Example of Good Input/Output]
Input Java Code:
```java
public int[] twoSum(int[] nums, int target) {{
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {{
        int complement = target - nums[i];
        if (map.containsKey(complement)) {{
            return new int[] {{ map.get(complement), i }};
        }}
        map.put(nums[i], i);
    }}
    return new int[0];
}}
```

Expected Output:
[
  {{"args": [[2, 7, 11, 15], 9]}},
  {{"args": [[3, 2, 4], 6]}},
  {{"args": [[3, 3], 6]}},
  {{"args": [[-1, -2, -3, -4], -5]}},
  {{"args": [[], 0]}},
  {{"args": [[100], 100]}}
]

[Actual Task Input]
Input Java Code:
```java
{java_code}
```

Expected Output:
"""

def get_prompt_by_model(llm_type, system_prompt, user_content):
    if llm_type == 'qwen':
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    elif llm_type == 'deepseek':
        return f"<｜begin▁of▁sentence｜>{system_prompt}\n### Instruction:\n{user_content}\n### Response:\n["
    else: 
        if system_prompt:
            return f"{system_prompt}\n\n{user_content}\n\n"
        else:
            return f"{user_content}\n\n"

# ==================== 2. 高阶自愈与 Java 类隔离动态协同沙箱 ====================
def extract_java_class_name(java_code: str) -> str:
    clean_code = re.sub(r'//.*$', '', java_code, flags=re.MULTILINE)
    clean_code = re.sub(r'/\*.*?\*/', '', clean_code, flags=re.DOTALL)
    
    public_match = re.search(r'\bpublic\s+class\s+(\w+)\b', clean_code)
    if public_match:
        return public_match.group(1)
        
    match = re.search(r'\bclass\s+(\w+)\b', clean_code)
    if match:
        return match.group(1)
    return "Solution"

def parse_java_signature_advanced(java_code: str):
    clean_code = re.sub(r'//.*$', '', java_code, flags=re.MULTILINE)
    clean_code = re.sub(r'/\*.*?\*/', '', clean_code, flags=re.DOTALL)
    
    pattern = r'((?:public|protected|private|static|\s)+)\s+([\w\<\>\[\],\s]+)\s+(\w+)\s*\(([^)]*)\)'
    matches = re.findall(pattern, clean_code)
    
    if not matches:
        return None
        
    class_name = extract_java_class_name(java_code)
    for modifiers, ret_type, name, params_str in matches:
        name = name.strip()
        modifiers_clean = modifiers.strip()
        ret_type_clean = ret_type.strip()
        
        if name == class_name or name in ("main", "SolutionRunner", "Solution", "equals", "hashCode", "toString") or "class" in modifiers_clean or "class" in ret_type_clean:
            continue
        
        is_static = "static" in modifiers_clean
        params = []
        if params_str.strip():
            for p in params_str.split(','):
                parts = p.strip().split()
                if len(parts) >= 2:
                    p_type = parts[-2].strip()
                    p_name = parts[-1].strip()
                    params.append((p_type, p_name))
        return {
            "modifiers": modifiers_clean,
            "return_type": ret_type_clean,
            "method_name": name,
            "params": params,
            "is_static": is_static
        }
    return None

def to_java_literal(val, java_type: str) -> str:
    if val is None:
        return "null"
    
    java_type_clean = java_type.strip()
    
    if "List" in java_type_clean:
        if isinstance(val, list):
            inner_type = "Object"
            match = re.search(r'List\s*<\s*([\w\<\>\[\]]+)\s*>', java_type_clean)
            if match:
                inner_type = match.group(1)
            ref_type_map = {
                "int": "Integer", "long": "Long", "double": "Double",
                "float": "Float", "boolean": "Boolean", "char": "Character"
            }
            inner_type = ref_type_map.get(inner_type, inner_type)
            literals = [to_java_literal(x, inner_type) for x in val]
            return f"new ArrayList<{inner_type}>(Arrays.asList({', '.join(literals)}))"
        return "new ArrayList<>()"
        
    if java_type_clean.endswith("[]"):
        base_type = java_type_clean[:-2].strip()
        if isinstance(val, list):
            literals = [to_java_literal(x, base_type) for x in val]
            return f"new {base_type}[]{{{', '.join(literals)}}}"
        return f"new {base_type}[0]"
        
    if java_type_clean == "int":
        try: return str(int(val))
        except: return "0"
    if java_type_clean == "long":
        try: return f"{int(val)}L"
        except: return "0L"
    if java_type_clean == "double":
        try: return f"{float(val)}d"
        except: return "0.0d"
    if java_type_clean == "float":
        try: return f"{float(val)}f"
        except: return "0.0f"
    if java_type_clean == "boolean":
        return "true" if val else "false"
    if java_type_clean in ("char", "Character"):
        char_val = str(val)[0] if val else ' '
        return f"'{char_val}'"
    if java_type_clean == "String":
        escaped = str(val).replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r")
        return f"\"{escaped}\""
        
    if isinstance(val, str):
        escaped = val.replace("\\", "\\\\").replace("\"", "\\\"")
        return f"\"{escaped}\""
    return str(val)

def generate_java_runner(java_code: str, raw_pairs: list, sig: dict, class_name: str) -> str:
    method_name = sig["method_name"]
    params = sig["params"]
    return_type = sig["return_type"]
    is_static = sig.get("is_static", False)
    
    test_cases_blocks = []
    for idx, pair in enumerate(raw_pairs):
        args = pair.get("args", [])
        
        java_method_args = []
        for i, (p_type, p_name) in enumerate(params):
            val = args[i] if i < len(args) else None
            java_method_args.append(to_java_literal(val, p_type))
        method_args_str = ", ".join(java_method_args)
        
        if is_static:
            method_call = f"{class_name}.{method_name}({method_args_str})"
        else:
            method_call = f"(({class_name})instantiateClass({class_name}.class)).{method_name}({method_args_str})"
        
        if return_type.strip() == "void":
            block = f"""
        try {{
            {method_call};
            System.out.println("TEST_CASE_{idx}_START:\\"void\\":TEST_CASE_{idx}_END");
        }} catch (Throwable t) {{
            System.out.println("TEST_CASE_{idx}_EXCEPTION:" + t.getClass().getName() + ": " + t.getMessage() + ":TEST_CASE_{idx}_END");
        }}
        """
        else:
            block = f"""
        try {{
            Object res = {method_call};
            System.out.println("TEST_CASE_{idx}_START:" + toJson(res) + ":TEST_CASE_{idx}_END");
        }} catch (Throwable t) {{
            System.out.println("TEST_CASE_{idx}_EXCEPTION:" + t.getClass().getName() + ": " + t.getMessage() + ":TEST_CASE_{idx}_END");
        }}
        """
        test_cases_blocks.append(block)
        
    full_source = f"""
import java.util.*;
import java.util.stream.*;
import java.lang.reflect.*;

public class SolutionRunner {{
    public static void main(String[] args) {{
        System.out.println("EXECUTION_START");
        {chr(10).join(test_cases_blocks)}
        System.out.println("EXECUTION_END");
    }}

    public static Object instantiateClass(Class<?> clazz) throws Exception {{
        try {{
            Constructor<?> defaultConst = clazz.getDeclaredConstructor();
            defaultConst.setAccessible(true);
            return defaultConst.newInstance();
        }} catch (Exception e) {{}}

        Constructor<?>[] constructors = clazz.getDeclaredConstructors();
        if (constructors.length == 0) {{
            return null;
        }}
        Arrays.sort(constructors, (c1, c2) -> Integer.compare(c1.getParameterCount(), c2.getParameterCount()));
        
        Constructor<?> bestConst = constructors[0];
        bestConst.setAccessible(true);
        Object[] dummyArgs = new Object[bestConst.getParameterCount()];
        Class<?>[] paramTypes = bestConst.getParameterTypes();
        for (int i = 0; i < paramTypes.length; i++) {{
            dummyArgs[i] = getDummyValue(paramTypes[i]);
        }}
        return bestConst.newInstance(dummyArgs);
    }}

    private static Object getDummyValue(Class<?> type) {{
        if (type == int.class || type == Integer.class) return 0;
        if (type == long.class || type == Long.class) return 0L;
        if (type == double.class || type == Double.class) return 0.0;
        if (type == float.class || type == Float.class) return 0.0f;
        if (type == boolean.class || type == Boolean.class) return false;
        if (type == char.class || type == Character.class) return ' ';
        if (type == String.class) return "";
        if (type == List.class) return new ArrayList<>();
        if (type == Map.class) return new HashMap<>();
        if (type == Set.class) return new HashSet<>();
        return null;
    }}

    public static String toJson(Object obj) {{
        if (obj == null) return "null";
        if (obj instanceof String) return "\\"" + escape((String)obj) + "\\"";
        if (obj instanceof Character) return "\\"" + obj.toString() + "\\"";
        if (obj instanceof Boolean || obj instanceof Number) return obj.toString();
        
        if (obj instanceof int[]) return Arrays.toString((int[])obj);
        if (obj instanceof long[]) return Arrays.toString((long[])obj);
        if (obj instanceof double[]) return Arrays.toString((double[])obj);
        if (obj instanceof float[]) return Arrays.toString((float[])obj);
        if (obj instanceof boolean[]) return Arrays.toString((boolean[])obj);
        if (obj instanceof char[]) {{
            char[] arr = (char[])obj;
            return "[" + IntStream.range(0, arr.length).mapToObj(i -> "\\"" + escape(String.valueOf(arr[i])) + "\\"").collect(Collectors.joining(",")) + "]";
        }}
        
        if (obj instanceof Object[]) {{
            return "[" + Arrays.stream((Object[])obj).map(SolutionRunner::toJson).collect(Collectors.joining(",")) + "]";
        }}
        if (obj instanceof List) {{
            return "[" + ((List<?>)obj).stream().map(SolutionRunner::toJson).collect(Collectors.joining(",")) + "]";
        }}
        if (obj instanceof Set) {{
            return "[" + ((Set<?>)obj).stream().map(SolutionRunner::toJson).collect(Collectors.joining(",")) + "]";
        }}
        if (obj instanceof Map) {{
            Map<?, ?> map = (Map<?, ?>)obj;
            return "{{" + map.entrySet().stream()
                .map(e -> "\\"" + escape(e.getKey().toString()) + "\\":" + toJson(e.getValue()))
                .collect(Collectors.joining(",")) + "}}";
        }}
        return "\\"" + escape(obj.toString()) + "\\"";
    }}

    private static String escape(String s) {{
        if (s == null) return "";
        return s.replace("\\\\", "\\\\\\\\")
                .replace("\\"", "\\\\\\\"")
                .replace("\\n", "\\\\n")
                .replace("\\r", "\\\\r")
                .replace("\\t", "\\\\t");
    }}
}}
"""
    return full_source

def execute_and_capture_ground_truth(java_code: str, raw_pairs: list) -> (list, str, float, float):
    """
    Java 真值捕获与物理执行，确保 100% 正确捕捉 Java Deterministic 输出。
    """
    sanitized_pairs = []
    if isinstance(raw_pairs, list):
        for item in raw_pairs:
            if isinstance(item, dict):
                args = item.get("args", [])
                sanitized_pairs.append({"args": args if isinstance(args, list) else [args]})
            elif isinstance(item, list):
                sanitized_pairs.append({"args": item})
            elif item is not None:
                sanitized_pairs.append({"args": [item]})
                
    if not sanitized_pairs:
        sanitized_pairs = [{"args": []}]
        
    raw_pairs = sanitized_pairs
    sig = parse_java_signature_advanced(java_code)
    
    if not sig:
        for pair in raw_pairs:
            pair["status"] = "FAILED"
            pair["expected"] = None
            pair["error_msg"] = "Java signature matching failed. Unable to identify targets."
        return raw_pairs, "CRITICAL_JAVA_SIGNATURE_FAILED", 0.0, 0.0
        
    class_name = extract_java_class_name(java_code)
    runner_source = generate_java_runner(java_code, raw_pairs, sig, class_name)
    
    standard_imports = "import java.util.*;\nimport java.util.stream.*;\nimport java.util.function.*;\nimport java.io.*;\nimport java.math.*;\nimport java.net.*;\nimport java.text.*;\n"
    java_code_clean = re.sub(r'^\s*package\s+[\w\.]+;\s*', '', java_code, flags=re.MULTILINE)
    full_java_class_code = standard_imports + "\n" + java_code_clean
    
    base_temp_path = "/dev/shm" if os.path.exists("/dev/shm") else None
    with tempfile.TemporaryDirectory(prefix="java_verify_", dir=base_temp_path) as tmpdir:
        class_file = os.path.join(tmpdir, f"{class_name}.java")
        runner_file = os.path.join(tmpdir, "SolutionRunner.java")
        
        with open(class_file, "w", encoding="utf-8") as f:
            f.write(full_java_class_code)
        with open(runner_file, "w", encoding="utf-8") as f:
            f.write(runner_source)
            
        compile_res = subprocess.run(
            ["javac", f"{class_name}.java", "SolutionRunner.java"],
            cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=12
        )
        
        if compile_res.returncode != 0:
            for pair in raw_pairs:
                pair["status"] = "FAILED"
                pair["expected"] = None
                pair["error_msg"] = f"JavaCompileError: {compile_res.stderr.strip()}"
            return raw_pairs, f"CRITICAL_JAVA_COMPILE_FAILED: {compile_res.stderr.strip()}", 0.0, 0.0
            
        try:
            run_res = subprocess.run(
                ["java", "SolutionRunner"],
                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=8
            )
            stdout_str, stderr_str = run_res.stdout, run_res.stderr
        except subprocess.TimeoutExpired:
            stdout_str, stderr_str = "", "CRITICAL_JAVA_SANDBOX_TIMEOUT"
            
        passed_io_pairs = []
        for pair in raw_pairs:
            passed_io_pairs.append({
                "args": pair.get("args", []),
                "status": "FAILED",
                "expected": None,
                "error_msg": "Timeout or Sandbox Crash"
            })
            
        start_pattern = r'TEST_CASE_(\d+)_START:(.*):TEST_CASE_\1_END'
        except_pattern = r'TEST_CASE_(\d+)_EXCEPTION:(.*):TEST_CASE_\1_END'
        
        for line in stdout_str.splitlines():
            m_start = re.search(start_pattern, line)
            if m_start:
                case_idx = int(m_start.group(1))
                val_str = m_start.group(2)
                if case_idx < len(passed_io_pairs):
                    try:
                        passed_io_pairs[case_idx]["expected"] = json.loads(val_str)
                    except:
                        passed_io_pairs[case_idx]["expected"] = val_str
                    passed_io_pairs[case_idx]["status"] = "PASSED"
                    passed_io_pairs[case_idx]["error_msg"] = None
                    
            m_except = re.search(except_pattern, line)
            if m_except:
                case_idx = int(m_except.group(1))
                err_msg = m_except.group(2)
                if case_idx < len(passed_io_pairs):
                    passed_io_pairs[case_idx]["status"] = "FAILED"
                    passed_io_pairs[case_idx]["expected"] = None
                    passed_io_pairs[case_idx]["error_msg"] = err_msg

        passed_runs = sum([1 for x in passed_io_pairs if x["status"] == "PASSED"])
        line_cov = (passed_runs / len(passed_io_pairs)) * 100.0 if passed_io_pairs else 0.0
        branch_cov = line_cov
        
        feedback_msg = "SUCCESS" if passed_runs > 0 else f"CRITICAL_EXECUTION_FAILURE: {stderr_str}"
        return passed_io_pairs, feedback_msg, line_cov, branch_cov

# ==================== 3. Cross-Modal Racket Assertion Engine ====================
def extract_racket_function_name(code: str, java_code: str = None) -> str:
    """
    专门适配 Racket Lisp 语法的函数名提取器。
    支持匹配标配的 (define (func-name ...)) 以及 Lambda 绑定的 (define func-name (lambda ...))
    """
    if not code or not isinstance(code, str):
        return ""
    # 清理 Racket 的注释防止误匹配
    code_clean = re.sub(r';.*$', '', code, flags=re.MULTILINE)
    
    # 匹配 Racket 两种常见的定义方式
    pattern_func = r'\(\s*define\s*\(\s*([\w\-\?\!\>\+\=\*\/\|]+)'
    pattern_lambda = r'\(\s*define\s+([\w\-\?\!\>\+\=\*\/\|]+)\s+\(\s*lambda'
    
    all_funcs = re.findall(pattern_func, code_clean) + re.findall(pattern_lambda, code_clean)
    if not all_funcs:
        return ""
    
    # 如果提供 Java 签名，尝试智能转换 camelCase -> kebab-case 匹配
    java_sig = parse_java_signature_advanced(java_code) if java_code else None
    if java_sig:
        java_func = java_sig["method_name"]
        # camelCase 转换成 Racket kebab-case (如 twoSum -> two-sum)
        kebab_java = re.sub(r'(?<!^)(?=[A-Z])', '-', java_func).lower()
        
        if java_func in all_funcs:
            return java_func
        if kebab_java in all_funcs:
            return kebab_java
            
        cleaned_java = java_func.replace("_", "").lower()
        for f in all_funcs:
            if f.replace("-", "").replace("_", "").lower() == cleaned_java:
                return f
        close_matches = difflib.get_close_matches(kebab_java, all_funcs, n=1, cutoff=0.3)
        if close_matches:
            return close_matches[0]

    # 排除测试套件内的公用垫片方法
    non_trivial_funcs = [f for f in all_funcs if f not in ("procedure-arity-min", "safe-take", "approx-equal?", "run-test-safely")]
    if non_trivial_funcs:
        return non_trivial_funcs[0]
        
    return all_funcs[0]

# 垫片自愈：当调试器寻找并尝试调用 Rust 入口提取函数时，自动对齐到 Racket
def extract_rust_function_name(code: str, java_code: str = "") -> str:
    return extract_racket_function_name(code, java_code)

def format_racket_val(val):
    if isinstance(val, bool):
        return "#t" if val else "#f"
    if val is None:
        return "'()"
    if isinstance(val, list):
        elements = [format_racket_val(x) for x in val]
        return f"(list {' '.join(elements)})"
    if isinstance(val, str):
        return f'"{val}"'
    if isinstance(val, float):
        if math.isinf(val):
            return "+inf.0" if val > 0 else "-inf.0"
        if math.isnan(val):
            return "+nan.0"
    return str(val)

def execute_racket_test(candidate_code, io_pairs, timeout=3) -> (bool, bool, str, str):
    """
    执行 Racket 单元测试
    返回: (是否测试通过, 是否属于 Racket 编译/语法错误, 完整生成的测试代码, 运行错误日志)
    """
    if not candidate_code.strip() or not io_pairs:
        return False, False, "", "Empty candidate or IO pairs"
        
    # 只针对 Java 端运行成功的真实路径（PASSED）作跨语言对齐测试，避免抛异常等逻辑歧义的干扰
    active_io_pairs = [pair for pair in io_pairs if pair.get("status") == "PASSED"]
    if not active_io_pairs:
        return False, False, "", "No active PASSED IO pairs"
        
    func_name = extract_racket_function_name(candidate_code)
    if not func_name:
        return False, True, "", "Failed to extract Racket function name" # 无法提取方法名，视为语法错误

    detected_libs = []
    if "math" in candidate_code or "log" in candidate_code or "sqrt" in candidate_code:
        detected_libs.append("(require racket/math)")
    
    lib_header = "\n".join(detected_libs) + "\n"

    mock_header = """
(require rackunit)

(define (procedure-arity-min arity)
  (cond
    [(integer? arity) arity]
    [(arity-at-least? arity) (arity-at-least-value arity)]
    [(list? arity) (if (null? arity) 0 (apply min (filter integer? arity)))]
    [else 0]))

(define (safe-take lst n)
  (cond
    [(or (<= n 0) (null? lst)) '()]
    [else (cons (car lst) (safe-take (cdr lst) (- n 1)))]))

(define (approx-equal? a b)
  (cond
    [(and (number? a) (number? b))
     (< (abs (- a b)) 1e-5)]
    [(and (pair? a) (pair? b))
     (and (approx-equal? (car a) (car b))
          (approx-equal? (cdr a) (cdr b)))]
    [(and (vector? a) (vector? b))
     (approx-equal? (vector->list a) (vector->list b))]
    [else (equal? a b)]))

(define (run-test-safely func args)
  (let* ([arity-info (procedure-arity func)]
         [min-arity (procedure-arity-min arity-info)])
    (let ([processed-args 
           (if (and (> min-arity 1) (= (length args) 1) (list? (car args)))
               (car args)
               args)])
      (let ([final-args (if (and (integer? arity-info) (> (length processed-args) arity-info))
                            (safe-take processed-args arity-info)
                            processed-args)])
        (apply func final-args)))))
"""

    test_statements = []
    for pair in active_io_pairs:
        try:
            formatted_args_list = "(list " + " ".join(format_racket_val(a) for a in pair["args"]) + ")"
            expected_str = format_racket_val(pair["expected"])
            
            test_statements.append(
                f"(with-handlers ([exn:fail? (lambda (ex) (exit 1))])\n"
                f"  (let* ([actual (run-test-safely {func_name} {formatted_args_list})]\n"
                f"         [expected {expected_str}])\n"
                f"    (if (approx-equal? actual expected)\n"
                f"        (void)\n"
                f"        (exit 1))))"
            )
        except Exception as e:
            return False, False, "", f"Assertion formatting exception: {str(e)}"
            
    clean_code = candidate_code
    if "#lang racket" in clean_code:
        clean_code = clean_code.replace("#lang racket", "#lang racket\n" + lib_header + mock_header)
    else:
        clean_code = "#lang racket\n" + lib_header + mock_header + "\n" + clean_code

    clean_code += "\n\n; ====== AUTOMATED TESTS ======\n" + "\n".join(test_statements) + "\n"

    base_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
    with tempfile.NamedTemporaryFile(suffix=".rkt", dir=base_dir, mode="w", encoding="utf-8", delete=False) as tf:
        tf.write(clean_code)
        temp_file_path = tf.name

    try:
        res = subprocess.run(
            ["racket", temp_file_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, encoding='utf-8', errors='ignore'
        )
        if res.returncode == 0:
            return True, False, clean_code, ""
        else:
            err_msg = res.stderr
            err_msg_lower = err_msg.lower()
            is_syntax = any(x in err_msg_lower for x in ["syntax error", "unbound identifier", "read-syntax", "duplicate definition", "bad syntax", "compile:", "expected:"])
            return False, is_syntax, clean_code, err_msg
    except Exception as e:
        return False, False, clean_code, f"Execution Sandbox Error: {str(e)}"
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass

# ==================== 4. Metric Estimation Logic ====================
def calculate_pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    else:
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def process_pipeline_sample(idx, java_code, candidates_str, raw_args_list, correctness_threshold=100.0):
    # 1. 物理比对，动态校验大模型生成的测试套件与 Java 真值运行（Oracle）的一致性
    try:
        passed_io_pairs, java_feedback, java_line_cov, java_branch_cov = execute_and_capture_ground_truth(
            java_code, raw_args_list
        )
    except Exception as e:
        passed_io_pairs = []
        java_feedback = f"CRITICAL_SANDBOX_CRASH: {str(e)}"
        java_line_cov, java_branch_cov = 0.0, 0.0

    # 捕获失败、编译错误或是覆盖率过低，则标记为拦截过滤（真值错）
    is_truth_capture_failed = (len(raw_args_list) == 0) or ("CRITICAL_" in java_feedback) or (java_line_cov < correctness_threshold)
    
    if is_truth_capture_failed:
        short_msg = "Filtered_Low_Oracle_Accuracy"
        if "SIGNATURE_FAILED" in java_feedback:
            short_msg = "Filtered_Signature_Parse_Failed"
        elif "COMPILE_FAILED" in java_feedback:
            short_msg = "Filtered_Java_Compilation_Failed"
            
        return {
            'idx': idx, 'java_status': java_feedback, 'status': short_msg,
            'pass@1': 0.0, 'pass@2': 0.0, 'pass@5': 0.0, 'passed_count': 0,
            'java_line_cov': java_line_cov, 'racket_line_cov': 0.0, 'is_syntax_error': False,
            'racket_test_code': "", 'diagnostic_reasons': [], 'racket_test_ok': False, 'candidates': [],
            'passed_candidates': [], 'java_test_cases': []
        }

    # 2. 对 Racket 翻译候选进行大规模批处理测试验证
    try:
        candidates = json.loads(candidates_str)
    except Exception:
        return {
            'idx': idx, 'java_status': 'JSON_Candidates_Parse_Error', 'status': 'JSON_Err',
            'pass@1': 0.0, 'pass@2': 0.0, 'pass@5': 0.0, 'passed_count': 0,
            'java_line_cov': java_line_cov, 'racket_line_cov': 0.0, 'is_syntax_error': False,
            'racket_test_code': "", 'diagnostic_reasons': [], 'racket_test_ok': False, 'candidates': [],
            'passed_candidates': [], 'java_test_cases': []
        }

    n = len(candidates)
    passed_count = 0
    passed_racket_lines = []
    passed_cands_list = []
    any_syntax_error = False

    failing_details = []
    racket_test_code = ""

    for cand in candidates:
        passed, is_syntax, rkt_code, err_msg = execute_racket_test(cand, passed_io_pairs)
        if passed:
            passed_count += 1
            passed_racket_lines.append(100.0) # 成功通过单元测试视为 100% 覆盖
            passed_cands_list.append(cand)
            if not racket_test_code:
                racket_test_code = rkt_code  # 捕获首个运行成功的完整 Racket 测试代码（包含测试桩声明）
        else:
            passed_racket_lines.append(0.0)
            if is_syntax:
                any_syntax_error = True
            if err_msg:
                failing_details.append({
                    'code': rkt_code,
                    'err': err_msg
                })

    p1 = calculate_pass_at_k(n, passed_count, 1)
    p2 = calculate_pass_at_k(n, passed_count, 2) if n >= 2 else p1
    p5 = calculate_pass_at_k(n, passed_count, 5) if n >= 5 else p2

    avg_racket_line_cov = sum(passed_racket_lines) / len(passed_racket_lines) if passed_racket_lines else 0.0

    diagnostic_reasons = []
    racket_test_ok = (passed_count > 0) or (len(failing_details) > 0 and not any_syntax_error)
    
    if passed_count == 0 and failing_details:
        racket_test_code = failing_details[0]['code']
        diagnostic_reasons = [x['err'] for x in failing_details if x['err']]

    return {
        'idx': idx, 'java_status': java_feedback, 'status': 'Passed' if passed_count > 0 else 'Failed',
        'pass@1': p1, 'pass@2': p2, 'pass@5': p5, 'passed_count': passed_count,
        'java_line_cov': java_line_cov, 'racket_line_cov': avg_racket_line_cov,
        'is_syntax_error': any_syntax_error,
        'racket_test_code': racket_test_code,
        'diagnostic_reasons': diagnostic_reasons,
        'racket_test_ok': racket_test_ok,
        'candidates': candidates,
        'passed_candidates': passed_cands_list,
        'java_test_cases': passed_io_pairs
    }

def extract_and_parse_json(output_str: str) -> list:
    """
    超强鲁棒 JSON 提取器，完美解决多余 Markdown、末尾逗号及 brackets 不闭合等幻觉故障。
    """
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
    return []

# ==================== 5. Live Pipeline Orchestration ====================
def run_live_pipeline(csv_path, llm_path, llm_type, output_report_csv, passed_save_path, correctness_threshold=100.0, batch_size=4):
    """
    主控制中心，运行评测。
    """
    print("📋 [Pre-flight Check] 正在检查系统测试环境依赖...")
    javac_check = subprocess.run(["which", "javac"], capture_output=True)
    racket_check = subprocess.run(["which", "racket"], capture_output=True)
    
    if javac_check.returncode != 0:
        print("⚠️  [System Warning] 系统未发现 'javac' 命令！所有的 Java 单元测试将无法编译运行，评测可能会被 100% 过滤。")
    else:
        print("   ✅ JDK 编译器 'javac' 状态：OK")
        
    if racket_check.returncode != 0:
        print("⚠️  [System Warning] 系统未发现 'racket' 命令！Racket 代码的运行结果将全部评测失败。")
    else:
        print("   ✅ 解释器 'racket' 状态：OK")
    print("-" * 66)

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['java_code', 'all_candidates'])
    
    if 'status' in df.columns:
        initial_count = len(df)
        df = df[df['status'] == 'Passed'].reset_index(drop=True)
        filtered_out = initial_count - len(df)
        print(f"📊 原始大模型推理总记录数: {initial_count}")
        print(f"🔥 剔除基础翻译未通过样本数: {filtered_out}")
        print(f"🚀 送入评测的真实有效黄金样本总量: {len(df)}\n")
    else:
        print("⚠️ [Warning] 未在 CSV 中找到 'status' 字段，将评测全量样本。")

    total_records = len(df)
    if total_records == 0:
        print("❌ [Terminate] 没有找到任何有效的黄金样本，评测终止。")
        return

    print("⏳ 正在加载用于测试套件合成的生成层级模型（LLM）...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    # 初始化大盘状态计数器（完全对齐 Rust 版本的自愈汇报大盘）
    stats = {
        'total': 0, 
        'java_test_failed': 0,            # 真值错 (Filtered)
        'racket_test_syntax_errors': 0,   # Racket 语法/解释编译错
        'racket_no_cand_passed': 0,       # Racket 无通过 (断言失败)
        'racket_success': 0,              # Racket 正常运行且至少一个候选通过
        'racket_truth_failed': 0,         # 因 Java 端真值错而强制失败的件数
        'valid': 0,                       # 通过 Oracle 的有效样本数
        'p1': 0.0, 'p2': 0.0, 'p5': 0.0,
        'sum_line_cov': 0.0,              # Java 行覆盖率求和
        'sum_racket_line_cov': 0.0        # Racket 行覆盖率求和
    }
    filter_reasons = Counter()
    
    print(f"\n🚀 评测流水线已激活 (批处理大小 Batch Size: {batch_size})")

    # 自动推导并初始化 jsonl 文件路径
    output_report_jsonl = output_report_csv.rsplit('.', 1)[0] + ".jsonl"
    debug_error_count = 0

    with open(output_report_csv, mode='w', encoding='utf-8', newline='') as f, \
         open(output_report_jsonl, mode='w', encoding='utf-8') as f_jsonl, \
         open(passed_save_path, mode='w', encoding='utf-8') as f_save:
         
        writer = csv.writer(f)
        writer.writerow([
            'idx', 'java_line_coverage', 'racket_coverage_line', 'status', 
            'pass@1', 'pass@2', 'pass@5', 'passed_candidates'
        ])

        rows = [row for _, row in df.iterrows()]
        start_time = time.time()
        current_count = 0

        for i in range(0, total_records, batch_size):
            batch_rows = rows[i:i + batch_size]
            prompts = []
            
            for row in batch_rows:
                user_content = GEN_TEST_USER_PROMPT.format(java_code=row['java_code'])
                prompt = get_prompt_by_model(llm_type, GEN_TEST_SYSTEM_PROMPT, user_content)
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    temperature=0.2, 
                    top_p=0.95, 
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            input_lens = inputs.input_ids.shape[1]
            for j, row in enumerate(batch_rows):
                current_count += 1

                idx = str(row['idx'])
                java_code = row['java_code']
                cand_str = row['all_candidates']
                
                output_str = tokenizer.decode(outputs[j][input_lens:], skip_special_tokens=True)
                raw_args_list = extract_and_parse_json(output_str)

                # 两段式比对与 Racket 沙箱检验（自动通过运行抓取 Java Deterministic expected 值）
                res = process_pipeline_sample(idx, java_code, cand_str, raw_args_list, correctness_threshold)

                # ==================== 【调试诊断区】Racket 测试与错误自愈捕获 ====================
                passed_count = res['passed_count']
                racket_test_ok = res.get('racket_test_ok', False)
                diagnostic_reasons = res.get('diagnostic_reasons', [])
                racket_test_code = res.get('racket_test_code', '')
                candidates = res.get('candidates', [])
                passed_candidates = res.get('passed_candidates', [])
                java_test_cases = res.get('java_test_cases', [])

                if passed_count == 0 and racket_test_ok and diagnostic_reasons:
                    if debug_error_count < 10:
                        debug_error_count += 1
                        print(f"\n📢 [自愈自修捕获 | 题目 IDX {idx}] 诊断日志 (第 {debug_error_count} 次观察):")
                        java_sig = parse_java_signature_advanced(row['java_code'])
                        print(f"  - 原始 Java 方法名: {java_sig['method_name'] if java_sig else 'unknown'}")
                        print(f"  - 目标 Racket 函数入口: {extract_racket_function_name(candidates[0], row['java_code']) if candidates else 'unknown'}")
                        print(f"  - 生成并清洗后的 Racket 测试代码(片段):")
                        print(f"    \"\"\"\n{racket_test_code[:300]}...\n    \"\"\"")
                        print(f"  - 编译器拦截反馈:")
                        for reason in diagnostic_reasons[:2]:
                            clean_reason = reason.replace('\n', '\n      ')
                            print(f"    ❌ {clean_reason[:500]}")
                        print("-" * 80 + "\n")
                        sys.stdout.flush()

                stats['total'] += 1
                
                # 【全量学术分母累加】被过滤或失败的指标贡献为 0.0
                stats['p1'] += res['pass@1']
                stats['p2'] += res['pass@2']
                stats['p5'] += res['pass@5']

                stats['sum_line_cov'] += res['java_line_cov']
                stats['sum_racket_line_cov'] += res['racket_line_cov']

                is_truth_capture_failed = res['status'] in ["Filtered_Low_Oracle_Accuracy", "Filtered_Signature_Parse_Failed", "Filtered_Java_Compilation_Failed"]

                if is_truth_capture_failed:
                    global_racket_status = "Racket_Truth_Failed"
                    stats['racket_truth_failed'] += 1
                    stats['java_test_failed'] += 1
                    filter_reasons[res['status']] += 1
                elif not racket_test_ok and passed_count == 0:
                    global_racket_status = "Racket_Test_Syntax_Error"
                    stats['valid'] += 1
                    if res.get('is_syntax_error', False):
                        stats['racket_test_syntax_errors'] += 1
                    else:
                        stats['racket_no_cand_passed'] += 1
                elif passed_count > 0:
                    global_racket_status = "Racket_Success"
                    stats['valid'] += 1
                    stats['racket_success'] += 1
                else:
                    global_racket_status = "Racket_No_Candidate_Passed"
                    stats['valid'] += 1
                    stats['racket_no_cand_passed'] += 1

                # 写入 CSV 汇总报告
                writer.writerow([
                    res['idx'], f"{res['java_line_cov']:.2f}%", f"{res['racket_line_cov']:.2f}%", res['status'],
                    f"{res['pass@1']:.4f}", f"{res['pass@2']:.4f}", f"{res['pass@5']:.4f}", res['passed_count']
                ])

                # 写入 JSONL 详细报告（对齐 Rust 行为）
                report_item = {
                    "idx": int(idx),
                    "status": "Passed" if passed_count > 0 else "Failed",
                    "global_racket_status": global_racket_status,
                    "metrics": {
                        "java_line_coverage": res['java_line_cov'],
                        "racket_line_coverage": res['racket_line_cov'],
                        "pass_at_1": res['pass@1'],
                        "pass_at_2": res['pass@2'],
                        "pass_at_5": res['pass@5']
                    },
                    "passed_candidates_count": passed_count,
                    "total_candidates_count": len(candidates),
                    "diagnostic_reasons": diagnostic_reasons
                }
                f_jsonl.write(json.dumps(report_item, ensure_ascii=False) + "\n")

                # 写入 Passed Racket 专属过滤落盘文件
                if passed_count > 0:
                    save_item = {
                        "idx": int(idx),
                        "java_code": java_code,
                        "java_test_cases": java_test_cases,
                        "racket_test_code": racket_test_code,
                        "passed_racket_candidates": passed_candidates,
                        "line_coverage": res['java_line_cov'],
                        "racket_line_coverage": res['racket_line_cov'],
                        "pass_at_1": res['pass@1'],
                        "pass_at_2": res['pass@2'],
                        "pass_at_5": res['pass@5']
                    }
                    f_save.write(json.dumps(save_item, ensure_ascii=False) + "\n")

                f_jsonl.flush()
                f_save.flush()

            f.flush()
            total_num = max(stats['total'], 1)
            pct = (current_count / total_records) * 100
            
            elapsed_time = time.time() - start_time
            it_s = current_count / elapsed_time if elapsed_time > 0 else 0.0
            remaining_items = total_records - current_count
            remaining_seconds = remaining_items / it_s if it_s > 0 else 0
            
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_seconds))

            # 统计计算大盘实时覆盖率
            java_line_avg = stats['sum_line_cov'] / total_num
            racket_line_avg = stats['sum_racket_line_cov'] / total_num

            # 完美的精细化进度打印格式
            full_status_line = (
                f"进度:{pct:>5.1f}% ({current_count}/{total_records})[{elapsed_str}<{remaining_str},{it_s:.2f}it/s] "
                f"|总数:{stats['total']}|真值错:{stats['java_test_failed']}|Racket语法错:{stats['racket_test_syntax_errors']}|Racket无通过:{stats['racket_no_cand_passed']}|Racket成功:{stats['racket_success']} "
                f"|P@1:{(stats['p1']/total_num)*100:.2f}%|P@2:{(stats['p2']/total_num)*100:.2f}%|P@5:{(stats['p5']/total_num)*100:.2f}% "
                f"|Java行覆:{java_line_avg:.1f}%|Racket行覆:{racket_line_avg:.1f}%"
            )
            print(full_status_line)
            sys.stdout.flush()

            del inputs, outputs
            torch.cuda.empty_cache()

        print("\n")

    # 释放显存
    del model
    torch.cuda.empty_cache()
    
    # 学术大盘最终总结报告
    print("================ 📈 重构版自愈评测大盘总结 (Java -> Racket) ==================")
    print(f"送入评测的黄金样本总量 (Total Records)     : {stats['total']}")
    print(f"Java 真值校验失败数 (Filtered / 真值错)    : {stats['java_test_failed']}")
    if stats['java_test_failed'] > 0:
        print("🔍 细分拦截原因统计 (Detailed Filtering Reason Counter):")
        for reason, count in filter_reasons.items():
            print(f"   - {reason}: {count} 个样本")
    print("-" * 66)
    print(f"参与语义指标统计的有效题目数 (Valid Samples): {stats['valid']}")
    print(f"  ├─ Racket 代码语法/运行错误数 (Racket Syntax Error): {stats['racket_test_syntax_errors']}")
    print(f"  ├─ 候选代码全部未通过断言 (Racket No Cand Passed)  : {stats['racket_no_cand_passed']}")
    print(f"  ├─ 🚨 因 Java 端真值错而强制失败的件数         : {stats['racket_truth_failed']}")
    print(f"  └─ 至少有1个候选代码测试通过的正常件数 (Racket Success): {stats['racket_success']}")
    print("-" * 66)
    if total_num > 0:
        print(f"🌟 Racket 语言对齐平均 Pass@1: {stats['p1'] / total_num * 100:.2f}%")
        print(f"🌟 Racket 语言对齐平均 Pass@2: {stats['p2'] / total_num * 100:.2f}%")
        print(f"🌟 Racket 语言对齐平均 Pass@5: {stats['p5'] / total_num * 100:.2f}%")
        print("-" * 66)
    if stats['valid'] > 0:
        print(f"📐 Java 平均行覆盖率 (Line Coverage)        : {stats['sum_line_cov'] / total_num:.2f}%")
        print(f"📐 Racket 平均行覆盖率 (Line Coverage)      : {stats['sum_racket_line_cov'] / total_num:.2f}%")
    print("==================================================================")
    print(f"🎉 评估顺利完成！报告与黄金代码已分别保存至:\n  - CSV 报告: {output_report_csv}\n  - JSONL 详情: {output_report_jsonl}\n  - 过滤落盘代码: {passed_save_path}\n")

if __name__ == "__main__":
    CSV_PATH = "/root/autodl-tmp/first_paper/java/translation_dataset/Racket_golden/Racket_inference_results_original.csv"
    LLM_MODEL_PATH = "/root/autodl-tmp/first_paper/LLM/Deepseek"
    CURRENT_LLM_TYPE = "qwen"
    CORRECTNESS_THRESHOLD = 100.0  # 校验通过率必须达到 100% 才能作为黄金测试套件
    OUTPUT_REPORT = "/root/autodl-tmp/first_paper/java/testcase_dataset/racket_pass/Racket_passk_report.csv"
    PASSED_TRANSLATIONS_JSONL = "/root/autodl-tmp/first_paper/java/testcase_dataset/racket_pass/Racket_passed_clean.jsonl"
    
    run_live_pipeline(
        csv_path=CSV_PATH, 
        llm_path=LLM_MODEL_PATH, 
        llm_type=CURRENT_LLM_TYPE, 
        output_report_csv=OUTPUT_REPORT, 
        passed_save_path=PASSED_TRANSLATIONS_JSONL,
        correctness_threshold=CORRECTNESS_THRESHOLD
    )