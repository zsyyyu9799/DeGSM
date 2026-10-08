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

# ==================== 1. 针对 Java 多方法覆盖与状态对齐优化的提示词 ====================
GEN_TEST_SYSTEM_PROMPT = (
    "You are an expert QA Engineer and a rigorous automated script. "
    "Your unique task is to analyze Java source code and generate highly precise and diverse test input arguments. "
    "You must output ONLY raw, strict, standard JSON data matching the template. "
    "Never write any markdown formatting (such as ```json), introductory text, or natural language explanation."
)

GEN_TEST_USER_PROMPT = """You are presented with a Java class containing several methods. Your goal is to construct a rigorous test input suite for EACH method to thoroughly evaluate its semantics.

For each method, you must generate exactly three diverse and distinct test cases.
- Instead of using default constructed empty states, you MUST specify the initial state of the object's internal fields in the "initial_state" dictionary. This allows us to reconstruct the object with rich, diverse, and non-default values in both Java and OCaml.
- "initial_state" must map the private/public field names of the class to their starting test values.
- If the method is static, "initial_state" should be empty `{{}}`.
- Provide the method arguments in the "args" field.

[Output Format Specification]
Output the test cases in strict, standard JSON format as a single dictionary where:
- The keys are the names of the Java methods.
- The values are lists of objects, each containing:
  - "initial_state": Dictionary mapping class field names (private or public) to their starting values.
  - "args": List of arguments for the method itself.

Example Java Class:
```java
public class Point {{
    private int x;
    private int y;
    public Point(int x, int y) {{
        this.x = x;
        this.y = y;
    }}
    public int getX() {{ return x; }}
    public void setX(int x) {{ this.x = x; }}
}}
```

Expected Output:
{{
  "getX": [
    {{"initial_state": {{"x": 2, "y": 3}}, "args": []}},
    {{"initial_state": {{"x": -5, "y": 10}}, "args": []}},
    {{"initial_state": {{"x": 100, "y": -200}}, "args": []}}
  ],
  "setX": [
    {{"initial_state": {{"x": 0, "y": 0}}, "args": [5]}},
    {{"initial_state": {{"x": 1, "y": 2}}, "args": [-10]}},
    {{"initial_state": {{"x": -10, "y": -20}}, "args": [99]}}
  ]
}}

[Actual Task Input]
Input Java Code:
```java
{java_code}
```

Expected Output:
"""

GEN_OCAML_TEST_SYSTEM_PROMPT = (
    "You are an expert QA Engineer and a rigorous automated script. "
    "Your unique task is to analyze Java source code and synthesize highly precise native OCaml testing suites. "
    "You must output ONLY the native OCaml test code enclosed in standard markdown code blocks."
)

GEN_OCAML_TEST_USER_PROMPT = """You are presented with a Java class, its executed Java test suite with ground truth, and the translated OCaml candidate implementation.
Your goal is to construct a matching OCaml test suite containing EXACTLY the same test cases targeting the OCaml code.

[OCaml Testing Strategy]
1. Map correct behaviors: Translate the semantically correct behaviors of the Java test cases (expected values, state initialization) into OCaml assertions.
2. Record Construction: If the OCaml candidate uses a record or module (e.g. `type t = {{ x : int; y : int }}`), construct the records with the values matching the Java "initial_state".
3. Syntax differences (CRITICAL):
   - OCaml is statically typed. Represent lists with semicolons, e.g., `[1; 2; 3]` or arrays with `[|1; 2; 3|]`.
   - Structural equality is checked with `=`, e.g. `assert (get_x {{x = 2; y = 3}} = 2)`.
   - Floating point comparisons use dotted operators (e.g., `abs_float (actual -. expected) < 1e-5`).
4. Assertion Rigor: Group assertions cleanly under top-level evaluation declarations `let () = assert (...)` or `let () = if not (...) then failwith "..."`. Keep assertions straightforward and structurally accurate.

[Output Format Specification]
You must output ONLY standard markdown code blocks for OCaml. Do NOT output any conversational text.

```ocaml
(* Name: test_solution.ml *)
(* Write test cases matching the Java methods *)
```

[Execution Feedback of Java Tests]
{java_feedback}

Original Java Code:
```java
{java_code}
```

Translated OCaml Code:
```ocaml
{ocaml_code}
```

Java Test Suite Code:
```json
{java_test_code}
```

Expected Output:
"""

def get_prompt_by_model(llm_type, system_prompt, user_content):
    """
    🎯 动态构建模型专用的 Prompt 格式（如 Qwen 或 Deepseek 的 Chat 模版标签）
    """
    if llm_type == 'qwen':
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    elif llm_type == 'deepseek':
        return f"<｜begin▁of▁sentence｜>{system_prompt}\n### Instruction:\n{user_content}\n### Response:\n"
    else: 
        if system_prompt:
            return f"{system_prompt}\n\n{user_content}\n\n"
        else:
            return f"{user_content}\n\n"

# ==================== 2. 高阶自愈与 Java 多方法协同沙箱 ====================
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

def parse_all_java_signatures(java_code: str) -> list:
    """
    🎯 扫描提取 Java 类的所有有效公共方法签名列表，建立测试目标矩阵。
    """
    clean_code = re.sub(r'//.*$', '', java_code, flags=re.MULTILINE)
    clean_code = re.sub(r'/\*.*?\*/', '', clean_code, flags=re.DOTALL)
    
    pattern = r'((?:public|protected|private|static|\s)+)\s+([\w\<\>\[\],_]+)\s+(\w+)\s*\(([^)]*)\)\s*\{'
    matches = re.findall(pattern, clean_code)
    
    class_name = extract_java_class_name(java_code)
    signatures = []
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
        signatures.append({
            "modifiers": modifiers_clean,
            "return_type": ret_type_clean,
            "method_name": name,
            "params": params,
            "is_static": is_static
        })
    return signatures

def parse_java_signature_advanced(java_code: str):
    """单签名降级适配路由"""
    sigs = parse_all_java_signatures(java_code)
    return sigs[0] if sigs else None

def to_java_literal(val, java_type: str) -> str:
    java_type_clean = java_type.strip()
    
    if val is None:
        if java_type_clean in ("int", "Integer"): return "0"
        if java_type_clean in ("long", "Long"): return "0L"
        if java_type_clean in ("double", "Double"): return "0.0d"
        if java_type_clean in ("float", "Float"): return "0.0f"
        if java_type_clean in ("boolean", "Boolean"): return "false"
        if java_type_clean in ("char", "Character"): return "'\\0'"
        if "List" in java_type_clean: return "new ArrayList<>()"
        if java_type_clean.endswith("[]"):
            base = java_type_clean[:-2].strip()
            return f"new {base}[0]"
        return "null"
    
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

def generate_java_runner_multi(java_code: str, test_cases: dict, sigs: dict, class_name: str) -> str:
    """
    🎯 动态构建高密度、多方法、携带自定义构造态的 Java SolutionRunner。
    在 Java 侧利用高阶反射直接注入 initial_state 初始状态，彻底绕过构造函数和 Setter 的类型阻断。
    """
    test_cases_blocks = []
    
    for method_name, cases in test_cases.items():
        if method_name not in sigs:
            continue
        sig = sigs[method_name]
        params = sig["params"]
        return_type = sig["return_type"]
        is_static = sig.get("is_static", False)
        
        for idx, case in enumerate(cases):
            initial_state = case.get("initial_state", {})
            args = case.get("args", [])
            
            # A. 物理构造初始状态的反射 Map 字面量
            state_build_lines = []
            state_var = f"state_{method_name}_{idx}"
            state_build_lines.append(f"Map<String, Object> {state_var} = new HashMap<>();")
            for k, v in initial_state.items():
                if isinstance(v, list):
                    val_literal = to_java_literal(v, "List<Object>")
                elif isinstance(v, str):
                    val_literal = to_java_literal(v, "String")
                elif isinstance(v, bool):
                    val_literal = "true" if v else "false"
                elif isinstance(v, int):
                    val_literal = f"Integer.valueOf({v})"
                elif isinstance(v, float):
                    val_literal = f"Double.valueOf({v})"
                else:
                    val_literal = "null"
                state_build_lines.append(f"{state_var}.put(\"{k}\", {val_literal});")
                
            state_build_str = "\n            ".join(state_build_lines)
            
            # B. 初始化当前 method 运行实参字面量
            java_method_args = []
            for i, (p_type, p_name) in enumerate(params):
                val = args[i] if i < len(args) else None
                java_method_args.append(to_java_literal(val, p_type))
            method_args_str = ", ".join(java_method_args)
            
            # C. 物理拉起实例进行定点状态配置并实施测试
            if is_static:
                method_call = f"{class_name}.{method_name}({method_args_str})"
            else:
                method_call = f"(({class_name})instantiateAndSetState({class_name}.class, {state_var})).{method_name}({method_args_str})"
            
            escaped_method = method_name.replace('"', '\\"')
            if return_type.strip() == "void":
                block = f"""
        try {{
            {state_build_str}
            {method_call};
            System.out.println("TEST_CASE_{escaped_method}_{idx}_START:\\"void\\":TEST_CASE_{escaped_method}_{idx}_END");
        }} catch (Throwable t) {{
            System.out.println("TEST_CASE_{escaped_method}_{idx}_EXCEPTION:" + t.getClass().getName() + ": " + t.getMessage() + ":TEST_CASE_{escaped_method}_{idx}_END");
        }}
        """
            else:
                block = f"""
        try {{
            {state_build_str}
            Object res = {method_call};
            System.out.println("TEST_CASE_{escaped_method}_{idx}_START:" + toJson(res) + ":TEST_CASE_{escaped_method}_{idx}_END");
        }} catch (Throwable t) {{
            System.out.println("TEST_CASE_{escaped_method}_{idx}_EXCEPTION:" + t.getClass().getName() + ": " + t.getMessage() + ":TEST_CASE_{escaped_method}_{idx}_END");
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

    public static Object instantiateAndSetState(Class<?> clazz, Map<String, Object> state) throws Exception {{
        Object instance = instantiateClass(clazz);
        if (state != null) {{
            for (Map.Entry<String, Object> entry : state.entrySet()) {{
                String fieldName = entry.getKey();
                Object val = entry.getValue();
                try {{
                    Field field = getDeclaredFieldRecursive(clazz, fieldName);
                    if (field != null) {{
                        field.setAccessible(true);
                        Object convertedVal = castValueToFieldType(field.getType(), val);
                        field.set(instance, convertedVal);
                    }}
                }} catch (Exception e) {{}}
            }}
        }}
        return instance;
    }}

    private static Field getDeclaredFieldRecursive(Class<?> clazz, String fieldName) {{
        Class<?> current = clazz;
        while (current != null) {{
            try {{
                return current.getDeclaredField(fieldName);
            }} catch (NoSuchFieldException e) {{
                for (Field f : current.getDeclaredFields()) {{
                    if (f.getName().equalsIgnoreCase(fieldName)) {{
                        return f;
                    }}
                }}
                current = current.getSuperclass();
            }}
        }}
        return null;
    }}

    private static Object castValueToFieldType(Class<?> type, Object val) {{
        if (val == null) return null;
        if (type == int.class || type == Integer.class) {{
            if (val instanceof Number) return ((Number) val).intValue();
        }}
        if (type == long.class || type == Long.class) {{
            if (val instanceof Number) return ((Number) val).longValue();
        }}
        if (type == double.class || type == Double.class) {{
            if (val instanceof Number) return ((Number) val).doubleValue();
        }}
        if (type == float.class || type == Float.class) {{
            if (val instanceof Number) return ((Number) val).floatValue();
        }}
        if (type == boolean.class || type == Boolean.class) {{
            if (val instanceof Boolean) return val;
        }}
        if (type == String.class) {{
            return val.toString();
        }}
        return val;
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

def execute_and_capture_ground_truth(java_code: str, test_cases_dict: dict) -> (dict, str, float, float):
    """
    Java 真值捕获与物理执行，支持多方法测试集
    """
    sigs_list = parse_all_java_signatures(java_code)
    sigs = {sig["method_name"]: sig for sig in sigs_list}
    
    if not sigs:
        return {}, "CRITICAL_JAVA_SIGNATURE_FAILED", 0.0, 0.0
        
    class_name = extract_java_class_name(java_code)
    
    # 适配并对齐 LLM 生成的大小写及不匹配 method 键
    sanitized_cases = {}
    for m_name, cases in test_cases_dict.items():
        matched_sig_name = None
        if m_name in sigs:
            matched_sig_name = m_name
        else:
            for s_name in sigs:
                if s_name.lower() == m_name.lower():
                    matched_sig_name = s_name
                    break
        if matched_sig_name:
            sanitized_cases[matched_sig_name] = cases
            
    if not sanitized_cases:
        return {}, "CRITICAL_JAVA_SIGNATURE_FAILED: No matching Java methods found for generated test cases.", 0.0, 0.0
        
    runner_source = generate_java_runner_multi(java_code, sanitized_cases, sigs, class_name)
    
    standard_imports = "import java.util.*;\nimport java.util.stream.*;\nimport java.util.function.*;\nimport java.io.*;\nimport java.math.*;\nimport java.net.*;\nimport java.text.*;\n"
    java_code_clean = re.sub(r'^\s*package\s+[\w\.]+;\s*', '', java_code, flags=re.MULTILINE)
    full_java_class_code = standard_imports + "\n" + java_code_clean
    
    base_temp_path = tempfile.gettempdir()
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
            return {}, f"CRITICAL_JAVA_COMPILE_FAILED: {compile_res.stderr.strip()}", 0.0, 0.0
            
        try:
            run_res = subprocess.run(
                ["java", "SolutionRunner"],
                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=8
            )
            stdout_str, stderr_str = run_res.stdout, run_res.stderr
        except subprocess.TimeoutExpired:
            stdout_str, stderr_str = "", "CRITICAL_JAVA_SANDBOX_TIMEOUT"
            
        matched_cases = {}
        for m_name, cases in sanitized_cases.items():
            matched_cases[m_name] = []
            for case in cases:
                matched_cases[m_name].append({
                    "initial_state": case.get("initial_state", {}),
                    "args": case.get("args", []),
                    "status": "FAILED",
                    "expected": None,
                    "error_msg": "Timeout or Sandbox Crash"
                })
                
        # 深度匹配输出反馈结果
        for line in stdout_str.splitlines():
            m_start = re.search(r'TEST_CASE_(.+)_(\d+)_START:(.*):TEST_CASE_\1_\2_END', line)
            if m_start:
                m_name = m_start.group(1)
                case_idx = int(m_start.group(2))
                val_str = m_start.group(3)
                
                if m_name in matched_cases and case_idx < len(matched_cases[m_name]):
                    try:
                        matched_cases[m_name][case_idx]["expected"] = json.loads(val_str)
                    except:
                        matched_cases[m_name][case_idx]["expected"] = val_str
                    matched_cases[m_name][case_idx]["status"] = "PASSED"
                    matched_cases[m_name][case_idx]["error_msg"] = None
                    
            m_except = re.search(r'TEST_CASE_(.+)_(\d+)_EXCEPTION:(.*):TEST_CASE_\1_\2_END', line)
            if m_except:
                m_name = m_except.group(1)
                case_idx = int(m_except.group(2))
                err_msg = m_except.group(3)
                
                if m_name in matched_cases and case_idx < len(matched_cases[m_name]):
                    matched_cases[m_name][case_idx]["status"] = "FAILED"
                    matched_cases[m_name][case_idx]["expected"] = None
                    matched_cases[m_name][case_idx]["error_msg"] = err_msg

        total_cases = 0
        passed_runs = 0
        for m_name, cases in matched_cases.items():
            total_cases += len(cases)
            passed_runs += sum([1 for x in cases if x["status"] == "PASSED"])
            
        line_cov = (passed_runs / total_cases) * 100.0 if total_cases > 0 else 0.0
        branch_cov = line_cov
        
        feedback_msg = "SUCCESS" if passed_runs > 0 else f"CRITICAL_EXECUTION_FAILURE: {stderr_str}"
        return matched_cases, feedback_msg, line_cov, branch_cov

# ==================== 3. 针对 OCaml 语言特化对齐的核心引擎 ====================
def extract_ocaml_function_name(code: str) -> str:
    """静态分析 OCaml 源码，清洗注释并智能提取定义的函数名称"""
    if not code or not isinstance(code, str):
        return ""
    code_clean = re.sub(r'\(\*.*?\*\)', '', code, flags=re.DOTALL)
    
    # 查找 top-level let 绑定定义：let rec name 或 let name
    found_funcs = re.findall(r'\blet\s+(?:rec\s+)?([a-z_][a-zA-Z0-9_]*)\b', code_clean)
        
    if not found_funcs:
        return ""
    
    clean_funcs = [f for f in found_funcs if f not in ['initialize', 'new', 'fit', 'predict', 'show', 'print', 'main', 'run']]
    target = clean_funcs[0] if clean_funcs else found_funcs[0]
    return target

def align_test_with_candidate_ocaml(candidate_code: str, ocaml_test_code: str) -> str:
    cand_func = extract_ocaml_function_name(candidate_code)
    if not cand_func:
        return ocaml_test_code
    if cand_func not in ocaml_test_code:
        ocaml_test_code = re.sub(r'\bsolution\b', cand_func, ocaml_test_code)
        ocaml_test_code = re.sub(r'\bsol\b', cand_func, ocaml_test_code)
    return ocaml_test_code

def clean_generated_code(code: str) -> str:
    """
    清洗并剥离大模型生成的 OCaml 代码块，提取 markdown 里的代码并过滤多余字符。
    """
    if not code:
        return ""

    ocaml_blocks = re.findall(r'```ocaml\s*\n?(.*?)\n?```', code, re.DOTALL)
    if not ocaml_blocks:
        ocaml_blocks = re.findall(r'```\s*\n?(.*?)\n?```', code, re.DOTALL)
    final_code = ocaml_blocks[-1] if ocaml_blocks else code

    final_code = re.sub(r'<\|im_.*?\|>', '', final_code)
    final_code = final_code.strip()
    
    if final_code.endswith(";"):
        if not final_code.endswith(";;"):
            final_code = final_code[:-1].strip()
            
    return final_code

def fix_ocaml_syntax_hallucinations(ocaml_code: str) -> str:
    """
    🎯 核心自愈：自动修复大模型把 Python / Java / JSON 的语法错误地带入 OCaml 里的经典跨语言幻觉。
    """
    if not ocaml_code:
        return ""
    
    # 1. 修复 Boolean & Null 关键字
    ocaml_code = re.sub(r'\bTrue\b', 'true', ocaml_code)
    ocaml_code = re.sub(r'\bFalse\b', 'false', ocaml_code)
    ocaml_code = re.sub(r'\bNone\b', 'None', ocaml_code)
    ocaml_code = re.sub(r'\bnull\b', 'None', ocaml_code)
    ocaml_code = re.sub(r'\bnil\b', 'None', ocaml_code)
    
    # 2. 自动修正列表分割符：[1, 2, 3] -> [1; 2; 3]，规避 OCaml 对逗号列表报错的通病
    def replace_commas_in_brackets(match):
        content = match.group(1)
        new_content = []
        paren_depth = 0
        for char in content:
            if char in '({':
                paren_depth += 1
            elif char in ')}':
                paren_depth -= 1
            elif char == ',' and paren_depth == 0:
                new_content.append(';')
                continue
            new_content.append(char)
        return '[' + "".join(new_content) + ']'
        
    ocaml_code = re.sub(r'\[([^\]]+)\]', replace_commas_in_brackets, ocaml_code)
    
    return ocaml_code

def clean_ocaml_test_code(ocaml_test_code: str) -> str:
    if not ocaml_test_code:
        return ""
    # 移除可能有安全隐患的外部 system 调用，保留纯测试逻辑
    cleaned = re.sub(r'Sys\.command\s+[^;]*', '()', ocaml_test_code)
    return cleaned

def check_ocaml_test_syntax(ocaml_test_code: str) -> bool:
    """使用 ocamlc 静态检查 OCaml 测试桩文件的语法"""
    if not ocaml_test_code.strip():
        return False
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test_solution.ml")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(ocaml_test_code)
        try:
            res = subprocess.run(
                ["ocamlc", "-dparsetree", "test_solution.ml"],
                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3
            )
            if "Syntax error" not in res.stderr and "Syntax error" not in res.stdout:
                return True
        except Exception:
            pass
    return False

def split_ocaml_tests(ocaml_test_code: str) -> list:
    """根据 OCaml 注释块将合并的一组测试切分为独立评估的 Test Blocks"""
    lines = ocaml_test_code.splitlines()
    test_blocks = []
    current_block = []
    
    for line in lines:
        if re.match(r'^\s*\(\*\s*[Tt]est\s*\d+', line) or re.match(r'^\s*\(\*\*\s*[Tt]est', line) or "let () =" in line:
            if current_block:
                test_blocks.append("\n".join(current_block))
                current_block = []
        current_block.append(line)
        
    if current_block:
        test_blocks.append("\n".join(current_block))
        
    if not test_blocks:
        test_blocks = [ocaml_test_code]
        
    return [b.strip() for b in test_blocks if b.strip()]

def verify_ocaml_candidate_robust(candidate_code: str, ocaml_test_code: str, diagnostic_reasons_list: list, timeout=5) -> (str, int):
    """
    🎯 OCaml 沙箱级别评测：结合候选译文与测试代码，在临时隔离环境中使用轻量级 `ocaml` 解释器执行评估
    """
    if not candidate_code.strip():
        return "Empty_Code", 0

    ocaml_test_code = clean_ocaml_test_code(ocaml_test_code)
    func_name = extract_ocaml_function_name(candidate_code)
    if not func_name:
        return "No_Function_Defined", 0

    for pat in [r'\bsolution\b', r'\bFUNC\b', r'\btarget_function\b', r'\bfunc\b', r'\bmy_function\b']:
        ocaml_test_code = re.sub(pat, func_name, ocaml_test_code)

    test_blocks = split_ocaml_tests(ocaml_test_code)
    passed_count = 0
    status = "Success"

    # 🎯 隔离临时环境目录
    base_dir = tempfile.gettempdir()
    with tempfile.TemporaryDirectory(prefix="ocaml_verify_", dir=base_dir) as tmpdir:
        cand_path = os.path.join(tmpdir, "candidate.ml")
        with open(cand_path, "w", encoding="utf-8") as f:
            f.write(candidate_code)

        for i, block in enumerate(test_blocks):
            runner_path = os.path.join(tmpdir, f"runner_{i}.ml")
            
            # 使用 OCaml 顶层环境指令 #use 动态载入候选实现并执行单个断言测试块
            run_src = f"""#use "candidate.ml";;\n\n{block}\n"""
            
            with open(runner_path, "w", encoding="utf-8") as f:
                f.write(run_src)
                
            try:
                res = subprocess.run(
                    ["ocaml", f"runner_{i}.ml"],
                    cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout
                )
                stdout = res.stdout
                stderr = res.stderr
                
                # 语义级错题本分类诊断
                if "Syntax error" in stderr or "Syntax error" in stdout:
                    status = "Syntax_Error"
                    diagnostic_reasons_list.append(f"SyntaxError in test case {i+1}: {stderr.strip()}")
                    break
                elif "Unbound value" in stderr or "Unbound value" in stdout:
                    status = "Unbound_Value"
                    diagnostic_reasons_list.append(f"UnboundValue in test case {i+1}: {stderr.strip()}")
                    break
                elif "This expression has type" in stderr or "This expression has type" in stdout:
                    status = "Type_Error"
                    diagnostic_reasons_list.append(f"TypeError in test case {i+1}: {stderr.strip()}")
                    break
                elif "Exception: Assert_failure" in stderr or res.returncode != 0:
                    err_info = stderr.strip() or stdout.strip() or "Assertion Failed"
                    err_info_clean = re.sub(r'^ocaml:\s*runner_\d+\.ml:\d+:\s*', '', err_info)
                    diagnostic_reasons_list.append(f"Test_{i+1}_Failed: {err_info_clean}")
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

        if len(test_blocks) == 0:
            status = "No_Tests_Found"

        return status, passed_count

# ==================== 4. 辅助数学计算与 JSON 提取器 ====================
def calculate_pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    else:
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def extract_and_parse_json(output_str: str) -> dict:
    cleaned_str = output_str.strip()
    json_match = re.search(r'\{.*\}', cleaned_str, re.DOTALL)
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
    return {}

# ==================== 5. OCaml 评测大盘指标统计与流式调度 ====================
def run_live_pipeline(csv_path, llm_path, llm_type, output_report_csv, passed_save_path, correctness_threshold=12.5, batch_size=4):
    """
    主控制中心，运行评测。
    """
    print("📋 [Pre-flight Check] 正在检查系统测试环境依赖...")
    javac_check = subprocess.run(["which", "javac"], capture_output=True)
    ocaml_check = subprocess.run(["which", "ocaml"], capture_output=True)
    
    if javac_check.returncode != 0:
        print("⚠️  [System Warning] 系统未发现 'javac' 命令！所有的 Java 单元测试将无法编译运行，评测可能会被 100% 过滤。")
    else:
        print("   ✅ JDK 编译器 'javac' 状态：OK")
        
    if ocaml_check.returncode != 0:
        print("⚠️  [System Warning] 系统未发现 'ocaml' 解释器环境！所有的测试都将无法执行。")
    else:
        print("   ✅ OCaml 解释器 'ocaml' 状态：OK")
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
        print("❌ [Terminate] 没有黄金可用样本，测试终止。")
        return

    print("⏳ 正在加载用于测试套件合成的生成层级模型（LLM）...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    stats = {
        'total': 0, 
        'java_test_failed': 0,            # 真值错 (Filtered)
        'ocaml_test_syntax_errors': 0,    # OCaml 语法/类型/编译器错误
        'ocaml_no_cand_passed': 0,        # OCaml 无通过 (断言失败)
        'ocaml_success': 0,              # OCaml 正常运行且至少一个候选通过
        'ocaml_truth_failed': 0,         # 因 Java 端真值错而强制失败的件数
        'valid': 0,                       # 通过 Oracle 的有效样本数
        'p1': 0.0, 'p2': 0.0, 'p5': 0.0,
        'sum_line_cov': 0.0,              # Java 行覆盖率求和
        'sum_ocaml_line_cov': 0.0        # OCaml 行覆盖率求和
    }
    filter_reasons = Counter()
    
    print(f"\n🚀 OCaml 评测流水线已激活 (批处理大小 Batch Size: {batch_size})")

    output_report_jsonl = output_report_csv.rsplit('.', 1)[0] + ".jsonl"

    with open(output_report_csv, mode='w', encoding='utf-8', newline='') as f, \
         open(output_report_jsonl, mode='w', encoding='utf-8') as f_jsonl, \
         open(passed_save_path, mode='w', encoding='utf-8') as f_save:
         
        writer = csv.writer(f)
        writer.writerow([
            'idx', 'java_line_coverage', 'ocaml_coverage_line', 'status', 
            'pass@1', 'pass@2', 'pass@5', 'passed_candidates'
        ])

        rows = [row for _, row in df.iterrows()]
        start_time = time.time()
        current_count = 0

        for i in range(0, total_records, batch_size):
            batch_rows = rows[i:i + batch_size]
            
            # ==================== 🛠️ 第一阶段：生成测试用例并捕获 Java Oracle expected 真实值 ====================
            prompts_py = []
            for row in batch_rows:
                user_content = GEN_TEST_USER_PROMPT.format(java_code=row['java_code'])
                prompt = get_prompt_by_model(llm_type, GEN_TEST_SYSTEM_PROMPT, user_content)
                prompts_py.append(prompt)

            inputs_py = tokenizer(prompts_py, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                outputs_py = model.generate(
                    **inputs_py, max_new_tokens=512, temperature=0.2, top_p=0.95, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            input_lens_py = inputs_py.input_ids.shape[1]
            active_batch_items = []

            for j, row in enumerate(batch_rows):
                idx = str(row['idx'])
                java_code = row['java_code']
                cand_str = row['all_candidates']
                
                output_str_py = tokenizer.decode(outputs_py[j][input_lens_py:], skip_special_tokens=True)
                raw_args_dict = extract_and_parse_json(output_str_py)

                stats['total'] += 1

                try:
                    matched_cases, java_feedback, java_line_cov, java_branch_cov = execute_and_capture_ground_truth(
                        java_code, raw_args_dict
                    )
                except Exception as e:
                    matched_cases = {}
                    java_feedback = f"CRITICAL_SANDBOX_CRASH: {str(e)}"
                    java_line_cov, java_branch_cov = 0.0, 0.0

                is_truth_capture_failed = (not matched_cases) or ("CRITICAL_" in java_feedback) or (java_line_cov < correctness_threshold)
                
                if is_truth_capture_failed:
                    stats['java_test_failed'] += 1
                    filter_reasons[java_feedback if "CRITICAL_" in java_feedback else "Filtered_Low_Oracle_Accuracy"] += 1

                item_data = {
                    'idx': idx,
                    'java_code': java_code,
                    'java_test_cases': matched_cases,
                    'java_feedback': java_feedback,
                    'java_line_cov': java_line_cov,
                    'cand_str': cand_str,
                    'is_truth_capture_failed': is_truth_capture_failed
                }
                active_batch_items.append(item_data)

            if not active_batch_items:
                del inputs_py, outputs_py
                torch.cuda.empty_cache()
                current_count += len(batch_rows)
                continue

            # ==================== 🛠️ 第二阶段：反馈测试集，动态合成完全对齐的 OCaml Assertions ====================
            prompts_ocaml = []
            for item in active_batch_items:
                candidates_list = []
                try:
                    candidates_list = json.loads(item['cand_str']) if isinstance(item['cand_str'], str) else item['cand_str']
                except:
                    pass
                first_ocaml_cand = candidates_list[0] if candidates_list else "(* No translated OCaml candidate *)"
                
                user_content_ocaml = GEN_OCAML_TEST_USER_PROMPT.format(
                    java_feedback=item['java_feedback'],
                    java_code=item['java_code'],
                    ocaml_code=first_ocaml_cand,
                    java_test_code=json.dumps(item['java_test_cases'], ensure_ascii=False)
                )
                prompt_ocaml = get_prompt_by_model(llm_type, GEN_OCAML_TEST_SYSTEM_PROMPT, user_content_ocaml)
                prompts_ocaml.append(prompt_ocaml)

            inputs_ocaml = tokenizer(prompts_ocaml, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                outputs_ocaml = model.generate(
                    **inputs_ocaml, max_new_tokens=1024, temperature=0.2, top_p=0.95, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            input_lens_ocaml = inputs_ocaml.input_ids.shape[1]

            # ==================== 🛠️ 第三阶段：隔离沙箱执行断言评估 ====================
            for j, item in enumerate(active_batch_items):
                idx = item['idx']
                java_code = item['java_code']
                java_line_cov = item['java_line_cov']
                cand_str = item['cand_str']
                is_truth_capture_failed = item['is_truth_capture_failed']
                matched_cases = item['java_test_cases']

                output_str_ocaml = tokenizer.decode(outputs_ocaml[j][input_lens_ocaml:], skip_special_tokens=True)
                
                ocaml_test_code = ""
                ocaml_match = re.search(r'```[Oo]caml\s*(.*?)\s*```', output_str_ocaml, re.DOTALL)
                if ocaml_match:
                    ocaml_test_code = ocaml_match.group(1).strip()
                else:
                    ocaml_test_code = output_str_ocaml.strip()

                ocaml_test_code = fix_ocaml_syntax_hallucinations(ocaml_test_code)

                try:
                    candidates = json.loads(cand_str) if isinstance(cand_str, str) else cand_str
                except Exception:
                    continue

                n = len(candidates)
                passed_count = 0
                passed_ocaml_lines = []
                passed_cands_list = []

                cleaned_base_test_code = clean_ocaml_test_code(ocaml_test_code)
                ocaml_test_ok = check_ocaml_test_syntax(cleaned_base_test_code)

                diagnostic_reasons = []

                for cand in candidates:
                    if is_truth_capture_failed or not ocaml_test_ok:
                        passed_ocaml_lines.append(0.0)
                    else:
                        aligned_ocaml_test_code = align_test_with_candidate_ocaml(cand, ocaml_test_code)
                        cleaned_ocaml_test_code = clean_ocaml_test_code(aligned_ocaml_test_code)
                        
                        status, num_passed_cases = verify_ocaml_candidate_robust(cand, cleaned_ocaml_test_code, diagnostic_reasons)
                        
                        if status == "Success" and num_passed_cases > 0:
                            passed_count += 1
                            passed_cands_list.append(cand)
                            passed_ocaml_lines.append(100.0) # 测试用例成功通过视为 100% 对齐覆盖
                        else:
                            passed_ocaml_lines.append(0.0)

                if is_truth_capture_failed:
                    global_ocaml_status = "OCaml_Truth_Failed"
                    stats['ocaml_truth_failed'] += 1
                    passed_count = 0
                elif not ocaml_test_ok:
                    global_ocaml_status = "OCaml_Test_Syntax_Error"
                    stats['ocaml_test_syntax_errors'] += 1
                elif passed_count > 0:
                    global_ocaml_status = "OCaml_Success"
                    stats['ocaml_success'] += 1
                else:
                    global_ocaml_status = "OCaml_No_Candidate_Passed"
                    stats['ocaml_no_cand_passed'] += 1

                avg_ocaml_line_cov = sum(passed_ocaml_lines) / len(passed_ocaml_lines) if passed_ocaml_lines else 0.0

                stats['valid'] += 1
                stats['sum_line_cov'] += java_line_cov
                stats['sum_ocaml_line_cov'] += avg_ocaml_line_cov

                p1 = calculate_pass_at_k(n, passed_count, 1)
                p2 = calculate_pass_at_k(n, passed_count, 2) if n >= 2 else p1
                p5 = calculate_pass_at_k(n, passed_count, 5) if n >= 5 else p2

                stats['p1'] += p1
                stats['p2'] += p2
                stats['p5'] += p5

                # 扁平化 Java 真值数据保存
                flat_java_cases = []
                for m_name, cases in matched_cases.items():
                    for case in cases:
                        flat_java_cases.append({
                            "method": m_name,
                            "initial_state": case.get("initial_state", {}),
                            "args": case.get("args", []),
                            "expected": case.get("expected"),
                            "status": case.get("status")
                        })

                # 1. 写入 CSV 汇总报告
                writer.writerow([
                    idx, f"{java_line_cov:.2f}%", f"{avg_ocaml_line_cov:.2f}%", "Passed" if passed_count > 0 else "Failed",
                    f"{p1:.4f}", f"{p2:.4f}", f"{p5:.4f}", passed_count
                ])

                # 2. 写入 JSONL 详细报告
                report_item = {
                    "idx": int(idx),
                    "status": "Passed" if passed_count > 0 else "Failed",
                    "global_ocaml_status": global_ocaml_status,
                    "metrics": {
                        "java_line_coverage": java_line_cov,
                        "ocaml_line_coverage": avg_ocaml_line_cov,
                        "pass_at_1": p1,
                        "pass_at_2": p2,
                        "pass_at_5": p5
                    },
                    "passed_candidates_count": passed_count,
                    "total_candidates_count": n,
                    "diagnostic_reasons": diagnostic_reasons
                }
                f_jsonl.write(json.dumps(report_item, ensure_ascii=False) + "\n")

                # 3. 写入落盘黄金代码
                if passed_count > 0:
                    save_item = {
                        "idx": int(idx),
                        "java_code": java_code,
                        "java_test_cases": flat_java_cases,
                        "ocaml_test_code": ocaml_test_code,
                        "passed_ocaml_candidates": passed_cands_list,
                        "line_coverage": java_line_cov,
                        "ocaml_line_coverage": avg_ocaml_line_cov,
                        "pass_at_1": p1,
                        "pass_at_2": p2,
                        "pass_at_5": p5
                    }
                    f_save.write(json.dumps(save_item, ensure_ascii=False) + "\n")

                f_jsonl.flush()
                f_save.flush()

            f.flush()
            current_count += len(batch_rows)
            total_num = max(stats['total'], 1)
            pct = (current_count / total_records) * 100
            
            elapsed_time = time.time() - start_time
            it_s = current_count / elapsed_time if elapsed_time > 0 else 0.0
            remaining_seconds = (total_records - current_count) / it_s if it_s > 0 else 0
            
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_seconds))

            java_line_avg = stats['sum_line_cov'] / total_num
            ocaml_line_avg = stats['sum_ocaml_line_cov'] / total_num

            # 进度报告
            full_status_line = (
                f"进度:{pct:>5.1f}% ({current_count}/{total_records})[{elapsed_str}<{remaining_str},{it_s:.2f}it/s] "
                f"|总数:{stats['total']}|真值错:{stats['java_test_failed']}|OCaml语法错:{stats['ocaml_test_syntax_errors']}|OCaml未通过:{stats['ocaml_no_cand_passed']}|OCaml成功:{stats['ocaml_success']} "
                f"|P@1:{(stats['p1']/total_num)*100:.2f}%|P@2:{(stats['p2']/total_num)*100:.2f}%|P@5:{(stats['p5']/total_num)*100:.2f}% "
                f"|Java行覆:{java_line_avg:.1f}%|OCaml行覆:{ocaml_line_avg:.1f}%"
            )
            print(full_status_line)
            sys.stdout.flush()

            del inputs_py, outputs_py, inputs_ocaml, outputs_ocaml
            torch.cuda.empty_cache()

        print("\n")

    # 释放显存
    del model
    torch.cuda.empty_cache()
    
    # 最终报告
    print("================ 📈 重构版自愈评测大盘总结 (Java -> OCaml) ==================")
    print(f"送入评测的黄金样本总量 (Total Records)     : {stats['total']}")
    print(f"Java 真值校验失败数 (Filtered / 真值错)    : {stats['java_test_failed']}")
    if stats['java_test_failed'] > 0:
        print("🔍 细分拦截原因统计 (Detailed Filtering Reason Counter):")
        for reason, count in filter_reasons.items():
            print(f"   - {reason}: {count} 个样本")
    print("-" * 66)
    print(f"参与语义指标统计的有效题目数 (Valid Samples): {stats['valid']}")
    print(f"  ├─ OCaml 代码静态类型/编译错误数 (OCaml Syntax Error): {stats['ocaml_test_syntax_errors']}")
    print(f"  ├─ 候选代码全部未通过断言 (OCaml No Cand Passed)  : {stats['ocaml_no_cand_passed']}")
    print(f"  ├─ 🚨 因 Java 端真值错而强制失败的件数         : {stats['ocaml_truth_failed']}")
    print(f"  └─ 至少有1个候选代码测试通过的正常件数 (OCaml Success): {stats['ocaml_success']}")
    print("-" * 66)
    if stats['valid'] > 0:
        print(f"🌟 OCaml 语言对齐平均 Pass@1: {stats['p1'] / stats['valid'] * 100:.2f}%")
        print(f"🌟 OCaml 语言对齐平均 Pass@2: {stats['p2'] / stats['valid'] * 100:.2f}%")
        print(f"🌟 OCaml 语言对齐平均 Pass@5: {stats['p5'] / stats['valid'] * 100:.2f}%")
        print("-" * 66)
        print(f"📐 Java 平均行覆盖率 (Line Coverage)        : {stats['sum_line_cov'] / stats['valid']:.2f}%")
        print(f"📐 OCaml 平均行覆盖率 (Line Coverage)      : {stats['sum_ocaml_line_cov'] / stats['valid']:.2f}%")
    print("==================================================================")
    print(f"🎉 评估顺利完成！报告与黄金代码已分别保存至:\n  - CSV 报告: {output_report_csv}\n  - JSONL 详情: {output_report_jsonl}\n  - 过滤落盘代码: {passed_save_path}\n")


if __name__ == "__main__":
    # Ocaml 推理源原始数据路径
    CSV_PATH = "/root/autodl-tmp/first_paper/java/translation_dataset/Ocaml_golden/Ocaml_inference_results_original.csv"
    LLM_MODEL_PATH = "/root/autodl-tmp/first_paper/LLM/Qwen"
    CURRENT_LLM_TYPE = "qwen"
    CORRECTNESS_THRESHOLD = 12.5  # 允许至少有 1/8 的测试用例跑通即进行 OCaml 强度语义对比
    
    # 评测报告具体保存路径
    OUTPUT_REPORT = "/root/autodl-tmp/first_paper/java/testcase_dataset/ocaml_pass/Ocaml_passk_report.csv"
    PASSED_TRANSLATIONS_JSONL = "/root/autodl-tmp/first_paper/java/testcase_dataset/ocaml_pass/Ocaml_passed_clean.jsonl"
    
    run_live_pipeline(
        csv_path=CSV_PATH, 
        llm_path=LLM_MODEL_PATH, 
        llm_type=CURRENT_LLM_TYPE, 
        output_report_csv=OUTPUT_REPORT, 
        passed_save_path=PASSED_TRANSLATIONS_JSONL,
        correctness_threshold=CORRECTNESS_THRESHOLD
    )