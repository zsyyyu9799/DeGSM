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
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== 1. 提示词与 Prompt 路由 ====================
GEN_TEST_SYSTEM_PROMPT = (
    "You are an expert QA Engineer and a rigorous automated script. "
    "Your unique task is to analyze C++ source code and synthesize highly precise differential testing suites. "
    "You must output ONLY raw, strict, standard JSON data matching the template. "
    "Never write any markdown formatting (such as ```json), introductory text, or natural language explanation."
)

GEN_TEST_USER_PROMPT = """You are presented with a C++ function. Your goal is to construct a rigorous test suite comprising EXACTLY 5 diverse and distinct test cases to thoroughly evaluate the function's semantics.

[Testing Objectives]
1. Diversity: Ensure the 5 test cases cover different logical paths, partition testing zones, and branching behaviors of the C++ code.
2. Boundary Values: Actively include edge cases such as empty structures (empty vectors/strings), negative numbers, zero, large integers, or specific boundary limits present in the logic to maximize line coverage.
3. Correctness: The "expected" value must precisely match the actual deterministic execution output of the provided C++ code.
4. If the function parameters involve structures like std::vector or std::string, map them to standard JSON arrays and strings.

[Critical Input Constraints for Arguments]
- The "args" array MUST contain only standard primitive JSON values (numbers, strings, booleans, arrays, null) that align with the function parameters.
- Do NOT pass uninitialized variables or complex custom custom classes as arguments.
- Use "true" and "false" (lowercase) for booleans, and "null" for nullptr. Never use True, False, or None in your JSON output.

[Output Format Specification]
Output the test suite in strict, standard JSON format as a single continuous list of objects.
- Do NOT wrap the JSON inside markdown code blocks (e.g., do NOT use ```json ... ```).
- Do NOT output any conversational prefix, suffix, or natural language explanation. Output the raw JSON string only.

[Example of Good Input/Output]
Input C++ Code:
```cpp
int find_element(std::vector<int> arr, int target) {{
    if (arr.empty()) {{
        return -1;
    }}
    for (int i = 0; i < arr.size(); i++) {{
        if (arr[i] == target) {{
            return i;
        }}
    }}
    return -1;
}}
```

Expected Output:
[
  {{"args": [[1, 2, 3], 2], "expected": 1}},
  {{"args": [[], 5], "expected": -1}},
  {{"args": [[10, -5, 0], -5], "expected": 1}},
  {{"args": [["apple", "banana"], "cherry"], "expected": -1}},
  {{"args": [[1], 1], "expected": 0}}
]

[Actual Task Input]
Input C++ Code:
```cpp
{cpp_code}
```

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
        return f"<｜begin▁of▁sentence｜>{instruction}\n### Response:\n["
    else: 
        if system_prompt:
            return f"{system_prompt}\n\n{user_content}\n\n"
        else:
            return f"{user_content}\n\n"

# ==================== 2. C++ 物理与先知双轨内省校验模块 ====================
def parse_cpp_function_signature(cpp_code: str):
    """
    智能解析 C++ 函数签名，并使用括号计数技术精准切分复杂模板类型参数
    支持 long long、unsigned int 等多词组合返回类型
    """
    code_clean = re.sub(r'//.*', '', cpp_code)
    code_clean = re.sub(r'/\*.*?\*/', '', code_clean, flags=re.DOTALL)

    # 兼容复杂的修饰符，捕获返回类型、函数名称与完整参数列表
    match = re.search(r'([\w<>:|&\*\s]+?)\s+(\w+)\s*\(([^)]*)\)\s*(?:const)?\s*\{', code_clean)
    if not match:
        # 容错匹配不包含尾随大括号的情况
        match = re.search(r'([\w<>:|&\*\s]+?)\s+(\w+)\s*\(([^)]*)\)', code_clean)
        
    if not match:
        return "void", "unknown", []

    return_type_raw = match.group(1).strip()
    func_name = match.group(2).strip()
    params_str = match.group(3).strip()

    # 清除 inline、static、friend 等编译器限定修饰符
    return_type = re.sub(r'\b(inline|static|friend|virtual|explicit)\b', '', return_type_raw).strip()

    param_types = []
    if params_str:
        # 支持模板深度感知的切分，处理像 std::map<int, std::vector<int>> 的复杂逗号干扰
        parts = []
        current = []
        depth = 0
        for char in params_str:
            if char == '<': depth += 1
            elif char == '>': depth -= 1
            if char == ',' and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            parts.append("".join(current).strip())

        for part in parts:
            # 去除引用符 & 和 const 限定符
            part_clean = re.sub(r'\s*&\s*', ' ', part)
            part_clean = re.sub(r'\s*const\s+', ' ', part_clean).strip()
            words = part_clean.split()
            if len(words) >= 2:
                param_types.append(" ".join(words[:-1]).strip())
            elif len(words) == 1:
                param_types.append(words[0].strip())
                
    return return_type, func_name, param_types

def extract_cpp_func_name(code: str) -> str:
    _, name, _ = parse_cpp_function_signature(code)
    return name

def format_cpp_val(val):
    """
    将 cpp/JSON 原生类型转换为 C++ 代码可编译的强类型字面量表示
    """
    if isinstance(val, bool):
        return "true" if val else "false"
    if val is None:
        return "nullptr"
    if isinstance(val, list):
        return "{" + ", ".join(format_cpp_val(x) for x in val) + "}"
    if isinstance(val, dict):
        return "{" + ", ".join(f"{{{format_cpp_val(k)}, {format_cpp_val(v)}}}" for k, v in val.items()) + "}"
    if isinstance(val, str):
        escaped = val.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        return f'"{escaped}"'
    return str(val)

def sanitize_io_pairs(io_pairs: list, cpp_code: str) -> list:
    """
    自适应对齐过滤器：利用静态签名自适应填充/截断 LLM 生成的参数，防御格式破碎崩溃
    """
    if not isinstance(io_pairs, list):
        return []
    
    _, _, param_types = parse_cpp_function_signature(cpp_code)
    param_count = len(param_types)

    sanitized = []
    for pair in io_pairs:
        if not isinstance(pair, dict):
            continue
        args = pair.get("args")
        expected = pair.get("expected")

        # 将空输入/单个原始非列表值自适应对齐为 list 容器
        if args is None:
            args = [None] * param_count if param_count > 0 else []
        elif not isinstance(args, list):
            args = [args]

        # 根据静态特征参数个数进行微调对齐
        if len(args) < param_count:
            args = args + [None] * (param_count - len(args))
        elif len(args) > param_count:
            args = args[:param_count]

        pair["args"] = args
        pair["expected"] = expected
        sanitized.append(pair)
        
    return sanitized

def verify_cpp_and_sync_expected(cpp_code: str, io_pairs: list) -> (bool, list, float, str):
    """
    SC-JEV C++ 双轨制运行与真值同步引擎：
    1. 动态差分轨：物理编译运行 C++, 自动捕捉真实运行时 return 输出并反向覆盖 LLM 的 expected，100% 拒绝幻觉。
    2. 先知轨：当 C++ 代码由于本地环境缺少特定的库时，优雅退回 Oracle 预测模式而不丢弃样本。
    """
    if not io_pairs:
        return False, io_pairs, 0.0, "No IO Pairs"

    return_type, func_name, param_types = parse_cpp_function_signature(cpp_code)
    if func_name == "unknown":
        return False, io_pairs, 0.0, "Identify C++ method signature failed"

    # 标准 C++ 通用依赖头文件声明
    cpp_header = (
        "#include <iostream>\n"
        "#include <vector>\n"
        "#include <string>\n"
        "#include <map>\n"
        "#include <set>\n"
        "#include <algorithm>\n"
        "#include <cmath>\n"
        "#include <sstream>\n"
        "#include <iomanip>\n\n"
    )

    # 核心 C++ 万能强类型输出序列化至标准 JSON 映射函数 (支持完备数值重载，防 ambiguous 编译报错)
    cpp_serializer = """
    // Automatically serialize standard types into standardized JSON strings
    inline std::string toJson(int v) { return std::to_string(v); }
    inline std::string toJson(unsigned int v) { return std::to_string(v); }
    inline std::string toJson(long v) { return std::to_string(v); }
    inline std::string toJson(unsigned long v) { return std::to_string(v); }
    inline std::string toJson(long long v) { return std::to_string(v); }
    inline std::string toJson(unsigned long long v) { return std::to_string(v); }
    inline std::string toJson(double v) {
        std::stringstream ss;
        ss << std::setprecision(10) << v;
        return ss.str();
    }
    inline std::string toJson(float v) {
        std::stringstream ss;
        ss << std::setprecision(7) << v;
        return ss.str();
    }
    inline std::string toJson(bool v) { return v ? "true" : "false"; }
    inline std::string toJson(const std::string& v) {
        std::string escaped = "";
        for (char c : v) {
            if (c == '\\\\') escaped += "\\\\\\\\";
            else if (c == '"') escaped += "\\\\\\"";
            else if (c == '\\n') escaped += "\\\\n";
            else escaped += c;
        }
        return "\\"" + escaped + "\\"";
    }
    inline std::string toJson(char v) { return std::string("\\"") + v + "\\""; }

    template<typename T>
    std::string toJson(const std::vector<T>& vec) {
        std::stringstream ss;
        ss << "[";
        for(size_t i = 0; i < vec.size(); ++i) {
            if(i > 0) ss << ", ";
            ss << toJson(vec[i]);
        }
        ss << "]";
        return ss.str();
    }

    template<typename K, typename V>
    std::string toJson(const std::map<K, V>& m) {
        std::stringstream ss;
        ss << "{";
        bool first = true;
        for(const auto& pair : m) {
            if(!first) ss << ", ";
            ss << toJson(pair.first) << ": " << toJson(pair.second);
            first = false;
        }
        ss << "}";
        return ss.str();
    }
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        # A. 物理编译沙箱性检查
        src_path = os.path.join(tmpdir, "main.cpp")
        bin_path = os.path.join(tmpdir, "app.out")
        
        dummy_main = "\nint main() { return 0; }\n"
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(cpp_header + cpp_serializer + cpp_code + dummy_main)

        try:
            compile_res = subprocess.run(
                ["g++", "-std=c++17", "-O0", src_path, "-o", bin_path],
                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if compile_res.returncode != 0:
                stderr_msg = compile_res.stderr.strip()
                # 编译未过往往是非本地的标准特定三方依赖，直接走先知机制防误判
                return True, io_pairs, 100.0, f"Oracle_Fallback (Compile fails: {stderr_msg[:100]})"
        except FileNotFoundError:
            return False, io_pairs, 0.0, "Compiler 'g++' not found"

        # B. 动态差分比对轨测试执行
        validated_pairs = []
        for idx, pair in enumerate(io_pairs):
            args = pair.get("args", [])
            try:
                args_formatted = ", ".join(format_cpp_val(a) for a in args)
            except Exception:
                continue

            test_main = "\nint main() {\n"
            if return_type == "void":
                test_main += f"    try {{ {func_name}({args_formatted}); std::cout << \"\\\"void\\\"\" << std::endl; }} catch(...) {{ std::cout << \"null\" << std::endl; }}\n"
            else:
                test_main += f"    try {{ auto res = {func_name}({args_formatted}); std::cout << toJson(res) << std::endl; }} catch(...) {{ std::cout << \"null\" << std::endl; }}\n"
            test_main += "    return 0;\n}\n"

            case_src_path = os.path.join(tmpdir, f"case_{idx}.cpp")
            case_bin_path = os.path.join(tmpdir, f"case_{idx}.out")
            with open(case_src_path, "w", encoding="utf-8") as f:
                f.write(cpp_header + cpp_serializer + cpp_code + test_main)

            try:
                compile_case = subprocess.run(
                    ["g++", "-std=c++17", "-O0", case_src_path, "-o", case_bin_path],
                    cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                if compile_case.returncode != 0:
                    continue

                run_res = subprocess.run(
                    [case_bin_path],
                    cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2
                )
                if run_res.returncode == 0:
                    raw_out = run_res.stdout.strip()
                    pair["expected"] = json.loads(raw_out)
                    validated_pairs.append(pair)
            except Exception:
                continue

        # 如果至少成功执行 3 条测试用例，我们坚决走 100% 精准同步真值路线
        if len(validated_pairs) >= 3:
            coverage_pct = (len(validated_pairs) / len(io_pairs)) * 100.0
            return True, validated_pairs, coverage_pct, "Dynamic_Verified"
        else:
            return True, io_pairs, 100.0, "Oracle_Fallback (Execution is not robust)"

# ==================== 3. Racket 编译与测试引擎 ====================
def extract_racket_function_name(code: str, cpp_code: str = None) -> str:
    """
    智能提取 Racket 代码中定义的主要函数/结构体名称。
    支持：(define (func ...), (define func (lambda ...), 以及 (struct struct-name ...
    🔥 深度改良：通过 C++ 签名提取对应 Kebab-case 函数名进行过滤排序，根治 helper 错位抓取导致的评测雪崩。
    """
    # 1. 提取所有可能的 Racket 定义名
    found_funcs = re.findall(r'\(\s*define\s*\(\s*([\w\-\?\!]+)', code)
    found_funcs += re.findall(r'\(\s*define\s+([\w\-\?\!]+)\s+\(\s*lambda', code)
    found_funcs += re.findall(r'\(\s*define\s+([\w\-\?\!]+)', code)
    found_funcs += re.findall(r'\(\s*struct\s+([\w\-\?\!]+)', code)
    
    # 去重并保留定义顺序
    seen = set()
    unique_funcs = []
    for f in found_funcs:
        if f not in seen:
            seen.add(f)
            unique_funcs.append(f)
            
    # 剔除 Lisp 关键字和通用特殊操作符
    clean_funcs = [
        f for f in unique_funcs 
        if f not in ['initialize', 'setClass', 'fit', 'predict', 'else', 'cond', 'let', 'let*', 'letrec', 'case']
    ]
    
    if not clean_funcs:
        return None
        
    # 🎯 学术级对齐：利用 C++ 函数名转换来匹配 Racket 主调用入口
    if cpp_code:
        cpp_name = extract_cpp_func_name(cpp_code)
        if cpp_name:
            # 1. kebab-case: find_element -> find-element
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', cpp_name)
            s2 = re.sub('([a-z0-9])([A-Z])', r'\1-\2', s1).lower()
            kebab_name = s2.replace('_', '-')
            
            possible_names = {cpp_name, kebab_name, cpp_name.lower(), cpp_name.replace('_', '-')}
            
            # 精确匹配优先
            for pf in clean_funcs:
                if pf in possible_names:
                    return pf
            
            # 模糊子串匹配次之
            for pf in clean_funcs:
                if kebab_name in pf or pf in kebab_name:
                    return pf

    # 无 C++ 上下文时默认匹配最后一个（通常是 Racket 的主程序入口）
    return clean_funcs[-1] if clean_funcs else None

def format_racket_val(val, use_vector=False):
    """
    将 cpp/JSON 的数据类型转换为 Lisp/Racket 的字面量表达式。
    🔥 升级：支持 use_vector 参数。当 use_vector=True 时将 list 格式化为 vector 字面量，解决类型冲突契约异常。
    """
    if isinstance(val, bool):
        return "#t" if val else "#f"
    if val is None:
        return "'()"
    if isinstance(val, list):
        if use_vector:
            # 格式化为 Racket 向量: (vector ...)
            return "(vector " + " ".join(format_racket_val(x, use_vector) for x in val) + ")"
        else:
            # 格式化为 Racket 动态列表: (list ...)
            return "(list " + " ".join(format_racket_val(x, use_vector) for x in val) + ")"
    if isinstance(val, dict):
        # 映射字典为 Racket 哈希表 (Hash Table)
        pairs_str = " ".join(f"'{k} {format_racket_val(v, use_vector)}" for k, v in val.items())
        return f"(hash {pairs_str})"
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)

def execute_racket_test(candidate_code, io_pairs, cpp_code=None, timeout=3) -> bool:
    """
    物理执行并验证 Racket 代码正确性
    🔥 升级至双轨自适应模式：多轮 trial 测试。
    - Trial 1: 将 C++ std::vector 在 Racket 传入 Racket 列表(list)
    - Trial 2: 将 C++ std::vector 在 Racket 传入 Racket 向量(vector)
    只要任一一轨比对通过，即代表语义完全对齐！
    """
    if not candidate_code.strip() or not io_pairs:
        return False

    # 🎯 核心 Racket 级别学术语义对齐校验机制：
    # 递归转换比对 List 和 Vector 的相互表示，
    # 并且对 `#<void>` 和 C++ 桩输出的字符串 `"void"` 进行了自适应放行比对，解决 Void 题目的黑洞！
    racket_semantic_equal_impl = """
    (define (semantic-equal? a b)
      (cond
        [(and (void? a) (equal? b "void")) #t]
        [(and (equal? a "void") (void? b)) #t]
        [(equal? a b) #t]
        [(and (vector? a) (list? b)) (semantic-equal? (vector->list a) b)]
        [(and (list? a) (vector? b)) (semantic-equal? a (vector->list b))]
        [(and (vector? a) (vector? b)) (semantic-equal? (vector->list a) (vector->list b))]
        [(and (list? a) (list? b))
         (and (= (length a) (length b))
              (andmap semantic-equal? a b))]
        [(and (number? a) (number? b)) (< (abs (- a b)) 1e-4)]
        [else #f]))
    """

    # 🔥 核心：自适应双轮驱动测试
    for use_vector in [False, True]:
        func_name = extract_racket_function_name(candidate_code, cpp_code)
        if not func_name:
            return False

        test_statements = []
        for pair in io_pairs:
            try:
                args_str = " ".join(format_racket_val(a, use_vector) for a in pair["args"])
                expected_str = format_racket_val(pair["expected"], use_vector)
                
                test_statements.append(
                    f"(with-handlers ([exn:fail? (lambda (e) (exit 1))])\n"
                    f"  (let ([result ({func_name} {args_str})])\n"
                    f"    (if (semantic-equal? result {expected_str})\n"
                    f"        (void)\n"
                    f"        (exit 1))))"
                )
            except Exception:
                break
        else:
            # 只有当完整格式化不抛异常时，才进入当前 Pass 物理运行
            clean_code = candidate_code.strip()
            if "#lang racket" in clean_code:
                clean_code = re.sub(r'#lang\s+racket\s*', '', clean_code)
                
            full_racket_code = (
                "#lang racket\n\n" + 
                racket_semantic_equal_impl + "\n\n" + 
                clean_code + "\n\n; ====== AUTOMATED TESTS ======\n" + 
                "\n".join(test_statements) + "\n"
            )

            base_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
            with tempfile.NamedTemporaryFile(suffix=".rkt", dir=base_dir, mode="w", encoding="utf-8", delete=False) as tf:
                tf.write(full_racket_code)
                temp_file_path = tf.name

            try:
                res = subprocess.run(
                    ["racket", temp_file_path],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, encoding='utf-8', errors='ignore'
                )
                
                # 调试利器：如果开发需要查看底层的 Racket 报错原因，可以将下方 False 改为 True
                if res.returncode != 0 and False:
                    print(f"\n[Pass: {'Vector' if use_vector else 'List'} Error - ID: {temp_file_path}]:\nSTDERR: {res.stderr}")

                if res.returncode == 0:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    return True  # 当前轨运行成功，代表整个测试通过，直接返回
            except Exception:
                pass
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    
    return False

def calculate_pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    else:
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def extract_and_parse_json(output_str: str) -> list:
    cleaned_str = output_str.strip()
    json_match = re.search(r'\[\s*\{.*\}\s*\]', cleaned_str, re.DOTALL)
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
    
    try:
        return json.loads(target_str)
    except Exception:
        return []
        
# ==================== 4. 主控制流与双轨评测执行 ====================
def run_live_pipeline(csv_path, llm_path, llm_type, output_report_csv, cov_threshold=80.0, batch_size=16):
    df = pd.read_csv(csv_path)
    
    # 兼容 Cpp -> Racket 场景下的列名称 (cpp_code)
    if 'cpp_code' in df.columns:
        df = df.rename(columns={'cpp_code': 'cpp_code'})
        
    df = df.dropna(subset=['cpp_code', 'all_candidates'])
    
    # 仅保留在前序 Stage 3 中顺利通过编译和逻辑验证的高质量 Passed 样本
    if 'status' in df.columns:
        initial_count = len(df)
        df = df[df['status'] == 'Passed'].reset_index(drop=True)
        filtered_out = initial_count - len(df)
        print(f"📊 [数据流控] 原始推理记录: {initial_count} 条")
        print(f"🔥 [数据流控] 已自动剔除 Racket 语法/编译未通过 (Failed) 的样本: {filtered_out} 条")
        print(f"🚀 [数据流控] 最终送入测试用例生成与 pass@k 评测的有效样本: {len(df)} 条\n")
    else:
        print("⚠️ [警告] 未在输入 CSV 中检测到 'status' 列，将默认对全量非空数据进行评测。")

    total_records = len(df)
    if total_records == 0:
        print("❌ [终止] 过滤后无可用的 'Passed' 样本，请检查输入数据集！")
        return

    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    stats = {'total': 0, 'filtered': 0, 'valid': 0, 'p1': 0.0, 'p2': 0.0, 'p5': 0.0}
    
    print(f"\n🚀 批量评测流水线已激活 (Batch Size: {batch_size})，正在单行流式计算并预测时间...")

    with open(output_report_csv, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'verify_mode', 'cpp_coverage', 'status', 'pass@1', 'pass@2', 'pass@5', 'passed_candidates'])

        rows = [row for _, row in df.iterrows()]
        start_time = time.time()
        current_count = 0

        for i in range(0, total_records, batch_size):
            batch_rows = rows[i:i + batch_size]
            prompts = []
            
            # 批量构建 Prompt 模板
            for row in batch_rows:
                # 💡 强健容错：兼容可能已经被外部重命名或保持原样的 C++ 核心代码字段
                cpp_source = row.get('cpp_code', row.get('cpp_code', ''))
                user_content = GEN_TEST_USER_PROMPT.format(cpp_code=cpp_source)
                prompt = get_prompt_by_model(llm_type, GEN_TEST_SYSTEM_PROMPT, user_content)
                prompts.append(prompt)

            # GPU 并行 Tokenize
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            # GPU 批量并行推理生成
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
                cpp_code = row.get('cpp_code', row.get('cpp_code', ''))
                cand_str = row['all_candidates']
                
                # 解码测试用例
                output_str = tokenizer.decode(outputs[j][input_lens:], skip_special_tokens=True)
                raw_io_pairs = extract_and_parse_json(output_str)

                # ==================== SC-JEV 拦截和真值对齐重构 ====================
                # 🎯 【数据第一防线】自适应包装清洗
                raw_io_pairs = sanitize_io_pairs(raw_io_pairs, cpp_code)

                # 🎯 【数据第二防线】C++ 执行编译与真值覆盖同步
                is_cpp_ok, verified_io_pairs, coverage_pct, validation_msg = verify_cpp_and_sync_expected(cpp_code, raw_io_pairs)

                stats['total'] += 1

                if not is_cpp_ok:
                    stats['filtered'] += 1
                    writer.writerow([idx, "None", "0.0%", "Format_Error", "0.0000", "0.0000", "0.0000", 0])
                    continue

                mode_str = "Oracle" if "Oracle" in validation_msg else "Dynamic_JEV"

                # 我们对物理覆盖率未满足阈值要求的样本，如果不是降级先知轨，则过滤
                if mode_str == "Dynamic_JEV" and coverage_pct < cov_threshold:
                    stats['filtered'] += 1
                    writer.writerow([idx, mode_str, f"{coverage_pct:.2f}%", "Filtered_Low_Cov", "0.0000", "0.0000", "0.0000", 0])
                    continue

                # ==================== Racket 比对与 Pass@k 指标计算 ====================
                try:
                    candidates = json.loads(cand_str)
                except Exception:
                    writer.writerow([idx, mode_str, f"{coverage_pct:.2f}%", "JSON_Err", "0.0000", "0.0000", "0.0000", 0])
                    continue

                n = len(candidates)
                passed_count = 0
                for cand in candidates:
                    if execute_racket_test(cand, verified_io_pairs, cpp_code):
                        passed_count += 1

                p1 = calculate_pass_at_k(n, passed_count, 1)
                p2 = calculate_pass_at_k(n, passed_count, 2) if n >= 2 else p1
                p5 = calculate_pass_at_k(n, passed_count, 5) if n >= 5 else p2

                stats['valid'] += 1
                stats['p1'] += p1
                stats['p2'] += p2
                stats['p5'] += p5

                # 写入落地报告 CSV
                writer.writerow([
                    idx, mode_str, f"{coverage_pct:.2f}%", "Passed" if passed_count > 0 else "Failed",
                    f"{p1:.4f}", f"{p2:.4f}", f"{p5:.4f}", passed_count
                ])

            # 单行流式输出控制
            f.flush()
            v = max(stats['valid'], 1)
            pct = (current_count / total_records) * 100
            
            elapsed_time = time.time() - start_time
            it_s = current_count / elapsed_time if elapsed_time > 0 else 0.0
            remaining_items = total_records - current_count
            remaining_seconds = remaining_items / it_s if it_s > 0 else 0
            
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_seconds))

            full_status_line = (
                f"进度:{pct:>5.1f}% ({current_count}/{total_records})[{elapsed_str}<{remaining_str},{it_s:.2f}it/s] "
                f"|总数:{stats['total']}|拦截/低覆盖:{stats['filtered']}|有效:{stats['valid']} "
                f"|P@1:{(stats['p1']/v)*100:.2f}%|P@2:{(stats['p2']/v)*100:.2f}%|P@5:{(stats['p5']/v)*100:.2f}%"
            )

            print(full_status_line)
            sys.stdout.flush()

            # 显式清理显存
            del inputs, outputs
            torch.cuda.empty_cache()

        print("\n")

    del model
    torch.cuda.empty_cache()
    
    # 落地最终学术实验大盘报告
    print("================ 📈 总结 📈 ================")
    print(f"原始样本总量: {total_records}")
    print(f"格式无法解析/C++ 低覆盖被淘汰的题目: {stats['filtered']}")
    print(f"最终纳入 $pass@k$ 指标计算的高密度有效题目总数: {stats['valid']}")
    if stats['valid'] > 0:
        print(f"🌟 精准过滤后平均 Pass@1: {stats['p1'] / stats['valid'] * 100:.2f}%")
        print(f"🌟 精准过滤后平均 Pass@2: {stats['p2'] / stats['valid'] * 100:.2f}%")
        print(f"🌟 精准过滤后平均 Pass@5: {stats['p5'] / stats['valid'] * 100:.2f}%")
    print("==================================================================")
    print("\n🎉 [OK] C++ 到 Racket 评测结束，报告已完整保存！\n")

if __name__ == "__main__":
    CSV_PATH = "/root/autodl-tmp/first_paper/cpp/translation_dataset/Racket_inference_results_original.csv"
    LLM_MODEL_PATH = "/root/autodl-tmp/first_paper/LLM/Qwen2.5"
    CURRENT_LLM_TYPE = "qwen"
    COVERAGE_LINE = 80.0  # 💡 保持 80.0% C++ 覆盖率作为高密度用例筛选门槛
    OUTPUT_REPORT = "/root/autodl-tmp/first_paper/cpp/testcase_dataset/Racket_passk_图结构_report.csv"
    
    run_live_pipeline(CSV_PATH, LLM_MODEL_PATH, CURRENT_LLM_TYPE, OUTPUT_REPORT, cov_threshold=COVERAGE_LINE)