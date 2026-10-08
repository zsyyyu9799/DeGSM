import os
import re
import csv
import json
import math
import tempfile
import subprocess
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
import torch
import sys  # 控制标准输出
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== 全局配置 ====================
COVERAGE_MODE = "jacoco" 

# 🛠️ JaCoCo 工具包存储目录
JACOCO_DIR = "/root/autodl-tmp/jacoco_tools"

# R 语言内置标准系统包
BUILTIN_R_PACKAGES = {
    'base', 'compiler', 'datasets', 'graphics', 'grDevices', 'grid', 'methods', 
    'parallel', 'splines', 'stats', 'stats4', 'tcltk', 'tools', 'utils'
}

# ==================== 0. JaCoCo 工具下载自愈模块 ====================
def download_file(url, dest_path):
    """带重试机制的网络文件下载器"""
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, dest_path)
            return True
        except Exception as e:
            time.sleep(2)
    return False

def ensure_jacoco_tools():
    """自愈式下载 JaCoCo Agent 和 CLI 工具包"""
    os.makedirs(JACOCO_DIR, exist_ok=True)
    agent_path = os.path.join(JACOCO_DIR, "jacocoagent.jar")
    cli_path = os.path.join(JACOCO_DIR, "jacococli.jar")
    
    agent_url = "https://repo1.maven.org/maven2/org/jacoco/org.jacoco.agent/0.8.11/org.jacoco.agent-0.8.11-runtime.jar"
    cli_url = "https://repo1.maven.org/maven2/org/jacoco/org.jacoco.cli/0.8.11/org.jacoco.cli-0.8.11-nodeps.jar"
    
    if not os.path.exists(agent_path):
        print(f"📥 [环境自愈] 正在从 Maven 中央仓库下载 JaCoCo Runtime Agent...")
        download_file(agent_url, agent_path)
            
    if not os.path.exists(cli_path):
        print(f"📥 [环境自愈] 正在从 Maven 中央仓库下载 JaCoCo CLI...")
        download_file(cli_url, cli_path)
            
    return agent_path, cli_path

# ==================== 1. 提示词与 Prompt 路由 ====================
GEN_TEST_SYSTEM_PROMPT = (
    "You are an expert QA Engineer and a rigorous automated script. "
    "Your unique task is to analyze Java source code and synthesize highly precise native testing suites. "
    "You must output ONLY the native test code enclosed in standard markdown code blocks."
)

# 🎯 第一阶段：专门针对 Java 生成测试用例
GEN_JAVA_TEST_USER_PROMPT = """You are presented with a Java class. Your goal is to construct a rigorous native Java test suite comprising EXACTLY 5 diverse and distinct test cases to thoroughly evaluate its correctness.

[Testing Objectives]
1. Diversity: Cover different logical paths, boundary conditions, and potential exception zones of the source code.
2. Isolated Reporting: To ensure we can diagnose each test case separately, you MUST wrap each of the 5 test cases in an individual try-catch block inside the main method and print its status using the exact format:
   `System.out.println("TEST_1: PASSED");` or `System.out.println("TEST_1: FAILED - " + e.toString());` (using TEST_1, TEST_2, TEST_3, TEST_4, TEST_5 respectively).
3. Summary output: At the very end of the main method, print: `System.out.println("TOTAL_PASSED: " + passedCount);` where passedCount is the integer of passed tests.
4. Self-Containment: Write a complete Java class named `{class_name}Test` with a `public static void main(String[] args)` method. Do NOT import JUnit. Use standard java assertions or comparisons.

[Output Format Specification]
You must output ONLY standard markdown code blocks for Java. Do NOT output any conversational text.

```java
public class {class_name}Test {{
    public static void main(String[] args) {{
        int passed = 0;
        // Test 1
        try {{
            // Write case 1 logic here
            System.out.println("TEST_1: PASSED");
            passed++;
        }} catch (Throwable e) {{
            System.out.println("TEST_1: FAILED - " + e.toString());
        }}
        
        // Test 2
        // ...
        // Test 5
        
        System.out.println("TOTAL_PASSED: " + passed);
    }}
}}
```

[Actual Task Input]
Input Java Code:
```java
{java_code}
```

Expected Output:
"""

# 🎯 第二阶段：根据 Java 测试的执行反馈，迁移生成 R 语言测试
GEN_R_TEST_USER_PROMPT = """You are presented with a Java class, its executed Java test suite, and the actual execution feedback logs.
Your goal is to construct a matching R test suite containing EXACTLY 5 test cases targeting the R function `{r_func_name}`.

[R Testing Strategy]
1. Map correct behaviors: Translate the semantically correct behaviors of the Java test cases (especially the ones that PASSED in Java) into R code.
2. Learn from Failures: For any Java test cases that FAILED, analyze their failure logs. If they failed due to wrong logical assumptions in Java tests, fix the logic when translating to R. If they failed because of boundary limit exceptions, verify that R asserts those bounds correctly.
3. **CRITICAL: Use the exact function name `{r_func_name}` in your test calls. Do NOT use placeholders like `solution`, `FUNC`, or `target_function`. The function `{r_func_name}` has already been defined in the environment.**
4. Use `stopifnot()` or custom error stops on failure. Assume the candidate R code has been sourced in the environment.

[Output Format Specification]
You must output ONLY standard markdown code blocks for R. Do NOT output any conversational text.

```R
# Name: test_solution.R
# Write exactly 5 test cases calling the function: {r_func_name}
# Use stopifnot() or custom stop() messages on mismatch
```

[Execution Feedback of Java Tests]
{java_feedback}

Original Java Code:
```java
{java_code}
```

Java Test Suite Code:
```java
{java_test_code}
```

Target R Function: `{r_func_name}`

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

# ==================== 2. 工具辅助函数 ====================
def extract_java_class_name(java_code: str) -> str:
    code_clean = re.sub(r'//.*', '', java_code)
    code_clean = re.sub(r'/\*.*?\*/', '', code_clean, flags=re.DOTALL)
    match_public = re.search(r'\bpublic\s+class\s+(\w+)', code_clean)
    if match_public:
        return match_public.group(1)
    matches = re.findall(r'\bclass\s+(\w+)', code_clean)
    if not matches:
        return "Solution"
    return "Solution" if "Solution" in matches else matches[0]

def extract_r_function_name(code: str) -> str:
    """
    精确提取 R 代码中的函数名。
    修复：去除了默认返回 "solution" 的硬编码设定，避免因未能提取到正确函数名时返回错误默认值，导致覆盖率统计完全失效。
    """
    if not code or not isinstance(code, str):
        return ""
        
    # 过滤单行注释以防误配
    code_clean = re.sub(r'#.*', '', code)
    
    # 支持字母、数字、点(.)及下划线(_)命名的 R 函数
    found_funcs = re.findall(r'^\s*([\w\._]+)\s*(?:<-|=)\s*function', code_clean, re.MULTILINE)
    if not found_funcs:
        found_funcs = re.findall(r'([\w\._]+)\s*(?:<-|=)\s*function', code_clean)
        
    if not found_funcs:
        return ""
        
    # 剔除常见的非评测主目标辅助函数
    clean_funcs = [f for f in found_funcs if f not in ['initialize', 'setClass', 'fit', 'predict', 'show', 'print']]
    return clean_funcs[0] if clean_funcs else found_funcs[0]

def align_test_with_candidate_func(candidate_code: str, r_test_code: str) -> str:
    """
    动态命名对齐：确保 R 测试用例中调用的核心函数名称与候选解答（Candidate）中的实际函数名完全一致。
    如果是硬编码的 "solution" 或 "sol"，将其动态替换为准确提取出的实际函数名。
    """
    cand_func = extract_r_function_name(candidate_code)
    if not cand_func:
        return r_test_code
        
    # 如果候选代码里包含具体的自定义函数名，但大模型生成的测试套件却误写为 'solution' 或 'sol'
    if cand_func not in r_test_code:
        r_test_code = re.sub(r'\bsolution\b', cand_func, r_test_code)
        r_test_code = re.sub(r'\bsol\b', cand_func, r_test_code)
        
    return r_test_code

def extract_packages_from_r_code(code: str) -> set:
    packages = set()
    if not code or not isinstance(code, str):
        return packages
    lib_pattern = r'library\s*\(\s*(?:"([^"]+)"|\'([^\'\s]+)\'|([\w\.]+))\s*\)'
    for m in re.finditer(lib_pattern, code):
        pkg = m.group(1) or m.group(2) or m.group(3)
        if pkg: packages.add(pkg.strip())
    return packages

# ==================== 3. Java 本地原生执行与覆盖率分析 ====================
def parse_jacoco_xml(xml_path, target_class_name):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        target_class_node = None
        for cls in root.findall('.//class'):
            name_attr = cls.attrib.get('name', '')
            if name_attr == target_class_name or name_attr.endswith('/' + target_class_name):
                target_class_node = cls
                break
        
        node_to_parse = target_class_node if target_class_node is not None else root
        line_cov, branch_cov = 100.0, 100.0
        has_branch = False
        for c in node_to_parse.findall('./counter'):
            c_type = c.attrib.get('type')
            missed = int(c.attrib.get('missed', 0))
            covered = int(c.attrib.get('covered', 0))
            total = missed + covered
            if total > 0:
                pct = (covered / total) * 100.0
                if c_type == 'LINE': line_cov = pct
                elif c_type == 'BRANCH':
                    branch_cov = pct
                    has_branch = True
        if not has_branch: branch_cov = 100.0
        return line_cov, branch_cov
    except Exception:
        return 0.0, 0.0

def verify_java_native_test_suite(java_code: str, java_test_code: str, jacoco_agent_path: str, jacoco_cli_path: str) -> (bool, int, str, float, float):
    """
    Java 本地原生多用例测试执行与解析器：
    编译运行生成的 try-catch 隔离测试类，精准解析捕获 TEST_1 ~ TEST_5 的真实成功状态与崩溃反馈。
    """
    if not java_test_code.strip():
        return False, 0, "No Java test code generated.", 0.0, 0.0

    java_code = re.sub(r'^\s*package\s+[\w\.]+;\s*', '', java_code, flags=re.MULTILINE)
    class_name = extract_java_class_name(java_code)
    test_class_name = f"{class_name}Test"

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, f"{class_name}.java"), "w", encoding="utf-8") as f:
            f.write(java_code)
        with open(os.path.join(tmpdir, f"{test_class_name}.java"), "w", encoding="utf-8") as f:
            f.write(java_test_code)

        shutil.copy(jacoco_agent_path, os.path.join(tmpdir, "jacocoagent.jar"))
        shutil.copy(jacoco_cli_path, os.path.join(tmpdir, "jacococli.jar"))

        # 编译
        try:
            compile_res = subprocess.run(
                ["javac", f"{class_name}.java", f"{test_class_name}.java"], 
                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10
            )
            if compile_res.returncode != 0:
                return False, 0, f"Java compilation failed:\n{compile_res.stderr}", 0.0, 0.0
        except subprocess.TimeoutExpired:
            return False, 0, "Java compilation timed out (possible compiler hang).", 0.0, 0.0

        # 执行原生测试（JaCoCo 字节码插桩追踪，增加 TimeoutExpired 的高容错防御）
        try:
            run_res = subprocess.run(
                ["java", f"-javaagent:jacocoagent.jar=destfile=jacoco.exec,includes={class_name}", "-cp", ".", test_class_name],
                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5
            )
            stdout_str = run_res.stdout
            stderr_str = run_res.stderr
        except subprocess.TimeoutExpired:
            # 🛡️ 发生死循环或挂起，优雅拦截，防止脚本中途中断崩溃！
            return False, 0, "Java execution timed out (possible infinite loop or blocked IO).", 0.0, 0.0

        # 🎯 精准解析多用例输出及通过率
        passed_count = 0
        feedback_lines = []
        
        status_matches = re.findall(r'(TEST_\d+:\s*(?:PASSED|FAILED.*?))(?:\n|$)', stdout_str)
        total_passed_match = re.search(r'TOTAL_PASSED:\s*(\d+)', stdout_str)
        
        if total_passed_match:
            passed_count = int(total_passed_match.group(1))
        else:
            passed_count = sum(1 for m in status_matches if "PASSED" in m)
            
        for line in stdout_str.split('\n'):
            if any(p in line for p in ["TEST_1", "TEST_2", "TEST_3", "TEST_4", "TEST_5", "TOTAL_PASSED"]):
                feedback_lines.append(line.strip())
                
        detailed_feedback = "\n".join(feedback_lines)
        if not detailed_feedback:
            detailed_feedback = f"Execution finished with exit code {run_res.returncode}.\nOutput:\n{stdout_str[:300]}\nError:\n{stderr_str[:300]}"

        # 只要有至少 1 个测试用例通过，就说明 Java 源码 and 对应的一部分用例是合法的
        success_at_least_one = (passed_count >= 1)

        # 生成 JaCoCo 覆盖率报告
        java_line_cov, java_branch_cov = 100.0, 100.0
        if success_at_least_one and os.path.exists(os.path.join(tmpdir, "jacoco.exec")):
            try:
                subprocess.run(
                    ["java", "-jar", "jacococli.jar", "report", "jacoco.exec", "--classfiles", ".", "--xml", "jacoco.xml"],
                    cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5
                )
                xml_path = os.path.join(tmpdir, "jacoco.xml")
                if os.path.exists(xml_path):
                    java_line_cov, java_branch_cov = parse_jacoco_xml(xml_path, class_name)
            except Exception:
                pass

        return success_at_least_one, passed_count, detailed_feedback, java_line_cov, java_branch_cov

# ==================== 4. R 本地原生执行与覆盖率分析 ====================
def clean_r_test_code(r_test_code: str) -> str:
    """
    清理 R 测试代码，去除任何可能导致执行中断或覆盖率失效的 source() 语句。
    1. 避免在测试执行时因找不到 'solution.R' 等硬编码文件名而崩溃。
    2. 避免在 covr 计算覆盖率时重新 source() 导致插桩代码被覆盖，产生 'no rows to aggregate' 错误。
    """
    if not r_test_code:
        return ""
    # 支持最多 3 层嵌套括号的 source() / sys.source() 匹配清理，确保不留下多余的括号
    pattern = r'(?i)\b(?:sys\.)?source\s*\((?:[^()]*|\((?:[^()]*|\([^()]*\))*\))*\)'
    cleaned = re.sub(pattern, '# [source call removed for safety]', r_test_code)
    return cleaned

def check_r_test_syntax(r_test_code: str) -> bool:
    """检查生成的 R 测试代码是否存在 R 语法错误"""
    if not r_test_code.strip():
        return False
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test_solution.R")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(r_test_code)
        try:
            res = subprocess.run(
                ["Rscript", "-e", "parse(file='test_solution.R')"],
                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3
            )
            return res.returncode == 0
        except Exception:
            return False

def verify_r_candidate(candidate_code: str, r_test_code: str, timeout=4) -> str:
    """
    检查单个 R 候选解答的执行状态并分类。
    返回状态:
      - "Success": 顺利通过所有测试用例并且退出码为 0
      - "Syntax_Error": 候选代码存在 R 语法错误，根本无法被 R解析执行
      - "Runtime_Error": 代码能解析，但运行期崩溃或断言失败 (如 stop/stopifnot 触发)
      - "Timeout": 运行超时 (可能存在死循环或锁死)
      - "Execution_Error": 其他不可预知的系统执行异常
    """
    if not candidate_code.strip():
        return "Empty_Code"

    # 执行断言前，去除测试代码内部一切硬编码的 source 干扰
    r_test_code = clean_r_test_code(r_test_code)

    with tempfile.TemporaryDirectory() as tmpdir:
        cand_path = os.path.join(tmpdir, "candidate.R")
        test_path = os.path.join(tmpdir, "test_solution.R")
        with open(cand_path, "w", encoding="utf-8") as f:
            f.write(candidate_code)
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(r_test_code)

        # 1. 静态语法解析检查
        try:
            syntax_res = subprocess.run(
                ["Rscript", "-e", "parse(file='candidate.R')"],
                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3
            )
            if syntax_res.returncode != 0:
                return "Syntax_Error"
        except subprocess.TimeoutExpired:
            return "Timeout"
        except Exception:
            return "Execution_Error"

        # 2. 物理运行与断言校验
        run_script = 'source("candidate.R"); source("test_solution.R")'
        try:
            res = subprocess.run(
                ["Rscript", "-e", run_script],
                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout
            )
            if res.returncode == 0:
                return "Success"
            else:
                return "Runtime_Error"
        except subprocess.TimeoutExpired:
            return "Timeout"
        except Exception:
            return "Execution_Error"

def calculate_r_native_coverage(candidate_code: str,
                                r_test_code: str) -> float:
    """
    计算 R 代码的行覆盖率（Line Coverage），返回单一浮点数。
    """
    r_test_code = align_test_with_candidate_func(candidate_code, r_test_code)
    func_name = extract_r_function_name(candidate_code)
    if not func_name:
        return 0.0

    # 1. 增强占位符替换（保留原有逻辑）
    placeholder_patterns = [
        r'\bsolution\b', r'\bFUNC\b', r'\btarget_function\b',
        r'\bfunc\b', r'\bmy_function\b'
    ]
    for pat in placeholder_patterns:
        r_test_code = re.sub(pat, func_name, r_test_code)

    # 2. 检查替换后是否包含函数调用
    if not re.search(rf'{re.escape(func_name)}\s*\(', r_test_code):
        print(f"[覆盖率警告] 测试代码中未检测到对 '{func_name}' 的调用，覆盖率可能为0")
        return 0.0

    if not candidate_code.strip() or not r_test_code.strip():
        return 0.0

    # 3. 清理测试代码中的 source() 语句
    r_test_code = clean_r_test_code(r_test_code)

    with tempfile.TemporaryDirectory() as tmpdir:
        candidate_path = os.path.join(tmpdir, "candidate.R")
        test_path = os.path.join(tmpdir, "test_solution.R")
        runner_path = os.path.join(tmpdir, "cov_runner.R")

        with open(candidate_path, "w", encoding="utf-8") as f:
            f.write(candidate_code)
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(r_test_code)

        # 4. 修改后的 covr 脚本：只计算行覆盖率
        cov_runner_src = r'''
        library(covr)
        result_line <- 0.0
        tryCatch({
            cov <- file_coverage(
                source_files = "candidate.R",
                test_files   = "test_solution.R"
            )
            if (!is.null(cov) && length(cov) > 0) {
                line_tbl <- tally_coverage(cov, by = "line")
                if (!is.null(line_tbl) && is.data.frame(line_tbl) && nrow(line_tbl) > 0) {
                    total_lines <- nrow(line_tbl)
                    covered_lines <- sum(line_tbl$value > 0)
                    result_line <- (covered_lines / total_lines) * 100.0
                }
            }
            if (is.na(result_line) || is.nan(result_line)) result_line <- 0.0
            cat(sprintf("RESULT:%.2f", result_line))
        }, error = function(e) {
            cat("ERROR_BEGIN\n")
            cat(conditionMessage(e))
            cat("\nERROR_END\n")
            cat("RESULT:0.00")
        })
        '''
        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(cov_runner_src)

        try:
            res = subprocess.run(
                ["Rscript", "cov_runner.R"],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20
            )
            if "ERROR_BEGIN" in res.stdout:
                print("\n========== COVR ERROR ==========")
                print(res.stdout)
                print("\n========== STDERR ==========")
                print(res.stderr)
                return 0.0

            match = re.search(r"RESULT:([\d\.]+)", res.stdout)
            if match:
                line_cov = float(match.group(1))
                return line_cov
            else:
                print("\n========== COVR PARSE FAIL ==========")
                print(res.stdout)
                print(res.stderr)
                return 0.0
        except subprocess.TimeoutExpired:
            print("\n[COVR TIMEOUT]\n")
            return 0.0
        except Exception as e:
            print(f"\n[COVR PYTHON ERROR] {e}\n")
            return 0.0
    return 0.0

def calculate_pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    else:
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)

# ==================== 5. 黄金数据集智能流控模块 ====================
def load_golden_data(file_path):
    """
    智能兼容 CSV 与 JSONL 双格式数据加载器。
    解决因配置 CSV 数据路径，但数据读取端强制按 JSONL 每行解析导致的崩溃幻觉。
    同时完美移植过滤逻辑：只将字段 'status' 为 'Passed' 的数据送入测试，规避脏数据干扰。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 找不到黄金数据集文件: {file_path}")

    print(f"🔄 [数据流控] 正在分析数据源格式并加载黄金子集: {file_path}")
    
    # 动态检测后缀
    if file_path.endswith('.csv'):
        try:
            df_clean = pd.read_csv(file_path)
            print("📁 [格式自检测] 已识别为标准 CSV 格式，已调用 pandas 进行抗噪读取。")
        except Exception as e:
            raise ValueError(f"❌ 解析 CSV 黄金文件失败，可能包含不规则换行或坏字符: {e}")
    else:
        # 兼容原 JSONL 降级读取
        print("📁 [格式自检测] 已识别为标准 JSONL 格式，已调用流式 JSON 逐行解码。")
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data.append(json.loads(line_str))
                except Exception as e:
                    print(f"⚠️  [行解析警告] 第 {line_num} 行 JSON 格式损坏并已被自动跳过: {e}")
        df_clean = pd.DataFrame(data)

    total_loaded = len(df_clean)
    if total_loaded == 0:
        return df_clean

    required_cols = {'java_code', 'all_candidates'}
    missing_cols = required_cols - set(df_clean.columns)
    if missing_cols:
        raise KeyError(f"❌ 黄金数据集字段缺失，请确认是否包含 {required_cols} 列。")

    df_clean = df_clean.dropna(subset=['java_code', 'all_candidates']).copy()
    df_clean['idx'] = df_clean['idx'].astype(int)

    # 🎯 核心修复：移植 generate_and_test (3).py 的 Passed 样本核心过滤逻辑
    if 'status' in df_clean.columns:
        initial_count = len(df_clean)
        # 精准筛选出基础翻译测试正常 Passed 的黄金样本
        df_clean = df_clean[df_clean['status'].astype(str).str.strip() == 'Passed'].reset_index(drop=True)
        filtered_out = initial_count - len(df_clean)
        print(f"🔥 [数据过滤成功] 原始加载记录总数: {initial_count} 条")
        print(f"🔥 [数据过滤成功] 已自动剔除未翻译成功 (非 Passed) 的脏数据样本数: {filtered_out} 条")
        print(f"🚀 [数据过滤成功] 实际送入评测的两阶段语义对齐黄金样本总量: {len(df_clean)} 条")
    else:
        print("⚠️  [流控警报] 未在输入数据集中检测到 'status' 字段，将对全量载入样本进行物理两阶段测试评测。")

    print(f"\n" + "="*50)
    print("📊 数据流控报告 (Data Stream Control Report)")
    print("=========================================")
    print(f"原始载入样本数 (Total Rows)              : {total_loaded}")
    print(f"🚀 送入评测的有效黄金样本数 (Golden Pairs)  : {len(df_clean)} 条")
    print("="*50 + "\n")

    return df_clean

# ==================== 6. 主控评测控制引擎 ====================
def run_live_pipeline(jsonl_path, llm_path, llm_type, output_report_csv, passed_save_path, batch_size=8):
    # 调用包含了 Passed 黄金过滤逻辑的智能加载引擎
    df = load_golden_data(jsonl_path)
    total_records = len(df)
    if total_records == 0: return

    # A. 环境自愈前置校验
    jacoco_agent_path, jacoco_cli_path = ensure_jacoco_tools()

    print("🔍 正在扫描分析 R 依赖包环境...")
    detected_r_libs = {"covr"}
    for _, row in df.iterrows():
        try:
            candidates = json.loads(row['all_candidates']) if isinstance(row['all_candidates'], str) else row['all_candidates']
            for cand in candidates:
                detected_r_libs.update(extract_packages_from_r_code(cand))
        except Exception: pass

    external_needed_libs = detected_r_libs - BUILTIN_R_PACKAGES
    if external_needed_libs:
        pkg_vector = ", ".join(f"'{p}'" for p in external_needed_libs)
        detect_script = f"needed<-c({pkg_vector});inst<-installed.packages()[,'Package'];missing<-needed[!needed %in% inst];cat(paste(missing,collapse=','))"
        res = subprocess.run(["Rscript", "-e", detect_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        missing_str = res.stdout.strip()
        if missing_str:
            missing_libs = [m.strip() for m in missing_str.split(",") if m.strip()]
            print(f"⚡ [R环境自愈] 批量安装缺失依赖: {missing_libs}")
            install_cmd = f"install.packages(c({','.join(f'\"{p}\"' for p in missing_libs)}), repos='https://cloud.r-project.org')"
            subprocess.run(["Rscript", "-e", install_cmd])

    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    stats = {
        'total': 0, 
        'java_test_failed': 0,       # 本地 Java 原生测试未跑通而剔除的样本数
        'r_test_syntax_errors': 0,   # R 测试用例本身存在语法错误的样本数
        'r_no_cand_passed': 0,       # R 测试用例没问题，但所有生成的 R 候选解全部测试失败
        'r_success': 0,              # 至少有 1 个 R 候选解答通过了测试的优秀样本数
        'p1': 0.0, 'p2': 0.0, 'p5': 0.0,
        'java_line_cov_sum': 0.0,
        'java_branch_cov_sum': 0.0,
        'r_line_cov_sum': 0.0
    }
    
    print(f"\n🚀 两阶段迁移反馈测试流水线启动 (Batch Size: {batch_size})")
    
    # 自动衍生详细评测报告 JSONL 文件的保存路径
    output_report_jsonl = output_report_csv.rsplit('.', 1)[0] + ".jsonl"
    
    os.makedirs(os.path.dirname(output_report_csv), exist_ok=True)
    os.makedirs(os.path.dirname(passed_save_path), exist_ok=True)

    # 同步写入精炼 CSV、全局 JSONL 详细报告、成功 R 语言资产库
    with open(output_report_csv, mode='w', encoding='utf-8', newline='') as f, \
         open(output_report_jsonl, mode='w', encoding='utf-8') as f_jsonl, \
         open(passed_save_path, mode='w', encoding='utf-8') as f_save:
         
        writer = csv.writer(f)
        writer.writerow([
            'idx', 'verify_mode', 'java_passed_tests', 'java_coverage_line', 'java_coverage_branch',
            'r_coverage_line', 'status', 'pass@1', 'pass@2', 'pass@5', 'passed_candidates'
        ])

        rows = [row for _, row in df.iterrows()]
        start_time = time.time()
        current_count = 0

        for i in range(0, total_records, batch_size):
            batch_rows = rows[i:i + batch_size]
            
            # ==================== 🛠️ 第一阶段：生成并评测 Java 原生测试 ====================
            prompts_java = []
            for row in batch_rows:
                java_code = row['java_code']
                class_name = extract_java_class_name(java_code)
                user_content_java = GEN_JAVA_TEST_USER_PROMPT.format(
                    class_name=class_name,
                    java_code=java_code
                )
                prompt_java = get_prompt_by_model(llm_type, GEN_TEST_SYSTEM_PROMPT, user_content_java)
                prompts_java.append(prompt_java)

            inputs_java = tokenizer(prompts_java, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                outputs_java = model.generate(
                    **inputs_java, max_new_tokens=1024, temperature=0.1, top_p=0.95, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            input_lens_java = inputs_java.input_ids.shape[1]
            
            # 保存该 Batch 中，通过了 Java 原生验证并保留的高质量样本信息
            active_batch_items = []
            
            for j, row in enumerate(batch_rows):
                idx = str(row['idx'])
                java_code = row['java_code']
                cand_str = row['all_candidates']
                
                output_str_java = tokenizer.decode(outputs_java[j][input_lens_java:], skip_special_tokens=True)
                
                # 提取 Java 原生测试 code
                java_test_code = ""
                java_match = re.search(r'```java\s*(.*?)\s*```', output_str_java, re.DOTALL)
                if java_match:
                    java_test_code = java_match.group(1).strip()
                else:
                    java_test_code = output_str_java.strip()

                stats['total'] += 1
                
                # 运行 Java 测试用例校验
                java_ok, java_passed, java_feedback, java_line_cov, java_branch_cov = verify_java_native_test_suite(
                    java_code, java_test_code, jacoco_agent_path, jacoco_cli_path
                )

                if not java_ok:
                    stats['java_test_failed'] += 1
                    # 5 个测试用例全部失败，或者编译不通过，判定为脏数据直接剔除。同步对齐写入 CSV 与详细 JSONL
                    writer.writerow([idx, "Java_Check_Failed", "0/5", "0.0%", "0.0%", "0.0%", "Java_Test_Failed", "0.0000", "0.0000", "0.0000", 0])
                    
                    report_item = {
                        "idx": int(idx),
                        "status": "Failed",
                        "global_r_status": "Java_Test_Failed",
                        "metrics": {
                            "java_coverage_line": 0.0,
                            "java_coverage_branch": 0.0,
                            "r_coverage_line_avg": 0.0,
                            "pass_at_1": 0.0,
                            "pass_at_2": 0.0,
                            "pass_at_5": 0.0
                        },
                        "passed_candidates_count": 0,
                        "total_candidates_count": 0,
                        "diagnostic_reasons": [java_feedback]
                    }
                    f_jsonl.write(json.dumps(report_item, ensure_ascii=False) + "\n")
                    f_jsonl.flush()
                    continue

                # 完美通过 Java 测试校验（至少 1 个用例跑通），保存数据信息供第二阶段调用
                item_data = {
                    'idx': idx,
                    'java_code': java_code,
                    'java_test_code': java_test_code,
                    'java_passed': java_passed,
                    'java_feedback': java_feedback,
                    'java_line_cov': java_line_cov,
                    'java_branch_cov': java_branch_cov,
                    'cand_str': cand_str
                }
                active_batch_items.append(item_data)
                
            # 若当前 batch 中所有数据都被剔除了，则直接进入下一次 Batch 循环
            if not active_batch_items:
                del inputs_java, outputs_java
                torch.cuda.empty_cache()
                current_count += len(batch_rows)
                continue

            # ==================== 🛠️ 第二阶段：迁移反馈并生成 R 测试用例 ====================
            prompts_r = []
            for item in active_batch_items:
                # 🎯 修复：不再强制默认 "solution"，而是多候选解答遍历安全提取 R 函数名称
                r_func_name = ""
                try:
                    cands = json.loads(item['cand_str']) if isinstance(item['cand_str'], str) else item['cand_str']
                    if cands:
                        for cand in cands:
                            name = extract_r_function_name(cand)
                            if name:
                                r_func_name = name
                                break
                    if not r_func_name:
                        r_func_name = "solution"  # 如果全部候选解都提取失败，才容错设定
                except Exception:
                    r_func_name = "solution"

                user_content_r = GEN_R_TEST_USER_PROMPT.format(
                    r_func_name=r_func_name,
                    java_feedback=item['java_feedback'],
                    java_code=item['java_code'],
                    java_test_code=item['java_test_code']
                )
                prompt_r = get_prompt_by_model(llm_type, GEN_TEST_SYSTEM_PROMPT, user_content_r)
                prompts_r.append(prompt_r)

            inputs_r = tokenizer(prompts_r, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                outputs_r = model.generate(
                    **inputs_r, max_new_tokens=1024, temperature=0.1, top_p=0.95, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            input_lens_r = inputs_r.input_ids.shape[1]

            # ==================== 🛠️ 第三阶段：跑通 R 候选评测与覆盖率计算并备份 ====================
            for j, item in enumerate(active_batch_items):
                current_count += 1
                idx = item['idx']
                java_passed = item['java_passed']
                java_line_cov = item['java_line_cov']
                java_branch_cov = item['java_branch_cov']
                cand_str = item['cand_str']

                output_str_r = tokenizer.decode(outputs_r[j][input_lens_r:], skip_special_tokens=True)
                
                # 提取 R 原生测试 code
                r_test_code = ""
                r_match = re.search(r'```[Rr]\s*(.*?)\s*```', output_str_r, re.DOTALL)
                if r_match:
                    r_test_code = r_match.group(1).strip()
                else:
                    r_test_code = output_str_r.strip()

                try:
                    candidates = json.loads(cand_str) if isinstance(cand_str, str) else cand_str
                except Exception:
                    writer.writerow([idx, "Java_Verified_Safe", f"{java_passed}/5", f"{java_line_cov:.2f}%", f"{java_branch_cov:.2f}%", "0.0%", "JSON_Err", "0.0000", "0.0000", "0.0000", 0])
                    report_item = {
                        "idx": int(idx),
                        "status": "Failed",
                        "global_r_status": "JSON_Err",
                        "metrics": {
                            "java_coverage_line": java_line_cov,
                            "java_coverage_branch": java_branch_cov,
                            "r_coverage_line_avg": 0.0,
                            "pass_at_1": 0.0,
                            "pass_at_2": 0.0,
                            "pass_at_5": 0.0
                        },
                        "passed_candidates_count": 0,
                        "total_candidates_count": 0,
                        "diagnostic_reasons": ["JSON candidates parse error"]
                    }
                    f_jsonl.write(json.dumps(report_item, ensure_ascii=False) + "\n")
                    f_jsonl.flush()
                    continue

                n = len(candidates)
                passed_count = 0
                candidate_r_line_covs = []
                
                # 用于追踪并精确保存成功与失败的代码形态（包含每条代码的细致执行状态）
                passed_candidates_list = []
                failed_candidates_list = []

                # 首先检查大模型生成的 R 测试用例本身是否存在语法错误
                # 检查时对原始代码进行一次清理，排除因缺失被 source 的文件引起的潜在解析波动
                cleaned_base_test_code = clean_r_test_code(r_test_code)
                r_test_ok = check_r_test_syntax(cleaned_base_test_code)

                for cand in candidates:
                    if not r_test_ok:
                        # 整个测试用例语法错误，导致候选解答无法进行测试
                        status = "Test_Syntax_Error"
                        candidate_r_line_covs.append(0.0)
                        failed_candidates_list.append({
                            "code": cand,
                            "status": status
                        })
                    else:
                        # 🎯 修复：动态检测并对齐当前 Candidate 函数名与 R 测试套件
                        aligned_r_test_code = align_test_with_candidate_func(cand, r_test_code)
                        
                        # 🎯 安全清理 R 测试代码中的 source() / sys.source() 语句，保障运行时不受加载异常文件干扰
                        cleaned_r_test_code = clean_r_test_code(aligned_r_test_code)
                        
                        # 用例通过，执行具体候选解分类状态测试
                        status = verify_r_candidate(cand, cleaned_r_test_code)
                        if status == "Success":
                            passed_count += 1
                            r_line_cov = calculate_r_native_coverage(cand, cleaned_r_test_code)
                            candidate_r_line_covs.append(r_line_cov)
                            passed_candidates_list.append({
                                "code": cand,
                                "status": "Success"
                            })
                        else:
                            candidate_r_line_covs.append(0.0)
                            failed_candidates_list.append({
                                "code": cand,
                                "status": status  # "Syntax_Error", "Runtime_Error", "Timeout", "Execution_Error"
                            })

                # 综合研判该题的 R 测试总体指标状态
                if not r_test_ok:
                    global_r_status = "R_Test_Syntax_Error"
                    stats['r_test_syntax_errors'] += 1
                elif passed_count > 0:
                    global_r_status = "R_Success"
                    stats['r_success'] += 1
                else:
                    global_r_status = "R_No_Candidate_Passed"
                    stats['r_no_cand_passed'] += 1

                valid_line_covs = [x for x in candidate_r_line_covs if x > 0]
                
                avg_r_line_cov = (
                    sum(valid_line_covs) / len(valid_line_covs)
                    if valid_line_covs else 0.0
                )

                stats['java_line_cov_sum'] += java_line_cov
                stats['java_branch_cov_sum'] += java_branch_cov
                stats['r_line_cov_sum'] += avg_r_line_cov

                p1 = calculate_pass_at_k(n, passed_count, 1)
                p2 = calculate_pass_at_k(n, passed_count, 2) if n >= 2 else p1
                p5 = calculate_pass_at_k(n, passed_count, 5) if n >= 5 else p2

                stats['p1'] += p1
                stats['p2'] += p2
                stats['p5'] += p5

                # 1. 写入精炼的 CSV 报表（包含精确整体 status）
                writer.writerow([
                    idx, "Java_Verified_Safe", f"{java_passed}/5", f"{java_line_cov:.2f}%", f"{java_branch_cov:.2f}%",
                    f"{avg_r_line_cov:.2f}%", global_r_status,
                    f"{p1:.4f}", f"{p2:.4f}", f"{p5:.4f}", passed_count
                ])

                # 2. 🎯 写入 JSONL 详细报告 (完全对齐 generate_and_test (3).py 的保存结构与键名)
                report_item = {
                    "idx": int(idx),
                    "status": "Passed" if passed_count > 0 else "Failed",
                    "global_r_status": global_r_status,
                    "metrics": {
                        "java_coverage_line": java_line_cov,
                        "java_coverage_branch": java_branch_cov,
                        "r_coverage_line_avg": avg_r_line_cov,
                        "pass_at_1": p1,
                        "pass_at_2": p2,
                        "pass_at_5": p5
                    },
                    "passed_candidates_count": passed_count,
                    "total_candidates_count": n,
                    "diagnostic_reasons": [item["status"] for item in failed_candidates_list if item.get("status")]
                }
                f_jsonl.write(json.dumps(report_item, ensure_ascii=False) + "\n")

                # 3. 🎯 写入 Passed R 专属过滤落盘文件（只有 passed_count > 0 时才落盘，完全对齐 (3) 的设计）
                if passed_count > 0:
                    passed_r_candidates = [item["code"] for item in passed_candidates_list]
                    save_item = {
                        "idx": int(idx),
                        "java_code": item['java_code'],
                        "java_test_code": item['java_test_code'],
                        "r_test_code": r_test_code,
                        "passed_r_candidates": passed_r_candidates,
                        "java_coverage_line": java_line_cov,
                        "java_coverage_branch": java_branch_cov,
                        "r_coverage_line": avg_r_line_cov,
                        "pass_at_1": p1,
                        "pass_at_2": p2,
                        "pass_at_5": p5
                    }
                    f_save.write(json.dumps(save_item, ensure_ascii=False) + "\n")

                f_jsonl.flush()
                f_save.flush()

            f.flush()
            
            # 过滤剔除数后，求得真正有效的题目数
            valid_num = max(stats['total'] - stats['java_test_failed'], 1)
            pct = (current_count / total_records) * 100
            
            elapsed_time = time.time() - start_time
            it_s = current_count / elapsed_time if elapsed_time > 0 else 0.0
            remaining_seconds = (total_records - current_count) / it_s if it_s > 0 else 0
            
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_seconds))

            # 滚动均值覆盖率计算
            java_line_avg = stats['java_line_cov_sum'] / valid_num
            java_branch_avg = stats['java_branch_cov_sum'] / valid_num
            r_line_avg = stats['r_line_cov_sum'] / valid_num

            # 实时进度看板（增加 R 语言测试状态分布显示，实时洞察微观分布变化）
            full_status_line = (
                f"进度:{pct:>5.1f}% ({current_count}/{total_records})[{elapsed_str}<{remaining_str},{it_s:.2f}it/s] "
                f"|总数:{stats['total']}|Java剔除:{stats['java_test_failed']}|R用例语法错:{stats['r_test_syntax_errors']}|R无通过:{stats['r_no_cand_passed']}|R成功:{stats['r_success']} "
                f"|P@1:{(stats['p1']/valid_num)*100:.2f}%|P@2:{(stats['p2']/valid_num)*100:.2f}%|P@5:{(stats['p5']/valid_num)*100:.2f}% "
                f"|Java行保:{java_line_avg:.1f}%|Java分保:{java_branch_avg:.1f}%|R行保:{r_line_avg:.1f}%"
            )
            print(full_status_line)
            sys.stdout.flush()

            del inputs_java, outputs_java, inputs_r, outputs_r
            torch.cuda.empty_cache()

        print("\n")

    del model
    torch.cuda.empty_cache()
    
    # 终期指标盘总结输出
    print("================ 📈 两阶段迁移评测总结 📈 ================")
    print(f"送入评测的优秀样本总量 (Total Records)     : {stats['total']}")
    print(f"Java 源码未跑通至少1个用例剔除数 (Java Fail): {stats['java_test_failed']}")
    
    valid_total = stats['total'] - stats['java_test_failed']
    print(f"完美参与语义指标统计的有效题目数           : {valid_total}")
    print(f"  ├─ R 测试用例代码语法错误件数 (R Test Syntax Error): {stats['r_test_syntax_errors']}")
    print(f"  ├─ R 用例没问题但候选代码全部不通过 (R No Cand Passed): {stats['r_no_cand_passed']}")
    print(f"  └─ 至少有1个候选代码测试通过的正常件数 (R Success)    : {stats['r_success']}")
    
    if valid_total > 0:
        print(f"🌟 R 语言语义对齐平均 Pass@1: {stats['p1'] / valid_total * 100:.2f}%")
        print(f"🌟 R 语言语义对齐平均 Pass@2: {stats['p2'] / valid_total * 100:.2f}%")
        print(f"🌟 R 语言语义对齐平均 Pass@5: {stats['p5'] / valid_total * 100:.2f}%")
        print("-" * 50)
        print(f"📐 Java 平均行覆盖率 (Line Coverage)      : {stats['java_line_cov_sum'] / valid_total:.2f}%")
        print(f"📐 Java 平均分支覆盖率 (Branch Coverage)  : {stats['java_branch_cov_sum'] / valid_total:.2f}%")
        print(f"📐 R 平均行覆盖率 (Line Coverage)         : {stats['r_line_cov_sum'] / valid_total:.2f}%")
    print("==================================================================")
    print(f"🎉 评估顺利完成！报告与黄金代码已分别保存至:\n  - CSV 报告: {output_report_csv}\n  - JSONL 详情: {output_report_jsonl}\n  - 过滤落盘代码: {passed_save_path}\n")

if __name__ == "__main__":
    JSONL_PATH = "/root/autodl-tmp/first_paper/java/translation_dataset/R_golden/R_inference_results_original.csv"
    LLM_MODEL_PATH = "/root/autodl-tmp/first_paper/LLM/Qwen"
    CURRENT_LLM_TYPE = "qwen"
    OUTPUT_REPORT = "/root/autodl-tmp/first_paper/java/testcase_dataset/r_pass/R_passk_report.csv"
    
    # 过滤落盘的成功 R 语言解答代码资产 JSONL 保存路径
    PASSED_SAVE_PATH = "/root/autodl-tmp/first_paper/java/testcase_dataset/r_pass/R_passed_clean.jsonl"
    
    run_live_pipeline(
        jsonl_path=JSONL_PATH, 
        llm_path=LLM_MODEL_PATH, 
        llm_type=CURRENT_LLM_TYPE, 
        output_report_csv=OUTPUT_REPORT,
        passed_save_path=PASSED_SAVE_PATH
    )