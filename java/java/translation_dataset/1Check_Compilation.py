import os
import re
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class JavaJsonlCompilationChecker:
    def __init__(self, jsonl_path, tmp_dir="./tmp_compile_check", max_workers=16):
        self.jsonl_path = jsonl_path
        self.tmp_dir = tmp_dir
        self.max_workers = max_workers
        
        # 创建干净的临时编译目录
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def _wrap_and_get_classname(self, code, idx):
        """
        解析类名，或者将碎片化的 Java 方法片段包裹成长合法的类
        """
        code = code.strip()
        
        # 1. 优先匹配 public class 名称
        class_match = re.search(r'public\s+class\s+([A-Za-z0-9_]+)', code)
        if class_match:
            return class_match.group(1), code
            
        # 2. 匹配普通 class 名称
        class_match_normal = re.search(r'class\s+([A-Za-z0-9_]+)', code)
        if class_match_normal:
            return class_match_normal.group(1), code

        # 3. 片段代码自动包裹，并补全高频基础包防止误报
        fallback_classname = f"TempClass_{idx}"
        wrapped_code = (
            "import java.util.*;\n"
            "import java.io.*;\n"
            "import java.math.*;\n"
            f"public class {fallback_classname} {{\n"
            f"    // Auto-wrapped snippet\n"
            f"    {code}\n"
            "}"
        )
        return fallback_classname, wrapped_code

    def _compile_single(self, idx, code):
        """单条代码编译线程任务"""
        # 为每条数据建立独立文件夹，防止多线程环境下类名撞车
        task_dir = os.path.join(self.tmp_dir, f"task_{idx}")
        os.makedirs(task_dir, exist_ok=True)
        
        try:
            classname, final_code = self._wrap_and_get_classname(code, idx)
            java_file_path = os.path.join(task_dir, f"{classname}.java")
            
            with open(java_file_path, 'w', encoding='utf-8') as f:
                f.write(final_code)
            
            # 调用系统 javac
            cmd = ["javac", "-nowarn", "-proc:none", java_file_path]
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                timeout=5
            )
            
            if result.returncode == 0:
                return idx, True, "Success"
            else:
                return idx, False, result.stderr.strip()
                
        except subprocess.TimeoutExpired:
            return idx, False, "Error: Compilation Timeout"
        except Exception as e:
            return idx, False, f"Error: {str(e)}"
        finally:
            if os.path.exists(task_dir):
                shutil.rmtree(task_dir)

    def run_check(self, output_report_path):
        # 1. 逐行读取 JSONL 数据
        print(f"Loading JSONL dataset from {self.jsonl_path}...")
        records = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                
                records.append({
                    'idx': data.get('idx', line_idx),
                    'src_code': data['src_code'],
                    'full_raw_data': data  # 保留整行原始字典
                })
        
        print(f"Total rows loaded: {len(records)}. Starting parallel compilation check with {self.max_workers} threads...")
        
        results = {}
        # 2. 多线程并发编译
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._compile_single, item['idx'], item['src_code'])
                for item in records
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Compiling"):
                idx, success, msg = future.result()
                results[idx] = {"success": success, "message": msg}

        # 3. 结果统计
        success_count = sum(1 for r in results.values() if r['success'])
        total_count = len(records)
        pass_rate = (success_count / total_count) * 100 if total_count > 0 else 0
        
        print(f"\n" + "="*40)
        print(f"📊 编译检查报告 (JSONL Compilation Report)")
        print(f"="*40)
        print(f"总测试代码量: {total_count}")
        print(f"编译通过数量: {success_count}")
        print(f"编译失败数量: {total_count - success_count}")
        print(f"🔥 最终编译通过率 (Pass Rate): {pass_rate:.2f}%")
        print(f"="*40)

        # 4. 【核心修改】只保存编译通过的数据，剔除错误数据
        print(f"Saving cleaned dataset (only passed items) to {output_report_path}...")
        saved_count = 0
        with open(output_report_path, 'w', encoding='utf-8') as f_out:
            for item in records:
                idx = item['idx']
                res = results.get(idx, {"success": False, "message": "Not Processed"})
                
                # 🎯 关键修改：只有编译成功才写入文件
                if res['success']:
                    f_out.write(json.dumps(item['full_raw_data'], ensure_ascii=False) + '\n')
                    saved_count += 1
                
        # 清理临时根目录
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        print(f"Done! Saved {saved_count} pure records.")

if __name__ == "__main__":
    # 配置路径
    JSONL_INPUT = "/root/autodl-tmp/first_paper/java/translation_dataset/nl_code_stage3_cleaned.jsonl"
    JSONL_OUTPUT = "/root/autodl-tmp/first_paper/java/translation_dataset/nl_code_stage3_compiled_clean.jsonl"
    
    checker = JavaJsonlCompilationChecker(jsonl_path=JSONL_INPUT, max_workers=16)
    checker.run_check(output_report_path=JSONL_OUTPUT)