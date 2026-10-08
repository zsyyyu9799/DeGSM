import json
import csv
import os
from tqdm import tqdm

def extract_java_matched_passed_data(compiled_jsonl_path, inference_csv_path, output_dir):
    """
    根据 Java 真正编译通过数据集的 idx (is_compile_pass == 1) 且推理结果中的 status == 'Passed'，
    双重过滤提取对应的黄金数据，同步输出为新的 CSV 和 JSONL 文件。
    """
    # 1. 验证文件路径
    if not os.path.exists(compiled_jsonl_path):
        print(f"❌ 错误: 找不到编译后的源文件 {compiled_jsonl_path}")
        return
    if not os.path.exists(inference_csv_path):
        print(f"❌ 错误: 找不到原始推理结果文件 {inference_csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, "R_inference_results_compiled_passed.csv")
    output_jsonl_path = os.path.join(output_dir, "R_inference_results_compiled_passed.jsonl")

    # 2. 第一步：从标记文件中，【严格筛选】is_compile_pass 为 1 的有效 idx
    print("Step 1: 正在读取 Java 验证文件并建立编译通过 (is_compile_pass==1) 的索引...")
    allowed_idxs = set()
    with open(compiled_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # 🔥 核心修正：只有当 is_compile_pass 显式等于 1 时，才算作真正编译通过！
                if 'idx' in data and data.get('is_compile_pass') == 1:
                    allowed_idxs.add(int(data['idx']))
            except Exception:
                continue
                
    print(f"✅ 严格筛选完成！共提取到 {len(allowed_idxs)} 个真正通过编译的 Java 有效 idx。")

    # 3. 第二步：流式读取推理 CSV 并进行【双重条件】过滤提取
    print("\nStep 2: 开始定点提取 [Java源码编译通过 且 R推理状态为 Passed] 的黄金数据...")
    
    match_count = 0
    total_inference_rows = 0

    with open(inference_csv_path, 'r', encoding='utf-8') as f_in_csv, \
         open(output_csv_path, 'w', encoding='utf-8', newline='') as f_out_csv, \
         open(output_jsonl_path, 'w', encoding='utf-8') as f_out_jsonl:

        reader = csv.DictReader(f_in_csv)
        headers = reader.fieldnames
        writer = csv.DictWriter(f_out_csv, fieldnames=headers)
        writer.writeheader()

        for row in tqdm(reader, desc="Filtering Java Golden Data"):
            total_inference_rows += 1
            try:
                current_idx = int(row['idx'])
            except (ValueError, TypeError):
                continue

            # 判定推理状态是否为 Passed
            current_status = row.get('status', '').strip()
            
            if current_idx in allowed_idxs and current_status == 'Passed':
                match_count += 1
                
                # 写入新的 CSV 文件
                writer.writerow(row)
                
                # 还原嵌套的 json 字符串为原生列表，避免双重转义
                try:
                    candidates_list = json.loads(row['all_candidates'])
                except Exception:
                    candidates_list = row['all_candidates']

                # 针对 Java 调整字典键名
                java_src_code = row.get('java_code', row.get('src_code', ''))

                jsonl_payload = {
                    "idx": current_idx,
                    "java_code": java_src_code,
                    "nl_desc": row.get('nl_desc', ''),
                    "all_candidates": candidates_list,
                    "passed_code": row.get('passed_code', ''),
                    "status": current_status
                }
                f_out_jsonl.write(json.dumps(jsonl_payload, ensure_ascii=False) + '\n')

    # 4. 打印可视化过滤报告
    print("\n" + "="*50)
    print("📊 Java 黄金数据集提取报告 (Java Golden Dataset Report)")
    print("=========================================")
    print(f"原始推理总样本数 (Inference Total Rows)  : {total_inference_rows}")
    print(f"Java 实际编译通过数 (True Clean Idxs)    : {len(allowed_idxs)}")
    print(f"🔥 最终满足双重条件提取数 (Passed Match) : {match_count}")
    print(f"📂 成果 CSV 保存至: {output_csv_path}")
    print(f"📂 成果 JSONL 保存至: {output_jsonl_path}")
    print("="*50)

if __name__ == "__main__":
    COMPILED_JSONL = "/root/autodl-tmp/first_paper/java/translation_dataset/nl_code_stage3_compiled_clean.jsonl"
    INFERENCE_CSV = "/root/autodl-tmp/first_paper/java/translation_dataset/R_inference_results_original.csv"
    OUTPUT_DIR = "/root/autodl-tmp/first_paper/java/translation_dataset/R_golden_纯文字"

    extract_java_matched_passed_data(COMPILED_JSONL, INFERENCE_CSV, OUTPUT_DIR)