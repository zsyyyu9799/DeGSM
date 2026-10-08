import os
import pandas as pd

def extract_multipl_e_exact(dataset_name, parquet_path):
    """
    针对包含复数列名 ['completions', 'statuses', 'tests'] 的数据集进行精准提取
    """
    print(f" └─ 正在处理 [{dataset_name}]...")
    
    # 1. 读取原始 Parquet 数据
    df = pd.read_parquet(parquet_path)

    # 2. 核心提取函数：从列表里捞出第一个成功的样本
    def get_first_passed_sample(row):
        completions = row.get('completions', [])
        # 兼容处理：有些版本叫 statuses，有些叫 results
        statuses = row.get('statuses', row.get('results', []))
        
        # 遍历该题目的所有采样结果
        for i, status in enumerate(statuses):
            # 转换为字符串并去空格小写化，同时兼容 'ok' 和 'pass'
            status_str = str(status).strip().lower()
            if status_str in ['ok', 'pass']:
                return completions[i], "OK"
                
        # 如果这道题的所有采样都没有通过，返回空和首个状态（或者标记未通过）
        if len(completions) > 0:
            return completions[0], "FAILED"
        return "", "No_Data"

    # 3. 内存中应用提取逻辑，拆分为代码和状态两列
    df['best_code'] = df.apply(get_first_passed_sample, axis=1, result_type='expand')[0]
    df['best_status'] = df.apply(get_first_passed_sample, axis=1, result_type='expand')[1]

    # 4. 严格筛选：只保留成功运行通过（'OK'）的完美样本
    perfect_df = df[df['best_status'] == 'OK'].copy()

    # 5. 仅保留你要求的 5 个核心字段
    # 从报错看，你的数据集里天然自带 ['problem', 'prompt', 'tests']
    keep_cols = ['problem', 'best_code', 'best_status', 'prompt', 'tests']
    
    # 确保字段安全过滤
    available_cols = [col for col in keep_cols if col in perfect_df.columns]
    final_df = perfect_df[available_cols]
    
    print(f"    💡 [{dataset_name}] 成功提取到真正通过的 Python 独立样本数: {len(final_df)} / 原始总数: {len(df)}")
    return final_df


def generate_combined_python_data(datasets_config, output_dir="./output"):
    """
    循环处理多个数据集并合并
    """
    os.makedirs(output_dir, exist_ok=True)
    ft_output_file = os.path.join(output_dir, "multipl_e_python_extracted.jsonl")

    print(f"==================================================")
    print(f"🚀 开始根据实际列名定制化提取 Python 数据")
    print(f"==================================================")

    all_dataset_dfs = []

    for dataset_name, paths in datasets_config.items():
        try:
            df_extracted = extract_multipl_e_exact(
                dataset_name=dataset_name,
                parquet_path=paths["py_parquet"]
            )
            all_dataset_dfs.append(df_extracted)
        except Exception as e:
            print(f"❌ 处理数据集 [{dataset_name}] 时出错: {e}，已跳过。")

    if not all_dataset_dfs:
        print("❌ 没有成功处理任何数据集，终止导出。")
        return

    print(f"\n[📊 正在合并所有数据集...]")
    combined_df = pd.concat(all_dataset_dfs, ignore_index=True)

    # 写入 JSONL 文件
    combined_df.to_json(ft_output_file, orient='records', lines=True, force_ascii=False)
    
    print(f"==================================================")
    print(f"✨ 联合收割完成！")
    print(f"📦 最终唯一微调文件: {ft_output_file}")
    print(f"📈 拼接后的总有效 Python 样本数: {len(combined_df)} 条")
    print(f"==================================================")


# ==========================================
# 🎛️ 用户配置区域
# ==========================================
if __name__ == "__main__":
    
    # 在这里配置你需要提取的 Python 数据集输入路径
    DATASETS_CONFIG = {
        "MBPP": {
            "py_parquet": "./human_python2.parquet"
        },
        "HumanEval": {
            "py_parquet": "./humaneval_python.parquet"
        }
    }

    # 指定输出目录
    OUTPUT_DIR = "./python_ok"

    # 执行生成流水线
    generate_combined_python_data(
        datasets_config=DATASETS_CONFIG,
        output_dir=OUTPUT_DIR
    )