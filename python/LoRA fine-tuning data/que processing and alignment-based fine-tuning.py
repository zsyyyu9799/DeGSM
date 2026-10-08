import os
import pandas as pd

def align_single_dataset(dataset_name, py_parquet_path, target_parquet_path, lang):
    """
    对单个数据集进行内存对齐与严格筛选（双语皆 OK）
    """
    print(f" └─ 正在处理 [{dataset_name}]...")
    
    # 1. 读取原始 Parquet 数据
    df_py = pd.read_parquet(py_parquet_path)
    df_target = pd.read_parquet(target_parquet_path)

    # 2. 内存中执行 'OK' 优先策略提取最优解
    def get_best_sample(row):
        completions = row.get('completions', [])
        statuses = row.get('statuses', [])
        for i, status in enumerate(statuses):
            if status == 'OK':
                return completions[i], "OK"
        if len(completions) > 0:
            return completions[0], statuses[0]
        return "", "No_Data"

    df_py['best_code'] = df_py.apply(get_best_sample, axis=1, result_type='expand')[0]
    df_py['best_status'] = df_py.apply(get_best_sample, axis=1, result_type='expand')[1]

    df_target['best_code'] = df_target.apply(get_best_sample, axis=1, result_type='expand')[0]
    df_target['best_status'] = df_target.apply(get_best_sample, axis=1, result_type='expand')[1]

    # 3. 双语基于 problem ID 内联对齐
    aligned_df = pd.merge(
        df_py[['problem', 'best_code', 'best_status', 'prompt', 'tests']],
        df_target[['problem', 'best_code', 'best_status', 'prompt', 'tests']],
        on='problem',
        suffixes=('_py', f'_{lang}')
    )

    # 4. 严格筛选双语状态均为 'OK' 的完美对齐子集
    perfect_df = aligned_df[
        (aligned_df['best_status_py'] == 'OK') &
        (aligned_df[f'best_status_{lang}'] == 'OK')
    ].copy()

    # 5. 字段精简与规范化重命名
    rename_mapping = {
        'best_code_py': 'python_source',
        f'prompt_{lang}': f'{lang}_prompt',
        f'best_code_{lang}': f'{lang}_target'
    }
    ft_df = perfect_df.rename(columns=rename_mapping)
    
    # 仅保留指定的 5 个核心字段
    keep_cols = ['problem', 'python_source', f'{lang}_prompt', f'{lang}_target', f'tests_{lang}']
    ft_df = ft_df[keep_cols]
    
    print(f"    💡 [{dataset_name}] 完美对齐样本数: {len(ft_df)} / 原始对齐数: {len(aligned_df)}")
    return ft_df


def generate_combined_fine_tune_data(lang_name, datasets_config, output_dir="./output"):
    """
    循环处理多个数据集，将结果在内存中拼接，最终仅输出一个合并后的微调 JSONL 文件
    """
    lang = lang_name.lower()
    os.makedirs(output_dir, exist_ok=True)
    
    # 动态定义最终的合并输出文件名
    ft_output_file = os.path.join(output_dir, f"raw_py_to_{lang}_combined.jsonl")

    print(f"==================================================")
    print(f"🚀 开始多数据集联合构建 | 目标语言: {lang_name.upper()}")
    print(f"==================================================")

    all_dataset_dfs = []

    # 遍历配置中的所有数据集进行对齐提取
    for dataset_name, paths in datasets_config.items():
        try:
            df_aligned = align_single_dataset(
                dataset_name=dataset_name,
                py_parquet_path=paths["py_parquet"],
                target_parquet_path=paths["target_parquet"],
                lang=lang
            )
            all_dataset_dfs.append(df_aligned)
        except Exception as e:
            print(f"❌ 处理数据集 [{dataset_name}] 时出错: {e}，已跳过。")

    if not all_dataset_dfs:
        print("❌ 没有成功处理任何数据集，终止导出。")
        return

    # 内存中高速拼接所有 DataFrame
    print(f"\n[📊 正在合并所有数据集...]")
    combined_df = pd.concat(all_dataset_dfs, ignore_index=True)

    # 写入最终唯一的微调 JSONL 文件
    combined_df.to_json(ft_output_file, orient='records', lines=True, force_ascii=False)
    
    print(f"==================================================")
    print(f"✨ 联合收割完成！")
    print(f"📦 最终唯一微调文件: {ft_output_file}")
    print(f"📈 拼接后的总微调样本数: {len(combined_df)} 条")
    print(f"==================================================")


# ==========================================
# 🎛️ 用户一键切换配置区域 (支持多数据集配置)
# ==========================================
if __name__ == "__main__":
    
    # 1. 切换目标语言：'rust'、'lua'、'racket'、'ocaml'、'r' 等
    TARGET_LANGUAGE = "ocaml" 
    
    # 2. 在这里配置你需要合并的所有数据集输入路径（可自由增减）
    DATASETS_CONFIG = {
        "MBPP": {
            "py_parquet": "./ocaml/humaneval.py.bigcode_15b_400m.0.8.reworded-00000-of-00001-37a13462da3478d4.parquet",
            "target_parquet": "./ocaml/humaneval.ml.starcoder2_15b.0.2.reworded.parquet"
        }#,
        # "HumanEval": {
        #     "py_parquet": "./ocaml/human_py.parquet",
        #     "target_parquet": "./ocaml/human_ocaml.parquet"
        # }
        # 如果后续还有新数据集，按下面这种格式直接往下加即可：
        # "ShareCoder": {
        #     "py_parquet": "sharecoder_py.parquet",
        #     "target_parquet": "sharecoder_rust.parquet"
        # }
    }

    # 3. 指定输出目录
    OUTPUT_DIR = "./ocaml"

    # 执行生成流水线
    generate_combined_fine_tune_data(
        lang_name=TARGET_LANGUAGE,
        datasets_config=DATASETS_CONFIG,
        output_dir=OUTPUT_DIR
    )