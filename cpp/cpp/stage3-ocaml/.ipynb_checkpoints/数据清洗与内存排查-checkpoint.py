import pandas as pd
import json
import os
import re
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

def analyze_and_clean_dataset(input_path, output_path, max_token_len=1500, model_path="/root/autodl-tmp/first_paper/LLM/Qwen2.5"):
    """
    排查数据集长度、校验数据质量并去除异常数据
    """
    print(f"正在加载分词器: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 1. 加载数据
    print(f"正在读取数据: {input_path}")
    if input_path.endswith('.jsonl'):
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(input_path)

    initial_count = len(df)
    
    # 2. 质量清洗阶段 (Quality Cleaning)
    print("正在进行质量清洗...")
    
    # A. 剔除缺失值
    df.dropna(subset=['src_code', 'nl_description', 'graph_emb'], inplace=True)
    
    # B. 过滤 Qwen 拒答或无效描述
    refusal_patterns = [
        r"I cannot explain", r"I am an AI", r"Sorry", r"抱歉", r"无法总结",
        r"The logic is unclear", r"is missing", r"invalid code"
    ]
    refusal_regex = "|".join(refusal_patterns)
    df = df[~df['nl_description'].str.contains(refusal_regex, case=False, na=False)]
    
    # C. 剔除重复代码（保证样本多样性，避免过拟合）
    df.drop_duplicates(subset=['src_code'], keep='first', inplace=True)

    # D. 校验 Graph Embedding 维度 (假设 Stage 2 输出为 [32, 256])
    def is_valid_emb(emb):
        try:
            # 如果是字符串形式则解析
            curr_emb = json.loads(emb) if isinstance(emb, str) else emb
            return np.array(curr_emb).shape == (32, 256)
        except:
            return False
    
    mask = df['graph_emb'].apply(is_valid_emb)
    df = df[mask]
    quality_cleaned_count = len(df)

    # 3. 统计长度与显存风险排查
    print("正在计算 Token 长度以评估显存风险...")
    # 统计 src_code 的 token 长度
    df['token_len'] = [len(tokenizer.encode(str(x))) for x in tqdm(df['src_code'], desc="Tokenizing")]
    df['char_len'] = df['src_code'].apply(lambda x: len(str(x)))

    # 4. 打印统计报告
    print("\n" + "="*30)
    print("--- 数据长度统计报告 ---")
    print(df[['token_len', 'char_len']].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))
    print("="*30)
    
    # 5. 最终长度过滤
    df_final = df[df['token_len'] <= max_token_len].copy()
    if 'idx' in df_final.columns:
        df_final.rename(columns={'idx': 'original_idx'}, inplace=True)
    df_final = df_final.reset_index(drop=True)
    df_final['idx'] = df_final.index
    final_count = len(df_final)
    
    # 6. 打印清洗结果汇总
    print(f"\n--- 清理结果汇总 ---")
    print(f"1. 原始总条数:     {initial_count}")
    print(f"2. 质量清洗后保留: {quality_cleaned_count} (过滤掉 {initial_count - quality_cleaned_count} 条垃圾数据)")
    print(f"3. 长度限制后保留: {final_count} (因超过 {max_token_len} tokens 过滤掉 {quality_cleaned_count - final_count} 条)")
    print(f"总过滤比例:       {((initial_count - final_count) / initial_count) * 100:.2f}%")

    # 7. 保存结果
    if output_path.endswith('.jsonl'):
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in df_final.to_dict('records'):
                # 移除临时统计列
                clean_record = {k: v for k, v in record.items() if k not in ['token_len', 'char_len']}
                f.write(json.dumps(clean_record, ensure_ascii=False) + '\n')
    else:
        df_final.drop(columns=['token_len', 'char_len']).to_csv(output_path, index=False)
    
    print(f"\n清洗完成！数据已保存至: {output_path}")

if __name__ == "__main__":
    # 输入输出路径保持一致
    INPUT = "/root/autodl-tmp/first_paper/cpp/translation_dataset/nl_code_stage3.jsonl"
    OUTPUT = "/root/autodl-tmp/first_paper/cpp/translation_dataset/nl_code_stage3_cleaned.jsonl"
    
    # 执行清洗与排查
    analyze_and_clean_dataset(INPUT, OUTPUT, max_token_len=1024)