import gzip
import json
import pandas as pd
import os
import re
import argparse

def remove_comments_and_docstrings_cpp(source):
    """
    专门针对 C++ 语言的清洗逻辑：
    - 移除 // 行注释
    - 移除 /* ... */ 块注释
    - 正确跳过普通字符串、字符以及 C++11 原始字符串 R"(...)"，防止误删其中的内容
    """
    if not isinstance(source, str) or not source.strip():
        return source

    # 1. /\*[\s\S]*?\*/ : 匹配块注释
    # 2. //.* : 匹配行注释
    # 3. R"([^(\s]*)\([\s\S]*?\)\1" : 匹配 C++11 原始字符串 R"delim(...)delim"
    # 4. "(?:\\.|[^\\"])*" : 匹配普通双引号字符串
    # 5. '(?:\\.|[^\\'])*' : 匹配单引号字符
    pattern = re.compile(
        r'/\*[\s\S]*?\*/|//.*|R"([^(\s]*)\([\s\S]*?\)\1"|"(?:\\.|[^\\"])*"|\'(?:\\.|[^\\\'])\'',
        re.MULTILINE
    )

    def replacer(match):
        s = match.group(0)
        # 如果匹配以 / 开头，说明是行注释或块注释，替换为空格
        if s.startswith('/'):
            return " "
        # 否则是字符串或字符，原样保留
        return s

    try:
        # 执行替换
        cleaned = re.sub(pattern, replacer, source)
        # 移除空行并重新拼接
        return '\n'.join([line for line in cleaned.splitlines() if line.strip() != ""])
    except Exception as e:
        print(f"Error processing C++ code: {e}")
        return source


def process_single_file(input_path, output_path, language):
    print(f"Reading: {input_path}")

    data = []
    try:
        with gzip.open(input_path, 'rt', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    df = pd.DataFrame(data)

    # 适配字段：通常包含 'content' 或 'code'
    code_col = 'content' if 'content' in df.columns else 'code'

    print(f"Cleaning comments for {len(df)} records (Language: {language})...")
    # 使用专门针对 C++ 的清洗函数
    df['cleaned_code'] = df[code_col].apply(remove_comments_and_docstrings_cpp)

    # 自动创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving to: {output_path}")
    df.to_pickle(output_path)
    print(f"Success! Shape: {df.shape}")


def main():
    parser = argparse.ArgumentParser(description='Process a single CodeSearchNet JSONL.GZ file for C++')
    # 默认路径更新为 cpp 相关路径
    parser.add_argument('--input', type=str, required=False, default=r'/root/autodl-tmp/first_paper/code_dataset/cpp/cpp_data.jsonl.gz', help='Path to the .jsonl.gz file')
    parser.add_argument('--output', type=str, required=False, default=r'/root/autodl-tmp/first_paper/code_dataset/cpp/cpp_data.pkl', help='Path to save the .pkl file')
    parser.add_argument('--lang', type=str, default='cpp', help='Language of the code')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    process_single_file(args.input, args.output, args.lang)


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()