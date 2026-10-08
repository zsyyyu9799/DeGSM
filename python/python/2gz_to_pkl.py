import gzip
import json
import pandas as pd
import os
import sys
import argparse
import re
from io import StringIO
import tokenize


def remove_comments_and_docstrings(source, lang):
    """
    核心清洗逻辑：移除代码中的注释和文档字符串
    """
    if not isinstance(source, str):
        return source

    try:
        if lang.lower() == 'python':
            io_obj = StringIO(source)
            out = ""
            prev_toktype = tokenize.INDENT
            last_lineno = -1
            last_col = 0

            try:
                for tok in tokenize.generate_tokens(io_obj.readline):
                    token_type, token_string, (start_line, start_col), (end_line, end_col), _ = tok

                    if start_line > last_lineno:
                        last_col = 0
                    if start_col > last_col:
                        out += (" " * (start_col - last_col))

                    if token_type == tokenize.COMMENT:
                        pass
                    elif token_type == tokenize.STRING:
                        # 仅保留非起始位置的字符串（简单过滤掉 docstrings）
                        if prev_toktype != tokenize.INDENT and prev_toktype != tokenize.NEWLINE:
                            out += token_string
                    else:
                        out += token_string

                    prev_toktype = token_type
                    last_col = end_col
                    last_lineno = end_line

                # 移除空行
                return '\n'.join([x for x in out.split('\n') if x.strip() != ""])
            except Exception:
                # Tokenize 失败后的正则保底方案
                no_comments = re.sub(r'#.*$', '', source, flags=re.MULTILINE)
                return re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', '', no_comments)

        else:
            # 非 Python 语言的通用正则处理
            pattern = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"', re.DOTALL | re.MULTILINE)

            def replacer(match):
                s = match.group(0)
                return " " if s.startswith('/') else s

            return '\n'.join([x for x in re.sub(pattern, replacer, source).split('\n') if x.strip() != ""])

    except Exception as e:
        print(f"Error processing code: {e}")
        return source


def process_single_file(input_path, output_path, language):
    """
    处理单个 jsonl.gz 文件并保存为 pkl
    """
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

    # 确保关键列存在
    cols_to_keep = ['code', 'docstring', 'func_name', 'language']
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[existing_cols]

    print(f"Cleaning comments for {len(df)} records...")
    df['cleaned_code'] = df['code'].apply(lambda x: remove_comments_and_docstrings(x, language))

    # 自动创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving to: {output_path}")
    df.to_pickle(output_path)
    print(f"Success! Shape: {df.shape}")


def main():
    parser = argparse.ArgumentParser(description='Process a single CodeSearchNet JSONL.GZ file')
    parser.add_argument('--input', type=str, required=False, default=r'/root/autodl-tmp/first_paper/code_dataset/python/python_data.jsonl.gz', help='Path to the .jsonl.gz file')
    parser.add_argument('--output', type=str, required=False, default=r'/root/autodl-tmp/first_paper/code_dataset/python/python_data.pkl', help='Path to save the .pkl file')
    parser.add_argument('--lang', type=str, default='python', help='Language of the code')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    process_single_file(args.input, args.output, args.lang)


if __name__ == '__main__':
    main()