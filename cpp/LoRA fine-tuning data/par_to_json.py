import pandas as pd

def parquet_to_jsonl(language, parquet_file_path, jsonl_file_path):
    # 1. 读取 Parquet 文件
    df = pd.read_parquet(parquet_file_path)
    
    # 2. 转换为 jsonl 格式
    df.to_json(jsonl_file_path, orient='records', lines=True, force_ascii=False)
    print(f"[{language.upper()}] 转换成功！已保存至: {jsonl_file_path}")

# 1. 定义你想处理的语言
# language1 = 'human_r'
# language2 = 'human_rust'
# language3 = 'human_racket'
language4 = 'racket/humaneval_racket1'
# language5 = 'human_ocaml'
# language6 = 'ocaml/humaneval_ocaml'
# 2. 在这里使用 f-string，把变量语言嵌入到字符串中
# 注意：f 必须写在双引号外面
# parquet_file1 = f"./{language1}.parquet"  
# jsonl_file1 = f"./{language1}.jsonl"    
# parquet_file2 = f"./{language2}.parquet"  
# jsonl_file2 = f"./{language2}.jsonl" 
# parquet_file3 = f"./{language3}.parquet"  
# jsonl_file3= f"./{language3}.jsonl" 
parquet_file4 = f"./{language4}.parquet"  
jsonl_file4 = f"./{language4}.jsonl" 
# parquet_file5 = f"./{language5}.parquet"  
# jsonl_file5 = f"./{language5}.jsonl" 
# parquet_file6 = f"./{language6}.parquet"  
# jsonl_file6 = f"./{language6}.jsonl" 
# 3. 传参时，位置参数必须放在关键字参数（如 language='rust'）的前面
# 或者直接像下面这样按顺序传参：
# parquet_to_jsonl(language1, parquet_file1, jsonl_file1)
# parquet_to_jsonl(language2, parquet_file2, jsonl_file2)
# parquet_to_jsonl(language3, parquet_file3, jsonl_file3)
parquet_to_jsonl(language4, parquet_file4, jsonl_file4)
# parquet_to_jsonl(language5, parquet_file5, jsonl_file5)
# parquet_to_jsonl(language6, parquet_file6, jsonl_file6)