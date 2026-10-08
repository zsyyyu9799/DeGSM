import torch
import os
import re
import yaml
import pandas as pd
import json
import csv
from tqdm import tqdm
from stage2.Stage2_AlignmentBridge import InnovativeStage2Bridge
from transformers import AutoTokenizer
from openai import OpenAI 
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)
from concurrent.futures import ThreadPoolExecutor, as_completed

class ExplicitLogicPromptGenerator:
    def __init__(self, stage2_config_path, stage2_model_path, api_key, model_name="gpt-4o-mini", max_workers=8):
        """
        - max_workers: 并发请求 API 的线程数，建议设为 5 ~ 15。取决于中转站和远端大模型的速率限制(RPM)。
        """
        with open(stage2_config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InnovativeStage2Bridge(self.config).to(self.device)
        self.model.load_state_dict(torch.load(stage2_model_path, map_location=self.device))
        self.model.eval()

        print(f"Initializing API Client for model: {model_name} with {max_workers} thread workers...")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://www.autodl.art/api/v1"
        )
        self.model_name = model_name
        self.max_workers = max_workers

        self.s2_tokenizer = AutoTokenizer.from_pretrained(self.config['model']['bert_model_dir'])
    
    def _single_api_call(self, index, code, system_prompt):
        """单个 API 请求的内部工作函数，方便线程池调用"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"### CODE TO ANALYZE:\n{code}"}
                ],
                max_completion_tokens=512,
                temperature=0.2,
            )
            summary = response.choices[0].message.content.strip()
            return index, summary
        except Exception as e:
            print(f"\n[API Error] Index {index} processed with exception: {e}")
            return index, "Error: Failed to generate summary via API."

    def generate_nl_description_parallel(self, codes):
        """【提速核心】使用线程池并发发起 API 请求"""
        system_prompt = (
            "Your only task is to extract the functional intent of the following code so that another AI can rewrite it.\n\n"
            "Please describe the code in the following simple format:\n"
            "1. **Overall Intent:** (What does this code do in 1 sentence?)\n"
            "2. **Logic Steps:** (Break down the execution flow into simple, numbered steps)\n"
            "3. **Core Inputs/Outputs:** (What goes in, what comes out)"
        )

        # 预先分配结果列表空间，保证多线程返回乱序后能按原顺序对齐
        summaries = [None] * len(codes)
        
        # 使用 ThreadPoolExecutor 并发发起网络 I/O 请求
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._single_api_call, idx, code, system_prompt)
                for idx, code in enumerate(codes)
            ]
            
            for future in as_completed(futures):
                idx, result = future.result()
                summaries[idx] = result
                
        return summaries

    @torch.no_grad()
    def process_and_save(self, shard_dir, output_path, batch_size=32):
        """
        核心处理逻辑优化版：
        - 扩大默认 batch_size (如 32)
        - 结合并发 API 生成
        - 增加自动 Flush 防止中断丢失数据
        """
        # 获取所有数据分片文件
        shard_files = sorted([
            os.path.join(shard_dir, f)
            for f in os.listdir(shard_dir)
            if f.startswith('processed_dataset_stage2_shard') and f.endswith('.pt')
        ])

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        jsonl_output_path = output_path.replace('.csv', '.jsonl')
        csv_output_path = output_path if output_path.endswith('.csv') else output_path + '.csv'

        print(f"Starting optimized parallel processing.")
        print(f"JSONL Output: {jsonl_output_path}")
        print(f"CSV Output: {csv_output_path}")

        csv_headers = ['idx', 'src_code', 'nl_description', 'graph_emb']

        with open(jsonl_output_path, 'w', encoding='utf-8') as f_jsonl, \
                open(csv_output_path, 'w', encoding='utf-8', newline='') as f_csv:

            csv_writer = csv.DictWriter(f_csv, fieldnames=csv_headers)
            csv_writer.writeheader()

            for f_shard in shard_files:
                shard_data = torch.load(f_shard, map_location='cpu')

                for i in tqdm(range(0, len(shard_data), batch_size), desc=f"Shard: {os.path.basename(f_shard)}"):
                    batch_items = shard_data[i : i + batch_size]
                    codes = [item['code'] for item in batch_items]
                
                    # --- A. 准备真实的文本输入 ---
                    text_inputs = self.s2_tokenizer(
                        codes,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config['data'].get('max_length', 512)
                    ).to(self.device)
                
                    # --- B. 准备模型输入负载 ---
                    node_list = [torch.tensor(item['node_embs'], dtype=torch.float32) for item in batch_items]
                    max_nodes = max([n.size(0) for n in node_list])
                    node_dim = node_list[0].size(1)
                    
                    padded_nodes = []
                    for n in node_list:
                        p = torch.zeros((max_nodes, node_dim), dtype=torch.float32)
                        p[:n.size(0), :] = n
                        padded_nodes.append(p)
                    
                    node_tensor = torch.stack(padded_nodes).to(self.device)
                
                    sample_input = {
                        'node_embs': node_tensor,
                        'text_input_ids': text_inputs.input_ids,
                        'text_atts': text_inputs.attention_mask,
                        'meta': [item['meta'] for item in batch_items]
                    }

                    # --- B. 提取 Stage 2 隐式图特征 ---
                    outputs = self.model(sample_input, task='test')
                    all_q_feats = outputs['query_output']
                    graph_soft_prompts = self.model.graph_proj(all_q_feats) 
                    
                    # --- C. 【高并发加速点】并发获取 API 描述 ---
                    nl_descs = self.generate_nl_description_parallel(codes)

                    # --- D. 保存结果 ---
                    for j, item in enumerate(batch_items):
                        current_graph_emb = graph_soft_prompts[j].cpu().tolist()
                        
                        payload = {
                            'idx': item.get('idx'),
                            'src_code': item['code'],
                            'nl_description': nl_descs[j],
                            'graph_emb': current_graph_emb
                        }
                        
                        # 写入 JSONL
                        f_jsonl.write(json.dumps(payload, ensure_ascii=False) + '\n')

                        # 写入 CSV
                        csv_payload = payload.copy()
                        csv_payload['graph_emb'] = json.dumps(current_graph_emb)
                        csv_writer.writerow(csv_payload)
                    
                    # 实时强制刷新数据缓冲区到磁盘，安全稳健
                    f_jsonl.flush()
                    f_csv.flush()

        print(f"\nSuccessfully generated both formats with optimized pipeline.")


if __name__ == "__main__":
    S2_CONFIG = "/root/autodl-tmp/first_paper/python/stage2/stage2.yaml"
    S2_MODEL = "/root/autodl-tmp/first_paper/output/python/second_stage/best_innovative_model.pt"
    SHARD_DIR = "/root/autodl-tmp/first_paper/graph_dataset/python/stage2_dataset/"
    OUTPUT_CSV = "/root/autodl-tmp/first_paper/python/translation_dataset/nl_code_stage3.csv"

    API_KEY = "4DpfEZViQVWMLqxnx0slgXFbSzCem8xVHzEKX9sxovINZ4tj" 
    MODEL_NAME = "Qwen3.5-397B-A17B"               

    # 💡 性能配置调优参数说明：
    # 1. MAX_WORKERS: 并发线程数。根据你和 autodl.art 接口服务商约定的 RPM (每分钟限制请求数) 来定。
    #    一般中转源设为 8 ~ 12 性能会有质的飞跃。如果频繁报错 429 (Rate Limit)，可以稍微调小该值。
    # 2. BATCH_SIZE: 控制 GPU 的单次吞吐。从 2 调大到 32 甚至 64，彻底喂饱显卡。
    MAX_WORKERS = 10
    BATCH_SIZE = 32

    generator = ExplicitLogicPromptGenerator(S2_CONFIG, S2_MODEL, API_KEY, MODEL_NAME, max_workers=MAX_WORKERS)
    generator.process_and_save(SHARD_DIR, OUTPUT_CSV, batch_size=BATCH_SIZE)