import os
import torch
import json
import pandas as pd
from tqdm import tqdm
import yaml
import sys
import gc

# 确保路径包含项目根目录，以便导入 stage1 和 stage2
python_project_dir = "/root/autodl-tmp/first_paper/python"

if python_project_dir not in sys.path:
    sys.path.insert(0, python_project_dir)
from transformers import AutoTokenizer
from python_code_graph import initialize_tree_sitter, initialize_encoder, code_to_graph
from stage1.HeteroCodeGraphEncoder import HeteroCodeGraphEncoder
from stage2.Stage2_AlignmentBridge import InnovativeStage2Bridge

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- 1. 配置基础路径 ---
    s1_config_path = "/root/autodl-tmp/first_paper/python/stage1/stage1.yaml"
    s1_weights_path = "/root/autodl-tmp/first_paper/output/python/first_stage/best_hetero_model.pt"
    
    s2_config_path = "/root/autodl-tmp/first_paper/python/stage2/stage2.yaml"
    s2_weights_path = "/root/autodl-tmp/first_paper/output/python/second_stage/best_innovative_model.pt"
    
    unixcoder_path = "/root/autodl-tmp/unixcoder-base"
    metadata_path = "/root/autodl-tmp/first_paper/python/graph_metadata.pt"
    
    # 🎯 定义我们需要批量处理的 7 种语言与配置（保持分开输出）
    languages = {
        'r': {'input': './mbpp_python_vs_r.jsonl', 'output': '/root/autodl-tmp/first_paper/human_stage3_llm/python/r/new_mbpp_python_vs_r.jsonl'},
        'rust': {'input': './mbpp_python_vs_rust.jsonl', 'output': '/root/autodl-tmp/first_paper/human_stage3_llm/python/rust/new_mbpp_python_vs_rust.jsonl'},
        'racket': {'input': './mbpp_python_vs_racket.jsonl', 'output': '/root/autodl-tmp/first_paper/human_stage3_llm/python/racket/new_mbpp_python_vs_racket.jsonl'},
        'lua': {'input': './mbpp_python_vs_lua.jsonl', 'output': '/root/autodl-tmp/first_paper/human_stage3_llm/python/lua/new_mbpp_python_vs_lua.jsonl'},
        'ocaml': {'input': './mbpp_python_vs_ocaml.jsonl', 'output': '/root/autodl-tmp/first_paper/human_stage3_llm/python/ocaml/new_mbpp_python_vs_ocaml.jsonl'}
    }

    # --- 2. 加载配置与模型（只在最开始加载一次，避免显存爆炸） ---
    with open(s1_config_path, 'r') as f: s1_cfg = yaml.safe_load(f)
    with open(s2_config_path, 'r') as f: s2_cfg = yaml.safe_load(f)

    print("加载 Stage 2 分词器...")
    s2_tokenizer = AutoTokenizer.from_pretrained(s2_cfg['model']['bert_model_dir'])
    
    meta_data = torch.load(metadata_path, map_location='cpu', weights_only=False)['metadata']
    stage1_model = HeteroCodeGraphEncoder(
        metadata=meta_data, in_channels=768, hidden_channels=512, out_channels=256,     
        gnn_type='gat', num_layers=4, dropout=0.1, num_heads=8           
    ).to(device)
    stage1_model.load_state_dict(torch.load(s1_weights_path, map_location=device))
    stage1_model.eval()

    stage2_model = InnovativeStage2Bridge(s2_cfg).to(device)
    stage2_model.load_state_dict(torch.load(s2_weights_path, map_location=device))
    stage2_model.eval()

    tokenizer, code_encoder = initialize_encoder(unixcoder_path, device)

    # --- 3. 外层大循环：一次性串行处理每个语言的数据集 ---
    for lang, paths in languages.items():
        input_jsonl = paths['input']
        output_jsonl = paths['output']
        
        if not os.path.exists(input_jsonl):
            print(f"⚠️ 跳过语言 [{lang.upper()}], 找不到输入对齐文件: {input_jsonl}")
            continue
            
        print(f"\n================ ⚙️ 开始处理语言: [{lang.upper()}] ================")
        df = pd.read_json(input_jsonl, lines=True)
        processed_data = []

        print(f"🚀 开始全管线特征提取，总计 {len(df)} 条数据...")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {lang}"):
            # 🌟 动态适配对齐数据中的 Python 代码字段
            py_code = row.get('python_code')
            if not py_code or not str(py_code).strip():
                continue
            
            try:
                # Step 1: 生成原始异构图
                graph = code_to_graph(py_code, tokenizer, code_encoder, edge_types=['AST', 'CFG', 'DFG'])
                
                has_edges = False
                for store in graph.edge_stores:
                    if 'edge_index' in store and store.edge_index.numel() > 0:
                        has_edges = True
                        break
                if not has_edges:
                    continue

                # 补丁 1：注入全零虚拟 Batch ID
                for nt in graph.node_types:
                    num_nodes = graph[nt].x.size(0)
                    graph[nt].batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

                graph = graph.to(device)

                with torch.no_grad():
                    # Step 2: 运行 Stage 1 异构图网络
                    node_emb_dict, _, _ = stage1_model(
                        graph.x_dict, graph.edge_index_dict,
                        edge_attr_dict=getattr(graph, 'edge_attr_dict', None),
                        batch_dict={nt: graph[nt].batch for nt in graph.node_types}
                    )
                    
                    # 补丁 2：全通道原生对齐拼接
                    ordered_embs = []
                    for nt in stage1_model.metadata[0]:
                        if nt in node_emb_dict and node_emb_dict[nt].size(0) > 0:
                            ordered_embs.append(node_emb_dict[nt])
                        else:
                            dummy_zero_emb = torch.zeros((1, 256), device=device, dtype=torch.float32)
                            ordered_embs.append(dummy_zero_emb)
                    
                    combined_node_embs = torch.cat(ordered_embs, dim=0).unsqueeze(0)

                    # Step 3: 构造完全符合 Stage 2 推理接口的 samples 字典
                    samples = {
                        'node_embs': combined_node_embs,
                        'text_input_ids': torch.zeros((1, 1), dtype=torch.long).to(device),
                        'text_atts': torch.zeros((1, 1), dtype=torch.long).to(device),
                        'meta': [{'node_type_map': []}]
                    }
                    
                    # Step 4: 运行 Stage 2 前向传播
                    outputs = stage2_model(samples, task='test')
                    query_feats = outputs['query_output'].squeeze(0).cpu().numpy().tolist()

                # 🌟 Step 5: 构造全新输出条目 (字段名和结构动态适配目标语言)
                new_row = {
                    'idx': row.get('idx', idx),
                    'python_code': py_code,
                    'nl_description': row.get(f'{lang}_prompt', ''), # 把当前语言的 prompt 统一映射为大模型认识的描述字段
                    'graph_emb': query_feats,                         # 完美的 32*768 大模型对齐指纹序列
                    f'{lang}_code': row.get(f'{lang}_code', ''),     # 目标语言参考代码（或称 r_target 等）
                    f'{lang}_tests': row.get(f'{lang}_tests', [])    # 目标语言过滤后的测试用例
                }
                processed_data.append(new_row)

            except Exception as e:
                # 遇到硬件级断言报错后，通过“软刷新软断开”绕过系统显存锁死
                if "cuda" in str(e).lower() or "device-side" in str(e).lower():
                    gc.collect()
                continue

        # --- 4. 分开保存每种语言的结果 ---
        if processed_data:
            os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
            with open(output_jsonl, 'w', encoding='utf-8') as f:
                for item in processed_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"✨ 成功导出 [{lang.upper()}] 独立数据集 -> {output_jsonl}")
            del processed_data
            gc.collect() # 及时释放内存/显存
        else:
            print(f"⚠️ 语言 [{lang.upper()}] 未产出有效数据，跳过文件写入。")

    print("\n🎉 所有语言的对齐数据集均已成功提取 graph_emb，并独立分开保存！")

if __name__ == "__main__":
    main()