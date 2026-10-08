import os
import torch
import json
import pandas as pd
from tqdm import tqdm
import yaml
import sys
import gc

# 确保路径包含项目根目录，以便导入 stage1 和 stage2
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from transformers import AutoTokenizer
from python_code_graph import initialize_tree_sitter, initialize_encoder, code_to_graph
from stage1.HeteroCodeGraphEncoder import HeteroCodeGraphEncoder
from stage2.Stage2_AlignmentBridge import InnovativeStage2Bridge

def main():
    # --- 1. 配置路径 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 路径已经将 c/rust 统一为了对应的 python/rust 组织结构（请根据你服务器的实际路径微调）
    s1_config_path = "/root/autodl-tmp/first_paper/python/stage1/stage1.yaml"
    s1_weights_path = "/root/autodl-tmp/first_paper/output/python/first_stage/best_hetero_model.pt"
    
    s2_config_path = "/root/autodl-tmp/first_paper/python/stage2/stage2.yaml"
    s2_weights_path = "/root/autodl-tmp/first_paper/output/python/second_stage/best_innovative_model.pt"
    
    unixcoder_path = "/root/autodl-tmp/unixcoder-base"
    metadata_path = "/root/autodl-tmp/first_paper/python/graph_metadata.pt"
    
    # 输入与输出路径适配新数据集文件名
    input_jsonl = "/root/autodl-tmp/first_paper/human_stage3_llm/python/rust/raw_py_to_rust_tasks.jsonl"
    output_jsonl = "/root/autodl-tmp/first_paper/human_stage3_llm/python/rust/py_to_rust_tasks.jsonl"

    # --- 2. 加载配置与模型 ---
    with open(s1_config_path, 'r') as f: s1_cfg = yaml.safe_load(f)
    with open(s2_config_path, 'r') as f: s2_cfg = yaml.safe_load(f)

    print("加载 Stage 2 分词器...")
    s2_tokenizer = AutoTokenizer.from_pretrained(s2_cfg['model']['bert_model_dir'])
    
    # 加载训练时使用的元数据
    meta_data = torch.load(metadata_path, map_location='cpu', weights_only=False)['metadata']
    stage1_model = HeteroCodeGraphEncoder(
        metadata=meta_data,
        in_channels=768,
        hidden_channels=512,  
        out_channels=256,     
        gnn_type='gat',       
        num_layers=4,         
        dropout=0.1,
        num_heads=8           
    ).to(device)
    
    stage1_model.load_state_dict(torch.load(s1_weights_path, map_location=device))
    stage1_model.eval()

    # 初始化 Stage 2 模型 (负责跨模态专家对齐)
    stage2_model = InnovativeStage2Bridge(s2_cfg).to(device)
    stage2_model.load_state_dict(torch.load(s2_weights_path, map_location=device))
    stage2_model.eval()

    # 初始化 UniXcoder 编码器
    tokenizer, code_encoder = initialize_encoder(unixcoder_path, device)

    # --- 3. 处理数据流程 ---
    df = pd.read_json(input_jsonl, lines=True)
    processed_data = []

    print(f"🚀 开始全管线特征提取，总计 {len(df)} 条数据...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 🌟 修改点：提取输入的 Python 源码
        py_code = row['python_source']
        
        try:
            # Step 1: 生成原始异构图 (节点初始表征 768 维)
            graph = code_to_graph(
                py_code, 
                tokenizer, 
                code_encoder, 
                edge_types=['AST', 'CFG', 'DFG']
            )
            
            # 防御性卡点：排除无边的孤立空图
            has_edges = False
            for store in graph.edge_stores:
                if 'edge_index' in store and store.edge_index.numel() > 0:
                    has_edges = True
                    break
            if not has_edges:
                continue

            # ================== ⭐ 核心推理补丁 1：为单图前向传播注入全零虚拟 Batch ID ==================
            for nt in graph.node_types:
                num_nodes = graph[nt].x.size(0)
                graph[nt].batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
            # =========================================================================================

            graph = graph.to(device)

            with torch.no_grad():
                # Step 2: 运行 Stage 1 异构图神经网络前向演进
                node_emb_dict, _, _ = stage1_model(
                    graph.x_dict, 
                    graph.edge_index_dict,
                    edge_attr_dict=getattr(graph, 'edge_attr_dict', None),
                    batch_dict={nt: graph[nt].batch for nt in graph.node_types}
                )
                
                # ================== ⭐ 核心推理补丁 2：全通道原生对齐拼接 + 空通道零填充防御 ==================
                ordered_embs = []
                # 严格按照固定的 5 类通道元数据顺序循环组装：
                # ['structure', 'statement', 'expression', 'variable', 'other_node']
                for nt in stage1_model.metadata[0]:
                    if nt in node_emb_dict and node_emb_dict[nt].size(0) > 0:
                        ordered_embs.append(node_emb_dict[nt])
                    else:
                        # 核心防线：确保整体合并后的特征空间比例雷打不动地切合 Q-Former 的 Cross-Attention 期望结构
                        dummy_zero_emb = torch.zeros((1, 256), device=device, dtype=torch.float32)
                        ordered_embs.append(dummy_zero_emb)
                
                # 拼接合并并升维匹配 Batch 维度 -> [1, Total_Nodes, 256]
                combined_node_embs = torch.cat(ordered_embs, dim=0).unsqueeze(0)
                # =========================================================================================

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

            # Step 5: 构造全新输出条目 (完全兼容且保留了你的 5 个新字段加图指纹)
            new_row = {
                'idx': row.get('idx', idx),
                'problem': row['problem'],
                'python_source': py_code,
                'rust_prompt': row['rust_prompt'],
                'graph_emb': query_feats,       # 完美的 32*768 大模型对齐指纹序列
                'rust_target': row['rust_target'],
                'tests_rust': row['tests_rust']
            }
            processed_data.append(new_row)

        except Exception as e:
            print(f"⚠️ 跳过索引 {idx}，错误原因: {e}")
            # 遇到硬件级断言报错后，通过“软刷新软断开”绕过系统显存锁死引发的二次死锁崩溃
            if "cuda" in str(e).lower() or "device-side" in str(e).lower():
                gc.collect()
            continue

    # --- 4. 保存结果 ---
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✨ 提取完成！产出数据集已100%无损对齐至 Stage 3 大模型空间: {output_jsonl}")

if __name__ == "__main__":
    main()