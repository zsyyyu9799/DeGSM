import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml
import os
import json

# 强制要求 transformers 使用原生实现以捕获注意力矩阵
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
from CGBridge_Stage3 import CGBridgeStage3, CodeTranslateDataset

def plot_attention_for_idx(target_idx, config_path):
    # 1. 加载配置与设备准备
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 初始化模型 (同步 Stage 3 的多模态架构)
    model = CGBridgeStage3(config).to(device)
    model.eval()
    
    # 3. 获取测试样本
    dataset = CodeTranslateDataset(config['data']['test_path'])
    sample_row = dataset.df[dataset.df['idx'] == target_idx]
    
    if sample_row.empty:
        print(f"❌ 错误: 找不到 idx 为 {target_idx} 的样本。")
        return

    # 4. 解析图特征 (同步 JSONL 逻辑)
    raw_emb = sample_row.iloc[0]['graph_emb']
    graph_emb = torch.tensor(raw_emb if not isinstance(raw_emb, str) else json.loads(raw_emb)).unsqueeze(0).to(device)
    
    python_code = [str(sample_row.iloc[0].get('python_code', "")) ]
    nl_desc = [str(sample_row.iloc[0].get('nl_description', ""))]

    # 5. 执行同步的前向传播流程
    with torch.no_grad():
        # A. 规范化图输入并提取特征 (同步 CGBridge_Stage3.py 的 forward 逻辑)
        model_dev = next(model.parameters()).device
        multimodal_embeds, multimodal_atts = model._normalize_graph_inputs(graph_emb, device=model_dev)
        query_tokens = model.query_tokens.expand(multimodal_embeds.shape[0], -1, -1).to(model_dev)
        
        query_output = model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=multimodal_embeds,
            encoder_attention_mask=multimodal_atts,
            return_dict=True,
        )
        
        # B. 获取投影后的 32 位图特征
        qformer_hidden_state = query_output.last_hidden_state[:, :query_tokens.size(1), :]
        gtokens = model.llm_proj(qformer_hidden_state)

        # C. 构造 LLM 输入 (同步最新的参数列表)
        input_ids, _, inputs_embeds, attention_mask, _ = model.prepare_lm_input(
            gtokens=gtokens,
            text_input=python_code,
            nl_desc=nl_desc,
            answer=None
        )

        # D. 捕获注意力矩阵
        outputs = model.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )

    # 6. 注意力数据提取
    # 取最后一层的所有 Head 的平均值
    last_layer_attn = outputs.attentions[-1][0] 
    avg_attn_matrix = last_layer_attn.mean(dim=0).float().cpu().numpy()

    # 7. 精准定位 <|graph|> 标记
    graph_token_id = model.llm_tokenizer.convert_tokens_to_ids("<|graph|>")
    graph_indices = (input_ids[0] == graph_token_id).nonzero(as_tuple=True)[0].cpu().numpy()

    if len(graph_indices) > 0:
        graph_start, graph_end = graph_indices[0], graph_indices[-1]
        print(f"✅ 定位成功: 图标记范围 [{graph_start}:{graph_end}]")
    else:
        print("⚠️ 未能定位图标记，请检查词表")
        return

    # 8. 绘制热力图
    # 截取局部视图，聚焦于图标记及其后续上下文
    x_view_start = max(0, graph_start - 10)
    x_view_end = min(avg_attn_matrix.shape[1], graph_end + 60)
    y_view_end = min(avg_attn_matrix.shape[0], graph_end + 120)

    plt.figure(figsize=(14, 9))
    sns.heatmap(
        avg_attn_matrix[:y_view_end, x_view_start:x_view_end], 
        cmap="rocket_r", 
        cbar_kws={'label': 'Attention Intensity'}
    )
    
    # 调整高亮框位置 (减去偏移量)
    plt.axvspan(graph_start - x_view_start, graph_end - x_view_start + 1, 
                color='cyan', alpha=0.3, label='Structured Graph Tokens')
    
    plt.title(f"Stage 3 Cross-Modal Attention Map (idx: {target_idx})")
    plt.xlabel("Key Tokens (Input Context)")
    plt.ylabel("Query Tokens (Generation Flow)")
    plt.legend()

    save_path = f"stage3_attn_idx_{target_idx}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图像已保存: {save_path}")

if __name__ == "__main__":
    CONFIG_PATH = "/root/autodl-tmp/first_paper/python/stage3/stage3.yaml"
    TARGET_IDX = 14 
    plot_attention_for_idx(TARGET_IDX, CONFIG_PATH)