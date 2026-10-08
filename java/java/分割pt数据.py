import torch
import random
from torch.utils.data import random_split

# 1. 加载原始数据
# 假设你的文件名为 data.pt
full_data = torch.load('/root/autodl-tmp/first_paper/graph_dataset/java/processed_dataset.pt', weights_only=False)
graph_list = full_data['graphs']  # 提取 HeteroData 列表
# is_partial = full_data.get('is_partial', True) # 保留原始元数据
edge_list = full_data['edge_label_to_name']
# 2. 计算各部分数量
total_count = len(graph_list)
train_size = int(0.8 * total_count)
val_size = int(0.1 * total_count)
test_size = total_count - train_size - val_size

print(f"总样本数: {total_count}")
print(f"训练集: {train_size}, 验证集: {val_size}, 测试集: {test_size}")

# 3. 使用 random_split 进行随机划分
# random_split 返回的是 Subset 对象，我们通过它索引原列表
generator = torch.Generator().manual_seed(42) # 固定随机种子
train_subs, val_subs, test_subs = random_split(
    graph_list,
    [train_size, val_size, test_size],
    generator=generator
)

# 4. 封装回原始字典格式
# 因为 HeteroData 通常需要这种特定的字典结构来保持兼容性
train_data = {'graphs': [graph_list[i] for i in train_subs.indices], 'edge_label_to_name': edge_list}
val_data = {'graphs': [graph_list[i] for i in val_subs.indices], 'edge_label_to_name': edge_list}
test_data = {'graphs': [graph_list[i] for i in test_subs.indices], 'edge_label_to_name': edge_list}

# 5. 保存为新的 .pt 文件
torch.save(train_data, '/root/autodl-tmp/first_paper/graph_dataset/java/train_graphs.pt')
torch.save(val_data, '/root/autodl-tmp/first_paper/graph_dataset/java/val_graphs.pt')
torch.save(test_data, '/root/autodl-tmp/first_paper/graph_dataset/java/test_graphs.pt')

print("划分完成并已保存为三个新文件。")