import torch
import os
from torch.utils.data import ConcatDataset, random_split
from tqdm import tqdm

# 1. 配置路径
data_dir = '/root/autodl-tmp/first_paper/graph_dataset/java/stage2_dataset/'
# 查找所有分片
shard_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt') and 'processed_dataset_stage2' in f])

print(f"检测到 {len(shard_files)} 个数据分片，准备合并...")

# 2. 物理加载数据
all_data = []
for file_name in tqdm(shard_files, desc="加载分片"):
    file_path = os.path.join(data_dir, file_name)
    # 加载后立即转为 list 存入，打破对原始文件的依赖
    shard = torch.load(file_path, weights_only=True)
    if isinstance(shard, list):
        all_data.extend(shard)
    else:
        all_data.append(shard)

total_size = len(all_data)
print(f"合并完成，总样本数: {total_size}")

# 3. 执行随机分割 (7:1.5:1.5)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

# 使用固定随机种子保证可重复性
train_data, val_data, test_data = random_split(
    all_data, 
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# 4. 封装保存函数：确保保存的是真实的 List[dict] 而不是 Subset 对象
def save_real_data(subset, filename):
    print(f"正在持久化 {filename}...")
    # 这一步是关键：将 Subset 转换为真正的列表
    actual_list = [subset[i] for i in range(len(subset))]
    torch.save(actual_list, os.path.join(data_dir, filename))
    print(f"保存成功！文件大小: {len(actual_list)}")

save_real_data(train_data, 'train_data_stage2.pt')
save_real_data(val_data, 'val_data_stage2.pt')
save_real_data(test_data, 'test_data_stage2.pt')

print("\n所有分割任务已完成，可以直接用于训练脚本。")