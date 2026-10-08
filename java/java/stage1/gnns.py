import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, ModuleList, Linear, Dropout, GRUCell, Sequential, ReLU
from torch_geometric.nn import GCNConv, TransformerConv, GATv2Conv
from torch_geometric.utils import softmax

# --- 新增：深度残差解码块，用于加强 768 维特征还原 ---
class DeepResidualBlock(torch.nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        # 引入中间压缩维度 (Bottleneck)，增加非线性表达能力
        mid_channels = channels // 2 
        self.net = Sequential(
            Linear(channels, mid_channels),
            LayerNorm(mid_channels),
            ReLU(),
            Dropout(dropout),
            Linear(mid_channels, mid_channels),
            ReLU(),
            Linear(mid_channels, channels),
            LayerNorm(channels)
        )
        self.shortcut = Sequential() # 如果输入输出维度不一致可加映射，这里一致

    def forward(self, x):
        # 残差连接 + 激活
        return F.relu(x + self.net(x))

class LogicGatedGATv2Conv(GATv2Conv):
    def __init__(self, in_channels, out_channels, heads=1, edge_dim=None, **kwargs):
        kwargs['add_self_loops'] = kwargs.get('add_self_loops', False)
        super(LogicGatedGATv2Conv, self).__init__(in_channels, out_channels, heads=heads, edge_dim=edge_dim, **kwargs)

        self.d_h = self.out_channels
        self.gate_norm = LayerNorm(self.d_h * 3)
        self.gate_mlp = Sequential(Linear(self.d_h, self.d_h // 2), ReLU(), Linear(self.d_h // 2, 1))
        self.temp = 0.5
        self.gru_gate = GRUCell(self.d_h * 3, self.d_h)

        # 保持可视化记录变量
        self.last_raw_alpha = None
        self.last_raw_logits = None
        self.last_logic_score = None
        self.last_filtered_alpha = None

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        E, H, D = x_j.size()

        # --- 标准 GATv2 评分逻辑 ---
        is_edge_attr_valid = (edge_attr is not None and torch.is_tensor(edge_attr) and edge_attr.numel() > 0)
        edge_feat = None
        x = x_i + x_j

        if is_edge_attr_valid:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_feat = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            x = x + edge_feat

        x = F.leaky_relu(x, self.negative_slope)
        raw_score = (x * self.att).sum(dim=-1) # 这是归一化前的 Logits
        self.last_raw_logits = raw_score.detach() # 新增记录项
        alpha = softmax(raw_score, index, ptr, size_i)

        # 【可视化记录 A】原始注意力
        self.last_raw_alpha = alpha.detach()

        # --- 强化版逻辑门控处理 ---
        x_j_flat = x_j.reshape(-1, D)
        x_i_flat = x_i.reshape(-1, D)
        
        if edge_feat is not None:
            e_feat_flat = edge_feat.reshape(-1, D)
        else:
            e_feat_flat = torch.zeros_like(x_j_flat)

        # 优化 4: 特征拼接后先进行归一化
        gate_inputs = torch.cat([x_j_flat, x_i_flat, e_feat_flat], dim=-1)
        gate_inputs = self.gate_norm(gate_inputs)

        gate_h = self.gru_gate(gate_inputs, x_i_flat)
        
        # 优化 5: 通过 MLP 计算 Logits，并应用温度系数
        gate_logits = self.gate_mlp(gate_h).view(E, H)
        gate_score = torch.sigmoid(gate_logits / self.temp)

        # 【可视化记录 B/c】
        self.last_logic_score = gate_score.detach()
        # 最终过滤后的注意力权重
        end_alpha = alpha * gate_score
        self.last_filtered_alpha = end_alpha.detach()

        return x_j * F.dropout(end_alpha, p=self.dropout, training=self.training).unsqueeze(-1)

class LogicGatedGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super(LogicGatedGAT, self).__init__()
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.skips = ModuleList()
        self.dropouts = ModuleList()

        for i in range(num_layers):
            # 关键：正确设置每一层的输入和输出维度
            layer_in = in_channels if i == 0 else hidden_channels
            layer_out = out_channels if i == num_layers - 1 else (hidden_channels // num_heads)
            current_heads = 1 if i == num_layers - 1 else num_heads

            # 每一层的 edge_dim 始终使用原始 in_channels (768)
            # 因为 CGE_Trainer.py 中 forward 传入的 data.edge_attr_dict 始终是原始特征
            self.convs.append(LogicGatedGATv2Conv(
                layer_in, layer_out, heads=current_heads, edge_dim=in_channels
            ))

            # 最后一层不需要 BN 和 Dropout (通常做法)
            if i < num_layers - 1:
                self.bns.append(LayerNorm(hidden_channels))
                self.skips.append(Linear(layer_in, hidden_channels))
                self.dropouts.append(Dropout(dropout))
            else:
                self.final_skip = Linear(layer_in, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        for i, conv in enumerate(self.convs[:-1]):
            residual = self.skips[i](x)
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.elu(x + residual)
            x = self.bns[i](x)
            x = self.dropouts[i](x)

        x = self.convs[-1](x, edge_index=edge_index, edge_attr=edge_attr) + self.final_skip(x)
        return x


class GCN(torch.nn.Module):
    """基础 GCN 保持作为 Baseline"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super(GCN, self).__init__()
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.skips = ModuleList()
        self.dropouts = ModuleList()

        for i in range(num_layers - 1):
            in_dim = in_channels if i == 0 else hidden_channels
            self.convs.append(GCNConv(in_dim, hidden_channels, add_self_loops=False))
            self.bns.append(LayerNorm(hidden_channels))
            self.skips.append(Linear(in_dim, hidden_channels))
            self.dropouts.append(Dropout(dropout))

        self.convs.append(GCNConv(hidden_channels, out_channels, add_self_loops=False))
        self.final_skip = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            residual = self.skips[i](x)
            x = conv(x, edge_index)
            x = F.relu(x + residual)
            x = self.bns[i](x)
            x = self.dropouts[i](x)
        return self.convs[-1](x, edge_index) + self.final_skip(x)

class StandardGAT(torch.nn.Module):
    """移除逻辑门控后的标准 GAT 模型"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super(StandardGAT, self).__init__()
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.skips = ModuleList()
        self.dropouts = ModuleList()

        for i in range(num_layers - 1):
            in_dim = in_channels if i == 0 else hidden_channels
            # 使用标准的 GATv2Conv 替换 LogicGatedGATv2Conv
            self.convs.append(GATv2Conv(in_dim, hidden_channels // num_heads,
                                        heads=num_heads, edge_dim=in_channels, add_self_loops=False))
            self.bns.append(LayerNorm(hidden_channels))
            self.skips.append(Linear(in_dim, hidden_channels))
            self.dropouts.append(Dropout(dropout))

        self.convs.append(GATv2Conv(hidden_channels, out_channels,
                                    heads=1, concat=False, edge_dim=in_channels, add_self_loops=False))
        self.final_skip = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        for i, conv in enumerate(self.convs[:-1]):
            residual = self.skips[i](x)
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.elu(x + residual)
            x = self.bns[i](x)
            x = self.dropouts[i](x)
        x = self.convs[-1](x, edge_index=edge_index, edge_attr=edge_attr) + self.final_skip(x)
        return x

class StandardTransformer(torch.nn.Module):
    """标准的 Graph Transformer"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=8):
        super(StandardTransformer, self).__init__()
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.skips = ModuleList()
        self.dropouts = ModuleList()

        for i in range(num_layers - 1):
            in_dim = in_channels if i == 0 else hidden_channels
            self.convs.append(TransformerConv(
                in_channels=in_dim,
                out_channels=hidden_channels // num_heads,
                heads=num_heads,
                edge_dim=in_channels
            ))
            self.bns.append(LayerNorm(hidden_channels))
            self.skips.append(Linear(in_dim, hidden_channels))
            self.dropouts.append(Dropout(dropout))

        self.convs.append(TransformerConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            heads=1,
            concat=False,
            edge_dim=in_channels
        ))
        self.final_skip = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        for i, conv in enumerate(self.convs[:-1]):
            residual = self.skips[i](x)
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.gelu(x + residual)
            x = self.bns[i](x)
            x = self.dropouts[i](x)
        x = self.convs[-1](x, edge_index=edge_index, edge_attr=edge_attr) + self.final_skip(x)
        return x

load_gnn_model = {'gat': LogicGatedGAT, 'gcn': GCN, 'gt': StandardTransformer}