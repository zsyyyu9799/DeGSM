import torch
import networkx as nx
from tqdm import tqdm


def extract_hybrid_logic_paths(hetero_data, max_paths=15, max_length=25):
    """
    整合 CFG (执行逻辑) 与 DFG (数据逻辑) 的混合路径提取。
    AST 信息已通过第一阶段编码器集成在节点向量中，此处通过索引保持关联。
    """
    # 1. 计算全局偏移量以支持跨节点类型搜索
    node_types = hetero_data.node_types
    offset = 0
    node_type_offsets = {}
    for n_type in node_types:
        num_nodes = hetero_data[n_type].x.size(0)
        node_type_offsets[n_type] = offset
        offset += num_nodes

    # 2. 构建混合逻辑图 (Hybrid Logic Graph)
    G = nx.DiGraph()

    # 核心：同时筛选 CFG 和 DFG 边
    # CFG 负责“代码怎么走”，DFG 负责“值怎么传”
    logic_edge_types = ['cfg', 'dfg']

    for edge_type in hetero_data.edge_types:
        rel_name = edge_type[1].lower()
        if rel_name in logic_edge_types:
            edge_index = hetero_data[edge_type].edge_index
            src_offset = node_type_offsets[edge_type[0]]
            dst_offset = node_type_offsets[edge_type[2]]

            for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                G.add_edge(s + src_offset, d + dst_offset, type=rel_name)

    if G.number_of_nodes() == 0:
        return []

    # 3. 提取具有语义意义的逻辑链条
    # 起点：入度为 0 的节点（通常是函数入口或变量初始定义）
    sources = [n for n, d in G.in_degree() if d == 0]
    # 终点：出度为 0 的节点（通常是 return 语句或变量的最终使用点）
    sinks = [n for n, d in G.out_degree() if d == 0]

    if not sources: sources = list(G.nodes())[:2]

    hybrid_paths = []
    try:
        for start in sources:
            for end in sinks:
                # 寻找简单路径，cutoff 限制长度防止递归爆炸
                paths = nx.all_simple_paths(G, source=start, target=end, cutoff=max_length)
                for p in paths:
                    hybrid_paths.append(p)
                    if len(hybrid_paths) >= max_paths: break
                if len(hybrid_paths) >= max_paths: break
            if len(hybrid_paths) >= max_paths: break
    except Exception:
        pass

    return hybrid_paths[:max_paths]


def main():
    graph_data_path = '/root/autodl-tmp/first_paper/graph_dataset/python/processed_dataset.pt'
    output_path = '/root/autodl-tmp/first_paper/graph_dataset/python/stage2_dataset/processed_logic_paths_dataset.pt'

    print(f"Loading hetero graph dataset...")
    data_dict = torch.load(graph_data_path, weights_only=False)
    graphs = data_dict['graphs']

    logic_paths_collection = []

    print("Extracting Hybrid (CFG+DFG) paths with AST-ready Global Indices...")
    for idx, graph in enumerate(tqdm(graphs)):
        # 提取融合了控制流与数据流的路径
        paths = extract_hybrid_logic_paths(graph)

        logic_paths_collection.append({
            'idx': getattr(graph, 'idx', idx),
            'hybrid_logic_paths': paths  # 格式：[[global_idx1, global_idx2, ...], ...]
        })

    torch.save(logic_paths_collection, output_path)
    print(f"Success: Saved {len(logic_paths_collection)} records to {output_path}")


if __name__ == "__main__":
    main()