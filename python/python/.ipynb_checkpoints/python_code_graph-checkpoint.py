import tree_sitter_python as tspython
from tree_sitter import Language, Parser
import os
import torch
from torch_geometric.data import HeteroData
from transformers import RobertaModel, RobertaTokenizer
import tree_sitter
from typing import Dict, List, Tuple
import argparse

# ================ 1. 异构图元数据定义 =================

def get_model_metadata():
    """
    通过笛卡尔积动态生成全量元数据，彻底杜绝 KeyError。
    """
    node_types = ['structure', 'statement', 'expression', 'variable', 'other_node']
    rel_types = ['ast', 'cfg', 'dfg']

    edge_types = []
    # 动态生成所有可能的 (src, rel, dst) 组合
    for src in node_types:
        for rel in rel_types:
            for dst in node_types:
                edge_types.append((src, rel, dst))

    # 自动为每个节点添加自环
    for n in node_types:
        edge_types.append((n, 'self_loop', n))

    return node_types, edge_types


# ================= 2. 异构图节点分类逻辑 =================

def get_node_category(node_type: str) -> str:
    """
    优化后的映射：更全面地覆盖 Python 语法节点。
    """
    # 结构/定义类
    structures = {
        'module', 'function_definition', 'class_definition', 'parameters',
        'default_parameter', 'decorated_definition', 'decorator', 'type_alias'
    }

    # 语句类
    statements = {
        'if_statement', 'for_statement', 'while_statement', 'try_statement',
        'with_statement', 'return_statement', 'break_statement', 'continue_statement',
        'expression_statement', 'assignment', 'augmented_assignment', 'import_statement',
        'import_from_statement', 'elif_clause', 'else_clause', 'except_clause', 'finally_clause',
        'block', 'assert_statement', 'global_statement', 'nonlocal_statement', 'pass_statement',
        'match_statement', 'case_clause'
    }

    # 表达式/字面量类
    expressions = {
        'comparison_operator', 'binary_operator', 'call', 'attribute',
        'subscript', 'parenthesized_expression', 'list', 'dictionary', 'set',
        'string', 'integer', 'float', 'boolean', 'none', 'list_comprehension',
        'dictionary_comprehension', 'set_comprehension', 'conditional_expression',
        'unary_operator', 'lambda', 'yield', 'await', 'tuple', 'slice'
    }

    if node_type == 'identifier':
        return 'variable'
    elif node_type in structures:
        return 'structure'
    elif node_type in statements:
        return 'statement'
    elif node_type in expressions:
        return 'expression'
    else:
        return 'other_node'

# ================= 3. Tree-sitter 基础解析 =================

def initialize_tree_sitter():
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    return parser


def build_node_map(code: str, parser: Parser) -> Tuple[tree_sitter.Node, Dict]:
    # 增加编码健壮性处理
    code_bytes = bytes(code, 'utf8', errors='replace')
    tree = parser.parse(code_bytes)
    root_node = tree.root_node
    node_map = {}
    code_lines = code.split('\n')

    def traverse(node, node_id=0):
        if node.is_named:
            node_map[node_id] = {
                'id': node_id,
                'type': node.type,
                'start_point': node.start_point,
                'end_point': node.end_point,
                'start_byte': node.start_byte,
                'end_byte': node.end_byte,
                'text': extract_node_text(node, code_lines),
                'children': []
            }
            children_ids = []
            next_id = node_id + 1
            for child in node.children:
                if child.is_named:
                    child_id = next_id
                    next_id, _ = traverse(child, next_id)
                    children_ids.append(child_id)
            node_map[node_id]['children'] = children_ids
            return next_id, node_id
        return node_id, None

    _, root_id = traverse(root_node)
    return root_node, node_map


def extract_node_text(node, code_lines):
    start_point = node.start_point
    end_point = node.end_point
    try:
        if start_point[0] == end_point[0]:
            return code_lines[start_point[0]][start_point[1]:end_point[1]]
        else:
            # 优化多行提取逻辑
            lines = [code_lines[start_point[0]][start_point[1]:]]
            for i in range(start_point[0] + 1, end_point[0]):
                lines.append(code_lines[i])
            lines.append(code_lines[end_point[0]][:end_point[1]])
            return "\n".join(lines)
    except IndexError:
        return ""


def refine_node_text(node_map: Dict) -> Dict:
    # 增加 match 语法处理
    statement_types = [
        'if_statement', 'for_statement', 'while_statement', 'try_statement',
        'elif_clause', 'else_clause', 'except_clause', 'finally_clause',
        'function_definition', 'class_definition', 'with_statement', 'match_statement'
    ]
    for node_id, info in node_map.items():
        original_text = info['text']
        if info['type'] in statement_types:
            info['refined_text'] = original_text.split('\n')[0].strip() if '\n' in original_text else original_text
        else:
            # 截断过长的表达式文本，防止 Embedding 溢出
            info['refined_text'] = original_text[:512] if len(original_text) > 512 else original_text
    return node_map


# ================= 4. 边构建逻辑 (CFG, DFG, AST) =================

def build_cfg(node_map: Dict) -> List[Tuple[int, int, str, str]]:
    cfg_edges = []
    # 块内顺序执行
    for node_id, info in node_map.items():
        if info['type'] == 'block':
            for i in range(len(info['children']) - 1):
                cfg_edges.append((info['children'][i], info['children'][i + 1], 'CFG', 'sequential'))

    # 控制流分支增强
    for node_id, info in node_map.items():
        if info['type'] == 'if_statement':
            for child_id in info['children']:
                c_info = node_map[child_id]
                if c_info['type'] == 'block':
                    cfg_edges.append((node_id, child_id, 'CFG', 'true_branch'))
                elif c_info['type'] in ['else_clause', 'elif_clause']:
                    cfg_edges.append((node_id, child_id, 'CFG', 'false_branch'))

        elif info['type'] in ['for_statement', 'while_statement']:
            for child_id in info['children']:
                if node_map[child_id]['type'] == 'block':
                    cfg_edges.append((node_id, child_id, 'CFG', 'loop_body'))
                    if node_map[child_id]['children']:
                        cfg_edges.append((node_map[child_id]['children'][-1], node_id, 'CFG', 'loop_back'))

        # 增加 With 结构
        elif info['type'] == 'with_statement':
            for child_id in info['children']:
                if node_map[child_id]['type'] == 'block':
                    cfg_edges.append((node_id, child_id, 'CFG', 'with_body'))

    return cfg_edges

def build_dfg(node_map: Dict) -> List[Tuple[int, int, str, str]]:
    dfg_edges = []

    def process_node(node_id, states):
        info = node_map[node_id]
        local_dfg = []

        if info['type'] == 'identifier':
            var_name = info['text']
            if var_name in states:
                for def_id in states[var_name]:
                    if def_id != node_id: local_dfg.append((def_id, node_id, 'DFG', 'data_flow'))

        elif info['type'] in ['assignment', 'augmented_assignment']:
            left_id, right_id = None, None
            for child_id in info['children']:
                if node_map[child_id]['type'] in ['identifier', 'subscript', 'attribute']:
                    left_id = child_id
                elif node_map[child_id]['type'] not in ['=', '+=', '-=', '*=', '/=']:
                    right_id = child_id

            if left_id and right_id:
                local_dfg.append((right_id, left_id, 'DFG', 'assignment'))
                if node_map[left_id]['type'] == 'identifier':
                    var_name = node_map[left_id]['text']
                    states[var_name] = [left_id]

        for child_id in info['children']:
            sub_dfg, _ = process_node(child_id, states)
            local_dfg.extend(sub_dfg)
        return local_dfg, states

    for root_id, info in node_map.items():
        if info['type'] == 'module':
            edges, _ = process_node(root_id, {})
            dfg_edges.extend(edges)
    return list(set(dfg_edges))

def extract_ast_edges(node_map: Dict) -> List[Tuple[int, int, str, str]]:
    ast_edges = []
    for node_id, info in node_map.items():
        for child_id in info['children']:
            ast_edges.append((node_id, child_id, 'AST', 'child_of'))
    return ast_edges


# ================= 5. 编码与嵌入 =================

def initialize_encoder(model_name_or_path: str, device: str):
    tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
    model = RobertaModel.from_pretrained(model_name_or_path).to(device)
    model.eval()
    return tokenizer, model


def encode_nodes(node_map: Dict, tokenizer, model, batch_size=64) -> torch.Tensor:
    device = next(model.parameters()).device
    node_ids = sorted(node_map.keys())
    texts = [node_map[nid].get('refined_text', '') for nid in node_ids]

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # 过滤掉空文本
        batch = [t if (t and t.strip()) else "empty" for t in batch]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # 使用 Mean Pooling 或 CLS，UniXcoder 建议使用 [CLS]
            batch_embs = outputs.last_hidden_state[:, 0, :].detach().cpu().half()
            all_embeddings.append(batch_embs)

    return torch.cat(all_embeddings, dim=0).float() if all_embeddings else torch.zeros((0, 768))


def encode_edges(all_edges: List, tokenizer, model, batch_size=128, edge_cache_path=None) -> torch.Tensor:
    if not all_edges: return torch.zeros((0, 768))

    # 尝试读取缓存
    if edge_cache_path and os.path.exists(edge_cache_path):
        try:
            cache = torch.load(edge_cache_path, weights_only=False)
            # 如果缓存是字典且包含所需键
            edge_embs = [cache.get(attr, torch.zeros(768)) for _, _, _, attr in all_edges]
            return torch.stack(edge_embs)
        except:
            pass

    device = next(model.parameters()).device
    texts = [attr for _, _, _, attr in all_edges]
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=64).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            all_embeddings.append(outputs.last_hidden_state[:, 0, :].detach().cpu().half())

    return torch.cat(all_embeddings, dim=0).float()

# ================= 6. 异构图组装 (含 node_positions 补全) =================

def build_pyg_hetero_graph(node_map: Dict, all_edges: List, node_embeddings: torch.Tensor,
                           edge_embeddings: torch.Tensor) -> HeteroData:
    data = HeteroData()
    node_ids = sorted(node_map.keys())

    # 1. 节点分类与索引映射
    global_to_local = {}
    cat_to_global_indices = {}
    cat_counts = {}

    for i, nid in enumerate(node_ids):
        cat = get_node_category(node_map[nid]['type'])
        if cat not in cat_counts:
            cat_counts[cat] = 0
            cat_to_global_indices[cat] = []
        global_to_local[nid] = (cat, cat_counts[cat])
        cat_to_global_indices[cat].append(i)
        cat_counts[cat] += 1

    # 2. 填充节点特征与位置信息
    for cat, indices in cat_to_global_indices.items():
        data[cat].x = node_embeddings[indices]
        positions = []
        for idx in indices:
            nid = node_ids[idx]
            positions.append([node_map[nid]['start_byte'], node_map[nid]['end_byte']])
        data[cat].node_positions = torch.tensor(positions, dtype=torch.long)

    # 3. 填充异构边
    for i, (src_id, dst_id, rel_label, _) in enumerate(all_edges):
        src_cat, src_idx = global_to_local[src_id]
        dst_cat, dst_idx = global_to_local[dst_id]
        edge_key = (src_cat, rel_label.lower(), dst_cat)

        if not hasattr(data[edge_key], 'edge_index'):
            data[edge_key].edge_index = torch.empty((2, 0), dtype=torch.long)
            data[edge_key].edge_attr = torch.empty((0, edge_embeddings.size(1)), dtype=torch.float)

        new_edge = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long)
        data[edge_key].edge_index = torch.cat([data[edge_key].edge_index, new_edge], dim=1)
        data[edge_key].edge_attr = torch.cat([data[edge_key].edge_attr, edge_embeddings[i].unsqueeze(0)], dim=0)

    return data


def code_to_graph(code: str, tokenizer, model, edge_types=['AST', 'CFG', 'DFG'], code_from_file=False,
                  batch_size=256, edge_cache_path=None, **kwargs) -> HeteroData:
    if code_from_file:
        with open(code, 'r', encoding='utf-8', errors='replace') as f:
            code = f.read()

    parser = initialize_tree_sitter()
    _, node_map = build_node_map(code, parser)
    if not node_map:
        return HeteroData()  # 返回空图处理空代码

    node_map = refine_node_text(node_map)
    node_map = dict(sorted(node_map.items()))

    all_edges = []
    e_types = [t.upper() for t in edge_types]
    if 'CFG' in e_types: all_edges.extend(build_cfg(node_map))
    if 'DFG' in e_types: all_edges.extend(build_dfg(node_map))
    if 'AST' in e_types: all_edges.extend(extract_ast_edges(node_map))

    node_embeddings = encode_nodes(node_map, tokenizer, model, batch_size=batch_size)
    edge_embeddings = encode_edges(all_edges, tokenizer, model, batch_size=batch_size, edge_cache_path=edge_cache_path)

    return build_pyg_hetero_graph(node_map, all_edges, node_embeddings, edge_embeddings)


def save_graph(graph: HeteroData, output_path: str, edge_types: List[str]):
    node_types, model_edge_types = get_model_metadata()
    torch.save({
        'graphs': [graph],
        'metadata': (node_types, model_edge_types),
        'edge_label_to_name': {i: t.upper() for i, t in enumerate(edge_types)}
    }, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Python Code to Hetero-Graph Converter')
    parser.add_argument('--file_path', type=str, default='example.py')
    parser.add_argument('--output', type=str, default='example_graph.pt')
    parser.add_argument('--model_name_or_path', type=str, default='/root/autodl-tmp/unixcoder-base')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--edge_cache_path', type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tokenizer, model = initialize_encoder(args.model_name_or_path, args.device)
    if not os.path.exists(args.file_path):
        with open(args.file_path, 'w') as f:
            f.write("def add(a, b):\n    result = a + b\n    return result")
    target_edges = ['AST', 'CFG', 'DFG']
    graph = code_to_graph(args.file_path, tokenizer, model, edge_types=target_edges,
                          code_from_file=True, batch_size=args.batch_size, edge_cache_path=args.edge_cache_path)
    save_graph(graph, args.output, target_edges)
    print("Graph Construction Completed.")