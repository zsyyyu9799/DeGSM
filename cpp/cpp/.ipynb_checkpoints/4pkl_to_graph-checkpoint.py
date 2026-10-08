import argparse
# 修改为导入 cpp_code_graph 模块
from cpp_code_graph import initialize_encoder, code_to_graph, get_model_metadata
import pandas as pd
import torch
import os
import sys
import gc
import shutil
import multiprocessing
from tqdm import tqdm

COLUMN_NAME = 'cleaned_code'

# --- Worker 辅助函数 ---
def worker_fn(worker_id, device, data_slice_df, output_path_slice, model_name_or_path_abs, batch_size, edge_types_list,
              edge_cache_path_worker, edge_id_to_type, current_column_name_val, start_position):
    """
    分片处理函数：具备错误日志拦截与显存定期回收。
    """
    print(f"Worker {worker_id} (on {device}) started. {len(data_slice_df)} items.")

    tokenizer, code_encoder = initialize_encoder(model_name_or_path_abs, device)
    graphs = []

    SAVE_INTERVAL = 200
    batch_idx = 0
    base_path, ext = os.path.splitext(output_path_slice)

    try:
        for position, (_, row) in enumerate(
                tqdm(data_slice_df.iterrows(), total=len(data_slice_df),
                     desc=f"Worker {worker_id} ({device})", position=worker_id)):

            global_position = start_position + position
            code = row[current_column_name_val]

            # 跳过空数据
            if not isinstance(code, str) or len(code.strip()) == 0:
                continue

            try:
                graph = code_to_graph(code, tokenizer, code_encoder,
                                      edge_types=edge_types_list,
                                      batch_size=batch_size,
                                      edge_cache_path=edge_cache_path_worker)
                graph.idx = global_position
                graphs.append(graph)
            except Exception as e:
                with open(f"worker_{worker_id}_error.log", "a") as ef:
                    ef.write(f"Error at idx {global_position}: {str(e)}\n")

            if (position + 1) % SAVE_INTERVAL == 0:
                current_batch_path = f"{base_path}_batch_{batch_idx}{ext}"
                torch.save({
                    'graphs': graphs,
                    'edge_label_to_name': edge_id_to_type
                }, current_batch_path)

                batch_idx += 1
                del graphs
                graphs = []
                gc.collect()
                if 'cuda' in device:
                    torch.cuda.empty_cache()

    except Exception as e:
        print(f"Worker {worker_id} critical: {e}")
    finally:
        if graphs:
            current_batch_path = f"{base_path}_batch_{batch_idx}{ext}"
            torch.save({
                'graphs': graphs,
                'edge_label_to_name': edge_id_to_type
            }, current_batch_path)

    # 释放显存
    del code_encoder
    if 'cuda' in device:
        torch.cuda.empty_cache()
    return output_path_slice

def parse_args():
    parser = argparse.ArgumentParser(description='.pkl dataset to PyG Hetero-Graph dataset for C++')
    # 默认路径修改为 C++ 的路径结构
    parser.add_argument('--pkl_path', type=str,
                        default=r'/root/autodl-tmp/first_paper/code_dataset/cpp/cpp_data.pkl')
    parser.add_argument('--output_path', type=str, default=r'/root/autodl-tmp/first_paper/graph_dataset/cpp/processed_dataset.pt')
    parser.add_argument('--model_name_or_path', type=str, default='/root/autodl-tmp/unixcoder-base')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--edge_types', type=str, default='AST,CFG,DFG')
    parser.add_argument('--slice_num', type=int, default=None)
    parser.add_argument('--column_name', type=str, default=None)
    return parser.parse_args()

def process_all(args):
    """全数据集并行处理主函数"""
    if 'cuda' in args.device and torch.cuda.is_available():
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                pass

    model_name_or_path = args.model_name_or_path
    simple_model_name = os.path.basename(model_name_or_path).split('-')[0]

    main_cache_dir = os.path.join(os.getcwd(), f"cache_enc_{simple_model_name}")
    os.makedirs(main_cache_dir, exist_ok=True)
    main_edge_cache_path = os.path.join(main_cache_dir, f"{simple_model_name}_edge_embeddings_cache.pt")

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading input file: {args.pkl_path}")
    if args.pkl_path.endswith('.csv'):
        df = pd.read_csv(args.pkl_path)
    else:
        df = pd.read_pickle(args.pkl_path)
    df = df.reset_index(drop=True)
    total_size = len(df)
    print(f"Total C++ dataset size: {total_size}")

    edge_types_list = args.edge_types.split(',')
    edge_id_to_type = {i: t.upper() for i, t in enumerate(edge_types_list)}

    if torch.cuda.is_available() and 'cuda' in args.device:
        devices_to_cycle = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    else:
        devices_to_cycle = ['cpu']

    slice_num = args.slice_num or (len(devices_to_cycle) if devices_to_cycle else 1)
    slice_num = min(slice_num, total_size)
    slice_size = (total_size + slice_num - 1) // slice_num

    temp_folder_name = "graph_slices_" + os.path.splitext(os.path.basename(args.output_path))[0]
    temp_output_dir = os.path.join(output_dir, temp_folder_name)
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)
    os.makedirs(temp_output_dir, exist_ok=True)

    curr_col = args.column_name or COLUMN_NAME
    worker_args_list = []

    for i in range(slice_num):
        start_idx = i * slice_size
        end_idx = min((i + 1) * slice_size, total_size)
        if start_idx >= end_idx: continue

        data_slice_df = df.iloc[start_idx:end_idx]
        assigned_device = devices_to_cycle[i % len(devices_to_cycle)]
        temp_path = os.path.join(temp_output_dir, f"slice_{i}.pt")

        worker_args_list.append((
            i, assigned_device, data_slice_df, temp_path,
            model_name_or_path,
            args.batch_size, edge_types_list,
            main_edge_cache_path, edge_id_to_type, curr_col, start_idx
        ))

    print(f"Starting {len(worker_args_list)} workers for C++ processing...")
    try:
        if len(worker_args_list) == 1:
            worker_fn(*worker_args_list[0])
        else:
            pool_processes = len(worker_args_list)
            with multiprocessing.Pool(processes=pool_processes) as pool:
                pool.starmap(worker_fn, worker_args_list)
    except KeyboardInterrupt:
        print("\n[Main] User interrupted.")

    print("\nStarting merge process...")
    all_graphs = []
    checkpoint_files = []
    for root, _, files in os.walk(temp_output_dir):
        for f in files:
            if f.endswith('.pt'):
                checkpoint_files.append(os.path.join(root, f))

    for f_path in tqdm(checkpoint_files, desc="Merging files"):
        try:
            data = torch.load(f_path, map_location='cpu', weights_only=False)
            if 'graphs' in data:
                all_graphs.extend(data['graphs'])
        except Exception as e:
            print(f"Error loading {f_path}: {e}")

    if all_graphs:
        print(f"Sorting {len(all_graphs)} graphs...")
        all_graphs.sort(key=lambda g: getattr(g, 'idx', 0))

        torch.save({
            'graphs': all_graphs,
            'edge_label_to_name': edge_id_to_type
        }, args.output_path)
        print(f"\nSUCCESS: Saved {len(all_graphs)} C++ Hetero-graphs to {args.output_path}")

        node_types, edge_types = get_model_metadata()
        metadata_save_path = os.path.join(output_dir, "graph_metadata.pt")
        torch.save({
            'node_types': node_types,
            'edge_types': edge_types,
            'metadata': (node_types, edge_types)
        }, metadata_save_path)
        print(f"Metadata file saved to: {metadata_save_path}")

        if len(all_graphs) == total_size:
            print("Full dataset processed. Cleaning up temporary directory.")
            shutil.rmtree(temp_output_dir)
    else:
        print("FAILED: No graphs were processed.")


if __name__ == '__main__':
    args = parse_args()
    process_all(args)