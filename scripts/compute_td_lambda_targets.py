#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""计算 TD(λ) target，用于训练 value net。

输入：
  - selfplay_td_*.pkl：每局一个 trajectory dict，含 samples 列表
  - 当前 value model（默认 output/nn_value_model_mc.pt）

输出：
  - .npz 文件，含 X (N, 175), y (N,), v (N,) TD(λ) target

性能：
  - V(s) 推理：GPU 批量，5000 局 < 5 分钟
  - TD target 计算：向量化 numpy
  - 多 GPU 并行：按 trajectory 分片到 4 GPU
"""
import sys
import os
import time
import pickle
import argparse
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from algo.nn.value_model import MahjongValueNetDeep


def load_value_model(model_path, config_path, device):
    with open(config_path) as f:
        import json
        cfg = json.load(f)
    model = MahjongValueNetDeep(
        input_dim=cfg['input_dim'],
        hidden_dims=cfg.get('hidden_dims', [512, 256, 128]),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def batch_inference(model, features_list, device, batch_size=4096):
    """批量推理 V(s)，返回 numpy array shape (N,)。"""
    n = len(features_list)
    out = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X = torch.tensor(np.stack(features_list[start:end]),
                             dtype=torch.float32, device=device)
            v = model(X).squeeze(-1).cpu().numpy()
            out[start:end] = v
    return out


def compute_td_targets_for_game(V_preds, outcome, lambda_):
    """向量化计算单局 TD(λ) target。

    V_preds: np.ndarray shape (T,) 每个 step 的 V(s_t) 预测
    outcome: 终局 +1/0/-1
    返回 np.ndarray shape (T,)，已 clip 到 [-1, 1]
    """
    T = len(V_preds)
    if T == 0:
        return np.zeros(0, dtype=np.float32)
    if T == 1:
        # 只有一局一步，纯 outcome
        return np.array([outcome], dtype=np.float32)

    # G_t^λ = (1-λ) * Σ_{n=1}^{T-t-1} λ^{n-1} V(s_{t+n}) + λ^{T-t-1} * outcome
    #
    # 设 w[k] = (1-λ) * λ^k for k = 0 .. T-2
    # 则对 t 来说：
    #   G_t^λ = Σ_{k=0}^{T-t-2} w[k] * V[s_{t+1+k}]  +  λ^{T-t-1} * outcome
    #        = (w[0..T-t-2] · V[s_{t+1..T-1}]) + λ^{T-t-1} * outcome
    #
    # 用 numpy cumsum/convolution 加速：
    #   inner_sum[t] = Σ_{k=0}^{T-t-2} w[k] * V[t+1+k]
    # 这等价于 (V * w_flip) 的反向 prefix sum，但简单实现是 double loop，
    # 单局 ≤30 步，开销可忽略。

    powers = lambda_ ** np.arange(T, dtype=np.float64)  # λ^k
    targets = np.zeros(T, dtype=np.float32)

    # 向量化：对每个 t，target = Σ_{k=0}^{T-t-2} (1-λ) λ^k V[t+1+k] + λ^{T-t-1} outcome
    # 改写：令 W = (1-λ) * λ^k for k=0..T-2
    # 对 t，求和范围 k=0..T-t-2，对应 V index t+1..T-1
    # 即 targets[t] = W[:T-t-1] · V[t+1:T] + powers[T-t-1] * outcome

    w = (1 - lambda_) * powers  # w[k] = (1-λ) * λ^k
    for t in range(T):
        n_v_terms = T - t - 1
        if n_v_terms > 0:
            # V[t+1..T-1] 长度 n_v_terms，对应权重 w[0..n_v_terms-1]
            targets[t] = float(np.dot(w[:n_v_terms], V_preds[t + 1:T]))
        # terminal
        targets[t] += float(powers[n_v_terms]) * outcome

    return np.clip(targets, -1.0, 1.0).astype(np.float32)


def process_shard(args):
    """处理一个 trajectory 分片：批量推理 + 算 TD target，返回 (X, y, v)。"""
    shard_path, model_path, config_path, lambda_, gpu_id, out_path = args
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    with open(shard_path, 'rb') as f:
        trajectories = pickle.load(f)

    # 1. 收集所有 features 做批量推理
    all_features = []
    bounds = []  # (start, end) 每局在 all_features 中的位置
    outcomes = []
    cursor = 0
    for traj in trajectories:
        samples = traj['samples']
        n = len(samples)
        if n == 0:
            continue
        for s in samples:
            all_features.append(s['features'])
        bounds.append((cursor, cursor + n, traj['outcome']))
        cursor += n

    if cursor == 0:
        return out_path, 0, 0.0

    # 2. 批量推理 V(s)
    model = load_value_model(model_path, config_path, device)
    V_all = batch_inference(model, all_features, device)

    # 3. 每局算 TD target
    all_targets = np.zeros(cursor, dtype=np.float32)
    all_actions = np.zeros(cursor, dtype=np.int64)
    for i, (start, end, outcome) in enumerate(bounds):
        V_traj = V_all[start:end]
        targets = compute_td_targets_for_game(V_traj, outcome, lambda_)
        all_targets[start:end] = targets
        for j, s in enumerate(trajectories[i]['samples']):
            all_actions[start + j] = s['action']

    X = np.stack(all_features)
    y = all_actions
    v = all_targets
    np.savez_compressed(out_path, X=X, y=y, v=v)
    return out_path, len(v), 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='selfplay_td_*.pkl (merged) or shard')
    parser.add_argument('output', help='output .npz path')
    parser.add_argument('--model', default='output/nn_value_model_mc.pt')
    parser.add_argument('--config', default='output/nn_value_model_mc_config.json')
    parser.add_argument('--lambda_', type=float, default=0.5)
    parser.add_argument('--shards', type=int, default=4,
                        help='split trajectories into N shards for multi-GPU')
    parser.add_argument('--shard_dir', default=None,
                        help='dir for shard files (default: /tmp/td_shards_<pid>)')
    args = parser.parse_args()

    print(f'Loading trajectories from {args.input} ...', flush=True)
    with open(args.input, 'rb') as f:
        trajectories = pickle.load(f)
    print(f'  {len(trajectories)} trajectories', flush=True)

    # 分片
    shard_dir = args.shard_dir or f'/tmp/td_shards_{os.getpid()}'
    os.makedirs(shard_dir, exist_ok=True)
    n_per_shard = (len(trajectories) + args.shards - 1) // args.shards
    shard_paths = []
    for i in range(args.shards):
        shard = trajectories[i * n_per_shard:(i + 1) * n_per_shard]
        p = os.path.join(shard_dir, f'shard_{i}.pkl')
        with open(p, 'wb') as f:
            pickle.dump(shard, f)
        shard_paths.append(p)
        print(f'  shard {i}: {len(shard)} trajectories -> {p}', flush=True)

    # 多 GPU 并行处理
    print(f'Computing TD(λ={args.lambda_}) targets on {args.shards} GPUs ...', flush=True)
    start = time.time()
    shard_outs = [os.path.join(shard_dir, f'shard_{i}.npz') for i in range(args.shards)]
    task_args = [(shard_paths[i], args.model, args.config, args.lambda_,
                  i % torch.cuda.device_count() if torch.cuda.is_available() else 0,
                  shard_outs[i])
                 for i in range(args.shards)]

    # 用 multiprocessing 跑多 GPU（每个 worker 绑定一个 GPU）
    # 注意：CUDA 默认 fork 不安全，用 spawn
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    with mp.Pool(args.shards) as pool:
        results = pool.map(process_shard, task_args)

    elapsed = time.time() - start
    total_samples = sum(r[1] for r in results)
    print(f'TD target computed in {elapsed:.1f}s, {total_samples} samples', flush=True)

    # 合并 shards
    Xs, ys, vs = [], [], []
    for out_path, _, _ in results:
        d = np.load(out_path)
        Xs.append(d['X'])
        ys.append(d['y'])
        vs.append(d['v'])
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    v = np.concatenate(vs)
    np.savez_compressed(args.output, X=X, y=y, v=v)
    print(f'Saved {args.output}: X{X.shape} y{y.shape} v{v.shape}', flush=True)
    print(f'V target stats: mean={v.mean():.3f}, std={v.std():.3f}, '
          f'min={v.min():.3f}, max={v.max():.3f}', flush=True)

    # 清理 shards
    import shutil
    shutil.rmtree(shard_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
