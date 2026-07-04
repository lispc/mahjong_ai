# -*- coding: utf-8 -*-
"""用 AlphaZero MCTS trace 训练 policy + value。

输入：
    trace.npz: X, visit_dist(34), value
    nn_full_action_data_128000.npz: 用于 response 的 BC 监督

训练目标：
    - discard policy: cross-entropy(softmax(logits), visit_dist)
    - response policy: 原始 hard label CE
    - value head: MSE(value, trace_value)

用法：
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/rl/train_alphazero.py \
        output/alphazero_trace_*.npz \
        output/nn_full_action_data_128000.npz \
        output/nn_full_action_best.pt \
        output/nn_full_action_az.pt
"""
import os
import sys
import time
import json
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algo.nn.model import build_model


def load_model(model_path, device):
    cfg_path = model_path.replace('.pt', '_config.json')
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(model_path), 'nn_model_config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = build_model(cfg).to(device)
    sd = torch.load(model_path, map_location=device, weights_only=False)
    state = sd.get('model_state', sd)
    model.load_state_dict(state, strict=False)
    return model, cfg


def load_trace_data(trace_paths):
    Xs, Vs, Ys = [], [], []
    for p in trace_paths:
        d = np.load(p)
        Xs.append(d['X'])
        Vs.append(d['value'])
        Ys.append(d['visit_dist'])
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    V = np.concatenate(Vs, axis=0).astype(np.float32)
    Y = np.concatenate(Ys, axis=0).astype(np.float32)
    return X, Y, V


def train_epoch(model, loader_discard, loader_response, optimizer, args, device):
    model.train()
    total_loss = 0.0
    n = 0

    # discard: soft target (visit_dist)
    for x, y_soft, v in loader_discard:
        x, y_soft, v = x.to(device), y_soft.to(device), v.to(device)
        optimizer.zero_grad()
        out = model(x)
        discard_logits = out[0]
        value = out[1].squeeze(-1)
        policy_loss = -(F.log_softmax(discard_logits, dim=-1) * y_soft).sum(dim=-1).mean()
        value_loss = F.mse_loss(value, v)
        loss = policy_loss + args.value_weight * value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n += 1

    # response: hard label BC
    for batch in loader_response:
        x, y, legal = batch
        x, y, legal = x.to(device), y.to(device), legal.to(device)
        optimizer.zero_grad()
        out = model(x)
        response_logits = out[-1]
        mask = (legal == 0).float() * -1e9
        loss = F.cross_entropy(response_logits + mask, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader_discard, loader_response, device):
    model.eval()
    total_ploss = 0.0
    total_vloss = 0.0
    n = 0
    for x, y_soft, v in loader_discard:
        x, y_soft, v = x.to(device), y_soft.to(device), v.to(device)
        out = model(x)
        discard_logits = out[0]
        value = out[1].squeeze(-1)
        ploss = -(F.log_softmax(discard_logits, dim=-1) * y_soft).sum(dim=-1).mean()
        vloss = F.mse_loss(value, v)
        total_ploss += ploss.item()
        total_vloss += vloss.item()
        n += 1
    r_loss = 0.0
    m = 0
    for batch in loader_response:
        x, y, legal = batch
        x, y, legal = x.to(device), y.to(device), legal.to(device)
        out = model(x)
        response_logits = out[-1]
        mask = (legal == 0).float() * -1e9
        r_loss += F.cross_entropy(response_logits + mask, y).item()
        m += 1
    return {
        'policy_loss': total_ploss / max(n, 1),
        'value_loss': total_vloss / max(n, 1),
        'response_loss': r_loss / max(m, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace_paths', nargs='+')
    ap.add_argument('response_data')
    ap.add_argument('init_model')
    ap.add_argument('out_model')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--value-weight', type=float, default=1.0)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--response-samples', type=int, default=1_000_000,
                    help='从 response 数据中随机采样的最大样本数（用于和 trace 平衡）')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)

    # 展开可能的 glob
    trace_paths = []
    for p in args.trace_paths:
        if '*' in p:
            trace_paths.extend(sorted(glob.glob(p)))
        else:
            trace_paths.append(p)
    print(f'Loading traces from {trace_paths}')
    Xd, Yd, Vd = load_trace_data(trace_paths)
    print(f'trace samples: {len(Xd)}')

    print(f'Loading response data from {args.response_data}')
    rd = np.load(args.response_data)
    Xr_full, yr_full, lr_full = rd['X_response'], rd['y_response'], rd['legal_response']
    if len(yr_full) > args.response_samples:
        idx = np.random.choice(len(yr_full), args.response_samples, replace=False)
        Xr_full, yr_full, lr_full = Xr_full[idx], yr_full[idx], lr_full[idx]
    Xr, yr, lr = Xr_full, yr_full, lr_full
    print(f'response samples: {len(yr)}')

    print(f'Loading init model from {args.init_model}')
    model, cfg = load_model(args.init_model, device)

    # discard dataset
    ds_d = TensorDataset(torch.tensor(Xd), torch.tensor(Yd), torch.tensor(Vd))
    n_val_d = int(len(ds_d) * args.val_ratio)
    n_train_d = len(ds_d) - n_val_d
    train_d, val_d = random_split(ds_d, [n_train_d, n_val_d],
                                  generator=torch.Generator().manual_seed(42))
    loader_d = DataLoader(train_d, batch_size=args.batch, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_loader_d = DataLoader(val_d, batch_size=args.batch, shuffle=False,
                              num_workers=4, pin_memory=True)

    # response dataset
    ds_r = TensorDataset(torch.tensor(Xr, dtype=torch.float32),
                         torch.tensor(yr, dtype=torch.long),
                         torch.tensor(lr, dtype=torch.float32))
    n_val_r = int(len(ds_r) * args.val_ratio)
    n_train_r = len(ds_r) - n_val_r
    train_r, val_r = random_split(ds_r, [n_train_r, n_val_r],
                                  generator=torch.Generator().manual_seed(42))
    loader_r = DataLoader(train_r, batch_size=args.batch, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_loader_r = DataLoader(val_r, batch_size=args.batch, shuffle=False,
                              num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float('inf')
    best_state = None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loader_d, loader_r, optimizer, args, device)
        val_m = evaluate(model, val_loader_d, val_loader_r, device)
        scheduler.step()
        print(f'Epoch {epoch:3d} | train_loss={train_loss:.4f} | '
              f'val_policy={val_m["policy_loss"]:.4f} val_value={val_m["value_loss"]:.4f} val_response={val_m["response_loss"]:.4f}')

        epoch_path = args.out_model.replace('.pt', f'_epoch_{epoch:02d}.pt')
        torch.save({'model_state': model.state_dict(), 'config': cfg, 'epoch': epoch}, epoch_path)

        val_sum = val_m['policy_loss'] + val_m['value_loss'] + val_m['response_loss']
        if val_sum < best_val:
            best_val = val_sum
            best_state = {k: v.to('cpu').clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out_cfg_path = args.out_model.replace('.pt', '_config.json')
    with open(out_cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    torch.save({'model_state': model.state_dict(), 'config': cfg}, args.out_model)
    print(f'Training finished in {time.time()-t0:.1f}s, best val sum {best_val:.4f}')
    print(f'Saved {args.out_model} + {out_cfg_path}')


if __name__ == '__main__':
    main()
