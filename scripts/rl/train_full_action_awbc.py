# -*- coding: utf-8 -*-
"""Advantage-Weighted Behavior Cloning (AWBC) on 128k full-action data。

核心思路：
- 用已训好的 deep value net 估计每个决策前状态的价值 V(s)；
- 用最终 seat reward R 计算优势 A = R - V(s)；
- 对优势高的样本加权（或过滤），做行为克隆微调。

这样可以把 outcome 里的噪声用价值基线抵消，留下“动作确实比预期好”的样本。

用法：
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/rl/train_full_action_awbc.py \
        output/nn_full_action_data_128000.npz \
        output/nn_full_action_best.pt \
        output/nn_value_model_mc.pt \
        output/nn_full_action_awbc.pt \
        --epochs 10 --batch 4096 --lr 5e-5 --weight-temp 1.0 --min-adv -0.2
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algo.nn.model import build_model
from algo.nn.value_model import MahjongValueNetDeep


def load_policy_model(model_path, device):
    cfg_path = model_path.replace('.pt', '_config.json')
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(model_path), 'nn_model_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        cfg = {'arch': 'mlp', 'input_dim': 175, 'hidden_dim': 256}
    model = build_model(cfg).to(device)
    sd = torch.load(model_path, map_location=device, weights_only=False)
    state = sd.get('model_state', sd)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        non_head = [k for k in missing if 'tenpai' not in k and 'dealin' not in k and 'value' not in k]
        if non_head:
            print(f'load_model warning: missing non-optional keys: {non_head[:5]}')
    if unexpected:
        print(f'load_model unexpected keys: {unexpected[:5]}')
    return model, cfg


def load_value_model(model_path, device):
    cfg_path = model_path.replace('.pt', '_config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = MahjongValueNetDeep(cfg['input_dim'], cfg.get('hidden_dims', [512, 256, 128])).to(device)
    sd = torch.load(model_path, map_location=device, weights_only=False)
    state = sd.get('model_state', sd)
    model.load_state_dict(state, strict=False)
    return model, cfg


@torch.no_grad()
def compute_values(net, X, batch_size, device, value_is_policy=False):
    net.eval()
    vals = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).float().to(device)
        out = net(xb)
        if value_is_policy:
            v = out[1] if isinstance(out, tuple) else out
        else:
            v = out
        if v.dim() > 1:
            v = v.squeeze(-1)
        vals.append(v.cpu().numpy())
    return np.concatenate(vals, axis=0).astype(np.float64)


def make_weighted_loader(X, y, reward, values, legal, batch_size, weight_temp, min_adv):
    adv = reward - values
    # 权重：仅保留 advantage >= min_adv 的样本；权重随 advantage 指数增长
    valid = adv >= min_adv
    if valid.sum() == 0:
        print('Warning: no samples satisfy min_adv; falling back to all')
        valid = np.ones(len(adv), dtype=bool)
    weights = np.exp((adv[valid] - adv[valid].min()) / max(weight_temp, 1e-6))
    weights = weights / weights.sum() * len(weights)  # 归一化使平均权重为 1

    X = torch.tensor(X[valid], dtype=torch.float32)
    y = torch.tensor(y[valid], dtype=torch.long)
    w = torch.tensor(weights, dtype=torch.float32)
    if legal is not None:
        legal = torch.tensor(legal[valid], dtype=torch.float32)
        ds = TensorDataset(X, y, w, legal)
    else:
        ds = TensorDataset(X, y, w)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4,
                      pin_memory=True)


def train_epoch(policy, loader_d, loader_r, optimizer, args, device):
    policy.train()
    total_loss = 0.0
    n = 0

    # 先训 discard，再训 response
    for batch in loader_d:
        if len(batch) == 4:
            x, y, w, _ = batch
        else:
            x, y, w = batch
        x, y, w = x.to(device), y.to(device), w.to(device)
        logits = policy(x)[0]
        loss = F.cross_entropy(logits, y, reduction='none')
        loss = (loss * w).mean()
        if args.bc_weight < 1.0:
            loss = loss * (1.0 - args.bc_weight) + F.cross_entropy(logits, y).mean() * args.bc_weight
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n += 1

    for batch in loader_r:
        x, y, w, legal = batch
        x, y, w, legal = x.to(device), y.to(device), w.to(device), legal.to(device)
        logits = policy(x)[-1]
        mask = (legal == 0).float() * -1e9
        loss = F.cross_entropy(logits + mask, y, reduction='none')
        loss = (loss * w).mean()
        if args.bc_weight < 1.0:
            loss = loss * (1.0 - args.bc_weight) + F.cross_entropy(logits + mask, y).mean() * args.bc_weight
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_path')
    ap.add_argument('policy_path')
    ap.add_argument('value_path')
    ap.add_argument('out_model')
    ap.add_argument('--value-is-policy', action='store_true',
                    help='value_path 是与 policy 同架构的模型（如 value head 微调后的 full-action model）')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--weight-temp', type=float, default=1.0,
                    help='advantage 加权温度，越小优势样本权重越高')
    ap.add_argument('--min-adv', type=float, default=-0.2,
                    help='只保留 advantage >= 该值的样本')
    ap.add_argument('--bc-weight', type=float, default=0.1,
                    help='保留多少 vanilla BC 正则（0=纯加权，1=纯BC）')
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f'Loading data from {args.data_path}')
    data = np.load(args.data_path)
    Xd, yd, vd = data['X_discard'], data['y_discard'], data['v_discard']
    Xr, yr, lr, vr = data['X_response'], data['y_response'], data['legal_response'], data['v_response']
    print(f'discard samples: {Xd.shape[0]}, response samples: {Xr.shape[0]}')

    print(f'Loading policy from {args.policy_path}')
    policy, cfg = load_policy_model(args.policy_path, device)
    print(f'Loading value net from {args.value_path}')
    if args.value_is_policy:
        value_net, _ = load_policy_model(args.value_path, device)
    else:
        value_net, vcfg = load_value_model(args.value_path, device)
    for p in value_net.parameters():
        p.requires_grad = False

    print('Computing state values ...')
    t0 = time.time()
    Vd = compute_values(value_net, Xd, args.batch * 4, device,
                        value_is_policy=args.value_is_policy)
    Vr = compute_values(value_net, Xr, args.batch * 4, device,
                        value_is_policy=args.value_is_policy)
    print(f'  discard V mean={Vd.mean():.3f} std={Vd.std():.3f}')
    print(f'  response V mean={Vr.mean():.3f} std={Vr.std():.3f}')
    print(f'  done in {time.time()-t0:.1f}s')

    loader_d = make_weighted_loader(Xd, yd, vd, Vd, None, args.batch,
                                    args.weight_temp, args.min_adv)
    loader_r = make_weighted_loader(Xr, yr, vr, Vr, lr, args.batch,
                                    args.weight_temp, args.min_adv)
    print(f'after filtering: discard batches/epoch={len(loader_d)}, response batches/epoch={len(loader_r)}')

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_metric = float('inf')
    best_state = None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(policy, loader_d, loader_r, optimizer, args, device)
        scheduler.step()
        print(f'Epoch {epoch:3d} | loss={loss:.4f}')

        epoch_path = args.out_model.replace('.pt', f'_epoch_{epoch:02d}.pt')
        torch.save({
            'model_state': policy.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'config': cfg,
        }, epoch_path)
        with open(epoch_path.replace('.pt', '_config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)

        if loss < best_metric:
            best_metric = loss
            best_state = {k: v.to('cpu').clone() for k, v in policy.state_dict().items()}

    if best_state is not None:
        policy.load_state_dict(best_state)

    out_cfg_path = args.out_model.replace('.pt', '_config.json')
    with open(out_cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    torch.save({'model_state': policy.state_dict(), 'config': cfg}, args.out_model)
    print(f'Training finished in {time.time()-t0:.1f}s, best loss {best_metric:.4f}')
    print(f'Saved {args.out_model} + {out_cfg_path}')


if __name__ == '__main__':
    main()
