# -*- coding: utf-8 -*-
"""微调 full-action policy 的 value head，用 128k 完整动作数据的状态-结局标签。

思路：
- `TileConvNet` 已有 value_head，但之前主要被当辅助任务训练，质量不足以支持 AWBC；
- 冻结 trunk/policy/response/dealin/tenpai head，只训 value_head；
- 用 discard + response 两个数据流的最终 seat reward 做 MSE 监督；
- 产物 `output/nn_full_action_valueft.pt` 用于后续 AWBC v3 的基线。

用法：
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/rl/finetune_value_head.py \
        output/nn_full_action_data_128000.npz \
        output/nn_full_action_best.pt \
        output/nn_full_action_valueft.pt \
        --epochs 20 --batch 4096 --lr 1e-3 --device cuda
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
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algo.nn.model import build_model


def load_policy(model_path, device):
    cfg_path = model_path.replace('.pt', '_config.json')
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(model_path), 'nn_model_config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f'config not found for {model_path}')
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = build_model(cfg).to(device)
    sd = torch.load(model_path, map_location=device, weights_only=False)
    state = sd.get('model_state', sd)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print('missing keys:', missing[:10])
    if unexpected:
        print('unexpected keys:', unexpected[:10])
    return model, cfg


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # 模型 forward 返回 tuple，取 value（倒数第二个元素）
        out = model(x)
        value = out[1] if isinstance(out, tuple) else out
        if value.dim() > 1:
            value = value.squeeze(-1)
        loss = F.mse_loss(value, y)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        value = out[1] if isinstance(out, tuple) else out
        if value.dim() > 1:
            value = value.squeeze(-1)
        loss = F.mse_loss(value, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_path')
    ap.add_argument('policy_path')
    ap.add_argument('out_model')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--val-ratio', type=float, default=0.05)
    ap.add_argument('--unfreeze-trunk', action='store_true',
                    help='默认只训 value_head；加此选项同时微调 trunk（小 lr 防遗忘）')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f'Loading policy from {args.policy_path}')
    model, cfg = load_policy(args.policy_path, device)

    # 冻结除 value_head 外的所有参数
    for name, p in model.named_parameters():
        if 'value_head' in name or 'value_fc' in name:
            p.requires_grad = True
        else:
            p.requires_grad = args.unfreeze_trunk
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable params: {trainable:,} / {total:,}')

    print(f'Loading data from {args.data_path}')
    data = np.load(args.data_path)
    X = np.concatenate([data['X_discard'], data['X_response']], axis=0)
    y = np.concatenate([data['v_discard'], data['v_response']], axis=0).astype(np.float32)
    print(f'Combined samples: {len(y)}, positive/negative/zero: {(y>0).mean():.3f}/{(y<0).mean():.3f}/{(y==0).mean():.3f}')

    full_ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    n_val = int(len(full_ds) * args.val_ratio)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_state = None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()
        print(f'Epoch {epoch:3d} | train_mse={train_loss:.4f} val_mse={val_loss:.4f}')

        epoch_path = args.out_model.replace('.pt', f'_epoch_{epoch:02d}.pt')
        torch.save({
            'model_state': model.state_dict(),
            'config': cfg,
            'epoch': epoch,
        }, epoch_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.to('cpu').clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out_cfg_path = args.out_model.replace('.pt', '_config.json')
    with open(out_cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    torch.save({'model_state': model.state_dict(), 'config': cfg}, args.out_model)
    print(f'Best val_mse {best_val_loss:.4f}, saved {args.out_model} + {out_cfg_path}')
    print(f'Training time {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
