# -*- coding: utf-8 -*-
"""用 Conv1d 架构训练对手听牌预测模型。

输入结构与完整动作 policy 类似：前 170 维 reshaped 成 (5,34) 做 tile conv，
后 5 维作为全局特征。输出 3 个听牌 logits。

用法：
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 scripts/rl/train_opponent_model_conv.py \
        output/opponent_model_data_16000.npz output/opponent_model_conv.pt
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


class OpponentConvNet(nn.Module):
    def __init__(self, n_tile_ch=5, channels=64, n_blocks=4, hidden=128, output_dim=3, dropout=0.2):
        super().__init__()
        self.n_tile_ch = n_tile_ch
        self.n_glob = 175 - n_tile_ch * 34  # 5
        blocks = []
        in_ch = n_tile_ch
        for _ in range(n_blocks):
            blocks += [
                nn.Conv1d(in_ch, channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(channels),
            ]
            in_ch = channels
        self.conv = nn.Sequential(*blocks)
        self.global_fc = nn.Sequential(
            nn.Linear(self.n_glob, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels + 32, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        b = x.size(0)
        tiles = x[:, :self.n_tile_ch * 34].reshape(b, self.n_tile_ch, 34)
        glob = x[:, self.n_tile_ch * 34:]
        c = self.conv(tiles)          # (B, channels, 34)
        c = F.adaptive_avg_pool1d(c, 1).squeeze(-1)  # (B, channels)
        g = self.global_fc(glob)      # (B, 32)
        h = torch.cat([c, g], dim=-1)
        return self.head(h)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_pos = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        total_loss += loss.item()
        pred = (torch.sigmoid(logits) > 0.5).float()
        total_acc += (pred == y).float().mean().item()
        total_pos += y.mean().item()
        n += 1
    return {
        'loss': total_loss / max(n, 1),
        'acc': total_acc / max(n, 1),
        'pos_rate': total_pos / max(n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_path')
    ap.add_argument('out_model')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--channels', type=int, default=64)
    ap.add_argument('--n-blocks', type=int, default=4)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)

    print(f'Loading data from {args.data_path}')
    data = np.load(args.data_path)
    X = torch.tensor(data['X'], dtype=torch.float32)
    Y = torch.tensor(data['opp_tenpai'], dtype=torch.float32)
    print(f'X: {tuple(X.shape)}, Y: {tuple(Y.shape)}, positive rate: {Y.mean(dim=0).numpy()}')

    ds = TensorDataset(X, Y)
    n_val = int(len(ds) * args.val_ratio)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = OpponentConvNet(
        n_tile_ch=5, channels=args.channels, n_blocks=args.n_blocks,
        hidden=args.hidden, output_dim=Y.size(1), dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_state = None
    cfg = {
        'input_dim': X.size(1),
        'output_dim': Y.size(1),
        'arch': 'opponent_conv',
        'channels': args.channels,
        'n_blocks': args.n_blocks,
        'hidden': args.hidden,
        'dropout': args.dropout,
        'framework': 'pytorch',
    }

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n += 1
        scheduler.step()

        val_m = evaluate(model, val_loader, device)
        print(f'Epoch {epoch:3d} | train_loss={train_loss/max(n,1):.4f} | '
              f'val_loss={val_m["loss"]:.4f} val_acc={val_m["acc"]:.3f} val_pos={val_m["pos_rate"]:.3f}')

        if val_m['loss'] < best_val_loss:
            best_val_loss = val_m['loss']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out_cfg_path = args.out_model.replace('.pt', '_config.json')
    with open(out_cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    torch.save({'model_state': model.state_dict(), 'config': cfg}, args.out_model)
    print(f'Training finished in {time.time() - t0:.1f}s, best val_loss {best_val_loss:.4f}')
    print(f'Saved {args.out_model} + {out_cfg_path}')


if __name__ == '__main__':
    main()
