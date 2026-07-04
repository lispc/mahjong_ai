# -*- coding: utf-8 -*-
"""训练 tile danger 预测模型。

输入：175-dim 公开特征。
输出：34-dim danger map，每个 tile 被任一对手荣和的概率。

标签来自 output/opponent_danger_data_16000.npz，极度稀疏（~0.016%），
因此使用带正样本权重的 BCE，并在评估时关注 AUC / recall。

用法：
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 scripts/rl/train_opponent_danger_model.py \
        output/opponent_danger_data_16000.npz output/opponent_danger_model.pt
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


class DangerConvNet(nn.Module):
    """Conv1d over 34 tile axis + global features, output 34 danger logits."""
    def __init__(self, n_tile_ch=5, channels=64, n_blocks=4, hidden=128, output_dim=34, dropout=0.2):
        super().__init__()
        self.n_tile_ch = n_tile_ch
        self.n_glob = 175 - n_tile_ch * 34
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
        c = self.conv(tiles)
        c = F.adaptive_avg_pool1d(c, 1).squeeze(-1)
        g = self.global_fc(glob)
        h = torch.cat([c, g], dim=-1)
        return self.head(h)


@torch.no_grad()
def evaluate(model, loader, device, pos_weight):
    model.eval()
    total_loss = 0.0
    all_pred = []
    all_y = []
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        total_loss += loss.item()
        all_pred.append(torch.sigmoid(logits).cpu().numpy())
        all_y.append(y.cpu().numpy())
        n += 1
    pred = np.concatenate(all_pred)
    y = np.concatenate(all_y)
    # 计算每个 tile 的 AUC 再平均
    from sklearn.metrics import roc_auc_score
    aucs = []
    for t in range(y.shape[1]):
        if y[:, t].max() == y[:, t].min():
            continue
        aucs.append(roc_auc_score(y[:, t], pred[:, t]))
    avg_auc = float(np.mean(aucs)) if aucs else 0.0
    # 整体 positive recall at top-K
    pred_flat = pred.ravel()
    y_flat = y.ravel()
    k = int(y_flat.sum())
    topk_idx = np.argpartition(pred_flat, -k)[-k:]
    recall_at_k = y_flat[topk_idx].mean() if k > 0 else 0.0
    return {
        'loss': total_loss / max(n, 1),
        'auc': avg_auc,
        'recall_at_k': recall_at_k,
        'pos_rate': y.mean(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_path')
    ap.add_argument('out_model')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--channels', type=int, default=64)
    ap.add_argument('--n-blocks', type=int, default=4)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--pos-weight', type=float, default=None,
                    help='正样本 BCE 权重；默认按 inverse frequency 计算')
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)

    print(f'Loading danger data from {args.data_path}')
    data = np.load(args.data_path)
    X = torch.tensor(data['X'], dtype=torch.float32)
    Y = torch.tensor(data['danger'], dtype=torch.float32)
    print(f'X: {tuple(X.shape)}, Y: {tuple(Y.shape)}, positive rate: {Y.mean():.6f}')

    ds = TensorDataset(X, Y)
    n_val = int(len(ds) * args.val_ratio)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = DangerConvNet(
        n_tile_ch=5, channels=args.channels, n_blocks=args.n_blocks,
        hidden=args.hidden, output_dim=Y.size(1), dropout=args.dropout).to(device)

    pos_weight = args.pos_weight
    if pos_weight is None:
        # inverse frequency: (1 - p) / p
        p = float(Y.mean())
        pos_weight = (1.0 - p) / max(p, 1e-9)
        print(f'Using auto pos_weight={pos_weight:.1f}')
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_state = None
    cfg = {
        'input_dim': X.size(1),
        'output_dim': Y.size(1),
        'arch': 'danger_conv',
        'channels': args.channels,
        'n_blocks': args.n_blocks,
        'hidden': args.hidden,
        'dropout': args.dropout,
        'pos_weight': pos_weight,
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
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n += 1
        scheduler.step()

        val_m = evaluate(model, val_loader, device, pos_weight_tensor)
        print(f'Epoch {epoch:3d} | train_loss={train_loss/max(n,1):.4f} | '
              f'val_loss={val_m["loss"]:.4f} val_auc={val_m["auc"]:.3f} '
              f'recall@k={val_m["recall_at_k"]:.3f} pos={val_m["pos_rate"]:.5f}')

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
