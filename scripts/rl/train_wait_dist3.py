# -*- coding: utf-8 -*-
"""训练 102 维待牌分布头（wait_dist3_head）。

从自对弈数据加载 (features, wait_label) 样本，在 current best backbone
上新增 wait_dist3_head 并只训练该 head。

用法：
    PYTHONPATH=. python3 scripts/rl/train_wait_dist3.py \
        output/wait_dist3_labels_10000.npz \
        output/nn_full_action_best.pt \
        output/nn_wait_dist3_10k.pt \
        --epochs 60 --batch 1024 --lr 5e-4
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.nn.model import build_model


_NUM_TILES = 34


def _load_data(path):
    with np.load(path) as data:
        samples = pickle.loads(bytes(data['samples']))
    X = np.stack([s['features'] for s in samples], axis=0).astype(np.float32)
    Y = np.stack([s['wait_label'] for s in samples], axis=0).astype(np.float32)
    return X, Y


def _recall_per_opp(Y, preds, k):
    """计算每个对手的 recall@k 并取平均。"""
    recalls = []
    for off in range(3):
        s = off * _NUM_TILES
        e = s + _NUM_TILES
        y_opp = Y[:, s:e]
        p_opp = preds[:, s:e]
        topk = np.argsort(-p_opp, axis=1)[:, :k]
        hits = 0
        total = 0
        for i in range(len(y_opp)):
            true_idx = set(np.where(y_opp[i] > 0.5)[0])
            pred_idx = set(topk[i])
            hits += len(true_idx & pred_idx)
            total += max(len(true_idx), 1)
        recalls.append(hits / total if total > 0 else 0.0)
    return np.mean(recalls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='.npz with pickled samples')
    parser.add_argument('base_model', help='current best .pt')
    parser.add_argument('output_model', help='output .pt path')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--train-ratio', type=float, default=0.9)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    cfg_path = args.base_model.replace('.pt', '_config.json')
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(args.base_model), 'nn_model_config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg['wait_dist3_head'] = True

    model = build_model(cfg).to(device)
    sd = torch.load(args.base_model, map_location=device)
    if isinstance(sd, dict):
        if 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        elif 'model_state' in sd:
            sd = sd['model_state']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print('Missing keys (expected for new wait_dist3_head):', missing[:10])
    if unexpected:
        print('Unexpected keys:', unexpected[:10])

    # Freeze backbone, only train wait_dist3 head
    for name, param in model.named_parameters():
        if 'wait_dist3' not in name:
            param.requires_grad = False
        else:
            print('Trainable:', name)

    X, Y = _load_data(args.data)
    n = len(X)
    n_train = int(n * args.train_ratio)
    perm = np.random.permutation(n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    print(f'Data: {n} total, {len(train_idx)} train, {len(val_idx)} val')
    print(f'Positive rate per tile: {Y.mean(axis=0).round(3)}')

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(Y_val))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.wd)
    pos_weight = torch.tensor((1.0 - Y.mean()) / Y.mean(), dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            wait_logits = out[-1]
            loss = criterion(wait_logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        train_loss = total_loss / len(train_idx)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_gts = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                wait_logits = out[-1]
                loss = criterion(wait_logits, yb)
                val_loss += loss.item() * len(xb)
                val_preds.append(torch.sigmoid(wait_logits).cpu().numpy())
                val_gts.append(yb.cpu().numpy())
        val_loss /= len(val_idx)
        val_preds = np.concatenate(val_preds)
        val_gts = np.concatenate(val_gts)

        r1 = _recall_per_opp(val_gts, val_preds, 1)
        r2 = _recall_per_opp(val_gts, val_preds, 2)
        r3 = _recall_per_opp(val_gts, val_preds, 3)
        r5 = _recall_per_opp(val_gts, val_preds, 5)

        print(f'Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f} '
              f'val_loss={val_loss:.4f} per-opp-recall@1/2/3/5='
              f'{r1:.3f}/{r2:.3f}/{r3:.3f}/{r5:.3f}')

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': cfg,
            }, args.output_model)
            print('  saved best')

    # Save config next to model
    out_cfg_path = args.output_model.replace('.pt', '_config.json')
    with open(out_cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'Done. Best val loss {best_val:.4f}. Model saved to {args.output_model}')


if __name__ == '__main__':
    main()
