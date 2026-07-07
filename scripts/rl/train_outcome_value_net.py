# -*- coding: utf-8 -*-
"""用 outcome labels (+1/0/-1) 训练独立 value net。

用法：
  PYTHONPATH=. python3 scripts/rl/train_outcome_value_net.py \
      output/nn_outcome_hybridbest_20k.npz \
      output/nn_value_model_outcome_hybridbest_20k.pt \
      --epochs 60 --batch 512 --lr 1e-3 --wd 1e-4 --hidden 512,256,128
"""

import os
import sys
import time
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from algo.nn.value_model import MahjongValueNetDeep


def evaluate(model, X_val, y_val, batch_size=1024, device='cpu'):
    model.eval()
    criterion = nn.MSELoss(reduction='sum')
    total_loss = 0.0
    total = 0
    n = X_val.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            Xb = X_val[start:end]
            yb = y_val[start:end]
            pred = model(Xb)
            total_loss += float(criterion(pred, yb))
            total += end - start
    return total_loss / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='.npz with X, y')
    parser.add_argument('output', help='output .pt path')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--hidden', default='512,256,128')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    data = np.load(args.data)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.float32)
    print(f'Loaded {len(X)} samples, feature dim {X.shape[1]}')
    print(f'Outcome mean={y.mean():.3f} std={y.std():.3f} win={(y>0).mean():.3f} lose={(y<0).mean():.3f}')

    n_val = min(5000, len(X) // 10)
    n_train = len(X) - n_val
    X_train = torch.tensor(X[:n_train], dtype=torch.float32, device=device)
    X_val = torch.tensor(X[n_train:], dtype=torch.float32, device=device)
    y_train = torch.tensor(y[:n_train], dtype=torch.float32, device=device)
    y_val = torch.tensor(y[n_val:], dtype=torch.float32, device=device)
    print(f'Train {n_train}, val {n_val}')

    hidden_dims = [int(x) for x in args.hidden.split(',')]
    model = MahjongValueNetDeep(input_dim=X.shape[1], hidden_dims=hidden_dims).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    n = len(X_train)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, args.batch):
            idx = perm[start:start + args.batch]
            xb = X_train[idx]
            yb = y_train[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(xb)
        train_loss = total_loss / n
        val_loss = evaluate(model, X_val, y_val, device=device)
        acc = ((pred.sign() == yb.sign()).float().mean()).item() if len(idx) > 0 else 0.0
        print(f'Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} batch_acc={acc:.3f}')
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({'model_state_dict': model.state_dict(),
                'config': {'input_dim': X.shape[1], 'arch': 'deep', 'hidden_dims': hidden_dims,
                           'framework': 'pytorch'}}, args.output)
    cfg_path = args.output.replace('.pt', '_config.json')
    with open(cfg_path, 'w') as f:
        json.dump({'input_dim': X.shape[1], 'arch': 'deep', 'hidden_dims': hidden_dims,
                   'framework': 'pytorch'}, f, indent=2)
    print(f'Done. Best val loss {best_val:.4f}. Saved to {args.output}')


if __name__ == '__main__':
    main()
