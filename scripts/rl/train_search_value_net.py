# -*- coding: utf-8 -*-
"""用 search-value labels 训练独立价值网络，可指定输出路径与初始化。

用法：
  PYTHONPATH=. python3 scripts/rl/train_search_value_net.py \
      output/nn_search_value_v3d2_200.npz \
      output/nn_value_model_search_v3d2_200.pt \
      --epochs 80 --batch 256 --lr 1e-3 --hidden 512,256,128 \
      [--init output/nn_value_model_mc.pt]
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
from algo.nn.model import build_model


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
    parser.add_argument('data', help='.npz with X, v')
    parser.add_argument('output', help='output .pt path')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--hidden', default='512,256,128')
    parser.add_argument('--init', default=None, help='optional init checkpoint .pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    data = np.load(args.data)
    X = data['X'].astype(np.float32)
    y = data['v'].astype(np.float32)
    print(f'Loaded {len(X)} samples, feature dim {X.shape[1]}')
    print(f'Value mean={y.mean():.3f} std={y.std():.3f} min={y.min():.3f} max={y.max():.3f}')

    n_val = min(2000, len(X) // 10)
    n_train = len(X) - n_val
    X_train = torch.tensor(X[:n_train], dtype=torch.float32, device=device)
    X_val = torch.tensor(X[n_train:], dtype=torch.float32, device=device)
    y_train = torch.tensor(y[:n_train], dtype=torch.float32, device=device)
    y_val = torch.tensor(y[n_train:], dtype=torch.float32, device=device)
    print(f'Train {n_train}, val {n_val}')

    hidden_dims = [int(x) for x in args.hidden.split(',')]
    model = MahjongValueNetDeep(input_dim=X.shape[1], hidden_dims=hidden_dims).to(device)

    if args.init and os.path.exists(args.init):
        print(f'Initializing from {args.init}')
        try:
            sd = torch.load(args.init, map_location=device)
            if isinstance(sd, dict):
                if 'model_state_dict' in sd:
                    sd = sd['model_state_dict']
                elif 'model_state' in sd:
                    sd = sd['model_state']
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f'Init failed: {e}, training from scratch')

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
            total_loss += float(loss) * len(xb)
        train_loss = total_loss / n
        val_loss = evaluate(model, X_val, y_val, device=device)
        print(f'Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}')
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict().copy()

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
