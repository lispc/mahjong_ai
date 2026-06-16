#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练 Rollout Policy Net（蒸馏 legacy eval2）。"""
import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from algo.nn.rollout_policy import RolloutPolicyNet


def evaluate(model, X_val, y_val, batch_size=1024):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.0
    correct = 0
    total = 0
    n = X_val.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = X_val[start:end]
            yb = y_val[start:end]
            logits = model(xb)
            total_loss += float(criterion(logits, yb))
            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == yb).sum())
            total += end - start
    return total_loss / total, correct / total


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'output/nn_training_data_rollout_policy.npz'
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
    hidden_dims_str = sys.argv[5] if len(sys.argv) > 5 else '512,256,128'
    hidden_dims = [int(x) for x in hidden_dims_str.split(',')]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print(f'Loading data from {data_path} ...')
    data = np.load(data_path)
    X = data['X']
    y = data['y']

    n_total = X.shape[0]
    n_val = min(5000, n_total // 10)

    X_train = torch.tensor(X[:n_total - n_val], dtype=torch.float32, device=device)
    X_val = torch.tensor(X[n_total - n_val:], dtype=torch.float32, device=device)
    y_train = torch.tensor(y[:n_total - n_val], dtype=torch.long, device=device)
    y_val = torch.tensor(y[n_total - n_val:], dtype=torch.long, device=device)

    print(f'Train: {len(y_train)}, Val: {len(y_val)}, features: {X.shape[1]}')

    model = RolloutPolicyNet(input_dim=X.shape[1], hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        perm = torch.randperm(X_train.shape[0], device=device)
        train_loss_sum = 0.0
        batches = 0
        for i in range(0, X_train.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss)
            batches += 1

        val_loss, val_acc = evaluate(model, X_val, y_val, batch_size=batch_size)
        elapsed = time.time() - start
        print(f'Epoch {epoch:2d}/{epochs}  '
              f'train_loss={train_loss_sum/batches:.4f}  '
              f'val_loss={val_loss:.4f}  '
              f'val_acc={val_acc:.3f}  '
              f'time={elapsed:.1f}s')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(out_dir, 'nn_rollout_policy.pt'))
            with open(os.path.join(out_dir, 'nn_rollout_policy_config.json'), 'w') as f:
                json.dump({'input_dim': int(X.shape[1]), 'hidden_dims': hidden_dims,
                           'framework': 'pytorch'}, f)
            print('  -> saved best rollout policy model')

    print(f'Training complete. Best val_acc={best_val_acc:.3f}')


if __name__ == '__main__':
    main()
