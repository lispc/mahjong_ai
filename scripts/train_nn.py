#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练轻量 Policy-Value 网络（PyTorch）。"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from algo.nn.model import MahjongNet


def evaluate(model, X_val, yp_val, yv_val, batch_size=1024, device='cuda'):
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    correct = 0
    total = 0
    n = X_val.shape[0]
    criterion = nn.CrossEntropyLoss(reduction='sum')
    mse = nn.MSELoss(reduction='sum')

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            Xb = X_val[start:end]
            yp_b = yp_val[start:end]
            yv_b = yv_val[start:end]

            logits, value = model(Xb)
            policy_loss = criterion(logits, yp_b)
            value_loss = mse(value.squeeze(-1), yv_b)
            loss = policy_loss + 0.5 * value_loss

            total_loss += float(loss)
            total_policy_loss += float(policy_loss)
            total_value_loss += float(value_loss)
            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == yp_b).sum())
            total += end - start

    return {
        'loss': total_loss / total,
        'policy_loss': total_policy_loss / total,
        'value_loss': total_value_loss / total,
        'acc': correct / total,
    }


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'output/nn_training_data.npz'
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
    hidden_dim = int(sys.argv[5]) if len(sys.argv) > 5 else 128
    wd = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print(f'Loading data from {data_path} ...')
    data = np.load(data_path)
    X = data['X']
    y_policy = data['y']
    if 'v' in data:
        y_value = data['v'].astype(np.float32)
    else:
        y_value = np.zeros_like(y_policy, dtype=np.float32)

    n_total = X.shape[0]
    n_val = min(5000, n_total // 10)
    n_train = n_total - n_val

    X_train = torch.tensor(X[:n_train], dtype=torch.float32, device=device)
    X_val = torch.tensor(X[n_train:], dtype=torch.float32, device=device)
    yp_train = torch.tensor(y_policy[:n_train], dtype=torch.long, device=device)
    yp_val = torch.tensor(y_policy[n_train:], dtype=torch.long, device=device)
    yv_train = torch.tensor(y_value[:n_train], dtype=torch.float32, device=device)
    yv_val = torch.tensor(y_value[n_train:], dtype=torch.float32, device=device)

    print(f'Train: {n_train}, Val: {n_val}, features: {X.shape[1]}')

    model = MahjongNet(input_dim=X.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    best_val_loss = float('inf')
    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            Xb, yp_b, yv_b = X_train[idx], yp_train[idx], yv_train[idx]

            optimizer.zero_grad()
            logits, value = model(Xb)
            policy_loss = criterion_policy(logits, yp_b)
            value_loss = criterion_value(value.squeeze(-1), yv_b)
            loss = policy_loss + 0.5 * value_loss
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss)
            train_batches += 1

        val_metrics = evaluate(model, X_val, yp_val, yv_val, batch_size=batch_size, device=device)
        elapsed = time.time() - start
        print(f'Epoch {epoch:2d}/{epochs}  '
              f'train_loss={train_loss_sum/train_batches:.4f}  '
              f'val_loss={val_metrics["loss"]:.4f}  '
              f'val_policy={val_metrics["policy_loss"]:.4f}  '
              f'val_value={val_metrics["value_loss"]:.4f}  '
              f'val_acc={val_metrics["acc"]:.3f}  '
              f'time={elapsed:.1f}s')

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), os.path.join(out_dir, 'nn_model.pt'))
            with open(os.path.join(out_dir, 'nn_model_config.json'), 'w') as f:
                json.dump({'input_dim': int(X.shape[1]), 'hidden_dim': hidden_dim,
                           'framework': 'pytorch'}, f)
            print('  -> saved best model')

    print('Training complete. Best model at output/nn_model.pt')


if __name__ == '__main__':
    main()
