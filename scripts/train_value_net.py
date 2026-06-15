# -*- coding: utf-8 -*-
"""训练独立的价值网络（MLX）。"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from algo.nn.value_model import MahjongValueNet


def evaluate(model, X, y_value, batch_size=1024):
    total_loss = 0.0
    total = 0
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Xb = mx.array(X[start:end])
        yv_b = mx.array(y_value[start:end])
        value = model(Xb)
        loss = mx.mean((value - yv_b) ** 2)
        total_loss += float(loss) * (end - start)
        total += end - start
    return total_loss / total


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'output/nn_training_data.npz'
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3

    print(f'Loading data from {data_path} ...')
    data = np.load(data_path)
    X = data['X']
    y_value = data['v'].astype(np.float32)

    n_total = X.shape[0]
    n_val = min(5000, n_total // 10)
    n_train = n_total - n_val
    X_train, X_val = X[:n_train], X[n_train:]
    yv_train, yv_val = y_value[:n_train], y_value[n_train:]

    print(f'Train: {n_train}, Val: {n_val}, features: {X.shape[1]}')

    hidden_dim = int(sys.argv[5]) if len(sys.argv) > 5 else 256
    model = MahjongValueNet(input_dim=X.shape[1], hidden_dim=hidden_dim)
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, Xb, yv_b):
        value = model(Xb)
        return mx.mean((value - yv_b) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    best_val_loss = float('inf')
    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)

    n = X_train.shape[0]
    for epoch in range(1, epochs + 1):
        start = time.time()
        # shuffle
        idx = np.arange(n)
        np.random.shuffle(idx)
        train_loss_sum = 0.0
        batches = 0
        for start_i in range(0, n, batch_size):
            end_i = min(start_i + batch_size, n)
            b = idx[start_i:end_i]
            Xb = mx.array(X_train[b])
            yv_b = mx.array(yv_train[b])
            loss, grads = loss_and_grad(model, Xb, yv_b)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            train_loss_sum += float(loss)
            batches += 1

        val_loss = evaluate(model, X_val, yv_val, batch_size=batch_size)
        elapsed = time.time() - start
        print(f'Epoch {epoch:2d}/{epochs}  '
              f'train_loss={train_loss_sum/batches:.4f}  '
              f'val_loss={val_loss:.4f}  '
              f'time={elapsed:.1f}s')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(os.path.join(out_dir, 'nn_value_model.npz'))
            with open(os.path.join(out_dir, 'nn_value_model_config.json'), 'w') as f:
                json.dump({'input_dim': int(X.shape[1]), 'hidden_dim': hidden_dim}, f)
            print('  -> saved best value model')

    print('Training complete. Best value model at output/nn_value_model.npz')


if __name__ == '__main__':
    main()
