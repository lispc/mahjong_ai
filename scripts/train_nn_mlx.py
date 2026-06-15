# -*- coding: utf-8 -*-
"""训练轻量 Policy-Value 网络（MLX）。"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from algo.nn.model import MahjongNet


def batch_iterator(X, y_policy, y_value, batch_size, shuffle=True):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b = idx[start:end]
        yield (mx.array(X[b]),
               mx.array(y_policy[b]),
               mx.array(y_value[b]))


def evaluate(model, X, y_policy, y_value, batch_size=1024):
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    correct = 0
    total = 0
    n = X.shape[0]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Xb = mx.array(X[start:end])
        yp_b = mx.array(y_policy[start:end])
        yv_b = mx.array(y_value[start:end])

        logits, value = model(Xb)
        log_probs = logits - mx.max(logits, axis=-1, keepdims=True)
        log_probs = log_probs - mx.log(mx.sum(mx.exp(log_probs), axis=-1, keepdims=True))
        policy_loss = -mx.mean(mx.take_along_axis(
            log_probs, mx.expand_dims(yp_b, -1), axis=-1))
        value_loss = mx.mean((value.squeeze(-1) - yv_b) ** 2)
        loss = policy_loss + 0.5 * value_loss

        total_loss += float(loss) * (end - start)
        total_policy_loss += float(policy_loss) * (end - start)
        total_value_loss += float(value_loss) * (end - start)
        preds = mx.argmax(logits, axis=-1)
        correct += int(mx.sum(preds == yp_b))
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

    print(f'Loading data from {data_path} ...')
    data = np.load(data_path)
    X = data['X']
    y_policy = data['y']
    if 'v' in data:
        y_value = data['v'].astype(np.float32)
    else:
        # 兼容旧版无 outcome label 的数据
        y_value = np.zeros_like(y_policy, dtype=np.float32)

    n_total = X.shape[0]
    n_val = min(5000, n_total // 10)
    n_train = n_total - n_val
    X_train, X_val = X[:n_train], X[n_train:]
    yp_train, yp_val = y_policy[:n_train], y_policy[n_train:]
    yv_train, yv_val = y_value[:n_train], y_value[n_train:]

    print(f'Train: {n_train}, Val: {n_val}, features: {X.shape[1]}')

    hidden_dim = int(sys.argv[5]) if len(sys.argv) > 5 else 128
    model = MahjongNet(input_dim=X.shape[1], hidden_dim=hidden_dim)
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=lr)

    def _log_softmax(logits):
        logits = logits - mx.max(logits, axis=-1, keepdims=True)
        return logits - mx.log(mx.sum(mx.exp(logits), axis=-1, keepdims=True))

    # 编译损失函数和梯度
    def loss_fn(model, Xb, yp_b, yv_b):
        logits, value = model(Xb)
        log_probs = _log_softmax(logits)
        policy_loss = -mx.mean(mx.take_along_axis(
            log_probs, mx.expand_dims(yp_b, -1), axis=-1))
        value_loss = mx.mean((value.squeeze(-1) - yv_b) ** 2)
        return policy_loss + 0.5 * value_loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    best_val_loss = float('inf')
    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss_sum = 0.0
        train_batches = 0

        for Xb, yp_b, yv_b in batch_iterator(X_train, yp_train, yv_train, batch_size):
            loss, grads = loss_and_grad(model, Xb, yp_b, yv_b)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            train_loss_sum += float(loss)
            train_batches += 1

        val_metrics = evaluate(model, X_val, yp_val, yv_val, batch_size=batch_size)
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
            model.save_weights(os.path.join(out_dir, 'nn_model.npz'))
            with open(os.path.join(out_dir, 'nn_model_config.json'), 'w') as f:
                json.dump({'input_dim': int(X.shape[1]), 'hidden_dim': hidden_dim}, f)
            print('  -> saved best model')

    print('Training complete. Best model at output/nn_model.npz')


if __name__ == '__main__':
    main()
