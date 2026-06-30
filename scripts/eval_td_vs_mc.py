#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比较 TD value model 和 MC best_1581 在真实 outcome 上的预测能力。

防止 TD target 循环（target 里包含 V(s')）导致虚低 val_loss 的误判：
直接用真实终局 outcome 作为 label 评估两个模型的预测 MSE。
"""
import sys
import os
import pickle
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from algo.nn.value_model import MahjongValueNetDeep


def load_model(weights_path, config_path, device):
    with open(config_path) as f:
        cfg = json.load(f)
    m = MahjongValueNetDeep(cfg['input_dim'], cfg.get('hidden_dims', [512, 256, 128])).to(device)
    m.load_state_dict(torch.load(weights_path, map_location=device))
    m.eval()
    return m


def predict(model, X, device, batch_size=4096):
    n = len(X)
    out = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            Xb = torch.tensor(X[start:end], dtype=torch.float32, device=device)
            v = model(Xb).squeeze(-1).cpu().numpy()
            out[start:end] = v
    return out


def main():
    td_pkl = sys.argv[1] if len(sys.argv) > 1 else 'output/selfplay_td_2000.pkl'
    td_model = sys.argv[2] if len(sys.argv) > 2 else 'output/nn_value_model_mc_td_v1_lam0.5.pt'
    td_config = sys.argv[3] if len(sys.argv) > 3 else 'output/nn_value_model_mc_td_v1_lam0.5.json'
    mc_model = sys.argv[4] if len(sys.argv) > 4 else 'output/nn_value_model_mc_best_1581.pt'
    mc_config = sys.argv[5] if len(sys.argv) > 5 else 'output/nn_value_model_mc_config_best_1581.json'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Loading trajectories from {td_pkl} ...')
    with open(td_pkl, 'rb') as f:
        trajectories = pickle.load(f)
    print(f'  {len(trajectories)} trajectories')

    # 收集所有 (features, actual_outcome) 对
    all_features = []
    all_outcomes = []
    for traj in trajectories:
        outcome = traj['outcome']
        for s in traj['samples']:
            all_features.append(s['features'])
            all_outcomes.append(outcome)  # 同一局所有 step 的 outcome 都是终局

    X = np.stack(all_features)
    y_outcome = np.array(all_outcomes, dtype=np.float32)
    print(f'Total samples: {len(X)}, outcome mean={y_outcome.mean():.3f}, std={y_outcome.std():.3f}')

    # 用最后 20% 作 val（避免和训练集重叠）
    n_val = len(X) // 5
    X_val = X[-n_val:]
    y_val = y_outcome[-n_val:]
    print(f'Val set: {n_val} samples')

    # 评估 MC best_1581
    print(f'\nEvaluating MC best_1581 ...')
    mc_m = load_model(mc_model, mc_config, device)
    mc_pred = predict(mc_m, X_val, device)
    mc_mse = float(np.mean((mc_pred - y_val) ** 2))
    mc_mae = float(np.mean(np.abs(mc_pred - y_val)))
    # 二分类准确率：outcome > 0 → 预测 > 0
    mc_acc = float(np.mean((mc_pred > 0) == (y_val > 0)))
    print(f'  MSE={mc_mse:.4f}, MAE={mc_mae:.4f}, win/loss acc={mc_acc:.3f}')
    print(f'  pred stats: mean={mc_pred.mean():.3f}, std={mc_pred.std():.3f}, '
          f'min={mc_pred.min():.3f}, max={mc_pred.max():.3f}')

    # 评估 TD model
    print(f'\nEvaluating TD model ...')
    td_m = load_model(td_model, td_config, device)
    td_pred = predict(td_m, X_val, device)
    td_mse = float(np.mean((td_pred - y_val) ** 2))
    td_mae = float(np.mean(np.abs(td_pred - y_val)))
    td_acc = float(np.mean((td_pred > 0) == (y_val > 0)))
    print(f'  MSE={td_mse:.4f}, MAE={td_mae:.4f}, win/loss acc={td_acc:.3f}')
    print(f'  pred stats: mean={td_pred.mean():.3f}, std={td_pred.std():.3f}, '
          f'min={td_pred.min():.3f}, max={td_pred.max():.3f}')

    # 对比
    print(f'\n=== Comparison (predicting actual outcome) ===')
    print(f'  MC best_1581:  MSE={mc_mse:.4f}, MAE={mc_mae:.4f}, acc={mc_acc:.3f}')
    print(f'  TD v1 (λ=0.5): MSE={td_mse:.4f}, MAE={td_mae:.4f}, acc={td_acc:.3f}')
    delta_mse = mc_mse - td_mse
    print(f'  Δ MSE (MC - TD): {delta_mse:+.4f}  ({">" if delta_mse > 0 else "<"} 0 means TD better)')
    delta_acc = td_acc - mc_acc
    print(f'  Δ acc (TD - MC): {delta_acc:+.4f}  ({">" if delta_acc > 0 else "<"} 0 means TD better)')

    if td_mse < mc_mse * 0.9:
        print(f'\n✓ TD model is significantly better at predicting real outcomes (>10% MSE reduction)')
        print(f'  Proceed to benchmark.')
    elif td_mse < mc_mse:
        print(f'\n~ TD model is marginally better. Benchmark may still help.')
    else:
        print(f'\n✗ TD model is NOT better at predicting real outcomes. Debug before benchmark.')


if __name__ == '__main__':
    main()
