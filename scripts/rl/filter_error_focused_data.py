# -*- coding: utf-8 -*-
"""从 BeliefExp 教师数据里筛选 NN 与老师意见不一致的样本，形成错题本数据集。

用法：
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/rl/filter_error_focused_data.py \
        output/nn_teacher_beliefexp_trace_16000.npz \
        output/nn_full_action_best.pt \
        output/nn_error_focused_16k.npz
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algo.nn.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('teacher_data')
    ap.add_argument('nn_model')
    ap.add_argument('out_npz')
    ap.add_argument('--response-data', type=str, default='output/nn_full_action_data_128000.npz')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg_path = args.nn_model.replace('.pt', '_config.json')
    cfg = json.load(open(cfg_path))
    net = build_model(cfg).to(device)
    sd = torch.load(args.nn_model, map_location=device, weights_only=False)
    net.load_state_dict(sd['model_state'] if isinstance(sd, dict) and 'model_state' in sd else sd)
    net.eval()

    td = np.load(args.teacher_data)
    X = torch.from_numpy(td['X']).float()
    y = td['y']

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            out = net(X[i:i+4096].to(device))
            logits = out[0] if isinstance(out, tuple) else out
            preds.append(logits.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(preds)

    mask = preds != y
    print(f'Teacher samples: {len(y)}, NN disagreements: {mask.sum()} ({mask.mean():.2%})')

    X_err = td['X'][mask]
    y_err = y[mask]
    v_err = td['v'][mask]
    tenpai_err = np.zeros(len(y_err), dtype=np.float32)

    # 保留 response 数据，让 response head 不崩
    rd = np.load(args.response_data)
    n_resp = min(len(rd['X_response']), 2_000_000)  # 限制数量，保持以 discard 为主
    idx = np.random.choice(len(rd['X_response']), n_resp, replace=False)

    np.savez(args.out_npz,
             X_discard=X_err, y_discard=y_err, v_discard=v_err, tenpai_discard=tenpai_err,
             X_response=rd['X_response'][idx], y_response=rd['y_response'][idx],
             legal_response=rd['legal_response'][idx], v_response=rd['v_response'][idx])
    print(f'Saved {args.out_npz}: {len(y_err)} discard + {n_resp} response samples')


if __name__ == '__main__':
    main()
