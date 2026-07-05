# -*- coding: utf-8 -*-
"""把多个同架构 checkpoint 的权重做平均（model soup），保存为一个新模型。

用于零训练成本验证：把当前 best 与若干候选（128k epoch7、error-focused、winner-only 等）
做权重平均，看 ensemble 是否能超过单个模型。
"""

import os
import sys
import json
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algo.nn.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_pt')
    ap.add_argument('checkpoints', nargs='+')
    ap.add_argument('--weights', type=float, nargs='+', default=None,
                    help='每个 checkpoint 的权重，默认等权')
    ap.add_argument('--config', type=str, default=None,
                    help='输出 config 路径，默认从第一个 checkpoint 推断')
    args = ap.parse_args()

    if args.weights is not None:
        assert len(args.weights) == len(args.checkpoints)
        weights = torch.tensor(args.weights, dtype=torch.float32)
    else:
        weights = torch.ones(len(args.checkpoints), dtype=torch.float32)
    weights = weights / weights.sum()
    print(f'Soup from {len(args.checkpoints)} checkpoints, weights={weights.tolist()}')

    # 用第一个 checkpoint 的 config
    first_cfg_path = args.checkpoints[0].replace('.pt', '_config.json')
    if not os.path.exists(first_cfg_path):
        first_cfg_path = os.path.join(os.path.dirname(args.checkpoints[0]), 'nn_model_config.json')
    with open(first_cfg_path, 'r') as f:
        cfg = json.load(f)

    model = build_model(cfg)
    avg_sd = None
    for w, pt in zip(weights, args.checkpoints):
        sd = torch.load(pt, map_location='cpu')
        if isinstance(sd, dict):
            if 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            elif 'model_state' in sd:
                sd = sd['model_state']
        if avg_sd is None:
            avg_sd = {k: v * w for k, v in sd.items()}
        else:
            for k in avg_sd:
                avg_sd[k] += sd[k] * w

    model.load_state_dict(avg_sd)
    torch.save(model.state_dict(), args.out_pt)
    out_cfg = args.config or args.out_pt.replace('.pt', '_config.json')
    json.dump(cfg, open(out_cfg, 'w'), indent=2)
    print(f'Saved soup model: {args.out_pt} + {out_cfg}')


if __name__ == '__main__':
    main()
