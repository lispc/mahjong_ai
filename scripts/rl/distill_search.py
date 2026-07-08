# -*- coding: utf-8 -*-
"""Search-value / search-policy distillation。

用 `gen_search_value_data.py` 生成的 exact search labels (X, v, a) 蒸馏学生网络：
- policy distillation: 用 action `a` 训练 discard policy head
- value distillation: 用 search value `v` 训练 value head

可选从现有 backbone（如 nn_full_action_best.pt）初始化，保留其 response/tenpai/dealin head。

用法:
    PYTHONPATH=. python3 scripts/rl/distill_search.py \
        output/nn_search_value_v3d2_exact_250.npz \
        output/nn_full_action_best.pt \
        output/nn_search_distill_v3d2_exact_250.pt \
        --epochs 60 --batch 512 --lr 5e-5 --policy-weight 1.0 --value-weight 1.0
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algo.nn.model import build_model


def evaluate(model, X, yp, yv, device, bs=2048):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')
    mse = nn.MSELoss(reduction='sum')
    n = X.shape[0]
    pl = vl = 0.0
    correct = 0
    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(s + bs, n)
            xb = X[s:e].to(device)
            logits, value = model(xb)[:2]
            pl += float(ce(logits, yp[s:e].to(device)))
            vl += float(mse(value.squeeze(-1), yv[s:e].to(device)))
            correct += int((logits.argmax(1) == yp[s:e].to(device)).sum())
    return pl / n, vl / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='search label .npz with X, v, a')
    parser.add_argument('backbone', help='backbone .pt path (or "init" for random init)')
    parser.add_argument('out', help='output .pt path')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--policy-weight', type=float, default=1.0)
    parser.add_argument('--value-weight', type=float, default=1.0)
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='freeze conv trunk, only train policy/value heads')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载数据
    d = np.load(args.data)
    X = torch.from_numpy(d['X'].astype(np.float32))
    yp = torch.from_numpy(d['a'].astype(np.int64))
    yv = torch.from_numpy(d['v'].astype(np.float32))
    n = X.shape[0]
    print(f'Loaded {n} samples from {args.data}; value mean={yv.mean():.3f} std={yv.std():.3f}')

    # 划分 train/val
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g)
    n_val = min(2000, n // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, yptr, yvtr = X[tr_idx], yp[tr_idx], yv[tr_idx]
    Xval, ypval, yvval = X[val_idx], yp[val_idx], yv[val_idx]

    # 加载 backbone
    if args.backbone.lower() == 'init':
        # 默认使用与 nn_full_action_best 相同的架构，但关闭辅助 head
        config = {'arch': 'conv', 'input_dim': X.shape[1], 'channels': 128,
                  'n_blocks': 6, 'hidden_dim': 512, 'n_tile_ch': 5,
                  'features': 'base', 'framework': 'pytorch',
                  'dealin_head': False, 'tenpai_head': False, 'response_head': False}
        print('Random init with default conv 128/6/512 config')
    else:
        ckpt = torch.load(args.backbone, map_location='cpu')
        if 'config' in ckpt:
            config = ckpt['config']
            state = ckpt.get('model_state', ckpt)
        else:
            config = json.load(open(args.backbone.replace('.pt', '_config.json')))
            state = ckpt
        print(f'Loaded backbone config from {args.backbone}: {config}')

    model = build_model(config).to(device)
    if args.backbone.lower() != 'init':
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f'  missing keys: {missing[:10]} ...' if len(missing) > 10 else f'  missing keys: {missing}')
        if unexpected:
            print(f'  unexpected keys: {unexpected[:10]} ...' if len(unexpected) > 10 else f'  unexpected keys: {unexpected}')

    if args.freeze_backbone:
        # 冻结 trunk + 所有已有 head；只训练 policy/value head 的最后层参数
        for p in model.parameters():
            p.requires_grad = False
        # 解冻 policy/value head
        for name in ['policy_conv', 'policy_glob', 'value_fc', 'value_head']:
            m = getattr(model, name, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = True
        print('Backbone frozen; only policy/value heads trainable')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce = nn.CrossEntropyLoss(reduction='none')
    mse = nn.MSELoss(reduction='none')

    best_metric = -float('inf')
    ntr = Xtr.shape[0]
    out_dir = os.path.dirname(args.out) or 'output'
    os.makedirs(out_dir, exist_ok=True)
    out_json = args.out.replace('.pt', '_config.json')

    for ep in range(args.epochs):
        model.train()
        perm_tr = torch.randperm(ntr)
        t0 = time.time()
        for s in range(0, ntr, args.batch):
            idx = perm_tr[s:s + args.batch]
            xb = Xtr[idx].to(device)
            yb = yptr[idx].to(device)
            vb = yvtr[idx].to(device)

            logits, value = model(xb)[:2]
            policy_loss = ce(logits, yb).mean()
            value_loss = mse(value.squeeze(-1), vb).mean()
            loss = args.policy_weight * policy_loss + args.value_weight * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        sched.step()
        pl, vl, acc = evaluate(model, Xval, ypval, yvval, device)
        dt = time.time() - t0

        # 保存最佳：policy acc 为主，value mse 为辅
        metric = acc - vl
        improved = metric > best_metric
        if improved:
            best_metric = metric
            torch.save({'model_state': model.state_dict(), 'config': config}, args.out)
            json.dump(config, open(out_json, 'w'))

        print(f'ep {ep:2d} | val acc {acc:.3f} policy_ce {pl:.3f} value_mse {vl:.3f} '
              f'{"*" if improved else " "} | {dt:.1f}s')

    print(f'Done. Best saved to {args.out}')


if __name__ == '__main__':
    main()
