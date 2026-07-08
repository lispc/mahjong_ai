# -*- coding: utf-8 -*-
"""Search-policy DPO / ranking distillation。

用 `gen_search_value_data.py` 生成的 search labels (X, a) 做偏好学习：
- preferred action = search 选出的动作 a
- rejected action = 同一状态下的其他合法动作（随机采样）
- 可选：用 search value v 给样本加权（高 value 样本更重要）

从 backbone（如 nn_full_action_best.pt）初始化作为 reference model，
训练 policy 网络最大化 DPO likelihood ratio。

用法:
    PYTHONPATH=. python3 scripts/rl/distill_search_dpo.py \
        output/nn_search_value_v3d2_exact_250.npz \
        output/nn_full_action_best.pt \
        output/nn_search_dpo_v3d2_eval0_5000.pt \
        --epochs 40 --batch 512 --lr 5e-5 --beta 0.1
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algo.nn.model import build_model


def evaluate(model, ref_model, X, yp, device, bs=2048, beta=0.1):
    """返回 policy acc、与 reference 的平均 KL、DPO loss。"""
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')
    n = X.shape[0]
    total_loss = correct = 0.0
    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(s + bs, n)
            xb = X[s:e].to(device)
            yb = yp[s:e].to(device)
            logits = model(xb)[0]
            ref_logits = ref_model(xb)[0]
            total_loss += float(ce(logits, yb))
            correct += int((logits.argmax(1) == yb).sum())
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='search label .npz with X, v, a')
    parser.add_argument('backbone', help='reference/backbone .pt path')
    parser.add_argument('out', help='output .pt path')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=0.1,
                        help='DPO temperature parameter')
    parser.add_argument('--rejected-per-sample', type=int, default=1,
                        help='number of rejected actions sampled per state')
    parser.add_argument('--value-weight', action='store_true',
                        help='weight samples by normalized search value')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载数据
    d = np.load(args.data)
    X = torch.from_numpy(d['X'].astype(np.float32))
    yp = torch.from_numpy(d['a'].astype(np.int64))
    yv = torch.from_numpy(d['v'].astype(np.float32)) if 'v' in d.files else None
    n = X.shape[0]
    print(f'Loaded {n} samples from {args.data}')

    # 划分 train/val
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g)
    n_val = min(2000, n // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, yptr = X[tr_idx], yp[tr_idx]
    Xval, ypval = X[val_idx], yp[val_idx]
    if yv is not None:
        yvtr = yv[tr_idx]
    else:
        yvtr = torch.ones(len(tr_idx), dtype=torch.float32)

    # 加载 backbone config
    ckpt = torch.load(args.backbone, map_location='cpu')
    if 'config' in ckpt:
        config = ckpt['config']
        state = ckpt.get('model_state', ckpt)
    else:
        config = json.load(open(args.backbone.replace('.pt', '_config.json')))
        state = ckpt
    print(f'Loaded backbone config from {args.backbone}: {config}')

    # policy model 和 reference model
    policy_model = build_model(config).to(device)
    policy_model.load_state_dict(state, strict=False)

    ref_model = build_model(config).to(device)
    ref_model.load_state_dict(state, strict=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = -1.0
    ntr = Xtr.shape[0]
    n_tile_ch = config.get('n_tile_ch', 5)
    tile_region = n_tile_ch * 34
    n_actions = 34

    for ep in range(args.epochs):
        policy_model.train()
        perm_tr = torch.randperm(ntr)
        t0 = time.time()
        epoch_loss = 0.0
        n_pairs = 0

        for s in range(0, ntr, args.batch):
            idx = perm_tr[s:s + args.batch]
            xb = Xtr[idx].to(device)
            yb_win = yptr[idx].to(device)
            vb = yvtr[idx].to(device)

            # 构建 rejected actions：随机选择不同于 winner 的合法动作
            # 简化：假设所有样本的合法动作都是全部 34 张牌（search data 来自闭手 14 张，
            #  winner 一定合法；rejected 随机选其他 33 张即可）
            B = xb.shape[0]
            yb_loss = []
            for _ in range(args.rejected_per_sample):
                rej = torch.randint(0, n_actions, (B,), device=device)
                # 确保 rejected != winner
                mask = rej == yb_win
                if mask.any():
                    rej[mask] = (rej[mask] + 1) % n_actions
                yb_loss.append(rej)
            yb_loss = torch.cat(yb_loss, dim=0)  # (B*R,)
            xb_exp = xb.repeat(args.rejected_per_sample, 1)
            yb_win_exp = yb_win.repeat(args.rejected_per_sample)
            vb_exp = vb.repeat(args.rejected_per_sample)

            logits = policy_model(xb_exp)[0]
            with torch.no_grad():
                ref_logits = ref_model(xb_exp)[0]

            # DPO loss: -log sigmoid(beta * (log pi(win)/pi(rej) - log ref(win)/ref(rej)))
            pi_win = logits.gather(1, yb_win_exp.unsqueeze(1)).squeeze(1)
            pi_rej = logits.gather(1, yb_loss.unsqueeze(1)).squeeze(1)
            ref_win = ref_logits.gather(1, yb_win_exp.unsqueeze(1)).squeeze(1)
            ref_rej = ref_logits.gather(1, yb_loss.unsqueeze(1)).squeeze(1)

            pi_ratio = pi_win - pi_rej
            ref_ratio = ref_win - ref_rej
            loss = -F.logsigmoid(args.beta * (pi_ratio - ref_ratio))

            # value-based 样本加权：高 search value 样本更重要
            if args.value_weight:
                # 归一化到 [0.5, 2.0]
                w = (vb_exp - vb_exp.min()) / (vb_exp.max() - vb_exp.min() + 1e-6)
                w = 0.5 + 1.5 * w
                loss = loss * w

            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss)
            n_pairs += B * args.rejected_per_sample

        sched.step()
        pl, acc = evaluate(policy_model, ref_model, Xval, ypval, device, beta=args.beta)
        dt = time.time() - t0

        improved = acc > best_acc
        if improved:
            best_acc = acc
            torch.save({'model_state': policy_model.state_dict(), 'config': config}, args.out)
            json.dump(config, open(args.out.replace('.pt', '_config.json'), 'w'))

        print(f'ep {ep:2d} | val acc {acc:.3f} policy_ce {pl:.3f} train_dpo_loss {epoch_loss/(n_pairs/args.batch):.3f} '
              f'{"*" if improved else " "} | {dt:.1f}s')

    print(f'Done. Best val acc = {best_acc:.3f}; saved to {args.out}')


if __name__ == '__main__':
    main()
