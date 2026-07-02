# -*- coding: utf-8 -*-
"""训练带 deal-in auxiliary head 的 conv-BC。

输入：gen_dealin_data.py 生成的 .npz，包含 X, dealin, y, v。
训练目标：
  policy CE + value MSE + lambda * deal-in BCE(logits, labels)
其中 deal-in 标签 -1 表示该位置不参与 loss（不在手牌中）。

可选从 conv-BC base 初始化 trunk/policy/value（--init），deal-in head 随机初始化。

用法：
    PYTHONPATH=. python3 scripts/rl/train_dealin_aux.py \
        output/nn_dealin_labels_500.npz \
        output/nn_conv_bc_dealin_500.pt \
        --init output/nn_conv_bc.pt \
        --lambda_dealin 1.0 --epochs 60 --bs 512 --lr 1e-3
"""

import sys
import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algo.nn.model import build_model

OUT = 'output'


def _suit_perms():
    import itertools
    blocks = {0: list(range(0, 9)), 1: list(range(9, 18)), 2: list(range(18, 27))}
    honors = list(range(27, 34))
    perms = []
    for p in itertools.permutations([0, 1, 2]):
        new_to_old = blocks[p[0]] + blocks[p[1]] + blocks[p[2]] + honors
        action_map = [0] * 34
        for i, o in enumerate(new_to_old):
            action_map[o] = i
        perms.append((np.array(new_to_old, dtype=np.int64),
                      np.array(action_map, dtype=np.int64)))
    return perms


def evaluate(model, X, yp, yv, D, device, bs=2048):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')
    mse = nn.MSELoss(reduction='sum')
    bce = nn.BCEWithLogitsLoss(reduction='none')
    tot = X.shape[0]
    pl = vl = dl = 0.0
    correct = 0
    dealin_valid = 0
    dealin_correct = 0
    dealin_pos = 0
    dealin_pred_pos = 0
    with torch.no_grad():
        for s in range(0, tot, bs):
            e = min(s + bs, tot)
            xb = X[s:e].to(device)
            out = model(xb)
            logits, value = out[0], out[1]
            dealin_logits = out[2]
            pl += float(ce(logits, yp[s:e].to(device)))
            vl += float(mse(value.squeeze(-1), yv[s:e].to(device)))
            correct += int((logits.argmax(1) == yp[s:e].to(device)).sum())

            dlbl = D[s:e].to(device)
            valid = (dlbl >= 0)
            if valid.any():
                dloss = bce(dealin_logits, dlbl.clamp(0, 1))
                dl += float(dloss[valid].sum())
                dealin_valid += int(valid.sum())
                preds = (dealin_logits > 0).long()
                dealin_correct += int((preds == dlbl.long())[valid].sum())
                dealin_pos += int((dlbl > 0.5)[valid].sum())
                dealin_pred_pos += int((preds == 1)[valid].sum())
    metrics = {
        'policy_ce': pl / tot,
        'value_mse': vl / tot,
        'acc': correct / tot,
        'dealin_bce': dl / max(dealin_valid, 1),
        'dealin_acc': dealin_correct / max(dealin_valid, 1),
    }
    if dealin_pos > 0:
        metrics['dealin_recall'] = dealin_pos / max(dealin_valid, 1)  # base rate
        metrics['dealin_precision'] = dealin_pred_pos / max(dealin_valid, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('out_tag')
    parser.add_argument('--init', default='output/nn_conv_bc.pt')
    parser.add_argument('--lambda_dealin', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--channels', type=int, default=96)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--freeze_epochs', type=int, default=0,
                        help='前 N epochs 只训 deal-in head（trunk 冻结）')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = np.load(args.data)
    X = torch.from_numpy(d['X'].astype(np.float32))
    D = torch.from_numpy(d['dealin'].astype(np.float32))
    yp = torch.from_numpy(d['y'].astype(np.int64))
    yv = torch.from_numpy(d['v'].astype(np.float32)) if 'v' in d.files \
        else torch.zeros(len(yp), dtype=torch.float32)
    n = X.shape[0]
    input_dim = int(X.shape[1])
    print(f'data {args.data}: {n} samples, dim={input_dim} lambda_dealin={args.lambda_dealin}')

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_val = min(8000, n // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, Dtr, yptr, yvtr = X[tr_idx], D[tr_idx], yp[tr_idx], yv[tr_idx]
    Xval, Dval, ypval, yvval = X[val_idx], D[val_idx], yp[val_idx], yv[val_idx]

    config = {'arch': 'conv', 'input_dim': input_dim, 'channels': args.channels,
              'n_blocks': args.n_blocks, 'hidden_dim': args.hidden, 'n_tile_ch': 5,
              'features': 'base', 'framework': 'pytorch', 'source': 'dealin_aux',
              'dealin_head': True}
    model = build_model(config).to(device)

    # 可选从 base conv-BC 初始化 trunk/policy/value
    if args.init and os.path.exists(args.init):
        sd = torch.load(args.init, map_location='cpu')
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'loaded init from {args.init}: missing={len(missing)} unexpected={len(unexpected)}')
        if missing:
            print('missing keys sample:', missing[:5])
    else:
        print('no init provided, training from scratch')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss(reduction='none')

    best_acc = -1.0
    ntr = Xtr.shape[0]
    perms = _suit_perms()
    perm_tensors = [(torch.from_numpy(n2o).to(device), torch.from_numpy(am).to(device))
                    for n2o, am in perms]
    import random as _random
    for ep in range(args.epochs):
        model.train()
        perm_tr = torch.randperm(ntr)
        t0 = time.time()
        for s in range(0, ntr, args.bs):
            idx = perm_tr[s:s + args.bs]
            xb = Xtr[idx].to(device)
            yb = yptr[idx].to(device)
            vb = yvtr[idx].to(device)
            db = Dtr[idx].to(device)

            # 花色置换
            n2o_t, am_t = perm_tensors[_random.randrange(6)]
            B = xb.shape[0]
            tile_region = 5 * 34
            tiles = xb[:, :tile_region].reshape(B, 5, 34)[:, :, n2o_t].reshape(B, tile_region)
            xb = torch.cat([tiles, xb[:, tile_region:]], dim=1)
            yb = am_t[yb]
            db = db[:, n2o_t]

            out = model(xb)
            logits, value, dealin_logits = out[0], out[1], out[2]
            policy_loss = ce(logits, yb)
            value_loss = mse(value.squeeze(-1), vb)
            valid = (db >= 0)
            if valid.any():
                dealin_loss = bce(dealin_logits, db.clamp(0, 1))[valid].mean()
            else:
                dealin_loss = torch.tensor(0.0, device=device)
            loss = policy_loss + value_loss + args.lambda_dealin * dealin_loss

            # freeze_epochs：只更新 deal-in head
            if ep < args.freeze_epochs:
                for name, p in model.named_parameters():
                    if 'dealin' not in name:
                        p.requires_grad = False
            else:
                for p in model.parameters():
                    p.requires_grad = True

            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        metrics = evaluate(model, Xval, ypval, yvval, Dval, device)
        dt = time.time() - t0
        print(f'ep {ep:2d} | acc {metrics["acc"]:.3f} pce {metrics["policy_ce"]:.3f} '
              f'vmse {metrics["value_mse"]:.3f} | dealin bce {metrics["dealin_bce"]:.3f} '
              f'dacc {metrics["dealin_acc"]:.3f} | {dt:.1f}s')
        if metrics['acc'] > best_acc:
            best_acc = metrics['acc']
            out_path = os.path.join(OUT, f'{args.out_tag}.pt')
            torch.save(model.state_dict(), out_path)
            json.dump(config, open(os.path.join(OUT, f'{args.out_tag}_config.json'), 'w'))
    print(f'best val acc = {best_acc:.3f}; saved {OUT}/{args.out_tag}.pt')


if __name__ == '__main__':
    main()
