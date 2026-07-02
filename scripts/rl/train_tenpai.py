# -*- coding: utf-8 -*-
"""训练带 tenpai 决策头的 conv-BC。

输入：gen_hybrid_tenpai_data.py 生成的 .npz，包含
  X, dealin, y, v, X_tenpai, t, v_tenpai
训练目标：
  policy CE + value MSE + lambda_dealin * deal-in BCE + lambda_tenpai * tenpai BCE
其中 tenpai 为正样本（报听）加权，以处理类别不平衡。

可选从已有 conv-BC base（含或不含 dealin_head）初始化，tenpai_head 随机初始化。

用法：
    PYTHONPATH=. python3 scripts/rl/train_tenpai.py \
        output/nn_teacher_hybrid_tenpai_1000.npz \
        nn_conv_bc_tenpai_1000_l1 \
        --init output/nn_conv_bc_dealin_2000_l07.pt \
        --lambda_dealin 0.5 --lambda_tenpai 1.0 \
        --epochs 40 --bs 512 --lr 1e-3
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


def _permute_batch(xb, yb, db, n2o_t, am_t):
    B = xb.shape[0]
    tile_region = 5 * 34
    tiles = xb[:, :tile_region].reshape(B, 5, 34)[:, :, n2o_t].reshape(B, tile_region)
    xb = torch.cat([tiles, xb[:, tile_region:]], dim=1)
    if yb is not None:
        yb = am_t[yb]
    if db is not None:
        db = db[:, n2o_t]
    return xb, yb, db


def evaluate(model, X, yp, yv, D, Xt, tt, device, bs=2048):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')
    mse = nn.MSELoss(reduction='sum')
    bce = nn.BCEWithLogitsLoss(reduction='none')
    tot = X.shape[0]
    pl = vl = dl = 0.0
    correct = 0
    dealin_valid = 0
    with torch.no_grad():
        for s in range(0, tot, bs):
            e = min(s + bs, tot)
            xb = X[s:e].to(device)
            out = model(xb)
            logits, value = out[0], out[1]
            pl += float(ce(logits, yp[s:e].to(device)))
            vl += float(mse(value.squeeze(-1), yv[s:e].to(device)))
            correct += int((logits.argmax(1) == yp[s:e].to(device)).sum())
            if model.use_dealin:
                dealin_logits = out[2]
                dlbl = D[s:e].to(device)
                valid = (dlbl >= 0)
                if valid.any():
                    dloss = bce(dealin_logits, dlbl.clamp(0, 1))
                    dl += float(dloss[valid].sum())
                    dealin_valid += int(valid.sum())

    metrics = {
        'policy_ce': pl / tot,
        'value_mse': vl / tot,
        'acc': correct / tot,
        'dealin_bce': dl / max(dealin_valid, 1),
    }

    # tenpai metrics
    if Xt is not None and len(Xt) > 0:
        ttot = Xt.shape[0]
        tl = tcorrect = tpos = tpred_pos = 0
        with torch.no_grad():
            for s in range(0, ttot, bs):
                e = min(s + bs, ttot)
                xb = Xt[s:e].to(device)
                logit = model.tenpai_logit(xb).squeeze(-1)
                tb = tt[s:e].to(device)
                tl += float(F.binary_cross_entropy_with_logits(logit, tb, reduction='sum'))
                pred = (logit > 0).long()
                tcorrect += int((pred == tb.long()).sum())
                tpos += int((tb > 0.5).sum())
                tpred_pos += int(pred.sum())
        metrics['tenpai_bce'] = tl / ttot
        metrics['tenpai_acc'] = tcorrect / ttot
        metrics['tenpai_pos_rate'] = tpos / ttot
        metrics['tenpai_precision'] = (tpred_pos > 0 and tpos > 0) and (tpos / tpred_pos) or 0.0
        metrics['tenpai_recall'] = (tpos > 0) and (tpos / ttot) or 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('out_tag')
    parser.add_argument('--init', default='output/nn_conv_bc_dealin_2000_l07.pt')
    parser.add_argument('--lambda_dealin', type=float, default=0.5)
    parser.add_argument('--lambda_tenpai', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--freeze_epochs', type=int, default=0,
                        help='前 N epochs 只训 tenpai/dealin head（trunk 冻结）')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = np.load(args.data)
    X = torch.from_numpy(d['X'].astype(np.float32))
    yp = torch.from_numpy(d['y'].astype(np.int64))
    yv = torch.from_numpy(d['v'].astype(np.float32))
    D = torch.from_numpy(d['dealin'].astype(np.float32)) if 'dealin' in d.files else None
    if D is None and args.lambda_dealin > 0:
        print('warning: no dealin labels, setting lambda_dealin=0')
        args.lambda_dealin = 0.0

    Xt = torch.from_numpy(d['X_tenpai'].astype(np.float32)) if 'X_tenpai' in d.files else None
    tt = torch.from_numpy(d['t'].astype(np.float32)) if 't' in d.files else None
    vt = torch.from_numpy(d['v_tenpai'].astype(np.float32)) if 'v_tenpai' in d.files else None

    n = X.shape[0]
    input_dim = int(X.shape[1])
    print(f'data {args.data}: discard={n}, dim={input_dim}, '
          f'tenpai={len(Xt) if Xt is not None else 0}')
    if Xt is not None:
        pos = int((tt > 0.5).sum())
        neg = len(tt) - pos
        print(f'tenpai label balance: pos={pos}, neg={neg}, pos_rate={pos/max(len(tt),1):.3f}')
        tenpai_pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    else:
        tenpai_pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_val = min(8000, n // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, yptr, yvtr = X[tr_idx], yp[tr_idx], yv[tr_idx]
    Xval, ypval, yvval = X[val_idx], yp[val_idx], yv[val_idx]
    Dtr = D[tr_idx] if D is not None else None
    Dval = D[val_idx] if D is not None else None

    if Xt is not None:
        nt = Xt.shape[0]
        n_val_t = min(2000, nt // 10)
        perm_t = torch.randperm(nt, generator=g)
        val_t_idx, tr_t_idx = perm_t[:n_val_t], perm_t[n_val_t:]
        Xttr, tttr, vttr = Xt[tr_t_idx], tt[tr_t_idx], vt[tr_t_idx]
        Xtval, ttval, vtval = Xt[val_t_idx], tt[val_t_idx], vt[val_t_idx]
    else:
        Xttr = tttr = vttr = Xtval = ttval = vtval = None

    # 读取 base 配置并加入 tenpai_head
    cfg_path = args.init.replace('.pt', '_config.json') if args.init else None
    if cfg_path and os.path.exists(cfg_path):
        config = json.load(open(cfg_path))
        print(f'loaded base config from {cfg_path}')
    else:
        config = {'arch': 'conv', 'input_dim': input_dim, 'channels': 96,
                  'n_blocks': 4, 'hidden_dim': 256, 'n_tile_ch': 5, 'features': 'base',
                  'framework': 'pytorch', 'source': 'tenpai'}
    config['tenpai_head'] = True
    config.setdefault('dealin_head', False)
    config['source'] = 'tenpai'
    model = build_model(config).to(device)

    if args.init and os.path.exists(args.init):
        sd = torch.load(args.init, map_location='cpu')
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'loaded init from {args.init}: missing={len(missing)} unexpected={len(unexpected)}')
        if missing:
            print('missing keys sample:', missing[:10])
    else:
        print('no init provided, training from scratch')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    bce_none = nn.BCEWithLogitsLoss(reduction='none')
    tenpai_bce = nn.BCEWithLogitsLoss(pos_weight=tenpai_pos_weight)

    best_acc = -1.0
    ntr = Xtr.shape[0]
    ntr_t = len(Xttr) if Xttr is not None else 0
    perms = _suit_perms()
    perm_tensors = [(torch.from_numpy(n2o).to(device), torch.from_numpy(am).to(device))
                    for n2o, am in perms]
    import random as _random

    for ep in range(args.epochs):
        model.train()
        perm_tr = torch.randperm(ntr)
        perm_tr_t = torch.randperm(ntr_t) if ntr_t > 0 else None
        t_idx = 0
        t0 = time.time()
        for s in range(0, ntr, args.bs):
            idx = perm_tr[s:s + args.bs]
            xb = Xtr[idx].to(device)
            yb = yptr[idx].to(device)
            vb = yvtr[idx].to(device)
            db = Dtr[idx].to(device) if Dtr is not None else None

            n2o_t, am_t = perm_tensors[_random.randrange(6)]
            xb, yb, db = _permute_batch(xb, yb, db, n2o_t, am_t)

            out = model(xb)
            logits, value = out[0], out[1]
            policy_loss = ce(logits, yb)
            value_loss = mse(value.squeeze(-1), vb)
            loss = policy_loss + 0.5 * value_loss

            if model.use_dealin and args.lambda_dealin > 0 and db is not None:
                valid = (db >= 0)
                if valid.any():
                    dealin_loss = bce_none(out[2], db.clamp(0, 1))[valid].mean()
                    loss = loss + args.lambda_dealin * dealin_loss

            # tenpai mini-batch
            if ntr_t > 0 and args.lambda_tenpai > 0:
                tbs = min(args.bs, ntr_t)
                t_idx = (s // args.bs) % max(1, (ntr_t + tbs - 1) // tbs)
                t_start = t_idx * tbs
                t_idx_end = min(t_start + tbs, ntr_t)
                if t_start < t_idx_end:
                    xtb = Xttr[t_start:t_idx_end].to(device)
                    ttb = tttr[t_start:t_idx_end].to(device)
                    xtb, _, _ = _permute_batch(xtb, None, None, n2o_t, am_t)
                    tenpai_loss = tenpai_bce(model.tenpai_logit(xtb).squeeze(-1), ttb)
                    loss = loss + args.lambda_tenpai * tenpai_loss

            # freeze_epochs：只更新 head
            if ep < args.freeze_epochs:
                for name, p in model.named_parameters():
                    if 'tenpai' not in name and 'dealin' not in name:
                        p.requires_grad = False
            else:
                for p in model.parameters():
                    p.requires_grad = True

            opt.zero_grad()
            loss.backward()
            opt.step()

        # 如果 tenpai 样本数远多于 discard batch 数，每个 epoch 末尾再扫一遍剩余 tenpai
        if ntr_t > 0 and args.lambda_tenpai > 0 and ntr_t > ntr:
            model.train()
            for s in range(0, ntr_t, args.bs):
                idx = perm_tr_t[s:s + args.bs]
                xtb = Xttr[idx].to(device)
                ttb = tttr[idx].to(device)
                n2o_t, _ = perm_tensors[_random.randrange(6)]
                xtb, _, _ = _permute_batch(xtb, None, None, n2o_t, None)
                tenpai_loss = tenpai_bce(model.tenpai_logit(xtb).squeeze(-1), ttb)
                opt.zero_grad()
                (args.lambda_tenpai * tenpai_loss).backward()
                opt.step()

        sched.step()
        metrics = evaluate(model, Xval, ypval, yvval, Dval, Xtval, ttval, device)
        dt = time.time() - t0
        tenpai_str = ''
        if 'tenpai_acc' in metrics:
            tenpai_str = (f' | tenpai bce {metrics["tenpai_bce"]:.3f} '
                          f'acc {metrics["tenpai_acc"]:.3f} '
                          f'pos {metrics["tenpai_pos_rate"]:.3f}')
        print(f'ep {ep:2d} | acc {metrics["acc"]:.3f} pce {metrics["policy_ce"]:.3f} '
              f'vmse {metrics["value_mse"]:.3f} | dealin bce {metrics["dealin_bce"]:.3f}'
              f'{tenpai_str} | {dt:.1f}s')
        if metrics['acc'] > best_acc:
            best_acc = metrics['acc']
            out_path = os.path.join(OUT, f'{args.out_tag}.pt')
            torch.save(model.state_dict(), out_path)
            json.dump(config, open(os.path.join(OUT, f'{args.out_tag}_config.json'), 'w'))
    print(f'best val acc = {best_acc:.3f}; saved {OUT}/{args.out_tag}.pt')


if __name__ == '__main__':
    main()
