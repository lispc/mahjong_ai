# -*- coding: utf-8 -*-
"""用搜索轨迹（search trace）蒸馏训练 conv-BC。

输入：gen_v3_trace_data.py 生成的 .npz，包含
  X, y, scores, selected_value, dealin, v
训练目标：
  policy CE + α * KL(student_policy || softmax(teacher_scores / T))
  + value MSE(v) + β * value MSE(tanh(selected_value / τ))
  + λ_dealin * deal-in BCE

其中 teacher_scores 是 V3-NN-PC 对每个候选的 expectimax 评分，
非候选位置为 -1e9；soft target 让网络学到教师对候选的偏好结构，
而不是只拟合最终选择的 hard label。

用法：
    PYTHONPATH=. python3 scripts/rl/train_search_distill.py \
        output/nn_teacher_v3_trace_500.npz \
        nn_conv_bc_searchdistill_500 \
        --init output/nn_conv_bc_dealin_2000_l07.pt \
        --alpha 0.5 --temp 2.0 --beta 0.3 --tau 10.0 \
        --lambda_dealin 0.5 --epochs 40 --bs 512 --lr 1e-3
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


def _permute_batch(xb, yb, db, sb, n2o_t, am_t):
    B = xb.shape[0]
    tile_region = 5 * 34
    tiles = xb[:, :tile_region].reshape(B, 5, 34)[:, :, n2o_t].reshape(B, tile_region)
    xb = torch.cat([tiles, xb[:, tile_region:]], dim=1)
    if yb is not None:
        yb = am_t[yb]
    if db is not None:
        db = db[:, n2o_t]
    if sb is not None:
        sb = sb[:, n2o_t]
    return xb, yb, db, sb


def evaluate(model, X, yp, yv, S, SV, HT, D, device, bs=2048):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')
    mse = nn.MSELoss(reduction='sum')
    bce = nn.BCEWithLogitsLoss(reduction='none')
    tot = X.shape[0]
    pl = vl = dl = kl = 0.0
    correct = 0
    dealin_valid = 0
    trace_tot = 0
    with torch.no_grad():
        for s in range(0, tot, bs):
            e = min(s + bs, tot)
            xb = X[s:e].to(device)
            out = model(xb)
            logits, value = out[0], out[1]
            pl += float(ce(logits, yp[s:e].to(device)))
            vl += float(mse(value.squeeze(-1), yv[s:e].to(device)))
            correct += int((logits.argmax(1) == yp[s:e].to(device)).sum())

            if S is not None and HT is not None:
                sb = S[s:e].to(device)
                htb = HT[s:e].to(device)
                trace_mask = (htb > 0.5)
                if trace_mask.any():
                    teacher_probs = F.softmax(sb, dim=1)
                    kl_per_sample = F.kl_div(F.log_softmax(logits, dim=1), teacher_probs,
                                             reduction='none').sum(dim=1)
                    kl += float(kl_per_sample[trace_mask].sum())
                    trace_tot += int(trace_mask.sum())

            if model.use_dealin and D is not None:
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
        'kl': kl / max(trace_tot, 1),
        'dealin_bce': dl / max(dealin_valid, 1),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('out_tag')
    parser.add_argument('--init', default='output/nn_conv_bc_dealin_2000_l07.pt')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='soft policy KL loss 权重')
    parser.add_argument('--temp', type=float, default=2.0,
                        help='teacher score softmax 温度')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='dense value MSE 权重')
    parser.add_argument('--tau', type=float, default=10.0,
                        help='dense value tanh 缩放')
    parser.add_argument('--lambda_dealin', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--channels', type=int, default=96)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--freeze_epochs', type=int, default=0,
                        help='前 N epochs 只训 policy head / value head（trunk 冻结）')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = np.load(args.data)
    X = torch.from_numpy(d['X'].astype(np.float32))
    yp = torch.from_numpy(d['y'].astype(np.int64))
    yv = torch.from_numpy(d['v'].astype(np.float32))
    S = torch.from_numpy(d['scores'].astype(np.float32)) if 'scores' in d.files else None
    SV = torch.from_numpy(d['selected_value'].astype(np.float32)) if 'selected_value' in d.files else None
    HT = torch.from_numpy(d['has_trace'].astype(np.float32)) if 'has_trace' in d.files else None
    D = torch.from_numpy(d['dealin'].astype(np.float32)) if 'dealin' in d.files else None

    n_trace = int((HT > 0.5).sum()) if HT is not None else 0

    if S is not None and args.alpha > 0:
        S = S / args.temp
    if SV is not None and args.beta > 0:
        SV = torch.tanh(SV / args.tau)

    n = X.shape[0]
    input_dim = int(X.shape[1])
    print(f'data {args.data}: {n} samples ({n_trace} with trace), dim={input_dim}, '
          f'channels={args.channels}, n_blocks={args.n_blocks}, hidden={args.hidden}, '
          f'alpha={args.alpha}, temp={args.temp}, beta={args.beta}, tau={args.tau}, '
          f'lambda_dealin={args.lambda_dealin}')

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_val = min(8000, n // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, yptr, yvtr = X[tr_idx], yp[tr_idx], yv[tr_idx]
    Xval, ypval, yvval = X[val_idx], yp[val_idx], yv[val_idx]
    Str, SVtr, HTtr = (S[tr_idx], SV[tr_idx], HT[tr_idx]) if S is not None else (None, None, None)
    Sval, SVval, HTval = (S[val_idx], SV[val_idx], HT[val_idx]) if S is not None else (None, None, None)
    Dtr = D[tr_idx] if D is not None else None
    Dval = D[val_idx] if D is not None else None

    # 读取/构建配置
    config = {'arch': 'conv', 'input_dim': input_dim, 'channels': args.channels,
              'n_blocks': args.n_blocks, 'hidden_dim': args.hidden, 'n_tile_ch': 5,
              'features': 'base', 'framework': 'pytorch', 'source': 'search_distill',
              'dealin_head': True}
    model = build_model(config).to(device)

    if args.init and os.path.exists(args.init):
        sd = torch.load(args.init, map_location='cpu')
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        # 只加载形状匹配的参数（允许大网络从小网络热启部分层）
        model_sd = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in sd.items():
            if k in model_sd and v.shape == model_sd[k].shape:
                filtered[k] = v
            else:
                skipped.append(k)
        model.load_state_dict(filtered, strict=False)
        print(f'loaded init from {args.init}: matched={len(filtered)}/{len(sd)} '
              f'skipped={len(skipped)}')
        if skipped:
            print('skipped keys sample:', skipped[:10])
    else:
        print('no init provided, training from scratch')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    bce_none = nn.BCEWithLogitsLoss(reduction='none')

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
            sb = Str[idx].to(device) if Str is not None else None
            svb = SVtr[idx].to(device) if SVtr is not None else None
            htb = HTtr[idx].to(device) if HTtr is not None else None
            db = Dtr[idx].to(device) if Dtr is not None else None

            n2o_t, am_t = perm_tensors[_random.randrange(6)]
            xb, yb, db, sb = _permute_batch(xb, yb, db, sb, n2o_t, am_t)
            if htb is not None:
                htb = htb  # has_trace 是标量，不受花色置换影响

            out = model(xb)
            logits, value = out[0], out[1]
            policy_loss = ce(logits, yb)
            value_loss = mse(value.squeeze(-1), vb)
            loss = policy_loss + 0.5 * value_loss

            if sb is not None and args.alpha > 0 and htb is not None:
                trace_mask = (htb > 0.5)
                if trace_mask.any():
                    teacher_probs = F.softmax(sb, dim=1)
                    kl_per_sample = F.kl_div(F.log_softmax(logits, dim=1), teacher_probs,
                                             reduction='none').sum(dim=1)
                    kl_loss = kl_per_sample[trace_mask].mean()
                    loss = loss + args.alpha * kl_loss

            if svb is not None and args.beta > 0 and htb is not None:
                trace_mask = (htb > 0.5)
                if trace_mask.any():
                    dense_value_loss = mse(value.squeeze(-1)[trace_mask], svb[trace_mask])
                    loss = loss + args.beta * dense_value_loss

            if model.use_dealin and args.lambda_dealin > 0 and db is not None:
                valid = (db >= 0)
                if valid.any():
                    dealin_loss = bce_none(out[2], db.clamp(0, 1))[valid].mean()
                    loss = loss + args.lambda_dealin * dealin_loss

            if ep < args.freeze_epochs:
                for name, p in model.named_parameters():
                    if 'policy' not in name and 'value' not in name and 'dealin' not in name:
                        p.requires_grad = False
            else:
                for p in model.parameters():
                    p.requires_grad = True

            opt.zero_grad()
            loss.backward()
            opt.step()

        sched.step()
        metrics = evaluate(model, Xval, ypval, yvval, Sval, SVval, HTval, Dval, device)
        dt = time.time() - t0
        print(f'ep {ep:2d} | acc {metrics["acc"]:.3f} pce {metrics["policy_ce"]:.3f} '
              f'vmse {metrics["value_mse"]:.3f} kl {metrics["kl"]:.3f} '
              f'dealin_bce {metrics["dealin_bce"]:.3f} | {dt:.1f}s')
        if metrics['acc'] > best_acc:
            best_acc = metrics['acc']
            out_path = os.path.join(OUT, f'{args.out_tag}.pt')
            torch.save(model.state_dict(), out_path)
            json.dump(config, open(os.path.join(OUT, f'{args.out_tag}_config.json'), 'w'))
    print(f'best val acc = {best_acc:.3f}; saved {OUT}/{args.out_tag}.pt')


if __name__ == '__main__':
    main()
