# -*- coding: utf-8 -*-
"""Oracle-Guided Distillation：用 oracle policy 蒸馏 normal policy。

输入：由 `gen_oracle_data.py` 生成的 .npz，包含 Xn(普通特征), Xo(oracle 特征), y, v。
步骤：
1. 加载已训好的 oracle policy（311-dim input）；
2. 对每对 (Xn, Xo) 计算 oracle policy 的 soft target（logits / probs）；
3. 训练 normal policy（175-dim input）拟合 y，同时 KL 接近 oracle probs。

用法：
    PYTHONPATH=. python3 scripts/rl/distill_oracle.py \
        output/nn_teacher_be_oracle_200.npz \
        output/nn_conv_bc_oracle_200.pt \
        40 512 1e-3 96 4 256 nn_conv_bc_distill_200 1.0
"""

import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algo.nn.model import build_model

OUT = 'output'


def _suit_perms():
    """花色置换增广。"""
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


def evaluate(student, X, yp, yv, device, bs=2048):
    student.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')
    mse = nn.MSELoss(reduction='sum')
    tot = X.shape[0]
    pl = vl = 0.0
    correct = 0
    with torch.no_grad():
        for s in range(0, tot, bs):
            e = min(s + bs, tot)
            xb = X[s:e].to(device)
            logits, value = student(xb)
            pl += float(ce(logits, yp[s:e].to(device)))
            vl += float(mse(value.squeeze(-1), yv[s:e].to(device)))
            correct += int((logits.argmax(1) == yp[s:e].to(device)).sum())
    return pl / tot, vl / tot, correct / tot


def main():
    data_path = sys.argv[1]
    oracle_path = sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    bs = int(sys.argv[4]) if len(sys.argv) > 4 else 512
    lr = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
    channels = int(sys.argv[6]) if len(sys.argv) > 6 else 96
    n_blocks = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    hidden = int(sys.argv[8]) if len(sys.argv) > 8 else 256
    tag = sys.argv[9] if len(sys.argv) > 9 else 'nn_conv_bc_distill'
    alpha = float(sys.argv[10]) if len(sys.argv) > 10 else 1.0  # KL weight

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = np.load(data_path)
    Xn = torch.from_numpy(d['Xn'].astype(np.float32))
    Xo = torch.from_numpy(d['Xo'].astype(np.float32))
    yp = torch.from_numpy(d['y'].astype(np.int64))
    yv = torch.from_numpy(d['v'].astype(np.float32)) if 'v' in d.files \
        else torch.zeros(len(yp), dtype=torch.float32)
    n = Xn.shape[0]
    input_dim = int(Xn.shape[1])
    oracle_dim = int(Xo.shape[1])
    n_tile_ch_normal = 5
    n_tile_ch_oracle = 9
    print(f'data {data_path}: {n} samples, normal_dim={input_dim} oracle_dim={oracle_dim} '
          f'alpha={alpha}')

    # 加载 oracle policy
    oracle_cfg_path = oracle_path.replace('.pt', '_config.json')
    oracle_cfg = json.load(open(oracle_cfg_path))
    oracle = build_model(oracle_cfg).to(device)
    sd = torch.load(oracle_path, map_location='cpu')
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    oracle.load_state_dict(sd)
    oracle.eval()
    for p in oracle.parameters():
        p.requires_grad = False

    # 预计算 oracle soft targets
    oracle_logits = []
    with torch.no_grad():
        for s in range(0, n, 2048):
            e = min(s + 2048, n)
            logits, _ = oracle(Xo[s:e].to(device))
            oracle_logits.append(logits.cpu())
    oracle_logits = torch.cat(oracle_logits, dim=0)
    oracle_probs = F.softmax(oracle_logits, dim=1)

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_val = min(8000, n // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, Xotr, yptr, yvtr = Xn[tr_idx], Xo[tr_idx], yp[tr_idx], yv[tr_idx]
    Xval, ypval, yvval = Xn[val_idx], yp[val_idx], yv[val_idx]
    oracle_probs_tr = oracle_probs[tr_idx]

    student_cfg = {'arch': 'conv', 'input_dim': input_dim, 'channels': channels,
                   'n_blocks': n_blocks, 'hidden_dim': hidden, 'n_tile_ch': n_tile_ch_normal,
                   'features': 'base', 'framework': 'pytorch', 'source': 'oracle_distill'}
    student = build_model(student_cfg).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    kl = nn.KLDivLoss(reduction='batchmean')

    best_acc = -1.0
    ntr = Xtr.shape[0]
    perms = _suit_perms()
    perm_tensors = [(torch.from_numpy(n2o).to(device), torch.from_numpy(am).to(device))
                    for n2o, am in perms]
    import random as _random
    for ep in range(epochs):
        student.train()
        perm_tr = torch.randperm(ntr)
        t0 = time.time()
        for s in range(0, ntr, bs):
            idx = perm_tr[s:s + bs]
            xb = Xtr[idx].to(device)
            yb = yptr[idx].to(device)
            vb = yvtr[idx].to(device)
            oracle_p = oracle_probs_tr[idx].to(device)
            # 花色置换
            n2o_t, am_t = perm_tensors[_random.randrange(6)]
            B = xb.shape[0]
            tile_region = n_tile_ch_normal * 34
            tiles = xb[:, :tile_region].reshape(B, n_tile_ch_normal, 34)[:, :, n2o_t].reshape(B, tile_region)
            xb = torch.cat([tiles, xb[:, tile_region:]], dim=1)
            yb = am_t[yb]
            oracle_p = oracle_p[:, am_t]  # 对 oracle probs 做同样置换
            logits, value = student(xb)
            policy_loss = ce(logits, yb)
            distill_loss = kl(F.log_softmax(logits, dim=1), oracle_p)
            value_loss = mse(value.squeeze(-1), vb)
            loss = policy_loss + alpha * distill_loss + 1.0 * value_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        pl, vl, acc = evaluate(student, Xval, ypval, yvval, device)
        dt = time.time() - t0
        print(f'ep {ep:2d} | val policy_ce {pl:.3f} acc {acc:.3f} value_mse {vl:.3f} | {dt:.1f}s')
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), f'{OUT}/{tag}.pt')
            json.dump(student_cfg, open(f'{OUT}/{tag}_config.json', 'w'))
    print(f'best val policy acc = {best_acc:.3f}; saved {OUT}/{tag}.pt')


if __name__ == '__main__':
    main()
