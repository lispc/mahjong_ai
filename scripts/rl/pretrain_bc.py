# -*- coding: utf-8 -*-
"""监督预训练（Behavior Cloning）TileConvNet：为 PPO 提供强初始化。

用现有 `nn_training_data_merged.npz`（96721 条：X 175 维, y 弃牌动作, v MC value）
训练带结构的卷积网络（policy CE + value MSE）。产出 `output/nn_conv_bc.pt` + config。

用法：
    PYTHONPATH=. python3 scripts/rl/pretrain_bc.py \
        [data=output/nn_training_data_merged.npz] [epochs=40] [bs=512] [lr=1e-3] \
        [channels=96] [n_blocks=4] [hidden=256] [tag=nn_conv_bc]
"""

import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algo.nn.model import build_model

OUT = 'output'


def _suit_perms():
    """晋北麻将 3 花色(万/条/筒)可互换 → 6 个置换，用于数据增广。

    34 牌轴布局：man[0:9] suo[9:18] tong[18:27] honors[27:34]。
    返回 [(new_to_old[34], action_map[34]), ...]：
      通道新值 c_new[i] = c_old[new_to_old[i]]；动作 a_new = action_map[a_old]。
    """
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


def evaluate(model, X, yp, yv, device, bs=2048):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')
    mse = nn.MSELoss(reduction='sum')
    tot = X.shape[0]
    pl = vl = 0.0
    correct = 0
    with torch.no_grad():
        for s in range(0, tot, bs):
            e = min(s + bs, tot)
            xb = X[s:e].to(device)
            logits, value = model(xb)
            pl += float(ce(logits, yp[s:e].to(device)))
            vl += float(mse(value.squeeze(-1), yv[s:e].to(device)))
            correct += int((logits.argmax(1) == yp[s:e].to(device)).sum())
    return pl / tot, vl / tot, correct / tot


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else f'{OUT}/nn_training_data_merged.npz'
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
    channels = int(sys.argv[5]) if len(sys.argv) > 5 else 96
    n_blocks = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    hidden = int(sys.argv[7]) if len(sys.argv) > 7 else 256
    tag = sys.argv[8] if len(sys.argv) > 8 else 'nn_conv_bc'
    n_tile_ch = int(sys.argv[9]) if len(sys.argv) > 9 else 5
    danger_weight = float(sys.argv[10]) if len(sys.argv) > 10 else 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = np.load(data_path)
    oracle_mode = os.environ.get('ORACLE_TRAIN', '0') == '1'
    if oracle_mode:
        if 'Xo' not in d.files:
            raise KeyError('ORACLE_TRAIN=1 but data has no Xo array')
        X = torch.from_numpy(d['Xo'].astype(np.float32))
    elif 'Xn' in d.files:
        X = torch.from_numpy(d['Xn'].astype(np.float32))
    elif 'X' in d.files:
        X = torch.from_numpy(d['X'].astype(np.float32))
    else:
        raise KeyError('No feature array found; expected X, Xn, or Xo')
    yp = torch.from_numpy(d['y'].astype(np.int64))
    yv = torch.from_numpy(d['v'].astype(np.float32)) if 'v' in d.files \
        else torch.zeros(len(yp), dtype=torch.float32)
    n = X.shape[0]
    input_dim = int(X.shape[1])
    tile_region = n_tile_ch * 34
    feature_type = 'oracle' if oracle_mode else ('ext' if n_tile_ch == 6 else 'base')
    print(f'data {data_path}: {n} samples, dim={input_dim} n_tile_ch={n_tile_ch} '
          f'features={feature_type} arch=conv ch={channels} blocks={n_blocks} hidden={hidden} '
          f'danger_weight={danger_weight}')

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_val = min(8000, n // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, yptr, yvtr = X[tr_idx], yp[tr_idx], yv[tr_idx]
    Xval, ypval, yvval = X[val_idx], yp[val_idx], yv[val_idx]

    config = {'arch': 'conv', 'input_dim': input_dim, 'channels': channels,
              'n_blocks': n_blocks, 'hidden_dim': hidden, 'n_tile_ch': n_tile_ch,
              'features': feature_type,
              'framework': 'pytorch', 'source': 'bc_pretrain'}
    model = build_model(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss(reduction='none')
    mse = nn.MSELoss(reduction='none')

    best_acc = -1.0
    ntr = Xtr.shape[0]
    # 花色置换增广（6×），只在训练 batch 上随机应用
    perms = _suit_perms()
    perm_tensors = [(torch.from_numpy(n2o).to(device), torch.from_numpy(am).to(device))
                    for n2o, am in perms]
    import random as _random
    for ep in range(epochs):
        model.train()
        perm_tr = torch.randperm(ntr)
        t0 = time.time()
        for s in range(0, ntr, bs):
            idx = perm_tr[s:s + bs]
            xb = Xtr[idx].to(device)
            yb = yptr[idx].to(device)
            vb = yvtr[idx].to(device)
            # 随机花色置换增广
            n2o_t, am_t = perm_tensors[_random.randrange(6)]
            B = xb.shape[0]
            tiles = xb[:, :tile_region].reshape(B, n_tile_ch, 34)[:, :, n2o_t].reshape(B, tile_region)
            xb = torch.cat([tiles, xb[:, tile_region:]], dim=1)
            yb = am_t[yb]
            logits, value = model(xb)
            # 高危险状态下样本加权，强制网络关注防守决策
            if danger_weight > 0 and n_tile_ch == 6:
                # ext 标量最后 3 维 = 对手危险等级 (已归一化到 [0,1])
                danger = xb[:, tile_region + 5:tile_region + 8].max(dim=1)[0]
                w = 1.0 + danger_weight * danger
            else:
                w = torch.ones(B, device=device)
            policy_loss = (ce(logits, yb) * w).mean()
            value_loss = (mse(value.squeeze(-1), vb) * w).mean()
            loss = policy_loss + 1.0 * value_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        pl, vl, acc = evaluate(model, Xval, ypval, yvval, device)
        dt = time.time() - t0
        print(f'ep {ep:2d} | val policy_ce {pl:.3f} acc {acc:.3f} value_mse {vl:.3f} | {dt:.1f}s')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'{OUT}/{tag}.pt')
            json.dump(config, open(f'{OUT}/{tag}_config.json', 'w'))
    print(f'best val policy acc = {best_acc:.3f}; saved {OUT}/{tag}.pt')


if __name__ == '__main__':
    main()
