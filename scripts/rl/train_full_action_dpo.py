# -*- coding: utf-8 -*-
"""用 outcome-labeled 128k 完整动作数据做离线 DPO。

偏好对构造：
- 用 `v_discard` / `v_response` 作为最终 seat reward（赢=1，点炮/被自摸=-1，流局=0）。
- 正例（chosen）= reward > 0 的样本；负例（rejected）= reward < 0 的样本。
- 每个训练 step 从正负例各采一个 batch，做标准 DPO loss：
    -log sigmoid(beta * ((log pi/pi_ref(chosen) - log pi/pi_ref(rejected))))

实现要点：
- 预先把所有 tensor 放到 GPU，避免 DataLoader CPU 瓶颈。
- 每 epoch 对正负索引做 in-place shuffle，按 chunk 取 batch；短边循环。

用法：
    PYTHONPATH=. python3 scripts/rl/train_full_action_dpo.py \
        output/nn_full_action_data_128000.npz \
        output/nn_full_action_best.pt \
        output/nn_full_action_dpo.pt \
        --epochs 10 --batch 1024 --lr 5e-5 --beta 0.1 --resp_weight 0.5
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algo.nn.model import build_model


def base_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def load_model(model_path, device):
    cfg_path = model_path.replace('.pt', '_config.json')
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(model_path), 'nn_model_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        cfg = {'arch': 'mlp', 'input_dim': 175, 'hidden_dim': 256}
    model = build_model(cfg).to(device)
    sd = torch.load(model_path, map_location=device, weights_only=False)
    state = sd.get('model_state', sd)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print('load_model missing keys:', missing)
    if unexpected:
        print('load_model unexpected keys:', unexpected)
    return model, cfg


@torch.no_grad()
def precompute_ref_logps(model, X, y, legal, head, batch_size, device):
    """预计算参考策略下每个样本自身动作的对数概率。X/y/legal 已在 device 上。"""
    model.eval()
    N = X.size(0)
    logps = []
    for i in range(0, N, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        out = model(xb)
        if head == 'discard':
            logits = out[0]
            logp = F.log_softmax(logits, dim=-1)
        else:
            legal_b = legal[i:i + batch_size]
            resp_logits = out[-1]
            mask = (legal_b == 0).float() * -1e9
            logp = F.log_softmax(resp_logits + mask, dim=-1)
        logps.append(logp.gather(1, yb.unsqueeze(1)).squeeze(1))
    return torch.cat(logps, dim=0)


def dpo_loss(logp_pi_chosen, logp_pi_rejected, logp_ref_chosen, logp_ref_rejected, beta):
    ratio_chosen = logp_pi_chosen - logp_ref_chosen
    ratio_rejected = logp_pi_rejected - logp_ref_rejected
    loss = -F.logsigmoid(beta * (ratio_chosen - ratio_rejected)).mean()
    with torch.no_grad():
        acc = (ratio_chosen > ratio_rejected).float().mean().item()
    return loss, acc


def train_epoch(policy, optimizer, X, y, logp_ref, legal, pos_idx, neg_idx,
                args, device, head='discard'):
    policy.train()
    n_pos = pos_idx.size(0)
    n_neg = neg_idx.size(0)
    n_steps = max(n_pos, n_neg) // args.batch
    # in-place shuffle
    pos_idx = pos_idx[torch.randperm(n_pos, device=device)]
    neg_idx = neg_idx[torch.randperm(n_neg, device=device)]

    total_loss = 0.0
    total_acc = 0.0
    for step in range(n_steps):
        # 取 batch；短边循环
        p_start = (step * args.batch) % n_pos
        n_start = (step * args.batch) % n_neg
        p_idx = pos_idx[p_start:p_start + args.batch]
        n_idx = neg_idx[n_start:n_start + args.batch]
        if p_idx.size(0) < args.batch:
            p_idx = torch.cat([p_idx, pos_idx[:args.batch - p_idx.size(0)]], dim=0)
        if n_idx.size(0) < args.batch:
            n_idx = torch.cat([n_idx, neg_idx[:args.batch - n_idx.size(0)]], dim=0)

        x_c, y_c, logp_ref_c = X[p_idx], y[p_idx], logp_ref[p_idx]
        x_r, y_r, logp_ref_r = X[n_idx], y[n_idx], logp_ref[n_idx]

        optimizer.zero_grad()
        x_all = torch.cat([x_c, x_r], dim=0)
        y_all = torch.cat([y_c, y_r], dim=0)
        logp_ref_all = torch.cat([logp_ref_c, logp_ref_r], dim=0)
        out = policy(x_all)

        if head == 'discard':
            logits = out[0]
            logp_pi = F.log_softmax(logits, dim=-1).gather(1, y_all.unsqueeze(1)).squeeze(1)
        else:
            legal_all = torch.cat([legal[p_idx], legal[n_idx]], dim=0)
            resp_logits = out[-1]
            mask = (legal_all == 0).float() * -1e9
            logp_pi = F.log_softmax(resp_logits + mask, dim=-1).gather(1, y_all.unsqueeze(1)).squeeze(1)

        b = y_all.size(0)
        logp_pi_c, logp_pi_r = logp_pi[:b//2], logp_pi[b//2:]
        logp_ref_c, logp_ref_r = logp_ref_all[:b//2], logp_ref_all[b//2:]
        loss, acc = dpo_loss(logp_pi_c, logp_pi_r, logp_ref_c, logp_ref_r, args.beta)

        if args.bc_weight > 0:
            bc_logits = policy(x_c)[0] if head == 'discard' else policy(x_c)[-1]
            if head == 'response':
                mask_c = (legal[p_idx] == 0).float() * -1e9
                bc_logits = bc_logits + mask_c
            bc_loss = F.cross_entropy(bc_logits, y_c)
            loss = loss + args.bc_weight * bc_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc

    n = max(n_steps, 1)
    return total_loss / n, total_acc / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_path')
    ap.add_argument('init_model')
    ap.add_argument('out_model')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--beta', type=float, default=0.1)
    ap.add_argument('--resp_weight', type=float, default=0.5)
    ap.add_argument('--bc_weight', type=float, default=0.0)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    # 数据整份放在单 GPU 上，DataParallel 与 GPU-resident 大数据容易触发 cuDNN OOM，
    # 这里用单卡训练即可（模型很小，batch 1024 完全吃得下）。
    use_dp = False

    print(f'Loading data from {args.data_path}')
    data = np.load(args.data_path)
    Xd, yd, vd = data['X_discard'], data['y_discard'], data['v_discard']
    Xr, yr, lr, vr = data['X_response'], data['y_response'], data['legal_response'], data['v_response']
    print(f'discard samples: {Xd.shape[0]}, response samples: {Xr.shape[0]}')

    print(f'Loading reference/policy init from {args.init_model}')
    ref_model, cfg = load_model(args.init_model, device)
    policy, _ = load_model(args.init_model, device)
    if use_dp:
        ref_model = torch.nn.DataParallel(ref_model)
        policy = torch.nn.DataParallel(policy)

    # 把数据搬到 GPU（整份放下，3090 24GB 足够）
    print('Moving data to GPU ...')
    t0 = time.time()
    Xd_t = torch.tensor(Xd, dtype=torch.float32, device=device)
    yd_t = torch.tensor(yd, dtype=torch.long, device=device)
    vd_t = torch.tensor(vd, dtype=torch.float32, device=device)
    Xr_t = torch.tensor(Xr, dtype=torch.float32, device=device)
    yr_t = torch.tensor(yr, dtype=torch.long, device=device)
    lr_t = torch.tensor(lr, dtype=torch.float32, device=device)
    vr_t = torch.tensor(vr, dtype=torch.float32, device=device)
    print(f'Data moved in {time.time() - t0:.1f}s')

    # 预计算参考策略 logp
    print('Precomputing reference log-probabilities ...')
    t0 = time.time()
    ref_batch = max(args.batch * 8, 8192)
    logp_ref_d = precompute_ref_logps(ref_model, Xd_t, yd_t, None, 'discard', ref_batch, device)
    logp_ref_r = precompute_ref_logps(ref_model, Xr_t, yr_t, lr_t, 'response', ref_batch, device)
    del ref_model
    torch.cuda.empty_cache()
    print(f'Precompute done in {time.time() - t0:.1f}s')

    pos_d = (vd_t > 0).nonzero(as_tuple=True)[0]
    neg_d = (vd_t < 0).nonzero(as_tuple=True)[0]
    pos_r = (vr_t > 0).nonzero(as_tuple=True)[0]
    neg_r = (vr_t < 0).nonzero(as_tuple=True)[0]
    print(f'discard pos={pos_d.size(0)} neg={neg_d.size(0)}')
    print(f'response pos={pos_r.size(0)} neg={neg_r.size(0)}')

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = -1.0
    best_state = None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ld, ad = train_epoch(policy, optimizer, Xd_t, yd_t, logp_ref_d, None,
                             pos_d, neg_d, args, device, head='discard')
        lr_, ar = train_epoch(policy, optimizer, Xr_t, yr_t, logp_ref_r, lr_t,
                              pos_r, neg_r, args, device, head='response')
        scheduler.step()
        acc = (ad + args.resp_weight * ar) / (1 + args.resp_weight)
        print(f'Epoch {epoch:3d} | disc dpo_loss={ld:.4f} dpo_acc={ad:.3f} | '
              f'resp dpo_loss={lr_:.4f} dpo_acc={ar:.3f} | combined_acc={acc:.3f}')

        epoch_path = args.out_model.replace('.pt', f'_epoch_{epoch:02d}.pt')
        torch.save({
            'model_state': base_model(policy).state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
            'config': cfg,
        }, epoch_path)
        with open(epoch_path.replace('.pt', '_config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.to(device).clone() for k, v in base_model(policy).state_dict().items()}

    if best_state is not None:
        base_model(policy).load_state_dict(best_state)

    out_cfg_path = args.out_model.replace('.pt', '_config.json')
    with open(out_cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    torch.save({'model_state': base_model(policy).state_dict(), 'config': cfg}, args.out_model)
    print(f'Training finished in {time.time() - t0:.1f}s, best combined dpo_acc {best_acc:.3f}')
    print(f'Saved {args.out_model} + {out_cfg_path}')


if __name__ == '__main__':
    main()
