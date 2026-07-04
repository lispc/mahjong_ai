# -*- coding: utf-8 -*-
"""用 outcome-labeled 128k 完整动作数据做 KTO（Kahneman-Tversky Optimization）。

与 DPO 不同，KTO 只需要每个样本标“好/坏”（二元反馈），不需要配对：
- desirable：最终 seat reward > 0（赢）
- undesirable：最终 seat reward < 0（点炮/被自摸）
- 流局 reward == 0 的样本跳过

loss（对每条样本）：
  desirable:  -λ_D * log σ(β * (r - z0))
  undesirable: -λ_U * log σ(β * (z0 - r))
其中 r = log πθ(y|x) - log π_ref(y|x)，
z0 是当前 batch 上 KL(πθ||π_ref) 的均值，作为 reference point。

实现上使用 DataParallel + CPU DataLoader（num_workers=0 避免 CUDA fork 问题），
比单卡 in-GPU 索引的 Python 循环快很多。

用法：
    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python3 scripts/rl/train_full_action_kto.py \
        output/nn_full_action_data_128000.npz \
        output/nn_full_action_best.pt \
        output/nn_full_action_kto.pt \
        --epochs 10 --batch 2048 --lr 5e-5 --beta 0.1 --lambda-d 1.0 --lambda-u 2.0
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algo.nn.model import build_model


def set_spawn():
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass


def base_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


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


def kto_loss(logp_pi_full, logp_ref_full, y, reward, beta, lambda_d, lambda_u):
    """
    logp_pi_full/logp_ref_full: [B, A] 完整动作对数概率
    y: [B] 实际动作
    reward: [B] 最终 seat reward
    """
    logp_pi_act = logp_pi_full.gather(1, y.unsqueeze(1)).squeeze(1)
    logp_ref_act = logp_ref_full.gather(1, y.unsqueeze(1)).squeeze(1)
    r = logp_pi_act - logp_ref_act

    pi_probs = logp_pi_full.exp()
    kl = (pi_probs * (logp_pi_full - logp_ref_full)).sum(dim=-1).mean()
    z0 = kl.detach()  # KTO 中 z0 是 reference point，不参与梯度回传

    desirable = reward > 0
    undesirable = reward < 0

    loss = torch.tensor(0.0, device=logp_pi_full.device)
    metrics = {'z0': z0.item()}
    if desirable.any():
        r_d = r[desirable]
        loss_d = -lambda_d * F.logsigmoid(beta * (r_d - z0)).mean()
        loss = loss + loss_d
        with torch.no_grad():
            metrics['d_acc'] = (r_d > z0).float().mean().item()
            metrics['d_ratio'] = r_d.mean().item()
    if undesirable.any():
        r_u = r[undesirable]
        loss_u = -lambda_u * F.logsigmoid(beta * (z0 - r_u)).mean()
        loss = loss + loss_u
        with torch.no_grad():
            metrics['u_acc'] = (r_u < z0).float().mean().item()
            metrics['u_ratio'] = r_u.mean().item()

    return loss, metrics


def train_epoch(policy, ref_model, loader, optimizer, args, device, head='discard'):
    policy.train()
    ref_model.eval()

    total_loss = 0.0
    total_z0 = 0.0
    total_d_acc = 0.0
    total_u_acc = 0.0
    cnt_d = 0
    cnt_u = 0
    n = 0

    for x, y, reward, legal in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        reward = reward.to(device, non_blocking=True)
        if legal is not None:
            legal = legal.to(device, non_blocking=True)

        optimizer.zero_grad()
        out_pi = policy(x)
        with torch.no_grad():
            out_ref = ref_model(x)

        if head == 'discard':
            logp_pi_full = F.log_softmax(out_pi[0], dim=-1)
            logp_ref_full = F.log_softmax(out_ref[0], dim=-1)
        else:
            mask = (legal == 0).float() * -1e9
            logp_pi_full = F.log_softmax(out_pi[-1] + mask, dim=-1)
            logp_ref_full = F.log_softmax(out_ref[-1] + mask, dim=-1)

        loss, metrics = kto_loss(
            logp_pi_full, logp_ref_full, y, reward,
            args.beta, args.lambda_d, args.lambda_u)

        if args.bc_weight > 0:
            bc_logits = out_pi[0] if head == 'discard' else out_pi[-1]
            if head == 'response':
                bc_logits = bc_logits + mask
            bc_loss = F.cross_entropy(bc_logits, y)
            loss = loss + args.bc_weight * bc_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_z0 += metrics['z0']
        if 'd_acc' in metrics:
            total_d_acc += metrics['d_acc']
            cnt_d += 1
        if 'u_acc' in metrics:
            total_u_acc += metrics['u_acc']
            cnt_u += 1
        n += 1

    return {
        'loss': total_loss / max(n, 1),
        'z0': total_z0 / max(n, 1),
        'd_acc': total_d_acc / max(cnt_d, 1),
        'u_acc': total_u_acc / max(cnt_u, 1),
    }


def make_loader(X, y, reward, legal, batch_size):
    if legal is None:
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.zeros(len(X), dtype=torch.float32),  # dummy
        )
    else:
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(legal, dtype=torch.float32),
        )
    valid = (reward != 0)
    valid_idx = np.nonzero(valid)[0].tolist()
    ds = Subset(ds, valid_idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=4, pin_memory=True, drop_last=True)


def main():
    set_spawn()
    ap = argparse.ArgumentParser()
    ap.add_argument('data_path')
    ap.add_argument('init_model')
    ap.add_argument('out_model')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--beta', type=float, default=0.1)
    ap.add_argument('--lambda-d', type=float, default=1.0)
    ap.add_argument('--lambda-u', type=float, default=2.0)
    ap.add_argument('--bc-weight', type=float, default=0.0)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    # 单 GPU + spawn DataLoader workers，避免 DataParallel 在多卡上的死锁/低效问题
    use_dp = False
    num_workers = 4

    print(f'Loading data from {args.data_path}')
    data = np.load(args.data_path)
    Xd, yd, vd = data['X_discard'], data['y_discard'], data['v_discard']
    Xr, yr, lr, vr = data['X_response'], data['y_response'], data['legal_response'], data['v_response']
    print(f'discard samples: {Xd.shape[0]}, response samples: {Xr.shape[0]}')

    print(f'Loading policy/reference from {args.init_model}')
    policy, cfg = load_model(args.init_model, device)
    ref_model, _ = load_model(args.init_model, device)
    for p in ref_model.parameters():
        p.requires_grad = False

    if use_dp:
        policy = nn.DataParallel(policy)
        ref_model = nn.DataParallel(ref_model)
        print(f'Using DataParallel on {torch.cuda.device_count()} GPUs')

    loader_d = make_loader(Xd, yd, vd, None, args.batch)
    loader_r = make_loader(Xr, yr, vr, lr, args.batch)
    print(f'discard batches/epoch: {len(loader_d)}, response batches/epoch: {len(loader_r)}')

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_metric = -1e9
    best_state = None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        md = train_epoch(policy, ref_model, loader_d, optimizer, args, device, head='discard')
        mr = train_epoch(policy, ref_model, loader_r, optimizer, args, device, head='response')
        scheduler.step()

        combined = md['d_acc'] + md['u_acc'] + 0.5 * (mr['d_acc'] + mr['u_acc'])
        print(f'Epoch {epoch:3d} | disc loss={md["loss"]:.4f} z0={md["z0"]:.4f} '
              f'd_acc={md["d_acc"]:.3f} u_acc={md["u_acc"]:.3f} | '
              f'resp loss={mr["loss"]:.4f} z0={mr["z0"]:.4f} '
              f'd_acc={mr["d_acc"]:.3f} u_acc={mr["u_acc"]:.3f} | combined={combined:.3f}')

        epoch_path = args.out_model.replace('.pt', f'_epoch_{epoch:02d}.pt')
        torch.save({
            'model_state': base_model(policy).state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'best_metric': best_metric,
            'config': cfg,
        }, epoch_path)
        with open(epoch_path.replace('.pt', '_config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)

        if combined > best_metric:
            best_metric = combined
            best_state = {k: v.to(device).clone() for k, v in base_model(policy).state_dict().items()}

    if best_state is not None:
        base_model(policy).load_state_dict(best_state)

    out_cfg_path = args.out_model.replace('.pt', '_config.json')
    with open(out_cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    torch.save({'model_state': base_model(policy).state_dict(), 'config': cfg}, args.out_model)
    print(f'Training finished in {time.time() - t0:.1f}s, best combined metric {best_metric:.3f}')
    print(f'Saved {args.out_model} + {out_cfg_path}')


if __name__ == '__main__':
    main()
