# -*- coding: utf-8 -*-
"""P1-1b：danger 头重训（冻结 trunk）+ policy 蒸馏 sanity 组
（docs/plan-beliefjax-0720.md §2-1b）。

数据：jaxenv/gen_belief_data.py 产物 output/belief_labels_shard_*.npz。
 backbone：output/jax_gumbel_iter92.pt（TileConvNet 128/6/512，带 dealin 头）。

- danger 头（核心交付物）：冻结 trunk，只用 belief 标签重训 dealin 头
  （dealin_conv/dealin_glob）。输出 34 维；top-8 候选监督 danger 值（MSE），
  非 top-8 不监督（mask）。val 指标：top-8 内 Kendall tau-b / min-danger top1
  命中率（BeliefExp 防守选择）/ max-danger top1 命中率；同时报告旧 dealin 头
  同指标（基线，量化旧头死亡程度）。
- policy 蒸馏（sanity 组）：同数据 CE(chosen) 全模型精调（iter92 出发），
  报告 val acc（含训练前基线）。
- 锁手状态（obs[170]>0.5，chosen=强制弃牌非真实决策）默认从两组训练剔除。

用法：
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python3 scripts/rl/train_danger_head.py \
        --shards 'output/belief_labels_shard_*.npz' \
        --out output/nn_danger_belief_v1.pt [--epochs 10] [--policy-epochs 3]

产物：--out（{'model_state','config'} 镜像 iter92 格式）+ 同名 _config.json +
      output/nn_policy_distill_belief_v1.pt（+_config.json）+ 训练日志
      <out 去后缀>_train.log。
"""

import argparse
import copy
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from algo.nn.model import build_model

DEALIN_KEYS = ('dealin_conv.weight', 'dealin_conv.bias',
               'dealin_glob.weight', 'dealin_glob.bias')


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------

def load_shards(pattern, exclude_locked=True):
    files = sorted(glob.glob(pattern))
    assert files, f'未匹配到 shard: {pattern}'
    obs, chosen, top8, offense, danger, dflag = [], [], [], [], [], []
    for f in files:
        d = np.load(f)
        obs.append(d['obs'])
        chosen.append(d['chosen'])
        top8.append(d['top8'])
        offense.append(d['offense'])
        danger.append(d['danger'])
        dflag.append(d['defense_flag'])
        print(f'[data] {f}: {len(d["chosen"])} rows', flush=True)
    obs = np.concatenate(obs).astype(np.float32)
    chosen = np.concatenate(chosen).astype(np.int64)
    top8 = np.concatenate(top8).astype(np.int64)
    offense = np.concatenate(offense).astype(np.int64)
    danger = np.concatenate(danger).astype(np.float32)
    dflag = np.concatenate(dflag)
    n0 = len(chosen)
    if exclude_locked:
        keep = obs[:, 170] <= 0.5            # obs[170] = 自家 locked flag
        obs, chosen, top8, offense, danger, dflag = (
            x[keep] for x in (obs, chosen, top8, offense, danger, dflag))
        print(f'[data] {n0} rows, 剔除锁手 {n0 - keep.sum()} -> {keep.sum()} '
              f'(defense_flag=True 占比 {dflag.mean():.3f})', flush=True)
    # top-8 监督掩码/目标
    valid8 = top8 >= 0                                   # (N,8)
    mask = np.zeros((len(chosen), 34), np.bool_)
    target = np.zeros((len(chosen), 34), np.float32)
    rows = np.arange(len(chosen))[:, None]
    mask[rows, np.clip(top8, 0, 33)] = valid8
    target[rows, np.clip(top8, 0, 33)] = np.where(valid8, danger, 0.0)
    legal = obs[:, :34] > 0                              # policy CE 合法掩码
    return dict(obs=obs, chosen=chosen, mask=mask, target=target,
                top8=top8, danger=danger, valid8=valid8, legal=legal,
                dflag=dflag, files=files)


# ---------------------------------------------------------------------------
# 指标：top-8 内 Kendall tau-b / min(danger) top1 / max top1（numpy，逐样本）
# ---------------------------------------------------------------------------

def rank_metrics(pred8, label8, valid8):
    """pred8/label8: (N,8) float；valid8: (N,8) bool。返回 (taub, hit_min, hit_max)。"""
    taus, hmin, hmax = [], [], []
    for p, l, v in zip(pred8, label8, valid8):
        p, l = p[v], l[v]
        k = len(p)
        if k < 2:
            continue
        c = d = tx = ty = 0
        for i in range(k):
            for j in range(i + 1, k):
                sp = np.sign(p[i] - p[j])
                sl = np.sign(l[i] - l[j])
                if sp == 0 and sl == 0:
                    continue
                if sp == 0:
                    tx += 1
                elif sl == 0:
                    ty += 1
                elif sp == sl:
                    c += 1
                else:
                    d += 1
        denom = np.sqrt((c + d + tx) * (c + d + ty))
        taus.append((c - d) / denom if denom > 0 else 0.0)
        lp = l[np.argmin(p)]
        hmin.append(float(lp == l.min()))
        hp = l[np.argmax(p)]
        hmax.append(float(hp == l.max()))
    return float(np.mean(taus)), float(np.mean(hmin)), float(np.mean(hmax))


def gather8(pred34, top8):
    """(N,34) -> (N,8)：按 top8 位置 gather（pad 位给 0）。"""
    return np.take_along_axis(pred34, np.clip(top8, 0, 33), axis=1)


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def forward_dealin(model, x):
    outs = model(x)
    # outs = (policy, value, dealin?, response?...)；iter92: dealin_head+response_head
    return outs[2] if len(outs) > 2 else None


@torch.no_grad()
def eval_danger(model, obs_t, top8, danger, valid8, bs=8192, max_n=20000):
    """top8/danger/valid8 为 numpy（与 obs_t 等长）；指标最多用 max_n 条（val 为
    随机排列切片，前缀即随机子集）。"""
    model.eval()
    obs_t, top8, danger, valid8 = (x[:max_n] for x in (obs_t, top8, danger, valid8))
    preds = []
    for s in range(0, len(obs_t), bs):
        preds.append(forward_dealin(model, obs_t[s:s + bs].float()).float().cpu().numpy())
    pred = np.concatenate(preds)
    pred8 = gather8(pred, top8)
    taub, hmin, hmax = rank_metrics(pred8, danger, valid8)
    mse = float(((pred8 - danger)[valid8] ** 2).mean())
    stats = (float(pred8[valid8].mean()), float(pred8[valid8].std()))
    return dict(taub=taub, hit_min=hmin, hit_max=hmax, mse=mse,
                pred_mean=stats[0], pred_std=stats[1])


def train_danger_head(model, data, tr, va, va_np, args, device, log):
    for k, v in model.named_parameters():
        v.requires_grad = k in DEALIN_KEYS
    opt = torch.optim.Adam([v for k, v in model.named_parameters() if v.requires_grad],
                           lr=args.lr)
    obs_t = torch.from_numpy(data['obs']).to(device).half()
    mask_t = torch.from_numpy(data['mask']).to(device)
    targ_t = torch.from_numpy(data['target']).to(device)
    n = len(tr)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = cnt = 0.0
        t0 = time.time()
        for s in range(0, n, args.batch):
            idx = tr[perm[s:s + args.batch]]
            x = obs_t[idx].float()
            pred = forward_dealin(model, x)
            m = mask_t[idx]
            loss = ((pred - targ_t[idx]) ** 2 * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            cnt += len(idx)
        va_m = eval_danger(model, obs_t[va], data['top8'][va_np],
                           data['danger'][va_np], data['valid8'][va_np])
        log(f'[danger] epoch {ep}/{args.epochs} train_mse={tot / cnt:.5f} '
            f'val_mse={va_m["mse"]:.5f} taub={va_m["taub"]:+.4f} '
            f'hit_min={va_m["hit_min"]:.4f} hit_max={va_m["hit_max"]:.4f} '
            f'({time.time() - t0:.1f}s)')
    return model


def train_policy(model, data, tr, va, args, device, log):
    opt = torch.optim.Adam(model.parameters(), lr=args.policy_lr)
    obs_t = torch.from_numpy(data['obs']).to(device).half()
    ch_t = torch.from_numpy(data['chosen']).to(device)
    legal_t = torch.from_numpy(data['legal']).to(device)
    n = len(tr)

    @torch.no_grad()
    def val_acc(idxs):
        model.eval()
        hit = cnt = 0
        for s in range(0, len(idxs), 8192):
            ii = idxs[s:s + 8192]
            logits = model(obs_t[ii].float())[0]
            logits = logits.masked_fill(~legal_t[ii], -1e9)
            hit += int((logits.argmax(-1) == ch_t[ii]).sum())
            cnt += len(ii)
        return hit / cnt

    log(f'[policy] baseline(iter92) val_acc={val_acc(va):.4f}')
    for ep in range(1, args.policy_epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = cnt = 0.0
        t0 = time.time()
        for s in range(0, n, args.batch):
            idx = tr[perm[s:s + args.batch]]
            logits = model(obs_t[idx].float())[0]
            logits = logits.masked_fill(~legal_t[idx], -1e9)
            loss = F.cross_entropy(logits, ch_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            cnt += len(idx)
        log(f'[policy] epoch {ep}/{args.policy_epochs} train_ce={tot / cnt:.4f} '
            f'val_acc={val_acc(va):.4f} ({time.time() - t0:.1f}s)')
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', default='output/belief_labels_shard_*.npz')
    ap.add_argument('--init', default='output/jax_gumbel_iter92.pt')
    ap.add_argument('--out', default='output/nn_danger_belief_v1.pt')
    ap.add_argument('--policy-out', default='output/nn_policy_distill_belief_v1.pt')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--policy-epochs', type=int, default=3)
    ap.add_argument('--policy-lr', type=float, default=1e-4)
    ap.add_argument('--val-frac', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--include-locked', action='store_true',
                    help='不剔除锁手状态（默认剔除）')
    ap.add_argument('--mem-frac', type=float, default=1.0,
                    help='torch.cuda.set_per_process_memory_fraction（与他任务共卡时限流）')
    args = ap.parse_args()

    torch.set_num_threads(8)
    if torch.cuda.is_available() and args.mem_frac < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_path = args.out.replace('.pt', '_train.log')
    log_f = open(log_path, 'a', buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + '\n')

    ck = torch.load(args.init, map_location='cpu')
    config = ck['config'] if isinstance(ck, dict) and 'config' in ck else None
    state = ck['model_state'] if isinstance(ck, dict) and 'model_state' in ck else ck
    assert config is not None, 'init checkpoint 缺少 config'
    log(f'[init] {args.init} config={config}')

    data = load_shards(args.shards, exclude_locked=not args.include_locked)
    n = len(data['chosen'])
    rng = np.random.default_rng(args.seed)
    perm = np.random.permutation(n)
    n_va = max(1000, int(n * args.val_frac))
    va_np = perm[:n_va]
    va = torch.from_numpy(va_np).to(device)
    tr = torch.from_numpy(perm[n_va:]).to(device)
    log(f'[split] train={len(tr)} val={len(va)}')

    # ---- danger 头（冻结 trunk）----
    model = build_model(config).to(device)
    model.load_state_dict(state)
    log('[danger] baseline（旧 dealin 头 + 现 trunk）指标：')
    obs_t = torch.from_numpy(data['obs']).to(device).half()
    base_m = eval_danger(model, obs_t[va], data['top8'][va_np],
                         data['danger'][va_np], data['valid8'][va_np])
    log(f'  OLD val_mse={base_m["mse"]:.5f} taub={base_m["taub"]:+.4f} '
        f'hit_min={base_m["hit_min"]:.4f} hit_max={base_m["hit_max"]:.4f} '
        f'pred_mean={base_m["pred_mean"]:.4f} pred_std={base_m["pred_std"]:.4f}')
    del obs_t

    t0 = time.time()
    model = train_danger_head(model, data, tr, va, va_np, args, device, log)
    log(f'[danger] trained in {time.time() - t0:.1f}s')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'config': config}, args.out)
    with open(args.out.replace('.pt', '_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    log(f'[save] {args.out} (+_config.json)')

    # ---- policy 蒸馏（sanity 组）----
    pmodel = build_model(config).to(device)
    pmodel.load_state_dict(state)
    t0 = time.time()
    pmodel = train_policy(pmodel, data, tr, va, args, device, log)
    log(f'[policy] trained in {time.time() - t0:.1f}s')
    torch.save({'model_state': pmodel.state_dict(), 'config': config},
               args.policy_out)
    with open(args.policy_out.replace('.pt', '_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    log(f'[save] {args.policy_out} (+_config.json)')
    log_f.close()


if __name__ == '__main__':
    main()
