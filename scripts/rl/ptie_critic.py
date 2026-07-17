# -*- coding: utf-8 -*-
"""方向 2：PTIE（Perfect-Training-Imperfect-Execution）完美信息 critic。

动机（docs/reports/web-research-directions-0717.md §3 方向 2）：
- v1 AWBC 失败的根因假设 = 不完美信息 critic corr 仅 0.231（信用分配 SNR 不足，
  `selfplay-bootstrap-0717.md` §2.5）；
- PerfectDou（NeurIPS 2022）的 PTIE：训练时 critic 看全部隐藏手牌，policy 部署时
  仍只看公开信息——critic 方差大幅降低，advantage 才有真信号；
- 本脚本：collect 时每个决策步额外记录「对手闭手 3×34 计数」（god 特征，
  同进程注入 `_table`，与 `oracle_endgame_gate.py` 同一模式），训练 god critic，
  用其 advantage 重跑 v1 的 AWBC 三候选（β=1.0/0.5、filtered）。

判读（预登记）：
- god critic val corr 应显著超过 v1 的 0.231（预期 0.5+）；若 corr 无大幅提升，
  说明 god 特征对 outcome 预测增益有限，PTIE 前提不成立；
- AWBC 候选仍按 eval-protocol：1000-pair duplicate 筛查，晋级线 +1.0%。

子命令：
  collect       同 selfplay_bootstrap collect，但每步多存 god(102,) 特征
  train_critic  冻结 best 主干 → gfeat + god → 新 value MLP；MSE 拟合 score/3
  finetune      同 selfplay_bootstrap finetune，但 V(s) 用 god critic

用法：
    PYTHONPATH=. python3 scripts/rl/ptie_critic.py collect \
        --games 24000 --workers 96 --out-prefix output/ptie_v1 \
        --init output/nn_full_action_best.pt
    PYTHONPATH=. python3 scripts/rl/ptie_critic.py train_critic \
        --data output/ptie_v1_merged.npz --init output/nn_full_action_best.pt \
        --out output/nn_ptie_v1_critic.pt --device cuda:0
    PYTHONPATH=. python3 scripts/rl/ptie_critic.py finetune \
        --data output/ptie_v1_merged.npz --init output/nn_full_action_best.pt \
        --critic output/nn_ptie_v1_critic.pt --mode awbc --beta 1.0 \
        --out output/nn_ptie_v1_awbc_b10.pt --device cuda:0
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from algo.rl.selfplay import build_net
from algo.nn.features import _TILE_TO_IDX
from scripts.rl import selfplay_bootstrap as sb

GOD_DIM = 3 * 34   # 三家对手闭手计数（next/face/prev 顺序）


# ---------------------------------------------------------------- collect

class PTIEActorAgent(sb.BootstrapActorAgent):
    """BootstrapActorAgent + 每个决策步记录对手闭手（god 特征）。

    `_table`/`_seat_idx` 由 _play_game 在同进程内注入。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._table = None
        self._seat_idx = 0

    def _god_vec(self):
        v = np.zeros(GOD_DIM, dtype=np.float32)
        if not self._table:
            return v
        n = len(self._table)
        for k in (1, 2, 3):
            opp = self._table[(self._seat_idx + k) % n]
            base = (k - 1) * 34
            for t in opp.cur:
                v[base + int(_TILE_TO_IDX[t])] += 1.0
        return v

    def next(self):
        r = super().next()
        if self.record and self.traj:
            self.traj[-1]['god'] = self._god_vec()
        return r

    def _record_response(self, tile_val, legal_mask, a):
        super()._record_response(tile_val, legal_mask, a)
        if self.record and self.traj_resp:
            self.traj_resp[-1]['god'] = self._god_vec()


def _play_game(net, cfg, seed, temperature):
    """4 个 PTIEActorAgent 自对弈，返回该局 4 条轨迹。"""
    from driver.engine import play_game
    random.seed(seed)
    np.random.seed(seed % (2 ** 31 - 1))
    agents = [PTIEActorAgent(f'A@{s}', net, cfg, device='cpu',
                             deterministic=False, temperature=temperature)
              for s in range(4)]
    for i, a in enumerate(agents):
        a._table = agents
        a._seat_idx = i
    result = play_game(agents)
    trajs = []
    from algo.rl.reward import terminal_reason
    for ag in agents:
        if not ag.traj and not ag.traj_resp:
            continue
        trajs.append({
            'steps': ag.traj,
            'resp': ag.traj_resp,
            'reason': terminal_reason(result, ag.name),
        })
    return trajs


def _flat_steps(trajs, game_id, temp_id):
    blk = sb._flat_steps(trajs, game_id, temp_id)
    if blk is None:
        return None
    gods, r_gods = [], []
    for tr in trajs:
        for s in tr['steps']:
            gods.append(s.get('god', np.zeros(GOD_DIM, dtype=np.float32)))
        for s in tr.get('resp', []):
            r_gods.append(s.get('god', np.zeros(GOD_DIM, dtype=np.float32)))
    if gods and 'feats' in blk:
        blk['god'] = np.asarray(gods, dtype=np.float32)
    if r_gods and 'resp_feats' in blk:
        blk['resp_god'] = np.asarray(r_gods, dtype=np.float32)
    return blk


_COUNTER = None


def _worker_init(counter):
    global _COUNTER
    _COUNTER = counter
    torch.set_num_threads(1)


def _shard_worker(args):
    (state_dict, config, n_games, seed_base, shard_path, progress_every,
     temps) = args
    torch.set_num_threads(1)
    net = build_net(state_dict, config, device='cpu')
    blocks = []
    for i in range(n_games):
        temp_id = i % len(temps)
        trajs = _play_game(net, config, seed_base + i, temps[temp_id])
        blk = _flat_steps(trajs, seed_base + i, temp_id)
        if blk is not None:
            blocks.append(blk)
        if _COUNTER is not None:
            with _COUNTER.get_lock():
                _COUNTER.value += 1
    if not blocks:
        raise RuntimeError(f'no data collected for {shard_path}')
    keys = set()
    for b in blocks:
        keys.update(b.keys())
    out = {k: np.concatenate([b[k] for b in blocks if k in b]) for k in keys}
    np.savez(shard_path, **out)
    return shard_path, len(out.get('actions', out.get('resp_actions')))


def cmd_collect(args):
    import multiprocessing as mp
    config = json.load(open(args.init.replace('.pt', '_config.json')))
    sd = {k: v.cpu() for k, v in sb._load_init(args.init).items()}

    per = int(np.ceil(args.games / args.workers))
    temps = sb._parse_temps(args.temps)
    tasks = []
    for w in range(args.workers):
        shard_path = f'{args.out_prefix}.shard{w:03d}.npz'
        if os.path.exists(shard_path):
            continue  # 断点续跑
        tasks.append((sd, config, per, args.seed_base + w * 1_000_000,
                      shard_path, args.progress_every, temps))
    total_planned = len(tasks) * per
    print(f'[collect] {args.games} games planned, {args.workers} workers, '
          f'{len(tasks)} shards todo, god features ON', flush=True)

    counter = mp.Value('l', 0)
    t0 = time.time()
    ctx = mp.get_context('fork')
    with ctx.Pool(args.workers, initializer=_worker_init, initargs=(counter,)) as pool:
        async_res = pool.map_async(_shard_worker, tasks)
        while not async_res.ready():
            async_res.wait(timeout=15)
            c = counter.value
            rate = c / max(time.time() - t0, 1e-9)
            eta = (total_planned - c) / max(rate, 1e-9)
            print(f'[collect] {c}/{total_planned} games '
                  f'({rate:.1f} g/s, ETA {eta/60:.1f} min)', flush=True)
        results = async_res.get()

    total_steps = sum(r[1] for r in results)
    print(f'[collect] done in {(time.time()-t0)/60:.1f} min, '
          f'new steps={total_steps}')

    shard_paths = [f'{args.out_prefix}.shard{w:03d}.npz'
                   for w in range(args.workers)]
    shard_paths = [p for p in shard_paths if os.path.exists(p)]
    print(f'[merge] {len(shard_paths)} shards')
    blocks = [np.load(p) for p in shard_paths]
    keys = set()
    for b in blocks:
        keys.update(b.files)
    merged = {k: np.concatenate([b[k] for b in blocks if k in b.files])
              for k in keys}
    out_path = f'{args.out_prefix}_merged.npz'
    np.savez(out_path, **merged)
    print(f'[merge] -> {out_path}: steps={len(merged["actions"])}, '
          f'god={merged["god"].shape if "god" in merged else None}')


# ------------------------------------------------------------ train_critic

class GodCritic(nn.Module):
    """冻结主干 gfeat + god 特征 → value MLP → tanh。"""

    def __init__(self, trunk, gfeat_dim, god_dim=GOD_DIM, hidden=512):
        super().__init__()
        self.trunk = trunk
        self.fc = nn.Linear(gfeat_dim + god_dim, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, god):
        _, gfeat = self.trunk._trunk(x)
        z = torch.cat([gfeat, god], dim=1)
        return torch.tanh(self.head(torch.relu(self.fc(z))))


def _gfeat_dim(net, cfg, device):
    input_dim = cfg.get('input_dim', 175)
    with torch.no_grad():
        dummy = torch.zeros(1, input_dim, device=device)
        _, gfeat = net._trunk(dummy)
    return gfeat.shape[1]


def cmd_train_critic(args):
    data = np.load(args.data)
    net, cfg = sb._build_from(args.init)
    device = args.device
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad = False   # 冻结主干

    gf_dim = _gfeat_dim(net, cfg, device)
    critic = GodCritic(net, gf_dim, hidden=args.hidden).to(device)
    print(f'[critic] gfeat={gf_dim} god={GOD_DIM} hidden={args.hidden}, '
          f'trainable={sum(p.numel() for p in critic.parameters() if p.requires_grad)}')

    scores = sb._score_from_code(data['reason_code'])
    tr_mask, va_mask = sb._split(data)
    X_tr = torch.from_numpy(data['feats'][tr_mask])
    G_tr = torch.from_numpy(data['god'][tr_mask])
    y_tr = torch.from_numpy(scores[tr_mask])
    X_va = torch.from_numpy(data['feats'][va_mask])
    G_va = torch.from_numpy(data['god'][va_mask])
    y_va = scores[va_mask]
    print(f'[critic] train={len(y_tr)} val={len(y_va)} '
          f'score mean={y_tr.mean():.4f} std={y_tr.std():.4f}')

    opt = torch.optim.Adam(critic.parameters(), lr=args.lr)
    N = len(y_tr)
    idx = np.arange(N)
    best_val = float('inf')
    best_state = None
    for ep in range(args.epochs):
        critic.train()
        np.random.shuffle(idx)
        for s in range(0, N, args.batch):
            mb = torch.from_numpy(idx[s:s + args.batch])
            pred = critic(X_tr[mb].to(device), G_tr[mb].to(device)).squeeze(-1)
            loss = F.mse_loss(pred, y_tr[mb].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        critic.eval()
        with torch.no_grad():
            preds = []
            for s in range(0, len(y_va), 8192):
                p = critic(X_va[s:s + 8192].to(device),
                           G_va[s:s + 8192].to(device)).squeeze(-1)
                preds.append(p.cpu().numpy())
            v_va = np.concatenate(preds)
        mse = float(np.mean((v_va - y_va) ** 2))
        corr = float(np.corrcoef(v_va, y_va)[0, 1])
        # decisive-only（非流局）corr
        dec = data['reason_code'][va_mask] != 4
        corr_dec = float(np.corrcoef(v_va[dec], y_va[dec])[0, 1]) if dec.sum() > 1 else 0.0
        print(f'[critic] epoch {ep}: val MSE={mse:.4f} corr={corr:.4f} '
              f'corr_decisive={corr_dec:.4f} '
              f'(baseline MSE={float(np.var(y_va)):.4f})', flush=True)
        if mse < best_val:
            best_val = mse
            best_state = {k: t.detach().cpu().clone()
                          for k, t in critic.state_dict().items()}

    print('[critic] calibration (mean V -> mean R):')
    for mv, mr, n in sb._calibration(v_va, y_va):
        print(f'  V={mv:+.3f}  R={mr:+.3f}  n={n}')

    critic.load_state_dict(best_state)
    torch.save({'critic_state_dict': critic.state_dict(),
                'init': args.init, 'gfeat_dim': gf_dim,
                'god_dim': GOD_DIM, 'hidden': args.hidden}, args.out)
    meta = {'init': args.init, 'gfeat_dim': gf_dim, 'god_dim': GOD_DIM,
            'hidden': args.hidden, 'source': 'ptie_god_critic'}
    json.dump(meta, open(args.out.replace('.pt', '_meta.json'), 'w'))
    print(f'[critic] saved {args.out}')


def load_god_critic(path, device):
    ckpt = torch.load(path, map_location='cpu')
    net, _ = sb._build_from(ckpt['init'])
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad = False
    critic = GodCritic(net, ckpt['gfeat_dim'], ckpt['god_dim'], ckpt['hidden'])
    critic.load_state_dict(ckpt['critic_state_dict'])
    critic.to(device).eval()
    return critic


# ------------------------------------------------------------- finetune

def cmd_finetune(args):
    data = np.load(args.data)
    device = args.device

    # 1) god critic 打分
    critic = load_god_critic(args.critic, device)
    scores = sb._score_from_code(data['reason_code'])
    tr_mask, va_mask = sb._split(data)
    N_all = len(scores)
    V = np.zeros(N_all, dtype=np.float32)
    feats_all = data['feats']
    god_all = data['god']
    with torch.no_grad():
        for s in range(0, N_all, 16384):
            x = torch.from_numpy(feats_all[s:s + 16384]).to(device)
            g = torch.from_numpy(god_all[s:s + 16384]).to(device)
            V[s:s + 16384] = critic(x, g).squeeze(-1).cpu().numpy()
    del critic
    A = scores - V
    a_mean, a_std = float(A[tr_mask].mean()), float(A[tr_mask].std())
    A_std = (A - a_mean) / max(a_std, 1e-6)
    print(f'[ft] A mean={a_mean:.4f} std={a_std:.4f}; '
          f'frac A>0 (train)={float((A[tr_mask] > 0).mean()):.3f}')

    if args.mode == 'awbc':
        W = np.exp(A_std / args.beta).astype(np.float32)
        W = np.clip(W, 0.0, args.wmax)
    elif args.mode == 'filtered':
        W = (A_std > 0).astype(np.float32)
    else:
        raise ValueError(args.mode)
    W = W / W[tr_mask].mean()
    W_tr = W[tr_mask]
    print(f'[ft] mode={args.mode} beta={args.beta} '
          f'w: p10={np.percentile(W_tr,10):.3f} p50={np.percentile(W_tr,50):.3f} '
          f'p90={np.percentile(W_tr,90):.3f} max={W_tr.max():.2f}')

    # 2) 从 init 微调 policy（与 sb.cmd_finetune 同配方）
    net, cfg = sb._build_from(args.init)
    net.to(device)
    ref_net, _ = sb._build_from(args.init)
    ref_net.to(device).eval()
    for p in ref_net.parameters():
        p.requires_grad = False

    X = data['feats']
    Act = data['actions']
    Msk = data['masks']
    tr_idx = np.where(tr_mask)[0]
    va_idx = np.where(va_mask)[0]
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def _eval_split(indices):
        net.eval()
        ces, agrees = [], []
        with torch.no_grad():
            for s in range(0, len(indices), 8192):
                mb = indices[s:s + 8192]
                x = torch.from_numpy(X[mb]).to(device)
                a = torch.from_numpy(Act[mb]).to(device)
                m = torch.from_numpy(Msk[mb]).to(device)
                logits = net(x)[0]
                ce, _ = sb._masked_ce(logits, m, a,
                                      torch.ones(len(mb), device=device))
                ref_logits = ref_net(x)[0]
                neg = (~m).float().mul(-1e9)
                agree = (logits + neg).argmax(1) == (ref_logits + neg).argmax(1)
                ces.append(float(ce) * len(mb))
                agrees.append(float(agree.float().mean()) * len(mb))
        return sum(ces) / len(indices), sum(agrees) / len(indices)

    ce0, ag0 = _eval_split(va_idx)
    print(f'[ft] before: val CE={ce0:.4f} argmax-agree={ag0:.4f}')

    for ep in range(args.epochs):
        net.train()
        np.random.shuffle(tr_idx)
        for s in range(0, len(tr_idx), args.batch):
            mb = tr_idx[s:s + args.batch]
            x = torch.from_numpy(X[mb]).to(device)
            a = torch.from_numpy(Act[mb]).to(device)
            m = torch.from_numpy(Msk[mb]).to(device)
            w = torch.from_numpy(W[mb].astype(np.float32)).to(device)
            logits = net(x)[0]
            loss, _ = sb._masked_ce(logits, m, a, w)
            if args.anchor_coef > 0:
                with torch.no_grad():
                    ref_logits = ref_net(x)[0]
                neg = (~m).float().mul(-1e9)
                logp = F.log_softmax(logits + neg, dim=1)
                ref_p = F.softmax(ref_logits + neg, dim=1)
                kl = (ref_p * (ref_p.clamp_min(1e-12).log() - logp)).sum(1).mean()
                loss = loss + args.anchor_coef * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
        ce_v, ag_v = _eval_split(va_idx)
        print(f'[ft] epoch {ep}: val CE={ce_v:.4f} argmax-agree={ag_v:.4f}',
              flush=True)

    torch.save(net.state_dict(), args.out)
    cfg_out = dict(cfg)
    cfg_out['source'] = f'ptie_{args.mode}_b{args.beta}'
    json.dump(cfg_out, open(args.out.replace('.pt', '_config.json'), 'w'))
    print(f'[ft] saved {args.out}')


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('collect')
    p.add_argument('--games', type=int, default=24000)
    p.add_argument('--workers', type=int, default=96)
    p.add_argument('--out-prefix', required=True)
    p.add_argument('--init', default='output/nn_full_action_best.pt')
    p.add_argument('--seed-base', type=int, default=0)
    p.add_argument('--temps', default='0.3,0.5,0.7')
    p.add_argument('--progress-every', type=int, default=15)
    p.set_defaults(fn=cmd_collect)

    p = sub.add_parser('train_critic')
    p.add_argument('--data', required=True)
    p.add_argument('--init', default='output/nn_full_action_best.pt')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-3)
    p.set_defaults(fn=cmd_train_critic)

    p = sub.add_parser('finetune')
    p.add_argument('--data', required=True)
    p.add_argument('--init', default='output/nn_full_action_best.pt')
    p.add_argument('--critic', required=True)
    p.add_argument('--mode', choices=['awbc', 'filtered'], default='awbc')
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--wmax', type=float, default=20.0)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--anchor-coef', type=float, default=0.0)
    p.set_defaults(fn=cmd_finetune)

    args = parser.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
