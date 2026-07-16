# -*- coding: utf-8 -*-
"""成对 rollout 评估「碰 vs 不碰」的因果效应（方向 F5，方案 E 的配对版）。

动机（docs/reports/selfplay-bootstrap-0717.md §2.5）：
outcome 级 RL（PPO/DPO/KTO/AWBC/AWR）五次失败，根因是终局 outcome 与单步决策
之间的信用分配 SNR 太低。本脚本把 SNR 打满：
- 在同一决策状态上，分别强制「碰」和「不碰」两个分支；
- 两个分支用**相同的 M 组牌山洗牌**（common random numbers）成对 rollout；
- rollout 用部署形态 agent（HybridNNBeliefAgent = NN + 搜索层），
  因此「碰 → 副露 → 本局失去搜索层」这一部署后果也被计入；
- Δ = meanR(碰) − meanR(不碰) 是该状态上碰的配对因果估计。

子命令：
  collect    用 Hybrid 自对弈，在每个碰决策点快照 god-state（全手牌/副露/
             context 副本/剩余牌山/locked），分 shard 存 pkl。
  evaluate   对快照做成对 rollout，输出每个状态的 meanR_pass/meanR_peng/Δ。
  train      用 |Δ|>τ 的状态生成 filtered 标签，微调 response head（冻主干）。

用法：
    PYTHONPATH=. python3 scripts/rl/peng_paired_eval.py collect \
        --games 3000 --workers 96 --out-prefix output/peng_states_v1
    PYTHONPATH=. python3 scripts/rl/peng_paired_eval.py evaluate \
        --states output/peng_states_v1_merged.pkl --out output/peng_eval_v1.npz \
        --max-states 8000 --rollouts 6 --workers 96
    PYTHONPATH=. python3 scripts/rl/peng_paired_eval.py train \
        --eval output/peng_eval_v1.npz --states output/peng_states_v1_merged.pkl \
        --init output/nn_full_action_best.pt --out output/nn_peng_paired_v1.pt
"""

import os
import sys
import json
import time
import math
import pickle
import random
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from agent import Message
from algo.eval.v2 import shanten
from driver.engine import (play_game, _discard_step, _process_claims,
                           _claim_peng, _claim_gang, _notify_meld, _WallPool)
from algo.rl.reward import terminal_reason
from scripts.rl.selfplay_bootstrap import _CODE_SCORE, _REASON_CODE

NUM_ACTIONS = 34


# ============================================================ collect

class _ProbeHybrid:
    """包一层 HybridNNBeliefAgent：在 respond_peng 被询问时快照 god-state。

    引擎只在 _can_peng 为真时才询问 respond_peng，且 hu/gang 阶段已无人生效
    （否则到不了碰阶段），所以这是合法的碰决策点。
    """

    def __init__(self, name, holder, seat_idx, **kw):
        from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent
        self._inner = HybridNNBeliefAgent(name, **kw)
        self._holder = holder
        self._seat = seat_idx

    def __getattr__(self, attr):
        return getattr(self._inner, attr)

    def respond_peng(self, tile_val, context=None):
        if self._can_peng(tile_val):
            _snapshot_peng(self._holder, tile_val, self._seat)
        return self._inner.respond_peng(tile_val, context)


def _snapshot_peng(holder, tile_val, claimer_idx):
    agents = holder['agents']
    info = holder['info']
    if info is None:
        return
    snap = {
        'hands': {i: list(ag.cur) for i, ag in enumerate(agents)},
        'melds': {i: [tuple(m) for m in ag.melds] for i, ag in enumerate(agents)},
        'contexts_nn': {i: ag.nn_agent.context.copy()
                        for i, ag in enumerate(agents)},
        'contexts_be': {i: ag.belief_agent.context.copy()
                        for i, ag in enumerate(agents)},
        'locked': sorted(holder['agents'][0].nn_agent.context.tenpai_players),
        'wall': list(info['wall']),
        'turn': holder['turn'],
        'claimer': claimer_idx,
        'tile': int(tile_val),
        'game_id': holder['game_id'],
    }
    holder['buf'].append(snap)


def _collect_shard(args):
    (nn_model_path, n_games, seed_base, shard_path) = args
    import torch
    torch.set_num_threads(1)
    buf = []
    for i in range(n_games):
        seed = seed_base + i
        holder = {'agents': None, 'info': None, 'turn': 0, 'buf': buf,
                  'game_id': seed}
        agents = [_ProbeHybrid(f'A@{s}', holder, s,
                               nn_model_path=nn_model_path,
                               belief_kind='beliefexp', tenpai_threshold=28,
                               device='cpu', temperature=0.0)
                  for s in range(4)]
        holder['agents'] = agents

        def _cb(ags, turn, kind, inf):
            holder['turn'] = turn
            holder['info'] = inf

        random.seed(seed)
        np.random.seed(seed % (2 ** 31 - 1))
        play_game(agents, seed=seed, state_callback=_cb)
    with open(shard_path, 'wb') as f:
        pickle.dump(buf, f)
    return shard_path, len(buf)


def cmd_collect(args):
    import multiprocessing as mp
    per = int(np.ceil(args.games / args.workers))
    tasks = []
    for w in range(args.workers):
        shard_path = f'{args.out_prefix}.shard{w:03d}.pkl'
        if os.path.exists(shard_path):
            continue
        tasks.append((args.nn_model, per, args.seed_base + w * 1_000_000,
                      shard_path))
    print(f'[collect] {args.games} games, {len(tasks)} shards todo')
    t0 = time.time()
    ctx = mp.get_context('fork')
    with ctx.Pool(args.workers) as pool:
        for j, (p, n) in enumerate(pool.imap_unordered(_collect_shard, tasks)):
            if (j + 1) % 4 == 0 or j + 1 == len(tasks):
                rate = (j + 1) / max(time.time() - t0, 1e-9)
                eta = (len(tasks) - j - 1) / max(rate, 1e-9)
                print(f'[collect] shards {j+1}/{len(tasks)} '
                      f'(ETA {eta/60:.1f} min)', flush=True)
    # merge
    shard_paths = [f'{args.out_prefix}.shard{w:03d}.pkl'
                   for w in range(args.workers)]
    all_states = []
    for p in shard_paths:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                all_states.extend(pickle.load(f))
    out_path = f'{args.out_prefix}_merged.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(all_states, f)
    print(f'[collect] -> {out_path}: {len(all_states)} peng states '
          f'in {(time.time()-t0)/60:.1f} min')


# ============================================================ evaluate

def _inject(agents, snap):
    """把快照状态注入 4 个 Hybrid agent（复用实例，init_tiles 会重置 context）。"""
    for i, ag in enumerate(agents):
        ag.init_tiles(list(snap['hands'][i]))
        melds = [tuple(m) for m in snap['melds'][i]]
        ag.melds = melds
        ag.nn_agent.melds = melds
        ag.belief_agent.melds = melds
        ag.nn_agent.context = snap['contexts_nn'][i].copy()
        ag.nn_agent._belief = None
        ag.belief_agent.context = snap['contexts_be'][i].copy()
        ag.belief_agent._fast_eval = None


def _play_from(agents, wall, start_turn, locked_names, skip_draw=False,
               entry_repl=False, max_steps=400):
    """从任意状态继续打完（复刻 engine 主循环，支持碰/杠后的入口状态）。"""
    wall = list(wall)
    wall_idx = 0
    turn = start_turn % len(agents)
    locked_names = set(locked_names)
    skip = skip_draw
    repl = entry_repl
    steps = 0
    while True:
        steps += 1
        if steps > max_steps:
            return {'winner': None, 'win_type': 'draw',
                    'players_order': [a.name for a in agents]}
        current = agents[turn]
        if not skip:
            if not wall or wall_idx >= len(wall):
                return {'winner': None, 'win_type': 'draw',
                        'players_order': [a.name for a in agents]}
            if repl:
                drawn = wall.pop()
                repl = False
            else:
                drawn = wall[wall_idx]
                wall_idx += 1
            if current.add(drawn):
                return {'winner': current.name, 'win_type': 'self',
                        'players_order': [a.name for a in agents]}
        else:
            skip = False
            drawn = None

        discarded, _, locked = _discard_step(
            current, drawn, locked_names, False, {}, False, [],
            lambda: len(wall) - wall_idx)

        if (not locked and current.name not in locked_names and
                len(current.full_hand()) == 13):
            if shanten(current.full_hand()) == 0:
                if current.declare_tenpai(current.cur,
                                          getattr(current, 'context', None)):
                    locked_names.add(current.name)
                    tenpai_msg = Message(current.name, 'tenpai', None)
                    for other in agents:
                        other.handle_msg(tenpai_msg)

        claim = _process_claims(agents, discarded, turn,
                                _WallPool(wall, wall_idx), locked_names,
                                [], False)
        if claim['type'] == 'win':
            return {'winner': claim['winner'], 'win_type': 'ron',
                    'dealer': claim['dealer'],
                    'players_order': [a.name for a in agents]}
        if claim['type'] == 'gang':
            turn = claim['claimer']
            repl = True
            continue
        if claim['type'] == 'peng':
            turn = claim['claimer']
            skip = True
            continue
        turn = (turn + 1) % len(agents)
        msg = Message(current.name, 'put', discarded)
        for other in agents:
            if other.name == current.name:
                continue
            other.handle_msg(msg)


def _claims_forced_pass(agents, discarded, discarder_turn, locked_names,
                        force_pass_idx):
    """复刻 engine._process_claims，但 force_pass_idx 玩家的碰被强制跳过。

    hu/gang 阶段与引擎一致（快照时已确定无人能 hu/gang——否则到不了碰阶段，
    且 rollout agent 的响应头是 argmax 确定性，会复现同样的「否」）。
    """
    n = len(agents)
    for offset in range(1, n):
        idx = (discarder_turn + offset) % n
        other = agents[idx]
        if other.respond_hu(discarded, getattr(other, 'context', None)):
            other.cur.append(discarded)
            return {'type': 'win', 'winner': other.name,
                    'dealer': agents[discarder_turn].name}
    for offset in range(1, n):
        idx = (discarder_turn + offset) % n
        other = agents[idx]
        if other.name in locked_names:
            continue
        if other._can_gang(discarded) and \
                other.respond_gang(discarded, getattr(other, 'context', None)):
            _claim_gang(other, discarded)
            _notify_meld(agents, other.name, 'gang', discarded)
            return {'type': 'gang', 'claimer': idx}
    for offset in range(1, n):
        idx = (discarder_turn + offset) % n
        other = agents[idx]
        if other.name in locked_names or idx == force_pass_idx:
            continue
        if other._can_peng(discarded) and \
                other.respond_peng(discarded, getattr(other, 'context', None)):
            _claim_peng(other, discarded)
            _notify_meld(agents, other.name, 'peng', discarded)
            return {'type': 'peng', 'claimer': idx}
    return {'type': 'pass'}


def _score_for(result, name):
    code = _REASON_CODE.get(terminal_reason(result, name), 0)
    return _CODE_SCORE[code]


def _run_branch(agents, snap, wall, branch):
    """跑一个分支，返回 claimer 的 score-proxy。agents 会被 _inject 重置。"""
    _inject(agents, snap)
    ci, di, T = snap['claimer'], snap['turn'], snap['tile']
    locked = set(snap['locked'])
    claimer_name = agents[ci].name
    if branch == 'peng':
        ag = agents[ci]
        _claim_peng(ag, T)
        _notify_meld(agents, ag.name, 'peng', T)
        result = _play_from(agents, wall, ci, locked, skip_draw=True)
    else:
        claim = _claims_forced_pass(agents, T, di, locked, ci)
        if claim['type'] == 'win':
            result = {'winner': claim['winner'], 'win_type': 'ron',
                      'dealer': claim['dealer'],
                      'players_order': [a.name for a in agents]}
        elif claim['type'] == 'gang':
            result = _play_from(agents, wall, claim['claimer'], locked,
                                entry_repl=True)
        elif claim['type'] == 'peng':
            result = _play_from(agents, wall, claim['claimer'], locked,
                                skip_draw=True)
        else:
            result = _play_from(agents, wall, (di + 1) % 4, locked)
    return _score_for(result, claimer_name)


_AGENTS = None


def _eval_worker_init(nn_model_path):
    import torch
    torch.set_num_threads(1)
    from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent
    global _AGENTS
    _AGENTS = [HybridNNBeliefAgent(f'A@{s}', nn_model_path=nn_model_path,
                                   belief_kind='beliefexp', tenpai_threshold=28,
                                   device='cpu', temperature=0.0)
               for s in range(4)]


def _eval_one_state(args):
    (snap, m_rollouts, seed) = args
    rng = random.Random(seed)
    wall0 = snap['wall']
    pass_scores, peng_scores = [], []
    for m in range(m_rollouts):
        wall_m = list(wall0)
        rng.shuffle(wall_m)
        sp = _run_branch(_AGENTS, snap, wall_m, 'pass')
        sg = _run_branch(_AGENTS, snap, wall_m, 'peng')
        pass_scores.append(sp)
        peng_scores.append(sg)
    pass_arr = np.asarray(pass_scores, dtype=np.float32)
    peng_arr = np.asarray(peng_scores, dtype=np.float32)
    return {
        'mean_pass': float(pass_arr.mean()),
        'mean_peng': float(peng_arr.mean()),
        'delta': float(peng_arr.mean() - pass_arr.mean()),
        'std_pass': float(pass_arr.std()),
        'std_peng': float(peng_arr.std()),
        'n_melds_claimer': len(snap['melds'][snap['claimer']]) // 3,
        'wall_len': len(wall0),
    }


def cmd_evaluate(args):
    import multiprocessing as mp
    with open(args.states, 'rb') as f:
        states = pickle.load(f)
    rng = random.Random(12345)
    idx_all = list(range(len(states)))
    rng.shuffle(idx_all)
    sel = idx_all[:args.max_states]
    states_sel = [states[i] for i in sel]
    print(f'[eval] {len(states_sel)} states (of {len(states)}), '
          f'M={args.rollouts} rollouts/branch, {args.workers} workers')
    tasks = [(snap, args.rollouts, 987654 + i)
             for i, snap in enumerate(states_sel)]
    t0 = time.time()
    ctx = mp.get_context('fork')
    results = []
    with ctx.Pool(args.workers,
                  initializer=_eval_worker_init,
                  initargs=(args.nn_model,)) as pool:
        for j, r in enumerate(pool.imap(_eval_one_state, tasks, chunksize=4)):
            results.append(r)
            if (j + 1) % 200 == 0 or j + 1 == len(tasks):
                rate = (j + 1) / max(time.time() - t0, 1e-9)
                eta = (len(tasks) - j - 1) / max(rate, 1e-9)
                print(f'[eval] {j+1}/{len(tasks)} states '
                      f'({rate:.1f} st/s, ETA {eta/60:.1f} min)', flush=True)
    keys = results[0].keys()
    out = {k: np.array([r[k] for r in results]) for k in keys}
    out['state_idx'] = np.array(sel, dtype=np.int64)
    np.savez(args.out, **out)
    d = out['delta']
    print(f'[eval] done in {(time.time()-t0)/60:.1f} min')
    print(f'[eval] delta: mean={d.mean():+.4f} std={d.std():.4f} '
          f'p10={np.percentile(d,10):+.3f} p50={np.percentile(d,50):+.3f} '
          f'p90={np.percentile(d,90):+.3f}')
    for nm in (0, 1, 2, 3):
        sel = out['n_melds_claimer'] == nm
        if sel.sum():
            print(f'  melds={nm}: n={sel.sum()} mean delta={d[sel].mean():+.4f}')


# ============================================================ train

def cmd_train(args):
    import torch
    import torch.nn.functional as F
    from algo.nn.features import extract_features
    from scripts.rl.selfplay_bootstrap import _build_from

    ev = np.load(args.eval)
    with open(args.states, 'rb') as f:
        states = pickle.load(f)
    sel = ev['state_idx']
    states = [states[int(i)] for i in sel]
    assert len(states) == len(ev['delta'])

    d = ev['delta']
    # 配对 SE：两分支各 M 个 rollout，用配对差分的经验标准误
    # 近似：se_delta ≈ std(delta)/sqrt(n) 用于总体；单状态阈值用合并噪声
    m = int(args.rollouts)
    se_state = np.sqrt((ev['std_pass'] ** 2 + ev['std_peng'] ** 2) / m)
    tau = float(np.median(se_state) * args.tau_mult)
    sig = np.abs(d) > tau
    print(f'[train] tau={tau:.4f} (median per-state SE={np.median(se_state):.4f}); '
          f'significant: {sig.sum()}/{len(d)} '
          f'(peng better: {(d > tau).sum()}, pass better: {(d < -tau).sum()})')
    print(f'[train] mean delta overall={d.mean():+.4f} '
          f'(SE of mean={d.std()/np.sqrt(len(d)):.4f})')

    # 构造标签：Δ>τ -> peng(1)，Δ<-τ -> pass(0)
    # one-sided（pass_only）：只用「头部当前会碰 且 Δ<−τ」的状态，目标=pass；
    # 不教「head-pass 但 Δ>0」的状态（rollout 盲区的稳健性优先，见报告 §2.10）
    feats, targets, weights = [], [], []
    head_preds = None
    if args.mode in ('pass_only', 'passfix_anchor'):
        import torch as _t
        _net0, _ = _build_from(args.init)
        _net0.eval()
        if args.device.startswith('cuda'):
            _net0 = _net0.to(args.device)
        _feats0 = []
        for snap in states:
            h = list(snap['hands'][snap['claimer']]) + \
                [t for _, t in snap['melds'][snap['claimer']]] + [snap['tile']]
            _feats0.append(extract_features(
                snap['contexts_nn'][snap['claimer']], h,
                f'A@{snap["claimer"]}'))
        _X0 = _t.from_numpy(np.asarray(_feats0, dtype=np.float32))
        head_preds = []
        with _t.no_grad():
            for st in range(0, len(_X0), 8192):
                out = _net0(_X0[st:st + 8192].to(args.device))
                for o in out:
                    if o.shape[-1] == 4:
                        head_preds.append(o[:, :2].argmax(1).cpu().numpy())
                        break
        head_preds = np.concatenate(head_preds)
        del _net0
    for i, snap in enumerate(states):
        if not sig[i]:
            continue
        if args.mode == 'pass_only':
            if not (head_preds[i] == 1 and d[i] < -tau):
                continue
        if args.mode == 'passfix_anchor':
            # 只在 head-peng 状态内重标定：Δ<−τ → pass（修错），Δ>+τ → peng（锚）
            if head_preds[i] != 1:
                continue
        ag_hand = list(snap['hands'][snap['claimer']]) + \
                  [t for _, t in snap['melds'][snap['claimer']]] + [snap['tile']]
        ctx = snap['contexts_nn'][snap['claimer']]
        f = extract_features(ctx, ag_hand, f'A@{snap["claimer"]}')
        feats.append(np.asarray(f, dtype=np.float32))
        targets.append(1 if d[i] > tau else 0)
        weights.append(min(1.0, abs(d[i]) / (2 * tau)))
    feats = np.asarray(feats, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    print(f'[train] labeled samples: {len(targets)} '
          f'(peng {int(targets.sum())}, pass {int((1 - targets).sum())})')
    if len(targets) < 100:
        print('[train] too few significant states, abort')
        return

    device = args.device
    net, cfg = _build_from(args.init)
    net.to(device)
    for name, p in net.named_parameters():
        p.requires_grad = name.startswith('response_')
    trainable = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr)

    def _resp_logits(model, x):
        for o in model(x):
            if o.shape[-1] == 4:
                return o
        raise RuntimeError('no response head')

    X = torch.from_numpy(feats)
    Y = torch.from_numpy(targets)
    Wt = torch.from_numpy(weights)
    N = len(Y)
    idx = np.arange(N)
    for ep in range(args.epochs):
        net.train()
        np.random.shuffle(idx)
        tot = 0.0
        for s in range(0, N, args.batch):
            mb = torch.from_numpy(idx[s:s + args.batch])
            x = X[mb].to(device)
            y = Y[mb].to(device)
            w = Wt[mb].to(device)
            logits = _resp_logits(net, x)
            # 只允许 pass(0)/peng(1) 两类
            logits2 = logits[:, :2]
            ce = F.cross_entropy(logits2, y, reduction='none')
            loss = (ce * w).sum() / w.sum().clamp_min(1e-9)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 0.5)
            opt.step()
            tot += float(loss) * len(mb)
        # 诊断：对显著状态的预测准确率
        net.eval()
        with torch.no_grad():
            correct = 0
            for s in range(0, N, 8192):
                x = X[s:s + 8192].to(device)
                pred = _resp_logits(net, x)[:, :2].argmax(1).cpu()
                correct += int((pred == Y[s:s + 8192]).sum())
        print(f'[train] epoch {ep}: loss={tot/N:.4f} '
              f'label-acc={correct/N:.3f}', flush=True)

    torch.save(net.state_dict(), args.out)
    cfg_out = dict(cfg)
    cfg_out['source'] = 'peng_paired_filtered'
    json.dump(cfg_out, open(args.out.replace('.pt', '_config.json'), 'w'))
    print(f'[train] saved {args.out}')


# ============================================================ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('collect')
    p.add_argument('--games', type=int, default=3000)
    p.add_argument('--workers', type=int, default=96)
    p.add_argument('--out-prefix', required=True)
    p.add_argument('--nn-model', default='output/nn_full_action_best.pt')
    p.add_argument('--seed-base', type=int, default=70_000_000)
    p.set_defaults(fn=cmd_collect)

    p = sub.add_parser('evaluate')
    p.add_argument('--states', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-states', type=int, default=8000)
    p.add_argument('--rollouts', type=int, default=6)
    p.add_argument('--workers', type=int, default=96)
    p.add_argument('--nn-model', default='output/nn_full_action_best.pt')
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser('train')
    p.add_argument('--eval', required=True)
    p.add_argument('--states', required=True)
    p.add_argument('--init', default='output/nn_full_action_best.pt')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--rollouts', type=int, default=6)
    p.add_argument('--tau-mult', type=float, default=2.0)
    p.add_argument('--mode', choices=['both', 'pass_only', 'passfix_anchor'],
                   default='both',
                   help='both=双向标签；pass_only=只修「头部会碰且 Δ<−τ」；'
                        'passfix_anchor=修错+peng 锚（不碰 head-pass 状态）')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch', type=int, default=2048)
    p.set_defaults(fn=cmd_train)

    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
