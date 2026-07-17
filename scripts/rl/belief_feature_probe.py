# -*- coding: utf-8 -*-
"""Belief 特征探针：检验「坏碰状态」在扩展特征下是否可分（方向 F 的判官）。

背景（docs/reports/selfplay-bootstrap-0717.md §2.9-§3）：
F5 成对 rollout 测得 875/12000 个「头部会碰但因果上不该碰」的状态
（Δ<−0.5，mean −0.85），但在 175 维基础特征上 label-acc 上限 ~0.7，
修复尝试全部失败或为零。本探针回答：加入 belief/eval 信号
（tile_danger / player_danger_level / shanten / ukeire 等，即 BeliefExp
搜索层在决策时使用的信息）后，这些状态是否变得可分。

判定规则（预登记）：
- base+belief 相对 base 的 AUC 提升 < 0.05，或绝对 AUC < 0.75
  → 特征扩容方向关闭；
- AUC 提升 ≥ 0.05 且 base+belief MLP ≥ 0.75
  → 值得做特征扩容重训（方向 F 立项）。

数据：output/peng_states_v1_merged.pkl + output/peng_eval_v1.npz
（12k 状态，god-state 快照 + 配对 Δ）。
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch

from algo.eval.v2 import shanten, is_win, ukeire, winning_tiles
from algo.eval.opponent import (tile_danger, tile_danger_for_player,
                                player_danger_level, discard_safety)
from algo.nn.features import extract_features

BELIEF_NAMES = [
    'shanten_full', 'shanten_after', 'peng_reduces_shanten',
    'ukeire_full', 'ukeire_after', 'wait_sum_full', 'wait_sum_after',
    'tenpai_full', 'tenpai_after', 'win_now',
    'danger_max_cur', 'danger_mean_cur', 'danger_frac_high', 'min_safety_cur',
    'opp_level_next', 'opp_level_across', 'opp_level_prev',
    'opp_level_max', 'opp_level_sum', 'opp_tenpai_declared',
    'wall_len', 'n_melds', 'total_discards',
    'tile_is_honor', 'tile_rank', 'danger_T_next',
]


def _post_peng_full_hand(snap):
    """碰后的完整手牌表示（cur−2T + 既有副露(每个3张已是正确张数) + TTT）。"""
    ci, T = snap['claimer'], snap['tile']
    cur = list(snap['hands'][ci])
    cur.remove(T)
    cur.remove(T)
    meld_tiles = [t for _, t in snap['melds'][ci]]
    return cur + meld_tiles + [T, T, T]


def _belief_features(snap):
    ci, T = snap['claimer'], snap['tile']
    ctx = snap['contexts_nn'][ci]
    name = f'A@{ci}'
    cur = list(snap['hands'][ci])
    meld_tiles = [t for _, t in snap['melds'][ci]]
    full = cur + meld_tiles
    post = _post_peng_full_hand(snap)

    sh_full = shanten(full)
    sh_post = shanten(post)
    remaining_full = ctx.remaining_wall(full)
    remaining_post = ctx.remaining_wall(post)
    uk_full = ukeire(full, remaining_full)
    uk_post = ukeire(post, remaining_post)
    wt_full = winning_tiles(full, remaining_full)
    wt_post = winning_tiles(post, remaining_post)
    wait_full = sum(remaining_full.get(t, 0) for t in wt_full)
    wait_post = sum(remaining_post.get(t, 0) for t in wt_post)

    dangers = [tile_danger(t, ctx, name) for t in set(cur)]
    safeties = [discard_safety(t, ctx) for t in set(cur)]

    seats = [f'A@{(ci + k) % 4}' for k in (1, 2, 3)]
    opp_levels = [player_danger_level(ctx.discards.get(p, [])) for p in seats]
    tenpai_declared = sum(1 for p in seats if p in ctx.tenpai_players)

    return np.array([
        sh_full, sh_post, sh_full - sh_post,
        uk_full, uk_post, wait_full, wait_post,
        float(sh_full == 0), float(sh_post == 0),
        float(is_win(full + [T])),
        max(dangers) if dangers else 0.0,
        float(np.mean(dangers)) if dangers else 0.0,
        float(np.mean([d > 1.0 for d in dangers])) if dangers else 0.0,
        min(safeties) if safeties else 0.0,
        *opp_levels,
        max(opp_levels), sum(opp_levels), tenpai_declared,
        float(len(snap['wall'])), float(len(snap['melds'][ci]) // 3),
        float(sum(len(v) for v in ctx.discards.values())),
        float(T >= 31), float(T % 10 if T < 30 else 0),
        tile_danger_for_player(T, seats[0], ctx),
    ], dtype=np.float32)


def _auc(scores, labels):
    """Mann-Whitney AUC（处理并列：平均秩）。"""
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 并列秩平均
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _fit_torch(Xtr, ytr, Xte, hidden=0, epochs=300, lr=1e-2, device='cuda:0'):
    """logistic（hidden=0）或单隐层 MLP；返回测试集得分。"""
    d = Xtr.shape[1]
    layers = []
    if hidden > 0:
        layers += [torch.nn.Linear(d, hidden), torch.nn.ReLU(),
                   torch.nn.Linear(hidden, 1)]
    else:
        layers += [torch.nn.Linear(d, 1)]
    model = torch.nn.Sequential(*layers).to(device)
    # 类别不均衡：pos_weight
    pw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pw, device=device))
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).float().to(device)
    Xte_t = torch.from_numpy(Xte).to(device)
    model.train()
    n = len(ytr)
    idx = np.arange(n)
    bs = 4096
    for ep in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, n, bs):
            mb = torch.from_numpy(idx[s:s + bs]).to(device)
            out = model(Xtr_t[mb]).squeeze(-1)
            loss = lossf(out, ytr_t[mb])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        scores = model(Xte_t).squeeze(-1).cpu().numpy()
    return scores, model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--states', default='output/peng_states_v1_merged.pkl')
    ap.add_argument('--eval', default='output/peng_eval_v1.npz')
    ap.add_argument('--init', default='output/nn_full_action_best.pt')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--delta-thr', type=float, default=-0.5)
    args = ap.parse_args()

    ev = np.load(args.eval)
    with open(args.states, 'rb') as f:
        states_all = pickle.load(f)
    states = [states_all[int(i)] for i in ev['state_idx']]
    d = ev['delta']
    print(f'[probe] {len(states)} states')

    # 1) 特征矩阵
    print('[probe] computing base + belief features ...')
    feats_base, feats_belief, game_ids = [], [], []
    for i, snap in enumerate(states):
        ci = snap['claimer']
        hand = list(snap['hands'][ci]) + \
               [t for _, t in snap['melds'][ci]] + [snap['tile']]
        feats_base.append(extract_features(
            snap['contexts_nn'][ci], hand, f'A@{ci}'))
        feats_belief.append(_belief_features(snap))
        game_ids.append(snap['game_id'])
        if (i + 1) % 3000 == 0:
            print(f'  {i+1}/{len(states)}', flush=True)
    Xb = np.asarray(feats_base, dtype=np.float32)
    Xe = np.asarray(feats_belief, dtype=np.float32)
    game_ids = np.asarray(game_ids)

    # 2) 头部预测（界定可修复区域）+ 头部 peng logit 作为知识基线
    from scripts.rl.selfplay_bootstrap import _build_from
    net, _ = _build_from(args.init)
    net.to(args.device).eval()
    peng_logit, pass_logit = [], []
    with torch.no_grad():
        Xt = torch.from_numpy(Xb)
        for s in range(0, len(Xt), 8192):
            out = net(Xt[s:s + 8192].to(args.device))
            for o in out:
                if o.shape[-1] == 4:
                    peng_logit.append(o[:, 1].cpu().numpy())
                    pass_logit.append(o[:, 0].cpu().numpy())
                    break
    peng_logit = np.concatenate(peng_logit)
    pass_logit = np.concatenate(pass_logit)
    head_peng = peng_logit > pass_logit
    head_margin = peng_logit - pass_logit
    del net

    # 3) 标签与总体：主分析 = head-peng 子集（部署可修复区）
    y_all = (d < args.delta_thr).astype(np.int64)
    pop = head_peng
    y = y_all[pop]
    print(f'[probe] head-peng subset: {pop.sum()} states, '
          f'positives (delta<{args.delta_thr}): {y.sum()} ({y.mean():.1%})')

    # 4) 按 game_id 切分 train/test（防同局泄漏）
    gid = game_ids[pop]
    uniq = np.unique(gid)
    rng = np.random.RandomState(7)
    rng.shuffle(uniq)
    n_test = int(len(uniq) * 0.2)
    test_games = set(uniq[:n_test].tolist())
    te = np.array([g in test_games for g in gid])
    tr = ~te

    Xb_p, Xe_p = Xb[pop], Xe[pop]
    margin_p = head_margin[pop].reshape(-1, 1)

    def _std(Xtr, Xte):
        mu = Xtr.mean(0, keepdims=True)
        sd = Xtr.std(0, keepdims=True) + 1e-6
        return (Xtr - mu) / sd, (Xte - mu) / sd

    sets = {
        'base175': Xb_p,
        'belief24': Xe_p,
        'base+belief': np.concatenate([Xb_p, Xe_p], axis=1),
        'head_margin': margin_p,
        'margin+belief': np.concatenate([margin_p, Xe_p], axis=1),
    }

    print(f'[probe] train={tr.sum()} test={te.sum()} '
          f'test base-rate={y[te].mean():.1%}')
    print(f'{"feature set":>16s} {"logistic":>9s} {"MLP-64":>9s}')
    table = {}
    for name, X in sets.items():
        Xtr, Xte = _std(X[tr], X[te])
        sc_lin, _ = _fit_torch(Xtr, y[tr], Xte, hidden=0, device=args.device)
        a_lin = _auc(sc_lin, y[te])
        if X.shape[1] >= 4:
            sc_mlp, _ = _fit_torch(Xtr, y[tr], Xte, hidden=64,
                                   epochs=200, device=args.device)
            a_mlp = _auc(sc_mlp, y[te])
        else:
            a_mlp = float('nan')
        table[name] = (a_lin, a_mlp)
        print(f'{name:>16s} {a_lin:9.4f} {a_mlp:9.4f}', flush=True)

    # 5) belief 特征的逻辑回归权重（解释性）
    Xtr, Xte = _std(Xe_p[tr], Xe_p[te])
    _, lin_model = _fit_torch(Xtr, y[tr], Xte, hidden=0, device=args.device)
    w = lin_model[0].weight.detach().cpu().numpy().reshape(-1)
    order = np.argsort(-np.abs(w))
    print('\n[probe] belief-only logistic weights (top 10):')
    for i in order[:10]:
        print(f'  {BELIEF_NAMES[i]:>22s} {w[i]:+.3f}')

    # 6) 判定
    a_base = table['base175'][1]
    a_bb = table['base+belief'][1]
    print(f'\n[probe] verdict: base+belief − base = {a_bb - a_base:+.4f}, '
          f'base+belief MLP = {a_bb:.4f}')
    if a_bb >= 0.75 and (a_bb - a_base) >= 0.05:
        print('[probe] => PASS: belief 特征带来可分的边际信息，方向 F 可立项')
    else:
        print('[probe] => FAIL: 边际信息不足，特征扩容方向建议关闭')


if __name__ == '__main__':
    main()
