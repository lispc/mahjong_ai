# -*- coding: utf-8 -*-
"""生成 Perfect-Info Safety Oracle 数据。

Oracle 决策：在当前决策点，用完美信息（能看见对手手牌）避免点炮：
- 枚举所有合法弃牌；
- 标记会立即点炮的弃牌（任一对手手牌加该牌即胡）；
- 在不会点炮的弃牌中，选 conv-BC policy 分数最高者；
- 若所有弃牌都会点炮，选点炮损失最小的（或 conv-BC 最高分）。

这个 oracle 显著降低点炮率，适合作为防守型 oracle guiding 教师。

输出 .npz 包含 Xn, Xo, y, v（与 gen_oracle_data.py 兼容）。

用法：
    PYTHONPATH=. python3 scripts/rl/gen_oracle_safety_data.py \
        output/nn_teacher_oracle_safety.npz 500 32 output/nn_conv_bc.pt 0
"""

import sys
import os
import json
import random
import numpy as np
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tile_pool
import tile
import algo.eval.v2 as eval_v2
import algo.context.v3 as context_v3
from algo.nn.features import extract_features, extract_features_oracle, _TILE_TO_IDX, _IDX_TO_TILE
from algo.nn.model import build_model

# 多线程推理：避免 OpenMP 超订与 torch+fork 死锁，单线程内核心效率更高
torch.set_num_threads(1)


NUM_ACTIONS = 34

GLOBAL_NET = None
GLOBAL_CFG = None
GLOBAL_DEVICE = None


def _load_conv_bc(path):
    cfg_path = path.replace('.pt', '_config.json')
    cfg = json.load(open(cfg_path))
    net = build_model(cfg)
    sd = torch.load(path, map_location='cpu')
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    net.load_state_dict(sd)
    net.eval()
    return net, cfg


def _conv_bc_scores(hand14, ctx, name, net, extract, device):
    feats = extract(ctx, hand14, name)
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = net(x)
    logits = logits.squeeze(0).detach().cpu().numpy()
    from algo.nn.features import _TILE_TO_IDX
    legal = np.zeros(34, dtype=np.float32)
    for t in hand14:
        legal[int(_TILE_TO_IDX[t])] = 1.0
    masked = logits + (legal - 1.0) * 1e9
    return masked


def _oracle_safety_discard(hand14, all_hands, current_name, net, extract, device):
    """用完美信息安全规则选弃牌。"""
    player_names = sorted(all_hands.keys(), key=lambda n: int(n.split('@')[-1]))
    opponents = [p for p in player_names if p != current_name]

    unique = sorted(set(hand14))
    safe = []
    unsafe = []
    for cand in unique:
        hand13 = list(hand14)
        hand13.remove(cand)
        deals_in = False
        for opp in opponents:
            if eval_v2.is_win(list(all_hands[opp]) + [cand]):
                deals_in = True
                break
        if deals_in:
            unsafe.append(cand)
        else:
            safe.append(cand)

    scores = _conv_bc_scores(hand14, context_v3.ContextV3(), current_name, net, extract, device)
    if safe:
        best = safe[0]
        best_score = -float('inf')
        for cand in safe:
            idx = int(_TILE_TO_IDX[cand])
            if scores[idx] > best_score:
                best_score = scores[idx]
                best = cand
        return best
    # 所有都点炮：选 conv-BC 最高分（无奈）
    best = unsafe[0]
    best_score = -float('inf')
    for cand in unsafe:
        idx = int(_TILE_TO_IDX[cand])
        if scores[idx] > best_score:
            best_score = scores[idx]
            best = cand
    return best


def _base_greedy_discard(hand14, ctx, name, net, extract, device):
    """conv-BC greedy discard（无安全规则），用于 mixed 模式中的普通玩家。"""
    scores = _conv_bc_scores(hand14, ctx, name, net, extract, device)
    best = None
    best_score = -float('inf')
    for t in hand14:
        idx = int(_TILE_TO_IDX[t])
        if scores[idx] > best_score:
            best_score = scores[idx]
            best = t
    return best


def _play_one_game(net, cfg, device, seed, mixed=False):
    random.seed(seed)
    np.random.seed(seed % (2 ** 31 - 1))

    pool = tile_pool.Pool()
    names = [f'P@{i}' for i in range(4)]
    hands = {n: pool.next_n(13) for n in names}
    ctx = context_v3.ContextV3()
    locked = set()
    turn = 0
    wall = list(pool.tiles[pool.idx:])
    oracle_seat = (seed % 4) if mixed else None

    extract = extract_features_ext if cfg.get('features') == 'ext' else extract_features
    samples = []
    outcome = {n: 0.0 for n in names}

    while True:
        if not wall:
            for n in names:
                outcome[n] = 0.0
            break
        pname = names[turn]
        drawn = wall.pop(0)
        hands[pname].append(drawn)

        if eval_v2.is_win(hands[pname]):
            for n in names:
                outcome[n] = 1.0 if n == pname else -1.0
            full_samples = []
            for xn, xo, y, pname in samples:
                full_samples.append((xn, xo, y, outcome.get(pname, 0.0)))
            return full_samples, outcome

        if pname in locked:
            discarded = drawn
            hands[pname].remove(discarded)
        else:
            is_oracle = (not mixed) or (turn == oracle_seat)
            if is_oracle:
                xn = extract_features(ctx, hands[pname], pname)
                xo = extract_features_oracle(ctx, hands[pname], pname, hands, wall)
                discarded = _oracle_safety_discard(hands[pname], hands, pname, net, extract, device)
                samples.append((xn, xo, int(_TILE_TO_IDX[discarded]), pname))
            else:
                discarded = _base_greedy_discard(hands[pname], ctx, pname, net, extract, device)
            hands[pname].remove(discarded)

        for j, other in enumerate(names):
            if j == turn:
                continue
            if eval_v2.is_win(hands[other] + [discarded]):
                for n in names:
                    if n == other:
                        outcome[n] = 1.0
                    else:
                        outcome[n] = -1.0
                full_samples = []
                for xn, xo, y, pname in samples:
                    full_samples.append((xn, xo, y, outcome.get(pname, 0.0)))
                return full_samples, outcome

        ctx.see_tile(discarded, pname)

        if (pname not in locked and len(hands[pname]) == 13 and
                eval_v2.shanten(hands[pname]) == 0):
            rem = ctx.remaining_wall(hands[pname])
            waits = eval_v2.winning_tiles(hands[pname], rem)
            if sum(rem.get(t, 0) for t in waits) >= 3:
                locked.add(pname)
                ctx.declare_tenpai(pname)

        turn = (turn + 1) % 4

    full_samples = []
    for xn, xo, y, pname in samples:
        full_samples.append((xn, xo, y, outcome.get(pname, 0.0)))
    return full_samples, outcome


def _thread_worker(args):
    """ThreadPool worker：复用主线程加载的模型。"""
    i, s, mixed = args
    return _play_one_game(GLOBAL_NET, GLOBAL_CFG, GLOBAL_DEVICE, s, mixed=mixed)[0]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'output/nn_teacher_oracle_safety.npz'
    total_games = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    model_path = sys.argv[4] if len(sys.argv) > 4 else 'output/nn_conv_bc.pt'
    seed_base = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    mixed = os.environ.get('SAFETY_MIXED', '0') == '1'

    device = 'cpu'
    net, cfg = _load_conv_bc(model_path)
    net.to(device)
    print(f'Oracle safety data: {total_games} games, {workers} workers, mixed={mixed}')

    t0 = time.time()
    all_samples = []
    if workers <= 1:
        for i in range(total_games):
            samples, _ = _play_one_game(net, cfg, device, seed_base + i, mixed=mixed)
            all_samples.extend(samples)
    else:
        global GLOBAL_NET, GLOBAL_CFG, GLOBAL_DEVICE
        GLOBAL_NET = net
        GLOBAL_CFG = cfg
        GLOBAL_DEVICE = device
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_thread_worker,
                                        [(i, seed_base + i, mixed) for i in range(total_games)]))
        for r in results:
            all_samples.extend(r)
    dt = time.time() - t0
    print(f'Generated {len(all_samples)} samples from {total_games} games in {dt:.1f}s')

    if not all_samples:
        print('No samples generated')
        return

    Xn = np.stack([s[0] for s in all_samples])
    Xo = np.stack([s[1] for s in all_samples])
    y = np.array([s[2] for s in all_samples], dtype=np.int64)
    v = np.array([s[3] for s in all_samples], dtype=np.float32)
    np.savez(out_path, Xn=Xn, Xo=Xo, y=y, v=v)
    print(f'Saved {out_path}: Xn={Xn.shape}, Xo={Xo.shape}, y={y.shape}, v={v.shape}')


if __name__ == '__main__':
    main()
