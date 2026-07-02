# -*- coding: utf-8 -*-
"""生成 Perfect-Info Rollout Oracle 数据。

Rollout Oracle：在当前决策点，用完美信息（知道对手手牌和牌山）评估每个合法弃牌：
- 排除会立即点炮的弃牌；
- 对剩余弃牌，从该状态开始用 conv-BC greedy 跑 N 次随机 wall 顺序的完整 rollout，
  取当前玩家最终 outcome 的均值；
- 选平均 outcome 最高的弃牌作为 oracle 动作。

输出 .npz：Xn, Xo, y, v（与 gen_oracle_data.py 兼容，可用于 pretrain_bc / distill_oracle）。

用法：
    PYTHONPATH=. python3 scripts/rl/gen_rollout_oracle_data.py \
        output/nn_teacher_rollout_oracle_100.npz 100 8 output/nn_conv_bc.pt 2 30000000

参数：out_path n_games n_workers model_path n_rollouts seed_base
"""

import sys
import os
import json
import random
import copy
import numpy as np
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tile_pool
import tile
import algo.eval.v2 as eval_v2
import algo.context.v3 as context_v3
from algo.nn.features import extract_features, extract_features_oracle, _TILE_TO_IDX
from algo.nn.model import build_model


torch.set_num_threads(1)

GLOBAL_NET = None
GLOBAL_CFG = None
GLOBAL_DEVICE = None
GLOBAL_N_ROLLOUTS = 2


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
        logits = net(x)[0]
    logits = logits.squeeze(0).detach().cpu().numpy()
    legal = np.zeros(34, dtype=np.float32)
    for t in hand14:
        legal[int(_TILE_TO_IDX[t])] = 1.0
    masked = logits + (legal - 1.0) * 1e9
    return masked


def _greedy_discard(hand14, ctx, name, net, extract, device):
    scores = _conv_bc_scores(hand14, ctx, name, net, extract, device)
    best = None
    best_score = -float('inf')
    for t in hand14:
        idx = int(_TILE_TO_IDX[t])
        if scores[idx] > best_score:
            best_score = scores[idx]
            best = t
    return best


def _simulate_from(hands, wall, ctx, locked, turn, current_name, net, cfg, extract, device,
                   max_steps=200):
    """从当前状态（turn 玩家已摸牌但尚未弃牌）模拟到终局，返回 current_name 的 outcome。"""
    names = sorted(hands.keys(), key=lambda n: int(n.split('@')[-1]))
    hands = {n: list(hands[n]) for n in names}
    wall = list(wall)
    ctx = ctx.copy()
    locked = set(locked)
    steps = 0
    while wall and steps < max_steps:
        pname = names[turn]
        drawn = wall.pop(0)
        hands[pname].append(drawn)

        if eval_v2.is_win(hands[pname]):
            return 1.0 if pname == current_name else -1.0

        if pname in locked:
            discarded = drawn
        else:
            discarded = _greedy_discard(hands[pname], ctx, pname, net, extract, device)
            # 报听锁手启发式
            if (pname not in locked and len(hands[pname]) == 13 and
                    eval_v2.shanten(hands[pname]) == 0):
                rem = ctx.remaining_wall(hands[pname])
                waits = eval_v2.winning_tiles(hands[pname], rem)
                if sum(rem.get(t, 0) for t in waits) >= 3:
                    locked.add(pname)
                    ctx.declare_tenpai(pname)
        hands[pname].remove(discarded)

        for j, other in enumerate(names):
            if j == turn:
                continue
            if eval_v2.is_win(hands[other] + [discarded]):
                return -1.0 if pname == current_name else (1.0 if other == current_name else -1.0)

        ctx.see_tile(discarded, pname)
        turn = (turn + 1) % 4
        steps += 1

    return 0.0


def _evaluate_candidate(hand14, all_hands, wall, ctx, locked, turn, current_name, cand,
                        net, cfg, extract, device, n_rollouts):
    """评估弃掉 cand 后的平均 rollout outcome。"""
    player_names = sorted(all_hands.keys(), key=lambda n: int(n.split('@')[-1]))
    opponents = [p for p in player_names if p != current_name]

    # 立即点炮：最差 outcome
    for opp in opponents:
        if eval_v2.is_win(list(all_hands[opp]) + [cand]):
            return -1.0

    # 构造弃牌后状态
    hands = {n: list(all_hands[n]) for n in player_names}
    hands[current_name].remove(cand)
    # 报听锁手启发式（若当前玩家在锁手状态则已在调用方处理）
    ctx2 = ctx.copy()
    ctx2.see_tile(cand, current_name)
    locked2 = set(locked)
    turn2 = (turn + 1) % 4

    total = 0.0
    for _ in range(n_rollouts):
        wall_copy = list(wall)
        random.shuffle(wall_copy)
        total += _simulate_from(hands, wall_copy, ctx2, locked2, turn2, current_name,
                                net, cfg, extract, device)
    return total / n_rollouts


def _oracle_rollout_discard(hand14, all_hands, wall, ctx, locked, turn, current_name,
                            net, cfg, extract, device, n_rollouts):
    """用 perfect-info rollout oracle 选弃牌。"""
    unique = sorted(set(hand14))
    best = None
    best_score = -float('inf')
    for cand in unique:
        score = _evaluate_candidate(hand14, all_hands, wall, ctx, locked, turn,
                                    current_name, cand, net, cfg, extract, device, n_rollouts)
        if score > best_score:
            best_score = score
            best = cand
    return best, best_score


def _play_one_game(net, cfg, device, seed, n_rollouts):
    random.seed(seed)
    np.random.seed(seed % (2 ** 31 - 1))

    pool = tile_pool.Pool()
    names = [f'P@{i}' for i in range(4)]
    hands = {n: pool.next_n(13) for n in names}
    ctx = context_v3.ContextV3()
    locked = set()
    turn = 0
    wall = list(pool.tiles[pool.idx:])

    extract = extract_features
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
            xn = extract_features(ctx, hands[pname], pname)
            xo = extract_features_oracle(ctx, hands[pname], pname, hands, wall)
            discarded, score = _oracle_rollout_discard(
                hands[pname], hands, wall, ctx, locked, turn, pname,
                net, cfg, extract, device, n_rollouts)
            samples.append((xn, xo, int(_TILE_TO_IDX[discarded]), pname))
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
    i, s, n_rollouts = args
    return _play_one_game(GLOBAL_NET, GLOBAL_CFG, GLOBAL_DEVICE, s, n_rollouts)[0]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'output/nn_teacher_rollout_oracle.npz'
    total_games = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    model_path = sys.argv[4] if len(sys.argv) > 4 else 'output/nn_conv_bc.pt'
    n_rollouts = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    seed_base = int(sys.argv[6]) if len(sys.argv) > 6 else 0

    device = 'cpu'
    net, cfg = _load_conv_bc(model_path)
    net.to(device)
    print(f'Rollout oracle data: {total_games} games, {workers} workers, n_rollouts={n_rollouts}')

    t0 = time.time()
    all_samples = []
    if workers <= 1:
        for i in range(total_games):
            all_samples.extend(_play_one_game(net, cfg, device, seed_base + i, n_rollouts)[0])
    else:
        global GLOBAL_NET, GLOBAL_CFG, GLOBAL_DEVICE
        GLOBAL_NET = net
        GLOBAL_CFG = cfg
        GLOBAL_DEVICE = device
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_thread_worker,
                                        [(i, seed_base + i, n_rollouts) for i in range(total_games)]))
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
