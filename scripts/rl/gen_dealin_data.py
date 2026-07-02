# -*- coding: utf-8 -*-
"""生成 per-tile 即时点炮标签数据，用于训练 deal-in auxiliary head。

对每个弃牌决策状态：
- 用 base conv-BC greedy 选动作（作为 policy 监督标签 y）；
- 用完美信息（知道对手手牌）计算每张合法弃牌是否立即点炮，生成 34 维 deal-in 标签：
  - 1.0：该牌在手牌中且会立即点炮；
  - 0.0：该牌在手牌中但不会立即点炮；
  - -1.0：该牌不在手牌中（不参与 loss）。

输出 .npz：X(n,175), dealin(n,34), y(n,), v(n,)。

用法：
    PYTHONPATH=. python3 scripts/rl/gen_dealin_data.py \
        output/nn_dealin_labels_2000.npz 2000 32 output/nn_conv_bc.pt 10000000
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
from algo.nn.features import extract_features, _TILE_TO_IDX
from algo.nn.model import build_model


torch.set_num_threads(1)

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


def _dealin_label(hand14, all_hands, current_name):
    """计算 34 维即时点炮标签（1=点炮，0=安全，-1=不在手牌）。"""
    player_names = sorted(all_hands.keys(), key=lambda n: int(n.split('@')[-1]))
    opponents = [p for p in player_names if p != current_name]
    label = np.full(34, -1.0, dtype=np.float32)
    for t in set(hand14):
        idx = int(_TILE_TO_IDX[t])
        hand13 = list(hand14)
        hand13.remove(t)
        deals_in = False
        for opp in opponents:
            if eval_v2.is_win(list(all_hands[opp]) + [t]):
                deals_in = True
                break
        label[idx] = 1.0 if deals_in else 0.0
    return label


def _play_one_game(net, cfg, device, seed):
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
            for x, d, y, pname in samples:
                full_samples.append((x, d, y, outcome.get(pname, 0.0)))
            return full_samples, outcome

        if pname in locked:
            discarded = drawn
            hands[pname].remove(discarded)
        else:
            x = extract_features(ctx, hands[pname], pname)
            d = _dealin_label(hands[pname], hands, pname)
            discarded = _greedy_discard(hands[pname], ctx, pname, net, extract, device)
            y = int(_TILE_TO_IDX[discarded])
            samples.append((x, d, y, pname))
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
                for x, d, y, pname in samples:
                    full_samples.append((x, d, y, outcome.get(pname, 0.0)))
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
    for x, d, y, pname in samples:
        full_samples.append((x, d, y, outcome.get(pname, 0.0)))
    return full_samples, outcome


def _thread_worker(args):
    i, s = args
    return _play_one_game(GLOBAL_NET, GLOBAL_CFG, GLOBAL_DEVICE, s)[0]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'output/nn_dealin_labels.npz'
    total_games = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    model_path = sys.argv[4] if len(sys.argv) > 4 else 'output/nn_conv_bc.pt'
    seed_base = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    device = 'cpu'
    net, cfg = _load_conv_bc(model_path)
    net.to(device)
    print(f'Deal-in label data: {total_games} games, {workers} workers')

    t0 = time.time()
    all_samples = []
    if workers <= 1:
        for i in range(total_games):
            all_samples.extend(_play_one_game(net, cfg, device, seed_base + i)[0])
    else:
        global GLOBAL_NET, GLOBAL_CFG, GLOBAL_DEVICE
        GLOBAL_NET = net
        GLOBAL_CFG = cfg
        GLOBAL_DEVICE = device
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_thread_worker,
                                        [(i, seed_base + i) for i in range(total_games)]))
        for r in results:
            all_samples.extend(r)
    dt = time.time() - t0
    print(f'Generated {len(all_samples)} samples from {total_games} games in {dt:.1f}s')

    if not all_samples:
        print('No samples generated')
        return

    X = np.stack([s[0] for s in all_samples])
    D = np.stack([s[1] for s in all_samples])
    y = np.array([s[2] for s in all_samples], dtype=np.int64)
    v = np.array([s[3] for s in all_samples], dtype=np.float32)
    np.savez(out_path, X=X, dealin=D, y=y, v=v)
    print(f'Saved {out_path}: X={X.shape}, dealin={D.shape}, y={y.shape}, v={v.shape}')


if __name__ == '__main__':
    main()
