# -*- coding: utf-8 -*-
"""生成 Perfect-Info Oracle Search 数据。

Oracle 决策：在当前决策点，枚举所有合法弃牌，对每个弃牌用已知的对手手牌和牌山
做确定性 rollout（所有玩家按 conv-BC policy 打牌），到达终局或截断深度后用 conv-BC value
评估。选择平均回报最高的弃牌。

由于用到了完美信息，这个 oracle 显著强于 BeliefExp/Baseline，适合作为 oracle guiding 的教师。

输出 .npz 包含 Xn, Xo, y, v（与 gen_oracle_data.py 兼容）。

用法：
    PYTHONPATH=. python3 scripts/rl/gen_oracle_search_data.py \
        output/nn_teacher_oracle_search.npz 200 32 output/nn_conv_bc.pt 0
"""

import sys
import os
import json
import random
import numpy as np
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent as base_agent_mod
import context as ctx_module
import algo.context.v3 as context_v3
import algo.eval.v2 as eval_v2
import algo.eval.v3 as eval_v3
import tile_pool
import tile
from algo.nn.features import extract_features, extract_features_oracle, _TILE_TO_IDX, _IDX_TO_TILE
from algo.nn.model import build_model


NUM_ACTIONS = 34


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


def _conv_bc_select(hand14, ctx, name, net, extract, device):
    feats = extract(ctx, hand14, name)
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = net(x)
    logits = logits.squeeze(0).detach().cpu().numpy()
    legal = np.zeros(34, dtype=np.float32)
    for t in hand14:
        legal[int(_TILE_TO_IDX[t])] = 1.0
    masked = logits + (legal - 1.0) * 1e9
    a = int(np.argmax(masked))
    return int(_IDX_TO_TILE[a])


def _conv_bc_value(hand14, ctx, name, net, extract, device):
    feats = extract(ctx, hand14, name)
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        _, value = net(x)
    return float(value.detach().cpu().reshape(-1)[0])


def _simulate_deterministic(candidate, current_hand, all_hands, wall,
                            ctx, current_name, locked_names,
                            net, extract, device):
    """对给定 candidate 做完美信息确定性 rollout，直到终局。"""
    player_names = sorted(all_hands.keys(), key=lambda n: int(n.split('@')[-1]))
    current_idx = player_names.index(current_name)

    hands = {p: list(h) for p, h in all_hands.items()}
    cur_hand = list(current_hand)
    cur_hand.remove(candidate)
    hands[current_name] = cur_hand

    sim_wall = list(wall)
    turn = (current_idx + 1) % 4
    locked = set(locked_names)
    sim_ctx = ctx.copy()

    while sim_wall:
        pname = player_names[turn]
        drawn = sim_wall.pop(0)
        hands[pname].append(drawn)

        if eval_v2.is_win(hands[pname]):
            return 1.0 if turn == current_idx else -1.0

        if pname in locked:
            discarded = drawn
            hands[pname].remove(discarded)
        else:
            discarded = _conv_bc_select(hands[pname], sim_ctx, pname, net, extract, device)
            hands[pname].remove(discarded)

        for j, other in enumerate(player_names):
            if j == turn:
                continue
            if eval_v2.is_win(hands[other] + [discarded]):
                if other == current_name:
                    return 1.0
                if pname == current_name:
                    return -1.0
                return -1.0

        sim_ctx.see_tile(discarded, pname)

        if (pname not in locked and len(hands[pname]) == 13 and
                eval_v2.shanten(hands[pname]) == 0):
            rem = sim_ctx.remaining_wall(hands[pname])
            waits = eval_v2.winning_tiles(hands[pname], rem)
            if sum(rem.get(t, 0) for t in waits) >= 3:
                locked.add(pname)
                sim_ctx.declare_tenpai(pname)

        turn = (turn + 1) % 4

    return 0.0


def _oracle_search_discard(hand14, all_hands, wall, ctx, current_name, locked_names,
                           net, extract, device):
    """用完美信息搜索选弃牌。"""
    unique = sorted(set(hand14))
    best_disc = unique[0]
    best_value = -float('inf')
    for cand in unique:
        val = _simulate_deterministic(cand, hand14, all_hands, wall, ctx, current_name,
                                      locked_names, net, extract, device)
        if val > best_value:
            best_value = val
            best_disc = cand
    return best_disc


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

    # 预计算 rollout 用的完整 wall
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
            break

        if pname in locked:
            discarded = drawn
            hands[pname].remove(discarded)
        else:
            # 记录决策前状态
            xn = extract_features(ctx, hands[pname], pname)
            xo = extract_features_oracle(ctx, hands[pname], pname, hands, wall)

            discarded = _oracle_search_discard(hands[pname], hands, wall, ctx, pname,
                                               locked, net, extract, device)
            samples.append((xn, xo, int(_TILE_TO_IDX[discarded]), pname))
            hands[pname].remove(discarded)

        # 点炮检查
        for j, other in enumerate(names):
            if j == turn:
                continue
            if eval_v2.is_win(hands[other] + [discarded]):
                for n in names:
                    if n == other:
                        outcome[n] = 1.0
                    elif n == pname:
                        outcome[n] = -1.0
                    else:
                        outcome[n] = -1.0
                return samples, outcome

        ctx.see_tile(discarded, pname)

        if (pname not in locked and len(hands[pname]) == 13 and
                eval_v2.shanten(hands[pname]) == 0):
            rem = ctx.remaining_wall(hands[pname])
            waits = eval_v2.winning_tiles(hands[pname], rem)
            if sum(rem.get(t, 0) for t in waits) >= 3:
                locked.add(pname)
                ctx.declare_tenpai(pname)

        turn = (turn + 1) % 4

    # 回填 outcome
    full_samples = []
    for xn, xo, y, pname in samples:
        full_samples.append((xn, xo, y, outcome.get(pname, 0.0)))
    return full_samples, outcome


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'output/nn_teacher_oracle_search.npz'
    total_games = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    model_path = sys.argv[4] if len(sys.argv) > 4 else 'output/nn_conv_bc.pt'
    seed_base = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    device = 'cpu'
    net, cfg = _load_conv_bc(model_path)
    net.to(device)
    print(f'Oracle search data: {total_games} games, {workers} workers')

    t0 = time.time()
    all_samples = []
    if workers <= 1:
        for i in range(total_games):
            samples, _ = _play_one_game(net, cfg, device, seed_base + i)
            all_samples.extend(samples)
    else:
        from concurrent.futures import ProcessPoolExecutor
        # 每个 worker 加载自己的 net
        def _worker(args):
            i, s = args
            local_net, local_cfg = _load_conv_bc(model_path)
            local_net.to(device)
            return _play_one_game(local_net, local_cfg, device, s)[0]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_worker, [(i, seed_base + i) for i in range(total_games)]))
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
