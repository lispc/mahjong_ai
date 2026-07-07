# -*- coding: utf-8 -*-
"""Standard Arena：固定对手、固定 seed 集、duplicate 赛制的基准评测。

用法：
    PYTHONPATH=. python3 scripts/rl/standard_arena.py \
        --a v3deep:1-nn:output/nn_full_action_best.pt \
        --b hybrid:Best:output/nn_full_action_best.pt \
        --n-seeds 1000 --workers 24 --mirror

默认对手（可覆盖）：
    baseline, beliefexp, hybrid:Best:output/nn_full_action_best.pt

输出：
    output/arena_<a_safe>_vs_<b_safe>_<n-seeds>.pkl / .log
"""

import argparse
import os
import sys
import time
import math
import pickle
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
# 多进程 + PyTorch CUDA 安全：用 spawn 避免 fork 后 CUDA 上下文死锁
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
torch.set_num_threads(1)

from driver.tournament import run_duplicate_tournament
from checker.report import compute_metrics
from scripts.rl import benchmark_pool


_OPPONENTS = (
    'baseline',
    'beliefexp',
    'hybrid:Best:output/nn_full_action_best.pt',
)


def _parse_token(tok):
    return benchmark_pool._make_factory(tok)


def _base_name(name):
    return name.split('@')[0]


def _safe_label(tok):
    return tok.replace('/', '_').replace(':', '_').replace('.', '_')


def _paired_ci(a_wins, b_wins, n_pairs, z=1.96):
    if n_pairs == 0:
        return 0.0, 0.0, 0.0
    diff = (a_wins - b_wins) / n_pairs
    var = (a_wins + b_wins) / n_pairs - diff ** 2
    var = max(var, 0.0)
    se = math.sqrt(var / n_pairs)
    lo = max(-1.0, diff - z * se)
    hi = min(1.0, diff + z * se)
    return diff, lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', required=True, help='candidate A token')
    parser.add_argument('--b', required=True, help='candidate B token')
    parser.add_argument('--opponents', default=','.join(_OPPONENTS),
                        help='comma-separated 3 opponent tokens')
    parser.add_argument('--n-seeds', type=int, default=1000)
    parser.add_argument('--mirror', action='store_true',
                        help='mirror all 4 positions (8 games/seed)')
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--out-dir', default='output')
    args = parser.parse_args()

    a_factory, a_name = _parse_token(args.a)
    b_factory, b_name = _parse_token(args.b)
    opp_tokens = [t.strip() for t in args.opponents.split(',')]
    assert len(opp_tokens) == 3, 'need exactly 3 opponents'
    opp_factories = [_parse_token(t)[0] for t in opp_tokens]
    opp_names = [_parse_token(t)[1] for t in opp_tokens]

    games_per_seed = 8 if args.mirror else 2
    total_games = args.n_seeds * games_per_seed
    label = f'{_safe_label(args.a)}_vs_{_safe_label(args.b)}_{args.n_seeds}'
    if args.mirror:
        label += '_mirror'
    out_pkl = os.path.join(args.out_dir, f'arena_{label}.pkl')
    out_log = os.path.join(args.out_dir, f'arena_{label}.log')

    print(f'Standard Arena: {a_name} vs {b_name}')
    print(f'Opponents: {opp_names}')
    print(f'Seeds: {args.n_seeds}, mirror: {args.mirror}, total games: {total_games}, workers: {args.workers}')

    t0 = time.time()
    results = run_duplicate_tournament(
        a_factory, b_factory, opp_factories,
        n_seeds=args.n_seeds, mirror_positions=args.mirror,
        verbose=False, n_workers=args.workers, seed_offset=args.seed_offset)
    dt = time.time() - t0

    metrics = compute_metrics(results, [a_name, b_name] + opp_names)

    positions = list(range(4)) if args.mirror else [0]
    n_pairs = args.n_seeds * len(positions)
    a_wins = b_wins = pair_draws = 0
    for i in range(0, len(results), 2):
        pos = positions[(i // 2) % len(positions)]
        candidate_a = f'{a_name}@{pos}_a'
        candidate_b = f'{b_name}@{pos}_b'
        winner_a = results[i].get('winner')
        winner_b = results[i + 1].get('winner')
        a_won = winner_a == candidate_a
        b_won = winner_b == candidate_b
        if a_won and not b_won:
            a_wins += 1
        elif b_won and not a_won:
            b_wins += 1
        else:
            pair_draws += 1

    diff, lo, hi = _paired_ci(a_wins, b_wins, n_pairs)

    lines = []
    lines.append(f'Duplicate benchmark: {a_name} vs {b_name}')
    lines.append(f'Opponents: {opp_names}')
    lines.append(f'Seeds: {args.n_seeds}, mirror: {args.mirror}, total games: {total_games}, workers: {args.workers}')
    lines.append('')
    lines.append('Per-agent metrics:')
    for name in [a_name, b_name] + opp_names:
        m = metrics[name]
        lines.append(f'  {name:20s}: win {m["win_rate"]:.3%}, self {m["self_rate"]:.3%}, '
                     f'ron {m["ron_rate"]:.3%}, deal-in {m["deal_in_rate"]:.3%}, draw {m["draw_rate"]:.3%}')
    lines.append('')
    lines.append(f'Total {dt:.1f}s')
    lines.append(f'Paired difference ({a_name} - {b_name}):')
    lines.append(f'  A wins {a_wins}/{n_pairs} ({a_wins/n_pairs:.3%})')
    lines.append(f'  B wins {b_wins}/{n_pairs} ({b_wins/n_pairs:.3%})')
    lines.append(f'  Ties   {pair_draws}/{n_pairs} ({pair_draws/n_pairs:.3%})')
    lines.append(f'  A-B = {diff:+.3%}, 95% CI [{lo:+.3%}, {hi:+.3%}]')
    if lo > 0:
        lines.append(f'  => {a_name} significantly stronger')
    elif hi < 0:
        lines.append(f'  => {b_name} significantly stronger')
    else:
        lines.append('  => difference not significant at 95%')

    out_text = '\n'.join(lines)
    with open(out_log, 'w') as f:
        f.write(out_text + '\n')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'args': vars(args),
            'results': results,
            'a_name': a_name,
            'b_name': b_name,
            'opp_names': opp_names,
            'metrics': metrics,
            'paired': {
                'n_pairs': n_pairs,
                'a_wins': a_wins,
                'b_wins': b_wins,
                'ties': pair_draws,
                'diff': diff,
                'ci_lo': lo,
                'ci_hi': hi,
            },
        }, f)
    print(out_text)
    print(f'\nSaved: {out_pkl}')


if __name__ == '__main__':
    main()
