# -*- coding: utf-8 -*-
"""Duplicate（复式）赛制 benchmark：配对消除发牌运气。

用法：
    PYTHONPATH=. python3 scripts/rl/benchmark_duplicate.py \
        --a hybrid:A:output/nn_full_action_best.pt \
        --b baseline \
        --opponents baseline,beliefexp,hybrid:Base:output/nn_full_action_best.pt \
        --n-seeds 1000 --workers 32

--a / --b 支持 benchmark_pool.py 的所有 token（baseline / beliefexp / hybrid:... 等）。
--opponents 是逗号分隔的 3 个固定对手 token。
默认只镜像 position 0（2 局/seed），加 --mirror-positions 则 8 局/seed。
"""

import argparse
import os
import sys
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driver.tournament import run_duplicate_tournament
from checker.report import compute_metrics
from scripts.rl import benchmark_pool


def _parse_token(tok):
    return benchmark_pool._make_factory(tok)


def _base_name(name):
    return name.split('@')[0]


def _paired_ci(a_wins, b_wins, n_pairs, z=1.96):
    """Paired difference (A - B) win-rate 95% CI."""
    if n_pairs == 0:
        return 0.0, 0.0, 0.0
    diff = (a_wins - b_wins) / n_pairs
    # Variance of paired difference
    var = (a_wins + b_wins) / n_pairs - diff ** 2
    var = max(var, 0.0)
    se = math.sqrt(var / n_pairs)
    lo = max(-1.0, diff - z * se)
    hi = min(1.0, diff + z * se)
    return diff, lo, hi


def main():
    import torch
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser(description='Duplicate tournament benchmark')
    parser.add_argument('--a', required=True, help='candidate A token')
    parser.add_argument('--b', required=True, help='candidate B token')
    parser.add_argument('--opponents', required=True,
                        help='exactly 3 opponent tokens, comma separated')
    parser.add_argument('--n-seeds', type=int, default=400)
    parser.add_argument('--mirror-positions', action='store_true')
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--output', default=None,
                        help='optional path to write raw results pickle')
    args = parser.parse_args()

    a_factory, a_name = _parse_token(args.a)
    b_factory, b_name = _parse_token(args.b)
    opp_tokens = [t.strip() for t in args.opponents.split(',')]
    if len(opp_tokens) != 3:
        raise ValueError('--opponents must contain exactly 3 tokens')
    opp_factories = [_parse_token(t)[0] for t in opp_tokens]
    opp_names = [_parse_token(t)[1] for t in opp_tokens]

    games_per_seed = 8 if args.mirror_positions else 2
    total_games = args.n_seeds * games_per_seed
    print(f'Duplicate benchmark: {a_name} vs {b_name}')
    print(f'Opponents: {opp_names}')
    print(f'Seeds: {args.n_seeds}, positions mirrored: {args.mirror_positions}, '
          f'total games: {total_games}, workers: {args.workers}')

    t0 = time.time()
    results = run_duplicate_tournament(
        a_factory, b_factory, opp_factories,
        n_seeds=args.n_seeds, mirror_positions=args.mirror_positions,
        verbose=False, n_workers=args.workers, seed_offset=args.seed_offset)
    dt = time.time() - t0

    # Aggregate simple win rates
    metrics = compute_metrics(results, [a_name, b_name] + opp_names)
    # Candidate-specific win rates.  The candidate sits at a known seat within
    # each pair; use players_order[seat] rather than name-prefix matching,
    # which breaks when a candidate shares its base name with an opponent.
    positions = list(range(4)) if args.mirror_positions else [0]

    def _candidate_wins(name, kind):
        wins = total = 0
        for k in range(0, len(results) - 1, 2):
            pos = positions[(k // 2) % len(positions)]
            r = results[k] if kind == 'a' else results[k + 1]
            cand = r['players_order'][pos]
            if not cand.startswith(name):
                continue
            total += 1
            if r.get('winner') == cand:
                wins += 1
        return wins, total

    a_wins_total, a_games = _candidate_wins(a_name, 'a')
    b_wins_total, b_games = _candidate_wins(b_name, 'b')
    print(f'\nCandidate-specific win rates:')
    print(f'  {a_name:20s}: {a_wins_total}/{a_games} = {a_wins_total/a_games:.1%}')
    print(f'  {b_name:20s}: {b_wins_total}/{b_games} = {b_wins_total/b_games:.1%}')
    print(f'  Simple A-B diff: {(a_wins_total/a_games - b_wins_total/b_games):+.1%}')

    # Opponent aggregate (for reference, same as compute_metrics but clearer)
    print(f'\nTotal {dt:.1f}s')
    for name in [a_name, b_name] + opp_names:
        m = metrics[name]
        print(f'  {name:20s}: win {m["win_rate"]:.1%}, '
              f'self {m["self_rate"]:.1%}, ron {m["ron_rate"]:.1%}, '
              f'draw {m["draw_rate"]:.1%}')

    # Paired comparison: A vs B on the same (seed, position) pairs
    n_pairs = args.n_seeds * len(positions)
    a_wins = 0
    b_wins = 0
    pair_draws = 0
    score_diff_sum = 0.0
    score_diff_sq = 0.0

    def _seat_score(r, cand):
        # 推倒胡计分代理：自摸三家付 (+3)，点和一家付 (+1)，放炮 -1，其余 0
        if r.get('winner') == cand:
            return 3.0 if r.get('win_type') == 'self' else 1.0
        if r.get('win_type') == 'ron' and r.get('dealer') == cand:
            return -1.0
        return 0.0

    # Results are ordered: for each seed, for each position, A then B.
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
        sd = (_seat_score(results[i], candidate_a)
              - _seat_score(results[i + 1], candidate_b))
        score_diff_sum += sd
        score_diff_sq += sd * sd

    diff, lo, hi = _paired_ci(a_wins, b_wins, n_pairs)
    print(f'\nPaired difference ({a_name} - {b_name}):')
    print(f'  A wins {a_wins}/{n_pairs} ({a_wins/n_pairs:.1%})')
    print(f'  B wins {b_wins}/{n_pairs} ({b_wins/n_pairs:.1%})')
    print(f'  Ties   {pair_draws}/{n_pairs} ({pair_draws/n_pairs:.1%})')
    print(f'  A-B = {diff:+.1%}, 95% CI [{lo:+.1%}, {hi:+.1%}]')
    score_mean = score_diff_sum / n_pairs
    score_var = max(score_diff_sq / n_pairs - score_mean ** 2, 0.0)
    score_se = math.sqrt(score_var / n_pairs)
    score_lo = score_mean - 1.96 * score_se
    score_hi = score_mean + 1.96 * score_se
    print(f'  Score-proxy A-B (self=+3, ron=+1, deal-in=-1): {score_mean:+.3f}, '
          f'95% CI [{score_lo:+.3f}, {score_hi:+.3f}]')
    if lo > 0:
        print(f'  => {a_name} significantly stronger (CI excludes 0)')
    elif hi < 0:
        print(f'  => {b_name} significantly stronger (CI excludes 0)')
    else:
        print('  => difference not significant at 95%')

    if args.output:
        import pickle
        with open(args.output, 'wb') as f:
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
                    'score_diff': score_mean,
                    'score_ci_lo': score_lo,
                    'score_ci_hi': score_hi,
                },
            }, f)
        print(f'Raw results saved to {args.output}')


if __name__ == '__main__':
    main()
