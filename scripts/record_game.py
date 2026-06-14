#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Record a single game with full event log for replay."""

import sys
import os
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.expectimax_eval2 import ExpectiMaxEval2Agent, ExpectiMaxEval2DefenseAgent
from algo.agents.mcts_eval2 import MCTSEval2Agent
from algo.agents.mcts import MCTSAgent
from algo.agents.shanten_ukeire import ShantenUkeireAgent, ShantenUkeireV3Agent
from driver.engine import play_game


def make_baseline():
    return agent.Agent('Baseline', verbose=False)


def make_eval2ctx():
    return ExpectiMaxEval2Agent('Eval2Ctx', verbose=False, max_candidates=6)


def make_eval2ctx_bd():
    return ExpectiMaxEval2DefenseAgent('Eval2Ctx+BD', verbose=False,
                                       defense_weight=3.0, safe_mode=True)


def make_mcts_eval2():
    return MCTSEval2Agent('MCTS-Eval2', samples=5, max_candidates=4, verbose=False)


def make_mcts():
    return MCTSAgent('MCTS', depth=1, samples=250, verbose=False)


def make_shanten_ukeire():
    return ShantenUkeireAgent('ShantenUkeire', verbose=False)


def make_shanten_ukeire_v3():
    return ShantenUkeireV3Agent('ShantenUkeireV3', verbose=False, defense_weight=2.0)


AGENTS = {
    'baseline': make_baseline,
    'eval2ctx': make_eval2ctx,
    'eval2ctx_bd': make_eval2ctx_bd,
    'mcts_eval2': make_mcts_eval2,
    'mcts': make_mcts,
    'shanten_ukeire': make_shanten_ukeire,
    'shanten_ukeire_v3': make_shanten_ukeire_v3,
}


def main():
    parser = argparse.ArgumentParser(description='Record one mahjong game as JSON log.')
    parser.add_argument('-o', '--output', default=None,
                        help='Output JSON path. Default: output/replay_<timestamp>.json')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed for reproducibility.')
    parser.add_argument('--agents', nargs=4, default=['eval2ctx', 'baseline', 'mcts', 'mcts_eval2'],
                        help='Four agent names from: baseline, eval2ctx, eval2ctx_bd, mcts_eval2, mcts')
    args = parser.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)

    factories = [AGENTS[name] for name in args.agents]
    agents = [f() for f in factories]

    result = play_game(agents, verbose=False, record_time=True, record_log=True)

    output_path = args.output
    if output_path is None:
        ts = int(time.time())
        output_path = 'output/replay_{}.json'.format(ts)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print('Saved replay to:', output_path)
    print('Winner:', result.get('winner'), 'Type:', result.get('win_type'))
    print('Events:', len(result.get('event_log', [])))


if __name__ == '__main__':
    main()
