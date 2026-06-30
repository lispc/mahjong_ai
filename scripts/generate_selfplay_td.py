#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成自对弈轨迹（含终局 outcome），用于 TD(λ) 训练。

与 generate_selfplay_raw.py 区别：
- 每局输出一个 trajectory dict（保留步序），不再 flatten
- 不调用 MC rollout，速度快 10×
- target_seat 玩家的 outcome/terminal_reason 回填到每一步
- 支持断点续跑：每 N 局 save 一次 .partial.pkl
- 多 GPU 并行：通过 CUDA_VISIBLE_DEVICES 拆 4 进程
"""
import sys
import os
import time
import random
import pickle
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.agents.data_collectors import DataCollectorV3NN
from driver import engine


def _classify_terminal(target_name, result):
    """返回 (outcome, terminal_reason)。outcome 从 target_name 视角。"""
    win_type = result.get('win_type')
    winner = result.get('winner')
    if win_type == 'draw' or winner is None:
        return 0.0, 'draw'
    if winner == target_name:
        # 自己赢
        if win_type == 'self':
            return 1.0, 'tsumo_win'
        if win_type == 'ron':
            return 1.0, 'ron_win'
        return 1.0, 'tsumo_win'  # fallback
    # 自己输
    if win_type == 'self':
        return -1.0, 'lose_tsumo'
    if win_type == 'ron':
        dealer = result.get('dealer')
        if dealer == target_name:
            return -1.0, 'ron_mine'
        return -1.0, 'lose_ron_others'
    return -1.0, 'lose_unknown'


def _play_and_collect_td(args):
    """跑一局，返回 target_seat 玩家的完整 trajectory dict。"""
    seed, target_seat, game_id = args
    random.seed(seed)
    agents = [DataCollectorV3NN('V3NN', verbose=False, game_id=game_id,
                                expectimax_depth=1, max_candidates=5)
              for _ in range(4)]
    random.shuffle(agents)
    for i, a in enumerate(agents):
        a.name = '{}@{}'.format(a.name, i)

    result = engine.play_game(agents, verbose=False, record_time=False)

    target_name = f'V3NN@{target_seat}'
    target_agent = None
    for a in agents:
        if a.name == target_name:
            target_agent = a
            break

    if target_agent is None or not target_agent.buffer:
        return None

    outcome, terminal_reason = _classify_terminal(target_name, result)
    target_agent.set_outcome(outcome, terminal_reason)

    return {
        'game_id': game_id,
        'outcome': outcome,
        'terminal_reason': terminal_reason,
        'n_steps': len(target_agent.buffer),
        'samples': target_agent.buffer,  # list of dict
    }


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    out_path = sys.argv[3] if len(sys.argv) > 3 else 'output/selfplay_td.pkl'
    seed_offset = int(sys.argv[4]) if len(sys.argv) > 4 else 950000
    target_seat = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    save_every = int(sys.argv[6]) if len(sys.argv) > 6 else 50

    print(f'Generating {n_games} TD self-play games, workers={n_workers}, '
          f'target_seat={target_seat}, seed_offset={seed_offset}', flush=True)
    print(f'Output: {out_path}', flush=True)

    # 断点续跑：检查 .partial.pkl
    partial_path = out_path + '.partial.pkl'
    trajectories = []
    start_game = 0
    if os.path.exists(partial_path):
        try:
            with open(partial_path, 'rb') as f:
                trajectories = pickle.load(f)
            start_game = len(trajectories)
            print(f'Resuming from checkpoint: {start_game}/{n_games} games already done', flush=True)
        except Exception as e:
            print(f'Checkpoint corrupt, starting fresh: {e}', flush=True)
            trajectories = []
            start_game = 0

    if start_game >= n_games:
        print(f'Already have {start_game} games, nothing to do', flush=True)
    else:
        start = time.time()
        tasks = [(seed_offset + i, target_seat, i)
                 for i in range(start_game, n_games)]
        completed = start_game

        with Pool(n_workers) as pool:
            for traj in pool.imap_unordered(_play_and_collect_td, tasks):
                if traj is not None:
                    trajectories.append(traj)
                completed += 1
                if completed % 10 == 0 or completed == n_games:
                    elapsed = time.time() - start
                    rate = (completed - start_game) / max(elapsed, 0.1)
                    eta = (n_games - completed) / max(rate, 0.01)
                    print(f'  ... {completed}/{n_games} games, '
                          f'{len(trajectories)} valid, '
                          f'{rate:.1f} games/s, eta {eta:.0f}s', flush=True)
                if completed % save_every == 0:
                    with open(partial_path, 'wb') as f:
                        pickle.dump(trajectories, f)

        elapsed = time.time() - start
        print(f'Done in {elapsed:.1f}s: {len(trajectories)} valid trajectories', flush=True)

    # 最终保存
    with open(out_path, 'wb') as f:
        pickle.dump(trajectories, f)
    if os.path.exists(partial_path):
        os.remove(partial_path)

    # 统计
    n = len(trajectories)
    if n > 0:
        wins = sum(1 for t in trajectories if t['outcome'] > 0.5)
        losses = sum(1 for t in trajectories if t['outcome'] < -0.5)
        draws = n - wins - losses
        total_steps = sum(t['n_steps'] for t in trajectories)
        print(f'Stats: {n} games, {wins} wins ({100*wins/n:.1f}%), '
              f'{losses} losses ({100*losses/n:.1f}%), {draws} draws ({100*draws/n:.1f}%)')
        print(f'Avg steps/game: {total_steps/n:.1f}, total samples: {total_steps}')
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
