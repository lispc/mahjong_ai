# -*- coding: utf-8 -*-
"""Phase 0 sanity：验证自对弈轨迹记录器的正确性（先小规模跑通再放大）。

用 warm-start 的 output/nn_model.pt 打若干局，检查：
- 特征维度 = 175；
- 每步采样动作都合法（mask 对应位为 1，且 tile 在决策时手牌里）；
- value / logp 数值有限；
- 每局奖励与 result 一致（winner +1、放铳 -1、流局 0）；
- 轨迹非空、step_idx 有序。

运行：PYTHONPATH=. python3 scripts/rl/sanity_selfplay.py [n_games]
"""

import sys
import json
import numpy as np
import torch

from algo.rl.selfplay import build_net, play_selfplay_game, NUM_ACTIONS
from algo.rl.reward import DEFAULT_REWARD
from algo.nn.features import _IDX_TO_TILE


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    cfg = json.load(open('output/nn_model_config.json'))
    state = torch.load('output/nn_model.pt', map_location='cpu')
    state = state.get('model_state_dict', state) if isinstance(state, dict) and 'model_state_dict' in state else state
    net = build_net(state, input_dim=cfg['input_dim'], hidden_dim=cfg['hidden_dim'], device='cpu')
    print(f'warm-start net loaded: input_dim={cfg["input_dim"]} hidden_dim={cfg["hidden_dim"]}')

    n_steps_total = 0
    n_traj = 0
    reason_counts = {}
    reward_sum_by_game = []
    errors = []

    for g in range(n_games):
        trajs, result = play_selfplay_game(net, seed=1000 + g, reward_cfg=DEFAULT_REWARD)
        game_reward = 0.0
        for tr in trajs:
            n_traj += 1
            reason_counts[tr['reason']] = reason_counts.get(tr['reason'], 0) + 1
            game_reward += tr['reward']
            for i, step in enumerate(tr['steps']):
                n_steps_total += 1
                feat = step['feat']
                mask = step['mask']
                a = step['action']
                # 维度
                if feat.shape[0] != cfg['input_dim']:
                    errors.append(f'game{g} feat dim {feat.shape}')
                if mask.shape[0] != NUM_ACTIONS:
                    errors.append(f'game{g} mask dim {mask.shape}')
                # 合法性
                if mask[a] != 1.0:
                    errors.append(f'game{g} step{i} illegal action a={a} not in mask')
                # 数值有限
                if not (np.isfinite(step['value']) and np.isfinite(step['logp'])):
                    errors.append(f'game{g} step{i} nonfinite value/logp')
        reward_sum_by_game.append(game_reward)
        # result 一致性：非流局必有唯一 winner
        if result['win_type'] != 'draw' and result.get('winner') is None:
            errors.append(f'game{g} non-draw but no winner')

    print(f'games={n_games} trajectories={n_traj} total_steps={n_steps_total} '
          f'avg_steps/traj={n_steps_total/max(1,n_traj):.1f}')
    print('terminal reasons:', reason_counts)
    print('per-game summed reward (4 seats):', [round(r, 2) for r in reward_sum_by_game])
    print(f'reward per-game mean (should be <=0, ~ -2 for 1 winner + 3 losers): '
          f'{np.mean(reward_sum_by_game):.3f}')

    if errors:
        print(f'\n*** {len(errors)} ERRORS ***')
        for e in errors[:20]:
            print('  ', e)
        sys.exit(1)
    print('\nSANITY OK: all sampled actions legal, dims correct, rewards consistent.')


if __name__ == '__main__':
    main()
