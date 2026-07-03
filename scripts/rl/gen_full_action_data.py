# -*- coding: utf-8 -*-
"""生成完整动作空间（弃牌 + 碰/杠/胡响应 + 报听）的行为克隆数据。

教师：
- 弃牌：当前最强 conv-BC NN policy（Hybrid）；
- 碰/杠/胡：HeuristicResponsePPOAgent 中的强启发式；
- 报听：原 NN tenpai head / 启发式。

输出 .npz 包含：
- X_discard, y_discard, v_discard, tenpai_discard
- X_response, y_response, legal_response, v_response

用法：
    PYTHONPATH=. python3 scripts/rl/gen_full_action_data.py \
        output/nn_full_action_data_1000.npz 1000 32 \
        output/nn_conv_bc_beliefexp_trace_16000_big_t8.pt 40000000
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import multiprocessing as mp

from driver.engine import play_game
from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent
from algo.agents.ppo_agent import HeuristicResponsePPOAgent
from algo.nn.features import extract_features, tile_to_index
from algo.rl.reward import seat_reward


class RecordingHybridAgent(HybridNNBeliefAgent):
    """在玩游戏的同时记录弃牌/响应样本。"""

    def __init__(self, name, nn_model_path, device='cpu'):
        super().__init__(
            name,
            nn_model_path=nn_model_path,
            belief_kind='beliefexp',
            tenpai_threshold=999,          # 数据生成阶段强制走 NN，避免 BeliefExp 未适配副露
            device=device,
            temperature=None,
            verbose=False,
            nn_agent_class=HeuristicResponsePPOAgent,
        )
        self.discard_samples = []
        self.response_samples = []

    def _feat_discard(self):
        return extract_features(self.nn_agent.context, self.full_hand(), self.name)

    def _feat_response(self, tile_val):
        return extract_features(self.nn_agent.context, self.full_hand() + [tile_val], self.name)

    def next(self):
        feat = self._feat_discard()
        tile_val = super().next()
        self.discard_samples.append({
            'feat': feat,
            'action': tile_to_index(tile_val),
            'tenpai': 1.0 if self.name in getattr(self.nn_agent.context, 'tenpai_players', set()) else 0.0,
        })
        return tile_val

    def respond_hu(self, tile_val, context=None):
        accept = self.nn_agent.respond_hu(tile_val, context)
        legal = [1.0, 0.0, 0.0, 1.0 if super().respond_hu(tile_val, context) else 0.0]
        action = 3 if accept else 0
        self.response_samples.append({
            'feat': self._feat_response(tile_val),
            'action': action,
            'legal': legal,
        })
        return accept

    def respond_peng(self, tile_val, context=None):
        accept = self.nn_agent.respond_peng(tile_val, context)
        legal = [1.0, 1.0 if self._can_peng(tile_val) else 0.0, 0.0, 0.0]
        action = 1 if accept else 0
        self.response_samples.append({
            'feat': self._feat_response(tile_val),
            'action': action,
            'legal': legal,
        })
        return accept

    def respond_gang(self, tile_val, context=None):
        accept = self.nn_agent.respond_gang(tile_val, context)
        legal = [1.0, 0.0, 1.0 if self._can_gang(tile_val) else 0.0, 0.0]
        action = 2 if accept else 0
        self.response_samples.append({
            'feat': self._feat_response(tile_val),
            'action': action,
            'legal': legal,
        })
        return accept


_WORKER_MODEL_PATH = None
_WORKER_DEVICE = None


def _init_worker(nn_path, device):
    global _WORKER_MODEL_PATH, _WORKER_DEVICE
    _WORKER_MODEL_PATH = nn_path
    _WORKER_DEVICE = device
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _play_one_game(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    global _WORKER_MODEL_PATH, _WORKER_DEVICE
    agents = [RecordingHybridAgent(f'P{s}', _WORKER_MODEL_PATH, device=_WORKER_DEVICE) for s in range(4)]
    result = play_game(agents, verbose=False, record_time=False)

    discards = []
    responses = []
    for ag in agents:
        reward = seat_reward(result, ag.name, {
            'win': 1.0, 'deal_in': -1.0, 'other_loss': -1.0, 'draw': 0.0,
        })
        for s in ag.discard_samples:
            discards.append((s['feat'], s['action'], reward, s['tenpai']))
        for s in ag.response_samples:
            responses.append((s['feat'], s['action'], s['legal'], reward))

    return discards, responses, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_path')
    ap.add_argument('total_games', type=int)
    ap.add_argument('workers', type=int)
    ap.add_argument('nn_path')
    ap.add_argument('seed_base', type=int)
    args = ap.parse_args()

    mp.set_start_method('spawn', force=True)

    seeds = [args.seed_base + i for i in range(args.total_games)]
    device = 'cpu'  # 数据生成在 CPU 上足够快，且避免多进程抢 GPU

    t0 = time.time()
    all_discard = []
    all_response = []
    results = []

    print(f'Starting {args.total_games} games with {args.workers} workers on {device}')
    # 避免多进程 NN 前向时抢 CPU 线程
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    with mp.Pool(processes=args.workers, initializer=_init_worker,
                 initargs=(args.nn_path, device)) as pool:
        for i, (disc, resp, res) in enumerate(pool.imap_unordered(_play_one_game, seeds)):
            all_discard.extend(disc)
            all_response.extend(resp)
            results.append(res)
            if (i + 1) % max(1, args.total_games // 20) == 0:
                print(f'  {i+1}/{args.total_games} games done')

    dt = time.time() - t0
    print(f'Generated {len(results)} games in {dt:.1f}s')
    print(f'discard samples: {len(all_discard)}, response samples: {len(all_response)}')

    if not all_discard:
        print('No samples')
        return

    Xd = np.stack([s[0] for s in all_discard])
    yd = np.array([s[1] for s in all_discard], dtype=np.int64)
    vd = np.array([s[2] for s in all_discard], dtype=np.float32)
    td = np.array([s[3] for s in all_discard], dtype=np.float32)

    Xr = np.stack([s[0] for s in all_response])
    yr = np.array([s[1] for s in all_response], dtype=np.int64)
    lr = np.array([s[2] for s in all_response], dtype=np.float32)
    vr = np.array([s[3] for s in all_response], dtype=np.float32)

    np.savez(args.out_path,
             X_discard=Xd, y_discard=yd, v_discard=vd, tenpai_discard=td,
             X_response=Xr, y_response=yr, legal_response=lr, v_response=vr)
    print(f'Saved {args.out_path}')


if __name__ == '__main__':
    main()
