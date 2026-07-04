# -*- coding: utf-8 -*-
"""生成对手建模训练数据。

在自对弈的每个决策点记录：
- 当前玩家的公开特征 X（175-dim，含自己手牌）
- 三个对手的隐藏手牌 multi-hot（3 x 34）
- 三个对手是否听牌（3）

用法：
    PYTHONPATH=. python3 scripts/rl/gen_opponent_data.py \
        output/opponent_model_data_32000.npz 32000 32 \
        output/nn_full_action_best.pt 50000000
"""
import os
import sys
import time
import random
import argparse
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from driver.engine import play_game
from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent
from algo.agents.ppo_agent import HeuristicResponsePPOAgent
from algo.nn.features import extract_features, tile_to_index
from algo.eval.v2 import shanten


class RecordingHybridAgent(HybridNNBeliefAgent):
    """在玩游戏的同时记录弃牌/响应样本，并允许回调读取隐藏状态。"""

    def __init__(self, name, nn_model_path, device='cpu'):
        super().__init__(
            name,
            nn_model_path=nn_model_path,
            belief_kind='beliefexp',
            tenpai_threshold=999,
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
        self.discard_samples.append({'feat': feat, 'action': tile_to_index(tile_val)})
        return tile_val

    def respond_hu(self, tile_val, context=None):
        accept = self.nn_agent.respond_hu(tile_val, context)
        self.response_samples.append({'feat': self._feat_response(tile_val), 'action': 3 if accept else 0})
        return accept

    def respond_peng(self, tile_val, context=None):
        accept = self.nn_agent.respond_peng(tile_val, context)
        self.response_samples.append({'feat': self._feat_response(tile_val), 'action': 1 if accept else 0})
        return accept

    def respond_gang(self, tile_val, context=None):
        accept = self.nn_agent.respond_gang(tile_val, context)
        self.response_samples.append({'feat': self._feat_response(tile_val), 'action': 2 if accept else 0})
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


def _hand_to_multihot(hand):
    arr = np.zeros(34, dtype=np.float32)
    for t in hand:
        arr[tile_to_index(t)] = 1.0
    return arr


def _play_one_game(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    global _WORKER_MODEL_PATH, _WORKER_DEVICE
    agents = [RecordingHybridAgent(f'P{s}', _WORKER_MODEL_PATH, device=_WORKER_DEVICE) for s in range(4)]

    snapshots = []

    def state_callback(agents, turn, event_type, info):
        if event_type != 'decision':
            return
        current = agents[turn]
        # 使用当前玩家视角的公开特征作为输入
        ctx = current.nn_agent.context
        feat = extract_features(ctx, current.full_hand(), current.name)
        # 三个对手：下家、对家、上家
        opp_tenpai = np.zeros(3, dtype=np.float32)
        opp_hand = np.zeros((3, 34), dtype=np.float32)
        for offset in range(1, 4):
            idx = (turn + offset) % 4
            opp = agents[idx]
            hand = opp.full_hand()
            opp_hand[offset - 1] = _hand_to_multihot(hand)
            try:
                opp_tenpai[offset - 1] = 1.0 if shanten(hand) == 0 else 0.0
            except Exception:
                opp_tenpai[offset - 1] = 0.0
        snapshots.append((feat, opp_tenpai, opp_hand, turn))

    result = play_game(agents, verbose=False, record_time=False, state_callback=state_callback)
    return snapshots, result


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
    device = 'cpu'

    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    t0 = time.time()
    all_snapshots = []
    results = []
    with mp.Pool(processes=args.workers, initializer=_init_worker,
                 initargs=(args.nn_path, device)) as pool:
        for i, (snaps, res) in enumerate(pool.imap_unordered(_play_one_game, seeds)):
            all_snapshots.extend(snaps)
            results.append(res)
            if (i + 1) % max(1, args.total_games // 20) == 0:
                print(f'  {i+1}/{args.total_games} games done, snapshots={len(all_snapshots)}')

    dt = time.time() - t0
    print(f'Generated {len(results)} games in {dt:.1f}s')
    print(f'snapshots: {len(all_snapshots)}')

    if not all_snapshots:
        print('No snapshots')
        return

    X = np.stack([s[0] for s in all_snapshots])
    opp_tenpai = np.stack([s[1] for s in all_snapshots])
    opp_hand = np.stack([s[2] for s in all_snapshots])
    self_seat = np.array([s[3] for s in all_snapshots], dtype=np.int64)

    np.savez(args.out_path,
             X=X, opp_tenpai=opp_tenpai, opp_hand=opp_hand, self_seat=self_seat)
    print(f'Saved {args.out_path}')


if __name__ == '__main__':
    main()
