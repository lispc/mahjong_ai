# -*- coding: utf-8 -*-
"""生成对手建模序列数据（方向 D1，docs/designs/silent-tenpai-seq-model.md）。

默认 BeliefExp 自对弈；--mix 启用混合agent池（BeliefExp/Baseline/Hybrid 随机排座），
用于匹配部署时的对手分布——Baseline 永不报听（默听标签的主要来源），
BeliefExp 积极报听，Hybrid 用 NN tenpai head。

每个决策 snapshot 记录：
- feats: 175 维公开特征（含自己手牌）；
- 每名对手（下家/对家/上家）：
  - discard 序列（tile idx，按时间顺序，pad 到 MAX_SEQ）；
  - 是否已报听 + 报听步（= 报听者第几次弃牌后报听，-1 未报听）；
  - 副露 tile multi-hot（34）；
  - agent 类型（mix 模式：0=Baseline 1=BeliefExp 2=Hybrid；非 mix 全 1）；
- 标签：opp_tenpai（3，含默听）、opp_wait（3x34 one-hot，仅听牌者非零）。

分 shard 保存（每 shard 1000 局），已存在的 shard 自动跳过（断点续跑）；
最后 merge 成单个 npz。

用法：
    PYTHONPATH=. python3 scripts/rl/gen_seq_opp_data.py \
        output/seq_opp_data_20000.npz 20000 32
    PYTHONPATH=. python3 scripts/rl/gen_seq_opp_data.py \
        output/seq_opp_mixed_20000.npz 20000 24 --mix \
        --nn-path output/nn_full_action_best.pt
"""

import os
import sys
import time
import argparse
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch
import agent as agent_module
import algo.context.v3 as context_v3
from driver import engine
from algo.agents.belief_expectimax import BeliefExpectimaxAgent
from algo.eval.v2 import winning_tiles, shanten
from algo.nn.features import extract_features, tile_to_index

MAX_SEQ = 40
SHARD_GAMES = 1000
_TYPE_CODE = {'Baseline': 0, 'BeliefExp': 1, 'Hybrid': 2}


def _seat(name):
    return int(name.split('@')[-1]) if '@' in name else 0


def _type_of(ag):
    cls = type(ag).__name__
    if 'Hybrid' in cls:
        return _TYPE_CODE['Hybrid']
    if 'Belief' in cls:
        return _TYPE_CODE['BeliefExp']
    return _TYPE_CODE['Baseline']


def _ctx_of(ag):
    """取 agent 的 ContextV3（Hybrid 的 context 在 nn_agent 上）。"""
    ctx = getattr(ag, 'context', None)
    if ctx is not None:
        return ctx
    nn_ag = getattr(ag, 'nn_agent', None)
    if nn_ag is not None:
        return getattr(nn_ag, 'context', None)
    return None


class TrackedBaseline(agent_module.Agent):
    """Baseline（永不报听）+ 被动 ContextV3 追踪（自记弃牌）。"""

    def __init__(self, name):
        super().__init__(name, verbose=False)
        self.context = context_v3.ContextV3()

    def init_tiles(self, l):
        super().init_tiles(l)
        self.context = context_v3.ContextV3()

    def handle_msg(self, msg):
        if msg.type == 'put':
            self.context.see_tile(msg.data, msg.sender)
        elif msg.type == 'tenpai':
            self.context.declare_tenpai(msg.sender)
        return super().handle_msg(msg)

    def next(self):
        t = super().next()
        self.context.see_tile(t, self.name)
        return t


def _make_agents(mix, nn_path, rng):
    if not mix:
        return [BeliefExpectimaxAgent('BE', verbose=False) for _ in range(4)]
    agents = []
    for _ in range(4):
        kind = rng.choice(['BE', 'Base', 'Hy'])
        if kind == 'BE':
            agents.append(BeliefExpectimaxAgent('BE', verbose=False))
        elif kind == 'Base':
            agents.append(TrackedBaseline('Base'))
        else:
            from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent
            agents.append(HybridNNBeliefAgent(
                'Hy', nn_model_path=nn_path, belief_kind='beliefexp',
                device='cpu', temperature=0.0, verbose=False))
    return agents


def _play_batch(args_tuple):
    seeds, mix, nn_path = args_tuple
    torch.set_num_threads(1)
    batch = []
    for seed in seeds:
        rng = random.Random(seed)
        agents = _make_agents(mix, nn_path, rng)
        for i, a in enumerate(agents):
            a.name = f'{a.name}@{i}'
        decl_step = {}   # seat -> 报听步（该玩家第几次弃牌后报听）
        samples = []

        def cb(ags, turn, event, info):
            if event != 'decision':
                return
            cur = ags[turn]
            if len(cur.cur) != 14:
                return
            ctx = _ctx_of(cur)
            if ctx is None:
                return
            # 更新报听步：locked_names 中新出现的玩家，其报听发生在其最近一次弃牌后
            for p in info.get('locked_names', set()):
                s = _seat(p)
                if s not in decl_step:
                    decl_step[s] = len(ctx.discards.get(p, []))

            feats = extract_features(ctx, list(cur.cur), cur.name)
            opp_seq = np.full((3, MAX_SEQ), -1, dtype=np.int64)
            opp_seq_len = np.zeros(3, dtype=np.int64)
            opp_decl = np.full(3, -1, dtype=np.int64)
            opp_decl_step = np.full(3, -1, dtype=np.int64)
            opp_meld_hot = np.zeros((3, 34), dtype=np.float32)
            opp_type = np.zeros(3, dtype=np.int64)
            opp_tenpai = np.zeros(3, dtype=np.float32)
            opp_wait = np.zeros((3, 34), dtype=np.float32)

            self_s = _seat(cur.name)
            for off in (1, 2, 3):
                opp = ags[(turn + off) % 4]
                os_ = _seat(opp.name)
                assert os_ == (self_s + off) % 4
                row = off - 1
                opp_type[row] = _type_of(opp)
                seq = ctx.discards.get(opp.name, [])
                ids = [tile_to_index(t) for t in seq][:MAX_SEQ]
                opp_seq[row, :len(ids)] = ids
                opp_seq_len[row] = len(ids)
                if opp.name in ctx.tenpai_players:
                    opp_decl[row] = 1
                    opp_decl_step[row] = decl_step.get(os_, -1)
                for _, t in opp.melds:
                    opp_meld_hot[row, tile_to_index(t)] = 1.0
                hand = opp.full_hand()
                try:
                    if shanten(list(hand)) == 0:
                        opp_tenpai[row] = 1.0
                        for t in winning_tiles(list(hand), None):
                            opp_wait[row, tile_to_index(t)] = 1.0
                except Exception:
                    pass

            samples.append({
                'feats': np.asarray(feats, dtype=np.float32),
                'opp_seq': opp_seq,
                'opp_seq_len': opp_seq_len,
                'opp_decl': opp_decl,
                'opp_decl_step': opp_decl_step,
                'opp_meld_hot': opp_meld_hot,
                'opp_type': opp_type,
                'opp_tenpai': opp_tenpai,
                'opp_wait': opp_wait,
            })

        engine.play_game(agents, seed=seed, state_callback=cb)
        batch.extend(samples)
    return batch


def _save_shard(path, samples):
    np.savez_compressed(
        path,
        feats=np.stack([s['feats'] for s in samples]),
        opp_seq=np.stack([s['opp_seq'] for s in samples]),
        opp_seq_len=np.stack([s['opp_seq_len'] for s in samples]),
        opp_decl=np.stack([s['opp_decl'] for s in samples]),
        opp_decl_step=np.stack([s['opp_decl_step'] for s in samples]),
        opp_meld_hot=np.stack([s['opp_meld_hot'] for s in samples]),
        opp_type=np.stack([s['opp_type'] for s in samples]),
        opp_tenpai=np.stack([s['opp_tenpai'] for s in samples]),
        opp_wait=np.stack([s['opp_wait'] for s in samples]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='最终合并输出 .npz 路径')
    parser.add_argument('n_games', type=int, default=20000)
    parser.add_argument('workers', type=int, default=32)
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--mix', action='store_true',
                        help='BeliefExp/Baseline/Hybrid 混合agent池（需 --nn-path）')
    parser.add_argument('--nn-path', default='output/nn_full_action_best.pt')
    parser.add_argument('--shard-dir', default=None,
                        help='shard 目录（默认 <output无后缀>_shards）')
    args = parser.parse_args()

    shard_dir = args.shard_dir or args.output.replace('.npz', '') + '_shards'
    os.makedirs(shard_dir, exist_ok=True)

    n_shards = (args.n_games + SHARD_GAMES - 1) // SHARD_GAMES
    todo = []
    for i in range(n_shards):
        path = os.path.join(shard_dir, f'shard_{i:04d}.npz')
        if os.path.exists(path):
            continue
        lo = args.seed_offset + i * SHARD_GAMES
        hi = min(lo + SHARD_GAMES, args.seed_offset + args.n_games)
        todo.append((i, (list(range(lo, hi)), args.mix, args.nn_path), path))

    print(f'total shards {n_shards}, todo {len(todo)}, mix={args.mix}, '
          f'shard_dir={shard_dir}')
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_play_batch, batch_args): (i, path)
                for i, batch_args, path in todo}
        for fut in as_completed(futs):
            i, path = futs[fut]
            samples = fut.result()
            _save_shard(path, samples)
            done += 1
            el = time.time() - t0
            eta = el / max(done, 1) * (len(todo) - done)
            print(f'  shard {i} saved ({len(samples)} samples)  '
                  f'{done}/{len(todo)}  elapsed {el:.0f}s eta {eta:.0f}s',
                  flush=True)

    # merge
    print('merging shards...')
    parts = []
    for i in range(n_shards):
        path = os.path.join(shard_dir, f'shard_{i:04d}.npz')
        parts.append(np.load(path))
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    np.savez_compressed(args.output, **merged)
    n = merged['feats'].shape[0]
    tenpai_rate = merged['opp_tenpai'].mean()
    silent_rate = (merged['opp_tenpai'] * (merged['opp_decl'] < 0.5)).mean()
    print(f'Done. {n} samples, per-opp tenpai rate {tenpai_rate:.3f}, '
          f'silent-tenpai rate {silent_rate:.4f}, saved to {args.output}')


if __name__ == '__main__':
    main()
