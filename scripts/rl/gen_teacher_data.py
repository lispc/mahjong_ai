# -*- coding: utf-8 -*-
"""蒸馏数据生成：用一个强教师自对弈，产 (features, teacher_action, outcome)。

teacher 可选：
  beliefexp  —— BeliefExpectimaxAgent（防守最好、eval2 级，CPU，快，推荐）
  v3deep     —— BeliefExpectimaxV3Agent(depth=D, conv-BC 候选, eval0 leaf)（深搜索，慢）
  baseline   —— agent.Agent（原始 eval2，空 context）

4 座位全是教师 → 每局 ~52 样本。CPU 专用（避免 32 进程抢 GPU）。带 checkpoint：
每完成一批把累计数据存 `<out>.checkpoint.npz`，最终存 `<out>`。断点续跑用不同 seed_base 追加。

用法：
  PYTHONPATH=. python3 scripts/rl/gen_teacher_data.py \
      [out=output/nn_teacher_data.npz] [total_games=3000] [workers=32] \
      [teacher=beliefexp] [depth=2] [cand=output/nn_conv_bc.pt] [seed_base=0]
"""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')   # 强制 CPU

import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from algo.nn.features import extract_features, extract_features_ext, tile_to_index

TEACHER = 'beliefexp'
DEPTH = 2
CAND = 'output/nn_conv_bc.pt'
FEATURES = 'base'   # 'base'(175) | 'ext'(212, 含危险度/防守)


def _extract(ctx, hand, name):
    return extract_features_ext(ctx, hand, name) if FEATURES == 'ext' \
        else extract_features(ctx, hand, name)


def _make_collector(name):
    """按 TEACHER 返回一个会记录 (features, action) 的 collector agent。"""
    if TEACHER == 'beliefexp':
        from algo.agents.belief_expectimax import BeliefExpectimaxAgent as Base
        kw = {}
    elif TEACHER == 'v3deep':
        from algo.agents.belief_expectimax_v3 import BeliefExpectimaxV3Agent as Base
        kw = dict(expectimax_depth=DEPTH, max_candidates=5, leaf_evaluator='eval0',
                  candidate_policy='nn', candidate_model_path=CAND)
    elif TEACHER == 'baseline':
        import agent as _am
        Base = _am.Agent
        kw = {}
    else:
        raise ValueError(TEACHER)

    class _Collector(Base):
        def __init__(self, nm):
            super().__init__(nm, verbose=False, **kw) if kw else super().__init__(nm, verbose=False)
            import algo.context.v3 as _cv3
            self._has_ctx = hasattr(self, 'context')
            if not self._has_ctx:
                self.context = _cv3.ContextV3()   # baseline 无 context，自己维护供特征用
            self.steps = []

        def init_tiles(self, l):
            super().init_tiles(l)
            import algo.context.v3 as _cv3
            if not getattr(self, '_has_ctx', False):
                self.context = _cv3.ContextV3()
            self.steps = []

        def handle_msg(self, msg):
            if not getattr(self, '_has_ctx', False):
                if msg.type == 'put':
                    self.context.see_tile(msg.data, msg.sender)
                elif msg.type == 'tenpai':
                    self.context.declare_tenpai(msg.sender)
            return super().handle_msg(msg)

        def next(self):
            feats = _extract(self.context, self.cur, self.name)
            disc = super().next()
            if not getattr(self, '_has_ctx', False):
                self.context.see_tile(disc, self.name)
            self.steps.append((np.asarray(feats, dtype=np.float32), tile_to_index(disc)))
            return disc

    return _Collector(name)


def _outcome(result, name):
    if result['win_type'] == 'draw':
        return 0.0
    return 1.0 if result.get('winner') == name else -1.0


def _init_worker(teacher, features, depth, cand):
    """spawn 子进程不继承运行时设置的全局，故用 initializer 显式注入。"""
    global TEACHER, FEATURES, DEPTH, CAND
    TEACHER, FEATURES, DEPTH, CAND = teacher, features, depth, cand


def _worker(args):
    n_games, seed_base = args
    torch.set_num_threads(1)
    import random
    from driver.engine import play_game
    Xs, ys, vs = [], [], []
    for g in range(n_games):
        random.seed(seed_base + g)
        agents = [_make_collector(f'T@{s}') for s in range(4)]
        result = play_game(agents)
        for a in agents:
            o = _outcome(result, a.name)
            for feats, act in a.steps:
                Xs.append(feats)
                ys.append(act)
                vs.append(o)
    if not Xs:
        return (np.zeros((0, 175), np.float32), np.zeros((0,), np.int64), np.zeros((0,), np.float32))
    return (np.stack(Xs), np.asarray(ys, np.int64), np.asarray(vs, np.float32))


def main():
    global TEACHER, DEPTH, CAND, FEATURES
    out = sys.argv[1] if len(sys.argv) > 1 else 'output/nn_teacher_data.npz'
    total_games = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    TEACHER = sys.argv[4] if len(sys.argv) > 4 else 'beliefexp'
    FEATURES = sys.argv[5] if len(sys.argv) > 5 else 'base'
    seed_base = int(sys.argv[6]) if len(sys.argv) > 6 else 0

    games_per_task = 5
    tasks = []
    g0, remaining = seed_base, total_games
    while remaining > 0:
        g = min(games_per_task, remaining)
        tasks.append((g, g0))
        g0 += g
        remaining -= g
    print(f'teacher data gen: teacher={TEACHER} features={FEATURES} total_games={total_games} '
          f'workers={workers} tasks={len(tasks)} -> {out}', flush=True)

    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    Xs, ys, vs = [], [], []
    done = 0
    t0 = time.time()
    ck = out.replace('.npz', '.checkpoint.npz')
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(TEACHER, FEATURES, DEPTH, CAND)) as pool:
        for i, (X, y, v) in enumerate(pool.imap_unordered(_worker, tasks)):
            if len(X):
                Xs.append(X); ys.append(y); vs.append(v)
            done += tasks[i][0]
            if (i + 1) % max(1, workers) == 0 or (i + 1) == len(tasks):
                nX = np.concatenate(Xs) if Xs else np.zeros((0, 175), np.float32)
                nY = np.concatenate(ys) if ys else np.zeros((0,), np.int64)
                nV = np.concatenate(vs) if vs else np.zeros((0,), np.float32)
                np.savez(ck, X=nX, y=nY, v=nV)
                rate = done / (time.time() - t0 + 1e-9)
                eta = (total_games - done) / max(rate, 1e-9) / 60
                print(f'  [{done}/{total_games}] samples={len(nX)} {rate:.2f} g/s eta={eta:.1f}min', flush=True)
    X = np.concatenate(Xs) if Xs else np.zeros((0, 175), np.float32)
    y = np.concatenate(ys) if ys else np.zeros((0,), np.int64)
    v = np.concatenate(vs) if vs else np.zeros((0,), np.float32)
    np.savez(out, X=X, y=y, v=v)
    print(f'saved {out}: {len(X)} samples in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
