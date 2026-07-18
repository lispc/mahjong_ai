# -*- coding: utf-8 -*-
"""jaxenv/greedy.py 验证：4×greedy 自对弈分布 vs Python 引擎镜像策略。

- 非 jit 验证（test_eager_smoke）：直接 eager 调用 jax.vmap(greedy_action) 与
  jax.vmap(env.step) 跑完整局，证明 greedy_action 不依赖 jit 即可用、动作恒合法。
  （纯 eager 跑全部 n 局不可行：eager lax.cond/switch 每步重 trace——
  is_win_counts 的 merge 为 Python 级循环，实测 ~5-6 s/step。）
- JAX 侧：host 循环 + 逐步 jit（act/step 分别 jit，非整局 while_loop；与
  test_env.py invariants 同风格）跑 n 局 4×greedy 自对弈，统计平均局长
  （n_draws）、流局率、胡牌构成（自摸/荣和）。
- Python 侧：driver/engine.py + 内联最小 shanten 贪心 agent（同 tie-break：
  字牌 > 幺九 > 中张，同组 idx 大者优先；能胡必胡、不碰不杠、报听恒 yes）跑 n 局。
- 容差：平均局长 ±10%，流局率 ±5pp。

用法：PYTHONPATH=. python3 jaxenv/test_greedy.py [--games 200] [--seed 7]
"""

import argparse
import random
import time

import numpy as np

import jax
import jax.numpy as jnp

from jaxenv import env, rules
from jaxenv.greedy import greedy_action


def test_eager_smoke(n=2, seed=99, max_steps=6):
    """非 jit 验证：直接 eager 调用 vmap(greedy_action)/vmap(env.step) 数步。

    证明 greedy_action 不依赖 jit 即可用、产出动作恒合法。只跑数步——
    纯 eager 每步重 trace（~5-6 s/step），完整局由下方逐步 jit 版本覆盖。
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    states = jax.vmap(env.init)(keys)
    steps = 0
    while not bool(jnp.all(states.done)) and steps < max_steps:
        acts = jax.vmap(greedy_action)(states)              # eager（非 jit）
        masks = jax.vmap(env.legal_mask)(states)
        a, m, d = np.asarray(acts), np.asarray(masks), np.asarray(states.done)
        assert m[np.arange(n), a][~d].all(), f'eager: illegal action at step {steps}'
        states, _, _ = jax.vmap(env.step)(states, acts)     # eager（非 jit）
        steps += 1
    print(f'[eager-smoke] {n} games x {steps} steps (non-jit): all actions legal',
          flush=True)


def jax_greedy_games(n, seed):
    """JAX 侧：host 循环 + 逐步 jit（非整局 while_loop）跑 n 局。"""
    act_v = jax.jit(jax.vmap(greedy_action))
    step_v = jax.jit(jax.vmap(env.step))
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    states = jax.vmap(env.init)(keys)
    steps = 0
    while not bool(jnp.all(states.done)):
        states, _, _ = step_v(states, act_v(states))
        steps += 1
        if steps > 1000:
            raise AssertionError('games did not finish in 1000 steps')
    print(f'  JAX side loop: {steps} steps', flush=True)
    return np.asarray(states.win_type), np.asarray(states.n_draws)


class _GreedyAgent:
    """Python 引擎侧镜像：最小 shanten 贪心（与 jaxenv/greedy.py 同 tie-break）。

    【胡牌语义对齐】基类 Agent.add/respond_hu 用 algo.is_succ 判胡（不含七对子），
    而 jaxenv env 实现的是 v2 语义（含七对子，见 env.py 头注第 6 条）；greedy 会
    主动追七对子，语义差会系统性抬高 PY 侧流局率（实测 ~11.5% 的 JAX 局为七对
    子胡牌）。此处把 add/respond_hu 改用 algo.eval.v2.is_win，使两侧规则一致
    （greedy 从不副露，v2.is_win 的 14 张判定在此是精确语义）。
    """

    @staticmethod
    def make(name):
        from agent import Agent
        from algo.eval.v2 import shanten, is_win

        class _A(Agent):
            def __init__(self):
                super().__init__(name, verbose=False)

            def next(self):
                best_t, best_key = None, None
                for t in set(self.cur):
                    h = list(self.cur)
                    h.remove(t)
                    s = shanten(h)                       # 13 张弃后向听
                    idx = rules.TILE_TO_IDX[t]
                    grp = 2 if idx >= 27 else (1 if idx % 9 in (0, 8) else 0)
                    key = (s, -grp, -idx)                # 与 JAX 侧排序键同序
                    if best_key is None or key < best_key:
                        best_key, best_t = key, t
                self.cur.remove(best_t)
                return best_t

            def add(self, t):
                self.cur.append(t)
                return is_win(self.full_hand())          # 自摸判定（v2 语义）

            def respond_hu(self, tile_val, context=None):
                return is_win(self.full_hand() + [tile_val])

            def declare_tenpai(self, hand, context):
                return True

            def respond_peng(self, tile_val, context=None):
                return False

            def respond_gang(self, tile_val, context=None):
                return False

        return _A()


def py_greedy_games(n, seed):
    """Python 引擎侧：n 局，返回 (win_types, lengths)（win_type: 1=self 2=ron 3=draw）。"""
    from driver import engine
    wt, ln = [], []
    for i in range(n):
        agents = [_GreedyAgent.make(f'p{k}') for k in range(4)]
        res = engine.play_game(agents, seed=seed * 1000003 + i, record_log=True)
        wt.append({'self': 1, 'ron': 2, 'draw': 3}[res['win_type']])
        ln.append(sum(1 for ev in res['event_log'] if ev['type'] == 'draw'))
        if (i + 1) % 50 == 0:
            print(f'  PY side {i + 1}/{n}', flush=True)
    return np.array(wt), np.array(ln)


def stats(wt, ln, name):
    n_dec = (wt != 3).sum()
    line = (f'  {name}: n={len(wt)} len_mean={ln.mean():.2f} '
            f'draw={100 * (wt == 3).mean():.1f}% self={100 * (wt == 1).mean():.1f}% '
            f'ron={100 * (wt == 2).mean():.1f}% '
            f'(self share of decided: {100 * (wt == 1).sum() / max(n_dec, 1):.1f}%)')
    print(line, flush=True)
    return ln.mean(), (wt == 3).mean(), (wt == 1).mean(), (wt == 2).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=200)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)

    t0 = time.time()
    test_eager_smoke()
    print(f'  eager smoke done ({time.time() - t0:.1f}s)', flush=True)

    t0 = time.time()
    jwt, jln = jax_greedy_games(args.games, args.seed)
    print(f'  JAX side done ({time.time() - t0:.1f}s)', flush=True)

    t0 = time.time()
    pwt, pln = py_greedy_games(args.games, args.seed)
    print(f'  PY side done ({time.time() - t0:.1f}s)', flush=True)

    print(f'[greedy-dist] {args.games} games/side, 策略=不碰/不杠/报听恒yes+'
          f'最小向听贪心弃牌(字牌>幺九>中张 tie-break)+能胡必胡')
    jl, jd, js, jr = stats(jwt, jln, 'JAX ')
    pl, pd, ps, pr = stats(pwt, pln, 'PY  ')
    len_ok = abs(jl - pl) / max(pl, 1e-9) <= 0.10
    draw_ok = abs(jd - pd) <= 0.05
    print(f'  局长差 {100 * (jl - pl) / max(pl, 1e-9):+.2f}% (容差 ±10%), '
          f'流局率差 {100 * (jd - pd):+.2f}pp (容差 ±5pp)')
    assert len_ok, '平均局长差异超容差'
    assert draw_ok, '流局率差异超容差'
    print('[greedy-dist] passed')


if __name__ == '__main__':
    main()
