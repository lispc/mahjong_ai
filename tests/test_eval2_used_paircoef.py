# -*- coding: utf-8 -*-
"""eval2 的 used 条件化与 pair_coef 参数的 parity / 回归测试（2026-07-20）。

覆盖：
1. Cython 快路径（used + 显式 pair_coef=0.6）与纯 Python 路径
   （eval_rec，config.pair_coef=0.6，Context.used 生效）逐值一致。
2. algo.eval2 的 pair_coef 参数透传正确。
3. 默认行为回归：algo.eval2(hand) 与 Cython 空 used、pair_coef=1.0 完全一致
   （A/B 期间默认值不得改变）。
4. UsedAwareContext 接线：algo.eval2 能拿到 used；普通 Context 仍然被忽略。
5. BeliefExpectimaxAgent.used_aware_eval2 开关的 context 类型。
"""

import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import context as ctx_module
import algo
from algo.eval import _fast_eval0
from algo.eval.legacy import eval_rec, eval1_counts

TILE_IDS = list(range(1, 10)) + list(range(11, 20)) + list(range(21, 30)) + list(range(31, 38))


def _rand_case(rng):
    """随机生成 (hand13, used_dict)，保证每种牌 hand + used <= 4。"""
    counts = {t: 0 for t in TILE_IDS}
    hand = []
    for _ in range(13):
        t = rng.choice(TILE_IDS)
        while counts[t] >= 4:
            t = rng.choice(TILE_IDS)
        counts[t] += 1
        hand.append(t)
    used = {}
    n_seen = rng.randint(0, 40)
    for _ in range(n_seen):
        t = rng.choice(TILE_IDS)
        if counts[t] + used.get(t, 0) < 4:
            used[t] = used.get(t, 0) + 1
    return hand, used


def _used_list(used):
    return [t for t, cnt in used.items() for _ in range(cnt)]


def _python_eval2(hand, used):
    """纯 Python 路径（eval_rec），Context.used 生效，pair_coef=config.pair_coef。"""
    c = ctx_module.Context()
    c.used = dict(used)
    return eval_rec(list(hand), eval1_counts, c)


def test_cython_python_parity():
    rng = random.Random(42)
    n_diff_used_effect = 0
    for i in range(150):
        hand, used = _rand_case(rng)
        py = _python_eval2(hand, used)
        cy = _fast_eval0.eval2_metric_tiles(list(hand), _used_list(used),
                                            float(config.pair_coef))
        assert abs(py - cy) < 1e-9, f'case {i}: python {py} != cython {cy} (used={used})'
        cy_empty = _fast_eval0.eval2_metric_tiles(list(hand), [], float(config.pair_coef))
        if used and abs(cy_empty - cy) > 1e-12:
            n_diff_used_effect += 1
    # used 应当对相当一部分随机状态产生实际影响（否则测试无鉴别力）
    assert n_diff_used_effect > 50, f'used 生效的样本太少: {n_diff_used_effect}/150'
    print(f'parity ok (150 cases, used 改变结果 {n_diff_used_effect}/150)')


def test_pair_coef_passthrough():
    rng = random.Random(7)
    n_diff = 0
    for i in range(60):
        hand, used = _rand_case(rng)
        v06 = algo.eval2(list(hand), pair_coef=0.6)
        direct = _fast_eval0.eval2_metric_tiles(list(hand), [], 0.6)
        assert abs(v06 - direct) < 1e-12, f'pair_coef 透传失败: {v06} != {direct}'
        v10 = algo.eval2(list(hand))
        if abs(v06 - v10) > 1e-12:
            n_diff += 1
    assert n_diff > 5, f'pair_coef 0.6 vs 1.0 几乎无差异 ({n_diff}/60)，参数可能没生效'
    print(f'pair_coef passthrough ok (0.6 vs 1.0 差异 {n_diff}/60)')


def test_default_behavior_unchanged():
    """默认 algo.eval2(hand) 必须仍等于 Cython 空 used + pair_coef=1.0。"""
    rng = random.Random(123)
    for i in range(60):
        hand, _ = _rand_case(rng)
        v = algo.eval2(list(hand))
        ref = _fast_eval0.eval2_metric_tiles(list(hand), [], 1.0)
        assert abs(v - ref) < 1e-12, f'默认行为漂移: {v} != {ref}'
        # 普通 Context（带 used 但无 all_tiles_as_dict）仍被忽略
        c = ctx_module.Context()
        c.used = {hand[0]: 1}
        v2 = algo.eval2(list(hand), c)
        assert abs(v2 - ref) < 1e-12, f'普通 Context 的 used 不再被忽略？{v2} != {ref}'
    print('default behavior unchanged ok (60 cases)')


def test_used_aware_context_wiring():
    rng = random.Random(99)
    n_diff = 0
    for i in range(60):
        hand, used = _rand_case(rng)
        if not used:
            continue
        c = ctx_module.UsedAwareContext()
        c.used = dict(used)
        v = algo.eval2(list(hand), c)
        ref = _fast_eval0.eval2_metric_tiles(list(hand), _used_list(used), 1.0)
        assert abs(v - ref) < 1e-12, f'UsedAwareContext 接线错误: {v} != {ref}'
        v_empty = algo.eval2(list(hand))
        if abs(v - v_empty) > 1e-12:
            n_diff += 1
    assert n_diff > 10, f'UsedAwareContext 几乎没有影响 ({n_diff})'
    print(f'UsedAwareContext wiring ok ({n_diff} cases differ)')


def test_belief_agent_switch():
    from algo.agents.belief_expectimax import BeliefExpectimaxAgent
    a = BeliefExpectimaxAgent('BE', used_aware_eval2=True)
    c = a._legacy_context()
    assert hasattr(c, 'all_tiles_as_dict'), 'used_aware_eval2=True 应有 all_tiles_as_dict'
    b = BeliefExpectimaxAgent('BE2')
    c2 = b._legacy_context()
    assert not hasattr(c2, 'all_tiles_as_dict'), '默认应保持历史行为（无 all_tiles_as_dict）'
    print('belief agent switch ok')


if __name__ == '__main__':
    test_cython_python_parity()
    test_pair_coef_passthrough()
    test_default_behavior_unchanged()
    test_used_aware_context_wiring()
    test_belief_agent_switch()
    print('All eval2 used/pair_coef tests passed.')
