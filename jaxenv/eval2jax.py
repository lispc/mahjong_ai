# -*- coding: utf-8 -*-
"""arena Baseline（algo.select 默认 eval2 度量）的 JAX 纯函数移植，用于 PPO 对手池。

Ground truth 与镜像要点（parity 以 arena 实际代码路径为准）：
- arena Baseline = agent.Agent.next() -> algo.select(cur, metric_f=algo.eval2)。
  algo.eval0/eval2 优先走 Cython algo/eval/_fast_eval0.pyx；该快路径**不传**
  config.pair_coef(=0.6)，用默认 pair_coef=1.0。eval0 单点 max 结构虽然与
  coef 无关（max = G* + coef*[G* 处可达对子]），但 eval2 是 eval0 的概率加权
  和，候选间比较依赖 coef —— 故本模块按 coef=1.0 复刻，eval0 用整数
  metric m = G + [P>0]（与 Cython double 值逐位相等）。
- 剩余分布镜像缺省空 Context：只看自己手牌，prob_k = (4 - hand_k) / Σ(4 - hand)
  （13 张分母 123，14 张分母 122）。不看弃牌/副露。
- eval1(hand) = Σ_k prob_k · eval0(hand+k)；eval2(hand13) = Σ_k prob_k · eval1(hand13+k)。
  公共分母 123·122 对所有弃牌候选相同，故比较用**整数分子**
  N = Σ_k w_k Σ_j (w_j − δ_jk) · m(hand+k+j)（w = 4 − hand），无 float 误差。
- 弃牌 tie-break 镜像 sorted((metric, tile), reverse=True)[0]：metric 降序，
  平局取 tile id 大者（id 序 == idx 序，故取 idx 大者）。注意 Cython 的 float64
  顺序求和在「数学上严格平局」时可能以 ±1ulp 打破该规则（见 test_eval2jax 对
  tie-split 样本的单独统计），JAX 侧用整数分子保证确定性。
- eval0 分解语义：每组（3 数牌花色 + 字牌）允许留孤张地提取面子/对子，每组
  只需两个数：a = 无对子分解的最大面子数、b = 含对子分解的最大面子数
  （不可达 -1；b<=a 恒成立，(g,有对) ⇒ (g,无对)）。跨组合并 = 16 种
  「哪几组提供对子」模式取 max Σg + [任一组出对子]。表见
  jaxenv/gen_eval2_tables.py（首次使用前先运行它生成 tables_eval2.npz）。
  注意 tables.npz 的 win 掩码 g 封顶 4 不够用：eval2 内层 eval0 会看到 15 张
  手牌，一个花色可达 5 面子。
- CLAIM：stage==HU 则胡(37)否则 pass(34)（agent.Agent：能胡必胡、不碰不杠；
  env 只对物理可胡的玩家开放 HU 阶段，与基类 respond_hu 等价）。
- TENPAI：恒 no(39)（agent.Agent.declare_tenpai 基类恒 False）。
- 不变式：本 agent 从不碰/杠 => n_melds 恒 0、DISCARD 时闭手恒 14 张
  （algo.select assert len==14 的镜像）。若违反（不会被本模块触发）结果无定义。
- done 状态返回 0（env.step 对 done 为 no-op，值无意义但有限）。

接口镜像 jaxenv/greedy.py：eval2_action(state) -> int8；批量用 vmap。
"""

import os

import numpy as np

import jax
import jax.numpy as jnp

from . import env

TABLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'tables_eval2.npz')

_TABLES = None


def load_tables(path=None):
    """加载 tables_eval2.npz（进程内缓存 numpy；原因同 rules.load_tables：
    jnp 数组缓存到全局再跨 trace 使用会触发 UnexpectedTracerError）。"""
    global _TABLES
    if _TABLES is None:
        d = np.load(path or TABLES_PATH)
        _TABLES = {k: d[k] for k in d.files}
    return _TABLES


_SUIT_N = 5 ** 9
_HONOR_N = 5 ** 7
_P5_9 = [5 ** i for i in range(9)]
_P5_7 = [5 ** i for i in range(7)]

# 16 种「哪几组提供对子」模式：bit i = 第 i 组走 b_i（含对子）路径
_PAT = np.array([[(s >> i) & 1 for i in range(4)] for s in range(16)], np.int32)

_IDX = jnp.arange(34, dtype=jnp.int32)
_EYE = jnp.eye(34, dtype=jnp.int32)

# eval2 公共分母（13 张外层的 123 与 14 张内层的 122），仅用于 float 形式展示
_DEN13 = 123 * 122


def _eval0_int(counts):
    """34 维计数 -> int32 eval0 metric m = G + [P>0]（pair_coef=1.0 语义）。

    与 Cython _metric_from_counts(pair_coef=1.0) 的 double 值逐位相等。
    counts 允许含 0..6 的计数（无效候选产生的越界 base-5 编码被 clip，
    其值只乘零权重，不影响结果）。
    """
    T = load_tables()
    suit_a = jnp.asarray(T['suit_a'])
    suit_b = jnp.asarray(T['suit_b'])
    honor_a = jnp.asarray(T['honor_a'])
    honor_b = jnp.asarray(T['honor_b'])
    p5_9 = jnp.asarray(_P5_9, jnp.int32)
    p5_7 = jnp.asarray(_P5_7, jnp.int32)
    c = counts.astype(jnp.int32)
    i0 = jnp.clip(jnp.dot(c[0:9], p5_9), 0, _SUIT_N - 1)
    i1 = jnp.clip(jnp.dot(c[9:18], p5_9), 0, _SUIT_N - 1)
    i2 = jnp.clip(jnp.dot(c[18:27], p5_9), 0, _SUIT_N - 1)
    ih = jnp.clip(jnp.dot(c[27:34], p5_7), 0, _HONOR_N - 1)
    a = jnp.stack([suit_a[i0], suit_a[i1], suit_a[i2], honor_a[ih]]).astype(jnp.int32)
    b = jnp.stack([suit_b[i0], suit_b[i1], suit_b[i2], honor_b[ih]]).astype(jnp.int32)

    pat = jnp.asarray(_PAT)                                   # (16,4)
    use_b = pat.astype(bool)
    chosen = jnp.where(use_b, b[None, :], a[None, :])         # (16,4)
    invalid = (use_b & (b[None, :] < 0)).any(axis=1)          # 该组无含对分解
    has_pair = pat.sum(axis=1) > 0                            # (16,)
    val = chosen.sum(axis=1) + has_pair.astype(jnp.int32)
    return jnp.where(invalid, jnp.int32(-1000), val).max()


def eval0_counts(counts):
    """34 维计数 -> float eval0 metric（镜像 algo.eval0，Cython 快路径）。"""
    return _eval0_int(counts).astype(jnp.float32)


def _eval2_num13(hand13):
    """13 张计数 -> int32 分子 N = 123·122·eval2(hand)（空 Context 分布）。

    整数精确：候选间比较/tie-break 无 float 噪声。
    """
    h = hand13.astype(jnp.int32)
    w = jnp.clip(jnp.int32(4) - h, 0, 4)                      # (34,) 外层权重
    H = (h[None, None, :] + _EYE[:, None, :]
         + _EYE[None, :, :])                                  # (k,j,34)
    m = jax.vmap(jax.vmap(_eval0_int))(H)                     # (k,j) 15 张 eval0
    wj = jnp.clip(w[None, :] - _EYE, 0, 4)                    # (k,j) 内层权重 w_j−δ_jk
    return (w[:, None] * wj * m).sum()


def eval2_counts(counts13):
    """13 张计数 -> float eval2 metric（镜像 algo.eval2 / Cython eval2_metric_tiles）。"""
    return _eval2_num13(counts13).astype(jnp.float32) / float(_DEN13)


def _discard_scores(hand14):
    """14 张计数 -> (34,) 每种弃牌的整数分子 N（非法弃牌值为垃圾，调用方掩掉）。"""
    hands13 = hand14.astype(jnp.int32)[None, :] - _EYE        # (34,34)
    return jax.vmap(_eval2_num13)(hands13)


def eval2_discard_idx(hand14):
    """14 张计数 -> 弃牌 idx（argmax N，平局取 idx 大者；镜像 algo.select(...)[0]）。

    不考虑锁手强制弃牌（arena Baseline 无此概念）；env 内的合法性交由
    eval2_action 用 legal_mask 处理。测试/调试入口。
    """
    n = _discard_scores(hand14)
    key = jnp.where(jnp.asarray(hand14) > 0, n * 64 + _IDX, jnp.int32(-1))
    return jnp.argmax(key).astype(jnp.int8)


def _discard_action(state, mask):
    """DISCARD 分支：合法集内 eval2 最大化弃牌，平局取 idx 大者。"""
    turn = state.turn.astype(jnp.int32)
    n = _discard_scores(state.hands[turn])
    # 排序键：N 为主键，idx 为次键（大者优先；镜像 (metric, tile) 降序 sorted）
    key = jnp.where(mask[:34], n * 64 + _IDX, jnp.int32(-1))
    return jnp.argmax(key).astype(jnp.int8)


def eval2_action(state):
    """单个 State -> int8 action（40 动作空间）。批量请 vmap(eval2_action)。"""
    mask = env.legal_mask(state)
    disc = _discard_action(state, mask)
    claim = jnp.where(state.claim_stage == jnp.int8(env.STAGE_HU),
                      jnp.int8(env.A_HU), jnp.int8(env.A_PASS))
    act = jnp.where(state.phase == jnp.int8(env.PHASE_DISCARD), disc,
          jnp.where(state.phase == jnp.int8(env.PHASE_CLAIM), claim,
                    jnp.int8(env.A_TENPAI_NO)))
    # 兜底：done 等 mask 全 False 时回退到合法集第一个动作（无实际影响）
    return jnp.where(mask[act.astype(jnp.int32)], act,
                     jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int8))
