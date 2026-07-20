# -*- coding: utf-8 -*-
"""JAX 复式（duplicate）评测 arena：scripts/rl/benchmark_duplicate.py 的可选第二后端。

**纯新增（multi-backend）**：Python 后端（driver/engine.py + benchmark_duplicate.py）
一字未动；本模块只为「有 JAX 对应物的 agent 子集」把配对 A/B 评测搬上 GPU，
目标是把 1000 pairs 的评测成本降一个数量级。

复式语义（镜像 driver/tournament.py::run_duplicate_tournament，默认只镜像 position 0）：
- 2*N 条 lane 一批 vmap 并行；lane 2k 与 2k+1 共用 init rng key
  ``jax.random.PRNGKey(seed_offset + k)`` —— env.init 同一 key 给出同一牌墙 +
  同一 4 家起手（deal identity，见 test_duplicate_arena.py G1）。
- lane 2k 座位 0 = 候选 A，lane 2k+1 座位 0 = 候选 B；座位 1-3 = 同 3 个对手、
  顺序两条 lane 完全一致。
- 全部 agent 均为确定性 argmax / 纯函数（无采样），整批跑完（max_steps 封顶），
  结果完全确定：同参重跑逐 bit 一致（G1 determinism 门）。
- 命名镜像 tournament.py 的 ``f'{name}@{pos}_{a|b}'`` 约定：
  A lane players_order = [f'{a}@0_a', f'{o1}@1_a', f'{o2}@2_a', f'{o3}@3_a']，
  B lane 同构换 _b。pkl schema 与 benchmark_duplicate.py 一致，复用既有
  reanalysis 工具链。

agent 类型与 dispatch（镜像 jaxenv/ppo.py ~150-230 的 seat_types int8 + jnp.where 模式）：
- TYPE_EVAL2 (0)  = arena Baseline 移植（jaxenv/eval2jax.py，Cython 路径精确 parity）；
- TYPE_BELIEF (1) = BeliefExpectimaxAgent 移植（jaxenv/beliefjax.py，top-1 ~98% parity，
  **非逐 bit** —— 跨后端对比只允许小 delta，见 G2 门注释）；
- TYPE_GREEDY (2) = shanten 贪心（jaxenv/greedy.py，Python 后端无对应物）；
- TYPE_NN_BASE+i (8+i) = 第 i 槽 flax NN，masked argmax 纯前馈（无搜索层），
  params 由调用方提供；同一 run 内全部 NN 座位共享同一架构（单个 model.apply），
  仅参数不同。

NN 座位行为（**AutoHu 风格**，对齐 algo/agents/auto_hu_ppo_agent.py 的部署形态，
以及 ppo.py 的 auto_hu/no_tenpai 强制动作处理）：
- 弃牌：policy(34) masked argmax（build_nn_logits，同 ppo.build_logits 的头映射）；
- 碰/杠声明：response 头 argmax（无 response 头的模型恒 pass，镜像 PPOAgent 回退）；
- 胡声明：默认强制能胡必胡（auto_hu=True，绕过 response 头的 hu 维）；
- 报听：默认强制恒否（no_tenpai=True，对齐 AutoHuPPOAgent.declare_tenpai=False；
  若关闭强制且无 tenpai 头则同样恒否）。
强制动作只对 NN 座位生效 —— eval2/belief/greedy 座位自带声明/报听行为，
不得覆盖（否则破坏与 Python 后端的 parity，尤其 BeliefExp 的报听启发式）。

统计（公式逐行复制 scripts/rl/benchmark_duplicate.py，**不要改**）：
- paired win diff 95% CI：``_paired_ci``（paired 方差公式原样）；
- score-proxy 配对差：自摸 +3 / 点和 +1 / 放炮 -1 / 其余 0（``_seat_score`` 原样）；
- ties = 同 pair 双方都赢或都没赢；candidate-specific 胜率含 startswith 守卫
  （``candidate_wins`` 镜像 `_candidate_wins`，A/B 同名的自配对场景下仍正确）。

已知限制（务必阅读）：
1. **无 hybridnm 座位**：Hybrid 的 BeliefExp 搜索层 + NN 候选未移植。规则断代后的
   标准三件套（baseline, beliefexp, hybridnm:Base）在本后端只能近似
   （例如用 beliefexp 或某个 NN argmax 座位顶替 hybridnm）——评测结论需注明此偏差。
2. beliefjax 是 BeliefExp 的 ~98% top-1 parity 移植，非逐 bit；跨后端（JAX vs
   Python）的 paired diff 允许小 delta（G2 门：|Δ| ≤ 4pp 且 CI 重叠）。
3. 同一 seed 号在两端产生**不同的实际牌墙**（Python 端 tile_pool.Pool(seed) vs
   本端 PRNGKey(seed)）；paired diff 是统计估计，按分布对比即可。
4. NN 座位为 masked argmax 纯前馈（无搜索层、无 belief），与 arena 中 Hybrid 系
   形态不可直接比较；只等价于 AutoHuPPOAgent 式部署。
5. max_steps=600 封顶仍未 done 的局按流局计（env 在牌山耗尽时必终局，正常远
   低于此上限；与 ppo.play_eval_generic 的 ``draw = (winner<0) | ~done`` 一致）。

用法（核心入口）：
    from jaxenv.duplicate_arena import run_duplicate, TYPE_EVAL2, TYPE_BELIEF
    out = run_duplicate(TYPE_EVAL2, TYPE_BELIEF,
                        (TYPE_EVAL2, TYPE_BELIEF, TYPE_BELIEF),
                        n_seeds=128, a_name='Baseline', b_name='BeliefExp',
                        opp_names=('Baseline', 'BeliefExp', 'BeliefExp'))
CLI 见 scripts/rl/benchmark_duplicate_jax.py；测试见 jaxenv/test_duplicate_arena.py。

性能注记：eval2/belief 的 jaxpr 巨大，step block 的 XLA 编译为一次性成本
（2048-lane 批约 2.5 min）。入口点（CLI / 测试）应调用
``enable_compile_cache()`` 开启持久化编译缓存（tmp/jax_dup_compile_cache，
gitignored）：同形状+同类型组合第二次起编译命中（~13s 级进程内开销），
重复 A/B 评测（本模块的主要用途）摊销后才有数量级加速。尾批零头会补齐到
整批，保证每个 run 只有一种 batch 形状（一条缓存项）。
实测（RTX3090，与另一 100%-util 租户共享 GPU）：2048-lane eval2+belief
混合批 ~21k env-steps/s；1000 seeds（2000 局）暖缓存墙钟 ~29s（Python
后端实测 ~64s/1000 seeds，32 workers）。
"""

import math
import os
import time

import numpy as np

import jax
import jax.numpy as jnp

from jaxenv import env
from jaxenv.beliefjax import belief_action
from jaxenv.eval2jax import eval2_action
from jaxenv.greedy import greedy_action
from jaxenv.obs import observe, actor_of

NEG = -1e9
N_ACTIONS = env.N_ACTIONS

# ---------------------------------------------------------------------------
# agent 类型
# ---------------------------------------------------------------------------

TYPE_EVAL2 = 0      # arena Baseline（algo.select eval2 度量）
TYPE_BELIEF = 1     # BeliefExpectimaxAgent 移植
TYPE_GREEDY = 2     # shanten 贪心（Python 后端无对应物）
TYPE_NN_BASE = 8    # NN 槽 i 的类型码 = TYPE_NN_BASE + i

TYPE_NAMES = {TYPE_EVAL2: 'EVAL2', TYPE_BELIEF: 'BELIEF', TYPE_GREEDY: 'GREEDY'}


def type_name(t):
    return TYPE_NAMES.get(int(t), f'NN[{int(t) - TYPE_NN_BASE}]')


DEFAULT_CACHE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'tmp',
    'jax_dup_compile_cache'))


def enable_compile_cache(cache_dir=DEFAULT_CACHE_DIR):
    """开启 XLA 持久化编译缓存（入口点调用；首次编译前调用才有效）。

    eval2/belief 的 step block 首次编译 ~2-2.5 min（一次性）；命中后 ~10s 级。
    返回缓存目录。"""
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', cache_dir)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)
    return cache_dir


# ---------------------------------------------------------------------------
# NN logits（ppo.build_logits 的缺头容忍版）
# ---------------------------------------------------------------------------

def build_nn_logits(out, phase, mask):
    """heads 输出 + phase(B,) + 合法 mask(B,40) -> (B,40) masked logits。

    头映射同 ppo.build_logits：policy(34)->0-33；response(4)->34-37；
    tenpai logit ±->38/39。缺 response 头时声明位除 pass 外全 NEG（恒 pass，
    镜像 PPOAgent 无 response 头的回退）；缺 tenpai 头时 38 NEG / 39=0（恒否）。
    """
    B = out['policy'].shape[0]
    base = jnp.full((B, N_ACTIONS), NEG, jnp.float32)
    pol = base.at[:, :34].set(out['policy'])
    if 'response' in out:
        clm = base.at[:, 34:38].set(out['response'])
    else:
        clm = base.at[:, env.A_PASS].set(0.0)
    if 'tenpai' in out:
        t = out['tenpai'][:, 0]
        ten = base.at[:, 38].set(t).at[:, 39].set(-t)
    else:
        ten = base.at[:, env.A_TENPAI_NO].set(0.0)
    logits = jnp.where((phase == jnp.int8(env.PHASE_DISCARD))[:, None], pol,
             jnp.where((phase == jnp.int8(env.PHASE_CLAIM))[:, None], clm, ten))
    return jnp.where(mask, logits, NEG)


# ---------------------------------------------------------------------------
# 单步 dispatch + scan 块
# ---------------------------------------------------------------------------

def _make_block_runner(model, used_base_types, used_slots, auto_hu, no_tenpai,
                       scan_steps):
    """返回 jitted (states, seat_types, nn_params) -> states（前进 scan_steps 步）。

    used_base_types: frozenset，本 run 实际出现的非 NN 类型（静态，未用类型不计算）。
    used_slots: tuple，本 run 实际出现的 NN 槽位号（静态，每槽一次前向）。
    nn_params: tuple，按 used_slots 顺序的 params pytree。
    """
    used_base_types = frozenset(int(t) for t in used_base_types)
    used_slots = tuple(int(s) for s in used_slots)

    def one_step(states, seat_types, nn_params):
        N = states.done.shape[0]
        masks = jax.vmap(env.legal_mask)(states)              # (N,40)
        phase32 = states.phase.astype(jnp.int32)
        actor = jax.vmap(actor_of)(states)                    # (N,)
        stype = seat_types[jnp.arange(N), actor]              # (N,)
        act = jnp.zeros(N, jnp.int8)
        if TYPE_EVAL2 in used_base_types:
            a = jax.vmap(eval2_action)(states).astype(jnp.int8)
            act = jnp.where(stype == jnp.int8(TYPE_EVAL2), a, act)
        if TYPE_BELIEF in used_base_types:
            a = jax.vmap(belief_action)(states).astype(jnp.int8)
            act = jnp.where(stype == jnp.int8(TYPE_BELIEF), a, act)
        if TYPE_GREEDY in used_base_types:
            a = jax.vmap(greedy_action)(states).astype(jnp.int8)
            act = jnp.where(stype == jnp.int8(TYPE_GREEDY), a, act)
        if used_slots:
            obs = jax.vmap(observe)(states)                   # (N,175)
            pre_done = states.done
            for pos, slot in enumerate(used_slots):
                out = model.apply({'params': nn_params[pos]}, obs)
                lg = build_nn_logits(out, phase32, masks)
                # done 状态 mask 全 False：logits 置零保证 argmax 有定义（no-op）
                safe = jnp.where(pre_done[:, None],
                                 jnp.zeros(N_ACTIONS, jnp.float32), lg)
                a = jnp.argmax(safe, -1).astype(jnp.int8)
                act = jnp.where(stype == jnp.int8(TYPE_NN_BASE + slot), a, act)
            is_nn = stype >= jnp.int8(TYPE_NN_BASE)
            # AutoHu 风格强制（仅 NN 座位；规则座位的声明/报听行为不覆盖）
            if auto_hu:
                act = jnp.where(is_nn
                                & (states.phase == jnp.int8(env.PHASE_CLAIM))
                                & (states.claim_stage == jnp.int8(env.STAGE_HU)),
                                jnp.int8(env.A_HU), act)
            if no_tenpai:
                act = jnp.where(is_nn
                                & (states.phase == jnp.int8(env.PHASE_TENPAI)),
                                jnp.int8(env.A_TENPAI_NO), act)
        return jax.vmap(env.step)(states, act)[0]

    @jax.jit
    def block(states, seat_types, nn_params):
        def f(carry, _):
            return one_step(carry, seat_types, nn_params), None
        states, _ = jax.lax.scan(f, states, None, length=scan_steps)
        return states

    return block


# ---------------------------------------------------------------------------
# 配对 key：lane 2k / 2k+1 共用 PRNGKey(seed_offset + k)
# ---------------------------------------------------------------------------

def _pair_keys(n_seeds, seed_offset=0):
    """(2*n_seeds, 2) uint32：keys[2k] == keys[2k+1] == PRNGKey(seed_offset+k)。"""
    base = jnp.stack([jax.random.PRNGKey(seed_offset + k)
                      for k in range(n_seeds)])
    return jnp.repeat(base, 2, axis=0)


# ---------------------------------------------------------------------------
# 统计（公式逐行复制 scripts/rl/benchmark_duplicate.py）
# ---------------------------------------------------------------------------

def _paired_ci(a_wins, b_wins, n_pairs, z=1.96):
    """Paired difference (A - B) win-rate 95% CI（benchmark_duplicate.py 原样）。"""
    if n_pairs == 0:
        return 0.0, 0.0, 0.0
    diff = (a_wins - b_wins) / n_pairs
    var = (a_wins + b_wins) / n_pairs - diff ** 2
    var = max(var, 0.0)
    se = math.sqrt(var / n_pairs)
    lo = max(-1.0, diff - z * se)
    hi = min(1.0, diff + z * se)
    return diff, lo, hi


def _seat_score(r, cand):
    """推倒胡计分代理：自摸三家付 (+3)，点和一家付 (+1)，放炮 -1，其余 0（原样）。"""
    if r.get('winner') == cand:
        return 3.0 if r.get('win_type') == 'self' else 1.0
    if r.get('win_type') == 'ron' and r.get('dealer') == cand:
        return -1.0
    return 0.0


def candidate_wins(results, name, kind, positions=(0,)):
    """候选席位胜率计数（镜像 benchmark_duplicate.py::_candidate_wins，
    含 startswith 守卫；kind='a' 取 results[k]，'b' 取 results[k+1]）。"""
    wins = total = 0
    for k in range(0, len(results) - 1, 2):
        pos = positions[(k // 2) % len(positions)]
        r = results[k] if kind == 'a' else results[k + 1]
        cand = r['players_order'][pos]
        if not cand.startswith(name):
            continue
        total += 1
        if r.get('winner') == cand:
            wins += 1
    return wins, total


def candidate_rates(results, name, kind, positions=(0,)):
    """候选席位的 win/self/ron/deal-in/draw 率（只数候选坐的那一个席位）。"""
    n = {'games': 0, 'win': 0, 'self': 0, 'ron': 0, 'deal_in': 0, 'draw': 0}
    for k in range(0, len(results) - 1, 2):
        pos = positions[(k // 2) % len(positions)]
        r = results[k] if kind == 'a' else results[k + 1]
        cand = r['players_order'][pos]
        if not cand.startswith(name):
            continue
        n['games'] += 1
        wt = r.get('win_type')
        if r.get('winner') == cand:
            n['win'] += 1
            n['self' if wt == 'self' else 'ron'] += 1
        elif wt == 'draw':
            n['draw'] += 1
        if wt == 'ron' and r.get('dealer') == cand:
            n['deal_in'] += 1
    g = max(n['games'], 1)
    return {**n,
            'win_rate': n['win'] / g, 'self_rate': n['self'] / g,
            'ron_rate': n['ron'] / g, 'deal_in_rate': n['deal_in'] / g,
            'draw_rate': n['draw'] / g}


def paired_block(results, a_name, b_name, n_pairs, positions=(0,)):
    """配对统计块（镜像 benchmark_duplicate.py 的 paired 循环与 score-proxy）。"""
    a_wins = b_wins = ties = 0
    sd_sum = sd_sq = 0.0
    for i in range(0, len(results), 2):
        pos = positions[(i // 2) % len(positions)]
        candidate_a = f'{a_name}@{pos}_a'
        candidate_b = f'{b_name}@{pos}_b'
        a_won = results[i].get('winner') == candidate_a
        b_won = results[i + 1].get('winner') == candidate_b
        if a_won and not b_won:
            a_wins += 1
        elif b_won and not a_won:
            b_wins += 1
        else:
            ties += 1
        sd = (_seat_score(results[i], candidate_a)
              - _seat_score(results[i + 1], candidate_b))
        sd_sum += sd
        sd_sq += sd * sd
    diff, lo, hi = _paired_ci(a_wins, b_wins, n_pairs)
    mean = sd_sum / n_pairs
    var = max(sd_sq / n_pairs - mean ** 2, 0.0)
    se = math.sqrt(var / n_pairs)
    return {
        'n_pairs': n_pairs,
        'a_wins': a_wins,
        'b_wins': b_wins,
        'ties': ties,
        'diff': diff,
        'ci_lo': lo,
        'ci_hi': hi,
        'score_diff': mean,
        'score_ci_lo': mean - 1.96 * se,
        'score_ci_hi': mean + 1.96 * se,
    }


# ---------------------------------------------------------------------------
# 结果组装（schema 镜像 driver/engine.py::play_game 的返回 dict）
# ---------------------------------------------------------------------------

def build_results(winner, win_type, dealer, done,
                  a_name, b_name, opp_names):
    """逐 lane 终局数组 -> results list（A/B 交替，命名 f'{name}@{pos}_{kind}'）。

    'dealer' 键仅 ron 局出现（与 engine.play_game 一致）；未 done 的局按流局计。
    """
    results = []
    for lane in range(len(winner)):
        kind = 'a' if lane % 2 == 0 else 'b'
        seat0 = a_name if kind == 'a' else b_name
        players = [f'{seat0}@0_{kind}', f'{opp_names[0]}@1_{kind}',
                   f'{opp_names[1]}@2_{kind}', f'{opp_names[2]}@3_{kind}']
        w = int(winner[lane])
        wt = int(win_type[lane])
        dl = int(dealer[lane])
        if not bool(done[lane]) or w < 0:
            results.append({'winner': None, 'win_type': 'draw',
                            'players_order': players})
        elif wt == env.WIN_SELF:
            results.append({'winner': players[w], 'win_type': 'self',
                            'players_order': players})
        else:
            results.append({'winner': players[w], 'win_type': 'ron',
                            'dealer': players[dl], 'players_order': players})
    return results


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_duplicate(type_a, type_b, opp_types, n_seeds, seed_offset=0,
                  a_name='A', b_name='B', opp_names=('Opp1', 'Opp2', 'Opp3'),
                  model=None, nn_params=(), chunk_pairs=1024, max_steps=600,
                  scan_steps=32, auto_hu=True, no_tenpai=True, verbose=False):
    """跑 n_seeds 对复式对局（2*n_seeds 条 lane），返回原始终局 + 配对统计。

    type_a/type_b/opp_types: 座位类型码（TYPE_EVAL2/TYPE_BELIEF/TYPE_GREEDY/
        TYPE_NN_BASE+i）；座位 0 = 候选（A lane/B lane），座位 1-3 = opp_types。
    model/nn_params: NN 座位用；nn_params[i] 为槽 i 的 flax params（未用槽可 None）。
    chunk_pairs: 每批 vmap 的 pair 数（内存/编译权衡；尾批零头用重复首 lane
        补齐到整批再切掉，避免二次编译）。
    返回 dict:
        winner/win_type/dealer/done: (2N,) numpy 原始终局（座位号 / env 常量）；
        results: benchmark_duplicate.py 兼容的 result dict 列表（A/B 交替）；
        paired: paired_block 统计（键与 Python 端 pkl['paired'] 一致）；
        candidate: {'a': candidate_rates, 'b': candidate_rates}；
        n_seeds, wall_time。
    """
    t0 = time.time()
    if len(opp_types) != 3:
        raise ValueError('opp_types must contain exactly 3 types')
    if len(opp_names) != 3:
        raise ValueError('opp_names must contain exactly 3 names')
    all_types = (int(type_a), int(type_b)) + tuple(int(t) for t in opp_types)
    used_base = frozenset(t for t in all_types if t < TYPE_NN_BASE)
    used_slots = tuple(sorted({t - TYPE_NN_BASE for t in all_types
                               if t >= TYPE_NN_BASE}))
    nn_params = tuple(nn_params)
    if used_slots:
        if model is None:
            raise ValueError('NN seats present but model is None')
        if used_slots[-1] >= len(nn_params) or any(
                nn_params[i] is None for i in used_slots):
            raise ValueError(f'nn_params missing for used slots {used_slots}')
    nn_arg = tuple(nn_params[i] for i in used_slots)

    seat_types = np.empty((2 * n_seeds, 4), np.int8)
    seat_types[0::2, 0] = type_a
    seat_types[1::2, 0] = type_b
    seat_types[:, 1:] = np.asarray(list(opp_types), np.int8)
    keys = _pair_keys(n_seeds, seed_offset)                    # (2N, 2) jnp

    block = _make_block_runner(model, used_base, used_slots,
                               auto_hu, no_tenpai, scan_steps)

    out_w, out_wt, out_dl, out_done = [], [], [], []
    chunk_pairs = max(int(chunk_pairs), 1)
    n_chunks = (n_seeds + chunk_pairs - 1) // chunk_pairs
    for c in range(n_chunks):
        s0 = c * chunk_pairs
        s1 = min(n_seeds, s0 + chunk_pairs)
        lanes = 2 * (s1 - s0)
        k = keys[2 * s0:2 * s1]
        st = jnp.asarray(seat_types[2 * s0:2 * s1])
        pad = 2 * chunk_pairs - lanes
        if pad > 0:          # 尾批补齐到整批，避免第二种 batch 形状重新编译
            k = jnp.concatenate([k, jnp.repeat(k[:1], pad, axis=0)], axis=0)
            st = jnp.concatenate([st, jnp.repeat(st[:1], pad, axis=0)], axis=0)
        tc = time.time()
        states = jax.vmap(env.init)(k)
        steps = 0
        while steps < max_steps and not bool(jnp.all(states.done)):
            states = block(states, st, nn_arg)
            steps += scan_steps
        out_w.append(np.asarray(states.winner)[:lanes])
        out_wt.append(np.asarray(states.win_type)[:lanes])
        out_dl.append(np.asarray(states.dealer)[:lanes])
        out_done.append(np.asarray(states.done)[:lanes])
        if verbose:
            print(f'[dup-arena] chunk {c + 1}/{n_chunks}: {lanes} lanes, '
                  f'{steps} steps (cap {max_steps}), {time.time() - tc:.1f}s',
                  flush=True)

    winner = np.concatenate(out_w)
    win_type = np.concatenate(out_wt)
    dealer = np.concatenate(out_dl)
    done = np.concatenate(out_done)
    results = build_results(winner, win_type, dealer, done,
                            a_name, b_name, tuple(opp_names))
    paired = paired_block(results, a_name, b_name, n_seeds)
    cand = {'a': candidate_rates(results, a_name, 'a'),
            'b': candidate_rates(results, b_name, 'b')}
    return {
        'winner': winner, 'win_type': win_type, 'dealer': dealer, 'done': done,
        'results': results, 'paired': paired, 'candidate': cand,
        'n_seeds': n_seeds, 'wall_time': time.time() - t0,
    }
