# -*- coding: utf-8 -*-
"""PPO + KL 锚训练（Mahjax 配方），作用于 jaxenv 晋北麻将环境。

- 单 GPU（CUDA_VISIBLE_DEVICES 外部指定），N_ENVS 场游戏 vmap 并行，T 个 env 步/iter。
- 4 座位共享 policy（--init 加载 best msgpack）；冻结 ref 副本做 KL 锚。
- 动作头映射：DISCARD→policy(34) 作 actions 0-33 logits；CLAIM→response(4) 映射到
  34-37（pass/peng/gang/hu）；TENPAI→tenpai logit ± 映射到 38/39。其余动作 mask -1e9。
  logp/entropy/KL 均按对应 head 的 masked softmax 计算。
- 每步记录：obs（行动玩家）、action、logp、value、mask、phase、player、done；
  done 的游戏同一步内先记终局、再 init 新局替换。
- GAE：γ=1，λ=0.95。每个玩家的决策子序列单独算：终局 reward[player] 挂在该玩家
  本局最后一次决策上，中间步 reward=0；delta = r + V(s_next_own) - V(s)
  （s_next_own = 该玩家下一次决策的 value；终局后 bootstrap=0）。
  每 iter 末尾未完成的游戏整段丢弃（跨局重置截断），不进 loss。
- PPO：clip 0.2，K epochs，Adam lr 3e-4，grad clip 0.5，vf 0.5，ent 0.01，
  KL 锚 coef 0.2（policy/response/tenpai 各 head 的 masked KL(ref‖cur)，按样本
  激活 head 计入）。
- 每 --eval-every iters：当前 argmax vs ref argmax，1v3 与 3v1 各 --eval-games 局
  （vmapped），打印胜率差/流局率/点炮率，写 <out-dir>/metrics.jsonl。
- 每 --save-every iters：存 params 到 <out-dir>/iter{N}.msgpack。
- matmul 精度：默认（TF32），M4 已验证小模型吞吐不受影响。

示例（smoke）：
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 jaxenv/ppo.py \
        --iters 5 --n-envs 32 --t-steps 64 --eval-every 5 --eval-games 64

方向 1b（Gumbel-top-k 1-ply 搜索目标，见 jaxenv/search.py）：
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 jaxenv/ppo.py \
        --target-mode gumbel --search-k 8 --search-draws 2 --search-beta 8.0 \
        --iters 5 --n-envs 32 --t-steps 64 --eval-every 5 --eval-games 64
    gumbel 模式：rollout 时对每个 DISCARD 决策 vmap 计算 π' 存入轨迹；DISCARD
    样本的 PPO surrogate 替换为 CE(π')（CLAIM/TENPAI 仍用 surrogate），出牌动作
    默认从 π' 采样（--no-play-searched 关闭）；日志/metrics.jsonl 增加 agree
    字段（prior argmax == π' argmax 比例，kept DISCARD 决策上统计）。
    --search-ply2：top-2 候选改算 best-first 2-ply Q（jump 截断取下一状态的
    search_value，--search-k2 控制叶状态候选数）；--search-chunk 控制每次
    vmap 的 env 数（lax.map 分块，防 ply2 大 batch 爆显存）。

对手池（--opp-pool CUR,GEN1,BC,GREEDY,EVAL2,BELIEF，缺省 1,0,0,0 = 纯自对弈，逐位
行为与旧版逐 bit 一致；也接受旧版 4/5 元权重，末尾自动补 0）：
- 每局 init/reset 时按权重为每个座位独立采样类型：CUR=当前 params（学习对象），
  GEN1=--opp-gen1 冻结 msgpack，BC=--init 冻结副本（=KL 锚 ref），GREEDY=
  jaxenv/greedy.py 的 shanten 贪心，EVAL2=jaxenv/eval2jax.py 的 arena Baseline
  移植（algo.select 默认 eval2 度量），BELIEF=jaxenv/beliefjax.py 的
  BeliefExpectimaxAgent 移植（eval2 进攻 + tile_danger 防守 + margin 规则 +
  报听启发式）。reset（done 自动开新局）时同步重采样。
- rollout 每步对全 batch 算 current/gen1/BC 三个 net 的前向 + greedy/eval2/belief
  action，按 seat_types[env, actor] 选择本步动作（3× 前向成本；gen1/BC 权重为 0
  的类型在 trace 时静态跳过其前向；greedy/eval2/belief 同理）。gumbel 搜索目标
  只用 current net 计算（静态形状下仍 vmap 全 batch，但仅 current 座位的样本被
  keep 消费，见下）。
- 训练数据只保留 current 座位：GAE keep 掩码加 seat_types==CUR 条件；gumbel CE、
  agree 统计均随之只在 current 座位决策上生效。终局 reward 仍按座位照常分配。
- 日志：每 iter 打印各类型座位占比；eval 除 1v3/3v1（vs ref）外，池含 GEN1 时
  加当前 argmax vs gen1 argmax 1v3，含 GREEDY/EVAL2/BELIEF 时加当前 argmax vs
  greedy/eval2/belief 1v3。
"""

import argparse
import json
import os
import time
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from flax import serialization
import optax

from jaxenv import env as env_mod
from jaxenv.beliefjax import belief_action
from jaxenv.eval2jax import eval2_action
from jaxenv.greedy import greedy_action
from jaxenv.model_flax import build_model_flax
from jaxenv.obs import observe, actor_of, OBS_DIM
from jaxenv.search import improved_policy, improved_policy_ply2

NEG = -1e9
N_ACTIONS = env_mod.N_ACTIONS

# 对手池座位类型
OPP_CUR, OPP_GEN1, OPP_BC, OPP_GREEDY, OPP_EVAL2, OPP_BELIEF = 0, 1, 2, 3, 4, 5
OPP_NAMES = ('CUR', 'GEN1', 'BC', 'GREEDY', 'EVAL2', 'BELIEF')


# ---------------------------------------------------------------------------
# 动作头映射
# ---------------------------------------------------------------------------

def build_logits(out, phase, mask):
    """heads 输出 + phase(B,) + 合法 mask(B,40) -> (B,40) masked logits。"""
    B = out['policy'].shape[0]
    base = jnp.full((B, N_ACTIONS), NEG, jnp.float32)
    pol = base.at[:, :34].set(out['policy'])
    clm = base.at[:, 34:38].set(out['response'])
    t = out['tenpai'][:, 0]
    ten = base.at[:, 38].set(t).at[:, 39].set(-t)
    logits = jnp.where((phase == jnp.int8(env_mod.PHASE_DISCARD))[:, None], pol,
             jnp.where((phase == jnp.int8(env_mod.PHASE_CLAIM))[:, None], clm, ten))
    return jnp.where(mask, logits, NEG)


# ---------------------------------------------------------------------------
# rollout（scan 版本：一次 jit 调用跑 T 步）
# ---------------------------------------------------------------------------

def _pi_prime_batched(params, model, states, skeys, search_cfg):
    """(N,) 状态 -> π' (N,34)。

    1-ply（无 ply2 键）与原 improved_policy vmap 完全一致。ply2 模式用
    improved_policy_ply2，并按 search_cfg['chunk'] 分块 lax.map：单次 vmap 的
    网络前向行数 ∝ chunk × 候选×子状态数，防大 N 下峰值显存过高；N 不能整除
    chunk（或 chunk<=0）时退回整批 vmap。
    """
    if not search_cfg.get('ply2'):
        pi, _, _ = jax.vmap(lambda st, kk: improved_policy(
            params, model, st, kk, search_cfg['k'], search_cfg['draws'],
            search_cfg['beta']))(states, skeys)
        return pi

    def one(st, kk):
        return improved_policy_ply2(params, model, st, kk, search_cfg['k'],
                                    search_cfg['draws'], search_cfg['beta'],
                                    search_cfg['k2'],
                                    n_top2=search_cfg.get('top2', 2))[0]
    n = states.done.shape[0]
    chunk = search_cfg.get('chunk') or 0
    if chunk <= 0 or n % chunk != 0:
        return jax.vmap(one)(states, skeys)
    c = n // chunk
    sr = jax.tree.map(lambda x: x.reshape((c, chunk) + x.shape[1:]), states)
    kr = skeys.reshape(c, chunk, 2)
    pi = jax.lax.map(lambda xs: jax.vmap(one)(*xs), (sr, kr))     # (c, chunk, 34)
    return pi.reshape(n, 34)


def make_rollout_fn(model, t_steps, search_cfg=None, play_searched=False,
                    pool_w=None, reward_kind=env_mod.REWARD_SCORE,
                    auto_hu=False, no_tenpai=False):
    """返回 jitted (params, opp_params, states, seat_types, rng)
        -> (states', seat_types', rng', batch) 函数。

    search_cfg: None（outcome 模式）或 dict(k=, draws=, beta=)（gumbel 模式，
    可选 ply2=/k2=/chunk= 键启用 best-first 2-ply，见 _pi_prime_batched）。
    gumbel 模式下 batch 额外含 (pi_prime(T,N,34), prior_arg(T,N), pi_arg(T,N))；
    play_searched=True 时 current 座位的 DISCARD 决策改从 π' 采样（AZ 闭环）。
    pool_w: None（纯自对弈，行为与旧版逐 bit 一致）或 (6,) 座位类型权重；
    非 None 时 batch 额外含 seat_types(T,N,4)，reset 的局按权重重采样座位类型。
    opp_params: (gen1_params, bc_params)；权重为 0 的类型传 None（静态跳过前向）。
    reward_kind: 环境奖励类型（score / winloss / score_dd），贯穿 init/reset。
    auto_hu: CLAIM-hu 阶段强制 action=hu（能胡必胡，删 hu 决策维度）。
    no_tenpai: TENPAI 阶段强制 action=no（报听恒否，删报听决策维度）。
    两者仅改变行为动作；batch 末尾附 cstage 供训练侧剔除这些强制决策样本。
    """
    pool_active = pool_w is not None
    use_gen1 = pool_active and pool_w[OPP_GEN1] > 0
    use_bc = pool_active and pool_w[OPP_BC] > 0
    use_greedy = pool_active and pool_w[OPP_GREEDY] > 0
    use_eval2 = pool_active and pool_w[OPP_EVAL2] > 0
    use_belief = pool_active and pool_w[OPP_BELIEF] > 0
    if pool_active:
        pool_logits = jnp.log(jnp.clip(jnp.asarray(pool_w, jnp.float32), 1e-9, 1.0))

    def body(carry, _):
        params, opp_params, states, seat_types, rng = carry
        if pool_active:
            rng, k_act, k_act2, k_reset, k_search, k_opp, k_seat = \
                jax.random.split(rng, 7)
        else:
            rng, k_act, k_act2, k_reset, k_search = jax.random.split(rng, 5)

        obs = jax.vmap(observe)(states)                       # (N,175)
        out = model.apply({'params': params}, obs)
        masks = jax.vmap(env_mod.legal_mask)(states)          # (N,40)
        phase = states.phase
        player = jax.vmap(actor_of)(states)                   # (N,)
        pre_done = states.done
        phase32 = phase.astype(jnp.int32)
        logits = build_logits(out, phase32, masks)
        # done 状态（天胡残留）mask 全 False：logits 置零保证采样有定义（no-op）
        safe = jnp.where(pre_done[:, None], jnp.zeros(N_ACTIONS, jnp.float32), logits)
        act = jax.random.categorical(k_act, safe, axis=-1).astype(jnp.int8)

        extra = ()
        if search_cfg is not None:
            skeys = jax.random.split(k_search, states.done.shape[0])
            pi_prime = _pi_prime_batched(params, model, states, skeys,
                                         search_cfg)              # (N,34)
            if play_searched:
                log_pi = jnp.where(pi_prime > 0, jnp.log(pi_prime), NEG)
                safe_pi = jnp.where(pre_done[:, None],
                                    jnp.zeros(34, jnp.float32), log_pi)
                act_s = jax.random.categorical(k_act2, safe_pi, axis=-1).astype(jnp.int8)
                act = jnp.where(phase == jnp.int8(env_mod.PHASE_DISCARD), act_s, act)
            prior_arg = jnp.argmax(jnp.where(masks[:, :34], out['policy'], NEG),
                                   -1).astype(jnp.int8)
            pi_arg = jnp.argmax(pi_prime, -1).astype(jnp.int8)
            extra = (pi_prime, prior_arg, pi_arg)

        # 对手池：非 current 座位按其类型覆盖动作（current 座位保持上面的 act）
        if pool_active:
            stype = seat_types[jnp.arange(states.done.shape[0]), player]   # (N,)
            k_og, k_ob = jax.random.split(k_opp)
            if use_gen1:
                out_g = model.apply({'params': opp_params[0]}, obs)
                lg = build_logits(out_g, phase32, masks)
                sg = jnp.where(pre_done[:, None],
                               jnp.zeros(N_ACTIONS, jnp.float32), lg)
                ag = jax.random.categorical(k_og, sg, axis=-1).astype(jnp.int8)
                act = jnp.where(stype == jnp.int8(OPP_GEN1), ag, act)
            if use_bc:
                out_b = model.apply({'params': opp_params[1]}, obs)
                lb = build_logits(out_b, phase32, masks)
                sb = jnp.where(pre_done[:, None],
                               jnp.zeros(N_ACTIONS, jnp.float32), lb)
                ab = jax.random.categorical(k_ob, sb, axis=-1).astype(jnp.int8)
                act = jnp.where(stype == jnp.int8(OPP_BC), ab, act)
            if use_greedy:
                agr = jax.vmap(greedy_action)(states)
                act = jnp.where(stype == jnp.int8(OPP_GREEDY), agr, act)
            if use_eval2:
                ae = jax.vmap(eval2_action)(states)
                act = jnp.where(stype == jnp.int8(OPP_EVAL2), ae, act)
            if use_belief:
                ab = jax.vmap(belief_action)(states)
                act = jnp.where(stype == jnp.int8(OPP_BELIEF), ab, act)

        # from-scratch 简化：hu 自动（能胡必胡）、报听恒否——对全部座位生效
        if auto_hu:
            act = jnp.where((phase == jnp.int8(env_mod.PHASE_CLAIM)) &
                            (states.claim_stage == jnp.int8(env_mod.STAGE_HU)),
                            jnp.int8(env_mod.A_HU), act)
        if no_tenpai:
            act = jnp.where(phase == jnp.int8(env_mod.PHASE_TENPAI),
                            jnp.int8(env_mod.A_TENPAI_NO), act)

        logp_all = jax.nn.log_softmax(safe, axis=-1)
        logp = jnp.take_along_axis(logp_all, act[:, None].astype(jnp.int32), -1)[:, 0]
        val = out['value'][:, 0]

        new_states, rew, done = jax.vmap(env_mod.step)(states, act)

        # 同一步内自动重置：先记终局（上方 batch 输出），再用新局替换
        keys = jax.random.split(k_reset, states.done.shape[0])
        fresh = jax.vmap(lambda k: env_mod.init(k, reward_kind))(keys)

        def pick(f, s):
            return jnp.where(done.reshape(-1, *([1] * (f.ndim - 1))), f, s)
        states2 = jax.tree.map(pick, fresh, new_states)
        if pool_active:
            # reset 的局按池权重为 4 个座位重采样类型
            new_st = jax.random.categorical(
                k_seat, jnp.broadcast_to(pool_logits,
                                         (*seat_types.shape, len(OPP_NAMES))),
                axis=-1).astype(jnp.int8)
            seat_types2 = jnp.where(done[:, None], new_st, seat_types)
        else:
            seat_types2 = seat_types

        batch = (obs, act, logp, val, masks, phase, player, pre_done, rew, done,
                 seat_types, states.claim_stage) + extra
        return (params, opp_params, states2, seat_types2, rng), batch

    @jax.jit
    def rollout(params, opp_params, states, seat_types, rng):
        (params, opp_params, states, seat_types, rng), batch = jax.lax.scan(
            body, (params, opp_params, states, seat_types, rng), None, length=t_steps)
        return states, seat_types, rng, batch

    return rollout


# ---------------------------------------------------------------------------
# GAE（host 侧，按玩家子序列）
# ---------------------------------------------------------------------------

def compute_gae(players, values, rewards, dones, pre_done, seat_types=None,
                lam=0.95):
    """players/values/dones/pre_done: (T,N) host 数组；rewards: (T,N,4)。

    γ=1。返回 adv/ret/keep (T,N)。每 env 最后一段未完成游戏整段 keep=False。
    pre_done（天胡 no-op 步）的决策不进任何子序列。
    seat_types: None（纯自对弈）或 (T,N,4)；非 None 时只有 seat_types==OPP_CUR
    的座位产生训练样本（对手池模式下非 current 座位不进 GAE）。
    """
    T, N = players.shape
    adv = np.zeros((T, N), np.float32)
    ret = np.zeros((T, N), np.float32)
    keep = np.zeros((T, N), bool)
    for e in range(N):
        t0 = 0
        for t1 in range(T):
            if not dones[t1, e]:
                continue
            final_rew = rewards[t1, e]                       # (4,)
            for p in range(4):
                if seat_types is not None and seat_types[t1, e, p] != OPP_CUR:
                    continue                                # 非 current 座位不产样本
                idxs = [t for t in range(t0, t1 + 1)
                        if players[t, e] == p and not pre_done[t, e]]
                if not idxs:
                    continue
                A = 0.0
                for i in range(len(idxs) - 1, -1, -1):
                    t = idxs[i]
                    if i == len(idxs) - 1:
                        delta = float(final_rew[p]) - values[t, e]   # bootstrap 0
                        A = delta
                    else:
                        delta = values[idxs[i + 1], e] - values[t, e]  # r=0, γ=1
                        A = delta + lam * A
                    adv[t, e] = A
                    ret[t, e] = A + values[t, e]
                    keep[t, e] = True
            t0 = t1 + 1
        # [t0..T-1] 未完成尾段：keep 保持 False（丢弃）
    return adv, ret, keep


# ---------------------------------------------------------------------------
# PPO 更新
# ---------------------------------------------------------------------------

def make_train_step(model, tx, clip, vf_coef, ent_coef, kl_coef, target_mode='outcome'):
    """target_mode='outcome'：全部样本 PPO clipped surrogate（原行为）。
    target_mode='gumbel'：DISCARD 样本改为 CE(π')（batch['pi']，搜索改进目标），
    CLAIM/TENPAI 样本保留 PPO surrogate（response/tenpai 头继续吃 outcome 信号）；
    value MSE / KL 锚 / entropy 不变。
    """
    @jax.jit
    def train_step(params, opt_state, ref_params, batch):
        def loss_fn(p):
            out = model.apply({'params': p}, batch['obs'])
            logits = build_logits(out, batch['phase'], batch['mask'])
            logp_all = jax.nn.log_softmax(logits, -1)
            logp = jnp.take_along_axis(logp_all, batch['act'][:, None].astype(jnp.int32), -1)[:, 0]
            ratio = jnp.exp(logp - batch['logp_old'])
            adv = batch['adv']
            pg_per = -jnp.minimum(ratio * adv,
                                  jnp.clip(ratio, 1.0 - clip, 1.0 + clip) * adv)
            aux = {}
            if target_mode == 'gumbel':
                is_disc = (batch['phase'] == jnp.int8(env_mod.PHASE_DISCARD)).astype(jnp.float32)
                ce_per = -(batch['pi'] * logp_all[:, :34]).sum(-1)
                ce = (ce_per * is_disc).sum() / jnp.maximum(is_disc.sum(), 1.0)
                nond = 1.0 - is_disc
                pg = (pg_per * nond).sum() / jnp.maximum(nond.sum(), 1.0)
                aux['ce'] = ce
                pg = pg + ce
            else:
                pg = pg_per.mean()
            v = ((out['value'][:, 0] - batch['ret']) ** 2).mean()
            prob = jax.nn.softmax(logits, -1)
            ent = -(prob * logp_all).sum(-1).mean()
            out_ref = model.apply({'params': ref_params}, batch['obs'])
            logits_ref = build_logits(out_ref, batch['phase'], batch['mask'])
            logp_ref = jax.nn.log_softmax(logits_ref, -1)
            prob_ref = jax.nn.softmax(logits_ref, -1)
            kl = (prob_ref * (logp_ref - logp_all)).sum(-1).mean()
            total = pg + vf_coef * v - ent_coef * ent + kl_coef * kl
            return total, {'pg': pg, 'v': v, 'ent': ent, 'kl': kl, **aux}

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    return train_step


# ---------------------------------------------------------------------------
# eval：当前 argmax vs ref argmax（1v3 / 3v1，vmapped）
# ---------------------------------------------------------------------------

def make_eval_step(model):
    @jax.jit
    def eval_step(cur_params, ref_params, states, cur_seat):
        obs = jax.vmap(observe)(states)
        masks = jax.vmap(env_mod.legal_mask)(states)
        phase = states.phase.astype(jnp.int32)
        out_c = model.apply({'params': cur_params}, obs)
        out_r = model.apply({'params': ref_params}, obs)
        lc = build_logits(out_c, phase, masks)
        lr = build_logits(out_r, phase, masks)
        actor = jax.vmap(actor_of)(states)
        use_cur = cur_seat[actor]                             # (N,) bool
        logits = jnp.where(use_cur[:, None], lc, lr)
        safe = jnp.where(states.done[:, None], jnp.zeros(N_ACTIONS, jnp.float32), logits)
        act = jnp.argmax(safe, -1).astype(jnp.int8)
        return jax.vmap(env_mod.step)(states, act)[0]

    return eval_step


def make_eval_step_greedy(model):
    """当前 argmax vs greedy_action 对手的 eval step。"""
    @jax.jit
    def eval_step(cur_params, states, cur_seat):
        obs = jax.vmap(observe)(states)
        masks = jax.vmap(env_mod.legal_mask)(states)
        phase = states.phase.astype(jnp.int32)
        out_c = model.apply({'params': cur_params}, obs)
        lc = build_logits(out_c, phase, masks)
        actor = jax.vmap(actor_of)(states)
        use_cur = cur_seat[actor]                             # (N,) bool
        safe = jnp.where(states.done[:, None], jnp.zeros(N_ACTIONS, jnp.float32), lc)
        act_c = jnp.argmax(safe, -1).astype(jnp.int8)
        act_g = jax.vmap(greedy_action)(states)
        act = jnp.where(use_cur, act_c, act_g)
        return jax.vmap(env_mod.step)(states, act)[0]

    return eval_step


def make_eval_step_eval2(model):
    """当前 argmax vs eval2_action（arena Baseline 移植）对手的 eval step。"""
    @jax.jit
    def eval_step(cur_params, states, cur_seat):
        obs = jax.vmap(observe)(states)
        masks = jax.vmap(env_mod.legal_mask)(states)
        phase = states.phase.astype(jnp.int32)
        out_c = model.apply({'params': cur_params}, obs)
        lc = build_logits(out_c, phase, masks)
        actor = jax.vmap(actor_of)(states)
        use_cur = cur_seat[actor]                             # (N,) bool
        safe = jnp.where(states.done[:, None], jnp.zeros(N_ACTIONS, jnp.float32), lc)
        act_c = jnp.argmax(safe, -1).astype(jnp.int8)
        act_e = jax.vmap(eval2_action)(states)
        act = jnp.where(use_cur, act_c, act_e)
        return jax.vmap(env_mod.step)(states, act)[0]

    return eval_step


def make_eval_step_belief(model):
    """当前 argmax vs belief_action（BeliefExpectimax 移植）对手的 eval step。"""
    @jax.jit
    def eval_step(cur_params, states, cur_seat):
        obs = jax.vmap(observe)(states)
        masks = jax.vmap(env_mod.legal_mask)(states)
        phase = states.phase.astype(jnp.int32)
        out_c = model.apply({'params': cur_params}, obs)
        lc = build_logits(out_c, phase, masks)
        actor = jax.vmap(actor_of)(states)
        use_cur = cur_seat[actor]                             # (N,) bool
        safe = jnp.where(states.done[:, None], jnp.zeros(N_ACTIONS, jnp.float32), lc)
        act_c = jnp.argmax(safe, -1).astype(jnp.int8)
        act_b = jax.vmap(belief_action)(states)
        act = jnp.where(use_cur, act_c, act_b)
        return jax.vmap(env_mod.step)(states, act)[0]

    return eval_step


def play_eval_generic(step_fn, n_games, cur_seat, rng, max_steps=600):
    """打 n_games 局（vmapped），step_fn(states, seat) -> states。返回指标 dict。"""
    keys = jax.random.split(rng, n_games)
    states = jax.vmap(env_mod.init)(keys)
    seat = jnp.asarray(cur_seat, dtype=bool)
    for _ in range(max_steps):
        if bool(jnp.all(states.done)):
            break
        states = step_fn(states, seat)
    winner = np.asarray(states.winner)
    win_type = np.asarray(states.win_type)
    dealer = np.asarray(states.dealer)
    done = np.asarray(states.done)
    seat_np = np.asarray(cur_seat)
    cur_win = float(np.mean((winner >= 0) & seat_np[np.clip(winner, 0, 3)] & done))
    ref_win = float(np.mean((winner >= 0) & ~seat_np[np.clip(winner, 0, 3)] & done))
    draw = float(np.mean((winner < 0) | ~done))
    ron = (win_type == env_mod.WIN_RON) & done
    cur_deal = float(np.mean(ron & seat_np[np.clip(dealer, 0, 3)]))
    ref_deal = float(np.mean(ron & ~seat_np[np.clip(dealer, 0, 3)]))
    return {'cur_win': cur_win, 'ref_win': ref_win,
            'win_diff': cur_win - ref_win, 'draw': draw,
            'cur_dealin': cur_deal, 'ref_dealin': ref_deal}


def play_eval(eval_step, cur_params, ref_params, n_games, cur_seat, rng,
              max_steps=600):
    """当前 argmax vs ref argmax。cur_seat: (4,) bool。"""
    return play_eval_generic(
        lambda s, seat: eval_step(cur_params, ref_params, s, seat),
        n_games, cur_seat, rng, max_steps)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def load_params(path):
    with open(path, 'rb') as f:
        variables = serialization.from_bytes(None, f.read())
    return variables['params']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--n-envs', type=int, default=512)
    ap.add_argument('--t-steps', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--kl-coef', type=float, default=0.2)
    ap.add_argument('--ent-coef', type=float, default=0.01)
    ap.add_argument('--vf-coef', type=float, default=0.5)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--gae-lambda', type=float, default=0.95)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--minibatch-size', type=int, default=8192)
    ap.add_argument('--max-grad-norm', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--init', default='output/nn_full_action_best_flax.msgpack')
    ap.add_argument('--config', default='output/nn_full_action_best_config.json')
    ap.add_argument('--out-dir', default='output/jax_ppo')
    ap.add_argument('--eval-every', type=int, default=25)
    ap.add_argument('--eval-games', type=int, default=128)
    ap.add_argument('--save-every', type=int, default=25)
    # 方向 1b：Gumbel-top-k 1-ply 搜索目标
    ap.add_argument('--target-mode', choices=['outcome', 'gumbel'], default='outcome')
    ap.add_argument('--search-k', type=int, default=8)
    ap.add_argument('--search-draws', type=int, default=2)
    ap.add_argument('--search-beta', type=float, default=8.0)
    ap.add_argument('--search-ply2', action='store_true',
                    help='gumbel 模式下 top-2 候选改算 best-first 2-ply Q'
                         '（jump 截断取下一状态的 search_value）')
    ap.add_argument('--search-k2', type=int, default=4,
                    help='2-ply 叶状态 search_value 的 top-k2 候选数（--search-ply2 时有效）')
    ap.add_argument('--search-top2', type=int, default=2,
                    help='2-ply 展开的根候选数 n_top2（--search-ply2 时有效；1 可约 2× 提速）')
    ap.add_argument('--search-chunk', type=int, default=64,
                    help='ply2 计算 π\' 时每次 vmap 的 env 数（lax.map 分块；0=整批）')
    ap.add_argument('--no-play-searched', action='store_true',
                    help='gumbel 模式下仍用 prior 采样出牌（消融用；默认从 π\' 采样）')
    ap.add_argument('--reward-kind', default=env_mod.REWARD_SCORE,
                    choices=[env_mod.REWARD_SCORE, env_mod.REWARD_WINLOSS,
                             env_mod.REWARD_DD],
                    help='环境奖励类型；score_dd = score + 流局全员 -0.25（from-scratch 用）')
    ap.add_argument('--anchor-refresh', type=int, default=0,
                    help='每 N iters 把 KL 锚 ref_params 刷新为当前 params（NPG 移动锚；0=不换）')
    ap.add_argument('--auto-hu', action='store_true',
                    help='CLAIM-hu 阶段强制能胡必胡（删 hu 决策维度；样本从训练剔除）')
    ap.add_argument('--no-tenpai', action='store_true',
                    help='TENPAI 阶段强制报听恒否（删报听决策维度；样本从训练剔除）')
    ap.add_argument('--eval-greedy', action='store_true',
                    help='eval 额外跑 vs shanten-greedy（from-scratch 里程碑指标）')
    # 对手池：CUR,GEN1,BC,GREEDY,EVAL2,BELIEF 六个 0-1 权重（和为 1；缺省纯自对弈；
    # 兼容旧版 4/5 元写法，末尾自动补 0）
    ap.add_argument('--opp-pool', default='1,0,0,0',
                    help='座位类型权重 CUR,GEN1,BC,GREEDY,EVAL2,BELIEF，'
                         '如 0.5,0.1,0.1,0.1,0.1,0.1（旧版 4/5 元写法末尾补 0）')
    ap.add_argument('--opp-gen1', default='output/jax_gumbel_pilot/iter92.msgpack',
                    help='GEN1 对手的冻结 msgpack')
    args = ap.parse_args()

    pool_w = np.array([float(x) for x in args.opp_pool.split(',')], np.float64)
    if pool_w.shape in ((4,), (5,)):                # 向后兼容旧版 4/5 元权重
        pool_w = np.concatenate([pool_w, np.zeros(len(OPP_NAMES) - pool_w.shape[0])])
    assert pool_w.shape == (len(OPP_NAMES),) and (pool_w >= 0).all(), \
        f'--opp-pool 需为 4-6 个非负权重: {args.opp_pool}'
    assert abs(pool_w.sum() - 1.0) < 1e-6, \
        f'--opp-pool 权重和须为 1: {pool_w.sum()}'
    pool_active = bool((pool_w[1:] > 0).any())

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.config) as f:
        config = json.load(f)
    # config 副本随产物落盘（convert_back 需要）
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    model = build_model_flax(config)
    params = load_params(args.init)
    ref_params = params                      # 冻结 KL 锚（params 为不可变 pytree）

    # 对手池参数：GEN1 从 --opp-gen1 加载；BC = --init 的冻结副本（=ref_params）
    gen1_params = load_params(args.opp_gen1) \
        if pool_active and pool_w[OPP_GEN1] > 0 else None
    bc_params = ref_params if pool_active and pool_w[OPP_BC] > 0 else None
    opp_params = (gen1_params, bc_params)

    tx = optax.chain(optax.clip_by_global_norm(args.max_grad_norm),
                     optax.adam(args.lr))
    opt_state = tx.init(params)

    search_cfg = None
    play_searched = False
    if args.target_mode == 'gumbel':
        search_cfg = {'k': args.search_k, 'draws': args.search_draws,
                      'beta': args.search_beta, 'ply2': args.search_ply2,
                      'k2': args.search_k2, 'top2': args.search_top2,
                      'chunk': args.search_chunk}
        play_searched = not args.no_play_searched

    rollout = make_rollout_fn(model, args.t_steps, search_cfg, play_searched,
                              pool_w if pool_active else None,
                              reward_kind=args.reward_kind,
                              auto_hu=args.auto_hu, no_tenpai=args.no_tenpai)
    train_step = make_train_step(model, tx, args.clip, args.vf_coef,
                                 args.ent_coef, args.kl_coef, args.target_mode)
    eval_step = make_eval_step(model)
    eval_step_greedy = make_eval_step_greedy(model) \
        if (pool_active and pool_w[OPP_GREEDY] > 0) or args.eval_greedy else None
    eval_step_eval2 = make_eval_step_eval2(model) \
        if pool_active and pool_w[OPP_EVAL2] > 0 else None
    eval_step_belief = make_eval_step_belief(model) \
        if pool_active and pool_w[OPP_BELIEF] > 0 else None

    rng = jax.random.PRNGKey(args.seed)
    rng, k_init, k_eval = jax.random.split(rng, 3)
    states = jax.vmap(lambda k: env_mod.init(k, args.reward_kind))(
        jax.random.split(k_init, args.n_envs))
    if pool_active:
        rng, k_st = jax.random.split(rng)
        pl = jnp.log(jnp.clip(jnp.asarray(pool_w, jnp.float32), 1e-9, 1.0))
        seat_types = jax.random.categorical(
            k_st, jnp.broadcast_to(pl, (args.n_envs, 4, len(OPP_NAMES))),
            axis=-1).astype(jnp.int8)
    else:
        seat_types = jnp.zeros((args.n_envs, 4), jnp.int8)

    metrics_path = os.path.join(args.out_dir, 'metrics.jsonl')
    mlog = open(metrics_path, 'a', buffering=1)
    print(f'[ppo] init={args.init} n_envs={args.n_envs} T={args.t_steps} '
          f'iters={args.iters} target={args.target_mode}'
          + (f' (k={args.search_k} draws={args.search_draws} '
             f'beta={args.search_beta} play_searched={play_searched}'
             + (f' ply2(k2={args.search_k2} chunk={args.search_chunk})'
                if args.search_ply2 else '') + ')'
             if search_cfg else '')
          + (f' opp_pool={args.opp_pool}'
             + (f' gen1={args.opp_gen1}' if gen1_params is not None else '')
             if pool_active else '')
          + f' -> {args.out_dir}', flush=True)

    total_decisions = 0
    total_time = 0.0
    for it in range(1, args.iters + 1):
        t0 = time.time()
        states, seat_types, rng, batch = rollout(
            params, opp_params, states, seat_types, rng)
        if search_cfg is not None:
            (obs_b, act_b, logp_b, val_b, mask_b, phase_b, player_b, predone_b,
             rew_b, done_b, st_b, cstage_b, pi_b, parg_b, piarg_b) = \
                [np.asarray(x) for x in batch]
        else:
            (obs_b, act_b, logp_b, val_b, mask_b, phase_b, player_b, predone_b,
             rew_b, done_b, st_b, cstage_b) = [np.asarray(x) for x in batch]
        total_decisions += args.n_envs * args.t_steps

        adv, ret, keep = compute_gae(player_b, val_b, rew_b, done_b, predone_b,
                                     seat_types=st_b if pool_active else None,
                                     lam=args.gae_lambda)
        # 强制决策（auto-hu / no-tenpai）是环境行为而非策略选择，样本从训练剔除
        if args.auto_hu:
            keep &= ~((phase_b == env_mod.PHASE_CLAIM) &
                      (cstage_b == env_mod.STAGE_HU))
        if args.no_tenpai:
            keep &= ~(phase_b == env_mod.PHASE_TENPAI)
        # G1 门指标：prior argmax == π' argmax 的比例（kept DISCARD 决策）
        agree = None
        if search_cfg is not None:
            m_ag = keep & (phase_b == env_mod.PHASE_DISCARD) & ~predone_b
            if m_ag.any():
                agree = float((parg_b == piarg_b)[m_ag].mean())
        idx = np.where(keep.ravel())[0]
        if len(idx) == 0:
            print(f'[ppo] iter {it}: no kept samples, skip update', flush=True)
            continue
        flat = lambda a: a.reshape(-1, *a.shape[2:])[idx]
        data = {
            'obs': jnp.asarray(flat(obs_b)),
            'act': jnp.asarray(flat(act_b)),
            'logp_old': jnp.asarray(flat(logp_b)),
            'mask': jnp.asarray(flat(mask_b)),
            'phase': jnp.asarray(flat(phase_b)),
            'adv': flat(adv).astype(np.float32),
            'ret': jnp.asarray(flat(ret)),
        }
        if search_cfg is not None:
            data['pi'] = jnp.asarray(flat(pi_b))
        data['adv'] = (data['adv'] - data['adv'].mean()) / (data['adv'].std() + 1e-8)
        data['adv'] = jnp.asarray(data['adv'])
        M = len(idx)

        rng_np = np.random.default_rng(args.seed + it)
        agg = {}
        for _ in range(args.epochs):
            perm = rng_np.permutation(M)
            for s in range(0, M, args.minibatch_size):
                mb_idx = perm[s:s + args.minibatch_size]
                mb = {k: v[mb_idx] for k, v in data.items()}
                params, opt_state, loss, aux = train_step(
                    params, opt_state, ref_params, mb)
                for kk, vv in [('loss', loss)] + list(aux.items()):
                    agg.setdefault(kk, []).append(float(vv))
        dt = time.time() - t0
        total_time += dt
        dps = args.n_envs * args.t_steps / dt
        msg = {k: float(np.mean(v)) for k, v in agg.items()}
        line = (f'[ppo] iter {it}/{args.iters} kept={M}/{keep.size} '
                + ' '.join(f'{k}={v:+.4f}' for k, v in sorted(msg.items())))
        if agree is not None:
            line += f' agree={agree:.4f}'
        seat_frac = None
        if pool_active:
            seat_frac = {OPP_NAMES[t]: float((st_b == t).mean())
                         for t in range(len(OPP_NAMES))
                         if pool_w[t] > 0}
            line += ' pool={' + ', '.join(f'{k}:{v:.2f}' for k, v in
                                          seat_frac.items()) + '}'
        line += f' time={dt:.1f}s ({dps:.0f} dec/s)'
        print(line, flush=True)
        rec = {'type': 'train', 'iter': it, 'kept': int(M), **msg, 'time': dt}
        if agree is not None:
            rec['agree'] = agree
        if seat_frac is not None:
            rec['pool'] = seat_frac
        mlog.write(json.dumps(rec) + '\n')

        if it % args.eval_every == 0:
            n_extra = (int(gen1_params is not None)
                       + int(eval_step_greedy is not None)
                       + int(eval_step_eval2 is not None)
                       + int(eval_step_belief is not None))
            ks = jax.random.split(k_eval, 3 + n_extra)
            k_eval = ks[0]
            k1, k3 = ks[1], ks[2]
            m1 = play_eval(eval_step, params, ref_params, args.eval_games,
                           np.array([True, False, False, False]), k1)
            m3 = play_eval(eval_step, params, ref_params, args.eval_games,
                           np.array([True, True, True, False]), k3)
            print(f'[eval] iter {it} 1v3: win_diff={m1["win_diff"]:+.3f} '
                  f'(cur {m1["cur_win"]:.3f} vs ref {m1["ref_win"]:.3f}) '
                  f'draw={m1["draw"]:.3f} dealin(cur/ref)='
                  f'{m1["cur_dealin"]:.3f}/{m1["ref_dealin"]:.3f}', flush=True)
            print(f'[eval] iter {it} 3v1: win_diff={m3["win_diff"]:+.3f} '
                  f'(cur {m3["cur_win"]:.3f} vs ref {m3["ref_win"]:.3f}) '
                  f'draw={m3["draw"]:.3f} dealin(cur/ref)='
                  f'{m3["cur_dealin"]:.3f}/{m3["ref_dealin"]:.3f}', flush=True)
            ev_rec = {'type': 'eval', 'iter': it, '1v3': m1, '3v1': m3}
            ki = 3
            if gen1_params is not None:
                mg = play_eval(eval_step, params, gen1_params, args.eval_games,
                               np.array([True, False, False, False]), ks[ki])
                ki += 1
                print(f'[eval] iter {it} vs_gen1 1v3: win_diff='
                      f'{mg["win_diff"]:+.3f} (cur {mg["cur_win"]:.3f} vs gen1 '
                      f'{mg["ref_win"]:.3f}) draw={mg["draw"]:.3f} '
                      f'dealin(cur/gen1)={mg["cur_dealin"]:.3f}/'
                      f'{mg["ref_dealin"]:.3f}', flush=True)
                ev_rec['vs_gen1'] = mg
            if eval_step_greedy is not None:
                mgr = play_eval_generic(
                    lambda s, seat: eval_step_greedy(params, s, seat),
                    args.eval_games, np.array([True, False, False, False]), ks[ki])
                print(f'[eval] iter {it} vs_greedy 1v3: win_diff='
                      f'{mgr["win_diff"]:+.3f} (cur {mgr["cur_win"]:.3f} vs greedy '
                      f'{mgr["ref_win"]:.3f}) draw={mgr["draw"]:.3f} '
                      f'dealin(cur/greedy)={mgr["cur_dealin"]:.3f}/'
                      f'{mgr["ref_dealin"]:.3f}', flush=True)
                ev_rec['vs_greedy'] = mgr
            if eval_step_eval2 is not None:
                me2 = play_eval_generic(
                    lambda s, seat: eval_step_eval2(params, s, seat),
                    args.eval_games, np.array([True, False, False, False]), ks[ki])
                ki += 1
                print(f'[eval] iter {it} vs_eval2 1v3: win_diff='
                      f'{me2["win_diff"]:+.3f} (cur {me2["cur_win"]:.3f} vs eval2 '
                      f'{me2["ref_win"]:.3f}) draw={me2["draw"]:.3f} '
                      f'dealin(cur/eval2)={me2["cur_dealin"]:.3f}/'
                      f'{me2["ref_dealin"]:.3f}', flush=True)
                ev_rec['vs_eval2'] = me2
            if eval_step_belief is not None:
                mbl = play_eval_generic(
                    lambda s, seat: eval_step_belief(params, s, seat),
                    args.eval_games, np.array([True, False, False, False]), ks[ki])
                ki += 1
                print(f'[eval] iter {it} vs_belief 1v3: win_diff='
                      f'{mbl["win_diff"]:+.3f} (cur {mbl["cur_win"]:.3f} vs belief '
                      f'{mbl["ref_win"]:.3f}) draw={mbl["draw"]:.3f} '
                      f'dealin(cur/belief)={mbl["cur_dealin"]:.3f}/'
                      f'{mbl["ref_dealin"]:.3f}', flush=True)
                ev_rec['vs_belief'] = mbl
            mlog.write(json.dumps(ev_rec) + '\n')

        if it % args.save_every == 0:
            ckpt = os.path.join(args.out_dir, f'iter{it}.msgpack')
            with open(ckpt, 'wb') as f:
                f.write(serialization.to_bytes({'params': params}))
            print(f'[ppo] saved {ckpt}', flush=True)

        if args.anchor_refresh > 0 and it % args.anchor_refresh == 0:
            ref_params = params   # NPG 移动锚：KL 锚推进到当前代
            print(f'[ppo] iter {it}: anchor refreshed', flush=True)

    if args.iters % args.save_every != 0:
        ckpt = os.path.join(args.out_dir, f'iter{args.iters}.msgpack')
        with open(ckpt, 'wb') as f:
            f.write(serialization.to_bytes({'params': params}))
        print(f'[ppo] saved final {ckpt}', flush=True)

    if total_time > 0:
        print(f'[ppo] throughput: {total_decisions / total_time:.0f} decisions/s '
              f'({total_decisions} decisions in {total_time:.1f}s, incl. training)',
              flush=True)
    mlog.close()


if __name__ == '__main__':
    main()
