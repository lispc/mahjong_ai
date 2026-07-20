# -*- coding: utf-8 -*-
"""批量生产「状态 + BeliefExp 全标签」数据（docs/plan-beliefjax-0720.md §2 P1-1a）。

用法：
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 jaxenv/gen_belief_data.py \
        --n-states 3000000 --shard-size 200000 --seed 0 --n-envs 256

- rollout 机制镜像 ppo.py：冻结 old-best net（--init）categorical 采样 +
  对手池（--opp-pool CUR,GEN1,BC,GREEDY,EVAL2,BELIEF；BELIEF 座位动作直接复用
  本步 belief_labels 的 chosen，与 belief_action 由构造保证一致，零额外成本）。
  座位类型每局 init/reset 重采样；done 自动开新局。
- 每条 DISCARD-phase 且未 done 的 pre-step 状态记录一条：
      obs          float16[N,175]
      chosen       int8[N]      belief_action 的选择（0-33）
      top8         int8[N,8]    top-8 候选 idx（top 顺序；不足 8 个合法填 -1）
      offense      int32[N,8]   top-8 的 eval2 整数分子（填充 -(1<<30)）
      danger       float16[N,8] top-8 的 danger 值（填充 -1）
      defense_flag bool[N]      危险信号
      margin       float16[N]   0.03+0.02*报听对手数
      best_offense int32[N]     top-8 内最大进攻分子（附加字段，便于复现 margin 过滤）
- shard 写 <out-dir>/belief_labels_shard_{NNNN}.npz；断点续跑（已有 shard 跳过；
  每 shard 独立子种子 seed+shard_idx*1000003）；每 2% 一行进度含 ETA。
- --self-check（默认开）：首个新 shard 写盘前，从记录行随机抽 100 条对应的
  pre-step State，重算 belief_action/observe/belief_labels 与落盘内容逐条一致
  （obs 按 float16 精度比较，danger 容差 1e-3，其余精确）。
"""

import argparse
import os
import time

import numpy as np

import jax
import jax.numpy as jnp

from jaxenv import env as env_mod
from jaxenv.beliefjax import belief_action, belief_labels
from jaxenv.eval2jax import eval2_action
from jaxenv.greedy import greedy_action
from jaxenv.model_flax import build_model_flax
from jaxenv.obs import observe, actor_of
from jaxenv.ppo import (OPP_CUR, OPP_GEN1, OPP_BC, OPP_GREEDY, OPP_EVAL2,
                        OPP_BELIEF, OPP_NAMES, build_logits, load_params)
import json

NEG = -1e9
N_ACTIONS = env_mod.N_ACTIONS

LABEL_FIELDS = ('chosen', 'top8', 'offense', 'danger', 'defense_flag', 'margin',
                'best_offense')


def make_step_fn(model, pool_w, self_check=False):
    """逐步 rollout + 标签。返回 jitted (params, states, seat_types, rng)
        -> (states', seat_types', rng', out)；out = (obs, labels, keep[, states])。

    pool_w: (6,) numpy 权重（GEN1 不支持——gen_belief_data 只用 CUR/BC/GREEDY/
    EVAL2/BELIEF；GEN1>0 会断言）。BC 与 CUR 共用同一冻结 params。
    """
    pool_w = np.asarray(pool_w, np.float64)
    assert pool_w[OPP_GEN1] == 0, 'gen_belief_data 不支持 GEN1（无 msgpack 参数槽）'
    use_bc = pool_w[OPP_BC] > 0
    use_greedy = pool_w[OPP_GREEDY] > 0
    use_eval2 = pool_w[OPP_EVAL2] > 0
    use_belief = pool_w[OPP_BELIEF] > 0
    pool_logits = jnp.log(jnp.clip(jnp.asarray(pool_w, jnp.float32), 1e-9, 1.0))

    @jax.jit
    def step_fn(params, states, seat_types, rng):
        rng, k_act, k_bc, k_reset, k_seat = jax.random.split(rng, 5)
        obs = jax.vmap(observe)(states)                       # (N,175)
        labels = jax.vmap(belief_labels)(states)              # dict（含 chosen）
        masks = jax.vmap(env_mod.legal_mask)(states)
        phase = states.phase
        player = jax.vmap(actor_of)(states)
        pre_done = states.done
        out = model.apply({'params': params}, obs)
        logits = build_logits(out, phase.astype(jnp.int32), masks)
        safe = jnp.where(pre_done[:, None],
                         jnp.zeros(N_ACTIONS, jnp.float32), logits)
        act = jax.random.categorical(k_act, safe, axis=-1).astype(jnp.int8)

        stype = seat_types[jnp.arange(states.done.shape[0]), player]
        if use_bc:
            # BC = 同一冻结权重的独立采样（镜像 ppo：BC 也走 categorical）
            out_b = model.apply({'params': params}, obs)
            lb = build_logits(out_b, phase.astype(jnp.int32), masks)
            sb = jnp.where(pre_done[:, None],
                           jnp.zeros(N_ACTIONS, jnp.float32), lb)
            ab = jax.random.categorical(k_bc, sb, axis=-1).astype(jnp.int8)
            act = jnp.where(stype == jnp.int8(OPP_BC), ab, act)
        if use_greedy:
            agr = jax.vmap(greedy_action)(states)
            act = jnp.where(stype == jnp.int8(OPP_GREEDY), agr, act)
        if use_eval2:
            ae = jax.vmap(eval2_action)(states)
            act = jnp.where(stype == jnp.int8(OPP_EVAL2), ae, act)
        if use_belief:
            # labels.chosen == belief_action（构造保证一致），零额外成本
            act = jnp.where(stype == jnp.int8(OPP_BELIEF), labels['chosen'], act)

        new_states, _, done = jax.vmap(env_mod.step)(states, act)

        keys = jax.random.split(k_reset, states.done.shape[0])
        fresh = jax.vmap(env_mod.init)(keys)

        def pick(f, s):
            return jnp.where(done.reshape(-1, *([1] * (f.ndim - 1))), f, s)
        states2 = jax.tree.map(pick, fresh, new_states)
        new_st = jax.random.categorical(
            k_seat, jnp.broadcast_to(pool_logits, (*seat_types.shape,
                                                   len(OPP_NAMES))),
            axis=-1).astype(jnp.int8)
        seat_types2 = jnp.where(done[:, None], new_st, seat_types)

        keep = (phase == jnp.int8(env_mod.PHASE_DISCARD)) & (~pre_done)
        out_data = (obs, labels, keep)
        if self_check:
            out_data = out_data + (states,)
        return states2, seat_types2, rng, out_data

    return step_fn


def _labels_to_host(labels, keep):
    """labels dict（jnp）-> host dict（numpy），按 keep 过滤并转目标 dtype。"""
    idx = np.where(np.asarray(keep))[0]
    out = {
        'chosen': np.asarray(labels['chosen'])[idx].astype(np.int8),
        'top8': np.asarray(labels['top8'])[idx].astype(np.int8),
        'offense': np.asarray(labels['offense'])[idx].astype(np.int32),
        'danger': np.asarray(labels['danger'])[idx].astype(np.float16),
        'defense_flag': np.asarray(labels['defense_flag'])[idx].astype(bool),
        'margin': np.asarray(labels['margin'])[idx].astype(np.float16),
        'best_offense': np.asarray(labels['best_offense'])[idx].astype(np.int32),
    }
    return idx, out


def _self_check(buf, raw_states_list, rng_np):
    """从记录行抽 100 条，用对应 pre-step State 重算并逐条比对。"""
    n = len(buf['chosen'])
    if n == 0:
        print('[self-check] no records, skipped', flush=True)
        return
    # 按步 vmap 重算全部 kept 行，再取样比对（单条重建 State 开销大，整步重算更省）
    acts, obses, labels_l = [], [], []
    for st, idxs in raw_states_list:
        acts.append(np.asarray(jax.vmap(belief_action)(st))[idxs])
        obses.append(np.asarray(jax.vmap(observe)(st))[idxs])
        lb = jax.vmap(belief_labels)(st)
        labels_l.append({k2: np.asarray(v)[idxs] for k2, v in lb.items()})
    act_all = np.concatenate(acts)
    obs_all = np.concatenate(obses)
    lab_all = {k2: np.concatenate([x[k2] for x in labels_l]) for k2 in LABEL_FIELDS}
    n = len(buf['chosen'])
    k = min(100, n)
    picks = rng_np.choice(n, k, replace=False)
    bad = []
    for i in picks:
        if int(act_all[i]) != int(buf['chosen'][i]):
            bad.append(('chosen', i, int(act_all[i]), int(buf['chosen'][i])))
        if not np.array_equal(obs_all[i].astype(np.float16), buf['obs'][i]):
            bad.append(('obs', i))
        for f in ('top8', 'offense', 'defense_flag', 'best_offense'):
            if not np.array_equal(lab_all[f][i], buf[f][i]):
                bad.append((f, i))
        if not np.allclose(lab_all['danger'][i].astype(np.float16),
                           buf['danger'][i], atol=1e-3):
            bad.append(('danger', i))
        if not np.allclose(lab_all['margin'][i].astype(np.float16),
                           buf['margin'][i], atol=1e-3):
            bad.append(('margin', i))
    assert not bad, f'self-check 失败 {len(bad)} 例: {bad[:5]}'
    print(f'[self-check] {k} 条抽检一致（chosen/obs/top8/offense/danger/'
          f'defense_flag/margin/best_offense）', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-states', type=int, default=3_000_000)
    ap.add_argument('--shard-size', type=int, default=200_000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-envs', type=int, default=256)
    ap.add_argument('--opp-pool', default='0.5,0,0.2,0.1,0.1,0.1',
                    help='座位类型权重 CUR,GEN1,BC,GREEDY,EVAL2,BELIEF')
    ap.add_argument('--init', default='output/nn_full_action_best_flax.msgpack')
    ap.add_argument('--config', default='output/nn_full_action_best_config.json')
    ap.add_argument('--out-dir', default='output')
    ap.add_argument('--self-check', action='store_true', default=True)
    ap.add_argument('--no-self-check', dest='self_check', action='store_false')
    args = ap.parse_args()

    pool_w = np.array([float(x) for x in args.opp_pool.split(',')], np.float64)
    if pool_w.shape in ((4,), (5,)):
        pool_w = np.concatenate([pool_w, np.zeros(6 - pool_w.shape[0])])
    assert pool_w.shape == (6,) and (pool_w >= 0).all() and \
        abs(pool_w.sum() - 1.0) < 1e-6, f'--opp-pool 需为和为 1 的 6 个权重: {pool_w}'

    with open(args.config) as f:
        config = json.load(f)
    model = build_model_flax(config)
    params = load_params(args.init)

    n_shards = (args.n_states + args.shard_size - 1) // args.shard_size
    step_fn = make_step_fn(model, pool_w, self_check=args.self_check)
    pool_desc = {OPP_NAMES[t]: float(pool_w[t]) for t in range(6) if pool_w[t] > 0}
    print(f'[gen] target={args.n_states} shard_size={args.shard_size} '
          f'n_shards={n_shards} n_envs={args.n_envs} pool={pool_desc} '
          f'seed={args.seed} -> {args.out_dir}', flush=True)

    t_start = time.time()
    produced = 0           # 已完成 shard 的记录数（含跳过）
    steps_total = 0
    checked = False
    for shard in range(n_shards):
        want = min(args.shard_size, args.n_states - shard * args.shard_size)
        path = os.path.join(args.out_dir, f'belief_labels_shard_{shard:04d}.npz')
        if os.path.exists(path):
            print(f'[gen] shard {shard} 已存在，跳过 ({path})', flush=True)
            produced += want
            continue
        rng = jax.random.PRNGKey(args.seed + shard * 1000003)
        rng, k_init, k_st = jax.random.split(rng, 3)
        states = jax.vmap(env_mod.init)(jax.random.split(k_init, args.n_envs))
        pl = jnp.log(jnp.clip(jnp.asarray(pool_w, jnp.float32), 1e-9, 1.0))
        seat_types = jax.random.categorical(
            k_st, jnp.broadcast_to(pl, (args.n_envs, 4, len(OPP_NAMES))),
            axis=-1).astype(jnp.int8)

        buf = {f: [] for f in ('obs',) + LABEL_FIELDS}
        raw_states_list = []                                   # self-check 用
        n_rec = 0
        t_shard = time.time()
        last_log = -1
        while n_rec < want:
            states, seat_types, rng, out_data = step_fn(params, states,
                                                        seat_types, rng)
            if args.self_check:
                obs, labels, keep, raw = out_data
            else:
                obs, labels, keep = out_data
                raw = None
            idx, rec = _labels_to_host(labels, keep)
            if len(idx):
                buf['obs'].append(np.asarray(obs)[idx].astype(np.float16))
                for f in LABEL_FIELDS:
                    buf[f].append(rec[f])
                if args.self_check:
                    raw_states_list.append((raw, idx))
                n_rec += len(idx)
            steps_total += 1
            pct = int(100 * (produced + min(n_rec, want)) / args.n_states)
            if pct >= last_log + 2:
                last_log = pct
                dt = time.time() - t_start
                done_total = produced + min(n_rec, want)
                eta = dt / max(done_total, 1) * (args.n_states - done_total)
                print(f'[gen] {done_total}/{args.n_states} ({pct}%) '
                      f'shard={shard} step={steps_total} '
                      f'rate={done_total / max(dt, 1e-9):.0f}/s ETA={eta / 60:.1f}min',
                      flush=True)

        buf = {f: np.concatenate(v)[:want] for f, v in buf.items()}
        if args.self_check and not checked:
            _self_check(buf, raw_states_list, np.random.default_rng(args.seed))
            checked = True
        np.savez(path, **buf)
        dt = time.time() - t_shard
        produced += len(buf['chosen'])
        print(f'[gen] wrote {path} ({len(buf["chosen"])} rows, {dt:.1f}s)', flush=True)

    dt = time.time() - t_start
    print(f'[gen] done: {produced} rows in {dt:.1f}s '
          f'({produced / max(dt, 1e-9):.0f} rows/s, {steps_total} env-steps)',
          flush=True)


if __name__ == '__main__':
    main()
