# -*- coding: utf-8 -*-
"""jaxenv 吞吐 benchmark（M3）：单 GPU vmap + jit，随机策略。

    PYTHONPATH=. python3 jaxenv/benchmark.py [--batches 64 256 1024 4096 8192] [--iters 300]

随机策略（合法 mask 内均匀；锁手/声明/报听决策由 mask 保证合法）整个放进 jit：
    masks -> categorical(logits=0/-inf) -> vmap(step)
终局环境不再演化（step 为 no-op；vmap 下 cond 两分支都执行，吞吐不虚高）。
GPU 上可能有其他租户小任务，绝对值仅供参考。
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from jaxenv import env


def make_fns():
    @jax.jit
    def random_step(states, key):
        masks = jax.vmap(env.legal_mask)(states)
        logits = jnp.where(masks, 0.0, -1e30)
        key, sub = jax.random.split(key)
        acts = jax.random.categorical(sub, logits, axis=-1).astype(jnp.int8)
        states, _, _ = jax.vmap(env.step)(states, acts)
        return states, key

    return random_step


def bench_batch(batch, iters, warmup, seed=0):
    random_step = make_fns()
    keys = jax.random.split(jax.random.PRNGKey(seed), batch)
    states = jax.vmap(env.init)(keys)
    key = jax.random.PRNGKey(seed + 1)
    # 编译 + warmup
    t_compile0 = time.time()
    states, key = random_step(states, key)
    jax.block_until_ready(states)
    t_compile = time.time() - t_compile0
    for _ in range(warmup):
        states, key = random_step(states, key)
    jax.block_until_ready(states)

    t0 = time.time()
    for _ in range(iters):
        states, key = random_step(states, key)
    jax.block_until_ready(states)
    dt = time.time() - t0
    sps = batch * iters / dt
    done_frac = float(np.array(states.done).mean())
    return sps, dt, t_compile, done_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', type=int, nargs='+',
                    default=[64, 256, 1024, 4096, 8192])
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--warmup', type=int, default=30)
    args = ap.parse_args()

    print(f'devices: {jax.devices()[:1]} (+{len(jax.devices()) - 1} more)')
    print(f'{"batch":>6} {"steps/s":>12} {"ms/step":>9} {"compile_s":>9} {"done%":>6}')
    for b in args.batches:
        sps, dt, tc, df = bench_batch(b, args.iters, args.warmup)
        print(f'{b:>6} {sps:>12.0f} {dt / args.iters * 1000:>9.3f} {tc:>9.1f} {df * 100:>5.0f}%',
              flush=True)


if __name__ == '__main__':
    main()
