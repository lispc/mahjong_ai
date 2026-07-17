# -*- coding: utf-8 -*-
"""PyTorch vs Flax 数值对齐验证 + 单卡批量推理吞吐。

- 从 output/ptie_v1_merged.npz 的 feats 取 1024 条真实特征（seed 固定）；
- PyTorch: cpu, eval, no_grad；Flax: jit（全精度 matmul，见下）逐 head 对比 max abs diff；
- 要求所有 head < 1e-4 (float32)；
- 最后测单 GPU batch=4096 推理吞吐（steps/s）。

注意：JAX 在 Ampere+ GPU 上默认对 f32 matmul/conv 用 TF32（相对误差 ~1e-3），
会超过 1e-4 容差，因此 parity 对比前设置 jax_default_matmul_precision='highest'。
吞吐数字分别报告 highest（与 parity 同数值路径）和 default(TF32) 两种。

用法：PYTHONPATH=. python3 jaxenv/test_parity.py
"""

import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import serialization

# 必须在 jax 计算前设置：关掉 TF32，用全精度 float32
jax.config.update('jax_default_matmul_precision', 'highest')

from algo.nn.model import build_model
from jaxenv.model_flax import build_model_flax

PT_PATH = 'output/nn_full_action_best.pt'
MSGPACK_PATH = 'output/nn_full_action_best_flax.msgpack'
FEATS_PATH = 'output/ptie_v1_merged.npz'
N_SAMPLES = 1024
TOL = 1e-4


def main():
    # ---- 数据 ----
    d = np.load(FEATS_PATH)
    feats = d['feats']
    rng = np.random.RandomState(0)
    idx = rng.choice(feats.shape[0], N_SAMPLES, replace=False)
    X = feats[idx].astype(np.float32)
    print(f'samples: {N_SAMPLES} from {FEATS_PATH} (feats {feats.shape})')

    # ---- PyTorch (cpu, eval, no_grad) ----
    ckpt = torch.load(PT_PATH, map_location='cpu')
    config = ckpt['config']
    print('config:', json.dumps(config))
    tmodel = build_model(config)
    tmodel.load_state_dict(ckpt['model_state'])
    tmodel.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X)
        outs = tmodel(xt)  # (policy, value, dealin, response)（按 config 开关顺序）
        tenpai = tmodel.tenpai_logit(xt)
    t_out = {
        'policy': outs[0].numpy(),
        'value': outs[1].numpy(),
        'dealin': outs[2].numpy(),
        'response': outs[3].numpy(),
        'tenpai': tenpai.numpy(),
    }

    # ---- Flax (jit) ----
    fmodel = build_model_flax(config)
    with open(MSGPACK_PATH, 'rb') as f:
        variables = serialization.from_bytes(None, f.read())
    apply_jit = jax.jit(lambda v, x: fmodel.apply(v, x))
    f_out = apply_jit(variables, jnp.asarray(X))

    # ---- 逐 head 对比 ----
    ok = True
    print('--- max abs diff per head (torch cpu vs flax jit) ---')
    for head in ('policy', 'value', 'dealin', 'tenpai', 'response'):
        a, b = t_out[head], np.asarray(f_out[head])
        assert a.shape == b.shape, f'{head} shape mismatch: {a.shape} vs {b.shape}'
        diff = float(np.abs(a - b).max())
        status = 'OK' if diff < TOL else 'FAIL'
        ok = ok and diff < TOL
        print(f'{head:10s} shape={str(a.shape):12s} max_abs_diff={diff:.3e}  [{status}]')
    print(f'ALL < {TOL:g}: {ok}')
    assert ok, 'parity check failed'

    # ---- 吞吐：单 GPU batch=4096 ----
    dev = jax.devices('gpu')[0]
    xb = jax.device_put(jnp.asarray(np.random.RandomState(1).randn(4096, 175).astype(np.float32)), dev)
    vb = jax.device_put(variables, dev)
    for _ in range(5):  # warmup + 编译
        jax.block_until_ready(apply_jit(vb, xb))
    n_steps = 100
    t0 = time.perf_counter()
    for _ in range(n_steps):
        jax.block_until_ready(apply_jit(vb, xb))
    dt = time.perf_counter() - t0
    print(f'throughput(highest): {n_steps / dt:.1f} steps/s '
          f'({4096 * n_steps / dt:.0f} samples/s) on {dev}')

    # default 精度（TF32）参考值：PPO 训练若想用 TF32 需另行验证数值
    jax.config.update('jax_default_matmul_precision', 'default')
    for _ in range(5):
        jax.block_until_ready(apply_jit(vb, xb))
    t0 = time.perf_counter()
    for _ in range(n_steps):
        jax.block_until_ready(apply_jit(vb, xb))
    dt = time.perf_counter() - t0
    print(f'throughput(default/TF32): {n_steps / dt:.1f} steps/s '
          f'({4096 * n_steps / dt:.0f} samples/s)')


if __name__ == '__main__':
    main()
