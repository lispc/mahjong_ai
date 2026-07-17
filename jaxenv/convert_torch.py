# -*- coding: utf-8 -*-
"""把 PyTorch TileConvNet checkpoint 转成 Flax params 并序列化为 msgpack。

用法：
    PYTHONPATH=. python3 jaxenv/convert_torch.py \
        output/nn_full_action_best.pt output/nn_full_action_best_flax.msgpack

权重布局转换：
- Conv1d: torch (out, in, k) -> flax kernel (k, in, out)，即 transpose(2, 1, 0)
- Linear: torch (out, in)     -> flax kernel (in, out)，即转置
- GroupNorm: weight/bias      -> scale/bias
- blocks.{i}.xxx              -> blocks_{i}/xxx（flax 模块名不能用 '.'）
"""

import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import serialization

from jaxenv.model_flax import build_model_flax


def convert_state_dict(sd: dict) -> dict:
    params = {}

    for key, tensor in sd.items():
        w = tensor.detach().cpu().float()
        parts = key.split('.')
        if parts[0] == 'stem':
            sub = params.setdefault('stem', {})
            sub['kernel' if parts[1] == 'weight' else 'bias'] = (
                jnp.asarray(w.numpy().transpose(2, 1, 0)) if parts[1] == 'weight'
                else jnp.asarray(w.numpy()))
        elif parts[0] == 'stem_bn':
            sub = params.setdefault('stem_bn', {})
            sub['scale' if parts[1] == 'weight' else 'bias'] = jnp.asarray(w.numpy())
        elif parts[0] == 'blocks':
            mod, layer, kind = f'blocks_{parts[1]}', parts[2], parts[3]
            sub = params.setdefault(mod, {}).setdefault(layer, {})
            if layer.startswith('c'):  # conv
                sub['kernel' if kind == 'weight' else 'bias'] = (
                    jnp.asarray(w.numpy().transpose(2, 1, 0)) if kind == 'weight'
                    else jnp.asarray(w.numpy()))
            else:  # GroupNorm b1/b2
                sub['scale' if kind == 'weight' else 'bias'] = jnp.asarray(w.numpy())
        else:
            mod, kind = parts[0], parts[1]
            sub = params.setdefault(mod, {})
            if w.ndim == 3:  # 1x1 conv (policy_conv / dealin_conv)
                sub['kernel' if kind == 'weight' else 'bias'] = (
                    jnp.asarray(w.numpy().transpose(2, 1, 0)) if kind == 'weight'
                    else jnp.asarray(w.numpy()))
            elif w.ndim == 2:  # linear
                sub['kernel' if kind == 'weight' else 'bias'] = (
                    jnp.asarray(w.numpy().T) if kind == 'weight'
                    else jnp.asarray(w.numpy()))
            else:
                sub['scale' if kind == 'weight' else 'bias'] = jnp.asarray(w.numpy())
    return params


def main():
    pt_path = sys.argv[1] if len(sys.argv) > 1 else 'output/nn_full_action_best.pt'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'output/nn_full_action_best_flax.msgpack'

    ckpt = torch.load(pt_path, map_location='cpu')
    config = ckpt['config']
    sd = ckpt['model_state']
    print('config:', json.dumps(config))

    model = build_model_flax(config)
    params = convert_state_dict(sd)
    variables = {'params': params}

    # 结构校验：与 model.init 的参数树 keys/shape 完全一致
    dummy = jnp.zeros((1, config.get('input_dim', 175)), jnp.float32)
    ref = model.init(jax.random.PRNGKey(0), dummy)
    ref_flat = jax.tree_util.tree_flatten_with_path(ref)[0]
    got_flat = jax.tree_util.tree_flatten_with_path(variables)[0]
    ref_map = {jax.tree_util.keystr(k): v.shape for k, v in ref_flat}
    got_map = {jax.tree_util.keystr(k): v.shape for k, v in got_flat}
    missing = set(ref_map) - set(got_map)
    extra = set(got_map) - set(ref_map)
    mismatch = {k for k in ref_map.keys() & got_map.keys() if ref_map[k] != got_map[k]}
    assert not missing and not extra and not mismatch, (
        f'missing={missing} extra={extra} shape_mismatch={mismatch}')
    print(f'param tree OK: {len(ref_map)} leaves')

    with open(out_path, 'wb') as f:
        f.write(serialization.to_bytes(variables))
    print(f'saved -> {out_path}')


if __name__ == '__main__':
    main()
