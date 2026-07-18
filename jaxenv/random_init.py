# -*- coding: utf-8 -*-
"""生成随机初始化的 Flax TileConvNet 权重（from-scratch 起点）。

用法：
    PYTHONPATH=. python3 jaxenv/random_init.py \
        --config output/nn_full_action_best_config.json \
        --out output/jax_scratch_init.msgpack --seed 0
"""

import argparse
import json

import jax
import jax.numpy as jnp
from flax import serialization

from jaxenv.model_flax import build_model_flax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='output/nn_full_action_best_config.json')
    ap.add_argument('--out', default='output/jax_scratch_init.msgpack')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    model = build_model_flax(config)
    variables = model.init(jax.random.PRNGKey(args.seed),
                           jnp.zeros((1, config.get('input_dim', 175))))
    with open(args.out, 'wb') as f:
        f.write(serialization.to_bytes(variables))
    n = sum(x.size for x in jax.tree.leaves(variables['params']))
    print(f'saved random init ({n} params) -> {args.out}')


if __name__ == '__main__':
    main()
