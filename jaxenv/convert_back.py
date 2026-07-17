# -*- coding: utf-8 -*-
"""Flax msgpack params -> PyTorch TileConvNet state_dict（jaxenv/convert_torch.py 的逆映射）。

用法：
    PYTHONPATH=. python3 jaxenv/convert_back.py \
        output/jax_ppo/iter5.msgpack output/jax_ppo/iter5.pt \
        [--config output/nn_full_action_best_config.json] \
        [--verify-against output/nn_full_action_best.pt] [--smoke-games 4]

- 映射规则（convert_torch 的逆）：
  Conv1d kernel (k,in,out) -> torch (out,in,k) = transpose(2,1,0)；
  Linear kernel (in,out) -> torch (out,in) = 转置；
  GroupNorm scale/bias -> weight/bias；blocks_{i} -> blocks.{i}。
- 产物：torch.save({'model_state': sd, 'config': config}) + 旁边落 <name>_config.json，
  可直接被 algo/agents/ppo_agent.py::PPOAgent 加载。
- --verify-against：与原 .pt checkpoint 做张量级对比（roundtrip 应 ~1e-6 内）。
- --smoke-games N：用 PPOAgent 加载产物，与 3 个 baseline 在 driver/engine.py
  play_game 里打 N 局（不崩、出牌合法即过，胜率无断言）。
"""

import argparse
import json
import os

import numpy as np
import torch
from flax import serialization


def convert_params_back(params):
    """flax params 树 -> torch state_dict（convert_torch.convert_state_dict 的逆）。"""
    sd = {}
    for mod, sub in params.items():
        if mod == 'stem' or mod.endswith('_conv'):      # Conv1d（含 1x1）
            sd[f'{mod}.weight'] = torch.from_numpy(
                np.asarray(sub['kernel']).transpose(2, 1, 0).copy())
            sd[f'{mod}.bias'] = torch.from_numpy(np.asarray(sub['bias']).copy())
        elif mod == 'stem_bn':                            # GroupNorm
            sd['stem_bn.weight'] = torch.from_numpy(np.asarray(sub['scale']).copy())
            sd['stem_bn.bias'] = torch.from_numpy(np.asarray(sub['bias']).copy())
        elif mod.startswith('blocks_'):                   # blocks_{i}.{c1,c2,b1,b2}
            i = mod.split('_', 1)[1]
            for layer, lsub in sub.items():
                key = f'blocks.{i}.{layer}'
                if layer.startswith('c'):                 # conv
                    sd[f'{key}.weight'] = torch.from_numpy(
                        np.asarray(lsub['kernel']).transpose(2, 1, 0).copy())
                    sd[f'{key}.bias'] = torch.from_numpy(np.asarray(lsub['bias']).copy())
                else:                                     # GroupNorm b1/b2
                    sd[f'{key}.weight'] = torch.from_numpy(np.asarray(lsub['scale']).copy())
                    sd[f'{key}.bias'] = torch.from_numpy(np.asarray(lsub['bias']).copy())
        else:                                             # Linear heads
            sd[f'{mod}.weight'] = torch.from_numpy(np.asarray(sub['kernel']).T.copy())
            sd[f'{mod}.bias'] = torch.from_numpy(np.asarray(sub['bias']).copy())
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('msgpack')
    ap.add_argument('out_pt')
    ap.add_argument('--config', default='output/nn_full_action_best_config.json')
    ap.add_argument('--verify-against', default=None)
    ap.add_argument('--smoke-games', type=int, default=0)
    args = ap.parse_args()

    with open(args.msgpack, 'rb') as f:
        variables = serialization.from_bytes(None, f.read())
    params = variables['params']
    with open(args.config) as f:
        config = json.load(f)

    from algo.nn.model import build_model
    net = build_model(config)
    sd = convert_params_back(params)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f'missing={missing} unexpected={unexpected}'
    print(f'state_dict OK: {len(sd)} tensors')

    if args.verify_against:
        ckpt = torch.load(args.verify_against, map_location='cpu')
        ref = ckpt['model_state'] if 'model_state' in ckpt else ckpt
        diffs = []
        for k in ref:
            if k in sd and ref[k].shape == sd[k].shape:
                diffs.append(float((ref[k].float() - sd[k].float()).abs().max()))
        md = max(diffs)
        print(f'roundtrip vs {args.verify_against}: {len(diffs)} tensors, max abs diff {md:.3e}')
        assert md < 1e-5, f'roundtrip diff too large: {md}'

    torch.save({'model_state': sd, 'config': config}, args.out_pt)
    cfg_path = args.out_pt.replace('.pt', '_config.json')
    config_out = dict(config)
    config_out['framework'] = 'pytorch'
    with open(cfg_path, 'w') as f:
        json.dump(config_out, f, indent=2)
    print(f'saved -> {args.out_pt} (+ {cfg_path})')

    if args.smoke_games > 0:
        import agent as agent_mod
        from algo.agents.ppo_agent import PPOAgent
        from driver.engine import play_game

        n_win = n_ron = n_draw = 0
        for g in range(args.smoke_games):
            agents = [PPOAgent('Converted@0', model_path=args.out_pt,
                               device='cpu', temperature=0.0)]
            agents += [agent_mod.Agent(f'Baseline@{s}', verbose=False)
                       for s in range(1, 4)]
            res = play_game(agents, seed=1000 + g, verbose=False)
            wt = res['win_type']
            n_win += (res['winner'] == 'Converted@0')
            n_ron += (wt == 'ron')
            n_draw += (wt == 'draw')
            print(f'  game {g}: winner={res["winner"]} type={wt}')
        print(f'smoke {args.smoke_games} games OK: converted wins={n_win}, '
              f'ron={n_ron}, draw={n_draw}（无胜率断言，不崩即过）')


if __name__ == '__main__':
    main()
