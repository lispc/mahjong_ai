# -*- coding: utf-8 -*-
"""Duplicate（复式）benchmark 的 JAX 后端 CLI：镜像 benchmark_duplicate.py 的接口与报告。

用法：
    PYTHONPATH=. python3 scripts/rl/benchmark_duplicate_jax.py \
        --a baseline --b beliefexp \
        --opponents baseline,beliefexp,beliefexp \
        --n-seeds 500 --output tmp/jax_dup_sanity_500.pkl

token：
    baseline  -> EVAL2（arena Baseline 移植，jaxenv/eval2jax.py）
    beliefexp -> BELIEF（BeliefExp 移植，jaxenv/beliefjax.py，~98% top-1 parity）
    greedy    -> GREEDY（shanten 贪心；Python 后端无对应物）
    nn:LABEL:PATH[:CONFIG] -> NN masked argmax 纯前馈（AutoHu 风格：能胡必胡、
        报听恒否；response 头决定碰/杠，无搜索层）。PATH 支持：
        - .msgpack（flax 序列化，ppo.py 同款 load_params）；CONFIG 缺省时依次尝试
          <dirname>/config.json（ppo out-dir 约定）、<stem>_config.json、
          以及 '_flax.msgpack' -> '_config.json' 替换（nn_full_action_best_flax 约定）；
        - .pt（PyTorch checkpoint，内含 config，经 jaxenv/convert_torch.py 转换）。
        同一 run 内全部 NN token 必须同架构；相同 PATH 共享同一槽位。

与 Python 后端的差异（详见 jaxenv/duplicate_arena.py 模块 docstring）：
    无 hybridnm 座位；断代后标准三件套只能近似。同 seed 号两端牌墙不同（统计等价）。

报告/pkl：与 benchmark_duplicate.py 相同的小节与 schema
（players_order 命名 f'{name}@{pos}_{a|b}'，paired 块公式逐行一致）。
"""

import argparse
import json
import math
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checker.report import compute_metrics
from jaxenv import duplicate_arena as dup


# ---------------------------------------------------------------------------
# token 解析
# ---------------------------------------------------------------------------

_BASE_TOKENS = {'baseline': (dup.TYPE_EVAL2, 'Baseline'),
                'beliefexp': (dup.TYPE_BELIEF, 'BeliefExp'),
                'greedy': (dup.TYPE_GREEDY, 'Greedy')}

# 架构一致性比较只取这些键（训练元数据等不参与）
_ARCH_KEYS = ('arch', 'input_dim', 'channels', 'n_blocks', 'hidden_dim',
              'n_tile_ch', 'dealin_head', 'tenpai_head', 'response_head',
              'se_ratio', 'attn_heads', 'attn_layers')


def _arch_signature(config):
    return tuple((k, config.get(k)) for k in _ARCH_KEYS)


def _find_msgpack_config(path):
    """msgpack 的 config 解析：返回路径或 None（候选按优先级列出）。"""
    stem = path[:-len('.msgpack')] if path.endswith('.msgpack') else path
    cands = [os.path.join(os.path.dirname(path), 'config.json'),
             stem + '_config.json']
    if os.path.basename(path).endswith('_flax.msgpack'):
        cands.append(stem[:-len('_flax')] + '_config.json')
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def _load_nn(path, config_path=None):
    """加载 NN：返回 (config, params)。支持 .msgpack 与 .pt。"""
    if path.endswith('.pt'):
        import torch                                     # 延迟导入，纯规则 seat 不需要
        from jaxenv.convert_torch import convert_state_dict
        ckpt = torch.load(path, map_location='cpu')
        config = ckpt['config']
        params = convert_state_dict(ckpt['model_state'])
        return config, params
    if config_path is None:
        config_path = _find_msgpack_config(path)
        if config_path is None:
            raise FileNotFoundError(
                f'config not found for {path}; tried <dirname>/config.json, '
                f'<stem>_config.json, _flax->_config.json; '
                f'use nn:LABEL:PATH:CONFIG to specify explicitly')
    with open(config_path) as f:
        config = json.load(f)
    from flax import serialization
    with open(path, 'rb') as f:
        variables = serialization.from_bytes(None, f.read())
    return config, variables['params']


class _TokenParser:
    """把 CLI token 映射到 (type_code, name)；NN token 分配槽位并去重 PATH。"""

    def __init__(self):
        self.nn_paths = []        # slot -> path
        self.configs = []         # slot -> config
        self.params = []          # slot -> params
        self._slot_of = {}        # path -> slot

    def parse(self, tok):
        tok = tok.strip()
        if tok in _BASE_TOKENS:
            return _BASE_TOKENS[tok]
        if tok.startswith('nn:'):
            parts = tok.split(':')
            if len(parts) not in (3, 4):
                raise ValueError(f'nn token needs 3 or 4 parts '
                                 f'(nn:LABEL:PATH[:CONFIG]): {tok}')
            _, label, path = parts[0], parts[1], parts[2]
            config_path = parts[3] if len(parts) == 4 else None
            if path not in self._slot_of:
                config, params = _load_nn(path, config_path)
                if self.configs and (_arch_signature(config)
                                     != _arch_signature(self.configs[0])):
                    raise ValueError(
                        f'all NN tokens in one run must share architecture; '
                        f'{path} differs from {self.nn_paths[0]}')
                self._slot_of[path] = len(self.nn_paths)
                self.nn_paths.append(path)
                self.configs.append(config)
                self.params.append(params)
            return dup.TYPE_NN_BASE + self._slot_of[path], f'NN-{label}'
        raise ValueError(f'unknown token: {tok} (supported: baseline, beliefexp, '
                         f'greedy, nn:LABEL:PATH[:CONFIG])')


def main():
    parser = argparse.ArgumentParser(
        description='Duplicate tournament benchmark (JAX backend)')
    parser.add_argument('--a', required=True, help='candidate A token')
    parser.add_argument('--b', required=True, help='candidate B token')
    parser.add_argument('--opponents', required=True,
                        help='exactly 3 opponent tokens, comma separated')
    parser.add_argument('--n-seeds', type=int, default=400)
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--output', default=None,
                        help='optional path to write raw results pickle')
    parser.add_argument('--chunk', type=int, default=1024,
                        help='pairs per vmap batch (default 1024 = 2048 lanes; '
                             '吞吐在 2048-lane 批饱和 ~21k env-steps/s@3090 共享)')
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--scan-steps', type=int, default=32,
                        help='env steps per jitted scan block')
    parser.add_argument('--auto-hu', action=argparse.BooleanOptionalAction,
                        default=True, help='NN seats: force hu at CLAIM-HU '
                                           '(AutoHu style, default on)')
    parser.add_argument('--force-no-tenpai', action=argparse.BooleanOptionalAction,
                        default=True, help='NN seats: force tenpai-no '
                                           '(AutoHu style, default on)')
    args = parser.parse_args()

    # eval2/belief 的 step block 首次 XLA 编译 ~2-2.5 min（一次性）；缓存命中后 ~10s
    dup.enable_compile_cache()

    tp = _TokenParser()
    type_a, a_name = tp.parse(args.a)
    type_b, b_name = tp.parse(args.b)
    opp_tokens = [t.strip() for t in args.opponents.split(',')]
    if len(opp_tokens) != 3:
        raise ValueError('--opponents must contain exactly 3 tokens')
    opp_parsed = [tp.parse(t) for t in opp_tokens]
    opp_types = tuple(t for t, _ in opp_parsed)
    opp_names = [n for _, n in opp_parsed]

    model = None
    if tp.nn_paths:
        import jax
        import jax.numpy as jnp
        from jaxenv.model_flax import build_model_flax
        model = build_model_flax(tp.configs[0])
        # 结构校验：params 树与 model.init 的 keys/shape 完全一致（convert_torch 同款检查）
        dummy = jnp.zeros((1, tp.configs[0].get('input_dim', 175)), jnp.float32)
        ref_flat = jax.tree_util.tree_flatten_with_path(
            model.init(jax.random.PRNGKey(0), dummy))[0]
        ref_map = {jax.tree_util.keystr(k): v.shape for k, v in ref_flat}
        for slot, params in enumerate(tp.params):
            got_map = {jax.tree_util.keystr(k): v.shape
                       for k, v in jax.tree_util.tree_flatten_with_path(
                           {'params': params})[0]}
            missing = set(ref_map) - set(got_map)
            extra = set(got_map) - set(ref_map)
            mismatch = {k for k in ref_map.keys() & got_map.keys()
                        if ref_map[k] != got_map[k]}
            if missing or extra or mismatch:
                raise ValueError(f'NN slot {slot} ({tp.nn_paths[slot]}): '
                                 f'missing={missing} extra={extra} '
                                 f'shape_mismatch={mismatch}')

    total_games = args.n_seeds * 2
    print(f'Duplicate benchmark (JAX backend): {a_name} vs {b_name}')
    print(f'Opponents: {opp_names}')
    print(f'Seeds: {args.n_seeds}, positions mirrored: False, '
          f'total games: {total_games}, backend: jax '
          f'(seat types: A={dup.type_name(type_a)}, B={dup.type_name(type_b)}, '
          f'opps={[dup.type_name(t) for t in opp_types]})')
    if tp.nn_paths:
        print(f'NN slots: {tp.nn_paths} '
              f'(auto_hu={args.auto_hu}, force_no_tenpai={args.force_no_tenpai})')

    t0 = time.time()
    out = dup.run_duplicate(
        type_a, type_b, opp_types, args.n_seeds, seed_offset=args.seed_offset,
        a_name=a_name, b_name=b_name, opp_names=tuple(opp_names),
        model=model, nn_params=tuple(tp.params), chunk_pairs=args.chunk,
        max_steps=args.max_steps, scan_steps=args.scan_steps,
        auto_hu=args.auto_hu, no_tenpai=args.force_no_tenpai, verbose=True)
    dt = time.time() - t0
    results = out['results']

    metrics = compute_metrics(results, [a_name, b_name] + opp_names)

    # Candidate-specific win rates（镜像 benchmark_duplicate.py 的小节与算法）
    a_wins_total, a_games = dup.candidate_wins(results, a_name, 'a')
    b_wins_total, b_games = dup.candidate_wins(results, b_name, 'b')
    print(f'\nCandidate-specific win rates:')
    print(f'  {a_name:20s}: {a_wins_total}/{a_games} = {a_wins_total/a_games:.1%}')
    print(f'  {b_name:20s}: {b_wins_total}/{b_games} = {b_wins_total/b_games:.1%}')
    print(f'  Simple A-B diff: '
          f'{(a_wins_total/a_games - b_wins_total/b_games):+.1%}')

    print(f'\nTotal {dt:.1f}s ({args.n_seeds/dt:.1f} seeds/s, '
          f'{total_games/dt:.0f} games/s)')
    for name in [a_name, b_name] + opp_names:
        m = metrics[name]
        print(f'  {name:20s}: win {m["win_rate"]:.1%}, '
              f'self {m["self_rate"]:.1%}, ron {m["ron_rate"]:.1%}, '
              f'draw {m["draw_rate"]:.1%}')

    p = out['paired']
    n_pairs = p['n_pairs']
    print(f'\nPaired difference ({a_name} - {b_name}):')
    print(f'  A wins {p["a_wins"]}/{n_pairs} ({p["a_wins"]/n_pairs:.1%})')
    print(f'  B wins {p["b_wins"]}/{n_pairs} ({p["b_wins"]/n_pairs:.1%})')
    print(f'  Ties   {p["ties"]}/{n_pairs} ({p["ties"]/n_pairs:.1%})')
    print(f'  A-B = {p["diff"]:+.1%}, 95% CI [{p["ci_lo"]:+.1%}, {p["ci_hi"]:+.1%}]')
    print(f'  Score-proxy A-B (self=+3, ron=+1, deal-in=-1): '
          f'{p["score_diff"]:+.3f}, '
          f'95% CI [{p["score_ci_lo"]:+.3f}, {p["score_ci_hi"]:+.3f}]')
    if p['ci_lo'] > 0:
        print(f'  => {a_name} significantly stronger (CI excludes 0)')
    elif p['ci_hi'] < 0:
        print(f'  => {b_name} significantly stronger (CI excludes 0)')
    else:
        print('  => difference not significant at 95%')

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'wb') as f:
            pickle.dump({
                'args': vars(args),
                'results': results,
                'a_name': a_name,
                'b_name': b_name,
                'opp_names': opp_names,
                'metrics': metrics,
                'paired': p,
            }, f)
        print(f'Raw results saved to {args.output}')


if __name__ == '__main__':
    main()
