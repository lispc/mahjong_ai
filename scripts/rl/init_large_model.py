# -*- coding: utf-8 -*-
"""从零初始化一个指定架构的 TileConvNet checkpoint，供 train_full_action.py 热启。

用法：
    PYTHONPATH=. python3 scripts/rl/init_large_model.py \
        output/nn_full_action_large_init.pt \
        --channels 256 --n-blocks 8 --hidden-dim 1024 --se-ratio 16
"""

import argparse
import json
import torch
from algo.nn.model import TileConvNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_path')
    ap.add_argument('--input-dim', type=int, default=175)
    ap.add_argument('--channels', type=int, default=128)
    ap.add_argument('--n-blocks', type=int, default=6)
    ap.add_argument('--hidden-dim', type=int, default=512)
    ap.add_argument('--n-tile-ch', type=int, default=5)
    ap.add_argument('--tile-len', type=int, default=34)
    ap.add_argument('--se-ratio', type=int, default=0)
    ap.add_argument('--attn-heads', type=int, default=0)
    ap.add_argument('--attn-layers', type=int, default=0)
    ap.add_argument('--dealin-head', type=int, default=1)
    ap.add_argument('--tenpai-head', type=int, default=1)
    ap.add_argument('--response-head', type=int, default=1)
    args = ap.parse_args()

    cfg = {
        'arch': 'conv',
        'input_dim': args.input_dim,
        'channels': args.channels,
        'n_blocks': args.n_blocks,
        'hidden_dim': args.hidden_dim,
        'n_tile_ch': args.n_tile_ch,
        'tile_len': args.tile_len,
        'se_ratio': args.se_ratio,
        'attn_heads': args.attn_heads,
        'attn_layers': args.attn_layers,
        'dealin_head': bool(args.dealin_head),
        'tenpai_head': bool(args.tenpai_head),
        'response_head': bool(args.response_head),
        'framework': 'pytorch',
        'source': 'large_init',
    }
    model = TileConvNet(**{k: cfg[k] for k in [
        'input_dim', 'channels', 'n_blocks', 'hidden_dim', 'n_tile_ch', 'tile_len',
        'dealin_head', 'tenpai_head', 'response_head', 'se_ratio', 'attn_heads', 'attn_layers']})
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Init {args.channels}/{args.n_blocks}/{args.hidden_dim} SE={args.se_ratio} '
          f'Attn={args.attn_heads}x{args.attn_layers}: {n_params:,} params')

    torch.save({'model_state': model.state_dict(), 'config': cfg}, args.out_path)
    with open(args.out_path.replace('.pt', '_config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'Saved {args.out_path}')


if __name__ == '__main__':
    main()
