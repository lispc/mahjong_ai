# -*- coding: utf-8 -*-
"""启动 large SE/attention 模型训练（从 init_large_model.py 生成的随机初始化）。"""

import os
import sys
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--data', default='output/nn_hybrid_soup_teacher_8000.npz')
    ap.add_argument('--init', default='output/nn_full_action_large_se_init.pt')
    ap.add_argument('--out', default='output/nn_full_action_large_se.pt')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--optimizer', default='adam', choices=['adam', 'adamw'])
    ap.add_argument('--scheduler', default='cosine', choices=['cosine', 'plateau', 'step'])
    ap.add_argument('--num-workers', type=int, default=4)
    args = ap.parse_args()

    cmd = (
        f'CUDA_VISIBLE_DEVICES={args.gpu} PYTHONPATH=. '
        f'python3 -u scripts/rl/train_full_action.py '
        f'{args.data} {args.init} {args.out} '
        f'--epochs {args.epochs} --batch {args.batch} --lr {args.lr} --wd {args.wd} '
        f'--optimizer {args.optimizer} --scheduler {args.scheduler} '
        f'--num_workers {args.num_workers} --dp 0 '
        f'> output/train_large_se_gpu{args.gpu}.log 2>&1'
    )
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    main()
