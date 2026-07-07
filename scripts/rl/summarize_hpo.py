# -*- coding: utf-8 -*-
"""汇总 HPO 训练日志，输出每个实验当前最佳 val disc_acc 与最近 epoch。"""

import os
import re
import sys
import glob

LOG_DIR = 'output'


def parse_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            m = re.search(
                r'Epoch\s+(\d+)\s+\|\s+disc_loss\s+([\d.]+)\s+resp_loss\s+([\d.]+)\s+'
                r'v_loss\s+([\d.]+)\s+t_loss\s+([\d.]+)\s+\|\s+'
                r'val disc_acc\s+([\d.]+)\s+resp_acc\s+([\d.]+)\s+v_mse\s+([\d.]+)\s+'
                r't_bce\s+([\d.]+)', line)
            if m:
                rows.append({
                    'epoch': int(m.group(1)),
                    'disc_loss': float(m.group(2)),
                    'resp_loss': float(m.group(3)),
                    'v_loss': float(m.group(4)),
                    't_loss': float(m.group(5)),
                    'val_disc_acc': float(m.group(6)),
                    'val_resp_acc': float(m.group(7)),
                    'val_v_mse': float(m.group(8)),
                    'val_t_bce': float(m.group(9)),
                })
    return rows


def main():
    logs = sorted(glob.glob(os.path.join(LOG_DIR, 'train_hpo_*.log')))
    if not logs:
        print('No train_hpo_*.log found')
        sys.exit(0)
    print(f'{"exp":<12} {"ep":>4} {"best_ep":>6} {"best_disc":>10} {"last_disc":>10} '
          f'{"last_v_mse":>11} {"status":>10}')
    for log in logs:
        rows = parse_log(log)
        name = os.path.basename(log).replace('train_', '').replace('.log', '')
        if not rows:
            print(f'{name:<12}    -      -          -          -           -     no-epochs')
            continue
        best = max(rows, key=lambda r: r['val_disc_acc'])
        last = rows[-1]
        status = 'done' if last['epoch'] >= 60 else 'running'
        print(f'{name:<12} {last["epoch"]:>4} {best["epoch"]:>6} {best["val_disc_acc"]:>10.4f} '
              f'{last["val_disc_acc"]:>10.4f} {last["val_v_mse"]:>11.4f} {status:>10}')


if __name__ == '__main__':
    main()
