# -*- coding: utf-8 -*-
"""等待 AlphaZero trace 生成完成后自动训练。

用法：
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/rl/wait_and_train_az.py \
        output/alphazero_trace_200.npz \
        output/nn_full_action_data_128000.npz \
        output/nn_full_action_best.pt \
        output/nn_full_action_az.pt \
        output/train_alphazero_200.log
"""
import os
import sys
import time
import subprocess
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _trace_ready(trace_path):
    return os.path.exists(trace_path) and os.path.getsize(trace_path) > 1024


def _gen_running():
    try:
        out = subprocess.check_output(['pgrep', '-f', 'gen_alphazero_data.py'], stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except subprocess.CalledProcessError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace_path')
    ap.add_argument('response_data')
    ap.add_argument('init_model')
    ap.add_argument('out_model')
    ap.add_argument('log_path')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--max-wait-hours', type=float, default=12.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.log_path) or '.', exist_ok=True)
    log = open(args.log_path, 'a', buffering=1)
    log.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Monitor started\n')

    deadline = time.time() + args.max_wait_hours * 3600
    while time.time() < deadline:
        if _trace_ready(args.trace_path):
            # 再确认生成进程已退出，避免读到半成品
            if not _gen_running():
                break
        elapsed = time.time() - (deadline - args.max_wait_hours * 3600)
        if int(elapsed) % 600 < 60:
            log.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] waiting... {args.trace_path}\n')
        time.sleep(60)

    if not _trace_ready(args.trace_path):
        log.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Timeout waiting for trace\n')
        log.close()
        return

    log.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Trace ready, start training\n')
    log.flush()

    cmd = [
        sys.executable, 'scripts/rl/train_alphazero.py',
        args.trace_path, args.response_data, args.init_model, args.out_model,
        '--epochs', '60',
        '--batch', '512',
        '--lr', '1e-4',
        '--device', args.device,
    ]
    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    log.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Training finished\n')
    log.close()


if __name__ == '__main__':
    main()
