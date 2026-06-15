# -*- coding: utf-8 -*-
"""用 BeliefExp 自对弈生成 NN 训练数据。

每个样本：(features, action_index)。
数据保存为 numpy 数组到 output/nn_training_data.npz。
"""

import sys
import os
import time
import numpy as np
from multiprocessing import Manager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.agents.data_collectors import DataCollectorBeliefExp
from driver.tournament import run_tournament


class CollectorFactory:
    """Picklable factory that appends samples to a shared buffer."""
    def __init__(self, buffer):
        self.buffer = buffer

    def __call__(self):
        return DataCollectorBeliefExp('BeliefExp', verbose=False, buffer=self.buffer)


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    manager = Manager()
    buffer = manager.list()
    factory = CollectorFactory(buffer)

    factories = [factory] * 4

    start = time.time()
    print(f'Generating {n_games} games with BeliefExp self-play (workers={n_workers}) ...')
    run_tournament(factories, n_games=n_games, verbose=False, n_workers=n_workers)
    elapsed = time.time() - start
    print(f'Generated {len(buffer)} samples in {elapsed:.1f}s '
          f'({len(buffer)/elapsed:.1f} samples/s)')

    if not buffer:
        print('No samples collected!')
        return

    X = np.stack([s[0] for s in buffer])
    y = np.array([s[1] for s in buffer], dtype=np.int64)

    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'nn_training_data.npz')
    np.savez_compressed(out_path, X=X, y=y)
    print(f'Saved to {out_path}: X shape {X.shape}, y shape {y.shape}')


if __name__ == '__main__':
    main()
