# -*- coding: utf-8 -*-
"""自对弈 + 重训练循环（带模型筛选门）。

每次迭代：
1. 用当前最强的 V3-NN leaf / NN policy 候选 agent 自己打数据；
2. 生成 MC rollout value 标签；
3. 把新数据与历史 MC 数据合并，训练 candidate policy/value 网络；
4. 让 candidate 与当前 best 分别和 baseline/BeliefExp 打 100 局，比较 Elo；
5. 只有 candidate 显著更强时才替换当前 best。
"""

import sys
import os
import time
import shutil
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.belief_expectimax import BeliefExpectimaxAgent
from algo.agents.belief_expectimax_v3 import BeliefExpectimaxV3Agent
from driver import engine
from driver.tournament import run_tournament
from checker.report import compute_metrics, compute_elo
import random


# ---------------------------------------------------------------------------
# Model file management
# ---------------------------------------------------------------------------

_MODEL_FILES = [
    ('nn_model_config.json', 'nn_model_config'),
    ('nn_model.npz', 'nn_model'),
    ('nn_value_model_mc_config.json', 'nn_value_model_mc_config'),
    ('nn_value_model_mc.npz', 'nn_value_model_mc'),
]


def _copy_model_pair(src_suffix, dst_suffix):
    """src/dst suffix: '' for default, '_best' for best, '_candidate' for candidate."""
    for fname, base in _MODEL_FILES:
        src = os.path.join('output', f'{base}{src_suffix}.npz' if 'npz' in fname else f'{base}{src_suffix}.json')
        dst = os.path.join('output', f'{base}{dst_suffix}.npz' if 'npz' in fname else f'{base}{dst_suffix}.json')
        if os.path.exists(src):
            shutil.copy2(src, dst)


def backup_best_models():
    """把当前 default 模型备份为 _best。"""
    print('Backing up current best models ...')
    _copy_model_pair('', '_best')


def save_candidate_from_default():
    """训练后 default 文件就是 candidate，复制到 _candidate。"""
    print('Saving candidate models ...')
    _copy_model_pair('', '_candidate')


def install_candidate_models():
    """把 _candidate 覆盖 default，用于评估。"""
    _copy_model_pair('_candidate', '')


def restore_best_models():
    """把 _best 恢复为 default。"""
    _copy_model_pair('_best', '')


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _outcome_for_agent(agent_name, result):
    win_type = result.get('win_type')
    if win_type == 'draw':
        return 0.0
    winner = result.get('winner')
    if agent_name == winner:
        return 1.0
    if win_type == 'ron' and result.get('dealer') == agent_name:
        return -1.0
    if win_type == 'self':
        return -1.0
    return 0.0


def _play_and_collect(seed, n_rollouts=4):
    from algo.agents.data_collectors import DataCollectorV3NN
    from algo.nn import mc_value
    random.seed(seed)
    agents = [DataCollectorV3NN('V3NN', verbose=False,
                                expectimax_depth=1, max_candidates=5)
              for _ in range(4)]
    random.shuffle(agents)
    for i, a in enumerate(agents):
        a.name = '{}@{}'.format(a.name, i)

    result = engine.play_game(agents, verbose=False, record_time=False)

    outcomes = {a.name: _outcome_for_agent(a.name, result) for a in agents}
    samples = []
    for a in agents:
        outcome = outcomes[a.name]
        for item in a.buffer:
            if n_rollouts > 0:
                mc_v = mc_value.estimate_win_rate(
                    item['context'], item['hand'], item['name'],
                    n_rollouts=n_rollouts)
            else:
                mc_v = outcome
            samples.append((item['features'], item['action'], mc_v))
    return samples


def _play_and_collect_wrapper(args):
    return _play_and_collect(*args)


def generate_self_play_data(n_games, n_workers, n_rollouts, seed_offset=0,
                            out_path='output/nn_training_data_selfplay.npz'):
    print(f'Self-play: generating {n_games} games (workers={n_workers}, '
          f'rollouts={n_rollouts}) ...')
    start = time.time()
    all_samples = []
    completed = 0
    tasks = [(seed, n_rollouts) for seed in range(seed_offset, seed_offset + n_games)]
    with Pool(n_workers) as pool:
        for samples in pool.imap_unordered(_play_and_collect_wrapper, tasks):
            all_samples.extend(samples)
            completed += 1
            if completed % 50 == 0 or completed == n_games:
                print(f'  ... {completed}/{n_games} games done, '
                      f'{len(all_samples)} samples', flush=True)
    elapsed = time.time() - start
    print(f'Generated {len(all_samples)} samples in {elapsed:.1f}s')

    if not all_samples:
        return None

    X = np.stack([s[0] for s in all_samples])
    y = np.array([s[1] for s in all_samples], dtype=np.int64)
    v = np.array([s[2] for s in all_samples], dtype=np.float32)
    np.savez_compressed(out_path, X=X, y=y, v=v)
    print(f'Saved to {out_path}: X{X.shape} y{y.shape} v{v.shape}')
    return out_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def merge_data_files(out_path, *paths):
    Xs, ys, vs = [], [], []
    for p in paths:
        if not os.path.exists(p):
            continue
        d = np.load(p)
        Xs.append(d['X'])
        ys.append(d['y'])
        vs.append(d['v'])
    if not Xs:
        return None
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    v = np.concatenate(vs)
    np.savez_compressed(out_path, X=X, y=y, v=v)
    print(f'Merged data: {out_path} X{X.shape} y{y.shape} v{v.shape}')
    return out_path


def train_candidate_models(data_path, policy_epochs=60, value_epochs=60):
    """训练 candidate 模型，结果写入 default 文件名。"""
    print('Training candidate policy-value net ...')
    ret = os.system(f'python scripts/train_nn.py {data_path} {policy_epochs} 256 0.001 256')
    if ret != 0:
        raise RuntimeError('policy-value training failed')
    print('Training candidate deep value net ...')
    ret = os.system(f'python scripts/train_value_net_mc.py {data_path} {value_epochs} 256 0.001')
    if ret != 0:
        raise RuntimeError('value net training failed')


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _make_baseline():
    return agent.Agent('Baseline', verbose=False)


def _make_beliefexp():
    return BeliefExpectimaxAgent('BeliefExp', verbose=False)


def _make_v3nn():
    return BeliefExpectimaxV3Agent('V3-NN', expectimax_depth=1,
                                   max_candidates=5, leaf_evaluator='nn')


def _make_v3nn_pc():
    return BeliefExpectimaxV3Agent('V3-NN-PC', expectimax_depth=1,
                                   max_candidates=5, leaf_evaluator='nn',
                                   candidate_policy='nn')


def evaluate_current_models(n_games=100, n_workers=4):
    """用当前 default 模型跑一个小型 benchmark，返回各 agent 指标。"""
    configs = [_make_baseline, _make_beliefexp, _make_v3nn, _make_v3nn_pc]
    names = ['Baseline', 'BeliefExp', 'V3-NN', 'V3-NN-PC']
    print(f'Evaluating current models: {n_games} games ...')
    start = time.time()
    results = run_tournament(configs, n_games=n_games, n_workers=n_workers,
                             verbose=False)
    elapsed = time.time() - start
    print(f'Evaluation done in {elapsed:.1f}s')
    metrics = compute_metrics(results, names)
    elo = compute_elo(results, names)
    summary = {}
    for name in names:
        summary[name] = {
            'win_rate': metrics[name]['win_rate'],
            'deal_in_rate': metrics[name]['deal_in_rate'],
            'elo': elo[name],
            'avg_time_ms': metrics[name]['avg_decision_time'] * 1000,
        }
        print(f"  {name}: win {summary[name]['win_rate']:.1%}, "
              f"deal-in {summary[name]['deal_in_rate']:.1%}, "
              f"Elo {summary[name]['elo']:.0f}")
    return summary


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    n_rollouts = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    loops = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    eval_games = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    elo_margin = float(sys.argv[6]) if len(sys.argv) > 6 else 20.0

    existing_mc_data = 'output/nn_training_data_mc.npz'

    for loop in range(1, loops + 1):
        print(f'\n========== Self-play loop {loop}/{loops} ==========')

        # 1) 备份当前 best
        backup_best_models()

        # 2) 生成自对弈数据
        seed_offset = (loop - 1) * n_games
        sp_path = generate_self_play_data(n_games, n_workers, n_rollouts,
                                          seed_offset=seed_offset)
        if sp_path is None:
            print('No data generated, abort.')
            restore_best_models()
            break

        # 3) 与历史 MC 数据合并训练 candidate
        merged_path = 'output/nn_training_data_merged.npz'
        merge_data_files(merged_path, existing_mc_data, sp_path)
        train_candidate_models(merged_path)

        # 4) 把 candidate 保存下来，恢复 best
        save_candidate_from_default()
        restore_best_models()

        # 5) 评估 candidate
        install_candidate_models()
        candidate_summary = evaluate_current_models(eval_games, n_workers=4)
        restore_best_models()

        # 6) 评估当前 best
        best_summary = evaluate_current_models(eval_games, n_workers=4)

        # 7) 比较：V3-NN 的 Elo
        cand_elo = candidate_summary['V3-NN']['elo']
        best_elo = best_summary['V3-NN']['elo']
        print(f'\nCandidate V3-NN Elo: {cand_elo:.0f}')
        print(f'Best    V3-NN Elo: {best_elo:.0f}')

        if cand_elo > best_elo + elo_margin:
            print(f'Candidate wins by {cand_elo - best_elo:.0f} Elo -> PROMOTE')
            install_candidate_models()
        else:
            print(f'Candidate does not beat best by {elo_margin:.0f} Elo -> KEEP BEST')

    print('\nSelf-play loop finished.')


if __name__ == '__main__':
    main()
