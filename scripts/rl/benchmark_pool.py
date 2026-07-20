# -*- coding: utf-8 -*-
"""通用 pool benchmark：把任意 4 个 agent 放同一 tournament，避免跨 run 的 Elo 漂移。

座位通过环境变量 SEATS 指定，逗号分隔，每个 token：
    baseline | beliefexp | v3nnpc
    | ppo:<label>:<model_path>
    | defensive:<label>:<model_path>
    | oppdef:<label>:<model_path>:<opp_model_path>
    | danger:<label>:<model_path>:<danger_model_path>
    | hybrid:<label>:<model_path>[:<belief_kind>]
    | hybridopp:<label>:<model_path>[:<opp_model_path>]
    | hybridwait:<label>:<model_path>[:<wait_threshold>]
    | hybridfilter:<label>:<model_path>[:<filter_kind>]
    | waitdef:<label>:<model_path>
    | exactdef:<label>:<model_path>
    | beend:<label>[:<wait_model_path>]
    | bewait:<label>[:<wait_model_path>]
    | gumbel:<label>:<model_path>[:<k>:<draws>]

环境变量：
    DEALIN_BETA          defensive/oppdef/waitdef/exactdef/hybridfilter 点炮惩罚系数（默认 2.0）
    WAIT_BETA            waitdef/hybridfilter 待牌概率放大系数（默认 0.5）
    DEF_BETA             exactdef/hybridfilter defensive EV 放大系数（默认 1.0）
    WAIT_MODEL_PATH      beend / waitdef / hybridwait 默认待牌模型路径
    HYBRID_FILTER_KIND   hybridfilter 启用哪种 filter（wait/def/both，默认 both）
    WAIT_TENPAI_THRESHOLD hybridwait 触发 BeliefExp 的待牌阈值（默认 0.5）
    OPP_BETA             oppdef 听牌概率放大系数（默认 2.0）
    DANGER_BETA          danger 模型惩罚系数（默认 2.0）
    OPP_TENPAI_THRESHOLD hybridopp 触发 BeliefExp 的听牌阈值（默认 0.5）
    OPP_MODEL_PATH       oppdef/hybridopp 默认对手模型路径
    DANGER_MODEL_PATH    danger 默认 danger 模型路径

例：
    SEATS="oppdef:opp:output/nn_full_action_best.pt:output/opponent_model.pt,\
hybridopp:hybo:output/nn_full_action_best.pt:output/opponent_model.pt,\
baseline,beliefexp" \
    PYTHONPATH=. python3 scripts/rl/benchmark_pool.py 400 32
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.belief_expectimax import BeliefExpectimaxAgent
from algo.agents.belief_expectimax_v3 import BeliefExpectimaxV3Agent
from algo.agents.ppo_agent import PPOAgent
from algo.agents.safety_aware_ppo_agent import SafetyAwarePPOAgent
from driver.tournament import run_tournament
from checker.report import compute_metrics, compute_elo

DEFAULT_SEATS = ('ppo:v1:output/nn_rl_ppo_selfplay_v1.pt,'
                 'ppo:A:output/nn_rl_ppo_A.pt,'
                 'ppo:C:output/nn_rl_ppo_C.pt,'
                 'beliefexp')


class AgentFactory:
    """picklable 工厂（ProcessPoolExecutor 需要顶层可 pickle 对象）。"""
    def __init__(self, kind, label=None, path=None):
        self.kind = kind
        self.label = label
        self.path = path

    def __call__(self):
        if self.kind == 'baseline':
            return agent.Agent('Baseline', verbose=False)
        if self.kind == 'baselinepc06':
            from algo.agents.baseline_pair_coef_agent import BaselinePairCoefAgent
            return BaselinePairCoefAgent('BaselinePC06', pair_coef=0.6, verbose=False)
        if self.kind == 'beliefexp':
            return BeliefExpectimaxAgent('BeliefExp', verbose=False)
        if self.kind == 'beliefexpused':
            return BeliefExpectimaxAgent('BeliefExpUsed', verbose=False,
                                         used_aware_eval2=True)
        if self.kind == 'beliefexp-fast2':
            return BeliefExpectimaxAgent('BeliefExp-fast2', verbose=False,
                                         eval_backend='fast2')
        if self.kind == 'be-nn':
            k = int(self.label)
            return BeliefExpectimaxAgent(f'BE-NN-{k}', verbose=False,
                                         nn_model_path=self.path, nn_top_k=k, device='cpu')
        if self.kind == 'v3nnpc':
            return BeliefExpectimaxV3Agent('V3-NN-PC', expectimax_depth=1,
                                           max_candidates=5, leaf_evaluator='nn',
                                           candidate_policy='nn')
        if self.kind == 'v3nnpck':
            k = int(self.label)
            return BeliefExpectimaxV3Agent(f'V3-NN-PC{k}', expectimax_depth=1,
                                           max_candidates=k, leaf_evaluator='nn',
                                           candidate_policy='nn')
        if self.kind == 'ppo':
            return PPOAgent(f'PPO-{self.label}', model_path=self.path,
                            device='cpu', temperature=0.0)
        if self.kind == 'autohu':
            from algo.agents.auto_hu_ppo_agent import AutoHuPPOAgent
            return AutoHuPPOAgent(f'AutoHu-{self.label}', model_path=self.path,
                                  device='cpu', temperature=0.0)
        if self.kind == 'adapt':
            from algo.agents.adaptive_conv_agent import AdaptiveConvAgent
            return AdaptiveConvAgent(f'Adapt-{self.label}', model_path=self.path,
                                     device='cpu', temperature=0.0)
        if self.kind == 'mctsconv':
            from algo.agents.mcts_conv_agent import MCTSConvAgent
            return MCTSConvAgent(f'MCTSconv-{self.label}', model_path=self.path,
                                 device='cpu')
        if self.kind == 'defensive':
            from algo.agents.defensive_conv_agent import DefensiveConvAgent
            return DefensiveConvAgent(f'Def-{self.label}', model_path=self.path,
                                      device='cpu', temperature=0.0)
        if self.kind == 'oppdef':
            from algo.agents.opp_defensive_agent import OppDefensiveAgent
            # path 格式：model_path:opp_model_path（opp 默认 output/opponent_model.pt）
            if ':' in self.path:
                model_path, opp_path = self.path.split(':', 1)
            else:
                model_path, opp_path = self.path, None
            return OppDefensiveAgent(f'OppDef-{self.label}', model_path=model_path,
                                     opp_model_path=opp_path,
                                     device='cpu', temperature=0.0)
        if self.kind == 'danger':
            from algo.agents.danger_aware_agent import DangerAwareAgent
            # path 格式：model_path:danger_model_path
            if ':' in self.path:
                model_path, danger_path = self.path.split(':', 1)
            else:
                model_path, danger_path = self.path, None
            return DangerAwareAgent(f'Danger-{self.label}', model_path=model_path,
                                    danger_model_path=danger_path,
                                    device='cpu', temperature=0.0)
        if self.kind == 'hybrid':
            from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent
            # path 格式：model_path:belief_kind（默认 beliefexp）
            if ':' in self.path:
                model_path, belief_kind = self.path.split(':', 1)
            else:
                model_path, belief_kind = self.path, 'beliefexp'
            return HybridNNBeliefAgent(f'Hybrid-{self.label}', nn_model_path=model_path,
                                       belief_kind=belief_kind, device='cpu',
                                       temperature=0.0)
        if self.kind == 'hybridt':
            # path 格式：model_path:belief_kind:tenpai_threshold
            from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent
            model_path, belief_kind, thr = self.path.rsplit(':', 2)
            return HybridNNBeliefAgent(f'HybridT-{self.label}', nn_model_path=model_path,
                                       belief_kind=belief_kind, device='cpu',
                                       temperature=0.0, tenpai_threshold=int(thr))
        if self.kind == 'hybridfix':
            from algo.agents.hybrid_nn_belief_tenpaifix_agent import \
                HybridNNBeliefTenpaiFixAgent
            return HybridNNBeliefTenpaiFixAgent(
                f'HybridFix-{self.label}', nn_model_path=self.path,
                belief_kind='beliefexp', device='cpu', temperature=0.0)
        if self.kind == 'hybridnm':
            from algo.agents.hybrid_nn_belief_nomeld_agent import \
                HybridNNBeliefNoMeldAgent
            return HybridNNBeliefNoMeldAgent(
                f'HybridNM-{self.label}', nn_model_path=self.path,
                belief_kind='beliefexp', device='cpu', temperature=0.0)
        if self.kind == 'hybridopp':
            from algo.agents.hybrid_nn_belief_opp_agent import HybridNNBeliefOppAgent
            # path 格式：model_path:opp_model_path（opp 默认 output/opponent_model.pt）
            if ':' in self.path:
                model_path, opp_path = self.path.split(':', 1)
            else:
                model_path, opp_path = self.path, None
            return HybridNNBeliefOppAgent(f'HybridOpp-{self.label}', nn_model_path=model_path,
                                          opp_model_path=opp_path, device='cpu',
                                          temperature=0.0)
        if self.kind == 'waitdef':
            from algo.agents.wait_dist_defensive_agent import WaitDistDefensiveAgent
            return WaitDistDefensiveAgent(f'WaitDef-{self.label}', model_path=self.path,
                                          device='cpu', temperature=0.0)
        if self.kind == 'hybridwait':
            from algo.agents.hybrid_nn_belief_waitdist_agent import HybridNNBeliefWaitDistAgent
            # path 格式：model_path:wait_threshold（默认 0.5）
            if ':' in self.path:
                model_path, wait_thr = self.path.split(':', 1)
                wait_thr = float(wait_thr)
            else:
                model_path, wait_thr = self.path, None
            return HybridNNBeliefWaitDistAgent(f'HybridWait-{self.label}', nn_model_path=model_path,
                                               wait_threshold=wait_thr, device='cpu',
                                               temperature=0.0)
        if self.kind == 'exactdef':
            from algo.agents.exact_defensive_agent import ExactDefensiveAgent
            return ExactDefensiveAgent(f'ExactDef-{self.label}', model_path=self.path,
                                       device='cpu', temperature=0.0)
        if self.kind == 'exactend':
            from algo.agents.exact_endgame_defensive_agent import ExactEndgameDefensiveAgent
            return ExactEndgameDefensiveAgent(f'ExactEnd-{self.label}', model_path=self.path,
                                              device='cpu', temperature=0.0)
        if self.kind == 'beend':
            from algo.agents.belief_endgame_agent import BeliefEndgameAgent
            # path 格式：可选 wait_model_path
            wait_path = self.path if self.path else None
            return BeliefEndgameAgent(f'BEEnd-{self.label}', verbose=False,
                                      wait_model_path=wait_path,
                                      max_candidates=8, defense_margin=0.03)
        if self.kind == 'besilent':
            from algo.agents.belief_silent_guard_agent import BeliefSilentGuardAgent
            # path 格式：可选 seq_model_path
            seq_path = self.path if self.path else None
            return BeliefSilentGuardAgent(f'BESilent-{self.label}', verbose=False,
                                          seq_model_path=seq_path,
                                          max_candidates=8, defense_margin=0.03)
        if self.kind == 'bewait':
            from algo.agents.belief_waitdist_agent import BeliefWaitDistAgent
            wait_path = self.path if self.path else None
            return BeliefWaitDistAgent(f'BEWait-{self.label}', verbose=False,
                                       wait_model_path=wait_path,
                                       max_candidates=8, defense_margin=0.03)
        if self.kind == 'hybridfilter':
            from algo.agents.hybrid_nn_belief_filter_agent import HybridNNBeliefFilterAgent
            # path 格式：model_path:filter_kind（默认 both）
            if ':' in self.path:
                model_path, filter_kind = self.path.split(':', 1)
            else:
                model_path, filter_kind = self.path, None
            return HybridNNBeliefFilterAgent(f'HybridFilter-{self.label}', nn_model_path=model_path,
                                             filter_kind=filter_kind, device='cpu',
                                             temperature=0.0)
        if self.kind == 'hybridend':
            from algo.agents.hybrid_nn_belief_endgame_agent import HybridNNBeliefEndgameAgent
            # path 格式：model_path[:wait_model_path]
            if ':' in self.path:
                model_path, wait_path = self.path.split(':', 1)
            else:
                model_path, wait_path = self.path, None
            return HybridNNBeliefEndgameAgent(f'HybridEnd-{self.label}', nn_model_path=model_path,
                                              wait_model_path=wait_path, device='cpu',
                                              temperature=0.0)
        if self.kind == 'hybridsilent':
            from algo.agents.hybrid_nn_besilent_agent import HybridNNBesilentAgent
            # path 格式：model_path[:seq_model_path]
            if ':' in self.path:
                model_path, seq_path = self.path.split(':', 1)
            else:
                model_path, seq_path = self.path, None
            return HybridNNBesilentAgent(f'HybridSil-{self.label}', nn_model_path=model_path,
                                         seq_model_path=seq_path, device='cpu',
                                         temperature=0.0)
        if self.kind == 'safetenpai':
            return SafetyAwarePPOAgent(f'SafeTenpai-{self.label}', model_path=self.path,
                                       device='cpu', temperature=0.0)
        if self.kind == 'hybridsafe':
            from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent
            if ':' in self.path:
                model_path, belief_kind = self.path.split(':', 1)
            else:
                model_path, belief_kind = self.path, 'beliefexp'
            return HybridNNBeliefAgent(f'HybridSafe-{self.label}', nn_model_path=model_path,
                                       belief_kind=belief_kind, device='cpu',
                                       temperature=0.0, nn_agent_class=SafetyAwarePPOAgent)
        if self.kind == 'hybridheur':
            from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent
            from algo.agents.ppo_agent import HeuristicResponsePPOAgent
            if ':' in self.path:
                model_path, belief_kind = self.path.split(':', 1)
            else:
                model_path, belief_kind = self.path, 'beliefexp'
            return HybridNNBeliefAgent(f'HybridHeur-{self.label}', nn_model_path=model_path,
                                       belief_kind=belief_kind, device='cpu',
                                       temperature=0.0, nn_agent_class=HeuristicResponsePPOAgent)
        if self.kind == 'gumbel':
            from algo.agents.gumbel_search_agent import GumbelSearchAgent
            # path 格式：model_path[:k:draws]（k/draws 可选，默认 8/2）
            parts = self.path.split(':')
            model_path = parts[0]
            k = int(parts[1]) if len(parts) > 1 else None
            draws = int(parts[2]) if len(parts) > 2 else None
            return GumbelSearchAgent(f'Gumbel-{self.label}', model_path=model_path,
                                     device='cpu', k=k, n_draws=draws)
        if self.kind == 'v3deep':
            # label = "depth-leaf[-candpolicy]"（如 "2-eval0" / "2-nn" / "2-nn-bc"）
            parts = self.label.split('-')
            depth_s, leaf = parts[0], parts[1]
            cand_policy = parts[2] if len(parts) > 2 else 'nn'
            return BeliefExpectimaxV3Agent(f'V3d-{self.label}', expectimax_depth=int(depth_s),
                                           max_candidates=5, leaf_evaluator=leaf,
                                           candidate_policy=cand_policy,
                                           candidate_model_path=self.path)
        if self.kind == 'v3rlcand':
            return BeliefExpectimaxV3Agent(f'V3-RLcand-{self.label}', expectimax_depth=1,
                                           max_candidates=5, leaf_evaluator='nn',
                                           candidate_policy='nn',
                                           candidate_model_path=self.path)
        if self.kind == 'v3rlunion':
            return BeliefExpectimaxV3Agent(f'V3-RLunion-{self.label}', expectimax_depth=1,
                                           max_candidates=5, leaf_evaluator='nn',
                                           candidate_policy='nn',
                                           candidate_model_path=self.path,
                                           candidate_union=True)
        raise ValueError(self.kind)


def _make_factory(token):
    if token in ('baseline', 'beliefexp', 'v3nnpc', 'beliefexp-fast2',
                 'baselinepc06', 'beliefexpused'):
        name = {'baseline': 'Baseline', 'beliefexp': 'BeliefExp',
                'v3nnpc': 'V3-NN-PC', 'beliefexp-fast2': 'BeliefExp-fast2',
                'baselinepc06': 'BaselinePC06', 'beliefexpused': 'BeliefExpUsed'}[token]
        return AgentFactory(token), name
    if token.startswith('v3nnpck:'):
        k = token.split(':', 1)[1]
        return AgentFactory('v3nnpck', label=k), f'V3-NN-PC{k}'
    for kind, prefix in (('ppo', 'PPO-'), ('autohu', 'AutoHu-'), ('v3rlcand', 'V3-RLcand-'),
                         ('v3rlunion', 'V3-RLunion-'), ('v3deep', 'V3d-'),
                         ('adapt', 'Adapt-'), ('mctsconv', 'MCTSconv-'),
                         ('defensive', 'Def-'), ('oppdef', 'OppDef-'),
                         ('waitdef', 'WaitDef-'), ('exactdef', 'ExactDef-'),
                         ('exactend', 'ExactEnd-'),
                         ('beend', 'BEEnd-'), ('bewait', 'BEWait-'),
                         ('besilent', 'BESilent-'),
                         ('danger', 'Danger-'), ('hybrid', 'Hybrid-'),
                         ('hybridt', 'HybridT-'), ('hybridfix', 'HybridFix-'), ('hybridnm', 'HybridNM-'),
                         ('hybridopp', 'HybridOpp-'), ('hybridwait', 'HybridWait-'),
                         ('hybridfilter', 'HybridFilter-'), ('hybridend', 'HybridEnd-'),
                         ('hybridsilent', 'HybridSil-'),
                         ('hybridsafe', 'HybridSafe-'), ('hybridheur', 'HybridHeur-'),
                         ('gumbel', 'Gumbel-'),
                         ('be-nn', 'BE-NN-')):
        if token.startswith(kind + ':'):
            if kind in ('oppdef', 'hybridopp', 'danger'):
                parts = token.split(':', 3)
                if len(parts) != 4:
                    raise ValueError(f'{kind} token needs 4 parts: {token}')
                _, label, path, extra_path = parts
                return AgentFactory(kind, label=label, path=f'{path}:{extra_path}'), f'{prefix}{label}'
            if kind == 'hybridt':
                # hybridt:LABEL:MODEL_PATH:THRESHOLD（MODEL_PATH 不含冒号）
                parts = token.split(':')
                if len(parts) != 4:
                    raise ValueError(f'hybridt token needs 4 parts: {token}')
                _, label, path, thr = parts
                return AgentFactory(kind, label=label,
                                    path=f'{path}:beliefexp:{thr}'), f'{prefix}{label}'
            if kind == 'gumbel':
                # gumbel:LABEL:MODEL_PATH[:K:DRAWS]（MODEL_PATH 不含冒号）
                parts = token.split(':')
                if len(parts) not in (3, 5):
                    raise ValueError(f'gumbel token needs 3 or 5 parts: {token}')
                label = parts[1]
                path = ':'.join(parts[2:])
                return AgentFactory(kind, label=label, path=path), f'{prefix}{label}'
            if kind in ('beend', 'bewait', 'besilent'):
                parts = token.split(':', 2)
                label = parts[1]
                path = parts[2] if len(parts) > 2 else ''
                return AgentFactory(kind, label=label, path=path), f'{prefix}{label}'
            _, label, path = token.split(':', 2)
            return AgentFactory(kind, label=label, path=path), f'{prefix}{label}'
    raise ValueError(token)


def main():
    import torch
    torch.set_num_threads(1)   # 防止多进程 fork 后 torch 线程过度订阅（fork 子进程继承）
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    seats = os.environ.get('SEATS', DEFAULT_SEATS).split(',')
    factories, names = [], []
    for tok in seats:
        f, n = _make_factory(tok.strip())
        factories.append(f)
        names.append(n)
    print('seats:', names)
    print(f'Running {n_games} games with {workers} workers ...')
    t0 = time.time()
    results = run_tournament(factories, n_games=n_games, verbose=False, n_workers=workers)
    dt = time.time() - t0
    metrics = compute_metrics(results, names)
    elo = compute_elo(results, names)
    print(f'\nTotal {dt:.1f}s')
    ranked = sorted(names, key=lambda n: elo[n], reverse=True)
    for name in ranked:
        m = metrics[name]
        print('  {:10s}: win {:.1%}, self {:.1%}, ron {:.1%}, deal-in {:.1%}, '
              'draw {:.1%}, Elo {:.0f}'.format(
                  name, m['win_rate'], m['self_rate'], m['ron_rate'],
                  m['deal_in_rate'], m['draw_rate'], elo[name]))


if __name__ == '__main__':
    main()
