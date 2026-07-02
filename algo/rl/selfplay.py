# -*- coding: utf-8 -*-
"""自对弈轨迹记录器：用当前 policy 网络采样打牌，记录 PPO 所需的逐步数据。

设计要点：
- PPOActorAgent 继承 BeliefExpectimaxV3Agent，只为复用其 ContextV3 上下文维护
  （handle_msg 的 see_tile / declare_tenpai）和 declare_tenpai 启发式；
  next() 被完全覆盖为「NN policy 采样 + 记录轨迹」，不走 expectimax。
- 牌用「种类值」表示（tile.all_tiles: 1..9,11..19,21..29,31..37 各 4 张），
  因此 action index a 对应的弃牌 tile 值 = _IDX_TO_TILE[a]，remove 一张即可。
- 奖励终局稀疏（γ=1），GAE 只需每步 V(s_t) + 末尾终局奖励，无需保存 s'。
"""

import random
import numpy as np
import torch

from algo.agents.belief_expectimax_v3 import BeliefExpectimaxV3Agent
from algo.nn.features import extract_features, _TILE_TO_IDX, _IDX_TO_TILE
from algo.rl.reward import seat_reward, terminal_reason

NUM_ACTIONS = 34


def build_net(state_dict, config, device='cpu'):
    """从 config + state_dict 构造网络（支持 mlp / conv 架构，eval 模式）。"""
    from algo.nn.model import build_model
    net = build_model(config)
    net.load_state_dict(state_dict)
    net.eval()
    net.to(device)
    return net


class PPOActorAgent(BeliefExpectimaxV3Agent):
    """用给定 policy-value 网络采样打牌，并记录逐步轨迹。"""

    def __init__(self, name, net, device='cpu', deterministic=False,
                 temperature=1.0, record=True, verbose=False):
        super().__init__(name, verbose=verbose)
        self.net = net
        self.device = device
        self.deterministic = deterministic
        self.temperature = temperature
        self.record = record
        self.traj = []

    def init_tiles(self, l):
        super().init_tiles(l)   # 重置 context / belief
        self.traj = []

    def _legal_mask(self):
        legal = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for t in self.cur:
            legal[int(_TILE_TO_IDX[t])] = 1.0
        return legal

    def next(self):
        assert len(self.cur) == 14
        feats = extract_features(self.context, self.cur, self.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.net(x)
            logits = out[0]
            value = out[1]
        logits = logits.squeeze(0).detach().cpu().numpy().astype(np.float64)
        value = float(value.detach().cpu().reshape(-1)[0])

        legal = self._legal_mask()
        masked = logits + (legal - 1.0) * 1e9           # 非法动作 -> -1e9
        masked = masked / max(self.temperature, 1e-6)
        masked = masked - masked.max()
        probs = np.exp(masked)
        probs = probs / probs.sum()

        if self.deterministic:
            a = int(np.argmax(probs))
        else:
            a = int(np.random.choice(NUM_ACTIONS, p=probs))
        logp = float(np.log(probs[a] + 1e-12))

        if self.record:
            self.traj.append({
                'feat': np.asarray(feats, dtype=np.float32),
                'action': a,
                'logp': logp,
                'value': value,
                'mask': legal,
            })

        tile_val = int(_IDX_TO_TILE[a])
        # tile_val 一定在手牌里（mask 保证 a 合法）
        self.cur.remove(tile_val)
        self.context.see_tile(tile_val, self.name)
        self._belief = None
        return tile_val


def _build_opponent(spec, name):
    """按 picklable spec 字符串构造对手 agent（worker 内调用）。"""
    if spec == 'baseline':
        import agent as _agent_mod
        return _agent_mod.Agent(name, verbose=False)
    if spec == 'beliefexp':
        from algo.agents.belief_expectimax import BeliefExpectimaxAgent
        return BeliefExpectimaxAgent(name, verbose=False)
    if spec == 'v3eval0':
        return BeliefExpectimaxV3Agent(name, verbose=False, expectimax_depth=1,
                                       max_candidates=5, leaf_evaluator='eval0',
                                       candidate_policy='baseline_eval1')
    raise ValueError(f'unknown opponent spec: {spec}')


def _make_agents(net, seat_specs, device, deterministic, temperature):
    """seat_specs: 长度 4 的列表，元素为 'ppo' 或对手 spec 字符串。"""
    agents = []
    for s in range(4):
        name = f'A@{s}'
        spec = seat_specs[s]
        if spec == 'ppo':
            agents.append(PPOActorAgent(name, net, device=device,
                                        deterministic=deterministic,
                                        temperature=temperature))
        else:
            agents.append(_build_opponent(spec, name))
    return agents


def play_selfplay_game(net, seed=None, reward_cfg=None, device='cpu',
                       deterministic=False, temperature=1.0, seat_specs=None):
    """打一局，返回 (trajectories, result)。

    seat_specs: 长度 4，'ppo'（学习者，记录轨迹）或对手 spec 字符串。
    trajectories: 每个记录轨迹的座位一条：
        {'steps': [...], 'reward': float, 'seat': name, 'reason': str}
    """
    from driver.engine import play_game

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2 ** 31 - 1))

    if seat_specs is None:
        seat_specs = ['ppo', 'ppo', 'ppo', 'ppo']

    agents = _make_agents(net, seat_specs, device, deterministic, temperature)
    result = play_game(agents)

    trajs = []
    for ag in agents:
        if not isinstance(ag, PPOActorAgent) or not ag.traj:
            continue
        trajs.append({
            'steps': ag.traj,
            'reward': seat_reward(result, ag.name, reward_cfg),
            'seat': ag.name,
            'reason': terminal_reason(result, ag.name),
        })
    return trajs, result


def _sample_seat_specs(rng, n_opponents, opponents):
    """随机决定 4 个座位：n_opponents 个对手座位（从 opponents 抽 spec），其余 'ppo'。"""
    specs = ['ppo', 'ppo', 'ppo', 'ppo']
    if n_opponents <= 0 or not opponents:
        return specs
    seats = list(range(4))
    rng.shuffle(seats)
    for s in seats[:min(n_opponents, 3)]:   # 至少留 1 个学习者座位
        specs[s] = rng.choice(opponents)
    return specs


def collect_games(state_dict, n_games, seed_base=0, reward_cfg=None,
                  config=None, device='cpu', temperature=1.0,
                  n_opponents=0, opponents=None, num_threads=1):
    """（可用于子进程）加载网络并跑 n_games 局，返回所有轨迹列表。

    config: 网络配置 dict（arch/input_dim/hidden_dim/...），传给 build_model。
    n_opponents>0 时，每局随机把若干座位换成 opponents 里的对手（其余为学习者）。
    """
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    net = build_net(state_dict, config, device=device)
    spec_rng = random.Random(seed_base ^ 0x9E3779B9)
    out = []
    for i in range(n_games):
        seat_specs = _sample_seat_specs(spec_rng, n_opponents, opponents)
        trajs, _ = play_selfplay_game(
            net, seed=seed_base + i, reward_cfg=reward_cfg, device=device,
            deterministic=False, temperature=temperature, seat_specs=seat_specs)
        out.extend(trajs)
    return out
