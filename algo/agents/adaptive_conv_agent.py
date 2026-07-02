# -*- coding: utf-8 -*-
"""Runtime Policy Adaptation（pMCPA）agent。

每局开始时，用当前 base conv-BC policy 快速跑 K 局 self-play，
收集轨迹后小步微调 policy head（只改 policy_conv + policy_glob），
然后在本局使用微调后的 policy。

利用 conv-BC ~1 ms/步的速度：K=64 局 × ~20 步 × 4 玩家 ≈ 5 秒 adaptation，
仍远快于传统搜索 agent 的 150–350 ms/步。
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from algo.agents.ppo_agent import PPOAgent, _load_net
from algo.rl.selfplay import PPOActorAgent
from algo.nn.features import extract_features, extract_features_ext
import tile_pool


def _clone_policy_head(state_dict):
    """提取并克隆 policy head 权重。"""
    return {k: state_dict[k].clone() for k in state_dict
            if 'policy_conv' in k or 'policy_glob' in k}


class AdaptiveConvAgent(PPOAgent):
    """每局开头做 online self-play 微调的 conv-BC agent。"""

    def __init__(self, name, model_path='output/nn_conv_bc.pt', device='cpu',
                 temperature=0.0, verbose=False,
                 n_adapt_games=None, adapt_epochs=None, adapt_lr=None,
                 adapt_batch_size=None, win_weight=None):
        super().__init__(name, model_path=model_path, device=device,
                         temperature=temperature, verbose=verbose)
        def _env_int(key, default):
            v = os.environ.get(key)
            return int(v) if v is not None else default
        def _env_float(key, default):
            v = os.environ.get(key)
            return float(v) if v is not None else default
        self.n_adapt_games = n_adapt_games if n_adapt_games is not None else _env_int('ADAPT_N_GAMES', 64)
        self.adapt_epochs = adapt_epochs if adapt_epochs is not None else _env_int('ADAPT_EPOCHS', 3)
        self.adapt_lr = adapt_lr if adapt_lr is not None else _env_float('ADAPT_LR', 1e-3)
        self.adapt_batch_size = adapt_batch_size if adapt_batch_size is not None else _env_int('ADAPT_BATCH_SIZE', 256)
        self.win_weight = win_weight if win_weight is not None else _env_float('ADAPT_WIN_WEIGHT', 2.0)
        self._base_policy_state = None
        self._cfg = None

    def _net_obj(self):
        if self._net is None:
            self._net, self._cfg = _load_net(self.model_path, self.device)
            self._extract = extract_features_ext if self._cfg.get('features') == 'ext' else extract_features
        return self._net

    def init_tiles(self, l):
        # 先调用父类重置 context
        super().init_tiles(l)
        # 每局开始时重置到 base policy，再执行 adaptation
        self._reset_to_base()
        self._online_adapt()

    def _reset_to_base(self):
        """把 policy head 恢复成 base 权重。"""
        if self._base_policy_state is None:
            net = self._net_obj()
            self._base_policy_state = _clone_policy_head(net.state_dict())
        else:
            net = self._net_obj()
            state = net.state_dict()
            for k, v in self._base_policy_state.items():
                state[k].copy_(v)

    def _online_adapt(self):
        """跑 K 局 self-play 并微调 policy head。"""
        net = self._net_obj()
        # 固定其它层，只训 policy head
        for param in net.parameters():
            param.requires_grad = False
        for name, param in net.named_parameters():
            if 'policy_conv' in name or 'policy_glob' in name:
                param.requires_grad = True

        feats, actions, weights = self._collect_adaptation_data(net)
        if len(feats) == 0:
            self._restore_grad()
            return

        # 转成 tensor
        X = torch.from_numpy(np.stack(feats)).to(self.device)
        y = torch.from_numpy(np.array(actions, dtype=np.int64)).to(self.device)
        w = torch.from_numpy(np.array(weights, dtype=np.float32)).to(self.device)
        w = w / w.mean()  # 归一化到平均 1

        opt = torch.optim.Adam(
            [p for p in net.parameters() if p.requires_grad],
            lr=self.adapt_lr, weight_decay=1e-4)

        n = X.shape[0]
        net.train()
        for ep in range(self.adapt_epochs):
            perm = torch.randperm(n)
            for s in range(0, n, self.adapt_batch_size):
                idx = perm[s:s + self.adapt_batch_size]
                logits, _ = net(X[idx])
                loss = F.cross_entropy(logits, y[idx], reduction='none')
                loss = (loss * w[idx]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()

        self._restore_grad()

    def _restore_grad(self):
        """恢复所有参数可训练（供后续 reset 使用）。"""
        net = self._net_obj()
        for param in net.parameters():
            param.requires_grad = True

    def _collect_adaptation_data(self, net):
        """跑 K 局 self-play，固定本座位初始手牌，收集 (feature, action, weight)。"""
        import tile as tile_mod
        from driver import engine

        # 从 name 解析座位，如 'Adapt-convBC@2' -> 2
        seat = 0
        if '@' in self.name:
            try:
                seat = int(self.name.split('@')[-1])
            except ValueError:
                seat = 0
        hand = sorted(self.cur)  # init_tiles 后当前手牌

        feats, actions, weights = [], [], []
        base_seed = random.randint(0, 1 << 30)
        for i in range(self.n_adapt_games):
            # 创建 custom pool，让 seat 拿到固定手牌
            pool_cls = self._make_fixed_seat_pool(hand, seat, base_seed + i)
            # 创建 4 个 agent，本座用 PPOActorAgent 记录轨迹，其它用 base policy
            agents = []
            for s in range(4):
                name = f'Adapt@{s}'
                if s == seat:
                    ag = PPOActorAgent(name, net, device=self.device,
                                       deterministic=False, temperature=1.0,
                                       record=True, verbose=False)
                else:
                    ag = PPOActorAgent(name, net, device=self.device,
                                       deterministic=False, temperature=1.0,
                                       record=False, verbose=False)
                agents.append(ag)
            result = engine.play_game(agents, tile_pool_cls=pool_cls, verbose=False)
            # 只收集本座轨迹
            target = agents[seat]
            reward = 1.0 if result.get('winner') == target.name else (
                -1.0 if result.get('win_type') != 'draw' else 0.0)
            sample_weight = self.win_weight if reward > 0.5 else 1.0
            for step in target.traj:
                feats.append(step['feat'])
                actions.append(step['action'])
                weights.append(sample_weight)
        return feats, actions, weights

    def _make_fixed_seat_pool(self, hand, seat, seed):
        """返回一个 tile_pool.Pool 子类，保证 seat 拿到 hand。"""
        import tile as tile_mod
        import random as _random

        class FixedSeatPool(tile_pool.Pool):
            def __init__(self):
                _random.seed(seed)
                all_t = list(tile_mod.all_tiles())
                remaining = list(all_t)
                for t in hand:
                    remaining.remove(t)
                _random.shuffle(remaining)
                hands = []
                for s in range(4):
                    if s == seat:
                        hands.append(list(hand))
                    else:
                        hands.append(remaining[:13])
                        remaining = remaining[13:]
                rest = list(remaining)
                _random.shuffle(rest)
                self.tiles = []
                for s in range(4):
                    self.tiles.extend(hands[s])
                self.tiles.extend(rest)
                self.idx = 0
        return FixedSeatPool
