# -*- coding: utf-8 -*-
"""轻量 Policy-Value 网络（PyTorch）。"""

import torch
import torch.nn as nn


class MahjongNet(nn.Module):
    """
    输入 175 维局面特征，输出：
    - policy_logits: 34 维（对应 34 种牌）
    - value: 1 维（当前玩家最终获胜期望，[-1, 1]）
    """

    def __init__(self, input_dim=175, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.policy_head = nn.Linear(hidden_dim // 2, 34)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h))
        return policy_logits, value


def loss_fn(model, X, y_policy, y_value, policy_weight=1.0, value_weight=0.5):
    """policy 交叉熵 + value MSE。"""
    logits, value = model(X)

    # policy loss: sparse cross entropy
    policy_loss = nn.functional.cross_entropy(logits, y_policy)

    # value loss: MSE
    value = value.squeeze(-1)
    value_loss = nn.functional.mse_loss(value, y_value)

    return policy_weight * policy_loss + value_weight * value_loss, {
        'policy_loss': policy_loss.detach(),
        'value_loss': value_loss.detach(),
    }


def _gn(ch):
    """GroupNorm（batch 无关，train/eval 一致，适配 PPO）。"""
    if ch % 8 == 0:
        g = 8
    elif ch % 4 == 0:
        g = 4
    else:
        g = 1
    return nn.GroupNorm(g, ch)


class _ResBlock1d(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, k, padding=k // 2)
        self.b1 = _gn(ch)
        self.c2 = nn.Conv1d(ch, ch, k, padding=k // 2)
        self.b2 = _gn(ch)

    def forward(self, x):
        h = torch.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return torch.relu(x + h)


class TileConvNet(nn.Module):
    """对牌结构敏感的 1D-Conv/ResNet Policy-Value 网络。

    输入沿用 175 维特征 = 5 个 34 维牌通道（手牌/牌山/3 家弃牌）+ 5 个标量
    （自家报听 + 3 对手报听 + 进度）。在 34 轴上卷积以捕捉顺子/刻子等局部牌型。
    - policy 头：1×1 卷积输出每张牌的 logit（保留位置）+ 全局上下文偏置；
    - value 头：全局池化（mean+max）+ 标量 → MLP → tanh；
    - 可选 dealin 头：结构同 policy 头，输出每张牌是否立即点炮的 logit；
    - 可选 tenpai 头：全局特征 → MLP → 1 logit，用于报听决策。
    """

    def __init__(self, input_dim=175, channels=96, n_blocks=4, hidden_dim=256,
                 n_tile_ch=5, tile_len=34, dealin_head=False, tenpai_head=False,
                 candidate_value_head=False):
        super().__init__()
        self.n_tile = n_tile_ch * tile_len            # 170
        self.n_tile_ch = n_tile_ch
        self.tile_len = tile_len
        self.n_glob = input_dim - self.n_tile         # 5
        self.use_dealin = dealin_head
        self.use_tenpai = tenpai_head
        self.use_candidate_value = candidate_value_head
        self.hidden_dim = hidden_dim
        self.stem = nn.Conv1d(n_tile_ch, channels, 3, padding=1)
        self.stem_bn = _gn(channels)
        self.blocks = nn.ModuleList([_ResBlock1d(channels) for _ in range(n_blocks)])
        gfeat_dim = 2 * channels + self.n_glob
        self.policy_conv = nn.Conv1d(channels, 1, 1)
        self.policy_glob = nn.Linear(gfeat_dim, 34)
        self.value_fc = nn.Linear(gfeat_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        if self.use_dealin:
            self.dealin_conv = nn.Conv1d(channels, 1, 1)
            self.dealin_glob = nn.Linear(gfeat_dim, 34)
        if self.use_tenpai:
            self.tenpai_fc = nn.Linear(gfeat_dim, hidden_dim // 2)
            self.tenpai_head = nn.Linear(hidden_dim // 2, 1)
        if self.use_candidate_value:
            self.cv_conv = nn.Conv1d(channels, 1, 1)
            self.cv_glob = nn.Linear(gfeat_dim, 34)
        # 输出层零初始化：初始 policy 近均匀、value≈0，避免 tanh 早期饱和崩溃
        for layer in (self.policy_conv, self.policy_glob, self.value_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        if self.use_dealin:
            nn.init.zeros_(self.dealin_conv.weight)
            nn.init.zeros_(self.dealin_conv.bias)
            nn.init.zeros_(self.dealin_glob.weight)
            nn.init.zeros_(self.dealin_glob.bias)
        if self.use_tenpai:
            nn.init.zeros_(self.tenpai_head.weight)
            nn.init.zeros_(self.tenpai_head.bias)
        if self.use_candidate_value:
            nn.init.zeros_(self.cv_conv.weight)
            nn.init.zeros_(self.cv_conv.bias)
            nn.init.zeros_(self.cv_glob.weight)
            nn.init.zeros_(self.cv_glob.bias)

    def _trunk(self, x):
        B = x.shape[0]
        tiles = x[:, :self.n_tile].reshape(B, self.n_tile_ch, self.tile_len)
        glob = x[:, self.n_tile:]
        h = torch.relu(self.stem_bn(self.stem(tiles)))
        for blk in self.blocks:
            h = blk(h)
        gfeat = torch.cat([h.mean(dim=2), h.amax(dim=2), glob], dim=1)
        return h, gfeat

    def forward(self, x):
        h, gfeat = self._trunk(x)
        policy_logits = self.policy_conv(h).squeeze(1) + self.policy_glob(gfeat)
        value = torch.tanh(self.value_head(torch.relu(self.value_fc(gfeat))))
        outs = [policy_logits, value]
        if self.use_dealin:
            dealin_logits = self.dealin_conv(h).squeeze(1) + self.dealin_glob(gfeat)
            outs.append(dealin_logits)
        if self.use_candidate_value:
            cv_logits = self.cv_conv(h).squeeze(1) + self.cv_glob(gfeat)
            outs.append(cv_logits)
        return tuple(outs)

    def tenpai_logit(self, x):
        """报听决策头：输入 175 维特征，输出 logit（>0 表示报听）。"""
        if not self.use_tenpai:
            raise RuntimeError('tenpai_logit called but tenpai_head=False')
        _, gfeat = self._trunk(x)
        return self.tenpai_head(torch.relu(self.tenpai_fc(gfeat)))


def build_model(config):
    """按 config 构造网络。config['arch'] in {'mlp'(默认), 'conv'}。"""
    arch = config.get('arch', 'mlp')
    input_dim = config.get('input_dim', 175)
    if arch == 'conv':
        return TileConvNet(input_dim=input_dim,
                           channels=config.get('channels', 96),
                           n_blocks=config.get('n_blocks', 4),
                           hidden_dim=config.get('hidden_dim', 256),
                           n_tile_ch=config.get('n_tile_ch', 5),
                           dealin_head=config.get('dealin_head', False),
                           tenpai_head=config.get('tenpai_head', False),
                           candidate_value_head=config.get('candidate_value_head', False))
    return MahjongNet(input_dim=input_dim, hidden_dim=config.get('hidden_dim', 128))
