# -*- coding: utf-8 -*-
"""TileConvNet 的 Flax (linen) 移植，数学上与 algo/nn/model.py 的 PyTorch 版完全一致。

实现说明（flax 0.12.7，选用 linen API——0.12 中 linen 为稳定 API，nnx 仍在演进）：
- JAX Conv 输入布局为 (B, L, C)，PyTorch Conv1d 为 (B, C, L)；这里内部做 reshape/transpose。
- linen ``nn.Conv(padding=1)`` 为两侧对称补 1，与 PyTorch ``padding=1`` 一致
  （两者都是 cross-correlation，无需翻转 kernel）。
- linen ``nn.GroupNorm`` 默认 ``reduction_axes=None`` 时对 (L, C) 归约（按 group 分组），
  与 PyTorch GroupNorm 对每个 (N, G) 在 (C/g × L) 上算均值/方差一致；epsilon 显式设为
  1e-5（PyTorch 默认值；flax 默认是 1e-6，必须覆盖）。GroupNorm 无 running stats，
  train/eval 数学一致，纯函数无需 collections。
- group 数计算与 ``algo/nn/model.py::_gn`` 相同：ch%8==0→8，ch%4==0→4，否则 1。
- 只实现当前 best 配置用到的 heads（policy/value/dealin/tenpai/response）；
  SE/attn/candidate_value/wait_dist/wait_dist3/defensive 未移植（config 全关）。
- PyTorch 里的输出层零初始化无关紧要（加载转换后的训练权重）。
"""

import flax.linen as nn
import jax.numpy as jnp


def gn_groups(ch: int) -> int:
    """与 algo/nn/model.py::_gn 相同的 group 数计算。"""
    if ch % 8 == 0:
        return 8
    if ch % 4 == 0:
        return 4
    return 1


class ResBlock1d(nn.Module):
    """对应 PyTorch _ResBlock1d（无 SE）。输入/输出 (B, L, C)。"""

    ch: int

    @nn.compact
    def __call__(self, x):
        g = gn_groups(self.ch)
        h = nn.Conv(self.ch, (3,), padding=1, name='c1')(x)
        h = nn.GroupNorm(num_groups=g, epsilon=1e-5, name='b1')(h)
        h = nn.relu(h)
        h = nn.Conv(self.ch, (3,), padding=1, name='c2')(h)
        h = nn.GroupNorm(num_groups=g, epsilon=1e-5, name='b2')(h)
        return nn.relu(x + h)


class TileConvNetFlax(nn.Module):
    """对应 PyTorch TileConvNet。输入 (B, 175)，输出 dict of heads。

    前 n_tile_ch*tile_len=170 维 reshape 成 (B, 5, 34)→transpose 为 (B, 34, 5) 过
    Conv stem + ResBlocks；后 5 维为全局标量。gfeat = [mean_L(h), max_L(h), glob]。
    """

    input_dim: int = 175
    channels: int = 128
    n_blocks: int = 6
    hidden_dim: int = 512
    n_tile_ch: int = 5
    tile_len: int = 34
    dealin_head: bool = True
    tenpai_head: bool = True
    response_head: bool = True

    @nn.compact
    def __call__(self, x):
        B = x.shape[0]
        n_tile = self.n_tile_ch * self.tile_len
        # 与 PyTorch reshape(B, 5, 34) 相同的元素顺序，再转成 JAX conv 布局 (B, L, C)
        tiles = x[:, :n_tile].reshape(B, self.n_tile_ch, self.tile_len)
        tiles = tiles.transpose(0, 2, 1)  # (B, L=34, C=5)
        glob = x[:, n_tile:]              # (B, 5)

        g = gn_groups(self.channels)
        h = nn.Conv(self.channels, (3,), padding=1, name='stem')(tiles)
        h = nn.GroupNorm(num_groups=g, epsilon=1e-5, name='stem_bn')(h)
        h = nn.relu(h)
        for i in range(self.n_blocks):
            h = ResBlock1d(self.channels, name=f'blocks_{i}')(h)

        # (B, L, C) 沿 L 池化
        gfeat = jnp.concatenate([h.mean(axis=1), h.max(axis=1), glob], axis=1)

        out = {}
        p = nn.Conv(1, (1,), padding=0, name='policy_conv')(h)[:, :, 0]
        out['policy'] = p + nn.Dense(34, name='policy_glob')(gfeat)
        v = nn.relu(nn.Dense(self.hidden_dim, name='value_fc')(gfeat))
        out['value'] = jnp.tanh(nn.Dense(1, name='value_head')(v))
        if self.dealin_head:
            d = nn.Conv(1, (1,), padding=0, name='dealin_conv')(h)[:, :, 0]
            out['dealin'] = d + nn.Dense(34, name='dealin_glob')(gfeat)
        if self.tenpai_head:
            t = nn.relu(nn.Dense(self.hidden_dim // 2, name='tenpai_fc')(gfeat))
            out['tenpai'] = nn.Dense(1, name='tenpai_head')(t)
        if self.response_head:
            r = nn.relu(nn.Dense(self.hidden_dim // 2, name='response_fc')(gfeat))
            out['response'] = nn.Dense(4, name='response_head')(r)
        return out


def build_model_flax(config: dict) -> TileConvNetFlax:
    """按 output/nn_full_action_best_config.json 的 config 构造 Flax 模型。"""
    assert config.get('arch', 'conv') == 'conv', 'only conv arch supported'
    for k in ('se_ratio', 'attn_heads', 'attn_layers', 'candidate_value_head',
              'wait_dist_head', 'wait_dist3_head', 'defensive_head'):
        assert not config.get(k, 0), f'{k} not supported by TileConvNetFlax'
    return TileConvNetFlax(
        input_dim=config.get('input_dim', 175),
        channels=config.get('channels', 128),
        n_blocks=config.get('n_blocks', 6),
        hidden_dim=config.get('hidden_dim', 512),
        n_tile_ch=config.get('n_tile_ch', 5),
        dealin_head=config.get('dealin_head', False),
        tenpai_head=config.get('tenpai_head', False),
        response_head=config.get('response_head', False),
    )
