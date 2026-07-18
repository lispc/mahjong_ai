# -*- coding: utf-8 -*-
"""From-scratch 部署形态：PPOAgent + hu 自动 + 报听恒否（plan-scratch-0718 §1）。

- respond_hu：能胡必胡（绕过 response head 的 hu 决策——训练环境同样自动）；
- declare_tenpai：恒 False（当前引擎报听只有代价没有收益，训练环境恒否）；
- 其余（弃牌、碰/杠响应）与 PPOAgent 完全一致。
"""

import algo
from algo.agents.ppo_agent import PPOAgent


class AutoHuPPOAgent(PPOAgent):
    def respond_hu(self, tile_val, context=None):
        return algo.is_succ(self.full_hand() + [tile_val])

    def declare_tenpai(self, hand, context):
        return False
