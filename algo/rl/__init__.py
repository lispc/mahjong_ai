# -*- coding: utf-8 -*-
"""晋北麻将 PPO 自对弈强化学习管线（方案 B）。

模块：
- reward:   终局 result -> 每座位标量奖励
- selfplay: PPOActorAgent（NN policy 采样 + 轨迹记录）+ 自对弈对局runner
"""
