# -*- coding: utf-8 -*-
"""PPO 的优势估计与轨迹展平（与网络/训练解耦，便于单测）。

奖励是终局稀疏（中间 reward=0，末步=终局奖励），γ 默认 1.0（单局 ≤30 步）。
GAE(λ) 把单个终局信号摊到每一步，配合 value baseline 降方差。
"""

import numpy as np


def compute_gae(values, terminal_reward, gamma=1.0, lam=0.95):
    """单条轨迹的 GAE。

    values: 长度 T 的 array，V(s_0..s_{T-1})（采样时旧网络给的估计）。
    terminal_reward: 终局标量奖励（只在末步获得）。
    返回 (advantages[T], returns[T])，returns = advantages + values（λ-return）。
    """
    values = np.asarray(values, dtype=np.float64)
    T = len(values)
    adv = np.zeros(T, dtype=np.float64)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        next_v = values[t + 1] if t < T - 1 else 0.0   # V(s_T)=0（终局）
        r_t = terminal_reward if t == T - 1 else 0.0
        delta = r_t + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


def flatten_trajectories(trajs, gamma=1.0, lam=0.95):
    """把 [{'steps':[...], 'reward':R}, ...] 展平成训练用 numpy 批。

    返回 dict：feats(N,D) actions(N,) old_logp(N,) masks(N,34)
              advantages(N,) returns(N,) values(N,)
    """
    feats, actions, old_logp, masks, advs, rets, vals = [], [], [], [], [], [], []
    for tr in trajs:
        steps = tr['steps']
        if not steps:
            continue
        v = np.array([s['value'] for s in steps], dtype=np.float64)
        adv, ret = compute_gae(v, tr['reward'], gamma=gamma, lam=lam)
        for i, s in enumerate(steps):
            feats.append(s['feat'])
            actions.append(s['action'])
            old_logp.append(s['logp'])
            masks.append(s['mask'])
            vals.append(s['value'])
        advs.append(adv)
        rets.append(ret)
    if not feats:
        return None
    return {
        'feats': np.asarray(feats, dtype=np.float32),
        'actions': np.asarray(actions, dtype=np.int64),
        'old_logp': np.asarray(old_logp, dtype=np.float32),
        'masks': np.asarray(masks, dtype=np.float32),
        'advantages': np.concatenate(advs).astype(np.float32),
        'returns': np.concatenate(rets).astype(np.float32),
        'values': np.asarray(vals, dtype=np.float32),
    }
