# -*- coding: utf-8 -*-
"""把麻将局面编码成神经网络的输入向量。"""

import numpy as np
import algo.eval.v3 as eval_v3


# 复用 eval_v3 的 tile -> idx 映射
_TILE_TO_IDX = eval_v3._TILE_TO_IDX
_IDX_TO_TILE = eval_v3._IDX_TO_TILE


def _hand_to_array(hand, length=34):
    arr = np.zeros(length, dtype=np.float32)
    for t in hand:
        arr[int(_TILE_TO_IDX[t])] += 1.0
    return arr


def extract_features(agent_context, hand14, self_name):
    """
    为当前玩家生成固定长度的特征向量。

    特征维度（共 175）：
    - 当前手牌 14 张 -> 34 维（计数，已归一化到 0-4）
    - 牌山有效剩余 -> 34 维（全局剩余 / 4）
    - 三名对手的弃牌计数 -> 3 * 34 = 102 维（每名对手累计弃出的牌数 / 20）
    - 当前玩家是否已报听 -> 1 维
    - 三名对手是否已报听 -> 3 维
    - 牌局进度 -> 1 维（已打出牌数 / 84）
    """
    ctx = agent_context

    hand_arr = _hand_to_array(hand14) / 4.0

    remaining = ctx.remaining_wall(hand14)
    rem_arr = np.zeros(34, dtype=np.float32)
    for idx in range(34):
        t = int(_IDX_TO_TILE[idx])
        rem_arr[idx] = remaining.get(t, 0) / 4.0

    opp_discard_arrs = []
    tenpai_flags = [1.0 if self_name in ctx.tenpai_players else 0.0]
    for player in ctx.discards:
        if player == self_name:
            continue
        discard_arr = _hand_to_array(ctx.discards[player]) / 20.0
        opp_discard_arrs.append(discard_arr)
        tenpai_flags.append(1.0 if player in ctx.tenpai_players else 0.0)

    # 如果对手不足 3 个，补零（正常情况下 4 人局不会触发）
    while len(opp_discard_arrs) < 3:
        opp_discard_arrs.append(np.zeros(34, dtype=np.float32))
    while len(tenpai_flags) < 4:
        tenpai_flags.append(0.0)

    progress = np.array([min(1.0, sum(len(v) for v in ctx.discards.values()) / 84.0)],
                        dtype=np.float32)

    features = np.concatenate([
        hand_arr,
        rem_arr,
        *opp_discard_arrs,
        np.array(tenpai_flags, dtype=np.float32),
        progress,
    ])
    return features


def tile_to_index(tile_value):
    return int(_TILE_TO_IDX[tile_value])
