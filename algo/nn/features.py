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


def _seat(name):
    return int(name.split('@')[-1]) if '@' in name else 0


def _context_features(agent_context, current_hand14, self_name):
    """返回局面上下文特征（不含手牌），共 141 维。

    兼容“当前玩家还没在 ctx.discards 里留下记录”、重名或多余记录等边界情况：
    始终按座位 0..3 重新对齐，把自己座位那一份只保留报听 flag，其余座位当作对手。
    """
    ctx = agent_context

    remaining = ctx.remaining_wall(current_hand14)
    rem_arr = np.zeros(34, dtype=np.float32)
    for idx in range(34):
        t = int(_IDX_TO_TILE[idx])
        rem_arr[idx] = remaining.get(t, 0) / 4.0

    self_seat = _seat(self_name)
    # 按座位收集已知玩家；同名冲突时优先保留 self_name
    players_by_seat = {}
    for p in set(ctx.discards.keys()) | {self_name}:
        s = _seat(p)
        if s not in players_by_seat or p == self_name:
            players_by_seat[s] = p

    opp_discard_arrs = []
    tenpai_flags = []
    for s in range(4):
        p = players_by_seat.get(s)
        if p is None:
            # 该座位玩家未知
            if s != self_seat:
                opp_discard_arrs.append(np.zeros(34, dtype=np.float32))
            tenpai_flags.append(0.0)
        elif s == self_seat:
            tenpai_flags.append(1.0 if p in ctx.tenpai_players else 0.0)
        else:
            discard_arr = _hand_to_array(ctx.discards.get(p, [])) / 20.0
            opp_discard_arrs.append(discard_arr)
            tenpai_flags.append(1.0 if p in ctx.tenpai_players else 0.0)

    progress = np.array([min(1.0, sum(len(v) for v in ctx.discards.values()) / 84.0)],
                        dtype=np.float32)

    return np.concatenate([
        rem_arr,
        *opp_discard_arrs,
        np.array(tenpai_flags, dtype=np.float32),
        progress,
    ])


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
    hand_arr = _hand_to_array(hand14) / 4.0
    ctx_arr = _context_features(agent_context, hand14, self_name)
    return np.concatenate([hand_arr, ctx_arr])


def tile_to_index(tile_value):
    return int(_TILE_TO_IDX[tile_value])
