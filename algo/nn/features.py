# -*- coding: utf-8 -*-
"""把麻将局面编码成神经网络的输入向量。"""

import numpy as np
import algo.eval.v3 as eval_v3
import algo.eval.v2 as eval_v2


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


def _suit_of_tile(tile_value):
    """返回牌的花色：0=万，1=索，2=筒，3=字。"""
    tv = int(tile_value)
    if 1 <= tv <= 9:
        return 0  # 万
    if 11 <= tv <= 19:
        return 1  # 索
    if 21 <= tv <= 29:
        return 2  # 筒
    return 3  # 字


def _suji_safe_array(ctx, self_name):
    """返回 34 维安全度估计（越高越安全）。

    简单规则：
    - 任何对手已经打过的牌 = 现物，安全度 +1.0
    - 与现物相隔 3 的牌（筋牌）安全度 +0.5
    """
    arr = np.zeros(34, dtype=np.float32)
    self_seat = _seat(self_name)
    players_by_seat = {}
    for p in set(ctx.discards.keys()) | {self_name}:
        s = _seat(p)
        if s not in players_by_seat or p == self_name:
            players_by_seat[s] = p

    seen_tiles = set()
    for s in range(4):
        if s == self_seat:
            continue
        p = players_by_seat.get(s)
        if p is None:
            continue
        for t in ctx.discards.get(p, []):
            idx = int(_TILE_TO_IDX[t])
            arr[idx] = max(arr[idx], 1.0)
            seen_tiles.add(idx)

    # 筋牌：如果 idx±3 的现物出现过，该牌相对安全
    for idx in list(seen_tiles):
        for delta in (-3, 3):
            j = idx + delta
            if 0 <= j < 34:
                # 只对同花色（非字牌）应用筋牌
                if _suit_of_tile(int(_IDX_TO_TILE[idx])) == _suit_of_tile(int(_IDX_TO_TILE[j])):
                    arr[j] = max(arr[j], 0.5)

    return arr


def _opp_suit_pref(ctx, self_name):
    """返回 12 维对手花色偏好：3 个对手 × 4 种花色比例。"""
    arr = np.zeros(12, dtype=np.float32)
    self_seat = _seat(self_name)
    players_by_seat = {}
    for p in set(ctx.discards.keys()) | {self_name}:
        s = _seat(p)
        if s not in players_by_seat or p == self_name:
            players_by_seat[s] = p

    idx = 0
    for s in range(4):
        if s == self_seat:
            continue
        p = players_by_seat.get(s)
        if p is None:
            idx += 4
            continue
        discards = ctx.discards.get(p, [])
        if not discards:
            idx += 4
            continue
        counts = [0, 0, 0, 0]
        for t in discards:
            counts[_suit_of_tile(t)] += 1
        total = sum(counts)
        for k in range(4):
            arr[idx + k] = counts[k] / total
        idx += 4

    return arr


def _self_discard_array(ctx, self_name):
    """返回 34 维自己的弃牌历史计数（归一化到 0-1）。"""
    arr = np.zeros(34, dtype=np.float32)
    for t in ctx.discards.get(self_name, []):
        arr[int(_TILE_TO_IDX[t])] += 1.0
    # 一局最多弃牌约 20 张，除以 20 归一化
    arr = np.minimum(arr / 20.0, 1.0)
    return arr


def _hand_quality_features(hand14, remaining):
    """返回手牌质量特征：向听数(1) + ukeire(1) + 待牌分布(34)。"""
    shanten = eval_v2.shanten(hand14)
    uke = eval_v2.ukeire(hand14, remaining)
    waits = eval_v2.winning_tiles(hand14, remaining)

    wait_arr = np.zeros(34, dtype=np.float32)
    for t in waits:
        wait_arr[int(_TILE_TO_IDX[t])] = remaining.get(t, 0) / 4.0

    # 向听数归一化：0~8 -> 0~1
    shanten_norm = min(float(shanten) / 8.0, 1.0)
    # ukeire 归一化：通常 0~30+，用 log 压缩
    uke_norm = min(float(uke) / 30.0, 1.0)

    return np.concatenate([
        np.array([shanten_norm, uke_norm], dtype=np.float32),
        wait_arr,
    ])


def _context_features(agent_context, current_hand14, self_name):
    """返回局面上下文特征（不含手牌），共 257 维。

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

    suji_arr = _suji_safe_array(ctx, self_name)
    suit_pref_arr = _opp_suit_pref(ctx, self_name)
    self_discard_arr = _self_discard_array(ctx, self_name)

    return np.concatenate([
        rem_arr,
        *opp_discard_arrs,
        np.array(tenpai_flags, dtype=np.float32),
        progress,
        self_discard_arr,
        suji_arr,
        suit_pref_arr,
    ])


def extract_features(agent_context, hand14, self_name):
    """
    为当前玩家生成固定长度的特征向量。

    特征维度（共 291）：
    - 当前手牌 14 张 -> 34 维（计数，已归一化到 0-4）
    - 牌山有效剩余 -> 34 维（全局剩余 / 4）
    - 三名对手的弃牌计数 -> 3 * 34 = 102 维（每名对手累计弃出的牌数 / 20）
    - 当前玩家是否已报听 -> 1 维
    - 三名对手是否已报听 -> 3 维
    - 牌局进度 -> 1 维（已打出牌数 / 84）
    - 自己的弃牌历史 -> 34 维
    - 壁牌/筋牌安全度 -> 34 维
    - 对手花色偏好 -> 12 维
    - 手牌质量：向听数(1) + ukeire(1) + 待牌分布(34) -> 36 维
    """
    hand_arr = _hand_to_array(hand14) / 4.0
    ctx_arr = _context_features(agent_context, hand14, self_name)
    remaining = agent_context.remaining_wall(hand14)
    quality_arr = _hand_quality_features(hand14, remaining)
    return np.concatenate([hand_arr, ctx_arr, quality_arr])


def tile_to_index(tile_value):
    return int(_TILE_TO_IDX[tile_value])
