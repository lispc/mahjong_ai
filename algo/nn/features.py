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
    if '@' not in name:
        return 0
    suffix = name.split('@')[-1]
    digits = ''
    for ch in suffix:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else 0


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


def extract_features_oracle(agent_context, hand14, self_name, all_hands, wall):
    """311 维 Oracle 特征 = 175 基础 + 3 对手当前手牌(102) + 完整牌山剩余(34)。

    all_hands: {player_name: hand13/14 list}，包含自己和对手；
    wall: list，牌山剩余牌（按摸牌顺序，只用集合计数）。
    """
    base = extract_features(agent_context, hand14, self_name)  # 175
    ctx = agent_context

    # 对手手牌（按座位顺序，排除自己）
    self_seat = _seat(self_name)
    seat_to_player = {}
    for p in ctx.discards:
        seat_to_player.setdefault(_seat(p), p)
    # 自己也可能在 all_hands 中，按座位补齐
    for p, h in all_hands.items():
        seat_to_player.setdefault(_seat(p), p)

    opp_hand_arrs = []
    for s in range(4):
        if s == self_seat:
            continue
        p = seat_to_player.get(s)
        hand = all_hands.get(p, [])
        opp_hand_arrs.append(_hand_to_array(hand) / 4.0)

    # 完整牌山剩余
    wall_arr = np.zeros(34, dtype=np.float32)
    for t in wall:
        wall_arr[int(_TILE_TO_IDX[t])] += 1.0
    wall_arr = wall_arr / 4.0

    return np.concatenate([base] + opp_hand_arrs + [wall_arr]).astype(np.float32)


def tile_to_index(tile_value):
    return int(_TILE_TO_IDX[tile_value])


# ---------------------------------------------------------------------------
# 扩展特征（ext, 212 维）：在 175 基础上加入 BeliefExp 用的「危险度/防守」信息，
# 让网络能"看见"打出每张牌的点炮风险。布局保持「先牌通道后标量」以适配 conv：
#   tile channels (6 * 34 = 204): 手牌 / 牌山剩余 / 3 对手弃牌 / 危险度地图
#   scalars (8): 自家报听 + 3 对手报听 + 进度 + 3 对手危险等级
# ---------------------------------------------------------------------------
import algo.eval.opponent as _opp


def _aggregate_danger(ctx, self_name, tile_value):
    """按 per-player 危险度加权聚合（与 BeliefExpV3._aggregate_danger 同构）。"""
    total = 0.0
    wsum = 0.0
    for player in ctx.discards:
        if player == self_name:
            continue
        d = _opp.tile_danger_for_player(tile_value, player, ctx)
        lvl = _opp.player_danger_level(ctx.discards.get(player, []))
        w = 1.0 + 0.5 * lvl
        total += d * w
        wsum += w
    return total / wsum if wsum > 0 else 0.0


def extract_features_ext(agent_context, hand14, self_name):
    """212 维扩展特征 = 175 基础重排 + 34 维危险度通道 + 3 维对手危险等级。"""
    base = extract_features(agent_context, hand14, self_name)   # 175
    tile_part = base[:170]          # 手牌+牌山+3对手弃牌（5*34）
    scalars = base[170:175]         # 报听4 + 进度1

    ctx = agent_context
    danger = np.zeros(34, dtype=np.float32)
    for idx in range(34):
        t = int(_IDX_TO_TILE[idx])
        danger[idx] = _aggregate_danger(ctx, self_name, t)
    danger = np.clip(danger / 2.0, 0.0, 1.0)

    self_seat = _seat(self_name)
    seat_to_player = {}
    for p in ctx.discards:
        seat_to_player.setdefault(_seat(p), p)
    dl = []
    for s in range(4):
        if s == self_seat:
            continue
        p = seat_to_player.get(s)
        lvl = _opp.player_danger_level(ctx.discards.get(p, [])) if p else 0
        dl.append(lvl / 2.0)
    dl = np.array((dl + [0.0, 0.0, 0.0])[:3], dtype=np.float32)

    return np.concatenate([tile_part, danger, scalars, dl]).astype(np.float32)
