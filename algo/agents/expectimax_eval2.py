# -*- coding: utf-8 -*-
"""使用原项目 algo.eval2 并带入已见牌信息的 Agent。"""

import agent
import tile
import algo
import context as ctx
import algo.context.v3 as context_v3
import algo.eval.opponent as opponent_model
import algo.eval.v2 as eval_v2


def _unique_tiles(hand):
    seen = set()
    out = []
    for t in hand:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _remove_one(hand, tile_value):
    hand = list(hand)
    hand.remove(tile_value)
    return hand


def _type_context(context):
    c = ctx.Context()
    c.used = context.used.copy()
    return c


def select(hand, context, max_candidates=6, defense_weight=0.0, self_name=None,
           safe_mode=True):
    """
    先用 eval0 快速预选 top 候选弃牌，再用 algo.eval2 精确评估。
    已见牌通过 context 传入。

    若 defense_weight > 0 且 self_name 提供：
    - safe_mode=True（默认）：仅当检测到某对手可能听牌（danger_level >= 1）时，
      在 eval2 最优解附近选一个危险度最低的弃牌。
    - safe_mode=False：直接把危险度作为惩罚项加入 eval2 分值。
    """
    assert len(hand) == 14
    type_ctx = _type_context(context)
    candidates = _unique_tiles(hand)

    # 快速预选：eval0 对每个候选弃牌打分
    scored = []
    for disc in candidates:
        hand13 = _remove_one(hand, disc)
        score = algo.eval0(hand13, type_ctx)
        scored.append((score, disc))
    scored.sort(reverse=True)

    if max_candidates <= 0 or len(scored) <= max_candidates:
        top_candidates = [disc for _, disc in scored]
    else:
        top_candidates = [disc for _, disc in scored[:max_candidates]]

    # 是否有对手已报听？报听是极强的危险信号
    opponents_tenpai = (self_name is not None and
                        bool(context.tenpai_players - {self_name}))

    # 评估每个候选的进攻分和危险度
    evaluated = []
    max_danger_level = 0
    if self_name is not None and (defense_weight > 0.0 or opponents_tenpai):
        for player in context.discards:
            if player == self_name:
                continue
            lvl = opponent_model.player_danger_level(context.discards[player])
            if lvl > max_danger_level:
                max_danger_level = lvl

    for disc in top_candidates:
        hand13 = _remove_one(hand, disc)
        offense = algo.eval2(hand13, type_ctx)
        danger = 0.0
        if self_name is not None and (defense_weight > 0.0 or opponents_tenpai):
            danger = opponent_model.tile_danger(disc, context, self_name)
        evaluated.append((offense, danger, disc))

    if not evaluated:
        return candidates[0]

    best_offense = max(item[0] for item in evaluated)

    # safe_mode：在对手可能听牌或已报听时，用危险度打破平局
    if safe_mode and (max_danger_level >= 1 or opponents_tenpai):
        # 已报听玩家越多，让步幅度越大
        margin = 0.02 * max_danger_level + 0.03 * len(context.tenpai_players)
        safe_candidates = [
            item for item in evaluated if item[0] >= best_offense - margin
        ]
        # 在可接受范围内选危险度最低的
        safe_candidates.sort(key=lambda x: x[1])
        return safe_candidates[0][2]

    # 普通模式：eval2 - defense_weight * danger
    best_tile = None
    best_value = -float('inf')
    for offense, danger, disc in evaluated:
        value = offense - defense_weight * danger
        if value > best_value:
            best_value = value
            best_tile = disc
    return best_tile


class ExpectiMaxEval2Agent(agent.Agent):
    """
    核心评估使用原项目 algo.eval2（本身就是 2-ply ExpectiMax），
    但把已见牌（对手弃牌）传入 eval2 的概率计算。
    """
    def __init__(self, name, verbose=False, max_candidates=6):
        super().__init__(name, verbose)
        self.max_candidates = max_candidates
        self.context = context_v3.ContextV3()

    def init_tiles(self, l):
        super().init_tiles(l)
        self.context = context_v3.ContextV3()

    def add(self, t):
        return super().add(t)

    def handle_msg(self, msg):
        if msg.type == 'put':
            self.context.see_tile(msg.data, msg.sender)
        elif msg.type == 'tenpai':
            self.context.declare_tenpai(msg.sender)
        return super().handle_msg(msg)

    def declare_tenpai(self, hand, context):
        """
        报听决策：简单启发式。
        hand 为弃牌后的 13 张手牌。
        """
        if context is None:
            return False
        # 至少需要一定轮数，避免过早锁死
        if sum(len(v) for v in context.discards.values()) < 12:
            return False
        if eval_v2.shanten(hand) != 0:
            return False
        counts = eval_v2._count(hand)
        winning_tiles = [t for t in eval_v2.VALID_TILES
                         if counts.get(t, 0) < 4 and eval_v2.is_win(hand + [t])]
        if not winning_tiles:
            return False
        remaining = context.remaining_wall(hand)
        total_wait = sum(remaining.get(t, 0) for t in winning_tiles)
        # 待牌总剩余张数 >= 4 才报听
        if total_wait >= 4:
            return True
        # 或有现物待牌也可报听
        for t in winning_tiles:
            if context.all_seen.get(t, 0) > 0 and remaining.get(t, 0) > 0:
                return True
        return False

    def next(self):
        assert len(self.cur) == 14
        result = select(
            self.cur, self.context, self.max_candidates,
            self_name=self.name, safe_mode=True
        )
        self.cur.remove(result)
        self.context.see_tile(result, self.name)
        if self.verbose:
            print('出牌:' + tile.tile_to_str(result))
        return result


class ExpectiMaxEval2DefenseAgent(ExpectiMaxEval2Agent):
    """
    在 Eval2Ctx 基础上加入对手建模防守惩罚（B+D）。
    """
    def __init__(self, name, verbose=False, max_candidates=8, defense_weight=3.0,
                 safe_mode=True):
        super().__init__(name, verbose, max_candidates)
        self.defense_weight = defense_weight
        self.safe_mode = safe_mode

    def next(self):
        assert len(self.cur) == 14
        result = select(
            self.cur, self.context, self.max_candidates,
            defense_weight=self.defense_weight,
            self_name=self.name,
            safe_mode=self.safe_mode
        )
        self.cur.remove(result)
        self.context.see_tile(result, self.name)
        if self.verbose:
            print('出牌:' + tile.tile_to_str(result))
        return result
