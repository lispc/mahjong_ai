# -*- coding: utf-8 -*-
"""方案 B：信念状态 + Expectimax Agent。

核心思想：
- 维护全局 tile-level 信念（context_v3 中的已见牌、每家弃牌序列、报听信息）。
- 使用项目已有的 2-ply expectimax 评估器 `algo.eval2` 作为叶子估值，
  但把概率分布替换为 ContextV3 维护的真实剩余分布。
- 弃牌时先用 `algo.eval0` 快速预选，再用 `algo.eval2` 精确评估；
  只在检测到对手听牌信号时才用危险度做安全 tie-breaking，避免过度防守。

这是一个"把不完全信息放进概率分布，再在前向搜索里取期望"的干净实现。
"""

import agent
import tile
import algo
import context as ctx_module
import algo.eval.opponent as opponent
import algo.context.v3 as context_v3
import algo.eval.v2 as eval_v2


class BeliefExpectimaxAgent(agent.Agent):
    """
    信念 Expectimax Agent（方案 B）。

    参数：
        max_candidates: eval0 预选后进入 eval2 精确评估的候选数。
        defense_margin: safe_mode 下，允许为安全让步的进攻分数比例。
        tenpai_min_wait: 报听所需的最小待牌剩余张数。
    """

    def __init__(self, name, verbose=False,
                 max_candidates=8,
                 defense_margin=0.03,
                 tenpai_min_wait=4):
        super().__init__(name, verbose)
        self.max_candidates = max_candidates
        self.defense_margin = defense_margin
        self.tenpai_min_wait = tenpai_min_wait
        self.context = context_v3.ContextV3()

    def init_tiles(self, l):
        super().init_tiles(l)
        self.context = context_v3.ContextV3()

    def handle_msg(self, msg):
        if msg.type == 'put':
            self.context.see_tile(msg.data, msg.sender)
        elif msg.type == 'tenpai':
            self.context.declare_tenpai(msg.sender)
        return super().handle_msg(msg)

    def _legacy_context(self):
        """把 ContextV3 的信念映射成 legacy context.Context（algo.eval2 所需）。"""
        c = ctx_module.Context()
        c.used = self.context.used.copy()
        return c

    def declare_tenpai(self, hand, context):
        """听牌且待牌足够好时才报听。"""
        if context is None:
            return False
        # 避免过早锁死
        if sum(len(v) for v in context.discards.values()) < 12:
            return False
        if eval_v2.shanten(hand) != 0:
            return False
        remaining = context.remaining_wall(hand)
        waits = eval_v2.winning_tiles(hand, remaining)
        if not waits:
            return False
        total_wait = sum(remaining.get(t, 0) for t in waits)
        if total_wait >= self.tenpai_min_wait:
            return True
        # 有现物待牌也可报听
        for t in waits:
            if context.all_seen.get(t, 0) > 0 and remaining.get(t, 0) > 0:
                return True
        return False

    def _unique_tiles(self, hand):
        seen = set()
        out = []
        for t in hand:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _remove_one(self, hand, tile_value):
        hand = list(hand)
        hand.remove(tile_value)
        return hand

    def _danger_signal(self):
        """是否需要进入安全模式：有对手报听或某对手危险等级 >= 1。"""
        if self.context.tenpai_players - {self.name}:
            return True
        for player, discards in self.context.discards.items():
            if player == self.name:
                continue
            if opponent.player_danger_level(discards) >= 1:
                return True
        return False

    def next(self):
        assert len(self.cur) == 14

        type_ctx = self._legacy_context()
        candidates = self._unique_tiles(self.cur)

        # 1) 用 eval0 快速预选 top 候选。
        scored = []
        for disc in candidates:
            hand13 = self._remove_one(self.cur, disc)
            score = algo.eval0(hand13, type_ctx)
            scored.append((score, disc))
        scored.sort(reverse=True)
        top = [disc for _, disc in scored[:self.max_candidates]]

        # 2) 对 top 候选用 eval2 精确评估进攻价值。
        evaluated = []
        for disc in top:
            hand13 = self._remove_one(self.cur, disc)
            offense = algo.eval2(hand13, type_ctx)
            danger = opponent.tile_danger(disc, self.context, self.name)
            evaluated.append((offense, danger, disc))

        best_offense = max(item[0] for item in evaluated)

        # 3) 安全 tie-breaking：只在有危险信号时，在 best_offense 附近选危险最低的。
        if self._danger_signal():
            margin = self.defense_margin + 0.02 * len(
                self.context.tenpai_players - {self.name})
            safe_candidates = [
                item for item in evaluated if item[0] >= best_offense - margin
            ]
            safe_candidates.sort(key=lambda x: x[1])
            result = safe_candidates[0][2]
        else:
            # 无危险信号：直接选进攻分最高者。
            evaluated.sort(reverse=True, key=lambda x: x[0])
            result = evaluated[0][2]

        self.cur.remove(result)
        self.context.see_tile(result, self.name)
        if self.verbose:
            print('出牌:' + tile.tile_to_str(result))
        return result
