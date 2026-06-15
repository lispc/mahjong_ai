# -*- coding: utf-8 -*-
"""方案 B 优化版：BeliefExpectimaxAgent V2。

在 BeliefExp 基础上：
1. 把全局 `tile_danger` 换成按对手听牌信号加权的 `aggregate_danger`，
   即把 tile-level 信念细化到 per-player 风格（花色偏好 + 危险等级）。
2. 保留 eval0 预选 + eval2 精确评估 + 安全 tie-breaking 的进攻框架。
3. 不维护精确 per-player 手牌分布，仍是 tile-level 信念。
"""

import agent
import tile
import algo
import context as ctx_module
import algo.eval.opponent as opponent
import algo.context.v3 as context_v3
import algo.eval.v2 as eval_v2


class BeliefExpectimaxV2Agent(agent.Agent):
    """
    Belief Expectimax Agent V2（per-player 危险度版）。

    参数：
        max_candidates: eval0 预选后进入 eval2 精确评估的候选数。
        defense_margin: 安全 tie-breaking 的进攻分允许差距。
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
        c = ctx_module.Context()
        c.used = self.context.used.copy()
        return c

    def declare_tenpai(self, hand, context):
        if context is None:
            return False
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

    def _danger_signal(self):
        if self.context.tenpai_players - {self.name}:
            return True
        for player, discards in self.context.discards.items():
            if player == self.name:
                continue
            if opponent.player_danger_level(discards) >= 1:
                return True
        return False

    def _aggregate_danger(self, disc):
        """按对手听牌信号加权的聚合危险度。"""
        total = 0.0
        weight_sum = 0.0
        for player in self.context.discards:
            if player == self.name:
                continue
            d = opponent.tile_danger_for_player(disc, player, self.context)
            level = opponent.player_danger_level(self.context.discards.get(player, []))
            w = 1.0 + 0.5 * level
            total += d * w
            weight_sum += w
        if weight_sum == 0:
            return 0.0
        return total / weight_sum

    def next(self):
        assert len(self.cur) == 14

        type_ctx = self._legacy_context()
        candidates = self._unique_tiles(self.cur)

        # 1) eval0 预选
        scored = []
        for disc in candidates:
            hand13 = list(self.cur)
            hand13.remove(disc)
            score = algo.eval0(hand13, type_ctx)
            scored.append((score, disc))
        scored.sort(reverse=True)
        top = [disc for _, disc in scored[:self.max_candidates]]

        # 2) eval2 精确评估
        evaluated = []
        for disc in top:
            hand13 = list(self.cur)
            hand13.remove(disc)
            offense = algo.eval2(hand13, type_ctx)
            danger = self._aggregate_danger(disc)
            evaluated.append((offense, danger, disc))

        best_offense = max(item[0] for item in evaluated)

        # 3) 安全 tie-breaking：只在有危险信号时，在 best_offense 附近选危险最低的。
        if self._danger_signal():
            n_tenpai = len(self.context.tenpai_players - {self.name})
            margin = self.defense_margin + 0.02 * n_tenpai
            safe_candidates = [item for item in evaluated if item[0] >= best_offense - margin]
            safe_candidates.sort(key=lambda x: x[1])
            result = safe_candidates[0][2]
        else:
            evaluated.sort(reverse=True, key=lambda x: x[0])
            result = evaluated[0][2]

        self.cur.remove(result)
        self.context.see_tile(result, self.name)
        if self.verbose:
            print('出牌:' + tile.tile_to_str(result))
        return result
