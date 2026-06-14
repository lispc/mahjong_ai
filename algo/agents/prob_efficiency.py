# -*- coding: utf-8 -*-
"""方案 A：纯概率牌效 Agent。

基于 Shanten + Ukeire 做进攻，基于 tile-level 危险度做防守，
通过统一的效用函数选择弃牌。听牌时必须报听（由引擎锁死手牌）。

提供两种模式：
- 静态评估（use_expectation=False）：直接对弃牌后的 13 张手牌打分。
- 1-ply 期望（use_expectation=True）：枚举下一张摸牌，按真实剩余概率
  加权，更适合目标为胜率的对局。
"""

import agent
import tile
import algo.eval.v2 as eval_v2
import algo.eval.opponent as opponent
import algo.context.v3 as context_v3


WIN_VALUE = 100.0


class ProbEfficiencyAgent(agent.Agent):
    """
    概率牌效 Agent（方案 A）。

    进攻价值 = 基础手牌评估（eval_v2.evaluate）+ 有效进张/听牌待牌奖励
    防守价值 = tile_danger（基于现物、筋牌、对手报听信号）
    最终选择 = argmax(进攻 - lambda_def * 防守)

    参数说明：
        use_expectation:  是否使用 1-ply 期望代替静态评估
        weight_ukeire:    未听牌时，每张有效进张的额外奖励
        weight_tenpai:    听牌后，每张剩余待牌的额外奖励
        lambda_def_base:  基础防守系数
        lambda_tenpai_opponent: 每个已报听对手的额外防守系数
    """

    def __init__(self, name, verbose=False,
                 use_expectation=False,
                 weight_ukeire=0.5,
                 weight_tenpai=2.0,
                 lambda_def_base=0.5,
                 lambda_tenpai_opponent=1.5):
        super().__init__(name, verbose)
        self.use_expectation = use_expectation
        self.weight_ukeire = weight_ukeire
        self.weight_tenpai = weight_tenpai
        self.lambda_def_base = lambda_def_base
        self.lambda_tenpai_opponent = lambda_tenpai_opponent
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

    def declare_tenpai(self, hand, context):
        """听牌必须报听。"""
        return eval_v2.shanten_fast(hand) == 0

    def _wall_remaining(self):
        """牌山中还能摸到的牌数（不含他家手牌）。"""
        total_seen = sum(self.context.all_seen.values())
        return max(0, 84 - total_seen - 1)

    def _lambda_def(self):
        """动态防守系数：对手报听越多、牌山越浅，越保守。"""
        n_tenpai = len(self.context.tenpai_players - {self.name})
        progress = 1.0 - self._wall_remaining() / 84.0
        return (self.lambda_def_base +
                self.lambda_tenpai_opponent * n_tenpai +
                0.5 * progress)

    def _static_offense(self, hand, remaining):
        """对任意长度手牌（通常为 13 或 14）的静态进攻评估。"""
        base = eval_v2.evaluate(hand)
        if len(hand) == 13 and eval_v2.shanten_fast(hand) == 0:
            waits = eval_v2.winning_tiles(hand, remaining)
            wait_count = sum(remaining.get(t, 0) for t in waits)
            return base + self.weight_tenpai * wait_count
        if len(hand) == 13:
            return base + self.weight_ukeire * eval_v2.ukeire(hand, remaining)
        return base

    def _offense_value(self, hand13, discarded_tile=None):
        """计算 13 张手牌的进攻价值。

        discarded_tile 表示为了得到 hand13 而刚刚打出的牌，需要从剩余牌中扣除，
        否则评估会错误地把这张牌当成还能摸到的牌。
        """
        remaining = self.context.remaining_wall(hand13)
        if discarded_tile is not None:
            remaining = dict(remaining)
            remaining[discarded_tile] = remaining.get(discarded_tile, 0) - 1
            if remaining[discarded_tile] <= 0:
                del remaining[discarded_tile]

        if not self.use_expectation:
            return self._static_offense(hand13, remaining)

        # 1-ply 期望：枚举下一张摸牌，按剩余概率加权。
        base13 = self._static_offense(hand13, remaining)
        total = sum(remaining.values())
        if total <= 0:
            return base13

        ev = 0.0
        for t, cnt in remaining.items():
            if cnt <= 0:
                continue
            hand14 = hand13 + [t]
            if eval_v2.is_win(hand14):
                v = WIN_VALUE
            else:
                # 摸到 t 后可以选择保留（近似用 14 张评估）或打回 t（回到 hand13）
                v = max(base13, self._static_offense(hand14, remaining))
            ev += (cnt / total) * v
        return ev

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

    def next(self):
        assert len(self.cur) == 14

        best_disc = None
        best_value = -float('inf')
        lam = self._lambda_def()

        for disc in self._unique_tiles(self.cur):
            hand13 = self._remove_one(self.cur, disc)
            off = self._offense_value(hand13, discarded_tile=disc)
            risk = opponent.tile_danger(disc, self.context, self.name)
            value = off - lam * risk
            if value > best_value:
                best_value = value
                best_disc = disc

        result = best_disc if best_disc is not None else self.cur[0]
        self.cur.remove(result)
        self.context.see_tile(result, self.name)
        if self.verbose:
            print('出牌:' + tile.tile_to_str(result))
        return result
