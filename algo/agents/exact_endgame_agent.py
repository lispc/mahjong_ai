# -*- coding: utf-8 -*-
"""终盘精确求解 agent：一旦有对手报听且牌山剩余较少，
用 exact endgame solver 选择最小点炮概率的弃牌；其他状态回退到基础 agent。"""

import agent
from algo.eval import endgame_solver
from algo.eval.v2 import winning_tiles


class ExactEndgameAgent(agent.Agent):
    """
    Wrapper agent：在对手报听后的终盘用精确求解器，其余时候用 base_agent。

    参数：
        name: agent 名字
        base_agent: 基础决策 agent（必须可 pickle / 有 next/respond_* 接口）
        wall_threshold: 牌山剩余 <= 此阈值且有人报听时触发精确求解
        deal_in_reward: 点炮收益（默认 -1）
    """

    def __init__(self, name, base_agent, wall_threshold=20,
                 deal_in_reward=-1.0, verbose=False):
        super().__init__(name, verbose=verbose)
        self.base_agent = base_agent
        self.wall_threshold = wall_threshold
        self.deal_in_reward = deal_in_reward

    def init_tiles(self, l):
        super().init_tiles(l)
        self.base_agent.init_tiles(l)

    def add_meld(self, meld_type, tile_val):
        super().add_meld(meld_type, tile_val)
        self.base_agent.add_meld(meld_type, tile_val)

    def handle_msg(self, msg):
        self.base_agent.handle_msg(msg)
        return super().handle_msg(msg)

    def declare_tenpai(self, hand, context):
        return self.base_agent.declare_tenpai(hand, context)

    def respond_hu(self, tile_val, context=None):
        return self.base_agent.respond_hu(tile_val, context)

    def respond_peng(self, tile_val, context=None):
        return self.base_agent.respond_peng(tile_val, context)

    def respond_gang(self, tile_val, context=None):
        return self.base_agent.respond_gang(tile_val, context)

    def _tenpai_opponents(self):
        ctx = getattr(self.base_agent, 'context', None)
        if ctx is None:
            return []
        tenpai = getattr(ctx, 'tenpai_players', set())
        return [p for p in tenpai if p != self.name]

    def _wall_remaining(self):
        ctx = getattr(self.base_agent, 'context', None)
        if ctx is None:
            return []
        # 用 context.remaining_wall 获取剩余牌山
        return ctx.remaining_wall(self.cur)

    def _infer_tenpai_waits(self, opponent_name):
        """
        推断报听者的待牌集合。
        当前简化版：从 context 中拿对手的真实 13 张闭手（仅适用于自对弈/数据生成，
        真实对局中不可用）。后续应替换为信念推断或 34 维待牌分布。
        """
        ctx = getattr(self.base_agent, 'context', None)
        if ctx is None:
            return set()
        # 在自对弈数据生成时，context 中可能保存了对手手牌快照
        opp_hand = getattr(ctx, '_player_hands', {}).get(opponent_name)
        if opp_hand is None:
            return set()
        rem = self._wall_remaining()
        return set(winning_tiles(opp_hand, rem))

    def next(self):
        opp_names = self._tenpai_opponents()
        wall = self._wall_remaining()
        wall_len = sum(wall.values())

        if (opp_names and wall_len <= self.wall_threshold and
                len(self.cur) == 14):
            # 简化：只考虑第一个报听对手
            waits = self._infer_tenpai_waits(opp_names[0])
            if waits:
                wall_list = []
                for t, c in wall.items():
                    wall_list.extend([t] * c)
                # tenpai_offset: 当前玩家下一家摸牌即 offset=0 对报听者？
                # 这里简化用 0，实际需要根据座位计算。
                best, _ = endgame_solver.best_defensive_discard(
                    self.cur, waits, wall_list, tenpai_offset=0,
                    deal_in_reward=self.deal_in_reward)
                self.cur.remove(best)
                ctx = getattr(self.base_agent, 'context', None)
                if ctx is not None:
                    ctx.see_tile(best, self.name)
                return best

        return self.base_agent.next()
