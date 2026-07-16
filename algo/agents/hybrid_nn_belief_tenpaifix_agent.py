# -*- coding: utf-8 -*-
"""HybridNNBeliefAgent 的 tenpai 触发修复变体（评测用，不改原类）。

原版 `_is_critical` 用 `getattr(ctx, 'tenpai', set())`，但 ContextV3 的属性名是
`tenpai_players`——「任一对手报听 → 触发搜索」分支因此从未生效（死代码），
只有总弃牌数阈值在起作用。本变体只修正这一处，其余行为完全一致。

用法（benchmark_pool token）：
    hybridfix:<label>:<nn_model_path>
"""

from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent


class HybridNNBeliefTenpaiFixAgent(HybridNNBeliefAgent):
    def _is_critical(self):
        if len(self.cur) != 14:
            return False
        ctx = self.nn_agent.context
        if ctx is None:
            return False
        tenpai_players = getattr(ctx, 'tenpai_players', set()) or set()
        if self.name in tenpai_players:
            tenpai_players = set(tenpai_players)
            tenpai_players.discard(self.name)
        if tenpai_players:
            return True
        total_discarded = sum(len(v) for v in getattr(ctx, 'discards', {}).values())
        if total_discarded >= self.tenpai_threshold:
            return True
        return False
