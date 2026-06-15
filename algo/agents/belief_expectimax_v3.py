# -*- coding: utf-8 -*-
"""方案 B 再升级：BeliefExpectimaxAgent V3。

核心改进：
1. 用完全自洽的 expectimax 替代 `algo.eval2`：
   - 叶子节点用 `eval_v2.evaluate + ukeire/wait`；
   - 中间节点枚举摸牌并按有效剩余张数加权；
   - 每次摸牌后做 max-over-discard，尊重“摸完必须打牌”的规则。
2. 引入 per-player tile-level 信念（`algo.eval.player_belief.PlayerBelief`），
   把“对手可能持有某张牌”折算成有效牌山剩余，从而更真实地估计 ukeire / 待牌。
3. 防守沿用 V2 的 per-player 危险度聚合 + 安全 tie-breaking。
"""

import functools
import agent
import tile
import algo
import context as ctx_module
import algo.context.v3 as context_v3
import algo.eval.v2 as eval_v2
import algo.eval.v3 as eval_v3
import algo.eval.opponent as opponent
from algo.eval.player_belief import PlayerBelief


_NN_LEAF_MOD = None
_NN_POLICY_MOD = None


def _get_nn_leaf():
    global _NN_LEAF_MOD
    if _NN_LEAF_MOD is None:
        from algo.nn import nn_leaf
        _NN_LEAF_MOD = nn_leaf
    return _NN_LEAF_MOD


def _get_nn_policy():
    global _NN_POLICY_MOD
    if _NN_POLICY_MOD is None:
        from algo.nn import nn_policy
        _NN_POLICY_MOD = nn_policy
    return _NN_POLICY_MOD


WIN_VALUE = 100.0


_EMPTY_CONTEXT = ctx_module.Context()


def _leaf_value_impl(hand, leaf_mode='eval0'):
    """叶子估值：eval0 或训练好的 NN value。"""
    if leaf_mode == 'nn':
        return _get_nn_leaf().nn_leaf_value(hand)
    return algo.eval0(hand, _EMPTY_CONTEXT)


@functools.lru_cache(maxsize=500000)
def _expectimax_cached(hand_tuple, rem_tuple, depth, leaf_mode='eval0'):
    hand = list(hand_tuple)
    remaining = {t: c for t, c in rem_tuple}

    if depth == 0:
        return _leaf_value_impl(hand, leaf_mode)

    total = sum(remaining.values())
    if total <= 0:
        return _leaf_value_impl(hand, leaf_mode)

    ev = 0.0
    unique_tiles = set(hand)
    for t, cnt in remaining.items():
        if cnt <= 0:
            continue
        prob = cnt / total
        hand14 = hand + [t]

        if eval_v3._is_win_14(eval_v3.hand_to_counts(hand14)):
            ev += prob * WIN_VALUE
            continue

        new_eff = dict(remaining)
        new_eff[t] = max(0.0, new_eff.get(t, 0) - 1.0)

        best = -float('inf')
        for x in unique_tiles | {t}:
            if x not in hand14:
                continue
            hand13p = list(hand14)
            hand13p.remove(x)
            rem_tuple_p = tuple(sorted(new_eff.items()))
            v = _expectimax_cached(tuple(sorted(hand13p)), rem_tuple_p,
                                   depth - 1, leaf_mode)
            if v > best:
                best = v
        ev += prob * best

    return ev


class BeliefExpectimaxV3Agent(agent.Agent):
    """
    Belief Expectimax Agent V3（自洽 expectimax + per-player 信念）。

    参数：
        expectimax_depth: 1 表示“一次摸牌+一次打牌”的期望；2 表示两轮。
        max_candidates: 进入 expectimax 精确评估的候选弃牌数。
        candidate_policy: 候选弃牌生成策略，'eval0' | 'nn' | 'baseline' | 'baseline_fast' | 'baseline_best' | 'baseline_rerank' | 'baseline_empty' | 'baseline_eval1'。
        weight_ukeire / weight_tenpai: 叶子节点奖励权重。
        defense_margin: 安全 tie-breaking 的进攻分允许差距。
    """

    def __init__(self, name, verbose=False,
                 expectimax_depth=1,
                 max_candidates=8,
                 defense_margin=0.03,
                 leaf_evaluator='eval0',
                 candidate_policy='baseline_eval1'):
        super().__init__(name, verbose)
        self.expectimax_depth = expectimax_depth
        self.max_candidates = max_candidates
        self.defense_margin = defense_margin
        self.leaf_evaluator = leaf_evaluator
        self.candidate_policy = candidate_policy
        self.context = context_v3.ContextV3()
        self._belief = None
        self._weights_tuple = tuple(sorted(eval_v3.DEFAULT_WEIGHTS.items()))

    def init_tiles(self, l):
        super().init_tiles(l)
        self.context = context_v3.ContextV3()
        self._belief = None

    def handle_msg(self, msg):
        if msg.type == 'put':
            self.context.see_tile(msg.data, msg.sender)
        elif msg.type == 'tenpai':
            self.context.declare_tenpai(msg.sender)
        self._belief = None
        return super().handle_msg(msg)

    def declare_tenpai(self, hand, context):
        if context is None:
            return False
        if sum(len(v) for v in context.discards.values()) < 12:
            return False
        if eval_v3.shanten_nb(eval_v3.hand_to_counts(hand)) != 0:
            return False
        remaining = context.remaining_wall(hand)
        waits = eval_v2.winning_tiles(hand, remaining)
        if not waits:
            return False
        total_wait = sum(remaining.get(t, 0) for t in waits)
        if total_wait >= 4:
            return True
        for t in waits:
            if context.all_seen.get(t, 0) > 0 and remaining.get(t, 0) > 0:
                return True
        return False

    def _belief_model(self):
        if self._belief is None:
            self._belief = PlayerBelief(self.context)
        return self._belief

    def _effective_remaining(self, hand13):
        """用 per-player 信念把全局剩余转换成“牌山有效剩余”。"""
        remaining = self.context.remaining_wall(hand13)
        belief = self._belief_model()
        effective = {}
        for t, cnt in remaining.items():
            if cnt <= 0:
                continue
            eff = cnt
            for player in self.context.discards:
                if player == self.name:
                    continue
                eff -= belief.expected_copies(player, t)
            effective[t] = max(0.0, eff)
        return effective

    def _unique_tiles(self, hand):
        seen = set()
        out = []
        for t in hand:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _expectimax_value(self, hand, effective_remaining, depth):
        """自洽 expectimax：depth 轮摸牌+打牌。"""
        if self.leaf_evaluator == 'nn' and depth == 1:
            return self._expectimax_value_batched_nn(hand, effective_remaining)
        hand_tuple = tuple(sorted(hand))
        rem_tuple = tuple(sorted(effective_remaining.items()))
        return _expectimax_cached(hand_tuple, rem_tuple, depth, self.leaf_evaluator)

    def _expectimax_value_batched_nn(self, hand13, effective_remaining):
        """对 depth=1 的 NN 叶子做批量评估，减少 MLX 前向调度次数。"""
        hand13 = list(hand13)
        unique_disc = self._unique_tiles(hand13)
        total = sum(effective_remaining.values())
        if total <= 0:
            return _get_nn_leaf().nn_leaf_value(hand13)

        ev = 0.0
        for t, cnt in effective_remaining.items():
            if cnt <= 0:
                continue
            prob = cnt / total
            hand14 = hand13 + [t]
            if eval_v3._is_win_14(eval_v3.hand_to_counts(hand14)):
                ev += prob * WIN_VALUE
                continue

            leaves = []
            for x in set(unique_disc) | {t}:
                if x not in hand14:
                    continue
                leaves.append(tuple(sorted([v for v in hand14 if v != x])))

            values = _get_nn_leaf().nn_leaf_values_batch(leaves)
            ev += prob * max(values)
        return ev

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
        candidates = self._unique_tiles(self.cur)

        # 预选 top_k 候选：eval0 / NN policy / baseline
        if self.candidate_policy == 'nn':
            top = _get_nn_policy().top_discards(self.cur, self.context, self.name,
                                                self.max_candidates)
        elif self.candidate_policy == 'baseline':
            # baseline 的 select 给出它认为最优的弃牌序列，作为 NN leaf 的候选池
            top = algo.select(list(self.cur), False, c=self.context)[:self.max_candidates]
        elif self.candidate_policy == 'baseline_fast':
            # 用 eval0 替代 eval2，生成候选更快，仍保留 baseline 风格的排序
            top = algo.select(list(self.cur), False, metric_f=algo.eval0,
                              c=self.context)[:self.max_candidates]
        elif self.candidate_policy == 'baseline_best':
            # 只取 baseline（eval2）认为最好的一张，再补充 eval0 top 候选
            # 兼顾 baseline 的稳健性与 eval0 的速度
            baseline_best = algo.select(list(self.cur), False, c=self.context)[0]
            eval0_top = algo.select(list(self.cur), False, metric_f=algo.eval0,
                                    c=self.context)[:self.max_candidates - 1]
            top = list(dict.fromkeys([baseline_best] + eval0_top))[:self.max_candidates]
        elif self.candidate_policy == 'baseline_rerank':
            # eval0 快速预选，再对少量候选用 eval2 重排序
            # 速度接近 baseline_fast，质量接近 baseline
            pre = algo.select(list(self.cur), False, metric_f=algo.eval0,
                              c=self.context)[:self.max_candidates + 2]
            scored = []
            for disc in pre:
                hand13 = list(self.cur)
                hand13.remove(disc)
                score = algo.eval2(hand13, self.context)
                scored.append((score, disc))
            scored.sort(reverse=True)
            top = [disc for _, disc in scored[:self.max_candidates]]
        elif self.candidate_policy == 'baseline_empty':
            # 用 baseline 原版的空 context select，速度最快，与 baseline agent 行为一致
            top = algo.select(list(self.cur), False)[:self.max_candidates]
        elif self.candidate_policy == 'baseline_eval1':
            # eval1 比 eval2 少一层递归，速度更快，仍保留 context-aware 的一阶 lookahead
            top = algo.select(list(self.cur), False, metric_f=algo.eval1,
                              c=self.context)[:self.max_candidates]
        else:
            scored = []
            for disc in candidates:
                hand13 = list(self.cur)
                hand13.remove(disc)
                score = eval_v2.evaluate(hand13)
                scored.append((score, disc))
            scored.sort(reverse=True)
            top = [disc for _, disc in scored[:self.max_candidates]]

        if self.leaf_evaluator == 'nn':
            _get_nn_leaf().set_leaf_context(self.context, self.name, list(self.cur))

        evaluated = []
        for disc in top:
            hand13 = list(self.cur)
            hand13.remove(disc)
            effective = self._effective_remaining(hand13)
            offense = self._expectimax_value(hand13, effective, self.expectimax_depth)
            danger = self._aggregate_danger(disc)
            evaluated.append((offense, danger, disc))

        if self.leaf_evaluator == 'nn':
            _get_nn_leaf().clear_leaf_context()

        best_offense = max(item[0] for item in evaluated)

        if self._danger_signal():
            n_tenpai = len(self.context.tenpai_players - {self.name})
            margin = self.defense_margin + 0.02 * n_tenpai
            safe = [item for item in evaluated if item[0] >= best_offense - margin]
            safe.sort(key=lambda x: x[1])
            result = safe[0][2]
        else:
            evaluated.sort(reverse=True, key=lambda x: x[0])
            result = evaluated[0][2]

        self.cur.remove(result)
        self.context.see_tile(result, self.name)
        self._belief = None
        if self.verbose:
            print('出牌:' + tile.tile_to_str(result))
        return result
