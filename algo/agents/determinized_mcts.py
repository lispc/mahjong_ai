# -*- coding: utf-8 -*-
"""方案 C：Determinized MCTS + 快速 Rollout Agent。

核心思想：
- 维护 tile-level 信念（ContextV3）。
- 每次决策采样若干“完整世界”（把未知牌随机分配给对手和牌山，与已见信息一致）。
- 在每个采样世界里，对每个候选弃牌跑一条快速 rollout（用 ShantenUkeire depth=0 作为默认 rollout policy）。
- 聚合所有 rollout 的收益，选择期望收益最高的弃牌。

注意：
- 这里为了实时性做了简化：每个世界里对每个候选只跑 1 条 rollout，且不建树，
  本质上是 "Determinized Flat Monte Carlo"，但保留了 MCTS + rollout 的核心思想。
- rollout 中不模拟报听锁手（保留接口但默认空），避免 rollout policy 复杂度爆炸。
"""

import random
import copy
from concurrent.futures import ProcessPoolExecutor

import agent
import tile
import algo
import context as ctx_module
from utils import dict_sub, count
import algo.context.v3 as context_v3
import algo.eval.v2 as eval_v2


# 快速 rollout policy：用原项目 eval0 选弃牌，不维护上下文。
_EMPTY_CONTEXT = ctx_module.Context()


def _fast_rollout_select(hand14):
    """为 rollout 选一个快速弃牌：最大化 algo.eval0(hand13, empty context)。"""
    best_disc = None
    best_score = -float('inf')
    seen = set()
    for disc in hand14:
        if disc in seen:
            continue
        seen.add(disc)
        hand13 = list(hand14)
        hand13.remove(disc)
        score = algo.eval0(hand13, _EMPTY_CONTEXT)
        if score > best_score:
            best_score = score
            best_disc = disc
    return best_disc


# ---------------------------------------------------------------------------
# NN value cutoff rollout worker
# ---------------------------------------------------------------------------


def _simulate_one_nn_value_cutoff(args):
    """在一个采样世界里用 NN value 截断评估候选弃牌。

    rollout 跑固定步数后，用 nn_leaf 评估当前玩家手牌价值。
    """
    from algo.nn import nn_leaf

    (candidate, current_hand, opp_hands, wall,
     public_ctx_dict, locked_indices, cutoff_depth) = args

    ctx = context_v3.ContextV3()
    ctx.used = public_ctx_dict['used'].copy()
    ctx.all_seen = public_ctx_dict['all_seen'].copy()
    ctx.discards = {p: list(seq) for p, seq in public_ctx_dict['discards'].items()}
    ctx.tenpai_players = set(public_ctx_dict['tenpai_players'])

    cur_hand = list(current_hand)
    cur_hand.remove(candidate)
    hands = [
        cur_hand,
        list(opp_hands[0]),
        list(opp_hands[1]),
        list(opp_hands[2]),
    ]
    player_names = ['cur@0', 'opp1@1', 'opp2@2', 'opp3@3']
    wall = list(wall)
    turn = 1
    current_idx = 0
    locked = set(locked_indices)

    # 设置 nn_leaf 全局上下文；子进程各自独立
    nn_leaf.set_leaf_context(ctx, player_names[current_idx], current_hand)

    step = 0
    max_steps = cutoff_depth if cutoff_depth > 0 else 10000

    while wall and step < max_steps:
        drawn = wall.pop(0)
        hands[turn].append(drawn)

        if eval_v2.is_win(hands[turn]):
            return 1.0 if turn == current_idx else -0.3

        if turn in locked:
            discarded = drawn
            hands[turn].remove(discarded)
        else:
            discarded = _fast_rollout_select(hands[turn])
            hands[turn].remove(discarded)

        for j in range(4):
            if j == turn:
                continue
            if eval_v2.is_win(hands[j] + [discarded]):
                if j == current_idx:
                    return 1.0
                if turn == current_idx:
                    return -1.0
                return -0.3

        ctx.see_tile(discarded, player_names[turn])

        if (turn not in locked and len(hands[turn]) == 13 and
                eval_v2.shanten(hands[turn]) == 0):
            rem = ctx.remaining_wall(hands[turn])
            waits = eval_v2.winning_tiles(hands[turn], rem)
            if sum(rem.get(t, 0) for t in waits) >= 3:
                locked.add(turn)
                ctx.declare_tenpai(player_names[turn])

        turn = (turn + 1) % 4
        step += 1

    # 截断：用 NN leaf 评估当前玩家手牌价值
    # 注意：当前玩家已经弃掉 candidate，手牌应为 13 张
    try:
        value = nn_leaf.nn_leaf_value(list(hands[current_idx]))
    except Exception:
        value = 0.0
    return float(value) / 100.0  # nn_leaf 返回 value = eval0 + 2*NN，范围约 [-100, 100]


# ---------------------------------------------------------------------------
# NN policy rollout worker
# ---------------------------------------------------------------------------

def _simulate_one_nn(args):
    """在一个采样世界里用 NN policy 作为 rollout policy 评估候选弃牌。"""
    from algo.nn import nn_policy

    (candidate, current_hand, opp_hands, wall,
     public_ctx_dict, locked_indices, rollout_depth) = args

    ctx = context_v3.ContextV3()
    ctx.used = public_ctx_dict['used'].copy()
    ctx.all_seen = public_ctx_dict['all_seen'].copy()
    ctx.discards = {p: list(seq) for p, seq in public_ctx_dict['discards'].items()}
    ctx.tenpai_players = set(public_ctx_dict['tenpai_players'])

    cur_hand = list(current_hand)
    cur_hand.remove(candidate)
    hands = [
        cur_hand,
        list(opp_hands[0]),
        list(opp_hands[1]),
        list(opp_hands[2]),
    ]
    # 带座位号的名字，让特征层能正确对齐自己/对手
    player_names = ['cur@0', 'opp1@1', 'opp2@2', 'opp3@3']
    wall = list(wall)
    turn = 1
    current_idx = 0
    locked = set(locked_indices)

    max_steps = rollout_depth if rollout_depth > 0 else 10000
    step = 0

    while wall and step < max_steps:
        drawn = wall.pop(0)
        hands[turn].append(drawn)

        if eval_v2.is_win(hands[turn]):
            return 1.0 if turn == current_idx else -0.3

        if turn in locked:
            discarded = drawn
            hands[turn].remove(discarded)
        else:
            discarded = nn_policy.top_discards(hands[turn], ctx,
                                               player_names[turn], 1)[0]
            hands[turn].remove(discarded)

        for j in range(4):
            if j == turn:
                continue
            if eval_v2.is_win(hands[j] + [discarded]):
                if j == current_idx:
                    return 1.0
                if turn == current_idx:
                    return -1.0
                return -0.3

        ctx.see_tile(discarded, player_names[turn])

        if (turn not in locked and len(hands[turn]) == 13 and
                eval_v2.shanten(hands[turn]) == 0):
            rem = ctx.remaining_wall(hands[turn])
            waits = eval_v2.winning_tiles(hands[turn], rem)
            if sum(rem.get(t, 0) for t in waits) >= 3:
                locked.add(turn)
                ctx.declare_tenpai(player_names[turn])

        turn = (turn + 1) % 4
        step += 1

    return 0.0


# ---------------------------------------------------------------------------
# Top-level picklable rollout worker
# ---------------------------------------------------------------------------

def _simulate_one(args):
    """在一个采样世界里评估某个候选弃牌。"""
    (candidate, current_hand, opp_hands, wall,
     public_ctx_dict, locked_indices, rollout_depth) = args

    # 重建公共信念上下文（供 rollout policy 使用）
    ctx = context_v3.ContextV3()
    ctx.used = public_ctx_dict['used'].copy()
    ctx.all_seen = public_ctx_dict['all_seen'].copy()
    ctx.discards = {p: list(seq) for p, seq in public_ctx_dict['discards'].items()}
    ctx.tenpai_players = set(public_ctx_dict['tenpai_players'])

    cur_hand = list(current_hand)
    cur_hand.remove(candidate)
    hands = [
        cur_hand,
        list(opp_hands[0]),
        list(opp_hands[1]),
        list(opp_hands[2]),
    ]
    player_names = ['cur', 'opp1', 'opp2', 'opp3']
    wall = list(wall)
    turn = 1  # 当前玩家已经打过 candidate，轮到下家
    current_idx = 0
    locked = set(locked_indices)

    max_steps = rollout_depth if rollout_depth > 0 else 10000
    step = 0

    while wall and step < max_steps:
        drawn = wall.pop(0)
        hands[turn].append(drawn)

        # 自摸
        if eval_v2.is_win(hands[turn]):
            return 1.0 if turn == current_idx else -0.3

        # 选弃牌
        if turn in locked:
            discarded = drawn
            hands[turn].remove(discarded)
        else:
            discarded = _fast_rollout_select(hands[turn])
            hands[turn].remove(discarded)

        # 点炮检查（只关心当前玩家的输赢）
        for j in range(4):
            if j == turn:
                continue
            if eval_v2.is_win(hands[j] + [discarded]):
                if j == current_idx:
                    return 1.0
                if turn == current_idx:
                    return -1.0
                return -0.3

        # 更新公共上下文
        ctx.see_tile(discarded, player_names[turn])

        # 简化报听：听牌且待牌>=3 时锁手（不处理现物待牌等细节）
        if (turn not in locked and len(hands[turn]) == 13 and
                eval_v2.shanten(hands[turn]) == 0):
            rem = ctx.remaining_wall(hands[turn])
            waits = eval_v2.winning_tiles(hands[turn], rem)
            if sum(rem.get(t, 0) for t in waits) >= 3:
                locked.add(turn)
                ctx.declare_tenpai(player_names[turn])

        turn = (turn + 1) % 4
        step += 1

    return 0.0


# ---------------------------------------------------------------------------
# BeliefExp hybrid rollout worker
# ---------------------------------------------------------------------------

def _simulate_one_belief(args):
    """
    使用 BeliefExpectimaxAgent 作为当前玩家的 rollout policy，
    对手仍用 _fast_rollout_select。
    """
    from algo.agents.belief_expectimax import BeliefExpectimaxAgent

    (candidate, current_hand, opp_hands, wall,
     public_ctx_dict, locked_indices, rollout_depth) = args

    ctx = context_v3.ContextV3()
    ctx.used = public_ctx_dict['used'].copy()
    ctx.all_seen = public_ctx_dict['all_seen'].copy()
    ctx.discards = {p: list(seq) for p, seq in public_ctx_dict['discards'].items()}
    ctx.tenpai_players = set(public_ctx_dict['tenpai_players'])

    # 当前玩家用 BeliefExpectimaxAgent 作为 rollout policy
    cur = BeliefExpectimaxAgent('cur', verbose=False)
    cur_hand = list(current_hand)
    cur_hand.remove(candidate)
    cur.init_tiles(cur_hand)
    cur.context = ctx.copy()

    opps = [list(opp_hands[0]), list(opp_hands[1]), list(opp_hands[2])]
    opp_names = ['opp1', 'opp2', 'opp3']
    wall = list(wall)
    turn = 1
    locked = set(locked_indices)

    max_steps = rollout_depth if rollout_depth > 0 else 10000
    step = 0

    while wall and step < max_steps:
        drawn = wall.pop(0)

        if turn == 0:
            if cur.add(drawn):
                return 1.0
            discarded = cur.next()  # 内部已经更新 cur.context
        else:
            opp_idx = turn - 1
            opps[opp_idx].append(drawn)
            if eval_v2.is_win(opps[opp_idx]):
                return -0.3
            discarded = _fast_rollout_select(opps[opp_idx])
            opps[opp_idx].remove(discarded)
            # 让当前玩家看到对手弃牌，更新信念
            cur.context.see_tile(discarded, opp_names[opp_idx])

        # 点炮检查
        for j in range(4):
            if j == turn:
                continue
            if j == 0:
                if eval_v2.is_win(list(cur.cur) + [discarded]):
                    return 1.0
            else:
                if eval_v2.is_win(opps[j - 1] + [discarded]):
                    if turn == 0:
                        return -1.0
                    return -0.3

        turn = (turn + 1) % 4
        step += 1

    # 截断启发式：根据当前玩家手牌向听数估计价值
    sh = eval_v2.shanten(cur.cur)
    rem = cur.context.remaining_wall(cur.cur)
    waits = eval_v2.winning_tiles(cur.cur, rem)
    wait_count = sum(rem.get(t, 0) for t in waits)
    value = -1.0 + 2.0 * max(0.0, (5.0 - sh) / 5.0) + 0.05 * wait_count
    return min(1.0, value)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DeterminizedMCTSAgent(agent.Agent):
    """
    Determinized MCTS Agent（方案 C 的实用化实现）。

    参数：
        n_worlds: 每次决策采样的世界数。
        n_rollouts_per_world: 每个世界每个候选的 rollout 次数（默认 1）。
        max_workers: 并行 rollout 的进程数；<=1 则串行。
        rollout_depth: 0 表示走完该局；>0 表示最多模拟这么多回合后截断。
        nn_value_cutoff: 是否用 NN leaf value 截断 rollout（默认 False）。
        cutoff_depth: NN value 截断的 rollout 深度。
    """

    def __init__(self, name, verbose=False,
                 n_worlds=5,
                 n_rollouts_per_world=1,
                 max_workers=1,
                 rollout_depth=0,
                 top_k=6,
                 belief_exp_rollout=False,
                 nn_rollout=False,
                 nn_value_cutoff=False,
                 cutoff_depth=20):
        super().__init__(name, verbose)
        self.n_worlds = n_worlds
        self.n_rollouts_per_world = n_rollouts_per_world
        self.max_workers = max_workers
        self.rollout_depth = rollout_depth
        self.top_k = top_k
        self.belief_exp_rollout = belief_exp_rollout
        self.nn_rollout = nn_rollout
        self.nn_value_cutoff = nn_value_cutoff
        self.cutoff_depth = cutoff_depth
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
        if context is None:
            return False
        if sum(len(v) for v in context.discards.values()) < 12:
            return False
        if eval_v2.shanten(hand) != 0:
            return False
        rem = context.remaining_wall(hand)
        waits = eval_v2.winning_tiles(hand, rem)
        if not waits:
            return False
        total_wait = sum(rem.get(t, 0) for t in waits)
        if total_wait >= 4:
            return True
        for t in waits:
            if context.all_seen.get(t, 0) > 0 and rem.get(t, 0) > 0:
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

    def _sample_world(self):
        """根据当前信念采样一个完整牌局（对手手牌 + 牌山）。"""
        all_tiles = tile.all_tiles_as_dict()
        unknown = dict_sub(dict_sub(all_tiles, self.context.used), count(self.cur))
        unknown_list = []
        for t, c in unknown.items():
            unknown_list.extend([t] * c)
        random.shuffle(unknown_list)

        opp1 = unknown_list[:13]
        opp2 = unknown_list[13:26]
        opp3 = unknown_list[26:39]
        wall = unknown_list[39:]
        return [opp1, opp2, opp3], wall

    def _public_context_dict(self):
        return {
            'used': self.context.used,
            'all_seen': self.context.all_seen,
            'discards': self.context.discards,
            'tenpai_players': list(self.context.tenpai_players),
        }

    def next(self):
        assert len(self.cur) == 14
        all_candidates = self._unique_tiles(self.cur)

        # 用 eval0 快速预选 top_k 候选，减少 rollout 数量
        type_ctx = ctx_module.Context()
        type_ctx.used = self.context.used.copy()
        scored = []
        for disc in all_candidates:
            hand13 = list(self.cur)
            hand13.remove(disc)
            score = algo.eval0(hand13, type_ctx)
            scored.append((score, disc))
        scored.sort(reverse=True)
        candidates = [disc for _, disc in scored[:self.top_k]]

        public_ctx = self._public_context_dict()

        depth_param = self.cutoff_depth if self.nn_value_cutoff else self.rollout_depth
        tasks = []
        for _ in range(self.n_worlds):
            opp_hands, wall = self._sample_world()
            for candidate in candidates:
                for _ in range(self.n_rollouts_per_world):
                    tasks.append((candidate, list(self.cur), opp_hands, wall,
                                  public_ctx, [], depth_param))

        if self.nn_value_cutoff:
            sim_fn = _simulate_one_nn_value_cutoff
        elif self.nn_rollout:
            sim_fn = _simulate_one_nn
        elif self.belief_exp_rollout:
            sim_fn = _simulate_one_belief
        else:
            sim_fn = _simulate_one
        if self.max_workers <= 1:
            rewards = list(map(sim_fn, tasks))
        else:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                rewards = list(executor.map(sim_fn, tasks))

        # 按候选聚合平均收益
        sums = {d: 0.0 for d in candidates}
        counts_map = {d: 0 for d in candidates}
        for task, reward in zip(tasks, rewards):
            d = task[0]
            sums[d] += reward
            counts_map[d] += 1

        best_disc = None
        best_value = -float('inf')
        for d in candidates:
            avg = sums[d] / max(counts_map[d], 1)
            if avg > best_value:
                best_value = avg
                best_disc = d

        if best_disc is None:
            best_disc = candidates[0]

        self.cur.remove(best_disc)
        self.context.see_tile(best_disc, self.name)
        if self.verbose:
            print('出牌:' + tile.tile_to_str(best_disc))
        return best_disc
