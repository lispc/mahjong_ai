# -*- coding: utf-8 -*-
"""GumbelSearchAgent：AlphaZero 式 1-ply gumbel 搜索改进策略 π' 的 Python arena 部署。

参考算法：jaxenv/search.py::improved_policy（1-ply 版）。训练环里 π' 能访问真实
状态（对手手牌、牌山）；部署版只有公开信息 + 自己手牌，隐藏信息用 belief 采样
近似（S 个世界）。

每步弃牌决策（k=8, n_draws=2, beta=32.0, S=4，确定性 top-k，无 Gumbel 噪声）：
1. prior：root obs 过一次网络得 policy logits 与 V（score/3 尺度，value head tanh）。
2. 按 logits 取前 k 个合法弃牌。
3. 每个候选 a 的 Q：
   a. 声明概率：从「不可见牌」（ctx.remaining_wall(full_hand)，= 牌山 + 对手手牌的
      多重集合）采 S 个世界，每个世界按各对手闭手数量（13 − 3×副露数）分配手牌。
      对每个对手 off∈{1,2,3}、每个世界做可行性硬掩码（hu ⇔ 样本手牌 + [a] 满足
      物理和牌判定 is_win_with_melds（含副露/七对）；peng ⇔ ≥2 张 a；gang ⇔ ≥3 张；报听锁手玩家
      不能碰/杠）。可行则构造该对手视角 obs 过 response head（4 logits
      [pass,peng,gang,hu]），P(claim X) = sigmoid(logit_X − logit_pass)，
      对 S 个世界取平均（不可行世界计 0），得 9 个声明位（胡 off1..3 → 杠
      off1..3 → 碰 off1..3）的 p_claim[9]。
   b. 7 个分支后状态（杠 off1..3 / 碰 off1..3 / 全pass）各自「跳到 root 下次
      摸牌后」截断：从不可见牌无放回采 n_draws 张（每个分支独立采），
      root 手牌' = 闭手 − a + 摸到的牌（+副露牌），若自摸（is_win_with_melds）
      该样本值 +1.0，否则过网络取 value head；均值即 jumpv[branch]。
      杠分支若牌山耗尽（世界分配后无余牌）→ dead → 价值 0。
   c. Q walk（同 search.py._walk_q）：val9 = [-1/3,-1/3,-1/3, jumpv[0..5]]，
      acc += rem·p_i·val_i，rem *= (1−p_i)，最后 acc + rem·jumpv[6]。
4. Q_completed：top-k 候选用 Q(a)，其余合法动作用 V 补齐；
   π' = masked softmax(logits + beta·(Q_completed − V))，取 argmax。

网络前向批量：response obs 行（≤ k×3×S，仅可行声明）拼一个 batch，value obs 行
（k×7×n_draws）拼一个 batch，各一次前向。device='cpu'。

与训练环 jaxenv 语义的对齐说明（重要）：
- response obs 用**未** see_tile(a) 的 context：引擎在 claims 处理完后才广播
  'put' 消息，arena 里响应方的 context 本就不含被弃的牌；jaxenv/obs.py 的 CLAIM
  obs 同样只把 pending tile 计入手牌通道、不计入弃牌堆。二者一致。
- 全 pass 分支的 value obs 用 see_tile(a, root) 后的 context 副本（对应 jaxenv
  _branch_states 的 discards[d]+=a）；碰/杠分支不用（jaxenv 中 a 进入对手副露，
  不进弃牌堆；ContextV3 本就不记副露，天然一致）。
- 和牌/自摸判定用物理和牌 `is_win_with_melds`（2026-07-19 起 arena 已支持
  副露和与七对子，与引擎实际判定一致）。

部署近似（vs 训练环真实状态）：
1. 隐藏信息用 S 个均匀 belief 世界代替真实手牌/牌山；世界在 k 个候选间复用
   （paired worlds，降低候选间比较方差）。
2. 对手副露牌（被碰/杠的牌不广播 'put'，对 context 不可见）通过引擎 'meld'
   消息自行跟踪并从不可见牌池剔除（碰 3 张/杠 4 张）。
3. 杠分支 dead 只检「牌山耗尽」（杠后补牌自摸概率忽略），
   与 search.py 的 dead 语义近似。

环境变量：MJ_GUMBEL_K(8)、MJ_GUMBEL_DRAWS(2)、MJ_GUMBEL_BETA(32.0)、
MJ_GUMBEL_S(4)、MJ_GUMBEL_DEALIN(1)。构造参数优先于环境变量。

防点炮通道（MJ_GUMBEL_DEALIN=1，默认开）：均匀 belief 世界下随机手牌成和概率
~0.1%，response head 的 hu 声明位恒 ~0，防点炮不生效。故 hu 声明位改用
dealin head（若模型带该头）：p_hu(a) = sigmoid(dealin_logit[a])，它直接以
对手弃牌/报听 flag 为条件，是聚合 P(任一家胡 a)。因三个胡位分支价值同为
-1/3，聚合与逐位 walk 等价：acc = p_hu*(-1/3) + (1-p_hu)*[杠/碰 walk + 全pass]。
此时 response head 的 hu 位弃用（peng/gang 位仍用）。
"""

import os
import random
import time

import numpy as np
import torch

from algo.agents.ppo_agent import PPOAgent
from algo.eval.v2 import is_win_with_melds
from algo.nn.features import extract_features, _TILE_TO_IDX, _IDX_TO_TILE

NUM_ACTIONS = 34
NEG = -1e9
HU_VALUE = -1.0 / 3.0      # 放炮（score=-1）的 /3 尺度
SELFDRAW_VALUE = 1.0       # 自摸（score=+3）的 /3 尺度

# 9 个声明位顺序（与 engine._process_claims / search.py 一致）：
# 胡 off1..3 -> 杠 off1..3 -> 碰 off1..3；response head 索引 pass=0 peng=1 gang=2 hu=3
_CLAIM_RESP_IDX = (3, 3, 3, 2, 2, 2, 1, 1, 1)


def _seat_of(name):
    """从 'Label@2' 形式的名字解析座位号（与 features._seat 同规则）。"""
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


class GumbelSearchAgent(PPOAgent):
    """1-ply gumbel 搜索改进策略 π' 的部署 agent。只覆盖 next()，其余继承 PPOAgent。"""

    def __init__(self, name, model_path='output/jax_gumbel_iter92.pt', device='cpu',
                 k=None, n_draws=None, beta=None, n_worlds=None, verbose=False):
        super().__init__(name, model_path=model_path, device=device,
                         temperature=0.0, verbose=verbose)
        self.k = int(k if k is not None else os.environ.get('MJ_GUMBEL_K', 8))
        self.n_draws = int(n_draws if n_draws is not None
                           else os.environ.get('MJ_GUMBEL_DRAWS', 2))
        self.beta = float(beta if beta is not None
                          else os.environ.get('MJ_GUMBEL_BETA', 32.0))
        self.n_worlds = int(n_worlds if n_worlds is not None
                            else os.environ.get('MJ_GUMBEL_S', 4))
        self.use_dealin = os.environ.get('MJ_GUMBEL_DEALIN', '1') == '1'
        self.use_claims = os.environ.get('MJ_GUMBEL_CLAIMS', '1') == '1'
        self._last_discarded = None   # 我们上一步打出的牌（被 claim 时撤回 used 用）
        self._meld_map = {}       # player -> [(meld_type, tile), ...]（引擎 'meld' 消息跟踪）
        # 决策统计（翻转率/耗时测量用）
        self.stat_decisions = 0
        self.stat_flips = 0
        self.stat_time = 0.0
        self.stat_resp_rows = 0
        self.stat_val_rows = 0

    def init_tiles(self, l):
        super().init_tiles(l)
        self._meld_map = {}
        self._last_discarded = None

    def handle_msg(self, msg):
        # claim 窗口只在我们弃牌后的瞬间有效；任何后续 'put' 都意味着窗口已过，
        # 清除 _last_discarded 防止误撤回（他人弃牌被碰时 tile 值可能巧合相同）。
        if msg.type == 'put':
            self._last_discarded = None
        # ContextV3 不记录副露；自行跟踪每家副露（belief 采样：被碰/杠的牌不会
        # 广播 'put'，整组副露牌对 context 不可见，必须从不可见牌池剔除）。
        if msg.type == 'meld' and isinstance(msg.data, dict):
            t = msg.data.get('tile')
            if t is not None:
                mtype = msg.data.get('type', 'peng')
                self._meld_map.setdefault(msg.sender, []).append((mtype, t))
                # 我们打出的牌被对手碰/杠：该牌已在 next() 里 see_tile 进了
                # used，被 claim 后不广播 'put' 也不撤回——used 多记 1 张，
                # 与副露物理牌重复计数，这里撤回。
                if msg.sender != self.name and t == self._last_discarded:
                    self._last_discarded = None
                    if self.context.used.get(t, 0) > 0:
                        self.context.used[t] -= 1
                        self.context.all_seen[t] = max(
                            0, self.context.all_seen.get(t, 0) - 1)
                    mine = self.context.discards.get(self.name)
                    if mine and mine[-1] == t:
                        mine.pop()
        return super().handle_msg(msg)

    @staticmethod
    def _meld_physical_tiles(entries):
        """(meld_type, tile) 列表 -> 物理副露牌（碰 3 张 / 杠 4 张）。"""
        out = []
        for mtype, t in entries:
            out.extend([t] * (4 if mtype == 'gang' else 3))
        return out

    # ------------------------------------------------------------------
    # 网络批量前向
    # ------------------------------------------------------------------

    def _forward(self, feats):
        """feats: (N,175) array -> (policy[N,34], value[N,1], response[N,4]|None, dealin[N,34]|None)。"""
        net = self._net_obj()
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            outs = net(x)
        policy = outs[0].detach().cpu().numpy().astype(np.float64)
        value = outs[1].detach().cpu().numpy().astype(np.float64)
        response = None
        for o in outs[2:]:
            if o.shape[-1] == 4:
                response = o.detach().cpu().numpy().astype(np.float64)
                break
        dealin = None
        if self.use_dealin and self._cfg.get('dealin_head', False):
            dealin = outs[2].detach().cpu().numpy().astype(np.float64)
        return policy, value, response, dealin

    # ------------------------------------------------------------------
    # belief 世界采样
    # ------------------------------------------------------------------

    def _seat_name_map(self):
        """座位号 -> 玩家名（真实名字来自弃牌/报听/副露记录）。"""
        m = {}
        for p in list(self.context.discards.keys()) + list(self.context.tenpai_players) \
                + list(self._meld_map.keys()) + [self.name]:
            m.setdefault(_seat_of(p), p)
        return m

    def _opponents(self):
        """off 1..3（下家起）的 (name, meld_entries, locked)。"""
        seat_map = self._seat_name_map()
        self_seat = _seat_of(self.name)
        out = []
        for off in (1, 2, 3):
            seat = (self_seat + off) % 4
            name = seat_map.get(seat, 'unknown@{}'.format(seat))
            entries = self._meld_map.get(name, [])
            out.append((name, entries, name in self.context.tenpai_players))
        return out

    @staticmethod
    def _deal(pool, sizes):
        """从不可见牌多重集合无放回给三家分配闭手。"""
        bag = list(pool)
        random.shuffle(bag)
        hands, i = [], 0
        for n in sizes:
            hands.append(bag[i:i + n])
            i += n
        return hands

    # ------------------------------------------------------------------
    # 搜索主流程
    # ------------------------------------------------------------------

    def next(self):
        assert len(self.cur) >= 1
        t0 = time.time()
        try:
            tile_val, info = self._search_select()
        except Exception:
            if os.environ.get('MJ_GUMBEL_DEBUG') == '1':
                import pickle
                dump = {
                    'cur': list(self.cur), 'melds': list(self.melds),
                    'meld_map': {k: v for k, v in self._meld_map.items()},
                    'used': dict(self.context.used),
                    'discards': {k: list(v) for k, v in self.context.discards.items()},
                    'tenpai_players': list(self.context.tenpai_players),
                    'name': self.name,
                }
                path = 'tmp/gumbel_crash_{}.pkl'.format(self.name.replace('@', '_'))
                with open(path, 'wb') as f:
                    pickle.dump(dump, f)
                print('[Gumbel] crash dump ->', path)
            raise
        dt = time.time() - t0
        self.stat_decisions += 1
        self.stat_flips += 1 if info['flip'] else 0
        self.stat_time += dt
        self.stat_resp_rows += info['resp_rows']
        self.stat_val_rows += info['val_rows']

        self.cur.remove(tile_val)
        self.context.see_tile(tile_val, self.name)
        self._last_discarded = tile_val
        self._belief = None
        if self.verbose:
            import tile as tile_mod
            print('[Gumbel] 出牌:{} flip={} prior={} q_best={}(Q={:+.3f}) '
                  'rows={}+{} t={:.0f}ms'.format(
                      tile_mod.tile_to_str(tile_val), info['flip'],
                      tile_mod.tile_to_str(info['prior_tile']),
                      tile_mod.tile_to_str(info['q_best_tile']), info['q_best'],
                      info['resp_rows'], info['val_rows'], dt * 1000))
        return tile_val

    def _search_select(self):
        ctx = self.context
        full = self.full_hand()
        root_meld_tiles = [t for _, t in self.melds]

        # 1) prior
        feats0 = self._extract(ctx, full, self.name)
        policy0, value0, _, dealin0 = self._forward(feats0[None])
        logits = policy0[0]
        v_root = float(value0[0, 0])

        legal_idx = sorted({int(_TILE_TO_IDX[t]) for t in self.cur})
        masked = np.full(NUM_ACTIONS, NEG)
        masked[legal_idx] = logits[legal_idx]
        order = np.argsort(-masked)
        top = [int(i) for i in order if masked[i] > NEG / 2][:self.k]
        prior_arg = int(np.argmax(masked))

        # 2) belief 世界（k 个候选复用）；剔除副露幻影牌（被碰/杠的牌不广播
        # 'put'，对 context 不可见，不剔除会被采进对手「手牌」导致计数 >4）：
        # - 对手副露：整组物理牌不可见（碰 3 / 杠 4）；
        # - 自家副露：full_hand 每组已记 1 张，额外幻影 = 碰 2 / 杠 3。
        m_dict = ctx.remaining_wall(full)
        opponents = self._opponents()
        for _, entries, _ in opponents:
            for t in self._meld_physical_tiles(entries):
                if m_dict.get(t, 0) > 0:
                    m_dict[t] -= 1
        for mtype, t in self.melds:
            for _ in range(3 if mtype == 'gang' else 2):
                if m_dict.get(t, 0) > 0:
                    m_dict[t] -= 1
        m_list = [t for t, c in m_dict.items() for _ in range(c)]
        # 对手闭手数量（post-discard）：碰进 1 张后随即打出 1 张、杠补牌后也打出
        # 1 张，故每组副露（无论碰/杠）闭手净减 3：13 − 3×副露数
        sizes = [13 - 3 * len(entries) for _, entries, _ in opponents]
        worlds = [self._deal(m_list, sizes) for _ in range(self.n_worlds)]
        # 杠分支 dead：世界分配后「牌山」耗尽（杠补自摸概率忽略，与 search.py
        # dead 语义近似）
        wall_empty = len(m_list) - sum(sizes) <= 0

        # 3a) response obs 行（仅可行声明；ctx 不 see_tile，与 CLAIM 训练语义一致）
        resp_rows, resp_meta = [], []
        if self.use_claims:
            for ia, aidx in enumerate(top):
                a = int(_IDX_TO_TILE[aidx])
                for off in (1, 2, 3):
                    name_c, entries_c, locked_c = opponents[off - 1]
                    meld_phys = self._meld_physical_tiles(entries_c)
                    for s in range(self.n_worlds):
                        closed = worlds[s][off - 1]
                        cnt_a = closed.count(a)
                        # 物理和牌判定（2026-07-19 起 arena 已支持副露和/七对）
                        hu_ok = is_win_with_melds(closed + [a], len(entries_c))
                        gang_ok = cnt_a >= 3 and not locked_c
                        peng_ok = cnt_a >= 2 and not locked_c
                        if hu_ok or gang_ok or peng_ok:
                            row = self._extract(ctx, closed + meld_phys + [a], name_c)
                            resp_rows.append(row)
                            resp_meta.append((ia, off, hu_ok, gang_ok, peng_ok))

        p_claim = np.zeros((len(top), 9))
        if resp_rows:
            _, _, resp, _ = self._forward(np.stack(resp_rows))
            if resp is not None:
                for r, (ia, off, hu_ok, gang_ok, peng_ok) in enumerate(resp_meta):
                    lg = resp[r]
                    for j, ok in ((0, hu_ok), (1, gang_ok), (2, peng_ok)):
                        if ok:
                            pos = j * 3 + (off - 1)
                            ridx = _CLAIM_RESP_IDX[pos]
                            p_claim[ia, pos] += 1.0 / (1.0 + np.exp(-(lg[ridx] - lg[0])))
        p_claim /= max(1, self.n_worlds)

        # 3b) value obs 行（k×7×n_draws 一个 batch）
        val_rows, val_meta = [], []
        for ia, aidx in enumerate(top):
            a = int(_IDX_TO_TILE[aidx])
            ctx2 = ctx.copy()
            ctx2.see_tile(a, self.name)
            cur_minus_a = list(self.cur)
            cur_minus_a.remove(a)
            for branch in range(7):
                # 全pass：a 入弃牌堆；碰/杠：a 进副露（ContextV3 不记副露）
                ctxb = ctx2 if branch == 6 else ctx
                nd = min(self.n_draws, len(m_list))
                draws = random.sample(m_list, nd) if nd > 0 else []
                for r in draws:
                    handp = cur_minus_a + [r] + root_meld_tiles
                    # 自摸判定用闭手 + n_melds 的物理和牌（obs 特征仍用含副露标记的 handp）
                    win = is_win_with_melds(cur_minus_a + [r], len(self.melds))
                    val_rows.append(self._extract(ctxb, handp, self.name))
                    val_meta.append((ia, branch, win))

        jumpv = np.zeros((len(top), 7))
        if val_rows:
            _, vals, _, _ = self._forward(np.stack(val_rows))
            counts = np.zeros((len(top), 7))
            for r, (ia, branch, win) in enumerate(val_meta):
                jumpv[ia, branch] += SELFDRAW_VALUE if win else float(vals[r, 0])
                counts[ia, branch] += 1.0
            nonzero = counts > 0
            jumpv[nonzero] /= counts[nonzero]
        if wall_empty:
            jumpv[:, 0:3] = 0.0    # 杠补牌时牌山耗尽 -> 流局 -> 价值 0

        # 3c) Q walk（search.py._walk_q；dealin head 可用时 hu 位用聚合 p_hu 门替代）
        p_hu_agg = np.zeros(len(top))
        if dealin0 is not None:
            dl = dealin0[0]
            for ia, aidx in enumerate(top):
                p_hu_agg[ia] = 1.0 / (1.0 + np.exp(-dl[aidx]))
        q_full = np.full(NUM_ACTIONS, v_root)
        q_top = np.full(len(top), v_root)
        for ia, aidx in enumerate(top):
            val9 = [HU_VALUE, HU_VALUE, HU_VALUE] + list(jumpv[ia, :6])
            if dealin0 is not None:
                # 聚合 hu 门：acc = p_hu*(-1/3) + (1-p_hu)*[杠/碰 walk + 全pass]
                acc = p_hu_agg[ia] * HU_VALUE
                rem = 1.0 - p_hu_agg[ia]
                for i in range(3, 9):
                    acc += rem * p_claim[ia, i] * val9[i]
                    rem *= (1.0 - p_claim[ia, i])
            else:
                acc, rem = 0.0, 1.0
                for i in range(9):
                    acc += rem * p_claim[ia, i] * val9[i]
                    rem *= (1.0 - p_claim[ia, i])
            q = acc + rem * jumpv[ia, 6]
            q_full[aidx] = q
            q_top[ia] = q

        # 4) π' = masked softmax(logits + beta*(Q_completed - V))，取 argmax
        adj = np.full(NUM_ACTIONS, NEG)
        for i in legal_idx:
            adj[i] = logits[i] + self.beta * (q_full[i] - v_root)
        action = int(np.argmax(adj))
        tile_val = int(_IDX_TO_TILE[action])

        info = {
            'flip': action != prior_arg,
            'prior_tile': int(_IDX_TO_TILE[prior_arg]),
            'q_best_tile': int(_IDX_TO_TILE[top[int(np.argmax(q_top))]]) if top else tile_val,
            'q_best': float(np.max(q_top)) if top else v_root,
            'resp_rows': len(resp_rows),
            'val_rows': len(val_rows),
            'q_top': {int(_IDX_TO_TILE[aidx]): float(q_top[ia])
                      for ia, aidx in enumerate(top)},
            'p_claim': {int(_IDX_TO_TILE[aidx]): [float(x) for x in p_claim[ia]]
                        for ia, aidx in enumerate(top)},
            'p_claim_max': float(p_claim.max()) if p_claim.size else 0.0,
            'p_hu_max': float(p_hu_agg.max()),
            'v_root': v_root,
        }
        return tile_val, info
