# -*- coding: utf-8 -*-
"""自对弈 bootstrap：on-policy outcome 数据 -> conv critic -> AWBC 策略微调。

动机（docs/reports/ablation_report.md §3.1/§7）：
- 旧 AWBC 未确认超越 BC，瓶颈是 value net 质量（旧 MLP MC value）；
- 本脚本用当前 best NN 自对弈（温度采样制造动作对比），训练同架构 conv
  critic 拟合终局 score-proxy，再以 advantage-weighted BC 微调 policy。

三个子命令：
  collect      并行自对弈，分 shard 落盘（断点续跑），最后 merge。
  train_value  冻结主干，只训 value_fc/value_head 拟合 score-proxy；输出校准诊断。
  finetune     w=exp(A_std/beta)（或 filtered: 1[A>0]）加权 masked-CE 微调。

部署形态：候选 .pt 替换 HybridNNBeliefAgent 的 nn_model_path，
评测走 scripts/rl/benchmark_duplicate.py（见 docs/eval-protocol.md）。

用法：
    PYTHONPATH=. python3 scripts/rl/selfplay_bootstrap.py collect \
        --games 24000 --workers 96 --out-prefix output/bootstrap_v1 \
        --init output/nn_full_action_best.pt
    PYTHONPATH=. python3 scripts/rl/selfplay_bootstrap.py train_value \
        --data output/bootstrap_v1_merged.npz --init output/nn_full_action_best.pt \
        --out output/nn_bootstrap_v1_critic.pt --device cuda:0
    PYTHONPATH=. python3 scripts/rl/selfplay_bootstrap.py finetune \
        --data output/bootstrap_v1_merged.npz --init output/nn_full_action_best.pt \
        --critic output/nn_bootstrap_v1_critic.pt --beta 1.0 \
        --out output/nn_bootstrap_v1_awbc_b1.pt --device cuda:0
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from algo.rl.selfplay import PPOActorAgent, build_net
from algo.rl.reward import terminal_reason
from algo.nn.features import extract_features, _IDX_TO_TILE, _TILE_TO_IDX

NUM_ACTIONS = 34

# reason_code -> int8
_REASON_CODE = {
    'other': 0, 'tsumo_win': 1, 'ron_win': 2, 'deal_in': 3,
    'draw': 4, 'lose_tsumo': 5, 'lose_ron_others': 6,
}
# score-proxy（与 duplicate 评测辅指标一致）：自摸 +3 / 点和 +1 / 放炮 -1 / 其余 0
_CODE_SCORE = {0: 0.0, 1: 3.0, 2: 1.0, 3: -1.0, 4: 0.0, 5: 0.0, 6: 0.0}
# value head 是 tanh（值域 [-1,1]），critic 拟合 score/_SCORE_SCALE
_SCORE_SCALE = 3.0

_TEMPS = (0.7, 0.85, 1.0, 1.2)


def _parse_temps(s):
    return tuple(float(x) for x in s.split(','))


# ---------------------------------------------------------------- collect

class BootstrapActorAgent(PPOActorAgent):
    """PPOActorAgent + 与部署一致的 NN response head / tenpai head 决策。

    基类只记录弃牌决策；响应动作用 response head（不记录），
    报听用 tenpai head——与 Hybrid 部署形态（PPOAgent 行为）一致，
    保证轨迹分布接近部署分布。
    """

    def __init__(self, name, net, cfg, **kw):
        super().__init__(name, net, **kw)
        self._cfg = cfg
        self.traj_resp = []

    def init_tiles(self, l):
        super().init_tiles(l)
        self.traj_resp = []

    def _record_response(self, tile_val, legal_mask, a):
        """记录一次响应决策（feat 含 offered tile，与部署 PPOAgent 一致）。"""
        if not self.record:
            return
        hand = self.full_hand() + [tile_val]
        feats = extract_features(self.context, hand, self.name)
        self.traj_resp.append({
            'feat': np.asarray(feats, dtype=np.float32),
            'action': a,
            'mask': legal_mask.astype(np.float32),
            'tile': int(_TILE_TO_IDX[tile_val]),
        })

    def next(self):
        """与部署（PPOAgent.next）一致：特征用 full_hand，合法集为闭手 cur。

        有副露时 len(cur) != 14，因此不能用基类的 cur==14 假设。
        """
        assert len(self.cur) >= 1
        feats = extract_features(self.context, self.full_hand(), self.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = self.net(x)
        logits = out[0].squeeze(0).detach().cpu().numpy().astype(np.float64)
        value = float(out[1].detach().cpu().reshape(-1)[0])

        legal = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for t in self.cur:
            legal[int(_TILE_TO_IDX[t])] = 1.0
        masked = logits + (legal - 1.0) * 1e9
        m = masked / max(self.temperature, 1e-6)
        m = m - m.max()
        probs = np.exp(m)
        probs = probs / probs.sum()
        if self.deterministic or self.temperature <= 1e-6:
            a = int(np.argmax(probs))
        else:
            a = int(np.random.choice(NUM_ACTIONS, p=probs))
        logp = float(np.log(probs[a] + 1e-12))

        if self.record:
            self.traj.append({
                'feat': np.asarray(feats, dtype=np.float32),
                'action': a,
                'logp': logp,
                'value': value,
                'mask': legal,
            })

        tile_val = int(_IDX_TO_TILE[a])
        self.cur.remove(tile_val)
        self.context.see_tile(tile_val, self.name)
        self._belief = None
        return tile_val

    def _response_action(self, tile_val, legal_mask):
        hand = self.full_hand() + [tile_val]
        feats = extract_features(self.context, hand, self.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = self.net(x)
        response_logits = None
        for o in out:
            if o.shape[-1] == 4:
                response_logits = o
                break
        if response_logits is None:
            return None
        logits = response_logits.squeeze(0).cpu().numpy().astype(np.float64)
        masked = logits + (legal_mask - 1.0) * 1e9
        rt = max(self.temperature, 1.0)   # 响应动作探索温度下限 1.0，保证动作对比
        if self.deterministic or rt <= 1e-6:
            return int(np.argmax(masked))
        m = masked / rt
        m = m - m.max()
        probs = np.exp(m)
        probs = probs / probs.sum()
        return int(np.random.choice(4, p=probs))

    def respond_hu(self, tile_val, context=None):
        legal = np.zeros(4, dtype=np.float32)
        legal[0] = 1.0
        if super(PPOActorAgent, self).respond_hu(tile_val, context):
            legal[3] = 1.0
        if legal.sum() <= 1:
            return False
        a = self._response_action(tile_val, legal)
        if a is None:
            return bool(legal[3])
        self._record_response(tile_val, legal, a)
        return a == 3

    def respond_peng(self, tile_val, context=None):
        legal = np.zeros(4, dtype=np.float32)
        legal[0] = 1.0
        if self._can_peng(tile_val):
            legal[1] = 1.0
        if legal.sum() <= 1:
            return False
        a = self._response_action(tile_val, legal)
        if a is None:
            return False
        self._record_response(tile_val, legal, a)
        return a == 1

    def respond_gang(self, tile_val, context=None):
        legal = np.zeros(4, dtype=np.float32)
        legal[0] = 1.0
        if self._can_gang(tile_val):
            legal[2] = 1.0
        if legal.sum() <= 1:
            return False
        a = self._response_action(tile_val, legal)
        if a is None:
            return False
        self._record_response(tile_val, legal, a)
        return a == 2

    def declare_tenpai(self, hand, context):
        if self._cfg.get('tenpai_head', False) and context is not None:
            try:
                feats = extract_features(context, hand, self.name)
                x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0)
                with torch.no_grad():
                    logit = self.net.tenpai_logit(x)
                return bool(logit.item() > 0.0)
            except Exception:
                pass
        return super().declare_tenpai(hand, context)


def _play_game(net, cfg, seed, temperature):
    """4 个 BootstrapActorAgent 自对弈，返回该局 4 条轨迹。"""
    from driver.engine import play_game
    random.seed(seed)
    np.random.seed(seed % (2 ** 31 - 1))
    agents = [BootstrapActorAgent(f'A@{s}', net, cfg, device='cpu',
                                  deterministic=False, temperature=temperature)
              for s in range(4)]
    result = play_game(agents)
    trajs = []
    for ag in agents:
        if not ag.traj and not ag.traj_resp:
            continue
        trajs.append({
            'steps': ag.traj,
            'resp': ag.traj_resp,
            'reason': terminal_reason(result, ag.name),
        })
    return trajs


def _flat_steps(trajs, game_id, temp_id):
    """把一局若干条轨迹展平成 dict of arrays（弃牌 + 响应；无数据部分为 None）。"""
    feats, actions, masks, old_logp, old_value, codes, gids, tids = \
        [], [], [], [], [], [], [], []
    r_feats, r_actions, r_masks, r_tiles, r_codes, r_gids, r_tids = \
        [], [], [], [], [], [], []
    for tr in trajs:
        code = _REASON_CODE.get(tr['reason'], 0)
        for s in tr['steps']:
            feats.append(s['feat'])
            actions.append(s['action'])
            masks.append(s['mask'])
            old_logp.append(s['logp'])
            old_value.append(s['value'])
            codes.append(code)
            gids.append(game_id)
            tids.append(temp_id)
        for s in tr.get('resp', []):
            r_feats.append(s['feat'])
            r_actions.append(s['action'])
            r_masks.append(s['mask'])
            r_tiles.append(s['tile'])
            r_codes.append(code)
            r_gids.append(game_id)
            r_tids.append(temp_id)
    out = {}
    if feats:
        out.update({
            'feats': np.asarray(feats, dtype=np.float32),
            'actions': np.asarray(actions, dtype=np.int64),
            'masks': np.asarray(masks, dtype=bool),
            'old_logp': np.asarray(old_logp, dtype=np.float32),
            'old_value': np.asarray(old_value, dtype=np.float32),
            'reason_code': np.asarray(codes, dtype=np.int8),
            'game_id': np.asarray(gids, dtype=np.int32),
            'temp_id': np.asarray(tids, dtype=np.int8),
        })
    if r_feats:
        out.update({
            'resp_feats': np.asarray(r_feats, dtype=np.float32),
            'resp_actions': np.asarray(r_actions, dtype=np.int64),
            'resp_masks': np.asarray(r_masks, dtype=bool),
            'resp_tiles': np.asarray(r_tiles, dtype=np.int8),
            'resp_reason_code': np.asarray(r_codes, dtype=np.int8),
            'resp_game_id': np.asarray(r_gids, dtype=np.int32),
            'resp_temp_id': np.asarray(r_tids, dtype=np.int8),
        })
    return out or None


_COUNTER = None


def _worker_init(counter):
    global _COUNTER
    _COUNTER = counter
    torch.set_num_threads(1)


def _shard_worker(args):
    (state_dict, config, n_games, seed_base, shard_path, progress_every,
     temps) = args
    torch.set_num_threads(1)
    net = build_net(state_dict, config, device='cpu')
    blocks = []
    for i in range(n_games):
        temp_id = i % len(temps)
        trajs = _play_game(net, config, seed_base + i, temps[temp_id])
        blk = _flat_steps(trajs, seed_base + i, temp_id)
        if blk is not None:
            blocks.append(blk)
        if _COUNTER is not None:
            with _COUNTER.get_lock():
                _COUNTER.value += 1
    if not blocks:
        raise RuntimeError(f'no data collected for {shard_path}')
    keys = set()
    for b in blocks:
        keys.update(b.keys())
    out = {k: np.concatenate([b[k] for b in blocks if k in b]) for k in keys}
    np.savez(shard_path, **out)
    return shard_path, len(out.get('actions', out.get('resp_actions')))


def cmd_collect(args):
    import multiprocessing as mp
    config = json.load(open(args.init.replace('.pt', '_config.json')))
    sd = {k: v.cpu() for k, v in _load_init(args.init).items()}

    per = int(np.ceil(args.games / args.workers))
    temps = _parse_temps(args.temps)
    tasks = []
    for w in range(args.workers):
        shard_path = f'{args.out_prefix}.shard{w:03d}.npz'
        if os.path.exists(shard_path):
            continue  # 断点续跑：已完成 shard 跳过
        tasks.append((sd, config, per, args.seed_base + w * 1_000_000,
                      shard_path, args.progress_every, temps))
    total_planned = len(tasks) * per
    print(f'[collect] {args.games} games planned, {args.workers} workers, '
          f'{len(tasks)} shards todo ({total_planned} games), '
          f'{args.workers - len(tasks)} shards resumed')

    counter = mp.Value('l', 0)
    t0 = time.time()
    ctx = mp.get_context('fork')
    with ctx.Pool(args.workers, initializer=_worker_init, initargs=(counter,)) as pool:
        async_res = pool.map_async(_shard_worker, tasks)
        done_games = 0
        while not async_res.ready():
            async_res.wait(timeout=15)
            c = counter.value
            rate = c / max(time.time() - t0, 1e-9)
            eta = (total_planned - c) / max(rate, 1e-9)
            print(f'[collect] {c}/{total_planned} games '
                  f'({rate:.1f} g/s, ETA {eta/60:.1f} min)', flush=True)
        results = async_res.get()

    total_steps = sum(r[1] for r in results)
    print(f'[collect] done in {(time.time()-t0)/60:.1f} min, '
          f'new steps={total_steps}')

    # merge 全部 shard（含此前已完成的）
    shard_paths = [f'{args.out_prefix}.shard{w:03d}.npz'
                   for w in range(args.workers)]
    shard_paths = [p for p in shard_paths if os.path.exists(p)]
    print(f'[merge] {len(shard_paths)} shards')
    blocks = [np.load(p) for p in shard_paths]
    keys = set()
    for b in blocks:
        keys.update(b.files)
    merged = {k: np.concatenate([b[k] for b in blocks if k in b.files])
              for k in keys}
    out_path = f'{args.out_prefix}_merged.npz'
    np.savez(out_path, **merged)
    n_games = len(np.unique(merged['game_id']))
    print(f'[merge] -> {out_path}: steps={len(merged["actions"])}, '
          f'games={n_games}')
    codes, counts = np.unique(merged['reason_code'], return_counts=True)
    for c, n in zip(codes, counts):
        print(f'  reason_code {c}: {n} ({n/len(merged["reason_code"]):.1%})')


# ------------------------------------------------------------ train_value

def _load_init(path):
    sd = torch.load(path, map_location='cpu')
    if isinstance(sd, dict):
        if 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        elif 'model_state' in sd:
            sd = sd['model_state']
    return sd


def _build_from(path):
    from algo.nn.model import build_model
    cfg = json.load(open(path.replace('.pt', '_config.json')))
    net = build_model(cfg)
    net.load_state_dict(_load_init(path), strict=False)
    return net, cfg


def _score_from_code(codes):
    lut = np.array([_CODE_SCORE[i] for i in range(7)], dtype=np.float32)
    return lut[codes] / _SCORE_SCALE


def _split(data, val_frac=0.02, seed=0):
    """按 game_id 切 train/val（同局不泄漏）。"""
    gids = np.unique(data['game_id'])
    rng = np.random.RandomState(seed)
    rng.shuffle(gids)
    n_val = max(1, int(len(gids) * val_frac))
    val_games = set(gids[:n_val].tolist())
    is_val = np.array([g in val_games for g in data['game_id']])
    return ~is_val, is_val


def _calibration(v, r, n_buckets=8):
    order = np.argsort(v)
    v, r = v[order], r[order]
    rows = []
    for i in range(n_buckets):
        lo = i * len(v) // n_buckets
        hi = (i + 1) * len(v) // n_buckets
        rows.append((float(v[lo:hi].mean()), float(r[lo:hi].mean()), hi - lo))
    return rows


def cmd_train_value(args):
    from algo.nn.model import build_model
    data = np.load(args.data)
    net, cfg = _build_from(args.init)
    device = args.device
    net.to(device)

    scores = _score_from_code(data['reason_code'])
    tr_mask, va_mask = _split(data)
    X_tr = torch.from_numpy(data['feats'][tr_mask])
    y_tr = torch.from_numpy(scores[tr_mask])
    X_va = torch.from_numpy(data['feats'][va_mask]).to(device)
    y_va = scores[va_mask]
    print(f'[value] train={len(y_tr)} val={len(y_va)} '
          f'score mean={y_tr.mean():.4f} std={y_tr.std():.4f}')

    # 冻结主干，只训 value_fc/value_head（--full-trunk 时全解冻，小 lr）
    for name, p in net.named_parameters():
        p.requires_grad = args.full_trunk or name.startswith('value_')
    trainable = [p for p in net.parameters() if p.requires_grad]
    print(f'[value] trainable params: {sum(p.numel() for p in trainable)}')
    opt = torch.optim.Adam(trainable, lr=args.lr)

    N = len(y_tr)
    idx = np.arange(N)
    best_val = float('inf')
    best_state = None
    for ep in range(args.epochs):
        net.train()
        np.random.shuffle(idx)
        for s in range(0, N, args.batch):
            mb = torch.from_numpy(idx[s:s + args.batch])
            out = net(X_tr[mb].to(device))
            loss = F.mse_loss(out[1].squeeze(-1), y_tr[mb].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            preds = []
            for s in range(0, len(y_va), 8192):
                out = net(X_va[s:s + 8192])
                preds.append(out[1].squeeze(-1).cpu().numpy())
            v_va = np.concatenate(preds)
        mse = float(np.mean((v_va - y_va) ** 2))
        corr = float(np.corrcoef(v_va, y_va)[0, 1])
        print(f'[value] epoch {ep}: val MSE={mse:.4f} corr={corr:.4f} '
              f'(baseline MSE={float(np.var(y_va)):.4f})', flush=True)
        if mse < best_val:
            best_val = mse
            best_state = {k: t.detach().cpu().clone()
                          for k, t in net.state_dict().items()}

    print('[value] calibration (mean V -> mean R):')
    for mv, mr, n in _calibration(v_va, y_va):
        print(f'  V={mv:+.3f}  R={mr:+.3f}  n={n}')

    net.load_state_dict(best_state)
    torch.save(net.state_dict(), args.out)
    cfg_out = dict(cfg)
    cfg_out['source'] = 'bootstrap_critic'
    json.dump(cfg_out, open(args.out.replace('.pt', '_config.json'), 'w'))
    print(f'[value] saved {args.out}')


# ------------------------------------------------------------- finetune

def _masked_ce(logits, masks, actions, weights):
    """masks: torch bool tensor (N,34)，True=合法。"""
    neg = (~masks).float().mul(-1e9)
    logp_all = F.log_softmax(logits + neg, dim=1)
    logp = logp_all.gather(1, actions.view(-1, 1)).squeeze(1)
    ce = -logp
    return (ce * weights).sum() / weights.sum().clamp_min(1e-9), ce


def cmd_finetune(args):
    data = np.load(args.data)
    device = args.device

    # 1) critic 打分
    critic, _ = _build_from(args.critic)
    critic.to(device).eval()
    scores = _score_from_code(data['reason_code'])
    tr_mask, va_mask = _split(data)
    N_all = len(scores)
    V = np.zeros(N_all, dtype=np.float32)
    feats_all = data['feats']
    with torch.no_grad():
        for s in range(0, N_all, 16384):
            x = torch.from_numpy(feats_all[s:s + 16384]).to(device)
            V[s:s + 16384] = critic(x)[1].squeeze(-1).cpu().numpy()
    del critic
    A = scores - V
    a_mean, a_std = float(A[tr_mask].mean()), float(A[tr_mask].std())
    A_std = (A - a_mean) / max(a_std, 1e-6)
    print(f'[ft] A mean={a_mean:.4f} std={a_std:.4f}; '
          f'frac A>0 (train)={float((A[tr_mask] > 0).mean()):.3f}')

    if args.mode == 'awbc':
        W = np.exp(A_std / args.beta).astype(np.float32)
        W = np.clip(W, 0.0, args.wmax)
    elif args.mode == 'filtered':
        W = (A_std > 0).astype(np.float32)   # 高于平均优势的样本
    else:
        raise ValueError(args.mode)
    # 全长权重按 train split 归一化；训练循环里按数据集索引 mb 直接索引，避免错位
    W = W / W[tr_mask].mean()
    W_tr = W[tr_mask]
    print(f'[ft] mode={args.mode} beta={args.beta} '
          f'w: p10={np.percentile(W_tr,10):.3f} p50={np.percentile(W_tr,50):.3f} '
          f'p90={np.percentile(W_tr,90):.3f} max={W_tr.max():.2f}')

    # 2) 从 init 微调 policy
    net, cfg = _build_from(args.init)
    net.to(device)
    ref_net, _ = _build_from(args.init)
    ref_net.to(device).eval()
    for p in ref_net.parameters():
        p.requires_grad = False

    X = data['feats']
    Act = data['actions']
    Msk = data['masks']
    tr_idx = np.where(tr_mask)[0]
    va_idx = np.where(va_mask)[0]
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def _eval_split(indices, weights=None):
        net.eval()
        ces, agrees = [], []
        with torch.no_grad():
            for s in range(0, len(indices), 8192):
                mb = indices[s:s + 8192]
                x = torch.from_numpy(X[mb]).to(device)
                a = torch.from_numpy(Act[mb]).to(device)
                m = torch.from_numpy(Msk[mb]).to(device)
                logits = net(x)[0]
                ce, _ = _masked_ce(logits, m, a,
                                   torch.ones(len(mb), device=device))
                ref_logits = ref_net(x)[0]
                neg = (~m).float().mul(-1e9)
                agree = (logits + neg).argmax(1) == (ref_logits + neg).argmax(1)
                ces.append(float(ce) * len(mb))
                agrees.append(float(agree.float().mean()) * len(mb))
        return sum(ces) / len(indices), sum(agrees) / len(indices)

    ce0, ag0 = _eval_split(va_idx)
    print(f'[ft] before: val CE={ce0:.4f} argmax-agree={ag0:.4f}')

    for ep in range(args.epochs):
        net.train()
        np.random.shuffle(tr_idx)
        for s in range(0, len(tr_idx), args.batch):
            mb = tr_idx[s:s + args.batch]
            x = torch.from_numpy(X[mb]).to(device)
            a = torch.from_numpy(Act[mb]).to(device)
            m = torch.from_numpy(Msk[mb]).to(device)
            w = torch.from_numpy(W[mb].astype(np.float32)).to(device)
            logits = net(x)[0]
            loss, _ = _masked_ce(logits, m, a, w)
            if args.anchor_coef > 0:
                with torch.no_grad():
                    ref_logits = ref_net(x)[0]
                neg = (~m).float().mul(-1e9)
                logp = F.log_softmax(logits + neg, dim=1)
                ref_p = F.softmax(ref_logits + neg, dim=1)
                kl = (ref_p * (ref_p.clamp_min(1e-12).log() - logp)).sum(1).mean()
                loss = loss + args.anchor_coef * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
        ce_v, ag_v = _eval_split(va_idx)
        print(f'[ft] epoch {ep}: val CE={ce_v:.4f} argmax-agree={ag_v:.4f}',
              flush=True)

    torch.save(net.state_dict(), args.out)
    cfg_out = dict(cfg)
    cfg_out['source'] = f'bootstrap_{args.mode}_b{args.beta}'
    json.dump(cfg_out, open(args.out.replace('.pt', '_config.json'), 'w'))
    print(f'[ft] saved {args.out}')


# ------------------------------------------------------ finetune_response

def cmd_finetune_resp(args):
    """响应头 AWR：只对 response_fc/response_head 做加权 CE（主干冻结）。

    数据：collect 阶段记录的响应决策（feat 含 offered tile，4 类动作
    0=pass/1=peng/2=gang/3=hu）。critic 复用弃牌版（feat 同分布）。
    """
    data = np.load(args.data)
    device = args.device
    assert 'resp_feats' in data.files, 'no response records in data'

    critic, _ = _build_from(args.critic)
    critic.to(device).eval()
    scores = _score_from_code(data['resp_reason_code'])
    tr_mask, va_mask = _split({
        'game_id': data['resp_game_id'],
    })
    N_all = len(scores)
    V = np.zeros(N_all, dtype=np.float32)
    feats_all = data['resp_feats']
    with torch.no_grad():
        for s in range(0, N_all, 16384):
            x = torch.from_numpy(feats_all[s:s + 16384]).to(device)
            V[s:s + 16384] = critic(x)[1].squeeze(-1).cpu().numpy()
    del critic
    A = scores - V
    a_mean, a_std = float(A[tr_mask].mean()), float(A[tr_mask].std())
    A_std = (A - a_mean) / max(a_std, 1e-6)
    acts = data['resp_actions']
    names = ['pass', 'peng', 'gang', 'hu']
    print(f'[rft] N={N_all} A mean={a_mean:.4f} std={a_std:.4f}')
    for a_id in range(4):
        sel = acts == a_id
        if sel.sum():
            print(f'  action {names[a_id]:5s}: n={sel.sum():6d} '
                  f'meanR={scores[sel].mean():+.4f} meanA={A[sel].mean():+.4f}')

    if args.mode == 'awbc':
        W = np.exp(A_std / args.beta).astype(np.float32)
        W = np.clip(W, 0.0, args.wmax)
    elif args.mode == 'filtered':
        W = (A_std > 0).astype(np.float32)
    else:
        raise ValueError(args.mode)
    W = W / W[tr_mask].mean()

    net, cfg = _build_from(args.init)
    net.to(device)
    ref_net, _ = _build_from(args.init)
    ref_net.to(device).eval()
    for p in ref_net.parameters():
        p.requires_grad = False
    for name, p in net.named_parameters():
        p.requires_grad = name.startswith('response_')
    trainable = [p for p in net.parameters() if p.requires_grad]
    print(f'[rft] trainable params: {sum(p.numel() for p in trainable)}')
    opt = torch.optim.Adam(trainable, lr=args.lr)

    X = data['resp_feats']
    Act = data['resp_actions']
    Msk = data['resp_masks']
    tr_idx = np.where(tr_mask)[0]
    va_idx = np.where(va_mask)[0]

    def _resp_logits(model, x):
        for o in model(x):
            if o.shape[-1] == 4:
                return o
        raise RuntimeError('no response head output')

    def _eval_split(indices):
        net.eval()
        ces, agrees = [], []
        with torch.no_grad():
            for s in range(0, len(indices), 8192):
                mb = indices[s:s + 8192]
                x = torch.from_numpy(X[mb]).to(device)
                a = torch.from_numpy(Act[mb]).to(device)
                m = torch.from_numpy(Msk[mb]).to(device)
                neg = (~m).float().mul(-1e9)
                logits = _resp_logits(net, x) + neg
                logp = F.log_softmax(logits, dim=1)
                ce = -logp.gather(1, a.view(-1, 1)).squeeze(1)
                ref_logits = _resp_logits(ref_net, x) + neg
                agree = logits.argmax(1) == ref_logits.argmax(1)
                ces.append(float(ce.mean()) * len(mb))
                agrees.append(float(agree.float().mean()) * len(mb))
        return sum(ces) / len(indices), sum(agrees) / len(indices)

    ce0, ag0 = _eval_split(va_idx)
    print(f'[rft] before: val CE={ce0:.4f} argmax-agree={ag0:.4f}')

    for ep in range(args.epochs):
        net.train()
        np.random.shuffle(tr_idx)
        for s in range(0, len(tr_idx), args.batch):
            mb = tr_idx[s:s + args.batch]
            x = torch.from_numpy(X[mb]).to(device)
            a = torch.from_numpy(Act[mb]).to(device)
            m = torch.from_numpy(Msk[mb]).to(device)
            w = torch.from_numpy(W[mb].astype(np.float32)).to(device)
            neg = (~m).float().mul(-1e9)
            logp = F.log_softmax(_resp_logits(net, x) + neg, dim=1)
            ce = -logp.gather(1, a.view(-1, 1)).squeeze(1)
            loss = (ce * w).sum() / w.sum().clamp_min(1e-9)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 0.5)
            opt.step()
        ce_v, ag_v = _eval_split(va_idx)
        print(f'[rft] epoch {ep}: val CE={ce_v:.4f} argmax-agree={ag_v:.4f}',
              flush=True)

    torch.save(net.state_dict(), args.out)
    cfg_out = dict(cfg)
    cfg_out['source'] = f'bootstrap_resp_{args.mode}_b{args.beta}'
    json.dump(cfg_out, open(args.out.replace('.pt', '_config.json'), 'w'))
    print(f'[rft] saved {args.out}')


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('collect')
    p.add_argument('--games', type=int, default=24000)
    p.add_argument('--workers', type=int, default=96)
    p.add_argument('--out-prefix', required=True)
    p.add_argument('--init', default='output/nn_full_action_best.pt')
    p.add_argument('--seed-base', type=int, default=50_000_000)
    p.add_argument('--progress-every', type=int, default=1)
    p.add_argument('--temps', type=str, default='0.7,0.85,1.0,1.2',
                   help='逗号分隔的采样温度（按局轮转；0=argmax）')
    p.set_defaults(fn=cmd_collect)

    p = sub.add_parser('train_value')
    p.add_argument('--data', required=True)
    p.add_argument('--init', default='output/nn_full_action_best.pt')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--full-trunk', action='store_true',
                   help='解冻全部参数（默认只训 value_fc/value_head）')
    p.set_defaults(fn=cmd_train_value)

    p = sub.add_parser('finetune')
    p.add_argument('--data', required=True)
    p.add_argument('--init', default='output/nn_full_action_best.pt')
    p.add_argument('--critic', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--mode', choices=['awbc', 'filtered'], default='awbc')
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--wmax', type=float, default=20.0)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--anchor-coef', type=float, default=0.0)
    p.set_defaults(fn=cmd_finetune)

    p = sub.add_parser('finetune_resp')
    p.add_argument('--data', required=True)
    p.add_argument('--init', default='output/nn_full_action_best.pt')
    p.add_argument('--critic', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--mode', choices=['awbc', 'filtered'], default='awbc')
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--wmax', type=float, default=20.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch', type=int, default=4096)
    p.set_defaults(fn=cmd_finetune_resp)

    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
