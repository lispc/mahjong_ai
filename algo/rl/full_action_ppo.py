# -*- coding: utf-8 -*-
"""完整动作空间（弃牌 + 碰/杠/胡响应）自对弈 PPO Agent 与采样工具。"""

import random
import numpy as np
import torch

import agent
from algo.agents.ppo_agent import PPOAgent
from algo.nn.features import extract_features, _TILE_TO_IDX, _IDX_TO_TILE
from algo.rl.reward import seat_reward, terminal_reason

NUM_DISCARD_ACTIONS = 34
NUM_RESPONSE_ACTIONS = 4  # pass/peng/gang/hu


def _split_outputs(net, cfg, out):
    """解析 TileConvNet 返回值。"""
    idx = 0
    disc = out[idx]; idx += 1
    val = out[idx]; idx += 1
    if cfg.get('dealin_head', False):
        idx += 1
    if cfg.get('candidate_value_head', False):
        idx += 1
    resp = None
    if cfg.get('response_head', False):
        resp = out[idx]
    return disc, val, resp


class FullActionPPOAgent(PPOAgent):
    """用同一 policy-value 网络同时采样弃牌与响应，并记录轨迹。

    继承 PPOAgent 只为复用上下文维护与网络加载缓存；
    next() / respond_* 被覆盖以支持完整动作空间。
    """

    def __init__(self, name, model_path='output/nn_full_action_best.pt',
                 device='cpu', deterministic=False, temperature=1.0,
                 record=True, verbose=False):
        # 绕过 PPOAgent.__init__ 的模型加载参数，直接设字段
        super(PPOAgent, self).__init__(name, verbose=verbose)
        self.model_path = model_path
        self.device = device
        self.deterministic = deterministic
        self.temperature = temperature
        self.record = record
        self.traj = []
        self._net = None
        self.config = None
        self._extract = extract_features

    def handle_msg(self, msg):
        # 只更新 ContextV3，不复用基类 Agent.handle_msg 的“摸到弃牌就加入手牌”副作用，
        # 否则在副露/响应场景下手牌数会错乱，触发 dict_sub assert。
        if msg.type == 'put':
            self.context.see_tile(msg.data, msg.sender)
        elif msg.type == 'tenpai':
            self.context.declare_tenpai(msg.sender)
        self._belief = None
        from agent import Message
        return Message(self.name, 'no_op', None)

    def init_tiles(self, l):
        super().init_tiles(l)
        self.traj = []

    def _net_obj(self):
        if self._net is None:
            from algo.nn.model import build_model
            import json as _json, os
            cfg_path = self.model_path.replace('.pt', '_config.json')
            if not os.path.exists(cfg_path):
                cfg_path = os.path.join(os.path.dirname(self.model_path), 'nn_model_config.json')
            self.config = _json.load(open(cfg_path)) if os.path.exists(cfg_path) else {'arch': 'mlp', 'input_dim': 175, 'hidden_dim': 256}
            self._net = build_model(self.config)
            sd = torch.load(self.model_path, map_location='cpu', weights_only=False)
            if isinstance(sd, dict):
                if 'model_state_dict' in sd:
                    sd = sd['model_state_dict']
                elif 'model_state' in sd:
                    sd = sd['model_state']
            self._net.load_state_dict(sd, strict=False)
            self._net.eval()
            self._net.to(self.device)
            self._extract = extract_features
        return self._net

    def _discard_legal_mask(self):
        legal = np.zeros(NUM_DISCARD_ACTIONS, dtype=np.float32)
        for t in self.cur:
            legal[int(_TILE_TO_IDX[t])] = 1.0
        return legal

    def _response_legal_mask(self, tile_val):
        legal = np.ones(NUM_RESPONSE_ACTIONS, dtype=np.float32)
        # 0=pass 永远合法
        legal[1] = 1.0 if self._can_peng(tile_val) else 0.0
        legal[2] = 1.0 if self._can_gang(tile_val) else 0.0
        legal[3] = 1.0 if super(PPOAgent, self).respond_hu(tile_val, self.context) else 0.0
        return legal

    def _sample(self, logits, legal_mask):
        masked = logits.clone()
        masked[legal_mask == 0] = -1e9
        masked = masked / max(self.temperature, 1e-6)
        masked = masked - masked.max()
        probs = torch.exp(masked)
        probs = probs / probs.sum()
        if self.deterministic:
            a = int(probs.argmax())
        else:
            a = int(torch.multinomial(probs, 1).item())
        logp = float(torch.log(probs[a] + 1e-12).item())
        return a, logp

    def next(self):
        assert len(self.cur) >= 1
        net = self._net_obj()
        feats = self._extract(self.context, self.full_hand(), self.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = net(x)
            disc_logits, value, resp_logits = _split_outputs(net, self.config, out)
            disc_logits = disc_logits.squeeze(0)
            value = float(value.detach().cpu().reshape(-1)[0])
        legal = self._discard_legal_mask()
        a, logp = self._sample(disc_logits, torch.from_numpy(legal).to(self.device))
        if self.record:
            self.traj.append({
                'feat': np.asarray(feats, dtype=np.float32),
                'action': a,
                'logp': logp,
                'value': value,
                'mask_disc': legal,
                'mask_resp': np.zeros(NUM_RESPONSE_ACTIONS, dtype=np.float32),
                'head': 0,
            })
        tile_val = int(_IDX_TO_TILE[a])
        self.cur.remove(tile_val)
        self.context.see_tile(tile_val, self.name)
        self._belief = None
        return tile_val

    def _record_response(self, tile_val):
        net = self._net_obj()
        feats = self._extract(self.context, self.full_hand() + [tile_val], self.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = net(x)
            disc_logits, value, resp_logits = _split_outputs(net, self.config, out)
            resp_logits = resp_logits.squeeze(0)
            value = float(value.detach().cpu().reshape(-1)[0])
        legal = self._response_legal_mask(tile_val)
        a, logp = self._sample(resp_logits, torch.from_numpy(legal).to(self.device))
        if self.record:
            self.traj.append({
                'feat': np.asarray(feats, dtype=np.float32),
                'action': a,
                'logp': logp,
                'value': value,
                'mask_disc': np.zeros(NUM_DISCARD_ACTIONS, dtype=np.float32),
                'mask_resp': legal,
                'head': 1,
            })
        return a

    def respond_hu(self, tile_val, context=None):
        a = self._record_response(tile_val)
        return a == 3

    def respond_peng(self, tile_val, context=None):
        a = self._record_response(tile_val)
        return a == 1

    def respond_gang(self, tile_val, context=None):
        a = self._record_response(tile_val)
        return a == 2

    def declare_tenpai(self, hand, context):
        # 复用 PPOAgent 的 tenpai head 逻辑
        if not getattr(self, '_tenpai_use_head', None):
            net = self._net_obj()
            self._tenpai_use_head = self.config.get('tenpai_head', False)
        if self._tenpai_use_head and context is not None:
            try:
                feats = self._extract(context, hand, self.name)
                x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logit = self._net.tenpai_logit(x)
                return bool(logit.item() > 0.0)
            except Exception:
                return super(PPOAgent, self).declare_tenpai(hand, context)
        return super(PPOAgent, self).declare_tenpai(hand, context)


def build_net(state_dict, config, device='cpu'):
    from algo.nn.model import build_model
    net = build_model(config)
    net.load_state_dict(state_dict)
    net.eval()
    net.to(device)
    return net


def play_selfplay_game(net, config, seed=None, reward_cfg=None, device='cpu',
                       deterministic=False, temperature=1.0):
    from driver.engine import play_game
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2 ** 31 - 1))
    agents = [FullActionPPOAgent(f'A@{s}', model_path='',
                                 device=device, deterministic=deterministic,
                                 temperature=temperature, record=True)
              for s in range(4)]
    # 把同一个网络对象或路径分给所有 agent；这里用路径避免多进程传 net 对象
    for ag in agents:
        ag.model_path = net.model_path if hasattr(net, 'model_path') else str(net)
        ag._net = net if isinstance(net, torch.nn.Module) else None
        if ag._net is not None:
            ag.config = config
    result = play_game(agents)
    trajs = []
    for ag in agents:
        if ag.traj:
            trajs.append({
                'steps': ag.traj,
                'reward': seat_reward(result, ag.name, reward_cfg),
                'seat': ag.name,
                'reason': terminal_reason(result, ag.name),
            })
    return trajs, result


def collect_games(state_dict, config, n_games, seed_base=0, reward_cfg=None,
                  device='cpu', deterministic=False, temperature=1.0,
                  num_threads=1):
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    net = build_net(state_dict, config, device=device)
    net.model_path = ''  # dummy
    out = []
    for i in range(n_games):
        trajs, _ = play_selfplay_game(net, config, seed=seed_base + i,
                                      reward_cfg=reward_cfg, device=device,
                                      deterministic=deterministic,
                                      temperature=temperature)
        out.extend(trajs)
    return out


def flatten_trajectories(trajs, gamma=1.0, lam=0.95):
    """展平完整动作轨迹，返回训练 batch。"""
    from algo.rl.ppo import compute_gae
    feats, actions, old_logp, mask_disc, mask_resp, head, advs, rets, vals = [], [], [], [], [], [], [], [], []
    for tr in trajs:
        steps = tr['steps']
        if not steps:
            continue
        v = np.array([s['value'] for s in steps], dtype=np.float64)
        adv, ret = compute_gae(v, tr['reward'], gamma=gamma, lam=lam)
        for i, s in enumerate(steps):
            feats.append(s['feat'])
            actions.append(s['action'])
            old_logp.append(s['logp'])
            mask_disc.append(s['mask_disc'])
            mask_resp.append(s['mask_resp'])
            head.append(s['head'])
            vals.append(s['value'])
        advs.append(adv)
        rets.append(ret)
    if not feats:
        return None
    return {
        'feats': np.asarray(feats, dtype=np.float32),
        'actions': np.asarray(actions, dtype=np.int64),
        'old_logp': np.asarray(old_logp, dtype=np.float32),
        'mask_disc': np.asarray(mask_disc, dtype=np.float32),
        'mask_resp': np.asarray(mask_resp, dtype=np.float32),
        'head': np.asarray(head, dtype=np.int64),
        'advantages': np.concatenate(advs).astype(np.float32),
        'returns': np.concatenate(rets).astype(np.float32),
        'values': np.asarray(vals, dtype=np.float32),
    }
