# -*- coding: utf-8 -*-
"""PPO 自对弈训练（方案 B）。

流程：warm-start from output/nn_model.pt -> 每 iter 自对弈采样 K 局（可多进程）
-> GAE 优势 -> PPO clip 更新（masked policy + value + entropy）-> checkpoint。

用法：
    PYTHONPATH=. python3 scripts/rl/train_ppo.py \
        --iters 5 --games-per-iter 100 --workers 0 --device cuda:0

产物（绝不覆盖 best）：
    output/nn_rl_ppo.pt / .json         最新权重
    output/nn_rl_ppo_iter{N}.pt         周期 snapshot
    output/nn_rl_ppo.checkpoint.pt      断点续训（含 optimizer/iter）
    output/rl_ppo_train.log             训练日志
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from algo.nn.model import MahjongNet, build_model
from algo.rl.selfplay import collect_games, play_selfplay_game, build_net, PPOActorAgent
from algo.rl.reward import DEFAULT_REWARD
from algo.rl.ppo import flatten_trajectories

OUT = 'output'
WARM_START = os.path.join(OUT, 'nn_model.pt')
WARM_CFG = os.path.join(OUT, 'nn_model_config.json')


# ----------------------------- 工具 -----------------------------

def _load_state(path):
    sd = torch.load(path, map_location='cpu')
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    return sd


def cpu_state_dict(net):
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}


def masked_logp_entropy(logits, masks, actions):
    """返回 (选中动作 logp, 每样本熵)，非法动作屏蔽。"""
    neg = (masks - 1.0) * 1e9            # 合法 0，非法 -1e9
    logits = logits + neg
    logp_all = F.log_softmax(logits, dim=1)
    probs = logp_all.exp()
    logp = logp_all.gather(1, actions.view(-1, 1)).squeeze(1)
    entropy = -(probs * logp_all * masks).sum(dim=1)
    return logp, entropy


# --------------------------- 并行采样 ---------------------------

def _collect_worker(args):
    (state_dict, n_games, seed_base, reward_cfg, config, temperature,
     n_opponents, opponents) = args
    return collect_games(state_dict, n_games, seed_base=seed_base,
                         reward_cfg=reward_cfg, config=config,
                         device='cpu', temperature=temperature,
                         n_opponents=n_opponents, opponents=opponents,
                         num_threads=1)


def collect_parallel(pool, state_dict, games_per_iter, workers, seed_base,
                     reward_cfg, config, temperature, n_opponents, opponents):
    per = max(1, games_per_iter // workers)
    tasks = []
    for w in range(workers):
        tasks.append((state_dict, per, seed_base + w * 100000, reward_cfg,
                      config, temperature, n_opponents, opponents))
    trajs = []
    for res in pool.imap_unordered(_collect_worker, tasks):
        trajs.extend(res)
    return trajs


# --------------------------- 评估 ---------------------------

def eval_vs_frozen(cur_net, frozen_net, n_games=200, seed_base=777, device='cpu'):
    """current 放 1 个座位，frozen 放其余 3 个座位，轮换座位统计 current 胜率。

    公平基线：4 个相同策略时，current 只占 1/4 座位，在 decisive 局中期望胜率 0.25。
    返回 (win_share_decisive, wins, decisive, draws)：
        win_share_decisive = wins / decisive，>0.25 说明 current 相对 frozen 有提升。
    """
    from driver.engine import play_game
    import random
    wins = 0
    decisive = 0
    for g in range(n_games):
        random.seed(seed_base + g)
        np.random.seed((seed_base + g) % (2 ** 31 - 1))
        cur_seat = g % 4
        agents = []
        for s in range(4):
            net = cur_net if s == cur_seat else frozen_net
            agents.append(PPOActorAgent(f'S@{s}', net, device=device,
                                        deterministic=True, record=False))
        result = play_game(agents)
        if result['win_type'] != 'draw':
            decisive += 1
            if result.get('winner') == f'S@{cur_seat}':
                wins += 1
    draws = n_games - decisive
    return wins / max(1, decisive), wins, decisive, draws


# --------------------------- 主训练 ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=5)
    ap.add_argument('--games-per-iter', type=int, default=100)
    ap.add_argument('--workers', type=int, default=0, help='0=串行')
    ap.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--lr', type=float, default=2.5e-4)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--minibatch', type=int, default=1024)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--ent-coef', type=float, default=0.01)
    ap.add_argument('--ent-coef-final', type=float, default=None,
                    help='线性退火到该值；None=不退火')
    ap.add_argument('--val-coef', type=float, default=0.5)
    ap.add_argument('--win-reward', type=float, default=1.0)
    ap.add_argument('--deal-in-reward', type=float, default=-1.0)
    ap.add_argument('--other-loss-reward', type=float, default=-1.0)
    ap.add_argument('--draw-reward', type=float, default=0.0)
    ap.add_argument('--eval-deterministic', type=int, default=1,
                    help='1=argmax 评估（更干净），0=采样评估')
    ap.add_argument('--n-opponents', type=int, default=0,
                    help='每局把多少个座位换成固定对手（0=纯自对弈，3=纯对手训练）')
    ap.add_argument('--opponents', type=str, default='baseline,beliefexp',
                    help='对手 spec，逗号分隔：baseline|beliefexp|v3eval0')
    ap.add_argument('--max-grad-norm', type=float, default=0.5)
    ap.add_argument('--target-kl', type=float, default=0.03)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--eval-every', type=int, default=1)
    ap.add_argument('--eval-games', type=int, default=200)
    ap.add_argument('--snapshot-every', type=int, default=5)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--init', type=str, default='',
                    help='初始权重路径（默认用 output/nn_model.pt warm-start）')
    ap.add_argument('--config', type=str, default='',
                    help='网络配置 json（默认 nn_model_config.json；conv 用 BC 的 config）')
    ap.add_argument('--tag', type=str, default='nn_rl_ppo')
    args = ap.parse_args()

    init_path = args.init or WARM_START
    # 配置来源：显式 --config > --init 同名 _config.json > 默认 nn_model_config.json
    if args.config:
        cfg_path = args.config
    else:
        cand = init_path.replace('.pt', '_config.json')
        cfg_path = cand if os.path.exists(cand) else WARM_CFG
    config = json.load(open(cfg_path))
    input_dim = config.get('input_dim', 175)
    hidden_dim = config.get('hidden_dim', 256)
    device = args.device
    print(f'[config] {cfg_path} arch={config.get("arch", "mlp")}')

    reward_cfg = {
        'win': args.win_reward,
        'deal_in': args.deal_in_reward,
        'other_loss': args.other_loss_reward,
        'draw': args.draw_reward,
    }
    opponents = args.opponents.split(',') if args.n_opponents > 0 else None

    net = build_model(config).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    start_iter = 0
    ckpt_path = os.path.join(OUT, f'{args.tag}.checkpoint.pt')
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ck['model_state_dict'])
        opt.load_state_dict(ck['optimizer_state_dict'])
        start_iter = ck['iter'] + 1
        print(f'[resume] from iter {start_iter}')
    else:
        net.load_state_dict(_load_state(init_path))
        print(f'[init] loaded {init_path}')

    # frozen 参照（评估用）= 初始模型，固定不变；衡量相对初始化的提升
    frozen_net = build_net(_load_state(init_path), config, device=device)

    pool = None
    if args.workers > 0:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(args.workers)
        print(f'[pool] spawned {args.workers} workers')

    logf = open(os.path.join(OUT, 'rl_ppo_train.log'), 'a')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line)
        logf.write(line + '\n')
        logf.flush()

    log(f'=== PPO train start: iters={args.iters} games/iter={args.games_per_iter} '
        f'workers={args.workers} lr={args.lr} device={device} ===')

    for it in range(start_iter, args.iters):
        t0 = time.time()
        net.eval()
        sd_cpu = cpu_state_dict(net)
        seed_base = 10_000_000 + it * 1_000_000
        if pool is not None:
            trajs = collect_parallel(pool, sd_cpu, args.games_per_iter, args.workers,
                                     seed_base, reward_cfg, config, args.temperature,
                                     args.n_opponents, opponents)
        else:
            trajs = collect_games(sd_cpu, args.games_per_iter, seed_base=seed_base,
                                  reward_cfg=reward_cfg, config=config,
                                  device='cpu', temperature=args.temperature,
                                  n_opponents=args.n_opponents, opponents=opponents,
                                  num_threads=None)
        t_collect = time.time() - t0

        batch = flatten_trajectories(trajs, gamma=args.gamma, lam=args.lam)
        if batch is None:
            log(f'[iter {it}] no data, skip')
            continue

        N = len(batch['actions'])
        feats = torch.from_numpy(batch['feats']).to(device)
        actions = torch.from_numpy(batch['actions']).to(device)
        old_logp = torch.from_numpy(batch['old_logp']).to(device)
        masks = torch.from_numpy(batch['masks']).to(device)
        adv = torch.from_numpy(batch['advantages']).to(device)
        rets = torch.from_numpy(batch['returns']).to(device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        net.train()
        idx = np.arange(N)
        stop = False
        last = {}
        if args.ent_coef_final is not None and args.iters > 1:
            frac = it / (args.iters - 1)
            ent_coef = args.ent_coef + frac * (args.ent_coef_final - args.ent_coef)
        else:
            ent_coef = args.ent_coef
        for ep in range(args.epochs):
            np.random.shuffle(idx)
            approx_kls = []
            for start in range(0, N, args.minibatch):
                mb = idx[start:start + args.minibatch]
                mb_t = torch.from_numpy(mb).to(device)
                out = net(feats[mb_t])
                logits = out[0]
                value = out[1].squeeze(-1)
                logp, entropy = masked_logp_entropy(logits, masks[mb_t], actions[mb_t])
                ratio = torch.exp(logp - old_logp[mb_t])
                a = adv[mb_t]
                s1 = ratio * a
                s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a
                pol_loss = -torch.min(s1, s2).mean()
                val_loss = F.mse_loss(value, rets[mb_t])
                ent = entropy.mean()
                loss = pol_loss + args.val_coef * val_loss - ent_coef * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    approx_kl = (old_logp[mb_t] - logp).mean().item()
                approx_kls.append(approx_kl)
                last = {'pol': pol_loss.item(), 'val': val_loss.item(),
                        'ent': ent.item(), 'kl': approx_kl}
            mean_kl = float(np.mean(approx_kls))
            if args.target_kl and mean_kl > 1.5 * args.target_kl:
                log(f'[iter {it}] early-stop epoch {ep} (kl {mean_kl:.4f})')
                stop = True
                break

        mean_ret = float(batch['returns'].mean())
        win_rate = np.mean([1.0 for tr in trajs if tr['reward'] > 0]) if trajs else 0.0
        n_win = sum(1 for tr in trajs if tr['reward'] > 0)
        n_deal_in = sum(1 for tr in trajs if tr['reason'] == 'deal_in')
        log(f'[iter {it}] N={N} traj={len(trajs)} collect={t_collect:.1f}s '
            f'pol={last.get("pol",0):.4f} val={last.get("val",0):.4f} '
            f'ent={last.get("ent",0):.3f} kl={last.get("kl",0):.4f} '
            f'meanRet={mean_ret:.3f} win={n_win}/{len(trajs)} dealin={n_deal_in}')

        # checkpoint（每 iter）
        torch.save({'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(), 'iter': it}, ckpt_path)
        torch.save(net.state_dict(), os.path.join(OUT, f'{args.tag}.pt'))
        json.dump(dict(config, source='ppo_selfplay'),
                  open(os.path.join(OUT, f'{args.tag}_config.json'), 'w'))
        if args.snapshot_every and (it + 1) % args.snapshot_every == 0:
            torch.save(net.state_dict(), os.path.join(OUT, f'{args.tag}_iter{it}.pt'))

        # 评估 vs frozen warm-start
        if args.eval_every and (it + 1) % args.eval_every == 0:
            net.eval()
            wr, w, dec, dr = eval_vs_frozen(net, frozen_net, n_games=args.eval_games,
                                            device=device)
            log(f'[iter {it}] eval vs frozen: current win_share={wr:.3f} '
                f'({w}/{dec} decisive, {dr} draws) [0.25=持平, >0.25=提升]')

    if pool is not None:
        pool.close()
        pool.join()
    logf.close()
    print('done.')


if __name__ == '__main__':
    main()
