# -*- coding: utf-8 -*-
"""完整动作空间 PPO 自对弈微调。

用法：
    PYTHONPATH=. python3 scripts/rl/train_full_action_ppo.py \
        --init output/nn_full_action_best.pt \
        --iters 10 --games-per-iter 100 --workers 8 --device cuda:0
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from algo.nn.model import build_model
from algo.rl.full_action_ppo import collect_games, flatten_trajectories, build_net
from algo.rl.reward import DEFAULT_REWARD

OUT = 'output'


def _load_state(path):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict):
        if 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        elif 'model_state' in sd:
            sd = sd['model_state']
    return sd


def cpu_state_dict(net):
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}


def masked_logp_entropy(logits, masks, actions):
    neg = (masks - 1.0) * 1e9
    logits = logits + neg
    logp_all = F.log_softmax(logits, dim=1)
    probs = logp_all.exp()
    logp = logp_all.gather(1, actions.view(-1, 1)).squeeze(1)
    entropy = -(probs * logp_all * masks).sum(dim=1)
    return logp, entropy


def _collect_worker(args):
    (state_dict, config, n_games, seed_base, reward_cfg,
     device, temperature, deterministic) = args
    return collect_games(state_dict, config, n_games, seed_base=seed_base,
                         reward_cfg=reward_cfg, device=device,
                         deterministic=deterministic, temperature=temperature,
                         num_threads=1)


def collect_parallel(pool, state_dict, config, games_per_iter, workers,
                     seed_base, reward_cfg, device, temperature, deterministic):
    per = max(1, games_per_iter // workers)
    tasks = []
    for w in range(workers):
        tasks.append((state_dict, config, per, seed_base + w * 100000,
                      reward_cfg, device, temperature, deterministic))
    trajs = []
    for res in pool.imap_unordered(_collect_worker, tasks):
        trajs.extend(res)
    return trajs


def eval_vs_frozen(cur_net, frozen_net, config, n_games=200,
                   seed_base=777, device='cpu'):
    from driver.engine import play_game
    import random
    from algo.rl.full_action_ppo import FullActionPPOAgent
    wins = 0
    decisive = 0
    for g in range(n_games):
        random.seed(seed_base + g)
        np.random.seed((seed_base + g) % (2 ** 31 - 1))
        cur_seat = g % 4
        agents = []
        for s in range(4):
            if s == cur_seat:
                ag = FullActionPPOAgent(f'S@{s}', model_path='', device=device,
                                        deterministic=True, record=False)
                ag._net = cur_net
                ag.config = config
            else:
                ag = FullActionPPOAgent(f'S@{s}', model_path='', device=device,
                                        deterministic=True, record=False)
                ag._net = frozen_net
                ag.config = config
            agents.append(ag)
        result = play_game(agents)
        if result['win_type'] != 'draw':
            decisive += 1
            if result.get('winner') == f'S@{cur_seat}':
                wins += 1
    draws = n_games - decisive
    return wins / max(1, decisive), wins, decisive, draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', type=str, default='output/nn_full_action_best.pt')
    ap.add_argument('--iters', type=int, default=10)
    ap.add_argument('--games-per-iter', type=int, default=100)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--minibatch', type=int, default=1024)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--ent-coef', type=float, default=0.01)
    ap.add_argument('--ent-coef-final', type=float, default=None)
    ap.add_argument('--val-coef', type=float, default=0.5)
    ap.add_argument('--reward', type=str, default='1.0,-1.0,-1.0,0.0',
                    help='win,deal_in,other_loss,draw')
    ap.add_argument('--max-grad-norm', type=float, default=0.5)
    ap.add_argument('--target-kl', type=float, default=0.03)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--eval-every', type=int, default=1)
    ap.add_argument('--eval-games', type=int, default=200)
    ap.add_argument('--snapshot-every', type=int, default=5)
    ap.add_argument('--tag', type=str, default='nn_full_action_ppo')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    cfg_path = args.init.replace('.pt', '_config.json')
    config = json.load(open(cfg_path))
    device = args.device
    print(f'[config] {cfg_path} arch={config.get("arch", "mlp")}')

    rw = list(map(float, args.reward.split(',')))
    reward_cfg = {'win': rw[0], 'deal_in': rw[1], 'other_loss': rw[2], 'draw': rw[3]}

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
        net.load_state_dict(_load_state(args.init))
        print(f'[init] loaded {args.init}')

    frozen_net = build_net(_load_state(args.init), config, device=device)

    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(args.workers)
    print(f'[pool] spawned {args.workers} workers')

    logf = open(os.path.join(OUT, f'{args.tag}_train.log'), 'a')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line)
        logf.write(line + '\n')
        logf.flush()

    log(f'=== FullAction PPO start: iters={args.iters} games/iter={args.games_per_iter} '
        f'workers={args.workers} lr={args.lr} device={device} ===')

    for it in range(start_iter, args.iters):
        t0 = time.time()
        net.eval()
        sd_cpu = cpu_state_dict(net)
        seed_base = 20_000_000 + it * 1_000_000
        trajs = collect_parallel(pool, sd_cpu, config, args.games_per_iter,
                                 args.workers, seed_base, reward_cfg, 'cpu',
                                 args.temperature, False)
        t_collect = time.time() - t0

        batch = flatten_trajectories(trajs, gamma=args.gamma, lam=args.lam)
        if batch is None:
            log(f'[iter {it}] no data, skip')
            continue

        N = len(batch['actions'])
        feats = torch.from_numpy(batch['feats']).to(device)
        actions = torch.from_numpy(batch['actions']).to(device)
        old_logp = torch.from_numpy(batch['old_logp']).to(device)
        mask_disc = torch.from_numpy(batch['mask_disc']).to(device)
        mask_resp = torch.from_numpy(batch['mask_resp']).to(device)
        head = torch.from_numpy(batch['head']).to(device)
        adv = torch.from_numpy(batch['advantages']).to(device)
        rets = torch.from_numpy(batch['returns']).to(device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        net.train()
        idx = np.arange(N)
        last = {}
        ent_coef = args.ent_coef
        if args.ent_coef_final is not None and args.iters > 1:
            frac = it / (args.iters - 1)
            ent_coef = args.ent_coef + frac * (args.ent_coef_final - args.ent_coef)

        for ep in range(args.epochs):
            np.random.shuffle(idx)
            approx_kls = []
            for start in range(0, N, args.minibatch):
                mb = idx[start:start + args.minibatch]
                mb_t = torch.from_numpy(mb).to(device)
                h = head[mb_t]
                d_idx = (h == 0).nonzero(as_tuple=True)[0]
                r_idx = (h == 1).nonzero(as_tuple=True)[0]

                out = net(feats[mb_t])
                disc_logits, value, resp_logits = out[0], out[1], out[-1]
                value = value.squeeze(-1)

                new_logp_list = []
                entropy_list = []
                if len(d_idx) > 0:
                    mb_d = mb_t[d_idx]
                    lp_d, ent_d = masked_logp_entropy(
                        disc_logits[d_idx], mask_disc[mb_d], actions[mb_d])
                    new_logp_list.append(lp_d)
                    entropy_list.append(ent_d)
                if len(r_idx) > 0:
                    mb_r = mb_t[r_idx]
                    lp_r, ent_r = masked_logp_entropy(
                        resp_logits[r_idx], mask_resp[mb_r], actions[mb_r])
                    new_logp_list.append(lp_r)
                    entropy_list.append(ent_r)

                new_logp = torch.cat(new_logp_list)
                entropy = torch.cat(entropy_list)

                ratio = torch.exp(new_logp - old_logp[mb_t])
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
                    approx_kl = (old_logp[mb_t] - new_logp).mean().item()
                approx_kls.append(approx_kl)
                last = {'pol': pol_loss.item(), 'val': val_loss.item(),
                        'ent': ent.item(), 'kl': approx_kl}
            mean_kl = float(np.mean(approx_kls))
            if args.target_kl and mean_kl > 1.5 * args.target_kl:
                log(f'[iter {it}] early-stop epoch {ep} (kl {mean_kl:.4f})')
                break

        mean_ret = float(batch['returns'].mean())
        n_win = sum(1 for tr in trajs if tr['reward'] > 0)
        n_deal_in = sum(1 for tr in trajs if tr['reason'] == 'deal_in')
        log(f'[iter {it}] N={N} traj={len(trajs)} collect={t_collect:.1f}s '
            f'pol={last.get("pol",0):.4f} val={last.get("val",0):.4f} '
            f'ent={last.get("ent",0):.3f} kl={last.get("kl",0):.4f} '
            f'meanRet={mean_ret:.3f} win={n_win}/{len(trajs)} dealin={n_deal_in}')

        torch.save({'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(), 'iter': it}, ckpt_path)
        torch.save(net.state_dict(), os.path.join(OUT, f'{args.tag}.pt'))
        json.dump(dict(config, source='full_action_ppo'),
                  open(os.path.join(OUT, f'{args.tag}_config.json'), 'w'))
        if args.snapshot_every and (it + 1) % args.snapshot_every == 0:
            torch.save(net.state_dict(), os.path.join(OUT, f'{args.tag}_iter{it}.pt'))

        if args.eval_every and (it + 1) % args.eval_every == 0:
            net.eval()
            wr, w, dec, dr = eval_vs_frozen(net, frozen_net, config,
                                            n_games=args.eval_games, device=device)
            log(f'[iter {it}] eval vs frozen: current win_share={wr:.3f} '
                f'({w}/{dec} decisive, {dr} draws) [0.25=持平, >0.25=提升]')

    pool.close()
    pool.join()
    logf.close()
    print('done.')


if __name__ == '__main__':
    main()
