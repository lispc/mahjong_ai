# -*- coding: utf-8 -*-
"""训练对手序列模型（方向 D2，docs/designs/silent-tenpai-seq-model.md）。

输入：gen_seq_opp_data.py 生成的 npz（feats 175 + 3 对手弃牌序列/报听/副露）。
任务：tenpai 3 维 BCE（含默听）+ tenpai 行 wait 3x34 sigmoid。
内置 --no-seq 消融（同架构去掉序列编码器），用于判定序列信息是否有增量。

注意：175 维特征含对手的报听 flag，已报听者的 tenpai 标签是"免费"的（泄漏）。
因此评估一律拆分：**all**（全部对手）与 **silent**（仅未报听对手）两档，
离线门只看 silent 档（docs/designs/silent-tenpai-seq-model.md §4）。

用法：
    PYTHONPATH=. python3 scripts/rl/train_seq_opp_model.py \
        output/seq_opp_data_20000.npz output/nn_seq_opp.pt \
        --epochs 30 --batch 512 --device cuda:0
    # 消融对照
    PYTHONPATH=. python3 scripts/rl/train_seq_opp_model.py \
        output/seq_opp_data_20000.npz output/nn_seq_opp_noseq.pt --no-seq
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

MAX_SEQ = 40
N_TILES = 34
PAD_IDX = 34


class OppSeqEncoder(nn.Module):
    """单对手弃牌序列 → 80 维向量（3 对手共享权重）。"""

    def __init__(self, emb_dim=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(N_TILES + 1, emb_dim, padding_idx=PAD_IDX)
        # step 特征：pos_norm, post_decl, is_decl_tile, is_pad
        self.gru = nn.GRU(emb_dim + 4, hidden, batch_first=True)
        self.meld_proj = nn.Linear(N_TILES, 16)

    def forward(self, seq, seq_len, decl_step, meld_hot):
        # seq: (B, 40) int（-1=pad）；seq_len: (B,)；decl_step: (B,)（-1 未报听）
        B = seq.shape[0]
        pad_idx = torch.full_like(seq, PAD_IDX)
        is_pad = torch.arange(MAX_SEQ, device=seq.device).unsqueeze(0) >= seq_len.unsqueeze(1)
        seq = torch.where(is_pad, pad_idx, seq.clamp(min=0))
        e = self.emb(seq)  # (B, 40, emb)

        pos = torch.arange(MAX_SEQ, device=seq.device, dtype=torch.float32)
        pos = pos.unsqueeze(0).expand(B, -1) / MAX_SEQ
        decl_pos = (decl_step - 1).unsqueeze(1).float()  # 报听牌在序列中的 index
        arange = torch.arange(MAX_SEQ, device=seq.device).unsqueeze(0).float()
        post_decl = ((arange >= decl_pos + 1) & (decl_pos >= 0)).float()
        is_decl_tile = (arange == decl_pos).float()
        step = torch.stack([pos, post_decl, is_decl_tile, is_pad.float()], dim=-1)
        x = torch.cat([e, step], dim=-1)

        lengths = seq_len.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        vec = h[-1]  # (B, hidden)
        return torch.cat([vec, self.meld_proj(meld_hot)], dim=-1)


class SeqOppModel(nn.Module):
    def __init__(self, use_seq=True, feat_dim=175, opp_vec=80, hidden=256):
        super().__init__()
        self.use_seq = use_seq
        if use_seq:
            self.encoder = OppSeqEncoder()
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim + 3 * opp_vec, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.tenpai_head = nn.Linear(hidden, 3)
        self.wait_head = nn.Linear(hidden, 3 * N_TILES)

    def forward(self, feats, opp_seq, opp_seq_len, opp_decl_step, opp_meld_hot):
        B = feats.shape[0]
        if self.use_seq:
            vecs = []
            for r in range(3):
                vecs.append(self.encoder(
                    opp_seq[:, r], opp_seq_len[:, r],
                    opp_decl_step[:, r], opp_meld_hot[:, r]))
            opp = torch.cat(vecs, dim=-1)
        else:
            opp = torch.zeros(B, 3 * 80, device=feats.device)
        h = self.trunk(torch.cat([feats, opp], dim=-1))
        return self.tenpai_head(h), self.wait_head(h).view(B, 3, N_TILES)


def _auc(scores, labels):
    """Mann-Whitney AUC；labels 全 0 或全 1 时返回 nan。"""
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _recall_at_k(probs, true_sets, k):
    """每行 true wait 集合与 top-k 的命中比例（任一命中即中）。"""
    if len(probs) == 0:
        return float('nan')
    topk = np.argsort(-probs, axis=1)[:, :k]
    hit = np.array([true_sets[i, topk[i]].any() for i in range(len(probs))])
    return hit.mean() if len(hit) else float('nan')


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp_p, tp_y, wt_p, wt_y, decls = [], [], [], [], []
    for batch in loader:
        feats, seq, seq_len, decl_step, meld, tenpai, wait, decl = \
            [b.to(device) for b in batch]
        tl, wl = model(feats, seq, seq_len, decl_step, meld)
        tp_p.append(torch.sigmoid(tl).cpu().numpy())
        wt_p.append(torch.sigmoid(wl).cpu().numpy())
        tp_y.append(tenpai.cpu().numpy())
        wt_y.append(wait.cpu().numpy())
        decls.append(decl.cpu().numpy())
    p = np.concatenate(tp_p)       # (N, 3)
    y = np.concatenate(tp_y)
    wp = np.concatenate(wt_p)      # (N, 3, 34)
    wy = np.concatenate(wt_y)
    dc = np.concatenate(decls)     # (N, 3)，1=已报听, -1/0=未报听

    out = {}
    for subset, mask in (('all', np.ones_like(y, dtype=bool)),
                         ('silent', dc < 0.5)):
        aucs = [_auc(p[mask[:, r], r], y[mask[:, r], r]) for r in range(3)]
        pos_rows = mask & (y > 0.5)
        probs = np.concatenate([wp[pos_rows[:, r], r] for r in range(3)])
        trues = np.concatenate([wy[pos_rows[:, r], r] for r in range(3)])
        out[subset] = {
            'auc': aucs,
            'recall': {k: _recall_at_k(probs, trues, k) for k in (1, 3, 5)},
            'frac': mask.mean(),
        }
    return out, y.mean()


def _fmt(metrics):
    s = []
    for subset in ('all', 'silent'):
        m = metrics[subset]
        auc_s = '/'.join(f'{a:.3f}' for a in m['auc'])
        r = m['recall']
        s.append(f'{subset}: AUC=[{auc_s}] '
                 f'r@1/3/5={r[1]:.3f}/{r[3]:.3f}/{r[5]:.3f}')
    return '  '.join(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('out')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-seq', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(args.data)
    n = d['feats'].shape[0]
    idx = np.random.permutation(n)
    n_val = max(1, n // 10)
    va, tr = idx[:n_val], idx[n_val:]
    silent_frac = (d['opp_decl'] < 0.5).mean()
    print(f'{n} samples -> train {len(tr)} val {len(va)}; '
          f'per-opp tenpai rate {d["opp_tenpai"].mean():.3f}, '
          f'silent frac {silent_frac:.3f}')

    def make_ds(ix):
        return torch.utils.data.TensorDataset(
            torch.from_numpy(d['feats'][ix]),
            torch.from_numpy(d['opp_seq'][ix]),
            torch.from_numpy(d['opp_seq_len'][ix]),
            torch.from_numpy(d['opp_decl_step'][ix]),
            torch.from_numpy(d['opp_meld_hot'][ix]),
            torch.from_numpy(d['opp_tenpai'][ix]),
            torch.from_numpy(d['opp_wait'][ix]),
            torch.from_numpy(d['opp_decl'][ix]),
        )

    tr_loader = torch.utils.data.DataLoader(
        make_ds(tr), batch_size=args.batch, shuffle=True, drop_last=True)
    va_loader = torch.utils.data.DataLoader(
        make_ds(va), batch_size=4096, shuffle=False)

    model = SeqOppModel(use_seq=not args.no_seq).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_rate = d['opp_tenpai'][tr].mean()
    pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-6)]).to(args.device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    wait_bce = nn.BCEWithLogitsLoss(reduction='none')

    best_val = float('inf')
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss = nb = 0
        for batch in tr_loader:
            feats, seq, seq_len, decl_step, meld, tenpai, wait, _decl = \
                [b.to(args.device) for b in batch]
            tl, wl = model(feats, seq, seq_len, decl_step, meld)
            loss = bce(tl, tenpai)
            wl_elem = wait_bce(wl, wait).mean(dim=2)      # (B, 3)
            wmask = tenpai > 0.5
            if wmask.any():
                loss = loss + wl_elem[wmask].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            nb += 1

        metrics, _ = evaluate(model, va_loader, args.device)
        auc_silent = np.nanmean(metrics['silent']['auc'])
        val_score = -auc_silent if not np.isnan(auc_silent) \
            else -np.nanmean(metrics['all']['auc'])
        marker = ''
        if val_score < best_val:
            best_val = val_score
            torch.save(model.state_dict(), args.out)
            marker = '  saved best'
        print(f'Epoch {ep}/{args.epochs}: loss={tot_loss/nb:.4f} '
              f'{_fmt(metrics)} ({time.time()-t0:.0f}s){marker}', flush=True)

    cfg = {'arch': 'seq_opp', 'use_seq': not args.no_seq,
           'feat_dim': 175, 'max_seq': MAX_SEQ}
    with open(args.out.replace('.pt', '_config.json'), 'w') as f:
        json.dump(cfg, f)
    print(f'Done. Best val AUC(silent mean) {-best_val:.3f}. Model saved to {args.out}')


if __name__ == '__main__':
    main()
