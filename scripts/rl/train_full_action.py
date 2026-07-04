# -*- coding: utf-8 -*-
"""训练完整动作空间 NN（弃牌 + 碰/杠/胡响应）。

用法：
    PYTHONPATH=. python3 scripts/rl/train_full_action.py \
        output/nn_full_action_data_1000.npz \
        output/nn_full_action_init.pt \
        output/nn_full_action_1000.pt \
        --epochs 60 --batch 256 --lr 0.001
"""

import os
import sys
import json
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from algo.nn.model import TileConvNet


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg_path = path.replace('.pt', '_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    elif isinstance(checkpoint, dict) and 'config' in checkpoint:
        cfg = checkpoint['config']
    else:
        raise ValueError(f'No config found for {path}')

    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state = checkpoint['model_state']
    else:
        state = checkpoint
    allowed = {'input_dim', 'channels', 'n_blocks', 'hidden_dim', 'n_tile_ch', 'tile_len',
               'dealin_head', 'tenpai_head', 'candidate_value_head', 'response_head'}
    ctor_cfg = {k: v for k, v in cfg.items() if k in allowed}
    model = TileConvNet(**ctor_cfg)
    model.load_state_dict(state)
    model.to(device)
    return model, cfg


def split_outputs(model, cfg, out):
    """根据配置解析 forward 返回元组。"""
    idx = 0
    d_logit = out[idx]; idx += 1
    val = out[idx]; idx += 1
    dealin_logit = None
    if cfg.get('dealin_head', False):
        dealin_logit = out[idx]; idx += 1
    cv_logit = None
    if cfg.get('candidate_value_head', False):
        cv_logit = out[idx]; idx += 1
    response_logit = None
    if cfg.get('response_head', False):
        response_logit = out[idx]; idx += 1
    return d_logit, val, dealin_logit, cv_logit, response_logit


def base_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def masked_response_loss(logits, actions, legal_mask):
    """只让 legal action 参与 softmax 的 CE。"""
    logits = logits.clone()
    logits[legal_mask == 0] = -1e9
    return F.cross_entropy(logits, actions)


def evaluate(model, cfg, loader_disc, loader_resp, device):
    model.eval()
    total_disc, corr_disc = 0, 0
    total_resp, corr_resp = 0, 0
    sum_v_loss = 0.0
    sum_t_loss = 0.0
    n_v = 0
    n_t = 0

    with torch.no_grad():
        for xb, yb, vb, tb in loader_disc:
            xb, yb, vb, tb = xb.to(device), yb.to(device), vb.to(device), tb.to(device)
            out = model(xb)
            d_logit, val, dealin_logit, cv_logit, r_logit = split_outputs(model, cfg, out)
            pred = d_logit.argmax(dim=1)
            corr_disc += (pred == yb).sum().item()
            total_disc += yb.size(0)
            sum_v_loss += F.mse_loss(val.squeeze(1), vb).item() * vb.size(0)
            n_v += vb.size(0)
            if cfg.get('tenpai_head', False):
                t_logit = base_model(model).tenpai_logit(xb)
                sum_t_loss += F.binary_cross_entropy_with_logits(t_logit.squeeze(1), tb).item() * tb.size(0)
                n_t += tb.size(0)

        for xb, yb, lb, vb in loader_resp:
            xb, yb, lb, vb = xb.to(device), yb.to(device), lb.to(device), vb.to(device)
            out = model(xb)
            d_logit, val, dealin_logit, cv_logit, r_logit = split_outputs(model, cfg, out)
            if r_logit is not None:
                loss = masked_response_loss(r_logit, yb, lb)
                # 仅统计 legal actions 中的 top1 准确率
                logits = r_logit.clone()
                logits[lb == 0] = -1e9
                pred = logits.argmax(dim=1)
                mask_ok = lb.gather(1, yb.unsqueeze(1)).squeeze(1).bool()
                corr_resp += ((pred == yb) & mask_ok).sum().item()
                total_resp += mask_ok.sum().item()

    disc_acc = corr_disc / total_disc if total_disc else 0.0
    resp_acc = corr_resp / total_resp if total_resp else 0.0
    v_mse = sum_v_loss / n_v if n_v else 0.0
    t_bce = sum_t_loss / n_t if n_t else 0.0
    return disc_acc, resp_acc, v_mse, t_bce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('init_model')
    ap.add_argument('out_model')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--resp_weight', type=float, default=1.0)
    ap.add_argument('--val_ratio', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dp', type=int, default=1,
                    help='use nn.DataParallel if multiple GPUs available (1=auto, 0=force single)')
    ap.add_argument('--num_workers', type=int, default=4,
                    help='DataLoader CPU workers for training loaders')
    ap.add_argument('--resume', type=str, default='',
                    help='resume from a per-epoch checkpoint (.pt)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_dp = args.dp and torch.cuda.device_count() > 1
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f'Loading data from {args.data}')
    data = np.load(args.data)
    Xd = torch.from_numpy(data['X_discard']).float()
    yd = torch.from_numpy(data['y_discard']).long()
    vd = torch.from_numpy(data['v_discard']).float()
    td = torch.from_numpy(data['tenpai_discard']).float()

    Xr = torch.from_numpy(data['X_response']).float()
    yr = torch.from_numpy(data['y_response']).long()
    lr = torch.from_numpy(data['legal_response']).float()
    vr = torch.from_numpy(data['v_response']).float()

    n_disc = len(Xd)
    n_resp = len(Xr)
    n_val_disc = max(1, int(n_disc * args.val_ratio))
    n_val_resp = max(1, int(n_resp * args.val_ratio))

    idxd = np.random.permutation(n_disc)
    idxr = np.random.permutation(n_resp)

    tr_id = idxd[n_val_disc:]
    va_id = idxd[:n_val_disc]
    tr_ir = idxr[n_val_resp:]
    va_ir = idxr[:n_val_resp]

    train_disc = TensorDataset(Xd[tr_id], yd[tr_id], vd[tr_id], td[tr_id])
    val_disc = TensorDataset(Xd[va_id], yd[va_id], vd[va_id], td[va_id])
    train_resp = TensorDataset(Xr[tr_ir], yr[tr_ir], lr[tr_ir], vr[tr_ir])
    val_resp = TensorDataset(Xr[va_ir], yr[va_ir], lr[va_ir], vr[va_ir])

    loader_disc = DataLoader(train_disc, batch_size=args.batch, shuffle=True,
                             drop_last=True, num_workers=args.num_workers,
                             pin_memory=True if device.type == 'cuda' else False)
    loader_resp = DataLoader(train_resp, batch_size=args.batch, shuffle=True,
                             drop_last=True, num_workers=args.num_workers,
                             pin_memory=True if device.type == 'cuda' else False)
    val_loader_disc = DataLoader(val_disc, batch_size=args.batch, shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True if device.type == 'cuda' else False)
    val_loader_resp = DataLoader(val_resp, batch_size=args.batch, shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True if device.type == 'cuda' else False)

    print(f'Loading init model {args.init_model}')
    model, cfg = load_model(args.init_model, device)
    print('Model cfg:', cfg)

    if use_dp:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        print(f'Using DataParallel on {torch.cuda.device_count()} GPUs')

    if not cfg.get('response_head', False):
        raise ValueError('init model must have response_head=True')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_disc_acc = 0.0
    best_state = None
    best_cfg = cfg.copy()

    if args.resume:
        print(f'Resuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        base_model(model).load_state_dict(ckpt['model_state'])
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_disc_acc = ckpt.get('best_disc_acc', 0.0)
        best_state = ckpt.get('best_state', None)
        if best_state is not None:
            best_state = {k: v.to(device).clone() for k, v in best_state.items()}
        if 'config' in ckpt:
            cfg = ckpt['config']
            best_cfg = cfg.copy()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - start_epoch + 1))

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_disc_loss = 0.0
        epoch_resp_loss = 0.0
        epoch_v_loss = 0.0
        epoch_t_loss = 0.0
        n_batches = 0

        # 双迭代器交替
        it_disc = iter(loader_disc)
        it_resp = iter(loader_resp)
        while True:
            try:
                xb, yb, vb, tb = next(it_disc)
            except StopIteration:
                break
            try:
                xbr, ybr, lbr, vbr = next(it_resp)
            except StopIteration:
                it_resp = iter(loader_resp)
                xbr, ybr, lbr, vbr = next(it_resp)

            xb, yb, vb, tb = xb.to(device), yb.to(device), vb.to(device), tb.to(device)
            xbr, ybr, lbr, vbr = xbr.to(device), ybr.to(device), lbr.to(device), vbr.to(device)

            optimizer.zero_grad()

            # discard branch
            out_d = model(xb)
            d_logit, val_d, dealin_logit, cv_logit, _ = split_outputs(model, cfg, out_d)
            loss_disc = F.cross_entropy(d_logit, yb)
            loss_v = F.mse_loss(val_d.squeeze(1), vb)
            loss_t = 0.0
            if cfg.get('tenpai_head', False):
                t_logit = base_model(model).tenpai_logit(xb)
                loss_t = F.binary_cross_entropy_with_logits(t_logit.squeeze(1), tb)

            # response branch
            out_r = model(xbr)
            _, val_r, _, _, r_logit = split_outputs(model, cfg, out_r)
            loss_resp = masked_response_loss(r_logit, ybr, lbr)
            loss_v += F.mse_loss(val_r.squeeze(1), vbr)

            loss = loss_disc + loss_resp * args.resp_weight + loss_v + loss_t * 0.5
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_disc_loss += loss_disc.item()
            epoch_resp_loss += loss_resp.item()
            epoch_v_loss += loss_v.item()
            epoch_t_loss += loss_t.item() if isinstance(loss_t, torch.Tensor) else 0.0
            n_batches += 1

        scheduler.step()

        disc_acc, resp_acc, v_mse, t_bce = evaluate(model, cfg, val_loader_disc, val_loader_resp, device)
        print(f'Epoch {epoch:3d} | disc_loss {epoch_disc_loss/n_batches:.4f} '
              f'resp_loss {epoch_resp_loss/n_batches:.4f} v_loss {epoch_v_loss/n_batches:.4f} '
              f't_loss {epoch_t_loss/n_batches:.4f} | '
              f'val disc_acc {disc_acc:.4f} resp_acc {resp_acc:.4f} v_mse {v_mse:.4f} t_bce {t_bce:.4f}')

        if disc_acc > best_disc_acc:
            best_disc_acc = disc_acc
            best_state = {k: v.to(device).clone() for k, v in base_model(model).state_dict().items()}
            best_cfg = cfg.copy()

        # 每 epoch 保存 checkpoint，方便中断/恢复
        epoch_path = args.out_model.replace('.pt', f'_epoch_{epoch:02d}.pt')
        torch.save({
            'model_state': base_model(model).state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'best_disc_acc': best_disc_acc,
            'best_state': best_state,
            'config': cfg,
        }, epoch_path)
        with open(epoch_path.replace('.pt', '_config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)

    dt = time.time() - t0
    print(f'Training finished in {dt:.1f}s, best val disc_acc {best_disc_acc:.4f}')

    if best_state is not None:
        base_model(model).load_state_dict(best_state)

    out_cfg_path = args.out_model.replace('.pt', '_config.json')
    with open(out_cfg_path, 'w') as f:
        json.dump(best_cfg, f, indent=2)

    torch.save({'model_state': base_model(model).state_dict(), 'config': best_cfg}, args.out_model)
    print(f'Saved {args.out_model} + {out_cfg_path}')


if __name__ == '__main__':
    main()
