
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_stream.py
- Streaming training over packed segments (per grid) with multi-head supervision.
- GroupKFold by grid id, type-balanced.
"""
from __future__ import annotations
import argparse, os, json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.packed_stream import PackedStreamingDataset, packed_collate
from utils.splits import load_grid_infos, make_group_kfold

from models.mamba_regressor import MambaRegressor

def epe(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((a-b)**2).sum(dim=-1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", required=True, help="preprocessed out_dir/all/*.npz")
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--input_dim", type=int, required=True)
    ap.add_argument("--proj_dim", type=int, default=64)
    ap.add_argument("--d_model",  type=int, default=128)
    ap.add_argument("--n_layer",  type=int, default=4)
    ap.add_argument("--patch_len", type=int, default=8)
    ap.add_argument("--stride",    type=int, default=4)

    # heads & loss weights
    ap.add_argument("--out_vel", action="store_true")
    ap.add_argument("--out_acc", action="store_true")
    ap.add_argument("--w_pos",  type=float, default=1.0)
    ap.add_argument("--w_next", type=float, default=0.4)
    ap.add_argument("--w_vel",  type=float, default=0.2)
    ap.add_argument("--w_acc",  type=float, default=0.1)

    # split
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2025)

    # train
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4, help="batches of grids")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out_dir", type=str, default="./ckpt_stream")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    infos = load_grid_infos(args.features_root)
    if len(infos) < args.kfold: raise SystemExit("Not enough grids for requested kfold")
    folds = make_group_kfold(infos, k=args.kfold, seed=args.seed)
    tr_grids, va_grids = folds[args.fold]


    all_dir = str(Path(args.features_root)/"all")
    # 1) 优先使用“当前 fold 的训练统计”
    stats_fold = Path(args.features_root) / f"stats_train_fold{args.fold}.npz"
    # 2) 其次使用通用别名（如果你手动复制/链接过）
    stats_train = Path(args.features_root) / "stats_train.npz"
    # 3) 最后兜底：全局统计
    stats_global = Path(args.features_root) / "stats_global.npz"

    def _compute_stats_from_grids(grids):
        import numpy as np, json
        root = Path(args.features_root) / "all"
        sumv = None; sumsq = None; cnt = 0; Din_local = None
        for g in grids:
            p = root / f"{g}.npz"
            if not p.exists():
                cand = list(root.glob(f"{g}*.npz"))
                if not cand: 
                    print(f"[WARN] stats: missing {g}.npz"); 
                    continue
                p = cand[0]
            d = np.load(p, allow_pickle=True)
            X = d["feats"].astype(np.float64)
            if Din_local is None: Din_local = X.shape[1]
            sumv  = X.sum(0) if sumv is None else sumv + X.sum(0)
            sumsq = (X**2).sum(0) if sumsq is None else sumsq + (X**2).sum(0)
            cnt  += X.shape[0]
        if cnt == 0:
            raise SystemExit("[STATS] No frames to compute train-only stats for this fold.")
        mean = (sumv / cnt).astype(np.float32)
        var  = (sumsq / cnt) - (mean.astype(np.float64)**2)
        std  = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
        std  = np.maximum(std, 1e-3)
        # 临时写一份，方便复用
        tmp = Path(args.features_root) / f"_stats_train_fold{args.fold}.npz"
        np.savez(tmp, feat_mean=mean, feat_std=std, count=int(cnt), Din=int(Din_local), std_floor=1e-3) # type: ignore
        print(f"[STATS] computed on-the-fly train-only stats for fold {args.fold} -> {tmp}")
        return str(tmp)

    # 依优先级选择统计文件；都没有时，现场计算
    if stats_fold.exists():
        stats_path = str(stats_fold)
    elif stats_train.exists():
        stats_path = str(stats_train)
    elif stats_global.exists():
        stats_path = str(stats_global)
    else:
        stats_path = _compute_stats_from_grids(tr_grids)

    stats_train = str(Path(args.features_root)/"stats_train.npz")
    tr_ds = PackedStreamingDataset(all_dir, seq_len=args.seq_len, use_vel=args.out_vel, use_acc=args.out_acc,
                                   grids=tr_grids, stats_path=stats_path)
    va_ds = PackedStreamingDataset(all_dir, seq_len=args.seq_len, use_vel=args.out_vel, use_acc=args.out_acc,
                                   grids=va_grids, stats_path=stats_path)
    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=packed_collate)
    va = DataLoader(va_ds, batch_size=max(1,args.batch_size//2), shuffle=False, num_workers=2, collate_fn=packed_collate)

    model = MambaRegressor(Din=args.input_dim, K=args.seq_len, proj_dim=args.proj_dim,
                           d_model=args.d_model, n_layer=args.n_layer, patch_len=args.patch_len, stride=args.stride,
                           out_vel=args.out_vel, out_acc=args.out_acc).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.amp and hasattr(torch, "amp"): scaler = torch.amp.GradScaler("cuda")  # type: ignore
    else: scaler=None
    def autocast():
        if args.amp and hasattr(torch, "amp"):
            return torch.amp.autocast("cuda")  # type: ignore
        from contextlib import nullcontext; return nullcontext()

    def run(loader, train: bool):
        model.train() if train else model.eval()
        tot_loss=0.0; tot_epe=0.0; seen=0
        pbar = tqdm(loader, total=len(loader), ncols=120, desc="train" if train else "val")
        for x, pos, nxt, vel, acc, msk, grids, kinds in pbar:
            # x: (B,S,K,Din) pack of chunks per grid
            B,S,K,Din = x.shape
            x   = x.to(device); pos=pos.to(device); nxt=nxt.to(device)
            vel = vel.to(device); acc=acc.to(device); msk=msk.to(device)
            # streaming across S segments
            states=None; epe_sum=0.0; ncnt=0
            loss = torch.zeros((), device=device)
            for s in range(S):
                xb = x[:,s]  # (B,K,Din)
                with autocast():
                    out, states = model(xb, states=states, stream=True)
                    # targets for segment end
                    e_main = epe(out["pos"], pos[:,s]) * msk[:,s]
                    e_aux  = epe(out["next"], nxt[:,s]) * msk[:,s]
                    l = args.w_pos*F.smooth_l1_loss(out["pos"], pos[:,s], beta=1.0, reduction='none').mean(dim=-1)
                    l = l + args.w_next*F.smooth_l1_loss(out["next"], nxt[:,s], beta=1.0, reduction='none').mean(dim=-1)
                    if args.out_vel:
                        l = l + args.w_vel*F.smooth_l1_loss(out["vel"], vel[:,s], beta=0.5, reduction='none').mean(dim=-1)
                    if args.out_acc:
                        l = l + args.w_acc*F.smooth_l1_loss(out["acc"], acc[:,s], beta=0.25, reduction='none').mean(dim=-1)
                    loss_s = (l * msk[:,s]).sum() / (msk[:,s].sum() + 1e-6)
                epe_sum += e_main.sum().item(); ncnt += float(msk[:,s].sum().item())
                loss = loss + loss_s
            if train:
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot_loss += float(loss.item()); tot_epe += (epe_sum / max(1.0, ncnt)); seen += 1
            pbar.set_postfix(loss=f"{tot_loss/seen:.4f}", epe_m=f"{tot_epe/seen:.3f}")
        return tot_loss/max(1,seen), tot_epe/max(1,seen)

    best = float("inf")
    for ep in range(1, args.epochs+1):
        tr_loss, tr_epe = run(tr, True)
        va_loss, va_epe = run(va, False)
        print(f"Epoch {ep:03d} | train {tr_loss:.4f} ({tr_epe:.3f} m) | val {va_loss:.4f} ({va_epe:.3f} m)")
        # save best on val_epe
        if va_epe < best:
            best = va_epe
            state = {"model": model.state_dict(), "args": vars(args), "meta": {"fold": args.fold, "val_epe_m": float(va_epe)}}
            torch.save(state, os.path.join(args.out_dir, f"best_fold{args.fold}.pt"))
            print(f"[OK] saved best_fold{args.fold}.pt (val_epe={va_epe:.4f})")

if __name__ == "__main__":
    main()
