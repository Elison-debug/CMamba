# Write a fixed, production-ready train_regression_lazy.py that:
# - uses absolute imports
# - builds a deterministic file split with pathlib + from_filelist()
# - prints dataset sizes, steps/epoch
# - tqdm progress with avg_loss, ips, mean_pos_err_m
# - AMP new API fallback-safe
# - gradient clipping + skip NaN batches
# - 'next' prediction by default
# - configurable num_workers and SmoothL1 beta
import os, textwrap, json, pathlib

#code = r'''#!/usr/bin/env python3
# models/train_regression_lazy.py
import argparse, os, math, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.frames_lazy import FramesLazyDataset
from models.mamba_regressor import MambaRegressor


def get_amp_scaler(enabled: bool):
    """Return (GradScaler, autocast_ctx_manager_factory)."""
    if not enabled:
        class Dummy:
            def scale(self, x): return x
            def step(self, opt): opt.step()
            def update(self): pass
            def unscale_(self, opt): pass
        def no_autocast():
            from contextlib import nullcontext
            return nullcontext()
        return Dummy(), no_autocast

    # Prefer new torch.amp API if available
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda") #type:ignore
        autocast_factory = lambda: torch.amp.autocast("cuda")#type:ignore
        return scaler, autocast_factory

    # Fallback to old API
    scaler = torch.cuda.amp.GradScaler()
    autocast_factory = lambda: torch.cuda.amp.autocast()
    return scaler, autocast_factory


def isnan(x: torch.Tensor) -> bool:
    return bool(torch.isnan(x).any().item()) or bool(torch.isinf(x).any().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", type=str, required=True)
    ap.add_argument("--seq_len", type=int, required=True)
    ap.add_argument("--input_dim", type=int, required=True)   # Din
    ap.add_argument("--proj_dim", type=int, default=64)       # C
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--patch_len", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--predict", choices=["current","next"], default="next")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--beta", type=float, default=0.5, help="SmoothL1 beta")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default="./ckpt")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- deterministic split (by files) ----------
    root = Path(args.features_root)
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz under {root}")
    import random
    rnd = random.Random(args.seed)
    rnd.shuffle(files)
    cut = int(len(files) * 0.8)
    train_files = files[:cut]
    val_files   = files[cut:] or files[:1]

    tr_ds = FramesLazyDataset.from_filelist(train_files, seq_len=args.seq_len, predict=args.predict, mmap=True)
    va_ds = FramesLazyDataset.from_filelist(val_files,   seq_len=args.seq_len, predict=args.predict, mmap=True)

    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True,
                    persistent_workers=(args.workers>0), prefetch_factor=2, drop_last=True)
    va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=max(1,args.workers//2), pin_memory=True,
                    persistent_workers=(args.workers>0), prefetch_factor=2)

    print(f"[SPLIT] train files={len(train_files)}  val files={len(val_files)}")
    print(f"[INFO] train samples={len(tr_ds)}  steps/epoch={len(tr)} | "
          f"val samples={len(va_ds)}  steps/epoch={len(va)}")

    # ---------- model ----------
    model = MambaRegressor(Din=args.input_dim, K=args.seq_len,
                           proj_dim=args.proj_dim, d_model=args.d_model,
                           n_layer=args.n_layer, patch_len=args.patch_len,
                           stride=args.stride).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = nn.SmoothL1Loss(beta=args.beta)
    scaler, autocast_factory = get_amp_scaler(enabled=args.amp)

    # Cosine schedule with floor
    total_steps = max(1, args.epochs * len(tr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        tot_loss, seen = 0.0, 0
        sum_err = 0.0
        pbar = tqdm(loader, ncols=110, desc="train" if train else "val")
        for i, (xb, yb) in enumerate(pbar, 1):
            xb = xb.to(device, non_blocking=True)       # (B,K,Din)
            yb = yb.squeeze(1).to(device, non_blocking=True)  # (B,2)

            # clamp inputs a bit to avoid rare outliers
            xb = xb.clamp_(-10, 10)

            with autocast_factory():
                yhat = model(xb)                         # (B,2)
                loss = crit(yhat, yb)

            if isnan(yhat) or isnan(loss):
                pbar.set_postfix_str("skip NaN batch")
                continue

            bs = xb.size(0)
            if train:
                if args.amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()

            tot_loss += loss.item() * bs
            seen += bs
            with torch.no_grad():
                err = torch.sqrt(((yhat - yb) ** 2).sum(dim=1)).mean().item()
                sum_err += err * bs
            ips = seen / max(1e-6, pbar.format_dict.get("elapsed", 0.0))
            pbar.set_postfix(avg_loss=f"{tot_loss/max(1,seen):.4f}",
                             ips=f"{ips:.1f}",
                             mean_pos_err_m=f"{(sum_err/max(1,seen)):.3f}")

        return tot_loss / max(1, seen), (sum_err / max(1, seen))

    os.makedirs(args.out_dir, exist_ok=True)
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_err = run_epoch(tr, True)
        va_loss, va_err = run_epoch(va, False)
        print(f"Epoch {ep:03d} | train {tr_loss:.4f} ({tr_err:.3f} m) "
              f"| val {va_loss:.4f} ({va_err:.3f} m) | lr {opt.param_groups[0]['lr']:.2e}")
        if va_loss < best:
            best = va_loss
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "meta": {
                    "train_files": [str(p) for p in train_files],
                    "val_files": [str(p) for p in val_files],
                }
            }, os.path.join(args.out_dir, "best.pt"))
            print(f"[OK] saved best.pt  (val_loss={best:.4f})")

if __name__ == "__main__":
    main()

