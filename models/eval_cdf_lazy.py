# Write a fixed, production-ready eval_cdf_lazy.py that:
# - uses absolute imports
# - deterministic split (prefer ckpt meta; else seed-based split)
# - keyword-only args for FramesLazyDataset
# - computes CDF/Hist, saves plots & CSV/NPZ
# - optional AMP for faster eval, safe fallback
# - supports predict=current|next
import os, textwrap, json, pathlib

# models/eval_cdf_lazy.py
import argparse, os, math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets.frames_lazy import FramesLazyDataset
from models.mamba_regressor import MambaRegressor


def get_autocast(enabled: bool):
    if not enabled:
        from contextlib import nullcontext
        return nullcontext
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return lambda: torch.amp.autocast("cuda") #type:ignore
    return lambda: torch.cuda.amp.autocast()


def euclid_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--seq_len", type=int, required=True)
    ap.add_argument("--input_dim", type=int, required=True)
    ap.add_argument("--proj_dim", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--patch_len", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--predict", choices=["current","next"], default="next")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_dir", type=str, default="./eval_out")
    ap.add_argument("--amp", action="store_true", help="enable autocast for eval")
    ap.add_argument("--save_csv", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # -------- Build dataset split: prefer ckpt meta --------
    ckpt = torch.load(args.ckpt, map_location="cpu")
    meta = ckpt.get("meta", {})
    if "val_files" in meta and meta["val_files"]:
        val_files = [Path(p) for p in meta["val_files"]]
    else:
        root = Path(args.features_root)
        files = sorted(root.glob("*.npz"))
        import random
        rnd = random.Random(args.seed)
        rnd.shuffle(files)
        cut = int(len(files) * 0.8)
        val_files = files[cut:] or files[:1]

    va_ds = FramesLazyDataset.from_filelist(val_files, seq_len=args.seq_len, predict=args.predict, mmap=True)
    va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=max(1,args.workers//2), pin_memory=True,
                    persistent_workers=(args.workers>0), prefetch_factor=2)

    # -------- Build model and load weights --------
    model = MambaRegressor(Din=args.input_dim, K=args.seq_len,
                           proj_dim=args.proj_dim, d_model=args.d_model,
                           n_layer=args.n_layer, patch_len=args.patch_len,
                           stride=args.stride).to(device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    # -------- Inference --------
    Autocast = get_autocast(args.amp)
    y_true, y_pred = [], []
    with torch.no_grad():
        pbar = tqdm(va, ncols=100, desc="eval")
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.squeeze(1).to(device, non_blocking=True)
            with Autocast():
                yhat = model(xb)
            y_true.append(yb.cpu().numpy())
            y_pred.append(yhat.cpu().numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    err = euclid_err(y_pred, y_true)

    # -------- Stats --------
    mean = float(err.mean())
    median = float(np.median(err))
    p80 = float(np.percentile(err, 80))
    p90 = float(np.percentile(err, 90))
    print(f"[STATS] N={len(err)}  mean={mean:.4f} m  median={median:.4f} m  P80={p80:.4f} m  P90={p90:.4f} m")

    # -------- Plots --------
    # CDF
    e = np.sort(err)
    y = np.arange(1, len(e)+1) / len(e)
    import matplotlib
    matplotlib.use("Agg")
    plt.figure(figsize=(5,4), dpi=160)
    plt.plot(e, y)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel("Position error (m)"); plt.ylabel("CDF"); plt.title("Error CDF")
    plt.tight_layout()
    cdf_path = os.path.join(args.out_dir, "cdf.png")
    plt.savefig(cdf_path); plt.close()

    # Histogram
    plt.figure(figsize=(5,4), dpi=160)
    plt.hist(err, bins=50)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel("Position error (m)"); plt.ylabel("Count"); plt.title("Error histogram")
    plt.tight_layout()
    hist_path = os.path.join(args.out_dir, "err_hist.png")
    plt.savefig(hist_path); plt.close()

    # -------- CSV / NPZ --------
    np.savez(
        os.path.join(args.out_dir, "val_preds.npz"),
        y_true=y_true,
        y_pred=y_pred,
        err=err,
        mean=np.float32(mean),
        median=np.float32(median),
        p80=np.float32(p80),
        p90=np.float32(p90),
)
    if args.save_csv:
        import csv
        csv_path = os.path.join(args.out_dir, "pred_vs_true.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["y_true_x","y_true_y","y_pred_x","y_pred_y","err_m"])
            for (yt, yp, e_) in zip(y_true, y_pred, err):
                w.writerow([yt[0], yt[1], yp[0], yp[1], e_])

    print(f"[OK] saved plots/stats under {args.out_dir}")
    print(f" - CDF: {cdf_path}")
    print(f" - Hist: {hist_path}")

if __name__ == "__main__":
    main()
