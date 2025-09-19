#!/usr/bin/env python3
# models/eval_cdf_lazy.py
import argparse, os, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets.frames_lazy import FramesLazyDataset
from models.mamba_regressor import MambaRegressor


def get_autocast(enabled: bool):
    if not enabled:
        from contextlib import nullcontext
        return lambda: nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return lambda: torch.amp.autocast("cuda") #type:ignore
    return lambda: torch.cuda.amp.autocast()


def euclid_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    # 会被 ckpt 覆盖（若存在），提供默认值即可
    ap.add_argument("--seq_len", type=int, default=12)
    ap.add_argument("--input_dim", type=int, default=2000)
    ap.add_argument("--proj_dim", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--patch_len", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--predict", choices=["current", "next"], default="next")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_dir", type=str, default="./eval_out")
    ap.add_argument("--amp", action="store_true", help="enable autocast for eval")
    ap.add_argument("--save_csv", action="store_true")

    # 实时统计参数
    ap.add_argument("--live_bins", type=int, default=100, help="在线直方图的分箱数")
    ap.add_argument("--live_max_err", type=float, default=10.0, help="直方图覆盖的最大误差（m）")
    ap.add_argument("--live_every", type=int, default=1, help="每 N 个 batch 刷新一次进度条统计")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # -------- 载入 ckpt，并用其中训练配置覆盖本地 args --------
    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = ckpt.get("args", {}) or {}
    for k in ["seq_len", "input_dim", "proj_dim", "d_model", "n_layer", "patch_len", "stride", "predict"]:
        if k in train_args:
            setattr(args, k, train_args[k])

    # -------- 构造验证集（优先使用 ckpt.meta.val_files）--------
    meta = ckpt.get("meta", {}) or {}
    if "val_files" in meta and meta["val_files"]:
        val_files = [Path(p) for p in meta["val_files"]]
    else:
        root = Path(args.features_root)
        files = sorted(root.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz under {root}")
        import random
        rnd = random.Random(args.seed)
        rnd.shuffle(files)
        cut = int(len(files) * 0.8)
        val_files = files[cut:] or files[:1]

    print(f"[CFG] seq_len={args.seq_len}, proj_dim={args.proj_dim}, d_model={args.d_model}, "
          f"n_layer={args.n_layer}, patch_len={args.patch_len}, stride={args.stride}, predict={args.predict}")
    print(f"[VAL] files={len(val_files)}")

    va_ds = FramesLazyDataset.from_filelist(val_files, seq_len=args.seq_len, predict=args.predict, mmap=True)
    va = DataLoader(
        va_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.workers // 2), pin_memory=True,
        persistent_workers=(args.workers > 0), prefetch_factor=2
    )

    # -------- 模型 --------
    model = MambaRegressor(
        Din=args.input_dim, K=args.seq_len,
        proj_dim=args.proj_dim, d_model=args.d_model,
        n_layer=args.n_layer, patch_len=args.patch_len, stride=args.stride
    ).to(device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    # -------- 推理：实时统计 --------
    Autocast = get_autocast(args.amp)

    nbins = int(args.live_bins)
    e_max = float(args.live_max_err)
    bins = np.linspace(0.0, e_max, nbins + 1)
    counts = np.zeros(nbins, dtype=np.int64)

    def _update_hist(err_batch: np.ndarray):
        h, _ = np.histogram(np.clip(err_batch, 0, e_max - 1e-9), bins=bins)
        counts[:] += h

    def _approx_percentile(p: float) -> float:
        target = p / 100.0 * counts.sum()
        if target <= 0:
            return 0.0
        cumsum = np.cumsum(counts)
        idx = np.searchsorted(cumsum, target)
        if idx <= 0:
            return float(bins[1])
        if idx >= nbins:
            return float(bins[-1])
        left_c = cumsum[idx - 1]
        right_c = cumsum[idx]
        frac = 0.0 if right_c == left_c else (target - left_c) / (right_c - left_c)
        return float(bins[idx] + frac * (bins[idx + 1] - bins[idx]))

    run_mean, run_n, run_max = 0.0, 0, 0.0
    ema_batch, ema_alpha = None, 0.2

    y_true, y_pred = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        pbar = tqdm(enumerate(va, 1), total=len(va), ncols=130, desc="eval")
        for i, (xb, yb) in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.squeeze(1).to(device, non_blocking=True)
            with Autocast():
                yhat = model(xb)

            yt_np = yb.cpu().numpy()
            yp_np = yhat.cpu().numpy()
            y_true.append(yt_np)
            y_pred.append(yp_np)

            batch_err = np.sqrt(((yp_np - yt_np) ** 2).sum(axis=1))
            bcnt = int(batch_err.size)
            if bcnt > 0:
                bsum = float(batch_err.sum())
                run_n += bcnt
                run_mean += (bsum - bcnt * run_mean) / run_n
                run_max = max(run_max, float(np.max(batch_err)))
                _update_hist(batch_err)
                bmean = float(batch_err.mean())
                ema_batch = bmean if (ema_batch is None) else (ema_alpha * bmean + (1 - ema_alpha) * ema_batch)

            if i % max(1, int(args.live_every)) == 0:
                p50 = _approx_percentile(50.0) if run_n > 0 else 0.0
                p80 = _approx_percentile(80.0) if run_n > 0 else 0.0
                elapsed = max(1e-6, time.perf_counter() - t0)
                ips = run_n / elapsed
                pbar.set_postfix(
                    batch=f"{(ema_batch or 0.0):.3f}",
                    mean=f"{run_mean:.3f}",
                    p50=f"{p50:.3f}",
                    p80=f"{p80:.3f}",
                    max=f"{run_max:.2f}",
                    ips=f"{ips:.1f}",
                )

    # -------- 汇总 & 保存 --------
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    err = euclid_err(y_pred, y_true)

    mean = float(err.mean())
    median = float(np.median(err))
    p80 = float(np.percentile(err, 80))
    p90 = float(np.percentile(err, 90))
    print(f"[STATS] N={len(err)}  mean={mean:.4f} m  median={median:.4f} m  P80={p80:.4f} m  P90={p90:.4f} m")

    # CDF
    e_sorted = np.sort(err)
    y = np.arange(1, len(e_sorted) + 1) / len(e_sorted)
    plt.figure(figsize=(5, 4), dpi=160)
    plt.plot(e_sorted, y)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel("Position error (m)"); plt.ylabel("CDF"); plt.title("Error CDF")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "cdf.png")); plt.close()

    # Histogram
    plt.figure(figsize=(5, 4), dpi=160)
    plt.hist(err, bins=50)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel("Position error (m)"); plt.ylabel("Count"); plt.title("Error histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "err_hist.png")); plt.close()

    # NPZ（标量单独字段，避免 dict 报错）
    np.savez(
        os.path.join(args.out_dir, "val_preds.npz"),
        y_true=y_true, y_pred=y_pred, err=err,
        mean=np.float32(mean), median=np.float32(median),
        p80=np.float32(p80), p90=np.float32(p90),
    )

    if args.save_csv:
        import csv
        with open(os.path.join(args.out_dir, "pred_vs_true.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["y_true_x", "y_true_y", "y_pred_x", "y_pred_y", "err_m"])
            for (yt, yp, e_) in zip(y_true, y_pred, err):
                w.writerow([yt[0], yt[1], yp[0], yp[1], e_])
                 # -------- Markdown summary --------
    md = []
    md.append("| Metric | Value (m) |")
    md.append("|--------|-----------|")
    md.append(f"| Mean   | {mean:.3f} |")
    md.append(f"| Median | {median:.3f} |")
    md.append(f"| P80    | {p80:.3f} |")
    md.append(f"| P90    | {p90:.3f} |")
    md.append(f"| Max    | {err.max():.3f} |")
    table = "\n".join(md)
    print("\n[SUMMARY]\n" + table + "\n")

    print(f"[OK] saved plots & stats under {args.out_dir}")


if __name__ == "__main__":
    main()
