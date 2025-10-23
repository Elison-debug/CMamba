#!/usr/bin/env python3
# models/eval/eval_multi.py
# Evaluate EACH *.npz individually, save plots per-file into its own folder,
# and finally write a mean-error ranking + detailed logs into eval_out.txt.

import argparse, os, time, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- project-local imports (same as the single-file version) ---
from datasets.frames_lazy import FramesLazyDataset
from models.eval.mamba_regressor import MambaRegressor


def get_autocast(enabled: bool):
    if not enabled:
        from contextlib import nullcontext
        return lambda: nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return lambda: torch.amp.autocast("cuda")  # type: ignore
    return lambda: torch.cuda.amp.autocast()


def euclid_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", type=str, default="./data/features")
    ap.add_argument("--ckpt", type=str, required=True)
    # Explicit eval root, otherwise read from ckpt.meta or fall back to features_root/eval
    ap.add_argument("--eval_root", type=str, default=None)

    # Model / data (will be overridden by ckpt args if present)
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
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--amp", action="store_true", help="enable autocast for eval")

    ap.add_argument("--out_dir", type=str, default="./eval_out")
    ap.add_argument("--save_csv", action="store_true", help="save pred_vs_true per file")

    # Online histogram params (per-file)
    ap.add_argument("--live_bins", type=int, default=100, help="histogram bins for live stats")
    ap.add_argument("--live_max_err", type=float, default=10.0, help="histogram max error (m)")
    ap.add_argument("--live_every", type=int, default=1, help="update tqdm every N batches")

    # Optional subset: evaluate only files whose name contains this substring (can repeat)
    ap.add_argument("--include", action="append", default=None, help="only eval files containing this substring (repeatable)")
    ap.add_argument("--exclude", action="append", default=None, help="skip files containing this substring (repeatable)")

    return ap.parse_args()


def should_keep(path: Path, includes, excludes) -> bool:
    name = path.name
    if includes:
        ok = any(s in name for s in includes)
        if not ok:
            return False
    if excludes:
        if any(s in name for s in excludes):
            return False
    return True


def make_dataloader(file_path: Path, stats_root: Path, args, train_args) -> DataLoader:
    ds = FramesLazyDataset.from_filelist(
        [file_path],
        seq_len=train_args.get("seq_len", args.seq_len),
        predict=train_args.get("predict", "current"),
        mmap=True,
        stats_root=stats_root,
    )
    va = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=4,
    )
    return va


def eval_one(npz_path: Path, model: torch.nn.Module, device: torch.device, stats_root: Path, args, train_args, log_write):
    Autocast = get_autocast(args.amp)

    # per-file output dir (folder named by npz file stem)
    out_dir = Path(args.out_dir) / npz_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # live histogram buffers (per file)
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

    va = make_dataloader(npz_path, stats_root, args, train_args)

    y_true, y_pred = [], []
    run_mean, run_n, run_max = 0.0, 0, 0.0
    ema_batch, ema_alpha = None, 0.2

    t0 = time.perf_counter()
    model.eval()

    with torch.inference_mode():
        pbar = tqdm(enumerate(va, 1), total=len(va), ncols=130, desc=f"eval[{npz_path.stem}]")
        for i, (xb, yb) in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.squeeze(1).to(device, non_blocking=True)
            with Autocast():
                yhat = model(xb)

            # GPU L2, then CPU only when needed
            batch_err_gpu = torch.linalg.vector_norm(yhat - yb, ord=2, dim=1)
            batch_err = batch_err_gpu.detach().cpu().numpy()

            # If we will plot/save later, keep a copy (CPU)
            y_true.append(yb.detach().cpu().numpy())
            y_pred.append(yhat.detach().cpu().numpy())

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

    # --- consolidate ---
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    err = euclid_err(y_pred, y_true)

    mean = float(err.mean())
    median = float(np.median(err))
    p80 = float(np.percentile(err, 80))
    p90 = float(np.percentile(err, 90))
    print(f"[STATS] N={len(err)}  mean={mean:.4f} m  median={median:.4f} m  P80={p80:.4f} m  P90={p90:.4f} m")

    # Log basic stats
    log_write(f"[FILE] {npz_path.name}\n")
    log_write(f"[STATS] N={len(err)}  mean={mean:.4f} m  median={median:.4f} m  P80={p80:.4f} m  P90={p90:.4f} m\n")

    # --- plots ---
    e_sorted = np.sort(err)
    y = np.arange(1, len(e_sorted) + 1) / len(e_sorted)

    # zoom-in CDF
    plt.figure(figsize=(5, 4), dpi=160)
    plt.plot(e_sorted, y)
    plt.xlim(0, 0.5)
    plt.xticks(np.arange(0, 0.51, 0.05))
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel("Position error (m)")
    plt.ylabel("CDF")
    plt.title("Error CDF (0-0.5 m zoom-in)")
    plt.tight_layout()
    plt.savefig(out_dir / "cdf_zoom.png")
    plt.close()

    # full CDF
    plt.figure(figsize=(5, 4), dpi=160)
    plt.plot(e_sorted, y)
    xticks = list(np.arange(0, 0.41, 0.10)) + list(np.arange(0, 2.1, 0.2))
    plt.xticks(xticks)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel("Position error (m)"); plt.ylabel("CDF"); plt.title("Error CDF")
    plt.tight_layout()
    plt.savefig(out_dir / "cdf.png")
    plt.close()

    # histogram
    plt.figure(figsize=(5, 4), dpi=160)
    plt.hist(err, bins=50)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel("Position error (m)"); plt.ylabel("Count"); plt.title("Error histogram")
    plt.tight_layout()
    plt.savefig(out_dir / "err_hist.png")
    plt.close()

    # Save arrays for this file
    np.savez(
        out_dir / "val_preds.npz",
        y_true=y_true, y_pred=y_pred, err=err,
        mean=np.float32(mean), median=np.float32(median),
        p80=np.float32(p80), p90=np.float32(p90),
    )

    if args.save_csv:
        import csv
        with open(out_dir / "pred_vs_true.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["y_true_x", "y_true_y", "y_pred_x", "y_pred_y", "err_m"])
            for (yt, yp, e_) in zip(y_true, y_pred, err):
                w.writerow([yt[0], yt[1], yp[0], yp[1], e_])

    # Markdown mini-table (also log)
    md = []
    md.append("| Metric | Value (m) |")
    md.append("|--------|-----------|")
    md.append(f"| Mean   | {mean:.4f} |")
    md.append(f"| Median | {median:.4f} |")
    md.append(f"| P80    | {p80:.4f} |")
    md.append(f"| P90    | {p90:.4f} |")
    md.append(f"| Max    | {err.max():.4f} |")
    table = "\n".join(md)
    log_write("[SUMMARY of " + npz_path.name + " ]\n" + table + "\n\n")

    return {
        "file": npz_path.name,
        "stem": npz_path.stem,
        "out_dir": str(out_dir),
        "N": int(len(err)),
        "mean": mean,
        "median": median,
        "p80": p80,
        "p90": p90,
        "max": float(err.max()),
    }


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # --- Load ckpt & training args override ---
    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = ckpt.get("args", {}) or {}

    for k in [
        "seq_len",
        "input_dim",
        "proj_dim",
        "d_model",
        "n_layer",
        "patch_len",
        "stride",
        "predict",
    ]:
        if k in train_args:
            setattr(args, k, train_args[k])

    # --- resolve eval file list ---
    meta = ckpt.get("meta", {}) or {}
    print(f"[info] args.eval_root is {args.eval_root}")
    if args.eval_root is None:
        if "val_files" in meta and meta["val_files"]:
            val_files = [Path(p) for p in meta["val_files"]]
            eval_root = Path(val_files[0]).parent if val_files else None
        else:
            root = Path(args.features_root) if args.features_root else None
            if root is None:
                raise SystemExit("Provide --eval_root or --features_root or let ckpt.meta provide val_files")
            eval_root = root / "eval"
            val_files = sorted(eval_root.glob("*.npz"))
    else:
        eval_root = Path(args.eval_root)
        print(f"[OK] eval root set to {eval_root}")
        val_files = sorted(eval_root.glob("*.npz"))
    if not val_files:
        raise FileNotFoundError(f"No .npz under {eval_root}")

    # filter if include/exclude requested
    val_files = [p for p in val_files if should_keep(p, args.include, args.exclude)]
    if not val_files:
        raise SystemExit("After include/exclude filtering, no files remain.")

    # --- dataset stats root: force to use train stats ---
    stats_root = Path(meta.get("train_root", eval_root.parent))  # type: ignore

    # --- build model once ---
    model = MambaRegressor(
        Din=args.input_dim,
        K=args.seq_len,
        proj_dim=args.proj_dim,
        d_model=args.d_model,
        sigma=args.sigma,
        n_layer=args.n_layer,
        patch_len=args.patch_len,
        stride=args.stride,
    ).to(device)
    
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    # --- open master log file ---
    log_path = Path(args.out_dir) / "eval_out.txt"
    f_log = open(log_path, "w", encoding="utf-8")

    def log_write(msg: str):
        f_log.write(msg)
        f_log.flush()
        # also mirror to stdout for real-time feedback
        sys.stdout.write(msg)
        sys.stdout.flush()

    # global header
    log_write(f"[CFG] seq_len={args.seq_len}, proj_dim={args.proj_dim}, d_model={args.d_model}, "
              f"n_layer={args.n_layer}, patch_len={args.patch_len}, stride={args.stride}, predict={args.predict}\n")
    log_write(f"[VAL] files={len(val_files)} (root={str(eval_root)})\n\n")

    # --- evaluate each file ---
    results = []
    for npz_path in val_files:
        res = eval_one(npz_path, model, device, stats_root, args, train_args, log_write)
        results.append(res)

    # --- ranking by mean error (ascending) ---
    results_sorted = sorted(results, key=lambda x: x["mean"])  # best first

    log_write("==================== RANKING (by mean error) ====================\n")
    for rank, r in enumerate(results_sorted, 1):
        log_write(f"{rank:2d}. mean={r['mean']:.4f} m | median={r['median']:.4f} | P80={r['p80']:.4f} | P90={r['p90']:.4f} | N={r['N']} | file={r['file']} | out={r['out_dir']}\n")
    log_write("=================================================================\n\n")

    # also save a small CSV of the ranking
    import csv
    with open(Path(args.out_dir) / "ranking.csv", "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["rank", "file", "mean", "median", "p80", "p90", "N", "out_dir"])
        for rank, r in enumerate(results_sorted, 1):
            w.writerow([rank, r["file"], f"{r['mean']:.6f}", f"{r['median']:.6f}", f"{r['p80']:.6f}", f"{r['p90']:.6f}", r["N"], r["out_dir"]])

    f_log.close()
    print(f"[OK] Wrote consolidated log to {log_path}")


if __name__ == "__main__":
    main()
