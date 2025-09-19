#!/usr/bin/env python3
# models/train_regression_lazy.py
# Clean, robust, and metrics-aligned training script.

import argparse, math, re
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.frames_lazy import FramesLazyDataset
from models.mamba_regressor import MambaRegressor


# ---------------------- AMP utilities ----------------------
def get_amp(enabled: bool):
    use_amp = bool(enabled and torch.cuda.is_available())
    if not use_amp:
        from contextlib import nullcontext
        class DummyScaler:
            def scale(self, x): return x
            def unscale_(self, opt): pass
            def step(self, opt): opt.step()
            def update(self): pass
        return use_amp, DummyScaler(), (lambda: nullcontext())

    # 优先新 API
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda")  # type: ignore
        ac = lambda: torch.amp.autocast("cuda")  # type: ignore
        return True, scaler, ac
    # 旧 API 回退
    scaler = torch.cuda.amp.GradScaler()
    ac = lambda: torch.cuda.amp.autocast()
    return True, scaler, ac


# ---------------------- LR schedule ----------------------
def make_warmup_cosine(total_steps: int, target_lr: float, warmup_ratio: float = 0.05, min_lr: float = 1e-6):
    warmup_steps = max(1000, int(warmup_ratio * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return target_lr * (step + 1) / float(warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + 0.5 * (target_lr - min_lr) * (1.0 + math.cos(math.pi * prog))
    return lr_at


# ---------------------- Loss: Huber over Euclidean error ----------------------
def epe_huber_loss(yp: torch.Tensor, yb: torch.Tensor, beta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    欧氏距离误差的 Huber（SmoothL1）：
        epe = ||yp - yb||_2
        per = SmoothL1(epe, 0, beta)   # reduction='none'
    返回 per (bs,) 逐样本损失
    """
    epe = torch.sqrt(((yp - yb) ** 2).sum(dim=1))  # (bs,)
    # SmoothL1(epe, 0) = Huber
    per = F.smooth_l1_loss(epe, torch.zeros_like(epe), beta=beta, reduction='none')  # (bs,)
    return per, epe


def main():
    ap = argparse.ArgumentParser()
    # Data / split
    ap.add_argument("--features_root", type=str, required=True)
    ap.add_argument("--seq_len", type=int, required=True)
    ap.add_argument("--input_dim", type=int, required=True)    # Din per frame
    ap.add_argument("--predict", choices=["current", "next"], default="next")
    ap.add_argument("--workers", type=int, default=4)

    # Model
    ap.add_argument("--proj_dim", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--patch_len", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)

    # Train
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1.0, help="SmoothL1 (Huber) delta, meter")

    # Tail weighting (optional; default off)
    ap.add_argument("--tail_tau", type=float, default=0.0, help=">0 启用尾部加权的起点阈值(m)，建议0.5")
    ap.add_argument("--tail_alpha", type=float, default=3.0)
    ap.add_argument("--tail_gamma", type=float, default=0.2)

    # Misc
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_dir", type=str, default="./ckpt")
    args = ap.parse_args()

    # Repro
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- deterministic split by files ----------
    root = Path(args.features_root)
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz under {root}")

    import random
    rnd = random.Random(args.seed)
    rnd.shuffle(files)
    cut = int(len(files) * 0.8)
    train_files = files[:cut]
    val_files = files[cut:] or files[:1]

    tr_ds = FramesLazyDataset.from_filelist(train_files, seq_len=args.seq_len, predict=args.predict, mmap=True)
    va_ds = FramesLazyDataset.from_filelist(val_files,   seq_len=args.seq_len, predict=args.predict, mmap=True)

    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=(args.workers > 0), prefetch_factor=2)
    va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=max(1, args.workers // 2), pin_memory=True, drop_last=False,
                    persistent_workers=(args.workers > 0), prefetch_factor=2)

    print(f"[SPLIT] train files={len(train_files)}  val files={len(val_files)}")
    print(f"[INFO]  train samples={len(tr_ds)} (steps/epoch={len(tr)}) | "
          f"val samples={len(va_ds)} (steps/epoch={len(va)})")

    # ---------- model ----------
    model = MambaRegressor(
        Din=args.input_dim, K=args.seq_len,
        proj_dim=args.proj_dim, d_model=args.d_model,
        n_layer=args.n_layer, patch_len=args.patch_len, stride=args.stride
    ).to(device)

    # Optimizer: no wd on norm/bias
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if (n.endswith("bias") or "norm" in n.lower()) else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr
    )

    # EMA (single init)
    try:
        from torch_ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    except Exception:
        ema = None

    # LR schedule
    total_steps = max(1, args.epochs * len(tr))
    lr_fn = make_warmup_cosine(total_steps, args.lr)

    def set_lr(lr: float):
        for g in opt.param_groups: g["lr"] = lr

    class Stepper:
        def __init__(self): self.step_num = 0
        def step(self) -> float:
            self.step_num += 1
            cur = lr_fn(self.step_num); set_lr(cur)
            if ema is not None: ema.update()
            return cur
    sched = Stepper()

    # AMP
    use_amp, scaler, autocast = get_amp(args.amp)

    # ------------- core loop -------------
    def run_epoch(loader, train: bool):
        model.train() if train else model.eval()
        tot_loss, sum_err, seen = 0.0, 0.0, 0
        pbar = tqdm(loader, total=len(loader), ncols=130, desc="train" if train else "val")
        opt.zero_grad(set_to_none=True)
        accum = max(1, args.accum if train else 1)
        kept_ratio = 1.0

        for i, (xb, yb) in enumerate(pbar, 1):
            xb = xb.to(device, non_blocking=True)         # (B,K,Din)
            yb = yb.squeeze(1).to(device, non_blocking=True)  # (B,2)
            # 轻度防极值；标准化已在 Dataset 完成
            xb = xb.clamp_(-10, 10)

            with autocast():
                per, epe = epe_huber_loss(model(xb), yb, beta=args.beta)  # (B,), (B,)

                # 可选：尾部加权（默认关闭：tau<=0）
                if args.tail_tau > 0.0:
                    w = 1.0 + args.tail_alpha * torch.sigmoid((epe - args.tail_tau) / args.tail_gamma)
                    per = per * w

                loss = per.mean()

            # 数值安全：跳过 NaN/Inf
            if not torch.isfinite(loss).all():
                if train: opt.zero_grad(set_to_none=True)
                pbar.set_postfix_str("skip NaN batch")
                continue

            # 反向/step（只在累积边界）
            did_step = False
            if train:
                if use_amp: scaler.scale(loss).backward()
                else:       loss.backward()

                if (i % accum) == 0:
                    if use_amp:
                        scaler.unscale_(opt)
                        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                        scaler.step(opt); scaler.update()
                    else:
                        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                    cur_lr = sched.step()
                    did_step = True
                else:
                    grad_norm = 0.0
                    cur_lr = opt.param_groups[0]["lr"]
            else:
                grad_norm = 0.0
                cur_lr = opt.param_groups[0]["lr"]

            # 运行统计（严格样本口径）
            bs = xb.size(0)
            tot_loss += float(per.sum().item())
            sum_err  += float(epe.sum().item())
            seen     += bs

            avg_loss = tot_loss / max(1, seen)
            mean_epe = sum_err  / max(1, seen)

            # 打印更有信息量的日志
            grad_disp = f"{grad_norm:.3e}" if did_step else "-"
            pbar.set_postfix(
                avg_loss=f"{avg_loss:.4f}",
                mean_pos_err_m=f"{mean_epe:.3f}",
                lr=f"{cur_lr:.2e}",
                grad=grad_disp,
            )

        return (tot_loss / max(1, seen)), (sum_err / max(1, seen))

    # ------------- train / eval / save -------------
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def next_ckpt_path(base: str) -> str:
        p = out_dir / base
        if not p.exists(): return str(p)
        patt = re.compile(r"best(\d*)\.pt")
        maxn = 1
        for f in out_dir.glob("best*.pt"):
            m = patt.fullmatch(f.name)
            if m:
                n = int(m.group(1) or "1"); maxn = max(maxn, n)
        return str(out_dir / f"best{maxn+1}.pt")

    best_loss = float("inf")
    best_epe  = float("inf")

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_epe = run_epoch(tr, True)

        if ema is not None:
            from contextlib import nullcontext
            with ema.average_parameters():
                va_loss, va_epe = run_epoch(va, False)
        else:
            va_loss, va_epe = run_epoch(va, False)

        print(f"Epoch {ep:03d} | train {tr_loss:.4f} ({tr_epe:.3f} m) | "
              f"val {va_loss:.4f} ({va_epe:.3f} m) | lr {opt.param_groups[0]['lr']:.2e}")

        # 以 loss 选一份
        if va_loss < best_loss:
            best_loss = va_loss
            path = next_ckpt_path("best.pt")
            state = {"model": model.state_dict(),
                     "args": vars(args),
                     "meta": {"train_files": [str(p) for p in train_files],
                              "val_files":   [str(p) for p in val_files],
                              "val_loss": float(va_loss),
                              "val_epe_m": float(va_epe)}}
            if ema is not None:
                with ema.average_parameters(): torch.save(state, path)
            else:
                torch.save(state, path)
            print(f"[OK] saved best.pt (val_loss={best_loss:.4f})")

        # 以 EPE 选主 best
        if va_epe < best_epe:
            best_epe = va_epe
            path = next_ckpt_path("best_epe.pt")
            state = {"model": model.state_dict(),
                     "args": vars(args),
                     "meta": {"train_files": [str(p) for p in train_files],
                              "val_files":   [str(p) for p in val_files],
                              "val_loss": float(va_loss),
                              "val_epe_m": float(va_epe)}}
            if ema is not None:
                with ema.average_parameters(): torch.save(state, path)
            else:
                torch.save(state, path)
            print(f"[OK] saved best_epe.pt (val_epe={best_epe:.4f} m)")


if __name__ == "__main__":
    main()
