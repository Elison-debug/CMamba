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
def make_warmup_cosine(total_steps: int, target_lr: float, warmup_ratio: float = 0.05, min_lr: float = 1e-5):
    warmup_steps = max(200, min(1500, int(round(total_steps * warmup_ratio))))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return target_lr * (step + 1) / float(warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + 0.5 * (target_lr - min_lr) * (1.0 + math.cos(math.pi * prog))
    return lr_at

#constant lr
def make_constant(target_lr: float):
    def lr_at(step: int) -> float:
        return target_lr
    return lr_at

# cosine decay
def make_cosine(total_steps: int,
                target_lr: float,
                min_lr: float = 1e-6):
    def lr_at(step: int) -> float:
        prog = step / max(1, total_steps)
        return min_lr + 0.5 * (target_lr - min_lr) * (1.0 + math.cos(math.pi * prog))
    return lr_at
# constant then cosine decay
def make_const_then_cosine(total_steps: int,
                           target_lr: float,
                           switch_ratio: float = 0.06,  # 前6%固定
                           min_lr: float = 1e-6):
    switch_step = int(total_steps * switch_ratio)

    def lr_at(step: int) -> float:
        if step < switch_step:
            return target_lr
        prog = (step - switch_step) / max(1, total_steps - switch_step)
        return min_lr + 0.5 * (target_lr - min_lr) * (1.0 + math.cos(math.pi * prog))
    return lr_at



# ---------------------- Loss: Huber over Euclidean error ----------------------
def epe_huber_loss(yp: torch.Tensor, yb: torch.Tensor, beta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    欧氏距离误差的 Huber(SmoothL1)：
        epe = ||yp - yb||_2
        per = SmoothL1(epe, 0, beta)   # reduction='none'
    返回 per (bs,) 逐样本损失
    """
    epe = torch.sqrt(((yp - yb) ** 2).sum(dim=1))  # (bs,)
    # SmoothL1(epe, 0) = Huber
    per = F.smooth_l1_loss(epe, torch.zeros_like(epe), beta=beta, reduction='none')  # (bs,)
    return per, epe

_RUN_TARGETS = {}
def _lock_target(out_dir: Path, subdir: str, base: str, suffix: str,
                 patt: str, glob_pat: str, cache_key: str) -> Path:
        """
        锁定一次运行内的固定写入目标：
        - 扫描 out_dir/subdir 下已有同类文件，取最大编号 n
        - 本次运行固定写 base{n+1}{suffix}
        - 结果缓存到 _RUN_TARGETS[cache_key]
        """
        if cache_key in _RUN_TARGETS:
            return _RUN_TARGETS[cache_key]

        d = out_dir / subdir
        d.mkdir(parents=True, exist_ok=True)

        rx = re.compile(patt)
        maxn = 0
        for f in d.glob(glob_pat):
            m = rx.fullmatch(f.name)
            if m:
                n = int(m.group(1))
                if n > maxn:
                    maxn = n

        target = d / f"{base}{maxn+1}{suffix}"
        _RUN_TARGETS[cache_key] = target
        return target

def main():
    ap = argparse.ArgumentParser()
    # Data / split
    ap.add_argument("--features_root", type=str, default="./data/features/")
    ap.add_argument("--train_root", type=str, default=None)
    ap.add_argument("--val_root",   type=str, default=None)
    ap.add_argument("--seq_len", type=int, required=True)
    ap.add_argument("--input_dim", type=int, required=True)
    ap.add_argument("--predict", choices=["current", "next"], default="current")
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
    ap.add_argument("--tail_tau", type=float, default=0.5, help=">0 启用尾部加权的起点阈值(m)，建议0.5")
    ap.add_argument("--tail_alpha", type=float, default=3.0)
    ap.add_argument("--tail_gamma", type=float, default=0.2)

    # Misc
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_dir", type=str, default="./ckpt")
    #args = ap.parse_args()
    args, _ = ap.parse_known_args()
    # Repro
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- deterministic split by files ----------
    if args.train_root is None and args.val_root is None:
        if args.features_root is None:
            raise SystemExit("Provide --train_root/--val_root or --features_root (containing train/ and eval/)")
        args.train_root = str(Path(args.features_root) / "train")
        args.val_root   = str(Path(args.features_root) / "eval")

    tr_dir = Path(args.train_root); va_dir = Path(args.val_root)
    if not tr_dir.exists(): raise FileNotFoundError(f"train_root not found: {tr_dir}")
    if not va_dir.exists(): raise FileNotFoundError(f"val_root not found: {va_dir}")

    # 构建数据集：显式使用 train/eval 目录
    tr_files = sorted(tr_dir.glob("*.npz"))
    va_files = sorted(va_dir.glob("*.npz")) or sorted(tr_dir.glob("*.npz"))[:1]

    stats_root = tr_dir.parent  # 强制使用 train 统计
    tr_ds = FramesLazyDataset.from_filelist(tr_files, seq_len=args.seq_len, predict=args.predict, mmap=True, stats_root=stats_root)
    va_ds = FramesLazyDataset.from_filelist(va_files, seq_len=args.seq_len, predict=args.predict, mmap=True, stats_root=stats_root)

    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=(args.workers>0), prefetch_factor=2)
    va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=max(1, args.workers//2), pin_memory=True, drop_last=False, persistent_workers=(args.workers>0), prefetch_factor=2)

    print(f"[SPLIT] train files={len(tr_files)}  val files={len(va_files)}")
    print(f"[INFO]  train samples={len(tr_ds)} | val samples={len(va_ds)}")

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
    # lr_fn = make_warmup_cosine(total_steps, args.lr)
    # lr_fn = make_constant(args.lr)
    lr_fn = make_const_then_cosine(total_steps,args.lr)
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
                
                if ema is not None:
                    ema.update(model.parameters())
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

            # 打印日志
            grad_disp = f"{grad_norm:.3e}"
            pbar.set_postfix(
                avg_loss=f"{avg_loss:.4f}",
                mean_pos_err_m=f"{mean_epe:.3f}",
                lr=f"{cur_lr:.2e}",
                grad=grad_disp,
            )

        return (tot_loss / max(1, seen)), (sum_err / max(1, seen))
    
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "log" ; log_dir.mkdir(parents=True, exist_ok=True)
    result_dir = out_dir / "result";result_dir.mkdir(parents=True, exist_ok=True)
    # ------------- train / eval / save -------------
    # 运行期缓存：同一次运行内，同一个 base 只计算一次目标文件名
    def next_log_path(out_dir: Path) -> Path:
        p = _lock_target(out_dir, "log", "train_log", ".txt",
                        r"train_log(\d+)\.txt", "train_log*.txt",
                        cache_key="log:train_log")
        return p

    def next_ckpt_path(out_dir: Path) -> Path:
        p = _lock_target(out_dir, "result", "best", ".pt",
                        r"best(\d+)\.pt", "best*.pt",
                        cache_key="result:best.pt")
        return p

    def next_ckpt_epe_path(out_dir: Path) -> Path:
        p = _lock_target(out_dir, "result", "best", "_epe.pt",
                        r"best(\d+)_epe\.pt", "best*_epe.pt",
                        cache_key="result:best_epe.pt")
        return p
    log_path     = next_log_path(out_dir)
    best_path    = next_ckpt_path(out_dir)
    best_epe_path= next_ckpt_epe_path(out_dir)

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
        
        
        with open(log_path, "a", encoding="utf-8") as f:
            if ep == 1:  # 第一次运行，先写参数
                f.write("===== Training Args =====\n")
                for k, v in sorted(vars(args).items()):
                    f.write(f"{k}: {v}\n")
                f.write("=========================\n")

            # 写 epoch 结果
            msg = (f"Epoch {ep+1:03d} | "
                f"train_loss={tr_loss:.4f} ({tr_epe:.3f} m)| "
                f"val_loss={va_loss:.4f} ({va_epe:.3f} m)| "
                f"lr={opt.param_groups[0]['lr']:.2e}\n")
            f.write(msg)

        # 以 loss 选一份
        if va_loss < best_loss:
            best_loss = va_loss
            state = {"model": model.state_dict(),
                     "args": vars(args),
                     "meta": {"train_files": [str(p) for p in tr_files],
                              "val_files":   [str(p) for p in va_files],
                              "val_loss": float(va_loss),
                              "val_epe_m": float(va_epe)}}
            if ema is not None:
                with ema.average_parameters(): torch.save(state, best_path)
            else:
                torch.save(state, best_path)
            print(f"[OK] saved {best_path} (val_loss={best_loss:.4f})")

        # 以 EPE 选主 best
        if va_epe < best_epe:
            best_epe = va_epe
            state = {"model": model.state_dict(),
                     "args": vars(args),
                     "meta": {"train_files": [str(p) for p in tr_files],
                              "val_files":   [str(p) for p in va_files],
                              "val_loss": float(va_loss),
                              "val_epe_m": float(va_epe)}}
            if ema is not None:
                with ema.average_parameters(): torch.save(state, best_epe_path)
            else:
                torch.save(state, best_epe_path)
            print(f"[OK] saved {best_epe_path} (val_epe={best_epe:.4f} m)")


if __name__ == "__main__":
    main()
