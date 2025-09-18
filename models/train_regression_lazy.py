#!/usr/bin/env python
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse, os, torch, torch.nn as nn
from torch.utils.data import DataLoader
import torch.amp
from pathlib import Path
from datasets.frames_lazy import FramesLazyDataset
from models.mamba_regressor import MambaRegressor
from time import perf_counter
from tqdm import tqdm

def isnan(x: torch.Tensor) -> bool:
    return bool(torch.isnan(x).any().item())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features_root",required=True)
    ap.add_argument("--seq_len",type=int,required=True)
    ap.add_argument("--input_dim",type=int,required=True)     # Din
    ap.add_argument("--proj_dim",type=int,default=64)         # C (num_channels for CMamba)
    ap.add_argument("--d_model",type=int,default=128)         # CMamba width
    ap.add_argument("--n_layer",type=int,default=4)           # CMamba depth
    ap.add_argument("--patch_len",type=int,default=8)
    ap.add_argument("--stride",type=int,default=4)
    ap.add_argument("--batch_size",type=int,default=32)
    ap.add_argument("--epochs",type=int,default=60)
    ap.add_argument("--lr",type=float,default=2e-3)
    ap.add_argument("--wd",type=float,default=5e-2)
    ap.add_argument("--amp",action="store_true")
    ap.add_argument("--out_dir",default="./ckpt")
    args=ap.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(args.features_root)
    all_files = sorted(root.glob("*.npz"))
    cut = int(len(all_files)*0.8)
    train_files = all_files[:cut]
    val_files   = all_files[cut:] or all_files[:1]
    tr = FramesLazyDataset.from_filelist(train_files, seq_len=args.seq_len, predict="next", mmap=True)
    va = FramesLazyDataset.from_filelist(val_files,   seq_len=args.seq_len, predict="next", mmap=True)

    model = MambaRegressor(Din=args.input_dim, K=args.seq_len,
                           proj_dim=args.proj_dim, d_model=args.d_model,
                           n_layer=args.n_layer, patch_len=args.patch_len, stride=args.stride).to(device)

    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit=nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)# type: ignore

    def run_epoch(loader, training=True):
        model.train() if training else model.eval()
        tot, n = 0.0, 0
        start = perf_counter()
        pbar = tqdm(loader, ncols=100, desc="train" if training else "val")
        with torch.set_grad_enabled(training):
            for i, (xb,yb) in enumerate(pbar, start=1):
                xb=xb.to(device,non_blocking=True)        # (B,K,Din)
                yb=yb.squeeze(1).to(device,non_blocking=True)  # (B,2)
                if isnan(xb):
                    print(f"[WARN] NaN in inputs at batch {i}")
                #with torch.amp.autocast('cuda', enabled=args.amp):# type: ignore
                with torch.amp.autocast('cuda', enabled=False):# type: ignore
                    yhat=model(xb)                # (B,2)

                    if isnan(yhat):
                        print(f"[WARN] NaN in model output at batch {i}")        

                    loss=crit(yhat,yb)
                    if isnan(loss):
                        print(f"[WARN] NaN in loss at batch {i} | yhat[{yhat.min().item():.3e},{yhat.max().item():.3e}] "
                            f"yb[{yb.min().item():.3e},{yb.max().item():.3e}]")
                # —— 新增：直接在米单位上报告误差 —— #
                with torch.no_grad():
                    err_m = torch.sqrt(((yhat - yb) ** 2).sum(dim=1))  # (B,)
                    mean_err_m = err_m.mean().item()
                    
                if training:

                    if torch.isnan(yhat).any() or torch.isinf(yhat).any():
                        print(f"[SKIP] NaN/Inf yhat at batch {i}, skip update")
                        continue
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"[SKIP] NaN/Inf loss at batch {i}, skip update")
                        continue
                    scaler.scale(loss).backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) ##nan issue

                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)
                bs = xb.size(0)
                tot += loss.item() * bs; n += bs
                # --- 新增: 每隔 50 个 batch 打印一次进度 ---
                '''
                if i % 50 == 0 or i == len(loader):
                    avg_loss = tot / max(1, n)
                    print(f"   [{'train' if training else 'val'}] batch {i}/{len(loader)} avg_loss={avg_loss:.4f}")
                '''
                elapsed = perf_counter() - start
                # 核心：显示平均loss与吞吐量（样本/秒）
                #pbar.set_postfix(avg_loss=f"{tot/max(1,n):.4f}", ips=f"{n/max(1e-6,elapsed):.1f}")
                pbar.set_postfix(avg_loss=f"{tot/max(1,n):.4f}", ips=f"{n/max(1e-6,elapsed):.1f}",
                 mean_pos_err_m=f"{mean_err_m:.3f}")
        return tot / max(1, n)

    os.makedirs(args.out_dir,exist_ok=True)
    best=1e9
    for ep in range(1,args.epochs+1):
        tr_loss=run_epoch(tr, True)
        va_loss=run_epoch(va, False)
        if va_loss<best: # type: ignore
            best=va_loss
            torch.save({"model":model.state_dict(),"args":vars(args)},
                       os.path.join(args.out_dir,"best.pt"))
        print(f"Epoch {ep:03d} | train {tr_loss:.4f} | val {va_loss:.4f} | best {best:.4f}")

if __name__=="__main__": main()
