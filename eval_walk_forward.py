
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-forward (short-horizon online) evaluation:
- For each grid, we simulate streaming inference; we do NOT update model weights, but
  we report metrics on consecutive eval windows of size ratio 5–10%.
  (For full retraining per step, run train_stream.py repeatedly with generated splits.)
"""
from __future__ import annotations
import argparse, os, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.packed_stream import PackedStreamingDataset, packed_collate
from utils.splits import walk_forward_windows
from models.mamba_regressor import MambaRegressor

def epe(a,b): return torch.sqrt(((a-b)**2).sum(dim=-1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--input_dim", type=int, required=True)
    ap.add_argument("--init_ratio", type=float, default=0.6)
    ap.add_argument("--step_ratio", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    carg = ckpt.get("args", {})
    model = MambaRegressor(Din=args.input_dim, K=args.seq_len, proj_dim=carg.get("proj_dim",64),
                           d_model=carg.get("d_model",128), n_layer=carg.get("n_layer",4),
                           patch_len=carg.get("patch_len",8), stride=carg.get("stride",4),
                           out_vel=carg.get("out_vel",False), out_acc=carg.get("out_acc",False)).to(device)
    model.load_state_dict(ckpt["model"], strict=False); model.eval()

    all_dir = str(Path(args.features_root)/"all")
    ds = PackedStreamingDataset(all_dir, seq_len=args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=packed_collate)

    tot_sum=0.0; tot_cnt=0.0
    with torch.no_grad():
        pbar = tqdm(dl, total=len(dl), ncols=120, desc="walk-fwd")
        for x, pos, nxt, vel, acc, msk, grids, kinds in pbar:
            B,S,K,D = x.shape
            # emulate windows by masking segments according to walk-forward pairs
            # approximate T=S*K
            wf_masks = torch.zeros_like(msk)
            for bi in range(B):
                T = int(S*K)
                spans = walk_forward_windows(T, init_ratio=args.init_ratio, step_ratio=args.step_ratio)
                # evaluate on each eval span: mark segments falling in eval span
                for (tr, ev) in spans:
                    s,e = ev
                    for si in range(S):
                        t_end = (si+1)*K - 1
                        if s <= t_end < e: wf_masks[bi,si]=1.0
            x=x.to(device); pos=pos.to(device); mask=(msk*wf_masks).to(device)
            states=None; e_sum=0.0; cnt=0.0
            for s in range(S):
                out, states = model(x[:,s], states=states, stream=True)
                e = epe(out["pos"], pos[:,s]) * mask[:,s]
                e_sum += e.sum().item(); cnt += float(mask[:,s].sum().item())
            tot_sum += e_sum; tot_cnt += cnt
            pbar.set_postfix(mean=f"{(tot_sum/max(1.0,tot_cnt)):.3f}")
    print(f"[RESULT] Walk-forward mean EPE = {tot_sum/max(1.0,tot_cnt):.4f} m (N={int(tot_cnt)})")

if __name__ == "__main__":
    main()
