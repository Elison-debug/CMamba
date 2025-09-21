
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate with Blocked+Embargo policy:
- For each grid file in features_root/all, cut into blocks of length --block.
- Odd blocks -> eval, even -> train, with an embargo of E frames between sets.
- We only EVALUATE a given checkpoint on the eval blocks (no retrain here).
- Report overall EPE and per-type EPE.
"""
from __future__ import annotations
import argparse, os, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.packed_stream import PackedStreamingDataset, packed_collate
from utils.splits import blocked_embargo_blocks, load_grid_infos
from models.mamba_regressor import MambaRegressor

def epe(a, b): return torch.sqrt(((a-b)**2).sum(dim=-1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--input_dim", type=int, required=True)
    ap.add_argument("--block", type=int, default=500)
    ap.add_argument("--embargo", type=int, default=64)
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
    ds = PackedStreamingDataset(all_dir, seq_len=args.seq_len, use_vel=carg.get("out_vel",False),
                                use_acc=carg.get("out_acc",False))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=packed_collate)

    per_type_sum = {"straight":0.0,"circle":0.0,"random":0.0}
    per_type_cnt = {"straight":0.0,"circle":0.0,"random":0.0}
    tot_sum=0.0; tot_cnt=0.0

    with torch.no_grad():
        pbar = tqdm(dl, total=len(dl), ncols=120, desc="eval B+E")
        for x, pos, nxt, vel, acc, msk, grids, kinds in pbar:
            # We simulate B+E by zeroing masks for segments whose end t is inside train blocks
            B,S,K,D = x.shape
            # Build mask per grid using its T inferred from original npz length S*K approx. Here we approximate blocks at segment granularity.
            be_mask = torch.ones_like(msk)
            # Note: packed segments end at t=end-1 with end=s*K + len-1
            # Using embargo ~ K is reasonable.
            for bi in range(B):
                # approximate T via segments * K
                # For more exact, one can load the specific npz again, skipped for speed.
                T_approx = int(S*K)
                train_blks, eval_blks = blocked_embargo_blocks(T_approx, args.block, args.embargo)
                # mark segments that end inside eval blocks
                valid = torch.zeros(S, dtype=torch.float32)
                for (s,e) in eval_blks:
                    # segment s index ends at t = (seg+1)*K - 1
                    for si in range(S):
                        t_end = (si+1)*K - 1
                        if s <= t_end < e: valid[si]=1.0
                be_mask[bi] = be_mask[bi] * valid.to(be_mask.device)
            x=x.to(device); pos=pos.to(device); msk=(msk*be_mask).to(device)
            states=None; epe_sum=0.0; cnt=0.0
            for s in range(S):
                out, states = model(x[:,s].to(device), states=states, stream=True)
                e = epe(out["pos"], pos[:,s]) * msk[:,s]
                epe_sum += e.sum().item(); cnt += float(msk[:,s].sum().item())
            k = kinds[0] if len(set(kinds))==1 else "random"
            per_type_sum[k] += epe_sum; per_type_cnt[k] += cnt
            tot_sum += epe_sum; tot_cnt += cnt
            pbar.set_postfix(mean=f"{(tot_sum/max(1.0,tot_cnt)):.3f}")
    print("[RESULT] Blocked+Embargo EPE (m):")
    for t in ["straight","circle","random"]:
        m = per_type_sum[t]/max(1.0,per_type_cnt[t])
        print(f"  {t:8s}: {m:.4f}  (N={int(per_type_cnt[t])})")
    print(f"  overall : {tot_sum/max(1.0,tot_cnt):.4f}  (N={int(tot_cnt)})")

if __name__ == "__main__":
    main()
