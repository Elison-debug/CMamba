
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datasets/packed_stream.py
- Packed streaming dataset: one sample = one grid packed into segments of length K.
- Returns tensors:
    x: (S, K, Din)  segments (last padded if needed)
    y_pos:  (S, 2)  position at segment end t
    y_next: (S, 2)  next position at t+1 (if available, else copy t)
    y_vel:  (S, 2)  velocity at t
    y_acc:  (S, 2)  acceleration at t
    mask:   (S,)    1 if valid segment
    grid:   str
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np, json
import torch
from torch.utils.data import Dataset

class PackedStreamingDataset(Dataset):
    def __init__(self, root:str, seq_len:int, use_vel:bool=False, use_acc:bool=False, mmap:bool=True,
                 grids: Optional[List[str]]=None, stats_path: Optional[str]=None):
        super().__init__()
        rootp = Path(root)
        files = sorted(rootp.glob("*.npz"))
        if grids is not None:
            files = [rootp/f"{g}.npz" for g in grids if (rootp/f"{g}.npz").exists()]
        if not files: raise FileNotFoundError(f"No .npz under {root}")
        self.files = files; self.K = int(seq_len); self.use_vel=bool(use_vel); self.use_acc=bool(use_acc)
        self._mmap = mmap
        # load stats
        self.stats = None
        if stats_path and Path(stats_path).exists():
            s = np.load(stats_path)
            m = s["feat_mean"].astype(np.float32)
            sd = s["feat_std"].astype(np.float32); sd = np.maximum(sd, float(s.get("std_floor", 1e-3)))
            self.stats = (m, sd)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx:int):
        d = np.load(self.files[idx], allow_pickle=True, mmap_mode=("r" if self._mmap else None))
        X = d["feats"].astype(np.float32)
        xy = d["xy"].astype(np.float32)
        ts = d["ts"].astype(np.float64)
        v  = d["v"].astype(np.float32) if "v" in d else np.zeros_like(xy, dtype=np.float32)
        a  = d["a"].astype(np.float32) if "a" in d else np.zeros_like(xy, dtype=np.float32)
        meta = json.loads(d["meta"].item()) if hasattr(d["meta"], "item") else json.loads(d["meta"].tolist())
        T, Din = X.shape
        if self.stats is not None:
            m, sd = self.stats; X = (X - m)/sd
        # pack into segments of length K
        K = self.K
        S = max(1, (T-1 + K-1)//K)  # ensure next available for last t if T-1 index exists
        x_segs = np.zeros((S, K, Din), dtype=np.float32)
        y_pos  = np.zeros((S, 2), dtype=np.float32)
        y_next = np.zeros((S, 2), dtype=np.float32)
        y_vel  = np.zeros((S, 2), dtype=np.float32)
        y_acc  = np.zeros((S, 2), dtype=np.float32)
        mask   = np.zeros((S,), dtype=np.float32)
        for s in range(S):
            start = s*K
            end   = min(T, start+K)
            xs = X[start:end]
            # pad front so that the end aligns to the last timestep 'end-1'
            if xs.shape[0] < K:
                pad = K - xs.shape[0]
                xs = np.pad(xs, ((pad,0),(0,0)), mode="edge")
            x_segs[s] = xs
            t = end - 1
            y_pos[s]  = xy[t]
            y_next[s] = xy[t+1] if t+1 < T else xy[t]
            y_vel[s]  = v[t]
            y_acc[s]  = a[t]
            mask[s]   = 1.0
        out = {
            "x": torch.from_numpy(x_segs),      # (S,K,Din)
            "pos": torch.from_numpy(y_pos),     # (S,2)
            "next": torch.from_numpy(y_next),   # (S,2)
            "vel": torch.from_numpy(y_vel),     # (S,2)
            "acc": torch.from_numpy(y_acc),     # (S,2)
            "mask": torch.from_numpy(mask),     # (S,)
            "grid": meta.get("grid", Path(self.files[idx]).stem),
            "kind": meta.get("kind","straight"),
        }
        return out

def packed_collate(batch: List[Dict[str, Any]]):
    # batch of size B; each element has (S_i,K,Din)
    B = len(batch)
    Smax = max(int(b["x"].shape[0]) for b in batch)
    K   = int(batch[0]["x"].shape[1]); Din=int(batch[0]["x"].shape[2])
    x   = torch.zeros((B, Smax, K, Din), dtype=torch.float32)
    pos = torch.zeros((B, Smax, 2), dtype=torch.float32)
    nxt = torch.zeros((B, Smax, 2), dtype=torch.float32)
    vel = torch.zeros((B, Smax, 2), dtype=torch.float32)
    acc = torch.zeros((B, Smax, 2), dtype=torch.float32)
    msk = torch.zeros((B, Smax), dtype=torch.float32)
    grids=[]; kinds=[]
    for i,b in enumerate(batch):
        Si = b["x"].shape[0]
        x[i,:Si]   = b["x"]
        pos[i,:Si] = b["pos"]
        nxt[i,:Si] = b["next"]
        vel[i,:Si] = b["vel"]
        acc[i,:Si] = b["acc"]
        msk[i,:Si] = b["mask"]
        grids.append(b["grid"]); kinds.append(b["kind"])
    return x, pos, nxt, vel, acc, msk, grids, kinds
