
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/splits.py
- GroupKFold by grid id (at least 5 folds)
- Blocked+Embargo split generator
- Walk-forward windows generator
- Type balancing helpers
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
import math
import json
from pathlib import Path

@dataclass(frozen=True)
class GridInfo:
    grid: str
    T: int
    kind: str  # "straight" | "circle" | "random"

def make_group_kfold(grids: List[GridInfo], k: int = 5, seed: int = 2025) -> List[Tuple[List[str], List[str]]]:
    """
    Return list of (train_grids, val_grids) using group-wise K folds.
    Try to balance by 'kind' across folds.
    """
    rng = np.random.default_rng(seed)
    kinds = sorted(set(g.kind for g in grids))
    # Partition grids by kind
    by_kind: Dict[str, List[GridInfo]] = {t: [] for t in kinds}
    for g in grids: by_kind[g.kind].append(g)
    # shuffle each bucket
    for t in kinds: rng.shuffle(by_kind[t]) # pyright: ignore[reportArgumentType]
    # Round-robin assign to k folds
    folds: List[List[GridInfo]] = [[] for _ in range(k)]
    for t in kinds:
        for i, g in enumerate(by_kind[t]):
            folds[i % k].append(g)
    # produce splits
    out: List[Tuple[List[str], List[str]]] = []
    for i in range(k):
        val = [g.grid for g in folds[i]]
        tr  = [g.grid for j in range(k) if j != i for g in folds[j]]
        out.append((tr, val))
    return out

def blocked_embargo_blocks(T:int, block:int, embargo:int) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
    """
    Cut 0..T-1 into contiguous blocks of length 'block' (last shorter ok).
    Return odd blocks as train and even blocks as eval, but enforce an embargo
    of E frames between the sets.
    """
    blocks = []
    s = 0
    while s < T:
        e = min(T, s + block)
        blocks.append((s, e))
        s = e
    train, eval_ = [], []
    for i, (s,e) in enumerate(blocks):
        if (i % 2) == 1: eval_.append((s,e))
        else: train.append((s,e))
    # Apply embargo
    def _shrink(blks, others):
        out=[]
        for (s,e) in blks:
            s2, e2 = s, e
            for (u,v) in others:
                # if touching, carve embargo gap
                if v <= s and (s - v) < embargo:
                    s2 = max(s2, v + embargo)
                if e <= u and (u - e) < embargo:
                    e2 = min(e2, u - embargo)
            if e2 > s2:
                out.append((s2,e2))
        return out
    train2 = _shrink(train, eval_)
    eval2  = _shrink(eval_, train)
    return train2, eval2

def walk_forward_windows(T:int, init_ratio:float=0.6, step_ratio:float=0.1) -> List[Tuple[Tuple[int,int], Tuple[int,int]]]:
    """
    Generate a sequence of (train_span, eval_span) pairs that roll forward.
    Each eval span is step_ratio of T (bounded to [1..]), appended to train for next round.
    """
    assert 0.05 <= step_ratio <= 0.5
    assert 0.2 <= init_ratio < 0.9
    step = max(1, int(round(T * step_ratio)))
    cur_end = max(1, int(round(T * init_ratio)))
    spans = []
    while cur_end < T:
        ev_e = min(T, cur_end + step)
        spans.append( ((0, cur_end), (cur_end, ev_e)) )
        cur_end = ev_e
    return spans

def load_grid_infos(features_root: str) -> List[GridInfo]:
    """
    Expect per-grid npz under features_root/all/GRIDxxxx.npz with meta json, fields: T, kind.
    If not found, try train/ and eval/ then dedup by basename.
    """
    root = Path(features_root)
    cand_dirs = [root / "all", root]
    # fallback to train/eval union
    if not (root/"all").exists():
        cand_dirs = [root/"train", root/"eval"]
    files = []
    for d in cand_dirs:
        if d is None or not d.exists(): continue
        files += list(d.glob("*.npz"))
    # Dedup by basename
    dedup = {}
    for p in files:
        k = p.name
        if k not in dedup: dedup[k] = p
    infos: List[GridInfo] = []
    for p in sorted(dedup.values()):
        try:
            d = np.load(p, allow_pickle=True)
            meta = json.loads(d["meta"].item()) if hasattr(d["meta"], "item") else json.loads(d["meta"].tolist())
            kind = meta.get("kind","straight")
            T = int(meta.get("T", d["xy"].shape[0]))
            grid = p.stem
            infos.append(GridInfo(grid=grid, T=T, kind=kind))
        except Exception:
            continue
    return infos
