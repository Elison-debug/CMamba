# =========================
# Updated: datasets/frames_lazy.py (explicit split & train-stats)
# =========================

# --- save as: datasets/frames_lazy.py ---
from __future__ import annotations
import os, glob
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

class FramesLazyDataset(Dataset):
    """
    懒滑窗数据集（不展开重叠窗口）。支持：
      • root 下直接是若干 .npz；或显式 files 列表；或 root 指向某 split 目录（train/ 或 eval/）。
      • 仅使用 **train 统计**(stats_train.npz) 做归一化，避免信息泄漏。
    """
    def __init__(
        self,
        root: Optional[str | os.PathLike] = None,
        *,
        files: Optional[Iterable[str | os.PathLike]] = None,
        seq_len: int = 12,
        predict: str = "current",
        mmap: bool = True,
        stats_root: Optional[str | os.PathLike] = None,   # 若不给：自动就近搜 => 本目录/父目录的 stats_train.npz
    ) -> None:
        super().__init__()
        assert (root is not None) ^ (files is not None), "root 与 files 必须二选一"

        if files is None:
            root_str = str(root)
            file_list: List[str] = sorted(glob.glob(os.path.join(root_str, "*.npz")))
        else:
            file_list = [str(p) for p in files]
        if not file_list:
            raise FileNotFoundError("No .npz files found for FramesLazyDataset")

        self.files: List[str] = file_list
        self.seq_len: int = int(seq_len)
        if predict not in ("current", "next"):
            raise ValueError("predict must be 'current' or 'next'")
        self.predict: str = predict
        self._mmap: bool = bool(mmap)

        # 预建索引 (file_idx, start, target_t)
        self.index: List[Tuple[int, int, int]] = []
        self._cache: dict = {"fi": -1, "data": None}

        missing_keys = 0
        total_frames = 0
        for fi, path in enumerate(self.files):
            d = np.load(path, allow_pickle=True, mmap_mode="r" if self._mmap else None)
            if "feats" not in d or "xy" not in d:
                missing_keys += 1
                continue
            T = int(d["feats"].shape[0])
            total_frames += T
            if self.predict == "current":
                for t in range(self.seq_len, T):
                    self.index.append((fi, t - self.seq_len, t))
            else:
                for t in range(self.seq_len, T - 1):
                    self.index.append((fi, t - self.seq_len, t + 1))

        if not self.index:
            raise RuntimeError(
                f"No samples built: files={len(self.files)}, missing_keys={missing_keys}, "
                f"total_frames={total_frames}, seq_len={self.seq_len}, predict={self.predict}"
            )

        # ---- 载入 train 统计（优先 stats_root 提供；否则就近搜） ----
        self.stats = None
        s_candidates: List[Path] = []
        if stats_root is not None:
            s_candidates.append(Path(stats_root) / "stats_train.npz")
        if len(self.files) > 0:
            p0 = Path(self.files[0]).parent
            s_candidates += [p0/"stats_train.npz", p0.parent/"stats_train.npz", p0/"stats.npz", p0.parent/"stats.npz"]
        for sp in s_candidates:
            if sp.exists():
                try:
                    s = np.load(sp)
                    m = s["feat_mean"].astype(np.float32)
                    sd = s["feat_std"].astype(np.float32)
                    std_floor = float(s["std_floor"]) if "std_floor" in s else 1e-3
                    sd = np.maximum(sd, std_floor).astype(np.float32)
                    self.stats = (m, sd)
                    print(f"[OK] Loaded stats (train): {sp} Din={m.shape[0]}")
                    break
                except Exception as e:
                    print(f"[WARN] fail to load stats from {sp}: {e}")

    def __len__(self) -> int:
        return len(self.index)

    def _load_file(self, fi: int):
        if self._cache["fi"] != fi:
            d = np.load(self.files[fi], allow_pickle=True, mmap_mode="r" if self._mmap else None)
            self._cache = {"fi": fi, "data": d}
        return self._cache["data"]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fi, s, t = self.index[idx]
        d = self._load_file(fi)
        x = d["feats"][s:t].astype(np.float32)  # (K, Din)
        y = d["xy"][t].astype(np.float32)       # (2,)
        if self.stats is not None:
            m, sd = self.stats
            x = (x - m) / sd
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.from_numpy(x), torch.from_numpy(y)[None, :]

    @classmethod
    def from_filelist(
        cls,
        files: Iterable[str | os.PathLike],
        *,
        seq_len: int = 12,
        predict: str = "current",
        mmap: bool = True,
        stats_root: Optional[str | os.PathLike] = None,
    ) -> "FramesLazyDataset":
        return cls(files=[str(p) for p in files], seq_len=seq_len, predict=predict, mmap=mmap, stats_root=stats_root)