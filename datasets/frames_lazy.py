# datasets/frames_lazy.py
from __future__ import annotations
import os
import glob
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class FramesLazyDataset(Dataset):
    """
    懒滑窗数据集（不在预处理阶段展开重叠窗口）：
      - 二选一：传入 `root`（目录下 *.npz）或显式 `files` 列表
      - 每个 .npz 至少包含：feats(T,Din) / xy(T,2)
      - seq_len = K；predict:
           'current' -> x=[t-K, t)  y=xy[t]
           'next'    -> x=[t-K, t)  y=xy[t+1]
      - mmap=True 使用内存映射加载，降低内存峰值

    典型用法：
        files = sorted(Path(features_root).glob("*.npz"))
        # 切分不串文件
        cut = int(len(files)*0.8)
        tr_ds = FramesLazyDataset.from_filelist(files[:cut],  seq_len=12, predict="next")
        va_ds = FramesLazyDataset.from_filelist(files[cut:],  seq_len=12, predict="next")
    """

    def __init__(
        self,
        root: Optional[str | os.PathLike] = None,
        *,
        files: Optional[Iterable[str | os.PathLike]] = None,
        seq_len: int = 12,
        predict: str = "current",
        split: Optional[str] = None,            # "train" | "val" | None
        split_ratio: float = 0.8,
        shuffle_files: bool = False,
        seed: int = 1234,
        mmap: bool = True,
    ) -> None:
        super().__init__()
        assert (root is not None) ^ (files is not None), "root 与 files 必须二选一"

        if files is None:
            root_str = str(root)  # Pylance 友好
            file_list: List[str] = sorted(glob.glob(os.path.join(root_str, "*.npz")))
            if shuffle_files:
                rng = np.random.RandomState(seed)
                rng.shuffle(file_list)
            if split in ("train", "val"):
                cut = int(len(file_list) * float(split_ratio))
                if split == "train":
                    file_list = file_list[:max(1, cut)]
                else:
                    file_list = file_list[max(1, cut):] or file_list[:1]
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
                # x=[t-K, t), y=t
                for t in range(self.seq_len, T):
                    self.index.append((fi, t - self.seq_len, t))
            else:
                # x=[t-K, t), y=t+1
                for t in range(self.seq_len, T - 1):
                    self.index.append((fi, t - self.seq_len, t + 1))

        if not self.index:
            raise RuntimeError(
                f"No samples built: files={len(self.files)}, missing_keys={missing_keys}, "
                f"total_frames={total_frames}, seq_len={self.seq_len}, predict={self.predict}"
            )
        self.stats = None
        # 约定：stats.npz 放在数据文件所在目录（或它的父目录），先就近找
        from pathlib import Path
        if len(self.files) > 0:
            p0 = Path(self.files[0]).parent
            cand = [p0/"stats.npz", p0.parent/"stats.npz"]
            for sp in cand:
                if sp.exists():
                    s = np.load(sp)
                    m = s["feat_mean"].astype(np.float32)
                    sd = s["feat_std"].astype(np.float32)
                    if "Din" in s and int(s["Din"]) != m.shape[0]:
                        print(f"[WARN] stats Din mismatch: stats={int(s['Din'])} != data={m.shape[0]}")
                    # 下限：优先 stats 里的 std_floor；否则 1e-3
                    std_floor = float(s["std_floor"]) if "std_floor" in s else 1e-3
                    sd = np.maximum(sd, std_floor).astype(np.float32)
                    self.stats = (m, sd)
                    print(f"[OK] Loaded stats: {sp} Din={m.shape[0]}")
                    break

    def __len__(self) -> int:
        return len(self.index)

    def _load_file(self, fi: int):
        if self._cache["fi"] != fi:
            d = np.load(
                self.files[fi],
                allow_pickle=True,
                mmap_mode="r" if self._mmap else None,
            )
            self._cache = {"fi": fi, "data": d}
        return self._cache["data"]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fi, s, t = self.index[idx]
        d = self._load_file(fi)
        x = d["feats"][s:t].astype(np.float32)  # (K, Din)
        y = d["xy"][t].astype(np.float32)       # (2,)
        # 若有 stats，进行在线标准化（逐样本逐维）
        if self.stats is not None:
            m, sd = self.stats
            # 形状兼容：x (K,Din), m/sd (Din,)
            x = (x - m) / sd
        # 基础健壮性：把非数/无穷替换掉，避免上游偶发 NaN 污染训练
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        if not np.isfinite(y).all():
            y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

        xb = torch.from_numpy(x)                    # (K, Din)
        yb = torch.from_numpy(y)[None, :]           # (1, 2)
        return xb, yb

    # ------- 推荐通过固定文件列表构建，避免切分漂移 -------
    @classmethod
    def from_filelist(
        cls,
        files: Iterable[str | os.PathLike],
        *,
        seq_len: int = 12,
        predict: str = "current",
        mmap: bool = True,
    ) -> "FramesLazyDataset":
        return cls(
            files=[str(p) for p in files],
            seq_len=seq_len,
            predict=predict,
            mmap=mmap,
        )
