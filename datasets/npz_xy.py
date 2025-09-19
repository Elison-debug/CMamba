# datasets/npz_xy.py
import numpy as np, torch
from torch.utils.data import Dataset
from pathlib import Path

class NPZXYDataset(Dataset):
    """
    兼容旧格式(单文件 X_train/X_val)，但更推荐使用 FramesLazyDataset + 目录切分。
    若传入 path 为目录，则自动读取该目录下 *.npz(与 FramesLazyDataset 对齐)。
    """
    def __init__(self, path, split='train'):
        p = Path(path)
        if p.is_dir():
            # 目录模式：聚合同 split 下的所有样本（期望每 npz 存 feats/xy，不适用于旧大文件）
            files = sorted(p.glob('*.npz'))
            if not files:
                raise FileNotFoundError(f"No .npz under {p}")
            Xs, Ys = [], []
            for f in files:
                d = np.load(f)
                if 'feats' in d and 'xy' in d:
                    # 将时间切成 (T,)->(T-1,K=?,Din=?) 的完整样本并不在这里做；此类文件建议用 FramesLazyDataset
                    # 这里只做最简单兼容：整文件作为一个样本
                    Xs.append(d['feats'][None, ...])
                    Ys.append(d['xy'][None, ...])
                else:
                    # 旧格式单大文件
                    Xs.append(d['X_train' if split=='train' else 'X_val'])
                    Ys.append(d['y_train' if split=='train' else 'y_val'])
            self.X = np.concatenate(Xs, axis=0)
            self.Y = np.concatenate(Ys, axis=0)
        else:
            data = np.load(p)
            self.X = data['X_train'] if split=='train' else data['X_val']
            self.Y = data['y_train'] if split=='train' else data['y_val']

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).float()
        y = torch.from_numpy(self.Y[i]).float()
        if y.ndim == 1: y = y[None, :]
        return x, y