# datasets/npz_xy.py
import numpy as np
import torch
from torch.utils.data import Dataset

class NPZXYDataset(Dataset):
    def __init__(self, path, split='train'):
        data = np.load(path)
        self.X = data['X_train'] if split=='train' else data['X_val']
        self.Y = data['y_train'] if split=='train' else data['y_val']
        # (N, K, Din), (N, 1, 2)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).float()   # (K, Din)
        y = torch.from_numpy(self.Y[i]).float()   # (1, 2)
        return x, y
