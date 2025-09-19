#!/usr/bin/env python3
"""
models/mamba_regressor.py
Wrap your CMamba (channel-first, multivariate forecasting) into a regression model for (x,y).

Pipeline:
  (B, K, Din) --per-frame Linear(Din->C)--> (B, K, C) --permute--> (B, C, K)
  --> CMamba(args)  # args.num_channels=C, args.seq_len=K, args.forecast_len=1
  Output: (B, C, 1) -> squeeze time -> (B, C) -> Linear(C->2)

Notes:
- Choose C (proj_dim) small, e.g., 64 or 128.
- Patch settings: patch_len=8, stride=4 by default (can be overridden).
- d_model ties to your CMamba's hidden width.
"""
import torch.nn as nn
import torch.nn.functional as F
# Import your CMamba & ModelArgs (adjust module path if needed)
try:
    from models.cmamba import CMamba, ModelArgs
except Exception:
    # fallback (some repos place it differently)
    from cmamba import CMamba, ModelArgs

class MambaRegressor(nn.Module):
    def __init__(self,
                 Din:int,           # input per-frame feature dim (e.g., 2000)
                 K:int,             # seq_len (window length)
                 proj_dim:int=64,   # num_channels C for CMamba
                 d_model:int=128,   # CMamba hidden width
                 n_layer:int=4,     # CMamba depth
                 patch_len:int=8,
                 stride:int=4):
        super().__init__()
        self.Din = Din
        self.K = K
        self.input_norm = nn.LayerNorm(proj_dim)
        self.proj = nn.Linear(Din, proj_dim)  # frame-wise projection to C
        # Build CMamba config
        args = ModelArgs(
            d_model=d_model,
            n_layer=n_layer,
            seq_len=K,
            num_channels=proj_dim,  # channels = projected features
            patch_len=patch_len,
            stride=stride,
            forecast_len=1,         # we only need one step
            pad_multiple=1,         # ← 加上这行，避免 1 被 pad 成 8
            sigma=0.0,
            # keep other defaults from ModelArgs
        )
        self.backbone = CMamba(args)
        self.head = nn.Linear(proj_dim, 2)    # map channels -> (x,y)

    def forward(self, x):  # x: (B,K,Din)
        B, K, Din = x.shape
        assert Din == self.Din, f"Expected Din={self.Din}, got {Din}"
        if K != self.K:
            if K > self.K:
                # 多了 → 取最近 self.K 帧
                x = x[:, -self.K:, :]
            else:
                # 少了 → 在时间维前端补零到 self.K
                pad = self.K - K
                # F.pad 的维度顺序是 (..., last_dim, second_last_dim, ...)
                # 这里 pad (features_dim=0, time_dim=pad 前, time_dim 后=0)
                x = F.pad(x, (0, 0, pad, 0))
            K = self.K  # 同步更新（仅用于后续 sanity）
        x = self.proj(x)            # (B,K,C)
        x = self.input_norm(x)      # 替代全局均值缩放
        x = x.permute(0,2,1)        # (B,C,K)
        y = self.backbone(x)        # (B,C,Tout) -- 注意 Tout 可能是 1 或被 pad 成 8
        # --- 关键修复：把最后时间维压成标量 ---
        if y.dim() == 3:
            y = y.mean(dim=-1)      # -> (B,C)
        out = self.head(y)          # (B,2)
        return out
