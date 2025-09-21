
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/mamba_regressor.py
- Wrap CMamba into a multi-head regressor:
    heads:
      pos_t     (2)  -- main
      pos_next  (2)  -- auxiliary
      vel_t     (2)  -- optional
      acc_t     (2)  -- optional
- Supports streaming over chunks (K = seq_len per chunk)
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict, Any, Tuple
try:
    from models.cmamba import CMamba, ModelArgs
except Exception:
    from cmamba import CMamba, ModelArgs

class MambaRegressor(nn.Module):
    def __init__(self,
                 Din:int,           # per-frame input dim
                 K:int,             # seq_len (window/chunk length)
                 proj_dim:int=64,   # channels for CMamba
                 d_model:int=128,
                 n_layer:int=4,
                 patch_len:int=8,
                 stride:int=4,
                 out_vel:bool=False,
                 out_acc:bool=False):
        super().__init__()
        self.Din = Din; self.K = K
        self.proj  = nn.Linear(Din, proj_dim)
        self.norm  = nn.LayerNorm(proj_dim)
        args = ModelArgs(
            d_model=d_model, n_layer=n_layer,
            seq_len=K, num_channels=proj_dim,
            patch_len=patch_len, stride=stride,
            forecast_len=1, pad_multiple=1, sigma=0.0)
        self.backbone = CMamba(args)
        self.head_pos  = nn.Linear(proj_dim, 2)
        self.head_next = nn.Linear(proj_dim, 2)
        self.out_vel = bool(out_vel); self.out_acc = bool(out_acc)
        if self.out_vel: self.head_vel = nn.Linear(proj_dim, 2)
        if self.out_acc: self.head_acc = nn.Linear(proj_dim, 2)

    def _forward_backbone(self, x, states=None, stream=False):
        # x: (B,K,Din) -> proj -> (B,C,K)
        x = self.proj(x); x = self.norm(x); x = x.permute(0,2,1)  # (B,C,K)
        if stream:
            y, states = self.backbone.forward_stream(x, states=states)
        else:
            y = self.backbone(x); states=None
        if y.dim()==3: y = y.mean(dim=-1)  # (B,C)
        return y, states

    def forward(self, x, *, states=None, stream=False) -> Tuple[Dict[str, torch.Tensor], Any]:
        feats, states = self._forward_backbone(x, states=states, stream=stream)  # (B,C)
        out = {
            "pos":  self.head_pos(feats),
            "next": self.head_next(feats),
        }
        if self.out_vel: out["vel"] = self.head_vel(feats)
        if self.out_acc: out["acc"] = self.head_acc(feats)
        return out, states
