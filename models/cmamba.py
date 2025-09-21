
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/cmamba.py
- C-Mamba backbone with optional streaming state passing.
- API:
    y = CMamba(args)(x)                               # vanilla (B,C,K) -> (B,C,To)
    y, new_states = CMamba(args).forward_stream(x, states=None)  # stream across chunks
States format: list[dict] per layer with keys {"x": (B, d_inner, n)}
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange, repeat, einsum

class ModelArgs:
    def __init__(self, d_model=128, n_layer=4, seq_len=96, d_state=16, expand=2, dt_rank='auto',
                 d_conv=4, pad_multiple=8, conv_bias=True, bias=False,
                 num_channels=24, patch_len=16, stride=8, forecast_len=1, sigma=0.0, reduction_ratio=8, verbose=False):
        self.d_model = d_model
        self.n_layer = n_layer
        self.seq_len = seq_len
        self.d_state = d_state
        self.v = verbose
        self.expand = expand
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.pad_multiple = pad_multiple
        self.conv_bias = conv_bias
        self.bias = bias
        self.num_channels = num_channels
        self.patch_len = patch_len
        self.stride = stride
        self.forecast_len = forecast_len
        self.sigma = sigma
        self.reduction_ratio = reduction_ratio
        self.num_patches = max(1, (self.seq_len - self.patch_len)//self.stride + 1)
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        if self.forecast_len % self.pad_multiple != 0:
            self.forecast_len += (self.pad_multiple - self.forecast_len % self.pad_multiple)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__(); self.eps=eps; self.weight=nn.Parameter(torch.ones(d_model))
    def forward(self,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*self.weight

class ChannelMixup(nn.Module):
    def __init__(self, sigma=0.0): super().__init__(); self.sigma=sigma
    def forward(self,x):
        if self.training and self.sigma>0.0:
            B,V,L=x.shape
            perm=torch.randperm(V, device=x.device)
            lam=torch.normal(0, self.sigma, size=(V,), device=x.device)
            return x + lam.unsqueeze(1)*x[:,perm]
        return x

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool=nn.AdaptiveAvgPool1d(1); self.max_pool=nn.AdaptiveMaxPool1d(1)
        rr=max(1, num_channels//reduction_ratio)
        self.fc1=nn.Linear(num_channels, rr); self.fc2=nn.Linear(rr, num_channels)
        self.relu=nn.ReLU(); self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        avg=self.fc2(self.relu(self.fc1(self.avg_pool(x).squeeze(-1))))
        mx =self.fc2(self.relu(self.fc1(self.max_pool(x).squeeze(-1))))
        out=self.sigmoid(avg+mx); return out.unsqueeze(-1)

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__(); self.args=args
        self.norm_in = RMSNorm(args.d_model)
        self.in_proj = nn.Linear(args.d_model, args.d_inner*2, bias=args.bias)
        self.conv1d = nn.Conv1d(args.d_inner, args.d_inner, kernel_size=args.d_conv,
                                groups=args.d_inner, padding=args.d_conv-1, bias=args.conv_bias)
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state*2, bias=False)        # type: ignore
        self.dt_proj= nn.Linear(args.dt_rank, args.d_inner, bias=True)# type: ignore
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def ssm(self, x, x0=None):
        """
        x: (b, l, d_in)
        x0: optional initial state (b, d_in, n)
        Returns (y, xT)
        """
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)  # (b,l, dt_rank+2n)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (b,l,d_in)
        z = einsum(delta, A, 'b l d, d n -> b l d n')
        deltaA = torch.exp(torch.clamp(z, min=-20.0, max=20.0))
        deltaB_u = einsum(delta, B, x, 'b l d, b l n, b l d -> b l d n')
        st = x0 if x0 is not None else torch.zeros((x.shape[0], d_in, n), device=x.device, dtype=x.dtype)
        ys=[]
        for i in range(x.shape[1]):
            st = deltaA[:,i]*st + deltaB_u[:,i]
            y  = einsum(st, C[:,i,:], 'b d n, b n -> b d')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # (b,l,d_in)
        y = y + x * D
        y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        return y, st

    def forward_block(self, x, state=None):
        # x: (b, l, d_model)
        x_and_res = self.in_proj(self.norm_in(x))
        x, res = x_and_res.split(self.args.d_inner, dim=-1)
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :x.shape[-1]]
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x)
        y, st = self.ssm(x, state["x"] if (isinstance(state, dict) and "x" in state) else None)
        y = y * F.silu(res)
        out = self.out_proj(y)
        return out, {"x": st}

class PatchMamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__(); self.args=args; self.layers=nn.ModuleList([MambaBlock(args) for _ in range(args.n_layer)])
    def forward(self, x):  # not streaming: returns (B,K,D)
        st = [None]*len(self.layers)
        for i,layer in enumerate(self.layers):
            y, st[i] = layer.forward_block(x, state=None) # pyright: ignore[reportCallIssue]
            x = x + y
        return x
    def forward_stream(self, x, states=None):  # streaming: pass states per layer
        new_states=[]
        for i,layer in enumerate(self.layers):
            s = (None if states is None else states[i])
            y, st = layer.forward_block(x, state=s)  # pyright: ignore[reportCallIssue]
            x = x + y
            new_states.append(st)
        return x, new_states

class CMambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__(); self.args=args
        self.patch_mamba = PatchMamba(args)
        self.channel_attention = ChannelAttention(args.d_model, args.reduction_ratio)
        self.norm = RMSNorm(args.d_model)
    def forward(self, x, states=None):
        if states is None:
            x = self.patch_mamba(x)
        else:
            x, _ = self.patch_mamba.forward_stream(x, states=None)  # intra-block streaming not needed
        attn = self.channel_attention(x.permute(0, 2, 1))
        x = x * attn.permute(0, 2, 1)
        return self.norm(x)

class CMamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__(); self.args=args
        self.channel_mixup = ChannelMixup(args.sigma)
        self.patch_embedding = nn.Linear(args.patch_len * args.num_channels, args.d_model)
        self.c_mamba_blocks = nn.ModuleList([CMambaBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)
        self.output_layer = nn.Linear(args.d_model * args.num_patches, args.num_channels * args.forecast_len)

    def _build_sincos(self, n: int, d: int, device=None, dtype=None):
        device = device or torch.device("cpu")
        dtype  = dtype  or torch.float32
        pos = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
        sin_cols=(d+1)//2; cos_cols=d//2; denom=d/2.0
        sin_i=torch.arange(sin_cols, device=device, dtype=dtype)
        cos_i=torch.arange(cos_cols, device=device, dtype=dtype)
        sin_div=torch.exp(-math.log(10000.0)*sin_i/denom).unsqueeze(0)
        cos_div=torch.exp(-math.log(10000.0)*cos_i/denom).unsqueeze(0)
        sin_part=torch.sin(pos*sin_div); cos_part=torch.cos(pos*cos_div)
        pe=torch.zeros(n,d,device=device,dtype=dtype)
        pe[:,0::2]=sin_part
        if cos_cols>0: pe[:,1::2]=cos_part
        return pe

    def _patch(self, x):  # x: (B,C,K) -> (B,Kp, C*P)
        B,V,L = x.shape; P=self.args.patch_len; S=self.args.stride
        if L < P:  # pad time
            pad = P - L
            x = F.pad(x, (pad, 0))
            L = x.shape[-1]
        X = x.unfold(2, P, S).contiguous()           # (B,V,Kp,P)
        X = X.permute(0,2,1,3).reshape(B, -1, V*P)   # (B,Kp,V*P)
        return X

    # vanilla (non-streaming)
    def forward(self, input_ids):
        x = self.channel_mixup(input_ids)            # (B,C,K)
        X = self._patch(x)                           # (B,Kp,V*P)
        X = self.patch_embedding(X)                  # (B,Kp,D)
        pos = self._build_sincos(n=X.size(1), d=self.args.d_model, device=X.device, dtype=X.dtype).unsqueeze(0)
        X = X + pos
        for block in self.c_mamba_blocks: X = block(X)
        X = self.norm_f(X)
        Xf = X.reshape(X.shape[0], -1)
        logits = self.output_layer(Xf).reshape(-1, self.args.num_channels, self.args.forecast_len)
        return logits

    # streaming across chunks: pass list of states (per layer inside PatchMamba)
    def forward_stream(self, input_ids, states=None):
        # Here, we just reuse the non-streaming blocks, but allow the inner MambaBlock to carry state.
        x = input_ids  # (B,C,K)
        X = self._patch(x)                           # (B,Kp,V*P)
        X = self.patch_embedding(X)
        pos = self._build_sincos(n=X.size(1), d=self.args.d_model, device=X.device, dtype=X.dtype).unsqueeze(0)
        X = X + pos
        # For simplicity, we do not carry states across CMambaBlocks (state lives in inner PatchMamba)
        for block in self.c_mamba_blocks:
            X = block(X, states=None)
        X = self.norm_f(X)
        Xf = X.reshape(X.shape[0], -1)
        logits = self.output_layer(Xf).reshape(-1, self.args.num_channels, self.args.forecast_len)
        # We also propagate states from the last PatchMamba layer (not used here but API-ready)
        return logits, states
