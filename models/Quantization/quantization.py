import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 基础工具 ----
def round_ste(x):
    # 直通估计：前向四舍五入，反向当作恒等映射
    class RoundSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp): return torch.round(inp)
        @staticmethod
        def backward(ctx, g):  return g
    return RoundSTE.apply(x)

def clamp_ste(x, lo, hi):
    class ClampSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, lo, hi): return torch.clamp(inp, lo, hi)
        @staticmethod
        def backward(ctx, g, *args):   return g, None, None
    return ClampSTE.apply(x, lo, hi)

# ---- LSQ 量化器（支持权重 per-channel，对称；激活 per-tensor，非对称）----
class LSQQuantizer(nn.Module):
    def __init__(self, bits=8, per_channel=False, ch_axis=0, symmetric=True, learn_scale=True):
        super().__init__()
        self.bits = bits
        self.qmin = -(2**(bits-1)) if symmetric else 0
        self.qmax = (2**(bits-1)-1) if symmetric else (2**bits-1)
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.symmetric = symmetric
        self.learn_scale = learn_scale
        self.register_parameter("scale", None)  # 延迟初始化
        self.register_buffer("zero_point", torch.zeros(1))  # 非对称时可学习/可缓冲

    def _init_params(self, x):
        # Kaiming-like：用统计量初始化 scale
        with torch.no_grad():
            if self.per_channel:
                dims = [i for i in range(x.dim()) if i != self.ch_axis]
                s = x.detach().abs().mean(dim=dims, keepdim=True) * 2.0
            else:
                s = x.detach().abs().mean() * 2.0
            if self.learn_scale:
                self.scale = nn.Parameter(s.clamp(min=1e-8))
            else:
                self.register_buffer("scale", s.clamp(min=1e-8))
            if not self.symmetric:
                # 激活常用：估计零点（这里用均值法，可换 minmax/KLD）
                zp = - (x.detach().mean() / self.scale).round()
                self.zero_point = zp.to(x).reshape_as(self.scale)

    def forward(self, x):
        if self.scale is None:
            self._init_params(x)

        s = self.scale
        if self.symmetric:
            zp = 0.0
            x_int = round_ste(clamp_ste(x / s, self.qmin, self.qmax))
        else:
            zp = self.zero_point
            x_int = round_ste(clamp_ste(x / s + zp, self.qmin, self.qmax))

        # 反量化（假量化）
        if self.symmetric:
            x_hat = (x_int * s) # type: ignore
        else:
            x_hat = (x_int - zp) * s # type: ignore

        return x_hat

# ---- 量化线性层（权重 per-channel，对称；激活 per-tensor，非对称）----
class QuantLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=True, w_bits=8, a_bits=8):
        super().__init__()
        self.lin = nn.Linear(in_f, out_f, bias=bias)
        self.qw = LSQQuantizer(bits=w_bits, per_channel=True, ch_axis=0, symmetric=True, learn_scale=True)
        self.qa = LSQQuantizer(bits=a_bits, per_channel=False, symmetric=False, learn_scale=True)

    def forward(self, x):
        # 激活先量化（假量化）
        xq = self.qa(x)
        # 权重量化（假量化）
        wq = self.qw(self.lin.weight)
        b  = self.lin.bias
        y  = F.linear(xq, wq, b)
        return y
