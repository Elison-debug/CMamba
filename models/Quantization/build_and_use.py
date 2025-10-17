import torch
from torch.utils.cpp_extension import load
import os

# Windows + MSVC + conda 环境可直接用 load JIT 编译
# 首次会比较慢，之后走缓存
src = os.path.abspath("quant_cpu.cpp")
quant = load(name="quant_cpu", sources=[src], verbose=True)

# ---- 前向用 C++位一致，反向用 STE 的 autograd.Function ----
class QuantPerTensorAsymSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zp, qmin, qmax):
        xq = quant.quant_per_tensor_asym(x.contiguous(), float(scale), int(zp), int(qmin), int(qmax)) # type: ignore
        # 反量化成 float，做“假量化”
        xhat = quant.dequant_per_tensor_asym(xq, float(scale), int(zp)) # type: ignore
        return xhat
    @staticmethod
    def backward(ctx, gO):
        # 直通梯度：把上游梯度原样传回
        return gO, None, None, None, None

class QuantPerChannelSymSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, scale, ch_axis, qmin, qmax):
        wq = quant.quant_per_channel_sym(w.contiguous(), scale.contiguous(), int(ch_axis), int(qmin), int(qmax)) # type: ignore
        # 反量化
        # zp=0（对称）
        # 这里简单写：w_hat = (wq * scale[chan])，按 ch_axis broadcast
        shape = [1]*w.dim(); shape[ch_axis] = -1
        w_hat = (wq.to(torch.float32) * scale.view(shape))
        return w_hat
    @staticmethod
    def backward(ctx, gO):
        return gO, None, None, None, None

# ---- 一个可直接替换 nn.Linear 的 QAT 线性层 ----
class QLinearINT8(torch.nn.Module):
    def __init__(self, in_f, out_f, bias=True, a_bits=8, w_bits=8, ch_axis=0):
        super().__init__()
        self.lin = torch.nn.Linear(in_f, out_f, bias=bias)
        # 可学习或冻结的 scale（QAT 推荐可学习）
        self.a_scale = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.a_zp    = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))  # 非对称可学
        self.w_scale = torch.nn.Parameter(torch.ones(out_f, dtype=torch.float32))  # per-channel
        self.a_bits, self.w_bits = a_bits, w_bits
        self.ch_axis = ch_axis

    @staticmethod
    def _qrange(bits, symmetric):
        if symmetric: return -(1<<(bits-1)), (1<<(bits-1))-1
        else:         return 0, (1<<bits)-1

    def forward(self, x):
        # 激活：per-tensor 非对称
        qmin_a, qmax_a = self._qrange(self.a_bits, symmetric=False)
        x_hat = QuantPerTensorAsymSTE.apply(x, self.a_scale.abs()+1e-8, self.a_zp, qmin_a, qmax_a)

        # 权重：per-channel 对称，ch_axis=0 对应 out_channels
        qmin_w, qmax_w = self._qrange(self.w_bits, symmetric=True)
        w = self.lin.weight
        w_hat = QuantPerChannelSymSTE.apply(w, self.w_scale.abs()+1e-8, self.ch_axis, qmin_w, qmax_w)

        y = torch.nn.functional.linear(x_hat, w_hat, self.lin.bias) # type: ignore
        return y
