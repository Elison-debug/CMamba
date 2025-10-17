// quant_cpu.cpp  --  PyTorch C++ extension (CPU)
#include <torch/extension.h>
#include <cmath>
#include <cstdint>
#include <limits>

// 最近偶数舍入（round to nearest even）
// Python 的 round/torch.round 在不同平台可能细节不同，这里固定行为
inline int64_t rnd_nearest_even(double v) {
    double r = std::nearbyint(v); // ties to even by default (IEEE-754), 依赖环境
    // 为了彻底可控，也可自写：手动判断 0.5 ties，然后改最末位为偶
    return static_cast<int64_t>(r);
}

template<typename T>
inline T sat(T x, T lo, T hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// per-tensor asymmetric: x_q = clamp(round(x/s + zp), qmin, qmax)
torch::Tensor quantize_per_tensor_asym(torch::Tensor x, double scale, int64_t zp, int64_t qmin, int64_t qmax) {
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    auto x_f = x.contiguous().to(torch::kFloat32);
    auto out = torch::empty_like(x_f, torch::kInt8);
    auto N = x_f.numel();

    const float* xp = x_f.data_ptr<float>();
    int8_t* op = out.data_ptr<int8_t>();

    const double s_inv = 1.0 / scale;
    for (int64_t i = 0; i < N; ++i) {
        double v = static_cast<double>(xp[i]) * s_inv + static_cast<double>(zp);
        int64_t qi = rnd_nearest_even(v);
        qi = sat<int64_t>(qi, qmin, qmax);
        op[i] = static_cast<int8_t>(qi);
    }
    return out;
}

// dequant: x_hat = (x_q - zp) * s
torch::Tensor dequantize_per_tensor_asym(torch::Tensor xq, double scale, int64_t zp) {
    TORCH_CHECK(xq.is_contiguous(), "xq must be contiguous");
    auto x_i = xq.contiguous().to(torch::kInt8);
    auto out = torch::empty(x_i.sizes(), torch::kFloat32);

    const int8_t* ip = x_i.data_ptr<int8_t>();
    float* op = out.data_ptr<float>();
    auto N = x_i.numel();

    for (int64_t i = 0; i < N; ++i) {
        int32_t qi = static_cast<int32_t>(ip[i]);
        op[i] = static_cast<float>((qi - zp) * scale);
    }
    return out;
}

// per-channel symmetric (weights): x_q[c, ...] = clamp(round(x/s[c]), qmin, qmax), zp=0
torch::Tensor quantize_per_channel_sym(torch::Tensor w, torch::Tensor scale, int64_t ch_axis, int64_t qmin, int64_t qmax) {
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(scale.dim() == 1, "scale must be 1D (per-channel)");
    TORCH_CHECK(ch_axis >= 0 && ch_axis < w.dim(), "ch_axis out of range");
    TORCH_CHECK(scale.size(0) == w.size(ch_axis), "scale length mismatch with channel size");

    auto w_f = w.contiguous().to(torch::kFloat32);
    auto out = torch::empty_like(w_f, torch::kInt8);

    const float* wp = w_f.data_ptr<float>();
    int8_t* op = out.data_ptr<int8_t>();

    auto sizes = w_f.sizes();
    int64_t C = sizes[ch_axis];

    // 展开：把 channel 维作为最外圈
    // 用线性索引访问：预计算每个通道的步长
    std::vector<int64_t> stride(w_f.dim(), 0), shape(w_f.dim(), 0);
    for (int i = 0; i < w_f.dim(); ++i) {
        stride[i] = w_f.stride(i);
        shape[i]  = sizes[i];
    }

    auto sp = scale.to(torch::kFloat64).contiguous().data_ptr<double>();

    // 朴素循环（可改为并行/AVX2）
    // 遍历所有元素，定位其通道索引
    int64_t total = w_f.numel();
    int64_t nd = w_f.dim();

    // 预计算：对每个位置如何取 channel 索引，避免反复 div/mod
    // 这里用计数器法
    std::vector<int64_t> idx(nd, 0);

    for (int64_t linear = 0; linear < total; ++linear) {
        // 线性 index -> 多维 index（按行主序）
        int64_t tmp = linear;
        for (int d = nd - 1; d >= 0; --d) {
            idx[d] = tmp % shape[d];
            tmp   /= shape[d];
        }
        int64_t c = idx[ch_axis];
        double s_inv = 1.0 / sp[c];

        double v = static_cast<double>(wp[linear]) * s_inv;
        int64_t qi = rnd_nearest_even(v);
        qi = sat<int64_t>(qi, qmin, qmax);
        op[linear] = static_cast<int8_t>(qi);
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant_per_tensor_asym", &quantize_per_tensor_asym, "INT8 per-tensor asymmetric quant (CPU)");
    m.def("dequant_per_tensor_asym", &dequantize_per_tensor_asym, "dequant (CPU)");
    m.def("quant_per_channel_sym",  &quantize_per_channel_sym,  "INT8 per-channel symmetric quant (CPU)");
}
