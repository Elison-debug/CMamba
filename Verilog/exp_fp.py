import numpy as np
import matplotlib.pyplot as plt

# --- 1. 参数定义 (Parameters) ---
# 定点数格式
DATA_WIDTH = 16
FRAC_BITS = 12
SEGMENTS = 16
INT_BITS = DATA_WIDTH - FRAC_BITS
Q_FORMAT = 2**FRAC_BITS  # 缩放因子，即 2^12 = 4096

# 近似系数 (Q4.12 格式)
A_TABLE = np.array([
    2902, 3030, 3164, 3304, 3451, 3603, 3763, 3929,
    4103, 4285, 4475, 4673, 4880, 5096, 5321, 5557
], dtype=np.int16)

B_TABLE = np.array([
    4096, 4088, 4071, 4045, 4008, 3961, 3901, 3828,
    3741, 3639, 3520, 3384, 3229, 3053, 2856, 2635
], dtype=np.int16)

# 其他常量 (Q4.12 格式)
LOG2E = int(round(1.442695 * Q_FORMAT)) # 1.442695 * 4096 = 5917.48

# --- 2. 核心函数：定点数运算模拟 (Hardware Simulation Functions) ---

def exp_core_calc(x_in_q412):
    """
    模拟 exp_sp_cell 的核心指数计算部分。
    """
    z_full = np.int64(x_in_q412) * np.int64(LOG2E)
    z = np.int16(z_full >> FRAC_BITS)
    
    u = np.int8(z >> FRAC_BITS)
    v = np.int16(z & (Q_FORMAT - 1))
    
    seg_idx = np.int8(v >> (FRAC_BITS - np.log2(SEGMENTS).astype(int)))
    
    a_mult_v = np.int64(A_TABLE[seg_idx]) * np.int64(v)
    lin_approx = np.int16((a_mult_v >> FRAC_BITS) + np.int64(B_TABLE[seg_idx]))
    
    exp_val = 0
    if u >= 0:
        exp_full_u = np.int64(lin_approx) << np.int64(u)
    else:
        exp_full_u = np.int64(lin_approx) >> np.int64(abs(u))

    if exp_full_u > 32767:
        exp_val = 32767
    elif exp_full_u < -32768:
        exp_val = -32768
    else:
        exp_val = np.int16(exp_full_u)

    return exp_val

def exp_sp_cell_model(x_in_q412, mode):
    """
    模拟 exp_sp_cell 模块的定点数计算。
    """
    if mode == 1: # Exp 模式
        return exp_core_calc(x_in_q412)
    else: # Softplus 模式
        if x_in_q412 < 0:
            return exp_core_calc(x_in_q412)
        else:
            # MODIFICATION: 当x>=0时，显式地计算 exp(-x)
            x_neg_q412 = -x_in_q412
            exp_neg_val = exp_core_calc(x_neg_q412)
            
            softplus_sum = np.int32(x_in_q412) + np.int32(exp_neg_val)
            
            if softplus_sum > 32767:
                return 32767
            elif softplus_sum < -32768:
                return -32768
            else:
                return np.int16(softplus_sum)

def evaluate_fixed_point_model(test_values_q412, mode_val):
    """
    评估定点数模型的精度
    """
    y_hw = np.array([exp_sp_cell_model(val, mode_val) for val in test_values_q412])
    return y_hw

# --- 3. 运行脚本 ---
if __name__ == '__main__':
    x_float = np.linspace(-1.0, 1.0, 1000)
    x_q412 = np.array([int(round(val * Q_FORMAT)) for val in x_float], dtype=np.int16)

    y_hw_exp = evaluate_fixed_point_model(x_q412, 1)
    y_hw_exp_float = y_hw_exp / Q_FORMAT
    y_th_exp = np.exp(x_float)
    mismatch_exp = np.sum(np.abs(y_hw_exp_float - y_th_exp) > 0.1)
    
    print("--- EXP 模式定点数模拟结果 ---")
    print(f"失配点数: {mismatch_exp} / {len(x_q412)} ({(mismatch_exp / len(x_q412)) * 100:.2f}%)")
    
    y_hw_sp = evaluate_fixed_point_model(x_q412, 0)
    y_hw_sp_float = y_hw_sp / Q_FORMAT
    y_th_sp = np.log(1 + np.exp(x_float))
    mismatch_sp = np.sum(np.abs(y_hw_sp_float - y_th_sp) > 0.1)
    
    print("\n--- SOFTPLUS 模式定点数模拟结果 ---")
    print(f"失配点数: {mismatch_sp} / {len(x_q412)} ({(mismatch_sp / len(x_q412)) * 100:.2f}%)")

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_float, y_th_exp, label='Theoretical Exp(x)')
    plt.plot(x_float, y_hw_exp_float, '--', label='Fixed-Point Exp(x)')
    plt.title('Exp(x) Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x_float, y_th_sp, label='Theoretical Softplus(x)')
    plt.plot(x_float, y_hw_sp_float, '--', label='Fixed-Point Softplus(x)')
    plt.title('Softplus(x) Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    plt.show()