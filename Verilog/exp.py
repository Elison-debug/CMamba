import numpy as np
import matplotlib.pyplot as plt

# --- 1. 参数定义 (Parameters) ---
# 定点数格式：4位整数 + 12位小数 = 16位总位宽
FRAC_BITS = 12
INT_BITS = 16 - FRAC_BITS
Q_FORMAT = 2**FRAC_BITS  # 缩放因子，即 2^12 = 4096

# 近似参数
SEGMENTS = 16  # 分段数量
TOLERANCE = 0.01  # 误差容忍度

# --- 2. 核心函数 (Core Functions) ---

def float_to_q(val, frac_bits):
    """将浮点数转换为定点数 (Q格式)"""
    return int(round(val * (2**frac_bits)))

def q_to_float(q_val, frac_bits):
    """将定点数转换回浮点数"""
    return q_val / (2**frac_bits)

def calculate_coefficients(segments, frac_bits):
    """
    计算分段线性近似的 a 和 b 系数，并以定点数格式返回。
    """
    a_table = np.zeros(segments, dtype=np.int16)
    b_table = np.zeros(segments, dtype=np.int16)
    
    # 将 v 的浮点数范围 [0, 1) 分成 segments 段
    v_segments = np.linspace(0, 1.0, segments + 1)
    
    for i in range(segments):
        v_start_float = v_segments[i]
        v_end_float = v_segments[i+1]
        
        # 计算每个分段的函数值
        y_start_float = 2**v_start_float
        y_end_float = 2**v_end_float
        
        # 计算斜率 (a) 和截距 (b)
        a_float = (y_end_float - y_start_float) / (v_end_float - v_start_float)
        b_float = y_start_float - a_float * v_start_float
        
        # 转换为定点数并存储
        a_table[i] = float_to_q(a_float, frac_bits)
        b_table[i] = float_to_q(b_float, frac_bits)
        
    return a_table, b_table

def evaluate_and_plot(a_table, b_table, segments, frac_bits, tolerance):
    """
    评估近似精度并绘制结果。
    """
    v_float = np.linspace(0, 1.0, 1000)
    
    # 理论值：2^v
    y_theoretical = 2**v_float
    
    # 硬件近似值
    y_hardware = np.zeros_like(v_float)
    
    mismatched_points = 0
    total_points = len(v_float)
    
    for i, v_val in enumerate(v_float):
        # --- FIX: Ensure the index is not out of bounds
        seg_idx = min(int(v_val * segments), segments - 1)
        
        # 从定点数中获取浮点数 a 和 b
        a_float = q_to_float(a_table[seg_idx], frac_bits)
        b_float = q_to_float(b_table[seg_idx], frac_bits)
        
        # 计算线性近似值
        y_approx = a_float * v_val + b_float
        y_hardware[i] = y_approx
        
        # 评估精度
        if abs(y_approx - y_theoretical[i]) > tolerance:
            mismatched_points += 1
            
    # 计算均方根误差 (Root Mean Square Error, RMSE)
    rmse = np.sqrt(np.mean((y_hardware - y_theoretical)**2))
    
    # 打印结果
    print("--- 近似精度评估 ---")
    print(f"定点数格式: Q{INT_BITS}.{FRAC_BITS}")
    print(f"分段数量: {segments}")
    print(f"均方根误差 (RMSE): {rmse:.6f}")
    print(f"失配点数: {mismatched_points} / {total_points} ({(mismatched_points / total_points) * 100:.2f}%)")
    print("-" * 20)

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(v_float, y_theoretical, label='Theoretical: 2^v')
    plt.plot(v_float, y_hardware, '--', label='Hardware Approximation')
    
    # 绘制分段点
    v_segments = np.linspace(0, 1.0, segments + 1)
    y_theoretical_at_segments = 2**v_segments
    plt.scatter(v_segments, y_theoretical_at_segments, color='red', label='Segment Endpoints')

    plt.title(f'2^v Approximation in Q{INT_BITS}.{FRAC_BITS} with {segments} Segments')
    plt.xlabel('v (Floating Point)')
    plt.ylabel('2^v (Floating Point)')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 3. 运行脚本 ---
if __name__ == '__main__':
    # 计算系数
    a_table_q, b_table_q = calculate_coefficients(SEGMENTS, FRAC_BITS)
    
    print(f"--- 重新计算的Q{INT_BITS}.{FRAC_BITS}定点数系数 ---")
    for i in range(SEGMENTS):
        print(f"a_table[{i}] = 16'd{a_table_q[i]:<5}; b_table[{i}] = 16'd{b_table_q[i]};")

    # 评估和画图
    evaluate_and_plot(a_table_q, b_table_q, SEGMENTS, FRAC_BITS, TOLERANCE)