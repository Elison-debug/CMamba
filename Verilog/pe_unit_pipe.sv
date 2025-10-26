//---------------------------------------------------------------
// Module: pe_unit_pipe
// Function: Pipelined PE with shared adder and correct sign extension
//
//   支持的三种模式：
//     00 - MAC : result = (a * b) + acc_in
//     01 - EWM : result = (a * b)
//     10 - EWA : result = (a + b) << FRAC_BITS   (并符号扩展)
//
//   特点：
//     - 无自反馈（不在PE内累积）
//     - 跨阵列传递 result_out → 下一级阵列 acc_in
//     - 1拍pipeline
//     - MAC/EWM/EWA 共用一个加法器
//---------------------------------------------------------------
module pe_unit_pipe #(
    parameter DATA_WIDTH = 16,  // 输入数据位宽 (Q8.8)
    parameter ACC_WIDTH  = 32,  // 输出数据位宽 (Q16.16)
    parameter FRAC_BITS  = 8    // 小数位数
)(
    input  logic                       clk,
    input  logic                       rst_n,

    input  logic valid_in,      // 输入有效信号 (用作 CE)
    output logic valid_out,     // 输出有效信号 (延迟1拍)

    input  logic [1:0]                 mode,        // 00:MAC, 01:EWM, 10:EWA

    input  logic signed [DATA_WIDTH-1:0] a_in,
    input  logic signed [DATA_WIDTH-1:0] b_in,
    input  logic signed [ACC_WIDTH-1:0]  acc_in,     // 上一级阵列的部分和

    output logic signed [ACC_WIDTH-1:0]  result_out  // 输出给下一级阵列
);

    // ===========================================================
    // 内部信号定义
    // ===========================================================
    localparam SHIFTED_WIDTH = (DATA_WIDTH + 1) + FRAC_BITS; // e.g. 17 + 8 = 25

    logic signed [ACC_WIDTH-1:0] mult_full;      // a*b
    logic signed [DATA_WIDTH:0]  ewa_sum;        // a+b
    logic signed [SHIFTED_WIDTH-1:0] ewa_shifted;// (a+b)<<FRAC_BITS
    logic signed [ACC_WIDTH-1:0] ewa_sum_aligned;// 对齐到 Q16.16
    logic signed [ACC_WIDTH-1:0] opA, opB;       // adder输入
    logic signed [ACC_WIDTH-1:0] add_result;     // adder输出
    logic signed [ACC_WIDTH-1:0] result_reg;     // pipeline寄存器
    logic valid_reg; // 与 result_reg 同步的 valid 寄存器
    // ===========================================================
    // 1. 乘法和加法准备
    // ===========================================================
    assign mult_full = $signed(a_in) * $signed(b_in);   // Q8.8 * Q8.8 = Q16.16
    assign ewa_sum   = a_in + b_in;                     // Q8.8 + Q8.8 = Q9.8
    assign ewa_shifted = $signed(ewa_sum) <<< FRAC_BITS;// Q9.8→Q9.16 (25bit)
    assign ewa_sum_aligned = 
        {{(ACC_WIDTH - SHIFTED_WIDTH){ewa_shifted[SHIFTED_WIDTH-1]}}, ewa_shifted}; // 符号扩展到32位

    // ===========================================================
    // 2. 共用加法器输入选择
    // ===========================================================
    always_comb begin
        unique case (mode)
            2'b00: begin
                // MAC: result = a*b + acc_in
                opA = mult_full;
                opB = acc_in;
            end
            2'b01: begin
                // EWM: result = a*b
                opA = mult_full;
                opB = '0;
            end
            2'b10: begin
                // EWA: result = (a+b)<<FRAC_BITS
                opA = ewa_sum_aligned;
                opB = '0;
            end
            default: begin
                opA = '0;
                opB = '0;
            end
        endcase
    end

    assign add_result = opA + opB;

    // ===========================================================
    // 3. 一拍pipeline寄存器 (已修改)
    // ===========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_reg <= '0;
            valid_reg  <= 1'b0;
        end else begin
            // valid_in 作为时钟使能 (CE)
            // 只有当输入有效时，才锁存新的计算结果
            // 这样可以节省大量功耗
            if (valid_in) begin
                result_reg <= add_result;
            end
            // else: result_reg 保持不变
            
            // valid 信号链总是随流水线传递
            // valid_out 将在 1 拍后与 result_out 同步变为有效
            valid_reg <= valid_in;
        end
    end

    // ===========================================================
    // 4. 输出
    // ===========================================================
    assign result_out = result_reg;
    assign valid_out  = valid_reg;
    
endmodule
