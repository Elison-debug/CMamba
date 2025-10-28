//---------------------------------------------------------------
// Module: pipeline_4array_with_reduction (v2 - with done_tile / clear)
// Function:
//   在 pipeline_4array_top 后级添加 reduction_accumulator
//   用于在 MAC 模式 (3'b000) 下，将 array4 输出矩阵逐拍规约为向量并累加。
//   新增 done_tile → clear 信号联动：
//   每个 tile 计算完成 (done_tile=1) 时自动清零累加寄存器。
//---------------------------------------------------------------
module pipeline_4array_with_reduction #(
    parameter int TILE_SIZE  = 4,
    parameter int DATA_WIDTH = 16,
    parameter int ACC_WIDTH  = 32,
    parameter int FRAC_BITS  = 8
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [2:0] mode,
    input  logic valid_in,

    // --- A/B 输入 ---
    input  logic signed [DATA_WIDTH-1:0] A0_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],
    input  logic signed [DATA_WIDTH-1:0] A1_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],
    input  logic signed [DATA_WIDTH-1:0] A2_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],
    input  logic signed [DATA_WIDTH-1:0] A3_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],

    input  logic signed [DATA_WIDTH-1:0] B0_vec [TILE_SIZE-1:0],
    input  logic signed [DATA_WIDTH-1:0] B1_vec [TILE_SIZE-1:0],
    input  logic signed [DATA_WIDTH-1:0] B2_vec [TILE_SIZE-1:0],
    input  logic signed [DATA_WIDTH-1:0] B3_vec [TILE_SIZE-1:0],

    // --- 最终输出 ---
    output logic signed [ACC_WIDTH-1:0] reduced_vec [TILE_SIZE-1:0],
    output logic                        valid_reduced
);

    // ==========================================================
    // 1. 连接 pipeline_4array_top
    // ==========================================================
    logic signed [ACC_WIDTH-1:0] result_out_0 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [ACC_WIDTH-1:0] result_out_1 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [ACC_WIDTH-1:0] result_out_2 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [ACC_WIDTH-1:0] result_out_3 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic                        valid_array4;
    logic                        done_tile;   // <<< 新增：来自 pipeline_4array_top

    pipeline_4array_top #(
        .TILE_SIZE (TILE_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH),
        .FRAC_BITS (FRAC_BITS)
    ) u_pipeline_4array_top (
        .clk(clk),
        .rst_n(rst_n),
        .mode(mode),
        .valid_in(valid_in),
        .valid_out(valid_array4),
        .done_tile(done_tile),   // <<< 新增：tile 完成脉冲输出

        .A0_mat(A0_mat), .A1_mat(A1_mat), .A2_mat(A2_mat), .A3_mat(A3_mat),
        .B0_vec(B0_vec), .B1_vec(B1_vec), .B2_vec(B2_vec), .B3_vec(B3_vec),

        .result_out_0(result_out_0),
        .result_out_1(result_out_1),
        .result_out_2(result_out_2),
        .result_out_3(result_out_3)
    );

    // ==========================================================
    // 2. 连接 reduction_accumulator (仅使用 array4 的结果)
    // ==========================================================
    reduction_accumulator #(
        .TILE_SIZE (TILE_SIZE),
        .ACC_WIDTH (ACC_WIDTH)
    ) u_reduction_accumulator (
        .clk(clk),
        .rst_n(rst_n),
        .mode(mode),
        .valid_in(valid_array4),   // array4 的 valid_out
        .clear(done_tile),         // <<< 新增：每个 tile 完成时清零
        .mat_in(result_out_3),     // array4 的 4x4 矩阵结果
        .vec_out(reduced_vec),     // 输出 4x1 向量
        .valid_out(valid_reduced)
    );

endmodule
