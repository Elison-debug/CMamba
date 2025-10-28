//---------------------------------------------------------------
// Module: pipeline_4array_top (v16 - 最终版，广播逻辑明确)
//   Mode Encoding (3-bit):
//   -----------------------------------------------------------
//   | mode | Operation Description          | Formula                           | Output Shape |
//   |------|--------------------------------|-----------------------------------|---------------|
//   | 000  | MAC (Matrix × Vector)          | RAW=Wx_pj⊙x_t; Δ_t=W_Δ⊙Δ_raw    | Vector (d_state×1) |
//   | 001  | EWM-Matrix (Matrix × Vector)   | ΔA = A ⊙ spΔ_t; ΔB_x=spΔ_t ⊙ B_x| Matrix (d_inner×d_state) |
//   | 010  | EWM-Vector (Vector × Vector)   | D_x = D ⊙ x_t                    | Vector (d_inner×1) |
//   | 011  | EWM-Outer (Outer Product)      | B_x = x_t ⊗ B_raw; C_h=ht⊗C_raw | Matrix (d_inner×d_state) |
//   | 100  | EWA-Vector (Vector + Vector)   | y = C_h + D_x; Δ_t_b=Δ_t+dt_bias  | Vector (d_inner×1) |
//   | 101  | EWA-Matrix (Matrix + Matrix)   | h_t = A_ht-1 + ΔB_x               | Matrix (d_inner×d_state) |
//   | 110  | EWM-Matrix2 (Matrix × Matrix)  | A_ht-1 = EXP_ΔA ⊙ h_t-1          | Matrix (d_inner×d_state) |
//   -----------------------------------------------------------
// Description:
//  - 架构: A=Mat, B=Vec (I/O: 4xA_mat, 4xB_vec)。
//  - A 输入 (A_mat) 直接透传。
//  - [修改] B 广播逻辑 (always_comb) 已被重构，
//    显式地将 3'b101 和 3'b110 映射到“行广播”，
//    以支持 4 拍串行化的 Mat-Mat 操作。
//---------------------------------------------------------------
module pipeline_4array_top #(
    parameter int TILE_SIZE  = 4,
    parameter int DATA_WIDTH = 16,
    parameter int ACC_WIDTH  = 32,
    parameter int FRAC_BITS  = 8
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [2:0] mode,
    input  logic valid_in,
    output logic valid_out,

    // --- 4 组并行的 A_mat 输入 (C-style) ---
    input  logic signed [DATA_WIDTH-1:0] A0_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],
    input  logic signed [DATA_WIDTH-1:0] A1_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],
    input  logic signed [DATA_WIDTH-1:0] A2_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],
    input  logic signed [DATA_WIDTH-1:0] A3_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],

    // --- 4 组并行的 B_vec 输入 (C-style) ---
    input  logic signed [DATA_WIDTH-1:0] B0_vec [TILE_SIZE-1:0],
    input  logic signed [DATA_WIDTH-1:0] B1_vec [TILE_SIZE-1:0],
    input  logic signed [DATA_WIDTH-1:0] B2_vec [TILE_SIZE-1:0],
    input  logic signed [DATA_WIDTH-1:0] B3_vec [TILE_SIZE-1:0],

    // --- 4 组独立的结果输出 (C-style) ---
    output logic signed [ACC_WIDTH-1:0] result_out_0 [TILE_SIZE-1:0][TILE_SIZE-1:0],
    output logic signed [ACC_WIDTH-1:0] result_out_1 [TILE_SIZE-1:0][TILE_SIZE-1:0],
    output logic signed [ACC_WIDTH-1:0] result_out_2 [TILE_SIZE-1:0][TILE_SIZE-1:0],
    output logic signed [ACC_WIDTH-1:0] result_out_3 [TILE_SIZE-1:0][TILE_SIZE-1:0]
);

    // ==========================================================
    // Mode mapping (for PEs)
    // ==========================================================
    logic [1:0] pe_mode;
    always_comb begin
        case (mode)
            3'b000: pe_mode = 2'b00; // MAC
            3'b001, 3'b010, 3'b011, 3'b110: pe_mode = 2'b01; // EWM
            3'b100, 3'b101: pe_mode = 2'b10; // EWA
            default: pe_mode = 2'b00;
        endcase
    end

    // ==========================================================
    // Internal Signals
    // ==========================================================
    logic v1, v2, v3, v4;
    logic is_mac_mode;
    //integer i, j; // 用于 B 广播循环

    logic signed [ACC_WIDTH-1:0] acc_1_out [TILE_SIZE-1:0][TILE_SIZE-1:0], 
                                acc_2_out [TILE_SIZE-1:0][TILE_SIZE-1:0],
                                acc_3_out [TILE_SIZE-1:0][TILE_SIZE-1:0],
                                acc_4_out [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [ACC_WIDTH-1:0] acc_in_1 [TILE_SIZE-1:0][TILE_SIZE-1:0],
                                acc_in_2 [TILE_SIZE-1:0][TILE_SIZE-1:0],
                                acc_in_3 [TILE_SIZE-1:0][TILE_SIZE-1:0],
                                acc_in_4 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic v_in_1, v_in_2, v_in_3, v_in_4;
    
    // --- 广播 B Tile 信号 ---
    logic signed [DATA_WIDTH-1:0] B0_tile [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] B1_tile [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] B2_tile [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] B3_tile [TILE_SIZE-1:0][TILE_SIZE-1:0];
    
    assign is_mac_mode = (mode == 3'b000);

    // ==========================================================
    // 1. B_tile 广播逻辑 (B_vec -> B_tile)
    // ==========================================================
    always_comb begin
        integer i, j;

        // 初始化
        // 初始化移除：由下方 case 分支全覆盖赋值
        for (i = 0; i < TILE_SIZE; i++) begin
            for (j = 0; j < TILE_SIZE; j++) begin
                B0_tile[i][j] = '0;
                B1_tile[i][j] = '0;
                B2_tile[i][j] = '0;
                B3_tile[i][j] = '0;
            end
        end
        unique case (mode)
            // 模式 1: 列广播 (Column Broadcast)
            // 用于: MAC (Mat*Vec) 和 Outer Product (Vec*Vec)
            3'b000, 3'b011: begin
                for (i = 0; i < TILE_SIZE; i++) begin
                    for (j = 0; j < TILE_SIZE; j++) begin
                        B0_tile[i][j] = B0_vec[j]; // 列广播
                        B1_tile[i][j] = B1_vec[j];
                        B2_tile[i][j] = B2_vec[j];
                        B3_tile[i][j] = B3_vec[j];
                    end
                end
            end

            // 模式 2: 行广播 (Row Broadcast)
            // 用于: EWM/EWA (Mat ⊙ Vec)
            // 并且用于: 3'b101, 3'b110 (串行化的 Mat + Mat)，一次只有一列有效，没充分利用阵列
            3'b001, 3'b010, 3'b100, 3'b101, 3'b110: begin
                for (i = 0; i < TILE_SIZE; i++) begin
                    for (j = 0; j < TILE_SIZE; j++) begin
                        B0_tile[i][j] = B0_vec[i]; // 行广播
                        B1_tile[i][j] = B1_vec[i];
                        B2_tile[i][j] = B2_vec[i];
                        B3_tile[i][j] = B3_vec[i];
                    end
                end
            end

            // 默认情况 (安全起见，设为行广播)
            default: begin
                for (i = 0; i < TILE_SIZE; i++) begin
                    for (j = 0; j < TILE_SIZE; j++) begin
                        B0_tile[i][j] = B0_vec[i]; // 行广播
                        B1_tile[i][j] = B1_vec[i];
                        B2_tile[i][j] = B2_vec[i];
                        B3_tile[i][j] = B3_vec[i];
                    end
                end
            end
        endcase
    end

    // ==========================================================
    // 2. MUX Logic for Valid & Acc Path (流水线控制)
    // ==========================================================
    assign v_in_1   = valid_in;
    assign v_in_2   = is_mac_mode ? v1        : valid_in;
    assign v_in_3   = is_mac_mode ? v2        : valid_in;
    assign v_in_4   = is_mac_mode ? v3        : valid_in;

    // 为避免 packed→unpacked 赋值错误，使用逐元素组合赋值
    always_comb begin
        integer i, j;
        for (i = 0; i < TILE_SIZE; i++) begin
            for (j = 0; j < TILE_SIZE; j++) begin
                acc_in_1[i][j] = '0;
                acc_in_2[i][j] = is_mac_mode ? acc_1_out[i][j] : '0;
                acc_in_3[i][j] = is_mac_mode ? acc_2_out[i][j] : '0;
                acc_in_4[i][j] = is_mac_mode ? acc_3_out[i][j] : '0;
            end
        end
    end

    // ==========================================================
    // 3. 4x Array instantiation
    // ==========================================================
    array4x4 #(
        .TILE_SIZE (TILE_SIZE), .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH), .FRAC_BITS(FRAC_BITS)
    ) u_array1 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(v_in_1), .valid_out(v1),
        .pe_mode(pe_mode),
        .a_in(A0_mat), .b_in(B0_tile), // A_mat 直接连接
        .acc_in(acc_in_1), .result_out(acc_1_out)
    );

    array4x4 #(
        .TILE_SIZE (TILE_SIZE), .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH), .FRAC_BITS(FRAC_BITS)
    ) u_array2 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(v_in_2), .valid_out(v2),
        .pe_mode(pe_mode),
        .a_in(A1_mat), .b_in(B1_tile), // A_mat 直接连接
        .acc_in(acc_in_2), .result_out(acc_2_out)
    );

    array4x4 #(
        .TILE_SIZE (TILE_SIZE), .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH), .FRAC_BITS(FRAC_BITS)
    ) u_array3 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(v_in_3), .valid_out(v3),
        .pe_mode(pe_mode),
        .a_in(A2_mat), .b_in(B2_tile), // A_mat 直接连接
        .acc_in(acc_in_3), .result_out(acc_3_out)
    );

    array4x4 #(
        .TILE_SIZE (TILE_SIZE), .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH), .FRAC_BITS(FRAC_BITS)
    ) u_array4 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(v_in_4), .valid_out(v4),
        .pe_mode(pe_mode),
        .a_in(A3_mat), .b_in(B3_tile), // A_mat 直接连接
        .acc_in(acc_in_4), .result_out(acc_4_out)
    );
    
    // ==========================================================
    // 4. Final Output Assignment
    // ==========================================================
    assign result_out_0 = acc_1_out;
    assign result_out_1 = acc_2_out;
    assign result_out_2 = acc_3_out;
    assign result_out_3 = acc_4_out;

    assign valid_out  = is_mac_mode ? v4 : v1;

endmodule