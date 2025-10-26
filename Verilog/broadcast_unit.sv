//---------------------------------------------------------------
// Module: broadcast_unit (Unified)
// Author: Shengjie Chen
//---------------------------------------------------------------
module broadcast_unit #(
    parameter int TILE_SIZE  = 4,
    parameter int DATA_WIDTH = 16
)(
    input  logic [2:0] mode,

    // --- 单向量输入（必要时广播） ---
    input  logic signed [TILE_SIZE-1:0][DATA_WIDTH-1:0] A_vec,
    input  logic signed [TILE_SIZE-1:0][DATA_WIDTH-1:0] B_vec,

    // --- 通用矩阵输入（既可是真矩阵，也可多向量并行） ---
    input  logic signed [TILE_SIZE-1:0][TILE_SIZE-1:0][DATA_WIDTH-1:0] A_mat,
    input  logic signed [TILE_SIZE-1:0][TILE_SIZE-1:0][DATA_WIDTH-1:0] B_mat,

    // --- 输出：阵列统一 tile ---
    output logic signed [TILE_SIZE-1:0][TILE_SIZE-1:0][DATA_WIDTH-1:0] A_tile,
    output logic signed [TILE_SIZE-1:0][TILE_SIZE-1:0][DATA_WIDTH-1:0] B_tile
);

    integer i, j;
    always_comb begin
        A_tile = '0;
        B_tile = '0;

        unique case (mode)

            // MAC: A=矩阵, B=向量 → B列广播
            3'b000: begin
                A_tile = A_mat;
                for (i = 0; i < TILE_SIZE; i++)
                    for (j = 0; j < TILE_SIZE; j++)
                        B_tile[i][j] = B_vec[j];
            end

            // EWM-Matrix: A=矩阵, B=向量 → B行广播
            3'b001: begin
                A_tile = A_mat;
                for (i = 0; i < TILE_SIZE; i++)
                    for (j = 0; j < TILE_SIZE; j++)
                        B_tile[i][j] = B_vec[i];
            end

            // EWM-Vector / EWA-Vector: 直接使用多向量矩阵
            3'b010, 3'b100: begin
                A_tile = A_mat; // 实际为多向量批矩阵
                B_tile = B_mat;
            end

            // EWM-Outer: A,B=向量 → A行 + B列广播
            3'b011: begin
                for (i = 0; i < TILE_SIZE; i++)
                    for (j = 0; j < TILE_SIZE; j++) begin
                        A_tile[i][j] = A_vec[i];
                        B_tile[i][j] = B_vec[j];
                    end
            end

            // EWA-Matrix / EWM-Matrix2: 矩阵直接送阵列
            3'b101, 3'b110: begin
                A_tile = A_mat;
                B_tile = B_mat;
            end

            default: begin
                A_tile = '0;
                B_tile = '0;
            end
        endcase
    end
endmodule
