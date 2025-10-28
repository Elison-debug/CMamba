//---------------------------------------------------------------
// Module: reduction_accumulator
// Function:
//   对 array4 输出的 4×4 矩阵做实时树形规约 (4→2→1)
//   并在时间上持续累加 (在线累加)
//---------------------------------------------------------------
module reduction_accumulator #(
    parameter int TILE_SIZE  = 4,
    parameter int ACC_WIDTH  = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  logic [2:0] mode,
    input  logic signed [ACC_WIDTH-1:0] mat_in [TILE_SIZE-1:0][TILE_SIZE-1:0],

    output logic signed [ACC_WIDTH-1:0] vec_out [TILE_SIZE-1:0],
    output logic valid_out
);

    //-----------------------------------------------------------
    // Mode 判定：仅在 MAC 模式下使能
    //-----------------------------------------------------------
    logic is_mac_mode;
    assign is_mac_mode = (mode == 3'b000);

    //-----------------------------------------------------------
    // Level 1: 4→2 并行加法
    //-----------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] sum_l1 [TILE_SIZE-1:0][1:0];
    always_comb begin
        for (int j = 0; j < TILE_SIZE; j++) begin
            sum_l1[j][0] = mat_in[0][j] + mat_in[1][j];
            sum_l1[j][1] = mat_in[2][j] + mat_in[3][j];
        end
    end

    //-----------------------------------------------------------
    // Level 2: 2→1 加法 (列规约完成)
    //-----------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] col_sum [TILE_SIZE-1:0];
    always_comb begin
        for (int j = 0; j < TILE_SIZE; j++)
            col_sum[j] = sum_l1[j][0] + sum_l1[j][1];
    end

    //-----------------------------------------------------------
    // Level 3: 累加寄存器 (跨周期累加)
    //-----------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] acc_vec [TILE_SIZE-1:0];
    logic valid_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int j = 0; j < TILE_SIZE; j++)
                acc_vec[j] <= '0;
            valid_reg <= 1'b0;
        end
        else if (is_mac_mode && valid_in) begin
            for (int j = 0; j < TILE_SIZE; j++)
                acc_vec[j] <= acc_vec[j] + col_sum[j];
            valid_reg <= 1'b1;
        end
        else begin
            valid_reg <= 1'b0;
        end
    end

    assign vec_out  = acc_vec;
    assign valid_out = valid_reg;

endmodule
