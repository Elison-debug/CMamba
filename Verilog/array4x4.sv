//---------------------------------------------------------------
// Module: array4x4 (or arrayNxN)
// Function: Parameterized PE Array (no mode mapping inside)
// Author: Shengjie Chen
//---------------------------------------------------------------
module array4x4 #(
    parameter int TILE_SIZE = 4,
    parameter int DATA_WIDTH = 16,
    parameter int ACC_WIDTH  = 32,
    parameter int FRAC_BITS  = 8
)(
    input  logic                              clk,
    input  logic                              rst_n,
    
    // 直接传入 PE 可识别的模式 (2-bit)
    input  logic [1:0]                        pe_mode,    

    input  logic signed [DATA_WIDTH-1:0] a_in [TILE_SIZE-1:0][TILE_SIZE-1:0],
    input  logic signed [DATA_WIDTH-1:0] b_in [TILE_SIZE-1:0][TILE_SIZE-1:0],
    input  logic signed [ACC_WIDTH-1:0] acc_in[TILE_SIZE-1:0][TILE_SIZE-1:0],
    input  logic valid_in,
    output logic valid_out,
    output logic signed [ACC_WIDTH-1:0] result_out[TILE_SIZE-1:0][TILE_SIZE-1:0]
);

    // 内部信号：用于从所有 PE 捕获 valid_out
    logic pe_valid_out [TILE_SIZE-1:0][TILE_SIZE-1:0];

    genvar i, j;
    generate
        for (i = 0; i < TILE_SIZE; i++) begin : ROW
            for (j = 0; j < TILE_SIZE; j++) begin : COL
                
                pe_unit_pipe #(
                    .DATA_WIDTH (DATA_WIDTH),
                    .ACC_WIDTH  (ACC_WIDTH),
                    .FRAC_BITS  (FRAC_BITS)
                ) u_pe (
                    .clk        (clk),
                    .rst_n      (rst_n),
                    
                    // --- valid 信号连接 ---
                    .valid_in   (valid_in),             // 向下传递 valid_in
                    .valid_out  (pe_valid_out[i][j]),   // 捕获 PE 的 valid_out
                    
                    .mode       (pe_mode),
                    
                    // --- [重要修正] ---
                    // 之前为 a_in[i] 和 b_in[j]，已修正为元素级连接
                    .a_in       (a_in[i][j]),
                    .b_in       (b_in[i][j]),
                    
                    .acc_in     (acc_in[i][j]),
                    .result_out (result_out[i][j])
                );
            end
        end
    endgenerate

    // --- 驱动模块的 valid_out ---
    // 所有 PE 具有相同的 1 拍延迟，因此它们所有的 valid_out 信号都是同步的。
    // 我们可以安全地选择任意一个 PE (例如 [3][3]) 的 valid_out
    // 作为整个 array4x4 模块的 valid_out 信号。
    assign valid_out = pe_valid_out[TILE_SIZE-1][TILE_SIZE-1];

endmodule
