//---------------------------------------------------------------
// Module : nonlinear_tile
// Function : Shared 16x16 exp_sp_cell array (Softplus / Exp)
// Author : Shengjie Chen
// Description :
//   - 16x16 parallel nonlinear array for Softplus / Exp
//   - Mode: 0=Softplus, 1=Exp
//   - Softplus: operates on vector mid_res_vec[16], but can be tiled.
//   - Exp: operates on matrix mid_res_mat[16x16]
//   - Unified output logic for consistent latency.
//   - done_tile is registered to be one cycle later than valid_out.
//---------------------------------------------------------------
module nonlinear_tile #(
    parameter DATA_WIDTH = 16,
    parameter TILE_SIZE  = 16,
    parameter FRAC_BITS  = 12
)(
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic                   valid_in,
    input  logic                   mode, // 0=Softplus, 1=Exp
    
    // Inputs
    input  logic signed [DATA_WIDTH-1:0] mid_res_vec [TILE_SIZE-1:0],
    input  logic signed [DATA_WIDTH-1:0] mid_res_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],

    // Outputs
    output logic signed [DATA_WIDTH-1:0] y_vec [TILE_SIZE-1:0],
    output logic signed [DATA_WIDTH-1:0] y_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],
    output logic                         valid_out, 
    output logic                         done_tile
);
    // ============================================================
    // Internal Signals
    // ============================================================
    genvar i, j;
    logic signed [DATA_WIDTH-1:0] exp_out [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic                        cell_valid_out [TILE_SIZE-1:0][TILE_SIZE-1:0];
    
    logic signed [DATA_WIDTH-1:0] y_vec_reg [TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] y_mat_reg [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic                        valid_out_reg;

    // MODIFICATION: 引入done_tile的内部寄存器
    logic done_tile_reg;

    // ============================================================
    // Parallel exp_sp_cell Array
    // ============================================================
    generate
        for (i = 0; i < TILE_SIZE; i++) begin : ROW
            for (j = 0; j < TILE_SIZE; j++) begin : COL
                exp_sp_cell #(
                    .DATA_WIDTH (DATA_WIDTH),
                    .FRAC_BITS  (FRAC_BITS)
                ) U_EXP_CELL (
                    .clk       (clk),
                    .rst_n     (rst_n),
                    .valid_in  (valid_in),
                    .x_in      (mode ? mid_res_mat[i][j] : mid_res_vec[i]),
                    .mode      (mode),
                    .y_out     (exp_out[i][j]),
                    .valid_out (cell_valid_out[i][j])
                );
            end
        end
    endgenerate

    // ============================================================
    // Unified Output Collection (1-cycle latency)
    // ============================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int k = 0; k < TILE_SIZE; k++) begin
                y_vec_reg[k] <= '0;
                for (int l = 0; l < TILE_SIZE; l++) begin
                    y_mat_reg[k][l] <= '0;
                end
            end
            valid_out_reg <= 1'b0;
        end else if (cell_valid_out[0][0]) begin
            if (mode) begin // Exp Mode
                for (int k = 0; k < TILE_SIZE; k++) begin
                    for (int l = 0; l < TILE_SIZE; l++) begin
                        y_mat_reg[k][l] <= exp_out[k][l];
                    end
                end
            end else begin // Softplus Mode
                for (int k = 0; k < TILE_SIZE; k++) begin
                    y_vec_reg[k] <= exp_out[k][0];
                end
            end
            valid_out_reg <= cell_valid_out[0][0]; // 所有单元的valid_out都是同步的
        end else begin
            valid_out_reg <= 1'b0;
        end
    end

    // ============================================================
    // Final Assignments and Control Logic
    // ============================================================
    assign y_vec = y_vec_reg;
    assign y_mat = y_mat_reg;
    assign valid_out = valid_out_reg;

    // MODIFICATION: done_tile 滞后 valid_out 一拍
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            done_tile_reg <= 1'b0;
        else
            done_tile_reg <= valid_out; // valid_out经过一拍后赋值给done_tile_reg
    end
    assign done_tile = done_tile_reg;

endmodule