//---------------------------------------------------------------
// Module: recfg_array
// Function: 16×16 systolic array with multi-mode PE support
// Author: Shengjie Chen
// Description:
//   - Supports unified PE (MAC / EWM / EWA)
//   - Currently MAC mode logic retained
//   - Future EWM/EWA control logic can be added later
//---------------------------------------------------------------
module recfg_array #(
    parameter DATA_WIDTH = 16,
    parameter TILE_SIZE  = 16
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         valid_in,
    input  logic [1:0]                   mode,   // 00:MAC, 01:EWM, 10:EWA
    input  logic signed [DATA_WIDTH-1:0] a_in [TILE_SIZE-1:0][TILE_SIZE-1:0], // weights tile
    input  logic signed [DATA_WIDTH-1:0] b_in [TILE_SIZE-1:0],                 // input vector tile
    output logic signed [DATA_WIDTH-1:0] result_out_mac [TILE_SIZE-1:0],             // 1D output for MAC
    output logic signed [DATA_WIDTH-1:0] result_out     [TILE_SIZE-1:0][TILE_SIZE-1:0], // 2D output for EWM
    output logic                         valid_out,
    output logic                         done_tile
);

    // ============================================================
    // Internal interconnects
    // ============================================================
    logic signed [DATA_WIDTH-1:0] acc_chain [TILE_SIZE-1:0][TILE_SIZE:0];
    logic                         pe_valid  [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic                         done_tile_n;
    //logic signed [DATA_WIDTH-1:0] result_out_n [TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] result_out_mac_n [TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] result_out_ewm_n [TILE_SIZE-1:0][TILE_SIZE-1:0];
    // ★ NEW: 用于EWM模式的输出缓存（不影响MAC累加链）
    //logic signed [DATA_WIDTH-1:0] ewm_result [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] pe_result_wire [TILE_SIZE-1:0][TILE_SIZE-1:0];
    genvar i, j;

    // ============================================================
    // Initialize leftmost accumulations to zero (start of each row)
    // ============================================================
    generate
        for (i = 0; i < TILE_SIZE; i++)
            assign acc_chain[i][0] = '0;
    endgenerate

    // ============================================================
    // Instantiate 16×16 Unified PEs (supporting MAC/EWM/EWA)
    // ------------------------------------------------------------
    // ============================================================
    // Instantiate 16×16 Unified PEs (supporting MAC/EWM/EWA)
    // ============================================================
    generate
        for (i = 0; i < TILE_SIZE; i++) begin : ROW
            for (j = 0; j < TILE_SIZE; j++) begin : COL
                pe_unit #(.DATA_WIDTH(DATA_WIDTH)) U_PE (
                    .clk        (clk),
                    .rst_n      (rst_n),
                    .mode       (mode),
                    .valid_in   (valid_in),
                    .a_in       (a_in[i][j]),
                    .b_in       ((mode == 2'b01) ? b_in[i] : b_in[j]),
                    .acc_in     (acc_chain[i][j]),
                    .result_out (pe_result_wire[i][j]),
                    .valid_out  (pe_valid[i][j])
                );
    
                // ★ Modified: 分离 EWM / MAC 路由逻辑（避免竞争）
                // MAC chain
                always_comb begin
                    if (mode == 2'b00)
                        acc_chain[i][j+1] = pe_result_wire[i][j];
                    else
                    //    acc_chain[i][j+1] = acc_chain[i][j+1]; // 保持原值（避免综合优化）
                          acc_chain[i][j+1] = '0;
                end
    
                // EWM matrix
                always_comb begin
                    if (mode == 2'b01)
                        result_out_ewm_n[i][j] = pe_result_wire[i][j];
                    else
                        result_out_ewm_n[i][j] = '0;
                end
            end
        end
    endgenerate


    // ============================================================
    // ★ MAC Mode Output (Rightmost Column Accumulated Result)
    // ============================================================
    generate
        for (i = 0; i < TILE_SIZE; i++) begin
            assign result_out_mac_n[i] = acc_chain[i][TILE_SIZE];

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    result_out_mac[i] <= '0;
                else if (mode == 2'b00)
                    result_out_mac[i] <= result_out_mac_n[i];
            end
        end
    endgenerate

    // ============================================================
    // ★ EWM Mode Output (Full 16×16 Matrix)
    // ============================================================
    generate
        for (i = 0; i < TILE_SIZE; i++) begin
            for (j = 0; j < TILE_SIZE; j++) begin
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n)
                        result_out[i][j] <= '0;
                    else if (mode == 2'b01)
                        result_out[i][j] <= result_out_ewm_n[i][j];
                end
            end
        end
    endgenerate
    // ============================================================
    // Valid signal: when last column's PEs are valid
    // ============================================================
    assign valid_out = pe_valid[0][TILE_SIZE-1];

    // ============================================================
    // Tile-level control (unchanged MAC control)
    // ============================================================
    logic [$clog2(TILE_SIZE+1)-1:0] col_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            col_cnt <= 0;
        else if (valid_out)
            col_cnt <= (col_cnt == TILE_SIZE - 1) ? 0 : col_cnt + 1;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            done_tile <= 1'b0;
        else 
            done_tile <= done_tile_n;
    end

    always_comb begin
        if (mode==2'b00 && valid_out && col_cnt == TILE_SIZE - 1)// MAC模式下，最后一列PE的valid_out且col_cnt满载表示完成
            done_tile_n = 1'b1;
        else if (mode==2'b01 && valid_out)// EWM模式下，valid_out即可表示完成
            done_tile_n = 1'b1;
        else
            done_tile_n = 1'b0;
    end

endmodule
