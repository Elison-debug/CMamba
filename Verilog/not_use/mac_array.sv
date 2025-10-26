//---------------------------------------------------------------
// Module: mac_array
// Function: 16×16 systolic array for row-wise MAC accumulation
// Supports: 
//   mode=0: X_PROJ   → RAW = W_xpj × x_t
//   mode=1: Δ_t_PROJ → Δ_t = W_Δ × Δ_raw
// Author: Shengjie Chen

//             x0 → x1 → x2 → x3
//             ↓    ↓    ↓    ↓
//Row0:  W00→PE00→PE01→PE02→PE03→ y0
//Row1:  W10→PE10→PE11→PE12→PE13→ y1
//Row2:  W20→PE20→PE21→PE22→PE23→ y2
//Row3:  W30→PE30→PE31→PE32→PE33→ y3
//---------------------------------------------------------------
module mac_array #(
    parameter DATA_WIDTH = 16,
    parameter TILE_SIZE  = 16
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         valid_in,
    input  logic                         mode,   // 0: X_PROJ, 1: Δ_t_PROJ
    input  logic signed [DATA_WIDTH-1:0] a_in [TILE_SIZE-1:0][TILE_SIZE-1:0], // weights tile
    input  logic signed [DATA_WIDTH-1:0] b_in [TILE_SIZE-1:0],                 // input vector tile
    output logic signed [DATA_WIDTH-1:0] result_out [TILE_SIZE-1:0],           // output vector tile
    output logic                         valid_out,
    output logic                         done_tile
);

    // ============================================================
    // Internal interconnects
    // ============================================================
    // Horizontal accumulation chain (→)
    logic signed [DATA_WIDTH-1:0] acc_chain [TILE_SIZE-1:0][TILE_SIZE:0];
    logic                         pe_valid  [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic                         done_tile_n;
    logic signed [DATA_WIDTH-1:0] result_out_n [TILE_SIZE-1:0];
    genvar i, j;

    // ============================================================
    // Initialize leftmost accumulations to zero (start of each row)
    // ============================================================
    generate
        for (i = 0; i < TILE_SIZE; i++)
            assign acc_chain[i][0] = '0;
    endgenerate

    // ============================================================
    // Instantiate 16×16 PE grid (row-wise accumulation)
    // ============================================================
    generate
        for (i = 0; i < TILE_SIZE; i++) begin : ROW
            for (j = 0; j < TILE_SIZE; j++) begin : COL
                pe_mac #(.DATA_WIDTH(DATA_WIDTH)) U_PE (
                    .clk        (clk),
                    .rst_n      (rst_n),
                    .valid_in   (valid_in),
                    .a_in       (a_in[i][j]),      // weight W[i][j]
                    .b_in       (b_in[j]),         // input x[j]
                    .acc_in     (acc_chain[i][j]), // ← from left neighbor
                    .result_out (acc_chain[i][j+1]), // → to right neighbor
                    .valid_out  (pe_valid[i][j])
                );
            end
        end
    endgenerate

    // ============================================================
    // Collect outputs from the rightmost column (final row sums)
    // ============================================================
    generate
        for (i = 0; i < TILE_SIZE; i++)begin
            assign result_out_n[i] = acc_chain[i][TILE_SIZE];
            always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n)
                result_out[i] <= '0;
            else 
                result_out[i] <= result_out_n[i];
            end
        end
    endgenerate

    // ============================================================
    // Valid signal: when last column's PEs are valid
    // ============================================================
    assign valid_out = pe_valid[0][TILE_SIZE-1];

    // ============================================================
    // Tile-level control (each tile has TILE_SIZE columns)
    // ============================================================
    logic [$clog2(TILE_SIZE+1)-1:0] col_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            col_cnt <= 0;
        else if (valid_out)
            col_cnt <= (col_cnt == TILE_SIZE - 1) ? 0 : col_cnt + 1;
    end

    // done_tile: high pulse after finishing one tile
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            done_tile <= 1'b0;
        else 
            done_tile <= done_tile_n;
    end
    //combinational logic for done_tile
    always_comb begin
        if (valid_out && col_cnt == TILE_SIZE - 1)
            done_tile_n = 1'b1;
        else
            done_tile_n = 1'b0;
    end
endmodule
