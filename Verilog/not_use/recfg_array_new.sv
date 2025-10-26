//---------------------------------------------------------------
// Module: recfg_array (High-Precision Version)
// Function: 16×16 systolic array using 32-bit accumulator datapath
// Author: Shengjie Chen
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
// Notes:
//   - Updated to support high-precision PE (Q16.16 internal format)
//   - Inputs remain 16-bit (Q8.8)
//   - Accumulator and PE results are 32-bit
//   - Final outputs truncated back to 16-bit (Q8.8)
//---------------------------------------------------------------
module recfg_array_new #(
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH  = 32,  // New: 32-bit internal precision
    parameter FRAC_BITS  = 8,
    parameter TILE_SIZE  = 16
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         valid_in,
    input  logic [2:0]                   mode,   // 000–110: operation mode

    // --- Matrix A (input weights)
    input  logic signed [DATA_WIDTH-1:0] a_in [TILE_SIZE-1:0][TILE_SIZE-1:0],
    // --- Vector b (used by most modes)
    input  logic signed [DATA_WIDTH-1:0] b_vec [TILE_SIZE-1:0],
    // --- Matrix b (for mode 101 & 110)
    input  logic signed [DATA_WIDTH-1:0] b_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],

    // --- Additional inputs for MAC accumulation (mode 000)
    input  logic accumulate_en, 
    input  logic signed [ACC_WIDTH-1:0] acc_in_vec [TILE_SIZE-1:0],

    // --- Outputs (externally remain 16-bit)
    output logic signed [DATA_WIDTH-1:0] result_out_vec [TILE_SIZE-1:0],
    output logic signed [DATA_WIDTH-1:0] result_out_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],
    output logic                         valid_out,
    output logic                         done_tile,
    output logic                         out_shape_flag
);

    // ============================================================
    // Internal high-precision signals
    // ============================================================
    logic signed [ACC_WIDTH-1:0] acc_chain [TILE_SIZE-1:0][TILE_SIZE:0];
    logic signed [ACC_WIDTH-1:0] pe_result_wire [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic                         pe_valid  [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic                         done_tile_n;
    logic [$clog2(TILE_SIZE+1)-1:0] col_cnt;

    genvar i, j;
    logic [1:0] pe_mode;

    logic done_mac;
    logic done_other;
    // ============================================================
    // Mode mapping (unchanged)
    // ============================================================
    always_comb begin
        case (mode)
            3'b000: pe_mode = 2'b00; // MAC
            3'b001: pe_mode = 2'b01; // EWM (Matrix)
            3'b010: pe_mode = 2'b01; // EWM (Vector)
            3'b011: pe_mode = 2'b01; // EWM (Outer)
            3'b100: pe_mode = 2'b10; // EWA (Vector)
            3'b101: pe_mode = 2'b10; // EWA (Matrix)
            3'b110: pe_mode = 2'b01; // EWM-Matrix2
            default: pe_mode = 2'b00;
        endcase
    end

    // ============================================================
    // Initialize high-precision accumulations
    // ============================================================
    //generate
    //    for (i = 0; i < TILE_SIZE; i++)
    //        assign acc_chain[i][0] = (accumulate_en) ? acc_in_vec[i] : '0;
    //endgenerate
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            for (int k=0; k<TILE_SIZE; k++) acc_chain[k][0] <= '0;
        //else if (valid_in)
        else begin
            for (int k=0; k<TILE_SIZE; k++)
                acc_chain[k][0] <= (accumulate_en) ? acc_in_vec[k] : '0;
        end
    end

    // ============================================================
    // PE Array instantiation
    // ============================================================
    generate
        for (i = 0; i < TILE_SIZE; i++) begin : ROW
            for (j = 0; j < TILE_SIZE; j++) begin : COL

                logic signed [DATA_WIDTH-1:0] b_sel;

                always_comb begin
                    case (mode)
                        3'b000, 3'b011: b_sel = b_vec[j];   // Column broadcast
                        3'b001, 3'b010, 3'b100: b_sel = b_vec[i]; // Row broadcast
                        3'b101, 3'b110: b_sel = b_mat[i][j];     // Matrix input
                        default: b_sel = '0;
                    endcase
                end

                // --- High-precision PE Instantiation ---
                pe_unit_new #(
                    .DATA_WIDTH (DATA_WIDTH),
                    .ACC_WIDTH  (ACC_WIDTH)
                ) U_PE (
                    .clk        (clk),
                    .rst_n      (rst_n),
                    .mode       (pe_mode),
                    .valid_in   (valid_in),
                    .a_in       (a_in[i][j]),
                    .b_in       (b_sel),
                    .acc_in     (acc_chain[i][j]),
                    .result_out (pe_result_wire[i][j]),
                    .valid_out  (pe_valid[i][j])
                );

                // --- MAC accumulation (mode 000 only) ---
                
                always_comb begin
                    if (mode == 3'b000)
                        acc_chain[i][j+1] = pe_result_wire[i][j];
                    else
                        acc_chain[i][j+1] = '0;
                end
                        
                        /*
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n)
                        acc_chain[i][j+1] <= '0;
                    else if (mode == 3'b000 && pe_valid[i][j]) // ✅ 注意这里
                        acc_chain[i][j+1] <= pe_result_wire[i][j];
                    else
                        acc_chain[i][j+1] <= acc_chain[i][j+1]; // ✅ 保持
                end
                */
            end
        end
    endgenerate

    // ============================================================
    // Output collection (truncate Q16.16 → Q8.8)
    // ============================================================
    generate
        for (i = 0; i < TILE_SIZE; i++) begin
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    result_out_vec[i] <= '0;
                else if (mode == 3'b000)
                    result_out_vec[i] <= $signed(pe_result_wire[i][TILE_SIZE-1]) >>> FRAC_BITS;  // Safe shift
                else if (mode inside {3'b010,3'b100})
                    result_out_vec[i] <= $signed(pe_result_wire[i][TILE_SIZE-1]) >>> FRAC_BITS;  // Safe shift
            end
        end

        for (i = 0; i < TILE_SIZE; i++) begin
            for (j = 0; j < TILE_SIZE; j++) begin
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n)
                        result_out_mat[i][j] <= '0;
                    else if (mode inside {3'b001,3'b011,3'b101,3'b110})
                        result_out_mat[i][j] <= $signed(pe_result_wire[i][j]) >>> FRAC_BITS;  // Safe shift
                end
            end
        end
    endgenerate

    // ============================================================
    // Control logic (unchanged)
    // ============================================================
    assign valid_out = pe_valid[TILE_SIZE-1][TILE_SIZE-1];
    assign out_shape_flag = (mode inside {3'b001,3'b011,3'b101,3'b110});

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            col_cnt <= 0;
        else if (valid_out)
            col_cnt <= (col_cnt == TILE_SIZE-1) ? 0 : col_cnt + 1;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            done_tile <= 1'b0;
        else
            done_tile <= done_tile_n;
    end

    always_comb begin
        // 条件 1: MAC 模式完成 (在第15个计数时)
        done_mac = (mode == 3'b000) && (col_cnt == TILE_SIZE - 1);
        
        // 条件 2: 任何其他模式完成 (立即完成)
        done_other = (mode inside {3'b001, 3'b010, 3'b011, 3'b100, 3'b101, 3'b110});

        // 最终决定
        if (valid_out && (done_mac || done_other))
            done_tile_n = 1'b1;
        else
            done_tile_n = 1'b0;
    end

endmodule
