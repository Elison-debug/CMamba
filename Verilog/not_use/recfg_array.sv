//---------------------------------------------------------------
// Module: recfg_array
// Function: 16×16 systolic array with full multi-mode PE support
// Author: Shengjie Chen
// Description:
//   Unified array supporting MAC, EWM, and EWA operations.
//
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
//
//   Notes:
//     - Both A and B are 16×16 matrices (vector = first column)
//     - PE mode[1:0]: 00=MAC, 01=Multiply, 10=Add
//     - mode[2] used to differentiate between vector/matrix classes
//     - out_shape_flag: 0=Vector Output, 1=Matrix Output
//---------------------------------------------------------------
module recfg_array #(
    parameter DATA_WIDTH = 16,
    parameter TILE_SIZE  = 16
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         valid_in,
    input  logic [2:0]                   mode,   // 000–110: operation mode

    // --- Matrix A ---
    input  logic signed [DATA_WIDTH-1:0] a_in [TILE_SIZE-1:0][TILE_SIZE-1:0],
    // --- Vector b (used by most modes) ---
    input  logic signed [DATA_WIDTH-1:0] b_vec [TILE_SIZE-1:0],
    // --- Matrix b (only for mode 101 & 110) ---
    input  logic signed [DATA_WIDTH-1:0] b_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],
    // --- Additional inputs for MAC accumulation (only for mode 000) ---
    input  logic accumulate_en, 
    input  logic signed [DATA_WIDTH-1:0] acc_in_vec [TILE_SIZE-1:0], 
    // --- Outputs ---
    output logic signed [DATA_WIDTH-1:0] result_out_vec [TILE_SIZE-1:0],
    output logic signed [DATA_WIDTH-1:0] result_out_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],
    output logic                         valid_out,
    output logic                         done_tile,
    output logic                         out_shape_flag
);

    // ============================================================
    // Internal signals
    // ============================================================
    logic signed [DATA_WIDTH-1:0] acc_chain [TILE_SIZE-1:0][TILE_SIZE:0];
    logic signed [DATA_WIDTH-1:0] pe_result_wire [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic                         pe_valid  [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic                         done_tile_n;
    logic [$clog2(TILE_SIZE+1)-1:0] col_cnt;

    genvar i, j;
    //mode mapping
    logic [1:0] pe_mode;

    always_comb begin
        case (mode)
            3'b000: pe_mode = 2'b00; // MAC
            3'b001: pe_mode = 2'b01; // EWM (Matrix)
            3'b010: pe_mode = 2'b01; // EWM (Vector) ✅ 修正！
            3'b011: pe_mode = 2'b01; // EWM (Outer)
            3'b100: pe_mode = 2'b10; // EWA (Vector)
            3'b101: pe_mode = 2'b10; // EWA (Matrix)
            3'b110: pe_mode = 2'b01; // EWM-Matrix2
            default: pe_mode = 2'b00;
        endcase
    end

    // ============================================================
    // Initialize accumulations
    // ============================================================
    /*
    generate
        for (i = 0; i < TILE_SIZE; i++)
            assign acc_chain[i][0] = '0;
    endgenerate
   */
    generate
    for (i = 0; i < TILE_SIZE; i++)
        assign acc_chain[i][0] = (accumulate_en) ? acc_in_vec[i] : '0;
    endgenerate
    // ============================================================
    // PE Array (vector/matrix selective input)
    // ============================================================
    generate
    for (i = 0; i < TILE_SIZE; i++) begin : ROW
        for (j = 0; j < TILE_SIZE; j++) begin : COL

            // --- b input selection logic ---
            //行广播（b_vec[i]） → 每行乘同一个标量（如 EWM-Matrix）。
            //列广播（b_vec[j]） → 每列乘同一个标量（如 外积、矩阵乘法）。
            logic signed [DATA_WIDTH-1:0] b_sel;
            always_comb begin
                case (mode)
                    // 列广播：MAC / Outer / EWA-Vector
                    3'b000, 3'b011:
                        b_sel = b_vec[j];
                    // 行广播：EWM-Matrix / EWM-Vector
                    3'b001, 3'b010, 3'b100:
                        b_sel = b_vec[i];
                    // 矩阵输入：EWA-Matrix / EWM-Matrix2
                    3'b101, 3'b110:
                        b_sel = b_mat[i][j];
                    default:
                        b_sel = '0;
                endcase
            end

                // --- PE Instantiation ---
                pe_unit #(.DATA_WIDTH(DATA_WIDTH)) U_PE (
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
                        acc_chain[i][j+1] <= pe_result_wire[i][j];
                    else
                        acc_chain[i][j+1] <= '0;
                end
            end
        end
    endgenerate

    // ============================================================
    // Output collection
    // ============================================================
    generate
        for (i = 0; i < TILE_SIZE; i++) begin
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    result_out_vec[i] <= '0;
                else if (mode == 3'b000)
                    result_out_vec[i] <= acc_chain[i][TILE_SIZE];
                else if (mode inside {3'b010,3'b100})
                    result_out_vec[i] <= pe_result_wire[i][TILE_SIZE-1];
            end
        end

        for (i = 0; i < TILE_SIZE; i++) begin
            for (j = 0; j < TILE_SIZE; j++) begin
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n)
                        result_out_mat[i][j] <= '0;
                    else if (mode inside {3'b001,3'b011,3'b101,3'b110})
                        result_out_mat[i][j] <= pe_result_wire[i][j];
                end
            end
        end
    endgenerate

    // ============================================================
    // Control logic
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
        if (mode==3'b000 && valid_out && col_cnt == TILE_SIZE - 1)
            done_tile_n = 1'b1;
        else if (mode inside {3'b001,3'b010,3'b011,3'b100,3'b101,3'b110} && valid_out)
            done_tile_n = 1'b1;
        else
            done_tile_n = 1'b0;
    end

endmodule
