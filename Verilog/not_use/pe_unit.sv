//---------------------------------------------------------------
// Module: pe_unit
// Function: Unified Processing Element supporting MAC / EWM / EWA
// Author: Shengjie Chen
// Description:
//   Performs one of three operations depending on mode:
//     mode = 2'b00 → MAC: result = a_in * b_in + acc_in
//     mode = 2'b01 → EWM: result = a_in * b_in
//     mode = 2'b10 → EWA: result = a_in + b_in
//   - Unified PE used for all phases of SSM
//   - One clock-cycle latency
//   - Supports fixed-point arithmetic for Q8.8 format
//   - a_in, b_in: Q8.8 (16-bit signed)
//   - Multiplier output: Q16.16 (32-bit)
//   - Shift right 16 bits to restore Q8.8
//---------------------------------------------------------------

module pe_unit #(
    parameter DATA_WIDTH = 16,
    parameter FRAC_BITS  = 8   // Q8.8 format → shift 16 bits (2*FRAC_BITS)
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         valid_in,
    input  logic [1:0]                   mode,       // 00:MAC, 01:EWM, 10:EWA
    input  logic signed [DATA_WIDTH-1:0] a_in,       // Q8.8
    input  logic signed [DATA_WIDTH-1:0] b_in,       // Q8.8
    input  logic signed [DATA_WIDTH-1:0] acc_in,     // Q8.8
    output logic signed [DATA_WIDTH-1:0] result_out, // Q8.8
    output logic                         valid_out
);

    // ============================================================
    // Internal extended signals
    // ============================================================
    logic signed [2*DATA_WIDTH-1:0] mult_full;    // Q16.16 full precision
    logic signed [DATA_WIDTH-1:0]   mult_scaled;  // after shift → Q8.8
    logic signed [DATA_WIDTH:0]     add_full;     // one extra bit for overflow
    logic signed [DATA_WIDTH-1:0]   add_trunc;    // truncated 16-bit result
    logic                           valid_out_n;

    // ============================================================
    // Combinational arithmetic logic
    // ============================================================
    always_comb begin
        mult_full   = '0;
        mult_scaled = '0;
        add_full    = '0;
        add_trunc   = '0;
        valid_out_n = 1'b0;

        if (valid_in) begin
            // ------------------------------------------------------------
            // 1. Fixed-point multiply (Q8.8 × Q8.8 = Q16.16)
            // ------------------------------------------------------------
            mult_full   = a_in * b_in;

            // ------------------------------------------------------------
            // 2. Shift right 16 bits ( FRAC_BITS) to restore Q8.8
            // ------------------------------------------------------------
            mult_scaled = mult_full >>>  FRAC_BITS;

            // ------------------------------------------------------------
            // 3. Mode selection
            // ------------------------------------------------------------
            case (mode)
                2'b00: add_full = mult_scaled + acc_in; // MAC
                2'b01: add_full = mult_scaled;          // EWM
                2'b10: add_full = a_in + b_in;          // EWA (no scaling)
                default: add_full = '0;
            endcase

            // ------------------------------------------------------------
            // 4. Truncate / (optional: add saturation later)
            // ------------------------------------------------------------
            add_trunc   = add_full[DATA_WIDTH-1:0];
            valid_out_n = 1'b1;
        end
    end

    // ============================================================
    // Sequential output register (1-cycle latency)
    // ============================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_out <= '0;
            valid_out  <= 1'b0;
        end else begin
            result_out <= add_trunc;
            valid_out  <= valid_out_n;
        end
    end

endmodule
