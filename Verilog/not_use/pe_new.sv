//---------------------------------------------------------------
// Module: pe_unit (MODIFIED for High-Precision & Hybrid Arch)
// Function: Unified PE supporting high-precision MAC / EWM / EWA
//
// Description:
//   - FIX 1: All inputs/outputs are now aligned to a high-precision
//     32-bit (Q16.16) datapath.
//   - FIX 2: MAC (2'b00) now correctly adds Q16.16 product to a
//     32-bit Q16.16 accumulator.
//   - FIX 3: EWM (2'b01) now correctly outputs the full 32-bit
//     Q16.16 product.
//   - FIX 4: EWA (2'b10) now correctly calculates the Q9.8 sum
//     and aligns it to the 32-bit Q16.16 output format.
//
//   Data Formats:
//   - a_in, b_in:   Q8.8  (16-bit)
//   - acc_in:       Q16.16 (32-bit)
//   - result_out:   Q16.16 (32-bit)
//---------------------------------------------------------------

module pe_unit_new #(
    parameter DATA_WIDTH = 16, // Input data width (Q8.8)
    parameter ACC_WIDTH  = 32, // Accumulator/Output width (Q16.16)
    parameter FRAC_BITS  = 8   // Fractional bits in input
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          valid_in,
    input  logic [1:0]                    mode,       // 00:MAC, 01:EWM, 10:EWA
    
    // 16-bit inputs (Q8.8)
    input  logic signed [DATA_WIDTH-1:0]  a_in,
    input  logic signed [DATA_WIDTH-1:0]  b_in,
    
    // 32-bit accumulator input (Q16.16)
    input  logic signed [ACC_WIDTH-1:0]   acc_in,
    
    // 32-bit result output (Q16.16)
    output logic signed [ACC_WIDTH-1:0]   result_out,
    output logic                          valid_out
);

    // ============================================================
    // Internal signals
    // ============================================================
    
    // Full precision multiplier output: Q8.8 * Q8.8 = Q16.16 (32-bit)
    logic signed [ACC_WIDTH-1:0]   mult_full;

    // Full precision adder output: Q8.8 + Q8.8 = Q9.8 (17-bit)
    logic signed [DATA_WIDTH:0]    ewa_sum; 

    // Aligned EWA result: Q9.8 aligned to Q16.16 format
    logic signed [ACC_WIDTH-1:0]   ewa_sum_aligned;

    // Internal result register
    logic signed [ACC_WIDTH-1:0]   result_n;
    logic                          valid_out_n;

    // ============================================================
    // Combinational arithmetic logic
    // ============================================================
    
    // Perform both main operations in parallel
    //assign mult_full = a_in * b_in;
    //assign mult_full =
    //$signed({{(ACC_WIDTH-DATA_WIDTH){a_in[DATA_WIDTH-1]}}, a_in}) *
    //$signed({{(ACC_WIDTH-DATA_WIDTH){b_in[DATA_WIDTH-1]}}, b_in});
    assign mult_full = a_in * b_in;
    assign ewa_sum   = a_in + b_in;
    
    // Align the EWA (add) result to the Q16.16 output format.
    // ewa_sum is Q9.8 (17 bits). We need to shift it left by 8 bits
    // to match the Q16.16 format (which has 16 fractional bits).
    // (17-bit Q9.8) << 8 = (25-bit Q9.16)
    // Then sign-extend this 25-bit value to our 32-bit ACC_WIDTH.
    always_comb begin
        logic signed [24:0] ewa_shifted;
        ewa_shifted = $signed(ewa_sum) <<< FRAC_BITS;
        ewa_sum_aligned = { {(ACC_WIDTH-25){ewa_shifted[24]}}, ewa_shifted };
    end

    always_comb begin
        result_n    = '0;
        valid_out_n = 1'b0;

        if (valid_in) begin
            case (mode)
                // MAC: (Q16.16) + (Q16.16) = (Q16.16)
                2'b00: result_n = mult_full + acc_in;
                
                // EWM: Output full Q16.16 product
                2'b01: result_n = mult_full;
                
                // EWA: Output Q9.8 sum, aligned to Q16.16
                2'b10: result_n = ewa_sum_aligned;
                
                default: result_n = '0;
            endcase
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
            result_out <= result_n;
            valid_out  <= valid_out_n;
        end
    end

endmodule