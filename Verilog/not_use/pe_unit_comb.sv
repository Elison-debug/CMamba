module pe_unit_comb #(
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH  = 32,
    parameter FRAC_BITS  = 8
)(
    input  logic [1:0]                    mode,       // 00:MAC, 01:EWM, 10:EWA
    input  logic signed [DATA_WIDTH-1:0]  a_in,
    input  logic signed [DATA_WIDTH-1:0]  b_in,
    input  logic signed [ACC_WIDTH-1:0]   acc_in,
    output logic signed [ACC_WIDTH-1:0]   result_out
);
    localparam SHIFTED_WIDTH = (DATA_WIDTH + 1) + FRAC_BITS; // 17 + 8 = 25
    logic signed [SHIFTED_WIDTH-1:0] ewa_shifted;
    logic signed [ACC_WIDTH-1:0] mult_full;
    logic signed [DATA_WIDTH:0]  ewa_sum;
    logic signed [ACC_WIDTH-1:0] ewa_sum_aligned;

    //assign mult_full = a_in * b_in;
    assign mult_full = $signed(a_in) * $signed(b_in);
    assign ewa_sum   = a_in + b_in;
    
    assign ewa_shifted = $signed(ewa_sum) <<< FRAC_BITS;
    assign ewa_sum_aligned = {{(ACC_WIDTH - SHIFTED_WIDTH){ewa_shifted[SHIFTED_WIDTH-1]}}, ewa_shifted};

    always_comb begin
        case(mode)
            2'b00: result_out = mult_full + acc_in;   // MAC
            2'b01: result_out = mult_full;            // EWM
            2'b10: result_out = ewa_sum_aligned;      // EWA
            default: result_out = '0;
        endcase
    end
endmodule
