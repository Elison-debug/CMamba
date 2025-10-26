//---------------------------------------------------------------
// Module: pe_mac
// Function: Single Processing Element for Multiply-Accumulate
// Author: Shengjie Chen
// Description:
//   Performs: result_out = a_in * b_in + acc_in
//   - Used in X_PROJ and Δ_t_PROJ phases of SSM
//   - One clock-cycle latency
//---------------------------------------------------------------

module pe_mac #(
    parameter DATA_WIDTH = 16
)(
    input  logic                     clk,        // system clock
    input  logic                     rst_n,      // asynchronous active-low reset
    input  logic                     valid_in,   // input data valid
    input  logic signed [DATA_WIDTH-1:0] a_in,   // weight input
    input  logic signed [DATA_WIDTH-1:0] b_in,   // input feature
    input  logic signed [DATA_WIDTH-1:0] acc_in, // accumulated input (partial sum)
    output logic signed [DATA_WIDTH-1:0] result_out, // accumulated result
    output logic                     valid_out   // output data valid
);

    // Internal signal declarations
    logic signed [DATA_WIDTH-1:0] mult_res;
    logic signed [DATA_WIDTH-1:0] add_res;
    logic valid_out_n;
    // ----------------------------------------------------------------
    // Combinational logic: multiplication and addition
    // ----------------------------------------------------------------
    always_comb begin
        if (valid_in)begin
            mult_res = a_in * b_in;
            add_res  = mult_res + acc_in;
            valid_out_n = 1'b1;
        end else begin
            mult_res = '0;
            add_res  = '0;
            valid_out_n = 1'b0;
        end
    end

    // ----------------------------------------------------------------
    // Sequential logic: register output for pipeline
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_out <= '0;
            valid_out  <= 1'b0;
        end else begin
            result_out <= add_res;  // update result when input valid
            valid_out  <= valid_out_n;
        end 
    end

endmodule
