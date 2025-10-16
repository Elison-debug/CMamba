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
//---------------------------------------------------------------

module pe_unit #(
    parameter DATA_WIDTH = 16
)(
    input  logic                     clk,        // system clock
    input  logic                     rst_n,      // asynchronous active-low reset
    input  logic                     valid_in,   // input data valid
    input  logic [1:0]               mode,       // 00:MAC, 01:EWM, 10:EWA
    input  logic signed [DATA_WIDTH-1:0] a_in,   // operand A (e.g. weight)
    input  logic signed [DATA_WIDTH-1:0] b_in,   // operand B (e.g. feature)
    input  logic signed [DATA_WIDTH-1:0] acc_in, // accumulated input (for MAC)
    output logic signed [DATA_WIDTH-1:0] result_out, // operation result
    output logic                     valid_out   // output data valid
);

    // Internal signal declarations
    logic signed [DATA_WIDTH-1:0] mult_res;
    logic signed [DATA_WIDTH-1:0] add_res;
    logic signed [DATA_WIDTH-1:0] mac_res;
    logic valid_out_n;

    // ----------------------------------------------------------------
    // Combinational logic: unified arithmetic with mode control
    // ----------------------------------------------------------------
    always_comb begin
        // default
        mult_res    = '0;
        add_res     = '0;
        mac_res     = '0;
        valid_out_n = 1'b0;

        if (valid_in) begin
            mult_res = a_in * b_in;  // always computed in parallel

            case (mode)
                2'b00: begin // MAC
                    mac_res = mult_res + acc_in;
                    add_res = mac_res;
                end
                2'b01: begin // EWM (multiply only)
                    add_res = mult_res;
                end
                2'b10: begin // EWA (adder bypasses multiplier)
                    add_res = a_in + b_in;
                end
                default: begin
                    add_res = '0;
                end
            endcase

            valid_out_n = 1'b1;
        end
    end

    // ----------------------------------------------------------------
    // Sequential logic: output register (one-cycle latency)
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_out <= '0;
            valid_out  <= 1'b0;
        end else begin
            result_out <= add_res;   // operation result
            valid_out  <= valid_out_n;
        end
    end

endmodule
