// ===============================================================
//  File: multi_bank_wbuf.sv
//  Function: 12-bank read-only WBUF subsystem for Mamba SSM
// ===============================================================
module multi_bank_wbuf #(
    parameter int N_BANK = 12,
    parameter int ADDR_W = 10,
    parameter int DATA_W = 256
)(
    input  logic                 clk,
    input  logic [N_BANK-1:0]    en_bank,
    input  logic [ADDR_W-1:0]    addr_bank [N_BANK],
    output logic [DATA_W-1:0]    dout_bank [N_BANK]
);

    // ------------------------------------------------------------
    // Instantiate 12 single-port ROM banks
    // ------------------------------------------------------------
    generate
        for (genvar i = 0; i < N_BANK; i++) begin : WBUF_BANK
            WBUF_Xproj_bank u_bank (
                .clka  (clk),
                .ena   (en_bank[i]),
                .addra (addr_bank[i]),
                .douta (dout_bank[i])
            );
        end
    endgenerate

endmodule
