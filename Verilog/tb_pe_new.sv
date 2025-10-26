//---------------------------------------------------------------
// Testbench : tb_pe_unit_new
// Purpose   : Verify PE (MAC/EWM/EWA) correctness in Q8.8 format
// Author    : Shengjie Chen
//---------------------------------------------------------------
`timescale 1ns/1ps

module tb_pe_unit_new;

  localparam DATA_WIDTH = 16;
  localparam ACC_WIDTH  = 32;
  localparam FRAC_BITS  = 8;

  // ------------------------------------------------------------
  // DUT I/O
  // ------------------------------------------------------------
  reg  clk, rst_n, valid_in;
  reg  [1:0] mode;
  reg  signed [DATA_WIDTH-1:0] a_in, b_in;
  reg  signed [ACC_WIDTH-1:0]  acc_in;
  wire signed [ACC_WIDTH-1:0]  result_out;
  wire valid_out;

  // Instantiate DUT
  pe_unit_new #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH (ACC_WIDTH),
    .FRAC_BITS (FRAC_BITS)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid_in),
    .mode(mode),
    .a_in(a_in),
    .b_in(b_in),
    .acc_in(acc_in),
    .result_out(result_out),
    .valid_out(valid_out)
  );

  // ------------------------------------------------------------
  // Clock & Reset
  // ------------------------------------------------------------
  initial clk = 0;
  always #5 clk = ~clk;

  initial begin
    rst_n = 0; valid_in = 0; mode = 0;
    a_in = 0; b_in = 0; acc_in = 0;
    #20 rst_n = 1;
  end

  // ------------------------------------------------------------
  // Reference Variables
  // ------------------------------------------------------------
  integer i;
  integer product_ref, acc_ref;
  integer ewa_ref, ewa_shifted;
  integer mismatch_cnt;

  // ------------------------------------------------------------
  // Stimulus
  // ------------------------------------------------------------
  initial begin
    @(posedge rst_n);
    $display("\n====================================");
    $display(">>> Testing pe_unit_new (Q8.8)");
    $display("====================================");

    mismatch_cnt = 0;

    // ---------- Mode 00: MAC ----------
    $display("\n[MODE 00] MAC : acc_out = acc_in + (a*b)");
    mode = 2'b00;

    for (i = 0; i < 10; i++) begin
      a_in   = ($random % 8) - 4;   // small integer range
      b_in   = ($random % 8) - 4;
      acc_in = $signed(($random % 64) - 32) <<< FRAC_BITS; // Q16.16
      valid_in = 1;
      @(posedge clk);
      valid_in = 0;
      @(posedge clk);

      // ref = (a*b + acc_in)  (all Q16.16)
      product_ref = $signed(a_in) * $signed(b_in);
      acc_ref     = product_ref + acc_in;

      if (result_out !== acc_ref) begin
        mismatch_cnt++;
        $display("Mismatch(MAC) #%0d : a=%0d b=%0d acc_in=%0d | HW=%0d REF=%0d diff=%0d",
                 i, a_in, b_in, acc_in, result_out, acc_ref, result_out-acc_ref);
      end else begin
        $display("Match(MAC) #%0d : HW=%0d REF=%0d", i, result_out, acc_ref);
      end
    end

    // ---------- Mode 01: EWM ----------
    $display("\n[MODE 01] EWM : result = a*b (Q16.16)");
    mode = 2'b01;

    for (i = 0; i < 10; i++) begin
      a_in = ($random % 8) - 4;
      b_in = ($random % 8) - 4;
      valid_in = 1;
      @(posedge clk);
      valid_in = 0;
      @(posedge clk);

      product_ref = $signed(a_in) * $signed(b_in);
      if (result_out !== product_ref) begin
        mismatch_cnt++;
        $display("Mismatch(EWM) #%0d : a=%0d b=%0d | HW=%0d REF=%0d diff=%0d",
                 i, a_in, b_in, result_out, product_ref, result_out-product_ref);
      end else begin
        $display("Match(EWM) #%0d : HW=%0d REF=%0d", i, result_out, product_ref);
      end
    end

    // ---------- Mode 10: EWA ----------
    $display("\n[MODE 10] EWA : result = (a+b) << 8 (aligned to Q16.16)");
    mode = 2'b10;

    for (i = 0; i < 10; i++) begin
      a_in = ($random % 8) - 4;
      b_in = ($random % 8) - 4;
      valid_in = 1;
      @(posedge clk);
      valid_in = 0;
      @(posedge clk);

      ewa_ref     = $signed(a_in) + $signed(b_in);      // Q9.8 (17 bits)
      ewa_shifted = ewa_ref <<< FRAC_BITS;              // align to Q16.16
      if (result_out !== ewa_shifted) begin
        mismatch_cnt++;
        $display("Mismatch(EWA) #%0d : a=%0d b=%0d | HW=%0d REF=%0d diff=%0d",
                 i, a_in, b_in, result_out, ewa_shifted, result_out-ewa_shifted);
      end else begin
        $display("Match(EWA) #%0d : HW=%0d REF=%0d", i, result_out, ewa_shifted);
      end
    end

    // ---------- Summary ----------
    $display("\n====================================");
    if (mismatch_cnt == 0)
      $display("[PASS] All modes matched reference.");
    else
      $display("[FAIL] Total mismatches = %0d", mismatch_cnt);
    $display("====================================");

    $finish;
  end

endmodule
