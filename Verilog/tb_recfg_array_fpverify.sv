//---------------------------------------------------------------
// Testbench : tb_recfg_array_fpverify
// Purpose   : Verify recfg_array_new (mode=000, Q8.8) with mismatch details
// Author    : Shengjie Chen
//---------------------------------------------------------------
`timescale 1ns/1ps
module tb_recfg_array_fpverify;

  localparam DATA_WIDTH = 16;
  localparam ACC_WIDTH  = 32;
  localparam FRAC_BITS  = 8;
  localparam TILE_SIZE  = 16;
  localparam D_INNER    = 256;
  localparam DT_RANK    = 8;
  localparam D_STATE    = 16;
  localparam OUT_SIZE   = DT_RANK + 2*D_STATE; // 40
  localparam N_ROWBLK   = (OUT_SIZE + TILE_SIZE - 1) / TILE_SIZE; // 3
  localparam N_COLBLK   = D_INNER / TILE_SIZE;                    // 16

  // ============================================================
  // DUT I/O
  // ============================================================
  reg  clk, rst_n, valid_in, accumulate_en;
  reg  [2:0] mode;
  reg  signed [ACC_WIDTH-1:0]  acc_in_vec [0:TILE_SIZE-1];
  reg  signed [DATA_WIDTH-1:0] a_in  [0:TILE_SIZE-1][0:TILE_SIZE-1];
  reg  signed [DATA_WIDTH-1:0] b_vec [0:TILE_SIZE-1];
  reg  signed [DATA_WIDTH-1:0] b_mat [0:TILE_SIZE-1][0:TILE_SIZE-1];
  wire signed [DATA_WIDTH-1:0] result_out_vec [0:TILE_SIZE-1];
  wire signed [DATA_WIDTH-1:0] result_out_mat [0:TILE_SIZE-1][0:TILE_SIZE-1];
  wire valid_out, done_tile, out_shape_flag;

  initial begin clk = 0; forever #5 clk = ~clk; end
  initial begin rst_n = 0; valid_in = 0; accumulate_en = 0; mode = 3'b000;
        #50 rst_n = 1; end

  recfg_array_new #(
      .DATA_WIDTH(DATA_WIDTH),
      .ACC_WIDTH (ACC_WIDTH),
      .FRAC_BITS (FRAC_BITS),
      .TILE_SIZE (TILE_SIZE)
  ) dut (
      .clk(clk), .rst_n(rst_n),
      .valid_in(valid_in), .mode(mode),
      .a_in(a_in), .b_vec(b_vec), .b_mat(b_mat),
      .accumulate_en(accumulate_en), .acc_in_vec(acc_in_vec),
      .result_out_vec(result_out_vec),
      .result_out_mat(result_out_mat),
      .valid_out(valid_out), .done_tile(done_tile),
      .out_shape_flag(out_shape_flag)
  );

  // ============================================================
  // Variables
  // ============================================================
  integer i, j, rb, cb;
  integer mismatch_cnt_000;
  reg signed [DATA_WIDTH-1:0] W   [0:OUT_SIZE-1][0:D_INNER-1];
  reg signed [DATA_WIDTH-1:0] x_t [0:D_INNER-1];
  reg signed [ACC_WIDTH-1:0]  ref_y [0:OUT_SIZE-1];
  reg signed [DATA_WIDTH-1:0] buffer_reg [0:OUT_SIZE-1];
  integer product, acc;

  // ============================================================
  // Stimulus
  // ============================================================
  initial begin
    @(posedge rst_n);

    // MODE000 : MAC (y = W × x_t)
    // --- small range integers (-4~+3)
    for(i=0;i<OUT_SIZE;i++)
      for(j=0;j<D_INNER;j++)
        W[i][j] = ($random % 8) - 4;

    for(j=0;j<D_INNER;j++)
      x_t[j] = ($random % 8) - 4;

    // --- software fixed-point reference (Q16.16 accumulate)
    for(i=0;i<OUT_SIZE;i++) begin
      acc = 0;
      for(j=0;j<D_INNER;j++) begin
        product = W[i][j] * x_t[j]; // 16×16→32
        acc += product;
      end
      ref_y[i] = acc >>> FRAC_BITS; // back to Q8.8
      buffer_reg[i] = 0;
    end

    $display("\n===============================================");
    $display(">>> MODE000 : MAC (40×256 × 256×1) [Q8.8 small-range]");
    $display("===============================================");

    mode = 3'b000;
    for(rb=0; rb<N_ROWBLK; rb++) begin
      for(cb=0; cb<N_COLBLK; cb++) begin
        accumulate_en = (cb!=0);
        for(i=0;i<TILE_SIZE;i++) begin
          int r = rb*TILE_SIZE+i;
          for(j=0;j<TILE_SIZE;j++) begin
            int c = cb*TILE_SIZE+j;
            a_in[i][j] = (r<OUT_SIZE) ? W[r][c] : 0;
          end
        end
        for(j=0;j<TILE_SIZE;j++)
          b_vec[j] = x_t[cb*TILE_SIZE+j];
        for(i=0;i<TILE_SIZE;i++) begin
          int r = rb*TILE_SIZE+i;
          acc_in_vec[i] = (r<OUT_SIZE) ? (buffer_reg[r] <<< FRAC_BITS) : 0;
        end

        @(posedge clk); valid_in=1; repeat(TILE_SIZE) @(posedge clk);
        valid_in=0; @(posedge done_tile); @(posedge clk);

        for(i=0;i<TILE_SIZE;i++) begin
          int r = rb*TILE_SIZE+i;
          if(r<OUT_SIZE) buffer_reg[r] = result_out_vec[i];
        end
      end
    end

    // ============================================================
    // Compare and show detailed mismatches
    // ============================================================
    mismatch_cnt_000 = 0;
    for(i=0;i<OUT_SIZE;i++) begin
      if(buffer_reg[i] !== ref_y[i][DATA_WIDTH-1:0]) begin
        mismatch_cnt_000++;
        $display("Mismatch @%0d : HW=%0d, REF=%0d (diff=%0d)", 
                 i, buffer_reg[i], ref_y[i][DATA_WIDTH-1:0],
                 buffer_reg[i]-ref_y[i][DATA_WIDTH-1:0]);
      end
    end

    if(mismatch_cnt_000==0)
      $display("[PASS] MODE000: All %0d outputs match reference.", OUT_SIZE);
    else
      $display("[FAIL] MODE000: %0d mismatches detected.", mismatch_cnt_000);

    $finish;
  end

endmodule
