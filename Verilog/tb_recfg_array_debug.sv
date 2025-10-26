//---------------------------------------------------------------
// Testbench : tb_recfg_array_debug_fixed (TILE_SIZE=2)
// Purpose   : Verify MAC accumulation across tiles in recfg_array_new
// Author    : Shengjie Chen
// --------------------------------------------------------------
// [MOD] Version with continuous valid_in and result_buffer mechanism
//---------------------------------------------------------------
`timescale 1ns/1ps
module tb_recfg_array_debug;

  localparam DATA_WIDTH = 16;
  localparam ACC_WIDTH  = 32;
  localparam FRAC_BITS  = 8;
  localparam TILE_SIZE  = 2;
  localparam D_INNER    = 4;
  localparam OUT_SIZE   = 2;

  // ============================================================
  // DUT I/O
  // ============================================================
  reg  clk, rst_n, valid_in, accumulate_en;
  reg  [2:0] mode;

  reg  signed [DATA_WIDTH-1:0] a_in [0:TILE_SIZE-1][0:TILE_SIZE-1];
  reg  signed [DATA_WIDTH-1:0] b_vec [0:TILE_SIZE-1];
  reg  signed [DATA_WIDTH-1:0] b_mat [0:TILE_SIZE-1][0:TILE_SIZE-1];
  reg  signed [ACC_WIDTH-1:0]  acc_in_vec [0:TILE_SIZE-1];

  wire signed [DATA_WIDTH-1:0] result_out_vec [0:TILE_SIZE-1];
  wire signed [DATA_WIDTH-1:0] result_out_mat [0:TILE_SIZE-1][0:TILE_SIZE-1];
  wire valid_out, done_tile, out_shape_flag;

  integer ref_y0, ref_y1;
  integer i, j, step;

  // ============================================================
  // Clock & Reset
  // ============================================================
  initial clk = 0;
  always #5 clk = ~clk;

  initial begin
    rst_n = 0;
    valid_in = 0;
    accumulate_en = 0;
    mode = 3'b000;
    #20 rst_n = 1;
  end

  // ============================================================
  // DUT Instantiation
  // ============================================================
  recfg_array_new #(
      .DATA_WIDTH(DATA_WIDTH),
      .ACC_WIDTH (ACC_WIDTH),
      .FRAC_BITS (FRAC_BITS),
      .TILE_SIZE (TILE_SIZE)
  ) dut (
      .clk(clk),
      .rst_n(rst_n),
      .valid_in(valid_in),
      .mode(mode),
      .a_in(a_in),
      .b_vec(b_vec),
      .b_mat(b_mat),
      .accumulate_en(accumulate_en),
      .acc_in_vec(acc_in_vec),
      .result_out_vec(result_out_vec),
      .result_out_mat(result_out_mat),
      .valid_out(valid_out),
      .done_tile(done_tile),
      .out_shape_flag(out_shape_flag)
  );

  // ============================================================
  // [MOD] 新增：result_buffer，用于跨 tile 保存结果
  // ============================================================
  reg signed [DATA_WIDTH-1:0] result_buffer [0:TILE_SIZE-1];

  // ============================================================
  // Test Data (Manual Small Matrix)
  // ============================================================
  // W = [1 2 3 4;
  //      5 6 7 8]
  // x = [1; 2; 3; 4]
  // Expected y = [30; 70]
  reg signed [DATA_WIDTH-1:0] W [0:OUT_SIZE-1][0:D_INNER-1];
  reg signed [DATA_WIDTH-1:0] x_t [0:D_INNER-1];

  initial begin
    @(posedge rst_n);
    mode = 3'b000;  // MAC mode

    // Initialize test data (Q8.8 integers)
    W[0][0] = 1 <<< FRAC_BITS;  W[0][1] = 2 <<< FRAC_BITS;
    W[0][2] = 3 <<< FRAC_BITS;  W[0][3] = 4 <<< FRAC_BITS;
    W[1][0] = 5 <<< FRAC_BITS;  W[1][1] = 6 <<< FRAC_BITS;
    W[1][2] = 7 <<< FRAC_BITS;  W[1][3] = 8 <<< FRAC_BITS;
    x_t[0]  = 1 <<< FRAC_BITS;
    x_t[1]  = 2 <<< FRAC_BITS;
    x_t[2]  = 3 <<< FRAC_BITS;
    x_t[3]  = 4 <<< FRAC_BITS;

    for (i=0;i<TILE_SIZE;i++) begin
      acc_in_vec[i] = 0;
      result_buffer[i] = 0;  // [MOD] 初始化 result_buffer
    end

    // ============================================================
    // Feed 2×2 tile each time
    // ============================================================
    for (step=0; step < D_INNER/TILE_SIZE; step++) begin

      // --- [MOD] 1️⃣ 加载当前tile数据 ---
      for (i=0;i<TILE_SIZE;i++)
        for (j=0;j<TILE_SIZE;j++)
          a_in[i][j] = W[i][step*TILE_SIZE+j];

      for (j=0;j<TILE_SIZE;j++)
        b_vec[j] = x_t[step*TILE_SIZE+j];

      // --- [MOD] 2️⃣ 设置控制信号 ---
      accumulate_en = (step != 0);
      $display("\n===== TILE STEP %0d =====", step);
      $display("accumulate_en = %b", accumulate_en);

      // --- [MOD] 3️⃣ 准备 acc_in_vec ---
      if (accumulate_en) begin
          for (i=0; i<TILE_SIZE; i++)
              acc_in_vec[i] = $signed({{(ACC_WIDTH-DATA_WIDTH){result_buffer[i][DATA_WIDTH-1]}},
                                       result_buffer[i]}) <<< FRAC_BITS;
      end else begin
          for (i=0; i<TILE_SIZE; i++)
              acc_in_vec[i] = 0;
      end

      // --- [MOD] 4️⃣ 连续两拍输入 valid_in ---
      @(posedge clk);
      valid_in = 1;
      repeat(TILE_SIZE) @(posedge clk);
      valid_in = 0;

      // --- [MOD] 5️⃣ 等待 tile 完成 ---
      @(posedge done_tile);
      @(posedge clk);

      // --- [MOD] 6️⃣ 更新 result_buffer 用于下一 tile ---
      for (i=0;i<TILE_SIZE;i++)
          result_buffer[i] = result_out_vec[i];

      // Debug print
      $display("Tile %0d done -> result_out_vec: [%0d, %0d]",
               step, result_out_vec[0], result_out_vec[1]);
      $display("Updated result_buffer: [%0d, %0d]",
               result_buffer[0], result_buffer[1]);
    end

    // ============================================================
    // Reference Computation (Q8.8)
    // ============================================================
    ref_y0 = ((1<<<FRAC_BITS)*(1<<<FRAC_BITS) +
              (2<<<FRAC_BITS)*(2<<<FRAC_BITS) +
              (3<<<FRAC_BITS)*(3<<<FRAC_BITS) +
              (4<<<FRAC_BITS)*(4<<<FRAC_BITS)) >>> FRAC_BITS;

    ref_y1 = ((5<<<FRAC_BITS)*(1<<<FRAC_BITS) +
              (6<<<FRAC_BITS)*(2<<<FRAC_BITS) +
              (7<<<FRAC_BITS)*(3<<<FRAC_BITS) +
              (8<<<FRAC_BITS)*(4<<<FRAC_BITS)) >>> FRAC_BITS;

    // ============================================================
    // Compare Results
    // ============================================================
    $display("\n===== FINAL OUTPUT =====");
    $display("HW Row0 = %0d  REF = %0d", result_out_vec[0], ref_y0);
    $display("HW Row1 = %0d  REF = %0d", result_out_vec[1], ref_y1);

    if ((result_out_vec[0] == ref_y0) && (result_out_vec[1] == ref_y1))
      $display("[PASS] MAC accumulation matches reference.");
    else
      $display("[FAIL] MAC accumulation mismatch.");

    $finish;
  end

endmodule
