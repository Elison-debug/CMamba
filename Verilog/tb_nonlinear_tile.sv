//---------------------------------------------------------------
// Testbench : tb_nonlinear_tile_shared
// Function  : Verify nonlinear_tile_shared (Softplus / Exp)
// Author    : Shengjie Chen
// Description:
//   - Tests both Exp (256x16) and Softplus (256x1)
//   - Includes a final summary of mismatch counts for all test modes.
//   - Uses done_tile handshake for tile streaming
//---------------------------------------------------------------
`timescale 1ns/1ps
module tb_nonlinear_tile_shared;

    // ============================================================
    // Parameters
    // ============================================================
    localparam DATA_WIDTH = 16;
    localparam TILE_SIZE  = 16;
    // MODIFICATION: 增加小数位宽到12位
    localparam FRAC_BITS  = 12;
    localparam D_INNER    = 256;
    localparam D_STATE    = 16;
    localparam N_TILE     = D_INNER / TILE_SIZE; // 16
    localparam real TOL   = 0.5;  // 理论值与硬件计算结果的容忍度

    // ============================================================
    // DUT I/O
    // ============================================================
    logic                    clk, rst_n;
    logic                    valid_in;
    logic                    mode; // 0=Softplus, 1=Exp
    logic signed [DATA_WIDTH-1:0] mid_res_vec [0:TILE_SIZE-1];
    logic signed [DATA_WIDTH-1:0] mid_res_mat [0:TILE_SIZE-1][0:TILE_SIZE-1];
    logic signed [DATA_WIDTH-1:0] y_vec [0:TILE_SIZE-1];
    logic signed [DATA_WIDTH-1:0] y_mat [0:TILE_SIZE-1][0:TILE_SIZE-1];
    logic                    valid_out;
    logic                    done_tile;

    // ============================================================
    // DUT Instance
    // ============================================================
    nonlinear_tile #(
        .DATA_WIDTH (DATA_WIDTH),
        .TILE_SIZE  (TILE_SIZE),
        // MODIFICATION: 将 FRAC_BITS 参数传递给 DUT
        .FRAC_BITS  (FRAC_BITS)
    ) DUT (
        .clk         (clk),
        .rst_n       (rst_n),
        .valid_in    (valid_in),
        .mode        (mode),
        .mid_res_vec (mid_res_vec),
        .mid_res_mat (mid_res_mat),
        .y_vec       (y_vec),
        .y_mat       (y_mat),
        .valid_out   (valid_out),
        .done_tile   (done_tile)
    );

    // ============================================================
    // Clock Generation
    // ============================================================
    initial clk = 0;
    always #5 clk = ~clk; // 100MHz

    // ============================================================
    // Utility Functions (Q4.12 <-> float)
    // ============================================================
    // MODIFICATION: 调整转换函数以匹配新的小数位宽
    function real q_to_float(input logic signed [DATA_WIDTH-1:0] qval);
        q_to_float = qval / 2.0**(FRAC_BITS);
    endfunction

    function logic signed [DATA_WIDTH-1:0] float_to_q(input real fval);
        float_to_q = $rtoi(fval * 2.0**(FRAC_BITS));
    endfunction

    // ============================================================
    // Main Test Process
    // ============================================================
    integer i, j, t;
    real xin, hw_val, th_val, diff;
    
    integer exp_mismatch_cnt;
    integer softplus_mismatch_cnt;
    integer exp_total_cnt, softplus_total_cnt;

    initial begin
        $display("\n==============================================");
        $display(" Testbench: tb_nonlinear_tile_shared");
        $display("==============================================");

        exp_mismatch_cnt = 0;
        softplus_mismatch_cnt = 0;
        exp_total_cnt = 0;
        softplus_total_cnt = 0;

        rst_n = 0;
        valid_in = 0;
        mode = 1'b0;
        #25 rst_n = 1;
        #10;

        // ============================================================
        // 1️⃣ EXP 模式测试 (mode=1)
        // ============================================================
        $display("---- [EXP Mode Test] ----");
        mode = 1'b1;

        for (t = 0; t < N_TILE; t++) begin
            for (i = 0; i < TILE_SIZE; i++) begin
                for (j = 0; j < TILE_SIZE; j++) begin
                    real val = -1.0 + 2.0 * (t*TILE_SIZE + i) / D_INNER;
                    // MODIFICATION: 使用新的转换函数
                    mid_res_mat[i][j] = float_to_q(val);
                end
            end

            valid_in = 1;
            @(posedge clk);
            valid_in = 0;
            
            wait(valid_out == 1'b1);
            
            @(posedge clk);
            wait(done_tile == 1'b1);
            
            for (i = 0; i < TILE_SIZE; i++) begin
                for (j = 0; j < TILE_SIZE; j++) begin
                    exp_total_cnt++;
                    // MODIFICATION: 使用新的转换函数
                    xin = q_to_float(mid_res_mat[i][j]);
                    hw_val = q_to_float(y_mat[i][j]);
                    th_val = $exp(xin);
                    diff = (hw_val > th_val) ? (hw_val - th_val) : (th_val - hw_val);
                    if (diff > TOL) begin
                        $display("Mismatch tile=%0d (%0d,%0d): x=%.3f hw=%.3f th=%.3f delta=%.3f",
                                 t, i, j, xin, hw_val, th_val, diff);
                        exp_mismatch_cnt++;
                    end
                end
            end
        end

        // ============================================================
        // 2️⃣ SOFTPLUS 模式测试 (mode=0)
        // ============================================================
        $display("---- [SOFTPLUS Mode Test] ----");
        mode = 1'b0;

        for (t = 0; t < N_TILE; t++) begin
            for (i = 0; i < TILE_SIZE; i++) begin
                real val = -1.0 + 2.0 * (t*TILE_SIZE + i) / D_INNER;
                // MODIFICATION: 使用新的转换函数
                mid_res_vec[i] = float_to_q(val);
            end

            valid_in = 1; 
            @(posedge clk);
            valid_in = 0;
            
            wait(valid_out == 1'b1);
            
            @(posedge clk);
            wait(done_tile == 1'b1);

            for (i = 0; i < TILE_SIZE; i++) begin
                softplus_total_cnt++;
                // MODIFICATION: 使用新的转换函数
                xin = q_to_float(mid_res_vec[i]);
                hw_val = q_to_float(y_vec[i]);
                th_val = $ln(1.0 + $exp(xin));
                diff = (hw_val > th_val) ? (hw_val - th_val) : (th_val - hw_val);
                if (diff > TOL) begin
                    $display("Mismatch tile=%0d i=%0d: x=%.3f hw=%.3f th=%.3f delta=%.3f",
                             t, i, xin, hw_val, th_val, diff);
                    softplus_mismatch_cnt++;
                end
            end
        end

        // ============================================================
        // END - 最终总结
        // ============================================================
        $display("\n==============================================");
        $display(" Simulation completed.");
        $display("==============================================\n");
        $display("Final Test Summary:");
        $display("-------------------");
        $display("EXP Mode Mismatches: %0d / %0d (%.2f%%)",
                  exp_mismatch_cnt, exp_total_cnt,
                  (exp_mismatch_cnt * 100.0) / exp_total_cnt);
        $display("Softplus Mode Mismatches: %0d / %0d (%.2f%%)",
                  softplus_mismatch_cnt, softplus_total_cnt,
                  (softplus_mismatch_cnt * 100.0) / softplus_total_cnt);
        $display("-------------------");
        $display("Total Mismatches: %0d / %0d (%.2f%%)",
                  exp_mismatch_cnt + softplus_mismatch_cnt,
                  exp_total_cnt + softplus_total_cnt,
                  (exp_mismatch_cnt + softplus_mismatch_cnt) * 100.0 / (exp_total_cnt + softplus_total_cnt));
        
        #20 $finish;
    end

endmodule