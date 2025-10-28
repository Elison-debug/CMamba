`timescale 1ns/1ps
//---------------------------------------------------------------
// Testbench for reduction_accumulator (with clear signal)
// 模拟 array4 在 cycle4~7 输出矩阵
//---------------------------------------------------------------
module tb_reduction_accumulator;

    localparam int TILE_SIZE  = 4;
    localparam int ACC_WIDTH  = 32;
    localparam CLK_PERIOD = 10;

    logic clk, rst_n;
    logic [2:0] mode;
    logic valid_in, clear;
    logic signed [ACC_WIDTH-1:0] mat_in [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [ACC_WIDTH-1:0] vec_out [TILE_SIZE-1:0];
    logic valid_out;

    //-----------------------------------------------------------
    // DUT
    //-----------------------------------------------------------
    reduction_accumulator #(
        .TILE_SIZE (TILE_SIZE),
        .ACC_WIDTH (ACC_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .mode(mode),
        .valid_in(valid_in),
        .clear(clear),              // <<<<<< 新增清零控制
        .mat_in(mat_in),
        .vec_out(vec_out),
        .valid_out(valid_out)
    );

    //-----------------------------------------------------------
    // Clock / Reset
    //-----------------------------------------------------------
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    initial begin
        rst_n = 0;
        mode  = 3'b000; // MAC 模式
        valid_in = 0;
        clear = 0;
        #(5*CLK_PERIOD);
        rst_n = 1;
    end

    //-----------------------------------------------------------
    // Stimulus: 模拟 array4 输出 4 个 tile
    //-----------------------------------------------------------
    int tile_id;
    initial begin
        @(posedge rst_n);
        #(3*CLK_PERIOD);

        for (tile_id = 0; tile_id < 4; tile_id++) begin
            // --- 每个 tile 开始前清零一次 ---
            @(negedge clk);
            clear    = 1'b1;
            valid_in = 1'b0;
            @(negedge clk);
            clear    = 1'b0;
            valid_in = 1'b1;

            // --- 输入 1 个矩阵 (4×4) ---
            for (int i = 0; i < TILE_SIZE; i++) begin
                for (int j = 0; j < TILE_SIZE; j++) begin
                    mat_in[i][j] = tile_id*100 + i*10 + j;
                end
            end

            // 停顿一个周期观察输出
            @(negedge clk);
            valid_in = 1'b0;
        end
    end

    //-----------------------------------------------------------
    // Monitor
    //-----------------------------------------------------------
    always_ff @(posedge clk) begin
        if (valid_in)
            $display("Cycle %0t : Input Tile[%0d]", $time/CLK_PERIOD, tile_id);
        if (valid_out)
            $display("Cycle %0t : Output Vector = [%0d, %0d, %0d, %0d]",
                     $time/CLK_PERIOD,
                     vec_out[0], vec_out[1], vec_out[2], vec_out[3]);
    end

    //-----------------------------------------------------------
    // Finish
    //-----------------------------------------------------------
    initial begin
        #(40*CLK_PERIOD);
        $display("Simulation finished at %0t ns", $time);
        $finish;
    end

endmodule
