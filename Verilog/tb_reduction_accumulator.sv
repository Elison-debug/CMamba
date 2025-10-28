`timescale 1ns/1ps
//---------------------------------------------------------------
// Testbench for reduction_accumulator
// 模拟 array4 在 cycle4~7 输出矩阵
// 检查规约 + 累加结果
//---------------------------------------------------------------
module tb_reduction_accumulator;

    // ===========================================================
    // 参数定义
    // ===========================================================
    localparam int TILE_SIZE  = 4;
    localparam int ACC_WIDTH  = 32;
    localparam CLK_PERIOD = 10;

    // ===========================================================
    // 信号声明
    // ===========================================================
    logic clk, rst_n;
    logic [2:0] mode;
    logic valid_in;
    logic signed [ACC_WIDTH-1:0] mat_in [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [ACC_WIDTH-1:0] vec_out [TILE_SIZE-1:0];
    logic valid_out;

    // ===========================================================
    // DUT 实例化
    // ===========================================================
    reduction_accumulator #(
        .TILE_SIZE (TILE_SIZE),
        .ACC_WIDTH (ACC_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .mode(mode),
        .valid_in(valid_in),
        .mat_in(mat_in),
        .vec_out(vec_out),
        .valid_out(valid_out)
    );

    // ===========================================================
    // 时钟 & 复位
    // ===========================================================
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    initial begin
        rst_n = 0;
        mode  = 3'b000; // MAC 模式
        valid_in = 0;
        #(5*CLK_PERIOD);
        rst_n = 1;
    end

    // ===========================================================
    // 模拟 array4 输出矩阵
    // 从 cycle4 开始每拍一个 tile，共 4 个 tile
    // 每个矩阵元素简单递增：mat_in[i][j] = tile_id*100 + i*10 + j
    // ===========================================================
    int tile_id;
    initial begin
        // 等待复位完成
        @(posedge rst_n);
        #(3*CLK_PERIOD); // 模拟 pipeline_4array_top 前3拍延迟

        for (tile_id = 0; tile_id < 4; tile_id++) begin
            @(negedge clk);
            valid_in = 1'b1;
            // 填入矩阵数据
            for (int i = 0; i < TILE_SIZE; i++) begin
                for (int j = 0; j < TILE_SIZE; j++) begin
                    mat_in[i][j] = tile_id*100 + i*10 + j;
                end
            end
        end

        // 停止输入
        @(negedge clk);
        valid_in = 1'b0;
    end

    // ===========================================================
    // 输出监视器
    // ===========================================================
    always_ff @(posedge clk) begin
        if (valid_in) begin
            $display("Cycle %0t : Input Tile[%0d]", $time/CLK_PERIOD, tile_id);
        end
        if (valid_out) begin
            $display("Cycle %0t : Output Vector = [%0d, %0d, %0d, %0d]",
                     $time/CLK_PERIOD, 
                     vec_out[0], vec_out[1], vec_out[2], vec_out[3]);
        end
    end

    // ===========================================================
    // 仿真结束
    // ===========================================================
    initial begin
        #(20*CLK_PERIOD);
        $display("Simulation finished at %0t ns", $time);
        $finish;
    end

endmodule
