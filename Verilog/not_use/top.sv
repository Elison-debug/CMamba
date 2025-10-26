//---------------------------------------------------------------
// Module: top_level_syntest (V7 - 移除了 rsta/rstb 端口)
//
// 修正：
//   1. 实例化 'fake_ram'
//   2. 移除了 'rsta' 和 'rstb' 端口连接，以匹配您的综合日志
//---------------------------------------------------------------
module top_level_syntest (
    input  logic clk,
    input  logic rst_n, // 您的顶层复位是低电平有效

    // --- 阵列的输出 (防止被优化掉) ---
    output logic signed [15:0] result_vec_out [15:0],
    output logic signed [15:0] result_mat_out [15:0][15:0],
    output logic             valid_out_sig,
    output logic             done_tile_sig,
    output logic             out_shape_flag_sig
);

    // --- 参数定义 ---
    parameter DATA_WIDTH = 16;
    parameter ACC_WIDTH  = 32;
    parameter FRAC_BITS  = 8;
    parameter TILE_SIZE  = 16;
    
    parameter BRAM_WIDTH = 4096;
    parameter BRAM_ADDR_WIDTH = 4; // 对应 [3:0]

    // --- BRAM 接口信号 ---
    logic [BRAM_ADDR_WIDTH-1:0]   read_addr;
    logic signed [BRAM_WIDTH-1:0] a_bram_dout; // 来自 fake_ram 1 的输出
    logic signed [BRAM_WIDTH-1:0] b_bram_dout; // 来自 fake_ram 2 的输出

    // --- 阵列接口信号 ---
    logic             valid_in_reg;
    logic [2:0]       mode_reg;
    logic             accumulate_en_reg;
    
    // 这些是连接 BRAM输出 和 阵列输入 的“物理线”
    logic signed [DATA_WIDTH-1:0] a_in_wires [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] b_mat_wires [TILE_SIZE-1:0][TILE_SIZE-1:0];

    // b_vec 和 acc_in 仍然是寄存器
    logic signed [DATA_WIDTH-1:0] b_vec_buf [TILE_SIZE-1:0];
    logic signed [ACC_WIDTH-1:0]  acc_in_vec_buf [TILE_SIZE-1:0];

    // ============================================================
    //               *** 1. 实例化您的 fake_ram IP ***
    // ============================================================

    // BRAM for a_in (实例化 'fake_ram')
    fake_ram U_BRAM_A (
        // --- Port A (Write - 未使用) ---
        .clka    (clk),
        .ena     (1'b0), // 不使能Port A
        .wea     (1'b0), 
        .addra   (4'b0),
        .dina    (4096'b0),
        .douta   (), // Port A 的读输出未连接
        // .rsta    (rst_a), // <-- 移除，因为您的IP没有这个端口

        // --- Port B (Read - 正在使用) ---
        .clkb    (clk),
        .enb     (1'b1), // 始终读取
        .web     (1'b0),
        .addrb   (read_addr),
        .dinb    (4096'b0),
        .doutb   (a_bram_dout)
        // .rstb    (rst_b)  // <-- 移除，因为您的IP没有这个端口
    );

    // BRAM for b_mat (实例化 'fake_ram')
    fake_ram U_BRAM_B (
        // --- Port A (Write - 未使用) ---
        .clka    (clk),
        .ena     (1'b0), // 不使能Port A
        .wea     (1'b0), 
        .addra   (4'b0),
        .dina    (4096'b0),
        .douta   (), 
        // .rsta    (rst_a), // <-- 移除，因为您的IP没有这个端口

        // --- Port B (Read - 正在使用) ---
        .clkb    (clk),
        .enb     (1'b1), // 始终读取
        .web     (1'b0),
        .addrb   (read_addr),
        .dinb    (4096'b0),
        .doutb   (b_bram_dout)
        // .rstb    (rst_b)  // <-- 移除，因为您的IP没有这个端口
    );

    // ============================================================
    //    *** 2. 关键：4096-bit 扇出 (Combinational Fan-out) ***
    // ============================================================
    genvar i, j;
    generate
        for (i = 0; i < TILE_SIZE; i++) begin
            for (j = 0; j < TILE_SIZE; j++) begin
                localparam int offset = (i * TILE_SIZE) + j;
                assign a_in_wires[i][j] = a_bram_dout[(offset + 1) * DATA_WIDTH - 1 -: DATA_WIDTH];
                assign b_mat_wires[i][j] = b_bram_dout[(offset + 1) * DATA_WIDTH - 1 -: DATA_WIDTH];
            end
        end
    endgenerate

    // ============================================================
    //               *** 3. 实例化您的阵列 ***
    // ============================================================
    recfg_array_new #(
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH),
        .FRAC_BITS  (FRAC_BITS),
        .TILE_SIZE  (TILE_SIZE)
    ) UUT (
        .clk          (clk),
        .rst_n        (rst_n),
        .valid_in     (valid_in_reg),
        .mode         (mode_reg),
        .accumulate_en(accumulate_en_reg),
        .a_in         (a_in_wires),
        .b_vec        (b_vec_buf),
        .b_mat        (b_mat_wires),
        .acc_in_vec   (acc_in_vec_buf),
        .result_out_vec (result_vec_out),
        .result_out_mat (result_mat_out),
        .valid_out      (valid_out_sig),
        .done_tile      (done_tile_sig),
        .out_shape_flag (out_shape_flag_sig)
    );

    // ============================================================
    //               *** 4. 简单的激励逻辑 ***
    // ============================================================
    
    logic valid_in_pipe; // 用于匹配BRAM的1周期读延迟
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_addr <= 4'b0;
            valid_in_pipe <= 1'b0;
            mode_reg <= 3'b000; // MAC
            accumulate_en_reg <= 1'b0;
        end else begin
            read_addr <= read_addr + 1; // 循环读取BRAM地址
            valid_in_pipe <= 1'b1;
        end
    end
    
    // valid_in_reg 必须与BRAM输出的数据对齐 (假设1周期读延迟)
    assign valid_in_reg = valid_in_pipe; 

endmodule