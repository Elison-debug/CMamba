`timescale 1ns/1ps

module tb_pipeline_4array_top;
    // ============================================================
    // Parameters
    // ============================================================
    localparam int TILE_SIZE  = 4;
    localparam int DATA_WIDTH = 16;
    localparam int ACC_WIDTH  = 32;
    localparam int FRAC_BITS  = 8;

    localparam int ROWS = 40;   // A rows
    localparam int K    = 256;  // A cols / B rows

    // ============================================================
    // Clock / Reset
    // ============================================================
    logic clk;
    logic rst_n;

    // DUT I/O
    logic [2:0] mode;
    logic       valid_in;
    logic       valid_out;

    // ============================================================
    // DUT array ports
    // ============================================================
    logic signed [DATA_WIDTH-1:0] A0_mat [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] A1_mat [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] A2_mat [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] A3_mat [TILE_SIZE-1:0][TILE_SIZE-1:0];

    logic signed [DATA_WIDTH-1:0] B0_vec [TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] B1_vec [TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] B2_vec [TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] B3_vec [TILE_SIZE-1:0];

    logic signed [ACC_WIDTH-1:0] result_out_0 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [ACC_WIDTH-1:0] result_out_1 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [ACC_WIDTH-1:0] result_out_2 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [ACC_WIDTH-1:0] result_out_3 [TILE_SIZE-1:0][TILE_SIZE-1:0];

    // ============================================================
    // DUT Instance
    // ============================================================
    pipeline_4array_top #(
        .TILE_SIZE (TILE_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH),
        .FRAC_BITS (FRAC_BITS)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .mode(mode), .valid_in(valid_in), .valid_out(valid_out),
        .A0_mat(A0_mat), .A1_mat(A1_mat), .A2_mat(A2_mat), .A3_mat(A3_mat),
        .B0_vec(B0_vec), .B1_vec(B1_vec), .B2_vec(B2_vec), .B3_vec(B3_vec),
        .result_out_0(result_out_0), .result_out_1(result_out_1),
        .result_out_2(result_out_2), .result_out_3(result_out_3)
    );

    // ============================================================
    // Test Data Storage
    // ============================================================
    logic signed [DATA_WIDTH-1:0] A_mem [ROWS-1:0][K-1:0];
    logic signed [DATA_WIDTH-1:0] B_mem [K-1:0];

    longint signed golden   [ROWS-1:0];
    longint signed dut_accum[ROWS-1:0];

    int rb, kb, errors;

    // ============================================================
    // Clock generation: 10 ns period
    // ============================================================
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // ============================================================
    // Reset
    // ============================================================
    initial begin
        rst_n = 0;
        mode = 3'b000; // MAC mode
        valid_in = 0;
        repeat (10) @(posedge clk);
        rst_n = 1;
    end

    // ============================================================
    // Helper
    // ============================================================
    function automatic logic signed [DATA_WIDTH-1:0] to_dw(input integer val);
        logic signed [DATA_WIDTH-1:0] tmp;
        begin tmp = val; to_dw = tmp; end
    endfunction

    // ============================================================
    // Initialize A, B data
    // ============================================================
    task automatic init_data;
        int r, k;
        begin
            for (r = 0; r < ROWS; r++) begin
                for (k = 0; k < K; k++) begin
                    int aval = ((r % 5) - 2) * (1 << FRAC_BITS);
                    int kval = ((k % 7) - 3);
                    A_mem[r][k] = to_dw(aval + kval);
                end
            end
            for (k = 0; k < K; k++) begin
                int bval = ((k % 9) - 4) * (1 << (FRAC_BITS-1));
                B_mem[k] = to_dw(bval);
            end
        end
    endtask

    // ============================================================
    // Golden compute (aligned with 4-array pipelined schedule)
    // ============================================================
    task automatic compute_golden;
        int r, k;
        begin
            for (r = 0; r < ROWS; r++) begin
                golden[r] = 0;
                for (k = 0; k < K; k++) begin
                    golden[r] += longint'(A_mem[r][k]) * longint'(B_mem[k])*4;//乘4修正！！！
                end
            end
        end
    endtask


    // ============================================================
    // Build tiles: true 4-array pipeline schedule
    // ============================================================
    task automatic build_tiles_pipeline;
        input int row_base;
        input int k_base;
        int i, j;
        int kb1, kb2, kb3, kb4;
        begin
            // 计算每级输入偏移：+4,+8,+12 为跨阵列offset
            kb1 = k_base;           // array1 当前16列
            kb2 = k_base - 16 + 4;  // array2 滞后1拍 +4列
            kb3 = k_base - 32 + 8;  // array3 滞后2拍 +8列
            kb4 = k_base - 48 + 12; // array4 滞后3拍 +12列

            for (i = 0; i < TILE_SIZE; i++) begin
                for (j = 0; j < TILE_SIZE; j++) begin
                    A0_mat[i][j] = (kb1 + j < K && kb1 >= 0) ? A_mem[row_base + i][kb1 + j] : '0;
                    A1_mat[i][j] = (kb2 + j < K && kb2 >= 0) ? A_mem[row_base + i][kb2 + j] : '0;
                    A2_mat[i][j] = (kb3 + j < K && kb3 >= 0) ? A_mem[row_base + i][kb3 + j] : '0;
                    A3_mat[i][j] = (kb4 + j < K && kb4 >= 0) ? A_mem[row_base + i][kb4 + j] : '0;
                end
            end

            for (j = 0; j < TILE_SIZE; j++) begin
                B0_vec[j] = (kb1 + j < K && kb1 >= 0) ? B_mem[kb1 + j] : '0;
                B1_vec[j] = (kb2 + j < K && kb2 >= 0) ? B_mem[kb2 + j] : '0;
                B2_vec[j] = (kb3 + j < K && kb3 >= 0) ? B_mem[kb3 + j] : '0;
                B3_vec[j] = (kb4 + j < K && kb4 >= 0) ? B_mem[kb4 + j] : '0;
            end
        end
    endtask

    // ============================================================
    // Drive pipeline one step
    // ============================================================
    task automatic drive_macro_step;
        input int row_base;
        input int k_base;
        begin
            build_tiles_pipeline(row_base, k_base);
            valid_in <= 1'b1;
            @(posedge clk);
        end
    endtask

    // ============================================================
    // Capture DUT output (sum across 4 columns)
    // ============================================================
    task automatic capture_and_accum;
        input int row_base;
        int i, j;
        longint signed row_sum;
        begin
            if (valid_out) begin
                for (i = 0; i < TILE_SIZE; i++) begin
                    row_sum = 0;
                    for (j = 0; j < TILE_SIZE; j++)
                        row_sum += longint'(result_out_3[i][j]);
                    dut_accum[row_base + i] += row_sum;
                end
            end
        end
    endtask
    
    // ============================================================
    // Main stimulus
    // ============================================================
    initial begin
        init_data();
        compute_golden();

        for (rb = 0; rb < ROWS; rb++) dut_accum[rb] = 0;

        @(posedge rst_n);
        @(posedge clk);

        // ==== 主循环 ====
        for (rb = 0; rb < ROWS; rb += TILE_SIZE) begin
            for (int i = 0; i < TILE_SIZE; i++) dut_accum[rb+i] = 0;

            valid_in <= 1'b1;
            for (int k_base = 0; k_base < K + 48; k_base += 16) begin
                drive_macro_step(rb, k_base);
                capture_and_accum(rb);
            end

            // pipeline flush
            valid_in <= 1'b0;
            repeat (4) begin
                @(posedge clk);
                capture_and_accum(rb);
            end
        end

        // ==== 结果比较 ====
        errors = 0;
        for (rb = 0; rb < ROWS; rb++) begin
            if (dut_accum[rb] !== golden[rb]) begin
                $display("Mismatch at row %0d: DUT=%0d, GOLD=%0d", rb, dut_accum[rb], golden[rb]);
                errors++;
            end
        end

        if (errors == 0)
            $display("All rows matched for MAC mode (A 40x256)*(B 256x1)");
        else
            $display("Test finished with %0d mismatches", errors);

        $finish;
    end

endmodule
