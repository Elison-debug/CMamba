`timescale 1ns/1ps
// golden model for pipeline_4array_with_reduction is not correct
module tb_pipeline_4array_with_reduction;

    localparam int TILE_SIZE  = 4;
    localparam int DATA_WIDTH = 16;
    localparam int ACC_WIDTH  = 32;
    localparam int FRAC_BITS  = 8;
    localparam int ROWS = 40;
    localparam int K    = 256;

    logic clk;
    logic rst_n;
    logic [2:0] mode;
    logic valid_in, valid_out;

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

    logic done_tile;  // <<< 修改1: 新增信号观测 tile 完成 >>>

    // DUT 实例化
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
        .result_out_2(result_out_2), .result_out_3(result_out_3),
        .done_tile(done_tile)  // <<< 修改2: 连接 done_tile >>>
    );

    // ---------------- 数据初始化 ----------------
    logic signed [DATA_WIDTH-1:0] A_mem [ROWS-1:0][K-1:0];
    logic signed [DATA_WIDTH-1:0] B_mem [K-1:0];
    longint signed golden [ROWS-1:0];
    longint signed dut_accum[ROWS-1:0];

    int rb, kb, errors;

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        rst_n = 0;
        mode = 3'b000;
        valid_in = 0;
        repeat (10) @(posedge clk);
        rst_n = 1;
    end

    function automatic logic signed [DATA_WIDTH-1:0] to_dw(input integer val);
        logic signed [DATA_WIDTH-1:0] tmp;
        begin tmp = val; to_dw = tmp; end
    endfunction

    task automatic init_data;
        int r, k;
        begin
            for (r = 0; r < ROWS; r++) begin
                for (k = 0; k < K; k++) begin
                    A_mem[r][k] = to_dw(((r % 5) - 2) * (1 << FRAC_BITS) + ((k % 7) - 3));
                end
            end
            for (k = 0; k < K; k++)
                B_mem[k] = to_dw(((k % 9) - 4) * (1 << (FRAC_BITS-1)));
        end
    endtask

    task automatic compute_golden;
        int r, k;
        begin
            for (r = 0; r < ROWS; r++) begin
                golden[r] = 0;
                for (k = 0; k < K; k++)
                    golden[r] += longint'(A_mem[r][k]) * longint'(B_mem[k]) * 4;
            end
        end
    endtask

    // ---------------- tile 数据生成 ----------------
    task automatic build_tiles_pipeline;
        input int row_base;
        input int k_base;
        int i, j;
        int kb1, kb2, kb3, kb4;
        begin
            kb1 = k_base;
            kb2 = k_base - 16 + 4;
            kb3 = k_base - 32 + 8;
            kb4 = k_base - 48 + 12;
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

    // ---------------- 主过程 ----------------
    initial begin
        init_data();
        compute_golden();
        for (rb = 0; rb < ROWS; rb++) dut_accum[rb] = 0;

        @(posedge rst_n);
        @(posedge clk);

        valid_in <= 1'b1; // <<< 修改3: 持续高，不在 tile 之间停4拍 >>>

        for (rb = 0; rb < ROWS; rb += TILE_SIZE) begin
            for (int k_base = 0; k_base < K + 48; k_base += 16) begin
                build_tiles_pipeline(rb, k_base);
                @(posedge clk);
                capture_and_accum(rb);
            end
        end

        valid_in <= 1'b0;

        repeat (8) @(posedge clk);

        // 验证结果
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