//---------------------------------------------------------------
// Module: controller_top
// Function:
//   Simple controller for MAC test with 12-bank WBUF + pipeline_4array_with_reduction
//   - Generates sequential bank access and valid_in timing
//   - Reads 4×4×4 weights from WBUF, feeds into pipeline array
//   - Feeds fixed input vectors for MAC test
//   - Monitors valid_reduced output for verification
//---------------------------------------------------------------
module controller_top #(
    parameter int TILE_SIZE  = 4,
    parameter int DATA_WIDTH = 16,
    parameter int ACC_WIDTH  = 32,
    parameter int ADDR_W     = 10,
    parameter int N_BANK     = 12
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,                      // 启动信号
    output logic finish,                     // 计算完成
    output logic valid_reduced,              // reduction 输出有效
    output logic signed [ACC_WIDTH-1:0] reduced_vec [TILE_SIZE-1:0]
);

    // ===========================================================
    // 1. 控制状态与计数器
    // ===========================================================
    typedef enum logic [1:0] {IDLE, LOAD, CALC, DONE} state_t;
    state_t state;
    int cycle_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            cycle_cnt <= 0;
            finish <= 0;
        end
        else begin
            case (state)
                IDLE: if (start) begin
                    state <= CALC;
                    cycle_cnt <= 1;
                    finish <= 0;
                end
                CALC: begin
                    if (cycle_cnt == 12) begin // 跑 12 个周期用于测试
                        state <= DONE;
                        finish <= 1;
                    end
                    else
                        cycle_cnt <= cycle_cnt + 1;
                end
                DONE: state <= DONE;
            endcase
        end
    end

    // ===========================================================
    // 2. 12-bank WBUF 实例化
    // ===========================================================
    logic [N_BANK-1:0] en_bank;
    logic [ADDR_W-1:0] addr_bank [N_BANK];
    logic [255:0]      dout_bank [N_BANK];

    multi_bank_wbuf #(
        .N_BANK (N_BANK),
        .ADDR_W (ADDR_W),
        .DATA_W (256)
    ) u_wbuf (
        .clk       (clk),
        .en_bank   (en_bank),
        .addr_bank (addr_bank),
        .dout_bank (dout_bank)
    );

    // ===========================================================
    // 3. bank 调度与地址生成
    // ===========================================================
    logic [3:0] bank_sel [4];  // 每个 array 选择的 bank id
    always_comb begin
        en_bank = '0;
        for (int i = 0; i < 4; i++) begin
            bank_sel[i] = (cycle_cnt + 3*i) % N_BANK;
            en_bank[ bank_sel[i] ] = (state == CALC);
            addr_bank[ bank_sel[i] ] = cycle_cnt[ADDR_W-1:0];
        end
    end

    // ===========================================================
    // 4. 从每个 bank 输出拆出 4×4 block 送阵列
    // ===========================================================
    logic signed [DATA_WIDTH-1:0] A_mat [4][TILE_SIZE-1:0][TILE_SIZE-1:0];

    always_comb begin
        for (int a = 0; a < 4; a++) begin
            for (int i = 0; i < TILE_SIZE; i++) begin
                for (int j = 0; j < TILE_SIZE; j++) begin
                    A_mat[a][i][j] = dout_bank[ bank_sel[a] ][(i*4+j)*DATA_WIDTH +: DATA_WIDTH];
                end
            end
        end
    end

    // ===========================================================
    // 5. 输入向量（FBUF）模拟，简单固定值
    // ===========================================================
    logic signed [DATA_WIDTH-1:0] B0_vec [TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] B1_vec [TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] B2_vec [TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] B3_vec [TILE_SIZE-1:0];

    always_comb begin
        for (int i = 0; i < TILE_SIZE; i++) begin
            B0_vec[i] = 16'sd1 * (i+1);
            B1_vec[i] = 16'sd2 * (i+1);
            B2_vec[i] = 16'sd3 * (i+1);
            B3_vec[i] = 16'sd4 * (i+1);
        end
    end

    // ===========================================================
    // 6. 启动 pipeline_4array_with_reduction
    // ===========================================================
    logic [2:0] mode = 3'b000; // MAC 模式
    logic valid_in;

    assign valid_in = (state == CALC);

    pipeline_4array_with_reduction #(
        .TILE_SIZE (TILE_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH)
    ) u_pipe4 (
        .clk(clk),
        .rst_n(rst_n),
        .mode(mode),
        .valid_in(valid_in),

        // A 矩阵输入
        .A0_mat(A_mat[0]),
        .A1_mat(A_mat[1]),
        .A2_mat(A_mat[2]),
        .A3_mat(A_mat[3]),

        // B 向量输入
        .B0_vec(B0_vec),
        .B1_vec(B1_vec),
        .B2_vec(B2_vec),
        .B3_vec(B3_vec),

        // 输出
        .reduced_vec(reduced_vec),
        .valid_reduced(valid_reduced)
    );

endmodule
