//---------------------------------------------------------------
// Module : exp_sp_cell
// Function : Approximate exp(x) and softplus(x) with valid timing
// Author : Shengjie Chen
// Description : 
//   - Supports exp(x) and Softplus(x) using shared linear approx.
//   - Parameterized segments for flexible approximation
//   - Q4.12 fixed-point, one-cycle latency (valid_in -> valid_out)
//   - mode bit: 0=Softplus, 1=Exp
//---------------------------------------------------------------
module exp_sp_cell #(
    parameter DATA_WIDTH = 16,
    parameter FRAC_BITS  = 12,
    parameter SEGMENTS   = 16
)(
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic                   valid_in,
    input  logic signed [DATA_WIDTH-1:0] x_in,
    input  logic                   mode,   // 0=Softplus, 1=Exp
    output logic signed [DATA_WIDTH-1:0] y_out,
    output logic                   valid_out
);
    // ============================================================
    // Internal signals and parameters
    // ============================================================
    localparam signed [15:0] LOG2E_Q4_12 = 16'd5917;
    localparam signed [15:0] SAT_POS     = {1'b0, {(DATA_WIDTH-1){1'b1}}};
    localparam signed [15:0] SAT_NEG     = {1'b1, {(DATA_WIDTH-1){1'b0}}};

    localparam U_INT_WIDTH = DATA_WIDTH - FRAC_BITS;
    localparam V_FRAC_WIDTH = FRAC_BITS;
    localparam SEG_IDX_WIDTH = $clog2(SEGMENTS);
    
    logic signed [31:0] z_full;
    logic signed [DATA_WIDTH-1:0] z;
    
    logic signed [U_INT_WIDTH-1:0] u;
    logic signed [V_FRAC_WIDTH-1:0] v;
    
    logic [SEG_IDX_WIDTH-1:0] seg_idx;
    
    logic signed [DATA_WIDTH-1:0] a_table [0:SEGMENTS-1];
    logic signed [DATA_WIDTH-1:0] b_table [0:SEGMENTS-1];
    
    // MODIFICATION: 定义一个Q4.12格式的v_q4_12
    logic signed [DATA_WIDTH-1:0] v_q4_12;
    
    // a*v是Q4.12*Q4.12，结果是Q8.24
    logic signed [DATA_WIDTH+FRAC_BITS-1:0] a_mult_v;
     logic signed [15:0] lin_approx; //linear approximation result of 2^v, a_i * v + b_i
    logic signed [31:0] exp_full, exp_neg_full;
    logic signed [15:0] exp_val, exp_neg_val, y_comb;
    logic valid_out_n;

    logic [31:0] exp_full_u, exp_neg_full_u;
    logic signed [DATA_WIDTH:0] softplus_sum;

    initial begin
        a_table[0] = 16'd2902 ; b_table[0] = 16'd4096;
        a_table[1] = 16'd3030 ; b_table[1] = 16'd4088;
        a_table[2] = 16'd3164 ; b_table[2] = 16'd4071;
        a_table[3] = 16'd3304 ; b_table[3] = 16'd4045;
        a_table[4] = 16'd3451 ; b_table[4] = 16'd4008;
        a_table[5] = 16'd3603 ; b_table[5] = 16'd3961;
        a_table[6] = 16'd3763 ; b_table[6] = 16'd3901;
        a_table[7] = 16'd3929 ; b_table[7] = 16'd3828;
        a_table[8] = 16'd4103 ; b_table[8] = 16'd3741;
        a_table[9] = 16'd4285 ; b_table[9] = 16'd3639;
        a_table[10] = 16'd4475 ; b_table[10] = 16'd3520;
        a_table[11] = 16'd4673 ; b_table[11] = 16'd3384;
        a_table[12] = 16'd4880 ; b_table[12] = 16'd3229;
        a_table[13] = 16'd5096 ; b_table[13] = 16'd3053;
        a_table[14] = 16'd5321 ; b_table[14] = 16'd2856;
        a_table[15] = 16'd5557 ; b_table[15] = 16'd2635;
    end

    always_comb begin
        y_comb      = '0;
        valid_out_n = 1'b0;

        if (valid_in) begin
            z_full = x_in * LOG2E_Q4_12;
            z      = $signed(z_full) >>> FRAC_BITS;
            
            u      = z[DATA_WIDTH-1:FRAC_BITS];
            v      = z[FRAC_BITS-1:0];
            
            seg_idx = v[V_FRAC_WIDTH-1 : V_FRAC_WIDTH-SEG_IDX_WIDTH];
            
            // MODIFICATION: 修正 lin_approx 的计算逻辑以保证定点数一致性
            // 1. 将Q0.12格式的v左移4位，使其成为Q4.12格式
            v_q4_12 = { {U_INT_WIDTH{v[V_FRAC_WIDTH-1]}}, v }; // 算术扩展
            
            // 2. 进行同格式Q4.12乘法，结果为Q8.24
            a_mult_v = $signed(a_table[seg_idx]) * $signed(v_q4_12);
            
            // 3. 将结果右移FRAC_BITS(12)位，转换为Q4.12
            // 4. 与b_table进行加法，得到最终的lin_approx (Q4.12)
            lin_approx = (a_mult_v >>> FRAC_BITS) + $signed(b_table[seg_idx]);

            exp_full = (u >= 0) ? ($unsigned(lin_approx) <<< u) : ($unsigned(lin_approx) >>> (-u));
            exp_neg_full = (u >= 0) ? ($unsigned(lin_approx) >>> u) : ($unsigned(lin_approx) <<< (-u));

            exp_val     = (exp_full > SAT_POS) ? SAT_POS : (exp_full < SAT_NEG) ? SAT_NEG : $signed(exp_full[15:0]);
            exp_neg_val = (exp_neg_full > SAT_POS) ? SAT_POS : (exp_neg_full < SAT_NEG) ? SAT_NEG : $signed(exp_neg_full[15:0]);
            
            case (mode)
                1'b1: y_comb = exp_val;
                1'b0: begin
                    if (x_in[DATA_WIDTH-1])
                        y_comb = exp_val;
                    else begin
                        softplus_sum = $signed(x_in) + $signed(exp_neg_val);
                        if (softplus_sum > SAT_POS)
                            y_comb = SAT_POS;
                        else if (softplus_sum < SAT_NEG)
                            y_comb = SAT_NEG;
                        else
                            y_comb = softplus_sum[15:0];
                    end
                end
                default: y_comb = '0;
            endcase
            valid_out_n = 1'b1;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_out     <= '0;
            valid_out <= 1'b0;
        end else begin
            y_out     <= y_comb;
            valid_out <= valid_out_n;
        end
    end
endmodule