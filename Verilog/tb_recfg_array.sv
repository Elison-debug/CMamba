//---------------------------------------------------------------
// Testbench : tb_recfg_array_full_verify
// Purpose   : Verify recfg_array (modes 000, 001, 010, 011)
// Author    : Shengjie Chen
//---------------------------------------------------------------
`timescale 1ns/1ps
module tb_recfg_array;

    // ============================================================
    // Parameters
    // ============================================================
    localparam DATA_WIDTH = 16;
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
    reg  signed [DATA_WIDTH-1:0] acc_in_vec [0:TILE_SIZE-1];
    reg  signed [DATA_WIDTH-1:0] a_in  [0:TILE_SIZE-1][0:TILE_SIZE-1];
    reg  signed [DATA_WIDTH-1:0] b_vec [0:TILE_SIZE-1];
    reg  signed [DATA_WIDTH-1:0] b_mat [0:TILE_SIZE-1][0:TILE_SIZE-1];
    wire signed [DATA_WIDTH-1:0] result_out_vec [0:TILE_SIZE-1];
    wire signed [DATA_WIDTH-1:0] result_out_mat [0:TILE_SIZE-1][0:TILE_SIZE-1];
    wire valid_out, done_tile, out_shape_flag;

    // clock/reset
    initial begin clk = 0; forever #5 clk = ~clk; end
    initial begin rst_n = 0; valid_in = 0; accumulate_en = 0; mode = 3'b000;
        #50 rst_n = 1;
    end

    // ============================================================
    // DUT
    // ============================================================
    recfg_array #(.DATA_WIDTH(DATA_WIDTH), .TILE_SIZE(TILE_SIZE)) dut (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in), .mode(mode),
        .a_in(a_in), .b_vec(b_vec), .b_mat(b_mat),
        .accumulate_en(accumulate_en), .acc_in_vec(acc_in_vec),
        .result_out_vec(result_out_vec), .result_out_mat(result_out_mat),
        .valid_out(valid_out), .done_tile(done_tile), .out_shape_flag(out_shape_flag)
    );

    // ============================================================
    // Variables
    // ============================================================
    integer i, j, rb, cb;
    integer mismatch_cnt_000,mismatch_cnt_001,mismatch_cnt_010,mismatch_cnt_011;
    
    // --- mode000
    reg signed [DATA_WIDTH-1:0] W [0:OUT_SIZE-1][0:D_INNER-1];
    reg signed [DATA_WIDTH-1:0] x_t [0:D_INNER-1];
    reg signed [31:0] ref_y [0:OUT_SIZE-1];
    reg signed [DATA_WIDTH-1:0] buffer_reg [0:OUT_SIZE-1];

    // --- mode001
    reg signed [DATA_WIDTH-1:0] A [0:D_INNER-1][0:D_STATE-1];
    reg signed [DATA_WIDTH-1:0] sp_delta_t [0:D_INNER-1];
    reg signed [31:0] ref_mat [0:D_INNER-1][0:D_STATE-1];
    reg signed [DATA_WIDTH-1:0] hw_mat [0:D_INNER-1][0:D_STATE-1];

    // --- mode010
    reg signed [DATA_WIDTH-1:0] D_vec [0:D_INNER-1];
    reg signed [DATA_WIDTH-1:0] x_vec [0:D_INNER-1];
    reg signed [31:0] ref_vec [0:D_INNER-1];
    reg signed [DATA_WIDTH-1:0] hw_vec [0:D_INNER-1];

    // --- mode011
    reg signed [DATA_WIDTH-1:0] B_raw [0:D_STATE-1];
    reg signed [31:0] ref_outer [0:D_INNER-1][0:D_STATE-1];
    reg signed [DATA_WIDTH-1:0] hw_outer [0:D_INNER-1][0:D_STATE-1];

    // ============================================================
    // Stimulus
    // ============================================================
    initial begin
        @(posedge rst_n);

        // ========================================================
        // MODE000 : MAC (y = W × x_t)
        // ========================================================
        for(i=0;i<OUT_SIZE;i++)
            for(j=0;j<D_INNER;j++) W[i][j]=($random%8)-4;
        for(j=0;j<D_INNER;j++) x_t[j]=($random%8)-4;

        for(i=0;i<OUT_SIZE;i++) begin
            ref_y[i]=0;
            for(j=0;j<D_INNER;j++)
                ref_y[i]+=W[i][j]*x_t[j];
            buffer_reg[i]=0;
        end

        $display("\n===============================================");
        $display(">>> MODE000 : MAC (40×256 × 256×1)");
        $display("===============================================");
        mode=3'b000;
        for(rb=0;rb<N_ROWBLK;rb++) begin
            for(cb=0;cb<N_COLBLK;cb++) begin
                accumulate_en=(cb!=0);
                for(i=0;i<TILE_SIZE;i++) begin
                    int r=rb*TILE_SIZE+i;
                    for(j=0;j<TILE_SIZE;j++) begin
                        int c=cb*TILE_SIZE+j;
                        a_in[i][j]=(r<OUT_SIZE)?W[r][c]:0;
                    end
                end
                for(j=0;j<TILE_SIZE;j++) b_vec[j]=x_t[cb*TILE_SIZE+j];
                for(i=0;i<TILE_SIZE;i++) begin
                    int r=rb*TILE_SIZE+i;
                    acc_in_vec[i]=(r<OUT_SIZE)?buffer_reg[r]:0;
                end
                @(posedge clk); valid_in=1; repeat(TILE_SIZE)@(posedge clk);
                valid_in=0; @(posedge done_tile); @(posedge clk);
                for(i=0;i<TILE_SIZE;i++) begin
                    int r=rb*TILE_SIZE+i;
                    if(r<OUT_SIZE) buffer_reg[r]=result_out_vec[i];
                end
            end
        end

        mismatch_cnt_000=0;
        for(i=0;i<OUT_SIZE;i++)
            if(buffer_reg[i]!==ref_y[i]) mismatch_cnt_000++;
        $display("[INFO] MODE000 mismatches=%0d", mismatch_cnt_000);

        // ========================================================
        // MODE001 : EWM-Matrix (ΔA = A ⊙ spΔ_t)
        // ========================================================
        $display("\n===============================================");
        $display(">>> MODE001 : EWM-Matrix (ΔA = A ⊙ spΔ_t)");
        $display("===============================================");
        mode=3'b001; accumulate_en=0;
        for(i=0;i<D_INNER;i++) begin
            sp_delta_t[i]=($random%8)-4;
            for(j=0;j<D_STATE;j++) A[i][j]=($random%8)-4;
        end
        for(i=0;i<D_INNER;i++)
            for(j=0;j<D_STATE;j++)
                ref_mat[i][j]=A[i][j]*sp_delta_t[i];

        for(rb=0;rb<D_INNER/TILE_SIZE;rb++) begin
            for(i=0;i<TILE_SIZE;i++) begin
                int r=rb*TILE_SIZE+i;
                for(j=0;j<TILE_SIZE;j++) a_in[i][j]=A[r][j];
            end
            for(i=0;i<TILE_SIZE;i++) b_vec[i]=sp_delta_t[rb*TILE_SIZE+i];
            @(posedge clk); valid_in=1; @(posedge clk); valid_in=0;
            @(posedge done_tile); @(posedge clk);
            for(i=0;i<TILE_SIZE;i++) begin
                int r=rb*TILE_SIZE+i;
                for(j=0;j<TILE_SIZE;j++) hw_mat[r][j]=result_out_mat[i][j];
            end
        end

        mismatch_cnt_001=0;
        for(i=0;i<D_INNER;i++)
            for(j=0;j<D_STATE;j++)
                if(hw_mat[i][j]!==ref_mat[i][j]) mismatch_cnt_001++;
        $display("[INFO] MODE001 mismatches=%0d", mismatch_cnt_001);

        // ========================================================
        // MODE010 : EWM-Vector (D_x = D ⊙ x_t)
        // ========================================================
        $display("\n===============================================");
        $display(">>> MODE010 : EWM-Vector (D_x = D ⊙ x_t)");
        $display("===============================================");
        mode=3'b010; accumulate_en=0;
        for(i=0;i<D_INNER;i++) begin
            D_vec[i]=($random%8)-4;
            x_vec[i]=($random%8)-4;
            ref_vec[i]=D_vec[i]*x_vec[i];
        end
        for(rb=0;rb<D_INNER/TILE_SIZE;rb++) begin
            for(i=0;i<TILE_SIZE;i++) begin
                int r=rb*TILE_SIZE+i;
                for(j=0;j<TILE_SIZE;j++)
                    a_in[i][j]=(j==0)?D_vec[r]:0;
                b_vec[i]=x_vec[r];
            end
            @(posedge clk); valid_in=1; @(posedge clk); valid_in=0;
            @(posedge done_tile); @(posedge clk);
            for(i=0;i<TILE_SIZE;i++) begin
                int r=rb*TILE_SIZE+i;
                hw_vec[r]=result_out_vec[i];
            end
        end
        mismatch_cnt_010=0;
        for(i=0;i<D_INNER;i++)
            if(hw_vec[i]!==ref_vec[i]) mismatch_cnt_010++;
        $display("[INFO] MODE010 mismatches=%0d", mismatch_cnt_010);

        // ========================================================
        // MODE011 : EWM-Outer (B_x = x_t ⊗ B_raw)
        // ========================================================
        $display("\n===============================================");
        $display(">>> MODE011 : EWM-Outer (B_x = x_t ⊗ B_raw)");
        $display("===============================================");
        mode=3'b011; accumulate_en=0;
        for(i=0;i<D_INNER;i++) x_vec[i]=($random%8)-4;
        for(j=0;j<D_STATE;j++) B_raw[j]=($random%8)-4;
        for(i=0;i<D_INNER;i++)
            for(j=0;j<D_STATE;j++)
                ref_outer[i][j]=x_vec[i]*B_raw[j];

        for(rb=0;rb<D_INNER/TILE_SIZE;rb++) begin
            for(i=0;i<TILE_SIZE;i++) begin
                int r=rb*TILE_SIZE+i;
                for(j=0;j<TILE_SIZE;j++) a_in[i][j]=x_vec[r];
            end
            for(j=0;j<TILE_SIZE;j++) b_vec[j]=B_raw[j];
            @(posedge clk); valid_in=1; @(posedge clk); valid_in=0;
            @(posedge done_tile); @(posedge clk);
            for(i=0;i<TILE_SIZE;i++) begin
                int r=rb*TILE_SIZE+i;
                for(j=0;j<TILE_SIZE;j++) hw_outer[r][j]=result_out_mat[i][j];
            end
        end
        mismatch_cnt_011=0;
        for(i=0;i<D_INNER;i++)
            for(j=0;j<D_STATE;j++)
                if(hw_outer[i][j]!==ref_outer[i][j]) mismatch_cnt_011++;
        $display("[INFO] MODE011 mismatches=%0d", mismatch_cnt_011);

        $display("\n==============================================================");
        $display("[SUMMARY] 000=%0d  001=%0d  010=%0d  011=%0d mismatches",
                 mismatch_cnt_000,mismatch_cnt_001,mismatch_cnt_010,mismatch_cnt_011);
        $display("==============================================================");
        $finish;
    end

    // monitor
    initial
        $monitor("T=%0t | mode=%b | valid_in=%b | valid_out=%b | done=%b",
                 $time,mode,valid_in,valid_out,done_tile);
endmodule
