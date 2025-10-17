//---------------------------------------------------------------
// Testbench: tb_recfg_array
// Purpose  : Verify recfg_array (modes 100–110, ASCII-safe)
// Author   : Shengjie Chen
//---------------------------------------------------------------
`timescale 1ns/1ps
module tb_recfg_array_EWA;

    localparam DATA_WIDTH = 16;
    localparam TILE_SIZE  = 16;
    localparam D_INNER    = 256;
    localparam DT_RANK    = 8;
    localparam D_STATE    = 16;
    localparam OUT_SIZE   = DT_RANK + 2*D_STATE;
    localparam N_ROWBLK   = (OUT_SIZE + TILE_SIZE - 1) / TILE_SIZE;
    localparam N_COLBLK   = D_INNER / TILE_SIZE;

    reg  clk, rst_n, valid_in, accumulate_en;
    reg  [2:0] mode;

    reg  signed [DATA_WIDTH-1:0] acc_in_vec [0:TILE_SIZE-1];
    reg  signed [DATA_WIDTH-1:0] a_in  [0:TILE_SIZE-1][0:TILE_SIZE-1];
    reg  signed [DATA_WIDTH-1:0] b_vec [0:TILE_SIZE-1];
    reg  signed [DATA_WIDTH-1:0] b_mat [0:TILE_SIZE-1][0:TILE_SIZE-1];
    wire signed [DATA_WIDTH-1:0] result_out_vec [0:TILE_SIZE-1];
    wire signed [DATA_WIDTH-1:0] result_out_mat [0:TILE_SIZE-1][0:TILE_SIZE-1];
    wire valid_out, done_tile, out_shape_flag;

    // Clock & reset
    initial begin clk=0; forever #5 clk=~clk; end
    initial begin rst_n=0; valid_in=0; accumulate_en=0; mode=3'b000; #50 rst_n=1; end

    recfg_array #(.DATA_WIDTH(DATA_WIDTH), .TILE_SIZE(TILE_SIZE)) dut (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in), .mode(mode),
        .a_in(a_in), .b_vec(b_vec), .b_mat(b_mat),
        .accumulate_en(accumulate_en), .acc_in_vec(acc_in_vec),
        .result_out_vec(result_out_vec), .result_out_mat(result_out_mat),
        .valid_out(valid_out), .done_tile(done_tile), .out_shape_flag(out_shape_flag)
    );

    integer i,j,rb;
    // -------- data storage --------
    reg signed [DATA_WIDTH-1:0] C_vec [0:D_INNER-1];
    reg signed [DATA_WIDTH-1:0] D_x   [0:D_INNER-1];
    reg signed [31:0] ref_ewa_vec [0:D_INNER-1];
    reg signed [DATA_WIDTH-1:0] hw_ewa_vec [0:D_INNER-1];

    reg signed [DATA_WIDTH-1:0] A_prev [0:D_INNER-1][0:D_STATE-1];
    reg signed [DATA_WIDTH-1:0] DeltaB [0:D_INNER-1][0:D_STATE-1];
    reg signed [31:0] ref_ewa_mat [0:D_INNER-1][0:D_STATE-1];
    reg signed [DATA_WIDTH-1:0] hw_ewa_mat [0:D_INNER-1][0:D_STATE-1];

    // -------- for MODE110 --------
    reg signed [DATA_WIDTH-1:0] EXP_DA [0:D_INNER-1][0:D_STATE-1];
    reg signed [DATA_WIDTH-1:0] H_prev [0:D_INNER-1][0:D_STATE-1];
    reg signed [31:0] ref_mat_110 [0:D_INNER-1][0:D_STATE-1];
    reg signed [DATA_WIDTH-1:0] hw_mat_110 [0:D_INNER-1][0:D_STATE-1];

    integer mismatch100, mismatch101, mismatch110;

    // ============================================================
    // Stimulus
    // ============================================================
    initial begin
        @(posedge rst_n);
        $display("===============================================");
        $display("MODE100–MODE110 verification start");
        $display("===============================================");

        //---------------------------
        // MODE100 : EWA-Vector (y = C_h + D_x)
        //---------------------------
        $display("\n[MODE100] EWA-Vector: y = C_h + D_x");
        mode = 3'b100; accumulate_en = 0;

        for (i=0;i<D_INNER;i++) begin
            C_vec[i] = ($random%8)-4;
            D_x[i]   = ($random%8)-4;
            ref_ewa_vec[i] = C_vec[i] + D_x[i];
        end

        for (rb=0; rb<D_INNER/TILE_SIZE; rb++) begin
            for (i=0;i<TILE_SIZE;i++) begin
                int r = rb*TILE_SIZE+i;
                for (j=0;j<TILE_SIZE;j++)
                    a_in[i][j] = (j==0) ? C_vec[r] : 0;
                b_vec[i] = D_x[r];
            end
            @(posedge clk); valid_in=1; @(posedge clk); valid_in=0;
            @(posedge done_tile); @(posedge clk);
            for (i=0;i<TILE_SIZE;i++) begin
                int r = rb*TILE_SIZE+i;
                hw_ewa_vec[r] = result_out_vec[i];
            end
        end

        mismatch100 = 0;
        for (i=0;i<D_INNER;i++)
            if (hw_ewa_vec[i] !== ref_ewa_vec[i])
                mismatch100++;

        $display("\nMODE100 Result Comparison (first 32 entries):");
        for (i=0;i<32;i++)
            $display("y_hw[%0d]=%0d  ref=%0d  %s",
                     i, hw_ewa_vec[i], ref_ewa_vec[i],
                     (hw_ewa_vec[i]===ref_ewa_vec[i])?"[PASS]":"[FAIL]");
        $display("[MODE100 mismatches=%0d]\n", mismatch100);

        //---------------------------
        // MODE101 : EWA-Matrix (h_t = A_ht-1 + ΔB_x)
        //---------------------------
        $display("\n[MODE101] EWA-Matrix: h_t = A_ht-1 + ΔB_x");
        mode = 3'b101; accumulate_en = 0;

        for (i=0;i<D_INNER;i++)
            for (j=0;j<D_STATE;j++) begin
                A_prev[i][j] = ($random%8)-4;
                DeltaB[i][j] = ($random%8)-4;
                ref_ewa_mat[i][j] = A_prev[i][j] + DeltaB[i][j];
            end

        for (rb=0; rb<D_INNER/TILE_SIZE; rb++) begin
            for (i=0;i<TILE_SIZE;i++) begin
                int r = rb*TILE_SIZE+i;
                for (j=0;j<TILE_SIZE;j++) begin
                    a_in[i][j] = A_prev[r][j];
                    b_mat[i][j] = DeltaB[r][j];
                end
            end
            @(posedge clk); valid_in=1; @(posedge clk); valid_in=0;
            @(posedge done_tile); @(posedge clk);
            for (i=0;i<TILE_SIZE;i++) begin
                int r = rb*TILE_SIZE+i;
                for (j=0;j<TILE_SIZE;j++)
                    hw_ewa_mat[r][j] = result_out_mat[i][j];
            end
        end

        mismatch101 = 0;
        for (i=0;i<D_INNER;i++)
            for (j=0;j<D_STATE;j++)
                if (hw_ewa_mat[i][j] !== ref_ewa_mat[i][j])
                    mismatch101++;

        $display("\nMODE101 Result Comparison (first 4x4 block):");
        for (i=0;i<4;i++) begin
            for (j=0;j<4;j++)
                $write("%6d ", hw_ewa_mat[i][j]);
            $display("");
        end
        $display("[MODE101 mismatches=%0d]\n", mismatch101);

        //---------------------------
        // MODE110 : EWM-Matrix2 (A_ht-1 = EXP_ΔA ⊙ h_t-1)
        //---------------------------
        $display("\n[MODE110] EWM-Matrix2: A_ht-1 = EXP_DA * H_prev");
        mode = 3'b110; accumulate_en = 0;

        for (i=0;i<D_INNER;i++)
            for (j=0;j<D_STATE;j++) begin
                EXP_DA[i][j] = ($random%8)-4;
                H_prev[i][j] = ($random%8)-4;
                ref_mat_110[i][j] = EXP_DA[i][j] * H_prev[i][j];
            end

        for (rb=0; rb<D_INNER/TILE_SIZE; rb++) begin
            for (i=0;i<TILE_SIZE;i++) begin
                int r = rb*TILE_SIZE+i;
                for (j=0;j<TILE_SIZE;j++) begin
                    a_in[i][j] = EXP_DA[r][j];
                    b_mat[i][j] = H_prev[r][j];
                end
            end
            @(posedge clk); valid_in=1; @(posedge clk); valid_in=0;
            @(posedge done_tile); @(posedge clk);
            for (i=0;i<TILE_SIZE;i++) begin
                int r = rb*TILE_SIZE+i;
                for (j=0;j<TILE_SIZE;j++)
                    hw_mat_110[r][j] = result_out_mat[i][j];
            end
        end

        mismatch110 = 0;
        for (i=0;i<D_INNER;i++)
            for (j=0;j<D_STATE;j++)
                if (hw_mat_110[i][j] !== ref_mat_110[i][j])
                    mismatch110++;

        $display("\nMODE110 Result Comparison (first 4x4 block):");
        for (i=0;i<4;i++) begin
            for (j=0;j<4;j++)
                $write("%6d ", hw_mat_110[i][j]);
            $display("");
        end
        $display("[MODE110 mismatches=%0d]\n", mismatch110);

        //---------------------------
        // Summary
        //---------------------------
        $display("==============================================================");
        $display("[SUMMARY] MODE100=%0d  MODE101=%0d  MODE110=%0d mismatches",
                 mismatch100, mismatch101, mismatch110);
        $display("==============================================================");
        $finish;
    end

    // Monitor
    initial
        $monitor("T=%0t | mode=%b | valid_in=%b | valid_out=%b | done=%b",
                 $time, mode, valid_in, valid_out, done_tile);

endmodule
