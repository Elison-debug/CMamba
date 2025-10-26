//---------------------------------------------------------------
// Module: recfg_array (Hybrid Systolic/Spatial Architecture)
//
// Description:
//   - Implements a 16x16 reconfigurable array using the high-precision pe_unit.
//   - ARCHITECTURE: Hybrid
//     - SPATIAL: PEs compute in parallel using data from local registers.
//     - SYSTOLIC: PEs compute using data flowed from neighbors
//       and/or local registers (Weight-Stationary).
//   Mode Encoding (3-bit):
//   -----------------------------------------------------------
//   | mode | Operation Description          | Formula                           | Output Shape |
//   |------|--------------------------------|-----------------------------------|---------------|
//   | 000  | MAC (Matrix × Vector)          | RAW=Wx_pj⊙x_t; Δ_t=W_Δ⊙Δ_raw    | Vector (d_state×1) |
//   | 001  | EWM-Matrix (Matrix × Vector)   | ΔA = A ⊙ spΔ_t; ΔB_x=spΔ_t ⊙ B_x| Matrix (d_inner×d_state) |
//   | 010  | EWM-Vector (Vector × Vector)   | D_x = D ⊙ x_t                    | Vector (d_inner×1) |
//   | 011  | EWM-Outer (Outer Product)      | B_x = x_t ⊗ B_raw; C_h=ht⊗C_raw | Matrix (d_inner×d_state) |
//   | 100  | EWA-Vector (Vector + Vector)   | y = C_h + D_x; Δ_t_b=Δ_t+dt_bias  | Vector (d_inner×1) |
//   | 101  | EWA-Matrix (Matrix + Matrix)   | h_t = A_ht-1 + ΔB_x               | Matrix (d_inner×d_state) |
//   | 110  | EWM-Matrix2 (Matrix × Matrix)  | A_ht-1 = EXP_ΔA ⊙ h_t-1          | Matrix (d_inner×d_state) |
//   -----------------------------------------------------------
//   I/O (NEW):
//   - I/O is now STREAMING, not parallel.
//   - 'load_en' signal controls Load vs. Compute phases.
//
//   DATAFLOW MODES:
//   - load_en = 1 (Load Phase):
//     - 'stream_a_in' -> Loads 'reg_a' (horizontal flow)
//     - 'stream_b_in' -> Loads 'reg_b' (vertical flow)
//   - load_en = 0 (Compute Phase):
//     - 'valid_in' triggers computation based on 'mode'.
//     - Mode 000 (MAC): Systolic. 'reg_a' (Weights) * 'stream_b_in' (Vector)
//     - Mode 011 (Outer): Systolic. 'stream_a_in' * 'stream_b_in'
//     - All Others: Spatial. 'reg_a' (op) 'reg_b'
//
//   PRECISION:
//   - PE operates on 32-bit Q16.16 internally.
//   - Outputs 'result_vec_out'/'result_mat_out' are scaled
//     back to 16-bit Q8.8.
//---------------------------------------------------------------
module recfg_array_new #(
    parameter DATA_WIDTH = 16,
    parameter TILE_SIZE  = 16,
    parameter ACC_WIDTH  = 32, // 32-bit Q16.16
    parameter FRAC_BITS  = 8   // 8 fractional bits
)(
    input  logic                          clk,
    input  logic                          rst_n,
    
    // --- Control Signals ---
    input  logic                          valid_in,     // Start computation (when load_en=0)
    input  logic                          load_en,      // 1=Load registers, 0=Compute
    input  logic [2:0]                    mode,         // 000-110: operation mode

    // --- Streaming I/O (Replaces parallel I/O) ---
    // Data stream for 'reg_a' (from left) or 'a_flow' (Outer prod)
    input  logic signed [DATA_WIDTH-1:0]  stream_a_in [TILE_SIZE-1:0],
    // Data stream for 'reg_b' (from top) or 'b_flow' (MAC/Outer)
    input  logic signed [DATA_WIDTH-1:0]  stream_b_in [TILE_SIZE-1:0],
    
    // Vector input for MAC accumulation (from left)
    input  logic                          accumulate_en, 
    input  logic signed [DATA_WIDTH-1:0]  acc_in_vec [TILE_SIZE-1:0], // Q8.8
    
    // --- Outputs ---
    // Streaming vector output (from right)
    output logic signed [DATA_WIDTH-1:0]  result_vec_out [TILE_SIZE-1:0],
    // Streaming matrix output (from bottom-right PEs)
    output logic signed [DATA_WIDTH-1:0]  result_mat_out [TILE_SIZE-1:0],
    
    output logic                          valid_out,
    output logic                          out_shape_flag
    // 'done_tile' is removed as latency is now mode-dependent
);

    // ============================================================
    // Internal Signals
    // ============================================================
    genvar i, j;

    // --- PE Grid Signals ---
    // Local Registers (inferred as LUT-RAM)
    logic signed [DATA_WIDTH-1:0] reg_a [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] reg_b [TILE_SIZE-1:0][TILE_SIZE-1:0];

    // PE Input Muxes
    logic signed [DATA_WIDTH-1:0] pe_a_in_sel [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] pe_b_in_sel [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [ACC_WIDTH-1:0]  pe_acc_in_sel [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic [1:0]                   pe_op_mode [TILE_SIZE-1:0][TILE_SIZE-1:0]; // 00, 01, 10

    // PE Output Wires
    logic signed [ACC_WIDTH-1:0]  pe_result_out [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic                         pe_valid_out [TILE_SIZE-1:0][TILE_SIZE-1:0];

    // Systolic Dataflow Registers
    logic signed [DATA_WIDTH-1:0] a_flow [TILE_SIZE-1:0][TILE_SIZE:0];
    logic signed [DATA_WIDTH-1:0] b_flow [TILE_SIZE:0][TILE_SIZE-1:0];
    logic signed [ACC_WIDTH-1:0]  acc_flow [TILE_SIZE-1:0][TILE_SIZE:0];

    // High-precision output registers (stores Q16.16)
    logic signed [ACC_WIDTH-1:0]  result_reg_hi [TILE_SIZE-1:0][TILE_SIZE-1:0];
    
    // Control/Status
    logic [$clog2(TILE_SIZE*2)-1:0] latency_cnt;
    logic                           compute_active;

    // ============================================================
    // PE Array Generation
    // ============================================================
    generate
    for (i = 0; i < TILE_SIZE; i++) begin : ROW
        for (j = 0; j < TILE_SIZE; j++) begin : COL

            // --- 1. Local Storage (LUT-RAM) ---
            // These registers hold the stationary data for Spatial mode
            // or Weight-Stationary Systolic mode.
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    reg_a[i][j] <= '0;
                    reg_b[i][j] <= '0;
                end else if (load_en) begin
                    // Load 'reg_a' from the left
                    reg_a[i][j] <= a_flow[i][j]; 
                    // Load 'reg_b' from the top
                    reg_b[i][j] <= b_flow[i][j];
                end
            end

            // --- 2. Systolic Dataflow Logic ---
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    a_flow[i][j+1]   <= '0;
                    b_flow[i+1][j]   <= '0;
                    acc_flow[i][j+1] <= '0;
                end else if (!load_en && compute_active) begin
                    // Data flows only during compute phase
                    case (mode)
                        // MAC: Accumulator flows left-to-right
                        3'b000: acc_flow[i][j+1] <= pe_result_out[i][j];
                        
                        // Outer Product: 'a' flows left-to-right
                        3'b011: a_flow[i][j+1] <= a_flow[i][j];
                        
                        default: begin
                           acc_flow[i][j+1] <= '0;
                           a_flow[i][j+1]   <= '0;
                        end
                    endcase
                    
                    // 'b' flow is common for MAC and Outer Product
                    if (mode == 3'b000 || mode == 3'b011) begin
                         b_flow[i+1][j] <= b_flow[i][j];
                    end else begin
                         b_flow[i+1][j] <= '0;
                    end
                end
            end

            // --- 3. Hybrid Input Muxes ---
            // This is the CORE of the hybrid architecture.
            // It selects the correct input for the PE based on the global mode.
            always_comb begin
                // Default: Spatial Mode
                pe_a_in_sel[i][j]   = reg_a[i][j];
                pe_b_in_sel[i][j]   = reg_b[i][j];
                pe_acc_in_sel[i][j] = '0;
                pe_op_mode[i][j]    = 2'b00; // Default to MAC

                case (mode)
                    // 000: MAC (Systolic)
                    // A = Weight (from reg), B = Vector (from top flow)
                    3'b000: begin
                        pe_a_in_sel[i][j]   = reg_a[i][j];
                        pe_b_in_sel[i][j]   = b_flow[i][j];
                        pe_acc_in_sel[i][j] = acc_flow[i][j];
                        pe_op_mode[i][j]    = 2'b00; // MAC
                    end
                    
                    // 001: EWM-Matrix (Spatial + Row Broadcast)
                    // A = Matrix (reg_a), B = Vector (broadcast from reg_b[i][0])
                    3'b001: begin
                        pe_a_in_sel[i][j]   = reg_a[i][j];
                        pe_b_in_sel[i][j]   = reg_b[i][0]; // Row Broadcast
                        pe_op_mode[i][j]    = 2'b01; // EWM
                    end

                    // 010: EWM-Vector (Spatial - Col 0 only)
                    // A = Vector (reg_a), B = Vector (reg_b)
                    3'b010: begin
                        pe_a_in_sel[i][j]   = reg_a[i][j];
                        pe_b_in_sel[i][j]   = reg_b[i][j];
                        pe_op_mode[i][j]    = 2'b01; // EWM
                    end

                    // 011: Outer Product (Systolic)
                    // A = Vector (from left flow), B = Vector (from top flow)
                    3'b011: begin
                        pe_a_in_sel[i][j]   = a_flow[i][j];
                        pe_b_in_sel[i][j]   = b_flow[i][j];
                        pe_op_mode[i][j]    = 2'b01; // EWM
                    end
                    
                    // 100: EWA-Vector (Spatial - Col 0 only)
                    // A = Vector (reg_a), B = Vector (reg_b)
                    3'b100: begin
                        pe_a_in_sel[i][j]   = reg_a[i][j];
                        pe_b_in_sel[i][j]   = reg_b[i][j];
                        pe_op_mode[i][j]    = 2'b10; // EWA
                    end

                    // 101: EWA-Matrix (Spatial)
                    // A = Matrix (reg_a), B = Matrix (reg_b)
                    3'b101: begin
                        pe_a_in_sel[i][j]   = reg_a[i][j];
                        pe_b_in_sel[i][j]   = reg_b[i][j];
                        pe_op_mode[i][j]    = 2'b10; // EWA
                    end

                    // 110: EWM-Matrix2 (Spatial)
                    // A = Matrix (reg_a), B = Matrix (reg_b)
                    3'b110: begin
                        pe_a_in_sel[i][j]   = reg_a[i][j];
                        pe_b_in_sel[i][j]   = reg_b[i][j];
                        pe_op_mode[i][j]    = 2'b01; // EWM
                    end
                endcase
            end

            // --- 4. PE Instantiation ---
            pe_unit_new #(
                .DATA_WIDTH(DATA_WIDTH),
                .ACC_WIDTH (ACC_WIDTH),
                .FRAC_BITS (FRAC_BITS)
            ) U_PE (
                .clk       (clk),
                .rst_n     (rst_n),
                .valid_in  (!load_en && compute_active), // Compute only when enabled
                .mode      (pe_op_mode[i][j]),
                .a_in      (pe_a_in_sel[i][j]),
                .b_in      (pe_b_in_sel[i][j]),
                .acc_in    (pe_acc_in_sel[i][j]),
                .result_out(pe_result_out[i][j]), // Q16.16
                .valid_out (pe_valid_out[i][j])
            );
            
            // --- 5. Spatial Result Latch ---
            // For Spatial modes, we latch the PE's result locally.
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    result_reg_hi[i][j] <= '0;
                else if (pe_valid_out[i][j])
                    result_reg_hi[i][j] <= pe_result_out[i][j];
            end

        end
    end
    endgenerate

    // ============================================================
    // Array Boundary Connections
    // ============================================================
    generate
    for (i = 0; i < TILE_SIZE; i++) begin
        // Connect stream inputs to the array edges
        assign a_flow[i][0] = stream_a_in[i];
        assign b_flow[0][i] = stream_b_in[i];
        
        // Connect high-precision accumulator input (sign-extended Q8.8 -> Q16.16)
        assign acc_flow[i][0] = (accumulate_en) ? 
            {{(ACC_WIDTH-DATA_WIDTH){acc_in_vec[i][DATA_WIDTH-1]}}, acc_in_vec[i]} : 
            '0;

        // --- Output Scaling (Q16.16 -> Q8.8) ---
        // Vector output taps the *end* of the accumulator chain (MAC)
        // or the *first column* of registers (Vector Spatial)
        always_comb begin
            logic signed [ACC_WIDTH-1:0] vec_out_hi;
            if (mode == 3'b000)
                vec_out_hi = acc_flow[i][TILE_SIZE]; // MAC result
            else
                vec_out_hi = result_reg_hi[i][0];   // Vector EWM/EWA result
            
            // Scale Q16.16 (16 frac bits) down to Q8.8 (8 frac bits)
            // This requires a right shift by (16 - 8) = 8 bits.
            // (Also handles saturation/truncation)
            result_vec_out[i] <= $signed(vec_out_hi >>> FRAC_BITS);
        end

        // Matrix output taps the *last row* of registers
        // (This is an arbitrary choice, could be any edge)
        always_comb begin
             logic signed [ACC_WIDTH-1:0] mat_out_hi;
             mat_out_hi = result_reg_hi[TILE_SIZE-1][i]; // Last row
             result_mat_out[i] <= $signed(mat_out_hi >>> FRAC_BITS);
        end
    end
    endgenerate
    
    // ============================================================
    // Control Logic
    // ============================================================
    
    // Compute pipeline is active one cycle *after* valid_in
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            compute_active <= 1'b0;
        else
            compute_active <= valid_in && !load_en;
    end
    
    // Latency counter for valid_out signal
    // This is a simplified counter. A robust design would have
    // different counts per mode (e.g., 1 for Spatial, 31 for Systolic).
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            latency_cnt <= '0;
            valid_out   <= 1'b0;
        end else if (valid_in && !load_en) begin
            latency_cnt <= (TILE_SIZE * 2) - 2; // Latency to fill (e.g., 31)
            valid_out   <= 1'b0;
        end else if (latency_cnt != 0) begin
            latency_cnt <= latency_cnt - 1;
            valid_out   <= (latency_cnt == 1);
        end else begin
            valid_out <= 1'b0;
        end
    end

    assign out_shape_flag = (mode inside {3'b001,3'b011,3'b101,3'b110});

endmodule