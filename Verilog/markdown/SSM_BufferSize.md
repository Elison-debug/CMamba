WBUF : 权重常量区(ROM, 32 KB)
INBUF :输入和Δ_raw缓冲区(单端口RAM, 0.6 KB)
FBUF : 动态中间结果(43 Bank SRAM, 48.6 KB)
HBUF : 状态区(2 Bank SRAM, 16 KB)
OBUF : 输出FIFO(<1 KB)
───────────────────────────────────────────────────────────────
[WBUF] — Weight Buffer (Read-Only ROM)
───────────────────────────────────────────────────────────────
Name        | Dimension                   | #Elements | Size(Bytes) | Description
-------------|-----------------------------|------------|-------------|----------------------------
Wx_pj        | (dt_rank + 2*d_state, d_inner) = (40, 256) | 10,240  | 20,480  | Projection weights for X_PROJ
W_Δ          | (d_inner, dt_rank) = (256, 8)              | 2,048   | 4,096   | Temporal projection weights
A            | (d_inner, d_state) = (256, 16)             | 4,096   | 8,192   | Elementwise multiply weights for ΔA
B_raw        | (d_state) = (16)                           | 16      | 32      | Outer-product base vector for B_x
C_raw        | (d_state) = (16)                           | 16      | 32      | Outer-product base vector for C_h
D            | (d_inner) = (256)                          | 256     | 512     | Elementwise multiply weights for D_x
dt_bias      | (d_inner) = (256)                          | 256     | 512     | Bias term added to Δ_t
---------------------------------------------------------------
Total WBUF Memory = 33,392 Bytes ≈ 32.6 KB
Recommended : 4 ROM Banks (Wx/WΔ | A | B_raw/C_raw/D | dt_bias)
───────────────────────────────────────────────────────────────


───────────────────────────────────────────────────────────────
[INBUF] — Input Buffer (Single Dual-Port RAM)
───────────────────────────────────────────────────────────────
Name        | Dimension                    | #Elements | Size(Bytes) | Description
-------------|------------------------------|------------|-------------|------------------------------
x_t          | (d_inner, 1) = (256,1)       | 256      | 512      | Input feature vector for current time-step
Δ_raw        | (dt_rank + 2*d_state, 1) = (40,1) | 40 | 80 | Output of X_PROJ used by DT_PROJ
---------------------------------------------------------------
Total INBUF Memory = 592 Bytes ≈ 0.6 KB
Configuration : Dual-Port RAM (Width=16bit, Depth=512)
Notes :
 - Serves low-bandwidth input stage (X_PROJ, DT_PROJ, SP_B_CALC, EXP_D_CALC)
 - Access pattern : Sequential read/write, no bank interleaving
 - Can be clock-gated when idle to save power
───────────────────────────────────────────────────────────────


───────────────────────────────────────────────────────────────
[FBUF] — Feature / Intermediate Buffer (3 Banks, Dual-Port)
───────────────────────────────────────────────────────────────
Name        | Dimension                    | #Elements | Size(Bytes) | Assigned Bank | Description
-------------|------------------------------|------------|-------------|----------------|------------------------------
Δ_t          | (d_inner,1) = (256,1)        | 256      | 512      | Bank1 | Intermediate temporal projection
Δ_t_b        | (d_inner,1) = (256,1)        | 256      | 512      | Bank1 | Bias-added vector
spΔ_t        | (d_inner,1) = (256,1)        | 256      | 512      | Bank1 | Softplus result
ΔA           | (d_inner, d_state) = (256,16)| 4,096    | 8,192    | Bank1 | ΔA matrix for EXP
B_x          | (d_inner, d_state) = (256,16)| 4,096    | 8,192    | Bank2 | Outer-product result
ΔB_x         | (d_inner, d_state) = (256,16)| 4,096    | 8,192    | Bank2 | Elementwise ΔB_x
EXP_ΔA       | (d_inner, d_state) = (256,16)| 4,096    | 8,192    | Bank3 | Exponentiated ΔA
A_ht-1       | (d_inner, d_state) = (256,16)| 4,096    | 8,192    | Bank3 | Weighted previous hidden state
C_h          | (d_inner, d_state) = (256,16)| 4,096    | 8,192    | Bank3 | Output outer-product term
D_x          | (d_inner,1) = (256,1)        | 256      | 512      | Bank3 | Elementwise D_x
---------------------------------------------------------------
Total FBUF Memory = 49,792 Bytes ≈ 48.6 KB
Configuration : 3 Banks × Dual-Port (Width=16bit, Depth≈4K)
Notes :
 - Bank1/2/3 handle Δ-, B-, and EXP-path intermediate data
 - Port A → Systolic Array (read/write), Port B → Nonlinear Unit or Controller Prefetch
 - Unified 12-bit addressing 
 - Each Bank internally organized as 16 subbanks (total 256-bit wide, 16-deep)
   → Each subbank outputs 16×16-bit per cycle, fully matching the 16×16 PE array bandwidth
 - Read/Write decoupled via true dual-port structure (PortA for Array, PortB for Softplus/EXP)
 - Nonlinear units access via PortB (Softplus→spΔ_t, EXP→EXP_ΔA)
 - Array-Interface Selector:
      · For matrix modes: 256-bit direct feed
      · For vector modes: 16→1 multiplexer per column extracts required 16-bit lane
        and optionally broadcasts it to all columns (no new RAM, only mux logic)
 - Write-Packer Controller:
      · Detects out_shape_flag
      · If vector output (16×1): pack 16×16-bit into one 256-bit word → single write
      · If matrix output (16×16): write 16 times (each row = 256-bit) to consecutive addresses
 - Ensures identical 256-bit interface for all modes, minimizing control divergence
 - Bandwidth summary:
      · Each Bank: 256-bit/port × 2 ports
      · Total array-side read bandwidth = 3 × 256bit = 768bit/cycle
      · Satisfies 16×16 PE input rate (512bit) with margin for nonlinear overlap
───────────────────────────────────────────────────────────────



───────────────────────────────────────────────────────────────
[HBUF] — Hidden State Buffer (2 Banks, Dual-Port)
───────────────────────────────────────────────────────────────
Name        | Dimension                 | #Elements | Size(Bytes) | Assigned Bank | Description
-------------|---------------------------|------------|-------------|----------------|------------------------------
h_{t-1}      | (d_inner, d_state)=(256,16) | 4,096 | 8,192 | Bank0 | Previous hidden state
h_t          | (d_inner, d_state)=(256,16) | 4,096 | 8,192 | Bank1 | Updated hidden state
---------------------------------------------------------------
Total HBUF Memory = 16,384 Bytes ≈ 16.0 KB
Configuration : 2 Banks (Ping-Pong scheme)
Ping-Pong Operation :
 - Each time-step (t) alternates read/write roles between two banks.
   • At step t:
       - Read  : Bank0 (h_{t-1}) → used by array in mode4 (EXP ⊙ h_{t-1})
       - Write : Bank1 (h_t) ← array output from EWA1 (A_ht-1 + ΔB_x)
   • At step t+1:
       - Read  : Bank1 (now h_t) → used by next iteration
       - Write : Bank0 (overwritten with new h_{t+1})
 - A single control bit pingpong_flag toggles every time-step:
       pingpong_flag = 0 → Read Bank0, Write Bank1
       pingpong_flag = 1 → Read Bank1, Write Bank0
 - This ensures continuous pipeline flow:
       While one bank feeds data to the array, the other bank receives results,
       avoiding read/write conflict and eliminating stall cycles.

Bank Architecture :
 - Each Bank stores a full hidden-state matrix (256×16 = 4,096 elements).
 - Implemented as 16 subbanks × (16-bit × 256-depth):
       · Each subbank outputs one 16-bit element per row.
       · 16 subbanks operate in parallel → 256-bit total per clock.
 - Depth = 256 words (covering d_inner dimension)
 - Width = 256 bits (16×16-bit) → matches one tile-row input/output of the 16×16 array.
 - Dual-Port configuration:
       · Port A → connected to array (read/write)
       · Port B → reserved for controller prefetch or off-chip checkpointing (optional)

Control and Access Summary :
 - Read mode4:  array reads h_{t-1} (BankX→PortA)
 - Write EWA1:  array writes h_t    (BankY←PortA)
 - On DONE state of each time-step, controller toggles pingpong_flag and swaps roles.
 - No address conflict; both banks accessible every cycle.
───────────────────────────────────────────────────────────────


───────────────────────────────────────────────────────────────
[OBUF] — Output Buffer
───────────────────────────────────────────────────────────────
Name        | Dimension            | #Elements | Size(Bytes) | Description
-------------|----------------------|------------|-------------|----------------------------
y_t          | (d_inner, 1) = (256,1) | 256 | 512 | Final output vector (EWA2 result)
---------------------------------------------------------------
Total OBUF Memory = 512 Bytes ≈ 0.5 KB
───────────────────────────────────────────────────────────────


───────────────────────────────────────────────────────────────
[SUMMARY]
───────────────────────────────────────────────────────────────
Buffer | Size(Bytes) | Size(KB) | Notes
--------|-------------|-----------|---------------------------
WBUF    | 33,392      | 32.6 KB   | ROM, read-only weights
FBUF    | 51,092      | 49.9 KB   | 4 Banks, dual-port SRAM
HBUF    | 16,384      | 16.0 KB   | 2 Banks, ping-pong SRAM
OBUF    | 512         | 0.5 KB    | Output FIFO / temporary
---------------------------------------------------------------
Total On-Chip Buffer Memory = 101,380 Bytes ≈ 99.0 KB
───────────────────────────────────────────────────────────────
