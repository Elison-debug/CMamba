Controller FSM for SSM Accelerator (x_t preloaded in FBUF)
----------------------------------------------------------

States:
IDLE → X_PROJ → DT_PROJ → SOFTPLUS → ΔA_CALC → EXP → A_B_CALC → EWA1 → C_D_CALC → EWA2 → DONE

----------------------------------------------------------
1. IDLE
----------------------------------------------------------
- Function: Wait for external start signal, reset address counters.
- Control Signals:
  start_array=0; reset_addr_fbuf=1; reset_addr_wbuf=1; reset_addr_hbuf=1; mode_array=IDLE; valid_in=0
- Handshake: Wait for start=1
- Wait: if (start==1)
- Next: X_PROJ

----------------------------------------------------------
2. X_PROJ (MAC)
----------------------------------------------------------
- Operation: RAW = W_xpj × x_t
- Control Signals:
  start_array=1; mode_array=MAC_MODE; mode_mac=XPROJ;
  rd_en_wbuf=1 (bank=Wx_pj); rd_en_fbuf=1 (zone=X_ZONE);
  wr_en_fbuf=1 (zone=RAW_ZONE);
  addr_wbuf+=16; addr_fbuf+=16
- Data Input: 16 data per cycle to array (x_t from FBUF, W_xpj from WBUF)
- Handshake: array_valid_in=1; wait done_mac=1
- Wait: if (done_mac==1)
- Next: DT_PROJ

----------------------------------------------------------
3. DT_PROJ (MAC + bias)
----------------------------------------------------------
- Operation: Δ_t = W_Δ × Δ_raw + bias
- Control Signals:
  start_array=1; mode_array=MAC_MODE; mode_mac=DTPROJ;
  rd_en_wbuf=1 (bank=W_Δ); rd_en_bbuf=1 (bank=bias);
  rd_en_fbuf=1 (zone=Δ_raw);
  wr_en_fbuf=1 (zone=Δ_t);
  addr_wbuf+=16; addr_fbuf+=16
- Handshake: array_valid_in=1; wait done_mac=1
- Wait: if (done_mac==1)
- Next: SOFTPLUS

----------------------------------------------------------
4. SOFTPLUS (Nonlinear)
----------------------------------------------------------
- Operation: spΔ_t = ln(1 + e^{Δ_t})
- Control Signals:
  start_sp=1; fbuf_sel_in=Δ_t; fbuf_sel_out=spΔ_t;
  rd_en_fbuf=1; wr_en_fbuf=1
- Handshake: sp_valid_in=1 → sp_ready_out; wait done_sp=1
- Wait: if (done_sp==1)
- Next: ΔA_CALC

----------------------------------------------------------
5. ΔA_CALC (EWM mode1)
----------------------------------------------------------
- Operation: ΔA = spΔ_t ⊙ A
- Control Signals:
  start_array=1; mode_array=EWM_MODE; mode_ewm=1;
  rd_en_fbuf=1 (spΔ_t); rd_en_wbuf=1 (A);
  wr_en_fbuf=1 (ΔA); addr_wbuf+=16; addr_fbuf+=16
- Handshake: array_valid_in=1; wait done_ewm=1
- Wait: if (done_ewm==1)
- Next: EXP

----------------------------------------------------------
6. EXP (Nonlinear)
----------------------------------------------------------
- Operation: EXP_ΔA = e^{ΔA}
- Control Signals:
  start_exp=1; rd_en_fbuf=1 (ΔA); wr_en_fbuf=1 (expΔA)
- Handshake: exp_valid_in=1; wait done_exp=1
- Wait: if (done_exp==1)
- Next: A_B_CALC

----------------------------------------------------------
7. A_B_CALC (EWM Parallel)
----------------------------------------------------------
- Operation:
  A_ht-1 = EXP(ΔA) ⊙ h_{t-1}
  ΔB_x = spΔ_t ⊙ (B_raw ⊗ x_t)
- Control Signals:
  start_array=1; mode_array=EWM_MODE;
  mode_ewm=4 (A-path) + mode_ewm=2 (B-path);
  rd_en_fbuf=1 (expΔA, spΔ_t, x_t, B_raw);
  rd_en_hbuf=1 (h_{t-1}); wr_en_fbuf=1 (A_ht-1, ΔB_x)
- Data Input: 16 data per row/column per cycle
- Handshake: array_valid_in=1; wait done_ewm_A & done_ewm_B
- Wait: if (done_ewm_A & done_ewm_B)
- Next: EWA1

----------------------------------------------------------
8. EWA1 (Add)
----------------------------------------------------------
- Operation: h_t = A_ht-1 + ΔB_x
- Control Signals:
  start_ewa=1; mode_array=EWA_MODE; mode_ewa=0;
  rd_en_fbuf=1 (A_ht-1, ΔB_x); wr_en_hbuf=1 (h_t)
- Handshake: ewa_valid_in=1; wait done_ewa=1
- Wait: if (done_ewa==1)
- Next: C_D_CALC

----------------------------------------------------------
9. C_D_CALC (EWM + Reduction Tree)
----------------------------------------------------------
- Operation:
  C_h = C_raw ⊗ h_t → Reduction
  D_x = D ⊙ x_t
- Control Signals:
  start_array=1; mode_array=EWM_MODE;
  mode_ewm=6 (C-path); reduce_en=1;
  rd_en_fbuf=1 (C_raw, x_t); rd_en_hbuf=1 (h_t);
  rd_en_wbuf=1 (D); wr_en_fbuf=1 (C_h, D_x)
- Data Input: 16×16 tile per iteration
- Handshake: array_valid_in=1; wait done_reduce & done_ewm
- Wait: if (done_reduce & done_ewm)
- Next: EWA2

----------------------------------------------------------
10. EWA2 (Add)
----------------------------------------------------------
- Operation: y_t = C_h + D_x
- Control Signals:
  start_ewa=1; mode_array=EWA_MODE; mode_ewa=1;
  rd_en_fbuf=1 (C_h, D_x); wr_en_obuf=1 (y_t)
- Handshake: ewa_valid_in=1; wait done_ewa=1
- Wait: if (done_ewa==1)
- Next: DONE

----------------------------------------------------------
11. DONE
----------------------------------------------------------
- Function: End of computation, signal completion.
- Control Signals:
  start_array=0; wr_en_fbuf=0; rd_en_fbuf=0; done_flag=1
- Next: IDLE

----------------------------------------------------------
Data Transfer Granularity:
- Each cycle, the array receives 16 data elements per input channel (one tile row/column per cycle).
- Each MAC/EWM/EWA operation processes one 16×16 tile block per iteration.
