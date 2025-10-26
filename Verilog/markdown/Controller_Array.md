───────────────────────────────────────────────────────────────
[0] IDLE
───────────────────────────────────────────────────────────────
• Function : Wait for start signal, clear internal flags/counters
• Mode     : — 
• Inputs   : — 
• Outputs  : — 
• Parallel : —
• Purpose  : Controller initialization and reset

───────────────────────────────────────────────────────────────
[1] X_PROJ
───────────────────────────────────────────────────────────────
• Mode     : 000 (MAC)
• Compute  : Δ_raw = Wx_pj ⊙ x_t
• Inputs   : WBUF(Wx_pj), FBUF(x_t)
• Outputs  : FBUF(Δ_raw)
• Parallel : —
• Purpose  : Project input feature x_t into Δ_raw domain

───────────────────────────────────────────────────────────────
[2] DT_PROJ
───────────────────────────────────────────────────────────────
• Mode     : 000 (MAC)
• Compute  : Δ_t = W_Δ ⊙ Δ_raw
• Inputs   : WBUF(W_Δ), FBUF(Δ_raw)
• Outputs  : FBUF(Δ_t)
• Parallel : —
• Purpose  : Compute intermediate Δ_t for temporal update

───────────────────────────────────────────────────────────────
[3] DT_PROJ_B
───────────────────────────────────────────────────────────────
• Mode     : 100 (EWA-Vector)
• Compute  : Δ_t_b = Δ_t + dt_bias
• Inputs   : FBUF(Δ_t), WBUF(dt_bias)
• Outputs  : FBUF(Δ_t_b)
• Parallel : —
• Purpose  : Add bias before Softplus nonlinearity

───────────────────────────────────────────────────────────────
[4] SP_B_CALC
───────────────────────────────────────────────────────────────
• Mode     : 011 (EWM-Outer)
• Compute  : B_x = x_t ⊗ B_raw
• Inputs   : FBUF(x_t), WBUF(B_raw)
• Outputs  : FBUF(B_x)
• Parallel : Softplus(Δ_t_b) → FBUF(spΔ_t)
• Purpose  : 
    - Generate B_x for later ΔB_x computation  
    - Concurrently compute Softplus(Δ_t_b) producing spΔ_t

───────────────────────────────────────────────────────────────
[5] A_CALC
───────────────────────────────────────────────────────────────
• Mode     : 001 (EWM-Matrix)
• Compute  : ΔA = spΔ_t ⊙ A
• Inputs   : FBUF(spΔ_t), WBUF(A)
• Outputs  : FBUF(ΔA)
• Parallel : Softplus module finalizing spΔ_t stream
• Purpose  : Compute ΔA matrix for EXP and A_ht-1 path

───────────────────────────────────────────────────────────────
[6] ΔB_CALC
───────────────────────────────────────────────────────────────
• Mode     : 001 (EWM-Matrix)
• Compute  : ΔB_x = spΔ_t ⊙ B_x
• Inputs   : FBUF(spΔ_t), FBUF(B_x)
• Outputs  : FBUF(ΔB_x)
• Parallel : Start EXP(ΔA)
• Purpose  : Produce ΔB_x for hidden state update (EWA1)

───────────────────────────────────────────────────────────────
[7] EXP_D_CALC
───────────────────────────────────────────────────────────────
• Mode     : 010 (EWM-Vector)
• Compute  : D_x = D ⊙ x_t
• Inputs   : WBUF(D), FBUF(x_t)
• Outputs  : FBUF(D_x)
• Parallel : EXP(ΔA) → FBUF(EXP_ΔA)
• Purpose  : 
    - Compute D_x in D path  
    - Concurrently compute EXP(ΔA) for A_ht-1 generation

───────────────────────────────────────────────────────────────
[8] A_HT_CALC
───────────────────────────────────────────────────────────────
• Mode     : 110 (EWM-Matrix2)
• Compute  : A_ht-1 = EXP_ΔA ⊙ h_{t-1}
• Inputs   : FBUF(EXP_ΔA), HBUF(h_{t-1})
• Outputs  : FBUF(A_ht-1)
• Parallel : —
• Purpose  : Apply exponential modulation to previous hidden state

───────────────────────────────────────────────────────────────
[9] EWA1
───────────────────────────────────────────────────────────────
• Mode     : 101 (EWA-Matrix)
• Compute  : h_t = A_ht-1 + ΔB_x
• Inputs   : FBUF(A_ht-1), FBUF(ΔB_x)
• Outputs  : HBUF(h_t)
• Parallel : Prefetch C_raw and D weights from WBUF
• Purpose  : Update hidden state matrix h_t

───────────────────────────────────────────────────────────────
[10] C_CALC
───────────────────────────────────────────────────────────────
• Mode     : 011 (EWM-Outer)
• Compute  : C_h = h_t ⊗ C_raw
• Inputs   : HBUF(h_t), WBUF(C_raw)
• Outputs  : FBUF(C_h)
• Parallel : —
• Purpose  : Compute outer product for output mixing

───────────────────────────────────────────────────────────────
[11] EWA2
───────────────────────────────────────────────────────────────
• Mode     : 100 (EWA-Vector)
• Compute  : y = C_h + D_x
• Inputs   : FBUF(C_h), FBUF(D_x)
• Outputs  : OBUF(y)
• Parallel : —
• Purpose  : Combine C and D paths to form final output vector y_t

───────────────────────────────────────────────────────────────
[12] DONE
───────────────────────────────────────────────────────────────
• Function : Assert finish=1; notify upper control layer
• Mode     : —
• Inputs   : —
• Outputs  : finish signal
• Parallel : —
• Purpose  : Mark the end of one SSM time-step
───────────────────────────────────────────────────────────────
