# ⚙️ Mamba SSM Controller State Machine

---

## [0] IDLE — Initialization  
───────────────────────────────────────────────────────────────  
**Function :** Wait for `start` signal, clear internal flags and counters  
**Mode :** —  
**Inputs :** —  
**Outputs :** —  
**Parallel :** —  
**Purpose :** Controller initialization and reset  

---

## [1] X_PROJ — Input Projection  
───────────────────────────────────────────────────────────────  
**Mode :** `000 (MAC)`  
**Compute :** Δ_raw = Wₓₚⱼ ⊙ xₜ  
**Inputs :** WBUF (Wₓₚⱼ), FBUF (xₜ)  
**Outputs :** FBUF (Δ_raw)  
**Parallel :** —  
**Purpose :** Project input feature xₜ into Δ_raw domain  

---

## [2] DT_PROJ — Temporal Projection  
───────────────────────────────────────────────────────────────  
**Mode :** `000 (MAC)`  
**Compute :** Δₜ = WΔ ⊙ Δ_raw  
**Inputs :** WBUF (WΔ), FBUF (Δ_raw)  
**Outputs :** FBUF (Δₜ)  
**Parallel :** —  
**Purpose :** Compute intermediate Δₜ for temporal update  

---

## [3] DT_PROJ_B — Bias Addition  
───────────────────────────────────────────────────────────────  
**Mode :** `100 (EWA-Vector)`  
**Compute :** Δₜ_b = Δₜ + dt_bias  
**Inputs :** FBUF (Δₜ), WBUF (dt_bias)  
**Outputs :** FBUF (Δₜ_b)  
**Parallel :** —  
**Purpose :** Add bias before Softplus nonlinearity  

---

## [4] SP_B_CALC — Softplus & Bₓ Computation  
───────────────────────────────────────────────────────────────  
**Mode :** `011 (EWM-Outer)`  
**Compute :** Bₓ = xₜ ⊗ B_raw  
**Inputs :** FBUF (xₜ), WBUF (B_raw)  
**Outputs :** FBUF (Bₓ)  
**Parallel :** Softplus (Δₜ_b) → FBUF (spΔₜ)  
**Purpose :**  
  - Generate Bₓ for later ΔBₓ computation  
  - Concurrently compute Softplus(Δₜ_b) producing spΔₜ  

---

## [5] A_CALC — ΔA Computation  
───────────────────────────────────────────────────────────────  
**Mode :** `001 (EWM-Matrix)`  
**Compute :** ΔA = spΔₜ ⊙ A  
**Inputs :** FBUF (spΔₜ), WBUF (A)  
**Outputs :** FBUF (ΔA)  
**Parallel :** Softplus module finalizing spΔₜ stream  
**Purpose :** Compute ΔA matrix for EXP and Aₜ₋₁ path  

---

## [6] ΔB_CALC — ΔBₓ Computation  
───────────────────────────────────────────────────────────────  
**Mode :** `001 (EWM-Matrix)`  
**Compute :** ΔBₓ = spΔₜ ⊙ Bₓ  
**Inputs :** FBUF (spΔₜ), FBUF (Bₓ)  
**Outputs :** FBUF (ΔBₓ)  
**Parallel :** Start EXP(ΔA)  
**Purpose :** Produce ΔBₓ for hidden state update (EWA1)  

---

## [7] EXP_D_CALC — Exponential & Dₓ Path  
───────────────────────────────────────────────────────────────  
**Mode :** `010 (EWM-Vector)`  
**Compute :** Dₓ = D ⊙ xₜ  
**Inputs :** WBUF (D), FBUF (xₜ)  
**Outputs :** FBUF (Dₓ)  
**Parallel :** EXP(ΔA) → FBUF (EXP_ΔA)  
**Purpose :**  
  - Compute Dₓ in D path  
  - Concurrently compute EXP(ΔA) for Aₜ₋₁ generation  

---

## [8] A_HT_CALC — Previous Hidden Modulation  
───────────────────────────────────────────────────────────────  
**Mode :** `110 (EWM-Matrix2)`  
**Compute :** Aₜ₋₁ = EXP_ΔA ⊙ hₜ₋₁  
**Inputs :** FBUF (EXP_ΔA), HBUF (hₜ₋₁)  
**Outputs :** FBUF (Aₜ₋₁)  
**Parallel :** —  
**Purpose :** Apply exponential modulation to previous hidden state  

---

## [9] EWA1 — Hidden State Update  
───────────────────────────────────────────────────────────────  
**Mode :** `101 (EWA-Matrix)`  
**Compute :** hₜ = Aₜ₋₁ + ΔBₓ  
**Inputs :** FBUF (Aₜ₋₁), FBUF (ΔBₓ)  
**Outputs :** HBUF (hₜ)  
**Parallel :** Prefetch C_raw and D weights from WBUF  
**Purpose :** Update hidden state matrix hₜ  

---

## [10] C_CALC — Output Outer Product  
───────────────────────────────────────────────────────────────  
**Mode :** `011 (EWM-Outer)`  
**Compute :** C_h = hₜ ⊗ C_raw  
**Inputs :** HBUF (hₜ), WBUF (C_raw)  
**Outputs :** FBUF (C_h)  
**Parallel :** —  
**Purpose :** Compute outer product for output mixing  

---

## [11] EWA2 — Output Aggregation  
───────────────────────────────────────────────────────────────  
**Mode :** `100 (EWA-Vector)`  
**Compute :** y = C_h + Dₓ  
**Inputs :** FBUF (C_h), FBUF (Dₓ)  
**Outputs :** OBUF (y)  
**Parallel :** —  
**Purpose :** Combine C and D paths to form final output vector yₜ  

---

## [12] DONE — Completion  
───────────────────────────────────────────────────────────────  
**Function :** Assert `finish = 1`; notify upper control layer  
**Mode :** —  
**Inputs :** —  
**Outputs :** finish signal  
**Parallel :** —  
**Purpose :** Mark the end of one SSM time-step  

---


