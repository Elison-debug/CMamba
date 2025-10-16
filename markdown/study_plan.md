# Study Plan: Mamba-style Models for Radio Positioning (Sept–Dec)

**Student:** Chenxi Zhang (LTH)

**Status:** Proposal (survey → implementation)

---

## 1. Motivation

State Space Models (SSMs) such as **Mamba** provide linear-time sequence modeling with strong long-range dependency capture. Building on Lund’s **FCNN** baselines for LuViRA radio positioning, this study focuses on **channel-enhanced CMamba** training and rigorous evaluation on LuViRA, producing **hardware-oriented metrics** (parameter count, FLOPs, memory/bandwidth, quantization sensitivity) for accelerator co-design.

>While the proposal begins with a survey, the plan proceeds—subject to time and compute availability—to a minimal implementation and evaluation of a cMamba model on LuViRA.

---

## 2. Research Questions (High-level)

1. **Suitability:** Do Mamba-style SSMs conceptually match the characteristics of radio positioning data (multi-channel, long sequences)?  
2. **Evaluation Design:** What would constitute a fair and reproducible evaluation protocol (splits, metrics, baselines) for LuViRA?  
3. **Efficiency Considerations:** What are the likely efficiency advantages/trade-offs (parameters, memory patterns, compute/bandwidth balance) compared with FCNNs, **in theory**?  
4. **Future Directions:** What minimal software experiments would be most informative, if time permits?

*Note:* These questions guide a **survey and design document**, not an implementation.

---

## 3. Preliminary Study Plan

- **3.1 Literature Review**
  - SSM foundations and Mamba-style models (core ideas, complexity, known strengths/limitations).
  - Radio positioning with machine learning (typical preprocessing, features, and evaluation practices).
  - Report gaps and open questions relevant to LuViRA.

- **3.2 Dataset Understanding (Desk Study)**
  - High-level data characteristics (sequence length ranges, channel structure).
  - Common train/validation/test patterns reported in prior work.
  - Typical error metrics (e.g., median error, MAE, tail percentiles).

- **3.3 Evaluation Design Draft (for future implementation)**
  - Baselines to include (e.g., existing FCNN configuration).
  - Proposed protocols: in-trajectory vs. cross-trajectory evaluation.
  - Metrics & reporting: accuracy, stability across splits, and **if later implemented**, efficiency indicators (parameters, FLOPs, peak memory).

- **3.4 Feasibility & Risk Assessment**
  - Identify likely bottlenecks (data I/O, sequence windowing, multi-channel handling).
  - Risks in fairness (data leakage, inconsistent preprocessing) and mitigation strategies for later work.

- **3.5 Optional (If time permits; still non-implementation)**
  - Sketch minimal experiment matrix (model variants, small-scale ablations) **without** committing to execution.

---

## 4. Objectives

1. **Reproducible Software Baselines** – Train and validate CMamba on LuViRA with conservative, well-documented settings(Reproducible).  
2. **Efficiency Assessment** – Compare training/inference time, peak memory, parameter count, and FLOPs at comparable accuracy.  
3. **Generalization & Robustness** – Evaluate **in-trajectory** and **cross-trajectory** settings; ablate power channel, and channel mixup (including $\sigma$=0).  
4. **Hardware-Oriented Analytics (software-side):** Per-layer params/FLOPs/bytes and quantization sensitivity to guide hardware acceleration.

---

## 5. Milestones & Timeline

| Week | Dates (approx.) |Milestone |
| ---: | :-------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| 1 | Sep 15–19    | Literature review.build reading list.|
| 2 | Sep 22–26    | Collect core SSM/Mamba & radio-positioning papers to a structured summary                                     |
| 3 | Sep 29–Oct 3 | Desk study of LuViRA: data characteristics, typical splits/metrics; draft risk log.                           |
| 4 | Oct 6–10     | Draft evaluation design: baselines (FCNN, CMamba), in-traj vs cross-traj protocols, reporting template.       |
| 5 | Oct 13–17    | Feasibility mapping: expected params/FLOPs/memory; identify I/O/sequence bottlenecks.                         |
| 6 | Oct 20–24    | Reproducible pipeline setup: data loader/splits verified;  FCNN baseline short run (sanity numbers).          |
| 7 | Oct 27–31    | Implement minimal CMamba; train/validate (in-trajectory) with conservative settings; log accuracy/efficiency. |
| 8 | Nov 3–7      | Ablations v1: power channel on/off; $\sigma$=0 vs light mixup; pick stable config for next steps.                    |
| 9 | Nov 10–14    | Cross-trajectory evaluation with the picked config; compare to FCNN baseline.                                 |
| 10 | Nov 17–21    | Efficiency assessment: per-layer params/FLOPs/activation bytes; throughput & peak-mem curves.                |
| 11 | Nov 24–28    | Quantization sensitivity (PTQ FP16/INT8) & hardware-oriented summary (Design Hints v1).                      |
| 12 | Dec 1–5      | Final training runs with locked config (in-traj & cross-traj); save best checkpoints & logs.                 |
| 13 | Dec 8–12     | Consolidate results: comparison tables, plots; write Results & Discussion draft.                             |
| 14 | Dec 15–19    | Finalize report & slides; package code/configs/checkpoints; submit.                                          |

## 6. Expected Outcomes

- **1.** Reproducible software pipeline (scripts/configs/logs) for **FCNN & CMamba** on LuViRA.  
- **2.** Accuracy & efficiency comparison:  
  - In-trajectory: CMamba achieves comparable or better error with similar or fewer parameters.  
  - Cross-trajectory: with mild regularization (small-$\sigma$ mixup/RevIN), improved robustness.  
- **3.** Hardware-oriented analytics: per-layer params/FLOPs/bytes, peak memory, arithmetic intensity, and quantization sensitivity, plus a concise **Design Hints** sheet for accelerator exploration.  
- **4.** Concise final report and presentation focusing on software training results and actionable guidance for future hardware co-design.

>If time permits, an appendix listing a minimal experiment instruction for a later implementation phase may be included.

---

## 7. Resources & Reproducibility Requirements

Single-GPU (RTX 3080) with FP16; fixed seeds; results and configs versioned in Git; one-click script to reproduce tables/plots.

## 8. References

1. Tian G., Yaman I., Sandra M., et al. *High-precision ML-based indoor localization with massive MIMO*, ICC 2023.  
2. Yaman I., Tian G., Tegler E., et al. *LuViRA dataset validation and discussion: comparing vision, radio, and audio sensors for indoor localization*, J-ISPN, 2024.

> Additional references on SSMs/Mamba and sequence modeling will be added during Weeks 1–2 of the review.
