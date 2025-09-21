
# LuViRA Streaming + Multi-head CMamba (Refactor)

This refactor adds:

- **Multi-head supervision**: predict `x_t` (main), `x_{t+1}` (aux), with optional `v_t`, `a_t` heads.
- **Packed streaming**: treat each grid as a stream; cut into segments of length `K=seq_len` and **carry hidden state** across segments.
- **GroupKFold by grid** (balanced by motion type) with `k>=5`.
- **Supplementary evaluations**:
  - **Blocked + Embargo** (same-scene isolation) — odd blocks train, even blocks eval with an `E>=K` frame gap.
  - **Walk-forward** (short horizon online) — roll forward `train -> eval 5–10% -> append -> ...`.
- **Type coverage**: `straight/circle/random` motion types are inferred heuristically per grid and reported.

## 1) Preprocess (per-grid consolidated files)

```bash
# 第一步：预处理 + 生成 folds.json（如已有人为分折，也可以手工提供 --folds_json）
python -m datasets.preprocess_luvira_stream `
--radio_dir=./data/radio `
--gt_dir=./data/ground_truth/radio `
--out_dir=./data/features `
--taps=10 --pos_units=mm --write_folds --folds=5
```

Outputs:

- `data/features/all/<GRID>.npz` with arrays: `feats (T,Din)`, `xy (T,2)`, `ts (T)`, `v (T,2)`, `a (T,2)`, `meta(kind,grid,T,...)`.
- `data/features/stats_global.npz` (global mean/std; training uses train-only is also supported if you wish to recompute).
- `data/features/folds.json` (optional), GroupKFold (balanced by type).

```bash
# 第二步：基于指定 fold 的 TRAIN grids 计算 train-only 统计（并写出 stats_train.npz）
python -m datasets.preprocess_luvira_stream `
--radio_dir=./data/radio `
--gt_dir=./data/ground_truth/radio `
--out_dir=./data/features `
--taps=10 --pos_units=mm `
--folds_json=./data/features/folds.json  `
--fold_for_stats=0
```

运行后会在 ./data/features/ 下生成：
stats_train_fold0.npz
stats_train.npz （直接给训练脚本用）

```bash
# 第二步：一次性为所有 folds 产出各自的 train-only 统计
python -m datasets.preprocess_luvira_stream `
--radio_dir=./data/radio `
--gt_dir=./data/ground_truth/radio `
--out_dir=./data/features `
--taps=10 --pos_units=mm `
--folds_json=./data/features/folds.json  `
--make_all_fold_stats
```

## 2) Train (streaming, multi-head)

```bash
python train_stream.py `
  --features_root=./data/features `
  --seq_len=12 --input_dim=2000 `
  --proj_dim=64 --d_model=128 `
  --n_layer=4 --patch_len=8 `
  --stride=4 --batch_size=4 `
  --kfold=5 --fold=0 `
  --out_vel --out_acc `
  --epochs=10 --lr=3e-4 `
  --w_pos=1.0 --w_next=0.4 --w_vel=0.2 --w_acc=0.1 `
  --amp
```

Each batch contains several **grids**, each grid **packed** into `S` segments of `K` frames. The loop streams through `S` with state being carried across segments for each grid.

Loss = SmoothL1 on heads with weights `w_*`, and primary metric is EPE (meters).

## 3) Evaluate

### 3.1 Blocked + Embargo

```bash
python eval_blocked_embargo.py \
  --features_root ./data/features \
  --ckpt ckpt_stream/best_fold0.pt \
  --seq_len 64 --input_dim <Din> \
  --block 500 --embargo 64
```

### 3.2 Walk-forward

```bash
python eval_walk_forward.py \
  --features_root ./data/features \
  --ckpt ckpt_stream/best_fold0.pt \
  --seq_len 64 --input_dim <Din> \
  --init_ratio 0.6 --step_ratio 0.1
```

Both evaluators report **overall EPE** and (for Blocked+Embargo) **per-type EPE**.

## Notes

- Streaming in this design passes state **across segments of length `K`** (chunk-level). This matches CMamba’s sequence processing while respecting causality.
- If you need **per-frame** outputs, we can expose an additional `step()` API and a tiny head computed at each patch position; current design supervises **segment end** (`t`) and `t+1`.
- Motion type labels are heuristic; if you have ground-truth labels, drop them into `meta.kind` during preprocess and they will be used verbatim.
- Use **GroupKFold** to strictly isolate grids across folds (no same-scene leakage).

---

Happy streaming! 🚀
