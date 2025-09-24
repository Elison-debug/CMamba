#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess LuViRA radio -> NPZ features (lazy frames)  [INDEX-FAST + train/eval split]

需求 / Changes:
- 每个 grid(.mat/.csv) 对齐后，将前 80% 帧保存到 out_dir/train/<base>.npz；
  将后 20% 帧保存到 out_dir/eval/<base>.npz。
- 仅用 **train** 部分统计全局 mean/std（避免信息泄漏），保存为 out_dir/stats_train.npz。
- 其余逻辑延续原脚本：
  • CSV 自动嗅探分隔符与编码；
  • .mat 自动选择 3D 复数张量，整理为 (T,F,A)；
  • IFFT 沿频率轴取前 L taps，按天线归一后展开为实/虚特征；
  • index-fast 对齐：若 MAT 与 CSV 帧数不同，截取公共长度并记录 kept_ratio。

输出：
- out_dir/train/<base>.npz
- out_dir/eval/<base>.npz
- out_dir/stats_train.npz

NPZ 内容：
- feats: (T', Din) float16/32(按 --dtype)
- xy:    (T', 2) float32(米)
- ts:    (T',)   float64(秒)
- meta:  json 字符串，含 taps/input_dim/scale/align/kept_ratio/shape_TFA/split(索引范围等)
"""

import argparse, os, glob, re, json, csv
from typing import Tuple, Dict, Any, Optional, List, cast

import numpy as np
from numpy.fft import ifft
from scipy.io import loadmat

# ----------------------- 时间单位统一：秒 -----------------------
def _infer_seconds(ts: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts).astype(np.float64).reshape(-1)
    if ts.size < 2:
        return ts
    dif = np.diff(ts)
    dif = dif[np.isfinite(dif) & (dif > 0)]
    if dif.size == 0:
        return ts
    md = float(np.median(dif))
    if md > 1e-2:
        scale = 1.0
    elif md > 1e-4:
        scale = 1e-3
    else:
        scale = 1e-6
    return ts * scale

# ----------------------- 读取 GT CSV（t,x,y）到秒/米 -----------------------
def _sniff_delimiter(path: str, encodings: tuple[str, ...] = ("utf-8", "utf-8-sig", "gbk", "latin1")) -> Tuple[str, str]:
    candidates = [",", ";", "\t", " "]
    candidate_str = ",;\t "
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                sample = f.read(65536)
            if not sample:
                continue
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=candidate_str)
                delim = dialect.delimiter
            except Exception:
                counts = {d: sample.count(d) for d in candidates}
                if sum(counts.values()) == 0:
                    delim = ","
                else:
                    delim = max(counts.items(), key=lambda kv: kv[1])[0]
            return enc, delim
        except Exception:
            continue
    return "utf-8", ","


def load_gt_csv(path_csv: str, pos_units: str = "mm") -> Tuple[np.ndarray, np.ndarray]:
    enc, delim = _sniff_delimiter(path_csv)
    data = None
    try:
        data = np.genfromtxt(path_csv, delimiter=delim, names=True, dtype=None, encoding=enc)
    except Exception:
        data = None

    have_header = (isinstance(data, np.ndarray) and getattr(data, "size", 0) > 0 and getattr(data.dtype, "names", None) is not None)

    if not have_header:
        try:
            raw = np.genfromtxt(path_csv, delimiter=delim, dtype=float, encoding=enc)
        except Exception:
            raw = np.genfromtxt(path_csv, delimiter=delim, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        raw = raw[np.isfinite(raw).all(axis=1)]
        if raw.shape[1] < 3 or raw.shape[0] < 2:
            raise ValueError(f"CSV {path_csv}: not enough numeric rows/cols after parsing (rows={raw.shape[0]}, cols={raw.shape[1]})")
        ts = raw[:, 0].astype(np.float64)
        xy = raw[:, 1:3].astype(np.float64)
    else:
        names = [n.lower() for n in (data.dtype.names or [])]  # type: ignore[attr-defined]
        def _find(keys):
            for k in keys:
                if k in names:
                    return k
            return None
        tk = _find(["time", "timestamp", "ts", "t"])
        xk = _find(["x", "pos_x", "px"])
        yk = _find(["y", "pos_y", "py"])
        if tk is None or xk is None or yk is None:
            raw = np.genfromtxt(path_csv, delimiter=delim, dtype=float, encoding=enc)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            raw = raw[np.isfinite(raw).all(axis=1)]
            if raw.shape[1] < 3 or raw.shape[0] < 2:
                raise ValueError(f"CSV {path_csv}: missing columns and fallback failed")
            ts = raw[:, 0].astype(np.float64)
            xy = raw[:, 1:3].astype(np.float64)
        else:
            rec = cast(np.ndarray, data)
            tk_s = cast(str, tk); xk_s = cast(str, xk); yk_s = cast(str, yk)
            ts = np.asarray(rec[tk_s], dtype=np.float64).reshape(-1)
            x  = np.asarray(rec[xk_s], dtype=np.float64).reshape(-1)
            y  = np.asarray(rec[yk_s], dtype=np.float64).reshape(-1)
            m = np.isfinite(ts) & np.isfinite(x) & np.isfinite(y)
            ts = ts[m]; x = x[m]; y = y[m]
            if ts.size < 2:
                raise ValueError(f"CSV {path_csv}: not enough numeric rows after masking")
            xy = np.stack([x, y], axis=1)

    ts = _infer_seconds(ts)
    if pos_units == "mm":
        xy = xy / 1000.0
    elif pos_units != "m":
        raise ValueError("pos_units must be 'mm' or 'm'")
    order = np.argsort(ts)
    ts = ts[order]
    xy = xy[order].astype(np.float32)

    if ts.shape[0] <= 1:
        raise ValueError(f"CSV {path_csv}: parsed only {ts.shape[0]} rows; check delimiter/encoding/header")

    print(f"[CSV] {os.path.basename(path_csv)} rows={ts.shape[0]} enc={enc} delim='{delim}'")
    return ts, xy

# ----------------------- 选择射频张量，统一成 (T,F,A) -----------------------
PREF_KEYS = ["H_ueprocess", "H", "Yc", "CSI", "CIR"]

def select_radio_tensor(md: Dict[str, Any]) -> np.ndarray:
    key: Optional[str] = None
    for k in PREF_KEYS:
        if k in md and isinstance(md[k], np.ndarray):
            key = k
            break
    if key is None:
        cands = [(k, v) for k, v in md.items() if isinstance(v, np.ndarray) and v.ndim == 3]
        if not cands:
            raise ValueError("No 3D ndarray in .mat")
        key, _ = max(cands, key=lambda kv: kv[1].size)

    A = np.asarray(md[key])
    if not np.iscomplexobj(A):
        re_key, im_key = key + "_real", key + "_imag"
        if re_key in md and im_key in md:
            A = np.asarray(md[re_key]) + 1j * np.asarray(md[im_key])
        else:
            raise ValueError(f"{key} is not complex and no ({re_key},{im_key}) provided")

    if A.ndim != 3:
        raise ValueError(f"{key} must be 3D, got {A.ndim}D")

    if key == "H_ueprocess":
        Y = A
    else:
        s0, s1, s2 = A.shape
        dims = [s0, s1, s2]
        t_ax = int(np.argmax(dims))
        if t_ax == 0:
            Yt = A
        elif t_ax == 1:
            Yt = np.transpose(A, (1, 0, 2))
        else:
            Yt = np.transpose(A, (2, 0, 1))
        T, a, b = Yt.shape
        if a >= b:
            Y = Yt
        else:
            Y = np.transpose(Yt, (0, 2, 1))
    return Y  # (T,F,A)


def load_radio_mat(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    md = loadmat(path)
    Y = select_radio_tensor(md)
    ts = None
    for k in ['timestamp', 'timestamps', 'time', 't', 'ts']:
        if k in md and isinstance(md[k], np.ndarray):
            ts0 = np.asarray(md[k]).squeeze()
            if ts0.ndim == 1:
                ts = ts0.astype(np.float64)
                break
    if ts is not None:
        ts = _infer_seconds(ts)
    return Y, ts

# ----------------------- IFFT 提特征（沿频率轴） -----------------------
# def cir_features_batch(Y: np.ndarray, L: int) -> np.ndarray:
#     D = ifft(Y, axis=1)[:, :L, :]                                  # (T, L, A)
#     p = np.sqrt(np.mean(np.abs(D) ** 2, axis=1, keepdims=True)) + 1e-8  # (T,1,A)
#     Dn = D / p
#     feat_cir = np.stack([Dn.real, Dn.imag], 2).transpose(0,1,3,2).reshape(T, L*A*2)
#     power = np.log1p(np.mean(np.abs(D)**2, axis=1))      # (T, A)
#     feats = np.stack([Dn.real, Dn.imag], axis=2)                    # (T, L, 2, A)
#     feats = feats.transpose(0, 1, 3, 2).reshape(D.shape[0], L * D.shape[2] * 2).astype(np.float32)
#     return feats

def cir_features_batch(Y: np.ndarray, L: int) -> np.ndarray:
    # Y: (T, F, A)  取前 L 个 CIR taps
    D = ifft(Y, axis=1)[:, :L, :]                        # (T, L, A)

    # 每根天线一个常数尺度（跨时+tap），避免把时间上的幅度信息抹掉
    p = np.sqrt(np.mean(np.abs(D)**2, axis=(0, 1), keepdims=True)) + 1e-8  # (1,1,A)
    Dn = D / p

    T, _, A = Dn.shape

    # 基础特征：实部/虚部展开 -> (T, 2*L*A)
    feat_cir = np.stack([Dn.real, Dn.imag], axis=2)      # (T, L, 2, A)
    feat_cir = feat_cir.transpose(0, 1, 3, 2).reshape(T, L * A * 2)

    # 追加功率通道（每时刻、每天线）：(T, A)
    power = np.log1p(np.mean(np.abs(D)**2, axis=1))      # (T, A)

    # 拼接得到最终特征：(T, 2*L*A + A)
    feats = np.concatenate([feat_cir.astype(np.float32), power.astype(np.float32)], axis=1)
    return feats


# ----------------------- 文件名匹配 -----------------------

def _extract_num_id(name: str) -> Optional[int]:
    m = re.findall(r'\d+', name)
    if not m:
        return None
    return int(m[-1])


def find_csv_for_base(base: str, csv_all: List[str]) -> Optional[str]:
    base_noext = os.path.splitext(os.path.basename(base))[0]
    for cp in csv_all:
        if os.path.splitext(os.path.basename(cp))[0].lower() == base_noext.lower():
            return cp
    bid = _extract_num_id(base_noext)
    if bid is None:
        return None
    cands = []
    for cp in csv_all:
        name = os.path.splitext(os.path.basename(cp))[0]
        cid = _extract_num_id(name)
        if cid is not None and cid == bid:
            cands.append(cp)
    if not cands:
        return None
    base_low = base_noext.lower()
    cands.sort(key=lambda p: (base_low not in os.path.basename(p).lower(), os.path.basename(p).lower()))
    return cands[0]

# ----------------------- 主程序：两遍流式 + index-fast 对齐 + split -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radio_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--taps", type=int, default=10)
    ap.add_argument("--fps", type=int, default=100)
    ap.add_argument("--pos_units", choices=["mm", "m"], default="mm")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--std_floor", type=float, default=1e-3)
    ap.add_argument("--split", type=float, default=0.8, help="train ratio (0-1)")
    ap.add_argument("--eval_ratio", type=float, default=1, help="eval ratio (0-1) > train ratio")

    args = ap.parse_args()

    train_ratio = float(np.clip(args.split, 0.0, 1.0))
    eval_ratio  = float(np.clip(args.eval_ratio, 0.0, 1.0))
    if not (0.0 < train_ratio < 1.0) or not (train_ratio < eval_ratio <= 1.0):
        raise ValueError("--split must be in (0,1) and --eval_ratio in (split,1].")

    # 子目录
    train_dir = os.path.join(args.out_dir, "train")
    eval_dir  = os.path.join(args.out_dir, "eval")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir,  exist_ok=True)

    mat_paths = sorted(glob.glob(os.path.join(args.radio_dir, "*.mat")))
    if not mat_paths:
        raise RuntimeError("No .mat files found")
    csv_all = sorted(glob.glob(os.path.join(args.gt_dir, "*.csv")))

    # ---------- Pass-1: 统计用于全局尺度（每文件的中位 RMS） ----------
    file_scales = []
    file_meta = []

    for mp in mat_paths:
        base = os.path.splitext(os.path.basename(mp))[0]
        csv_match = find_csv_for_base(base, csv_all)
        if not csv_match:
            print(f"[WARN] no GT for {base}")
            continue
        Y, ts_radio = load_radio_mat(mp)  # (T,F,A)
        T, F, A = Y.shape
        if ts_radio is None:
            ts_radio = np.arange(T, dtype=np.float64) / max(1, args.fps)
        ts_gt, xy_gt = load_gt_csv(csv_match, pos_units=args.pos_units)

        if ts_gt.shape[0] != T:
            L = min(T, ts_gt.shape[0])
            print(f"[WARN] length mismatch for {base}: radio T={T}, gt T={ts_gt.shape[0]} -> cut to {L}")
            kept_ratio = float(L / T)
        else:
            L = T
            kept_ratio = 1.0

        Ltap = args.taps
        D = ifft(Y, axis=1)[:L, :Ltap, :]                           # (L,Ltap,A)
        frms = np.sqrt(np.mean(np.abs(D) ** 2, axis=(1, 2)))        # (L,)
        file_scales.append(float(np.median(frms)))
        file_meta.append((base, mp, csv_match, int(T), int(F), int(A), int(L), kept_ratio))
        print(f"[ALIGN] {base} T={T} F={F} A={A} | mode=index-fast kept={kept_ratio:.2f}")

    if not file_scales:
        raise RuntimeError("No matched MAT/CSV pairs (after index-fast).")

    gscale = float(np.median(file_scales))
    print(f"[GSCALE] gscale={gscale:.6g} from {len(file_scales)} files")

    # ---------- Pass-2: 生成特征 + 仅用 train 部分统计 mean/std ----------
    count_total = 0
    sum_vec: Optional[np.ndarray] = None
    sumsq_vec: Optional[np.ndarray] = None
    Din_ref: int = -1

    train_ratio = float(np.clip(args.split, 0.0, 1.0))
    eval_ratio  = float(np.clip(args.eval_ratio, 0.0, 1.0))

    for base, mp, csv_match, T, F, A, L, kept_ratio in file_meta:
        Y, ts_radio = load_radio_mat(mp)
        if ts_radio is None:
            ts_radio = np.arange(T, dtype=np.float64) / max(1, args.fps)
        ts_gt, xy_gt = load_gt_csv(csv_match, pos_units=args.pos_units)

        # index-fast 对齐到 L
        Y_use = Y[:L]
        ts_use = ts_radio[:L]
        xy_use = xy_gt[:L].astype(np.float32)

        feats = cir_features_batch(Y_use, args.taps)  # (L, Din)
        feats = feats / (gscale + 1e-12)
        Din = feats.shape[1]
        if Din_ref < 0:
            Din_ref = int(Din)
        elif Din != Din_ref:
            raise ValueError(f"Din mismatch: {Din} vs {Din_ref}")

        # split indices
        i_split = max(1, int(round(L * train_ratio)))
        i_split = min(i_split, L - 1) if L >= 2 else 1  # 保证 eval 至少 1 帧（若 L==1 则全放 train）

        # --- train part ---
        feats_tr, xy_tr, ts_tr = feats[:i_split], xy_use[:i_split], ts_use[:i_split]
        meta_tr = dict(
            taps=args.taps, input_dim=Din, scale=gscale,
            align="index-fast", kept_ratio=float(kept_ratio), shape_TFA=[int(T), int(F), int(A)],
            split=dict(role="train", idx=[0, int(i_split)], total=int(L), ratio=float(train_ratio))
        )
        out_tr = os.path.join(train_dir, f"{base}.npz")
        np.savez(out_tr,
                 feats=feats_tr.astype(args.dtype),
                 xy=xy_tr.astype(np.float32),
                 ts=ts_tr.astype(np.float64),
                 meta=json.dumps(meta_tr))
        print(f"[OK][train] {out_tr} feats={feats_tr.shape}")

        # 更新统计（仅 train）
        f64 = feats_tr.astype(np.float64, copy=False)
        if f64.shape[0] > 0:
            if sum_vec is None:
                sum_vec = f64.sum(axis=0)
                sumsq_vec = (f64 ** 2).sum(axis=0)
            else:
                sum_vec += f64.sum(axis=0)
                sumsq_vec += (f64 ** 2).sum(axis=0)
            count_total += f64.shape[0]
        # split eval indices
        e_split = min(int(round(L * eval_ratio)), L)
        if e_split <= i_split:
            e_split = L
        # --- eval part ---
        feats_ev, xy_ev, ts_ev = feats[i_split:e_split], xy_use[i_split:e_split], ts_use[i_split:e_split]
        meta_ev = dict(
            taps=args.taps, input_dim=Din, scale=gscale,
            align="index-fast", kept_ratio=float(kept_ratio), shape_TFA=[int(T), int(F), int(A)],
            split=dict(
                role="eval",
                idx=[int(i_split), int(e_split)],   # [start, end)
                total=int(L),
                ratio=float(e_split - i_split) / float(L),
                exclusive_end=True
                )
            )
        out_ev = os.path.join(eval_dir, f"{base}.npz")
        np.savez(out_ev,
                    feats=feats_ev.astype(args.dtype),
                    xy=xy_ev.astype(np.float32),
                    ts=ts_ev.astype(np.float64),
                    meta=json.dumps(meta_ev))
        print(f"[OK][eval ] {out_ev} feats={feats_ev.shape}")

    if count_total == 0 or sum_vec is None or sumsq_vec is None:
        raise RuntimeError("No train frames to compute stats.")

    mean = (sum_vec / count_total).astype(np.float32)
    var = (sumsq_vec / count_total) - (mean.astype(np.float64) ** 2)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var).astype(np.float32)
    std = np.maximum(std, args.std_floor).astype(np.float32)

    stats_path = os.path.join(args.out_dir, "stats_train.npz")
    np.savez(stats_path,
             feat_mean=mean,
             feat_std=std,
             count=int(count_total),
             Din=int(Din_ref),
             gscale=float(gscale),
             std_floor=float(args.std_floor),
             split_ratio=float(train_ratio))
    print(f"[STATS] {stats_path} Din={Din_ref} count={count_total} gscale={gscale:.6g} train_ratio={train_ratio}")


if __name__ == "__main__":
    main()
