#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess LuViRA radio -> NPZ features (lazy frames)  [INDEX-FAST]
前提：每个 Grid 的 CSV(ref) 与 MAT(radio) 帧数完全一致(例如 4274)。
对齐策略：优先走 “index-fast” 一一对齐(kept=1.00)。

输出：
- 每文件：{base}.npz  (feats[T', Din], xy[T',2], ts[T'], meta=json)；通常 T'=T(index-fast 保留全部)。
- 全局：stats.npz (feat_mean/feat_std/count/Din/gscale/std_floor)
"""
import argparse, os, glob, re, json
from typing import Tuple, Dict, Any, Optional

import numpy as np
from numpy.fft import ifft
from scipy.io import loadmat
import os, re
from typing import Optional, List

# ----------------------- 时间单位统一：秒 -----------------------
def _infer_seconds(ts: np.ndarray) -> np.ndarray:
    """
    根据中位采样间隔把时间戳转成秒。
    - median_dt > 1e-2 认为已是秒
    - 1e-4 < median_dt <= 1e-2 认为是毫秒 → /1e3
    - 否则认为是微秒 → /1e6
    """
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
import csv
from typing import Tuple
def _sniff_delimiter(path: str, encodings: tuple[str, ...] = ("utf-8", "utf-8-sig", "gbk", "latin1")) -> Tuple[str, str]:
    """
    返回 (encoding, delimiter)。优先用 csv.Sniffer 从文件前 64KB 嗅探分隔符。
    若嗅探失败，按候选集计数回退。最终失败则返回 ('utf-8', ',').
    """
    candidates = [",", ";", "\t", " "]
    candidate_str = ",;\t "  # <-- 关键：传给 sniff 的必须是字符串，而不是 list

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                sample = f.read(65536)
            if not sample:
                continue

            # 首选：Sniffer 嗅探（注意 delimiters 传字符串）
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=candidate_str)
                delim = dialect.delimiter
            except Exception:
                # 回退：统计候选分隔符出现次数，取最多的那个
                counts = {d: sample.count(d) for d in candidates}
                if sum(counts.values()) == 0:
                    delim = ","
                else:
                    # 用 items() + key=lambda kv: kv[1]，避免 Pylance 对 max(dict, key=...) 的报错
                    delim = max(counts.items(), key=lambda kv: kv[1])[0]

            return enc, delim
        except Exception:
            # 尝试下一个编码
            continue

    # 编码遍历失败时的兜底
    return "utf-8", ","

def load_gt_csv(path_csv: str, pos_units: str = "mm") -> Tuple[np.ndarray, np.ndarray]:
    """
    鲁棒读取 CSV：自动嗅探分隔符与编码；优先按表头取列，否则退化为无表头前三列。
    返回：
        ts: (T,) 秒
        xy: (T,2) 米
    """
    enc, delim = _sniff_delimiter(path_csv)
    # 尝试带表头
    data = None
    try:
        data = np.genfromtxt(path_csv, delimiter=delim, names=True, dtype=None, encoding=enc)
    except Exception:
        data = None

    have_header = (isinstance(data, np.ndarray) and data.size > 0 and getattr(data.dtype, "names", None) is not None)

    if not have_header:
        # 无表头或失败：退化为“前三列”
        try:
            raw = np.genfromtxt(path_csv, delimiter=delim, dtype=float, encoding=enc)
        except Exception:
            # 最后一搏：不指定编码让 numpy 自行处理
            raw = np.genfromtxt(path_csv, delimiter=delim, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        # 过滤空行/非数值
        raw = raw[np.isfinite(raw).all(axis=1)]
        if raw.shape[1] < 3 or raw.shape[0] < 2:
            raise ValueError(f"CSV {path_csv}: not enough numeric rows/cols after parsing (rows={raw.shape[0]}, cols={raw.shape[1]})")
        ts = raw[:, 0].astype(np.float64)
        xy = raw[:, 1:3].astype(np.float64)
    else:
        names = [n.lower() for n in data.dtype.names]  # type: ignore[attr-defined]
        def _find(keys):
            for k in keys:
                if k in names:
                    return k
            return None
        tk = _find(["time", "timestamp", "ts", "t"])
        xk = _find(["x", "pos_x", "px"])
        yk = _find(["y", "pos_y", "py"])
        if tk is None or xk is None or yk is None:
            # 表头不规范：退化到无表头模式再试一次
            raw = np.genfromtxt(path_csv, delimiter=delim, dtype=float, encoding=enc)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            raw = raw[np.isfinite(raw).all(axis=1)]
            if raw.shape[1] < 3 or raw.shape[0] < 2:
                raise ValueError(f"CSV {path_csv}: missing columns and fallback failed")
            ts = raw[:, 0].astype(np.float64)
            xy = raw[:, 1:3].astype(np.float64)
        else:
            ts = np.asarray(data[tk], dtype=np.float64).reshape(-1)
            x  = np.asarray(data[xk], dtype=np.float64).reshape(-1)
            y  = np.asarray(data[yk], dtype=np.float64).reshape(-1)
            # 过滤非数值
            m = np.isfinite(ts) & np.isfinite(x) & np.isfinite(y)
            ts = ts[m]; x = x[m]; y = y[m]
            if ts.size < 2:
                raise ValueError(f"CSV {path_csv}: not enough numeric rows after masking")
            xy = np.stack([x, y], axis=1)

    # 单位与排序
    ts = _infer_seconds(ts)
    if pos_units == "mm":
        xy = xy / 1000.0
    elif pos_units != "m":
        raise ValueError("pos_units must be 'mm' or 'm'")
    order = np.argsort(ts)
    ts = ts[order]
    xy = xy[order].astype(np.float32)

    # 关键自检：如果只剩 1 行，直接报错（防止你的 T=1）
    if ts.shape[0] <= 1:
        raise ValueError(f"CSV {path_csv}: parsed only {ts.shape[0]} rows; check delimiter/encoding/header")

    # 调试打印（可注释）
    print(f"[CSV] {os.path.basename(path_csv)} rows={ts.shape[0]} enc={enc} delim='{delim}'")
    return ts, xy


# ----------------------- 选择射频张量，统一成 (T,F,A) -----------------------
PREF_KEYS = ["H_ueprocess", "H", "Yc", "CSI", "CIR"]

def select_radio_tensor(md: Dict[str, Any]) -> np.ndarray:
    """
    从 .mat 选择复数射频张量，返回 shape=(T,F,A) 复数数组。
    优先变量名 H_ueprocess；若无则回退到体量最大 3D 数组，并整理到 (T,F,A)。
    """
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

    # 目标：(T, F, A)
    if key == "H_ueprocess":
        Y = A  # 数据已是 (T,F,A)
    else:
        s0, s1, s2 = A.shape
        # 先把“时间”放 axis=0：经验上 T≥F,A（fallback 情形）
        dims = [s0, s1, s2]
        t_ax = int(np.argmax(dims))
        if t_ax == 0:
            Yt = A
        elif t_ax == 1:
            Yt = np.transpose(A, (1, 0, 2))
        else:
            Yt = np.transpose(A, (2, 0, 1))
        # 再把较大的那一维作为频率轴 F
        T, a, b = Yt.shape
        if a >= b:
            Y = Yt
        else:
            Y = np.transpose(Yt, (0, 2, 1))
    return Y  # (T,F,A)


def load_radio_mat(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """返回 Y:(T,F,A) 复数, ts: 秒级 1D 或 None"""
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
def cir_features_batch(Y: np.ndarray, L: int) -> np.ndarray:
    """
    Y: (T, F, A) 复数
    返回 feats: (T, Din=2*L*A) float32
    流程：IFFT(axis=1) → 取前 L tap → 每天线归一 → 实/虚展平
    """
    D = ifft(Y, axis=1)[:, :L, :]                                  # (T, L, A)
    p = np.sqrt(np.mean(np.abs(D) ** 2, axis=1, keepdims=True)) + 1e-8  # (T,1,A)
    Dn = D / p
    feats = np.stack([Dn.real, Dn.imag], axis=2)                    # (T, L, 2, A)
    feats = feats.transpose(0, 1, 3, 2).reshape(D.shape[0], L * D.shape[2] * 2).astype(np.float32)
    return feats


def _extract_num_id(name: str) -> Optional[int]:
    """
    从文件名中提取核心编号，如 'Grid0142' -> 142, 'grid101' -> 101。
    若没有数字，返回 None。
    """
    m = re.findall(r'\d+', name)
    if not m:
        return None
    # 取最后一段数字，并去掉前导 0
    return int(m[-1])

def find_csv_for_base(base: str, csv_all: List[str]) -> Optional[str]:
    """
    优先精确同名（不含扩展名）；否则用“数字编号相等”匹配（忽略大小写、前导零）。
    多个候选时优先包含 base（大小写不敏感）的那一个。
    """
    base_noext = os.path.splitext(os.path.basename(base))[0]
    # 1) 先找同名
    for cp in csv_all:
        if os.path.splitext(os.path.basename(cp))[0].lower() == base_noext.lower():
            return cp

    # 2) 用编号匹配
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

    # 3) 有多个候选：优先名称中包含 base 关键字的
    base_low = base_noext.lower()
    cands.sort(key=lambda p: (base_low not in os.path.basename(p).lower(), os.path.basename(p).lower()))
    return cands[0]




# ----------------------- 主程序：两遍流式 + index-fast 对齐 -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radio_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--taps", type=int, default=10)
    ap.add_argument("--fps", type=int, default=100)  # 当 MAT 无时间戳时生成秒级 ts
    ap.add_argument("--pos_units", choices=["mm", "m"], default="mm")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--std_floor", type=float, default=1e-3)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mat_paths = sorted(glob.glob(os.path.join(args.radio_dir, "*.mat")))
    if not mat_paths:
        raise RuntimeError("No .mat files found")

    csv_all = sorted(glob.glob(os.path.join(args.gt_dir, "*.csv")))

    # ---------- Pass-1: 稳健幅度统计（不驻留特征） ----------
    file_scales = []
    file_meta = []

    for mp in mat_paths:
        base = os.path.splitext(os.path.basename(mp))[0]

        # 匹配 CSV：优先同名，否则按数字片段
        csv_match = find_csv_for_base(base, csv_all)
        if not csv_match:
            print(f"[WARN] no GT for {base}")
            continue

        Y, ts_radio = load_radio_mat(mp)  # (T,F,A)
        T, F, A = Y.shape
        if ts_radio is None:
            ts_radio = np.arange(T, dtype=np.float64) / max(1, args.fps)

        ts_gt, xy_gt = load_gt_csv(csv_match, pos_units=args.pos_units)

        # index-fast 对齐：要求长度完全一致，否则截短到公共长度并告警
        if ts_gt.shape[0] != T:
            L = min(T, ts_gt.shape[0])
            print(f"[WARN] length mismatch for {base}: radio T={T}, gt T={ts_gt.shape[0]} -> cut to {L}")
            xy = xy_gt[:L].astype(np.float32)
            ts_r_use = ts_radio[:L]
            kept_ratio = float(L / T)
        else:
            xy = xy_gt.astype(np.float32)
            ts_r_use = ts_radio
            kept_ratio = 1.0

        # 用 IFFT 前 L taps 的逐帧 RMS 的中位数，作为该文件的“尺度”
        Ltap = args.taps
        D = ifft(Y, axis=1)[:, :Ltap, :]                            # (T,L,A)
        frms = np.sqrt(np.mean(np.abs(D) ** 2, axis=(1, 2)))        # (T,)
        # 与 index-fast 一致，若截短则也按 L 统计
        frms = frms[:xy.shape[0]]
        file_scales.append(float(np.median(frms)))
        file_meta.append((base, mp, csv_match, int(T), int(F), int(A), kept_ratio))
        print(f"[ALIGN] {base} T={T} F={F} A={A} | mode=index-fast kept={kept_ratio:.2f}")

    if not file_scales:
        raise RuntimeError("No matched MAT/CSV pairs (after index-fast).")

    gscale = float(np.median(file_scales))
    print(f"[GSCALE] gscale={gscale:.6g} from {len(file_scales)} files")

    # ---------- Pass-2: 生成特征 + 全局 mean/std ----------
    count_total = 0
    sum_vec: Optional[np.ndarray] = None
    sumsq_vec: Optional[np.ndarray] = None
    Din_ref: int = -1

    for base, mp, csv_match, T, F, A in file_meta:
        Y, ts_radio = load_radio_mat(mp)  # (T,F,A)
        if ts_radio is None:
            ts_radio = np.arange(T, dtype=np.float64) / max(1, args.fps)
        ts_gt, xy_gt = load_gt_csv(csv_match, pos_units=args.pos_units)

        # index-fast 对齐
        if ts_gt.shape[0] != T:
            L = min(T, ts_gt.shape[0])
            xy = xy_gt[:L].astype(np.float32)
            ts = ts_radio[:L]
            Y_use = Y[:L]
            kept_ratio = float(L / T)
        else:
            xy = xy_gt.astype(np.float32)
            ts = ts_radio
            Y_use = Y
            kept_ratio = 1.0

        feats = cir_features_batch(Y_use, args.taps)  # (T', Din=2*L*A)
        Din = feats.shape[1]
        if Din_ref < 0:
            Din_ref = int(Din)
        elif Din != Din_ref:
            raise ValueError(f"Din mismatch: {Din} vs {Din_ref}")

        # 应用全局尺度
        feats = feats / (gscale + 1e-12)

        # 统计（流式 sum/sumsq）
        f64 = feats.astype(np.float64, copy=False)
        if sum_vec is None:
            sum_vec = f64.sum(axis=0)
            sumsq_vec = (f64 ** 2).sum(axis=0)
        else:
            sum_vec += f64.sum(axis=0)
            sumsq_vec += (f64 ** 2).sum(axis=0)
        count_total += f64.shape[0]

        # 落盘
        feats_out = feats.astype(args.dtype)
        out = os.path.join(args.out_dir, f"{base}.npz")
        meta = dict(
            taps=args.taps, input_dim=Din, scale=gscale,
            align="index-fast", kept_ratio=float(kept_ratio),
            shape_TFA=[int(T), int(F), int(A)],
        )
        np.savez(out,
                 feats=feats_out,
                 xy=xy.astype(np.float32),
                 ts=ts.astype(np.float64),
                 meta=json.dumps(meta))
        print(f"[OK] {out} feats={feats_out.shape} Din={Din} kept={kept_ratio:.2f}")

    if count_total == 0 or sum_vec is None or sumsq_vec is None:
        raise RuntimeError("No valid frames to compute stats.")

    mean = (sum_vec / count_total).astype(np.float32)
    var = (sumsq_vec / count_total) - (mean.astype(np.float64) ** 2)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var).astype(np.float32)
    std = np.maximum(std, args.std_floor).astype(np.float32)

    stats_path = os.path.join(args.out_dir, "stats.npz")
    np.savez(stats_path,
             feat_mean=mean,
             feat_std=std,
             count=int(count_total),
             Din=int(Din_ref),
             gscale=float(gscale),
             std_floor=float(args.std_floor))
    print(f"[STATS] {stats_path} Din={Din_ref} count={count_total} gscale={gscale:.6g}")


if __name__ == "__main__":
    main()
