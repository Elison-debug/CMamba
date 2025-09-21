
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datasets/preprocess_luvira_stream.py
- Read LuViRA .mat radio and .csv GT
- IFFT features (taps=L)
- Save per-grid consolidated file under out/all/GRIDxxxx.npz, with fields:
    feats (T, Din), xy (T,2), ts (T,), v (T,2), a (T,2), meta (json)
- Compute 'kind' of motion per grid: straight / circle / random (heuristic)
- Save stats_train.npz if --train_grids provided (only train used for mean/std)
- Optionally write folds.json with GroupKFold mapping (balanced by kind)
"""
from __future__ import annotations
import argparse, os, glob, re, json
from typing import Optional, List ,Iterable
import numpy as np
from numpy.fft import ifft
from scipy.io import loadmat
from pathlib import Path
from utils.splits import make_group_kfold, GridInfo

PREF_KEYS = ["H_ueprocess", "H", "Yc", "CSI", "CIR"]

def _extract_num_id(name: str) -> Optional[int]:
    m = re.findall(r'\d+', name)
    return int(m[-1]) if m else None

def _infer_seconds(ts: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts).astype(np.float64).reshape(-1)
    if ts.size < 2: return ts
    dif = np.diff(ts); dif = dif[np.isfinite(dif) & (dif > 0)]
    if dif.size == 0: return ts
    md = float(np.median(dif))
    scale = 1.0 if md > 1e-2 else (1e-3 if md > 1e-4 else 1e-6)
    return ts * scale

def load_radio_mat(path: str):
    md = loadmat(path)
    key = None
    for k in PREF_KEYS:
        if k in md and isinstance(md[k], np.ndarray): key=k; break
    if key is None:
        cands = [(k, v) for k, v in md.items() if isinstance(v, np.ndarray) and v.ndim==3]
        if not cands: raise ValueError("No 3D ndarray in .mat")
        key, _ = max(cands, key=lambda kv: kv[1].size)
    A = np.asarray(md[key])
    if not np.iscomplexobj(A):
        rk,ik = key+"_real", key+"_imag"
        if rk in md and ik in md: A = np.asarray(md[rk]) + 1j*np.asarray(md[ik])
        else: raise ValueError(f"{key} not complex and no ({rk},{ik})")
    if A.ndim!=3: raise ValueError(f"{key} must be 3D")
    s0,s1,s2=A.shape; dims=[s0,s1,s2]; t_ax=int(np.argmax(dims))
    if t_ax==0: Yt=A
    elif t_ax==1: Yt=np.transpose(A,(1,0,2))
    else: Yt=np.transpose(A,(2,0,1))
    T,a,b=Yt.shape; Y = Yt if a>=b else np.transpose(Yt,(0,2,1))
    ts=None
    for k in ['timestamp','timestamps','time','t','ts']:
        if k in md and isinstance(md[k], np.ndarray):
            ts0 = np.asarray(md[k]).squeeze()
            if ts0.ndim==1: ts = ts0.astype(np.float64); break
    if ts is not None: ts=_infer_seconds(ts)
    return Y, ts

def cir_features_batch(Y: np.ndarray, L: int) -> np.ndarray:
    D = ifft(Y, axis=1)[:, :L, :]                                  # (T,L,A)
    p = np.sqrt(np.mean(np.abs(D)**2, axis=1, keepdims=True)) + 1e-8
    Dn = D / p
    feats = np.stack([Dn.real, Dn.imag], axis=2)                    # (T,L,2,A)
    feats = feats.transpose(0,1,3,2).reshape(D.shape[0], L*D.shape[2]*2).astype(np.float32)
    return feats

def compute_kinematics(xy: np.ndarray, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = xy.shape[0]
    if ts is None or ts.shape[0] != T:
        dt = np.ones(T, dtype=np.float32)
    else:
        dt = np.diff(ts, prepend=ts[0]).astype(np.float32)
        dt[dt<=0] = np.median(dt[dt>0]) if np.any(dt>0) else 1.0
    v = np.zeros_like(xy, dtype=np.float32)
    a = np.zeros_like(xy, dtype=np.float32)
    v[1:] = (xy[1:] - xy[:-1]) / dt[1:, None]
    a[2:] = (v[2:]  - v[1:-1]) / dt[2:, None]
    return v, a

def classify_motion(xy: np.ndarray) -> str:
    # Straight: high straightness ratio
    return "straight"
    disp = np.linalg.norm(xy[-1]-xy[0]) + 1e-6
    path = np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)) + 1e-6
    straightness = disp / path
    # Curvature proxy
    dv = np.diff(xy, axis=0)
    ang = []
    for i in range(1, dv.shape[0]):
        u, v = dv[i-1], dv[i]
        nu = np.linalg.norm(u)+1e-9; nv=np.linalg.norm(v)+1e-9
        cos = (u@v)/(nu*nv)
        cos = np.clip(cos, -1.0, 1.0)
        ang.append(np.arccos(cos))
    kappa = float(np.mean(ang)) if ang else 0.0
    # Closedness
    closed = np.linalg.norm(xy[-1]-xy[0]) < 0.2*path
    if straightness > 0.9 and kappa < 0.1: return "straight"
    if closed and kappa > 0.3: return "circle"
    return "random"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radio_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--taps", type=int, default=10)
    ap.add_argument("--fps", type=int, default=100)
    ap.add_argument("--pos_units", choices=["mm","m"], default="mm")
    ap.add_argument("--dtype", choices=["float16","float32"], default="float16")
    ap.add_argument("--std_floor", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--write_folds", action="store_true")
    ap.add_argument("--folds_json", type=str, default="", help="Optional: use an existing folds.json (overrides --write_folds).")
    ap.add_argument("--fold_for_stats", type=int, default=-1, help="If >=0, recompute stats from the TRAIN grids of this fold.")
    ap.add_argument("--make_all_fold_stats", action="store_true", help="If set, compute stats_train_fold{i}.npz for all folds in folds.json.")

    args = ap.parse_args()

    out_all = Path(args.out_dir)/"all"; out_all.mkdir(parents=True, exist_ok=True)
    mat_paths = sorted(glob.glob(os.path.join(args.radio_dir, "*.mat")))
    if not mat_paths: raise SystemExit("No .mat files")
    csv_all = sorted(glob.glob(os.path.join(args.gt_dir, "*.csv")))

    def find_csv_for_base(base: str) -> Optional[str]:
        base_noext = os.path.splitext(os.path.basename(base))[0]
        for cp in csv_all:
            if os.path.splitext(os.path.basename(cp))[0].lower() == base_noext.lower():
                return cp
        bid = _extract_num_id(base_noext)
        if bid is None: return None
        cands = []
        for cp in csv_all:
            name = os.path.splitext(os.path.basename(cp))[0]
            m = re.findall(r'\d+', name)
            if m and int(m[-1]) == bid: cands.append(cp)
        if not cands: return None
        base_low = base_noext.lower()
        cands.sort(key=lambda p: (base_low not in os.path.basename(p).lower(), os.path.basename(p).lower()))
        return cands[0]

    Din_ref = None
    stats_sum = None; stats_sumsq=None; count=0
    grid_infos: List[GridInfo] = []

    for mp in mat_paths:
        base = os.path.splitext(os.path.basename(mp))[0]
        cp = find_csv_for_base(base)
        if not cp:
            print(f"[WARN] no GT for {base}"); continue
        Y, ts_radio = load_radio_mat(mp)
        T,F,A = Y.shape
        if ts_radio is None:
            ts_radio = np.arange(T, dtype=np.float64)/max(1,args.fps)
        # load CSV
        # flexible load
        import csv
        with open(cp, "r", encoding="utf-8", newline="") as f:
            sample = f.read(65536)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
            delim = dialect.delimiter
        except Exception:
            delim = ","
        raw = np.genfromtxt(cp, delimiter=delim, dtype=float)
        if raw.ndim==1: raw = raw.reshape(1,-1)
        raw = raw[np.isfinite(raw).all(axis=1)]
        ts_gt = raw[:,0].astype(np.float64); xy_gt = raw[:,1:3].astype(np.float64)
        # normalize units
        if args.pos_units == "mm": xy_gt = xy_gt / 1000.0
        ts_gt = _infer_seconds(ts_gt)
        L = min(Y.shape[0], ts_gt.shape[0])
        Y = Y[:L]; ts = ts_radio[:L]; xy = xy_gt[:L].astype(np.float32)

        feats = cir_features_batch(Y, args.taps)  # (L, Din)
        Din = feats.shape[1]
        if Din_ref is None: Din_ref = Din
        elif Din != Din_ref: raise RuntimeError(f"Din mismatch: {Din} vs {Din_ref}")

        v, a = compute_kinematics(xy, ts)
        kind = classify_motion(xy)

        meta = dict(grid=base, T=int(L), F=int(F), A=int(A), taps=int(args.taps), kind=kind)
        outp = out_all/f"{base}.npz"
        np.savez(outp,
                 feats=feats.astype(args.dtype),
                 xy=xy.astype(np.float32),
                 ts=ts.astype(np.float64),
                 v=v.astype(np.float32),
                 a=a.astype(np.float32),
                 meta=json.dumps(meta))
        print(f"[OK] {outp.name} T={L} Din={Din} kind={kind}")
        # Accumulate stats for train later (we don't know folds yet -> compute overall; users can recompute for train only)
        f64 = feats.astype(np.float64)
        stats_sum = f64.sum(axis=0) if stats_sum is None else stats_sum + f64.sum(axis=0)
        stats_sumsq = (f64**2).sum(axis=0) if stats_sumsq is None else stats_sumsq + (f64**2).sum(axis=0)
        count += f64.shape[0]

        grid_infos.append(GridInfo(grid=base, T=L, kind=kind))

    # Save global stats (optional baseline)
    if count>0:
        mean = (stats_sum / count).astype(np.float32) # type: ignore
        var  = (stats_sumsq / count) / count - (mean.astype(np.float64)**2) # type: ignore
        std = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
        std = np.maximum(std, args.std_floor)
        np.savez(Path(args.out_dir)/"stats_global.npz", feat_mean=mean, feat_std=std, count=int(count),
                 Din=int(Din_ref), std_floor=float(args.std_floor)) # type: ignore
        print(f"[STATS] stats_global.npz Din={Din_ref} count={count}")

    # Folds.json（生成或加载）
    folds_json_path = Path(args.out_dir)/"folds.json"
    folds_data = None
    if args.folds_json:
        p = Path(args.folds_json)
        if not p.exists():
            raise SystemExit(f"--folds_json not found: {p}")
        folds_json_path = p
    elif args.write_folds and len(grid_infos) >= args.folds:
        folds = make_group_kfold(grid_infos, k=args.folds)
        data = [{"fold":i, "train":tr, "val":va} for i,(tr,va) in enumerate(folds)]
        with open(folds_json_path,"w",encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK] wrote folds.json with {args.folds} folds -> {folds_json_path}")

    if folds_json_path.exists():
        with open(folds_json_path, "r", encoding="utf-8") as f:
            folds_data = json.load(f)

    # ===== 新增：基于 fold 的 TRAIN grids 重算 feat_mean/std =====
    def _compute_stats_from_grids(grids: Iterable[str], all_dir: Path, std_floor: float, din_hint: int|None=None):
        sumv = None; sumsq = None; cnt = 0; Din_local=None
        for g in grids:
            p = all_dir / f"{g}.npz"
            if not p.exists():
                # 兼容已有文件名不带 .npz 后缀的情况
                cand = list(all_dir.glob(f"{g}*.npz"))
                if not cand: 
                    print(f"[WARN] stats: missing {g}.npz"); 
                    continue
                p = cand[0]
            d = np.load(p, allow_pickle=True)
            feats = d["feats"].astype(np.float64)
            if Din_local is None: Din_local = feats.shape[1]
            if din_hint is not None and Din_local != din_hint:
                raise RuntimeError(f"[STATS] Din mismatch inside train-only stats: {Din_local} vs hint {din_hint} at {p.name}")
            sumv   = feats.sum(0) if sumv is None else sumv + feats.sum(0)
            sumsq  = (feats**2).sum(0) if sumsq is None else sumsq + (feats**2).sum(0)
            cnt   += feats.shape[0]
        if cnt == 0 or sumv is None or sumsq is None:
            raise SystemExit("[STATS] No frames found for train-only stats. Check folds/train list.")
        mean = (sumv / cnt).astype(np.float32)
        var  = (sumsq / cnt) - (mean.astype(np.float64)**2)
        std  = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
        std  = np.maximum(std, std_floor)
        return mean, std, cnt, int(Din_local if Din_local is not None else (din_hint or -1))

    if folds_data is not None:
        all_dir = Path(args.out_dir)/"all"
        # 按需为指定 fold 生成 train-only 统计
        if args.fold_for_stats >= 0:
            entry = next((e for e in folds_data if int(e.get("fold",-1)) == int(args.fold_for_stats)), None)
            if entry is None:
                raise SystemExit(f"[STATS] fold {args.fold_for_stats} not found in {folds_json_path}")
            tr_grids = entry.get("train", [])
            mean,std,cnt,din = _compute_stats_from_grids(tr_grids, all_dir, args.std_floor, din_hint=Din_ref)
            outp = Path(args.out_dir)/f"stats_train_fold{args.fold_for_stats}.npz"
            np.savez(outp, feat_mean=mean, feat_std=std, count=int(cnt), Din=int(din), std_floor=float(args.std_floor))
            # 方便训练脚本直接用
            np.savez(Path(args.out_dir)/"stats_train.npz", feat_mean=mean, feat_std=std, count=int(cnt), Din=int(din), std_floor=float(args.std_floor))
            print(f"[STATS] wrote {outp.name} and stats_train.npz (Din={din}, count={cnt})")

        # 或者一次性为所有 folds 生成
        if args.make_all_fold_stats:
            for e in folds_data:
                k = int(e.get("fold",-1))
                tr_grids = e.get("train", [])
                if not tr_grids:
                    print(f"[STATS] skip fold {k}: empty train list"); 
                    continue
                mean,std,cnt,din = _compute_stats_from_grids(tr_grids, all_dir, args.std_floor, din_hint=Din_ref)
                outp = Path(args.out_dir)/f"stats_train_fold{k}.npz"
                np.savez(outp, feat_mean=mean, feat_std=std, count=int(cnt), Din=int(din), std_floor=float(args.std_floor))
                print(f"[STATS] wrote {outp.name} (Din={din}, count={cnt})")

    print("[DONE]")
if __name__ == "__main__":
    main()
