#!/usr/bin/env python3
"""
preprocess_luvira_lazy.py
把 LuViRA 的 radio .mat + ground truth .csv 转成帧级特征：
- feats (T, Din): float16
- xy    (T, 2):  float32，单位米
- ts    (T,)
- meta  (JSON字符串)
"""
import argparse, os, glob, re, json
import numpy as np, pandas as pd
from numpy.fft import ifft
from scipy.io import loadmat


def guess_radio_tensor(md):
    cands = [(k,v) for k,v in md.items() if isinstance(v,np.ndarray) and v.ndim==3]
    if not cands: raise ValueError("No 3D ndarray in .mat")
    key, arr = cands[0]
    if np.iscomplexobj(arr):
        Y = arr
    else:
        re_key, im_key = key+"_real", key+"_imag"
        if re_key in md and im_key in md:
            Y = md[re_key] + 1j*md[im_key]
        else:
            raise ValueError("Need complex or *_real/_imag")
    T,N,M = Y.shape
    if N < M and M < 16:  # heuristics
        Y = np.transpose(Y,(0,2,1))
    return Y

def load_radio_mat(path):
    md = loadmat(path)
    Y = guess_radio_tensor(md)
    ts = None
    for k in ['timestamp','timestamps','time','t']:
        if k in md and isinstance(md[k],np.ndarray) and md[k].ndim==1:
            ts = md[k].astype(np.float64).reshape(-1)
            break
    return Y, ts

def load_gt_csv(path_csv, pos_units="mm"):
    with open(path_csv,"r",encoding="utf-8") as f:
        first = ""
        for line in f:
            if line.strip():
                first=line.strip(); break
    def _to_m(xy): return xy/1000.0 if pos_units=="mm" else xy
    if first.startswith("#"):
        import re as _re
        cols=[c.strip() for c in _re.split(r"[,\s]+",first.lstrip("#").strip()) if c.strip()]
        df=pd.read_csv(path_csv,comment="#",header=None,names=cols,sep=r"[,\s]+",engine="python")
        cols_l={c.lower():c for c in df.columns}
        ts=df[cols_l["timestamps"]].to_numpy(float)
        xy=df[[cols_l["x"],cols_l["y"]]].to_numpy(float)
        xy=_to_m(xy)
    else:
        df=pd.read_csv(path_csv,sep=r"\s+",engine="python")
        if df.shape[1]==17:
            df.columns=["timestamps","X","Y","Z","Roll","Pitch","Yaw","Residual",
                        "R00","R01","R02","R10","R11","R12","R20","R21","R22"]
            ts=df["timestamps"].to_numpy(float)
            xy=df[["X","Y"]].to_numpy(float); xy=_to_m(xy)
        else:
            raise ValueError("Unexpected CSV format")
    order=ts.argsort()
    return ts[order], xy[order].astype("float32")

def nearest_align(ts_radio, ts_gt, xy_gt):
    idx=np.abs(ts_gt[None,:]-ts_radio[:,None]).argmin(axis=1)
    return xy_gt[idx]

def cir_feature(H,L):
    D=ifft(H,axis=0)[:L,:]
    p=np.sqrt(np.mean(np.abs(D)**2,axis=0,keepdims=True))+1e-8
    Dn=D/p
    return np.stack([Dn.real,Dn.imag],axis=0).reshape(-1).astype(np.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--radio_dir",required=True)
    ap.add_argument("--gt_dir",required=True)
    ap.add_argument("--out_dir",required=True)
    ap.add_argument("--taps",type=int,default=10)
    ap.add_argument("--fps",type=int,default=100)
    ap.add_argument("--pos_units",choices=["mm","m"],default="mm")
    ap.add_argument("--dtype",choices=["float16","float32"],default="float16")
    args=ap.parse_args()
    os.makedirs(args.out_dir,exist_ok=True)

    mat_paths=sorted(glob.glob(os.path.join(args.radio_dir,"*.mat")))
    scales, outputs=[],[]
    for mp in mat_paths:
        base=os.path.splitext(os.path.basename(mp))[0]
        csv_cands = glob.glob(os.path.join(args.gt_dir, '*.csv'))
        csv_match = None
        for cp in csv_cands:
            if os.path.splitext(os.path.basename(cp))[0] == base:
                csv_match = cp; break
        if csv_match is None:
            # 用数字 id 模糊匹配
            num = ''.join(re.findall(r'\d+', base))
            for cp in csv_cands:
                if num and num in os.path.basename(cp):
                    csv_match = cp; break
        if not csv_match:
            print(f"[WARN] no GT for {base}"); continue
        Yc,ts_radio=load_radio_mat(mp)
        T,N,M=Yc.shape
        if ts_radio is None: ts_radio=np.arange(T)/args.fps
        ts_gt,xy_gt=load_gt_csv(csv_match,pos_units=args.pos_units)
        xy=nearest_align(ts_radio,ts_gt,xy_gt)
        Din=2*M*args.taps
        feats=np.empty((T,Din),np.float32)
        for t in range(T): feats[t]=cir_feature(Yc[t],args.taps)
        scale=np.linalg.norm(feats)/np.sqrt(feats.size)+1e-8
        scales.append(scale); outputs.append((base,feats,xy,ts_radio,Din))
    gscale=float(np.median(scales))
    for base,feats,xy,ts_radio,Din in outputs:
        feats=(feats/gscale).astype(args.dtype)
        out=os.path.join(args.out_dir,f"{base}.npz")
        meta=dict(taps=args.taps,input_dim=Din,scale=gscale)
        np.savez(out,feats=feats,xy=xy,ts=ts_radio,meta=json.dumps(meta))
        print(f"[OK] {out} feats={feats.shape} Din={Din}")

if __name__=="__main__": main()
