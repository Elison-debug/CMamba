#!/usr/bin/env python3
import argparse, os, numpy as np, torch, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.frames_lazy import FramesLazyDataset
from models.mamba_regressor import MambaRegressor

def euclid_err(a,b): return np.sqrt(np.sum((a-b)**2,axis=1))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features_root",required=True)
    ap.add_argument("--ckpt",required=True)
    ap.add_argument("--seq_len",type=int,required=True)
    ap.add_argument("--input_dim",type=int,required=True)
    ap.add_argument("--proj_dim",type=int,default=64)
    ap.add_argument("--d_model",type=int,default=128)
    ap.add_argument("--n_layer",type=int,default=4)
    ap.add_argument("--patch_len",type=int,default=8)
    ap.add_argument("--stride",type=int,default=4)
    ap.add_argument("--batch_size",type=int,default=64)
    ap.add_argument("--out_dir",default="./eval_out")
    args=ap.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl=DataLoader(FramesLazyDataset(args.features_root,args.seq_len,"current","val"),
                  batch_size=args.batch_size,shuffle=False,num_workers=2,pin_memory=True)

    model = MambaRegressor(Din=args.input_dim, K=args.seq_len,
                           proj_dim=args.proj_dim, d_model=args.d_model,
                           n_layer=args.n_layer, patch_len=args.patch_len, stride=args.stride).to(device)

    ckpt=torch.load(args.ckpt,map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    yts, yps = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb=xb.to(device,non_blocking=True)
            yb=yb.squeeze(1).to(device,non_blocking=True)
            yhat=model(xb)
            yts.append(yb.cpu().numpy()); yps.append(yhat.cpu().numpy())
    y_true=np.concatenate(yts); y_pred=np.concatenate(yps)
    err=euclid_err(y_pred,y_true)
    os.makedirs(args.out_dir,exist_ok=True)
    e=np.sort(err); y=np.arange(1,len(e)+1)/len(e)
    plt.figure(figsize=(5,4),dpi=160); plt.plot(e,y); plt.grid(True,linestyle="--",linewidth=0.5)
    plt.xlabel("Position error (m)"); plt.ylabel("CDF"); plt.title("Error CDF"); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,"cdf.png")); plt.close()
    print(f"N={len(err)}  mean={err.mean():.4f}  median={np.median(err):.4f}")

    np.savez(os.path.join(args.out_dir,"val_preds.npz"), y_true=y_true, y_pred=y_pred, err=err)
    print(f"[OK] saved plots/arrays under {args.out_dir}")

if __name__=="__main__": main()
