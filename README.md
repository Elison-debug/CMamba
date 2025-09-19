# CMamba for Wireless Positioning on LuViRA Dataset

This repo contains a PyTorch implementation of a **channel-sequence regression model** based on
[Simple-CMamba](https://github.com/Prismadic/simple-CMamba), adapted for **wireless localization** tasks
on the [LuViRA Dataset](https://github.com/ilaydayaman/LuViRA_Dataset).

Our goal is to explore **lightweight Mamba-style architectures** for radio-based positioning,
and compare them with CNN baselines (e.g. FCNN).


## 📂 Project Structure
```
.
├── datasets/ # Lazy sliding-window dataset loader  /
├── models/ # CMamba regressor, training & evaluation scripts/
├── data/ # Put LuViRA preprocessed .npz features here/
├── ckpt/ # Model checkpoints/
└── eval_out/ # Evaluation results (plots, csv, npz)/
```


## ⚙️ Setup

```bash
conda create -n mamba-pos python=3.10
conda activate mamba-pos
pip install -r requirements.txt
```

## Installation
To use this model, you need to have the following libraries installed:
- `torch`
- `einops`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `wandb`

You can install them using pip:

```bash
pip install torch einops numpy pandas scikit-learn matplotlib wandb
```

📊 Data Preparation

Download the [LuViRA Dataset](https://github.com/ilaydayaman/LuViRA_Dataset)
 
Convert raw .mat/.csv into .npz feature files using the provided preprocessing scripts:
```bash
python dataprocess/preprocess_luvira.py --input_root ./LuViRA --output_root ./data/features
```
Each `-.npz` will contain:
- `-feats`: time-series feature tensor
- `-xy`: ground-truth positions

## 🚀 Training
```bash
python models/train_regression_lazy.py \
  --features_root ./data/features \
  --seq_len 12 --input_dim 2000 \
  --proj_dim 64 --d_model 128 --n_layer 4 \
  --patch_len 8 --stride 4 \
  --batch_size 32 --epochs 60 \
  --lr 2e-3 --wd 0.05 --amp \
  --out_dir ./ckpt
```
## 📈 Evaluation
After training, run:
```bash
python -m models.eval_cdf_lazy \
  --features_root ./data/features \
  --ckpt ./ckpt/best.pt \
  --out_dir ./eval_out \
  --amp --save_csv
```
This will produce:
- `-cdf.png`: error CDF plot
- `-err_hist.png`: histogram of position errors
- `-val_preds.npz`: predictions + ground truth + errors
- `-pred_vs_true.csv`: optional CSV for analysis

## 🙏 Acknowledgements

- Dataset  :  [LuViRA Dataset](https://github.com/ilaydayaman/LuViRA_Dataset)
- Base Code  : [Prismadic/simple-CMamba](https://github.com/Prismadic/simple-CMamba/tree/main))
- Inspired by the Mamba architecture (Selective State Spaces).
