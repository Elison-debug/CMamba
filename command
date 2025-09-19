#python ./dataprocess/preprocess_luvira.py   --radio_dir ./data/radio   --gt_dir    ./data/ground_truth/radio   --out_npz   ./data/data_transfered/prepared_data.npz  --seq_len 12 --taps 10 --fps 100 --train_split 0.8   --predict current
#python ./dataprocess/preprocess_luvira.py   --radio_dir ./data/radio   --gt_dir    ./data/ground_truth/radio   --out_npz   ./data/data_transfered/prepared_data.npz  --seq_len 12 --taps 10 --fps 100 --train_split 0.8   --predict current
  #transfer data to npz file
  python -m datasets.preprocess_luvira_lazy `
  --radio_dir=./data/radio `
  --gt_dir=./data/ground_truth/radio `
  --out_dir=./data/features   `
  --taps=10 --fps=100 `
  --pos_units=mm --dtype=float16 `
  --split=0.8



  #train & eval
  python -m models.train_regression_lazy `
  --features_root=./data/features `
  --seq_len=12 --input_dim=2000 `
  --proj_dim=64 --d_model=128 `
  --n_layer=4 --patch_len=6 `
  --stride=3 --batch_size=32 `
  --epochs=60 --lr=3e-4 `
  --wd=0.01 --out_dir=./ckpt `
  --amp --accum=4
  
  #train & eval
  python -m models.train_regression_lazy `
  --features_root=./data/features `
  --seq_len=12 --input_dim=2000 `
  --proj_dim=64 --d_model=128 `
  --n_layer=4 --patch_len=8 `
  --stride=4 --batch_size=32 `
  --epochs=60 --lr=3e-4 `
  --wd=0.05 --out_dir=./ckpt `
  --amp --accum=4
  
  
  #eval
  python -m models.eval_cdf_lazy `
  --features_root=.\data\features `
  --ckpt=.\ckpt\best.pt `
  --seq_len=12 --input_dim=2000 `
  --proj_dim=64 --d_model=128 --n_layer=4 `
  --patch_len=8 --stride=4 `
  --predict=next `
  --batch_size=64 --workers=4 `
  --out_dir=.\eval_out --save_csv

  #eval with live plotting and without specifying model structure
  python -m models.eval_cdf_lazy `
    --features_root=.\data\features `
    --ckpt=.\ckpt\best.pt `
    --out_dir=.\eval_out `
    --amp `
    --live_bins=120 --live_max_err=10 --live_every=1 `
    --save_csv



   
  
  
