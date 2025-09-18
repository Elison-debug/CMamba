#python ./dataprocess/preprocess_luvira.py   --radio_dir ./data/radio   --gt_dir    ./data/ground_truth/radio   --out_npz   ./data/data_transfered/prepared_data.npz  --seq_len 12 --taps 10 --fps 100 --train_split 0.8   --predict current
#python ./dataprocess/preprocess_luvira.py   --radio_dir ./data/radio   --gt_dir    ./data/ground_truth/radio   --out_npz   ./data/data_transfered/prepared_data.npz  --seq_len 12 --taps 10 --fps 100 --train_split 0.8   --predict current
  #transfer data to npz file
  python ./datasets/preprocess_luvira_lazy.py --radio_dir ./data/radio --gt_dir    ./data/ground_truth/radio --out_dir   ./data/features   --taps 10 --fps 100 --pos_units mm --dtype float16
  #train & eval
  python ./models/train_regression_lazy.py --features_root ./data/features --seq_len 12 --input_dim 2000 --proj_dim 64 --d_model 128 --n_layer 4 --patch_len 8 --stride 4 --batch_size 32 --epochs 60 --lr 2e-3 --wd 0.05 --amp --out_dir ./ckpt
  #eval
  python -m ./models/eval_cdf_lazy --features_root ./data/features  --ckpt ./ckpt/best.pt   --seq_len 12 --input_dim 2000   --proj_dim 64 --d_model 128 --n_layer 4   --patch_len 8 --stride 4   --batch_size 64 --out_dir ./eval_out





   
  
  
