@echo off
set args=%1

if "%args%" NEQ "" goto run1

echo Unknown argument: %arg%
goto :eof

:run1
echo Running eval...
python -m models.eval_cdf_lazy ^
  --features_root=./data/features ^
  --ckpt=ckpt/grid/result/best%args%_epe.pt ^
  --predict=next ^
  --seq_len=64 --input_dim=2100 ^
  --out_dir=.\eval_out --save_csv

goto :eof
