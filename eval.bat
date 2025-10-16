@echo off
set args=%1

if "%arg%"=="" (
    echo Usage: train.bat grid ^| random
    goto :eof
)

if "%arg%"=="grid" goto run1
if "%arg%"=="random" goto run2

:run1
echo Running grid eval...
python -m models.eval_cdf_lazy ^
  --ckpt=ckpt/grid/result/best%args%_epe.pt ^
  --predict=current --sigma=0.1^
  --out_dir=.\eval_out --save_csv^
  --amp
goto :eof

:run1
echo Running random eval...
python -m models.eval_cdf_lazy ^
  --ckpt=ckpt/grid/result/best%args%_epe.pt ^
  --predict=current --sigma=0.1^
  --out_dir=.\eval_out --save_csv^
  --amp
goto :eof