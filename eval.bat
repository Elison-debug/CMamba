@echo off
set arg=%1
set args=%2
set multi=%3

if "%arg%"=="" (
    echo Usage: train.bat grid ^| random | parity
    goto :eof
)

if "%multi%"=="multi"  (goto run4)  else  echo invalid args %multi%...
if "%arg%"=="grid" goto run1
if "%arg%"=="random" goto run2
if "%arg%"=="parity" goto run3
if "%arg%"=="less" goto run5

:run1
echo Running grid eval...
python -m models.eval.eval_cdf_lazy ^
  --ckpt=ckpt/grid/result/best%args%_epe.pt ^
  --predict=current --sigma=0.1^
  --out_dir=.\eval_out --save_csv^
  --amp
goto :eof

:run2
echo Running random eval...
python -m models.eval.eval_cdf_lazy ^
  --ckpt=ckpt/grid/result/best%args%_epe.pt ^
  --predict=current --sigma=0.1^
  --out_dir=eval_out --save_csv^
  --amp
goto :eof

:run3
echo Running parity grid eval...
python -m models.eval.eval_cdf_lazy ^
  --ckpt=ckpt/parity/result/best%args%_epe.pt ^
  --predict=current --sigma=0.1^
  --out_dir=eval_out --save_csv^
  --amp
goto :eof

:run4
echo Running multi eval...
python -m models.eval.eval_multi ^
  --ckpt=ckpt/parity/result/best%args%_epe.pt ^
  --predict=current --sigma=0.1^
  --out_dir=eval_out_parity_%args% --save_csv^
  --amp
goto :eof

:run5
echo Running testing eval...
python -m models.eval.eval_multi ^
  --ckpt=ckpt/lessData/result/best%args%_epe.pt ^
  --predict=current --sigma=0.1^
  --out_dir=eval_out_testing_%args% --save_csv^
  --amp --eval_root=./data/features/lessData/test
goto :eof