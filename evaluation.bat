@echo off
set arg=%1
set fold=%2

if "%arg%"=="" (
    echo Usage: train.bat be(blocked_embargo) ^| wf(eval_walk_forward)
    goto :eof
)

if "%arg%"=="be" goto run1
if "%arg%"=="wf" goto run2

echo Unknown argument: %arg%
goto :eof

:run1
echo Running eval_blocked_embargo...
python -m eval_blocked_embargo ^
  --features_root=./data/features ^
  --ckpt=ckpt_stream/best_fold0.pt ^
  --seq_len=64 --input_dim=2000 ^
  --block=500 --embargo=64
goto :eof

:run2
echo Running eval_walk_forward...
python -m eval_walk_forward ^
  --features_root=./data/features ^
  --ckpt=ckpt_stream/best_fold%fold%.pt ^
  --seq_len=64 --input_dim=2000 ^
  --init_ratio=0.6 --step_ratio=0.1
goto :eof
