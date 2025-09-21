@echo off
set arg=%1

if "%arg%"=="" (
    echo Usage: train.bat grid ^| random
    goto :eof
)

if "%arg%"=="grid"   goto run1
if "%arg%"=="random" goto run2

echo Unknown argument: %arg%
goto :eof

:run1
echo Running grid...
python -m datasets.preprocess_luvira_lazy ^
  --radio_dir=./data/radio/grid ^
  --gt_dir=./data/truth/grid ^
  --out_dir=./data/features/grid   ^
  --taps=10 --fps=100 ^
  --pos_units=mm --dtype=float16 ^
  --split=0.2 --eval_ratio=0.5
goto :eof

:run2
echo Running random...
python -m datasets.preprocess_luvira_lazy ^
  --radio_dir=./data/radio/random ^
  --gt_dir =./data/truth/random ^
  --out_dir=./data/features/random   ^
  --taps=10 --fps=100 ^
  --pos_units=mm --dtype=float16 ^
  --split=0.2 --eval_ratio=0.5
goto :eof

