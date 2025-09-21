@echo off
set arg=%1

if "%arg%"=="" (
    echo Usage: train.bat grid ^| random
    goto :eof
)

if "%arg%"=="grid" goto run1
if "%arg%"=="random" goto run2

echo Unknown argument: %arg%
goto :eof

:run1
echo Running grid...
python -m models.train_regression_lazy ^
    --features_root=./data/features/grid ^
    --seq_len=12 --input_dim=2000 ^
    --proj_dim=64 --d_model=128 ^
    --n_layer=4 --patch_len=8 ^
    --stride=4 --batch_size=32 ^
    --epochs=20 --lr=3e-4 ^
    --wd=0.01 --out_dir=./ckpt/grid ^
    --amp --accum=4
goto :eof

:run2
echo Running random...
python -m models.train_regression_lazy ^
    --features_root=./data/features/random ^
    --seq_len=64 --input_dim=2000 ^
    --proj_dim=128 --d_model=256 ^
    --n_layer=6 --patch_len=8 ^
    --stride=4 --batch_size=16 ^
    --epochs=100 --lr=1e-4 ^
    --wd=0.05 --out_dir=./ckpt/random ^
    --amp --accum=2
goto :eof
