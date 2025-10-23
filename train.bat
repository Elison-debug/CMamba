@echo off
set arg=%1
set resume=%2

if "%arg%"=="" (
    echo Usage: train.bat grid ^| random
    goto :eof
)
if "%resume%"=="resume" (
    if "%arg%"=="less" goto run6
    goto run4
)
if "%arg%"=="grid"   goto run1
if "%arg%"=="random" goto run2
if "%arg%"=="parity" goto run3
if "%arg%"=="less"   goto run5


echo Unknown argument: %arg%
goto :eof

:run1
echo Running grid...
python -m models.training.train_regression_lazy ^
    --features_root=./data/features/grid ^
    --seq_len=12 --input_dim=2100 ^
    --proj_dim=64 --d_model=128 ^
    --n_layer=4 --patch_len=8 ^
    --stride=4 --batch_size=32 ^
    --epochs=20 --lr=3e-4 --sigma=0.15 ^
    --wd=0.01 --out_dir=./ckpt/grid ^
    --amp --accum=4
goto :eof

:run2
echo Running random...
python -m models.training.train_regression_lazy ^
    --features_root=./data/features/random ^
    --seq_len=12 --input_dim=2100 ^
    --proj_dim=64 --d_model=128 ^
    --n_layer=4 --patch_len=8 ^
    --stride=4 --batch_size=32 ^
    --epochs=60 --lr=3e-4 --sigma=0.1^
    --wd=0.01 --out_dir=./ckpt/random ^
    --amp --accum=4
goto :eof

:run3
echo Running parity grid...
python -m models.training.train_regression_lazy ^
    --features_root=./data/features/parity ^
    --seq_len=12 --input_dim=2100 ^
    --proj_dim=64 --d_model=128 ^
    --n_layer=4 --patch_len=8 ^
    --stride=4 --batch_size=32 ^
    --epochs=20 --lr=3e-4 --sigma=0.1 ^
    --wd=0.01 --out_dir=./ckpt/parity ^
    --amp --accum=4
goto :eof

:run4
echo Running parity grid resume from ./ckpt/parity/checkpoint.pt...
python -m models.training.train_regression_lazy ^
    --features_root=./data/features/parity ^
    --seq_len=12 --input_dim=2100 ^
    --proj_dim=64 --d_model=128 ^
    --n_layer=4 --patch_len=8 ^
    --stride=4 --batch_size=32 ^
    --epochs=20 --lr=3e-4 --sigma=0.1 ^
    --wd=0.01 --out_dir=./ckpt/parity ^
    --amp --accum=4 ^
    --resume=./ckpt/parity/checkpoint.pt ^
    --resume_strict --load_ema_for_eval

goto :eof

:run5
echo Running parity with lessData...
python -m models.training.train_regression_lazy ^
    --features_root=./data/features/lessData ^
    --seq_len=12 --input_dim=2100 ^
    --proj_dim=64 --d_model=128 ^
    --n_layer=4 --patch_len=8 ^
    --stride=4 --batch_size=32 ^
    --epochs=20 --lr=3e-4 --sigma=0.1 ^
    --wd=0.01 --out_dir=./ckpt/lessData ^
    --amp --accum=4
goto :eof

:run6
echo Running parity grid resume from ./ckpt/lessData/checkpoint.pt...
python -m models.training.train_regression_lazy ^
    --features_root=./data/features/lessData ^
    --seq_len=12 --input_dim=2100 ^
    --proj_dim=64 --d_model=128 ^
    --n_layer=4 --patch_len=8 ^
    --stride=4 --batch_size=32 ^
    --epochs=20 --lr=3e-4 --sigma=0.1 ^
    --wd=0.01 --out_dir=./ckpt/lessData ^
    --amp --accum=4 ^
    --resume=./ckpt/lessData/checkpoint.pt ^
    --resume_strict --load_ema_for_eval

goto :eof
