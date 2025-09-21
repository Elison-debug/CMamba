@echo off
set arg=%1

if "%arg%"=="" (
    echo Usage: train.bat run1 ^| run2
    echo using default：    --features_root=./data/features
    echo --seq_len=64 --input_dim=2000
    echo --proj_dim=64 --d_model=128
    echo --n_layer=4 --patch_len=8
    echo --stride=4 --batch_size=8
    echo --kfold=5 --fold=0
    echo --epochs=60 --lr=3e-4
    echo --w_pos=1.0 --w_next=0.3 --w_vel=0.0 --w_acc=0.0
    echo --amp
    goto :run1
)

if "%arg%"=="run1" goto run1
if "%arg%"=="run2" goto run2

echo Unknown argument: %arg%
goto :eof

:run1
echo Running run1...
python -m train_stream ^
   --features_root=./data/features ^
   --seq_len=64 --input_dim=2000 ^
   --proj_dim=64 --d_model=128 ^
   --n_layer=4 --patch_len=8 ^
   --stride=4 --batch_size=8 ^
   --kfold=5 --fold=0 ^
   --epochs=60 --lr=3e-4 ^
   --w_pos=1.0 --w_next=0.3 --w_vel=0.0 --w_acc=0.0 ^
   --amp
goto :eof

:run2
echo Running run2...
python -m train_stream ^
   --features_root=./data/features ^
   --seq_len=64 --input_dim=2000 ^
   --proj_dim=64 --d_model=128 ^
   --n_layer=4 --patch_len=8 ^
   --stride=4 --batch_size=8 ^
   --kfold=5 --fold=0 ^
   --epochs=60 --lr=3e-4 ^
   --w_pos=1.0 --w_next=0.3 --w_vel=0.0 --w_acc=0.0 ^
   --amp
goto :eof
