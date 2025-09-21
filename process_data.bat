@echo off
set arg=%1

if "%arg%"=="" (
    echo Usage: runall ^| global
    echo Using default：runall
    goto :runall
)

if "%arg%"=="runall"   goto runall
if "%arg%"=="global"   goto global

echo Unknown argument: %arg%
goto :eof

:runall
echo Running grid...
python -m datasets.preprocess_luvira_stream `
  --radio_dir=./data/radio `
  --gt_dir=./data/ground_truth/radio `
  --out_dir=./data/features `
  --taps=10 --pos_units=mm --write_folds --folds=5

python -m datasets.preprocess_luvira_stream `
  --radio_dir=./data/radio `
  --gt_dir=./data/ground_truth/radio `
  --out_dir=./data/features `
  --taps=10 --pos_units=mm `
  --folds_json=./data/features/folds.json  `
  --make_all_fold_stats
goto :eof

:global
echo Running random...
python -m datasets.preprocess_luvira_stream `
  --radio_dir=./data/radio `
  --gt_dir=./data/ground_truth/radio `
  --out_dir=./data/features `
  --taps=10 --pos_units=mm --write_folds --folds=5
goto :eof

