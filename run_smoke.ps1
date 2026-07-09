$ErrorActionPreference = "Stop"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:TOKENIZERS_PARALLELISM = "false"

python smoke_test.py
python collect_data.py --samples 64 --clear
python train_oracle.py --epochs 1
python inference_pipeline.py --max-new-tokens 10

