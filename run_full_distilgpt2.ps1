$ErrorActionPreference = "Stop"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:TOKENIZERS_PARALLELISM = "false"

python smoke_test.py --preset distilgpt2
python collect_data.py --preset distilgpt2 --samples 512 --clear
python train_oracle.py --preset distilgpt2 --epochs 3
python inference_pipeline.py --preset distilgpt2 --max-new-tokens 20
python evaluate.py --preset distilgpt2 --max-texts 20 --max-prompts 5

