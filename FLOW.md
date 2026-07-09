# PathOracle CPU MVP Execution Flow

## Phase 1: Environment

Run from `outputs/pathoracle_cpu_mvp`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set CPU-friendly environment variables:

```powershell
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"
$env:TOKENIZERS_PARALLELISM="false"
```

Checkpoint:

```powershell
python smoke_test.py
```

Expected key output:

```text
model=distilgpt2
layers=6
hidden_size=768
early_layers=0..1
skipped_layers=2..3
late_layers=4..5
```

## Phase 2: Smoke Data Collection

```powershell
python collect_data.py --samples 64 --clear
```

Checkpoint:

- `oracle_data_distilgpt2` exists.
- It contains 64 `pair_*.npz` files.
- `metadata.json` reports `samples: 64`.

Stop and reduce scale if Windows starts swapping heavily.

## Phase 3: Smoke Training

```powershell
python train_oracle.py --epochs 1
```

Checkpoint:

- `oracle_distilgpt2.pth` exists.
- `oracle_distilgpt2.json` exists.
- Console prints `avg_mse` and `avg_cosine`.

Continue if training completes and loss is finite.

## Phase 4: Smoke Inference

```powershell
python inference_pipeline.py --max-new-tokens 10
```

Checkpoint:

- Output text is printed.
- `sample_ppl` is finite.
- No NaN or Inf errors occur.

## Phase 5: Full Distilgpt2 MVP

```powershell
python collect_data.py --samples 512 --clear
python train_oracle.py --epochs 3
python inference_pipeline.py --max-new-tokens 20
python evaluate.py --max-texts 20 --max-prompts 5
```

Record:

- Final `avg_mse`.
- Final `avg_cosine`.
- Original model PPL.
- PathOracle PPL.
- Relative PPL increase.
- Whether generated text is readable.

## Phase 6: Optional GPT-2 Small Upgrade

Run only after `distilgpt2` completes.

```powershell
python smoke_test.py --preset gpt2
python collect_data.py --preset gpt2 --samples 256 --clear
python train_oracle.py --preset gpt2 --epochs 1
python inference_pipeline.py --preset gpt2 --max-new-tokens 10
```

If stable:

```powershell
python collect_data.py --preset gpt2 --samples 2000 --clear
python train_oracle.py --preset gpt2 --epochs 3
python evaluate.py --preset gpt2 --max-texts 20 --max-prompts 5
```

## Go / No-Go Rules

Continue when:

- Model loading succeeds.
- Data collection completes.
- Training loss is finite.
- Inference logits are finite.
- Short generation completes.

Reduce scale when:

- Python memory use approaches 6 GB.
- System becomes unresponsive from swapping.
- One `distilgpt2` epoch exceeds 2 hours.
- PPL or logits produce NaN or Inf.

Fallback reductions:

- Use `--samples 64`.
- Use `--batch-size 2`.
- Use `--max-len 32`.
- Keep `--oracle-type mlp`.

## Non-Goals On This Device

Do not run:

- `gpt2-medium`
- 100K hidden-state samples
- sequence length 256 or 512
- GPU acceleration through Vega 8
- GLM-scale experiments
- long-context experiments

