# PathOracle CPU MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a CPU-only PathOracle MVP on `distilgpt2`, with an optional `gpt2` upgrade path.

**Architecture:** The original model is run manually through early layers to collect `early` hidden states and through skipped layers to collect `target` hidden states. A tiny oracle learns `early -> target`, and inference runs early layers, oracle, late layers, final layer norm, and LM head.

**Tech Stack:** Python, PyTorch CPU, Hugging Face Transformers, Hugging Face Datasets, NumPy, tqdm, PowerShell.

## Global Constraints

- Hardware target: Ryzen 5 2500U / 8 GB RAM / CPU-only.
- Default model: `distilgpt2`.
- Default sequence length: 64 tokens.
- Default smoke samples: 64.
- Default full samples: 512.
- Default oracle: MLP with `small_dim=96`.
- Do not attempt GPT-2 Medium on this device.

---

### Task 1: Verify Environment And Model Shape

**Files:**
- Use: `requirements.txt`
- Use: `config.py`
- Use: `smoke_test.py`

**Interfaces:**
- Consumes: `get_config(preset: str) -> ExperimentConfig`
- Produces: Verified layer split and oracle parameter count.

- [x] **Step 1: Verify available Python environment**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- [x] **Step 2: Set CPU variables**

```powershell
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"
$env:TOKENIZERS_PARALLELISM="false"
```

- [x] **Step 3: Run shape smoke test**

```powershell
python smoke_test.py
```

Expected output includes:

```text
model=distilgpt2
layers=6
hidden_size=768
early_layers=0..1
skipped_layers=2..3
late_layers=4..5
```

### Task 2: Collect Smoke Hidden-State Data

**Files:**
- Use: `collect_data.py`
- Output: `oracle_data_distilgpt2/pair_*.npz`
- Output: `oracle_data_distilgpt2/metadata.json`

**Interfaces:**
- Consumes: Hugging Face `wikitext-2-raw-v1`.
- Produces: compressed float16 arrays named `early` and `target`.

- [x] **Step 1: Collect 64 samples**

```powershell
python collect_data.py --samples 64 --clear
```

- [x] **Step 2: Inspect output count**

```powershell
(Get-ChildItem .\oracle_data_distilgpt2\pair_*.npz).Count
```

Expected output:

```text
64
```

### Task 3: Train Smoke Oracle

**Files:**
- Use: `oracle_model.py`
- Use: `train_oracle.py`
- Input: `oracle_data_distilgpt2/pair_*.npz`
- Output: `oracle_distilgpt2.pth`
- Output: `oracle_distilgpt2.json`

**Interfaces:**
- Consumes: `build_oracle(oracle_type: str, in_dim: int, small_dim: int) -> nn.Module`
- Produces: checkpoint payload with `model_state_dict`, `config`, and `history`.

- [x] **Step 1: Train one epoch**

```powershell
python train_oracle.py --epochs 1
```

- [x] **Step 2: Verify checkpoint exists**

```powershell
Test-Path .\oracle_distilgpt2.pth
```

Expected output:

```text
True
```

### Task 4: Run Smoke PathOracle Inference

**Files:**
- Use: `inference_pipeline.py`
- Input: `oracle_distilgpt2.pth`

**Interfaces:**
- Consumes: `PathOracleGPT2.forward_logits(input_ids)`.
- Produces: generated text and finite sample perplexity.

- [x] **Step 1: Generate 10 tokens**

```powershell
python inference_pipeline.py --max-new-tokens 10
```

Expected output includes:

```text
prompt=The capital of France is
output=
sample_ppl=
```

### Task 5: Run Full Distilgpt2 MVP

**Files:**
- Use: `collect_data.py`
- Use: `train_oracle.py`
- Use: `evaluate.py`
- Output: `oracle_data_distilgpt2`
- Output: `oracle_distilgpt2.pth`
- Output: console evaluation report.

**Interfaces:**
- Consumes: same APIs as Tasks 2-4.
- Produces: generation comparison and PPL comparison.

- [x] **Step 1: Collect 512 samples**

```powershell
python collect_data.py --samples 512 --clear
```

- [x] **Step 2: Train three epochs**

```powershell
python train_oracle.py --epochs 3
```

- [x] **Step 3: Evaluate**

```powershell
python evaluate.py --max-texts 20 --max-prompts 5
```

Expected output includes:

```text
=== generation comparison ===
=== perplexity comparison ===
original_ppl=
pathoracle_ppl=
relative_increase=
```

### Task 6: Optional GPT-2 Small Upgrade

**Files:**
- Use: all Python scripts with `--preset gpt2`
- Output: `oracle_data_gpt2`
- Output: `oracle_gpt2.pth`

**Interfaces:**
- Consumes: same command-line interface with `--preset gpt2`.
- Produces: a 12-layer GPT-2 small run with layers 2-9 skipped.

- [ ] **Step 1: Verify GPT-2 small shape**

```powershell
python smoke_test.py --preset gpt2
```

Expected output includes:

```text
model=gpt2
layers=12
hidden_size=768
skipped_layers=2..9
late_layers=10..11
```

- [ ] **Step 2: Run reduced GPT-2 smoke**

```powershell
python collect_data.py --preset gpt2 --samples 256 --clear
python train_oracle.py --preset gpt2 --epochs 1
python inference_pipeline.py --preset gpt2 --max-new-tokens 10
```

- [ ] **Step 3: Run full GPT-2 only if stable**

```powershell
python collect_data.py --preset gpt2 --samples 2000 --clear
python train_oracle.py --preset gpt2 --epochs 3
python evaluate.py --preset gpt2 --max-texts 20 --max-prompts 5
```
