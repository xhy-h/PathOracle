# PathOracle

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20tested-ee4c2c.svg)](https://pytorch.org/)

**Skip layers, keep accuracy. PathOracle predicts hidden states to accelerate Transformer prefill with a small external oracle.**

PathOracle is a CPU-tested research framework for prefill-time layer skipping. It runs the early Transformer layers, predicts the hidden state that would have been produced after skipped middle layers, and then resumes the final layers. This is designed to be complementary to speculative decoding systems such as DSpark: PathOracle targets **prefill**, while speculative decoding targets **decode**.

- Verified on `distilgpt2`: **1.66x PPL ratio** with a 2-block Transformer oracle.
- Verified on GPT-2 Small: **1.92x PPL ratio** while skipping 8 of 12 blocks.
- Fully parameterized experiment scripts for data collection, oracle training, inference, and evaluation.
- Designed as a stepping stone toward MoE prefill acceleration, including a DeepSeek-V4 migration plan.

[Quick Start](#quick-start) · [Results](RESULTS.md) · [Whitepaper](docs/whitepaper.md) · [DeepSeek Migration Plan](docs/deepseek_v4_migration.md)

## Core Idea

```text
tokens
  -> embeddings
  -> early Transformer blocks
  -> PathOracle(hidden-state predictor)
  -> late Transformer blocks
  -> logits
```

Instead of running every middle block during prefill, PathOracle learns to predict the target hidden state at a later anchor layer. The current CPU MVP uses:

- hidden-state pair collection from the original model;
- MSE loss plus cosine similarity loss;
- MLP or compact Transformer oracle models;
- patched inference that runs early blocks, oracle, and late blocks.

## Best Results

| Model | Data | Skip Pattern | Oracle | Params | PPL Ratio | Final Cosine |
|---|---:|---|---|---:|---:|---:|
| `distilgpt2` | 5,000 x len128 | run 0-1, skip 2-3, run 4-5 | Transformer, dim 192, 2 blocks | 889K | **1.66x** | **0.9052** |
| `gpt2` | 5,000 x len64 | run 0-1, skip 2-9, run 10-11 | Transformer, dim 256, 2 blocks | 1.45M | **1.92x** | **0.8871** |

The GPT-2 Small result is the current best checkpoint:

```text
run_tag = gpt2_wiki5000_len64_tr_sdim256_nb2_cos0.5_e5
original_ppl = 65.7304
pathoracle_ppl = 126.2876
ppl_ratio = 1.92x
```

See [RESULTS.md](RESULTS.md) for the full experiment history.

## Quick Start

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"
$env:TOKENIZERS_PARALLELISM="false"
```

Run a small smoke test:

```powershell
python smoke_test.py --preset distilgpt2
python collect_data.py --preset distilgpt2 --samples 64 --max-length 64 --clear
python train_oracle.py --preset distilgpt2 --epochs 1
python inference_pipeline.py --preset distilgpt2 --max-new-tokens 10
```

Reproduce the stronger GPT-2 Small configuration:

```powershell
python collect_data.py --preset gpt2 --samples 5000 --max-length 64 --run-tag gpt2_wiki5000_len64_tr_sdim256_nb2_cos0.5_e5 --clear

python train_oracle.py --preset gpt2 --oracle-type transformer --small-dim 256 --num-blocks 2 --cos-weight 0.5 --batch-size 1 --epochs 5 --run-tag gpt2_wiki5000_len64_tr_sdim256_nb2_cos0.5_e5

python evaluate.py --preset gpt2 --run-tag gpt2_wiki5000_len64_tr_sdim256_nb2_cos0.5_e5 --max-texts 20 --max-prompts 5
```

## Repository Layout

- `config.py`: shared presets and experiment configuration.
- `collect_data.py`: collects early and target hidden-state pairs.
- `oracle_model.py`: MLP and Transformer oracle definitions.
- `train_oracle.py`: oracle training, checkpointing, resume, and fine-tuning.
- `inference_pipeline.py`: PathOracle inference pipeline.
- `evaluate.py`: generation and perplexity comparison.
- `tests/`: unit tests for configuration, oracle construction, and resume behavior.
- `docs/whitepaper.md`: technical whitepaper draft.
- `docs/deepseek_v4_migration.md`: DeepSeek-V4 migration design.
- `docs/deepseek_issue.md`: GitHub proposal draft.
- `docs/deepseek_email.md`: email proposal draft.

## Design Notes

PathOracle is not intended to replace speculative decoding. It addresses a different phase:

| Method | Phase | Mechanism |
|---|---|---|
| DSpark / speculative decoding | Decode | Predict candidate future tokens |
| PathOracle | Prefill | Predict future hidden states and skip blocks |

The two methods can be stacked: PathOracle reduces prefill block execution, then speculative decoding accelerates token-by-token decode.

## Current Limitations

- The strongest GPT-2 result still shows repeated phrase templates in greedy generation.
- The project has not yet been validated on a production MoE checkpoint.
- FLOPs reductions are estimated from skipped block counts and should be remeasured in the target inference runtime.
- DeepSeek-V4 migration details must be checked against the final target model config before implementation.

## License

MIT License. See [LICENSE](LICENSE).
