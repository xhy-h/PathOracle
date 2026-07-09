# Experiment Parameterization Design

## Goal

Add a reusable experiment parameter layer so larger PathOracle runs can be varied without overwriting the current `distilgpt2` baseline artifacts.

## Scope

The first supported follow-up experiment is:

- Model: `distilgpt2`
- Dataset: `wikitext-2-raw-v1`
- Samples: 1024
- Sequence length: 128
- Oracle: transformer
- Oracle small dim: 128
- Transformer blocks: 1
- Cosine loss weight: 0.5
- Batch size: 2
- Epochs: 3
- Learning rate: 1e-3
- Run tag: `distilgpt2_wiki1024_len128_tr_sdim128_cos0.5_e3`

## Design

Keep the existing script entry points. Extend `config.py` with fields for dataset, run tag, cosine weight, and transformer block count, plus a helper that applies CLI overrides and derives artifact paths.

Default behavior remains backward compatible: scripts without a run tag still use `oracle_data_distilgpt2` and `oracle_distilgpt2.pth`.

When `--run-tag` is supplied, scripts derive:

- data directory: `oracle_data_<run_tag>`
- checkpoint path: `oracle_<run_tag>.pth`

Training stores `small_dim`, `num_blocks`, `cos_weight`, and data/checkpoint metadata in the checkpoint payload so inference and evaluation can reconstruct the correct oracle.

## Validation

Use `unittest` for configuration and oracle construction checks, then run a tiny end-to-end command sequence before the 1024-sample experiment.
