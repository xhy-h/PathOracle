# Experiment Parameterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CLI-configurable experiment parameters and non-overwriting artifact naming for PathOracle runs.

**Architecture:** Extend the existing `ExperimentConfig` dataclass and script CLIs rather than introducing a new framework. Scripts share a helper that applies overrides and derives artifact paths from `--run-tag`.

**Tech Stack:** Python, PyTorch, Hugging Face Datasets/Transformers, standard-library `unittest`.

## Global Constraints

- Preserve the existing default `distilgpt2` 512-sample baseline paths.
- Do not introduce new dependencies.
- Use `unittest` tests before production changes.
- Keep CPU-only behavior.
- This folder is not a Git repository, so commit steps are skipped.

---

### Task 1: Config And Oracle Parameters

**Files:**
- Modify: `config.py`
- Modify: `oracle_model.py`
- Test: `tests/test_config.py`
- Test: `tests/test_oracle_model.py`

**Interfaces:**
- Produces: `build_experiment_config(...) -> ExperimentConfig`
- Produces: `build_oracle(oracle_type, in_dim, small_dim, num_blocks=1) -> nn.Module`

- [ ] Write failing tests for run-tag path derivation and transformer block count.
- [ ] Run `python -m unittest tests.test_config tests.test_oracle_model` and verify failure.
- [ ] Implement config override helper and `num_blocks` support.
- [ ] Re-run tests and verify pass.

### Task 2: Script CLI Integration

**Files:**
- Modify: `collect_data.py`
- Modify: `train_oracle.py`
- Modify: `inference_pipeline.py`
- Modify: `evaluate.py`
- Modify: `smoke_test.py`

**Interfaces:**
- Consumes: `build_experiment_config`
- Produces: scripts accepting `--run-tag`, `--small-dim`, `--num-blocks`, `--cos-weight`, `--dataset`, and `--streaming` where relevant.

- [ ] Add CLI flags and route them through the shared config helper.
- [ ] Store new metadata in data metadata and checkpoint payload.
- [ ] Ensure inference reconstructs transformer oracle with checkpoint `num_blocks`.
- [ ] Run `python -m compileall .`.

### Task 3: Tiny Experiment Verification

**Files:**
- Output: `oracle_data_distilgpt2_param_smoke`
- Output: `oracle_distilgpt2_param_smoke.pth`

**Interfaces:**
- Verifies: the new parameter layer supports isolated artifacts.

- [ ] Run a tiny collection with `--samples 8 --max-length 32 --run-tag distilgpt2_param_smoke`.
- [ ] Train one epoch with `--oracle-type transformer --small-dim 128 --cos-weight 0.5 --batch-size 2 --epochs 1 --run-tag distilgpt2_param_smoke`.
- [ ] Run inference with `--checkpoint oracle_distilgpt2_param_smoke.pth`.
- [ ] Verify the original baseline artifacts still exist.
