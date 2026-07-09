# DeepSeek-V4 Migration Plan

This document describes a proposed PathOracle migration path for DeepSeek-V4-style MoE models. It is a design document, not a completed DeepSeek-V4 experiment. All layer counts, hidden sizes, and routing details must be checked against the exact target checkpoint before implementation.

## Goal

Combine two complementary acceleration strategies:

- **PathOracle for prefill**: predict hidden states and skip intermediate layers.
- **DSpark for decode**: speculative decoding over generated tokens.

Together, the two methods can target the full inference pipeline.

## Starting Point from CPU Experiments

| Model | Best Oracle | Skip Pattern | PPL Ratio | Final Cosine |
|---|---|---|---:|---:|
| `distilgpt2` | dim 192, 2 blocks | skip 2 of 6 blocks | 1.66x | 0.9052 |
| GPT-2 Small | dim 256, 2 blocks | skip 8 of 12 blocks | 1.92x | 0.8871 |

The GPT-2 Small result is the most relevant stress test because it skips a large middle span and still stays below a 2.0x PPL ratio.

## Proposed DeepSeek-V4 Strategy

For a target model with 61 layers and hidden size 7168, use multi-anchor skipping instead of one long jump.

### Candidate Segment Layout

```text
Segment 1: run early layers -> oracle -> resume later layers
Segment 2: run early layers -> oracle -> resume later layers
Segment 3: run early layers -> oracle -> resume later layers
...
```

The exact segmentation should follow the model's block grouping, MoE placement, attention pattern, and any IndexShare-style sharing rules. A safe starting target is to skip roughly 45-52% of prefill blocks, then increase the skip ratio only after quality is stable.

## Oracle Architecture

Start with the smallest architecture that worked on GPT-2 Small, scaled to the target hidden size:

```text
Input hidden state: [batch, seq, 7168]
Projection: Linear(7168 -> 512)
Backbone: 2 Transformer encoder blocks
Output projection: Linear(512 -> 7168)
Loss: MSE + 0.5 * cosine loss
```

If quality is insufficient, scale in this order:

1. Increase small dimension from 512 to 768.
2. Add a third Transformer block.
3. Add intermediate-layer auxiliary targets.
4. Add logits or KL distillation.

## Training Data

The small-model experiments show that data volume is critical. A GPU-scale DeepSeek run should start with:

- at least 50,000 prompt samples for each anchor design;
- sequence lengths 128 and 256;
- a mixture of web, code, math, and instruction prompts;
- held-out evaluation prompts covering both short and long prefill.

## Evaluation

Use both distributional and task-level metrics:

- hidden-state MSE;
- cosine similarity;
- next-token negative log-likelihood;
- perplexity ratio against the unmodified model;
- generation repetition rate;
- downstream task pass rates if available;
- actual wall-clock prefill latency.

## Expected Benefit

If roughly half of prefill blocks are skipped, the expected block-compute reduction is around 48-52% before accounting for oracle overhead. The measured speedup will depend on:

- MoE routing overhead;
- attention kernel implementation;
- quantization format;
- CPU/GPU placement of the oracle;
- batching and prompt length.

## Integration with DSpark

PathOracle can be placed before decode starts:

```text
Prompt tokens
  -> PathOracle-accelerated prefill
  -> KV cache / final hidden states
  -> DSpark speculative decode
  -> final tokens
```

This keeps the integration boundary clean: PathOracle modifies prefill block execution, while DSpark modifies decode token generation.

## Risks

- Long-span hidden-state prediction can cause local phrase repetition.
- MoE routing may amplify hidden-state prediction error.
- A single oracle may not generalize across all layer groups.
- PPL improvement may not fully predict user-visible generation quality.

## Recommended Next Experiment

On a GPU machine:

1. Validate one short DeepSeek-V4 segment with a conservative skip ratio.
2. Train a dim-512, 2-block oracle.
3. Compare original vs PathOracle prefill on a held-out prompt set.
4. Measure wall-clock prefill latency, not only PPL.
5. Add DSpark only after prefill quality is stable.
