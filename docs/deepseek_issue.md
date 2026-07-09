# GitHub Issue Draft for DeepSeek

## Title

```text
[Proposal] PathOracle: Hidden-State Prediction Layer Skipping for Prefill Acceleration, Complementary to DSpark
```

## Body

````markdown
Dear DeepSeek team,

We have been working on **PathOracle**, a lightweight prefill acceleration method that predicts hidden states at later anchor layers and skips intermediate Transformer blocks.

The key point is complementarity:

- **DSpark** accelerates the decode phase through speculative decoding.
- **PathOracle** accelerates the prefill phase through hidden-state prediction and layer skipping.

Together, the two methods could cover more of the end-to-end inference pipeline.

## What PathOracle Does

PathOracle runs the early Transformer blocks, predicts the hidden state that would have been produced after skipped middle blocks, and then resumes the original late blocks:

```text
embeddings -> early blocks -> PathOracle -> late blocks -> logits
```

The oracle is a compact external Transformer trained with MSE plus cosine similarity loss.

## Experimental Evidence

All experiments are CPU-tested and reproducible in the open-source framework.

| Model | Skip Pattern | Best Oracle | PPL Ratio | Final Cosine |
|---|---|---|---:|---:|
| distilgpt2 | run 0-1, skip 2-3, run 4-5 | dim 192, 2 blocks | 1.66x | 0.9052 |
| GPT-2 Small | run 0-1, skip 2-9, run 10-11 | dim 256, 2 blocks | 1.92x | 0.8871 |

For GPT-2 Small, PathOracle skips 8 of 12 Transformer blocks and keeps the PPL ratio below 2.0x.

## DeepSeek-V4 Migration Plan

We also prepared a concrete migration design for DeepSeek-V4-style MoE models:

- multi-anchor layer grouping instead of one long jump;
- compact per-segment or shared hidden-state oracles;
- target prefill block-compute reduction around 48-52%;
- compatibility with DSpark because PathOracle operates before decode starts.

The exact layer grouping should be validated against the final DeepSeek-V4 checkpoint configuration.

## Open-Source Materials

- Repository: https://github.com/xhy-h/PathOracle
- Whitepaper: https://github.com/xhy-h/PathOracle/blob/main/docs/whitepaper.md
- DeepSeek migration plan: https://github.com/xhy-h/PathOracle/blob/main/docs/deepseek_v4_migration.md
- Full experiment results: https://github.com/xhy-h/PathOracle/blob/main/RESULTS.md

## Request

We would appreciate feedback from the DeepSeek inference team on whether PathOracle could be explored as a prefill acceleration component alongside DSpark.

We would also be happy to provide additional implementation details, experiment logs, or a more targeted DeepSeek-V4 prototype plan.

Thank you for the excellent work on DeepSeek and DSpark.
````
