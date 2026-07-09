# PathOracle: Hidden-State Prediction for Efficient Transformer Prefill

## Abstract

Large Transformer inference has two distinct cost centers: prefill and decode. Speculative decoding methods such as DSpark target decode by proposing future tokens. PathOracle targets the complementary prefill stage by learning a lightweight external network that predicts later hidden states from early hidden states, allowing intermediate Transformer blocks to be skipped.

On `distilgpt2`, PathOracle reaches a 1.66x perplexity ratio against the original model. On GPT-2 Small, it reaches a 1.92x perplexity ratio while skipping blocks 2-9 and running only blocks 0-1 and 10-11. The implementation is fully CPU-tested and released as an MIT-licensed experiment framework.

## 1. Motivation

Prefill computes the prompt representation before autoregressive decoding starts. For long prompts and MoE models, this stage can dominate latency and memory bandwidth because every prompt token must pass through many Transformer layers.

PathOracle asks a direct question: if early layers already encode enough semantic information, can a compact oracle predict the hidden state that would have been produced after several middle layers? If so, the model can skip expensive blocks during prefill and resume computation near the output layers.

This is orthogonal to speculative decoding:

| Method | Target Phase | Acceleration Target |
|---|---|---|
| Speculative decoding / DSpark | Decode | Token proposal and verification |
| PathOracle | Prefill | Hidden-state prediction and layer skipping |

## 2. Method

PathOracle uses paired hidden states collected from a frozen base model.

1. Run the prompt through the early Transformer blocks.
2. Save the early hidden state.
3. Continue the original model through the skipped middle blocks.
4. Save the target hidden state at the late anchor layer.
5. Train an oracle to map early hidden states to target hidden states.

During inference, the original model runs:

```text
embedding -> early blocks -> oracle -> late blocks -> logits
```

The current oracle variants are:

- MLP oracle: a small baseline.
- Transformer oracle: projection to a smaller dimension, several self-attention blocks, and projection back to the model hidden size.

The training objective is:

```text
loss = MSE(predicted_hidden, target_hidden)
     + cos_weight * (1 - cosine_similarity(predicted_hidden, target_hidden))
```

## 3. Experiments

All experiments were run on a CPU-only Windows laptop. The framework includes scripts for data collection, training, inference, evaluation, and checkpoint resume.

### 3.1 distilgpt2

The best `distilgpt2` run uses 5,000 WikiText-2 samples at sequence length 128.

| Configuration | PPL | PPL Ratio | Final Cosine |
|---|---:|---:|---:|
| Original model | 110.9334 | 1.00x | - |
| PathOracle, dim 192, 2 blocks | 184.0707 | **1.66x** | **0.9052** |

### 3.2 GPT-2 Small

The GPT-2 Small experiment is harder because it skips 8 of 12 blocks.

| Configuration | Samples | PPL | PPL Ratio | Final Cosine |
|---|---:|---:|---:|---:|
| Original model | - | 65.7304 | 1.00x | - |
| PathOracle, dim 192, 2 blocks | 5,000 | 142.5196 | 2.17x | 0.8748 |
| PathOracle, dim 192, 2 blocks, low-LR fine-tune | 5,000 | 138.2357 | 2.10x | 0.8753 |
| PathOracle, dim 192, 2 blocks, WikiText+c4 | 7,000 | 137.8258 | 2.10x | 0.8671 |
| PathOracle, dim 256, 2 blocks | 5,000 | **126.2876** | **1.92x** | **0.8871** |

The main empirical finding is that larger oracle capacity is required as the skipped layer span becomes deeper. Data scaling is valuable, but once the dim-192 oracle saturates, dim-256 produces the first GPT-2 result below a 2.0x PPL ratio.

## 4. DeepSeek-V4 Migration Design

PathOracle has not yet been tested on DeepSeek-V4. The migration plan is designed around the target model configuration and must be validated against the actual checkpoint config before implementation.

Assuming a 61-layer model with hidden size 7168, a practical migration design is:

- keep early anchor blocks in each segment;
- train compact per-segment or shared oracles;
- skip roughly half of the dense/MoE blocks during prefill;
- resume the original late blocks for final representation refinement.

A candidate oracle shape is:

```text
Linear(7168 -> 512)
2 x Transformer encoder block
Linear(512 -> 7168)
```

The expected benefit is prefill block-compute reduction on the order of 48-52%, excluding oracle overhead. The actual speedup must be measured in the target runtime because MoE routing, attention kernels, quantization, and memory bandwidth can dominate wall-clock performance.

## 5. Limitations and Future Work

PathOracle currently improves perplexity but still shows repeated phrase templates in greedy generation. Promising next steps include:

- adding KL or logits distillation from the original model;
- adding intermediate-layer auxiliary targets;
- using repetition-aware decoding during qualitative evaluation;
- scaling oracle capacity on GPU;
- testing multi-anchor skipping on an actual MoE checkpoint.

## 6. Conclusion

The CPU MVP validates the PathOracle prefill-skipping paradigm across `distilgpt2` and GPT-2 Small. The GPT-2 Small result reaches a 1.92x PPL ratio while skipping 8 of 12 blocks. This is strong enough to justify GPU-scale validation on larger MoE models and to explore integration with decode-time systems such as DSpark.
