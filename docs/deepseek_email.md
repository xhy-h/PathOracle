# DeepSeek Email Draft

## Recipients

```text
To: service@deepseek.com
Cc: info@deepseek.ai
```

## Subject

```text
[Technical Proposal] PathOracle: Prefill Acceleration Complementary to DSpark
```

## Body

```text
Dear DeepSeek Team,

I am reaching out to share PathOracle, an open-source hidden-state prediction method for accelerating the prefill phase of Transformer and MoE model inference.

PathOracle is complementary to DSpark. DSpark accelerates decode through speculative decoding, while PathOracle accelerates prefill by predicting later hidden states and skipping intermediate Transformer blocks.

Key results from the CPU-tested MVP:

- distilgpt2: 1.66x PPL ratio with a 2-block Transformer oracle.
- GPT-2 Small: 1.92x PPL ratio while skipping 8 of 12 Transformer blocks.
- Full training, inference, evaluation, checkpoint resume, and experiment parameterization are open-sourced under the MIT License.

We also prepared a DeepSeek-V4 migration design based on multi-anchor layer skipping. The target is to reduce prefill block compute by roughly 48-52%, with PathOracle handling prefill and DSpark continuing to handle decode.

Materials:

- Repository: https://github.com/xhy-h/PathOracle
- Whitepaper: https://github.com/xhy-h/PathOracle/blob/main/docs/whitepaper.md
- DeepSeek migration plan: https://github.com/xhy-h/PathOracle/blob/main/docs/deepseek_v4_migration.md
- Full experiment results: https://github.com/xhy-h/PathOracle/blob/main/RESULTS.md

I would be grateful for any feedback from the DeepSeek inference optimization team. If this direction is interesting, I would be happy to provide a more detailed DeepSeek-V4 prototype plan or collaborate on a targeted validation experiment.

Best regards,
xhy-h
https://github.com/xhy-h
```
