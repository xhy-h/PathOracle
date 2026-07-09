# PathOracle CPU MVP Results

Run date: 2026-07-09

## Environment

- Python: 3.11.9
- PyTorch: 2.10.0+cpu
- Transformers: 5.1.0
- Datasets: 4.5.0
- NumPy: 2.4.2
- Runtime: CPU-only

## Distilgpt2 Smoke Run

- Shape check: passed
- Layers: 6
- Hidden size: 768
- Early layers: 0..1
- Skipped layers: 2..3
- Late layers: 4..5
- Oracle type: MLP
- Oracle parameters: 159,168
- Smoke samples: 64
- Smoke train: 1 epoch
- Smoke train history: MSE 80.7036, cosine 0.2155
- Smoke inference: finite sample PPL 121313.7891

## Distilgpt2 512-Sample Run

- Samples collected: 512
- Train epochs: 3
- Epoch 1: MSE 65.0559, cosine 0.6194
- Epoch 2: MSE 8.7452, cosine 0.6745
- Epoch 3: MSE 4.5086, cosine 0.7156
- Inference sample PPL: 5955.1963

## Evaluation

- Eval texts: 20
- Eval prompts: 5
- Original PPL: 110.9334
- PathOracle PPL: 2214.6071
- Relative PPL increase: 1896.34%

## Distilgpt2 1024-Sample Transformer Run

- Run tag: `distilgpt2_wiki1024_len128_tr_sdim128_cos0.5_e3`
- Samples collected: 1024
- Max length: 128
- Oracle type: Transformer
- Oracle parameters: 329,344
- Small dim: 128
- Transformer blocks: 1
- Cosine loss weight: 0.5
- Batch size: 2
- Train epochs: 3
- Epoch 1: MSE 69.8550, cosine 0.7816
- Epoch 2: MSE 52.2843, cosine 0.8609
- Epoch 3: MSE 41.4324, cosine 0.8710
- Inference sample PPL: 921.4484

## 1024-Sample Transformer Evaluation

- Eval texts: 20
- Eval prompts: 5
- Original PPL: 110.9334
- PathOracle PPL: 377.9450
- Relative PPL increase: 240.70%
- PPL ratio: 3.41x
- Go/No-Go PPL criterion: passed (`ratio < 5`)
- Generation quality: no `the the the` collapse, but repeated short phrase templates remain (`first year`, `first book`, `first day`).

## Distilgpt2 5000-Sample Transformer Run

- Run tag: `distilgpt2_wiki5000_len128_tr_sdim128_cos0.5_e3`
- Samples collected: 5000
- Max length: 128
- Data directory size: about 1.45 GB
- Oracle type: Transformer
- Oracle parameters: 329,344
- Small dim: 128
- Transformer blocks: 1
- Cosine loss weight: 0.5
- Batch size: 2
- Train epochs: 3
- Epoch 1: MSE 33.1437, cosine 0.8493
- Epoch 2: MSE 2.0549, cosine 0.8765
- Epoch 3: MSE 1.9959, cosine 0.8792
- Inference sample PPL: 978.2251

## 5000-Sample Transformer Evaluation

- Eval texts: 20
- Eval prompts: 5
- Original PPL: 110.9334
- PathOracle PPL: 260.2357
- Relative PPL increase: 134.59%
- PPL ratio: 2.35x
- Go/No-Go PPL criterion: passed (`ratio < 2.5`)
- Go/No-Go cosine criterion: not passed (`0.8792 < 0.90`)
- Generation quality: stronger semantic outputs than prior runs, but repeated phrase templates remain (`book that is a book`, `first time the first time`).

## Distilgpt2 5000-Sample Transformer Resume-to-5 Run

- Run tag: `distilgpt2_wiki5000_len128_tr_sdim128_cos0.5_e5`
- Resumed from: `distilgpt2_wiki5000_len128_tr_sdim128_cos0.5_e3`
- Data directory: `oracle_data_distilgpt2_wiki5000_len128_tr_sdim128_cos0.5_e3`
- Oracle type: Transformer
- Oracle parameters: 329,344
- Small dim: 128
- Transformer blocks: 1
- Cosine loss weight: 0.5
- Batch size: 2
- Total train epochs: 5
- Epoch 4: MSE 2.0204, cosine 0.8777
- Epoch 5: MSE 2.0022, cosine 0.8788
- Inference sample PPL: 933.2794

## 5000-Sample Transformer Resume-to-5 Evaluation

- Eval texts: 20
- Eval prompts: 5
- Original PPL: 110.9334
- PathOracle PPL: 252.0380
- Relative PPL increase: 127.20%
- PPL ratio: 2.27x
- Go/No-Go PPL criterion: passed (`ratio < 2.5`)
- Migration PPL criterion: not passed (`ratio > 2.0`)
- Go/No-Go cosine criterion: not passed (`0.8788 < 0.90`)
- Generation quality: slight PPL gain over epoch 3, but repeated phrase templates remain (`book that is a book`, `first time the first time`).

## Distilgpt2 5000-Sample Low-LR Fine-Tune Run

- Run tag: `distilgpt2_wiki5000_len128_tr_sdim128_cos0.5_e8_ft_lr5e-4`
- Resumed from: `distilgpt2_wiki5000_len128_tr_sdim128_cos0.5_e5`
- Reset optimizer: yes
- Learning rate: 5e-4
- Data directory: `oracle_data_distilgpt2_wiki5000_len128_tr_sdim128_cos0.5_e3`
- Oracle type: Transformer
- Oracle parameters: 329,344
- Small dim: 128
- Transformer blocks: 1
- Cosine loss weight: 0.5
- Batch size: 2
- Total train epochs: 8
- Epoch 6: MSE 1.9764, cosine 0.8803
- Epoch 7: MSE 1.9705, cosine 0.8807
- Epoch 8: MSE 1.9650, cosine 0.8810
- Inference sample PPL: 866.1578

## Low-LR Fine-Tune Evaluation

- Eval texts: 20
- Eval prompts: 5
- Original PPL: 110.9334
- PathOracle PPL: 237.6965
- Relative PPL increase: 114.27%
- PPL ratio: 2.14x
- Migration PPL criterion: not passed (`ratio > 2.0`)
- Go/No-Go cosine criterion: not passed (`0.8810 < 0.90`)
- Generation quality: sample prompt improved, but repeated phrase templates remain (`first time the first time`).

## Distilgpt2 5000-Sample Transformer 2-Block Run

- Run tag: `distilgpt2_wiki5000_len128_tr_sdim128_nb2_cos0.5_e5`
- Data directory: `oracle_data_distilgpt2_wiki5000_len128_tr_sdim128_cos0.5_e3`
- Oracle type: Transformer
- Oracle parameters: 461,824
- Small dim: 128
- Transformer blocks: 2
- Cosine loss weight: 0.5
- Batch size: 2
- Train epochs: 5
- Epoch 1: MSE 32.9211, cosine 0.8497
- Epoch 2: MSE 2.0354, cosine 0.8777
- Epoch 3: MSE 1.9814, cosine 0.8802
- Epoch 4: MSE 1.9556, cosine 0.8816
- Epoch 5: MSE 1.9393, cosine 0.8826
- Inference sample PPL: 832.7229

## 2-Block Transformer Evaluation

- Eval texts: 20
- Eval prompts: 5
- Original PPL: 110.9334
- PathOracle PPL: 234.6546
- Relative PPL increase: 111.53%
- PPL ratio: 2.12x
- Migration PPL criterion: not passed (`ratio > 2.0`)
- Go/No-Go cosine criterion: not passed (`0.8826 < 0.90`)
- Generation quality: modest improvement over 1-block fine-tune on some prompts, but `book` and `first time` loops remain.

## Distilgpt2 5000-Sample Transformer 2-Block Small-Dim-192 Run

- Run tag: `distilgpt2_wiki5000_len128_tr_sdim192_nb2_cos0.5_e5`
- Data directory: `oracle_data_distilgpt2_wiki5000_len128_tr_sdim128_cos0.5_e3`
- Oracle type: Transformer
- Oracle parameters: 889,344
- Small dim: 192
- Transformer blocks: 2
- Cosine loss weight: 0.5
- Batch size: 2
- Train epochs: 5
- Epoch 1: MSE 25.9746, cosine 0.8718
- Epoch 2: MSE 1.7028, cosine 0.8989
- Epoch 3: MSE 1.6397, cosine 0.9020
- Epoch 4: MSE 1.6054, cosine 0.9039
- Epoch 5: MSE 1.5832, cosine 0.9052
- Inference sample PPL: 667.0175

## 2-Block Small-Dim-192 Evaluation

- Eval texts: 20
- Eval prompts: 5
- Original PPL: 110.9334
- PathOracle PPL: 184.0707
- Relative PPL increase: 65.93%
- PPL ratio: 1.66x
- Migration PPL criterion: passed (`ratio <= 2.0`)
- Go/No-Go cosine criterion: passed (`0.9052 >= 0.90`)
- Generation quality: best run so far and no token-level collapse, but repeated phrase templates remain on some prompts (`weather is a very different`, `book about the book`).
- Status: best distilgpt2 checkpoint so far.

## GPT-2 Small 2000-Sample Migration Run

- Run tag: `gpt2_wiki2000_len64_tr_sdim192_nb2_cos0.5_e5`
- Model: `gpt2`
- Samples collected: 2000
- Max length: 64
- Data directory size: about 337 MB
- Total layers: 12
- Early layers: 0..1
- Skipped layers: 2..9
- Late layers: 10..11
- Oracle type: Transformer
- Oracle parameters: 889,344
- Small dim: 192
- Transformer blocks: 2
- Cosine loss weight: 0.5
- Batch size: 2
- Train epochs: 5
- Epoch 1: MSE 192.9691, cosine 0.7841
- Epoch 2: MSE 83.5233, cosine 0.8499
- Epoch 3: MSE 19.2601, cosine 0.8607
- Epoch 4: MSE 7.9127, cosine 0.8670
- Epoch 5: MSE 7.2648, cosine 0.8705
- Inference sample PPL: 414.9333

## GPT-2 Small Migration Evaluation

- Eval texts: 20
- Eval prompts: 5
- Original PPL: 65.7304
- PathOracle PPL: 195.8084
- Relative PPL increase: 197.90%
- PPL ratio: 2.98x
- Migration proof criterion: passed (`ratio < 3.0`)
- Acceptable baseline criterion: passed (`ratio < 4.0`)
- Generation quality: pipeline transfers to GPT-2 Small without NaNs or token-level collapse, but phrase loops remain severe on several prompts (`book that is a book`, `United States, the United States`, `first of the first`).

## GPT-2 Small 5000-Sample Data Expansion Run

- Run tag: `gpt2_wiki5000_len64_tr_sdim192_nb2_cos0.5_e5`
- Model: `gpt2`
- Samples collected: 5000
- Max length: 64
- Data directory size: about 841 MB
- Total layers: 12
- Early layers: 0..1
- Skipped layers: 2..9
- Late layers: 10..11
- Oracle type: Transformer
- Oracle parameters: 889,344
- Small dim: 192
- Transformer blocks: 2
- Cosine loss weight: 0.5
- Batch size: 2
- Train epochs: 5
- Epoch 1: MSE 113.5796, cosine 0.8230
- Epoch 2: MSE 7.9075, cosine 0.8618
- Epoch 3: MSE 7.3880, cosine 0.8684
- Epoch 4: MSE 7.1709, cosine 0.8723
- Epoch 5: MSE 7.0313, cosine 0.8748
- Inference sample PPL: 388.6868

## GPT-2 Small 5000-Sample Evaluation

- Eval texts: 20
- Eval prompts: 5
- Original PPL: 65.7304
- PathOracle PPL: 142.5196
- Relative PPL increase: 116.82%
- PPL ratio: 2.17x
- Go/No-Go PPL criterion: passed (`ratio <= 2.5`)
- Cosine criterion: not passed (`0.8748 < 0.88`)
- Generation quality: improved materially over the 2000-sample GPT-2 run, but phrase templates remain on several prompts (`world of the world`, `year of the year`, `game of the game`).

## GPT-2 Small 5000-Sample Low-LR Fine-Tune Run

- Run tag: `gpt2_wiki5000_len64_tr_sdim192_nb2_cos0.5_e8_ft_lr5e-4`
- Resumed from: `gpt2_wiki5000_len64_tr_sdim192_nb2_cos0.5_e5`
- Reset optimizer: yes
- Learning rate: 5e-4
- Data directory: `oracle_data_gpt2_wiki5000_len64_tr_sdim192_nb2_cos0.5_e5`
- Oracle type: Transformer
- Oracle parameters: 889,344
- Small dim: 192
- Transformer blocks: 2
- Cosine loss weight: 0.5
- Batch size: 2
- Total train epochs: 8
- Epoch 6: MSE 7.1303, cosine 0.8730
- Epoch 7: MSE 7.0695, cosine 0.8741
- Epoch 8: MSE 7.0040, cosine 0.8753
- Inference sample PPL: 384.9074

## GPT-2 Small Low-LR Fine-Tune Evaluation

- Eval texts: 20
- Eval prompts: 5
- Original PPL: 65.7304
- PathOracle PPL: 138.2357
- Relative PPL increase: 110.31%
- PPL ratio: 2.10x
- Go/No-Go PPL criterion: passed (`ratio <= 2.5`)
- Target PPL criterion: not passed (`ratio > 2.0`)
- Cosine criterion: not passed (`0.8753 < 0.88`)
- Generation quality: PPL improved slightly over epoch 5, but phrase templates remain severe (`very difficult to be`, `team of the team`).

## GPT-2 Small Mixed WikiText+c4 Run

- Run tag: `gpt2_mixed_wiki5000_c4_2000_len64_tr_sdim192_nb2_cos0.5_e5`
- Data directory: `oracle_data_gpt2_mixed_wiki5000_c4_2000_len64`
- Source data:
  - WikiText-2: 5000 samples from `oracle_data_gpt2_wiki5000_len64_tr_sdim192_nb2_cos0.5_e5`
  - c4: 2000 samples from `oracle_data_gpt2_c4_2000_len64`
- c4 source note: used `allenai/c4` with streaming because the legacy `c4` dataset script is not supported by the installed `datasets` version.
- Mixed samples: 7000
- Max length: 64
- Oracle type: Transformer
- Oracle parameters: 889,344
- Small dim: 192
- Transformer blocks: 2
- Cosine loss weight: 0.5
- Batch size: 2
- Train epochs: 5
- Epoch 1: MSE 81.8153, cosine 0.8250
- Epoch 2: MSE 8.0769, cosine 0.8562
- Epoch 3: MSE 7.7877, cosine 0.8615
- Epoch 4: MSE 7.5931, cosine 0.8650
- Epoch 5: MSE 7.4724, cosine 0.8671
- Inference sample PPL: 253.3163

## GPT-2 Small Mixed WikiText+c4 Evaluation

- Eval texts: 20
- Eval prompts: 5
- Original PPL: 65.7304
- PathOracle PPL: 137.8258
- Relative PPL increase: 109.68%
- PPL ratio: 2.10x
- Go/No-Go PPL criterion: passed (`ratio <= 2.5`)
- Target PPL criterion: not passed (`ratio > 2.0`)
- Cosine criterion: not passed (`0.8671 < 0.88`)
- Generation quality: PPL is slightly better than the low-LR fine-tune, but phrase templates remain severe (`capital of the capital`, `book that I've been a book`, `United States, the United States`, `first, the first`).

## GPT-2 Small 5000-Sample Small-Dim-256 Run

- Run tag: `gpt2_wiki5000_len64_tr_sdim256_nb2_cos0.5_e5`
- Data directory: `oracle_data_gpt2_wiki5000_len64_tr_sdim192_nb2_cos0.5_e5`
- Model: `gpt2`
- Samples: 5000
- Max length: 64
- Oracle type: Transformer
- Oracle parameters: 1,447,936
- Small dim: 256
- Transformer blocks: 2
- Cosine loss weight: 0.5
- Batch size: 1
- Train epochs: 5
- Epoch 1: MSE 57.2251, cosine 0.8426
- Epoch 2: MSE 7.2082, cosine 0.8724
- Epoch 3: MSE 6.8239, cosine 0.8794
- Epoch 4: MSE 6.5613, cosine 0.8841
- Epoch 5: MSE 6.3907, cosine 0.8871
- Inference sample PPL: 283.1636

## GPT-2 Small Small-Dim-256 Evaluation

- Eval texts: 20
- Eval prompts: 5
- Original PPL: 65.7304
- PathOracle PPL: 126.2876
- Relative PPL increase: 92.13%
- PPL ratio: 1.92x
- Target PPL criterion: passed (`ratio < 2.0`)
- Cosine criterion: partial (`0.8871`, in the 0.88-0.89 band)
- Generation quality: best GPT-2 PPL so far, but repeated phrase templates remain (`great book that is a book`, `United States was the United States`, `most beautiful`).
- Status: best GPT-2 checkpoint so far.

## Notes

The MVP validates the end-to-end CPU pipeline: hidden-state collection, oracle training, patched inference, generation comparison, and PPL measurement all run locally. Generation quality is poor with the current 512-sample MLP oracle and tends to repeat common tokens, so this checkpoint should be treated as a pipeline proof, not a quality target.
