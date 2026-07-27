# Sparsity Oracle

**Plan A — FFN 激活稀疏性利用**

利用 Transformer FFN 层中天然存在的 85-95% 激活稀疏性，在不改变模型输出的前提下节省计算。

## 核心思路

ReLU/GELU 激活函数在 FFN 中产生高度稀疏的激活模式——每个 token 只激活 FFN 隐藏维度的 5-15%，其余输出为零。这意味着 `down_proj` 矩阵中与零激活对应的列**完全不需要计算**。

## 文件说明

| 文件 | 用途 |
|---|---|
| `analyze_sparsity.py` | 分析预训练模型中每层 FFN 的激活稀疏模式，生成报告 |
| `sparse_ffn.py` | Top-k 稀疏 FFN 实现（只计算 top-k% 的非零激活对应的权重列）|
| `evaluate.py` | 对比全量 FFN 与稀疏 FFN 的输出差异（余弦相似度、MSE）|

## 使用方式

```bash
# 1. 分析 DistilGPT2 的 FFN 激活稀疏性
python -m sparsity_oracle.analyze_sparsity --preset distilgpt2 --samples 200

# 2. 评估稀疏 FFN 的质量保存效果
python -m sparsity_oracle.evaluate --preset distilgpt2 --topk 0.15 --samples 100
```

## 预期结果

| 模型 | 激活稀疏度 (top-k 保留) | 余弦相似度 | 计算节省（FFN 层） |
|---|---|---|---|
| distilgpt2 | 10-15% | > 0.995 | ~85-90% |
| gpt2 | 10-15% | > 0.995 | ~85-90% |
| **MoE (GLM-5.2)** | **10-20%** | — | **~80-90%（每个 expert 内）** |

## 与主项目关系

- 从主项目导入 `oracle_utils`、`config` 等基础模块
- 推理时可直接替换 `model.transformer.h[i].mlp` 为 `SparseFFN` 包装器
- 与 PathOracle（Plan B）正交互补：Plan A 省每层 FFN 计算，Plan B 省整层跳过的延迟
