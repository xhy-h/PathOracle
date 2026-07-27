# PathOracle CPU MVP

用一个小型"神谕网络"（Oracle）跳过 Transformer 模型的中间层，在推理时减少计算量。

本项目包含两套正交的算力节省方案：

| 方案 | 思路 | 质量保障 |
|---|---|---|
| **Plan A — FFN 稀疏性利用** (``sparsity_oracle/``) | 利用 FFN 激活层固有 85%+ 的稀疏性，只计算非零激活对应的权重列 | **零退化** — 数学上等价于原始模型 |
| **Plan B — 投机式 PathOracle** (当前管线) | Oracle 预测跳过层的 hidden state + 置信度检查 + 低置信度时回退到原始计算 | **严格保证** — 回退机制确保输出与原始模型一致 |

---

## 前提

- Python ≥ 3.10
- 虚拟环境（推荐）

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> CPU-only 运行：设置环境变量 `OMP_NUM_THREADS=4`（默认值），各脚本启动时自动读取。

---

## 管线概览（Plan B — 基础管线）

五个独立脚本串联成一条数据管线，步骤间通过文件系统传递中间产物：

```
smoke_test.py  →  collect_data.py  →  train_oracle.py  →  inference_pipeline.py
                                                              ↓
                                                         evaluate.py
```

| 步骤 | 脚本 | 产物 |
|---|---|---|
| ① 环境验证 | `smoke_test.py` | 打印模型结构和 Oracle 参数量 |
| ② 采集配对 | `collect_data.py` | `oracle_data_*/pair_*.npz` + `metadata.json`（支持增量恢复）|
| ③ 训练 Oracle | `train_oracle.py` | `oracle_*.pth`（含权重、配置、历史、优化器状态、**置信度头**）|
| ④ 推理管线 | `inference_pipeline.py` | 生成文本 + PPL（标准模式 / **投机模式**）|
| ⑤ 评测 | `evaluate.py` | 生成对比 + WikiText-2 PPL → `eval_results_*.json` |

### 快速运行（distilgpt2）

```powershell
# ① 看模型结构和 Oracle 有多少参数
python smoke_test.py --preset distilgpt2

# ② 采集 512 条 hidden-state 配对
python collect_data.py --preset distilgpt2 --run-tag my_first_run

# ③ 训练 Oracle（3 个 epoch），自动训练 MSE 回归置信度头
python train_oracle.py --preset distilgpt2 --run-tag my_first_run --epochs 3

# ④ 标准推理：Oracle 直接近似跳过层
python inference_pipeline.py --preset distilgpt2 --run-tag my_first_run

# ④ 投机推理：置信度检查 + 回退 → 质量保证
python inference_pipeline.py --preset distilgpt2 --run-tag my_first_run --speculative

# ⑤ 评测
python evaluate.py --preset distilgpt2 --run-tag my_first_run
```

也提供了封装脚本：

```powershell
.\run_smoke.ps1                # 端到端冒烟
.\run_full_distilgpt2.ps1      # 完整 MVP 运行
```

---

## Plan A — FFN 稀疏性利用（`sparsity_oracle/`）

### 核心原理

Transformer FFN 的 GELU 激活层输出有 **85-95% 的稀疏性**——即每个 token 只激活 FFN 隐藏维度的 5-15%，其余输出为零。

```
对任意 token：GELU(v) 中 ~90% 的元素 ≈ 0
     → down_proj 矩阵中对应的列完全不需要计算
```

这是模型训练后**自发 learned 的固有特性**，不是人为引入的。利用它不改变任何模型行为。

### 工具

```bash
# 分析 FFN 激活稀疏性（分布、百分位、top-k 浓度）
python -m sparsity_oracle.analyze_sparsity --preset distilgpt2 --samples 200 --dist

# 评估稀疏 FFN 质量（全量 vs 稀疏的余弦相似度）
python -m sparsity_oracle.evaluate --preset distilgpt2 --topk 0.15 --samples 100

# FLOPs 节省估算（含 MoE 模型外推）
python -m sparsity_oracle.benchmark --preset distilgpt2 --seq-len 4096
```

### 实测结果（distilgpt2）

| 指标 | 数值 |
|---|---|
| 总体激活稀疏度 | **40.7%**（均值，但层间差异大：96.4% ~ 12.5%）|
| Top-15% 质量保留 | **0.991 余弦相似度**（所有层 > 0.97）|
| FLOPs 节省（sparsity FFN） | **~19.6%** 整模型 |
| MoE 外推（61 层/7168h） | **~17.7%** 整模型 |

---

## Plan B — 投机式 PathOracle（置信度 + 回退）

### 工作流程

```
输入 hidden
      │
      ├── Oracle 预测跳过层的输出 h_pred
      │
      ├── 置信度头检查预测质量
      │     （MSE 回归预测 / 二分类置信度）
      │
      ├── 置信度 ≥ 阈值 → 接受 h_pred，跳过 k 层
      │
      └── 置信度 < 阈值 → 回退到原始 k 层计算
           输出与原始模型严格一致
```

### 置信度头

| 类型 | 输出 | 含义 |
|---|---|---|
| `ConfidenceHead`（二分类） | `[0, 1]` | 越高 = 预测越可靠 |
| `MSEPredictionHead`（回归，**默认**） | `(0, +∞)` | 值越低 = 误差越小 |

训练时，置信度头与 Oracle 联合训练：

```python
# 对每个 token，计算 Oracle 预测与目标之间的 per-token MSE
# 回归目标：直接用这个 MSE 值
# 损失函数：oracle_loss + confidence_coeff * MSE(pred_mse, actual_mse)
```

### 两种推理模式

```bash
# 标准模式（近似跳过）
python inference_pipeline.py --preset distilgpt2 --run-tag myrun

# 投机模式（质量保证）
python inference_pipeline.py --preset distilgpt2 --run-tag myrun --speculative

# 调整置信度阈值
python inference_pipeline.py --preset distilgpt2 --run-tag myrun --speculative \
    --confidence-threshold 1.5
```

> 注意：MSE 回归模式使用 `≤ threshold` 判断接受；二分类模式使用 `≥ threshold`。阈值含义见训练时的 MSE 分布。

---

## 配置

通过 `--preset` 选择预设（`distilgpt2` / `gpt2`），各脚本支持同名参数覆盖默认值。`--run-tag` 自动派生独立的数据目录和检查点路径，多个实验互不干扰。

| 参数 | distilgpt2 默认 | gpt2 默认 |
|---|---|---|
| `total_layers` | 6 | 12 |
| `early_count`（前置层数） | 2 | 2 |
| `late_count`（后置层数） | 2 | 2 |
| `target_layer_start` | 4 | 10 |
| `small_dim`（Oracle 隐层宽度） | 96 | 128 |
| `oracle_type` | mlp | mlp |
| `cos_weight` | 0.1 | 0.1 |
| `confidence_threshold` | 0.8 | 0.8 |
| `confidence_coeff` | 0.1 | 0.1 |
| 默认数据集 | wikitext-2-raw-v1 | wikitext-2-raw-v1 |

### 层跳过模式

```
distilgpt2 (6 层):   [0][1] → 跳过 [2][3] → [4][5]
                       ↑ early     Oracle     ↑ late

gpt2 (12 层):        [0][1] → 跳过 [2..9] → [10][11]
                       ↑ early     Oracle     ↑ late
```

---

## 架构

```
pathoracle_cpu_mvp/
│
├── config.py                  # 配置数据类 + 预设（含置信度参数）
├── oracle_utils.py            # 共享工具（embedding、tokenizer、PPL、logging、CPU 线程）
├── oracle_model.py            # MLP / Transformer Oracle + ConfidenceHead + MSEPredictionHead
│
├── collect_data.py            # ② 采集 (early, target) 配对（支持增量恢复）
├── train_oracle.py            # ③ 训练 Oracle + 置信度头（MSE + cosine + confidence loss）
├── inference_pipeline.py      # ④ PathOracleGPT2 推理类（标准 / 投机模式）
├── evaluate.py                # ⑤ 生成 + PPL 对比评测（结构化 JSON 输出）
├── smoke_test.py              # ① 环境验证
│
├── sparsity_oracle/           # Plan A：FFN 稀疏性利用
│   ├── analyze_sparsity.py    #   稀疏性分析（分布、百分位、per-position）
│   ├── sparse_ffn.py          #   Top-k 稀疏 FFN 实现
│   ├── evaluate.py            #   稀疏 FFN 质量评估（余弦相似度）
│   └── benchmark.py           #   FLOPs 节省估算（含 MoE 外推）
│
├── tests/                     # 16 个单元测试
│   ├── test_config.py
│   ├── test_oracle_model.py
│   ├── test_train_resume.py
│   ├── test_utils.py          # 新增：工具函数测试
│   └── test_inference.py      # 新增：配置校验测试
│
├── docs/                      # 白皮书 & 迁移计划
├── pyproject.toml             # 项目元数据（Python ≥ 3.10）
└── requirements.txt           # 依赖
```

---

## 检查点恢复

```powershell
# 从中断处恢复
python train_oracle.py --preset distilgpt2 --run-tag myrun --resume

# 从另一个 run_tag 的检查点恢复（微调）
python train_oracle.py --preset distilgpt2 --run-tag myrun_v2 --resume-from myrun

# 恢复权重但重置优化器（支持微调）
python train_oracle.py --preset distilgpt2 --run-tag myrun --resume --reset-optimizer
```

检查点保存内容：
- `model_state_dict` — Oracle 权重
- `confidence_head_state_dict` — 置信度头权重（Plan B 投机模式需要）
- `optimizer_state_dict` / `scheduler_state_dict` — 恢复训练
- `config` — 完整训练配置（`oracle_type`, `small_dim`, `confidence_threshold`, 等）
- `history` — 每个 epoch 的 MSE 和 cosine 曲线

---

## 测试

```powershell
python -m pytest tests/ -v
```

16 个测试覆盖：配置系统、Oracle 模型构造、检查点恢复逻辑、共享工具函数（CPU 线程、tokenizer、PPL 计算）、配置校验（层边界、形状验证）。

---

## 两个方案的协同

Plan A 和 Plan B 正交互补，可叠加使用：

```
Plan A ─── 每层的 FFN 计算减少 85% → 约 20% 整模型节省（零退化）
Plan B ─── 跳过整层 + 置信度回退  → 约 50% 层跳过（质量保证）
─────────────────────────────────────────────────────────────
叠加估算          → 潜在总节省 ~60-70%（严格质量保证）
```
