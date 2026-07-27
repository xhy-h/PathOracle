# PathOracle CPU MVP

用一个小型"神谕网络"（Oracle）跳过 Transformer 模型的中间层，在推理时减少计算量。

## 前提

- Python ≥ 3.10
- 虚拟环境（推荐）

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> CPU-only 运行：设置环境变量 `OMP_NUM_THREADS=4`（默认值），各脚本启动时自动读取。

## 管线概览

五个独立脚本串联成一条数据管线，步骤间通过文件系统传递中间产物：

```
smoke_test.py  →  collect_data.py  →  train_oracle.py  →  inference_pipeline.py
                                                              ↓
                                                         evaluate.py
```

| 步骤 | 脚本 | 产物 |
|---|---|---|
| ① 环境验证 | `smoke_test.py` | 打印模型结构和 Oracle 参数量 |
| ② 采集配对 | `collect_data.py` | `oracle_data_*/pair_*.npz` + `metadata.json` |
| ③ 训练 Oracle | `train_oracle.py` | `oracle_*.pth`（含权重、配置、历史、优化器状态） |
| ④ 推理管线 | `inference_pipeline.py` | 生成文本 + PPL |
| ⑤ 评测 | `evaluate.py` | 生成对比 + WikiText-2 PPL = `eval_results_*.json` |

## 快速运行（distilgpt2）

```powershell
# ① 看模型结构和 Oracle 有多少参数
python smoke_test.py --preset distilgpt2

# ② 采集 512 条 hidden-state 配对
python collect_data.py --preset distilgpt2 --run-tag my_first_run

# ③ 训练 Oracle（3 个 epoch）
python train_oracle.py --preset distilgpt2 --run-tag my_first_run --epochs 3

# ④ 用 Oracle 生成文本
python inference_pipeline.py --preset distilgpt2 --run-tag my_first_run

# ⑤ 评测：对比原始模型 vs PathOracle
python evaluate.py --preset distilgpt2 --run-tag my_first_run
```

也提供了封装脚本：

```powershell
.\run_smoke.ps1                # 端到端冒烟
.\run_full_distilgpt2.ps1      # 完整 MVP 运行
```

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
| 默认数据集 | wikitext-2-raw-v1 | wikitext-2-raw-v1 |

### 层跳过模式

```
distilgpt2 (6 层):   [0][1] → 跳过 [2][3] → [4][5]
                       ↑ early     Oracle     ↑ late

gpt2 (12 层):        [0][1] → 跳过 [2..9] → [10][11]
                       ↑ early     Oracle     ↑ late
```

## 架构

```
oracle_utils.py        ← 共享工具（embedding、tokenizer 初始化、PPL 计算）
config.py              ← 配置数据类 + 预设
oracle_model.py        ← MLP / Transformer Oracle 定义
collect_data.py        ← 步骤②：提取 (early, target) 配对
train_oracle.py        ← 步骤③：训练 Oracle（MSE + cosine loss）
inference_pipeline.py  ← 步骤④：PathOracleGPT2 推理类
evaluate.py            ← 步骤⑤：生成 + PPL 对比评测
smoke_test.py          ← 步骤①：环境验证
```

## 检查点恢复

```powershell
# 从中断处恢复
python train_oracle.py --preset distilgpt2 --run-tag myrun --resume

# 从另一个 run_tag 的检查点恢复（微调）
python train_oracle.py --preset distilgpt2 --run-tag myrun_v2 --resume-from myrun

# 恢复权重但重置优化器
python train_oracle.py --preset distilgpt2 --run-tag myrun --resume --reset-optimizer
```

## 测试

```powershell
python -m pytest tests/ -v
```

涵盖：配置系统、Oracle 模型构造、检查点恢复逻辑、共享工具函数、配置校验。

## 项目结构

```
pathoracle_cpu_mvp/
├── config.py                  # 共享配置
├── oracle_utils.py            # 共享工具
├── oracle_model.py            # Oracle 模型
├── collect_data.py            # 采集配对
├── train_oracle.py            # 训练
├── inference_pipeline.py      # 推理管线
├── evaluate.py                # 评测
├── smoke_test.py              # 冒烟测试
├── tests/                     # 单元测试
├── docs/                      # 白皮书 & 设计文档
├── pyproject.toml             # 项目元数据
└── requirements.txt           # 依赖
```
