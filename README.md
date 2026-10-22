<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
-->

# KernelBench Pro

用于 GPU 内核生成、正确性验证和性能评测的批量工作流。
由 [shadowfall09/KernelBench](https://github.com/shadowfall09/KernelBench) 维护。

## 当前能力

- 统一运行基线、生成、评测和分析，支持按题目范围运行和多样本尝试。
- 生成支持本地、Docker、Modal 和 Harbor；验证支持本地、Docker、Modal 和 HTTP 队列。
- GPU 型号、架构、设备和云端类型统一使用 TOML 配置；支持型号预设及自定义目标。
- 原生生成接入 mini-swe-agent：支持 one-shot 和有预算的反馈循环，开发反馈与最终隐藏评测分开。
- 默认比较四种基线，可配置扩展compile/Graph组合和可信框架改写；支持交替计时、噪声控制与冻结后独立重测。
- 使用同步墙钟计时，逐次更换输入并校验输出，记录稳态耗时和冷启动成本。
- 严格检查输出结构、shape、dtype、整数、bool 和非有限值；隐藏测试使用三个种子，29 道 Level 1 题另有数值压力测试。
- 可选四种Compute Sanitizer工具与题目级重复一致性检查，检查耗时与性能采样分开。
- 可选受限Docker验证器和Harbor网络策略；支持冻结已有候选后用当前验证器重评分，默认仅生成计划。
- 评分包含失败和缺失样本，恢复时核对源码、配置和测量证据。

默认批量流程支持 Level 1–3。`KernelBench/level4/` 保留实验题目，不属于默认批量评测范围。

项目按五条 motivation 组织：题目区分度、正确性与执行可信度、性能指标、Agent 回路和基础设施。
[动机与逐项对照](docs/MOTIVATION.md)覆盖每版改动和未完成设计；
[统一研究稿](research/headroom/PAPER.md)说明 Headroom 目标与当前实现的差额。
当前提供 NVIDIA CUDA 和[原生Neuron/NKI设备工作流](docs/NEURON.md)；后者为软件接入及CPU/模拟验证，实机部署与性能未验证。可选Headroom评分要求真实设备校准和可信规格，缺证据不计分。
显式严格模式已为17道L1加入附加FP64契约与证据等级，见[严格验证](docs/STRICT_VALIDATION.md)。

## 安装

使用 Python 3.12。项目固定依赖 Torch 2.12.0。

```bash
git clone https://github.com/shadowfall09/KernelBench.git
cd KernelBench
uv sync --extra dev
source .venv/bin/activate
python scripts/run_batch.py --help
```

需要 Triton、CuTe、TileLang 等可选工具时，使用 `uv sync --extra dev --extra gpu`。
本地验证需要 CUDA GPU 和兼容的编译工具链；本地生成还需要 Claude Code CLI 及运行时认证。
也可配置 `agent_backend = "litellm"` 和显式供应商/模型，使用对应 API 认证。

## Harbor

Harbor 使用同一个批量入口和结果格式；安装 `uv sync --extra dev --extra harbor` 后，选择 `--generation harbor`。也可用 `--stage export-harbor` 只导出任务。完整配置与独立验证环境说明见 [Harbor 接入](docs/HARBOR.md)。

## 运行

```bash
# 预览配置与题目范围，不创建运行目录或调用模型。
python scripts/run_batch.py --dry-run \
  --config configs/a6000.toml

# 基线 → 生成 → 评测 → 分析；命令行可覆盖配置中的运行参数。
python scripts/run_batch.py --config configs/a6000.toml --resume
```

修改配置中的 `[gpu] profile` 和 `devices` 即可切换目标。自定义 GPU 可直接填写硬件标签和架构；不再默认选择 H100/Hopper。配置格式、覆盖优先级和云端示例见 [GPU 配置](docs/GPU_CONFIG.md)。

一次生成使用 `configs/a6000-one-shot.toml`；反馈循环使用 `configs/a6000-feedback.toml`，默认每个样本最多 5 次候选、40 分钟，可配置。两种模式使用相同最终评测；完整协议、模型入口、预算及比较方式见 [Agent loops](docs/AGENT_LOOPS.md)。Claude Code 可在仓库中使用 `/kernelbench-pro`；Codex 的 `.agents/skills` 入口共享同一份 skill。

在集群上通过 `srun` 或 `sbatch` 获取 GPU 后执行本地评测。Docker、Slurm、Modal、远端队列的完整配置见 [批量工作流](docs/BATCH_WORKFLOWS.md)。

显式添加 `--suite kernelbench105` 可使用研究选集；`start/end` 与选集取交集。
105 道主集、5 道探针与140 道扩展题分别运行和报告，默认仍为完整题集。
选择规则、`--suite-role` 用法和未校准输入等局限见 [Headroom 研究索引](research/headroom/README.md)。

运行产物写入 `runs/NAME/`，基线写入 `results/timing/HARDWARE/`：

| 文件 | 内容 |
| --- | --- |
| `eval_results.json` | 标准正确性、耗时和显存 |
| `eval_results_hidden.json` | 隐藏正确性与检查记录 |
| `validation_summary.json` | 编译、正确性和计时阶段统计 |
| `paper_analysis.json` | 保留现有文件名的性能评分报告 |
| `eval_results_scored.json` | 通过证据检查后的计分资格 |
| `pass_at_k_results.json` | 正确性 pass@k |
| `generation_budget.json` | 生成模式、声明预算、实际调用及费用/token 回执；未知值保留 null |
| `generation_attempts.json` | 包含重试的样本尝试历史；重复运行不隐藏前次成本 |
| `phase_budget.json` | 基线、生成、最终验证与分析各阶段的墙钟时间 |

阶段统计中的正耗时不等于有效加速；性能结论以评分报告和对应测量证据为准。

## 开发与文档

```bash
python -m pytest -q tests src/unit_tests/test_dataset.py src/unit_tests/test_score.py
python -m ruff check kb_workflows tests src/reliability.py src/validation.py
python -m scripts.check_motivations
```

| 文档 | 用途 |
| --- | --- |
| [功能清单](docs/FEATURE_INVENTORY.md) | 逐版变化、实现入口与验证状态 |
| [Motivation](docs/MOTIVATION.md) | 每项改动对应的问题，以及论文设计和实现之间的差额 |
| [Headroom 研究](research/headroom/README.md) | 统一论文、105 题选集、审查证据与来源摘要 |
| [GPU 配置](docs/GPU_CONFIG.md) | GPU 预设、自定义架构、设备选择与覆盖规则 |
| [Agent loops](docs/AGENT_LOOPS.md) | one-shot、mini-swe-agent 循环、预算、交互 skill 与比较 |
| [Harbor 接入](docs/HARBOR.md) | 统一生成后端、任务导出与 reward |
| [批量工作流](docs/BATCH_WORKFLOWS.md) | 各后端、恢复、基线与评分配置 |
| [隔离与重评分](docs/ISOLATION_REGRADE.md) | 受限验证器、Harbor网络配置、冻结产物与重评分计划 |
| [内存工具与重复检查](docs/TOOL_CHECKS.md) | 可选Sanitizer、审核原子操作策略、证据与恢复 |
| [交替与最终计时](docs/PAIRED_TIMING.md) | 扩展基线、平衡配对、噪声控制、独立重测与恢复 |
| [Headroom评分](docs/HEADROOM_SCORING.md) | 设备校准、可信工作量证书、独立见证与时间线诊断 |
| [Production-41](docs/PRODUCTION_TASKS.md) | Level 5生产题族、逐角色dtype、显存计划和独立反向轨道 |
| [研究工作流](docs/STUDIES.md) | 当前证据导出、选集区分度、多会话计划与得分—预算曲线 |
| [资源预算](docs/RESOURCE_BUDGETS.md) | 精确本地token、API输出预留、开发worker占用与恢复证据 |
| [RL工作流](docs/RL_WORKFLOW.md) | GRPO计划、算法证书奖励、训练来源、恢复、审计与持出效果对照 |
| [严格验证](docs/STRICT_VALIDATION.md) | FP64附加契约、输入/状态检查、证据等级与聚合消融 |
| [正确性可靠性](docs/RELIABILITY.md) | 检查协议、适配范围与成本 |
| [输入盲化](docs/INPUT_BLIND_GENERATION.md) | 生成材料如何去掉测试输入 |
| [贡献指南](CONTRIBUTING.md) | 本仓库的开发与评审方式 |

Python 3.12 / Torch 2.12.0 的 CPU/模拟回归已执行；真实 CUDA 和各云后端仍需对应环境的实机验证。

## 许可证与来源

代码采用 [MIT License](LICENSE)。继承代码的版权声明及来源记录见 [NOTICE](NOTICE.md)。
