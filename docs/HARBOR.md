# KernelBench Pro 与 Harbor

Harbor 是现有批量流程的生成后端，和 local、Docker、Modal 共用 `RunConfig`、题目范围、样本槽位、结果文件与恢复逻辑。Python 统一为 3.12，验证依赖保持 Torch 2.12.0。

## 安装

```bash
uv sync --extra dev --extra harbor
source .venv/bin/activate
harbor --version
python scripts/run_batch.py --help
```

可选 `harbor` extra 安装 Harbor 0.24.0 及其 Modal 支持；不使用 Harbor 时无需安装。当前适配依据 [Harbor 源码 6c9d3d6](https://github.com/harbor-framework/harbor/tree/6c9d3d6609854375005a7b5c8f1906371d64c6ff)，使用 schema 1.4、独立 verifier 和显式产物传递。自带其他 Harbor 版本时需核对这些接口。

## 同一入口运行完整流程

先为所选 Harbor agent 配置认证，再运行：

```bash
python scripts/run_batch.py \
  --generation harbor --evaluation local \
  --harbor-agent claude-code --harbor-environment docker \
  --model 'your-provider/your-model' \
  --level 1 --start 1 --end 10 --num-samples 2 \
  --run-name harbor_run --hardware RTX_A6000 --gpu RTX_A6000 \
  --gpu-arch Ampere --gpus 0 --parallel 1 --resume
```

将 `--model` 换成你已配置的模型。Docker 需要 NVIDIA Container Toolkit；集群上先通过 Slurm 获取 GPU。各本地 GPU 上的 Harbor 任务串行执行，生成的 Compose 定义把 agent 和 verifier 绑定到所分配的同一设备。不同 GPU 可以并行。

完整流程仍按原有顺序运行：基线 → Harbor 生成 → 本项目正式评测 → 分析。每个 `(题号, sample_id)` 对应一个 Harbor job、一个 attempt，Harbor 回收的 `solution.py` 写入原有 `level_L_problem_P_sample_S_kernel.py`。正式评测重新验证回收的源码，不直接把 Harbor reward 当作可信性能结果。

Harbor 在生成任务内部也会执行 verifier，所以完整流程会额外产生一轮验证与基线成本。可以使用 `--stage generate` 先只生成，再用同一配置执行 `evaluate` / `analyze`。Harbor agent、模型、环境或验证器源码变化会使生成恢复指纹失效。

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--harbor-command` | `harbor` | Harbor 可执行文件，可使用虚拟环境中的绝对路径 |
| `--harbor-agent` | `claude-code` | Harbor agent 标识 |
| `--harbor-environment` | `docker` | Docker 或 Modal；这是 Harbor 内部环境，不是正式评测的 `--evaluation` |
| `--harbor-reward` | `correctness` | `correctness` 或 `bounded_gain` |
| `--harbor-memory-mb` | 65536 | agent/verifier 的 CPU RAM 上限；大输入题可提高，显存由 GPU 型号决定 |
| `--agent-timeout` | 2400 秒 | agent 执行预算 |
| `--timeout` | 300 秒 | 单个原生 GPU 验证/基线子任务预算 |

Modal 示例使用 `--harbor-environment modal --gpu H100 --gpu-arch Hopper`，并配置 Harbor/Modal 运行时凭据；正式评测后端仍可独立选择。`--profile` 提供的 AWS profile 在宿主上解析为临时运行时凭据，不复制 profile 文件或把凭据写入导出任务。其他认证按所选 Harbor agent 的原生支持配置；`--small-model` 和原生 `--optimization-rounds` 不控制 Harbor agent 的内部策略。

## 只导出 Harbor 任务

```bash
python scripts/run_batch.py --stage export-harbor \
  --level 1 --start 1 --end 10 --num-samples 1 \
  --run-name harbor_export --gpu H100 --gpu-arch Hopper

harbor run -p runs/harbor_export/harbor_tasks \
  -a claude-code -m 'your-provider/your-model' -e modal -k 1 -n 1
```

每个样本已经单独导出为一个任务，默认使用 `-k 1`。新 run name 避免覆盖既有导出。`--dry-run` 只预览配置，不创建任务或调用模型。

```text
runs/NAME/harbor_tasks/l1-p1-s0/
├── task.toml
├── instruction.md
├── environment/          # Python 3.12 / CUDA、依赖和 model.py
├── solution/             # Harbor oracle 的参考控制实现
└── tests/
    ├── Dockerfile
    ├── test.sh
    ├── config.json
    └── repo/             # 独立 verifier 的可信代码、题目和隐藏测试
```

agent 镜像不包含 `tests/`、oracle、隐藏测试或完整仓库。`model.py` 沿用当前输入盲化策略。根据 [Harbor 独立 verifier 协议](https://docs.harborframework.com/tasks/verifier)，正式验证在新环境中进行，仅传递 `/app/solution.py`；其他 agent 文件系统修改不继承。

oracle 使用参考 PyTorch 实现检查任务接线，因此 correctness reward 可以通过，但不代表自定义内核或性能加速。检查导出接线可用 `harbor run -p PATH -a oracle -e docker -k 1 -n 1`，仍然需要兼容 GPU。

## 验证与输出

verifier 复用四变体基线、标准正确性、多种子隐藏检查、同步墙钟计时、冷成本和证据评分。初始化 reward 为 0；缺失提交、协议不匹配或运行异常均保留 0，清理可能过期的 `reward.json`。

| 输出 | 含义 |
| --- | --- |
| `/logs/verifier/reward.txt` | correctness 模式为全部标准/隐藏正确时的 1，否则 0；bounded_gain 模式使用有当前测量证据的原生 `[0,1]` 评分 |
| `/logs/verifier/kernelbench_pro_report.json` | 正确性、性能分数、资格状态及异常 |
| `/logs/verifier/kernelbench-pro/runs/harbor/` | 原格式正确性结果、计时收据、pass@k、评分与子任务日志 |
| `runs/NAME/work/pP_sS/harbor_jobs/` | 统一生成后端的 Harbor job、轨迹和回收产物 |

独立 verifier 隔离的是 agent 阶段的文件改动。候选内核仍会作为不可信代码在 GPU 子进程中执行，此适配不宣称完成所有运行时反作弊或安全隔离检查。

## 验证记录

Python 3.12.12 / Torch 2.12.0 下，完整回归 **204 项通过**（原有 172 项和新增 32 项）。正式发布的 Harbor 0.24.0 的真实 `Task` 校验、CLI 配置解析及依赖一致性检查通过。Docker 构建定义检查通过，未据此宣称镜像完整构建或 GPU agent 流程实机跑通。

Harbor Docker 的 CPU 容器 smoke 尝试因宿主 NAT 网络创建失败而中断；即使命令退出码为 0，也没有产物或 reward，未将其记为通过。生成后端同样要求唯一非空提交，随后重新执行原生正式评测。

当前验证宿主的驱动被 Torch CUDA 13 构建报告为过旧，因此尚未完成真实 GPU / Harbor agent 端到端验证；需要在兼容的已分配 GPU 上检查。
