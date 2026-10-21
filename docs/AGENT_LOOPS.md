# One-shot 与 mini-swe-agent 反馈循环

原生生成现在使用 **mini-swe-agent 2.4.6 的 `DefaultAgent`**。它负责消息历史、查询、步数与费用检查和轨迹保存；本项目实现 Model 和 Environment 适配器，提供完整源码提交动作以及可信开发反馈，不使用默认的任意 shell 环境。

## 两种可控模式

| 模式 | 每个样本的候选上限 | 生成阶段反馈 | 最终选择 |
| --- | --- | --- | --- |
| `one-shot` | 1 | 不执行标准测试、profiling 或隐藏测试 | 唯一源码候选 |
| `feedback` | `optimization_rounds`，默认 5 | 标准编译、正确性、同步墙钟耗时及可选 profiler 诊断 | 开发验证通过的最低平均耗时候选 |

每个样本默认总墙钟预算为 `agent_timeout = 2400` 秒，包含模型调用和开发评测等待；原生控制器超时会停止自己的模型和评测子进程树，保留已完成的最佳候选。远端排队任务的取消/计费取决于服务端，不据此宣称远端 GPU 立即停止。

`num_samples` 表示每题独立尝试，和循环轮数分开。多个 one-shot 样本仍是独立采样，不能称为“一次调用得到最佳结果”。默认评分继续计入失败和缺失样本，不用 best-of-n 替代失败包含均值；原有 pass@k 可另行报告。

```bash
python scripts/run_batch.py --config configs/a6000-one-shot.toml --dry-run
python scripts/run_batch.py --config configs/a6000-feedback.toml --dry-run

# 从已分配 GPU 的环境运行；配置中的 GPU 必须与实际设备一致。
python scripts/run_batch.py --config configs/a6000-one-shot.toml --resume
python scripts/run_batch.py --config configs/a6000-feedback.toml --resume
```

GPU 由 `[gpu]` 统一配置。local / Docker / Modal 生成复用同一个 loop；验证可使用 local / Docker / Modal / HTTP 队列。依赖由 `requirements.txt` 安装，Docker 和 Modal 镜像使用同一版本。

## 模型与预算配置

```toml
[run]
generation_mode = "feedback"
agent_backend = "claude-code"
optimization_rounds = 5
agent_timeout = 2400
agent_cost_limit = 0.0
agent_profile = true
```

`claude-code` 使用已配置的 CLI 认证，每个候选请求关闭内置工具、MCP 和自定义加载，限制为一个 agent turn，由控制器接收源码。`one-shot` 不靠 prompt 劝 agent 少测试。为便于比较，这个适配器不开放 Claude 自己的文件编辑或 Bash 工具。CLI 必须支持 `--safe-mode`、`--tools`、`--strict-mcp-config` 等当前官方参数。

`agent_backend = "litellm"` 则使用 LiteLLM，`model` 需显式填写供应商/模型名。凭据沿用对应供应商的运行时环境变量；常用 Anthropic/OpenAI 配置会传给 Docker/Modal，不写入任务或日志配置。模型响应中可观测到的实际模型名与声明模型分别记录。

`agent_cost_limit` 为客户端估算费用上限，0 表示不设美元上限。循环在调用之间检查，单次请求可能超过上限；它不是账单硬限制。要求费用上限但供应商没有费用证据时，停止继续调用。未知费用或 token 不记为零。API SDK 内部重试、服务端推理和 GPU 账单不能由候选次数精确推算；记录的 model_calls 是适配器调用尝试数。

可选资源预留增加本地`local-transformers`精确总token计数、单请求LiteLLM输出限额和local/Docker开发worker占用额度；配置、失败处理、恢复限制和计分证据见[资源预算](RESOURCE_BUDGETS.md)。设备额度按父进程墙钟结算，尚未提供纯GPU busy时间硬限。不会自动下载或运行本地checkpoint。

策略训练另提供固定版本TRL的one-shot/反馈GRPO入口，开发工具复用本节控制器，终局才进入完整奖励验证；计划、恢复和审计范围见[RL工作流](RL_WORKFLOW.md)。训练不自动启动，持出题不进入训练callback。

## 反馈与最终验证分开

控制器只把经过现有 input-blind 策略处理的公开参考发给模型。反馈通道明确关闭隐藏测试，模型没有读取测试源码的 shell/file 工具。开发正确性和计时复用当前评测器，仍检查输出结构、dtype、非有限值、变化输入及逐次计时输出。

开发正确后，可在独立子任务中用 `torch.profiler` 收集 CPU/device 时间、调用次数和 kernel/operator 名称。失败的 profiler 标为 unavailable；这些时间不参加最佳候选选择或正式评分，也不冒充 Nsight 硬件计数器。profiler 运行的仍是当前候选，而不是已保留的其他版本。

候选冻结后，原生评测重新运行标准和多种子隐藏测试，再做基线/测量证据评分。隐藏结果不返回给自动循环。候选源码仍作为不可信 Python 在评测进程执行；此适配不提供完整 OS 安全隔离，也不宣称杜绝所有反作弊攻击。

## 结果与比较

每个样本保留 `trajectory.json`、`model_calls/`、`rounds/`、`development_history.json`、`best_candidate.json` 和 `agent_result.json`。Modal 收回轨迹、开发历史和结果摘要；详细子目录留在临时 sandbox，摘要包含预算与每轮诊断。

父运行的 `generation_budget.json` 记录模式、框架、模型、独立样本数、候选/墙钟/估算费用限制及实际资源回执。缺失样本、被强制终止而未完成的回执或外部框架不可观测的费用保持 unknown/null。源码 hash 绑定生成回执；生成后改写源码会标为 unattributed，不能继续冒充原来的实验模式。

`--resume` 保留指纹和产物匹配的已完成失败，不自动再花一轮预算。显式重跑同一目录会在 `generation_attempts.json` 中追加尝试，累计可观测成本，并标记为 `repeated-attempts`，不再当作单次 one-shot/固定预算实验。中断的 started 尝试保留未知成本；历史尝试无法追溯时总成本保持未知。新的独立实验使用新 run name。

`phase_budget.json` 分开记录本次命令的基线、生成、最终验证与分析墙钟时间，包含请求等待和编译，不包含前置镜像构建与环境启动；它不是 GPU 实际执行时间或云账单。`paper_analysis.json` 附带协议及资源回执。

```bash
python -m kb_workflows.experiment runs/a6000_one_shot runs/a6000_feedback
```

比较器按模式、框架、模型适配器、声明预算、采样数、题集、硬件、精度和验证参数分组；不同模型可列在同组，实际模型名显示在各自记录中。不同组不合并排名。该报告展示已有评分，不能替代正式验证或当前测量证据检查。旧 HTML 榜单是历史展示工具，不作为新实验模式混合比较的入口。

Harbor 仍可用于外部 agent，但标为 `external`，不声称其内部执行符合本项目 5 次候选限制或隐藏反馈隔离；不允许把 Harbor 外部 agent 标成可控 one-shot。

## 直接使用 Claude Code / Codex

项目 skill 位于 [SKILL.md](../.claude/skills/kernelbench-pro/SKILL.md)，Claude Code 可用 `/kernelbench-pro`；`.agents/skills/kernelbench-pro` 指向同一份 skill，避免维护两套说明。

在当前 coding-agent 会话中，`python -m kb_workflows.agent_task prepare ...` 准备一个任务，`verify.py` 提供标准开发反馈，`finalize` 冻结最佳源码，再由原生 CLI 执行最终验证。这个模式不递归调用 Claude CLI，也不清除嵌套会话保护。由于当前会话模型/tool 调用不可完全观测，它标记为 `interactive`，不冒充受测 one-shot 或自动 mini-swe-agent 轨迹。交互状态文件不是安全边界；最终正确性仍来自可信评测。

## 参考与选择依据

核对日期：2026-10-21。没有找到在本项目相同题集、GPU、模型与预算下证明某一框架普遍更好的对照，因此这里按接入边界选择，不转用论文中不同设定的加速比作排名。

| 项目 | 可取之处 | 本次取舍 |
| --- | --- | --- |
| [MLC / TIRx Harness](https://mlc.ai/agentic-gpu-programming-for-mlsys/) | 将编译分析、诊断、基准与 agent workflow 分开 | 参考诊断/正式计时分离和固定契约；不引入 TIRx、KCoral、Flame Chase 依赖 |
| [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) | 简单可扩展的 Model/Environment 接口、步数/费用限制、线性轨迹 | 实际接入发布版 2.4.6 的 DefaultAgent，使用自有源码提交与反馈环境 |
| [Meta KernelAgent](https://github.com/meta-pytorch/KernelAgent) | 面向 KernelBench 的生成与 profiler 引导优化 | 本次不迁入它的多 worker/fuser 流程和默认计时体系 |
| [Atrex Kernel Agent](https://github.com/alibaba/atrex-kernel-agent) | supervisor 管理候选提升、恢复、生产评测与 coding-agent 工作流 | 参考由控制器决定接纳候选的边界；不迁入其问题格式与运行基础设施 |
| [FlashInfer agent baseline](https://github.com/flashinfer-ai/mlsys26-agent-baseline) | 现成 iterative / evolve 两类搜索 | 本次保留独立采样与有限轮反馈，不引入额外 elite-pool 进化预算 |

TIRx 的模拟、同步与竞争检查只覆盖受支持的 TIRx 程序和给定输入，不是通用 CUDA/Triton 的正确性证明；本次没有把这些能力写成已经接入。

Multi-model/session budget plans and a Harbor general-agent baseline are now available through [study workflows](STUDIES.md). Planning is nonexecuting by default; current score export, tie-aware selection diagnostics and independent budget curves do not launch models or workloads. Resource receipts distinguish declared caps, observed token/agent time, unknown values and uncertified GPU billing.
