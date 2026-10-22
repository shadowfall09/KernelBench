# KernelBench Pro 功能改进清单（相比 KernelBench-Verified）

本清单按版本维护，记录功能改进、实现入口和验证程度。第一版记录 `2bce940` 的批量工作流整合，第二版记录正确性判断与输入隔离，第三版记录基线与计时，第四版记录评分与恢复，第五版记录从相关工作引入的可靠性检查，第六版统一 KernelBench Pro 名称、Harbor 工作流与 Python 3.12，第七版统一 GPU 配置并接入有界 Agent 工作流，第八版整合 Headroom 研究、任务选集与 motivation 追踪，第九版加入显式严格契约、证据等级和聚合消融。

当前能力为第一至第十七版的累计功能；23项设计仍按代码状态和真实实验状态分别追踪，见对应版本节。此前搁置的评测完整性整合及清理草稿仍保存在原 stash 中，没有恢复；第六版的仓库清理单独完成。

当前版本继续沿用统一入口：

| 能力 | 当前配置 |
| --- | --- |
| Python | 3.12；验证依赖 Torch 2.12.0 |
| 生成后端 | local / Docker / Modal / Harbor |
| GPU 目标 | TOML 配置与预设，显式 CLI 参数覆盖 |
| 生成协议 | 原生 one-shot / mini-swe-agent feedback；Harbor external；会话 skill interactive |
| 反馈预算 | 默认每样本最多 5 次候选、2400 秒，可配置 |
| 正式验证 | local / Docker / Modal / HTTP 队列 |
| Harbor 任务 | 同一 CLI 的 `export-harbor` 阶段 |
| 结果与恢复 | 原生样本编号、文件格式、基线、计时、评分及指纹 |

下面各版分别记录对应提交、功能变化和验证范围。按动机组织的完整对照见 [Motivation](MOTIVATION.md)，统一研究稿见 [论文](../research/headroom/PAPER.md)。

第七版同时包含 GPU 配置、one-shot / mini-swe-agent 反馈循环和 coding-agent skill，见后文对应记录。

## 第一版：Verified + 批量工作流

| 项目 | 版本信息 |
| --- | --- |
| 本版提交 | `2bce940c631c01086c262eef9c9efc9d3693e1de` |
| 提交时间 | 2026-10-04 17:02:18（America/New_York，UTC−04:00） |
| 提交标题 | Replace legacy fork with Verified and integrated batch workflows |
| 记录的上游基准 | Meta KernelBench-Verified `3fdf6fec7372a4d0cb682635f00e7bdcbc55d50e` |
| 工作流来源 | 原 KernelBench fork / `KernelBench-modified` |
| 核对日期 | 2026-10-20（America/New_York） |

这一版把 Verified 的参考题目、隐藏测试和评测器，与原 fork 的批量运行能力整合到同一条流程中。使用者可以统一配置题目范围、生成方式、验证方式和结果分析，在本地、集群或云端运行实验。

### 功能清单

以下各项在本版均已提交。“迁移整合”表示把原 fork 的能力接入 Verified；“流程增强”表示统一配置、可靠性或结果处理的改进，不表示上游完全没有相关基础能力。

| 编号 | 功能 | 相对 Verified 的改进与用途 | 实现入口（本版快照） | Motivation |
| --- | --- | --- | --- | --- |
| V1-01 | 统一批量入口 | **流程增强。** 用一套配置串联基线、生成、评测和分析；支持 `all`、`generate`、`evaluate`、`analyze`、`baseline` 五种阶段，减少多脚本间的手工传参。 | [cli.py][v1-cli]、[config.py][v1-config] | [M5](MOTIVATION.md#m5) |
| V1-02 | 生成与最终验证独立选择后端 | **迁移整合。** 生成支持 local / Docker / Modal，验证支持 local / Docker / Modal / HTTP 队列；例如 CPU 容器生成、远端 GPU 验证。不同后端共用参考源码、评测设置与结果协议。 | [engine.py][v1-engine]、[transports.py][v1-transports]、[modal_backend.py][v1-modal] | [M5](MOTIVATION.md#m5) |
| V1-03 | 题目范围与多样本批量生成 | **迁移整合。** 支持 level 1–3、闭区间题号和每题多次独立 agent 尝试；每次尝试有独立工作目录、日志及内核文件。 | [config.py][v1-config]、[generation.py][v1-generation] | [M1](MOTIVATION.md#m1) |
| V1-04 | 多 GPU 调度与并发限制 | **迁移整合。** 本地 GPU 生成按选定设备限制并发；最终评测在同一 GPU 上串行、不同 GPU 间并行；远端队列可限制在途任务数。 | [generation.py][v1-generation]、[engine.py][v1-engine]、[transports.py][v1-transports] | [M5](MOTIVATION.md#m5) |
| V1-05 | Docker 生成与验证 | **迁移整合。** 统一 Dockerfile，提供带 CUDA 工具链的 `agent` 和用于远端验证的 CPU 生成目标 `agent-remote`；镜像包含当前 checkout 的题目、隐藏测试与评测代码。 | [Dockerfile][v1-docker]、[generation.py][v1-generation]、[transports.py][v1-transports] | [M5](MOTIVATION.md#m5) |
| V1-06 | Slurm 集群提交 | **迁移整合。** 提供批量作业脚本，使用提交目录、可配置 Python 和分配到的 `CUDA_VISIBLE_DEVICES`；通过环境变量指定题目、设备与后端。 | [batch_run.sbatch][v1-slurm]、[transports.py][v1-transports] | [M5](MOTIVATION.md#m5) |
| V1-07 | Modal 云端执行 | **迁移整合。** 支持云端最终验证及 GPU Sandbox 内的生成、迭代验证；退出 Sandbox 前收集内核和日志，使用同一套 Verified 评测入口。 | [modal_backend.py][v1-modal]、[batch_run_modal.sh][v1-modal-sh] | [M5](MOTIVATION.md#m5) |
| V1-08 | 持久化 HTTP 验证队列 | **迁移整合与可靠性增强。** HTTP 提交/查询、SQLite 保存任务与结果、重启后重新排队未完成任务；相同请求幂等、瞬时 HTTP 错误重试，并限制待处理任务数量。 | [queue_worker.py][v1-queue]、[transports.py][v1-transports] | [M5](MOTIVATION.md#m5) |
| V1-09 | 带内容指纹的断点恢复 | **流程增强。** 评测恢复核对参考、内核、评测器、隐藏测试和配置；生成恢复核对模型、prompt 与输出校验和。修改内核后重新执行对应的标准及隐藏评测，旧格式结果重新验证。 | [common.py][v1-common]、[engine.py][v1-engine]、[generation.py][v1-generation] | [M5](MOTIVATION.md#m5) |
| V1-10 | 批量任务的超时与失败处理 | **流程增强。** 单次验证使用独立子进程与编译目录；超时终止进程组，Docker 验证同时清理容器；缺失内核写入失败记录，失败的重新生成不会继续使用旧内核冒充新结果。 | [transports.py][v1-transports]、[generation.py][v1-generation]、[engine.py][v1-engine] | [M5](MOTIVATION.md#m5) |
| V1-11 | 将 Verified 隐藏门接入全部验证后端 | **评测整合。** 默认先做标准正确性、CUDA-event 计时和峰值显存，再做隐藏正确性；最终正确性要求两者通过，隐藏结果缺失也不算通过。耗时和显存保留标准评测值。 | [evaluate_task.py][v1-task]、[engine.py][v1-engine]、[benchmark_eval_analysis.py][v1-analysis] | [M2](MOTIVATION.md#m2) |
| V1-12 | 基线记录接入批量流程 | **流程增强。** 可在选定验证后端记录指定题目基线；完整流程自动补测缺失基线。FP32 默认 TF32，FP16/BF16 分开存储，也可显式选择 `torch.compile` 基线；记录耗时、显存与环境信息。 | [evaluate_task.py][v1-task]、[engine.py][v1-engine]、[config.py][v1-config] | [M3](MOTIVATION.md#m3) |
| V1-13 | 子集分析、pass@k 与统一结果归档 | **流程增强。** 批量分析使用选定题目作为分母；pass@k 使用声明样本数，缺失和隐藏失败样本计失败；输出分析摘要、逐样本耗时/显存对比，并保持 Verified 结果键格式与原有指标公式。 | [engine.py][v1-engine]、[benchmark_eval_analysis.py][v1-analysis] | [M3](MOTIVATION.md#m3) |
| V1-14 | 旧入口兼容与配置预览 | **流程增强。** 旧 shell / Python 入口转接统一配置，兼容常用 pydra 参数；`--dry-run` 预览配置与题目，不创建运行目录、不调用模型或云服务。 | [legacy.py][v1-legacy]、[cli.py][v1-cli]、[batch_run.sh][v1-shell] | [M5](MOTIVATION.md#m5) |

### Verified 继承能力与本版行为

TF32 基线、四分布隐藏测试、显存效率指标和特定题目的 input-blind 策略来自 Verified。本版的贡献是把它们接入上述批量流程，统一后端的配置、调用与输出。

本版遵循上游 input-blind 题目列表：相应题目的生成材料去掉测试输入，隐藏测试不放进 prompt。其他题目仍按本版原有策略提供源码。工作目录和 prompt 管理不构成阻止 agent 访问文件的安全沙箱。

本版关键配置如下，便于与后续版本对照：

| 配置或行为 | 第一版取值 |
| --- | --- |
| 默认生成 / 验证后端 | Docker / local |
| 默认隐藏测试 | 开启 |
| 标准性能计时 | CUDA event |
| 默认 FP32 基线名 | `baseline_time_torch_tf32` |
| 可选编译基线 | 通过 `--compile-backend`、`--compile-mode` 显式配置 |
| 默认 pass@k | 在样本数足够时计算 k=1、5 |
| 恢复失败结果 | 指纹匹配时保留已记录失败；重试需修改源码/设置或关闭 resume |

六类根因检查、四变体基线组合、同步墙钟计时、H1–H14、独立最终重测和新的 K 次摊销评分均不属于本版清单。它们在核对时仍是本地未提交改动，后续应另列版本。

### 结果放在哪里

| 路径 | 内容 |
| --- | --- |
| `runs/NAME/run_config.json` | 本次运行配置 |
| `runs/NAME/generation_summary.json` | 各次生成状态与校验信息 |
| `runs/NAME/level_L_problem_P_sample_S_kernel.py` | 生成的内核 |
| `runs/NAME/eval_results.json` | 标准正确性、耗时与显存 |
| `runs/NAME/eval_results_hidden.json` | 隐藏正确性 |
| `runs/NAME/pass_at_k_results.json` | 隐藏门约束后的 pass@k |
| `runs/NAME/analysis_summary.json` | Verified 指标与逐题比较 |
| `runs/NAME/work/pP_sS/`、`runs/NAME/logs/` | agent 工作目录及执行日志 |
| `results/timing/HARDWARE/BASELINE.json` | 对应硬件标签下的基线 |

### 验证程度

本版提交说明与随提交保存的 [工作流文档][v1-doc] 记录：**17 项工作流测试 + 4 项上游 dataset/score 测试，共 21 项通过**。测试使用临时 Python 3.10 / Torch 2.9.0 环境。测试源码见 [test_workflows.py][v1-tests]。

| 检查范围 | 本版记录的验证情况 |
| --- | --- |
| 配置、题目范围、GPU 索引与旧参数兼容 | CPU 测试覆盖 |
| HTTP 队列、持久化、幂等与重启恢复 | 真实 HTTP 服务配合模拟 GPU 评测器验证 |
| 隐藏门、缺失样本、pass@k、子集分母与显存字段 | CPU / 模拟任务测试覆盖 |
| 内核变化后的恢复、生成失败后的旧文件处理、超时与 dry-run | 工作流测试覆盖 |
| 项目固定 Torch 2.12.0 | 依赖解析通过；运行验证仍待完成 |
| 两个 Docker 目标 | 构建定义检查通过；未证明实际镜像构建和 GPU 运行成功 |
| Modal | SDK 镜像定义构造通过；实际云端执行仍待完成 |
| Slurm、真实 CUDA 内核与模型生成 | GPU 申请因节点忙未成功；完整执行仍待完成 |

这里记录的是该提交已有的验证结果。本次维护清单没有重跑历史测试，也不据此宣称性能提升或全部后端已经实机跑通。

## 第二版：正确性判断与输入隔离

| 项目 | 版本信息 |
| --- | --- |
| 提交标题 | Fix output comparison and isolate correctness inputs |
| 提交日期 | 2026-10-20（America/New_York） |
| 基础提交 | `2bce940c631c01086c262eef9c9efc9d3693e1de` |
| 本版提交 | 实现、回归测试与本节文档在同一提交；哈希可用下方命令查询 |
| 状态 | 已提交；CPU 回归通过；真实 CUDA 验证待完成 |

```bash
git log -1 --format='%h %s' --grep='^Fix output comparison and isolate correctness inputs$'
```

本版统一标准评测和隐藏评测的输出判断，防止离散结果被浮点容差放过，以及输入或输出缓冲区被修改后污染比较。

| 编号 | 功能 | 具体变化 | 实现入口 | Motivation |
| --- | --- | --- | --- | --- |
| V2-01 | 严格比较输出结构、shape 和 dtype | 递归检查 Tensor、dict、list、tuple；核对字典键、序列类型和长度，以及各 Tensor 的 shape、dtype；记录具体不一致的输出路径。 | [validation.py](../src/validation.py)：`compare_outputs` | [M2](MOTIVATION.md#m2) |
| V2-02 | 整数与 bool 精确比较 | 离散 Tensor 使用精确比较。比如整数索引 `100000` 与 `100001` 即使满足相对浮点容差，也会被判为不一致；整数和 bool 输入不再被统一转换为浮点精度。 | [validation.py](../src/validation.py)、[eval.py](../src/eval.py)：`_process_input_tensor` | [M2](MOTIVATION.md#m2) |
| V2-03 | 核对 NaN、Inf 的位置与符号 | NaN、正 Inf、负 Inf 的掩码分别匹配；匹配后，仅有限数值继续按原 FP32/FP16/BF16 容差比较。 | [validation.py](../src/validation.py)：`compare_outputs` | [M2](MOTIVATION.md#m2) |
| V2-04 | 隔离参考与候选输入 | 两个模型分别使用独立的 Tensor 副本，支持嵌套容器，避免参考或候选的原地操作修改另一方的输入和调用方原始数据。 | [validation.py](../src/validation.py)：`clone_tensors`；[eval.py](../src/eval.py)：两个正确性入口 | [M2](MOTIVATION.md#m2) |
| V2-05 | 保存参考输出快照 | 在调用候选前保存参考结果，避免共享或复用输出缓冲区把参考答案覆盖，导致错误实现被判通过。 | [eval.py](../src/eval.py)：`run_and_check_correctness`、`run_correctness_on_fixed_inputs` | [M2](MOTIVATION.md#m2) |
| V2-06 | 对齐 forward 随机种子 | 参考与候选 forward 使用相同种子，避免 dropout 等随机计算因执行先后产生不同结果。 | [eval.py](../src/eval.py)：两个正确性入口 | [M2](MOTIVATION.md#m2) |

### 验证程度

在以 `2bce940` 为基础、仅包含本版代码的独立快照中执行：

```bash
python -m pytest -q tests/test_correctness_validation.py tests/test_workflows.py \
  src/unit_tests/test_dataset.py src/unit_tests/test_score.py
```

结果为 **64 passed**：43 项新增正确性回归用例，加上 21 项原有工作流与 dataset/score 测试。环境为 Python 3.10 / Torch 2.9.0。

新增用例覆盖嵌套输出、结构/shape/dtype 不匹配、整数偏差、bool、非有限值位置与符号、原浮点容差、原地操作、共享输出缓冲区、嵌套输入副本及随机种子。Ruff 和 diff 格式检查通过。测试源码为 [test_correctness_validation.py](../tests/test_correctness_validation.py)。

真实 CUDA 执行和项目固定 Torch 2.12.0 的运行验证仍待完成。本版没有加入根因探针、反作弊、墙钟计时、四变体基线或新计分流程；参考报错和空隐藏测试集的处理规则也未在本版修改。

本节实现入口的相对链接会打开当前 checkout；查看本版准确源码时，使用上述命令取得提交哈希，再执行 `git show <hash>:src/eval.py` 或对应文件路径。

## 第三版：基线与计时

| 项目 | 版本信息 |
| --- | --- |
| 提交标题 | Add validated baseline portfolio and synchronized wall timing |
| 本版提交哈希 | `d4eed06` |
| 基础提交 | `d488df8` |
| 本版提交 | 与本节首次加入同一提交；可用下方命令查询 |
| 状态 | 已提交；CPU/模拟 CUDA 回归通过；真实 GPU 工具链待验证 |

```bash
git log -1 --format='%h %s' --grep='^Add validated baseline portfolio and synchronized wall timing$'
```

| 编号 | 功能 | 具体变化 | 实现入口 | Motivation |
| --- | --- | --- | --- | --- |
| V3-01 | 默认四变体基线 | eager、compile default、compile max-autotune、CUDA Graph 各自独立任务运行；验证初次及每次计时输出，保留失败变体，在正确变体中选最优。组合的 FP32 使用 IEEE；显式单基线仍可使用 TF32。 | `baseline_variants.py`、`evaluate_task.py`、`engine.py` | [M3](MOTIVATION.md#m3) |
| V3-02 | 同步墙钟计时 | 调用前后同步，计入 Python 调度、调用、输出消费及 GPU 完成等待；输入准备和完整参考校验在计时外。Graph 的输入复制和输出克隆计入 replay 耗时。 | `measurement.py`、`src/eval.py` | [M2](MOTIVATION.md#m2) |
| V3-03 | 变化输入与逐次输出验证 | 批量基线和候选每次重新调用可信输入生成器，记录新种子；保留输入副本，每次检查输出，失败不作为有效性能结果。固定生成器仍可能产生相同内容。 | `measurement.py`、`evaluate_task.py`、`src/eval.py` | [M2](MOTIVATION.md#m2) |
| V3-04 | 冷成本与 K 次摊销 | 记录构建/初始化、首次调用和预热；按 `(cold + (K-1) × steady) / K` 报告平均成本，默认 K=10000，同时保留稳态最优与摊销最优基线。 | `measurement.py`、`baseline_variants.py`、`config.py` | [M3](MOTIVATION.md#m3) |
| V3-05 | 原始样本和测量信息 | 保存毫秒样本、均值/标准差及正态近似 95% 均值区间；记录真实设备、软件版本、输入形状/stride/dtype 和源码指纹，供后续兼容性检查。 | `measurement.py`、`evaluate_task.py` | [M3](MOTIVATION.md#m3) |
| V3-06 | 区分旧计时与新结果 | 新默认文件名为 `baseline_time_suite_fp32_wall`；`--no-baseline-suite` 可选单基线，编译单基线需同时指定该开关；基线组合未完整记录时重新执行。 | `config.py`、`cli.py`、`engine.py` | [M3](MOTIVATION.md#m3) |

验证：Python 3.10 / Torch 2.9.0 下，**84 项测试通过**，含 20 项新增基线/计时回归及此前 64 项测试。新增测试为 [test_baseline_timing.py](../tests/test_baseline_timing.py)，包含真实 CPU 公式、模拟异步完成与 Graph replay、原评测入口的计时校验接入。Ruff、目标模块编译和 diff 检查通过。

范围：本版记录摊销成本，不改变聚合评分；评分与恢复另列下一版。H1–H14、FP64 精度门、sanitizer、根因探针、独立最终重测继续暂缓。冷成本不清空编译缓存，未计进程启动和输入准备；不代表完整应用端到端成本。真实 CUDA、torch.compile、Graph 和 Torch 2.12.0 尚待部署验证。旧名 `time_execution_with_cuda_event` 保留为墙钟计时兼容入口；仅原生批量入口提供变化输入及逐次验证。

## 第四版：评分与恢复

| 项目 | 版本信息 |
| --- | --- |
| 提交标题 | Add failure-inclusive scoring and evidence-aware recovery |
| 基础提交 | `d4eed06` |
| 本版提交 | 与本节首次加入同一提交；可用下方命令查询 |
| 状态 | 已提交；CPU/持久化结果回归通过；真实 GPU 与完整反作弊实验待完成 |

```bash
git log -1 --format='%h %s' --grep='^Add failure-inclusive scoring and evidence-aware recovery$'
```

| 编号 | 功能 | 具体变化 | 实现入口 | Motivation |
| --- | --- | --- | --- | --- |
| V4-01 | 失败和缺失样本计入分母 | 每个请求题目 × 声明样本数形成固定槽位；失败、超时、缺失、过期或比较证据不足的槽位记 0。先按题取所有样本均值，再按层宏平均，不选择最快样本作为新评分。 | `paper_analysis.py`：`analyze_paper` | [M3](MOTIVATION.md#m3) |
| V4-02 | 按 K 摊销成本评分 | 在有当前证据的变体中，按声明 K 重新选择摊销最优基线；评分为 `clip(log(T_ref/T_cand)/log(1.25), 0, 1)`，同时要求所选基线与候选的稳态均值区间不重叠。保留稳态加速比和摊销加速比。 | `paper_analysis.py`、`measurement.py` | [M3](MOTIVATION.md#m3) |
| V4-03 | 兼容性和测量证据检查 | 检查设备类别、Torch/CUDA/HIP、精度/容差、计时协议、输入签名、源码与任务指纹；要求变化输入、逐次校验收据、有效原始样本和冷成本。组合基线必须记录完整，具体变体也须匹配当前任务。 | `paper_analysis.py`：`compatible`、`measurement_evidence`、`current_result` | [M3](MOTIVATION.md#m3) |
| V4-04 | 分层统计与区间 | `--problem-specs` 可声明 I/F/D 层，未声明为 `unclassified`；保留失败槽位，输出按题和样本两层重采样的 bootstrap 区间。该版本不加入设备校准或 headroom 评分。 | `config.py`、`paper_analysis.py` | [M3](MOTIVATION.md#m3) |
| V4-05 | 扩大恢复指纹覆盖 | 指纹覆盖共享验证、计时、基线变体、配置、任务编排、传输和分析模块；版本或源码变化使旧任务失效。完整流程发现过期基线时重新记录，客户端与 worker 版本不匹配则拒绝执行。 | `common.py`、`cli.py`、`engine.py`、`evaluate_task.py` | [M5](MOTIVATION.md#m5) |
| V4-06 | 旧分析与榜单不能绕过证据检查 | 新批量结果经 `eval_results_scored.json` 提供资格；读取时重新核对当前内核、参考题目、隐藏源码、版本、运行配置及原始结果/基线内容。缺失或过期资格拒绝计分；隐藏合并不会把标准失败覆盖为通过。 | `common.py`：`apply_score_gate`；两个分析/榜单脚本 | [M3](MOTIVATION.md#m3) |

新增输出为 `paper_analysis.json` 与 `eval_results_scored.json`。前者记录固定分母评分、区间、原始与摊销加速比及逐项拒绝原因；后者保存计分资格并保留 `raw_correctness`。原 Verified 指标继续使用原有公式，只读取资格检查后的结果；pass@k 仍表示正确性统计，含义与新性能评分不同。

验证：Python 3.10 / Torch 2.9.0 下，**114 项测试通过**，含 30 项新增评分/恢复回归及第三版的 84 项测试。新增测试为 [test_scoring_recovery.py](../tests/test_scoring_recovery.py)，覆盖失败分母、置信区间、冷成本、兼容性、证据失效、旧分析接入、指纹变化重跑、过期基线重测、worker 版本拒绝和子验证器配置传递。Ruff 对工作流/共享验证/测试模块、目标模块编译与 diff 检查通过。

这里的资格只证明本版要求的测量与兼容性证据齐备，报告明确写 `anti_cheat_verified=false`、`cross_device_comparison_certified=false`；不等于 H1–H14 通过。没有完整反作弊模块时，不接受标为反作弊评测的结果获得资格。置信区间为近似统计，冷成本为一次观测，不构成正式的性能或安全证明。真实 CUDA、Torch 2.12.0、跨设备实验、根因探针、sanitizer 和独立最终重测继续暂缓。

## 第五版：参考相关工作的可靠性检查

| 项目 | 版本信息 |
| --- | --- |
| 提交标题 | Strengthen correctness checks from kernel benchmark research |
| 日期 | 2026-10-21 |
| 基础提交 | `81c7055` |
| 本版提交 | 实现、测试、来源说明和本节在同一提交；用下方命令定位 |
| 状态 | 已提交；172 项 CPU/模拟测试通过；真实 GPU 验证尚未完成 |

```bash
git log -1 --format='%h %s' --grep='^Strengthen correctness checks from kernel benchmark research$'
```

本次核对了 KernelBench-Hard、KernelBenchX、robust-kbench 的论文/源码及 TritonRL 论文，按当前题目与工作流独立适配。各项来源、固定上游 commit、采用与未采用部分见 [可靠性说明](RELIABILITY.md)。

| 编号 | 功能 | 具体变化与来源 | 实现入口 | Motivation |
| --- | --- | --- | --- | --- |
| V5-01 | 参考前后配对检查 | 借鉴 robust-kbench 的前后调用，标准与隐藏检查均执行 C→R、R→C 两组，两个模型调用次数一致；同步、重放 RNG、输入隔离与输出快照防止当前参考缓存和共享缓冲污染。 | `src/reliability.py`、`src/eval.py` | [M2](MOTIVATION.md#m2) |
| V5-02 | 隔离源码全局字典与后端设置 | 候选同名变量/函数不能覆盖参考定义；每次 forward 后恢复共享后端设置，防止候选改变精度设置影响参考比较。属于本地配套实现，不声称提供安全沙箱。 | `src/eval.py`、`src/reliability.py` | [M2](MOTIVATION.md#m2) |
| V5-03 | 三个种子重新生成隐藏输入 | 借鉴 Hard 的三种子验证，分别在 42、123、456 下调用可信隐藏生成器，全部配置必须通过。硬编码生成器仍可能返回相同内容。 | `src/eval.py`：`run_hidden_correctness_check` | [M2](MOTIVATION.md#m2) |
| V5-04 | 按输入域限定的数值压力测试 | 借鉴 Hard 的按题压力输入及 KernelBenchX 的 0.1%/×50 离群分布。29 道 Level 1 题的首个隐藏配置增加零值、×32 和稀疏离群值；不改变 shape/stride/dtype 或其他输入，不把扰动施加到未经核对的受限域。 | `src/reliability.py`：`STRESS_TASKS`、`stress_cases` | [M2](MOTIVATION.md#m2) |
| V5-05 | 无效验证不能算通过 | 空、缺失、格式错误的隐藏集显式报错；参考报错记录来源并判验证失败；标准零次测试不能通过。是落实“所有配置必须实际验证”的本地修补。 | `src/eval.py`、`src/reliability.py` | [M2](MOTIVATION.md#m2) |
| V5-06 | 分阶段诊断与逐项收据 | 参考 KernelBenchX/TritonRL 对编译、正确性、性能的区分，新增 `validation_summary.json`，失败/缺失样本保留分母；metadata 保存种子、配置、调用顺序和失败来源。阶段诊断不代替第四版的计分资格，也不证明功能合法性。 | `kb_workflows/engine.py`、`src/reliability.py` | [M3](MOTIVATION.md#m3) |
| V5-07 | 新协议参与恢复指纹 | 新可靠性模块纳入 evaluator digest，使旧缓存不能继续冒充新协议结果；沿用既有 worker 版本核对和过期证据拒绝。 | `kb_workflows/common.py` | [M5](MOTIVATION.md#m5) |

验证：Python 3.10 / Torch 2.9.0 下 **172 passed**，其中新增 [test_reliability.py](../tests/test_reliability.py) 的 58 项，原有 114 项全部通过。覆盖全部 29 道适配题的小形状真实 CPU 参考执行，以及缓存答案、第二次调用错误、无效隐藏集、非法域排除、源码全局覆盖和恢复指纹回归。Ruff 与 diff 检查通过。

范围与成本：原四配置隐藏集变为每题 12 次配置检查；适配题增加压力输入后为 21 次，每配置两组配对。额外检查不进入稳态计时；首次候选 forward 提前到首次参考 forward 前，冷成本仍记录。Level 2/3 本次没有新增数值扰动。保留当前浮点容差与匹配 NaN/Inf 规则，没有迁入 Hard/X 题集、SOTA 外部库、TFLOPS/带宽评分、模型裁判或 backward 验证，不宣称完整反作弊通过。

本次 Slurm GPU smoke 作业 50702 因 `Requested nodes are busy` 未能启动；真实 CUDA 和固定 Torch 2.12.0 尚未验证。

原先未提交的整合与清理保存在 stash `ba76365ce0dd755bce1361e38420bb5f4ec3e196`，本次没有恢复。以后恢复需处理与本版重叠的正确性入口，并重新验证。

## 第六版：KernelBench Pro、Harbor 与 Python 3.12

| 项目 | 版本信息 |
| --- | --- |
| 提交标题 | Unify KernelBench Pro with Harbor and Python 3.12 |
| 定位命令 | `git log -1 --format='%H %s' --grep='^Unify KernelBench Pro with Harbor and Python 3.12$'` |
| 基础提交 | `66fd65c5272c72572f0144989e97da1e98778771` |
| 记录日期 | 2026-10-21 |
| 状态 | 已提交；CPU/模拟回归通过，真实 GPU 与 agent 端到端验证待兼容环境 |

### 仓库定位与文件清理

项目名称统一为 **KernelBench Pro**，包、镜像、Modal 与 Slurm 标识使用 `kernelbench-pro`。现有 checkout 路径、题库目录和 GitHub 地址保留。

本版以 `66fd65c` 为基础，把仓库清理、更名、Harbor 接入和 Python 3.12 升级纳入同一次提交。此前搁置的整合改动仍在 stash `ba76365ce0dd755bce1361e38420bb5f4ec3e196`，没有恢复。

| 项目 | 当前变化 | Motivation |
| --- | --- | --- |
| 论文与旧实验材料 | 删除 1 份上游报告 PDF、3 份 LaTeX 稿、1 份参考文献库、9 份图表、1 个论文绘图脚本和 5 份旧实验分析文档，共 20 个文件。 | [M5](MOTIVATION.md#m5) |
| 重复生成材料 | 删除 4 份未被运行流程读取的 `stripped_prompts` 快照；输入盲化仍由可信配置动态生成。 | [M5](MOTIVATION.md#m5) |
| 仓库入口 | README 改为本仓库的安装、能力、运行与文档入口；移除上游作者名单、论文宣传、旧榜单图与论文结论。 | [M5](MOTIVATION.md#m5) |
| 维护方式 | 贡献与行为指南改为本仓库的 GitHub 流程，移除 Meta 内部导入、CLA 和举报渠道；LICENSE 和继承版权头保留，来源集中记录在 `NOTICE.md`。 | [M5](MOTIVATION.md#m5) |
| 项目名称 | Python distribution 为 `kernelbench-pro`，默认镜像为 `kernelbench-pro:agent`，Modal app 为 `kernelbench-pro`，Slurm job 为 `kernelbench-pro`；CLI、生成 prompt、worker 与结果页面使用 KernelBench Pro 名称。 | [M5](MOTIVATION.md#m5) |
| 结果页面 | 硬件标签取自本次数据；不再写死 H200、Best-of-5 或 2026 年 5 月，也不替未标注的运行猜测推理强度。 | [M3](MOTIVATION.md#m3) |
| 忽略规则 | 按目录忽略运行产物、缓存与生成预览，补充凭据和 Slurm 日志规则；取消对全部 JSON、CUDA/C++ 源码及部分个人脚本名的全局忽略。 | [M5](MOTIVATION.md#m5) |
| 使用中的文件 | 题库、隐藏测试、动态 prompt 示例和兼容脚本保留；`paper_analysis.py` / `paper_analysis.json` 是现有评分接口，保持名称以兼容结果读取。 | [M5](MOTIVATION.md#m5) |

验证：172 项 CPU/模拟回归和 Ruff 通过；CLI dry-run、实际 HTML 生成、Python 包元数据、脚本语法、文档链接与忽略规则检查通过。270 个参考题目和 250 个隐藏测试文件完整保留，LICENSE、核心比较/计时/评分实现没有改动。此轮不恢复搁置的评测完整性整合，也不重写 Git 历史；删除的研究材料仍可从历史提交恢复。

### Harbor 统一接入与 Python 3.12

本轮扩展的是第一版建立的统一工作流，沿用同一个 `RunConfig`、题目范围、样本槽位、结果文件、GPU 调度、正式验证和分析；不修改第一版历史提交的能力描述。此前搁置的整合改动仍在原 stash 中。

| 项目 | 当前变化 | Motivation |
| --- | --- | --- |
| 统一生成后端 | `--generation harbor` 调用 Harbor，每个原生样本对应一个 job/attempt；提交文件回收到原内核命名位置，再由原生正式评测和评分处理。 | [M5](MOTIVATION.md#m5) |
| 统一任务导出 | `--stage export-harbor` 使用同一配置按题号/样本导出 schema 1.4 任务；保留当前输入盲化策略，使用新目录避免覆盖。 | [M5](MOTIVATION.md#m5) |
| 验证与 reward | 独立 verifier 镜像保存可信代码及隐藏测试，只传递 `solution.py`；复用原生基线、正确性与证据评分，异常 reward 为 0。默认 correctness 与可选 bounded_gain 分开定义。 | [M2](MOTIVATION.md#m2) |
| GPU 与恢复 | Docker 任务绑定原生 GPU lane，单卡串行；Harbor agent/环境/模型/协议纳入恢复指纹，正式评测不直接复用 Harbor reward。AWS profile 只解析为运行时凭据。 | [M5](MOTIVATION.md#m5) |
| Python 升级 | `requires-python ==3.12.*`、Ruff py312、Docker Python 3.12 / Ubuntu 24.04、Modal Python 3.12；增加可选 `harbor` extra，固定 Torch 2.12.0。 | [M5](MOTIVATION.md#m5) |
| 文档 | [Harbor 接入](HARBOR.md)、README 和批量文档同步更新；历史 Python 3.10 测试记录仍保留其真实环境。 | [M5](MOTIVATION.md#m5) |

验证：Python 3.12.12 / Torch 2.12.0 / 正式发布的 Harbor 0.24.0 下 **204 passed**，含 32 项 Harbor 回归；Ruff、依赖一致性、实际 Task 模型校验和 CLI 配置解析通过。两个 Docker 目标的构建定义检查通过。Harbor Docker CPU smoke 因宿主 NAT 网络错误未启动，没有产物或 reward，不算通过；Torch CUDA 13 构建报告当前宿主驱动过旧，真实 GPU 与 agent 端到端验证仍待兼容环境。

## 第七版：GPU 配置与有界 Agent 工作流

| 项目 | 版本信息 |
| --- | --- |
| 提交标题 | Add configurable GPU targets and bounded mini-swe-agent workflows |
| 提交定位 | `git log -1 --format='%H %s' --grep='^Add configurable GPU targets and bounded mini-swe-agent workflows$'` |
| 基础提交 | `d2550fb60a36e2555635072a8eab6a1411392504` |
| 记录日期 | 2026-10-21 |
| 功能状态 | 已提交；270 项 CPU/模拟回归通过，真实 GPU 与付费模型端到端验证待完成 |

本版把 GPU 文件配置、one-shot / mini-swe-agent 反馈循环和 coding-agent skill 纳入同一次提交。此前搁置的评测完整性整合及清理草稿仍保留在原 stash，没有恢复或纳入本版。

### GPU 配置统一

| 功能 | 相对第六版的变化 | 实现入口 | Motivation |
| --- | --- | --- | --- |
| 文件配置 | 新增 `--config FILE.toml`，`[gpu]` 管 GPU，`[run]` 管工作流；命令行显式覆盖文件，文件覆盖默认值；检查类型、未知字段及枚举值。 | `kb_workflows/config.py`、[GPU 配置](GPU_CONFIG.md) | [M5](MOTIVATION.md#m5) |
| GPU 预设 | 型号到架构和云端类型的映射移出 Python，集中维护在随包分发的 TOML；提供 A6000 本地和 H100 Modal 示例。 | `kb_workflows/gpu_profiles.toml`、`configs/`、`pyproject.toml` | [M5](MOTIVATION.md#m5) |
| 自定义目标 | `[gpu]` 可直接填写硬件标签、架构、设备和云端类型；编译目标接受架构名或计算能力，不再依赖旧的固定架构白名单。 | `kb_workflows/gpu.py`、`src/utils.py` | [M5](MOTIVATION.md#m5) |
| 明确选择硬件 | 去掉默认 H100 和未知型号回退 Hopper；未知硬件需显式架构，实际 GPU 任务要求完整目标，Modal 要求云端类型。更换 CLI 目标时清除文件中旧的目标配置。 | `kb_workflows/config.py`、`kb_workflows/cli.py` | [M3](MOTIVATION.md#m3) |
| 后端与子任务 | 同一解析结果传给现有生成、评测、基线、调度和恢复流程；子任务保存展开后的 JSON，不依赖宿主 TOML 路径；Harbor 省略空云端类型。 | `kb_workflows/harbor.py`、原生 `RunConfig` 及任务协议 | [M5](MOTIVATION.md#m5) |
| 集群与恢复 | Slurm 使用 `KB_CONFIG`，只让显式环境变量覆盖文件；shell 入口尊重文件后端；GPU 解析模块和预设参与 evaluator digest，切换目标影响恢复指纹。 | `batch_run.sbatch`、`batch_run.sh`、`batch_run_modal.sh`、`kb_workflows/common.py` | [M5](MOTIVATION.md#m5) |

GPU 配置阶段验证：Python 3.12.12 / Torch 2.12.0 / Harbor 0.24.0 下 **244 passed**，新增 40 项配置与边界回归；Ruff、脚本语法和 diff 检查通过。离线 wheel 构建成功，解包后在仓库外验证 GPU 预设可加载；配置预览不导入 Torch、Modal 或 Harbor，Harbor 任务通过实际 Task 模型校验。真实 GPU 与 agent 端到端验证仍受宿主驱动限制，未新增实机通过声明。

范围：一个运行配置描述一种 GPU 类别，异构卡使用独立运行和基线；执行后端仍为 NVIDIA CUDA，不代表 AMD、Intel、MPS 或 CPU 性能评测已适配。

### One-shot、mini-swe-agent 与 coding-agent skill

与前述 GPU 配置统一改动一起提交。采用 mini-swe-agent 发布版 2.4.6，参考 TIRx 的固定评测契约、诊断与基准分离设计；没有迁入 TIRx 编译器、NumSim、Synccheck、Racecheck 或 KCoral。此前 stash 仍未恢复。

| 功能 | 相对第六版的变化 | 实现入口 | Motivation |
| --- | --- | --- | --- |
| 真实 agent loop | 使用上游 DefaultAgent 的查询/反馈循环、线性消息历史、步数/费用检查与轨迹序列化；自有 Model/Environment 适配器提交源码，控制器负责执行反馈。 | `kb_workflows/agent_loop.py`、`kb_workflows/agent_models.py` | [M4](MOTIVATION.md#m4) |
| One-shot | 一个样本只发起一次模型适配器调用，不给执行反馈；Claude Code 内置工具、MCP 和自定义加载关闭。多样本采样与单次生成分别记录。 | `configs/a6000-one-shot.toml`、`generation_mode` | [M4](MOTIVATION.md#m4) |
| 有界反馈 | 默认每样本尝试最多 5 次候选、2400 秒；CLI/TOML 可调。控制器保留开发正确且平均耗时最低的候选，后续错误或超时不覆盖已完成最佳版本。 | `configs/a6000-feedback.toml`、`kb_workflows/generation.py` | [M4](MOTIVATION.md#m4) |
| 开发与最终评测分离 | 开发任务明确关闭隐藏测试；冻结源码后原生流程重新做标准与多种子隐藏验证。新增独立 Torch profiler 诊断子任务，诊断时间不进入正式评分，也不冒充 NCU 硬件计数器。 | `kb_workflows/agent_loop.py`、`kb_workflows/profile_task.py`、`kb_workflows/evaluate_task.py` | [M4](MOTIVATION.md#m4) |
| 资源与来源记录 | 记录调用、候选、开发/诊断次数、可观测 token 和估算费用；未知值不当作零。源码 hash 绑定生成回执；已完成失败可恢复而不自动重试，显式重复尝试累计历史成本并标为 repeated-attempts。 | `generation_budget.json`、`generation_attempts.json`、`phase_budget.json`、`kb_workflows/experiment.py` | [M4](MOTIVATION.md#m4) |
| 模式比较 | 按模式、框架、预算、采样数、题集、硬件和精度分组，模型信息逐行记录；不跨不同实验组直接排名。生成后修改源码标为 unattributed。 | `python -m kb_workflows.experiment`、`paper_analysis.json` | [M4](MOTIVATION.md#m4) |
| 模型与后端 | Claude Code / LiteLLM 模型入口共用 mini-swe-agent；local / Docker / Modal 生成接入同一循环；队列支持诊断任务。Harbor 保持外部 agent 模式，不冒充原生 one-shot 或受控轮数。 | `requirements.txt`、`kb_workflows/modal_backend.py`、`kb_workflows/queue_worker.py` | [M4](MOTIVATION.md#m4) |
| Coding-agent skill | Claude Code 的 `/kernelbench-pro` 与 `.agents/skills` 共享 skill。当前会话可准备任务、提交标准开发检查、冻结最佳源码并执行最终评测，不递归启动 Claude CLI；模型调用不可观测时单列 interactive。 | [.claude/skills/kernelbench-pro/SKILL.md](../.claude/skills/kernelbench-pro/SKILL.md)、`kb_workflows/agent_task.py` | [M4](MOTIVATION.md#m4) |

验证：Python 3.12.12 / Torch 2.12.0 / Harbor 0.24.0 / mini-swe-agent 2.4.6 下，**270 项完整 CPU/模拟回归通过**。覆盖真实 DefaultAgent、伪模型 CLI 的完整 worker/多样本/恢复路径、LiteLLM SDK 适配、未知费用、费用超限、保留正确候选、诊断反馈、标准/隐藏分离、重试成本、来源归属和嵌套子进程超时清理。Ruff、168 个包的依赖一致性及 skill 校验通过；离线 wheel 构建后，在仓库外核对适配模块、GPU 预设与轻量导入。测试没有调用付费模型，也没有执行真实 GPU 内核；当前宿主仍报告 NVIDIA 驱动不兼容 CUDA 13，实机端到端验证待兼容的已分配 GPU。

运行说明与方案比较见 [Agent loops](AGENT_LOOPS.md)。费用限制是调用间的客户端估算上限，单次可超限；GPU 计费时间保持未知，不宣称完整 OS 沙箱或反作弊认证。

## 第八版：Headroom 研究与 Motivation 统一

| 项目 | 版本信息 |
| --- | --- |
| 提交标题 | Integrate headroom research with motivation traceability and task suites |
| 提交定位 | `git log -1 --format='%H %s' --grep='^Integrate headroom research with motivation traceability and task suites$'` |
| 基础提交 | `92c27f4c020a4860d5ca546b88a50b007f629e3f` |
| 日期 | 2026-10-21 |
| 状态 | 已提交；验证结果见本节末尾 |

本版择取用户提供的 `headroom-bench-20261021.zip`，保留研究证据、固定选集与未完成设计，按当前实现重写研究稿；不导入旧论文备份、内部对话或缺依赖的攻击/加固原型。此前 stash 没有恢复，核心正确性/计时检查没有被原型替换。

| 编号 | 功能 | 变化 | 实现入口 | Motivation |
| --- | --- | --- | --- | --- |
| V8-01 | 五条动机与逐项追踪 | 将历史改动归到具体问题，并校验覆盖完整性。 | `docs/MOTIVATION.md` | [M5](MOTIVATION.md#m5) |
| V8-02 | 统一论文与能力状态 | 已实现与研究目标使用同一套名称和预算，不将设计写成实验结果。 | `research/headroom/PAPER.md` | [M5](MOTIVATION.md#m5) |
| V8-03 | 研究证据归档与重建 | 保留逐实现审查依据与来源摘要，不执行缺依赖的原型。 | `scripts/headroom_evidence.py` | [M2](MOTIVATION.md#m2) |
| V8-04 | KernelBench-105 选集 | 接入105个scored、5个probe、140个extended的固定选集。 | `kb_workflows/suites/kernelbench105.json` | [M1](MOTIVATION.md#m1) |
| V8-05 | 全流程统一选题 | 生成、基线、隐藏验证、评分、预算与Harbor均使用同一子集。 | `kb_workflows/common.py` | [M1](MOTIVATION.md#m1) |
| V8-06 | 分层来源与探针隔离 | 启发式I/F/D标注来源；probe和extended分组且不标为主榜资格。 | `kb_workflows/paper_analysis.py` | [M3](MOTIVATION.md#m3) |
| V8-07 | 选集与参考版本证据 | 记录manifest摘要并检查参考快照，阻止静默任务漂移。 | `kb_workflows/suites.py` | [M5](MOTIVATION.md#m5) |

KernelBench-105 使用原选集的 105 scored / 5 probe / 140 extended，暂定 I/F/D 为 30/56/19。它是原始题目上的研究选集，未实现论文输入校准，也不把启发式 D 标签当作算法证书。默认仍跑完整题集；显式选择 `--suite kernelbench105`，范围与选集取交集，probe/extended 单独运行和报告。参考源码改变时需重新审阅清单。详细用法与证据限制见 [研究索引](../research/headroom/README.md)。

相对原稿，本版修正了“仅第一版实现”“默认60分钟”“已支持Trainium”“已有完整FP64/硬件计数器/Headroom评分”等过时或超前表述；当前默认仍为5候选/2400秒，计算后端仍为NVIDIA CUDA，评分仍为测量证据约束的bounded gain。完整未完成项见 Motivation 的23条差额，实验不因文档整合而记为完成。

验证：Python 3.12.12 / Torch 2.12.0 / Harbor 0.24.0 / mini-swe-agent 2.4.6 下 **287 项 CPU/模拟回归通过**，含本轮17项选集与追踪回归。覆盖三类角色与250题分区、范围交集、参考漂移拒绝、无GPU轻量预览、缺失样本分母、实际批量生成路径、基线/预算/子任务一致性、真实Harbor Task导出、层来源标记、证据摘要及全部改动的motivation覆盖。Ruff、168个包依赖一致性、来源重建逐文件比较、文档本地链接和离线wheel检查通过；wheel在仓库外可加载选集且配置不导入Torch。未运行真实GPU内核或付费模型，现有CUDA驱动兼容性限制仍保留。

## 第九版：严格数值契约与证据等级（第一批设计落地）

| 项目 | 版本信息 |
| --- | --- |
| 提交标题 | Implement strict numerical contracts and evidence-gated evaluation |
| 基础提交 | `7915423553f5e0cc385dd800a9497aef276d5936` |
| 日期 | 2026-10-21 |
| 状态 | 已提交；368项CPU/模拟测试通过，GPU校准与实机验证待完成 |

本版推进23项设计中的R-04/05/06/07/09，交付R-17聚合消融和R-22证据收据的代码。严格校验显式开启；默认既有协议保留。严格契约先覆盖17道L1（1–3、19–27、47–49、51–52），使用小规模附加实例，不改变原计时输入，也不声称原尺寸已通过FP64校准。完整差额仍在Motivation逐项记录。

| 编号 | 功能 | 变化 | 实现入口 | Motivation |
| --- | --- | --- | --- | --- |
| V9-01 | FP64数值契约 | 在17道已审阅L1的附加实例上，参考与候选均须通过RMS及全元素FP64门。 | `kb_workflows/numerical.py` | [M2](MOTIVATION.md#m2) |
| V9-02 | 参考信号诊断 | 记录输出量级、非有限值、常数与容差覆盖；至少一组合法实例有信息量。 | `kb_workflows/numerical.py` | [M2](MOTIVATION.md#m2) |
| V9-03 | 合法输入族与隐藏形状 | 限定17题的耦合维度、符号、近常数、动态范围、边界、GELU辨别输入及最终非连续stride。 | `kb_workflows/strict_validation.py` | [M2](MOTIVATION.md#m2) |
| V9-04 | 模式状态与同对象缓存检查 | 审计train/eval、同名buffer、参数原地更新与同对象输入修改后的响应。 | `kb_workflows/strict_validation.py` | [M2](MOTIVATION.md#m2) |
| V9-05 | 进程设置审计 | 记录并恢复候选导入和执行修改的后端、默认dtype、线程和环境变量；收据不含环境变量值。 | `kb_workflows/strict_validation.py` | [M2](MOTIVATION.md#m2) |
| V9-06 | 逐项证据等级 | 实现current/L0/L1/L2要求、逐字段状态、缺工具值为null；按真实检查决定达到的等级。 | `kb_workflows/evidence.py` | [M5](MOTIVATION.md#m5) |
| V9-07 | 严格评分门与协议隔离 | 显式严格模式下，未适配题、失败检查和缺失收据拒绝资格；后端设置与等级参与任务及比较指纹。 | `kb_workflows/paper_analysis.py` | [M3](MOTIVATION.md#m3) |
| V9-08 | 样本聚合消融 | 对同一批声明槽位计算mean、median、best-of-N，保留失败零分和成本记录，正式分数不变。 | `kb_workflows/aggregation.py` | [M3](MOTIVATION.md#m3) |

通过`--strict-validation`或`--evidence-level L0`启用；L1/L2会要求尚未实现的设备证据，缺失时不给相应等级或评分资格。未适配题在严格模式下明确失败，不退回旧容差冒充严格通过。详细契约、误差公式、范围及等级定义见[严格验证](STRICT_VALIDATION.md)。

验证：Python 3.12.12 / Torch 2.12.0 / Harbor 0.24.0 / mini-swe-agent 2.4.6 下 **368项CPU/模拟回归通过**，较第八版新增81项。17道实际参考分别以FP32、FP16、BF16执行小规模FP64及隐藏形状检查；反例覆盖BF16计算冒充FP32、GELU近似、单元素错误、缓存、参数更新、BN状态、进程设置恢复、极端量级RMS溢出、首轮原生评测篡改、评分收据缺失/伪汇总和轻量预检。Ruff、168包依赖一致性、motivation覆盖、本地链接与独立wheel轻量加载通过。真实GPU、GPU阈值校准和模型实验未执行；搁置的stash没有恢复。

## 第十版：交替计时、独立最终重测与扩展基线

| 项目 | 版本信息 |
| --- | --- |
| 提交标题 | Add paired timing and independent frozen-source remeasurement |
| 基础提交 | `bcab5645b8e0dbad3937226706fb405e316ef1f2` |
| 日期 | 2026-10-21 |
| 状态 | 已提交；404项CPU/模拟回归通过；真实GPU验收仍待完成 |

本版交付R-11/12的代码并推进R-13，继续保持全部23项设计目标。新增机制可配置，不修改未启用它们的既有计分口径；当前设备相关结论仍需真实GPU验收。

| 编号 | 功能 | 变化 | 实现入口 | Motivation |
| --- | --- | --- | --- | --- |
| V10-01 | 交替计时与噪声对照 | 同进程平衡参考/候选/参考控制顺序，逐次验证输出，按重复进程与配对样本重采样估计倍率区间。 | `kb_workflows/paired_task.py` | [M3](MOTIVATION.md#m3) |
| V10-02 | 冻结后独立最终重测 | 冻结源码按新种子在独立任务重测标准和隐藏正确性，保留全部重复，正式分数使用重测而非最快重复。 | `kb_workflows/final_verification.py` | [M3](MOTIVATION.md#m3) |
| V10-03 | 扩展基线组合与可信改写 | 增加compile default/max-autotune各自Graph组合；3道归一化题增加源码指纹约束的框架等价改写。 | `kb_workflows/baseline_variants.py` | [M3](MOTIVATION.md#m3) |
| V10-04 | 附加证据与旧评分门 | 最终和交替测量必须匹配当前源码/基线/配置，收据文件及冻结内容参与旧评分资格复核。 | `kb_workflows/common.py` | [M3](MOTIVATION.md#m3) |
| V10-05 | 重测恢复与失败分母 | 恢复保留未变样本的新种子和完成记录，仅源码改变的样本重跑；独立重测失败进入正确性pass@k分母。 | `kb_workflows/final_verification.py` | [M5](MOTIVATION.md#m5) |

新增`baseline_portfolio = "extended"`、`paired_timing = true`、`paired_trials = 30`、`final_repeats = 2`。交替计时要求已验证的组合基线；普通四变体组合仍默认。最终重测和交替计时不向agent返回隐藏反馈。原有均值槽位聚合保留，重测使用全部新进程样本，并保留原冷成本点估计。完整流程、假设和输出见[配对与最终计时](PAIRED_TIMING.md)。

另外修正严格验证将受信任的`TORCH_USE_CUDA_DSA`编译设置误记为候选修改的风险：设置在进入候选审计之前完成。实际候选的首次导入/执行修改仍受审计。

验证：Python 3.12.12 / Torch 2.12.0 / Harbor 0.24.0 / mini-swe-agent 2.4.6 下 **404项CPU/模拟回归通过**，新增36项。覆盖六种顺序平衡、输入准备排除、数值错误拒绝、噪声控制、重复进程bootstrap、3个真实参考改写、扩展任务/Graph编排、冻结来源与种子、仅变更样本恢复、最终失败门、附加收据和旧资格篡改拒绝、Harbor预算与reward。Ruff、来源/文档检查及离线wheel独立轻量导入通过。Slurm作业50703（taurus）和50704（aries）在即时申请窗口返回`Requested nodes are busy`，未运行GPU；不能据此声明CUDA Graph、真实计时或GPU隔离已验证。之前stash保持原样。

## 第十一版：设备模型与Headroom评分

| 项目 | 版本信息 |
| --- | --- |
| 提交标题 | Add device calibration and certificate-gated headroom scoring |
| 日期 | 2026-10-21 |
| 基础提交 | `b0d4b09` |
| 状态 | 已提交；只进行CPU/模拟回归，按用户要求不运行任何实验 |

| 编号 | 功能 | 具体变化 | 实现入口 | Motivation |
| --- | --- | --- | --- | --- |
| V11-01 | 设备校准接口与原始收据 | 提供显式设备探针与cache flush/time接口，绑定物理UUID、驱动、软件和评测器，重算原始样本速率；本轮只实现不运行校准 | `kb_workflows/calibration.py` | [M3](MOTIVATION.md#m3) |
| V11-02 | 工作量证书与独立可达性见证 | 核对参考源码、输入、精度和校准；D标签要求算法说明与至少两次独立标准/隐藏记录，历史暂定标签不能代替证书 | `kb_workflows/headroom.py` | [M3](MOTIVATION.md#m3) |
| V11-03 | Headroom评分与证据失效 | 按SOL_target归一稳态收益，缺证书置零并保留固定分母；另报摊销bounded gain，外部规格/校准修改使旧资格失效 | `kb_workflows/paper_analysis.py` | [M3](MOTIVATION.md#m3) |
| V11-04 | 设备时间并集与性能解释 | 计算重叠活动并集与kernel数，来源未知不猜测；缺工作计数器时A/E为null，启动收益须有测量解释，暂为诊断 | `kb_workflows/performance.py` | [M3](MOTIVATION.md#m3) |
| V11-05 | Verifier配置与精度设置恢复 | 可信规格和校准只进入Harbor verifier；开发子任务保留原评分模式；补齐半精度归约开关恢复与修改审计 | `kb_workflows/harbor.py` | [M5](MOTIVATION.md#m5) |

默认评分保持不变；显式选择 `score_mode = "headroom"` 并提供可信规格和校准收据后，使用论文分母计算稳态分数。原K摊销bounded gain另报。达到的微基准速率是设备模型依据，不包装成物理上限证明。缺证书、输入/设备/精度不匹配及独立见证失败不能计分；低于模型估计只提示核查。完整格式与边界见[Headroom评分说明](HEADROOM_SCORING.md)。

验证：Python 3.12.12 / Torch 2.12.0 下446项CPU/模拟回归通过，其中42项为本版新增。覆盖校准调度和CUDA events调用顺序的模拟、原始数据/驱动/UUID/版本绑定、SOL资源最大值、证书与D独立见证、持久化评分与外部证据失效、Harbor可信文件分离、区间并集与未知计数器。Ruff、动机表、本地文档链接与离线wheel轻量导入检查通过。

本版未运行GPU/Trainium校准、模型生成、训练或科研实验。设备收据与实验结果维持未采集；后续实现按同一范围推进。来源归因和实际硬件计数器采集仍未完成；A/E不会填造数值。之前stash保持不变。

## 第十二版：生产题族、逐角色精度与反向轨道

| 项目 | 版本信息 |
| --- | --- |
| 提交标题 | Add production task families with role-aware inputs and backward tracks |
| 日期 | 2026-10-21 |
| 基础提交 | `d2594bd` |
| 状态 | 已提交；CPU/模拟回归，不运行科研实验 |

| 编号 | 功能 | 具体变化 | 实现入口 | Motivation |
| --- | --- | --- | --- | --- |
| V12-01 | Production-41参数化题集 | 独立Level 5增加12设计、6数值设计、12固定格式与11反向任务，保留旧250题及旧Level 4模型 | `kb_workflows/production_specs.py` | [M1](MOTIVATION.md#m1) |
| V12-02 | 独立反向轨道与梯度参考 | 11个VJP参考支持外层no_grad下执行；FP64 gradcheck和方向有限差分核对梯度，轨道分别选题与评分 | `kb_workflows/production_ops.py` | [M1](MOTIVATION.md#m1) |
| V12-03 | 逐输入角色的dtype契约 | 保留FP32缩放、bool mask与离散索引，支持FP8存储/packed INT4及E2M1软件量化；有效精度写入任务而非统一强转 | `kb_workflows/production.py` | [M1](MOTIVATION.md#m1) |
| V12-04 | 显存计划与合法隐藏实例 | 按声明包络及设备容量确定性缩放并保存计划，私有尺寸/stride/动态范围/分页/mask及归一化近常量遵守输入域 | `kb_workflows/production.py` | [M2](MOTIVATION.md#m2) |
| V12-05 | 生产题数值预算与零答案门 | 所有41题使用相对RMS及逐元素比较，六个数值设计预算独立；逐题验证全零反例，参考比较与独立FP64自资格明确区分 | `src/validation.py` | [M2](MOTIVATION.md#m2) |
| V12-06 | 生产题全流程与来源恢复 | 选题/轨道贯穿原生与Harbor，prompt保留数学语义并移除生成器；生成参考与manifest可重建核对，协议参与恢复指纹 | `kb_workflows/engine.py` | [M5](MOTIVATION.md#m5) |

使用 `--suite production41 --level 5`，按 `--task-track design|numerical|implementation|backward` 选择独立轨道。`all`用于收集，混合分数只作诊断。small/standard/large与显存包络显式配置；角色格式、FP8及FP4的实现边界、隐藏实例和数值预算见[生产题说明](PRODUCTION_TASKS.md)。

验证：Python 3.12.12 / Torch 2.12.0 下614项CPU/模拟回归通过，本版新增168项。覆盖全部41题的合法角色、支持dtype、隐藏奇数/stride/范围、全零反例、11个FP64 gradcheck与有限差分、10个独立Torch公式对照、pack/tie规则、显存缩放、原生失败分母及Harbor/轨道配置。生成文件一致性、Ruff、动机表、本地文档链接与离线wheel轻量导入检查通过。

参考和原生/Harbor流程均已实现；GPU编译、计时、厂商最强baseline、原生FP4吞吐和可达性实验未运行，不能据此声称实机性能或完整L0资格。之前stash保持不变。

## 第十三版：选集分析与预算研究工作流

| 项目 | 版本信息 |
| --- | --- |
| 提交标题 | Add evidence-bound selection studies and budget curves |
| 日期 | 2026-10-21 |
| 基础提交 | `caa7796` |
| 状态 | 已提交；只运行CPU/模拟回归，不启动实验 |

| 编号 | 功能 | 具体变化 | 实现入口 | Motivation |
| --- | --- | --- | --- | --- |
| V13-01 | 研究结果的当前证据导出 | 重新核对当前源码与计分资格后导出固定槽位，合并互不重叠层级并拒绝复制运行冒充独立会话；不调用生成或评测 | `kb_workflows/study_data.py` | [M1](MOTIVATION.md#m1) |
| V13-02 | 选集区分度与统计不确定性 | 支持并列秩Spearman、Kendall tau-b、差值/排名反转、任务/会话/样本bootstrap与显式二元power假设；常量相关为null，不假称旧排名已复现 | `kb_workflows/study_statistics.py` | [M1](MOTIVATION.md#m1) |
| V13-03 | 多预算多会话与通用agent计划 | 生成原生one-shot/反馈及Harbor coding-agent网格，保留真实候选限制；默认不执行，显式执行时经过原生配置校验与resume | `kb_workflows/study_plan.py` | [M4](MOTIVATION.md#m4) |
| V13-04 | 得分预算曲线与共同区间AUC | 独立预算点保留失败分母与非单调结果，区间覆盖任务/会话/样本；未知成本不免费，AUC不外推或取最好历史 | `kb_workflows/budget_curves.py` | [M4](MOTIVATION.md#m4) |
| V13-05 | 缓存与外部agent用量核算 | 补总token和样本执行时间，区分Claude与Harbor cache口径；多步缺项保持未知，外部agent收据绑定结果文件，不冒充GPU账单 | `kb_workflows/agent_models.py` | [M4](MOTIVATION.md#m4) |
| V13-06 | 研究CLI与静态图导出 | 提供计划/选集/预算分析入口、可选Matplotlib静态图；分析实现单独指纹，残缺选集必须显式标记，文档与功能清单同步 | `scripts/analyze_budget_curves.py` | [M5](MOTIVATION.md#m5) |

新增 `scripts.plan_study`、`scripts.analyze_selection` 和 `scripts.analyze_budget_curves`。计划默认不执行，分析只重算当前文件证据；研究用的任务/样本分母、方法/预算差异、遗漏选集与未知资源均显式保存。完整格式、假设和使用范围见[研究工作流](STUDIES.md)。

验证：Python 3.12.12 / Torch 2.12.0 下644项CPU/模拟回归通过，本版新增30项；覆盖实际计分重算、来源失效、并列与全失败排名、分母/重复会话拒绝、曲线回落与AUC区间、真实CLI解析24个模拟job、Harbor/cache用量及Matplotlib静态导出。Ruff、动机表、文档链接和离线wheel轻量导入检查通过。

本版没有运行模型/GPU/Trainium或训练实验，也没有补造原7模型数据或声称精确排名重现。token/设备时间硬上限仍未实现，估算费用和agent墙钟不等于GPU账单；之前stash保持不变。

## 第十四版：资源预留、本地模型与预算证据

| 项目 | 版本信息 |
| --- | --- |
| 提交标题 | Enforce auditable agent token reservations and development leases |
| 日期 | 2026-10-21 |
| 基础提交 | `511b1ab` |
| 状态 | 软件实现与CPU/模拟回归；没有运行实验 |

| 编号 | 功能 | 具体变化 | 实现入口 | Motivation |
| --- | --- | --- | --- | --- |
| V14-01 | 调用前token预留与用量结算 | 本地总token与支持的API输出上限；请求前预留、已知用量返还、缺失或超额停机；不支持的后端拒绝配置 | `kb_workflows/agent_models.py` | [M4](MOTIVATION.md#m4) |
| V14-02 | 离线本地模型适配 | 精确chat template输入ID、单序列生成、禁止下载与远端代码、safetensors加载、模型及tokenizer内容绑定 | `kb_workflows/local_model.py` | [M4](MOTIVATION.md#m4) |
| V14-03 | 开发worker占用预留 | local/Docker父进程计时覆盖编译/验证/profiling与清理，预留清理时间；超额可见，不宣称纯GPU时间物理抢占 | `kb_workflows/resource_budgets.py` | [M4](MOTIVATION.md#m4) |
| V14-04 | 预算恢复与计分资格 | 限额样本只许一个可审计尝试；重算账本并绑定历史/汇总/源码/配置，改收据使新旧评分门失效 | `kb_workflows/paper_analysis.py` | [M3](MOTIVATION.md#m3) |
| V14-05 | token与设备占用预算网格 | 增加三种预留额度的研究计划与曲线横轴，按后端能力预检，其他预算继续分组；默认不执行 | `kb_workflows/study_plan.py` | [M4](MOTIVATION.md#m4) |

配置、作用范围、未知用量处理与恢复规则见[资源预算](RESOURCE_BUDGETS.md)。R-18推进为部分实现；纯GPU busy时间硬限与远端可信取消仍待实现。不会把worker墙钟、模型token计数或客户端估算费用写成GPU账单。

验证：Python 3.12.12 / Torch 2.12.0下681项CPU/模拟回归通过，本版新增37项；覆盖预留/返还/未知扣款、超额与不支持后端、最佳候选保留、模型资产变更、评分证据失效与预算网格。真实模型与设备实验均未运行，Ruff、动机表与文档链接检查通过。此前stash保持原样。

## 第十五版：GRPO训练入口与奖励验证

| 项目 | 版本信息 |
| --- | --- |
| 提交标题 | Integrate local GRPO with native terminal rewards and audit workflows |
| 日期 | 2026-10-21 |
| 基础提交 | `2ba12d6` |
| 状态 | 软件实现与CPU/模拟回归；真实模型和训练实验未运行 |

| 编号 | 功能 | 具体变化 | 实现入口 | Motivation |
| --- | --- | --- | --- | --- |
| V15-01 | 冻结训练计划与持出划分 | 显式任务/源码/family划分、模型资产及实现指纹，公开参考沿用input-blind策略，默认不执行 | `kb_workflows/rl_plan.py` | [M4](MOTIVATION.md#m4) |
| V15-02 | 原生GRPO与标准开发工具 | 固定TRL 1.8.0，one-shot/反馈环境及候选选择，离线加载；持出题不进入训练callback | `kb_workflows/rl_training.py` | [M4](MOTIVATION.md#m4) |
| V15-03 | 分阶段小实例资格任务 | 新增qualify_small worker，在原尺寸分配与计时前运行审查契约；基线不合格时先停止模型加载 | `kb_workflows/evaluate_task.py` | [M2](MOTIVATION.md#m2) |
| V15-04 | 同题交替终局奖励 | 共享repeat种子、同设备UUID、独立进程交替重测；严格资格分与弱raw-speedup消融分开，失败保留零分 | `kb_workflows/rl_rewards.py` | [M3](MOTIVATION.md#m3) |
| V15-05 | 训练预算与checkpoint恢复 | 生成前预留group、中断仍扣额、checkpoint绑定文件/step/奖励历史，父进程墙钟监督包含清理 | `kb_workflows/rl_training.py` | [M5](MOTIVATION.md#m5) |
| V15-06 | 奖励重评分与可靠性审计 | 预声明案例、重复重评、ICC/排序/攻击接受率及公共最终差；缺检查与基础设施故障不冒充防攻击通过 | `kb_workflows/rl_audit.py` | [M4](MOTIVATION.md#m4) |
| V15-07 | 持出评测计划与CLI | 生成冻结checkpoint的原生持出任务/会话计划，提供计划、基线预检、训练、重评和离线审计命令 | `scripts/rl_workflow.py` | [M4](MOTIVATION.md#m4) |

完整配置与限制见[RL工作流](RL_WORKFLOW.md)。默认只规划/预览；真实执行需要显式开关。本版不把TRL含工具观察的completion序列算成精确计费token，也不宣称纯GPU时间硬限、完整OS隔离或训练收益已经验证。

验证：Python 3.12.12 / Torch 2.12.0下724项CPU/模拟回归通过，本版新增43项；实际原生评分处理模拟worker证据，覆盖坏候选分母、早停、交替种子、同设备检查、文件篡改、丢失账本、checkpoint恢复、TRL真实配置API、重评样本重排及持出权重冻结。未加载真实模型或运行训练；Ruff、动机表、文档链接和依赖检查通过。

R-20推进为部分实现；候选算法证书加分和绑定训练来源的完整效果对照报告仍需完成。其余研究设计继续保留原状态，不以新增训练入口替代全部23项的验收。此前stash保持原样。

## 第十六版：算法证书奖励与训练效果证据

| 项目 | 版本信息 |
| --- | --- |
| 提交标题 | Bind RL policies to training lineage and reviewed scaling rewards |
| 日期 | 2026-10-22 |
| 基础提交 | `9ebca56` |
| 状态 | 软件实现与CPU/模拟回归；真实模型、设备和训练实验未运行 |

| 编号 | 功能 | 具体变化 | 实现入口 | Motivation |
| --- | --- | --- | --- | --- |
| V16-01 | 显式尺寸实例与证据绑定 | Production-41维度覆盖、整除/显存检查；内容参与基线/冻结/评分/比较，私有隐藏实例保持独立 | `kb_workflows/production.py` | [M1](MOTIVATION.md#m1) |
| V16-02 | 多尺寸算法证书 | 当前D/SOL审查绑定，至少5尺寸跨度16倍，固定源码/设备/变体，完整原生验证与配对斜率区间 | `kb_workflows/scaling_certificate.py` | [M3](MOTIVATION.md#m3) |
| V16-03 | 有界证书奖励 | 默认关闭；仅资格通过且达到阈值的候选可请求，加分封顶1；失败和恒定倍率优化无算法加分 | `kb_workflows/rl_rewards.py` | [M4](MOTIVATION.md#m4) |
| V16-04 | 训练来源与父进程凭据 | 最终权重绑定计划、奖励文件、证书和实际优化路径；中断成本保留，nonce/锁拒绝旧完成记录及并发监督 | `kb_workflows/rl_lineage.py` | [M5](MOTIVATION.md#m5) |
| V16-05 | 预注册奖励对照与持出评测 | 固定两奖励组×全部种子、优化器/数据/预算与持出会话；当前lineage和权重匹配，缺少作业不缩小研究 | `kb_workflows/rl_effectiveness.py` | [M4](MOTIVATION.md#m4) |
| V16-06 | 按训练种子的效果分析 | 重算持出资格分，配对种子/分层任务区间与双侧检验；失败检查比例保留未知上下界，模拟数据不能支持正式效果 | `kb_workflows/rl_effectiveness.py` | [M3](MOTIVATION.md#m3) |

配置与运行规则见[RL工作流](RL_WORKFLOW.md)和[算法证书](SCALING_CERTIFICATES.md)。证书只支持有限审查区间；恒定倍数提速不等于斜率改进。各组相同声明额度不等于实际GPU busy时间相同，训练收益仍未测量。

验证：Python 3.12.12 / Torch 2.12.0下743项CPU/模拟回归通过，本版新增19项。覆盖完整模拟尺寸扫描、恒定倍率反例、基线互不覆盖、证据变更失效、旧完成文件、恢复后优化路径、全部种子/会话/槽位保留及持出权重来源；模拟训练只生成模拟权重，不加载真实模型。Ruff、动机表、19份Markdown本地链接、177包依赖兼容性及离线wheel轻量导入检查通过。

R-20的软件范围已补全，23项合计11项代码已实现、8项部分实现、4项待实现；其余设计继续推进。没有运行真实模型、GPU、Trainium、校准或训练实验，此前stash保持原样。

## 第十七版：原生Neuron/NKI与设备适配接口

| 项目 | 版本信息 |
| --- | --- |
| 提交标题 | Add native Neuron evaluation through shared device adapters |
| 日期 | 2026-10-22 |
| 基础提交 | `e743d71` |
| 状态 | 软件实现与CPU/模拟回归；未运行设备、校准、模型或训练实验 |

| 编号 | 功能 | 具体变化 | 实现入口 | Motivation |
| --- | --- | --- | --- | --- |
| V17-01 | 通用设备适配协议 | 同一caps/calibrate/flush/time/attribute接口，CUDA与Neuron两种实现，导入与预览不加载SDK | `kb_workflows/devices.py` | [M5](MOTIVATION.md#m5) |
| V17-02 | Neuron配置与任务执行 | device节、独立worker解释器、core分配与LNC=1，接入原生标准/隐藏/最终及开发任务，拒绝CUDA环境混用 | `kb_workflows/neuron_worker.py` | [M5](MOTIVATION.md#m5) |
| V17-03 | Neuron基线与配对重测 | eager/compile与4题源码绑定的框架库变体，复用最快正确基线、平衡配对与冻结重测；平台独立分组 | `kb_workflows/baseline_variants.py` | [M3](MOTIVATION.md#m3) |
| V17-04 | CPU FP64与输入角色保护 | 独立oracle留在CPU，逐次验证与双向调用保持模型状态/RNG；冻结NKI源码与环境，禁止FP8存储降级和CPU代理计分 | `kb_workflows/neuron_worker.py` | [M2](MOTIVATION.md#m2) |
| V17-05 | Neuron校准与物理身份 | 序列号/core/驱动/SDK身份，原始copy与分dtype GEMM及启动代理；同步墙钟口径和未知cache明确保留 | `kb_workflows/neuron_device.py` | [M3](MOTIVATION.md#m3) |
| V17-06 | 设备轨迹与每引擎诊断 | 收集框架trace/NEFF/NTFF及哈希，显式单位/窗口分析tensor/vector/scalar/DMA活动并集；不冒充工作计数器或host墙钟 | `kb_workflows/neuron_profile.py` | [M3](MOTIVATION.md#m3) |
| V17-07 | NKI提示与coding skill | 保留ModelNew接口，提示现代NKI及native注册；skill复用设备配置和vendor解释器，客户端/验证器环境分开 | `kb_workflows/generation.py` | [M4](MOTIVATION.md#m4) |

使用方法见[Neuron/NKI说明](NEURON.md)。Production-41中40个可传输存储接口通过CPU前端回归，FP8存储题21显式拒绝；这不等于40题已在Neuron上运行或支持全部算子。原生后端是TorchNeuron 2.14的Beta接口，旧XLA产品、LNC=2和CUDA容器不是这一实现的运行环境。

验证：Python 3.12.12 / Torch 2.12.0下823项CPU/模拟回归通过，本版新增80项。覆盖设备配置/分配、40题前端、4题库等价改写、CPU FP64、小实例资格、状态/RNG进度、模拟校准与profile、首次执行篡改、源码检查及Harbor完整摘要；原生SDK调用用模拟对象替代。Ruff、动机表、20份Markdown本地链接、skill校验、177包依赖兼容性及离线wheel轻量导入检查通过。没有安装或运行Neuron SDK及实机实验。

R-21的软件范围已接入；23项当前为12项代码已实现、8项部分实现、3项待实现。来源归因、硬件计数器和正式归因门继续归R-16，隔离容器仍归R-23。此前stash没有恢复。

## 维护方式

1. 本文作为功能清单入口；后续新增版本节，写明提交、日期和功能状态。
2. 功能编号保持稳定。后续改变默认值、替换实现或删除功能时，引用原编号说明变化。
3. “已提交”“CPU/模拟验证”“真实 GPU/云端验证”分别记录；仅代码接入不等于完成实机验证。
4. 未提交工作单独标记，提交后再固定版本，保留旧版行为，避免混淆历史能力。
5. 实现证据使用固定提交链接。即使当前工作区继续变化，仍可核查本版内容。
6. 每次提交功能变化时，同步更新对应清单条目与验证结果，并将文档纳入同一提交。记录本提交时可用唯一提交标题与定位命令，避免文档引用自身哈希的问题。

上游基准来自该提交的 provenance 记录；本地没有 `3fdf6fec` 对象，本清单依据 `git show 2bce940:<path>` 核对整合内容，不是对上游源码逐文件 diff 的证明。第一版完整操作说明见 [本版工作流文档][v1-doc]。

[v1-cli]: https://github.com/shadowfall09/KernelBench/blob/2bce940c631c01086c262eef9c9efc9d3693e1de/kb_workflows/cli.py
[v1-config]: https://github.com/shadowfall09/KernelBench/blob/2bce940c631c01086c262eef9c9efc9d3693e1de/kb_workflows/config.py
[v1-engine]: https://github.com/shadowfall09/KernelBench/blob/2bce940c631c01086c262eef9c9efc9d3693e1de/kb_workflows/engine.py
[v1-transports]: https://github.com/shadowfall09/KernelBench/blob/2bce940c631c01086c262eef9c9efc9d3693e1de/kb_workflows/transports.py
[v1-modal]: https://github.com/shadowfall09/KernelBench/blob/2bce940c631c01086c262eef9c9efc9d3693e1de/kb_workflows/modal_backend.py
[v1-generation]: https://github.com/shadowfall09/KernelBench/blob/2bce940c631c01086c262eef9c9efc9d3693e1de/kb_workflows/generation.py
[v1-docker]: https://github.com/shadowfall09/KernelBench/blob/2bce940c631c01086c262eef9c9efc9d3693e1de/Dockerfile
[v1-slurm]: https://github.com/shadowfall09/KernelBench/blob/2bce940c631c01086c262eef9c9efc9d3693e1de/batch_run.sbatch
[v1-modal-sh]: https://github.com/shadowfall09/KernelBench/blob/2bce940c631c01086c262eef9c9efc9d3693e1de/batch_run_modal.sh
[v1-queue]: https://github.com/shadowfall09/KernelBench/blob/2bce940c631c01086c262eef9c9efc9d3693e1de/kb_workflows/queue_worker.py
[v1-common]: https://github.com/shadowfall09/KernelBench/blob/2bce940c631c01086c262eef9c9efc9d3693e1de/kb_workflows/common.py
[v1-task]: https://github.com/shadowfall09/KernelBench/blob/2bce940c631c01086c262eef9c9efc9d3693e1de/kb_workflows/evaluate_task.py
[v1-analysis]: https://github.com/shadowfall09/KernelBench/blob/2bce940c631c01086c262eef9c9efc9d3693e1de/scripts/benchmark_eval_analysis.py
[v1-legacy]: https://github.com/shadowfall09/KernelBench/blob/2bce940c631c01086c262eef9c9efc9d3693e1de/kb_workflows/legacy.py
[v1-shell]: https://github.com/shadowfall09/KernelBench/blob/2bce940c631c01086c262eef9c9efc9d3693e1de/batch_run.sh
[v1-doc]: https://github.com/shadowfall09/KernelBench/blob/2bce940c631c01086c262eef9c9efc9d3693e1de/docs/BATCH_WORKFLOWS.md
[v1-tests]: https://github.com/shadowfall09/KernelBench/blob/2bce940c631c01086c262eef9c9efc9d3693e1de/tests/test_workflows.py
