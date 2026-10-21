# 相比 KernelBench-Verified 的功能改进清单

本清单按版本维护，记录功能改进、实现入口和验证程度。第一版记录 `2bce940` 的批量工作流整合，第二版记录正确性判断与输入隔离，第三版记录基线与计时，第四版记录评分与恢复，第五版记录从相关工作引入的可靠性检查。

当前已提交能力为第一至第五版的累计功能。此前的评测完整性整合与文件清理已保存为 stash，继续暂缓，未混入第五版。

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

| 编号 | 功能 | 相对 Verified 的改进与用途 | 实现入口（本版快照） |
| --- | --- | --- | --- |
| V1-01 | 统一批量入口 | **流程增强。** 用一套配置串联基线、生成、评测和分析；支持 `all`、`generate`、`evaluate`、`analyze`、`baseline` 五种阶段，减少多脚本间的手工传参。 | [cli.py][v1-cli]、[config.py][v1-config] |
| V1-02 | 生成与最终验证独立选择后端 | **迁移整合。** 生成支持 local / Docker / Modal，验证支持 local / Docker / Modal / HTTP 队列；例如 CPU 容器生成、远端 GPU 验证。不同后端共用参考源码、评测设置与结果协议。 | [engine.py][v1-engine]、[transports.py][v1-transports]、[modal_backend.py][v1-modal] |
| V1-03 | 题目范围与多样本批量生成 | **迁移整合。** 支持 level 1–3、闭区间题号和每题多次独立 agent 尝试；每次尝试有独立工作目录、日志及内核文件。 | [config.py][v1-config]、[generation.py][v1-generation] |
| V1-04 | 多 GPU 调度与并发限制 | **迁移整合。** 本地 GPU 生成按选定设备限制并发；最终评测在同一 GPU 上串行、不同 GPU 间并行；远端队列可限制在途任务数。 | [generation.py][v1-generation]、[engine.py][v1-engine]、[transports.py][v1-transports] |
| V1-05 | Docker 生成与验证 | **迁移整合。** 统一 Dockerfile，提供带 CUDA 工具链的 `agent` 和用于远端验证的 CPU 生成目标 `agent-remote`；镜像包含当前 checkout 的题目、隐藏测试与评测代码。 | [Dockerfile][v1-docker]、[generation.py][v1-generation]、[transports.py][v1-transports] |
| V1-06 | Slurm 集群提交 | **迁移整合。** 提供批量作业脚本，使用提交目录、可配置 Python 和分配到的 `CUDA_VISIBLE_DEVICES`；通过环境变量指定题目、设备与后端。 | [batch_run.sbatch][v1-slurm]、[transports.py][v1-transports] |
| V1-07 | Modal 云端执行 | **迁移整合。** 支持云端最终验证及 GPU Sandbox 内的生成、迭代验证；退出 Sandbox 前收集内核和日志，使用同一套 Verified 评测入口。 | [modal_backend.py][v1-modal]、[batch_run_modal.sh][v1-modal-sh] |
| V1-08 | 持久化 HTTP 验证队列 | **迁移整合与可靠性增强。** HTTP 提交/查询、SQLite 保存任务与结果、重启后重新排队未完成任务；相同请求幂等、瞬时 HTTP 错误重试，并限制待处理任务数量。 | [queue_worker.py][v1-queue]、[transports.py][v1-transports] |
| V1-09 | 带内容指纹的断点恢复 | **流程增强。** 评测恢复核对参考、内核、评测器、隐藏测试和配置；生成恢复核对模型、prompt 与输出校验和。修改内核后重新执行对应的标准及隐藏评测，旧格式结果重新验证。 | [common.py][v1-common]、[engine.py][v1-engine]、[generation.py][v1-generation] |
| V1-10 | 批量任务的超时与失败处理 | **流程增强。** 单次验证使用独立子进程与编译目录；超时终止进程组，Docker 验证同时清理容器；缺失内核写入失败记录，失败的重新生成不会继续使用旧内核冒充新结果。 | [transports.py][v1-transports]、[generation.py][v1-generation]、[engine.py][v1-engine] |
| V1-11 | 将 Verified 隐藏门接入全部验证后端 | **评测整合。** 默认先做标准正确性、CUDA-event 计时和峰值显存，再做隐藏正确性；最终正确性要求两者通过，隐藏结果缺失也不算通过。耗时和显存保留标准评测值。 | [evaluate_task.py][v1-task]、[engine.py][v1-engine]、[benchmark_eval_analysis.py][v1-analysis] |
| V1-12 | 基线记录接入批量流程 | **流程增强。** 可在选定验证后端记录指定题目基线；完整流程自动补测缺失基线。FP32 默认 TF32，FP16/BF16 分开存储，也可显式选择 `torch.compile` 基线；记录耗时、显存与环境信息。 | [evaluate_task.py][v1-task]、[engine.py][v1-engine]、[config.py][v1-config] |
| V1-13 | 子集分析、pass@k 与统一结果归档 | **流程增强。** 批量分析使用选定题目作为分母；pass@k 使用声明样本数，缺失和隐藏失败样本计失败；输出分析摘要、逐样本耗时/显存对比，并保持 Verified 结果键格式与原有指标公式。 | [engine.py][v1-engine]、[benchmark_eval_analysis.py][v1-analysis] |
| V1-14 | 旧入口兼容与配置预览 | **流程增强。** 旧 shell / Python 入口转接统一配置，兼容常用 pydra 参数；`--dry-run` 预览配置与题目，不创建运行目录、不调用模型或云服务。 | [legacy.py][v1-legacy]、[cli.py][v1-cli]、[batch_run.sh][v1-shell] |

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

| 编号 | 功能 | 具体变化 | 实现入口 |
| --- | --- | --- | --- |
| V2-01 | 严格比较输出结构、shape 和 dtype | 递归检查 Tensor、dict、list、tuple；核对字典键、序列类型和长度，以及各 Tensor 的 shape、dtype；记录具体不一致的输出路径。 | [validation.py](../src/validation.py)：`compare_outputs` |
| V2-02 | 整数与 bool 精确比较 | 离散 Tensor 使用精确比较。比如整数索引 `100000` 与 `100001` 即使满足相对浮点容差，也会被判为不一致；整数和 bool 输入不再被统一转换为浮点精度。 | [validation.py](../src/validation.py)、[eval.py](../src/eval.py)：`_process_input_tensor` |
| V2-03 | 核对 NaN、Inf 的位置与符号 | NaN、正 Inf、负 Inf 的掩码分别匹配；匹配后，仅有限数值继续按原 FP32/FP16/BF16 容差比较。 | [validation.py](../src/validation.py)：`compare_outputs` |
| V2-04 | 隔离参考与候选输入 | 两个模型分别使用独立的 Tensor 副本，支持嵌套容器，避免参考或候选的原地操作修改另一方的输入和调用方原始数据。 | [validation.py](../src/validation.py)：`clone_tensors`；[eval.py](../src/eval.py)：两个正确性入口 |
| V2-05 | 保存参考输出快照 | 在调用候选前保存参考结果，避免共享或复用输出缓冲区把参考答案覆盖，导致错误实现被判通过。 | [eval.py](../src/eval.py)：`run_and_check_correctness`、`run_correctness_on_fixed_inputs` |
| V2-06 | 对齐 forward 随机种子 | 参考与候选 forward 使用相同种子，避免 dropout 等随机计算因执行先后产生不同结果。 | [eval.py](../src/eval.py)：两个正确性入口 |

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

| 编号 | 功能 | 具体变化 | 实现入口 |
| --- | --- | --- | --- |
| V3-01 | 默认四变体基线 | eager、compile default、compile max-autotune、CUDA Graph 各自独立任务运行；验证初次及每次计时输出，保留失败变体，在正确变体中选最优。组合的 FP32 使用 IEEE；显式单基线仍可使用 TF32。 | `baseline_variants.py`、`evaluate_task.py`、`engine.py` |
| V3-02 | 同步墙钟计时 | 调用前后同步，计入 Python 调度、调用、输出消费及 GPU 完成等待；输入准备和完整参考校验在计时外。Graph 的输入复制和输出克隆计入 replay 耗时。 | `measurement.py`、`src/eval.py` |
| V3-03 | 变化输入与逐次输出验证 | 批量基线和候选每次重新调用可信输入生成器，记录新种子；保留输入副本，每次检查输出，失败不作为有效性能结果。固定生成器仍可能产生相同内容。 | `measurement.py`、`evaluate_task.py`、`src/eval.py` |
| V3-04 | 冷成本与 K 次摊销 | 记录构建/初始化、首次调用和预热；按 `(cold + (K-1) × steady) / K` 报告平均成本，默认 K=10000，同时保留稳态最优与摊销最优基线。 | `measurement.py`、`baseline_variants.py`、`config.py` |
| V3-05 | 原始样本和测量信息 | 保存毫秒样本、均值/标准差及正态近似 95% 均值区间；记录真实设备、软件版本、输入形状/stride/dtype 和源码指纹，供后续兼容性检查。 | `measurement.py`、`evaluate_task.py` |
| V3-06 | 区分旧计时与新结果 | 新默认文件名为 `baseline_time_suite_fp32_wall`；`--no-baseline-suite` 可选单基线，编译单基线需同时指定该开关；基线组合未完整记录时重新执行。 | `config.py`、`cli.py`、`engine.py` |

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

| 编号 | 功能 | 具体变化 | 实现入口 |
| --- | --- | --- | --- |
| V4-01 | 失败和缺失样本计入分母 | 每个请求题目 × 声明样本数形成固定槽位；失败、超时、缺失、过期或比较证据不足的槽位记 0。先按题取所有样本均值，再按层宏平均，不选择最快样本作为新评分。 | `paper_analysis.py`：`analyze_paper` |
| V4-02 | 按 K 摊销成本评分 | 在有当前证据的变体中，按声明 K 重新选择摊销最优基线；评分为 `clip(log(T_ref/T_cand)/log(1.25), 0, 1)`，同时要求所选基线与候选的稳态均值区间不重叠。保留稳态加速比和摊销加速比。 | `paper_analysis.py`、`measurement.py` |
| V4-03 | 兼容性和测量证据检查 | 检查设备类别、Torch/CUDA/HIP、精度/容差、计时协议、输入签名、源码与任务指纹；要求变化输入、逐次校验收据、有效原始样本和冷成本。组合基线必须记录完整，具体变体也须匹配当前任务。 | `paper_analysis.py`：`compatible`、`measurement_evidence`、`current_result` |
| V4-04 | 分层统计与区间 | `--problem-specs` 可声明 I/F/D 层，未声明为 `unclassified`；保留失败槽位，输出按题和样本两层重采样的 bootstrap 区间。该版本不加入设备校准或 headroom 评分。 | `config.py`、`paper_analysis.py` |
| V4-05 | 扩大恢复指纹覆盖 | 指纹覆盖共享验证、计时、基线变体、配置、任务编排、传输和分析模块；版本或源码变化使旧任务失效。完整流程发现过期基线时重新记录，客户端与 worker 版本不匹配则拒绝执行。 | `common.py`、`cli.py`、`engine.py`、`evaluate_task.py` |
| V4-06 | 旧分析与榜单不能绕过证据检查 | 新批量结果经 `eval_results_scored.json` 提供资格；读取时重新核对当前内核、参考题目、隐藏源码、版本、运行配置及原始结果/基线内容。缺失或过期资格拒绝计分；隐藏合并不会把标准失败覆盖为通过。 | `common.py`：`apply_score_gate`；两个分析/榜单脚本 |

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

| 编号 | 功能 | 具体变化与来源 | 实现入口 |
| --- | --- | --- | --- |
| V5-01 | 参考前后配对检查 | 借鉴 robust-kbench 的前后调用，标准与隐藏检查均执行 C→R、R→C 两组，两个模型调用次数一致；同步、重放 RNG、输入隔离与输出快照防止当前参考缓存和共享缓冲污染。 | `src/reliability.py`、`src/eval.py` |
| V5-02 | 隔离源码全局字典与后端设置 | 候选同名变量/函数不能覆盖参考定义；每次 forward 后恢复共享后端设置，防止候选改变精度设置影响参考比较。属于本地配套实现，不声称提供安全沙箱。 | `src/eval.py`、`src/reliability.py` |
| V5-03 | 三个种子重新生成隐藏输入 | 借鉴 Hard 的三种子验证，分别在 42、123、456 下调用可信隐藏生成器，全部配置必须通过。硬编码生成器仍可能返回相同内容。 | `src/eval.py`：`run_hidden_correctness_check` |
| V5-04 | 按输入域限定的数值压力测试 | 借鉴 Hard 的按题压力输入及 KernelBenchX 的 0.1%/×50 离群分布。29 道 Level 1 题的首个隐藏配置增加零值、×32 和稀疏离群值；不改变 shape/stride/dtype 或其他输入，不把扰动施加到未经核对的受限域。 | `src/reliability.py`：`STRESS_TASKS`、`stress_cases` |
| V5-05 | 无效验证不能算通过 | 空、缺失、格式错误的隐藏集显式报错；参考报错记录来源并判验证失败；标准零次测试不能通过。是落实“所有配置必须实际验证”的本地修补。 | `src/eval.py`、`src/reliability.py` |
| V5-06 | 分阶段诊断与逐项收据 | 参考 KernelBenchX/TritonRL 对编译、正确性、性能的区分，新增 `validation_summary.json`，失败/缺失样本保留分母；metadata 保存种子、配置、调用顺序和失败来源。阶段诊断不代替第四版的计分资格，也不证明功能合法性。 | `kb_workflows/engine.py`、`src/reliability.py` |
| V5-07 | 新协议参与恢复指纹 | 新可靠性模块纳入 evaluator digest，使旧缓存不能继续冒充新协议结果；沿用既有 worker 版本核对和过期证据拒绝。 | `kb_workflows/common.py` |

验证：Python 3.10 / Torch 2.9.0 下 **172 passed**，其中新增 [test_reliability.py](../tests/test_reliability.py) 的 58 项，原有 114 项全部通过。覆盖全部 29 道适配题的小形状真实 CPU 参考执行，以及缓存答案、第二次调用错误、无效隐藏集、非法域排除、源码全局覆盖和恢复指纹回归。Ruff 与 diff 检查通过。

范围与成本：原四配置隐藏集变为每题 12 次配置检查；适配题增加压力输入后为 21 次，每配置两组配对。额外检查不进入稳态计时；首次候选 forward 提前到首次参考 forward 前，冷成本仍记录。Level 2/3 本次没有新增数值扰动。保留当前浮点容差与匹配 NaN/Inf 规则，没有迁入 Hard/X 题集、SOTA 外部库、TFLOPS/带宽评分、模型裁判或 backward 验证，不宣称完整反作弊通过。

本次 Slurm GPU smoke 作业 50702 因 `Requested nodes are busy` 未能启动；真实 CUDA 和固定 Torch 2.12.0 尚未验证。

原先未提交的整合与清理保存在 stash `ba76365ce0dd755bce1361e38420bb5f4ec3e196`，本次没有恢复。以后恢复需处理与本版重叠的正确性入口，并重新验证。

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
