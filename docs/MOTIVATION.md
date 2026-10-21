# KernelBench Pro：Motivation 与实现对照

本文是项目动机和范围的统一入口；[论文](../research/headroom/PAPER.md)沿用同样的五条主线，[功能清单](FEATURE_INVENTORY.md)保留逐版历史。实现、研究设计和实验结果分别记录。当前计算后端是 NVIDIA CUDA；Trainium/NKI 属于后续设计。

<a id="m1"></a>

## M1：题目与区分度

大量相似题消耗评测预算，而少量需要算法设计的题在总分中被稀释。压缩题集可以降低运行成本，但不能据此声称已经提升区分度。新增生产难题、反向任务和数值格式设计，才有可能测出不同于常规实现优化的能力。

当前支持 Level 1–3 范围、多样本，以及从研究包接入的 KernelBench-105 固定选集：105 道主集、5 道审查探针、140 道扩展题。三种角色独立运行。I/F/D 是研究者给出的暂定分类，不是算法最优性或可达性证书；原始输入尚未按论文的退化审计重新校准。用法和选择规则见[研究索引](../research/headroom/README.md)。

还需要生产风格难题、逐题 dtype 与合法输入域契约、backward、隐藏形状、按显存缩放，以及在当前题目和真实反馈循环下重做区分度实验。历史 7 模型原始结果未在压缩包中提供；论文声称的精确排名一致性不能在本仓库独立重现。

<a id="m2"></a>

## M2：正确性与执行可信度

“原随机输入上接近参考”不足以证明程序实现了题意。输出格式、离散结果、非有限值、数值量级、状态和执行路径都可能让错误实现通过；更多反馈还会增加针对评测器优化的机会。

当前实现严格的结构/shape/dtype 比较、整数与 bool 精确比较、NaN/Inf 对齐、输入副本和输出快照、RNG 配对、参考前后调用检查、源码命名空间和后端设置恢复。隐藏验证重新使用三个种子；29 道经过输入域核对的 L1 题增加零值、放大和离群压力。墙钟计时包含输出消费和设备同步，每个计时输出也要校验。

这些能力分别减少特定错误，不能合并成“完整反作弊通过”。目前没有逐题 FP64 oracle、退化输出自动校准、全部根因输入族、隐藏 shape 覆盖、自写内核来源证明或完整模式/状态/参数更新检查。研究包的审查记录保留为**源码审查结论**；1,574 份候选和反例没有在这里上 GPU 复测。原型中的统一扰动、跳过非有限参考、固定自写时间占比阈值均未直接纳入默认评测。

<a id="m3"></a>

## M3：可解释且可比较的性能指标

加速比依赖所用基线、计时协议、冷启动摊销和样本选择；弱基线、噪声和只统计成功样本都会改变结论。同样的倍率也不能解释收益来自算法、访存、硬件利用率还是启动开销。

当前默认比较四个经过正确性验证的基线变体，保留原始墙钟样本、设备与软件信息、冷成本及 K 次摊销结果。评分要求当前源码和测量证据匹配，按声明题目和样本计入失败；先按题求均值，再分层宏平均，并报告 bootstrap 区间。I/F/D 可由规格文件或选集暂定分类提供。不同设备、协议和预算分开比较。

当前 bounded gain 的分母为 `ln(1.25)`；没有设备实测上限时，它**不是论文提出的 headroom 分数**。现有 Torch profiler 是开发诊断，不提供自写时间占比、硬件计数器或 A/E/O 归因。交替计时、参考对参考噪声底、实测 roofline、算法边界及见证、独立获胜结果重测、完整性能分解和 `launch_only` 门仍是研究设计。

<a id="m4"></a>

## M4：现实的 Agent 回路与预算

现实内核开发包含写、编译、检查、分析和修改。一次生成、多样本挑最好、多轮反馈与进化搜索必须分开声明预算；隐藏分不能成为开发反馈。

当前原生流程使用 mini-swe-agent 2.4.6，区分 one-shot 与 feedback。默认每样本最多 5 次候选、2400 秒，可配置；开发检查只看公开实例，控制器保留开发正确且最快的候选，再冻结源码执行正式标准和隐藏验证。开发 profiler 单独运行。Claude Code/LiteLLM 使用相同协议；Harbor 标为 external，当前 coding-agent skill 会话标为 interactive。

保存轨迹、每轮源码、生成回执和重试历史；可观测 token 与费用如实记录，未知为 null。费用限制是调用间的客户端估算检查，不能保证单次请求不超限；纯 GPU 时间及 token 硬预算尚未实现。得分—预算曲线、多会话稳定性、同预算通用 agent 基线和 RL 奖励效果仍需实验。

<a id="m5"></a>

## M5：可复现且可部署的基础设施

评测需要把不可信候选的运行、可信参考与计时、设备调度和结果证据组织起来。生成与验证分开选择设备，可减少 GPU 空等；内容指纹和持久化队列可防止中断后重复花费或误用旧分。

当前有统一 TOML/CLI、显式 GPU 目标、local/Docker/Modal/HTTP 队列、Slurm、Harbor 任务导出与独立 verifier 配置；运行采用独立任务、超时清理和内容恢复。Python 3.12 与依赖版本固定，项目命名、文档、来源和版权集中维护。不同执行后端不等于不同计算平台：现有 GPU 配置仍描述 NVIDIA CUDA。

CPU/模拟测试、实际 Harbor Task schema 校验和离线 wheel 校验不等于部署成功。真实 CUDA、容器/云端、Harbor 隔离与 regrade 仍待实机验证。Trainium/NKI、通用设备接口、工具能力收据和分档报告属于设计；“缺工具可选”不能关闭已有正确性门或把未知字段写成通过。

## 逐项归档

下表覆盖第一至第七版的 **67 项**改动及本轮 **7 项**整合。每项指定一个主要 motivation，必要时附关联动机。仓库清理、命名和兼容性属于 M5 的维护工作，不包装成新的研究贡献。V6/V7 编号在本轮追踪表中补齐；不改变历史提交内容。

<!-- changes:start -->
### M1：题目与区分度的改动

| 改动 | 回应 | 关联 | 证据 |
| --- | --- | --- | --- |
| V1-03 题目范围与多样本批量生成 | 统一题集与独立样本槽位，避免各家子集不明 | M4 | [generation.py](../kb_workflows/generation.py) |
| V8-04 KernelBench-105 选集 | 接入105个scored、5个probe、140个extended的固定选集 | M3 | [kernelbench105.json](../kb_workflows/suites/kernelbench105.json) |
| V8-05 全流程统一选题 | 生成、基线、隐藏验证、评分、预算与Harbor均使用同一子集 | M5 | [common.py](../kb_workflows/common.py) |

### M2：正确性与执行可信度的改动

| 改动 | 回应 | 关联 | 证据 |
| --- | --- | --- | --- |
| V1-11 将 Verified 隐藏门接入全部验证后端 | 所有验证后端统一接受标准和隐藏门约束 | M5 | [evaluate_task.py](../kb_workflows/evaluate_task.py) |
| V2-01 严格比较输出结构、shape 和 dtype | 防止输出结构或格式不同却通过数值检查 | — | [validation.py](../src/validation.py) |
| V2-02 整数与 bool 精确比较 | 索引和逻辑结果不能套用浮点相对容差 | — | [validation.py](../src/validation.py) |
| V2-03 核对 NaN、Inf 的位置与符号 | 防止非有限值位置或符号不同被掩盖 | — | [validation.py](../src/validation.py) |
| V2-04 隔离参考与候选输入 | 原地修改不污染另一方输入 | — | [validation.py](../src/validation.py) |
| V2-05 保存参考输出快照 | 防止复用输出缓冲区覆盖参考答案 | — | [eval.py](../src/eval.py) |
| V2-06 对齐 forward 随机种子 | 同步随机计算条件，减少顺序造成的假失败 | — | [eval.py](../src/eval.py) |
| V3-02 同步墙钟计时 | 覆盖主机调用与全部设备工作完成时间 | M3 | [measurement.py](../kb_workflows/measurement.py) |
| V3-03 变化输入与逐次输出验证 | 不让缓存固定输入答案取得虚假稳态成绩 | M3 | [measurement.py](../kb_workflows/measurement.py) |
| V5-01 参考前后配对检查 | 用调用顺序和输出快照检测参考缓存与缓冲污染 | — | [reliability.py](../src/reliability.py) |
| V5-02 隔离源码全局字典与后端设置 | 防止同名定义和后端设置污染参考计算 | M5 | [eval.py](../src/eval.py) |
| V5-03 三个种子重新生成隐藏输入 | 多种子重建隐藏输入，不只验证一份随机实例 | — | [eval.py](../src/eval.py) |
| V5-04 按输入域限定的数值压力测试 | 只对已核对输入域添加数值压力，避免非法输入 | — | [reliability.py](../src/reliability.py) |
| V5-05 无效验证不能算通过 | 空测试、参考异常与零次检查不能冒充通过 | M3 | [eval.py](../src/eval.py) |
| V6-11 验证与 reward | 单独 verifier 重新判断产物，异常奖励为零 | M5 | [harbor_verifier.py](../kb_workflows/harbor_verifier.py) |
| V8-03 研究证据归档与重建 | 保留逐实现审查依据与来源摘要，不执行缺依赖的原型 | M1, M3 | [headroom_evidence.py](../scripts/headroom_evidence.py) |

### M3：可解释且可比较的性能指标的改动

| 改动 | 回应 | 关联 | 证据 |
| --- | --- | --- | --- |
| V1-12 基线记录接入批量流程 | 在实际验证设备上记录对应精度的参考基线 | M5 | [engine.py](../kb_workflows/engine.py) |
| V1-13 子集分析、pass@k 与统一结果归档 | 按声明题集和样本数归档分析，缺失不消失 | M1 | [engine.py](../kb_workflows/engine.py) |
| V3-01 默认四变体基线 | 用正确且最快的真实框架变体替代弱基线 | M2 | [baseline_variants.py](../kb_workflows/baseline_variants.py) |
| V3-04 冷成本与 K 次摊销 | 将构建、初始化和首次执行成本摊到声明调用次数 | M2 | [measurement.py](../kb_workflows/measurement.py) |
| V3-05 原始样本和测量信息 | 保留原始时间样本与设备环境供复核 | M5 | [measurement.py](../kb_workflows/measurement.py) |
| V3-06 区分旧计时与新结果 | 防止旧计时协议与新墙钟结果混用 | M5 | [config.py](../kb_workflows/config.py) |
| V4-01 失败和缺失样本计入分母 | 固定分母，防止只平均成功或最快样本 | — | [paper_analysis.py](../kb_workflows/paper_analysis.py) |
| V4-02 按 K 摊销成本评分 | 按调用预算比较成本并用区间排除未分离的小收益 | — | [paper_analysis.py](../kb_workflows/paper_analysis.py) |
| V4-03 兼容性和测量证据检查 | 计分前核对源码、设备、精度和测量证据 | M2 | [paper_analysis.py](../kb_workflows/paper_analysis.py) |
| V4-04 分层统计与区间 | 分层分析能力并保留题目与样本不确定性 | M1 | [paper_analysis.py](../kb_workflows/paper_analysis.py) |
| V4-06 旧分析与榜单不能绕过证据检查 | 旧结果展示也必须遵守当前证据门 | M5 | [common.py](../kb_workflows/common.py) |
| V5-06 分阶段诊断与逐项收据 | 区分编译、数值正确性、有效计时和计分资格 | M2 | [engine.py](../kb_workflows/engine.py) |
| V6-06 结果页面 | 结果页面使用实际硬件与采样声明 | M5 | [generate_leaderboard.py](../scripts/generate_leaderboard.py) |
| V7-04 明确选择硬件 | 显式硬件选择，避免未知设备被当成 Hopper | M5 | [config.py](../kb_workflows/config.py) |
| V8-06 分层来源与探针隔离 | 启发式I/F/D标注来源；probe和extended分组且不标为主榜资格 | M1 | [paper_analysis.py](../kb_workflows/paper_analysis.py) |

### M4：现实的Agent回路与预算的改动

| 改动 | 回应 | 关联 | 证据 |
| --- | --- | --- | --- |
| V7-07 真实 agent loop | 接入真实 DefaultAgent 而非仅有多次生成脚本 | — | [agent_loop.py](../kb_workflows/agent_loop.py) |
| V7-08 One-shot | 定义严格一次调用且没有执行反馈的比较协议 | — | [agent_models.py](../kb_workflows/agent_models.py) |
| V7-09 有界反馈 | 有界反馈保留最快开发正确候选，后续失败不丢结果 | — | [agent_loop.py](../kb_workflows/agent_loop.py) |
| V7-10 开发与最终评测分离 | 诊断与最终隐藏分分离，profiler 时间不计分 | M2 | [profile_task.py](../kb_workflows/profile_task.py) |
| V7-11 资源与来源记录 | 记录可观测成本与重试历史，未知不算免费 | M3 | [experiment.py](../kb_workflows/experiment.py) |
| V7-12 模式比较 | 按设定和预算分组，修改源码丢失生成归属 | M3 | [experiment.py](../kb_workflows/experiment.py) |
| V7-13 模型与后端 | 模型与执行后端共享协议，外部 agent 不伪装受控循环 | M5 | [agent_models.py](../kb_workflows/agent_models.py) |
| V7-14 Coding-agent skill | 现成 coding agent 可直接完成任务且会话成本不可观测时单列 | M5 | [SKILL.md](../.claude/skills/kernelbench-pro/SKILL.md) |

### M5：可复现且可部署的基础设施的改动

| 改动 | 回应 | 关联 | 证据 |
| --- | --- | --- | --- |
| V1-01 统一批量入口 | 一套配置贯穿实验阶段，降低手工交接成本 | M4 | [cli.py](../kb_workflows/cli.py) |
| V1-02 生成与最终验证独立选择后端 | 生成设备与可信验证设备分别选择 | M4 | [engine.py](../kb_workflows/engine.py) |
| V1-04 多 GPU 调度与并发限制 | 单卡串行、跨卡并行，减少资源竞争 | — | [engine.py](../kb_workflows/engine.py) |
| V1-05 Docker 生成与验证 | 在容器中运行生成和验证，提供可部署环境 | M2 | [Dockerfile](../Dockerfile) |
| V1-06 Slurm 集群提交 | 使用集群分配的设备运行批量任务 | — | [batch_run.sbatch](../batch_run.sbatch) |
| V1-07 Modal 云端执行 | 云端执行与回收内核、日志 | — | [modal_backend.py](../kb_workflows/modal_backend.py) |
| V1-08 持久化 HTTP 验证队列 | 持久化远端任务，重启后继续处理 | M4 | [queue_worker.py](../kb_workflows/queue_worker.py) |
| V1-09 带内容指纹的断点恢复 | 按内容恢复，防止旧证据冒充新结果 | M3 | [common.py](../kb_workflows/common.py) |
| V1-10 批量任务的超时与失败处理 | 明确超时失败并清理任务，禁止复用旧内核 | M3 | [transports.py](../kb_workflows/transports.py) |
| V1-14 旧入口兼容与配置预览 | 保留入口兼容并允许无副作用预览 | — | [legacy.py](../kb_workflows/legacy.py) |
| V4-05 扩大恢复指纹覆盖 | 评测代码变化使旧运行和基线失效 | M3 | [common.py](../kb_workflows/common.py) |
| V5-07 新协议参与恢复指纹 | 可靠性协议升级使旧检查缓存失效 | M2 | [common.py](../kb_workflows/common.py) |
| V6-01 论文与旧实验材料 | 清除上游宣传和失效实验材料，维护来源 | — | [NOTICE.md](../NOTICE.md) |
| V6-02 重复生成材料 | 删除重复快照，输入盲化由可信配置动态生成 | M2 | [INPUT_BLIND_GENERATION.md](../docs/INPUT_BLIND_GENERATION.md) |
| V6-03 仓库入口 | 让仓库入口描述本项目实际安装和能力 | — | [README.md](../README.md) |
| V6-04 维护方式 | 维护流程与本仓库一致，并保留许可证和版权 | — | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| V6-05 项目名称 | 统一项目、包、镜像和任务标识 | — | [pyproject.toml](../pyproject.toml) |
| V6-07 忽略规则 | 保留应审阅源码，排除凭据和运行产物 | — | [.gitignore](../.gitignore) |
| V6-08 使用中的文件 | 保留运行中的题库和结果接口，避免清理破坏评测 | M3 | [paper_analysis.py](../kb_workflows/paper_analysis.py) |
| V6-09 统一生成后端 | 外部 agent 产物回到原生样本与评分流程 | M4 | [harbor.py](../kb_workflows/harbor.py) |
| V6-10 统一任务导出 | 同一配置导出可重放任务 | M4 | [harbor.py](../kb_workflows/harbor.py) |
| V6-12 GPU 与恢复 | GPU lane 和 Harbor 配置纳入调度与恢复 | M3 | [harbor.py](../kb_workflows/harbor.py) |
| V6-13 Python 升级 | 固定 Python 与依赖版本以复现实验环境 | — | [pyproject.toml](../pyproject.toml) |
| V6-14 文档 | 文档同步当前运行方式和真实验证范围 | — | [HARBOR.md](../docs/HARBOR.md) |
| V7-01 文件配置 | 用配置统一硬件目标与工作流参数 | — | [config.py](../kb_workflows/config.py) |
| V7-02 GPU 预设 | 将 GPU 型号映射集中为可维护数据 | M3 | [gpu_profiles.toml](../kb_workflows/gpu_profiles.toml) |
| V7-03 自定义目标 | 接受自定义架构而非扩充固定白名单 | — | [gpu.py](../kb_workflows/gpu.py) |
| V7-05 后端与子任务 | 子任务保存展开配置，消除宿主文件路径依赖 | — | [generation.py](../kb_workflows/generation.py) |
| V7-06 集群与恢复 | 集群尊重配置且硬件变化触发恢复失效 | M3 | [batch_run.sbatch](../batch_run.sbatch) |
| V8-01 五条动机与逐项追踪 | 将历史改动归到具体问题，并校验覆盖完整性 | M1, M2, M3, M4 | [MOTIVATION.md](../docs/MOTIVATION.md) |
| V8-02 统一论文与能力状态 | 已实现与研究目标使用同一套名称和预算，不将设计写成实验结果 | M1, M2, M3, M4 | [PAPER.md](../research/headroom/PAPER.md) |
| V8-07 选集与参考版本证据 | 记录manifest摘要并检查参考快照，阻止静默任务漂移 | M3 | [suites.py](../kb_workflows/suites.py) |
<!-- changes:end -->

## 设计与实现的差额

这部分是论文与实现取并集后仍需完成的工作。每条都保留，避免未来只读论文或只读代码时遗漏；它们没有因为写入本文就变成已实现。之前搁置的 stash 独立保留，不作为下面任何条目的实现证据。

<!-- roadmap:start -->
| 设计 | Motivation | 尚缺什么 |
| --- | --- | --- |
| R-01 生产难题与反向轨道 | M1 | 约41道设计、数值设计、实现和backward题需参考程序、测试、基线与见证 |
| R-02 多dtype任务族与显存缩放 | M1 | 现有运行精度选项不等于已审阅的逐题dtype契约或按显存自动实例化 |
| R-03 选集区分度重新验证 | M1 | 原7模型数据缺失；归档输出分别涉及104和105题，不能据此确认精确排名复现 |
| R-04 FP64数值契约与精度门 | M2 | 需逐题oracle及阈值校准，当前仍用既有dtype容差；误差预算轨道尚未实现 |
| R-05 参考退化审计与输入校准 | M2 | 先诊断全零/小输出，再按合法域和语义修复；不能统一放大全部输入 |
| R-06 针对算子的输入族与隐藏形状 | M2 | 目前只有29道L1数值压力，隐藏多种子不等于隐藏shape或stride覆盖 |
| R-07 模式、状态、参数更新与输入敏感性 | M2 | 需要逐题train/eval、BN、参数可变性与输入域契约；不能生搬原型 |
| R-08 自写内核执行归因与库轨道 | M2 | Torch算子汇总不证明来源；20%和95%阈值未校准，不作为默认资格 |
| R-09 执行完整性与缓存审计 | M2 | NaN预填、同对象原地更新、吞吐对照、运行环境拒绝规则尚未完整接入 |
| R-10 内存与竞态工具 | M2 | compute-sanitizer及题目允许非确定性的规则需GPU复现，重复一致不证明无竞态 |
| R-11 交替计时与噪声底 | M3 | 当前为独立基线和候选样本的近似均值区间；分位数范围不是置信区间 |
| R-12 独立最终重测 | M3 | 冻结后的正式验证已有；按当前评分策略独立复测所有获胜结果的完整证据门仍待设计实现 |
| R-13 扩展基线组合 | M3 | 框架等价重写与每个compile变体的Graph组合需逐题验证；目前默认仅四变体 |
| R-14 设备实测上限与边界证书 | M3 | 不将历史H100规格表ledger当成当前设备校准结果；需实测和见证 |
| R-15 Headroom分层评分与设计证书 | M3 | 当前归一分母为ln(1.25)，不是含SOL_target的论文公式；D标签不是设计证明 |
| R-16 完整性能归因 | M3 | busy时间需并集；FLOPs/bytes与A/E/O拆分、每引擎计数器、launch_only门待实现 |
| R-17 样本聚合消融 | M3 | 当前所有声明槽位求均值；论文原提议的中位数、最快样本与均值需同预算对照 |
| R-18 设备时间与token硬预算 | M4 | 现有墙钟和候选上限已实现；模型token及纯GPU时间未硬限，费用为调用间估算 |
| R-19 得分预算曲线与通用agent基线 | M4 | 当前协议和轨迹已记录；多预算点、多会话及同预算真实模型实验待完成 |
| R-20 RL训练与奖励验证 | M4 | Harbor reward接入不等于完成训练；需红队、排序稳定性与持出集实验 |
| R-21 Trainium/NKI与设备通用接口 | M5 | caps/calibrate/flush/time/attribute是目标设计；当前计算后端仍为NVIDIA CUDA |
| R-22 工具能力收据与可选分层 | M5 | 逐字段not_measured及L0/L1/L2要求尚未实现；不得因此降级已有隐藏门和证据门 |
| R-23 实机隔离与重评分实验 | M5 | Harbor独立verifier配置已接入；GPU、容器网络、沙箱边界、升级后regrade尚待实测 |
<!-- roadmap:end -->

## 维护约定

[motivation_map.json](motivation_map.json)保存对应关系、状态、原因及仓库证据入口。新增功能时，同时更新功能清单、映射与论文中的能力状态；用 `python -m scripts.check_motivations` 检查覆盖，并用 `--write` 重建本文的两张表。论文中的目标公式或平台名称不自动升级实现状态；每条已实现记录必须指向实际文件。
