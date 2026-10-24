# KernelBench Pro：Motivation 与实现对照

本文是项目动机和范围的统一入口；[论文](../research/headroom/PAPER.md)沿用同样的五条主线，[功能清单](FEATURE_INVENTORY.md)保留逐版历史。实现、研究设计和实验结果分别记录。现有CUDA工作流与新增原生Neuron/NKI轨道共享设备接口；Neuron仅完成软件与CPU/模拟回归，实机兼容性和性能未验证。

<a id="m1"></a>

## M1：题目与区分度

大量相似题消耗评测预算，而少量需要算法设计的题在总分中被稀释。压缩题集可以降低运行成本，但不能据此声称已经提升区分度。新增生产难题、反向任务和数值格式设计，才有可能测出不同于常规实现优化的能力。

当前支持 Level 1–3 范围、多样本，以及从研究包接入的 KernelBench-105 固定选集：105 道主集、5 道审查探针、140 道扩展题。三种角色独立运行。I/F/D 是研究者给出的暂定分类，不是算法最优性或可达性证书；原始输入尚未按论文的退化审计重新校准。用法和选择规则见[研究索引](../research/headroom/README.md)。

第十二版已新增独立Level 5的Production-41：12设计、6数值设计、12固定格式与11反向题；支持逐角色dtype、合法输入与隐藏形状、按设备容量的确定性显存计划。11个反向家族有FP64 gradcheck与有限差分回归，四轨分开评分；软件实现不等于GPU可达性或区分度实验已完成。原250题名义配方、候选前资格集合和阈值拟合代码已实现，设备实测未运行；第十三版已实现当前协议下的选集分析与独立研究runner，实际区分度实验按用户范围不运行。历史 7 模型原始结果未在压缩包中提供；论文声称的精确排名一致性不能在本仓库独立重现。

<a id="m2"></a>

## M2：正确性与执行可信度

“原随机输入上接近参考”不足以证明程序实现了题意。输出格式、离散结果、非有限值、数值量级、状态和执行路径都可能让错误实现通过；更多反馈还会增加针对评测器优化的机会。

当前实现严格的结构/shape/dtype 比较、整数与 bool 精确比较、NaN/Inf 对齐、输入副本和输出快照、RNG 配对、参考前后调用检查、源码命名空间和后端设置恢复。隐藏验证重新使用三个种子；29 道经过输入域核对的 L1 题增加零值、放大和离群压力。墙钟计时包含输出消费和设备同步，每个计时输出也要校验。

新增生产题还使用相对RMS/逐元素预算，逐题拒绝全零候选；参考比较门不冒充独立FP64自资格。

这些能力分别减少特定错误，不能合并成“完整反作弊通过”。第九版新增显式严格模式，覆盖17道L1的小规模附加实例：FP64契约、信号诊断、合法输入族、隐藏奇数尺寸/stride、模式状态与缓存检查。第二十三版支持注册状态的重命名/融合视图和逐原生实例复核；第二十四版为原L1全部100题增加独立FP64算式与实际输入复核。第二十六版增加原L1参考专用名义值配方校准，第二十八版扩展到原L2全部100题的独立复合数值算式，第二十九版补齐原L2/L3来源绑定的形状族并显式修订旧概率/标签变换。第三十五版加入实际权重抵消、GELU/极值边界和按组保真的offset/存储别名，原250题每种子共2551例，完成R-06的有限输入族软件范围。第三十六版将独立数值入口补至原250题，L3采用冻结的逐层图与专用模型算式。第三十七版补齐全250题名义配方与固定前向RNG下的输出变化审计，校准计时使用调用前真值票据。第三十八版在候选前冻结合格参考集合、排除账本及样本分母，完成R-05有限软件范围。第三十九版完成有无名义配方的逐次原生计时数值门与状态/RNG时序，绑定评分恢复。第四十版实现参考专用阈值拟合、分离种子验证、固定上限和运行前冻结，完成R-04有限软件范围。执行来源和性能计数/剩余硬预算仍待完成；任意状态表示/内部信号传播不作为通用证明。研究包的审查记录保留为**源码审查结论**；1,574 份候选和反例没有在这里上 GPU 复测。原型中的统一扰动、跳过非有限参考、固定自写时间占比阈值均未直接纳入默认评测。

<a id="m3"></a>

## M3：可解释且可比较的性能指标

加速比依赖所用基线、计时协议、冷启动摊销和样本选择；弱基线、噪声和只统计成功样本都会改变结论。同样的倍率也不能解释收益来自算法、访存、硬件利用率还是启动开销。

当前默认比较四个经过正确性验证的基线变体，保留原始墙钟样本、设备与软件信息、冷成本及 K 次摊销结果。评分要求当前源码和测量证据匹配，按声明题目和样本计入失败；先按题求均值，再分层宏平均，并报告 bootstrap 区间。I/F/D 可由规格文件或选集暂定分类提供。不同设备、协议和预算分开比较。

聚合报告已新增同一批槽位上的mean/median/best-of-N消融，正式均值口径不变。当前 bounded gain 的分母为 `ln(1.25)`；没有设备实测上限时，它**不是论文提出的 headroom 分数**。第十版已实现交替计时、参考控制噪声、冻结后的独立重测、扩展Graph组合和部分框架等价改写。第十一版实现显式设备校准、工作量证书、D类独立可达性见证及可选Headroom评分；缺证据不能计分。开发profiler增加设备活动并集，A/E缺计数器时保持null，launch_only只作为诊断；正式来源归因与实际硬件计数器采集仍需实现。按用户要求不运行校准、GPU/Trainium、模型或训练实验，结果保持未采集。

<a id="m4"></a>

## M4：现实的 Agent 回路与预算

现实内核开发包含写、编译、检查、分析和修改。一次生成、多样本挑最好、多轮反馈与进化搜索必须分开声明预算；隐藏分不能成为开发反馈。

当前原生流程使用 mini-swe-agent 2.4.6，区分 one-shot 与 feedback。默认每样本最多 5 次候选、2400 秒，可配置；开发检查只看公开实例，控制器保留开发正确且最快的候选，再冻结源码执行正式标准和隐藏验证。开发 profiler 单独运行。Claude Code/LiteLLM 使用相同协议；Harbor 标为 external，当前 coding-agent skill 会话标为 interactive。

保存轨迹、每轮源码、生成回执和重试历史；可观测 token 与费用如实记录，未知为 null。费用限制仍是调用间估算。第十四版增加离线本地模型的精确总token预留、支持的API输出上限，以及local/Docker开发worker占用额度；未知或超额用量停止并拒绝计分，恢复不能重置额度。worker墙钟包含清理，不冒充纯GPU时间硬限。完整边界见[资源预算](RESOURCE_BUDGETS.md)。第十三版的得分—预算曲线和多会话计划已扩展三种新额度，保留真实回落、共同区间AUC及未知成本，默认不执行实验。第十五版新增TRL GRPO训练入口、标准开发工具、同题交替终局奖励、预算恢复、红队/稳定性审计和持出评测计划；第十六版补全审查绑定的多尺寸算法证书加分、训练权重与奖励来源追踪、预注册种子对照和持出效果分析。模拟训练不能进入正式结论，实际模型比较和RL奖励效果均未测量，详见[RL工作流](RL_WORKFLOW.md)。

<a id="m5"></a>

## M5：可复现且可部署的基础设施

评测需要把不可信候选的运行、可信参考与计时、设备调度和结果证据组织起来。生成与验证分开选择设备，可减少 GPU 空等；内容指纹和持久化队列可防止中断后重复花费或误用旧分。

当前有统一 TOML/CLI、显式 GPU 目标、local/Docker/Modal/HTTP 队列、Slurm、Harbor 任务导出与独立 verifier 配置；运行采用独立任务、超时清理和内容恢复。Python 3.12 与依赖版本固定，项目命名、文档、来源和版权集中维护。不同执行后端不等于不同计算平台：现有 GPU 配置仍描述 NVIDIA CUDA。

CPU/模拟测试、实际 Harbor Task schema 校验和离线 wheel 校验不等于部署成功。真实 CUDA、容器/云端、Harbor 隔离与 regrade 未进行实机验证。第十七版补通用设备接口、原生TorchNeuron/NKI执行、该平台基线与校准、源码/物理身份绑定及每引擎诊断；SDK部署与算子覆盖仍未做实机验证，CPU fallback与模拟计时不能建立正式设备资格。逐项证据收据和等级门已实现，L1/L2的来源/计数器证据尚未采集；“缺工具可选”不能关闭已有正确性门或把未知字段写成通过。限制和配置见[Neuron说明](NEURON.md)。

第十八版交付可选四工具流程与题目级快照重复策略，原始日志、工具/源码/设备身份、无插桩复核及派生判定进入评分和恢复；审核原子操作策略不放宽原生精度。未运行Sanitizer或设备实验，也不将工具通过等同无竞态证明，见[工具说明](TOOL_CHECKS.md)。

第十九版交付受限验证器、Harbor网络/权限配置和已有产物重评分。原生Docker核对有效配置与Linux观察，Harbor使用独立只读verifier及显式agent host策略；native/Harbor重评分保留完整旧cohort和候选，默认只预览。容器部署、逃逸/GPU独占验收和实际regrade实验未运行，收据不提供安全认证，见[隔离与重评分](ISOLATION_REGRADE.md)。

当前任务范围是完成23项设计所需的实现、配置、runner和必要回归，**不运行科研实验**。实际GPU、Trainium、真实模型和训练结果保持未运行；它们不作为本轮代码交付的前置条件。

## 逐项归档

下表覆盖第一至第七版的 **67 项**改动、第八版 **7 项**整合、第九版 **8 项**、第十版 **5 项**、第十一版 **5 项**、第十二版 **6 项**与第十三版 **6 项**设计落地。每项指定一个主要 motivation，必要时附关联动机。仓库清理、命名和兼容性属于 M5 的维护工作，不包装成新的研究贡献。V6/V7 编号在本轮追踪表中补齐；不改变历史提交内容。

<!-- changes:start -->
### M1：题目与区分度的改动

| 改动 | 回应 | 关联 | 证据 |
| --- | --- | --- | --- |
| V1-03 题目范围与多样本批量生成 | 统一题集与独立样本槽位，避免各家子集不明 | M4 | [generation.py](../kb_workflows/generation.py) |
| V8-04 KernelBench-105 选集 | 接入105个scored、5个probe、140个extended的固定选集 | M3 | [kernelbench105.json](../kb_workflows/suites/kernelbench105.json) |
| V8-05 全流程统一选题 | 生成、基线、隐藏验证、评分、预算与Harbor均使用同一子集 | M5 | [common.py](../kb_workflows/common.py) |
| V12-01 Production-41参数化题集 | 独立Level 5增加12设计、6数值设计、12固定格式与11反向任务，保留旧250题及旧Level 4模型 | — | [production_specs.py](../kb_workflows/production_specs.py), [production41.json](../kb_workflows/suites/production41.json) |
| V12-02 独立反向轨道与梯度参考 | 11个VJP参考支持外层no_grad下执行；FP64 gradcheck和方向有限差分核对梯度，轨道分别选题与评分 | M2 | [production_ops.py](../kb_workflows/production_ops.py), [test_production.py](../tests/test_production.py) |
| V12-03 逐输入角色的dtype契约 | 保留FP32缩放、bool mask与离散索引，支持FP8存储/packed INT4及E2M1软件量化；有效精度写入任务而非统一强转 | M2 | [production.py](../kb_workflows/production.py), [eval.py](../src/eval.py) |
| V13-01 研究结果的当前证据导出 | 重新核对当前源码与计分资格后导出固定槽位，合并互不重叠层级并拒绝复制运行冒充独立会话；不调用生成或评测 | M5 | [study_data.py](../kb_workflows/study_data.py) |
| V13-02 选集区分度与统计不确定性 | 支持并列秩Spearman、Kendall tau-b、差值/排名反转、任务/会话/样本bootstrap与显式二元power假设；常量相关为null，不假称旧排名已复现 | M3 | [study_statistics.py](../kb_workflows/study_statistics.py), [analyze_selection.py](../scripts/analyze_selection.py) |
| V16-01 显式尺寸实例与证据绑定 | Production-41已知正整数维度与整除/显存门；内容绑定基线、冻结、评分与比较，隐藏实例独立 | M2, M3 | [production.py](../kb_workflows/production.py), [engine.py](../kb_workflows/engine.py) |

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
| V12-04 显存计划与合法隐藏实例 | 按声明包络及设备容量确定性缩放并保存计划，私有尺寸/stride/动态范围/分页/mask及归一化近常量遵守输入域 | M1, M5 | [production.py](../kb_workflows/production.py), [build_production_tasks.py](../scripts/build_production_tasks.py) |
| V12-05 生产题数值预算与零答案门 | 所有41题使用相对RMS及逐元素比较，六个数值设计预算独立；逐题验证全零反例，参考比较与独立FP64自资格明确区分 | M1 | [validation.py](../src/validation.py), [production_specs.py](../kb_workflows/production_specs.py), [test_production.py](../tests/test_production.py) |
| V9-01 FP64数值契约 | 在17道已审阅L1的附加实例上，参考与候选均须通过RMS及全元素FP64门 | — | [numerical.py](../kb_workflows/numerical.py) |
| V9-02 参考信号诊断 | 记录输出量级、非有限值、常数与容差覆盖；至少一组合法实例有信息量 | — | [numerical.py](../kb_workflows/numerical.py) |
| V9-03 合法输入族与隐藏形状 | 限定17题的耦合维度、符号、近常数、动态范围、边界、GELU辨别输入及最终非连续stride | M1 | [strict_validation.py](../kb_workflows/strict_validation.py) |
| V9-04 模式状态与同对象缓存检查 | 审计train/eval、同名buffer、参数原地更新与同对象输入修改后的响应 | — | [strict_validation.py](../kb_workflows/strict_validation.py) |
| V9-05 进程设置审计 | 记录并恢复候选导入和执行修改的后端、默认dtype、线程和环境变量；收据不含环境变量值 | M5 | [strict_validation.py](../kb_workflows/strict_validation.py) |
| V15-03 分阶段小实例资格任务 | qualify_small在原尺寸分配前运行审查契约；源码与小实例失败先停止昂贵阶段，基线过期则在模型加载前停止 | M4 | [evaluate_task.py](../kb_workflows/evaluate_task.py), [rl_rewards.py](../kb_workflows/rl_rewards.py) |
| V17-04 CPU FP64与输入角色保护 | 独立oracle留在CPU，逐次验证保持输入角色、模型调用状态及Neuron RNG；审计冻结NKI文件和环境，禁止FP8存储降级与模拟设备计分 | M3 | [neuron_worker.py](../kb_workflows/neuron_worker.py), [strict_validation.py](../kb_workflows/strict_validation.py), [measurement.py](../kb_workflows/measurement.py), [reliability.py](../src/reliability.py) |
| V18-01 四工具独立诊断流程 | 分别运行memcheck/racecheck/synccheck/initcheck，要求完成摘要与正调用覆盖，缺工具/日志或超时保持未知，工具范围含参考及框架 | M5 | [tool_runner.py](../kb_workflows/tool_runner.py), [evaluate_task.py](../kb_workflows/evaluate_task.py) |
| V18-02 模型快照与逐位重复验证 | 逐实例复制模型和输入并重放RNG，不推进原模型；每次保留原生精度门，记录输出位摘要，缺实例或快照不给通过 | — | [determinism.py](../kb_workflows/determinism.py), [reliability.py](../src/reliability.py) |
| V18-03 审核绑定的原子操作策略 | 默认逐位一致，有限波动须有参考源码摘要、审核理由及非零界限；不放宽离散值、非有限值或原生精度 | M1 | [determinism.py](../kb_workflows/determinism.py), [generation.py](../kb_workflows/generation.py), [harbor.py](../kb_workflows/harbor.py) |
| V19-04 隔离凭据进入评分与恢复 | 重算配置/结果摘要和原生观察，基线变体与标准/隐藏/最终/配对数据不能混用image；模拟凭据拒绝正式资格，保留诊断与原始计时分离 | M3, M5 | [paper_analysis.py](../kb_workflows/paper_analysis.py), [final_verification.py](../kb_workflows/final_verification.py), [paired_task.py](../kb_workflows/paired_task.py), [tool_checks.py](../kb_workflows/tool_checks.py) |
| V20-02 CUDA编译产物与实际launch关联 | 可选CUPTI/Triton适配器关联实际cubin、模块/函数代次和成功launch；复制完整二进制摘要，卸载及句柄复用不继承来源 | M5 | [cuda_origin.py](../kb_workflows/cuda_origin.py), [test_kernel_origin.py](../tests/test_kernel_origin.py) |
| V20-05 来源诊断与正式资格边界 | R-08转为部分实现；库/Graph/Neuron和正式轨道门槛未完成，95%仅可配置诊断，不授予L1，不替换原生计时或声称算法/安全证明 | M3 | [KERNEL_ORIGIN.md](../docs/KERNEL_ORIGIN.md), [PAPER.md](../research/headroom/PAPER.md), [kernel_origin.py](../kb_workflows/kernel_origin.py) |
| V21-01 受信安装包与实际库二进制 | 候选导入前固定RECORD与逐文件哈希，只提取实际加载且前后身份一致的DSO；CUDA ELF字节匹配实际模块，缺工具/篡改/冲突/模拟不能建立库来源 | M5 | [origin_binaries.py](../kb_workflows/origin_binaries.py), [test_origin_binaries.py](../tests/test_origin_binaries.py) |
| V21-02 CUDA Graph节点与执行快照 | 关联实际创建/克隆/instantiate的节点与函数代次，每次replay保存快照；源模板销毁不抹除既有执行，未知更新或可执行对象销毁不沿用旧来源 | M3 | [origin_graphs.py](../kb_workflows/origin_graphs.py), [test_kernel_origin.py](../tests/test_kernel_origin.py) |
| V21-05 诊断容差与实际模式后置检查 | 导入前冻结配置容差，核对执行前后training/grad并记录恢复进程设置；协议v2要求后置观察，模式或环境改变保持未知 | M5 | [origin_task.py](../kb_workflows/origin_task.py), [test_kernel_origin.py](../tests/test_kernel_origin.py) |
| V22-01 执行前冻结JIT文本与源码字面量 | 在执行候选前固定函数文本和保留代码对象，不依赖运行时inspect/linecache；记录完整字符串字面量供CUDA编译输入核对 | M5 | [submission_source.py](../kb_workflows/submission_source.py), [test_compiler_origins.py](../tests/test_compiler_origins.py) |
| V22-02 实际前端依赖与审核语言primitive | 观察真实AST解析和CodeGenerator调用，逐callee核对候选/语言源码与字节码和callable闭包；外层wrapper或builtin标签不代表完整来源 | M5 | [triton_origin.py](../kb_workflows/triton_origin.py), [test_compiler_origins.py](../tests/test_compiler_origins.py) |
| V22-03 编译器产物及初始依赖冻结 | 在CUDA后端输出与真实构造器边界固定cubin、source、metadata和原始依赖收据；替换产物或事后洗白依赖无效，缓存/IR override缺证据不通过 | M3 | [triton_origin.py](../kb_workflows/triton_origin.py), [cuda_origin.py](../kb_workflows/cuda_origin.py), [test_compiler_origins.py](../tests/test_compiler_origins.py) |
| V22-05 CUDA object回溯实际模块函数 | 只提取已核对CUDA object的cubin，再匹配实际模块和函数/launch；不是整个.so都算自写，未关联Runtime注册及模拟构建不能建立实测来源 | M3 | [cpp_origin.py](../kb_workflows/cpp_origin.py), [kernel_origin.py](../kb_workflows/kernel_origin.py), [origin_task.py](../kb_workflows/origin_task.py) |
| V23-01 声明式注册状态映射 | 支持重命名、融合切片、转置与不复制存储的reshape，核对值/元数据/共享关系；映射不能豁免状态比较 | M5 | [state_contract.py](../kb_workflows/state_contract.py), [test_state_contract.py](../tests/test_state_contract.py) |
| V23-02 连续模式与累计状态检查 | 同一模型连续train/train/eval/train/eval，检查累计buffer、输出及training/grad后置状态，不只检查首次调用 | — | [state_contract.py](../kb_workflows/state_contract.py), [test_state_contract.py](../tests/test_state_contract.py) |
| V23-03 零值参数与缓存失效探针 | 分别在train/eval预热后两轮修改同一参数存储；非零偏移覆盖全零参数，共享参数只修改一次，记录参考敏感性 | — | [state_contract.py](../kb_workflows/state_contract.py), [test_state_contract.py](../tests/test_state_contract.py) |
| V23-05 状态快照与证据重算 | 拒绝共享模型/状态存储和改变快照值，逐例重算绑定摘要与模式/参数步骤；改容差/漏实例/模拟执行不能获得设备资格 | M5 | [state_contract.py](../kb_workflows/state_contract.py), [state_audit.py](../kb_workflows/state_audit.py), [common.py](../kb_workflows/common.py), [test_state_audit.py](../tests/test_state_audit.py) |
| V24-01 L1独立FP64算式与源码绑定 | 为原L1全部100题提供冻结源码/域/初始误差预算的CPU公式，不调用参考forward；原题不被改写，L2/L3缺适配不冒充覆盖 | M5 | [operator_specs.py](../kb_workflows/operator_specs.py), [operator_oracles.py](../kb_workflows/operator_oracles.py), [operator_oracles.json](../kb_workflows/operator_oracles.json), [test_operator_oracles.py](../tests/test_operator_oracles.py) |
| V24-02 卷积与池化窗口参考 | 显式逐组空间收缩与转置散射，保留stride/dilation/padding/output padding；窗口池化保留ceil边界、分母与并列索引，不调用原融合算子 | — | [operator_oracles.py](../kb_workflows/operator_oracles.py), [test_operator_oracles.py](../tests/test_operator_oracles.py) |
| V24-03 数值语义与低精度自资格反例 | 归一化/attention/loss采用显式FP64公式并保留状态、标签、epsilon和精确激活语义；固定BF16抵消反例不能因候选等于参考而通过 | — | [operator_oracles.py](../kb_workflows/operator_oracles.py), [test_operator_oracles.py](../tests/test_operator_oracles.py) |
| V24-05 参考结构与离散自资格修正 | 参考资格独立计算，oracle shape/结构/离散dtype不符时不能因没有数值叶子而通过；92题exclusive scan参考缺陷明确保留失败 | — | [numerical.py](../kb_workflows/numerical.py), [test_operator_oracles.py](../tests/test_operator_oracles.py) |
| V24-06 数值与信号证据重算 | 核对逐原生实例、冻结规格/误差/内存政策、RMS限值、元素计数和信号叶子；源码漂移/缺项/模拟不能获得设备资格，配置贯穿Harbor及评分恢复 | M5, M3 | [numerical_audit.py](../kb_workflows/numerical_audit.py), [tool_checks.py](../kb_workflows/tool_checks.py), [cli.py](../kb_workflows/cli.py), [test_numerical_audit.py](../tests/test_numerical_audit.py) |
| V25-01 Exclusive scan参考有版本修正 | 沿声明轴补零/累加/裁剪，保留负轴、singleton与空轴shape；独立移位公式和旧错误反例同时核对，不改变名义计时尺寸 | M5 | [92_cumsum_exclusive.py](../KernelBench/level1/92_cumsum_exclusive.py), [operator_oracles.py](../kb_workflows/operator_oracles.py), [test_reference_revisions.py](../tests/test_reference_revisions.py) |
| V25-02 扫描输入族与原分布保留 | 五题追加奇数/非连续/rank-3/单元素/空轴，保留原3或4分布；乘积保持信号、mask保持bool，生成器不丢新增入口 | M1 | [scan_cases.py](../kb_workflows/scan_cases.py), [89_hidden.py](../hidden_tests/level1/89_hidden.py), [90_hidden.py](../hidden_tests/level1/90_hidden.py), [91_hidden.py](../hidden_tests/level1/91_hidden.py), [92_hidden.py](../hidden_tests/level1/92_hidden.py), [93_hidden.py](../hidden_tests/level1/93_hidden.py), [test_scan_cases.py](../tests/test_scan_cases.py) |
| V25-05 空输出数值契约 | 合法空Tensor按结构/dtype检查，RMS及误差保留null，不编造测得零误差；空输出不能单独提供参考信号 | — | [numerical.py](../kb_workflows/numerical.py), [numerical_audit.py](../kb_workflows/numerical_audit.py), [test_numerical_audit.py](../tests/test_numerical_audit.py) |
| V26-01 参考专用输入配方选择 | 按固定顺序与三个种子检查参考FP64资格、信号和探针变化，保留失败尝试，不加载候选或采集速度 | — | [input_calibration_task.py](../kb_workflows/input_calibration_task.py), [input_calibration.py](../kb_workflows/input_calibration.py) |
| V26-02 L1角色保持的值分布 | 100题适配保留shape/stride/离散角色；乘积随机有界前缀、概率与三角结构受约束，隐藏工厂不套用名义配方 | — | [input_profiles.py](../kb_workflows/input_profiles.py), [test_input_calibration.py](../tests/test_input_calibration.py) |
| V26-03 冻结强数值政策与逐输出复核 | 绝对误差项收紧为零且不放宽原预算，参考和候选均须通过独立FP64；首基线、warmup及每个计时输出在计时外核对 | — | [input_calibration.py](../kb_workflows/input_calibration.py), [evaluate_task.py](../kb_workflows/evaluate_task.py), [paired_task.py](../kb_workflows/paired_task.py), [eval.py](../src/eval.py) |
| V27-01 原L1真实初始化的隐藏输入族 | 100题使用参考实际参数构造合法形状/布局，LN固定normalized_shape及卷积通道/分组不替换，计时输入不变 | — | [operator_cases.py](../kb_workflows/operator_cases.py), [test_operator_inputs.py](../tests/test_operator_inputs.py) |
| V27-02 算子边界、抵消与并列极值 | 矩阵耦合/结构、卷积池化空间、扫描mask/乘积、概率标签及attention独立处理，每种子632例且保留旧隐藏分布 | — | [operator_cases.py](../kb_workflows/operator_cases.py), [operator_input_policy.py](../kb_workflows/operator_input_policy.py) |
| V27-03 显式非有限域与空数值测量 | 五种激活追加NaN/Inf及全NaN，传播严格匹配；全非有限叶子的RMS/误差为null，不能贡献参考信号 | — | [numerical.py](../kb_workflows/numerical.py), [numerical_audit.py](../kb_workflows/numerical_audit.py), [operator_oracles.json](../kb_workflows/operator_oracles.json) |
| V27-04 转换后保留正stride布局 | 修正dtype/device转换可能压紧切片的问题，将真实正stride与离散dtype送入候选及隔离副本 | — | [eval.py](../src/eval.py), [validation.py](../src/validation.py), [test_operator_inputs.py](../tests/test_operator_inputs.py) |
| V27-07 参考近常量失格证据保留 | BN/IN固定CPU FP32反例超过既定FP64预算，保留已知问题与失败；来源变化的旧结果在分析时明确失效 | — | [operator_oracles.json](../kb_workflows/operator_oracles.json), [paper_analysis.py](../kb_workflows/paper_analysis.py), [test_operator_inputs.py](../tests/test_operator_inputs.py) |
| V28-01 全部L2来源绑定的复合程序 | 100题冻结参考SHA、图节点、构造属性、状态形状与随机节点；构造变化或未知算术不回退到参考forward | — | [composite_oracles.py](../kb_workflows/composite_oracles.py), [composite_programs.json](../kb_workflows/composite_programs.json), [composite_specs.py](../kb_workflows/composite_specs.py) |
| V28-02 复合层独立FP64算式 | Linear/卷积/池化/归一化/激活显式CPU公式，保留GELU近似、inplace别名及原dtype常量舍入，不调用融合数值forward | — | [composite_math.py](../kb_workflows/composite_math.py), [operator_oracles.py](../kb_workflows/operator_oracles.py), [test_composite_oracles.py](../tests/test_composite_oracles.py) |
| V28-03 原生随机掩码条件下的数值参考 | Dropout保持实际模式，以原dtype/shape/stride/device和seed重放掩码；FP64算式独立计算数值，外部RNG恢复 | — | [composite_oracles.py](../kb_workflows/composite_oracles.py), [numerical_audit.py](../kb_workflows/numerical_audit.py) |
| V28-07 零信号复合参考问题保留 | 83题min后clamp在固定参数下恒零，数值相等不能恢复reference_signal；源文件与总体不变，不放宽误差 | — | [operator_oracles.json](../kb_workflows/operator_oracles.json), [test_composite_oracles.py](../tests/test_composite_oracles.py) |
| V29-01 L2融合输入族与残差耦合 | 全100题600例每种子，保留学习特征、LayerNorm末轴及外部转置卷积残差几何 | — | [model_input_families.json](../kb_workflows/model_input_families.json) |
| V29-02 L3序列状态与固定形状规则 | 50题297例，区分序列与状态batch轴，三题保留固定初始化并明示原因 | — | [model_input_cases.py](../kb_workflows/model_input_cases.py) |
| V29-03 旧概率和浮点标签语义修订 | KL温度归一化及hinge只变预测；保留槽位/公式并冻结修订历史，生成器不退回旧域 | — | [reference_revisions.json](../kb_workflows/reference_revisions.json) |
| V29-04 全题逐例隐藏覆盖门 | 250题原生与Neuron按seed、双向调用、内存、布局复核，缺完整执行证据不能恢复正结果 | M5 | [operator_input_policy.py](../kb_workflows/operator_input_policy.py) |
| V30-01 双模式候选分配预填充 | 两组不同填充值检查浮点/复数/整数/bool，首模式与参考碰巧一致仍须通过第二模式 | — | [allocation_audit.py](../kb_workflows/allocation_audit.py) |
| V30-02 导入构造与实际调用作用域 | 候选全局/构造缓存与逐原生双向调用预填，参考/输入不预填，也不强制确定性算法 | — | [reliability.py](../src/reliability.py) |
| V30-03 连续扩容前缀保护与未知存储 | 只填连续扩容新逻辑尾部；不支持存储保留未知，metadata-only存储量为null | — | [allocation_audit.py](../kb_workflows/allocation_audit.py) |
| V30-06 预填充反例与覆盖范围声明 | CPU多dtype分配、局部未写与缓存、首模式碰撞和模拟进程篡改/超时；原始allocator与Graph等未知不宣称覆盖 | M5 | [test_allocation_audit.py](../tests/test_allocation_audit.py) |
| V31-01 独立同步与连续提交进程 | 两种新公开worker对照逐次同步和连续N次后同步，参考/候选交替且原计时保持 | M3 | [tool_runner.py](../kb_workflows/tool_runner.py) |
| V31-02 完整输出快照与复用缓冲区 | 相同快照消费成本保留每次复用缓冲区输出，逐例结构/形状/精度/设备与数值核对 | — | [throughput_audit.py](../kb_workflows/throughput_audit.py) |
| V31-03 连续状态随机与执行政策 | 可信工厂字节和序列种子跨模式一致，模型连续推进；根模式/no_grad/线程及设置恢复 | — | [reliability.py](../src/reliability.py) |
| V31-07 排队模拟与状态输出反例 | CPU排队/合法摊销、状态随机、复用与结构化输出、缺项篡改/超时和Neuron前端证明代码路径 | M5 | [test_throughput_audit.py](../tests/test_throughput_audit.py) |
| V32-01 全实际实例的同对象更新 | 独立worker覆盖原生公开/隐藏实例，三次复用Tensor身份并逐次验证输出，原计时保持 | — | [cache_audit.py](../kb_workflows/cache_audit.py) |
| V32-02 来源绑定的合法值配方 | L1/L2/L3和生产41题保持几何/耦合与离散角色；公开继续可信工厂，隐藏不借公开配方 | — | [cache_recipes.py](../kb_workflows/cache_recipes.py) |
| V32-03 输入存储别名与几何恢复 | 隔离重复对象/正stride/offset/广播别名、分离计算图并原地复制；几何恢复记录且未知表示不通过 | M5 | [input_reuse.py](../kb_workflows/input_reuse.py) |
| V32-04 连续状态与稳定输出快照 | 参考候选相同调用数和随机重放，不重置状态；立即快照复用输出，常量参考不误判作弊 | — | [cache_audit.py](../kb_workflows/cache_audit.py) |
| V32-05 运行设置拒绝与组合诊断修复 | 导入/构造/原生/更新调用记录并恢复设置；修复状态收据覆盖父文档而破坏后续诊断 | — | [tool_checks.py](../kb_workflows/tool_checks.py) |
| V33-01 稳定普通Tensor状态路径 | 按真实字典键/类型、列表位置和普通对象属性绑定，跳过注册参数的框架别名，不执行property | — | [state_paths.py](../kb_workflows/state_paths.py) |
| V33-02 循环模型逐题状态语义 | 源绑定L3/33持久hidden替换和放置，以及L3/41未使用初始化h0；候选可省略无用属性而不省输入 | — | [state_policies.py](../kb_workflows/state_policies.py) |
| V33-03 普通状态快照隔离与保真 | 普通属性/容器/对象的值、元数据、路径及对象/重叠存储别名保真；deepcopy不能修复断开的共享或重连失效缓存 | M5 | [state_contract.py](../kb_workflows/state_contract.py) |
| V33-04 持久状态替换与规范参数更新 | 按实际参考连续比较替换后的普通状态，支持映射注册buffer；注册参数仍要求同对象存储及非零更新 | — | [state_contract.py](../kb_workflows/state_contract.py) |
| V33-05 状态所有者模式与收据重放 | 每次调用记录并比较全部映射状态所有者的train/eval和no_grad，漏模式/改角色/缺持久状态拒绝资格 | — | [state_audit.py](../kb_workflows/state_audit.py) |
| V35-01 复合与序列专用压力政策 | 150题冻结七类角色与特征轴，追加异号对、激活分支、孤立极值和坐标不对称输入 | M5 | [model_input_families.json](../kb_workflows/model_input_families.json) |
| V35-02 当前实际权重的成对抵消 | 100个直接线性/标准卷积/循环入口按权重比值构造有界输入，保留参数与偏置并记录实际系数摘要 | M5 | [model_stress.py](../kb_workflows/model_stress.py) |
| V35-03 GELU与有限极值判别输入 | 15个激活题覆盖±32/±32752，26/88加±2.699，池化/极值归约检查实际窗口内的单个边界极值 | M5 | [operator_cases.py](../kb_workflows/operator_cases.py) |
| V35-04 偏移与共享存储隐藏输入 | 250题追加非零offset，33个合法多输入题追加重叠存储，初始化守卫并在分配前核对更大包络 | M5 | [input_layouts.py](../kb_workflows/input_layouts.py) |
| V35-05 按存储组隔离与转移 | 原生/Neuron转换、双向比较、状态/重复/数值检查保留stride、offset、别名和重复对象；输出快照保持独立 | M5 | [input_groups.py](../src/input_groups.py), [reliability.py](../src/reliability.py), [neuron_worker.py](../kb_workflows/neuron_worker.py) |
| V36-01 全L3来源与构造冻结 | 50题绑定来源、完整module树、标量属性、状态几何与算式；保留精度/模式/物理参数布局变化，拒绝缩小正式构造 | M5 | [model_programs.json](../kb_workflows/model_programs.json) |
| V36-02 独立attention与encoder算术 | 显式投影、分头、mask、softmax、残差、前馈及norm，保持源码batch-first方向；原内置随机概率为零 | M5 | [model_layers.py](../kb_workflows/model_layers.py) |
| V36-03 循环网络与普通状态算术 | RNN逐步递推、LSTM/GRU门公式、多层双向和h/c返回，保持33的状态copy舍入；不调用融合RNN | M5 | [model_math.py](../kb_workflows/model_math.py) |
| V36-04 Swin、GPT、VLAD与SSD算术 | 窗口/shift、两类patch合并、余弦/因果/ReLU注意力、ghost残差及块内外状态递推分别实现，保持未使用层语义 | M5 | [model_math.py](../kb_workflows/model_math.py) |
| V36-05 meta准入与条件随机掩码 | meta仅给形状/布局；全尺寸输入/状态与attention workspace先准入，原生值无关mask保留dtype/stride/RNG；闭包指向隔离meta模块 | M5 | [model_oracles.py](../kb_workflows/model_oracles.py) |
| V37-01 L2/L3来源绑定的名义菜单 | 七类模型角色共731个配方声明，保留实际特征/显式初始状态、尺寸、参数和隐藏分布 | M5 | [model_input_profiles.py](../kb_workflows/model_input_profiles.py) |
| V37-02 按存储组生成值与极值修复 | 共享backing仅采样一次并保持视图/精度；七个L1极值题增加正态配方，修复FP16 uniform max恒值 | M5 | [input_profiles.py](../kb_workflows/input_profiles.py) |
| V37-03 固定前向RNG与输出变化审计 | 三个输入种子共享前向seed 42，同时要求输入探针变化和输出至少两次不同；恒零/非零常量不合格 | M5 | [input_calibration.py](../kb_workflows/input_calibration.py) |
| V37-04 全题参考专用校准器 | L1公式/L2复合/L3模型算式统一准入，保留程序与mask收据；候选、时间样本和误差放宽不参与选择 | M5 | [input_calibration_task.py](../kb_workflows/input_calibration_task.py) |
| V37-05 当前状态的调用前真值票据 | 计时参考调用前计算独立真值并恢复RNG，绑定参考/输入计划；模型路径拒绝缺票据的事后计算 | M5 | [reference_oracles.py](../kb_workflows/reference_oracles.py), [input_calibration.py](../kb_workflows/input_calibration.py) |
| V38-01 冻结参考资格与排除清单 | 按预定原题范围核对完整参考校准，固定第一个合格配方、每个失败配方的原因和case摘要；候选和时间不参与选择 | M3 | [reference_population.py](../kb_workflows/reference_population.py) |
| V38-02 未解决失败不缩小题集 | 缺项、未测完菜单、资源/worker错误、模拟或来源漂移拒绝冻结；没有合格题目不能产生空集得分 | M5 | [reference_population.py](../kb_workflows/reference_population.py), [test_reference_population.py](../tests/test_reference_population.py) |
| V39-01 不依赖值校准的逐次FP64门 | 启用数值验证后，原250题的原生计时自动采用独立实际输入真值，不再要求输入配方；实际冷调用/双向正确性输出、预热与每次正式输出均核对参考和候选 | M5 | [timed_numerics.py](../kb_workflows/timed_numerics.py), [evaluate_task.py](../kb_workflows/evaluate_task.py), [input_calibration.py](../kb_workflows/input_calibration.py), [eval.py](../src/eval.py) |
| V39-02 逐输出持久化数值证据 | 记录每次输出的类别、序号、RMS/全元素门、参考信号、内存与模型/随机mask收据；独立核对覆盖数、固定政策和源身份，不虚构前向seed | M3 | [timed_numerics.py](../kb_workflows/timed_numerics.py) |
| V39-03 配对共用调用前状态真值 | 每轮先计算一次参考调用前真值，再逐一验证三方输出；验证不额外推进普通Tensor/注册状态或随机流，保持原同步墙钟样本 | M3 | [paired_task.py](../kb_workflows/paired_task.py), [neuron_worker.py](../kb_workflows/neuron_worker.py) |
| V40-01 固定预算内的误差使用比例 | 从实际存储值与未舍入FP64真值计算逐叶RMS和逐元素最大预算使用比例；原上限失格不能通过拟合加宽 | M3 | [numerical.py](../kb_workflows/numerical.py) |
| V40-02 参考专用阈值拟合与验证 | 原L1/L2/L3按题目/dtype使用三个拟合种子和三个不重叠验证种子，固定构造/前向RNG、隔离状态，不加载候选、不采性能时间 | M5 | [precision_calibration_task.py](../kb_workflows/precision_calibration_task.py) |
| V40-03 五项预算统一缩放与禁止重拟合 | 固定margin4/floor1e-6且factor最多1，五项预算一同收紧；独立种子验证失败保留原政策，离散输出精确且不虚构浮点拟合 | M3 | [precision_calibration.py](../kb_workflows/precision_calibration.py) |
| V41-01 Graph模板核函数参数更新 | 观察cuGraphKernelNodeSetParams的v1/v2参数前缀及匹配CUPTI事件，更新模板代次，保留已实例化可执行Graph | — | [origin_graphs.py](../kb_workflows/origin_graphs.py), [KERNEL_ORIGIN.md](../docs/KERNEL_ORIGIN.md) |
| V41-02 可执行Graph更新与克隆别名 | 可执行Graph核参数更新只修改目标实例的后续提交；原模板、其他实例和旧调用快照保留，克隆节点沿原节点身份更新 | M3 | [origin_graphs.py](../kb_workflows/origin_graphs.py), [KERNEL_ORIGIN.md](../docs/KERNEL_ORIGIN.md) |
| V41-03 Graph更新事件与代次复核 | v4来源收据重放API、函数与context、节点/实例事件、前后代次和调用边界，拒绝回滚节点或借用其他实例的更新 | M3 | [kernel_origin.py](../kb_workflows/kernel_origin.py), [KERNEL_ORIGIN.md](../docs/KERNEL_ORIGIN.md) |
| V41-04 含糊更新失效与有界重放 | 成功却缺事件、错绑事件或失败伴更新事件使后续关联未知；原始代次及失效边界保留，有界重放超额拒绝而不省略证据 | — | [origin_graphs.py](../kb_workflows/origin_graphs.py), [KERNEL_ORIGIN.md](../docs/KERNEL_ORIGIN.md) |

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
| V11-01 设备校准接口与原始收据 | 提供显式设备探针与cache flush/time接口，绑定物理UUID、驱动、软件和评测器，重算原始样本速率；本轮只实现不运行校准 | M5 | [calibration.py](../kb_workflows/calibration.py), [calibrate_device.py](../scripts/calibrate_device.py) |
| V11-02 工作量证书与独立可达性见证 | 核对参考源码、输入、精度和校准；D标签要求算法说明与至少两次独立标准/隐藏记录，历史暂定标签不能代替证书 | M2 | [headroom.py](../kb_workflows/headroom.py), [export_reachability.py](../scripts/export_reachability.py) |
| V11-03 Headroom评分与证据失效 | 按SOL_target归一稳态收益，缺证书置零并保留固定分母；另报摊销bounded gain，外部规格/校准修改使旧资格失效 | — | [paper_analysis.py](../kb_workflows/paper_analysis.py), [common.py](../kb_workflows/common.py) |
| V11-04 设备时间并集与性能解释 | 计算重叠活动并集与kernel数，来源未知不猜测；缺工作计数器时A/E为null，启动收益须有测量解释，暂为诊断 | — | [performance.py](../kb_workflows/performance.py), [profile_task.py](../kb_workflows/profile_task.py) |
| V9-07 严格评分门与协议隔离 | 显式严格模式下，未适配题、失败检查和缺失收据拒绝资格；后端设置与等级参与任务及比较指纹 | M2 | [paper_analysis.py](../kb_workflows/paper_analysis.py) |
| V9-08 样本聚合消融 | 对同一批声明槽位计算mean、median、best-of-N，保留失败零分和成本记录，正式分数不变 | M4 | [aggregation.py](../kb_workflows/aggregation.py) |
| V10-01 交替计时与噪声对照 | 同进程平衡参考/候选/参考控制顺序，逐次验证输出，按重复进程与配对样本重采样估计倍率区间 | M2 | [paired_task.py](../kb_workflows/paired_task.py) |
| V10-02 冻结后独立最终重测 | 冻结源码按新种子在独立任务重测标准和隐藏正确性，保留全部重复，正式分数使用重测而非最快重复 | M2 | [final_verification.py](../kb_workflows/final_verification.py) |
| V10-03 扩展基线组合与可信改写 | 增加compile default/max-autotune各自Graph组合；3道归一化题增加源码指纹约束的框架等价改写 | — | [baseline_variants.py](../kb_workflows/baseline_variants.py) |
| V10-04 附加证据与旧评分门 | 最终和交替测量必须匹配当前源码/基线/配置，收据文件及冻结内容参与旧评分资格复核 | M5 | [common.py](../kb_workflows/common.py) |
| V14-04 预算恢复与计分资格 | 限额槽位只允许一次可审计尝试，匹配完成记录可resume；重算账本、绑定历史与汇总，修改预算证据使新旧计分门均失效 | M4, M5 | [generation.py](../kb_workflows/generation.py), [paper_analysis.py](../kb_workflows/paper_analysis.py), [common.py](../kb_workflows/common.py) |
| V15-04 同题交替终局奖励 | 同题共享repeat种子并交替启动独立标准/隐藏/配对计时任务，核对设备UUID；严格资格分与弱raw-speedup消融分开，失败槽位为零 | M2, M4 | [rl_rewards.py](../kb_workflows/rl_rewards.py) |
| V16-02 多尺寸算法证书 | 至少5尺寸且跨度16倍，固定源码/设备/变体和形状规律，重做原生资格与配对测量，斜率须匹配当前D/SOL审查且显著低于参考 | M2, M4 | [scaling_certificate.py](../kb_workflows/scaling_certificate.py), [SCALING_CERTIFICATES.md](../docs/SCALING_CERTIFICATES.md) |
| V16-06 按训练种子的效果分析 | 重算持出资格分，报告配对训练种子效应、分层区间与双侧检验；失败检查未知槽位给上下界，模拟数据不能支持正式效果 | M4 | [rl_effectiveness.py](../kb_workflows/rl_effectiveness.py), [test_rl_workflow.py](../tests/test_rl_workflow.py) |
| V17-03 Neuron基线与配对重测 | eager/compile及4题源码绑定的框架库改写，最快正确基线、平衡配对与冻结重测复用；平台单独分组 | M2, M5 | [neuron_worker.py](../kb_workflows/neuron_worker.py), [baseline_variants.py](../kb_workflows/baseline_variants.py) |
| V17-05 Neuron校准与物理身份 | 绑定序列号/core/驱动/SDK并保留原始copy、分dtype GEMM和启动代理，明确同步墙钟口径、无FP64速率和未知缓存驱逐 | M5 | [neuron_device.py](../kb_workflows/neuron_device.py), [calibration.py](../kb_workflows/calibration.py) |
| V17-06 设备轨迹与每引擎诊断 | 收集框架trace/NEFF/NTFF与哈希；显式单位/窗口的tensor/vector/scalar/DMA并集，不把指令数冒充kernel数或capture跨度冒充host墙钟 | M5 | [neuron_profile.py](../kb_workflows/neuron_profile.py), [neuron_device.py](../kb_workflows/neuron_device.py), [analyze_device_trace.py](../scripts/analyze_device_trace.py) |
| V18-05 诊断与性能采样分离 | 保留首次无插桩原生/配对样本与冷启动，诊断共用原请求预算；四工具全通过才给内存工具证据，L2另需重复一致性 | M2, M4 | [tool_runner.py](../kb_workflows/tool_runner.py), [tool_checks.py](../kb_workflows/tool_checks.py), [evidence.py](../kb_workflows/evidence.py) |
| V19-07 重评分来源与测量凭据 | 单独标记artifact-regrade与零新模型调用，绑定原始测量和基线后再评分，保留失败阶段成本；历史费用不重新认证，模拟数据不能建立资格 | M4, M5 | [regrade.py](../kb_workflows/regrade.py), [experiment.py](../kb_workflows/experiment.py), [common.py](../kb_workflows/common.py) |
| V20-03 来源时间并集与未知重放 | 逐调用重算保守自写时间占比，重叠活动只计一次，未知/Graph/内存活动不获自写信用；丢记录、错绑和模式变化不通过 | M2 | [kernel_origin.py](../kb_workflows/kernel_origin.py), [test_kernel_origin.py](../tests/test_kernel_origin.py) |
| V21-03 原生内存操作与双轨道诊断 | 成功原生copy/set及Graph节点记为runtime而非custom，逐调用分别给出自写和库轨道诊断；重叠并集与未知保留，正式资格仍未开放 | M2 | [kernel_origin.py](../kb_workflows/kernel_origin.py), [cuda_origin.py](../kb_workflows/cuda_origin.py), [test_kernel_origin.py](../tests/test_kernel_origin.py) |
| V25-06 诊断设置贯穿冻结与比较 | 统一状态/数值/内存/工具/重复选项，补齐早期摘要遗漏；修改开关或额度改变生成/最终冻结/研究/RL契约并分组报告 | M2, M5 | [tool_checks.py](../kb_workflows/tool_checks.py), [experiment.py](../kb_workflows/experiment.py), [final_verification.py](../kb_workflows/final_verification.py), [study_data.py](../kb_workflows/study_data.py), [test_reference_revisions.py](../tests/test_reference_revisions.py) |
| V26-04 新基线与恢复证据 | 输入记录进入基线指纹，缺当前配方、执行设备、工厂调用或逐输出检查数不能恢复/评分，旧分布基线须重测 | — | [engine.py](../kb_workflows/engine.py), [paper_analysis.py](../kb_workflows/paper_analysis.py), [input_calibration.py](../kb_workflows/input_calibration.py) |
| V31-05 吞吐差异复核与性能资格 | 逐轮10%标记可来自正常并行，不抹除原生答案；未消除差异不获性能分，分母不减 | M2 | [paper_analysis.py](../kb_workflows/paper_analysis.py) |
| V34-01 全题来源绑定的基线清单 | 291题冻结参考SHA、11个数学改写和491个适用后端策略；没有额外改写的题目保留基础组合 | M5 | [baseline_rewrites.json](../kb_workflows/baseline_rewrites.json) |
| V34-02 注意力与归一化库改写 | SDPA/GQA、LayerNorm/var_mean、残差RMS及行缩放/范数形式保持名义域、mask、eps、affine和输出格式 | M2 | [baseline_rewrites.py](../kb_workflows/baseline_rewrites.py) |
| V34-03 卷积布局与后端偏好作用域 | channels-last/3d、cuDNN benchmark、cuBLASLt偏好按调用设置恢复，转换与设置成本计时；不冒充GPU来源 | M2 | [baseline_policies.py](../kb_workflows/baseline_policies.py) |
| V34-04 每个扩展项的Graph对应 | 六基础组合加每项库/后端Graph版本，完整声明2750个分题变体；不支持捕获和错误输出保留失败 | — | [baseline_variants.py](../kb_workflows/baseline_variants.py) |
| V38-03 候选前冻结固定评分分母 | 在生成、导出和评测前保存集合及样本数，已有候选/结果不能补选集合；冻结后失败和缺失样本保留零分 | M4 | [reference_population.py](../kb_workflows/reference_population.py), [generation.py](../kb_workflows/generation.py), [engine.py](../kb_workflows/engine.py) |
| V39-05 评分恢复要求计时数值证据 | 基线记录/缓存、候选恢复、配对、最终重测和普通评分核对完整逐次收据；缺检查、改预算或篡改误差计数不能取得正分 | M2, M5 | [tool_checks.py](../kb_workflows/tool_checks.py), [engine.py](../kb_workflows/engine.py), [paper_analysis.py](../kb_workflows/paper_analysis.py) |
| V40-06 模型生成前冻结阈值政策 | 冻结校准文件、类别、范围与样本数，候选出现后不能补选/替换阈值；预算、最终冻结和研究计划识别文件变化 | M2, M4 | [precision_calibration.py](../kb_workflows/precision_calibration.py), [reference_population.py](../kb_workflows/reference_population.py) |

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
| V13-03 多预算多会话与通用agent计划 | 生成原生one-shot/反馈及Harbor coding-agent网格，保留真实候选限制；默认不执行，显式执行时经过原生配置校验与resume | M5 | [study_plan.py](../kb_workflows/study_plan.py), [plan_study.py](../scripts/plan_study.py) |
| V13-04 得分预算曲线与共同区间AUC | 独立预算点保留失败分母与非单调结果，区间覆盖任务/会话/样本；未知成本不免费，AUC不外推或取最好历史 | M3 | [budget_curves.py](../kb_workflows/budget_curves.py) |
| V13-05 缓存与外部agent用量核算 | 补总token和样本执行时间，区分Claude与Harbor cache口径；多步缺项保持未知，外部agent收据绑定结果文件，不冒充GPU账单 | M5 | [agent_models.py](../kb_workflows/agent_models.py), [harbor_budget.py](../kb_workflows/harbor_budget.py), [experiment.py](../kb_workflows/experiment.py) |
| V14-01 调用前token预留与用量结算 | 调用前预留本地精确总token或API输出额度，已知用量返还，缺失或超额停止并拒绝计分；不支持的后端显式拒绝 | M3 | [agent_models.py](../kb_workflows/agent_models.py), [resource_budgets.py](../kb_workflows/resource_budgets.py) |
| V14-02 离线本地模型适配 | 直接使用实际chat template输入ID与单条生成序列；只加载本地safetensors且禁止远端代码，权重和tokenizer内容绑定恢复与评分 | M5 | [local_model.py](../kb_workflows/local_model.py) |
| V14-03 开发worker占用预留 | local/Docker父进程记录包含启动/清理的开发worker墙钟，评测和profile均先预留；超额/未知不合格，不冒充物理GPU抢占或账单 | M5 | [resource_budgets.py](../kb_workflows/resource_budgets.py), [transports.py](../kb_workflows/transports.py) |
| V14-05 token与设备占用预算网格 | 研究计划和曲线增加总token、输出token和开发worker占用三种额度；按真实后端能力预检，保持其他预算分组，默认不执行 | M3 | [study_plan.py](../kb_workflows/study_plan.py), [budget_curves.py](../kb_workflows/budget_curves.py) |
| V15-01 冻结训练计划与持出划分 | 显式任务、参考与隐藏源码哈希、审查family、初始模型及实现指纹固定计划；公开参考沿用input-blind策略，默认不执行 | M1, M5 | [rl_plan.py](../kb_workflows/rl_plan.py) |
| V15-02 原生GRPO与标准开发工具 | 固定TRL 1.8.0及DAPO损失变体，接入one-shot/反馈工具、原生最佳候选选择和离线模型；持出题不进入训练callback | M2 | [rl_training.py](../kb_workflows/rl_training.py) |
| V15-06 奖励重评分与可靠性审计 | 预声明源码案例与重复计划，重算ICC/奖励排序/攻击接受率和公共最终差；缺数据与基础设施故障不能冒充抵御攻击，统计依赖假设显式保留 | M2, M3 | [rl_evaluation.py](../kb_workflows/rl_evaluation.py), [rl_audit.py](../kb_workflows/rl_audit.py) |
| V15-07 持出评测计划与CLI | 持出checkpoint资产冻结并复用原生one-shot/反馈任务及独立会话；提供默认非执行CLI，模型改变在运行前拒绝 | M1, M5 | [rl_evaluation.py](../kb_workflows/rl_evaluation.py), [study_plan.py](../kb_workflows/study_plan.py), [rl_workflow.py](../scripts/rl_workflow.py) |
| V16-03 有界证书奖励 | 默认关闭；高资格候选才可请求证书，加分后封顶1，失败或恒定倍率优化无算法加分，证书墙钟记入训练预算 | M2, M3 | [rl_rewards.py](../kb_workflows/rl_rewards.py), [rl_plan.py](../kb_workflows/rl_plan.py) |
| V16-05 预注册奖励对照与持出评测 | 执行前冻结两奖励组与全部种子、优化器/数据/预算及评测会话；当前lineage/权重不符或缺作业不能缩小研究，默认不执行 | M1, M5 | [rl_effectiveness.py](../kb_workflows/rl_effectiveness.py), [study_plan.py](../kb_workflows/study_plan.py), [rl_workflow.py](../scripts/rl_workflow.py) |
| V17-07 NKI提示与coding skill | 保留ModelNew接口并提示现代NKI/native注册，skill复用设备配置与vendor解释器；客户端模型环境与可信验证环境分开 | M5 | [generation.py](../kb_workflows/generation.py), [SKILL.md](../.claude/skills/kernelbench-pro/SKILL.md), [trainium2.toml](../configs/trainium2.toml) |
| V38-08 内层模型超时清理嵌套进程 | 模型CLI自身先超时时先清理包括独立session的后代再终止父进程，避免外层控制器已无法找到孤儿；直接与外层墙钟两种超时均做CPU子进程回归，未启动模型 | M5 | [agent_models.py](../kb_workflows/agent_models.py), [generation.py](../kb_workflows/generation.py), [test_agent_loop.py](../tests/test_agent_loop.py) |

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
| V11-05 Verifier配置与精度设置恢复 | 可信规格和校准只进入Harbor verifier；开发子任务保留原评分模式；补齐半精度归约开关恢复与修改审计 | M2 | [harbor.py](../kb_workflows/harbor.py), [generation.py](../kb_workflows/generation.py), [validation.py](../src/validation.py) |
| V12-06 生产题全流程与来源恢复 | 选题/轨道贯穿原生与Harbor，prompt保留数学语义并移除生成器；生成参考与manifest可重建核对，协议参与恢复指纹 | M4 | [engine.py](../kb_workflows/engine.py), [generation.py](../kb_workflows/generation.py), [harbor.py](../kb_workflows/harbor.py) |
| V13-06 研究CLI与静态图导出 | 提供计划/选集/预算分析入口、可选Matplotlib静态图；分析实现单独指纹，残缺选集必须显式标记，文档与功能清单同步 | M1, M4 | [analyze_budget_curves.py](../scripts/analyze_budget_curves.py), [analyze_selection.py](../scripts/analyze_selection.py), [pyproject.toml](../pyproject.toml) |
| V9-06 逐项证据等级 | 实现current/L0/L1/L2要求、逐字段状态、缺工具值为null；按真实检查决定达到的等级 | M2, M3 | [evidence.py](../kb_workflows/evidence.py) |
| V10-05 重测恢复与失败分母 | 恢复保留未变样本的新种子和完成记录，仅源码改变的样本重跑；独立重测失败进入正确性pass@k分母 | M3, M4 | [final_verification.py](../kb_workflows/final_verification.py) |
| V15-05 训练预算与checkpoint恢复 | 生成前持久化预留group，中断仍扣额度；checkpoint绑定文件、step及奖励前缀；父进程监督总墙钟含清理，缺账本拒绝重置 | M4 | [rl_training.py](../kb_workflows/rl_training.py) |
| V16-04 训练来源与父进程凭据 | 绑定最终权重、计划、全部奖励/证书文件和当前优化路径；恢复保留丢弃更新的成本，nonce与锁拒绝旧完成文件和并发监督 | M4 | [rl_lineage.py](../kb_workflows/rl_lineage.py), [rl_training.py](../kb_workflows/rl_training.py) |
| V17-01 通用设备适配协议 | 定义caps/calibrate/flush/time/attribute，CUDA与Neuron两实现，导入及配置预览不加载SDK | M3 | [devices.py](../kb_workflows/devices.py), [calibration.py](../kb_workflows/calibration.py) |
| V17-02 Neuron配置与任务执行 | device节、独立vendor解释器、物理core分配及LNC=1，接入标准/隐藏/最终与开发任务，拒绝CUDA镜像和旧XLA混用 | M2, M4 | [config.py](../kb_workflows/config.py), [neuron_worker.py](../kb_workflows/neuron_worker.py), [transports.py](../kb_workflows/transports.py) |
| V18-04 原始工具证据与负结果恢复 | 绑定工具版本/二进制、逐进程日志、阶段nonce/源码/物理设备；重算派生结果，已记录失败可恢复，篡改收据失效 | M2 | [tool_checks.py](../kb_workflows/tool_checks.py), [engine.py](../kb_workflows/engine.py), [final_verification.py](../kb_workflows/final_verification.py) |
| V18-06 统一配置与跨流程指纹 | 配置可选工具/重复预算和题目策略；所有transport复用CLI，策略内容进入生成、最终冻结、比较、RL和旧评分指纹，Neuron拒绝CUDA工具 | M4 | [config.py](../kb_workflows/config.py), [experiment.py](../kb_workflows/experiment.py), [rl_plan.py](../kb_workflows/rl_plan.py), [a6000-safety.toml](../configs/a6000-safety.toml) |
| V19-01 受限验证器配置与镜像固定 | 显式CUDA/Docker受限策略，每请求把标签解析为不可变image ID；清空worker环境，使用非root、只读根和私有编译缓存，不把不支持的transport降格当作满足策略 | M2 | [isolation.py](../kb_workflows/isolation.py), [restricted_transport.py](../kb_workflows/restricted_transport.py), [a6000-restricted.toml](../configs/a6000-restricted.toml) |
| V19-02 实际权限与资源观察 | 核对前后Docker配置与Linux capability/no-new-privileges/seccomp、namespace和cgroup；单设备/主机资源有明确上限，缺观察不通过，不认证GPU独占或驱动逃逸安全 | M2 | [isolation.py](../kb_workflows/isolation.py), [isolated_worker.py](../kb_workflows/isolated_worker.py) |
| V19-03 结果交接与容器清理 | 只读请求和独立结果目录，退出后只读有大小界限的普通JSON；拒绝symlink/FIFO/非有限值，确认删除自己创建的容器并处理create客户端超时 | M2, M4 | [restricted_transport.py](../kb_workflows/restricted_transport.py), [isolation.py](../kb_workflows/isolation.py) |
| V19-05 Harbor网络与只读验证器 | 独立verifier禁止联网，受限Compose隔离可写路径并复用bundled tests；agent执行使用显式host allowlist，实际SDK schema已校验，容器部署未运行 | M4, M2 | [harbor.py](../kb_workflows/harbor.py), [harbor_verifier.py](../kb_workflows/harbor_verifier.py), [test_isolation.py](../tests/test_isolation.py) |
| V19-06 冻结候选的原生重评分 | 固定完整题目与样本槽位、候选/历史轨迹及当前验证输入；缺样本保留，新基线另存，仅调用当前验证阶段，旧运行不改动 | M3 | [regrade.py](../kb_workflows/regrade.py), [regrade_artifacts.py](../scripts/regrade_artifacts.py) |
| V19-08 Harbor产物重评分与默认预览 | 固定SDK/CLI/当前verifier及原trial/产物，用上游NopAgent重评分；核对隐式artifact覆盖、完整cohort和源码哈希，默认只生成计划/预览 | M4 | [harbor_regrade.py](../kb_workflows/harbor_regrade.py), [regrade_artifacts.py](../scripts/regrade_artifacts.py), [test_harbor_regrade.py](../tests/test_harbor_regrade.py) |
| V20-01 冻结候选源码与JIT身份 | 用实际内容哈希文件和保留代码对象加载候选，支持Triton源码检查与future导入；相同文件名或重新编译的相等对象不继承身份 | M2 | [submission_source.py](../kb_workflows/submission_source.py), [eval.py](../src/eval.py), [test_kernel_origin.py](../tests/test_kernel_origin.py) |
| V20-04 独立来源诊断与默认预览 | 独立公开输入worker保留请求/源码/设备/模式和逐次输出验证，CLI默认零设备预览；结果与日志不覆盖原实验 | M4 | [origin_task.py](../kb_workflows/origin_task.py), [audit_kernel_origin.py](../scripts/audit_kernel_origin.py), [test_kernel_origin.py](../tests/test_kernel_origin.py) |
| V21-04 CUPTI活动关联与原始提取记录 | 补齐DRIVER/RUNTIME activity生成correlation，运行时父关联唯一时才使用；二进制工具/日志按原字节绑定，兼容非UTF-8记录 | M2 | [cuda_origin.py](../kb_workflows/cuda_origin.py), [origin_binaries.py](../kb_workflows/origin_binaries.py), [test_kernel_origin.py](../tests/test_kernel_origin.py) |
| V21-06 来源任务进入队列验证入口 | 队列接收origin_audit并强制候选源码，验证器摘要包含新增入口；CPU请求校验覆盖预览生成的实际payload，不运行设备 | — | [queue_worker.py](../kb_workflows/queue_worker.py), [common.py](../kb_workflows/common.py), [test_kernel_origin.py](../tests/test_kernel_origin.py) |
| V22-04 CUDA C++真实构建观察 | 核对Torch适配器RECORD、实际loader调用点、CUDA字面量、Ninja/source/object和工具/环境；未知header、编译wrapper或未完成构建不赋予来源 | M2 | [cpp_origin.py](../kb_workflows/cpp_origin.py), [test_compiler_origins.py](../tests/test_compiler_origins.py) |
| V22-06 来源协议v3与实际SDK结构兼容 | 新增前端/产物必需凭据及真实GPUTarget结构摘要；CPU检查实际AST到MLIR与构造器，不调用GPU或编译工具，仍不授予正式L1 | M2 | [kernel_origin.py](../kb_workflows/kernel_origin.py), [triton_origin.py](../kb_workflows/triton_origin.py), [test_compiler_origins.py](../tests/test_compiler_origins.py), [KERNEL_ORIGIN.md](../docs/KERNEL_ORIGIN.md) |
| V23-04 原生逐实例状态复核 | 可选state_validation经过统一后端与Harbor，独立复核覆盖实际标准/隐藏输入，共享原请求预算，诊断不替换性能样本 | M2, M3 | [state_audit.py](../kb_workflows/state_audit.py), [evaluate_task.py](../kb_workflows/evaluate_task.py), [tool_checks.py](../kb_workflows/tool_checks.py), [test_state_audit.py](../tests/test_state_audit.py) |
| V24-04 实际输入复核与内存准入 | 可选numerical_validation覆盖实际原生标准/隐藏输入，分配前核对主机内存估计包络；不缩小原尺寸，独立复核不替换性能样本 | M2, M3 | [numerical_audit.py](../kb_workflows/numerical_audit.py), [config.py](../kb_workflows/config.py), [evaluate_task.py](../kb_workflows/evaluate_task.py), [test_numerical_audit.py](../tests/test_numerical_audit.py) |
| V25-03 任务修订与共享依赖绑定 | 记录参考/隐藏前后哈希、连续版本链和共享生成器；worker/队列执行前核对，新结果记录修订，评分恢复拒绝无当前修订的正结果 | M2 | [reference_revisions.py](../kb_workflows/reference_revisions.py), [reference_revisions.json](../kb_workflows/reference_revisions.json), [test_reference_revisions.py](../tests/test_reference_revisions.py) |
| V25-04 选集运行快照升级与历史档案 | KernelBench-105清单v2保留250题身份/105主选/5探针/140扩展及seed，显式基底加修订；原研究CSV不重写为新结果 | M1, M3 | [suites.py](../kb_workflows/suites.py), [kernelbench105.json](../kb_workflows/suites/kernelbench105.json), [test_reference_revisions.py](../tests/test_reference_revisions.py) |
| V25-07 修订任务跨入口回归 | CPU核对轴/空边界/精度提升、实际隐藏入口、队列与全部dispatch阶段、Harbor私有文件及旧证据拒绝，不运行设备实验 | M2 | [test_reference_revisions.py](../tests/test_reference_revisions.py), [test_scan_cases.py](../tests/test_scan_cases.py), [generate_hidden_inputs.py](../scripts/generate_hidden_inputs.py), [REFERENCE_REVISIONS.md](../docs/REFERENCE_REVISIONS.md) |
| V26-05 多设备分布复用与实测身份 | 分别记录校准设备和执行设备；复用配方须在本设备再次数值资格，不借用性能或可达性证据 | — | [input_calibration.py](../kb_workflows/input_calibration.py), [neuron_worker.py](../kb_workflows/neuron_worker.py), [test_input_calibration.py](../tests/test_input_calibration.py) |
| V26-06 校准文件贯穿冻结与任务桥接 | 文件内容进入最终冻结、研究/RL及重新评分；单题子任务和Harbor私有文件保持一致，公开提示词说明分布及政策 | — | [generation.py](../kb_workflows/generation.py), [harbor.py](../kb_workflows/harbor.py), [final_verification.py](../kb_workflows/final_verification.py), [rl_plan.py](../kb_workflows/rl_plan.py), [study_data.py](../kb_workflows/study_data.py) |
| V26-07 轻量校准预览与准入回归 | 默认预览无SDK/输入/目录副作用，执行显式开启；内存超限在变换/公式前拒绝，CPU反例覆盖证据删改及所有计时输出 | — | [calibrate_inputs.py](../scripts/calibrate_inputs.py), [test_input_calibration.py](../tests/test_input_calibration.py), [INPUT_CALIBRATION.md](../docs/INPUT_CALIBRATION.md) |
| V27-05 原生隐藏执行与覆盖复核 | CUDA及Neuron共同追加案例，逐例记录有效输入/种子/双向调用；评分恢复拒绝漏例、改布局、重复标签或缺记录 | — | [operator_input_policy.py](../kb_workflows/operator_input_policy.py), [reliability.py](../src/reliability.py), [neuron_worker.py](../kb_workflows/neuron_worker.py), [tool_checks.py](../kb_workflows/tool_checks.py) |
| V27-06 输入内存准入与全流程冻结 | 默认256MiB输入估计在分配前检查，超限不缩尺寸；计划和额度贯穿队列、生成、Harbor、最终冻结及研究比较 | — | [config.py](../kb_workflows/config.py), [engine.py](../kb_workflows/engine.py), [cli.py](../kb_workflows/cli.py), [harbor.py](../kb_workflows/harbor.py) |
| V27-08 全L1元数据与原生入口回归 | 真实100题初始化meta检查及CPU算式、私有seed、NaN信号、Neuron模拟、评分/RL/恢复收据回归，不运行设备实验 | — | [test_operator_inputs.py](../tests/test_operator_inputs.py), [_operator_evidence.py](../tests/_operator_evidence.py), [OPERATOR_INPUTS.md](../docs/OPERATOR_INPUTS.md) |
| V28-04 复合收据与缺项拒绝 | 逐例复核冻结程序、节点、采样模式/概率/布局/计数/摘要，缺失随机证据或图身份不匹配不能取得数值资格 | — | [composite_specs.py](../kb_workflows/composite_specs.py), [numerical_audit.py](../kb_workflows/numerical_audit.py), [test_composite_oracles.py](../tests/test_composite_oracles.py) |
| V28-05 meta状态副本与分配前准入 | 不先复制真实GPU权重，先汇总meta输入/状态/节点元素估算主机包络，超限不执行FP64或采样也不缩题 | — | [composite_oracles.py](../kb_workflows/composite_oracles.py), [numerical_audit.py](../kb_workflows/numerical_audit.py) |
| V28-06 独立数值入口扩至L1及L2 | numerical_validation支持原200题实际标准/隐藏输入，Source/程序/额度贯穿现有请求与部署；L3仍拒绝泛化fallback | — | [config.py](../kb_workflows/config.py), [operator_specs.py](../kb_workflows/operator_specs.py), [common.py](../kb_workflows/common.py), [pyproject.toml](../pyproject.toml) |
| V28-08 全L2数学与名义形状回归 | 100题小配置CPU数值与原构造/名义shape的meta检查，禁用融合forward、两类低精度、随机收据及准入反例 | — | [_composite_cases.py](../tests/_composite_cases.py), [test_composite_oracles.py](../tests/test_composite_oracles.py), [COMPOSITE_NUMERICS.md](../docs/COMPOSITE_NUMERICS.md) |
| V29-05 形状精度独立重放与来源预检 | L2/L3从冻结清单重建shape/dtype，CLI/生成/队列/Harbor同源预检且无SDK导入 | M2 | [engine.py](../kb_workflows/engine.py) |
| V29-06 全模型元数据与域迁移反例 | 150原工厂/初始化meta与CPU小输入、数值控制、Neuron模拟覆盖；超限分配、篡改与旧域回归 | M2 | [test_model_input_cases.py](../tests/test_model_input_cases.py) |
| V30-04 独立进程关闭缓存并同步报错 | 两模式各用父进程配置的fresh CUDA正确性worker，直接后续复核，不替换原生性能时间 | M2 | [tool_runner.py](../kb_workflows/tool_runner.py) |
| V30-05 来源调用收据与冻结部署 | 复核设备/来源/模式/环境/导入构造/实际case-seed-order，开关进入冻结、研究、RL与Harbor | M2 | [tool_checks.py](../kb_workflows/tool_checks.py) |
| V31-04 名义元数据和存储准入 | 参考meta输出推导后原尺寸存储准入，未知/超额不换实例、不缩题或降N | M2 | [throughput_audit.py](../kb_workflows/throughput_audit.py) |
| V31-06 请求冻结与多后端私有部署 | 来源/设备/模式/调用/seed/字节/时间重放，统一冻结/研究/RL/队列/Harbor且无隐藏诊断 | M2 | [tool_checks.py](../kb_workflows/tool_checks.py) |
| V32-06 存储准入和私有部署冻结 | 全底层存储和暂存估计先准入；源码/设备/原生覆盖/更新收据进入队列、Harbor及最终/研究/RL合同 | M2 | [engine.py](../kb_workflows/engine.py) |
| V32-07 全族回归与原始范围验收 | 小输入验证L1/L2/生产族、缓存反例及私有流程；R09按最初四项验收，不增加万能GPU证明条件 | M2 | [test_cache_audit.py](../tests/test_cache_audit.py) |
| V33-06 全题状态政策与私有工作流 | 291题source绑定政策贯穿CLI/生成/队列/Harbor/最终，普通状态白名单和未使用角色不借可执行回调 | M2 | [engine.py](../kb_workflows/engine.py) |
| V33-07 原构造元数据与循环状态回归 | 291题仅构造meta模型、不调用输入工厂；小循环状态/对象缓存、别名、快照和收据反例验证有限软件范围 | M2 | [test_ordinary_state.py](../tests/test_ordinary_state.py) |
| V34-05 基线政策收据与复核门 | 新变体的配方/来源/策略/Graph收据进入恢复、配对与评分；缺项失格但保留原始时间和诊断 | M3 | [engine.py](../kb_workflows/engine.py) |
| V34-06 原生预检与多后端边界 | SDK轻量预览列逐题变体和总数，原生/最终/私有Harbor共用清单；Neuron保留自己的非CUDA组合 | M3 | [cli.py](../kb_workflows/cli.py) |
| V34-07 数学格式与后端模拟回归 | 三个dtype的合法值配方、291题清单、2D/3D布局、设置异常恢复、原生错误拒绝与收据删改；不运行设备实验 | M2 | [test_baseline_recipes.py](../tests/test_baseline_recipes.py) |
| V35-06 实际副本交付收据 | 重算有效输入及双向四次隔离副本布局，缺API/压紧/拆存储/错dtype不能取得别名资格；实际权重收据拒绝meta替代 | M2 | [operator_input_policy.py](../kb_workflows/operator_input_policy.py), [test_model_input_cases.py](../tests/test_model_input_cases.py) |
| V35-07 别名与低精度反例回归 | 全250题转移及隔离的meta布局、小输入权重抵消/参数不变、GELU近似与BF16反例、边界极值及收据删改 | M2 | [test_input_layouts.py](../tests/test_input_layouts.py) |
| V36-06 模型证据与统一数值入口 | L3配置、SDK轻量预览、逐原生实例收据、内部FX图和mask复核贯通已有控制worker、评分/恢复与打包 | M2 | [model_specs.py](../kb_workflows/model_specs.py), [numerical_audit.py](../kb_workflows/numerical_audit.py), [config.py](../kb_workflows/config.py), [pyproject.toml](../pyproject.toml) |
| V36-07 小构造对照与参考缺陷保留 | 36个小配置的两模式算式、融合forward禁用、50题双meta计划、随机/可选状态/收据反例；35省略状态的FP32默认不被暗改 | M2 | [test_model_oracles.py](../tests/test_model_oracles.py), [_model_cases.py](../tests/_model_cases.py), [MODEL_NUMERICS.md](../docs/MODEL_NUMERICS.md) |
| V37-06 原生与Neuron计时顺序统一 | 基线/候选/配对/Neuron逐对票据，回调重放trial RNG，转移/快照/Graph输入保持别名；诊断不替换原始时间 | M2 | [paired_task.py](../kb_workflows/paired_task.py), [eval.py](../src/eval.py), [reliability.py](../src/reliability.py), [measurement.py](../kb_workflows/measurement.py), [neuron_worker.py](../kb_workflows/neuron_worker.py), [evaluate_task.py](../kb_workflows/evaluate_task.py), [baseline_variants.py](../kb_workflows/baseline_variants.py) |
| V37-07 角色、退化与连续状态回归 | 150原构造的所有配方仅做meta；小CPU输入验证三精度别名、常量/全零、固定mask、连续状态/RNG和模拟Neuron三阶段 | M2 | [test_model_input_profiles.py](../tests/test_model_input_profiles.py), [INPUT_CALIBRATION.md](../docs/INPUT_CALIBRATION.md) |
| V38-04 工作流与恢复共用题集 | 生成、基线、最终/配对验证、评分和研究分析统一选择器；全校准文件及清单绑定执行/评分/恢复指纹 | M2, M3 | [cli.py](../kb_workflows/cli.py), [paper_analysis.py](../kb_workflows/paper_analysis.py), [common.py](../kb_workflows/common.py), [final_verification.py](../kb_workflows/final_verification.py) |
| V38-05 父集合与单题委托绑定 | 私有子任务保留父清单与当前题校准，运行收据校验父摘要；Harbor、RL、预算、研究和重新评分保留完整排除账本与实际子范围 | M2, M4 | [input_calibration.py](../kb_workflows/input_calibration.py), [harbor.py](../kb_workflows/harbor.py), [rl_plan.py](../kb_workflows/rl_plan.py), [regrade.py](../kb_workflows/regrade.py) |
| V38-06 无执行的集合冻结命令 | 读取已有校准预览，显式write只写新清单；TOML/CLI路径和任务委托可配置，不导入设备SDK或启动实验 | M2 | [freeze_reference_population.py](../scripts/freeze_reference_population.py), [config.py](../kb_workflows/config.py), [INPUT_CALIBRATION.md](../docs/INPUT_CALIBRATION.md) |
| V38-07 资格集合的失败与传递回归 | 小CPU与不计分的序列化夹具覆盖缺项/篡改/未解决、固定分母、零分保留、私有委托、运行收据和轻量命令；不建立实测资格 | M2, M3 | [test_reference_population.py](../tests/test_reference_population.py) |
| V39-04 原生与Neuron数值路径一致 | 未配置输入配方也启用Neuron双向/首次/预热/正式/配对强门；数值模式的FP32非组合基线关闭TF32，不改原尺寸和误差政策 | M2 | [neuron_worker.py](../kb_workflows/neuron_worker.py), [evaluate_task.py](../kb_workflows/evaluate_task.py), [paired_task.py](../kb_workflows/paired_task.py) |
| V39-06 验证失败时恢复真实调用后RNG | 数值或其他验证回调抛错时，finally仍恢复候选实际调用后的CPU/CUDA/Neuron随机状态，不把参考诊断的消耗带出计时器 | M2 | [measurement.py](../kb_workflows/measurement.py) |
| V39-07 无配方的模型与状态计时回归 | 小CPU原构造验证L1/L2/L3票据、旧容差漏错、首/预热/正式覆盖、收据重封篡改和连续随机状态；模拟Neuron验证三阶段，不建立设备测量资格 | M2 | [test_timed_numerics.py](../tests/test_timed_numerics.py), [TIMED_NUMERICS.md](../docs/TIMED_NUMERICS.md) |
| V40-04 阈值预览与单次配方应用 | 预览不导入设备SDK或创建输出，显式execute才运行参考；TOML/CLI路径、有效政策和缺项拒绝接入，名义配方仅变换一次 | M2 | [calibrate_precision.py](../scripts/calibrate_precision.py), [config.py](../kb_workflows/config.py), [cli.py](../kb_workflows/cli.py) |
| V40-05 校准与执行设备分离核对 | 校准设备保持来源身份，同硬件/软件类别可复用契约且逐次当前设备FP64自资格；异类拒绝，运行记录不能冒充新设备已校准 | M2 | [precision_calibration.py](../kb_workflows/precision_calibration.py), [numerical_audit.py](../kb_workflows/numerical_audit.py), [timed_numerics.py](../kb_workflows/timed_numerics.py) |
| V40-07 私有阈值贯通Agent与重评分 | 子任务携带本题/dtype记录，公开提示只给有效政策；Harbor/RL/评分恢复、研究导出及重新评分绑定内容和实际执行收据 | M2, M4 | [generation.py](../kb_workflows/generation.py), [harbor.py](../kb_workflows/harbor.py), [rl_plan.py](../kb_workflows/rl_plan.py), [regrade.py](../kb_workflows/regrade.py), [tool_checks.py](../kb_workflows/tool_checks.py) |
| V40-08 拟合反例与批量传递回归 | 小CPU/模拟与格式夹具覆盖三dtype、原复合/循环构造、验证失败不重拟合、原上限失败、篡改、设备类别/多GPU、冻结、Harbor/研究/重新评分和全250题SDK轻量预览 | M2, M3 | [test_precision_calibration.py](../tests/test_precision_calibration.py), [PRECISION_CALIBRATION.md](../docs/PRECISION_CALIBRATION.md) |
| V41-05 Graph更新反例与主机ABI核对 | CPU夹具覆盖两类API的v1/v2、克隆、相邻实例、失败/缺事件、重复更新和篡改；主机C编译核对供应商头文件布局，不调用CUDA | M2 | [test_graph_updates.py](../tests/test_graph_updates.py), [KERNEL_ORIGIN.md](../docs/KERNEL_ORIGIN.md) |
<!-- changes:end -->

## 设计与实现的差额

这部分保留全部23项设计，分别标记待实现、部分实现或代码已实现；代码交付和真实设备/模型实验分别验收。之前搁置的 stash 独立保留，不作为下面任何条目的实现证据。

<!-- roadmap:start -->
| 设计 | 状态 | Motivation | 已交付与剩余范围 |
| --- | --- | --- | --- |
| R-01 生产难题与反向轨道 | 代码已实现 | M1 | 已实现12设计、6数值设计、12固定格式和11反向题的参考、私有输入、数值门及原生/Harbor全流程；11个反向家族有FP64 gradcheck与有限差分，实际性能/可达性实验不运行，库baseline优化归R-13 |
| R-02 多dtype任务族与显存缩放 | 代码已实现 | M1 | 已实现逐输入角色dtype、独立精度实例、FP8/packed格式语义和按声明显存/设备容量确定性缩放；计划与角色绑定恢复，内存为显式模型而非任意候选OOM保证 |
| R-03 选集区分度重新验证 | 代码已实现 | M1 | 已实现当前证据导出、全量/选集成对比较、并列秩相关、分层会话/样本bootstrap、冗余与假设power设计；缺失/复制数据拒绝，原7模型数据仍缺，按用户范围不运行重新验证实验 |
| R-04 FP64数值契约与精度门 | 代码已实现 | M2 | 原250题独立FP64公式/模型算法、实际正确性/冷调用/逐次计时门及评分恢复已交付；现有按题目/dtype/校准设备的参考专用阈值拟合，固定上限内五项收紧、分离种子验证不重拟合，保存原始证据并在候选前冻结。当前设备必须独立自资格，记录和预算/Agent/Harbor/RL/研究/重新评分绑定。有限软件范围已完成；用户要求不运行设备/模型/阈值实验，未产生实测阈值或泛化/硬件认证 |
| R-05 参考退化审计与输入校准 | 代码已实现 | M2 | 原250题来源绑定名义配方与参考专用FP64选择、输入/输出变化审计、固定RNG、程序/mask收据和新基线已接入；统一入口在候选前冻结合格参考集合和全部失败/排除账本，未解决记录不能缩小题集，冻结后所有候选失败保留分母。子任务/预算/评分/恢复/重新评分绑定父集合。有限软件范围已完成，阈值校准归R-04；按要求不运行设备/模型实验 |
| R-06 针对算子的输入族与隐藏形状 | 代码已实现 | M2 | 原250题来源绑定输入族每种子2551例：角色/形状、异号与实际权重抵消、GELU/有限极值/索引、offset与33题共享存储；转移和双向隔离副本逐项核对布局与实际系数。291题另有生产角色及模式/参数/重复检查。有限模板不保证内部传播或全部错误检出，原尺寸只做meta，按用户要求不运行实验 |
| R-07 模式、状态、参数更新与输入敏感性 | 代码已实现 | M2 | 原始逐题模式/BN/参数/输入域软件契约已实现：291来源绑定政策、注册与普通Tensor状态/容器/对象、参考对应替换、逐状态所有者模式、同对象规范参数更新及R-06/R-09输入变化。L3/33持久hidden与41未使用self.h0分别审核；任意表示/非线性解码不作为通用证明，候选须保留可映射规范参数；按用户要求不跑实验 |
| R-08 自写内核执行归因与库轨道 | 部分实现 | M2 | 已实现CUDA/Triton实际AST/审核primitive依赖与产物冻结、CUDA C++ object、受信wheel cubin、Graph快照及两类核参数API的v1/v2更新重绑定诊断；未关联Runtime/其他loader、动态编译器、整体/其他Graph更新、CUkernel、Neuron与正式资格门继续实现，未知不通过 |
| R-09 执行完整性与缓存审计 | 代码已实现 | M2 | 原始四项已提供可选软件路径：候选分配双预填、全实际正确性实例同对象更新、公开同步/连续提交对照、候选导入/构造/调用运行设置拒绝与恢复。来源/设备/覆盖收据贯穿评分恢复；不支持路径不通过。完整GPU/线程/Graph证明不是原设计验收条件，来源与正式归因仍归R-08/R-16；按用户要求不运行实验 |
| R-10 内存与竞态工具 | 代码已实现 | M2 | 已实现可选四工具独立正确性worker、原始日志/二进制/设备/请求绑定、无插桩复核与题目级快照重复策略；审核原子操作仅允许有界波动，不放宽精度门；评分与恢复重算证据，未运行工具或GPU实验，重复一致不证明无竞态 |
| R-11 交替计时与噪声底 | 代码已实现 | M3 | 已实现平衡交替墙钟、参考控制噪声、重复进程/配对bootstrap及评分门；CPU路径和持久化反例验证完成，真实设备漂移/时钟试点仍待验收 |
| R-12 独立最终重测 | 代码已实现 | M3 | 已实现冻结源码、新标准/隐藏种子、独立最终任务、所有重复平均与分层区间、失败门和来源复核；GPU独立重测实验仍待执行 |
| R-13 扩展基线组合 | 代码已实现 | M3 | 291题来源清单已覆盖论文指定的SDPA、LayerNorm/var_mean、卷积benchmark/布局和cuBLASLt偏好；六基础组合及每个改写/策略的Graph对应共2750个分题声明。逐输出验证、失败保留、政策收据、冷成本/稳态选择与恢复/配对/评分门已实现；最强指声明且通过的集合，非全库最优性证明；不运行硬件实验 |
| R-14 设备实测上限与边界证书 | 代码已实现 | M3 | 已实现显式CUDA校准、原始探针/驱动与物理设备绑定、资源工作量证书校验及独立见证导出；按用户要求不运行校准或实验，不把达到的速率当成物理定理 |
| R-15 Headroom分层评分与设计证书 | 代码已实现 | M3 | 已接入SOL_target分母、I/F/D可信规格与D独立可达性见证门，缺证据置零；稳态Headroom与K摊销bounded gain分开报告并绑定外部文件，未生成设备实验结果 |
| R-16 完整性能归因 | 部分实现 | M3 | 已实现CUDA busy并集与部分源码/库/Graph来源诊断、Neuron框架/NEFF/NTFF及每引擎指令并集、缺计数器时AE/O与launch_only；硬件工作计数器、完整来源与正式归因门待实现，没有运行设备实验 |
| R-17 样本聚合消融 | 代码已实现 | M3 | 已在同一批声明槽位上并列计算mean/median/best-of-N，保留失败、缺失和成本；代码与持久化评分测试完成，真实模型实验另行报告 |
| R-18 设备时间与token硬预算 | 部分实现 | M4 | 已实现本地精确总token预留、支持的API输出限额、local/Docker开发worker墙钟额度及评分/恢复证明，模型CLI内层超时清理嵌套子进程的竞态也已修复；纯GPU busy时间物理硬限、远端可信计量取消仍待实现，没有运行模型或设备实验 |
| R-19 得分预算曲线与通用agent基线 | 代码已实现 | M4 | 已实现one-shot/反馈/Harbor通用agent多预算多会话计划、显式执行runner、当前证据曲线/区间/共同区间AUC和静态图；默认只生成计划，真实模型实验未运行，设备/token硬预算仍归R-18 |
| R-20 RL训练与奖励验证 | 代码已实现 | M4 | 已实现TRL GRPO、原生终局奖励与审计、审查绑定的多尺寸算法证书加分、模型/奖励/优化路径与父进程凭据、预注册种子对照及持出效果分析；真实训练与模型/设备实验按用户要求不运行，模拟证据不能支持正式效果 |
| R-21 Trainium/NKI与设备通用接口 | 代码已实现 | M5 | 已实现五方法设备接口、原生TorchNeuron/NKI worker、平台基线/校准、core与物理/软件身份、CPU FP64与原生最终/配对重测、每引擎轨迹诊断；40个存储接口仅CPU前端回归，FP8存储明确拒绝，SDK部署、算子覆盖和设备实验未验证 |
| R-22 工具能力收据与可选分层 | 代码已实现 | M5 | 已实现逐项状态收据、缺项null、等级判定与评分门；L1/L2缺少的设备工具证据不会降格冒充通过，GPU验收仍独立记录 |
| R-23 实机隔离与重评分实验 | 代码已实现 | M5 | 已实现CUDA/Docker受限策略与实际配置/进程观察、Harbor禁止联网/只读验证器/agent allowlist，以及完整槽位冻结的native和Harbor产物重评分；原始测量与历史来源分开绑定，默认不执行，不认证OS逃逸或GPU独占；真实部署、隔离和重评分实验按用户范围未运行 |
<!-- roadmap:end -->

## 维护约定

[motivation_map.json](motivation_map.json)保存对应关系、状态、原因及仓库证据入口。新增功能时，同时更新功能清单、映射与论文中的能力状态；用 `python -m scripts.check_motivations` 检查覆盖，并用 `--write` 重建本文的两张表。规划项逐步落地时保留R编号，并记录实现范围与剩余验收；论文中的目标公式或平台名称不自动升级实现状态；每条已实现记录必须指向实际文件。
