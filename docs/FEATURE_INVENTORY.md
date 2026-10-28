# KernelBench Pro 功能改进清单（相比 KernelBench-Verified）

本页记录当前能力，按五条 motivation 归类。`V版本-编号`保留首次交付的追踪标识；完整理由、关联动机和全部证据集中在 [Motivation](MOTIVATION.md)。历史提交、当时的环境和验证数量可在 Git 中查阅，不再逐版重复。

## 当前工作流

| 能力 | 实现范围 |
| --- | --- |
| 运行环境 | Python 3.12；验证依赖 Torch 2.12.0 |
| 任务 | 原Level 1–3、KernelBench-105选集、Production-41四轨；[研究索引](../research/headroom/README.md) |
| 生成 | one-shot / mini-swe-agent feedback、Harbor external、coding-agent skill interactive |
| 默认反馈预算 | 每样本最多5次候选、2400秒，可配置；开发反馈与最终隐藏验证分开 |
| 设备与后端 | TOML/CLI硬件配置；CUDA与原生Neuron/NKI分别适配，local / Docker / Modal / HTTP队列执行 |
| 正确性 | 结构/dtype/离散值/非有限值比较、实际输入数值与状态审计、隐藏输入族、缓存和执行检查；[严格契约](STRICT_VALIDATION.md) |
| 性能与评分 | 经过验证的基线组合、配对噪声、冷成本摊销、冻结复测、Headroom模型与固定分母；[评分说明](HEADROOM_SCORING.md) |
| 来源与归因 | CUDA实际计分来源门、实际R/C/B活动并集及launch_only零分门；完整计数器/Neuron归因仍按剩余范围追踪 |
| 预算与恢复 | token预留、受控开发worker租约、CUDA进程活动/保守上界配额、队列/Modal请求与工件绑定；[资源边界](RESOURCE_BUDGETS.md) |
| 部署与复现 | Harbor独立verifier、只读任务导出、Slurm、结果/轨迹/收据、指纹恢复与已有产物重评分 |

当前23项设计中20项代码已实现、3项部分实现。未完成的是来源路径覆盖、完整性能归因，以及纯GPU时间硬预算/Harbor其他agent及设备资源契约；逐项交付与剩余范围见 [23项状态](MOTIVATION.md#设计与实现的差额)。软件实现与真实实验分别验收；按用户要求，本轮不运行GPU、Trainium、真实模型、云资源、容器或训练实验。

此前搁置的整合与清理继续保存在 stash `ba76365ce0dd755bce1361e38420bb5f4ec3e196`，没有恢复。

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

## 按 Motivation 查找功能

每项列出当前实现入口之一；完整行为、局限及所有入口见对应 motivation 的统一对照表。

### M1：题目与区分度

| 编号 | 功能 | 当前证据入口 | Motivation |
| --- | --- | --- | --- |
| V1-03 | 题目范围与多样本批量生成 | [generation.py](../kb_workflows/agents/generation.py) | [M1](MOTIVATION.md#m1) |
| V8-04 | KernelBench-105 选集 | [kernelbench105.json](../kb_workflows/benchmarks/manifests/kernelbench105.json) | [M1](MOTIVATION.md#m1) |
| V8-05 | 全流程统一选题 | [common.py](../kb_workflows/common.py) | [M1](MOTIVATION.md#m1) |
| V12-01 | Production-41参数化题集 | [production_specs.py](../kb_workflows/benchmarks/production_specs.py) | [M1](MOTIVATION.md#m1) |
| V12-02 | 独立反向轨道与梯度参考 | [production_ops.py](../kb_workflows/benchmarks/production_ops.py) | [M1](MOTIVATION.md#m1) |
| V12-03 | 逐输入角色的dtype契约 | [production.py](../kb_workflows/production.py) | [M1](MOTIVATION.md#m1) |
| V13-01 | 研究结果的当前证据导出 | [study_data.py](../kb_workflows/analysis/study_data.py) | [M1](MOTIVATION.md#m1) |
| V13-02 | 选集区分度与统计不确定性 | [study_statistics.py](../kb_workflows/analysis/study_statistics.py) | [M1](MOTIVATION.md#m1) |
| V16-01 | 显式尺寸实例与证据绑定 | [production.py](../kb_workflows/production.py) | [M1](MOTIVATION.md#m1) |

### M2：正确性与执行可信度

| 编号 | 功能 | 当前证据入口 | Motivation |
| --- | --- | --- | --- |
| V1-11 | 将 Verified 隐藏门接入全部验证后端 | [evaluate_task.py](../kb_workflows/evaluation/evaluate_task.py) | [M2](MOTIVATION.md#m2) |
| V2-01 | 严格比较输出结构、shape 和 dtype | [validation.py](../src/validation.py) | [M2](MOTIVATION.md#m2) |
| V2-02 | 整数与 bool 精确比较 | [validation.py](../src/validation.py) | [M2](MOTIVATION.md#m2) |
| V2-03 | 核对 NaN、Inf 的位置与符号 | [validation.py](../src/validation.py) | [M2](MOTIVATION.md#m2) |
| V2-04 | 隔离参考与候选输入 | [validation.py](../src/validation.py) | [M2](MOTIVATION.md#m2) |
| V2-05 | 保存参考输出快照 | [eval.py](../src/eval.py) | [M2](MOTIVATION.md#m2) |
| V2-06 | 对齐 forward 随机种子 | [eval.py](../src/eval.py) | [M2](MOTIVATION.md#m2) |
| V3-02 | 同步墙钟计时 | [measurement.py](../kb_workflows/evaluation/measurement.py) | [M2](MOTIVATION.md#m2) |
| V3-03 | 变化输入与逐次输出验证 | [measurement.py](../kb_workflows/evaluation/measurement.py) | [M2](MOTIVATION.md#m2) |
| V5-01 | 参考前后配对检查 | [reliability.py](../src/reliability.py) | [M2](MOTIVATION.md#m2) |
| V5-02 | 隔离源码全局字典与后端设置 | [eval.py](../src/eval.py) | [M2](MOTIVATION.md#m2) |
| V5-03 | 三个种子重新生成隐藏输入 | [eval.py](../src/eval.py) | [M2](MOTIVATION.md#m2) |
| V5-04 | 按输入域限定的数值压力测试 | [reliability.py](../src/reliability.py) | [M2](MOTIVATION.md#m2) |
| V5-05 | 无效验证不能算通过 | [eval.py](../src/eval.py) | [M2](MOTIVATION.md#m2) |
| V6-11 | 验证与 reward | [harbor_verifier.py](../kb_workflows/runtime/harbor_verifier.py) | [M2](MOTIVATION.md#m2) |
| V8-03 | 研究证据归档与重建 | [headroom_evidence.py](../scripts/headroom_evidence.py) | [M2](MOTIVATION.md#m2) |
| V12-04 | 显存计划与合法隐藏实例 | [production.py](../kb_workflows/production.py) | [M2](MOTIVATION.md#m2) |
| V12-05 | 生产题数值预算与零答案门 | [validation.py](../src/validation.py) | [M2](MOTIVATION.md#m2) |
| V9-01 | FP64数值契约 | [numerical.py](../kb_workflows/numerics/numerical.py) | [M2](MOTIVATION.md#m2) |
| V9-02 | 参考信号诊断 | [numerical.py](../kb_workflows/numerics/numerical.py) | [M2](MOTIVATION.md#m2) |
| V9-03 | 合法输入族与隐藏形状 | [strict_validation.py](../kb_workflows/evaluation/strict_validation.py) | [M2](MOTIVATION.md#m2) |
| V9-04 | 模式状态与同对象缓存检查 | [strict_validation.py](../kb_workflows/evaluation/strict_validation.py) | [M2](MOTIVATION.md#m2) |
| V9-05 | 进程设置审计 | [strict_validation.py](../kb_workflows/evaluation/strict_validation.py) | [M2](MOTIVATION.md#m2) |
| V15-03 | 分阶段小实例资格任务 | [evaluate_task.py](../kb_workflows/evaluation/evaluate_task.py) | [M2](MOTIVATION.md#m2) |
| V17-04 | CPU FP64与输入角色保护 | [neuron_worker.py](../kb_workflows/hardware/neuron_worker.py) | [M2](MOTIVATION.md#m2) |
| V18-01 | 四工具独立诊断流程 | [tool_runner.py](../kb_workflows/evaluation/tool_runner.py) | [M2](MOTIVATION.md#m2) |
| V18-02 | 模型快照与逐位重复验证 | [determinism.py](../kb_workflows/evaluation/determinism.py) | [M2](MOTIVATION.md#m2) |
| V18-03 | 审核绑定的原子操作策略 | [determinism.py](../kb_workflows/evaluation/determinism.py) | [M2](MOTIVATION.md#m2) |
| V19-04 | 隔离凭据进入评分与恢复 | [paper_analysis.py](../kb_workflows/analysis/paper_analysis.py) | [M2](MOTIVATION.md#m2) |
| V20-02 | CUDA编译产物与实际launch关联 | [cuda_origin.py](../kb_workflows/provenance/cuda_origin.py) | [M2](MOTIVATION.md#m2) |
| V20-05 | 来源诊断与正式资格边界 | [KERNEL_ORIGIN.md](../docs/KERNEL_ORIGIN.md) | [M2](MOTIVATION.md#m2) |
| V21-01 | 受信安装包与实际库二进制 | [origin_binaries.py](../kb_workflows/provenance/origin_binaries.py) | [M2](MOTIVATION.md#m2) |
| V21-02 | CUDA Graph节点与执行快照 | [origin_graphs.py](../kb_workflows/provenance/origin_graphs.py) | [M2](MOTIVATION.md#m2) |
| V21-05 | 诊断容差与实际模式后置检查 | [origin_task.py](../kb_workflows/provenance/origin_task.py) | [M2](MOTIVATION.md#m2) |
| V22-01 | 执行前冻结JIT文本与源码字面量 | [submission_source.py](../kb_workflows/provenance/submission_source.py) | [M2](MOTIVATION.md#m2) |
| V22-02 | 实际前端依赖与审核语言primitive | [triton_origin.py](../kb_workflows/provenance/triton_origin.py) | [M2](MOTIVATION.md#m2) |
| V22-03 | 编译器产物及初始依赖冻结 | [triton_origin.py](../kb_workflows/provenance/triton_origin.py) | [M2](MOTIVATION.md#m2) |
| V22-05 | CUDA object回溯实际模块函数 | [cpp_origin.py](../kb_workflows/provenance/cpp_origin.py) | [M2](MOTIVATION.md#m2) |
| V23-01 | 声明式注册状态映射 | [state_contract.py](../kb_workflows/evaluation/state_contract.py) | [M2](MOTIVATION.md#m2) |
| V23-02 | 连续模式与累计状态检查 | [state_contract.py](../kb_workflows/evaluation/state_contract.py) | [M2](MOTIVATION.md#m2) |
| V23-03 | 零值参数与缓存失效探针 | [state_contract.py](../kb_workflows/evaluation/state_contract.py) | [M2](MOTIVATION.md#m2) |
| V23-05 | 状态快照与证据重算 | [state_contract.py](../kb_workflows/evaluation/state_contract.py) | [M2](MOTIVATION.md#m2) |
| V24-01 | L1独立FP64算式与源码绑定 | [operator_specs.py](../kb_workflows/numerics/operator_specs.py) | [M2](MOTIVATION.md#m2) |
| V24-02 | 卷积与池化窗口参考 | [operator_oracles.py](../kb_workflows/numerics/operator_oracles.py) | [M2](MOTIVATION.md#m2) |
| V24-03 | 数值语义与低精度自资格反例 | [operator_oracles.py](../kb_workflows/numerics/operator_oracles.py) | [M2](MOTIVATION.md#m2) |
| V24-05 | 参考结构与离散自资格修正 | [numerical.py](../kb_workflows/numerics/numerical.py) | [M2](MOTIVATION.md#m2) |
| V24-06 | 数值与信号证据重算 | [numerical_audit.py](../kb_workflows/numerics/numerical_audit.py) | [M2](MOTIVATION.md#m2) |
| V25-01 | Exclusive scan参考有版本修正 | [92_cumsum_exclusive.py](../KernelBench/level1/92_cumsum_exclusive.py) | [M2](MOTIVATION.md#m2) |
| V25-02 | 扫描输入族与原分布保留 | [scan_cases.py](../kb_workflows/scan_cases.py) | [M2](MOTIVATION.md#m2) |
| V25-05 | 空输出数值契约 | [numerical.py](../kb_workflows/numerics/numerical.py) | [M2](MOTIVATION.md#m2) |
| V26-01 | 参考专用输入配方选择 | [input_calibration_task.py](../kb_workflows/numerics/input_calibration_task.py) | [M2](MOTIVATION.md#m2) |
| V26-02 | L1角色保持的值分布 | [input_profiles.py](../kb_workflows/numerics/input_profiles.py) | [M2](MOTIVATION.md#m2) |
| V26-03 | 冻结强数值政策与逐输出复核 | [input_calibration.py](../kb_workflows/numerics/input_calibration.py) | [M2](MOTIVATION.md#m2) |
| V27-01 | 原L1真实初始化的隐藏输入族 | [operator_cases.py](../kb_workflows/numerics/operator_cases.py) | [M2](MOTIVATION.md#m2) |
| V27-02 | 算子边界、抵消与并列极值 | [operator_cases.py](../kb_workflows/numerics/operator_cases.py) | [M2](MOTIVATION.md#m2) |
| V27-03 | 显式非有限域与空数值测量 | [numerical.py](../kb_workflows/numerics/numerical.py) | [M2](MOTIVATION.md#m2) |
| V27-04 | 转换后保留正stride布局 | [eval.py](../src/eval.py) | [M2](MOTIVATION.md#m2) |
| V27-07 | 参考近常量失格证据保留 | [operator_oracles.json](../kb_workflows/numerics/operator_oracles.json) | [M2](MOTIVATION.md#m2) |
| V28-01 | 全部L2来源绑定的复合程序 | [composite_oracles.py](../kb_workflows/numerics/composite_oracles.py) | [M2](MOTIVATION.md#m2) |
| V28-02 | 复合层独立FP64算式 | [composite_math.py](../kb_workflows/numerics/composite_math.py) | [M2](MOTIVATION.md#m2) |
| V28-03 | 原生随机掩码条件下的数值参考 | [composite_oracles.py](../kb_workflows/numerics/composite_oracles.py) | [M2](MOTIVATION.md#m2) |
| V28-07 | 零信号复合参考问题保留 | [operator_oracles.json](../kb_workflows/numerics/operator_oracles.json) | [M2](MOTIVATION.md#m2) |
| V29-01 | L2融合输入族与残差耦合 | [model_input_families.json](../kb_workflows/numerics/model_input_families.json) | [M2](MOTIVATION.md#m2) |
| V29-02 | L3序列状态与固定形状规则 | [model_input_cases.py](../kb_workflows/numerics/model_input_cases.py) | [M2](MOTIVATION.md#m2) |
| V29-03 | 旧概率和浮点标签语义修订 | [reference_revisions.json](../kb_workflows/benchmarks/reference_revisions.json) | [M2](MOTIVATION.md#m2) |
| V29-04 | 全题逐例隐藏覆盖门 | [operator_input_policy.py](../kb_workflows/numerics/operator_input_policy.py) | [M2](MOTIVATION.md#m2) |
| V30-01 | 双模式候选分配预填充 | [allocation_audit.py](../kb_workflows/evaluation/allocation_audit.py) | [M2](MOTIVATION.md#m2) |
| V30-02 | 导入构造与实际调用作用域 | [reliability.py](../src/reliability.py) | [M2](MOTIVATION.md#m2) |
| V30-03 | 连续扩容前缀保护与未知存储 | [allocation_audit.py](../kb_workflows/evaluation/allocation_audit.py) | [M2](MOTIVATION.md#m2) |
| V30-06 | 预填充反例与覆盖范围声明 | [test_allocation_audit.py](../tests/evaluation/test_allocation_audit.py) | [M2](MOTIVATION.md#m2) |
| V31-01 | 独立同步与连续提交进程 | [tool_runner.py](../kb_workflows/evaluation/tool_runner.py) | [M2](MOTIVATION.md#m2) |
| V31-02 | 完整输出快照与复用缓冲区 | [throughput_audit.py](../kb_workflows/evaluation/throughput_audit.py) | [M2](MOTIVATION.md#m2) |
| V31-03 | 连续状态随机与执行政策 | [reliability.py](../src/reliability.py) | [M2](MOTIVATION.md#m2) |
| V31-07 | 排队模拟与状态输出反例 | [test_throughput_audit.py](../tests/evaluation/test_throughput_audit.py) | [M2](MOTIVATION.md#m2) |
| V32-01 | 全实际实例的同对象更新 | [cache_audit.py](../kb_workflows/evaluation/cache_audit.py) | [M2](MOTIVATION.md#m2) |
| V32-02 | 来源绑定的合法值配方 | [cache_recipes.py](../kb_workflows/evaluation/cache_recipes.py) | [M2](MOTIVATION.md#m2) |
| V32-03 | 输入存储别名与几何恢复 | [input_reuse.py](../kb_workflows/evaluation/input_reuse.py) | [M2](MOTIVATION.md#m2) |
| V32-04 | 连续状态与稳定输出快照 | [cache_audit.py](../kb_workflows/evaluation/cache_audit.py) | [M2](MOTIVATION.md#m2) |
| V32-05 | 运行设置拒绝与组合诊断修复 | [tool_checks.py](../kb_workflows/evaluation/tool_checks.py) | [M2](MOTIVATION.md#m2) |
| V33-01 | 稳定普通Tensor状态路径 | [state_paths.py](../kb_workflows/evaluation/state_paths.py) | [M2](MOTIVATION.md#m2) |
| V33-02 | 循环模型逐题状态语义 | [state_policies.py](../kb_workflows/evaluation/state_policies.py) | [M2](MOTIVATION.md#m2) |
| V33-03 | 普通状态快照隔离与保真 | [state_contract.py](../kb_workflows/evaluation/state_contract.py) | [M2](MOTIVATION.md#m2) |
| V33-04 | 持久状态替换与规范参数更新 | [state_contract.py](../kb_workflows/evaluation/state_contract.py) | [M2](MOTIVATION.md#m2) |
| V33-05 | 状态所有者模式与收据重放 | [state_audit.py](../kb_workflows/evaluation/state_audit.py) | [M2](MOTIVATION.md#m2) |
| V35-01 | 复合与序列专用压力政策 | [model_input_families.json](../kb_workflows/numerics/model_input_families.json) | [M2](MOTIVATION.md#m2) |
| V35-02 | 当前实际权重的成对抵消 | [model_stress.py](../kb_workflows/numerics/model_stress.py) | [M2](MOTIVATION.md#m2) |
| V35-03 | GELU与有限极值判别输入 | [operator_cases.py](../kb_workflows/numerics/operator_cases.py) | [M2](MOTIVATION.md#m2) |
| V35-04 | 偏移与共享存储隐藏输入 | [input_layouts.py](../kb_workflows/numerics/input_layouts.py) | [M2](MOTIVATION.md#m2) |
| V35-05 | 按存储组隔离与转移 | [input_groups.py](../src/input_groups.py) | [M2](MOTIVATION.md#m2) |
| V36-01 | 全L3来源与构造冻结 | [model_programs.json](../kb_workflows/numerics/model_programs.json) | [M2](MOTIVATION.md#m2) |
| V36-02 | 独立attention与encoder算术 | [model_layers.py](../kb_workflows/numerics/model_layers.py) | [M2](MOTIVATION.md#m2) |
| V36-03 | 循环网络与普通状态算术 | [model_math.py](../kb_workflows/numerics/model_math.py) | [M2](MOTIVATION.md#m2) |
| V36-04 | Swin、GPT、VLAD与SSD算术 | [model_math.py](../kb_workflows/numerics/model_math.py) | [M2](MOTIVATION.md#m2) |
| V36-05 | meta准入与条件随机掩码 | [model_oracles.py](../kb_workflows/numerics/model_oracles.py) | [M2](MOTIVATION.md#m2) |
| V37-01 | L2/L3来源绑定的名义菜单 | [model_input_profiles.py](../kb_workflows/numerics/model_input_profiles.py) | [M2](MOTIVATION.md#m2) |
| V37-02 | 按存储组生成值与极值修复 | [input_profiles.py](../kb_workflows/numerics/input_profiles.py) | [M2](MOTIVATION.md#m2) |
| V37-03 | 固定前向RNG与输出变化审计 | [input_calibration.py](../kb_workflows/numerics/input_calibration.py) | [M2](MOTIVATION.md#m2) |
| V37-04 | 全题参考专用校准器 | [input_calibration_task.py](../kb_workflows/numerics/input_calibration_task.py) | [M2](MOTIVATION.md#m2) |
| V37-05 | 当前状态的调用前真值票据 | [reference_oracles.py](../kb_workflows/numerics/reference_oracles.py) | [M2](MOTIVATION.md#m2) |
| V38-01 | 冻结参考资格与排除清单 | [reference_population.py](../kb_workflows/benchmarks/reference_population.py) | [M2](MOTIVATION.md#m2) |
| V38-02 | 未解决失败不缩小题集 | [reference_population.py](../kb_workflows/benchmarks/reference_population.py) | [M2](MOTIVATION.md#m2) |
| V39-01 | 不依赖值校准的逐次FP64门 | [timed_numerics.py](../kb_workflows/evaluation/timed_numerics.py) | [M2](MOTIVATION.md#m2) |
| V39-02 | 逐输出持久化数值证据 | [timed_numerics.py](../kb_workflows/evaluation/timed_numerics.py) | [M2](MOTIVATION.md#m2) |
| V39-03 | 配对共用调用前状态真值 | [paired_task.py](../kb_workflows/evaluation/paired_task.py) | [M2](MOTIVATION.md#m2) |
| V40-01 | 固定预算内的误差使用比例 | [numerical.py](../kb_workflows/numerics/numerical.py) | [M2](MOTIVATION.md#m2) |
| V40-02 | 参考专用阈值拟合与验证 | [precision_calibration_task.py](../kb_workflows/numerics/precision_calibration_task.py) | [M2](MOTIVATION.md#m2) |
| V40-03 | 五项预算统一缩放与禁止重拟合 | [precision_calibration.py](../kb_workflows/numerics/precision_calibration.py) | [M2](MOTIVATION.md#m2) |
| V41-01 | Graph模板核函数参数更新 | [origin_graphs.py](../kb_workflows/provenance/origin_graphs.py) | [M2](MOTIVATION.md#m2) |
| V41-02 | 可执行Graph更新与克隆别名 | [origin_graphs.py](../kb_workflows/provenance/origin_graphs.py) | [M2](MOTIVATION.md#m2) |
| V41-03 | Graph更新事件与代次复核 | [kernel_origin.py](../kb_workflows/provenance/kernel_origin.py) | [M2](MOTIVATION.md#m2) |
| V41-04 | 含糊更新失效与有界重放 | [origin_graphs.py](../kb_workflows/provenance/origin_graphs.py) | [M2](MOTIVATION.md#m2) |
| V42-05 | 计数器父校验与失败工件保存 | [counter_task.py](../kb_workflows/provenance/counter_task.py) | [M2](MOTIVATION.md#m2) |
| V43-03 | Execution与NKI调用点联结 | [neuron_tables.py](../kb_workflows/hardware/neuron_tables.py) | [M2](MOTIVATION.md#m2) |
| V43-05 | Profile输出与调用前票据验证 | [neuron_worker.py](../kb_workflows/hardware/neuron_worker.py) | [M2](MOTIVATION.md#m2) |
| V43-06 | 推理模式统一与梯度设置恢复 | [strict_validation.py](../kb_workflows/evaluation/strict_validation.py) | [M2](MOTIVATION.md#m2) |
| V44-01 | Neuron采集来源与调用收据 | [neuron_work.py](../kb_workflows/hardware/neuron_work.py) | [M2](MOTIVATION.md#m2) |
| V44-04 | 显式execution身份核对 | [neuron_tables.py](../kb_workflows/hardware/neuron_tables.py) | [M2](MOTIVATION.md#m2) |
| V45-01 | 采集字节与可见core清单 | [neuron_processing.py](../kb_workflows/hardware/neuron_processing.py) | [M2](MOTIVATION.md#m2) |
| V45-03 | 固定Explorer处理工具与命令 | [neuron_processing.py](../kb_workflows/hardware/neuron_processing.py) | [M2](MOTIVATION.md#m2) |
| V45-04 | 输入输出复查与失败工件保留 | [neuron_processing.py](../kb_workflows/hardware/neuron_processing.py) | [M2](MOTIVATION.md#m2) |
| V48-01 | 原生调用覆盖与独立来源对照 | [origin_task.py](../kb_workflows/provenance/origin_task.py) | [M2](MOTIVATION.md#m2) |
| V51-01 | 显存前向输出、峰值与来源关联 | [eval.py](../src/eval.py) | [M2](MOTIVATION.md#m2) |
| V54-01 | 原始计时可调用对象与实际调用收据 | [origin_task.py](../kb_workflows/provenance/origin_task.py) | [M2](MOTIVATION.md#m2) |
| V55-01 | 逐调用模型数据、入口与别名绑定 | [state_paths.py](../kb_workflows/evaluation/state_paths.py) | [M2](MOTIVATION.md#m2) |
| V56-01 | Neuron完整候选状态与内容绑定 | [neuron_worker.py](../kb_workflows/hardware/neuron_worker.py) | [M2](MOTIVATION.md#m2) |
| V57-01 | 实际计分持续采集与自写/库轨道门 | [origin_task.py](../kb_workflows/provenance/origin_task.py) | [M2](MOTIVATION.md#m2) |
| V59-01 | L1测量证据汇总与评分重放 | [evidence.py](../kb_workflows/evaluation/evidence.py) | [M2](MOTIVATION.md#m2) |

### M3：可解释且可比较的性能指标

| 编号 | 功能 | 当前证据入口 | Motivation |
| --- | --- | --- | --- |
| V1-12 | 基线记录接入批量流程 | [engine.py](../kb_workflows/engine.py) | [M3](MOTIVATION.md#m3) |
| V1-13 | 子集分析、pass@k 与统一结果归档 | [engine.py](../kb_workflows/engine.py) | [M3](MOTIVATION.md#m3) |
| V3-01 | 默认四变体基线 | [baseline_variants.py](../kb_workflows/evaluation/baseline_variants.py) | [M3](MOTIVATION.md#m3) |
| V3-04 | 冷成本与 K 次摊销 | [measurement.py](../kb_workflows/evaluation/measurement.py) | [M3](MOTIVATION.md#m3) |
| V3-05 | 原始样本和测量信息 | [measurement.py](../kb_workflows/evaluation/measurement.py) | [M3](MOTIVATION.md#m3) |
| V3-06 | 区分旧计时与新结果 | [config.py](../kb_workflows/config.py) | [M3](MOTIVATION.md#m3) |
| V4-01 | 失败和缺失样本计入分母 | [paper_analysis.py](../kb_workflows/analysis/paper_analysis.py) | [M3](MOTIVATION.md#m3) |
| V4-02 | 按 K 摊销成本评分 | [paper_analysis.py](../kb_workflows/analysis/paper_analysis.py) | [M3](MOTIVATION.md#m3) |
| V4-03 | 兼容性和测量证据检查 | [paper_analysis.py](../kb_workflows/analysis/paper_analysis.py) | [M3](MOTIVATION.md#m3) |
| V4-04 | 分层统计与区间 | [paper_analysis.py](../kb_workflows/analysis/paper_analysis.py) | [M3](MOTIVATION.md#m3) |
| V4-06 | 旧分析与榜单不能绕过证据检查 | [common.py](../kb_workflows/common.py) | [M3](MOTIVATION.md#m3) |
| V5-06 | 分阶段诊断与逐项收据 | [engine.py](../kb_workflows/engine.py) | [M3](MOTIVATION.md#m3) |
| V6-06 | 结果页面 | [generate_leaderboard.py](../scripts/generate_leaderboard.py) | [M3](MOTIVATION.md#m3) |
| V7-04 | 明确选择硬件 | [config.py](../kb_workflows/config.py) | [M3](MOTIVATION.md#m3) |
| V8-06 | 分层来源与探针隔离 | [paper_analysis.py](../kb_workflows/analysis/paper_analysis.py) | [M3](MOTIVATION.md#m3) |
| V11-01 | 设备校准接口与原始收据 | [calibration.py](../kb_workflows/analysis/calibration.py) | [M3](MOTIVATION.md#m3) |
| V11-02 | 工作量证书与独立可达性见证 | [headroom.py](../kb_workflows/analysis/headroom.py) | [M3](MOTIVATION.md#m3) |
| V11-03 | Headroom评分与证据失效 | [paper_analysis.py](../kb_workflows/analysis/paper_analysis.py) | [M3](MOTIVATION.md#m3) |
| V11-04 | 设备时间并集与性能解释 | [performance.py](../kb_workflows/analysis/performance.py) | [M3](MOTIVATION.md#m3) |
| V9-07 | 严格评分门与协议隔离 | [paper_analysis.py](../kb_workflows/analysis/paper_analysis.py) | [M3](MOTIVATION.md#m3) |
| V9-08 | 样本聚合消融 | [aggregation.py](../kb_workflows/evaluation/aggregation.py) | [M3](MOTIVATION.md#m3) |
| V10-01 | 交替计时与噪声对照 | [paired_task.py](../kb_workflows/evaluation/paired_task.py) | [M3](MOTIVATION.md#m3) |
| V10-02 | 冻结后独立最终重测 | [final_verification.py](../kb_workflows/evaluation/final_verification.py) | [M3](MOTIVATION.md#m3) |
| V10-03 | 扩展基线组合与可信改写 | [baseline_variants.py](../kb_workflows/evaluation/baseline_variants.py) | [M3](MOTIVATION.md#m3) |
| V10-04 | 附加证据与旧评分门 | [common.py](../kb_workflows/common.py) | [M3](MOTIVATION.md#m3) |
| V14-04 | 预算恢复与计分资格 | [generation.py](../kb_workflows/agents/generation.py) | [M3](MOTIVATION.md#m3) |
| V15-04 | 同题交替终局奖励 | [rl_rewards.py](../kb_workflows/agents/rl_rewards.py) | [M3](MOTIVATION.md#m3) |
| V16-02 | 多尺寸算法证书 | [scaling_certificate.py](../kb_workflows/analysis/scaling_certificate.py) | [M3](MOTIVATION.md#m3) |
| V16-06 | 按训练种子的效果分析 | [rl_effectiveness.py](../kb_workflows/agents/rl_effectiveness.py) | [M3](MOTIVATION.md#m3) |
| V17-03 | Neuron基线与配对重测 | [neuron_worker.py](../kb_workflows/hardware/neuron_worker.py) | [M3](MOTIVATION.md#m3) |
| V17-05 | Neuron校准与物理身份 | [neuron_device.py](../kb_workflows/hardware/neuron_device.py) | [M3](MOTIVATION.md#m3) |
| V17-06 | 设备轨迹与每引擎诊断 | [neuron_tables.py](../kb_workflows/hardware/neuron_tables.py) | [M3](MOTIVATION.md#m3) |
| V18-05 | 诊断与性能采样分离 | [tool_runner.py](../kb_workflows/evaluation/tool_runner.py) | [M3](MOTIVATION.md#m3) |
| V19-07 | 重评分来源与测量凭据 | [原始结果与关联工件](ISOLATION_REGRADE.md#原生已有候选重评分) | [M3](MOTIVATION.md#m3) |
| V20-03 | 来源时间并集与未知重放 | [kernel_origin.py](../kb_workflows/provenance/kernel_origin.py) | [M3](MOTIVATION.md#m3) |
| V21-03 | 原生内存操作与双轨道诊断 | [kernel_origin.py](../kb_workflows/provenance/kernel_origin.py) | [M3](MOTIVATION.md#m3) |
| V25-06 | 诊断设置贯穿冻结与比较 | [tool_checks.py](../kb_workflows/evaluation/tool_checks.py) | [M3](MOTIVATION.md#m3) |
| V26-04 | 新基线与恢复证据 | [engine.py](../kb_workflows/engine.py) | [M3](MOTIVATION.md#m3) |
| V31-05 | 吞吐差异复核与性能资格 | [paper_analysis.py](../kb_workflows/analysis/paper_analysis.py) | [M3](MOTIVATION.md#m3) |
| V34-01 | 全题来源绑定的基线清单 | [baseline_rewrites.json](../kb_workflows/evaluation/baseline_rewrites.json) | [M3](MOTIVATION.md#m3) |
| V34-02 | 注意力与归一化库改写 | [baseline_rewrites.py](../kb_workflows/evaluation/baseline_rewrites.py) | [M3](MOTIVATION.md#m3) |
| V34-03 | 卷积布局与后端偏好作用域 | [baseline_policies.py](../kb_workflows/evaluation/baseline_policies.py) | [M3](MOTIVATION.md#m3) |
| V34-04 | 每个扩展项的Graph对应 | [baseline_variants.py](../kb_workflows/evaluation/baseline_variants.py) | [M3](MOTIVATION.md#m3) |
| V38-03 | 候选前冻结固定评分分母 | [reference_population.py](../kb_workflows/benchmarks/reference_population.py) | [M3](MOTIVATION.md#m3) |
| V39-05 | 评分恢复要求计时数值证据 | [tool_checks.py](../kb_workflows/evaluation/tool_checks.py) | [M3](MOTIVATION.md#m3) |
| V40-06 | 模型生成前冻结阈值政策 | [precision_calibration.py](../kb_workflows/numerics/precision_calibration.py) | [M3](MOTIVATION.md#m3) |
| V42-01 | 原始工作计数器逐核重放 | [work_counters.py](../kb_workflows/provenance/work_counters.py) | [M3](MOTIVATION.md#m3) |
| V42-02 | 审核资源映射与完整实例契约 | [work_counters.py](../kb_workflows/provenance/work_counters.py) | [M3](MOTIVATION.md#m3) |
| V42-03 | 计数器驱动的A/E分解 | [performance.py](../kb_workflows/analysis/performance.py) | [M3](MOTIVATION.md#m3) |
| V42-04 | 独立Nsight参考与候选采集 | [counter_task.py](../kb_workflows/provenance/counter_task.py) | [M3](MOTIVATION.md#m3) |
| V43-02 | Neuron逐指令工作与DMA分账 | [neuron_tables.py](../kb_workflows/hardware/neuron_tables.py) | [M3](MOTIVATION.md#m3) |
| V44-02 | 逐指令审核与完整资源工作模型 | [neuron_work.py](../kb_workflows/hardware/neuron_work.py) | [M3](MOTIVATION.md#m3) |
| V44-03 | Neuron同设备校准与A/E | [performance.py](../kb_workflows/analysis/performance.py) | [M3](MOTIVATION.md#m3) |
| V44-05 | 采集前输入隔离 | [neuron_worker.py](../kb_workflows/hardware/neuron_worker.py) | [M3](MOTIVATION.md#m3) |
| V45-05 | 处理收据驱动Neuron模型 | [neuron_work.py](../kb_workflows/hardware/neuron_work.py) | [M3](MOTIVATION.md#m3) |
| V49-01 | 保存报告重放与原生时间/设备活动关联 | [origin_task.py](../kb_workflows/provenance/origin_task.py) | [M3](MOTIVATION.md#m3) |
| V50-01 | 原生序列计数器与逐调用A/E关联 | [counter_task.py](../kb_workflows/provenance/counter_task.py) | [M3](MOTIVATION.md#m3) |
| V60-01 | 实际配对设备时间与启动收益评分门 | [origin_task.py](../kb_workflows/provenance/origin_task.py) | [M3](MOTIVATION.md#m3) |
| V61-01 | 实际计分工作量指纹与计数模型关联 | [counter_task.py](../kb_workflows/provenance/counter_task.py)、[活动流量](WORK_COUNTERS.md#复制与清零活动) | [M3](MOTIVATION.md#m3) |

### M4：现实的Agent回路与预算

| 编号 | 功能 | 当前证据入口 | Motivation |
| --- | --- | --- | --- |
| V63-01 | Harbor受控输出额度与评分恢复 | [harbor_budget.py](../kb_workflows/runtime/harbor_budget.py) | [M4](MOTIVATION.md#m4) |
| V62-01 | 开发CUDA活动计量与软件配额 | [resource_budgets.py](../kb_workflows/agents/resource_budgets.py) | [M4](MOTIVATION.md#m4) |
| V7-07 | 真实 agent loop | [agent_loop.py](../kb_workflows/agents/agent_loop.py) | [M4](MOTIVATION.md#m4) |
| V7-08 | One-shot | [agent_models.py](../kb_workflows/agents/agent_models.py) | [M4](MOTIVATION.md#m4) |
| V7-09 | 有界反馈 | [agent_loop.py](../kb_workflows/agents/agent_loop.py) | [M4](MOTIVATION.md#m4) |
| V7-10 | 开发与最终评测分离 | [profile_task.py](../kb_workflows/runtime/profile_task.py) | [M4](MOTIVATION.md#m4) |
| V7-11 | 资源与来源记录 | [experiment.py](../kb_workflows/analysis/experiment.py) | [M4](MOTIVATION.md#m4) |
| V7-12 | 模式比较 | [experiment.py](../kb_workflows/analysis/experiment.py) | [M4](MOTIVATION.md#m4) |
| V7-13 | 模型与后端 | [agent_models.py](../kb_workflows/agents/agent_models.py) | [M4](MOTIVATION.md#m4) |
| V7-14 | Coding-agent skill | [SKILL.md](../.claude/skills/kernelbench-pro/SKILL.md) | [M4](MOTIVATION.md#m4) |
| V13-03 | 多预算多会话与通用agent计划 | [study_plan.py](../kb_workflows/analysis/study_plan.py) | [M4](MOTIVATION.md#m4) |
| V13-04 | 得分预算曲线与共同区间AUC | [budget_curves.py](../kb_workflows/analysis/budget_curves.py) | [M4](MOTIVATION.md#m4) |
| V13-05 | 缓存与外部agent用量核算 | [agent_models.py](../kb_workflows/agents/agent_models.py) | [M4](MOTIVATION.md#m4) |
| V14-01 | 调用前token预留与用量结算 | [agent_models.py](../kb_workflows/agents/agent_models.py) | [M4](MOTIVATION.md#m4) |
| V14-02 | 离线本地模型适配 | [local_model.py](../kb_workflows/agents/local_model.py) | [M4](MOTIVATION.md#m4) |
| V14-03 | 开发worker占用预留 | [resource_budgets.py](../kb_workflows/agents/resource_budgets.py) | [M4](MOTIVATION.md#m4) |
| V14-05 | token与设备占用预算网格 | [study_plan.py](../kb_workflows/analysis/study_plan.py) | [M4](MOTIVATION.md#m4) |
| V15-01 | 冻结训练计划与持出划分 | [rl_plan.py](../kb_workflows/agents/rl_plan.py) | [M4](MOTIVATION.md#m4) |
| V15-02 | 原生GRPO与标准开发工具 | [rl_training.py](../kb_workflows/agents/rl_training.py) | [M4](MOTIVATION.md#m4) |
| V15-06 | 奖励重评分与可靠性审计 | [rl_evaluation.py](../kb_workflows/agents/rl_evaluation.py) | [M4](MOTIVATION.md#m4) |
| V15-07 | 持出评测计划与CLI | [rl_evaluation.py](../kb_workflows/agents/rl_evaluation.py) | [M4](MOTIVATION.md#m4) |
| V16-03 | 有界证书奖励 | [rl_rewards.py](../kb_workflows/agents/rl_rewards.py) | [M4](MOTIVATION.md#m4) |
| V16-05 | 预注册奖励对照与持出评测 | [rl_effectiveness.py](../kb_workflows/agents/rl_effectiveness.py) | [M4](MOTIVATION.md#m4) |
| V17-07 | NKI提示与coding skill | [generation.py](../kb_workflows/agents/generation.py) | [M4](MOTIVATION.md#m4) |
| V38-08 | 内层模型超时清理嵌套进程 | [agent_models.py](../kb_workflows/agents/agent_models.py) | [M4](MOTIVATION.md#m4) |
| V47-01 | 远端队列租约、取消与未知用量拒绝 | [queue_worker.py](../kb_workflows/runtime/queue_worker.py) | [M4](MOTIVATION.md#m4) |
| V53-01 | Modal开发预算与单次租约执行 | [modal_backend.py](../kb_workflows/runtime/modal_backend.py) | [M4](MOTIVATION.md#m4) |
| V58-01 | Modal原生生成预算与工件核对 | [modal_backend.py](../kb_workflows/runtime/modal_backend.py) | [M4](MOTIVATION.md#m4) |

### M5：可复现且可部署的基础设施

| 编号 | 功能 | 当前证据入口 | Motivation |
| --- | --- | --- | --- |
| V1-01 | 统一批量入口 | [cli.py](../kb_workflows/cli.py) | [M5](MOTIVATION.md#m5) |
| V1-02 | 生成与最终验证独立选择后端 | [engine.py](../kb_workflows/engine.py) | [M5](MOTIVATION.md#m5) |
| V1-04 | 多 GPU 调度与并发限制 | [engine.py](../kb_workflows/engine.py) | [M5](MOTIVATION.md#m5) |
| V1-05 | Docker 生成与验证 | [Dockerfile](../Dockerfile) | [M5](MOTIVATION.md#m5) |
| V1-06 | Slurm 集群提交 | [batch_run.sbatch](../batch_run.sbatch) | [M5](MOTIVATION.md#m5) |
| V1-07 | Modal 云端执行 | [modal_backend.py](../kb_workflows/runtime/modal_backend.py) | [M5](MOTIVATION.md#m5) |
| V1-08 | 持久化 HTTP 验证队列 | [queue_worker.py](../kb_workflows/runtime/queue_worker.py) | [M5](MOTIVATION.md#m5) |
| V1-09 | 带内容指纹的断点恢复 | [common.py](../kb_workflows/common.py) | [M5](MOTIVATION.md#m5) |
| V1-10 | 批量任务的超时与失败处理 | [transports.py](../kb_workflows/runtime/transports.py) | [M5](MOTIVATION.md#m5) |
| V1-14 | 旧入口兼容与配置预览 | [legacy.py](../kb_workflows/legacy.py) | [M5](MOTIVATION.md#m5) |
| V4-05 | 扩大恢复指纹覆盖 | [common.py](../kb_workflows/common.py) | [M5](MOTIVATION.md#m5) |
| V5-07 | 新协议参与恢复指纹 | [common.py](../kb_workflows/common.py) | [M5](MOTIVATION.md#m5) |
| V6-01 | 论文与旧实验材料 | [NOTICE.md](../NOTICE.md) | [M5](MOTIVATION.md#m5) |
| V6-02 | 重复生成材料 | [INPUT_BLIND_GENERATION.md](../docs/INPUT_BLIND_GENERATION.md) | [M5](MOTIVATION.md#m5) |
| V6-03 | 仓库入口 | [README.md](../README.md) | [M5](MOTIVATION.md#m5) |
| V6-04 | 维护方式 | [CONTRIBUTING.md](../CONTRIBUTING.md) | [M5](MOTIVATION.md#m5) |
| V6-05 | 项目名称 | [pyproject.toml](../pyproject.toml) | [M5](MOTIVATION.md#m5) |
| V6-07 | 忽略规则 | [.gitignore](../.gitignore) | [M5](MOTIVATION.md#m5) |
| V6-08 | 使用中的文件 | [paper_analysis.py](../kb_workflows/analysis/paper_analysis.py) | [M5](MOTIVATION.md#m5) |
| V6-09 | 统一生成后端 | [harbor.py](../kb_workflows/runtime/harbor.py) | [M5](MOTIVATION.md#m5) |
| V6-10 | 统一任务导出 | [harbor.py](../kb_workflows/runtime/harbor.py) | [M5](MOTIVATION.md#m5) |
| V6-12 | GPU 与恢复 | [harbor.py](../kb_workflows/runtime/harbor.py) | [M5](MOTIVATION.md#m5) |
| V6-13 | Python 升级 | [pyproject.toml](../pyproject.toml) | [M5](MOTIVATION.md#m5) |
| V6-14 | 文档 | [HARBOR.md](../docs/HARBOR.md) | [M5](MOTIVATION.md#m5) |
| V7-01 | 文件配置 | [config.py](../kb_workflows/config.py) | [M5](MOTIVATION.md#m5) |
| V7-02 | GPU 预设 | [gpu_profiles.toml](../kb_workflows/hardware/gpu_profiles.toml) | [M5](MOTIVATION.md#m5) |
| V7-03 | 自定义目标 | [gpu.py](../kb_workflows/hardware/gpu.py) | [M5](MOTIVATION.md#m5) |
| V7-05 | 后端与子任务 | [generation.py](../kb_workflows/agents/generation.py) | [M5](MOTIVATION.md#m5) |
| V7-06 | 集群与恢复 | [batch_run.sbatch](../batch_run.sbatch) | [M5](MOTIVATION.md#m5) |
| V8-01 | 五条动机与逐项追踪 | [MOTIVATION.md](../docs/MOTIVATION.md) | [M5](MOTIVATION.md#m5) |
| V8-02 | 统一论文与能力状态 | [PAPER.md](../research/headroom/PAPER.md) | [M5](MOTIVATION.md#m5) |
| V8-07 | 选集与参考版本证据 | [suites.py](../kb_workflows/benchmarks/suites.py) | [M5](MOTIVATION.md#m5) |
| V11-05 | Verifier配置与精度设置恢复 | [harbor.py](../kb_workflows/runtime/harbor.py) | [M5](MOTIVATION.md#m5) |
| V12-06 | 生产题全流程与来源恢复 | [engine.py](../kb_workflows/engine.py) | [M5](MOTIVATION.md#m5) |
| V13-06 | 研究CLI与静态图导出 | [analyze_budget_curves.py](../scripts/analyze_budget_curves.py) | [M5](MOTIVATION.md#m5) |
| V9-06 | 逐项证据等级 | [evidence.py](../kb_workflows/evaluation/evidence.py) | [M5](MOTIVATION.md#m5) |
| V10-05 | 重测恢复与失败分母 | [final_verification.py](../kb_workflows/evaluation/final_verification.py) | [M5](MOTIVATION.md#m5) |
| V15-05 | 训练预算与checkpoint恢复 | [rl_training.py](../kb_workflows/agents/rl_training.py) | [M5](MOTIVATION.md#m5) |
| V16-04 | 训练来源与父进程凭据 | [rl_lineage.py](../kb_workflows/agents/rl_lineage.py) | [M5](MOTIVATION.md#m5) |
| V17-01 | 通用设备适配协议 | [devices.py](../kb_workflows/hardware/devices.py) | [M5](MOTIVATION.md#m5) |
| V17-02 | Neuron配置与任务执行 | [config.py](../kb_workflows/config.py) | [M5](MOTIVATION.md#m5) |
| V18-04 | 原始工具证据与负结果恢复 | [tool_checks.py](../kb_workflows/evaluation/tool_checks.py) | [M5](MOTIVATION.md#m5) |
| V18-06 | 统一配置与跨流程指纹 | [config.py](../kb_workflows/config.py) | [M5](MOTIVATION.md#m5) |
| V19-01 | 受限验证器配置与镜像固定 | [isolation.py](../kb_workflows/runtime/isolation.py) | [M5](MOTIVATION.md#m5) |
| V19-02 | 实际权限与资源观察 | [isolation.py](../kb_workflows/runtime/isolation.py) | [M5](MOTIVATION.md#m5) |
| V19-03 | 结果交接与容器清理 | [restricted_transport.py](../kb_workflows/runtime/restricted_transport.py) | [M5](MOTIVATION.md#m5) |
| V19-05 | Harbor网络与只读验证器 | [harbor.py](../kb_workflows/runtime/harbor.py) | [M5](MOTIVATION.md#m5) |
| V19-06 | 冻结候选的原生重评分 | [冻结配置与采集入口](../scripts/audit_kernel_origin.py) | [M5](MOTIVATION.md#m5) |
| V19-08 | Harbor产物重评分与默认预览 | [harbor_regrade.py](../kb_workflows/runtime/harbor_regrade.py) | [M5](MOTIVATION.md#m5) |
| V20-01 | 冻结候选源码与JIT身份 | [submission_source.py](../kb_workflows/provenance/submission_source.py) | [M5](MOTIVATION.md#m5) |
| V20-04 | 独立来源诊断与默认预览 | [origin_task.py](../kb_workflows/provenance/origin_task.py) | [M5](MOTIVATION.md#m5) |
| V21-04 | CUPTI活动关联与原始提取记录 | [cuda_origin.py](../kb_workflows/provenance/cuda_origin.py) | [M5](MOTIVATION.md#m5) |
| V21-06 | 来源任务进入队列验证入口 | [queue_worker.py](../kb_workflows/runtime/queue_worker.py) | [M5](MOTIVATION.md#m5) |
| V22-04 | CUDA C++真实构建观察 | [cpp_origin.py](../kb_workflows/provenance/cpp_origin.py) | [M5](MOTIVATION.md#m5) |
| V22-06 | 来源协议v3与实际SDK结构兼容 | [kernel_origin.py](../kb_workflows/provenance/kernel_origin.py) | [M5](MOTIVATION.md#m5) |
| V23-04 | 原生逐实例状态复核 | [state_audit.py](../kb_workflows/evaluation/state_audit.py) | [M5](MOTIVATION.md#m5) |
| V24-04 | 实际输入复核与内存准入 | [numerical_audit.py](../kb_workflows/numerics/numerical_audit.py) | [M5](MOTIVATION.md#m5) |
| V25-03 | 任务修订与共享依赖绑定 | [reference_revisions.py](../kb_workflows/benchmarks/reference_revisions.py) | [M5](MOTIVATION.md#m5) |
| V25-04 | 选集运行快照升级与历史档案 | [suites.py](../kb_workflows/benchmarks/suites.py) | [M5](MOTIVATION.md#m5) |
| V25-07 | 修订任务跨入口回归 | [test_reference_revisions.py](../tests/benchmarks/test_reference_revisions.py) | [M5](MOTIVATION.md#m5) |
| V26-05 | 多设备分布复用与实测身份 | [input_calibration.py](../kb_workflows/numerics/input_calibration.py) | [M5](MOTIVATION.md#m5) |
| V26-06 | 校准文件贯穿冻结与任务桥接 | [generation.py](../kb_workflows/agents/generation.py) | [M5](MOTIVATION.md#m5) |
| V26-07 | 轻量校准预览与准入回归 | [calibrate_inputs.py](../scripts/calibrate_inputs.py) | [M5](MOTIVATION.md#m5) |
| V27-05 | 原生隐藏执行与覆盖复核 | [operator_input_policy.py](../kb_workflows/numerics/operator_input_policy.py) | [M5](MOTIVATION.md#m5) |
| V27-06 | 输入内存准入与全流程冻结 | [config.py](../kb_workflows/config.py) | [M5](MOTIVATION.md#m5) |
| V27-08 | 全L1元数据与原生入口回归 | [test_operator_inputs.py](../tests/numerics/test_operator_inputs.py) | [M5](MOTIVATION.md#m5) |
| V28-04 | 复合收据与缺项拒绝 | [composite_specs.py](../kb_workflows/numerics/composite_specs.py) | [M5](MOTIVATION.md#m5) |
| V28-05 | meta状态副本与分配前准入 | [composite_oracles.py](../kb_workflows/numerics/composite_oracles.py) | [M5](MOTIVATION.md#m5) |
| V28-06 | 独立数值入口扩至L1及L2 | [config.py](../kb_workflows/config.py) | [M5](MOTIVATION.md#m5) |
| V28-08 | 全L2数学与名义形状回归 | [_composite_cases.py](../tests/support/_composite_cases.py) | [M5](MOTIVATION.md#m5) |
| V29-05 | 形状精度独立重放与来源预检 | [engine.py](../kb_workflows/engine.py) | [M5](MOTIVATION.md#m5) |
| V29-06 | 全模型元数据与域迁移反例 | [test_model_input_cases.py](../tests/numerics/test_model_input_cases.py) | [M5](MOTIVATION.md#m5) |
| V30-04 | 独立进程关闭缓存并同步报错 | [tool_runner.py](../kb_workflows/evaluation/tool_runner.py) | [M5](MOTIVATION.md#m5) |
| V30-05 | 来源调用收据与冻结部署 | [tool_checks.py](../kb_workflows/evaluation/tool_checks.py) | [M5](MOTIVATION.md#m5) |
| V31-04 | 名义元数据和存储准入 | [throughput_audit.py](../kb_workflows/evaluation/throughput_audit.py) | [M5](MOTIVATION.md#m5) |
| V31-06 | 请求冻结与多后端私有部署 | [tool_checks.py](../kb_workflows/evaluation/tool_checks.py) | [M5](MOTIVATION.md#m5) |
| V32-06 | 存储准入和私有部署冻结 | [engine.py](../kb_workflows/engine.py) | [M5](MOTIVATION.md#m5) |
| V32-07 | 全族回归与原始范围验收 | [test_cache_audit.py](../tests/evaluation/test_cache_audit.py) | [M5](MOTIVATION.md#m5) |
| V33-06 | 全题状态政策与私有工作流 | [engine.py](../kb_workflows/engine.py) | [M5](MOTIVATION.md#m5) |
| V33-07 | 原构造元数据与循环状态回归 | [test_ordinary_state.py](../tests/evaluation/test_ordinary_state.py) | [M5](MOTIVATION.md#m5) |
| V34-05 | 基线政策收据与复核门 | [engine.py](../kb_workflows/engine.py) | [M5](MOTIVATION.md#m5) |
| V34-06 | 原生预检与多后端边界 | [cli.py](../kb_workflows/cli.py) | [M5](MOTIVATION.md#m5) |
| V34-07 | 数学格式与后端模拟回归 | [test_baseline_recipes.py](../tests/evaluation/test_baseline_recipes.py) | [M5](MOTIVATION.md#m5) |
| V35-06 | 实际副本交付收据 | [operator_input_policy.py](../kb_workflows/numerics/operator_input_policy.py) | [M5](MOTIVATION.md#m5) |
| V35-07 | 别名与低精度反例回归 | [test_input_layouts.py](../tests/evaluation/test_input_layouts.py) | [M5](MOTIVATION.md#m5) |
| V36-06 | 模型证据与统一数值入口 | [model_specs.py](../kb_workflows/numerics/model_specs.py) | [M5](MOTIVATION.md#m5) |
| V36-07 | 小构造对照与参考缺陷保留 | [test_model_oracles.py](../tests/numerics/test_model_oracles.py) | [M5](MOTIVATION.md#m5) |
| V37-06 | 原生与Neuron计时顺序统一 | [paired_task.py](../kb_workflows/evaluation/paired_task.py) | [M5](MOTIVATION.md#m5) |
| V37-07 | 角色、退化与连续状态回归 | [test_model_input_profiles.py](../tests/numerics/test_model_input_profiles.py) | [M5](MOTIVATION.md#m5) |
| V38-04 | 工作流与恢复共用题集 | [cli.py](../kb_workflows/cli.py) | [M5](MOTIVATION.md#m5) |
| V38-05 | 父集合与单题委托绑定 | [input_calibration.py](../kb_workflows/numerics/input_calibration.py) | [M5](MOTIVATION.md#m5) |
| V38-06 | 无执行的集合冻结命令 | [freeze_reference_population.py](../scripts/freeze_reference_population.py) | [M5](MOTIVATION.md#m5) |
| V38-07 | 资格集合的失败与传递回归 | [test_reference_population.py](../tests/benchmarks/test_reference_population.py) | [M5](MOTIVATION.md#m5) |
| V39-04 | 原生与Neuron数值路径一致 | [neuron_worker.py](../kb_workflows/hardware/neuron_worker.py) | [M5](MOTIVATION.md#m5) |
| V39-06 | 验证失败时恢复真实调用后RNG | [measurement.py](../kb_workflows/evaluation/measurement.py) | [M5](MOTIVATION.md#m5) |
| V39-07 | 无配方的模型与状态计时回归 | [test_timed_numerics.py](../tests/evaluation/test_timed_numerics.py) | [M5](MOTIVATION.md#m5) |
| V40-04 | 阈值预览与单次配方应用 | [calibrate_precision.py](../scripts/calibrate_precision.py) | [M5](MOTIVATION.md#m5) |
| V40-05 | 校准与执行设备分离核对 | [precision_calibration.py](../kb_workflows/numerics/precision_calibration.py) | [M5](MOTIVATION.md#m5) |
| V40-07 | 私有阈值贯通Agent与重评分 | [generation.py](../kb_workflows/agents/generation.py) | [M5](MOTIVATION.md#m5) |
| V40-08 | 拟合反例与批量传递回归 | [test_precision_calibration.py](../tests/numerics/test_precision_calibration.py) | [M5](MOTIVATION.md#m5) |
| V41-05 | Graph更新反例与主机ABI核对 | [test_graph_updates.py](../tests/provenance/test_graph_updates.py) | [M5](MOTIVATION.md#m5) |
| V42-06 | 计数器路由与精度校准队列修复 | [audit_work_counters.py](../scripts/audit_work_counters.py) | [M5](MOTIVATION.md#m5) |
| V42-07 | 计数器反例与软件集成回归 | [test_work_counters.py](../tests/provenance/test_work_counters.py) | [M5](MOTIVATION.md#m5) |
| V43-01 | Schema12分片完整读取与资源准入 | [neuron_tables.py](../kb_workflows/hardware/neuron_tables.py) | [M5](MOTIVATION.md#m5) |
| V43-04 | Neuron采集导出与源码缓存 | [运行时key与NEFF缓存快照](NEURON_WORK.md#运行时key与neff缓存快照) | [M5](MOTIVATION.md#m5) |
| V43-07 | Parquet与Neuron模拟回归 | [test_neuron_tables.py](../tests/hardware/test_neuron_tables.py) | [M5](MOTIVATION.md#m5) |
| V44-06 | 离线模型命令与失败门 | [analyze_device_trace.py](../scripts/analyze_device_trace.py) | [M5](MOTIVATION.md#m5) |
| V44-07 | 注释模型与篡改回归 | [test_neuron_work.py](../tests/hardware/test_neuron_work.py) | [M5](MOTIVATION.md#m5) |
| V45-02 | 原始工件完整快照与资源准入 | [neuron_processing.py](../kb_workflows/hardware/neuron_processing.py) | [M5](MOTIVATION.md#m5) |
| V45-06 | 默认预览与离线处理入口 | [analyze_device_trace.py](../scripts/analyze_device_trace.py) | [M5](MOTIVATION.md#m5) |
| V45-07 | 自动处理关联与失败反例 | [test_neuron_processing.py](../tests/hardware/test_neuron_processing.py) | [M5](MOTIVATION.md#m5) |
| V46-01 | 按职责归类并合并重复实现 | [README.md](../README.md) | [M5](MOTIVATION.md#m5) |
| V52-01 | 设备配置驱动队列与后端预算握手 | [queue_worker.py](../kb_workflows/runtime/queue_worker.py) | [M5](MOTIVATION.md#m5) |
| V61-02 | 功能清单按动机归类与历史重复精简 | [FEATURE_INVENTORY.md](../docs/FEATURE_INVENTORY.md) | [M5](MOTIVATION.md#m5) |

## 维护与验证

清单按动机维护，详细理由集中到motivation映射。CUDA worker活动实测与保守扣额分开，范围见[资源预算](RESOURCE_BUDGETS.md#cuda进程活动计量与软件配额)。A/E/O使用实际计分时间；CUDA L2组装重放各阶段计数器、逐内核组模型、独占观察与安全/重复收据，缺失或含糊保持未知。支持范围及Neuron/传输缺项见[计数器与评分](WORK_COUNTERS.md#cuda-l2评分组装)。

清单、motivation映射和实现证据由`python -m scripts.check_motivations`核对，所有历史功能ID继续保留。软件回归和离线打包验证不替代真实设备、隔离、安全或模型效果实验。
