# KernelBench Pro 评测可靠性

可靠性检查统一通过 `scripts/run_batch.py` 和 TOML 配置运行。普通正确性、可选诊断和正式证据等级分别记录；没有采集到的检查保持 `not_measured`，不把它改成通过。

## 默认正确性检查

- 输出结构、shape、dtype 必须一致；整数和 bool 精确比较，浮点使用配置容差。
- NaN、Inf 的位置与符号必须一致。输入副本保持视图和别名关系，避免原地修改污染参考比较。
- 每个实例分别执行 candidate-first 和 reference-first，保存输出快照，并检查状态、RNG 和运行设置。
- 隐藏验证按冻结种子重新生成合法输入，并保留原始分布；附加压力输入遵守逐题数值域。
- 计时使用变化输入，逐次验证输出；隐藏失败不会被公开结果或旧分析入口覆盖。

实现集中在 [validation.py](../src/validation.py)、[reliability.py](../src/reliability.py) 和 [evaluate_task.py](../kb_workflows/evaluation/evaluate_task.py)。输入族及参考修订见 [OPERATOR_INPUTS.md](OPERATOR_INPUTS.md) 和 [REFERENCE_REVISIONS.md](REFERENCE_REVISIONS.md)。

## 配置

按实际需要开启诊断；预览不启动设备任务：

```bash
python scripts/run_batch.py --config configs/a6000.toml --dry-run
```

```toml
[run]
strict_validation = true
numerical_validation = true
allocation_validation = true
state_validation = true
cache_validation = true
throughput_validation = true
```

| 配置 | 检查目的 | 实现 |
| --- | --- | --- |
| `strict_validation` | 审核过的辅助输入、精度和参考信号 | [strict_validation.py](../kb_workflows/evaluation/strict_validation.py) |
| `numerical_validation` | 实际调用输入上的独立高精度真值 | [numerical_audit.py](../kb_workflows/numerics/numerical_audit.py) |
| `allocation_validation` | 未初始化候选存储的依赖 | [allocation_audit.py](../kb_workflows/evaluation/allocation_audit.py) |
| `state_validation` | 参数、缓冲区、训练模式和普通状态的更新 | [state_contract.py](../kb_workflows/evaluation/state_contract.py) |
| `cache_validation` | 同一输入对象更新内容后是否重算 | [cache_audit.py](../kb_workflows/evaluation/cache_audit.py) |
| `throughput_validation` | 公开同步与连续提交的吞吐差异 | [throughput_audit.py](../kb_workflows/evaluation/throughput_audit.py) |

设备、阶段和任务不支持某项检查时会明确拒绝或保留未知。完整配置与运行方式见 [BATCH_WORKFLOWS.md](BATCH_WORKFLOWS.md)。

<a id="strict-validation"></a>
## 辅助数值与输入契约

严格验证使用审核过的合法输入和数值预算。原始名义输入和正式计时尺寸不缩小；辅助小输入不能替代完整任务的原生验证。规格、源码摘要、oracle 选择及收据校验统一在 [specs.py](../kb_workflows/numerics/specs.py)。

`verification_contracts.json` 保存辅助契约，`operator_oracles.json` 选择独立公式。原 L1/L2/L3 的高精度算式、模型结构和随机掩码策略分别由对应 oracle 实现，不通过调用参考的数值 forward 冒充独立真值。未知任务、源码变化、非法输入域或不受支持的精度不会自动沿用旧契约。

进一步的误差预算、算式和输入域说明见 [NUMERICAL_VALIDATION.md](NUMERICAL_VALIDATION.md)、[COMPOSITE_NUMERICS.md](COMPOSITE_NUMERICS.md) 和 [MODEL_NUMERICS.md](MODEL_NUMERICS.md)。

<a id="timed-numerics"></a>
## 实际计时调用的数值验证

启用 `numerical_validation` 后，验证覆盖实际名义输入和变化后的预热、计时及配对输入，而非只检查几次单独的正确性调用。调用前准备参考状态、RNG 和 oracle 票据，调用后核对实际输出；这些准备与比较在计时区间外。原始被计时 callable 和已有完成屏障保持原路径。

真值计算使用当前模型状态，不提前消费候选调用，也不通过事后重算替代原调用的输出。Dropout 等随机算子保留实际掩码及模式。票据与输入、状态、源码和精度绑定；缺失或改变时不接受既有数值证据。独立公式仍受主机内存预算约束，不支持的情况明确报告。

相关执行入口为 [timed_numerics.py](../kb_workflows/evaluation/timed_numerics.py)；输入与精度校准见 [INPUT_CALIBRATION.md](INPUT_CALIBRATION.md) 和 [PRECISION_CALIBRATION.md](PRECISION_CALIBRATION.md)。

<a id="allocation-validation"></a>
## 候选分配检查

`allocation_validation` 在独立诊断进程中，使用两种确定性预填模式检查候选导入、构造和调用期间观察到的未初始化存储。浮点、复数、整数和 bool 分别按其存储类型填充；正常完整写入的候选应不依赖预填内容。视图、偏移、resize 和已有内容的保留按分配接口核对。

分配检查不替换普通正确性或性能样本，也不强制切换参考算法。原生 dispatcher、私有分配器、驱动或第三方路径未被覆盖时不能宣称检查完整；它不是所有 GPU 内存活动的安全证明。需要设备级内存错误检查时使用 [TOOL_CHECKS.md](TOOL_CHECKS.md)。

<a id="state-validation"></a>
## 模型状态检查

`state_validation` 对照参考检查参数、注册 buffer、普通状态、训练/推理模式和显式输入状态的变化。按审核过的调用顺序和非零参数更新再次运行，避免一次前向正确却缓存旧参数或旧状态。

普通属性的持久状态和初始化用途按任务策略区分；输入与状态的别名也必须保留。未知对象或无法可靠描述的状态保持未知，不能靠清空候选缓存或重建模型掩盖问题。状态观察不等于任意 Python 控制流或进程隔离证明。

策略与状态路径集中在 [state_policies.py](../kb_workflows/evaluation/state_policies.py) 和 [state_paths.py](../kb_workflows/evaluation/state_paths.py)。正式调用的前后状态绑定也由来源记录复用。

<a id="input-cache"></a>
## 同对象输入更新

`cache_validation` 保持输入对象、存储地址、视图和别名关系，改变合法内容后重新执行。只按 `id`、`data_ptr` 或首次输入值缓存答案的实现应在该检查中暴露。参数、离散标签、概率、量化比例等角色按任务策略更新，不随意替换成无效随机数。

原生正确性实例、种子、顺序和输出快照参与诊断。缺少完整实例、更新未生效、输入被污染或原始收据不一致时，不接受缓存检查。共享的输入复制/更新实现见 [input_reuse.py](../kb_workflows/evaluation/input_reuse.py)，逐题规则见 [cache_recipes.py](../kb_workflows/evaluation/cache_recipes.py)。

<a id="throughput-validation"></a>
## 同步与连续提交

`throughput_validation` 在公开输入上比较逐次完成和连续提交；保留实际候选状态与输入更新，不为了获得稳定结果而隐藏清理候选缓存。输出在完成后验证，原始事件及吞吐样本进入收据。

这项诊断检查吞吐行为，不替代原生墙钟性能，也不单独授予执行来源、算法效率或内存安全资格。并发活动时间使用区间并集；host dispatch、device busy 和整个请求墙钟分别记录。更完整的性能归因见 [WORK_COUNTERS.md](WORK_COUNTERS.md)。

## 结果、评分与恢复

`eval_results*.json` 的 metadata 保存实际检查和证据等级。`validation_summary.json` 汇总编译、公开正确、隐藏正确及有效计时数量；失败和缺失样本留在分母。有正耗时不等于获得计分资格，正式结果读取 `paper_analysis.json` / `eval_results_scored.json`。

源码、配置、输入/精度策略和评测器摘要参与恢复与评分校验；相关文件变化后旧证据失效。独立诊断不能补造实际计分过程的来源。L1/L2 仍需其余规定的校准、来源、计数器和安全证据，见 [HEADROOM_SCORING.md](HEADROOM_SCORING.md)。

验证命令为 `python -m pytest -q`。CPU 和模拟测试不证明实际 CUDA/Neuron 硬件正确性或性能；按当前用户约束，不运行设备、模型或供应商后端实验。

## 设计来源

采用并独立实现了 KernelBench-Hard 的多种子及合法数值域压力检查、KernelBenchX 的稀疏离群值与分阶段报告，以及 robust-kbench 的双向调用、输出快照和运行设置检查。没有迁入其整套任务或直接套用其精度/评分政策。

依据：[KernelBench-Hard SPEC](https://github.com/Infatoshi/KernelBench-Hard/blob/77d02f401b40c26226c8f5b753d7da46ff070e0a/SPEC.md)、[KernelBenchX §3.2.2](https://arxiv.org/html/2605.04956v2#S3.SS2.SSS2)、[robust-kbench 前向验证](https://github.com/SakanaAI/robust-kbench/blob/078f5bab29934a822268d59a4e707d449abf9b4e/robust_kbench/sandbox/correct_forward_fn.py)。
