# 逐次同步与连续提交的吞吐对照

`throughput_validation`默认关闭。它是独立诊断，不替换原生性能样本。

```toml
[run]
throughput_validation = true
throughput_calls = 8
throughput_repeats = 4
throughput_memory_mb = 1024
timeout = 1800
```

调用数允许2–128；重复数允许4–100且必须为偶数，以交替运行参考和候选。每种模式固定预热3次。CUDA和Neuron共用观察与同步接口；实现只做了CPU队列模拟和Neuron前端模拟，没有运行设备或模型实验。

## 两个新进程测相同流水线

父进程先运行原来的原生评测，再为latency和burst各创建一个新的正确性worker。它们使用相同候选来源、初始化参数和可信名义输入工厂，在首个实际公开正确性实例处进入对照。输入种子、轮次顺序与随机序列种子由同一个父进程nonce导出；两个进程的输入完整字节摘要必须相同，不重新挑选容易通过的实例。

| 模式 | 诊断计时区间 |
| --- | --- |
| latency | 每次执行forward、消费输出并复制到独立快照，再做设备级同步；记录逐次耗时及总和。 |
| burst | 连续执行N次相同的forward/消费/快照操作，最后做一次设备级同步，记录总耗时与每次均摊。 |

输入集和输出快照在计时前分配，快照复制本身计入两种模式。这样即使候选反复返回同一个Graph或工作区缓冲区，也能在最后同步后逐次验证全部输出，不只核对最后一次。结构、shape、dtype、device均核对；启用名义输入校准时，这些输出也须通过相应独立复核。

这是包含输出快照的消费流水线对照，不能把它的均值当作原始forward时间。快照复制要求返回值能够立即由当前流消费；侧流实现必须建立正确依赖。流间依赖与内存生命周期依据[Torch CUDA语义](https://docs.pytorch.org/docs/stable/notes/cuda)及[record_stream说明](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)。没有运行真实侧流或Graph实验。

每个模式在自己的新进程内连续推进参考和候选状态，不在每次调用间重置模型。两个模型调用次数相等；每段序列开始前重放相同RNG状态，序列内部自然推进。模型根模块的training、每次调用后的no_grad与CPU线程数被记录；环境、精度后端、梯度和确定性设置变化会失败并恢复，不强制切换算法。它不替代完整子模块状态审计。

## 差异标记与评分

每个重复分别计算参考、候选的相对差异`abs(latency_ms - burst_ms) / latency_ms`。任一超过10%就记录`review_required`；边界只容许1e-12的浮点舍入差，不提供可调放宽阈值。

正常启动开销摊薄、并行、噪声或同步错误都可能产生差异，标记不等于作弊或错误答案。保存的原生正确性和原始时间保留；启用本项时，论文性能评分把未消除的差异标为`throughput_discrepancy_requires_review`并不给性能分，失败槽位仍留在分母。兼容评分门沿用这项资格判断。不能删除标记或改摘要后恢复计分。

每段结束后还记录第二次设备屏障的耗时。它只是额外排空探针，既不单独证明有残留工作，也不证明任意后台线程以后不会再提交工作。正式计时仍只取最初原生进程的样本、冷启动和配对记录，诊断毫秒数不覆盖它们。

## 原尺寸存储准入与未知值

先在meta上推导真实参考的输出结构，再估算三组输入、两组输出快照、哈希临时量、消费标量和预热存储。正stride的存储跨度也计入估算。元数据推导失败或超过`throughput_memory_mb`时保留`not_measured`，不缩小模型、输入、N或重复数，也不改用后续实例。

该包络只约束本诊断估算的Tensor存储，不包括已存在模型、编译器/workspace或Python对象；它不是OS或设备内存硬限。CUDA的`CUDA_LAUNCH_BLOCKING=1`会破坏异步对照，保留未知，不在运行中悄悄切换；Neuron不受这个CUDA变量限制。

隐藏验证不会运行本诊断或提供隐藏工厂。所有子进程仍共用原请求的墙钟上限和每阶段`sanitizer_timeout`，没有额外设备预算。超时、设备不一致、输入内容不一致或丢失调用，均不能取得已配置检查的资格。

## 收据与部署

`metadata.tool_checks.throughput_controls`保留两个进程的请求、退出状态和设备身份。每个`throughput_audit`记录首个原生实例、存储准入、名义输入签名、完整输入/输出字节摘要、种子与平衡顺序、调用数、逐次/整段时间、线程/no_grad/根模式及排空探针。

评分与恢复重新计算覆盖、存储估计、时间均值和10%标记，核对普通正确性复核，拒绝缺模式、改seed、少输出、改时间总和和模拟设备冒充实测。政策进入最终冻结、研究/RL合同、运行分组、队列预检及Harbor私有配置；旧正结果不能跨政策复用。

实现与证据见[对照收集和重放](../kb_workflows/evaluation/throughput_audit.py)、[进程监督](../kb_workflows/evaluation/tool_runner.py)、[评分政策](../kb_workflows/analysis/paper_analysis.py)及[CPU/模拟回归](../tests/evaluation/test_throughput_audit.py)。完整GPU来源、任意线程/allocator、所有异步提交与硬件完整性仍未完成，R-09保持部分实现。
