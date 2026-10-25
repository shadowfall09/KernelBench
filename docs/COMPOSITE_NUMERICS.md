# L2复合算子的独立数值参考

`numerical_validation`现覆盖原250题：L1的100题显式公式、L2的100题复合计算，以及[L3的50题模型算式](MODEL_NUMERICS.md)。本文描述L2的冻结符号图与独立CPU FP64算式；名义配方已覆盖原250题；候选前的合格参考集合与排除账本已实现，[阈值拟合与独立种子验证](PRECISION_CALIBRATION.md)代码已实现。当前证据为CPU小配置、meta元数据和模拟测试，没有运行GPU、Trainium或预训练模型实验。

## 入口与范围

```toml
[run]
level = 2
numerical_validation = true
oracle_memory_mb = 4096
```

预览：

```bash
python -m scripts.run_batch --level 2 --start 1 --end 100 \
  --numerical-validation --dry-run
```

预览不导入Torch、不创建输入或运行模型。正式请求复用既有独立诊断控制进程，对实际标准/隐藏双向正确性实例逐一复核；检查不替换原始计时样本。启用数值验证后，不论是否配置输入配方，首次、预热、每个计时和配对输出均须独立FP64合格，见[计时验证](TIMED_NUMERICS.md)。原250题名义配方及新基线见[输入校准](INPUT_CALIBRATION.md)。

## 来源、构造和图绑定

每题的参考SHA、图摘要、节点数和随机节点记录在[算子清单](../kb_workflows/numerics/operator_oracles.json)中。完整的图、使用的模块构造属性、状态形状和常量身份记录在[冻结程序](../kb_workflows/numerics/composite_programs.json)。图身份绑定实际的通道、分组、卷积/池化参数、归一化形状、epsilon、激活近似和Dropout概率等构造配置；合法参数数值更新不需要改变形状。

worker先将参考的已存在参数和buffer映射为meta元数据，再复制Python模块结构，不先深拷贝一份真实GPU权重。FX符号追踪会执行参考Python控制逻辑，shape propagation会执行meta层forward；它们用于构图和推导形状，不计算真实参考数值输出。程序摘要必须与冻结清单一致。构造变化、未知函数/方法/模块或无法追踪的控制流明确失败，不回退到`reference.double().forward`。

FX的图解释和叶子模块机制可参见[官方FX文档](https://docs.pytorch.org/docs/stable/fx.html)；实现依据本环境Torch 2.12.0源码与回归。冻结图的代码与依赖身份进入评测器摘要，不将其他Torch版本或设备部署视作已验证。

## 哪些算式独立实现

真正的FP64计算逐节点执行，既不调用生成图的数值forward，也不调用原模块的融合数值forward：

- Linear使用CPU FP64矩阵算术及实际存储的权重/bias。
- 标准/转置/分组卷积复用逐空间kernel偏移的收缩/散射参考，不调用原卷积算子。
- 固定池化使用显式窗口与索引；adaptive average pooling按每个输出窗口的floor/ceil边界求均值。
- BN、IN、GN、LN使用实际模式、仿射参数、running state及显式中心化/方差或分组公式。
- GELU保留精确与tanh近似的区别；Mish、Softplus、HardSwish、LeakyReLU、Softmax和LogSumExp等有显式公式。
- 激活的inplace行为及视图别名保留；图内Tensor常量先保留观测到的原dtype舍入，再提升FP64。整数与离散结果不被浮点提升。

CPU Tensor算术、索引及基础归约仍是受信计算原语，不是底层库无误或精确实数计算的证明。该解释器只对已冻结、已审核的图提供资格，不声称支持任意Python模型。L3的RNN、attention、动态控制与状态语义使用另行冻结的模型计划及专用算式。

## Dropout：随机掩码与数值运算分开

Dropout不被关闭，也不通过切换eval让训练路径变得确定。独立控制在每例开始重放该例seed。对活跃的Dropout节点，使用原dtype、形状、stride、设备与RNG，在全一输入上调用已捕获的原生Dropout采样器，取得掩码；CPU FP64算式在该掩码条件下计算缩放后的数值。真值完成后再次重放seed，分别调用参考和候选，整个控制过程恢复外部RNG。

这里依赖Dropout掩码与输入值独立的原生实现语义；[官方说明](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.dropout.Dropout.html)描述了其Bernoulli采样及训练/评估行为。CPU测试核对了当前代码路径，原dtype/device掩码重放在真实GPU或Neuron上尚未实验验证。它不是完全脱离框架随机原语的oracle，也不支持借此替代其他未审核的融合随机算法。

收据保存逐随机节点的模式、概率、布局、采样数、保留数和mask摘要。评分复核完整节点集合、冻结程序、dtype/device、形状计数与摘要格式；缺掩码、改概率或把数值CPU执行冒充meta推导都不能只改外层“passed”通过。摘要是完整性记录，不是签名认证、随机性证明或任意代码安全沙箱。

## 数值、信号和资源

默认使用清单已有的初始分精度预算；可选阈值文件在该上限内拟合并冻结更严格的政策，不为复合参考自动放宽容差。本次没有从实机误差生成阈值文件。参考和候选均须通过同一逐元素及RMS门；整数/bool、结构、dtype及非有限表示仍单独核对。

第83题的冻结公式在`min_value=0`时先`min(x,0)`再`clamp(min=0,max=1)`，恒为零。Dropout无法恢复参考信号。原文件和题目分母保留，独立数值比较即使相等也不能通过`reference_signal`；记录这一问题，不把相等输出当作有效测试。

原始构造和名义输入形状的meta检查覆盖100题。CPU数值对照使用显式的小构造配置，只用于算式单元测试；worker不会用这些小配置替代正式任务。构造差异使冻结程序摘要不匹配，不能用小实例收据冒充原题资格。

`oracle_memory_mb`仍是主机增量内存估计准入包络。L2先用meta形状汇总输入、状态和图中Tensor元素，按每元素64字节保守估计复制和中间量，然后才进行实际FP64状态复制或原生掩码采样。它不是OS RSS硬限，也不是GPU sampler workspace保证。大名义尺寸可能显著超出默认4096 MiB；超包络、分配失败、超时或缺完整实例都不能通过，不自动缩尺寸或调整模型参数。

新清单、算式、程序和策略进入评测器与请求身份，贯穿本地、队列、Docker、Harbor、最终冻结、研究/RL和恢复。合格参考集合已在候选前冻结；阈值校准代码已实现；更完整执行来源/性能证据仍是剩余代码范围，真实设备部署未运行。

数值验证现在不依赖名义输入配方即可检查首次、预热与全部计时输出；逐次误差/信号/程序证据进入评分和恢复，见[计时数值验证](TIMED_NUMERICS.md)。配对每轮共用一次调用前真值，不额外推进状态或RNG。
