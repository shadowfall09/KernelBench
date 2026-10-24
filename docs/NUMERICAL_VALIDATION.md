# 原生输入的独立数值复核

`numerical_validation`增加原生输入上的CPU FP64/complex128数值门，当前覆盖原L1全部100题显式公式、L2全部100题来源绑定的复合算式，以及L3全部50题的模型算式，合计250题。默认关闭，不替换原生正确性或性能协议，不自动授予完整L0。

```toml
[run]
level = 1
numerical_validation = true
oracle_memory_mb = 4096
timeout = 1800
sanitizer_timeout = 600
```

只预览配置及源码覆盖：

```bash
python scripts/run_batch.py --config configs/a6000-feedback.toml \
  --numerical-validation --oracle-memory-mb 4096 --dry-run
```

`--dry-run`不加载Torch，不分配题目输入、不运行设备；输出`numerical_coverage`及已记录的参考问题。源码SHA与[适配器清单](../kb_workflows/operator_oracles.json)不一致时，预检和worker都拒绝套用旧公式。详见[L2复合数值参考](COMPOSITE_NUMERICS.md)与[L3模型数值参考](MODEL_NUMERICS.md)。生产任务仍使用自身声明的数值门；本选项不接受它们，也不允许通用`.double()`冒充独立适配器。

## 实际检查什么

原生worker先按现有协议完成正确性及计时。随后独立复核worker关闭性能采样，在每个实际标准/隐藏双向正确性实例前，先由CPU公式计算真值，再在隔离模型与输入副本上分别调用参考和候选。每个实例重放原种子，恢复RNG和进程设置，不推进原生模型。

原始shape、stride、输入值和模型超参数不被重采样、缩小或自动修复。整数、bool及标签保留角色；输入浮点存储的实际值被提升到FP64，参数/buffer同样使用其实际存储值。参考和候选输出搬到CPU时保留原dtype；真值不先舍入到输出dtype。

独立公式不调用参考或候选的`forward`，也不调用对应的原生融合算子：

- 矩阵家族使用CPU FP64矩阵算术，并保留转置、对角、三角等题义。
- 卷积按空间kernel偏移做逐组收缩，转置卷积按合法目标索引累加；保持stride、padding、dilation、groups、output padding和bias。
- 池化显式枚举窗口，处理padding、ceil边界、平均池分母和最大池返回索引；索引与并列结果精确比较。
- 归一化按声明轴显式计算中心化/方差或范数，保留仿射状态及train/eval中的running state；attention显式计算缩放、稳定softmax和加权和。
- 激活、scan及loss保留精确GELU与tanh近似的区别、exclusive/reverse/masked语义、cross-entropy标签、KL零target项及triplet epsilon/margin等参数。

实现仍依赖安装的CPU Tensor算术和FP64精度，不能解释为底层库无误或数学精确算术的证明。完整公式见[实现](../kb_workflows/operator_oracles.py)。

## 误差、参考自资格与信号

每个浮点叶子同时满足RMS和所有元素误差门，公式与[严格契约](STRICT_VALIDATION.md)相同。当前清单公布的是**初始误差预算**，没有运行设备校准；不能把“配置了阈值”写成“阈值已校准”。参考自身也必须满足同一FP64门；不为使参考通过而自动放宽阈值。

结构、shape、dtype、离散输出和NaN/Inf位置另行核对。参考资格独立于候选判断：oracle shape或结构不一致时，缺少数值叶子不能算参考合格；候选自身结构错误也不能凭空否定一个已通过参考自检查的参考。

每个实际实例还记录参考输出的有限值、RMS、最大值、超过绝对阈值的比例和常数标记。该次任务至少有一个信息量足够的实例才能通过`reference_signal`；常数本身不是语义错误。弱信号会被报告；可选[参考输入校准](INPUT_CALIBRATION.md)已实现原250题名义值分布选择及相应新基线，隐藏分布不自动变换；合格参考集合现可在候选前冻结，阈值校准与未配置输入校准时的完整计时数值门仍由R-04推进。

## 主机内存与请求预算

`oracle_memory_mb`是1–1048576 MiB的增量主机内存**估计准入包络**，默认4096 MiB。worker在复制或执行oracle前，根据实际输入、状态、输出及attention/卷积等中间量估计峰值。超包络记`not_measured`，不缩小原尺寸、不换低精度；分配失败、超时和不完整执行也不能通过。

该估计不是OS RSS硬限制或无OOM保证；实际操作系统资源限制仍由[受限验证器](ISOLATION_REGRADE.md)和部署负责。原生运行、可选工具与复核共用请求`timeout`，保留5秒清理，复核受`sanitizer_timeout`阶段上限约束。大尺寸CPU FP64可能很慢；本实现没有执行原尺寸设备实验。

## 证据、评分与恢复

`metadata.tool_checks.control.result.metadata.numerical_audit`保存规格摘要、数值策略、内存包络，以及逐实例label、seed、输入签名、估计内存、数值门、参考自资格、信号诊断和错误。

评分和恢复重新核对实际原生实例集合、冻结规格/误差/内存政策、RMS限值、误差计数和逐例完成状态。缺实例、改seed、加宽误差或内存政策、把非零失败计数改成“passed”均不能只改外层摘要取得资格。模拟执行不能建立设备资格。该路径进入本地、Docker、Modal、HTTP队列和Harbor的同一监督进程，可与状态和重复检查共用一次复核。诊断时间及输出不替换最初的性能样本。

## 发现的参考问题与未完成范围

原L1第92题的旧版曾沿`dim`拼接零、却裁剪第0维，已被独立公式检出。当前revision 2沿正确轴补零、累加与裁剪；独立公式采用inclusive sum后移位。修订保留前后源码身份和旧反例，选集分母不变，旧结果不能直接转为新版通过。详情及五类扫描隐藏输入见[参考修订](REFERENCE_REVISIONS.md)。

另有固定CPU反例表明，BF16 triplet loss在接近抵消的结果上，参考也可能超出公布预算；与参考相同的候选不会因此自动获得资格。测试保留此反例，而不是放宽误差门。

合法空Tensor及精确传播匹配的全非有限Tensor保留结构/dtype契约，有限误差测量为`null`，均不作为有效参考信号；非空参考仍须提供信息量。当前新增路径覆盖原250题，原17题小实例严格契约继续保留；原250题可选输入校准和强数值新基线已实现。候选前冻结合格参考集合/排除账本已实现；设备阈值校准代码及未配置输入校准时的完整计时数值门仍未完成。本文的CPU/模拟回归不能替代设备或模型实验。没有运行GPU、Trainium、真实模型、容器或训练实验。

原L1现有[算子专用隐藏输入](OPERATOR_INPUTS.md)也进入这条实际输入复核。新近常量CPU FP32 BN/IN反例保留参考失格；五题声明非有限域，其他算子不使用通用NaN注入。

原L2全部100题现有[复合FP64算式与冻结图](COMPOSITE_NUMERICS.md)，复用实际输入诊断控制。L1仍使用显式独立公式；符号图的Python/meta调用与数值计算分开记录。
