# L3 模型的独立数值复核

`numerical_validation`现在覆盖原L3的全部50题，与L1的100题显式公式、L2的100题复合算式合计250题。检查使用实际原生正确性输入和当前参考参数，不调用参考的数值`forward`来生成真值。默认关闭；它不替换原始计时，也不自动提供完整L0或设备资格。

```toml
[run]
level = 3
numerical_validation = true
oracle_memory_mb = 4096
```

只预览来源与覆盖，不导入设备SDK、不分配模型输入：

```bash
python scripts/run_batch.py --level 3 --numerical-validation --dry-run
```

## 解释器与模型算式

[模型清单](../kb_workflows/model_programs.json)冻结50个来源构造计划，包含完整module树、公开标量属性、状态形状/种类、算式类型与dropout调用顺序；[任务规格](../kb_workflows/operator_oracles.json)另绑定参考SHA和初始误差政策。浮点存储dtype、train/eval模式与物理参数布局可以按已有协议变化；这些变化不改数学拓扑，也不能取消实际输入上的数值门。维度、层数、激活形式、概率等构造变化不能复用原计划。

| 题目 | 独立算术路径 |
| --- | --- |
| 1–28、31–32、45，共31题 | 来源Python符号图与逐层CPU FP64解释器；卷积、窗口、BN/归一化、激活、矩阵等复用已审核算式。 |
| 28、31、32的内置attention/encoder | 独立Q/K/V投影、分头、缩放、mask、softmax、加权和、输出投影、残差、前馈和pre/post norm，保留源码的batch-first方向。 |
| 29–30 Swin | 独立窗口/shift或padding、分头空间MLP/余弦attention、相对位置bias、patch合并和归一化顺序；不把SwinMLP与SwinV2的不同顺序合并。 |
| 33–34普通RNN | 保留当前hidden或传入h0、逐时间递推与输出投影；33的显式`copy_`保留目标状态存储的舍入。 |
| 35–38 LSTM、39–42 GRU | 显式门公式、多层、正反方向、h0/c0和返回状态；GRU保持PyTorch的reset-after隐藏偏置形式。 |
| 43–44、50 GPT类 | 因果mask、声明的softmax/ReLU、残差和精确的tanh GELU形式；50不使用源码中未调用的c_proj。 |
| 46–47 NetVLAD | 分配、BN、ghost截断、残差累积、逐簇与最终L2归一化。 |
| 48–49 Mamba SSD | 分块segment sum、块内/跨块递推与可选initial state；分别返回Y或最终状态。 |

CPU Tensor矩阵、索引、归约及einsum仍是受信原语，FP64也不是精确实数算术。未知模块/构造、未审核的Transformer内部随机概率、投影RNN等不回退到原生融合层或参考`.double()`。当前冻结的Transformer与LSTM/GRU构造使用零内部dropout；不为通过检查把用户实际模型切换成eval或关闭其随机性。

## meta规划与随机性

参考的隔离meta副本只提供形状、布局和内存元数据。LSTM/GRU的已知矩形输入/状态尺寸用审核过的公式验证，避免在meta上逐步展开巨型序列。模型算式的数值执行不调用这些meta `forward`，也不调用原生融合attention、RNN、归一化或卷积来取得真值。

普通`nn.Dropout`只重放值无关的原生随机mask：ones保留观察到的dtype、shape、stride、设备和RNG，随后使用FP64算术缩放。每次调用记录顺序、模式、概率、保留数量和摘要；共享Dropout模块的多次调用分别保留。零概率的内置encoder路径明确是恒等操作。掩码重放依赖安装的原生随机primitive，不能视为跨设备随机实现证明。

MiniGPT的`mlpf`闭包在普通deepcopy后仍可能引用原ModuleDict；meta观察单独重绑到隔离副本，数值解释器直接按已审核算式读取当前层参数，不执行该闭包。

## 内存与参考问题

准入先于实际FP64输入/状态复制和随机采样。L3估计使用输入和状态元素、最大meta结果及显式attention矩阵workspace的保守64倍包络，并取FX计划估计的较大值。它不把逐时间步的同一参数视图累计成独立权重分配。该估计不是操作系统RSS硬限制，也不包含任意候选workspace；仍受请求墙钟约束。默认4096 MiB不足以复核某些名义模型时，保留`not_measured`，不能缩小原输入、层数或状态来过门。

L3/35省略h0/c0时，源码的`torch.zeros`没有指定dtype，固定使用当时默认dtype。小配置CPU回归确认：默认FP32环境下，FP64参数/输入与省略状态会因dtype不同而失败；显式同dtype状态是另一条合法调用路径。清单保留这个参考问题，正式原输入工厂和参考代码未改写。meta接受形状不等于其他存储精度或GPU后端已接受该参考。

误差门默认使用公布的初始政策，参考与候选都必须对独立真值合格。全题名义配方已接入参考专用校准；候选前冻结合格参考集合/排除账本已实现；有无配方的原生逐次计时强门已实现，[参考专用阈值校准](PRECISION_CALIBRATION.md)已实现，只在原上限内收紧政策，验证失败不重拟合，不因为添加适配器就放宽误差或删除失格样本。

## 证据与验证范围

`model_receipt`绑定任务、构造/算式摘要、meta计划、dropout顺序与mask，FX路径还保留内部图收据。缺失/改摘要、执行参考数值forward、漏掩码、变模式/布局或不完整原生实例集合不能只修改外层passed取得资格。复用已有独立控制worker，进入原生、队列、Harbor、最终验证、评分与恢复的相同请求身份。

全部50个原始构造和名义形状仅在meta上检查。数值单元测试使用36个明确的小构造配置，覆盖全部专用模型算式及代表性CNN/MLP；禁止参考/融合数值forward的负例、两种模式、原生随机mask、可选状态和收据删改另行验证。原生入口用小输入、原始LeNet构造验证；小配置收据不能冒充冻结原题构造。

没有运行GPU、Trainium、原尺寸模型、预训练模型、容器、校准或训练实验。原生算式、模型构造和收据的代码验证不提供上述实验结论。实现见[模型解释器](../kb_workflows/model_oracles.py)、[独立模型算式](../kb_workflows/model_math.py)、[attention/encoder算式](../kb_workflows/model_layers.py)、[证据复核](../kb_workflows/model_specs.py)和[回归](../tests/test_model_oracles.py)。

数值验证现在不依赖名义输入配方即可检查首次、预热与全部计时输出；逐次误差/信号/程序证据进入评分和恢复，见[计时数值验证](TIMED_NUMERICS.md)。配对每轮共用一次调用前真值，不额外推进状态或RNG。
