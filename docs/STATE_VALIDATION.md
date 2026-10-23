# 模式、状态与参数更新

`--state-validation`为标准/隐藏验证增加独立的状态复核，默认关闭。TOML中使用：

```toml
[run]
state_validation = true
timeout = 1800
sanitizer_timeout = 600
```

配置预览不运行设备：

```bash
python scripts/run_batch.py --config configs/a6000-feedback.toml --state-validation --dry-run
```

该选项通过统一payload进入本地、Docker、Modal、HTTP队列和Harbor的原生验证器。独立复核worker关闭性能采样，在每个实际原生双向正确性实例前执行状态检查；覆盖该次任务的全部label和seed。标准、隐藏和最终独立重测各自收集自己的证据，隐藏信息不会进入开发反馈。已有原生失败不会因诊断而变成成功。

它和[重复检查及内存工具](TOOL_CHECKS.md)共用一次请求的总timeout及5秒清理预留，复核阶段受`sanitizer_timeout`限制。状态和重复检查可以在同一次复核中开启。第一轮原生worker的runtime、样本和冷启动成本用于计分；额外模型调用只进入请求资源核算。

## 检查的行为

参考和候选先取得隔离副本，检查注册状态的种类、shape、dtype、device、requires_grad、初始值和共享关系。快照不得共享原模型或另一个模型的Module、参数、buffer或普通Tensor属性存储，也不得改变注册状态的值和元数据。

同一对模型按`train → train → eval → train → eval`连续调用，每次重放该原生实例的RNG种子、复制输入及输出，比较输出和所有要求的注册状态。连续两次训练可识别“首次正确、第二次不再更新”的计数错误；切回训练及eval可检查累计状态。根training标志、no_grad后置状态也须符合请求。

有参数的模型分别在train和eval模式下重新取得隔离副本，先调用以填充缓存，再在**同一参数存储**上执行两轮确定性缩放和非零偏移，逐轮比较输出与状态。偏移使全零参数也发生变化；不同参数使用不同偏移。共享参数只更新一次。收据记录参考输出是否真的响应每轮修改；一次全体参数扰动不证明每个单独参数都影响输出。

整数/bool状态精确比较，浮点状态沿用该验证阶段的数值比较策略；非有限值位置和Inf符号仍须一致。状态映射不能提供比较函数、预期值或容差。进程环境和后端设置变化由外层审计恢复并记为失败，秘密环境变量值不进入收据。

## 候选状态映射

没有声明时使用同名注册状态。需要重命名或融合时，在`ModelNew`的类或实例上提供普通数据字典`kernelbench_state_map`，键是参考的完整注册名称，值是候选注册名称或声明式视图。未列名称仍按同名查找。例：

```python
class ModelNew(torch.nn.Module):
    kernelbench_state_map = {
        "first.weight": {"target": "packed_weight", "slice": [[0, 4], None]},
        "second.weight": {"target": "packed_weight", "slice": [[4, 8], None]},
        "projection.weight": {"target": "projection_transposed", "permute": [1, 0]},
        "norm.running_mean": "block.norm.running_mean",
    }
```

此例只说明格式；实际shape须与对应参考一致。每个slice项是`null`/Python `None`或非负、非空、界内的`[start, stop]`；维数须完整。操作顺序为slice、permute、reshape。permute必须是维度双射；reshape须为显式正整数shape，并能用`view()`保持原存储。不能使用索引gather、广播、任意函数、property或动态mapping方法。

目标必须是实际注册参数或buffer，并保持参考状态种类。独立参考状态不能映射到重叠的候选字节范围；共享参考状态必须保持同一个候选视图。对复杂交错视图采用保守范围检查，无法确认不重叠时拒绝。运行中更改映射、替换要求的状态对象或参数存储不能通过。候选可以保留额外缓存，但实际缓存响应仍须通过输出和更新探针。

## 收据与范围

`metadata.tool_checks.control.result.metadata.state_audit`记录逐实例输入结构、seed、状态绑定与摘要、共享组、模式序列、参数更新数和失败原因。`mode_state`从完整实例及步骤重新推导；删实例、改seed、改容差、丢绑定、少做参数更新不能只靠重写汇总或外层摘要获得通过。评分和恢复使用同一重算规则，评测器指纹包含状态实现。CPU模拟收据不能用于设备资格。

状态检查入口不再受17道FP64附加契约题的选题限制；这不等于250题均已适配或在设备上验证。当前支持注册的普通稠密Tensor/Parameter，及上述存储视图。参考含未注册Tensor属性（包括普通容器中的Tensor）、需要非线性重参数化、未初始化或稀疏状态、动态注册/替换状态的任务，须先提供对应任务适配；缺适配返回失败，不忽略。任意Python对象中的状态、标量业务计数和每个子模块的自定义模式语义也尚无通用证明，因此R-07继续保留部分实现状态。

这是一套有限功能探针，依赖可信worker；不认证任意Python/native代码安全，不证明全部参数的因果使用，也不授予完整L0。严格FP64模式中的小实例复用同一状态实现。当前只运行CPU和模拟回归，没有运行GPU、Trainium、模型推理或训练实验。
