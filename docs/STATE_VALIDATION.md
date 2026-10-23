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

同一对模型按`train → train → eval → train → eval`连续调用，每次重放该原生实例的RNG种子、复制输入及输出，比较输出、所有要求的注册状态及普通持久Tensor状态。连续两次训练可识别“首次正确、第二次不再更新”的计数错误；切回训练及eval可检查累计状态。根training标志、状态所属子模块的train/eval以及no_grad后置状态也须符合请求。子模块通过状态映射定位，融合模型可将多个参考所有者映射到同一候选所有者；每个调用前后的所有要求模式均记录并比较。

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

注册目标保持参考参数/buffer种类。普通Tensor参考状态可以映射到普通Tensor或注册buffer；它不是注册参数的替代声明。独立参考状态不能映射到重叠的候选字节范围；共享参考状态必须保持同一个候选视图。对复杂交错视图采用保守范围检查，无法确认不重叠时拒绝。运行中更改映射、替换要求的状态对象或参数存储不能通过。候选可以保留额外缓存，但实际缓存响应仍须通过输出和更新探针。

## 收据与范围

`metadata.tool_checks.control.result.metadata.state_audit`记录逐实例输入结构、seed、状态绑定与摘要、共享组、模式序列、参数更新数和失败原因。`mode_state`从完整实例及步骤重新推导；删实例、改seed、改容差、丢绑定、少做参数更新不能只靠重写汇总或外层摘要获得通过。评分和恢复使用同一重算规则，评测器指纹包含状态实现。CPU模拟收据不能用于设备资格。

## 普通Tensor状态与逐题政策

状态名以`@`开头，例如`@hidden`、`@cache["states"][0]`和`@cache.inner.value`。字典按真实键名和类型定位，改变插入顺序不改变绑定；字符串键`"1"`和整数键`1`不同。支持普通模块属性、内建dict/list/tuple、原生OrderedDict和具有普通`__dict__`的对象；不读取property或执行查找回调。只作为框架别名存在、指向同一注册Tensor对象的GRU/LSTM/RNN扁平权重列表由注册状态覆盖，不重复要求候选实现框架内部缓存。

普通持久状态允许forward替换Tensor对象；每次都按当前参考状态比较形状、dtype、device、requires_grad、值与共享关系。注册参数仍要求同对象、同存储；参数更新探针不会因为允许持久状态替换而放宽。快照必须保留普通Tensor的值、元数据、路径，并隔离原模型和另一模型的存储。还须保留复制前后的Tensor对象别名与重叠存储视图关系；自定义deepcopy不能修复原模型断开的共享关系，或把失效缓存重新连接到参数来绕过更新探针。独立参数在框架平铺存储中不重叠的打包位置可以变化。候选声明例如：

```python
kernelbench_state_map = {"@hidden": "running_hidden"}  # running_hidden为候选注册buffer
```

原250题和生产41题使用绑定当前参考SHA的`state_policy`，通过CLI/生成/队列/私有Harbor统一预检。所有注册参数和buffer都检查。全291题的原始构造仅做meta回归，普通Tensor角色与逐题政策完全匹配；不会调用名义输入工厂或构造真实大权重。其中两项有明确规则：

| 题目 | 普通属性 | 规则 |
| --- | --- | --- |
| L3/33 VanillaRNN | `@hidden` | forward复制显式初始状态、迁移到输入设备、再替换为循环结果；逐调用比较持久状态，允许参考对应的替换及初始CPU/输入设备放置 |
| L3/41 GRUBidirectional | `@h0` | forward仅使用输入参数h0，不读取self.h0；以源码绑定的理由记录为未使用初始化，候选可省略该属性，输入h0仍须通过原生比较 |

其余任务不得静默新增未声明的普通Tensor参考状态。删改政策、漏掉持久状态、改动未使用角色或更换参考源码不能取得资格。持久状态的全部更新仍在独立诊断worker中运行，原生计时和状态轨迹不替换。

循环对象、容器子类、未适配slots、未注册子模块、稀疏/量化/未初始化状态、复杂存储重叠及动态注册参数仍需要专门适配；不作为通用支持。候选非线性重参数化须同时暴露可映射的规范参数，并在原地更新后同步派生缓存；映射不执行解码或比较代码。任意Python业务状态、瞬时框架篡改和每个参数的因果使用不属于这组有限探针的证明。

R-07按最初的逐题模式、BN、参数可变性和输入域契约验收，软件已实现；输入域与同对象变化还由[R-06输入族](OPERATOR_INPUTS.md)和[R-09更新检查](INPUT_CACHE.md)覆盖。真实设备实验按用户要求不运行，不把任意表示的通用证明追加为验收条件。

这是一套有限功能探针，依赖可信worker；不认证任意Python/native代码安全，不证明全部参数的因果使用，也不授予完整L0。严格FP64模式中的小实例复用同一状态实现。当前只运行CPU和模拟回归，没有运行GPU、Trainium、模型推理或训练实验。
