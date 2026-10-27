# 原生调用计数器与审核后的 A/E 分解

计数器采集复用原生评测序列，不再另造一个初始实例测一次前向。默认仅预览；没有运行 ncu、CUPTI、GPU、Neuron、模型、容器或校准实验。所有插桩时间属于诊断，不写回正式分数。

## 入口

```bash
python -m scripts.audit_work_counters --config /path/to/device.toml \
  --recipe /path/to/recipe.json --problem 2 --sample 0 \
  --output /path/to/new-counters.json
```

预览不导入设备SDK、不查工具、不创建输出。仅在实验已授权时增加 `--execute`。执行使用同一总超时：先冻结工具路径、字节与版本，再运行原生对照和CUPTI来源采集，最后在新进程中采集完整序列的计数器；每个阶段均受父进程剩余预算约束。复用local、Docker、Modal、HTTP队列，CUDA工具缺失或Neuron请求不能降级成通过。

默认目标为候选的原生 `evaluate`，覆盖双顺序正确性、所有预热、正式计时及实际显存前向。配对路径使用 `--task paired_timing --baseline-variant eager`，变体须填写实际选中的来源绑定基线；加 `--target reference` 测该真实配对基线，不再用另造的公开参考实例代替。原生初始化、连续状态、输入角色/精度、数值门和实际调用顺序保持原样。旧的单次 `--seed` 路径和 `counter_worker.py` 已移除；原生评测seed来自配置，计时输入计划在候选前冻结。

## 逐次采集与重放

采集进程在候选导入前捕获profiler start/stop和NVTX push/pop函数。每次被观察的调用使用父进程新nonce和唯一ID标记，独立启停profiler；前向异常也关闭采集并弹出范围，栈深须平衡。NVTX仅关联调用，不作为源码归属证据。

ncu禁用用户配置和额外kernel重命名，显式关闭cache flush与时钟控制，采用kernel replay、base units、完整子进程范围和原始逐launch CSV，开启NVTX并将最近push/pop范围打印为标签。不按NVTX过滤或只取前若干kernel。解析须覆盖每项请求指标、实际PID和device；标签必须唯一对应可信窗口，且每个调用的launch数与独立来源采集中的kernel数相等。嵌套NVTX覆盖标签、无范围/后台工作、缺launch或重复身份无法匹配时明确失败，不凭名称猜测或把未知记录丢掉。[NVIDIA CLI说明](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)记录了这些选项；push/pop深度按当前Torch绑定核对。kernel replay仍会改变并发和缓存历史，不保证与未插桩的工作量相同；这些SOL和分解保留为审核后的诊断模型。真实部署的vendor输出仍未实验验证。

原生对照、来源采集和ncu三阶段必须具有相同的完整调用序列：完整逻辑输入/输出字节、布局、模式、seed、顺序、当前源码、精度和设备绑定逐项核对，所有输出仍过原来的数值检查。原生计时样本和窗口由[来源重放](KERNEL_ORIGIN.md)再次检查。显存前向新增独立memory调用，输出、原始/调整峰值与验证副本增量均须匹配各自原生收据；其他独立诊断仍不在此序列内；跨进程可观察一致不证明未插桩路径或任意参数状态等价。

来源与计数器CLI共用`--scored-result FILE`读取既有L1/L2原生结果，按实际计时进程生成的seed及完整调用观察关联三阶段，不能用另抽的一组seed代表原进程。候选公共/配对请求支持该关联；参考角色不能借用候选收据。离线时间表单列`native_scored_wall_ms`，仍用CUPTI同一时钟计算O，不将实际时间与另一个进程的busy相除。默认预览只核对已有工件，关联本身不提供路径等价、真实硬件计数或正式来源资格。

嵌入的原生收据现为v2，实际/对照/来源/ncu按调用核对模型前后数据、入口和输入/状态别名摘要。状态更新可以合法发生，但不能把不同参数或入口下测到的工作量借给实际调用；未知状态不能补成空状态。取摘要不进入原生计时或CUPTI/NVTX窗口，也不把这组观察当作host控制流等价证明。

报告保留一份原始CSV、可用时的ncu二进制报告、源码采集和计数器进程记录，按调用重算，不向每个调用复制完整原始报告。协议为 `kernelbench_pro_native_work_counters_v2`；`counter_task.validate()`重放全部关联及工作量，不信任保存的passed标记或重封的汇总。旧 `kernelbench_pro_work_counters_v1` 的原始CSV读取和离线模型API仍可用于历史诊断，但不能建立本版原生调用关联。

## 复制与清零活动

持续采集除kernel外还保留`MEMCPY`、`MEMCPY2`（peer复制）和`MEMSET`。原始记录增加`bytes`，复制还保留`copy_kind/src_kind/dst_kind`，peer记录保留来源/目标设备与上下文，清零保留内存类型和写入值。字段依据[CUPTI memcpy记录](https://docs.nvidia.com/cupti/13.3.0/api/structCUpti__ActivityMemcpy6.html)与[Python绑定](https://docs.nvidia.com/cupti-python/api-reference/topics/bindings.html)。

来源摘要、原生诊断时间表使用`memory_activity`；计数报告将对应数据明确命名为`profile_memory_activity`，表示它来自独立来源采集，不是ncu进程自身的memory观察。按操作类型保留次数与payload，区分逻辑设备读/写量。D2D复制n字节计逻辑读n、写n；memset计逻辑写n。Host↔device/array按方向计算设备侧逻辑读写；host↔host不产生设备侧逻辑读写。未知方向、managed或peer归属不明时，逻辑设备读写保持null，已知payload可以单独保留。peer流量不冒充单设备GPU实测，开发配额回退到确认的worker墙钟扣额，实测GPU值保持未知。

字节数缺失/非法、活动只部分落在调用窗口、捕获不完整或丢记录时，相应总量保持null。零仅用于完整记录中确实没有该类操作；窗口截断不按时间比例猜测字节量。非法原始byte字段还会使当前采集记录错误。所有值从原始活动重算，修改汇总无法继承原资格。

逻辑payload不是DRAM/cache硬件计数，也不包含kernel内访存、链路协议开销或managed页迁移。当前A/E模型继续拒绝含非kernel工作而没有专用传输校准的报告，不能把`2*n`直接加入DRAM计数后宣称完整执行SOL。该边界仍归R-16，未运行设备或校准实验。

## 指标与可信审核

规则是显式聚合计数器、单位和正换算系数，不执行Python或正则指标表达式。例如下面只记录DRAM流量，不能声称完整工作量：

```json
{
  "protocol": "kernelbench_pro_counter_recipe_v1",
  "coverage": "reported_resources_only",
  "review": {"reviewer": "maintainer", "rationale": "Partial memory report"},
  "resources": {
    "memory_bytes": {
      "rate_unit": "bytes/s",
      "terms": [
        {"metric": "dram__bytes_read.sum", "unit": "byte", "coefficient": 1},
        {"metric": "dram__bytes_write.sum", "unit": "byte", "coefficient": 1}
      ]
    }
  }
}
```

每launch必须有全部请求指标、相同单位及精确非负uint64计数；缺项、N/A、缩放、分数、实例列表、重复值或其他进程/设备数据均不能成为完整报告。浮点/Tensor指令类别与系数需逐架构、dtype和指令审核；不能用一个FMA项或引擎busy比例代表所有运算。

只有 `reviewed_complete_workload` 才能建执行模型。审核保留来源列表、硬件/软件类别和 `workload_contracts` 的来源/参考/输入签名/精度政策/precision；另须用 `review.native_contract_sha256` 绑定原生配置、观察角色和基线变体。预览会打印该摘要；它排除每次的新计时seed与采集参数，实际输入值和完整调用序列由各自收据绑定，不能借用其他基线/模式的审核。审核属于可信控制端输入，不是候选声明或签名认证。

`--calibration FILE` 在采集后重算每个调用的资源需求/当前设备实测速率；设备、单位或审核不符时保留完整采集，模型标为未测。CUDA memcpy/memset活动没有对应的完整kernel计数，本版拒绝将其自动补零后建立完整执行模型。不能将校准达到的速率称为物理上限。模型协议为 `kernelbench_pro_native_execution_models_v1`，每调用保留自身的SOL与组件，校准仅存一次。

## 离线原生 A/E/O

对既有计数器报告，不启动任何工具或设备：

```bash
python -m scripts.analyze_device_trace native /path/to/counters.json \
  --calibration /path/to/calibration.json --certificate /path/to/bound.json \
  --output /path/to/new-decomposition.json
```

算法证书复用[Headroom](HEADROOM_SCORING.md)的审核规则：当前参考SHA、精度、实际输入签名与同设备校准须匹配；多种几何可给证书列表，每次恰好匹配一项，缺失或重复不选择性跳过。执行SOL来自当前调用的原始计数，算法SOL来自该调用的审核证书，busy来自原生来源窗口的活动并集；O仅用同一CUPTI时钟的窗口长度/busy，原生对照时间与ncu重放时间分别保存，不混用。

每次及每个阶段按 `A=SOL_exec/SOL_alg`、`E=busy/SOL_exec`、`AE=busy/SOL_alg`、`O=window/busy` 报告，阶段使用各自总量之比。没有请求完整模型时A/E保持null；缺活动仍为未知。模型顺序与测量不一致会保留 `model_consistent=false`，不冒充错误答案。模拟输入永远保留simulated且不能成为hardware-counter证据；当前仅有CPU状态模型、模拟采集边界、vendor格式、重封篡改及软件集成回归。

缺工具、工具替换、超时、输出/模式/窗口/launch错配均保存not_measured和有界失败工件。CSV及worker单文件上限64MiB，原始行数上限200000；失败日志保留各自前64KiB、长度、摘要及完整性，截断不是完整证据。来源完整性、未覆盖独立诊断/后端、正式L1/L2门和设备资源硬限继续按原23项跟踪，绝不从诊断状态直接授予资格。

## 关联实际计分时间

```bash
python -m scripts.analyze_device_trace native /path/to/counters.json \
  --scored-result /path/to/eval_results.json --problem 2 --sample 0 \
  --calibration /path/to/calibration.json --certificate /path/to/bound.json \
  --output /path/to/new-scored-decomposition.json
```

配对工件使用`--task paired_timing`。该入口只读既有文件，不查询或执行设备、工具和模型；要求完整计数器报告、原始L1/L2实际结果、同物理设备校准和覆盖全部几何的算法证书。

实际计分与来源重放逐调用核对完整输入/状态/输出/模式及物理设备，随后比较“已关联二进制SHA、解析入口名、实际grid/block、静态/动态共享内存”的多重集合，保留并发重复次数而不假定活动完成顺序相同。入口名只有在函数/模块/launch关联和活动名称一致时才进入身份；名字本身不能建立来源。缺字段、不同内核/配置/次数或非kernel活动不能借用kernel计数模型。采集字段依据[NVIDIA CUPTI Python接口](https://docs.nvidia.com/cupti-python/api-reference/topics/bindings.html)。

每个调用的busy来自实际计分CUPTI活动并集，墙钟来自对应原始timer，A/E/O不借用另一个进程的时钟。没有原始timer的正确性/显存调用保留`scored_wall_ms=null`、`O=null`；模型与实际时间不一致会显式标记。模拟结果不能成为正式证据，CLI返回非零。

原生v2在同一次ncu采集中额外请求`launch__kernel_name`、`launch__grid_dim_[xyz]`、`launch__block_dim_[xyz]`和`launch__shared_mem_per_block_{static,dynamic}`。原始kernel name独立于CSV中用于关联调用的NVTX标签；输出要求mangled名称，不猜测名称别名。launch指标的定义及不增加额外replay pass的性质见[NVIDIA指标说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#launch-metrics)。

解析要求每个action都有唯一、完整的launch属性，维度是正整数，共享内存是非负整数，单位明确且没有缩放；缺项、重复或异常值不能借独立来源采集补齐。工作计数与配置分开保存，逐调用关联时将ncu原始名称/grid/block/shared的多重集合与实际计分来源记录比较，配置或重复次数变化均拒绝关联。它复用同一份CSV，没有新进程、CUPTI订阅或独立采集。v1原生报告缺少这些字段，需重新采集；旧通用CSV读取API仍可用于历史诊断。

同一次ncu命令还通过`--export`保存供应商自己的二进制报告。`raw_report`只保留一份base64原始字节、大小与SHA-256，单文件上限64 MiB。现有计数器模块按[NVIDIA version-7文件格式](https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html#report-file-format)读取未压缩`NVR\0`报告，从`SourceData.Binary`提取原始ELF cubin，解析`ProfileResult`的原始符号、grid/block、context、stream及进程所属，并解析源码引用。字段依据本机SDK随附的`ProfilerReport.proto`与`ProfilerReportCommon.proto`核对；没有引入供应商生成代码、protobuf运行依赖或新运行模块。

二进制报告与CSV必须具有相同的进程、context/stream、原始符号、grid/block组及完整launch数量。一个组若出现不同cubin就保留未知，不猜CSV action ID与供应商内部KernelID的关系。逐调用`kernel_identities`保存已关联cubin哈希、字节数和CSV共享内存配置，再与实际计分二进制/入口/完整配置多重集合比较；即使重封全部汇总，替换cubin仍会失配。缺少、过大、截断、未解析引用、压缩或其他未支持格式保留`binary_identity_issue`与空身份，仍可读取工作计数，不能借另一进程补二进制。

成功关联记录`counter_process_launch_identity_observed=true`；仅当全部调用的二进制也核对一致时，`counter_process_binary_identity_observed`及`counter_process_workload_identity_observed`才为true。模拟收据保留模拟标记。`formal_qualification_implemented=false`继续保留：这批没有完成非kernel传输需求/校准、Neuron来源与原生时间关联及正式L2组装门，不能由二进制关联自动授予L2资格。

采集没有增加CUPTI订阅或独立采集进程。当前固定的CUPTI Python 13.3将`allow_multiple_subscribers`固定为0，见[绑定限制](https://docs.nvidia.com/cupti-python/13.3.0/api-reference/topics/bindings.html#cupti.cupti._SubscriberParams.allow_multiple_subscribers)；这里使用ncu自身报告的字节证据，不叠加Python采集器。软件回归另用SDK `.proto`字段定义和官方protobuf实现核对生成的报告消息；没有执行ncu或设备实验。
