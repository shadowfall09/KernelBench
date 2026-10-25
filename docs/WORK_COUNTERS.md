# 原生调用计数器与审核后的 A/E 分解

计数器采集复用原生评测序列，不再另造一个初始实例测一次前向。默认仅预览；没有运行 ncu、CUPTI、GPU、Neuron、模型、容器或校准实验。所有插桩时间属于诊断，不写回正式分数。

## 入口

```bash
python -m scripts.audit_work_counters --config /path/to/device.toml \
  --recipe /path/to/recipe.json --problem 2 --sample 0 \
  --output /path/to/new-counters.json
```

预览不导入设备SDK、不查工具、不创建输出。仅在实验已授权时增加 `--execute`。执行使用同一总超时：先冻结工具路径、字节与版本，再运行原生对照和CUPTI来源采集，最后在新进程中采集完整序列的计数器；每个阶段均受父进程剩余预算约束。复用local、Docker、Modal、HTTP队列，CUDA工具缺失或Neuron请求不能降级成通过。

默认目标为候选的原生 `evaluate`，覆盖双顺序正确性、所有预热和正式计时调用。配对路径使用 `--task paired_timing --baseline-variant eager`，变体须填写实际选中的来源绑定基线；加 `--target reference` 测该真实配对基线，不再用另造的公开参考实例代替。原生初始化、连续状态、输入角色/精度、数值门和实际调用顺序保持原样。旧的单次 `--seed` 路径和 `counter_worker.py` 已移除；原生评测seed来自配置，计时输入计划在候选前冻结。

## 逐次采集与重放

采集进程在候选导入前捕获profiler start/stop和NVTX push/pop函数。每次被观察的调用使用父进程新nonce和唯一ID标记，独立启停profiler；前向异常也关闭采集并弹出范围，栈深须平衡。NVTX仅关联调用，不作为源码归属证据。

ncu禁用用户配置和额外kernel重命名，显式关闭cache flush与时钟控制，采用kernel replay、base units、完整子进程范围和原始逐launch CSV，开启NVTX并将最近push/pop范围打印为标签。不按NVTX过滤或只取前若干kernel。解析须覆盖每项请求指标、实际PID和device；标签必须唯一对应可信窗口，且每个调用的launch数与独立来源采集中的kernel数相等。嵌套NVTX覆盖标签、无范围/后台工作、缺launch或重复身份无法匹配时明确失败，不凭名称猜测或把未知记录丢掉。[NVIDIA CLI说明](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)记录了这些选项；push/pop深度按当前Torch绑定核对。kernel replay仍会改变并发和缓存历史，不保证与未插桩的工作量相同；这些SOL和分解保留为审核后的诊断模型。真实部署的vendor输出仍未实验验证。

原生对照、来源采集和ncu三阶段必须具有相同的完整调用序列：完整逻辑输入/输出字节、布局、模式、seed、顺序、当前源码、精度和设备绑定逐项核对，所有输出仍过原来的数值检查。原生计时样本和窗口由[来源重放](KERNEL_ORIGIN.md)再次检查。设备内存/其他诊断探针仍不在此序列内；跨进程可观察一致不证明未插桩路径或任意参数状态等价。

报告只保留一份原始CSV、源码采集和计数器进程记录，按调用重算计数，不复制整份CSV到每个调用。协议为 `kernelbench_pro_native_work_counters_v1`；`counter_task.validate()`重放全部关联及工作量，不信任保存的passed标记或重封的汇总。旧 `kernelbench_pro_work_counters_v1` 的原始CSV读取和离线模型API仍可用于历史诊断，但不能建立本版原生调用关联。

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

缺工具、工具替换、超时、输出/模式/窗口/launch错配均保存not_measured和有界失败工件。CSV及worker单文件上限64MiB，原始行数上限200000；失败日志保留各自前64KiB、长度、摘要及完整性，截断不是完整证据。来源完整性、未覆盖探针/后端、正式L1/L2门和设备资源硬限继续按原23项跟踪，绝不从诊断状态直接授予资格。
