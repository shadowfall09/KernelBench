# 候选分配预填充检查

`allocation_validation`默认关闭，可通过`--allocation-validation`或TOML启用：

```toml
[run]
allocation_validation = true
timeout = 1800
```

当前仅接受CUDA后端。实现固定Torch 2.12.0的ATen Python dispatch接口；已做CPU/模拟回归，没有执行GPU实验。它检查可观察分配下的正确性，不提供完整GPU内存、安全或自写内核来源证明。

## 两种填充值

每次启用后，父进程在正常原生评测之外，分别启动两个新的正确性进程。候选的模块导入、构造/设备移动和每个实际双向比较中的forward，都在预填充作用域中；参考调用和输入工厂不做预填充。

| 存储dtype | `nan_max` | `finite_min` |
| --- | --- | --- |
| 实数浮点 | NaN | −0.5 |
| 复数 | 实部、虚部均NaN | −0.5 + 0.5j |
| 整数 | 本dtype最大值 | 本dtype最小值 |
| bool | True | False |

候选仍须通过原生结构、shape、dtype、非有限传播及数值比较。第二种值可揭露“参考本来全NaN/整数最大值/bool True，未写入输出却碰巧相同”的情况。预填充不消耗RNG，不强制开启确定性算法，不改变容差。连续`resize_`/`resize_as_`只填新增逻辑尾部，保留已有前缀；非连续增长无法安全处理时保留`not_measured`，不破坏原数据来模拟覆盖。

基本原理依据Torch的[未初始化empty语义](https://docs.pytorch.org/docs/2.14/generated/torch.empty.html)和[resize保留原存储的语义](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.resize_.html)。其内置确定性预填需要同时开启确定性算法，见[官方确定性说明](https://docs.pytorch.org/docs/main/utils/deterministic.html)；本实现另用作用域观察，避免为这项诊断强制选择另一算法。链接为当前在线文档，实际实现与单测使用2.12.0安装源码。

## 独立进程与计时

父进程在两个子进程启动前配置`PYTORCH_NO_CUDA_MEMORY_CACHING=1`、`CUDA_LAUNCH_BLOCKING=1`、`PYTORCH_ALLOC_CONF=backend:native`，并去掉冲突的旧allocator配置。官方环境变量说明见[CUDA环境配置](https://docs.pytorch.org/docs/2.14/cuda_environment_variables.html)。审计记录入口前CUDA是否已初始化；过早初始化或环境不匹配不能取得本项资格。

子请求强制为`evaluate`且关闭性能采样。原有标准/隐藏范围和种子不变，两个模式分别创建候选，不能把原进程里已写好的缓存拿来复核。每例仍比较候选先行与参考先行。最后另有一次不预填的直接正确性复核。原生计时样本、冷启动成本、baseline和配对时间保持来自最初的原生进程；这几次诊断不替换评分时间，也不把填充kernel当作候选性能或来源。

所有阶段共用父请求的墙钟上限，各诊断阶段还受`sanitizer_timeout`约束。这里没有额外发放设备预算；超时保留未知并阻止当前请求取得已配置检查的资格。

## 收据与恢复

`metadata.tool_checks.allocation_controls`保留两个进程的请求哈希、退出状态、设备身份、实际原生覆盖和逐模式文档。`allocation_audit`逐作用域记录调用标签、seed/order、ATen分配类型、dtype/device、次数与逻辑元素数；meta/fake分配没有实测存储量，元素数保留null。

父进程和评分/恢复重新核对模式、源码/请求/设备、导入与构造覆盖、每个实际调用及后续未预填复核。不接受删模式、改seed、丢调用、错误设备、模式被移除、不可处理分配或超时。候选在预填条件下的原生比较失败会阻止资格；没有完整证据标为`not_measured`。模拟过程不能用作设备资格。

开关属于统一诊断政策，进入最终冻结、研究/RL合同、运行分组和Harbor私有配置。更改开关或评测器不能复用旧正结果。实现见[分配观察](../kb_workflows/evaluation/allocation_audit.py)、[进程监督](../kb_workflows/evaluation/tool_runner.py)、[收据复核](../kb_workflows/evaluation/tool_checks.py)和[CPU/模拟测试](../tests/evaluation/test_allocation_audit.py)。

## 观察边界

当前作用域只覆盖当前线程可观察到的ATen逻辑分配与安全的连续扩容。原始CUDA/自定义allocator、其他线程、不可观察的扩展内部、已捕获Graph缓存及某些编译路径仍可能绕过观察；零分配记录只说明该作用域未观察到分配，不能证明内核执行或排除旧显存读取。

量化/非标准Tensor和无法填充的存储不猜测通过。meta/fake记录不代表设备写入。编译器可能对dispatch mode有额外限制；不自动禁用编译或退回另一实现来取得通过。观察模式不是安全沙箱，源码/二进制归因、Sanitizer和独立数值检查继续各自判断。现已另有[吞吐对照](THROUGHPUT_VALIDATION.md)，完整GPU执行证据仍未完成，R-09保持部分实现。
