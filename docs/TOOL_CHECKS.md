# 内存工具与重复输出检查

R-10提供可选的Compute Sanitizer流程和题目级重复一致性策略。默认关闭；本版只完成代码及CPU/模拟回归，没有运行Sanitizer、GPU或模型实验。

## 配置

```toml
[run]
sanitizer_tools = "all"
sanitizer_command = "compute-sanitizer"
sanitizer_timeout = 300
determinism_checks = true
determinism_repeats = 5
# state_validation = true  # 可独立开启的模式/注册状态/参数更新检查
# numerical_validation = true  # L1/L2原生实际输入的独立CPU FP64复核
# allocation_validation = true  # CUDA候选分配的独立预填充控制
# oracle_memory_mb = 4096
timeout = 1800
# determinism_policy_file = "reviewed-policies.json"
```

`sanitizer_tools`接受`none`、`all`或逗号分隔的`memcheck,racecheck,synccheck,initcheck`子集。命令是单个可执行文件路径或PATH名称，不经过shell。`sanitizer_timeout`是每个诊断阶段的上限；原生计时、工具、版本查询和复核共用一次请求的`timeout`，预留5秒清理，没有额外发放设备预算。工具较慢时需显式调整原请求超时。

预览示例：

```bash
python scripts/run_batch.py --config configs/a6000-safety.toml --dry-run
```

预览不会调用工具或设备。真正评测需要在验证环境中另外提供兼容的NVIDIA Compute Sanitizer。缺工具是`not_measured`，不会自动安装、跳过后算通过，或退回CPU计时。CUDA镜像的具体工具安装情况须由部署者确认。Neuron拒绝CUDA工具选项；重复输出检查可独立启用，无法快照模型或后端RNG时不能得到通过证据。

## 执行顺序与计时

1. 独立原生worker运行原来的正确性和性能协议；已有失败直接返回。
2. 每个选定工具分别启动新的正确性worker，沿用冻结源码、标准/隐藏范围和种子，关闭性能采样。
3. 若开启分配检查，两个独立子进程分别使用NaN/极值与另一组确定值，覆盖候选导入、构造和实际正确性调用；不用这些进程采样性能。
4. 全部选定工具通过后，再进行一次无插桩正确性复核。开启重复、状态、数值或分配检查时，即使工具失败也保留这次复核诊断。
5. 结果保留第一步的runtime、原始样本、冷启动成本和配对记录。工具和复核耗时只进入请求墙钟与资源核算。

所有transport使用同一个CLI监督进程，因此本地、Docker、Modal、HTTP队列和Harbor验证复用这条路径；部署与实机可用性未验证。配对计时的工具子任务使用独立标准正确性流程，不对插桩后的速度做配对推断。标准与隐藏结果分别绑定自己的请求；隐藏日志不会反馈给开发agent。

`state_validation`在这次独立复核中为每个原生实例增加连续模式、注册状态映射及参数更新探针，与重复检查可以同时开启；接口和覆盖边界见[状态审计](STATE_VALIDATION.md)。它不要求安装Sanitizer，也不受17题FP64附加契约清单限制。缺少完整实例、绑定、模式或更新记录时拒绝资格；不改变原生采样时间。

`numerical_validation`在同一次复核中为L1/L2的每个实际输入增加CPU独立算式的FP64门与参考信号诊断，保留原shape和值，并在分配前核对主机内存估计包络。参考也须通过；原尺寸超包络不能缩小后算通过。范围和参考问题见[数值复核](NUMERICAL_VALIDATION.md)。

`allocation_validation`另用两个新进程进行候选分配预填充，启动前关闭CUDA缓存allocator并要求同步报错；不切换确定性算法，参考分配不被填充。当前线程ATen观察的范围、连续扩容保护、量化/Graph/原始分配限制和完整收据见[分配检查](ALLOCATION_VALIDATION.md)。它不替代四工具的memory_safety证据。

工具跟踪整个worker及子进程，要求CUDA初始化，设置非零错误退出码，逐进程保留日志。memcheck启用32字节padding，并在Torch导入前禁用PyTorch CUDA内存缓存，增强对逻辑分配越界的可见性。racecheck保留warning及error，不强制阻塞launch，不设置suppression或跳过kernel的过滤器。监督进程清理外部`NV_COMPUTE_SANITIZER_*`及CUDA injection路径覆盖。

通过要求：原始日志哈希、已识别的完成摘要、零错误/警告、成功且未超时的进程、冻结候选的正调用覆盖，以及匹配的源码与物理设备/驱动身份。日志缺失、未完成、超限、符号链接或采集失败均不能通过。工具确实报告的错误单独标为`failed`；基础设施缺证据为`not_measured`。

日志覆盖包括参考和框架执行，错误不能自动归因到候选源码。racecheck主要检查共享内存访问危险，不证明所有全局内存竞态都不存在；通过有限输入上的四工具流程也不等于形式化内存安全证明。调用覆盖不证明自写kernel做了有效工作，该证据仍归R-08。

## 重复一致性与原子操作

开启后，每个原生双向正确性实例在独立模型快照上重复3–100次，默认5次。每次重新复制输入和模型状态，重放种子，消费并同步输出，同时保留原有正确性检查；最终恢复调用前RNG状态，原模型不会因这些额外调用而前进。

默认`bitwise`策略要求输出位模式、结构、shape、dtype一致，包括浮点有符号零。整数/bool始终精确。重复不一致表示违反声明的确定性契约，不能据此断言存在竞态；重复一致同样不能证明无竞态。

合法使用非确定性原子规约的题目，可提交审核后的策略文件：

```json
{
  "protocol": "kernelbench_pro_determinism_policies_v1",
  "tasks": {
    "1/2": {
      "reference_sha256": "替换为参考文件UTF-8原文的SHA-256",
      "mode": "bounded_atomics",
      "atomics_allowed": true,
      "atol": 0.000001,
      "rtol": 0.000001,
      "review": {
        "reviewer": "填写审核者",
        "rationale": "填写该任务允许原子规约及数值波动的理由"
      }
    }
  }
}
```

这只是格式示意，没有为矩阵乘法或其他题目授予默认豁免。实际文件必须有64位十六进制摘要、有限非负且不全为零的界限和审核说明。未列题目仍使用逐位策略；源码改变必须重新审核。浮点重复差异需满足声明的`atol + rtol * abs(首次输出)`，非有限值位置和Inf符号仍须一致；该策略不放宽每次输出对参考的原生精度门、离散值或严格FP64门。

收据记录每个原生实例的label、seed、输入结构、快照策略、参考/候选实际调用数、输出位摘要及失败原因。遗漏实例、更换种子、未完成全部重复不能通过。它依赖可信worker的实际比较记录，不是独立可验证的数值证明或OS来源认证。

## 评分与恢复

`metadata.tool_checks`保存完整原始结果、工具二进制/版本摘要、每个日志全文和摘要、阶段请求与nonce、设备身份、无插桩复核、重复检查及派生判定。模拟执行显式标注，正式评分拒绝。当前评分和断点恢复会重算这些判定；仅修改汇总布尔值或外层哈希不能弥补日志/覆盖缺项。已记录的匹配失败可以恢复，避免重复花费；被修改或丢失的收据需重跑。

显式选择的每个工具和重复检查都是正确性附加门。只选一个工具可以满足该选择，但`memory_safety`汇总仍为未知；只有四工具全通过才提供该项证据。L2还要求`repeat_consistency`以及此前的L1、计数器与算法证书；不会因本功能已实现就自动获得L2资格。隔离认证始终保留`false`，容器权限和重评分属于R-23。

策略文件内容进入生成、最终冻结、研究比较、RL验证器输入和旧评分恢复指纹。开发任务只带本题公开数值策略；Harbor把审核文件放进独立verifier，其他题目的策略不导出给agent。

工具选项和限制依据[NVIDIA Compute Sanitizer文档](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。独立进程、原始日志和直接复核的流程也参考[NVIDIA compute-eval的Sanitizer实现](https://github.com/NVIDIA/compute-eval/blob/main/compute_eval/evaluation/sanitizer.py)；仓库没有直接引入其代码。
