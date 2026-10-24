# Neuron逐指令工作与离线Parquet分析

本路径读取已有的Neuron Explorer处理结果，不启动Explorer、SDK或设备。本轮仅做真实Parquet文件读写、CPU小模型和SDK模拟测试，没有运行Trainium、GPU、真实模型或采集实验。它记录逐指令工作注释与DMA包事实；尚不建立执行来源、完整工作模型或正式评分资格。

## 输入与命令

读取固定版本[Neuron SDK 2.33.0的schema 12.0.1](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.33.0/tools/neuron-explorer/profile-schema-reference.html)，不猜其他版本。官方[完整导出说明](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.33.0/tools/neuron-explorer/how-to-analyze-profile-output.html)支持：

```bash
neuron-explorer view --session-dir ./profile_output --output-format parquet --output-file ./profile_parquet
```

这是供已有授权实验使用的供应商命令，本轮没有执行。UI下载可能采样或截断，不能代替完整表；SDK 2.33也是[最后支持旧JSON的版本](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.33.0/about-neuron/announcements/neuron2.x/announce-intent-eos-explorer-json.html)。旧JSON时间线分析仍是独立入口，不能直接套新schema。

在装有`analysis`可选依赖的Python 3.12环境中，对一个已处理的profile目录运行：

```bash
python -m scripts.analyze_neuron_work ./profile_parquet --output ./new-neuron-work.json
```

命令只处理本地文件并写新报告，已有输出会拒绝。必需表是`Metadata`、`ExecutionInfo`、`Instruction`、`DmaPacket`；可选读取`NkiKernelCallSites`、`Warning`和`Error`。每表支持一个完整文件或从0连续编号的分片，不能混用、缺片、重复或用软链接。所有分片先完成准入，再分批读全部行，只投影审核过的字段；不读取`Summary`或DMA聚合表来补工作量。

默认上限为全部选中表200,000行、每文件64 MiB、全部文件及Parquet footer/row-group解码估计256 MiB、4096个表文件。可用`--max-rows`、`--max-file-mb`和`--max-decoded-mb`调整；估计不是进程RSS硬限制。超限拒绝整个报告，不丢弃分片或抽样。报告保留每文件字节数、SHA-256、行数和字段投影，读取前后复查文件与分片清单。

## 工作量与时间口径

只接受明确capture类型和LNC=1。指令及DMA身份、时间戳、字节和FLOPs保留精确整数；超过浮点精确范围的值也不经过float。每行验证结束时间与时长，busy时间取区间并集，不累加重叠。零时长事件仍保留工作事实，不能贡献busy时间。

`raw_flops`和BF16等价的`adjusted_flops`分别保存，不能混用；dtype注释保留在原表。供应商说明FP32_LO_HI每条记录半个逻辑matmul的FLOPs，分析器直接保留逐指令值，不再重复乘二。每引擎的FLOPs/HBM读写分别报告已知行之和、缺失行数和完整性，null不是零。

DMA包的read/write/transfer字节另列。它们包含不同存储端点和host传输，不能全部重命名HBM流量，也不能与指令HBM注释或DMA聚合表相加。未出现的Error/Warning表与已出现的空表区别保留。

## 执行关联与证据边界

使用ExecutionInfo的完整时间区间关联指令，只有恰好一个包含区间才分配execution；缺失、重叠或含糊时保留未分配。NKI调用点必须同时匹配`execution_index`与`nki_call_site_id`，并核对NEFF名称；相同site编号在不同execution不能互借。kernel/NEFF名称只是标签，所有来源仍为`unknown`。

`neuron_tables.validate()`从报告保留的原始投影表重算工作、busy与关联，复查文件/分片和投影收据，拒绝改汇总后重新封装的报告。这是离线一致性校验，不是设备采集的密码学认证。Metadata中的`capture_type=hardware`只记录供应商声明，不会使`hardware_counters_collected`或`formal_origin_qualified`变真，也不替代正式墙钟样本。

当前尚未把这些注释接成审核后的Neuron执行工作模型、同物理设备校准A/E或正式来源/评分门；R-16继续保持部分实现。CUDA现有模型见[工作计数器](WORK_COUNTERS.md)。本实现及回归分别位于[表读取器](../kb_workflows/neuron_tables.py)、[命令](../scripts/analyze_neuron_work.py)和[测试](../tests/test_neuron_tables.py)。
