# Neuron逐指令工作与离线Parquet分析

离线读取器只读取已有Parquet，不启动工具或设备；新增处理入口默认预览，显式执行也只让Explorer处理已有文件，不启动采集或工作负载。本轮仅做真实Parquet文件读写、CPU小模型和SDK/runner模拟，没有运行Explorer、Trainium、GPU、真实模型或采集实验。逐指令工作注释可构建审核后的资源需求模型，但不建立内核执行所有权或正式评分资格。

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

指令提供显式`execution_index`时，必须指向已知execution且时间区间完全包含该指令，不能用另一execution替代。旧行未提供该字段时，使用ExecutionInfo的完整时间区间，只有恰好一个包含区间才分配；缺失、重叠或含糊保留未分配。NKI调用点必须同时匹配`execution_index`与`nki_call_site_id`，并核对NEFF名称；相同site编号在不同execution不能互借。kernel/NEFF名称只是标签，所有来源仍为`unknown`。

`neuron_tables.validate()`从报告保留的原始投影表重算工作、busy与关联，复查文件/分片和投影收据，拒绝改汇总后重新封装的报告。这是离线一致性校验，不是设备采集的密码学认证。Metadata中的`capture_type=hardware`只记录供应商声明，不会使`hardware_counters_collected`或`formal_origin_qualified`变真，也不替代正式墙钟样本。

## 审核后的资源需求模型

Neuron worker的profile返回`capture_receipt`。它在冻结编译源码再次核对后生成，保留候选/参考源码、配置、输入几何、输入及注册状态探针、此前正确性用例、实际设备/软件、模式、seed、比较政策和原生工件哈希/字节数，并记录原生worker的可见core清单。探针只覆盖每Tensor最多256点，不能代表完整内容或普通Python状态哈希。输入隔离复制在采集区间前完成；采集中只有候选前向及完成屏障。CPU/注入适配器收据保留模拟标记，不能建真实设备模型。

模型需要可信控制端维护者提供`kernelbench_pro_neuron_work_review_v1`审核。`coverage`必须是`reviewed_complete_workload`；保留非空`reviewer`与`rationale`，不能由候选自行声明。审核同时包含：

- `workload_contract`：与采集收据的整个`binding`一致，包括源码、参考、配置、输入、状态、精度和政策。
- `numeric_class_sha256`：当前采集设备的软件/硬件类别；校准还须匹配同一物理UUID、core拓扑及软件版本。
- `pairing`：精确固定`capture_receipt_sha256`、`report_receipt_sha256`、`physical_device_sha256`及报告的`neuron_device_idx`/`neuroncore_idx`。
- `memory_accounting`：明确为`instruction_hbm_annotations_only; DMA_packets_not_added`。维护者须审查所用注释是否覆盖该工作负载，不能把未覆盖传输称为完整需求。
- `rules`：逐条给出精确的`engine`、`opcode`、`dtypes`（三个键`src1_dtype/src2_dtype/dst_dtype`）和`arithmetic_resource`。每条指令恰好匹配一条规则；资源只允许`fp32_flops/fp16_flops/bf16_flops`，或仅在原始FLOPs为零时用null。

dtype标签按报告原值填写，不猜拼写或用输出存储dtype推导全部算术。每条指令的`raw_flops/hbm_read_bytes/hbm_write_bytes`必须是已报告的非负精确整数；null不能由审核补零。正FLOPs还要求三个dtype都有明确值。只用原始FLOPs，BF16等价的adjusted计数不参与加总；DMA包继续保留，不再叠加。所有指令须关联execution，所有执行的NEFF名称须对应采集工件；硬件声明、显式空Error和Warning表也都是前置条件。

未提供处理收据时，以上report/capture配对仍是**维护者人工核对的关联**。名称相同或hash配对不是自动NTFF处理证据。使用下一节的处理链时，`pairing`改为仅包含`processing_receipt_sha256`，由处理收据核对输入和报告；资源规则及完整工作量仍须可信维护者审核。两条路径都不证明源码执行所有权或密码学认证。

对已有文件增加三个选项即可离线构建：

```bash
python -m scripts.analyze_neuron_work ./profile_parquet --output ./new-neuron-model.json \
  --capture ./profile-result.json --review ./reviewed-neuron-work.json \
  --calibration ./same-device-calibration.json
```

命令不会采集、校准或调用Explorer。成功结果包含全部原表、采集、审核、校准原始样本、逐引擎/资源需求及SOL分量；模型失格时退出2，仍在新输出保存完整原始报告及`not_measured`原因。单独省略全部三个选项仍输出原来的工作报告。

`neuron_work.execution_model()`将正资源需求除以同设备实测速率，取各分量最大值；`validate_model()`从完整证据重算。`performance.decomposition(...,counters=model)`可据此拆A/E，但标注`work_evidence_kind=reviewed_neuron_instruction_annotations`、`hardware_counters_collected=false`，不能称为硬件计数器。缺证据仍只给AE/O；分解也不替代原生计分时间或把实测率当物理定理。

## 从原始工件自动关联处理结果

已有真实原生profile结果和工件时，先预览：

```bash
python -m scripts.process_neuron_work --capture ./profile-result.json \
  --artifacts /path/to/native-profile --output ./new-processing.json
```

预览读取收据并检查声明，不查找/运行工具、不读取整个工件目录、不创建路径。只有明确需要处理时才增加`--execute`；本轮没有执行。可用`--binary`选择已安装Explorer，默认`neuron-explorer`。该入口仅执行固定的`view --session-dir INPUT --output-format parquet --output-file OUTPUT`，不提供capture、inspect、用户附加参数或过滤选项。

自动路径要求收据有完整字节清单，且可见core清单只有已核对的那个物理core。它不猜供应商`neuroncore_idx`与全局/局部编号的换算；设备关联范围是这次单core原生采集提供的全部工件。原始工件目录与工作目录必须分离，不能借用或覆写已有工作目录。

执行先按收据逐文件核对完整树的哈希、大小和NEFF/NTFF，拒绝额外/缺失文件、软链接、特殊文件与超限；再复制到独立`input/`，复制时逐字节验证摘要，文件设为只读。工具路径解析为绝对路径，版本查询前冻结二进制SHA，查询后和处理后再核对。完成表读取后还会复查工具、原始输入、快照及全部输出，晚到的修改也不能生成收据。

默认工作目录是输出同级的`<output-stem>-processing/`，可用`--work-dir`指定新目录。原始快照、全部Parquet（包括未投影的Summary等表）、版本日志与处理日志均保留。处理收据保存完整输入/输出文件描述、选中表投影、命令、工具身份、进程退出/超时状态、评测器与capture/report摘要。失败退出2，保存`not_measured`原因及每日志最多64 KiB的可校验前缀；已有部分Parquet仍留在工作目录，不变成部分通过。

默认原始/输出工件各最多4096个文件、每文件128 MiB、合计512 MiB，可用`--max-artifact-file-mb`和`--max-artifact-mb`调整；目录项另有限制。表读取仍默认200,000行、单表文件64 MiB和解码估计256 MiB，对应`--max-rows`、`--max-table-file-mb`、`--max-decoded-mb`。`--timeout`默认120秒，涵盖步骤间剩余额度检查和子进程超时；文件/估计门不是进程RSS、实时调度或文件写入时的磁盘硬配额。

处理成功后，在前面的模型命令中增加：

```bash
--processing ./new-processing.json
```

同时把可信审核的`pairing`设为该处理收据的`processing_receipt_sha256`。模型重放处理清单、工具和命令、capture/report及逐指令需求。指定的处理证据缺失、无效、模拟、超时或被改写时，拒绝自动绑定并保留原始报告，**不会退回人工配对**。模型记录`automatic_capture_report_binding=true`，仍保持`formal_origin_qualified=false`及`hardware_counters_collected=false`。

这里自动关联的是受信控制进程所观察的文件处理。它没有隔离供应商工具的所有外部依赖，也不是对候选或设备驱动的密码学认证；更不证明与未插桩原生计时路径、全部状态或隐藏用例等价。R-16继续保持部分实现：与原生时间/执行来源的完整关联及正式评分门尚未完成。

CUDA现有模型见[工作计数器](WORK_COUNTERS.md)。实现及回归分别位于[表读取器](../kb_workflows/neuron_tables.py)、[审核模型](../kb_workflows/neuron_work.py)、[读取命令](../scripts/analyze_neuron_work.py)、[自动处理器](../kb_workflows/neuron_processing.py)、[处理命令](../scripts/process_neuron_work.py)、[表测试](../tests/test_neuron_tables.py)、[模型测试](../tests/test_neuron_work.py)和[处理链测试](../tests/test_neuron_processing.py)。
