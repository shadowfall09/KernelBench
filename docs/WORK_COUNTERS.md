# 工作计数器与审核后的 A/E 分解

本路径提供 Nsight Compute 采集入口、原始 CSV 重放和审核工作模型的软件实现。默认只预览；本次没有运行 ncu、GPU、Trainium、模型、容器或校准实验。计数器工件属于独立诊断，不建立完整来源/原生用例覆盖，也不修改正式计分时间。

## 入口与范围

对已有候选先预览：

```bash
python -m scripts.audit_work_counters --config /path/to/device.toml \
  --recipe /path/to/counter-recipe.json --problem 2 --sample 0 \
  --output /path/to/new-counter-report.json
```

预览不导入设备/模型 SDK、不查询工具、不创建输出目录。只有明确要运行采集时才增加 `--execute`；本次仅用模拟executor测试执行分支，没有实际采集。可用 `--target reference` 对参考运行同一采集路径，默认目标为候选。输入 seed 默认 42、可配置；参数初始化和前向 seed 固定为 42。采集复用 local、Docker、Modal 或 HTTP 队列，worker 中需已有 ncu。Neuron 明确拒绝这条 CUDA 路径。

worker 保留原构造和实际公开工厂尺寸，预热单独的初始实例，然后测另一个初始实例的一次前向。它在导入候选前固定输入、参考输出、容差和设备身份；可选名义配方仍经现有入口应用。启用 `numerical_validation` 时另作实际输入的独立 FP64 检查，超内存估计包络不缩小尺寸。Profiler start/stop 只包围目标前向和完成屏障；异常也执行 stop。模式或已观察的环境修改拒绝输出资格，可信编译目录在审计前设置并在退出后恢复。

这不是原生连续状态、全部标准/隐藏用例或参考信号资格的证明。输入与参考注册参数/buffer记录每Tensor最多256点探针，不是完整内容或普通未注册状态哈希。所有工件明确标记该范围；不能将它们直接升级为正式来源或高级证据等级。

## 指标规则

规则只声明要读取的聚合计数器及单位换算。例如下面只报告 DRAM 流量和一个 FP32 指令类别，**不具备完整工作量覆盖资格**：

```json
{
  "protocol": "kernelbench_pro_counter_recipe_v1",
  "coverage": "reported_resources_only",
  "review": {
    "reviewer": "counter mapping maintainer",
    "rationale": "Partial resource report; excludes other arithmetic and engines"
  },
  "resources": {
    "memory_bytes": {
      "rate_unit": "bytes/s",
      "terms": [
        {"metric": "dram__bytes_read.sum", "unit": "byte", "coefficient": 1},
        {"metric": "dram__bytes_write.sum", "unit": "byte", "coefficient": 1}
      ]
    },
    "fp32_flops": {
      "rate_unit": "operations/s",
      "terms": [
        {"metric": "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", "unit": "inst", "coefficient": 2}
      ]
    }
  }
}
```

不能用示例的 FMA 项代表所有浮点运算，也不能从 Tensor engine busy 时间猜 FLOPs。不同架构、dtype、指令形状和实现需要相应的审核映射；不支持的指标保持失败，不换算成零。规则允许明确的 byte/inst/op/sector 单位和正换算系数，不执行 Python，也不接受 regex 指标表达式。每个 launch 均须具备全部请求指标、相同单位及精确非负 uint64 聚合值；缺项、N/A、缩放单位、分数计数、重复值、实例列表或其他进程/设备数据不能建立完整报告。

## 采集与重放

父进程冻结 ncu 路径、文件哈希与版本，禁用用户配置文件，显式选择 kernel replay、base units、完整子进程范围和单次 profiler API 窗口。原始报告保留 launch/PID/context/stream/device 身份，kernel 名称只作为标签。kernel replay 可能序列化或重放设备工作，因此其时间不能替代未插桩时间，也不证明两条路径等价。[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)说明了这些选项。

父进程再次核对请求源码、参考、设置、输入签名、精度政策、seed 与实际设备身份；报告 PID 必须对应该次 worker，结束后工具字节也要一致。记录保存原 CSV、换算规则、逐 launch 原值与汇总、来源与设备、输出检查和独立程序收据。整数计数先精确解析，避免经 float 丢失大整数。记录进入当前评测器摘要；CPU/注入 runner 或适配器保留 simulated 标记，不能成为设备测量。

超时、工具/报告变更、输出失格、缺指标或解析失败保存 `not_measured` 及有界失败工件：每个日志/报告保留前64 KiB的base64、长度、摘要和是否完整，不把截断内容当完整证据。成功报告也有64 MiB及原始行数上限。该路径沿用现有任务超时和进程清理，不提供 GPU busy 时间的物理硬预算。

## 执行工作模型与 A/E

只有 `coverage = "reviewed_complete_workload"` 才能构建执行工作模型。审核还须固定 `source_sha256` 列表、硬件/软件类别 `numeric_class_sha256`，以及 `workload_contracts` 中的候选/参考来源 SHA、输入签名 SHA、精度政策 SHA 和 precision；缺项、其他题/形状/精度/政策不能借用完整资格。该审核是可信控制端输入，不是候选自行声明或密码学认证。

在已有完整规则和当前设备校准文件时，可在显式采集命令中增加 `--calibration FILE`。模型校验失败仍保存已完成的计数器报告，退出码为2。也可对现有完整记录离线调用：

```python
from kb_workflows.provenance.work_counters import execution_model
from kb_workflows.analysis.performance import decomposition

model = execution_model(counter_record, device_calibration)
factors = decomposition(wall_ms, busy_ms, sol_algorithm_ms, counters=model)
```

执行 SOL 从原始计数器换算的各资源需求除以相同物理设备的实测率，取最大值；零需求不虚构工作，缺率或单位不同拒绝。分解自动使用该 SOL，重新核对完整记录、审核范围、校准原样本和资源分量。仅有 `status="measured"`/来源字符串或手填 `sol_execution_ms` 不能启用 A/E；缺证据时继续只给 AE/O。与模型不一致的结果保留标记，不判作弊。所用实测率不是已证明的物理极限。

这是数值分析接口，调用者给出的 wall/busy/SOL_alg 尚未被一条完整正式归因门连接到原生结果和执行来源。另有[Neuron Parquet分析与审核模型](NEURON_WORK.md)，可重放逐指令工作注释、独立DMA包及execution/NKI调用点，并用审核后的完整资源映射和同设备校准拆A/E。report/capture可以由维护者人工配对，也可以重放固定Explorer工具、完整工件快照及全部输出的自动处理收据；后者默认预览，只处理已有文件。两者都不提升硬件计数器或正式来源资格。与原生时间/执行来源的完整关联和正式计分门仍由 R-16/R-08 继续实现。设备实验按用户要求不运行，不能用软件回归冒充这些证据。

实现见[原始记录与工作模型](../kb_workflows/provenance/work_counters.py)、[父采集器](../kb_workflows/provenance/counter_task.py)、[单次前向](../kb_workflows/provenance/counter_worker.py)、[命令](../scripts/audit_work_counters.py)、[分解](../kb_workflows/analysis/performance.py)和[回归](../tests/provenance/test_work_counters.py)。
