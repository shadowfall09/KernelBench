# 设备校准、边界证书与 Headroom 评分

本轮只实现代码并进行 CPU/模拟回归，**不运行校准、GPU、Trainium 或真实模型实验**。下列命令说明将来的使用方式。本仓库没有生成当前设备的测量收据，也没有把历史规格表转换成“实测”证书。

`score_mode = "bounded_gain"` 保留原有 K 次摊销评分。显式选择 `headroom` 后，使用当前有效变体中稳态均值最小的基线，按论文公式计算稳态分数；另报同一已选基线的冷成本与摊销 bounded gain。后者是诊断值，不冒充重新选择 K 最优基线的独立榜单；报告保留两种最优变体名称和选择策略。失败、缺失和证据不匹配仍占声明槽位，正式聚合仍为按题取样本均值、再按层宏平均。

## 校准接口与收据

`kb_workflows.devices.DeviceAdapter`定义`caps()`、`flush()`、`time()`、`calibrate()`和`attribute()`，提供CUDA与[原生Neuron适配器](NEURON.md)。校准只执行可信探针，不导入候选，不会由评测自动启动。CUDA使用events采集探针时间；Neuron标明同步墙钟口径，缺少真实FP64能力时不生成FP64速率。两者均未在本轮运行实机校准。

```bash
python -m scripts.calibrate_device --device cuda:0 --output calibration.json
```

探针包括超过两倍已知 L2 的设备内拷贝、多个矩阵尺寸的 FP32/FP16/FP64 GEMM、支持时的 BF16 GEMM，以及同步墙钟下单元素加法的启动代理成本。矩阵计时用当前流 CUDA events；拷贝计入读与写字节，FMA 按两次操作计算。保留每个原始时间样本、工作量、矩阵扫描结果和中位时间对应速率。FP32 禁用 TF32，半精度归约禁用 reduced precision；退出或报错后恢复后端设置。未知 L2、缺物理 UUID/驱动版本、显存不足或探针失败均拒绝生成有效收据。

收据绑定物理 UUID、设备属性、驱动、Torch/CUDA/HIP 版本与 evaluator digest。读取时重新计算样本对应的速率、启动中位数和收据内容哈希。跨设备和旧评测器收据不能复用。`flush()` 写入超过已知 L2 的缓冲，但不证明每级缓存完全驱逐；默认微基准测稳态，不使用该写入伪称冷 HBM 测量。

这些速率是微基准达到的速率，**不是已证明的物理上限**。时钟、功耗、争用、微基准规模及矩阵算法仍可能改变可达速率；收据明确写 `clock_and_power_state_certified=false`。基于它们的 SOL 是经过审查的模型估计。

## 可信规格与见证

在原 `problem_specs` 的逐题记录中新增 `headroom_certificate`。下例仅展示格式，省略号和占位哈希不是有效数据：

```json
{
  "level1": {
    "1": {
      "layer": "I",
      "headroom_certificate": {
        "protocol": "reviewed_headroom_certificate_v1",
        "layer": "I",
        "reference_sha256": "参考文件原始字节的SHA256",
        "precision": "fp32",
        "input_signature": [{"shape": [1024, 1024], "stride": [1024, 1], "dtype": "torch.float32"}],
        "calibration_sha256": "校准收据的receipt_sha256",
        "review": {"reviewer": "审查者", "rationale": "列明计数推导、合法格式、最低流量与假设"},
        "reference_work": {"memory_bytes": 12582912, "fp32_flops": 1048576},
        "algorithm_work": {"memory_bytes": 12582912, "fp32_flops": 1048576}
      }
    }
  }
}
```

工作量键必须对应校准资源；可额外声明有审查依据的 `critical_path_ms`。各资源时间取最大值，不相加。参考与最优已知算法分别给出工作量；算法模型不能比参考模型更慢。I/F 使用参考模型，D 使用算法模型作为目标。计数及数值合法性由可信规格的审查承担，不能使用候选自行声明的 FLOPs、dtype 或来源。

D 还必须有 `design.algorithm`、`comparison`、`numerical_contract`、`maximum_witness_gap >= 1` 和 `reachability`。见证只能从已经完成、至少两次独立最终复测且标准/隐藏均通过的记录导出：

```bash
python -m scripts.export_reachability runs/example --problem 1 --sample 0 \
  --calibration calibration.json --output witness.json
```

此命令只读取既有结果，不启动评测。导出前执行当前冻结源码、基线、配置、正确性及测量证据检查。见证包含全部独立标准/隐藏记录；读证书时重新核对其来源/精度/输入/设备、独立 nonce、原始时间与均值。达到算法 SOL 的倍数必须落在审查者声明的区间内。见证可来自另一个参考设备，但必须附 `design.witness_calibration`；报告分别标记参考设备已见证与当前目标设备已见证。

这些文件应放在可信 verifier 一侧。内容哈希检测修改和错误复用，不提供签名认证或候选进程安全隔离。

## 评分接入

```toml
[run]
score_mode = "headroom"
device_calibration = "calibration.json"
problem_specs = "reviewed_specs.json"
paired_timing = true
paired_trials = 30
final_repeats = 2
```

两个文件路径相对 TOML 文件解析。缺文件在配置阶段报错，证书缺失/未审查/来源不匹配在计分阶段将对应槽位置零。已有的标准/隐藏、测量、区间及最终复测门继续生效。配对计时和独立复测按上例推荐开启，是否要求更高级证据仍由 `evidence_level` 控制；本功能不会把尚未收集的来源证明或硬件计数器标为通过。

```
SOL_ref = max_resource(work_ref / measured_rate)
SOL_alg = max_resource(work_algorithm / measured_rate)
target  = SOL_alg for D, otherwise SOL_ref
score   = clip(log(T_ref / T_candidate) /
               max(log(T_ref / target), log(1.25)), 0, 1)
```

`paper_analysis.json` 分开保存 `score_macro`、`bounded_gain_macro`、`headroom_macro` 和逐层结果；未选 headroom 时，其分数为 null。逐项保存目标、各资源分量、证书/校准哈希与低于模型估计的标记。低于估计只提示核查模型，不自动判作弊。旧评分读取器也重新核对外部规格与校准文件内容；修改或删除后，旧资格失效。Harbor 只把所需单题规格和校准放入 verifier 文件，不放入 agent 环境。

## 时间线与性能解释

开发 profiler 现在导出并解析一个标记墙钟区间的 Chrome/Kineto 时间线，报告设备活动并集和 kernel 数；重叠核执行不重复累计。CPU CUDA runtime 调用、其他设备和区间外活动排除。开发诊断不替换计分墙钟。

```bash
python -m scripts.analyze_device_trace trace.json --device 0 --output activity.json
```

`performance.decomposition()` 在没有可信、来源绑定的工作计数器时，只给出 `AE=T_dev/SOL_alg`、`O=T_wall/T_dev`，A/E 为 null。现在必须提供可重放原始计数、完整审核实例契约与同物理设备校准模型才能分开计算A/E；执行SOL自动重算，`status="measured"`字符串或手填数值不能启用。模型不一致会保留标记，见[工作计数器](WORK_COUNTERS.md)。`launch_only()` 还要求配对区间超出参考控制噪声、核数减少、busy 时间基本不变，且墙钟收益不超过校准启动成本解释的范围；这是诊断分类，不自动提供计分资格。

来源关联图只接受可信 verifier 的 correlation 映射，未知保持 unknown，不能根据 kernel 名字宣称自写。Nsight采集入口和计数器重放已有软件路径，未运行设备采集；[Neuron Parquet分析](NEURON_WORK.md)已有完整分片、逐指令工作注释和DMA分账软件，来源仍未知。Neuron执行工作模型、完整来源关联和正式计分归因门仍需后续实现。

实现依据：[PyTorch profiler](https://docs.pytorch.org/docs/2.12/profiler.html)、[数值精度说明](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html)、[NVIDIA nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。
