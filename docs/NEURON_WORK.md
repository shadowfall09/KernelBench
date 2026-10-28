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
python -m scripts.analyze_device_trace neuron-tables ./profile_parquet --output ./new-neuron-work.json
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

## 运行时key与NEFF缓存快照

Neuron worker与CUDA共用源码冻结、字节核对和编译注册，`source_registration=frozen_compile_code_objects`记录本次实际执行的代码对象；相同文件名或重新编译的相同文本不能借用注册身份。源文件保持可检查，旧`.pyc`不能替代本次编译；原始换行字节也参与核对。候选前后状态观察复用同一个注册表。

原生profile的`runtime_neff_bindings`从导出的Chrome trace选择`privateuse1_driver`完整事件，排除显式host worker及runtime提交事件。只使用原始`neff_cache_key`与`persistent_cache_key`，不靠kernel名称猜身份。每对key调用一次在候选导入前保存的`torch_neuronx._C.get_neff_info`，要求返回的persistent key精确相同，再保存NEFF字节、SHA-256与原始事件索引。接口和事件字段分别依据固定SDK源码的[缓存查询实现](https://github.com/aws-neuron/torch-neuronx/blob/50b14171affaa365f1df6aa015232427336f6a4d/torch_neuronx/csrc/core/NeuronBindings.cpp#L2259)与[设备事件导出实现](https://github.com/aws-neuron/torch-neuronx/blob/50b14171affaa365f1df6aa015232427336f6a4d/torch_neuronx/csrc/profiler/Session.cpp#L691)核对，未调用该运行时或运行设备实验。

实时采集和离线处理复用已有完整工件清单检查：默认4096文件、单文件128 MiB、总计512 MiB。trace另限64 MiB及200,000事件。不重复保存已有同哈希`.neff`；缺少时才写入`runtime_neff/<SHA-256>.neff`。清单与收据同时核对哈希、字节数及事件覆盖。缺少SDK接口、字段不完整、查询失败或key不匹配均保留未解析事件；`observed/partial/not_measured`区分完整关联、部分关联和未测量。

这里观测的是执行事件key与查询时的缓存字节快照；它没有捕获编译前端到NEFF的完整生成链，也没有读取实际加载到设备的二进制。因此`loaded_binary_identity_proven`与`formal_origin_qualified`保持false，不能据此通过Neuron自写内核门或授予L1/L2资格。

## 编译前端与框架降低观察

开发 profile 现在同时保留 `kernelbench_pro_neuron_compilation_observation_v2`。观察器在导入候选前固定 NKI 编译入口的代码对象，以当前 Python 线程的实际调用/返回事件关联 `compile_to_bir`、`NKIFrontend.compile`、`compile_mlir_to_bir`、`NirResult.build_config` 和框架 `_lower_custom_nki`。原有入口通过函数帧观察；为保存原生 AST，开发 profile 会临时代理 Parser 的原生 Kernel 构造入口，并在原生降低前保存输入、结束后恢复。它不覆盖已有 Python profiler；已有 profiler、回调被替换或入口代码变化都会留下未完成原因。正式 evaluate、baseline、paired_timing 路径不启用这项开发插桩。

只有实际前端返回的 MLIR 对象传给实际 BIR 降低、同一个 NirResult 返回框架配置、配置和 BIR 字节在框架降低时仍一致，才标记 `fresh_lowering_observed`。内存缓存、持久 trace 缓存跳过前端，或模块对象不对应时保留缺失阶段。源码身份来自本次冻结编译注册的代码对象；相同文件名和另行编译的相等代码对象不等于已注册对象。外部根函数单独记录为代码观察，不能据此认领自写内核。

NKI 0.7.0 的实际 `CompilationResult.dep_files` 另保留 Gather 文件清单。只有本次冻结注册的源码，或在候选导入前通过 wheel RECORD 校验的 NKI language/ISA/experimental 原语源码，才读取并保存；未知路径不会由评测器打开。每个根函数还核对静态可解析的辅助函数引用：实际注册代码对象、模块属性、闭包引用及批准原语的字节码/可调用闭包分别记录，原生 Kernel 包装只接受提前捕获的 SDK 类型。独立编译的相等文件名、外部函数、动态调用、缺失文件及前后引用变化保留 `partial`。注册函数引用的源码必须出现在实际 Gather 集合中，不能仅填文件名认领函数。

原语源码校验复用 [WheelSources](../kb_workflows/provenance/submission_source.py)，与 Triton 共用 RECORD、源码、字节码及可调用闭包检查；Triton 自己的版本、语言路径和 `__init__` 函数限制保持原政策。这里只核对 Gather 文件和静态引用视图，收据明确标记 `not_native_ast_callee_attestation`：它没有解码原生 Parser 的全部 AST，也不把动态依赖或静态引用当成实际编译调用证明。

每次 BIR 快照最多16 MiB、每个依赖源码和原生 AST 各4 MiB，三者共用64 MiB总额。源码保存在既有采集工件树 `compiler_sources/<sha256>.py`，纳入同一文件/字节清单；不会复制 SDK 源码进仓库。每次 Gather 最多1024文件，静态引用最多1024对象、4096调用位置。流水线、配置对象和降低调用分别最多4096项。原始字节保存到既有 profile 工件树的 `compiler_ir/<sha256>.bir`，纳入统一哈希/字节清单。收据记录入口代码摘要、逐阶段观察、BIR 摘要及实际 StableHLO 摘要/cache key；验证器检查阶段依赖、原始工件和模拟标记。StableHLO 摘要不等于保留了完整 HLO 文件。

实现依据是 [NKI 0.7.0 官方 wheel](https://pip.repos.neuron.amazonaws.com/nki/nki-0.7.0%2B37713544100.gfede04bb-cp312-cp312-linux_x86_64.whl) 的编译驱动，以及固定提交的 [Torch Neuron NKI 降低](https://github.com/aws-neuron/torch-neuronx/blob/50b14171affaa365f1df6aa015232427336f6a4d/torch_neuronx/python_ops/neuron_nki_call_lowering.py)。这不是所有 SDK 版本的兼容保证：其他前端、线程、缓存或编译路径未观察到时保持未知。本次另在隔离临时目录用真实 NKI 0.7.0 CPU Gather/序列化验证接口，禁止 Torch、Neuron runtime/后端工具、子进程与网络；未执行原生 Kernel.compile、BIR/NEFF 后端、设备或模型实验。模块交接与工厂恢复另由明确模拟的软件回归覆盖。

编译交接与静态引用不是原生 AST 证据的替代；原生定义输入另由下一节的独立收据核对。动态、外部及不支持的定义继续未知，正式连续计分和完整编译输出关联仍未完成。因此编译观察和运行时缓存快照都保持 `formal_origin_qualified=false`、`loaded_binary_identity_proven=false`；下一节的可选原生观察器单独核对加载字节与提交序号，R-08/R-16 继续部分实现。旧的 root-only v1 编译收据不能改标签升级为 v2，须重新采集。来源观察统一放在 [neuron_origin.py](../kb_workflows/provenance/neuron_origin.py)；[neuron_work.py](../kb_workflows/hardware/neuron_work.py)只组合采集与资源需求模型。

## 原生 Parser 的实际源码输入

[neuron_ast.py](../kb_workflows/provenance/neuron_ast.py)独立处理 NKI 0.7.0 的 MessagePack 输入和源码读取证据，避免把二进制格式解析继续堆进采集器。编译观察中的可选 `native_ast` 使用 `kernelbench_pro_neuron_native_ast_v1`。候选导入前固定 Parser、原始 `inspect.getsourcelines` 代码与 SDK Kernel 构造/编译/序列化方法；正常 Parser 入口临时返回透明代理，同一个原生对象在真正降低前先序列化，然后调用原有编译方法。前端返回的 module 对象继续核对是否传给上层前端。

收集真实源码读取的返回值，要求调用直接来自原生 Parser/构造边界，使用原始 inspect 的 globals/builtins，并匹配注册代码或候选导入前通过 RECORD 核对的 SDK 函数/类。返回源码须逐字匹配提前冻结的文本（按 Python 源码语义规范换行）；代码对象、原始文件 SHA、实际返回块 SHA 和定义范围分别保留。普通 Python 包装器中转的读取不能取得真实原生输入身份。注册表额外缓存原始源码，不能用运行中的 linecache 替代冻结文本。

解析原始 MessagePack 中的函数、类及类方法定义，把名称和完整源码范围逐项关联到实际读取记录及冻结源码 AST。重复读取只有来源与返回内容一致才可合并；缺失、含糊、未知外部定义、arch/sub-version/LNC 不同或不支持格式保留未完成。原始 blob 还保留参数及内联全局值，不能只靠 Gather 文件列表推断这些值的来源。`root_source_sha256`来自原生根定义的实际读取，而不是外层包装器的声明。

原始 AST 放在 `compiler_ast/<sha256>.msgpack`，与 BIR/源码共用64 MiB总额；收据同时保留有界编码，供离线重新解析并核对原始工件、定义覆盖和源码清单。解码最多4 MiB、1,000,000值、128层，定义最多20,000项、源码读取4096次。`msgpack`依赖仅加入 Neuron worker 和离线 analysis extra，导入模块不会加载 SDK或解码器。

只有全部定义都匹配直接读取、根定义存在、配置一致且非模拟时，`native_definition_sources_bound`为true；`module_handoff_observed`单独记录 Parser→上层前端的对象交接。它们证明的是受信 SDK 的源码输入关联，不证明输出因果/算法质量、完整 NEFF 编译链或正式墙钟等价，`formal_origin_qualified`保持false。格式/配额无法观察时记录原因并继续原有 SDK 编译，不能把诊断失败改成成功来源。构造入口被替换也会记录失格并恢复。

CPU 软件核验使用已记录官方 wheel 的真实 `Kernel.specialize()`、`serialize_python()` 和 `inspect` 读取；一个包含 load/arange/辅助函数的输入关联了180个函数、类和方法定义，全部有冻结输入记录。没有调用原生 `Kernel.compile`、neuronx-cc、NRT、设备或真实模型。该核验只确认 Gather 证据与格式，不认证实际 SDK 部署、硬件正确性或性能；常规测试保留明确模拟标记，不能冒充设备来源。

## 可选原生加载与提交观察

在 Neuron config 的 `[run]` 中启用：

```toml
neuron_runtime_origin = true
neuron_runtime_include = "/opt/aws/neuron/include"
neuron_runtime_compiler = "c++"
```

默认关闭，只用于公开开发 profile。SDK include 路径可配置；配置文件里的相对路径按配置目录解析。正式 evaluate、baseline 和 paired_timing 不启动观察器。启用时要求同一原生 worker 主机已安装 C++ 编译器及 NRT 头文件；缺失、编译失败或收集失败会明确失败，不退回一个声称已观察的结果。本轮没有安装或调用实际 NRT。

轻量 worker 入口先编译仓库自己的 [C++ 观察器](../kb_workflows/provenance/neuron_runtime.cpp)，然后在导入 Torch/Neuron 和候选前，以 `LD_PRELOAD` 重启同一 worker。编译直接使用所配置 SDK 的 `nrt.h`、`nrt_async.h` 和 `nrt_experimental.h` 声明；不自行猜测 `nrta_seq_t` 布局。构建收据固定源码、编译器及这三个直接头文件的摘要，并在构建后复查，保留命令与 SO 摘要。它不认证工具链的全部传递依赖。

观察外层 `nrt_load`、`nrt_load_collectives`、`nrt_unload` 和 `nrta_execute_schedule`。加载时保存字节副本，并把这个副本传给真实 NRT；成功加载的缓冲区持续保留到成功卸载。模型地址复用产生不同代次，库内部嵌套转发不会重复认领一次加载。全局符号不可见时只查找已经加载的唯一 `libnrt.so` 路径，兼容 Python 的局部加载；不会因此另行加载 SDK，多版本含糊时拒绝。复制最多128 MiB/次、总计512 MiB、4096代，日志最多200,000条收集操作；超限留下错误，不取得加载身份。

profile 同步完成后封口原生日志。`native_runtime/` 保存 SO、构建清单、原始 JSONL 和每次加载的 NEFF；它们仍受整个采集树的4096文件、每文件128 MiB、总计512 MiB限制，原生快照不额外增加磁盘额度。worker 复查加载观察器的工件，再按日志重放成功加载/卸载的代次，用原生异步序号关联框架 `privateuse1_driver` 事件。两种观察器共用事件选择与区间有效性政策，显式 host worker 不在该范围。序号缺失、重复含糊、没有本次活模型或文件不一致时拒绝相应关联；没有用事件名称或当前缓存字节认领加载身份。

`kernelbench_pro_neuron_native_runtime_v1` 的 `loaded_binary_identity_proven` 仅在真实收据覆盖全部选中 driver 事件时为true。这里证明的是观察到的加载与异步提交关联；调度返回成功本身不证明设备完成、输出有用性或全部硬件活动覆盖。模拟收据始终不能取得真实加载身份，默认验证也拒绝模拟标记。`formal_origin_qualified` 仍为false：辅助函数编译归属、实际评分连续调用和原生时间的完整关联尚未完成，不能凭这项开发采集通过自写内核门或授予 L1/L2。

接口依据固定提交的 [NRT 调用封装](https://github.com/aws-neuron/torch-neuronx/blob/50b14171affaa365f1df6aa015232427336f6a4d/torch_neuronx/csrc/core/runtime/NRTUtils.cpp)及其正式头文件。当前验证只编译并运行仓库自写的 CPU 假 NRT，覆盖实际预加载、嵌套转发、卸载后同址复用、序号匹配/缺失、原始工件及启动参数；它不证明某一实际 SDK 安装的 ABI 或设备兼容性。其他加载/执行 API、不同进程、绕过预加载的路径不在支持范围。

## 审核后的资源需求模型

Neuron worker的profile返回`kernelbench_pro_neuron_verified_capture_v2`收据。它在冻结编译源码再次核对后生成，保留候选/参考源码、配置、完整输入/输出内容摘要、输入几何和布局、此前正确性用例、实际设备/软件、模式、seed、比较政策和原生工件哈希/字节数，并记录原生worker的可见core清单。实际候选模型在采集前后分别通过共享状态遍历记录参数/buffer、普通属性、类数据、入口代码/default/closure及输入/状态别名摘要；合法状态更新可使前后摘要不同。内容摘要分块复制到主机，不再用参考模型的256点注册状态探针替代候选状态。所有摘要及输入隔离复制都在采集区间外完成，采集中只有候选前向及完成屏障。

不支持的模型/容器/Tensor子类、opaque对象或存储布局保留`not_measured`或缺失摘要，原始采集可保留，但不能通过完整需求模型验证。旧v1收据必须重新采集，不能改标签升级。代码摘要只是入口观察，不证明全局依赖、host控制流、执行来源或与正式原生计时等价；CPU/注入适配器收据继续保留模拟标记，不能建真实设备模型。

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
python -m scripts.analyze_device_trace neuron-tables ./profile_parquet --output ./new-neuron-model.json \
  --capture ./profile-result.json --review ./reviewed-neuron-work.json \
  --calibration ./same-device-calibration.json
```

命令不会采集、校准或调用Explorer。成功结果包含全部原表、采集、审核、校准原始样本、逐引擎/资源需求及SOL分量；模型失格时退出2，仍在新输出保存完整原始报告及`not_measured`原因。单独省略全部三个选项仍输出原来的工作报告。

`neuron_work.execution_model()`将正资源需求除以同设备实测速率，取各分量最大值；`validate_model()`从完整证据重算。`performance.decomposition(...,counters=model)`可据此拆A/E，但标注`work_evidence_kind=reviewed_neuron_instruction_annotations`、`hardware_counters_collected=false`，不能称为硬件计数器。缺证据仍只给AE/O；分解也不替代原生计分时间或把实测率当物理定理。

## 从原始工件自动关联处理结果

已有真实原生profile结果和工件时，先预览：

```bash
python -m scripts.analyze_device_trace neuron-process --capture ./profile-result.json \
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

CUDA现有模型见[工作计数器](WORK_COUNTERS.md)。实现及回归分别位于[表读取器](../kb_workflows/hardware/neuron_tables.py)、[审核模型](../kb_workflows/hardware/neuron_work.py)、[读取命令](../scripts/analyze_device_trace.py)、[自动处理器](../kb_workflows/hardware/neuron_processing.py)、[处理命令](../scripts/analyze_device_trace.py)、[表测试](../tests/hardware/test_neuron_tables.py)、[模型测试](../tests/hardware/test_neuron_work.py)和[处理链测试](../tests/hardware/test_neuron_processing.py)。
