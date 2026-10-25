# Trainium / NKI设备工作流

本版实现通用设备适配器和独立Neuron轨道，**没有运行Trainium、GPU、校准、真实模型或训练实验**。CPU及模拟回归不能证明SDK部署、算子覆盖、硬件正确性或性能。

## 配置与环境

旧`[gpu]`配置继续用于CUDA。`[device]`选择计算平台，不能与`[gpu]`同时设置：

```toml
[device]
backend = "neuron"
hardware = "Trainium2"
target = "trn2"
devices = [0]
python = "/absolute/path/to/neuron-venv/bin/python"

[run]
generation = "local"
evaluation = "local"
enable_ncu = false
level = 5
suite = "production41"
task_track = "implementation"
start = 19
end = 20
paired_timing = true
final_repeats = 3
```

可从[Trainium2配置](../configs/trainium2.toml)开始，Trainium3对应`hardware="Trainium3"`、`target="trn3"`。这里的`devices`选择NeuronCore，存在`NEURON_RT_VISIBLE_CORES`时按分配列表中的相对索引映射；不读取CUDA设备分配。每个worker使用一个物理core、LNC=1，包含区间形式的分配会先展开校验。设备身份记录序列号、core、驱动、软件与拓扑，缺失或矛盾时拒绝；没有用同名设备替代物理身份。

验证器使用[AWS原生TorchNeuron后端](https://github.com/aws-neuron/torch-neuronx)，要求Python 3.12、Torch 2.14、`torch-neuronx>=2.14.3,<2.15`及`nki>=0.6,<0.8`，支持文档列出的Trainium2/3。旧XLA产品不符合该接口。CUDA环境仍用原来的Torch 2.12；通过`device.python`使用独立vendor环境，源码从同一checkout加载，避免混装两个Torch。可在已安装Neuron驱动/runtime/tools的机器上按官方说明准备环境：

```bash
python3.12 -m venv /path/to/neuron-venv
/path/to/neuron-venv/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
/path/to/neuron-venv/bin/python -m pip install -r requirements-neuron-worker.txt
export PATH=/opt/aws/neuron/bin:$PATH
```

本次没有执行上述安装或设备命令。主仓库的[worker依赖](../requirements-neuron-worker.txt)不替代系统驱动/runtime。序列号来自[Neuron sysfs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-sysfs-user-guide.html)的`info/serial_number`，core/内存映射来自`neuron-ls --json-output`。客户端与worker必须使用相同源码；解释器路径只用于本地验证，不发送给queue服务执行，远端服务使用自己的环境。

```bash
python scripts/run_batch.py --config configs/trainium2.toml --dry-run
```

预览不会导入Neuron SDK或请求硬件。实际执行仍需显式运行相应stage。本版支持本地及HTTP queue验证、本地agent生成；CUDA Docker/Modal/Harbor环境、CUDA Graph、Inductor和Nsight选项会拒绝。设备容器接入继续属于隔离工作的范围。

## HTTP验证worker

worker读取同一TOML中的设备字段，无需复制控制端的模型或校准文件：

```bash
python -m kb_workflows.runtime.queue_worker --config configs/trainium2.toml \
  --state cache/verify_queue/neuron.sqlite3 --dry-run
```

预览只显示后端、分配内映射的物理core、解释器和状态路径，不加载SDK、不创建状态库。部署时移除`--dry-run`，按需指定`--host`与`--port`；本次未启动设备服务。单worker使用一个core，多设备配置须通过`--device`选择一个相对索引。

`device.python`可为相对TOML的路径或绝对路径，保留venv的`bin/python`符号链接路径；解析到基础Python二进制会丢失vendor环境。服务端在SDK加载前切换至该解释器，并设置单core、LNC=1、禁用fallback及隐藏CUDA；实际SDK与配置硬件不匹配时不启动监听。客户端不能指定服务端解释器。每个worker/core使用独立状态库，库内绑定后端，不能将旧CUDA状态改作Neuron；所有Neuron请求要求v3协议和后端匹配。详情见[队列](BATCH_WORKFLOWS.md#persistent-local-verification-queue)与[预算契约](RESOURCE_BUDGETS.md)。

## NKI提交与基线

保留`ModelNew(torch.nn.Module)`接口，forward接收原生`neuron`张量。使用`import nki`、`@nki.jit`及`torch_neuronx.nki_op`注册内核；API示例见[官方NKI集成说明](https://github.com/aws-neuron/torch-neuronx#using-nki-kernels)。编译源文件实际写入独立构建目录，支持NKI源码检查；返回前再次核对源码，运行中改写文件不能获得有效结果。

基线组合包括原生eager和`torch.compile(backend="neuron",fullgraph=True,dynamic=False)`。Production的attention、layer norm、residual RMSNorm和GQA另有经过语义回归的框架库改写变体。每个变体独立验证/计时，再取当前正确且最快者；没有把CUDA基线或CPU耗时迁移为Neuron基线。库改写不保证编译器在所有版本上选择同一个NKI库内核。

输入角色、shape和dtype不变。Production的40个可传输存储接口（除FP8存储题21）已有CPU前端回归，包括反向题；这不是40题的Neuron算子覆盖认证。题21明确拒绝隐式降精度。其他题的编译/运行不支持会保留为失败，不能事后从声明分母删除。CPU fallback在worker加载SDK前关闭，输出必须位于声明设备；模拟适配器的结果带标记，正式计时资格会拒绝。

正确性复用两种调用顺序、隔离副本、逐次变化输入与原生比较。隐藏阶段使用冻结的新种子及私有输入；开发工具不含隐藏反馈。Production数值契约及已审查辅助契约的FP64 oracle在CPU执行：原生Neuron文档明确float64按float32运行，设备上的`.double()`不能被当作FP64证明。错误结构、离散值、非有限位置与符号、精度和修改环境均不能借平台变化放宽。

同步墙钟、输出消费、逐次验证、冷启动摊销、参考/候选/参考控制配对和冻结后独立重测沿用原生协议。参考和候选各自保留真实来源与设备身份。Headroom校准与D类可达性见证必须来自同一Neuron设备和当前配置，CUDA见证不能替用。平台各自报告，不汇成一个跨硬件排行榜。

## 五个适配方法与诊断

`kb_workflows.hardware.devices.DeviceAdapter`定义`caps/calibrate/flush/time/attribute`，工厂只注册CUDA与Neuron，实现导入时不加载SDK。可使用以下显式探针入口；本轮未运行：

```bash
/path/to/neuron-venv/bin/python -m scripts.calibrate_device --backend neuron --device neuron:0 --output neuron_calibration.json
```

Neuron校准保留copy、FP32/FP16/BF16编译GEMM的原始样本和全部尺寸扫描，以及同步启动代理；缺少真实FP64能力不会生成FP64速率。计时是含dispatch/输出消费的同步墙钟，单独标明，不能冒充CUDA events。缓存拓扑未知时`flush()`要求显式scratch字节数，记录已写字节且不声称完全驱逐。

开发profile使用原生`NeuronConfig`与`PrivateUse1`活动，并按[当前SDK采集接口](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.33.0/tools/neuron-explorer/how-to-profile-workload.html)设置`NeuronProfiler.export_trace`回调，收集框架trace、NEFF/NTFF并保存文件哈希。worker在导入和编译前设置本次私有NEFF缓存，结束或异常后恢复原环境；所有Neuron评测统一使用no_grad，合法的局部enable_grad仍须自行恢复。环境守卫记录并恢复候选泄漏的梯度模式。

采集前从参考当前状态和相同seed冻结预期输出；启用独立数值验证时，先准备参考前向前的oracle票据。采集只执行对应的候选调用，关闭profiler后验证该次输出及模式/环境，没有从已经推进的参考状态再算真值。profile仍是开发诊断，不替代正式时间或隐藏测试。上述SDK接法仅模拟验证，未实机采集。

新[Parquet工作分析](NEURON_WORK.md)读取schema 12.0.1完整分片、逐指令FLOPs/HBM注释和独立DMA包，并保留未知来源。旧版本设备指令JSON可由官方[neuron-profile工具](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/neuron_profile_for_nki.html)导出；其已有文件仍使用下面的独立离线入口，不启动设备：

```bash
python -m scripts.analyze_device_trace profile.json --backend neuron --timestamp-unit ns --window START END --output profile_summary.json
```

单位和capture窗口必须按实际导出版本明确提供，不能猜。分析计算tensor/vector/scalar/DMA及总活动的时间并集，重叠不累加；未知标签与来源保留未知。指令条数不冒充kernel数，capture跨度不冒充host墙钟；硬件工作计数器、二进制来源关联和正式归因门仍归R-16。profiler耗时不替代正式计分的墙钟样本。
