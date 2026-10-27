# Agent资源预留与核算

本版交付配置、控制器与CPU/模拟回归。没有运行模型、GPU、Trainium或训练实验。

每个样本保留原来的候选数与总墙钟上限，另可开启以下限制；`0`表示关闭：

| 配置 | 范围 | 支持入口 |
| --- | --- | --- |
| `agent_token_limit` | 累计完整prompt token IDs + 返回的生成token IDs；每轮prompt重新计入 | `local-transformers` |
| `agent_output_token_limit` | 累计输出token；供应商包含的推理token按completion usage计入 | `local-transformers`、支持completion上限参数的LiteLLM模型 |
| `agent_max_output_tokens` | 每次调用最多输出数，默认8192；用于本地模型或开启token预留的调用 | 同上 |
| `agent_device_seconds` | 开发编译、标准测试与profiling占用独立worker的墙钟总额，包含子进程启动与清理 | 原生local/Docker/Modal生成；Modal生成使用Sandbox内local验证，其他原生生成可选local/Docker、Modal或新版queue |

设备额度不包含最终验证，也不包含本地语言模型推理；`local_model_device`单独选择模型设备，默认`cpu`。worker占用时间不等于GPU busy时间、供应商账单或跨进程独占证明。

## 预留与停止

模型请求前，控制器先预留允许的最大输出；本地总token预算还预留本轮完整prompt。已知实际用量结算后返还剩余预留。用量缺失、请求异常或超限时保留证据、扣除预留并停止后续调用；超限的已知实际数仍原样记录。模型响应没有合格用量证据时不会交给开发评测器。

本地模型先用相同chat template/tokenizer生成实际输入ID，再提交这组ID。只生成一条序列，不使用beam search。预算衡量提交和返回的ID，不衡量模型内部浮点计算量。

LiteLLM在请求前检查模型支持的`max_completion_tokens`/`max_tokens`参数，设置`drop_params=false`、单结果和零SDK重试；不支持便拒绝调用。服务端硬上限依赖供应商参数契约，收到缺失或超额usage会拒绝该样本资格。供应商的prompt计数估算无法证明严格总token上限，因此该限制只向本地适配器开放。Claude CLI可能有不可观测的辅助调用，两个token预留选项均拒绝在该后端启用。原有美元费用限制仍是调用间估算检查。

开发worker每次执行前预留`min(任务超时+5秒清理余量, 剩余总墙钟, 剩余设备额度)`。不足5秒时不启动新任务。父进程把执行超时缩至预留减5秒，并用单调时钟测量整个transport调用，包括超时终止与清理；子进程提供的同名收据会被覆盖。profiling也要单独申请和结算。

已知设备占用未超额时，预算耗尽仍可保留此前开发通过的最佳候选。缺失收据、无法完成清理或实际超额会拒绝计分资格。进程终止不能保证任意驱动或内核被立即抢占；收据明确写`physical_preemption_certified=false`，没有实现纯GPU busy时间的物理硬限。新版HTTP队列与Modal验证已有下述服务端租约；Modal原生生成复用Sandbox内的token/worker预留，Harbor外部agent仍不支持这些硬预留。

可选[内存工具与重复检查](TOOL_CHECKS.md)在同一次验证请求的超时和设备worker墙钟内执行；插桩、版本查询及无插桩复核没有独立额度。它们的耗时不进入性能分数，但计入实际开发资源占用。

受限Docker仍按父进程完整transport墙钟结算，不把容器CPU/cgroup或设备可见性当成GPU busy硬限。[产物重评分](ISOLATION_REGRADE.md)只运行新验证阶段，旧生成成本保存在历史快照，模式单独标记为`artifact-regrade`；不会把旧费用/token重新认证成新预算证明。

## HTTP队列的开发租约

`evaluation = "queue"` 可搭配同一个 `agent_device_seconds`，不新增预算配置项。客户端在提交前要求 `/health` 返回 `kernelbench_pro_queue_v3`、状态正常且`device_backend`匹配，并逐次核对响应的协议、后端与请求指纹；旧worker不会收到带预算的任务。Neuron请求即使没有设备预算也需要该握手；旧的无预算CUDA服务保持兼容。服务端以完整请求指纹绑定任务和唯一租约nonce，同一nonce不能换任务重花。worker从统一设备配置启动，状态库绑定后端，客户端不能指定服务端解释器；这些软件约束不构成GPU busy时间的物理硬限。

服务端从接收租约起，以单调时钟限制排队加执行窗口；排队消耗可用窗口，但不收取设备用量。到期未启动的任务不执行；运行中的任务会通知本地验证父进程终止进程组，包括主进程退出后仍存活的子进程。服务端父进程按实际worker墙钟（含启动、校验和清理）覆盖子进程收据，客户端用匹配的请求、nonce与收据结算。它不计网络等待、GPU busy时间或账单。

客户端把提交、轮询和重试放在同一等待期限内。等待超时、轮询异常或提交响应丢失后，使用原请求ID与指纹发送取消，并在5秒清理余量内查询终态；原调用仍为失败。`cancelling`不是清理成功。确认尚未启动的任务可按零用量结算；用量未知、取消无法确认或清理异常时扣除整笔预留并拒绝资格。清理异常的worker停止接单，不能把短暂父进程耗时当成后续用量已结束。

取消状态和nonce绑定持久化。服务重启后，未完成的带租约任务标为失败且用量未知，不重新排队；已完成记录仍可幂等读取。未带租约的旧运行任务保留原有重排策略。服务端是配置中信任的验证器；这些内容指纹不提供独立网络身份认证或物理设备抢占证明。

## Modal验证的开发租约

原生local/Docker生成可配置`evaluation="modal"`和`agent_device_seconds`，无需新预算项。Modal生成的开发验证固定在自己的Sandbox内local执行，按下一节处理；Harbor外部agent仍拒绝这些资源预留。验证session为带设备预算的运行建立临时共享Dict；远端在启动子进程前，通过`put(skip_if_exists=True)`占用nonce并保存完整请求指纹。重复、并发或平台重试不能重新执行同一nonce；存储缺失或查询异常也不会启动工作。父进程墙钟覆盖该准入查询、编译/验证子进程及清理，覆盖子进程自报收据。

客户端使用单次spawn，显式关闭函数重试，绑定`kernelbench_pro_modal_lease_v1`、请求指纹、nonce和CUDA后端。提交与取结果共用不超过执行窗口的等待期限；每次函数的服务端timeout另设为预留秒数向上取整，即使控制端消失也不采用原来的长任务timeout。平台timeout以整数秒设置，不等于精确设备抢占；真实父进程用量超出预留仍会拒绝资格。取消请求要求终止容器，并有至多5秒等待上限；丢失提交响应、错绑、等待到期或取消失败均保留未知用量，扣除整笔预留，不能靠重发nonce重置预算。

成功响应只按远端父进程的单调时钟结算，云端调度、容器冷启动、网络等待和GPU账单不冒充该worker用量。取消API确认也不提供可退款的实际用量或物理抢占证明。云端服务及共享存储属于配置中信任的执行平台。本次只做CPU/模拟软件回归，并静态核对已安装Modal SDK 1.6.2接口，没有创建Modal资源或启动云函数。API依据为[FunctionCall](https://modal.com/docs/sdk/py/latest/FunctionCall)、[函数配置](https://modal.com/docs/sdk/py/latest/Function)和[Dict](https://modal.com/docs/sdk/py/latest/Dict)。

## Modal原生生成

`generation="modal"`现在复用同一mini-swe-agent控制器、输出token预留与开发worker租约，不增加预算配置。外层`evaluation`仍选择最终验证后端；开发job固定`evaluation="local"`、设备0，父/子预算协议统一记为`development_verifier="modal_sandbox_local"`。LiteLLM输出额度及worker墙钟可同时启用；Claude单次CLI适配器只能使用设备额度，精确总token与本地checkpoint仍限本地适配器。

启动前绑定实际job、公开参考、verify脚本和可选输入/精度/重复政策文件、源码版本及随机nonce。`run_agent.py --admission`在调用控制器前核对绑定并以独占文件创建准入标记；同一Sandbox的重复exec不能再启动一次，也不能重置账本。SDK创建Sandbox时不提交agent命令，agent只由这个受控入口启动。

原生控制器继续限制`agent_timeout`，Sandbox另有`agent_timeout+120`秒的平台timeout。上传、exec、等待与工件下载共用同一个客户端期限，不逐次重开；终止通过`terminate(wait=True)`有界等待，最多额外5秒。终止API确认是平台终态观察，不是GPU物理抢占或退款依据。缺收据、绑定不符、丢失回复、到期或终止未知均保留失败/未知用量，不补跑同一尝试。失败时可在原期限内保留标为未核对的源码和日志。

返回收据绑定控制器结果、账本和逐文件字节数/摘要。完整的结果JSON和工件集合分别受现有`isolation_report_mb`限制，不静默截断；下载保留trajectory、每轮候选/反馈/标准结果和model_calls。拒绝越界路径及symlink。评分再次核对job/源码绑定、终止观察与全部工件，模型调用或候选工件被改写后不能继续获得预算资格。正常输出token/worker额度耗尽仍可使用此前合格的最佳候选；用量未知或缺失完整收据不能获得资格。

预算只计算受控模型调用与开发worker墙钟。云端持卡空闲、登录/镜像构建、调度、网络及账单不冒充GPU busy或worker用量；本版没有运行云函数、Sandbox、设备或模型实验。文件传输按已安装SDK 1.6.2及[官方filesystem接口](https://modal.com/docs/guide/sandbox-files)实现，使用`filesystem.write_text/stat/read_bytes`，移除旧`Sandbox.open`路径。

## 离线本地模型

安装可选依赖：`pip install '.[local-model]'`。配置中的checkpoint必须事先存在；适配器不下载模型、禁用远端Python代码并只加载safetensors权重。下面是配置示意，需填写本地路径和实际GPU目标：

```toml
[run]
generation = "local"
evaluation = "local"
generation_mode = "feedback"
agent_backend = "local-transformers"
model = "/path/to/local/checkpoint"
local_model_device = "cpu"
agent_temperature = 0.0
agent_token_limit = 32768
agent_output_token_limit = 8192
agent_max_output_tokens = 4096
agent_device_seconds = 120.0

[gpu]
profile = "RTX_A6000"
devices = [0]
```

相对checkpoint路径以TOML所在目录解析。权重、配置及tokenizer资产内容哈希参与生成签名；实际加载的快照哈希也保存进模型用量，评分时复核。模型资产变化不能继续沿用旧预算证明。未运行真实checkpoint兼容性或生成质量实验。

## 恢复、评分与研究

预算快照写入模型调用记录、`agent_result.json`，再汇入`generation_summary.json`、`generation_attempts.json`及`generation_budget.json`。正式分析重算账本条目、额度、调用数和当前生成签名，核对原始尝试与汇总一致；未知或超限的条目不合格。这三份文件也进入旧评分门的内容指纹，分析后改动会使旧分失效。这些内容校验不是密码学来源认证或OS沙箱。

同一开启预留的样本只允许一个可审计的生成尝试。`resume`可以复用签名和源码均匹配的已完成结果；崩溃、不匹配配置或人为重跑必须使用新的run名称，不能清空账本后悄悄重花一次预算。当前交互skill会话无法观测全部模型调用，拒绝使用这些预留。

[研究工作流](STUDIES.md)的`vary`增加`total_tokens`、`output_tokens`和`device_lease_seconds`；对应曲线横轴是`declared_total_tokens`、`declared_output_tokens`和`declared_device_lease_seconds`。配置预检按后端能力拒绝不可兑现的计划；默认仍只生成计划，不启动实验。不同口径、未变化的其他预算和硬件分别比较。

R-18仍标为部分实现，剩余包括纯GPU时间物理硬限及Harbor外部agent的资源契约，均没有用worker墙钟或终止确认代替。
