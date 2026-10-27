# 设备执行来源与轨道判定

CUDA L1/L2现在从实际计分进程采集源码→产物→函数→launch证据，支持Triton直接launch、受信wheel cubin、已适配的CUDA C++ object、Graph replay/核参数更新及原生复制/memset。自写/库轨道门使用这份实际证据，独立的对照/重放报告继续只作诊断。未关联的CUDA Runtime注册、其他编译器/loader、整体/其他Graph更新、CUkernel及Neuron来源保持未知；R-08仍为部分实现。

## 实际计分模式与来源门

```toml
[run]
evidence_level = "L1"
kernel_track = "custom"  # 或 library；两种轨道分组报告
origin_minimum_custom_fraction = 0.95
```

CUDA L1/L2的`evaluate`、隐藏验证、`paired_timing`及`baseline`在实际工作进程启动持续CUPTI采集，早于候选导入；原始计时仍调用同一个model对象，不增加转发包装器或同步。计时前打开候选窗口，计时结束立即关闭，再观察模式、模型状态和输出；摘要与验证不计入自写份额。持续记录窗口外的launch/activity，使配对参考、控制及独立基线也使用同一种callback/activity模式。该模式明确标为`synchronized_wall_continuous_cupti_v1`，含真实采集开销，不声称与未采集计时等价。普通`current`/L0保留原模式。

`metadata.origin_capture`使用`kernelbench_pro_scored_capture_v1`，只保存一份实际raw trace，并绑定请求、实际时间、设备及逐调用收据；`native_call_receipt`通过摘要引用它。基线也保留绑定实际样本及seed的采集收据。评分重算来源门，不能只写passed；同时拒绝无采集基线、混合模式、模拟/不完整/丢记录的观察。工具缺失时保留数值结果与失败原因，候选只执行一次，不重跑或补造来源。

原生v2在每次调用前后另取模型状态摘要：参数、buffer、普通属性与容器、类数据、实际入口代码/default/closure及输入与状态之间的存储别名。合法连续状态可以改变，要求的是实际进程与重放逐次匹配前后状态，而非强制状态不变。重新编译的等值代码不能继承本次注册的代码身份；外部Python/builtin入口只记录观察，不据名称授予所有权。未知描述符、容器/张量子类、定制metaclass或无可靠适配的对象标为未测，不作为空状态。摘要在计时区间和CUPTI/NVTX候选窗口外完成；校验不执行候选property、iterator或state_dict。 这些区间外观察可能影响缓存，未运行设备实验量化影响；它们不闭合任意全局依赖或证明其他host路径。

`kernel_track="custom"`要求每次实际调用来源完整且独占自写busy时间比例达到冻结阈值；`library`允许已经建立来源的自写、库、框架及运行时工作，仍不接受未知活动。库工作是合法答案，未过自写门不改变数值正确性。轨道与阈值纳入生成提示、请求/恢复指纹、评分及实验比较组；更改声明必须生成新匹配工件。诊断CLI省略阈值时沿用配置，绑定已存实际结果时不能改变其冻结政策。

`origin_task.scored_origin()`离线核对实际采集并重算事实；`origin_check()`生成`kernel_origin`证据。只有这个实际执行上下文实现来源门；通用`summarize()`和后面的独立诊断仍不授予资格。来源门通过不等于整套L1/L2通过，其他正确性、配对/噪声、校准、边界及工具门仍须分别满足。开发循环与交互式verify.py分别显示数值正确性、紧凑来源状态及performance_eligible；只有数值与当前轨道来源均合格的候选可成为最佳实现，失败不会覆盖此前合格项。开发选择使用可信worker反馈，最终评分仍重算完整来源证据；不把其他最终等级门或隐藏反馈提前交给agent。

这是可信验证器工件的来源判定，不是对整份工件伪造、任意Python/native-memory攻击、输出因果用途或算法有效工作量的安全认证。实际模式无需用重放推断未插桩分支，但也不声称证明任意未观察路径。Neuron及未支持来源适配器不能借用CUDA证据。

## 源码、产物与实际执行

采集器在导入候选前订阅CUPTI callback/activity，同时启用DRIVER与RUNTIME activity以生成correlation ID。候选从带内容哈希的实际Python文件加载；工作进程保留该次`compile()`产生的嵌套代码对象。文件名相同、源码相同但重新编译的另一个对象，不继承身份。Triton可通过`inspect`取得函数源码；只有候选自己声明的future特性生效。

对于Triton 3.7.0，编译产物关联同时要求：

1. `ASTSource`的Python函数代码对象属于本次冻结候选，实际JIT文本与执行候选前保存的函数文本一致；不信任运行时inspect/linecache。
2. 记录实际AST→IR调用与JIT依赖，冻结CUDA后端刚生成的cubin及CompiledKernel初始source/metadata；`_init_handles`后还须保持一致并取得真实CUmodule/CUfunction。
3. CUPTI模块加载回调在生命周期内复制完整二进制字节并取摘要，包括NUL和非UTF-8字节；通过对应成功的driver module-load返回值绑定CUmodule。
4. 成功的`cuModuleGetFunction`把CUfunction绑定到该次模块加载；句柄卸载或复用不会继承上一代身份。
5. 直接launch或Graph节点必须绑定到这个函数代次；设备活动还须匹配成功launch的context、correlation及调用窗口。运行时包装器的correlation只在关联唯一时使用。

内核名称、导入路径、候选自报的`custom`标签或外部origin map不参与上述关联。回调内只读取生命周期有效的数据，并使用官方样例中的CUPTI Graph ID查询；不调用CUDA runtime/driver查询或同步API。当前模块关联适配`cuModuleLoad*`/`cuModuleGetFunction`；没有把未适配的`cuLibrary*`或其他编译器偷偷当作支持。

## 编译输入、依赖与产物

Triton 3.7适配器在真实`ASTSource.make_ir`、解析器和CodeGenerator调用位置记录输入，避免把缓存key或外层函数当作完整来源。每个实际JIT callee均须来自本次提交，或来自执行候选前已固定RECORD/文件SHA的语言primitive；必须观察到匹配的真实AST。普通Python编译期调用只接受明确允许的内建对象与已核对字节码、callable闭包的语言primitive，不凭`__triton_builtin__`或模块名授予身份。未审核helper/自定义IR builder继续未知。候选数值constexpr生成器的通用来源仍未实现。

诊断使用新的私有TRITON_CACHE_DIR，结束后恢复环境；缓存命中、绕过已观察解析器或缺少本次编译记录不能继承证据。另在CUDA后端cubin返回和真实CompiledKernel构造时冻结字节/source/metadata。之后替换`kernel`、source或metadata，IR override、编译stage hook或装载字节不符均不给自写信用。metadata按结构处理真实GPUTarget dataclass，避免把repr或不可序列化对象当摘要。CPU测试实际执行Triton前端到MLIR及真实构造器，未执行ptxas、nvcc、GPU编译后端或kernel。

Torch 2.12.0的`load_inline`/`load`有可选编译观察器。只有实际loader调用点的代码对象属于冻结提交、CUDA文本对应提交中的完整字符串字面量、源依赖与编译选项处于当前审核范围，才建立候选构建记录。观察真实Ninja文件、CUDA source→object映射及完成的构建；固定nvcc/C++/Ninja工具和编译环境摘要，前后核对source、规则及工具。Torch编译适配器自身源码须匹配安装包RECORD。动态文本生成、外部实现header、编译wrapper、设备链接或未支持选项保留未知，不改变原生loader行为。

提取对象是完成构建的CUDA object，而不是将最终.so里的所有cubin归给候选。每个object仍须通过有界CUDA ELF提取与身份核对，且CUPTI实际模块字节相同、CUfunction及launch已完成关联，才可建立来源。仅有load_inline、.so路径、声明的kernel或编译成功不够；尚未关联的CUDA Runtime注册/隐式loader不自动通过。注入的编译器或提取runner标为模拟，不能建立实测来源。

这些是受信诊断worker中的编译输入/产物凭据，不是任意Python/编译器/native-memory攻击的安全认证，也不证明输出因果依赖或算法有效工作量。

## 可信库产物

候选导入前，读取明确允许的安装包RECORD：Torch为framework，cuBLAS/cuDNN/cuFFT/cuSolver/cuSparse/cuRand/NCCL/nvJitLink的已知CUDA wheel名称为library。清单固定版本、RECORD哈希、逐文件SHA-256和大小；候选提供的包名或路径不能扩展清单。

诊断结束后只处理实际映射的DSO。其字节须与冻结RECORD一致，再由预先固定路径和二进制哈希的`cuobjdump --extract-elf all`提取cubin。前后重查DSO和工具哈希，限制尝试时间、记录数、文件与日志大小；保留原始日志的base64与摘要。只有普通CUDA ELF（EM_CUDA=190）的字节摘要，才可与CUPTI实际模块字节匹配。名称、文件所在目录和单独加载过某个库都不足以证明执行来源。

未完成提取、特殊文件、缺工具、错绑、模拟runner及冲突所有权不建立来源。相同设备二进制若同时有已验证的库身份和候选root身份，优先保守地记为非自写；多个受信来源冲突则明确保持未知，不回退到custom。这个清单证明的是当前受信安装镜像中的文件完整性，不是供应商签名认证，也不证明完整的Python/JIT依赖闭包。

## CUDA Graph快照、核参数更新与内存操作

通过真实创建API、CUPTI `GRAPHNODE_CREATED`、`GRAPHNODE_CLONED`与成功instantiate保留节点到函数的关联。每个graph launch保存独立快照，之后的修改不改变旧测量。源模板销毁不抹除已经实例化的快照；可执行graph销毁、未适配参数更新或未知节点会使后续来源保持未知。克隆必须继承已观察到的旧节点代次；设备activity同时匹配本次graph、node、context和launch。

直接memcpy/memset必须匹配同一调用内成功的原生driver/runtime API；Graph内存活动必须匹配实际节点类型和成功launch。满足这些条件记为runtime，进入总设备时间分母，不计入自写内核时间。无法关联的内存活动、后台线程活动、设备发起的child graph和不支持的更新路径继续保持未知。

### 核参数更新的范围

现支持`cuGraphKernelNodeSetParams`、`cuGraphExecKernelNodeSetParams`及二者的`_v2`版本，另识别`cuGraphAddKernelNode_v2`的共同参数前缀。模板setter创建新模板节点代次，既有可执行实例保持原快照；executable setter只改变该实例后续提交的快照，同时处理从原节点继承的clone别名，不修改模板、相邻实例或已提交调用。这遵循[CUDA Graph API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html)的更新语义。

关联须同时满足：进入API时复制的非空CUfunction能对应已观察的函数代次；CUPTI `GRAPH_NODE_UPDATED`或`GRAPH_NODE_SET_PARAMS`事件的节点、executable和context匹配本次API；退出API返回成功。v2的非空`func`沿用v1前缀；`func=NULL`时的CUkernel/ctx路径仍未知，不凭kernel名称猜测。事件字段依据[CUPTI 13.3 Python绑定](https://docs.nvidia.com/cupti-python/13.3.0/api-reference/topics/bindings.html)，布局与已安装供应商`cuda.h`/`generated_cuda_meta.h`由仅主机C程序核对，未链接或调用CUDA。

v4原始收据新增`graph_updates`：保留API/context/correlation、目标节点、函数、资源事件、前后实例代次、替换关系及调用边界。复核要求后续snapshot实际使用最新替换节点，并核对clone祖先、代次和实例，不允许复制旧节点、借用其他实例更新或将未来更新套到旧调用。重放工作量有显式包络，超额拒绝，不跳过节点取得通过。

成功却缺匹配事件、事件错绑，或失败API伴随更新事件时，保留失效边界，后续旧实例来源未知；失败且没有更新事件时保留原快照。`cuGraphExecUpdate`整体更新、其他节点setter、设备发起Graph以及尚未关联的CUkernel继续使后续来源未知。当前只有CPU夹具、序列化篡改反例及主机ABI核对，没有执行CUPTI、GPU、nvcc或设备实验；这些更新记录仍不授予正式轨道资格。

## 独立重放的调用与轨道诊断

诊断已替换原先三个公开seed的独立前向循环，直接复用原生`evaluate`或`paired_timing`入口。每个请求在同一总超时内使用两个新Python进程：一个不启用CUPTI的对照，一个来源采集进程。两者分别使用私有构建和Triton缓存，复用同一组预先冻结的计时输入seed；不改变原始精度、输入族、参数或数值契约。

原生钩子记录首次/普通正确性检查的两种调用顺序、全部预热与计时输出、实际显存探针，以及配对计时的全部预热和正式调用。显式隐藏请求走真实隐藏入口，保留原来的多seed和附加输入族。每次比较完整逻辑输入/输出字节摘要、shape/stride/dtype、存储偏移/别名、模式与实际seed/顺序；摘要在计时区间外分块读取，单块最多1MiB。原生正确性收据、冻结seed计划与CUPTI窗口逐一联结，缺失、额外、换绑或未经验证的调用都不能通过覆盖复核。

采集器只保留与候选窗口重叠的活动，仍保留重叠的未知后台工作；所有观察到的context/stream都参与丢记录核对。窗口发布和回调过滤使用同一把锁，防止过滤尚未发布窗口的活动。离线关联通过时间索引匹配重叠，不再对每个窗口扫描全部活动；重叠展开过多时明确拒绝。可信输出消费、摘要拷贝和参考调用不计为候选自写工作。

两阶段的`native_result`和墙钟耗时仅供诊断，不能写回正式计时、基线或评分。对照阶段也带有调用观察与额外同步，因此逐次字节一致不证明未观察的原生进程采用相同执行路径。CUDA原生峰值显存前向现已验证并作为独立memory窗口纳入：核对输入、输出、原始allocator峰值及验证副本分配增量，FP64收据也有独立memory类别。其他独立诊断仍未纳入。该独立重放不授予来源门资格。

保存的计分结果可通过`--scored-result`关联至后续诊断：它接受单个原生结果，或按题目/样本选择`eval_results.json`、隐藏结果、`paired_results.json`中的结果。请求、源码、设备、seed、原生时间样本和逐次输出均重放核对，随后对照/采集必须与实际观察一致。原生收据缺失、错配或设备身份未知会拒绝，不用新抽取的seed补造实际观察；参考角色目前不接受候选角色的实际收据。最终重复的非默认请求应通过保存的完整请求显式关联，不能借用另一重复或种子。

占比使用同一设备、调用窗口内GPU activity的时间区间并集。自写时间只计入没有与任何非自写活动重叠的部分；并发stream不能重复累加。分别报告custom/library/framework/runtime/unknown的区间并集，这些分类小计之间可以重叠，不能直接相加成总busy时间。每次调用分别检查，不能用一个长调用掩盖其他调用。

`track_verdicts.custom`要求来源完整且每次保守自写占比达到阈值；`track_verdicts.library`允许已建立来源的库、框架和运行时工作，不要求自写占比。库优化不是错误答案，custom门失败也不修改原生正确性。缺活动、丢失记录、未平衡回调、source/hook/模式变化、SDK错误或任一未知活动使相关轨道保持`not_measured`。诊断不能建立正式轨道资格；`formal_qualification_implemented`仍为false。

原始模块/函数/Graph/launch/activity、库提取记录及摘要一并保存。离线`summarize()`重算关联和区间，拒绝重复、冲突或失效代次。CUDA来源协议为`kernelbench_pro_cuda_origin_v4`；原生收据升级为`kernelbench_pro_native_origin_v2`，增加逐次状态与入口关联，并继续要求training后置观察、Triton实际前端/产物凭据或完整CUDA object编译凭据。原生v1缺少状态证据，不能重新封装成v2；旧CUDA v1/v2也不能冒充当前CUDA v4。配置默认0.95是论文声明的来源政策，不是实测校准结论；只适用于新建且绑定该政策的实际计分模式，历史运行不会追认为合格。

以上不能证明输出因果依赖、算法有效工作量、任意动态编译器的依赖闭包或未插桩时执行相同路径，也不提供Python/驱动安全认证。R-08继续标为部分实现。

## 保存报告的离线时间关联

新报告同时保存一次完整请求、两阶段逐调用原生墙钟样本和内容摘要。收集与离线分析共用 `origin_task.replay()`：重新核对当前参考/评测器、请求与精度、设备、完整调用计划、输入/输出/布局/模式摘要，以及逐一对应的CUPTI窗口；原生稳态样本、首个预热时间、预热总和或配对候选时间必须匹配观察记录。非模拟报告还须保留物理设备UUID、驱动与软件身份。随后重算来源汇总，保存的passed标记或重封的总摘要不能代替这些关联。内容摘要用于可信验证器工件的一致性检查，不是对伪造整份工件的密码学认证。

已有报告可以离线分析，不启动采集器或设备：

```bash
python -m scripts.analyze_device_trace native /path/to/origin-result.json \
  --output /path/to/new-native-timeline.json
```

入口只读取既有普通文件（最多64MiB），拒绝覆盖输出；缺新请求/计时记录的旧报告须保持未关联。模拟报告不能产生设备测量。每次原生调用保留其seed、阶段、顺序、摘要、对照/采集两份原生时间、来源并集、未知活动数和保守custom占比。正确性或显存检查没有原生计时样本时对应字段为null，不借用其他调用时间。

`O`只使用同一CUPTI时钟下的候选窗口长度除以设备busy并集。窗口包括前向与完成屏障，排除输入/输出摘要及参考调用；它与包含输出消费和其他主机开销的原生计时区间分别命名、分别保存，不混用跨进程的分子分母。按阶段汇总为总窗口长度/总busy，而非逐调用比值的平均；阶段间隙不属于窗口总和。某次缺活动时该次busy、O及完整阶段总量保持null，同时保留已观察量，不能将缺失当零成本。重叠的未知工作仍进入分母；来源未建立时来源轨道继续未知。

原生ncu采集现复用相同完整调用序列，以唯一NVTX窗口、实际输出及来源kernel数量逐次关联；提供完整审核、同设备校准与逐几何算法证书后，离线分析可重算A/E/AE，详见[工作计数器](WORK_COUNTERS.md)。未请求或证据缺项时保持null，不借用其他seed、状态、基线或形状的模型。所有结果仍是诊断，不修改正式时间、评分或L1；其他独立诊断与Neuron来源继续跟踪。最终隐藏报告限可信分析使用，不作为开发反馈。

## 使用入口

已有软件环境需要额外的`cupti-python==13.3.0`和`triton==3.7.0`，可选依赖为`kernelbench-pro[origin]`；CUDA C++观察器要求Torch 2.12.0和正常的nvcc/C++/Ninja工具链，库/object匹配还需cuobjdump。适配器要求64位Linux，供应商运行库和驱动仍须兼容；缺包、ABI或运行时不兼容不会自动换成名称推断。普通配置预览、模块导入和current/L0批量验证不加载CUPTI；L1/L2请求显式需要上述采集能力。Docker/Modal/queue worker需事先包含可选工具；受限Docker策略仍由原有transport执行。

下面命令默认只预览，读取已保存候选并检查任务/样本范围，不申请设备、不创建结果目录：

```bash
python -m scripts.audit_kernel_origin --config configs/a6000-feedback.toml \
  --problem 2 --sample 0 --output /tmp/origin-diagnostics/result.json
```

使用已有L1/L2计分调用观察时，在同一配置下追加`--scored-result /path/to/eval_results.json`；默认仍只预览并验证既有文件，不启动设备。旧结果没有调用收据时必须明确失败，不能事后重放并冒充原进程。

只有在实验已授权且环境可用时才追加`--execute`。`--minimum-custom-fraction 0.95`可调整诊断阈值；旧`--seeds`简化循环已移除。默认复核公开原生评测；`--hidden`只用于可信最终隐藏请求；`--task paired_timing --baseline-variant eager`复核显式声明的配对基线变体；`--role reference`观察该真实基线调用，应填写实际选中的来源绑定变体。结果路径必须新建，诊断worker日志与结果放在独立目录；失败结果保留已完成阶段及有限日志前缀，不写回原运行的分数、配置或轨迹。

当前实现工作没有执行设备诊断、cuobjdump、CUPTI、容器或模型实验。CPU/模拟回归覆盖真实Triton源码检查、已安装包RECORD静态读取、C ABI布局、句柄复用、Graph克隆/销毁/更新失效、原生内存活动、库产物错绑、非UTF-8日志、并发并集、模式与容差修改、缺工具/丢记录和默认零设备预览。

代码入口：[冻结源码](../kb_workflows/provenance/submission_source.py)、[CUDA采集器](../kb_workflows/provenance/cuda_origin.py)、[库产物清单](../kb_workflows/provenance/origin_binaries.py)、[Triton编译来源](../kb_workflows/provenance/triton_origin.py)、[CUDA C++观察器](../kb_workflows/provenance/cpp_origin.py)、[Graph关联](../kb_workflows/provenance/origin_graphs.py)、[离线重放](../kb_workflows/provenance/kernel_origin.py)、[诊断worker](../kb_workflows/provenance/origin_task.py)、[CLI](../scripts/audit_kernel_origin.py)。

接口依据为[NVIDIA CUPTI Python 13.3绑定](https://docs.nvidia.com/cupti-python/13.3.0/api-reference/topics/bindings.html)、[CUPTI回调与Graph说明](https://docs.nvidia.com/cupti/main/main.html)、[cuobjdump](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#cuobjdump)、[Triton 3.7编译器](https://github.com/triton-lang/triton/blob/v3.7.0/python/triton/compiler/compiler.py)、[JIT源码](https://github.com/triton-lang/triton/blob/v3.7.0/python/triton/runtime/jit.py)及[PyTorch 2.12扩展加载器](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/utils/cpp_extension.py)。CUPTI官方wheel类型声明、已安装供应商头文件与Graph样例已静态核对；未导入或执行CUPTI二进制。
