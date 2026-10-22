# 设备执行来源诊断

R-08当前支持CUDA/Triton直接launch、受信wheel cubin匹配、不可变CUDA Graph replay以及有原生调用/节点证据的复制与memset。仍是独立进程诊断：CPP等其他编译器、现代CUDA library loader、Graph参数更新后的重新绑定、Neuron来源和正式轨道资格尚需完成。诊断结果不能写入正式`kernel_origin`证据，也不能替换原生墙钟、冷启动、配对或最终重复测量。

## 源码、产物与实际执行

采集器在导入候选前订阅CUPTI callback/activity，同时启用DRIVER与RUNTIME activity以生成correlation ID。候选从带内容哈希的实际Python文件加载；工作进程保留该次`compile()`产生的嵌套代码对象。文件名相同、源码相同但重新编译的另一个对象，不继承身份。Triton可通过`inspect`取得函数源码；只有候选自己声明的future特性生效。

对于Triton 3.7.0，编译产物关联同时要求：

1. `ASTSource`的Python函数代码对象属于本次冻结候选，实际JIT文本与该文件中的函数一致。
2. `CompiledKernel._init_handles`完成后取得真实CUmodule、CUfunction和编译器输出cubin摘要。
3. CUPTI模块加载回调在生命周期内复制完整二进制字节并取摘要，包括NUL和非UTF-8字节；通过对应成功的driver module-load返回值绑定CUmodule。
4. 成功的`cuModuleGetFunction`把CUfunction绑定到该次模块加载；句柄卸载或复用不会继承上一代身份。
5. 直接launch或Graph节点必须绑定到这个函数代次；设备活动还须匹配成功launch的context、correlation及调用窗口。运行时包装器的correlation只在关联唯一时使用。

内核名称、导入路径、候选自报的`custom`标签或外部origin map不参与上述关联。回调内只读取生命周期有效的数据，并使用官方样例中的CUPTI Graph ID查询；不调用CUDA runtime/driver查询或同步API。当前模块关联适配`cuModuleLoad*`/`cuModuleGetFunction`；没有把未适配的`cuLibrary*`或其他编译器偷偷当作支持。

## 可信库产物

候选导入前，读取明确允许的安装包RECORD：Torch为framework，cuBLAS/cuDNN/cuFFT/cuSolver/cuSparse/cuRand/NCCL/nvJitLink的已知CUDA wheel名称为library。清单固定版本、RECORD哈希、逐文件SHA-256和大小；候选提供的包名或路径不能扩展清单。

诊断结束后只处理实际映射的DSO。其字节须与冻结RECORD一致，再由预先固定路径和二进制哈希的`cuobjdump --extract-elf all`提取cubin。前后重查DSO和工具哈希，限制尝试时间、记录数、文件与日志大小；保留原始日志的base64与摘要。只有普通CUDA ELF（EM_CUDA=190）的字节摘要，才可与CUPTI实际模块字节匹配。名称、文件所在目录和单独加载过某个库都不足以证明执行来源。

未完成提取、特殊文件、缺工具、错绑、模拟runner及冲突所有权不建立来源。相同设备二进制若同时有已验证的库身份和候选root身份，优先保守地记为非自写；多个受信来源冲突则明确保持未知，不回退到custom。这个清单证明的是当前受信安装镜像中的文件完整性，不是供应商签名认证，也不证明完整的Python/JIT依赖闭包。

## CUDA Graph与原生内存操作

通过真实创建API、CUPTI `GRAPHNODE_CREATED`、`GRAPHNODE_CLONED`与成功instantiate保留节点到函数的关联。每个graph launch保存独立快照，之后的修改不改变旧测量。源模板销毁不抹除已经实例化的快照；可执行graph销毁、未适配参数更新或未知节点会使后续来源保持未知。克隆必须继承已观察到的旧节点代次；设备activity同时匹配本次graph、node、context和launch。

直接memcpy/memset必须匹配同一调用内成功的原生driver/runtime API；Graph内存活动必须匹配实际节点类型和成功launch。满足这些条件记为runtime，进入总设备时间分母，不计入自写内核时间。无法关联的内存活动、后台线程活动、设备发起的child graph和不支持的更新路径继续保持未知。

## 调用、模式与轨道诊断

默认使用公开输入构造器的三个seed（42、43、44），保留实际shape/stride/dtype及标量签名、调用前后的training/grad观察、源码/请求/评测器摘要和设备身份。参考与候选使用独立输入副本；每次候选执行同步结束，并按候选导入前冻结的配置容差通过原生结构/数值比较。导入/执行期间的已知进程设置变化会被记录并恢复，变化使诊断保持未知。这个范围不声称覆盖全部原生、压力或隐藏用例，也不证明完整参数/状态契约。

占比使用同一设备、调用窗口内GPU activity的时间区间并集。自写时间只计入没有与任何非自写活动重叠的部分；并发stream不能重复累加。分别报告custom/library/framework/runtime/unknown的区间并集，这些分类小计之间可以重叠，不能直接相加成总busy时间。每次调用分别检查，不能用一个长调用掩盖其他调用。

`track_verdicts.custom`要求来源完整且每次保守自写占比达到阈值；`track_verdicts.library`允许已建立来源的库、框架和运行时工作，不要求自写占比。库优化不是错误答案，custom门失败也不修改原生正确性。缺活动、丢失记录、未平衡回调、source/hook/模式变化、SDK错误或任一未知活动使相关轨道保持`not_measured`。诊断不能建立正式轨道资格；`formal_qualification_implemented`仍为false。

原始模块/函数/Graph/launch/activity、库提取记录及摘要一并保存。离线`summarize()`重算关联和区间，拒绝重复、冲突或失效代次。来源协议升级为`kernelbench_pro_cuda_origin_v2`，要求实际training后置观察；旧v1诊断不能冒充当前收据。默认0.95仅是论文设计中的可配置诊断阈值；历史20%与95%未做设备校准，不设为所有旧运行的默认资格门槛。

以上不能证明输出因果依赖、算法有效工作量、完整依赖闭包或未插桩时执行相同路径，也不提供Python/驱动安全认证。R-08继续标为部分实现。

## 使用入口

已有软件环境需要额外的`cupti-python==13.3.0`和`triton==3.7.0`，可选依赖为`kernelbench-pro[origin]`；库匹配还需当前CUDA工具链的cuobjdump。适配器要求64位Linux，供应商运行库和驱动仍须兼容；缺包、ABI或运行时不兼容不会自动换成名称推断。普通配置预览、模块导入、批量验证均不加载CUPTI。Docker/Modal/queue worker需事先包含可选工具；受限Docker策略仍由原有transport执行。

下面命令默认只预览，读取已保存候选并检查任务/样本范围，不申请设备、不创建结果目录：

```bash
python -m scripts.audit_kernel_origin --config configs/a6000-feedback.toml \
  --problem 2 --sample 0 --output /tmp/origin-diagnostics/result.json
```

只有在实验已授权且环境可用时才追加`--execute`。`--seeds 42,43,44`和`--minimum-custom-fraction 0.95`可调整设置。结果路径必须新建，诊断worker日志放在独立目录；不写回原运行的分数、配置或轨迹。

当前实现工作没有执行设备诊断、cuobjdump、CUPTI、容器或模型实验。CPU/模拟回归覆盖真实Triton源码检查、已安装包RECORD静态读取、C ABI布局、句柄复用、Graph克隆/销毁/更新失效、原生内存活动、库产物错绑、非UTF-8日志、并发并集、模式与容差修改、缺工具/丢记录和默认零设备预览。

代码入口：[冻结源码](../kb_workflows/submission_source.py)、[CUDA采集器](../kb_workflows/cuda_origin.py)、[库产物清单](../kb_workflows/origin_binaries.py)、[Graph关联](../kb_workflows/origin_graphs.py)、[离线重放](../kb_workflows/kernel_origin.py)、[诊断worker](../kb_workflows/origin_task.py)、[CLI](../scripts/audit_kernel_origin.py)。

接口依据为[NVIDIA CUPTI Python 13.3绑定](https://docs.nvidia.com/cupti-python/13.3.0/api-reference/topics/bindings.html)、[CUPTI回调与Graph说明](https://docs.nvidia.com/cupti/main/main.html)、[cuobjdump](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#cuobjdump)、[Triton 3.7编译器](https://github.com/triton-lang/triton/blob/v3.7.0/python/triton/compiler/compiler.py)及[JIT源码](https://github.com/triton-lang/triton/blob/v3.7.0/python/triton/runtime/jit.py)。CUPTI官方wheel类型声明、已安装供应商头文件与Graph样例已静态核对；未导入或执行CUPTI二进制。
