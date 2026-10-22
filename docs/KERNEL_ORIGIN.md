# 设备执行来源诊断

R-08当前交付CUDA/Triton直接launch的来源采集器和离线重放，尚未交付完整库轨道、CUDA Graph和Neuron来源资格判断。所有结果都是独立进程的诊断，不能写入正式`kernel_origin`证据，也不能替换原生墙钟、冷启动、配对或最终重复测量。

## 已实现的关联

采集器在导入候选前订阅CUPTI callback/activity。候选从带内容哈希的实际Python文件加载；工作进程保留该次`compile()`产生的嵌套代码对象。文件名相同、源码相同但重新编译的另一个对象，不继承身份。这也使Triton能够通过`inspect`取得函数源码，并修复了旧加载器在源码前插入环境变量设置、影响`__future__`导入及含引号build路径的问题。

对于Triton 3.7.0，编译产物关联同时要求：

1. `ASTSource`的Python函数代码对象属于本次冻结候选，实际JIT文本与该文件中的函数一致。
2. `CompiledKernel._init_handles`完成后取得真实CUmodule、CUfunction和编译器输出cubin摘要。
3. CUPTI模块加载回调在生命周期内复制完整二进制字节并取摘要，包括NUL和非UTF-8字节；通过对应成功的driver module-load返回值绑定CUmodule。
4. 成功的`cuModuleGetFunction`把CUfunction绑定到该次模块加载；句柄卸载或复用不会继承上一代身份。
5. `cuLaunchKernel`、`cuLaunchKernelEx`及其`_ptsz`版本记录CUfunction、context、correlation和受信调用范围；GPU activity必须通过同一context/correlation关联到成功的launch。

内核名称、导入路径、候选自报的`custom`标签或外部origin map不参与上述关联。回调内不调用CUDA runtime/driver查询或同步API。

## 调用与时间范围

默认在公开输入构造器上取三个seed（42、43、44），保留实际shape/stride/dtype及标量签名、`no_grad`、模型training状态、源码/请求/评测器摘要和设备身份。参考与候选使用独立输入副本；每次候选执行同步结束，并通过原生结构/数值比较后才能进入完整诊断。这个范围不声称覆盖全部原生、压力或隐藏用例，也不证明参数/状态已具备完整任务契约。

占比使用同一设备、调用窗口内GPU activity的时间区间并集。自写时间只计入没有与任何未知活动重叠的部分；并发stream不能重复累加，后台线程的未关联活动不能因没有call标签而被忽略。memcpy/memset、未关联函数、CUDA Graph/node replay、其他编译器和库内核目前保持未知，进入分母，不获得自写信用。每次调用分别检查，不能用一个长调用掩盖其他调用。

`custom_exclusive_fraction_lower_bound`是已建立来源部分的保守占比。缺活动、丢失记录、未平衡回调、source/hook变化、SDK错误或任一未知活动使该调用的结论保持`not_measured`。原始模块/函数/launch/activity记录及摘要一并保存；离线`summarize()`重算关联和区间，拒绝重复或冲突身份。模拟绑定不能建立实测诊断结论。

默认0.95仅是论文设计中的可配置诊断阈值；历史20%与95%未做设备校准，不设为所有旧运行的默认资格门槛。目前未实现库/框架的可信二进制清单，因此混合执行通常保留未知；没有把未知宣告为库轨道或错误答案。阈值规则不能证明输出因果依赖、算法有效工作量、完整依赖闭包来源或未插桩时执行相同路径，也不提供Python/驱动安全认证。

## 使用入口

已有软件环境需要额外的`cupti-python==13.3.0`和`triton==3.7.0`，可选依赖为`kernelbench-pro[origin]`。适配器要求64位Linux，供应商运行库和驱动仍须兼容；缺包、ABI或运行时不兼容不会自动换成名称推断。普通配置预览、模块导入、批量验证均不加载CUPTI。Docker/Modal/queue镜像需事先包含可选工具，受限Docker策略仍由原有transport执行。

下面命令默认只预览，读取已保存候选并检查任务/样本范围，不申请设备、不创建结果目录：

```bash
python -m scripts.audit_kernel_origin --config configs/a6000-feedback.toml \
  --problem 2 --sample 0 --output /tmp/origin-diagnostics/result.json
```

只有在实验已授权且环境可用时才追加`--execute`。`--seeds 42,43,44`和`--minimum-custom-fraction 0.95`可调整诊断设置。结果路径必须新建，诊断worker日志放在独立目录；不写回原运行的分数、配置或轨迹。当前实现工作没有执行该选项、安装CUPTI、调用设备或运行模型；测试的执行分支使用合成dispatch。

代码入口：[冻结源码](../kb_workflows/submission_source.py)、[CUDA采集器](../kb_workflows/cuda_origin.py)、[离线重放](../kb_workflows/kernel_origin.py)、[诊断worker](../kb_workflows/origin_task.py)、[CLI](../scripts/audit_kernel_origin.py)。CPU回归覆盖句柄复用、二进制中的NUL、来源/产物错绑、伪造标签、丢记录、并发重叠、Graph未知、源码/模式变化、特殊文件、真实Triton源码检查与零设备预览。

接口依据为[NVIDIA CUPTI Python 13.3绑定](https://docs.nvidia.com/cupti-python/13.3.0/api-reference/topics/bindings.html)、[CUPTI回调约束](https://docs.nvidia.com/cupti/main/main.html#callback-api)、[Triton 3.7编译器](https://github.com/triton-lang/triton/blob/v3.7.0/python/triton/compiler/compiler.py)及[JIT源码](https://github.com/triton-lang/triton/blob/v3.7.0/python/triton/runtime/jit.py)。CUPTI 13.3官方wheel的类型声明和样例已静态核对，未导入或执行其二进制。
