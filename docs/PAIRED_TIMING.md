# 交替计时与独立最终重测

R-11/12已接入同一批量入口，R-13已加入组合扩展但仍需继续补题目级框架改写。它们不缩小其余23项设计的范围；真实CUDA、设备校准、归因和Trainium等仍继续实现与验证。

## 配置

在既有GPU配置的`[run]`增加：

```toml
baseline_portfolio = "extended"
paired_timing = true
paired_trials = 30
final_repeats = 2
```

也可使用`--baseline-portfolio extended --paired-timing --paired-trials 30 --final-repeats 2`。默认仍为四变体基线、不启用额外交替计时和重复；`final_repeats`非负，配对次数至少10。交替计时要求组合基线，不使用旧单基线模式。恢复、CLI/TOML、Docker/Modal/队列与Harbor原生验证器共用参数。

开发反馈仍只用公开实例。agent提交后执行的独立隐藏重测不会返回给搜索回路。更严格数值契约可另外配置，不能仅因打开计时选项就获得L0/L1/L2等级。

## 组合基线

`standard`保留eager、compile default、compile max-autotune和eager CUDA Graph。`extended`增加两个compile变体各自的外层Graph。max-autotune的外层捕获使用`max-autotune-no-cudagraphs`，避免Inductor自己持有另一层自动Graph。输入复制及输出快照仍在重放计时内；相关CUDA Graph约束依据[PyTorch 2.12文档](https://docs.pytorch.org/docs/2.12/notes/cuda.html#cuda-graphs)。

L1 #36/#38/#39另有源码SHA-256约束的框架改写及其Graph版本。RMSNorm保留eps与通道语义；#38参考是除以`mean(abs(x))`，不是L1总和，因此使用F.normalize后还原通道数因子；L2Norm保留零值时的NaN行为。这些变体仍逐次核对输出，失败变体保留且不作为赢家。不是把未经核对的F函数统一套到所有题目上。

## 平衡配对与噪声

在一个新的评测进程中加载当前摊销最优基线、候选和独立的同变体参考控制。每轮生成一份输入、分别复制给三者，重放RNG并恢复注册状态；六种调用顺序平衡出现。每个计时包含输出消费与设备同步，输入准备、状态重置和完整数值检查在区间外。错误输出与候选设置修改使测量失败。

记录每轮参考、候选、控制耗时以及种子和顺序。对`log(T_ref/T_candidate)`求均值并指数化，通过配对bootstrap得到95%区间；参考/控制比同样估计噪声。只有加速区间下界大于`max(1, 噪声区间上界)`时，才允许正性能分。多个独立重测进程时，先重采样进程、再重采样其中配对，保留进程间方差。

这是一项有明确可交换性假设的近似推断。它不保证消除共享设备漂移；控制样本也不能替代独占设备和时钟试点。计时生成器可能仍返回相同内容，重新调用与记录种子不证明所有值都不同。

## 冻结与新进程

启用`final_repeats`后，所有声明样本都冻结到`final_artifacts/<sha256>.py`。`final_manifest.json`绑定源码、参考、评测版本和配置，并保存各次独立任务的新标准种子、三种隐藏种子与nonce。每个标准/隐藏任务依旧由独立评测进程执行。

评分要求所有重复的标准和隐藏检查均通过，使用所有标准重测均值，不选最快一次；时间区间按进程再按样本重采样。已测冷成本仍是一项点估计，不冒充清空编译缓存后重复测量的冷启动分布。启用交替计时时，正式稳态时间来自相应新进程的配对数据。

resume保留未变样本的种子、nonce和已完成结果，包含已记录失败；源码变化的样本重新冻结和执行，其他样本不因此重花预算。源码、参考、配置、快照或评测器变化会拒绝过期证据。进入`analyze`阶段不改变实验身份。

## 结果和评分门

新增`final_manifest.json`、`final_results.json`、`final_artifacts/`与`paired_results.json`。性能评分核对当前源码、设备类别、实际获胜基线、全部重复结果和原始配对样本；配对汇总会重新计算。旧榜单资格还绑定附加结果文件摘要和冻结源码内容，删掉/修改重测记录不能继续使用旧资格。

独立重测失败计入正确性pass@k的失败槽位；原始阶段诊断仍不等于计分资格。正式分母与样本均值口径不变，资源成本不因追加复测被写成模型生成成本。

CPU回归使用真实Torch输出验证调度、结构、改写和状态逻辑，并用受控时钟验证计时边界。真实GPU计时、Graph捕获、共享设备稳定性及外部沙箱仍需实际设备验收。
