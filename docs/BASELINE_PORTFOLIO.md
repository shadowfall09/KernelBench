# 扩展基线组合

标准组合仍为eager、compile default、compile max-autotune、CUDA Graph四项。扩展组合通过配置选择，预览不执行实验：

```toml
[run]
baseline_suite = true
baseline_portfolio = "extended"
```

```bash
python -m kb_workflows.cli --baseline-portfolio extended --dry-run --start 1 --end 100
```

预览列出每题声明的变体及总数。扩展项不是已测得更快的实现；实际运行时，每项在独立任务进程中完成初始输出及全部计时输出验证。错误、不支持、编译失败和超时保留，不能参与最快有效基线选择。标准组合不自动增加这些任务。

## 来源绑定的声明清单

`baseline_rewrites.json`绑定原250题与生产41题的源码SHA。全部任务拥有六种扩展基础组合：eager及其Graph、compile default及其Graph、compile max-autotune及其Graph。max-autotune外层Graph使用`max-autotune-no-cudagraphs`以避免嵌套捕获。

当前清单另外声明11个数学／库改写和491个适用后端策略，各有普通与Graph版本。两类共有1,004项额外声明，与六个基础组合合计2,750个分题变体；这不是2,750次已执行实验。清单按当前源码中的算子家族保守选择；没有额外改写的题目仍使用基础组合，不用同义包装冒充独立算法。

| 范围 | 替代实现 |
| --- | --- |
| L1/12 | 对角矩阵乘改为行缩放；限定名义有限值域，非有限语义差异由实际输出验证拒绝 |
| L1/36、38、39 | 保留channel RMSNorm、mean-absolute L1和L2归一化；保留原eps与mean/sum含义 |
| L1/37、40 | 全张量vector norm，以及保留原LayerNorm仿射参数/eps的var_mean形式 |
| 生产1、30 | SDPA布尔mask及GQA形式，保留原FP32／FP64累加语义和输出dtype |
| 生产3 | functional LayerNorm及var_mean两种形式，保留affine、eps及输入角色 |
| 生产27 | 残差加法后的functional RMSNorm，保留原累加与输出格式 |
| 来源中包含卷积家族 | cuDNN benchmark；2D/3D布局候选，以及布局与benchmark的组合 |
| 来源中包含GEMM／Linear等家族 | 通过PyTorch请求cuBLASLt偏好，不改变TF32、缩减精度或确定性政策 |

SDPA的bool mask以True表示参与注意力；GQA保持头数耦合，dropout明确为0，与这两道参考一致。[PyTorch 2.12 SDPA文档](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html)说明这些接口。var_mean使用`correction=0`，不能误用默认Bessel修正。[方差接口](https://docs.pytorch.org/docs/2.12/generated/torch.var_mean.html)定义了该参数。

## 计时、状态和边界

所有改写遵守同一输出与精度合同，不能通过改输入尺度、eps、mask、输入dtype或容差取得资格。保留公开名义工厂、输入校准、原生RNG重放、train/eval和逐输出验证。数学改写不是对所有IEEE非有限输入的等价定理；实际违反原生比较或已配置独立数值门时，该变体失败。

布局转换不改变逻辑shape和dtype。模型权重格式转换计入setup；每次输入的channels-last转换、后端偏好进入/恢复及forward都在调用流水线中计时。Graph版本捕获相同转换与forward，输入复制和稳定输出复制仍包含在重放成本内。参考中不兼容某布局的view、动态scalar或不支持捕获的操作可以让该变体失败；不修改参考或静默换回另一变体。

cuDNN benchmark和BLAS偏好按调用作用域设置并在异常后恢复，不污染其他变体或无插桩参考。Graph中的策略在预热／捕获时生效，重放执行已捕获操作。记录的是此流水线的墙钟时间，不能冒充去掉转换／设置成本的裸算子耗时。编译、预热与格式转换仍计入冷成本，按声明K选择有效摊销基线。

cuBLASLt是请求的偏好，PyTorch仍可能对不支持的算子选用其他库；该接口也标为experimental。[PyTorch后端文档](https://docs.pytorch.org/docs/2.12/backends.html#torch.backends.cuda.preferred_blas_library)明确这项边界。声明、函数名称或通过数值检查不证明实际GPU库路径；执行来源仍归R-08/R-16。没有运行cuDNN、cuBLASLt、CUDA Graph或模型设备实验。

## 结果、恢复与多后端

额外变体保存`baseline_plan`：协议、题目、参考SHA、变体、数学配方、名义域、后端策略与Graph标志。原生worker从当前参考清单重建，不接受候选自带实现。正式基线、配对重测、恢复和评分核对清单及参考身份；漏政策、换配方或把别题的收据移来不能通过。未通过的变体保留原时间和诊断，设为无基线资格。

“最强基线”指当前声明并通过验证的组合中，按稳态或K摊销政策选择的基线，不是全库全算法最优性证明。软件、源文件、政策或参考变化使旧记录失效，需要独立重测；本次只实现流程，不生成性能结果。

CUDA扩展不套用到Trainium。Neuron继续使用自己的eager／compile及四道来源验证的library候选，不请求cuDNN、cuBLASLt或CUDA Graph。普通原生、最终重测和私有Harbor使用同一源码清单及评测器指纹；[配对计时](PAIRED_TIMING.md)与[Neuron工作流](NEURON.md)说明各自边界。
