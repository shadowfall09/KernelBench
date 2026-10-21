# Production-41：参数化题族与反向轨道

新增题集为 `--suite production41 --level 5`，包含设计12题、数值设计6题、固定格式实现12题、反向11题。旧 Level 1–3 的250道题及旧 Level 4 的20个模型文件保持原样。新题有独立参考程序、逐角色输入契约和隐藏生成器，进入原生生成、标准/隐藏验证、组合基线、独立复测、评分与 Harbor 流程。

本轮只实现代码并进行CPU/模拟回归，**不运行GPU、Trainium、真实模型或训练实验**。没有提交实测性能、设计可达性结果或已校准的硬件边界。

## 选题与轨道

只查看配置，不启动生成或评测：

```bash
python -m scripts.run_batch --suite production41 --level 5 \
  --task-track backward --task-profile small --dry-run
```

`task_track` 可选 `design`、`numerical`、`implementation`、`backward`、`all`。每个轨道独立声明题目和样本槽位，`paper_analysis.json` 的 `score_by_track` 分开汇总。`all` 用于收集四类任务，其混合宏平均只作诊断，`leaderboard_eligible=false`。反向任务不混入前向榜单。

`small` 用于回归；`standard` 和 `large` 是较大实例配置。题目是参数化家族，隐藏尺寸不是新的独立题目。默认固定格式任务支持FP32/BF16/FP16逐次独立运行；任务19、20、21、22、29及数值设计轨道另有固定存储/激活格式，任务负载会明确覆盖通用精度家族。实际每个输入和输出的dtype以契约与收据为准。

## 输入角色与格式

参考、候选、组合基线、开发profiler和配对计时都在同一个逐角色输入策略中运行。它保留FP32缩放因子、归一化权重、步长/衰减，保持int64标签/路由/页表、uint8打包权重和bool mask；不对所有浮点输入统一转为全局精度。模型也只移动设备，不隐式改变已有buffer的格式。作用域结束后恢复原处理策略，旧题默认行为不变。

公开参考包含完整数学实现、角色顺序、格式、数值预算和反向上游梯度接口。生成prompt移除 `get_inputs`、`get_init_inputs` 及私有生成器依赖；agent不需要猜测题意，但不能从prompt取得实际测试配置。参考文件可独立阅读，生成器和数学函数分别维护于 `production.py`、`production_specs.py`、`production_ops.py`。

格式细节：

- FP8 GEMM使用实际 `torch.float8_e4m3fn` 输入存储、逐行FP32缩放和BF16输出。数学参考显式反量化后计算；这不证明baseline使用原生FP8矩阵指令。
- W4权重每字节按低半字节、再高半字节排列，码值减8得到有符号整数；每组FP32 scale沿K维展开。
- FP4量化融合采用E2M1数值集合，最近值舍入、等距时选偶数幅值编码，按块缩放。它是明确的软件语义任务，不冒充NVFP4/MXFP4原生吞吐测量，也不要求所有设备有原生FP4指令。
- top-k按值降序，同值保留较低原始索引。索引精确比较，避免未定义的tie规则使合法实现无法核对。
- 状态空间任务是带逐头标量衰减的SSD递推，不是完整Mamba模型。长卷积采用直接因果深度卷积参考，允许候选使用FFT等算法。

## 数值门与反向

六道数值设计任务给出逐题RMS和逐元素预算。参考在规定量化步骤之后使用FP64算术，再舍入到声明输出格式。格式选择题允许候选选择内部格式，只要满足该误差契约。精度模拟题13与格式预算题14具有不同预算，不能互换判分条件。CPU回归中，单遍TF32/BF16近似不能通过题13，六项BF16拆分可通过；同一BF16近似可通过较宽的题14。这些小规模回归不代替GPU误差分布实验。

固定格式题也使用与有效激活格式匹配的相对RMS及逐元素门，避免绝对容差掩盖小输出。离散输出仍精确比较，结构/shape/dtype及NaN、正负Inf位置仍必须匹配。比较发生在标准、隐藏、每次计时输出校验和配对计时中，不计入测量区间。全41题均有全零候选反例测试。

这个门比较的是声明参考输出；把该输出转为FP64进行误差计算，**不等于重新取得独立FP64 oracle并完成参考自资格**。后者属于仍在扩展的R-04工作。原有L0覆盖范围没有因为新增题目自动扩大；请求尚未支持的严格等级仍会在预检或评分时拒绝。

反向题的forward接口直接返回声明输入的VJP，最后一个输入是与前向输出同形状的上游梯度。标签、页表、packed权重等不要求梯度。参考在独立的 `torch.enable_grad()` 作用域执行，即使外层计时使用no_grad也能工作；候选可以直接实现反向kernel。11个家族均有FP64 gradcheck及逐输入方向有限差分回归。

## 显存计划和隐藏实例

`task_memory_mb` 默认512。worker在导入候选前读取设备总容量，以 `min(声明字节数, 设备总容量/10)` 为包络，按确定性规则缩小维度。保留packed分组整除、偶数rotary维度、GQA头数比例等约束，记录请求维度、实际维度、角色、估计峰值和 `instance_sha256`。使用总容量让同设备上的基线与候选得到可重放计划；不是根据每次空闲显存悄悄更换形状。

峰值模型包含输入副本、参考、FP64/梯度及主要中间量，但不能保证任意编译器workspace或候选实现不会OOM。无法缩到包络内时拒绝该实例；OOM/超时仍是失败槽位，不替换成更小的未声明题目。

隐藏生成器覆盖不对齐尺寸、非连续stride、合法动态范围、不同mask/长度/分页索引、以及归一化的近常量输入。索引保持合法，mask保证每行有可见元素，概率gate归一化，步长正、衰减负。标准与隐藏共用相同角色约束；隐藏case的输入准备不进入性能计时。

## 基线、证据与生成文件

沿用eager、compile default/max-autotune、CUDA Graph及已有扩展组合；每个变体独立验证，失败保留。逐题优化的库改写和原生低精度基线仍在R-13中扩展，不把当前数学参考视为厂商最强实现。I/D标签只是任务设计分类；正式Headroom评分仍需可信工作量证书和独立可达性见证，已有导出与校验流程可处理Level 5结果。

配置、角色契约、轨道、显存计划参数、生成文件和manifest都参与任务/恢复指纹。Harbor保留 `production41` 和轨道设置，将完整私有生成器留在verifier侧。开发只接收公开验证反馈。

```bash
python -m scripts.build_production_tasks --check
```

此命令核对41个自包含参考、41个隐藏生成器及manifest与维护源一致；修改数学实现或角色契约后应重新生成，再审查源码diff。不得手工改manifest哈希来掩盖参考变化。

数学API依据：[PyTorch autograd.grad](https://docs.pytorch.org/docs/2.12/generated/torch.autograd.grad.html)、[scaled dot product attention](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html)、[cross entropy](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.cross_entropy.html)。

<!-- tasks:start -->
| ID | 任务 | 轨道 |
| --- | --- | --- |
| 1 | [online_attention](../KernelBench/level5/1_online_attention.py) | 设计 |
| 2 | [linear_cross_entropy](../KernelBench/level5/2_linear_cross_entropy.py) | 设计 |
| 3 | [welford_normalization](../KernelBench/level5/3_welford_normalization.py) | 设计 |
| 4 | [partial_topk](../KernelBench/level5/4_partial_topk.py) | 设计 |
| 5 | [chunked_linear_attention](../KernelBench/level5/5_chunked_linear_attention.py) | 设计 |
| 6 | [chunked_state_space](../KernelBench/level5/6_chunked_state_space.py) | 设计 |
| 7 | [persistent_recurrent](../KernelBench/level5/7_persistent_recurrent.py) | 设计 |
| 8 | [fft_long_convolution](../KernelBench/level5/8_fft_long_convolution.py) | 设计 |
| 9 | [grouped_moe](../KernelBench/level5/9_grouped_moe.py) | 设计 |
| 10 | [dequantization_gemm](../KernelBench/level5/10_dequantization_gemm.py) | 设计 |
| 11 | [paged_attention](../KernelBench/level5/11_paged_attention.py) | 设计 |
| 12 | [split_k_gemm](../KernelBench/level5/12_split_k_gemm.py) | 设计 |
| 13 | [accurate_gemm_emulation](../KernelBench/level5/13_accurate_gemm_emulation.py) | 数值设计 |
| 14 | [format_budget_gemm](../KernelBench/level5/14_format_budget_gemm.py) | 数值设计 |
| 15 | [format_budget_attention](../KernelBench/level5/15_format_budget_attention.py) | 数值设计 |
| 16 | [format_budget_moe](../KernelBench/level5/16_format_budget_moe.py) | 数值设计 |
| 17 | [block_fp4_quantization_gemm](../KernelBench/level5/17_block_fp4_quantization_gemm.py) | 数值设计 |
| 18 | [token_fp8_quantization_attention](../KernelBench/level5/18_token_fp8_quantization_attention.py) | 数值设计 |
| 19 | [bf16_gemm](../KernelBench/level5/19_bf16_gemm.py) | 固定格式实现 |
| 20 | [fp16_gemm](../KernelBench/level5/20_fp16_gemm.py) | 固定格式实现 |
| 21 | [fp8_scaled_gemm](../KernelBench/level5/21_fp8_scaled_gemm.py) | 固定格式实现 |
| 22 | [w4a16_gemm](../KernelBench/level5/22_w4a16_gemm.py) | 固定格式实现 |
| 23 | [batched_grouped_gemm](../KernelBench/level5/23_batched_grouped_gemm.py) | 固定格式实现 |
| 24 | [paged_decode](../KernelBench/level5/24_paged_decode.py) | 固定格式实现 |
| 25 | [batched_topk](../KernelBench/level5/25_batched_topk.py) | 固定格式实现 |
| 26 | [swiglu_projection](../KernelBench/level5/26_swiglu_projection.py) | 固定格式实现 |
| 27 | [residual_rmsnorm](../KernelBench/level5/27_residual_rmsnorm.py) | 固定格式实现 |
| 28 | [rotary_embedding](../KernelBench/level5/28_rotary_embedding.py) | 固定格式实现 |
| 29 | [quantized_moe](../KernelBench/level5/29_quantized_moe.py) | 固定格式实现 |
| 30 | [gqa_decode](../KernelBench/level5/30_gqa_decode.py) | 固定格式实现 |
| 31 | [online_attention_backward](../KernelBench/level5/31_online_attention_backward.py) | 反向 |
| 32 | [linear_cross_entropy_backward](../KernelBench/level5/32_linear_cross_entropy_backward.py) | 反向 |
| 33 | [welford_normalization_backward](../KernelBench/level5/33_welford_normalization_backward.py) | 反向 |
| 34 | [chunked_linear_attention_backward](../KernelBench/level5/34_chunked_linear_attention_backward.py) | 反向 |
| 35 | [chunked_state_space_backward](../KernelBench/level5/35_chunked_state_space_backward.py) | 反向 |
| 36 | [persistent_recurrent_backward](../KernelBench/level5/36_persistent_recurrent_backward.py) | 反向 |
| 37 | [fft_long_convolution_backward](../KernelBench/level5/37_fft_long_convolution_backward.py) | 反向 |
| 38 | [grouped_moe_backward](../KernelBench/level5/38_grouped_moe_backward.py) | 反向 |
| 39 | [dequantization_gemm_backward](../KernelBench/level5/39_dequantization_gemm_backward.py) | 反向 |
| 40 | [paged_attention_backward](../KernelBench/level5/40_paged_attention_backward.py) | 反向 |
| 41 | [swiglu_projection_backward](../KernelBench/level5/41_swiglu_projection_backward.py) | 反向 |
<!-- tasks:end -->
