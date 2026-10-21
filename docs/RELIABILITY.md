# 从相关工作引入的评测可靠性改进

核对日期：2026-10-21。基础版本：`81c7055`。本次按论文和公开源码独立实现适合当前 Verified 工作流的检查，没有整仓迁入其他基准。

## 来源与取舍

| 工作 | 核对材料 | 本次采用 | 暂未采用及原因 |
| --- | --- | --- | --- |
| KernelBench v3 / KernelBench-Hard | 用户提供的 [v3 地址](https://kernelbench.com/v3) 当前重定向到总站。实现依据为 Hard 的 [SPEC](https://github.com/Infatoshi/KernelBench-Hard/blob/77d02f401b40c26226c8f5b753d7da46ff070e0a/SPEC.md)、[数值压力测试](https://github.com/Infatoshi/KernelBench-Hard/blob/77d02f401b40c26226c8f5b753d7da46ff070e0a/src/eval/numeric_stress.py) 和各题 `check.py`。 | 三个种子逐一验证；按题目的合法数值域添加压力输入；压力检查与性能计时分开。 | 未迁入 Hard 题集、SOTA 依赖或硬件峰值评分。其特定精度容差不能直接替换当前题目的容差；本次保留既有 dtype 容差。 |
| KernelBenchX | [论文 §3.2.2](https://arxiv.org/html/2605.04956v2#S3.SS2.SSS2)、[输入生成器](https://github.com/BonnieW05/KernelBenchX/blob/fd4192293bf9a8c645327a9d46aa1e807f1f9cf2/utils/data_utils.py)、[README](https://github.com/BonnieW05/KernelBenchX/blob/fd4192293bf9a8c645327a9d46aa1e807f1f9cf2/README.md)。 | 稀疏离群值：约 0.1% 位置替换为标准差 50 的高斯值；编译、数值正确性、有效计时分开报告。 | 未迁入 176 道任务或直接套用其分数。TFLOPS、带宽需要逐题可信计算量和字节量定义，当前不凭输入大小猜测。 |
| robust-kbench | [论文](https://arxiv.org/abs/2509.14279)、[前向验证源码](https://github.com/SakanaAI/robust-kbench/blob/078f5bab29934a822268d59a4e707d449abf9b4e/robust_kbench/sandbox/correct_forward_fn.py)、[说明](https://github.com/SakanaAI/robust-kbench/blob/078f5bab29934a822268d59a4e707d449abf9b4e/README.md)。 | 在参考计算前后都运行候选，检查依赖参考结果缓存和重复调用不一致；每次调用隔离输入、保存输出快照。 | 原源码采用 C→R→C；这里采用 C→R、R→C 两组配对，让有状态模型的调用次数也一致。未迁入 backward 任务、权重初始化网格或任务过滤器。 |
| TritonRL | [论文 §2，Fine-Grained Verification](https://arxiv.org/html/2510.17891v1#S2)。论文区分语法、功能合法性、编译、数值正确性和性能，并使用规则与模型裁判。 | 采用分阶段诊断的原则，明确现有检查只覆盖哪些证据。 | 没有接入模型裁判、训练奖励或“出现 Triton 语法即合法”的规则。本次不宣称已证明计算没有转交高层库。未找到可以核实并直接迁入的官方验证器源码，依据限于论文。 |

源码核对版本分别固定为 Hard `77d02f401b40c26226c8f5b753d7da46ff070e0a`、KernelBenchX `fd4192293bf9a8c645327a9d46aa1e807f1f9cf2`、robust-kbench `078f5bab29934a822268d59a4e707d449abf9b4e`。本次是方法借鉴与独立适配，不是这些项目对当前实现的背书。

## 实际行为

标准与隐藏正确性检查都执行两组调用：

1. 候选 → 参考：候选先得到独立输入，执行并保存输出，随后才计算参考答案。
2. 参考 → 候选：再次配对计算，检查第二次候选调用。

每组内重放相同随机种子，两边分别接收输入副本；调用后同步，再保存输出快照。两个模型各前进两次，支持状态按调用次数变化的情况。候选和参考的源码在不同全局字典加载，避免候选同名变量或函数覆盖参考定义。每次正确性 forward 后恢复共享后端设置，避免候选改变 TF32 等设置后污染参考计算。

隐藏输入生成器分别在种子 **42、123、456** 下重新调用；每个种子的所有配置都必须验证。生成器本身若硬编码数据，三个种子仍可能得到同样输入，不能据此声称内容一定不同。

对以下 **29 道 Level 1 题目**，每个种子的第一个隐藏配置再增加三组输入：input 0 全零、input 0 乘 32、input 0 注入稀疏离群值。其他参数、shape、stride 和 dtype 保持不变。小 Tensor 如果没有随机选中任何位置，强制注入一个值为 50 的离群值，保证该测试实际发生。

| 题号 | 已核对的输入约束 |
| --- | --- |
| 1–12、16–18 | 普通矩阵、批量矩阵、转置矩阵、向量或标量乘法；input 0 接受任意有限实数。12 的 input 0 是表示对角线的一维向量，仍保持该表示。 |
| 19–27 | ReLU、LeakyReLU、Sigmoid、Tanh、Softmax、LogSoftmax、Swish、GELU、SELU 的实数输入。 |
| 47–49、51–52 | sum、mean、max、argmax、argmin 的实数输入。全零时索引结果仍须与参考的 tie 规则精确一致。 |

对称/三角矩阵、概率、索引、mask 等限制没有经过逐题适配，不自动增加上述扰动。Level 2/3 本次只增加调用顺序与多种子检查。允许列表只针对仓库可信 `hidden_tests/level1/` 路径生效，不依据外部文件名猜测题目。维护者修改题目语义后须重新核对列表。

原隐藏集为四配置时，无压力适配的题目执行 **12 个配置检查**，适配题目执行 **21 个配置检查**；每个配置包含两组调用。额外成本只发生在正确性阶段，性能仍使用原可信输入生成器和已有同步墙钟协议。标准阶段的首次候选调用现在发生在首次参考 forward 前，冷成本仍按既有规则记录。

隐藏集加载失败、返回空集或配置格式错误会显式中止验证；参考 forward 报错记录 `reference_error` 并使检查失败；标准正确性次数为零不能通过。这些情况不能算作候选已经验证正确，错误来源仍可从元数据区分。

## 输出与恢复

`eval_results*.json` 的 metadata 增加：

- `reliability_protocol=paired_order_multiseed_stress_v1`。
- `reliability_checks`：每项配置的标签、种子、调用顺序、通过状态及失败来源。
- 隐藏阶段的 `hidden_validation_seeds`、`numeric_stress_enabled`、`hidden_completed_checks` 和失败配置。

`validation_summary.json` 按所有声明样本记录编译数、标准正确数、隐藏正确数、两者均正确数，以及其中有有限正耗时的样本数。失败和缺失样本保留在分母。编译数沿用原结果的 `compiled` 标记（源码加载/初始化阶段），不额外证明所有延迟编译内核都已构建。它是原始阶段诊断，**没有检查计分资格**；完整性能评分仍读取 `paper_analysis.json` / `eval_results_scored.json`，不会把有耗时直接当作有效加速。

`src/reliability.py` 纳入 evaluator 指纹，修改协议会使旧恢复结果失效，客户端与远端 worker 版本也必须一致。没有新增默认开关；`--no-hidden-tests` 仍可关闭整个隐藏阶段，此时没有隐藏多种子与压力检查。报告会明确记录隐藏阶段是否开启。

这些检查不能取代进程/文件系统安全隔离、GPU 内存检查、功能合法性审计或高精度 oracle。保留已有非有限值比较规则：NaN/Inf 的位置和符号必须与参考一致，并不全局禁止参考本身产生非有限结果。

## 验证

Python 3.10 / Torch 2.9.0 下，172 项 CPU/模拟测试通过，其中新增 58 项。新增用例覆盖缓存答案、共享缓冲覆盖、第二次调用错误、状态与 RNG 配对、后端设置恢复、无效参考/空隐藏集、三种子重生成、三类压力分布、输入域限制、全部 29 道适配题的小形状参考执行、源码命名空间隔离、阶段统计与恢复指纹。

```bash
python -m pytest -q tests/test_reliability.py tests/test_correctness_validation.py \
  tests/test_baseline_timing.py tests/test_scoring_recovery.py tests/test_workflows.py \
  src/unit_tests/test_dataset.py src/unit_tests/test_score.py
```

Slurm 作业 50702 使用 `srun --immediate=5` 申请 taurus 的一张 GPU，返回 `Requested nodes are busy` 后终止，GPU 测试未执行。真实 CUDA 与项目固定 Torch 2.12.0 的运行验证尚未完成。
