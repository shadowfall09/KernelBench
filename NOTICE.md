# KernelBench Pro 来源与版权说明

KernelBench Pro 由 shadowfall09 维护，维护地址为 https://github.com/shadowfall09/KernelBench。
当前工作包含批量工作流、验证与计时修正、评分和恢复、可靠性检查，以及仓库文档整理。

## 继承代码

参考题目、隐藏测试及部分评测和生成工具继承自
[Meta KernelBench-Verified](https://github.com/facebookresearch/kernel_bench_verified)。
初始整合提交 `2bce940` 记录的上游版本为 `3fdf6fec7372a4d0cb682635f00e7bdcbc55d50e`；
该对象不在当前本地历史中，此处记录来源，不声称完成上游逐文件比对。

该项目建立在 [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench) 基础上。
批量流程也整合了本维护者先前 KernelBench fork 的工作。
保留的 `LICENSE` 和继承文件中的版权声明适用于对应代码；项目介绍调整不改变这些代码的来源。

## 示例与方法来源

`src/prompts/` 中的部分示例参考 Christian Mills 的 GPU MODE 讲解和 Peter Kim 的
Minimal Flash Attention；具体链接保留在该目录的说明中。

正确性协议借鉴了 KernelBench-Hard、KernelBenchX、robust-kbench 的公开方法，
分阶段诊断参考 TritonRL 的验证划分。固定源码版本、采用范围和本地适配见
[可靠性说明](docs/RELIABILITY.md)。

上游论文 PDF、LaTeX 源稿、图表及旧实验报告不随本仓库分发。
