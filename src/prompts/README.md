<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
-->

# KernelBench Pro Prompt 示例

本目录保存运行时读取的 PyTorch / CUDA 示例及硬件说明，供 `src/prompt_constructor.py` 构造生成材料。

| 路径 | 内容 |
| --- | --- |
| `model_ex_*.py`、`model_new_ex_*.py` | 参考模型与候选实现的配对示例 |
| `few_shot/` | 矩阵乘法、GELU、MNIST、Attention 等示例 |
| `cot/` | 示例实现的推导材料 |
| `hardware/` | GPU 参数说明 |

这些文件通过路径动态加载，不能仅因没有静态 import 就认定无用。

部分示例来源：

- Fused GELU / Tiled Matmul：[Christian Mills, GPU MODE Lecture 04](https://christianjmills.com/posts/cuda-mode-notes/lecture-004/)。
- Minimal Flash Attention：[Peter Kim, Minimal Flash Attention](https://github.com/tspeterkim/flash-attention-minimal/tree/main)。
