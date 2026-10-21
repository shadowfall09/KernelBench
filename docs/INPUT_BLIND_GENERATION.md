<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
-->

# KernelBench Pro 输入盲化生成

指定题目的生成材料去掉 `get_inputs()`、`get_init_inputs()` 和测试配置，仅提供 import、模型类及其形状说明。完整题目和隐藏测试供验证器使用。

## 当前题目

配置以 `src/prompt_constructor.py` 中的 `STRIP_TEST_CONFIG_PIDS` 为准：

| Level | 题号 | 操作 |
| --- | --- | --- |
| 1 | 90 | cumprod |
| 1 | 100 | HingeLoss |
| 3 | 35 | LSTM |
| 3 | 47 | NetVLAD |

批量入口通过 `kb_workflows/generation.py` 的 `prompt_source()` 按配置动态提取源码，放入本次工作目录的 `model.py`；原生生成入口使用 prompt constructor。无需维护预先导出的 prompt 文件。

## 添加题目

1. 核对题目是否需要隐藏测试配置，并补全模型 `forward()` 的参数与 shape 说明。
2. 将 `(level, problem_id)` 加入 `STRIP_TEST_CONFIG_PIDS`。
3. 检查动态生成材料中仍有完整模型类，且没有输入函数和测试常量。
4. 重新生成并验证候选内核。

可在终端预览批量入口实际使用的材料：

```bash
python - <<'PYTHON'
from kb_workflows.common import select_problems
from kb_workflows.generation import prompt_source

path = select_problems(1, problem_ids=[90])[90]
print(prompt_source(path.read_text(), 1, 90))
PYTHON
```

输入盲化限制的是 prompt 内容，不是文件系统安全沙箱；本地、Docker 和 Modal agent 的文件访问能力仍取决于执行环境。
