<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
-->

# KernelBench Pro 贡献指南

本仓库在 [shadowfall09/KernelBench](https://github.com/shadowfall09/KernelBench) 通过 GitHub issue 和 pull request 维护。

## 开发流程

1. 从 `main` 创建分支，围绕一个具体问题修改。
2. 在说明中写清问题、修改后的行为和验证结果；接口或默认值变化时同步更新文档。
3. 功能变化同步更新 `docs/FEATURE_INVENTORY.md` 和 `docs/motivation_map.json`，用 `python -m scripts.check_motivations --write` 刷新逐项动机表；区分设计、CPU/模拟验证和真实 GPU 验证。
4. 运行与修改相关的测试及 Ruff 检查，再提交 pull request 供维护者评审。

```bash
# 按改动所在职责运行，例如：
python -m pytest -q tests/runtime/test_harbor.py
# 跨模块重构时运行完整软件回归：
python -m pytest -q tests src/unit_tests/test_dataset.py src/unit_tests/test_score.py
python -m ruff check kb_workflows tests src/reliability.py src/validation.py
python -m scripts.check_motivations
```

Python 使用 4 空格缩进，项目 Ruff 配置位于 `pyproject.toml`。GPU 测试在分配到的设备上执行；报告设备、依赖版本和实际完成的检查。

## 文件与结果

题目放在 `KernelBench/`，隐藏测试放在 `hidden_tests/`，批量流程放在 `kb_workflows/`。
实现与测试按 [README 的职责目录](README.md#代码布局)归类，优先扩展现有模块；不要为每个设计编号新增模块或脚本。入口相近时合并子命令，删除旧实现并更新调用者，不保留无用转发层。
生成内核、运行结果、编译缓存和凭据不作为项目源码提交。隐藏测试不放入生成 prompt。

测试围绕不同算法、边界和真实错误反例编写。全题清单与 meta 元数据检查集中在 `tests/benchmarks/test_catalogs.py`，复用模型构造；通用来源拒绝规则验证一次，不按每题复制。共用线程设置放在 `tests/conftest.py`。不为简单文件移动或可逆文档修改新增测试，不以参数化条目数衡量覆盖质量。
本项目现行研究稿和可核对证据放在 `research/headroom/`；上游论文宣传、历史草稿和旧实验产物不随仓库维护。
论文的每项已实现声明应能追踪到代码和验证记录；规划功能不能写成实验结果。需要核对继承来源时使用 `NOTICE.md` 和版本历史。

继承代码的版权与许可证声明应保留。贡献按仓库的 [MIT License](LICENSE) 提供。
