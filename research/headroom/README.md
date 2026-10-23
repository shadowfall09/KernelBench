# KernelBench Pro 的 Headroom 研究

本目录整合用户提供的 `headroom-bench-20261021.zip`。项目仍叫 **KernelBench Pro**；HR-Eval 指目标评测协议，HEADROOM-Bench 指目标任务设计，两者不是额外发布的已完成产品。

| 入口 | 用途 |
| --- | --- |
| [论文](PAPER.md) | 唯一现行英文稿，按五条 motivation 组织，区分已实现与研究设计 |
| [Motivation](../../docs/MOTIVATION.md) | 中文主线、67 项历史改动、本轮改动与待完成项 |
| [功能清单](../../docs/FEATURE_INVENTORY.md) | 保留逐版提交与验证历史 |
| [来源清单](evidence/source_manifest.json) | 压缩包与采用文件的 SHA-256、转换产物摘要、未导入类别 |
| [任务选集](../../kb_workflows/suites/kernelbench105.json) | 随包分发的运行清单，含参考源码快照、角色和暂定层 |

没有导入 archive 下的旧论文与旧大纲、内部对话记录、重复清单、未上 GPU 的攻击程序、依赖缺失的 `strict_check.py` 和本机路径脚本。证据表经过排序、字段标准化和一致性核对，论文与 motivation 按当前实现重写；不在仓库根目录堆放压缩包副本。

## KernelBench-105

选集来自原 `evidence/kb100.csv`，实际主集为 105 题：L1 39、L2 40、L3 26；暂定 I/F/D 各 30/56/19。另有 5 个 probe 和 140 个 extended，三种角色合起来覆盖原 Level 1–3 的 250 题。保持原选择种子 `20261002` 和全部选择理由；没有根据当前模型成绩重新挑题。

默认仍为完整题集，使用 `--suite kernelbench105` 显式选择。现有 `start/end` 与选集取交集，配置若只写了 `end = 10`，不会自动扩到整个 level；交集为空显式报错。

```bash
# 预览某一 level 的完整主集；不需要 GPU 或模型调用。
python scripts/run_batch.py --config configs/a6000-feedback.toml \
  --suite kernelbench105 --level 1 --start 1 --end 100 --dry-run

# 使用相同入口运行；L2 的范围为 1–100，L3 为 1–50，各 level 用独立 run_name。
python scripts/run_batch.py --config configs/a6000-feedback.toml \
  --suite kernelbench105 --level 1 --start 1 --end 100 --run-name kb105_l1

# 探针独立报告，不计作主集；也可选择 extended。
python scripts/run_batch.py --config configs/a6000-feedback.toml \
  --suite kernelbench105 --suite-role probe --level 2 --start 1 --end 100 --dry-run
```

TOML 的 `[run]` 可写 `suite = "kernelbench105"`、`suite_role = "scored"`。选择结果贯穿基线、生成、验证、pass@k、性能评分、资源记录和 Harbor 导出。子任务展开为单题配置，不依赖宿主研究目录。suite receipt 记录清单摘要，按角色分组比较；probe/extended 的评分报告标记 `leaderboard_eligible = false`。参考源码与清单快照不符时拒绝运行，维护者需重新审阅并版本化清单。

运行清单当前为version 2：保留原选题与角色，声明第92题exclusive scan参考修正，以及89–93五题的隐藏输入契约更新。原始研究CSV保持不变，当前修订、旧结果失效与来源绑定见[参考修订](../../docs/REFERENCE_REVISIONS.md)。

此接入没有复现原稿声称的 τ=1.00：7 模型原始结果不在压缩包中，另有历史分析输出使用 104 题，不能混作 105 题结果。选集中仍有输出退化或低信号题；原L1名义值分布校准与强数值新基线已有实现，见[输入校准](../../docs/INPUT_CALIBRATION.md)；全题适配和逐设备阈值校准尚缺，也没有运行真实校准实验。因此这是**可运行的研究选集**，还不是完成验证的 HEADROOM 发布版。层标签可以用于描述性统计，不作为设计证书或 headroom 分母。

## 保留的证据及局限

| 文件 | 保留内容 | 解释边界 |
| --- | --- | --- |
| [selection.csv](evidence/selection.csv) | 250 题的角色、层、选择单元与理由 | 固定选择，不证明当前区分度 |
| [audit_implementations.csv](evidence/audit_implementations.csv) | 1,574 份实现的原审查结论、问题标签、依据和历史源码路径 | `raw_source_available=false`；没有原候选源码，不能据此复跑 |
| [audit_statistics.json](evidence/audit_statistics.json) | 原审查汇总，逐实现最严重结论和可重叠问题标签 | 159 无效 + 60 无法验证 = 219；217 为最终 noise 分类，346 为带 noise 标签，两者口径不同 |
| [ledger_compact.csv](evidence/ledger_compact.csv) | 250 题 meta 跟踪的历史 FLOPs/bytes 与解析估计 | 规格表模型，不是当前设备实测上限或严格算法下界；失败行也保留 |
| [l1_value_table.csv](evidence/l1_value_table.csv)、[l2_l3_value_table.csv](evidence/l2_l3_value_table.csv) | 历史任务价值分析 | 名称、尺寸与来源混杂；不自动参与当前计分 |
| [sakana_l1_sol.csv](evidence/sakana_l1_sol.csv) | 历史发布倍率与下界估计的对照 | 原硬件部分为推断；不把异常倍率判成已复现作弊 |

审查总数按最严重结论划分为：159 无效、60 无法验证、88 风险、217 noise、14 待核、6 需说明、34 说明、996 可信。“可信”只是该次源码审查未发现问题；“noise”是审查者的倍率分档，不是测量得到的噪声置信区间。没有在本次运行任何候选 GPU 程序。

原稿的 E0（Claude Code 历史实验）没有逐题产物，本目录只在论文中保留为作者报告，不能当成本仓库当前协议的基线。未复核的外部数字和实验计划见论文的证据表。

## 重建与校验

```bash
python -m scripts.headroom_evidence --archive /path/to/headroom-bench-20261021.zip \
  --output /tmp/headroom-evidence
python -m scripts.check_motivations
```

导入器使用标准库，按固定成员名读取 ZIP，验证选集和审查统计后写出标准化证据；不执行归档 Python、反例或候选。比较重建目录与本目录 evidence 的文件摘要即可核对转换。运行清单单独受版本控制，修改研究证据不会悄悄改动题集。

Production-41代码已接入独立Level 5，四轨包含12/6/12/11题；角色、显存计划、反向与数值门见[生产题说明](../../docs/PRODUCTION_TASKS.md)。本轮不运行设备、模型或训练实验；原105选集的排名结论仍未重做。

原L1新增[算子专用隐藏输入](../../docs/OPERATOR_INPUTS.md)：真实初始化下的100题形状/布局/边界实例和覆盖收据已接入，不改变名义计时。CPU近常量参考失格保留；这不等于完成L2/L3或实机验证。

原L2的100题现有[来源绑定的复合FP64算式](../../docs/COMPOSITE_NUMERICS.md)，与原L1合计200题。算式、原构造meta和原生控制路径已作CPU/模拟检查；L3及设备实验没有因此完成。
