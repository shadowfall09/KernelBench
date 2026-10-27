# 选集区分度与得分—预算研究流程

本轮只实现分析、计划和执行入口，并运行CPU/模拟回归，**不运行模型、GPU、Trainium或训练实验**。原研究包缺少7模型原始运行数据；新代码不会把已有选集和归档统计标为“排名已复现”。

## 先生成可审查计划

`study_plan` 可以生成one-shot、原生反馈循环与Harbor通用coding agent的多模型、多预算、多会话网格。以下模型名称是占位符，需要替换为所选adapter实际支持的名称：

```json
{
  "name": "protocol_study",
  "base_config": {
    "hardware": "RTX_A6000",
    "generation": "local",
    "evaluation": "local"
  },
  "models": ["YOUR_MODEL_A", "YOUR_MODEL_B"],
  "methods": ["one-shot", "feedback", "harbor"],
  "harbor_agent": "claude-code",
  "levels": [1, 2, 3],
  "sessions": 3,
  "vary": "wall",
  "budgets": [900, 2400, 3600]
}
```

`base_config` 也可指向相对该spec文件的现有TOML。原生agent/backend、GPU目标、精度、题集、验证等级等沿用RunConfig。每个job在生成计划时就经过配置及选题校验；每个会话、预算和层使用独立run目录。

`vary`还支持`total_tokens`、`output_tokens`、`device_lease_seconds`，分别改变`agent_token_limit`、`agent_output_token_limit`和`agent_device_seconds`，并提供对应`declared_*`曲线横轴。token额度是正整数，设备额度可为正小数；后端支持范围、设备占用口径和证明要求见[资源预算](RESOURCE_BUDGETS.md)。Harbor支持wall网格；受控`terminus-2`还支持`output_tokens`网格，其他资源网格拒绝。

```bash
python -m scripts.plan_study --spec study_spec.json --output-dir study
```

默认只写 `plan.json` 和 `study.json`，不启动任何实验。计划绑定评测器与计划实现的内容指纹，修改或过期后须重新生成和审查。将来明确要运行实验时才使用：

```bash
python -m scripts.plan_study --plan study/plan.json --execute
```

runner串行运行已审查的job，复用原生resume逻辑并记录退出状态；失败默认停止，`--keep-going`才继续后面的job。原生任务内部仍可使用其配置的多设备并行。初始化、镜像、verifier与agent执行时间有各自边界，不把墙钟声称为GPU账单。

one-shot始终只生成一个候选且无开发反馈。`vary="candidates"`时，one-shot只保留预算1，其他点明确跳过；Harbor没有原生候选次数保证，因此此网格拒绝包含Harbor。Harbor可用于共同的声明墙钟预算网格，其agent时间上限写入真实Task配置。独立会话是分开的声明尝试，不证明provider随机性可控，也不证明GPU时间或token硬预算已经落实；受控Terminus 2输出额度已实现，其他缺口仍属于R-18。

## 从当前证据导出

分析命令读取 `study.json` 的runs数组：

```json
{"runs": [
  {"path": "runs/model-a-session-0-level1", "session": "session_000"},
  {"path": "runs/model-a-session-0-level2", "session": "session_000"}
]}
```

相对路径以manifest所在目录为准。同一模型、协议与session可以合并不同层或互不重叠的范围；同一任务重复出现会报错，不挑最好的一次。每个session必须包含相同的声明任务集合，每题所有声明样本保持分母。失败、缺失和当前证据不合格的槽位为0。缺一整层运行不能悄悄缩小总体。

导出会重新运行**文件分析**，核对当前源码、隐藏验证、基线与所需证据，不执行候选或生成新内核。每个record保存参考源码、评测协议、用量和输入文件哈希。改动候选后，旧分数不再沿用。相同run目录或同一份生成收据不能被改名为多个独立会话。

可用指标：

| 指标 | 含义 |
| --- | --- |
| `score` | 当前运行选择的正式性能分数 |
| `bounded_gain` | 已验证比较中的bounded gain |
| `headroom_score` | 要求该运行已启用且取得合格Headroom证据 |
| `qualified_correctness` | 同时满足当前计分证据的正确槽位；不是原生pass@k |
| `qualified_fast_p` | 合格、显著快于基线且摊销加速比超过声明阈值的槽位 |

同组要求评测器、硬件/精度、基线内容、输入协议、样本数、预算、生成框架等相容。模型身份优先使用provider收据，配置名和身份来源也保留；多模型混用或无法归因的记录不能冒充单模型正式比较。`--audit-only`允许输出明确标记的探索性诊断，不补齐缺失身份或成本。内容哈希用于追踪变更，不是数据真实性的签名认证。

## 选集与排名

```bash
python -m scripts.analyze_selection --manifest study/study.json \
  --selection-suite kernelbench105 --output study/selection.json
```

也可用 `--selection selection.json` 指定 `[{"level":1,"problem_id":1}, ...]`。默认要求完整覆盖所请求选集；只有显式 `--allow-partial-selection` 才取已覆盖部分，同时逐项列出遗漏，不能把104题写成105题。选集只改变成员，不把暂定I/F/D标签转换成边界证书。

分析给出全量/选集的逐模型分数、差值区间、平均并列秩的Spearman、Kendall tau-b、严格名次反转，以及模型间分数标准差和范围。常量分数的相关系数为null，不写成1。任务相关性只作探索性冗余诊断，不据此自动删除题目；少于3个subject不提供这种诊断。

bootstrap保持选集成员与层的数量固定：同一次任务抽样同时用于全量和选集，再对各subject独立抽会话、对任务抽声明样本槽位。`--aggregation task_mean|layer_macro`明确控制分析权重。该方法以当前选集为条件，不测量选题算法本身的不确定性；不校正所有算子族相关性，也不能用单次会话估计会话间方差。没有使用少量模型下的渐近相关性p值。

二元指标可额外指定 `--assumed-discordance 0.3 --target-power 0.8 --alpha 0.05`，输出一个**假设每独立任务只有一个成对二元结果**时的正态设计近似。它不是当前多样本/多会话数据的经验power，也不自动重申旧稿的数值。零discordance或固定discordance下不可达的差值返回null。

## 得分—预算曲线

安装可选绘图依赖 `pip install '.[analysis]'` 后可同时导出静态图：

```bash
python -m scripts.analyze_budget_curves --manifest study/study.json \
  --vary wall --axis declared_wall_seconds \
  --output study/curves.json --plot study/curves.svg
```

每个点来自单独声明的预算运行，不把隐藏结果的逐轮最好值当作在线曲线。均值保留失败槽位和全部会话，不取最好会话，也不把曲线强行变成单调。点区间重采样任务、会话和声明槽位。`vary`之外的限制继续写在series协议中；改变其他限制会分成不同series，设备/精度/评分不同则分组。

可选择的x轴：声明墙钟、声明候选数，或观测到的总token、输出token、模型调用数、估算美元费用、agent执行墙钟。观测总量按声明样本槽位归一；模型调用或agent时间包含其已记录失败与重试。未知资源坐标为null，该点仍保留分数和拒绝原因，但不参与该资源曲线，绝不当成零成本。x轴本身的不确定性没有估计。

归一化AUC仅在所有series共同的已知预算区间内对实际点作线性插值和梯形积分，不外推；点不足、重复坐标或区间不相交时不输出AUC。图为Matplotlib独立静态文件，不需要交互服务。

## 用量边界

原生LiteLLM采用返回的 `total_tokens`；Claude CLI只有在普通输入、输出、缓存读取、缓存创建字段全都明确可见时才求总量。缺字段保持未知。Harbor 0.24的输入token已包括cache，不能再加一次cache；多步结果有缺项时也不只累加可见部分。Harbor的agent执行计时与verifier/setup分开，缺失或无效时间不填0。

Harbor预算收据还绑定trial结果文件；正式外部agent比较要求可见的输入/输出总量和有效执行时间。缺少模型调用数或美元费用时仍保持null。估算费用、声明墙钟及agent执行秒数都不等于已认证GPU账单，曲线也不声称不同方法耗用了相同设备时间。

统计定义参照[SciPy Spearman文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)与[Kendall文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html)；本实现使用标准库计算，不依赖SciPy运行。token接口说明见[LiteLLM用量文档](https://docs.litellm.ai/docs/completion/token_usage)。

CUDA worker进程配额可用`vary="gpu_busy_seconds"`和秒数预算网格。声明配额、观测进程busy和配额扣额是三个不同的轴；缺完整观测时保留未知，不能把保守worker墙钟扣额写成实测GPU时间。范围和完整性见[资源说明](RESOURCE_BUDGETS.md#cuda进程活动计量与软件配额)，不会据此认证跨进程物理硬限。
