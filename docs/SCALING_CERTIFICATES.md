# 候选算法证书与可选奖励

证书检验同一候选在声明的尺寸区间是否表现出审查预期的缩放斜率，回应M3的“常数优化与算法改进”区分和M4的可靠训练奖励。默认不启用，也没有运行真实设备扫描。

## 配置与审查来源

训练配置可增加`certificate_bonus=0.05`、`certificate_min_score=0.5`和`scaling_contracts=/path/to/scaling_contracts.json`。只允许`reward_mode=qualified`及`score_mode=headroom`；规格和校准仍须满足[Headroom证据门](HEADROOM_SCORING.md)。失败候选不能请求证书，奖励上限为1。

缩放文件的顶层为`protocol: kernelbench_pro_scaling_contracts_v1`及`tasks`对象。每个任务键如`5/8`声明以下字段：

| 字段 | 含义 |
| --- | --- |
| `review.reviewer/rationale` | 对算法、尺寸区间与形状关系的人工审查 |
| `reference_sha256` | 当前参考Python源码的SHA256 |
| `bound_certificate_sha256` | 已通过原生验证的D类Headroom证书指纹 |
| `axis` | `input`和`dimension`，实际输入签名中的尺寸轴 |
| `shape_laws` | `input`、`dimension`及`power`列表；尺寸轴为1，未列维度必须固定 |
| `candidate_exponent/reference_exponent` | 审查的候选与参考缩放指数，候选须更低 |
| `tolerance` | 指数置信区间允许的绝对偏差，范围0至0.5 |
| `baseline_variant` | 固定参考变体，默认`eager`，各尺寸必须相同 |
| `instances` | 至少5个不同的`dimensions`与正整数`memory_mb`实例 |

`axis`、`shape_laws`、两个指数和`tolerance`必须与D边界证书中的`design.scaling`完全一致。模型响应不能自行声明这些期望值；问题边界证书也不会自动变成候选算法证书。

自动扫描目前支持Production-41。普通运行也可通过`task_dimensions`指向JSON，例如`{"S":16,"L":16}`，覆盖所选profile的维度。只接受已知正整数维度，核对head/group/rotary等整除约束；超出显存计划报错，不悄悄缩小尺寸。隐藏测试仍使用独立隐藏实例。尺寸内容参与基线、最终冻结、评分、比较与恢复指纹；Harbor只将它交给可信verifier。

## 检验与留档

至少5个不同尺寸且最大/最小不少于16倍；每个尺寸至少2组独立最终重复，每组至少10个已验证参考/候选/参考控制配对。所有点固定候选/参考源码、参考变体、完整设备与软件身份、dtype、输入结构及声明形状关系。每次扫描拥有独立基线文件，不能覆盖此前训练奖励依赖的记录。

拟合`log(mean T) = alpha log(N) + c`，先按重复组、再按配对样本bootstrap，同时保留参考与候选配对。两个指数的95%区间都须落入各自审查指数的容差范围，且“参考指数−候选指数”的区间下界大于0。仅获得相同斜率的恒定倍率加速，即使快100倍，也不会通过算法证书。

每点完整运行源码、小实例、标准/隐藏及冻结配对验证；缺失、失败或跨设备点不能从拟合中删除。证书保存anchor计划/输入签名、审查契约、所有点的原生奖励与原始配对文件。重读时重新核对当前D边界证书、任务/评测器、文件哈希、模型与计划绑定，并重算拟合；修改任一证明文件使证书及关联训练lineage失效。证书请求时间记入奖励墙钟和父训练进程预算。

检验只支持声明的有限尺寸区间。启动开销、缓存层级或算法切换可能使斜率检验失败；通过也不证明普遍渐近复杂度、物理最优性或训练有效性。CPU回归使用明确标记的模拟计时证据，不能作为实机结果。
