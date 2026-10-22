# GRPO训练、奖励审计与持出评测

本版提供软件入口和CPU/模拟回归，**没有运行真实模型、GPU、Trainium或训练实验**。所有计划与预览命令默认不执行。

## 训练协议

训练使用[TRL 1.8.0的GRPOTrainer](https://huggingface.co/docs/trl/v1.8.0/en/grpo_trainer)，固定`loss_type=dapo`、组内奖励归一化、每组一次优化迭代。优化器、梯度、采样及工具消息的loss mask由开源训练器处理。本项目负责题目、开发工具、最终奖励、证据和预算。安装入口为`pip install '.[rl]'`，支持Python 3.12、TRL 1.8.0和Transformers 5.2–5.x；本次核对了实际1.8.0/5.20.2 API，模型加载与训练调用均用模拟对象替代。

| 模式 | 模型动作 | 开发反馈 | 终局候选 |
| --- | --- | --- | --- |
| `one-shot` | 一次源码响应 | 无工具、无开发测试 | 唯一响应 |
| `feedback` | 源码响应及`submit_kernel(source)`工具 | 标准正确性、同步墙钟及可选profiling；复用原生候选/墙钟限制 | 保留开发通过的最佳源码 |

feedback环境每个rollout独立，公开工具只有`submit_kernel`。终局评分前关闭环境；已用完候选额度的失败episode不能另加一次终局尝试。没有调用开发工具时，可把唯一终局响应作为候选。隐藏检查结果不加入对话；训练任务的终局奖励只用于策略更新，持出题不会传给训练器或奖励callback。

## 冻结计划与分组

下面是配置示意，checkpoint路径必须已经存在：

```json
{
  "training": {
    "checkpoint": "/path/to/local/checkpoint",
    "mode": "feedback",
    "reward_mode": "qualified",
    "group_size": 4,
    "max_steps": 100,
    "max_prompt_tokens": 8192,
    "max_completion_tokens": 4096,
    "learning_rate": 0.000001,
    "temperature": 0.8,
    "beta": 0.0,
    "wall_seconds": 14400.0,
    "use_cpu": true
  },
  "verifier_config": {
    "hardware": "RTX_A6000",
    "generation": "local",
    "evaluation": "local",
    "gpus": "0",
    "parallel": 1,
    "optimization_rounds": 5,
    "agent_timeout": 2400,
    "strict_validation": true,
    "evidence_level": "L0",
    "paired_timing": true,
    "final_repeats": 3
  },
  "train": [{"level": 1, "problem_id": 2}],
  "heldout": [{"level": 1, "problem_id": 26}]
}
```

任务ID、参考及隐藏源码哈希、初始checkpoint内容、训练/验证配置和实现指纹共同固定计划。默认按任务、参考源码和已审查family分开train/heldout；family标签不是语义等价证明。模型只收到现有input-blind策略处理的公开参考。超出`max_prompt_tokens`时拒绝加载策略权重，不截断题意。当前默认L0限于已有17道审查契约，其他题必须先补契约或显式选择较低证据要求。

```bash
python -m scripts.rl_workflow plan --spec rl_spec.json --output-dir rl_study
python -m scripts.rl_workflow baselines --training-plan rl_study/plan.json
python -m scripts.rl_workflow train --plan rl_study/plan.json
```

以上命令只建计划、读基线或预览训练。基线准备、训练、重评分分别需要显式`--execute`；本次没有使用真实执行入口。训练前要求当前完整基线组合及设备UUID证据，缺失或过期便在模型加载前停止。基线准备是独立作业，其成本不记成策略训练免费额度。

初始模型只从本地safetensors加载，禁止远端Python代码，运行时关闭Hub联网、遥测与外部报告上传。默认`use_cpu=true`；设为false时，TRL使用该训练进程的首个可见CUDA设备，验证设备仍由`verifier_config.gpus`显式选择。第一版只支持一个训练进程与一条local/Docker验证设备通道，拒绝分布式和远端奖励排队。

## 终局奖励与计时

每个完整同题group先做源码结构检查，再做审查小实例；合格者才进入原尺寸标准、隐藏验证和冻结重测。`qualify_small`是独立worker任务，不分配原尺寸计时输入。所有声明样本保留在分母，失败记零。

冻结后，各候选使用同一repeat的公共/隐藏随机种子，在同一设备上按随机次序交替启动独立重测进程；标准、隐藏与候选/参考/参考控制计时均保留全部重复。它不把所有候选模型常驻同一进程，也不宣称消除了漂移。基线与实际计时结果的UUID必须一致；缺失或跨设备收据不给组奖励。某个候选编译/正确性失败不会把有效同组候选一并归零。

`qualified`奖励直接使用当前原生评分：任何必需检查或证据不合格都为零。`raw_speedup`是明确的弱奖励消融，使用标准正确时的`min(基线均值/候选均值, raw_speedup_clip)/raw_speedup_clip`，默认clip为4；它不等于完整资格分。消融会继续测量小实例失败但能通过旧标准检查的源码，便于记录奖励漏洞。两个奖励值、失败检查、源码、种子、调度顺序、请求墙钟、基线快照及原生结果一起保存，不能以响应格式或代码长度替代性能奖励。

可选算法证书奖励默认关闭。启用`certificate_bonus`需要Headroom评分及事先审查的D/SOL缩放契约；只有已通过完整资格、达到`certificate_min_score`的候选才扫描尺寸。至少5个尺寸、跨度至少16倍，保持源码、参考变体、物理设备、dtype和声明的形状关系不变；每个尺寸重新验证并采集独立配对重复。候选斜率区间须符合审查的算法指数，并显著低于参考指数。恒定倍数加速不能获得算法设计加分；证书失败只取消加分，原资格失败仍为零，总奖励上限为1。它是有限尺寸区间的经验支持，不能证明普遍渐近复杂度。配置与证据格式见[算法证书](SCALING_CERTIFICATES.md)。

## 预算与恢复

每个优化step在生成前持久化预留一个group。中断发生在模型响应、奖励或优化步骤之间时，该group仍占用总额度；不能靠删除结果或退回旧checkpoint重置预算。checkpoint绑定模型/优化器等文件、trainer step和当时的奖励日志前缀；恢复会复核已有奖励证据，保留checkpoint之后的中断成本。缺失账本或变更的checkpoint拒绝恢复。

最终策略保存`final_lineage.json`：绑定初始/最终权重、计划、软件版本、奖励及其底层文件、算法证书、全部已消费group和实际进入当前权重的优化路径。退回checkpoint后的旧更新继续算成本，但不进入恢复后策略的优化路径。父进程使用独立nonce关联本次worker、墙钟账本和lineage，并用锁防止同目录并发监督；旧的完成文件或只返回零退出码的worker不能证明当前完成。这些是评测器维护的内容绑定，不是硬件签名或OS隔离证明。

训练CLI由父进程监管总墙钟，预留5秒用于清理嵌套子进程；请求和清理耗时均记账。丢失的父进程预留整笔扣除。驱动/内核的物理抢占和GPU账单仍未认证。TRL有自己的生成路径，本版不把原生`agent_token_limit`等选项套用到训练后声称硬限，而会拒绝这些配置。

计划给出保守输出上界：`max_steps × group_size × max_completion_tokens × (1 + 最大工具轮数)`，one-shot最后一项为1。实际保存的completion序列可包含工具观察，不能当作精确模型输出token或计费收据；超额工具再生成可能被训练器截断，已生成但丢弃的token也不能从最终序列反推。`usage_complete=false`会保留在结果里。

## 红队、稳定性与持出集

`regrade-plan`接收预先声明的候选源码、task、重复次数和split；`regrade`默认预览，显式执行才调用同一奖励管线。每次重复保留新进程/种子，resume只复用完全一致且仍有效的收据；候选重排也会拒绝旧样本映射。

```json
{
  "split": "train",
  "repeats": 30,
  "cases": [
    {"case_id": "candidate_a", "task_key": "1/2", "source": "candidate_a.py"},
    {"case_id": "zero_output", "task_key": "1/2", "source": "zero_output.py",
     "expected_invalid": true, "expected_block_checks": ["precision_contract"],
     "rationale": "输出为零，违反数值契约"}
  ]
}
```

`audit`只分析已有证据，报告ICC(A,1)、5%及以上独立runtime锚点的奖励排序、攻击接受率及Wilson区间。ICC至少需要两个不同内核与矩形重复矩阵；单个内核即使重复30次也不能识别ICC。常量矩阵、缺锚点、缺数据保持null/未测，奖励并列不算排序正确。共享内核/组的排序事件不独立，因此Wilson区间明确保留该假设限制，不作为统计认证。ICC公式对应[双向绝对一致的单次测量定义](https://pingouin-stats.org/generated/pingouin.intraclass_corr.html)。

红队案例必须声明预期失败检查，并有该检查确实失败的证据；缺GPU、缺基线或工具不可用不能算“成功抵御攻击”。公共/最终奖励差可从同一源码的两份当前原生运行离线重算，必须仅在是否启用隐藏检查上不同；失败样本和配对bootstrap保留。

`heldout-plan`为命名的本地checkpoint生成原生one-shot/反馈评测计划，模型资产内容冻结，使用独立任务/会话目录。任意命名checkpoint只作来源声明；要比较训练效果，使用下面绑定lineage的预注册入口。

## 绑定训练来源的奖励对照

对照规格声明`base`（上面的训练规格）、`seeds`（如`[11,22,33,44,55,66]`）、`evaluation_sessions`和`evaluation_samples`。系统在执行前生成全部`qualified`与`raw_speedup`种子组合，两组固定优化器、任务划分、初始权重与声明预算；弱奖励组关闭证书加分，差异仅为奖励构造与种子。成本包含中断group，额外证书请求也记墙钟；相同声明额度不表示实际GPU busy时间完全相同。

```bash
python -m scripts.rl_workflow experiment-plan --spec reward_experiment.json --output-dir reward_study
python -m scripts.rl_workflow experiment-train --experiment reward_study/experiment.json
python -m scripts.rl_workflow experiment-evaluation-plan --experiment reward_study/experiment.json --output-dir reward_heldout
python -m scripts.plan_study --plan reward_heldout/plan.json
python -m scripts.rl_workflow experiment-analyze --plan reward_heldout/plan.json --output reward_effect.json
```

训练与持出runner仍默认不执行。未训练时评测计划返回`not_ready`；缺少预注册作业的执行记录时分析返回`not_measured`，不能删掉缺失种子后缩小研究。全部策略必须匹配预声明训练计划、当前奖励/源码/模型资产、软件版本及父进程墙钟收据。模拟训练只能在显式`--audit-only`中检查管线，不能进入正式效果结论。

分析从原生持出结果重新计分，保留全部候选槽位和失败零分，以训练种子为独立重复，按层重采样任务、独立重采样各组评测会话/槽位。同时报告配对差、95%区间和双侧符号随机化检验；检验依赖零假设下的组标签可交换性或符号对称性。两个完全正向种子仍有`p=0.5`，不会因退化的正区间而声称显著。训练中触发失败检查的比例单列，未知槽位给上下界，不假设没有违规。正式效果支持还要求全部当前推理与训练证据合格。

R-20的软件入口已补全；预注册阈值和效果支持条件均为待实际数据检验的规则，当前没有实测结果。候选Python仍由现有独立进程执行，完整OS隔离继续归R-23。
