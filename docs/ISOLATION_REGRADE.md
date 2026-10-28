# 验证器隔离与产物重评分

R-23的软件实现包括受限Docker验证器、Harbor网络与权限配置，以及原生/Harbor的已有产物重评分。本轮只实现代码和CPU/模拟回归，没有构建或启动真实容器，没有运行设备、模型、校准或训练实验。配置和运行时观察不等于安全认证，`security_isolation_certified`保持`false`。

## 受限验证器

原有执行方式为`isolation_mode = "standard"`。显式使用：

```toml
[run]
generation = "local"
evaluation = "docker"
isolation_mode = "restricted"
isolation_cpus = 4
isolation_memory_mb = 65536
isolation_pids = 512
isolation_tmp_mb = 16384
isolation_report_mb = 64
```

这些是主机CPU、内存、进程数、临时存储及结果交接上限，GPU显存仍由任务配置及实际设备决定。`isolation_tmp_mb`分别约束tmp与编译缓存挂载，总内存还受cgroup约束。tmpfs保留执行权限以支持JIT动态库，同时使用`nosuid,nodev`。预览：

```bash
python scripts/run_batch.py --config configs/a6000-restricted.toml --dry-run
```

当前受限原生transport支持CUDA/Docker。local进程、Modal、HTTP队列和Neuron不会被自动降格视为满足该策略。Docker/Modal生成环境内的本地验证也不能冒充独立受限worker；原生生成可使用local控制器，或使用独立Harbor。这里限制的是提交候选的验证环境；local生成控制器的shell权限不因此被沙箱化。

父进程先把本地镜像标签解析为不可变image ID，再创建而不立即启动容器。它使用：

- 单个明确设备分配；禁止默认扩展到全部GPU。
- 非root的65532:65532身份、清空capabilities、no-new-privileges和有效seccomp。
- 禁止联网、只读根文件系统、独立PID/IPC/网络/mount及cgroup命名空间。
- CPU、主机内存与相同的memory+swap上限、PID上限和有限的Docker日志。
- 只读请求目录、单独结果目录，以及每请求独立tmpfs编译缓存；没有宿主仓库、共享缓存或Docker socket挂载。
- 清空进程环境后设置必要的Python/CUDA路径和缓存位置；不向worker传递宿主模型API认证。

启动前和退出后都从Docker读取实际配置，核对image、命令、用户、设备、挂载、网络与资源。worker在导入候选前核对评测器摘要，读取Linux UID/GID、有效/许可/边界/ambient capability、no-new-privileges、seccomp、网络接口、根挂载、cgroup上限及命名空间。父进程还核对worker与自身的命名空间不同。cgroup v2或必要观察不可用时，没有受限资格。

每次运行使用新目录、新容器和新缓存。退出后先确认容器停止，再读取有大小上限的普通JSON文件；符号链接、FIFO、超限文件和非有限JSON不被接受。worker为私有父目录中的结果设置可读取权限，兼容rootful Docker下不同UID的交接。父进程删除自己创建的容器及匿名卷，并再次查询确认清理；create客户端超时时只认领带本次nonce标签的容器。超时或清理未确认时不给通过。

原生生命周期共用任务的请求截止时间，预留清理时间；parent记录的开发设备lease仍覆盖完整transport。守护进程失去响应、宿主进程崩溃或GPU驱动卡死并不因此获得物理抢占保证。实际GPU独占、busy时间硬限与账单仍不在这份收据中，见[资源预算](RESOURCE_BUDGETS.md)。未运行容器逃逸、设备独占或兼容性实验。

`metadata.execution_envelope`绑定当前请求、评测器、image ID、实际配置、worker观察、结果摘要及清理。评分与恢复会重新检查候选和基线变体的凭据；工具检查的原始计时不因外层隔离收据而改变。标准、隐藏、基线变体、最终重复与配对结果的image身份必须一致；新库或规则使用新重评分计划，不把保留的旧观察重新标成新runtime。模拟执行显式标注，正式资格拒绝模拟收据。来源真实性、计数器和算法证书仍是各自的附加门，不能由容器策略替代。

## Harbor

Harbor任务继续使用schema 1.4和独立verifier，验证器显式禁止联网。受限模式为verifier生成只读、非root、capability清空及资源受限的Compose配置。可信代码和隐藏测试已放进image，实际0.24 SDK会识别bundled tests；`/app`交接区、编译缓存、临时基线和verifier日志各有自己的可写位置。桥接器在候选评测前读取Linux限制；这份容器内观察没有独立宿主inspect证明，不冒充原生Docker外层凭据。

使用Harbor agent时，执行阶段的域名需显式配置，例如：

```toml
[run]
generation = "harbor"
evaluation = "docker"
harbor_environment = "docker"
isolation_mode = "restricted"
# 填写实际模型服务的域名/IP/CIDR，逗号分隔；不要填写URL路径。
harbor_agent_hosts = "api.example.test"
```

示例域名需替换。SDK分别处理agent执行、环境准备和验证阶段；此allowlist控制agent执行阶段，不是安装阶段的网络认证。导出和regrade没有agent调用。配置中的主机名也不是“这些地址只提供模型API”的证明。

显式候选产物为`/app/solution.py`；Harbor还自动记录约定的`/logs/artifacts`目录。重评分会核对两者的manifest覆盖，保留旧目录状态，不伪造缺失记录。原生最终评测只重新接收候选源码，其他agent数据不能替代原生验证。

## 原生已有候选重评分

先准备当前验证器配置。源运行必须有`run_config.json`及记录的完整题目×样本范围；不能只从尚存的几个kernel推断分母。下列命令只生成计划：

```bash
python -m scripts.regrade_artifacts native-plan \
  --source-run runs/old_run --config configs/a6000-restricted.toml \
  --output runs/regraded_run

python -m scripts.regrade_artifacts native-run \
  --manifest runs/regraded_run/regrade_manifest.json
```

`native-run`默认仍只校验并预览。实际调用验证器需要额外显式`--execute`；本轮没有执行该选项。

计划保留全部声明槽位，包括缺失候选。现存、UTF-8且不超过8MiB的源码按SHA冻结；缺失、空文件、超限或无效编码保留为缺失/不支持槽位，不能被后来补上的实现悄悄替换。符号链接拒绝。旧运行的配置、分数、生成成本/尝试、标准/隐藏/final/paired原始结果和已有agent轨迹复制到私有`source_history`。计数器只保留这些原始结果实际引用的规范工件，不复制无人引用的诊断文件或worker日志树。所有保留文件逐字节冻结核对；历史材料不直接取得新等级，冻结后无需继续读取原目录，原目录不被修改。

当前参考、隐藏源码、评测器和新的审核输入也会冻结。重评分必须保留原题目及样本范围，选择变化会拒绝计划。新精度、输入规格或验证规则属于新的协议，报告不把分数变化解释成因果性能提升。新的基线使用独立名称；原运行引用的全局基线不会被覆盖，计划还保存当时可取得的旧基线快照。

执行runner只调用image准备（需要时）、基线、标准/隐藏及最终/配对验证、分析。它不调用生成器或模型。阶段失败仍保留墙钟和执行日志；完成或中断过的计划不能静默重试，需另建记录。`regrade_measurements.json`单独绑定原始测量文件及本次基线，再进行分析，避免汇总与执行日志互相循环签名。模拟dispatch不能提供正式重评分资格。

新报告标记`artifact-regrade`及`new_model_calls=0`；历史生成方式和成本留作来源，不重新认证旧费用、token或公平预算。新分析和旧兼容评分门都绑定冻结计划与测量证明。已有计划可以单独分析；批量CLI不会覆盖其冻结配置或启动生成。

CUDA L2来源与计数器补充采集可直接使用这个冻结manifest，选择当前标准、隐藏、冻结或配对结果，不必再维护一份匹配的TOML。两个CLI的`--regrade-manifest`、`--scored-result`和`--result-key`均默认预览；源码/计划变更、缺少或错用当前测量会拒绝，详见[L2工件与结果选择](WORK_COUNTERS.md#cuda-l2评分组装)。

## Harbor记录重评分

```bash
python -m scripts.regrade_artifacts harbor-plan \
  --source-job jobs/old_job --config configs/a6000-restricted.toml \
  --output runs/harbor_regraded

python -m scripts.regrade_artifacts harbor-run \
  --manifest runs/harbor_regraded/harbor_regrade_plan.json
```

这两个命令默认不启动实验。未来明确执行时才加`--execute`。准备过程为完整旧cohort导出当前verifier，固定源码、任务、原trial、产物哈希、SDK及CLI身份。缺少trial、manifest、约定目录覆盖或候选时拒绝，不删掉失败记录后缩小分母。当前入口限定本地记录、单步KernelBench Pro任务和Docker provider；不自动下载Hub作业。

执行使用固定Harbor 0.24的`harbor job regrade`。上游`RegradeTrial`恢复已记录产物并使用NopAgent，source job保持不变。新结果必须逐一指向原`source_trial`，数量和任务名一致，候选字节哈希不变；旧agent成本不会被当成新调用。模拟launcher只产生审计数据。SDK、命令、source或当前verifier改变会使原计划失效。

工具语义依据[Docker create](https://docs.docker.com/reference/cli/docker/container/create/)、[禁止联网](https://docs.docker.com/engine/network/drivers/none/)及[默认seccomp](https://docs.docker.com/engine/security/seccomp/)文档。Harbor恢复接口及限制核对了[0.24.0的RegradeTrial源码](https://github.com/harbor-framework/harbor/blob/v0.24.0/src/harbor/trial/regrade.py)及安装版本；兼容性回归使用实际Task schema、bundled-tests解析、artifact覆盖函数和CLI参数定义，未启动容器。
