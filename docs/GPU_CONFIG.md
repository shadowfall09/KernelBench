# GPU 配置

CUDA沿用本文的`[gpu]`预设。通用`[device]`配置及Trainium2/3的原生NKI工作流见[Neuron说明](NEURON.md)；两种配置节不能同时声明一个目标，CUDA设备镜像不会被当作Neuron环境。

批量入口通过 TOML 的 `[gpu]` 配置目标设备，通过 `[run]` 配置题目与工作流。配置解析只依赖 Python 3.12 标准库，预览和 Harbor 任务导出不需要导入 Torch、探测 GPU 或调用云服务。

```toml
[gpu]
profile = "RTX_A6000"
devices = [0, 1]

[run]
run_name = "a6000_run"
generation = "docker"
evaluation = "docker"
level = 1
start = 1
end = 10
precision = "fp32"
resume = true
```

仓库提供单卡 [a6000.toml](../configs/a6000.toml) 和云端验证 [h100-modal.toml](../configs/h100-modal.toml) 示例。先复制或按实际环境修改，再运行：

```bash
python scripts/run_batch.py --config configs/a6000.toml --dry-run
python scripts/run_batch.py --config configs/a6000.toml --resume
```

## 型号与自定义目标

内置型号集中维护在 [gpu_profiles.toml](../kb_workflows/gpu_profiles.toml)，随 Python 包一起分发。当前提供 A10G、A100、A100-80GB、RTX_A6000、L4、L40S、H100、H200 和 T4；这是配置预设列表，不表示所有型号都已在当前 CUDA 镜像上完成实机验证。

每个预设用 `architecture` 描述编译目标，支持云端的预设另有 `modal_type`。预设名用作基线目录标签。增加常用型号可以直接修改 TOML；临时目标直接在运行配置中填写，无需修改 Python 代码：

```toml
[gpu]
hardware = "My_GPU"
architecture = "8.6"
devices = [0]
# 使用 Modal 时，另填该服务接受的 GPU 类型：
# modal_type = "provider-gpu-name"
```

| `[gpu]` 字段 | 含义 |
| --- | --- |
| `profile` | 可选，选择内置型号，大小写与预设一致 |
| `hardware` | 基线目录标签；省略时使用预设名 |
| `architecture` | PyTorch CUDA 架构名或计算能力，如 `Ampere`、`8.6`、`8.6+PTX`；多个目标用逗号或分号分隔 |
| `devices` | 本地 GPU 序号列表，必须非空、不重复且为非负整数；省略时为 `[0]` |
| `modal_type` | 云端 GPU 选择值；没有该值的预设不会隐式租用 H100 |

文件先展开预设，再应用同一个 `[gpu]` 中显式填写的字段。架构语法在配置解析时检查，具体架构是否可编译仍取决于安装的 Torch/nvcc 和目标硬件。

未知预设会报错。未知硬件标签必须显式提供架构，不再回退为 Hopper。执行生成、验证、基线或 Harbor 导出前必须有硬件标签和架构；Modal 任务还必须有云端类型。仅 `--dry-run` 可保留未选定的 GPU，且不会自动补入 H100。

## 运行参数与覆盖

`[run]` 使用 `RunConfig` 的参数名，例如 `num_samples`、`harbor_agent`、`hidden_tests`、`parallel`。GPU 字段集中放在 `[gpu]`，不能在 `[run]` 再写 `gpu`、`gpus`、`gpu_arch` 或 `hardware`。未知字段、拼写错误、错误类型和非法枚举值会在创建运行目录或调用后端前报错。

优先级为：**显式命令行参数 > 文件配置 > 程序默认值**，包括 `--no-resume` 等布尔覆盖。文件中的相对 `runs_dir` 和 `problem_specs` 路径相对于配置文件目录解析；命令行相对路径仍相对于执行目录。

```bash
# 同一份配置换为 H100；同时替换硬件标签、架构和云端类型。
python scripts/run_batch.py --config configs/a6000.toml --gpu-profile H100 --dry-run

# 切换生成后端，仍使用配置中的目标 GPU。
python scripts/run_batch.py --config configs/a6000.toml \
  --generation harbor --harbor-environment docker --model 'your-provider/your-model'

# 使用配置中的 GPU 导出自包含 Harbor 任务。
python scripts/run_batch.py --config configs/a6000.toml \
  --stage export-harbor --run-name harbor_export
```

旧的 `--hardware`、`--gpu-arch`、`--gpus` 和 `--gpu` 仍可使用。若命令行显式改变 `--hardware` 或 `--gpu`，文件中的硬件标签、架构和云端类型会一起清除，再按新目标解析，避免 H100 继承 A6000 的架构；自定义目标需同时传 `--gpu-arch`。单独 `--gpu-profile` 可以整体切换目标，保留文件中的设备序号。

`batch_run.sh` 和 `batch_run_modal.sh` 收到 `--config` 后使用文件中的后端选择；没有文件时保留旧的 Docker/Modal 默认行为。Slurm 使用：

```bash
sbatch --gpus=1 --export=ALL,KB_CONFIG=configs/a6000.toml batch_run.sbatch
```

`KB_CONFIG` 模式下，只有显式提供的 `GPUS`、`GPU_PROFILE`、`GPU_ARCH`、`HARDWARE`、`GENERATION` 等环境变量覆盖配置，脚本不再注入默认 Ampere。`devices` 相对于 Slurm 分配后的 `CUDA_VISIBLE_DEVICES`，不会绕过作业分配。

## 任务、基线与兼容范围

解析后的具体 GPU 参数写入 `run_config.json`，再传递给本地子进程、Docker、Modal、队列或 Harbor。子任务使用展开后的 JSON，不依赖宿主配置文件路径；Harbor verifier 包含预设文件。Harbor Docker 对没有 `modal_type` 的 GPU 省略云端类型约束，按分配的 GPU 绑定设备。

基线继续按 `results/timing/HARDWARE/` 分开保存。切换目标会改变生成和评测指纹；GPU 配置解析器和预设内容也纳入评测器摘要。评分仍核对实际设备/软件指纹，不能靠改标签把别的硬件成绩当作匹配基线。

一个运行配置描述一种 GPU 类别，多张同类卡可以并行。异构卡使用独立配置、运行目录和基线。当前执行器仍基于 NVIDIA CUDA；新增配置入口没有增加 AMD、Intel、MPS 或 CPU 性能评测后端，也没有解决 CUDA 镜像与旧硬件的工具链兼容问题。
