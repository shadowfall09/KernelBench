# Unified batch workflows

The workflows migrated from `KernelBench-modified` share one configuration and one
Verified evaluator. The original Verified generation and evaluation scripts remain
available. New workflows use local references and hidden tests from this checkout.

## Install

Use Python 3.10, as in the original repository:

```bash
uv sync --extra dev
source .venv/bin/activate
python scripts/run_batch.py --help
```

The project keeps upstream's `torch==2.12.0` requirement. Add `--extra gpu` for
optional Triton, CuTe, TileLang, CuPy and Nsight Python packages. A local verifier
needs a CUDA GPU and a compatible CUDA toolkit/compiler for generated extensions.
Local generation also needs the Claude Code CLI. Docker and Modal generation
images install that CLI themselves.

Authenticate Claude Code through an API key, or use Bedrock with `--profile NAME`
after logging into that AWS profile. `--model` and `--small-model` can override
model choices. Credentials are passed at runtime; they are not saved in run JSON
or baked into cloud images.

## One entrypoint, independent generation and verification

```bash
python scripts/run_batch.py \
  --generation docker --evaluation docker \
  --level 1 --start 1 --end 10 --run-name my_run \
  --hardware RTX_A6000 --gpu-arch Ampere --gpus 0,1 \
  --parallel 2 --profile bedrock --resume
```

`--start` and `--end` are inclusive. Verified supports levels 1–3. `--num-samples`
controls independent agent attempts per problem. Every attempt has its own work
directory, logs and output file. Local GPU generation runs at most one agent per
selected GPU. CPU generation for queue/Modal verification uses `--parallel`.

| Option | Choices | Purpose |
| --- | --- | --- |
| `--generation` | `local`, `docker`, `modal` | Where Claude Code generates kernels |
| `--evaluation` | `local`, `docker`, `modal`, `queue` | Where the final evaluator runs |
| `--stage` | `all`, `generate`, `evaluate`, `analyze`, `baseline` | Run the complete pipeline or one phase |
| `--hardware` | A baseline directory label | Identify the actual verifier GPU; e.g. `RTX_A6000` |
| `--gpu-arch` | `Ampere`, `Ada`, `Hopper`, etc. | CUDA compilation target |
| `--gpu` | Modal GPU type; default `H100` | Cloud GPU selection |
| `--gpus` | Comma-separated indices | Local GPUs, relative to `CUDA_VISIBLE_DEVICES` |
| `--precision` | `fp32`, `fp16`, `bf16` | Computation precision |
| `--fp32-tolerance` | Default `1e-3` | FP32 correctness tolerance, consistent across backends |
| `--hidden-tests` | Default `true` | Perform hidden correctness gating |
| `--resume` | Default `false` | Reuse matching generation/evaluation results |
| `--enable-ncu` | Default `true` | Allow agent profiling when the generation GPU has `ncu` |

Add `--dry-run` to print the selected problems, backends and output directory without
creating files, building images, contacting Modal or calling a model.

The `all` stage records missing baselines, generates kernels, evaluates them and
analyzes the selected range. A baseline must exist for every selected problem;
missing or failed baseline measurements cause an explicit analysis error.
`--hardware` is required for `all`, `baseline` and `analyze` with local/queue GPUs.
Modal defaults that label to `--gpu`.

## Verified evaluation is the same on every backend

Every run defaults to two phases:

1. Standard correctness, CUDA-event timing and peak GPU memory measurement.
2. Hidden correctness using the repository's four input distributions; no repeated timing.

Results use Verified's flat `"problem_id_sample_id"` keys, so its analysis and
leaderboard scripts can read them directly. Reported correctness is gated by the
hidden result while runtime and memory come from the standard result. A missing
hidden result counts as incorrect.

Correctness runs use full FP32 arithmetic. FP32 baseline timing enables TF32 for
matmul and cuDNN. FP16/BF16 baselines have separate names. New PyTorch precision
controls are used on supported versions; see the
[PyTorch CUDA documentation](https://docs.pytorch.org/docs/2.12/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices).

The default FP32 baseline is `baseline_time_torch_tf32`. Old measurements from the
modified repository are not copied: they used different reference problems,
precision settings and hardware. Record a baseline on the actual verifier GPU.

Resume checks reference code, kernel code, evaluator source, hidden test source,
precision, tolerance and evaluation settings. Editing a kernel invalidates its
standard and hidden results. Generation resume also checks the model/prompt and
the generated file's checksum. Existing legacy files without these fingerprints
are evaluated again. Matching recorded failures are retained; use a new kernel,
changed settings or omit `--resume` to retry them.

Agent prompts use the canonical input-blind problem list in
`src/prompt_constructor.py`. Those staged `model.py` files contain imports and
model classes without test inputs. Hidden tests never appear in the prompt.
The agent is instructed to use this staged source and `python verify.py`.
This preserves prompt input blindness; local/Docker/Modal agents still have
filesystem tools, so this is not a security boundary against a hostile agent.

## Local and Docker

```bash
# Existing kernels, evaluated on a locally allocated GPU.
python scripts/run_batch.py --stage evaluate --evaluation local \
  --level 1 --start 1 --end 10 --run-name my_run --gpu-arch Ampere --resume

# Full Docker generation and verification, using the old shell entrypoint.
./batch_run.sh --level 1 --start 1 --end 10 --run-name my_run \
  --hardware RTX_A6000 --gpu-arch Ampere --gpus 0,1 --profile bedrock
```

Docker images contain the current checkout, including the migrated code. The
single Dockerfile has `agent` (CUDA toolkit) and `agent-remote` (CPU generation)
targets. Source is copied during the build; rebuild after changing evaluator code.
`--no-build-image` reuses an image you already built.

Each evaluation uses an isolated subprocess and compilation directory. Timeouts
terminate the process group, including compilation children. Local GPU tasks are
serialized per GPU while different GPUs run in parallel. Docker verification
also removes its container on timeout.

## Slurm

Submit from the checkout root, selecting your cluster's partition/account:

```bash
sbatch --partition=taurus --account=yichengtao --gpus=2 \
  --export=ALL,LEVEL=1,START=1,END=10,HARDWARE=RTX_A6000,GPU_ARCH=Ampere,RUN_NAME=slurm_test \
  batch_run.sbatch
```

The script uses `SLURM_SUBMIT_DIR` instead of a hardcoded repository path and
honors the allocated `CUDA_VISIBLE_DEVICES`. Set `KB_REPO_DIR` if submitting from
another directory, and `KB_PYTHON` to select a Python environment. `GENERATION`
and `EVALUATION` select the same backends as the unified CLI. The default is Docker.

## Persistent local verification queue

Start one worker inside a GPU allocation:

```bash
python scripts/local_verify_worker.py --host 0.0.0.0 --port 8000 --gpu 0
```

The HTTP interface is unchanged:

- `POST /v1/verify` submits an evaluation or baseline task.
- `GET /v1/verify/{request_id}` returns its state and result.
- `GET /health` reports worker/queue health.

Jobs and results persist in `cache/verify_queue/jobs.sqlite3`. Restarted running
jobs are requeued. Identical requests are idempotent; the client retries transient
HTTP failures. One worker processes one task at a time, and `--capacity` bounds
pending tasks. Use a different `--state` file per worker GPU.

```bash
./batch_run_queue.sh --level 1 --start 1 --end 10 --run-name queue_test \
  --hardware RTX_A6000 --gpu-arch Ampere --parallel 4 \
  --queue-url http://WORKER_HOST:8000 --queue-max-inflight 8 --resume
```

Use the worker's reachable hostname when generation and verification are on
different hosts. When both are on the same host, the Docker generation client
translates `127.0.0.1`/`localhost` to `host.docker.internal`. The worker must listen
on a reachable interface for containers to connect. The client and worker must
use the same reference, evaluator and hidden-test sources.

`--queue-wait-timeout` bounds total polling time, separate from each GPU task's
`--timeout`. Requests that outlive the client remain in the persistent worker
store and can be fetched/reused later.

The compatibility Python entrypoint still accepts common pydra options:

```bash
python scripts/eval_from_generations_queue.py \
  run_name=queue_test dataset_src=local level=1 subset='(1,10)' \
  gpu_arch="['Ampere']" queue_base_url=http://WORKER_HOST:8000
```

## Modal

Authenticate Modal locally first. Docker CPU generation plus cloud verification:

```bash
./batch_run_modal.sh --level 1 --start 1 --end 10 --run-name modal_test \
  --gpu H100 --parallel 4 --profile bedrock --resume
```

For generation inside GPU-enabled Modal Sandboxes:

```bash
python scripts/run_batch.py --generation modal --evaluation modal \
  --level 1 --start 1 --end 10 --run-name sandbox_test \
  --gpu H100 --parallel 4 --modal-secret MY_MODEL_CREDENTIALS --resume
```

Modal images include Verified's references, hidden tests and evaluator. Sandbox
agents use their allocated GPU for iterative verification; final verification
uses the selected evaluation backend. Kernels and logs are downloaded before
sandbox termination. Credentials use a named Modal secret, or environment
credentials; `--profile` resolves AWS credentials at runtime for SSO/Bedrock.

`scripts/batch_run_modal.py` is a compatibility wrapper for the same workflow and
accepts options such as `level=1 start=1 end=10 gpu=H100 run_name=sandbox_test`.

## Baselines, analysis and output files

```bash
python scripts/generate_baseline_time.py --evaluation local \
  --level 1 --start 1 --end 10 --hardware RTX_A6000 --gpu-arch Ampere

# Optional torch.compile baseline; receives a distinct default filename.
python scripts/generate_baseline_time.py --evaluation local \
  --level 1 --start 1 --end 10 --hardware RTX_A6000 --gpu-arch Ampere \
  --compile-backend inductor --compile-mode max-autotune

python scripts/run_batch.py --stage analyze --level 1 --start 1 --end 10 \
  --run-name my_run --hardware RTX_A6000
```

The subset is used as the denominator for batch analysis. Native analysis retains
its full-dataset default when no subset is given. It prints the first 30 per-sample
timing/memory comparisons and writes `analysis_summary.json`. Invalid timing
values are excluded from speedup calculations. Memory and speed metrics retain
Verified's formulas.

| Output | Contents |
| --- | --- |
| `runs/NAME/run_config.json` | Shared configuration |
| `runs/NAME/generation_summary.json` | Per-attempt generation status and checksums |
| `runs/NAME/level_L_problem_P_sample_S_kernel.py` | Generated kernels |
| `runs/NAME/eval_results.json` | Standard correctness, timing and memory |
| `runs/NAME/eval_results_hidden.json` | Hidden correctness results |
| `runs/NAME/pass_at_k_results.json` | Hidden-gated pass@k; missing samples count as failures |
| `runs/NAME/analysis_summary.json` | Native Verified metrics and per-problem comparisons |
| `runs/NAME/work/pP_sS/` | Agent workspace, staged reference and logs |
| `runs/NAME/logs/` | Local/Docker evaluator logs |
| `results/timing/HARDWARE/BASELINE.json` | Baseline timing, memory and environment information |

Pass@k defaults to the values 1 and 5 where at least k samples are requested.
Override with `--pass-at-k 1,2,5`. Existing Verified leaderboard generation can
consume the result files with `--use_hidden_eval` and a matching TF32 baseline.

## Validation

```bash
python -m pytest -q tests src/unit_tests/test_dataset.py src/unit_tests/test_score.py
python -m ruff check kb_workflows tests
docker build --check --target agent .
bash -n batch_run.sh batch_run_modal.sh batch_run_queue.sh batch_run.sbatch
```

Workflow tests exercise a real HTTP server with a simulated GPU evaluator,
persistent queue restart, hidden gating, evaluator API compatibility, baseline
memory fields, pass@k, subset analysis, changed-kernel resume and failed overwrite.
They do not invoke a model or rent cloud GPUs. A real CUDA/cloud execution remains
necessary to validate toolchain compatibility and measured performance on each
deployment environment.

The migration was validated with 17 workflow tests and 4 upstream dataset/score
tests, using a temporary Python 3.10 environment with the existing Torch 2.9.0
installation. Dependency resolution for the project's pinned Torch 2.12.0 passed.
Both Docker targets passed build-definition checks, and Modal image definitions
were constructed with the installed SDK. A Slurm GPU smoke-test allocation was
unavailable because the requested nodes were busy; no real GPU, model-generation
or Modal execution is claimed by these checks.
