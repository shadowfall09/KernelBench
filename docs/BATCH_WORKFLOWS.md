# KernelBench Pro batch workflows

Local, Docker, Modal and Harbor generation share one configuration and evaluator, using local references and
hidden tests from this checkout. Compatibility entrypoints remain available.
See [NOTICE](../NOTICE.md) for code provenance.

Native local/Docker/Modal generation uses mini-swe-agent 2.4.6. Choose
`generation_mode = "one-shot"` or `"feedback"` in `[run]`; feedback defaults to
five candidate submissions and 2400 seconds per sample. Development feedback is
standard-only; hidden correctness runs after selection. See [Agent loops](AGENT_LOOPS.md)
for model adapters, resource receipts, replay behavior and the Claude Code/Codex skill.

## Paired timing and frozen final repeats

Set `baseline_portfolio = "extended"`, `paired_timing = true`, `paired_trials = 30`,
and `final_repeats = 2` in `[run]` to enable the expanded reference portfolio,
balanced same-process timing and independent frozen-source remeasurement.
All repeats are required; final failures count in correctness pass@k and score gates.
Defaults retain the standard portfolio and no additional repeats. See
[paired and final timing](PAIRED_TIMING.md) for assumptions, evidence and outputs.

Optional `score_mode = "headroom"` additionally requires `problem_specs` containing reviewed
work certificates and a `device_calibration` receipt. It uses steady-state gains with
the SOL target denominator and reports amortized bounded gain separately. Calibration is
an explicit opt-in command, never an automatic phase. See [headroom scoring](HEADROOM_SCORING.md).

## Optional memory and consistency checks

`sanitizer_tools = "all"` enables four independent CUDA correctness-tool workers;
`determinism_checks = true` adds seeded model snapshots with task-reviewed atomic
variation policies. Both are disabled by default, share the original request
timeout, and never replace uninstrumented timing samples. Missing tools or logs
remain unknown. See [tool checks](TOOL_CHECKS.md) and `configs/a6000-safety.toml`.

## Restricted verifiers and artifact regrading

`isolation_mode = "restricted"` selects the CUDA/Docker policy in
`configs/a6000-restricted.toml`. The parent pins each worker image by ID,
checks effective configuration and Linux limits, and verifies bounded result
handoff and cleanup. Scored cohorts must use the same image identity.
Source-only native/Harbor regrading uses current verifiers without another
agent run; plan and run commands preview by default. See
[isolation and regrading](ISOLATION_REGRADE.md) for platform scope and trust limits.

## Production families

Select `suite = "production41"`, `level = 5` and
`task_track = "design" | "numerical" | "implementation" | "backward"`.
The `all` selection collects all four tracks and reports them separately; its mixed macro
is diagnostic. `task_profile = "small" | "standard" | "large"` and `task_memory_mb = 512`
declare instantiation limits. Per-role storage preserves FP32 scales and discrete inputs;
fixed-format tasks may override the requested activation family. See
[production tasks](PRODUCTION_TASKS.md) for exact representations and limits.

## Install

Use Python 3.12:

```bash
uv sync --extra dev
source .venv/bin/activate
python scripts/run_batch.py --help
```

The project pins `torch==2.12.0`. Add `--extra gpu` for
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
python scripts/run_batch.py --config configs/a6000.toml \
  --run-name my_run --gpus 0,1 --profile bedrock --resume
```

TOML `[gpu]` selects the target preset, architecture and local devices; `[run]`
sets the workflow. Explicit CLI options override the file. Presets live in
`kb_workflows/hardware/gpu_profiles.toml`; custom GPUs can be configured inline. Unknown
hardware no longer falls back to Hopper, and there is no implicit H100 target.
See [GPU configuration](GPU_CONFIG.md) for the schema and precedence rules.

`--start` and `--end` are inclusive. The batch workflow supports levels 1–3. `--num-samples`
controls independent agent attempts per problem. Every attempt has its own work
directory, logs and output file. Local GPU generation runs at most one agent per
selected GPU. CPU generation for queue/Modal verification uses `--parallel`.

The default suite is `full`. Explicit `--suite kernelbench105` intersects the range
with the versioned research selection. `--suite-role scored` selects 105 tasks
across the three levels; `probe` (5) and `extended` (140) are separate runs and
are not marked main-leaderboard eligible. Baselines, agents, verification,
analysis, budgets and Harbor export use the same selected IDs. Reports carry the
manifest fingerprint; changed reference snapshots are rejected until reviewed.
I/F/D labels are provisional, and importing the selection does not calibrate
inputs or enable headroom scoring. See [research scope and examples](../research/headroom/README.md).

| Option | Choices | Purpose |
| --- | --- | --- |
| `--config` | TOML path | Load `[gpu]` and `[run]` settings |
| `--gpu-profile` | A preset name | Select one coherent GPU target, e.g. `RTX_A6000` |
| `--suite` | `full`, `kernelbench105` | Select the task manifest before intersecting the range |
| `--suite-role` | `scored`, `probe`, `extended` | Keep research roles in separate runs; `full` requires `scored` |
| `--generation-mode` | `one-shot`, `feedback`, `external` | Controlled native mode or external Harbor agent |
| `--agent-backend` | `claude-code`, `litellm` | Single-response model adapter used by mini-swe-agent |
| `--optimization-rounds` | Default `5` | Maximum candidates per feedback attempt |
| `--agent-timeout` | Default `2400` | Generation wall seconds per sample attempt |
| `--agent-cost-limit` | Default `0` | Optional estimated USD cap checked between model calls |
| `--agent-profile` | Default `true` | Separate development profiler diagnostics |
| `--generation` | `local`, `docker`, `modal`, `harbor` | Where the agent generates kernels |
| `--evaluation` | `local`, `docker`, `modal`, `queue` | Where the final evaluator runs |
| `--stage` | `all`, `generate`, `evaluate`, `analyze`, `baseline`, `export-harbor` | Run the complete pipeline, one phase, or task export |
| `--hardware` | A baseline directory label | Identify the actual verifier GPU; e.g. `RTX_A6000` |
| `--gpu-arch` | Architecture name or compute capability | CUDA compilation target; overrides the config |
| `--gpu` | Explicit cloud GPU type | Cloud GPU selection; no implicit default |
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
Select the actual verifier GPU through `[gpu]`, `--gpu-profile` or the legacy
hardware/architecture flags. Presets populate the baseline label automatically.
Custom GPUs require an explicit architecture; Modal also needs a cloud GPU type.

## Evaluation is the same on every backend

Every run defaults to two phases:

1. Standard correctness, synchronized wall timing and peak GPU memory measurement; every timed output is verified on freshly generated inputs.
2. Hidden correctness using the repository's four input distributions regenerated under three seeds, plus domain-reviewed stress cases; no repeated timing.

Results use flat `"problem_id_sample_id"` keys, so its analysis and
leaderboard scripts can read them directly. Reported correctness is gated by the
hidden result while runtime and memory come from the standard result. A missing
hidden result counts as incorrect.

Correctness runs use full FP32 arithmetic. Explicit single-baseline FP32 timing enables TF32 for
matmul and cuDNN; the default portfolio uses IEEE FP32 for all variants. FP16/BF16 baselines have separate names. New PyTorch precision
controls are used on supported versions; see the
[PyTorch CUDA documentation](https://docs.pytorch.org/docs/2.12/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices).

The default FP32 baseline is `baseline_time_suite_fp32_wall` (eager, compile default, compile max-autotune and CUDA Graph). Suite timing uses IEEE FP32; explicitly select `--no-baseline-suite` for a single TF32 baseline. Old measurements from the
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
sbatch --partition=taurus --account=yichengtao --gpus=1 \
  --export=ALL,KB_CONFIG=configs/a6000.toml,RUN_NAME=slurm_test \
  batch_run.sbatch
```

The script uses `SLURM_SUBMIT_DIR` instead of a hardcoded repository path and
honors the allocated `CUDA_VISIBLE_DEVICES`. Set `KB_REPO_DIR` if submitting from
another directory, and `KB_PYTHON` to select a Python environment. `GENERATION`
and `EVALUATION` select the same backends as the unified CLI. With `KB_CONFIG`,
only explicitly supplied environment settings override the file. Without a config,
the default is Docker; no default architecture is injected.

## Persistent local verification queue

Start one worker inside a device allocation, using the shared hardware config:

```bash
python -m kb_workflows.runtime.queue_worker --config configs/trainium2.toml \
  --state cache/verify_queue/neuron.sqlite3 --dry-run
```

The preview imports no device SDK and creates no queue state. To serve requests,
remove `--dry-run` and set the appropriate `--host`/`--port`. CUDA uses the same
entrypoint with a CUDA TOML; the legacy `python scripts/local_verify_worker.py
--gpu 0` command remains supported. A worker selects one allocation-relative
device; pass `--device` to override a config containing multiple devices.
Only device settings are loaded from the shared config, so controller calibration
files and model settings need not exist on the worker. `device.python` selects
the worker's own vendor environment before SDK imports; clients cannot select
its interpreter. The actual backend must be available before the listener starts.

The HTTP interface is:

- `POST /v1/verify` submits an evaluation or baseline task.
- `GET /v1/verify/{request_id}` returns its state and result.
- `GET /health` reports worker/queue health and protocol capabilities.
- `POST /v1/verify/{request_id}/cancel`, with `{"request_sha256": "..."}`, cancels the matching saved request. The submit/get response supplies its fingerprint.

Jobs, cancellation states and lease nonce bindings persist in `cache/verify_queue/jobs.sqlite3`. Restarted unfinished leased/cancelling jobs fail with unknown usage; unleased running jobs retain the old requeue behavior. Identical requests are idempotent; the client retries transient HTTP failures. One worker processes one task at a time, and `--capacity` bounds
pending tasks. Use a different `--state` file per worker device. Each database
is bound to its backend; an existing CUDA database cannot be reopened as Neuron.

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

`--queue-wait-timeout` bounds submission/retries/polling, separate from each device task's `--timeout`. On timeout or lost communication the client requests cancellation, retaining the failed call and any confirmed server-parent usage. With `agent_device_seconds`, the worker also expires queued/running leases without requiring a live client. Budgeted requests and all Neuron requests require a healthy `kernelbench_pro_queue_v3` worker with a matching backend; older workers must be upgraded. Legacy unbudgeted CUDA services remain compatible. See [resource budgets](RESOURCE_BUDGETS.md) for accounting and cleanup limits.

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

Modal images include the repository references, hidden tests and evaluator. Sandbox
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
  --no-baseline-suite --compile-backend inductor --compile-mode max-autotune

python scripts/run_batch.py --stage analyze --level 1 --start 1 --end 10 \
  --run-name my_run --hardware RTX_A6000
```

The subset is used as the denominator for batch analysis. Native analysis retains
its full-dataset default when no subset is given. It prints the first 30 per-sample
timing/memory comparisons and writes `analysis_summary.json`. Invalid timing
values are excluded from speedup calculations. Memory and speed metrics retain
the inherited metric formulas.

| Output | Contents |
| --- | --- |
| `runs/NAME/run_config.json` | Shared configuration |
| `runs/NAME/generation_summary.json` | Per-attempt generation status and checksums |
| `runs/NAME/level_L_problem_P_sample_S_kernel.py` | Generated kernels |
| `runs/NAME/eval_results.json` | Standard correctness, timing and memory |
| `runs/NAME/eval_results_hidden.json` | Hidden correctness results |
| `runs/NAME/pass_at_k_results.json` | Hidden-gated pass@k; missing samples count as failures |
| `runs/NAME/analysis_summary.json` | Inherited benchmark metrics and per-problem comparisons |
| `runs/NAME/work/pP_sS/` | Agent workspace, staged reference and logs |
| `runs/NAME/logs/` | Local/Docker evaluator logs |
| `results/timing/HARDWARE/BASELINE.json` | Baseline timing, memory and environment information |

Pass@k defaults to the values 1 and 5 where at least k samples are requested.
Override with `--pass-at-k 1,2,5`. Leaderboard generation can
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

## Wall timing and startup costs

Batch candidate and baseline timing synchronizes the device before and after each
call and consumes its outputs. Input generation and full reference comparisons are
outside the interval. Each timed call uses a new seed with the trusted input
factory, and all warmup/timed outputs are verified. A fixed input factory may still
produce identical contents. CUDA Graph input copies and stable output copies are
included in replay time. This adds host/synchronization overhead to short kernels.

Results retain raw samples and normal-approximation 95% mean intervals. The cold
cost is observed setup, first forward and warmup, with existing compiler caches;
it excludes process startup, input preparation and correctness validation overhead.
`--cold-calls K` defaults to 10000 and reports `(cold + (K - 1) * steady) / K`.
The portfolio retains steady and amortized winners; aggregate scoring is separate.
New baseline filenames end in `_wall`; do not reuse CUDA-event baselines.

The added regression suite passes on CPU and simulated CUDA scheduling. Real GPU,
compile, CUDA Graph and the pinned Torch 2.12.0 toolchain still require validation.

## Failure-inclusive scoring and recovery

`--stage analyze` first writes `paper_analysis.json` and `eval_results_scored.json`,
then produces the compatible benchmark summary when its inputs are available. Keep
the same declared range, sample count, precision, backend and call budget across
stages. Every declared problem/sample slot remains in the new score denominator;
failed, missing, stale or incompatible slots earn zero. Eligible variants are
reselected by amortized cost at K, and a positive bounded log-gain score additionally
requires non-overlapping steady mean intervals. Inherited metric formulas remain.

Optional `--problem-specs FILE` supports `{"level1":{"1":{"layer":"F"}}}` for
I/F/D macro aggregation; otherwise entries are `unclassified`. Bootstrap intervals
resample problems and declared sample slots. This version does not calibrate devices
or certify cross-device scores.

Compatibility checks require matching recorded device/software, reference/kernel
sources, evaluation settings, wall-timing protocol, input signatures, varied-input
verification receipts, raw timing samples and cold costs. Recovery fingerprints
include the measurement, validation, baseline, configuration, orchestration,
transport and analysis modules. Rebuild remote images/workers after changes.

New batch results require scored evidence in the original analysis and leaderboard
entrypoints as well. Their gate rejects changed or missing kernels, reference or
hidden source, configuration, raw results, baseline records and evaluator versions.
Legacy runs without this policy retain their original behavior. Correctness/pass@k
and performance eligibility remain separate. Additional integrity checks, sanitizers and independent retests are deferred;
reports explicitly do not certify anti-cheat.


## Research-informed correctness checks

Standard and hidden correctness now compare both candidate→reference and
reference→candidate pairs with isolated inputs, output snapshots and RNG replay.
Hidden generators run separately under seeds 42, 123 and 456. For 29 reviewed
Level 1 tasks, the first hidden configuration additionally exercises zeros,
32× values and sparse Gaussian outliers. Empty suites and reference failures
cannot count as successful validation. These checks are enabled in both native
and batch evaluation; `--no-hidden-tests` disables the hidden stage as before.

`validation_summary.json` reports compilation, standard/hidden correctness and
positive finite timing counts over all declared samples, including missing ones.
It is a raw diagnostic, not the performance score or an anti-cheat certificate.
The existing scored-results gate still controls performance eligibility.
New protocol code participates in the evaluator digest, invalidating old resume
records. Verifier images/checkouts must be updated together with the client.

See [sources, task coverage, costs and limits](RELIABILITY.md). The original
four-config hidden suite now runs 12 configuration checks, or 21 for adapted
tasks; each check has two paired calls per implementation. Performance inputs
and the synchronized wall timing protocol remain as documented above.


## Local project names

The Python distribution is `kernelbench-pro`; CLI commands remain
`kb-batch` and `kb-verify-worker`. New Docker runs default to `kernelbench-pro:agent`,
Modal uses `kernelbench-pro`, and Slurm jobs use `kernelbench-pro`.
Rebuild the Docker image after this update. To use a previously built image,
select its tag explicitly with `--docker-image` and ensure its evaluator version
matches the client. Evaluator/configuration fingerprints will invalidate old
cached measurements when affected code or settings change.

`paper_analysis.py` and `paper_analysis.json` keep their existing names for
compatibility; they implement the active scoring workflow and are not stored
paper assets. The results page uses the requested hardware label and does not
claim an evaluation date or a fixed sample count.


## Harbor and Python 3.12

The current project requires Python 3.12. Docker uses Python 3.12 and a CUDA
Ubuntu 24.04 base; Modal uses `add_python="3.12"`. Historical validation records
above describe their original Python 3.10 environments, not the current requirement.

`--generation harbor` reuses the native generation slots, GPU lanes, recovery and
formal evaluation/analysis pipeline. Install `uv sync --extra dev --extra harbor`,
configure the Harbor agent credentials, and use `--harbor-agent`,
`--harbor-environment` and `--harbor-reward` as needed. The `export-harbor` stage
uses the same level/range/sample/precision/trial settings without starting an agent.
See [Harbor integration](HARBOR.md) for commands, task layout and validation limits.
