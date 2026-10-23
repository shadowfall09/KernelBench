---
name: kernelbench-pro
description: Generate, optimize, or evaluate GPU kernels with KernelBench Pro using its GPU configs, bounded feedback and final hidden verification. Use for benchmark runs or implementing an individual KernelBench task.
---

# KernelBench Pro

Work from the checkout root with Python 3.12. Read `docs/AGENT_LOOPS.md` for the chosen mode and `docs/GPU_CONFIG.md` when selecting hardware. For a Neuron target, read `docs/NEURON.md` and start from `configs/trainium2.toml`. Reuse the user's configured device, backend, model and budgets. Local GPU work runs inside an existing Slurm allocation; configuration preview does not need a device.

## Run an automated experiment

Use `configs/a6000-one-shot.toml` or `configs/a6000-feedback.toml` as a starting point, adjusting the GPU to the actual verifier. Preview with `python scripts/run_batch.py --config FILE --dry-run`, then use the same config to run. CLI overrides are supported. Independent samples use `num_samples`; feedback candidates use `optimization_rounds`. Defaults for feedback are five candidates and 2400 seconds per sample.

The automated loop uses mini-swe-agent. Claude Code and LiteLLM are model adapters; the controller owns evaluation and source selection. Inside an active Claude Code session, use the interactive route below to implement the task yourself, or use an already configured LiteLLM provider for automation. Do not bypass Claude's nested-session guard or change authentication settings to launch another Claude process.

## Implement a task in this coding session

Prepare one task without launching a nested model:

```bash
python -m kb_workflows.agent_task prepare --config configs/a6000-feedback.toml \
  --generation local --evaluation local --start 90 --end 90 --num-samples 1
```

Choose the task and verifier backend from the user's request. For Docker verification, build the current project image first; queue and Modal verification use their configured credentials/services. The command prints the task directory, public reference, output path and feedback command.

Read only the supplied `model.py` for the task and implement `ModelNew` at the printed candidate path. Use `python verify.py` from the task directory for development feedback. Inspect correctness before speed; use profiler events to form a concrete bottleneck hypothesis, change the implementation, and resubmit within the configured candidate/time budget. Keep notes on changes that helped or regressed. The bridge retains the best development-correct source at finalization.

When `device_backend=neuron`, keep the `torch.nn.Module` interface and use native NKI kernels registered through `torch_neuronx.nki_op`; the verifier supplies `neuron` tensors. Use the configured vendor interpreter for device verification. CPU or NumPy proxy execution, legacy XLA packages, CUDA images and simulated timings do not establish Neuron performance. Preserve role dtypes and keep unsupported FP8 storage explicit.

For one-shot, produce one implementation and finalize without calling `verify.py`; execution feedback is disabled. A live coding session has unobserved model/tool calls, so it is recorded as `interactive`, not as a measured one-shot experiment.

Freeze the chosen source:

```bash
python -m kb_workflows.agent_task finalize --directory TASK_DIRECTORY
```

Then run the native `evaluate` and `analyze` stages with the original parent run name, task selection, hardware and precision. Use `baseline` first if there is no current matching baseline. These final stages perform standard and hidden verification. Do not revise the frozen candidate in response to held-out failures within the same experiment; start a separately labeled run for further development.

## Preserve the experiment contract

- Do not inspect hidden tests, recover stripped inputs, change tolerances, or modify the evaluator/reference to improve a result. Preserve output structure, shape, dtype, parameters and input-dependent behavior.
- Development feedback uses standard tests only. Final hidden checks do not enter the automated conversation or candidate selection.
- For restricted verification or rule/library upgrades, read `docs/ISOLATION_REGRADE.md`. Keep the configured Docker/Harbor restrictions and use frozen-artifact regrading when the request is to reassess saved candidates. Its CLI previews by default; use `--execute` only when running experiments is within the user’s requested scope.
- For execution-origin work, read `docs/KERNEL_ORIGIN.md`. Its CUDA/Triton collector runs separately and previews by default. Wheel-bound cubins, immutable Graph replay, observed Triton frontend/emission and CUDA object builds have diagnostic adapters. Keep full CUDA source literals and the reviewed compiler options when using that observer; outer wrappers, replaced artifacts and cache-only records confer no provenance. Unsupported Runtime loaders, updates, compilers and Neuron origins remain unknown. Never turn a source declaration, diagnostic track verdict or custom fraction into formal L1 evidence or replace scored times with profile durations.
- When configured, preserve sanitizer and repeat checks. Read `docs/TOOL_CHECKS.md`; do not disable tools, edit task atomics policies or treat missing diagnostics as passing. Instrumented times never enter performance scores.
- Preserve configured `state_validation`. Read `docs/STATE_VALIDATION.md` when renaming or packing state: use the declarative `kernelbench_state_map` for registered tensors and supported storage views. Preserve tied state and cache invalidation under updates; a mapping cannot waive comparisons. Missing adapters for plain tensor attributes or nonlinear representations are explicit failures, not passing state evidence.
- Preserve configured `numerical_validation` and `oracle_memory_mb`; read `docs/NUMERICAL_VALIDATION.md`. Its L1 explicit formulas and source-bound L2 composite arithmetic check actual native input sizes and require reference self-qualification. Read `docs/COMPOSITE_NUMERICS.md` for frozen constructor plans and native dropout-mask replay; do not disable dropout, substitute small unit-test configurations, or fall back to reference double-forward. L3 remains unsupported. Never resize inputs, loosen error budgets or replace scored samples to obtain a passing diagnostic. Preserve explicit reference defects and unknown/over-budget cases; full L0 and device calibration remain separate requirements.
- When `input_calibration` is configured, read `docs/INPUT_CALIBRATION.md`. Keep its source/precision/evaluator bindings and fixed error policy; record matching new baselines and preserve per-output FP64 checks. A transferred value distribution is not device measurement evidence. Calibration previews by default; never add `--execute` when the user requested implementation only. Do not transform held-out factories to obtain passing results.
- Read `docs/OPERATOR_INPUTS.md` for default L1/L2/L3 hidden families. Preserve constructor parameters, sequence/state/residual shape coupling, discrete roles and positive strides. L3/33/48/49 have explicit fixed-shape reasons; never invent broadcast or replace state to shrink them. Keep `input_family_memory_mb` and full case/seed/native-call receipts; L2/L3 replay also checks exact shapes and dtype. Never skip an over-budget case or resize LayerNorm parameters. NaN/Inf cases apply only to declared tasks; null finite-error measurements do not supply reference signal.
- Preserve configured `allocation_validation`; read `docs/ALLOCATION_VALIDATION.md`. Its two fresh CUDA correctness workers poison observed candidate allocations across import, construction and native calls. Keep both patterns, environment, case/order receipts and the final unpoisoned control. Their timings never replace scored samples. Raw allocators, threads, Graph buffers and unsupported paths are not full device-memory evidence; do not disable compilation or deterministic policy to obtain a passing control. Neuron rejects this option.
- Preserve `throughput_validation`, calls/repeats and storage budget; read `docs/THROUGHPUT_VALIDATION.md`. Fresh public-only latency/burst workers use matching input bytes and stable per-call snapshots, with continuous state/RNG and mode/gradient/thread checks. Both modes include snapshot-copy costs; these times never replace native samples. A >10% review flag excludes qualified performance credit without relabeling the saved native answer as cheating. Do not drop snapshots, shrink nominal workloads, turn on launch blocking, reuse hidden factories or erase flags to qualify.
- Read `docs/REFERENCE_REVISIONS.md` for repaired tasks. Keep current reference, hidden-source and shared-input-generator bindings; do not relabel a legacy positive result as a current revision. Preserve reviewed scan cases, empty-output/null semantics, normalized KL probability transforms and ±1 hinge labels. State/numerical policies and oracle memory also belong to frozen repeats and experiment comparison groups.
- Profiler durations diagnose execution; final performance comes from the native synchronized-wall benchmark and matching baseline.
- Read `paper_analysis.json`, `generation_budget.json` and `phase_budget.json` when reporting results. Missing costs/tokens/GPU billing remain unknown. Generated source, development correctness and final verified speedup are separate claims.
- Compare matching settings with `python -m kb_workflows.experiment RUN_A RUN_B`. Keep one-shot, feedback, external Harbor and interactive sessions separate. Report independent sample count and per-sample limits alongside outcomes.
