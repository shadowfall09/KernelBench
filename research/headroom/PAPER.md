# Speedup Is Not Skill: Toward Reliable Evaluation of Agentic Kernel Engineering

**KernelBench Pro: an implemented workflow and a headroom-oriented research design**

Draft, unified on 2026-10-21. This is the current research manuscript. The implementation baseline is commit `92c27f4`; the accompanying integration adds the research selection and traceability records. [Motivation](../../docs/MOTIVATION.md), the [change ledger](../../docs/motivation_map.json), and the [version inventory](../../docs/FEATURE_INVENTORY.md) distinguish implementation from proposed mechanisms. The earlier HR-Eval and HEADROOM-Bench names denote the target evaluator and benchmark designs within KernelBench Pro. They do not denote two completed releases.

## Abstract

Evaluating generated GPU kernels requires more than measuring speedup over one PyTorch reference. Repeated tasks consume evaluation budgets; loose numerical checks can accept incorrect programs; baseline choice and startup accounting change the interpretation of speedup; and one-shot generation does not measure how an agent uses execution feedback. Reproducible infrastructure is necessary to study these questions at scale.

We organize KernelBench Pro around five motivations: task discrimination, correctness and execution validity, interpretable performance measurement, realistic agent interaction, and reproducible infrastructure. The current implementation combines the KernelBench-Verified task and hidden-test workflow with independent generation and verification backends, stricter output comparison, paired-order checks, multi-seed hidden validation, a validated four-variant baseline portfolio, synchronized wall timing with changing inputs, failure-inclusive evidence-gated scoring, and bounded mini-swe-agent sessions. It supports native one-shot and feedback protocols, Harbor tasks, and a skill for existing coding-agent sessions. The integration also exposes the supplied 105-task research selection through the same workflow.

The supplied research archive motivates a broader design: calibrated per-task inputs and FP64 precision contracts, hidden shapes, execution-origin attribution, measured device ceilings, algorithm-aware headroom scores, performance decomposition, production tasks, and a Trainium/NKI track. These remain explicit research objectives where implementation is absent. CPU and simulated regression tests validate workflow contracts; they do not establish GPU correctness, isolation, speedups, or cross-platform support. We retain historical audit evidence with its limitations rather than treating source review as device reproduction.

## 1. Scope, terminology, and evidence

KernelBench Pro is the repository and software package. A **generation backend** supplies an agent environment; a **verification backend** executes trusted evaluation against a submitted source file. Several execution backends do not imply several accelerator platforms: current compute execution is NVIDIA CUDA. Python 3.12, Torch 2.12.0, mini-swe-agent 2.4.6, and the optional Harbor 0.24.0 integration define the software baseline.

The research archive contains different kinds of evidence:

| Evidence | What is available | What it supports |
| --- | --- | --- |
| KernelBench-Verified audit | Standardized records for 1,574 historically green implementations across a 247-task leaderboard snapshot; issue descriptions and source-review bases | A taxonomy and motivation for checks; not fresh GPU correctness or latency verdicts |
| Analytic ledger | Historical meta-device traces, operation counts, bytes, FLOPs and spec-based estimates | Approximate fusion/precision sensitivity; not measured ceilings or universal algorithmic bounds |
| Task selection | A 250-row manifest: 105 scored, 5 probes and 140 extended; public selection seed 20261002 | Reproducible membership and provisional strata |
| Ranking studies | Claims in the draft and historical summaries, with the underlying seven-model result directory absent | Hypotheses to re-evaluate; no independently reproduced τ=1.00 claim here |
| E0 author experiment | Aggregate Fast_p numbers reported in the supplied manuscript, without per-task artifacts | A historical reference point, outside the current protocol |
| Current software | Source, tests, configuration, task manifests and workflow receipts | Implemented behavior at the validation level actually exercised |

The [source manifest](evidence/source_manifest.json) records the supplied archive's hash, input member hashes, derived artifact hashes, and omitted material. The original candidate source bundle is absent. Historical source paths in the standardized audit table are provenance strings, not files present in this checkout. We do not execute archived candidate or audit scripts during import.

## 2. Motivation 1: tasks should discriminate at affordable cost

Many problems share computation patterns, and relatively few ask for an algorithmic change. Averaging many similar implementation problems can dilute the signal of a small design stratum. The archive's analyses suggest this redundancy, but fusion headroom alone cannot identify useless tasks: a single-kernel normalization can still leave substantial implementation room, while GEMM and convolution bounds depend on the algorithm and numerical contract.

### 2.1 Implemented task scope and selection

The default workflow retains Level 1–3, with explicit ranges and declared sample counts. Level 4 remains experimental. `--suite kernelbench105` selects the research manifest; ranges intersect with that selection. Every workflow stage uses the same IDs. An empty intersection is an error.

| Role | Tasks | Use |
| --- | --- | --- |
| scored | 105: L1 39, L2 40, L3 26 | Main research selection |
| probe | 5 | Audit diagnostics, reported separately |
| extended | 140 | Remaining coverage, reported separately |

The main selection's provisional strata are I=30 implementation, F=56 fusion, and D=19 design. The original procedure retains design candidates, groups other tasks by computation pattern and optimization type, and selects representatives using a public seed rather than model results. The manifest preserves every original membership decision and reason. Its name is standardized to KernelBench-105; the archive's `kb100.csv` filename was misleading.

These strata are descriptive research labels. Some design candidates were not certified, and some selected tasks have low-signal or degenerate original inputs. In particular, importing membership does not implement the draft's proposed input recalibration. Reference names and source hashes bind the selection to the reviewed checkout; changing a reference requires reviewing and versioning the manifest. No old tensor shape, input constructor, or baseline is silently substituted to match a historical study.

### 2.2 Target benchmark extension

The implementation now provides a separate `production41` suite in Level 5: 12 design, 6 numerical-design, 12 fixed-format implementation and 11 backward references. The original 250 tasks and the pre-existing Level 4 model files are preserved. These are software additions, not measured GPU results. The families follow the earlier draft’s categories:

- Algorithmic design: online/flash attention and backward, fused linear plus cross-entropy, Welford normalization, partial top-k, chunked linear attention and state-space scans, persistent or wavefront RNNs, FFT convolution, grouped MoE GEMM, dequantization fusion, and paged attention.
- Numerical design: precision emulation such as multi-pass low-precision matmul; choosing format, scaling block, and accumulation precision under an error budget; quantization fused with downstream work.
- Fixed-format implementation: production low-precision GEMM, grouped operators, decode attention, and related library-backed workloads.
- A separate backward track, with reference gradients and suitable high-precision checks.

The implemented families declare per-input storage roles, legal domains, hidden dimensions/strides and dynamic knobs. A deterministic memory plan uses the smaller of an explicit envelope and one tenth of device capacity, preserving shape constraints; this estimate is not a guarantee against arbitrary compiler/candidate workspace OOM. Four tracks are selected and scored separately, while their all-track collection macro is diagnostic. FP8 storage and packed INT4 are explicit. The FP4 quantization task defines software E2M1 semantics, not native NVFP4/MXFP4 throughput. All 11 backward families have FP64 gradcheck and directional finite-difference regressions. Existing baseline variants run through the same role-preserving path; strongest per-task library rewrites remain an ongoing extension. Design labels do not provide headroom eligibility without reviewed bounds and independent reachability witnesses. [Production tasks](../../docs/PRODUCTION_TASKS.md) lists every reference, representation and limit.

Selection-analysis software now exports current evidence-gated scores, compares full and selected populations with tie-aware rank statistics, bootstraps tasks/sessions/declared slots and reports missing scope explicitly. Constant score vectors have undefined correlation, not perfect agreement. Copied runs do not become independent sessions. A paired-binary normal design approximation is available under explicit assumptions, rather than replaying the draft’s unverified power numbers. Shapes within a task remain correlated instances; family correlation and selection-algorithm uncertainty are not automatically calibrated. Original seven-model raw data remain unavailable, and no compression/discrimination experiment is run in this scope. [Study workflows](../../docs/STUDIES.md) defines the inputs and limitations.

## 3. Motivation 2: passing a test should mean something

### 3.1 Diagnosis and its limits

The archived audit assigns each implementation its most severe source-review conclusion: 159 invalid, 60 unverifiable, 88 at risk, 217 in a small-speedup/noise category, 14 requiring investigation, 6 requiring explanation, 34 explanatory notes, and 996 with no issue found. Thus 219 is the sum of invalid and unverifiable records. Issue labels overlap: 346 records carry a noise label, while only 217 have noise as their final category. A displayed speedup ≤1.05 is a review threshold, not a measured confidence interval. Likewise, the 996 records are not GPU-certified correct implementations.

The records motivate checks for library substitution, unused custom paths, mode-dependent execution, weakened precision, low-output-magnitude comparisons, unstable statistics, indexing defects, cached parameters, and synchronization risks. A separate root-cause review in the supplied draft discusses representative implementations of 102 tasks, based on source review and counterexamples. Its counts and GPU detection rates have not been reproduced under the current harness. We do not attribute intent to every incorrect implementation.

### 3.2 Implemented correctness contract

Current standard and hidden checks share recursive comparison of structure, tensor shape and dtype. Integer and boolean tensors are exact. Floating outputs retain the existing dtype tolerances; NaN positions and positive/negative infinity positions must match before finite values are compared. Legacy tasks retain this contract; opt-in strict validation adds independent FP64 gates on reviewed auxiliary cases. Production tasks additionally apply per-instance relative RMS and all-element budgets to declared reference outputs, so small magnitudes cannot make zero answers pass. Numerical-design references use FP64 arithmetic after prescribed quantization and round to declared output formats. A reference-output comparison is not independent FP64 self-qualification.

Reference and candidate receive separate input snapshots. Outputs are saved before another forward can overwrite a reused buffer. RNG state is replayed so stochastic operations face matching conditions. Paired-order checks run candidate→reference and reference→candidate, advancing each model equally. Separate source namespaces prevent candidate definitions replacing reference globals; shared backend settings are restored around candidate activity. These measures address specific contamination paths, not arbitrary host compromise.

Hidden input generators run under seeds 42, 123 and 456. For 29 reviewed Level 1 tasks, the first hidden configuration also receives zero, scaled and sparse-outlier variants, without changing shape, dtype or unreviewed input roles. No generic perturbation is applied to probability, mask, index or structural domains. Hidden sets that are empty, malformed, unavailable or fail in the reference cannot count as successful validation. The legacy default additions do not provide hidden-shape coverage. Opt-in strict contracts separately cover odd-sized and non-contiguous auxiliary cases for 17 reviewed L1 tasks. Production-41 has role-preserving private odd-shape, strided and dynamic-range cases, legal paged/masked variants, and near-constant normalization cases.

### 3.3 Union of implemented checks and target checks

| Concern | Current implementation | Remaining target |
| --- | --- | --- |
| Wrong output representation | Recursive structure/shape/dtype/discrete/non-finite checks; production role-preserving storage | Legacy full-suite per-task numerical and role contracts |
| Weak absolute tolerance / degenerate reference | 17 legacy auxiliary FP64/signal contracts; all production tasks have relative RMS/all-element gates and zero-answer regressions | Full independent oracle qualification and legacy input calibration |
| Rounding and unstable statistics | Limited legacy stresses; production legal dynamic range and near-constant normalization inputs | Broader legacy per-operator cancellation and calibrated precision contracts |
| Extremes and boundary behavior | Matching non-finite masks, limited input stresses | Legal-domain extremes, GELU discriminators, asymmetric indexing and border cases |
| Reference-result or input caching | Paired order, snapshots, changing timed inputs; strict same-object mutation probes | Same-object mutation probes and complete parameter/state contracts |
| Wrong train/eval path or BN state | Strict same-name mode/buffer/parameter contract implemented and tested on CPU counterexamples | Check both declared modes and running state, time in the required mode |
| Work outside the timed stream | Device-wide synchronization and output consumption | Throughput contrast, leftover-work checks and broader tampering rejection |
| No custom kernel doing useful work | No certified origin attribution | Attribute launched device work by submission/library/framework origin; separate tracks |
| Synchronization / memory safety | Some repeat behavior is checked | Optional sanitizer, task-specific determinism policies and GPU red-team reproduction |
| Overfitting a fixed input shape | Legacy auxiliary odd/strided cases; production private shape/stride/knob families | Remaining legacy full-suite shape and sensitivity contracts |

The archived hardening prototype is not imported as the evaluator. Its uniform transforms can violate input roles, its numerical oracle assumptions need review, its percentile interval is not a confidence interval, and its suggested 20% custom-time threshold conflicts with the paper's proposed 95% track threshold. Neither threshold is calibrated here. Silently skipping a non-finite reference or treating every repeated bitwise difference as a race would also change semantics incorrectly.

### 3.4 Target numerical contract

The target evaluator distinguishes fixed-format tasks from error-budget tasks. A fixed-format contract declares all input, output and scaling formats. An error-budget task permits internal format selection but compares with a reviewed FP64 oracle, using a normwise bound plus an all-element bound; integers, booleans and non-finite handling remain explicit. Thresholds need legitimate implementation error distributions or format-derived analysis, published per task.

Input design has three layers: diagnose and repair non-informative reference outputs using legal inputs or parameters; instantiate operator-specific counterexamples; and vary hidden shapes and strides. Calibration inputs would also define the timing workload, while adversarial families remain correctness-only. Calibrating a reference changes the workload and requires new baselines and protocol labels. A constant-output task can be semantically valid, so degeneracy is evidence of a weak test rather than automatic proof that the task or candidate is wrong.

The intended precision gate should distinguish acceptable precision emulation from unacceptable single-pass lower precision under the same contract. This discrimination is an experiment, not an assumption. The first implementation supplies role-preserving FP64 oracle conversion and explicit initial thresholds for auxiliary instances of 17 L1 tasks (1–3, 19–27, 47–49, 51–52). Both the reference and candidate must meet RMS and all-element bounds. Element limits include an explicit global RMS-scaled term to accommodate legitimate cancellation error. Original-size legacy timing inputs retain the native comparator; their full FP64 calibration and GPU error distributions remain outstanding. The new numerical-design track now has task-specific budgets and FP64-derived mathematical references, while independent oracle qualification of every production path is still separate work.

## 4. Motivation 3: explain and compare performance honestly

### 4.1 Implemented baseline and timing protocol

The default baseline portfolio contains eager, torch.compile default, torch.compile max-autotune, and CUDA Graph. Variants run as independent tasks, validate initial and timed outputs, retain failures, and choose among valid variants. The suite's fp32 contract uses IEEE settings; explicit legacy single-baseline behavior is recorded separately. An optional extended portfolio now adds default/max-autotune compile variants with outer CUDA Graph capture and reviewed framework rewrites for channel RMSNorm and L1/L2 normalization. The latter are tied to three reference-source hashes and retain their exact eps and mean-vs-sum semantics. All variants still validate every measured output, retain failures, and select only validated members.

Synchronized wall time covers host invocation, output consumption, and completion of device work. Input construction is outside the interval; each timed call uses newly generated inputs with recorded seeds, and its output is checked outside the timed region. Graph replay includes input copying and output cloning. A deterministic generator can still yield identical contents despite a different seed.

Raw millisecond samples, mean, median, standard deviation, an approximate normal mean interval, device/software fingerprint and input signatures are saved. Cold cost includes recorded construction/initialization, first execution and warmup; it is not a cache-cleared application launch measurement. For a declared call count K, the average is:

```
T_K = (T_cold + (K - 1) T_steady) / K
```

The default is K=10000; reporting K=1 or other values is configurable. Input preparation and process startup are not silently claimed as included. Profiling runs separately and never supplies official latency.

### 4.2 Implemented scoring and recovery

Every selected task and declared sample contributes a slot. Failed, missing, stale, incompatible or evidence-incomplete slots receive zero. Current analysis uses all declared slots, first averaging samples per task, then macro-averaging task layers; it does not take the fastest sample. The report now computes mean, median and best-of-N sensitivity on the identical declared slots, retaining failures and cost receipts. This ablation does not change official mean scoring, spend additional model calls or certify unknown budgets.

Eligible comparisons require matching reference and kernel sources, evaluator version, input signature, precision/tolerance, device/software class, workflow configuration, raw timing samples, changing-input/verification receipts and cold costs. The strongest valid amortized baseline is selected at the declared K. A positive score additionally requires separated approximate steady-state mean intervals. Cold-cost uncertainty is not modeled by this interval test. When independent repeats or paired timing are enabled, their current measured samples and interval protocol replace the corresponding legacy steady-state values; incomplete or stale auxiliary evidence blocks the score.

Current bounded gain is:

```
G = clip(ln(T_ref,K / T_candidate,K) / ln(1.25), 0, 1)
```

This remains the default failure-inclusive measurement score. An optional certificate-gated headroom mode is implemented separately below; it does not turn these default numbers into measured headroom results. Two-level bootstrap resamples tasks and declared sample slots. Original Verified metrics remain available with their original formulas, subject to the measurement evidence gate. Correctness pass@k remains distinct from performance scoring. Raw stage diagnostics do not confer score eligibility.

Content-based recovery covers current sources, settings, hidden tests and evaluator modules. Generation recovery binds receipts to the candidate hash; explicit repeated attempts retain prior costs. Missing evidence is not converted to zero cost or successful verification. The scorer reports `anti_cheat_verified=false` and `cross_device_comparison_certified=false`. Suite roles and provisional label sources are explicit; probes and extended runs are not labeled as main-leaderboard eligible.

### 4.3 Target headroom score and certificates

The implemented optional headroom mode accepts device-specific calibration receipts and reviewed work-model certificates, rather than importing the historical spec-sheet ledger as measurements:

```
SOL_ref = max_k W_ref,k / rho_device,k
SOL_alg = max_k W_best_known_algorithm,k / rho_device,k
S = clip(ln(T_ref / T_candidate) /
         max(ln(T_ref / SOL_target), ln(1.25)), 0, 1)
```

The target can be the reference-graph bound for implementation/fusion tasks or the best-known-algorithm bound for design tasks. Minimum traffic, operation counts, legal numerical formats and resource ceilings must be explicit. A reference graph is not a universal bound: algebraic simplification or a better convolution algorithm can legitimately beat it. Recurrent workloads can additionally need a critical-path bound.

A design label needs a reviewed algorithmic comparison, a credible bound and a reference-device reachability witness; other devices may remain unwitnessed. Physically implausible latencies warrant investigation, not an automatic accusation based on a faulty bound. Optional scaling-exponent evidence needs multiple sizes and uncertainty estimates. The certificate reader binds the reference source, inputs, precision and calibration. D requires a reviewed algorithmic/numerical comparison and a reachability witness exported from at least two completed independent final standard/hidden verifications. Witness raw samples, source identities and distinct nonces are rechecked. A witness may belong to an explicitly recorded reference device; target-device reachability is reported separately. Imported provisional D labels do not meet this gate. Attained microbenchmark rates are model inputs, not proven physical upper bounds; below-model performance is an investigation flag. No calibration or reachability experiment is run in this implementation-only scope.

### 4.4 Target attribution and performance model

The desired separate profiling pass measures wall time T, the union of device busy intervals T_dev, kernel count, origin attribution, and optional operation/traffic counters. Overlapping kernel durations cannot simply be summed into busy time. The proposed explanatory model is:

```
T_X = SOL_alg * A_X * E_X * O_X
A_X = SOL_X / SOL_alg           # algorithm and traffic gap
E_X = T_dev,X / SOL_X           # device efficiency gap
O_X = T_X / T_dev,X             # host, launch and synchronization overhead
```

Its usefulness depends on trustworthy work counts and ceilings; deviations can expose model errors rather than candidate wrongdoing. Without counters, A and E should remain unresolved or be reported as their product, never fabricated. Launch-only attribution needs measured launch latency and kernel-count changes. It is not inferred solely from a speedup over eager.

NVIDIA's target tooling combines activity timelines with Nsight Compute and optional compute-sanitizer. Trainium's target uses Neuron timelines, per-engine tensor/vector/scalar/DMA information, and compiler reports where available. CUDA calibration software now supplies raw copy/GEMM timing sweeps and a synchronized launch proxy, bound to physical UUID, driver, software and evaluator. Development traces now use device-activity interval unions and report kernel counts. Offline model code returns A/E as unresolved without measured source-bound counters, with AE/O retained; a launch-only diagnostic additionally requires measured launch cost, kernel-count changes, similar busy time and paired noise separation. Binary origin association, vendor hardware-counter collectors and a formal attribution scoring gate remain incomplete. Profiler durations never replace scored wall times. [Headroom scoring](../../docs/HEADROOM_SCORING.md) defines the implemented schema and limits.

Balanced same-process timing is now implemented as an optional task against the selected validated baseline, with an independent same-variant reference control. Every output is verified, measured order is balanced, and candidate process settings are audited/restored. Bootstrap first resamples independent final processes, then their paired log ratios; speedup requires its lower interval bound to exceed both 1 and the reference/control upper bound. The exchangeability assumptions and shared-device limits remain explicit. The archived p10–p90 range is not used as a confidence interval.

Independent final remeasurement now freezes every declared source by hash, issues fresh initialization and hidden seeds, and runs all declared standard/hidden repeats in independent tasks. All repeats must pass. The score uses their mean rather than the fastest repeat, with process-then-sample bootstrap uncertainty when pairing is disabled. Original cold costs remain point estimates. The saved nonce/source/evaluator/configuration plan supports resume without silently repeating completed attempts; changed source invalidates only affected sample repeats. Auxiliary results and frozen source content are rechecked by the legacy score gate as well. [Timing protocol](../../docs/PAIRED_TIMING.md) records implementation and deployment limits.

Scores are defined per device and protocol. No device-independent efficiency transfer or merged cross-platform ranking is claimed.

## 5. Motivation 4: evaluate the feedback loop and its cost

### 5.1 Implemented interaction protocols

The native controller uses upstream mini-swe-agent DefaultAgent with local Model and Environment adapters. Each action submits a complete source file. Claude Code and LiteLLM adapters share this protocol; vendor-specific nested tools are not part of the controlled loop.

| Protocol | Model interaction | Development checks | Selection |
| --- | --- | --- | --- |
| one-shot | One adapter call per sample | None | That submission |
| feedback | At most 5 candidate submissions and 2400 seconds per sample by default | Public standard correctness/timing and optional profiler diagnostics | Fastest development-correct candidate |
| external | Harbor-managed agent | External policy | Recorded as external, without native round guarantees |
| interactive | Existing Claude Code/Codex session using the skill | Public bridge, with its own observable development limits | Frozen best development candidate; unobserved model calls remain unknown |

Budgets are configurable in TOML/CLI. Multiple independent samples do not become one-shot best-of-N without being labeled. The controller retains a completed correct candidate through later wrong answers or timeouts, and cleans up nested worker processes on timeout. Development payloads disable hidden tests. After source selection, native final evaluation performs standard and hidden verification in new evaluation tasks; hidden outcomes do not drive the native search.

The skill reuses the current coding session instead of recursively launching another agent. It prepares the public interface, exposes standard-only verification and freezes a selected candidate. This is useful practitioner behavior, but cannot certify one observed model call or complete session cost.

### 5.2 Artifacts, accounting and outstanding experiments

Trajectories, candidate snapshots, per-call usage, generation summaries and resource receipts are retained. `generation_attempts.json` preserves explicit retries; failed matching attempts are resumed without silently spending another budget. Hash mismatches remove claimed generation attribution. Phase wall times include transport and waiting, not GPU billing.

Token usage and estimated model cost are reported where observable; unknown fields remain null. The cost cap is checked between requests and can overshoot by one call. A controlled offline local-transformers adapter now reserves exact prompt and returned generated token IDs before each call; supported single-request LiteLLM providers can enforce a completion-token parameter. Unknown or exceeded usage stops the loop and fails budget qualification. Claude CLI and provider-side prompt estimates cannot certify a total-token cap and are rejected for that setting. Local weights and tokenizer assets are content-bound to generation and scoring; no checkpoint has been executed here. The old draft’s tentative 60-minute default is replaced by the implemented 40-minute/5-candidate default. Multi-budget, multi-model and multi-session plan/runner software is available for native one-shot, feedback and a general Harbor coding-agent baseline. Plans are nonexecuting by default and require an explicit execution switch; no real study is launched here.

Development compilation, validation and profiling can also reserve an aggregate local/Docker worker wall lease. Parent-owned monotonic timing includes cleanup; a five-second cleanup allowance reduces the child execution timeout. Missing or exceeded receipts fail qualification, while ordinary admission exhaustion retains the best completed candidate. This is an exclusive development lane accounting contract, not a hardware preemption guarantee or a pure GPU busy-time cap. Remote cancellation and physical GPU-time enforcement remain unfinished (R-18). Capped samples cannot silently restart their budget: only matching completed work can resume, and altered/crashed attempts require a distinct run. Scoring recomputes ledger totals and binds the attempt, summary and current-source receipts. Detailed scope and configuration are in [resource budgets](../../docs/RESOURCE_BUDGETS.md).

The implemented offline curve analysis uses independent declared budget points, retains failed slots and all sessions, provides hierarchical point intervals and restricts normalized trapezoidal AUC to shared known interpolation domains. It does not take a hidden best-so-far trace or impose monotonic gains. Unknown observed costs stay null; native provider totals and Harbor cache-inclusive input semantics are distinct. Agent execution time excludes verifier/setup and is not GPU billing; equal device time is not certified. Hardware, precision, scores, unvaried limits and generation methods remain explicit groups/series. No model performance comparison has yet been produced under this unified protocol.

### 5.3 Training and reward-audit software

An explicit offline-local entrypoint now integrates [TRL 1.8.0 GRPOTrainer](https://huggingface.co/docs/trl/v1.8.0/en/grpo_trainer), using its DAPO loss variant, group reward scaling, optimizer and tool-observation masks. One-shot and standard-feedback environments share native candidate selection. Training and held-out task/reference identities are disjoint; reviewed family separation is the default. Policy weights and tokenizer assets are local, content-bound, and loaded only after explicit execution and baseline preflight. No real model loading or training has been performed here.

Terminal rewards stage syntax, reviewed small-instance checks, standard/hidden validation and frozen remeasurement. Same-prompt candidates share repeat seeds and interleave fresh verifier processes on one UUID-bound device; drift elimination is not certified. Native qualified scores and an explicitly weaker public-speedup ablation remain separate. Reservations precede generation, interrupted groups remain charged, checkpoints bind optimizer steps and reward-log prefixes, and a parent process accounts for aggregate training wall time. Completion sequences containing tool observations are not presented as exact model-token billing.

Regrading plans and offline audits cover ICC(A,1), reward ordering against independent runtime anchors, predeclared attacks with evidence of their intended failed checks, and paired public/final score gaps. Single-kernel ICC, missing tools and constant/absent data are not fabricated into passing results; correlated ordering events retain explicit inference limits. Held-out native plans freeze checkpoint assets and cannot feed tasks back into the optimizer. Candidate algorithm-certificate bonuses and a training-lineage-bound effectiveness comparison remain unfinished (R-20). These are software capabilities, not evidence that any preregistered threshold or training benefit has been achieved. Details and commands are in [the RL workflow](../../docs/RL_WORKFLOW.md).

## 6. Motivation 5: reproducible infrastructure is part of the evaluation

### 6.1 Implemented workflow

The original integration starts at `2bce940`, recording KernelBench-Verified upstream `3fdf6fe`. Subsequent committed versions add strict comparison, baseline/timing, evidence-gated scoring, reliability checks, project cleanup, Harbor/Python 3.12 and GPU/agent configuration. [The inventory](../../docs/FEATURE_INVENTORY.md) preserves exact history and validation states. The former manuscript's claim that all later checks were uncommitted is obsolete.

One `RunConfig` drives baseline, generation, evaluation, analysis and Harbor export. TOML hardware profiles and explicit overrides resolve architecture, visible devices and cloud type; unknown hardware never silently falls back to Hopper. Generation supports local, Docker, Modal and Harbor. Verification supports local, Docker, Modal and an HTTP queue. Slurm allocates resources around the same workflow. Local execution serializes tasks per GPU while allowing different devices to progress.

The HTTP queue persists jobs in SQLite, supports idempotent requests and restart recovery, and bounds outstanding work. Evaluation subprocesses have independent build directories and timeouts; missing kernels and failed regenerations are failures rather than stale successes. Portable child configurations contain resolved settings rather than host-only TOML paths. Project names, documentation, generated pages and source notices identify this maintained repository while retaining inherited copyright and licenses.

### 6.2 Harbor and trust boundaries

Each selected task can be exported into an instruction, environment, tests and solution layout accepted by Harbor's actual Task schema. A separate verifier configuration contains trusted evaluator code and hidden tests; the candidate handoff is `solution.py`. The native workflow re-ingests the source into its own sample and evidence format. Correctness and bounded-gain rewards have distinct meanings; verifier failure produces zero reward.

This is an implemented export and invocation path, not a proven production sandbox. Agent working directories, source namespaces, protocol flags and ordinary subprocesses are not OS security boundaries. Docker/provider configuration, network restrictions, GPU access and Harbor artifact transfer require real environment validation. The draft's illustrative TOML is replaced by the actual exporter and its tests; an unsupported schema example is not an interface contract.

Regrading saved artifacts after baseline/library/rule updates is an important design goal. It should execute the current verifier and regenerate measurement evidence without rerunning the agent, while retaining historical provenance. Content-based invalidation and native re-evaluation provide part of this workflow; the full Harbor regrade deployment experiment remains unverified.

### 6.3 Two-platform target and optional diagnostics

The target device interface has five responsibilities: `caps()`, `calibrate()`, `flush()`, `time()`, and `attribute()`. It would isolate per-device capabilities, measured ceilings, cache policy, timing and origin profiling. The current GPU configuration module is not this interface. A Trainium/NKI implementation still needs Neuron-compatible tasks, baselines, execution, profiling and reachability evidence; it is not present merely because AWS credentials or a Modal backend exist.

Implemented evidence receipts now distinguish passed, failed, not_measured and justified not_applicable checks, with unknown measurement values remaining null. The current level requires native correctness; L0 additionally requires the reviewed auxiliary precision, signal, input-family, sensitivity, environment and mode/state contracts. L1 adds paired timing, noise, origin, calibration and bound evidence; L2 adds counters, memory safety and algorithm evidence. Device measurement results are not collected in this implementation-only scope, and full origin/counter collectors remain incomplete, so requesting L1/L2 cannot receive that level or score eligibility. The scorer recomputes eligibility from checks, scope and requested level. These operational levels and their limited task coverage are specified in [strict validation](../../docs/STRICT_VALIDATION.md); they do not imply the entire broader research design has been validated. Missing diagnostics never weaken existing gates.

## 7. Validation and experimental plan

The first design-delivery batch adds strict contract and evidence-level code, tested using the 17 actual CPU references plus known bad arithmetic, input/parameter caches, BN state errors and process-setting changes. It also adds saved-slot aggregation sensitivity. Current full regression results are maintained in version nine of the inventory; this extends software validation, not GPU or paid-model research evidence.

At baseline `92c27f4`, 270 CPU/simulated tests passed under Python 3.12.12, Torch 2.12.0, Harbor 0.24.0 and mini-swe-agent 2.4.6. These include real DefaultAgent control flow and actual Task model validation, with model/GPU execution mocked where appropriate. The integration's selection, provenance and motivation-coverage tests are recorded separately in the current feature inventory. Ruff, dependency consistency, skill validation and offline packaging were checked. These are software validation results, not E1–E9 research findings.

Real GPU end-to-end validation remains outstanding. Prior attempts encountered busy allocated devices, an incompatible host driver for the pinned Torch CUDA build, and a container networking failure; none is counted as a passing GPU run. No paid-model experiment is included in this integration.

| Experiment | Question | Available now / remaining work |
| --- | --- | --- |
| E0 | How does a general-purpose agent perform? | Historical author report only; raw artifacts absent; rerun under current protocol |
| E1 | Which gains survive reliable evaluation and how are they attributed? | Audit taxonomy and workflow exist; original candidates, calibrated checks and GPU attribution needed |
| E1b | How does feedback change scores and budgets? | Both protocols and receipts exist; same-budget multi-session model experiments and curves needed |
| E2 | How sensitive are conclusions to stronger baselines at current sizes? | Standard and extended portfolios, paired controls and final repeats exist; real GPU measurements and drift pilot needed |
| E3 | Do design and implementation strata rank agents differently? | Provisional labels exist; certified design tasks, model population and held-out analysis needed |
| E4 | Are design targets valid and reachable? | Witnesses, bounds and high-precision equivalence checks needed |
| E5 | Do hardening checks block real loopholes without false positives? | Source-review table retained; raw sources missing, known-good cases and GPU red-team reproduction needed |
| E6 | Does the numerical gate accept legitimate arithmetic and reject unacceptable loss? | FP64 oracles, calibrated thresholds and precision-emulation implementations needed |
| E7 | Can artifacts be regraded after rule/library updates? | Source artifacts and invalidation exist; full deployed Harbor regrade study needed |
| E8 | How do the same workloads behave on NVIDIA and Trainium? | CUDA workflow exists; NKI backend, supported tasks and both-platform profiling needed |
| E9 | Does the proposed reward improve held-out kernel engineering? | Full reward protocol, RL training and held-out experiments needed |

The historical E0 author report gives L1/L2 Fast_0 of 0.87/0.94, Fast_1 of 0.56/0.93, Fast_1.5 of 0.36/0.73 and Fast_2 of 0.30/0.56. It used Claude Code with Sonnet 4.6 and remote H100 verification, without the current budget cap or development/final separation; L2 used an fp32 baseline. These figures are not directly comparable to current scores and do not substantiate superiority over specialized methods.

## 8. Related work and limitations

KernelBench supplies the task-based kernel-generation setting, including earlier multi-turn experiments. KernelBench-Verified supplies inherited tasks, hidden value distributions and the TF32 baseline motivation. Hard, KernelBenchX, robust-kbench and TritonRL motivate multi-seed/domain-aware tests, granular diagnostics and ordering checks; [the reliability document](../../docs/RELIABILITY.md) records exact adopted methods and upstream versions. mini-swe-agent supplies the control loop; MLC/TIRx motivates a fixed feedback contract and separation of diagnostic profiling from timing, as recorded in [Agent loops](../../docs/AGENT_LOOPS.md). This integration does not import TIRx's compiler or checkers.

The supplied manuscript also discusses SOL-ExecBench, MultiKernelBench, production/low-precision benchmarks and measured-roofline tools. Those comparisons motivate the research plan; they do not establish that this implementation already surpasses their validation, profiling or hardware coverage. A publication-ready novelty claim needs a refreshed primary-source review and the planned experiments.

The audit snapshot, analytic ledger and public-kernel studies mix historical reference sizes and software versions. Some device identities were inferred; spec-based ceilings and meta traces have blind spots, especially on convolution algorithms and recurrent models. A reference-graph estimate is not a best-known-algorithm proof. Historical speedups and aggregate rankings must not be treated as newly measured data.

Other limitations are substantive: no universal numerical oracle, strict auxiliary contracts limited to 17 L1 tasks, incomplete general state/mode mappings, no full-suite hidden-shape guarantee, no verified custom-work attribution, no collected calibration or reachability results, no completed two-platform implementation, no complete security isolation and no model/GPU experiment for the unified protocol. Bootstrap intervals and approximate timing intervals do not remove shared-device drift. Changing inputs does not guarantee different values. Optional diagnostics do not prove absence of races or cheating.

## 9. Conclusion

KernelBench Pro connects practical kernel-agent workflows to five explicit evaluation problems. The implementation provides auditable generation, validation, measurement, scoring and recovery contracts; the research plan adds task discrimination, stronger numerical and execution evidence, and hardware-aware explanations. Each existing change is mapped to its motivation, and each remaining design has a recorded gap. The current scope completes software and regression coverage without running GPU, Trainium, model or training experiments. Such experiments would be a separate step toward empirical validation; their results remain explicitly unrun.
