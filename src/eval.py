# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Helpers for Evaluations
"""

import torch
import torch.nn as nn
import os, subprocess
from pydantic import BaseModel
import numpy as np
import random
import json
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from contextvars import ContextVar
from io import StringIO
import sys

from kb_workflows.measurement import CLOCK, input_signature, synchronized_times, timing_stats

_PRESERVE_INPUT_DTYPES = ContextVar("preserve_declared_input_dtypes", default=False)


@contextmanager
def input_dtype_policy(*, preserve=False):
    token = _PRESERVE_INPUT_DTYPES.set(preserve)
    try:
        yield
    finally:
        _PRESERVE_INPUT_DTYPES.reset(token)


def move_model(model, device, precision):
    return model.to(device=device) if _PRESERVE_INPUT_DTYPES.get() else model.to(device=device, dtype=precision)

from . import utils
from .reliability import HIDDEN_SEEDS, check_paired_calls, stress_cases, stress_enabled
from .validation import clone_tensors, compare_outputs, synchronize

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")
HIDDEN_TESTS_PATH = os.path.join(REPO_TOP_PATH, "hidden_tests")


def fetch_ref_arch_from_problem_id(problem_id, problems, with_name=False) -> str:
    """
    Fetches the reference architecture in string for a given problem_id
    """
    if isinstance(problem_id, str):
        problem_id = int(problem_id)

    problem_path = problems[problem_id]

    # problem_path = os.path.join(REPO_ROOT_PATH, problem)
    if not os.path.exists(problem_path):
        raise FileNotFoundError(f"Problem file at {problem_path} does not exist.")

    ref_arch = utils.read_file(problem_path)
    if not with_name:
        return ref_arch
    else:
        return (problem_path, ref_arch)


def fetch_ref_arch_from_level_problem_id(level, problem_id, with_name=False):
    PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level" + str(level))
    dataset = utils.construct_problem_dataset_from_problem_dir(PROBLEM_DIR)
    return fetch_ref_arch_from_problem_id(problem_id, dataset, with_name)


def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)


def get_torch_dtype_from_string(precision: str) -> torch.dtype:
    """
    Get the torch dtype for specific precision
    """
    if precision == "fp32":
        return torch.float32
    elif precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    else:
        raise ValueError(f"Invalid precision not supported: {precision}")


def get_tolerance_for_precision(precision) -> float:
    """
    Get the tolerance from a string representing the precision.
    These tolerances are inspired by torchbench (PyTorch Benchmarking Suite):
    Reference:
    https://github.com/pytorch/benchmark/blob/cfd835c35d04513ced9a59bd074eeb21dc8187d7/torchbenchmark/util/env_check.py#L519
    """
    if isinstance(precision, str):
        precision = get_torch_dtype_from_string(precision)

    PRECISION_TOLERANCES = {
        # 1e-3 for fp32: torchbench uses 1e-4 for PyTorch-vs-PyTorch comparisons, but
        # KernelBench evaluates custom CUDA kernels which can have larger accumulated
        # floating-point errors from parallel reductions (~eps * sqrt(N)).  1e-3 is more
        # appropriate here.  A strict 1e-4 leaderboard is also published for reference.
        torch.float32: 1e-3,
        # torchbench states for bf16 and fp16, use 1e-3 as tolerance and 1e-2 if too strict.
        torch.float16: 1e-2,
        torch.bfloat16: 1e-2,
    }
    assert precision in PRECISION_TOLERANCES, f"Invalid precision not supported: {precision}"
    # Optional override for the strict-tolerance leaderboard runs (e.g. fp32 @ 1e-4).
    # Defaults to the standard values above when KB_FP32_TOL is unset.
    if precision == torch.float32:
        _ov = os.environ.get("KB_FP32_TOL")
        if _ov:
            return float(_ov)
    return PRECISION_TOLERANCES[precision]


def _process_input_tensor(input, device, precision=torch.float32):
    """
    Helper function to move tensors to the correct device and apply dtype casting.

    Args:
        input: Input tensor or non-tensor value
        device: Target CUDA device
        precision: torch.dtype
    Returns:
        Processed tensor on correct device with correct dtype, or original value if not a tensor
    """
    # sometimes things like init inputs are floats (like in the case of labels / targets, classification losses, etc.)
    if not isinstance(input, torch.Tensor):
        return input

    # cast to the desired precision dtype for activations
    input_tensor = input.to(dtype=precision) if input.is_floating_point() and not _PRESERVE_INPUT_DTYPES.get() else input

    # Default for all other backends and float types
    output = input_tensor.to(device=device)
    if output.stride() != input.stride() and all(stride > 0 for stride in input.stride()):
        # dtype/device conversion can compact sliced inputs. Keep the reviewed
        # positive-stride layout visible to the actual candidate, including bool.
        output = torch.empty_strided(input.shape, input.stride(), dtype=output.dtype, device=output.device).copy_(output)
    return output


def _process_input_group(inputs, device, precision=torch.float32):
    from .input_groups import copy_inputs

    return copy_inputs(inputs, lambda value: _process_input_tensor(value, device, precision), isolate=False)


class KernelExecResult(BaseModel):
    """
    Single Kernel Execution
    """

    compiled: bool = False
    correctness: bool = False
    metadata: dict = {}
    runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    runtime_stats: dict = {}  # only recorded if we decide to measure performance
    peak_memory: float = -1.0  # peak GPU memory in bytes, -1.0 if not measured
    memory_stats: dict = {}    # dict with peak_bytes, hardware, device


def load_original_model_and_inputs(
    model_original_src: str, context: dict
) -> tuple[nn.Module, callable, callable]:
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement
    """

    try:
        compile(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None

    try:
        exec(model_original_src, context)  # expose to current namespace
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None

    # these should be defined in the original model code and present in the context
    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get("Model")
    from kb_workflows.input_profiles import wrap_factory

    get_inputs_fn = wrap_factory(model_original_src, get_inputs_fn)
    return (Model, get_init_inputs_fn, get_inputs_fn)


def load_custom_model(
    model_custom_src: str, context: dict, build_directory: str = None
) -> nn.Module:
    """
    Load class from custom NN.module pytorch code
    this is the code output by LLM with calls to custom cuda kernels
    """
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        os.environ["TORCH_EXTENSIONS_DIR"] = str(build_directory)

    try:
        from kb_workflows.allocation_audit import candidate_scope
        from kb_workflows.cache_audit import candidate_guard as cache_guard
        from kb_workflows.throughput_audit import candidate_guard

        with candidate_scope("import"), candidate_guard(), cache_guard():
            if build_directory:
                from kb_workflows.submission_source import execute

                execute(model_custom_src, context, build_directory)
            else:
                exec(compile(model_custom_src, "<string>", "exec"), context)
        # DANGER: need to delete refernece from global namespace
    except SyntaxError as e:
        print(f"Syntax Error in custom generated code or Compilation Error {e}")
        return None

    ModelNew = context.get("ModelNew")
    return ModelNew


def _cleanup_cuda_extensions():
    """Helper function to cleanup compiled CUDA extensions"""
    # SIMON NOTE: is this necessary?
    import shutil

    torch_extensions_path = os.path.join(
        os.path.expanduser("~"), ".cache", "torch_extensions"
    )
    if os.path.exists(torch_extensions_path):
        shutil.rmtree(torch_extensions_path)


def graceful_eval_cleanup(curr_context: dict, device: torch.device):
    """
    Clean up env, gpu cache, and compiled CUDA extensions after evaluation
    """  # delete ran-specific function definitions before next eval run
    del curr_context
    # Clear CUDA cache and reset GPU state
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

        # does this help?
        torch.cuda.reset_peak_memory_stats(device=device)

        torch.cuda.synchronize(
            device=device
        )  # Wait for all CUDA operations to complete

    # _cleanup_cuda_extensions() # SIMON NOTE: is this necessary?

def build_compile_cache_legacy(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None,
) -> tuple[bool, str, str]:
    """
    Try to build the compiled cuda code for sample and store in the cache directory
    Should be able to run on CPUs to do this massively in parallel

    Don't limit ninja to set default number of workers, let it use all the cpu cores possible

    NOTE: currently stdout_buffer does not capture all the compiler warning and failure messages
    Returns:
        tuple[bool, str]: whether compilation is successful, stdout content as string
    """
    context = {}
    stdout_buffer = StringIO()

    if verbose:
        print("[Compilation] Pre-compile custom cuda binaries")

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # sys.stdout.flush()

        # Capture stdout during compilation
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            load_custom_model(custom_model_src, context, build_dir)
            # sys.stdout.flush()

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
    except Exception as e:
        print(f"[Compilation] Failed to compile custom CUDA kernel. Unable to cache, \nError: {e}")
        return False, stdout_buffer.getvalue(), str(e)

    return True, stdout_buffer.getvalue(), None



def build_compile_cache(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None,
) -> tuple[bool, str, str]:
    """
    Try to build the compiled cuda code for sample and store in the cache directory
    Should be able to run on CPUs to do this massively in parallel

    Don't limit ninja to set default number of workers, let it use all the cpu cores possible
    # try do this with a subprocess
    NOTE: currently stdout_buffer does not capture all the compiler warning and failure messages
    Returns:
        tuple[bool, str]: whether compilation is successful, stdout content as string
    """
    context = {}
    stdout_buffer = StringIO()

    if verbose:
        print("[Compilation] Pre-compile custom cuda binaries")

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # sys.stdout.flush()

        # Capture stdout during compilation
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            load_custom_model(custom_model_src, context, build_dir)
            # sys.stdout.flush()

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
    except Exception as e:
        print(f"[Compilation] Failed to compile custom CUDA kernel. Unable to cache, \nError: {e}")
        return False, stdout_buffer.getvalue(), str(e)

    return True, stdout_buffer.getvalue(), None


def build_compile_cache_with_capturing(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None
) -> tuple[int, str, str]:
    """
    Write a temporary python file to compile the custom model on CPU
    Captures the return code, stdout, and stderr
    This works for capturing, build_compile_cache does not
    """
    if build_dir:
        # Add import at the start of the source code
        custom_model_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
        ) + custom_model_src

    kernel_hash = hash(custom_model_src)
    # tmp is a temp python file we write to for compilation
    tmp = os.path.join(build_dir, f"tmp_{kernel_hash}.py")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)

    with open(tmp, "w", encoding="utf-8") as f:
        f.write(custom_model_src)

    # Execute the temporary Python file and capture output
    process = subprocess.Popen(['python', tmp], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    returncode = process.returncode

    # Clean up temporary file
    os.remove(tmp)


    if verbose:
        print("[CPU Precompile] return code: ", returncode)
        print("[CPU Precompile] stdout: \n", stdout.decode('utf-8'))
        print("[CPU Precompile] stderr: \n", stderr.decode('utf-8'))

    return returncode, stdout.decode('utf-8'), stderr.decode('utf-8')




def get_hidden_test_path(level: int, problem_id: int) -> str:
    """
    Return the path to the hidden test file for a given (level, problem_id).
    Returns None if the file doesn't exist.
    """
    path = os.path.join(HIDDEN_TESTS_PATH, f"level{level}", f"{problem_id}_hidden.py")
    return path if os.path.exists(path) else None


def load_hidden_inputs(hidden_test_path: str) -> list | None:
    """
    Load and return the hidden input configs from a hidden test file.
    Returns a list of input-lists, or None on failure.

    Each element of the returned list is a list of raw tensors / values —
    exactly what get_hidden_inputs() returns.
    """
    if hidden_test_path is None or not os.path.exists(hidden_test_path):
        return None

    context = {"torch": torch, "__file__": hidden_test_path, "__builtins__": __builtins__}
    try:
        with open(hidden_test_path, "r") as f:
            src = f.read()
        exec(compile(src, hidden_test_path, "exec"), context)
    except Exception as e:
        print(f"[Hidden] Failed to load {hidden_test_path}: {e}")
        return None

    get_hidden_inputs_fn = context.get("get_hidden_inputs")
    if get_hidden_inputs_fn is None:
        return None

    try:
        return get_hidden_inputs_fn()
    except Exception as e:
        print(f"[Hidden] get_hidden_inputs() failed for {hidden_test_path}: {e}")
        return None


def run_correctness_on_fixed_inputs(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    raw_inputs: list,
    metadata: dict,
    verbose: bool = False,
    device=None,
    precision: torch.dtype = torch.float32,
    config_label: str = "",
    seed: int = 42,
) -> bool:
    """
    Run one correctness check on a fixed list of pre-generated inputs.
    Returns True if the custom model matches the reference model, False otherwise.

    Unlike run_and_check_correctness(), this function does NOT call get_inputs()
    to generate random tensors — it uses the provided raw_inputs directly.
    This is used for hidden test configs where inputs are fixed and deterministic.
    """
    inputs = _process_input_group(raw_inputs, device, precision)
    with torch.no_grad():
        issue = check_paired_calls(
            move_model(original_model_instance, device, precision),
            move_model(new_model_instance, device, precision), inputs,
            seed=seed, tolerance=get_tolerance_for_precision(precision), device=device,
            set_seed=set_seed, metadata=metadata, label=config_label,
        )
    if issue:
        metadata["hidden_correctness_issue"] = issue
        if verbose:
            print(f"[Hidden] FAIL: {issue}")
    return issue is None


def run_hidden_correctness_check(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    hidden_test_path: str,
    metadata: dict,
    verbose: bool = False,
    device=None,
    precision: torch.dtype = torch.float32,
    seeds=None,
) -> bool:
    """Regenerate every hidden config under three seeds; every check must execute.

    Reviewed tasks additionally stress their first nominal config. Reference errors
    invalidate validation rather than silently counting as successful checks.
    """
    failed_configs = []
    completed = 0
    seeds = HIDDEN_SEEDS if seeds is None else seeds
    if (not isinstance(seeds, (tuple, list)) or not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("Hidden seeds must be a nonempty list of distinct nonnegative integers")
    metadata["hidden_validation_seeds"] = list(seeds)
    metadata["numeric_stress_enabled"] = stress_enabled(hidden_test_path)
    for seed in seeds:
        set_seed(seed)
        hidden_configs = load_hidden_inputs(hidden_test_path)
        if not isinstance(hidden_configs, (list, tuple)) or not hidden_configs:
            metadata["hidden_validation_error"] = f"seed {seed}: missing, malformed or empty hidden suite"
            raise RuntimeError(f"[Hidden] {metadata['hidden_validation_error']}: {hidden_test_path}")
        for i, raw_inputs in enumerate(hidden_configs):
            if not isinstance(raw_inputs, (list, tuple)):
                metadata["hidden_validation_error"] = f"seed {seed}/config_{i + 1}: inputs must be a list or tuple"
                raise RuntimeError(f"[Hidden] {metadata['hidden_validation_error']}")

            def variants():
                yield "nominal", raw_inputs
                if i == 0 and metadata["numeric_stress_enabled"]:
                    yield from stress_cases(raw_inputs, seed)

            for name, inputs in variants():
                label = f"seed_{seed}/config_{i + 1}/{name}"
                passed = run_correctness_on_fixed_inputs(
                    original_model_instance, new_model_instance, inputs, metadata,
                    verbose=verbose, device=device, precision=precision,
                    config_label=label, seed=seed,
                )
                completed += 1
                if not passed:
                    failed_configs.append(label)
        # Release this seed's tensors before generating the next seed's suite.
        del hidden_configs, raw_inputs, inputs

        from kb_workflows.operator_input_policy import completed as operator_case_completed
        from kb_workflows.operator_input_policy import extra_cases

        for row in extra_cases(original_model_instance, seed):
            effective = _process_input_group(row["inputs"], device, precision)
            passed = run_correctness_on_fixed_inputs(
                original_model_instance, new_model_instance, effective, metadata,
                verbose=verbose, device=device, precision=precision,
                config_label=row["case_label"], seed=seed)
            operator_case_completed(row, effective, passed)
            completed += 1
            if not passed:
                failed_configs.append(row["case_label"])
            del effective
    metadata["hidden_failed_configs"] = failed_configs
    metadata["hidden_completed_checks"] = completed
    metadata["hidden_correctness_trials"] = f"({completed - len(failed_configs)}/{completed} hidden configs passed)"
    return not failed_configs


def measure_hidden_speedup(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    hidden_test_path: str,
    num_perf_trials: int = 10,
    verbose: bool = False,
    device=None,
    precision: torch.dtype = torch.float32,
) -> float | None:
    """
    Measure geometric mean speedup across all hidden test configs.

    For each config, time both the reference model and custom model, compute
    speedup = ref_time / custom_time, then return geomean across all configs.

    Returns None if hidden test file is not found or timing fails.

    NOTE: The original standard test inputs (torch.rand, default shape) are NOT
    included in this metric, per design. Only hidden configs are used.
    """
    hidden_configs = load_hidden_inputs(hidden_test_path)

    if hidden_configs is None:
        return None

    speedups = []
    for i, raw_inputs in enumerate(hidden_configs):
        config_label = f"config_{i + 1}"
        try:
            inputs = _process_input_group(raw_inputs, device, precision)
            model = move_model(original_model_instance, device, precision)
            model_new = move_model(new_model_instance, device, precision)

            # Time reference model
            ref_times = time_execution_with_cuda_event(
                model, *inputs, num_trials=num_perf_trials, verbose=False, device=device
            )
            # Time custom model
            custom_times = time_execution_with_cuda_event(
                model_new, *inputs, num_trials=num_perf_trials, verbose=False, device=device
            )

            ref_mean = np.mean(ref_times)
            custom_mean = np.mean(custom_times)

            if custom_mean > 0 and ref_mean > 0:
                speedup = ref_mean / custom_mean
                speedups.append(speedup)
                if verbose:
                    print(f"[Hidden] {config_label}: ref={ref_mean:.3g}ms, custom={custom_mean:.3g}ms, speedup={speedup:.3f}x")
        except Exception as e:
            if verbose:
                print(f"[Hidden] Speedup measurement failed for {config_label}: {e}")

    if not speedups:
        return None

    geomean = float(np.exp(np.mean(np.log(speedups))))
    if verbose:
        print(f"[Hidden] Geometric mean speedup across {len(speedups)} configs: {geomean:.3f}x")
    return geomean


def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    verbose: bool = False,
    measure_performance: bool = False,
    build_dir: os.PathLike = None,
    device: torch.device = torch.cuda.current_device() if torch.cuda.is_available() else None, # have to run on GPU
    precision: torch.dtype = torch.float32,
    hidden_test_path: str = None,  # path to {pid}_hidden.py, or None to skip
    hidden_seeds=None,
) -> KernelExecResult:
    """
    Evaluate the custom kernel against the original model

    num_correct_trials: number of trials to initialize different random inputs; correctness pass only if all trials pass
    num_perf_trials: run the evalutation many times to take the average
    device: GPU (cuda) device to run the evalutation on
    """
    # TODO: check device is busy
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elements at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )

    # set CUDA device
    torch.cuda.set_device(device)

    context = {}

    if verbose:
        print(f"[Eval] Start Evalulation! on device: {device}")
        print("[Eval] Loading Original Model")

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        original_model_src, context
    )
    set_seed(seed_num)  # set seed for reproducible input
    init_inputs = get_init_inputs()
    init_inputs = [_process_input_tensor(x, device, precision) for x in init_inputs]

    with torch.no_grad():
        set_seed(seed_num)  # set seed for reproducible weights
        original_model = Model(*init_inputs)
        assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")
    if verbose:
        print("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

    metadata = {}  # for storing result metadata
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)  # for debugging

    original_model = move_model(original_model, device, precision)
    synchronize(device)
    candidate_setup_started = CLOCK()

    # this is where compilation happens
    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # add hash for later to distinguish between multi-turn kernels
        ModelNew = load_custom_model(custom_model_src, {}, build_dir)
        torch.cuda.synchronize(device=device)  # not sure if this is too much
    except Exception as e:
        print(
            f"Failed to compile custom CUDA kernel: Record as compilation failure. \nError: {e}"
        )
        # TODO: add metadata for compilation error (how to we get the compilation error message?)

        if "/lock" in str(e):
            # this is a lock file error, likely due to concurrent compilation
            # this does not necessarily mean the compilation failed, but we should retry
            print(
                f"[Eval] Lock file error during compilation, Please retry. Error: {e}"
            )
            graceful_eval_cleanup(context, device)
            return None
        else:
            metadata["compilation_error"] = e
            graceful_eval_cleanup(context, device)
            return KernelExecResult(
                compiled=False, metadata=metadata
            )  # skip further steps

    # at this point we passed compilation
    try:
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            from kb_workflows.allocation_audit import candidate_scope
            from kb_workflows.cache_audit import candidate_guard as cache_guard
            from kb_workflows.throughput_audit import candidate_guard

            with candidate_scope("constructor"), candidate_guard(), cache_guard():
                custom_model = ModelNew(*init_inputs)
                assert hasattr(custom_model, "forward")
                custom_model = move_model(custom_model, device, precision)
                torch.cuda.synchronize(device=device)
            original_model = move_model(original_model, device, precision)
        if verbose:
            print("[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        print(
            f"Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. \nError: {e}"
        )
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        graceful_eval_cleanup(context, device)
        metadata["runtime_error"] = e
        return KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )  # skip further steps

    metadata["candidate_setup_ms"] = (CLOCK() - candidate_setup_started) * 1000
    kernel_exec_result = None

    # Check Correctness
    if verbose:
        print("[Eval] Checking Correctness")
    try:
        kernel_exec_result = run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
            precision=precision,
        )
    except Exception as e:
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        metadata["runtime_error"] = e
        kernel_exec_result = KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )

    # Hidden Correctness Check [Optional] | run after standard correctness
    # Only runs if hidden_test_path is provided; uses adversarial distributions & shape variants.
    # Standard correctness and every hidden config must pass to mark a kernel correct.
    if hidden_test_path is not None and kernel_exec_result and kernel_exec_result.correctness:
        print(f"[Hidden] Running hidden correctness check: {os.path.basename(hidden_test_path)}")
        try:
            hidden_passed = run_hidden_correctness_check(
                original_model,
                custom_model,
                hidden_test_path,
                metadata=metadata,
                verbose=verbose,
                device=device,
                precision=precision,
                **({"seeds": hidden_seeds} if hidden_seeds is not None else {}),
            )
            if not hidden_passed:
                print("[Hidden] HIDDEN CHECK FAILED — marking kernel as incorrect")
                kernel_exec_result = KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )
            else:
                # Propagate hidden pass metadata back into the result
                kernel_exec_result = KernelExecResult(
                    compiled=True, correctness=True, metadata=metadata
                )
        except RuntimeError as e:
            # Fatal config error (e.g. hidden file not found) — crash loudly
            raise
        except Exception as e:
            # Per-sample failure (CUDA error, OOM from scaled inputs, etc.)
            # Mark as failed but don't crash the worker — let other samples continue
            print(f"[Hidden] Exception during hidden correctness check: {e}")
            metadata["hidden_check_error"] = str(e)
            kernel_exec_result = KernelExecResult(
                compiled=True, correctness=False, metadata=metadata
            )

    # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness:
                if verbose:
                    print("[Eval] Measuring Performance as Sample is Correct")

                torch.cuda.synchronize(device=device)
                set_seed(seed_num)
                inputs = get_inputs()
                inputs = [_process_input_tensor(x, device, precision) for x in inputs]
                model_new = move_model(custom_model, device, precision)
                torch.cuda.synchronize(device=device)

                timing_receipt = metadata["timing_receipt"] = {}

                def input_factory(trial):
                    from kb_workflows.measurement import fresh_seed
                    seed = fresh_seed()
                    timing_receipt.setdefault("input_seeds", []).append(seed)
                    set_seed(seed)
                    return [_process_input_tensor(x, device, precision) for x in get_inputs()]

                def verifier(fresh, actual):
                    from kb_workflows.input_calibration import verify_output

                    expected = clone_tensors(original_model(*fresh))
                    synchronize(device)
                    verify_output(original_model, fresh, expected, actual)
                    issue = compare_outputs(expected, actual, get_tolerance_for_precision(precision))
                    if issue:
                        raise RuntimeError(f"Timed output did not match varied input: {issue}")

                elapsed_times = synchronized_times(
                    model_new, inputs, device, num_trials=num_perf_trials,
                    input_factory=input_factory, verifier=verifier, receipt=timing_receipt,
                )
                metadata["cold_start_ms"] = metadata.get("cold_start_ms", 0) + timing_receipt.get("warmup_total_ms", 0)
                metadata.update(input_signature=input_signature(inputs), timing_method="synchronized_wall",
                                output_consumed=True, fresh_timing_inputs=True)
                runtime_stats = get_timing_stats(elapsed_times, device=device)

                if verbose:
                    print(f"[Eval] Performance Stats: {runtime_stats}")
                kernel_exec_result.runtime = runtime_stats["mean"]
                kernel_exec_result.runtime_stats = runtime_stats

                memory_stats = get_memory_stats(model_new, *inputs, device=device)
                kernel_exec_result.peak_memory = memory_stats["peak_bytes"]
                kernel_exec_result.memory_stats = memory_stats
        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance"] = e
            kernel_exec_result.correctness = False
            kernel_exec_result.runtime = -1.0

    kernel_exec_result.metadata.update(metadata)
    graceful_eval_cleanup(context, device)
    return kernel_exec_result


def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate=False,
    max_length=200,
):
    """
    max_length characters

    NOTE: I can't get torch truncate to work during exception handling so I have this for now
    """
    # Truncate exception message if too long
    exception_str = str(exception_msg)
    if truncate and len(exception_str) > max_length:
        exception_str = exception_str[: max_length - 3] + "..."

    if verbose:
        print(f"[Exception {exception_type}] {exception_str} ")
    metadata[exception_type] = exception_str

    return metadata


def time_execution_with_cuda_event(
    kernel_fn: callable, *args, num_warmup: int = 3, num_trials: int = 10,
    verbose: bool = True, device: torch.device = None,
) -> list[float]:
    """Compatibility name for synchronized wall milliseconds, including dispatch.

    This legacy interface clones inputs; batch evaluation additionally supplies
    fresh input contents and verifies every timed output.
    """
    device = torch.cuda.current_device() if device is None else device
    return synchronized_times(kernel_fn, args, device, num_warmup, num_trials)


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose=False,
    seed=42,
    device=None,
    precision: torch.dtype = torch.float32,
) -> KernelExecResult:
    """
    run the model and check correctness,
    assume model already loaded and compiled (loaded and compiled in the caller)
    this is all on GPU, requiring cuda device and transfer .cuda()

    num_correct_trials: run the evalutation multiple times with (ideally) different random inputs to ensure correctness
    """
    if num_correct_trials <= 0:
        metadata["correctness_issue"] = "Correctness requires at least one trial"
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)
    pass_count = 0
    from kb_workflows.throughput_audit import bind_factory

    bind_factory(lambda: _process_input_group(get_inputs_fn(), device, precision))
    from kb_workflows.cache_audit import bind_factory as bind_cache_factory

    bind_cache_factory(lambda: _process_input_group(get_inputs_fn(), device, precision))
    generator = torch.Generator().manual_seed(seed)
    seeds = torch.randint(0, 2**32 - 1, (num_correct_trials,), generator=generator).tolist()
    with torch.no_grad():
        model = move_model(original_model_instance, device, precision)
        model_new = move_model(new_model_instance, device, precision)
        for trial, trial_seed in enumerate(seeds):
            set_seed(trial_seed)
            inputs = _process_input_group(get_inputs_fn(), device, precision)
            issue = check_paired_calls(
                model, model_new, inputs, seed=trial_seed,
                tolerance=get_tolerance_for_precision(precision), device=device,
                set_seed=set_seed, metadata=metadata, label=f"standard_{trial + 1}",
                record_first_call=trial == 0, clock=CLOCK,
            )
            if issue:
                metadata["correctness_issue"] = issue
                if verbose:
                    print(f"[FAIL] {issue}")
            else:
                pass_count += 1
    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"
    return KernelExecResult(compiled=True, correctness=pass_count == num_correct_trials, metadata=metadata)


def check_metadata_serializable(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings
    """
    try:
        json.dumps(metadata)
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings
        metadata = {
            "eval_0": {
                k: (
                    str(v)
                    if not isinstance(
                        v, (dict, list, str, int, float, bool, type(None))
                    )
                    else v
                )
                for k, v in metadata["eval_0"].items()
            }
        }
        print(
            f"[WARNING] Metadata now converted to string: {metadata} to be JSON serializable"
        )

    return metadata

def check_metadata_serializable_all_types(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings recursively
    """
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    try:
        json.dumps(metadata)
        return metadata
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings recursively
        converted_metadata = convert_to_serializable(metadata)
        print(
            f"[WARNING] Metadata now converted to be JSON serializable: {converted_metadata}"
        )
        return converted_metadata


################################################################################
# Performance Eval
################################################################################


def fetch_baseline_time(
    level_name: str, problem_id: int, dataset: list[str], baseline_time_filepath: str
) -> dict:
    """
    Fetch the baseline time from the time
    """
    if not os.path.exists(baseline_time_filepath):
        raise FileNotFoundError(
            f"Baseline time file not found at {baseline_time_filepath}"
        )

    with open(baseline_time_filepath, "r") as f:
        baseline_json = json.load(f)

    problem_name = dataset[problem_id].split("/")[-1]
    baseline_time = baseline_json[level_name].get(problem_name, None)
    return baseline_time


def get_timing_stats(elapsed_times: list[float], device: torch.device = None) -> dict:
    """Get timing statistics from a list of elapsed times.

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: CUDA device, record device info
    Returns:
        Dict containing mean, std, min, max and num_trials
        all timing are in ms
    """

    stats = timing_stats(elapsed_times)

    if device:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)  # for debugging

    return stats


def get_memory_stats(kernel_fn: callable, *args, device: torch.device = None) -> dict:
    """Measure peak GPU memory allocation for a single kernel invocation.

    Args:
        kernel_fn: Function to measure
        *args: Arguments to pass to kernel_fn
        device: CUDA device to use, if None uses current device

    Returns:
        Dict with peak_bytes, hardware, device
    """
    if device is None:
        device = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(device)
    kernel_fn(*args)
    torch.cuda.synchronize(device)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return {
        "peak_bytes": peak_bytes,
        "hardware": torch.cuda.get_device_name(device=device),
        "device": str(device),
    }


# if __name__ == "__main__":
# print(fetch_ref_arch_from_level_problem_id("2", 1, with_name=True))
# fetch_baseline_time("level1", 0, ["1_Square_matrix_multiplication_.py"], "tests/baseline_time_matx3.json")
