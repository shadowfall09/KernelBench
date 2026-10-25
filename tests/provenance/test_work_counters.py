"""CPU controls and vendor-format fixtures, never hardware counter measurements."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from kb_workflows import engine
from kb_workflows.analysis import calibration, performance
from kb_workflows.common import ROOT, atomic_json, evaluator_digest, fingerprint, kernel_name
from kb_workflows.config import RunConfig
from kb_workflows.numerics.precision_calibration import numeric_class
from kb_workflows.provenance import counter_task, counter_worker
from kb_workflows.provenance import work_counters as wc
from scripts.audit_work_counters import prepare
from tests.numerics.test_input_calibration import CAPS, CPUAdapter

READ, WRITE = "dram__bytes_read.sum", "dram__bytes_write.sum"
FMA = "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
SOURCE = "import torch\nclass ModelNew(torch.nn.Module):\n    def forward(self,a,b):\n        return a @ b\n"


def policy(*, complete=True, source=None):
    review = {"reviewer": "unscored format fixture", "rationale": "toy scalar/memory mapping, never device evidence"}
    if complete:
        review.update(
            source_sha256=[source or hashlib.sha256(SOURCE.encode()).hexdigest()],
            numeric_class_sha256=numeric_class(CAPS),
        )
        review["workload_contracts"] = [
            {
                **{k: "1" * 64 for k in wc.WORKLOAD_FIELDS if k != "precision"},
                "source_sha256": review["source_sha256"][0],
                "precision": "fp32",
            }
        ]
    return {
        "protocol": wc.RECIPE,
        "coverage": "reviewed_complete_workload" if complete else "reported_resources_only",
        "review": review,
        "resources": {
            "memory_bytes": {
                "rate_unit": "bytes/s",
                "terms": [
                    {"metric": READ, "unit": "byte", "coefficient": 1},
                    {"metric": WRITE, "unit": "byte", "coefficient": 1},
                ],
            },
            "fp32_flops": {"rate_unit": "operations/s", "terms": [{"metric": FMA, "unit": "inst", "coefficient": 2}]},
        },
    }


def csv_report(pid=17, *, launches=2, bom=False):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(wc.FIELDS)
    for index in range(launches):
        for name, unit, value in ((READ, "byte", "32"), (WRITE, "byte", "64"), (FMA, "inst", "10")):
            writer.writerow([index, pid, 1, 2, 0, f"opaque name {index}", name, unit, value])
    return ("\ufeff" if bom else "").encode() + output.getvalue().encode()


def binding():
    return {
        **{key: "1" * 64 for key in wc.BINDINGS},
        "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
        "evaluator_sha256": evaluator_digest(),
        "precision": "fp32",
        "target": "candidate",
        "seed": 42,
        "pid": 17,
        "device": copy.deepcopy(CAPS),
        "device_identity_sha256": calibration.identity(CAPS),
        "output_verified": True,
        "environment_changes": [],
        "initialization_seed": 42,
        "forward_seed": 42,
        "training": True,
        "no_grad": True,
    }


def tool():
    version = "Unscored simulated NCU schema, not an installed device capture"
    return {
        "name": "ncu",
        "sha256": "2" * 64,
        "version": version,
        "version_sha256": hashlib.sha256(version.encode()).hexdigest(),
        "binary_unchanged": True,
        "exit_code": 0,
        "timed_out": False,
        "configuration": counter_task.CONFIGURATION,
    }


def reseal(record):
    record["receipt_sha256"] = fingerprint({key: value for key, value in record.items() if key != "receipt_sha256"})


@pytest.fixture
def format_record():
    record = wc.record(csv_report(), policy(), binding(), tool(), simulated=True)
    wc.validate(record, allow_simulated=True)
    # Unscored serialization fixture only. Never persisted in the repository or used in experiments.
    record["simulated"] = False
    reseal(record)
    return record


def device_calibration():
    probes = {
        "memory_bytes": calibration.probe_record(100_000, [100] * 5, unit="bytes/s", parameters={}),
        "fp32_flops": calibration.probe_record(20_000, [10] * 5, unit="operations/s", parameters={}),
    }
    return calibration.seal(CAPS, probes, [0.01] * 5, settings={"basis": "unscored schema fixture only"})


@pytest.mark.parametrize("bom", [False, True])
def test_raw_vendor_counts_keep_every_launch_and_exact_units(bom):
    raw = csv_report(bom=bom)
    result = wc.parse_csv(raw, policy(), pid=17)
    assert result["launch_count"] == 2 and result["work"] == {"memory_bytes": 192, "fp32_flops": 40}
    assert result["report_sha256"] == hashlib.sha256(raw).hexdigest()
    record = wc.record(raw, policy(), binding(), tool(), simulated=True)
    assert wc.validate(json.loads(json.dumps(record)), allow_simulated=True) == record
    with pytest.raises(ValueError, match="simulated"):
        wc.validate(record)


@pytest.mark.parametrize(
    "value,expected", [("1,234", 1234), ("1.000e3", 1000), ("0.0", 0), (str(2**64 - 1), 2**64 - 1)]
)
def test_counts_parse_exact_base_units_without_float_rounding(value, expected):
    assert wc.count(value) == expected


@pytest.mark.parametrize(
    "value", ["N/A", "nan", "inf", "-1", "1.5", "1,2", "1,,234", "12 Kbyte", "1;2", str(2**64), "1e100"]
)
def test_invalid_scaled_instanced_or_missing_counts_cannot_become_zero(value):
    with pytest.raises(ValueError):
        wc.count(value)


@pytest.mark.parametrize(
    "damage", ["unit", "duplicate", "missing", "pid", "device", "context", "bad_row", "bad_header", "empty"]
)
def test_incomplete_wrong_process_and_ambiguous_csv_remains_unmeasured(damage):
    rows = list(csv.reader(io.StringIO(csv_report().decode())))
    if damage == "unit":
        rows[1][-2] = "Kbyte"
    elif damage == "duplicate":
        rows.append(rows[1])
    elif damage == "missing":
        rows.pop()
    elif damage == "pid":
        rows[1][1] = "99"
    elif damage == "device":
        rows[1][4] = "1"
    elif damage == "context":
        rows[2][2] = "2"
    elif damage == "bad_row":
        rows[1].pop()
    elif damage == "bad_header":
        rows[0][-1] = "Unknown Value"
    elif damage == "empty":
        rows = rows[:1]
    output = io.StringIO()
    csv.writer(output).writerows(rows)
    with pytest.raises(ValueError):
        wc.parse_csv(output.getvalue().encode(), policy(), pid=17)


@pytest.mark.parametrize("damage", ["totals", "raw", "binding", "tool", "schema", "coverage", "signal"])
def test_resealed_counter_summaries_cannot_replace_raw_observations(format_record, damage):
    row = copy.deepcopy(format_record)
    if damage == "totals":
        row["observed"]["work"]["fp32_flops"] += 1
    elif damage == "raw":
        row["raw_csv"] = row["raw_csv"].replace("10\r\n", "11\r\n", 1)
    elif damage == "binding":
        row["binding"]["evaluator_sha256"] = "a" * 64
    elif damage == "tool":
        row["tool"]["binary_unchanged"] = False
    elif damage == "schema":
        row["recipe"]["resources"]["fp32_flops"]["terms"][0]["metric"] = "regex:.*"
    elif damage == "coverage":
        row["recipe"]["coverage"] = "silently_complete"
    elif damage == "signal":
        row["binding"]["output_verified"] = False
    reseal(row)
    with pytest.raises(ValueError):
        wc.validate(row)


def test_execution_model_replays_units_counts_rates_and_populates_ae_without_manual_latency(format_record):
    model = wc.execution_model(format_record, device_calibration())
    assert model["sol_execution_ms"] == 0.192
    result = performance.decomposition(1, 0.5, 0.1, counters=model)
    assert result["hardware_counters_collected"] and math.isclose(0.1 * result["A"] * result["E"] * result["O"], 1)
    for damage in ("sol", "calibration", "components"):
        changed = copy.deepcopy(model)
        if damage == "sol":
            changed["sol_execution_ms"] *= 2
        elif damage == "calibration":
            changed["calibration"]["probes"]["memory_bytes"]["rate"] *= 2
        else:
            changed["components_ms"]["memory_bytes"] *= 2
        reseal(changed)
        with pytest.raises(ValueError, match="source-bound"):
            performance.decomposition(1, 0.5, 0.1, sol_execution_ms=0.192, counters=changed)
    with pytest.raises(ValueError):
        performance.decomposition(1, 0.5, 0.1, sol_execution_ms=99, counters=model)


@pytest.mark.parametrize(
    "damage", ["partial", "wrong_source", "wrong_class", "units", "other_device", "input", "precision", "policy"]
)
def test_partial_or_wrongly_bound_work_cannot_separate_algorithm_and_efficiency(format_record, damage):
    row, calibrated = copy.deepcopy(format_record), device_calibration()
    if damage == "partial":
        row["recipe"] = policy(complete=False)
    elif damage == "wrong_source":
        row["recipe"]["review"]["source_sha256"] = ["f" * 64]
    elif damage == "wrong_class":
        row["recipe"]["review"]["numeric_class_sha256"] = "f" * 64
    elif damage == "units":
        row["recipe"]["resources"]["memory_bytes"]["rate_unit"] = "operations/s"
    elif damage == "other_device":
        row["binding"]["device"]["uuid"] = "other-serialization-device"
        row["binding"]["device_identity_sha256"] = calibration.identity(row["binding"]["device"])
    elif damage == "input":
        row["binding"]["input_signature_sha256"] = "f" * 64
    elif damage == "precision":
        row["binding"]["precision"] = "fp16"
    elif damage == "policy":
        row["binding"]["policy_sha256"] = "f" * 64
    reseal(row)
    with pytest.raises(ValueError):
        wc.execution_model(row, calibrated)
    result = performance.decomposition(1, 0.5, 0.1, counters={"status": "measured", "source_sha256": "fake"})
    assert (
        result["A"] is None
        and result["E"] is None
        and not result["hardware_counters_collected"]
        and result["counter_issue"]
    )


def cpu_call(
    tmp_path,
    *,
    wrong=False,
    numerical=False,
    tolerance=1e-4,
    target="candidate",
    precision="fp32",
    source=None,
    events=None,
):
    config = RunConfig(
        level=1,
        start=2,
        end=2,
        generation="local",
        runs_dir=str(tmp_path),
        fp32_tolerance=tolerance,
        precision=precision,
        numerical_validation=numerical,
        oracle_memory_mb=64,
    ).validate()
    config.run_dir.mkdir(parents=True)
    (config.run_dir / kernel_name(1, 2, 0)).write_text(
        source or (SOURCE.replace("a @ b", "a @ b + 1e-5") if wrong else SOURCE)
    )
    request, _ = prepare(config, 2, 0, policy(), target=target)
    from src.eval import load_original_model_and_inputs

    def loader(source, namespace):
        Model, init, _ = load_original_model_and_inputs(source, namespace)
        return (
            Model,
            init,
            lambda: [torch.randint(-4, 5, (3, 7)).float() * 0.25, torch.randint(-4, 5, (7, 5)).float() * 0.25],
        )

    events = [] if events is None else events
    result = counter_worker.run_call(
        request,
        adapter=CPUAdapter(),
        loader=loader,
        profiler=NS(start=lambda: events.append("start"), stop=lambda: events.append("stop")),
    )
    return config, request, result, events


@pytest.mark.parametrize("numerical", [False, True])
def test_one_public_cpu_call_observes_profiler_boundaries_and_freezes_policy_before_submission(tmp_path, numerical):
    _, request, result, events = cpu_call(tmp_path, numerical=numerical)
    assert events == ["start", "stop"] and result["simulated"] and result["binding"]["output_verified"]
    assert bool(result["numerical_gate"]) == numerical
    assert result["binding"]["input_signature_sha256"] == fingerprint({"signature": result["input_signature"]})
    counter_task.validate_call(request, result, CAPS)
    altered = copy.deepcopy(result)
    altered["binding"]["source_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        counter_task.validate_call(request, altered, CAPS)


def test_strict_configured_fp32_tolerance_is_used_in_the_instrumented_call(tmp_path):
    with pytest.raises(ValueError, match="frozen native"):
        cpu_call(tmp_path, wrong=True, tolerance=1e-8)


def test_parent_ncu_command_uses_only_fixed_arguments_and_simulated_results_stay_simulated(tmp_path):
    _, request, child, _ = cpu_call(tmp_path)
    binary = tmp_path / "unscored-ncu-stub"
    binary.write_text("unscored placeholder, never executed")
    binary.chmod(0o700)
    observed = []

    def runner(argv, timeout, log, **kwargs):
        observed.append(argv)
        if argv[-1] == "--version":
            Path(log).write_text("unscored NCU version fixture")
        else:
            assert "--config-file" in argv and argv[argv.index("--config-file") + 1] == "off"
            report = Path(argv[argv.index("--log-file") + 1])
            report.write_bytes(csv_report(child["binding"]["pid"]))
            atomic_json(Path(argv[-1]), child)
        return {"exit_code": 0, "timed_out": False}

    result = counter_task.audit(request, run=runner, binary=str(binary), device_caps=CAPS)
    assert result["status"] == "ok", result
    assert len(observed) == 2 and result["counters"]["simulated"]
    wc.validate(result["counters"], allow_simulated=True)


def test_counter_cli_preview_and_queue_calibration_dispatch_do_not_load_device_sdks(tmp_path):
    config, request, _, _ = cpu_call(tmp_path)
    from kb_workflows.runtime.queue_worker import validate_payload

    validate_payload(request)
    calibration_request = engine.task_payload(config, 2, task="calibrate_precision")
    validate_payload(calibration_request)
    calibration_request["kernel_src"] = SOURCE
    with pytest.raises(ValueError, match="no candidate"):
        validate_payload(calibration_request)
    policy_path = tmp_path / "counter-recipe.json"
    atomic_json(policy_path, policy())
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'[run]\nlevel=1\nstart=2\nend=2\nruns_dir={json.dumps(str(tmp_path))}\nrun_name="{config.run_name}"\n'
    )
    output = tmp_path / "never-created.json"
    code = (
        "import sys; from scripts.audit_work_counters import main; "
        f"assert main({['--config', str(config_path), '--recipe', str(policy_path), '--problem', '2', '--output', str(output)]!r}) == 0; "
        "assert not any(n in sys.modules for n in ('torch','triton','cupti','harbor','transformers'))"
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["experiments_started"] is False and not output.exists()


@pytest.mark.parametrize("target", ["candidate", "reference"])
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_stored_precision_and_reference_target_keep_unrounded_truth_and_do_not_invent_counter_measurements(
    tmp_path, target, precision
):
    _, request, result, events = cpu_call(tmp_path, target=target, precision=precision, numerical=True)
    assert result["binding"]["precision"] == precision and result["binding"]["target"] == target
    assert result["numerical_gate"]["reference_qualified"] and events == ["start", "stop"]
    counter_task.validate_call(request, result, CAPS)


def test_profiler_stops_even_when_the_instrumented_forward_raises(tmp_path):
    events = []
    source = (
        "import torch\nclass ModelNew(torch.nn.Module):\n"
        "    calls=0\n    def forward(self,a,b):\n        ModelNew.calls += 1\n"
        "        if ModelNew.calls == 2: raise RuntimeError('unit counter call failed')\n        return a @ b\n"
    )
    with pytest.raises(RuntimeError, match="unit counter call failed"):
        cpu_call(tmp_path, source=source, events=events)
    assert events == ["start", "stop"]


@pytest.mark.parametrize("mode", ["self.eval()", "torch.set_grad_enabled(True)"])
def test_constructor_cannot_weaken_the_frozen_training_and_grad_mode(tmp_path, mode):
    source = SOURCE.replace(
        "    def forward", f"    def __init__(self):\n        super().__init__()\n        {mode}\n    def forward"
    )
    events = []
    with pytest.raises(ValueError, match="training/gradient"):
        cpu_call(tmp_path, source=source, events=events)
    assert events == []


@pytest.mark.parametrize("mutation", ["mode", "environment"])
def test_trusted_build_setup_is_restored_but_submission_setting_changes_are_rejected(tmp_path, mutation):
    before = os.environ.get("TORCH_EXTENSIONS_DIR")
    line = (
        "self.eval()" if mutation == "mode" else "__import__('os').environ['TORCH_EXTENSIONS_DIR']='submission-change'"
    )
    source = SOURCE.replace("        return", f"        {line}\n        return")
    with pytest.raises(ValueError, match="bindings"):
        cpu_call(tmp_path, source=source)
    assert os.environ.get("TORCH_EXTENSIONS_DIR") == before


@pytest.mark.parametrize("failure", ["timeout", "metric_missing", "source", "device", "binary_changed"])
def test_parent_preserves_bounded_failure_artifacts_and_never_returns_partial_counts(tmp_path, failure):
    _, request, child, _ = cpu_call(tmp_path)
    binary = tmp_path / "unscored-ncu"
    binary.write_text("unused fixture tool")
    binary.chmod(0o700)

    def runner(argv, timeout, log, **kwargs):
        if argv[-1] == "--version":
            Path(log).write_text("unscored tool version")
        else:
            Path(log).write_text("unscored failure log")
            raw = csv_report(child["binding"]["pid"])
            if failure == "metric_missing":
                raw = b"\n".join(raw.splitlines()[:-1])
            Path(argv[argv.index("--log-file") + 1]).write_bytes(raw)
            changed = copy.deepcopy(child)
            if failure == "source":
                changed["binding"]["source_sha256"] = "f" * 64
            if failure == "device":
                changed["binding"]["device"]["uuid"] = "wrong-worker"
                changed["binding"]["device_identity_sha256"] = calibration.identity(changed["binding"]["device"])
            if failure == "binary_changed":
                binary.write_text("changed during simulated capture")
            atomic_json(Path(argv[-1]), changed)
            if failure == "timeout":
                return {"exit_code": 124, "timed_out": True}
        return {"exit_code": 0, "timed_out": False}

    result = counter_task.audit(request, run=runner, binary=str(binary), device_caps=CAPS)
    assert result["status"] == "not_measured" and "counters" not in result
    assert set(result["failure_artifacts"]) >= {"counters.csv", "application.log"}
    assert all(item["complete"] for item in result["failure_artifacts"].values())


def test_extreme_vendor_exponent_is_a_failed_count_instead_of_a_parser_exception():
    with pytest.raises(ValueError, match="envelope"):
        wc.count("1e999999999999999999999999999")


def test_failed_offline_model_keeps_the_completed_counter_report(tmp_path, monkeypatch, format_record):
    from scripts import audit_work_counters as cli

    config, _, _, _ = cpu_call(tmp_path)
    config_path = tmp_path / "source.toml"
    config_path.write_text(
        f'[run]\nlevel=1\nstart=2\nend=2\nruns_dir={json.dumps(str(tmp_path))}\nrun_name="{config.run_name}"\n'
    )
    recipe_path, calibration_path, output = (
        tmp_path / "recipe.json",
        tmp_path / "invalid-calibration.json",
        tmp_path / "counter-result.json",
    )
    atomic_json(recipe_path, policy())
    atomic_json(calibration_path, {})
    monkeypatch.setattr(
        cli, "evaluate_tasks", lambda *args: iter([("counters", {"status": "ok", "counters": format_record})])
    )
    assert (
        cli.main(
            [
                "--config",
                str(config_path),
                "--recipe",
                str(recipe_path),
                "--problem",
                "2",
                "--calibration",
                str(calibration_path),
                "--output",
                str(output),
                "--execute",
            ]
        )
        == 2
    )
    saved = json.loads(output.read_text())
    assert saved["counters"] == format_record and saved["execution_model"]["status"] == "not_measured"
