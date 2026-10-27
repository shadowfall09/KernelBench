"""CPU controls and vendor-format fixtures, never hardware counter measurements."""

from __future__ import annotations

import base64
import copy
import csv
import hashlib
import io
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from kb_workflows import engine
from kb_workflows.analysis import calibration, performance
from kb_workflows.common import ROOT, atomic_json, evaluator_digest, fingerprint, kernel_name
from kb_workflows.config import RunConfig
from kb_workflows.numerics.precision_calibration import numeric_class
from kb_workflows.provenance import counter_task, origin_task
from kb_workflows.provenance import work_counters as wc
from scripts.audit_work_counters import prepare
from tests.numerics.test_input_calibration import CAPS
from tests.provenance.test_kernel_origin import native_origin_runner as native_origin_runner

READ, WRITE = "dram__bytes_read.sum", "dram__bytes_write.sum"
FMA = "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
SOURCE = "import torch\nclass ModelNew(torch.nn.Module):\n    def forward(self,a,b):\n        return a @ b\n"


def _ncu_report(pid, count, *, binary=None, symbol="fixture_kernel", inline=False):
    """Synthetic version-7 bytes using NVIDIA's installed proto2 field numbers."""
    from tests.provenance.test_kernel_origin import BINARY

    def varint(value):
        output = bytearray()
        while value > 127:
            output.append((value & 127) | 128)
            value >>= 7
        return bytes(output) + bytes([value])
    def proto(*fields):
        return b"".join(varint(number * 8 + (2 if isinstance(value, bytes) else 0)) +
                        (varint(len(value)) + value if isinstance(value, bytes) else varint(value))
                        for number, value in fields)
    def sized(value):
        return struct.pack("<I", len(value)) + value
    source = proto((1, 77), (4, BINARY if binary is None else binary))
    definitions = b"" if inline else sized(source)
    payload = definitions + b"".join(sized(proto(
        (1, 1), (2, index + 1), (3, 1), (4, index), (5, symbol.encode()), (6, symbol.encode()), (7, symbol.encode()),
        (8, 3), (9, proto((1, 0), (2, 0), (3, 0))),
        (10, proto((1, 1), (2, 1), (3, 1))), (11, proto((1, 32), (2, 1), (3, 1))),
        (15, proto((1, 77 + index), (4, BINARY if binary is None else binary)) if inline else proto((1, 77))),
        (16, 0), (22, 1), (23, 2))) for index in range(count))
    header = proto((1, 0 if inline else 1), (2, count), (3, proto((1, pid))), (5, len(payload)))
    return b"NVR\0" + sized(proto((1, 7))) + sized(header) + payload


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
        "configuration": "config_file_off; kernel_replay; profiler_API_single_call; base_units; all_children",
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



@pytest.fixture
def native_counter_runner(tmp_path, native_origin_runner):
    report = native_origin_runner(activities=True, caps=CAPS)
    binary = tmp_path / "unscored-ncu-stub"
    binary.write_text("Unexecuted software-format fixture")
    binary.chmod(0o700)
    events, damage = [], {"kind": None}

    def runner(argv, timeout, log, **options):
        if argv[-1] == "--version":
            if damage["kind"] == "sdk":
                raise ImportError("simulated missing optional profiler SDK")
            Path(log).write_text("Unscored Nsight native protocol format fixture")
            return {"exit_code": 0, "timed_out": False}
        if "--log-file" not in argv:
            return native_origin_runner.runner(argv, timeout, log, **options)
        assert "--nvtx" in argv and argv[argv.index("--print-nvtx-rename") + 1] == "kernel"
        assert argv[argv.index("--print-kernel-base") + 1] == "mangled"
        assert set(wc.LAUNCH_METRICS) <= set(argv[argv.index("--metrics") + 1].split(","))
        for flag, expected in (("--config-file", "off"), ("--profile-from-start", "off"),
                               ("--cache-control", "none"), ("--clock-control", "none"), ("--rename-kernels", "off")):
            assert argv[argv.index(flag) + 1] == expected
        index = argv.index("-m")
        payload = json.loads(Path(argv[index + 2]).read_text())
        active = {"depth": -1}
        def push(marker):
            active["depth"] += 1
            events.append(("push", marker))
            return active["depth"]
        def pop():
            depth = active["depth"]
            active["depth"] -= 1
            events.append(("pop", None))
            return depth
        profiler = counter_task.Profiler(payload, functions=(lambda: events.append(("start", None)),
            lambda: events.append(("stop", None)), push, pop))
        document = origin_task._run_native(payload, lambda: native_origin_runner.invoke(payload), observer=profiler)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(wc.FIELDS)
        action = 0
        for window in document["counter_windows"]:
            for _ in range(2):
                for metric, unit, count in ((READ, "byte", 32), (WRITE, "byte", 64), (FMA, "inst", 10)):
                    writer.writerow([action, document["pid"], 1, 2, 0, window["marker"], metric, unit, count])
                for metric in wc.LAUNCH_METRICS:
                    if damage["kind"] == "missing_launch_identity" and metric == "launch__kernel_name":
                        continue
                    unit = "byte" if "shared_mem" in metric else "thread" if "block_dim" in metric else "block" if "grid_dim" in metric else ""
                    value = ("fixture_kernel" if metric == "launch__kernel_name" else 0 if "shared_mem" in metric else
                             32 if metric == "launch__block_dim_x" else 1)
                    writer.writerow([action, document["pid"], 1, 2, 0, window["marker"], metric, unit, value])
                    if damage["kind"] == "duplicate_launch_identity" and metric == "launch__kernel_name":
                        writer.writerow([action, document["pid"], 1, 2, 0, window["marker"], metric, unit, value])
                action += 1
        raw = output.getvalue()
        if damage["kind"] == "unscoped":
            raw = raw.replace(document["counter_windows"][0]["marker"], "unscoped_kernel")
        elif damage["kind"] == "output":
            document["calls"][-1]["output_content_sha256"] = "f" * 64
        elif damage["kind"] == "window":
            document["counter_windows"].pop()
        elif damage["kind"] == "binary":
            binary.write_text("changed simulated binary")
        Path(argv[argv.index("--log-file") + 1]).write_text(raw)
        Path(argv[argv.index("--export") + 1]).write_bytes(_ncu_report(document["pid"], action))
        Path(log).write_text("CPU native invocation simulation, no profiler or GPU loaded")
        atomic_json(Path(argv[index + 3]), document)
        return {"exit_code": 0, "timed_out": damage["kind"] == "timeout"}

    def collect(*, target="evaluate", role="candidate", failure=None, mutation="", scored=False):
        nonlocal report
        report = native_origin_runner(target=target, role=role, mutation=mutation, activities=True, caps=CAPS)
        payload = copy.deepcopy(report["native_request"])
        if scored:
            actual = native_origin_runner.scored(payload)
            payload, _ = origin_task.validate_scored(actual, allow_simulated=True)
            payload.update(_origin_target=target, _scored_result=actual)
        payload.update(task="counter_audit", _native_counter={"recipe": policy(complete=False)})
        damage["kind"] = failure
        events.clear()
        return counter_task.audit(payload, run=runner, binary=str(binary)), list(events)
    return collect


def test_native_counters_cover_complete_stateful_sequence_and_both_paired_roles(native_counter_runner):
    for target, role, count in (("evaluate", "candidate", 8), ("paired_timing", "candidate", 13),
                                ("paired_timing", "reference", 13)):
        result, events = native_counter_runner(target=target, role=role, scored=role == "candidate")
        assert result["status"] == "ok", result.get("reason")
        assert result["simulated"] and len(result["calls"]) == count
        assert [event[0] for event in events] == ["push", "start", "stop", "pop"] * count
        assert all(call["work"] == {"memory_bytes": 192, "fp32_flops": 40} for call in result["calls"])
        assert result["native_run"]["calls"] == result["origin"]["native_runs"]["control"]["calls"]
        assert result["origin"]["native_coverage"].get("scored_call_observations_match", False) is (role == "candidate")
        assert not result["scored_wall_times_replaced"] and not result["formal_qualification_implemented"]
        counter_task.validate(result, allow_simulated=True)
        archive = base64.b64decode(result["raw_report"]["base64"])
        identities = wc.report_binaries(archive, pid=result["native_run"]["pid"])
        assert len(identities) == count * 2
        assert all(call["kernel_identities"] and call["binary_identity_issue"] is None for call in result["calls"])
        with pytest.raises(ValueError, match="simulated"):
            counter_task.validate(result)
    inline = _ncu_report(result["native_run"]["pid"], count * 2, inline=True)
    assert wc.report_binaries(inline, pid=result["native_run"]["pid"]) == identities
    for damaged in (archive[:-1], archive + b"\0", b"NVR\0", archive.replace(b"\x08\x07", b"\x08\x08", 1),
                    archive.replace(b"\x08M", b"\x08N", 1),
                    _ncu_report(result["native_run"]["pid"] + 1, 1),
                    _ncu_report(result["native_run"]["pid"], 1, binary=b"PTX is not an ELF")):
        with pytest.raises(ValueError):
            wc.report_binaries(damaged, pid=result["native_run"]["pid"])
    for data in (b"\x08\x80", b"\0", b"\x0b"):
        with pytest.raises(ValueError):
            wc._proto(data)
    with pytest.raises(ValueError, match="Ambiguous"):
        wc._field(wc._proto(b"\x08\x01\x08\x02"), 1, 0)


def test_native_counters_reject_unmatched_ranges_state_outputs_and_incomplete_captures(native_counter_runner):
    for failure in ("unscoped", "output", "window", "binary", "timeout", "missing_launch_identity", "duplicate_launch_identity"):
        result, _ = native_counter_runner(failure=failure)
        assert result["status"] == "not_measured", failure
        assert "calls" not in result and result["failure_artifacts"]
    result, _ = native_counter_runner()
    for damage in (lambda value: value["calls"][0]["work"].update(memory_bytes=1),
                   lambda value: value["native_run"]["calls"][-1].update(input_content_sha256="e" * 64),
                   lambda value: value["native_run"]["counter_windows"][0].update(range_balanced=False),
                   lambda value: value["calls"][0]["launch_identities"][0].update(symbol="forged"),
                   lambda value: value["calls"][0]["kernel_identities"][0].update(binary_sha256="f" * 64),
                   lambda value: value["raw_report"].update(sha256="f" * 64),
                   lambda value: value["raw_report"].update(bytes=1),
                   lambda value: value["raw_report"].update(base64="!")):
        changed = copy.deepcopy(result)
        damage(changed)
        changed["receipt_sha256"] = counter_task._receipt(changed)
        with pytest.raises(ValueError):
            counter_task.validate(changed, allow_simulated=True)
    events = []
    observer = counter_task.Profiler({"_native_counter": {"nonce": "a" * 32}}, functions=(
        lambda: events.append("start"), lambda: events.append("stop"), lambda marker: 0,
        lambda: events.append("pop") or 0))
    with pytest.raises(RuntimeError, match="forward failed"):
        with observer.call(seed=42, signature=[], training=True):
            raise RuntimeError("forward failed")
    assert events == ["start", "stop", "pop"]
    assert observer.windows[0]["stopped"] and observer.windows[0]["range_balanced"]
    broken = counter_task.Profiler({"_native_counter": {"nonce": "a" * 32}}, functions=(
        lambda: events.append("unexpected start"), lambda: None, lambda marker: -1, lambda: -1))
    with pytest.raises(ValueError, match="valid range"):
        with broken.call(seed=42, signature=[], training=True):
            pytest.fail("Invalid NVTX range must not enter the candidate")
    unavailable, _ = native_counter_runner(failure="sdk")
    assert unavailable["status"] == "not_measured" and "calls" not in unavailable


def test_native_counter_models_replay_per_call_work_review_and_device_rates(native_counter_runner):
    result, _ = native_counter_runner()
    rows, payload = result["native_run"]["calls"], result["origin"]["native_request"]
    source = hashlib.sha256(payload["kernel_src"].encode()).hexdigest()
    result["recipe"] = policy(source=source)
    result["recipe"]["review"]["native_contract_sha256"] = counter_task.native_contract(payload)
    result["recipe"]["review"]["workload_contracts"] = [
        {"source_sha256": source, "reference_sha256": hashlib.sha256(payload["ref_arch_src"].encode()).hexdigest(),
         "input_signature_sha256": row["input_signature_sha256"], "precision": "fp32",
         "policy_sha256": fingerprint({"native_tolerance": 1e-4})} for row in rows]
    result["receipt_sha256"] = counter_task._receipt(result)
    modeled = counter_task.execution_models(result, device_calibration(), allow_simulated=True)
    assert len(modeled["calls"]) == len(rows) and modeled["simulated"]
    assert not modeled["hardware_counters_collected"]
    assert all(row["sol_execution_ms"] == pytest.approx(0.192) for row in modeled["calls"])
    calibrated = device_calibration()
    bound = {"protocol": "reviewed_headroom_certificate_v1", "layer": "I", "precision": "fp32",
             "review": {"reviewer": "unscored format fixture", "rationale": "CPU serialization only"},
             "reference_sha256": hashlib.sha256(payload["ref_arch_src"].encode()).hexdigest(),
             "input_signature": result["origin"]["raw"]["windows"][0]["input_signature"],
             "calibration_sha256": calibrated["receipt_sha256"],
             "reference_work": {"memory_bytes": 192}, "algorithm_work": {"memory_bytes": 96}}
    timeline = performance.native_timeline(result["origin"], counter_report=result, calibration=calibrated,
                                           bound=bound, allow_simulated=True)
    assert timeline["execution_model_replayed"] and not timeline["hardware_counters_collected"]
    for call in timeline["calls"]:
        assert call["A"] == pytest.approx(2)
        assert call["sol_algorithm_ms"] * call["A"] * call["E"] * call["O"] == pytest.approx(call["profile_window_wall_ms"])
        assert not call["model_consistent"]  # Intentionally impossible CPU-format timings are not physical evidence.
    mismatched = {**bound, "reference_sha256": "f" * 64}
    with pytest.raises(ValueError, match="mismatch"):
        performance.native_timeline(result["origin"], counter_report=result, calibration=calibrated,
                                    bound=mismatched, allow_simulated=True)
    rebound = copy.deepcopy(result)
    rebound["recipe"]["review"]["native_contract_sha256"] = "f" * 64
    rebound["receipt_sha256"] = counter_task._receipt(rebound)
    with pytest.raises(ValueError, match="native configuration"):
        counter_task.execution_models(rebound, calibrated, allow_simulated=True)
    partial = copy.deepcopy(result)
    partial["recipe"]["coverage"] = "reported_resources_only"
    partial["receipt_sha256"] = counter_task._receipt(partial)
    with pytest.raises(ValueError, match="complete reviewed"):
        counter_task.execution_models(partial, device_calibration(), allow_simulated=True)
    memory = copy.deepcopy(result)
    origin = memory["origin"]
    raw = origin["raw"]
    window = raw["windows"][0]
    raw["activities"].append({"kind": "memcpy", "start_ns": window["start_ns"] + 10,
        "end_ns": window["start_ns"] + 90, "device": 0, "stream": 3, "context_id": 1,
        "correlation": 800_000, "graph_id": 0, "graph_node_id": 0, "bytes": 64,
        "copy_kind": 8, "src_kind": 3, "dst_kind": 3})
    raw["memory_calls"] = [{"context_id": 1, "correlation": 800_000, "call": window["id"],
                            "kind": "memcpy", "exit_success": True}]
    origin["summary"] = origin_task.summarize(raw)
    origin["receipt_sha256"] = fingerprint({key: value for key, value in origin.items() if key not in {"receipt_sha256", "metadata"}})
    memory["calls"] = counter_task._observations(memory, allow_simulated=True)
    memory["receipt_sha256"] = counter_task._receipt(memory)
    counter_task.validate(memory, allow_simulated=True)
    observed = performance.native_timeline(origin, counter_report=memory, allow_simulated=True)
    assert observed["calls"][0]["memory_activity"]["payload_bytes"] == 64
    assert memory["calls"][0]["profile_memory_activity"]["logical_device_read_bytes"] == 64
    with pytest.raises(ValueError, match="transfer calibration"):
        counter_task.execution_models(memory, calibrated, allow_simulated=True)


def test_counter_cli_preview_and_queue_validation_are_sdk_light(tmp_path):
    config = RunConfig(level=1, start=2, end=2, generation="local", runs_dir=str(tmp_path)).validate()
    config.run_dir.mkdir(parents=True)
    (config.run_dir / kernel_name(1, 2)).write_text(SOURCE)
    request, _ = prepare(config, 2, 0, policy())
    from kb_workflows.runtime.queue_worker import validate_payload

    validate_payload(request)
    with pytest.raises(ValueError, match="explicit paired"):
        prepare(config, 2, 0, policy(), target="reference")
    calibration_request = engine.task_payload(config, 2, task="calibrate_precision")
    validate_payload(calibration_request)
    calibration_request["kernel_src"] = SOURCE
    with pytest.raises(ValueError, match="no candidate"):
        validate_payload(calibration_request)
    policy_path, config_path, output = tmp_path / "recipe.json", tmp_path / "config.toml", tmp_path / "result.json"
    atomic_json(policy_path, policy())
    config_path.write_text(f'[run]\nruns_dir={json.dumps(str(tmp_path))}\nrun_name="test_run"\nstart=2\nend=2\ngeneration="local"\n')
    (tmp_path / "test_run").mkdir()
    (tmp_path / "test_run" / kernel_name(1, 2)).write_text(SOURCE)
    code = "from scripts.audit_work_counters import main; import sys; status=main(sys.argv[1:]); assert not {'torch','triton','cupti','harbor'} & set(sys.modules); raise SystemExit(status)"
    completed = subprocess.run([sys.executable, "-c", code, "--config", str(config_path), "--recipe", str(policy_path),
                                "--problem", "2", "--output", str(output)], cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not output.exists()


def test_scored_counter_join_uses_actual_clock_and_rejects_changed_kernel_workloads(native_counter_runner, tmp_path, monkeypatch):
    from scripts import analyze_device_trace
    from tests.provenance.test_kernel_origin import BINARY, frontend_receipt, raw_capture

    report, _ = native_counter_runner(scored=True)
    payload = report["origin"]["native_request"]
    actual = payload["_scored_result"]
    source = payload["kernel_src"]
    def install(raw):
        pattern = raw_capture()
        for key in ("modules", "functions", "proofs"):
            raw[key] = copy.deepcopy(pattern[key])
        proof = raw["proofs"][0]
        proof.update(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                     jit_source_sha256=hashlib.sha256(source.encode()).hexdigest(), frontend=frontend_receipt(source))
        proof["emission"]["frontend_receipt_sha256"] = proof["frontend"]["receipt_sha256"]
        raw["launches"], raw["activities"] = [], []
        for window in raw["windows"]:
            for index, (left, right) in enumerate(((100, 700), (400, 900))):
                correlation = window["id"] * 10 + index
                raw["launches"].append({**pattern["launches"][0], "call": window["id"], "correlation": correlation})
                raw["activities"].append({**pattern["activities"][0], "start_ns": window["start_ns"] + left,
                                          "end_ns": window["start_ns"] + right, "stream": index + 1, "correlation": correlation})
    install(actual["metadata"]["origin_capture"]["raw"])
    install(report["origin"]["raw"])
    def seal_all(value):
        origin = value["origin"]
        result = origin["native_request"]["_scored_result"]
        observer = result["metadata"]["origin_capture"]
        observer["receipt_sha256"] = fingerprint({k: v for k, v in observer.items() if k != "receipt_sha256"})
        calls = result["metadata"]["native_call_receipt"]
        calls["origin_capture_sha256"] = observer["receipt_sha256"]
        calls["receipt_sha256"] = fingerprint({k: v for k, v in calls.items() if k != "receipt_sha256"})
        origin["summary"] = kernel_summary = origin_task.summarize(origin["raw"])
        assert kernel_summary["calls"]
        origin["receipt_sha256"] = fingerprint({k: v for k, v in origin.items() if k not in {"receipt_sha256", "metadata"}})
        value["receipt_sha256"] = counter_task._receipt(value)
    report["recipe"] = policy(source=hashlib.sha256(source.encode()).hexdigest())
    report["recipe"]["review"]["native_contract_sha256"] = counter_task.native_contract(payload)
    report["recipe"]["review"]["workload_contracts"] = [
        {"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
         "reference_sha256": hashlib.sha256(payload["ref_arch_src"].encode()).hexdigest(),
         "input_signature_sha256": row["input_signature_sha256"], "precision": "fp32",
         "policy_sha256": fingerprint({"native_tolerance": 1e-4})} for row in report["native_run"]["calls"]]
    seal_all(report)
    calibrated = calibration.seal(CAPS, {
        "memory_bytes": calibration.probe_record(10**9, [1] * 5, unit="bytes/s", parameters={}),
        "fp32_flops": calibration.probe_record(10**9, [1] * 5, unit="operations/s", parameters={})},
        [0.01] * 5, settings={"basis": "unscored parser fixture"})
    bound = {"protocol": "reviewed_headroom_certificate_v1", "layer": "I", "precision": "fp32",
             "review": {"reviewer": "software fixture", "rationale": "No device evidence"},
             "reference_sha256": hashlib.sha256(payload["ref_arch_src"].encode()).hexdigest(),
             "input_signature": report["origin"]["raw"]["windows"][0]["input_signature"],
             "calibration_sha256": calibrated["receipt_sha256"],
             "reference_work": {"memory_bytes": 192}, "algorithm_work": {"memory_bytes": 96}}
    joined = performance.scored_timeline(report, actual, calibrated, bound, allow_simulated=True)
    assert joined["status"] == "simulated" and not joined["formal_qualification_implemented"]
    assert joined["counter_process_launch_identity_observed"]
    assert joined["counter_process_workload_identity_observed"] and joined["counter_process_binary_identity_observed"]
    # Preserve unknowns for unsupported/compressed archives, or absent binary evidence.
    ambiguous = _ncu_report(report["native_run"]["pid"], len(report["calls"]) * 2, inline=True).replace(BINARY, BINARY[:-1] + b"X", 1)
    for raw in (None, b"Unsupported compressed Nsight archive", ambiguous,
                _ncu_report(report["native_run"]["pid"], len(report["calls"]) * 2 - 1)):
        changed = copy.deepcopy(report)
        changed["raw_report"] = None if raw is None else {"base64": base64.b64encode(raw).decode(),
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        changed["calls"] = counter_task._observations(changed, allow_simulated=True)
        seal_all(changed)
        unknown = performance.scored_timeline(changed, actual, calibrated, bound, allow_simulated=True)
        assert not unknown["counter_process_workload_identity_observed"] and not unknown["formal_qualification_implemented"]
        assert all(call["binary_identity_issue"] for call in changed["calls"])
    # Even a resealed archive with internally consistent counts cannot replace the actual cubin.
    changed = copy.deepcopy(report)
    raw = _ncu_report(changed["native_run"]["pid"], len(changed["calls"]) * 2, binary=b"\x7fELFother-cubin")
    changed["raw_report"] = {"base64": base64.b64encode(raw).decode(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    changed["calls"] = counter_task._observations(changed, allow_simulated=True)
    seal_all(changed)
    with pytest.raises(ValueError, match="Counter-process cubin"):
        performance.scored_timeline(changed, actual, calibrated, bound, allow_simulated=True)
    assert all(row["model_consistent"] and row["workload"]["kernel_count"] == 2 for row in joined["calls"])
    for row in joined["calls"]:
        assert row["scored_device_busy_ms"] == pytest.approx(0.0008)
        if row["wall_decomposition_available"]:
            assert row["sol_algorithm_ms"] * row["A"] * row["E"] * row["O"] == pytest.approx(row["scored_wall_ms"])
        else:
            assert row["O"] is None and row["scored_wall_ms"] is None
    with pytest.raises(ValueError):
        performance.scored_timeline(report, actual, calibrated, bound)
    for damage in ("binary", "symbol", "grid", "block", "shared", "missing_identity", "memory", "missing_call"):
        changed = copy.deepcopy(report)
        raw = changed["origin"]["raw"]
        if damage == "binary":
            raw["modules"][0]["binary_sha256"] = "f" * 64
            raw["proofs"][0]["binary_sha256"] = "f" * 64
            raw["proofs"][0]["emission"]["binary_sha256"] = "f" * 64
        elif damage == "symbol":
            raw["functions"][0]["symbol"] = "other_entrypoint"
            for activity in raw["activities"]:
                activity["name"] = "other_entrypoint"
        elif damage in {"grid", "block"}:
            raw["activities"][0]["configuration"][damage][0] += 1
        elif damage == "shared":
            raw["activities"][0]["configuration"]["dynamic_shared_bytes"] += 1
        elif damage == "missing_identity":
            raw["activities"][0].pop("configuration")
        elif damage == "memory":
            raw["activities"][0]["kind"] = "memcpy"
        else:
            changed["native_run"]["calls"].pop()
        seal_all(changed)
        with pytest.raises(ValueError):
            performance.scored_timeline(changed, changed["origin"]["native_request"]["_scored_result"], calibrated, bound,
                                        allow_simulated=True)
    for metric, replacement in (("launch__kernel_name", "other_kernel"), ("launch__grid_dim_x", "2"),
                                ("launch__block_dim_x", "64"), ("launch__shared_mem_per_block_dynamic", "1"),
                                ("launch__shared_mem_per_block_static", "1")):
        changed = copy.deepcopy(report)
        rows = list(csv.reader(io.StringIO(changed["raw_csv"])))
        for row in rows[1:]:
            if row[0] == "0" and row[6] == metric:
                row[8] = replacement
        output = io.StringIO()
        csv.writer(output).writerows(rows)
        changed["raw_csv"] = output.getvalue()
        # Rebuild every declaration: the original ncu values must still fail
        # association with the unchanged actual-scored source/configuration.
        changed["calls"] = counter_task._observations(changed, allow_simulated=True)
        seal_all(changed)
        with pytest.raises(ValueError, match="Counter-process original"):
            performance.scored_timeline(changed, actual, calibrated, bound, allow_simulated=True)
    # Real CLI routing; the serialization fixture remains explicitly simulated.
    report_path, scored_path, cal_path, bound_path = [tmp_path / name for name in ("counter.json", "scored.json", "cal.json", "bound.json")]
    for path, value in ((report_path, report), (scored_path, actual), (cal_path, calibrated), (bound_path, bound)):
        atomic_json(path, value)
    invoke = performance.scored_timeline
    monkeypatch.setattr(analyze_device_trace, "scored_timeline", lambda *a: invoke(*a, allow_simulated=True))
    output = tmp_path / "joined.json"
    assert analyze_device_trace.native_main([str(report_path), "--scored-result", str(scored_path), "--problem", "2",
        "--calibration", str(cal_path), "--certificate", str(bound_path), "--output", str(output)]) == 2
    assert json.loads(output.read_text())["simulated"]
