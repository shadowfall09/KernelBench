"""Actual local files/Parquet and an injected tool runner; no Explorer/device run."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_neuron_work import (
    artifacts,  # noqa: F401 -- shared local schema fixture
    review,
)

from kb_workflows import neuron_processing as np
from kb_workflows import neuron_tables, neuron_work, performance
from kb_workflows.common import ROOT
from scripts import process_neuron_work
from scripts.analyze_neuron_work import main as analyze


@pytest.fixture
def native_files(request, tmp_path):
    parquet, _, capture, _, rates = request.getfixturevalue("artifacts")
    source = tmp_path / "native"
    source.mkdir()
    for name in capture["artifact_files"]:
        (source / name).write_bytes(b"Synthetic native-format fixture, never executed: " + name.encode())
    capture["artifact_files"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    capture["artifact_file_bytes"] = {p.name: p.stat().st_size for p in source.iterdir()}
    capture["neuron_core_allocation"] = [12]
    neuron_work.seal(capture)
    binary = tmp_path / "explorer-format-stub"
    binary.write_text("Synthetic non-program bytes; the injected runner never executes this file")
    binary.chmod(0o700)
    return parquet, capture, rates, source, binary


def runner(parquet, *, damage=None, source=None, binary=None):
    calls = []

    def run(argv, timeout, log, *, environment):
        assert timeout > 0 and isinstance(environment, dict)
        calls.append(argv)
        log = Path(log)
        result = {"exit_code": 0, "timed_out": False, "request_wall_seconds": 0.001}
        if argv[1:] == ["--version"]:
            log.write_text("SIMULATED Neuron Explorer version schema fixture; not an installed SDK")
            if damage == "version_fail":
                result["exit_code"] = 2
            elif damage == "version_timeout":
                result.update(exit_code=124, timed_out=True)
            elif damage == "empty_version":
                log.write_text("")
            elif damage == "version_binary_change":
                binary.write_bytes(b"changed during version query")
            return result
        assert argv[1:3] == ["view", "--session-dir"] and argv[4:7] == ["--output-format", "parquet", "--output-file"]
        frozen, target = Path(argv[3]), Path(argv[7])
        assert frozen != source and all(not (p.stat().st_mode & 0o222) for p in frozen.rglob("*") if p.is_file())
        shutil.copytree(parquet, target)
        log.write_text("Injected offline view copied real Parquet fixtures, without invoking a tool")
        if damage == "view_fail":
            result["exit_code"] = 2
            log.write_bytes(b"F" * (70 * 1024))
        elif damage == "view_timeout":
            result.update(exit_code=124, timed_out=True)
        elif damage == "source_change":
            (source / "a.neff").write_bytes(b"changed source")
        elif damage == "snapshot_change":
            (frozen / "a.neff").chmod(0o644)
            (frozen / "a.neff").write_bytes(b"changed snapshot")
        elif damage == "binary_change":
            binary.write_bytes(b"changed during view")
        elif damage == "output_symlink":
            (target / "Instruction.parquet").unlink()
            (target / "Instruction.parquet").symlink_to(parquet / "Instruction.parquet")
        elif damage == "output_root_symlink":
            shutil.rmtree(target)
            target.symlink_to(parquet, target_is_directory=True)
        elif damage == "snapshot_root_symlink":
            shutil.rmtree(frozen)
            frozen.symlink_to(source, target_is_directory=True)
        elif damage == "schema":
            values = pq.read_table(target / "Metadata.parquet").to_pylist()
            values[0]["profile_schema_version"] = "unsupported"
            pq.write_table(pa.Table.from_pylist(values), target / "Metadata.parquet")
        return result

    return run, calls


def execute(native_files, tmp_path, *, damage=None, **kwargs):
    parquet, capture, _, source, binary = native_files
    run, calls = runner(parquet, damage=damage, source=source, binary=binary)
    value = np.process_capture(capture, source, directory=tmp_path / "processed", binary=str(binary), run=run, **kwargs)
    return value, calls


def test_complete_native_snapshot_and_every_output_file_bind_real_parquet_to_the_capture(native_files, tmp_path):
    result, calls = execute(native_files, tmp_path)
    _, capture, _, source, _ = native_files
    assert result["status"] == "ok" and len(calls) == 2 and calls[0][1:] == ["--version"]
    assert result["tool_invoked"]
    report, record = result["report"], result["processing"]
    assert record["simulated"] and not record["device_workload_started"] and not record["formal_origin_qualified"]
    assert record["copy_policy"] == np.COPY_POLICY and record["core_binding_scope"] == np.CORE_SCOPE
    assert "Summary.parquet" in record["output_files"] and "Summary.parquet" not in record["report_files"]
    assert {name: row["sha256"] for name, row in record["source_files"].items()} == capture["artifact_files"]
    assert {name: row["bytes"] for name, row in record["source_files"].items()} == capture["artifact_file_bytes"]
    assert np.validate(json.loads(json.dumps(record)), report, capture, allow_simulated=True) == record
    with pytest.raises(ValueError, match="simulated"):
        np.validate(record, report, capture)
    work = Path(result["work_directory"])
    assert (work / "parquet/Instruction.parquet").is_file() and (work / "explorer.log").is_file()
    for name in capture["artifact_files"]:
        assert (work / "input" / name).read_bytes() == (source / name).read_bytes()


@pytest.mark.parametrize(
    "damage", ["missing", "extra", "bytes", "content", "symlink", "special", "allocation", "no_sizes"]
)
def test_invalid_native_inventory_is_refused_before_any_tool_call(native_files, tmp_path, damage):
    _, capture, _, source, _ = native_files
    if damage == "missing":
        (source / "a.neff").unlink()
    elif damage == "extra":
        (source / "unrecorded.neff").write_bytes(b"unrecorded")
    elif damage == "bytes":
        capture["artifact_file_bytes"]["a.neff"] += 1
        neuron_work.seal(capture)
    elif damage == "content":
        (source / "a.neff").write_bytes(b"changed")
    elif damage == "symlink":
        (source / "extra").symlink_to(tmp_path)
    elif damage == "special":
        import os

        os.mkfifo(source / "fifo")
    elif damage == "allocation":
        capture["neuron_core_allocation"] = [12, 13]
        neuron_work.seal(capture)
    elif damage == "no_sizes":
        capture["artifact_file_bytes"] = None
        neuron_work.seal(capture)
    value, calls = execute(native_files, tmp_path)
    assert value["status"] == "not_measured" and value["reason"] and not calls
    assert not value["tool_invoked"]


@pytest.mark.parametrize(
    "damage",
    [
        "version_fail",
        "version_timeout",
        "empty_version",
        "version_binary_change",
        "view_fail",
        "view_timeout",
        "source_change",
        "snapshot_change",
        "binary_change",
        "output_symlink",
        "output_root_symlink",
        "snapshot_root_symlink",
        "schema",
    ],
)
def test_partial_changed_or_failed_processing_never_mints_a_receipt(native_files, tmp_path, damage):
    value, calls = execute(native_files, tmp_path, damage=damage)
    assert value["status"] == "not_measured" and "processing" not in value and value["reason"]
    assert not value["device_workload_started"] and calls
    assert "explorer-version.log" in value["failure_artifacts"]
    if damage == "view_fail":
        log = value["failure_artifacts"]["explorer.log"]
        assert log["bytes"] == 70 * 1024 and not log["complete"]
        assert (Path(value["work_directory"]) / "parquet/Instruction.parquet").is_file()


@pytest.mark.parametrize("damage", ["source", "snapshot", "output", "binary"])
def test_inputs_tool_and_outputs_are_rechecked_after_table_decoding(native_files, tmp_path, monkeypatch, damage):
    original = neuron_tables.load

    def load(directory, **kwargs):
        report = original(directory, **kwargs)
        _, _, _, source, binary = native_files
        if damage == "source":
            (source / "a.neff").write_bytes(b"late source mutation")
        elif damage == "snapshot":
            path = Path(directory).parent / "input/a.neff"
            path.chmod(0o644)
            path.write_bytes(b"late snapshot mutation")
        elif damage == "output":
            (Path(directory) / "late.txt").write_text("late output")
        else:
            binary.write_bytes(b"late executable mutation")
        return report

    monkeypatch.setattr(neuron_tables, "load", load)
    value, _ = execute(native_files, tmp_path)
    assert value["status"] == "not_measured" and "processing" not in value


@pytest.mark.parametrize("limit", [{"max_files": 1}, {"max_file_bytes": 1}, {"max_bytes": 1, "max_file_bytes": 1}])
def test_complete_inventory_limits_are_admitted_before_tool_discovery(native_files, tmp_path, limit):
    value, calls = execute(native_files, tmp_path, limits={**np.DEFAULT_LIMITS, **limit})
    assert value["status"] == "not_measured" and not calls


@pytest.mark.parametrize(
    "damage",
    [
        "source_hash",
        "source_bytes",
        "missing_output",
        "output_hash",
        "table_rows",
        "capture",
        "report",
        "command",
        "options",
        "tool_hash",
        "version",
        "exit",
        "timed_out",
        "snapshot",
        "formal",
        "simulated",
    ],
)
def test_resealed_processing_cannot_borrow_another_capture_table_or_tool_contract(native_files, tmp_path, damage):
    result, _ = execute(native_files, tmp_path)
    record = copy.deepcopy(result["processing"])
    _, capture, _, _, _ = native_files
    record["simulated"] = False
    if damage == "source_hash":
        record["source_files"]["a.neff"]["sha256"] = "0" * 64
    elif damage == "source_bytes":
        record["source_files"]["a.neff"]["bytes"] += 1
    elif damage == "missing_output":
        del record["output_files"]["Instruction.parquet"]
    elif damage == "output_hash":
        record["output_files"]["Instruction.parquet"]["sha256"] = "0" * 64
    elif damage == "table_rows":
        record["report_files"]["Instruction.parquet"]["rows"] += 1
    elif damage == "capture":
        record["capture_receipt_sha256"] = "0" * 64
    elif damage == "report":
        record["report_receipt_sha256"] = "0" * 64
    elif damage == "command":
        record["command"][1] = "capture"
    elif damage == "options":
        record["command"] += ["--ignore-device-profile"]
    elif damage == "tool_hash":
        record["tool"]["sha256"] = "not a hash"
    elif damage == "version":
        record["tool"]["version"] = "borrowed version"
    elif damage == "exit":
        record["process"]["exit_code"] = True
    elif damage == "timed_out":
        record["process"]["timed_out"] = True
    elif damage == "snapshot":
        record["snapshot_after_processing_unchanged"] = False
    elif damage == "formal":
        record["formal_origin_qualified"] = True
    elif damage == "simulated":
        record["simulated"] = True
    neuron_work.seal(record)
    with pytest.raises(ValueError):
        np.validate(record, result["report"], capture)


def test_valid_serialized_processing_replays_into_model_and_offline_cli_without_promoting_origin(
    native_files, tmp_path, monkeypatch
):
    result, _ = execute(native_files, tmp_path)
    _, capture, rates, _, _ = native_files
    report, record = result["report"], result["processing"]
    # Unscored format fixture only, never real processing/device evidence.
    record["simulated"] = False
    neuron_work.seal(record)
    policy = review(report, capture)
    policy["pairing"] = {"processing_receipt_sha256": record["receipt_sha256"]}
    model = neuron_work.execution_model(report, capture, policy, rates, processing=record)
    assert model["automatic_capture_report_binding"] and not model["formal_origin_qualified"]
    assert "verified_Explorer_view" in model["association"]
    assert not performance.decomposition(1, 0.8, 0.001, counters=model)["hardware_counters_collected"]
    assert neuron_work.validate_model(model) == model
    argv = [str(Path(result["work_directory"]) / "parquet"), "--output", str(tmp_path / "model.json")]
    for name, value in (
        ("capture", capture),
        ("processing", {"processing": record}),
        ("review", policy),
        ("calibration", rates),
    ):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        argv += ["--" + name, str(path)]
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("offline model launched a tool"))
    assert analyze(argv) == 0
    saved = json.loads((tmp_path / "model.json").read_text())
    assert neuron_work.validate_model(saved)["automatic_capture_report_binding"]
    saved["processing"]["snapshot_after_processing_unchanged"] = False
    neuron_work.seal(saved["processing"])
    neuron_work.seal(saved)
    factors = performance.decomposition(1, 0.8, 0.001, counters=saved)
    assert factors["A"] is None and factors["counter_issue"]


def test_preview_is_sdk_light_and_does_not_resolve_a_tool_or_create_paths(native_files, tmp_path, monkeypatch, capsys):
    _, capture, _, source, _ = native_files
    path = tmp_path / "capture.json"
    path.write_text(json.dumps({"capture_receipt": capture}))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("preview launched a process"))
    monkeypatch.setattr(shutil, "which", lambda *a, **k: pytest.fail("preview resolved a tool"))
    output = tmp_path / "preview/report.json"
    assert process_neuron_work.main(["--capture", str(path), "--artifacts", str(source), "--output", str(output)]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert not preview["tool_invoked"] and not preview["device_workload_started"]
    assert not output.parent.exists()
    for timeout in ("nan", "0", "-1"):
        assert (
            process_neuron_work.main(
                ["--capture", str(path), "--artifacts", str(source), "--output", str(output), "--timeout", timeout]
            )
            == 2
        )


def test_explicit_processing_cli_saves_the_result_and_refuses_existing_paths(native_files, tmp_path, monkeypatch):
    parquet, capture, _, source, binary = native_files
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(capture))
    output = tmp_path / "result.json"
    run, _ = runner(parquet, source=source, binary=binary)
    original = np.process_capture
    monkeypatch.setattr(process_neuron_work, "process_capture", lambda *a, **k: original(*a, **k, run=run))
    args = [
        "--capture",
        str(path),
        "--artifacts",
        str(source),
        "--output",
        str(output),
        "--binary",
        str(binary),
        "--execute",
    ]
    assert process_neuron_work.main(args) == 0
    assert json.loads(output.read_text())["processing"]["simulated"]
    assert process_neuron_work.main(args) == 2


def test_processing_modules_import_without_arrow_device_or_model_sdks():
    code = 'import sys; import kb_workflows.neuron_processing, scripts.process_neuron_work; assert not {"torch","pyarrow","torch_neuronx","triton","transformers"} & sys.modules.keys()'
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_processing_work_directory_cannot_overlap_capture_or_overwrite_previous_artifacts(native_files, tmp_path):
    _, capture, _, source, binary = native_files
    requested = source / "child"
    value = np.process_capture(
        capture,
        source,
        directory=requested,
        binary=str(binary),
        run=lambda *a, **k: pytest.fail("overlap launched tool"),
    )
    assert value["status"] == "not_measured" and not value["tool_invoked"] and not requested.exists()
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "preserve.txt").write_text("prior artifacts")
    with pytest.raises(ValueError, match="new work directory"):
        np.process_capture(
            capture,
            source,
            directory=occupied,
            binary=str(binary),
            run=lambda *a, **k: pytest.fail("existing work launched tool"),
        )
    assert (occupied / "preserve.txt").read_text() == "prior artifacts"
    outside = tmp_path / "unexpected-target"
    link = tmp_path / "work-link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="new work directory"):
        np.process_capture(
            capture,
            source,
            directory=link,
            binary=str(binary),
            run=lambda *a, **k: pytest.fail("work symlink launched tool"),
        )
    assert not outside.exists()


def test_selected_table_envelope_refuses_a_partial_report_and_keeps_raw_parquet(native_files, tmp_path):
    result, calls = execute(native_files, tmp_path, table_limits={"max_rows": 1})
    assert len(calls) == 2 and result["status"] == "not_measured" and "processing" not in result
    assert (Path(result["work_directory"]) / "parquet/Instruction.parquet").is_file()


@pytest.mark.parametrize("supplied", ["missing", "null", "empty_result"])
def test_requested_processing_evidence_never_falls_back_to_manual_pairing(native_files, tmp_path, supplied):
    result, _ = execute(native_files, tmp_path)
    _, capture, rates, _, _ = native_files
    policy = review(result["report"], capture)
    argv = [str(Path(result["work_directory"]) / "parquet"), "--output", str(tmp_path / "failure.json")]
    for name, value in (("capture", capture), ("review", policy), ("calibration", rates)):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        argv += ["--" + name, str(path)]
    evidence = tmp_path / "processing.json"
    if supplied != "missing":
        evidence.write_text(json.dumps(None if supplied == "null" else {"processing": None}))
    assert analyze([*argv, "--processing", str(evidence)]) == 2
    saved = json.loads((tmp_path / "failure.json").read_text())
    assert (
        saved["execution_model"]["status"] == "not_measured"
        and "no manual fallback" in saved["execution_model"]["reason"]
    )
