"""Replay vendor counter values and reviewed unit conversions without SDK imports."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import struct
from decimal import Decimal, InvalidOperation

from ..analysis.calibration import identity, positive
from ..common import evaluator_digest, fingerprint

PROTOCOL = "kernelbench_pro_work_counters_v1"
RECIPE = "kernelbench_pro_counter_recipe_v1"
MODEL = "kernelbench_pro_counter_execution_model_v1"
MAX_BYTES = 64 * 1024**2
MAX_ROWS = 200_000
FIELDS = ("ID", "Process ID", "Context", "Stream", "Device", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value")
BINDINGS = ("source_sha256", "reference_sha256", "evaluator_sha256", "evaluation_contract_sha256",
            "input_signature_sha256", "input_probe_sha256", "state_probe_sha256", "policy_sha256")
WORKLOAD_FIELDS = ("source_sha256", "reference_sha256", "input_signature_sha256", "policy_sha256", "precision")
LAUNCH_METRICS = {"launch__kernel_name": {""},
                  **{f"launch__grid_dim_{axis}": {"", "block"} for axis in "xyz"},
                  **{f"launch__block_dim_{axis}": {"", "thread"} for axis in "xyz"},
                  **{f"launch__shared_mem_per_block_{kind}": {"byte"} for kind in ("dynamic", "static")}}


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def recipe(value):
    if (not isinstance(value, dict) or value.get("protocol") != RECIPE
            or not isinstance(value.get("review"), dict)
            or any(not isinstance(value["review"].get(k), str) or not value["review"][k].strip() for k in ("reviewer", "rationale"))
            or value.get("coverage") not in {"reported_resources_only", "reviewed_complete_workload"}
            or not isinstance(value.get("resources"), dict) or not 1 <= len(value["resources"]) <= 32):
        raise ValueError("Counter recipe needs explicit review, resources and coverage")
    units = {}
    for resource, rule in value["resources"].items():
        if (not isinstance(resource, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", resource)
                or not isinstance(rule, dict) or rule.get("rate_unit") not in {"bytes/s", "operations/s"}
                or not isinstance(rule.get("terms"), list) or not 1 <= len(rule["terms"]) <= 32):
            raise ValueError("Counter resource has no reviewed unit conversion")
        seen = set()
        for term in rule["terms"]:
            if (not isinstance(term, dict) or not isinstance(term.get("metric"), str)
                    or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*\.sum", term["metric"])
                    or term["metric"] in seen or term.get("unit") not in {"byte", "inst", "op", "sector"}
                    or not positive(term.get("coefficient")) or term["coefficient"] > 2**32):
                raise ValueError("Counter terms require aggregate counts, explicit units and positive coefficients")
            name = term["metric"]
            if name in units and units[name] != term["unit"]:
                raise ValueError("Counter recipe assigns conflicting units")
            units[name] = term["unit"]
            seen.add(name)
    if len(units) > 128:
        raise ValueError("Counter recipe exceeds the metric envelope")
    if value["coverage"] == "reviewed_complete_workload":
        review = value["review"]
        if (not isinstance(review.get("source_sha256"), list) or not review["source_sha256"]
                or any(not sha(v) for v in review["source_sha256"])
                or len(review["source_sha256"]) != len(set(review["source_sha256"])) or not sha(review.get("numeric_class_sha256"))):
            raise ValueError("Complete counter coverage needs reviewed sources and hardware/software class")
        workloads = review.get("workload_contracts")
        if (not isinstance(workloads, list) or not 1 <= len(workloads) <= 10_000
                or any(not isinstance(row, dict) or set(row) != set(WORKLOAD_FIELDS)
                    or any(not sha(row.get(k)) for k in WORKLOAD_FIELDS if k != "precision")
                    or row["source_sha256"] not in review["source_sha256"] or row.get("precision") not in {"fp32", "fp16", "bf16"}
                    for row in workloads)):
            raise ValueError("Complete counter review must bind reference, input geometry, precision and numerical policy")
    return units


def count(value):
    # CSV values use base units; never turn failed conversion/N/A into zero.
    if not isinstance(value, str) or not re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?(?:[eE][+-]?\d+)?", value):
        raise ValueError("Counter value is missing, negative, scaled or not numeric")
    try:
        result = Decimal(value.replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError("Counter exponent exceeds its exact numeric envelope") from exc
    if not result.is_finite() or result != result.to_integral_value() or not 0 <= result <= 2**64 - 1:
        raise ValueError("Counter counts must be exact nonnegative uint64 values")
    return int(result)


def work_totals(actions, policy):
    """Shared aggregate-count arithmetic for standalone and native reports."""
    return {resource: sum(sum(action["metrics"][term["metric"]]["count"] * term["coefficient"]
                             for term in rule["terms"]) for action in actions)
            for resource, rule in policy["resources"].items()}


def parse_csv(raw, policy, *, pid, device=0, launch_identity=False):
    units = recipe(policy)
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_BYTES or type(pid) is not int or pid <= 0:
        raise ValueError("Counter CSV requires a bounded report and observed worker PID")
    rows = csv.reader(io.StringIO(raw.decode("utf-8-sig", errors="strict")))
    header, actions, messages = None, {}, []
    for ordinal, row in enumerate(rows):
        if ordinal >= MAX_ROWS:
            raise ValueError("Counter report exceeds its raw-row envelope")
        if not row:
            continue
        if row[0].startswith("=="):
            messages.append(",".join(row)[:1000])
            continue
        if row[0] == "ID":
            if len(set(row)) != len(row) or not set(FIELDS) <= set(row) or (header is not None and header != row):
                raise ValueError("Counter CSV has a missing, ambiguous or changed schema")
            header = row
            continue
        if header is None or len(row) != len(header):
            raise ValueError("Counter report contains an unparsed or truncated row")
        item = dict(zip(header, row, strict=True))
        fields = ("ID", "Process ID", "Context", "Stream", "Device")
        if any(not re.fullmatch(r"0|[1-9][0-9]*", item[k]) for k in fields):
            raise ValueError("Counter report lacks exact process/context/stream/device identities")
        if int(item["Process ID"]) != pid or int(item["Device"]) != device:
            raise ValueError("Counter report belongs to another process or device")
        key = item["ID"]
        metadata = {k: item[k] for k in fields + ("Kernel Name",)}
        action = actions.setdefault(key, {"identity": metadata, "metrics": {}})
        if action["identity"] != metadata:
            raise ValueError("Counter launch ID was reused with different metadata")
        name = item["Metric Name"]
        if launch_identity and name in LAUNCH_METRICS:
            attributes = action.setdefault("launch_attributes", {})
            if name in attributes or item["Metric Unit"] not in LAUNCH_METRICS[name]:
                raise ValueError("Counter launch attribute has duplicate values or an unsupported unit")
            value = item["Metric Value"]
            if name == "launch__kernel_name":
                if not value or len(value) > 2048 or "\x00" in value:
                    raise ValueError("Counter launch lacks an original kernel name")
            else:
                value = count(value)
                if "_dim_" in name and value == 0:
                    raise ValueError("Counter launch dimensions must be positive")
            attributes[name] = value
            continue
        if name not in units:
            continue  # NCU also emits launch/device attributes; raw bytes remain retained.
        if item["Metric Unit"] != units[name] or name in action["metrics"]:
            raise ValueError("Counter metric has duplicate values or the wrong unit")
        action["metrics"][name] = {"unit": units[name], "count": count(item["Metric Value"])}
    if not actions or any(set(action["metrics"]) != set(units) for action in actions.values()):
        raise ValueError("Counter report omitted a required metric or has no measured launches")
    if launch_identity:
        for action in actions.values():
            attributes = action.pop("launch_attributes", {})
            if set(attributes) != set(LAUNCH_METRICS):
                raise ValueError("Counter report omitted its same-process launch identity")
            action["launch_identity"] = {"symbol": attributes["launch__kernel_name"], "configuration": {
                **{kind: [attributes[f"launch__{kind}_dim_{axis}"] for axis in "xyz"] for kind in ("grid", "block")},
                **{f"{kind}_shared_bytes": attributes[f"launch__shared_mem_per_block_{kind}"] for kind in ("dynamic", "static")}}}
    work = work_totals(actions.values(), policy)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in work.values()):
        raise ValueError("Counter work overflows its finite quantity envelope")
    return {"actions": list(actions.values()), "work": work, "messages": messages,
            "launch_count": len(actions), "report_sha256": hashlib.sha256(raw).hexdigest()}


def _proto(data):
    """Read bounded protobuf wire fields; no generated/vendor runtime is needed."""
    if not isinstance(data, (bytes, memoryview)):
        raise ValueError("Missing Nsight protobuf message")
    data, position, result, fields = memoryview(data), 0, {}, 0

    def varint():
        nonlocal position
        value = 0
        for shift in range(0, 70, 7):
            if position >= len(data):
                break
            byte = data[position]
            position += 1
            value |= (byte & 127) << shift
            if byte < 128 and value < 2**64:
                return value
        raise ValueError("Truncated or excessive Nsight protobuf integer")

    while position < len(data):
        tag = varint()
        number, wire = tag >> 3, tag & 7
        fields += 1
        if not 0 < number < 2**29 or fields > MAX_ROWS:
            raise ValueError("Invalid or excessive Nsight protobuf fields")
        if wire == 0:
            value = varint()
        elif wire in (1, 2, 5):
            size = varint() if wire == 2 else 8 if wire == 1 else 4
            if position + size > len(data):
                raise ValueError("Truncated Nsight protobuf field")
            value = data[position:position + size]
            position += size
        else:
            raise ValueError("Unsupported Nsight protobuf wire type")
        result.setdefault(number, []).append((wire, value))
    return result


def _field(message, number, wire, default=None):
    values = message.get(number)
    if values is None:
        return default
    if len(values) != 1 or values[0][0] != wire:
        raise ValueError("Ambiguous or mistyped Nsight protobuf field")
    return values[0][1]


def report_binaries(raw, *, pid):
    """Extract original cubin bytes from an uncompressed version-7 ncu report.

    Field numbers follow the installed NVIDIA ProfilerReport{,Common}.proto.
    Unsupported formats never become binary observations through a CSV label.
    """
    if (not isinstance(raw, bytes) or not 8 <= len(raw) <= MAX_BYTES or raw[:4] != b"NVR\0"
            or type(pid) is not int or pid <= 0):
        raise ValueError("Nsight binary identity needs a bounded uncompressed report")
    data, position, sources, results, default_pid = memoryview(raw), 4, {}, [], None

    def message(end):
        nonlocal position
        if position + 4 > end:
            raise ValueError("Truncated Nsight report message length")
        size = struct.unpack_from("<I", data, position)[0]
        position += 4
        if position + size > end:
            raise ValueError("Truncated Nsight report message")
        value = _proto(data[position:position + size])
        position += size
        return value

    if _field(message(len(data)), 1, 0) != 7:
        raise ValueError("Unsupported Nsight report version")

    def source(value, process):
        reference, binary = _field(value, 1, 0), _field(value, 4, 2)
        if binary is not None:
            if len(binary) < 4 or bytes(binary[:4]) != b"\x7fELF":
                raise ValueError("Nsight source lacks original ELF cubin bytes")
            identity = {"binary_sha256": hashlib.sha256(binary).hexdigest(), "binary_bytes": len(binary)}
            if reference is not None:
                key = process, reference
                if key in sources and sources[key] != identity:
                    raise ValueError("Nsight source reference has conflicting cubins")
                sources[key] = identity
            return identity
        if reference is None:
            raise ValueError("Nsight launch lacks a cubin or source reference")
        return process, reference

    blocks, entries = 0, 0
    while position < len(data):
        header = message(len(data))
        blocks += 1
        counts = [_field(header, field, 0, 0) for field in (1, 2, 7)]
        size = _field(header, 5, 0)
        entries += sum(counts)
        if blocks > MAX_ROWS or entries > MAX_ROWS or counts[2] or size is None or position + size > len(data):
            raise ValueError("Unsupported or excessive Nsight report block")
        end = position + size
        session = _field(header, 3, 2)
        if session is not None:
            default_pid = _field(_proto(session), 1, 0, default_pid)
        process_info = _field(header, 6, 2)
        process = _field(_proto(process_info), 1, 0) if process_info is not None else default_pid
        if counts[0] + counts[1] and process != pid:
            raise ValueError("Nsight cubins/results belong to another or unknown process")
        for _ in range(counts[0]):
            value = source(message(end), process)
            if not isinstance(value, dict):
                raise ValueError("Nsight source definition has no binary")
        for _ in range(counts[1]):
            row = message(end)
            symbol = _field(row, 5, 2)
            symbol = bytes(symbol).decode("utf-8") if symbol is not None else ""
            dimensions = {}
            for name, field in (("grid", 10), ("block", 11)):
                vector = _proto(_field(row, field, 2))
                dimensions[name] = [_field(vector, axis, 0) for axis in (1, 2, 3)]
            context, stream = _field(row, 22, 0), _field(row, 23, 0)
            if (not symbol or len(symbol) > 2048 or "\x00" in symbol or context is None or stream is None
                    or any(value is None or value == 0 for dims in dimensions.values() for value in dims)):
                raise ValueError("Nsight launch lacks exact symbol/context/stream/dimensions")
            results.append({"context": context, "stream": stream, "symbol": symbol, **dimensions,
                            "binary": source(_proto(_field(row, 15, 2)), process)})
        if position != end:
            raise ValueError("Nsight report block has unparsed payload")
    if not results:
        raise ValueError("Nsight report contains no kernel results")
    for row in results:
        binary = row.pop("binary")
        if not isinstance(binary, dict):
            binary = sources.get(binary)
        if binary is None:
            raise ValueError("Nsight launch has an unresolved cubin reference")
        row.update(binary)
    return results


def bind_report_binaries(raw, actions, *, pid):
    """Join exact report/CSV launch groups; differing binaries are ambiguous."""
    groups, selected = {}, {}
    def key(context, stream, symbol, configuration):
        return context, stream, symbol, tuple(configuration["grid"]), tuple(configuration["block"])
    for row in report_binaries(raw, pid=pid):
        group = groups.setdefault(key(row["context"], row["stream"], row["symbol"], row), [])
        group.append((row["binary_sha256"], row["binary_bytes"]))
    for action in actions:
        launch, identity = action["launch_identity"], action["identity"]
        selected.setdefault(key(int(identity["Context"]), int(identity["Stream"]), launch["symbol"], launch["configuration"]), []).append(action)
    if set(groups) != set(selected):
        raise ValueError("Nsight report and raw CSV select different launches")
    result = {}
    for group, launches in selected.items():
        binaries = groups[group]
        if len(binaries) != len(launches) or len(set(binaries)) != 1:
            raise ValueError("Nsight report/CSV binary or launch multiplicity is ambiguous")
        digest, size = binaries[0]
        for action in launches:
            result[action["identity"]["ID"]] = {**action["launch_identity"], "binary_sha256": digest, "binary_bytes": size}
    return result


def binding_valid(binding):
    if (not isinstance(binding, dict) or any(not sha(binding.get(k)) for k in BINDINGS)
            or binding["evaluator_sha256"] != evaluator_digest() or binding.get("precision") not in {"fp32", "fp16", "bf16"}
            or binding.get("target") not in {"candidate", "reference"}
            or type(binding.get("seed")) is not int or not 0 <= binding["seed"] < 2**32
            or binding.get("initialization_seed") != 42 or binding.get("forward_seed") != 42
            or type(binding.get("training")) is not bool or binding.get("no_grad") is not True
            or type(binding.get("pid")) is not int or binding["pid"] <= 0
            or binding.get("device_identity_sha256") != identity(binding.get("device", {}))
            or binding.get("output_verified") is not True or binding.get("environment_changes") != []):
        raise ValueError("Counter capture lacks current source/input/device/output bindings")
    return binding


def record(raw, policy, binding, tool, *, simulated=False):
    binding_valid(binding)
    observed = parse_csv(raw, policy, pid=binding["pid"])
    value = {"protocol": PROTOCOL, "recipe": policy, "binding": binding, "tool": tool,
             "raw_csv": raw.decode("utf-8"), "observed": observed, "simulated": simulated,
             "scope": "single_public_initial_state_forward; separate_instrumented_kernel_replay",
             "scored_wall_times_replaced": False, "origin_or_complete_native_coverage_proven": False}
    value["receipt_sha256"] = fingerprint(value)
    validate(value, allow_simulated=simulated)
    return value


def validate(value, *, allow_simulated=False):
    if (not isinstance(value, dict) or value.get("protocol") != PROTOCOL
            or value.get("receipt_sha256") != fingerprint({k: v for k, v in value.items() if k != "receipt_sha256"})
            or (value.get("simulated") is not False and not allow_simulated)
            or value.get("scope") != "single_public_initial_state_forward; separate_instrumented_kernel_replay"
            or value.get("scored_wall_times_replaced") is not False or value.get("origin_or_complete_native_coverage_proven") is not False):
        raise ValueError("Counter record is missing, changed or simulated")
    binding_valid(value.get("binding"))
    tool = value.get("tool", {})
    if (not isinstance(tool, dict) or tool.get("name") != "ncu" or not sha(tool.get("sha256"))
            or not isinstance(tool.get("version"), str) or not tool["version"].strip()
            or tool.get("version_sha256") != hashlib.sha256(tool["version"].encode()).hexdigest()
            or tool.get("binary_unchanged") is not True or tool.get("exit_code") != 0
            or tool.get("timed_out") is not False or tool.get("configuration") != "config_file_off; kernel_replay; profiler_API_single_call; base_units; all_children"):
        raise ValueError("Counter record lacks a completed, unchanged vendor-tool identity")
    raw = value.get("raw_csv")
    if not isinstance(raw, str) or parse_csv(raw.encode(), value.get("recipe"), pid=value["binding"]["pid"]) != value.get("observed"):
        raise ValueError("Counter totals differ from their complete raw report")
    return value


def execution_values(binding, policy, work, calibration, *, digest=None):
    from ..analysis.calibration import validate as validate_calibration
    from ..analysis.headroom import sol
    from ..numerics.precision_calibration import numeric_class

    validate_calibration(calibration, binding["device"], digest=digest)
    if (policy["coverage"] != "reviewed_complete_workload"
            or binding["source_sha256"] not in policy["review"]["source_sha256"]
            or {key: binding[key] for key in WORKLOAD_FIELDS} not in policy["review"]["workload_contracts"]
            or numeric_class(binding["device"]) != policy["review"]["numeric_class_sha256"]):
        raise ValueError("Separate A/E needs complete reviewed workload/resource coverage on this device class")
    work = {k: v for k, v in work.items() if v > 0}
    for name in work:
        if calibration.get("probes", {}).get(name, {}).get("unit") != policy["resources"][name]["rate_unit"]:
            raise ValueError("Counter work and calibration rates use different resource units")
    return sol(work, calibration)


def execution_model(counters, calibration):
    validate(counters)
    latency, components = execution_values(counters["binding"], counters["recipe"], counters["observed"]["work"], calibration)
    result = {"protocol": MODEL, "counters": counters, "calibration": calibration,
              "sol_execution_ms": latency, "components_ms": components,
              "interpretation": "reviewed_resource_demand_over_attained_rates; not_a_physical_lower_bound"}
    result["receipt_sha256"] = fingerprint(result)
    return result


def validate_model(value):
    if (not isinstance(value, dict) or value.get("protocol") != MODEL
            or value.get("receipt_sha256") != fingerprint({k: v for k, v in value.items() if k != "receipt_sha256"})):
        raise ValueError("Separate A/E requires replayable source-bound work counters and calibration")
    expected = execution_model(value.get("counters"), value.get("calibration"))
    if value != expected:
        raise ValueError("Execution SOL differs from its raw counters and calibrated rates")
    return expected
