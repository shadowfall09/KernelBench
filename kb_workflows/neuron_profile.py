"""Explicit-unit Neuron instruction timelines; host events are never device work."""

from __future__ import annotations

from .calibration import positive
from .performance import interval_union

ENGINE_LABELS = {"TensorMatrix": "tensor", "Vector": "vector", "VectorE": "vector", "Scalar": "scalar",
                 "ScalarE": "scalar", "DMA": "dma", "Dma": "dma", "DMAEngine": "dma"}
UNITS_TO_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0}


def timeline(trace, *, timestamp_unit, window, origins=None):
    """Parse neuron-profile view JSON for one declared capture window.

    Source locations alone do not establish authorship. These are diagnostics
    until a trusted NEFF/source association has been established.
    """
    if timestamp_unit not in UNITS_TO_MS or not isinstance(window, (list, tuple)) or len(window) != 2:
        raise ValueError("Neuron timelines require explicit ns/us/ms units and a capture window")
    interval_union([window])
    events = trace.get("instruction")
    if not isinstance(events, list) or not events:
        raise ValueError("No Neuron device instruction data; host/runtime events cannot substitute")
    left, right = window
    rows = []
    for index, event in enumerate(events):
        start, duration = event.get("timestamp"), event.get("duration")
        if type(start) not in (int, float) or not positive(duration):
            raise ValueError("Malformed Neuron instruction timing")
        interval_union([(start, start + duration)])
        if start + duration <= left or start >= right:
            continue
        origin = (origins or {}).get(str(index), "unknown")
        if origin not in {"custom", "library", "unknown"}:
            raise ValueError("Unknown instruction origin")
        rows.append({"start": max(start, left), "end": min(start + duration, right),
                     "engine": ENGINE_LABELS.get(event.get("label"), "unknown"), "origin": origin,
                     "opcode": event.get("opcode"), "nki_source_location": event.get("nki_source_location")})
    if not rows:
        raise ValueError("No device instructions fall inside the declared capture window")
    factor = UNITS_TO_MS[timestamp_unit]

    def union(group):
        return interval_union([(row["start"], row["end"]) for row in group]) * factor

    busy = union(rows)
    other = union([row for row in rows if row["origin"] != "custom"])
    return {"protocol": "neuron_instruction_union_v1", "device_backend": "neuron", "timestamp_unit": timestamp_unit,
            "capture_window": list(window), "capture_span_ms": (right - left) * factor,
            "wall_ms": None, "device_busy_ms": busy, "kernel_count": None, "instruction_count": len(rows),
            "per_engine_busy_ms": {engine: union([row for row in rows if row["engine"] == engine])
                                   for engine in ("tensor", "vector", "scalar", "dma", "unknown")},
            "custom_exclusive_fraction": max(0, busy - other) / busy,
            "unknown_origin_count": sum(row["origin"] == "unknown" for row in rows),
            "origin_map_supplied": origins is not None, "activities": rows,
            "timings_are_diagnostic": True, "hardware_counters_collected": False,
            "scored_wall_times_replaced": False,
            "scope": "instruction_union; not_host_wall_time_or_kernel_count; explicit_capture_binding_required"}
