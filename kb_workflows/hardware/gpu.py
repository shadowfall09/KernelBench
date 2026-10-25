"""GPU configuration helpers; usable without Torch, CUDA or cloud SDKs."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

PROFILES_PATH = Path(__file__).with_name("gpu_profiles.toml")


def read_toml(path):
    path = Path(path).expanduser()
    try:
        with path.open("rb") as source:
            return tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Cannot load config {path}: {exc}") from exc


def cuda_architectures(value):
    """Accept Torch architecture names or compute capabilities, leaving support to Torch/nvcc."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("GPU architecture must be a nonempty string, e.g. Ampere or 8.6")
    targets = [item.strip() for item in re.split(r"[,;]", value)]
    if any(not re.fullmatch(r"(?:[A-Za-z][A-Za-z0-9_-]*|[0-9]+\.[0-9]+[a-z]?)(?:\+PTX)?", t) for t in targets):
        raise ValueError(f"Invalid GPU architecture: {value!r}")
    return ",".join(dict.fromkeys(targets))


def gpu_profiles():
    document = read_toml(PROFILES_PATH)
    if set(document) != {"profiles"} or not isinstance(document["profiles"], dict):
        raise ValueError("GPU presets must contain a [profiles] table")
    profiles = {}
    for name, entry in document["profiles"].items():
        if not isinstance(entry, dict) or set(entry) - {"architecture", "modal_type"}:
            raise ValueError(f"Invalid GPU preset: {name}")
        architecture = cuda_architectures(entry.get("architecture"))
        cloud = entry.get("modal_type", "")
        if not isinstance(cloud, str) or ("modal_type" in entry and not cloud.strip()):
            raise ValueError(f"Invalid modal_type for GPU preset {name}")
        profiles[name] = {"hardware": name, "gpu_arch": architecture, "gpu": cloud}
    return profiles


def gpu_options(section):
    if not isinstance(section, dict):
        raise ValueError("[gpu] must be a table")
    allowed = {"profile", "hardware", "architecture", "devices", "modal_type"}
    if unknown := set(section) - allowed:
        raise ValueError(f"Unknown [gpu] options: {', '.join(sorted(unknown))}")
    options = {}
    if "profile" in section:
        name = section["profile"]
        profiles = gpu_profiles()
        if not isinstance(name, str) or name not in profiles:
            raise ValueError(f"Unknown GPU profile {name!r}; choose from {', '.join(profiles)} or set hardware/architecture")
        options.update(profiles[name])
    for key, field in (("hardware", "hardware"), ("architecture", "gpu_arch"), ("modal_type", "gpu")):
        if key in section:
            if not isinstance(section[key], str) or not section[key].strip():
                raise ValueError(f"gpu.{key} must be a nonempty string")
            options[field] = section[key]
    if "devices" in section:
        devices = section["devices"]
        if (not isinstance(devices, list) or not devices
                or any(type(d) is not int or d < 0 for d in devices) or len(set(devices)) != len(devices)):
            raise ValueError("gpu.devices must be a nonempty list of distinct nonnegative integers")
        options["gpus"] = ",".join(map(str, devices))
    return options
