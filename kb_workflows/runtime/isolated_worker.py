"""Restricted-container bootstrap; checks the image and Linux policy first."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..common import atomic_json, evaluator_digest, failure, fingerprint, read_json
from .isolation import policy, proc_issues, proc_observation


def main():
    source, target = map(Path, sys.argv[1:3])
    payload = read_json(source)
    try:
        settings = payload["eval_config"]
        if settings.get("evaluator_sha256") != evaluator_digest():
            raise ValueError("Restricted verifier image does not contain the frozen evaluator")
        spec = policy(settings["isolation"])
        observation = proc_observation()
        if issues := proc_issues(observation, spec):
            raise ValueError("Linux restrictions were not observed: " + ", ".join(issues))
        Path("/tmp/kb-home").mkdir(exist_ok=True)
        from ..evaluation.evaluate_task import main as evaluate

        sys.argv = [sys.argv[0], str(source), str(target)]
        evaluate()
        result = read_json(target)
        result.setdefault("metadata", {})["restricted_worker"] = {"observation": observation, "payload_sha256": fingerprint(payload),
            "scope": "Linux process/mount/network/cgroup observations before candidate import; not a complete security certification"}
    except Exception as exc:
        result = failure(str(exc), error_name=type(exc).__name__)
    atomic_json(target, result)
    # The rootful daemon maps the worker to UID 65532; the owning host user
    # reads this handoff after container exit, within a private parent directory.
    os.chmod(target, 0o644, follow_symlinks=False)


if __name__ == "__main__":
    main()
