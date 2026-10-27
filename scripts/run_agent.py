#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kb_workflows.agents.generation import run_agent
from kb_workflows.common import atomic_json
from kb_workflows.config import RunConfig


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--admission", type=Path)
    args = parser.parse_args(argv)
    job = json.loads(args.config.read_text())
    config = RunConfig(**job).validate()
    if args.admission:
        from kb_workflows.runtime.modal_backend import admitted_generation

        result = admitted_generation(args.directory, job, json.loads(args.admission.read_text()),
                                     lambda: run_agent(args.directory, config, config.start))
    else:
        result = run_agent(args.directory, config, config.start)
    atomic_json(args.directory / "agent_result.json", result)
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
