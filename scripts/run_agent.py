#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kb_workflows.common import atomic_json
from kb_workflows.config import RunConfig
from kb_workflows.generation import run_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    config = RunConfig(**json.loads(args.config.read_text())).validate()
    result = run_agent(args.directory, config, config.start)
    atomic_json(args.directory / "agent_result.json", result)
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
