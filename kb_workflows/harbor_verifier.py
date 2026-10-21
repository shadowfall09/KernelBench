"""Harbor reward bridge. Runs only in the fresh, trusted verifier environment."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

from .common import atomic_json, evaluator_digest, kernel_name, read_json
from .config import RunConfig


def verify(manifest, candidate, logs):
    logs = Path(logs)
    logs.mkdir(parents=True, exist_ok=True)
    reward_path = logs / "reward.txt"
    reward_path.write_text("0\n")
    (logs / "reward.json").unlink(missing_ok=True)
    report = {"reward": 0.0, "correctness": False, "bounded_gain": 0.0}
    try:
        spec = read_json(Path(manifest))
        if spec.get("protocol") != "kernelbench-pro-harbor-v1" or spec.get("evaluator_sha256") != evaluator_digest():
            raise ValueError("Harbor task and verifier protocol/source versions differ")
        mode = spec["reward"]
        if mode not in {"correctness", "bounded_gain"}:
            raise ValueError("Unknown Harbor reward mode")
        config = RunConfig(**spec["config"]).validate()
        if not config.hidden_tests or config.num_samples != 1 or config.start != config.end or config.evaluation != "local":
            raise ValueError("Harbor verification requires one local sample and hidden tests")
        # Every invocation is a new evaluation, including regrades in one container.
        config.resume = False
        config.runs_dir = str(logs / "kernelbench-pro" / "runs")
        if config.run_dir.exists():
            shutil.rmtree(config.run_dir)
        config.run_dir.mkdir(parents=True)
        destination = config.run_dir / kernel_name(config.level, config.start)
        submission = Path(candidate)
        if submission.is_symlink() or not submission.is_file() or not submission.stat().st_size:
            raise ValueError("Missing or empty /app/solution.py submission")
        destination.write_bytes(submission.read_bytes())
        atomic_json(config.run_dir / "run_config.json", config.to_dict())
        from .engine import evaluate_run, record_baseline
        from .paper_analysis import analyze_paper

        record_baseline(config)
        evaluate_run(config)
        scores = analyze_paper(config)
        key = f"{config.start}_0"
        standard = read_json(config.run_dir / "eval_results.json", {}).get(key, {})
        hidden = read_json(config.run_dir / "eval_results_hidden.json", {}).get(key, {})
        correct = bool(standard.get("compiled") and standard.get("correctness")
                       and hidden.get("compiled") and hidden.get("correctness"))
        gain = scores["bounded_gain_macro"]
        if type(gain) not in (float, int) or not math.isfinite(gain) or not 0 <= gain <= 1:
            raise ValueError("Invalid native performance score")
        if not correct or scores["details"][0]["status"] != "verified":
            gain = 0.0
        report.update(correctness=correct, bounded_gain=gain, reward_mode=mode,
                      reward=float(correct) if mode == "correctness" else float(gain),
                      performance_status=scores["details"][0]["status"],
                      native_results=str(config.run_dir))
    except Exception as exc:
        report.update(reward=0.0, error=f"{type(exc).__name__}: {exc}")
    atomic_json(logs / "kernelbench_pro_report.json", report)
    reward_path.write_text(f"{report['reward']}\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="KernelBench Pro Harbor verifier")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=Path("/app/solution.py"))
    parser.add_argument("--logs", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args(argv)
    return 2 if "error" in verify(args.manifest, args.candidate, args.logs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
