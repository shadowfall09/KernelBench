"""Read Harbor 0.24 trial accounting without treating absent step fields as zero."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from ..common import fingerprint, read_json


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def harbor_budget(directory):
    files = list(Path(directory).glob("*/result.json"))
    result = {"model_calls": None, "candidate_submissions": None, "development_evaluations": None, "profile_requests": None,
              "input_tokens": None, "output_tokens": None, "total_tokens": None, "estimated_cost_usd": None,
              "generation_wall_seconds": None, "development_request_wall_seconds": None, "gpu_execution_seconds": None,
              "models": None, "resource_receipt_verified": False, "billing_cost_certified": False,
              "scope": "Harbor_agent_execution; input_includes_cache; excludes_setup_and_verifier"}
    if len(files) != 1 or files[0].is_symlink():
        result["reason"] = "Missing or ambiguous single-trial result"
        return result
    raw = read_json(files[0])
    if not isinstance(raw, dict):
        result["reason"] = "Malformed Harbor result"
        return result
    contexts = ([raw["agent_result"]] if isinstance(raw.get("agent_result"), dict) else
                [step.get("agent_result") for step in raw.get("step_results") or []])
    # Multi-step omissions are unknown, even if another step has a measurement.
    def known_sum(name):
        values = [ctx.get(name) if isinstance(ctx, dict) else None for ctx in contexts]
        valid = all(finite(value) and value >= 0 and (name == "cost_usd" or type(value) is int) for value in values)
        return sum(values) if values and valid else None
    result.update(input_tokens=known_sum("n_input_tokens"), output_tokens=known_sum("n_output_tokens"),
                  estimated_cost_usd=known_sum("cost_usd"), cache_tokens=known_sum("n_cache_tokens"))
    if result["input_tokens"] is not None and result["output_tokens"] is not None:
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    models = sorted({name for ctx in contexts if isinstance(ctx, dict) for name in ctx.get("model_usage") or {}})
    result["models"] = models or None
    timing = raw.get("agent_execution") or {}
    try:
        started, finished = (datetime.fromisoformat(timing[key].replace("Z", "+00:00")) for key in ("started_at", "finished_at"))
        if started.tzinfo is not None and finished.tzinfo is not None:
            elapsed = (finished - started).total_seconds()
            if elapsed >= 0:
                result["generation_wall_seconds"] = elapsed
    except (KeyError, TypeError, ValueError):
        pass
    result["resource_receipt_verified"] = result["total_tokens"] is not None and result["generation_wall_seconds"] is not None
    result["harbor_result_sha256"] = fingerprint(raw)
    return result


OUTPUT_PROTOCOL = "kernelbench_pro_harbor_output_quota_v1"
ADAPTER = "kb_workflows.runtime.harbor_budget:BudgetedTerminus2"


class _QuotaStop(BaseException):
    """Escape Harbor's Exception-based retry loops; caught by our run boundary."""


def _binding(config, pid, sid, nonce):
    from ..agents.generation import generation_signature
    from ..common import evaluator_digest

    return {"protocol": OUTPUT_PROTOCOL, "nonce": nonce, "problem_id": pid, "sample_id": sid,
            "generation_signature": generation_signature(config, pid, sid), "evaluator_sha256": evaluator_digest(),
            "model": config.model, "limit": config.agent_output_token_limit,
            "max_output_tokens": config.agent_max_output_tokens, "wall_seconds": config.agent_timeout,
            "adapter": ADAPTER, "harbor_version": "0.24.0", "max_provider_calls": 10000,
            "network_scope": "terminal_no_network; host_provider_only"}


def prepare_quota(directory, config, pid, sid):
    import uuid

    from ..common import atomic_json

    root = Path(directory) / "harbor_quota"
    root.mkdir(exist_ok=False)
    atomic_json(root / "admission.json", _binding(config, pid, sid, uuid.uuid4().hex))
    return root / "admission.json"


class OutputGate:
    """One host-controlled provider-request ledger, including summary/retry calls."""

    def __init__(self, admission, expected_sha256):
        from ..agents.resource_budgets import Ledger

        path = Path(admission)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Harbor quota admission must be a regular local file")
        self.root, self.binding = path.parent, read_json(path)
        if fingerprint(self.binding) != expected_sha256:
            raise ValueError("Harbor quota admission differs from its controller argument")
        self.ledger = Ledger(self.binding["limit"], "tokens", integer=True)
        self.calls, self.reason, self.poisoned = [], None, False
        self.finished = False

    def save(self):
        from ..common import atomic_json

        value = {"binding": self.binding, "ledger": self.ledger.snapshot(), "calls": self.calls,
                 "finished": self.finished, "reason": self.reason, "poisoned": self.poisoned,
                 "scope": "host_LiteLLM_requests_including_summaries; output_tokens_only; no_billing_certification"}
        value["receipt_sha256"] = fingerprint(value)
        atomic_json(self.root / "state.json", value)

    def begin(self):
        import time

        from ..common import evaluator_digest

        if (self.binding.get("protocol") != OUTPUT_PROTOCOL or self.binding.get("evaluator_sha256") != evaluator_digest()
                or read_json(self.root / "admission.json") != self.binding):
            raise ValueError("Harbor quota admission differs from the current evaluator")
        with (self.root / "admitted.json").open("x") as stream:
            import json

            json.dump(self.binding, stream)
        self.deadline = time.monotonic() + self.binding["wall_seconds"]
        self.save()

    async def request(self, provider, **kwargs):
        import asyncio
        import time

        import litellm

        from ..agents.resource_budgets import amount

        remaining = self.deadline - time.monotonic()
        if self.poisoned or remaining <= 0 or self.ledger.remaining <= 0 or len(self.calls) >= self.binding["max_provider_calls"]:
            self.reason = ("usage_unknown" if self.poisoned else "wall_expired" if remaining <= 0 else
                           "quota_exhausted" if self.ledger.remaining <= 0 else "provider_call_envelope_exhausted")
            self.save()
            raise _QuotaStop()
        if (kwargs.get("model") != self.binding["model"] or kwargs.get("n", 1) != 1
                or getattr(litellm, "model_fallbacks", None)
                or any(kwargs.get(key) for key in ("fallbacks", "context_window_fallback_dict", "mock_response"))):
            self.poisoned, self.reason = True, "unsupported_provider_route"
            self.save()
            raise _QuotaStop()
        supported = litellm.get_supported_openai_params(model=self.binding["model"]) or []
        parameter = next((name for name in ("max_completion_tokens", "max_tokens") if name in supported), None)
        if parameter is None:
            self.poisoned, self.reason = True, "provider_limit_unavailable"
            self.save()
            raise _QuotaStop()
        maximum = min(self.binding["max_output_tokens"], self.ledger.remaining)
        options = {key: value for key, value in kwargs.items() if key not in {"max_tokens", "max_completion_tokens", "num_retries", "max_retries", "stream", "n", "drop_params"}}
        options.update({parameter: maximum, "num_retries": 0, "max_retries": 0, "stream": False, "n": 1, "drop_params": False, "timeout": remaining})
        nonce = self.ledger.reserve(maximum)
        row = {"nonce": nonce, "reserved": maximum, "parameter": parameter, "request_sha256": fingerprint(options),
               "output_tokens": None, "input_tokens": None, "cache_tokens": None, "cost": None, "model": None,
               "request_policy": {"n": 1, "stream": False, "num_retries": 0, "max_retries": 0, "drop_params": False}}
        self.calls.append(row)
        self.save()  # A lost reply leaves an outstanding reservation, never a fresh budget.
        try:
            response = await asyncio.wait_for(provider(**options), timeout=remaining)
            usage = getattr(response, "usage", None)
            observed = getattr(usage, "completion_tokens", None)
            cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
            row.update(cache_tokens=cached if amount(cached, integer=True) else None,
                       output_tokens=observed if amount(observed, integer=True) else None,
                       input_tokens=getattr(usage, "prompt_tokens", None) if amount(getattr(usage, "prompt_tokens", None), integer=True) else None,
                       model=getattr(response, "model", None) if isinstance(getattr(response, "model", None), str) and len(response.model) <= 512 else None)
            try:
                cost = litellm.completion_cost(completion_response=response)
                row["cost"] = cost if amount(cost) else None
            except Exception:
                pass
            compliant = self.ledger.settle(nonce, observed)
            if not compliant:
                self.poisoned, self.reason = True, "usage_missing_or_exceeded"
                self.save()
                raise _QuotaStop()
            self.save()
            return response
        except _QuotaStop:
            raise
        except BaseException as exc:
            if nonce in self.ledger.pending:
                self.ledger.settle(nonce)
            self.poisoned, self.reason = True, type(exc).__name__
            self.save()
            raise _QuotaStop() from exc


def __getattr__(name):
    if name != "BudgetedTerminus2":
        raise AttributeError(name)
    from importlib.metadata import version

    if version("harbor") != "0.24.0":
        raise ValueError("Controlled Harbor output quotas require Harbor 0.24.0")
    import litellm
    from harbor.agents.terminus_2 import Terminus2

    class BudgetedTerminus2(Terminus2):
        def __init__(self, *args, budget_admission, budget_binding_sha256, **kwargs):
            if (kwargs.get("use_responses_api") or kwargs.get("llm_backend", "litellm") != "litellm"
                    or kwargs.get("llm_kwargs") or kwargs.get("llm_call_kwargs") or kwargs.get("mcp_servers")):
                raise ValueError("Controlled Harbor quotas require the pinned direct LiteLLM path")
            self._quota = OutputGate(budget_admission, budget_binding_sha256)
            super().__init__(*args, **kwargs)
            if self._model_name != self._quota.binding["model"]:
                raise ValueError("Harbor model differs from its quota admission")

        async def run(self, *args, **kwargs):
            gate = self._quota
            gate.begin()
            original = litellm.acompletion
            async def bounded(**options):
                return await gate.request(original, **options)
            litellm.acompletion = bounded
            try:
                await super().run(*args, **kwargs)
            except _QuotaStop:
                pass
            finally:
                if litellm.acompletion is not bounded:
                    gate.poisoned, gate.reason = True, "provider_hook_changed"
                litellm.acompletion = original
                gate.finished = True
                gate.save()
    BudgetedTerminus2.__module__, BudgetedTerminus2.__qualname__ = __name__, name
    globals()[name] = BudgetedTerminus2
    return BudgetedTerminus2


def quota_receipt(directory, config, pid, sid):
    from ..agents.resource_budgets import amount, validate_snapshot

    root = Path(directory) / "harbor_quota"
    admission, value = read_json(root / "admission.json"), read_json(root / "state.json")
    binding = _binding(config, pid, sid, admission["nonce"])
    if (admission != binding or read_json(root / "admitted.json") != binding or value.get("binding") != binding
            or value.get("receipt_sha256") != fingerprint({key: item for key, item in value.items() if key != "receipt_sha256"})
            or value.get("finished") is not True or value.get("poisoned") is not False
            or not validate_snapshot(value.get("ledger"), config.agent_output_token_limit, "tokens")):
        raise ValueError("Harbor output quota is missing, unfinished, changed or exceeded")
    calls, entries = value.get("calls"), value["ledger"]["entries"]
    if not isinstance(calls, list) or len(calls) != len(entries) or len(calls) > 10000:
        raise ValueError("Harbor output quota lost provider-call evidence")
    by_nonce = {row["nonce"]: row for row in entries}
    if len({row.get("nonce") for row in calls}) != len(calls):
        raise ValueError("Harbor quota repeats a provider nonce")
    for call in calls:
        entry = by_nonce.get(call.get("nonce"), {})
        if (call.get("nonce") != entry["nonce"] or call.get("reserved") != entry["reserved"]
                or call["reserved"] > config.agent_max_output_tokens or call.get("output_tokens") != entry["observed"]
                or call.get("parameter") not in {"max_tokens", "max_completion_tokens"}
                or call.get("request_policy") != {"n": 1, "stream": False, "num_retries": 0, "max_retries": 0, "drop_params": False}
                or not isinstance(call.get("request_sha256"), str) or len(call["request_sha256"]) != 64):
            raise ValueError("Harbor output quota differs from its admitted provider calls")
    def total(name):
        values = [row.get(name) for row in calls]
        return sum(values) if all(amount(item, integer=name != "cost") for item in values) else None
    return value, {"model_calls": len(calls), "input_tokens": total("input_tokens"), "output_tokens": total("output_tokens"),
                   "cache_tokens": total("cache_tokens"), "estimated_cost_usd": total("cost"), "models": sorted({row["model"] for row in calls if isinstance(row.get("model"), str) and row["model"]}) or None}


def attach_quota(budget, directory, config, pid, sid):
    from ..analysis.experiment import protocol

    value, usage = quota_receipt(directory, config, pid, sid)
    budget["reported_harbor_usage"] = {key: budget.get(key) for key in ("input_tokens", "output_tokens", "cache_tokens", "estimated_cost_usd")}
    budget.update(usage)
    budget.update(total_tokens=usage["input_tokens"] + usage["output_tokens"] if usage["input_tokens"] is not None else None,
                  resource_budgets={"output_tokens": value["ledger"]}, protocol={**protocol(config), "samples_per_problem": 1},
                  controlled_output_receipt_sha256=value["receipt_sha256"], controlled_output_binding=value["binding"],
                  scope="host_LiteLLM_normalized_usage_including_summary_calls; input_includes_reported_cache; output_quota_only")
    return budget


def quota_artifact_hashes(run_dir, pid, sid, summary, *, level):
    if type(level) is not int or level not in {1, 2, 3, 5} or type(pid) is not int or pid < 1 or type(sid) is not int or sid < 0:
        raise ValueError("Invalid Harbor quota slot identity")
    base = Path(run_dir).resolve()
    root = base / "work" / f"p{pid}_s{sid}"
    job = Path(summary["harbor_job"]).resolve()
    if not job.resolve().is_relative_to((root / "harbor_jobs").resolve()):
        raise ValueError("Harbor quota job is outside its generation slot")
    results = list(job.glob("*/result.json"))
    sources = list(job.glob("*/artifacts/app/solution.py"))
    if len(results) != 1 or len(sources) != 1:
        raise ValueError("Harbor quota needs one original result and one source artifact")
    import hashlib

    from ..common import kernel_name

    output = {}
    paths = [root / "harbor_quota" / name for name in ("admission.json", "admitted.json", "state.json")] + results + sources
    for path in paths:
        if (not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                or path.stat().st_size > 64 * 1024**2):
            raise ValueError("Harbor quota artifact is missing, excessive or outside its job")
        output[path.relative_to(base).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if output[sources[0].relative_to(base).as_posix()] != hashlib.sha256((base / kernel_name(level, pid, sid)).read_bytes()).hexdigest():
        raise ValueError("Harbor source differs from the selected generation artifact")
    return output


def score_issue(config, pid, sid):
    from ..agents.generation import generation_signature
    from ..analysis.experiment import existing_budget, protocol

    budget = existing_budget(config)
    history = read_json(config.run_dir / "generation_attempts.json", {})
    summary = read_json(config.run_dir / "generation_summary.json", {}).get(f"{pid}_{sid}", {})
    attempts = history.get("slots", {}).get(f"{pid}_{sid}", [])
    try:
        quota_artifact_hashes(config.run_dir, pid, sid, summary, level=config.level)
        receipt = attach_quota(harbor_budget(Path(summary["harbor_job"])), config.run_dir / "work" / f"p{pid}_s{sid}", config, pid, sid)
        if (budget.get("provenance_valid") is not True or budget.get("protocol") != protocol(config)
                or budget.get("attempt_history") != history or history.get("historical_attempts_unknown")
                or len(attempts) != 1 or attempts[0].get("status") != "success" or summary.get("status") != "success"
                or summary.get("signature") != attempts[0].get("signature")
                or summary.get("signature") != generation_signature(config, pid, sid)
                or summary.get("budget") != receipt or attempts[0].get("budget") != receipt
                or budget.get("slots", {}).get(f"{pid}_{sid}") != receipt):
            return "resource_budget_harbor_provenance_or_usage_mismatch"
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return "resource_budget_harbor_output_evidence_missing_or_invalid"
    return None
