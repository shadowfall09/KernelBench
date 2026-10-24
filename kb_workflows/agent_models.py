"""Single-response model adapters for the mini-swe-agent kernel environment."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

from .common import atomic_json
from .resource_budgets import BudgetExhausted, Ledger, amount
from .transports import terminate


def nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def source_from_response(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty model response")
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        if len(blocks) != 1:
            raise ValueError("Return exactly one self-contained Python code block")
        text = blocks[0]
    return text.strip() + "\n"


def claude_response(messages, config, directory, timeout, remaining_cost=None):
    # Disable tools, MCP and customizations, keeping the user's configured auth.
    command = ["claude", "-p", "--output-format", "json", "--tools", "", "--disallowedTools", "mcp__*",
               "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--max-turns", "1",
               "--no-session-persistence", "--safe-mode"]
    if remaining_cost is not None:
        command += ["--max-budget-usd", str(max(remaining_cost, 0.000001))]
    if config.model:
        from .generation import agent_environment

        command += ["--model", agent_environment(config).get("ANTHROPIC_MODEL", config.model)]
    from .generation import agent_environment

    with (Path(directory) / "claude.stderr.log").open("a") as log:
        process = subprocess.Popen(command, cwd=directory, env={**os.environ, **agent_environment(config)},
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                                   start_new_session=True)
        try:
            output, _ = process.communicate(json.dumps(messages), timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill descendants before their parent exits and they are reparented.
            # They may have created their own sessions, outside Claude's group.
            from .generation import terminate_agent_tree

            terminate_agent_tree(process)
            raise
        finally:
            terminate(process)
    response = json.loads(output)
    usage = response.get("usage", {})
    token_fields = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    total_tokens = sum(usage[key] for key in token_fields) if all(nonnegative(usage.get(key)) for key in token_fields) else None
    return {"text": response.get("result", ""), "cost": response.get("total_cost_usd"),
            "models": sorted(response.get("modelUsage", {})) or None,
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "total_tokens": total_tokens,
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "provider_error": response.get("is_error", False) or process.returncode != 0 or response.get("num_turns", 1) != 1,
            "num_turns": response.get("num_turns")}


def litellm_response(messages, config, directory, timeout, remaining_cost=None, *, max_output_tokens=None):
    import litellm

    options = {}
    if max_output_tokens is not None:
        supported = litellm.get_supported_openai_params(model=config.model) or []
        parameter = next((key for key in ("max_completion_tokens", "max_tokens") if key in supported), None)
        if parameter is None:
            raise ValueError("Provider cannot enforce a completion-token limit; refusing the request")
        options = {parameter: max_output_tokens, "drop_params": False, "n": 1}
    response = litellm.completion(model=config.model, messages=[{"role": m["role"], "content": m["content"]}
                                                            for m in messages], timeout=timeout, num_retries=0, **options)
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = None
    usage = response.usage
    return {"text": response.choices[0].message.content, "cost": cost,
            "models": [response.model] if response.model else None,
            "input_tokens": getattr(usage, "prompt_tokens", None), "output_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None)}


class KernelModel:
    """mini-swe-agent Model protocol with a code-submission action, without shell access."""

    def __init__(self, config, directory, remaining, response_fn=None):
        self.config, self.directory, self.remaining = config, Path(directory), remaining
        if response_fn is None and config.agent_backend == "local-transformers":
            from .local_model import LocalTransformersModel

            response_fn = LocalTransformersModel(config)
        self.response_fn = response_fn or (claude_response if config.agent_backend == "claude-code" else litellm_response)
        self.receipts = []
        self.ledgers = {name: Ledger(limit, "tokens", integer=True) for name, limit in (
            ("total_tokens", config.agent_token_limit), ("output_tokens", config.agent_output_token_limit)) if limit}

    def query(self, messages):
        reservations, prompt_tokens = {}, None
        cap = self.config.agent_max_output_tokens
        if "total_tokens" in self.ledgers:
            prompt_tokens = self.response_fn.count_tokens(messages)
            if not amount(prompt_tokens, integer=True) or prompt_tokens < 1:
                raise ValueError("Exact local tokenizer evidence is required before reserving tokens")
            cap = min(cap, self.ledgers["total_tokens"].remaining - prompt_tokens)
        if "output_tokens" in self.ledgers:
            cap = min(cap, self.ledgers["output_tokens"].remaining)
        if cap <= 0:
            raise BudgetExhausted("No room for another prompt and completion; retained prior best candidate")
        for name, ledger in self.ledgers.items():
            reservations[name] = ledger.reserve(cap + (prompt_tokens if name == "total_tokens" else 0))
        remaining_cost = None
        if self.config.agent_cost_limit:
            remaining_cost = self.config.agent_cost_limit - sum(r["cost"] for r in self.receipts if nonnegative(r.get("cost")))
        try:
            options = {"max_output_tokens": cap} if self.ledgers or self.config.agent_backend == "local-transformers" else {}
            response = self.response_fn(messages, self.config, self.directory, self.remaining(), remaining_cost, **options)
            if not isinstance(response, dict):
                raise ValueError("Model response must be a usage-bearing object")
        except Exception as exc:
            for name, nonce in reservations.items():
                self.ledgers[name].settle(nonce)
            self.receipts.append({"error": type(exc).__name__})
            atomic_json(self.directory / "model_calls" / f"{len(self.receipts):03}.json", self.receipts[-1])
            raise
        resource_error = False
        for name, nonce in reservations.items():
            observed = response.get(name)
            if name == "total_tokens" and (response.get("input_tokens") != prompt_tokens
                    or not amount(response.get("output_tokens"), integer=True)
                    or observed != prompt_tokens + response["output_tokens"] or response["output_tokens"] > cap):
                observed = None
            resource_error |= not self.ledgers[name].settle(nonce, observed)
        if reservations:
            response["resource_reservations"] = {"nonces": reservations, "max_output_tokens": cap,
                                                  "prompt_tokens": prompt_tokens, "compliant": not resource_error}
        for key in ("cost", "input_tokens", "output_tokens", "total_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            if not nonnegative(response.get(key)):
                response[key] = None
        self.receipts.append(response)
        atomic_json(self.directory / "model_calls" / f"{len(self.receipts):03}.json", response)
        action = {}
        try:
            if resource_error:
                action["stop"] = True
                action["exit_status"] = "ResourceEvidenceMissingOrExceeded"
                raise ValueError("Resource usage was missing or exceeded its reservation; stopping without evaluating this response")
            if self.config.agent_cost_limit and not nonnegative(response.get("cost")):
                action["stop"] = True
                raise ValueError("Estimated-cost cap requires cost evidence; provider returned none")
            if response.get("provider_error"):
                raise ValueError("Model provider reported an unsuccessful or multi-turn response")
            action["source"] = source_from_response(response.get("text"))
        except ValueError as exc:
            action["error"] = str(exc)
        return self.format_message(role="assistant", content=response.get("text") or "", extra={
            "actions": [action], "cost": response["cost"] if nonnegative(response.get("cost")) else 0.0})

    @staticmethod
    def format_message(**kwargs):
        return kwargs

    @staticmethod
    def format_observation_messages(message, outputs, template_vars=None):
        return [{"role": "user", "content": json.dumps(output, ensure_ascii=False)} for output in outputs]

    @staticmethod
    def get_template_vars(**kwargs):
        return kwargs

    def serialize(self):
        usage = self.usage()
        return {"info": {"model": {"backend": self.config.agent_backend, "name": self.config.model},
                         "model_stats": {"instance_cost": usage["estimated_cost_usd"], "api_calls": usage["model_calls"],
                                         "cost_status": "unknown" if usage["estimated_cost_usd"] is None else "estimated"}}}

    def usage(self):
        result = {"model_calls": len(self.receipts)}
        result["models"] = sorted({name for item in self.receipts for name in item.get("models") or []}) or None
        if self.config.agent_backend == "local-transformers":
            snapshots = [item.get("model_snapshot_sha256") for item in self.receipts]
            result["local_checkpoint_sha256"] = sorted(set(snapshots)) if snapshots and all(isinstance(value, str) for value in snapshots) else None
        for field, source in (("estimated_cost_usd", "cost"), ("input_tokens", "input_tokens"), ("total_tokens", "total_tokens"),
                              ("output_tokens", "output_tokens"), ("cache_read_input_tokens", "cache_read_input_tokens"),
                              ("cache_creation_input_tokens", "cache_creation_input_tokens")):
            values = [item.get(source) for item in self.receipts]
            result[field] = sum(values) if values and all(nonnegative(value) for value in values) else None
        result["resource_budgets"] = {name: ledger.snapshot() for name, ledger in self.ledgers.items()}
        return result
