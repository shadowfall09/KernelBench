"""Single-response model adapters for the mini-swe-agent kernel environment."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

from .common import atomic_json
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


def litellm_response(messages, config, directory, timeout, remaining_cost=None):
    import litellm

    response = litellm.completion(model=config.model, messages=[{"role": m["role"], "content": m["content"]}
                                                            for m in messages], timeout=timeout, num_retries=0)
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
        self.response_fn = response_fn or (claude_response if config.agent_backend == "claude-code" else litellm_response)
        self.receipts = []

    def query(self, messages):
        remaining_cost = None
        if self.config.agent_cost_limit:
            remaining_cost = self.config.agent_cost_limit - sum(r["cost"] for r in self.receipts if nonnegative(r.get("cost")))
        try:
            response = self.response_fn(messages, self.config, self.directory, self.remaining(), remaining_cost)
        except Exception as exc:
            self.receipts.append({"error": type(exc).__name__})
            atomic_json(self.directory / "model_calls" / f"{len(self.receipts):03}.json", self.receipts[-1])
            raise
        for key in ("cost", "input_tokens", "output_tokens", "total_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            if not nonnegative(response.get(key)):
                response[key] = None
        self.receipts.append(response)
        atomic_json(self.directory / "model_calls" / f"{len(self.receipts):03}.json", response)
        action = {}
        try:
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
        for field, source in (("estimated_cost_usd", "cost"), ("input_tokens", "input_tokens"), ("total_tokens", "total_tokens"),
                              ("output_tokens", "output_tokens"), ("cache_read_input_tokens", "cache_read_input_tokens"),
                              ("cache_creation_input_tokens", "cache_creation_input_tokens")):
            values = [item.get(source) for item in self.receipts]
            result[field] = sum(values) if values and all(nonnegative(value) for value in values) else None
        return result
