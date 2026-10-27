"""Controller accounting with fake model/worker responses; no model or GPU execution."""

from types import SimpleNamespace

import pytest

from kb_workflows.agents import agent_loop, generation
from kb_workflows.agents import resource_budgets as budgets
from kb_workflows.agents.agent_models import KernelModel, litellm_response
from kb_workflows.agents.local_model import LocalTransformersModel, checkpoint_digest
from kb_workflows.analysis.study_plan import make_plan
from kb_workflows.common import fingerprint, kernel_name, read_json
from kb_workflows.config import RunConfig, parse_config


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("MSWEA_SILENT_STARTUP", "1")
    return RunConfig(run_name="capped", runs_dir=str(tmp_path / "runs"), generation="local", start=90, end=90,
                     hardware="RTX_A6000", agent_profile=False, optimization_rounds=4, agent_timeout=30).validate()


def loop(config, response, evaluator=None):
    directory = generation.prepare_job(config, 90, 0)
    child = RunConfig(**read_json(directory / "job_config.json")).validate()
    result = agent_loop.run_loop(directory, child, 90, response_fn=response, **({"evaluator": evaluator} if evaluator else {}))
    return directory, result


def response(*args, max_output_tokens=None):
    return {"text": "class ModelNew: pass", "input_tokens": 3, "output_tokens": min(2, max_output_tokens or 2),
            "total_tokens": 3 + min(2, max_output_tokens or 2), "cost": 0.01}


response.count_tokens = lambda messages: 3


@pytest.mark.parametrize("bad", [True, -1, float("nan"), float("inf"), "9"])
def test_ledger_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        budgets.Ledger(bad, "tokens", integer=True)


def test_reservations_refund_known_usage_and_burn_unknown_usage():
    ledger = budgets.Ledger(12, "tokens", integer=True)
    first = ledger.reserve(10)
    with pytest.raises(budgets.BudgetExhausted):
        ledger.reserve(3)
    assert ledger.settle(first, 4) and ledger.remaining == 8
    assert budgets.validate_snapshot(ledger.snapshot(), 12, "tokens")
    assert not ledger.settle(ledger.reserve(7))
    assert ledger.remaining == 1 and ledger.spent == 11
    assert not budgets.validate_snapshot(ledger.snapshot(), 12, "tokens")
    with pytest.raises(ValueError):
        ledger.settle(first, 0)


def test_overrun_is_visible_and_malformed_snapshot_fails_closed():
    ledger = budgets.Ledger(10, "tokens", integer=True)
    assert not ledger.settle(ledger.reserve(5), 11)
    assert ledger.remaining == 0 and ledger.spent == 11 and not ledger.snapshot()["compliant"]
    snapshot = budgets.Ledger(10, "tokens", integer=True).snapshot()
    snapshot["entries"] = [None]
    assert not budgets.validate_snapshot(snapshot, 10, "tokens")


def test_exact_total_token_budget_stops_before_dispatch_and_keeps_best(config):
    config.agent_backend, config.model, config.agent_token_limit = "local-transformers", str(config.run_dir.parent), 9
    calls = []

    def model(*args, **kwargs):
        calls.append(kwargs["max_output_tokens"])
        return response(*args, **kwargs)
    model.count_tokens = response.count_tokens
    directory, result = loop(config, model, lambda *args, **kwargs: {"compiled": True, "correctness": True, "runtime": 1})
    assert calls == [6, 1]
    assert result["status"] == "success" and result["exit_status"] == "ResourceBudgetLimit"
    assert result["budget"]["model_calls"] == 2 and result["budget"]["total_tokens"] == 9
    assert budgets.validate_snapshot(result["budget"]["resource_budgets"]["total_tokens"], 9, "tokens")
    assert (directory / "runs/agent" / kernel_name(1, 90)).is_file()


@pytest.mark.parametrize("missing", [False, True])
def test_output_budget_rejects_provider_overrun_or_missing_usage_before_evaluation(config, missing):
    config.agent_backend, config.model, config.agent_output_token_limit = "litellm", "fake/provider", 3
    config.generation_mode = "feedback"
    evaluations = []

    def model(*args, **kwargs):
        item = response(*args, **kwargs)
        item["output_tokens"] = None if missing else 4
        return item
    _, result = loop(config, model, lambda *a, **k: evaluations.append(a))
    assert not evaluations and result["status"] == "failed" and result["budget"]["model_calls"] == 1
    assert not result["budget"]["resource_budgets"]["output_tokens"]["compliant"]


def test_provider_exception_burns_inflight_tokens(config):
    config.agent_output_token_limit = 7
    config.agent_backend, config.model = "litellm", "fake/provider"
    def broken(*a, **k):
        raise TimeoutError("provider stopped returning usage")
    _, result = loop(config, broken)
    record = result["budget"]["resource_budgets"]["output_tokens"]
    assert record["charged"] == 7 and record["pending"] == {} and not record["compliant"]


def test_prompt_that_cannot_fit_never_dispatches_a_model_request(config, tmp_path):
    config.agent_token_limit, config.agent_backend = 3, "local-transformers"
    calls = []
    def model(*args, **kwargs):
        calls.append(args)
    model.count_tokens = lambda messages: 3
    controlled = KernelModel(config, tmp_path, lambda: 30, model)
    with pytest.raises(budgets.BudgetExhausted):
        controlled.query([])
    assert calls == [] and controlled.usage()["model_calls"] == 0


def test_profiling_is_not_free_and_retains_a_completed_candidate(config):
    config.agent_device_seconds, config.agent_profile = 6.2, True
    calls = []
    def worker(*args, lease, task, **kwargs):
        calls.append(task)
        receipt = {"protocol": budgets.LEASE_PROTOCOL, "nonce": lease["nonce"], "reserved_seconds": lease["reserved_seconds"],
                   "elapsed_seconds": 0.8, "method": "trusted_parent_monotonic_wall", "physical_preemption_certified": False}
        receipt["receipt_sha256"] = fingerprint(receipt)
        return {"compiled": True, "correctness": True, "runtime": 1, "metadata": {"device_lease_receipt": receipt}}
    _, result = loop(config, response, worker)
    assert calls == ["evaluate", "profile"] and result["selected"]["round"] == 1
    assert result["status"] == "success" and result["budget"]["profile_requests"] == 1
    assert result["budget"]["resource_budgets"]["device_lease"]["charged"] == pytest.approx(1.6)


def test_development_payload_carries_fractional_execution_window_and_nonce(config, monkeypatch):
    from kb_workflows import engine

    config.evaluation = "queue"
    directory = generation.prepare_job(config, 90, 0)
    child = RunConfig(**read_json(directory / "job_config.json"))
    output = child.run_dir / kernel_name(1, 90)
    output.parent.mkdir(parents=True)
    output.write_text("class ModelNew: pass")
    ledger = budgets.Ledger(5.25, "device_lease_seconds")
    lease = budgets.device_lease(ledger, 30, 30)
    payloads = []
    def evaluate(config, tasks):
        payloads.extend(payload for _, payload in tasks)
        yield tasks[0][0], {}
    monkeypatch.setattr(engine, "evaluate_tasks", evaluate)
    agent_loop.development_evaluate(child, 90, 1, lambda: 20, lease=lease)
    assert payloads[0]["eval_config"]["timeout"] == 0.25
    assert payloads[0]["eval_config"]["device_lease"] == lease
    assert payloads[0]["eval_config"]["use_hidden_tests"] is False
    assert payloads[0]["request_id"] == "kb-" + fingerprint(payloads[0])


def test_parent_detects_worker_overrun(monkeypatch):
    ledger = budgets.Ledger(6, "device_lease_seconds")
    lease = budgets.device_lease(ledger, 30, 30)
    times = iter([100, 107])
    monkeypatch.setattr(budgets.time, "monotonic", lambda: next(times))
    wrapped = budgets.record_device_lease(lambda payload: {"compiled": True, "correctness": True})
    result = wrapped({"eval_config": {"timeout": 1, "device_lease": lease}})
    assert not result["correctness"] and not budgets.settle_device_lease(ledger, lease, result)
    assert ledger.spent == 7 and not ledger.snapshot()["compliant"]


def test_local_checkpoint_path_resolves_relative_to_toml(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    path = tmp_path / "run.toml"
    path.write_text('[run]\ngeneration="local"\nagent_backend="local-transformers"\nmodel="checkpoint"\n')
    assert parse_config(["--config", str(path)]).model == str(checkpoint)


def test_litellm_cap_is_forwarded_without_silent_parameter_drop(config, monkeypatch):
    import litellm

    calls = []
    monkeypatch.setattr(litellm, "get_supported_openai_params", lambda **kwargs: ["max_completion_tokens"])
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: calls.append(kwargs) or SimpleNamespace(
        model="fake", choices=[SimpleNamespace(message=SimpleNamespace(content="source"))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)))
    monkeypatch.setattr(litellm, "completion_cost", lambda **kwargs: 0.1)
    litellm_response([], config, None, 1, max_output_tokens=17)
    assert calls[0]["max_completion_tokens"] == 17 and calls[0]["drop_params"] is False and calls[0]["n"] == 1
    monkeypatch.setattr(litellm, "get_supported_openai_params", lambda **kwargs: [])
    with pytest.raises(ValueError, match="cannot enforce"):
        litellm_response([], config, None, 1, max_output_tokens=17)
    assert len(calls) == 1


def test_parent_timing_overwrites_child_receipt_and_includes_cleanup(monkeypatch):
    ledger = budgets.Ledger(8, "device_lease_seconds")
    lease = budgets.device_lease(ledger, 10, 30)
    assert lease["execution_timeout_seconds"] == 3
    times = iter([100, 101.25])
    monkeypatch.setattr(budgets.time, "monotonic", lambda: next(times))
    @budgets.record_device_lease
    def worker(payload):
        return {"compiled": True, "correctness": True, "metadata": {"device_lease_receipt": {"elapsed_seconds": 0}}}
    result = worker({"eval_config": {"timeout": 3, "device_lease": lease}})
    assert result["metadata"]["device_lease_receipt"]["elapsed_seconds"] == 1.25
    assert budgets.settle_device_lease(ledger, lease, result) and ledger.remaining == 6.75
    other = budgets.device_lease(ledger, 10, 30)
    assert not budgets.settle_device_lease(ledger, other, result)
    assert ledger.remaining == 0


def test_device_budget_exhaustion_preserves_best_and_never_dispatches_extra_work(config):
    config.agent_device_seconds = 6.2
    calls = []
    def worker(*args, lease, **kwargs):
        calls.append(lease)
        receipt = {"protocol": budgets.LEASE_PROTOCOL, "nonce": lease["nonce"], "reserved_seconds": lease["reserved_seconds"],
                   "elapsed_seconds": 0.8, "method": "trusted_parent_monotonic_wall", "physical_preemption_certified": False}
        receipt["receipt_sha256"] = fingerprint(receipt)
        return {"compiled": True, "correctness": True, "runtime": 1 / len(calls),
                "metadata": {"device_lease_receipt": receipt}}
    _, result = loop(config, response, worker)
    assert len(calls) == 2 and result["selected"]["round"] == 2
    assert result["exit_status"] == "ResourceBudgetLimit" and result["status"] == "success"
    record = result["budget"]["resource_budgets"]["device_lease"]
    assert budgets.validate_snapshot(record, 6.2, "device_lease_seconds")
    assert result["budget"]["development_evaluations"] == 2


def test_device_usage_missing_stops_without_accepting_candidate(config):
    config.agent_device_seconds = 10
    _, result = loop(config, response, lambda *a, **k: {"compiled": True, "correctness": True, "runtime": 1})
    assert result["status"] == "failed" and result["budget"]["development_evaluations"] == 1
    assert not result["budget"]["resource_budgets"]["device_lease"]["compliant"]


def test_offline_adapter_counts_exact_ids_without_loading_any_checkpoint(config):
    import torch

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}
        def decode(self, ids, **kwargs):
            return str(ids)
    class FakeModel:
        def generate(self, input_ids, **kwargs):
            assert kwargs["num_beams"] == kwargs["num_return_sequences"] == 1
            assert kwargs["max_new_tokens"] == 2
            return torch.cat([input_ids, torch.tensor([[4, 5]])], dim=1)
    adapter = LocalTransformersModel(config, tokenizer=Tokenizer(), model=FakeModel())
    messages = [{"role": "user", "content": "fake"}]
    assert adapter.count_tokens(messages) == 3
    result = adapter(messages, config, None, 1, max_output_tokens=2)
    assert result["total_tokens"] == 5 and result["output_tokens"] == 2 and not result["provider_error"]


def test_checkpoint_asset_changes_invalidate_generation_resume(config):
    config.agent_backend, config.model = "local-transformers", str(config.run_dir.parent)
    path = config.run_dir.parent / "vocab.txt"
    path.parent.mkdir(parents=True)
    path.write_text("before")
    before = generation.generation_signature(config, 90, 0)
    path.write_text("after!")
    assert generation.generation_signature(config, 90, 0) != before
    assert checkpoint_digest(config.model) == checkpoint_digest(config.model)


@pytest.mark.parametrize("options", [
    {"agent_token_limit": 10}, {"agent_output_token_limit": 10},
    {"agent_device_seconds": 10, "generation": "harbor"}, {"agent_device_seconds": float("nan")},
    {"agent_output_token_limit": True}, {"agent_backend": "local-transformers", "model": ""},
])
def test_unsupported_or_invalid_budget_configuration_is_rejected(config, options):
    with pytest.raises(ValueError):
        RunConfig(**{**config.to_dict(), **options}).validate()


def test_budget_grid_and_cli_share_caps(config, tmp_path):
    spec = {"base_config": {**config.to_dict(), "agent_backend": "litellm"}, "models": ["fake/provider"],
            "methods": ["one-shot", "feedback"], "sessions": 1, "vary": "output_tokens", "budgets": [10, 20]}
    plan = make_plan(spec, directory=tmp_path / "plan")
    assert [row["config"]["agent_output_token_limit"] for row in plan["jobs"]] == [10, 20, 10, 20]
    assert not plan["experiments_started"]
    parsed = parse_config(["--generation", "local", "--agent-backend", "litellm", "--model", "fake/provider",
                           "--agent-output-token-limit", "99", "--agent-device-seconds", "8.5", "--agent-gpu-busy-seconds", "7.5"])
    assert parsed.agent_output_token_limit == 99 and parsed.agent_device_seconds == 8.5 and parsed.agent_gpu_busy_seconds == 7.5
    spec.update(vary="gpu_busy_seconds", budgets=[6.5, 8.5])
    gpu_plan = make_plan(spec, directory=tmp_path / "gpu-plan")
    assert [row["config"]["agent_gpu_busy_seconds"] for row in gpu_plan["jobs"]] == [6.5, 8.5, 6.5, 8.5]
    for value in (-1, True, float("inf")):
        with pytest.raises(ValueError):
            RunConfig(**{**config.to_dict(), "agent_gpu_busy_seconds": value}).validate()


def test_capped_attempt_cannot_be_restarted_with_fresh_budget(config, monkeypatch):
    from kb_workflows.common import atomic_json

    config.agent_output_token_limit, config.agent_backend, config.model = 10, "litellm", "fake/provider"
    config.run_dir.mkdir(parents=True)
    atomic_json(config.run_dir / "generation_attempts.json", {"slots": {"90_0": [{"status": "started"}]}})
    monkeypatch.setattr(generation, "prepare_job", lambda *args: pytest.fail("Must reject before modifying prior evidence"))
    with pytest.raises(ValueError, match="already has an attempt"):
        generation.generate_run(config)


def test_interactive_sessions_reject_unobservable_caps(config):
    from kb_workflows.agents.agent_task import require_interactive_budget

    config.agent_device_seconds = 10
    with pytest.raises(ValueError, match="unobserved"):
        require_interactive_budget(config)


def test_legacy_opt_out_cannot_hide_missing_resource_budget_evidence(config):
    from kb_workflows.common import apply_score_gate, atomic_json

    config.agent_device_seconds, config.score_gate_required = 10, False
    atomic_json(config.run_dir / "run_config.json", config.to_dict())
    raw = {"90_0": {"compiled": True, "correctness": True, "metadata": {"evaluation_config": {"score_gate_required": False}}}}
    assert not apply_score_gate(raw, config.run_dir)["90_0"]["correctness"]


def test_queue_budget_requires_capability_and_cancels_a_lost_submission(config, monkeypatch):
    from kb_workflows.runtime import transports

    config.evaluation, config.agent_device_seconds = "queue", 10
    config.validate()
    ledger = budgets.Ledger(10, "device_lease_seconds")
    lease = budgets.device_lease(ledger, 3, 10)
    payload = {"request_id": "owned", "eval_config": {"timeout": 3, "device_lease": lease}}
    calls = []
    supported = False

    def http(method, url, value=None, timeout=30, *, deadline=None):
        calls.append((method, url))
        if url.endswith("/health"):
            return {"protocol": transports.QUEUE_PROTOCOL if supported else "old-worker", "status": "ok", "device_backend": "cuda"}
        if url.endswith("/cancel"):
            result = budgets.stamp_device_lease({"correctness": False}, lease, 0.25)
            return {"request_id": "owned", "protocol": transports.QUEUE_PROTOCOL,
                    "device_backend": "cuda",
                    "request_sha256": fingerprint(payload), "status": "cancelled", "result": result}
        raise TimeoutError("Submission response lost after server admission")

    monkeypatch.setattr(transports, "http_json", http)
    result = transports.queue_evaluate(payload, config)
    assert not result["correctness"] and len(calls) == 1  # Old server never receives work.
    supported = True
    result = transports.queue_evaluate(payload, config)
    assert result["metadata"]["queue_cancellation_confirmed"] is True
    assert calls[-1][0] == "POST" and calls[-1][1].endswith("/owned/cancel")
    assert budgets.settle_device_lease(ledger, lease, result) and ledger.spent == 0.25

    lease = budgets.device_lease(ledger, 3, 10)
    payload["eval_config"]["device_lease"] = lease

    @budgets.record_device_lease
    def broken_cleanup(payload):
        raise TimeoutError("Worker did not exit after SIGKILL")

    result = budgets.record_device_lease(broken_cleanup)(payload)
    assert result["metadata"]["resource_usage_unknown"] and "device_lease_receipt" not in result["metadata"]
    assert not budgets.settle_device_lease(ledger, lease, result) and ledger.spent == 8.25


def test_modal_budget_runs_the_native_loop_with_remote_parent_receipts(config, monkeypatch):
    import sys
    from contextlib import nullcontext

    from kb_workflows.runtime import modal_backend, transports

    start_session = modal_backend.modal_session
    config.evaluation, config.agent_device_seconds = "modal", 6.2
    config.validate()
    requests, options, executed = [], [], []
    admitted = {}

    def put(nonce, signature, *, skip_if_exists):
        assert skip_if_exists
        if nonce in admitted:
            return False
        admitted[nonce] = signature
        return True

    leases = SimpleNamespace(put=put)

    @budgets.record_device_lease
    def remote_parent(payload):
        executed.append(fingerprint(payload))
        # Simulate the isolated candidate returning an invented receipt. The
        # existing remote parent must replace it before cloud serialization.
        return {"compiled": True, "correctness": True, "runtime": 1,
                "metadata": {"device_lease_receipt": {"elapsed_seconds": 0}}}

    class Function:
        def with_options(self, **kwargs):
            options.append(kwargs)
            return self

        async def submit(self, payload, store):
            requests.append(payload)

            async def get(*, timeout):
                assert timeout > 0
                return modal_backend._remote_evaluate(payload, store)

            async def cancel(**kwargs):
                pytest.fail("Successful remote execution must not be cancelled")

            return SimpleNamespace(get=SimpleNamespace(aio=get), cancel=SimpleNamespace(aio=cancel))

        @property
        def spawn(self):
            return SimpleNamespace(aio=self.submit)

    monkeypatch.setattr(transports, "local_evaluate", remote_parent)
    monkeypatch.setattr(modal_backend, "_SESSION", (None, Function(), None, leases))
    monkeypatch.setattr(modal_backend, "modal_session", lambda config: nullcontext())
    _, result = loop(config, response)
    assert result["status"] == "success" and result["selected"]["round"] == 1
    assert len(requests) == config.optimization_rounds == len(options)
    assert all(row["retries"] == 0 and 0 < row["timeout"] <= 7 for row in options)
    record = result["budget"]["resource_budgets"]["device_lease"]
    assert 0 < record["charged"] < 6.2 and budgets.validate_snapshot(record, 6.2, "device_lease_seconds")
    assert len({row["eval_config"]["device_lease"]["nonce"] for row in requests}) == len(requests)
    assert admitted == {row["eval_config"]["device_lease"]["nonce"]: fingerprint(row) for row in requests}
    replay = modal_backend._remote_evaluate(requests[0], leases)
    assert replay["result"]["metadata"]["resource_usage_unknown"]
    assert "device_lease_receipt" not in replay["result"]["metadata"]
    assert len(executed) == len(requests)
    for fields in ({"worker_python": "/client/python"}, {"eval_config": {"device_backend": "neuron"}}):
        with pytest.raises(ValueError, match="server-owned interpreter"):
            modal_backend._remote_evaluate({**requests[0], **fields})
    # Exercise the actual session wiring with an inert SDK, including nested
    # restoration. Nothing may build an image, contact Modal or allocate a GPU.
    function_options = {}
    def decorate(**kwargs):
        function_options.update(kwargs)
        return lambda handler: Function()
    app = SimpleNamespace(function=decorate, run=lambda: nullcontext())
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(App=lambda name: app, enable_output=lambda: nullcontext(),
                        Dict=SimpleNamespace(ephemeral=lambda: nullcontext(leases))))
    monkeypatch.setattr(modal_backend, "make_image", lambda **kwargs: "simulated-image")
    previous = modal_backend._SESSION
    with start_session(config):
        assert modal_backend._SESSION[0] is app and modal_backend._SESSION[3] is leases
    assert modal_backend._SESSION is previous and function_options["retries"] == 0
    config.generation = "modal"
    with start_session(config):
        assert modal_backend._SESSION[3] is None  # Sandbox-local leases need no extra cloud Dict.
    assert modal_backend._SESSION is previous


def test_modal_unknown_submission_binding_and_cleanup_burn_the_reservation(config, monkeypatch):
    import asyncio

    from kb_workflows.runtime import modal_backend

    monkeypatch.setattr(modal_backend, "CLEANUP_SECONDS", 0.02)
    config.queue_wait_timeout = 0.02
    for mode in ("lost_submit", "blocked_submit", "blocked_cleanup", "wrong_request", "missing_receipt"):
        ledger = budgets.Ledger(7, "device_lease_seconds")
        lease = budgets.device_lease(ledger, 1, 7)
        payload = {"eval_config": {"timeout": 1, "device_lease": lease}}
        cancellations = []

        async def submit(payload, store):
            if mode == "lost_submit":
                raise ConnectionError("Submission reply lost after possible admission")
            if mode == "blocked_submit":
                await asyncio.sleep(3600)

            async def get(*, timeout):
                if mode == "blocked_cleanup":
                    await asyncio.sleep(3600)
                return {"protocol": modal_backend.LEASE_PROTOCOL, "nonce": lease["nonce"],
                        "request_sha256": "another-task" if mode == "wrong_request" else fingerprint(payload),
                        "device_backend": "cuda", "result": {"compiled": True, "correctness": True}}

            async def cancel(*, terminate_containers):
                cancellations.append(terminate_containers)
                if mode == "blocked_cleanup":
                    await asyncio.sleep(3600)
                raise OSError("Cancellation reply lost")

            return SimpleNamespace(get=SimpleNamespace(aio=get), cancel=SimpleNamespace(aio=cancel))

        function = SimpleNamespace(spawn=SimpleNamespace(aio=submit))
        function.with_options = lambda **kwargs: function
        monkeypatch.setattr(modal_backend, "_SESSION", (None, function, None, object()))
        result = modal_backend.modal_evaluate(payload, config)
        assert not budgets.settle_device_lease(ledger, lease, result) and ledger.spent == lease["reserved_seconds"]
        if mode != "missing_receipt":
            assert not result["correctness"] and result["metadata"]["resource_usage_unknown"]
        assert cancellations == ([True] if mode in {"blocked_cleanup", "wrong_request"} else [])


@pytest.fixture
def modal_generator(config, tmp_path, monkeypatch):
    """Local files and real controller accounting behind an inert cloud API."""
    import asyncio
    import sys
    import time

    from kb_workflows.common import atomic_json
    from kb_workflows.runtime import modal_backend
    from scripts import run_agent as entrypoint

    config.generation, config.evaluation, config.stage = "modal", "queue", "generate"
    config.hardware, config.gpu, config.gpu_arch = "H100", "H100", "Hopper"
    config.agent_backend, config.model = "litellm", "fake/provider"
    config.agent_output_token_limit, config.agent_max_output_tokens, config.agent_device_seconds = 6, 3, 6.2
    config.validate()
    state = SimpleNamespace(mode="success", boxes=[], controller_calls=0, evaluations=0, real_time=time)

    def api(function):
        async def call(*args, **kwargs):
            value = function(*args, **kwargs)
            return await value if hasattr(value, "__await__") else value
        return SimpleNamespace(aio=call)

    @budgets.record_device_lease
    def worker(payload):
        state.evaluations += 1
        return {"compiled": True, "correctness": True, "runtime": 1,
                "metadata": {"device_lease_receipt": {"elapsed_seconds": 0}}}

    def evaluate(child, pid, index, remaining, task="evaluate", *, lease=None):
        assert child.evaluation == "local" and lease is not None
        return worker({"eval_config": {"timeout": lease["execution_timeout_seconds"], "device_lease": lease}})

    def controlled_run(directory, child, pid):
        state.controller_calls += 1
        child = RunConfig(**{**child.to_dict(), "runs_dir": str(directory / "runs")}).validate()
        return agent_loop.run_loop(directory, child, pid, response_fn=response, evaluator=evaluate)
    monkeypatch.setattr(entrypoint, "run_agent", controlled_run)

    class Box:
        def __init__(self):
            self.root = tmp_path / f"remote-{len(state.boxes)}"
            self.root.mkdir()
            self.terminated = 0
            def path(name):
                assert name.startswith("/work/")
                return self.root / name.removeprefix("/work/")
            def write(data, name):
                path(name).parent.mkdir(parents=True, exist_ok=True)
                path(name).write_text(data)
            self.filesystem = SimpleNamespace(
                make_directory=api(lambda name: path(name).mkdir(parents=True, exist_ok=True)),
                write_text=api(write), stat=api(lambda name: SimpleNamespace(size=path(name).stat().st_size)),
                read_bytes=api(lambda name: path(name).read_bytes()))
            self.exec, self.terminate = api(self.execute), api(self.stop)

        def execute(self, *args, **kwargs):
            assert args[:2] == ("bash", "-c") and "--admission" in args[2] and kwargs["timeout"] > 0
            job = read_json(self.root / "job_config.json")
            self.job, self.admission = job, read_json(self.root / "generation_admission.json")
            assert entrypoint.main(["--config", str(self.root / "job_config.json"), "--directory", str(self.root),
                                    "--admission", str(self.root / "generation_admission.json")]) == 0
            result = read_json(self.root / "agent_result.json")
            if state.mode == "binding":
                result["modal_generation_receipt"]["binding"]["nonce"] = "0" * 32
            elif state.mode == "missing":
                result.pop("modal_generation_receipt")
            elif state.mode == "path":
                receipt = result["modal_generation_receipt"]
                receipt["artifacts"]["../outside"] = {"bytes": 0, "sha256": "0" * 64}
                receipt["receipt_sha256"] = fingerprint({k: v for k, v in receipt.items() if k != "receipt_sha256"})
            atomic_json(self.root / "agent_result.json", result)
            (self.root / "sandbox.log").write_text("Explicit CPU simulation; no Modal or device execution")
            if state.mode == "lost":
                raise OSError("exec response lost after the single controller completed")
            if state.mode == "deadline":
                expired = time.monotonic() + config.agent_timeout + 121
                monkeypatch.setattr(modal_backend, "time", SimpleNamespace(monotonic=lambda: expired))
            async def wait():
                if state.mode == "deadline":
                    await asyncio.Future()
                return 0
            return SimpleNamespace(wait=api(wait))

        def stop(self, *, wait):
            assert wait is True
            self.terminated += 1
            if state.mode == "cleanup":
                raise OSError("termination acknowledgement lost")
            return 0

    def create(**kwargs):
        assert kwargs["timeout"] == config.agent_timeout + 120 and kwargs["gpu"] == "H100"
        box = Box()
        state.boxes.append(box)
        return box
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Sandbox=SimpleNamespace(create=api(create))))
    monkeypatch.setattr(modal_backend, "_SESSION", (None, None, "inert-image", None))
    monkeypatch.setattr(modal_backend, "sandbox_secrets", lambda config: [])
    return config, state


def test_modal_native_generation_reuses_token_and_device_ledgers_with_single_admission(modal_generator):
    from kb_workflows.runtime import modal_backend

    config, state = modal_generator
    assert generation.generate_run(config)
    summary = read_json(config.run_dir / "generation_summary.json")["90_0"]
    assert summary["status"] == "success" and state.controller_calls == 1 and state.evaluations == 3
    assert summary["budget"]["output_tokens"] == 6 and summary["budget"]["model_calls"] == 3
    assert summary["budget"]["protocol"]["development_verifier"] == "modal_sandbox_local"
    assert state.boxes[0].job["evaluation"] == "local" and config.evaluation == "queue"
    assert state.boxes[0].terminated == 1 and budgets.score_issue(config, 90, 0) is None
    directory = config.run_dir / "work/p90_s0"
    assert (directory / "model_calls/001.json").is_file() and (directory / "rounds/003/candidate.py").is_file()
    modal_backend.validate_generation(summary, directory)
    box = state.boxes[0]
    with pytest.raises(FileExistsError):
        modal_backend.admitted_generation(box.root, box.job, box.admission, lambda: pytest.fail("duplicate controller"))
    source = directory / "rounds/001/candidate.py"
    source.write_text("changed after transfer")
    assert budgets.score_issue(config, 90, 0) == "resource_budget_modal_generation_missing_or_invalid"
    config.resume = True
    generation.generate_run(config)
    assert state.controller_calls == 1  # Resume never buys another cloud attempt.


def test_modal_native_generation_lost_results_and_cleanup_cannot_reset_caps(modal_generator, monkeypatch):
    from kb_workflows.runtime import modal_backend

    config, state = modal_generator
    for mode in ("binding", "missing", "path", "lost", "deadline", "cleanup"):
        state.mode, config.run_name = mode, "modal_" + mode
        monkeypatch.setattr(modal_backend, "time", state.real_time)
        before = state.controller_calls
        assert not generation.generate_run(config)
        summary = read_json(config.run_dir / "generation_summary.json")["90_0"]
        assert summary["status"] == "error" and summary["budget"] is None and summary["resource_receipt_incomplete"]
        assert state.controller_calls == before + 1 and state.boxes[-1].terminated == 1
        assert budgets.score_issue(config, 90, 0) is not None
        assert summary["modal_cleanup"]["termination_wait_completed"] is (mode != "cleanup")
        with pytest.raises(ValueError, match="already has an attempt"):
            generation.generate_run(config)
        assert state.controller_calls == before + 1


def test_gpu_quota_admits_before_work_replays_artifacts_and_keeps_measurements_unknown(config):
    from kb_workflows.common import atomic_json

    config.agent_device_seconds, config.agent_gpu_busy_seconds = 6.2, 6.5
    reservations = []
    def evaluate(*args, lease, **kwargs):
        budgets.validate_device_lease({"eval_config": {"timeout": lease["execution_timeout_seconds"], "device_lease": lease}})
        reservations.append(lease)
        return budgets.stamp_device_lease({"compiled": True, "correctness": True, "runtime": 1}, lease, 0.6)
    directory, result = loop(config, response, evaluate)
    receipt = result["budget"]
    gpu = receipt["resource_budgets"]["gpu_busy_quota"]
    assert len(reservations) == 2 and receipt["development_evaluations"] == 2
    assert reservations[0]["execution_timeout_seconds"] == pytest.approx(1.2)
    assert reservations[1]["execution_timeout_seconds"] == pytest.approx(0.6)
    assert result["status"] == "success" and result["exit_status"] == "ResourceBudgetLimit"
    assert gpu["charged"] == pytest.approx(1.2) and budgets.validate_snapshot(gpu, 6.5, "gpu_busy_upper_bound_seconds")
    assert receipt["gpu_execution_seconds"] is None and receipt["gpu_busy_charge_upper_bound_seconds"] == pytest.approx(1.2)
    assert all(row["basis"]["method"] == "confirmed_worker_wall_upper_bound" for row in gpu["entries"])
    assert budgets.validate_gpu_artifacts(config, 90, 0, gpu)
    path = directory / "rounds/001/standard.json"
    saved = read_json(path)
    saved["metadata"]["device_lease_receipt"]["elapsed_seconds"] = 0.1
    atomic_json(path, saved)
    assert not budgets.validate_gpu_artifacts(config, 90, 0, gpu)
    # GPU quota can be used without charging the separate worker-wall ledger.
    only_gpu = budgets.Ledger(6, "gpu_busy_upper_bound_seconds")
    lease = budgets.device_lease(None, 30, 30, gpu_ledger=only_gpu)
    assert lease["gpu_busy"]["nonce"] == lease["nonce"]
    wrong = budgets.stamp_device_lease({}, lease, 0.5)
    wrong["metadata"]["device_lease_receipt"].pop("lease_sha256")
    assert not budgets.settle_gpu_lease(only_gpu, lease, wrong, index=1, task="evaluate")
    assert only_gpu.spent == 6 and not only_gpu.snapshot()["compliant"]
