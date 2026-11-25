"""Shared lanes send the best-ranked ready request and adapt to rate limits.

A study's contexts collect at once through one lane per model, with no
context holding slots; admission order never reaches a capture, and an
interrupted study resumes from its stored calls.
"""

import asyncio
import json
from typing import cast

import litellm
import pytest

from catmodel.collection import contextual_collection
from catmodel.collection.collection_scheduler import (
    BACKOFF,
    CollectionScheduler,
    ModelLane,
)
from catmodel.collection.contextual_collection import collect_study
from catmodel.config import ExecutionSettings
from catmodel.contracts.contextual_risk import (
    ContextualRiskArtifact,
    ContextualRiskInstrument,
)
from catmodel.elicitation.contextual_risk import contextual_prompts
from catmodel.llm import litellm_provider
from catmodel.llm.litellm_provider import LiteLLMProvider
from catmodel.llm.models import ProviderName
from catmodel.llm.rate_limits import report_rate_limited
from catmodel.llm.transport import LLMProvider, ProviderError, ProviderResponse
from catmodel.setup.contextual_inputs import (
    read_contextual_capture_files,
    validate_contextual_capture,
)
from test_support.collection_trace import (
    CollectionTrace,
    LatencyProvider,
    ModelLatency,
    StudyResponder,
    TriggerRates,
    request_kind,
    run_warped,
    traced_collection,
)
from test_support.contextual_risk import hierarchy_instrument

SCENARIOS = ("output_reliance_legal", "fraud_organizations")
EVERY_TURN = TriggerRates(
    direct_retry=1.0,
    chain_guardrail_retry=1.0,
    crosscheck=1.0,
    crosscheck_retry=1.0,
    parent_gap=1.0,
)
MODELS = ("fixture-a", "fixture-b")


# Lanes


def test_a_free_slot_goes_to_the_best_ranked_ready_request():
    async def scenario():
        lane = ModelLane(1)
        await lane.acquire((0,))
        order = []

        async def wait(name, rank):
            await lane.acquire(rank)
            order.append(name)

        waiting = [
            asyncio.ensure_future(wait("new", (0, 5))),
            asyncio.ensure_future(wait("round", (-7, 9))),
            asyncio.ensure_future(wait("older", (0, 1))),
            asyncio.ensure_future(wait("retry", (-1, 3))),
        ]
        await asyncio.sleep(0)
        assert lane.in_flight == 1 and order == []
        for _ in waiting:
            lane.release()
            await asyncio.sleep(0)
        await asyncio.gather(*waiting)
        return order

    assert asyncio.run(scenario()) == ["round", "retry", "older", "new"]


def test_the_longest_remaining_chain_ranks_first():
    rank = contextual_collection.request_rank
    question, retry, last_turn = rank(10, 9, 9), rank(9, 0, 0), rank(1, 0, 0)
    assert question < retry < last_turn
    # Among equal chains, earlier collections and then groups go first.
    assert rank(5, 0, 3) < rank(5, 1, 0) < rank(5, 1, 1)


def test_a_provider_failure_stops_every_lane():
    async def scenario():
        scheduler = CollectionScheduler({"fixture-a": 2, "fixture-b": 2})
        first, second = scheduler.lanes["fixture-a"], scheduler.lanes["fixture-b"]
        await first.acquire((0,))
        first.failed(ConnectionError("down"))
        assert isinstance(scheduler.failure, ConnectionError)
        waiting = [
            asyncio.ensure_future(lane.acquire((0,))) for lane in (first, second)
        ]
        first.release()
        await asyncio.sleep(0)
        assert not any(row.done() for row in waiting)
        assert (first.in_flight, second.in_flight) == (0, 0)
        for row in waiting:
            row.cancel()
        await asyncio.gather(*waiting, return_exceptions=True)

    asyncio.run(scenario())


def test_a_cancelled_request_never_keeps_a_slot():
    async def scenario():
        lane = ModelLane(1)
        await lane.acquire((0,))
        gone = asyncio.ensure_future(lane.acquire((-1,)))
        kept = asyncio.ensure_future(lane.acquire((0,)))
        await asyncio.sleep(0)
        gone.cancel()
        await asyncio.sleep(0)
        lane.release()
        await asyncio.wait_for(kept, 1)
        assert lane.in_flight == 1
        # Admitted in the same step as it is cancelled: the slot moves on.
        late = asyncio.ensure_future(lane.acquire((0,)))
        after = asyncio.ensure_future(lane.acquire((0,)))
        await asyncio.sleep(0)
        lane.release()
        late.cancel()
        with pytest.raises(asyncio.CancelledError):
            await late
        await asyncio.wait_for(after, 1)
        assert lane.in_flight == 1

    asyncio.run(scenario())


def test_a_rate_limit_lowers_the_limit_once_per_window_and_success_regrows_it():
    async def scenario():
        lane = ModelLane(10)
        admitted = [await lane.acquire((0,)) for _ in range(10)]
        assert admitted == [0] * 10
        # Refusals of requests sent before the limit fell count once.
        lane.rate_limited(0)
        lane.rate_limited(0)
        assert lane.limit == pytest.approx(10 * BACKOFF)
        assert (lane.rate_limits, lane.decreases) == (2, 1)
        # Nothing new is sent past the lowered limit.
        blocked = asyncio.ensure_future(lane.acquire((0,)))
        behind = asyncio.ensure_future(lane.acquire((0,)))
        lane.release()
        await asyncio.sleep(0)
        assert not blocked.done() and lane.in_flight == 9
        lane.release()
        await asyncio.sleep(0)
        assert blocked.done() and not behind.done() and lane.in_flight == 9
        # A refusal of a request sent since the decrease lowers it again.
        lane.rate_limited(blocked.result())
        assert lane.limit == pytest.approx(10 * BACKOFF**2)
        # Successes with the lane at least half busy add one slot each.
        lane.succeeded()
        assert lane.limit == pytest.approx(10 * BACKOFF**2 + 1)
        for _ in range(5):
            lane.succeeded()
        assert lane.limit == lane.ceiling == 10
        await asyncio.sleep(0)
        assert behind.done() and lane.in_flight == 10
        # A lane barely in use does not grow.
        lane.rate_limited(lane.decreases)
        for _ in range(8):
            lane.release()
        lane.succeeded()
        assert lane.limit == pytest.approx(10 * BACKOFF)
        for _ in range(100):
            lane.rate_limited(lane.decreases)
        assert lane.limit == 1 and lane.slots == 1

    asyncio.run(scenario())


class _Limited:
    model_id = "fixture-a"
    provider_name = "openai"
    reasoning_effort = "max"

    def __init__(self) -> None:
        self.seen = []

    async def complete(self, **request) -> ProviderResponse:
        self.seen.append(request)
        report_rate_limited()
        return ProviderResponse(text="{}", model_id=self.model_id)


def test_an_admitted_provider_keeps_its_identity_and_hears_rate_limits():
    inner = _Limited()
    # At two slots, one request in flight is half the lane: a clean success
    # would grow it, so only the reported rate limit keeps it lowered.
    scheduler = CollectionScheduler({"fixture-a": 2})
    admitted = scheduler.provider(cast(LLMProvider, inner), (0,))
    assert (admitted.model_id, admitted.provider_name, admitted.reasoning_effort) == (
        "fixture-a",
        "openai",
        "max",
    )
    request = {
        "system_prompt": "s",
        "user_prompt": "u",
        "max_tokens": 1,
        "request_timeout_seconds": 1.0,
    }
    response = asyncio.run(admitted.complete(**request))
    lane = scheduler.lanes["fixture-a"]
    assert response.text == "{}" and inner.seen == [request]
    assert lane.rate_limits == 1 and lane.limit == 2 * BACKOFF
    assert lane.in_flight == 0
    # Outside an admitted request a report reaches no one.
    report_rate_limited()
    assert lane.rate_limits == 1

    class _Failing(_Limited):
        def __init__(self, error: Exception) -> None:
            super().__init__()
            self.error = error

        async def complete(self, **request) -> ProviderResponse:
            raise self.error

    # A transient failure, once the transport's retries ran out, stops no lane.
    for transient in (
        ConnectionError("down"),
        ProviderError("submission failed 5 times: 502", transient=True),
    ):
        failing = scheduler.provider(cast(LLMProvider, _Failing(transient)), (0,))
        with pytest.raises(type(transient)):
            asyncio.run(failing.complete(**request))
        assert lane.in_flight == 0 and scheduler.failure is None
    # A systematic one, such as a 401, stops every lane.
    unauthorized = ProviderError("AuthenticationError: 401 invalid key")
    failing = scheduler.provider(cast(LLMProvider, _Failing(unauthorized)), (0,))
    with pytest.raises(ProviderError):
        asyncio.run(failing.complete(**request))
    assert lane.in_flight == 0 and lane.limit == 2 * BACKOFF
    assert scheduler.failure is unauthorized


# Study collection


def _instrument(context_id: str) -> ContextualRiskInstrument:
    payload = hierarchy_instrument(SCENARIOS).model_dump(mode="json")
    payload["context_id"] = context_id
    # A question names no context: each context's lab reads differently.
    payload["lab"]["display_name"] = f"Synthetic lab {context_id}"
    payload["respondents"] = [
        {"respondent_id": f"fixture-{index}", "model_id": model, "sample_index": 0}
        for index, model in enumerate(MODELS)
    ]
    return ContextualRiskInstrument.model_validate(payload)


CONTEXTS = tuple(_instrument(name) for name in ("lab-a", "lab-b", "lab-c"))
LAB_B = "Lab: Synthetic lab lab-b."
"""Text only the lab-b context's questions contain."""


def _providers(
    trace, *, seed=0, slots=3, tokens_per_minute=float("inf"), responder=None
):
    responder = responder or StudyResponder(CONTEXTS, EVERY_TURN)
    return [
        LatencyProvider(
            model_id=model,
            provider_name="openai",
            latency=ModelLatency(
                output_tokens=1000,
                tokens_per_second=50,
                tokens_per_minute=tokens_per_minute,
                retry_after_seconds=5.0,
            ),
            responder=responder,
            trace=trace,
            max_concurrent=slots,
            seed=seed + index,
        )
        for index, model in enumerate(MODELS)
    ]


CONFIG = ExecutionSettings(
    samples_per_question=1,
    max_output_tokens_by_model=dict.fromkeys(MODELS, 100),
    request_timeout_seconds=7200,
    concurrency_by_model=dict.fromkeys(MODELS, 3),
)


A_WEEK = 7 * 24 * 3600.0
"""A virtual-time bound on every study here, far beyond any of them: a
regression that leaves a study waiting forever fails with a timeout, in
moments of real time, instead of hanging the suite."""


def _study(
    root,
    *,
    seed=0,
    trace=None,
    timeout=A_WEEK,
    wrap=list,
    config=CONFIG,
    **provider_options,
):
    trace = trace or CollectionTrace()
    contexts = [(row, root / row.context_id) for row in CONTEXTS]
    run = asyncio.wait_for(
        collect_study(
            contexts,
            providers=cast(
                list[LLMProvider],
                wrap(_providers(trace, seed=seed, **provider_options)),
            ),
            config=config,
        ),
        timeout,
    )
    with traced_collection(trace):
        results, _ = run_warped(run)
    return results, trace


def _capture(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_collection_order_never_reaches_a_capture(tmp_path, monkeypatch):
    shared, _ = _study(tmp_path / "shared")
    assert all(not isinstance(row, Exception) for row in shared)
    # Other latencies and the opposite rank order send every request at
    # another time and in another order.
    monkeypatch.setattr(
        contextual_collection,
        "request_rank",
        lambda remaining, collection, group: (remaining, -collection, -group),
    )
    reordered, _ = _study(tmp_path / "reordered", seed=7)
    assert reordered == shared
    assert _capture(tmp_path / "reordered") == _capture(tmp_path / "shared")
    for instrument, artifact in zip(CONTEXTS, shared, strict=True):
        run = tmp_path / "shared" / instrument.context_id
        assert (run / "contextual-risk.json").exists()
        assert not isinstance(artifact, Exception)
        # Cases keep the instrument's question order, whatever finished first.
        assert [
            (case.node_id, case.anchor_score, case.respondent)
            for case in artifact.cases
        ] == [
            (prompt.node.node_id, prompt.anchor_score, prompt.respondent)
            for prompt in contextual_prompts(instrument)
        ]
        validate_contextual_capture(
            instrument, artifact, read_contextual_capture_files(run)
        )


def test_an_interrupted_study_resumes_to_the_same_capture(tmp_path):
    complete, _ = _study(tmp_path / "complete")
    with pytest.raises(TimeoutError):
        _study(tmp_path / "resumed", timeout=60.0)
    partial = _capture(tmp_path / "resumed")
    assert not any(name.endswith("contextual-risk.json") for name in partial)
    resumed, trace = _study(tmp_path / "resumed", seed=3)
    assert resumed == complete
    assert _capture(tmp_path / "resumed") == _capture(tmp_path / "complete")
    # Stored calls replay; only the rest were sent again.
    sent = sum(row.start is not None for row in trace.requests)
    assert 0 < sent < len(trace.requests)


def test_each_parent_is_asked_once_beside_its_group_s_leaves(tmp_path):
    results, trace = _study(tmp_path)
    assert {row.kind for row in trace.requests} == set(TURNS)
    first_answer: dict[tuple, float] = {}
    parents: dict[tuple, list] = {}
    for row in trace.requests:
        group = row.context_id, row.anchor_score, row.respondent
        assert row.done is not None
        if row.kind == "parent":
            parents.setdefault(group, []).append(row)
        elif row.kind == "direct":
            first_answer[group] = min(first_answer.get(group, row.done), row.done)
    # One question per parent in every group, although every parent's answer
    # lies ten times above what its leaves imply, and each is queued with the
    # group's leaf questions, before any of them is answered.
    for instrument in CONTEXTS:
        groups = [group for group in parents if group[0] == instrument.context_id]
        assert len(groups) == len(instrument.anchor_scores) * len(
            instrument.respondents
        )
        for group in groups:
            assert sorted(row.subject for row in parents[group]) == sorted(
                row.parent_id for row in instrument.parents
            )
            assert max(row.ready for row in parents[group]) < first_answer[group]
    for artifact in results:
        assert not isinstance(artifact, Exception)
        assert all(case.pool is not None for case in artifact.parent_cases)


TURNS = {
    "direct": 4,
    "chain": 2,
    "direct_retry": 3,
    "chain_guardrail_retry": 1,
    "crosscheck": 2,
    "crosscheck_retry": 1,
    "parent": 1,
}


def test_no_context_holds_slots_and_the_longest_chain_goes_first(tmp_path):
    _, trace = _study(tmp_path)
    rows = trace.requests
    # Every context starts at once; none waits for another to finish.
    first = {
        context: min(row.ready for row in rows if row.context_id == context)
        for context in {row.context_id for row in rows}
    }
    last_done = max(row.done for row in rows if row.done is not None)
    assert max(first.values()) - min(first.values()) < 0.01 * last_done

    def remaining(row):
        return TURNS[row.kind]

    # Across contexts, a request never waits behind one with a shorter chain
    # still to come that was itself waiting when the request reached the
    # lane (allowing a second of local work before a request reaches it).
    started = [row for row in rows if row.start is not None]
    passed = 0
    for longer in started:
        for shorter in started:
            if (
                shorter.model_id != longer.model_id
                or shorter.context_id == longer.context_id
                or remaining(shorter) >= remaining(longer)
                or not shorter.ready <= longer.ready < shorter.start - 1.0
            ):
                continue
            passed += 1
            assert longer.start <= shorter.start + 1e-9
    assert passed


def test_a_failed_context_leaves_the_others_to_finish(tmp_path):
    (tmp_path / "lab-b").mkdir()
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "lab-b" / "calls").symlink_to(tmp_path / "elsewhere")
    results, _ = _study(tmp_path)
    assert isinstance(results[1], ValueError)
    assert "symbolic links" in str(results[1])
    for index in (0, 2):
        assert not isinstance(results[index], Exception)
        assert (tmp_path / CONTEXTS[index].context_id / "contextual-risk.json").exists()


class _Down:
    """A provider that fails its requests with ``error`` once ``healthy`` of
    them have been sent, or only those whose prompt holds ``only``, at most
    ``failures`` of them."""

    def __init__(
        self,
        inner,
        *,
        error: Exception,
        healthy: int = 0,
        only: str = "",
        failures: float = float("inf"),
    ) -> None:
        self._inner = inner
        self._left = healthy
        self._error = error
        self._only = only
        self._failures = failures

    model_id = property(lambda self: self._inner.model_id)
    provider_name = property(lambda self: self._inner.provider_name)
    reasoning_effort = property(lambda self: self._inner.reasoning_effort)

    async def complete(self, **request) -> ProviderResponse:
        if self._only in request["user_prompt"]:
            self._left -= 1
            if self._left < 0 and self._failures > 0:
                self._failures -= 1
                raise self._error
        return await self._inner.complete(**request)


def _terminal(root):
    return [
        path
        for path in root.rglob("attempt-*.json")
        if json.loads(path.read_text())["raw_response"] is None
    ]


def _artifact(root, artifact):
    """``artifact`` with its frozen file, validated from its capture."""
    instrument = next(row for row in CONTEXTS if row.context_id == root.name)
    validate_contextual_capture(
        instrument, artifact, read_contextual_capture_files(root)
    )
    return artifact


def test_a_systematic_failure_stops_the_study_and_a_rerun_resends_it(tmp_path):
    clean, _ = _study(tmp_path / "clean")

    def unauthorized(providers):
        error = ProviderError("AuthenticationError: 401 invalid key")
        return [_Down(providers[0], error=error, healthy=40), *providers[1:]]

    with pytest.raises(ValueError, match="401 invalid key"):
        _study(tmp_path / "stopped", wrap=unauthorized)
    # Every lane stopped at the first failure: one call is frozen as failed,
    # however many more the bad key would have refused.
    (failed,) = _terminal(tmp_path / "stopped")
    holder = failed.relative_to(tmp_path / "stopped").parts[0]
    assert not list((tmp_path / "stopped").rglob("contextual-risk.json"))
    # A rerun in the same run directories sends the failed call again and
    # completes every context exactly as an uninterrupted study would; only
    # the holder's capture also keeps the failed attempt.
    resumed, _ = _study(tmp_path / "stopped", seed=5)
    assert resumed == clean
    for instrument, artifact in zip(CONTEXTS, resumed, strict=True):
        name = instrument.context_id
        _artifact(tmp_path / "stopped" / name, artifact)
        stopped = _capture(tmp_path / "stopped" / name)
        if name == holder:
            failed_name = str(failed.relative_to(tmp_path / "stopped" / name))
            assert failed_name in stopped and failed_name.endswith("attempt-001.json")
            assert set(stopped) - set(_capture(tmp_path / "clean" / name)) == {
                failed_name.replace("attempt-001", "attempt-002")
            }
        else:
            assert stopped == _capture(tmp_path / "clean" / name)


def test_a_transient_failure_fails_only_its_group(tmp_path):
    clean, _ = _study(tmp_path / "clean")

    def blip(providers):
        error = ProviderError("submission failed 5 times: 502", transient=True)
        failing = _Down(providers[0], error=error, only=LAB_B, failures=1)
        return [failing, *providers[1:]]

    results, _ = _study(tmp_path / "blip", wrap=blip)
    # The other contexts complete exactly as an uninterrupted study would.
    for index in (0, 2):
        assert results[index] == clean[index]
        name = CONTEXTS[index].context_id
        assert _capture(tmp_path / "blip" / name) == _capture(tmp_path / "clean" / name)
    # The holder is incomplete and names its failed call.
    (failed,) = _terminal(tmp_path / "blip")
    assert failed.relative_to(tmp_path / "blip").parts[0] == "lab-b"
    error = results[1]
    assert isinstance(error, ValueError)
    assert "1 calls failed at the provider" in str(error)
    assert failed.parent.name in str(error)
    assert not (tmp_path / "blip" / "lab-b" / "contextual-risk.json").exists()
    # Every other group of the holder finished: a rerun sends only the failed
    # call and the calls of its own group that depend on it.
    resumed, rerun = _study(tmp_path / "blip", seed=5)
    assert resumed == clean
    _artifact(tmp_path / "blip" / "lab-b", resumed[1])
    sent = [row for row in rerun.requests if row.start is not None]
    assert sent and {row.context_id for row in sent} == {"lab-b"}
    assert len({(row.anchor_score, row.respondent) for row in sent}) == 1


class _Rejected:
    """A provider whose direct question about one leaf of one context goes to
    the transport's own provider for *provider_name*, whose stream LiteLLM
    refuses with an error the test installs."""

    def __init__(
        self, inner, *, provider_name: str, context_id: str, node_id: str
    ) -> None:
        self._inner = inner
        # A question names its context only through the context's own text,
        # here its lab, and its leaf in the scenario heading.
        self._marks = (
            f"Lab: Synthetic lab {context_id}.",
            f"\n## Scenario {node_id}\n",
        )
        self.rejecting = LiteLLMProvider(
            api_key="fixture-key",
            provider_name=cast(ProviderName, provider_name),
            model=inner.model_id,
            max_concurrent=3,
            reasoning_effort="low",
        )

    model_id = property(lambda self: self._inner.model_id)
    provider_name = property(lambda self: self._inner.provider_name)
    reasoning_effort = property(lambda self: self._inner.reasoning_effort)

    def rejects(self, user_prompt: str) -> bool:
        return request_kind(user_prompt) == "direct" and all(
            mark in user_prompt for mark in self._marks
        )

    async def complete(self, **request) -> ProviderResponse:
        if self.rejects(request["user_prompt"]):
            return await self.rejecting.complete(**request)
        return await self._inner.complete(**request)


def _xai_403(message: str) -> litellm.APIError:
    """An xAI 403 as LiteLLM raises it, quoting xAI's reply."""
    reply = json.dumps(
        {
            "code": "The caller does not have permission to execute the "
            "specified operation",
            "error": message,
        }
    )
    return litellm.APIError(403, f"XaiException - {reply}", "xai", "fixture-a")


@pytest.mark.parametrize(
    ("provider_name", "error", "stops"),
    [
        (
            "anthropic",
            litellm.ContentPolicyViolationError(
                "Output blocked by content filtering policy", "fixture-a", "anthropic"
            ),
            None,
        ),
        # xAI's safety checks, as reported: a 403 naming its usage guidelines.
        (
            "xai",
            _xai_403("Content violates usage guidelines. Failed check: SAFETY_CHECK"),
            None,
        ),
        # Any other 400, or a 403 for a missing permission, is systematic.
        (
            "anthropic",
            litellm.BadRequestError("max_tokens: too large", "fixture-a", "anthropic"),
            "max_tokens: too large",
        ),
        (
            "xai",
            _xai_403("Team t does not have permission to access model fixture-a"),
            "does not have permission to access model",
        ),
    ],
)
def test_a_policy_rejection_refuses_its_call_and_the_study_goes_on(
    tmp_path, monkeypatch, provider_name, error, stops
):
    leaf = CONTEXTS[1].collectible_nodes[0].node_id

    async def rejected(**_kwargs):
        raise error

    monkeypatch.setattr(litellm_provider, "acompletion", rejected)
    wrapped: list[_Rejected] = []

    def rejecting(providers):
        wrapped.append(
            _Rejected(
                providers[0],
                provider_name=provider_name,
                context_id="lab-b",
                node_id=leaf,
            )
        )
        return [wrapped[0], *providers[1:]]

    if stops is not None:
        # A systematic failure stops every lane.
        with pytest.raises(ValueError, match=stops):
            _study(tmp_path / "study", wrap=rejecting)
        (failed,) = _terminal(tmp_path / "study")
        assert failed.relative_to(tmp_path / "study").parts[0] == "lab-b"
        return
    clean, _ = _study(tmp_path / "clean")
    results, _ = _study(tmp_path / "study", wrap=rejecting)
    # Every context completes; the others exactly as an uninterrupted study.
    assert not any(isinstance(row, Exception) for row in results)
    for index in (0, 2):
        name = CONTEXTS[index].context_id
        assert results[index] == clean[index]
        assert _capture(tmp_path / "study" / name) == _capture(
            tmp_path / "clean" / name
        )
    # Each rejected direct question was refused three times, which exhausted
    # it and left its leaf unanswered; nothing failed at the provider.
    assert not _terminal(tmp_path / "study")
    attempts = [
        json.loads(path.read_text())
        for path in sorted((tmp_path / "study" / "lab-b").rglob("attempt-*.json"))
    ]
    refused = [
        row
        for row in attempts
        if row["model_id"] == wrapped[0].model_id
        and wrapped[0].rejects(row["rendered_user_prompt"])
    ]
    assert refused and len(refused) % 3 == 0
    for row in refused:
        assert (row["raw_response"], row["stop_reason"]) == ("", "refusal")
        assert "policy_rejection" in row["provider_evidence"]
    artifact = _artifact(tmp_path / "study" / "lab-b", results[1])
    assert {row.node_id for row in artifact.unanswered_cases} == {leaf}
    assert len(artifact.unanswered_cases) == len(refused) // 3


class _Dropped:
    """From virtual second 30 on, one model refuses new requests for a bad
    key, and every request in flight at second 30 fails for it then."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._origin: float | None = None

    model_id = property(lambda self: self._inner.model_id)
    provider_name = property(lambda self: self._inner.provider_name)
    reasoning_effort = property(lambda self: self._inner.reasoning_effort)

    async def complete(self, **request) -> ProviderResponse:
        loop = asyncio.get_running_loop()
        if self._origin is None:
            self._origin = loop.time()
        now = loop.time() - self._origin
        if now >= 30:
            raise ProviderError("AuthenticationError: 401 refused")
        work = asyncio.ensure_future(self._inner.complete(**request))
        done, _ = await asyncio.wait({work}, timeout=30 - now)
        if work in done:
            return work.result()
        work.cancel()
        raise ProviderError("AuthenticationError: 401 key revoked")


def test_requests_failing_beside_the_first_failure_freeze_nothing(tmp_path):
    """Every in-flight request of a model fails in the same instant: the
    first failure stops the study, and the rest are cancelled, not frozen.

    At virtual second 30 the in-flight requests fail ("key revoked") and any
    request sent from then on is refused at once ("refused"). The warping
    loop's clock includes real time, so which of the two reaches the
    scheduler first depends on load; either stops the study, and exactly one
    attempt is frozen whichever it is."""
    wide = CONFIG.model_copy(update={"concurrency_by_model": dict.fromkeys(MODELS, 12)})

    def dropping(providers):
        return [_Dropped(providers[0]), *providers[1:]]

    with pytest.raises(ValueError, match=r"401 (key revoked|refused)"):
        _study(tmp_path, wrap=dropping, config=wide, slots=12)
    assert len(_terminal(tmp_path)) == 1


def test_an_unresolved_retry_must_have_answered_its_last_attempt(tmp_path):
    study = StudyResponder(CONTEXTS, EVERY_TURN)

    def responder(system_prompt, user_prompt):
        if request_kind(user_prompt) == "direct_retry":
            return "not an answer"
        return study(system_prompt, user_prompt)

    results, _ = _study(tmp_path, responder=responder)
    artifact = results[0]
    assert not isinstance(artifact, Exception)
    run = tmp_path / CONTEXTS[0].context_id
    validate_contextual_capture(
        CONTEXTS[0], artifact, read_contextual_capture_files(run)
    )
    # Recast an exhausted retry's last attempt as a provider failure: the
    # recorded unresolved retry no longer matches its capture.
    last = next(
        path
        for path in sorted(run.glob("calls/*/attempt-003.json"))
        if json.loads(path.read_text())["raw_response"] == "not an answer"
    )
    row = json.loads(last.read_text())
    row.update(
        raw_response=None,
        error="transport unavailable",
        diagnostic_error="transport unavailable",
        response_model_id=None,
    )
    last.write_text(json.dumps(row, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError):
        validate_contextual_capture(
            CONTEXTS[0], artifact, read_contextual_capture_files(run)
        )


def test_a_context_whose_calls_exhaust_validation_still_completes(tmp_path):
    study = StudyResponder(CONTEXTS, EVERY_TURN)

    def responder(system_prompt, user_prompt):
        if (
            LAB_B in user_prompt
            and request_kind(user_prompt) == "chain_guardrail_retry"
        ):
            return "not an answer"
        return study(system_prompt, user_prompt)

    results, trace = _study(tmp_path, responder=responder)
    assert all(isinstance(row, ContextualRiskArtifact) for row in results)
    artifact = cast(ContextualRiskArtifact, results[1])
    # Every exhausted guardrail retry is recorded, the original chain answer
    # drives its law, and the context goes on to cross-checks and its parents.
    calls = artifact.exhausted_calls()
    assert calls and {row.kind for row in calls} == {"chain_guardrail_retry"}
    assert len(calls) == sum(
        case.chain_guardrail is not None for case in artifact.cases
    )
    lab_b = [row for row in trace.requests if row.context_id == "lab-b"]
    assert any(row.kind == "crosscheck" for row in lab_b)
    assert artifact.parent_cases
    validate_contextual_capture(
        CONTEXTS[1], artifact, read_contextual_capture_files(tmp_path / "lab-b")
    )


def test_rate_limits_shrink_the_lane_and_collection_still_completes(
    tmp_path, monkeypatch
):
    made = []

    class Recorded(CollectionScheduler):
        def __init__(self, ceilings):
            super().__init__(ceilings)
            made.append(self)

    monkeypatch.setattr(contextual_collection, "CollectionScheduler", Recorded)
    results, trace = _study(tmp_path, slots=3, tokens_per_minute=3000)
    assert all(not isinstance(row, Exception) for row in results)
    refused = sum(len(times) for times in trace.rate_limits.values())
    assert refused > 0
    # One scheduler serves the whole study and heard every refusal.
    assert len(made) == 1
    assert sum(lane.rate_limits for lane in made[0].lanes.values()) == refused
