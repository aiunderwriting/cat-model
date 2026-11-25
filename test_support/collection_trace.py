"""A virtual-time trace of contextual collection, for scheduling work only.

Collection is run unchanged against a fake provider whose response times
follow a configurable latency law. A warping event loop lets provider waits
pass instantly while local work, including executor threads, runs in real
time, so a study of many hours of provider time is traced in minutes and the
trace still charges every local CPU cost to the clock.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import math
import re
import selectors
import time
from collections import defaultdict
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from catmodel.llm.rate_limits import report_rate_limited
from catmodel.llm.transport import ProviderResponse
from test_support.contextual_risk import CHAIN_MARKER, CROSSCHECK_MARKER

# Warping event loop


class _WarpSelector(selectors.BaseSelector):
    """A selector that returns real I/O when ready and otherwise advances the
    loop's virtual clock to its next timer instead of sleeping."""

    def __init__(self) -> None:
        self._real = selectors.DefaultSelector()
        self.loop: WarpLoop | None = None

    def register(self, fileobj, events, data=None):
        return self._real.register(fileobj, events, data)

    def unregister(self, fileobj):
        return self._real.unregister(fileobj)

    def modify(self, fileobj, events, data=None):
        return self._real.modify(fileobj, events, data)

    def get_map(self):
        return self._real.get_map()

    def close(self) -> None:
        self._real.close()

    def select(self, timeout=None):
        events = self._real.select(0)
        loop = self.loop
        if events or loop is None or timeout == 0:
            return events
        if loop.pending_threads or timeout is None:
            # Executor work runs in real time and wakes the loop when done.
            return self._real.select(timeout)
        loop.warp += timeout
        return []


class WarpLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock is real time plus every skipped wait."""

    def __init__(self) -> None:
        selector = _WarpSelector()
        super().__init__(selector)
        selector.loop = self
        self.warp = 0.0
        self.pending_threads = 0

    def time(self) -> float:
        return time.monotonic() + self.warp

    def run_in_executor(self, executor, func, *args):
        future = super().run_in_executor(executor, func, *args)
        self.pending_threads += 1

        def finished(_future) -> None:
            self.pending_threads -= 1

        future.add_done_callback(finished)
        return future


def run_warped(coroutine) -> tuple[Any, float]:
    """Run ``coroutine`` on a warping loop; return its result and the
    virtual seconds it took."""
    loop = WarpLoop()
    try:
        start = loop.time()
        result = loop.run_until_complete(coroutine)
        return result, loop.time() - start
    finally:
        # As ``asyncio.run`` does, cancel what is left, such as the rest of
        # a failed collection, before closing.
        left = asyncio.all_tasks(loop)
        for task in left:
            task.cancel()

        async def settled() -> None:
            await asyncio.gather(*left, return_exceptions=True)

        loop.run_until_complete(settled())
        loop.close()


# Latency and rate limits


@dataclass(frozen=True)
class ModelLatency:
    """One respondent model's response-time law and server-side limits.

    Output tokens are lognormal with mean ``output_tokens`` and log standard
    deviation ``sigma``; a response takes ``first_token_seconds`` plus its
    tokens over ``tokens_per_second``. ``tokens_per_minute`` is a server token
    budget: a request that would exceed it is answered 429, reported as a
    rate limit and retried after ``retry_after_seconds``, as a provider's
    retry layer does.
    """

    output_tokens: float
    tokens_per_second: float
    sigma: float = 0.5
    first_token_seconds: float = 2.0
    tokens_per_minute: float = math.inf
    retry_after_seconds: float = 20.0


_TRACE_TOKEN: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "collection_trace_token", default=None
)


class LatencyProvider:
    """A fake provider: ``responder`` writes the answer text, the latency law
    sets how long it takes, and ``max_concurrent`` bounds its own slots like
    a real provider's semaphore."""

    def __init__(
        self,
        *,
        model_id: str,
        provider_name: str,
        latency: ModelLatency,
        responder: Callable[[str, str], str],
        trace: CollectionTrace,
        max_concurrent: int,
        seed: int = 0,
    ) -> None:
        self.model_id = model_id
        self.provider_name = provider_name
        self.reasoning_effort = "max"
        self.latency = latency
        self.responder = responder
        self.trace = trace
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._rng = np.random.default_rng(seed)
        self._window: list[tuple[float, float]] = []

    def _tokens(self) -> float:
        law = self.latency
        mu = math.log(law.output_tokens) - law.sigma**2 / 2
        return float(self._rng.lognormal(mu, law.sigma))

    def _admit(self, now: float, tokens: float) -> bool:
        budget = self.latency.tokens_per_minute
        self._window = [(at, used) for at, used in self._window if at > now - 60]
        # A request alone in the window is always served, as a provider serves
        # one request above its per-minute budget rather than none.
        if self._window and sum(used for _, used in self._window) + tokens > budget:
            return False
        self._window.append((now, tokens))
        return True

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        request_timeout_seconds: float,
    ) -> ProviderResponse:
        loop = asyncio.get_running_loop()
        await self.semaphore.acquire()
        try:
            token = _TRACE_TOKEN.get()
            start = loop.time()
            tokens = self._tokens()
            while not self._admit(loop.time(), tokens):
                self.trace.rate_limited(self.model_id, loop.time())
                report_rate_limited()
                await asyncio.sleep(self.latency.retry_after_seconds)
            admitted = loop.time()
            seconds = (
                self.latency.first_token_seconds
                + tokens / self.latency.tokens_per_second
            )
            await asyncio.sleep(seconds)
            if token is not None:
                self.trace.served(token, start, admitted, loop.time())
            return ProviderResponse(
                text=self.responder(system_prompt, user_prompt),
                model_id=self.model_id,
                provider_name=self.provider_name,
                latency_ms=0,
                input_tokens=None,
                output_tokens=None,
            )
        finally:
            self.semaphore.release()


# Request identity from the frozen request


CHAIN_GUARDRAIL_MARKER = "Keeping your answer is as acceptable as changing it"
PARENT_MARKER = "societal harm from the whole category above"
CONTRADICTION_MARKER = "Your answer to this question contradicted itself"

KINDS = {
    "ContextualPrompt": "direct",
    "ContextualSeverityRetryPrompt": "direct_retry",
    "ContextualChainPrompt": "chain",
    "ChainGuardrailRetryPrompt": "chain_guardrail_retry",
    "EalCrosscheckPrompt": "crosscheck",
    "EalCrosscheckRetryPrompt": "crosscheck_retry",
    "ContextualParentPrompt": "parent",
}
"""Each question class's kind of request."""


def request_kind(user_prompt: str) -> str:
    """A question's kind, read from its text as a respondent reads it."""
    if PARENT_MARKER in user_prompt:
        return "parent"
    if CROSSCHECK_MARKER in user_prompt:
        return (
            "crosscheck_retry" if CONTRADICTION_MARKER in user_prompt else "crosscheck"
        )
    if CHAIN_MARKER in user_prompt:
        return (
            "chain_guardrail_retry"
            if CHAIN_GUARDRAIL_MARKER in user_prompt
            else "chain"
        )
    if CONTRADICTION_MARKER in user_prompt:
        return "direct_retry"
    return "direct"


def _heading(user_prompt: str, name: str) -> str | None:
    """The subject a question's ``## Scenario`` or ``## Category`` heading names."""
    found = re.search(rf"^## {name} ([^\s:]+)", user_prompt, re.MULTILINE)
    return None if found is None else found.group(1)


def question_identity(prompt: Any) -> tuple[str, str, str, float]:
    """A frozen question's kind, context, subject and coordinate."""
    kind = KINDS[type(prompt).__name__]
    base = getattr(prompt, "direct", None) or getattr(prompt, "chain", None)
    base = getattr(prompt, "crosscheck", None) or base or prompt
    subject = base.parent.parent_id if hasattr(base, "parent") else base.node.node_id
    return kind, base.instrument.context_id, subject, base.anchor_score


def _shown_eal_means(user_prompt: str) -> tuple[float, float]:
    """The direct and implied EAL means an EAL cross-check question shows."""
    found = [
        re.search(pattern, user_prompt)
        for pattern in (
            r"its mean is USD ([0-9,.e+-]+) \(four significant",
            r"The EAL your count and severity answers imply: USD ([0-9,.e+-]+),",
        )
    ]
    assert all(row is not None for row in found)
    direct, implied = (float(row.group(1).replace(",", "")) for row in found if row)
    return direct, implied


@dataclass
class TracedRequest:
    token: int
    kind: str
    context_id: str
    subject: str
    anchor_score: float
    respondent: tuple[str, str, int]
    model_id: str
    ready: float
    start: float | None = None
    """When the request took a provider slot."""
    admitted: float | None = None
    """When the token budget let it through: service time excludes back-off."""
    end: float | None = None
    done: float | None = None


@dataclass
class CollectionTrace:
    """Every traced request with its ready, service and completion times."""

    requests: list[TracedRequest] = field(default_factory=list)
    rate_limits: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def begin(self, request: Mapping[str, Any], question: Any, now: float) -> int:
        kind, context_id, subject, anchor = question_identity(question)
        respondent = request["respondent"]
        self.requests.append(
            TracedRequest(
                token=len(self.requests),
                kind=kind,
                context_id=context_id,
                subject=subject,
                anchor_score=anchor,
                respondent=(
                    respondent["respondent_id"],
                    respondent["model_id"],
                    respondent["sample_index"],
                ),
                model_id=respondent["model_id"],
                ready=now,
            )
        )
        return len(self.requests) - 1

    def served(self, token: int, start: float, admitted: float, end: float) -> None:
        row = self.requests[token]
        row.start, row.admitted, row.end = start, admitted, end

    def finished(self, token: int, now: float) -> None:
        self.requests[token].done = now

    def rate_limited(self, model_id: str, now: float) -> None:
        self.rate_limits[model_id].append(now)


@contextmanager
def traced_collection(trace: CollectionTrace) -> Generator[None]:
    """Record every frozen call that contextual collection makes."""
    import catmodel.collection.contextual_collection as collection

    original = collection.collect_frozen_call

    async def recorded(**kwargs):
        loop = asyncio.get_running_loop()
        token = trace.begin(
            kwargs["request"], kwargs["parse_answer"].keywords["prompt"], loop.time()
        )
        _TRACE_TOKEN.set(token)
        try:
            return await original(**kwargs)
        finally:
            # Each call sets its own token, so clearing (not resetting) is
            # enough, and stays safe when an aborted run's leftover call is
            # closed from another context.
            _TRACE_TOKEN.set(None)
            trace.finished(token, loop.time())

    setattr(collection, "collect_frozen_call", recorded)  # noqa: B010
    try:
        yield
    finally:
        setattr(collection, "collect_frozen_call", original)  # noqa: B010


# Synthetic study answers


def _unit(*parts: object) -> float:
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


@dataclass(frozen=True)
class TriggerRates:
    """The share of leaf nodes that take each conditional turn, and of
    parents whose category answer disagrees with what is beneath them. The
    draw is fixed per turn, context and node (or parent): a question's text
    does not name its respondent, so every respondent and sample of a node in
    a context takes the same turns, and every schedule of the same study sees
    the same turns.

    Collection itself decides from its stored values which turns are asked,
    as in production. ``chain_guardrail_retry`` is the share of chain questions
    whose synthetic answer trips a guardrail and takes the retry. ``crosscheck``
    is the share whose direct EAL lies a hundred times above the EAL its own
    direct count and marks imply, which fires the ``eal_gap`` trigger.
    ``crosscheck_count_revision`` is the share whose cross-check adopts the
    count that brings those EAL means into line. ``parent_gap`` is the share
    of parents whose category answer lies ten times above the EAL its leaves'
    direct answers imply, so that its pool moves what is beneath it; a parent
    takes no conditional turn."""

    direct_retry: float = 0.05
    chain_guardrail_retry: float = 0.2
    crosscheck: float = 0.2
    crosscheck_count_revision: float = 0.0
    crosscheck_retry: float = 0.15
    parent_gap: float = 0.3


class StudyResponder:
    """Answers every question of a study's instruments, taking each
    conditional turn at its rate.

    ``implied`` holds each leaf's EAL from its direct count and marks. The
    direct EAL of a ``crosscheck`` turn is set far from it to trigger the EAL
    cross-check."""

    def __init__(self, instruments, rates: TriggerRates = TriggerRates()) -> None:
        from catmodel.elicitation.question_text import question_preamble
        from catmodel.setup.implied_eal import implied_eal
        from test_support.contextual_risk import scenario_node_answer

        self.rates = rates
        self.instruments = {row.context_id: row for row in instruments}
        # A question names no context: each context's questions begin with
        # its own preamble, its lab, deployment and trajectory.
        self.preambles = {
            (row.context_id, score): question_preamble(row, score) + "\n\n"
            for row in instruments
            for score in row.anchor_scores
        }
        self.base = {}
        self.implied: dict[tuple[str, str], float] = {}
        for instrument in instruments:
            for node in instrument.collectible_nodes:
                answer = scenario_node_answer(node)
                self.base[instrument.context_id, node.node_id] = answer
                value = implied_eal(
                    instrument, node, answer, instrument.anchor_scores[0]
                )
                # The EAL is linear in the count; every synthetic large-loss
                # count that drives a run is zero, so the law is unchanged.
                self.implied[instrument.context_id, node.node_id] = (
                    0.0 if value is None or value.eal_mean is None else value.eal_mean
                )

    def _takes(self, turn: str, context_id: str, subject: str) -> bool:
        return _unit(turn, context_id, subject) < getattr(self.rates, turn)

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        from catmodel.elicitation.eal_crosscheck import EAL_CROSSCHECK_RATIO
        from catmodel.probability.numerics import differs_by_more_than
        from test_support.contextual_risk import (
            crosscheck_answer,
            declined_crosscheck,
        )

        kind = request_kind(user_prompt)
        context_id = next(
            context
            for (context, _), preamble in self.preambles.items()
            if user_prompt.startswith(preamble)
        )
        if kind == "parent":
            return self._parent(context_id, _heading(user_prompt, "Category"))
        node_id = _heading(user_prompt, "Scenario") or ""
        base = self.base[context_id, node_id]
        node = next(
            row
            for row in self.instruments[context_id].collectible_nodes
            if row.node_id == node_id
        )
        levels = node.total is not None and node.total.large_loss_level is not None
        very = levels and node.total is not None and node.total.very_large_loss_level
        assert base.rate is not None

        def contradicting(answer):
            assert answer.rate is not None
            update = {
                "large_loss_count": answer.rate.model_copy(
                    update={
                        key: 0.06 * value
                        for key, value in answer.rate.model_dump().items()
                    }
                )
            }
            return answer.model_copy(update=update)

        direct = base
        if self._takes("crosscheck", context_id, node_id):
            direct = direct.model_copy(
                update={
                    "direct_eal": base.rate.model_copy(
                        update={
                            key: 100 * max(self.implied[context_id, node_id], 1.0)
                            for key in ("p5", "p25", "p50", "p75", "p95")
                        }
                    )
                }
            )
        if kind == "direct":
            if levels and self._takes("direct_retry", context_id, node_id):
                return contradicting(direct).model_dump_json()
            return direct.model_dump_json()
        if kind in ("direct_retry",):
            return direct.model_dump_json()
        count = base.rate.p50
        if kind == "chain":
            assert node.frequency_chain is not None
            exposure = self.instruments[context_id].chain_exposure(node.frequency_chain)
            return self._chain(
                node,
                count / exposure.count,
                guardrail=self._takes("chain_guardrail_retry", context_id, node_id),
            )
        if kind == "chain_guardrail_retry":
            assert node.frequency_chain is not None
            exposure = self.instruments[context_id].chain_exposure(node.frequency_chain)
            return self._chain(node, count / exposure.count)
        if kind == "crosscheck":
            # Close a direct EAL gap in the requested judgment.
            direct_mean, implied_mean = _shown_eal_means(user_prompt)
            gap = differs_by_more_than(direct_mean, implied_mean, EAL_CROSSCHECK_RATIO)
            if (
                gap
                and implied_mean > 0
                and self._takes("crosscheck_count_revision", context_id, node_id)
            ):
                factor = direct_mean / implied_mean
                return crosscheck_answer(
                    {
                        "rate": {
                            key: value * factor
                            for key, value in base.rate.model_dump().items()
                        }
                    }
                ).model_dump_json()
            matched = dict.fromkeys(
                ("p5", "p25", "p50", "p75", "p95"), max(implied_mean, 1.0)
            )
            if levels and self._takes("crosscheck_retry", context_id, node_id):
                return crosscheck_answer(
                    {
                        "direct_eal": matched,
                        "large_loss_count": dict.fromkeys(matched, 0.06 * count),
                        **(
                            {"very_large_loss_count": dict.fromkeys(matched, 0.0)}
                            if very
                            else {}
                        ),
                    }
                ).model_dump_json()
            if not gap:
                return declined_crosscheck().model_dump_json()
            return crosscheck_answer({"direct_eal": matched}).model_dump_json()
        if kind == "crosscheck_retry":
            return crosscheck_answer(
                {
                    "large_loss_count": dict.fromkeys(
                        ("p5", "p25", "p50", "p75", "p95"), 0.0
                    ),
                    **(
                        {
                            "very_large_loss_count": dict.fromkeys(
                                ("p5", "p25", "p50", "p75", "p95"), 0.0
                            )
                        }
                        if very
                        else {}
                    ),
                }
            ).model_dump_json()
        raise ValueError(f"unexpected question kind {kind}")

    @staticmethod
    def _chain(node, median: float, *, guardrail: bool = False) -> str:
        """A spread chain whose implied median, per unit of exposure, is
        ``median``: ordinary chains have a moderate rate spread and terminal
        probability; flagged chains have a fine opportunity grain for retry."""
        from catmodel.contracts.contextual_risk import ContextualChainAnswer

        chain = node.frequency_chain
        assert chain is not None
        stages = len(chain.stages)
        terminal = 0.0005 if guardrail else 0.9
        center = median / (0.9 ** (stages - 1) * terminal)
        keys = ("p5", "p25", "p50", "p75", "p95")
        rate_factors = (
            (0.0001, 0.01, 1, 10, 2_000) if guardrail else (0.3, 0.6, 1, 1.6, 3)
        )
        return ContextualChainAnswer.model_validate(
            {
                "status": "available",
                "opportunity_rate": dict(
                    zip(keys, (center * f for f in rate_factors), strict=True)
                ),
                "stages": [
                    {
                        "stage_id": stage.stage_id,
                        "probability": dict(
                            zip(
                                keys,
                                (
                                    (0.0001, 0.0003, terminal, 0.0007, 0.0009)
                                    if guardrail
                                    else (0.5, 0.6, terminal, 0.95, 1.0)
                                )
                                if stage.stage_id == chain.stages[-1].stage_id
                                else (0.6, 0.8, 0.9, 0.95, 1.0),
                                strict=True,
                            )
                        ),
                    }
                    for stage in chain.stages
                ],
                "reasoning": {
                    "median_reason": "Synthetic",
                    "low_case": "Synthetic",
                    "high_case": "Synthetic",
                    "method": "direct_judgment",
                    "evidence_used": [],
                },
            }
        ).model_dump_json()

    def _subtree(self, context_id: str, parent_id: str) -> float:
        from catmodel.elicitation.contextual_hierarchy import subtree_leaves

        instrument = self.instruments[context_id]
        return math.fsum(
            self.implied[context_id, leaf]
            for leaf in subtree_leaves(instrument, parent_id)
        )

    def _parent(self, context_id: str, parent_id: str | None) -> str:
        from test_support.contextual_risk import parent_answer

        assert parent_id is not None
        parent = next(
            row
            for row in self.instruments[context_id].parents
            if row.parent_id == parent_id
        )
        children = max(self._subtree(context_id, parent_id), 1.0)
        factor = 10.0 if self._takes("parent_gap", context_id, parent_id) else 1.0
        return parent_answer(
            factor * 1.01 * children, 0.01 * children if parent.children else None
        ).model_dump_json()
