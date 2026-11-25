"""Shared, ranked and adaptive admission of provider requests.

Every collection of one run sends its provider requests through one lane per
model. A lane keeps the requests that are ready to send in rank order and lets
as many run as its limit allows; no collection holds slots of its own, so a
free slot goes to the best-ranked ready request of any context.

The limit starts at the model's configured ceiling and falls multiplicatively
when the provider reports a rate limit (a 429 or an exhausted token budget)
on a request sent since the limit last fell: it keeps ``BACKOFF`` of itself,
and refusals of requests sent before then are the same overload already
answered, so a burst of them counts once, as in TCP's one decrease per
window. Each response that succeeds without a rate limit while at least half
the limit is in use adds one slot back, never past the ceiling. That growth
is per success, not per window, so the limit can double in one round trip.
The provider's retry layer reports every 429 it retries, a poll's included,
through ``catmodel.llm.rate_limits.report_rate_limited``.

When a provider request fails with an error that ``transport.is_transient``
does not class as transient, such as a 400 or 401, every lane stops sending: a
bad key or request fails every request alike and fast, so each freed slot would
otherwise go to the next request and freeze a failure for that one too.
Requests already waiting stay queued until their collection is cancelled, and
a request in flight that fails after the first is cancelled rather than
frozen. A transient failure, one whose transport retries ran out, stops nothing
here.

Admission order never reaches a capture: frozen calls are keyed by their
request's identity, so ranks decide only when a request is sent.

Every request in flight is served by the one event loop of its process, so
the scheduler also keeps that loop answering. Preparing a request (rendering,
replaying and freezing it) holds the loop for milliseconds, and a study
starts thousands at once: back to back they would hold it for minutes while
the exchanges already in flight go unread and time out. Each request
therefore takes a turn (``CollectionScheduler.turn``), one per loop
iteration. A process at the limit of its CPU core then starts requests only
as fast as it serves those in flight, so it holds fewer than its limits allow
instead of more than it can read; ``watch_event_loop`` reports that, and a
loop that runs late all the same.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import math
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import structlog

from catmodel.llm.rate_limits import listen_for_rate_limits
from catmodel.llm.transport import is_transient

if TYPE_CHECKING:
    from catmodel.llm.models import ReasoningEffort
    from catmodel.llm.transport import LLMProvider, ProviderResponse

logger = structlog.get_logger(__name__)

type Rank = tuple[int, ...]

LOOP_LATE_SECONDS = 5.0
"""How late the event loop may run a timer before ``watch_event_loop`` counts
it: later than the single holds a collection makes by design, such as the
few seconds it validates and freezes a finished context in, and about when
exchanges start to fail, since the transport's HTTP client gives a connection
5 s to open."""
LOOP_REPORT_SECONDS = 60.0
"""The time over which ``watch_event_loop`` measures the process's CPU, and
the least time between two of its ``event loop late`` warnings."""
BUSY_SHARE = 0.9
"""The share of one CPU core, over ``LOOP_REPORT_SECONDS``, from which
``watch_event_loop`` reports the process as bound by its CPU."""


async def watch_event_loop(interval: float = 1.0) -> None:
    """Warn while this process's CPU core bounds its collection, until
    cancelled.

    A process that used ``BUSY_SHARE`` of a core over the last
    ``LOOP_REPORT_SECONDS`` is serving all the requests it can: its lanes'
    turns keep it from taking on more than it reads, so it holds fewer in
    flight than its limits allow and answers arrive no faster for a higher
    ceiling. That is reported every ``LOOP_REPORT_SECONDS`` while it lasts.

    A timer due every ``interval`` seconds that fires more than
    ``LOOP_LATE_SECONDS`` late means the loop was held that long, and every
    exchange in flight went unread meanwhile; provider latencies, timeouts
    and failed submissions are then local. The first late timer is reported
    at once and later ones at most every ``LOOP_REPORT_SECONDS``, with the
    worst lateness and the count since the last report.

    Each warning carries the seconds since the watch began, so that a report
    equal to an earlier one is still written (``catmodel.logging`` writes a
    warning repeated with identical fields once).
    """
    loop = asyncio.get_running_loop()
    began = loop.time()
    worst, late, reported = 0.0, 0, -LOOP_REPORT_SECONDS
    measured, used = began, time.process_time()
    while True:
        due = loop.time() + interval
        await asyncio.sleep(interval)
        now = loop.time()
        elapsed = round(now - began, 1)
        if now - due > LOOP_LATE_SECONDS:
            worst, late = max(worst, now - due), late + 1
        if late and now - reported >= LOOP_REPORT_SECONDS:
            logger.warning(
                "event loop late: exchanges in flight went unread; use more "
                "collector processes (--workers) or lower the ceilings",
                worst_s=round(worst, 1),
                late_checks=late,
                elapsed_s=elapsed,
            )
            worst, late, reported = 0.0, 0, now
        if now - measured >= LOOP_REPORT_SECONDS:
            share = (time.process_time() - used) / (now - measured)
            measured, used = now, time.process_time()
            if share >= BUSY_SHARE:
                logger.warning(
                    "collector bound by its CPU core, not its ceilings; use "
                    "more collector processes (--workers) or lower the ceilings",
                    cpu_share=round(share, 2),
                    elapsed_s=elapsed,
                )


BACKOFF = 0.9
"""The share of its limit a lane keeps after a rate limit. Provider limits
are rates (tokens or requests per minute), not queues: the in-flight count a
rate sustains is the rate times the response time, so a gentle decrease
settles near it, where halving would idle up to half of it for a round trip
of minutes."""


class _Stop:
    """The first systematic provider failure of a run, shared by its lanes."""

    def __init__(self) -> None:
        self.failure: BaseException | None = None


class ModelLane:
    """One model's ready requests and its adaptive in-flight limit."""

    def __init__(self, ceiling: int, stop: _Stop | None = None) -> None:
        if isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling < 1:
            raise ValueError("a lane's ceiling must be a positive integer")
        self._stop = stop or _Stop()
        self.ceiling = ceiling
        self.limit = float(ceiling)
        self.in_flight = 0
        self.rate_limits = 0
        self.decreases = 0
        self._ready: list[tuple[Rank, int, asyncio.Future[int]]] = []
        self._order = itertools.count()

    @property
    def slots(self) -> int:
        return max(1, math.floor(self.limit))

    async def acquire(self, rank: Rank) -> int:
        """Wait until this request ranks first among the ready ones and a
        slot is free; return the number of decreases before its admission."""
        if (
            not self._ready
            and self.in_flight < self.slots
            and self._stop.failure is None
        ):
            self.in_flight += 1
            return self.decreases
        waiter = asyncio.get_running_loop().create_future()
        heapq.heappush(self._ready, (rank, next(self._order), waiter))
        try:
            return await waiter
        except BaseException:
            if waiter.done() and not waiter.cancelled():
                # Admitted as it was cancelled: hand the slot on.
                self.release()
            else:
                waiter.cancel()
            raise

    def release(self) -> None:
        self.in_flight -= 1
        self._admit()

    @property
    def stopped(self) -> bool:
        return self._stop.failure is not None

    def failed(self, error: BaseException) -> None:
        """A request failed systematically at the provider: stop every lane
        of the run."""
        if self._stop.failure is None:
            self._stop.failure = error

    def succeeded(self) -> None:
        if self.in_flight * 2 >= self.limit:
            self.limit = min(float(self.ceiling), self.limit + 1)
            self._admit()

    def rate_limited(self, admitted_after: int) -> None:
        """A refusal of a request admitted after ``admitted_after`` decreases."""
        self.rate_limits += 1
        if admitted_after < self.decreases:
            return
        self.decreases += 1
        self.limit = max(1.0, self.limit * BACKOFF)

    def _admit(self) -> None:
        while (
            self._ready and self.in_flight < self.slots and self._stop.failure is None
        ):
            _, _, waiter = heapq.heappop(self._ready)
            if waiter.done():
                continue
            waiter.set_result(self.decreases)
            self.in_flight += 1


class CollectionScheduler:
    """The lanes shared by every collection of one run, keyed by model."""

    def __init__(self, ceilings: Mapping[str, int]) -> None:
        self._stop = _Stop()
        self.lanes = {
            model: ModelLane(ceiling, self._stop) for model, ceiling in ceilings.items()
        }
        self._turns = asyncio.Lock()

    async def turn(self) -> None:
        """Wait for a turn to do work that holds the event loop, such as
        preparing one request. Turns are taken in the order asked for, one
        per loop iteration, so the loop serves the exchanges in flight
        between two of them however many callers are waiting."""
        async with self._turns:
            await asyncio.sleep(0)

    @property
    def failure(self) -> BaseException | None:
        """The systematic provider failure that stopped every lane, if one did."""
        return self._stop.failure

    def provider(self, provider: LLMProvider, rank: Rank) -> LLMProvider:
        """``provider`` for one request, admitted through its model's lane."""
        return _Admitted(provider, self.lanes[provider.model_id], rank)


class _Admitted:
    """A provider whose requests wait for their lane, keeping its identity."""

    def __init__(self, provider: LLMProvider, lane: ModelLane, rank: Rank) -> None:
        self._provider = provider
        self._lane = lane
        self._rank = rank

    @property
    def model_id(self) -> str:
        return self._provider.model_id

    @property
    def provider_name(self) -> str:
        return self._provider.provider_name

    @property
    def reasoning_effort(self) -> ReasoningEffort:
        return self._provider.reasoning_effort

    async def complete(self, **request: Any) -> ProviderResponse:
        admitted_after = await self._lane.acquire(self._rank)
        limited = False

        def heard() -> None:
            nonlocal limited
            limited = True
            self._lane.rate_limited(admitted_after)

        try:
            with listen_for_rate_limits(heard):
                response = await self._provider.complete(**request)
        except Exception as error:
            if self._lane.stopped:
                # Another request's failure already stopped the run: this one,
                # in flight beside it, is cancelled like the rest, so it
                # freezes nothing and is sent again on resume.
                raise asyncio.CancelledError from error
            if not is_transient(error):
                self._lane.failed(error)
            raise
        else:
            if not limited:
                self._lane.succeeded()
            return response
        finally:
            self._lane.release()
