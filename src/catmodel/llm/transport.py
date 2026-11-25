"""Provider-neutral response and timeout boundaries."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, Literal, Protocol, runtime_checkable

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field

from .models import ReasoningEffort

logger = structlog.get_logger(__name__)

type StopReason = Literal["completed", "output_limit", "refusal"]
"""Why the provider ended a response: it finished, it reached the output-token
limit, or it refused (an explicit refusal, a content filter or a rejection of
the prompt on content-policy grounds)."""


class ProviderResponse(BaseModel):
    """Normalized response retained by cat-model call records."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    model_id: str
    provider_name: str = ""
    provider_request_id: str | None = None
    latency_ms: int = 0
    input_tokens: int | None = None
    """Uncached input tokens only: the provider's input count less its cache
    reads and, where reported, cache writes, which stay in the usage that
    ``evidence`` keeps. A frozen attempt records this as ``input_tokens``."""
    output_tokens: int | None = None
    """Output tokens, reasoning included; a frozen attempt records this as
    ``response_tokens``."""
    stop_reason: StopReason = "completed"
    evidence: dict[str, Any] = Field(default_factory=dict)
    """What the provider returned, kept with the call record: the mechanism
    that collected it, the final response object (which carries the provider's
    own stop or finish reason, ``incomplete_details`` and usage), every lost
    sample (``connection_lost``), for a server-side run its id, failed
    submissions and when each status was first seen, and for a prompt the
    provider rejected on content-policy grounds the rejection
    (``policy_rejection``: its HTTP status, provider error code and error)."""


class ProviderError(Exception):
    """A request that ended without a response, with the evidence gathered
    before it failed (``ProviderResponse.evidence``'s keys).

    ``transient`` marks a failure that sending the request again later may
    not repeat: the transport's own retries or re-sends of a transient
    failure (``is_transient``) ran out. Any other failure, a client error
    such as 400 or 401 or a run the provider ended for a reason of its own,
    is systematic. A rejection of the prompt on content-policy grounds is
    not a failure but a response with stop reason ``refusal``."""

    transient_by_default = False

    def __init__(
        self,
        message: str,
        *,
        evidence: dict[str, Any] | None = None,
        transient: bool | None = None,
    ):
        super().__init__(message)
        self.evidence = evidence
        self.transient = self.transient_by_default if transient is None else transient


class ProviderTimeout(ProviderError, TimeoutError):
    """A request that reached its deadline; transient."""

    transient_by_default = True


TRANSIENT_STATUS_CODES = frozenset({408, 429})
"""Besides every 5xx: the statuses worth sending again unchanged."""


def status_code(error: BaseException) -> int | None:
    """The HTTP status an error carries, if any."""
    for holder in (error, getattr(error, "response", None)):
        code = getattr(holder, "status_code", None)
        if isinstance(code, int) and not isinstance(code, bool):
            return code
    return None


def is_transient(error: BaseException) -> bool:
    """Whether sending the same request again may succeed: a lost connection,
    a rate limit (429), a request timeout (408) or a server error (5xx), or a
    ``ProviderError`` marked transient. Never a client error such as 400 or
    401."""
    if isinstance(error, ProviderError):
        return error.transient
    code = status_code(error)
    if code is not None:
        return code in TRANSIENT_STATUS_CODES or 500 <= code < 600
    return isinstance(error, ConnectionError | TimeoutError | httpx.TransportError)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol implemented by one configured model endpoint."""

    @property
    def model_id(self) -> str: ...

    @property
    def provider_name(self) -> str: ...

    @property
    def reasoning_effort(self) -> ReasoningEffort: ...

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        request_timeout_seconds: float,
    ) -> ProviderResponse: ...


def validate_request_timeout_seconds(timeout_seconds: float) -> float:
    """Return a finite positive provider deadline."""
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("request_timeout_seconds must be finite and greater than 0")
    return timeout


_DETACHED_PROVIDER_TASKS: set[asyncio.Future[Any]] = set()


def _consume_detached_provider_task(task: asyncio.Future[Any]) -> None:
    _DETACHED_PROVIDER_TASKS.discard(task)
    with suppress(BaseException):
        task.exception()


def _cancel_without_waiting(task: asyncio.Future[Any]) -> None:
    """Request cancellation without letting lower-layer cleanup hold a slot."""
    task.cancel()
    _DETACHED_PROVIDER_TASKS.add(task)
    task.add_done_callback(_consume_detached_provider_task)


async def await_with_hard_timeout[T](
    awaitable: Awaitable[T],
    timeout_seconds: float,
) -> T:
    """Enforce a deadline without waiting for a cancelled provider task to unwind.

    ``asyncio.timeout`` waits for the cancelled awaitable to unwind, so a wedged
    HTTP cleanup could hold a provider slot forever. Here the awaitable runs in
    a child task observed with ``asyncio.wait``: the caller raises at the
    deadline while the child finishes on its own, at the cost that wedged
    cleanup can keep a connection.
    """
    timeout = validate_request_timeout_seconds(timeout_seconds)
    task = asyncio.ensure_future(awaitable)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except BaseException:
        _cancel_without_waiting(task)
        raise
    if task in done:
        return task.result()
    _cancel_without_waiting(task)
    raise TimeoutError


_OPEN_RUNS: dict[str, Callable[[], Awaitable[object]]] = {}
"""Server-side runs submitted and not yet seen to end, by provider and run
id, with the call that cancels each."""


def open_run(key: str, cancel: Callable[[], Awaitable[object]]) -> None:
    """Record a submitted server-side run until ``close_run``."""
    _OPEN_RUNS[key] = cancel


def close_run(key: str) -> None:
    """Forget a run the provider reported ended."""
    _OPEN_RUNS.pop(key, None)


async def cancel_open_runs(timeout_seconds: float = 30.0) -> None:
    """Cancel every server-side run still open.

    A run outlives the request that submitted it when the request reached
    its deadline, was cancelled with its study or was interrupted; the
    provider would keep generating, and billing, it. Await this before the
    event loop that ran the requests closes (``with_open_runs_cancelled``).
    """
    runs = dict(_OPEN_RUNS)
    _OPEN_RUNS.clear()
    if not runs:
        return
    tasks = {asyncio.ensure_future(cancel()): key for key, cancel in runs.items()}
    done, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
    for task in pending:
        task.cancel()
    for task, key in tasks.items():
        error = None if task in done and not task.cancelled() else "no reply"
        if error is None and task.exception() is not None:
            error = f"{type(task.exception()).__name__}: {task.exception()}"[:500]
        if error is None:
            logger.info("cancelled an open server-side run", run=key)
        else:
            logger.error("could not cancel a server-side run", run=key, error=error)


async def with_open_runs_cancelled[T](work: Awaitable[T]) -> T:
    """Await *work*, then cancel every server-side run it left open, however
    it ended. Wrap the coroutine handed to ``asyncio.run`` with it."""
    try:
        return await work
    finally:
        await cancel_open_runs()


async def complete(
    provider: LLMProvider,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    request_timeout_seconds: float,
) -> ProviderResponse:
    """Complete one prompt and translate a provider deadline failure."""
    timeout_seconds = validate_request_timeout_seconds(request_timeout_seconds)
    try:
        return await provider.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            request_timeout_seconds=timeout_seconds,
        )
    except TimeoutError as exc:
        logger.warning(
            "provider request timed out",
            provider=provider.provider_name,
            model=provider.model_id,
            timeout_s=timeout_seconds,
        )
        detail = f" ({exc})" if str(exc) else ""
        raise ProviderTimeout(
            f"{provider.provider_name}/{provider.model_id} request timed out "
            f"after {timeout_seconds:g}s{detail}",
            evidence=getattr(exc, "evidence", None),
        ) from exc
