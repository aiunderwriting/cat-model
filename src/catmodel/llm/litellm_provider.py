"""LiteLLM-backed transport for the providers cat-model elicits.

Every provider call goes through LiteLLM. A request may run for up to its
deadline (2,400 s in the example and test profiles), and a network path may
not hold a connection that long: the transport was developed behind an egress
proxy that answers 502 when a response's first byte takes more than about
30 s and periodically cuts long-lived connections. Each provider is therefore
called through one mechanism, chosen in ``PROVIDERS``:

``background`` (OpenAI)
    Submit a Responses request with ``background: true`` through
    ``litellm.aresponses``, poll it by id with ``litellm.aget_responses`` and
    cancel it with ``litellm.acancel_responses``, so no connection stays open
    while the answer is generated. LiteLLM sends each exchange through the
    transport's HTTP handler, which passes it to LiteLLM's pooled client and
    keeps the provider's reply: the final response object is kept exactly as
    the provider returned it, not as LiteLLM's response object, which
    re-encodes the response id.
``stream`` (Anthropic, xAI, Google, Meta, OpenRouter)
    Stream the chat completion through ``litellm.acompletion`` and aggregate
    it with ``litellm.stream_chunk_builder``. xAI's Responses API does not
    support background mode, and its deferred completions are offered only
    through its own REST endpoint and SDK, so xAI streams too: each call holds
    a connection for its whole answer, so a proxy cut is a re-send recorded
    as ``connection_lost``. Meta is the Meta Model API, LiteLLM's ``meta``
    provider. A stream holds a whole answer only once the provider's own
    finish reason arrived, except that LiteLLM gives every Gemini chunk a
    finish reason of its own, so a Gemini stream that ends without an error
    counts as whole.

Only transient failures (a lost connection, 408, 429, 5xx) are retried. A
poll, the opening of a stream and a submission refused with 429 are retried
until the deadline: polling is idempotent, a stream that failed to open
generated nothing and a refused submission created no run. LiteLLM sends
Gemini's request only when its stream is first read, so every stream is read
open before its answer is awaited. Any other failed submission may have
created a run, so it is tried at most ``TRANSIENT_ATTEMPTS`` times; a
duplicate run it leaves is billed but never polled, so it selects nothing.
LiteLLM's HTTP client itself sends a request again once, on a new connection,
when its connection fails, so one try may submit twice.

A stream cut once open, a background run the provider ends ``failed`` with a
code in ``RESENT_FAILURE_CODES`` and a background run at its queue limit or
deadline, which is cancelled at once, are lost samples: the request is sent
again at most ``RESENDS`` times in all, and every re-send is recorded as
``connection_lost`` because answers that take longer are lost more often.
Background runs still open when a study stops are cancelled by
``transport.with_open_runs_cancelled``.

A provider that rejects the prompt on content-policy grounds refuses that
call: the call returns a response with no text and stop reason ``refusal``,
which collection counts as a failed validation attempt, and the evidence
keeps the rejection (``policy_rejection``). A rejection is a
``litellm.ContentPolicyViolationError`` (LiteLLM's class for an OpenAI, xAI
or Meta error naming ``content_policy_violation``, an invalid prompt flagged
as violating the usage policy or a request rejected by the safety system, and
for Anthropic's content filtering policy block), a 400 whose provider error
code is in ``POLICY_REJECTION_CODES``, any 4xx whose provider error message
contains ``content violates usage guidelines`` in any case
(``POLICY_REJECTION_PHRASES``: xAI's safety checks are reported to answer
with a 403 saying so, though xAI's documentation does not specify the
format), or a background run the provider ends ``failed`` with a code in
``POLICY_REJECTION_CODES``; a rejection raised mid-stream counts too. A
prompt Gemini blocks, and an answer it stops for safety, end with a content
filter finish reason, which is a refusal like any other.

A call that fails once these retries and re-sends run out, or at its deadline,
raises a ``ProviderError`` marked ``transient``: collection fails only that
call's group and sends it again on resume. Any other failure (another client
error, such as any other 400, a 401, a 403 for a missing permission or model
access, or a 404, or a run the provider ended ``failed`` with another code,
``cancelled`` or otherwise unanswered) is systematic and stops the study.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

import httpx
import structlog

from .models import REASONING_EFFORT_LEVELS, ProviderName, ReasoningEffort
from .rate_limits import report_rate_limited
from .transport import (
    ProviderError,
    ProviderResponse,
    ProviderTimeout,
    StopReason,
    await_with_hard_timeout,
    close_run,
    is_transient,
    open_run,
    status_code,
    validate_request_timeout_seconds,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

_IMPORT_ENV = {
    "LITELLM_LOCAL_MODEL_COST_MAP": "true",
    "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS": "true",
}
# Importing LiteLLM refreshes pricing metadata and beta headers over the
# network. Imports and dry runs must stay offline, so use the bundled maps
# during import and restore the process environment immediately afterward.
_previous_import_env = {name: os.environ.get(name) for name in _IMPORT_ENV}
try:
    for _name, _value in _IMPORT_ENV.items():
        os.environ.setdefault(_name, _value)
    import litellm
    from litellm.llms.custom_httpx.http_handler import (
        AsyncHTTPHandler,
        get_async_httpx_client,
    )
finally:
    for _name, _previous in _previous_import_env.items():
        if _previous is None:
            os.environ.pop(_name, None)
        else:
            os.environ[_name] = _previous

# LiteLLM fails a stream whose last 100 chunks are identical, a check on the
# answer's content; a re-send would then select on content.
litellm.REPEATED_STREAMING_CHUNK_LIMIT = sys.maxsize
# LiteLLM prints a help banner to stdout for every error it maps, a transient
# poll failure included; the transport logs its own failures.
litellm.suppress_debug_info = True

logger = structlog.get_logger(__name__)

type Mechanism = Literal["background", "stream"]


@dataclass(frozen=True)
class ProviderSpec:
    """How cat-model calls one provider through LiteLLM."""

    litellm_prefix: str
    mechanism: Mechanism


PROVIDERS: dict[ProviderName, ProviderSpec] = {
    "openai": ProviderSpec("openai", "background"),
    "anthropic": ProviderSpec("anthropic", "stream"),
    "xai": ProviderSpec("xai", "stream"),
    "google": ProviderSpec("gemini", "stream"),
    # The Meta Model API, LiteLLM's ``meta`` provider; not ``meta_llama``,
    # the Llama API.
    "meta": ProviderSpec("meta", "stream"),
    "openrouter": ProviderSpec("openrouter", "stream"),
}
_NATIVE_REASONING_EFFORT: frozenset[ProviderName] = frozenset({"xai", "meta"})
"""Providers whose OpenAI-compatible API takes ``reasoning_effort`` as it is."""

TRANSIENT_ATTEMPTS = 5
"""Tries of a submission that fails transiently other than by a 429, with
exponential backoff between them."""
RESENDS = 2
"""Re-sends of one request after lost samples; one bound shared by every
cause."""
RESENT_FAILURE_CODES = frozenset({"server_error", "rate_limit_exceeded"})
"""Error codes of a ``failed`` background run that are transient: the run
produced no answer, so the request is submitted again as a new run."""
POLICY_REJECTION_CODES = frozenset({"invalid_prompt", "content_policy_violation"})
"""Provider error codes that reject the prompt on content-policy grounds, in
a 400's error body or as a ``failed`` background run's error code: OpenAI's
``invalid_prompt`` (a prompt flagged as potentially violating its usage
policy) and the OpenAI-compatible ``content_policy_violation``."""
POLICY_REJECTION_PHRASES = ("content violates usage guidelines",)
"""Phrases that mark any 4xx error as a rejection of the prompt on
content-policy grounds when the provider's error message contains one, in
any case: xAI's ``Content violates usage guidelines``, which its safety checks
are reported to answer with a 403. xAI's documentation does not specify the
format, so the phrase rests on that reported behavior."""
POLL_SECONDS = 10.0
"""Wait between polls of a background run; a response is seen at most this
long after it completes."""
_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 30.0
_EXCHANGE_TIMEOUT_SECONDS = 60.0
"""Timeout of one submission, poll or cancellation, each of which returns at
once."""
_CANCEL_SECONDS = 30.0
"""The wait for the cancellation of a background run at its queue limit or
deadline before the request is sent again; a run whose cancellation is not
confirmed stays open for ``transport.cancel_open_runs``."""
_PENDING_STATUSES = frozenset({"queued", "in_progress"})
"""A background run's statuses before it ends."""


def litellm_model_name(provider_name: ProviderName, model_id: str) -> str:
    prefix = PROVIDERS[provider_name].litellm_prefix
    if model_id.startswith(f"{prefix}/"):
        return model_id
    return f"{prefix}/{model_id}"


acompletion = litellm.acompletion
aresponses = litellm.aresponses
aget_responses = litellm.aget_responses
acancel_responses = litellm.acancel_responses
stream_chunk_builder = litellm.stream_chunk_builder


def http_client(provider_name: ProviderName) -> AsyncHTTPHandler:
    """LiteLLM's pooled HTTP handler for *provider_name*, with its proxy and
    TLS settings."""
    return get_async_httpx_client(
        llm_provider=litellm.LlmProviders(PROVIDERS[provider_name].litellm_prefix)
    )


class _ReplyKeeper(AsyncHTTPHandler):
    """The HTTP handler LiteLLM sends one Responses API exchange through: it
    passes the exchange unchanged to LiteLLM's pooled handler and keeps the
    provider's reply."""

    def __init__(self, pooled: AsyncHTTPHandler) -> None:
        # AsyncHTTPHandler.__init__ would open a client of its own; the pooled
        # handler owns the connection.
        self._pooled = pooled
        self.reply: httpx.Response | None = None

    async def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
        self.reply = await self._pooled.post(*args, **kwargs)
        return self.reply

    async def get(self, *args: Any, **kwargs: Any) -> httpx.Response:
        self.reply = await self._pooled.get(*args, **kwargs)
        return self.reply


class StreamCut(Exception):
    """A stream ended before the provider's finish reason."""


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:500]


def _retry_after_seconds(error: BaseException) -> float | None:
    """The wait a provider asked for in its ``Retry-After`` header, if any."""
    for headers in (
        getattr(error, "headers", None),
        getattr(getattr(error, "response", None), "headers", None),
        getattr(error, "litellm_response_headers", None),
    ):
        mapping = _as_mapping(headers)
        if mapping is None:
            continue
        value = next(
            (item for key, item in mapping.items() if key.lower() == "retry-after"),
            None,
        )
        if value is None:
            continue
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            try:
                when = parsedate_to_datetime(str(value))
            except (TypeError, ValueError):
                continue
            seconds = (when - datetime.now(UTC)).total_seconds()
        if math.isfinite(seconds):
            return max(seconds, 0.0)
    return None


def backoff_seconds(failures: int, error: BaseException) -> float:
    """The wait before the next try: exponential with jitter, capped at
    ``_MAX_BACKOFF_SECONDS``, or longer if the provider asked for it."""
    delay = min(_MAX_BACKOFF_SECONDS, _BACKOFF_SECONDS * 2 ** min(failures - 1, 16))
    delay *= random.uniform(0.5, 1.0)
    return max(delay, _retry_after_seconds(error) or 0.0)


async def _retrying[T](
    send: Callable[[], Awaitable[T]], *, attempts: int | None = None
) -> T:
    """*send*, again after each transient failure until the request's
    deadline cancels it, but after at most *attempts* failures other than a
    429: a request refused for rate was not acted on. Any other failure is
    raised at once. Each 429 is reported to the request's lane."""
    failures = 0
    counted = 0
    while True:
        try:
            return await send()
        except Exception as error:
            failures += 1
            refused = status_code(error) == 429
            counted += not refused
            if not is_transient(error) or (
                attempts is not None and counted >= attempts
            ):
                raise
            if refused:
                report_rate_limited()
            delay = backoff_seconds(failures, error)
            logger.warning(
                "transient provider failure; retrying",
                error=_describe(error),
                failures=failures,
                delay_s=round(delay, 2),
            )
            await asyncio.sleep(delay)


def _cut_cause(error: BaseException) -> str:
    """Why an open stream ended early: it closed without a finish reason, the
    connection was lost, or the provider sent an error mid-stream."""
    if isinstance(error, StreamCut):
        return "ended_early"
    original = getattr(error, "original_exception", None) or error
    if isinstance(
        original,
        litellm.APIConnectionError | httpx.TransportError | ConnectionError,
    ):
        return "connection_error"
    return "server_error"


def _provider_error_code(error: BaseException) -> str | None:
    """The error code of the provider's JSON error reply that *error* carries:
    its ``body``, or the reply LiteLLM quotes in its message. OpenAI's layout
    nests the code under ``error``."""
    body = _as_mapping(getattr(error, "body", None))
    if body is None:
        text = str(error)
        decoder = json.JSONDecoder()
        start = text.find("{")
        while body is None and start != -1:
            try:
                body = _as_mapping(decoder.raw_decode(text, start)[0])
            except ValueError:
                start = text.find("{", start + 1)
    if body is None:
        return None
    nested = _as_mapping(body.get("error"))
    code = (nested if nested is not None else body).get("code")
    return code if isinstance(code, str) else None


def _names_policy_violation(error: BaseException) -> bool:
    """Whether the provider's error message that *error* carries, in its
    ``body`` or quoted in its message, contains one of
    ``POLICY_REJECTION_PHRASES``, in any case."""
    body = getattr(error, "body", None)
    text = f"{error} {'' if body is None else json.dumps(body, default=str)}"
    return any(phrase in text.casefold() for phrase in POLICY_REJECTION_PHRASES)


def _policy_rejection(error: BaseException) -> dict[str, Any] | None:
    """The record of a provider's rejection of the prompt on content-policy
    grounds, if *error*, or the error LiteLLM wrapped in it, is one: a
    ``litellm.ContentPolicyViolationError``, a 400 whose provider error code
    is in ``POLICY_REJECTION_CODES``, or a 4xx whose provider error message
    contains one of ``POLICY_REJECTION_PHRASES``."""
    for candidate in (error, getattr(error, "original_exception", None)):
        if not isinstance(candidate, BaseException):
            continue
        status = status_code(candidate)
        code = _provider_error_code(candidate)
        if (
            isinstance(candidate, litellm.ContentPolicyViolationError)
            or (status == 400 and code in POLICY_REJECTION_CODES)
            or (
                status is not None
                and 400 <= status < 500
                and _names_policy_violation(candidate)
            )
        ):
            return {"status_code": status, "code": code, "error": _describe(candidate)}
    return None


def _rejected_run(response: object) -> dict[str, Any] | None:
    """The record of a background run the provider ended ``failed`` with a
    code in ``POLICY_REJECTION_CODES``, if *response* is one."""
    error = _get(response, "error")
    code = _get(error, "code")
    if _get(response, "status") != "failed" or code not in POLICY_REJECTION_CODES:
        return None
    return {"status_code": None, "code": code, "error": _get(error, "message")}


_PROMPT_CACHE_KEY_PREFIX = "catmodel-system-"


def _prompt_cache_key(system_prompt: str) -> str:
    digest = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    return f"{_PROMPT_CACHE_KEY_PREFIX}{digest[:32]}"


def _as_mapping(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return cast("Mapping[str, Any]", value)
    return None


def _get(value: object, key: str, default: Any = None) -> Any:
    """Read *key* from a mapping, an attribute or a pydantic model's extras."""
    mapping = _as_mapping(value)
    if mapping is not None:
        return mapping.get(key, default)
    if hasattr(value, key):
        attr = getattr(value, key)
        if attr is not None:
            return attr
    extra = getattr(value, "model_extra", None)
    extra_mapping = _as_mapping(extra)
    if extra_mapping is not None:
        return extra_mapping.get(key, default)
    return default


def _as_items(value: object) -> list[Any]:
    """Return *value* as a list.

    ``None`` gives ``[]``; a string, bytes, mapping or non-iterable is one item.
    """
    if value is None:
        return []
    if isinstance(value, str | bytes) or _as_mapping(value) is not None:
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(cast("Any", value))
    except (TypeError, ValueError):
        return None


def _first_int(*values: object) -> int | None:
    for value in values:
        coerced = _coerce_int(value)
        if coerced is not None:
            return coerced
    return None


def _usage_value(usage: object, key: str) -> int | None:
    return _coerce_int(_get(usage, key))


def _uncached_input_tokens(
    total_input_tokens: int | None,
    cache_read_tokens: int | None,
    cache_creation_tokens: int | None = None,
) -> int | None:
    """Return input tokens not covered by provider cache subsets.

    Provider usage reports cache reads and, for chat responses, cache writes as
    subsets of total input. Call records retain only the uncached remainder.
    """
    if total_input_tokens is None:
        return None
    return max(
        total_input_tokens - (cache_read_tokens or 0) - (cache_creation_tokens or 0),
        0,
    )


def _message_text(message: object) -> str:
    content = _get(message, "content")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in _as_items(content):
        text = _get(block, "text")
        if text is None:
            text = _get(block, "output_text")
        if text is not None:
            parts.append(str(text))
    return "".join(parts)


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _chat_stop_reason(response: object) -> StopReason:
    choices = _as_items(_get(response, "choices"))
    choice = choices[0] if choices else None
    finish_reason = _get(choice, "finish_reason")
    if finish_reason == "length":
        return "output_limit"
    if finish_reason == "content_filter" or _get(_get(choice, "message"), "refusal"):
        return "refusal"
    return "completed"


def _responses_stop_reason(response: object) -> StopReason:
    status = _get(response, "status")
    reason = _get(_get(response, "incomplete_details"), "reason")
    if status == "incomplete" and reason == "max_output_tokens":
        return "output_limit"
    if (status == "incomplete" and reason == "content_filter") or _rejected_run(
        response
    ):
        return "refusal"
    if status != "completed":
        raise ProviderError(
            f"response ended {status}: {_get(response, 'error') or reason}"
        )
    for item in _as_items(_get(response, "output")):
        for part in _as_items(_get(item, "content")):
            if _get(part, "type") == "refusal":
                return "refusal"
    return "completed"


def _resent_failure(response: object) -> str | None:
    """The error code of a background run that failed transiently, if any."""
    code = _get(_get(response, "error"), "code")
    if _get(response, "status") == "failed" and code in RESENT_FAILURE_CODES:
        return code
    return None


class LiteLLMProvider:
    """One concurrent provider endpoint, called through its ``ProviderSpec``.

    Blank completions are successful responses, so the caller's parser judges
    them. Output-limit stops, refusals and policy rejections are responses
    with their ``stop_reason``; a request that ends without a response raises
    ``ProviderError`` with the evidence gathered.
    """

    def __init__(
        self,
        *,
        api_key: str,
        provider_name: ProviderName,
        model: str,
        max_concurrent: int,
        reasoning_effort: ReasoningEffort,
        api_base: str | None = None,
    ) -> None:
        if provider_name not in PROVIDERS:
            raise ValueError(f"Unsupported LiteLLM provider {provider_name!r}")
        if (
            isinstance(max_concurrent, bool)
            or not isinstance(max_concurrent, int)
            or max_concurrent < 1
        ):
            raise ValueError("max_concurrent must be a positive integer")
        if reasoning_effort not in REASONING_EFFORT_LEVELS:
            raise ValueError(
                "reasoning_effort must be one of "
                f"{REASONING_EFFORT_LEVELS}, got {reasoning_effort!r}"
            )
        self._api_key = api_key
        self._provider_name = provider_name
        self._spec = PROVIDERS[provider_name]
        self._api_base = (api_base or "").rstrip("/") or None
        self._model = model
        self._litellm_model = litellm_model_name(provider_name, model)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._reasoning_effort: ReasoningEffort = reasoning_effort

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider_name(self) -> ProviderName:
        return self._provider_name

    @property
    def reasoning_effort(self) -> ReasoningEffort:
        return self._reasoning_effort

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        request_timeout_seconds: float,
    ) -> ProviderResponse:
        timeout_seconds = validate_request_timeout_seconds(request_timeout_seconds)
        evidence: dict[str, Any] = {
            "mechanism": self._spec.mechanism,
            "connection_lost": [],
        }
        # No cancellation point separates a successful acquire from the
        # try/finally that releases it. The request deadline starts after the
        # slot is acquired, so queue time is outside it.
        await self._semaphore.acquire()
        try:
            if self._spec.mechanism == "stream":
                work = self._stream(
                    system_prompt, user_prompt, max_tokens, timeout_seconds, evidence
                )
                hard_timeout = timeout_seconds
            else:
                work = self._run_in_background(
                    system_prompt, user_prompt, max_tokens, timeout_seconds, evidence
                )
                # Each run enforces its own queue limit and deadline, a run at
                # either is cancelled within _CANCEL_SECONDS, and a request may
                # submit RESENDS + 1 runs; this only backs them up.
                hard_timeout = (RESENDS + 1) * (
                    2 * timeout_seconds
                    + POLL_SECONDS
                    + _EXCHANGE_TIMEOUT_SECONDS
                    + _CANCEL_SECONDS
                )
            try:
                return await await_with_hard_timeout(work, hard_timeout)
            except ProviderError as error:
                error.evidence = evidence
                raise
            except TimeoutError as error:
                raise ProviderTimeout(
                    f"no response within {hard_timeout:g}s", evidence=evidence
                ) from error
            except Exception as error:
                raise ProviderError(
                    f"{type(error).__name__}: {error}",
                    evidence=evidence,
                    transient=is_transient(error),
                ) from error
        finally:
            self._semaphore.release()

    def _refused(
        self, rejection: dict[str, Any], started: float, evidence: dict[str, Any]
    ) -> ProviderResponse:
        """The response to a prompt the provider rejected on content-policy
        grounds: no text, stop reason ``refusal`` and the rejection kept."""
        evidence["policy_rejection"] = rejection
        logger.warning(
            "provider rejected the prompt on content-policy grounds",
            provider=self._provider_name,
            model=self._model,
            code=rejection["code"],
        )
        return ProviderResponse(
            text="",
            model_id=self._model,
            provider_name=self._provider_name,
            latency_ms=_elapsed_ms(started),
            stop_reason="refusal",
            evidence=evidence,
        )

    async def _responses_exchange(
        self, call: Callable[..., Awaitable[object]], **kwargs: Any
    ) -> Any:
        """One Responses API exchange through LiteLLM's *call*: the provider's
        JSON reply as sent.

        LiteLLM's response object re-encodes the response id, and it rejects
        a polled reply its own model does not fit, so a reply the provider
        sent successfully is read as sent even where LiteLLM failed to build
        its object from it.
        """
        keeper = _ReplyKeeper(http_client(self._provider_name))
        if self._api_base is not None:
            kwargs["api_base"] = self._api_base
        try:
            async with asyncio.timeout(_EXCHANGE_TIMEOUT_SECONDS):
                await call(
                    **kwargs,
                    api_key=self._api_key,
                    timeout=_EXCHANGE_TIMEOUT_SECONDS,
                    client=keeper,
                )
        except Exception as error:
            if keeper.reply is None or not keeper.reply.is_success:
                raise
            logger.warning(
                "LiteLLM could not read a provider reply; keeping the reply",
                provider=self._provider_name,
                model=self._model,
                error=_describe(error),
            )
        if keeper.reply is None:
            raise ProviderError("LiteLLM did not send the exchange through its client")
        return keeper.reply.json()

    def _cancel_call(self, run_id: str) -> Callable[[], Awaitable[object]]:
        """The exchange that cancels background run *run_id*."""
        return partial(
            self._responses_exchange,
            acancel_responses,
            response_id=run_id,
            custom_llm_provider=self._spec.litellm_prefix,
        )

    async def _cancel_lost_run(self, run_id: str) -> str | None:
        """Cancel a background run this request gave up on at its limit and
        close it; ``None`` once the provider confirmed it, otherwise why not.
        An unconfirmed run stays open for ``cancel_open_runs``."""
        key = f"{self._provider_name}:{run_id}"
        try:
            async with asyncio.timeout(_CANCEL_SECONDS):
                await self._cancel_call(run_id)()
        except Exception as error:
            logger.warning(
                "could not cancel a background run at its limit",
                provider=self._provider_name,
                model=self._model,
                run_id=run_id,
                error=_describe(error),
            )
            return _describe(error)
        close_run(key)
        return None

    async def _run_in_background(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        timeout_seconds: float,
        evidence: dict[str, Any],
    ) -> ProviderResponse:
        """Submit a background run and poll it until the provider returns its
        final response.

        Submission and queueing get the request's deadline, the queue limit.
        The deadline starts again when the run is first seen in progress, so
        queueing under load cannot time out a long answer. Each status's first
        sighting is recorded in milliseconds from the run's submission. A lost
        run is submitted again as a new run with a queue limit and deadline of
        its own; a submission still unanswered at the queue limit fails the
        call.
        """
        started = time.monotonic()
        provider = self._spec.litellm_prefix
        # Background mode needs a stored response: the provider keeps it under
        # the account's retention terms so it can be polled.
        submit = partial(
            self._responses_exchange,
            aresponses,
            model=self._litellm_model,
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=max_tokens,
            reasoning={"effort": self._reasoning_effort},
            prompt_cache_key=_prompt_cache_key(system_prompt),
            background=True,
            store=True,
            custom_llm_provider=provider,
            # cat-model's own retries are the only ones: LiteLLM's need
            # tenacity, which is not installed.
            num_retries=0,
        )
        poll = partial(
            self._responses_exchange, aget_responses, custom_llm_provider=provider
        )
        submit_failures: list[dict[str, Any]] = []
        evidence["submit_failures"] = submit_failures
        lost: list[dict[str, Any]] = evidence["connection_lost"]

        async def submit_once() -> Any:
            try:
                return await submit()
            except Exception as error:
                # The provider may have accepted it: a retry may leave a
                # duplicate run billed, though only one is ever polled.
                submit_failures.append(
                    {"elapsed_ms": _elapsed_ms(started), "error": _describe(error)}
                )
                logger.warning(
                    "submission failed",
                    provider=self._provider_name,
                    model=self._model,
                    elapsed_ms=_elapsed_ms(started),
                    error=_describe(error),
                )
                raise

        while True:
            # The evidence describes the run being polled; a lost run's is in
            # its connection_lost record.
            run_started = time.monotonic()
            timeline: dict[str, int] = {}
            evidence["status_first_seen_ms"] = timeline
            evidence.pop("response", None)
            run_id = None
            try:
                async with asyncio.timeout(timeout_seconds) as limit:
                    try:
                        created = await _retrying(
                            submit_once, attempts=TRANSIENT_ATTEMPTS
                        )
                    except Exception as error:
                        rejection = _policy_rejection(error)
                        if rejection is None:
                            raise
                        return self._refused(rejection, started, evidence)
                    run_id = str(created["id"])
                    evidence["run_id"] = run_id
                    key = f"{self._provider_name}:{run_id}"
                    logger.info(
                        "server-side run submitted",
                        provider=self._provider_name,
                        model=self._model,
                        run_id=run_id,
                    )
                    open_run(key, self._cancel_call(run_id))
                    final, status = created, created.get("status")
                    while True:
                        if status not in timeline:
                            timeline[status] = _elapsed_ms(run_started)
                            if status == "in_progress":
                                limit.reschedule(
                                    asyncio.get_running_loop().time() + timeout_seconds
                                )
                        if status not in _PENDING_STATUSES:
                            break
                        await asyncio.sleep(POLL_SECONDS)
                        final = await _retrying(partial(poll, response_id=run_id))
                        status = final.get("status")
            except TimeoutError:
                elapsed_ms = _elapsed_ms(run_started)
                generating = "in_progress" in timeline
                timed_out = "no response " + (
                    f"{timeout_seconds:g}s after it was first seen in progress"
                    if generating
                    else f"within {timeout_seconds:g}s of submission"
                )
                if run_id is None:
                    raise ProviderTimeout(timed_out) from None
                # A run at its limit is a lost sample like a failed one, but
                # it may still be generating, and billed: cancel it first.
                cancel_error = await self._cancel_lost_run(run_id)
                lost.append(
                    {
                        "elapsed_ms": elapsed_ms,
                        "cause": "deadline" if generating else "queue_deadline",
                        "run_id": run_id,
                        "status_first_seen_ms": timeline,
                        "error": timed_out,
                        "cancel_error": cancel_error,
                    }
                )
                logger.warning(
                    "background run reached its limit; submitting again as a "
                    "new sample",
                    provider=self._provider_name,
                    model=self._model,
                    run_id=run_id,
                    cause=lost[-1]["cause"],
                    connection_lost=len(lost),
                )
                if len(lost) > RESENDS:
                    raise ProviderTimeout(
                        f"{timed_out}; {len(lost)} background runs lost"
                    ) from None
                continue
            # A run that ended needs no cancelling; one left by a call that
            # was itself cancelled stays open for cancel_open_runs.
            close_run(key)
            evidence["response"] = final
            cause = _resent_failure(final)
            if cause is None:
                break
            lost.append(
                {
                    "elapsed_ms": _elapsed_ms(run_started),
                    "cause": cause,
                    "run_id": run_id,
                    "status_first_seen_ms": timeline,
                    "error": _get(final, "error"),
                }
            )
            if cause == "rate_limit_exceeded":
                report_rate_limited()
            logger.warning(
                "background run failed; submitting again as a new sample",
                provider=self._provider_name,
                model=self._model,
                run_id=run_id,
                cause=cause,
                connection_lost=len(lost),
            )
            if len(lost) > RESENDS:
                raise ProviderError(
                    f"{len(lost)} background runs lost; the last failed after "
                    f"{lost[-1]['elapsed_ms']} ms: {lost[-1]['error']}",
                    transient=True,
                )
        return self._responses_result(final, started, evidence)

    async def _stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        timeout_seconds: float,
        evidence: dict[str, Any],
    ) -> ProviderResponse:
        """Stream through LiteLLM, sending a stream cut once open again as a
        new sample."""
        started = time.monotonic()
        kwargs: dict[str, Any] = {
            "model": self._litellm_model,
            "messages": self._messages(system_prompt, user_prompt),
            "max_tokens": max_tokens,
            "api_key": self._api_key,
            "timeout": timeout_seconds,
            "stream": True,
            "stream_options": {"include_usage": True},
            # cat-model's own retries are the only ones: LiteLLM's retry
            # every error, a 400 included, and need tenacity, which is not
            # installed.
            "num_retries": 0,
            "max_retries": 0,
        }
        if self._api_base is not None:
            kwargs["api_base"] = self._api_base
        self._add_reasoning_params(kwargs, self._reasoning_effort)
        self._add_openrouter_data_collection_policy(kwargs)
        lost: list[dict[str, Any]] = evidence["connection_lost"]

        async def open_stream() -> Any:
            # ``stream=True`` makes LiteLLM return its stream wrapper.
            stream = cast("CustomStreamWrapper", await acompletion(**kwargs))
            # LiteLLM sends some providers' requests, Gemini's, only when their
            # stream is first read; send it now, so that it opens or fails here
            # like the others.
            await stream.fetch_stream()
            return stream

        while True:
            # A provider's error status (429, 5xx, 400) is raised here, before
            # the provider has started an answer, so a transient one is
            # retried until the deadline.
            try:
                stream = await _retrying(open_stream)
            except Exception as error:
                rejection = _policy_rejection(error)
                if rejection is None:
                    raise
                return self._refused(rejection, started, evidence)
            opened = time.monotonic()
            chunks: list[Any] = []
            try:
                async for chunk in stream:
                    chunks.append(chunk)
                if not self._whole(stream, chunks):
                    raise StreamCut("stream ended before the provider's finish reason")
                break
            except Exception as error:
                rejection = _policy_rejection(error)
                if rejection is not None:
                    return self._refused(rejection, started, evidence)
                if not (isinstance(error, StreamCut) or is_transient(error)):
                    raise
                lost.append(
                    {
                        "elapsed_ms": _elapsed_ms(opened),
                        "chunks": len(chunks),
                        "cause": _cut_cause(error),
                        "error": _describe(error),
                    }
                )
                logger.warning(
                    "stream cut; sending again as a new sample",
                    provider=self._provider_name,
                    model=self._model,
                    connection_lost=len(lost),
                )
                if len(lost) > RESENDS:
                    raise ProviderError(
                        f"stream cut {len(lost)} times; the last after "
                        f"{lost[-1]['elapsed_ms']} ms: {lost[-1]['error']}",
                        transient=True,
                    ) from error
        response = stream_chunk_builder(chunks, messages=kwargs["messages"])
        if not isinstance(response, litellm.ModelResponse):
            raise ProviderError("LiteLLM could not assemble the streamed response")
        evidence["response"] = response.model_dump(mode="json")
        return self._chat_result(response, started, evidence)

    def _whole(self, stream: object, chunks: list[Any]) -> bool:
        """Whether *stream* ended with the provider's own finish reason, and
        so holds a whole answer.

        LiteLLM closes a stream that ends early with a finish reason of its
        own and records the provider's as ``received_finish_reason`` (LiteLLM
        internals, coupled to the pinned version). For Gemini it records none
        and gives every chunk a finish reason, ``stop`` where Gemini sent
        none, so a Gemini stream that ends without an error after any chunk
        counts as whole; a lost connection still raises and is a lost sample.
        """
        if self._provider_name == "google":
            return bool(chunks)
        return getattr(stream, "received_finish_reason", None) is not None

    def _messages(self, system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
        # Implements PROVIDER_MESSAGE_LAYOUT, which resumable runs freeze.
        if self._provider_name == "anthropic":
            return [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
            ]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _add_reasoning_params(
        self,
        kwargs: dict[str, Any],
        reasoning_effort: ReasoningEffort,
    ) -> None:
        # LiteLLM maps this setting to each provider's native control:
        # Anthropic adaptive thinking effort and Gemini 3 ``thinkingLevel``;
        # a numeric thinking budget is ignored for gemini-3. Gemini 3 has no
        # ``xhigh`` or ``max`` level, and LiteLLM rejects them before sending.
        if self._provider_name == "openrouter":
            kwargs.setdefault("extra_body", {})["reasoning"] = {
                "effort": reasoning_effort
            }
            return
        kwargs["reasoning_effort"] = reasoning_effort
        if self._provider_name in _NATIVE_REASONING_EFFORT:
            # These providers take ``reasoning_effort`` as it is. LiteLLM sends
            # it only for models its bundled model map lists as reasoning
            # models and refuses it for any other, such as a model newer than
            # the map, unless it is allowed.
            kwargs["allowed_openai_params"] = ["reasoning_effort"]

    def _add_openrouter_data_collection_policy(self, kwargs: dict[str, Any]) -> None:
        if self._provider_name != "openrouter":
            return
        # ``data_collection=deny`` restricts routing to upstream providers
        # that do not collect prompts for training or retention.
        extra_body = kwargs.setdefault("extra_body", {})
        extra_body["provider"] = {"data_collection": "deny"}

    def _responses_result(
        self, resp: object, started: float, evidence: dict[str, Any]
    ) -> ProviderResponse:
        stop_reason = _responses_stop_reason(resp)
        rejection = _rejected_run(resp)
        if rejection is not None:
            evidence["policy_rejection"] = rejection
        text_parts: list[str] = []
        for item in _as_items(_get(resp, "output")):
            if _get(item, "type") != "message":
                continue
            for part in _as_items(_get(item, "content")):
                if _get(part, "type") != "output_text":
                    continue
                text = _get(part, "text")
                if text:
                    text_parts.append(str(text))

        text = "".join(text_parts)
        usage = _get(resp, "usage")
        input_details = _get(usage, "input_tokens_details")
        cache_read_tokens = _first_int(
            _get(usage, "cache_read_input_tokens"),
            _get(input_details, "cached_tokens"),
        )
        return ProviderResponse(
            text=text,
            model_id=self._model,
            provider_name=self._provider_name,
            provider_request_id=_get(resp, "id"),
            input_tokens=_uncached_input_tokens(
                _usage_value(usage, "input_tokens"), cache_read_tokens
            ),
            output_tokens=_usage_value(usage, "output_tokens"),
            latency_ms=_elapsed_ms(started),
            stop_reason=stop_reason,
            evidence=evidence,
        )

    def _chat_result(
        self, resp: object, started: float, evidence: dict[str, Any]
    ) -> ProviderResponse:
        choices = _as_items(_get(resp, "choices"))
        choice = choices[0] if choices else None
        message = _get(choice, "message") if choice is not None else None
        text = _message_text(message)
        usage = _get(resp, "usage")
        prompt_details = _get(usage, "prompt_tokens_details")
        cache_read_tokens = _first_int(
            _get(usage, "cache_read_input_tokens"),
            _get(prompt_details, "cached_tokens"),
        )
        cache_creation_tokens = _usage_value(usage, "cache_creation_input_tokens")
        return ProviderResponse(
            text=text,
            model_id=self._model,
            provider_name=self._provider_name,
            provider_request_id=_get(resp, "id"),
            input_tokens=_uncached_input_tokens(
                _usage_value(usage, "prompt_tokens"),
                cache_read_tokens,
                cache_creation_tokens,
            ),
            output_tokens=_usage_value(usage, "completion_tokens"),
            latency_ms=_elapsed_ms(started),
            stop_reason=_chat_stop_reason(resp),
            evidence=evidence,
        )
