"""Tests for the provider-neutral transport boundary."""

from __future__ import annotations

import asyncio
import gc
import math

import pytest
import structlog.testing

from catmodel.llm import litellm_provider, transport
from catmodel.llm.litellm_provider import LiteLLMProvider
from catmodel.llm.models import ReasoningEffort
from catmodel.llm.transport import (
    ProviderResponse,
    await_with_hard_timeout,
    complete,
    validate_request_timeout_seconds,
)
from test_support.llm_fakes import FakeStream, answer_chunks


class _Provider:
    model_id = "model-test"
    provider_name = "provider-test"
    reasoning_effort: ReasoningEffort = "low"

    def __init__(
        self, result: ProviderResponse | None = None, error: Exception | None = None
    ):
        self.result = result or ProviderResponse(text="answer", model_id=self.model_id)
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def complete(self, **kwargs: object) -> ProviderResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.parametrize("timeout", [0, -1, math.inf, -math.inf, math.nan])
def test_request_timeout_must_be_finite_and_positive(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and greater than 0"):
        validate_request_timeout_seconds(timeout)


@pytest.mark.asyncio
async def test_complete_translates_timeout_and_emits_structured_warning() -> None:
    provider = _Provider(error=TimeoutError("underlying timeout"))

    with (
        structlog.testing.capture_logs() as logs,
        pytest.raises(
            TimeoutError,
            match=r"provider-test/model-test request timed out after 2\.5s",
        ),
    ):
        await complete(
            provider,
            system_prompt="system",
            user_prompt="user",
            max_tokens=123,
            request_timeout_seconds=2.5,
        )

    assert provider.calls == [
        {
            "system_prompt": "system",
            "user_prompt": "user",
            "max_tokens": 123,
            "request_timeout_seconds": 2.5,
        }
    ]
    assert logs == [
        {
            "event": "provider request timed out",
            "provider": "provider-test",
            "model": "model-test",
            "timeout_s": 2.5,
            "log_level": "warning",
        }
    ]


@pytest.mark.asyncio
async def test_complete_propagates_non_timeout_exception_unchanged() -> None:
    error = RuntimeError("provider failed")
    provider = _Provider(error=error)

    with pytest.raises(RuntimeError) as caught:
        await complete(
            provider,
            system_prompt="system",
            user_prompt="user",
            max_tokens=123,
            request_timeout_seconds=2.5,
        )

    assert caught.value is error


@pytest.mark.asyncio
async def test_hard_timeout_releases_owner_while_cancelled_cleanup_is_stuck() -> None:
    cleanup = asyncio.Event()
    started = asyncio.Event()

    async def work() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await cleanup.wait()
            raise

    with pytest.raises(TimeoutError):
        await await_with_hard_timeout(work(), 0.01)
    assert started.is_set()

    cleanup.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_detached_task_exception_is_consumed() -> None:
    cleanup = asyncio.Event()
    loop = asyncio.get_running_loop()
    contexts: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()

    def collect_exception(
        _loop: asyncio.AbstractEventLoop, context: dict[str, object]
    ) -> None:
        contexts.append(context)

    async def work() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await cleanup.wait()
            raise RuntimeError("late provider failure") from None

    loop.set_exception_handler(collect_exception)
    try:
        with pytest.raises(TimeoutError):
            await await_with_hard_timeout(work(), 0.01)
        cleanup.set()
        await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)
        assert contexts == []
        assert not transport._DETACHED_PROVIDER_TASKS
    finally:
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_active_call_cancellation_cancels_child_without_waiting() -> None:
    cleanup = asyncio.Event()
    entered = asyncio.Event()

    async def work() -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await cleanup.wait()
            raise

    task = asyncio.create_task(await_with_hard_timeout(work(), 10))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    cleanup.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_cancelled_semaphore_waiter_does_not_consume_next_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    calls = 0

    async def fake_chat(**_kwargs: object) -> FakeStream:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            await release_first.wait()
        return FakeStream(answer_chunks("answer"))

    monkeypatch.setattr(litellm_provider, "acompletion", fake_chat)
    provider = LiteLLMProvider(
        api_key="key",
        provider_name="anthropic",
        model="model",
        max_concurrent=1,
        reasoning_effort="low",
    )

    first = asyncio.create_task(
        provider.complete(
            system_prompt="system",
            user_prompt="first",
            max_tokens=10,
            request_timeout_seconds=1.0,
        )
    )
    await first_entered.wait()
    cancelled = asyncio.create_task(
        provider.complete(
            system_prompt="system",
            user_prompt="cancelled",
            max_tokens=10,
            request_timeout_seconds=1.0,
        )
    )
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    release_first.set()
    await first
    await provider.complete(
        system_prompt="system",
        user_prompt="third",
        max_tokens=10,
        request_timeout_seconds=1.0,
    )
    assert calls == 2


@pytest.mark.asyncio
async def test_provider_queue_time_is_outside_transport_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    calls = 0

    async def fake_chat(**_kwargs: object) -> FakeStream:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            await release_first.wait()
        return FakeStream(answer_chunks("done"))

    monkeypatch.setattr(litellm_provider, "acompletion", fake_chat)
    provider = LiteLLMProvider(
        api_key="key",
        provider_name="anthropic",
        model="model",
        max_concurrent=1,
        reasoning_effort="low",
    )

    first = asyncio.create_task(
        provider.complete(
            system_prompt="system",
            user_prompt="first",
            max_tokens=10,
            request_timeout_seconds=1.0,
        )
    )
    await first_entered.wait()
    queued = asyncio.create_task(
        provider.complete(
            system_prompt="system",
            user_prompt="queued",
            max_tokens=10,
            request_timeout_seconds=0.05,
        )
    )
    await asyncio.sleep(0.3)
    release_first.set()
    assert (await queued).text == "done"
    await first
    assert calls == 2


@pytest.mark.asyncio
async def test_provider_hard_timeout_releases_slot_while_cleanup_is_stuck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup = asyncio.Event()
    cleanup_entered = asyncio.Event()
    first_entered = asyncio.Event()
    calls = 0

    async def fake_chat(**_kwargs: object) -> FakeStream:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_entered.set()
                await cleanup.wait()
                raise RuntimeError("late provider failure") from None
        return FakeStream(answer_chunks("second"))

    monkeypatch.setattr(litellm_provider, "acompletion", fake_chat)
    provider = LiteLLMProvider(
        api_key="key",
        provider_name="anthropic",
        model="model",
        max_concurrent=1,
        reasoning_effort="low",
    )
    try:
        first = asyncio.create_task(
            provider.complete(
                system_prompt="system",
                user_prompt="first",
                max_tokens=10,
                request_timeout_seconds=0.01,
            )
        )
        await first_entered.wait()
        with pytest.raises(TimeoutError):
            await first
        await cleanup_entered.wait()
        assert not cleanup.is_set()
        second = await provider.complete(
            system_prompt="system",
            user_prompt="second",
            max_tokens=10,
            request_timeout_seconds=0.2,
        )
        assert second.text == "second"
        assert calls == 2
    finally:
        cleanup.set()
        await asyncio.sleep(0)
