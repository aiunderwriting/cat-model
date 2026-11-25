"""Tests for each provider mechanism's requests, polling, streams and retries.

Background runs go through LiteLLM's own Responses API calls to an in-memory
API; streams go through a scripted ``acompletion`` or, for xAI, Meta and
Google, LiteLLM's own stream paths to a local server.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import litellm
import pytest
from aiohttp import web
from litellm.exceptions import MidStreamFallbackError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from catmodel.llm import litellm_provider, transport
from catmodel.llm.litellm_provider import (
    PROVIDERS,
    RESENDS,
    TRANSIENT_ATTEMPTS,
    LiteLLMProvider,
    backoff_seconds,
)
from catmodel.llm.models import ProviderName
from catmodel.llm.rate_limits import listen_for_rate_limits
from catmodel.llm.transport import ProviderError, ProviderTimeout, is_transient
from test_support.llm_fakes import (
    FakeStream,
    InMemoryHandler,
    ScriptedAcompletion,
    ScriptedAPI,
    answer_chunks,
    chunk,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "llm"
RESPONSES_API = ("aresponses", "aget_responses", "acancel_responses")
STREAMED = tuple(
    sorted(name for name, spec in PROVIDERS.items() if spec.mechanism == "stream")
)


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm_provider, "POLL_SECONDS", 0.0)
    monkeypatch.setattr(litellm_provider, "_BACKOFF_SECONDS", 0.0)


@pytest.fixture(scope="module", autouse=True)
def _warm_litellm_responses_api() -> None:
    """Call LiteLLM's Responses API once per worker against an in-memory API,
    so the imports its first call makes do not count against the short run
    deadlines below."""
    handler = ScriptedAPI(after=(200, _openai_response())).client_factory()("openai")

    async def warm() -> None:
        await litellm.aresponses(
            model="openai/fixture-model", input="warm-up", api_key="k", client=handler
        )
        for call in (litellm.aget_responses, litellm.acancel_responses):
            await call(
                response_id="resp_fixture",
                custom_llm_provider="openai",
                api_key="k",
                client=handler,
            )

    asyncio.run(warm())


@pytest.fixture(autouse=True)
def _litellm_responses_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """LiteLLM's own Responses API calls. They send only through the
    transport's ``http_client``, which reaches nothing until a test installs
    an in-memory API."""
    for name in RESPONSES_API:
        monkeypatch.setattr(litellm_provider, name, getattr(litellm, name))


def _provider(
    provider_name: str, model: str = "fixture-model", effort: str = "high"
) -> LiteLLMProvider:
    return LiteLLMProvider(
        api_key="fixture-key",
        provider_name=cast(ProviderName, provider_name),
        model=model,
        max_concurrent=1,
        reasoning_effort=cast(Any, effort),
    )


# A deadline that a request expected to succeed never reaches. The first
# request through LiteLLM in a process imports the OpenAI SDK's resources and
# a tokenizer, about 1.5 s on an idle machine and several times that on a
# loaded CI runner, all inside the request's deadline. Tests of the deadlines
# themselves pass their own.
_DEADLINE_SECONDS = 60.0


async def _complete(
    provider: LiteLLMProvider, timeout: float = _DEADLINE_SECONDS, **kwargs: Any
):
    return await provider.complete(
        system_prompt=kwargs.get("system_prompt", "system"),
        user_prompt=kwargs.get("user_prompt", "user"),
        max_tokens=kwargs.get("max_tokens", 100),
        request_timeout_seconds=timeout,
    )


def _openai_response(
    status: str = "completed",
    *,
    text: str = '{"value": 3}',
    incomplete: str | None = None,
    content: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": "resp_fixture",
        "object": "response",
        "created_at": 1741476542,
        "status": status,
        "background": True,
        "model": "fixture-model",
        "incomplete_details": None if incomplete is None else {"reason": incomplete},
        "output": [
            {
                "id": "rs_1",
                "type": "reasoning",
                "summary": [],
                "encrypted_content": None,
            },
            {
                "id": "msg_1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": content
                or [{"type": "output_text", "annotations": [], "text": text}],
            },
        ],
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 4},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 1},
            "total_tokens": 12,
        },
    }


# ---- request construction ------------------------------------------------


def _logical_prompt(payload: dict[str, Any]) -> list[tuple[str, str]]:
    def text(content: Any) -> str:
        if isinstance(content, str):
            return content
        assert all(part["type"] == "text" for part in content)
        return "".join(part["text"] for part in content)

    if "instructions" in payload:
        return [("system", payload["instructions"]), ("user", payload["input"])]
    return [
        (message["role"], text(message["content"])) for message in payload["messages"]
    ]


async def _captured_request(
    monkeypatch: pytest.MonkeyPatch, provider: LiteLLMProvider, **kwargs: Any
) -> dict[str, Any]:
    """The request *provider* sends: the body LiteLLM submits for a
    background run, or the keyword arguments of a stream."""
    if PROVIDERS[provider.provider_name].mechanism == "stream":
        acompletion = ScriptedAcompletion(FakeStream(answer_chunks("answer")))
        monkeypatch.setattr(litellm_provider, "acompletion", acompletion)
        await _complete(provider, **kwargs)
        return acompletion.calls[0]
    api = ScriptedAPI((200, _openai_response()))
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    await _complete(provider, **kwargs)
    [submitted] = api.requests
    assert submitted["method"] == "POST"
    assert submitted["authorization"] == "Bearer fixture-key"
    return {"path": submitted["path"], "json": submitted["json"]}


@pytest.mark.parametrize("provider_name", sorted(PROVIDERS))
async def test_provider_prompts_preserve_roles(
    monkeypatch: pytest.MonkeyPatch, provider_name: str
) -> None:
    system = "Use only supplied evidence; do not browse or call tools.\nReturn JSON."
    user = (
        "\n<context>Evidence α; counts are annual.</context>\nEstimate p5, p50, p95.\n"
    )
    request = await _captured_request(
        monkeypatch, _provider(provider_name), system_prompt=system, user_prompt=user
    )
    payload = request.get("json", request)
    assert _logical_prompt(payload) == [("system", system), ("user", user)]


async def test_request_payload_goldens(monkeypatch: pytest.MonkeyPatch) -> None:
    records = json.loads((FIXTURE_ROOT / "request_payloads.json").read_bytes())
    for record in records:
        provider = LiteLLMProvider(
            api_key=record["api_key"],
            provider_name=record["provider"],
            model=record["model"],
            max_concurrent=1,
            reasoning_effort=record["reasoning_effort"],
        )
        request = await _captured_request(
            monkeypatch,
            provider,
            system_prompt=record["system_prompt"],
            user_prompt=record["user_prompt"],
            max_tokens=record["max_tokens"],
            timeout=record["request_timeout_seconds"],
        )
        assert record["mechanism"] == PROVIDERS[record["provider"]].mechanism
        assert request == record["request"], record["name"]


async def test_background_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    openai = await _captured_request(
        monkeypatch, _provider("openai", "openai/fixture-model")
    )
    assert openai["path"] == "/v1/responses"
    assert openai["json"]["background"] is True
    assert openai["json"]["store"] is True
    assert openai["json"]["model"] == "fixture-model"
    assert openai["json"]["reasoning"] == {"effort": "high"}
    assert openai["json"]["max_output_tokens"] == 100
    assert (
        openai["json"]["prompt_cache_key"]
        == "catmodel-system-" + hashlib.sha256(b"system").hexdigest()[:32]
    )
    assert not {"temperature", "top_p", "top_logprobs"} & openai["json"].keys()


async def test_stream_requests_disable_litellm_retries(monkeypatch: pytest.MonkeyPatch):
    kwargs = await _captured_request(monkeypatch, _provider("anthropic"))
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}
    # LiteLLM's own retry layer retries every error, a 400 included.
    assert kwargs["num_retries"] == 0
    assert kwargs["max_retries"] == 0


@pytest.mark.parametrize(
    ("provider_name", "model"),
    [
        # An xAI model newer than the pinned map's grok-4.7.
        ("xai", "grok-5"),
        # A Meta model newer than the pinned map's muse-spark-1.2.
        ("meta", "muse-spark-2"),
    ],
)
async def test_native_reasoning_effort_is_allowed(
    monkeypatch: pytest.MonkeyPatch, provider_name: str, model: str
) -> None:
    kwargs = await _captured_request(monkeypatch, _provider(provider_name, model))
    assert kwargs["model"] == f"{provider_name}/{model}"
    assert kwargs["stream"] is True
    assert kwargs["max_tokens"] == 100
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["allowed_openai_params"] == ["reasoning_effort"]
    # LiteLLM internal, coupled to the pinned version: its mapping for these
    # providers refuses reasoning effort for a model its bundled map does not
    # list as a reasoning model unless it is allowed, and then passes it on.
    params: dict[str, Any] = {
        "model": model,
        "custom_llm_provider": provider_name,
        "max_tokens": 100,
        "reasoning_effort": "high",
    }
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.get_optional_params(**params)
    allowed = litellm.get_optional_params(
        **params, allowed_openai_params=["reasoning_effort"]
    )
    assert (allowed["reasoning_effort"], allowed["max_tokens"]) == ("high", 100)


@pytest.mark.parametrize(
    ("provider_name", "model"),
    [
        # The roster's xAI model.
        ("xai", "grok-4.7"),
        ("meta", "muse-spark-1.2"),
    ],
)
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_litellm_passes_native_reasoning_effort_for_a_mapped_model(
    provider_name: str, model: str, effort: str
) -> None:
    # LiteLLM internal, coupled to the pinned version: the map lists grok-4.7
    # as a reasoning model, and muse-spark-1.2 as one with an xhigh level.
    params = litellm.get_optional_params(
        model=model,
        custom_llm_provider=provider_name,
        max_tokens=100,
        reasoning_effort=effort,
    )
    assert (params["reasoning_effort"], params["max_tokens"]) == (effort, 100)


async def test_anthropic_cache_layout_and_reasoning_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = await _captured_request(
        monkeypatch, _provider("anthropic", "claude-opus-5-5")
    )
    assert kwargs["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["messages"][1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert not {"temperature", "top_p", "top_k", "allowed_openai_params"} & (
        kwargs.keys()
    )

    # LiteLLM internal, coupled to the pinned version; the only offline
    # native-request check.
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    native = AnthropicConfig().transform_request(
        model="claude-opus-5-5",
        messages=kwargs["messages"],
        optional_params={"max_tokens": 100},
        litellm_params={},
        headers={},
    )
    assert native["system"][0]["text"] == "system"
    assert "cache_control" in native["system"][0]
    params = litellm.get_optional_params(
        model=kwargs["model"],
        custom_llm_provider="anthropic",
        max_tokens=kwargs["max_tokens"],
        reasoning_effort=kwargs["reasoning_effort"],
    )
    assert params["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert params["output_config"]["effort"] == "high"


async def test_google_reasoning_payload_maps_to_thinking_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = await _captured_request(
        monkeypatch, _provider("google", "gemini-3.1-pro-preview", "medium")
    )
    from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig

    optional_params = GoogleAIStudioGeminiConfig().map_openai_params(
        non_default_params={"reasoning_effort": kwargs["reasoning_effort"]},
        optional_params={},
        model=kwargs["model"],
        drop_params=False,
    )
    assert optional_params["thinkingConfig"]["thinkingLevel"] == "medium"


async def test_openrouter_denies_provider_data_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = await _captured_request(
        monkeypatch, _provider("openrouter", "deepseek/deepseek-v4.1-flash-1017")
    )
    assert kwargs["extra_body"] == {
        "reasoning": {"effort": "high"},
        "provider": {"data_collection": "deny"},
    }
    assert kwargs["model"] == "openrouter/deepseek/deepseek-v4.1-flash-1017"


# ---- OpenAI background mode ------------------------------------------------


async def test_background_polls_through_completion_and_keeps_the_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _openai_response()
    api = ScriptedAPI(
        (200, {**_openai_response("queued"), "output": [], "usage": None}),
        (200, {**_openai_response("in_progress"), "output": [], "usage": None}),
        (200, {**_openai_response("in_progress"), "output": [], "usage": None}),
        (200, final),
    )
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    response = await _complete(_provider("openai"))
    assert [(r["method"], r["path"]) for r in api.requests] == [
        ("POST", "/v1/responses"),
        ("GET", "/v1/responses/resp_fixture"),
        ("GET", "/v1/responses/resp_fixture"),
        ("GET", "/v1/responses/resp_fixture"),
    ]
    assert all(r["authorization"] == "Bearer fixture-key" for r in api.requests)
    assert response.text == '{"value": 3}'
    assert response.stop_reason == "completed"
    # The provider's own id; LiteLLM's response object re-encodes it.
    assert response.provider_request_id == "resp_fixture"
    assert (response.input_tokens, response.output_tokens) == (6, 2)
    # The final object exactly as the provider returned it.
    evidence = response.evidence
    assert evidence["response"] == final
    assert evidence["mechanism"] == "background"
    assert evidence["run_id"] == "resp_fixture"
    assert evidence["connection_lost"] == evidence["submit_failures"] == []
    seen = evidence["status_first_seen_ms"]
    assert list(seen) == ["queued", "in_progress", "completed"]
    assert seen["queued"] <= seen["in_progress"] <= seen["completed"]
    # A finished run is not left for cancelling.
    assert transport._OPEN_RUNS == {}


async def test_a_reply_litellm_cannot_read_is_kept_as_the_provider_sent_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # LiteLLM's polled response object takes an integer created_at only.
    final = {**_openai_response(), "created_at": 1741476542.5, "extra": [1, None]}
    api = ScriptedAPI((200, _openai_response("queued")), (200, final))
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    response = await _complete(_provider("openai"))
    assert api.calls("GET") == 1
    assert response.text == '{"value": 3}'
    assert response.evidence["response"] == final


async def test_background_runs_go_through_litellms_responses_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm_provider, "POLL_SECONDS", 0.01)
    calls: list[tuple[str, dict[str, Any]]] = []
    for name in RESPONSES_API:

        def spy(*args: Any, _name: str = name, **kwargs: Any) -> Any:
            calls.append((_name, kwargs))
            return getattr(litellm, _name)(*args, **kwargs)

        monkeypatch.setattr(litellm_provider, name, spy)
    # The first run reaches its deadline and is cancelled; the second answers.
    api = _Runs(["queued", "in_progress"], ["queued", "completed"])
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    response = await _complete(_provider("openai", "gpt-6-luna"), timeout=0.5)
    assert response.provider_request_id == "resp_2"
    names = [name for name, _ in calls]
    assert names[0] == names[-2] == "aresponses"
    assert names.count("aresponses") == 2
    assert names.count("acancel_responses") == 1
    assert names[-1] == "aget_responses"
    submit = calls[0][1]
    assert submit["model"] == "openai/gpt-6-luna"
    assert (submit["background"], submit["store"]) == (True, True)
    assert submit["custom_llm_provider"] == "openai"
    assert submit["num_retries"] == 0
    assert submit["api_key"] == "fixture-key"
    assert isinstance(submit["client"], AsyncHTTPHandler)
    polls = [kwargs for name, kwargs in calls if name == "aget_responses"]
    assert {kwargs["response_id"] for kwargs in polls} == {"resp_1", "resp_2"}
    assert {kwargs["custom_llm_provider"] for kwargs in polls} == {"openai"}
    [cancel] = [kwargs for name, kwargs in calls if name == "acancel_responses"]
    assert (cancel["response_id"], cancel["custom_llm_provider"]) == (
        "resp_1",
        "openai",
    )


async def test_background_poll_errors_are_retried_without_resubmitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # More consecutive failures than a submission may have: nothing is
    # generated for the call when a poll fails, so polls retry until the
    # deadline.
    api = ScriptedAPI(
        (200, _openai_response("queued")),
        httpx.RemoteProtocolError("tunnel drained"),
        *[(502, {"error": "bad gateway"})] * TRANSIENT_ATTEMPTS,
        (429, {"error": "slow down"}),
        (200, _openai_response()),
    )
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    heard: list[None] = []
    with listen_for_rate_limits(lambda: heard.append(None)):
        response = await _complete(_provider("openai"))
    assert response.text == '{"value": 3}'
    assert api.calls("POST") == 1
    assert api.calls("GET") == TRANSIENT_ATTEMPTS + 3
    assert len(heard) == 1


def test_backoff_doubles_to_its_cap_and_honours_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm_provider, "_BACKOFF_SECONDS", 1.0)

    def refused(headers: dict[str, str]) -> litellm.RateLimitError:
        request = httpx.Request("POST", "http://localhost")
        reply = httpx.Response(429, headers=headers, request=request)
        return litellm.RateLimitError("slow down", "openai", "gpt", response=reply)

    error = litellm.ServiceUnavailableError("overloaded", "openai", "gpt")
    delays = [backoff_seconds(failures, error) for failures in range(1, 12)]
    assert all(0.5 <= delay <= 1.0 for delay in delays[:1])
    assert all(15.0 <= delay <= 30.0 for delay in delays[6:])
    assert backoff_seconds(1, refused({"Retry-After": "45"})) == 45.0
    later = (datetime.now(UTC) + timedelta(seconds=90)).strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )
    assert 80.0 <= backoff_seconds(1, refused({"retry-after": later})) <= 90.0


async def test_a_retry_after_header_reaches_the_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(litellm_provider.asyncio, "sleep", sleep)
    api = ScriptedAPI((429, {"error": "slow down"}), (200, _openai_response()))
    api_handle = api.handle

    def handle(request: httpx.Request) -> httpx.Response:
        response = api_handle(request)
        if response.status_code == 429:
            response.headers["Retry-After"] = "45"
        return response

    handler = InMemoryHandler(httpx.MockTransport(handle))
    monkeypatch.setattr(litellm_provider, "http_client", lambda _name: handler)
    response = await _complete(_provider("openai"))
    assert response.text == '{"value": 3}'
    assert waits == [45.0]


def _run(run_id: str, status: str, code: str | None = None) -> dict[str, Any]:
    """A background run's response with *run_id*, failed with *code* if given."""
    response = {**_openai_response(status), "id": run_id}
    if code is not None:
        response.update(
            output=[], usage=None, error={"code": code, "message": "fixture error"}
        )
    return response


@pytest.mark.parametrize(
    ("status", "code"),
    [
        ("failed", "vector_store_timeout"),
        ("failed", None),
        # A transient code fails the call unless the run itself failed.
        ("cancelled", "server_error"),
        ("expired", "server_error"),
    ],
)
async def test_a_background_run_that_ends_otherwise_fails_the_call_at_once(
    monkeypatch: pytest.MonkeyPatch, status: str, code: str | None
) -> None:
    ended = {**_openai_response(status), "error": code and {"code": code}}
    api = ScriptedAPI((200, _openai_response("queued")), (200, ended))
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    with pytest.raises(ProviderError, match=f"response ended {status}") as caught:
        await _complete(_provider("openai"))
    assert not caught.value.transient
    assert caught.value.evidence is not None
    assert caught.value.evidence["response"] == ended
    assert caught.value.evidence["connection_lost"] == []
    assert api.calls("POST", "/v1/responses") == 1
    assert api.calls("POST", "/v1/responses/resp_fixture/cancel") == 0
    assert transport._OPEN_RUNS == {}


@pytest.mark.parametrize(
    ("code", "rate_limited"), [("server_error", 0), ("rate_limit_exceeded", 1)]
)
async def test_a_background_run_that_failed_transiently_is_sent_again_and_recorded(
    monkeypatch: pytest.MonkeyPatch, code: str, rate_limited: int
) -> None:
    failed = _run("resp_1", "failed", code)
    api = ScriptedAPI(
        (200, _run("resp_1", "queued")),
        (200, _run("resp_1", "in_progress")),
        (200, failed),
        (200, _run("resp_2", "queued")),
        (200, _run("resp_2", "completed")),
    )
    # The runs open when each request is sent: the failed run is closed and
    # the new one opened.
    open_at_request: list[list[str]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        open_at_request.append(sorted(transport._OPEN_RUNS))
        return api.handle(request)

    handler = InMemoryHandler(httpx.MockTransport(handle))
    monkeypatch.setattr(litellm_provider, "http_client", lambda _name: handler)
    heard: list[None] = []
    with listen_for_rate_limits(lambda: heard.append(None)):
        response = await _complete(_provider("openai"))
    assert [(r["method"], r["path"]) for r in api.requests] == [
        ("POST", "/v1/responses"),
        ("GET", "/v1/responses/resp_1"),
        ("GET", "/v1/responses/resp_1"),
        ("POST", "/v1/responses"),
        ("GET", "/v1/responses/resp_2"),
    ]
    assert api.requests[3]["json"] == api.requests[0]["json"]
    assert open_at_request == [
        [],
        ["openai:resp_1"],
        ["openai:resp_1"],
        [],
        ["openai:resp_2"],
    ]
    assert transport._OPEN_RUNS == {}
    assert len(heard) == rate_limited
    assert response.text == '{"value": 3}'
    assert response.stop_reason == "completed"
    assert response.provider_request_id == "resp_2"
    evidence = response.evidence
    assert evidence["run_id"] == "resp_2"
    assert evidence["response"] == _run("resp_2", "completed")
    assert list(evidence["status_first_seen_ms"]) == ["queued", "completed"]
    [lost] = evidence["connection_lost"]
    assert lost["cause"] == code
    assert lost["run_id"] == "resp_1"
    assert lost["error"] == failed["error"]
    assert isinstance(lost["elapsed_ms"], int)
    assert list(lost["status_first_seen_ms"]) == ["queued", "in_progress", "failed"]
    json.dumps(evidence)


async def test_background_resends_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = [f"resp_{index}" for index in range(RESENDS + 1)]
    api = ScriptedAPI(
        *(
            reply
            for run_id in runs
            for reply in (
                (200, _run(run_id, "queued")),
                (200, _run(run_id, "failed", "server_error")),
            )
        )
    )
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    with pytest.raises(
        ProviderError, match="3 background runs lost; the last failed"
    ) as caught:
        await _complete(_provider("openai"))
    assert caught.value.transient
    assert api.calls("POST", "/v1/responses") == RESENDS + 1
    assert caught.value.evidence is not None
    lost = caught.value.evidence["connection_lost"]
    assert [row["run_id"] for row in lost] == runs
    assert caught.value.evidence["response"] == _run(runs[-1], "failed", "server_error")
    assert transport._OPEN_RUNS == {}


async def test_a_resent_background_run_has_its_own_queue_limit_and_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm_provider, "POLL_SECONDS", 0.03)
    # The lost run queues for 30 polls; the next queues for 30 and generates
    # for 30. Each phase is within the deadline only if measured from its own
    # run's submission. A poll's own exchange may take up to 20 ms.
    api = ScriptedAPI(
        (200, _run("resp_1", "queued")),
        *[(200, _run("resp_1", "queued"))] * 30,
        (200, _run("resp_1", "failed", "server_error")),
        (200, _run("resp_2", "queued")),
        *[(200, _run("resp_2", "queued"))] * 30,
        *[(200, _run("resp_2", "in_progress"))] * 30,
        (200, _run("resp_2", "completed")),
    )
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    response = await _complete(_provider("openai"), timeout=1.5)
    assert response.latency_ms > 2700
    assert response.evidence["connection_lost"][0]["elapsed_ms"] >= 900
    assert 900 <= response.evidence["status_first_seen_ms"]["in_progress"] < 1500


async def test_background_queue_limit_and_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm_provider, "POLL_SECONDS", 0.03)
    # Queued for 30 polls, then in progress for 30: longer in all than the
    # deadline, but neither phase is, so the answer arrives. A poll's own
    # exchange may take up to 20 ms.
    api = ScriptedAPI(
        (200, _openai_response("queued")),
        *[(200, _openai_response("queued"))] * 30,
        *[(200, _openai_response("in_progress"))] * 30,
        (200, _openai_response()),
    )
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    response = await _complete(_provider("openai"), timeout=1.5)
    assert response.latency_ms > 1500
    assert response.evidence["status_first_seen_ms"]["in_progress"] >= 900
    assert response.evidence["connection_lost"] == []


type _Status = str | tuple[str, str]


class _Runs:
    """A provider API whose background run ``resp_<n>``, the nth submitted,
    answers its submission and each poll with the next of the nth scripted
    statuses, the last repeated: a status, or ``failed`` with an error code.
    A cancel is answered with *cancel_status* after *cancel_delay* seconds.
    Each request is recorded with the open runs when it arrived."""

    def __init__(
        self,
        *runs: Sequence[_Status],
        cancel_delay: float = 0.0,
        cancel_status: int = 200,
    ) -> None:
        self.runs = [list(statuses) for statuses in runs]
        self.cancel_delay = cancel_delay
        self.cancel_status = cancel_status
        self.requests: list[tuple[str, str, list[str]]] = []
        self.submissions: list[Any] = []
        self.cancels: list[tuple[str | None, bytes]] = []

    def _reply(self, run: int, request: httpx.Request) -> httpx.Response:
        statuses = self.runs[run]
        status = statuses.pop(0) if len(statuses) > 1 else statuses[0]
        status, code = status if isinstance(status, tuple) else (status, None)
        body = _run(f"resp_{run + 1}", status, code)
        return httpx.Response(200, json=body, request=request)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1")
        self.requests.append((request.method, path, sorted(transport._OPEN_RUNS)))
        if path == "/responses":
            self.submissions.append(json.loads(request.content))
            if len(self.submissions) > len(self.runs):
                raise AssertionError("unscripted submission")
            return self._reply(len(self.submissions) - 1, request)
        run = int(path.split("/")[2].removeprefix("resp_")) - 1
        if path.endswith("/cancel"):
            self.cancels.append((request.headers.get("content-type"), request.content))
            await asyncio.sleep(self.cancel_delay)
            body = _run(f"resp_{run + 1}", "cancelled")
            return httpx.Response(self.cancel_status, json=body, request=request)
        return self._reply(run, request)

    def client_factory(self) -> Any:
        handler = InMemoryHandler(httpx.MockTransport(self.handle))
        return lambda _provider_name: handler

    def steps(self) -> list[tuple[str, str, list[str]]]:
        """The requests, each run of identical ones (polls) recorded once."""
        steps: list[tuple[str, str, list[str]]] = []
        for step in self.requests:
            if not steps or steps[-1] != step:
                steps.append(step)
        return steps


@pytest.mark.parametrize(
    ("statuses", "cause", "phase"),
    [
        (
            ["queued", "in_progress"],
            "deadline",
            "0.5s after it was first seen in progress",
        ),
        (["queued"], "queue_deadline", "within 0.5s of submission"),
    ],
)
async def test_a_background_run_at_its_limit_is_cancelled_and_sent_again(
    monkeypatch: pytest.MonkeyPatch,
    statuses: list[_Status],
    cause: str,
    phase: str,
) -> None:
    monkeypatch.setattr(litellm_provider, "POLL_SECONDS", 0.01)
    api = _Runs(statuses, ["queued", "completed"])
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    response = await _complete(_provider("openai"), timeout=0.5)
    # The stalled run is cancelled, with a JSON body, before the request is
    # submitted again; only the live run is ever open.
    assert api.steps() == [
        ("POST", "/responses", []),
        ("GET", "/responses/resp_1", ["openai:resp_1"]),
        ("POST", "/responses/resp_1/cancel", ["openai:resp_1"]),
        ("POST", "/responses", []),
        ("GET", "/responses/resp_2", ["openai:resp_2"]),
    ]
    assert api.cancels == [("application/json", b"{}")]
    assert api.submissions[1] == api.submissions[0]
    assert transport._OPEN_RUNS == {}
    assert response.text == '{"value": 3}'
    assert response.provider_request_id == "resp_2"
    evidence = response.evidence
    assert evidence["run_id"] == "resp_2"
    assert list(evidence["status_first_seen_ms"]) == ["queued", "completed"]
    [lost] = evidence["connection_lost"]
    assert lost["cause"] == cause
    assert lost["run_id"] == "resp_1"
    assert lost["elapsed_ms"] >= 500
    assert list(lost["status_first_seen_ms"]) == statuses
    assert lost["error"] == f"no response {phase}"
    assert lost["cancel_error"] is None
    json.dumps(evidence)


async def test_background_runs_at_their_limits_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm_provider, "POLL_SECONDS", 0.01)
    # A fourth run would answer, but only RESENDS re-sends are allowed.
    api = _Runs(
        ["queued", "in_progress"],
        ["queued"],
        ["in_progress"],
        ["queued", "completed"],
    )
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    with pytest.raises(
        ProviderTimeout,
        match=r"0\.5s after it was first seen in progress; 3 background runs lost",
    ) as caught:
        await _complete(_provider("openai"), timeout=0.5)
    assert isinstance(caught.value, TimeoutError)
    assert len(api.submissions) == RESENDS + 1
    assert [path for method, path, _ in api.steps() if method == "POST"] == [
        "/responses",
        "/responses/resp_1/cancel",
        "/responses",
        "/responses/resp_2/cancel",
        "/responses",
        "/responses/resp_3/cancel",
    ]
    assert transport._OPEN_RUNS == {}
    evidence = caught.value.evidence
    assert evidence is not None
    assert [(row["run_id"], row["cause"]) for row in evidence["connection_lost"]] == [
        ("resp_1", "deadline"),
        ("resp_2", "queue_deadline"),
        ("resp_3", "deadline"),
    ]
    assert evidence["run_id"] == "resp_3"
    assert "response" not in evidence


@pytest.mark.parametrize(
    ("runs", "causes", "ends"),
    [
        (
            [
                ["queued", ("failed", "server_error")],
                ["queued", "in_progress"],
                ["queued", "completed"],
            ],
            ["server_error", "deadline"],
            "answered",
        ),
        (
            [
                ["queued", ("failed", "server_error")],
                ["in_progress"],
                ["queued"],
                ["queued", "completed"],
            ],
            ["server_error", "deadline", "queue_deadline"],
            "timeout",
        ),
        (
            [
                ["queued"],
                ["in_progress"],
                ["queued", ("failed", "server_error")],
                ["queued", "completed"],
            ],
            ["queue_deadline", "deadline", "server_error"],
            "failed",
        ),
    ],
)
async def test_failed_runs_and_runs_at_their_limits_share_one_bound(
    monkeypatch: pytest.MonkeyPatch,
    runs: list[list[_Status]],
    causes: list[str],
    ends: str,
) -> None:
    monkeypatch.setattr(litellm_provider, "POLL_SECONDS", 0.01)
    api = _Runs(*runs)
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    if ends == "answered":
        response = await _complete(_provider("openai"), timeout=0.5)
        assert response.provider_request_id == "resp_3"
        evidence = response.evidence
        assert api.steps() == [
            ("POST", "/responses", []),
            ("GET", "/responses/resp_1", ["openai:resp_1"]),
            ("POST", "/responses", []),
            ("GET", "/responses/resp_2", ["openai:resp_2"]),
            ("POST", "/responses/resp_2/cancel", ["openai:resp_2"]),
            ("POST", "/responses", []),
            ("GET", "/responses/resp_3", ["openai:resp_3"]),
        ]
    else:
        # A bound per cause would send a fourth run, which would answer.
        with pytest.raises(ProviderError, match="3 background runs lost") as caught:
            await _complete(_provider("openai"), timeout=0.5)
        assert isinstance(caught.value, TimeoutError) == (ends == "timeout")
        assert caught.value.evidence is not None
        evidence = caught.value.evidence
    assert len(api.submissions) == RESENDS + 1
    assert [row["cause"] for row in evidence["connection_lost"]] == causes
    assert len(api.cancels) == causes.count("deadline") + causes.count("queue_deadline")
    assert transport._OPEN_RUNS == {}


@pytest.mark.parametrize(("delay", "status"), [(0.0, 500), (1.0, 200)])
async def test_a_run_whose_cancel_is_unconfirmed_stays_open_for_shutdown(
    monkeypatch: pytest.MonkeyPatch, delay: float, status: int
) -> None:
    monkeypatch.setattr(litellm_provider, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(litellm_provider, "_CANCEL_SECONDS", 0.1)
    api = _Runs(
        ["in_progress"],
        ["queued", "completed"],
        cancel_delay=delay,
        cancel_status=status,
    )
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    response = await _complete(_provider("openai"), timeout=0.5)
    assert response.provider_request_id == "resp_2"
    [lost] = response.evidence["connection_lost"]
    assert lost["cause"] == "deadline"
    assert lost["cancel_error"].startswith(
        "InternalServerError" if status == 500 else "TimeoutError"
    )
    assert list(transport._OPEN_RUNS) == ["openai:resp_1"]


async def test_the_hard_timeout_covers_every_run_and_its_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every run reaches its deadline and no cancel is answered within its
    # exchange's timeout: the runs' own limits, not the backstop, end the call.
    monkeypatch.setattr(litellm_provider, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(litellm_provider, "_EXCHANGE_TIMEOUT_SECONDS", 0.25)
    monkeypatch.setattr(litellm_provider, "_CANCEL_SECONDS", 0.5)
    api = _Runs(*[["in_progress"]] * (RESENDS + 1), cancel_delay=10.0)
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    with pytest.raises(ProviderTimeout, match="3 background runs lost"):
        await _complete(_provider("openai"), timeout=0.5)
    assert len(api.cancels) == RESENDS + 1
    assert sorted(transport._OPEN_RUNS) == [
        "openai:resp_1",
        "openai:resp_2",
        "openai:resp_3",
    ]


class _CancelLog:
    """A provider API whose background runs never finish and whose cancel
    reply takes a round trip; it logs each cancel sent and acknowledged, and
    each cancel's content type and body."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.cancels: list[tuple[str | None, bytes]] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/cancel"):
            # OpenAI answers 400 unsupported_content_type without a JSON body.
            self.cancels.append((request.headers.get("content-type"), request.content))
            self.events.append("cancel sent")
            await asyncio.sleep(0.05)
            self.events.append("cancel acknowledged")
            return httpx.Response(
                200, json=_openai_response("cancelled"), request=request
            )
        status = "queued" if request.method == "POST" else "in_progress"
        return httpx.Response(200, json=_openai_response(status), request=request)


@pytest.mark.parametrize("ending", ["deadline", "another call failed", "interrupt"])
def test_open_runs_are_cancelled_before_the_event_loop_closes(
    monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    # A plain asyncio.run, as collection uses: no test event loop outlives it,
    # so the cancel must be delivered before the loop closes.
    monkeypatch.setattr(litellm_provider, "POLL_SECONDS", 0.01)
    api = _CancelLog()
    handler = InMemoryHandler(httpx.MockTransport(api.handle))
    monkeypatch.setattr(litellm_provider, "http_client", lambda _name: handler)

    async def study() -> None:
        call = _complete(
            _provider("openai"), timeout=0.3 if ending == "deadline" else 60
        )
        if ending == "deadline":
            with pytest.raises(ProviderTimeout):
                await call
            return
        main = asyncio.current_task()
        assert main is not None

        async def stop_after_submission() -> None:
            await asyncio.sleep(0.3)
            if ending == "interrupt":
                main.cancel()  # what asyncio.run does on Ctrl-C
                await asyncio.sleep(1)
            raise ValueError("another call's provider failure")

        async with asyncio.TaskGroup() as group:
            group.create_task(call)
            group.create_task(stop_after_submission())

    def run() -> None:
        asyncio.run(transport.with_open_runs_cancelled(study()))

    if ending == "deadline":
        run()
    else:
        expected = asyncio.CancelledError if ending == "interrupt" else ExceptionGroup
        with pytest.raises(expected):
            run()
    # A run at its deadline is cancelled by its call, which sends the request
    # again; a run left by a stopped study is cancelled at shutdown.
    cancels = RESENDS + 1 if ending == "deadline" else 1
    assert api.events == ["cancel sent", "cancel acknowledged"] * cancels
    assert api.cancels == [("application/json", b"{}")] * cancels
    assert transport._OPEN_RUNS == {}


@pytest.mark.parametrize(
    ("final", "stop_reason"),
    [
        (
            _openai_response("incomplete", incomplete="max_output_tokens"),
            "output_limit",
        ),
        (_openai_response("incomplete", incomplete="content_filter"), "refusal"),
        (
            _openai_response(content=[{"type": "refusal", "refusal": "I can't"}]),
            "refusal",
        ),
        (_openai_response(), "completed"),
    ],
)
async def test_background_stop_reasons(
    monkeypatch: pytest.MonkeyPatch, final: dict[str, Any], stop_reason: str
) -> None:
    api = ScriptedAPI((200, final))
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    response = await _complete(_provider("openai"))
    assert response.stop_reason == stop_reason
    assert (
        response.evidence["response"]["incomplete_details"]
        == final["incomplete_details"]
    )


async def test_an_unknown_incomplete_reason_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = ScriptedAPI((200, _openai_response("incomplete", incomplete="unknown")))
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    with pytest.raises(ProviderError, match="response ended incomplete: unknown"):
        await _complete(_provider("openai"))


# ---- streams through LiteLLM -----------------------------------------------------


@asynccontextmanager
async def _serving(path: str, handler: Any) -> AsyncGenerator[str]:
    """A local server answering POST *path* with *handler*; yields its root."""
    app = web.Application()
    app.router.add_post(path, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}"
    finally:
        await runner.cleanup()


_OPENAI_COMPATIBLE_USAGE = {
    # xAI counts reasoning tokens apart from its completion tokens.
    "xai": {
        "completion_tokens": 7,
        "completion_tokens_details": {"reasoning_tokens": 100},
    },
    # Meta's OpenAI-compatible usage counts them within.
    "meta": {
        "completion_tokens": 107,
        "completion_tokens_details": {"reasoning_tokens": 100},
    },
}


# LiteLLM's OpenAI-compatible stream handler, xAI's and Meta's, reads every
# attribute of a chunk's empty delta, pydantic's deprecated instance
# attributes included.
@pytest.mark.filterwarnings(
    "ignore:Accessing the 'model_:pydantic.warnings.PydanticDeprecatedSince211"
)
@pytest.mark.parametrize(
    ("provider_name", "model"), [("xai", "grok-4.7"), ("meta", "muse-spark-1.2")]
)
@pytest.mark.parametrize(
    ("finish_reason", "stop_reason"),
    [
        ("stop", "completed"),
        ("end_turn", "completed"),
        ("length", "output_limit"),
        ("content_filter", "refusal"),
    ],
)
async def test_an_openai_compatible_stream_through_litellm(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    model: str,
    finish_reason: str,
    stop_reason: str,
) -> None:
    """LiteLLM's own xAI and Meta stream paths, against a local server playing
    the provider's streamed chat completion: each sends the model's reasoning
    effort, aggregates the text and usage, and maps each finish reason."""
    monkeypatch.setattr(litellm_provider, "acompletion", litellm.acompletion)
    received: list[tuple[dict[str, Any], str | None]] = []

    async def chat(request: web.Request) -> web.StreamResponse:
        received.append((await request.json(), request.headers.get("authorization")))
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        base = {
            "id": "chatcmpl-fixture",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
        }

        async def send(**fields: Any) -> None:
            await response.write(f"data: {json.dumps({**base, **fields})}\n\n".encode())

        for delta, reason in (
            ({"role": "assistant", "content": ""}, None),
            ({"reasoning_content": "thinking"}, None),
            ({"content": '{"value": '}, None),
            ({"content": "3}"}, None),
            ({}, finish_reason),
        ):
            await send(choices=[{"index": 0, "delta": delta, "finish_reason": reason}])
        usage = {
            "prompt_tokens": 19,
            "total_tokens": 126,
            "prompt_tokens_details": {"cached_tokens": 5},
            **_OPENAI_COMPATIBLE_USAGE[provider_name],
        }
        await send(choices=[], usage=usage)
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async with _serving("/v1/chat/completions", chat) as root:
        provider = LiteLLMProvider(
            api_key="fixture-key",
            provider_name=cast(ProviderName, provider_name),
            model=model,
            max_concurrent=1,
            reasoning_effort="high",
            api_base=f"{root}/v1",
        )
        response = await _complete(provider, max_tokens=123)
    [(body, authorization)] = received
    assert authorization == "Bearer fixture-key"
    assert body["model"] == model
    assert body["reasoning_effort"] == "high"
    assert body["max_tokens"] == 123
    assert (body["stream"], body["stream_options"]) == (True, {"include_usage": True})
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert response.text == '{"value": 3}'
    assert response.stop_reason == stop_reason
    assert response.provider_request_id == "chatcmpl-fixture"
    # Output tokens include reasoning: LiteLLM folds in xAI's separate count.
    assert (response.input_tokens, response.output_tokens) == (14, 107)
    assert response.evidence["mechanism"] == "stream"
    assert response.evidence["connection_lost"] == []
    json.dumps(response.evidence)


@pytest.mark.parametrize(
    ("message", "refused"),
    [
        (
            "Content violates usage guidelines. Team: t, API key ID: k, "
            "Model: grok-4.7, Failed check: SAFETY_CHECK_TYPE_BIO",
            True,
        ),
        ("Team t does not have permission to access model grok-4.7", False),
    ],
)
async def test_an_xai_403_through_litellm(
    monkeypatch: pytest.MonkeyPatch, message: str, refused: bool
) -> None:
    """A request xAI's safety checks reject is reported to be answered with
    a 403 naming its usage guidelines (xAI's documentation does not say):
    through LiteLLM's own xAI path that is a refusal of the call, while a 403
    for a missing permission stays systematic."""
    monkeypatch.setattr(litellm_provider, "acompletion", litellm.acompletion)
    sent: list[None] = []

    async def chat(request: web.Request) -> web.Response:
        sent.append(None)
        return web.json_response({"code": _XAI_DENIED, "error": message}, status=403)

    async with _serving("/v1/chat/completions", chat) as root:
        provider = LiteLLMProvider(
            api_key="fixture-key",
            provider_name="xai",
            model="grok-4.7",
            max_concurrent=1,
            reasoning_effort="high",
            api_base=f"{root}/v1",
        )
        if refused:
            response = await _complete(provider)
            assert (response.text, response.stop_reason) == ("", "refusal")
            rejection = response.evidence["policy_rejection"]
            assert (rejection["status_code"], rejection["code"]) == (403, _XAI_DENIED)
            assert "Content violates usage guidelines" in rejection["error"]
        else:
            with pytest.raises(ProviderError) as caught:
                await _complete(provider)
            assert not caught.value.transient
    assert len(sent) == 1


_GEMINI_USAGE = {
    "promptTokenCount": 19,
    "candidatesTokenCount": 7,
    "thoughtsTokenCount": 100,
    "totalTokenCount": 126,
    "cachedContentTokenCount": 5,
}


def _gemini_chunk(text: str, finish_reason: str | None = None) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "content": {"parts": [{"text": text}], "role": "model"},
        "index": 0,
    }
    if finish_reason is not None:
        candidate["finishReason"] = finish_reason
    return {
        "candidates": [candidate],
        "usageMetadata": _GEMINI_USAGE,
        "responseId": "r1",
    }


class _Gemini:
    """Gemini's streamed generateContent, answering each request with the next
    scripted reply: the streamed chunks, the chunks and then a lost connection
    (``("cut", chunks)``), or an error status."""

    def __init__(
        self, *replies: list[dict[str, Any]] | tuple[str, list[dict[str, Any]]] | int
    ) -> None:
        self.replies = list(replies)
        self.requests: list[tuple[str, dict[str, Any]]] = []

    async def handle(self, request: web.Request) -> web.StreamResponse:
        self.requests.append((str(request.rel_url), await request.json()))
        reply = self.replies.pop(0)
        if isinstance(reply, int):
            error = {"code": reply, "message": "fixture", "status": "UNAVAILABLE"}
            return web.json_response(
                {"error": error}, status=reply, headers={"Retry-After": "0"}
            )
        events = reply[1] if isinstance(reply, tuple) else reply
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        for event in events:
            await response.write(f"data: {json.dumps(event)}\r\n\r\n".encode())
        if isinstance(reply, tuple) and request.transport is not None:
            request.transport.close()
            return response
        await response.write_eof()
        return response


def _google(root: str) -> LiteLLMProvider:
    return LiteLLMProvider(
        api_key="fixture-key",
        provider_name="google",
        model="gemini-3.1-pro-preview",
        max_concurrent=1,
        reasoning_effort="high",
        api_base=root,
    )


@pytest.mark.parametrize(
    ("chunks", "text", "stop_reason"),
    [
        (
            [_gemini_chunk('{"value": '), _gemini_chunk("3}", "STOP")],
            '{"value": 3}',
            "completed",
        ),
        (
            [_gemini_chunk('{"value": '), _gemini_chunk("3}", "MAX_TOKENS")],
            '{"value": 3}',
            "output_limit",
        ),
        # Gemini's safety stops are refusals.
        (
            [_gemini_chunk('{"value": '), _gemini_chunk("", "SAFETY")],
            '{"value": ',
            "refusal",
        ),
        (
            [_gemini_chunk("", "PROHIBITED_CONTENT")],
            "",
            "refusal",
        ),
        # So is a prompt Gemini blocks before answering.
        (
            [
                {
                    "promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
                    "usageMetadata": {"promptTokenCount": 19, "totalTokenCount": 19},
                    "responseId": "r1",
                }
            ],
            "",
            "refusal",
        ),
    ],
)
async def test_a_google_stream_through_litellm(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[dict[str, Any]],
    text: str,
    stop_reason: str,
) -> None:
    """LiteLLM's own Gemini stream path, against a local server playing
    Gemini's streamed generateContent: it maps the reasoning effort to a
    thinking level, aggregates the text and usage, and maps each finish
    reason, though LiteLLM records Gemini's finish reason apart from the
    others'."""
    monkeypatch.setattr(litellm_provider, "acompletion", litellm.acompletion)
    gemini = _Gemini(chunks)
    async with _serving("/{path:.*}", gemini.handle) as root:
        response = await _complete(_google(root), max_tokens=123)
    [(path, body)] = gemini.requests
    assert path.startswith("/models/gemini-3.1-pro-preview:streamGenerateContent")
    assert body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "high"
    assert body["generationConfig"]["max_output_tokens"] == 123
    assert (response.text, response.stop_reason) == (text, stop_reason)
    assert response.provider_request_id == "r1"
    if "promptFeedback" not in chunks[0]:
        # Output tokens include Gemini's thinking tokens.
        assert (response.input_tokens, response.output_tokens) == (14, 107)
    assert response.evidence["connection_lost"] == []
    json.dumps(response.evidence)


async def test_a_google_stream_is_opened_when_sent_and_its_cut_is_a_lost_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LiteLLM sends Gemini's request only when its stream is first read: a
    refusal for rate then is a stream that failed to open, retried and told to
    the lane, and a connection lost once the stream is open is a lost
    sample."""
    monkeypatch.setattr(litellm_provider, "acompletion", litellm.acompletion)
    answer = [_gemini_chunk('{"value": '), _gemini_chunk("3}", "STOP")]
    gemini = _Gemini(429, 503, ("cut", [_gemini_chunk('{"value": ')]), answer)
    heard: list[None] = []
    async with _serving("/{path:.*}", gemini.handle) as root:
        with listen_for_rate_limits(lambda: heard.append(None)):
            response = await _complete(_google(root))
    assert len(gemini.requests) == 4
    assert len(heard) == 1
    assert (response.text, response.stop_reason) == ('{"value": 3}', "completed")
    [lost] = response.evidence["connection_lost"]
    assert (lost["cause"], lost["chunks"]) == ("connection_error", 1)


@pytest.mark.parametrize("provider_name", STREAMED)
async def test_a_stream_that_fails_to_open_when_first_read_is_retried(
    monkeypatch: pytest.MonkeyPatch, provider_name: str
) -> None:
    acompletion = ScriptedAcompletion(
        FakeStream(
            [], open_error=litellm.RateLimitError("429", provider_name, "model")
        ),
        FakeStream(answer_chunks("{}")),
    )
    monkeypatch.setattr(litellm_provider, "acompletion", acompletion)
    heard: list[None] = []
    with listen_for_rate_limits(lambda: heard.append(None)):
        response = await _complete(_provider(provider_name))
    assert response.text == "{}"
    assert len(acompletion.calls) == 2
    assert response.evidence["connection_lost"] == []
    assert len(heard) == 1


@pytest.mark.parametrize("provider_name", sorted(PROVIDERS))
async def test_every_provider_is_refused_a_live_call(provider_name: str) -> None:
    # The conftest guards refuse each entry point the transport calls through.
    with pytest.raises(ProviderError, match="tests never call a live provider"):
        await _complete(_provider(provider_name))


# ---- policy rejections ---------------------------------------------------------

_FLAGGED = (
    "Invalid prompt: your prompt was flagged as potentially violating our usage "
    "policy. Please try again with a different prompt."
)
_XAI_DENIED = "The caller does not have permission to execute the specified operation"
"""The ``code`` of xAI's 403 replies, a safety check's and a permission's alike."""


def _xai_403(message: str) -> litellm.APIError:
    """An xAI 403 as LiteLLM raises it, quoting xAI's reply."""
    reply = json.dumps({"code": _XAI_DENIED, "error": message})
    return litellm.APIError(403, f"APIError: XaiException - {reply}", "xai", "grok")


@pytest.mark.parametrize(
    ("message", "code", "classed"),
    [
        # LiteLLM's class for OpenAI's flagged prompt.
        (_FLAGGED, "invalid_prompt", "ContentPolicyViolationError"),
        # The provider's error code alone makes a rejection.
        ("rejected", "invalid_prompt", "BadRequestError"),
        ("rejected", "content_policy_violation", "ContentPolicyViolationError"),
    ],
)
async def test_a_submission_rejected_for_policy_is_a_refusal(
    monkeypatch: pytest.MonkeyPatch, message: str, code: str, classed: str
) -> None:
    error = {
        "message": message,
        "type": "invalid_request_error",
        "param": None,
        "code": code,
    }
    api = ScriptedAPI((400, {"error": error}))
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    response = await _complete(_provider("openai"))
    assert (response.text, response.stop_reason) == ("", "refusal")
    assert response.provider_request_id is None
    rejection = response.evidence["policy_rejection"]
    assert (rejection["status_code"], rejection["code"]) == (400, code)
    assert rejection["error"].startswith(classed)
    assert len(api.requests) == 1
    assert "run_id" not in response.evidence
    assert "response" not in response.evidence
    assert transport._OPEN_RUNS == {}
    json.dumps(response.evidence)


async def test_a_background_run_failed_for_policy_is_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = _run("resp_1", "failed", "invalid_prompt")
    api = ScriptedAPI((200, _run("resp_1", "queued")), (200, failed))
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    response = await _complete(_provider("openai"))
    assert (response.text, response.stop_reason) == ("", "refusal")
    assert response.provider_request_id == "resp_1"
    evidence = response.evidence
    assert evidence["response"] == failed
    assert evidence["policy_rejection"] == {
        "status_code": None,
        "code": "invalid_prompt",
        "error": "fixture error",
    }
    # A refusal is not a lost sample: the request is not sent again.
    assert evidence["connection_lost"] == []
    assert api.calls("POST", "/v1/responses") == 1
    assert transport._OPEN_RUNS == {}


@pytest.mark.parametrize("provider_name", STREAMED)
@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (
            litellm.ContentPolicyViolationError(
                "Output blocked by content filtering policy", "model", "anthropic"
            ),
            400,
            None,
        ),
        (
            litellm.BadRequestError(
                'XaiException - {"error": {"message": "rejected", '
                '"type": "invalid_request_error", "code": "invalid_prompt"}}',
                "model",
                "xai",
            ),
            400,
            "invalid_prompt",
        ),
        # xAI's safety checks, as reported: a 403 naming the guidelines.
        (
            _xai_403(
                "Content violates usage guidelines. Team: t, API key ID: k, "
                "Model: grok-4.7, Failed check: SAFETY_CHECK_TYPE_BIO"
            ),
            403,
            _XAI_DENIED,
        ),
        (_xai_403("CONTENT VIOLATES USAGE GUIDELINES"), 403, _XAI_DENIED),
    ],
)
async def test_a_stream_rejected_for_policy_is_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    error: Exception,
    status: int,
    code: str | None,
) -> None:
    acompletion = ScriptedAcompletion(error)
    monkeypatch.setattr(litellm_provider, "acompletion", acompletion)
    response = await _complete(_provider(provider_name))
    assert (response.text, response.stop_reason) == ("", "refusal")
    assert len(acompletion.calls) == 1
    rejection = response.evidence["policy_rejection"]
    assert (rejection["status_code"], rejection["code"]) == (status, code)
    assert response.evidence["connection_lost"] == []


async def test_a_policy_rejection_mid_stream_is_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = MidStreamFallbackError(
        "blocked",
        "claude",
        "anthropic",
        original_exception=litellm.ContentPolicyViolationError(
            "Output blocked by content filtering policy", "claude", "anthropic"
        ),
    )
    acompletion = ScriptedAcompletion(FakeStream([chunk("partial")], error=blocked))
    monkeypatch.setattr(litellm_provider, "acompletion", acompletion)
    response = await _complete(_provider("anthropic"))
    assert (response.text, response.stop_reason) == ("", "refusal")
    assert len(acompletion.calls) == 1
    assert response.evidence["policy_rejection"]["error"].startswith(
        "ContentPolicyViolationError"
    )
    assert response.evidence["connection_lost"] == []


# ---- retries ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (
            400,
            {
                "message": "Unsupported parameter: 'top_k'",
                "type": "invalid_request_error",
                "param": "top_k",
                "code": "unsupported_parameter",
            },
        ),
        (
            400,
            {
                "message": "The requested model does not exist",
                "type": "invalid_request_error",
                "param": None,
                "code": "model_not_found",
            },
        ),
        (401, {"message": "Incorrect API key provided", "code": "invalid_api_key"}),
        (403, {"message": "Project does not have access", "code": None}),
        (404, {"message": "Not found", "code": None}),
    ],
)
async def test_other_client_errors_stay_systematic(
    monkeypatch: pytest.MonkeyPatch, status: int, error: dict[str, Any]
) -> None:
    api = ScriptedAPI((status, {"error": error}))
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    with pytest.raises(ProviderError) as caught:
        await _complete(_provider("openai"))
    assert not caught.value.transient and not is_transient(caught.value)
    assert len(api.requests) == 1
    assert caught.value.evidence is not None
    assert "policy_rejection" not in caught.value.evidence


@pytest.mark.parametrize("provider_name", STREAMED)
@pytest.mark.parametrize(
    "error",
    [
        litellm.BadRequestError("max_tokens: too large", "model", "anthropic"),
        litellm.BadRequestError(
            'XaiException - {"code": "Client specified an invalid argument", '
            '"error": "Model grok-9 does not exist"}',
            "model",
            "xai",
        ),
        litellm.AuthenticationError("bad key", "anthropic", "model"),
        litellm.PermissionDeniedError(
            "forbidden",
            "anthropic",
            "model",
            httpx.Response(403, request=httpx.Request("POST", "http://localhost")),
        ),
        litellm.NotFoundError("no such model", "model", "anthropic"),
        # A 403 for a missing permission names no usage guidelines.
        _xai_403("Team t does not have permission to access model grok-4.7"),
    ],
)
async def test_stream_client_errors_are_never_retried(
    monkeypatch: pytest.MonkeyPatch, provider_name: str, error: Exception
) -> None:
    acompletion = ScriptedAcompletion(error)
    monkeypatch.setattr(litellm_provider, "acompletion", acompletion)
    with pytest.raises(ProviderError) as caught:
        await _complete(_provider(provider_name))
    assert not caught.value.transient
    assert len(acompletion.calls) == 1


async def test_submissions_are_retried_a_bounded_number_of_times_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = ScriptedAPI(after=(503, {"error": "overloaded"}))
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    with pytest.raises(ProviderError, match="ServiceUnavailableError") as caught:
        await _complete(_provider("openai"))
    # Collection fails only this call's group and sends it again on resume.
    assert caught.value.transient and is_transient(caught.value)
    assert len(api.requests) == TRANSIENT_ATTEMPTS
    assert caught.value.evidence is not None
    assert len(caught.value.evidence["submit_failures"]) == TRANSIENT_ATTEMPTS

    # A client error is systematic: tried once, it stops the study.
    api = ScriptedAPI(after=(401, {"error": "invalid key"}))
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    with pytest.raises(ProviderError, match="AuthenticationError") as caught:
        await _complete(_provider("openai"))
    assert not caught.value.transient and not is_transient(caught.value)
    assert len(api.requests) == 1

    # A lost reply may hide an accepted run: every failed submission is kept.
    api = ScriptedAPI(
        httpx.ReadTimeout("no reply"), (429, None), (200, _openai_response())
    )
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    response = await _complete(_provider("openai"))
    assert response.text == '{"value": 3}'
    assert api.calls("POST") == 3
    failures = response.evidence["submit_failures"]
    assert [failure["error"].split(":")[0] for failure in failures] == [
        "Timeout",
        "RateLimitError",
    ]
    assert all(isinstance(failure["elapsed_ms"], int) for failure in failures)


async def test_a_lost_connection_is_sent_again_on_a_new_one_by_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # LiteLLM's HTTP client re-sends a request once on a new connection when
    # its connection fails, inside one of the transport's tries.
    api = ScriptedAPI(httpx.ConnectError("refused"), (200, _openai_response()))
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    response = await _complete(_provider("openai"))
    assert response.text == '{"value": 3}'
    assert api.calls("POST") == 2
    assert response.evidence["submit_failures"] == []


async def test_submissions_refused_for_rate_are_retried_until_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A 429 created no run, so it does not use up the submission's tries;
    # each is reported to the lane, and a failure that may have created a run
    # still does.
    refused = [(429, {"error": "rate limited"})] * (2 * TRANSIENT_ATTEMPTS)
    api = ScriptedAPI(*refused, (503, None), (200, _openai_response()))
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    heard: list[None] = []
    with listen_for_rate_limits(lambda: heard.append(None)):
        response = await _complete(_provider("openai"))
    assert response.text == '{"value": 3}'
    assert api.calls("POST") == 2 * TRANSIENT_ATTEMPTS + 2
    assert len(heard) == 2 * TRANSIENT_ATTEMPTS

    # A submission refused until the queue limit created no run to send again.
    api = ScriptedAPI(after=(429, {"error": "rate limited"}))
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    with pytest.raises(ProviderTimeout, match=r"within 0\.5s of submission$"):
        await _complete(_provider("openai"), timeout=0.5)
    assert api.calls("POST") > TRANSIENT_ATTEMPTS


@pytest.mark.parametrize("provider_name", STREAMED)
async def test_streams_refused_for_rate_are_retried_until_the_deadline(
    monkeypatch: pytest.MonkeyPatch, provider_name: str
) -> None:
    opened: list[None] = []

    async def refused(**_kwargs: Any) -> Any:
        opened.append(None)
        await asyncio.sleep(0.01)
        raise litellm.RateLimitError("429", provider_name, "model")

    monkeypatch.setattr(litellm_provider, "acompletion", refused)
    with pytest.raises(ProviderTimeout, match=r"no response within 0\.2s"):
        await _complete(_provider(provider_name), timeout=0.2)
    assert len(opened) > TRANSIENT_ATTEMPTS


# ---- streams ------------------------------------------------------------------


@pytest.mark.parametrize("provider_name", STREAMED)
async def test_stream_is_aggregated_with_usage_and_evidence(
    monkeypatch: pytest.MonkeyPatch, provider_name: str
) -> None:
    chunks = [
        chunk('{"value": '),
        chunk("3}", finish_reason="stop"),
        chunk(
            usage={
                "prompt_tokens": 19,
                "completion_tokens": 7,
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 2,
            }
        ),
    ]
    monkeypatch.setattr(
        litellm_provider, "acompletion", ScriptedAcompletion(FakeStream(chunks))
    )
    response = await _complete(_provider(provider_name))
    assert response.text == '{"value": 3}'
    assert response.stop_reason == "completed"
    assert (response.input_tokens, response.output_tokens) == (12, 7)
    assert response.provider_request_id == "msg-fixture"
    assert response.evidence["mechanism"] == "stream"
    assert response.evidence["connection_lost"] == []
    final = response.evidence["response"]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["usage"]["completion_tokens"] == 7
    json.dumps(response.evidence)


@pytest.mark.parametrize("provider_name", STREAMED)
@pytest.mark.parametrize(
    ("finish_reason", "stop_reason"),
    [("length", "output_limit"), ("content_filter", "refusal"), ("stop", "completed")],
)
async def test_stream_stop_reasons(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    finish_reason: str,
    stop_reason: str,
) -> None:
    stream = FakeStream(answer_chunks('{"value"', finish_reason=finish_reason))
    monkeypatch.setattr(litellm_provider, "acompletion", ScriptedAcompletion(stream))
    assert (await _complete(_provider(provider_name))).stop_reason == stop_reason


@pytest.mark.parametrize("provider_name", STREAMED)
async def test_a_cut_stream_is_sent_again_and_recorded(
    monkeypatch: pytest.MonkeyPatch, provider_name: str
) -> None:
    cut = litellm.ServiceUnavailableError(
        "Response payload is not completed", provider_name, "model"
    )
    acompletion = ScriptedAcompletion(
        FakeStream([chunk('{"value": ')], error=cut),
        FakeStream(answer_chunks('{"value": 3}')),
    )
    monkeypatch.setattr(litellm_provider, "acompletion", acompletion)
    response = await _complete(_provider(provider_name))
    assert response.text == '{"value": 3}'
    assert len(acompletion.calls) == 2
    assert acompletion.calls[1] == acompletion.calls[0]
    [lost] = response.evidence["connection_lost"]
    assert lost["chunks"] == 1
    assert lost["cause"] == "server_error"
    assert "ServiceUnavailableError" in lost["error"]
    assert isinstance(lost["elapsed_ms"], int)


@pytest.mark.parametrize(
    ("error", "cause"),
    [
        (httpx.ReadError("reset"), "connection_error"),
        (
            MidStreamFallbackError(
                "payload not completed",
                "claude",
                "anthropic",
                original_exception=litellm.APIConnectionError(
                    "payload not completed", "anthropic", "claude"
                ),
            ),
            "connection_error",
        ),
        (
            MidStreamFallbackError(
                "overloaded",
                "claude",
                "anthropic",
                original_exception=litellm.InternalServerError(
                    "overloaded", "anthropic", "claude"
                ),
            ),
            "server_error",
        ),
    ],
)
async def test_a_lost_stream_records_its_cause(
    monkeypatch: pytest.MonkeyPatch, error: Exception, cause: str
) -> None:
    acompletion = ScriptedAcompletion(
        FakeStream([chunk("partial")], error=error), FakeStream(answer_chunks("{}"))
    )
    monkeypatch.setattr(litellm_provider, "acompletion", acompletion)
    response = await _complete(_provider("anthropic"))
    assert [lost["cause"] for lost in response.evidence["connection_lost"]] == [cause]


async def test_a_stream_without_the_providers_finish_reason_was_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # LiteLLM closes a stream that ended early with a "stop" of its own.
    truncated = FakeStream(
        answer_chunks('{"value": ', finish_reason="stop"), provider_finished=False
    )
    acompletion = ScriptedAcompletion(truncated, FakeStream(answer_chunks("{}")))
    monkeypatch.setattr(litellm_provider, "acompletion", acompletion)
    response = await _complete(_provider("anthropic"))
    assert response.text == "{}"
    [lost] = response.evidence["connection_lost"]
    assert lost["error"].startswith("StreamCut")
    assert lost["cause"] == "ended_early"


async def test_stream_resends_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    acompletion = ScriptedAcompletion(
        *(
            FakeStream([chunk("partial")], error=httpx.ReadError("drained"))
            for _ in range(RESENDS + 2)
        )
    )
    monkeypatch.setattr(litellm_provider, "acompletion", acompletion)
    with pytest.raises(ProviderError, match="stream cut 3 times") as caught:
        await _complete(_provider("anthropic"))
    assert caught.value.transient
    assert len(acompletion.calls) == RESENDS + 1
    assert caught.value.evidence is not None
    assert len(caught.value.evidence["connection_lost"]) == RESENDS + 1


async def test_failures_before_the_stream_opens_are_retries_not_new_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # More consecutive failures than a submission may have: they are retried
    # until the deadline.
    acompletion = ScriptedAcompletion(
        litellm.RateLimitError("429", "anthropic", "claude"),
        *[litellm.InternalServerError("529 overloaded", "anthropic", "claude")]
        * TRANSIENT_ATTEMPTS,
        httpx.ConnectError("connection refused"),
        FakeStream(answer_chunks("{}")),
    )
    monkeypatch.setattr(litellm_provider, "acompletion", acompletion)
    heard: list[None] = []
    with listen_for_rate_limits(lambda: heard.append(None)):
        response = await _complete(_provider("anthropic"))
    assert response.text == "{}"
    assert len(acompletion.calls) == TRANSIENT_ATTEMPTS + 3
    assert response.evidence["connection_lost"] == []
    assert len(heard) == 1


async def test_a_stream_cut_before_its_first_chunk_is_a_lost_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An open stream may run for minutes before LiteLLM yields a chunk.
    acompletion = ScriptedAcompletion(
        FakeStream([], error=httpx.ReadError("tunnel drained")),
        FakeStream(answer_chunks("{}")),
    )
    monkeypatch.setattr(litellm_provider, "acompletion", acompletion)
    response = await _complete(_provider("anthropic"))
    [lost] = response.evidence["connection_lost"]
    assert lost["chunks"] == 0


# ---- shared behavior ------------------------------------------------------------


async def test_blank_completions_are_successful(monkeypatch: pytest.MonkeyPatch):
    api = ScriptedAPI((200, {**_openai_response(), "output": []}))
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    assert (await _complete(_provider("openai"))).text == ""
    monkeypatch.setattr(
        litellm_provider,
        "acompletion",
        ScriptedAcompletion(FakeStream([chunk(finish_reason="stop")])),
    )
    assert (await _complete(_provider("anthropic"))).text == ""


@pytest.mark.parametrize("max_concurrent", [0, -1, True, "1"])
def test_provider_requires_a_positive_integer_limit(max_concurrent: object) -> None:
    with pytest.raises(ValueError, match="max_concurrent must be a positive integer"):
        LiteLLMProvider(
            api_key="fixture-key",
            provider_name="openai",
            model="gpt-test",
            max_concurrent=cast(Any, max_concurrent),
            reasoning_effort="low",
        )


def test_provider_rejects_unknown_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="reasoning_effort must be one of"):
        LiteLLMProvider(
            api_key="fixture-key",
            provider_name="openai",
            model="gpt-test",
            max_concurrent=1,
            reasoning_effort=cast(Any, "extreme"),
        )


def test_adapter_import_restores_litellm_environment() -> None:
    names = (
        "LITELLM_LOCAL_MODEL_COST_MAP",
        "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS",
    )
    script = (
        "import os\n"
        "import catmodel.llm.litellm_provider\n"
        f"assert all(name not in os.environ for name in {names!r})\n"
    )

    unset_env = os.environ.copy()
    for name in names:
        unset_env.pop(name, None)
    unset = subprocess.run(
        [sys.executable, "-c", script],
        env=unset_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert unset.returncode == 0, unset.stderr

    preset_env = os.environ.copy()
    preset_env.update({name: "false" for name in names})
    preset_script = (
        "import os\n"
        "import catmodel.llm.litellm_provider\n"
        f"assert all(os.environ[name] == 'false' for name in {names!r})\n"
    )
    preset = subprocess.run(
        [sys.executable, "-c", preset_script],
        env=preset_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert preset.returncode == 0, preset.stderr


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        # A direct cache count takes precedence over the nested one, and the
        # uncached remainder never goes below zero.
        (
            {
                "input_tokens": 3,
                "cache_read_input_tokens": 5,
                "input_tokens_details": {"cached_tokens": 1},
                "output_tokens": 7,
            },
            (0, 7),
        ),
        ({"input_tokens": "many", "output_tokens": None}, (None, None)),
        (None, (None, None)),
    ],
)
async def test_usage_normalization(
    monkeypatch: pytest.MonkeyPatch, usage: Any, expected: tuple[Any, Any]
) -> None:
    api = ScriptedAPI((200, {**_openai_response(), "usage": usage}))
    monkeypatch.setattr(litellm_provider, "http_client", api.client_factory())
    response = await _complete(_provider("openai"))
    assert (response.input_tokens, response.output_tokens) == expected
