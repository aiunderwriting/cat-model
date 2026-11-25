"""Offline demonstration of each mechanism through a proxy-like local server.

The server plays the providers' APIs behind the two limits of the egress proxy
the transport was developed against, with time scaled so that one proxy second
takes ``SCALE`` real seconds:

- a response whose first byte would come more than 30 s after its request
  gets 502 at 30 s;
- an open streaming response is cut with a constant hazard, set so that a
  20-minute stream is cut with probability 0.167. The rate and the hazard
  form are assumptions of this demonstration.

Each call's answer takes 5 to 35 proxy minutes, drawn from a seed keyed by the
call and its send, so every call has the same history whatever the order.
LiteLLM's own Responses API calls, stream paths and HTTP client talk to the
server.

The longest answer ends 5 proxy minutes before the 40-minute deadline, however
often a background run is polled (its deadline starts when it is first seen in
progress). ``SCALE`` makes that margin 0.9 real seconds, and a poll every
``POLL`` proxy seconds keeps the 40 runs' polling light, so a loaded machine
cannot delay a run past its deadline.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
from collections import Counter
from typing import Any

import httpx
import litellm
import pytest
from aiohttp import web

from catmodel.llm import litellm_provider
from catmodel.llm.litellm_provider import LiteLLMProvider
from catmodel.llm.litellm_provider import http_client as litellm_http_client
from catmodel.llm.models import ProviderName
from catmodel.llm.transport import ProviderError, ProviderTimeout

SCALE = 0.003
"""Real seconds per proxy second."""
POLL = 120.0
"""Proxy seconds between polls of a background run."""
FIRST_BYTE_LIMIT = 30.0
CUT_HAZARD = -math.log(1 - 0.167) / 1200
"""Cuts per second of an open stream."""
DEADLINE = 2400.0
CALLS = 40


class ProxiedProviders:
    """The providers' APIs as seen through the proxy, in proxy seconds."""

    def __init__(self) -> None:
        self.sends: dict[str, int] = {}
        self.cuts: Counter[str] = Counter()
        self.runs: dict[str, dict[str, Any]] = {}
        self.app = web.Application()
        self.app.router.add_post("/v1/responses", self.openai_submit)
        self.app.router.add_get("/v1/responses/{id}", self.openai_poll)
        self.app.router.add_post("/v1/responses/{id}/cancel", self.openai_cancel)
        self.app.router.add_post("/v1/chat/completions", self.xai)
        self.app.router.add_post("/v1/messages", self.anthropic)

    def _answer(self, prompt: str) -> tuple[str, float, float]:
        """This send's answer, its duration and when the proxy would cut it."""
        send = self.sends[prompt] = self.sends.get(prompt, 0) + 1
        draw = random.Random(f"{prompt}/{send}")
        minutes = draw.uniform(5, 35)
        text = json.dumps({"call": prompt, "send": send, "minutes": round(minutes, 3)})
        return text, minutes * 60, draw.expovariate(CUT_HAZARD)

    async def _first_byte(self, ready_after: float) -> web.Response | None:
        if ready_after > FIRST_BYTE_LIMIT:
            await asyncio.sleep(FIRST_BYTE_LIMIT * SCALE)
            return web.Response(status=502, text="proxy: no first byte in 30 s")
        await asyncio.sleep(ready_after * SCALE)
        return None

    def _now(self) -> float:
        return asyncio.get_running_loop().time() / SCALE

    async def _cut(self, provider: str, request: web.Request, cut_at: float) -> None:
        """Cut an open stream *cut_at* proxy seconds in, unless the client
        gave up first."""
        await asyncio.sleep(cut_at * SCALE)
        if request.transport is not None:
            self.cuts[provider] += 1
            request.transport.close()

    def _run(self, run_id: str, status: str, text: str | None = None) -> dict:
        """A background run's response object as OpenAI returns it."""
        message = {
            "id": f"msg_{run_id}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "annotations": [], "text": text}],
        }
        usage = {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 30,
        }
        return {
            "id": run_id,
            "object": "response",
            "created_at": 1741476542,
            "status": status,
            "background": True,
            "incomplete_details": None,
            "output": [] if text is None else [message],
            "usage": None if text is None else usage,
        }

    async def openai_submit(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        text, seconds, _ = self._answer(body["input"])
        if not body.get("background"):
            return await self._first_byte(seconds) or web.json_response({})
        run_id = f"resp_{len(self.runs)}"
        self.runs[run_id] = {"due": self._now() + seconds, "text": text}
        return web.json_response(self._run(run_id, "queued"))

    async def openai_poll(self, request: web.Request) -> web.Response:
        run_id = request.match_info["id"]
        run = self.runs[run_id]
        if run.get("cancelled"):
            return web.json_response(self._run(run_id, "cancelled"))
        if self._now() < run["due"]:
            return web.json_response(self._run(run_id, "in_progress"))
        return web.json_response(self._run(run_id, "completed", run["text"]))

    async def openai_cancel(self, request: web.Request) -> web.Response:
        run_id = request.match_info["id"]
        self.runs[run_id]["cancelled"] = True
        return web.json_response(self._run(run_id, "cancelled"))

    async def xai(self, request: web.Request) -> web.StreamResponse:
        try:
            return await self._xai(request)
        except ConnectionResetError:  # the client gave up at its deadline
            return web.Response(status=499)

    async def _xai(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        prompt = body["messages"][-1]["content"]
        text, seconds, cut_at = self._answer(prompt)
        if not body.get("stream"):
            return await self._first_byte(seconds) or web.json_response({})
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        base = {
            "id": f"chatcmpl-{prompt}",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": body["model"],
        }

        async def send(**fields: Any) -> None:
            await response.write(f"data: {json.dumps({**base, **fields})}\n\n".encode())

        def choice(delta: dict[str, Any], finish_reason: str | None = None) -> dict:
            return {"index": 0, "delta": delta, "finish_reason": finish_reason}

        await send(choices=[choice({"role": "assistant", "content": ""})])
        if cut_at < seconds:
            await self._cut("xai", request, cut_at)
            return response
        await asyncio.sleep(seconds * SCALE)
        await send(choices=[choice({"content": text})])
        await send(choices=[choice({}, "stop")])
        await send(
            choices=[],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def anthropic(self, request: web.Request) -> web.StreamResponse:
        try:
            return await self._anthropic(request)
        except ConnectionResetError:  # the client gave up at its deadline
            return web.Response(status=499)

    async def _anthropic(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        prompt = body["messages"][-1]["content"][0]["text"]
        text, seconds, cut_at = self._answer(prompt)
        if not body.get("stream"):
            return await self._first_byte(seconds) or web.json_response({})
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)

        async def event(name: str, data: dict[str, Any]) -> None:
            await response.write(
                f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()
            )

        await event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": f"msg_{prompt}",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            },
        )
        await event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        await event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": ""},
            },
        )
        if cut_at < seconds:
            await self._cut("anthropic", request, cut_at)
            return response
        await asyncio.sleep(seconds * SCALE)
        await event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        )
        await event("content_block_stop", {"type": "content_block_stop", "index": 0})
        await event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 20},
            },
        )
        await event("message_stop", {"type": "message_stop"})
        await response.write_eof()
        return response


def _first_draw_minutes(prompt: str) -> float:
    return random.Random(f"{prompt}/1").uniform(5, 35)


async def _complete_all(provider: LiteLLMProvider) -> list[Any]:
    async def one(index: int) -> Any:
        try:
            return await provider.complete(
                system_prompt="system",
                user_prompt=f"{provider.provider_name}-{index:02}",
                max_tokens=100_000,
                request_timeout_seconds=DEADLINE * SCALE,
            )
        except ProviderError as error:
            return error

    return await asyncio.gather(*(one(index) for index in range(CALLS)))


def _summary(name: str, outcomes: list[Any]) -> dict[str, Any]:
    answers = [item for item in outcomes if not isinstance(item, Exception)]
    lost = sum(
        len((item.evidence or {}).get("connection_lost", ())) for item in outcomes
    )
    accepted = [json.loads(item.text)["minutes"] for item in answers]
    first = [_first_draw_minutes(f"{name}-{index:02}") for index in range(CALLS)]
    return {
        "provider": name,
        "completed": len(answers),
        "completed_after_resend": sum(
            bool(item.evidence["connection_lost"]) for item in answers
        ),
        "failed": len(outcomes) - len(answers),
        "failed_at_deadline": sum(
            isinstance(item, ProviderTimeout) for item in outcomes
        ),
        "connection_lost": lost,
        "mean_minutes_first_sends": round(sum(first) / len(first), 1),
        "mean_minutes_accepted": round(sum(accepted) / len(accepted), 1)
        if accepted
        else None,
    }


# Scaled milliseconds stand for minutes, so a loaded machine can move a cut
# or an answer across the deadline. LiteLLM's OpenAI-compatible stream handler,
# xAI's, reads every attribute of a chunk's empty delta, pydantic's deprecated
# instance attributes included.
@pytest.mark.stochastic
@pytest.mark.filterwarnings(
    "ignore:Accessing the 'model_:pydantic.warnings.PydanticDeprecatedSince211"
)
async def test_each_mechanism_through_the_proxy_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm_provider, "POLL_SECONDS", POLL * SCALE)
    monkeypatch.setattr(litellm_provider, "_BACKOFF_SECONDS", SCALE)
    monkeypatch.setattr(litellm_provider, "http_client", litellm_http_client)
    for entry_point in (
        "acompletion",
        "aresponses",
        "aget_responses",
        "acancel_responses",
    ):
        monkeypatch.setattr(
            litellm_provider, entry_point, getattr(litellm, entry_point)
        )
    server = ProxiedProviders()
    runner = web.AppRunner(server.app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    base = f"http://127.0.0.1:{port}"
    try:

        def provider(name: ProviderName, model: str) -> LiteLLMProvider:
            return LiteLLMProvider(
                api_key="offline",
                provider_name=name,
                model=model,
                max_concurrent=CALLS,
                reasoning_effort="high",
                api_base=f"{base}/v1" if name != "anthropic" else base,
            )

        # A plain long request sees no first byte within 30 s and gets a 502.
        async with httpx.AsyncClient() as client:
            plain = await asyncio.gather(
                *(
                    client.post(
                        f"{base}/v1/responses",
                        json={"input": f"plain-{index:02}"},
                        timeout=DEADLINE * SCALE,
                    )
                    for index in range(CALLS)
                )
            )
        assert {response.status_code for response in plain} == {502}

        background = await _complete_all(provider("openai", "fixture-model"))
        # Roster models: LiteLLM maps reasoning effort only for models it
        # knows, and the transport allows it for xAI's. The streams mostly
        # wait, so both providers' streams run at once.
        anthropic, xai = await asyncio.gather(
            _complete_all(provider("anthropic", "claude-haiku-5-5")),
            _complete_all(provider("xai", "grok-4.7")),
        )
        streamed = {"anthropic": anthropic, "xai": xai}
    finally:
        await runner.cleanup()

    rows = [
        _summary("openai", background),
        *(_summary(name, outcomes) for name, outcomes in streamed.items()),
    ]
    print("\n" + "\n".join(json.dumps(row) for row in rows))
    # Background runs: every call completes from one submission, nothing is
    # re-asked and nothing is lost.
    assert rows[0] == {
        **rows[0],
        "completed": CALLS,
        "completed_after_resend": 0,
        "failed": 0,
        "connection_lost": 0,
    }
    assert all(server.sends[f"openai-{index:02}"] == 1 for index in range(CALLS))
    # Streams: every cut the proxy made is on its call's record and each call
    # was sent at most three times.
    for row in rows[1:]:
        name = row["provider"]
        # A call that reaches its deadline may stop waiting just as its last
        # stream is cut, before it can record the cut.
        assert server.cuts[name] > 0
        assert (
            server.cuts[name] - row["failed_at_deadline"]
            <= row["connection_lost"]
            <= server.cuts[name]
        )
        assert {
            lost["cause"]
            for item in streamed[name]
            for lost in (item.evidence or {}).get("connection_lost", ())
        } == {"connection_error"}
        assert max(server.sends[f"{name}-{index:02}"] for index in range(CALLS)) <= 3
        assert row["completed"] + row["failed"] == CALLS
        assert all(
            isinstance(item, ProviderError) or json.loads(item.text)["send"] >= 1
            for item in streamed[name]
        )


async def test_repeated_chunks_do_not_cut_a_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LiteLLM fails a stream whose last 100 chunks are identical, a check on
    the answer's content; the transport turns it off, so a long run of equal
    values is one answer, not a cut and a re-send."""
    monkeypatch.setattr(litellm_provider, "acompletion", litellm.acompletion)
    sends = 0

    async def messages(request: web.Request) -> web.StreamResponse:
        nonlocal sends
        sends += 1
        body = await request.json()
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)

        async def event(data: dict[str, Any]) -> None:
            await response.write(
                f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()
            )

        def text(value: str) -> dict[str, Any]:
            return {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": value},
            }

        await event(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_repeat",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            }
        )
        await event(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        )
        await event(text('{"values": ['))
        for _ in range(120):
            await event(text("0.0, "))
        await event(text("0.0]}"))
        await event({"type": "content_block_stop", "index": 0})
        await event(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 130},
            }
        )
        await event({"type": "message_stop"})
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/messages", messages)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        provider = LiteLLMProvider(
            api_key="offline",
            provider_name="anthropic",
            # A roster model: LiteLLM maps reasoning effort only for models it knows.
            model="claude-haiku-5-5",
            max_concurrent=1,
            reasoning_effort="high",
            api_base=f"http://127.0.0.1:{runner.addresses[0][1]}",
        )
        response = await provider.complete(
            system_prompt="system",
            user_prompt="user",
            max_tokens=1000,
            request_timeout_seconds=30,
        )
    finally:
        await runner.cleanup()
    assert sends == 1
    assert response.evidence["connection_lost"] == []
    assert json.loads(response.text) == {"values": [0.0] * 121}
