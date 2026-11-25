"""Offline stand-ins for the provider APIs the LLM transport calls."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import AsyncIterator, Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import httpx
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices, Usage


def chunk(
    text: str = "",
    *,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
    response_id: str = "msg-fixture",
) -> ModelResponseStream:
    """One streamed chat chunk as LiteLLM yields it."""
    streamed = ModelResponseStream(
        id=response_id,
        choices=[
            StreamingChoices(
                index=0, delta=Delta(content=text or None), finish_reason=finish_reason
            )
        ],
    )
    if usage is not None:
        streamed.usage = cast(Any, Usage)(**usage)
    return streamed


def answer_chunks(
    text: str,
    *,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 2,
) -> list[ModelResponseStream]:
    """A whole streamed answer: its text in two pieces, then its usage."""
    middle = len(text) // 2
    return [
        chunk(text[:middle]),
        chunk(text[middle:], finish_reason=finish_reason),
        chunk(
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        ),
    ]


class FakeStream:
    """A LiteLLM stream: yields *chunks*, then raises *error* if given.

    ``received_finish_reason`` is set, as LiteLLM sets it, only when the
    provider's own finish reason arrived; a stream without one ended early.
    ``fetch_stream`` raises *open_error*, if given, as a stream that LiteLLM
    opens only when it is first read does when its request is refused.
    """

    def __init__(
        self,
        chunks: list[ModelResponseStream],
        *,
        error: BaseException | None = None,
        open_error: BaseException | None = None,
        provider_finished: bool = True,
    ) -> None:
        self._chunks = chunks
        self._error = error
        self._open_error = open_error
        self._provider_finished = provider_finished
        self.received_finish_reason: str | None = None

    async def fetch_stream(self) -> None:
        if self._open_error is not None:
            raise self._open_error

    def __aiter__(self) -> AsyncIterator[ModelResponseStream]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[ModelResponseStream]:
        for item in self._chunks:
            reason = item.choices[0].finish_reason if item.choices else None
            if reason is not None and self._provider_finished:
                self.received_finish_reason = reason
            yield item
        if self._error is not None:
            raise self._error


class ScriptedAcompletion:
    """``acompletion`` returning each scripted stream, or raising each
    scripted error, in turn, and recording every call's arguments."""

    def __init__(self, *outcomes: FakeStream | BaseException) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class InMemoryHandler(AsyncHTTPHandler):
    """LiteLLM's HTTP handler over an in-memory transport: every request
    LiteLLM sends through it, its reconnect after a lost connection included,
    goes to *transport*."""

    def __init__(self, transport: httpx.AsyncBaseTransport) -> None:
        self._transport = transport
        super().__init__()

    def create_client(self, timeout: Any, event_hooks: Any, *_: Any, **__: Any):
        return httpx.AsyncClient(
            transport=self._transport, timeout=timeout, event_hooks=event_hooks
        )


type Reply = tuple[int, Any] | BaseException


class ScriptedAPI:
    """A provider API answering each request with the next scripted reply:
    a status and JSON body (``None`` for no body), or an exception raised as
    a transport failure. Every request is recorded."""

    def __init__(self, *replies: Reply, after: Reply | None = None) -> None:
        self.replies = list(replies)
        self.after = after
        self.requests: list[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(
            {
                "method": request.method,
                "path": request.url.path,
                "authorization": request.headers.get("authorization"),
                "json": json.loads(request.content) if request.content else None,
            }
        )
        reply = self.replies.pop(0) if self.replies else self.after
        if reply is None:
            raise AssertionError(f"unscripted request {request.method} {request.url}")
        if isinstance(reply, BaseException):
            raise reply
        status, body = reply
        if body is None:
            return httpx.Response(status, request=request)
        return httpx.Response(status, json=body, request=request)

    def client_factory(self) -> Callable[[str], AsyncHTTPHandler]:
        """A stand-in for the transport's ``http_client``: LiteLLM's HTTP
        handler, answered by this API."""
        handler = InMemoryHandler(httpx.MockTransport(self.handle))
        return lambda _provider_name: handler

    def calls(self, method: str, path: str | None = None) -> int:
        """Requests made with *method*, to *path* if given."""
        return sum(
            request["method"] == method and path in (None, request["path"])
            for request in self.requests
        )


class BackgroundRunServer:
    """OpenAI's Responses API on a local port, answering collectors that run
    in other processes: submissions, polls and cancellations of background
    runs.

    ``answer`` writes a run's text from its instructions and input. A run is
    returned ``completed`` in the reply to its submission, under an id that
    depends only on its prompt; with ``hold`` it is returned ``queued``, under
    an id of its own, and stays ``in_progress`` until it is cancelled. A
    submission after the first ``reject_after`` is refused with a 401, sent
    half a second late so that the replies to the submissions before it have
    been read. ``submitted`` and ``cancelled`` list the run ids in the order
    received, and ``polled`` holds those polled at least once: a collector
    polls a run only after it has read the reply to its submission.
    """

    def __init__(
        self,
        answer: Callable[[str, str], str],
        *,
        hold: bool = False,
        reject_after: int | None = None,
    ) -> None:
        self.submitted: list[str] = []
        self.cancelled: list[str] = []
        self.polled: set[str] = set()
        self._answer = answer
        self._hold = hold
        self._reject_after = reject_after
        self._lock = threading.Lock()
        api = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                pass

            def _reply(self, status: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                run_id = self.path.rsplit("/", 1)[-1]
                with api._lock:
                    api.polled.add(run_id)
                status = "cancelled" if run_id in api.cancelled else "in_progress"
                self._reply(200, api._run(run_id, status))

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path.endswith("/cancel"):
                    run_id = self.path.split("/")[-2]
                    with api._lock:
                        api.cancelled.append(run_id)
                    self._reply(200, api._run(run_id, "cancelled"))
                    return
                with api._lock:
                    rejected = (
                        api._reject_after is not None
                        and len(api.submitted) >= api._reject_after
                    )
                    digest = hashlib.sha256(
                        f"{body['instructions']}\0{body['input']}".encode()
                    ).hexdigest()[:24]
                    run_id = f"resp_{digest}"
                    if api._hold:
                        run_id += f"_{len(api.submitted)}"
                    if not rejected:
                        api.submitted.append(run_id)
                if rejected:
                    time.sleep(0.5)
                    error = {
                        "message": "Incorrect API key provided.",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    }
                    self._reply(401, {"error": error})
                elif api._hold:
                    self._reply(200, api._run(run_id, "queued"))
                else:
                    text = api._answer(body["instructions"], body["input"])
                    self._reply(200, api._run(run_id, "completed", text))

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @staticmethod
    def _run(run_id: str, status: str, text: str | None = None) -> dict[str, Any]:
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

    def __enter__(self) -> BackgroundRunServer:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._server.shutdown()
        self._server.server_close()
