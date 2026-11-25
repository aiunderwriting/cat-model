"""Guards that apply to every test; tests build their data with ``test_support``.

Provider credentials can be present wherever tests run, in the environment or
supplied by a proxy, so a provider request that escaped a test would be a
live, billed call. The transport's provider entry points (LiteLLM's completion
and Responses API calls and the HTTP handler a background run's exchanges go
through) refuse to run, and an HTTP client refuses any host but this machine
unless a test gave it an in-memory transport. Tests that exercise the transport replace the entry points
with fakes or point them at an in-memory API or a local server. Model profiles
come from AIUC's packaged example unless a test names another file, whatever
``CATMODEL_PROFILES`` holds where tests run.

Every results file a run or a replay writes in a test is read back, so the
loader accepts each file the suite's end-to-end tests produce.

``--shard I/N`` runs part I of N of the collected tests, so that N jobs can
share the suite: every test is in exactly one part.
"""

import httpx
import pytest

from catmodel.config import PROFILES_ENV

# Imported first so LiteLLM loads its bundled model map, as cat-model does,
# rather than fetching the current one when a test imports it directly.
from catmodel.llm import litellm_provider, transport
from catmodel.results import execution
from catmodel.results.results_file import load_results, write_results
from test_support.shards import parse_shard, shard_of

_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
_send = httpx.AsyncClient.send
_send_sync = httpx.Client.send


def _refuse_remote(client: httpx.AsyncClient | httpx.Client, request: httpx.Request):
    in_memory = isinstance(client._transport_for_url(request.url), httpx.MockTransport)
    if request.url.host not in _LOOPBACK and not in_memory:
        raise RuntimeError(f"tests never call {request.url.host}")


async def _guarded_send(self: httpx.AsyncClient, request: httpx.Request, **kwargs):
    _refuse_remote(self, request)
    return await _send(self, request, **kwargs)


def _guarded_send_sync(self: httpx.Client, request: httpx.Request, **kwargs):
    _refuse_remote(self, request)
    return _send_sync(self, request, **kwargs)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--shard",
        metavar="I/N",
        default=None,
        help="run only part I of N of the collected tests",
    )


def pytest_configure(config: pytest.Config) -> None:
    shard = config.getoption("--shard")
    if shard is not None:
        try:
            parse_shard(shard)
        except ValueError as error:
            raise pytest.UsageError(str(error)) from None


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    shard = config.getoption("--shard")
    if shard is None:
        return
    part, parts = parse_shard(shard)
    left_out = [item for item in items if shard_of(item.nodeid, parts) != part]
    if left_out:
        config.hook.pytest_deselected(items=left_out)
        items[:] = [item for item in items if shard_of(item.nodeid, parts) == part]


@pytest.fixture(autouse=True)
def _no_live_provider_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("tests never call a live provider")

    for entry_point in (
        "http_client",
        "acompletion",
        "aresponses",
        "aget_responses",
        "acancel_responses",
    ):
        monkeypatch.setattr(litellm_provider, entry_point, refuse)
    monkeypatch.setattr(httpx.AsyncClient, "send", _guarded_send)
    monkeypatch.setattr(httpx.Client, "send", _guarded_send_sync)
    monkeypatch.setattr(transport, "_OPEN_RUNS", {})


@pytest.fixture(autouse=True)
def _packaged_model_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PROFILES_ENV, raising=False)


@pytest.fixture(autouse=True)
def _written_results_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run's or a replay's results file, with its diagnostics file, must be
    one ``load_results`` reads: each matches its committed JSON Schema
    document, resolves its reasons and carries its analysis identity."""

    def write_and_read(path, report) -> None:
        write_results(path, report)
        load_results(path, diagnostics=True)

    monkeypatch.setattr(execution, "write_results", write_and_read)
