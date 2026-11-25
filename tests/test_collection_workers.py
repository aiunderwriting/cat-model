"""A study collected in several collector processes: the same frozen files,
divided ceilings, and stops that cancel every open server-side run.

The collectors are real processes running the real transport; only the
provider is local (``BackgroundRunServer``): the collectors find it through
``OPENAI_BASE_URL``, with a key that is no provider's.
"""

import asyncio
import json
import multiprocessing
import os
import pickle
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import litellm
import pytest
import structlog

from catmodel.collection import collection_scheduler
from catmodel.collection.collection_scheduler import (
    CollectionScheduler,
    watch_event_loop,
)
from catmodel.collection.collection_workers import (
    STOP_SECONDS,
    ceiling_shares,
    collector_count,
)
from catmodel.collection.contextual_collection import run_collections
from catmodel.collection.frozen_call_collection import ProviderFailed
from catmodel.config import settings
from catmodel.contracts.contextual_risk import ContextualRiskInstrument
from catmodel.evidence.source_bundle import capture_source_bundle_snapshot
from catmodel.llm import litellm_provider
from catmodel.llm.litellm_provider import http_client as litellm_http_client
from test_support.capability_sources import authored_source_files, write_source_bundle
from test_support.contextual_risk import (
    contextual_answer,
    contextual_instrument,
    fixture_response,
)
from test_support.llm_fakes import BackgroundRunServer

PROFILES = """\
profiles:
  mine:
    models:
      - provider: openai
        model_id: fixture-model
        reasoning_effort: low
        max_output_tokens: 100000
    settings:
      max_concurrency: 4
"""
CALLS_PER_CONTEXT = 6
"""The fixture instrument's initial requests: a direct and a chain question at
each of its three coordinates. Its fixture answers ask no follow-up."""


def _answer(_instructions: str, question: str) -> str:
    return fixture_response(question, contextual_answer())


@dataclass(frozen=True)
class _Study:
    """A source bundle, the fixture instrument bound to it and a profile
    whose one model is the local server's."""

    root: Path
    instrument: ContextualRiskInstrument
    source_bundle: Path
    profiles: Path

    def collect(self, names, *, workers=1):
        """Collect one context of the instrument into each named directory."""
        return run_collections(
            [(self.instrument, self.root / name) for name in names],
            source_bundle=self.source_bundle,
            profile="mine",
            profiles=self.profiles,
            workers=workers,
        )


def _study(tmp_path, monkeypatch, server) -> _Study:
    source = write_source_bundle(tmp_path / "source", files=authored_source_files())
    data = contextual_instrument().model_dump(mode="json")
    data["source_bundle_manifest_sha256"] = capture_source_bundle_snapshot(
        source
    ).manifest_sha256
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text(PROFILES)
    # The collectors read the key and the base URL from their environment;
    # this process's settings were read when it started.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-offline")
    monkeypatch.setenv("OPENAI_BASE_URL", server.url)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "127.0.0.1,localhost")
    # A collection in this process reaches the local server through LiteLLM
    # too; tests otherwise refuse its entry points.
    monkeypatch.setattr(litellm_provider, "http_client", litellm_http_client)
    for entry_point in ("aresponses", "aget_responses", "acancel_responses"):
        monkeypatch.setattr(
            litellm_provider, entry_point, getattr(litellm, entry_point)
        )
    return _Study(
        root=tmp_path,
        instrument=ContextualRiskInstrument.model_validate(data),
        source_bundle=source,
        profiles=profiles,
    )


def _frozen(run_directory):
    """Every file of a capture by relative path, without what differs between
    two collections of the same answers: when each attempt's response came."""
    files = {}
    for path in sorted(run_directory.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        if path.name.startswith("attempt-"):
            row = json.loads(data)
            del row["latency_ms"]
            del row["provider_evidence"]["status_first_seen_ms"]
            data = json.dumps(row, sort_keys=True).encode()
        files[str(path.relative_to(run_directory))] = data
    return files


def test_shares_divide_each_ceiling_and_no_collector_goes_without():
    ceilings = {"a": 200, "b": 400, "c": 7}
    shares = ceiling_shares(ceilings, 3)
    assert shares == [
        {"a": 67, "b": 134, "c": 3},
        {"a": 67, "b": 133, "c": 2},
        {"a": 66, "b": 133, "c": 2},
    ]
    for model, ceiling in ceilings.items():
        assert sum(share[model] for share in shares) == ceiling
    # No more collectors than contexts, or than the smallest ceiling.
    assert collector_count(8, 48, ceilings) == 7
    assert collector_count(8, 3, ceilings) == 3
    assert collector_count(1, 48, ceilings) == 1


def test_a_provider_failure_survives_the_trip_from_a_collector():
    failure = pickle.loads(pickle.dumps(ProviderFailed("no answer", transient=True)))
    assert isinstance(failure, ProviderFailed)
    assert (str(failure), failure.transient) == ("no answer", True)


async def test_turns_let_the_event_loop_run_between_requests():
    scheduler = CollectionScheduler({"model": 1})
    iterations = 0

    async def loop_iterations():
        nonlocal iterations
        while True:
            iterations += 1
            await asyncio.sleep(0)

    order = []

    async def prepare(index):
        await scheduler.turn()
        order.append(index)

    counting = asyncio.ensure_future(loop_iterations())
    await asyncio.gather(*(prepare(index) for index in range(50)))
    counting.cancel()
    # In the order asked for, and the loop came round for every one of them:
    # without turns, the fifty would run within one iteration.
    assert order == list(range(50))
    assert iterations >= 50


async def test_a_held_event_loop_is_reported(monkeypatch):
    monkeypatch.setattr(collection_scheduler, "LOOP_LATE_SECONDS", 0.05)
    with structlog.testing.capture_logs() as logs:
        watch = asyncio.ensure_future(watch_event_loop(interval=0.01))
        await asyncio.sleep(0.05)
        assert logs == []
        time.sleep(0.2)
        await asyncio.sleep(0.05)
        watch.cancel()
    assert [row["late_checks"] for row in logs] == [1]
    assert logs[0]["event"].startswith("event loop late") and logs[0]["worst_s"] >= 0.1


async def test_a_process_at_the_limit_of_its_core_is_reported(monkeypatch):
    monkeypatch.setattr(collection_scheduler, "LOOP_REPORT_SECONDS", 0.1)
    # A fifth of a core, so that a loaded test machine still shows it.
    monkeypatch.setattr(collection_scheduler, "BUSY_SHARE", 0.2)
    with structlog.testing.capture_logs() as logs:
        watch = asyncio.ensure_future(watch_event_loop(interval=0.01))
        # Idle, the process is not reported.
        await asyncio.sleep(0.3)
        assert logs == []
        # Working without holding the loop, it is.
        until = time.monotonic() + 0.4
        while time.monotonic() < until:
            sum(range(1000))
            await asyncio.sleep(0)
        watch.cancel()
    assert logs and all(
        row["event"].startswith("collector bound by its CPU core") for row in logs
    )
    assert logs[0]["cpu_share"] >= 0.2
    # No two reports are the same warning, which the log would write once.
    assert len({row["elapsed_s"] for row in logs}) == len(logs) > 1


def test_collectors_freeze_what_one_process_freezes(tmp_path, monkeypatch):
    shared = ["a", "b", "c"]
    with BackgroundRunServer(_answer) as server:
        study = _study(tmp_path, monkeypatch, server)
        with structlog.testing.capture_logs() as logs:
            artifacts = study.collect(shared, workers=2)
        assert artifacts is not None and len(artifacts) == 3
        started = [row for row in logs if row["event"] == "collector started"]
        # Two collectors, the contexts dealt in turn and the ceiling of four
        # divided between them.
        assert [(row["contexts"], row["concurrency_by_model"]) for row in started] == [
            (2, {"fixture-model": 2}),
            (1, {"fixture-model": 2}),
        ]
        assert len(server.submitted) == 3 * CALLS_PER_CONTEXT
        assert server.cancelled == []
        reference = study.collect(["alone"])
        assert reference is not None
    frozen = _frozen(tmp_path / "alone")
    assert sum(name.startswith("calls/") for name in frozen) == 2 * CALLS_PER_CONTEXT
    for artifact, name in zip(artifacts, shared, strict=True):
        assert artifact == reference[0]
        assert _frozen(tmp_path / name) == frozen
    # A rerun finds every context complete and starts no collector.
    with structlog.testing.capture_logs() as logs:
        assert study.collect(shared, workers=2) == artifacts
    assert not [row for row in logs if row["event"] == "collector started"]


def _when_open(server, runs, action):
    """Call ``action`` once the server holds ``runs`` open runs that their
    collectors know of: each has been polled, ten seconds after it was
    submitted."""

    def wait():
        # A collection that never opens them is stopped all the same, and
        # the test then fails on its counts instead of waiting for ever.
        patience = time.monotonic() + 120
        while len(server.polled) < runs and time.monotonic() < patience:
            time.sleep(0.05)
        action()

    thread = threading.Thread(target=wait, daemon=True)
    thread.start()
    return thread


@pytest.mark.parametrize(
    ("stop", "workers"), [(signal.SIGINT, 2), (signal.SIGTERM, 2), (signal.SIGTERM, 1)]
)
def test_an_interrupted_study_cancels_every_open_run(
    tmp_path, monkeypatch, stop, workers
):
    raised = KeyboardInterrupt if stop == signal.SIGINT else SystemExit
    with BackgroundRunServer(_answer, hold=True) as server:
        study = _study(tmp_path, monkeypatch, server)
        # The ceiling of four is full: two runs in each of two collectors,
        # or four in the one process.
        _when_open(server, 4, lambda: os.kill(os.getpid(), stop))
        began = time.monotonic()
        with pytest.raises(raised):
            study.collect(["a", "b"], workers=workers)
        assert time.monotonic() - began < STOP_SECONDS
        assert len(server.submitted) == 4
        assert sorted(server.cancelled) == sorted(server.submitted)
    assert multiprocessing.active_children() == []
    # Nothing was answered, so nothing but the plans is frozen, and the
    # interrupted contexts resume.
    for name in ("a", "b"):
        assert not list((tmp_path / name).glob("calls/*/attempt-*.json"))
        assert not (tmp_path / name / "contextual-risk.json").exists()


def test_a_systematic_failure_in_one_collector_stops_them_all(tmp_path, monkeypatch):
    with BackgroundRunServer(_answer, hold=True, reject_after=3) as server:
        study = _study(tmp_path, monkeypatch, server)
        with pytest.raises(ProviderFailed, match="Collection incomplete") as failure:
            study.collect(["a", "b"], workers=2)
        assert not failure.value.transient
        # The refused collector cancelled its own open run and the other,
        # told to stop, its two.
        assert len(server.submitted) == 3
        assert sorted(server.cancelled) == sorted(server.submitted)
    assert multiprocessing.active_children() == []
    failed = [
        json.loads(path.read_text())
        for path in tmp_path.glob("[ab]/calls/*/attempt-*.json")
    ]
    assert len(failed) == 1 and failed[0]["raw_response"] is None


def test_a_killed_collector_stops_the_others(tmp_path, monkeypatch):
    with BackgroundRunServer(_answer, hold=True) as server:
        study = _study(tmp_path, monkeypatch, server)

        def kill_one():
            multiprocessing.active_children()[0].kill()

        _when_open(server, 4, kill_one)
        with pytest.raises(ChildProcessError, match="killed by SIGKILL"):
            study.collect(["a", "b"], workers=2)
        # The collector left running cancelled its two runs; the killed one
        # could not.
        assert len(server.cancelled) == 2
        assert set(server.cancelled) < set(server.submitted)
    assert multiprocessing.active_children() == []
