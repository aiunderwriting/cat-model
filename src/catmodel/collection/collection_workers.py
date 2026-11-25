"""Collector processes: a study collected on more than one CPU core.

A collector serves every provider exchange of its contexts from one event
loop, on one core. What loads that core is the transport, not the questions:
LiteLLM parses each streamed chunk and each poll in Python, a streamed answer
arrives in hundreds to thousands of chunks, and so every stream in flight
costs a steady share of the core for as long as it runs (``runbook.md``
section 3 gives sizing guidance). Past the load one core serves, the loop
reads its providers late: latencies stretch, an exchange times out locally
though the provider answered, and a submission that timed out may have
created a billed run that is never polled.

``collect_in_processes`` therefore runs several collectors. The pending
contexts are dealt to them in order, one each in turn, and each model's
ceiling of requests in flight is divided among them (``ceiling_shares``), so
the collectors together never exceed it and no collector serves more than its
share however the work is spread. Each collector is an ordinary study
collection of its contexts (``collect_study``) with lanes of its own: a rate
limit lowers only the limit of the collector that heard it, and a collector
that finishes early leaves its share unused.

Frozen files do not depend on the number of collectors: a context is
collected by one collector, into its own directory, exactly as one process
collects it, and the answer table takes several writers.

Stopping is cooperative, because a collector that is killed cannot cancel the
server-side runs it left open. On Ctrl-C or SIGTERM, on a systematic provider
failure in one collector, or when a collector dies, the parent tells every
collector to stop; each cancels its collection, cancels its open runs at the
provider and exits, and only one still running ``STOP_SECONDS`` later is
terminated. A collector whose parent has died stops the same way by itself.
"""

from __future__ import annotations

import asyncio
import multiprocessing.connection
import os
import pickle
import signal
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from catmodel.worker_pool import (
    SpawnContext,
    sigterm_exits,
    start_worker,
    warning_filters,
)

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from catmodel.config import ExecutionSettings
    from catmodel.llm.models import ModelSpec

logger = structlog.get_logger(__name__)

STOP_SECONDS = 90.0
"""How long a collector told to stop may take to cancel its open server-side
runs and exit before it is terminated: the 30 s ``cancel_open_runs`` allows,
with room for a loop that is late."""


def collector_count(workers: int, contexts: int, ceilings: Mapping[str, int]) -> int:
    """The collector processes a collection of ``contexts`` pending contexts
    uses when asked for ``workers``: no more than its contexts, and no more
    than the smallest ceiling, so every collector may send to every model."""
    return max(1, min(workers, contexts, *ceilings.values()))


def ceiling_shares(
    ceilings: Mapping[str, int], collectors: int
) -> list[dict[str, int]]:
    """Each collector's share of every model's ceiling. The shares of a model
    sum to its ceiling and differ by at most one."""
    return [
        {
            model: ceiling // collectors + (index < ceiling % collectors)
            for model, ceiling in ceilings.items()
        }
        for index in range(collectors)
    ]


@dataclass(frozen=True)
class _Task:
    """What one collector collects: its contexts' prepared capture
    directories, and the run's models and settings with this collector's
    share of the ceilings."""

    run_directories: tuple[Path, ...]
    models: tuple[ModelSpec, ...]
    execution: ExecutionSettings
    answers: Path | None


def _portable(error: BaseException) -> BaseException:
    """``error`` if it survives the trip to the parent process, otherwise an
    error of its family with its message."""
    try:
        clone = pickle.loads(pickle.dumps(error))
        if type(clone) is type(error) and str(clone) == str(error):
            return error
    except Exception:
        pass
    if isinstance(error, ValueError):
        return ValueError(str(error))
    if isinstance(error, OSError):
        return OSError(str(error))
    return RuntimeError(f"{type(error).__name__}: {error}")


def _stop_when_told(control: Connection) -> None:
    """Cancel this collector's collection when the parent says stop, or when
    the parent has ended and its end of ``control`` closed. The collection
    then cancels its open runs (``with_open_runs_cancelled``); a collector
    that has not exited ``STOP_SECONDS`` later exits at once."""
    loop = asyncio.get_running_loop()
    work = asyncio.current_task()
    assert work is not None

    def listen() -> None:
        with suppress(EOFError, OSError):
            control.recv()
        last_resort = threading.Timer(STOP_SECONDS, os._exit, (1,))
        last_resort.daemon = True
        last_resort.start()
        # The loop is closed when the collection has already ended.
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(work.cancel)

    threading.Thread(target=listen, name="collector-stop", daemon=True).start()


async def _collect(task: _Task, control: Connection) -> list[BaseException | None]:
    """Collect this collector's contexts; each context's error, or None for a
    context whose artifact is frozen."""
    from catmodel.collection.answer_table import AnswerTable
    from catmodel.collection.contextual_collection import collect_study
    from catmodel.config import settings
    from catmodel.contracts.contextual_risk import ContextualRiskInstrument
    from catmodel.json_io import strict_json_loads
    from catmodel.llm.factory import build_providers

    _stop_when_told(control)
    # The instrument each context's preparation froze in its directory.
    contexts = [
        (
            ContextualRiskInstrument.from_saved(
                strict_json_loads(
                    (run_directory / "contextual-instrument.json").read_bytes()
                )
            ),
            run_directory,
        )
        for run_directory in task.run_directories
    ]
    providers = build_providers(
        task.models,
        api_keys=settings.provider_api_keys(),
        concurrency_limits=task.execution.concurrency_by_model,
    )
    results = await collect_study(
        contexts,
        providers=providers,
        config=task.execution,
        answers=None if task.answers is None else AnswerTable(task.answers),
    )
    return [
        _portable(result) if isinstance(result, Exception) else None
        for result in results
    ]


def _collector(
    index: int,
    task: _Task,
    results: Connection,
    control: Connection,
    filters: list[tuple[Any, str, type[Warning], str, int]],
) -> None:
    """One collector process. It sends the parent one message before it
    exits: ``done`` with its contexts' outcomes, ``stopped`` when it was told
    to stop, or ``failed`` with the error that ended its collection."""
    from catmodel.llm.transport import with_open_runs_cancelled

    # A collector whose parent has ended stops itself (``_stop_when_told``),
    # so that it cancels its open runs first.
    start_worker(filters, exit_with_parent=False)
    structlog.contextvars.bind_contextvars(collector=index)
    message: tuple[str, Any]
    try:
        message = (
            "done",
            asyncio.run(with_open_runs_cancelled(_collect(task, control))),
        )
    except asyncio.CancelledError:
        message = ("stopped", None)
    except Exception as error:
        # A provider failure is logged where it happened; anything else is a
        # defect, reported with where it was raised.
        if not isinstance(error, ValueError):
            logger.error("collector failed", error=str(error), exc_info=True)
        message = ("failed", _portable(error))
    with suppress(OSError):
        results.send(message)
    results.close()


@dataclass
class _Collector:
    process: Any
    results: Connection
    control: Connection
    indices: tuple[int, ...]
    """The places, among the pending contexts, of this collector's contexts."""
    told: float | None = None
    """When the collector was told to stop."""

    def stop(self) -> None:
        """Tell the collector to stop, once."""
        if self.told is None:
            self.told = time.monotonic()
            with suppress(OSError):
                self.control.send("stop")

    def message(self) -> tuple[str, Any]:
        """The collector's one message, once its pipe is readable: ``done``
        with its contexts' outcomes, ``stopped`` if it was told to stop, or
        ``failed`` with an error, which for a collector that sent no message
        or stopped untold says how it ended."""
        try:
            kind, payload = self.results.recv()
        except (EOFError, OSError):
            return "failed", ChildProcessError(_ended(self.process))
        if kind == "stopped" and self.told is None:
            return "failed", ChildProcessError(
                "a collector process stopped without being told to"
            )
        return kind, payload


def _ended(process: Any) -> str:
    """How a collector that sent no message ended."""
    process.join(5)
    code = process.exitcode
    if code is not None and code < 0:
        name = signal.Signals(-code).name
        hint = (
            ", as when the system runs out of memory" if -code == signal.SIGKILL else ""
        )
        return (
            f"a collector process was killed by {name}{hint}; server-side runs "
            "it left open were not cancelled"
        )
    return f"a collector process exited with code {code} before reporting"


def _reap(collectors: Sequence[_Collector]) -> None:
    """Wait for every collector to exit, terminating one that is still
    running ``STOP_SECONDS`` after it was told to stop, or after now if it
    was not, and close the pipes."""
    now = time.monotonic()
    for collector in collectors:
        process = collector.process
        if process.pid is None:
            continue
        deadline = (collector.told or now) + STOP_SECONDS
        process.join(max(0.0, deadline - time.monotonic()))
        if process.exitcode is None:
            logger.error(
                "collector did not stop; terminating it, so server-side runs "
                "it left open are not cancelled",
                pid=process.pid,
            )
            process.terminate()
            process.join(5)
            if process.exitcode is None:
                process.kill()
                process.join()
    for collector in collectors:
        collector.results.close()
        collector.control.close()


def collect_in_processes(
    run_directories: Sequence[Path],
    *,
    models: Sequence[ModelSpec],
    execution: ExecutionSettings,
    answers: Path | None,
    collectors: int,
) -> list[BaseException | None]:
    """Collect the prepared contexts ``run_directories`` in ``collectors``
    collector processes; return each context's error, or None for a context
    whose artifact is frozen, in the order given.

    Context number ``i`` goes to collector ``i mod collectors``, and each
    collector takes its share of every model's ceiling in ``execution``
    (``ceiling_shares``). A systematic provider failure in one collector, or
    any other error that ends its collection, stops them all and is raised;
    so is the death of a collector, as a ``ChildProcessError``. An
    interruption of this process (Ctrl-C, or SIGTERM in the main thread)
    stops them all before it is raised again. Every stop lets each collector
    cancel its open server-side runs first (module docstring).
    """
    if collectors < 1 or collectors > collector_count(
        collectors, len(run_directories), execution.concurrency_by_model
    ):
        raise ValueError(
            "collectors must be at least one and no more than the contexts "
            "or the smallest ceiling"
        )
    shares = ceiling_shares(execution.concurrency_by_model, collectors)
    context = SpawnContext()
    filters = warning_filters()
    running: list[_Collector] = []
    outcomes: dict[int, BaseException | None] = {}
    failure: BaseException | None = None
    stopping = 0.0
    with ExitStack() as stack:
        stack.enter_context(sigterm_exits())
        # Whatever ends this block, every collector has exited after it.
        stack.callback(_reap, running)
        try:
            for index, share in enumerate(shares):
                indices = tuple(range(index, len(run_directories), collectors))
                results, send_results = context.Pipe(duplex=False)
                receive_control, control = context.Pipe(duplex=False)
                task = _Task(
                    run_directories=tuple(run_directories[place] for place in indices),
                    models=tuple(models),
                    execution=execution.model_copy(
                        update={"concurrency_by_model": share}
                    ),
                    answers=answers,
                )
                process = context.Process(
                    target=_collector,
                    args=(index, task, send_results, receive_control, filters),
                    name=f"collector-{index}",
                )
                running.append(_Collector(process, results, control, indices))
                process.start()
                # The collector holds its own ends: closing these lets each side
                # see the other's end.
                send_results.close()
                receive_control.close()
                logger.info(
                    "collector started",
                    collector=index,
                    pid=process.pid,
                    contexts=len(indices),
                    concurrency_by_model=share,
                )
            waiting = {collector.results: collector for collector in running}
            while waiting:
                # Once the collectors are told to stop, one that stays silent
                # past its time is left to ``_reap``.
                patience = (
                    None
                    if failure is None
                    else max(0.0, STOP_SECONDS - (time.monotonic() - stopping))
                )
                arrived = multiprocessing.connection.wait(list(waiting), patience)
                if not arrived:
                    break
                for ready in arrived:
                    collector = waiting.pop(ready)  # ty: ignore[invalid-argument-type]
                    kind, payload = collector.message()
                    if kind == "done":
                        outcomes.update(zip(collector.indices, payload, strict=True))
                    elif kind != "stopped" and failure is None:
                        failure, stopping = payload, time.monotonic()
                        logger.error(
                            "a collector ended the study; stopping the others",
                            collector=running.index(collector),
                            error=str(payload),
                        )
                        for other in waiting.values():
                            other.stop()
            if failure is not None:
                raise failure
        except BaseException:
            for collector in running:
                collector.stop()
            raise
    return [outcomes[place] for place in range(len(run_directories))]
