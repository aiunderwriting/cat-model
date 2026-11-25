"""Worker processes for independent per-state tasks.

A state's simulation depends only on its inputs: every stream is keyed by the
plan seed, the state and the year (``simulate_incident_state``). Running such
tasks in worker processes therefore gives the results of running them one
after another on the same platform and dependency versions, and
``WorkerPool.map`` returns them in task order whatever order they finish in.
The number of processes is an operational choice: the requested cap is
recorded with a run's execution context and never enters a setup, a capture
or a result's identity.
"""

from __future__ import annotations

import copyreg
import math
import multiprocessing
import multiprocessing.connection
import multiprocessing.context
import os
import pickle
import resource
import signal
import sys
import threading
import warnings
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from catmodel.logging import configure_logging

# Before a round has measured its tasks: a bound on a worker's peak resident
# memory while it extends one state to ``years`` years, base + per_year *
# years, which allows 0.4 GB and 20 KB a year. Per-year memory grows with a
# state's origins, so later rounds scale the last measured peak instead
# (``simulation_worker_bytes``). The base also bounds a worker's own imports
# where a task's peak is measured in the main process (``_run_here``). The
# bound only limits how many processes run at once.
WORKER_BASE_BYTES = 400_000_000
WORKER_BYTES_PER_YEAR = 20_000


def _read_only_mapping(items: dict) -> MappingProxyType:
    return MappingProxyType(items)


def _reduce_read_only_mapping(mapping: MappingProxyType) -> tuple:
    return _read_only_mapping, (dict(mapping),)


# Validated models and compiled laws keep their mappings read-only
# (``MappingProxyType``), which pickle refuses; tasks and results cross to and
# from worker processes as read-only copies of the same items.
copyreg.pickle(MappingProxyType, _reduce_read_only_mapping)

_CGROUP = Path("/sys/fs/cgroup")
_PROC_CGROUP = Path("/proc/self/cgroup")
# A cgroup v1 limit at or above this is unlimited.
_UNLIMITED = 1 << 60


def _cgroup_directories(controller: str, proc: Path, root: Path) -> list[Path]:
    """This process's cgroup directories for a v1 ``controller`` and for v2.

    Without a cgroup namespace, ``proc`` names the host's path, which does not
    exist inside the container; the container's own cgroup is then the mount
    itself."""
    try:
        lines = proc.read_text(encoding="ascii").splitlines()
    except OSError:
        return []
    directories = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if controller in controllers.split(","):
            mounts = [root / controller]
        elif controllers == "":
            mounts = [root, root / "unified"]
        else:
            continue
        for mount in mounts:
            directory = mount / path.lstrip("/")
            directories.append(directory if directory.is_dir() else mount)
    return directories


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    return int(text) if text.lstrip("-").isdigit() else None


def _memory_stat(directory: Path, key: str) -> int:
    """A counter from a memory cgroup's ``memory.stat``; 0 when absent."""
    try:
        lines = (directory / "memory.stat").read_text(encoding="ascii").splitlines()
    except OSError:
        return 0
    for line in lines:
        name, _, value = line.partition(" ")
        if name == key and value.strip().isdigit():
            return int(value)
    return 0


def cgroup_memory_headroom(
    proc: Path = _PROC_CGROUP, root: Path = _CGROUP
) -> int | None:
    """Bytes this process's cgroup may still allocate under its memory limit;
    None without a limit.

    Usage counts the page cache, which the kernel reclaims before it runs out
    of memory under the limit, so file pages on either LRU list are not counted
    as used, except those mapped into a process, as ``MemAvailable`` counts the
    file cache. Files a run has just written or read sit on the active list;
    counting them as used would hold back workers memory can serve."""
    headroom = None
    for directory in _cgroup_directories("memory", proc, root):
        for limit_file, usage_file, prefix, mapped_key in (
            ("memory.max", "memory.current", "", "file_mapped"),
            (
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
                "total_",
                "total_mapped_file",
            ),
        ):
            limit = _read_int(directory / limit_file)
            usage = _read_int(directory / usage_file)
            if limit is None or usage is None or limit >= _UNLIMITED:
                continue
            cache = _memory_stat(directory, f"{prefix}inactive_file") + _memory_stat(
                directory, f"{prefix}active_file"
            )
            reclaimable = max(cache - _memory_stat(directory, mapped_key), 0)
            used = max(usage - reclaimable, 0)
            room = max(limit - used, 0)
            headroom = room if headroom is None else min(headroom, room)
    return headroom


def cgroup_cpu_quota(proc: Path = _PROC_CGROUP, root: Path = _CGROUP) -> int | None:
    """CPUs this process's cgroup may use, rounded up; None without a quota."""
    for directory in _cgroup_directories("cpu", proc, root):
        try:
            quota, period = (directory / "cpu.max").read_text().split()[:2]
            if quota != "max":
                return max(1, math.ceil(int(quota) / int(period)))
        except (OSError, ValueError):
            pass
        quota_us = _read_int(directory / "cpu.cfs_quota_us")
        period_us = _read_int(directory / "cpu.cfs_period_us")
        if quota_us is not None and period_us and quota_us > 0:
            return max(1, math.ceil(quota_us / period_us))
    return None


def available_cpus() -> int:
    """The CPUs this process may run on, within its cgroup's CPU quota."""
    cpus = os.process_cpu_count() or 1
    quota = cgroup_cpu_quota()
    return cpus if quota is None else min(cpus, quota)


def _reported_available_memory() -> int | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        # Where available memory is not reported, as on macOS: physical memory.
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, AttributeError):
        return None


def available_memory() -> int | None:
    """Memory new processes may use, in bytes: Linux's ``MemAvailable`` (or
    physical memory where that is not reported), within the cgroup's memory
    limit; None where neither is known."""
    known = [
        value
        for value in (_reported_available_memory(), cgroup_memory_headroom())
        if value is not None
    ]
    return min(known) if known else None


def simulation_worker_bytes(years: int, measured: tuple[int, int] | None = None) -> int:
    """A worker's memory bound for extending a state to ``years`` years.

    ``measured`` is the last measured peak of a round's tasks and that round's
    years; the bound scales it by the ratio of years, never down, since part of
    it does not grow with the years. Before any measurement, the formula."""
    if measured is not None:
        peak, at = measured
        return math.ceil(peak * max(years, at) / at)
    return WORKER_BASE_BYTES + WORKER_BYTES_PER_YEAR * years


def memory_bounded_workers(workers: int, per_worker: int, reserved: int = 0) -> int:
    """At most ``workers``, and no more processes than available memory, less
    ``reserved`` bytes the main process will still allocate, holds at
    ``per_worker`` bytes each; one where available memory is unknown, and
    always at least one."""
    memory = available_memory()
    if memory is None:
        return 1
    return max(1, min(workers, (memory - reserved) // max(per_worker, 1)))


def _exit_with_parent() -> None:
    """End this worker when its parent process ends, however it ends."""
    parent = multiprocessing.parent_process()
    if parent is None or parent.sentinel is None:
        return

    def watch() -> None:
        multiprocessing.connection.wait([parent.sentinel])
        os._exit(1)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()


def _prefer_oom_kill() -> None:
    """Ask Linux to kill a worker before the main process when memory runs
    out, so the run stops with a message instead of stranding workers."""
    try:
        with open("/proc/self/oom_score_adj", "w", encoding="ascii") as handle:
            handle.write("500")
    except OSError:
        pass


def start_worker(
    filters: list[tuple[Any, str, type[Warning], str, int]],
    *,
    exit_with_parent: bool = True,
) -> None:
    """Set a spawned worker up: it ignores Ctrl-C, ends with its parent
    unless ``exit_with_parent`` is false (a worker that must clean up first
    watches for its parent's end itself), is killed before its parent when
    memory runs out, and takes the parent's warning filters (``filters``, from
    ``warning_filters``) and logging."""
    # Ctrl-C reaches the whole process group; the parent stops the workers, so
    # they ignore it rather than each printing its own interruption. It has
    # been blocked since the spawn (``_WorkerProcess``); ignoring it discards
    # one that arrived while the worker imported the code.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    if exit_with_parent:
        _exit_with_parent()
    _prefer_oom_kill()
    # The parent's warning filters, so a warning that is an error there (as in
    # the test suite) is one here too.
    warnings.resetwarnings()
    for action, message, category, module, lineno in reversed(filters):
        warnings.filterwarnings(
            action,
            message=message,
            category=category,
            module=module,
            lineno=lineno,
        )
    configure_logging()


def _pattern(value: Any) -> str:
    """A warning filter's message or module pattern as text."""
    return "" if value is None else getattr(value, "pattern", value)


def warning_filters() -> list[tuple[Any, str, type[Warning], str, int]]:
    """The current warning filters on built-in categories, which any worker
    can unpickle. A filter on several categories (``simplefilter`` takes a
    tuple) becomes one filter per category, in the same place."""
    return [
        (
            action,
            _pattern(message),
            category,
            _pattern(module),
            lineno,
        )
        for action, message, categories, module, lineno in warnings.filters
        for category in (categories if isinstance(categories, tuple) else (categories,))
        if category.__module__ == "builtins"
    ]


def _peak_bytes() -> int:
    """This process's peak resident memory: on Linux its address space's
    high-water mark (``VmHWM``), since it started or since ``_reset_peak``;
    elsewhere ``ru_maxrss``. Linux's ``ru_maxrss`` would not do for a worker:
    a spawned process's starts at its parent's resident memory when it was
    forked, so every worker would report the main process's size."""
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _resident_bytes() -> int | None:
    """This process's resident memory now, where ``/proc`` reports it."""
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _reset_peak() -> None:
    """Reset this process's peak resident memory to its resident memory now,
    where Linux allows it (``/proc/self/clear_refs``)."""
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as handle:
            handle.write("5")
    except OSError:
        pass


def _run_here[T, R](
    function: Callable[[T], R], tasks: Iterable[T]
) -> tuple[list[R], int | None]:
    """``function`` applied to every task in this process, with a bound on the
    peak a worker would reach for the largest: what the task raised this
    process's peak above its resident memory before the task, plus a worker's
    own imports (``WORKER_BASE_BYTES``). The peak is reset before each task
    where Linux allows it; elsewhere it is the process's own since it started,
    so an earlier, higher peak counts in full. None where resident memory is
    not reported."""
    results, peak = [], None
    for task in tasks:
        before = _resident_bytes()
        _reset_peak()
        results.append(function(task))
        if before is not None:
            grown = WORKER_BASE_BYTES + max(_peak_bytes() - before, 0)
            peak = grown if peak is None else max(peak, grown)
    return results, peak


def _call[T, R](function: Callable[[T], R], task: T) -> tuple[R, int]:
    """``function(task)`` in a worker, with the worker's peak resident memory.

    An exception that would not survive the trip to the parent is raised as a
    ``RuntimeError`` naming it, rather than breaking the pool."""
    try:
        result = function(task)
    except Exception as error:
        try:
            pickle.loads(pickle.dumps(error))
        except Exception:
            raise RuntimeError(f"{type(error).__name__}: {error}") from None
        raise
    return result, _peak_bytes()


class _WorkerProcess(multiprocessing.context.SpawnProcess):
    """A spawned worker that starts with Ctrl-C blocked.

    A worker ignores Ctrl-C once it has imported the code (``start_worker``);
    one that reached it before then would kill it and break the pool. A
    process starts with its parent thread's signal mask, so blocking SIGINT
    while spawning holds it back until the worker ignores it."""

    def start(self) -> None:
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        try:
            super().start()
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)


class SpawnContext(multiprocessing.context.SpawnContext):
    """The spawn start method, starting workers with Ctrl-C blocked and
    remembering them so their starter can stop them and say how they ended."""

    def __init__(self) -> None:
        self.processes: list[Any] = []

    # The base class's ``Process`` is the process class itself; this factory
    # makes a ``_WorkerProcess`` and records each one.
    def Process(self, *args, **kwargs):  # ty: ignore[incompatible-attribute-override]
        process = _WorkerProcess(*args, **kwargs)
        self.processes.append(process)
        return process


def _broken_reason(error: BrokenProcessPool, processes: list[Any]) -> str:
    """Why a pool broke: a worker killed by a signal, a result or error that
    could not reach this process, or a worker that exited."""
    killed = sorted(
        {-process.exitcode for process in processes if (process.exitcode or 0) < 0}
        - {signal.SIGTERM}
    )
    if killed:
        names = ", ".join(signal.Signals(number).name for number in killed)
        hint = (
            "; if the system ran out of memory, run again with fewer --workers"
            if signal.SIGKILL in killed
            else ""
        )
        return f"a worker process was killed by {names}{hint}"
    # The remote traceback, quoted; its last line is the error.
    cause = [
        line.strip()
        for line in str(error.__cause__ or "").replace("'''", "").splitlines()
        if line.strip()
    ]
    if cause:
        return f"a worker process could not return its result: {cause[-1]}"
    codes = sorted({process.exitcode for process in processes if process.exitcode})
    if codes:
        return (
            f"a worker process exited with code {codes[0]} before returning its "
            "result; a Python script that starts workers must guard its entry "
            'point with if __name__ == "__main__":'
        )
    return f"a worker process ended without returning its result: {error}"


def _exit_on_sigterm(signum: int, _frame: Any) -> None:
    raise SystemExit(128 + signum)


@contextmanager
def sigterm_exits() -> Generator[None]:
    """Inside the block, when it is entered from the main thread, SIGTERM
    raises ``SystemExit`` there instead of ending the process at once, so
    the work in the block can stop what it started. Entered from another
    thread, SIGTERM keeps the handling it had."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        yield
    finally:
        # None: a handler not installed from Python, which cannot be
        # reinstalled; the default ends the process as it would.
        signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)


class WorkerPool:
    """Up to ``workers`` worker processes for ``map``.

    The processes start when a ``map`` first needs more than one and are kept
    for later maps of the same size. Leaving the context stops them, at once
    when it is left by an exception. While worker processes are running, and
    the pool started them from the main thread, SIGTERM raises ``SystemExit``
    there, so the workers are stopped and enclosing ``with`` and ``finally``
    blocks run. Otherwise SIGTERM keeps the handling it had; by default it
    ends the process at once, so temporary files written outside the
    workers' time, such as a run's staged results, can remain. A worker ends
    by itself if this process dies.
    """

    def __init__(self, workers: int) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.workers = workers
        # The largest peak resident memory of a task in the last map: a
        # worker's report, or the bound measured in this process
        # (``_run_here``); None where it was not measured.
        self.peak_task_bytes: int | None = None
        self._executor: ProcessPoolExecutor | None = None
        self._context: SpawnContext | None = None
        self._size = 0
        self._sigterm: Any = None

    def __enter__(self) -> WorkerPool:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop(terminate=exc_type is not None)

    def map[T, R](
        self,
        function: Callable[[T], R],
        tasks: Iterable[T],
        *,
        concurrency: int | None = None,
    ) -> list[R]:
        """``function`` applied to every task, in task order.

        At most ``concurrency`` (default ``workers``) tasks run at once; with
        one, the tasks run here, one after another. Tasks are taken from
        ``tasks`` as workers come free, no more than twice that many unfinished
        at once, so an iterable that builds each task when asked holds few of
        them. A task's exception is raised as running the tasks in order would
        raise it: the earliest failing task's, once every earlier task has
        finished; the other workers are then stopped. Once a task is seen to
        have failed, no more are taken from ``tasks``.
        """
        limit = min(self.workers, self.workers if concurrency is None else concurrency)
        self.peak_task_bytes = None
        if limit <= 1:
            results, self.peak_task_bytes = _run_here(function, tasks)
            return results
        executor = self._executor_for(limit)
        try:
            futures: list[Future] = []
            unfinished: set[Future] = set()
            for task in tasks:
                future = executor.submit(_call, function, task)
                futures.append(future)
                unfinished.add(future)
                if len(unfinished) >= 2 * limit:
                    done, unfinished = wait(unfinished, return_when=FIRST_COMPLETED)
                    if any(row.exception() is not None for row in done):
                        break
            outcomes = cast(
                list[tuple[R, int]], [future.result() for future in futures]
            )
        except BrokenProcessPool as error:
            processes = self._context.processes if self._context else []
            self._stop(terminate=True)
            raise ChildProcessError(_broken_reason(error, processes)) from error
        except BaseException:
            self._stop(terminate=True)
            raise
        self.peak_task_bytes = max((peak for _, peak in outcomes), default=None)
        return [result for result, _ in outcomes]

    def _executor_for(self, size: int) -> ProcessPoolExecutor:
        if self._executor is None or self._size != size:
            self._stop(terminate=False)
            self._context = SpawnContext()
            # Spawned workers import the code afresh rather than inheriting
            # the parent's threads and memory.
            self._executor = ProcessPoolExecutor(
                max_workers=size,
                mp_context=self._context,
                initializer=start_worker,
                initargs=(warning_filters(),),
            )
            self._size = size
            if threading.current_thread() is threading.main_thread():
                previous = signal.signal(signal.SIGTERM, _exit_on_sigterm)
                # None: a handler not installed from Python, which cannot be
                # reinstalled; the default ends the process as it would.
                self._sigterm = signal.SIG_DFL if previous is None else previous
        return self._executor

    def _stop(self, *, terminate: bool) -> None:
        """Stop the workers: terminate them first when asked, then wait for
        the executor, which reaps them, so none outlives the call; then give
        SIGTERM its own handling back."""
        executor, context = self._executor, self._context
        self._executor, self._context, self._size = None, None, 0
        try:
            if executor is None:
                return
            if terminate and context is not None:
                for process in context.processes:
                    if process.pid is not None and process.exitcode is None:
                        process.terminate()
            executor.shutdown(wait=True, cancel_futures=True)
        finally:
            if self._sigterm is not None:
                signal.signal(signal.SIGTERM, self._sigterm)
                self._sigterm = None
