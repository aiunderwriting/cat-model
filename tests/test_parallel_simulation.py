"""Worker processes simulate a tail gate round's states without changing results.

A run with several workers must reproduce the serial run exactly: the same
capture arrays and records, the same results and the same replay, whatever
order the workers finish in. A worker's refusal must stop the run with the
serial path's message, and an interruption must leave no worker running.
"""

import contextlib
import json
import multiprocessing
import os
import pickle
import shutil
import signal
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from catmodel import worker_pool
from catmodel.cli import cli
from catmodel.results import execution
from catmodel.results.execution import replay_capture, run_setup
from catmodel.results.results_export import export_results
from catmodel.results.results_file import diagnostics_path
from catmodel.setup.store import load_setup
from catmodel.simulation import incident_runtime
from catmodel.simulation.execution_budget import WorkLimitExceeded
from catmodel.simulation.incident import compile_incident_state
from catmodel.validation_errors import EvidenceValidationError
from catmodel.worker_pool import (
    WorkerPool,
    available_cpus,
    memory_bounded_workers,
    simulation_worker_bytes,
)
from tests.test_small_events import _kernel
from tests.test_tail_gate import _arms_package, _package

# Each arm doubles to its cap without converging, so the reference arm and
# the comparison arm's two states (one shared with the reference arm only
# while their gates agree) are extended over several rounds.
_GATE = {
    "batches": 4,
    "max_reference_state_years": 800,
    "max_comparison_state_years": 800,
    "levels": [0.999],
}
_BOTH_GATE = {**_GATE, "max_comparison_state_years": 400}


def _capture_files(run: Path) -> dict[str, bytes]:
    capture = run / "capture"
    return {
        path.relative_to(capture).as_posix(): path.read_bytes()
        for path in sorted(capture.rglob("*"))
        if path.is_file()
    }


def _without_execution_context(report: Mapping[str, Any]) -> dict:
    """A report without the worker count it records outside every identity."""
    return {key: value for key, value in report.items() if key != "execution_context"}


def test_workers_reproduce_the_serial_run_capture_results_and_replay(tmp_path):
    package = _arms_package(tmp_path, _GATE, both_gate=_BOTH_GATE)
    runs = {}
    for workers in (1, 2, 3):
        output = tmp_path / f"run-{workers}"
        report = run_setup(
            setup_path=package.root,
            output=output,
            code_commit="fixture",
            workers=workers,
        )
        assert report["execution_context"] == {"workers": workers}
        replay = replay_capture(
            setup_path=package.root,
            capture_directory=output / "capture",
            output=tmp_path / f"replay-{workers}.json",
            code_commit="fixture",
            workers=workers,
        )
        assert replay["execution_context"] == {"workers": workers}
        runs[workers] = (output, report, replay)
    serial, serial_report, serial_replay = runs[1]
    # Several rounds extend three simulations, so workers took tasks out of
    # order and across rounds.
    assert serial_report["tail_gate"]["doublings"] >= 2
    assert len(_capture_files(serial)) >= 3
    for workers in (2, 3):
        output, report, replay = runs[workers]
        # Every capture file, its record and content digest included, is the
        # serial run's byte for byte, and so is every identity in the results.
        assert _capture_files(output) == _capture_files(serial)
        assert report["analysis_sha256"] == serial_report["analysis_sha256"]
        assert _without_execution_context(report) == _without_execution_context(
            serial_report
        )
        assert _without_execution_context(replay) == _without_execution_context(
            serial_replay
        )


def _reversed(task):
    index, count = task
    time.sleep(0.15 * (count - index))
    return index


def _fail_late_or_early(task):
    index, delay = task
    time.sleep(delay)
    raise ValueError(f"task {index}")


def _sleep(seconds):
    time.sleep(seconds)
    return seconds


def test_results_come_back_in_task_order_whatever_order_workers_finish():
    with WorkerPool(3) as pool:
        tasks = [(index, 6) for index in range(6)]
        assert pool.map(_reversed, tasks) == list(range(6))
        # A later map reuses the same processes.
        assert pool.map(_reversed, tasks, concurrency=2) == list(range(6))


def test_the_earliest_failing_task_is_raised_as_in_order():
    # Task 1 fails first, but task 0, still running, fails too: running the
    # tasks in order would raise task 0's error, so the pool waits for it.
    with pytest.raises(ValueError, match=r"^task 0$"), WorkerPool(2) as pool:
        pool.map(_fail_late_or_early, [(0, 1.0), (1, 0.0), (2, 0.0)])
    assert multiprocessing.active_children() == []


def test_a_refusal_survives_the_process_boundary():
    refusal = WorkLimitExceeded(
        "max_vectorized_events_per_year", 50_000, "state", 10**9, 50
    ).in_arm("reference")
    copy = pickle.loads(pickle.dumps(refusal))
    assert type(copy) is WorkLimitExceeded
    assert (str(copy), copy.details, copy.arm) == (
        str(refusal),
        refusal.details,
        "reference",
    )


def test_an_interruption_stops_every_worker():
    # Ctrl-C while both workers run: the parent stops and reaps them at once
    # instead of waiting for their tasks.
    timer = threading.Timer(2.0, os.kill, (os.getpid(), signal.SIGINT))
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt), WorkerPool(2) as pool:
            pool.map(_sleep, [60, 60, 60])
    finally:
        timer.cancel()
    assert multiprocessing.active_children() == []
    assert time.monotonic() - started < 30


def test_a_worker_refusal_stops_the_run_with_the_serial_message(tmp_path):
    # Every state of a runaway law is refused in the first round; with two
    # workers the first two simulations fail in worker processes, and the run
    # still stops with the first one's refusal, as the serial run does.
    messages = {}
    package = _arms_package(tmp_path, _GATE, both_gate=_BOTH_GATE, rate=1e9)
    for workers in (1, 2):
        with pytest.raises(WorkLimitExceeded) as refusal:
            run_setup(
                setup_path=package.root,
                output=tmp_path / f"run-{workers}",
                code_commit="fixture",
                workers=workers,
            )
        messages[workers] = str(refusal.value)
        assert not (tmp_path / f"run-{workers}").exists()
    assert messages[2] == messages[1]
    assert messages[1].startswith(
        "individually drawn events in state 'high' of arm 'reference'"
    )
    assert multiprocessing.active_children() == []


def test_memory_bounds_each_rounds_workers(monkeypatch):
    first = simulation_worker_bytes(1)
    assert first == worker_pool.WORKER_BASE_BYTES + worker_pool.WORKER_BYTES_PER_YEAR
    # Later rounds scale the last measured peak by the ratio of years, never
    # down.
    assert simulation_worker_bytes(200, measured=(300, 100)) == 600
    assert simulation_worker_bytes(100, measured=(300, 64)) == 469
    assert simulation_worker_bytes(50, measured=(300, 100)) == 300
    monkeypatch.setattr(worker_pool, "available_memory", lambda: 3 * first)
    assert memory_bounded_workers(8, first) == 3
    assert memory_bounded_workers(2, first) == 2
    # What this process will still hold leaves less for workers.
    assert memory_bounded_workers(8, first, reserved=first) == 2
    # More memory per worker; one worker always runs.
    assert memory_bounded_workers(8, simulation_worker_bytes(10**7)) == 1
    # Unknown memory runs one worker.
    monkeypatch.setattr(worker_pool, "available_memory", lambda: None)
    assert memory_bounded_workers(8, first) == 1


def test_cgroup_limits_bound_memory_and_cpus(tmp_path):
    proc = tmp_path / "cgroup"
    proc.write_text("4:memory:/job\n2:cpu,cpuacct:/job\n0::/job\n")
    (tmp_path / "memory" / "job").mkdir(parents=True)
    (tmp_path / "memory" / "job" / "memory.limit_in_bytes").write_text("1000\n")
    (tmp_path / "memory" / "job" / "memory.usage_in_bytes").write_text("400\n")
    (tmp_path / "job").mkdir()
    (tmp_path / "job" / "memory.max").write_text("900\n")
    (tmp_path / "job" / "memory.current").write_text("100\n")
    assert worker_pool.cgroup_memory_headroom(proc, tmp_path) == 600
    # Inactive page cache is reclaimable, so it is not counted as used: v1's
    # hierarchical total and v2's own counter.
    (tmp_path / "memory" / "job" / "memory.stat").write_text(
        "cache 500\ninactive_file 10\ntotal_inactive_file 300\n"
    )
    (tmp_path / "job" / "memory.stat").write_text("file 200\ninactive_file 50\n")
    assert worker_pool.cgroup_memory_headroom(proc, tmp_path) == 850
    # Active page cache, such as files just written, is reclaimable too, but
    # file pages mapped into a process are not counted as free.
    (tmp_path / "memory" / "job" / "memory.stat").write_text(
        "total_inactive_file 300\ntotal_active_file 200\ntotal_mapped_file 50\n"
    )
    (tmp_path / "job" / "memory.stat").write_text(
        "inactive_file 50\nactive_file 40\nfile_mapped 60\n"
    )
    # v1: 400 used less 450 reclaimable; v2: 100 used less 30.
    assert worker_pool.cgroup_memory_headroom(proc, tmp_path) == 830
    (tmp_path / "job" / "memory.stat").write_text("file 200\ninactive_file 50\n")
    (tmp_path / "memory" / "job" / "memory.stat").write_text(
        "cache 500\ninactive_file 10\ntotal_inactive_file 300\n"
    )
    (tmp_path / "job" / "memory.max").write_text("max\n")
    assert worker_pool.cgroup_memory_headroom(proc, tmp_path) == 900
    (tmp_path / "job" / "memory.max").write_text("900\n")
    for path in (tmp_path / "memory" / "job", tmp_path / "job"):
        (path / "memory.stat").unlink()
    (tmp_path / "cpu" / "job").mkdir(parents=True)
    (tmp_path / "cpu" / "job" / "cpu.cfs_quota_us").write_text("250000\n")
    (tmp_path / "cpu" / "job" / "cpu.cfs_period_us").write_text("100000\n")
    assert worker_pool.cgroup_memory_headroom(proc, tmp_path) == 600
    assert worker_pool.cgroup_cpu_quota(proc, tmp_path) == 3
    (tmp_path / "job" / "cpu.max").write_text("max 100000\n")
    (tmp_path / "cpu" / "job" / "cpu.cfs_quota_us").write_text("-1\n")
    assert worker_pool.cgroup_cpu_quota(proc, tmp_path) is None
    # Only the cgroup v2 quota: 1.5 CPUs round up to 2.
    (tmp_path / "job" / "cpu.max").write_text("150000 100000\n")
    assert worker_pool.cgroup_cpu_quota(proc, tmp_path) == 2
    (tmp_path / "memory" / "job" / "memory.limit_in_bytes").write_text(f"{1 << 62}\n")
    (tmp_path / "job" / "memory.max").write_text("max\n")
    assert worker_pool.cgroup_memory_headroom(proc, tmp_path) is None


@pytest.mark.parametrize(
    ("line", "mount", "files"),
    [
        (
            "4:memory:/docker/abc",
            "memory",
            ("memory.limit_in_bytes", "memory.usage_in_bytes"),
        ),
        ("0::/system.slice/abc", "", ("memory.max", "memory.current")),
    ],
)
def test_a_container_without_a_cgroup_namespace_reads_its_mount(
    tmp_path, line, mount, files
):
    # The path names the host's cgroup, which does not exist in the container;
    # the container's own cgroup is mounted at the controller's root.
    proc = tmp_path / "cgroup"
    proc.write_text(line + "\n")
    root = tmp_path / "fs"
    (root / mount).mkdir(parents=True)
    limit, usage = files
    (root / mount / limit).write_text("1000\n")
    (root / mount / usage).write_text("300\n")
    assert worker_pool.cgroup_memory_headroom(proc, root) == 700


def test_a_round_starts_no_more_workers_than_memory_holds(tmp_path, monkeypatch):
    package = _arms_package(tmp_path, _GATE, both_gate=_BOTH_GATE)
    monkeypatch.setattr(worker_pool, "available_memory", lambda: 1)

    def refuse(self, size):
        raise AssertionError("a worker process was started")

    monkeypatch.setattr(WorkerPool, "_executor_for", refuse)
    report = run_setup(
        setup_path=package.root,
        output=tmp_path / "run",
        code_commit="fixture",
        workers=3,
    )
    assert report["execution_context"] == {"workers": 3}


def test_the_run_command_takes_and_records_the_global_workers(tmp_path):
    package = _arms_package(tmp_path, _GATE)
    runner = CliRunner()
    for workers in ("0", "x"):
        result = runner.invoke(
            cli,
            [
                "--workers",
                workers,
                "run",
                "--setup",
                str(package.root),
                "--output",
                str(tmp_path / "refused"),
            ],
        )
        assert result.exit_code == 2
    for name, option in (("two", ["--workers", "2"]), ("default", [])):
        result = runner.invoke(
            cli,
            [
                *option,
                "run",
                "--setup",
                str(package.root),
                "--output",
                str(tmp_path / name),
            ],
        )
        assert result.exit_code == 0, result.output
    # The default is the CPUs available, each round bounded by memory.
    for name, workers in (("two", 2), ("default", available_cpus())):
        report = json.loads((tmp_path / name / "results.json").read_bytes())
        assert report["execution_context"] == {"workers": workers}


def _worker_pids():
    return [process.pid for process in multiprocessing.active_children()]


def test_workers_ignore_ctrl_c_and_finish_their_tasks():
    # Ctrl-C reaches every process of the group; a worker ignores it and the
    # main process decides. Here only the workers receive it, from the moment
    # each exists, while it still imports the code, until its task is done.
    done = threading.Event()

    def interrupt():
        while not done.is_set():
            for pid in _worker_pids():
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGINT)
            time.sleep(0.005)

    interrupter = threading.Thread(target=interrupt, daemon=True)
    interrupter.start()
    with WorkerPool(2) as pool:
        try:
            results = pool.map(_sleep, [1, 1])
        finally:
            done.set()
            interrupter.join()
    assert results == [1, 1]


_ORPHAN_SCRIPT = """
import multiprocessing, sys, threading, time
from catmodel.worker_pool import WorkerPool

def report():
    while len(multiprocessing.active_children()) < 2:
        time.sleep(0.05)
    print(" ".join(str(p.pid) for p in multiprocessing.active_children()), flush=True)

if __name__ == "__main__":
    threading.Thread(target=report, daemon=True).start()
    with WorkerPool(2) as pool:
        pool.map(time.sleep, [60, 60])
"""


def _ended(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
    except OSError:
        return True
    return state in {"Z", "X"}


@pytest.mark.parametrize("stop", [signal.SIGTERM, signal.SIGKILL])
def test_workers_end_when_the_main_process_dies(tmp_path, stop):
    script = tmp_path / "pool_script.py"
    script.write_text(_ORPHAN_SCRIPT)
    main = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert main.stdout is not None
    workers = []
    try:
        workers = [int(pid) for pid in main.stdout.readline().split()]
        assert len(workers) == 2
        main.send_signal(stop)
        # SIGTERM exits through the pool, which stops its workers; after
        # SIGKILL each worker sees its parent gone and exits by itself.
        assert main.wait(timeout=30) == (
            128 + signal.SIGTERM if stop == signal.SIGTERM else -stop
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not all(map(_ended, workers)):
            time.sleep(0.1)
        assert all(_ended(pid) for pid in workers)
    finally:
        # A failure must not leave the script or its workers running.
        main.kill()
        main.wait()
        for pid in workers:
            if not _ended(pid):
                os.kill(pid, signal.SIGKILL)


class _TwoArguments(ValueError):
    def __init__(self, first, second):
        super().__init__(f"{first} and {second}")


def _raise_two_arguments(_):
    raise _TwoArguments("one", "two")


def _fail_to_load():
    raise ValueError("this result cannot be loaded")


class _Unloadable:
    def __reduce__(self):
        return _fail_to_load, ()


def _return_unloadable(_):
    return _Unloadable()


def _kill_self(_):
    os.kill(os.getpid(), signal.SIGKILL)


def test_a_broken_pool_says_why():
    # An error that cannot be rebuilt here keeps its type name and message.
    with (
        pytest.raises(RuntimeError, match=r"^_TwoArguments: one and two$"),
        WorkerPool(2) as pool,
    ):
        pool.map(_raise_two_arguments, [0, 1])
    # A result that cannot be loaded here names the cause, not memory.
    with pytest.raises(ChildProcessError) as broken, WorkerPool(2) as pool:
        pool.map(_return_unloadable, [0, 1])
    assert "could not return its result" in str(broken.value)
    assert "this result cannot be loaded" in str(broken.value)
    assert "memory" not in str(broken.value)
    # A worker killed by a signal names it, and memory only for SIGKILL.
    with (
        pytest.raises(ChildProcessError, match=r"killed by SIGKILL.*fewer --workers"),
        WorkerPool(2) as pool,
    ):
        pool.map(_kill_self, [0, 1])
    assert multiprocessing.active_children() == []


def test_evidence_errors_survive_the_process_boundary():
    error = EvidenceValidationError("code", "message", location=("a", 1))
    copy = pickle.loads(pickle.dumps(error))
    assert (type(copy), copy.code, str(copy), copy.location) == (
        EvidenceValidationError,
        "code",
        "message",
        ("a", 1),
    )


def _warn(_):
    warnings.warn("a worker warning", RuntimeWarning, stacklevel=1)
    return True


def test_workers_keep_the_parents_warning_filters():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with (
            pytest.raises(RuntimeWarning, match="a worker warning"),
            WorkerPool(2) as pool,
        ):
            pool.map(_warn, [0, 1])


def test_workers_keep_a_filter_on_several_categories():
    with warnings.catch_warnings():
        warnings.simplefilter("error", (DeprecationWarning, RuntimeWarning))
        with (
            pytest.raises(RuntimeWarning, match="a worker warning"),
            WorkerPool(2) as pool,
        ):
            pool.map(_warn, [0, 1])


def test_round_tasks_are_built_on_demand_and_take_their_earlier_years():
    simulated = {"a": "years of a", "b": "years of b"}
    caches = {"a": {"split": 1}}
    plans = {"arm": "plan"}
    compiled = []

    class Layout:
        """Compiles a state when a task is built, and records it."""

        def compiled(self, arm, key):
            compiled.append((arm, key))
            return f"law of {key}"

    tasks = execution._round_tasks(
        {"a": ("arm", "state a"), "b": ("arm", "state b")},
        Layout(),
        plans,
        "response",
        {"arm": 100},
        simulated,
        caches,
    )
    assert compiled == []
    first = next(tasks)
    assert (first.compiled, first.previous, first.split_cache) == (
        "law of state a",
        "years of a",
        {"split": 1},
    )
    assert simulated == {"b": "years of b"} and compiled == [("arm", "state a")]
    second = next(tasks)
    assert (second.previous, second.split_cache, simulated) == ("years of b", {}, {})
    assert compiled == [("arm", "state a"), ("arm", "state b")]


def test_a_state_reuses_its_small_event_splits_across_rounds(tmp_path, monkeypatch):
    # The split cache a task returns, as a worker would send it, serves the
    # next round without recomputing the splits.
    gate = {"batches": 4, "max_reference_state_years": 128, "levels": [0.999]}
    package = _package(
        tmp_path, simulations=64, gate=gate, rate=4_000.0, kernel=_kernel(member=0.9)
    )
    setup = load_setup(package.root)
    plan = setup.execution.for_arm("reference")
    compiled = compile_incident_state(
        setup, setup.joint_states[0].state_id, "reference"
    )
    computed = []
    splits = incident_runtime._small_event_splits

    def counted(*args, **kwargs):
        computed.append(True)
        return splits(*args, **kwargs)

    monkeypatch.setattr(incident_runtime, "_small_event_splits", counted)
    first, cache = execution._extend(
        execution._Extension("reference", plan, compiled, "r", 64, None, {})
    )
    assert cache and len(computed) == 1
    cache = pickle.loads(pickle.dumps(cache))
    first = pickle.loads(pickle.dumps(first))
    execution._extend(
        execution._Extension("reference", plan, compiled, "r", 128, first, cache)
    )
    assert len(computed) == 1
    # A run keeps each state's cache between rounds.
    computed.clear()
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    assert report["tail_gate"]["doublings"] == 1
    assert len(computed) == 1


def test_export_accepts_only_a_worker_count_as_execution_context(tmp_path):
    package = _arms_package(tmp_path, _GATE)
    run_setup(
        setup_path=package.root,
        output=tmp_path / "run",
        code_commit="fixture",
        workers=2,
    )
    results = tmp_path / "run" / "results.json"
    report = json.loads(results.read_bytes())
    for index, (context, message) in enumerate(
        (
            # The results schema closes the record and types its count.
            (
                {"workers": 2, "note": "approved for external reliance"},
                "execution_context.note is not a field of the schema",
            ),
            ({"workers": True}, "execution_context.workers is true, not an integer"),
            (
                {"workers": 2, "expected_annual_loss": 1.0},
                "execution_context.expected_annual_loss is not a field of the schema",
            ),
            ([2], "execution_context is an array, not an object"),
            (None, "execution_context is null, not an object"),
            # An integer the schema admits, which no run records.
            ({"workers": 0}, "exactly a worker count"),
        )
    ):
        forged = tmp_path / f"forged-{index}.json"
        forged.write_text(json.dumps({**report, "execution_context": context}))
        # The run's diagnostics file, which holds its exhibits, beside it.
        shutil.copy(diagnostics_path(results), diagnostics_path(forged))
        with pytest.raises(ValueError, match=message):
            export_results(
                setup_path=package.root,
                capture_directory=tmp_path / "run" / "capture",
                results_path=forged,
                output=tmp_path / f"export-{index}",
                code_commit="fixture",
            )
    export_results(
        setup_path=package.root,
        capture_directory=tmp_path / "run" / "capture",
        results_path=results,
        output=tmp_path / "export",
        code_commit="fixture",
    )


def test_a_script_without_an_entry_point_guard_is_told_so(tmp_path):
    script = tmp_path / "unguarded.py"
    script.write_text(
        "import time\n"
        "from catmodel.worker_pool import WorkerPool\n"
        "with WorkerPool(2) as pool:\n"
        "    pool.map(time.sleep, [0.1, 0.1])\n"
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert result.returncode != 0
    assert 'guard its entry point with if __name__ == "__main__":' in result.stderr


def test_each_round_reserves_its_arrays_and_sizes_workers_from_the_last_peak(
    tmp_path, monkeypatch
):
    package = _arms_package(tmp_path, _GATE, both_gate=_BOTH_GATE)
    bounds, peaks = [], []
    bound = execution.memory_bounded_workers

    def recorded_bound(workers, per_worker, reserved=0):
        bounds.append((per_worker, reserved))
        return bound(workers, per_worker, reserved)

    pooled = WorkerPool.map

    def recorded_map(self, function, tasks, **kwargs):
        result = pooled(self, function, tasks, **kwargs)
        peaks.append(self.peak_task_bytes)
        return result

    monkeypatch.setattr(execution, "memory_bounded_workers", recorded_bound)
    monkeypatch.setattr(WorkerPool, "map", recorded_map)
    run_setup(
        setup_path=package.root,
        output=tmp_path / "run",
        code_commit="fixture",
        workers=2,
    )
    # The first round extends three states to 50 years of one view, so this
    # process will hold 3 x 50 floats; its workers are sized by the formula.
    assert bounds[0] == (simulation_worker_bytes(50), 8 * 50 * 3)
    # The next round's workers are sized at twice the first round's peak.
    assert peaks[0] is not None and bounds[1] == (2 * peaks[0], 8 * 100 * 3)


def test_a_round_run_here_measures_its_tasks_for_the_next(tmp_path, monkeypatch):
    # Rounds of 50, 100, 200 and more years; the second runs in this
    # process, as when memory is short, and the third is sized from it.
    package = _arms_package(tmp_path, _GATE, both_gate=_BOTH_GATE)
    sizes, peaks = [], []
    pooled = WorkerPool.map

    def recorded_bound(workers, per_worker, reserved=0):
        sizes.append(per_worker)
        return 1 if len(sizes) == 2 else 2

    def recorded_map(self, function, tasks, **kwargs):
        result = pooled(self, function, tasks, **kwargs)
        peaks.append(self.peak_task_bytes)
        return result

    monkeypatch.setattr(execution, "memory_bounded_workers", recorded_bound)
    monkeypatch.setattr(WorkerPool, "map", recorded_map)
    run_setup(
        setup_path=package.root,
        output=tmp_path / "run",
        code_commit="fixture",
        workers=2,
    )
    assert len(sizes) >= 3 and None not in peaks
    # Measured here: what the task raised this process's peak above its
    # resident memory, plus a worker's own imports.
    assert peaks[1] >= worker_pool.WORKER_BASE_BYTES
    assert sizes[1:3] == [2 * peaks[0], 2 * peaks[1]]
    # Where this process cannot measure, the earlier measurement is scaled.
    sizes.clear()
    peaks.clear()
    monkeypatch.setattr(worker_pool, "_resident_bytes", lambda: None)
    run_setup(
        setup_path=package.root,
        output=tmp_path / "unmeasured",
        code_commit="fixture",
        workers=2,
    )
    assert peaks[1] is None
    assert peaks[0] is not None and sizes[1:3] == [2 * peaks[0], 4 * peaks[0]]


def _callers_handler(signum, frame):
    raise AssertionError("the caller's own SIGTERM handler")


def test_sigterm_is_handled_only_while_workers_run():
    previous = signal.signal(signal.SIGTERM, _callers_handler)
    try:
        with WorkerPool(2) as pool:
            # Tasks run here leave a serial caller's SIGTERM alone.
            assert pool.map(_sleep, [0, 0], concurrency=1) == [0, 0]
            assert signal.getsignal(signal.SIGTERM) is _callers_handler
            assert pool.map(_sleep, [0, 0]) == [0, 0]
            assert signal.getsignal(signal.SIGTERM) is worker_pool._exit_on_sigterm
        assert signal.getsignal(signal.SIGTERM) is _callers_handler
    finally:
        signal.signal(signal.SIGTERM, previous)


def _resettable() -> bool:
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as handle:
            handle.write("5")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _resettable(), reason="the peak cannot be reset here")
def test_a_task_run_here_is_not_charged_an_earlier_peak(monkeypatch):
    # An earlier peak 400 MB above resident memory, released before the task.
    earlier = b"\x01" * (400 * 2**20)
    del earlier
    _, peak = worker_pool._run_here(lambda task: task, [1])
    assert peak is not None
    assert peak < worker_pool.WORKER_BASE_BYTES + 200 * 2**20
    # Where the peak cannot be reset, the process's own peak counts in full.
    monkeypatch.setattr(worker_pool, "_reset_peak", lambda: None)
    earlier = b"\x01" * (400 * 2**20)
    del earlier
    _, peak = worker_pool._run_here(lambda task: task, [1])
    assert peak is not None
    assert peak >= worker_pool.WORKER_BASE_BYTES + 300 * 2**20


def _peak_of_worker(_: int) -> int:
    return worker_pool._peak_bytes()


@pytest.mark.skipif(sys.platform != "linux", reason="VmHWM is Linux's")
def test_a_worker_reports_its_own_peak_not_its_parents():
    # A parent 600 MB above a worker's size: Linux starts a spawned child's
    # ru_maxrss at the parent's resident memory, but not its own VmHWM.
    held = b"\x01" * (600 * 2**20)
    with WorkerPool(2) as pool:
        pool.map(_peak_of_worker, [0, 1], concurrency=2)
        assert pool.peak_task_bytes is not None
        assert pool.peak_task_bytes < len(held)
