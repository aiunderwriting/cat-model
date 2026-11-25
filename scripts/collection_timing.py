"""Timing-only aggregates of a collected public study's stored attempts.

A development tool for scheduling work. It reads every call's stored
attempt files under a public study's ``collections/`` directory and prints
when each round ran, how long calls took, how many were in flight minute by
minute, how much time background runs lost to their queue limit or deadline
before being sent again, and the data-dependency critical path of the
observed latencies. It prints no prompt or response text: a call's round,
context and subject are read from its frozen request only to group it.

Each attempt ends when its file was written (its modification time, which a
copy must preserve) and started its ``latency_ms`` earlier; the latency
covers every background run the attempt sent. Queue time is the provider's
first ``in_progress`` sighting of the run that answered.

    uv run python scripts/collection_timing.py ../hfl/public-study
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts.trace_collection import LEAF_STAGES, critical_path  # noqa: E402
from test_support.collection_trace import (  # noqa: E402
    CollectionTrace,
    TracedRequest,
    request_kind,
)

ROUNDS = (*LEAF_STAGES, "parent")


@dataclass
class Attempt:
    start: float
    end: float
    parsed: bool
    queue_seconds: float | None
    lost: list[tuple[str, float]]
    """Each lost background run's cause and seconds."""
    submit_failures: int


@dataclass
class Call:
    collection: str
    kind: str
    subject: str
    anchor_score: float
    respondent: tuple[str, str, int]
    model_id: str
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.attempts[0].start

    @property
    def end(self) -> float:
        return self.attempts[-1].end


def _subject(prompt: str, kind: str) -> str:
    """The scenario or category a question's heading names."""
    heading = "Category" if kind == "parent" else "Scenario"
    found = re.search(rf"^## {heading} ([^\s:]+)", prompt, re.MULTILINE)
    return "" if found is None else found.group(1)


def _anchor(prompt: str) -> float:
    """A fixed capability coordinate's score; 0 on a public trajectory, whose
    questions state no score."""
    found = re.search(r"capability \([^)]*\) at exactly ([-+.0-9e]+) ", prompt)
    return 0.0 if found is None else float(found.group(1))


def _attempt(path: Path) -> Attempt:
    row = json.loads(path.read_bytes())
    end = path.stat().st_mtime
    latency = (row.get("latency_ms") or 0) / 1000
    evidence = row.get("provider_evidence") or {}
    seen = evidence.get("status_first_seen_ms") or {}
    queue = seen.get("in_progress")
    return Attempt(
        start=end - latency,
        end=end,
        parsed=bool(row.get("parse_success")),
        queue_seconds=None if queue is None else queue / 1000,
        lost=[
            (str(lost.get("cause")), (lost.get("elapsed_ms") or 0) / 1000)
            for lost in evidence.get("connection_lost") or ()
        ],
        submit_failures=len(evidence.get("submit_failures") or ()),
    )


def load_calls(study: Path) -> tuple[list[Call], int]:
    """Every call with at least one stored attempt, and the count of calls
    frozen without one."""
    calls: list[Call] = []
    pending = 0
    for capture in sorted((study / "collections").iterdir()):
        for directory in sorted((capture / "calls").iterdir()):
            attempts = sorted(directory.glob("attempt-*.json"))
            if not attempts:
                pending += 1
                continue
            request = json.loads((directory / "request.json").read_bytes())
            prompt = request["user_prompt"]
            kind = request_kind(prompt)
            respondent = request["respondent"]
            calls.append(
                Call(
                    collection=capture.name,
                    kind=kind,
                    subject=_subject(prompt, kind),
                    anchor_score=_anchor(prompt),
                    respondent=(
                        respondent["respondent_id"],
                        respondent["model_id"],
                        respondent["sample_index"],
                    ),
                    model_id=respondent["model_id"],
                    attempts=[_attempt(path) for path in attempts],
                )
            )
    return calls, pending


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return dict.fromkeys(("p50", "p90", "p99", "max"), None)
    p50, p90, p99 = np.percentile(values, [50, 90, 99])
    return {
        "p50": round(float(p50), 1),
        "p90": round(float(p90), 1),
        "p99": round(float(p99), 1),
        "max": round(max(values), 1),
    }


def _group(calls: Iterable[Call], origin: float) -> dict[str, Any]:
    rows = list(calls)
    attempts = [attempt for call in rows for attempt in call.attempts]
    lost = [seconds for attempt in attempts for _, seconds in attempt.lost]
    return {
        "calls": len(rows),
        "attempts": len(attempts),
        "unparsed_attempts": sum(not attempt.parsed for attempt in attempts),
        "first_start_min": round((min(c.start for c in rows) - origin) / 60, 1),
        "last_end_min": round((max(c.end for c in rows) - origin) / 60, 1),
        "attempt_latency_s": _percentiles([a.end - a.start for a in attempts]),
        "queue_s": _percentiles(
            [a.queue_seconds for a in attempts if a.queue_seconds is not None]
        ),
        "lost_runs": len(lost),
        "lost_run_hours": round(math.fsum(lost) / 3600, 2),
    }


IN_FLIGHT_COLUMNS = ("minute", "mean_in_flight", "peak_in_flight", "calls_finished")


def in_flight_by_minute(calls: Sequence[Call], origin: float) -> list[list[float]]:
    """Each minute's mean and peak attempts in flight and calls finished,
    as rows of ``IN_FLIGHT_COLUMNS``."""
    attempts = [attempt for call in calls for attempt in call.attempts]
    minutes = math.ceil((max(a.end for a in attempts) - origin) / 60) + 1
    mean = np.zeros(minutes)
    delta = np.zeros(minutes * 60 + 2)
    finished = np.zeros(minutes, dtype=int)
    for attempt in attempts:
        start, end = attempt.start - origin, attempt.end - origin
        for minute in range(int(start // 60), int(end // 60) + 1):
            mean[minute] += max(
                0.0, min(end, 60 * minute + 60) - max(start, 60 * minute)
            )
        delta[int(start)] += 1
        delta[int(end) + 1] -= 1
    for call in calls:
        finished[int((call.end - origin) // 60)] += 1
    level = np.cumsum(delta)[: minutes * 60]
    return [
        [
            minute,
            round(mean[minute] / 60, 1),
            int(level[60 * minute : 60 * minute + 60].max()),
            int(finished[minute]),
        ]
        for minute in range(minutes)
    ]


def tail(calls: Sequence[Call], origin: float) -> dict[str, Any]:
    """What finished after 90%, 99% and 99.9% of calls had."""
    ends = sorted(call.end for call in calls)
    last = ends[-1]
    report = {}
    for share in (0.9, 0.99, 0.999):
        cut = ends[min(len(ends) - 1, int(share * len(ends)))]
        late = [call for call in calls if call.end > cut]
        report[f"after_{share:g}"] = {
            "at_min": round((cut - origin) / 60, 1),
            "minutes_to_last": round((last - cut) / 60, 1),
            "calls": len(late),
            "by_round": dict(Counter(call.kind for call in late).most_common()),
            "with_lost_runs": sum(
                any(attempt.lost for attempt in call.attempts) for call in late
            ),
        }
    return report


def resends(calls: Sequence[Call]) -> dict[str, Any]:
    """Background runs lost to their queue limit, deadline or a failure,
    each sent again; and how many calls needed a second validation attempt."""
    by_cause: dict[str, list[float]] = defaultdict(list)
    for call in calls:
        for attempt in call.attempts:
            for cause, seconds in attempt.lost:
                by_cause[cause].append(seconds)
    return {
        "lost_runs_by_cause": {
            cause: {
                "runs": len(values),
                "hours": round(math.fsum(values) / 3600, 2),
                "seconds": _percentiles(values),
            }
            for cause, values in sorted(by_cause.items())
        },
        "calls_with_lost_runs": sum(
            any(attempt.lost for attempt in call.attempts) for call in calls
        ),
        "submit_failures": sum(a.submit_failures for c in calls for a in c.attempts),
        "calls_with_validation_retries": sum(len(c.attempts) > 1 for c in calls),
    }


def observed_critical_path(calls: Sequence[Call], *, lost: bool) -> dict[str, Any]:
    """The longest dependency chain of one (context, anchor, respondent)
    group at the observed call times with unlimited slots (see
    ``trace_collection.critical_path``); with ``lost`` false, time spent on
    lost background runs is taken out."""
    trace = CollectionTrace()
    for token, call in enumerate(calls):
        spent = math.fsum(a.end - a.start for a in call.attempts)
        if not lost:
            spent -= math.fsum(s for a in call.attempts for _, s in a.lost)
        trace.requests.append(
            TracedRequest(
                token=token,
                kind=call.kind,
                context_id=call.collection,
                subject=call.subject,
                anchor_score=call.anchor_score,
                respondent=call.respondent,
                model_id=call.model_id,
                ready=call.start,
                start=call.start,
                admitted=0.0,
                end=max(spent, 0.0),
            )
        )
    path = critical_path(trace)
    return {"minutes": round(path["seconds"] / 60, 1), "turns": path["turns"]}


def report(study: Path) -> dict[str, Any]:
    calls, pending = load_calls(study)
    if not calls:
        raise SystemExit("no stored attempts")
    origin = min(call.start for call in calls)
    wall = max(call.end for call in calls) - origin
    attempt_seconds = math.fsum(a.end - a.start for c in calls for a in c.attempts)
    by_round = defaultdict(list)
    by_collection = defaultdict(list)
    for call in calls:
        by_round[call.kind].append(call)
        by_collection[call.collection].append(call)
    mtimes = sorted(call.end for call in calls)
    return {
        "calls": len(calls),
        "calls_without_attempts": pending,
        "wall_minutes": round(wall / 60, 1),
        "attempt_slot_hours": round(attempt_seconds / 3600, 1),
        "mean_in_flight": round(attempt_seconds / wall, 1) if wall else None,
        "end_times_spread_minutes": round((mtimes[-1] - mtimes[0]) / 60, 1),
        "by_round": {
            kind: _group(by_round[kind], origin) for kind in ROUNDS if kind in by_round
        },
        "by_collection": {
            name: _group(rows, origin) for name, rows in sorted(by_collection.items())
        },
        "resends": resends(calls),
        "tail": tail(calls, origin),
        "critical_path": {
            "observed": observed_critical_path(calls, lost=True),
            "without_lost_runs": observed_critical_path(calls, lost=False),
        },
        "in_flight_columns": list(IN_FLIGHT_COLUMNS),
        "in_flight_by_minute": in_flight_by_minute(calls, origin),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("study", type=Path, help="a public study's directory")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.study.is_file():
        parser.error(f"{args.study} is a file; give the public study's directory")
    values = report(args.study)
    rows = values.pop("in_flight_by_minute")
    # One line per minute keeps the per-minute table readable.
    text = json.dumps(values, indent=1)[:-2] + ',\n "in_flight_by_minute": [\n'
    text += ",\n".join(f"  {json.dumps(row)}" for row in rows) + "\n ]\n}"
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
