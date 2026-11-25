"""The collection timing tool aggregates stored attempts by their times only."""

import json
import os

import pytest

from catmodel.artifacts import freeze_json
from catmodel.frozen_calls import frozen_call_key
from catmodel.setup.contextual_inputs import contextual_requests
from scripts.collection_timing import main
from test_support.contextual_risk import contextual_instrument

SECRET = "response text that must not be printed"


@pytest.fixture
def stored_requests(tmp_path):
    """A study at ``tmp_path / "study"`` with one stored attempt per request,
    the first of which lost a background run to its deadline."""
    instrument = contextual_instrument()
    capture = tmp_path / "study" / "collections" / "node-00-public"
    capture.mkdir(parents=True)
    freeze_json(
        capture / "contextual-instrument.json", instrument.model_dump(mode="json")
    )
    requests = contextual_requests(instrument)
    origin = 1_800_000_000.0
    for index, request in enumerate(requests):
        call = capture / "calls" / frozen_call_key(request)
        call.mkdir(parents=True)
        freeze_json(call / "request.json", request)
        lost = [{"elapsed_ms": 2_400_000, "cause": "deadline"}] if index == 0 else []
        attempt = call / "attempt-001.json"
        freeze_json(
            attempt,
            {
                "parse_success": True,
                "latency_ms": 300_000 + (2_400_000 if lost else 0),
                "raw_response": SECRET,
                "provider_evidence": {
                    "status_first_seen_ms": {"queued": 0, "in_progress": 5_000},
                    "connection_lost": lost,
                },
            },
        )
        end = origin + 300 + (2_400 if lost else 0) + 60 * index
        os.utime(attempt, (end, end))
    return requests


def test_the_timing_tool_reports_rounds_resends_and_flight(
    stored_requests, tmp_path, capsys
):
    requests = stored_requests
    output = tmp_path / "timing.json"
    assert main([str(tmp_path / "study"), "--output", str(output)]) == 0
    assert SECRET not in capsys.readouterr().out
    report = json.loads(output.read_text())
    assert report["calls"] == len(requests)
    assert sum(row["calls"] for row in report["by_round"].values()) == len(requests)
    assert report["resends"]["lost_runs_by_cause"]["deadline"]["runs"] == 1
    assert report["resends"]["calls_with_lost_runs"] == 1
    assert report["by_round"]["direct"]["queue_s"]["p50"] == 5.0
    # The call that lost a run waited 2,700 s; taking the lost run out leaves
    # the 300 s every call took.
    path = report["critical_path"]
    assert path["observed"]["minutes"] == 45.0
    assert path["without_lost_runs"]["minutes"] == 5.0
    flight = report["in_flight_by_minute"]
    assert max(row[2] for row in flight) >= 2
    assert sum(row[3] for row in flight) == len(requests)


def test_the_timing_tool_refuses_a_file_inside_the_study(
    stored_requests, tmp_path, capsys
):
    plan = tmp_path / "study" / "public-study.json"
    plan.write_text("{}")
    output = tmp_path / "timing.json"
    with pytest.raises(SystemExit) as stopped:
        main([str(plan), "--output", str(output)])
    assert stopped.value.code == 2
    captured = capsys.readouterr()
    assert str(plan) in captured.err and captured.out == ""
    assert not output.exists()
