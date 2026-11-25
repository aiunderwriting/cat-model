"""The collection trace tool runs end to end on one context.

It traces scheduling changes against the real collection code, so this smoke
test keeps it from drifting as collection changes; one context at one sample
takes about 20 seconds with its CPU profile, most of it building the study.
"""

import cProfile
import json

import pytest
import yaml

from catmodel.config import EXAMPLE_PROFILES
from catmodel.resources import data_path
from scripts import trace_collection
from scripts.trace_collection import CPU_STEPS, cpu_breakdown, main


def test_the_trace_tool_traces_one_context(tmp_path):
    # A profiles file of its own: the example's test profile at another
    # ceiling, read both for the slots and for the study's respondents.
    payload = yaml.safe_load(data_path(EXAMPLE_PROFILES).read_text(encoding="utf-8"))
    payload["profiles"]["test"]["settings"]["max_concurrency"] = 7
    profiles = tmp_path / "my-profiles.yaml"
    profiles.write_text(yaml.safe_dump(payload), encoding="utf-8")
    output = tmp_path / "trace.json"
    arguments = ["--samples", "1", "--contexts", "1", "--profile-cpu"]
    assert main([*arguments, "--profiles", str(profiles), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["contexts"] == 1 and report["requests"] > 0
    assert sum(report["requests_by_kind"].values()) == report["requests"]
    path = report["critical_path"]
    assert path["turns"] >= 2
    assert 0 < path["seconds"] <= report["wall_clock_hours"] * 3600
    # The supplied profile's ceiling for every model, each of which answered.
    assert {row["ceiling"] for row in report["by_model"].values()} == {7}
    assert all(row["requests"] > 0 for row in report["by_model"].values())
    # The CPU profile, run inline, measures every step: each of its functions
    # ran, and none took more than the process did.
    cpu = report["cpu_ms_per_request"]
    assert set(cpu) == {"whole process", "catmodel frames", *CPU_STEPS}
    assert report["cpu_entries_without_samples"] == []
    assert all(
        value is not None and 0 < value <= cpu["whole process"]
        for value in cpu.values()
    )
    assert cpu["whole process"] >= cpu["catmodel frames"]


def test_a_step_whose_functions_never_ran_is_unmeasured_not_zero(monkeypatch):
    def worked() -> int:
        return sum(range(1000))

    monkeypatch.setattr(
        trace_collection,
        "CPU_STEPS",
        {
            "ran": ((__file__, "worked"),),
            "renamed": ((__file__, "no_such_function"),),
            "partly": ((__file__, "worked"), ("no/such/file.py", "worked")),
        },
    )
    profile = cProfile.Profile()
    profile.enable()
    worked()
    profile.disable()
    rows, without_samples = cpu_breakdown(profile, requests=1, process_seconds=1.0)
    assert rows["renamed"] is None
    ran = rows["ran"]
    assert ran is not None and ran >= 0
    assert rows["partly"] == ran
    assert without_samples == [
        f"renamed: {__file__} no_such_function",
        "partly: no/such/file.py worked",
    ]


def test_a_profile_file_needs_the_cpu_profile(tmp_path, capsys):
    target = tmp_path / "trace.prof"
    with pytest.raises(SystemExit) as stopped:
        main(["--samples", "1", "--contexts", "1", "--profile-output", str(target)])
    assert stopped.value.code == 2
    assert "--profile-cpu" in capsys.readouterr().err
    assert not target.exists()
