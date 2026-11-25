"""The coverage-gap register names live scenarios and reaches result limitations."""

from pathlib import Path

import pytest
import yaml

from catmodel import resources
from catmodel.resources import scenarios_path
from catmodel.scenarios.coverage_register import (
    coverage_register_errors,
    load_coverage_register,
)
from catmodel.scenarios.loader import load_scenarios

REGISTER_PATH = Path(__file__).parents[1] / "registries" / "coverage_gaps.yaml"


def test_repository_register_satisfies_the_library_rules():
    scenarios = load_scenarios(scenarios_path())
    register = load_coverage_register(REGISTER_PATH)
    assert register.gaps
    assert coverage_register_errors(register, scenarios) == []


def test_denominator_rows_name_the_scenarios_with_a_normalisation_base():
    """Library validation requires a row for every scenario with a normalisation
    base; the repository register also keeps none for a scenario without one."""
    register = load_coverage_register(REGISTER_PATH)
    named = {
        scenario_id
        for gap in register.gaps
        if gap.gap_type == "ungoverned_exposure_denominator"
        for scenario_id in gap.scenario_ids
    }
    assert named == {
        scenario.id
        for scenario in load_scenarios(scenarios_path())
        if scenario.normalisation_bases
    }


def test_rows_are_selected_by_any_named_scenario():
    register = load_coverage_register(REGISTER_PATH)
    selected = register.for_scenarios({"private_claims_privacy"})
    assert selected.gaps
    assert all("private_claims_privacy" in gap.scenario_ids for gap in selected.gaps)
    assert not register.for_scenarios({"not_a_scenario"}).gaps


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (
            lambda payload: payload["gaps"].append(dict(payload["gaps"][0])),
            "unique IDs",
        ),
        (
            lambda payload: payload["gaps"][0].update(
                scenario_ids=payload["gaps"][0]["scenario_ids"] * 2
            ),
            "repeats a scenario",
        ),
        (lambda payload: payload["gaps"][0].update(scenario_ids=[]), "at least 1"),
        (lambda payload: payload["gaps"][0].update(rationale=" "), "must not be blank"),
        (
            lambda payload: payload.update(unknown_field="value"),
            "Extra inputs are not permitted",
        ),
        (
            lambda payload: payload["gaps"][0].update(unknown_field="value"),
            "Extra inputs are not permitted",
        ),
    ],
)
def test_malformed_registers_are_refused(tmp_path, edit, match):
    payload = yaml.safe_load(REGISTER_PATH.read_text(encoding="utf-8"))
    edit(payload)
    malformed = tmp_path / "coverage_gaps.yaml"
    malformed.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_coverage_register(malformed)


def test_results_list_the_gap_rows_of_their_scenarios():
    from types import SimpleNamespace
    from typing import cast

    from catmodel.contracts.results import NativeResults
    from catmodel.contracts.run_setup import RunSetup
    from catmodel.results.result_disclosures import result_limitations

    populations = [
        SimpleNamespace(scenario_id=s) for s in ("synthetic", "input_hijack")
    ]
    package = SimpleNamespace(
        limitations=("A setup note.",), scope=SimpleNamespace(populations=populations)
    )
    report = {"coverage": {"comparisons": {"status": "complete"}, "scenarios": []}}
    rows = result_limitations(cast(RunSetup, package), cast(NativeResults, report))
    register = load_coverage_register(REGISTER_PATH)
    expected = {gap.gap_id for gap in register.for_scenarios({"input_hijack"}).gaps}
    gap_rows = [
        row for row in rows if row["text"].startswith("Registered coverage gap")
    ]
    assert len(expected) == len(gap_rows) >= 2
    assert all(row["source"] == "setup" for row in gap_rows)
    assert all("of input_hijack (" in row["text"] for row in gap_rows)
    assert {row["text"].split()[3] for row in gap_rows} == expected
    # The setup's own limitations come first, then its scenarios' gaps.
    assert rows[0] == {"source": "setup", "text": "A setup note."}
    assert rows[1 : 1 + len(gap_rows)] == gap_rows


def test_default_register_uses_packaged_resources_outside_a_checkout(
    tmp_path, monkeypatch
):
    payload = yaml.safe_load(REGISTER_PATH.read_text(encoding="utf-8"))
    payload["gaps"][0]["rationale"] = "Packaged register fixture."
    bundled = tmp_path / "_resources" / "registries" / "coverage_gaps.yaml"
    bundled.parent.mkdir(parents=True)
    bundled.write_text(yaml.safe_dump(payload), encoding="utf-8")
    monkeypatch.setattr(resources, "source_checkout_root", lambda: None)
    monkeypatch.setattr(resources, "files", lambda _: tmp_path)
    monkeypatch.chdir(tmp_path)
    assert load_coverage_register().gaps[0].rationale == "Packaged register fixture."
