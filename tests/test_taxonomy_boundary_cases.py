"""Authored narrative boundaries, classification replay and incident accounting.

These tests replay the prose examples' explicitly resolved facts.
"""

from pathlib import Path
from typing import get_args

import pytest
import yaml

from catmodel.economic_loss import EconomicNature
from catmodel.resources import scenarios_path
from catmodel.scenarios.hierarchy_contract import (
    load_hierarchy_contract,
    nodes_by_id,
)
from catmodel.scenarios.loader import load_scenarios
from test_support.hierarchy import scenario_owner_for_history, terminal_for_history

EXAMPLES = yaml.safe_load(
    (Path(__file__).parent / "fixtures/taxonomy_boundary_cases.yaml").read_text()
)
CASES = EXAMPLES["cases"]
# Histories at the boundaries between leaves that the fixture must resolve.
REQUIRED_HISTORIES = {
    "phishing_written_ransomware_human",
    "phishing_and_ransomware_both_ai_written",
    "intrusion_supporting_bio_attack",
    "one_defect_across_roles",
    "worm_and_vendor_update",
    "whole_enforcement_matter",
    "model_crafted_prompt_injection",
    "stolen_weights_then_bio_uplift",
    "drug_checker_defect",
    "small_parties_large_campaigns",
    "serving_bug_outage_then_wrong_answers",
    "designation_with_fine",
    "deepfake_moves_stock_and_solicits",
    "insurance_claim_ring",
    "defamation_and_self_harm",
    "agent_cascade_size",
    "teen_self_harm_identified_defect",
    "teen_self_harm_without_identified_defect",
    "state_action_over_several_injury_incidents",
    "model_used_without_gain",
    "another_model_could_have_helped",
    "speed_only_help",
    "injury_claim_over_attack_without_gain",
    "victims_assistant_gives_up_credentials",
    "plant_halt_without_damage",
    "intrusion_edits_agent_system_prompt",
    "intrusion_edits_agent_system_prompt_data_theft_dominates",
}


@pytest.fixture(scope="module")
def contract():
    return load_hierarchy_contract()


@pytest.fixture(scope="module")
def scenarios():
    return {scenario.id: scenario for scenario in load_scenarios(scenarios_path())}


@pytest.mark.parametrize(
    "case,answer",
    [
        pytest.param(case, answer, id=f"{case['id']}-{i + 1}")
        for case in CASES
        for i, answer in enumerate(case["expected"]["classifications"])
    ],
)
def test_authored_boundary_has_the_expected_terminal_and_forecast(
    case, answer, contract, scenarios
):
    assert answer["subject"] in case["prompt"]["subjects"]
    terminal = terminal_for_history(answer["facts"], contract=contract)
    assert terminal == answer["terminal_id"]
    assert (
        scenario_owner_for_history(
            answer["facts"], contract=contract, scenarios=list(scenarios.values())
        )
        == answer["scenario_id"]
    )
    if answer["scenario_id"] is None:
        disposition = nodes_by_id(contract)[terminal].unestimated_disposition
        assert disposition is not None
        assert disposition.disposition == (
            "unsupported"
            if answer["facts"]["origin_regime"] == "other"
            else "unestimated"
        )


def test_examples_cover_the_required_histories_and_each_root_answer(contract):
    assert EXAMPLES["evidence_kind"] == "authored_examples"
    assert len({case["id"] for case in CASES}) == len(CASES)
    assert {case["id"] for case in CASES} >= REQUIRED_HISTORIES
    assert {
        answer["facts"]["origin_regime"]
        for case in CASES
        for answer in case["expected"]["classifications"]
    } == set(contract.axes["origin_regime"].values)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_each_incident_belongs_wholly_to_one_leaf(case):
    by_incident: dict[str, set[tuple[str, str | None]]] = {}
    for answer in case["expected"]["classifications"]:
        by_incident.setdefault(answer["incident_id"], set()).add(
            (answer["terminal_id"], answer["scenario_id"])
        )
    assert all(len(owners) == 1 for owners in by_incident.values()), by_incident


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_admission_uses_the_whole_incidents_gross_loss(case, contract):
    threshold = contract.admission.threshold
    answers = case["expected"]["classifications"]
    gross: dict[str, float] = {}
    for answer in answers:
        gross[answer["incident_id"]] = (
            gross.get(answer["incident_id"], 0) + answer["gross_amount_usd"]
        )
    assert case["expected"]["admitted"] == {
        incident: total >= threshold for incident, total in gross.items()
    }


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_incident_accounting_counts_unique_losses_once(case):
    answers = case["expected"]["classifications"]
    accounting = case["expected"]["accounting"]
    admitted = case["expected"]["admitted"]
    assert len({answer["loss_id"] for answer in answers}) == len(answers)
    assert (
        sum(answer["gross_amount_usd"] for answer in answers)
        == accounting["unique_gross_usd"]
    )
    assert (
        len({answer["incident_id"] for answer in answers})
        == accounting["unique_incident_count"]
    )
    counted: dict[str, set[str]] = {}
    for answer in answers:
        if answer["scenario_id"] is not None and admitted[answer["incident_id"]]:
            counted.setdefault(answer["scenario_id"], set()).add(answer["incident_id"])
    assert {
        scenario_id: len(incidents) for scenario_id, incidents in counted.items()
    } == accounting["admitted_scenario_incident_counts"]


def test_enforcement_matter_is_dated_by_its_conduct_start(scenarios):
    (case,) = [case for case in CASES if case["id"] == "whole_enforcement_matter"]
    assert case["expected"]["matter_year"] == 2027
    year_rule = scenarios["enforcement_matter"].frequency_semantics.year_rule
    assert "course of conduct it concerns first occurs" in year_rule


def test_each_loss_names_an_economic_nature():
    natures = set(get_args(EconomicNature))
    for case in CASES:
        for answer in case["expected"]["classifications"]:
            assert answer["economic_nature"] in natures, (case["id"], answer["loss_id"])


def test_exclusivity_lost_with_stolen_weights_is_an_appropriated_transfer(scenarios):
    (case,) = [case for case in CASES if case["id"] == "stolen_weights_then_bio_uplift"]
    (exclusivity,) = [
        answer
        for answer in case["expected"]["classifications"]
        if answer["loss_id"] == "exclusivity"
    ]
    # The theft scenario's composition holds stolen exclusivity in this cell.
    cell = next(
        cell
        for cell in scenarios["model_asset_compromise"].estimand_composition.cells
        if cell.cell_id == "provider_distinct_transfer_appropriated"
    )
    assert "exclusivity" in cell.loss_form
    assert exclusivity["economic_nature"] == cell.nature == "appropriated_transfer"


@pytest.mark.parametrize(
    "case,unresolved",
    [
        pytest.param(case, unresolved, id=f"{case['id']}-{i + 1}")
        for case in CASES
        for i, unresolved in enumerate(case["expected"].get("unresolved", []))
    ],
)
def test_unresolved_classification_does_not_become_other(case, unresolved, contract):
    with pytest.raises(ValueError, match=unresolved["missing_fact"]):
        terminal_for_history(unresolved["facts"], contract=contract)
    assert unresolved["interpretation"]
