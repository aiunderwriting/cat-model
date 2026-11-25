from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from catmodel.scenarios.loader import (
    EstimandCompositionDef,
    ScenarioDefinition,
    load_scenarios,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCENARIOS_DIR = _PROJECT_ROOT / "scenarios"


def _composition_payload() -> dict[str, Any]:
    return {
        "status": "INITIAL_ESTIMATE",
        "rationale": "  Classification of gross loss for economic views.  ",
        "evidence": None,
        "cells": [
            {
                "cell_id": "third_party_injury",
                "loss_form": "  Third-party injury valuation.  ",
                "nature": "injury_valuation",
                "subject": "third_party",
            },
            {
                "cell_id": "provider_remediation",
                "loss_form": "Provider remediation resources.",
                "nature": "resource_cost",
                "subject": "provider",
            },
        ],
    }


def _scenario_payload(scenario_id: str) -> dict[str, Any]:
    return next(
        scenario
        for scenario in load_scenarios(_SCENARIOS_DIR)
        if scenario.id == scenario_id
    ).model_dump(mode="json")


def test_valid_estimand_composition_is_accepted_and_normalized() -> None:
    composition = EstimandCompositionDef.model_validate(_composition_payload())

    assert composition.rationale == "Classification of gross loss for economic views."
    assert composition.cells[0].loss_form == "Third-party injury valuation."
    assert composition.tracked_cells == ()


@pytest.mark.parametrize("field", ["weight", "share"])
def test_authored_shares_are_not_part_of_the_schema(field: str) -> None:
    """Shares are elicited, so a cell cannot author one."""
    payload = _composition_payload()
    payload["cells"][0][field] = 1.0

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EstimandCompositionDef.model_validate(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["cells"][1].update(
                {"cell_id": "third_party_injury"}
            ),
            "cell_id values repeat",
        ),
        (
            lambda payload: payload["cells"][1].update(
                {
                    "loss_form": "Third-party injury valuation.",
                    "subject": "third_party",
                }
            ),
            "cells must be unique",
        ),
        (lambda payload: payload.update({"rationale": "   "}), "must not be blank"),
        (lambda payload: payload.update({"cells": payload["cells"][:1]}), "at least 2"),
    ],
)
def test_invalid_estimand_composition_is_rejected(
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    payload = _composition_payload()
    mutation(payload)

    with pytest.raises(ValidationError, match=message):
        EstimandCompositionDef.model_validate(payload)


def test_repo_composition_rosters_are_explicit() -> None:
    scenarios = {scenario.id: scenario for scenario in load_scenarios(_SCENARIOS_DIR)}
    # Every whole-incident roster has several loss forms, so every one is elicited.
    assert all(len(row.estimand_composition.cells) > 1 for row in scenarios.values())
    service = scenarios["service_outage_degradation"].estimand_composition
    assert service.roster is None
    assert [cell.cell_id for cell in service.cells] == [
        "customer_interruption_and_response",
        "customer_performance_reliance_transfer",
        "customer_deployer_interruption_and_response",
        "customer_deployer_performance_reliance_transfer",
        "provider_response_and_restoration",
        "provider_civil_fine",
    ]
    enforcement = scenarios["enforcement_matter"].estimand_composition
    assert {cell.cell_id for cell in enforcement.cells} == {
        "provider_civil_fine",
        "provider_resource_cost",
        "provider_noncivil_penalty",
        "provider_ordinary_transfer",
        "customer_migration_and_switching",
        "deployer_migration_and_switching",
    }
    # Customers forced off the provider by a ban bear their own migration
    # resources; the provider does not.
    migration = {
        cell.cell_id: (cell.nature, cell.subject)
        for cell in enforcement.cells
        if cell.cell_id.endswith("_migration_and_switching")
    }
    assert migration == {
        "customer_migration_and_switching": ("resource_cost", "third_party"),
        "deployer_migration_and_switching": ("resource_cost", "other"),
    }


def test_an_incidents_civil_fine_is_a_tracked_provider_transfer_cell() -> None:
    consequential = [
        (scenario.id, cell)
        for scenario in load_scenarios(_SCENARIOS_DIR)
        for cell in scenario.estimand_composition.tracked_cells
        if cell.attachment_basis == "incident_consequence"
    ]
    assert consequential
    for scenario_id, cell in consequential:
        assert cell.cell_id == "provider_civil_fine", scenario_id
        assert (cell.nature, cell.subject, cell.loss_type) == (
            "transfer",
            "provider",
            "civil_fines",
        ), scenario_id


def test_a_tracked_cell_must_appear_in_the_severity_scope() -> None:
    payload = _scenario_payload("enforcement_matter")
    payload["estimand_composition"]["cells"] = [
        cell
        for cell in payload["estimand_composition"]["cells"]
        if cell["cell_id"] != "provider_noncivil_penalty"
    ]
    with pytest.raises(ValidationError, match="not tracked composition cells"):
        ScenarioDefinition.model_validate(payload)
