"""Authoring checks for resource loss, redistribution and non-rival weights."""

import math
from pathlib import Path

import pytest

from catmodel.economic_loss import welfare_weight
from catmodel.scenarios.loader import load_scenarios, rendered_severity_occurrence_scope


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios(Path(__file__).resolve().parents[1] / "scenarios")


def test_resource_cells_cannot_call_displaced_profit_resource_destruction(scenarios):
    for scenario in scenarios:
        for cell in scenario.estimand_composition.cells:
            if cell.nature != "resource_cost":
                continue
            text = " ".join(cell.loss_form.split())
            assert "lost operating contribution" not in text, (
                scenario.id,
                cell.cell_id,
            )
            assert (
                "Exclude displaced profit" in text
                or "Do not automatically charge historic training cost" in text
            ), (scenario.id, cell.cell_id)


def test_non_rival_weights_have_separate_resource_and_appropriation_cells(scenarios):
    for scenario_id in ("model_asset_compromise", "developer_control_loss"):
        _assert_non_rival_weight_cells(scenarios, scenario_id)


def _assert_non_rival_weight_cells(scenarios, scenario_id):
    scenario = next(s for s in scenarios if s.id == scenario_id)
    cells = {c.cell_id: c for c in scenario.estimand_composition.cells}
    resource = cells["provider_resource_cost"]
    appropriation = cells["provider_distinct_transfer_appropriated"]
    assert resource.nature == "resource_cost"
    assert appropriation.nature == "appropriated_transfer"
    assert "does not itself destroy the original model" in " ".join(
        resource.loss_form.split()
    )
    assert "do not substitute its historic training bill" in " ".join(
        appropriation.loss_form.split()
    )
    assert "stolen_weights_and_provider_response" not in cells


def test_named_noncivil_penalty_cell_does_not_expand_into_ip_or_profit_losses(
    scenarios,
):
    scenario = next(s for s in scenarios if s.id == "enforcement_matter")
    cell = next(
        c
        for c in scenario.estimand_composition.cells
        if c.cell_id == "provider_noncivil_penalty"
    )
    assert cell.nature == "transfer"
    assert "IP rents" not in cell.loss_form
    assert "displaced profit" not in cell.loss_form


@pytest.mark.parametrize(
    "scenario_id,amounts,expected_gross,expected_welfare,expected_resources",
    [
        (
            "government_designation",
            {"provider_resource_cost": 1_000_000, "provider_transfer": 10_000_000},
            11_000_000,
            1_000_000,
            1_000_000,
        ),
        (
            "fraud_organizations",
            {
                "external_resource_loss": 4_000_000,
                "external_financial_loss_appropriated": 8_000_000,
                "external_financial_loss": 3_000_000,
                "provider_civil_fine": 750_000,
            },
            15_750_000,  # The fine is included once within the whole incident total.
            12_000_000,
            4_000_000,
        ),
    ],
)
def test_business_loss_compositions_preserve_gross_and_project_distinct_natures(
    scenarios,
    scenario_id,
    amounts,
    expected_gross,
    expected_welfare,
    expected_resources,
):
    scenario = next(s for s in scenarios if s.id == scenario_id)
    cells = scenario.estimand_composition.cells
    amounts = {**dict.fromkeys((c.cell_id for c in cells), 0), **amounts}
    assert set(amounts) == {c.cell_id for c in cells}
    # An authored history with illustrative amounts checks the nature and bearer
    # the contextual questions attach to each cell.
    natures = {c.cell_id: c.nature for c in cells}
    bearers = {c.cell_id: c.subject for c in cells}
    assert math.fsum(amounts.values()) == pytest.approx(expected_gross)
    welfare = math.fsum(
        amount * welfare_weight(natures[cell_id]) for cell_id, amount in amounts.items()
    )
    assert welfare == pytest.approx(expected_welfare)
    resources = math.fsum(
        amount
        for cell_id, amount in amounts.items()
        if natures[cell_id] == "resource_cost"
    )
    assert resources == pytest.approx(expected_resources)
    if scenario_id == "government_designation":
        assert {bearers[cell_id] for cell_id, amount in amounts.items() if amount} == {
            "provider"
        }


def test_campaign_appropriation_is_distinct_from_ordinary_transfers(scenarios):
    scenario = next(s for s in scenarios if s.id == "fraud_organizations")
    cells = {c.cell_id: c for c in scenario.estimand_composition.cells}
    ransom = cells["external_financial_loss_appropriated"]
    assert ransom.nature == "appropriated_transfer"
    assert "theft, fraud" in ransom.loss_form
    assert "ordinary" in ransom.loss_form
    assert cells["external_financial_loss"].nature == "transfer"


def test_enforcement_rendered_scope_matches_resource_cell(scenarios):
    scenario = next(s for s in scenarios if s.id == "enforcement_matter")
    cell = next(
        c
        for c in scenario.estimand_composition.tracked_cells
        if c.cell_id == "provider_resource_cost"
    )
    scope = " ".join(rendered_severity_occurrence_scope(scenario).split())
    assert cell.loss_form in scope
    assert "lost operating contribution" not in scope
    assert "Exclude displaced profit" in scope
