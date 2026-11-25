"""Service-customer loss forms keep provider-own loss outside their roster."""

from pathlib import Path

import pytest

from catmodel.scenarios.loader import load_scenarios

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scenarios():
    return {s.id: s for s in load_scenarios(ROOT / "scenarios")}


def test_service_scope_adds_provider_restoration_once(scenarios):
    scenario = scenarios["service_outage_degradation"]
    cells = {cell.cell_id: cell for cell in scenario.estimand_composition.cells}
    assert {cell.subject for cell in cells.values()} == {
        "other",
        "third_party",
        "provider",
    }
    assert [
        cell_id for cell_id, cell in cells.items() if cell.subject == "provider"
    ] == ["provider_response_and_restoration", "provider_civil_fine"]
    # The whole incident owns the provider's legal defense with its restoration.
    assert "legal defense" in cells["provider_response_and_restoration"].loss_form
    for cell_id, cell in cells.items():
        if cell_id == "provider_civil_fine":
            continue
        assert "compensation for already-recorded harm" in cell.loss_form or (
            "compensate customer harm already recorded" in cell.loss_form
        )
        if cell.subject == "other":
            assert "provider-own" in cell.loss_form
            assert "customer-deployer's own" in cell.loss_form
        elif cell.subject == "third_party":
            assert "provider-own" in cell.loss_form
            assert (
                "neither the frontier provider nor the non-provider system deployer"
                in cell.loss_form
            )
