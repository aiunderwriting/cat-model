"""A disjoint terminal union remains one forecast with one placement projection."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from catmodel.scenarios.facets import scenario_facet_projection
from catmodel.scenarios.hierarchy_contract import (
    load_hierarchy_contract,
    scenario_placement_coordinates,
    validate_hierarchy_contract,
)
from catmodel.scenarios.loader import ScenarioDefinition, load_scenarios

ROOT = Path(__file__).resolve().parents[1]
UNION_NODES = ("n_reliance_health", "n_reliance_legal")


@dataclass(frozen=True)
class Placement:
    id: str
    node_ids: tuple[str, ...]


@pytest.fixture(scope="module")
def scenarios() -> dict[str, ScenarioDefinition]:
    return {scenario.id: scenario for scenario in load_scenarios(ROOT / "scenarios")}


def union_scenario(scenarios: dict[str, ScenarioDefinition]) -> ScenarioDefinition:
    payload = scenarios["output_reliance_legal"].model_dump()
    payload["node_ids"] = UNION_NODES
    return ScenarioDefinition.model_validate(payload)


@pytest.mark.parametrize(
    "node_ids", [(), ("n_reliance_legal", "n_reliance_legal"), ("bad",)]
)
def test_scenario_rejects_invalid_terminal_vectors(scenarios, node_ids) -> None:
    payload = scenarios["output_reliance_legal"].model_dump()
    payload["node_ids"] = node_ids
    with pytest.raises(ValidationError):
        ScenarioDefinition.model_validate(payload)


def test_union_keeps_common_domain_and_facets(scenarios) -> None:
    original = scenarios["output_reliance_legal"]
    union = union_scenario(scenarios)
    assert union.node_ids == UNION_NODES
    assert union.risk_domain == original.risk_domain == "eo"
    assert scenario_facet_projection(union) == scenario_facet_projection(original)


@pytest.mark.parametrize(
    "node_ids",
    [
        ("n_reliance_legal", "n_fraud_organizations"),
        ("n_fraud_organizations", "n_misuse_asset"),
        ("n_reliance_legal", "n_service_outage"),
    ],
)
def test_union_rejects_different_placement_coordinates(node_ids) -> None:
    with pytest.raises(ValueError, match="share origin, AI role and reporting"):
        scenario_placement_coordinates(load_hierarchy_contract(), node_ids)


def test_typed_scenario_rejects_cross_domain_union(scenarios) -> None:
    payload = scenarios["output_reliance_legal"].model_dump()
    payload["node_ids"] = ("n_reliance_legal", "n_fraud_organizations")
    with pytest.raises(ValidationError, match="share origin, AI role"):
        ScenarioDefinition.model_validate(payload)


def test_union_placement_passes_live_hierarchy_validation(scenarios) -> None:
    contract = load_hierarchy_contract()
    union = union_scenario(scenarios)
    selected = [
        scenario
        for scenario in scenarios.values()
        if scenario.id not in {"output_reliance_legal", "output_reliance_health"}
    ] + [union]
    assert (
        validate_hierarchy_contract(contract=contract, scenarios=selected).errors == ()
    )


def test_overlapping_forecasts_reject_shared_terminal() -> None:
    result = validate_hierarchy_contract(
        contract=load_hierarchy_contract(),
        scenarios=[
            Placement("union", UNION_NODES),
            Placement("other", ("n_reliance_legal",)),
        ],
    )
    assert "scenario node assignments must be unique" in result.errors
