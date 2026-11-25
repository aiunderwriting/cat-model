"""Scenario authoring describes populations and native structure directly."""

import pytest

from catmodel.resources import scenarios_path
from catmodel.scenarios.catalog import (
    build_scenario_catalog_from_snapshot,
    load_scenario_library_snapshot,
)
from catmodel.scenarios.loader import (
    ScenarioDefinition,
    load_scenarios,
)


def test_a_scenario_refuses_a_field_its_schema_does_not_define():
    payload = load_scenarios(scenarios_path())[0].model_dump(mode="json")
    payload["unknown_field"] = []
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ScenarioDefinition.model_validate(payload)


def test_catalog_reports_each_scenarios_governed_severity():
    catalog = build_scenario_catalog_from_snapshot(
        load_scenario_library_snapshot(scenarios_path())
    )
    assert catalog["meta"]["schema_version"] == 1
    entries = {entry["id"]: entry for entry in catalog["scenarios"]}
    assert entries["private_claims_privacy"]["severity_governance"] == {
        "tail_class": "liability_enforcement",
        "gpd_shape": 0.5,
        "cap_posture": "uncapped",
        "severity_cap": None,
        "uncapped_endpoint_stress": None,
    }
    assert entries["output_reliance_health"]["severity_governance"] == {
        "tail_class": "cyber_operational",
        "gpd_shape": 0.7,
        "cap_posture": "uncapped",
        "severity_cap": None,
        "uncapped_endpoint_stress": None,
    }
