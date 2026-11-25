import pytest

from catmodel.elicitation.contextual_risk import contextual_node_from_scenario
from catmodel.resources import scenarios_path
from catmodel.scenarios.catalog import (
    build_scenario_catalog_from_snapshot,
    load_scenario_library_snapshot,
)
from catmodel.scenarios.facets import (
    FACET_ROLLUP_AXES,
    scenario_facet_violations,
)
from catmodel.scenarios.loader import ScenarioDefinition, load_scenarios

SCENARIOS = {scenario.id: scenario for scenario in load_scenarios(scenarios_path())}


def _facet_copy(scenario_id: str, **facet_updates):
    scenario = SCENARIOS[scenario_id]
    return scenario.model_copy(
        update={"facets": scenario.facets.model_copy(update=facet_updates)}
    )


class TestFacetCoherenceValidators:
    def test_scenario_load_rejects_incoherent_facets(self) -> None:
        payload = SCENARIOS["enforcement_matter"].model_dump()
        payload["facets"]["party_basis"] = "third_party"
        with pytest.raises(ValueError, match="facet coherence"):
            ScenarioDefinition.model_validate(payload)

    def test_lifecycle_stage_must_be_admissible_for_domain(self) -> None:
        modified = _facet_copy(
            "compromise_organizations", lifecycle_stage="training_evaluation"
        )
        violations = scenario_facet_violations(modified)
        assert any("lifecycle_stage" in violation for violation in violations)

    def test_self_exfiltrated_mode_restricted_to_ai_rd(self) -> None:
        modified = _facet_copy(
            "compromise_organizations", deployment_control_mode="self_exfiltrated"
        )
        violations = scenario_facet_violations(modified)
        assert any("self_exfiltrated" in violation for violation in violations)

    def test_domain_exclusive_mechanisms_rejected_elsewhere(self) -> None:
        modified = _facet_copy(
            "compromise_organizations",
            peril_mechanism=["technical_intrusion", "enforcement_process"],
        )
        violations = scenario_facet_violations(modified)
        assert any("enforcement_process" in violation for violation in violations)

    def test_regulatory_cells_must_be_first_party(self) -> None:
        modified = _facet_copy("enforcement_matter", party_basis="third_party")
        violations = scenario_facet_violations(modified)
        assert any("first_party" in violation for violation in violations)

    def test_derived_facets_cannot_be_authored(self) -> None:
        scenario = SCENARIOS["compromise_organizations"]
        payload = scenario.model_dump(mode="json", exclude_none=True)
        payload["facets"]["causal_origin"] = "autonomous_model_action"
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            type(scenario).model_validate(payload)


class TestFacetsStayOutOfQuestions:
    def test_facet_reclassification_changes_no_question(self) -> None:
        scenario = SCENARIOS["compromise_organizations"]
        reclassified = _facet_copy(
            "compromise_organizations",
            harm_types=["economic_loss", "financial_theft"],
            provisional=True,
        )
        for access in ("public", "internal"):
            assert contextual_node_from_scenario(
                reclassified, "lab", access
            ) == contextual_node_from_scenario(scenario, "lab", access)


class TestCatalogProjections:
    def test_catalog_projects_every_axis_and_scenario(self) -> None:
        catalog = build_scenario_catalog_from_snapshot(
            load_scenario_library_snapshot(scenarios_path())
        )
        projections = catalog["facet_projections"]
        assert set(projections) == set(FACET_ROLLUP_AXES)
        all_ids = {scenario_id for scenario_id in SCENARIOS}
        for axis, additivity in FACET_ROLLUP_AXES.items():
            assert projections[axis]["additivity"] == additivity
            projected: list[str] = []
            for scenario_ids in projections[axis]["values"].values():
                projected.extend(scenario_ids)
            assert set(projected) == all_ids
            if additivity == "partition":
                assert len(projected) == len(all_ids)
        for entry in catalog["scenarios"]:
            assert entry["facets"]["causal_origin"]
            assert entry["occurrence_unit_id"]
