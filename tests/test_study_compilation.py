import math
from copy import deepcopy

import pytest

from catmodel.contracts.incident import EntityPath, LocalResponse
from catmodel.simulation.incident import (
    _integrate_local_response,
    compile_incident_state,
)
from test_support.run_setup import setup_from_payload
from test_support.study import study_payload


def test_nonlinear_response_precedes_average_path() -> None:
    payload = study_payload()
    path = payload["joint_states"][0]["entity_paths"][0]
    path["knots"] = [
        {"time": 0, "value": 0, "model_revision": "r1"},
        {"time": 0.5, "value": 1, "model_revision": "r1"},
        {"time": 1, "value": 1, "model_revision": "r1"},
    ]
    study = setup_from_payload(payload)
    state = compile_incident_state(study, "state")
    assert state.incidents[0].annual_rate == pytest.approx(2.5)
    path["knots"] = [
        {"time": 0, "value": 0.5, "model_revision": "r1"},
        {"time": 1, "value": 0.5, "model_revision": "r1"},
    ]
    averaged = compile_incident_state(setup_from_payload(payload), "state")
    assert averaged.incidents[0].annual_rate == pytest.approx(2)


def test_exact_linear_path_log_response_integral() -> None:
    payload = study_payload()
    payload["joint_states"][0]["entity_paths"][0]["interpolation"] = "linear"
    payload["joint_states"][0]["entity_paths"][0]["knots"][-1]["value"] = 1
    assert compile_incident_state(setup_from_payload(payload), "state").incidents[
        0
    ].annual_rate == pytest.approx(3 / math.log(4))


def test_changing_marks_are_intensity_weighted_not_duration_averaged() -> None:
    payload = study_payload()
    state = payload["joint_states"][0]
    state["entity_paths"][0]["knots"] = [
        {"time": 0, "value": 0, "model_revision": "r1"},
        {"time": 0.5, "value": 1, "model_revision": "r1"},
        {"time": 1, "value": 1, "model_revision": "r1"},
    ]
    first = state["laws"][0]["segments"][0]
    first.update(end=0.5, reference_exposure_share=0.5)
    second = deepcopy(first)
    second.update(start=0.5, end=1)
    second["kernel"]["member_outcomes"][0]["components"][0]["amount"] = 100
    state["laws"][0]["segments"].append(second)
    incident = compile_incident_state(setup_from_payload(payload), "state").incidents[0]
    assert incident.annual_rate == pytest.approx(2.5)
    assert [segment.weight for segment in incident.segments] == pytest.approx(
        [0.2, 0.8]
    )
    marked_mean = 0.0
    for segment in incident.segments:
        amount = segment.kernel.member_outcomes[0].components[0].amount
        assert amount is not None
        marked_mean += segment.integrated_rate * amount
    assert marked_mean == pytest.approx(205)
    assert marked_mean != pytest.approx(2.5 * (10 + 100) / 2)


def test_partial_exposure_is_never_normalized() -> None:
    payload = study_payload()
    payload["joint_states"][0]["laws"][0]["segments"][0]["reference_exposure_share"] = (
        0.2
    )
    assert compile_incident_state(setup_from_payload(payload), "state").incidents[
        0
    ].annual_rate == pytest.approx(0.2)


def test_sparse_correlated_epistemic_paths_preserve_eal() -> None:
    payload = study_payload()
    first = payload["joint_states"][0]
    first.update(state_id="A_high", epistemic_group_id="A", weight=0.5)
    second = deepcopy(first)
    second.update(state_id="B_low", epistemic_group_id="B")
    payload["joint_states"].append(second)
    study = setup_from_payload(payload)
    # Each authored joint cell has loss 10. No missing off-diagonal cell is inferred.
    means = [
        compile_incident_state(study, row.state_id).incidents[0].annual_rate * 10
        for row in study.joint_states
    ]
    assert (
        math.fsum((row.weight * mean for row, mean in zip(study.joint_states, means)))
        == 10
    )


def test_metadata_and_origin_do_not_dispatch_mathematics() -> None:
    payload = study_payload()
    first = setup_from_payload(payload)
    changed = deepcopy(payload)
    changed["title"] = "Editorial correction"
    changed["entities"][0].update(status="actual", display_name="Observed label")
    second = setup_from_payload(changed)
    assert first.scientific_sha256 != second.scientific_sha256
    assert (
        compile_incident_state(first, "state").conditional_key
        == compile_incident_state(second, "state").conditional_key
    )


def test_weight_only_reduction_preserves_conditional_kernel() -> None:
    payload = study_payload()
    first = payload["joint_states"][0]
    first.update(weight=0.5)
    second = deepcopy(first)
    second.update(state_id="other", weight=0.5)
    payload["joint_states"].append(second)
    original = setup_from_payload(payload)
    payload["joint_states"][0]["weight"] = 0.25
    payload["joint_states"][1]["weight"] = 0.75
    reweighted = setup_from_payload(payload)
    assert [
        compile_incident_state(original, state.state_id).conditional_key
        for state in original.joint_states
    ] == [
        compile_incident_state(reweighted, state.state_id).conditional_key
        for state in reweighted.joint_states
    ]


def test_negative_local_intensity_is_not_hidden_by_positive_integral() -> None:
    payload = study_payload()
    path = payload["joint_states"][0]["entity_paths"][0]
    path["interpolation"] = "linear"
    path["knots"][1]["value"] = 2.5
    response = payload["joint_states"][0]["laws"][0]["response"]
    # The capped last-slope continuation reaches multiplier -1 at x = 2, while
    # the path's integral stays positive (0.6).
    response.update(
        interpolation="linear",
        points=[{"value": 0, "multiplier": 3}, {"value": 1, "multiplier": 1}],
    )
    with pytest.raises(ValueError, match="negative"):
        _integrate_local_response(
            EntityPath.model_validate(path),
            LocalResponse.model_validate(response),
            0,
            1,
        )


def test_numerical_refinement_cannot_change_the_law() -> None:
    payload = study_payload()
    payload["alternatives"] = [
        {
            "alternative_id": "refinement",
            "purpose": "numerical_refinement",
            "description": "Numerical settings may change, scientific laws may not.",
            "overrides": [
                {"state_id": "state", "law": payload["joint_states"][0]["laws"][0]}
            ],
        }
    ]
    with pytest.raises(ValueError, match="cannot mutate"):
        compile_incident_state(setup_from_payload(payload), "state", "refinement")


def test_report_split_does_not_change_mathematical_identity() -> None:
    payload = study_payload()
    before = compile_incident_state(setup_from_payload(payload), "state")
    payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"]["member_outcomes"][
        0
    ]["components"][0]["report_labels"] = ["one display", "another display"]
    after = compile_incident_state(setup_from_payload(payload), "state")
    assert before.law_sha256 == after.law_sha256
    assert before.conditional_key == after.conditional_key


def test_population_and_component_ownership_change_scientific_identity() -> None:
    payload = study_payload()
    before = compile_incident_state(setup_from_payload(payload), "state")
    changed = deepcopy(payload)
    changed["populations"][0]["causal_qualification"] = (
        "A different actual target population"
    )
    assert (
        compile_incident_state(setup_from_payload(changed), "state").conditional_key
        != before.conditional_key
    )
    changed = deepcopy(payload)
    changed["joint_states"][0]["laws"][0]["segments"][0]["kernel"]["member_outcomes"][
        0
    ]["components"][0]["bearer_id"] = "other_victim"
    assert (
        compile_incident_state(setup_from_payload(changed), "state").law_sha256
        != before.law_sha256
    )


def test_evidence_origin_changes_provenance_without_changing_law() -> None:
    payload = study_payload()
    evidence = {
        "evidence_id": "input",
        "origin": "stipulation",
        "original_sha256": "a" * 64,
        "selected_sha256": "a" * 64,
        "target": "Conditional initiating count",
        "conditioning": "Given the authored state",
        "units": "campaigns per study year",
        "transport": "None, directly stipulated for this exact target.",
        "review": "Unreviewed fixture",
        "reliance": "Internal numerical test only",
        "rights": "Openly authored test",
    }
    payload["evidence"] = [evidence]
    payload["joint_states"][0]["laws"][0]["evidence_ids"] = ["input"]
    first = compile_incident_state(setup_from_payload(payload), "state")
    evidence["origin"] = "observation"
    evidence["original_sha256"] = "b" * 64
    second = compile_incident_state(setup_from_payload(payload), "state")
    assert first.evidence_sha256 != second.evidence_sha256
    assert first.law_sha256 == second.law_sha256
    assert first.conditional_key == second.conditional_key


def test_pooled_accounting_standing_cannot_borrow_the_first_population_label():
    payload = study_payload()
    population = deepcopy(payload["populations"][0])
    population.update(
        scenario_id="provider_only", accounting_standing="Provider-only standing"
    )
    payload["populations"].append(population)
    law = deepcopy(payload["joint_states"][0]["laws"][0])
    law.update(origin_id="provider-origin", scenario_id="provider_only")
    law["segments"][0]["kernel"]["member_outcomes"][0]["components"][0][
        "scenario_ids"
    ] = ["provider_only"]
    payload["joint_states"][0]["laws"].append(law)
    compiled = compile_incident_state(setup_from_payload(payload), "state")
    assert "accounting standing" in compiled.unavailable_views["pooled"]
    assert "pooled" not in compiled.view_bases
    assert (
        compiled.view_bases["scenario::provider_only"]["accounting_standing"]
        == "Provider-only standing"
    )


def test_numerical_refinement_cannot_mutate_a_scientific_law():
    payload = study_payload()
    state = payload["joint_states"][0]
    payload["alternatives"] = [
        {
            "alternative_id": "refinement",
            "purpose": "numerical_refinement",
            "description": "A refinement may change budgets, never the law.",
            "overrides": [{"state_id": state["state_id"], "law": state["laws"][0]}],
        }
    ]
    study = setup_from_payload(payload)
    with pytest.raises(ValueError, match="numerical refinement cannot mutate"):
        compile_incident_state(study, state["state_id"], "refinement")


def test_compiled_law_carries_frozen_contracts_and_declared_output_roles():
    payload = study_payload()
    payload["contracts"] = [
        {
            "contract_id": "policy",
            "contract_year": "2027",
            "entity_id": "model",
            "occurrence_limit": 50,
        }
    ]
    study = setup_from_payload(payload)
    compiled = compile_incident_state(study, "state")
    assert compiled.contracts == study.contracts
    assert compiled.contracts[0].occurrence_limit == 50
    assert compiled.declared_view_ids["bearer"] == frozenset({"victim"})
    assert compiled.declared_view_ids["risk_unit"] == frozenset({"origin"})
    with pytest.raises(TypeError):
        compiled.declared_view_ids["bearer"] = frozenset({"unwritten"})  # ty: ignore[invalid-assignment]
    payload["contracts"][0]["occurrence_limit"] = 0
    changed = compile_incident_state(setup_from_payload(payload), "state")
    assert compiled.contracts[0].occurrence_limit == 50
    assert changed.contracts[0].occurrence_limit == 0
    assert compiled.conditional_key != changed.conditional_key


def test_pooled_disclosure_retains_every_incompatible_basis():
    payload = study_payload()
    population = deepcopy(payload["populations"][0])
    population.update(
        scenario_id="other",
        currency="EUR",
        monetary_basis="2028 EUR",
        accounting_standing="Other stakeholder standing",
    )
    payload["populations"].append(population)
    law = deepcopy(payload["joint_states"][0]["laws"][0])
    law.update(origin_id="other-origin", scenario_id="other")
    law["segments"][0]["kernel"]["member_outcomes"][0]["components"][0][
        "scenario_ids"
    ] = ["other"]
    payload["joint_states"][0]["laws"].append(law)
    compiled = compile_incident_state(setup_from_payload(payload), "state")
    reason = compiled.unavailable_views["pooled"]
    for phrase in (
        "currency and monetary basis",
        "same accounting standing",
    ):
        assert phrase in reason
    assert "pooled" not in compiled.view_bases
    assert compiled.view_bases["scenario::scenario"]["currency"] == "USD"
    assert compiled.view_bases["scenario::other"] == {
        "currency": "EUR",
        "monetary_basis": "2028 EUR",
        "accounting_standing": "Other stakeholder standing",
        "severity_unit": population["severity_unit"],
    }
