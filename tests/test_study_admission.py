"""Independent admission counterexamples for absent populations and qualifications."""

from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from catmodel.contracts.incident import ExecutionPlan
from catmodel.contracts.native_severity import native_severity_from_sampler
from catmodel.probability.distributions import Distribution
from catmodel.results.execution import simulate_incident_state
from catmodel.simulation import incident
from catmodel.simulation.incident import check_incident_state, compile_incident_state
from test_support.results import available_metric, present, reduced_arm
from test_support.run_setup import setup_from_json, setup_from_payload, with_execution
from test_support.study import study_payload


def _second_population(payload):
    other = deepcopy(payload["populations"][0])
    other["scenario_id"] = "unmodeled"
    payload["populations"].append(other)


def test_unstated_missing_population_cannot_become_zero() -> None:
    payload = study_payload()
    _second_population(payload)
    with pytest.raises(ValidationError, match="every declared population"):
        setup_from_payload(payload)


def test_explicit_structural_absence_is_distinct_from_unavailable() -> None:
    payload = study_payload()
    _second_population(payload)
    state = payload["joint_states"][0]
    state["absent_populations"] = [
        {
            "scenario_id": "unmodeled",
            "disposition": "structural_zero",
            "reason": "The selected model is not deployed to this population in this state.",
        }
    ]
    study = setup_from_payload(payload)
    plan = ExecutionPlan(
        plan_id="absence",
        required_arms=("reference",),
        simulations=100,
        seed=42,
        requested_views=(
            "gross",
            "scenario::scenario::gross",
            "scenario::unmodeled::gross",
        ),
    )
    draw, reasons, _ = simulate_incident_state(
        plan, compile_incident_state(study, "state"), response_sha256="a" * 64
    )
    assert not reasons
    np.testing.assert_array_equal(draw.annual_losses["scenario::unmodeled::gross"], 0)
    state["absent_populations"][0].update(
        disposition="unavailable",
        reason="No incident law has been authored for this population.",
    )
    study = setup_from_payload(payload)
    draw, reasons, _ = simulate_incident_state(
        plan, compile_incident_state(study, "state"), response_sha256="a" * 64
    )
    assert set(draw.annual_losses) == {"scenario::scenario::gross"}
    assert set(reasons) == {"gross", "scenario::unmodeled::gross"}


def test_all_unavailable_state_needs_no_fabricated_zero_law() -> None:
    payload = study_payload()
    state = payload["joint_states"][0]
    state.update(
        laws=[],
        entity_paths=[],
        absent_populations=[
            {
                "scenario_id": "scenario",
                "disposition": "unavailable",
                "reason": "New incident inputs are absent; the annual aggregate is retained as evidence only.",
            }
        ],
    )
    compiled = compile_incident_state(setup_from_payload(payload), "state")
    assert compiled.incidents == ()
    assert "scenario::scenario" in compiled.unavailable_views
    assert "unquantified populations" in compiled.unavailable_views["pooled"]


def test_pooled_admission_retains_basis_and_missing_population_reasons() -> None:
    payload = study_payload()
    _second_population(payload)
    payload["populations"][1]["currency"] = "EUR"
    payload["joint_states"][0]["absent_populations"] = [
        {
            "scenario_id": "unmodeled",
            "disposition": "unavailable",
            "reason": "The additional physical population has no incident inputs.",
        }
    ]
    compiled = compile_incident_state(setup_from_payload(payload), "state")
    assert "currency and monetary basis" in compiled.unavailable_views["pooled"]
    assert "unquantified populations" in compiled.unavailable_views["pooled"]
    assert "scenario::unmodeled" in compiled.unavailable_views
    assert "scenario::scenario" not in compiled.unavailable_views


def test_all_structural_zero_state_is_an_explicit_zero_law() -> None:
    payload = study_payload()
    state = payload["joint_states"][0]
    state.update(
        laws=[],
        entity_paths=[],
        absent_populations=[
            {
                "scenario_id": "scenario",
                "disposition": "structural_zero",
                "reason": "No selected entity is deployed in the population.",
            }
        ],
    )
    study = setup_from_payload(payload)
    plan = ExecutionPlan(
        plan_id="zero",
        required_arms=("reference",),
        simulations=20,
        seed=4,
        requested_views=("gross", "scenario::scenario::gross"),
    )
    draw, reasons, _ = simulate_incident_state(
        plan, compile_incident_state(study, "state"), response_sha256="a" * 64
    )
    assert not reasons
    assert all(np.all(values == 0) for values in draw.annual_losses.values())


def _threshold_payload(comparator="gt", threshold=10):
    payload = study_payload()
    payload["populations"][0].update(
        threshold=threshold, threshold_comparator=comparator
    )
    return payload


@pytest.mark.parametrize("comparator,accepted", [("gt", False), ("ge", True)])
def test_threshold_equality_is_part_of_the_terminal_law(comparator, accepted) -> None:
    study = setup_from_payload(_threshold_payload(comparator))
    if accepted:
        assert compile_incident_state(study, "state").incidents[0].annual_rate == 1
    else:
        with pytest.raises(ValueError, match="terminal-total support contradicts"):
            compile_incident_state(study, "state")


def test_threshold_is_member_total_and_excludes_campaign_response_cost() -> None:
    payload = _threshold_payload()
    kernel = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"]
    prototype = kernel["member_outcomes"][0]["components"][0]
    prototype["amount"] = 6
    other = {**deepcopy(prototype), "component_id": "second", "amount": 4}
    kernel["member_outcomes"][0]["components"].append(other)
    kernel["campaign_components"] = [
        {**deepcopy(prototype), "component_id": "response", "amount": 100}
    ]
    with pytest.raises(ValueError, match="lower support is 10 with an atom"):
        compile_incident_state(setup_from_payload(payload), "state")
    other["amount"] = 5
    assert (
        compile_incident_state(setup_from_payload(payload), "state")
        .incidents[0]
        .annual_rate
        == 1
    )


def test_prevention_gate_may_produce_zero_without_requalifying_survivors() -> None:
    payload = _threshold_payload()
    payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
        "unit_probability"
    ] = 0
    assert (
        compile_incident_state(setup_from_payload(payload), "state")
        .incidents[0]
        .annual_rate
        == 1
    )


def _native_component(payload, *, atom: bool):
    component = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
        "member_outcomes"
    ][0]["components"][0]
    component.pop("amount")
    knots = (10, 10, 10, 20, 30) if atom else (11, 12, 13, 14, 15)
    component.update(
        rank_group_id="severity",
        severity=native_severity_from_sampler(
            Distribution(
                *knots,
                severity_minimum=None if atom else 10,
                severity_minimum_strict=not atom,
            )
        ).model_dump(mode="json"),
    )
    return component


def test_atomic_native_floor_is_excluded_by_strict_threshold() -> None:
    payload = _threshold_payload()
    _native_component(payload, atom=True)
    with pytest.raises(ValueError, match="with an atom"):
        compile_incident_state(setup_from_payload(payload), "state")
    payload["populations"][0]["threshold_comparator"] = "ge"
    compile_incident_state(setup_from_payload(payload), "state")


def test_continuous_native_floor_satisfies_strict_threshold_without_clipping() -> None:
    payload = _threshold_payload()
    _native_component(payload, atom=False)
    compiled = compile_incident_state(setup_from_payload(payload), "state")
    assert compiled.incidents[0].annual_rate == 1
    assert compiled.incidents[0].segments[0].kernel.unit_probability == 1


def test_transform_that_crosses_qualification_is_rejected_without_rate_change() -> None:
    payload = _threshold_payload()
    component = _native_component(payload, atom=False)
    component["multiplier"] = 0.5
    with pytest.raises(ValueError, match="contradicts qualification"):
        compile_incident_state(setup_from_payload(payload), "state")


def _refusal(function, setup, alternative: str = "reference") -> str | None:
    try:
        function(setup, "state", alternative)
    except ValueError as error:
        return str(error)
    return None


@pytest.mark.parametrize(
    "change,refused",
    [
        ("equality allowed", False),
        ("equality excluded", True),
        ("atom", True),
        ("continuous floor", False),
        ("crossing", True),
        ("unknown arm", True),
    ],
)
def test_checking_a_state_refuses_what_compiling_it_refuses(change, refused) -> None:
    payload = _threshold_payload("ge" if change == "equality allowed" else "gt")
    if change == "atom":
        _native_component(payload, atom=True)
    if change in ("continuous floor", "crossing", "unknown arm"):
        component = _native_component(payload, atom=False)
        if change == "crossing":
            component["multiplier"] = 0.5
    setup = setup_from_payload(payload)
    arm = "absent" if change == "unknown arm" else "reference"
    compiling = _refusal(compile_incident_state, setup, arm)
    assert (compiling is not None) == refused
    assert _refusal(check_incident_state, setup, arm) == compiling


def test_checking_a_state_works_out_no_identity(monkeypatch) -> None:
    """Admission asks only whether a law is refused, so the check stops before
    the digests and view bases a compiled state carries."""
    setup = setup_from_payload(_threshold_payload("ge"))

    def refuse(_payload: object) -> str:
        raise AssertionError("a check needs no digest")

    monkeypatch.setattr(incident, "canonical_json_sha256", refuse)
    check_incident_state(setup, "state")
    with pytest.raises(AssertionError, match="needs no digest"):
        compile_incident_state(setup, "state")


def test_source_annual_conditioning_is_not_relabelled_as_local_response() -> None:
    payload = study_payload()
    law = payload["joint_states"][0]["laws"][0]
    law["response"]["domain"] = "conditional_annual_kernel"
    with pytest.raises(ValidationError, match="cannot be relabelled"):
        setup_from_payload(payload)


@pytest.mark.parametrize(
    "changed_field", ["rate", "component", "ownership", "count_family"]
)
def test_named_count_control_must_hold_complete_conditional_law_fixed(
    changed_field,
) -> None:
    payload = study_payload()
    experiment = {
        "kind": "fixed_marginal_gaussian",
        "origin_groups": {"origin": "all"},
        "correlation": 0,
    }
    control_law = deepcopy(payload["joint_states"][0]["laws"][0])
    if changed_field == "rate":
        control_law["reference_annual_rate"] = 7
    elif changed_field == "component":
        control_law["segments"][0]["kernel"]["member_outcomes"][0]["components"][0][
            "amount"
        ] = 700
    elif changed_field == "ownership":
        control_law["segments"][0]["kernel"]["member_outcomes"][0]["components"][0][
            "bearer_id"
        ] = "another_bearer"
    else:
        control_law.update(count_family="negative_binomial", count_variance_to_mean=2)
    payload["alternatives"] = [
        {
            "alternative_id": "control",
            "purpose": "scientific_alternative",
            "description": "Incorrect unmatched control",
            "count_experiment": experiment,
            "matched_control_id": "control",
            "overrides": [{"state_id": "state", "law": control_law}],
        },
        {
            "alternative_id": "stress",
            "purpose": "scientific_alternative",
            "description": "The rate and complete marks must stay fixed",
            "count_experiment": {**experiment, "correlation": 0.7},
            "matched_control_id": "control",
        },
    ]
    with pytest.raises(ValidationError, match="hold every conditional scientific"):
        setup_from_payload(payload)


def test_explicit_zero_rate_law_covers_a_population_without_assumed_absence() -> None:
    payload = study_payload()
    _second_population(payload)
    zero = deepcopy(payload["joint_states"][0]["laws"][0])
    zero.update(
        origin_id="no_origins", scenario_id="unmodeled", reference_annual_rate=0
    )
    payload["joint_states"][0]["laws"].append(zero)
    study = setup_from_payload(payload)
    assert len(compile_incident_state(study, "state").incidents) == 2


def test_cap_atom_and_post_cap_scaling_are_part_of_threshold_support() -> None:
    from catmodel.contracts.native_severity import native_severity_from_sampler
    from catmodel.probability.distributions import Distribution

    payload = _threshold_payload()
    component = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
        "member_outcomes"
    ][0]["components"][0]
    component.pop("amount")
    component.update(
        rank_group_id="capped",
        severity=native_severity_from_sampler(
            Distribution(p5=20, p25=20, p50=20, p75=40, p95=60, severity_cap=10)
        ).model_dump(mode="json"),
    )
    with pytest.raises(ValueError, match="with an atom"):
        compile_incident_state(setup_from_payload(payload), "state")
    payload["populations"][0]["threshold_comparator"] = "ge"
    compile_incident_state(setup_from_payload(payload), "state")
    component["post_multiplier"] = 0.5
    with pytest.raises(ValueError, match="lower support is 5"):
        compile_incident_state(setup_from_payload(payload), "state")


def _report_payload(*, target_changes=None, binding=True):
    payload = study_payload()
    target = {
        **deepcopy(payload["populations"][0]),
        "scenario_id": "report",
        **(target_changes or {}),
    }
    payload["populations"].append(target)
    zero = deepcopy(payload["joint_states"][0]["laws"][0])
    zero.update(
        origin_id="report_origin", scenario_id="report", reference_annual_rate=0
    )
    zero["segments"][0]["kernel"]["member_outcomes"][0]["components"][0][
        "scenario_ids"
    ] = ["report"]
    payload["joint_states"][0]["laws"].append(zero)
    source = payload["joint_states"][0]["laws"][0]
    source["segments"][0]["kernel"]["member_outcomes"][0]["components"][0][
        "scenario_ids"
    ] = ["report"]
    if binding:
        payload["report_population_bindings"] = [
            {
                "source_scenario_id": "scenario",
                "target_scenario_id": "report",
                "rationale": "Explicitly stipulated overlapping reports of the same physical terminal members.",
            }
        ]
    return payload


@pytest.mark.parametrize(
    "attribute,changed",
    [
        ("currency", "EUR"),
        ("monetary_basis", "USD valued in a different year"),
        ("accounting_standing", "A different welfare standing"),
    ],
)
def test_report_labels_cannot_convert_a_scientific_basis(attribute, changed) -> None:
    payload = _report_payload(target_changes={attribute: changed})
    with pytest.raises(
        ValueError, match="a report label cannot convert a scientific basis"
    ):
        compile_incident_state(setup_from_payload(payload), "state")


@pytest.mark.parametrize("comparator,accepted", [("gt", False), ("ge", True)])
def test_cross_report_threshold_uses_exact_terminal_atom(comparator, accepted) -> None:
    payload = _report_payload(
        target_changes={"threshold": 10, "threshold_comparator": comparator}
    )
    study = setup_from_payload(payload)
    if accepted:
        compile_incident_state(study, "state")
    else:
        with pytest.raises(ValueError, match="terminal-total support contradicts"):
            compile_incident_state(study, "state")


def test_report_threshold_cannot_hide_behind_zero_rate_target_origin() -> None:
    payload = _report_payload(
        target_changes={"threshold": 100, "threshold_comparator": "gt"}
    )
    with pytest.raises(ValueError, match="lower support is 10 with an atom"):
        compile_incident_state(setup_from_payload(payload), "state")


def test_report_qualification_uses_full_member_not_tagged_component_sum() -> None:
    payload = _report_payload(
        target_changes={"threshold": 10, "threshold_comparator": "ge"}
    )
    member = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
        "member_outcomes"
    ][0]
    member["components"][0]["amount"] = 6
    member["components"].append(
        {
            **deepcopy(member["components"][0]),
            "component_id": "other_harm",
            "amount": 4,
            "scenario_ids": ["scenario"],
        }
    )
    compile_incident_state(setup_from_payload(payload), "state")


def test_campaign_cost_does_not_establish_cross_report_terminal_qualification() -> None:
    payload = _report_payload(
        target_changes={"threshold": 100, "threshold_comparator": "gt"}
    )
    kernel = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"]
    member = kernel["member_outcomes"][0]["components"][0]
    member["scenario_ids"] = ["scenario"]
    kernel["campaign_components"] = [
        {
            **deepcopy(member),
            "component_id": "response",
            "amount": 150,
            "scenario_ids": ["report"],
        }
    ]
    with pytest.raises(ValueError, match="lower support is 10 with an atom"):
        compile_incident_state(setup_from_payload(payload), "state")


def test_nontrivial_report_population_needs_an_explicit_scientific_binding() -> None:
    payload = _report_payload(
        target_changes={
            "causal_qualification": "A separately described causal population"
        },
        binding=False,
    )
    with pytest.raises(ValueError, match="requires an explicit same-terminal-members"):
        compile_incident_state(setup_from_payload(payload), "state")
    payload["report_population_bindings"] = [
        {
            "source_scenario_id": "scenario",
            "target_scenario_id": "report",
            "rationale": "An explicit target-membership assumption for the same terminal members.",
        }
    ]
    first = compile_incident_state(setup_from_payload(payload), "state")
    payload["report_population_bindings"][0]["rationale"] += (
        " Changed declared transport."
    )
    second = compile_incident_state(setup_from_payload(payload), "state")
    assert first.conditional_key != second.conditional_key


def test_legitimate_overlapping_report_aliases_keep_unique_gross() -> None:
    payload = _report_payload(binding=False)
    payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"]["member_outcomes"][
        0
    ]["components"][0]["scenario_ids"] = ["scenario", "report"]
    study = setup_from_payload(payload)
    plan = ExecutionPlan(
        plan_id="overlap",
        required_arms=("reference",),
        simulations=200,
        seed=9,
        requested_views=(
            "gross",
            "scenario::scenario::gross",
            "scenario::report::gross",
        ),
    )
    draw, unavailable, allocation = simulate_incident_state(
        plan, compile_incident_state(study, "state"), response_sha256="a" * 64
    )
    total = allocation.total_initiators
    assert isinstance(total, int)
    assert draw.annual_losses["gross"].sum() == 10 * total
    assert set(unavailable) == {"scenario::scenario::gross", "scenario::report::gross"}


def test_undeclared_report_tags_are_rejected_at_contract_boundary() -> None:
    payload = study_payload()
    payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"]["member_outcomes"][
        0
    ]["scenario_ids"] = ["unknown"]
    with pytest.raises(ValidationError, match="report tags must refer to declared"):
        setup_from_payload(payload)


@pytest.mark.parametrize("location", ["outcome", "campaign", "joint_campaign"])
def test_adjacent_report_tag_locations_cannot_bypass_basis_binding(location) -> None:
    payload = _report_payload(target_changes={"currency": "EUR"})
    kernel = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"]
    member = kernel["member_outcomes"][0]
    member["components"][0]["scenario_ids"] = ["scenario"]
    if location == "outcome":
        member["scenario_ids"] = ["report"]
    elif location == "campaign":
        kernel["campaign_components"] = [
            {
                **deepcopy(member["components"][0]),
                "component_id": "response",
                "scenario_ids": ["report"],
            }
        ]
    else:
        kernel.pop("cluster")
        kernel.pop("member_outcomes")
        kernel["explicit_joint_members"] = [
            {
                "probability": 1,
                "members": [member],
                "campaign_components": [
                    {
                        **deepcopy(member["components"][0]),
                        "component_id": "response",
                        "scenario_ids": ["report"],
                    }
                ],
            }
        ]
    with pytest.raises(ValueError, match="incompatible currency"):
        compile_incident_state(setup_from_payload(payload), "state")


def test_own_campaign_response_cost_remains_available_after_perfect_prevention() -> (
    None
):
    payload = study_payload()
    payload["populations"][0].update(threshold=100, threshold_comparator="gt")
    kernel = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"]
    kernel["unit_probability"] = 0
    kernel["campaign_components"] = [
        {
            **deepcopy(kernel["member_outcomes"][0]["components"][0]),
            "component_id": "response",
            "amount": 20,
        }
    ]
    compiled = compile_incident_state(setup_from_payload(payload), "state")
    assert compiled.incidents[0].annual_rate == 1


def _identified_candidate_payload(target_changes=None):
    payload = _report_payload(target_changes=target_changes, binding=False)
    member = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
        "member_outcomes"
    ][0]
    member.update(
        components=[],
        composition_complete=False,
        identified_gross_total=40,
        scenario_ids=["scenario", "report"],
    )
    return payload


@pytest.mark.parametrize(
    "target_changes",
    [
        {"currency": "EUR"},
        {"threshold": 100, "threshold_comparator": "gt"},
        {"terminal_unit": "A different terminal unit"},
    ],
)
def test_unidentified_gross_candidates_do_not_assert_target_population(
    target_changes,
) -> None:
    from catmodel.simulation.incident_runtime import view_admission

    study = setup_from_payload(_identified_candidate_payload(target_changes))
    compiled = compile_incident_state(study, "state")
    assert set(compiled.unavailable_views) == {
        "scenario::scenario",
        "scenario::report",
    }
    assert "pooled" not in compiled.unavailable_views
    reasons, _, _ = view_admission(compiled, ("gross", "scenario::report::gross"))
    assert "scenario::report::gross" in reasons


def test_candidate_allocation_is_unavailable_even_with_no_realized_losses() -> None:
    payload = _identified_candidate_payload(
        {"threshold": 100, "threshold_comparator": "gt"}
    )
    payload["joint_states"][0]["laws"][0]["reference_annual_rate"] = 1e-200
    study = setup_from_payload(payload)
    plan = ExecutionPlan(
        plan_id="rare-candidates",
        required_arms=("reference",),
        simulations=20,
        seed=2,
        requested_views=("gross", "scenario::report::gross"),
    )
    draw, unavailable, _ = simulate_incident_state(
        plan, compile_incident_state(study, "state"), response_sha256="a" * 64
    )
    assert set(draw.annual_losses) == {"gross"}
    assert np.all(draw.annual_losses["gross"] == 0)
    assert "scenario::report::gross" in unavailable


def test_single_target_incomplete_gross_still_requires_basis_and_qualification() -> (
    None
):
    for changed in (
        {"currency": "EUR"},
        {"threshold": 100, "threshold_comparator": "gt"},
    ):
        payload = _identified_candidate_payload(changed)
        payload["report_population_bindings"] = [
            {
                "source_scenario_id": "scenario",
                "target_scenario_id": "report",
                "rationale": "Explicit single-target report assumption cannot convert basis or qualification.",
            }
        ]
        member = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
            "member_outcomes"
        ][0]
        member["scenario_ids"] = ["report"]
        with pytest.raises(
            ValueError,
            match=r"incompatible currency|terminal-total support contradicts",
        ):
            compile_incident_state(setup_from_payload(payload), "state")


def test_ambiguous_fallback_cannot_launder_an_explicit_component_target() -> None:
    payload = _identified_candidate_payload({"currency": "EUR"})
    member = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
        "member_outcomes"
    ][0]
    member["components"] = [
        {"component_id": "known_subset", "amount": 10, "scenario_ids": ["report"]}
    ]
    with pytest.raises(ValueError, match="incompatible currency"):
        compile_incident_state(setup_from_payload(payload), "state")


def test_candidate_ids_must_be_distinct_and_declared() -> None:
    for tags, reason in (
        (["report", "report"], "must be unique"),
        (["scenario", "unknown"], "report tags must refer to declared"),
    ):
        payload = _identified_candidate_payload()
        payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
            "member_outcomes"
        ][0]["scenario_ids"] = tags
        with pytest.raises(ValidationError, match=reason):
            setup_from_payload(payload)


def test_checked_in_stipulated_example_keeps_identified_gross_with_ambiguous_allocation(
    tmp_path,
) -> None:
    from catmodel.resources import config_path
    from catmodel.results.execution import run_setup
    from catmodel.setup.store import build_setup

    source = config_path("setups", "stipulated-campaign")
    study = setup_from_json((source / "setup.json").read_bytes())
    plan = ExecutionPlan.model_validate_json(
        (source / "plan.json").read_bytes()
    ).model_copy(update={"simulations": 64})
    package = build_setup(
        destination=tmp_path / "package",
        setup=with_execution(
            study,
            plan,
            setup_id="checked-in-example",
            evidence_files={
                "assumptions.json": (
                    source / "evidence" / "assumptions.json"
                ).read_bytes()
            },
        ),
        code_commit="fixture",
    )
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    assert set(report["arms"]) == set(plan.required_arms)
    for arm in report["arms"].values():
        metrics = reduced_arm(arm)["metrics"]
        assert present(available_metric(metrics["gross"])["expected_annual_loss"]) > 0
        assert metrics["scenario::campaign_harm::gross"]["status"] == "unavailable"
        assert metrics["net_resource_cost"]["status"] == "unavailable"


def _pure_candidate_basis_payload():
    return _identified_candidate_payload(
        {"currency": "EUR", "monetary_basis": "EUR at the study start"}
    )


def _candidate_public_package(tmp_path, payload, *, name="package"):
    from catmodel.setup.store import build_setup

    return build_setup(
        destination=tmp_path / name,
        setup=with_execution(
            setup_from_payload(payload),
            ExecutionPlan(
                plan_id="candidate-basis",
                required_arms=("reference",),
                simulations=80,
                seed=22,
                requested_views=("gross", "scenario::report::gross"),
            ),
            setup_id=name,
        ),
        code_commit="fixture",
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("zero_origin", [False, True])
def test_pure_eur_candidate_does_not_change_known_usd_gross_basis(
    tmp_path, reverse, zero_origin
) -> None:
    from catmodel.results.execution import run_setup

    payload = _pure_candidate_basis_payload()
    if not zero_origin:
        state = payload["joint_states"][0]
        state["laws"] = state["laws"][:1]
        state["absent_populations"] = [
            {
                "scenario_id": "report",
                "disposition": "structural_zero",
                "reason": "There is no independent physical report-population origin in this state.",
            }
        ]
    if reverse:
        payload["populations"].reverse()
    compiled = compile_incident_state(setup_from_payload(payload), "state")
    assert "pooled" not in compiled.unavailable_views
    assert compiled.view_bases["pooled"]["currency"] == "USD"
    assert compiled.view_bases["scenario::report"]["currency"] == "EUR"
    package = _candidate_public_package(tmp_path, payload)
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    metrics = reduced_arm(report["arms"]["reference"])["metrics"]
    assert available_metric(metrics["gross"])["expected_annual_loss"] == pytest.approx(
        40
    )
    assert present(available_metric(metrics["gross"])["basis"])["currency"] == "USD"
    assert metrics["scenario::report::gross"]["status"] == "unavailable"


def test_positive_eur_physical_origin_still_blocks_pooled_currency_mix() -> None:
    from catmodel.simulation.incident_runtime import view_admission

    payload = _pure_candidate_basis_payload()
    payload["joint_states"][0]["laws"][1]["reference_annual_rate"] = 1
    study = setup_from_payload(payload)
    compiled = compile_incident_state(study, "state")
    assert "currency and monetary basis" in compiled.unavailable_views["pooled"]
    assert "pooled" not in compiled.view_bases
    assert "gross" in view_admission(compiled, ("gross",))[0]


def test_candidate_basis_exclusion_does_not_hide_unknown_additional_population() -> (
    None
):
    from catmodel.simulation.incident_runtime import view_admission

    payload = _pure_candidate_basis_payload()
    state = payload["joint_states"][0]
    state["laws"] = state["laws"][:1]
    state["absent_populations"] = [
        {
            "scenario_id": "report",
            "disposition": "unavailable",
            "reason": "Additional physical origins have no authored incident law.",
        }
    ]
    study = setup_from_payload(payload)
    compiled = compile_incident_state(study, "state")
    assert "currency" not in compiled.unavailable_views["pooled"]
    assert "unquantified populations" in compiled.unavailable_views["pooled"]
    assert "gross" in view_admission(compiled, ("gross",))[0]


def test_public_reweight_checks_actual_conditional_bases_after_candidate_exclusion(
    tmp_path,
) -> None:
    from catmodel.results.execution import replay_capture, run_setup

    payload = _pure_candidate_basis_payload()
    usd = payload["joint_states"][0]
    usd.update(state_id="usd", weight=1)
    eur = deepcopy(usd)
    eur.update(state_id="eur", weight=0)
    eur["laws"][0]["reference_annual_rate"] = 0
    eur["laws"][1]["reference_annual_rate"] = 1
    eur["laws"][1]["segments"][0]["kernel"]["member_outcomes"][0].update(
        components=[],
        composition_complete=False,
        identified_gross_total=60,
        scenario_ids=["scenario", "report"],
    )
    payload["joint_states"].append(eur)
    package = _candidate_public_package(tmp_path, payload)
    original = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    assert available_metric(
        reduced_arm(original["arms"]["reference"])["metrics"]["gross"]
    )["expected_annual_loss"] == pytest.approx(40)
    mixed = replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "mixed.json",
        state_weights={"usd": 0.5, "eur": 0.5},
        code_commit="fixture",
    )
    assert (
        reduced_arm(mixed["arms"]["reference"])["metrics"]["gross"]["status"]
        == "unavailable"
    )
    euro_only = replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "eur.json",
        state_weights={"usd": 0, "eur": 1},
        code_commit="fixture",
    )
    metric = available_metric(
        reduced_arm(euro_only["arms"]["reference"])["metrics"]["gross"]
    )
    assert metric["expected_annual_loss"] == pytest.approx(60)
    assert present(metric["basis"])["currency"] == "EUR"
