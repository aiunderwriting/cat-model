"""Comparison arms of context-bound public-trajectory setups."""

import json
import math
from functools import partial
from typing import Any

import numpy as np
import pytest

from catmodel.contracts.contextual_risk import (
    ContextualRiskAnswer,
    ContextualRiskInstrument,
    ContextualRiskNode,
)
from catmodel.contracts.elicitation import Quantiles
from catmodel.contracts.incident import IncidentLaw
from catmodel.contracts.run_setup import RunSetup
from catmodel.hashing import canonical_json_sha256
from catmodel.resources import scenarios_path
from catmodel.results.reduction import tail_diagnostics
from catmodel.scenarios.loader import load_scenarios
from catmodel.setup import public_frontier
from catmodel.setup import public_frontier_comparisons as comparisons
from catmodel.setup.contextual_node import CompileOptions, compile_contextual_node
from catmodel.setup.public_frontier import (
    PublicFrontierRiskInput,
    public_frontier_setup_fields,
)
from catmodel.setup.public_frontier_comparisons import (
    GENERATED_FAMILIES,
    AuthoredCapStresses,
    PublicForecastAlternative,
    _cap_alternative,
    _endpoint,
    _Excluded,
    _shifted,
    _uncapped,
)
from catmodel.setup.public_study import _cap_stresses
from catmodel.setup.rank_spread import REFERENCE_SPREAD_TAIL_FORM
from catmodel.setup.sensitivity_catalog import (
    DEFAULT_CONTEXT_BOUND_FAMILIES,
    FRONTIER_LAB_VALUE_SENSITIVITY_USD,
    FRONTIER_LAB_VALUE_USD,
    PUBLIC_STUDY_UNAVAILABLE_CHALLENGES,
    SENSITIVITY_FAMILIES,
)
from catmodel.simulation.capability_mixture import CapabilityMixture
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_moments import analytic_linear_eals
from catmodel.simulation.incident_runtime import (
    score_multiplier,
    simulate_incident_state,
    view_admission,
)
from test_support.contextual_risk import (
    contextual_answer,
    contextual_instrument,
    fixture_artifact,
    governed_contextual_answer,
    governed_contextual_instrument,
)
from test_support.count_law_oracle import CountLawOracle
from test_support.run_setup import scientific_digest

RATIONALES: dict[str, Any] = {
    "exposure_rationale": "Synthetic source-status populations are disjoint, with unique incident costs.",
    "residual_independence_rationale": "Synthetic independent counts conditional on public path and respondent.",
    "parameter_coupling_rationale": "Synthetic matched respondents and independent ranks per source status.",
}
MODELS = ("model-a", "model-b", "model-c")
CATASTROPHE, UNCAPPED = "compromise_critical_infrastructure", "output_reliance_legal"
CATALOG = {family.family_id for family in SENSITIVITY_FAMILIES}
STRESSES = {
    CATASTROPHE: AuthoredCapStresses(
        cap_values=(1e13, 5e14),
        uncapped_alternative=True,
        endpoint_from_tail_class=True,
    ),
    UNCAPPED: AuthoredCapStresses(),
}


GOVERNED_RATE = (0.05, 0.1, 0.2, 0.4, 1.0)
"""The governed fixture answer's count quantiles at unit scale."""


def _with_large_loss_counts(answer, shares):
    """The answer with large and very-large loss counts that are ``shares``
    of its counts, where its node asks for them."""
    if shares is None or answer.large_loss_count is None:
        return answer
    large, very_large = shares
    payload = answer.model_dump(mode="json")
    payload["large_loss_count"] = {
        key: large * value for key, value in payload["rate"].items()
    }
    payload["very_large_loss_count"] = {
        key: very_large * value for key, value in payload["rate"].items()
    }
    return ContextualRiskAnswer.model_validate(payload)


def _inputs(
    models=MODELS,
    draws=(("low", 1, 0.25), ("high", 3, 0.75)),
    lab_caps=None,
    lab_id=None,
    large_loss_shares=None,
):
    result = []
    for draw, score, weight in draws:
        for status in ("public", "not_yet_public"):
            payload = governed_contextual_instrument(model_ids=models).model_dump(
                mode="json"
            )
            if lab_id is not None:
                payload["lab"].update(entity_id=lab_id, status="actual")
                payload["model"]["related_entity_ids"] = [lab_id]
            payload.update(
                context_id=f"{draw}-{status}",
                coordinate_kind="public_trajectory",
                reference_eci=0,
                anchor_scores=[0],
                integer_min=0,
                integer_max=0,
                capability_access="internal",
            )
            payload["capability_scale"]["scale_id"] = "eci"
            payload["model"]["entity_id"] = f"model-{status}"
            payload["deployment"]["entity_id"] = f"deployment-{status}"
            payload["lab"]["related_entity_ids"] = [f"model-{status}"]
            payload["deployment_binding"].update(
                model_id=f"model-{status}", deployment_id=f"deployment-{status}"
            )
            for node in payload["nodes"]:
                node["population"]["deployment_ids"] = [f"deployment-{status}"]
                if lab_caps and node["node_id"] in lab_caps:
                    node["lab_loss_severity_cap"] = lab_caps[node["node_id"]]
            payload["public_trajectory"] = {
                "trajectory_sha256": "e" * 64,
                "draw_id": draw,
                "observed_at": "2026-12-01T00:00:00Z",
                "observed_values": [score, 100 + score],
                "earlier_frontier": None,
                "times": [f"2027-{month:02d}-01T00:00:00Z" for month in range(1, 13)]
                + ["2028-01-01T00:00:00Z"],
                "coordinates": [
                    {
                        "coordinate_id": lab,
                        "lab_id": lab,
                        "domain_id": "general",
                        "access": "public",
                        "scale_id": "eci",
                        "definition": f"Synthetic {lab} general ECI",
                    }
                    for lab in ("lab_a", "lab_b")
                ],
                "values": [[score, 100 + score]] * 13,
                "interpolation": "linear",
                "target_lab": "public_frontier",
                "source_model_status": status,
                "source_references": ["urn:fixture:public-frontier"],
                "rationale": "Synthetic joint public forecast condition.",
                "limitations": ["Synthetic values validate software only."],
            }
            instrument = ContextualRiskInstrument.model_validate(payload)
            nodes = {node.node_id: node for node in instrument.nodes}
            artifact = fixture_artifact(
                instrument,
                lambda prompt, nodes=nodes, score=score, status=status: (
                    _with_large_loss_counts(
                        governed_contextual_answer(
                            nodes[prompt.node.node_id],
                            scale=score
                            * (1 + MODELS.index(prompt.respondent.model_id))
                            * (1 if status == "public" else 2),
                        ),
                        large_loss_shares,
                    )
                ),
            )
            result.append(
                PublicFrontierRiskInput(
                    instrument, artifact, weight, quadrature_sha256="q" * 64
                )
            )
    return tuple(result)


def _compile(inputs=None, **options):
    settings: dict[str, Any] = {
        "cap_stresses": STRESSES,
        **RATIONALES,
        **options,
    }
    return RunSetup(
        **public_frontier_setup_fields(
            _inputs() if inputs is None else inputs,
            setup_id="synthetic-public",
            years_per_state=8,
            seed=4,
            **settings,
        )
    )


def test_a_draw_s_selections_are_made_once_and_reused_by_every_arm(monkeypatch):
    made = []
    selections = public_frontier.contextual_respondent_selections

    def counted(instrument, artifact, weights, **options):
        made.append(instrument.context_id)
        return selections(instrument, artifact, weights, **options)

    monkeypatch.setattr(public_frontier, "contextual_respondent_selections", counted)
    batches = []

    def recorded(function, tasks):
        batches.append(list(tasks))
        return [function(task) for task in tasks]

    inputs = _inputs()
    shared = _compile(inputs, map_tasks=recorded)
    # Making them checks the whole collection, so it happens once a context
    # however many arms compile the draw.
    assert sorted(made) == sorted(row.instrument.context_id for row in inputs)
    reference, *arms = batches
    assert len(arms) >= 2
    assert all(
        task.selections is None
        and all(item.artifact is not None for item in task.pair.values())
        for task in reference
    )
    # An arm's task carries the selections and not the collection.
    assert all(
        task.selections is not None
        and set(task.selections) == set(task.pair)
        and all(item.artifact is None for item in task.pair.values())
        for batch in arms
        for task in batch
    )
    # A setup whose every compilation makes its own selections is the same.
    compile_cells = public_frontier._compile_cells
    monkeypatch.setattr(
        public_frontier,
        "_compile_cells",
        lambda *args, selections=None, **options: compile_cells(*args, **options),
    )
    made.clear()
    alone = _compile(inputs)
    assert len(made) == len(batches) * len(inputs)
    assert alone == shared
    assert scientific_digest(alone) == scientific_digest(shared)


def test_selections_are_made_from_outcomes_the_inputs_carry(monkeypatch):
    """Inputs read where their collections were checked carry the outcomes:
    each draw's selections are made from them, and no collection is checked
    again."""
    from dataclasses import replace

    from catmodel.setup import contextual_setup

    inputs = _inputs()
    carried = tuple(
        replace(
            row,
            outcomes=contextual_setup.checked_hierarchy_outcomes(
                row.instrument, row.artifact
            ),
        )
        for row in inputs
    )
    checks = []
    for name in (
        "validate_contextual_artifact",
        "validate_eal_crosschecks",
        "validate_hierarchy_cases",
    ):
        monkeypatch.setattr(
            contextual_setup, name, lambda *_args, name=name: checks.append(name)
        )
    batches = []

    def recorded(function, tasks):
        batches.append(list(tasks))
        return [function(task) for task in tasks]

    given = _compile(carried, map_tasks=recorded)
    assert checks == []
    reference, *arms = batches
    assert len(arms) >= 2
    # The reference draws make the selections, from the outcomes they carry.
    assert all(
        task.selections is None
        and all(
            item.artifact is not None and item.outcomes is not None
            for item in task.pair.values()
        )
        for task in reference
    )
    assert all(
        task.selections is not None
        and all(
            item.artifact is None and item.outcomes is None
            for item in task.pair.values()
        )
        for batch in arms
        for task in batch
    )
    checked = _compile(inputs)
    assert len(checks) == 3 * len(inputs)
    assert given == checked
    assert scientific_digest(given) == scientific_digest(checked)


def _rows(setup, family):
    return [
        row
        for row in setup.execution.comparison_requirements
        if row.family_id == family
    ]


def _numbers(state):
    return canonical_json_sha256(
        [
            law.model_dump(mode="json", exclude={"conditioning_context"})
            for law in state.laws
        ]
    )


def _eal(setup, weights, view="welfare"):
    return math.fsum(
        weights[state.state_id]
        * analytic_linear_eals(
            compile_incident_state(setup, state.state_id), (view,)
        ).values[view]
        for state in setup.joint_states
    )


def _arm_eal(setup, arm, view="welfare"):
    return math.fsum(
        state.weight
        * analytic_linear_eals(
            compile_incident_state(setup, state.state_id, arm), (view,)
        ).values[view]
        for state in setup.joint_states
    )


def test_the_route_generates_every_catalog_family():
    assert set(GENERATED_FAMILIES) == CATALOG
    assert set(DEFAULT_CONTEXT_BOUND_FAMILIES) <= set(GENERATED_FAMILIES)


def test_default_selection_freezes_every_family_with_arms_or_reasons():
    alternatives = {
        "higher_pace_shape": PublicForecastAlternative(
            {"low": 0.4, "high": 0.6}, {"pace_shape": 7.0}
        ),
        "upper_trend_discrepancy": PublicForecastAlternative(
            None, {}, "Synthetic missing density."
        ),
    }
    setup = _compile(forecast_alternatives=alternatives)
    plan = setup.execution
    frozen = CATALOG | set(PUBLIC_STUDY_UNAVAILABLE_CHALLENGES)
    assert set(plan.requested_comparison_families) == frozen
    assert {row.family_id for row in plan.comparison_requirements} == frozen
    challenges = [
        row
        for row in plan.comparison_requirements
        if row.family_id in PUBLIC_STUDY_UNAVAILABLE_CHALLENGES
    ]
    assert len(challenges) == len(PUBLIC_STUDY_UNAVAILABLE_CHALLENGES)
    assert all(row.status == "unavailable" and row.reason for row in challenges)
    assert all(row.scenario_ids == (CATASTROPHE, UNCAPPED) for row in challenges)
    assert plan.required_arms == (
        "reference",
        "public_forecast::higher_pace_shape",
        *(f"model_pooling_leave_one_out::{model}" for model in MODELS),
        "tail_shape_lower",
        "tail_shape_higher",
        "uncapped_severity",
    )
    required = {
        row.arm_id: row.scenario_ids
        for row in plan.comparison_requirements
        if row.status == "required"
    }
    assert required["tail_shape_higher"] == (CATASTROPHE, UNCAPPED)
    assert required["uncapped_severity"] == (CATASTROPHE,)
    (uncapped_note,) = [
        row for row in _rows(setup, "uncapped_severity") if row.status != "required"
    ]
    assert (uncapped_note.status, uncapped_note.scenario_ids) == (
        "inapplicable",
        (UNCAPPED,),
    )
    assert any(
        row.status == "unavailable"
        and row.reason.startswith("upper_trend_discrepancy:")
        for row in _rows(setup, "public_forecast")
    )
    (severity_cap,) = _rows(setup, "severity_cap")
    assert severity_cap.status == "unavailable"
    assert "--comparison-family severity_cap" in severity_cap.reason
    # The uncapped alternative removes the catastrophe endpoint from the societal and lab
    # laws; the stipulated frontier-lab value still caps the lab law.
    arm = next(
        row for row in setup.alternatives if row.alternative_id == "uncapped_severity"
    )
    assert setup.overrides.of(arm.alternative_id)
    assert arm.description.endswith(
        f"Count laws use k = 1 with tail_form = {REFERENCE_SPREAD_TAIL_FORM}."
    )
    assert (
        "severity laws, branch probabilities, shares, exposure and controls are held fixed"
        not in arm.description
    )
    for override in setup.overrides.of(arm.alternative_id):
        assert override.law.scenario_id == CATASTROPHE
        kernel = override.law.segments[0].kernel
        assert override.law.annual_lab_losses[0].severity is not None
        positive = override.law.annual_lab_losses[0].severity.components[-1]
        assert positive.severity_cap == FRONTIER_LAB_VALUE_USD
        societal = kernel.member_outcomes[0].components[0].severity
        assert societal.components[0].severity_cap is None
    # The higher tail alternative reaches 1.0 in the capped catastrophe class and 0.9
    # in the cyber/operational class.
    higher = next(
        row for row in setup.alternatives if row.alternative_id == "tail_shape_higher"
    )
    shapes = {
        override.law.scenario_id: override.law.annual_lab_losses[0]
        .severity.components[-1]
        .gpd_shape
        for override in setup.overrides.of(higher.alternative_id)
    }
    assert shapes == {CATASTROPHE: 1.0, UNCAPPED: 0.9}


def test_explicit_selection_and_missing_governance_are_recorded():
    setup = _compile(
        comparison_families=("severity_cap", "model_pooling", "count_family"),
        cap_stresses={
            CATASTROPHE: STRESSES[CATASTROPHE],
            UNCAPPED: "Synthetic governance mismatch.",
        },
    )
    assert setup.execution.required_arms == (
        "reference",
        *(f"model_pooling::{model}" for model in MODELS),
        "severity_cap_lower",
        "severity_cap_higher",
        "fixed_vmr_nb_2",
    )
    (tail,) = _rows(setup, "tail_shape")
    assert tail.status == "unavailable"
    assert "this import's comparison-family list" in tail.reason
    notes = {
        (row.status, row.scenario_ids, row.reason)
        for row in _rows(setup, "severity_cap")
        if row.status != "required"
    }
    assert notes == {("unavailable", (UNCAPPED,), "Synthetic governance mismatch.")}
    with pytest.raises(ValueError, match="unknown comparison families"):
        _compile(comparison_families=("not_a_family",))


def test_fewer_than_three_models_make_leave_one_out_inapplicable():
    setup = _compile(_inputs(models=MODELS[:2]))
    (row,) = _rows(setup, "model_pooling_leave_one_out")
    assert row.status == "inapplicable"
    assert "at least 3 respondent models" in row.reason


def test_leave_one_model_out_equals_an_import_without_that_model():
    full = _compile(comparison_families=("model_pooling_leave_one_out",))
    reference = {state.state_id: state.weight for state in full.joint_states}
    for dropped in MODELS:
        retained = tuple(model for model in MODELS if model != dropped)
        reduced = _compile(
            _inputs(models=retained), comparison_families=("model_pooling",)
        )
        arm = next(
            row
            for row in full.alternatives
            if row.alternative_id == f"model_pooling_leave_one_out::{dropped}"
        )
        assert arm.state_weights is not None
        assert not full.overrides.of(arm.alternative_id)
        assert _eal(full, arm.state_weights) == pytest.approx(
            _eal(reduced, {s.state_id: s.weight for s in reduced.joint_states}),
            rel=1e-12,
        )
        # Retained states keep their laws; the reweighting changes weights only.
        # Conditioning contexts differ only by the respondent roster's digest.
        laws = sorted(
            _numbers(state)
            for state in full.joint_states
            if arm.state_weights[state.state_id] > 0
        )
        assert laws == sorted(_numbers(state) for state in reduced.joint_states)
    assert _eal(full, reference) != pytest.approx(
        _eal(full, full.alternatives[0].state_weights)
    )


def test_forecast_arm_with_the_base_node_weights_reproduces_the_reference():
    setup = _compile(
        comparison_families=("public_forecast",),
        forecast_alternatives={
            "same": PublicForecastAlternative({"low": 0.25, "high": 0.75}, {})
        },
    )
    (arm,) = setup.alternatives
    assert arm.state_weights == {
        state.state_id: state.weight for state in setup.joint_states
    }
    assert arm.description.endswith(
        f"Count laws use k = 1 with tail_form = {REFERENCE_SPREAD_TAIL_FORM}."
    )


def test_capped_lab_law_admits_the_catastrophe_higher_tail_alternative():
    node = governed_contextual_instrument((CATASTROPHE,)).nodes[0]
    payload = node.model_dump(mode="json")
    payload["lab_loss_gpd_shape"] = 1.0
    assert ContextualRiskNode.model_validate(payload).lab_loss_gpd_shape == 1.0
    payload["lab_loss_severity_cap"] = None
    with pytest.raises(ValueError, match="uncapped lab-loss tail shape"):
        ContextualRiskNode.model_validate(payload)


def test_uncapped_alternative_refuses_a_lab_tail_without_a_finite_mean():
    node = governed_contextual_instrument((CATASTROPHE,)).nodes[0]
    payload = node.model_dump(mode="json")
    payload["lab_loss_gpd_shape"] = 1.0
    with pytest.raises(_Excluded, match="finite-mean bound") as refused:
        _uncapped(ContextualRiskNode.model_validate(payload), STRESSES[CATASTROPHE])
    assert refused.value.status == "unavailable"


@pytest.mark.parametrize(
    ("shape", "step", "cap", "expected"),
    (
        (0.7, -0.2, None, 0.5),
        (0.7, 0.2, None, 0.9),
        (0.8, -0.2, 1e14, 0.6),
        # The capped sanity bound and the lower bound are inclusive; binary
        # addition puts -0.8 - 0.15 below the lower bound.
        (2.3, 0.2, 1e14, 2.5),
        (-0.8, -0.15, None, -0.95),
    ),
)
def test_a_shifted_shape_is_the_decimal_sum_and_meets_its_bounds_as_one(
    shape, step, cap, expected
):
    assert _shifted(shape, step, cap, "shared-total") == expected


@pytest.mark.parametrize(("shape", "step"), ((0.75, 0.2), (0.8, 0.15)))
def test_an_uncapped_shape_shifted_onto_the_finite_mean_bound_is_unavailable(
    shape, step
):
    with pytest.raises(_Excluded, match=r"tail shape at 0\.95, outside") as refused:
        _shifted(shape, step, None, "shared-total")
    assert refused.value.status == "unavailable"
    assert _shifted(shape, step, 1e14, "shared-total") == 0.95


def _governed_shapes(law):
    """The GPD shapes of a law's societal total and annual lab business loss."""
    societal = law.segments[0].kernel.member_outcomes[0].components[0].severity
    lab = law.annual_lab_losses[0].severity
    return societal.components[0].gpd_shape, lab.components[-1].gpd_shape


def test_tail_shape_arms_carry_decimal_shapes_and_one_half_keeps_infinite_variance():
    """The step moves 0.7 to 0.5 and 0.9, and 0.8 to 0.6 and 1.0, where binary
    addition gives 0.49999999999999994, 0.8999999999999999 and
    0.6000000000000001.

    At 0.5 an uncapped law has infinite variance, as at 0.7, so the lower arm
    reads the cyber/operational cell's moments as the reference arm does: its
    TVaR coefficient is one by law, and its years reconcile with the analytic
    EAL through their limited mean. The capped catastrophe cell has a finite
    variance in every arm.
    """
    setup = _compile(
        _inputs(models=MODELS[:1], draws=(("only", 1, 1.0),)),
        comparison_families=("tail_shape",),
    )
    assert {
        arm: {
            override.law.scenario_id: _governed_shapes(override.law)
            for override in setup.overrides.of(arm)
        }
        for arm in ("tail_shape_lower", "tail_shape_higher")
    } == {
        "tail_shape_lower": {CATASTROPHE: (0.6, 0.6), UNCAPPED: (0.5, 0.5)},
        "tail_shape_higher": {CATASTROPHE: (1.0, 1.0), UNCAPPED: (0.9, 0.9)},
    }

    (state,) = setup.joint_states
    plan = setup.execution
    uncapped, capped = (
        f"scenario::{scenario}::gross" for scenario in (UNCAPPED, CATASTROPHE)
    )
    reference = compile_incident_state(setup, state.state_id)
    lower = compile_incident_state(setup, state.state_id, "tail_shape_lower")
    _, finite_variance, _ = view_admission(lower, plan.requested_views)
    assert finite_variance == view_admission(reference, plan.requested_views)[1]
    assert (finite_variance[uncapped], finite_variance[capped]) == (False, True)

    draws, unavailable, _ = simulate_incident_state(
        plan, lower, response_sha256=setup.scientific_sha256, years=64
    )
    assert not unavailable
    key = draws.stratum_sha256
    mixture = CapabilityMixture({key: draws}, {key: 1.0})
    for view, expected in (
        (uncapped, ("infinite_variance", "limited_mean")),
        (capped, ("regression", "mean")),
    ):
        diagnostics = tail_diagnostics(mixture, view, levels=(0.99,))
        assert (
            diagnostics["tails"]["0.99"]["tvar_estimator"]["coefficient_basis"],
            diagnostics["convergence"]["eal_reconciliation"]["statistic"],
        ) == expected


def test_cap_stresses_leave_a_lab_cap_on_another_basis():
    setup = _compile(
        _inputs(lab_caps={CATASTROPHE: 5e11}),
        comparison_families=("uncapped_severity",),
    )
    (arm,) = setup.alternatives
    for override in setup.overrides.of(arm.alternative_id):
        kernel = override.law.segments[0].kernel
        assert kernel.campaign_lab_losses[0].episode_cap == 5e11
        assert (
            override.law.annual_lab_losses[0].severity.components[-1].severity_cap
            == 5e11
        )
        societal = kernel.member_outcomes[0].components[0].severity
        assert societal.components[0].severity_cap is None


def _lab_caps(laws):
    return {
        law.annual_lab_losses[0].severity.components[-1].severity_cap for law in laws
    }


def test_lab_loss_cap_alternatives_run_only_when_selected_and_say_so():
    default = _compile()
    (row,) = _rows(default, "lab_loss_cap")
    assert row.status == "unavailable"
    assert any(
        "USD 1 trillion" in text and "not executed" in text
        for text in default.limitations
    )
    lower, higher = FRONTIER_LAB_VALUE_SENSITIVITY_USD
    # The same stipulated value caps a hypothetical and a named lab.
    for inputs in (_inputs(), _inputs(lab_id="lab_b")):
        setup = _compile(inputs, comparison_families=("lab_loss_cap",))
        assert _lab_caps(law for state in setup.joint_states for law in state.laws) == {
            FRONTIER_LAB_VALUE_USD
        }
        arms = {row.alternative_id: row for row in setup.alternatives}
        assert list(arms) == [
            "lab_loss_cap_lower",
            "lab_loss_cap_higher",
            "lab_loss_cap_removed",
        ]
        # Removing the value leaves the catastrophe endpoint and no other cap.
        for arm, expected in (
            ("lab_loss_cap_lower", {lower}),
            ("lab_loss_cap_higher", {higher}),
            ("lab_loss_cap_removed", {1e14, None}),
        ):
            assert _lab_caps(row.law for row in setup.overrides.of(arm)) == expected, (
                arm
            )
        assert any(
            "USD 300 billion, USD 3 trillion and no lab cap" in text
            and "are executed" in text
            for text in setup.limitations
        )


def test_spread_tail_width_and_endpoint_arms_change_only_their_target():
    setup = RunSetup(
        **public_frontier_setup_fields(
            _inputs(),
            setup_id="synthetic-public",
            years_per_state=8,
            seed=4,
            comparison_families=(
                "epistemic_spread_calibration",
                "epistemic_tail_width",
                "severity_endpoint_stress",
            ),
            cap_stresses={
                CATASTROPHE: STRESSES[CATASTROPHE],
                UNCAPPED: AuthoredCapStresses(endpoint_stress=5e12),
            },
            **RATIONALES,
        )
    )
    arms = {row.alternative_id: row for row in setup.alternatives}
    assert set(arms) == {
        "epistemic_spread_widened",
        "epistemic_spread_narrowed",
        "epistemic_tail_narrowed",
        "epistemic_tail_widened",
        "severity_endpoint",
    }
    reference = {
        (state.state_id, law.origin_id): law
        for state in setup.joint_states
        for law in state.laws
    }
    for arm in ("epistemic_spread_widened", "epistemic_tail_widened"):
        for override in setup.overrides.of(arm):
            before = reference[override.state_id, override.law.origin_id]
            # Rank changes move only the expected count's law and its mean over
            # the rank; the marks are unchanged.
            assert override.law.reference_annual_rate != before.reference_annual_rate
            assert override.law.segments == before.segments
    (endpoint,) = [
        row
        for row in setup.execution.comparison_requirements
        if row.arm_id == "severity_endpoint"
    ]
    assert endpoint.scenario_ids == (UNCAPPED,)
    for override in setup.overrides.of("severity_endpoint"):
        kernel = override.law.segments[0].kernel
        severity = kernel.member_outcomes[0].components[0].severity
        assert severity is not None
        assert severity.components[0].severity_cap == 5e12
        # A scenario endpoint bounds the scenario's severity, not the lab's costs.
        before = reference[override.state_id, override.law.origin_id]
        assert (
            kernel.campaign_lab_losses == before.segments[0].kernel.campaign_lab_losses
        )


def test_comparison_identity_includes_the_effective_rank_tail_widths():
    instrument = governed_contextual_instrument((CATASTROPHE,))
    node = instrument.nodes[0]
    answer = governed_contextual_answer(node).model_copy(
        update={
            "rate": Quantiles(
                p5=1,
                p25=10,
                p50=20,
                p75=30,
                p95=86.9644417849956,
            )
        }
    )
    answers = {instrument.reference_eci: answer}
    pure_stretch = compile_contextual_node(
        instrument,
        node,
        answers,
        instrument.reference_eci,
        options=CompileOptions(rank_spread=1.5, spread_tail_form="pure_stretch"),
    )
    calibrated = compile_contextual_node(
        instrument,
        node,
        answers,
        instrument.reference_eci,
        options=CompileOptions(
            rank_spread=1.5, spread_tail_form=REFERENCE_SPREAD_TAIL_FORM
        ),
    )
    assert isinstance(pure_stretch, IncidentLaw)
    assert isinstance(calibrated, IncidentLaw)
    assert pure_stretch.reference_annual_rate == pytest.approx(
        calibrated.reference_annual_rate, rel=1e-12
    )

    pure_numbers = comparisons._numbers(pure_stretch)
    calibrated_numbers = comparisons._numbers(calibrated)
    assert (
        pure_numbers["effective_parameter_rank_tail_widths"]
        != calibrated_numbers["effective_parameter_rank_tail_widths"]
    )


def _laws(setup, arm=None):
    """The reference laws, or the laws ``arm`` overrides, by state and origin."""
    if arm is None:
        return {
            (state.state_id, law.origin_id): law
            for state in setup.joint_states
            for law in state.laws
        }
    (alternative,) = [row for row in setup.alternatives if row.alternative_id == arm]
    return {
        (override.state_id, override.law.origin_id): override.law
        for override in setup.overrides.of(alternative.alternative_id)
    }


def _assert_reads(law, stated, *, k, factor, tail_scale=1.0):
    """``law`` is the ``stated`` count law read at spread ``k`` and scaled by
    ``factor``: its recorded reading, its five quantiles and its mean, which
    ``CountLawOracle`` integrates from the methodology's formulas."""
    rank = law.parameter_rank
    assert rank is not None
    assert (rank.rank_spread, rank.tail_form, rank.epistemic_tail_scale) == (
        k,
        REFERENCE_SPREAD_TAIL_FORM,
        tail_scale,
    )
    assert law.conditioning_context["count_factor"] == pytest.approx(factor, rel=1e-15)
    for name in ("p5", "p25", "p50", "p75", "p95"):
        assert getattr(rank.count, name) == pytest.approx(
            factor * getattr(stated, name), rel=1e-15
        )
    oracle = CountLawOracle(
        stated, tail_scale=tail_scale, spread=k, tail_form=REFERENCE_SPREAD_TAIL_FORM
    )
    assert law.reference_annual_rate == pytest.approx(factor * oracle.mean(), rel=1e-12)


@pytest.mark.parametrize(
    ("spread", "bias"), ((1.0, 0.4), (1.5, 1.0), (1.5, 0.8), (0.75, 2.5))
)
def test_a_count_law_under_spread_and_bias_is_the_bias_times_the_law_at_that_spread(
    spread, bias
):
    instrument = governed_contextual_instrument((CATASTROPHE,))
    node = instrument.nodes[0]
    stated = Quantiles(p5=1, p25=10, p50=20, p75=30, p95=86.9644417849956)
    answer = governed_contextual_answer(node).model_copy(update={"rate": stated})
    law = compile_contextual_node(
        instrument,
        node,
        {instrument.reference_eci: answer},
        instrument.reference_eci,
        options=CompileOptions(rank_spread=spread, count_factor=bias),
    )
    assert isinstance(law, IncidentLaw)
    _assert_reads(law, stated, k=spread, factor=bias)
    # A simulated year's expected count at normal score z is b Q(Phi(k z)).
    assert law.parameter_rank is not None
    oracle = CountLawOracle(stated, spread=spread, tail_form=REFERENCE_SPREAD_TAIL_FORM)
    scores = np.asarray([-3.0, -1.0, 0.0, 0.5, 2.0, 4.0])
    np.testing.assert_allclose(
        law.reference_annual_rate * score_multiplier(law.parameter_rank)(scores),
        [bias * oracle.at_score(float(score)) for score in scores],
        rtol=1e-11,
    )


COUNT_LAW_FAMILIES = (
    "epistemic_spread_calibration",
    "epistemic_tail_width",
    "common_frequency_bias",
)
# Each fixed arm of those families, in arm order: the spread it reads count
# laws at, the factor on their stated quantiles and any other tail scale.
COUNT_LAW_ARMS: dict[str, dict[str, Any]] = {
    "epistemic_spread_widened": {"k": 1.5, "factor": 1.0},
    "epistemic_spread_narrowed": {"k": 0.75, "factor": 1.0},
    "epistemic_tail_narrowed": {"k": 1.0, "factor": 1.0, "tail_scale": 0.75},
    "epistemic_tail_widened": {"k": 1.0, "factor": 1.0, "tail_scale": 1.5},
    "common_frequency_bias_lower": {"k": 1.0, "factor": 0.5},
    "common_frequency_bias_higher": {"k": 1.0, "factor": 2.0},
}


@pytest.fixture(scope="module")
def count_law_arms():
    """The synthetic study compiled with the fixed count-law families."""
    return _compile(comparison_families=COUNT_LAW_FAMILIES)


def test_the_reference_reads_count_laws_at_face_value_and_each_fixed_arm_at_the_law_it_names(
    count_law_arms,
):
    setup = count_law_arms
    # Every reference law carries its stated count law, read at face value.
    reference = _laws(setup)
    stated = {}
    for key, law in reference.items():
        assert law.parameter_rank is not None
        stated[key] = law.parameter_rank.count
        _assert_reads(law, stated[key], k=1.0, factor=1.0)

    assert setup.execution.required_arms == ("reference", *COUNT_LAW_ARMS)
    arms = {row.alternative_id: row for row in setup.alternatives}
    for arm, reading in COUNT_LAW_ARMS.items():
        laws = _laws(setup, arm)
        # Every law differs from the reference in each arm, so each overrides all.
        assert laws.keys() == stated.keys(), arm
        for key, law in laws.items():
            _assert_reads(law, stated[key], **reading)
        description = arms[arm].description
        assert (
            f"Count laws use k = {reading['k']:g} with "
            f"tail_form = {REFERENCE_SPREAD_TAIL_FORM}." in description
        )
        assert (f"Count factor = {reading['factor']:g}." in description) == (
            reading["factor"] != 1
        )
        assert arms[arm].selection_provenance["count_factor"] == pytest.approx(
            reading["factor"], rel=1e-15
        )

    required = [
        row
        for row in setup.execution.comparison_requirements
        if row.status == "required"
    ]
    assert {row.family_id for row in required} == set(COUNT_LAW_FAMILIES)
    assert {row.arm_id for row in required} == set(COUNT_LAW_ARMS)
    assert all(row.held_fixed == comparisons.HELD_FIXED for row in required)


def test_the_count_law_families_are_optional_and_unavailable_unless_selected():
    setup = _compile(comparison_families=())
    assert setup.execution.required_arms == ("reference",)
    for family in COUNT_LAW_FAMILIES:
        (row,) = _rows(setup, family)
        assert (row.status, row.reason) == (
            "unavailable",
            "Not selected by this import's comparison-family list; request it "
            f"with --comparison-family {family} or --comparison-family all.",
        )
    # No family compares tail forms, and none can be asked for.
    assert "epistemic_spread_tail_form" not in {
        row.family_id for row in setup.execution.comparison_requirements
    }
    with pytest.raises(ValueError, match="unknown comparison families"):
        _compile(comparison_families=("epistemic_spread_tail_form",))


def test_bias_arms_scale_the_reference_eal_exactly(count_law_arms):
    setup = count_law_arms
    weights = {state.state_id: state.weight for state in setup.joint_states}
    reference_eal = _eal(setup, weights)
    # Welfare is linear in the expected count, so each arm moves the EAL by
    # its factor on the reference count laws.
    for arm, factor in (
        ("common_frequency_bias_lower", 0.5),
        ("common_frequency_bias_higher", 2.0),
    ):
        assert _arm_eal(setup, arm) == pytest.approx(factor * reference_eal, rel=1e-12)

    state = next(state for state in setup.joint_states if state.weight > 0)
    reference = compile_incident_state(setup, state.state_id)
    doubled = compile_incident_state(
        setup, state.state_id, "common_frequency_bias_higher"
    )
    reference_incident = next(
        incident for incident in reference.incidents if incident.annual_rate > 0
    )
    doubled_incident = next(
        incident
        for incident in doubled.incidents
        if incident.origin_id == reference_incident.origin_id
    )
    assert doubled_incident.annual_rate == pytest.approx(
        2 * reference_incident.annual_rate
    )

    def probability_of_any_count(incident):
        if incident.count_family == "poisson":
            return -math.expm1(-incident.annual_rate)
        vmr = incident.count_variance_to_mean
        return -math.expm1(-incident.annual_rate * math.log(vmr) / (vmr - 1))

    probability_reference = probability_of_any_count(reference_incident)
    probability_doubled = probability_of_any_count(doubled_incident)
    assert probability_doubled != pytest.approx(2 * probability_reference)


def _exhibit(setup, name):
    return json.loads(setup.evidence.files[f"derived/{name}.json"])


def test_the_exhibits_read_every_count_at_face_value():
    # One draw and one model: every state holds the governed count at scale 1
    # in the public population and at scale 2 in the not-yet-public one.
    inputs = _inputs(
        models=("model-a",),
        draws=(("only", 1, 1.0),),
        large_loss_shares=(0.75, 0.04),
    )
    setup = _compile(inputs, comparison_families=())
    weights = {state.state_id: state.weight for state in setup.joint_states}

    estimate = _exhibit(setup, "frequency-estimate")
    arms = estimate["views"]["welfare"]["by_arm"]
    available = [arm for arm, row in arms.items() if row["status"] == "available"]
    assert {"final", "direct", "chain", "before_eal_crosscheck"} <= set(available)
    # The final arm is the reference run's law in every view, the gross and
    # lab views' annual term included.
    for view, by_arm in estimate["views"].items():
        assert by_arm["by_arm"]["final"]["expected_annual_loss"] == pytest.approx(
            _eal(setup, weights, view), rel=1e-9
        )
    assert "k = 1 and b = 1" in estimate["interpretation"]
    assert "k_hat" not in estimate["interpretation"]

    stated_mean = CountLawOracle(
        GOVERNED_RATE, tail_form=REFERENCE_SPREAD_TAIL_FORM
    ).mean()
    large_loss = _exhibit(setup, "large-loss-anchor")
    for scenario in (CATASTROPHE, UNCAPPED):
        row = large_loss["by_scenario"][scenario]
        assert row["stated_large_count"] == pytest.approx(
            (1 + 2) * 0.75 * stated_mean, rel=1e-12
        )
        assert row["stated_very_large_count"] == pytest.approx(
            (1 + 2) * 0.04 * stated_mean, rel=1e-12
        )
        # The modelled count reproduces the pooled anchor.
        assert row["modelled_large_count"] == pytest.approx(
            row["pooled_large_count"], rel=1e-4
        )


@pytest.mark.parametrize(
    ("endpoint", "cap_text"),
    ((5e10, "USD 50 billion"), (1e11, "USD 100 billion")),
)
def test_endpoint_at_or_below_the_very_large_level_drops_that_diagnostic(
    endpoint, cap_text
):
    # USD 50 billion is the enforcement leaf's authored endpoint.
    setup = _compile(
        comparison_families=("severity_endpoint_stress",),
        cap_stresses={
            CATASTROPHE: STRESSES[CATASTROPHE],
            UNCAPPED: AuthoredCapStresses(endpoint_stress=endpoint),
        },
    )
    (arm,) = setup.alternatives
    assert arm.alternative_id == "severity_endpoint"
    rows = _rows(setup, "severity_endpoint_stress")
    (required,) = [row for row in rows if row.status == "required"]
    assert required.scenario_ids == (UNCAPPED,)
    for override in setup.overrides.of(arm.alternative_id):
        severity = (
            override.law.segments[0].kernel.member_outcomes[0].components[0].severity
        )
        assert severity is not None
        assert severity.components[0].severity_cap == endpoint
    assert arm.selection_provenance["large_loss_level_exclusions"] == {
        UNCAPPED: (
            "The very-large-loss level USD 100 billion is at or above the alternative's cap "
            f"{cap_text}, so no capped loss exceeds it and its diagnostic count "
            "does not apply under this alternative.",
        )
    }
    assert [(row.scenario_ids, row.status, row.reason) for row in rows] == [
        ((UNCAPPED,), "required", None),
        (
            (CATASTROPHE,),
            "inapplicable",
            "No finite endpoint stress is authored for this source law.",
        ),
    ]


@pytest.mark.parametrize("endpoint", (1e9, 5e8))
def test_endpoint_at_or_below_the_large_loss_anchor_is_unavailable(endpoint):
    setup = _compile(
        comparison_families=("severity_endpoint_stress",),
        cap_stresses={
            CATASTROPHE: STRESSES[CATASTROPHE],
            UNCAPPED: AuthoredCapStresses(endpoint_stress=endpoint),
        },
    )
    assert not any(
        row.alternative_id == "severity_endpoint" for row in setup.alternatives
    )
    (uncapped,) = [
        row
        for row in _rows(setup, "severity_endpoint_stress")
        if row.scenario_ids == (UNCAPPED,)
    ]
    assert uncapped.status == "unavailable"
    assert uncapped.reason is not None
    assert "lies at or below the large-loss level USD 1 billion" in uncapped.reason
    assert "dropping the anchor would change more than the cap" in uncapped.reason


def test_endpoint_stress_does_not_change_other_comparison_families():
    families = (
        "severity_endpoint_stress",
        "uncapped_severity",
        "severity_cap",
        "count_family",
        "tail_shape",
    )
    stressed = _compile(
        comparison_families=families,
        cap_stresses={
            CATASTROPHE: STRESSES[CATASTROPHE],
            UNCAPPED: AuthoredCapStresses(endpoint_stress=5e10),
        },
    )
    no_endpoint = _compile(
        comparison_families=families,
        cap_stresses={
            CATASTROPHE: STRESSES[CATASTROPHE],
            UNCAPPED: AuthoredCapStresses(),
        },
    )

    def outside_endpoint_alternatives(setup):
        return {
            row.alternative_id: row
            for row in setup.alternatives
            if row.selection_provenance["family_id"] != "severity_endpoint_stress"
        }

    def outside_endpoint_requirements(setup):
        return [
            row
            for row in setup.execution.comparison_requirements
            if row.family_id != "severity_endpoint_stress"
        ]

    assert outside_endpoint_alternatives(stressed) == outside_endpoint_alternatives(
        no_endpoint
    )
    assert outside_endpoint_requirements(stressed) == outside_endpoint_requirements(
        no_endpoint
    )


def test_repository_cap_edits_return_valid_nodes_or_governed_exclusions():
    scenario_ids = tuple(row.id for row in load_scenarios(scenarios_path()))
    instrument = governed_contextual_instrument(scenario_ids)
    stresses = _cap_stresses(instrument)
    edits = (
        _uncapped,
        partial(_cap_alternative, index=0),
        partial(_cap_alternative, index=1),
        _endpoint,
    )
    for node in instrument.nodes:
        authored = stresses[node.population.scenario_id]
        for edit in edits:
            try:
                edited = edit(node, authored)
            except _Excluded:
                continue
            assert isinstance(edited, ContextualRiskNode)

    enforcement = next(
        node
        for node in instrument.nodes
        if node.population.scenario_id == "enforcement_matter"
    )
    endpoint = _endpoint(enforcement, stresses["enforcement_matter"])
    assert endpoint.total is not None
    assert endpoint.total.cap == 5e10
    assert endpoint.total.large_loss_level == 1e9
    assert endpoint.total.very_large_loss_level is None


def test_invalid_edit_validation_errors_are_not_recorded_as_unavailable(monkeypatch):
    def invalid_endpoint(node, stresses):
        assert node.total is not None
        assert node.total.large_loss_level is not None
        payload = node.model_dump(mode="json")
        payload["total"]["cap"] = node.total.large_loss_level
        return ContextualRiskNode.model_validate(payload)

    monkeypatch.setattr(comparisons, "_endpoint", invalid_endpoint)
    with pytest.raises(ValueError, match="large-loss levels must lie below the cap"):
        _compile(
            comparison_families=("severity_endpoint_stress",),
            cap_stresses={
                CATASTROPHE: STRESSES[CATASTROPHE],
                UNCAPPED: AuthoredCapStresses(endpoint_stress=5e10),
            },
        )


def test_stochastic_lab_loss_with_partial_probability_compiles():
    instrument = contextual_instrument()
    payload = contextual_answer().model_dump(mode="json")
    payload["lab_annual_business_loss"] = {
        "status": "available",
        "positive_probability": 0.4,
        "loss": dict(
            zip(
                ("p5", "p25", "p50", "p75", "p95", "p99", "p999"),
                (20, 40, 80, 120, 200, 600, 1000),
                strict=True,
            )
        ),
    }
    answer = ContextualRiskAnswer.model_validate(payload)
    law = compile_contextual_node(
        instrument,
        instrument.nodes[0],
        dict.fromkeys(instrument.anchor_scores, answer),
        0,
    )
    assert isinstance(law, IncidentLaw)
    severity = law.annual_lab_losses[0].severity
    assert severity is not None and severity.representation == "mixture"
    zero, positive = severity.components
    assert severity.weights == pytest.approx((0.6, 0.4))
    assert zero.p95 == 0 and zero.severity_minimum == positive.severity_minimum
