"""Captured contextual answers generate native studies and exact point kernels."""

import math
from copy import deepcopy

import numpy as np
import pytest

from catmodel.contracts.contextual_risk import (
    ContextualRiskAnswer,
    ContextualRiskArtifact,
    ContextualRiskInstrument,
)
from catmodel.contracts.incident import PopulationAbsence
from catmodel.elicitation.contextual_risk import (
    contextual_tail_diagnostic_limitations,
    contextual_tail_diagnostics,
)
from catmodel.probability.distributions import p999_implied_gpd_shape
from catmodel.results.conditional import ConditionalCapture
from catmodel.results.execution import load_run_capture, run_setup
from catmodel.results.results_export import _conditioning
from catmodel.setup.contextual_node import (
    ContextualLawUnavailable,
    compile_contextual_node,
)
from catmodel.setup.contextual_setup import compile_contextual_setup
from catmodel.setup.sensitivity_catalog import FIXED_ANCHOR_UNAVAILABLE_CHALLENGES
from catmodel.setup.store import build_setup
from catmodel.simulation.incident import _integrate_local_response
from catmodel.simulation.incident_runtime import state_key
from test_support.contextual_risk import (
    contextual_answer,
    contextual_artifact,
    contextual_instrument,
    fixture_artifact,
    leaves_instrument,
)
from test_support.results import available_metric, reduced_arm


def _capture(tmp_path, instrument=None, artifact=None):
    instrument = instrument or contextual_instrument()
    artifact = artifact or contextual_artifact(instrument)
    composed_setup = compile_contextual_setup(
        instrument,
        artifact,
        setup_id="contextual-fixture",
        years_per_state=48,
        seed=35,
    )
    study = composed_setup
    plan = composed_setup.execution
    package = build_setup(
        destination=tmp_path / "package",
        setup=composed_setup,
        code_commit="fixture",
    )
    report = run_setup(
        setup_path=package.root,
        output=tmp_path / "run",
        code_commit="fixture",
    )
    return study, plan, report, ConditionalCapture(package, tmp_path / "run/capture")


def _changed_artifact(instrument, transform):
    """The fixture capture with each case's direct answer transformed; chain
    nodes get chains matched to the new answers."""
    payload = contextual_artifact(instrument).model_dump(mode="json")
    answers = {}
    for case in payload["cases"]:
        transform(case)
        answers[
            case["node_id"], case["anchor_score"], case["respondent"]["respondent_id"]
        ] = case["answer"]
    return fixture_artifact(
        instrument,
        lambda prompt: ContextualRiskAnswer.model_validate(
            answers[
                prompt.node.node_id,
                prompt.anchor_score,
                prompt.respondent.respondent_id,
            ]
        ),
    )


def test_generated_native_study_keeps_exact_reference_and_dense_zero_prior_grid(
    tmp_path,
):
    instrument = contextual_instrument()

    def scale_rate(case):
        value = {0: 1, 0.5: 2, 2: 8}[case["anchor_score"]]
        case["answer"]["rate"] = dict.fromkeys(
            ("p5", "p25", "p50", "p75", "p95"), value
        )

    artifact = _changed_artifact(instrument, scale_rate)
    study, plan, report, capture = _capture(tmp_path, instrument, artifact)
    assert plan.required_arms == ("reference",)
    # No alternative arm executes; the unavailable register challenges keep
    # comparison coverage from reading complete and are listed with reasons.
    assert report["coverage"]["comparisons"]["status"] == "partial"
    assert report["coverage"]["comparisons"]["required_alternative_arms"] == []
    assert not report["coverage"]["comparisons"]["requirements_complete"]
    assert [row["family_id"] for row in report["comparison_requirements"]] == list(
        FIXED_ANCHOR_UNAVAILABLE_CHALLENGES
    )
    assert all(
        row["status"] == "unavailable" and row["reason"]
        for row in report["comparison_requirements"]
    )
    challenges = {
        row["family_id"]: row
        for row in report["coverage"]["sensitivity_families"]
        if row["kind"] == "register_challenge"
    }
    assert set(challenges) == set(FIXED_ANCHOR_UNAVAILABLE_CHALLENGES)
    assert challenges["OBS-CAUSAL"]["status"] == "unavailable"
    assert report["status"] == "partial"
    assert not report["rankings_available"]
    assert len(study.joint_states) == 4
    assert sum(state.weight for state in study.joint_states) == pytest.approx(1)
    assert {
        state.entity_paths[0].knots[0].value
        for state in study.joint_states
        if state.weight > 0
    } == {0.5}
    assert set(plan.required_state_ids_by_arm["reference"]) == {
        state.state_id for state in study.joint_states
    }
    assert report["arms"]["reference"]["metrics"]["welfare"][
        "expected_annual_loss"
    ] == pytest.approx(200)
    kernel = capture.kernel_at(instrument.context_id, 1, "reference")
    assert kernel.metrics("welfare")["expected_annual_loss"] == pytest.approx(400)
    # Each USD100-welfare episode costs the lab its USD20 fine, a provider-borne
    # component, and it pays no compensation; a year with an episode adds USD70
    # of business loss once.
    assert kernel.metrics("lab_loss::lab")["expected_annual_loss"] == pytest.approx(
        20 * 4 + 70 * -math.expm1(-4)
    )
    societal, weights = kernel.samples("welfare")
    lab, lab_weights = kernel.samples("lab_loss::lab")
    np.testing.assert_allclose(lab, societal * 0.2 + 70 * (societal > 0))
    np.testing.assert_array_equal(weights, lab_weights)
    assert {row.entity_id for row in study.scope.entities} == {
        "lab",
        "model",
        "deployment",
    }


def test_integer_reference_reuses_states_without_double_counting():
    payload = contextual_instrument().model_dump(mode="json")
    payload.update(reference_eci=1, anchor_scores=[0, 1, 2])
    instrument = ContextualRiskInstrument.model_validate(payload)
    composed_setup = compile_contextual_setup(
        instrument,
        contextual_artifact(instrument),
        setup_id="integer",
    )
    study = composed_setup
    assert len(study.joint_states) == 3
    assert len([state for state in study.joint_states if state.weight]) == 1


def test_anchor_count_laws_interpolate_and_stay_separate_from_process_loss():
    instrument = contextual_instrument()

    def uncertain_rate(case):
        factor = 1 if case["anchor_score"] < 2 else 3
        case["answer"]["rate"] = {
            key: factor * value
            for key, value in zip(
                ("p5", "p25", "p50", "p75", "p95"), (1, 2, 4, 8, 12), strict=True
            )
        }

    artifact = _changed_artifact(instrument, uncertain_rate)
    composed_setup = compile_contextual_setup(
        instrument,
        artifact,
        setup_id="ranks",
    )
    study = composed_setup
    by_group = {}
    for state in study.joint_states:
        by_group.setdefault(state.epistemic_group_id, {})[
            state.entity_paths[0].knots[0].value
        ] = state.laws[0]
    for laws in by_group.values():
        assert laws[2].reference_annual_rate == pytest.approx(
            3 * laws[0].reference_annual_rate
        )
        assert laws[1].reference_annual_rate == pytest.approx(
            5 / 3 * laws[0.5].reference_annual_rate
        )
        assert laws[1].segments[0].kernel.member_outcomes[0].components[0].amount == 100


@pytest.mark.parametrize("field", ["lab_compensation", "lab_annual_business_loss"])
def test_unavailable_lab_answer_retains_welfare(tmp_path, field):
    instrument = contextual_instrument()

    def unknown_lab(case):
        case["answer"][field] = {
            "status": "unavailable",
            "unavailable_reason": "No lab-loss judgment.",
        }

    artifact = _changed_artifact(instrument, unknown_lab)
    _, _, report, capture = _capture(tmp_path, instrument, artifact)
    metrics = report["arms"]["reference"]["metrics"]
    assert metrics["welfare"]["expected_annual_loss"] == pytest.approx(100)
    assert metrics["lab_loss::lab"]["status"] == "unavailable"
    with pytest.raises(ValueError, match="societal or lab loss is missing"):
        capture.kernel_at(instrument.context_id, 1, "reference")


def test_a_respondent_left_out_of_the_lab_loss_and_gross_views_keeps_its_welfare(
    tmp_path,
):
    """A fixed-anchor setup of two models with two samples each. One sample's
    direct answers declare the lab's annual business loss unavailable, so
    the respondent is left out of the views that count the lab's loss, the
    lab-loss view and the gross views, at every coordinate: its model's other
    sample carries the model's half there, and the welfare view keeps all
    four at a quarter each. A kernel that weighs the respondent holds the
    welfare views alone."""
    instrument = leaves_instrument(("leaf_a",), {"a": 2, "b": 2})
    scale = {"a-0": 1, "a-1": 5, "b-0": 2, "b-1": 3}

    def answer_for(prompt):
        respondent = prompt.respondent.respondent_id
        payload = contextual_answer().model_dump(mode="json")
        payload["branches"][0]["components"][0]["amount"] *= scale[respondent]
        payload["direct_eal"] = {
            level: value * scale[respondent]
            for level, value in payload["direct_eal"].items()
        }
        payload["lab_annual_business_loss"]["amount"] *= scale[respondent] ** 2
        if respondent == "a-1":
            payload["lab_annual_business_loss"] = {
                "status": "unavailable",
                "unavailable_reason": "No lab-loss judgment.",
            }
        return ContextualRiskAnswer.model_validate(payload)

    setup = compile_contextual_setup(
        instrument,
        fixture_artifact(instrument, answer_for),
        setup_id="left-out",
        years_per_state=48,
        seed=35,
        # One of model a's two samples.
        max_excluded_sample_share=0.5,
    )
    left_out = [state for state in setup.joint_states if state.lab_loss_left_out]
    assert {state.epistemic_group_id for state in left_out} == {"respondent-1"}
    assert len(left_out) == len(setup.joint_states) // 4
    weighted = {
        state.epistemic_group_id: state for state in setup.joint_states if state.weight
    }
    assert {group: state.weight for group, state in weighted.items()} == {
        f"respondent-{index}": 0.25 for index in range(4)
    }
    package = build_setup(
        destination=tmp_path / "package", setup=setup, code_commit="fixture"
    )
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    arm = reduced_arm(report["arms"]["reference"])
    draws, _ = load_run_capture(tmp_path / "run/capture")
    eal = {
        group: {
            view: found.native_mean.value
            for view, found in draws["reference"][
                state_key(state.state_id)
            ].views.items()
        }
        for group, state in weighted.items()
    }
    # Societal harm: 100 a year times each sample's scale, a quarter each.
    assert available_metric(arm["metrics"]["welfare"])[
        "expected_annual_loss"
    ] == pytest.approx(100 * (1 + 5 + 2 + 3) / 4)
    # The lab's loss and gross, which adds the lab's annual business loss:
    # a's first sample at a half, b's two at a quarter each. The left-out
    # state's own gross, which holds no business loss, enters neither.
    assert "lab_loss::lab" not in eal["respondent-1"]
    assert eal["respondent-1"]["gross"] == pytest.approx(5 * 100 + 20)
    for view in ("lab_loss::lab", "gross", "scenario::leaf_a::gross"):
        assert available_metric(arm["metrics"][view])[
            "expected_annual_loss"
        ] == pytest.approx(
            0.5 * eal["respondent-0"][view]
            + 0.25 * eal["respondent-2"][view]
            + 0.25 * eal["respondent-3"][view]
        ), view
    assert {
        weight
        for key, weight in arm["lab_loss_state_weights"].items()
        if key == state_key(weighted["respondent-1"].state_id)
    } == {0.0}
    assert sorted(arm["lab_loss_state_weights"].values())[-3:] == pytest.approx(
        [0.25, 0.25, 0.5]
    )
    # The kernel at the reference coordinate weighs the respondent, so it
    # holds the welfare views at the authored weights and no view that counts
    # the lab's loss.
    capture = ConditionalCapture(package, tmp_path / "run/capture")
    kernel = capture.kernel_at(instrument.context_id, 1, "reference")
    assert kernel.metrics("welfare")["expected_annual_loss"] == pytest.approx(275)
    assert {view for draw in kernel.draws.values() for view in draw.views} == {
        "welfare",
        "scenario::leaf_a::welfare",
    }
    reason = capture.lab_loss_unavailable_reason(instrument.context_id, 1, "reference")
    assert reason is not None
    assert "left out of the views that count the lab's own loss" in reason
    assert reason.endswith("neither the lab-loss views nor the gross views.")
    assert ":respondent-1'," in reason and ":respondent-0" not in reason
    (context,) = _conditioning(package, tmp_path / "run/capture")["contexts"]
    assert context["status"] == "supported" and context["failures"] == []
    assert {row["eci"] for row in context["lab_loss_unavailable"]} == {
        kernel.eci for kernel in setup.conditional_contexts[0].kernels
    }


@pytest.mark.parametrize("lab_status", ["zero", "positive", "unavailable"])
def test_business_loss_is_independent_of_zero_societal_components(tmp_path, lab_status):
    instrument = contextual_instrument()

    def zero_amounts(case):
        for component in case["answer"]["branches"][0]["components"]:
            component["amount"] = 0
        if lab_status == "zero":
            case["answer"]["lab_annual_business_loss"] = {
                "status": "available",
                "positive_probability": 0,
            }
        elif lab_status == "unavailable":
            case["answer"]["lab_annual_business_loss"] = {
                "status": "unavailable",
                "unavailable_reason": "Unknown aggregate",
            }

    _, _, report, _ = _capture(
        tmp_path, instrument, _changed_artifact(instrument, zero_amounts)
    )
    metrics = report["arms"]["reference"]["metrics"]
    assert metrics["welfare"]["expected_annual_loss"] == 0
    if lab_status == "zero":
        assert metrics["lab_loss::lab"]["expected_annual_loss"] == 0
    elif lab_status == "positive":
        # USD 70 once in a year with an incident, at a constant count of one.
        assert metrics["lab_loss::lab"]["expected_annual_loss"] == pytest.approx(
            70 * -math.expm1(-1)
        )
    else:
        assert metrics["lab_loss::lab"]["status"] == "unavailable"
        assert "Unknown aggregate" in " ".join(
            metrics["lab_loss::lab"]["state_reasons"].values()
        )


def test_direct_lab_amount_and_positive_probability_do_not_attenuate_societal_loss(
    tmp_path,
):
    instrument = contextual_instrument()

    def aggregate(case):
        case["answer"]["lab_annual_business_loss"] = {
            "status": "available",
            "positive_probability": 0.25,
            "amount": 1000,
        }

    _, _, report, _ = _capture(
        tmp_path, instrument, _changed_artifact(instrument, aggregate)
    )
    metrics = report["arms"]["reference"]["metrics"]
    assert metrics["welfare"]["expected_annual_loss"] == pytest.approx(100)
    # Gross counts the same business loss as the lab view; welfare does not.
    assert metrics["gross"]["expected_annual_loss"] == pytest.approx(
        120 + 250 * -math.expm1(-1)
    )
    # The provider-borne USD20 fine plus 0.25 x USD1000 of business loss in a
    # year with an incident, at a constant count of one.
    assert metrics["lab_loss::lab"]["expected_annual_loss"] == pytest.approx(
        20 + 250 * -math.expm1(-1)
    )


def test_zero_lab_anchor_is_usable_and_does_not_identify_an_interior_positive_law():
    instrument = contextual_instrument()
    payload = contextual_answer().model_dump(mode="json")
    payload["lab_annual_business_loss"] = {
        "status": "available",
        "positive_probability": 0,
    }
    zero = ContextualRiskAnswer.model_validate(payload)
    positive = contextual_answer()
    answers: dict[float, ContextualRiskAnswer] = {0: zero, 2: positive}
    node = instrument.nodes[0]
    low = compile_contextual_node(
        instrument,
        node,
        answers,
        0,
    )
    high = compile_contextual_node(
        instrument,
        node,
        answers,
        2,
    )
    middle = compile_contextual_node(
        instrument,
        node,
        answers,
        1,
    )
    assert not isinstance(low, PopulationAbsence)
    assert not isinstance(high, PopulationAbsence)
    assert not isinstance(middle, PopulationAbsence)
    assert low.annual_lab_losses == ()
    assert high.annual_lab_losses[0].amount == 70
    assert not middle.segments[0].kernel.campaign_lab_losses
    assert not middle.annual_lab_losses
    assert "both bounding anchors" in middle.unavailable_lab_losses["lab"]
    assert middle.segments[0].kernel.member_outcomes[0].components[0].amount == 100


def test_heavy_stated_lab_p999_is_a_diagnostic_and_keeps_lab_mean_finite(tmp_path):
    instrument = contextual_instrument()
    quantiles = (1, 2, 3, 4, 5, 10, 1000)

    def heavy_lab(case):
        case["answer"]["lab_annual_business_loss"] = {
            "status": "available",
            "positive_probability": 1,
            "loss": dict(
                zip(
                    ("p5", "p25", "p50", "p75", "p95", "p99", "p999"),
                    quantiles,
                    strict=True,
                )
            ),
        }

    artifact = _changed_artifact(instrument, heavy_lab)
    implied = p999_implied_gpd_shape(*quantiles[4:])
    assert implied > 1
    diagnostics = contextual_tail_diagnostics(artifact, instrument)
    lab_rows = [
        row for row in diagnostics if row.observable == "lab_annual_business_loss"
    ]
    assert len(lab_rows) == len(artifact.cases)
    assert all(
        row.p999_implied_gpd_shape == pytest.approx(implied)
        and row.governed_gpd_shape == 0.25
        for row in lab_rows
    )
    (limitation,) = (
        line
        for line in contextual_tail_diagnostic_limitations(((artifact, instrument),))
        if "lab_annual_business_loss" in line
    )
    assert "governed shape 0.25 sets the law" in limitation
    assert "across" not in limitation
    other_payload = instrument.model_dump(mode="json")
    other_payload["context_id"] = "synthetic-lab-other"
    other = ContextualRiskInstrument.model_validate(other_payload)
    (pooled,) = (
        line
        for line in contextual_tail_diagnostic_limitations(
            ((artifact, instrument), (_changed_artifact(other, heavy_lab), other))
        )
        if "lab_annual_business_loss" in line
    )
    assert f"{2 * len(lab_rows)} captured" in pooled
    assert "across 2 contexts" in pooled
    study, _, report, _ = _capture(tmp_path, instrument, artifact)
    law = study.joint_states[0].laws[0]
    severity = law.annual_lab_losses[0].severity
    assert severity is not None
    assert severity.components[0].gpd_shape == 0.25
    diagnostic = law.conditioning_context["lab_annual_business_loss_tail_diagnostic"]
    assert diagnostic["governed_gpd_shape"] == 0.25
    assert diagnostic["interpolated_p999_implied_gpd_shape"] == pytest.approx(implied)
    metrics = report["arms"]["reference"]["metrics"]
    assert metrics["welfare"]["expected_annual_loss"] == pytest.approx(100)
    assert math.isfinite(metrics["lab_loss::lab"]["expected_annual_loss"])


def test_unavailable_anchor_is_not_interpolated_through():
    instrument = contextual_instrument()

    def unknown_high(case):
        if case["anchor_score"] == 2:
            case["answer"] = {
                "status": "unavailable",
                "unavailable_reason": "High-score mechanism has not been characterized.",
                "reasoning": case["answer"]["reasoning"],
            }

    with pytest.raises(
        ContextualLawUnavailable, match=r"ECI 1.*not been characterized"
    ):
        compile_contextual_setup(
            instrument,
            _changed_artifact(instrument, unknown_high),
            setup_id="missing",
        )


def test_authored_unavailable_node_remains_visible_without_hiding_known_node(tmp_path):
    payload = contextual_instrument().model_dump(mode="json")
    payload["independent_node_origins_rationale"] = (
        "Synthetic disjoint initiating incidents and unique cross-node amounts with independent node counts."
    )
    unknown = deepcopy(payload["nodes"][0])
    unknown.update(
        node_id="unmodeled",
        count_family=None,
        branches=[],
        components=[],
        unavailable_reason="No qualifying incident-count law.",
    )
    unknown["population"]["scenario_id"] = "unmodeled"
    payload["nodes"].append(unknown)
    instrument = ContextualRiskInstrument.model_validate(payload)
    study, _, report, _ = _capture(tmp_path, instrument)
    assert len(study.scope.populations) == 2
    assert all(
        state.absent_populations[0].scenario_id == "unmodeled"
        for state in study.joint_states
    )
    metrics = report["arms"]["reference"]["metrics"]
    assert metrics["welfare"]["status"] == "unavailable"
    assert metrics["scenario::scenario::welfare"][
        "expected_annual_loss"
    ] == pytest.approx(100)
    assert metrics["scenario::unmodeled::welfare"]["status"] == "unavailable"


def test_unknown_component_preserves_node_unavailability():
    instrument = contextual_instrument()

    def unknown_component(case):
        case["answer"]["branches"][0]["components"][0] = {
            "component_id": "victim",
            "status": "unavailable",
            "unavailable_reason": "No loss law evidence.",
        }

    with pytest.raises(ContextualLawUnavailable, match="No loss law evidence"):
        compile_contextual_setup(
            instrument,
            _changed_artifact(instrument, unknown_component),
            setup_id="unknown",
        )


def test_exact_zero_count_retains_structural_zero_even_without_amount_evidence(
    tmp_path,
):
    instrument = contextual_instrument()

    def zero(case):
        case["answer"]["rate"] = dict.fromkeys(("p5", "p25", "p50", "p75", "p95"), 0)
        case["answer"]["lab_annual_business_loss"] = {
            "status": "unavailable",
            "unavailable_reason": "Unused lab loss missing.",
        }
        case["answer"]["branches"][0]["components"][0] = {
            "component_id": "victim",
            "status": "unavailable",
            "unavailable_reason": "Unused severity missing.",
        }

    study, _, report, capture = _capture(
        tmp_path, instrument, _changed_artifact(instrument, zero)
    )
    assert all(
        state.absent_populations[0].disposition == "structural_zero"
        for state in study.joint_states
    )
    assert (
        report["arms"]["reference"]["metrics"]["welfare"]["expected_annual_loss"] == 0
    )
    assert (
        capture.kernel_at(instrument.context_id, 1, "reference").metrics(
            "lab_loss::lab"
        )["expected_annual_loss"]
        == 0
    )


def test_native_severity_retains_governed_tail_and_rank_group_but_not_p999_authority():
    instrument = contextual_instrument()
    payload = contextual_answer().model_dump(mode="json")
    component = payload["branches"][0]["components"][0]
    component.update(
        amount=None,
        loss=dict(
            zip(
                ("p5", "p25", "p50", "p75", "p95", "p99", "p999"),
                (20, 40, 80, 120, 200, 600, 1000),
                strict=True,
            )
        ),
    )
    answer = ContextualRiskAnswer.model_validate(payload)
    node = instrument.nodes[0]
    first = compile_contextual_node(
        instrument,
        node,
        {0: answer},
        0,
    )
    assert not isinstance(first, PopulationAbsence)
    component["loss"]["p999"] = 2000
    second = compile_contextual_node(
        instrument,
        node,
        {0: ContextualRiskAnswer.model_validate(payload)},
        0,
    )
    assert first == second
    assert (
        first.segments[0].kernel.member_outcomes[0].components[0].rank_group_id
        == "severity"
    )


def test_native_qualification_rejection_marks_only_that_node_unavailable():
    payload = contextual_instrument().model_dump(mode="json")
    payload["nodes"][0]["population"].update(threshold=200, threshold_comparator="gt")
    instrument = ContextualRiskInstrument.model_validate(payload)
    with pytest.raises(ContextualLawUnavailable, match="contradicts qualification"):
        compile_contextual_setup(
            instrument,
            contextual_artifact(instrument),
            setup_id="threshold",
        )


def test_tampered_or_incomplete_artifact_is_rejected_before_compilation():
    instrument = contextual_instrument()
    payload = contextual_artifact(instrument).model_dump(mode="json")
    payload["cases"].pop()
    with pytest.raises(ValueError, match="every collectible"):
        compile_contextual_setup(
            instrument,
            ContextualRiskArtifact.model_validate(payload),
            setup_id="incomplete",
        )


@pytest.mark.parametrize("tamper", ["prompt", "citation"])
def test_direct_compiler_validates_prompt_and_citation_binding(tamper):
    instrument = contextual_instrument()
    payload = contextual_artifact(instrument).model_dump(mode="json")
    if tamper == "prompt":
        payload["cases"][0]["prompt_sha256"] = "f" * 64
    else:
        payload["cases"][0]["answer"]["reasoning"]["evidence_used"] = ["invented-card"]
    with pytest.raises(ValueError):
        compile_contextual_setup(
            instrument,
            ContextualRiskArtifact.model_validate(payload),
            setup_id="tampered",
        )


def test_capture_retains_all_zero_prior_states(tmp_path):
    study, _, _, capture = _capture(tmp_path)
    _, record = load_run_capture(capture.capture_directory)
    assert len(record.arms["reference"]) == len(study.joint_states)


def test_missing_respondent_is_not_dropped_or_averaged_as_zero():
    payload = contextual_instrument().model_dump(mode="json")
    payload["respondents"].append(
        {"respondent_id": "other", "model_id": "other-model", "sample_index": 0}
    )
    instrument = ContextualRiskInstrument.model_validate(payload)

    def missing(case):
        if case["respondent"]["respondent_id"] == "other":
            case["answer"] = {
                "status": "unavailable",
                "unavailable_reason": "No defensible estimate.",
                "reasoning": case["answer"]["reasoning"],
            }

    with pytest.raises(
        ContextualLawUnavailable, match=r"respondent.*other.*No defensible estimate"
    ):
        compile_contextual_setup(
            instrument,
            _changed_artifact(instrument, missing),
            setup_id="mixed",
        )


def test_partial_respondent_coverage_withholds_only_affected_views(tmp_path):
    payload = contextual_instrument().model_dump(mode="json")
    payload["independent_node_origins_rationale"] = (
        "Synthetic disjoint initiating incidents and unique cross-node amounts with independent node counts."
    )
    payload["respondents"].append(
        {"respondent_id": "other", "model_id": "other-model", "sample_index": 0}
    )
    control = deepcopy(payload["nodes"][0])
    control["node_id"] = "control"
    control["population"]["scenario_id"] = "control"
    payload["nodes"].append(control)
    instrument = ContextualRiskInstrument.model_validate(payload)

    def missing(case):
        if (
            case["node_id"] == "synthetic-node"
            and case["respondent"]["respondent_id"] == "other"
        ):
            case["answer"] = {
                "status": "unavailable",
                "unavailable_reason": "Node A has no estimate from this respondent.",
                "reasoning": case["answer"]["reasoning"],
            }

    _, _, report, _ = _capture(
        tmp_path,
        instrument,
        _changed_artifact(instrument, missing),
    )
    metrics = report["arms"]["reference"]["metrics"]
    assert metrics["welfare"]["status"] == "unavailable"
    assert metrics["scenario::scenario::welfare"]["status"] == "unavailable"
    assert metrics["scenario::control::welfare"][
        "expected_annual_loss"
    ] == pytest.approx(100)


@pytest.mark.parametrize("interpolation", ["step", "linear"])
def test_fixed_score_law_cannot_be_reattached_to_a_moving_path(interpolation):
    instrument = contextual_instrument()
    composed_setup = compile_contextual_setup(
        instrument,
        contextual_artifact(instrument),
        setup_id="fixed",
    )
    study = composed_setup
    state = next(row for row in study.joint_states if row.weight)
    law = state.laws[0]
    assert _integrate_local_response(state.entity_paths[0], law.response, 0, 1) == 1
    path = state.entity_paths[0].model_copy(
        update={
            "interpolation": interpolation,
            "knots": (
                state.entity_paths[0].knots[0],
                state.entity_paths[0]
                .knots[0]
                .model_copy(update={"time": 0.5, "value": 1.5}),
                state.entity_paths[0].knots[-1].model_copy(update={"value": 1.5}),
            ),
        }
    )
    with pytest.raises(ValueError, match="outside its authored coordinate support"):
        _integrate_local_response(path, law.response, 0, 1)


def test_multiple_nodes_need_an_authored_origin_and_amount_partition():
    payload = contextual_instrument().model_dump(mode="json")
    other = deepcopy(payload["nodes"][0])
    other["node_id"] = "overlapping"
    other["population"]["scenario_id"] = "overlapping"
    payload["nodes"].append(other)
    instrument = ContextualRiskInstrument.model_validate(payload)
    with pytest.raises(ContextualLawUnavailable, match="disjoint initiating incidents"):
        compile_contextual_setup(
            instrument,
            contextual_artifact(instrument),
            setup_id="overlap",
        )
