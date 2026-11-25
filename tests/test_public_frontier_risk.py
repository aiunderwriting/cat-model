"""Synthetic whole-trajectory conditions preserve mixtures and source status."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from catmodel.contracts.contextual_risk import (
    ContextualRiskArtifact,
    ContextualRiskInstrument,
)
from catmodel.contracts.elicitation import Quantiles
from catmodel.contracts.public_frontier_condition import PublicTrajectoryCondition
from catmodel.contracts.run_setup import RunSetup
from catmodel.elicitation.contextual_risk import contextual_prompts
from catmodel.results.execution import run_setup
from catmodel.setup.contextual_node import ContextualLawUnavailable
from catmodel.setup.contextual_setup import compile_contextual_setup
from catmodel.setup.public_frontier import (
    PublicFrontierRiskInput,
    public_frontier_setup_fields,
)
from catmodel.setup.store import build_setup
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_moments import (
    analytic_linear_eals,
    at_least_one_probability,
)
from test_support.contextual_risk import (
    contextual_answer,
    contextual_artifact,
    contextual_instrument,
    fixture_artifact,
)

RATIONALES: dict[str, Any] = {
    "exposure_rationale": "Synthetic source-status populations are disjoint, with unique incident costs.",
    "residual_independence_rationale": "Synthetic independent counts conditional on public path and respondent.",
    "parameter_coupling_rationale": "Synthetic matched respondents and independent ranks per source status.",
}


def _inputs():
    result = []
    for draw, score, weight in (("low", 1, 0.25), ("high", 3, 0.75)):
        for status in ("public", "not_yet_public"):
            payload = contextual_instrument().model_dump(mode="json")
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
            payload["nodes"][0]["population"]["deployment_ids"] = [
                f"deployment-{status}"
            ]
            times = [f"2027-{month:02d}-01T00:00:00Z" for month in range(1, 13)] + [
                "2028-01-01T00:00:00Z"
            ]
            payload["public_trajectory"] = {
                "trajectory_sha256": "e" * 64,
                "draw_id": draw,
                "observed_at": "2026-12-01T00:00:00Z",
                "observed_values": [score, 100 + score],
                "earlier_frontier": None,
                "times": times,
                "coordinates": [
                    {
                        "coordinate_id": lab,
                        "lab_id": lab,
                        "domain_id": "general",
                        "access": "public",
                        "scale_id": scale,
                        "definition": definition,
                    }
                    for lab, scale, definition in (
                        ("lab_a", "eci", "Synthetic lab A general ECI"),
                        ("lab_b", "eci", "Synthetic lab B general ECI"),
                    )
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
            rate = score**2 * (1 if status == "public" else 2)
            answer = contextual_answer().model_copy(
                update={
                    "rate": Quantiles(p5=rate, p25=rate, p50=rate, p75=rate, p95=rate)
                }
            )
            result.append(
                PublicFrontierRiskInput(
                    instrument,
                    fixture_artifact(instrument, lambda _prompt, a=answer: a),
                    weight,
                    quadrature_sha256="q" * 64,
                )
            )
    return tuple(result)


def _compile(inputs=None, setup_id="synthetic-public"):
    return RunSetup(
        **public_frontier_setup_fields(
            inputs or _inputs(),
            setup_id=setup_id,
            years_per_state=32,
            seed=4,
            epistemic_tail_scale=1.0,
            **RATIONALES,
        )
    )


def test_prompt_conditions_on_whole_public_path_without_internal_forecast():
    instrument = _inputs()[0].instrument
    text = contextual_prompts(instrument)[0].render()
    internal = "No path is given for the lab's not-yet-public models"
    assert internal not in text
    assert internal in contextual_prompts(_inputs()[1].instrument)[0].render()
    assert "A publicly available model used inside the lab remains public" in text
    assert "including its changes within the year" in text
    assert "date        lab_a  lab_b  frontier" in text
    assert "general capability" in text
    assert "lab_annual_business_loss, defined as: " in text
    assert "lab_compensation, defined as: " in text
    assert instrument.lab_compensation_definition in text
    assert instrument.lab_annual_business_loss_definition in text
    for stale in (
        "Hold the lab's",
        '"anchor_score"',
        '"reference_eci"',
        '"capability_access"',
        "exact supplied ECI held constant",
        "Constant-year answers",
        "lab_loss_fraction",
        "lab_costs_complete",
        "Exact-zero anchors",
    ):
        assert stale not in text
    assert instrument.digest() != _inputs()[2].instrument.digest()
    with pytest.raises(ValueError, match="public-frontier compiler"):
        compile_contextual_setup(
            instrument,
            _inputs()[0].artifact,
            setup_id="invalid",
        )


def test_public_condition_rejects_missing_measurements_or_internal_axes():
    condition = _inputs()[0].instrument.public_trajectory
    assert condition is not None
    for change, message in (
        ("missing", "finite numeric values"),
        ("internal", "distinct public labs"),
        ("domain", "require general ECI lab coordinates"),
        ("declining", "must be nondecreasing"),
        ("target", "target_lab"),
        ("scale", "share one general-ECI scale"),
        ("duplicate", "distinct public labs"),
    ):
        data = condition.model_dump(mode="json")
        if change == "missing":
            data["values"][0][1] = None
        elif change == "internal":
            data["coordinates"][1]["access"] = "internal"
        elif change == "domain":
            data["coordinates"][1]["domain_id"] = "bio"
        elif change == "target":
            data["target_lab"] = "lab_a"
        elif change == "scale":
            data["coordinates"][1]["scale_id"] = "other_eci"
        elif change == "duplicate":
            data["coordinates"][1]["lab_id"] = "lab_a"
        else:
            data["values"][1][0] = data["values"][0][0] - 1
        with pytest.raises(ValidationError, match=message):
            PublicTrajectoryCondition.model_validate(data)


def test_domain_benchmark_evidence_remains_in_general_conditioned_prompt():
    instrument = _inputs()[0].instrument
    payload = instrument.model_dump(mode="json")
    payload["nodes"][0]["source_cards"] = [
        {
            "card_id": "synthetic-cyber-benchmark",
            "observation_through_utc": "2026-09-01",
            "availability_date_utc": "2026-09-02",
            "availability_basis": "Synthetic source date for software validation.",
            "text": "Synthetic cyber benchmark result retained as contextual evidence.",
            "sources": ["https://example.org/cyber-benchmark"],
        },
        {
            "card_id": "synthetic-bci-benchmark",
            "observation_through_utc": "2026-09-01",
            "availability_date_utc": "2026-09-02",
            "availability_basis": "Synthetic source date for software validation.",
            "text": "Synthetic BCI result retained in its own units as contextual evidence.",
            "sources": ["https://example.org/bci-benchmark"],
        },
    ]
    payload["public_trajectory"]["source_references"].append(
        "urn:fixture:cyber-benchmark"
    )
    changed = ContextualRiskInstrument.model_validate(payload)
    prompt = contextual_prompts(changed)[0].render()
    assert "general capability" in prompt and "cyber or biology" in prompt
    assert "Benchmark results in the evidence are context" in prompt
    assert '<excerpt id="synthetic-cyber-benchmark">' in prompt
    assert '<excerpt id="synthetic-bci-benchmark">' in prompt
    assert (
        "Synthetic BCI result retained in its own units as contextual evidence."
        in prompt
    )
    assert "Synthetic cyber benchmark result retained as contextual evidence." in prompt
    # The benchmarks stay evidence: the trajectory has general-ECI labs only.
    assert "date        lab_a  lab_b  frontier" in prompt
    assert '"domain_id": "bio"' not in prompt
    assert changed.digest() != instrument.digest()


def test_risk_mixture_preserves_nonlinearity_and_both_source_populations():
    setup = _compile()
    assert len(setup.joint_states) == 2
    expected = 0
    lab_expected = 0
    occurrence = 0.0
    for state in setup.joint_states:
        assert len(state.laws) == 2
        assert {law.segments[0].source_public_status for law in state.laws} == {
            "public",
            "not_yet_public",
        }
        assert all(
            law.response.domain == "conditional_annual_kernel" for law in state.laws
        )
        assert all(
            len(law.conditioning_context["public_trajectory"]["values"]) == 13
            for law in state.laws
        )
        assert all(
            law.conditioning_context["forecast_integration"]
            == {
                "scope": "trajectory_quadrature_node",
                "continuous_forecast_loss_moments": "not_established",
            }
            for law in state.laws
        )
        expected += (
            state.weight
            * analytic_linear_eals(
                compile_incident_state(setup, state.state_id), ("welfare",)
            ).values["welfare"]
        )
        for incident in compile_incident_state(setup, state.state_id).incidents:
            result = at_least_one_probability(incident)
            assert result is not None
            occurrence += state.weight * result[0]
        lab_expected += (
            state.weight
            * analytic_linear_eals(
                compile_incident_state(setup, state.state_id), ("lab_loss::lab",)
            ).values["lab_loss::lab"]
        )
    assert expected == pytest.approx(2100)
    # USD 20 of fine per incident, plus USD 70 of business loss once in each
    # year with an incident: fewer than one per incident.
    assert 0.2 * expected < lab_expected < 0.9 * expected
    assert lab_expected == pytest.approx(0.2 * expected + 70 * occurrence)
    assert _inputs()[0].instrument.lab_compensation_definition in setup.limitations
    assert (
        _inputs()[0].instrument.lab_annual_business_loss_definition in setup.limitations
    )
    assert any(
        "Lab annual business loss is drawn once per year" in row
        for row in setup.assumptions.independence
    )
    assert expected != pytest.approx(3 * 2.5**2 * 100)
    assert all(
        "not a forecast" in path.history_rationale
        for state in setup.joint_states
        for path in state.entity_paths
    )
    assert any("integrated node mixture" in text for text in setup.limitations)


def test_missing_status_or_unavailable_positive_path_is_not_dropped():
    with pytest.raises(ValueError, match="every public draw"):
        _compile(_inputs()[:-1])
    inputs = list(_inputs())
    data = inputs[-1].artifact.model_dump(mode="json")
    data["cases"][0]["answer"].update(
        status="unavailable",
        rate=None,
        lab_compensation=None,
        lab_annual_business_loss=None,
        branches=[],
        direct_eal=None,
        unavailable_reason="Synthetic missing judgment",
    )
    data["cases"][0].pop("eal_crosscheck", None)
    inputs[-1] = replace(
        inputs[-1], artifact=ContextualRiskArtifact.model_validate(data)
    )
    with pytest.raises(ContextualLawUnavailable, match="Incomplete source-status"):
        _compile(tuple(inputs))


def test_source_status_annual_vectors_add_to_total_even_for_internal_use(tmp_path):
    from catmodel.results.conditional import ConditionalCapture

    setup = _compile()
    package = build_setup(
        destination=tmp_path / "package", setup=setup, code_commit="fixture"
    )
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")
    capture = ConditionalCapture(package, tmp_path / "run/capture")
    for draw in capture.draws["reference"].values():
        for view in ("welfare", "lab_loss::lab"):
            np.testing.assert_allclose(
                draw.views[view].array,
                draw.views[f"source_status::public::{view}"].array
                + draw.views[f"source_status::not_yet_public::{view}"].array,
            )


@pytest.mark.parametrize(
    "field", ["lab_compensation_definition", "lab_annual_business_loss_definition"]
)
def test_public_conditions_share_one_high_level_lab_loss_definition(field):
    inputs = list(_inputs())
    original = inputs[-1]
    data = original.instrument.model_dump(mode="json")
    data[field] = "Different synthetic economic-loss boundary"
    instrument = ContextualRiskInstrument.model_validate(data)
    inputs[-1] = replace(
        original, instrument=instrument, artifact=contextual_artifact(instrument)
    )
    with pytest.raises(ValueError, match="same lab, loss scope"):
        _compile(tuple(inputs))


def test_one_studys_inputs_share_one_trajectory_quadrature():
    # Questions identify only their node's trajectory; the rule, weights
    # included, travels beside them.
    inputs = list(_inputs())
    inputs[-1] = replace(inputs[-1], quadrature_sha256="r" * 64)
    with pytest.raises(ValueError, match="same trajectory quadrature"):
        _compile(tuple(inputs))


def test_frontier_target_includes_exact_leader_switches():
    condition = _inputs()[0].instrument.public_trajectory
    assert condition is not None
    data = condition.model_dump(mode="json")
    data["times"] = ["2027-01-01T00:00:00Z", "2027-01-03T00:00:00Z"]
    data["values"] = [[100, 98], [102, 104]]
    data["observed_values"] = [100, 98]
    frontier = PublicTrajectoryCondition.model_validate(data)
    times, path = frontier.target_path()
    assert [at.day for at in times] == [1, 2, 3]
    assert path == pytest.approx((100, 101, 104))


def test_setup_uses_within_draw_frontier_without_changing_laws():
    setup = _compile()
    assert {
        path.knots[0].value
        for state in setup.joint_states
        for path in state.entity_paths
    } == {101, 103}
    assert all(
        law.conditioning_context["public_trajectory"]["target_lab"] == "public_frontier"
        for state in setup.joint_states
        for law in state.laws
    )
    assert all(
        law.response.points[0].multiplier == 1
        for state in setup.joint_states
        for law in state.laws
    )


def test_imports_differing_only_in_setup_id_draw_identical_reference_losses(tmp_path):
    from catmodel.results.conditional import ConditionalCapture

    draws = []
    for setup_id in ("first-import", "second-import"):
        package = build_setup(
            destination=tmp_path / setup_id,
            setup=_compile(setup_id=setup_id),
            code_commit="fixture",
        )
        run_setup(
            setup_path=package.root,
            output=tmp_path / f"{setup_id}-run",
            code_commit="fixture",
        )
        draws.append(
            ConditionalCapture(package, tmp_path / f"{setup_id}-run/capture").draws[
                "reference"
            ]
        )
    first, second = draws
    assert set(first) == set(second)
    assert any(np.any(draw.views["welfare"].array > 0) for draw in first.values())
    for state, draw in first.items():
        assert set(draw.views) == set(second[state].views)
        for view, values in draw.views.items():
            np.testing.assert_array_equal(values.array, second[state].views[view].array)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"observed_at": "2027-06-01T00:00:00Z"}, "cannot follow the path's start"),
        ({"observed_values": [1]}, "start at or above its observed score"),
        ({"observed_values": [1e6, 1e6]}, "start at or above its observed score"),
        (
            {"observed_at": "2027-01-01T00:00:00Z", "observed_values": [0, 0]},
            "must start at the observed scores",
        ),
        (
            {"earlier_frontier": {"at": "2027-01-01T00:00:00Z", "value": 0}},
            "must precede the observed scores",
        ),
        (
            {"earlier_frontier": {"at": "2025-12-01T00:00:00Z", "value": 1e6}},
            "must precede the observed scores",
        ),
    ],
)
def test_a_trajectory_condition_refuses_observations_its_path_contradicts(
    change, message
):
    """The observed scores are the reference a question gives for the scale:
    they precede the path, each lab's path starts at or above its own (at it,
    when the path starts on the observed date), and the earlier frontier lies
    before and at or below them."""
    condition = _inputs()[0].instrument.public_trajectory
    assert condition is not None
    data = condition.model_dump(mode="json") | change
    with pytest.raises(ValueError, match=message):
        PublicTrajectoryCondition.model_validate(data)
