"""Small-event aggregation: exact large cells, exact body moments, caps, budget."""

import json
import math
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from scipy import integrate
from scipy.special import ndtri

from catmodel.contracts.components import AggregateLabLossLaw, ComponentAmountLaw
from catmodel.contracts.elicitation import Quantiles
from catmodel.contracts.incident import (
    ClusterCount,
    CountExperiment,
    ExecutionPlan,
    JointMarkOutcome,
    MarkKernel,
    ParameterRankLaw,
    UnitCountMass,
)
from catmodel.contracts.native_severity import native_severity_from_sampler
from catmodel.elicitation.epistemic import epistemic_mean
from catmodel.hashing import canonical_json_sha256
from catmodel.probability.distributions import Distribution, MixtureSeverityDistribution
from catmodel.results.execution import replay_capture, run_setup
from catmodel.setup.rank_spread import REFERENCE_SPREAD_TAIL_FORM
from catmodel.setup.store import build_setup
from catmodel.simulation import incident_runtime as runtime
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_linear import LinearProjection, single_member_law
from catmodel.simulation.small_events import (
    SIMULATED_SCORE_RANGE,
    SMALL_EVENT_LIMITATION,
    SmallEventControls,
    SmallEventLaw,
    body_variance,
    fixed_rate_buckets,
    small_event_results,
    tvar_envelope,
)
from test_support.count_law_oracle import CountLawOracle
from test_support.run_setup import setup_from_payload, with_execution
from test_support.study import study_payload

GROSS = Distribution(
    1.05e5, 1.3e5, 2e5, 4.5e5, 2.5e6, gpd_shape=0.7, severity_minimum=1e5
)
ANCHORED = Distribution(
    1.05e5,
    1.3e5,
    2e5,
    4.5e5,
    2.5e6,
    gpd_shape=0.7,
    severity_minimum=1e5,
    large_loss_amount=1e9,
    large_loss_exceedance=1e-4,
)
AGGREGATE_LAB = MixtureSeverityDistribution(
    [
        Distribution(0, 0, 0, 0, 0),
        Distribution(5e3, 2e4, 5e4, 1.5e5, 1e6, gpd_shape=0.7),
    ],
    np.array([0.9, 0.1]),
)
SHARES = {"third_party": 0.8, "provider": 0.15, "transfer": 0.05}
VIEWS = ("gross", "oep_gross", "welfare", "lab_loss::lab", "scenario::scenario::gross")


def _kernel(
    *, severity=GROSS, cap=None, lab_cap=1e12, member=1.0, scale=1.0, shares=None
):
    law = native_severity_from_sampler(
        severity if cap is None else replace(severity, severity_cap=cap)
    )
    support = [UnitCountMass(count=1, probability=member)]
    if member < 1:
        support.append(UnitCountMass(count=0, probability=1 - member))
    return MarkKernel(
        cluster=ClusterCount(family="finite", support=tuple(support)),
        shared_probability=1,
        unit_probability=1,
        gate_meaning="One qualified incident",
        selection_assumption="One member",
        displacement_assumption="No displacement",
        campaign_lab_losses=(
            AggregateLabLossLaw(
                lab_id="lab",
                severity=native_severity_from_sampler(AGGREGATE_LAB),
                episode_cap=lab_cap,
            ),
        ),
        member_outcomes=(
            JointMarkOutcome(
                probability=1,
                components=tuple(
                    ComponentAmountLaw(
                        component_id=name,
                        nature="transfer" if name == "transfer" else "resource_cost",
                        lab_loss_weights={"lab": 1.0} if name == "provider" else {},
                        scenario_ids=("scenario",),
                        severity=law,
                        rank_group_id="total",
                        post_multiplier=share * scale,
                    )
                    for name, share in (shares or SHARES).items()
                ),
            ),
        ),
    )


def _state(rate, **kernel):
    payload = study_payload()
    payload["entities"].append(
        {
            "entity_id": "lab",
            "kind": "lab",
            "definition": "Synthetic lab",
            "display_name": "lab",
            "status": "hypothetical",
        }
    )
    payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"] = _kernel(
        **kernel
    ).model_dump(mode="json")
    compiled = compile_incident_state(setup_from_payload(payload), "state")
    return replace(
        compiled,
        incidents=tuple(
            replace(incident, annual_rate=rate) for incident in compiled.incidents
        ),
    )


def _plan(years, **controls):
    return ExecutionPlan(
        plan_id="small-events",
        required_arms=("reference",),
        simulations=years,
        seed=7,
        requested_views=VIEWS,
        numerical_controls=controls,
    )


def _run(compiled, years, **controls):
    draw, reasons, allocation = runtime.simulate_incident_state(
        _plan(years, **controls), compiled, response_sha256="a" * 64
    )
    assert reasons == {}
    return draw, allocation


def _law(**kernel):
    law = single_member_law(_kernel(**kernel))
    assert law is not None
    return law


def _midpoint_moments(sampler, upper, scale, points=2_000_000):
    ranks = (np.arange(points) + 0.5) / points * upper
    values = scale * sampler.sample_from_uniforms_uncapped(ranks)
    return [float(np.mean(values**power)) for power in (1, 2)]


@pytest.mark.parametrize("severity", [GROSS, ANCHORED], ids=["one-piece", "two-piece"])
def test_split_hits_the_target_and_its_moments_are_exact(severity):
    law = _law(severity=severity)
    controls = SmallEventControls()
    split = SmallEventLaw(law, controls).split(50_000.0)
    assert not isinstance(split, str)
    gross_cdf = float(severity.cdf(np.array([split.split_point]), apply_cap=False)[0])
    assert split.rank_gross == pytest.approx(gross_cdf, rel=1e-12)
    lab_cdf = float(
        AGGREGATE_LAB.cdf(np.array([split.lab_split_point]), apply_cap=False)[0]
    )
    assert split.rank_lab == pytest.approx(lab_cdf, rel=1e-12)
    expected_large = 50_000 * (
        split.large_gross_probability + split.large_lab_probability
    )
    # Each mark targets half the individually drawn events; the skewness bound
    # can only lower a split, which adds individually drawn events.
    assert expected_large >= controls.target * (1 - 1e-6)
    assert expected_large < 4 * controls.target
    assert split.minimum_count >= 100
    expected = 50_000 * split.small_probability
    assert expected - 5 * math.sqrt(expected) >= split.minimum_count
    # The split lies above p99, so the p99 break and the anchor are inside
    # the integrated ranks for the anchored law.
    assert split.rank_gross > 0.99
    mean, second = _midpoint_moments(severity, split.rank_gross, 1.0)
    assert split.gross_moments[0] == pytest.approx(mean, rel=1e-6)
    assert split.gross_moments[1] == pytest.approx(second, rel=1e-5)
    lab_mean, lab_second = _midpoint_moments(AGGREGATE_LAB, split.rank_lab, 1.0)
    assert split.lab_moments[0] == pytest.approx(lab_mean, rel=1e-5)
    assert split.lab_moments[1] == pytest.approx(lab_second, rel=1e-4)


def test_both_marks_share_the_individually_drawn_target():
    """Where the skewness bound does not bind, the gross and aggregate-lab splits
    each take half the target, so an origin-year draws about kappa events."""
    split = SmallEventLaw(_law(), SmallEventControls()).split(1e7)
    assert not isinstance(split, str)
    large = 1e7 * (split.large_gross_probability + split.large_lab_probability)
    assert large == pytest.approx(SmallEventControls().target, rel=0.01)
    assert 1e7 * split.large_gross_probability == pytest.approx(8, rel=0.01)


def test_a_heavy_aggregate_lab_law_can_keep_only_its_zero_atom_small():
    """Where no positive aggregate-lab split meets the skewness bound, every event
    with positive aggregate lab loss is drawn and the small events' aggregate-lab sum
    is exactly zero."""
    split = SmallEventLaw(_law(), SmallEventControls()).split(1_000.0)
    assert not isinstance(split, str)
    assert split.lab_split_point == 0.0
    assert split.lab_moments == (0.0, 0.0, 0.0)
    assert split.rank_lab == pytest.approx(0.9)
    assert split.large_lab_probability == pytest.approx(
        (1 - split.large_gross_probability) * 0.1
    )


def test_gate_moments_count_gated_out_events_as_zero():
    law = _law(member=0.6)
    split = SmallEventLaw(law, SmallEventControls()).split(50_000.0)
    assert not isinstance(split, str)
    member = 0.6 * split.rank_gross / (0.4 + 0.6 * split.rank_gross)
    assert split.small_member_probability == pytest.approx(member)
    mean, _ = _midpoint_moments(GROSS, split.rank_gross, 1.0)
    assert split.gross_moments[0] == pytest.approx(member * mean, rel=1e-6)
    assert split.large_gross_probability == pytest.approx(0.6 * (1 - split.rank_gross))


def test_caps_bound_the_split_and_a_cap_at_the_floor_refuses():
    free = SmallEventLaw(_law(), SmallEventControls()).split(50_000.0)
    assert not isinstance(free, str) and free.split_point > 3e6
    capped = SmallEventLaw(_law(cap=3e6), SmallEventControls()).split(50_000.0)
    assert not isinstance(capped, str) and capped.split_point < 3e6
    lab_capped = SmallEventLaw(_law(lab_cap=2e6), SmallEventControls()).split(50_000.0)
    # A small event's lab view is below 0.15 * H + H_L, within the cap.
    assert not isinstance(lab_capped, str)
    assert 0.15 * lab_capped.split_point + lab_capped.lab_split_point < 2e6
    # Half the cap to the gross mark: 0.15 * H <= 2e4 / 2 leaves H below the floor.
    refused = SmallEventLaw(_law(lab_cap=2e4), SmallEventControls()).split(50_000.0)
    assert refused == "A cap lies at or below the smallest admitted loss."
    assert (
        SmallEventLaw(_law(), SmallEventControls()).split(16.0)
        == "Its expected count does not exceed the individually drawn target."
    )


def test_ineligible_kernels_are_drawn_event_by_event():
    payload = _kernel().model_dump(mode="json")
    components = payload["member_outcomes"][0]["components"]
    components[1]["rank_group_id"] = "independent"
    law = single_member_law(MarkKernel.model_validate(payload))
    assert law is not None
    reason = SmallEventLaw(law, SmallEventControls()).split(1e5)
    assert isinstance(reason, str) and "one severity draw" in reason
    components[1]["rank_group_id"] = "total"
    components[1].update(amount=5.0, severity=None, rank_group_id=None)
    law = single_member_law(MarkKernel.model_validate(payload))
    assert law is not None
    reason = SmallEventLaw(law, SmallEventControls()).split(1e5)
    assert isinstance(reason, str) and "one severity draw" in reason


def test_low_rates_keep_todays_event_by_event_path_bit_for_bit():
    compiled = _state(12.0)
    exact, _ = _run(compiled, 64, small_events={"target_large_events_per_year": 1e9})
    default, default_allocation = _run(compiled, 64)
    for view in VIEWS:
        np.testing.assert_array_equal(
            default.views[view].array, exact.views[view].array
        )
    record = default_allocation.small_events["by_origin"]["origin"]
    assert record["status"] == "event_by_event"
    assert default_allocation.summed_small_initiators == 0


@pytest.mark.parametrize("lab_cap", [1e12, 4e6], ids=["free", "lab-capped"])
def test_summed_and_individual_bodies_share_large_events_and_two_moments(lab_cap):
    compiled = _state(4_000.0, lab_cap=lab_cap, member=0.9)
    years = 1_500
    summed, summed_allocation = _run(
        compiled, years, max_vectorized_events_per_year=10**5
    )
    individual, individual_allocation = _run(
        compiled, years, small_events={"body": "individual"}
    )
    summed_record = summed_allocation.small_events["by_origin"]["origin"]
    individual_record = individual_allocation.small_events["by_origin"]["origin"]
    assert summed_record["status"] == "summed_small_events"
    assert summed_record["large_events"] == individual_record["large_events"]
    assert summed_allocation.summed_small_initiators > 0
    assert individual_allocation.summed_small_initiators == 0
    law = single_member_law(compiled.incidents[0].segments[0].kernel)
    assert law is not None
    split = SmallEventLaw(law, SmallEventControls()).split(4_000.0)
    assert not isinstance(split, str)
    small = 4_000.0 * split.small_probability
    for view, coefficient, lab in (
        ("gross", 1.0, 0.0),
        ("welfare", 0.95, 0.0),
        ("lab_loss::lab", 0.15, 1.0),
    ):
        # Large events are common to both runs, so each year's difference is
        # the two bodies' difference: mean zero and twice the conditional
        # small-event variance per small event.
        difference = summed.views[view].array - individual.views[view].array
        g1, g2, _ = split.gross_moments
        l1, l2, _ = split.lab_moments
        variance = coefficient**2 * (g2 - g1**2) + lab**2 * (l2 - l1**2)
        scale = math.sqrt(2 * small * variance)
        assert abs(difference.mean()) < 4 * scale / math.sqrt(years)
        assert difference.std() == pytest.approx(scale, rel=0.08)
        # The analytic EAL is the same law's mean in both runs.
        assert summed.views[view].native_mean == individual.views[view].native_mean
    # OEP: the body maximum lies below the split; a year's maximum is common
    # whenever a large gross event occurs.
    assert np.all(summed.views["oep_gross"].array > 0)
    np.testing.assert_array_equal(
        summed.views["gross"].array >= summed.views["oep_gross"].array, True
    )


def test_summed_bodies_have_the_exact_conditional_mean_and_variance():
    compiled = _state(4_000.0, member=0.9)
    (incident,) = compiled.incidents
    law = single_member_law(incident.segments[0].kernel)
    years = 6_000
    counts = np.random.default_rng(5).poisson(incident.annual_rate, years)
    splits, _ = runtime._small_event_splits(
        compiled, {incident.origin_id: {0: law}}, SmallEventControls()
    )
    plans, _ = runtime._small_event_plans(
        compiled,
        {incident.origin_id: counts},
        splits,
        SmallEventControls(),
        scores={},
        seed=3,
        simulations=years,
    )
    plan = plans[incident.origin_id]
    (_, large_gross), (_, large_lab), (_, drawn_small) = plan.drawn
    summed = counts - large_gross - large_lab - drawn_small
    assert summed.min() > 0
    (split,) = plan.buckets.splits
    assert not isinstance(split, str)
    for body, (first, second, _) in (
        (plan.body_gross, split.gross_moments),
        (plan.body_lab, split.lab_moments),
    ):
        if second == 0:
            continue
        scores = (body - summed * first) / np.sqrt(summed * (second - first**2))
        assert abs(scores.mean()) < 0.06
        assert scores.var() == pytest.approx(1.0, rel=0.06)


def test_a_years_draws_do_not_depend_on_how_many_years_are_simulated():
    compiled = _state(4_000.0, member=0.9)
    short, _ = _run(compiled, 300)
    long, allocation = _run(compiled, 600)
    assert allocation.summed_small_initiators > 0
    for view, values in short.views.items():
        np.testing.assert_array_equal(values.array, long.views[view].array[:300])


def test_summed_body_reproduces_the_annual_mean_and_variance_of_full_simulation():
    compiled = _state(4_000.0)
    draw, _ = _run(compiled, 2_000)
    reference, _ = _run(compiled, 2_000, small_events={"body": "individual"})
    for view in ("gross", "lab_loss::lab"):
        summed, exact = draw.views[view].array, reference.views[view].array
        assert summed.mean() == pytest.approx(exact.mean(), rel=0.01)
        assert summed.std() == pytest.approx(exact.std(), rel=0.1)


@pytest.mark.parametrize(
    ("member", "scale", "rate"),
    [(1.0, 1.0, 4_000.0), (0.7, 2.0, 4_000.0), (1.0, 1.0, 1_000.0)],
    ids=["certain", "gated-doubled", "zero-atom-lab-split"],
)
def test_summed_years_match_the_analytic_mean_of_every_view(member, scale, rate):
    """The analytic EAL is the exact law's mean, so the summed run's annual
    mean must agree with it: this fixes the cell probabilities and the rank
    intervals, which a comparison with the audit mode cannot."""
    compiled = _state(rate, member=member, scale=scale)
    draw, allocation = _run(compiled, 3_000)
    assert allocation.summed_small_initiators > 0
    for view in ("gross", "welfare", "lab_loss::lab"):
        values = draw.views[view].array
        native = draw.views[view].native_mean
        assert native is not None
        error = values.std() / math.sqrt(len(values))
        assert abs(values.mean() - native.value) < 4 * error, view


def test_years_below_the_summed_minimum_draw_their_small_events(monkeypatch):
    """A year with fewer small events than the minimum draws them under the
    same addresses as the audit mode, so the two runs agree exactly there."""
    compiled = _state(4_000.0)
    counts = np.where(np.arange(12) % 2 == 0, 4_000, 300).astype(np.int64)
    monkeypatch.setattr(runtime, "_origin_counts", lambda *_: {"origin": counts})
    summed, allocation = _run(compiled, 12)
    individual, _ = _run(compiled, 12, small_events={"body": "individual"})
    record = allocation.small_events["by_origin"]["origin"]
    assert record["summed_years"] == 6
    low = counts == 300
    for view in VIEWS:
        np.testing.assert_array_equal(
            summed.views[view].array[low], individual.views[view].array[low]
        )
        # Summed years differ, except where a shared large event sets OEP.
        if view != "oep_gross":
            assert not np.array_equal(
                summed.views[view].array[~low], individual.views[view].array[~low]
            )


def test_draws_are_invariant_to_batches_and_requested_views():
    compiled = _state(3_000.0)
    baseline, _ = _run(compiled, 40)
    batched, _ = _run(compiled, 40, event_batch_size=7)
    for view in VIEWS:
        np.testing.assert_array_equal(
            batched.views[view].array, baseline.views[view].array
        )
    plan = _plan(40).model_copy(update={"requested_views": ("welfare",)})
    narrow = runtime.simulate_incident_state(plan, compiled, response_sha256="a" * 64)[
        0
    ]
    np.testing.assert_array_equal(
        narrow.views["welfare"].array, baseline.views["welfare"].array
    )


def _two_status_state():
    """A public origin frequent enough to sum its small events, with lab
    allocations, and a sparse not-yet-public origin in another scenario with
    none, so each leaves some requested views untouched."""
    payload = study_payload()
    payload["entities"].append(
        {
            "entity_id": "lab",
            "kind": "lab",
            "definition": "Synthetic lab",
            "display_name": "lab",
            "status": "hypothetical",
        }
    )
    payload["populations"].append({**payload["populations"][0], "scenario_id": "other"})
    summed = payload["joint_states"][0]["laws"][0]
    summed["segments"][0].update(
        kernel=_kernel(member=0.9).model_dump(mode="json"),
        source_model_id="model",
        source_model_revision="r1",
        source_public_status="public",
    )
    other = json.loads(json.dumps(summed))
    other.update(origin_id="other", scenario_id="other")
    segment = other["segments"][0]
    segment["source_public_status"] = "not_yet_public"
    segment["kernel"]["campaign_lab_losses"] = []
    for component in segment["kernel"]["member_outcomes"][0]["components"]:
        component.update(lab_loss_weights={}, scenario_ids=["other"])
    payload["joint_states"][0]["laws"].append(other)
    compiled = compile_incident_state(setup_from_payload(payload), "state")
    rates = {"origin": 4_000.0, "other": 3.0}
    return replace(
        compiled,
        incidents=tuple(
            replace(incident, annual_rate=rates[incident.origin_id])
            for incident in compiled.incidents
        ),
    )


def test_views_an_origin_cannot_touch_are_skipped_without_changing_any_bit(
    monkeypatch,
):
    """Each origin reduces only the views its events can make nonzero, its
    summed small events included; reducing every view adds +0.0 and gives the
    same arrays and allocation."""
    compiled = _two_status_state()
    plan = _plan(48).model_copy(
        update={
            "requested_views": (
                *VIEWS,
                "scenario::other::welfare",
                "source_status::public::welfare",
                "source_status::not_yet_public::welfare",
                "source_status::not_yet_public::lab_loss::lab",
            )
        }
    )
    skipped = runtime.simulate_incident_state(plan, compiled, response_sha256="a" * 64)
    draw, reasons, allocation = skipped
    assert reasons == {}
    assert allocation.summed_small_initiators > 0
    by_origin = cast(dict, allocation.initiating_counts)
    assert by_origin["other"] > 0
    # No origin touches the not-yet-public lab view; each touches its own
    # scenario's and status's views.
    assert not np.any(draw.views["source_status::not_yet_public::lab_loss::lab"].array)
    for view in ("scenario::other::welfare", "scenario::scenario::gross"):
        assert np.any(draw.views[view].array), view
    monkeypatch.setattr(LinearProjection, "touched", lambda projection: projection)
    every = runtime.simulate_incident_state(plan, compiled, response_sha256="a" * 64)
    assert (every[1], every[2]) == (reasons, allocation)
    assert every[0].views.keys() == draw.views.keys()
    for view, row in draw.views.items():
        assert row.array.tobytes() == every[0].views[view].array.tobytes(), view


def test_the_vectorized_budget_counts_individually_drawn_events():
    compiled = _state(20_000.0)
    _, allocation = _run(compiled, 8)
    drawn = allocation.vectorized_initiators
    assert drawn < allocation.total_initiators / 50
    # The smallest allowance whose eight years admit them, and one less.
    allowance = -(-drawn // 8)
    _, admitted = _run(compiled, 8, max_vectorized_events_per_year=allowance)
    assert admitted.work_limits["vectorized_initiators"] == 8 * allowance
    with pytest.raises(ValueError, match="max_vectorized_events_per_year"):
        _run(compiled, 8, max_vectorized_events_per_year=allowance - 1)


def test_body_maximum_has_the_law_of_the_largest_small_event():
    law = _law(member=0.7)
    split = SmallEventLaw(law, SmallEventControls()).split(50_000.0)
    assert not isinstance(split, str)
    rng = np.random.default_rng(3)
    counts = np.full(20_000, 40)
    maxima = split.body_maximum(counts, rng.random(20_000))
    member = rng.random((20_000, 40)) < split.small_member_probability
    values = GROSS.sample_from_uniforms_uncapped(
        rng.random((20_000, 40)).ravel() * split.rank_gross
    ).reshape(20_000, 40)
    direct = np.where(member, values, 0.0).max(axis=1)
    assert np.all(maxima < split.split_point)
    for level in (0.1, 0.5, 0.9):
        assert np.quantile(maxima, level) == pytest.approx(
            np.quantile(direct, level), rel=0.03
        )


def test_variance_includes_count_dispersion_and_lab_covariance():
    split = SmallEventLaw(_law(), SmallEventControls()).split(50_000.0)
    assert not isinstance(split, str)
    g1, g2, _ = split.gross_moments
    l1, l2, _ = split.lab_moments
    p = split.small_probability
    poisson = body_variance(split, 0.15, 1.0, 50_000.0, 1.0)
    assert poisson == pytest.approx(
        50_000 * p * (0.15**2 * g2 + 2 * 0.15 * g1 * l1 + l2)
    )
    dispersed = body_variance(split, 1.0, 0.0, 50_000.0, 2.0)
    assert dispersed == pytest.approx(
        50_000 * p * (g2 - g1**2) + (p * p * 2 * 50_000 + p * (1 - p) * 50_000) * g1**2
    )


def test_results_report_controls_split_points_and_the_tvar_envelope():
    record = {
        "controls": SmallEventControls().record(),
        "by_origin": {
            "origin": {
                "status": "summed_small_events",
                "scenario_id": "scenario",
                "source_public_status": None,
                "rank_buckets": {"split_point_usd": [None, 1e8, 2e8]},
                "fallback_years": 3,
            }
        },
        "body_variance_by_view": {"gross": 4e18},
        "envelope_reason": None,
    }
    other = {**record, "body_variance_by_view": {"gross": 16e18}}
    executions: dict[str, dict[str, dict[str, Any]]] = {
        "reference": {
            "a": {
                "small_events": record,
                "vectorized_initiators": 10,
                "summed_small_initiators": 90,
            },
            "b": {
                "small_events": other,
                "vectorized_initiators": 5,
                "summed_small_initiators": 95,
            },
        }
    }
    result = small_event_results(
        executions, {"reference": {"a": 0.75, "b": 0.25}}, "reference", (0.99,)
    )
    deviation = math.sqrt(0.75 * 4e18 + 0.25 * 16e18)
    view = result["error_envelope"]["reference"]["views"]["gross"]
    assert view["body_standard_deviation"] == pytest.approx(deviation)
    assert view["tvar_bound"]["0.99"] == pytest.approx(deviation * math.sqrt(99))
    assert tvar_envelope(1.0, 0.99) == pytest.approx(math.sqrt(99))
    assert result["events"] == {
        "individually_drawn": 15,
        "summed": 185,
        "fallback_years": 6,
    }
    assert result["split_points_by_origin"]["scenario::None"] == {
        "states": 2,
        "min_usd": 1e8,
        "max_usd": 2e8,
    }
    assert result["controls"]["target_large_events_per_year"] == 16.0
    uncertified = {**record, "envelope_reason": "Not Poisson."}
    executions["reference"]["a"]["small_events"] = uncertified
    result = small_event_results(
        executions, {"reference": {"a": 1.0, "b": 0.0}}, "reference", (0.99,)
    )
    assert result["error_envelope"]["reference"]["reason"] == "Not Poisson."
    assert result["error_envelope"]["reference"]["views"]["gross"]["tvar_bound"] == {
        "0.99": None
    }
    assert "NUM-SMALL-EVENTS" in SMALL_EVENT_LIMITATION


def test_controls_are_validated():
    with pytest.raises(ValueError, match="take only"):
        SmallEventControls.from_controls({"kappa": 4})
    with pytest.raises(ValueError, match="positive and finite"):
        SmallEventControls.from_controls({"max_body_skewness": 0})
    with pytest.raises(ValueError, match="body must be one of"):
        SmallEventControls.from_controls({"body": "gamma"})
    with pytest.raises(ValueError, match="must exceed one"):
        SmallEventControls.from_controls({"bucket_rate_ratio": 1})
    with pytest.raises(ValueError, match="take only"):
        SmallEventControls.from_controls({"rank_buckets": 32})
    assert SmallEventControls.from_controls(None) == SmallEventControls()


@pytest.mark.parametrize("ranked", [False, True], ids=["fixed-rate", "ranked"])
@pytest.mark.parametrize(
    "tamper",
    [
        "split_point",
        "rate",
        "edge",
        "minimum",
        "variance",
        "envelope",
        "status",
        "realized",
    ],
)
def test_replay_rederives_small_event_records_and_refuses_changes(
    tmp_path, tamper, ranked
):
    payload = study_payload()
    payload["entities"].append(
        {
            "entity_id": "lab",
            "kind": "lab",
            "definition": "Synthetic lab",
            "display_name": "lab",
            "status": "hypothetical",
        }
    )
    law = payload["joint_states"][0]["laws"][0]
    law["segments"][0]["kernel"] = _kernel(member=0.9).model_dump(mode="json")
    law["reference_annual_rate"] = 4_000.0
    if ranked:
        count = _lognormal_count(400.0, 3_000.0)
        law["reference_annual_rate"] = epistemic_mean(count)
        law["parameter_rank"] = {
            "group_id": "group",
            "count": count.model_dump(),
            "tail_form": REFERENCE_SPREAD_TAIL_FORM,
        }
    package = build_setup(
        destination=tmp_path / "package",
        setup=with_execution(
            setup_from_payload(payload), _plan(64), setup_id="small-events"
        ),
        code_commit="fixture",
    )
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")
    capture = tmp_path / "run" / "capture"
    replay_capture(
        setup_path=package.root,
        capture_directory=capture,
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )

    def change(record):
        (row,) = record["arms"]["reference"].values()
        small = row["execution"]["small_events"]
        (origin,) = small["by_origin"].values()
        assert origin["status"] == "summed_small_events"
        buckets = origin["rank_buckets"]
        assert (len(buckets["split_point_usd"]) > 2) == ranked
        # The highest bucket with a split.
        (bucket, *_) = (
            index
            for index in reversed(range(len(buckets["split_point_usd"])))
            if buckets["split_point_usd"][index] is not None
        )
        if tamper == "split_point":
            buckets["split_point_usd"][bucket] *= 1.01
        elif tamper == "rate":
            buckets["representative_rate"][bucket] *= 1.01
        elif tamper == "edge":
            if not ranked:
                pytest.skip("a fixed-rate law has no rank edges")
            buckets["lowest_score"][bucket] += 0.000001
        elif tamper == "minimum":
            buckets["minimum_summed_count"][bucket] += 1
        elif tamper == "realized":
            origin["summed_events"] += 1
        elif tamper == "variance":
            small["body_variance_by_view"]["gross"] *= 0.5
        elif tamper == "envelope":
            small["envelope_reason"] = "Withheld."
        else:
            origin["status"] = "event_by_event"

    record = json.loads((capture / "run-capture.json").read_text())
    change(record)
    record["content_sha256"] = canonical_json_sha256(
        {key: value for key, value in record.items() if key != "content_sha256"}
    )
    (capture / "run-capture.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="small-event"):
        replay_capture(
            setup_path=package.root,
            capture_directory=capture,
            output=tmp_path / "tampered.json",
            code_commit="fixture",
        )


def test_replay_rederives_a_split_without_a_lab_cap(tmp_path):
    """A law whose lab loss has no episode cap, as the lab-loss-cap-removed
    alternative gives, has no lab split point. The record states it as null,
    which a capture's JSON holds and replay re-derives."""
    payload = study_payload()
    payload["entities"].append(
        {
            "entity_id": "lab",
            "kind": "lab",
            "definition": "Synthetic lab",
            "display_name": "lab",
            "status": "hypothetical",
        }
    )
    law = payload["joint_states"][0]["laws"][0]
    kernel = _kernel(member=0.9).model_copy(update={"campaign_lab_losses": ()})
    law["segments"][0]["kernel"] = kernel.model_dump(mode="json")
    law["reference_annual_rate"] = 4_000.0
    package = build_setup(
        destination=tmp_path / "package",
        setup=with_execution(
            setup_from_payload(payload), _plan(64), setup_id="small-events"
        ),
        code_commit="fixture",
    )
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")
    capture = tmp_path / "run" / "capture"
    record = json.loads((capture / "run-capture.json").read_text())
    (row,) = record["arms"]["reference"].values()
    (origin,) = row["execution"]["small_events"]["by_origin"].values()
    assert origin["status"] == "summed_small_events"
    assert origin["rank_buckets"]["lab_split_point_usd"] == [None]
    replay_capture(
        setup_path=package.root,
        capture_directory=capture,
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )


def test_the_envelope_holds_for_negative_binomial_counts_but_not_count_experiments():
    compiled = _state(4_000.0)
    negative_binomial = replace(
        compiled,
        incidents=tuple(
            replace(
                incident, count_family="negative_binomial", count_variance_to_mean=2.0
            )
            for incident in compiled.incidents
        ),
    )
    controls = SmallEventControls()
    splits, audit = runtime.small_event_audit(negative_binomial, VIEWS, controls)
    assert splits and audit["envelope_reason"] is None
    # Count dispersion widens the body's variance by the variance-to-mean ratio
    # of the count law it thins.
    _, poisson = runtime.small_event_audit(compiled, VIEWS, controls)
    assert (
        audit["body_variance_by_view"]["gross"]
        > poisson["body_variance_by_view"]["gross"]
    )
    coupled = replace(
        compiled,
        count_experiment=CountExperiment(
            kind="shared_gamma_intensity",
            origin_groups={
                incident.origin_id: "portfolio" for incident in compiled.incidents
            },
            coefficient_of_variation=1.0,
        ),
    )
    _, experiment = runtime.small_event_audit(coupled, VIEWS, controls)
    assert "count experiment" in experiment["envelope_reason"]


# Rank buckets: a year's split points follow its parameter rank.

_SCORES = (-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817)


def _lognormal_count(median, ratio):
    """Count quantiles log-symmetric about ``median`` with p95/p5 ``ratio``."""
    sigma = math.log(ratio) / (2 * -_SCORES[0])
    return Quantiles(
        **dict(
            zip(
                ("p5", "p25", "p50", "p75", "p95"),
                (median * math.exp(sigma * z) for z in (*_SCORES, -_SCORES[0])),
                strict=True,
            )
        )
    )


# The count spread of the real test-run answers: p95/p5 of about 3,000.
WIDE_COUNT = _lognormal_count(20_000.0, 3_000.0)
# PCHIP ladders far from lognormal: a steep one with a near-zero lowest knot,
# one that is zero below p25 and one with a lowest knot a millionth of p25.
STEEP_COUNT = Quantiles(p5=0.001, p25=0.5, p50=2, p75=30, p95=4000)
ZERO_BELOW_P25 = Quantiles(p5=0, p25=0, p50=0.5, p75=2, p95=8)
NEAR_ZERO_P5 = Quantiles(p5=1e-6, p25=1, p50=2, p75=4, p95=10)


def _rank(incident, count, mean, group="group"):
    law = ParameterRankLaw(
        group_id=group, count=count, tail_form=REFERENCE_SPREAD_TAIL_FORM
    )
    return replace(incident, annual_rate=mean, parameter_rank=law)


def _ranked_state(count=WIDE_COUNT, *, mean=None, group="group", **kernel):
    """A state whose one origin reads ``count`` at a fresh rank each year, with
    mean count ``mean`` (the count law's own mean by default)."""
    compiled = _state(1.0, **kernel)
    mean = epistemic_mean(count) if mean is None else mean
    return replace(
        compiled,
        incidents=tuple(
            _rank(incident, count, mean, group) for incident in compiled.incidents
        ),
    )


def _two_origin_state():
    """Two ranked origins with different count laws, groups and shares, so the
    gross, welfare and lab views are not multiples of one another."""
    payload = study_payload()
    payload["entities"].append(
        {
            "entity_id": "lab",
            "kind": "lab",
            "definition": "Synthetic lab",
            "display_name": "lab",
            "status": "hypothetical",
        }
    )
    first = payload["joint_states"][0]["laws"][0]
    first["segments"][0]["kernel"] = _kernel(member=0.9).model_dump(mode="json")
    second = json.loads(json.dumps(first))
    second["origin_id"] = "second"
    second["segments"][0]["kernel"] = _kernel(
        severity=ANCHORED,
        shares={"third_party": 0.2, "provider": 0.3, "transfer": 0.5},
    ).model_dump(mode="json")
    payload["joint_states"][0]["laws"].append(second)
    compiled = compile_incident_state(setup_from_payload(payload), "state")
    first_rank, second_rank = compiled.incidents
    return replace(
        compiled,
        incidents=(
            _rank(first_rank, _lognormal_count(1_000.0, 3_000.0), 12_000.0),
            _rank(second_rank, STEEP_COUNT, 6_000.0, group="other"),
        ),
    )


def _mean_rate_buckets(incident, prepared, segment):
    """Every year at the reference rate's split, as without rank buckets."""
    return fixed_rate_buckets(prepared, segment, incident.annual_rate)


def _year_buckets(compiled, years, controls=SmallEventControls()):
    plan = _plan(years)
    splits, _ = runtime.small_event_audit(compiled, VIEWS, controls)
    scores = runtime.parameter_scores(
        compiled.incidents, plan.seed, years, compiled.rank_stream_id
    )
    (buckets,) = splits.values()
    return buckets, scores["origin"], buckets.assign(scores["origin"], years)


def _drawn_by_year(compiled, buckets, years, controls=SmallEventControls()):
    plan = _plan(years)
    scores = runtime.parameter_scores(
        compiled.incidents, plan.seed, years, compiled.rank_stream_id
    )
    plans, realized = runtime._small_event_plans(
        compiled,
        runtime._origin_counts(compiled, plan, scores),
        {"origin": buckets},
        controls,
        scores=scores,
        seed=runtime.state_stream_seed(plan.seed, compiled.state_id),
        simulations=years,
    )
    return sum(counts for _, counts in plans["origin"].drawn), realized["origin"]


@pytest.mark.parametrize(
    ("count", "spread", "tail_scale"),
    [
        (WIDE_COUNT, 1.0, 1.0),
        (WIDE_COUNT, 1.5, 0.5),
        (STEEP_COUNT, 0.7, 1.5),
        (ZERO_BELOW_P25, 1.0, 1.0),
        (NEAR_ZERO_P5, 1.0, 1.0),
    ],
    ids=["wide", "stretched", "steep", "zero-below-p25", "near-zero-p5"],
)
def test_buckets_are_cut_by_rate_ratio_and_share_the_mean_count(
    count, spread, tail_scale
):
    """Below the top bucket each edge is the lowest score whose expected count
    reaches the top score's divided by a power of the ratio, so every bucket
    with a split spans at most that ratio; cutting stops where the count is too
    low to split. Each bucket's share of the mean count is the integral of the
    year's multiplier over its scores, and together they integrate the
    representable score range."""
    law = ParameterRankLaw(
        group_id="group",
        count=count,
        rank_spread=spread,
        epistemic_tail_scale=tail_scale,
        tail_form=REFERENCE_SPREAD_TAIL_FORM,
    )
    incident = replace(
        _state(1.0).incidents[0], annual_rate=256_000.0, parameter_rank=law
    )
    controls = SmallEventControls()
    buckets = runtime._origin_buckets(incident, SmallEventLaw(_law(), controls), 0)
    assert not isinstance(buckets, str)
    multiplier = runtime.score_multiplier(law)
    edges = np.array(buckets.edges)
    assert len(buckets.splits) == len(edges) + 1 <= 64
    assert edges[-1] == controls.top_rank_score
    lowest = np.array([SIMULATED_SCORE_RANGE[0], *edges])
    np.testing.assert_array_equal(buckets.rates, 256_000.0 * multiplier(lowest))
    # Cut by the ratio below the top bucket, at the lowest score reaching it.
    top = buckets.rates[-1]
    for bucket, edge in enumerate(edges[:-1], start=1):
        target = top / 2.0 ** (len(edges) - bucket)
        assert target <= buckets.rates[bucket] <= target * (1 + 1e-10)
        below = edge - 1e-7
        assert 256_000.0 * multiplier(np.array([below]))[0] < target
    # Only the lowest bucket lacks a split, and its rate refuses one.
    assert isinstance(buckets.splits[0], str)
    assert all(not isinstance(split, str) for split in buckets.splits[1:])
    assert buckets.largest_rate_ratio == pytest.approx(2.0, rel=1e-5)
    low, high = SIMULATED_SCORE_RANGE
    # The shares integrate the independent oracle's law, not the multiplier.
    oracle = CountLawOracle(
        count,
        tail_scale=tail_scale,
        spread=spread,
        tail_form=REFERENCE_SPREAD_TAIL_FORM,
    )
    mean = oracle.mean()

    def integrand(score):
        value = oracle.at_score(score) / mean
        return value * math.exp(-score * score / 2) / math.sqrt(2 * math.pi)

    bounds = [low, *edges, high]
    kinks = oracle.kinks()
    for bucket, share in enumerate(buckets.mean_shares):
        left, right = bounds[bucket], bounds[bucket + 1]
        inside = [kink for kink in kinks if left < kink < right]
        reference = integrate.quad(
            integrand, left, right, points=inside or None, limit=400, epsrel=1e-11
        )[0]
        assert share == pytest.approx(reference, rel=1e-10, abs=1e-12)
    representable = integrate.quad(
        integrand,
        low,
        high,
        points=sorted(score for score in kinks if low < score < high) or None,
        limit=400,
        epsrel=1e-11,
    )[0]
    assert math.fsum(buckets.mean_shares) == pytest.approx(
        representable, rel=1e-10, abs=1e-12
    )


def test_each_year_takes_its_bucket_from_its_rank_alone():
    """For a fixed law a year's bucket is a fixed function of its parameter
    rank: prefix-stable in the year count and independent of the realized
    counts, and its split was found at the lowest rate a year in that bucket
    can have."""
    compiled = _ranked_state()
    buckets, scores, assigned = _year_buckets(compiled, 2_048)
    np.testing.assert_array_equal(
        assigned, np.searchsorted(np.array(buckets.edges), scores, side="right")
    )
    _, _, short = _year_buckets(compiled, 512)
    np.testing.assert_array_equal(short, assigned[:512])
    # The representative rate is conservative: no year in a bucket reads a
    # lower expected count, and none reads the next bucket's.
    (incident,) = compiled.incidents
    assert incident.parameter_rank is not None
    rates = incident.annual_rate * runtime.score_multiplier(incident.parameter_rank)(
        scores
    )
    lowest = np.asarray(buckets.rates)
    assert np.all(rates >= lowest[assigned])
    upper = assigned < len(lowest) - 1
    assert np.all(rates[upper] <= lowest[assigned[upper] + 1])
    # Cell counts follow the year's bucket whatever the counts are.
    plan = _plan(2_048)
    poisson = runtime._origin_counts(compiled, plan)["origin"]
    for counts in (poisson, np.full(2_048, 50_000)):
        plans, _ = runtime._small_event_plans(
            compiled,
            {"origin": counts},
            {"origin": buckets},
            SmallEventControls(),
            scores={"origin": scores},
            seed=3,
            simulations=2_048,
        )
        for cell, cell_counts in plans["origin"].drawn:
            if cell is None:
                own = np.array([isinstance(split, str) for split in buckets.splits])
            else:
                own = np.array(
                    [
                        not isinstance(split, str) and cell in split.cells().values()
                        for split in buckets.splits
                    ]
                )
            assert np.all(own[assigned[cell_counts > 0]])


@pytest.mark.parametrize(
    "count",
    [WIDE_COUNT, STEEP_COUNT, ZERO_BELOW_P25, NEAR_ZERO_P5],
    ids=["wide", "steep", "zero-below-p25", "near-zero-p5"],
)
def test_a_wide_count_law_keeps_the_event_budget(count):
    """At a mean of 256,000 a year, count laws with p95/p5 of 3,000 or PCHIP
    ladders with a zero or near-zero lowest knot keep every year's
    individually drawn events near the target: each bucket with a split spans
    at most the rate ratio, and the lowest bucket, drawn event by event, only
    counts too low to sum."""
    compiled = _ranked_state(count, mean=256_000.0, member=0.9)
    years = 1_024
    _, allocation = _run(compiled, years, max_vectorized_events_per_year=300)
    record = allocation.small_events["by_origin"]["origin"]
    assert record["fallback_years"] == 0
    assert record["rank_buckets"]["largest_rate_ratio"] == pytest.approx(2.0)
    assert allocation.summed_small_initiators > 0.999 * allocation.total_initiators
    buckets, scores, _ = _year_buckets(compiled, years)
    drawn, _ = _drawn_by_year(compiled, buckets, years)
    assert drawn.sum() == allocation.vectorized_initiators
    # The upper ranks, which carry most events, draw few individually, and no
    # year draws more than twice the lowest bucket's highest count.
    assert drawn[scores > ndtri(0.9)].max() < 4 * SmallEventControls().target
    assert drawn.max() < 1_000
    assert buckets.rates[1] < 500


def test_one_split_at_the_reference_rate_breaks_the_budget_rank_buckets_keep():
    """With one split at the count law's mean, most years of a law with
    p95/p5 = 3,000 have too few small events to sum and draw them all."""
    compiled = _ranked_state(mean=256_000.0, member=0.9)
    years = 1_024
    _, allocation = _run(compiled, years, max_vectorized_events_per_year=300)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runtime, "_origin_buckets", _mean_rate_buckets)
        with pytest.raises(ValueError, match="max_vectorized_events_per_year"):
            _run(compiled, years, max_vectorized_events_per_year=300)
        _, mean_rate = _run(compiled, years, max_vectorized_events_per_year=10**6)
    record = mean_rate.small_events["by_origin"]["origin"]
    assert record["fallback_years"] > 0.3 * years
    assert mean_rate.vectorized_initiators > 100 * allocation.vectorized_initiators


def test_a_bucket_refused_above_a_split_one_draws_every_event(monkeypatch):
    """A refusal a lower count need not share, such as moments that cannot be
    certified, leaves that bucket drawn event by event while cutting goes on
    below it; its years match the fully event-by-event path exactly."""
    compiled = _ranked_state(_lognormal_count(400.0, 3_000.0), member=0.9)
    original = SmallEventLaw.split
    refused = "The small events' moments cannot be integrated to the tolerance."

    def split(self, rate):
        return refused if 2_000 <= rate < 8_000 else original(self, rate)

    monkeypatch.setattr(SmallEventLaw, "split", split)
    years = 2_048
    buckets, _, assigned = _year_buckets(compiled, years)
    unsplit = np.array([isinstance(split, str) for split in buckets.splits])
    middle = [
        bucket
        for bucket in range(1, len(unsplit) - 1)
        if unsplit[bucket] and not unsplit[bucket - 1]
    ]
    assert middle and buckets.splits[middle[0]] == refused
    summed, allocation = _run(compiled, years)
    exact, _ = _run(compiled, years, small_events={"target_large_events_per_year": 1e9})
    in_middle = np.isin(assigned, middle)
    below = (assigned < middle[0]) & ~unsplit[assigned]
    above = (assigned > max(middle)) & ~unsplit[assigned]
    assert in_middle.sum() > 20 and below.sum() > 20 and above.sum() > 20
    record = allocation.small_events["by_origin"]["origin"]
    assert record["event_by_event_years"] >= in_middle.sum() * 0.9
    for view in VIEWS:
        np.testing.assert_array_equal(
            summed.views[view].array[in_middle], exact.views[view].array[in_middle]
        )
        if view != "oep_gross":
            assert not np.array_equal(
                summed.views[view].array[below], exact.views[view].array[below]
            )


def test_events_above_a_years_split_point_are_drawn_exactly():
    """The large cells of a year's bucket are common to the summed and the
    event-by-event audit runs, so every year whose largest event reaches its
    own split point has the same OEP in both, and a year in a bucket without a
    split draws every event as an origin without any split would."""
    compiled = _ranked_state(_lognormal_count(400.0, 3_000.0), member=0.9)
    years = 1_024
    summed, summed_allocation = _run(compiled, years)
    individual, individual_allocation = _run(
        compiled, years, small_events={"body": "individual"}
    )
    summed_record = summed_allocation.small_events["by_origin"]["origin"]
    individual_record = individual_allocation.small_events["by_origin"]["origin"]
    assert summed_record["large_events"] == individual_record["large_events"]
    assert summed_record["summed_years"] > years / 4
    buckets, _, assigned = _year_buckets(compiled, years)
    points = np.array(
        [
            math.inf if isinstance(split, str) else split.split_point
            for split in buckets.splits
        ]
    )[assigned]
    oep = individual.views["oep_gross"].array
    above = oep >= points
    assert above.sum() > years / 4
    np.testing.assert_array_equal(summed.views["oep_gross"].array[above], oep[above])
    unsplit = np.isinf(points)
    assert unsplit.sum() > 10
    exact, exact_allocation = _run(
        compiled, years, small_events={"target_large_events_per_year": 1e9}
    )
    assert exact_allocation.small_events["by_origin"]["origin"]["status"] == (
        "event_by_event"
    )
    for view in VIEWS:
        np.testing.assert_array_equal(
            summed.views[view].array[unsplit], exact.views[view].array[unsplit]
        )
        np.testing.assert_array_equal(
            individual.views[view].array[unsplit], exact.views[view].array[unsplit]
        )
    # Each year's large gross events are as many as its own split implies.
    plan = _plan(years)
    counts = runtime._origin_counts(compiled, plan)["origin"]
    plans, _ = runtime._small_event_plans(
        compiled,
        {"origin": counts},
        {"origin": buckets},
        SmallEventControls(),
        scores=runtime.parameter_scores(
            compiled.incidents, plan.seed, years, compiled.rank_stream_id
        ),
        seed=5,
        simulations=years,
    )
    large_gross = sum(
        cell_counts
        for cell, cell_counts in plans["origin"].drawn
        if cell is not None and cell.purpose_prefix == "small-events::large-gross::"
    )
    probability = np.array(
        [
            0.0 if isinstance(split, str) else split.large_gross_probability
            for split in buckets.splits
        ]
    )[assigned]
    expected = counts * probability
    assert abs(large_gross.sum() - expected.sum()) < 4 * math.sqrt(expected.sum())


@pytest.mark.stochastic
def test_results_do_not_depend_on_the_bucket_rate_ratio():
    """A finer rate ratio moves every year's split, and with it which events are
    drawn individually, so the two runs differ by Monte Carlo noise: the annual
    means and tails of the gross, welfare and lab views, which are not
    multiples of one another, differ by under four paired batch errors, under
    1.5% (observed 0.01% to 0.3%) and by less than the two envelopes (observed
    at most a seventh of them). The batch error of a 99% tail mean rests on a
    few extreme years, so it is not itself compared with the envelope."""
    compiled = _two_origin_state()
    years = 32_768
    runs = {}
    for ratio in (2.0, 1.5):
        draw, allocation = _run(
            compiled, years, small_events={"bucket_rate_ratio": ratio}
        )
        runs[ratio] = (draw, allocation.small_events["body_variance_by_view"])
    batches = 16
    gross, welfare = (runs[2.0][0].views[view].array for view in ("gross", "welfare"))
    assert np.corrcoef(gross, welfare)[0, 1] < 0.999
    for view in ("gross", "welfare", "lab_loss::lab"):
        coarse, fine = (runs[ratio][0].views[view].array for ratio in (2.0, 1.5))
        mean_differences = (coarse - fine).reshape(batches, -1).mean(axis=1)
        error = mean_differences.std(ddof=1) / math.sqrt(batches)
        assert abs(mean_differences.mean()) < 4 * error + 1e-9 * fine.mean(), view
        for level in (0.9, 0.99):

            def tail_mean(values, level=level):
                ordered = np.sort(values)
                return ordered[int(level * len(ordered)) :].mean()

            differences = [
                tail_mean(a) - tail_mean(b)
                for a, b in zip(
                    coarse.reshape(batches, -1), fine.reshape(batches, -1), strict=True
                )
            ]
            noise = np.std(differences, ddof=1) / math.sqrt(batches)
            envelope = sum(
                tvar_envelope(math.sqrt(runs[ratio][1][view]), level)
                for ratio in (2.0, 1.5)
            )
            difference = tail_mean(coarse) - tail_mean(fine)
            reference = tail_mean(fine)
            assert abs(difference) <= 4 * noise, (view, level)
            assert abs(difference) <= 0.015 * reference, (view, level)
            assert abs(difference) < envelope, (view, level)


def test_the_envelope_variance_mixes_the_buckets_by_their_share_of_the_mean():
    """A year's small-event variance is linear in its expected count, so the
    envelope's variance is each bucket's at its share of the mean count; the
    simulated sums' variance about their conditional means matches it."""
    compiled = _ranked_state(_lognormal_count(8_000.0, 20.0), member=0.9)
    controls = SmallEventControls()
    splits, audit = runtime.small_event_audit(compiled, VIEWS, controls)
    (buckets,) = splits.values()
    (incident,) = compiled.incidents
    mixed = math.fsum(
        body_variance(split, 1.0, 0.0, incident.annual_rate * share, 1.0)
        for split, share in zip(buckets.splits, buckets.mean_shares, strict=True)
        if not isinstance(split, str)
    )
    assert audit["body_variance_by_view"]["gross"] == pytest.approx(mixed, rel=1e-12)
    years = 60_000
    plan = _plan(years)
    counts = runtime._origin_counts(compiled, plan)["origin"]
    scores = runtime.parameter_scores(
        compiled.incidents, plan.seed, years, compiled.rank_stream_id
    )
    plans, realized = runtime._small_event_plans(
        compiled,
        {"origin": counts},
        splits,
        controls,
        scores=scores,
        seed=11,
        simulations=years,
    )
    assert realized["origin"]["fallback_years"] == 0
    assigned = buckets.assign(scores["origin"], years)
    assert incident.parameter_rank is not None
    rates = incident.annual_rate * runtime.score_multiplier(incident.parameter_rank)(
        scores["origin"]
    )
    # Gross and aggregate-lab sums, whose moments differ by bucket: the upper
    # buckets keep positive aggregate lab losses small, the lower ones only its
    # zero atom.
    lab_mixed = math.fsum(
        body_variance(split, 0.0, 1.0, incident.annual_rate * share, 1.0)
        for split, share in zip(buckets.splits, buckets.mean_shares, strict=True)
        if not isinstance(split, str)
    )
    lab_means = {
        split.lab_moments[0] for split in buckets.splits if not isinstance(split, str)
    }
    assert 0.0 in lab_means and max(lab_means) > 0
    for body, moments, variance in (
        (plans["origin"].body_gross, "gross_moments", mixed),
        (plans["origin"].body_lab, "lab_moments", lab_mixed),
    ):
        conditional = np.array(
            [
                0.0
                if isinstance(split, str)
                else split.small_probability * getattr(split, moments)[0]
                for split in buckets.splits
            ]
        )[assigned]
        deviations = body - rates * conditional
        assert np.mean(deviations**2) == pytest.approx(variance, rel=0.05)


def test_an_origin_needing_too_many_buckets_is_refused():
    compiled = _ranked_state(mean=256_000.0)
    with pytest.raises(ValueError, match="more than 64 rank buckets"):
        runtime.small_event_audit(compiled, VIEWS, SmallEventControls(rate_ratio=1.05))


def test_the_split_cache_reuses_splits_and_leaves_draws_unchanged(monkeypatch):
    """A caller's cache for one compiled state reuses its splits at a later
    year count; the draws and records equal those computed afresh."""
    compiled = _ranked_state(_lognormal_count(400.0, 3_000.0), member=0.9)
    cache: dict = {}
    calls = []
    original = runtime._small_event_splits

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(runtime, "_small_event_splits", counted)
    for years in (256, 512):
        cached, _, cached_allocation = runtime.simulate_incident_state(
            _plan(years), compiled, response_sha256="a" * 64, split_cache=cache
        )
    assert len(calls) == 1
    fresh, _, fresh_allocation = runtime.simulate_incident_state(
        _plan(512), compiled, response_sha256="a" * 64
    )
    assert len(calls) == 2
    for view in VIEWS:
        np.testing.assert_array_equal(cached.views[view].array, fresh.views[view].array)
    assert cached_allocation.small_events == fresh_allocation.small_events
    # Other controls find their own splits in the same cache.
    coarse = SmallEventControls(rate_ratio=4.0)
    splits, _ = runtime.small_event_audit(compiled, VIEWS, coarse, cache=cache)
    assert len(calls) == 3
    fresh_splits, _ = runtime.small_event_audit(compiled, VIEWS, coarse)
    assert splits == fresh_splits
    default = cache[SmallEventControls()][0]["origin"]
    assert len(splits["origin"].splits) < len(default.splits)
