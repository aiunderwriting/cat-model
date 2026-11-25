"""Independent count and financial oracles for the incident engine."""

import math

import numpy as np
import pytest
from scipy.stats import poisson

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.probability.distributions import Distribution, MixtureSeverityDistribution
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_moments import analytic_linear_eals
from catmodel.simulation.incident_runtime import _origin_counts, simulate_incident_state
from test_support.incident import common_case as _case
from test_support.run_setup import setup_from_payload


def _common(
    *,
    rate=1,
    amount=1,
    sampler=None,
    simulations=4000,
    financial=False,
    unit_count=1,
    shared=1,
    unit=1,
    scope="member",
    retention=0,
    occurrence_limit=None,
    annual_limit=None,
    coverage=1,
):
    study, plan, _, _ = _case(
        sampler=sampler, financial=financial, rate=rate, simulations=simulations
    )
    payload = encode_scientific(study)
    for state in payload["joint_states"]:
        law = state["laws"][0]
        kernel = law["segments"][0]["kernel"]
        kernel.update(shared_probability=shared, unit_probability=unit)
        kernel["cluster"]["support"] = [{"count": unit_count, "probability": 1}]
        outcome = kernel["member_outcomes"][0]
        if sampler is None:
            outcome["components"][0]["amount"] = amount
        if financial:
            membership = outcome["memberships"][0]
            membership.update(occurrence_scope=scope, covered_fraction=coverage)
    if financial:
        payload["contracts"][0].update(
            occurrence_retention=retention,
            occurrence_limit=occurrence_limit,
            annual_retention=0,
            annual_limit=annual_limit,
        )
    study = setup_from_payload(payload)
    views = ("gross", "oep_gross") + (
        ("financial", "oep_financial") if financial else ()
    )
    plan = plan.model_copy(update={"requested_views": views})
    return study, plan, compile_incident_state(study, "low")


def _finite(values, weights):
    return MixtureSeverityDistribution(
        [Distribution(value, value, value, value, value) for value in values],
        np.asarray(weights),
    )


def _probability_bound(probability, n):
    return 6 * math.sqrt(probability * (1 - probability) / n) + 1 / n


def test_occurrence_hook_is_applied_to_the_selected_grouping_unit():
    for scope, expected_per_origin in (("member", 0), ("campaign", 2)):
        _study, plan, compiled = _common(
            rate=0.7,
            amount=6,
            simulations=128,
            financial=True,
            unit_count=2,
            scope=scope,
            retention=10,
        )
        draw, unavailable, allocation = simulate_incident_state(
            plan, compiled, response_sha256=_study.scientific_sha256
        )
        assert not unavailable
        initiators = allocation.total_initiators
        assert isinstance(initiators, int)
        assert draw.annual_losses["financial"].sum() == expected_per_origin * initiators


def test_incommensurate_occurrence_limit_preserves_the_cap_atom():
    probabilities = np.array([0.10, 0.20, 0.20, 0.20, 0.30])
    cap = 7.701
    _study, plan, compiled = _common(
        rate=0.8,
        sampler=_finite([0, 5, 10, 15, 20], probabilities),
        financial=True,
        retention=4.9,
        occurrence_limit=cap,
        simulations=8000,
    )
    result, unavailable, _ = simulate_incident_state(
        plan, compiled, response_sha256="a" * 64
    )
    assert not unavailable
    maximum = result.annual_losses["oep_financial"]
    expected_zero = math.exp(-0.8 * (1 - probabilities[0]))
    expected_cap = 1 - math.exp(-0.8 * probabilities[3:].sum())
    assert np.mean(maximum == 0) == pytest.approx(
        expected_zero, abs=_probability_bound(expected_zero, plan.simulations)
    )
    assert np.mean(np.isclose(maximum, cap, rtol=0, atol=1e-12)) == pytest.approx(
        expected_cap, abs=_probability_bound(expected_cap, plan.simulations)
    )
    assert maximum.max() == pytest.approx(cap, abs=1e-12)
    assert np.all((maximum == 0) | (maximum >= 0.1 - 1e-12))


def test_per_unit_oep_uses_episode_gate_and_unit_gate():
    rate, shared, unit = 1.2, 0.5, 0.5
    _study, plan, compiled = _common(
        rate=rate,
        amount=5,
        financial=True,
        unit_count=2,
        shared=shared,
        unit=unit,
        simulations=1500,
    )
    draw, unavailable, _ = simulate_incident_state(
        plan, compiled, response_sha256=_study.scientific_sha256
    )
    assert not unavailable
    no_positive_one_origin = (1 - shared) + shared * (1 - unit) ** 2
    expected_zero = math.exp(rate * (no_positive_one_origin - 1))
    maximum = draw.annual_losses["oep_financial"]
    assert np.mean(maximum == 0) == pytest.approx(
        expected_zero, abs=_probability_bound(expected_zero, plan.simulations)
    )
    assert set(maximum) <= {0, 5}


def test_incommensurate_aggregate_limit_preserves_the_cap_atom(monkeypatch):
    counts = np.array([0, 1, 50, 1000])
    _study, plan, compiled = _common(
        amount=100, financial=True, simulations=len(counts), annual_limit=7777.3
    )
    monkeypatch.setattr(
        "catmodel.simulation.incident_runtime._origin_counts",
        lambda *_: {"campaign": counts},
    )
    result, unavailable, _ = simulate_incident_state(
        plan, compiled, response_sha256="a" * 64
    )
    assert not unavailable
    np.testing.assert_allclose(
        result.annual_losses["financial"],
        np.minimum(100 * counts, 7777.3),
        rtol=0,
        atol=1e-12,
    )
    assert result.annual_losses["financial"][-1] == 7777.3


@pytest.mark.parametrize("rate", [10_000, 100_000, 1_000_000])
def test_high_rate_native_counts_preserve_the_exact_poisson_law(rate):
    _study, plan, compiled = _common(rate=rate, simulations=4000)
    counts = _origin_counts(compiled, plan)["campaign"]
    assert analytic_linear_eals(compiled, ("gross",)).values == {"gross": rate}
    for quantile in (0.95, 0.99, 0.999):
        threshold = poisson.ppf(quantile, rate)
        probability = poisson.cdf(threshold, rate)
        assert np.mean(counts <= threshold) == pytest.approx(
            probability, abs=_probability_bound(probability, plan.simulations)
        )


def test_native_marks_preserve_a_small_coverage_transform(monkeypatch):
    counts = np.array([0, 1, 3, 100])
    _study, plan, compiled = _common(
        rate=100,
        amount=100,
        financial=True,
        coverage=0.01,
        simulations=len(counts),
    )
    monkeypatch.setattr(
        "catmodel.simulation.incident_runtime._origin_counts",
        lambda *_: {"campaign": counts},
    )
    result, unavailable, _ = simulate_incident_state(
        plan, compiled, response_sha256="a" * 64
    )
    assert not unavailable
    np.testing.assert_array_equal(result.annual_losses["gross"], 100 * counts)
    np.testing.assert_array_equal(result.annual_losses["financial"], counts)


def test_positive_occurrence_support_retains_no_incident_oep_atom(monkeypatch):
    count = poisson.ppf((np.arange(4000) + 0.5) / 4000, 1).astype(np.int64)
    _study, plan, compiled = _common(simulations=len(count))
    monkeypatch.setattr(
        "catmodel.simulation.incident_runtime._origin_counts",
        lambda *_: {"campaign": count},
    )
    result, unavailable, _ = simulate_incident_state(
        plan, compiled, response_sha256="a" * 64
    )
    assert not unavailable
    maximum = result.annual_losses["oep_gross"]
    assert np.mean(maximum == 0) == pytest.approx(math.exp(-1), abs=1 / len(count))
    assert set(maximum) == {0, 1}
