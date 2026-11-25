"""Independent finite-law checks for physical incident marks."""

import math

import numpy as np
import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.probability.distributions import Distribution, MixtureSeverityDistribution
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_runtime import simulate_incident_state
from test_support.incident import common_case as _case
from test_support.run_setup import setup_from_payload


def _financial_law(*, sampler=None, rate=1, simulations=6000):
    study, plan, _, _ = _case(
        sampler=sampler, financial=True, rate=rate, simulations=simulations
    )
    payload = encode_scientific(study)
    payload["contracts"][0].update(
        occurrence_retention=0, occurrence_limit=1, annual_retention=0
    )
    study = setup_from_payload(payload)
    plan = plan.model_copy(
        update={
            "requested_views": ("gross", "financial", "oep_gross", "oep_financial"),
            "numerical_controls": {},
        }
    )
    return study, plan, compile_incident_state(study, "low")


def test_joint_gross_payout_tail_matches_multivariate_poisson_oracle():
    severity = MixtureSeverityDistribution(
        components=[Distribution(1, 1, 1, 1, 1), Distribution(100, 100, 100, 100, 100)],
        weights=np.array([0.5, 0.5]),
    )
    _unused_study, plan, compiled = _financial_law(sampler=severity)
    draw, unavailable, _ = simulate_incident_state(
        plan, compiled, response_sha256="a" * 64
    )
    assert not unavailable
    losses = draw.annual_losses
    gross, payout = losses["gross"], losses["financial"]
    expected = 1 - math.exp(-0.5) - 0.5 * math.exp(-1)
    assert np.mean((gross > 50) & (payout > 1)) == pytest.approx(expected, abs=0.02)
    assert np.mean(losses["oep_financial"] == 0) == pytest.approx(
        math.exp(-1), abs=0.02
    )
    assert np.all(losses["oep_financial"] <= payout)
    maximum = losses["oep_gross"]
    for threshold, cdf in (
        (0, math.exp(-1)),
        (1, math.exp(-0.5)),
        (50, math.exp(-0.5)),
        (100, 1),
    ):
        assert np.mean(maximum <= threshold) == pytest.approx(cdf, abs=0.02)
    assert np.all(maximum <= gross)
    # These integer constraints fail under rank-paired marginal inversions.
    np.testing.assert_allclose(
        (gross - payout) / 99, np.rint((gross - payout) / 99), atol=1e-12
    )
    assert np.all(gross >= payout)


def test_canonical_gate_states_keep_post_cluster_limit_and_zero_mass():
    estimates = []
    for shared, unit in ((0.01, 1), (1, 0.01)):
        study, plan, _ = _financial_law(simulations=5000)
        payload = encode_scientific(study)
        for state in payload["joint_states"]:
            kernel = state["laws"][0]["segments"][0]["kernel"]
            kernel.update(shared_probability=shared, unit_probability=unit)
            kernel["cluster"]["support"] = [{"count": 100, "probability": 1}]
            outcome = kernel["member_outcomes"][0]
            outcome["components"][0]["amount"] = 1
            outcome["memberships"][0].update(occurrence_scope="campaign")
        study = setup_from_payload(payload)
        draw, unavailable, _ = simulate_incident_state(
            plan, compile_incident_state(study, "low"), response_sha256="a" * 64
        )
        assert not unavailable
        estimates.append(
            (
                np.mean(draw.annual_losses["gross"] > 0),
                np.mean(draw.annual_losses["financial"]),
            )
        )
    annual_positive = 0.5 * (-math.expm1(-0.01) - math.expm1(-(1 - 0.99**100)))
    expected_payout = 0.5 * (0.01 + 1 - 0.99**100)
    assert np.mean(np.array(estimates)[:, 0]) == pytest.approx(
        annual_positive, abs=0.025
    )
    assert np.mean(np.array(estimates)[:, 1]) == pytest.approx(
        expected_payout, abs=0.025
    )


def test_direct_common_marks_keep_a_capped_heavy_tail_finite():
    severity = Distribution(
        1, 2, 4, 8, 16, gpd_shape=5.0, gpd_scale=1000, severity_cap=100
    )
    _unused_study, plan, compiled, _ = _case(
        sampler=severity, multiplier=0.5, simulations=1000, rate=1
    )
    plan = plan.model_copy(update={"requested_views": ("gross",)})
    draw, unavailable, _ = simulate_incident_state(
        plan, compiled, response_sha256="a" * 64
    )
    assert not unavailable
    assert np.all(np.isfinite(draw.annual_losses["gross"]))
    assert np.any(draw.annual_losses["gross"] >= 100)
