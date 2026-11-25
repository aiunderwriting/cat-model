"""Incident captures retain their moment meaning in shared readers."""

import json

import numpy as np
import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.results.reduction import mean_diagnostics, tail_diagnostics
from catmodel.simulation.capability_mixture import (
    CapabilityMixture,
)
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_runtime import simulate_incident_state
from catmodel.simulation.tail_gate import TailGateControls, view_gate
from test_support.run_setup import setup_from_payload
from test_support.study import _financial_study, _plan, _study


def _mixture(rows, weights=None):
    draws = {draw.stratum_sha256: draw for draw, _ in rows}
    target = weights or {next(iter(draws)): 1.0}
    return CapabilityMixture(draws=draws, stratum_weights=target)


def _run(study, *, state="low", years=64, seed=42):
    compiled = compile_incident_state(study, state)
    plan = _plan(years).model_copy(update={"requested_views": ("gross",), "seed": seed})
    draw, reasons, _ = simulate_incident_state(
        plan, compiled, response_sha256=study.scientific_sha256
    )
    assert not reasons
    return draw, compiled


@pytest.mark.parametrize("shape", [0.6, 1.2])
def test_runtime_moment_admission_reaches_metrics_and_tail_reader(shape):
    draw, compiled = _run(_financial_study(shape=shape))
    mixture = _mixture(((draw, compiled),))
    metrics = mixture.metrics("gross")
    tail = tail_diagnostics(mixture, "gross")["tails"]["0.99"]
    assert metrics["normalized_empirical_mean"] > 0
    assert tail["var"] is not None
    # 64 years leave about 0.04 years above the 99% VaR per batch.
    assert tail["convergence"]["var"]["status"] == "unconverged"
    if shape > 1:
        assert metrics["expected_annual_loss"] is None
        assert metrics["tail_metrics"]["0.99"]["tvar"] is None
        assert metrics["eal_reconciliation_difference"] is None
        assert metrics["eal_precision"]["numerical_standard_error"] is None
        assert metrics["statistics"]["expected_annual_loss"]["status"] == "unavailable"
        assert "finite mean" in metrics["statistics"]["expected_annual_loss"]["reason"]
        assert tail["tvar"] is None
        assert "finite mean" in tail["tvar_interval_unavailable_reason"]
    else:
        assert metrics["expected_annual_loss"] is not None
        assert tail["tvar"] is not None
        # Infinite variance fixes the control-variate coefficient at one, so
        # the TVaR's error is a bounded mean's and its batch half-width is
        # defined; 64 years give 4 per batch.
        assert tail["tvar_estimator"]["coefficient"] == 1.0
        assert tail["tvar_estimator"]["coefficient_basis"] == "infinite_variance"
        assert tail["tvar_numerical_standard_error"] is not None
        assert tail["convergence"]["tvar"]["status"] == "unconverged"
    json.dumps(metrics, allow_nan=False)
    json.dumps(tail, allow_nan=False)


def test_zero_weight_infinite_mean_state_is_reassessed_when_positive():
    payload = encode_scientific(_financial_study(shape=1.2))
    component = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
        "member_outcomes"
    ][0]["components"][0]
    component.update(amount=10, severity=None, rank_group_id=None)
    study = setup_from_payload(payload)
    low, high = _run(study, state="low"), _run(study, state="high")
    before = _mixture(
        (low, high), {low[0].stratum_sha256: 1.0, high[0].stratum_sha256: 0.0}
    )
    after = _mixture(
        (low, high), {low[0].stratum_sha256: 0.5, high[0].stratum_sha256: 0.5}
    )
    assert before.metrics("gross")["expected_annual_loss"] is not None
    assert after.metrics("gross")["expected_annual_loss"] is None
    assert after.metrics("gross")["tail_metrics"]["0.99"]["var"] is not None


@pytest.mark.parametrize("zero", [False, True])
def test_single_independent_year_needs_structural_zero_proof_for_precision(zero):
    draw, compiled = _run(_study(rate=0 if zero else 1), years=1, seed=0)
    mixture = _mixture(((draw, compiled),))
    assert mixture.metrics("gross")["expected_annual_loss"] == (0 if zero else 200)
    mean = mean_diagnostics(mixture, "gross")
    tail = tail_diagnostics(mixture, "gross")["tails"]["0.99"]
    if zero:
        assert mean["empirical_mean_standard_error"] == 0
        assert mean["empirical_mean_numerical_interval_95"] == [0, 0]
        assert tail["tvar_numerical_interval_95"] == [0, 0]
    else:
        assert mean["empirical_mean_standard_error"] is None
        assert "fewer than two independent" in mean["interval_unavailable_reason"]
        assert tail["var_numerical_interval_95"] is None
        assert tail["tvar_numerical_interval_95"] is None


def test_positive_rare_origin_without_sampled_losses_has_no_zero_width_precision():
    draw, compiled = _run(_study(rate=1e-200), years=16)
    mixture = _mixture(((draw, compiled),))
    assert any(view.native_mean is not None for view in draw.views.values())
    assert draw.views["gross"].native_mean.value > 0
    assert not np.any(draw.annual_losses["gross"])
    gate = view_gate(mixture, "gross", {}, TailGateControls(), mean_exists=True)
    assert "No positive loss" in gate["unavailable_reason"]
    tail = tail_diagnostics(mixture, "gross")["tails"]["0.99"]
    assert tail["var"] == 0
    assert tail["var_numerical_interval_95"] is None
    assert tail["tvar_numerical_interval_95"] is None
