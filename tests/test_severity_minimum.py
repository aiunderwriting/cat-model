"""Numerical and native-runtime oracles for fixed qualifying severity support."""

from dataclasses import replace

import numpy as np
import pytest

from catmodel.probability.distributions import (
    Distribution,
    MixtureSeverityDistribution,
    apply_severity_excess_scale,
    severity_minimum,
)


def _law(*, cap=None, **tail):
    return Distribution(
        120.0,
        160.0,
        220.0,
        350.0,
        800.0,
        gpd_shape=0.25,
        severity_cap=cap,
        severity_minimum=100.0,
        severity_minimum_strict=True,
        **tail,
    )


@pytest.mark.parametrize("strict", [False, True])
def test_positive_floor_continuation_preserves_all_knots_without_boundary_atom(strict):
    law = Distribution(
        1_000_001.0,
        50_000_000.0,
        100_000_000.0,
        200_000_000.0,
        500_000_000.0,
        severity_minimum=1_000_000.0,
        severity_minimum_strict=strict,
    )
    u = np.array([0.0, 1e-16, 1e-6, 0.001, 0.025, 0.049, 0.05, 0.25, 0.5, 0.75, 0.95])
    values = law.sample_from_uniforms(u)
    assert values[0] == 1_000_000.0
    assert np.all(values[1:] > 1_000_000.0)
    assert np.all(np.diff(values) >= 0)
    assert np.array_equal(values[-5:], [law.p5, law.p25, law.p50, law.p75, law.p95])
    assert law.survival_probability(1_000_000.0) == 1.0
    assert law.survival_probability(1_000_000.5) == pytest.approx(0.975)


def test_inclusive_quantile_at_floor_retains_authored_atom():
    law = Distribution(100.0, 160.0, 220.0, 350.0, 800.0, severity_minimum=100.0)
    assert np.all(
        law.sample_from_uniforms(np.array([0.0, 0.001, 0.049, 0.05])) == 100.0
    )
    assert law.survival_probability(100.0) == pytest.approx(0.95)
    with pytest.raises(ValueError, match="strictly above"):
        replace(law, severity_minimum_strict=True)


@pytest.mark.parametrize(
    "changes",
    [
        {"severity_minimum": float("nan")},
        {"severity_minimum": -1.0},
        {"severity_minimum": 121.0},
        {"severity_minimum": None, "severity_minimum_strict": True},
        {"severity_cap": 100.0},
    ],
)
def test_support_rejects_invalid_parameters_without_changing_answers(changes):
    with pytest.raises(ValueError):
        replace(_law(), **changes)


@pytest.mark.parametrize("factor", [0.6, 1.0, 2.0])
@pytest.mark.parametrize(
    "tail",
    [{}, {"gpd_tail_ratio": 2.5}, {"gpd_scale": 60.0}],
)
def test_excess_scaling_preserves_floor_exact_law_and_cdf(tail, factor):
    law = _law(**tail)
    mean = law.analytic_mean()
    # The excess scale that multiplies the mean by ``factor``.
    k = (factor * mean - 100) / (mean - 100)
    scaled = law.with_excess_scale(k)
    u = np.array([0.0, 0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
    assert scaled.sample_from_uniforms(u) == pytest.approx(
        100 + k * (law.sample_from_uniforms(u) - 100), rel=1e-10, abs=1e-9
    )
    assert scaled.analytic_mean() == pytest.approx(factor * mean, rel=2e-9)
    assert severity_minimum(scaled) == 100
    assert scaled.survival_probability(100) == 1
    for probability in [0.01, 0.3, 0.7, 0.99]:
        assert scaled.survival_probability(
            scaled.quantile(probability)
        ) == pytest.approx(1 - probability, abs=1e-10)
    again = scaled.with_excess_scale(1 / k)
    assert again.sample_from_uniforms(u) == pytest.approx(
        law.sample_from_uniforms(u), rel=1e-10
    )


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_excess_scale_must_be_finite_and_positive(scale):
    for law in (_law(), MixtureSeverityDistribution([_law()], np.array([1.0]))):
        with pytest.raises(ValueError, match="finite and positive"):
            law.with_excess_scale(scale)


def test_tail_copies_and_mixture_preserve_floor():
    law = _law(gpd_scale=60.0).with_excess_scale(1.5)
    for copied in [
        law.with_gpd_shape(law.gpd_shape + 0.1),
    ]:
        assert severity_minimum(copied) == 100
        assert copied.quantile(0) == pytest.approx(100)
        assert copied.survival_probability(100) == pytest.approx(1)
    mix = MixtureSeverityDistribution(
        [_law(), _law(gpd_scale=60.0)], np.array([0.3, 0.7])
    )
    scaled = mix.with_excess_scale(0.8)
    assert scaled.analytic_mean() == pytest.approx(
        100 + 0.8 * (mix.analytic_mean() - 100), rel=1e-9
    )
    assert severity_minimum(scaled) == 100


@pytest.mark.parametrize("factor", [0.7, 1.05])
def test_common_draws_and_native_mean_use_same_floor_mapping(factor):
    from catmodel.simulation.incident_runtime import simulate_incident_state
    from test_support.incident import common_case as _case

    law = _law(cap=500.0)
    mean = law.analytic_mean(apply_severity_cap=False)
    slope = (factor * mean - 100) / (mean - 100)
    transformed = law.with_excess_scale(slope)
    _unused_study, plan, compiled, _ = _case(
        sampler=transformed, rate=1, simulations=10_000
    )
    plan = plan.model_copy(update={"requested_views": ("gross", "oep_gross")})
    draws, unavailable, _ = simulate_incident_state(
        plan, compiled, response_sha256="a" * 64
    )
    assert not unavailable
    maximum = draws.annual_losses["oep_gross"]
    assert np.all(maximum[maximum > 0] > 100)
    assert np.max(maximum) <= 500
    expected = 100 + slope * (
        law.analytic_mean(
            apply_severity_cap=False,
            severity_cap_override=100 + 400 / slope,
        )
        - 100
    )
    assert any(view.native_mean is not None for view in draws.views.values())
    assert draws.views["gross"].native_mean is not None
    assert draws.views["gross"].native_mean.value == pytest.approx(expected, rel=1e-12)
    losses = draws.annual_losses["gross"]
    standard_error = np.std(losses, ddof=1) / np.sqrt(len(losses))
    assert abs(losses.mean() - expected) < 5 * standard_error


def test_excess_scale_preserves_a_small_positive_distance_from_the_floor():
    # An extreme affine offset must not cancel a small but positive excess.
    near = Distribution(
        1_000_001.0,
        2_000_000.0,
        3_000_000.0,
        4_000_000.0,
        5_000_000.0,
        severity_minimum=1_000_000.0,
        severity_minimum_strict=True,
    )
    values = np.array([np.nextafter(1_000_000.0, np.inf), 1_000_001.0])
    projected = apply_severity_excess_scale(near, values, 1e-8)
    assert np.all(projected > 1_000_000.0)
    assert projected[1] == pytest.approx(1_000_000.0 + 1e-8)


def test_zero_weight_component_cannot_change_active_population_floor():
    active = _law()
    ignored = Distribution(1.0, 2.0, 3.0, 4.0, 5.0)
    mix = MixtureSeverityDistribution([active, ignored], np.array([1.0, 0.0]))
    assert severity_minimum(mix) == 100.0
    assert mix.severity_minimum_strict
    k = 0.1
    values = apply_severity_excess_scale(
        mix, active.sample_from_uniforms(np.array([0.001, 0.5, 0.99])), k
    )
    assert np.all(values > 100.0)


def test_native_mixture_rejects_incompatible_active_qualifying_populations():
    with pytest.raises(ValueError, match="authored qualifying minimum"):
        MixtureSeverityDistribution(
            [_law(), Distribution(1.0, 2.0, 3.0, 4.0, 5.0)], np.array([0.5, 0.5])
        )
    with pytest.raises(ValueError, match="authored qualifying minimum"):
        MixtureSeverityDistribution(
            [_law(), replace(_law(), severity_minimum_strict=False)],
            np.array([0.5, 0.5]),
        )
