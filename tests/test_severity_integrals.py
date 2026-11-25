"""Survival-integral oracles for native floor-preserving limited means."""

from dataclasses import replace

import numpy as np
import pytest

from catmodel.probability.distributions import Distribution


def _gpd_law(shape=0.75, scale=10.0, threshold=1.0, cap=None):
    """A quantile-spline law through the knots of threshold+GPD(shape,scale)."""

    def q(p):
        return threshold + scale / shape * ((1 - p) ** (-shape) - 1)

    return Distribution(
        p5=q(0.05),
        p25=q(0.25),
        p50=q(0.5),
        p75=q(0.75),
        p95=q(0.95),
        gpd_shape=shape,
        gpd_scale=scale,
        severity_cap=cap,
    )


def _survival_integral(law, factor, limit):
    """E[min(aX, limit)] by integrating the law's survival function, whose
    knots are breakpoints, independently of Distribution.analytic_mean."""
    from scipy.integrate import quad

    knots = [law.p5, law.p25, law.p50, law.p75, law.p95]
    return quad(
        lambda y: law.survival_probability(y / factor),
        0,
        limit,
        points=[factor * knot for knot in knots if factor * knot < limit],
        limit=200,
        epsabs=0,
        epsrel=1e-12,
    )[0]


@pytest.mark.parametrize("shape", [0.75, 1.2])
def test_conditional_mean_preserves_multiplier_before_outer_cap(shape):
    cap = 100.0
    law = _gpd_law(shape, cap=cap)
    factors = np.array([0.25, 1.0, 2.0])
    raw = replace(law, severity_cap=None)
    actual = [
        factor * raw.analytic_mean(severity_cap_override=cap / factor)
        for factor in factors
    ]
    expected = [_survival_integral(raw, factor, cap) for factor in factors]
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


@pytest.mark.parametrize("by_override", [False, True])
@pytest.mark.parametrize("has_tail", [False, True])
def test_limited_mean_across_the_tail_threshold(has_tail, by_override):
    shape, scale = 0.75, 10.0
    raw = _gpd_law(shape, scale)
    if not has_tail:
        raw = replace(raw, gpd_shape=None, gpd_scale=None)
    threshold = raw.p95

    def limited_mean(cap):
        if by_override:
            return raw.analytic_mean(severity_cap_override=cap)
        return replace(raw, severity_cap=cap).analytic_mean()

    for cap in (0.01 * threshold, 0.4 * threshold, threshold, 3.0 * threshold):
        assert limited_mean(cap) == pytest.approx(
            _survival_integral(raw, 1.0, cap), rel=1e-10
        )

    # The formulas for a cap inside the body and inside the tail meet at the
    # threshold.
    for neighbour in (np.nextafter(threshold, 0.0), np.nextafter(threshold, np.inf)):
        assert limited_mean(float(neighbour)) == pytest.approx(
            limited_mean(threshold), rel=1e-13
        )

    if has_tail:
        # Without a cap the tail contributes its closed-form GPD mean.
        uncapped = _survival_integral(raw, 1.0, threshold) + 0.05 * scale / (1 - shape)
        assert raw.analytic_mean() == pytest.approx(uncapped, rel=1e-10)


def test_conditional_mean_preserves_intrinsic_mixture_caps_before_scaling():
    from catmodel.probability.distributions import MixtureSeverityDistribution

    component = _gpd_law(cap=20.0)
    mixture = MixtureSeverityDistribution(
        components=[component], weights=np.array([1.0]), severity_cap=30.0
    )
    for factor in [0.5, 2.0]:
        expected = _survival_integral(component, 1.0, min(20, 30 / factor))
        actual = replace(mixture, severity_cap=None).analytic_mean(
            severity_cap_override=30 / factor
        )
        assert actual is not None
        assert factor * actual == pytest.approx(factor * expected, rel=1e-10)


def test_uncapped_tail_cannot_invent_a_finite_mean():
    law = Distribution(1, 2, 4, 8, 16, gpd_shape=1.2, gpd_scale=1000)
    with pytest.raises(ValueError, match="uncapped GPD tail mean"):
        law.analytic_mean()
    # A limited mean needs no uncapped moment.
    assert 0 < law.analytic_mean(severity_cap_override=1e6) < 1e6


def test_zero_affine_scale_needs_no_raw_moment():
    from catmodel.probability.distributions import sample_affine_capped_severity

    law = Distribution(
        101, 102, 104, 108, 116, gpd_shape=1.2, gpd_scale=1000, severity_minimum=100
    )
    np.testing.assert_array_equal(
        sample_affine_capped_severity(
            law, np.array([0.0, 0.5, 1.0]), scale=0, offset=100
        ),
        [100, 100, 100],
    )
