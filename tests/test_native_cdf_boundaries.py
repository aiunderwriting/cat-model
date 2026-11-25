"""Independent inverse inequalities at authored knots, atoms, splices and caps."""

from dataclasses import replace

import numpy as np
import pytest

from catmodel.probability.distributions import (
    Distribution,
    MixtureSeverityDistribution,
    sample_affine_capped_severity,
)
from test_support.severity_oracles import SplineOracle


def _inverse_cdf_oracle(law, threshold):
    knots = [law.p5, law.p25, law.p50, law.p75, law.p95]
    if knots[0] <= threshold < knots[-1]:
        return SplineOracle(knots).cdf(threshold)
    lower, upper = 0.0, 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2
        if law.quantile(midpoint) <= threshold:
            lower = midpoint
        else:
            upper = midpoint
    return lower


@pytest.mark.parametrize("units", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("gpd", [False, True])
def test_authored_knots_and_adjacent_amounts_have_valid_cdf_brackets(units, gpd):
    law = (
        Distribution(3.9e5, 1.5e6, 2.6e6, 5.5e6, 1.3e7, gpd_shape=0.4)
        if gpd
        else Distribution(0.1, 0.1, 0.1, 0.3, 1.0)
    ).with_currency_scale(units)
    knots = np.array([law.p5, law.p25, law.p50, law.p75, law.p95])
    amounts = np.sort(
        np.concatenate(
            (np.nextafter(knots, -np.inf), knots, np.nextafter(knots, np.inf))
        )
    )
    expected = [_inverse_cdf_oracle(law, x) for x in amounts]
    np.testing.assert_allclose(law.cdf(amounts), expected, rtol=0, atol=4e-15)
    np.testing.assert_allclose(
        law.survival(amounts), 1 - np.array(expected), rtol=0, atol=4e-15
    )
    assert np.all(np.diff(law.cdf(amounts)) >= 0)
    if gpd:
        assert law.quantile(0.95) == law.p95


@pytest.mark.parametrize("units", [0.01, 1.0, 100.0])
def test_left_right_atom_masses_survive_mixture_and_cap_transforms(units):
    law = Distribution(1, 1, 1, 11, 11).with_currency_scale(units)
    mixture = MixtureSeverityDistribution(
        [law, Distribution(3, 3, 3, 3, 3).with_currency_scale(units)],
        np.array([0.4, 0.6]),
    )
    for threshold, expected in ((1 * units, 0.5), (11 * units, 1.0)):
        assert law.cdf(np.array([threshold]))[0] == expected
        left = np.nextafter(threshold, -np.inf)
        assert law.cdf(np.array([left]))[0] == pytest.approx(
            _inverse_cdf_oracle(law, left), abs=4e-15
        )
    ranks = np.array([0, 0.1, 0.2, np.nextafter(0.2, 1), 0.5, 0.8, 0.9, 0.999, 1])
    draws = mixture.sample_from_uniforms(ranks)
    assert np.all(np.diff(draws) >= 0)
    for probability, amount in zip(ranks, draws, strict=True):
        cdf = 0.4 * _inverse_cdf_oracle(law, amount) + 0.6 * (amount >= 3 * units)
        assert cdf >= probability - 4e-15
    capped = replace(law, severity_cap=3 * units)
    assert capped.cdf(np.array([3 * units]))[0] == 1
    assert capped.cdf(np.array([np.nextafter(3 * units, -np.inf)]))[0] < 1
    np.testing.assert_allclose(
        sample_affine_capped_severity(capped, ranks, scale=0.5),
        np.minimum(0.5 * law.sample_from_uniforms_uncapped(ranks), 3 * units),
        rtol=1e-14,
    )


@pytest.mark.parametrize("units", [0.01, 1.0, 100.0])
def test_cap_at_realistic_p95_remains_evaluable_before_and_after_scaling(units):
    raw = Distribution(
        3.9e5, 1.5e6, 2.6e6, 5.5e6, 1.3e7, gpd_shape=0.4
    ).with_currency_scale(units)
    law = replace(raw, severity_cap=raw.p95)
    assert law.severity_cap is not None
    ranks = np.array([0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
    for scale in (0.5, 1.0, 2.0):
        np.testing.assert_allclose(
            sample_affine_capped_severity(law, ranks, scale=scale),
            np.minimum(
                scale * raw.sample_from_uniforms_uncapped(ranks), law.severity_cap
            ),
            rtol=1e-14,
        )
    assert np.isfinite(law.analytic_mean())
    mixture = MixtureSeverityDistribution(
        [law, replace(law, severity_cap=2 * raw.p95)],
        np.array([0.5, 0.5]),
        severity_cap=raw.p95,
    )
    assert np.all(np.isfinite(mixture.sample_from_uniforms(ranks)))


@pytest.mark.parametrize("units", [0.01, 1.0, 100.0])
def test_dense_interior_cdf_inverts_spline_body_not_continuation_slope(units):
    law = Distribution(
        390000, 1500000, 2600000, 5500000, 13000000, gpd_shape=0.4
    ).with_currency_scale(units)
    probabilities = np.unique(np.r_[np.linspace(0.050001, 0.949999, 10001), 0.9])
    amounts = law.sample_from_uniforms_uncapped(probabilities)
    np.testing.assert_allclose(law.cdf(amounts), probabilities, rtol=0, atol=2e-14)
    np.testing.assert_allclose(
        law.survival(amounts), 1 - probabilities, rtol=0, atol=2e-14
    )
    mixture = MixtureSeverityDistribution(
        [law, law.with_currency_scale(1.7)], np.array([0.4, 0.6])
    )
    ranks = np.linspace(0.050001, 0.949999, 101)
    draws = mixture.sample_from_uniforms_uncapped(ranks)
    cdf = 0.4 * law.cdf(draws) + 0.6 * mixture.components[1].cdf(draws)
    np.testing.assert_allclose(cdf, ranks, rtol=0, atol=2e-14)


@pytest.mark.parametrize("units", [0.01, 1.0, 100.0])
def test_q90_cap_does_not_censor_lower_native_or_mixture_quantiles(units):
    law = Distribution(
        390000, 1500000, 2600000, 5500000, 13000000, gpd_shape=0.4
    ).with_currency_scale(units)
    ranks = np.array([0.89999, 0.899999, 0.8999995, 0.8999999, 0.9, 0.9000005])
    for sampler in (
        law,
        MixtureSeverityDistribution(
            [law, law.with_currency_scale(1.7)], np.array([0.4, 0.6])
        ),
    ):
        cap = sampler.quantile(0.9)
        capped = replace(sampler, severity_cap=cap)
        expected = np.minimum(sampler.sample_from_uniforms_uncapped(ranks), cap)
        actual = sample_affine_capped_severity(capped, ranks)
        np.testing.assert_allclose(
            actual, expected, rtol=8 * np.finfo(float).eps, atol=0
        )
        assert np.all(actual[ranks < 0.9] < cap)
