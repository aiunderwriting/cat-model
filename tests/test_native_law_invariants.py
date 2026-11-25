"""Independent law and transform oracles, including atoms and deep tails."""

import math
from itertools import pairwise

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import genpareto

from catmodel.probability.distributions import (
    Distribution,
    MixtureSeverityDistribution,
    sample_affine_capped_severity,
)


@pytest.mark.parametrize("shape", [0.49, 0.5, 0.51, 0.95, 1.0, 1.2])
def test_duplicate_and_zero_weight_components_preserve_atoms_and_raw_tail(shape):
    law = Distribution(1, 2, 4, 8, 16, gpd_shape=shape, gpd_scale=1000)
    inactive = Distribution(1, 1, 1, 11, 11)
    mixture = MixtureSeverityDistribution(
        [inactive, law, law], np.array([0.0, 0.25, 0.75])
    )
    probabilities = np.array([0.0, 0.5, 0.95, 0.975, 1 - 1e-13, np.nextafter(1.0, 0.0)])
    np.testing.assert_array_equal(
        mixture.sample_from_uniforms(probabilities),
        law.sample_from_uniforms(probabilities),
    )
    assert mixture.analytic_mean(severity_cap_override=50) == pytest.approx(
        law.analytic_mean(severity_cap_override=50), rel=2e-15
    )
    if shape < 1.0:
        assert mixture.analytic_mean() == pytest.approx(law.analytic_mean(), rel=2e-15)


def test_mixture_atom_and_gap_use_generalized_inverse():
    law = Distribution(1, 1, 1, 11, 11)
    mixture = MixtureSeverityDistribution(
        [law, Distribution(3, 3, 3, 3, 3)], np.array([0.5, 0.5])
    )
    assert law.survival_probability(1) == pytest.approx(0.5)
    assert mixture.quantile(0.25) == pytest.approx(1, abs=2e-14)
    assert mixture.quantile(0.5) == pytest.approx(3, abs=2e-14)
    assert mixture.quantile(0.875) == pytest.approx(11, abs=2e-14)
    for threshold in (1, 3, np.nextafter(11.0, 0.0), 11):
        assert mixture.survival_probability(threshold) == pytest.approx(
            0.5 * law.survival_probability(threshold) + 0.5 * (threshold < 3)
        )


def _gpd_law(shape, scale, location):
    """A quantile-spline law through a shifted GPD's knots, with its shape."""
    probabilities = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    knots = genpareto.ppf(probabilities, c=shape, loc=location, scale=scale)
    return Distribution(*knots, gpd_shape=shape)


@pytest.mark.parametrize("probability", [0.01, 0.25, 0.5, 0.99, 1 - 1e-13])
def test_heterogeneous_mixture_inverse_matches_independent_survival_root(probability):
    first = _gpd_law(0.6, 2.0, 1.0)
    second = _gpd_law(0.9, 7.0, 2.0)
    mixture = MixtureSeverityDistribution([first, second], np.array([0.2, 0.8]))
    target = 1.0 - probability

    # A scalar root of the components' own survival functions, independently
    # of the mixture's vectorized inverse.
    def oracle(x):
        return (
            0.2 * first.survival_probability(x)
            + 0.8 * second.survival_probability(x)
            - target
        )

    lo, hi = first.quantile(probability), second.quantile(probability)
    expected = brentq(oracle, min(lo, hi), max(lo, hi), xtol=1e-14, rtol=2e-14)
    assert mixture.quantile(probability) == pytest.approx(expected, rel=3e-13)


@pytest.mark.parametrize("units", [1e-6, 0.01, 100.0])
def test_currency_conversion_scales_the_entire_law(units):
    first = Distribution(
        1, 2, 4, 8, 16, gpd_shape=0.8, severity_cap=100, severity_minimum=0.5
    )
    second = Distribution(2, 4, 8, 16, 32, gpd_shape=0.6, severity_minimum=0.5)
    mixture = MixtureSeverityDistribution(
        [first, second], np.array([0.4, 0.6]), severity_cap=80
    )
    converted = mixture.with_currency_scale(units)
    probabilities = np.array([0.0, 0.25, 0.5, 0.99, 1 - 1e-13])
    np.testing.assert_allclose(
        converted.sample_from_uniforms(probabilities) / units,
        mixture.sample_from_uniforms(probabilities),
        rtol=5e-13,
    )
    assert converted.severity_cap == pytest.approx(80 * units)
    assert converted.components[0].severity_cap == pytest.approx(100 * units)
    assert converted.components[0].severity_minimum == pytest.approx(0.5 * units)
    assert converted.analytic_mean() / units == pytest.approx(
        mixture.analytic_mean(), rel=2e-12
    )


def test_outer_cap_does_not_cap_heavy_components_before_response():
    sampler = MixtureSeverityDistribution(
        [Distribution(1, 2, 4, 8, 16, gpd_shape=1.5, gpd_scale=1000)],
        np.array([1.0]),
        severity_cap=100,
    )
    raw = sampler.sample_from_uniforms_uncapped(np.array([0.975]))[0]
    assert raw == pytest.approx(16 + genpareto.ppf(0.5, c=1.5, scale=1000))
    assert min(raw * 0.5, 100) == 100
    assert 0.5 * min(raw, 100) == 50
    expected = sampler.components[0].analytic_mean(severity_cap_override=200) * 0.5
    law = sampler.components[0]
    integral = sum(
        quad(
            lambda u: min(
                float(law.sample_from_uniforms_uncapped(np.array([u]))[0]), 200
            ),
            a,
            b,
            epsabs=0,
            epsrel=1e-12,
            limit=200,
        )[0]
        for a, b in pairwise((0, 0.05, 0.25, 0.5, 0.75, 0.95, 1))
    )
    assert expected == pytest.approx(0.5 * integral, rel=1e-10)


@pytest.mark.parametrize("minimum", [0.0, 1.0])
def test_capped_heavy_excess_scale_needs_no_raw_mean(minimum):
    law = Distribution(
        2,
        3,
        4,
        8,
        16,
        gpd_shape=1.2,
        gpd_scale=1000,
        severity_minimum=minimum,
        severity_cap=10000,
    )
    # The excess scale that doubles the median.
    transformed = law.with_excess_scale((8 - minimum) / (4 - minimum))
    assert transformed.p50 == pytest.approx(8)
    assert math.isfinite(transformed.analytic_mean())
    with pytest.raises(ValueError, match="uncapped GPD tail mean"):
        law.analytic_mean(apply_severity_cap=False)
    assert law.with_excess_scale(1).sample_from_uniforms(
        np.array([0.05, 0.5, 0.99])
    ) == pytest.approx(law.sample_from_uniforms(np.array([0.05, 0.5, 0.99])))


def test_deterministic_mixture_endpoint_atoms():
    mixture = MixtureSeverityDistribution(
        [Distribution(1, 1, 1, 1, 1), Distribution(3, 3, 3, 3, 3)],
        np.array([0.9, 0.1]),
    )
    probabilities = np.array([0.0, 0.5, 0.9, np.nextafter(0.9, 1.0), 1.0])
    np.testing.assert_allclose(
        mixture.sample_from_uniforms(probabilities), [1, 1, 1, 3, 3], atol=2e-14
    )
    assert Distribution(0, 0, 0, 0, 0).analytic_mean() == 0.0


def test_raw_tail_keeps_its_endpoint_and_does_not_clip_remote_quantiles():
    law = Distribution(1, 2, 4, 8, 16, gpd_shape=0.95, gpd_scale=1000)
    p = np.nextafter(1.0, 0.0)
    expected = 16 + genpareto.ppf(1 - (1 - p) / 0.05, c=0.95, scale=1000)
    assert law.quantile(p) == pytest.approx(expected, rel=2e-14)
    assert law.quantile(1.0) == math.inf
    bounded = Distribution(1, 2, 4, 8, 16, gpd_shape=-0.5, gpd_scale=1000)
    assert bounded.quantile(1.0) == 2016


def test_physical_response_transforms_intrinsic_component_support_before_outer_cap():
    component = Distribution(1, 2, 4, 8, 16, gpd_shape=0.4, severity_cap=100)
    mixture = MixtureSeverityDistribution(
        [component], np.array([1.0]), severity_cap=200
    )
    transformed = mixture.with_excess_scale(2)
    probabilities = np.array([0.5, 0.99, 0.999999])
    np.testing.assert_allclose(
        transformed.sample_from_uniforms(probabilities),
        np.minimum(2 * component.sample_from_uniforms(probabilities), 200),
    )
    assert transformed.components[0].severity_cap == 200


def test_the_heaviest_admitted_tail_stays_finite_at_every_interior_rank():
    law = Distribution(1, 2, 4, 8, 16, gpd_shape=5.0, gpd_scale=1e15)
    assert law.quantile(0.5) == 4
    assert np.isfinite(law.quantile(np.nextafter(1.0, 0.0)))
    capped = Distribution(
        1, 2, 4, 8, 16, gpd_shape=5.0, gpd_scale=1e3, severity_cap=100
    )
    assert capped.quantile(0.99) == 100


def test_authored_inclusive_zero_floor_retains_its_atom():
    law = Distribution(0, 2, 4, 8, 16, severity_minimum=0)
    assert law.survival_probability(0) == pytest.approx(0.95)
    assert law.cdf(np.array([0.0]))[0] == pytest.approx(0.05)
    mixture = MixtureSeverityDistribution([law, law], np.array([0.5, 0.5]))
    assert mixture.quantile(0.025) == 0
    assert mixture.survival_probability(0) == pytest.approx(0.95)


@pytest.mark.parametrize("multiplier", [0.5, 1.0, 2.0])
def test_capped_mixture_resolves_ranks_above_its_cap(multiplier):

    law = MixtureSeverityDistribution(
        [Distribution(1, 2, 3, 4, 5, gpd_shape=2.5, gpd_scale=1e15)],
        np.array([1.0]),
        severity_cap=100,
    )
    assert law.quantile(1 - 1e-10) == 100
    assert (
        sample_affine_capped_severity(law, np.array([1 - 1e-10]), scale=multiplier)[0]
        == 100
    )


def test_affine_inverse_preserves_strict_floor_under_rounding():

    minimum = 1.0
    law = Distribution(
        *(np.nextafter(minimum, np.inf) for _ in range(5)),
        severity_minimum=minimum,
        severity_minimum_strict=True,
    )
    slope = 1e-10
    result = sample_affine_capped_severity(
        law, np.array([0.5]), scale=slope, offset=minimum * (1 - slope)
    )
    assert result[0] == np.nextafter(minimum, np.inf)


def test_cap_preclassification_preserves_an_upper_probability_gap_boundary():
    law = MixtureSeverityDistribution(
        [Distribution(1, 1, 1, 1, 1), Distribution(3, 3, 3, 3, 3)],
        np.array([0.9999, 0.0001]),
        severity_cap=2,
    )
    assert law.quantile(0.9999) == 1
    assert law.quantile(np.nextafter(0.9999, 1.0)) == 2
