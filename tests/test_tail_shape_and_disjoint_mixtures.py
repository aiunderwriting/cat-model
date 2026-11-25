"""Tail-shape transforms and pooled inverses of disjoint mixture components."""

import numpy as np
import pytest

from catmodel.probability.distributions import (
    Distribution,
    MixtureSeverityDistribution,
)


def test_absolute_shape_change_preserves_raw_body_explicit_scale_and_cap():
    law = Distribution(1, 2, 4, 8, 16, gpd_shape=0.9, gpd_scale=1000, severity_cap=100)
    changed = law.with_gpd_shape(1.4)
    assert changed.gpd_shape == 1.4
    assert changed.effective_gpd_scale == law.effective_gpd_scale
    assert changed.severity_cap == 100
    ranks = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    np.testing.assert_array_equal(
        changed.sample_from_uniforms_uncapped(ranks),
        law.sample_from_uniforms_uncapped(ranks),
    )
    assert np.isfinite(changed.analytic_mean())
    with pytest.raises(ValueError, match="uncapped GPD tail mean"):
        changed.analytic_mean(apply_severity_cap=False)


def test_absolute_shape_change_retains_p99_anchor_and_named_menu_limits():
    law = Distribution(1, 2, 4, 8, 16, gpd_shape=0.9, gpd_tail_ratio=3)
    changed = law.with_gpd_shape(1.2)
    assert changed.quantile(0.99) / changed.quantile(0.95) == pytest.approx(3)
    assert changed.gpd_shape == 1.2


def test_absolute_shape_change_holds_an_explicit_scale():
    ranks = np.array([0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    law = Distribution(1, 2, 4, 8, 16, gpd_shape=0.6, gpd_scale=7)
    changed = law.with_gpd_shape(1.2)
    assert changed.effective_gpd_scale == 7
    # The knots stay; above p95 the tail is a GPD of shape 1.2 and scale 7.
    np.testing.assert_allclose(
        changed.sample_from_uniforms(ranks),
        [1, 2, 4, 8, 16, 16 + 7 / 1.2 * ((0.01 / 0.05) ** -1.2 - 1)],
        rtol=2e-14,
    )
    mixture = MixtureSeverityDistribution(
        [law, Distribution(1, 2, 3, 4, 5)], np.array([0.6, 0.4]), severity_cap=100
    )
    transformed = mixture.with_gpd_shape(1.2)
    assert transformed.components[0].gpd_shape == 1.2
    assert transformed.components[1] is mixture.components[1]
    assert transformed.severity_cap == 100


@pytest.mark.parametrize("units", [0.01, 1.0, 100.0])
def test_disjoint_inverse_preserves_left_rank_boundaries_atoms_and_roster(units):
    low = Distribution(0.05, 0.25, 0.5, 0.75, 0.95).with_currency_scale(units)
    atom = Distribution(2, 2, 2, 2, 2).with_currency_scale(units)
    high = Distribution(10.05, 10.25, 10.5, 10.75, 10.95).with_currency_scale(units)
    ranks = np.array(
        [0, 0.1, 0.25, np.nextafter(0.25, 1), 0.4, 0.5, np.nextafter(0.5, 1), 0.75, 1]
    )
    expected = np.array(
        [
            low.quantile(0),
            low.quantile(0.4),
            low.quantile(1),
            2 * units,
            2 * units,
            2 * units,
            high.quantile(0),
            high.quantile(0.5),
            high.quantile(1),
        ]
    )
    one = MixtureSeverityDistribution([low, atom, high], np.array([0.25, 0.25, 0.5]))
    two = MixtureSeverityDistribution(
        [high, low, atom, low, high], np.array([0.5, 0.1, 0.25, 0.15, 0])
    )
    np.testing.assert_allclose(
        one.sample_from_uniforms_uncapped(ranks),
        expected,
        rtol=1e-13,
        atol=1e-14 * units,
    )
    np.testing.assert_array_equal(
        one.sample_from_uniforms_uncapped(ranks),
        two.sample_from_uniforms_uncapped(ranks),
    )
    assert one.quantile(0.25) == pytest.approx(low.quantile(1))
    assert one.quantile(0.5) == 2 * units


def test_disjoint_inverse_retains_a_tiny_positive_upper_support_and_mean():
    law = MixtureSeverityDistribution(
        [Distribution(1, 1, 1, 1, 1), Distribution(1e12, 1e12, 1e12, 1e12, 1e12)],
        np.array([1.0, 1e-12]),
    )
    assert law.quantile(1 - 1e-11) == 1
    assert law.quantile(1) == 1e12
    assert law.analytic_mean() == pytest.approx(2)


def test_touching_intrinsic_supports_preserve_the_pooled_inverse():
    upper = Distribution(10, 20, 30, 40, 50)
    touch = upper.quantile(0)
    lower = Distribution(1, 2, 3, 4, 5, gpd_shape=0.3, severity_cap=touch)
    assert lower.quantile(1) == touch
    law = MixtureSeverityDistribution([lower, upper], np.array([0.5, 0.5]))
    ranks = np.array([0, 0.25, 0.5, np.nextafter(0.5, 1), 0.75, 1])
    expected = np.r_[
        lower.sample_from_uniforms(2 * ranks[:3]),
        upper.sample_from_uniforms(2 * ranks[3:] - 1),
    ]
    np.testing.assert_allclose(
        law.sample_from_uniforms_uncapped(ranks), expected, rtol=1e-13, atol=1e-14
    )
