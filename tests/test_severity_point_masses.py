"""Degenerate severity laws keep their exact point mass."""

from dataclasses import replace

import numpy as np
import pytest

from catmodel.probability.distributions import Distribution

_KNOTS = ("p5", "p25", "p50", "p75", "p95")


@pytest.mark.parametrize("value", [0.0, 0.37, 1.0])
def test_point_mass_preserves_the_represented_law(value: float) -> None:
    distribution = Distribution(value, value, value, value, value)
    uniforms = np.array([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])

    for law in (distribution, replace(distribution)):
        np.testing.assert_array_equal(
            law.sample_from_uniforms(uniforms), np.full(uniforms.shape, value)
        )
        np.testing.assert_array_equal(
            law.sample_from_uniforms_uncapped(uniforms), np.full(uniforms.shape, value)
        )
        np.testing.assert_array_equal(
            law.sample_from_uniforms(np.random.default_rng(17).random(32)),
            np.full(32, value),
        )
        assert [law.quantile(float(p)) for p in uniforms] == [value] * len(uniforms)
        assert law.analytic_mean() == pytest.approx(value, rel=1e-14, abs=0.0)
        assert law.analytic_mean(apply_severity_cap=False) == pytest.approx(
            value, rel=1e-14, abs=0.0
        )

        # Strict exceedance makes the corresponding CDF right-continuous at the atom.
        assert law.survival_probability(float(np.nextafter(value, -np.inf))) == 1.0
        assert law.survival_probability(value) == 0.0
        assert law.survival_probability(float(np.nextafter(value, np.inf))) == 0.0


@pytest.mark.parametrize("value", [0.0, 0.37, 1.0])
def test_constant_laws_preserve_their_quantity_constraints(value: float) -> None:
    law = Distribution(p5=value, p25=value, p50=value, p75=value, p95=value)
    np.testing.assert_array_equal(
        law.sample_from_uniforms(np.array([0.0, 0.5, 1.0])), [value] * 3
    )
