import math

import numpy as np
import pytest
from scipy.stats import norm

from catmodel.probability.distributions import Distribution
from test_support.severity_oracles import SCORES, SplineOracle

BODY = dict(p5=1.0, p25=5.0, p50=20.0, p75=60.0, p95=200.0)


class TestExplicitScaleConstruction:
    @pytest.mark.parametrize("shape", [-0.3, 0.0, 0.3, 0.8])
    def test_default_scale_matches_the_body_density_at_the_splice(self, shape):
        """Without an explicit or anchored scale, the tail density matches the body.

        The body is log q against the normal score z, so the oracle takes SciPy's
        analytic PCHIP slope of log q at z(p95), sets
        ``sigma = 0.05 q(p95) dlogq/dz / phi(z)`` and evaluates the closed-form
        GPD quantile ``p95 + sigma g(q)``, ``q = (p - 0.95) / 0.05``, with
        ``g(q) = ((1 - q)^-xi - 1) / xi`` (``-log(1 - q)`` at ``xi = 0``).
        """
        distribution = Distribution(
            severity_minimum_strict=False, **BODY, gpd_shape=shape
        )
        assert distribution.gpd_scale is None
        body = SplineOracle([1.0, 5.0, 20.0, 60.0, 200.0]).spline
        sigma = 0.05 * 200.0 * body.derivative()(SCORES[-1]) / norm.pdf(SCORES[-1])
        assert distribution.effective_gpd_scale == pytest.approx(sigma, rel=1e-12)
        for level in (0.9625, 0.99, 0.999):
            exceedance = (level - 0.95) / 0.05
            growth = (
                -math.log1p(-exceedance)
                if shape == 0.0
                else ((1 - exceedance) ** -shape - 1) / shape
            )
            assert distribution.quantile(level) == pytest.approx(
                200.0 + sigma * growth, rel=1e-12
            )

    @pytest.mark.parametrize("xi", [-0.3, 0.0, 0.5])
    def test_explicit_scale_sampling_matches_closed_form(self, xi):
        distribution = Distribution(
            severity_minimum_strict=False, **BODY, gpd_shape=xi, gpd_scale=300.0
        )
        probabilities = np.array([0.95, 0.99, 0.999])
        tail_survival = (1 - probabilities) / 0.05
        excess = (
            -300 * np.log(tail_survival)
            if xi == 0
            else 300 / xi * (tail_survival**-xi - 1)
        )
        expected = 200 + excess
        np.testing.assert_allclose(
            distribution.sample_from_uniforms(probabilities), expected, rtol=1e-12
        )
        np.testing.assert_allclose(
            [distribution.quantile(p) for p in probabilities], expected, rtol=1e-12
        )

    def test_larger_sigma_is_strictly_heavier_tail(self):
        eff = Distribution(
            severity_minimum_strict=False, **BODY, gpd_shape=0.5
        ).effective_gpd_scale
        assert eff is not None
        light = Distribution(
            severity_minimum_strict=False, **BODY, gpd_shape=0.5, gpd_scale=eff * 0.5
        )
        base = Distribution(
            severity_minimum_strict=False, **BODY, gpd_shape=0.5, gpd_scale=eff
        )
        heavy = Distribution(
            severity_minimum_strict=False, **BODY, gpd_shape=0.5, gpd_scale=eff * 2.0
        )
        for p in (0.99, 0.999, 0.9999):
            assert light.quantile(p) < base.quantile(p) < heavy.quantile(p)


class TestValidation:
    def test_scale_without_shape_rejected(self):
        with pytest.raises(ValueError, match="require gpd_shape"):
            Distribution(severity_minimum_strict=False, **BODY, gpd_scale=100.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
    def test_nonpositive_or_nonfinite_scale_rejected(self, bad):
        with pytest.raises(ValueError):
            Distribution(
                severity_minimum_strict=False, **BODY, gpd_shape=0.5, gpd_scale=bad
            )


class TestRepresentability:
    """Threshold stability maps a fixed peaks-over-threshold fit to the p95
    splice.

    An illustrative fit (shape 0.5, scale 1e8 above a threshold of 1e7) has
    scale ``beta + xi * (p95 - u)`` at the p95 splice, which puts p99 at
    ``p95 + sigma / xi * (5**xi - 1)``. Body-slope continuity falls short of
    that ratio.
    """

    BODY = dict(p5=1.0e6, p25=5.0e6, p50=2.0e7, p75=1.0e8, p95=5.0e8)
    XI, BETA, THRESHOLD = 0.5, 1.0e8, 1.0e7

    def test_continuity_falls_short_but_explicit_reaches_the_fitted_p99(self):
        p95 = self.BODY["p95"]
        cont = Distribution(
            severity_minimum_strict=False, **self.BODY, gpd_shape=self.XI
        )
        sigma_runtime = self.BETA + self.XI * (p95 - self.THRESHOLD)
        expl = Distribution(
            severity_minimum_strict=False,
            **self.BODY,
            gpd_shape=self.XI,
            gpd_scale=sigma_runtime,
        )
        fitted_p99 = p95 + sigma_runtime / self.XI * (5**self.XI - 1)

        assert expl.quantile(0.99) == pytest.approx(fitted_p99, rel=1e-9)
        assert fitted_p99 / p95 == pytest.approx(2.71, abs=0.005)
        assert cont.quantile(0.99) < expl.quantile(0.99)


class TestPropagation:
    def test_shape_change_holds_explicit_sigma_fixed(self):
        d = Distribution(
            severity_minimum_strict=False, **BODY, gpd_shape=0.5, gpd_scale=300.0
        ).with_gpd_shape(0.6)
        assert d.gpd_shape == pytest.approx(0.6)
        assert d.effective_gpd_scale == pytest.approx(300.0)

    def test_excess_scale_scales_explicit_sigma(self):
        d = Distribution(
            severity_minimum_strict=False, **BODY, gpd_shape=0.5, gpd_scale=300.0
        )
        d2 = d.with_excess_scale(2.0)
        assert d2.effective_gpd_scale == pytest.approx(600.0)
        assert d2.quantile(0.99) == pytest.approx(2.0 * d.quantile(0.99), rel=1e-6)

    def test_continuity_stays_continuity_through_perturbations(self):
        d = Distribution(severity_minimum_strict=False, **BODY, gpd_shape=0.5)
        assert d.with_gpd_shape(0.6).gpd_scale is None
        assert d.with_excess_scale(2.0).gpd_scale is None
