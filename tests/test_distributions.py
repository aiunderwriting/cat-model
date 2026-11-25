import math

import numpy as np
import pytest
import structlog.testing

from catmodel.probability.distributions import (
    Distribution,
    MixtureSeverityDistribution,
    _limited_gpd_excess_mean,
    p999_implied_gpd_shape,
)
from catmodel.scenarios.sensitivity_coverage import (
    CAPPED_TAIL_SHAPE_SANITY_BOUND,
    UNCAPPED_TAIL_SHAPE_BOUND,
)
from test_support.sampling import sample_severity


def _gpd_body(**kwargs) -> Distribution:
    return Distribution(p5=1e6, p25=5e6, p50=1e7, p75=5e7, p95=2e8, **kwargs)


class TestAnalyticMean:
    def test_matches_mc_for_moderate_tail(self):
        d = _gpd_body(gpd_shape=0.3)
        mc = float(sample_severity(d, 2_000_000, rng=np.random.default_rng(7)).mean())
        assert mc == pytest.approx(d.analytic_mean(), rel=0.01)

    def test_matches_mc_for_no_gpd_body(self):
        d = Distribution(p5=1.0, p25=2.0, p50=3.0, p75=4.0, p95=5.0)
        mc = float(sample_severity(d, 2_000_000, rng=np.random.default_rng(1)).mean())
        assert mc == pytest.approx(d.analytic_mean(), rel=0.002)

    @pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
    def test_capped_tail_closed_form_matches_quadrature(self):
        from scipy.integrate import quad

        d = _gpd_body(gpd_shape=0.6, severity_cap=1e9)

        def capped_quantile(u: float) -> float:
            return float(d.sample_from_uniforms(np.array([u]))[0])

        numeric, _err = quad(capped_quantile, 0.0, 1.0, limit=200)
        assert d.analytic_mean() == pytest.approx(numeric, rel=1e-4)

    @pytest.mark.parametrize("cap_multiple", [1.0, 2.0, 100.0])
    def test_negative_shape_cap_at_or_above_endpoint_preserves_mean(self, cap_multiple):
        uncapped = _gpd_body(gpd_shape=-0.5)
        assert uncapped.effective_gpd_scale is not None
        assert uncapped.gpd_shape is not None
        endpoint = uncapped.p95 - uncapped.effective_gpd_scale / uncapped.gpd_shape
        capped = _gpd_body(gpd_shape=-0.5, severity_cap=endpoint * cap_multiple)

        assert capped.analytic_mean() == pytest.approx(uncapped.analytic_mean())
        assert uncapped.analytic_mean(
            severity_cap_override=endpoint * cap_multiple
        ) == (pytest.approx(uncapped.analytic_mean()))

    @pytest.mark.parametrize("shape", [-0.5, 0.0, 0.6, 0.999999, 1.0, 1.000001, 2.5])
    def test_tail_mean_matches_survival_integral(self, shape):
        from scipy.integrate import quad

        cap = 1e9
        dist = _gpd_body(gpd_shape=shape, severity_cap=cap)
        scale = dist.effective_gpd_scale
        assert scale is not None
        excess_limit = (cap - dist.p95) / scale
        if shape < 0.0:
            excess_limit = min(excess_limit, -1.0 / shape)

        def survival(excess):
            if shape == 0.0:
                return np.exp(-excess)
            return (1.0 + shape * excess) ** (-1.0 / shape)

        expected = scale * quad(survival, 0.0, excess_limit)[0]
        assert _limited_gpd_excess_mean(cap - dist.p95, shape, scale) == pytest.approx(
            expected
        )

    def test_remote_cap_does_not_invent_residual_tail_mass(self):
        uncapped = _gpd_body(gpd_shape=0.6)
        capped = _gpd_body(gpd_shape=0.6, severity_cap=1e15)

        assert capped.analytic_mean() <= uncapped.analytic_mean()
        # Only the GPD stop-loss above the cap leaves the mean:
        # 0.05 S(c) (sigma + xi (c - u)) / (1 - xi) with S the GPD survival.
        sigma, xi, u, c = capped.effective_gpd_scale, 0.6, 2e8, 1e15
        assert sigma is not None
        survival = (1 + xi * (c - u) / sigma) ** (-1 / xi)
        stop_loss = 0.05 * survival * (sigma + xi * (c - u)) / (1 - xi)
        assert capped.analytic_mean() == pytest.approx(
            uncapped.analytic_mean() - stop_loss, rel=1e-12
        )

    @pytest.mark.parametrize("cap", [4.5, 5.1])
    def test_no_gpd_cap_censors_without_reshaping_uncapped_law(self, cap):
        uncapped = Distribution(p5=1.0, p25=2.0, p50=3.0, p75=4.0, p95=5.0)
        capped = Distribution(
            p5=1.0, p25=2.0, p50=3.0, p75=4.0, p95=5.0, severity_cap=cap
        )
        uniforms = np.linspace(0.0, 1.0, 10_001)
        raw = uncapped.sample_from_uniforms(uniforms)

        assert capped.sample_from_uniforms_uncapped(uniforms) == pytest.approx(raw)
        assert capped.sample_from_uniforms(uniforms) == pytest.approx(
            np.minimum(raw, cap)
        )
        assert capped.analytic_mean() == pytest.approx(
            uncapped.analytic_mean(severity_cap_override=cap)
        )
        # A severity scale factor applies to the raw law before the cap.
        assert np.minimum(
            capped.sample_from_uniforms_uncapped(uniforms) * 0.5, cap
        ) == (pytest.approx(np.minimum(raw * 0.5, cap)))

    def test_cap_override_lowers_mean(self):
        d = _gpd_body(gpd_shape=0.5)
        assert d.analytic_mean(severity_cap_override=5e8) < d.analytic_mean()

    def test_mixture_analytic_mean_is_weighted_component_mean(self):
        c1 = _gpd_body(gpd_shape=0.4)
        c2 = _gpd_body(gpd_shape=0.6)
        mix = MixtureSeverityDistribution(
            components=[c1, c2], weights=np.array([0.25, 0.75])
        )
        expected = 0.25 * c1.analytic_mean() + 0.75 * c2.analytic_mean()
        assert mix.analytic_mean() == pytest.approx(expected, rel=1e-12)

    def test_limited_mean_handles_separated_components_and_cap(self):
        mix = MixtureSeverityDistribution(
            components=[
                Distribution(p5=0.8, p25=0.9, p50=1.0, p75=1.1, p95=1.2),
                Distribution(p5=80, p25=90, p50=100, p75=110, p95=120),
            ],
            weights=np.array([0.5, 0.5]),
        )

        actual = mix.analytic_mean(severity_cap_override=105.0)
        uniforms = np.linspace(0.0, 1.0, 1_000_001)
        expected = np.trapezoid(
            np.minimum(mix.sample_from_uniforms_uncapped(uniforms), 105.0),
            uniforms,
        )

        # The trapezoid rule's error across the component jump is about 1e-6.
        assert actual == pytest.approx(expected, rel=2e-6)


@pytest.mark.stochastic
class TestMcMeanRecoveryRegression:
    """At 500K simulations, ξ = 0.95 recovers 55-85% of analytic mean.

    Missing deep-tail draws cause the shortfall, which CV diagnostics cannot detect.
    """

    N_SIMS = 500_000
    SEED = 0

    def _recovery(self, xi: float) -> float:
        d = _gpd_body(gpd_shape=xi)
        draws = sample_severity(d, self.N_SIMS, rng=np.random.default_rng(self.SEED))
        mc = float(draws.mean())
        return mc / d.analytic_mean()

    @pytest.mark.parametrize("xi", [0.3, 0.65])
    def test_recovery_meets_truncated_mean_lower_bound_below_clamp(self, xi: float):
        """Shortfall below the clamp stays within a seed-free sampling bound.

        At ``xi >= 0.5`` the sample mean has no standard error, so the bound
        uses ``min(X, t)`` with ``t = Q(1 - 5 / n)``. The sample mean of ``X``
        is at least that of ``min(X, t)``, whose mean ``mu_t`` and variance
        ``v_t`` are finite. Under a correct sampler, recovery below
        ``(mu_t - 5 sqrt(v_t / n)) / mu`` therefore has probability of about
        ``Phi(-5)`` or less at any seed. The quadrature integrates the public
        quantile function.
        """
        from scipy.integrate import quad

        d = _gpd_body(gpd_shape=xi)
        upper = 1 - 5 / self.N_SIMS
        truncation = d.quantile(upper)
        knots = [0.05, 0.25, 0.50, 0.75, 0.95]
        first = quad(d.quantile, 0, upper, points=knots, limit=200)[0]
        first += truncation * (1 - upper)
        second = quad(lambda u: d.quantile(u) ** 2, 0, upper, points=knots, limit=200)[
            0
        ]
        second += truncation**2 * (1 - upper)
        standard_error = math.sqrt((second - first**2) / self.N_SIMS)
        bound = (first - 5 * standard_error) / d.analytic_mean()
        assert self._recovery(xi) >= bound

    def test_clamp_boundary_shortfall_documented_band(self):
        recovery = self._recovery(0.95)
        assert 0.55 <= recovery <= 0.85


class TestXiClampBoundaryWarning:
    def test_construction_at_clamp_boundary_logs_warning(self):
        with structlog.testing.capture_logs() as logs:
            _gpd_body(gpd_shape=0.95)
        clamp_logs = [
            entry for entry in logs if entry.get("gpd_shape") == pytest.approx(0.95)
        ]
        assert len(clamp_logs) == 1
        assert clamp_logs[0]["log_level"] == "warning"
        assert "depends on tail shape and cap binding" in clamp_logs[0]["event"]
        assert (
            "body-binding cap can approach 100% recovery"
            in clamp_logs[0]["mean_recovery_context"]
        )

    def test_no_warning_below_clamp_boundary(self):
        with structlog.testing.capture_logs() as logs:
            _gpd_body(gpd_shape=0.65)
        assert logs == []


class TestMixtureSeverityCapFloor:
    def test_cap_between_component_p50_and_p95_censors(self):
        component = _gpd_body(gpd_shape=0.5)
        mix = MixtureSeverityDistribution(
            components=[component],
            weights=np.array([1.0]),
            severity_cap=5e7,
        )
        assert mix.quantile(0.95) == pytest.approx(5e7)

    def test_multi_component_cap_censors_only_above_cap_mass(self):
        ok = Distribution(p5=1e5, p25=2e5, p50=5e5, p75=1e6, p95=2e6)
        offending = _gpd_body(gpd_shape=0.5)
        mix = MixtureSeverityDistribution(
            components=[ok, offending],
            weights=np.array([0.5, 0.5]),
            severity_cap=1e8,
        )
        draws = mix.sample_from_uniforms(np.array([0.5, 0.99, 1.0]))
        assert np.all(draws <= 1e8)
        assert draws[-1] == pytest.approx(1e8)

    def test_excess_scale_preserves_precap_law_then_censors_draws(self):
        mix = MixtureSeverityDistribution(
            components=[_gpd_body(gpd_shape=0.5)],
            weights=np.array([1.0]),
            severity_cap=3e8,
        )
        scaled = mix.with_excess_scale(1.78)
        assert scaled.severity_cap == pytest.approx(3e8)
        assert scaled.components[0].p95 == pytest.approx(2e8 * 1.78)
        assert scaled.components[0].analytic_mean() == pytest.approx(
            1.78 * mix.components[0].analytic_mean()
        )
        probabilities = np.array([0.05, 0.5, 0.95, 0.99])
        expected = np.minimum(
            1.78 * mix.components[0].sample_from_uniforms(probabilities), 3e8
        )
        np.testing.assert_allclose(scaled.sample_from_uniforms(probabilities), expected)

    def test_near_point_mass_accepts_non_degenerate_severity_scaling(self):
        # Scaling ordered knots does not collapse them, even near a fixed cap.
        tight = MixtureSeverityDistribution(
            components=[
                Distribution(
                    p5=2e8 * (1 - 5e-5),
                    p25=2e8 * (1 - 4e-5),
                    p50=2e8 * (1 - 3e-5),
                    p75=2e8 * (1 - 2e-5),
                    p95=2e8,
                )
            ],
            weights=np.array([1.0]),
            severity_cap=2e8,
        )
        scaled = tight.with_excess_scale(0.5)
        assert scaled is not tight
        assert scaled.severity_cap == tight.severity_cap
        assert scaled.components[0].analytic_mean() == pytest.approx(
            0.5 * tight.components[0].analytic_mean()
        )
        probabilities = np.array([0.01, 0.05, 0.5, 0.95])
        np.testing.assert_allclose(
            scaled.sample_from_uniforms(probabilities),
            0.5 * tight.sample_from_uniforms(probabilities),
        )


class TestSingleComponentMixtureFidelity:
    def test_single_component_passthrough_is_exact(self):
        component = _gpd_body(gpd_shape=0.55)
        mix = MixtureSeverityDistribution(
            components=[component], weights=np.array([1.0])
        )
        probs = np.array([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999])
        np.testing.assert_array_equal(
            mix.sample_from_uniforms(probs),
            component.sample_from_uniforms(probs),
        )
        # Elicited knots bypass inverse-cache error.
        assert mix.quantile(0.50) == component.p50
        assert mix.quantile(0.95) == component.p95

    def test_single_component_mixture_cap_still_applies(self):
        component = _gpd_body(gpd_shape=0.55)
        mix = MixtureSeverityDistribution(
            components=[component], weights=np.array([1.0]), severity_cap=1e9
        )
        deep = np.array([0.99999])
        assert float(mix.sample_from_uniforms(deep)[0]) == pytest.approx(1e9)
        assert float(mix.sample_from_uniforms_uncapped(deep)[0]) > 1e9

    def test_duplicate_components_preserve_the_exact_inverse(self):
        """Duplicating an opinion cannot change its atoms or tail law."""
        component = _gpd_body(gpd_shape=0.55)
        mix = MixtureSeverityDistribution(
            components=[
                _gpd_body(gpd_shape=0.55),
                _gpd_body(gpd_shape=0.55),
            ],
            weights=np.array([0.5, 0.5]),
        )
        for p in (0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999):
            rel_err = abs(mix.quantile(p) - component.quantile(p)) / component.quantile(
                p
            )
            assert rel_err == 0.0, f"p={p}: rel_err={rel_err}"


# Test reference tail: Pareto alpha = 0.60 within a test range of 0.43-0.83, so
# xi ~= 1.67 exceeds the finite-mean bound and requires a governed cap.
_IDENTIFIED_XI = 1.0 / 0.60
_GOVERNED_CAP = 1e10


def _capped_mean_by_quadrature(dist: Distribution, cap: float) -> float:
    """Independently integrate E[min(X, cap)] on a quantile grid refined near 1."""

    body = np.linspace(0.0, 0.95, 400_001)
    tail = 1.0 - np.geomspace(5e-2, 1e-13, 400_000)
    grid = np.unique(np.concatenate([body, tail, np.array([1.0])]))
    values = np.minimum(dist.sample_from_uniforms_uncapped(grid), cap)
    return float(np.trapezoid(values, grid))


class TestConditionalXiBound:
    """The finite-mean xi bound is conditional on a governed severity cap.

    A cap makes E[min(X, cap)] finite for xi >= 1. The capped sanity bound limits
    governed scenario shapes, not the raw law.
    """

    def test_raw_law_admission_is_separate_from_mean_requirements(self):
        law = _gpd_body(gpd_shape=0.96)
        assert math.isfinite(law.analytic_mean())
        assert law.quantile(0.99) > law.p95

    def test_capped_component_at_identified_shape_priceable(self):
        dist = _gpd_body(gpd_shape=_IDENTIFIED_XI, severity_cap=_GOVERNED_CAP)
        draws = sample_severity(dist, 20_000, rng=np.random.default_rng(7))
        assert np.all(np.isfinite(draws))
        assert np.all(draws <= _GOVERNED_CAP)
        assert draws.max() > dist.quantile(0.95)

    def test_raw_law_does_not_inherit_a_sensitivity_menu_bound(self):
        law = _gpd_body(gpd_shape=2.6, severity_cap=_GOVERNED_CAP)
        assert math.isfinite(law.analytic_mean())
        assert math.isfinite(law.quantile(0.99))

    def test_sanity_bound_sits_above_identified_interval(self):
        heaviest_identified_xi = 1.0 / 0.43  # lower alpha of the test range
        assert heaviest_identified_xi <= CAPPED_TAIL_SHAPE_SANITY_BOUND


class TestUncappedMeanDivergence:
    def test_valid_infinite_mean_law_retains_quantiles_and_limited_moments(self):
        for xi in (1.0, 1.5, _IDENTIFIED_XI, 2.5):
            law = _gpd_body(gpd_shape=xi)
            assert math.isfinite(law.quantile(0.99))
            assert math.isfinite(law.analytic_mean(severity_cap_override=_GOVERNED_CAP))
            with pytest.raises(ValueError, match="uncapped GPD tail mean"):
                law.analytic_mean()


class TestCappedAnalyticMean:
    def test_deep_cap_below_positive_xi_tail_endpoint_matches_quadrature(self):
        cap = 5e11
        dist = Distribution(
            p5=1e11,
            p25=5e11,
            p50=9e11,
            p75=2e12,
            p95=5e12,
            gpd_shape=0.45,
            severity_cap=cap,
        )

        reference = _capped_mean_by_quadrature(dist, cap)
        assert dist.analytic_mean() == pytest.approx(reference, rel=5e-5)
        assert dist.analytic_mean() <= cap

    @pytest.mark.parametrize("xi", [1.0, 1.2, _IDENTIFIED_XI, 2.0, 2.5])
    def test_capped_analytic_mean_finite_and_correct_across_heavy_shapes(self, xi):
        dist = _gpd_body(gpd_shape=xi, severity_cap=_GOVERNED_CAP)
        analytic = dist.analytic_mean()
        assert np.isfinite(analytic)
        reference = _capped_mean_by_quadrature(dist, _GOVERNED_CAP)
        assert analytic == pytest.approx(reference, rel=5e-3)

    def test_capped_mean_is_monotone_in_shape(self):
        means = [
            _gpd_body(gpd_shape=xi, severity_cap=_GOVERNED_CAP).analytic_mean()
            for xi in (0.5, 0.95, 1.2, _IDENTIFIED_XI, 2.0)
        ]
        assert means == sorted(means)


class TestCapSensitivityUnderHeavyTail:
    def test_cap_move_materially_changes_mean_at_identified_shape(self):
        base = _gpd_body(
            gpd_shape=_IDENTIFIED_XI, severity_cap=_GOVERNED_CAP
        ).analytic_mean()
        raised = _gpd_body(
            gpd_shape=_IDENTIFIED_XI, severity_cap=5 * _GOVERNED_CAP
        ).analytic_mean()
        # The 35% floor keeps the current ~56% uplift visibly material with headroom.
        assert (raised / base - 1.0) > 0.35

    def test_cap_sensitivity_is_muted_at_the_clamped_shape(self):
        base = _gpd_body(
            gpd_shape=UNCAPPED_TAIL_SHAPE_BOUND, severity_cap=_GOVERNED_CAP
        ).analytic_mean()
        raised = _gpd_body(
            gpd_shape=UNCAPPED_TAIL_SHAPE_BOUND, severity_cap=5 * _GOVERNED_CAP
        ).analytic_mean()
        clamped_uplift = raised / base - 1.0
        identified_uplift = (
            _gpd_body(
                gpd_shape=_IDENTIFIED_XI, severity_cap=5 * _GOVERNED_CAP
            ).analytic_mean()
            / _gpd_body(
                gpd_shape=_IDENTIFIED_XI, severity_cap=_GOVERNED_CAP
            ).analytic_mean()
            - 1.0
        )
        # The clamped ~9% uplift must stay below one-fifth of the identified ~56%.
        assert clamped_uplift < identified_uplift / 5.0


@pytest.mark.parametrize("shape", [-0.5, 0.0, 0.3, 0.5, 0.9, 1.2, 3.0])
def test_p999_implied_gpd_shape_recovers_the_generating_tail(shape):
    from scipy.stats import genpareto

    tail = genpareto(c=shape, loc=100.0, scale=40.0)
    p99, p999 = tail.ppf(0.8), tail.ppf(0.98)
    assert p999_implied_gpd_shape(100.0, p99, p999) == pytest.approx(shape, abs=1e-9)


def test_p999_implied_gpd_shape_is_monotone_and_requires_ordered_quantiles():
    shapes = [p999_implied_gpd_shape(100, 200, p999) for p999 in (210, 400, 1e4, 1e9)]
    assert shapes == sorted(shapes)
    for values in ((100, 100, 200), (100, 200, 200), (0, 1, 2), (1, 2, math.inf)):
        with pytest.raises(ValueError, match="0 < p95 < p99 < p999"):
            p999_implied_gpd_shape(*values)
