import math

import numpy as np
import pytest

from catmodel.probability.distributions import (
    Distribution,
    MixtureSeverityDistribution,
)


class TestDeterministicSurvivalProbability:
    def test_body_quantiles_invert_to_exact_survival(self):
        distribution = Distribution(
            p5=1.0,
            p25=2.0,
            p50=3.0,
            p75=4.0,
            p95=5.0,
        )
        assert distribution.survival_probability(1.0) == pytest.approx(0.95)
        assert distribution.survival_probability(3.0) == pytest.approx(0.50)
        assert distribution.survival_probability(5.0) == pytest.approx(0.05)

    def test_flat_authored_body_uses_right_edge_for_strict_exceedance(self):
        distribution = Distribution(
            p5=1.0,
            p25=2.0,
            p50=2.0,
            p75=3.0,
            p95=4.0,
        )
        assert distribution.survival_probability(2.0) == pytest.approx(0.50)

    def test_gpd_survival_matches_authored_quantiles(self):
        distribution = Distribution(
            p5=1e5,
            p25=2e5,
            p50=4e5,
            p75=1e6,
            p95=4e6,
            gpd_shape=0.55,
            gpd_tail_ratio=4.0,
        )
        assert distribution.survival_probability(distribution.p95) == pytest.approx(
            0.05
        )
        q99 = distribution.quantile(0.99)
        assert distribution.survival_probability(q99) == pytest.approx(0.01, rel=1e-12)

    def test_bounded_gpd_has_zero_survival_at_endpoint(self):
        distribution = Distribution(
            p5=1.0,
            p25=2.0,
            p50=3.0,
            p75=4.0,
            p95=5.0,
            gpd_shape=-0.5,
            gpd_scale=2.0,
        )
        endpoint = distribution.p95 - 2.0 / -0.5
        assert distribution.survival_probability(endpoint) == 0.0
        assert distribution.survival_probability(endpoint + 1.0) == 0.0

    def test_severity_cap_is_an_exact_strict_exceedance_boundary(self):
        uncapped = Distribution(
            p5=1.0,
            p25=2.0,
            p50=3.0,
            p75=4.0,
            p95=5.0,
            gpd_shape=0.5,
            gpd_scale=2.0,
        )
        capped = Distribution(
            p5=1.0,
            p25=2.0,
            p50=3.0,
            p75=4.0,
            p95=5.0,
            gpd_shape=0.5,
            gpd_scale=2.0,
            severity_cap=10.0,
        )
        assert capped.survival_probability(10.0) == 0.0
        assert capped.survival_probability(np.nextafter(10.0, -math.inf)) == (
            pytest.approx(uncapped.survival_probability(np.nextafter(10.0, -math.inf)))
        )

    def test_mixture_survival_is_exact_component_weighted_pool(self):
        first = Distribution(
            p5=1.0,
            p25=2.0,
            p50=3.0,
            p75=4.0,
            p95=5.0,
            gpd_shape=0.3,
        )
        second = Distribution(
            p5=2.0,
            p25=4.0,
            p50=6.0,
            p75=8.0,
            p95=10.0,
            gpd_shape=0.6,
        )
        mixture = MixtureSeverityDistribution(
            components=[first, second],
            weights=np.array([0.25, 0.75]),
        )
        threshold = 7.5
        expected = 0.25 * first.survival_probability(
            threshold
        ) + 0.75 * second.survival_probability(threshold)
        assert mixture.survival_probability(threshold) == pytest.approx(
            expected, rel=1e-15
        )

    def test_mixture_level_cap_is_exact(self):
        component = Distribution(
            p5=1.0,
            p25=2.0,
            p50=3.0,
            p75=4.0,
            p95=5.0,
            gpd_shape=0.5,
        )
        mixture = MixtureSeverityDistribution(
            components=[component],
            weights=np.array([1.0]),
            severity_cap=8.0,
        )
        assert mixture.survival_probability(8.0) == 0.0
        assert mixture.survival_probability(7.0) == pytest.approx(
            component.survival_probability(7.0)
        )

    @pytest.mark.parametrize("threshold", [math.inf, -math.inf, math.nan])
    def test_nonfinite_survival_threshold_rejected(self, threshold):
        distribution = Distribution(
            p5=1.0,
            p25=2.0,
            p50=3.0,
            p75=4.0,
            p95=5.0,
        )
        with pytest.raises(ValueError, match="threshold must be finite"):
            distribution.survival_probability(threshold)
