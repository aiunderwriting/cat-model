"""Native distribution and keyed-stream invariants independent of orchestration."""

import numpy as np
import pytest

from catmodel.probability.distributions import Distribution, MixtureSeverityDistribution
from test_support.sampling import sample_severity
from test_support.severity_oracles import SplineOracle


class TestDistributionValidation:
    def test_reversed_knots_raises(self):
        with pytest.raises(ValueError, match="non-decreasing"):
            Distribution(p5=10, p25=50, p50=30, p75=40, p95=50)

    def test_constant_knots_preserve_a_deterministic_law(self):
        law = Distribution(p5=30, p25=30, p50=30, p75=30, p95=30)
        np.testing.assert_array_equal(
            law.sample_from_uniforms(np.array([0.0, 0.5, 1.0])), [30, 30, 30]
        )
        assert law.analytic_mean() == 30

    def test_p5_greater_than_p95_raises(self):
        with pytest.raises(ValueError, match="non-decreasing"):
            Distribution(p5=50, p25=40, p50=30, p75=20, p95=10)

    def test_negative_knots_are_not_a_severity_law(self):
        with pytest.raises(ValueError, match="nonnegative"):
            Distribution(p5=-100, p25=-50, p50=0, p75=50, p95=100)

    def test_positive_lower_continuation_has_no_zero_atom(self):
        d = Distribution(p5=0.01, p25=1.0, p50=2.0, p75=3.0, p95=4.0)
        u = np.linspace(0.0, 0.05, 1001)
        values = d.sample_from_uniforms(u)

        assert values[0] == pytest.approx(0.0)
        assert np.all(values[1:] > 0.0)
        assert values[-1] == pytest.approx(d.p5)
        assert np.all(np.diff(values) >= -1e-12)

    def test_zero_p5_positive_body_has_no_zero_atom(self):
        d = Distribution(
            p5=0.0,
            p25=76_988_131.0,
            p50=464_912_281.0,
            p75=3_352_678_571.0,
            p95=28_196_428_571.0,
            gpd_shape=0.65,
        )
        u = np.linspace(0.0, 0.05, 1001)
        values = d.sample_from_uniforms(u)

        assert values[0] == pytest.approx(0.0)
        assert np.all(values[1:] > 0.0)
        assert np.count_nonzero(values == 0.0) == 1
        assert np.all(np.diff(values) >= -1e-12)
        assert values[-1] > d.p5
        assert values[-1] < d.p25 * 1e-3

    def test_severity_cap_censors_the_unmodified_upper_continuation(self):
        d = Distribution(
            p5=10.0,
            p25=20.0,
            p50=40.0,
            p75=60.0,
            p95=99.0,
            severity_cap=100.0,
        )
        u = np.linspace(0.95, 1.0, 1001)
        values = d.sample_from_uniforms(u)

        assert values[0] == pytest.approx(d.p95)
        assert np.all(values <= 100.0)
        np.testing.assert_array_equal(
            values, np.minimum(d.sample_from_uniforms_uncapped(u), 100.0)
        )
        assert np.count_nonzero(values == 100.0) > 1
        assert np.all(np.diff(values) >= -1e-12)

    def test_non_finite_distribution_knot_rejected(self):
        with pytest.raises(ValueError, match="must be finite"):
            Distribution(p5=10, p25=np.nan, p50=30, p75=40, p95=50)

    def test_non_finite_severity_cap_rejected(self):
        with pytest.raises(ValueError, match="severity_cap must be finite"):
            Distribution(
                p5=10,
                p25=20,
                p50=30,
                p75=40,
                p95=50,
                gpd_shape=0.4,
                severity_cap=np.inf,
            )

    def test_non_finite_gpd_shape_rejected(self):
        with pytest.raises(ValueError, match="gpd_shape must be finite"):
            Distribution(p5=10, p25=20, p50=30, p75=40, p95=50, gpd_shape=np.nan)


class TestDistributionSampling:
    @pytest.fixture
    def dist(self):
        return Distribution(p5=10, p25=25, p50=50, p75=75, p95=90)

    def test_no_uniforms_return_no_amounts(self, dist):
        samples = dist.sample_from_uniforms(np.empty(0))
        assert samples.shape == (0,)


class TestMixtureSeverityDistribution:
    @pytest.fixture
    def separated_mixture(self):
        return MixtureSeverityDistribution(
            components=[
                Distribution(p5=1, p25=2, p50=3, p75=4, p95=5),
                Distribution(p5=10, p25=20, p50=30, p75=40, p95=50),
            ],
            weights=np.array([0.5, 0.5]),
        )

    def test_sample_from_uniforms_is_monotone(self, separated_mixture):
        uniforms = np.linspace(0.0, 1.0, 1_000)
        values = separated_mixture.sample_from_uniforms(uniforms)
        assert np.all(np.diff(values) >= -1e-9)

    def test_disjoint_component_quantiles_match_linear_pool(self, separated_mixture):
        assert separated_mixture.quantile(0.25) == pytest.approx(3.0, abs=0.15)
        assert separated_mixture.quantile(0.75) == pytest.approx(30.0, abs=1.0)

    def test_disjoint_component_inverse_preserves_jump(self, separated_mixture):
        # Bounded continuations leave a gap between component supports; at the
        # 0.5 mixture boundary, the inverse jumps from the lower to upper support.
        lower, upper = separated_mixture.components
        c1_max = float(lower.sample_from_uniforms(np.array([1.0]))[0])
        c2_min = float(upper.sample_from_uniforms(np.array([0.0]))[0])
        assert c1_max < c2_min

        assert separated_mixture.quantile(0.5) == pytest.approx(c1_max, rel=1e-3)
        assert separated_mixture.quantile(np.nextafter(0.5, 1.0)) == pytest.approx(
            c2_min, rel=1e-3
        )
        near_jump = separated_mixture.sample_from_uniforms(
            np.array([0.5 - 1e-13, 0.5, np.nextafter(0.5, 1.0)])
        )
        assert near_jump[0] <= c1_max + 1e-6
        assert near_jump[1] == pytest.approx(c1_max, rel=1e-3)
        assert near_jump[2] == pytest.approx(c2_min, rel=1e-3)
        assert near_jump[2] - near_jump[1] == pytest.approx(c2_min - c1_max, rel=1e-3)

    def test_cap_clips_public_mixture_inverse(self):
        # Keep the cap above component p95s so it cannot collapse a body.
        mixture = MixtureSeverityDistribution(
            components=[
                Distribution(p5=1, p25=2, p50=3, p75=4, p95=5),
                Distribution(p5=10, p25=20, p50=30, p75=40, p95=50, gpd_shape=0.5),
            ],
            weights=np.array([0.5, 0.5]),
            severity_cap=60.0,
        )
        values = mixture.sample_from_uniforms(np.linspace(0.0, 1.0, 10_000))
        assert values.max() == pytest.approx(60.0)
        assert np.any(values == 60.0)

    def test_uncapped_mixture_inverse_bypasses_policy_clip(self):
        mixture = MixtureSeverityDistribution(
            components=[
                Distribution(p5=1, p25=2, p50=3, p75=4, p95=5),
                Distribution(p5=10, p25=20, p50=30, p75=40, p95=50, gpd_shape=0.5),
            ],
            weights=np.array([0.5, 0.5]),
            severity_cap=60.0,
        )
        values = mixture.sample_from_uniforms_uncapped(np.linspace(0.0, 1.0, 10_000))
        assert values.max() > 60.0
        assert not np.any(values == 60.0)

    def test_cap_below_component_body_censors_mixture(self):
        mixture = MixtureSeverityDistribution(
            components=[
                Distribution(p5=1, p25=2, p50=3, p75=4, p95=5),
                Distribution(p5=10, p25=20, p50=30, p75=40, p95=50),
            ],
            weights=np.array([0.5, 0.5]),
            severity_cap=4.0,
        )
        samples = sample_severity(mixture, 20_000, rng=np.random.default_rng(42))
        assert np.max(samples) == 4.0
        assert np.mean(samples == 4.0) > 0.5

    def test_inversion_is_repeatable(self, separated_mixture):
        uniforms = np.random.default_rng(42).random(100)
        a = separated_mixture.sample_from_uniforms(uniforms)
        b = separated_mixture.sample_from_uniforms(uniforms)
        np.testing.assert_array_equal(a, b)


class TestDistributionGPDTail:
    def test_negative_gpd_shape_bounded_tail(self):
        d = Distribution(
            p5=10,
            p25=20,
            p50=30,
            p75=40,
            p95=50,
            gpd_shape=-0.5,
        )
        samples = sample_severity(d, 20_000, rng=np.random.default_rng(42))
        # With ξ=-0.5, upper bound is p95 + σ/|ξ|.
        assert d.gpd_shape is not None
        upper_bound = d._gpd_threshold + d._gpd_scale / abs(d.gpd_shape)
        assert np.max(samples) <= upper_bound + 1.0

    def test_severity_cap_bounds_samples(self):
        cap = 250e6
        d = Distribution(
            p5=1e6,
            p25=5e6,
            p50=15e6,
            p75=50e6,
            p95=80e6,
            gpd_shape=0.6,
            gpd_tail_ratio=2.0,
            severity_cap=cap,
        )
        samples = sample_severity(d, 20_000, rng=np.random.default_rng(42))
        assert np.max(samples) <= cap
        assert np.any(samples > cap * 0.90)

    def test_severity_cap_below_p95_censors_instead_of_rejecting(self):
        cap = 100e6
        distribution = Distribution(
            p5=1e6,
            p25=5e6,
            p50=15e6,
            p75=50e6,
            p95=200e6,
            gpd_shape=0.6,
            severity_cap=cap,
        )
        samples = sample_severity(distribution, 20_000, rng=np.random.default_rng(42))
        assert np.max(samples) == cap
        assert distribution.quantile(0.95) == cap

    @pytest.mark.parametrize("gpd_shape", [0.45, 0.0, -0.5])
    def test_severity_cap_equal_p95_censors_gpd_tail(self, gpd_shape: float):
        cap = 100e6
        distribution = Distribution(
            p5=1e6,
            p25=5e6,
            p50=15e6,
            p75=50e6,
            p95=cap,
            gpd_shape=gpd_shape,
            severity_cap=cap,
        )
        samples = sample_severity(distribution, 20_000, rng=np.random.default_rng(42))
        assert np.max(samples) == cap
        assert distribution.survival_probability(cap) == 0.0

    @pytest.mark.parametrize("invalid_cap", [0.0, -1.0])
    def test_nonpositive_severity_cap_rejected(self, invalid_cap: float):
        with pytest.raises(ValueError, match="severity_cap must be strictly positive"):
            Distribution(
                p5=1e6,
                p25=5e6,
                p50=15e6,
                p75=50e6,
                p95=100e6,
                severity_cap=invalid_cap,
            )

    def test_severity_cap_above_p95_clips_public_gpd(self):
        d = Distribution(
            p5=1e6,
            p25=5e6,
            p50=20e6,
            p75=80e6,
            p95=150e6,
            gpd_shape=0.50,
            gpd_tail_ratio=2.0,
            severity_cap=500e6,
        )
        samples = sample_severity(d, 100_000, rng=np.random.default_rng(42))
        assert np.all(samples <= 500e6)
        assert np.any(samples > 150e6)
        uncapped = d.sample_from_uniforms_uncapped(
            np.random.default_rng(42).uniform(0.0, 1.0, 100_000)
        )
        assert np.any(uncapped > 500e6)


class TestKeyedStreamIndependence:
    def test_same_key_reproduces_and_keys_separate_streams(self):
        from catmodel.probability.keyed_streams import keyed_stream_generator

        first = keyed_stream_generator(
            root_seed=42,
            risk_unit_id="lab::cell",
            purpose="scenario-marginal-stream-v1",
        ).random(64)
        again = keyed_stream_generator(
            root_seed=42,
            risk_unit_id="lab::cell",
            purpose="scenario-marginal-stream-v1",
        ).random(64)
        np.testing.assert_array_equal(first, again)

        other_row = keyed_stream_generator(
            root_seed=42,
            risk_unit_id="lab::other",
            purpose="scenario-marginal-stream-v1",
        ).random(64)
        other_purpose = keyed_stream_generator(
            root_seed=42,
            risk_unit_id="lab::cell",
            purpose="portfolio-shared-dependence-v1",
        ).random(64)
        other_seed = keyed_stream_generator(
            root_seed=43,
            risk_unit_id="lab::cell",
            purpose="scenario-marginal-stream-v1",
        ).random(64)
        assert not np.array_equal(first, other_row)
        assert not np.array_equal(first, other_purpose)
        assert not np.array_equal(first, other_seed)

    def test_invalid_keys_rejected(self):
        from catmodel.probability.keyed_streams import keyed_stream_generator

        with pytest.raises(ValueError, match="nonnegative"):
            keyed_stream_generator(root_seed=-1, risk_unit_id="r", purpose="p")
        with pytest.raises(ValueError, match="must not be blank"):
            keyed_stream_generator(root_seed=1, risk_unit_id=" ", purpose="p")
        with pytest.raises(ValueError, match="must not be blank"):
            keyed_stream_generator(root_seed=1, risk_unit_id="r", purpose="")


class TestSampleFromUniforms:
    def test_body_uniform_is_the_percentile_no_gpd(self):
        d = Distribution(p5=0.01, p25=0.05, p50=0.10, p75=0.20, p95=0.40)
        u = np.linspace(0.05, 0.95, 50)
        np.testing.assert_allclose(
            d.sample_from_uniforms(u),
            SplineOracle([0.01, 0.05, 0.10, 0.20, 0.40]).quantile(u),
            rtol=1e-13,
        )

    def test_round_trip_recovers_elicited_percentiles_no_gpd(self):
        d = Distribution(p5=0.01, p25=0.05, p50=0.10, p75=0.20, p95=0.40)
        u = (np.arange(2_000_000) + 0.5) / 2_000_000
        samples = d.sample_from_uniforms(u)
        for q, knot in [
            (5, d.p5),
            (25, d.p25),
            (50, d.p50),
            (75, d.p75),
            (95, d.p95),
        ]:
            assert np.percentile(samples, q) == pytest.approx(knot, rel=1e-3)

    def test_round_trip_recovers_elicited_percentiles_with_gpd(self):
        d = Distribution(p5=1e6, p25=5e6, p50=15e6, p75=50e6, p95=200e6, gpd_shape=0.5)
        u = (np.arange(2_000_000) + 0.5) / 2_000_000
        samples = d.sample_from_uniforms(u)
        for q, knot in [
            (5, d.p5),
            (25, d.p25),
            (50, d.p50),
            (75, d.p75),
            (95, d.p95),
        ]:
            assert np.percentile(samples, q) == pytest.approx(knot, rel=2e-3)

    def test_no_gpd_copula_uniforms_do_not_create_boundary_atoms(self):
        d = Distribution(p5=0.01, p25=0.05, p50=0.10, p75=0.20, p95=0.40)
        u = np.linspace(0.0, 1.0, 100_001, endpoint=False)

        result = d.sample_from_uniforms(u)

        assert np.count_nonzero(result == d.p5) <= 1
        assert np.count_nonzero(np.isclose(result, d.p95, atol=1e-12)) <= 1

    def test_gpd_body_copula_uniforms_do_not_create_boundary_atoms(self):
        d = Distribution(
            p5=1e6,
            p25=5e6,
            p50=15e6,
            p75=50e6,
            p95=200e6,
            gpd_shape=0.5,
        )
        u = np.linspace(0.0, 1.0, 100_001, endpoint=False)

        result = d.sample_from_uniforms(u)

        assert np.count_nonzero(result == d.p5) <= 1

    def test_lower_continuation_is_batch_invariant(self):
        d = Distribution(p5=0.01, p25=1.0, p50=2.0, p75=3.0, p95=4.0)

        singleton = d.quantile(0.049)
        batched = d.sample_from_uniforms(np.array([0.049, 0.04]))[0]

        assert batched == pytest.approx(singleton)

    def test_upper_continuation_is_batch_invariant(self):
        d = Distribution(
            p5=10.0,
            p25=20.0,
            p50=40.0,
            p75=50.0,
            p95=99.0,
            severity_cap=100.0,
        )

        singleton = d.quantile(0.951)
        batched = d.sample_from_uniforms(np.array([0.951, 0.999]))[0]

        assert batched == pytest.approx(singleton)
        assert d.severity_cap is not None
        assert d.quantile(1.0) <= d.severity_cap
