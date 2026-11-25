from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from catmodel.probability.distributions import Distribution
from catmodel.probability.liability import LiabilityAttenuation
from test_support.sampling import sample_liability_shares, sample_severity


def _authored(**overrides: Any) -> LiabilityAttenuation:
    """Build an attenuation from neutral explicit parameters plus overrides."""
    parameters: dict[str, Any] = {
        "mean": 0.5,
        "concentration": 10.0,
        "recovery_rate": 1.0,
        "severity_slope": 0.0,
        "regulatory_recovery_threshold": 1e9,
        "regulatory_recovery_uplift": 0.0,
    }
    return LiabilityAttenuation(**(parameters | overrides))


class TestLiabilityAttenuation:
    def test_basic_construction(self) -> None:
        attn = _authored(
            mean=0.5,
            concentration=10.0,
        )
        assert attn.conditional_mean == 0.5

    def test_parameters_have_no_defaults(self) -> None:
        with pytest.raises(TypeError, match="recovery_rate"):
            LiabilityAttenuation(  # ty: ignore[missing-argument]
                mean=0.5,
                concentration=10.0,
            )

    def test_invalid_mean_raises(self) -> None:
        with pytest.raises(ValueError, match="mean"):
            _authored(
                mean=0.0,
                concentration=10.0,
            )
        with pytest.raises(ValueError, match="mean"):
            _authored(
                mean=1.0,
                concentration=10.0,
            )

    def test_invalid_concentration_raises(self) -> None:
        with pytest.raises(ValueError, match="concentration"):
            _authored(
                mean=0.5,
                concentration=-1.0,
            )

    def test_invalid_severity_slope_raises(self) -> None:
        with pytest.raises(ValueError, match="severity_slope"):
            _authored(
                mean=0.5,
                concentration=10.0,
                severity_slope=-0.01,
            )

    def test_excessive_severity_slope_raises(self) -> None:
        with pytest.raises(ValueError, match=r"severity_slope.*\[0, 0.1\]"):
            _authored(
                mean=0.5,
                concentration=10.0,
                severity_slope=0.1001,
            )


class TestSeverityConditionalSampling:
    def test_higher_severity_higher_attenuation(self) -> None:
        attn = _authored(
            mean=0.4,
            concentration=10.0,
            severity_slope=0.025,
            median_severity=1_000_000,
        )
        rng = np.random.default_rng(42)
        n = 100_000

        low_sev = np.full(n, 50_000.0)  # well below median
        high_sev = np.full(n, 50_000_000.0)  # well above median

        low_attn = sample_liability_shares(attn, low_sev, rng=rng)
        rng2 = np.random.default_rng(42)
        high_attn = sample_liability_shares(attn, high_sev, rng=rng2)

        assert np.mean(high_attn) > np.mean(low_attn)

    @pytest.mark.parametrize(
        ("severity_slope", "median_severity"), [(0.0, 1_000_000), (0.025, None)]
    )
    def test_unconditioned_share_keeps_recovery_rate_and_unconditional_mean(
        self, severity_slope, median_severity
    ) -> None:
        # Zero slope, or no median, leaves the recovered share at
        # mean / recovery_rate whatever the severity.
        attn = _authored(
            mean=0.4,
            concentration=10.0,
            severity_slope=severity_slope,
            median_severity=median_severity,
            recovery_rate=0.55,
            regulatory_recovery_uplift=0.0,
        )
        draws = sample_liability_shares(
            attn, np.logspace(4, 9, 100_000), rng=np.random.default_rng(42)
        )
        assert np.mean(draws > 0) == pytest.approx(0.55, abs=0.01)
        assert np.mean(draws) == pytest.approx(0.4, abs=0.01)

    def test_positive_correlation_increases_eal(self) -> None:
        attn_corr = _authored(
            mean=0.4,
            concentration=8.0,
            severity_slope=0.025,
            median_severity=1_000_000,
        )
        attn_ind = _authored(
            mean=0.4,
            concentration=8.0,
            severity_slope=0.0,
        )

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        n = 200_000

        # Pair severity draws to isolate the correlation effect.

        sev_dist = Distribution(
            p5=50_000,
            p25=100_000,
            p50=500_000,
            p75=10_000_000,
            p95=50_000_000,
        )
        sevs = sample_severity(sev_dist, n, rng=np.random.default_rng(0))
        sevs = np.maximum(sevs, 1.0)

        corr_factors = sample_liability_shares(attn_corr, sevs, rng=rng1)
        ind_factors = sample_liability_shares(attn_ind, sevs, rng=rng2)

        eal_corr = np.mean(sevs * corr_factors)
        eal_ind = np.mean(sevs * ind_factors)

        assert eal_corr > eal_ind

    def test_sensitivity_range(self) -> None:
        n = 200_000

        sev_dist = Distribution(
            p5=50_000,
            p25=100_000,
            p50=500_000,
            p75=10_000_000,
            p95=50_000_000,
        )
        sevs = sample_severity(sev_dist, n, rng=np.random.default_rng(0))
        sevs = np.maximum(sevs, 1.0)

        eals = {}
        for slope in [0.020, 0.025, 0.030]:
            attn = _authored(
                mean=0.4,
                concentration=8.0,
                severity_slope=slope,
                median_severity=1_000_000,
            )
            factors = sample_liability_shares(attn, sevs, rng=np.random.default_rng(42))
            eals[slope] = np.mean(sevs * factors)

        assert eals[0.020] < eals[0.025] < eals[0.030]


class TestSeverityConditionalUniformTransform:
    @staticmethod
    def _attenuation() -> LiabilityAttenuation:
        return _authored(
            mean=0.35,
            concentration=8.0,
            recovery_rate=0.55,
            severity_slope=0.025,
            median_severity=1_000_000,
            regulatory_recovery_threshold=1_000_000_000,
            regulatory_recovery_uplift=0.15,
        )

    def test_full_batch_equals_arbitrary_contiguous_slices(self) -> None:
        attn = self._attenuation()
        n = 4_117
        severities = np.geomspace(10_000.0, 100_000_000_000.0, n)
        recovery_u = np.random.default_rng(123).random(n)
        share_u = np.random.default_rng(456).random(n)

        full = attn.sample_conditional_from_uniforms(
            severities,
            recovery_u,
            share_u,
        )
        bounds = (0, 1, 19, 257, 1_024, 3_333, n)
        sliced = np.concatenate(
            [
                attn.sample_conditional_from_uniforms(
                    severities[start:stop],
                    recovery_u[start:stop],
                    share_u[start:stop],
                )
                for start, stop in pairwise(bounds)
            ]
        )

        np.testing.assert_array_equal(sliced, full)

    def test_matches_zero_inflated_beta_moment_oracle(self) -> None:
        attn = self._attenuation()
        draws_per_level = 120_000
        severity_levels = np.array([100_000.0, 1_000_000_000.0, 100_000_000_000.0])
        severities = np.repeat(severity_levels, draws_per_level)
        recovery_u = np.random.default_rng(321).random(severities.size)
        share_u = np.random.default_rng(654).random(severities.size)

        factors = attn.sample_conditional_from_uniforms(
            severities,
            recovery_u,
            share_u,
        )
        assert attn.median_severity is not None
        for index, severity in enumerate(severity_levels):
            start = index * draws_per_level
            stop = start + draws_per_level
            sample = factors[start:stop]
            recovered_sample = sample[sample > 0.0]

            log10_severity = np.log10(severity)
            uplift_fraction = 1.0 / (
                1.0
                + np.exp(
                    -(log10_severity - np.log10(attn.regulatory_recovery_threshold))
                )
            )
            recovery = min(
                attn.recovery_rate + attn.regulatory_recovery_uplift * uplift_fraction,
                1.0,
            )
            effective_unconditional = np.clip(
                attn.mean
                + attn.severity_slope * np.log10(severity / attn.median_severity),
                0.01,
                0.99,
            )
            conditional_mean = np.clip(
                effective_unconditional / attn.recovery_rate,
                0.01,
                0.99,
            )
            alpha = conditional_mean * attn.concentration
            conditional_second_moment = (
                alpha
                * (alpha + 1.0)
                / (attn.concentration * (attn.concentration + 1.0))
            )
            expected_mean = recovery * conditional_mean
            expected_variance = recovery * conditional_second_moment - expected_mean**2

            assert np.mean(sample > 0.0) == pytest.approx(recovery, abs=0.004)
            assert np.mean(recovered_sample) == pytest.approx(
                conditional_mean,
                abs=0.004,
            )
            assert np.mean(sample) == pytest.approx(expected_mean, abs=0.004)
            assert np.var(sample) == pytest.approx(expected_variance, abs=0.004)

    @pytest.mark.parametrize(
        ("recovery_u", "share_u", "match"),
        [
            (np.array([0.1, 0.2]), np.array([0.2]), "share_u"),
            (np.array([np.nan, 0.1]), np.array([0.2, 0.3]), "recovery_u"),
            (np.array([0.1, 0.2]), np.array([-0.1, 0.3]), "share_u"),
            (np.array([0.1, 0.2]), np.array([0.3, 1.0]), "share_u"),
        ],
    )
    def test_rejects_invalid_uniform_vectors(
        self,
        recovery_u: np.ndarray,
        share_u: np.ndarray,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            self._attenuation().sample_conditional_from_uniforms(
                np.array([1_000_000.0, 2_000_000.0]),
                recovery_u,
                share_u,
            )


class TestRecoveryRateDecomposition:
    def test_conditional_mean_computed(self) -> None:
        attn = _authored(
            mean=0.35,
            concentration=8.0,
            recovery_rate=0.50,
        )
        assert abs(attn.conditional_mean - 0.70) < 1e-10

    def test_mean_over_recovery_rate_raises(self) -> None:
        with pytest.raises(ValueError, match=r"mean.*recovery_rate"):
            _authored(
                mean=0.95,
                concentration=8.0,
                recovery_rate=0.50,
            )

    def test_mean_at_boundary_ok(self) -> None:
        attn = _authored(
            mean=0.495,
            concentration=8.0,
            recovery_rate=0.50,
        )
        assert abs(attn.conditional_mean - 0.99) < 1e-10

    def test_invalid_recovery_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="recovery_rate"):
            _authored(
                mean=0.35,
                concentration=8.0,
                recovery_rate=0.0,
            )
        with pytest.raises(ValueError, match="recovery_rate"):
            _authored(
                mean=0.35,
                concentration=8.0,
                recovery_rate=1.5,
            )


def test_zero_uplift_disables_severity_recovery_ramp() -> None:
    attenuation = _authored(
        mean=0.35,
        concentration=8.0,
        recovery_rate=0.55,
        severity_slope=0.025,
        median_severity=1_000_000,
        regulatory_recovery_threshold=1e9,
        regulatory_recovery_uplift=0,
    )
    # Recovery remains 0.55 on either side of the threshold. Probe the exact
    # gate boundary independently of the conditional Beta share.
    severities = np.repeat([1e5, 1e10], 2)
    values = attenuation.sample_conditional_from_uniforms(
        severities, np.tile([np.nextafter(0.55, 0), 0.55], 2), np.full(4, 0.5)
    )
    np.testing.assert_array_equal(values > 0, [True, False, True, False])
