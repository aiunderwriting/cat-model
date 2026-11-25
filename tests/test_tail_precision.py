"""Tail estimators, support diagnostics and abstention for heavy-tailed reporting."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import brentq

from catmodel.results.reduction import (
    mean_diagnostics,
    tail_diagnostics,
)
from catmodel.simulation.capability_mixture import (
    CapabilityMixture,
    ConditionalCapabilityDraws,
    ConditionalView,
    NativeMean,
    weighted_var_tvar,
)

ZERO_LAW = NativeMean(
    0,
    "exact_zero",
    {
        "precision_basis": "deterministic_integration",
        "integration_error_bound": 0.0,
        "known_zero_law": True,
    },
)


def stratum(values, *, key="s", shape=0.4, **kwargs):
    return ConditionalCapabilityDraws(
        stratum_sha256=key,
        response_sha256="response",
        seed=0,
        views={
            "gross": ConditionalView(
                np.asarray(values, dtype=float),
                mean_exists=shape < 1,
                variance_exists=shape < 0.5,
            )
        },
        **kwargs,
    )


def mixture(values, *, shape=0.4, **kwargs):
    return CapabilityMixture({"s": stratum(values, shape=shape, **kwargs)}, {"s": 1.0})


def gpd(rng, n, shape, scale=1.0):
    return scale * np.expm1(-shape * np.log1p(-rng.random(n))) / shape


def zero_stratum(key="z"):
    return ConditionalCapabilityDraws(
        stratum_sha256=key,
        response_sha256="response",
        seed=0,
        views={"gross": ConditionalView(np.zeros(1), native_mean=ZERO_LAW)},
    )


def _exact_mean(value):
    return NativeMean(
        value,
        "native_analytic_mean",
        {
            "numerical_standard_error": 0.0,
            "standard_error_upper_bound": 0.0,
            "precision_basis": "deterministic_integration",
            "integration_error_bound": 0.0,
        },
    )


@pytest.mark.stochastic
@pytest.mark.parametrize("shape", [0.45, 0.65])
def test_control_variate_tvar_removes_sample_tail_bias(shape):
    # The design: 12 equal-mass parameter states of 256 years. A year has
    # one loss with probability q, GPD with a state-specific scale, so each
    # state's exact mean and the mixture's VaR/TVaR are closed form. Shape 0.65
    # has infinite variance, like the governed tail classes; 0.45 does not.
    q, levels, trials = 0.05, (0.99, 0.995, 0.999), 200
    scales = 1 + np.arange(12) / 4
    keys = [f"s{index}" for index in range(12)]

    def survival(x):
        return q * (1 + shape * x / scales) ** (-1 / shape)

    truth = {}
    for level in levels:
        var = brentq(
            lambda x, level=level: float(np.mean(survival(x))) - (1 - level), 0, 1e12
        )
        excess = np.mean((scales + shape * var) / (1 - shape) * survival(var))
        truth[level] = var + excess / (1 - level)
    rng = np.random.default_rng(round(1000 * shape) + 7)
    sample = {level: [] for level in levels}
    reported_tvar = {level: [] for level in levels}
    for _ in range(trials):
        draws = {}
        for key, scale in zip(keys, scales, strict=True):
            losses = np.where(rng.random(256) < q, gpd(rng, 256, shape, scale), 0.0)
            draws[key] = ConditionalCapabilityDraws(
                stratum_sha256=key,
                response_sha256="response",
                seed=0,
                views={
                    "gross": ConditionalView(
                        losses,
                        variance_exists=shape < 0.5,
                        native_mean=_exact_mean(q * scale / (1 - shape)),
                    )
                },
            )
        law = CapabilityMixture(draws, dict.fromkeys(keys, 1 / 12))
        tails = tail_diagnostics(law, "gross", levels=levels)["tails"]
        for level in levels:
            tail = tails[str(level)]
            # Infinite variance sets the coefficient to one by law; finite
            # variance estimates it.
            assert tail["tvar_estimator"]["coefficient_basis"] == (
                "infinite_variance" if shape >= 0.5 else "regression"
            )
            reported_tvar[level].append(tail["tvar"] / truth[level])
            sample[level].append(
                weighted_var_tvar(*law.samples("gross"), level)[1] / truth[level]
            )
    # About three expected exceedances at 0.999: the sample tail mean is badly
    # biased low.
    assert np.median(sample[0.999]) < 0.8
    for level in levels:
        reported, plain = np.array(reported_tvar[level]), np.array(sample[level])
        # The reported estimator never has the larger mean squared error.
        assert np.mean((reported - 1) ** 2) <= np.mean((plain - 1) ** 2)
        if shape >= 0.5 or level < 0.999:
            assert 0.93 < np.median(reported) < 1.05
        # With finite variance and about three exceedances, a sample that
        # misses the far tail also understates the regression coefficient, so
        # much of the low median bias remains.


@pytest.mark.stochastic
@pytest.mark.parametrize("shape", [0.3, 0.65])
def test_control_variate_tvar_is_unbiased_and_its_interval_covers(shape):
    """Four equal-mass states of 4,096 years, a loss with probability 0.05
    and GPD severities of unit scale, whose VaR and TVaR are closed form: at
    the 1-in-100 level, with about 160 years beyond the VaR, the reported
    TVaR's mean ratio to the truth is within 2% of one, and its 95%
    batch-means interval covers the truth in at least 85 of 100 trials, both
    with finite variance (shape 0.3, an estimated coefficient) and without
    (shape 0.65, coefficient one)."""
    q, level, trials, states, years = 0.05, 0.99, 100, 4, 4096
    var = ((q / (1 - level)) ** shape - 1) / shape
    truth = var + (1 + shape * var) / (1 - shape)
    keys = [f"s{index}" for index in range(states)]
    rng = np.random.default_rng(round(1000 * shape) + 11)
    ratios, covered = [], 0
    for _ in range(trials):
        draws = {
            key: ConditionalCapabilityDraws(
                stratum_sha256=key,
                response_sha256="response",
                seed=0,
                views={
                    "gross": ConditionalView(
                        np.where(rng.random(years) < q, gpd(rng, years, shape), 0.0),
                        variance_exists=shape < 0.5,
                        native_mean=_exact_mean(q / (1 - shape)),
                    )
                },
            )
            for key in keys
        }
        law = CapabilityMixture(draws, dict.fromkeys(keys, 1 / states))
        tail = tail_diagnostics(law, "gross", levels=(level,))["tails"][str(level)]
        coefficient = tail["tvar_estimator"]["coefficient"]
        if shape >= 0.5:
            assert coefficient == 1.0
        else:
            assert 0 < coefficient < 1
        ratios.append(tail["tvar"] / truth)
        low, high = tail["tvar_numerical_interval_95"]
        covered += low <= truth <= high
    assert abs(np.mean(ratios) - 1) < 0.02
    assert covered >= 85


def test_tail_support_distinguishes_weight_and_dollar_concentration():
    values = np.concatenate((np.zeros(100), np.ones(99), [1e300]))
    law = mixture(values)
    tail = tail_diagnostics(law, "gross", levels=(0.5,))["tails"]["0.5"]
    support = tail["tail_support_by_stratum"]["s"]
    assert tail["tail_weight_effective_sample_size"] == pytest.approx(100)
    assert support["tail_mass_effective_units"] == pytest.approx(100)
    assert support["tail_excess_effective_units"] == pytest.approx(1)


def test_mean_reports_batch_standard_error_without_a_coverage_claim():
    values = np.arange(1, 161, dtype=float)
    result = mean_diagnostics(mixture(values), "gross")
    means = values.reshape(16, 10).mean(axis=1)
    assert result["empirical_mean_standard_error"] == pytest.approx(
        np.std(means, ddof=1) / 4
    )
    assert result["standard_error_basis"] == "batch_means"
    assert result["empirical_mean_standard_error_unavailable_reason"] is None
    assert result["empirical_mean_numerical_interval_95"] is None
    assert (
        "No coverage-valid empirical-mean interval"
        in (result["interval_unavailable_reason"])
    )
    short = mean_diagnostics(mixture(values[:15]), "gross")
    assert short["empirical_mean_standard_error"] is None
    assert (
        "needs at least 16" in short["empirical_mean_standard_error_unavailable_reason"]
    )


def test_structural_zero_law_keeps_exact_intervals_with_one_year():
    zero = ConditionalView(np.zeros(1), native_mean=ZERO_LAW)
    law = mixture([0])
    draw = replace(law.draws["s"], views={"gross": zero})
    law = CapabilityMixture({"s": draw}, {"s": 1.0})
    tail = tail_diagnostics(law, "gross")["tails"]["0.999"]
    assert tail["strict_exceedance_count"] == 0
    assert tail["var_numerical_interval_95"] == [0, 0]
    assert tail["tvar_numerical_interval_95"] == [0, 0]
    assert tail["var_interval_unavailable_reason"] is None
    assert tail["tvar_interval_unavailable_reason"] is None
    mean = mean_diagnostics(law, "gross")
    assert mean["empirical_mean_numerical_interval_95"] == [0, 0]
    assert mean["empirical_mean_standard_error"] == 0


def test_loss_contract_rejects_negative_views_before_interval_construction():
    with pytest.raises(ValueError, match="nonnegative"):
        mixture([-1, 0, 1, 2])


def test_infinite_variance_mean_standard_error_is_withheld_with_explicit_reason():
    report = mean_diagnostics(mixture([1, 2, 3, 4], shape=0.7), "gross")
    assert report["empirical_mean_standard_error"] is None
    assert "second moment" in report["empirical_mean_standard_error_unavailable_reason"]
    assert report["empirical_mean_numerical_interval_95"] is None


def test_tail_levels_without_enough_exceedances_are_withheld():
    from catmodel.results.reduction import withhold_unsupported_tails
    from catmodel.simulation.capability_mixture import statistic_status

    rng = np.random.default_rng(5)
    law = mixture(gpd(rng, 4000, 0.4))
    levels = (0.99, 0.999)
    metric = {
        "statistics": {
            f"{name}::{level}": statistic_status()
            for level in levels
            for name in ("var", "tvar")
        },
        "tail_diagnostics": tail_diagnostics(law, "gross", levels=levels),
    }
    withhold_unsupported_tails(metric)
    statistics = metric["statistics"]
    # About 40 exceedances leave about 2.5 per batch at the 1-in-100 level and
    # about 0.25 at 1-in-1,000: neither converges.
    for key in ("var::0.99", "tvar::0.99", "var::0.999", "tvar::0.999"):
        assert statistics[key]["status"] == "withheld"
        assert "at least 10 in each" in str(statistics[key]["reason"])
