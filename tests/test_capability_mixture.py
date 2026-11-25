"""Exact finite-law checks for mixture atoms, masses and batch precision."""

import numpy as np
import pytest

from catmodel.results.reduction import mean_diagnostics
from catmodel.simulation.capability_mixture import (
    AnalyticMean,
    CapabilityMixture,
    ConditionalCapabilityDraws,
    ConditionalView,
    NativeMean,
    analytic_mixture_mean,
    control_variate_coefficient,
    control_variate_tvar,
    native_mixture_mean,
    summed_view,
    tail_parts,
    weighted_var_tvar,
)
from catmodel.simulation.tail_gate import TailGateControls, view_gate


def draw(path, values, **kwargs):
    view_options = {
        key: kwargs.pop(key)
        for key in ("mean_exists", "variance_exists", "native_mean")
        if key in kwargs
    }
    return ConditionalCapabilityDraws(
        stratum_sha256=path,
        response_sha256="r",
        seed=42,
        views={
            name: ConditionalView(values, **view_options)
            for name, values in (
                {k: np.asarray(v, dtype=float) for k, v in values.items()}
            ).items()
        },
        **kwargs,
    )


def mix(draws, weights):
    return CapabilityMixture({d.stratum_sha256: d for d in draws}, weights)


def test_unequal_stratum_effort_preserves_exact_probability_mass_and_tail():
    zero = draw("zero", {"loss": np.zeros(7)})
    tail = draw("tail", {"loss": (np.arange(10000) + 0.5) / 100})
    result = mix([zero, tail], {"zero": 0.9, "tail": 0.1})
    metrics = result.metrics("loss")
    assert metrics["normalized_empirical_mean"] == pytest.approx(5)
    assert metrics["tail_metrics"]["0.99"]["var"] == pytest.approx(90, abs=0.01)
    assert metrics["tail_metrics"]["0.99"]["tvar"] == pytest.approx(95)
    assert metrics["stratum_diagnostics"]["zero"]["sample_mass"] == pytest.approx(0.9)
    assert metrics["stratum_diagnostics"]["tail"]["sample_mass"] == pytest.approx(0.1)


def test_atoms_at_caps_use_fractional_tail_mass():
    result = mix(
        [
            draw(key, {"loss": [value]})
            for key, value in (("a", 0), ("b", 10), ("c", 100))
        ],
        {"a": 0.9, "b": 0.095, "c": 0.005},
    )
    tail = result.metrics("loss")["tail_metrics"]["0.99"]
    assert {"var": tail["var"], "tvar": tail["tvar"]} == pytest.approx(
        {"var": 10, "tvar": 55}
    )
    assert tail["tvar_estimator"] == SAMPLE_TAIL_MEAN
    assert tail["exceedance_count"] == 1


EXACT_PRECISION = {
    "numerical_standard_error": 0.0,
    "standard_error_upper_bound": 0.0,
    "precision_basis": "deterministic_integration",
    "integration_error_bound": 0.0,
}
SAMPLE_TAIL_MEAN = {
    "method": "control_variate",
    "coefficient": 0.0,
    "coefficient_basis": "no_exact_mean",
}


@pytest.mark.parametrize(
    ("exact", "coefficient", "tvar"),
    [
        # The sample tail mean: the 0.005 atom at 100 plus 0.005 of the VaR
        # atom, 10 + (100 - 10) * 0.005 / 0.01.
        (None, 0.0, 55),
        (10, 0.0, 55),
        # E[min(S, 10)] = 0.1 * 10 = 1, so TVaR = 10 + (10 - 1) / 0.01.
        (10, 1.0, 910),
        # Halfway: the sample mean is 1.45, so 55 + 0.5 (10 - 1.45) / 0.01.
        (10, 0.5, 482.5),
        # A limited expected value above the exact mean cannot resolve the
        # excess, so the estimate is raised to VaR.
        (0.5, 1.0, 10),
    ],
)
def test_control_variate_tvar_weights_the_sample_excess_against_the_exact_mean(
    exact, coefficient, tvar
):
    losses, masses = np.array([0.0, 10, 100]), np.array([0.9, 0.095, 0.005])
    parts = tail_parts(losses, masses, 0.99)
    assert (parts.var, parts.excess, parts.limited) == pytest.approx((10, 0.45, 1.0))
    assert control_variate_tvar(parts, 0.99, exact, coefficient) == pytest.approx(tvar)
    assert weighted_var_tvar(losses, masses, 0.99) == (10, pytest.approx(55))
    with pytest.raises(ValueError, match="analytic mean"):
        control_variate_tvar(parts, 0.99, None, 0.5)


def _single_state(values, *, native_mean=None, variance_exists=True):
    return mix(
        [
            draw(
                "one",
                {"loss": values},
                native_mean=native_mean,
                variance_exists=variance_exists,
            )
        ],
        {"one": 1},
    )


def _regression(excess, loss):
    """Cov(excess, loss) / Var(loss) over equally weighted units."""
    return float(np.cov(excess, loss)[0, 1] / np.var(loss, ddof=1))


def test_the_coefficient_is_estimated_or_set_by_the_law():
    rng = np.random.default_rng(11)
    level = 0.99
    # A light Poisson count with an exact mean: the regression coefficient of
    # the excess over the VaR on the loss, from the years.
    values = 7.0 * rng.poisson(4.0, 2000)
    light = _single_state(
        values, native_mean=NativeMean(28.0, "native_analytic_mean", EXACT_PRECISION)
    )
    tail = light.metrics("loss", levels=(level,))["tail_metrics"][str(level)]
    var, sample = weighted_var_tvar(values, np.full(2000, 1 / 2000), level)
    beta = _regression(np.maximum(values - var, 0.0), values)
    assert 0 < beta < 1
    assert tail["tvar_estimator"] == {
        "method": "control_variate",
        "coefficient": pytest.approx(beta, rel=1e-9),
        "coefficient_basis": "regression",
    }
    assert tail["tvar"] == pytest.approx(
        sample + beta * (28.0 - values.mean()) / (1 - level), rel=1e-12
    )
    # Infinite variance: the excess has infinite variance, so the coefficient
    # is one and only the bounded limited expected value comes from the years.
    values = np.expm1(-0.6 * np.log1p(-rng.random(2000))) / 0.6
    heavy = _single_state(
        values,
        native_mean=NativeMean(2.5, "native_analytic_mean", EXACT_PRECISION),
        variance_exists=False,
    )
    tail = heavy.metrics("loss", levels=(level,))["tail_metrics"][str(level)]
    var = tail["var"]
    assert tail["tvar_estimator"] == {
        "method": "control_variate",
        "coefficient": 1.0,
        "coefficient_basis": "infinite_variance",
    }
    assert tail["tvar"] == pytest.approx(
        var + max(2.5 - np.mean(np.minimum(values, var)), 0.0) / (1 - level),
        rel=1e-12,
    )
    # Without a deterministic mean there is no control: the sample tail mean.
    sampled = _single_state(
        [0, 10, 100],
        native_mean=NativeMean(10, "sampled_mean", {"precision_basis": "monte_carlo"}),
    )
    tail = sampled.metrics("loss")["tail_metrics"]["0.99"]
    assert tail["tvar_estimator"] == SAMPLE_TAIL_MEAN
    assert (
        tail["tvar"]
        == weighted_var_tvar(np.array([0.0, 10, 100]), np.full(3, 1 / 3), 0.99)[1]
    )


def test_the_coefficient_sums_aligned_years_over_states_and_is_clipped():
    """States with the same years are summed year by year, since they can
    share a year's draws; states with different numbers of years count apart;
    and the estimate is clipped to [0, 1], the range of the true coefficient."""
    rng = np.random.default_rng(17)
    a, b = rng.exponential(10.0, 64), rng.exponential(30.0, 64)
    shared = mix(
        [draw("a", {"loss": a}), draw("b", {"loss": b})], {"a": 0.25, "b": 0.75}
    )
    var = 40.0
    loss = 0.25 * a + 0.75 * b
    excess = 0.25 * np.maximum(a - var, 0.0) + 0.75 * np.maximum(b - var, 0.0)
    assert control_variate_coefficient(shared, "loss", var) == pytest.approx(
        _regression(excess, loss), rel=1e-9
    )
    # 64 years in one state and 32 in the other: each state's years count
    # apart, centred on that state's mean, each year weighted by its mass.
    short = b[:32]
    unaligned = mix(
        [draw("a", {"loss": a}), draw("b", {"loss": short})], {"a": 0.25, "b": 0.75}
    )
    covariance = variance = 0.0
    for values, mass in ((a, 0.25), (short, 0.75)):
        weight = mass / len(values)
        x = weight * np.maximum(values - var, 0.0)
        y = weight * values
        covariance += float(np.dot(x - x.mean(), y - y.mean()))
        variance += float(np.dot(y - y.mean(), y - y.mean()))
    assert control_variate_coefficient(unaligned, "loss", var) == pytest.approx(
        covariance / variance, rel=1e-9
    )
    # Year totals whose excess grows faster than their loss (raw 5/3) and
    # one whose excess falls as its loss grows (raw -2) at VaR 10.
    steep = mix(
        [draw("a", {"loss": [9, 30]}), draw("b", {"loss": [9, 0]})],
        {"a": 0.5, "b": 0.5},
    )
    assert control_variate_coefficient(steep, "loss", 10.0) == 1.0
    falling = mix(
        [draw(key, {"loss": [first, 10]}) for key, first in zip("abcd", (30, 0, 0, 0))],
        dict.fromkeys("abcd", 0.25),
    )
    assert control_variate_coefficient(falling, "loss", 10.0) == 0.0
    # A loss that never varies has no regression; the coefficient is zero.
    flat = mix([draw("a", {"loss": [5, 5, 5]})], {"a": 1})
    assert control_variate_coefficient(flat, "loss", 5.0) == 0.0


def test_a_certified_native_mean_is_analytic_with_its_bound():
    """A view has one analytic mean: its states' deterministic native means
    mixed by mass, with their certified integration error bounds mixed the
    same way, zero in closed form. A numerically integrated mean is analytic,
    so it controls the TVaR like a closed-form one; a sampled or missing
    native mean is not."""
    exact = NativeMean(5.0, "native_analytic_mean", EXACT_PRECISION)
    integrated = NativeMean(
        10.0,
        "native_analytic_mean",
        {**EXACT_PRECISION, "integration_error_bound": 0.01},
    )
    sampled = NativeMean(10.0, "sampled_mean", {"precision_basis": "monte_carlo"})

    def state(key, gross, lab):
        return ConditionalCapabilityDraws(
            stratum_sha256=key,
            response_sha256="r",
            seed=42,
            views={
                "gross": ConditionalView(np.array([0.0, 10.0]), native_mean=gross),
                "lab": ConditionalView(np.array([0.0, 20.0]), native_mean=lab),
            },
        )

    one, other = state("one", exact, integrated), state("other", integrated, exact)
    strata = [(one, 0.25), (other, 0.75)]
    assert analytic_mixture_mean(strata, ("gross",)) == AnalyticMean(
        0.25 * 5.0 + 0.75 * 10.0, 0.75 * 0.01
    )
    assert analytic_mixture_mean([(one, 1.0)], ("gross",)) == AnalyticMean(5.0, 0.0)
    # Summed views add their means and their bounds.
    total = analytic_mixture_mean(strata, ("gross", "lab"))
    assert total is not None
    assert (total.value, total.error_bound) == (15.0, pytest.approx(0.01))
    summed = summed_view(one, ("gross", "lab")).native_mean
    assert summed is not None and summed.value == 15.0
    assert summed.precision is not None
    assert summed.precision["integration_error_bound"] == 0.01
    assert summed.precision["precision_basis"] == "deterministic_integration"
    # A sampled native mean is the reported EAL but no analytic mean, and a
    # missing one leaves the EAL to the simulated years.
    estimated = state("estimated", sampled, None)
    assert native_mixture_mean([(estimated, 1.0)], ("gross",)) == 10.0
    assert analytic_mixture_mean([(estimated, 1.0)], ("gross",)) is None
    assert native_mixture_mean([(estimated, 1.0)], ("lab",)) is None
    assert analytic_mixture_mean([(one, 0.5), (estimated, 0.5)], ("gross",)) is None
    assert summed_view(estimated, ("gross",)).native_mean is None
    # The loss 0, 10 or 100 with equal mass: at 0.99 the VaR is 100, the
    # excess over it is zero in every year, and so is its regression
    # coefficient; the integrated mean is admitted as the control, where a
    # sampled one leaves the sample tail mean by rule.
    tail = _single_state([0, 10, 100], native_mean=integrated).metrics("loss")[
        "tail_metrics"
    ]["0.99"]
    assert tail["tvar_estimator"] == {
        "method": "control_variate",
        "coefficient": 0.0,
        "coefficient_basis": "regression",
    }
    assert (
        _single_state([0, 10, 100], native_mean=sampled).metrics("loss")[
            "tail_metrics"
        ]["0.99"]["tvar_estimator"]
        == SAMPLE_TAIL_MEAN
    )


def test_separate_eal_estimator_is_not_silently_renormalized():
    a = draw("a", {"loss": [0, 10]}, native_mean=NativeMean(8, "analytic_native_mean"))
    b = draw("b", {"loss": [0, 20]}, native_mean=NativeMean(12, "analytic_native_mean"))
    result = mix([a, b], {"a": 0.25, "b": 0.75})
    metrics = result.metrics("loss")
    assert metrics["expected_annual_loss"] == 11
    assert metrics["normalized_empirical_mean"] == 8.75
    assert metrics["eal_reconciliation_difference"] == -2.25
    reweighted = mix([a, b], {"a": 0.8, "b": 0.2})
    assert reweighted.metrics("loss")["expected_annual_loss"] == pytest.approx(8.8)
    assert reweighted.draws["a"] is a


def test_zero_weight_unsimulated_strata_cannot_acquire_positive_mass():
    a = draw("a", {"loss": [1, 2]})
    assert mix([a], {"a": 1, "missing": 0}).metrics("loss")[
        "unsimulated_zero_mass_strata"
    ] == ["missing"]
    with pytest.raises(ValueError, match="stored conditional"):
        mix([a], {"a": 0.9, "missing": 0.1})


def _mean_standard_error(mixture, batches=16):
    gate = view_gate(
        mixture, "loss", {}, TailGateControls(batches=batches), mean_exists=True
    )
    return gate["unavailable_reason"], gate.get("empirical_mean_standard_error")


def test_batch_means_standard_error_uses_years_and_path_mass():
    a = draw("a", {"loss": [0, 2, 0, 2]})
    b = draw("b", {"loss": [0, 20, 0, 20]})
    years = mix([a, b], {"a": 0.9, "b": 0.1})
    # Two years per batch: the batch means are equal, so the batches show no
    # numerical uncertainty.
    assert _mean_standard_error(years, batches=2) == (None, 0)
    # One year per batch: the mixture's batch means are 0, 3.8, 0, 3.8, whose
    # standard deviation over sqrt(4) is 1.9 / sqrt(3).
    assert _mean_standard_error(years, batches=4) == (
        None,
        pytest.approx(1.9 / np.sqrt(3)),
    )


def test_infinite_variance_loss_law_withholds_mean_standard_error():
    a = draw("a", {"loss": [0, 100000]}, variance_exists=False)
    mean = mean_diagnostics(mix([a], {"a": 1}), "loss")
    assert mean["empirical_mean_standard_error"] is None
    assert "second moment" in mean["empirical_mean_standard_error_unavailable_reason"]


def test_a_single_year_cannot_be_batched():
    a = draw("a", {"loss": [2]})
    reason, error = _mean_standard_error(mix([a], {"a": 1}))
    assert "fewer than two" in reason
    assert error is None
