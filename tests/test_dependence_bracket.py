"""Small exact couplings and conditional-state controls."""

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from catmodel.results.dependence_bracket import dependence_bracket
from catmodel.simulation.capability_mixture import (
    CapabilityMixture,
    ConditionalCapabilityDraws,
    ConditionalView,
    NativeMean,
)


def _point(tail):
    """A reference's VaR and TVaR without its tail-gate fields."""
    return {key: tail[key] for key in ("var", "tvar")}


def _draw(key, x, y, native=None, mean_exists=True):
    return ConditionalCapabilityDraws(
        stratum_sha256=key,
        response_sha256="response",
        seed=1,
        views={
            name: ConditionalView(
                np.asarray(values, dtype=float),
                mean_exists=mean_exists,
                native_mean=NativeMean(native, "fixture")
                if native is not None
                else None,
            )
            for name, values in {"x": x, "y": y}.items()
        },
    )


def _report(draws, masses, **kwargs):
    kwargs.setdefault("levels", (0.5, 0.9))
    return dependence_bracket(
        CapabilityMixture({row.stratum_sha256: row for row in draws}, masses),
        scenario_views={"first": "x", "second": "y"},
        **kwargs,
    )


def test_shared_rank_preserves_every_marginal_and_sums_ordered_losses():
    # Ten equally likely years with each scenario's losses at shares 0.2, 0.3
    # and 0.5: the comonotone sums of the ordered years are 0, 10, 50 and 60
    # with masses 0.2, 0.3, 0.3 and 0.2.
    x = [0] * 2 + [10] * 3 + [20] * 5
    y = [40] * 2 + [30] * 3 + [0] * 5
    record = _report([_draw("one", x, y)], {"one": 1})
    references = record["references"]
    assert references["executed"]["normalized_empirical_mean"] == pytest.approx(30)
    stress = references["comonotone"]
    assert stress["normalized_empirical_mean"] == pytest.approx(30)
    assert _point(stress["tail"]["0.5"]) == pytest.approx({"var": 10, "tvar": 54})
    assert stress["mean_changing"] is False


def test_couplings_do_not_remove_driver_state_dependence():
    draws = [_draw("quiet", [0], [0]), _draw("severe", [10], [20])]
    references = _report(draws, {"quiet": 0.75, "severe": 0.25})["references"]
    for row in references.values():
        assert row["expected_annual_loss"] == pytest.approx(7.5)
        assert row["normalized_empirical_mean"] == pytest.approx(7.5)
        assert _point(row["tail"]["0.9"]) == pytest.approx({"var": 30, "tvar": 30})


def test_native_eal_is_retained_without_rescaling_empirical_tail_samples():
    references = _report([_draw("one", [0, 10], [0, 20], native=20)], {"one": 1})[
        "references"
    ]
    assert all(row["expected_annual_loss"] == 40 for row in references.values())
    assert references["comonotone"]["normalized_empirical_mean"] == 15
    assert references["comonotone"]["tail"]["0.9"]["var"] == 30


def test_exact_native_means_correct_every_reference_tail_by_the_same_excess():
    exact = NativeMean(
        10,
        "native_analytic_mean",
        {
            "numerical_standard_error": 0.0,
            "standard_error_upper_bound": 0.0,
            "precision_basis": "deterministic_integration",
            "integration_error_bound": 0.0,
        },
    )
    draw = ConditionalCapabilityDraws(
        stratum_sha256="one",
        response_sha256="response",
        seed=1,
        views={
            # Sixteen batches of the same two years.
            "x": ConditionalView(np.array([0.0, 10.0] * 16), native_mean=exact),
            "y": ConditionalView(np.array([20.0, 0.0] * 16), native_mean=exact),
        },
    )
    report = _report([draw], {"one": 1}, levels=(0.5,))
    executed, comonotone = (
        report["references"][key] for key in ("executed", "comonotone")
    )
    # The executed sum is 10 or 20, always at or above its VaR of 10, so its
    # excess moves one for one with the loss: the regression coefficient is one.
    assert report["tvar_estimators"] == {
        "0.5": {
            "method": "control_variate",
            "coefficient": pytest.approx(1.0),
            "coefficient_basis": "regression",
        }
    }
    # The exact mean 20 exceeds the empirical 15; both couplings gain (20 - 15) / 0.5.
    assert _point(executed["tail"]["0.5"]) == pytest.approx({"var": 10, "tvar": 30})
    assert _point(comonotone["tail"]["0.5"]) == pytest.approx({"var": 0, "tvar": 40})


def test_a_certified_scenario_mean_keeps_the_control_and_widens_the_half_width():
    """The two scenarios above with the second's mean of 10 certified only to
    0.2: the sum's mean is 20 within 0.2, so the executed sum keeps its
    control, its coefficient of one and both references' TVaRs. Sixteen
    batches of the same two years give every batch the same TVaR, so the
    batch-means half-width is zero and each reference's TVaR half-width is
    the integration margin alone, 1 x 0.2 / 0.5 = 0.4."""
    precision = {
        "numerical_standard_error": 0.0,
        "standard_error_upper_bound": 0.0,
        "precision_basis": "deterministic_integration",
    }
    exact = NativeMean(
        10, "native_analytic_mean", {**precision, "integration_error_bound": 0.0}
    )
    certified = NativeMean(
        10, "native_analytic_mean", {**precision, "integration_error_bound": 0.2}
    )
    draw = ConditionalCapabilityDraws(
        stratum_sha256="one",
        response_sha256="response",
        seed=1,
        views={
            "x": ConditionalView(np.array([0.0, 10.0] * 16), native_mean=exact),
            "y": ConditionalView(np.array([20.0, 0.0] * 16), native_mean=certified),
        },
    )
    report = _report([draw], {"one": 1}, levels=(0.5,))
    assert report["tvar_estimators"] == {
        "0.5": {
            "method": "control_variate",
            "coefficient": pytest.approx(1.0),
            "coefficient_basis": "regression",
        }
    }
    for key, point in (
        ("executed", {"var": 10, "tvar": 30}),
        ("comonotone", {"var": 0, "tvar": 40}),
    ):
        tail = report["references"][key]["tail"]["0.5"]
        assert _point(tail) == pytest.approx(point)
        assert tail["half_width"] == pytest.approx({"var": 0.0, "tvar": 0.4})


def test_nonfinite_target_mean_withholds_eal_and_tvar():
    references = _report([_draw("one", [0, 1], [1, 0], mean_exists=False)], {"one": 1})[
        "references"
    ]
    for row in references.values():
        assert row["expected_annual_loss"] is None
        assert row["tail"]["0.9"]["tvar"] is None


def test_comonotone_retains_rare_state_mass_below_machine_epsilon():
    report = _report(
        [_draw("usual", [0], [0]), _draw("rare", [1e30], [0])],
        {"usual": 1, "rare": 1e-30},
    )
    shared = report["references"]["comonotone"]
    assert shared["normalized_empirical_mean"] == pytest.approx(1)
    assert shared["tail"]["0.9"]["tvar"] == pytest.approx(10)


def test_overflowing_recoupled_sums_are_unavailable_without_losing_reference():
    report = _report([_draw("one", [0, 1e308], [1e308, 0])], {"one": 1})
    references = report["references"]
    assert references["executed"]["status"] == "available"
    assert references["executed"]["expected_annual_loss"] == pytest.approx(1e308)
    assert references["comonotone"]["status"] == "unavailable"
    assert "exceeds finite numeric range" in references["comonotone"]["reason"]


def test_comonotone_changes_severity_tails_when_occurrence_is_certain():
    references = _report([_draw("one", [1, 10], [10, 1])], {"one": 1})["references"]
    assert references["executed"]["tail"]["0.5"]["tvar"] == 11
    shared = references["comonotone"]
    assert shared["expected_annual_loss"] == 11
    assert _point(shared["tail"]["0.5"]) == pytest.approx({"var": 2, "tvar": 20})


def test_comonotone_bounds_tvar_but_not_var():
    # Independent Bernoulli(0.04) occurrences over 10,000 equally likely
    # years: the executed sum reaches 1 with probability 0.0784, while the
    # comonotone sum is 2 with probability 0.04.
    counts = (16, 384, 384, 9216)
    draw = _draw(
        "one",
        np.repeat([1, 1, 0, 0], counts),
        np.repeat([1, 0, 1, 0], counts),
    )
    references = _report([draw], {"one": 1}, levels=(0.95,))["references"]
    executed = references["executed"]["tail"]["0.95"]
    comonotone = references["comonotone"]["tail"]["0.95"]
    assert _point(executed) == pytest.approx({"var": 1, "tvar": 1.032})
    assert _point(comonotone) == pytest.approx({"var": 0, "tvar": 1.6})
    assert comonotone["tvar"] > executed["tvar"] and comonotone["var"] < executed["var"]


def test_precision_reports_kish_effective_sample_size_of_executed_weights():
    report = _report(
        [_draw("first", [0, 1], [1, 0]), _draw("second", [0, 2], [2, 0])],
        {"first": 0.5, "second": 0.5},
    )
    precision = report["precision"]
    assert precision["effective_sample_size"] == pytest.approx(4)
    assert precision["effective_tail_observations"] == pytest.approx(
        {"0.5": 2, "0.9": 0.4}
    )


def test_bracket_and_portfolio_metric_choose_the_same_estimator_for_one_sum():
    from catmodel.simulation.capability_mixture import summed_view

    rng = np.random.default_rng(17)
    exact = NativeMean(
        2.5,
        "native_analytic_mean",
        {
            "numerical_standard_error": 0.0,
            "standard_error_upper_bound": 0.0,
            "precision_basis": "deterministic_integration",
            "integration_error_bound": 0.0,
        },
    )
    draw = ConditionalCapabilityDraws(
        stratum_sha256="one",
        response_sha256="response",
        seed=1,
        views={
            name: ConditionalView(
                np.expm1(-0.6 * np.log1p(-rng.random(2000))) / 0.6,
                variance_exists=False,
                native_mean=exact,
            )
            for name in ("x", "y")
        },
    )
    report = _report([draw], {"one": 1}, levels=(0.99,))
    portfolio = CapabilityMixture(
        {"one": replace(draw, views={"sum": summed_view(draw, ("x", "y"))})},
        {"one": 1},
    )
    tails = cast(
        dict[str, Any], portfolio.metrics("sum", levels=(0.99,))["tail_metrics"]
    )
    tail = tails["0.99"]
    # Infinite variance travels with the sum, so both take coefficient one.
    assert tail["tvar_estimator"] == {
        "method": "control_variate",
        "coefficient": 1.0,
        "coefficient_basis": "infinite_variance",
    }
    assert report["tvar_estimators"] == {"0.99": tail["tvar_estimator"]}
    assert report["references"]["executed"]["tail"]["0.99"]["tvar"] == pytest.approx(
        tail["tvar"]
    )
