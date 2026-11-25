"""The tail convergence gate: batch means over years, withholding and doubling."""

import json
import math
from dataclasses import replace
from functools import partial
from typing import Any, cast

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import t as student_t

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.contracts.incident import DEFAULT_WORK_ALLOWANCES, ExecutionPlan
from catmodel.hashing import canonical_json_sha256
from catmodel.results import execution
from catmodel.results.execution import (
    load_run_capture,
    replay_capture,
    run_setup,
    setup_workload,
)
from catmodel.results.reduction import (
    ReductionSpec,
    mean_diagnostics,
    reduce_captures,
    tail_diagnostics,
    withhold_unsupported_tails,
)
from catmodel.results.results_file import load_results
from catmodel.setup.contextual_setup import compile_contextual_setup
from catmodel.setup.store import build_setup
from catmodel.simulation import incident_runtime
from catmodel.simulation.capability_mixture import (
    CapabilityMixture,
    ConditionalCapabilityDraws,
    ConditionalView,
    NativeMean,
    control_variate_tvar,
    tail_estimates,
    tail_parts,
    weighted_var_tvar,
)
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_runtime import simulate_incident_state
from catmodel.simulation.tail_gate import (
    INTEGRATION_MARGIN_REASON,
    MIN_TAIL_EFFECTIVE_UNITS,
    SAMPLE_TVAR_MOMENT_REASON,
    TailGateControls,
    batch_layout,
    doubling_limits,
    paired_effect,
    unconverged_drivers,
    view_gate,
    year_prefix,
)
from test_support.contextual_risk import (
    fixture_artifact,
    governed_contextual_answer,
    governed_contextual_instrument,
)
from test_support.results import (
    available_metric,
    present,
    reduced_arm,
    view_diagnostics,
)
from test_support.run_setup import setup_from_payload, with_execution
from test_support.study import _study, study_payload
from tests.test_small_events import _kernel

ZERO_LAW = NativeMean(
    0,
    "exact_zero",
    {
        "precision_basis": "deterministic_integration",
        "integration_error_bound": 0.0,
        "known_zero_law": True,
    },
)


_VECTORIZED = DEFAULT_WORK_ALLOWANCES["max_vectorized_events_per_year"]


def _stratum(values, key="s", *, mean=None, bound=0.0, variance=True, **kwargs):
    """One state's years of ``gross``; ``mean`` is its analytic native mean,
    closed-form unless ``bound`` gives it a certified integration error."""
    return ConditionalCapabilityDraws(
        stratum_sha256=key,
        response_sha256="response",
        seed=0,
        views={
            "gross": ConditionalView(
                np.asarray(values, dtype=float),
                variance_exists=variance,
                native_mean=None
                if mean is None
                else NativeMean(
                    mean,
                    "native_analytic_mean",
                    {
                        "precision_basis": "deterministic_integration",
                        "integration_error_bound": bound,
                    },
                ),
            )
        },
        **kwargs,
    )


def _mixture(*strata, masses=None):
    keys = [draw.stratum_sha256 for draw in strata]
    masses = masses or [1 / len(strata)] * len(strata)
    return CapabilityMixture(
        dict(zip(keys, strata, strict=True)), dict(zip(keys, masses, strict=True))
    )


def _gate(mixture, levels=(0.9,), **controls):
    controls = TailGateControls(**controls)
    return view_gate(
        mixture,
        "gross",
        tail_estimates(mixture, "gross", levels),
        controls,
        mean_exists=True,
    )


def test_controls_validate_and_default_to_headline_drivers():
    controls = TailGateControls.from_controls(None)
    assert (
        controls.tolerance,
        controls.batches,
        controls.max_reference_state_years,
        controls.max_comparison_state_years,
    ) == (0.1, 16, None, None)
    assert controls.drivers(
        ("gross", "welfare", "lab_loss::lab", "scenario::a::gross")
    ) == (
        "gross",
        "welfare",
        "lab_loss::lab",
    )
    chosen = TailGateControls.from_controls(
        {"views": ["scenario::a::gross"], "levels": [0.9]}
    )
    assert chosen.drivers(("gross", "scenario::a::gross")) == ("scenario::a::gross",)
    assert chosen.record()["levels"] == [0.9]
    # A null cap, views or levels is the default, as the frozen control writes.
    assert TailGateControls.from_controls(
        {
            "max_reference_state_years": None,
            "max_comparison_state_years": None,
            "views": None,
            "levels": None,
        }
    ) == TailGateControls.from_controls(TailGateControls().record_controls())
    with pytest.raises(ValueError, match="tail gate views"):
        TailGateControls(views="gross")  # ty: ignore[invalid-argument-type]
    for bad in (
        {"tolerance": 0},
        {"tolerance": 1.5},
        {"batches": 1},
        {"batches": True},
        {"max_reference_state_years": 0},
        {"max_comparison_state_years": True},
        {"max_years": 4096},
        {"views": []},
        {"views": "gross"},
        {"views": [""]},
        {"views": [7]},
        {"levels": [1.0]},
        {"levels": 0.9},
        {"levels": []},
        {"levels": [True]},
        {"tolerance": None},
        {"tolerance": "0.1"},
        {"batches": None},
        {"batches": 16.0},
        {"max_reference_state_years": 4096.0},
        {"other": 1},
        ["views"],
    ):
        with pytest.raises(ValueError, match=r"tail gate|tail_gate"):
            TailGateControls.from_controls(bad)


def test_batches_split_each_strata_years_contiguously():
    years = _stratum(np.arange(64), "a")
    uneven = _stratum(np.arange(40), "b")
    layout = batch_layout(_mixture(years, uneven).positive_strata(), 16)
    assert not isinstance(layout, str)
    batch, stratum = layout
    np.testing.assert_array_equal(batch[:64], np.arange(64) // 4)
    # Batch sizes differ by one year where the years are not a multiple of 16.
    np.testing.assert_array_equal(
        np.bincount(batch[64:]), [3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2]
    )
    np.testing.assert_array_equal(batch[64:], np.sort(batch[64:]))
    np.testing.assert_array_equal(stratum, np.repeat([0, 1], [64, 40]))
    short = _mixture(_stratum(np.arange(15))).positive_strata()
    assert "needs at least 16" in batch_layout(short, 16)


def test_half_widths_are_the_spread_of_the_batch_statistics():
    rng = np.random.default_rng(3)
    first, second = rng.exponential(1.0, 640), rng.exponential(2.0, 640)
    mixture = _mixture(_stratum(first, "a"), _stratum(second, "b"), masses=[0.25, 0.75])
    gate = _gate(mixture, batches=8)
    row = gate["levels"]["0.9"]
    estimate = tail_estimates(mixture, "gross", (0.9,))[0.9]
    var_k, tvar_k, means = [], [], []
    for k in range(8):
        part = slice(80 * k, 80 * (k + 1))
        losses = np.concatenate([first[part], second[part]])
        masses = np.concatenate([np.full(80, 0.25 / 80), np.full(80, 0.75 / 80)])
        var, tvar = weighted_var_tvar(losses, masses, 0.9)
        var_k.append(var)
        tvar_k.append(tvar)
        means.append(float(np.dot(losses, masses)))
    critical = float(student_t.ppf(0.975, 7))
    assert row["var"]["half_width"] == pytest.approx(
        critical * np.std(var_k, ddof=1) / math.sqrt(8), rel=1e-12
    )
    assert row["tvar"]["half_width"] == pytest.approx(
        critical * np.std(tvar_k, ddof=1) / math.sqrt(8), rel=1e-12
    )
    assert row["var"]["relative_half_width"] == pytest.approx(
        row["var"]["half_width"] / estimate.var
    )
    assert gate["empirical_mean_standard_error"] == pytest.approx(
        np.std(means, ddof=1) / math.sqrt(8), rel=1e-12
    )


def test_states_sharing_a_years_draws_are_not_treated_as_independent():
    """Two states with the same years: the mixture is one state, so its
    standard error must equal one state's, not one state's over sqrt(2)."""
    values = np.random.default_rng(5).exponential(1.0, 1600)
    shared = _mixture(_stratum(values, "a"), _stratum(values, "b"))
    single = _mixture(_stratum(values, "a"))
    assert mean_diagnostics(shared, "gross")[
        "empirical_mean_standard_error"
    ] == pytest.approx(
        mean_diagnostics(single, "gross")["empirical_mean_standard_error"], rel=1e-12
    )


def test_unconverged_and_sparse_levels_are_withheld_with_their_half_width():
    rng = np.random.default_rng(7)
    law = _mixture(_stratum(rng.pareto(1.5, 1600) + 1))
    levels = (0.5, 0.99)
    tails = tail_diagnostics(
        law, "gross", levels=levels, controls=TailGateControls(tolerance=0.02)
    )
    metric = {
        "statistics": {
            f"{functional}::{level}": {"status": "available", "reason": None}
            for level in levels
            for functional in ("var", "tvar")
        },
        "tail_diagnostics": tails,
    }
    withhold_unsupported_tails(metric)
    statistics = metric["statistics"]
    median = tails["tails"]["0.5"]["convergence"]
    if median["var"]["status"] == "converged":
        assert statistics["var::0.5"]["status"] == "available"
    else:
        assert "half-width is ±" in str(statistics["var::0.5"]["reason"])
    # 1,600 years in 16 batches leave about one year above the 99% VaR per batch.
    sparse = tails["tails"]["0.99"]["convergence"]
    assert sparse["minimum_batch_exceedances"] < MIN_TAIL_EFFECTIVE_UNITS
    for functional in ("var", "tvar"):
        status = statistics[f"{functional}::0.99"]
        assert status["status"] == "withheld"
        assert "sparsest of 16 batches" in str(status["reason"])
    low, high = tails["tails"]["0.99"]["var_numerical_interval_95"]
    var = tails["tails"]["0.99"]["var"]
    assert low == pytest.approx(max(var - sparse["var"]["half_width"], 0.0))
    assert high == pytest.approx(var + sparse["var"]["half_width"])


def test_a_tight_tail_converges_and_stays_available():
    law = _mixture(_stratum(np.random.default_rng(9).normal(100.0, 1.0, 3200).clip(0)))
    tails = tail_diagnostics(law, "gross", levels=(0.9,))
    row = tails["tails"]["0.9"]["convergence"]
    assert row["var"]["status"] == row["tvar"]["status"] == "converged"
    assert row["minimum_batch_exceedances"] >= MIN_TAIL_EFFECTIVE_UNITS
    assert tails["tails"]["0.9"]["var_interval_method"] == "batch_means"


def test_the_tvar_half_width_is_the_reported_estimators_own():
    """A finite-variance law with an exact mean: every batch's TVaR is the
    control-variate estimate at the coefficient set on all years, and the
    half-width is their spread alone, not the larger of it and another
    estimate's."""
    rng = np.random.default_rng(11)
    losses = np.where(rng.random(1600) < 0.2, rng.exponential(1.0, 1600), 0.0)
    mixture = _mixture(_stratum(losses, mean=0.2))
    estimate = tail_estimates(mixture, "gross", (0.9,))[0.9]
    row = _gate(mixture)["levels"]["0.9"]
    assert estimate.coefficient_basis == "regression"
    assert 0 < estimate.coefficient < 1
    batches = {coefficient: [] for coefficient in (estimate.coefficient, 0.0, 1.0)}
    for k in range(16):
        parts = tail_parts(losses[100 * k : 100 * (k + 1)], np.full(100, 0.01), 0.9)
        for coefficient, values in batches.items():
            values.append(control_variate_tvar(parts, 0.9, 0.2, coefficient))
    critical = float(student_t.ppf(0.975, 15))
    widths = {
        coefficient: critical * np.std(values, ddof=1) / 4
        for coefficient, values in batches.items()
    }
    assert row["tvar"]["half_width"] == pytest.approx(
        widths[estimate.coefficient], rel=1e-12
    )
    assert widths[estimate.coefficient] < max(widths[0.0], widths[1.0])


def test_a_zero_var_tvar_is_the_exact_mean_over_one_minus_the_level():
    """At a level inside the zero atom the excess over VaR = 0 is the loss
    itself, so the regression coefficient is one and every batch's TVaR is
    EAL / (1 - level): exact, with a zero half-width."""
    rng = np.random.default_rng(71)
    losses = np.where(rng.random(1600) < 0.2, rng.exponential(1.0, 1600), 0.0)
    mixture = _mixture(_stratum(losses, mean=0.2))
    estimate = tail_estimates(mixture, "gross", (0.5,))[0.5]
    row = _gate(mixture, levels=(0.5,))["levels"]["0.5"]
    assert estimate.var == 0
    assert (estimate.coefficient, estimate.coefficient_basis) == (1.0, "regression")
    assert estimate.tvar == pytest.approx(0.2 / 0.5, rel=1e-12)
    assert row["tvar"]["half_width"] == pytest.approx(0.0, abs=1e-12)
    assert row["tvar"]["status"] == "converged"


def test_a_finite_variance_tvar_converges_where_the_exact_mean_estimate_would_not():
    """The golden example's law, L = 10 A + 100 B with A ~ Poisson(0.4) and
    B ~ Poisson(0.1), at 16,000 years: the control-variate TVaR at 0.98 has
    an expected half-width of about 4% of the value, while the exact-mean
    estimate's is about 18%, from the whole body's spread over 0.02. Gating
    only the reported estimator lets it converge at this budget."""
    rng = np.random.default_rng(23)
    losses = 10.0 * rng.poisson(0.4, 16000) + 100.0 * rng.poisson(0.1, 16000)
    mixture = _mixture(_stratum(losses, mean=14.0))
    level = 0.98
    estimate = tail_estimates(mixture, "gross", (level,))[level]
    row = _gate(mixture, levels=(level,))["levels"][str(level)]
    assert row["minimum_batch_exceedances"] >= MIN_TAIL_EFFECTIVE_UNITS
    assert row["tvar"]["relative_half_width"] < 0.06
    assert row["tvar"]["status"] == "converged"
    exact_mean_estimates = [
        control_variate_tvar(
            tail_parts(losses[1000 * k : 1000 * (k + 1)], np.full(1000, 1e-3), level),
            level,
            14.0,
            1.0,
        )
        for k in range(16)
    ]
    width = float(student_t.ppf(0.975, 15)) * np.std(exact_mean_estimates, ddof=1) / 4
    assert width / estimate.tvar > 0.1


def test_structural_zero_laws_keep_exact_intervals():
    zero = ConditionalCapabilityDraws(
        stratum_sha256="z",
        response_sha256="response",
        seed=0,
        views={"gross": ConditionalView(np.zeros(1), native_mean=ZERO_LAW)},
    )
    tail = tail_diagnostics(_mixture(zero), "gross", levels=(0.99,))["tails"]["0.99"]
    assert tail["var_interval_method"] == "known_zero_law"
    assert tail["var_numerical_interval_95"] == [0.0, 0.0]
    assert tail["tvar_numerical_interval_95"] == [0.0, 0.0]
    # A level inside the zero atom of a partly zero mixture needs no exceedances.
    positive = _stratum(np.random.default_rng(1).exponential(1.0, 1600), "p")
    row = _gate(_mixture(zero, positive, masses=[0.995, 0.005]), levels=(0.995,))[
        "levels"
    ]["0.995"]
    assert row["minimum_batch_exceedances"] is None
    assert row["var"]["status"] == "converged"


@pytest.mark.stochastic
def test_batch_means_var_intervals_cover_for_twelve_heavy_tailed_states():
    """Twelve equal-mass states of 1,024 years, a loss with probability 0.2 and
    GPD shape 0.4: the 90% and 99% VaR intervals cover near nominal."""
    shape, q, trials, levels = 0.4, 0.2, 200, (0.9, 0.99)
    scales = 1 + np.arange(12) / 4

    def survival(x):
        return q * (1 + shape * x / scales) ** (-1 / shape)

    truth = {
        level: brentq(
            lambda x, level=level: float(np.mean(survival(x))) - (1 - level), 0, 1e9
        )
        for level in levels
    }
    rng = np.random.default_rng(2024)
    covered = dict.fromkeys(levels, 0)
    for _ in range(trials):
        strata = []
        for index, scale in enumerate(scales):
            uniforms = rng.random(1024)
            losses = np.where(
                rng.random(1024) < q,
                scale * np.expm1(-shape * np.log1p(-uniforms)) / shape,
                0.0,
            )
            strata.append(_stratum(losses, f"s{index}"))
        tails = tail_diagnostics(_mixture(*strata), "gross", levels=levels)["tails"]
        for level in levels:
            low, high = tails[str(level)]["var_numerical_interval_95"]
            covered[level] += low <= truth[level] <= high
    for level in levels:
        assert 0.89 <= covered[level] / trials <= 0.99


def _package(tmp_path, *, simulations, gate, rate=50.0, name="gate", kernel=None):
    payload = study_payload()
    law = payload["joint_states"][0]["laws"][0]
    law["reference_annual_rate"] = rate
    if kernel is not None:
        payload["entities"].append(
            {
                "entity_id": "lab",
                "kind": "lab",
                "definition": "Synthetic lab",
                "display_name": "lab",
                "status": "hypothetical",
            }
        )
        law["segments"][0]["kernel"] = kernel.model_dump(mode="json")
    plan = ExecutionPlan(
        plan_id=name,
        required_arms=("reference",),
        simulations=simulations,
        seed=17,
        requested_views=("gross",),
        numerical_controls={"tail_gate": gate},
    )
    return build_setup(
        destination=tmp_path / name,
        setup=with_execution(setup_from_payload(payload), plan, setup_id=name),
        code_commit="fixture",
    )


def _run_and_prefix(tmp_path, gate, *, simulations=50, **package):
    """The doubled run, and the same plan run without doubling."""
    doubled = _package(tmp_path, simulations=simulations, gate=gate, **package)
    report = run_setup(
        setup_path=doubled.root, output=tmp_path / "run", code_commit="fixture"
    )
    short = _package(
        tmp_path,
        simulations=simulations,
        gate={**gate, "max_reference_state_years": None},
        name="short",
        **package,
    )
    run_setup(
        setup_path=short.root, output=tmp_path / "short-run", code_commit="fixture"
    )
    draws, _ = load_run_capture(tmp_path / "run" / "capture")
    first, _ = load_run_capture(tmp_path / "short-run" / "capture")
    (draw,) = draws["reference"].values()
    (prefix,) = first["reference"].values()
    return doubled, report, draw, prefix


def test_an_arm_doubles_its_years_until_its_drivers_converge(tmp_path):
    gate = {
        "batches": 4,
        "tolerance": 0.05,
        "max_reference_state_years": 1600,
        "levels": [0.9],
    }
    package, report, draw, prefix = _run_and_prefix(tmp_path, gate)
    record = report["tail_gate"]
    doublings = record["doublings"]
    assert doublings >= 1 and record["by_arm"]["reference"]["drivers_converged"]
    assert [row["doubling"] for row in record["rounds"]] == list(range(doublings + 1))
    assert all(
        row["unconverged_views"] == {"reference": ["gross"]}
        for row in record["rounds"][:-1]
    )
    assert record["rounds"][-1]["unconverged_views"] == {"reference": []}
    years = record["by_arm"]["reference"]["years_per_state"]
    assert years == draw.n_simulations == 50 * 2**doublings
    # Doubling keeps the earlier years.
    np.testing.assert_array_equal(
        prefix.annual_losses["gross"], draw.annual_losses["gross"][:50]
    )
    # Replay re-derives every round from the captured years' prefixes.
    replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )


def test_doubling_keeps_the_years_of_summed_small_events(tmp_path):
    gate = {"batches": 4, "max_reference_state_years": 128, "levels": [0.999]}
    _, report, draw, prefix = _run_and_prefix(
        tmp_path, gate, simulations=64, rate=4_000.0, kernel=_kernel(member=0.9)
    )
    assert report["tail_gate"]["doublings"] == 1
    assert report["small_event_aggregation"]["events"]["summed"] > 0
    for view in ("gross", "oep_gross"):
        if view in prefix.annual_losses:
            np.testing.assert_array_equal(
                prefix.annual_losses[view], draw.annual_losses[view][:64]
            )


@pytest.mark.parametrize(
    ("states", "reference", "comparison"),
    [(16, 65_536, 16_384), (120, 16_384, 4_096)],
)
def test_state_year_caps_set_each_arms_years_per_state(states, reference, comparison):
    """The import caps at 16 joint states and at 120."""
    controls = TailGateControls(
        max_reference_state_years=2_000_000, max_comparison_state_years=500_000
    )
    for is_reference, years, cap in (
        (True, reference, 2_000_000),
        (False, comparison, 500_000),
    ):
        doublings = controls.max_doublings(
            reference=is_reference, states=states, years=1024
        )
        assert 1024 * 2**doublings == years
        assert states * years <= cap < states * 2 * years


def test_a_cap_admits_exactly_its_state_years_and_refuses_fewer_than_planned():
    controls = TailGateControls(
        max_reference_state_years=3 * 50 * 4, max_comparison_state_years=3 * 50 * 4 - 1
    )
    assert controls.max_doublings(reference=True, states=3, years=50) == 2
    assert controls.max_doublings(reference=False, states=3, years=50) == 1
    assert TailGateControls().max_doublings(reference=True, states=3, years=50) == 0
    with pytest.raises(ValueError, match="exceed its tail gate cap"):
        TailGateControls(max_comparison_state_years=149).max_doublings(
            reference=False, states=3, years=50
        )


def _arms_package(
    tmp_path, gate, *, both_gate=None, rate=0.3, follower_rate=None, name="arms"
):
    """Two states of one law. The reference arm weights and simulates only the
    high state; ``both`` weights both, so it simulates the low state itself
    and shares the high state's simulation with the reference arm unless
    ``both_gate`` gives it its own gate; ``follower`` reweights the reference
    arm's state, unless ``follower_rate`` gives it its own law there."""
    payload = encode_scientific(_study(rate=rate))
    for state in payload["joint_states"]:
        state["weight"] = float(state["state_id"] == "high")
    alternatives: list[dict[str, Any]] = [
        {
            "alternative_id": "both",
            "purpose": "scientific_alternative",
            "description": "Both states.",
            "state_weights": {"low": 0.5, "high": 0.5},
        },
        {
            "alternative_id": "follower",
            "purpose": "scientific_alternative",
            "description": "The reference arm's state.",
            "state_weights": {"low": 0.0, "high": 1.0},
        },
    ]
    if follower_rate is not None:
        (high,) = [row for row in payload["joint_states"] if row["state_id"] == "high"]
        law = {**high["laws"][0], "reference_annual_rate": follower_rate}
        alternatives[1]["overrides"] = [{"state_id": "high", "law": law}]
    payload["alternatives"] = alternatives
    plan = ExecutionPlan(
        plan_id="arms",
        required_arms=("reference", "both", "follower"),
        simulations=50,
        seed=23,
        requested_views=("gross",),
        numerical_controls={"tail_gate": gate},
        numerical_controls_by_arm={}
        if both_gate is None
        else {"both": {"tail_gate": both_gate}},
        required_state_ids_by_arm={
            "reference": ("high",),
            "both": ("low", "high"),
            "follower": ("high",),
        },
    )
    return build_setup(
        destination=tmp_path / name,
        setup=with_execution(setup_from_payload(payload), plan, setup_id="arms"),
        code_commit="fixture",
    )


def test_each_arm_doubles_to_its_own_state_year_cap(tmp_path):
    # The reference arm's one state reaches 400 years within its 400
    # state-years and the comparison arm's two states 100 within its 200;
    # neither converges at 1-in-1,000. The follower simulates nothing and
    # takes the reference arm's years.
    gate = {
        "batches": 4,
        "max_reference_state_years": 400,
        "max_comparison_state_years": 200,
        "levels": [0.999],
    }
    package = _arms_package(tmp_path, gate)
    workload = setup_workload(package)["by_arm"]
    assert {arm: row["max_conditional_years"] for arm, row in workload.items()} == {
        "reference": 400,
        "both": 200,
        "follower": 400,
    }
    assert workload["follower"]["follows_arm"] == "reference"
    # Each state's work may reach its allowance per year times its arm's
    # largest years.
    assert {
        arm: row["max_work_per_state"]["vectorized_initiators"]
        for arm, row in workload.items()
    } == {
        "reference": 400 * _VECTORIZED,
        "both": 100 * _VECTORIZED,
        "follower": 400 * _VECTORIZED,
    }
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    record = report["tail_gate"]
    assert record["doublings"] == 3
    rows = record["by_arm"]
    assert [
        (
            rows[arm]["states"],
            rows[arm]["years_per_state"],
            rows[arm]["max_years_per_state"],
            rows[arm]["follows_arm"],
            rows[arm]["drivers_converged"],
        )
        for arm in ("reference", "both", "follower")
    ] == [
        (1, 400, 400, None, False),
        (2, 100, 100, None, False),
        (1, 400, 400, "reference", None),
    ]
    # Rounds evaluate the deciding arms until each stops.
    assert [sorted(row["unconverged_views"]) for row in record["rounds"]] == [
        ["both", "reference"],
        ["both", "reference"],
        ["reference"],
        ["reference"],
    ]
    draws, capture = load_run_capture(tmp_path / "run" / "capture")
    (high,) = draws["reference"]
    # The comparison arm keeps the first 100 years of the shared simulation the
    # reference arm extended, with their work limits; the follower holds the
    # reference arm's arrays.
    assert {
        arm: capture.arms[arm][high].execution["work_limits"]["vectorized_initiators"]
        for arm in ("reference", "both", "follower")
    } == {
        "reference": 400 * _VECTORIZED,
        "both": 100 * _VECTORIZED,
        "follower": 400 * _VECTORIZED,
    }
    np.testing.assert_array_equal(
        draws["both"][high].annual_losses["gross"],
        draws["reference"][high].annual_losses["gross"][:100],
    )
    assert draws["follower"][high] is draws["reference"][high]
    # Effects pair the arms' common years: the comparison arm's 100.
    comparisons = cast(dict, report["comparisons"])
    assert comparisons["both"]["gross"]["var::0.999"]["years_per_state"] == 100
    assert comparisons["follower"]["gross"]["var::0.999"]["years_per_state"] == 400
    replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )


def test_a_required_zero_weight_state_counts_toward_the_cap(tmp_path):
    # A hand-authored plan may require a zero-weight state; it costs its years,
    # so the two states' 400 state-years allow 200 years each.
    payload = encode_scientific(_study(rate=0.3))
    for state in payload["joint_states"]:
        state["weight"] = float(state["state_id"] == "high")
    payload["alternatives"] = []
    plan = ExecutionPlan(
        plan_id="zero",
        required_arms=("reference",),
        simulations=50,
        seed=23,
        requested_views=("gross",),
        numerical_controls={
            "tail_gate": {
                "batches": 4,
                "max_reference_state_years": 400,
                "levels": [0.999],
            }
        },
        required_state_ids_by_arm={"reference": ("low", "high")},
    )
    package = build_setup(
        destination=tmp_path / "zero",
        setup=with_execution(setup_from_payload(payload), plan, setup_id="zero"),
        code_commit="fixture",
    )
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    row = report["tail_gate"]["by_arm"]["reference"]
    assert (row["states"], row["years_per_state"]) == (2, 200)
    replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )


def test_an_arm_stops_on_its_own_converged_drivers(tmp_path):
    # The reference arm's 0.8 level converges in some round below its cap of
    # 3,200 years per state; the comparison arm's 1-in-1,000 level keeps
    # doubling to its own cap without the reference arm.
    caps = {"max_reference_state_years": 3200, "max_comparison_state_years": 6400}
    package = _arms_package(
        tmp_path,
        {"batches": 4, **caps, "levels": [0.8]},
        both_gate={"batches": 4, **caps, "levels": [0.999]},
    )
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    record = report["tail_gate"]
    rows = record["by_arm"]
    rounds = record["rounds"]
    stop = next(
        index
        for index, row in enumerate(rounds)
        if row["unconverged_views"].get("reference") == []
    )
    years = 50 * 2**stop
    assert years < 3200
    assert [sorted(row["unconverged_views"]) for row in rounds[stop:]] == [
        ["both", "reference"],
        *[["both"]] * (len(rounds) - stop - 1),
    ]
    assert rows["reference"]["drivers_converged"]
    assert [rows[arm]["years_per_state"] for arm in ("reference", "both")] == [
        years,
        3200,
    ]
    assert rows["follower"]["years_per_state"] == years
    comparisons = cast(dict, report["comparisons"])
    assert comparisons["both"]["gross"]["var::0.9"]["years_per_state"] == years
    replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )


def test_a_comparison_arm_doubles_while_its_control_fails(tmp_path):
    # The comparison arm's median converges early, but its control, the
    # reference arm, fails at 1-in-1,000 until its cap, so the comparison arm
    # keeps doubling with it and its effects are measured on 6,400 years.
    package = _arms_package(
        tmp_path,
        {
            "batches": 4,
            "max_reference_state_years": 6400,
            "max_comparison_state_years": 12800,
            "levels": [0.999],
        },
        both_gate={
            "batches": 4,
            "max_reference_state_years": 6400,
            "max_comparison_state_years": 12800,
            "levels": [0.5],
        },
    )
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    record = report["tail_gate"]
    rows = record["by_arm"]
    converged = [
        index
        for index, row in enumerate(record["rounds"])
        if row["unconverged_views"].get("both") == []
    ]
    assert converged and converged[0] < len(record["rounds"]) - 1
    assert rows["both"]["drivers_converged"]
    assert rows["both"]["years_per_state"] == rows["reference"]["years_per_state"]
    assert rows["reference"]["years_per_state"] == 6400
    comparisons = cast(dict, report["comparisons"])
    assert comparisons["both"]["gross"]["var::0.9"]["years_per_state"] == 6400
    replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )


def test_replay_refuses_a_follower_capture_unlike_the_reference_arms(tmp_path):
    gate = {
        "batches": 4,
        "max_reference_state_years": 100,
        "max_comparison_state_years": 100,
        "levels": [0.999],
    }
    package = _arms_package(tmp_path, gate)
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")
    capture = tmp_path / "run" / "capture"
    record = json.loads((capture / "run-capture.json").read_text())
    (high,) = record["arms"]["follower"]
    # Point the follower at the comparison arm's draw of the shared state.
    record["arms"]["follower"][high]["draw"] = record["arms"]["both"][high]["draw"]
    record["content_sha256"] = canonical_json_sha256(
        {key: value for key, value in record.items() if key != "content_sha256"}
    )
    (capture / "run-capture.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="following arm's capture differs"):
        load_run_capture(capture)


def test_a_comparison_arms_control_is_its_matched_control_or_the_reference():
    from types import SimpleNamespace

    from catmodel.results.execution import _gate_controls

    def arm(name, control=None):
        return SimpleNamespace(alternative_id=name, matched_control_id=control)

    package = SimpleNamespace(
        execution=SimpleNamespace(
            reference_arm="reference",
            required_arms=("reference", "alternative", "control", "other", "self"),
        ),
        alternatives=(
            arm("alternative", "control"),
            arm("control"),
            arm("other"),
            arm("self", "self"),
        ),
    )
    # A follower control stands for the reference arm, whose years it takes.
    assert _gate_controls(package, {"control"}) == {
        "reference": None,
        "alternative": "reference",
        "control": "reference",
        "other": "reference",
        "self": None,
    }
    assert _gate_controls(package, set())["alternative"] == "control"


def test_followers_take_the_reference_arms_limit_without_a_cap_of_their_own():
    plan = ExecutionPlan(
        plan_id="followers",
        required_arms=("reference", "follower"),
        simulations=100,
        seed=1,
        requested_views=("gross",),
        numerical_controls={
            "tail_gate": {
                "max_reference_state_years": 800,
                "max_comparison_state_years": 100,
            }
        },
    )
    states = {"reference": 2, "follower": 2}
    with pytest.raises(ValueError, match="exceed its tail gate cap"):
        doubling_limits(plan, states)
    assert doubling_limits(plan, states, ("follower",)) == {
        "reference": 2,
        "follower": 2,
    }


def test_an_effect_pairs_both_arms_on_their_common_years():
    """An arm identical to the reference over its fewer years has no effect,
    exactly, whichever arm ran longer."""
    values = np.random.default_rng(71).lognormal(0.0, 1.0, 1600)
    for base, other in ((values, values[:800]), (values[:800], values)):
        report = _reduced(
            {"base": {"s": _stratum(base)}, "other": {"s": _stratum(other)}}
        )
        effect = report["comparisons"]["other"]["gross"]["annual_probability_positive"]
        tail = report["comparisons"]["other"]["gross"]["var::0.9"]
        assert effect["years_per_state"] == tail["years_per_state"] == 800
        assert tail["absolute_difference"] == 0 and tail["half_width"] == 0
        assert tail["reference"] == pytest.approx(
            weighted_var_tvar(values[:800], np.full(800, 1 / 800), 0.9)[0]
        )


def test_doubling_limits_read_each_arms_planned_years():
    plan = ExecutionPlan(
        plan_id="limits",
        required_arms=("reference", "alternative"),
        simulations=100,
        seed=1,
        requested_views=("gross",),
        simulations_by_arm={"alternative": 25},
        numerical_controls={
            "tail_gate": {
                "max_reference_state_years": 800,
                "max_comparison_state_years": 800,
            }
        },
    )
    assert doubling_limits(plan, {"reference": 2, "alternative": 2}) == {
        "reference": 2,
        "alternative": 4,
    }


def test_a_year_prefix_cannot_exceed_the_simulated_years():
    draw = _stratum(np.arange(8.0))
    assert year_prefix(draw, 8) is draw
    np.testing.assert_array_equal(
        year_prefix(draw, 4).annual_losses["gross"], np.arange(4.0)
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        year_prefix(draw, 16)


@pytest.mark.parametrize("tamper", ["drop_round", "converged_early", "drivers"])
def test_replay_refuses_rounds_the_captured_years_do_not_reproduce(tmp_path, tamper):
    gate = {"batches": 4, "max_reference_state_years": 200, "levels": [0.999]}
    package = _package(tmp_path, simulations=50, gate=gate)
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")
    capture = tmp_path / "run" / "capture"
    record = json.loads((capture / "run-capture.json").read_text())
    rounds = record["tail_gate_rounds"]
    assert len(rounds) == 3
    if tamper == "drop_round":
        rounds.pop()
    elif tamper == "converged_early":
        rounds[0]["unconverged_views"] = {"reference": []}
    else:
        rounds[-1]["unconverged_views"] = {"reference": []}
    record["content_sha256"] = canonical_json_sha256(
        {key: value for key, value in record.items() if key != "content_sha256"}
    )
    (capture / "run-capture.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match=r"numerical controls|tail gate rounds"):
        replay_capture(
            setup_path=package.root,
            capture_directory=capture,
            output=tmp_path / "replay.json",
            code_commit="fixture",
        )


@pytest.mark.parametrize(
    "tamper", ["count", "count_and_limit", "first_round_limit", "missing_count"]
)
def test_replay_rederives_each_states_work_limits(tmp_path, tamper):
    """Replay re-derives a state's work limits, the plan's allowances per year
    times its captured years, and checks each count within its limit: it
    refuses a count above its limit, alone or beside a recorded limit raised to
    admit it, a recorded limit of the first round's years and a missing count."""
    gate = {"batches": 4, "max_reference_state_years": 200, "levels": [0.999]}
    package = _package(tmp_path, simulations=50, gate=gate)
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")
    capture = tmp_path / "run" / "capture"
    record = json.loads((capture / "run-capture.json").read_text())
    (state,) = record["arms"]["reference"].values()
    limits = state["execution"]["work_limits"]
    assert limits["vectorized_initiators"] == 200 * _VECTORIZED
    if tamper == "count":
        state["execution"]["vectorized_initiators"] = 200 * _VECTORIZED + 1
    elif tamper == "count_and_limit":
        state["execution"]["vectorized_initiators"] = limits[
            "vectorized_initiators"
        ] = 200 * _VECTORIZED + 1
    elif tamper == "first_round_limit":
        limits["vectorized_initiators"] = 50 * _VECTORIZED
    else:
        del state["execution"]["campaign_members"]
    record["content_sha256"] = canonical_json_sha256(
        {key: value for key, value in record.items() if key != "content_sha256"}
    )
    (capture / "run-capture.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="work allowances per year times the captured"):
        replay_capture(
            setup_path=package.root,
            capture_directory=capture,
            output=tmp_path / "replay.json",
            code_commit="fixture",
        )


def test_a_runaway_law_is_refused_at_the_first_round_before_drawing(
    tmp_path, monkeypatch
):
    """A law of a billion events a year exceeds the default vectorized
    allowance at once: the run stops in its first round, before any event is
    drawn, and leaves no capture."""
    gate = {"batches": 4, "max_reference_state_years": 3200, "levels": [0.999]}
    package = _package(tmp_path, simulations=50, gate=gate, rate=1e9)
    years, drawing = [], []
    simulate = execution.simulate_incident_state
    vectorized = incident_runtime._vectorized_incidents

    def rounds(*args, **kwargs):
        years.append(kwargs["years"])
        return simulate(*args, **kwargs)

    def tracked(*args, **kwargs):
        drawing.append(True)
        return vectorized(*args, **kwargs)

    monkeypatch.setattr(execution, "simulate_incident_state", rounds)
    monkeypatch.setattr(incident_runtime, "_vectorized_incidents", tracked)
    with pytest.raises(ValueError) as refusal:
        run_setup(
            setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
        )
    # The refusal names the count, arm, years, limit and allowance, and how to
    # raise it.
    message = str(refusal.value)
    assert message.startswith("individually drawn events in state ")
    for part in (
        "of arm 'reference': ",
        " over 50 simulated years, above the limit of 2,500,000, "
        "max_vectorized_events_per_year (50,000 a year) times the years; "
        "they average ",
        "raise max_vectorized_events_per_year in the plan's numerical controls",
    ):
        assert part in message
    assert (years, drawing) == ([50], [])
    assert not (tmp_path / "run").exists()


def test_the_cap_stops_doubling_and_unconverged_tails_are_withheld(tmp_path):
    gate = {"batches": 4, "max_reference_state_years": 200, "levels": [0.999]}
    package = _package(tmp_path, simulations=50, gate=gate)
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    record = report["tail_gate"]
    assert record["doublings"] == 2
    assert record["by_arm"]["reference"]["years_per_state"] == 200
    assert not record["by_arm"]["reference"]["drivers_converged"]
    assert "gross::var::0.999" in record["by_arm"]["reference"]["withheld_statistics"]
    statistics = available_metric(
        reduced_arm(report["arms"]["reference"])["metrics"]["gross"]
    )["statistics"]
    assert statistics["var::0.999"]["status"] == "withheld"
    assert str(statistics["var::0.999"]["reason"]).startswith(
        "Unconverged at 200 years per state"
    )


def test_a_tight_half_width_with_sparse_batches_does_not_converge():
    """The same distinct losses in every batch: every batch's VaR is equal, so
    the half-width is zero, but only two years of each lie at or above it."""
    values = np.tile(np.arange(1.0, 101.0), 16)
    row = _gate(_mixture(_stratum(values)), levels=(0.99,))["levels"]["0.99"]
    assert row["var"]["half_width"] == 0
    assert row["minimum_batch_exceedances"] == 2
    assert row["var"]["status"] == "unconverged"


def _capped_mixture(years=3200, seed=29):
    """An annual payout capped at a limit of 50, as a limited policy pays:
    a loss in 30% of years, exponential with mean 100 before the cap, so
    P(payout = 50) = 0.3 exp(-0.5) and the law has no support above 50."""
    rng = np.random.default_rng(seed)
    loss = 100.0 * rng.exponential(size=years)
    return np.where(rng.random(years) < 0.3, np.minimum(loss, 50.0), 0.0)


def test_a_var_at_the_top_of_a_capped_law_converges_without_years_above_it():
    """At levels above the cap's atom, VaR = TVaR = 50 exactly: no year lies
    above the VaR, but the years at it fill the tail, so the gate converges."""
    values = _capped_mixture()
    atom = 0.3 * math.exp(-0.5)
    levels = (0.95, 0.99)
    assert all(1 - level < atom for level in levels)
    mixture = _mixture(_stratum(values))
    tails = tail_diagnostics(mixture, "gross", levels=levels)
    metric = {
        "statistics": {
            f"{functional}::{level}": {"status": "available", "reason": None}
            for level in levels
            for functional in ("var", "tvar")
        },
        "tail_diagnostics": tails,
    }
    withhold_unsupported_tails(metric)
    batch = values.reshape(16, -1)
    for level in levels:
        tail = tails["tails"][str(level)]
        assert (tail["var"], tail["tvar"]) == (50.0, pytest.approx(50.0, abs=1e-12))
        # The descriptive strict count stays zero; the gate counts the atom.
        assert tail["strict_exceedance_count"] == 0
        row = tail["convergence"]
        assert row["minimum_batch_exceedances"] == int(
            np.min(np.sum(batch >= 50.0, axis=1))
        )
        assert row["minimum_batch_exceedances"] >= MIN_TAIL_EFFECTIVE_UNITS
        assert row["var"]["half_width"] == 0
        assert row["tvar"]["half_width"] == pytest.approx(0.0, abs=1e-9)
        for functional in ("var", "tvar"):
            assert row[functional]["status"] == "converged"
            assert metric["statistics"][f"{functional}::{level}"]["status"] == (
                "available"
            )


def test_years_at_an_interior_atom_var_count_as_tail_years():
    """A lattice law whose 0.9 VaR is an interior atom with mass on both sides
    of the level: the gate counts the years at the VaR, which the TVaR's
    fractional atom averages, as well as those above it."""
    rng = np.random.default_rng(31)
    values = 10.0 * rng.poisson(1.0, 3200)
    level = 0.9
    var = weighted_var_tvar(values, np.full(3200, 1 / 3200), level)[0]
    assert np.mean(values < var) < level < np.mean(values <= var)
    row = _gate(_mixture(_stratum(values)), levels=(level,))["levels"][str(level)]
    batch = values.reshape(16, -1)
    at_or_above = int(np.min(np.sum(batch >= var, axis=1)))
    assert row["minimum_batch_exceedances"] == at_or_above
    assert at_or_above > int(np.min(np.sum(batch > var, axis=1)))


def test_the_tvar_interval_never_falls_below_the_var():
    values = np.random.default_rng(1).pareto(1.2, 1600) + 1
    tail = tail_diagnostics(_mixture(_stratum(values)), "gross", levels=(0.99,))[
        "tails"
    ]["0.99"]
    low, high = tail["tvar_numerical_interval_95"]
    half = tail["convergence"]["tvar"]["half_width"]
    assert tail["tvar"] - half < tail["var"]
    assert low == tail["var"] and high == pytest.approx(tail["tvar"] + half)


def test_the_tolerance_decides_convergence_when_every_batch_has_support():
    law = _mixture(_stratum(np.random.default_rng(21).lognormal(0.0, 2.0, 1600)))
    tight = _gate(law, levels=(0.5,))["levels"]["0.5"]
    loose = _gate(law, levels=(0.5,), tolerance=0.5)["levels"]["0.5"]
    assert tight["minimum_batch_exceedances"] >= MIN_TAIL_EFFECTIVE_UNITS
    assert 0.05 < tight["var"]["relative_half_width"] < 0.5
    assert tight["var"]["status"] == "unconverged"
    assert loose["var"]["status"] == "converged"


def test_an_eal_above_every_simulated_year_leaves_every_tail_unconverged():
    """An exact mean the years cannot reach: they miss the upper ranks or
    severities that carry it, so no tail statistic has converged."""
    values = np.random.default_rng(13).exponential(1.0, 1600)
    mixture = _mixture(_stratum(values, mean=10 * float(values.max())))
    row = _gate(mixture)["levels"]["0.9"]
    for key in ("var", "tvar"):
        assert row[key]["status"] == "unconverged"
        assert "analytic EAL exceeds" in row[key]["reason"]


def test_a_certified_eal_above_the_years_within_its_bound_is_coherent():
    """A loss of exactly 7 in every year: its mean, largest year, VaR and
    TVaR are all 7. An EAL of 7.000007 exceeds them, which no law's mean can,
    so as a closed-form mean it leaves every tail unconverged. Certified only
    to 0.00007 it may be the exact mean, 7: the years are coherent with it,
    in the headline record as in the gate, and its tails converge."""
    values = np.full(1600, 7.0)
    for bound, coherent in ((0.0, False), (7e-5, True)):
        mixture = _mixture(_stratum(values, mean=7.000007, bound=bound))
        record = cast(dict, mixture.metrics("gross", levels=(0.9,)))
        assert record["eal_tail_coherence"]["status"] == (
            "coherent" if coherent else "incoherent"
        )
        row = _gate(mixture)["levels"]["0.9"]
        for key in ("var", "tvar"):
            assert row[key]["half_width"] == 0
            if coherent:
                assert row[key]["status"] == "converged" and "reason" not in row[key]
            else:
                assert row[key]["status"] == "unconverged"
                assert "analytic EAL exceeds" in row[key]["reason"]


def _withheld(tails, levels):
    """The statuses ``withhold_unsupported_tails`` gives every VaR and TVaR."""
    metric = {
        "statistics": {
            f"{functional}::{level}": {"status": "available", "reason": None}
            for level in levels
            for functional in ("var", "tvar")
        },
        "tail_diagnostics": tails,
    }
    withhold_unsupported_tails(metric)
    return metric["statistics"]


def test_years_that_disagree_with_the_analytic_eal_withhold_every_tail():
    """The golden example's law, L = 10 A + 100 B with A ~ Poisson(0.4) and
    B ~ Poisson(0.1), mean 14 and variance 1,040, at 16,000 years. z is the
    years' mean less the EAL over the standard deviation of the 16 batch
    means over 4, about t-distributed with 15 degrees of freedom at the exact
    EAL, so within ±5 but with probability 1.6e-4. An analytic EAL of 21, as
    an implementation that disagrees with the simulation could give, lies
    7 / sqrt(1,040 / 16,000) = 27 standard errors above the years' mean, so z
    is below -5 unless the batches overstate the standard error more than
    fourfold. Every VaR and TVaR is then withheld with z, even the TVaR at
    0.5, inside the zero atom, which is exactly EAL / 0.5 in every batch and
    so converges on its own half-width."""
    rng = np.random.default_rng(23)
    losses = 10.0 * rng.poisson(0.4, 16000) + 100.0 * rng.poisson(0.1, 16000)
    error = float(np.std(losses.reshape(16, -1).mean(axis=1), ddof=1)) / 4
    levels = (0.5, 0.9)
    for eal, status in ((14.0, "consistent"), (21.0, "inconsistent")):
        tails = tail_diagnostics(
            _mixture(_stratum(losses, mean=eal)), "gross", levels=levels
        )
        record = tails["convergence"]["eal_reconciliation"]
        assert (record["statistic"], record["limit"]) == ("mean", None)
        assert record["expected_annual_loss"] == eal
        assert record["simulated_mean_standard_error"] == pytest.approx(
            error, rel=1e-12
        )
        assert record["z"] == pytest.approx((losses.mean() - eal) / error, rel=1e-9)
        assert record["status"] == status
        assert tails["tails"]["0.5"]["convergence"]["tvar"][
            "half_width"
        ] == pytest.approx(0.0, abs=1e-12)
        statistics = _withheld(tails, levels)
        if status == "consistent":
            for key in ("var::0.5", "tvar::0.5", "tvar::0.9"):
                assert statistics[key]["status"] == "available"
            continue
        assert record["z"] < -5
        for key, value in statistics.items():
            assert value["status"] == "withheld", key
            assert str(value["reason"]).startswith(
                "Unconverged: the simulated years' mean differs from the analytic "
                f"EAL by z = {record['z']:+.1f} batch-means standard errors, beyond ±5"
            )


def test_a_limited_mean_above_the_analytic_eal_withholds_an_infinite_variance_tail():
    """GPD shape 0.65 and scale 1: mean 1 / 0.35 = 2.857 and no finite
    variance, so the years' mean has no standard error. The mean of the years
    limited at their VaR at the highest level, 0.99 (about 29.2), estimates
    E[min(X, 29.2)] = 2.857 (1 - 19.95^(1 - 1 / 0.65)) = 2.287, whose standard
    error at 16,000 years is 4.28 / sqrt(16,000) = 0.034, and no law's
    limited mean exceeds its mean. It lies about 17 standard errors below the
    exact EAL, which passes however far below; an EAL understated by half,
    1.43, lies about 25 standard errors below it, which withholds every VaR
    and TVaR."""
    shape, years, levels = 0.65, 16000, (0.9, 0.99)
    uniforms = np.random.default_rng(37).random(years)
    losses = np.expm1(-shape * np.log1p(-uniforms)) / shape
    # The left-inverse VaR at 0.99: the 15,840th smallest year.
    limit = float(np.sort(losses)[int(np.ceil(0.99 * years)) - 1])
    limited = np.minimum(losses, limit)
    error = float(np.std(limited.reshape(16, -1).mean(axis=1), ddof=1)) / 4
    exact = 1 / (1 - shape)
    for eal, status in ((exact, "consistent"), (exact / 2, "inconsistent")):
        tails = tail_diagnostics(
            _mixture(_stratum(losses, mean=eal, variance=False)),
            "gross",
            levels=levels,
        )
        record = tails["convergence"]["eal_reconciliation"]
        assert (record["statistic"], record["limit_level"]) == ("limited_mean", "0.99")
        assert record["limit"] == limit
        assert record["simulated_mean"] == pytest.approx(limited.mean(), rel=1e-12)
        assert record["simulated_mean_standard_error"] == pytest.approx(
            error, rel=1e-12
        )
        assert record["z"] == pytest.approx((limited.mean() - eal) / error, rel=1e-9)
        assert record["status"] == status
        statistics = _withheld(tails, levels)
        if status == "consistent":
            assert record["z"] < -5
            assert statistics["tvar::0.9"]["status"] == "available"
            continue
        assert record["z"] > 5
        for key, value in statistics.items():
            assert value["status"] == "withheld", key
            assert (
                "limited at their VaR at level 0.99 exceeds the analytic EAL by z = +"
                in str(value["reason"])
            )


def test_a_mismatched_analytic_eal_withholds_tails_through_doubling_and_replay(
    tmp_path, monkeypatch
):
    """The fixture's annual loss is 10 times a Poisson(50) count: mean 500 and
    variance 5,000. At 6,400 years its 0.9 VaR and TVaR converge in the first
    round. An analytic path that disagrees with the simulation, here every
    analytic EAL raised by a fifth, puts the EAL 100 / sqrt(5,000 / 6,400) =
    113 standard errors above the years' mean: the driver fails in every
    round, the arm doubles to its cap and its VaR and TVaR are withheld with
    z, and replay, which rederives the analytic EAL and every round from the
    capture, reproduces them."""
    gate = {"levels": [0.9], "max_reference_state_years": 12800}
    control = _package(tmp_path, simulations=6400, gate=gate, name="control")
    report = run_setup(
        setup_path=control.root, output=tmp_path / "control-run", code_commit="fixture"
    )
    assert report["tail_gate"]["doublings"] == 0
    metric = available_metric(
        reduced_arm(report["arms"]["reference"])["metrics"]["gross"]
    )
    # The reconciliation is reported beside the statistics it can withhold.
    assert present(metric["eal_reconciliation"])["status"] == "consistent"
    assert metric["statistics"]["tvar::0.9"]["status"] == "available"
    analytic = incident_runtime.analytic_linear_eals

    def raised(*args, **kwargs):
        means = analytic(*args, **kwargs)
        return replace(
            means, values={view: 1.2 * value for view, value in means.values.items()}
        )

    monkeypatch.setattr(incident_runtime, "analytic_linear_eals", raised)
    monkeypatch.setattr(execution, "analytic_linear_eals", raised)
    package = _package(tmp_path, simulations=6400, gate=gate)
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    record = report["tail_gate"]
    assert record["doublings"] == 1
    assert all(
        row["unconverged_views"] == {"reference": ["gross"]} for row in record["rounds"]
    )
    assert not record["by_arm"]["reference"]["drivers_converged"]
    assert {"gross::var::0.9", "gross::tvar::0.9"} <= set(
        record["by_arm"]["reference"]["withheld_statistics"]
    )
    metric = available_metric(
        reduced_arm(report["arms"]["reference"])["metrics"]["gross"]
    )
    reconciliation = present(metric["eal_reconciliation"])
    assert reconciliation["expected_annual_loss"] == pytest.approx(600, rel=1e-12)
    assert reconciliation["status"] == "inconsistent"
    assert present(reconciliation["z"]) < -5
    for key in ("var::0.9", "tvar::0.9"):
        status = metric["statistics"][key]
        assert status["status"] == "withheld"
        assert str(status["reason"]).startswith(
            "Unconverged: the simulated years' mean differs from the analytic EAL by z = -"
        )
    replayed = replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    assert replayed["tail_gate"] == record
    again = available_metric(
        reduced_arm(replayed["arms"]["reference"])["metrics"]["gross"]
    )
    assert again["statistics"] == metric["statistics"]
    assert again["eal_reconciliation"] == reconciliation
    # results.json carries the record, so the export's reproduction check
    # covers its z and standard error.
    saved = load_results(tmp_path / "replay.json")
    saved_metric = available_metric(
        reduced_arm(saved["arms"]["reference"])["metrics"]["gross"]
    )
    assert saved_metric["eal_reconciliation"] == reconciliation


def _gpd(shape, years, seed):
    """Unit-scale GPD losses, one a year: survival (1 + shape x)^(-1 / shape)
    and mean 1 / (1 - shape), with no finite variance from shape 0.5."""
    uniforms = np.random.default_rng(seed).random(years)
    return np.expm1(-shape * np.log1p(-uniforms)) / shape


def _numpy_tvar(losses, level, mean, coefficient):
    """The control-variate TVaR of equally weighted years from NumPy means:
    ``v + ((1 - b) mean((L - v)+) + b (mean - mean(min(L, v)))) / (1 - level)``
    at the years' VaR ``v``, and at least ``v``."""
    var = weighted_var_tvar(losses, np.full(len(losses), 1 / len(losses)), level)[0]
    excess = float(np.mean(np.maximum(losses - var, 0.0)))
    limited = float(np.mean(np.minimum(losses, var)))
    correction = (1 - coefficient) * excess + coefficient * (mean - limited)
    return var + max(correction, 0.0) / (1 - level)


@pytest.mark.stochastic
@pytest.mark.parametrize(
    ("shape", "basis"), [(0.3, "regression"), (0.65, "infinite_variance")]
)
def test_a_certified_mean_controls_the_tvar_and_its_bound_widens_the_half_width(
    shape, basis
):
    """A unit GPD loss every year for 16,000 years, at level 0.9: the VaR is
    v = (0.1^-shape - 1) / shape and the excess over it is GPD with scale
    s = 1 + shape v, so the TVaR is v + s / (1 - shape): 6.168 at shape 0.3
    and 18.096 at shape 0.65. The mean 1 / (1 - shape) is given as a
    certified integral, 0.05% above the truth with a bound of 0.1% of it.

    That mean is the control: the coefficient is the regression's with a
    finite variance and one without, never the zero of a view with no mean.
    The TVaR is the control-variate formula at the certified mean, so it
    differs from the estimate at the true mean by at most the margin,
    coefficient x bound / 0.1. Its half-width is the 16 batches' Student-t
    half-width plus that margin, the interval is the estimate plus or minus
    it, and the standard error leaves the margin out.

    The estimate lies within five standard errors, and the margin, of the
    closed-form TVaR. At coefficient one the error is that of the years'
    mean of min(X, v) over 0.1, and E[min(X, v)^2] is the integral of
    2 x S(x) to v: sd 1.686, so 1.686 / (sqrt(16,000) 0.1) = 0.133, 0.7% of
    the TVaR. A regression coefficient's error is at most the sample tail
    mean's, sd((X - v)+) / (sqrt(16,000) 0.1) = 0.131, 2.1% of the TVaR.
    """
    level, years = 0.9, 16000
    losses = _gpd(shape, years, seed=round(100 * shape) + 3)
    exact = 1 / (1 - shape)
    mean, bound = exact * (1 + 5e-4), exact * 1e-3
    mixture = _mixture(_stratum(losses, mean=mean, bound=bound, variance=shape < 0.5))
    tails = tail_diagnostics(mixture, "gross", levels=(level,))
    tail = tails["tails"][str(level)]
    estimator = tail["tvar_estimator"]
    assert estimator["coefficient_basis"] == basis
    coefficient = 1.0
    if basis == "regression":
        var = weighted_var_tvar(losses, np.full(years, 1 / years), level)[0]
        excess = np.maximum(losses - var, 0.0)
        coefficient = float(np.cov(excess, losses)[0, 1] / np.var(losses, ddof=1))
        assert 0 < coefficient < 1
    assert estimator["coefficient"] == pytest.approx(coefficient, rel=1e-9)
    tvar = _numpy_tvar(losses, level, mean, coefficient)
    assert tail["tvar"] == pytest.approx(tvar, rel=1e-9)
    margin = coefficient * bound / (1 - level)
    assert 0 < abs(tvar - _numpy_tvar(losses, level, exact, coefficient)) <= margin
    batches = [
        _numpy_tvar(part, level, mean, coefficient) for part in losses.reshape(16, -1)
    ]
    critical = float(student_t.ppf(0.975, 15))
    sampling = critical * float(np.std(batches, ddof=1)) / 4
    row = tail["convergence"]["tvar"]
    assert row["half_width"] == pytest.approx(sampling + margin, rel=1e-9)
    assert row["status"] == "converged"
    assert tail["tvar_numerical_interval_95"] == pytest.approx(
        [tvar - sampling - margin, tvar + sampling + margin], rel=1e-9
    )
    assert tail["tvar_numerical_standard_error"] == pytest.approx(
        sampling / critical, rel=1e-6
    )
    assert _withheld(tails, (level,))[f"tvar::{level}"]["status"] == "available"
    # The closed-form TVaR and the estimator's standard error under the law.
    truth_var = ((1 - level) ** -shape - 1) / shape
    scale = 1 + shape * truth_var
    truth = truth_var + scale / (1 - shape)
    if basis == "regression":
        first = (1 - level) * scale / (1 - shape)
        second = (1 - level) * 2 * scale**2 / ((1 - shape) * (1 - 2 * shape))
    else:
        first = (1 - scale ** (1 - 1 / shape)) / (1 - shape)
        a, b = 2 - 1 / shape, 1 - 1 / shape
        second = 2 / shape**2 * ((scale**a - 1) / a - (scale**b - 1) / b)
    error = math.sqrt(second - first**2) / (math.sqrt(years) * (1 - level))
    assert abs(tail["tvar"] - truth) <= 5 * error + margin


@pytest.mark.stochastic
def test_a_tvar_its_certified_bound_alone_withholds_never_drives_doubling():
    """Unit GPD losses of shape 0.65 at 16,000 years, level 0.9, with the
    exact mean 1 / 0.35 certified only to 0.2 (7%). At coefficient one the
    margin is 0.2 / 0.1 = 2.0, 11% of the TVaR of about 18.1 and above the
    10% tolerance at any number of years, so the TVaR is withheld for it.
    More years cannot converge it, so it does not make its view a failing
    driver: the VaR of 5.33 converges, its standard error being
    sqrt(0.09 / 16,000) over the density 0.0224 there, 0.106 or 2%, and the
    arm would stop. A VaR that more years could converge, at 0.999 with
    about one year beyond it in each batch, still drives. With the mean
    certified to 0.1% the same years converge the TVaR."""
    level = 0.9
    losses = _gpd(0.65, 16000, seed=68)
    exact = 1 / 0.35
    drivers = partial(
        unconverged_drivers,
        weights={"s": 1.0},
        controls=TailGateControls(),
        requested=("gross",),
    )
    for bound, converged in ((0.2, False), (exact * 1e-3, True)):
        draw = _stratum(losses, mean=exact, bound=bound, variance=False)
        tails = tail_diagnostics(_mixture(draw), "gross", levels=(level,))
        tail = tails["tails"][str(level)]
        row = tail["convergence"]
        margin = bound / (1 - level)
        assert (margin > 0.1 * tail["tvar"]) is not converged
        assert row["var"]["status"] == "converged"
        statistics = _withheld(tails, (level,))
        assert statistics[f"var::{level}"]["status"] == "available"
        assert drivers({"s": draw}, levels=(level,)) == []
        if converged:
            assert row["tvar"]["status"] == "converged"
            assert "reason" not in row["tvar"]
            assert statistics[f"tvar::{level}"]["status"] == "available"
            continue
        assert row["tvar"]["status"] == "unconverged"
        assert row["tvar"]["reason"] == INTEGRATION_MARGIN_REASON
        assert row["tvar"]["half_width"] > margin
        assert statistics[f"tvar::{level}"] == {
            "status": "withheld",
            "reason": INTEGRATION_MARGIN_REASON,
        }
        assert drivers({"s": draw}, levels=(level, 0.999)) == ["gross"]


@pytest.mark.stochastic
def test_a_tvar_with_no_half_width_at_any_years_never_drives_doubling():
    """Unit GPD losses of shape 0.65 with no analytic mean: the TVaR is the
    sample tail mean of a law without a second moment, which has no
    half-width at any number of years. It is withheld, but doubling could
    never converge it, so its view fails only on a statistic more years
    could converge: not on its converged 0.9 VaR, and on its 0.999 VaR, with
    about one year beyond it in each batch."""
    draw = _stratum(_gpd(0.65, 16000, seed=68), variance=False)
    tails = tail_diagnostics(_mixture(draw), "gross", levels=(0.9,))
    row = tails["tails"]["0.9"]["convergence"]
    assert row["tvar"] == {
        "half_width": None,
        "relative_half_width": None,
        "status": "unconverged",
        "reason": SAMPLE_TVAR_MOMENT_REASON,
    }
    assert row["var"]["status"] == "converged"
    assert _withheld(tails, (0.9,))["tvar::0.9"] == {
        "status": "withheld",
        "reason": SAMPLE_TVAR_MOMENT_REASON,
    }
    drivers = partial(
        unconverged_drivers, {"s": draw}, {"s": 1.0}, TailGateControls(), ("gross",)
    )
    assert drivers((0.9,)) == []
    assert drivers((0.9, 0.999)) == ["gross"]


def _governed_setup(**options):
    """The fraud-on-individuals node under one synthetic respondent. Its
    governed severity tail is uncapped with a GPD shape of at least 0.5, so
    welfare and gross have no finite second moment; welfare's mean is
    closed-form, and gross adds the lab's annual business loss, whose mean
    carries the certified error of P(N >= 1) integrated over the count's
    parameter rank."""
    instrument = governed_contextual_instrument(("fraud_individuals",))
    (node,) = instrument.nodes
    artifact = fixture_artifact(
        instrument, lambda _prompt: governed_contextual_answer(node)
    )
    return compile_contextual_setup(
        instrument, artifact, setup_id="governed", **options
    )


@pytest.mark.stochastic
def test_a_certified_infinite_variance_gross_reports_its_tvar_like_welfare():
    """The governed state at 65,536 years, at the 1-in-10 and 1-in-100
    levels. Gross is welfare plus the lab's loss in every year, so their EALs
    differ by the lab's, and the lab's loss has a finite variance: its
    simulated mean, a separate implementation from the certified integral,
    lies within five standard errors, and the bound, of that difference.

    Gross's TVaR is then controlled by its certified mean like welfare's by
    its closed-form one: coefficient one for want of a second moment, the
    estimate VaR + (EAL - mean(min(L, VaR))) / (1 - level) on the years, and
    a half-width of the 16 batches' Student-t half-width plus
    bound / (1 - level). It is reported wherever welfare's is, with an
    interval; a year-by-year larger loss also has the larger VaR. The lab's
    view, with a finite variance, takes the regression coefficient."""
    setup = _governed_setup(years_per_state=1024, seed=42)
    state = max(setup.joint_states, key=lambda row: row.weight)
    years, levels = 65536, (0.9, 0.99)
    draw = simulate_incident_state(
        setup.execution,
        compile_incident_state(setup, state.state_id),
        response_sha256="0" * 64,
        years=years,
    ).draws
    views = ("gross", "welfare", "lab_loss::lab")
    report = reduce_captures(
        scientific_sha256="0" * 64,
        plan_sha256="0" * 64,
        draws_by_arm={"reference": {draw.stratum_sha256: draw}},
        state_weights={draw.stratum_sha256: 1.0},
        required_arms=("reference",),
        reference_arm="reference",
        views=views,
        reducer_provenance={"code": "fixture"},
        spec=ReductionSpec(levels=levels),
    )
    metrics = {
        view: available_metric(
            reduced_arm(report["arms"]["reference"])["metrics"][view]
        )
        for view in views
    }
    gross, welfare, lab = (draw.annual_losses[view] for view in views)
    np.testing.assert_allclose(gross, welfare + lab, rtol=1e-12)
    assert not draw.views["gross"].variance_exists
    assert draw.views["lab_loss::lab"].variance_exists
    eal = present(metrics["gross"]["expected_annual_loss"])
    bound = metrics["gross"]["eal_precision"]["integration_error_bound"]
    assert 0 < bound <= 1e-3 * eal
    assert metrics["welfare"]["eal_precision"]["integration_error_bound"] == 0
    error = float(np.std(lab, ddof=1)) / math.sqrt(years)
    difference = eal - present(metrics["welfare"]["expected_annual_loss"])
    assert abs(float(lab.mean()) - difference) <= 5 * error + bound
    critical = float(student_t.ppf(0.975, 15))
    for level in levels:
        key = str(level)
        tail = metrics["gross"]["tail_metrics"][key]
        reference = metrics["welfare"]["tail_metrics"][key]
        for view in ("gross", "welfare"):
            assert metrics[view]["tail_metrics"][key]["tvar_estimator"] == {
                "method": "control_variate",
                "coefficient": 1.0,
                "coefficient_basis": "infinite_variance",
            }
            status = metrics[view]["statistics"][f"tvar::{key}"]
            assert status == {"status": "available", "reason": None}
        tvar = _numpy_tvar(gross, level, eal, 1.0)
        assert tail["tvar"] == pytest.approx(tvar, rel=1e-9)
        batches = [_numpy_tvar(part, level, eal, 1.0) for part in gross.reshape(16, -1)]
        half = critical * float(np.std(batches, ddof=1)) / 4 + bound / (1 - level)
        assert tail["tvar_numerical_interval_95"] == pytest.approx(
            [tvar - half, tvar + half], rel=1e-9
        )
        assert present(tail["var"]) >= present(reference["var"])
        estimator = present(
            metrics["lab_loss::lab"]["tail_metrics"][key]["tvar_estimator"]
        )
        assert estimator["coefficient_basis"] == "regression"
        assert 0 < estimator["coefficient"] <= 1


@pytest.mark.stochastic
def test_an_arm_stops_doubling_once_its_certified_drivers_converge(tmp_path):
    """The governed study with welfare and gross as the drivers at level
    0.9, from 1,024 years per state under a cap of 262,144 (its four anchor
    states' 1,048,576 state-years). Gross has a certified mean and no finite
    second moment: as a sample tail mean its TVaR would have no half-width
    at any years, and the arm could only run to its cap. Controlled by the
    certified mean it converges as welfare's does, so the arm stops below
    its cap with every driver converged and the gross TVaR reported.

    The first round cannot converge: each batch has 64 years, of which 6.4
    are expected at or above a 0.9 VaR and ten are needed. Replay re-derives
    every round from the captured years."""
    setup = _governed_setup(
        years_per_state=1024, seed=42, max_reference_state_years=4 * 262144
    )
    plan = setup.execution
    gate = {
        **plan.numerical_controls["tail_gate"],
        "views": ["gross", "welfare"],
        "levels": [0.9],
    }
    controls = {**plan.numerical_controls, "tail_gate": gate}
    package = build_setup(
        destination=tmp_path / "setup",
        setup=with_execution(
            setup,
            plan.model_copy(update={"numerical_controls": controls}),
            setup_id="governed",
        ),
        code_commit="fixture",
    )
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    record = report["tail_gate"]
    arm = record["by_arm"]["reference"]
    assert arm["driver_views"] == ["gross", "welfare"]
    assert record["doublings"] >= 1
    assert set(record["rounds"][0]["unconverged_views"]["reference"]) == {
        "gross",
        "welfare",
    }
    assert record["rounds"][-1]["unconverged_views"] == {"reference": []}
    assert arm["drivers_converged"]
    assert arm["years_per_state"] == 1024 * 2 ** record["doublings"]
    assert arm["years_per_state"] < arm["max_years_per_state"] == 262144
    assert "gross::tvar::0.9" not in arm["withheld_statistics"]
    metric = available_metric(
        reduced_arm(report["arms"]["reference"])["metrics"]["gross"]
    )
    assert metric["eal_precision"]["integration_error_bound"] > 0
    assert present(metric["tail_metrics"]["0.9"]["tvar_estimator"]) == {
        "method": "control_variate",
        "coefficient": 1.0,
        "coefficient_basis": "infinite_variance",
    }
    assert metric["statistics"]["tvar::0.9"]["status"] == "available"
    replayed = replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    assert replayed["tail_gate"] == record


def test_stratum_masses_that_round_above_one_leave_no_negative_batch_mass():
    rng = np.random.default_rng(17)
    mixture = _mixture(
        _stratum(rng.exponential(1.0, 320), "a"),
        _stratum(rng.exponential(1.0, 320), "b"),
        masses=[0.5 + 1e-11, 0.5],
    )
    assert _gate(mixture)["unavailable_reason"] is None


def test_paired_effects_cancel_shared_noise():
    rng = np.random.default_rng(31)
    base = rng.exponential(1.0, 16)
    shifted = paired_effect(base.tolist(), (base + 0.1).tolist(), 1.0, 0.05)
    independent = paired_effect(
        base.tolist(), (rng.exponential(1.0, 16) + 0.1).tolist(), 1.0, 0.05
    )
    assert shifted is not None and independent is not None
    assert shifted["half_width"] == pytest.approx(0, abs=1e-12)
    assert shifted["status"] == "converged"
    # Independent batches do not cancel.
    assert independent["half_width"] > 0.1
    assert independent["status"] == "unconverged"


def _reduced(arms, **kwargs):
    from catmodel.results.reduction import reduce_captures

    return reduce_captures(
        scientific_sha256="study",
        plan_sha256="plan",
        draws_by_arm=arms,
        state_weights={"s": 1.0},
        required_arms=tuple(arms),
        reference_arm="base",
        views=("gross",),
        reducer_provenance={"commit": "fixture"},
        **kwargs,
    )


def test_arm_effects_pair_the_same_batches_and_withhold_noisy_ones():
    values = np.random.default_rng(37).lognormal(0.0, 1.0, 1600)
    noise = np.random.default_rng(41).lognormal(0.0, 1.0, 1600)
    report = _reduced(
        {
            "base": {"s": _stratum(values)},
            "scaled": {"s": _stratum(1.5 * values)},
            "resampled": {"s": _stratum(noise)},
        }
    )
    scaled = report["comparisons"]["scaled"]["gross"]["var::0.9"]
    resampled = report["comparisons"]["resampled"]["gross"]["annual_loss_median"]
    # Scaling every year: each batch's effect is half its base value, so the
    # paired half-width is half the base's.
    convergence = view_diagnostics(report, "base", "gross")["tail_diagnostics"][
        "tails"
    ]["0.9"]["convergence"]
    base = present(present(convergence)["var"]["half_width"])
    assert scaled["years_per_state"] == 1600
    assert scaled["half_width"] == pytest.approx(0.5 * base, rel=1e-9)
    # Independent years do not cancel: the median effect is withheld.
    assert resampled["status"] == "withheld"
    assert "batch-means half-width" in resampled["reason"]


def test_sparse_positive_years_withhold_the_probability_of_a_loss():
    values = np.zeros(1600)
    values[::200] = 5.0
    tails = tail_diagnostics(_mixture(_stratum(values)), "gross", levels=(0.5,))
    metric = {
        "statistics": {
            "annual_probability_positive": {"status": "available", "reason": None},
            "var::0.5": {"status": "available", "reason": None},
            "tvar::0.5": {"status": "available", "reason": None},
        },
        "tail_diagnostics": tails,
    }
    withhold_unsupported_tails(metric)
    status = metric["statistics"]["annual_probability_positive"]
    assert status["status"] == "withheld"
    assert "years with a loss" in str(status["reason"])


def test_dependence_bracket_tails_carry_batch_half_widths():
    from catmodel.results.dependence_bracket import dependence_bracket

    rng = np.random.default_rng(43)
    draw = ConditionalCapabilityDraws(
        stratum_sha256="s",
        response_sha256="response",
        seed=0,
        views={
            "a": ConditionalView(rng.exponential(1.0, 6400)),
            "b": ConditionalView(rng.exponential(1.0, 6400)),
        },
    )
    bracket = dependence_bracket(
        CapabilityMixture({"s": draw}, {"s": 1.0}),
        scenario_views={"a": "a", "b": "b"},
        levels=(0.5, 0.999),
    )
    for reference in cast(dict, bracket["references"]).values():
        median, extreme = reference["tail"]["0.5"], reference["tail"]["0.999"]
        assert median["half_width"]["var"] > 0
        assert median["status"] == "available"
        # At the 10% tolerance the 1-in-1,000 half-width can pass while some
        # batch has no year above the VaR; either withholds it.
        assert extreme["half_width"]["var"] > 0
        assert extreme["status"] == "withheld"
        assert extreme["reason"].startswith("Unconverged")


def test_the_tvar_coefficient_treats_states_sharing_years_as_one():
    values = np.random.default_rng(47).exponential(1.0, 1600)
    single = _mixture(_stratum(values, "a", mean=1.0))
    shared = _mixture(_stratum(values, "a", mean=1.0), _stratum(values, "b", mean=1.0))
    (one,) = tail_estimates(single, "gross", (0.9,)).values()
    (two,) = tail_estimates(shared, "gross", (0.9,)).values()
    assert one.coefficient_basis == two.coefficient_basis == "regression"
    assert two.coefficient == pytest.approx(one.coefficient, rel=1e-9)
    assert two.tvar == pytest.approx(one.tvar, rel=1e-12)


def _tvar_effect(base, alternative, level=0.9):
    from catmodel.results.reduction import ReductionSpec

    report = _reduced(
        {"base": {"s": base}, "alternative": {"s": alternative}},
        spec=ReductionSpec(levels=(level,)),
    )
    tails = {
        arm: report["arms"][arm]["metrics"]["gross"]["tail_metrics"][str(level)]
        for arm in ("base", "alternative")
    }
    return report["comparisons"]["alternative"]["gross"][f"tvar::{level}"], tails


def test_a_tvar_effect_applies_the_controls_coefficient_to_both_arms():
    """Two finite-variance arms on the same years with different regression
    coefficients: the effect re-estimates the alternative's TVaR with the
    control's coefficient, so the effect and every batch of it difference one
    estimator, and the shared control term cancels."""
    rng = np.random.default_rng(59)
    base = np.where(rng.random(1600) < 0.3, rng.exponential(1.0, 1600), 0.0)
    alternative = base**1.5
    base_mean, alternative_mean = 0.3, 0.3 * math.gamma(2.5)
    effect, tails = _tvar_effect(
        _stratum(base, mean=base_mean), _stratum(alternative, mean=alternative_mean)
    )
    coefficient = tails["base"]["tvar_estimator"]["coefficient"]
    own = tails["alternative"]["tvar_estimator"]["coefficient"]
    assert abs(coefficient - own) > 0.01
    assert effect["control_variate_coefficient"] == coefficient
    assert effect["reference"] == tails["base"]["tvar"]
    masses = np.full(1600, 1 / 1600)
    assert effect["alternative"] == pytest.approx(
        control_variate_tvar(
            tail_parts(alternative, masses, 0.9), 0.9, alternative_mean, coefficient
        ),
        rel=1e-12,
    )
    differences = []
    for k in range(16):
        part, batch = slice(100 * k, 100 * (k + 1)), np.full(100, 0.01)
        differences.append(
            control_variate_tvar(
                tail_parts(alternative[part], batch, 0.9),
                0.9,
                alternative_mean,
                coefficient,
            )
            - control_variate_tvar(
                tail_parts(base[part], batch, 0.9), 0.9, base_mean, coefficient
            )
        )
    assert effect["half_width"] == pytest.approx(
        float(student_t.ppf(0.975, 15)) * np.std(differences, ddof=1) / 4, rel=1e-9
    )


def test_an_infinite_variance_arm_sets_the_effects_coefficient_to_one():
    """A capped control (finite variance) against the same years uncapped
    (no second moment): only one suits the uncapped arm, and the capped arm
    admits it, so both arms use the exact-mean estimate."""
    rng = np.random.default_rng(61)
    shape, cap = 0.6, 50.0
    uncapped = np.expm1(-shape * np.log1p(-rng.random(1600))) / shape
    capped_mean = (1 - (1 + shape * cap) ** (1 - 1 / shape)) / (1 - shape)
    effect, tails = _tvar_effect(
        _stratum(np.minimum(uncapped, cap), mean=capped_mean),
        _stratum(uncapped, mean=1 / (1 - shape), variance=False),
    )
    assert tails["base"]["tvar_estimator"]["coefficient_basis"] == "regression"
    assert tails["alternative"]["tvar_estimator"]["coefficient"] == 1.0
    assert effect["control_variate_coefficient"] == 1.0
    assert effect["alternative"] == tails["alternative"]["tvar"]
    masses = np.full(1600, 1 / 1600)
    assert effect["reference"] == pytest.approx(
        control_variate_tvar(
            tail_parts(np.minimum(uncapped, cap), masses, 0.9), 0.9, capped_mean, 1.0
        ),
        rel=1e-12,
    )


def test_no_effect_coefficient_suits_arms_fixed_at_zero_and_one():
    """A control without an exact mean (coefficient zero) against an arm
    without a second moment (coefficient one): each keeps its own."""
    rng = np.random.default_rng(67)
    uncapped = np.expm1(-0.6 * np.log1p(-rng.random(1600))) / 0.6
    effect, tails = _tvar_effect(
        _stratum(np.minimum(uncapped, 50.0)),
        _stratum(uncapped, mean=2.5, variance=False),
    )
    assert effect["control_variate_coefficient"] is None
    assert effect["reference"] == tails["base"]["tvar"]
    assert effect["alternative"] == tails["alternative"]["tvar"]


def test_a_tvar_effect_adds_both_arms_integration_margins():
    """Unit GPD losses of shape 0.65 against the same years scaled by 1.5:
    neither arm has a second moment, so both and the effect take coefficient
    one. Scaling scales every batch's TVaR, so the effect is half the
    control's TVaR in every batch and its batch-means half-width half the
    control's. Each mean is certified to 0.1%, e for the control and 1.5 e
    for the scaled arm; their errors need not cancel, so the effect's
    half-width adds both margins, (e + 1.5 e) / 0.1."""
    level = 0.9
    losses = _gpd(0.65, 16000, seed=68)
    mean = 1 / 0.35
    bound = 1e-3 * mean
    effect, tails = _tvar_effect(
        _stratum(losses, mean=mean, bound=bound, variance=False),
        _stratum(1.5 * losses, mean=1.5 * mean, bound=1.5 * bound, variance=False),
        level,
    )
    assert effect["control_variate_coefficient"] == 1.0
    assert effect["reference"] == tails["base"]["tvar"]
    assert effect["absolute_difference"] == pytest.approx(
        0.5 * _numpy_tvar(losses, level, mean, 1.0), rel=1e-9
    )
    batches = [_numpy_tvar(part, level, mean, 1.0) for part in losses.reshape(16, -1)]
    sampling = float(student_t.ppf(0.975, 15)) * float(np.std(batches, ddof=1)) / 4
    assert effect["half_width"] == pytest.approx(
        0.5 * sampling + 2.5 * bound / (1 - level), rel=1e-9
    )
    assert "status" not in effect


@pytest.mark.stochastic
def test_an_effect_on_an_analytic_eal_has_no_simulation_error_to_gate():
    """Two states at even weights, and an arm that only reweights them to
    0.1 and 0.9. The first state's loss is exponential with mean 1; the
    second's is 100 with probability 0.02, mean 2 and standard deviation 14.

    With analytic state means, closed-form or certified integrals, both EALs
    are those means under the weights, 1.5 and 1.9, and the effect is
    0.4 x 2 - 0.4 x 1 = 0.4 with no sampling error: it has no half-width and
    is not withheld. Where the EAL is the years' mean, the effect is
    0.4 (mean of the second - mean of the first) and its half-width comes
    from the 16 batches' differences: about 2.13 x 0.4 x 1.4 / 4 = 0.30, a
    fifth of the reference EAL, so it is withheld."""
    rng = np.random.default_rng(83)
    first = rng.exponential(1.0, 1600)
    second = 100.0 * (rng.random(1600) < 0.02)

    def effect_of(a, b):
        draws = {"a": a, "b": b}
        weights = {"base": {"a": 0.5, "b": 0.5}, "reweighted": {"a": 0.1, "b": 0.9}}
        report = reduce_captures(
            scientific_sha256="study",
            plan_sha256="plan",
            draws_by_arm=dict.fromkeys(weights, draws),
            state_weights=weights["base"],
            state_weights_by_arm=weights,
            required_arms=tuple(weights),
            reference_arm="base",
            views=("gross",),
            reducer_provenance={"commit": "fixture"},
        )
        return cast(dict, report["comparisons"])["reweighted"]["gross"][
            "expected_annual_loss"
        ]

    for bound in (0.0, 1e-3):
        effect = effect_of(
            _stratum(first, "a", mean=1.0, bound=bound),
            _stratum(second, "b", mean=2.0, bound=2 * bound),
        )
        assert effect["reference"] == pytest.approx(1.5, rel=1e-12)
        assert effect["alternative"] == pytest.approx(1.9, rel=1e-12)
        assert effect["absolute_difference"] == pytest.approx(0.4, rel=1e-12)
        assert "half_width" not in effect and "status" not in effect
    effect = effect_of(_stratum(first, "a"), _stratum(second, "b"))
    assert effect["absolute_difference"] == pytest.approx(
        0.4 * (second.mean() - first.mean()), rel=1e-9
    )
    differences = 0.4 * (
        second.reshape(16, -1).mean(axis=1) - first.reshape(16, -1).mean(axis=1)
    )
    width = float(student_t.ppf(0.975, 15)) * float(np.std(differences, ddof=1)) / 4
    assert effect["half_width"] == pytest.approx(width, rel=1e-9)
    assert width > 0.1 * effect["reference"]
    assert effect["status"] == "withheld"
    assert "batch-means half-width" in effect["reason"]


def test_a_withheld_arm_statistic_withholds_its_effect():
    values = np.random.default_rng(53).exponential(1.0, 1600)
    report = _reduced(
        {"base": {"s": _stratum(values)}, "same": {"s": _stratum(values)}}
    )
    effect = report["comparisons"]["same"]["gross"]["var::0.999"]
    # Identical arms: the paired half-width is zero, yet each arm's 1-in-1,000
    # VaR is withheld, so the effect is too.
    assert effect["half_width"] == 0
    assert effect["status"] == "withheld" and "sparsest" in effect["reason"]


def _rounds_capture(rounds, *, simulations=800, level=0.9):
    from types import SimpleNamespace

    plan = ExecutionPlan(
        plan_id="rounds",
        required_arms=("reference",),
        simulations=simulations,
        seed=1,
        requested_views=("gross",),
        numerical_controls={
            "tail_gate": {
                "batches": 4,
                "max_reference_state_years": 1600,
                "levels": [level],
            }
        },
    )
    return SimpleNamespace(
        plan=plan,
        arms={"reference": {"s": SimpleNamespace(conditional_key="law")}},
        state_weights_by_arm={"reference": {"s": 1.0}},
        tail_gate_rounds=rounds,
    )


def test_replay_refuses_a_doubling_after_converged_drivers_and_an_early_stop():
    from catmodel.results.execution import _validate_gate_rounds

    tight = {
        "reference": {"s": _stratum(np.random.default_rng(59).normal(100, 1, 1600))}
    }
    converged = {"reference": []}
    with pytest.raises(ValueError, match="doubled an arm whose drivers had converged"):
        _validate_gate_rounds(
            _rounds_capture(
                [
                    {"doubling": 0, "unconverged_views": converged},
                    {"doubling": 1, "unconverged_views": converged},
                ]
            ),
            tight,
            {"reference": None},
        )
    sparse = {
        "reference": {"s": _stratum(np.random.default_rng(61).exponential(1.0, 800))}
    }
    with pytest.raises(ValueError, match="cap allowed another doubling"):
        _validate_gate_rounds(
            _rounds_capture(
                [{"doubling": 0, "unconverged_views": {"reference": ["gross"]}}],
                level=0.999,
            ),
            sparse,
            {"reference": None},
        )


def test_comonotone_bracket_batches_sort_each_scenario_within_the_batch():
    """b = 10 - a year by year, so the executed sum is constant while the
    comonotone sum pairs each batch's sorted years."""
    from catmodel.results.dependence_bracket import dependence_bracket

    a = np.random.default_rng(67).uniform(0.0, 10.0, 1600)
    b = 10.0 - a
    draw = ConditionalCapabilityDraws(
        stratum_sha256="s",
        response_sha256="response",
        seed=0,
        views={"a": ConditionalView(a), "b": ConditionalView(b)},
    )
    bracket = dependence_bracket(
        CapabilityMixture({"s": draw}, {"s": 1.0}),
        scenario_views={"a": "a", "b": "b"},
        levels=(0.5,),
    )
    medians = []
    for k in range(16):
        part = slice(100 * k, 100 * (k + 1))
        total = np.sort(a[part]) + np.sort(b[part])
        medians.append(weighted_var_tvar(total, np.full(100, 0.01), 0.5)[0])
    expected = float(student_t.ppf(0.975, 15)) * np.std(medians, ddof=1) / 4
    references = cast(dict, bracket["references"])
    assert references["executed"]["tail"]["0.5"]["half_width"]["var"] == 0
    assert references["comonotone"]["tail"]["0.5"]["half_width"][
        "var"
    ] == pytest.approx(expected, rel=1e-9)
    assert expected > 0


def test_the_probability_of_a_loss_needs_ten_positive_years_in_every_batch():
    values = np.zeros(1600)
    for k in range(16):
        values[100 * k : 100 * k + 5] = 3.0
    tails = tail_diagnostics(_mixture(_stratum(values)), "gross", levels=(0.5,))
    row = tails["convergence"]["probability_positive"]
    # Every batch has the same probability, so only the floor fails.
    assert row["half_width"] == 0
    assert row["minimum_batch_exceedances"] == 5
    assert row["status"] == "unconverged"


def test_an_unconverged_joint_exceedance_is_withheld():
    from catmodel.results.reduction import ReductionSpec

    values = np.zeros(1600)
    values[::400] = 9.0
    draw = ConditionalCapabilityDraws(
        stratum_sha256="s",
        response_sha256="response",
        seed=0,
        views={"gross": ConditionalView(values)},
        joint_law_semantics="physical_joint",
        evaluator_provenance_by_risk_unit={
            "unit": {"joint_law_semantics": "physical_joint"}
        },
    )
    report = _reduced(
        {"base": {"s": draw}},
        spec=ReductionSpec(joint_thresholds=(("gross", 1.0),)),
        capture_provenance={"physical_joint_semantics": "physical_joint_samples"},
    )
    arm = report["arms"]["base"]
    assert arm["joint_exceedance_probability"] == pytest.approx(4 / 1600)
    assert arm["joint_exceedance_status"]["status"] == "withheld"


def test_the_tornado_does_not_rank_a_withheld_effect():
    from catmodel.results.sensitivity_summary import sensitivity_summary

    def contrast(difference, withheld=False):
        effect = {
            "reference": 10.0,
            "alternative": 10.0 + difference,
            "absolute_difference": difference,
            "relative_change": difference / 10,
        }
        if withheld:
            effect.update(status="withheld", reason="Unconverged: noisy effect.")
        return {"gross": {"expected_annual_loss": effect}}

    report = {
        "arms": {
            "reference": {
                "metrics": {
                    "gross": {"status": "available", "expected_annual_loss": 10.0}
                }
            },
            "steady": {},
            "noisy": {},
        },
        "comparisons": {
            "steady": contrast(4.0),
            "noisy": contrast(40.0, withheld=True),
        },
        "coverage": {"comparisons": {"status": "complete"}},
    }
    tornado = sensitivity_summary(report, reference_arm="reference", purposes={})[
        "tornado"
    ]
    assert [row["arm"] for row in tornado["rows"]] == ["steady"]
    assert "Unconverged" in tornado["unavailable_arms"]["noisy"]


def test_rankings_never_order_a_withheld_effect():
    values = np.random.default_rng(53).exponential(1.0, 1600)
    report = _reduced(
        {"base": {"s": _stratum(values)}, "scaled": {"s": _stratum(1.2 * values)}}
    )
    rankings = cast(dict, report["rankings"])["gross"]
    assert rankings["annual_loss_median"]["status"] == "available"
    assert rankings["var::0.999"]["status"] == "withheld"
    assert "tail gate withheld" in rankings["var::0.999"]["reason"]
    assert report["rankings_available"]


def test_bracket_levels_need_ten_years_at_or_above_the_var_in_every_batch():
    from catmodel.results.dependence_bracket import dependence_bracket

    def executed_tail(pattern):
        values = np.tile(pattern, 16)
        draw = ConditionalCapabilityDraws(
            stratum_sha256="s",
            response_sha256="response",
            seed=0,
            views={"a": ConditionalView(values), "b": ConditionalView(values)},
        )
        bracket = dependence_bracket(
            CapabilityMixture({"s": draw}, {"s": 1.0}),
            scenario_views={"a": "a", "b": "b"},
            levels=(0.98,),
        )
        return cast(dict, bracket["references"])["executed"]["tail"]["0.98"]

    # Every batch is the same, so only the floor can fail: distinct losses
    # leave three years at or above the 0.98 VaR in each batch.
    sparse = executed_tail(np.arange(1.0, 101.0))
    assert sparse["half_width"]["var"] == 0
    assert sparse["minimum_batch_exceedances"] == 3
    assert sparse["status"] == "withheld" and "sparsest" in sparse["reason"]
    # An atom of 11 years at the largest loss: VaR and TVaR are that loss, no
    # year lies above it, and the years at it support the level.
    pattern = np.ones(100)
    pattern[:11] = 2.0
    atom = executed_tail(pattern)
    assert atom["var"] == atom["tvar"] == 4.0
    assert atom["minimum_batch_exceedances"] == 11
    assert atom["status"] == "available" and atom["reason"] is None
