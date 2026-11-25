"""Counts drawn by inversion: exact for their law, comonotone across laws."""

from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy import stats

from catmodel.contracts.incident import ExecutionPlan
from catmodel.probability.count_inversion import (
    binomial_counts,
    negative_binomial_counts,
    poisson_counts,
)
from catmodel.probability.keyed_streams import event_uniforms
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_runtime import simulate_incident_state
from test_support.study import stipulated_study

# Both ends of a 53-bit uniform, both sides of one half and the body.
EDGES = (0.0, 2.0**-53, 1e-12, 0.25, 0.5, 0.5 + 2.0**-53, 0.75, 1 - 1e-12, 1 - 2.0**-53)


def _uniforms(count=30, seed=1):
    return np.array([*EDGES, *np.random.default_rng(seed).random(count)])


def _assert_least(
    counts,
    uniforms,
    first: Callable[[], Decimal],
    ratio: Callable[[int], Decimal],
):
    """Each count is the least whose CDF reaches its uniform, by a 60-digit CDF
    built from the pmf at zero and the ratio of consecutive pmf terms."""
    with localcontext() as context:
        context.prec = 60
        start = first()
        for count, uniform in zip(counts.tolist(), uniforms.tolist(), strict=True):
            term = cdf = start
            previous = Decimal(0)
            for k in range(1, count + 1):
                previous = cdf
                term *= ratio(k)
                cdf += term
            target = Decimal(uniform)
            assert cdf >= target, (count, uniform)
            assert count == 0 or previous < target, (count, uniform)


@pytest.mark.parametrize("rate", [1e-9, 0.3, 2.5, 40.0, 700.0])
def test_poisson_counts_are_the_least_whose_cdf_reaches_the_uniform(rate):
    uniforms = _uniforms()
    counts = poisson_counts(uniforms, np.full(len(uniforms), rate))
    lam = Decimal(rate)
    _assert_least(counts, uniforms, lambda: (-lam).exp(), lambda k: lam / k)


@pytest.mark.parametrize("mean,ratio", [(0.4, 2.0), (6.0, 3.0), (90.0, 1.5)])
def test_negative_binomial_counts_are_the_least_whose_cdf_reaches_the_uniform(
    mean, ratio
):
    uniforms = _uniforms()
    counts = negative_binomial_counts(uniforms, np.full(len(uniforms), mean), ratio)
    # Failures before mean / (v - 1) successes of probability 1 / v.
    successes = Decimal(mean) / (Decimal(ratio) - 1)
    success = 1 / Decimal(ratio)
    _assert_least(
        counts,
        uniforms,
        lambda: (successes * success.ln()).exp(),
        lambda k: (k - 1 + successes) / k * (1 - success),
    )


@pytest.mark.parametrize(
    "trials,probability", [(1, 0.3), (40, 0.02), (500, 0.6), (3000, 0.005)]
)
def test_binomial_counts_are_the_least_whose_cdf_reaches_the_uniform(
    trials, probability
):
    uniforms = _uniforms()
    counts = binomial_counts(
        uniforms, np.full(len(uniforms), trials), np.full(len(uniforms), probability)
    )
    p = Decimal(probability)
    _assert_least(
        counts,
        uniforms,
        lambda: (1 - p) ** trials,
        lambda k: Decimal(trials - k + 1) / k * p / (1 - p),
    )


def test_degenerate_laws_and_the_zero_uniform():
    uniforms = _uniforms()
    size = len(uniforms)
    assert not poisson_counts(uniforms, np.zeros(size)).any()
    assert not negative_binomial_counts(uniforms, np.zeros(size), 2.0).any()
    assert not binomial_counts(uniforms, np.zeros(size, dtype=int), 0.5).any()
    assert not binomial_counts(uniforms, np.full(size, 7), np.zeros(size)).any()
    np.testing.assert_array_equal(
        binomial_counts(uniforms[1:], np.full(size - 1, 20), np.ones(size - 1)), 20
    )
    # A subnormal rate or probability draws nothing, without overflow.
    tiny = np.full(size, 5e-324)
    assert not poisson_counts(uniforms, tiny).any()
    assert not negative_binomial_counts(uniforms, tiny, 2.0).any()
    assert not binomial_counts(uniforms, np.full(size, 10), tiny).any()
    # A zero uniform gives the least count however high the rate.
    for rate in (1e3, 1e9, 1e12):
        assert poisson_counts(np.zeros(1), np.full(1, rate))[0] == 0


@pytest.mark.parametrize("rate", [1e4, 1e9, 1e12])
def test_huge_rates_keep_their_mean_and_variance(rate):
    years = np.arange(100_000, dtype=np.int64)
    uniforms = event_uniforms(
        years,
        np.zeros_like(years),
        root_seed=3,
        state_id="s",
        origin_id="o",
        purpose="p",
    )
    counts = poisson_counts(uniforms, np.full(len(years), rate))
    assert abs(counts.mean() - rate) < 5 * np.sqrt(rate / len(years))
    assert counts.var() / rate == pytest.approx(1.0, abs=0.03)


def test_invalid_inputs_are_refused():
    with pytest.raises(ValueError, match="uniforms"):
        poisson_counts(np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="rates"):
        poisson_counts(np.array([0.5]), np.array([np.inf]))
    with pytest.raises(ValueError, match="ratio must exceed one"):
        negative_binomial_counts(np.array([0.5]), np.array([1.0]), 1.0)
    with pytest.raises(ValueError, match="trials"):
        binomial_counts(np.array([0.5]), np.array([2.5]), np.array([0.5]))
    with pytest.raises(ValueError, match="at most one"):
        binomial_counts(np.array([0.5]), np.array([2]), np.array([1.5]))
    with pytest.raises(ValueError, match="representable"):
        poisson_counts(np.array([0.5]), np.array([1e17]))


def _addressed(years: int):
    index = np.arange(years, dtype=np.int64)
    return event_uniforms(
        index,
        np.zeros_like(index),
        root_seed=11,
        state_id="state",
        origin_id="origin",
        purpose="origin-counts",
    )


@pytest.mark.stochastic
def test_counts_of_addressed_uniforms_follow_their_exact_law():
    uniforms = _addressed(200_000)
    size = len(uniforms)
    for counts, law in (
        (poisson_counts(uniforms, np.full(size, 3.7)), stats.poisson(3.7)),
        (
            negative_binomial_counts(uniforms, np.full(size, 5.0), 2.5),
            stats.nbinom(5.0 / 1.5, 1 / 2.5),
        ),
        (binomial_counts(uniforms, np.full(size, 40), 0.1), stats.binom(40, 0.1)),
    ):
        # Pearson's test on every count with at least 20 expected years, the
        # rest pooled into the upper tail.
        support = np.arange(counts.max() + 1)
        expected = size * law.pmf(support)
        cells = int(np.flatnonzero(expected >= 20)[-1]) + 1
        observed = np.bincount(counts, minlength=len(support))
        observed = np.append(observed[:cells], observed[cells:].sum())
        expected = np.append(expected[:cells], size * law.sf(cells - 1))
        _, p_value = stats.chisquare(observed, expected)
        assert p_value > 1e-4


def test_counts_read_at_one_uniform_are_comonotone_across_laws():
    uniforms = np.sort(_addressed(50_000))
    size = len(uniforms)
    reference = poisson_counts(uniforms, np.full(size, 40.0))
    scaled = poisson_counts(uniforms, np.full(size, 44.0))
    overdispersed = negative_binomial_counts(uniforms, np.full(size, 40.0), 3.0)
    # Every law's counts rise with the uniform, so any two are comonotone, and
    # a higher rate never lowers a count.
    for counts in (reference, scaled, overdispersed):
        assert np.all(np.diff(counts) >= 0)
    assert np.all(scaled >= reference)
    # Comonotone counts at 40 and 44 differ by 4 plus at most a count or so;
    # independent ones would differ with variance 84, twice a count's.
    assert np.var(scaled - reference) < 0.02 * np.var(reference)


def test_an_arm_with_more_rate_keeps_every_simulated_year_paired():
    """Each incident of the stipulated study loses exactly 10, so a year's
    gross loss is ten times its count: an arm with a tenth more rate loses at
    least as much as its control in every year, and the paired difference
    keeps under 2% of the control's variance, against about 210% for
    independent counts."""
    compiled = compile_incident_state(stipulated_study(), "state")
    (incident,) = compiled.incidents
    plan = ExecutionPlan(
        plan_id="paired-counts",
        required_arms=("reference",),
        simulations=20_000,
        seed=5,
        requested_views=("gross",),
    )

    def gross(rate):
        state = replace(compiled, incidents=(replace(incident, annual_rate=rate),))
        draw, _, _ = simulate_incident_state(plan, state, response_sha256="a" * 64)
        return draw.views["gross"].array

    control, arm = gross(40.0), gross(44.0)
    assert np.all(arm >= control)
    assert np.var(arm - control) < 0.02 * np.var(control)
