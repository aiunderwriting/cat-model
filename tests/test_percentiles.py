"""Independent CDF and tail-law checks for study reporting."""

import math
from fractions import Fraction
from itertools import permutations, product
from typing import cast

import numpy as np
import pytest

from catmodel.probability.distributions import (
    Distribution,
    MixtureSeverityDistribution,
    uncapped_infinite_variance_gpd_shapes,
)
from catmodel.probability.numerics import TIE_RELATIVE_TOLERANCE
from catmodel.probability.percentiles import SortedSample, _running_sums
from catmodel.simulation.capability_mixture import (
    CapabilityMixture,
    ConditionalCapabilityDraws,
    ConditionalView,
    tail_parts,
    weighted_var_tvar,
)


def weighted_percentile(values, weights, q):
    return SortedSample(values, weights).percentile(q)


def exact_steps(values, weights) -> list[tuple[float, Fraction]]:
    """The weighted empirical distribution function in rational arithmetic:
    each value of positive weight, in rising order, with the share of the
    weight at or below it. A float weight is read as its exact binary value."""
    weighted = sorted(
        (float(value), Fraction(weight))
        for value, weight in zip(values, weights, strict=True)
        if weight
    )
    total = sum(weight for _, weight in weighted)
    steps, below = [], Fraction(0)
    for value, weight in weighted:
        below += weight
        steps.append((value, below / total))
    return steps


def exact_left_inverse(steps: list[tuple[float, Fraction]], level: Fraction) -> float:
    """The least value of ``exact_steps`` whose share reaches ``level``."""
    return next(value for value, share in steps if share >= level)


@pytest.mark.parametrize("representation", ["empirical", "state_masses"])
def test_capability_metrics_match_hand_calculated_loss_table(
    representation: str,
) -> None:
    # A 1,000-year table: 990 zeros, four 10s, three 100s, two 1,000s,
    # and one 2,000. Split and scramble the 100 atom to exercise tied masses.
    losses = np.array([1_000.0, 0.0, 100.0, 10.0, 2_000.0, 100.0])
    frequencies = np.array([2, 990, 1, 4, 1, 2])
    if representation == "state_masses":
        # One single-year state per atom, at the atom's probability.
        draws = {
            f"path-{index}": ConditionalCapabilityDraws(
                f"path-{index}",
                "response",
                1,
                {"gross": ConditionalView(np.array([loss]))},
            )
            for index, loss in enumerate(losses)
        }
        masses = {
            f"path-{index}": float(frequency) / 1000
            for index, frequency in enumerate(frequencies)
        }
    else:
        draws = {
            "path": ConditionalCapabilityDraws(
                "path",
                "response",
                1,
                {"gross": ConditionalView(np.repeat(losses, frequencies))},
            )
        }
        masses = {"path": 1.0}
    result = CapabilityMixture(draws, masses).metrics(
        "gross", levels=(0.99, 0.995, 0.996, 0.999)
    )
    assert result["expected_annual_loss"] == pytest.approx(4.34)
    assert result["annual_probability_positive"] == pytest.approx(0.01)
    assert result["annual_loss_median"] == 0.0
    tails = cast(dict[str, dict[str, float]], result["tail_metrics"])
    assert (tails["0.99"]["var"], tails["0.996"]["var"], tails["0.999"]["var"]) == (
        0.0,
        100.0,
        1_000.0,
    )
    assert tails["0.99"]["tvar"] == pytest.approx(434.0)
    assert tails["0.995"]["tvar"] == pytest.approx(840.0)


def test_annual_median_is_the_mixed_law_left_inverse_not_a_stratum_median() -> None:
    # Stratum masses 0.3 and 0.7 give atoms 0 (0.225), 5 (0.35), 10 (0.075)
    # and 20 (0.35). The mixed CDF first reaches one half at 5, while the
    # stratum medians are 0 and 12.5.
    draws = {
        "a": ConditionalCapabilityDraws(
            "a",
            "response",
            1,
            {"gross": ConditionalView(np.array([0.0, 0.0, 0.0, 10.0]))},
        ),
        "b": ConditionalCapabilityDraws(
            "b",
            "response",
            2,
            {"gross": ConditionalView(np.array([5.0, 5.0, 20.0, 20.0]))},
        ),
    }
    result = CapabilityMixture(draws, {"a": 0.3, "b": 0.7}).metrics("gross")
    assert result["annual_loss_median"] == 5.0
    assert result["annual_probability_positive"] == pytest.approx(0.775)


def test_tail_variance_detector_respects_executed_caps_and_mixture_mass() -> None:
    light = Distribution(p5=1, p25=2, p50=3, p75=4, p95=5, gpd_shape=0.2)
    heavy = Distribution(p5=1, p25=2, p50=3, p75=4, p95=5, gpd_shape=0.6)
    capped = Distribution(
        p5=1, p25=2, p50=3, p75=4, p95=5, gpd_shape=0.7, severity_cap=100
    )
    assert uncapped_infinite_variance_gpd_shapes(heavy) == (0.6,)
    assert uncapped_infinite_variance_gpd_shapes(capped) == ()
    mixture = MixtureSeverityDistribution(
        [light, heavy, capped], np.array([0.5, 0, 0.5])
    )
    assert uncapped_infinite_variance_gpd_shapes(mixture) == ()
    mixture = MixtureSeverityDistribution(
        [light, heavy, capped], np.array([0.25, 0.5, 0.25])
    )
    assert uncapped_infinite_variance_gpd_shapes(mixture) == (0.6,)


def test_empirical_percentile_exact_boundary_rank():
    losses = np.arange(1_000, dtype=float)

    assert weighted_percentile(losses, np.ones_like(losses), 99.9) == 998.0
    assert (
        weighted_percentile(losses, np.full(len(losses), 1.0 / len(losses)), 99.9)
        == 998.0
    )


def test_p0_returns_minimum():
    values = np.array([5.0, 10.0, 15.0])
    weights = np.ones(3)
    result = weighted_percentile(values, weights, 0.0)
    assert result == 5.0


def test_p100_returns_maximum():
    values = np.array([5.0, 10.0, 15.0])
    weights = np.ones(3)
    result = weighted_percentile(values, weights, 100.0)
    assert result == 15.0


def test_heavy_weight_shifts_percentile():
    values = np.array([10.0, 20.0, 30.0])
    weights = np.array([100.0, 1.0, 1.0])
    median = weighted_percentile(values, weights, 50.0)
    assert median == 10.0


def test_unsorted_input():
    values = np.array([30.0, 10.0, 50.0, 20.0, 40.0])
    weights = np.ones(5)
    result = weighted_percentile(values, weights, 50.0)
    assert result == pytest.approx(30.0)


@pytest.mark.parametrize("weight", [1.0, 0.1, None])
def test_equal_weight_percentiles_are_the_exact_order_statistic(weight):
    """With n equal weights the left inverse at p is order ceil(n p), counted in
    rational arithmetic; a floating cumulative sum misplaces it for many n."""
    for count in range(1, 2001):
        values = np.arange(count, dtype=float)[::-1].copy()
        ordered = SortedSample(
            values, np.full(count, 1 / count if weight is None else weight)
        )
        for percent in (5, 50, 95, 99.9):
            order = math.ceil(Fraction(str(percent)) / 100 * count)
            assert ordered.percentile(percent) == max(order, 1) - 1


def test_equal_weights_beside_zero_weight_values_keep_the_exact_order_statistic():
    """A value of zero weight is no part of the sample, so equal weights
    beside it still count the exact order. The tail gate's batches are such
    samples: their equally weighted years beside the structurally zero
    strata's atom, whose mass is zero when there are no such strata."""
    for count in range(1, 401):
        years = np.arange(count, 0, -1, dtype=float)
        ordered = SortedSample(
            np.concatenate([[0.0], years, [count + 1.0]]),
            np.concatenate([[0.0], np.full(count, 1 / count), [0.0]]),
        )
        for percent in (0, 5, 50, 95, 99.9, 100):
            order = math.ceil(Fraction(str(percent)) / 100 * count)
            assert ordered.percentile(percent) == max(order, 1)


def test_a_level_on_a_step_between_two_states_takes_the_lower_loss():
    # Two states of mass one half: ten years of losses 0 to 9 and twenty at a
    # million and above. P(L <= 9) is exactly one half, so the left inverse at
    # one half is 9, though a float running sum of the years' masses, 0.05 and
    # 0.025, reaches 0.49999999999999994 there and 1.0000000000000002 in all.
    small, large = np.arange(10.0), 1e6 + np.arange(20.0)
    ordered = SortedSample(
        np.concatenate([small, large]),
        np.concatenate([np.full(10, 0.5 / 10), np.full(20, 0.5 / 20)]),
    )
    assert ordered.percentile(50) == 9.0
    assert ordered.quantile(0.5) == 9.0
    draws = {
        name: ConditionalCapabilityDraws(
            name, "response", seed, {"gross": ConditionalView(losses)}
        )
        for seed, (name, losses) in enumerate((("small", small), ("large", large)))
    }
    result = CapabilityMixture(draws, {"small": 0.5, "large": 0.5}).metrics(
        "gross", levels=(0.5,)
    )
    assert result["annual_loss_median"] == 9.0
    tail = cast(dict[str, dict[str, float]], result["tail_metrics"])["0.5"]
    assert tail["var"] == 9.0
    # The worst half is exactly the large state, whichever end of the step
    # the VaR takes.
    assert tail["tvar"] == pytest.approx(float(np.mean(large)), rel=1e-12)


STATE_YEARS = (4, 5, 8, 10, 16, 20, 25, 40, 50, 100)
STATE_MASSES = tuple(Fraction(mass) for mass in ("1/2", "1/3", "1/4", "1/5", "3/10"))
PERCENTS = ("0", "50", "75", "90", "95", "99", "99.5", "100")


@pytest.mark.parametrize(("layout", "ties"), [("stacked", 486), ("interleaved", 678)])
def test_unequal_weight_percentiles_are_the_exact_left_inverse(layout, ties):
    """Two states' years, each weighted by its state's mass over its years,
    against the left inverse in rational arithmetic, with values of zero
    weight below, between and above them. ``ties`` of the levels strictly
    between 0 and 1 fall exactly on a step, where a float sum of the weights
    lands on either side."""
    found, wrong = 0, []
    for (first, second), mass in product(permutations(STATE_YEARS, 2), STATE_MASSES):
        if layout == "stacked":
            years = [*range(first), *range(10**6, 10**6 + second)]
        else:
            years = [*range(0, 2 * first, 2), *range(1, 2 * second, 2)]
        values = np.array([*years, -1, 5 * 10**5, 2 * 10**6], dtype=float)
        shares = [mass / first] * first + [(1 - mass) / second] * second + [0] * 3
        weights = np.array(
            [float(mass) / first] * first
            + [float(1 - mass) / second] * second
            + [0.0] * 3
        )
        ordered = SortedSample(values, weights)
        steps = exact_steps(values, shares)
        for percent in PERCENTS:
            level = Fraction(percent) / 100
            found += 0 < level < 1 and any(share == level for _, share in steps)
            expected = exact_left_inverse(steps, level)
            got = (ordered.percentile(float(percent)), ordered.quantile(float(level)))
            if got != (expected, expected):
                wrong.append((first, second, str(mass), percent, got, expected))
    assert wrong == []
    assert found == ties


def test_a_cumulative_weight_within_the_tie_tolerance_reaches_the_level():
    """A float weight holds its share of the total only to rounding, so a
    cumulative weight within the tie tolerance below a level's share reaches
    it; one further below does not."""
    values = np.array([1.0, 2.0, 3.0])
    for shortfall, median in ((0.5, 1.0), (4.0, 2.0)):
        # The first value falls ``shortfall`` tolerances short of half the
        # weight, and the middle value carries twice that.
        weights = np.array([1.0, 2 * shortfall * TIE_RELATIVE_TOLERANCE, 1.0])
        assert SortedSample(values, weights).percentile(50) == median


def test_cumulative_weights_do_not_drift_with_their_number():
    """Each of 10,000 weights below half the last place of the first leaves
    a float running sum where it was, 1e-12 short at the end; every corrected
    sum is within a rounding of the exact one, and none is below the one
    before it."""
    weights = np.concatenate([[1.0], np.full(10_000, 1e-16), [1.0]])
    sums = _running_sums(weights)
    exact = Fraction(0)
    for weight, corrected in zip(weights, sums, strict=True):
        exact += Fraction(float(weight))
        assert abs(Fraction(float(corrected)) - exact) <= math.ulp(float(exact))
    assert np.all(np.diff(sums) >= 0)


@pytest.mark.parametrize("exponent", [-1070, -500, 0, 500, 1024])
def test_percentiles_do_not_depend_on_the_scale_of_the_weights(exponent):
    """Unequal weights from the least floats to ones whose sum is beyond the
    largest give the percentiles of the same weights at unit scale."""
    values = np.array([30.0, 10.0, 20.0])
    weights = np.array([math.ldexp(share, exponent) for share in (0.25, 0.5, 0.25)])
    ordered = SortedSample(values, weights)
    assert [ordered.percentile(q) for q in (0, 50, 51, 75, 76, 100)] == [
        10.0,
        10.0,
        20.0,
        20.0,
        30.0,
        30.0,
    ]


@pytest.mark.parametrize("level", [0.07, 0.14, 0.28, 0.56])
def test_a_tail_level_is_read_as_the_decimal_it_was_written_as(level):
    """The level's share of 100 equal years is a whole number of years, the
    VaR's order. ``100 * level`` in floats is just above the percent the
    level states, which would take the next year up."""
    assert 100 * level > round(100 * level)
    losses = np.arange(100.0)[::-1].copy()
    masses = np.full(100, 0.01)
    var = round(100 * level) - 1.0
    assert SortedSample(losses, masses).quantile(level) == var
    assert tail_parts(losses, masses, level).var == var
    assert weighted_var_tvar(losses, masses, level)[0] == var


@pytest.mark.parametrize(
    ("values", "weights", "message"),
    [
        ([[1.0, 2.0]], [[1.0, 1.0]], "values and weights must be 1-D arrays"),
        ([1.0, 2.0], [[1.0, 1.0]], "values and weights must be 1-D arrays"),
        ([1.0, 2.0], [1.0], "values and weights must have the same shape"),
        ([], [], "values must not be empty"),
        ([1.0, math.nan], [1.0, 1.0], "values must be finite"),
        ([1.0, math.inf], [1.0, 1.0], "values must be finite"),
        ([1.0, 2.0], [1.0, math.nan], "weights must be finite"),
        ([1.0, 2.0], [1.0, math.inf], "weights must be finite"),
        ([1.0, 2.0], [1.0, -1e-300], "weights must be non-negative"),
        ([1.0, 2.0], [0.0, 0.0], "weights must have positive total weight"),
    ],
)
def test_a_sample_must_be_aligned_finite_and_weighted(values, weights, message):
    with pytest.raises(ValueError, match=message):
        SortedSample(np.array(values), np.array(weights))


@pytest.mark.parametrize("uniform", [True, False])
@pytest.mark.parametrize("outside", [-1e-9, 1.000000001, math.nan, math.inf])
def test_a_level_must_lie_in_its_range(uniform, outside):
    ordered = SortedSample(
        np.array([1.0, 2.0]), np.array([1.0, 1.0 if uniform else 2.0])
    )
    with pytest.raises(ValueError, match=r"level must be in \[0, 1\]"):
        ordered.quantile(outside)
    with pytest.raises(ValueError, match=r"q must be in \[0, 100\]"):
        ordered.percentile(100 * outside)
