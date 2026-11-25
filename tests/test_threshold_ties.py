"""Threshold decisions are the same on every platform and at exact ties."""

import math

import pytest

from catmodel.probability.distributions import (
    Distribution,
    uncapped_infinite_variance_gpd_shapes,
)
from catmodel.probability.numerics import (
    decimal_value,
    differs_by_more_than,
    significant,
)
from catmodel.scenarios.sensitivity_coverage import (
    TAIL_SHAPE_SENSITIVITY_STEP,
    shifted_tail_shape,
)


def _ulps(value: float, count: int) -> float:
    for _ in range(abs(count)):
        value = math.nextafter(value, math.inf if count > 0 else -math.inf)
    return value


@pytest.mark.parametrize(
    ("low", "high", "ratio"),
    [(0.3, 0.9, 3.0), (0.7, 2.1, 3.0), (0.07, 0.7, 10.0), (1.1, 3.3, 3.0)],
)
def test_stored_decimals_exactly_at_the_factor_do_not_differ(low, high, ratio):
    assert decimal_value(high) == decimal_value(ratio) * decimal_value(low)
    assert not differs_by_more_than(low, high, ratio)
    assert not differs_by_more_than(high, low, ratio)
    assert differs_by_more_than(low, _ulps(high, 1), ratio)


@pytest.mark.parametrize("ulps", (-2, -1, 0, 1, 2))
@pytest.mark.parametrize(
    ("midpoint", "rounded"),
    [(1.2345, 1.234), (1.2355, 1.236), (0.012345, 0.01234), (98765.0, 98760.0)],
)
def test_four_significant_figures_round_a_midpoint_the_same_way(
    ulps, midpoint, rounded
):
    assert significant(_ulps(midpoint, ulps)) == rounded


@pytest.mark.parametrize(
    ("shape", "step", "threshold"),
    [
        # Infinite variance from 0.5 and no mean from 1, then the operating
        # bounds of an uncapped shape.
        (0.7, -0.2, 0.5),
        (1.15, -0.15, 1.0),
        (0.8, 0.15, 0.95),
        (-0.8, -0.15, -0.95),
    ],
)
def test_a_stepped_tail_shape_is_the_decimal_sum_where_binary_addition_misses_it(
    shape, step, threshold
):
    assert shape + step != threshold
    assert shifted_tail_shape(shape, step) == threshold
    assert decimal_value(shape) + decimal_value(step) == decimal_value(threshold)
    assert shifted_tail_shape(threshold, -step) == shape


def test_a_shape_stepped_down_to_one_half_has_infinite_variance():
    step = TAIL_SHAPE_SENSITIVITY_STEP
    below = 0.7 - step
    # Binary subtraction gives the float just below one half.
    assert below == math.nextafter(0.5, 0.0) == 0.49999999999999994
    lower = shifted_tail_shape(0.7, -step)
    assert lower == 0.5
    law = Distribution(p5=1, p25=2, p50=3, p75=4, p95=5, gpd_shape=0.7)
    assert uncapped_infinite_variance_gpd_shapes(law) == (0.7,)
    assert uncapped_infinite_variance_gpd_shapes(law.with_gpd_shape(lower)) == (0.5,)
    # The threshold is a stated decimal compared exactly, with no tie
    # tolerance, so the step must give the decimal: the float below one half
    # is a finite-variance shape.
    assert uncapped_infinite_variance_gpd_shapes(law.with_gpd_shape(below)) == ()
