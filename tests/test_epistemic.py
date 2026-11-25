"""Independent checks of the five-quantile epistemic law and rank stretching."""

import math

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtri

from catmodel.contracts.elicitation import Quantiles
from catmodel.elicitation.epistemic import (
    EPISTEMIC_TAIL_WIDTH_CAP,
    epistemic_quantiles,
)

RANKS = (0.05, 0.25, 0.5, 0.75, 0.95)


def epistemic_quantile(estimate, rank, **options):
    """The law at one rank, through the vectorized reading."""
    return float(epistemic_quantiles(estimate, np.asarray([rank]), **options)[0])


def ladder(*values):
    return Quantiles(
        **dict(zip(("p5", "p25", "p50", "p75", "p95"), values, strict=True))
    )


@pytest.mark.parametrize("scale", [0.75, 1, 1.5])
def test_five_anchors_and_central_interpolation_are_invariant(scale):
    values = (0.1, 1, 5, 20, 100)
    estimate = ladder(*values)
    for rank, value in zip(RANKS, values, strict=True):
        assert epistemic_quantile(estimate, rank, epistemic_tail_scale=scale) == value
    for rank in (0.06, 0.13, 0.3, 0.7, 0.94):
        assert epistemic_quantile(
            estimate, rank, epistemic_tail_scale=scale
        ) == PchipInterpolator(RANKS, values)(rank)


@pytest.mark.parametrize(
    "values",
    [
        (0.1, 1, 5, 20, 100),
        (2, 2, 3, 5, 5),
        (0, 0, 0, 0, 100),
        (0, 100, 100, 100, 100),
    ],
)
def test_open_ranks_monotonicity_and_boundary_fallbacks(values):
    estimate = ladder(*values)
    ranks = [np.nextafter(0.0, 1.0), 1e-12, 0.01, *RANKS, 0.99, np.nextafter(1.0, 0.0)]
    results = [epistemic_quantile(estimate, rank) for rank in ranks]
    assert np.all(np.isfinite(results))
    assert np.all(np.diff(results) >= 0)
    assert results[0] >= 0
    # A tied edge pair continues its flat quantile function.
    if values[3] == values[4]:
        assert results[-1] == values[-1]
    else:
        assert results[-1] > values[-1]
    if values[0] in (values[1], 0):
        assert results[0] == values[0]
    else:
        assert 0 < results[0] < values[0]


def test_single_positive_anchor_takes_the_capped_log_tail():
    z95 = float(ndtri(0.95))
    assert epistemic_quantile(ladder(0, 0, 0, 0, 100), 0.99) == pytest.approx(
        100 * math.exp(z95 * (float(ndtri(0.99)) - z95))
    )


@pytest.mark.parametrize(
    "limit,approaching",
    [
        ((0, 0, 0, 0, 5), lambda e: (0, 0, 0, e, 5)),
        ((1, 2, 3, 10, 10), lambda e: (1, 2, 3, 10 - e, 10)),
        ((0, 1, 2, 3, 4), lambda e: (e, 1, 2, 3, 4)),
        ((2, 2, 3, 4, 5), lambda e: (2, 2 + e, 3, 4, 5)),
    ],
)
def test_tails_are_continuous_at_ties_and_zero(limit, approaching):
    ranks = (1e-9, 0.01, 0.04, 0.96, 0.99, 1 - 1e-9)
    reference = [epistemic_quantile(ladder(*limit), u) for u in ranks]
    for epsilon in (1e-3, 1e-6, 1e-9, 1e-12):
        estimate = ladder(*approaching(epsilon))
        assert [epistemic_quantile(estimate, u) for u in ranks] == pytest.approx(
            reference, rel=10 * epsilon, abs=10 * epsilon
        )


@pytest.mark.parametrize("scale", [0.75, 1, 1.5])
@pytest.mark.parametrize(
    "values",
    [
        (0.1, 1, 5, 20, 100),
        (0.1, 1, 5, 20, 1000),
        (1e-6, 1e-4, 1e-3, 1e-2, 1),
        (0, 0, 0, 0.001, 5),
    ],
)
def test_wide_ladders_take_the_capped_tail_width(values, scale):
    estimate = ladder(*values)
    z99, z999 = ndtri([0.99, 0.999])
    upper99, upper999 = (
        epistemic_quantile(estimate, u, epistemic_tail_scale=scale)
        for u in (0.99, 0.999)
    )
    assert pytest.approx(float(ndtri(0.95))) == EPISTEMIC_TAIL_WIDTH_CAP
    # Every one of these raw secants exceeds the cap, so the cap is the width.
    assert math.log(upper999 / upper99) / (z999 - z99) == pytest.approx(
        scale * EPISTEMIC_TAIL_WIDTH_CAP
    )
    # Scaling every quantile scales the law; no tail width depends on level.
    assert epistemic_quantile(
        ladder(*(10 * value for value in values)), 0.999, epistemic_tail_scale=scale
    ) == pytest.approx(10 * upper999)


@pytest.mark.parametrize("value", [0, 1, 1e6])
def test_constants_are_deterministic(value):
    assert epistemic_quantile(ladder(*([value] * 5)), 0.999) == value


@pytest.mark.parametrize(
    "rank,scale", [(0, 1), (1, 1), (float("nan"), 1), (0.5, 0), (0.5, float("inf"))]
)
def test_invalid_arguments_rejected_even_for_constants(rank, scale):
    with pytest.raises(ValueError):
        epistemic_quantile(ladder(1, 1, 1, 1, 1), rank, epistemic_tail_scale=scale)


def test_lower_tail_rounding_to_zero_stays_positive():
    value = epistemic_quantile(
        ladder(1e-300, 1, 2, 3, 4), np.nextafter(0.0, 1.0), epistemic_tail_scale=50
    )
    assert value == np.nextafter(0.0, 1.0)
