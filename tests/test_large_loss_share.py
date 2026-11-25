"""The large-loss share, its tail bands and when the anchor shapes the law."""

import math

import pytest

from catmodel.contracts.elicitation import QUANTILES, Quantiles
from catmodel.elicitation.epistemic import epistemic_mean
from catmodel.elicitation.large_loss import (
    amount_band,
    contradiction,
    large_loss_pin,
    large_loss_share,
    quantile_large_loss_share,
    share_band,
)

P95, P99, AMOUNT = 6e6, 2e7, 1e9


def test_share_is_the_ratio_of_means_and_exact_zero_means_no_anchor():
    assert large_loss_share(2_000.0, 0.4) == pytest.approx(2e-4)
    assert large_loss_share(2_000.0, 0.0) is None
    assert large_loss_share(0.0, 0.0) is None
    with pytest.raises(ValueError, match="nonnegative"):
        large_loss_share(-1.0, 0.1)


@pytest.mark.parametrize(
    ("amount", "band"),
    [(5e6, "body"), (P95, "body"), (1e7, "near"), (P99, "near"), (2.0001e7, "far")],
)
def test_amount_band_uses_inclusive_upper_edges(amount, band):
    assert amount_band(amount, P95, P99) == band


@pytest.mark.parametrize(
    ("share", "band"),
    [(1.0, "body"), (0.05, "body"), (0.0499, "near"), (0.01, "near"), (0.0099, "far")],
)
def test_share_band_edges(share, band):
    assert share_band(share) == band


@pytest.mark.parametrize("share", (0.0, -0.1, 1.5))
def test_share_band_refuses_values_outside_the_unit_interval(share):
    with pytest.raises(ValueError, match="share"):
        share_band(share)


@pytest.mark.parametrize(
    ("p95", "p99", "share"),
    [
        (P95, P99, 2e-4),  # far and far
        (2e8, 2e9, 0.02),  # near and near
        (2e9, 5e10, 0.3),  # body and body
        (2e8, AMOUNT, 0.01),  # p99 exactly at the level with share 1 in 100
    ],
)
def test_coherent_answers_have_no_contradiction(p95, p99, share):
    assert contradiction(AMOUNT, share, p95, p99) is None


@pytest.mark.parametrize(
    ("p95", "p99", "share", "fragment"),
    [
        (P95, P99, 0.01, "fewer than 1 in 100"),
        (P95, P99, 0.2, "fewer than 1 in 100"),
        (2e8, 2e9, 0.005, "from 1 in 100 up to but not including 1 in 20"),
        (2e8, 2e9, 0.05, "from 1 in 100 up to but not including 1 in 20"),
        (AMOUNT, 5e10, 0.049, "at least 1 in 20"),
    ],
)
def test_contradictions_state_the_numbers_neutrally(p95, p99, share, fragment):
    message = contradiction(AMOUNT, share, p95, p99)
    assert message is not None
    assert fragment in message
    assert "USD 1,000,000,000" in message
    assert f"share {share:.4g}" in message


def test_share_above_one_contradicts_any_quantiles():
    message = contradiction(AMOUNT, 1.2, 2e9, 5e10)
    assert message is not None and "more than every qualified unit" in message


@pytest.mark.parametrize(
    ("share", "p99", "pinned"),  # p99 is the law's own computed p99
    [
        (2e-4, P99, True),
        (0.0099, P99, True),
        (0.01, P99, False),
        (None, P99, False),
        (2e-4, AMOUNT, False),
        (2e-4, 2e9, False),
        (2e-4, None, False),
    ],
)
def test_anchor_shapes_the_law_only_above_its_p99_with_a_share_below_one_percent(
    share, p99, pinned
):
    result = large_loss_pin(AMOUNT, share, p99)
    assert (result == (AMOUNT, share)) if pinned else result is None


# Quantiles whose large-loss count is a decimal multiple of the count: the
# ratio of computed means misses the multiple by an ulp, either way by case.
_RATES = ((12, 30, 55, 140, 410), (3, 7, 15, 40, 120), (20, 50, 100, 250, 800))


def _quantiles(values):
    return Quantiles(**dict(zip(QUANTILES, values, strict=True)))


@pytest.mark.parametrize("rate", _RATES)
@pytest.mark.parametrize("divisor", (100, 20))
def test_a_stated_proportional_anchor_has_its_exact_decimal_share(rate, divisor):
    large = [float(repr(value / divisor)) for value in rate]
    share = quantile_large_loss_share(_quantiles(rate), _quantiles(large))
    assert share == 1 / divisor


def test_a_non_proportional_anchor_keeps_the_ratio_of_means():
    rate, large = _quantiles((12, 30, 55, 140, 410)), _quantiles((0, 0, 0.1, 1, 5))
    assert quantile_large_loss_share(rate, large) == pytest.approx(
        epistemic_mean(large) / epistemic_mean(rate), rel=1e-15
    )
    assert quantile_large_loss_share(rate, _quantiles((0,) * 5)) is None


@pytest.mark.parametrize("boundary", (0.05, 0.01))
@pytest.mark.parametrize("ulps", (-2, -1, 0, 1, 2))
def test_a_share_at_a_band_boundary_takes_the_same_band_whichever_way_it_rounds(
    boundary, ulps
):
    share = boundary
    for _ in range(abs(ulps)):
        share = math.nextafter(share, math.inf if ulps > 0 else 0.0)
    assert share_band(share) == share_band(boundary)
    if boundary == 0.01:
        # p99 exactly at the level and a share of 1 in 100: coherent, and the
        # anchor never shapes the law, on every platform.
        assert contradiction(AMOUNT, share, 2e8, AMOUNT) is None
        assert large_loss_pin(AMOUNT, share, 1e8) is None
        assert contradiction(AMOUNT, share, P95, P99) is not None


def test_a_share_of_one_reached_by_roundoff_is_every_unit_not_more():
    share = math.nextafter(1.0, 2.0)
    assert share_band(share) == "body"
    assert contradiction(AMOUNT, share, AMOUNT, 5e10) is None
