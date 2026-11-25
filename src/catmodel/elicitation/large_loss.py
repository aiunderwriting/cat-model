"""The large-loss anchor's share, its tail band and when it shapes the law.

A respondent states the expected annual number of admitted incidents at or
above a fixed amount A beside the expected annual count of all admitted
incidents (``SEV-LARGE-LOSS-ANCHOR``). Their epistemic means give the
large-loss share pi = E[large-loss count] / E[count], the probability that an
admitted incident reaches A. The share is a severity mark: held fixed across
count ranks, so severity never varies with the rank and the compiled model's
expected number of incidents at or above A is the count mean times S(A).

The anchor shapes the law only where A lies above the law's own computed p99
(``Distribution.large_loss_break``, which equals the stated p99 unless a
scale pin places it elsewhere) and the share is below 0.01: the tail above
that p99 then passes through A with survival pi
(``Distribution.large_loss_amount``). Elsewhere the stated quantiles govern
and the anchor is a diagnostic. A share is coherent with the stated
quantiles when A and the share fall in the same tail band: at or below p95
with a share of at least 0.05, between p95 and p99 with a share in
[0.01, 0.05), or above p99 with a share below 0.01.

The share is a ratio of computed means, so its last bit can differ between
platforms. Where the large-loss count's five quantiles are one decimal
multiple of the count's, the share is that multiple exactly, read from the
stated decimals; otherwise it is the ratio of the computed means, and a share
within ``TIE_RELATIVE_TOLERANCE`` below a band boundary is at the boundary,
in the band above it. Every decision that reads the share's band uses
``share_band``, so a share of exactly 0.01 is in the near band everywhere.

The same question states the count at a second, higher amount. No reference
law reads it. The two-anchor comparison (``CompileOptions.
very_large_loss_anchor``) takes its share pi2 = E[very-large-loss count] /
E[count] as a second point of the tail: above a first anchor that shapes the
law where 0 < pi2 < pi (``very_large_loss_pin``), and otherwise above the
law's own p99, under the first anchor's own rule, where the first amount lies
within the stated quantiles.
"""

from __future__ import annotations

from typing import Literal

from catmodel.contracts.elicitation import QUANTILES, Quantiles
from catmodel.elicitation.epistemic import epistemic_mean
from catmodel.probability.distributions import LARGE_LOSS_BREAK_EXCEEDANCE
from catmodel.probability.numerics import at_least, decimal_value, significant

TailBand = Literal["body", "near", "far"]
"""Where an amount lies against stated p95 and p99, or a share against 0.05 and 0.01."""

SPLICE_EXCEEDANCE = 0.05
"""Exceedance probability of the stated p95, where the GPD tail begins."""


def large_loss_share(count_mean: float, large_loss_mean: float) -> float | None:
    """The share of admitted incidents at or above the level, or None.

    None when the count mean is zero (no admitted incident to share) or the
    large-loss mean is exactly zero: an exact-zero anchor means "no anchor",
    and the stated quantiles govern.
    """
    if count_mean < 0 or large_loss_mean < 0:
        raise ValueError("expected counts must be nonnegative")
    if count_mean == 0 or large_loss_mean == 0:
        return None
    return large_loss_mean / count_mean


def quantile_large_loss_share(
    rate: Quantiles, large_loss_count: Quantiles
) -> float | None:
    """The share two count laws' epistemic means give, or None
    (``large_loss_share``).

    The epistemic mean scales with its quantiles, so where every stated
    large-loss quantile is the same decimal multiple of the count's (a zero
    count quantile allowing only a zero beside it) the share is that multiple
    exactly, computed from the stated decimals; otherwise it is the ratio of
    the computed means.
    """
    count_mean = epistemic_mean(rate)
    pairs = [
        (
            decimal_value(getattr(rate, key)),
            decimal_value(getattr(large_loss_count, key)),
        )
        for key in QUANTILES
    ]
    multiples = {large / count for count, large in pairs if count}
    if (
        count_mean > 0
        and len(multiples) == 1
        and all(large == 0 for count, large in pairs if not count)
    ):
        (multiple,) = multiples
        return float(multiple) if multiple else None
    return large_loss_share(count_mean, epistemic_mean(large_loss_count))


def amount_band(amount: float, p95: float, p99: float) -> TailBand:
    """Where an amount lies against the stated p95 and p99."""
    if amount <= p95:
        return "body"
    if amount <= p99:
        return "near"
    return "far"


def share_band(share: float) -> TailBand:
    """Which stated tail region a positive share of at most one implies.

    A share within ``TIE_RELATIVE_TOLERANCE`` below 0.05 or 0.01 is at that
    boundary, in the band above it: the band of the exact share whose
    computed value may round either way.
    """
    if not share > 0 or exceeds_one(share):
        raise ValueError(f"a large-loss share must lie in (0, 1], got {share}")
    if at_least(share, SPLICE_EXCEEDANCE):
        return "body"
    if at_least(share, LARGE_LOSS_BREAK_EXCEEDANCE):
        return "near"
    return "far"


def exceeds_one(share: float) -> bool:
    """Whether a share is more than one beyond a tie: one reached by
    roundoff is every qualified unit, not more."""
    return not at_least(1.0, share)


def contradiction(amount: float, share: float, p95: float, p99: float) -> str | None:
    """Why a positive share contradicts the stated quantiles, or None.

    Only a real contradiction counts: the share implies a different tail band
    from the one the amount occupies, or it exceeds one. The message states
    the numbers, in the questions' term "qualified units", and is neutral
    about which answer should change.
    """
    one_in = significant(1 / share)
    implied = f"imply that about 1 in {one_in:,.4g} qualified units (share {significant(share):.4g}) reaches {_usd(amount)}"
    if exceeds_one(share):
        return f"Your large-loss count and count {implied}, more than every qualified unit."
    band = amount_band(amount, p95, p99)
    if band == share_band(share):
        return None
    required = {
        "body": f"your p95 ({_usd(p95)}) is at or above {_usd(amount)}, which means at least 1 in 20 qualified units reach it",
        "near": f"your p95 ({_usd(p95)}) is below and your p99 ({_usd(p99)}) at or above {_usd(amount)}, which means from 1 in 100 up to but not including 1 in 20 qualified units reach it",
        "far": f"your p99 ({_usd(p99)}) is below {_usd(amount)}, which means fewer than 1 in 100 qualified units reach it",
    }[band]
    return f"Your large-loss count and count {implied}, but {required}."


def large_loss_pin(
    amount: float, share: float | None, break_amount: float | None
) -> tuple[float, float] | None:
    """The anchor point that shapes the law above its p99, or None.

    ``break_amount`` is the law's own computed p99 (``Distribution.
    large_loss_break``), not the stated one, so the law can always take the
    pin. The anchor acts only where the amount lies above it and the share is
    positive and in the far band (``share_band``): below 0.01 beyond a tie.
    Anywhere else the stated quantiles govern.
    """
    if (
        share is None
        or not 0 < share < LARGE_LOSS_BREAK_EXCEEDANCE
        or share_band(share) != "far"
    ):
        return None
    if break_amount is None or amount <= break_amount:
        return None
    return amount, share


def very_large_loss_pin(
    amount: float, share: float | None, first: tuple[float, float]
) -> tuple[float, float] | None:
    """The second point of a two-anchor tail above a first anchor that
    shapes the law, or None.

    ``first`` is the first anchor's point (``large_loss_pin``). The second
    acts only where its amount lies above the first's and its share is
    positive and below the first's beyond a tie: incidents at or above the
    higher amount are a subset of those at or above the lower, and a share
    within ``TIE_RELATIVE_TOLERANCE`` below the first's is the first's, which
    would leave no incident between the two amounts. An exact-zero share is
    None (``large_loss_share``) and means no second anchor.
    """
    first_amount, first_share = first
    if (
        share is None
        or not share > 0
        or at_least(share, first_share)
        or amount <= first_amount
    ):
        return None
    return amount, share


def _usd(value: float) -> str:
    return f"USD {value:,.0f}" if value >= 1 else f"USD {value:g}"
