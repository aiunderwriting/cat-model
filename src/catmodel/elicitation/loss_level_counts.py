"""Loss-level counts: the expected annual numbers of qualified units at or
above loss levels that a leaf's count and severity answers imply.

The leaf EAL cross-check question shows them and its exhibit reports them
(``FREQ-EAL-CROSSCHECK``). Within one case the severity law does not move
with the count's epistemic rank, so the expected annual number of qualified
units whose gross loss is at least ``x`` is the count law's epistemic mean
times ``P(X >= x)``. Each count is computed from a stored
four-significant-figure count mean, so the question, the exhibit and replay
agree on every platform.

The stated ``total_loss`` quantiles bound ``P(X >= x)`` whatever the law's
form and whether or not it has atoms. Read each stated quantile as
``Q(p) = inf{y : F(y) >= p}``. If ``Q(p) >= x`` then
``P(X >= x) >= P(X >= Q(p)) = 1 - F(Q(p)-) >= 1 - p``, because ``F(y) < p``
below ``Q(p)``. If ``Q(p) < x`` then ``P(X >= x) <= P(X > Q(p)) =
1 - F(Q(p)) <= 1 - p``, because ``F`` is right-continuous. The lower bound
is the largest such ``1 - p`` and the upper bound the smallest; with no
stated quantile below ``x`` the upper bound is one, and above the stated
p99 the lower bound is zero.

At a level equal to exactly one stated quantile ``Q(p)`` the count is one
value, the count mean times ``1 - p``: the stated quantile read as the share
of units at or above it, exact for a law with no mass exactly at that amount
and a lower bound for any law. It is rounded to the nearest four significant
figures. At a fixed reporting level, and at tied quantiles, which force an
atom, the count is a pair of bounds rounded outward to four significant
figures, the lower one down and the upper one up, so rounding never narrows
them.

Above p99 the large-loss counts and their anchor govern the run
(``SEV-LARGE-LOSS-ANCHOR``); they are reported elsewhere and not inferred
here, and a stated p999 is not used. The counts describe the stated, uncapped
shared total before any taxonomy-parent factor: additive components outside
it are excluded, and where compilation admits or reshapes the law they
describe the answers, not the compiled law. They change no law, count, weight
or result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any

from catmodel.probability.numerics import significant

FIXED_LOSS_LEVELS: tuple[float, ...] = (1_000_000.0, 100_000_000.0)
"""Fixed gross-loss levels, in policy-start USD, at which every leaf case
reports its loss-level counts: a reporting choice between the admission
threshold and the large-loss level, not an assumption of any law."""

STATED_TAILS: tuple[tuple[str, float], ...] = (
    ("p5", 0.95),
    ("p25", 0.75),
    ("p50", 0.5),
    ("p75", 0.25),
    ("p95", 0.05),
    ("p99", 0.01),
)
"""Each stated ``total_loss`` quantile used, with ``1 - p``."""

REPORTED_QUANTILES = ("p50", "p75", "p95", "p99")
"""The stated quantiles at which each case reports its loss-level counts,
beside ``FIXED_LOSS_LEVELS``."""

LOSS_LEVEL_COUNTS_NOTE = (
    "Loss-level counts: implied expected annual counts of qualified units with "
    "gross loss at or above each level, the stored count mean times the share "
    "of units the stated total_loss quantiles put there. At a level equal to "
    "one stated quantile the count is one value, the mean times one minus the "
    "quantile's probability, exact for a law with no mass exactly at that "
    "amount and a lower bound otherwise; at a fixed reporting level, or at "
    "tied quantiles, it is a pair of bounds that hold for any law with the "
    "stated quantiles, rounded outward to four significant figures. Above p99 "
    "only the one-in-a-hundred bound is given; the large-loss anchor governs "
    "there. Stated answers before any taxonomy-parent factor, without "
    "additive components; nothing here changes a law, count, weight or result."
)


def _outward(value: float, rounding: str) -> float:
    """``value`` at four significant figures, rounded down (``ROUND_FLOOR``)
    or up (``ROUND_CEILING``) and never across it.

    The float's exact binary value is rounded in decimal; converting the
    result back rounds to the nearest float, which is monotone, so the result
    stays on the requested side of ``value``.
    """
    if value == 0 or not math.isfinite(value):
        return float(value)
    exact = Decimal(value)
    step = Decimal(1).scaleb(exact.adjusted() - 3)
    return float(exact.quantize(step, rounding=rounding))


def stated_tails(total_loss: object) -> tuple[tuple[str, float, float], ...]:
    """Each stated quantile's name and value with ``1 - p``."""
    return tuple(
        (name, float(getattr(total_loss, name)), tail) for name, tail in STATED_TAILS
    )


def exceedance_bounds(total_loss: object, level: float) -> tuple[float, float]:
    """Bounds on ``P(X >= level)`` valid for every law with the stated
    quantiles, atoms included (see the module docstring)."""
    tails = stated_tails(total_loss)
    low = max((tail for _, value, tail in tails if value >= level), default=0.0)
    high = min((tail for _, value, tail in tails if value < level), default=1.0)
    return low, high


@dataclass(frozen=True)
class LossLevelCount:
    """The implied expected annual count of qualified units with gross loss at
    least ``level``.

    ``basis`` names the stated quantile equal to ``level``, the tied
    quantiles joined by ``=``, or ``fixed``. At a level equal to exactly one
    stated quantile the count is one value and ``annual_count_low`` equals
    ``annual_count_high``; otherwise they are bounds rounded outward.
    """

    level: float
    basis: str
    annual_count_low: float
    annual_count_high: float

    @property
    def single(self) -> bool:
        return self.annual_count_low == self.annual_count_high

    def record(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "basis": self.basis,
            "annual_count_low": self.annual_count_low,
            "annual_count_high": self.annual_count_high,
        }


def loss_level_counts(mean: float, total_loss: object) -> tuple[LossLevelCount, ...]:
    """The loss-level counts at the stored count mean ``mean`` beside the
    stated ``total_loss`` quantiles, at ``REPORTED_QUANTILES`` and
    ``FIXED_LOSS_LEVELS``, by level."""
    mean = float(mean)
    tails = stated_tails(total_loss)
    levels = {float(getattr(total_loss, name)) for name in REPORTED_QUANTILES} | set(
        FIXED_LOSS_LEVELS
    )
    counts: list[LossLevelCount] = []
    for level in sorted(levels):
        named = [(name, tail) for name, value, tail in tails if value == level]
        if len(named) == 1:
            low = high = significant(mean * named[0][1])
        else:
            share_low, share_high = exceedance_bounds(total_loss, level)
            low = _outward(mean * share_low, ROUND_FLOOR)
            high = _outward(mean * share_high, ROUND_CEILING)
        counts.append(
            LossLevelCount(
                level=level,
                basis="=".join(name for name, _ in named) or "fixed",
                annual_count_low=low,
                annual_count_high=high,
            )
        )
    return tuple(counts)
