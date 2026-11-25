"""Weighted percentiles of a loss sample, shared by simulation and reduction."""

import math
from fractions import Fraction

import numpy as np

from catmodel.probability.numerics import TIE_RELATIVE_TOLERANCE, decimal_value


def _share(value: float, whole: int, name: str) -> Fraction:
    """``value / whole`` as a fraction, with ``value`` read as the decimal it
    was written as (``decimal_value``): 0.07 of 1 is seven hundredths."""
    value = float(value)
    if not 0.0 <= value <= whole:
        raise ValueError(f"{name} must be in [0, {whole}], got {value}")
    return decimal_value(value) / whole


def _running_sums(weights: np.ndarray) -> np.ndarray:
    """The running sums of nonnegative ``weights``, never decreasing and each
    within a rounding of its exact value.

    A float running sum rounds once per addition, which over two million
    weights compounds to as much as 2e-10 of the sum and depends on their
    order. Each addition's rounding error is recovered exactly (Knuth's
    two-sum) and the errors' own running sum is added back. That sum rounds
    too, but to second order: by less than one rounding of the result for
    fewer than 10⁸ weights.
    """
    sums = np.cumsum(weights)
    before, after = sums[:-1], sums[1:]
    virtual = after - before
    lost = (before - (after - virtual)) + (weights[1:] - virtual)
    # No rounding error exceeds the smaller of the two numbers added, so a
    # corrected sum is never below the one before it.
    after += np.cumsum(lost)
    return sums


class SortedSample:
    """A weighted sample sorted once, so its quantiles at several levels share
    one sort and one validation.

    ``quantile`` and ``percentile`` are the left inverse of the weighted
    empirical distribution function: the least value whose cumulative weight
    reaches the level's share of the total. A level that falls on a step
    takes the lower value, and a value of zero weight is no part of the
    sample.

    With equal weights the order ``ceil(n p)`` is counted in rational
    arithmetic, so p5 of 20 values is the first. Unequal weights are floats
    that hold a share, such as a state's mass over its years, only to
    rounding. So a cumulative weight within ``TIE_RELATIVE_TOLERANCE`` below
    the level's share reaches it, and each cumulative weight is summed to
    within a rounding (``_running_sums``). The rounding of the weights, how
    many are summed and the order of equal values then move no level off
    its step.
    """

    def __init__(self, values: np.ndarray, weights: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if values.ndim != 1 or weights.ndim != 1:
            raise ValueError("values and weights must be 1-D arrays")
        if values.shape != weights.shape:
            raise ValueError("values and weights must have the same shape")
        if len(values) == 0:
            raise ValueError("values must not be empty")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must be finite")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite")
        if np.any(weights < 0.0):
            raise ValueError("weights must be non-negative")
        weighted = weights > 0.0
        if not np.any(weighted):
            raise ValueError("weights must have positive total weight")
        if not np.all(weighted):
            values, weights = values[weighted], weights[weighted]
        idx = np.argsort(values)
        self._values = values[idx]
        self._cumulative: np.ndarray | None = None
        if not np.all(weights == weights[0]):
            # A power of two rescales the weights exactly, the largest to
            # below one, so that no sum of them can overflow.
            exponent = math.frexp(float(np.max(weights)))[1]
            self._cumulative = _running_sums(np.ldexp(weights[idx], -exponent))
        # Read-only, so a shared sort cannot be changed under its users.
        self._values.flags.writeable = False
        if self._cumulative is not None:
            self._cumulative.flags.writeable = False

    def quantile(self, level: float) -> float:
        """The left-inverse weighted quantile at probability ``level`` in
        [0, 1]."""
        return self._left_inverse(_share(level, 1, "level"))

    def percentile(self, q: float) -> float:
        """The left-inverse weighted percentile at ``q`` in [0, 100]."""
        return self._left_inverse(_share(q, 100, "q"))

    def _left_inverse(self, share: Fraction) -> float:
        if self._cumulative is None:
            # The least order whose share ``order / count`` reaches the level.
            count = len(self._values)
            order = -((-share.numerator * count) // share.denominator)
            return float(self._values[min(max(order, 1), count) - 1])
        # The last cumulative weight is the total, and no level's share of it
        # exceeds it, so the search always ends on a value.
        reached = float(share) * float(self._cumulative[-1])
        position = np.searchsorted(
            self._cumulative, reached * (1 - TIE_RELATIVE_TOLERANCE)
        )
        return float(self._values[position])
