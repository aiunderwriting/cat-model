"""Scalar numerical composition, probability-weight admission and the
platform-independent threshold decisions.

A decision that compares a value with a threshold, and so changes a
question, a prompt or a law, must not depend on how a platform rounds the
last bit of a computed float. Two rules cover every such decision:

- A stated or stored decimal (an answer's quantile, a value stored at four
  significant figures) is compared exactly as the decimal it was written as
  (``decimal_value``), so a ratio of exactly three is exactly three. A stated
  decimal moved by a stated step is their decimal sum
  (``catmodel.scenarios.sensitivity_coverage.shifted_tail_shape``), so a tail
  shape of 0.7 less 0.2 is exactly 0.5.
- A freshly computed value (an epistemic mean, a ratio of means) is compared
  with ``TIE_RELATIVE_TOLERANCE``: a value within it of the threshold is a
  tie, and each decision states which side a tie takes.
"""

import math
from collections.abc import Iterable
from decimal import Decimal
from fractions import Fraction

TIE_RELATIVE_TOLERANCE = 1e-9
"""Relative distance from a threshold within which a computed value is a tie.

Cross-platform differences in computed means and ratios are a few units in
the last place (relative ~1e-15); 1e-9 leaves a wide margin above that, and
is far below any difference a judgment or a threshold means to resolve. A
computed value within it of a threshold is not a statement on either side.
"""


def decimal_value(value: float) -> Fraction:
    """The exact decimal a finite float was written as: its shortest
    round-trip representation, which is the same on every platform."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"a decimal value must be finite, got {value!r}")
    return Fraction(repr(value))


def differs_by_more_than(left: float, right: float, ratio: float) -> bool:
    """Whether two nonnegative stated or stored decimals differ by more than
    ``ratio`` times, compared exactly as decimals (``decimal_value``).

    Equality of the ratio is not a difference; two zeros never differ and
    exactly one zero always does.
    """
    if left == right == 0:
        return False
    low, high = sorted((decimal_value(left), decimal_value(right)))
    return high > decimal_value(ratio) * low


def at_least(value: float, bound: float) -> bool:
    """Whether a computed nonnegative value is at least ``bound``, a value
    within ``TIE_RELATIVE_TOLERANCE`` below it counting as equal."""
    return value >= bound * (1 - TIE_RELATIVE_TOLERANCE)


def significant(value: float, digits: int = 4) -> float:
    """``value`` at ``digits`` significant figures, ties to even.

    The value is first written to twelve significant figures, which absorbs
    platform roundoff, so a computed value whose exact counterpart lies on a
    rounding midpoint rounds the same way on every platform.
    """
    return float(format(Decimal(f"{value:.12g}"), f".{digits}g"))


def valid_probability_weights(
    values: Iterable[object],
    *,
    rel_tol: float = 0,
    abs_tol: float = 1e-12,
    require_unit_sum: bool = True,
) -> bool:
    """Admit finite nonnegative mass, optionally requiring a unit total."""
    weights: list[float] = []
    try:
        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                return False
            weights.append(value)
        total = math.fsum(weights)
        return math.isfinite(total) and (
            not require_unit_sum
            or math.isclose(total, 1, rel_tol=rel_tol, abs_tol=abs_tol)
        )
    except OverflowError:
        return False


def scaled_product(*factors: float) -> float:
    """Multiply finite factors without intermediate underflow or overflow.

    Exponents accumulate separately, so only the final product can leave the
    binary64 range; it then rounds to zero or infinity for callers to handle
    in their own domain.
    """
    if any(value == 0 for value in factors):
        return 0.0
    mantissa, exponent = 1.0, 0
    for value in factors:
        part, power = math.frexp(value)
        mantissa, adjustment = math.frexp(mantissa * part)
        exponent += power + adjustment
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError:
        return math.copysign(math.inf, mantissa)


def product_at_least(left: float, right: float, threshold: float) -> bool:
    """Compare a finite product exactly, including zero and subnormal factors."""
    left_numerator, left_denominator = left.as_integer_ratio()
    right_numerator, right_denominator = right.as_integer_ratio()
    threshold_numerator, threshold_denominator = threshold.as_integer_ratio()
    return left_numerator * right_numerator * threshold_denominator >= (
        threshold_numerator * left_denominator * right_denominator
    )


def positive_product_ratio(
    factors, divisors=(), *, quantity: str, exponential=0.0
) -> float:
    """Compose positive factors before subnormal intermediates lose precision."""
    mantissa, exponent = 1.0, 0
    for values, divide in ((factors, False), (divisors, True)):
        for value in values:
            part, power = math.frexp(value)
            mantissa = mantissa / part if divide else mantissa * part
            mantissa, adjustment = math.frexp(mantissa)
            exponent += adjustment - power if divide else adjustment + power
    if exponential != 0:
        combined_log = math.log(mantissa) + exponent * math.log(2) + exponential
        if not math.isfinite(combined_log) or not -746 < combined_log < 710:
            raise ValueError(
                f"positive {quantity} is outside the numerical floating-point range"
            )
        pieces = max(1, math.ceil(abs(exponential) / 500))
        part, power = math.frexp(math.exp(exponential / pieces))
        for _ in range(pieces):
            mantissa, adjustment = math.frexp(mantissa * part)
            exponent += power + adjustment
    try:
        result = math.ldexp(mantissa, exponent)
    except OverflowError:
        result = math.inf
    if not math.isfinite(result) or result <= 0:
        raise ValueError(
            f"positive {quantity} is outside the numerical floating-point range"
        )
    return result
