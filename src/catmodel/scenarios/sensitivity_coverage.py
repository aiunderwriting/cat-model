"""Tail-shape bounds, the required tail-shape sensitivity step and the shape
it gives.

This module sits with the scenario schema so that the schema and the
severity laws can share it without an import cycle.
"""

from fractions import Fraction
from typing import Final

# Tail-shape bounds shared by severity-governance validation and the severity
# laws. An uncapped GPD mean requires shape < 1; 0.95 leaves headroom because
# Monte Carlo recovery of the analytic mean is poor near divergence. The same
# magnitude bounds shapes from below.
UNCAPPED_TAIL_SHAPE_BOUND: Final = 0.95
# With a finite cap, E[min(X, cap)] is finite for every shape, so this bound
# guards against fit pathologies and authoring errors rather than limiting the
# model. A shape of 2.5 corresponds to a Pareto tail index of 0.4.
CAPPED_TAIL_SHAPE_SANITY_BOUND: Final = 2.5
# Required tail-shape sensitivity step. A governed shape moves by this step with
# the GPD scale re-derived from the unchanged p95/p99 anchor. It is an
# INITIAL_ESTIMATE judgment sized to
# the analog evidence for the tail classes: it spans the 0.37-0.63 threshold spread
# of the NPDB payment fit and takes the cyber/operational class to 0.9, the
# heaviest shape an uncapped finite-mean law represents with margin. Published
# operational-risk and cyber estimates above that are outside the range, so it
# is a stress, not an uncertainty interval for the shape.
TAIL_SHAPE_SENSITIVITY_STEP: Final = 0.2


def shifted_tail_shape(shape: float, step: float) -> float:
    """``shape`` moved by ``step``, as the decimal the two stated decimals sum to.

    A governed shape and the step are stated decimals, and decisions compare
    a shifted shape with stated thresholds: 0.5, from which an uncapped law
    has infinite variance; 1, from which it has no mean; and the operating
    bounds above. Binary addition can miss the decimal sum by a unit in the
    last place: ``0.7 - 0.2`` is 0.49999999999999994, a finite-variance shape,
    where the alternative meant is 0.5. So the sum is taken exactly on the
    decimals the two floats were written as (their shortest round-trip forms,
    as ``catmodel.probability.numerics.decimal_value`` reads them) and rounded
    once.

    Every site that applies the step uses this function. A law's shape is
    then always a stated decimal, and its thresholds are compared exactly,
    without a tie tolerance.
    """
    return float(Fraction(repr(float(shape))) + Fraction(repr(float(step))))
