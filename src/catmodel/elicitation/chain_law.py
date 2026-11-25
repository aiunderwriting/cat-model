"""The chain count law and its fixed pool with the direct count.

A frequency chain's count is the stipulated base count times the elicited
opportunity rate per base unit times every stage's conditional probability
(``FREQ-EVENT-CHAIN``). ``chain_count_law`` recomposes it deterministically in
log space over a fixed quasi-random design, reading each component's quantile
law only between its own p5 and p95 (``FREQ-CHAIN-INDEPENDENCE``), and returns
the product's five quantiles. ``pooled_count`` averages the direct and chain
quantile laws in log space at each of the five probabilities
(``FREQ-CHAIN-POOL``). Chain and direct are compared on medians and central
intervals, never on means: a mean of a product of wide judged laws is driven
by extrapolated tails that no respondent stated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from scipy.stats import qmc

from catmodel.contracts.elicitation import QUANTILES, Quantiles
from catmodel.elicitation.epistemic import epistemic_quantiles
from catmodel.probability.numerics import significant

CHAIN_DESIGN_LOG2_POINTS = 14
"""The chain design holds 2**14 = 16,384 points of a scrambled Sobol sequence,
one dimension per component; quantiles of the product are read from them."""

CHAIN_DESIGN_SEED = 20_261_103
"""The fixed seed of the chain design's scrambling: every replay of a chain
answer reads the same points."""

CENTRAL_LOWER = 0.05
CENTRAL_UPPER = 0.95
"""Each component law is read between its own p5 and p95 and held at them
beyond: the lowest and highest 5% of each component's rank sit at its p5 and
p95, so no extrapolated tail enters the product."""

CHAIN_POOL_WEIGHT = 0.5
"""INITIAL_ESTIMATE weight of the chain law in the log-quantile pool
(``FREQ-CHAIN-POOL``); the direct law takes the rest."""

CHAIN_COUNT_SIGNIFICANT_FIGURES = 4
"""Significant figures a case stores of its chain count law."""

OPPORTUNITY_RATE_SPREAD_LIMIT = 1000.0
"""The opportunity rate's p95/p5 above which a chain answer gets one
validation retry (``FREQ-CHAIN-GUARDRAIL``)."""

TERMINAL_STAGE_FLOOR = 1e-3
"""The terminal stage's p50 below which a chain answer gets one validation
retry (``FREQ-CHAIN-GUARDRAIL``)."""

_PROBABILITIES = np.asarray([0.05, 0.25, 0.5, 0.75, 0.95])


@lru_cache(maxsize=8)
def chain_design(dimensions: int) -> np.ndarray:
    """The fixed design: ranks in (0, 1), one column per component."""
    if dimensions < 1:
        raise ValueError("a chain design needs at least one dimension")
    sampler = qmc.Sobol(d=dimensions, scramble=True, seed=CHAIN_DESIGN_SEED)
    points = sampler.random_base2(m=CHAIN_DESIGN_LOG2_POINTS)
    points.setflags(write=False)
    return points


def central_values(law: Quantiles, ranks: np.ndarray) -> np.ndarray:
    """``law``'s quantile function at ``ranks``, held at its p5 and p95 below
    and above them: the law is read only between its own p5 and p95."""
    clipped = np.clip(ranks, CENTRAL_LOWER, CENTRAL_UPPER)
    return epistemic_quantiles(law, clipped)


def _quantiles(values: Sequence[float] | np.ndarray) -> Quantiles:
    ordered = np.maximum.accumulate(np.asarray(values, dtype=float))
    return Quantiles(
        **{
            key: significant(float(value), CHAIN_COUNT_SIGNIFICANT_FIGURES)
            for key, value in zip(QUANTILES, ordered, strict=True)
        }
    )


def chain_count_values(
    base_count: float, rate: Quantiles, stages: Sequence[Quantiles]
) -> np.ndarray:
    """The chain count at every design point, from its log-space sum.

    Each stage probability is capped at one. The logs are added component by
    component in chain order, an elementwise sum whose order does not depend
    on the platform's vectorized reduction. A point where any component is
    zero has a zero count.
    """
    if not (math.isfinite(base_count) and base_count > 0):
        raise ValueError("a chain's stipulated base count must be positive and finite")
    design = chain_design(1 + len(stages))
    columns = [central_values(rate, design[:, 0])]
    columns.extend(
        np.minimum(central_values(stage, design[:, index]), 1.0)
        for index, stage in enumerate(stages, start=1)
    )
    zero = np.zeros(design.shape[0], dtype=bool)
    total = np.full(design.shape[0], math.log(base_count))
    for column in columns:
        positive = column > 0
        zero |= ~positive
        total = total + np.log(np.where(positive, column, 1.0))
    return np.where(zero, 0.0, np.exp(total))


def chain_count_law(
    base_count: float, rate: Quantiles, stages: Sequence[Quantiles]
) -> Quantiles:
    """The chain count's p5, p25, p50, p75 and p95 over the fixed design, at
    ``CHAIN_COUNT_SIGNIFICANT_FIGURES``; linear interpolation between order
    statistics."""
    values = np.sort(chain_count_values(base_count, rate, stages))
    return _quantiles(np.quantile(values, _PROBABILITIES, method="linear"))


def _log_average(direct: float, chain: float, weight: float) -> float:
    if direct == 0 or chain == 0:
        return 0.0
    return math.exp((1 - weight) * math.log(direct) + weight * math.log(chain))


def is_zero_law(law: Quantiles) -> bool:
    """Whether every quantile of ``law`` is zero: a wholly zero estimate."""
    return law.p95 == 0


def pooled_count(
    direct: Quantiles,
    chain: Quantiles | None,
    weight: float = CHAIN_POOL_WEIGHT,
) -> Quantiles:
    """The final count law: the weighted average, in log space, of the direct
    and chain quantile laws at p5, p25, p50, p75 and p95 (Vincentization).

    At each probability the pooled quantile is
    ``exp((1 - weight) ln direct + weight ln chain)``; a probability where
    either is zero pools to zero, so the result stays ordered. An unavailable
    chain, or a wholly zero direct or chain law, is not pooled: the other is
    used alone, and two wholly zero laws pool to zero.

    ``weight`` 0 is the direct law and 1 the chain law, exactly: the
    exhibit's direct-only and chain-only arms. Each reports the estimate it
    names, a wholly zero one included, and only a missing chain leaves the
    chain-only arm on the direct law.
    """
    if not 0 <= weight <= 1:
        raise ValueError("the chain pool weight lies in [0, 1]")
    if chain is None or weight == 0:
        return direct
    if weight == 1:
        return chain
    if is_zero_law(direct) and is_zero_law(chain):
        return direct
    if is_zero_law(direct):
        return chain
    if is_zero_law(chain):
        return direct
    values = [
        _log_average(getattr(direct, key), getattr(chain, key), weight)
        for key in QUANTILES
    ]
    return _quantiles(values)


def rate_spread_tripped(rate: Quantiles) -> bool:
    """Whether the opportunity rate's p95/p5 exceeds
    ``OPPORTUNITY_RATE_SPREAD_LIMIT``; a zero p5 under a positive p95 is
    unbounded."""
    return rate.p95 > 0 and (
        rate.p5 == 0 or rate.p95 > OPPORTUNITY_RATE_SPREAD_LIMIT * rate.p5
    )


def terminal_stage_tripped(terminal: Quantiles) -> bool:
    """Whether the terminal stage's p50 is below ``TERMINAL_STAGE_FLOOR``."""
    return terminal.p50 < TERMINAL_STAGE_FLOOR


def guardrail_violations(rate: Quantiles, terminal: Quantiles) -> tuple[str, ...]:
    """The chain guardrail's findings on one available chain answer: a very
    wide opportunity rate (``rate_spread_tripped``) or a rare terminal stage
    (``terminal_stage_tripped``). Either earns one retry that asks the
    respondent to confirm or revise the answer, neutral about any change."""
    findings = []
    if rate_spread_tripped(rate):
        findings.append(
            f"opportunity_rate p95/p5 exceeds {OPPORTUNITY_RATE_SPREAD_LIMIT:,.0f}"
        )
    if terminal_stage_tripped(terminal):
        findings.append(f"the terminal stage's p50 is below {TERMINAL_STAGE_FLOOR:g}")
    return tuple(findings)
