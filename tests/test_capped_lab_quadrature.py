"""The capped episode lab mean lies inside independent monotone bounds.

``E[min(s*T + k + A, c)] = g(0) + integral over (0, L) of F_T((L - a)/s) S_A(a) da``
with ``L = c - k``. The integrand is a product of two non-increasing functions
of ``a``, so on any partition its integral lies between the right-endpoint and
left-endpoint Riemann sums. These tests build that bracket on their own fine
partition and check the reported mean and its certified error bound against it.
"""

import math

import numpy as np
import pytest

from catmodel.contracts.native_severity import (
    NativeSeverityComponent,
    NativeSeverityLaw,
    native_severity_from_sampler,
    native_severity_sampler,
)
from catmodel.probability.distributions import Distribution
from catmodel.simulation.incident_moments import (
    CERTIFIED_INTEGRATION_RELATIVE_ERROR,
    _capped_branch_mean,
)


def _bracket(other, total, scale, fixed, cap, points=400_000):
    """Monotone lower and upper bounds of the capped mean on a fine partition."""
    shared = native_severity_sampler(total)
    aggregate = native_severity_sampler(other)
    limit = cap - fixed
    knots = np.array(
        [
            value
            for law in (other,)
            for component in law.components
            for value in (
                component.p5,
                component.p25,
                component.p50,
                component.p75,
                component.p95,
            )
            if 0 < value < limit
        ]
        + [
            limit - scale * value
            for component in total.components
            for value in (
                component.p5,
                component.p25,
                component.p50,
                component.p75,
                component.p95,
            )
            if 0 < limit - scale * value < limit
        ]
    )
    edges = np.unique(
        np.concatenate(
            (
                [0.0, limit],
                np.linspace(0.0, limit, points // 4),
                np.geomspace(limit * 1e-15, limit, points),
                limit - np.geomspace(limit * 1e-15, limit, points // 4)[::-1],
                knots,
            )
        )
    )
    edges = edges[(edges >= 0) & (edges <= limit)]

    def integrand(amounts):
        return shared.cdf((limit - amounts) / scale) * (1.0 - aggregate.cdf(amounts))

    values = integrand(edges)
    widths = np.diff(edges)
    base = fixed + scale * shared.analytic_mean(severity_cap_override=limit / scale)
    lower = base + math.fsum(widths * values[1:])
    upper = base + math.fsum(widths * values[:-1])
    return lower, upper


def _positive_lab_law(quantiles, shape, cap, probability):
    """A business-loss law as contextual compilation builds it."""
    positive = NativeSeverityComponent(
        p5=quantiles[0],
        p25=quantiles[1],
        p50=quantiles[2],
        p75=quantiles[3],
        p95=quantiles[4],
        gpd_shape=shape,
        gpd_tail_ratio=quantiles[5] / quantiles[4],
        severity_minimum=0,
        severity_cap=cap,
    )
    if probability == 1:
        return NativeSeverityLaw(
            representation="distribution", components=(positive,), weights=(1.0,)
        )
    zero = NativeSeverityComponent(p5=0, p25=0, p50=0, p75=0, p95=0, severity_minimum=0)
    return NativeSeverityLaw(
        representation="mixture",
        components=(zero, positive),
        weights=(1 - probability, probability),
    )


def test_wide_business_loss_below_a_far_cap_is_not_understated():
    """A GPD-tailed business loss whose p95 is far below the cap.

    Fixed linear-scale nodes on the segment from p95 to the cap miss the mass
    just above p95; the bracket shows the true mean.
    """
    cap = 1e12
    other = _positive_lab_law((1e3, 1e4, 3e4, 1e5, 5e5, 2e6), 0.7, cap, 1.0)
    total = native_severity_from_sampler(
        Distribution(1e3, 1e4, 3e4, 1e5, 3e5, gpd_shape=0.5, gpd_tail_ratio=4)
    )
    lower, upper = _bracket(other, total, 1.0, 0.0, cap)
    assert (upper - lower) / lower < 1e-4
    result = _capped_branch_mean(
        other.model_dump_json(), total.model_dump_json(), 1.0, 0.0, cap
    )
    assert result is not None
    value, bound = result
    assert value - bound <= upper and value + bound >= lower
    assert 0 < bound <= CERTIFIED_INTEGRATION_RELATIVE_ERROR * value
    assert value > 0.999 * lower


@pytest.mark.parametrize("seed", range(12))
def test_capped_mean_lies_within_monotone_bounds_for_random_wide_heavy_laws(seed):
    rng = np.random.default_rng(seed)
    cap = float(10 ** rng.uniform(9, 13))
    # Business loss: p5 anywhere from 1e2 to 1e8, p5 to p95 spanning half a
    # decade to four, governed tail shape 0.35-0.95, sometimes a zero atom.
    start = float(10 ** rng.uniform(2, 8))
    ladder = np.cumsum(rng.uniform(0.2, 1.0, 6))
    ladder = ladder / ladder[4] * rng.uniform(1, 4)
    quantiles = tuple(start * 10 ** (ladder - ladder[0]))
    other = _positive_lab_law(
        quantiles,
        float(rng.uniform(0.35, 0.95)),
        cap,
        float(rng.choice([1.0, rng.uniform(0.05, 0.9)])),
    )
    total_start = float(10 ** rng.uniform(2, 10))
    total_ladder = np.cumsum(rng.uniform(0.2, 1.0, 5))
    total_ladder = total_ladder / total_ladder[4] * rng.uniform(1, 3)
    total = native_severity_from_sampler(
        Distribution(
            *(total_start * 10 ** (total_ladder - total_ladder[0])),
            gpd_shape=float(rng.uniform(0.2, 0.9)),
            gpd_tail_ratio=float(rng.uniform(2, 6)),
        )
    )
    scale = float(rng.uniform(0.05, 1.0))
    fixed = float(rng.choice([0.0, cap * rng.uniform(0, 0.5)]))
    lower, upper = _bracket(other, total, scale, fixed, cap)
    result = _capped_branch_mean(
        other.model_dump_json(), total.model_dump_json(), scale, fixed, cap
    )
    assert result is not None
    value, bound = result
    # Both the certified interval [value - bound, value + bound] and the
    # independent bracket contain the exact mean, so they overlap.
    assert value - bound <= upper * (1 + 1e-9)
    assert value + bound >= lower * (1 - 1e-9)
    assert bound <= CERTIFIED_INTEGRATION_RELATIVE_ERROR * value
    assert 0 < value <= cap
