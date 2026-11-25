"""The chain count law's recomposition and its log-quantile pool with the direct law."""

from __future__ import annotations

import math

import numpy as np
import pytest

from catmodel.contracts.elicitation import QUANTILES, Quantiles
from catmodel.elicitation.chain_law import (
    CHAIN_POOL_WEIGHT,
    OPPORTUNITY_RATE_SPREAD_LIMIT,
    TERMINAL_STAGE_FLOOR,
    central_values,
    chain_count_law,
    chain_count_values,
    guardrail_violations,
    pooled_count,
)

RATE = Quantiles(p5=0.2, p25=1.0, p50=3.0, p75=9.0, p95=60.0)
STAGE = Quantiles(p5=0.01, p25=0.05, p50=0.1, p75=0.2, p95=0.6)
TERMINAL = Quantiles(p5=0.05, p25=0.2, p50=0.4, p75=0.6, p95=0.9)


def _law(*values: float) -> Quantiles:
    return Quantiles(**dict(zip(QUANTILES, values, strict=True)))


def _values(law: Quantiles) -> np.ndarray:
    return np.asarray([getattr(law, key) for key in QUANTILES])


def _brute_force(base: float, laws: list[Quantiles], draws: int) -> np.ndarray:
    """Independent pseudo-random draws of the same capped components."""
    rng = np.random.default_rng(7)
    product = np.full(draws, base)
    for index, law in enumerate(laws):
        values = central_values(law, rng.random(draws))
        product *= values if index == 0 else np.minimum(values, 1.0)
    return np.quantile(product, [0.05, 0.25, 0.5, 0.75, 0.95])


def test_recomposition_matches_brute_force_on_known_component_laws():
    laws = [RATE, STAGE, TERMINAL]
    law = chain_count_law(600.0, laws[0], laws[1:])
    expected = _brute_force(600.0, laws, 2_000_000)
    np.testing.assert_allclose(_values(law), expected, rtol=0.03)


def test_constant_components_multiply_exactly():
    law = chain_count_law(
        1600.0, _law(*[0.5] * 5), [_law(*[0.25] * 5), _law(*[0.1] * 5)]
    )
    assert _values(law) == pytest.approx([20.0] * 5, rel=1e-9)


def test_one_component_reads_its_own_central_law():
    """With every stage certain, the chain law is the base count times the
    rate law, with the rate's own p5 and p95 as its end points."""
    certain = _law(*[1.0] * 5)
    law = chain_count_law(10.0, RATE, [certain, certain])
    assert _values(law) == pytest.approx(10 * _values(RATE), rel=2e-3)


def test_each_component_is_held_at_its_own_p5_and_p95():
    """No design point leaves the box the components' p5 and p95 span: no
    extrapolated tail enters the product."""
    values = chain_count_values(600.0, RATE, [STAGE, TERMINAL])
    low = 600.0 * RATE.p5 * STAGE.p5 * TERMINAL.p5
    high = 600.0 * RATE.p95 * STAGE.p95 * TERMINAL.p95
    assert values.min() >= low * (1 - 1e-12)
    assert values.max() <= high * (1 + 1e-12)
    # The extremes are reached: 5% of each component's rank sits at its edge.
    assert values.max() == pytest.approx(high, rel=1e-9)


def test_stage_probabilities_are_capped_at_one():
    above = _law(0.5, 0.8, 1.0, 1.0, 1.0)
    values = chain_count_values(10.0, _law(*[1.0] * 5), [above])
    assert values.max() <= 10.0 * (1 + 1e-12)


def test_a_zero_component_gives_a_zero_count_where_it_is_zero():
    stage = _law(0.0, 0.0, 0.1, 0.2, 0.3)
    law = chain_count_law(100.0, RATE, [stage])
    # A quarter of the stage's rank is exactly zero, so the count's p5 is.
    assert law.p5 == 0
    assert 0 <= law.p25 < 1e-3 * law.p50


def test_recomposition_is_deterministic():
    first = chain_count_law(600.0, RATE, [STAGE, TERMINAL])
    second = chain_count_law(600.0, RATE, [STAGE, TERMINAL])
    assert first == second


def test_median_of_product_is_near_the_product_of_medians():
    """Log-space recomposition leaves the median near the product of the
    component medians; a mean-based composition would not."""
    law = chain_count_law(600.0, RATE, [STAGE, TERMINAL])
    product = 600.0 * RATE.p50 * STAGE.p50 * TERMINAL.p50
    assert 0.5 < law.p50 / product < 2


def test_pool_is_the_log_average_at_every_quantile():
    direct = _law(1, 2, 4, 8, 16)
    chain = _law(4, 8, 16, 32, 64)
    pooled = pooled_count(direct, chain)
    assert CHAIN_POOL_WEIGHT == 0.5
    assert _values(pooled) == pytest.approx([2, 4, 8, 16, 32], rel=1e-9)


def test_pool_is_not_a_linear_pool_or_a_mean_average():
    direct = _law(0.5, 0.8, 1, 1.2, 2)
    chain = _law(50, 80, 100, 120, 200)
    pooled = pooled_count(direct, chain)
    assert pooled.p50 == pytest.approx(10, rel=1e-9)
    # A linear pool of the two laws would put its median between them far
    # from 10; an average of quantiles in value space would give 50.5.
    assert not math.isclose(pooled.p50, 50.5, rel_tol=0.1)


def test_pool_reads_only_quantiles_not_means():
    """The pool reads the five quantiles, not a mean: changing the direct
    law's p95 moves only the pooled p95."""
    direct = _law(1, 2, 4, 8, 16)
    wider = _law(1, 2, 4, 8, 1600)
    chain = _law(4, 8, 16, 32, 64)
    narrow, wide = pooled_count(direct, chain), pooled_count(wider, chain)
    assert (narrow.p5, narrow.p25, narrow.p50, narrow.p75) == (
        wide.p5,
        wide.p25,
        wide.p50,
        wide.p75,
    )
    assert wide.p95 == pytest.approx(math.sqrt(1600 * 64), rel=1e-3)


def test_pool_weights_zero_and_one_are_the_arms_exactly():
    direct = _law(1, 2, 4, 8, 16)
    chain = _law(4, 8, 16, 32, 64)
    assert pooled_count(direct, chain, 0.0) == direct
    assert pooled_count(direct, chain, 1.0) == chain
    assert pooled_count(direct, None, 1.0) == direct


def test_a_single_source_arm_reports_its_source_s_zero():
    """The pool sets a wholly zero law aside and uses the other. An arm that
    names one source reports that source, so a zero chain is a chain-only
    count of zero, not the direct count."""
    direct = _law(1, 2, 4, 8, 16)
    chain = _law(4, 8, 16, 32, 64)
    zero = _law(0, 0, 0, 0, 0)
    assert pooled_count(direct, zero) == direct
    assert pooled_count(direct, zero, 1.0) == zero
    assert pooled_count(zero, chain) == chain
    assert pooled_count(zero, chain, 0.0) == zero
    assert pooled_count(zero, zero, 1.0) == zero
    # No chain is not a zero chain: the chain-only arm keeps the direct law.
    assert pooled_count(direct, None, 1.0) == direct


@pytest.mark.parametrize("weight", [0.25, 0.5, 0.75])
def test_pool_is_monotone_in_each_input_and_between_them(weight):
    direct = _law(1, 2, 4, 8, 16)
    chain = _law(4, 8, 16, 32, 64)
    higher = _law(5, 9, 20, 40, 90)
    pooled = _values(pooled_count(direct, chain, weight))
    assert np.all(np.diff(pooled) >= 0)
    assert np.all(pooled >= _values(direct)) and np.all(pooled <= _values(chain))
    assert np.all(_values(pooled_count(direct, higher, weight)) >= pooled)
    lower = _values(pooled_count(direct, chain, weight / 2))
    assert np.all(lower <= pooled)


def test_pool_edge_cases():
    direct = _law(1, 2, 4, 8, 16)
    chain = _law(4, 8, 16, 32, 64)
    zero = _law(0, 0, 0, 0, 0)
    # A zero direct answer uses the chain alone; an unavailable or wholly
    # zero chain uses the direct alone; two zeros pool to zero.
    assert pooled_count(zero, chain) == chain
    assert pooled_count(direct, None) == direct
    assert pooled_count(direct, zero) == direct
    assert pooled_count(zero, zero) == zero
    assert pooled_count(zero, None) == zero
    # A zero quantile in one law pools to zero at that probability only.
    partial = pooled_count(_law(0, 0, 4, 8, 16), chain)
    assert (partial.p5, partial.p25) == (0, 0)
    assert partial.p50 == pytest.approx(8, rel=1e-9)


def test_guardrail_flags_a_wide_rate_or_a_negligible_terminal_stage():
    assert guardrail_violations(RATE, TERMINAL) == ()
    wide = _law(0.5, 1, 3, 9, 0.5 * OPPORTUNITY_RATE_SPREAD_LIMIT * 1.01)
    assert len(guardrail_violations(wide, TERMINAL)) == 1
    edge = _law(0.5, 1, 3, 9, 0.5 * OPPORTUNITY_RATE_SPREAD_LIMIT * 0.99)
    assert guardrail_violations(edge, TERMINAL) == ()
    assert len(guardrail_violations(_law(0, 1, 2, 3, 4), TERMINAL)) == 1
    assert guardrail_violations(_law(0, 0, 0, 0, 0), TERMINAL) == ()
    small = _law(0, 1e-4, TERMINAL_STAGE_FLOOR / 2, 0.01, 0.1)
    assert len(guardrail_violations(RATE, small)) == 1
    assert len(guardrail_violations(wide, small)) == 2
