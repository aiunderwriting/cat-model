"""The anchored tail of a severity law with a large-loss anchor.

The law without the anchor is the quantile-spline body through p5..p95 and a
GPD from p95 whose scale reproduces p99. With an anchor (A, pi) above that
law's own p99 b, the survival function is 0.01 ((x - m)/(b - m))^-alpha on
[b, A], a power law of the excess over the floor m through (b, 0.01) and
(A, pi), and pi S1(x)/S1(A) above A, the law without the anchor conditioned on
exceeding A and scaled to the anchor's mass. At and below p99 the law is
bit-identical to the law without the anchor; it is continuous in pi up to
the flat segment at pi = 0.01 and continuous where p99 reaches A with pi
reaching 0.01.
"""

import math
from itertools import pairwise

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import integrate

from catmodel.contracts.native_severity import (
    NativeSeverityComponent,
    NativeSeverityLaw,
    native_severity_from_sampler,
    native_severity_sampler,
)
from catmodel.probability.distributions import (
    LARGE_LOSS_BREAK_EXCEEDANCE,
    Distribution,
    MixtureSeverityDistribution,
    SeverityDomainError,
    SeverityMomentUnavailable,
    sample_affine_capped_severity,
)
from catmodel.setup.contextual_node import odds_scaled_large_loss_share
from catmodel.setup.sensitivity_catalog import LARGE_LOSS_ANCHOR_FACTORS
from catmodel.simulation.incident_moments import (
    CERTIFIED_INTEGRATION_RELATIVE_ERROR,
    _capped_branch_mean,
    _knots,
)

FLOOR = 1e5
P99 = 2e7
AMOUNT = 1e9
SHARE = 2e-4
P95 = 6e6
KNOTS = (1.2e5, 2.5e5, 6e5, 1.8e6, P95)
BOUND = LARGE_LOSS_BREAK_EXCEEDANCE


def _law(
    shape: float = 0.7,
    *,
    anchored: bool = True,
    share: float | None = SHARE,
    amount: float | None = AMOUNT,
    severity_cap: float | None = None,
    p99: float = P99,
) -> Distribution:
    return Distribution(
        *KNOTS,
        gpd_shape=shape,
        gpd_tail_ratio=p99 / P95,
        severity_minimum=FLOOR,
        severity_cap=severity_cap,
        large_loss_amount=amount if anchored else None,
        large_loss_exceedance=share if anchored else None,
    )


def _break(law: Distribution) -> float:
    value = law.large_loss_break
    assert value is not None
    return value


def _index(law: Distribution, segment: int = 0) -> float:
    """The power-law index of a segment of the anchored tail: the first runs
    from the p99 break to the anchor."""
    return law._large_loss_segments[segment].index


def _survival(law: Distribution, amount: float) -> float:
    return law.survival_probability(amount)


def _survival_integral(law: Distribution, upper: float | None = None) -> float:
    """Integral of S(x) over [0, upper] by adaptive quadrature in log amount,
    split at the knots, the p99 break and the anchor. Without an upper limit
    it runs to e^650 times the anchor, beyond which even shape 0.9 leaves
    under 1e-28 of the integral."""
    edges = sorted({FLOOR, *KNOTS, _break(law), AMOUNT})
    if upper is not None:
        edges = [edge for edge in edges if edge < upper] + [upper]

    def piece(lo: float, start: float, stop: float) -> float:
        def integrand(t: float) -> float:
            amount = lo * math.exp(t)
            return _survival(law, amount) * amount

        return integrate.quad(
            integrand, start, stop, limit=400, epsabs=0, epsrel=1e-12
        )[0]

    terms = [piece(lo, 0.0, math.log(hi / lo)) for lo, hi in pairwise(edges)]
    if upper is None:
        terms += [
            piece(edges[-1], start, stop)
            for start, stop in pairwise((0.0, 5.0, 25.0, 100.0, 300.0, 650.0))
        ]
    return FLOOR + math.fsum(terms)


SHAPES = (1e-9, 0.3, 0.5, 0.7, 0.9, 1.0)


@pytest.mark.parametrize("shape", SHAPES)
def test_law_at_and_below_p99_is_bit_identical_to_the_unanchored_law(shape):
    base, anchored = _law(shape, anchored=False), _law(shape)
    ranks = np.concatenate((np.linspace(0.0, 0.99, 20_001), [0.95, 0.99]))
    assert np.array_equal(
        base.sample_from_uniforms(ranks), anchored.sample_from_uniforms(ranks)
    )
    amounts = np.concatenate(
        (np.geomspace(FLOOR, _break(anchored), 5_001), [0.0, FLOOR])
    )
    assert np.array_equal(base.cdf(amounts), anchored.cdf(amounts))
    assert np.array_equal(base.survival(amounts), anchored.survival(amounts))
    # The body, its p95 slope and the first piece's scale are unchanged.
    assert base._body.upper_slope == anchored._body.upper_slope
    assert base.effective_gpd_scale == anchored.effective_gpd_scale
    for cap in (3e6, 1.5e7, _break(anchored)):
        assert base.analytic_mean(severity_cap_override=cap) == (
            anchored.analytic_mean(severity_cap_override=cap)
        )


@pytest.mark.parametrize("shape", SHAPES)
def test_anchor_is_reproduced_exactly(shape):
    law = _law(shape)
    assert law.large_loss_break == pytest.approx(P99, rel=1e-12)
    assert law.survival_probability(AMOUNT) == pytest.approx(SHARE, rel=1e-12)
    assert law.quantile(1 - SHARE) == pytest.approx(AMOUNT, rel=1e-9)
    # The distribution function is continuous across the break and the anchor.
    for edge in (_break(law), AMOUNT):
        below, above = law.cdf(np.array([np.nextafter(edge, 0.0), edge]))
        assert above - below < 1e-15
    assert law.cdf(np.array([_break(law)]))[0] == pytest.approx(0.99, abs=1e-15)


@pytest.mark.parametrize("shape", SHAPES)
def test_survival_is_a_power_law_to_the_anchor_then_the_conditional_tail(shape):
    law, base = _law(shape), _law(shape, anchored=False)
    edge = _break(law)
    alpha = math.log(BOUND / SHARE) / math.log((AMOUNT - FLOOR) / (edge - FLOOR))
    for amount in np.geomspace(edge * (1 + 1e-9), AMOUNT * (1 - 1e-9), 7):
        expected = BOUND * ((amount - FLOOR) / (edge - FLOOR)) ** -alpha
        assert _survival(law, amount) == pytest.approx(expected, rel=1e-11)
    # Beyond A the law is the unanchored law given X > A, with mass pi.
    for multiple in (1.5, 3.0, 10.0, 100.0):
        conditional = _survival(base, multiple * AMOUNT) / _survival(base, AMOUNT)
        if conditional == 0.0:
            assert _survival(law, multiple * AMOUNT) == 0.0
            continue
        assert _survival(law, multiple * AMOUNT) == pytest.approx(
            SHARE * conditional, rel=1e-9
        )
    # Its scale is the first piece's own scale at A.
    expected_scale = base.effective_gpd_scale + shape * (AMOUNT - P95)
    assert law._large_loss_tail_scale == pytest.approx(expected_scale, rel=1e-14)


@pytest.mark.parametrize(("shape", "one_in"), [(0.5, 10_000), (0.7, 720), (0.8, 316)])
def test_the_shape_alone_sets_how_many_anchor_level_incidents_go_a_hundredfold_higher(
    shape, one_in
):
    """docs/limitations.md, item 3. Beyond the anchor the law is the GPD of
    the governed shape conditioned on reaching it, so with p95 and p99 far
    below the anchor an incident of at least USD 1 billion reaches USD 100
    billion with probability about 100^(-1/shape), whatever the anchor's
    share."""
    scale = shape * (P99 - P95) / (5**shape - 1)
    expected = (1 + shape * (1e11 - AMOUNT) / (scale + shape * (AMOUNT - P95))) ** (
        -1 / shape
    )
    for share in (1e-6, SHARE, 5e-3):
        law = _law(shape, share=share)
        conditional = _survival(law, 1e11) / _survival(law, AMOUNT)
        assert conditional == pytest.approx(expected, rel=1e-9)
        assert 1 / conditional == pytest.approx(100 ** (1 / shape), rel=0.02)
        assert 1 / conditional == pytest.approx(one_in, rel=0.02)


@pytest.mark.parametrize("shape", (0.5, 0.7, 0.8))
def test_a_p99_nearer_the_anchor_raises_the_share_that_goes_a_hundredfold_higher(
    shape,
):
    """The GPD's scale at the anchor is s1 + xi (A - p95), which rises with
    the stated p99, and S(100 A)/S(A) rises with it: within three times
    100^(-1/xi), the far-below-anchor limit that docs/limitations.md (item 3)
    quotes, up to a p99 of USD 900 million on this leaf."""
    ratios = [
        _survival(law, 100 * AMOUNT) / _survival(law, AMOUNT)
        for law in (_law(shape, p99=p99) for p99 in (P99, 1e8, 3e8, 9e8))
    ]
    assert all(lower < higher for lower, higher in pairwise(ratios))
    assert ratios[0] == pytest.approx(100 ** (-1 / shape), rel=0.02)
    assert ratios[-1] < 3 * 100 ** (-1 / shape)


@pytest.mark.parametrize("shape", (1e-9, 0.5, 0.7, 0.9))
def test_uncapped_mean_is_the_survival_integral(shape):
    law = _law(shape)
    assert law.analytic_mean() == pytest.approx(_survival_integral(law), rel=1e-9)


@pytest.mark.parametrize("cap", (5e7, 6e8, AMOUNT, 2e9, 3e11))
@pytest.mark.parametrize("share", (1e-6, SHARE, 9.9e-3, BOUND))
def test_capped_mean_is_the_survival_integral(cap, share):
    law = _law(0.7, share=share)
    assert law.analytic_mean(severity_cap_override=cap) == pytest.approx(
        _survival_integral(law, cap), rel=1e-9
    )


@pytest.mark.parametrize("cap", (5e7, 2e9, 3e11))
def test_capped_mean_matches_the_quantile_integral(cap):
    law = _law(0.7)
    quantile = lambda u: min(float(law.quantile(u)), cap)  # noqa: E731
    points = sorted({0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1 - SHARE})
    pieces = [0.0, *points, 1.0]
    value = math.fsum(
        integrate.quad(quantile, lo, hi, limit=400, epsabs=0, epsrel=1e-11)[0]
        for lo, hi in pairwise(pieces)
    )
    assert law.analytic_mean(severity_cap_override=cap) == pytest.approx(
        value, rel=1e-8
    )


@pytest.mark.parametrize("share", (1e-7, SHARE, 5e-3, 9.99e-3, BOUND))
def test_quantile_and_distribution_function_invert_each_other(share):
    law = _law(0.7, share=share)
    edge = _break(law)
    ranks = np.concatenate(
        (
            0.99 + (BOUND - share) * np.array([1e-6, 0.3, 0.9, 1 - 1e-6]),
            1 - share * np.array([0.999, 0.5, 1e-3, 1e-9]),
        )
    )
    ranks = ranks[ranks > 0.99]
    amounts = law.sample_from_uniforms(ranks)
    assert np.all(np.diff(amounts) >= 0)
    np.testing.assert_allclose(law.survival(amounts), 1 - ranks, rtol=1e-9, atol=0)
    probes = [*np.geomspace(1.5 * AMOUNT, 1e3 * AMOUNT, 4)]
    if share < BOUND:
        probes += list(np.geomspace(edge * 1.01, AMOUNT * 0.99, 4))
    for amount in probes:
        survival = _survival(law, amount)
        if survival < 1e-8:
            # 1 - S keeps too few digits of S to read the amount back.
            continue
        assert law.quantile(1 - survival) == pytest.approx(amount, rel=1e-6)


def test_sampled_mean_matches_the_analytic_mean():
    law = _law(0.7, share=3e-3, severity_cap=5e10)
    rng = np.random.default_rng(20261104)
    draws = law.sample_from_uniforms(rng.uniform(size=4_000_000))
    error = draws.std() / math.sqrt(len(draws))
    assert abs(draws.mean() - law.analytic_mean()) < 5 * error
    assert float(np.mean(draws >= AMOUNT)) == pytest.approx(3e-3, rel=0.03)


SHARES = (1e-12, 1e-7, 1e-5, SHARE, 1e-3, 3e-3, 6e-3, 9e-3, 9.9e-3, 9.99e-3)


@pytest.mark.parametrize(("shape", "cap"), [(0.5, None), (0.7, None), (0.8, 1e14)])
def test_law_is_monotone_and_continuous_in_the_share(shape, cap):
    laws = [_law(shape, share=share, severity_cap=cap) for share in (*SHARES, BOUND)]
    means = [law.analytic_mean() for law in laws]
    assert all(lower < higher for lower, higher in pairwise(means))
    amounts = np.geomspace(1.01 * P99, 1e3 * AMOUNT, 50)
    survivals = [law.survival(amounts) for law in laws]
    assert all(np.all(low <= high) for low, high in pairwise(survivals))
    ranks = np.array([0.995, 0.999, 1 - 1e-5, 1 - 1e-8])
    quantiles = [law.sample_from_uniforms(ranks) for law in laws]
    assert all(np.all(low <= high) for low, high in pairwise(quantiles))
    # The law approaches its flat-segment limit as the share approaches 0.01.
    limit = means[-1]
    gaps = [
        abs(
            _law(shape, share=BOUND * (1 - step), severity_cap=cap).analytic_mean()
            - limit
        )
        / limit
        for step in (1e-2, 1e-4, 1e-6)
    ]
    assert gaps[0] < 2e-2 and gaps[1] < 2e-4 and gaps[2] < 2e-6


@pytest.mark.parametrize("share", (SHARE, 2e-3, 9.9e-3))
def test_odds_scaled_alternatives_bracket_the_law_below_its_flat_limit(share):
    """The large_loss_anchor alternatives multiply the conditional odds above
    p99 by 1/3 and 3. Their shares stay below 0.01, their laws bracket the
    law at every amount above p99 and lie below the flat-segment limit, and
    near 0.01, where they barely move the share, they barely move the mean."""
    lower, higher = (
        _law(share=odds_scaled_large_loss_share(share, factor))
        for factor in LARGE_LOSS_ANCHOR_FACTORS
    )
    base, flat = _law(share=share), _law(share=BOUND)
    for alternative in (lower, higher):
        exceedance = alternative.large_loss_exceedance
        assert exceedance is not None and 0 < exceedance < BOUND
    amounts = np.geomspace(1.01 * P99, 1e3 * AMOUNT, 40)
    survivals = [law.survival(amounts) for law in (lower, base, higher, flat)]
    assert all(np.all(low <= high) for low, high in pairwise(survivals))
    means = [law.analytic_mean() for law in (lower, base, higher, flat)]
    assert all(low < high for low, high in pairwise(means))
    if share > 9e-3:
        assert means[2] / means[1] - 1 < 0.01
        assert 1 - means[0] / means[1] < 0.03


@pytest.mark.parametrize("shape", (0.5, 0.7, 0.9))
def test_share_at_the_break_exceedance_gives_the_flat_segment_limit(shape):
    """At pi = 0.01 every incident above p99 is at or above A: the survival
    function is flat on [b, A) and the tail beyond A carries 0.01, so the mean
    is the law's limited mean at b plus 0.01 (A - b) plus 0.01 s_A/(1 - xi)."""
    law, base = _law(shape, share=BOUND), _law(shape, anchored=False)
    edge = _break(law)
    for amount in np.geomspace(edge * (1 + 1e-9), AMOUNT * (1 - 1e-9), 5):
        assert _survival(law, amount) == BOUND
    assert law.quantile(0.99) == pytest.approx(edge, rel=1e-15)
    assert law.quantile(math.nextafter(0.99, 1.0)) == pytest.approx(AMOUNT, rel=1e-9)
    scale = base.effective_gpd_scale + shape * (AMOUNT - P95)
    expected = (
        base.analytic_mean(severity_cap_override=edge)
        + BOUND * (AMOUNT - edge)
        + BOUND * scale / (1 - shape)
    )
    assert law.analytic_mean() == pytest.approx(expected, rel=1e-12)
    # A share within rounding of the bound gives the same law, not a blow-up.
    near = _law(shape, share=BOUND * (1 - 1e-15))
    assert near.analytic_mean() == pytest.approx(expected, rel=1e-12)


def test_a_coherent_share_below_the_bound_is_far_heavier_than_the_fallback():
    """An unresolved share of 0.01 or more compiles the p99-only tail: on this
    leaf (shape 0.7, uncapped) about a sixteenth of the mean of a coherent
    share just below 0.01."""
    below = _law(0.7, share=BOUND * (1 - 1e-9)).analytic_mean()
    fallback = _law(0.7, anchored=False).analytic_mean()
    assert below / fallback == pytest.approx(16.5, abs=0.1)


@pytest.mark.parametrize("share", (1e-4, 5e-3))
def test_mean_rises_with_p99_and_jumps_up_where_p99_passes_the_anchor(share):
    """At a fixed share below 0.01 the mean rises with p99. Once the law's p99
    reaches A, P(X >= A) is at least 0.01 > pi, so the anchor contradicts the
    quantiles and the law is the unanchored one: the mean jumps up there."""
    p99s = (2e7, 5e7, 1e8, 3e8, 5e8, 9e8, 9.9e8, 9.99e8, 0.99999 * AMOUNT)
    means = [_law(share=share, p99=p99).analytic_mean() for p99 in p99s]
    assert all(lower < higher for lower, higher in pairwise(means))
    beyond = [_law(anchored=False, p99=p99).analytic_mean() for p99 in (AMOUNT, 2e9)]
    assert means[-1] < beyond[0] < beyond[1]


def test_law_is_continuous_where_p99_reaches_the_anchor_with_the_share_at_the_bound():
    """Coherent answers can cross p99 = A only with pi reaching 0.01 there.
    The anchored law then approaches the unanchored law at p99 = A, so the
    law does not jump between the two sides."""
    reached = _law(anchored=False, p99=AMOUNT)
    ranks = np.array([0.9, 0.99, 0.995, 0.999, 1 - 1e-6])
    for step in (1e-3, 1e-6, 1e-9):
        for share in (BOUND, BOUND * (1 - step)):
            law = _law(share=share, p99=AMOUNT * (1 - step))
            assert law.analytic_mean() == pytest.approx(
                reached.analytic_mean(), rel=20 * step
            )
            np.testing.assert_allclose(
                law.sample_from_uniforms(ranks),
                reached.sample_from_uniforms(ranks),
                rtol=20 * step,
            )


@pytest.mark.parametrize("shape", (0.3, 0.5, 0.7, 0.9))
def test_anchor_on_the_unanchored_law_keeps_its_tail_beyond_the_anchor(shape):
    """With pi = S1(A) the law beyond A is the unanchored law; between p99 and
    A the power law replaces the GPD, which moves the mean by under 3%."""
    base = _law(shape, anchored=False)
    on_law = base.survival_probability(AMOUNT)
    law = _law(shape, share=on_law)
    ranks = 1 - on_law * np.array([0.5, 1e-3, 1e-6])
    np.testing.assert_allclose(
        law.sample_from_uniforms(ranks), base.sample_from_uniforms(ranks), rtol=1e-12
    )
    assert law.analytic_mean() == pytest.approx(base.analytic_mean(), rel=0.03)


@pytest.mark.parametrize("factor", (0.25, 2.0, 7.5))
@pytest.mark.parametrize("share", (SHARE, BOUND))
def test_excess_scale_maps_the_whole_tail_and_the_anchor(factor, share):
    law = _law(0.7, share=share)
    scaled = law.with_excess_scale(factor)
    mapped = FLOOR + factor * (AMOUNT - FLOOR)
    assert scaled.large_loss_amount == pytest.approx(mapped, rel=1e-15)
    assert scaled.large_loss_exceedance == share
    assert scaled.survival_probability(mapped) == pytest.approx(share, rel=1e-12)
    assert scaled.large_loss_break == pytest.approx(
        FLOOR + factor * (_break(law) - FLOOR), rel=1e-12
    )
    assert _index(scaled) == pytest.approx(_index(law), rel=1e-9)
    assert scaled._large_loss_tail_scale == pytest.approx(
        factor * law._large_loss_tail_scale, rel=1e-12
    )
    ranks = np.array([0.1, 0.5, 0.96, 0.99, 0.995, 1 - share, 1 - 1e-7])
    assert np.allclose(
        scaled.sample_from_uniforms(ranks),
        FLOOR + factor * (law.sample_from_uniforms(ranks) - FLOOR),
        rtol=1e-11,
    )
    assert scaled.analytic_mean() == pytest.approx(
        FLOOR + factor * (law.analytic_mean() - FLOOR), rel=1e-11
    )


def test_currency_scale_maps_the_anchor():
    law = _law(0.7)
    scaled = law.with_currency_scale(0.9, 1.1)
    assert scaled.large_loss_amount == pytest.approx(AMOUNT * 0.99, rel=1e-15)
    assert scaled.survival_probability(AMOUNT * 0.99) == pytest.approx(SHARE, rel=1e-12)
    assert scaled.analytic_mean() == pytest.approx(
        0.99 * law.analytic_mean(), rel=1e-11
    )


@pytest.mark.parametrize("shape", (0.3, 0.5, 0.9, 1.0))
def test_shape_alternatives_re_derive_both_pins(shape):
    alternative = _law(0.7).with_gpd_shape(shape)
    assert alternative.large_loss_break == pytest.approx(P99, rel=1e-12)
    assert alternative.survival_probability(AMOUNT) == pytest.approx(SHARE, rel=1e-12)
    assert alternative.survival_probability(P99) == pytest.approx(0.01, rel=1e-12)
    # The power law between p99 and A does not depend on the shape; the tail
    # beyond A takes the alternative's shape and its first piece's scale at A.
    for amount in (5e7, 3e8, 9e8):
        assert alternative.survival_probability(amount) == pytest.approx(
            _law(0.7).survival_probability(amount), rel=1e-11
        )
    assert alternative._large_loss_tail_scale == pytest.approx(
        _law(shape, anchored=False).effective_gpd_scale + shape * (AMOUNT - P95),
        rel=1e-14,
    )


def test_break_is_the_laws_own_p99_with_or_without_an_anchor():
    base = _law(0.7, anchored=False)
    assert base.large_loss_break == base.quantile(0.99)
    assert _law(0.7).large_loss_break == base.large_loss_break
    assert Distribution(*KNOTS).large_loss_break is None
    # An explicit scale puts the computed p99 away from any stated one; the
    # anchor must clear the computed value.
    explicit = Distribution(
        *KNOTS, gpd_shape=0.7, gpd_scale=4e8, severity_minimum=FLOOR
    )
    computed = _break(explicit)
    assert computed > AMOUNT > P99
    with pytest.raises(ValueError, match="must exceed the law's p99"):
        Distribution(
            *KNOTS,
            gpd_shape=0.7,
            gpd_scale=4e8,
            severity_minimum=FLOOR,
            large_loss_amount=AMOUNT,
            large_loss_exceedance=SHARE,
        )


def test_a_heavy_shape_and_a_tiny_share_spread_the_mass_above_p99():
    """Shape 0.9, share 1e-6 and p99 USD 400 million: the power law between
    p99 and A has index alpha = ln(1e4)/ln(2.5), about 10, so half of the
    incidents above p99 lie below 2^(1/alpha) p99, about 7% above it, and
    about 1.5% within 0.1% of the gap to A. The shape does not enter."""
    law = Distribution(
        1e6,
        5e6,
        2e7,
        5e7,
        1e8,
        gpd_shape=0.9,
        gpd_tail_ratio=4.0,
        large_loss_amount=AMOUNT,
        large_loss_exceedance=1e-6,
    )
    edge = _break(law)
    alpha = math.log(1e4) / math.log(AMOUNT / edge)
    assert _index(law) == pytest.approx(alpha, rel=1e-12)
    assert law.quantile(1 - 0.005) == pytest.approx(edge * 2 ** (1 / alpha), rel=1e-9)
    assert law.quantile(1 - 0.005) / edge == pytest.approx(1.0714, abs=1e-4)
    within = 1 - law.survival_probability(edge + 1e-3 * (AMOUNT - edge)) / BOUND
    assert within == pytest.approx(0.015, abs=1e-3)


def test_caps_apply_last():
    law = _law(0.7, severity_cap=5e9)
    assert law.survival_probability(5e9) == 0.0
    assert law.survival_probability(AMOUNT) == pytest.approx(SHARE, rel=1e-12)
    assert law.quantile(1.0) == 5e9
    assert law.analytic_mean() == pytest.approx(
        _law(0.7).analytic_mean(severity_cap_override=5e9), rel=1e-15
    )
    # A cap between p99 and A censors the power law and leaves nothing beyond.
    between = _law(0.7, severity_cap=3e8)
    assert between.survival_probability(AMOUNT) == 0.0
    assert between.quantile(1 - SHARE) == 3e8
    assert between.analytic_mean() == pytest.approx(
        _law(0.7).analytic_mean(severity_cap_override=3e8), rel=1e-15
    )
    assert between.analytic_mean() < law.analytic_mean()
    # A cap below the break leaves the anchor without effect.
    low = _law(0.7, severity_cap=1e7)
    assert (
        low.analytic_mean()
        == _law(0.7, anchored=False, severity_cap=1e7).analytic_mean()
    )


def test_uncapped_mean_needs_a_shape_below_one():
    with pytest.raises(SeverityMomentUnavailable):
        _law(1.0).analytic_mean()
    assert math.isfinite(_law(1.0, severity_cap=1e14).analytic_mean())


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ((AMOUNT, 0.0), "positive and at most"),
        ((AMOUNT, -1e-3), "positive and at most"),
        ((AMOUNT, math.nextafter(BOUND, 1.0)), "positive and at most"),
        ((AMOUNT, math.nan), "positive and at most"),
        ((1.5e7, SHARE), "must exceed the law's p99"),
        ((None, SHARE), "both its amount and its exceedance"),
    ],
)
def test_invalid_anchors_are_refused(changes, message):
    amount, share = changes
    with pytest.raises(ValueError, match=message):
        _law(0.7, amount=amount, share=share)


def test_anchor_needs_an_elicited_gpd_tail_and_its_support():
    with pytest.raises(ValueError, match="elicited law's GPD tail"):
        Distribution(*KNOTS, large_loss_amount=AMOUNT, large_loss_exceedance=SHARE)
    with pytest.raises(SeverityDomainError):
        _law(0.7, amount=1e101)
    # A negative shape ends the first piece near USD 31 million.
    with pytest.raises(ValueError, match="inside the support"):
        _law(-0.5)
    inside = _law(-0.5, amount=2.5e7)
    assert inside.survival_probability(2.5e7) == pytest.approx(SHARE, rel=1e-12)
    assert inside.quantile(1.0) == pytest.approx(
        _law(-0.5, anchored=False).quantile(1.0)
    )


def test_native_law_round_trip_and_schema_version():
    law = _law(0.7, severity_cap=1e14)
    native = native_severity_from_sampler(law)
    assert native.schema_version == 1
    component = native.components[0]
    assert component.large_loss_amount == AMOUNT
    assert component.large_loss_exceedance == SHARE
    restored = native_severity_sampler(native)
    assert restored.analytic_mean() == law.analytic_mean()
    flat = native_severity_sampler(native_severity_from_sampler(_law(share=BOUND)))
    assert flat.analytic_mean() == _law(share=BOUND).analytic_mean()
    payload = native.model_dump(mode="json")
    payload["schema_version"] = 2
    with pytest.raises(ValidationError):
        NativeSeverityLaw.model_validate(payload)
    with pytest.raises(ValidationError):
        NativeSeverityComponent(
            p5=KNOTS[0],
            p25=KNOTS[1],
            p50=KNOTS[2],
            p75=KNOTS[3],
            p95=P95,
            large_loss_amount=AMOUNT,
        )


def test_sampling_and_affine_capped_evaluation_use_the_whole_tail():
    law = _law(0.7)
    ranks = np.linspace(0.98, 1 - 1e-9, 2_001)
    direct = law.sample_from_uniforms(ranks)
    assert np.all(np.diff(direct) > 0)
    capped = sample_affine_capped_severity(law, ranks, cap=5e9)
    assert np.array_equal(capped, np.minimum(direct, 5e9))
    rng = np.random.default_rng(20261030)
    draws = law.sample_from_uniforms(rng.uniform(size=2_000_000))
    share = float(np.mean(draws >= AMOUNT))
    assert abs(share - SHARE) < 4 * math.sqrt(SHARE / 2_000_000)


def test_mixture_mean_weights_an_anchored_component():
    atom = Distribution(
        p5=FLOOR, p25=FLOOR, p50=FLOOR, p75=FLOOR, p95=FLOOR, severity_minimum=FLOOR
    )
    for share in (SHARE, BOUND):
        law = _law(0.7, share=share)
        mixture = MixtureSeverityDistribution(
            components=[atom, law], weights=np.array([0.4, 0.6])
        )
        assert mixture.analytic_mean() == pytest.approx(
            0.4 * FLOOR + 0.6 * law.analytic_mean()
        )
        assert mixture.survival_probability(AMOUNT) == pytest.approx(0.6 * share)
        ranks = 0.4 + 0.6 * np.array([0.5, 0.995, 1 - share / 2, 1 - 1e-7])
        np.testing.assert_allclose(
            mixture.sample_from_uniforms(ranks),
            law.sample_from_uniforms((ranks - 0.4) / 0.6),
            rtol=1e-8,
        )


def test_knots_include_the_break_and_the_anchor():
    native = native_severity_from_sampler(_law(0.7))
    knots = _knots(native)
    assert AMOUNT in knots
    assert any(math.isclose(value, P99, rel_tol=1e-12) for value in knots)
    unanchored = _knots(native_severity_from_sampler(_law(0.7, anchored=False)))
    assert knots - unanchored == {AMOUNT, _law(0.7).large_loss_break}


@pytest.mark.parametrize("share", (SHARE, BOUND))
def test_certified_capped_lab_mean_brackets_an_anchored_total(share):
    anchored = _law(0.7, share=share)
    total = native_severity_from_sampler(anchored)
    other = native_severity_from_sampler(
        Distribution(
            1e4, 1e5, 3e5, 1e6, 5e6, gpd_shape=0.7, gpd_tail_ratio=4, severity_cap=1e12
        )
    )
    scale, fixed, cap = 0.3, 0.0, 1e12
    shared, aggregate = native_severity_sampler(total), native_severity_sampler(other)
    limit = cap - fixed
    edges = np.unique(
        np.concatenate(
            (
                [0.0, limit],
                np.geomspace(limit * 1e-15, limit, 400_000),
                limit - np.geomspace(limit * 1e-15, limit, 100_000)[::-1],
                # Where the anchored total's distribution function kinks: its
                # floor, knots, p99 break and anchor amount.
                [
                    limit - scale * value
                    for value in (FLOOR, *KNOTS, _break(anchored), AMOUNT)
                ],
            )
        )
    )
    edges = edges[(edges >= 0) & (edges <= limit)]
    values = shared.cdf((limit - edges) / scale) * (1.0 - aggregate.cdf(edges))
    widths = np.diff(edges)
    base = fixed + scale * shared.analytic_mean(severity_cap_override=limit / scale)
    lower = base + math.fsum(widths * values[1:])
    upper = base + math.fsum(widths * values[:-1])
    result = _capped_branch_mean(
        other.model_dump_json(), total.model_dump_json(), scale, fixed, cap
    )
    assert result is not None
    value, bound = result
    assert value - bound <= upper * (1 + 1e-9)
    assert value + bound >= lower * (1 - 1e-9)
    assert bound <= CERTIFIED_INTEGRATION_RELATIVE_ERROR * value
