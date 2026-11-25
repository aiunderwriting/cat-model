"""The two-anchor tail of a severity law: a second anchor above the first.

A law with a large-loss anchor (A, pi) is a power law of the excess over the
floor m from its own p99 b to A and, above A, its GPD tail conditioned on
reaching A (``tests/test_large_loss_anchor_law.py``). A second anchor
(A2, pi2), with A2 > A and 0 < pi2 <= pi, replaces the tail above A: the
survival function is pi ((x - m)/(A - m))^-alpha2 on [A, A2], a second power
law through (A, pi) and (A2, pi2), and pi2 S1(x)/S1(A2) above A2, the law
without anchors conditioned on reaching A2. At and below A the law is the
one-anchor law bit for bit, it is continuous and rising in both shares, and
given X > A2 it is the one-anchor law's own conditional law, so the second
anchor moves how many incidents reach A2 and not how losses grow beyond it.
"""

import math
from itertools import pairwise

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import integrate

from catmodel.contracts.native_severity import (
    NativeSeverityComponent,
    native_severity_from_sampler,
    native_severity_sampler,
)
from catmodel.probability.distributions import (
    LARGE_LOSS_BREAK_EXCEEDANCE,
    Distribution,
    MixtureSeverityDistribution,
    SeverityMomentUnavailable,
    sample_affine_capped_severity,
)
from catmodel.simulation.incident_moments import (
    CERTIFIED_INTEGRATION_RELATIVE_ERROR,
    _capped_branch_mean,
    _knots,
)

FLOOR = 1e5
P95 = 6e6
P99 = 2e7
AMOUNT = 1e9
HIGHER = 1e11
SHARE = 2e-4
SECOND = 4e-7
KNOTS = (1.2e5, 2.5e5, 6e5, 1.8e6, P95)
BOUND = LARGE_LOSS_BREAK_EXCEEDANCE
# The conditional share at which the second segment's index is one: beyond it
# an unbounded power law of that index would have no mean.
UNIT_INDEX_RATIO = (AMOUNT - FLOOR) / (HIGHER - FLOOR)
# The three tail classes: the governed shape and the outer cap.
TAIL_CLASSES = ((0.5, None), (0.7, None), (0.8, 1e14))


def _law(
    shape: float = 0.7,
    *,
    share: float = SHARE,
    second: float | None = SECOND,
    severity_cap: float | None = None,
    p99: float = P99,
    amount: float = AMOUNT,
    higher: float | None = HIGHER,
) -> Distribution:
    """The hand-built leaf with one anchor (``second`` None) or two."""
    return Distribution(
        *KNOTS,
        gpd_shape=shape,
        gpd_tail_ratio=p99 / P95,
        severity_minimum=FLOOR,
        severity_cap=severity_cap,
        large_loss_amount=amount,
        large_loss_exceedance=share,
        very_large_loss_amount=None if second is None else higher,
        very_large_loss_exceedance=second,
    )


def _unanchored(shape: float = 0.7) -> Distribution:
    return Distribution(
        *KNOTS, gpd_shape=shape, gpd_tail_ratio=P99 / P95, severity_minimum=FLOOR
    )


def _break(law: Distribution) -> float:
    value = law.large_loss_break
    assert value is not None
    return value


def _survival(law: Distribution, amount: float) -> float:
    return law.survival_probability(amount)


def _gpd_survival(shape: float, scale: float, excess: float) -> float:
    """The survival function of a GPD at an excess over its threshold."""
    if shape < 1e-8:
        return math.exp(-excess / scale)
    return (1 + shape * excess / scale) ** (-1 / shape)


def _survival_integral(
    law: Distribution, upper: float | None = None, lower: float = 0.0
) -> float:
    """Integral of S(x) over [lower, upper] by adaptive quadrature in log
    amount, split at the knots, the p99 break and both anchors. Without an
    upper limit it runs to e^650 times the second anchor, beyond which even
    shape 0.9 leaves under 1e-28 of the integral."""
    edges = sorted({FLOOR, *KNOTS, _break(law), AMOUNT, HIGHER} | {lower or FLOOR})
    edges = [edge for edge in edges if edge >= max(lower, FLOOR)]
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
    return (FLOOR if lower == 0.0 else 0.0) + math.fsum(terms)


SHAPES = (1e-9, 0.3, 0.5, 0.7, 0.9, 1.0)
# Conditional shares pi2/pi: light, at the unit index, heavy and nearly flat.
RATIOS = (1e-6, 1e-3, UNIT_INDEX_RATIO, 0.1, 0.9)


@pytest.mark.parametrize("shape", SHAPES)
def test_law_at_and_below_the_first_anchor_is_bit_identical_to_the_one_anchor_law(
    shape,
):
    one, two = _law(shape, second=None), _law(shape)
    # Every rank below the first anchor's; at that rank itself each law reads
    # the anchor amount through its own piece.
    ranks = np.concatenate((np.linspace(0.0, 1 - SHARE, 20_001)[:-1], [0.95, 0.99]))
    assert np.array_equal(
        one.sample_from_uniforms(ranks), two.sample_from_uniforms(ranks)
    )
    assert one.quantile(1 - SHARE) == pytest.approx(two.quantile(1 - SHARE), rel=1e-9)
    amounts = np.concatenate((np.geomspace(FLOOR, AMOUNT, 5_001), [0.0, FLOOR]))
    amounts = amounts[amounts < AMOUNT]
    assert np.array_equal(one.cdf(amounts), two.cdf(amounts))
    assert np.array_equal(one.survival(amounts), two.survival(amounts))
    assert one.large_loss_break == two.large_loss_break
    for cap in (3e6, 1.5e7, _break(two), 3e8):
        assert one.analytic_mean(severity_cap_override=cap) == (
            two.analytic_mean(severity_cap_override=cap)
        )
    assert one.analytic_mean(severity_cap_override=AMOUNT) == pytest.approx(
        two.analytic_mean(severity_cap_override=AMOUNT), rel=1e-15
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_both_anchors_are_reproduced_exactly(shape):
    law = _law(shape)
    assert law.survival_probability(AMOUNT) == pytest.approx(SHARE, rel=1e-12)
    assert law.survival_probability(HIGHER) == pytest.approx(SECOND, rel=1e-12)
    assert law.quantile(1 - SHARE) == pytest.approx(AMOUNT, rel=1e-9)
    assert law.quantile(1 - SECOND) == pytest.approx(HIGHER, rel=1e-9)
    # The distribution function is continuous across the break and both anchors.
    for edge in (_break(law), AMOUNT, HIGHER):
        below, above = law.cdf(np.array([np.nextafter(edge, 0.0), edge]))
        assert above - below < 1e-15


@pytest.mark.parametrize("shape", SHAPES)
def test_survival_is_two_power_laws_then_the_conditional_tail(shape):
    law, base = _law(shape), _unanchored(shape)
    edge = _break(law)
    first = math.log(BOUND / SHARE) / math.log((AMOUNT - FLOOR) / (edge - FLOOR))
    for amount in np.geomspace(edge * (1 + 1e-9), AMOUNT * (1 - 1e-9), 5):
        expected = BOUND * ((amount - FLOOR) / (edge - FLOOR)) ** -first
        assert _survival(law, amount) == pytest.approx(expected, rel=1e-11)
    second = math.log(SHARE / SECOND) / math.log((HIGHER - FLOOR) / (AMOUNT - FLOOR))
    for amount in np.geomspace(AMOUNT * (1 + 1e-9), HIGHER * (1 - 1e-9), 7):
        expected = SHARE * ((amount - FLOOR) / (AMOUNT - FLOOR)) ** -second
        assert _survival(law, amount) == pytest.approx(expected, rel=1e-11)
    # Beyond A2 the law is the unanchored law given X > A2, with mass pi2: a
    # GPD of the same shape whose scale is the first piece's own scale at A2.
    scale = base.effective_gpd_scale + shape * (HIGHER - P95)
    assert law._large_loss_tail_scale == pytest.approx(scale, rel=1e-14)
    for multiple in (1.0001, 1.5, 3.0, 10.0, 100.0):
        conditional = _gpd_survival(shape, scale, (multiple - 1) * HIGHER)
        assert _survival(law, multiple * HIGHER) == pytest.approx(
            SECOND * conditional, rel=1e-9, abs=1e-300
        )
        own = _survival(base, HIGHER)
        if own > 0:
            assert _survival(base, multiple * HIGHER) / own == pytest.approx(
                conditional, rel=1e-9
            )


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize(("shape", "cap"), TAIL_CLASSES)
def test_given_the_second_anchor_the_law_is_the_one_anchor_laws(shape, cap, ratio):
    """Given X > A2 the two-anchor law and the one-anchor law are the same
    GPD: its scale at A2 does not depend on which anchors lie below. So the
    second anchor multiplies the expected loss beyond A2 by pi2 over the
    one-anchor law's own S(A2), at every shape and cap, and changes nothing
    else there."""
    one = _law(shape, second=None, severity_cap=cap)
    two = _law(shape, second=ratio * SHARE, severity_cap=cap)
    own = _survival(one, HIGHER)
    for multiple in (1.0001, 2.0, 30.0, 900.0):
        assert _survival(two, multiple * HIGHER) / (ratio * SHARE) == pytest.approx(
            _survival(one, multiple * HIGHER) / own, rel=1e-9
        )
    beyond = [
        _survival_integral(law, cap, lower=HIGHER) + HIGHER * _survival(law, HIGHER)
        for law in (one, two)
    ]
    assert beyond[1] / beyond[0] == pytest.approx(ratio * SHARE / own, rel=1e-8)


@pytest.mark.parametrize(
    ("shape", "cap", "per_incident"),
    [
        (0.5, None, 2.0e11),
        (0.7, None, 3.33e11),
        (0.8, 1e14, 4.29e11),
        # The tail-shape and cap stresses of the three classes.
        (0.3, None, 1.43e11),
        (0.9, None, 1.0e12),
        (0.6, 1e14, 2.49e11),
        (1.0, 1e14, 7.91e11),
        (0.8, 1e13, 3.74e11),
        (0.8, 5e14, 4.52e11),
        (0.8, None, 5.0e11),
    ],
)
def test_mean_loss_of_an_incident_that_reaches_the_second_anchor(
    shape, cap, per_incident
):
    """E[min(X, C) | X >= A2] is A2 plus the limited mean excess of a GPD of
    the governed shape and scale s1 + xi (A2 - p95), about xi A2 here: USD
    200 billion, 333 billion and 429 billion per incident in the three tail
    classes, whatever either share is."""
    base = _unanchored(shape)
    scale = base.effective_gpd_scale + shape * (HIGHER - P95)
    if cap is None:
        excess = scale / (1 - shape)
    else:
        reach = 1 + shape * (cap - HIGHER) / scale
        excess = (
            scale * math.log(reach)
            if shape == 1.0
            else scale / (1 - shape) * (1 - reach ** (1 - 1 / shape))
        )
    assert HIGHER + excess == pytest.approx(per_incident, rel=0.005)
    for share, second in ((SHARE, SECOND), (5e-3, 2e-3), (1e-6, 1e-9)):
        law = _law(shape, share=share, second=second, severity_cap=cap)
        beyond = _survival_integral(law, cap, lower=HIGHER) + HIGHER * second
        assert beyond / second == pytest.approx(HIGHER + excess, rel=1e-8)


@pytest.mark.parametrize(
    ("shape", "ratio"),
    [
        (1e-9, 1e-3),
        *((shape, ratio) for shape in (0.5, 0.7, 0.9) for ratio in RATIOS),
    ],
)
def test_uncapped_mean_is_the_survival_integral(shape, ratio):
    law = _law(shape, second=ratio * SHARE)
    assert law.analytic_mean() == pytest.approx(_survival_integral(law), rel=1e-9)


@pytest.mark.parametrize("cap", (6e8, AMOUNT, 4e10, HIGHER, 3e11, 1e14))
@pytest.mark.parametrize("ratio", (1e-3, UNIT_INDEX_RATIO, 0.1, 1.0))
def test_capped_mean_is_the_survival_integral(cap, ratio):
    law = _law(0.7, second=ratio * SHARE)
    assert law.analytic_mean(severity_cap_override=cap) == pytest.approx(
        _survival_integral(law, cap), rel=1e-9
    )


@pytest.mark.parametrize("cap", (5e7, 2e9, 4e10, 3e11))
def test_capped_mean_matches_the_quantile_integral(cap):
    law = _law(0.7)
    quantile = lambda u: min(float(law.quantile(u)), cap)  # noqa: E731
    points = sorted({0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1 - SHARE, 1 - SECOND})
    pieces = [0.0, *points, 1.0]
    value = math.fsum(
        integrate.quad(quantile, lo, hi, limit=400, epsabs=0, epsrel=1e-11)[0]
        for lo, hi in pairwise(pieces)
    )
    assert law.analytic_mean(severity_cap_override=cap) == pytest.approx(
        value, rel=1e-8
    )


@pytest.mark.parametrize("ratio", (*RATIOS, 1.0))
@pytest.mark.parametrize("share", (1e-6, SHARE, 9.99e-3, BOUND))
def test_quantile_and_distribution_function_invert_each_other(share, ratio):
    second = ratio * share
    law = _law(0.7, share=share, second=second)
    edge = _break(law)
    ranks = np.concatenate(
        (
            0.99 + (BOUND - share) * np.array([1e-6, 0.3, 0.9, 1 - 1e-6]),
            1 - share + (share - second) * np.array([1e-6, 0.2, 0.7, 1 - 1e-6]),
            1 - second * np.array([0.999, 0.5, 1e-3, 1e-9]),
        )
    )
    ranks = np.unique(ranks[(ranks > 0.99) & (ranks < 1)])
    amounts = law.sample_from_uniforms(ranks)
    assert np.all(np.diff(amounts) >= 0)
    np.testing.assert_allclose(law.survival(amounts), 1 - ranks, rtol=1e-9, atol=0)
    probes = [*np.geomspace(1.5 * HIGHER, 1e3 * HIGHER, 4)]
    if ratio < 1:
        probes += list(np.geomspace(AMOUNT * 1.01, HIGHER * 0.99, 5))
    if share < BOUND:
        probes += list(np.geomspace(edge * 1.01, AMOUNT * 0.99, 4))
    for amount in probes:
        survival = _survival(law, amount)
        if survival < 1e-8:
            # 1 - S keeps too few digits of S to read the amount back.
            continue
        assert law.quantile(1 - survival) == pytest.approx(amount, rel=1e-6)


@pytest.mark.stochastic
def test_sampled_mean_and_anchor_shares_match_the_law():
    """Four million inverse-transform draws of a capped law, whose variance
    is finite: the sample mean lies within five standard errors of the
    closed-form mean, and the shares at or above each anchor within five
    binomial standard errors of the anchors' exceedances."""
    share, second = 3e-3, 4e-4
    law = _law(0.7, share=share, second=second, severity_cap=5e12)
    rng = np.random.default_rng(20261111)
    draws = law.sample_from_uniforms(rng.uniform(size=4_000_000))
    error = draws.std() / math.sqrt(len(draws))
    assert abs(draws.mean() - law.analytic_mean()) < 5 * error
    for amount, probability in ((AMOUNT, share), (HIGHER, second)):
        observed = float(np.mean(draws >= amount))
        deviation = math.sqrt(probability * (1 - probability) / len(draws))
        assert abs(observed - probability) < 5 * deviation
    # Between the anchors the draws follow the second power law: the share
    # at or above the geometric midpoint of the excess is sqrt(pi pi2).
    midpoint = FLOOR + math.sqrt((AMOUNT - FLOOR) * (HIGHER - FLOOR))
    expected = math.sqrt(share * second)
    observed = float(np.mean(draws >= midpoint))
    assert abs(observed - expected) < 5 * math.sqrt(expected / len(draws))


SECOND_SHARES = (1e-15, 1e-10, 1e-8, SECOND, 2e-6, 2e-5, 1e-4, 1.9e-4, 1.999e-4)


@pytest.mark.parametrize(("shape", "cap"), TAIL_CLASSES)
def test_law_is_monotone_and_continuous_in_the_second_share(shape, cap):
    laws = [
        _law(shape, second=second, severity_cap=cap)
        for second in (*SECOND_SHARES, SHARE)
    ]
    means = [law.analytic_mean() for law in laws]
    assert all(lower < higher for lower, higher in pairwise(means))
    amounts = np.geomspace(1.01 * AMOUNT, 1e3 * HIGHER, 60)
    survivals = [law.survival(amounts) for law in laws]
    assert all(np.all(low <= high) for low, high in pairwise(survivals))
    ranks = 1 - SHARE * np.array([0.9, 0.3, 1e-2, 1e-5])
    quantiles = [law.sample_from_uniforms(ranks) for law in laws]
    assert all(np.all(low <= high) for low, high in pairwise(quantiles))
    # The law approaches its flat-segment limit as pi2 approaches pi.
    limit = means[-1]
    gaps = [
        abs(
            _law(shape, second=SHARE * (1 - step), severity_cap=cap).analytic_mean()
            - limit
        )
        / limit
        for step in (1e-2, 1e-4, 1e-6)
    ]
    assert gaps[0] < 2e-2 and gaps[1] < 2e-4 and gaps[2] < 2e-6


@pytest.mark.parametrize(("shape", "cap"), TAIL_CLASSES)
def test_law_is_monotone_and_continuous_in_the_first_share(shape, cap):
    """At a fixed second share the law rises with the first share at every
    amount between the break and A2 and does not move beyond A2, up to the
    first segment's flat limit at 0.01."""
    shares = (4.1e-7, 1e-6, 1e-5, SHARE, 3e-3, 9e-3, 9.99e-3, BOUND)
    laws = [_law(shape, share=share, severity_cap=cap) for share in shares]
    means = [law.analytic_mean() for law in laws]
    assert all(lower < higher for lower, higher in pairwise(means))
    between = np.geomspace(1.01 * P99, 0.99 * HIGHER, 60)
    survivals = [law.survival(between) for law in laws]
    assert all(np.all(low <= high) for low, high in pairwise(survivals))
    beyond = np.geomspace(HIGHER, 0.5 * (cap or 1e16), 20)
    for law in laws[1:]:
        np.testing.assert_allclose(
            law.survival(beyond), laws[0].survival(beyond), rtol=1e-11
        )
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


@pytest.mark.parametrize(("shape", "cap"), TAIL_CLASSES)
def test_second_share_equal_to_the_first_gives_the_flat_segment_limit(shape, cap):
    """At pi2 = pi every incident that reaches A reaches A2: the survival
    function is flat on [A, A2) and the tail beyond A2 carries pi, so the
    mean is the limited mean at A plus pi (A2 - A) plus pi times the limited
    mean excess beyond A2."""
    law = _law(shape, second=SHARE, severity_cap=cap)
    one = _law(shape, second=None, severity_cap=cap)
    for amount in np.geomspace(AMOUNT, HIGHER * (1 - 1e-9), 5):
        assert _survival(law, amount) == pytest.approx(SHARE, rel=1e-15)
    assert law.quantile(math.nextafter(1 - SHARE, 1.0)) == pytest.approx(
        HIGHER, rel=1e-6
    )
    excess = _survival_integral(law, cap, lower=HIGHER)
    expected = (
        one.analytic_mean(severity_cap_override=AMOUNT)
        + SHARE * (HIGHER - AMOUNT)
        + excess
    )
    assert law.analytic_mean() == pytest.approx(expected, rel=1e-9)
    # A share within rounding of the first gives the same law, not a blow-up.
    near = _law(shape, second=SHARE * (1 - 1e-15), severity_cap=cap)
    assert near.analytic_mean() == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(("shape", "cap"), TAIL_CLASSES)
def test_a_vanishing_second_share_piles_the_first_anchors_mass_at_its_amount(
    shape, cap
):
    """As pi2 falls to zero the second power law steepens without limit: the
    incidents that reach A stay at A, and the expected loss from them falls
    to pi A, below the one-anchor law's pi (A + s_A/(1 - xi)). An exact zero
    is no second anchor, the one-anchor law, so the law is not continuous at
    zero: the convention the first anchor has at its own zero."""
    one = _law(shape, second=None, severity_cap=cap)
    below = one.analytic_mean(severity_cap_override=AMOUNT)
    own = one.analytic_mean() - below + AMOUNT * SHARE
    assert own > 1.9 * SHARE * AMOUNT
    previous = math.inf
    for second in (1e-20, 1e-60, 1e-200):
        law = _law(shape, second=second, severity_cap=cap)
        above = law.analytic_mean() - below + AMOUNT * SHARE
        assert SHARE * AMOUNT < above < previous
        previous = above
    assert previous == pytest.approx(SHARE * AMOUNT, rel=0.02)


@pytest.mark.parametrize(("shape", "cap"), TAIL_CLASSES)
def test_second_share_on_the_one_anchor_law_keeps_its_tail_beyond_the_second_anchor(
    shape, cap
):
    """With pi2 = the one-anchor law's own S(A2), the two-anchor law is that
    law beyond A2; between the anchors a power law of the excess over the
    floor replaces a GPD that is nearly one, which moves the mean by under
    0.1%."""
    one = _law(shape, second=None, severity_cap=cap)
    own = one.survival_probability(HIGHER)
    law = _law(shape, second=own, severity_cap=cap)
    ranks = 1 - own * np.array([0.5, 1e-3, 1e-6])
    np.testing.assert_allclose(
        law.sample_from_uniforms(ranks), one.sample_from_uniforms(ranks), rtol=1e-9
    )
    assert law.analytic_mean() == pytest.approx(one.analytic_mean(), rel=1e-3)


@pytest.mark.parametrize(("shape", "cap"), TAIL_CLASSES)
@pytest.mark.parametrize("ratio", (UNIT_INDEX_RATIO, 0.03, 0.3))
def test_a_segment_too_heavy_for_an_unbounded_mean_has_a_finite_mean(shape, cap, ratio):
    """A stated pi2/pi of (A - m)/(A2 - m), about 1 in 100, or more gives the
    second power law an index of at most one. An unbounded power law of that
    index has no mean; this one ends at A2, so its part of the mean is
    pi (A - m) (R^(1 - alpha2) - 1)/(1 - alpha2), with R = (A2 - m)/(A - m),
    or pi (A - m) ln R at alpha2 = 1. The mean is finite exactly when the
    one-anchor law's is: the tail beyond A2 is the same GPD."""
    second = ratio * SHARE
    law = _law(shape, second=second, severity_cap=cap)
    span = (HIGHER - FLOOR) / (AMOUNT - FLOOR)
    index = math.log(SHARE / second) / math.log(span)
    assert index <= 1 + 1e-12
    segment = SHARE * (AMOUNT - FLOOR)
    segment *= (
        math.log(span)
        if ratio == UNIT_INDEX_RATIO
        else (span ** (1 - index) - 1) / (1 - index)
    )
    assert _survival_integral(law, HIGHER, lower=AMOUNT) == pytest.approx(
        segment, rel=1e-9
    )
    one = _law(shape, second=None, severity_cap=cap)
    expected = (
        one.analytic_mean(severity_cap_override=AMOUNT)
        + segment
        + _survival_integral(law, cap, lower=HIGHER)
    )
    assert law.analytic_mean() == pytest.approx(expected, rel=1e-9)
    assert math.isfinite(law.analytic_mean())
    # Without a cap the mean needs a shape below one, as the one-anchor law's.
    with pytest.raises(SeverityMomentUnavailable):
        _law(1.0, second=second).analytic_mean()


@pytest.mark.parametrize(
    ("shape", "cap", "at_unit_index", "at_one_in_ten"),
    [(0.5, None, 0.30, 0.69), (0.7, None, 0.42, 0.79), (0.8, 1e14, 0.48, 0.83)],
)
def test_a_heavy_second_segment_hands_the_far_tail_to_the_governed_shape(
    shape, cap, at_unit_index, at_one_in_ten
):
    """The share of the expected loss from incidents of at least A that lies
    beyond A2, where the shape and the cap set each incident's size: about
    30%, 42% and 48% in the three tail classes at a stated 1 in 100, where
    alpha2 = 1, and 69%, 79% and 83% at 1 in 10. It does not depend on the
    first share."""
    for ratio, expected in ((UNIT_INDEX_RATIO, at_unit_index), (0.1, at_one_in_ten)):
        shares = []
        for share in (SHARE, 5e-3):
            law = _law(shape, share=share, second=ratio * share, severity_cap=cap)
            beyond = _survival_integral(law, cap, lower=HIGHER) + HIGHER * (
                ratio * share
            )
            above = _survival_integral(law, cap, lower=AMOUNT) + AMOUNT * share
            shares.append(beyond / above)
        assert shares[0] == pytest.approx(shares[1], rel=1e-6)
        assert shares[0] == pytest.approx(expected, abs=0.006)


@pytest.mark.parametrize("p99", (2e7, 3e8, 9.9e8))
def test_law_is_continuous_where_p99_reaches_the_first_anchor(p99):
    """As the law's p99 rises to A with pi reaching 0.01, the first segment
    vanishes and the two-anchor law approaches the one-anchor law whose only
    anchor is (A2, pi2) above a p99 at A: the law a coherent answer with p99
    at or above A compiles."""
    reached = Distribution(
        *KNOTS,
        gpd_shape=0.7,
        gpd_tail_ratio=AMOUNT / P95,
        severity_minimum=FLOOR,
        large_loss_amount=HIGHER,
        large_loss_exceedance=SECOND,
    )
    ranks = np.array([0.9, 0.99, 0.995, 0.999, 1 - 1e-6, 1 - SECOND / 3])
    for step in (1e-3, 1e-6, 1e-9):
        law = _law(share=BOUND * (1 - step), p99=AMOUNT * (1 - step))
        assert law.analytic_mean() == pytest.approx(
            reached.analytic_mean(), rel=20 * step
        )
        np.testing.assert_allclose(
            law.sample_from_uniforms(ranks),
            reached.sample_from_uniforms(ranks),
            rtol=20 * step,
        )
    # Away from that limit the two laws differ.
    assert _law(p99=p99).analytic_mean() < reached.analytic_mean()


@pytest.mark.parametrize("factor", (0.25, 2.0, 7.5))
def test_excess_scale_maps_both_anchors_and_the_whole_tail(factor):
    law = _law(0.7)
    scaled = law.with_excess_scale(factor)
    for amount, share in ((AMOUNT, SHARE), (HIGHER, SECOND)):
        mapped = FLOOR + factor * (amount - FLOOR)
        assert scaled.survival_probability(mapped) == pytest.approx(share, rel=1e-12)
    assert scaled.very_large_loss_amount == pytest.approx(
        FLOOR + factor * (HIGHER - FLOOR), rel=1e-15
    )
    assert scaled.very_large_loss_exceedance == SECOND
    assert scaled._large_loss_tail_scale == pytest.approx(
        factor * law._large_loss_tail_scale, rel=1e-12
    )
    ranks = np.array([0.5, 0.995, 1 - SHARE, 1 - 3e-5, 1 - SECOND, 1 - 1e-8])
    assert np.allclose(
        scaled.sample_from_uniforms(ranks),
        FLOOR + factor * (law.sample_from_uniforms(ranks) - FLOOR),
        rtol=1e-11,
    )
    assert scaled.analytic_mean() == pytest.approx(
        FLOOR + factor * (law.analytic_mean() - FLOOR), rel=1e-11
    )


def test_currency_scale_maps_both_anchors():
    law = _law(0.7)
    scaled = law.with_currency_scale(0.9, 1.1)
    assert scaled.very_large_loss_amount == pytest.approx(HIGHER * 0.99, rel=1e-15)
    assert scaled.survival_probability(HIGHER * 0.99) == pytest.approx(
        SECOND, rel=1e-12
    )
    assert scaled.analytic_mean() == pytest.approx(
        0.99 * law.analytic_mean(), rel=1e-11
    )


@pytest.mark.parametrize("shape", (0.3, 0.5, 0.9, 1.0))
def test_shape_alternatives_keep_both_anchors_and_move_the_tail_beyond_the_second(
    shape,
):
    alternative = _law(0.7).with_gpd_shape(shape)
    assert alternative.survival_probability(AMOUNT) == pytest.approx(SHARE, rel=1e-12)
    assert alternative.survival_probability(HIGHER) == pytest.approx(SECOND, rel=1e-12)
    # Neither power law depends on the shape; the tail beyond A2 takes the
    # alternative's shape and its first piece's scale at A2.
    for amount in (5e7, 3e8, 5e9, 6e10):
        assert alternative.survival_probability(amount) == pytest.approx(
            _law(0.7).survival_probability(amount), rel=1e-11
        )
    assert alternative._large_loss_tail_scale == pytest.approx(
        _unanchored(shape).effective_gpd_scale + shape * (HIGHER - P95), rel=1e-14
    )


def test_caps_apply_last():
    law = _law(0.7, severity_cap=5e12)
    assert law.survival_probability(5e12) == 0.0
    assert law.survival_probability(HIGHER) == pytest.approx(SECOND, rel=1e-12)
    assert law.quantile(1.0) == 5e12
    assert law.analytic_mean() == pytest.approx(
        _law(0.7).analytic_mean(severity_cap_override=5e12), rel=1e-15
    )
    # A cap between the anchors censors the second power law and leaves
    # nothing beyond.
    between = _law(0.7, severity_cap=3e10)
    assert between.survival_probability(HIGHER) == 0.0
    assert between.survival_probability(AMOUNT) == pytest.approx(SHARE, rel=1e-12)
    assert between.quantile(1 - SECOND) == 3e10
    assert between.analytic_mean() == pytest.approx(
        _law(0.7).analytic_mean(severity_cap_override=3e10), rel=1e-15
    )
    # A cap at or below the first anchor leaves the second without effect.
    low = _law(0.7, severity_cap=AMOUNT)
    assert low.analytic_mean() == pytest.approx(
        _law(0.7, second=None, severity_cap=AMOUNT).analytic_mean(), rel=1e-15
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"second": 0.0}, "positive and at most the first anchor's"),
        ({"second": -1e-9}, "positive and at most the first anchor's"),
        ({"second": math.nextafter(SHARE, 1.0)}, "at most the first anchor's"),
        ({"second": math.nan}, "positive and at most the first anchor's"),
        ({"higher": AMOUNT}, "must exceed the first anchor amount"),
        ({"higher": 5e8}, "must exceed the first anchor amount"),
    ],
)
def test_invalid_second_anchors_are_refused(changes, message):
    with pytest.raises(ValueError, match=message):
        _law(0.7, **changes)


def test_second_anchor_needs_both_its_fields_and_the_first_anchor():
    with pytest.raises(ValueError, match="both its amount and its exceedance"):
        Distribution(
            *KNOTS,
            gpd_shape=0.7,
            gpd_tail_ratio=P99 / P95,
            severity_minimum=FLOOR,
            large_loss_amount=AMOUNT,
            large_loss_exceedance=SHARE,
            very_large_loss_amount=HIGHER,
        )
    with pytest.raises(ValueError, match="needs the first anchor"):
        Distribution(
            *KNOTS,
            gpd_shape=0.7,
            gpd_tail_ratio=P99 / P95,
            severity_minimum=FLOOR,
            very_large_loss_amount=HIGHER,
            very_large_loss_exceedance=SECOND,
        )
    # A negative shape ends the first piece near USD 31 million, below A2.
    with pytest.raises(ValueError, match="inside the support"):
        _law(-0.5, amount=2.5e7, higher=4e7)
    inside = _law(-0.5, amount=2.5e7, higher=2.9e7)
    assert inside.survival_probability(2.9e7) == pytest.approx(SECOND, rel=1e-12)


def test_native_law_round_trip_keeps_the_second_anchor():
    law = _law(0.7, severity_cap=1e14)
    native = native_severity_from_sampler(law)
    component = native.components[0]
    assert component.very_large_loss_amount == HIGHER
    assert component.very_large_loss_exceedance == SECOND
    restored = native_severity_sampler(native)
    assert restored.analytic_mean() == law.analytic_mean()
    assert restored.rank_breaks() == law.rank_breaks()
    with pytest.raises(ValidationError):
        NativeSeverityComponent(
            p5=KNOTS[0],
            p25=KNOTS[1],
            p50=KNOTS[2],
            p75=KNOTS[3],
            p95=P95,
            gpd_shape=0.7,
            gpd_tail_ratio=P99 / P95,
            large_loss_amount=AMOUNT,
            large_loss_exceedance=SHARE,
            very_large_loss_amount=HIGHER,
        )


def test_a_law_without_a_second_anchor_is_stored_as_before():
    """A stored law's encoding enters every state's conditional key, so a law
    without a second anchor must keep the fields it had before the second
    anchor existed."""
    component = native_severity_from_sampler(_law(0.7, second=None)).components[0]
    for mode in ("json", "python"):
        assert set(component.model_dump(mode=mode)) == {
            "p5",
            "p25",
            "p50",
            "p75",
            "p95",
            "gpd_shape",
            "severity_cap",
            "severity_minimum",
            "severity_minimum_strict",
            "gpd_scale",
            "gpd_tail_ratio",
            "large_loss_amount",
            "large_loss_exceedance",
        }
    stored = native_severity_from_sampler(_law(0.7)).components[0]
    assert {"very_large_loss_amount", "very_large_loss_exceedance"} <= set(
        stored.model_dump(mode="json")
    )


def test_sampling_and_affine_capped_evaluation_use_the_whole_tail():
    law = _law(0.7)
    ranks = np.linspace(0.98, 1 - 1e-9, 2_001)
    direct = law.sample_from_uniforms(ranks)
    assert np.all(np.diff(direct) > 0)
    capped = sample_affine_capped_severity(law, ranks, cap=5e11)
    assert np.array_equal(capped, np.minimum(direct, 5e11))


def test_mixture_mean_and_inverse_weight_a_two_anchor_component():
    atom = Distribution(
        p5=FLOOR, p25=FLOOR, p50=FLOOR, p75=FLOOR, p95=FLOOR, severity_minimum=FLOOR
    )
    law = _law(0.7)
    mixture = MixtureSeverityDistribution(
        components=[atom, law], weights=np.array([0.4, 0.6])
    )
    assert mixture.analytic_mean() == pytest.approx(
        0.4 * FLOOR + 0.6 * law.analytic_mean()
    )
    assert mixture.survival_probability(HIGHER) == pytest.approx(0.6 * SECOND)
    # In each piece of the tail: below the first anchor, between the anchors
    # and beyond the second.
    survivals = np.array([0.5, SHARE * 2, SHARE / 50, SECOND / 2])
    np.testing.assert_allclose(
        mixture.sample_from_uniforms(1 - 0.6 * survivals),
        law.sample_from_uniforms(1 - survivals),
        rtol=1e-8,
    )


def test_knots_and_rank_breaks_include_both_anchors():
    law = _law(0.7)
    knots = _knots(native_severity_from_sampler(law))
    one = _knots(native_severity_from_sampler(_law(0.7, second=None)))
    assert knots - one == {HIGHER}
    assert law.rank_breaks()[-2:] == (1 - SHARE, 1 - SECOND)
    assert _law(0.7, second=None).rank_breaks()[-1] == 1 - SHARE


@pytest.mark.parametrize("ratio", (1e-3, 0.3))
def test_certified_capped_lab_mean_brackets_a_two_anchor_total(ratio):
    anchored = _law(0.7, second=ratio * SHARE)
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
                # Where the total's distribution function kinks: its floor,
                # knots, p99 break and both anchor amounts.
                [
                    limit - scale * value
                    for value in (FLOOR, *KNOTS, _break(anchored), AMOUNT, HIGHER)
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
