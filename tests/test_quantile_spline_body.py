"""The quantile-spline severity body: shape, accuracy, conventions and splice."""

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad
from scipy.special import ndtri

from catmodel.probability.distributions import Distribution, severity_lower_support
from test_support.severity_oracles import PROBABILITIES, SplineOracle

_INTERIOR = np.linspace(0.05, 0.95, 181)


def _integral(law, lower: float, upper: float) -> float:
    """Integrate the body quantile over [lower, upper] within [p5, p95]."""
    return law._body.integral(lower, upper)


def _tail_slope(law) -> float:
    scale = law.effective_gpd_scale
    assert scale is not None
    return scale / 0.05


def _true_body_integral(ppf) -> float:
    return quad(ppf, 0.05, 0.95, points=PROBABILITIES[1:-1], limit=400)[0]


class _LeftTruncatedLognormal:
    def __init__(self, lower: float, scale: float, sigma: float):
        self.parent = stats.lognorm(sigma, scale=scale)
        self.mass_below = self.parent.cdf(lower)

    def ppf(self, u):
        return self.parent.ppf(self.mass_below + u * (1 - self.mass_below))


@pytest.mark.parametrize("sigma", [0.5, 1.8, 3.0])
@pytest.mark.parametrize("floor", [None, 1e4])
def test_lognormal_ladder_is_reproduced_exactly(sigma, floor):
    shift = 0.0 if floor is None else floor
    parent = stats.lognorm(sigma, scale=5e4)
    law = Distribution(*(shift + parent.ppf(PROBABILITIES)), severity_minimum=floor)
    np.testing.assert_allclose(
        law.sample_from_uniforms(_INTERIOR), shift + parent.ppf(_INTERIOR), rtol=1e-13
    )
    assert _integral(law, 0.05, 0.95) == pytest.approx(
        _true_body_integral(lambda u: shift + parent.ppf(u)), rel=1e-12
    )


# Validation evidence for the interpolation assumption (methodology.md,
# assumptions.md `SEV-QUANTILE-LAW`): from only the five knots and without a
# pinned tail, the body's [p5, p95] integral stays within 1.5% of each analytic
# family's own integral. The largest measured error is 1.2% (Lomax shape 0.8).
_FAMILIES = {
    "lognormal sigma 1": (stats.lognorm(1.0, scale=5e6).ppf, None),
    "lognormal sigma 3": (stats.lognorm(3.0, scale=5e6).ppf, None),
    "lomax shape 1.5": (stats.lomax(1.5, scale=1e6).ppf, None),
    "lomax shape 0.8": (stats.lomax(0.8, scale=1e6).ppf, None),
    "log-logistic shape 1.2": (stats.fisk(1.2, scale=1e6).ppf, None),
    "weibull shape 0.5": (stats.weibull_min(0.5, scale=1e6).ppf, None),
    "weibull shape 2": (stats.weibull_min(2.0, scale=1e6).ppf, None),
    "gamma shape 2": (stats.gamma(2.0, scale=1e6).ppf, None),
    "gpd shape 0.5 above 1e5": (
        stats.genpareto(0.5, loc=1e5, scale=1e5).ppf,
        1e5,
    ),
    "gpd shape 0.7 above 1e7": (stats.genpareto(0.7, loc=1e7, scale=5e6).ppf, 1e7),
    "left-truncated lognormal above 1e4": (
        _LeftTruncatedLognormal(1e4, 5e4, 1.38).ppf,
        1e4,
    ),
}


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_body_integral_tracks_common_severity_families(family):
    ppf, floor = _FAMILIES[family]
    knots = ppf(PROBABILITIES)
    law = Distribution(*knots, severity_minimum=floor)
    body = _integral(law, 0.05, 0.95)
    assert body == pytest.approx(_true_body_integral(ppf), rel=0.015)


@pytest.mark.parametrize(
    "knots,floor,tail",
    [
        ((1e5, 1e6, 5e6, 2e7, 1e8), None, {}),
        ((1e5, 1e6, 5e6, 2e7, 1e8), None, {"gpd_shape": 0.6, "gpd_tail_ratio": 3.0}),
        ((13e3, 28e3, 62e3, 146e3, 530e3), 1e4, {"gpd_shape": 0.4, "gpd_scale": 4e5}),
        ((0.1, 0.1, 0.1, 0.3, 1.0), None, {}),
        ((10, 25, 50, 100, 200), 0.0, {"gpd_shape": 0.25, "gpd_tail_ratio": 3.0}),
    ],
)
def test_body_matches_an_independent_scipy_spline(knots, floor, tail):
    law = Distribution(*knots, severity_minimum=floor, **tail)
    end_slope = (
        None if law._gpd_scale_source in {"none", "continuity"} else _tail_slope(law)
    )
    oracle = SplineOracle(knots, floor=floor or 0.0, end_slope=end_slope)
    np.testing.assert_allclose(
        law.sample_from_uniforms_uncapped(_INTERIOR),
        oracle.quantile(_INTERIOR),
        rtol=1e-13,
    )
    for lower, upper in ((0.05, 0.95), (0.1, 0.3), (0.6, 0.94)):
        assert _integral(law, lower, upper) == pytest.approx(
            oracle.integral(lower, upper), rel=1e-12
        )
    amounts = np.unique(
        np.r_[
            knots[:-1],
            np.nextafter(np.asarray(knots[1:]), -np.inf),
            oracle.quantile(_INTERIOR[1:-1]),
        ]
    )
    amounts = amounts[(amounts >= knots[0]) & (amounts < knots[-1])]
    np.testing.assert_allclose(
        law.cdf(amounts),
        [oracle.cdf(x) for x in amounts],
        rtol=0,
        atol=4e-15,
    )


def test_floor_preserving_affine_changes_map_the_whole_law():
    law = Distribution(
        13e3, 28e3, 62e3, 146e3, 530e3, gpd_shape=0.4, severity_minimum=1e4
    )
    ranks = np.r_[0.0, 0.01, _INTERIOR, 0.99, 0.999]
    raw = law.sample_from_uniforms(ranks)
    for k in (0.4, 2.2):
        scaled = law.with_excess_scale(k)
        np.testing.assert_allclose(
            scaled.sample_from_uniforms(ranks), 1e4 + k * (raw - 1e4), rtol=1e-12
        )
    converted = law.with_currency_scale(0.9)
    np.testing.assert_allclose(
        converted.sample_from_uniforms(ranks), 0.9 * raw, rtol=1e-12
    )


def _atom(law, amount: float) -> float:
    below = np.nextafter(amount, -np.inf)
    return float(law.cdf(np.array([amount]))[0] - law.cdf(np.array([below]))[0])


def _quantile_slopes(law, probability, step=1e-7):
    left = (law.quantile(probability) - law.quantile(probability - step)) / step
    right = (law.quantile(probability + step) - law.quantile(probability)) / step
    return left, right


@pytest.mark.parametrize("ratio", [1.5, 3.0])
def test_a_pinned_tail_scale_makes_the_density_continuous_at_p95(ratio):
    knots = (1e5, 1e6, 5e6, 2e7, 1e8)
    plain = Distribution(*knots)
    law = Distribution(*knots, gpd_shape=0.6, gpd_tail_ratio=ratio)
    left, right = _quantile_slopes(law, 0.95)
    assert right == pytest.approx(_tail_slope(law), rel=1e-5)
    assert left == pytest.approx(right, rel=1e-5)
    assert [law.quantile(p) for p in PROBABILITIES] == list(knots)
    assert law.quantile(0.99) == pytest.approx(ratio * law.p95, rel=1e-14)
    # The body through p75 depends on the knots alone; the p75-p95 segment
    # follows the tail scale.
    lower = _INTERIOR[_INTERIOR <= 0.75]
    np.testing.assert_array_equal(
        law.sample_from_uniforms(lower), plain.sample_from_uniforms(lower)
    )
    assert law.quantile(0.9) != plain.quantile(0.9)


def test_unreachable_tail_slope_leaves_a_bounded_density_step():
    # The sufficient monotone bound caps the body's end slope at three times
    # the last secant.
    law = Distribution(1, 2, 4, 8, 16, gpd_shape=0.5, gpd_scale=1000)
    left, right = _quantile_slopes(law, 0.95)
    secant = (np.log(16) - np.log(8)) / (ndtri(0.95) - ndtri(0.75))
    density = stats.norm.pdf(ndtri(0.95))
    assert left == pytest.approx(16 * 3 * secant / density, rel=1e-5)
    assert right == pytest.approx(1000 / 0.05, rel=1e-5)
    assert np.all(np.diff(law.sample_from_uniforms(np.linspace(0.75, 0.96, 2001))) > 0)


@pytest.mark.parametrize("gpd_shape", [None, 0.5])
def test_end_continuations_never_invent_an_atom_the_knots_do_not_imply(gpd_shape):
    # ln(p50/p25) > 1.87 ln(p25/p5) zeroes PCHIP's slope at p5, and a strongly
    # log-concave top zeroes it at p95; the continuations use the end secants.
    law = Distribution(1e6, 1.4e6, 3e6, 1e7, 3e7, gpd_shape=gpd_shape)
    assert law._body._segment_slope(0, law._body.left[0], 0) == 0.0
    # PCHIP's zero end slope leaves only a vanishing quadratic flat point.
    assert law.quantile(0.01) < law.quantile(0.049) < 1e6
    assert _atom(law, 1e6) < 1e-6
    assert severity_lower_support(law) == (law.quantile(0.0), False)
    concave = Distribution(1e6, 5e6, 15e6, 50e6, 80e6, gpd_shape=gpd_shape)
    assert concave._body._segment_slope(3, concave._body.right[3], 4) == 0.0
    secant = (80e6 - 50e6) / 0.2
    assert concave._upper_slope == pytest.approx(secant)
    assert _atom(concave, 80e6) < 1e-6
    assert 80e6 < concave.quantile(0.951) < concave.quantile(0.99)
    if gpd_shape is not None:
        assert concave.effective_gpd_scale == pytest.approx(0.05 * secant)
    # Equal knots imply an atom, which the continuation keeps.
    assert _atom(Distribution(2, 2, 3, 4, 5), 2) == pytest.approx(0.25)


# Ladders whose PCHIP end slope vanishes within a 1e-12 relative change of the
# end knot. Above p95 the continuity GPD scale and the linear continuation
# continue the same body; below p5, p25 = 1.4 p5 gives a linear continuation
# and p25 = 10 p5 (the ladder 1, 10, 750, 1e4, 1e5 with p5 moved to where the
# slope vanishes) a power continuation.
_P95_SLOPE_VANISHES = (
    95.33794514311101,
    1058.3955525794756,
    1058.3955525794756,
    10055.209835726824,
    33444.094314239715,
)
# Each family: the ladder, the index of the end knot moved, the GPD shape.
_VANISHING_END_SLOPE = {
    "continuity GPD scale above p95": (_P95_SLOPE_VANISHES, 4, 0.6),
    "linear above p95": (_P95_SLOPE_VANISHES, 4, None),
    "linear below p5": ((999980.5183114671, 1.4e6, 2.6296e6, 1e7, 3e7), 0, None),
    "power below p5": ((0.9978723844369357, 10.0, 750.0, 1e4, 1e5), 0, None),
}
_CONTINUITY_RANKS = np.r_[
    0.0, 1e-12, 1e-4, 0.01, 0.049, 0.05, 0.5, 0.95, 0.951, 0.99, 0.999, 1 - 1e-12
]


@pytest.mark.parametrize("family", sorted(_VANISHING_END_SLOPE))
def test_continuations_stay_continuous_where_the_pchip_end_slope_vanishes(family):
    # A secant fallback only where the slope vanishes exactly, with a tiny
    # positive slope used as is, moves the mean by 2e-6 to 10% and quantiles
    # by 10% or more across this change.
    knots, end, shape = _VANISHING_END_SLOPE[family]
    laws = []
    for factor in (1 - 1e-12, 1.0, 1 + 1e-12):
        p5, p25, p50, p75, p95 = (
            knot * factor if index == end else knot for index, knot in enumerate(knots)
        )
        laws.append(Distribution(p5, p25, p50, p75, p95, gpd_shape=shape))
    index, side = (0, "left") if end == 0 else (3, "right")
    pchip = [
        law._body._segment_slope(index, getattr(law._body, side)[index], end)
        for law in laws
    ]
    assert min(pchip) == 0.0 < max(pchip)
    middle = laws[1]
    expected = middle.sample_from_uniforms(_CONTINUITY_RANKS)
    if end == 0:
        # A power continuation reaches zero; a linear one stops above it.
        assert (expected[0] == 0.0) == family.startswith("power")
    for law in (laws[0], laws[2]):
        np.testing.assert_allclose(
            law.sample_from_uniforms(_CONTINUITY_RANKS), expected, rtol=1e-9, atol=0
        )
        assert law.analytic_mean() == pytest.approx(middle.analytic_mean(), rel=1e-9)


@pytest.mark.parametrize("sigma", [0.5, 1.8, 3.0])
@pytest.mark.parametrize("floor", [None, 1e4])
def test_lognormal_ladder_continues_at_its_own_end_slopes(sigma, floor):
    # dq/du of floor + lognormal at a knot is its excess times sigma over the
    # normal density at the knot's score. Below p5 at sigma 1.8 and 3 that slope
    # lies under the [p5, p25] probability secant, which a floor at the secant
    # alone would substitute.
    shift = 0.0 if floor is None else floor
    knots = shift + stats.lognorm(sigma, scale=5e4).ppf(PROBABILITIES)
    law = Distribution(*knots, severity_minimum=floor)
    density, step = stats.norm.pdf(ndtri(0.95)), 1e-8
    above = (law.quantile(0.95 + step) - law.quantile(0.95)) / step
    assert above == pytest.approx((knots[4] - shift) * sigma / density, rel=1e-6)
    if floor is None:
        below = (law.quantile(0.05) - law.quantile(0.05 - step)) / step
        assert below == pytest.approx(knots[0] * sigma / density, rel=1e-6)


def test_zero_p5_keeps_a_support_endpoint_without_an_atom():
    law = Distribution(0.0, 1e5, 5e5, 2.5e6, 2e7, gpd_shape=0.5, gpd_tail_ratio=5.0)
    assert law.quantile(0.0) == 0.0
    assert 0.0 < law.quantile(0.05) < 1e5 * 1e-5
    assert _atom(law, law.quantile(0.05)) < 1e-15
    assert [law.quantile(p) for p in PROBABILITIES[1:]] == [1e5, 5e5, 2.5e6, 2e7]
    # The zero-origin segment rises from a zero slope and continues the
    # following log segment's slope.
    left, right = _quantile_slopes(law, 0.25, step=1e-8)
    assert left == pytest.approx(right, rel=1e-4)
    assert _quantile_slopes(law, 0.05 + 1e-6, step=1e-8)[1] < 1e-3 * right
    values = law.sample_from_uniforms(np.linspace(0.0, 0.999, 100_001))
    assert np.all(np.diff(values) >= 0)


def test_steep_first_log_segment_bounds_the_zero_origin_slope():
    # Matching a decade step from zero would exceed the monotone bound of three
    # times the secant, so the slope steps up at p25.
    law = Distribution(0.0, 1e5, 1e6, 1e7, 1e8)
    left, right = _quantile_slopes(law, 0.25, step=1e-8)
    secant = (1e5 - law.quantile(0.05)) / (ndtri(0.25) - ndtri(0.05))
    assert left == pytest.approx(3 * secant / stats.norm.pdf(ndtri(0.25)), rel=1e-4)
    assert right > left


def test_inclusive_floor_and_repeated_zero_knots_keep_their_atoms():
    floored = Distribution(1e4, 2e4, 5e4, 1e5, 4e5, severity_minimum=1e4)
    assert floored.cdf(np.array([1e4]))[0] == 0.05
    assert _atom(floored, 1e4) == 0.05
    zeros = Distribution(0.0, 0.0, 1e3, 1e4, 1e5)
    assert zeros.cdf(np.array([0.0]))[0] == 0.25
    assert zeros.quantile(0.25) == 0.0
    assert zeros.quantile(0.5) == 1e3


def test_random_ladders_reproduce_knots_and_stay_monotone():
    rng = np.random.default_rng(20261029)
    ranks = np.linspace(0.0, 0.9999, 20_001)
    for _ in range(50):
        steps = rng.lognormal(0.5, 1.0, size=5)
        steps[rng.random(5) < 0.15] = 0.0
        knots = 1e3 * np.cumsum(steps)
        if knots[2] <= 0:
            continue
        law = Distribution(
            *knots, gpd_shape=0.5, gpd_tail_ratio=float(rng.uniform(1.2, 10))
        )
        authored = [law.quantile(p) for p in PROBABILITIES]
        assert authored[1:] == list(knots[1:])
        assert authored[0] == knots[0] or (knots[0] == 0 and authored[0] > 0)
        values = law.sample_from_uniforms(ranks)
        assert np.all(np.diff(values) >= 0)
        if np.all(np.diff(knots) > 0):
            np.testing.assert_allclose(
                law.cdf(law.sample_from_uniforms(_INTERIOR[1:-1])),
                _INTERIOR[1:-1],
                rtol=0,
                atol=1e-13,
            )
