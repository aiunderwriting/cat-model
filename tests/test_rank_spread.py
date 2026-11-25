"""Score-native epistemic count laws and the POOL-SPREAD sampler.

Expected values come from ``CountLawOracle``, which reads the methodology's
formulas with SciPy alone.
"""

from __future__ import annotations

import math
from itertools import pairwise
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from scipy import integrate
from scipy.special import ndtr, ndtri

from catmodel.contracts.elicitation import Quantiles
from catmodel.contracts.incident import SPREAD_TAIL_FORMS, IncidentLaw, ParameterRankLaw
from catmodel.elicitation.epistemic import (
    epistemic_knots,
    epistemic_mean,
    epistemic_quantiles,
    stretched_score_quantiles,
    tail_widths,
)
from catmodel.setup.contextual_compilation import _log_no_incident
from catmodel.setup.rank_spread import REFERENCE_SPREAD_TAIL_FORM
from catmodel.simulation.incident_runtime import (
    parameter_rank_multipliers,
    parameter_scores,
    score_multiplier,
)
from catmodel.simulation.small_events import SIMULATED_SCORE_RANGE
from test_support.count_law_oracle import RANKS, CountLawOracle

LAWS = (
    Quantiles(p5=1, p25=2.2, p50=4.5, p75=9, p95=20),
    Quantiles(p5=1, p25=5, p50=20, p75=80, p95=400),
    Quantiles(p5=0.01, p25=0.1, p50=1, p75=10, p95=100),
    Quantiles(p5=0, p25=0, p50=2, p75=5, p95=9),
    Quantiles(p5=1e-9, p25=0.5, p50=1, p75=2, p95=4),
    Quantiles(p5=1, p25=2, p50=3, p75=5, p95=5),
)
LAW_IDS = (
    "narrow",
    "wide",
    "very-wide",
    "zero-below-p25",
    "near-zero-p5",
    "tied-upper",
)
SPREADS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
TAIL_SCALES = (0.5, 0.75, 1.0, 1.5)


def _rank_law(count, spread=1.0, tail_scale=1.0, tail_form=REFERENCE_SPREAD_TAIL_FORM):
    return ParameterRankLaw(
        group_id="group",
        count=count,
        rank_spread=spread,
        epistemic_tail_scale=tail_scale,
        tail_form=tail_form,
    )


def _log_symmetric_count_law(p95_p5_ratio):
    z5, z25, z75 = map(float, ndtri((0.05, 0.25, 0.75)))
    z95 = -z5
    sigma = math.log(p95_p5_ratio) / (z95 - z5)
    median = math.sqrt(p95_p5_ratio)
    return Quantiles(
        p5=median * math.exp(sigma * z5),
        p25=median * math.exp(sigma * z25),
        p50=median,
        p75=median * math.exp(sigma * z75),
        p95=median * math.exp(sigma * z95),
    )


def _normal_integral(multiplier, low, high, breaks):
    points = sorted({low, high, *(x for x in breaks if low < x < high)})

    def integrand(score):
        value = float(multiplier(np.asarray([score]))[0])
        return value * math.exp(-score * score / 2) / math.sqrt(2 * math.pi)

    return math.fsum(
        integrate.quad(
            integrand,
            left,
            right,
            limit=400,
            epsabs=0,
            epsrel=2e-11,
        )[0]
        for left, right in pairwise(points)
    )


def _range_mean_shortfall(count, spread, tail_scale):
    """The share of the pure-stretch mean from scores outside the simulated range."""
    oracle = CountLawOracle(count, tail_scale=tail_scale, spread=spread)
    low, high = SIMULATED_SCORE_RANGE
    return (oracle.mean(high=low) + oracle.mean(low=high)) / oracle.mean()


@pytest.mark.parametrize("count", LAWS, ids=LAW_IDS)
@pytest.mark.parametrize("spread", (0.75, 1.5, 3.0))
@pytest.mark.parametrize("tail_scale", (0.5, 1.0, 1.5))
def test_stretched_scores_agree_with_unsaturated_probability_quantiles(
    count, spread, tail_scale
):
    # Stay far enough from the representability edge that ndtr/ndtri's
    # probability round trip is accurate to 1e-12 in normal-score space.
    source = np.linspace(-3.0 / spread, 3.0 / spread, 1001)
    stretched = spread * source
    expected = epistemic_quantiles(
        count,
        ndtr(stretched),
        epistemic_tail_scale=tail_scale,
    )
    actual = stretched_score_quantiles(
        count, source, spread=spread, epistemic_tail_scale=tail_scale
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("count", LAWS, ids=LAW_IDS)
def test_stretched_scores_are_exact_at_the_five_knots(count):
    values = np.asarray([count.p5, count.p25, count.p50, count.p75, count.p95])
    actual = stretched_score_quantiles(count, ndtri(np.asarray(RANKS)))
    np.testing.assert_array_equal(actual, values)


def test_stretched_scores_refuse_bad_spreads_and_nonfinite_scores():
    count = LAWS[0]
    for spread in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(
            ValueError, match="rank spread factor must be positive and finite"
        ):
            stretched_score_quantiles(count, np.asarray([0.0]), spread=spread)
    with pytest.raises(ValueError, match="normal scores must be finite"):
        stretched_score_quantiles(count, np.asarray([0.0, math.nan]))


@pytest.mark.parametrize("count", LAWS, ids=LAW_IDS)
@pytest.mark.parametrize("spread", SPREADS)
@pytest.mark.parametrize("tail_scale", TAIL_SCALES)
def test_score_multiplier_normalizes_over_the_whole_normal_score(
    count, spread, tail_scale
):
    law = _rank_law(count, spread, tail_scale, tail_form="pure_stretch")
    multiplier = score_multiplier(law)
    breaks = CountLawOracle(count, tail_scale=tail_scale, spread=spread).kinks()
    limit = max(map(abs, breaks)) + 40
    whole = _normal_integral(multiplier, -limit, limit, breaks)
    assert whole == pytest.approx(1.0, rel=1e-9)

    low, high = SIMULATED_SCORE_RANGE
    represented = _normal_integral(multiplier, low, high, breaks)
    expected = 1 - _range_mean_shortfall(count, spread, tail_scale)
    assert represented == pytest.approx(expected, rel=2e-10, abs=2e-12)
    if spread <= 1.5 and tail_scale <= 1.5:
        # The specified p95/p5=400 and 10,000 laws retain about 3.3e-6 of
        # mean above the 53-bit score ceiling at spread=tail_scale=1.5.
        assert 1 - represented < 3.3e-6


def test_zero_mean_score_multiplier_is_zero():
    zero = Quantiles(p5=0, p25=0, p50=0, p75=0, p95=0)
    multiplier = score_multiplier(_rank_law(zero, spread=4.0))
    np.testing.assert_array_equal(multiplier(np.asarray([-40.0, 0.0, 40.0])), 0.0)


def test_upper_tail_mean_beyond_the_score_ceiling():
    wide = LAWS[1]
    reference_tail_shortfall = _range_mean_shortfall(wide, 1.5, 1.0)
    widened_tail_shortfall = _range_mean_shortfall(wide, 1.5, 1.5)
    assert reference_tail_shortfall < 1e-8
    assert widened_tail_shortfall == pytest.approx(3.25103e-6, rel=1e-5)


def test_k_four_no_incident_integral_uses_stretched_scores():
    count = LAWS[1]
    law = _rank_law(count, spread=4.0, tail_scale=1.5)
    rate = epistemic_mean(
        count, spread=law.rank_spread, epistemic_tail_scale=law.epistemic_tail_scale
    )
    incident = cast(
        IncidentLaw,
        SimpleNamespace(parameter_rank=law, reference_annual_rate=rate),
    )
    survival = 0.01
    actual = _log_no_incident(incident, survival)
    oracle = CountLawOracle(
        count,
        tail_scale=law.epistemic_tail_scale,
        spread=law.rank_spread,
        tail_form=law.tail_form,
    )
    breaks = oracle.kinks()
    limit = max(map(abs, breaks)) + 40

    def integrand(score):
        return (
            math.exp(-survival * oracle.at_score(score))
            * math.exp(-score * score / 2)
            / math.sqrt(2 * math.pi)
        )

    expected = math.log(
        integrate.quad(
            integrand,
            -limit,
            limit,
            points=sorted(x for x in breaks if -limit < x < limit),
            limit=600,
            epsabs=1e-14,
            epsrel=1e-12,
        )[0]
    )
    assert actual == pytest.approx(expected, abs=2e-7)


@pytest.mark.parametrize("count", LAWS, ids=LAW_IDS)
@pytest.mark.parametrize("tail_form", SPREAD_TAIL_FORMS)
def test_score_multiplier_at_unit_spread_matches_the_probability_formula(
    count, tail_form
):
    uniforms = np.concatenate(
        (
            np.geomspace(1e-300, 0.049, 80),
            np.linspace(0.05, 0.95, 501),
            1 - np.geomspace(1e-16, 0.049, 80),
            np.asarray([np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)]),
        )
    )
    uniforms = np.unique(uniforms)
    law = _rank_law(count, tail_form=tail_form)
    actual = score_multiplier(law)(ndtri(uniforms))
    expected = epistemic_quantiles(count, uniforms) / epistemic_mean(count)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0)


@pytest.mark.parametrize("tail_form", SPREAD_TAIL_FORMS)
def test_runtime_rank_multiplier_uses_and_normalizes_each_tail_form(tail_form):
    law = _rank_law(LAWS[1], spread=1.5, tail_scale=1.0, tail_form=tail_form)
    incident = SimpleNamespace(origin_id="origin", parameter_rank=law)
    scores = np.asarray([-4.0, -1.0, 0.0, 1.0, 4.0])
    actual = parameter_rank_multipliers((incident,), {"origin": scores})["origin"]
    oracle = CountLawOracle(
        law.count,
        tail_scale=law.epistemic_tail_scale,
        spread=law.rank_spread,
        tail_form=tail_form,
    )
    expected = [oracle.at_score(float(score)) / oracle.mean() for score in scores]
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=0)

    breaks = oracle.kinks()
    limit = max(map(abs, breaks)) + 40
    normalized = _normal_integral(score_multiplier(law), -limit, limit, breaks)
    assert normalized == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("spread", (0.5, 1.0))
def test_post_stretch_tail_forms_are_identical_at_spread_at_most_one(spread):
    count = LAWS[1]
    scores = np.linspace(-6, 6, 301)
    reads = [
        stretched_score_quantiles(count, scores, spread=spread, tail_form=form)
        for form in SPREAD_TAIL_FORMS
    ]
    for other in reads[1:]:
        np.testing.assert_array_equal(other, reads[0])
    means = [
        epistemic_mean(count, spread=spread, tail_form=form)
        for form in SPREAD_TAIL_FORMS
    ]
    assert means[1] == means[0]


@pytest.mark.parametrize(
    ("ratio", "spread", "tail_form", "expected"),
    (
        (20, 1.5, "pure_stretch", 1.6614633),
        (160, 1.5, "pure_stretch", 4.2115388),
        (400, 1.5, "pure_stretch", 5.3604265),
        (20, 2.0, "pure_stretch", 3.4003860),
        (160, 2.0, "pure_stretch", 33.3435879),
        (400, 2.0, "pure_stretch", 56.8937945),
        (20, 1.5, "calibrated_p95_peak", 1.6614633),
        (160, 1.5, "calibrated_p95_peak", 2.1541236),
        (400, 1.5, "calibrated_p95_peak", 2.1997500),
        (20, 2.0, "calibrated_p95_peak", 2.8459781),
        (160, 2.0, "calibrated_p95_peak", 3.3147893),
        (400, 2.0, "calibrated_p95_peak", 3.4370468),
    ),
)
def test_log_symmetric_count_mean_multipliers_match_the_methodology(
    ratio, spread, tail_form, expected
):
    count = _log_symmetric_count_law(ratio)
    face_value = epistemic_mean(count, spread=1.0, tail_form=REFERENCE_SPREAD_TAIL_FORM)
    actual = epistemic_mean(count, spread=spread, tail_form=tail_form) / face_value
    assert actual == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("spread", (1.5, 2.0))
@pytest.mark.parametrize("ratio", (20, 400))
def test_reference_count_mean_share_above_p95_is_closed_form(ratio, spread):
    count = _log_symmetric_count_law(ratio)
    mean = epistemic_mean(
        count,
        spread=spread,
        tail_form=REFERENCE_SPREAD_TAIL_FORM,
    )
    upper_width = CountLawOracle(
        count, spread=spread, tail_form=REFERENCE_SPREAD_TAIL_FORM
    ).upper_width
    p95_score = -float(ndtri(0.05))
    mean_above_p95 = (
        count.p95
        * math.exp(upper_width**2 * spread**2 / 2 - upper_width * p95_score)
        * float(ndtr(upper_width * spread - p95_score / spread))
    )
    expected = {
        (20, 1.5): 0.5963623,
        (400, 1.5): 0.7794979,
        (20, 2.0): 0.8028922,
        (400, 2.0): 0.8785939,
    }[ratio, spread]
    assert mean_above_p95 / mean == pytest.approx(expected, rel=1e-6)


def test_simulated_stretched_multipliers_have_unit_mean_within_sampling_error():
    law = _rank_law(
        LAWS[1], spread=1.5, tail_scale=1.0, tail_form=REFERENCE_SPREAD_TAIL_FORM
    )
    incidents = (SimpleNamespace(origin_id="origin", parameter_rank=law),)
    scores = parameter_scores(
        incidents, seed=9, simulations=2**18, rank_stream_id="state"
    )
    values = parameter_rank_multipliers(incidents, scores)["origin"]
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    assert abs(mean - 1.0) < 5 * standard_error


def test_extreme_tails_keep_representable_values():
    """The tail combines logs where exp alone would overflow or underflow, and
    a constant law is its constant whatever the spread."""
    tiny = Quantiles(p5=1e-305, p25=1e-304, p50=1e-303, p75=1e-302, p95=1e-301)
    (high,) = stretched_score_quantiles(tiny, np.asarray([500.0]))
    width = CountLawOracle(tiny).upper_width
    expected = math.exp(math.log(1e-301) + width * (500.0 - float(ndtri(0.95))))
    assert high == pytest.approx(expected, rel=1e-12)
    huge = Quantiles(p5=1e280, p25=1e281, p50=1e282, p75=1e282, p95=1e282)
    (low,) = stretched_score_quantiles(huge, np.asarray([-500.0]))
    width = CountLawOracle(huge).lower_width
    expected = math.exp(math.log(1e280) + width * (-500.0 - float(ndtri(0.05))))
    assert low == pytest.approx(expected, rel=1e-12)
    constant = Quantiles(p5=10, p25=10, p50=10, p75=10, p95=10)
    np.testing.assert_array_equal(
        stretched_score_quantiles(constant, np.asarray([2.0]), spread=1e308), 10.0
    )


@pytest.mark.parametrize("count", LAWS, ids=LAW_IDS)
@pytest.mark.parametrize("spread", (0.5, 1.0, 1.5, 3.0))
def test_a_tail_form_that_does_not_bind_reads_the_frozen_mean_exactly(count, spread):
    """Wherever the post-stretch cap leaves the widths unchanged (always at
    k <= 1), the mean is bit-for-bit the POOL-PCHIP mean, so k = 1 readings
    (leaf EALs, collection-time comparisons) are unchanged."""
    values = epistemic_knots(count)
    pure = tail_widths(values, spread=spread)
    for form in SPREAD_TAIL_FORMS:
        if tail_widths(values, spread=spread, tail_form=form) == pure:
            assert epistemic_mean(
                count, spread=spread, tail_form=form
            ) == epistemic_mean(count, spread=spread)
