"""An authored lower ramp remains part of a mixture's pooled law and mean."""

from fractions import Fraction

import numpy as np
import pytest

from catmodel.probability.distributions import Distribution, MixtureSeverityDistribution


def _mixture():
    return MixtureSeverityDistribution(
        [
            Distribution(10, 10, 10, 10, 10, severity_minimum=1),
            Distribution(20, 20, 20, 20, 20, severity_minimum=1),
        ],
        np.array([0.5, 0.5]),
    )


# Pool two 2.5% uniform ramps from the floor and two 47.5% atoms.
_DENSITY = Fraction(1, 40) * (Fraction(1, 9) + Fraction(1, 19))
_FIRST_ATOM = 9 * _DENSITY
_FIRST_ATOM_END = _FIRST_ATOM + Fraction(19, 40)
_SECOND_ATOM = Fraction(21, 40)
_PIECES = (
    (Fraction(0), _FIRST_ATOM, Fraction(1), 1 / _DENSITY),
    (_FIRST_ATOM, _FIRST_ATOM_END, Fraction(10), Fraction(0)),
    (_FIRST_ATOM_END, _SECOND_ATOM, Fraction(10), Fraction(760)),
    (_SECOND_ATOM, Fraction(1), Fraction(20), Fraction(0)),
)


def _quantile_oracle(rank):
    rank = Fraction(str(rank))
    for start, stop, value, slope in _PIECES:
        if start <= rank <= stop:
            return float(value + slope * (rank - start))
    raise AssertionError(rank)


@pytest.mark.parametrize("rank", [0.001, 0.005, 0.01, 0.02, 0.3, 0.5, 0.51, 0.52, 0.9])
def test_mixture_inverse_retains_authored_ramps(rank):
    assert _mixture().quantile(rank) == pytest.approx(_quantile_oracle(rank), rel=2e-12)


def test_full_mixture_core_mean_matches_its_component_laws():
    mixture = _mixture()
    assert [
        component.analytic_mean() for component in mixture.components
    ] == pytest.approx([9.775, 19.525])
    assert mixture.analytic_mean() == pytest.approx(14.65)


def test_true_point_mass_mixture_retains_exact_atoms():
    mixture = MixtureSeverityDistribution(
        [Distribution(10, 10, 10, 10, 10), Distribution(20, 20, 20, 20, 20)],
        np.array([0.5, 0.5]),
    )
    assert mixture.quantile(0.25) == 10
    assert mixture.quantile(0.75) == 20
    assert mixture.analytic_mean() == 15


def test_intrinsically_capped_component_keeps_its_lower_ramp_in_mixture_means():
    capped = Distribution(10, 10, 10, 10, 10, severity_minimum=1, severity_cap=5)
    other = Distribution(20, 20, 20, 20, 20, severity_minimum=1)
    mixture = MixtureSeverityDistribution([capped, other], np.array([0.5, 0.5]))
    capped_mean = float(Fraction(223, 45))
    expected = float((Fraction(223, 45) + Fraction(781, 40)) / 2)
    assert capped.analytic_mean() == pytest.approx(capped_mean)
    assert mixture.analytic_mean() == pytest.approx(expected)
    # Below the cap crossing, both original uniform lower ramps remain intact.
    assert mixture.quantile(0.005) == pytest.approx(_quantile_oracle(0.005))
    assert expected != pytest.approx((5 + other.analytic_mean()) / 2)
