"""Analytic zero-atom pooling retains tail probabilities and guarded fallbacks."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from catmodel.probability import distributions
from catmodel.probability.distributions import Distribution, MixtureSeverityDistribution


def _mixture(*, mass=0.65, shape=0.7, intrinsic_cap=None, outer_cap=None):
    return MixtureSeverityDistribution(
        [
            Distribution(0, 0, 0, 0, 0, severity_minimum=0),
            Distribution(
                5000,
                25000,
                150000,
                750000,
                5000000,
                severity_minimum=0,
                severity_cap=intrinsic_cap,
                gpd_shape=shape,
                gpd_tail_ratio=5,
            ),
        ],
        np.array([1 - mass, mass]),
        severity_cap=outer_cap,
    )


def _tail_truth(law, mass, probability):
    with localcontext() as context:
        context.prec = 100

        def number(value):
            return Decimal.from_float(float(value))

        survival = (1 - number(probability)) / number(mass) / (1 - number(0.95))
        shape, scale = number(law.gpd_shape), number(law._gpd_scale)
        excess = (
            -scale * survival.ln()
            if abs(shape) < Decimal("1e-8")
            else scale / shape * ((-shape * survival.ln()).exp() - 1)
        )
        return float(number(law._gpd_threshold) + excess)


def test_zero_atom_shortcut_avoids_numerical_roots(monkeypatch):
    mixture = _mixture()
    ranks = np.array([0, 0.1, 0.34, 0.351, 0.41, 0.64, 0.88, 0.98, 0.99999])

    def unexpected(*args, **kwargs):
        raise AssertionError("eligible zero-atom inverse invoked root solver")

    monkeypatch.setattr(distributions.elementwise, "find_root", unexpected)
    actual = mixture.sample_from_uniforms_uncapped(ranks)
    assert np.all(actual[ranks <= 0.35] == 0)
    assert np.all(np.diff(actual[ranks > 0.35]) > 0)


@pytest.mark.parametrize("mass", [0.01, 0.15, 0.3, 0.65, 0.99])
@pytest.mark.parametrize("shape", [0, 1e-9, 0.3, 0.7, 1.2, 3])
def test_zero_atom_tail_keeps_original_survival(mass, shape):
    mixture = _mixture(mass=mass, shape=shape)
    ranks = 1 - np.geomspace(np.finfo(float).eps / 2, 1e-7, 12)
    actual = mixture.sample_from_uniforms_uncapped(ranks)
    expected = np.array(
        [_tail_truth(mixture.components[1], mixture.weights[1], rank) for rank in ranks]
    )
    # Float64 log/expm1 accuracy, including the rank immediately below one.
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0)


def test_zero_atom_and_knot_boundaries_retain_original_inverse_values():
    mixture = _mixture()
    boundaries = mixture.weights[0] + mixture.weights[1] * np.array(
        [0, 0.05, 0.25, 0.5, 0.75, 0.95]
    )
    ranks = np.concatenate(
        [
            np.nextafter(boundaries, -np.inf),
            boundaries,
            np.nextafter(boundaries, np.inf),
        ]
    )
    actual = mixture.sample_from_uniforms_uncapped(ranks)
    expected = mixture.sample_from_uniforms_uncapped(ranks, _allow_zero_atom=False)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("shape", [0, 0.7])
def test_zero_atom_endpoint_and_outer_cap(shape):
    mixture = _mixture(shape=shape, outer_cap=1e9)
    ranks = np.array([0, 0.35, np.nextafter(1, 0), 1])
    raw = mixture.sample_from_uniforms_uncapped(ranks)
    capped = mixture.sample_from_uniforms(ranks)
    assert np.all(raw[:2] == 0)
    assert raw[-1] == mixture.components[1].quantile(1)
    np.testing.assert_array_equal(capped, np.minimum(raw, 1e9))


@pytest.mark.parametrize("kind", ["flat", "intrinsic_cap", "no_gpd"])
def test_unsupported_zero_atom_laws_keep_general_inverse(monkeypatch, kind):
    mixture = _mixture(intrinsic_cap=1e8 if kind == "intrinsic_cap" else None)
    if kind == "flat":
        mixture.components[1] = Distribution(
            1, 2, 2, 3, 4, severity_minimum=0, gpd_shape=0.7
        )
    elif kind == "no_gpd":
        mixture.components[1] = Distribution(1, 2, 3, 4, 5, severity_minimum=0)
    calls = []
    original = distributions.elementwise.find_root

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(distributions.elementwise, "find_root", counted)
    result = mixture.sample_from_uniforms_uncapped(np.array([0.4, 0.6, 0.9]))
    assert np.all(np.isfinite(result))
    if kind == "flat":
        assert calls
    else:
        # Bounded disjoint supports already have their own exact pooling path.
        expected = mixture.components[1].sample_from_uniforms(
            (np.array([0.4, 0.6, 0.9]) - mixture.weights[0]) / mixture.weights[1]
        )
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("shape", [-0.5, -1.0])
def test_bounded_negative_tail_retains_existing_disjoint_inverse(shape):
    mixture = _mixture(shape=shape)
    ranks = np.array([0, 0.35, 0.351, 0.4, 0.9, 0.999, np.nextafter(1, 0), 1])
    actual = mixture.sample_from_uniforms_uncapped(ranks)
    expected = mixture.sample_from_uniforms_uncapped(ranks, _allow_zero_atom=False)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("exceedance", [1e-6, 1e-4, 9e-3])
def test_anchored_component_is_pooled_through_both_tail_pieces(exceedance):
    """A large-loss anchor's tail above p99 survives pooling.

    The component's p99 is 5 x 5e6 = 2.5e7 and its anchor puts the given
    share of incidents at or above 1e9. Pooled ranks read the component at
    the conditional rank on both sides of the p99 break.
    """
    mass = 0.65
    anchored = Distribution(
        5000,
        25000,
        150000,
        750000,
        5000000,
        severity_minimum=0,
        gpd_shape=0.7,
        gpd_tail_ratio=5,
        large_loss_amount=1e9,
        large_loss_exceedance=exceedance,
    )
    mixture = MixtureSeverityDistribution(
        [Distribution(0, 0, 0, 0, 0, severity_minimum=0), anchored],
        np.array([1 - mass, mass]),
    )
    conditional = np.array([0.5, 0.98, 0.995, 0.999, 1 - 1e-6])
    np.testing.assert_allclose(
        mixture.sample_from_uniforms(1 - mass + mass * conditional),
        anchored.sample_from_uniforms(conditional),
        rtol=1e-8,
    )
    assert mixture.survival_probability(1e9) == pytest.approx(mass * exceedance)
