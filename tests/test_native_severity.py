"""Native law serialization and exact lower-support admission invariants."""

from dataclasses import fields

import numpy as np
import pytest
from pydantic import ValidationError

from catmodel.contracts.native_severity import (
    NativeSeverityComponent,
    NativeSeverityLaw,
    native_severity_from_sampler,
    native_severity_sampler,
)
from catmodel.probability.distributions import (
    Distribution,
    MixtureSeverityDistribution,
    sample_affine_capped_severity,
    severity_lower_support,
)


@pytest.mark.parametrize(
    "tail",
    [
        {},
        {"gpd_shape": 0.3},
        {"gpd_shape": 0.3, "gpd_scale": 40.0},
        {"gpd_shape": 0.3, "gpd_tail_ratio": 3.0},
    ],
)
def test_native_law_round_trip_after_affine_change(tail):
    minimum = 20.0
    original = Distribution(
        25.0,
        40.0,
        70.0,
        110.0,
        180.0,
        severity_minimum=minimum,
        severity_cap=1000.0,
        **tail,
    )
    offset, factor = minimum * (1 - 1.7), 1.7
    original = original._with_affine_severity(offset, factor)
    contract = native_severity_from_sampler(original)
    restored_contract = NativeSeverityLaw.model_validate_json(
        contract.model_dump_json()
    )
    restored = native_severity_sampler(restored_contract)
    assert isinstance(restored, Distribution)
    for field in fields(Distribution):
        if field.init:
            assert getattr(restored, field.name) == getattr(original, field.name)
    ranks = np.array([0.0, 0.001, 0.05, 0.5, 0.95, 0.99, 1 - 1e-13])
    np.testing.assert_array_equal(
        restored.sample_from_uniforms(ranks), original.sample_from_uniforms(ranks)
    )
    np.testing.assert_array_equal(
        restored.sample_from_uniforms_uncapped(ranks),
        original.sample_from_uniforms_uncapped(ranks),
    )
    assert restored.analytic_mean() == original.analytic_mean()
    assert severity_lower_support(restored) == severity_lower_support(original)


def test_round_trip_retains_intrinsic_caps_and_zero_weights():
    first = Distribution(
        p5=25,
        p25=40,
        p50=60,
        p75=85,
        p95=130,
        gpd_shape=0.7,
        gpd_scale=10,
        severity_cap=70,
    )
    second = Distribution(
        1,
        3,
        5,
        9,
        20,
        gpd_shape=1.2,
        gpd_tail_ratio=None,
        gpd_scale=20,
    )
    inactive = Distribution(0, 0, 0, 0, 0)
    original = MixtureSeverityDistribution(
        [first, second, inactive], np.array([0.25, 0.75, 0.0]), severity_cap=100
    )
    contract = native_severity_from_sampler(original)
    restored = native_severity_sampler(
        NativeSeverityLaw.model_validate_json(contract.model_dump_json())
    )
    assert isinstance(restored, MixtureSeverityDistribution)
    assert restored.components[0].severity_cap == 70
    assert restored.severity_cap == 100
    np.testing.assert_array_equal(restored.weights, original.weights)
    ranks = np.array([0, 0.01, 0.25, 0.5, 0.95, 0.999, 1 - 1e-13])
    np.testing.assert_array_equal(
        restored.sample_from_uniforms(ranks), original.sample_from_uniforms(ranks)
    )
    np.testing.assert_array_equal(
        restored.sample_from_uniforms_uncapped(ranks),
        original.sample_from_uniforms_uncapped(ranks),
    )
    for multiplier in (0.5, 1.0, 2.0):
        np.testing.assert_array_equal(
            sample_affine_capped_severity(restored, ranks, scale=multiplier),
            sample_affine_capped_severity(original, ranks, scale=multiplier),
        )
    assert restored.analytic_mean() == original.analytic_mean()


def test_native_component_contract_covers_every_distribution_generative_field():
    runtime_fields = {field.name for field in fields(Distribution) if field.init}
    assert set(NativeSeverityComponent.model_fields) == runtime_fields


@pytest.mark.parametrize(
    "law,expected",
    [
        (Distribution(1, 1, 1, 1, 1), (1.0, True)),
        (Distribution(1, 2, 4, 8, 16, severity_minimum=0), (0.0, False)),
        (Distribution(0, 2, 4, 8, 16, severity_minimum=0), (0.0, True)),
        (Distribution(0, 2, 4, 8, 16), (0.0, False)),
        (
            Distribution(
                2, 3, 4, 8, 16, severity_minimum=1, severity_minimum_strict=True
            ),
            (1.0, False),
        ),
        (Distribution(1, 1, 1, 1, 1, severity_cap=0.5), (0.5, True)),
    ],
)
def test_lower_support_distinguishes_infimum_from_atom(law, expected):
    assert severity_lower_support(law) == expected
    restored = native_severity_sampler(native_severity_from_sampler(law))
    assert severity_lower_support(restored) == expected


def test_lower_support_preserves_tiny_positive_atom_and_intrinsic_cap():
    original = MixtureSeverityDistribution(
        [Distribution(1, 1, 1, 1, 1, severity_cap=0.5), Distribution(2, 3, 4, 5, 6)],
        np.array([1e-300, 1.0]),
        severity_cap=0.25,
    )
    assert severity_lower_support(original) == (0.25, True)
    assert severity_lower_support(original, apply_severity_cap=False) == (0.5, True)
    restored = native_severity_sampler(native_severity_from_sampler(original))
    assert severity_lower_support(restored, apply_severity_cap=False) == (0.5, True)


def test_native_distribution_does_not_admit_hidden_extra_mixture_cap():
    contract = native_severity_from_sampler(Distribution(1, 2, 3, 4, 5))
    with pytest.raises(ValidationError, match="no additional cap"):
        NativeSeverityLaw.model_validate({**contract.model_dump(), "severity_cap": 3})


def test_round_trip_does_not_renormalize_persisted_weight_bits():
    rng = np.random.default_rng(7)
    components = [
        Distribution(1, 2, 3, 4, 5),
        Distribution(2, 3, 4, 5, 6),
        Distribution(3, 4, 5, 6, 7),
    ]
    for _ in range(20):
        original = MixtureSeverityDistribution(components, rng.random(3))
        restored = native_severity_sampler(native_severity_from_sampler(original))
        assert isinstance(restored, MixtureSeverityDistribution)
        np.testing.assert_array_equal(restored.weights, original.weights)
        assert restored.analytic_mean() == original.analytic_mean()
