"""Component amount laws and native payments at given ranks."""

import numpy as np
import pytest

from catmodel.contracts.components import (
    ComponentAmountLaw,
    NativeLiabilityParameters,
    NativePaymentKernel,
)
from catmodel.contracts.native_severity import native_severity_from_sampler
from catmodel.probability.distributions import Distribution
from catmodel.simulation.authored.payments import (
    native_payment_positive_mean_lower_bound,
)
from test_support.components import sample_component_amounts, sample_native_payments


def test_native_severity_rank_coupling_and_multiplier_before_cap():
    severity = native_severity_from_sampler(
        Distribution(1, 2, 4, 8, 16, gpd_shape=1.5, gpd_scale=1000, severity_cap=100)
    )
    laws = (
        ComponentAmountLaw(
            component_id="one", severity=severity, rank_group_id="shared"
        ),
        ComponentAmountLaw(
            component_id="two",
            severity=severity,
            rank_group_id="shared",
            multiplier=0.5,
        ),
    )
    sampled = sample_component_amounts(
        laws, np.random.default_rng(1), rank_uniforms={"shared": 0.975}
    )
    assert [item.amount for item in sampled] == [100, 100]
    light = native_severity_from_sampler(
        Distribution(1, 2, 4, 8, 16, gpd_shape=0.5, gpd_scale=1000)
    )
    shared = tuple(law.model_copy(update={"severity": light}) for law in laws)
    sampled = sample_component_amounts(
        shared, np.random.default_rng(1), rank_uniforms={"shared": 0.9}
    )
    assert sampled[0].amount == 2 * sampled[1].amount


def test_zero_multiplier_does_not_evaluate_unused_stochastic_law(monkeypatch):
    def unused(*args):
        raise AssertionError("unused stochastic severity evaluated")

    monkeypatch.setattr("catmodel.contracts.native_severity._shared_sampler", unused)
    law = ComponentAmountLaw(
        component_id="zero",
        multiplier=0,
        severity=native_severity_from_sampler(
            Distribution(1, 2, 3, 4, 5, gpd_shape=1.2, gpd_scale=1)
        ),
        rank_group_id="unused",
    )
    values = sample_component_amounts((law,), np.random.default_rng(1))
    assert values[0].amount == 0


def test_component_ratio_follows_estimand_cap():
    severity = native_severity_from_sampler(
        Distribution(100, 100, 100, 100, 100, severity_cap=50)
    )
    law = ComponentAmountLaw(
        component_id="additive",
        severity=severity,
        rank_group_id="source",
        post_multiplier=2,
    )
    assert sample_component_amounts((law,), np.random.default_rng(1))[0].amount == 100


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_native_payment_endpoints_are_exact_and_skip_unused_source(
    fraction, monkeypatch
):
    def unused(*args):
        raise AssertionError("unused source severity evaluated")

    monkeypatch.setattr("catmodel.contracts.native_severity._shared_sampler", unused)
    kernel = NativePaymentKernel(
        source_severity=native_severity_from_sampler(Distribution(1, 2, 3, 4, 5)),
        rank_group_id="unused",
        entity_id="payer",
        liability_component_ids=("harm",),
        liability=NativeLiabilityParameters(
            mean=fraction,
            concentration=5,
            recovery_rate=fraction,
            severity_slope=0,
            median_severity=None,
            regulatory_recovery_threshold=1,
            regulatory_recovery_uplift=0,
        ),
    )
    obligations, payments = sample_native_payments(
        kernel, np.random.default_rng(1), rank_uniforms={}
    )
    assert obligations[0].fraction == payments[0].fraction == fraction
    assert native_payment_positive_mean_lower_bound(kernel, "harm") == fraction


@pytest.mark.parametrize("multiplier", [0.5, 1.0, 2.0])
def test_outer_cap_resolves_ranks_before_the_raw_inverse(multiplier):
    law = ComponentAmountLaw(
        component_id="bounded",
        multiplier=multiplier,
        severity=native_severity_from_sampler(
            Distribution(1, 2, 3, 4, 5, gpd_shape=2.5, gpd_scale=1e15, severity_cap=100)
        ),
        rank_group_id="bounded",
    )
    values = sample_component_amounts(
        (law,), np.random.default_rng(1), rank_uniforms={"bounded": 1 - 1e-10}
    )
    assert values[0].amount == 100
