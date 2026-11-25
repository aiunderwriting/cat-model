"""Independent native amount, view weight and payment oracles."""

from itertools import pairwise

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from pydantic import ValidationError

from catmodel.contracts.components import (
    ComponentAmountLaw,
    ContractTerms,
    NativeLiabilityParameters,
    NativePaymentKernel,
)
from catmodel.contracts.native_severity import (
    NativeSeverityComponent,
    native_severity_from_sampler,
    native_severity_sampler,
)
from catmodel.economic_loss import EconomicNature
from catmodel.probability.distributions import Distribution
from catmodel.simulation.components import component_view_weight
from catmodel.simulation.policy_terms import CoveredParts, PolicyReduction
from test_support.components import sample_component_amounts, sample_native_payments


def test_public_sampler_mutation_cannot_change_authored_component_or_payment_law():
    specification = native_severity_from_sampler(Distribution(100, 100, 100, 100, 100))
    law = ComponentAmountLaw(
        component_id="loss", severity=specification, rank_group_id="severity"
    )
    ranks: dict[str | tuple[str, str], float] = {"severity": 0.5}
    rng = np.random.default_rng(2)
    assert sample_component_amounts((law,), rng, rank_uniforms=ranks)[0].amount == 100
    exposed = native_severity_sampler(specification)
    exposed.severity_cap = 1
    assert native_severity_sampler(specification) is not exposed
    assert sample_component_amounts((law,), rng, rank_uniforms=ranks)[0].amount == 100
    kernel = NativePaymentKernel(
        source_severity=specification,
        rank_group_id="severity",
        entity_id="operator",
        liability_component_ids=("loss",),
        liability=NativeLiabilityParameters(
            mean=0.4,
            concentration=20,
            recovery_rate=1,
            severity_slope=0.05,
            median_severity=100,
            regulatory_recovery_threshold=1e6,
            regulatory_recovery_uplift=0,
        ),
    )
    first = sample_native_payments(
        kernel, np.random.default_rng(8), rank_uniforms=ranks
    )
    exposed.severity_cap = 0.1
    second = sample_native_payments(
        kernel, np.random.default_rng(8), rank_uniforms=ranks
    )
    assert first == second


def _unused_law() -> Distribution:
    """A law the tests below must never evaluate."""
    return Distribution(1, 2, 4, 8, 16)


def test_signed_quantile_knots_are_not_a_native_severity_law():
    with pytest.raises(ValidationError, match="nonnegative"):
        NativeSeverityComponent(p5=-10, p25=-5, p50=0, p75=5, p95=10)


@pytest.mark.parametrize("zero", [{"multiplier": 0}, {"post_multiplier": 0}])
def test_structural_zero_component_skips_unused_severity(zero, monkeypatch):
    specification = native_severity_from_sampler(_unused_law())

    def unused(*args):
        raise AssertionError("structural zero evaluated its unused severity")

    monkeypatch.setattr("catmodel.contracts.native_severity._shared_sampler", unused)
    law = ComponentAmountLaw(
        component_id="zero", severity=specification, rank_group_id="unused", **zero
    )
    assert (
        sample_component_amounts(
            (law,), np.random.default_rng(1), rank_uniforms={"unused": 1}
        )[0].amount
        == 0
    )


def _active_payment_kernel():
    return NativePaymentKernel(
        source_severity=native_severity_from_sampler(
            Distribution(100, 100, 100, 100, 100)
        ),
        rank_group_id="severity",
        entity_id="operator",
        liability_component_ids=("loss",),
        liability=NativeLiabilityParameters(
            mean=0.4,
            concentration=10,
            recovery_rate=0.8,
            severity_slope=0.025,
            median_severity=100,
            regulatory_recovery_threshold=1e6,
            regulatory_recovery_uplift=0,
        ),
    )


@pytest.mark.parametrize("mean", [0.0, 1.0])
def test_endpoint_payment_needs_no_unused_severity_rank(mean, monkeypatch):
    payload = _active_payment_kernel().model_dump(mode="json")
    payload["source_severity"] = native_severity_from_sampler(_unused_law()).model_dump(
        mode="json"
    )
    payload["liability"].update(mean=mean, recovery_rate=mean, severity_slope=0)

    def unused(*args):
        raise AssertionError("endpoint evaluated unused severity")

    monkeypatch.setattr("catmodel.contracts.native_severity._shared_sampler", unused)
    kernel = NativePaymentKernel.model_validate(payload)
    obligations, payments = sample_native_payments(
        kernel, np.random.default_rng(1), rank_uniforms={"severity": 1}
    )
    assert obligations[0].fraction == payments[0].fraction == mean


def test_direct_payment_needs_no_unused_liability_severity_rank(monkeypatch):
    payload = _active_payment_kernel().model_dump(mode="json")
    payload.update(liability_component_ids=[], direct_component_ids=["loss"])
    payload["source_severity"] = native_severity_from_sampler(_unused_law()).model_dump(
        mode="json"
    )

    def unused(*args):
        raise AssertionError("direct payment evaluated unused liability severity")

    monkeypatch.setattr("catmodel.contracts.native_severity._shared_sampler", unused)
    kernel = NativePaymentKernel.model_validate(payload)
    _, payments = sample_native_payments(
        kernel, np.random.default_rng(1), rank_uniforms={"severity": 1}
    )
    assert payments[0].fraction == 1


def test_absent_liability_needs_no_unused_severity_source(monkeypatch):
    payload = _active_payment_kernel().model_dump(mode="json")
    payload["liability"] = None
    payload["source_severity"] = native_severity_from_sampler(_unused_law()).model_dump(
        mode="json"
    )

    def unused(*args):
        raise AssertionError("absent liability evaluated unused severity")

    monkeypatch.setattr("catmodel.contracts.native_severity._shared_sampler", unused)
    kernel = NativePaymentKernel.model_validate(payload)
    _, payments = sample_native_payments(
        kernel, np.random.default_rng(1), rank_uniforms={"severity": 1}
    )
    assert payments[0].fraction == 1


def test_contributors_are_not_conserving_but_attribution_is():
    law = ComponentAmountLaw(
        component_id="harm",
        amount=100,
        contributor_ids=("a", "b"),
        attribution_weights={"a": 0.25, "b": 0.75},
    )
    weights = {
        view: component_view_weight(law, view)
        for view in ("gross", "contributor::a", "contributor::b")
    }
    assert weights == dict.fromkeys(weights, 1.0)
    assert (
        component_view_weight(law, "attribution::a")
        + component_view_weight(law, "attribution::b")
        == 1
    )


def test_appropriation_is_welfare_harm_but_not_resource_consumption():
    natures: tuple[EconomicNature, ...] = (
        "resource_cost",
        "injury_valuation",
        "transfer",
        "appropriated_transfer",
    )
    laws = [
        ComponentAmountLaw(component_id="loss", amount=100, nature=nature)
        for nature in natures
    ]
    assert {
        view: sum(100 * component_view_weight(law, view) for law in laws)
        for view in ("gross", "welfare", "net_resource_cost")
    } == {"gross": 400, "welfare": 300, "net_resource_cost": 200}


def test_annual_cap_bounds_payment_of_infinite_mean_native_amount():
    floor, cap = 1.0, 40.0
    raw = Distribution(2, 3, 4, 6, 10, gpd_shape=1.2, severity_minimum=floor)
    with pytest.raises(ValueError, match="mean"):
        raw.analytic_mean()
    law = ComponentAmountLaw(
        component_id="loss",
        severity=native_severity_from_sampler(raw),
        rank_group_id="severity",
    )
    contract = ContractTerms(
        contract_id="policy",
        contract_year="2027",
        entity_id="operator",
        annual_limit=cap,
    )

    def annual_payment(amount: float) -> float:
        """One year of two occurrences, a random amount and a fixed 10."""
        reduction = PolicyReduction(
            (contract,), 1, financial=True, occurrence_maximum=False
        )
        reduction.add(
            np.zeros(2, dtype=np.int64),
            CoveredParts(
                np.array([0, 1]),
                np.full(2, -1),
                np.zeros(2, dtype=np.int64),
                np.zeros(2, dtype=np.int64),
                np.zeros(2, dtype=bool),
                np.array([amount, 10.0]),
                np.zeros(2, dtype=bool),
            ),
            origin="origin",
        )
        return float(reduction.views()["financial"][0])

    # Y=min(X+10,40): split at the law's knots and at the exact cap atom.
    cut = float(raw.cdf(np.array([cap - 10]))[0])
    nodes, weights = leggauss(48)
    average = 0.0
    for lower, upper in pairwise(sorted({0.0, 0.05, 0.25, 0.5, 0.75, 0.95, cut, 1.0})):
        ranks = lower + (nodes + 1) * (upper - lower) / 2
        for rank, weight in zip(ranks, weights, strict=True):
            amount = sample_component_amounts(
                (law,),
                np.random.default_rng(1),
                rank_uniforms={"severity": float(rank)},
            )[0].amount
            payment = annual_payment(amount)
            assert 0 <= payment <= cap
            average += weight * (upper - lower) / 2 * payment
    # The law's closed-form limited mean, independently of the payment path.
    expected = 10 + raw.analytic_mean(severity_cap_override=cap - 10)
    assert average == pytest.approx(expected, rel=1e-10)
