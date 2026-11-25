"""Finite joint cluster laws retain member dependence rather than just marginals."""

from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from catmodel.contracts.incident import ExecutionPlan, MarkKernel
from catmodel.results.execution import simulate_incident_state
from catmodel.simulation.incident import compile_incident_state
from test_support.run_setup import setup_from_payload
from test_support.study import study_payload


def _joint_kernel():
    kernel = deepcopy(
        study_payload()["joint_states"][0]["laws"][0]["segments"][0]["kernel"]
    )
    prototype = kernel.pop("member_outcomes")[0]
    kernel.pop("cluster")
    low = deepcopy(prototype)
    low["components"][0].update(amount=10, bearer_id="left")
    high = deepcopy(prototype)
    high["components"][0].update(amount=100, bearer_id="right")
    swapped_high = deepcopy(high)
    swapped_high["components"][0]["bearer_id"] = "left"
    swapped_low = deepcopy(low)
    swapped_low["components"][0]["bearer_id"] = "right"
    kernel["explicit_joint_members"] = [
        {"probability": 0.5, "members": [low, high]},
        {"probability": 0.5, "members": [swapped_high, swapped_low]},
    ]
    return kernel


def test_joint_cluster_uses_its_sole_count_authority() -> None:
    kernel = _joint_kernel()
    model = MarkKernel.model_validate(kernel)
    assert model.expected_member_count() == 2
    assert len(model.possible_member_outcomes()) == 4
    kernel["cluster"] = {
        "family": "finite",
        "support": [{"count": 2, "probability": 1}],
    }
    with pytest.raises(ValidationError, match="sole member count authority"):
        MarkKernel.model_validate(kernel)


def test_joint_cluster_preserves_negative_member_dependence() -> None:
    payload = study_payload()
    payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"] = _joint_kernel()
    study = setup_from_payload(payload)
    plan = ExecutionPlan(
        plan_id="joint",
        required_arms=("reference",),
        simulations=12000,
        seed=173,
        requested_views=("gross", "bearer::left", "bearer::right"),
    )
    draw, unavailable, allocation = simulate_incident_state(
        plan, compile_incident_state(study, "state"), response_sha256="a" * 64
    )
    assert not unavailable
    gross = draw.annual_losses["gross"]
    # The complete physical campaign mark is always 110. Independent member
    # resampling would introduce impossible campaign totals 20 and 200.
    assert np.all(gross == np.rint(gross / 110) * 110)
    total_initiators = allocation.total_initiators
    assert isinstance(total_initiators, int)
    assert gross.sum() == 110 * total_initiators
    np.testing.assert_array_equal(
        gross, draw.annual_losses["bearer::left"] + draw.annual_losses["bearer::right"]
    )
    assert np.mean(draw.annual_losses["bearer::left"]) == pytest.approx(55, abs=2)
    assert np.mean(draw.annual_losses["bearer::right"]) == pytest.approx(55, abs=2)
    # Exact branch oracle: Cov(X,Y | one campaign)=1000-55^2=-2025.
    assert (10 * 100 + 100 * 10) / 2 - 55**2 == -2025


def test_joint_branch_response_cost_survives_perfect_prevention_once() -> None:
    payload = study_payload()
    kernel = _joint_kernel()
    kernel["unit_probability"] = 0
    for branch, amount in zip(kernel["explicit_joint_members"], (10, 20)):
        branch["campaign_components"] = [
            {
                "component_id": "response",
                "amount": amount,
                "bearer_id": "operator",
                "nature": "resource_cost",
                "scenario_ids": ["scenario"],
            }
        ]
    payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"] = kernel
    study = setup_from_payload(payload)
    plan = ExecutionPlan(
        plan_id="joint_cost",
        required_arms=("reference",),
        simulations=12000,
        seed=371,
        requested_views=("gross",),
    )
    draw, unavailable, allocation = simulate_incident_state(
        plan, compile_incident_state(study, "state"), response_sha256="a" * 64
    )
    assert not unavailable
    assert np.mean(draw.annual_losses["gross"]) == pytest.approx(15, abs=0.6)
    total_initiators = allocation.total_initiators
    assert isinstance(total_initiators, int)
    assert (
        10 * total_initiators
        <= draw.annual_losses["gross"].sum()
        <= 20 * total_initiators
    )


def test_zero_weight_joint_branch_does_not_become_a_member_authority() -> None:
    kernel = _joint_kernel()
    kernel["explicit_joint_members"].append(
        {
            "probability": 0,
            "members": [deepcopy(kernel["explicit_joint_members"][0]["members"][0])]
            * 300,
        }
    )
    model = MarkKernel.model_validate(kernel)
    assert model.expected_member_count() == 2
    assert len(model.possible_member_outcomes()) == 4


@pytest.mark.parametrize("changed_basis", ["currency", "monetary_basis"])
def test_mixed_bases_withhold_only_requested_pooled_views(changed_basis) -> None:
    payload = study_payload()
    second_population = deepcopy(payload["populations"][0])
    second_population["scenario_id"] = "other"
    second_population[changed_basis] = {
        "currency": "EUR",
        "monetary_basis": "Different fixed valuation date",
    }[changed_basis]
    payload["populations"].append(second_population)
    second_law = deepcopy(payload["joint_states"][0]["laws"][0])
    second_law.update(origin_id="other_origin", scenario_id="other")
    second_law["segments"][0]["kernel"]["member_outcomes"][0]["components"][0][
        "scenario_ids"
    ] = ["other"]
    payload["joint_states"][0]["laws"].append(second_law)
    study = setup_from_payload(payload)
    compiled = compile_incident_state(study, "state")
    assert "pooled" in compiled.unavailable_views
    assert (
        compiled.view_bases["scenario::other"]["currency"]
        == second_population["currency"]
    )
    plan = ExecutionPlan(
        plan_id="scoped",
        required_arms=("reference",),
        simulations=500,
        seed=173,
        requested_views=(
            "gross",
            "scenario::scenario::gross",
            "scenario::other::gross",
        ),
    )
    draw, unavailable, _ = simulate_incident_state(
        plan, compiled, response_sha256="a" * 64
    )
    assert set(draw.annual_losses) == {
        "scenario::scenario::gross",
        "scenario::other::gross",
    }
    assert "gross" in unavailable
    assert np.mean(draw.annual_losses["scenario::other::gross"]) > 0
