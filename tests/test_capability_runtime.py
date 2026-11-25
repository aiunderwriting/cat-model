"""Physical Poisson oracles under an explicitly weighted joint-state law."""

from copy import deepcopy

import numpy as np
import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.simulation.capability_mixture import CapabilityMixture
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_runtime import simulate_incident_state
from test_support.run_setup import setup_from_payload
from test_support.study import _plan, _study


def setup_runtime(*, shared=False, simulations=4_000):
    payload = encode_scientific(_study())
    payload["alternatives"] = []
    population = payload["populations"][0]
    population["scenario_id"] = "one"
    if not shared:
        payload["populations"].append({**population, "scenario_id": "two"})
    states = []
    for name, probability, weight in (
        ("low", 0.1, 0.5),
        ("high", 0.9, 0.5),
        ("zero_mass", 0.5, 0),
    ):
        state = deepcopy(payload["joint_states"][0])
        state.update(state_id=name, epistemic_group_id=name, weight=weight)
        law = state["laws"][0]
        law.update(
            origin_id="one",
            scenario_id="one",
            reference_annual_rate=-np.log1p(-probability),
        )
        kernel = law["segments"][0]["kernel"]
        kernel["cluster"]["support"] = [{"count": 1, "probability": 1}]
        component = kernel["member_outcomes"][0]["components"][0]
        component.update(amount=1, scenario_ids=["one"])
        if shared:
            component["contributor_ids"] = ["provider-a", "provider-b"]
        else:
            other = deepcopy(law)
            other.update(origin_id="two", scenario_id="two")
            other["segments"][0]["kernel"]["member_outcomes"][0]["components"][0][
                "scenario_ids"
            ] = ["two"]
            state["laws"].append(other)
        states.append(state)
    payload["joint_states"] = states
    study = setup_from_payload(payload)
    plan = _plan(simulations).model_copy(
        update={
            "required_arms": ("reference",),
            "requested_views": (
                "gross",
                "welfare",
                "net_resource_cost",
                "provider_contribution",
                "scenario::one::gross",
                *(("scenario::two::gross",) if not shared else ()),
            ),
        }
    )
    compiled = {
        state.state_id: compile_incident_state(study, state.state_id)
        for state in study.joint_states
    }
    return study, plan, compiled


def _runs(study, plan, compiled):
    return {
        key: simulate_incident_state(plan, state, response_sha256="a" * 64)[0]
        for key, state in compiled.items()
    }


def _mixture(runs, weights):
    return CapabilityMixture(
        {draw.stratum_sha256: draw for draw in runs.values()},
        {runs[name].stratum_sha256: weight for name, weight in weights.items()},
    )


def _positive_probabilities(mixture):
    strata = mixture.positive_strata()
    marginals = np.zeros(2)
    joint = 0.0
    for draw, weight in strata:
        one = draw.annual_losses["scenario::one::gross"] > 0
        two = draw.annual_losses["scenario::two::gross"] > 0
        marginals += weight * np.array([one.mean(), two.mean()])
        joint += weight * np.mean(one & two)
    return marginals, joint


def _probability_tolerance(probabilities, weights, simulations):
    # The conditional Bernoulli laws give an independent standard-error oracle.
    # A few thousand years distinguish the .41 joint probability from .25 while
    # avoiding tens of thousands of Python-level physical-mark evaluations.
    variance = sum(
        weight**2 * probability * (1 - probability) / simulations
        for probability, weight in zip(probabilities, weights, strict=True)
    )
    return 6 * np.sqrt(variance)


def test_native_joint_state_mixture_reproduces_dependence_and_accounting_views():
    study, plan, compiled = setup_runtime()
    runs = _runs(study, plan, compiled)
    mixture = _mixture(runs, {"low": 0.5, "high": 0.5, "zero_mass": 0})
    # Independent conditional Poisson counts have positive probabilities .1/.9.
    # Mixing their complete states gives joint .5*(.1**2+.9**2), not .5**2.
    marginals, joint = _positive_probabilities(mixture)
    assert marginals == pytest.approx(
        [0.5, 0.5], abs=_probability_tolerance((0.1, 0.9), (0.5, 0.5), plan.simulations)
    )
    assert joint == pytest.approx(
        0.41, abs=_probability_tolerance((0.01, 0.81), (0.5, 0.5), plan.simulations)
    )
    gross = mixture.metrics("gross")
    expected = -np.log(0.9) - np.log(0.1)
    assert gross["expected_annual_loss"] == pytest.approx(expected)
    contribution = mixture.metrics("provider_contribution")
    assert contribution["expected_annual_loss"] == gross["expected_annual_loss"]
    assert contribution["eal_precision"] == gross["eal_precision"]
    for draw in runs.values():
        for view in ("welfare", "net_resource_cost", "provider_contribution"):
            np.testing.assert_array_equal(
                draw.annual_losses["gross"], draw.annual_losses[view]
            )
        assert not draw.annual_losses["gross"].flags.writeable


def test_fixed_joint_state_removes_mixture_induced_dependence():
    _unused_study, plan, compiled = setup_runtime()
    draw = simulate_incident_state(plan, compiled["high"], response_sha256="a" * 64)[0]
    mixture = _mixture({"high": draw}, {"high": 1})
    marginals, joint = _positive_probabilities(mixture)
    assert marginals == pytest.approx(
        [0.9, 0.9], abs=_probability_tolerance((0.9,), (1,), plan.simulations)
    )
    assert joint == pytest.approx(
        0.81, abs=_probability_tolerance((0.81,), (1,), plan.simulations)
    )


def test_one_physical_cohort_is_counted_once_before_contributor_views():
    _unused_study, plan, compiled = setup_runtime(shared=True, simulations=4_000)
    draw, unavailable, allocation = simulate_incident_state(
        plan, compiled["zero_mass"], response_sha256="a" * 64
    )
    assert not unavailable
    gross = draw.annual_losses["gross"]
    contribution = draw.annual_losses["provider_contribution"]
    np.testing.assert_array_equal(contribution, 2 * gross)
    assert gross.sum() == allocation.total_initiators
    assert gross.mean() == pytest.approx(
        np.log(2), abs=6 * np.sqrt(np.log(2) / plan.simulations)
    )
    assert any(view.native_mean is not None for view in draw.views.values())
    assert draw.views["gross"].native_mean is not None
    assert draw.views["gross"].native_mean.value == pytest.approx(np.log(2))
    assert draw.views["provider_contribution"].native_mean is not None
    assert draw.views["provider_contribution"].native_mean.value == pytest.approx(
        2 * np.log(2)
    )
