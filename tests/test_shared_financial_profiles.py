"""Scenario expense marks share policy terms without multiplying annual limits."""

import numpy as np
import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.probability.distributions import Distribution
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_runtime import simulate_incident_state
from test_support.incident import common_case as _case
from test_support.run_setup import setup_from_payload
from test_support.study import _financial_study, _plan


def test_native_financial_marks_retain_component_expense_loading():
    study, plan, _compiled, _counts = _case(
        sampler=Distribution(40, 40, 40, 40, 40), financial=True, simulations=32
    )
    payload = encode_scientific(study)
    payload["contracts"][0].update(
        occurrence_retention=0, occurrence_limit=50, annual_retention=0, annual_limit=85
    )
    for state in payload["joint_states"]:
        membership = state["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0][
            "memberships"
        ][0]
        membership.update(defense_loading=0.2, defense_outside_limits=True)
    study = setup_from_payload(payload)
    result, unavailable, _ = simulate_incident_state(
        plan.model_copy(
            update={"requested_views": ("gross", "financial", "oep_financial")}
        ),
        compile_incident_state(study, "low"),
        response_sha256="a" * 64,
    )
    assert not unavailable
    np.testing.assert_array_equal(result.annual_losses["financial"], np.full(32, 85))
    np.testing.assert_array_equal(
        result.annual_losses["oep_financial"], np.full(32, 48)
    )


@pytest.mark.parametrize("outside,finite_variance", [(False, True), (True, False)])
def test_outside_component_expense_removes_an_occurrence_variance_bound(
    outside, finite_variance
):
    payload = encode_scientific(_financial_study(shape=0.6))
    for state in payload["joint_states"]:
        membership = state["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0][
            "memberships"
        ][0]
        membership.update(defense_loading=0.2, defense_outside_limits=outside)
    study = setup_from_payload(payload)
    plan = _plan(128).model_copy(
        update={"required_arms": ("reference",), "requested_views": ("financial",)}
    )
    draw, _, _ = simulate_incident_state(
        plan, compile_incident_state(study, "low"), response_sha256="a" * 64
    )
    assert draw.views["financial"].mean_exists
    assert draw.views["financial"].variance_exists is finite_variance
