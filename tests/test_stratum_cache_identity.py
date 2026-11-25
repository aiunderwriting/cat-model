"""A different seed gives the same compiled law distinct, reproducible streams."""

import numpy as np

from catmodel.contracts.incident import ExecutionPlan
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_runtime import simulate_incident_state
from test_support.study import stipulated_study


def test_same_conditional_law_with_a_different_seed_requires_different_streams():
    contract = stipulated_study()
    compiled = compile_incident_state(contract, "state")
    first = ExecutionPlan(
        plan_id="first",
        required_arms=("reference",),
        simulations=128,
        seed=11,
        requested_views=("gross",),
    )
    second = first.model_copy(update={"seed": 12})
    left, _, audit_left = simulate_incident_state(
        first, compiled, response_sha256="a" * 64
    )
    right, _, audit_right = simulate_incident_state(
        second, compiled, response_sha256="a" * 64
    )
    assert first.digest() != second.digest()
    assert left.seed != right.seed
    assert audit_left.stream_seed != audit_right.stream_seed
    assert not np.array_equal(left.annual_losses["gross"], right.annual_losses["gross"])
    repeated, _, _ = simulate_incident_state(second, compiled, response_sha256="a" * 64)
    np.testing.assert_array_equal(
        repeated.annual_losses["gross"], right.annual_losses["gross"]
    )
