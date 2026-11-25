"""Compiled incident laws own every binding consumed during execution."""

from dataclasses import replace

import numpy as np
import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_runtime import simulate_incident_state
from test_support.run_setup import setup_from_payload
from test_support.study import _financial_study, _plan


def test_changed_contract_requires_a_new_compiled_law_for_direct_execution():
    study = _financial_study()
    original = compile_incident_state(study, "low")
    payload = encode_scientific(study)
    payload["contracts"][0]["occurrence_limit"] = 0
    changed = compile_incident_state(setup_from_payload(payload), "low")
    assert original.conditional_key != changed.conditional_key
    assert original.contracts[0].occurrence_limit == 50
    assert changed.contracts[0].occurrence_limit == 0
    plan = _plan(32).model_copy(update={"requested_views": ("gross", "financial")})
    first, _, first_execution = simulate_incident_state(
        plan, original, response_sha256="a" * 64
    )
    second, _, second_execution = simulate_incident_state(
        plan, changed, response_sha256="a" * 64
    )
    assert first_execution.compiled_law_sha256 != second_execution.compiled_law_sha256
    np.testing.assert_array_equal(
        first.annual_losses["gross"], second.annual_losses["gross"]
    )
    assert np.sum(first.annual_losses["financial"]) > 0
    assert np.sum(second.annual_losses["financial"]) == 0
    repeated, _, _ = simulate_incident_state(plan, original, response_sha256="a" * 64)
    np.testing.assert_array_equal(
        first.annual_losses["financial"], repeated.annual_losses["financial"]
    )


def test_compiled_role_registry_is_fixed_and_editorial_title_preserves_science():
    study = _financial_study()
    original = compile_incident_state(study, "low")
    editorial = compile_incident_state(
        replace(
            study,
            scope=study.scope.model_copy(update={"title": "Edited display title"}),
        ),
        "low",
    )
    assert original.conditional_key == editorial.conditional_key
    assert original.contracts == editorial.contracts
    assert original.declared_view_ids == editorial.declared_view_ids
    with pytest.raises(TypeError):
        original.declared_view_ids["bearer"] = frozenset({"invented"})  # ty: ignore[invalid-assignment]


def test_every_state_of_a_setup_declares_the_views_of_its_own_laws():
    from catmodel.simulation.incident import declared_view_identifiers

    study = _financial_study()
    states = [compile_incident_state(study, row.state_id) for row in study.joint_states]
    assert all(
        state.declared_view_ids == declared_view_identifiers(study) for state in states
    )
    # Another setup object, even of the same laws, reads its own.
    copy = setup_from_payload(encode_scientific(study))
    assert compile_incident_state(copy, "low").declared_view_ids == (
        declared_view_identifiers(copy)
    )
