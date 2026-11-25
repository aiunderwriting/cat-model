"""Pre-recovery lab costs use an explicit allocation of unique loss components."""

import numpy as np
import pytest
from pydantic import ValidationError

from catmodel.contracts.components import ComponentAmountLaw
from catmodel.economic_loss import LAB_LOSS_BASIS
from catmodel.simulation import incident_runtime as runtime
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_linear import mark_projection
from catmodel.simulation.incident_runtime import simulate_incident_state
from test_support.run_setup import setup_from_payload
from test_support.study import _plan, study_payload

LAB = "lab_loss::lab"


def _study_payload(weights):
    payload = study_payload()
    payload["entities"].append(
        {
            "entity_id": "lab",
            "kind": "lab",
            "definition": "Named hypothetical lab before recoveries",
            "display_name": "Lab",
            "status": "hypothetical",
        }
    )
    component = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
        "member_outcomes"
    ][0]["components"][0]
    component["lab_loss_weights"] = weights
    return payload


def _simulate(payload):
    study = setup_from_payload(payload)
    plan = _plan(16).model_copy(
        update={
            "required_arms": ("reference",),
            "requested_views": ("welfare", "lab_loss::lab"),
        }
    )
    state = study.joint_states[0].state_id
    return simulate_incident_state(
        plan, compile_incident_state(study, state), response_sha256="a" * 64
    )


def test_lab_and_society_arrays_and_native_means_remain_matched():
    payload = _study_payload({"lab": 0.5})
    payload["joint_states"][0]["laws"][0]["reference_annual_rate"] = 100
    draw, unavailable, allocation = _simulate(payload)
    assert not unavailable
    np.testing.assert_allclose(
        draw.annual_losses["lab_loss::lab"], draw.annual_losses["welfare"] * 0.5
    )
    native = draw.views["lab_loss::lab"].native_mean
    assert native is not None
    assert native.value == pytest.approx(500)
    assert (
        allocation.view_bases["lab_loss::lab"]["accounting_standing"] == LAB_LOSS_BASIS
    )


def test_missing_map_unavailable_even_when_no_event_is_sampled():
    payload = _study_payload(None)
    payload["joint_states"][0]["laws"][0]["reference_annual_rate"] = 1e-20
    draw, unavailable, _ = _simulate(payload)
    assert np.count_nonzero(draw.annual_losses["welfare"]) == 0
    assert "lab_loss::lab" in unavailable
    assert "lab_loss::lab" not in draw.views


def test_explicit_empty_map_is_known_zero():
    draw, unavailable, _ = _simulate(_study_payload({}))
    assert not unavailable
    np.testing.assert_array_equal(draw.annual_losses["lab_loss::lab"], np.zeros(16))
    native = draw.views["lab_loss::lab"].native_mean
    assert native is not None and native.value == 0
    assert native.precision["known_zero_law"]


@pytest.mark.parametrize("entity", ["missing", "model"])
def test_lab_map_keys_must_be_declared_labs(entity):
    with pytest.raises(ValidationError, match="declared lab entities"):
        setup_from_payload(_study_payload({entity: 1}))


def test_absent_lab_map_preserves_historical_serialization():
    component = ComponentAmountLaw(component_id="cost", amount=100)
    assert "lab_loss_weights" not in component.model_dump(mode="json")
    explicit = component.model_copy(update={"lab_loss_weights": {}})
    assert explicit.model_dump(mode="json")["lab_loss_weights"] == {}


def test_unsampled_positive_probability_branch_still_needs_lab_map():
    payload = _study_payload({"lab": 0.5})
    kernel = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"]
    first = kernel["member_outcomes"][0]
    first["probability"] = 1 - 1e-12
    kernel["member_outcomes"].append(
        {
            "probability": 1e-12,
            "components": [{**first["components"][0], "lab_loss_weights": None}],
        }
    )
    _, unavailable, _ = _simulate(payload)
    assert "lab_loss::lab" in unavailable


def test_zero_lab_allocation_does_not_inherit_unrelated_heavy_tail_precision():
    payload = _study_payload({})
    component = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
        "member_outcomes"
    ][0]["components"][0]
    component.pop("amount")
    component.update(
        severity={
            "representation": "distribution",
            "components": [
                {
                    "p5": 1,
                    "p25": 2,
                    "p50": 3,
                    "p75": 4,
                    "p95": 5,
                    "gpd_shape": 0.75,
                    "gpd_scale": 1,
                }
            ],
            "weights": [1],
        },
        rank_group_id="tail",
    )
    draw, unavailable, _ = _simulate(payload)
    assert not unavailable
    assert not draw.views["welfare"].variance_exists
    assert draw.views["lab_loss::lab"].variance_exists


def test_unrequested_lab_observable_keeps_existing_capture_basis_shape():
    payload = _study_payload(None)
    compiled = compile_incident_state(setup_from_payload(payload), "state")
    plan = _plan(16).model_copy(update={"requested_views": ("welfare",)})
    _, _, execution = simulate_incident_state(plan, compiled, response_sha256="a" * 64)
    bases = execution.view_bases
    assert isinstance(bases, dict)
    assert set(bases) == {"scenario::scenario", "pooled"}


def test_mixed_outcome_lab_tail_keeps_infinite_variance_applicability():
    payload = _study_payload({})
    kernel = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"]
    light = kernel["member_outcomes"][0]
    light["probability"] = 0.5
    kernel["member_outcomes"].append(
        {
            "probability": 0.5,
            "components": [
                {
                    "component_id": "lab_tail",
                    "nature": "resource_cost",
                    "lab_loss_weights": {"lab": 1},
                    "rank_group_id": "lab_tail",
                    "severity": {
                        "representation": "distribution",
                        "components": [
                            {
                                "p5": 1,
                                "p25": 2,
                                "p50": 3,
                                "p75": 4,
                                "p95": 5,
                                "gpd_shape": 0.75,
                                "gpd_scale": 1,
                            }
                        ],
                        "weights": [1],
                    },
                }
            ],
        }
    )
    draw, unavailable, _ = _simulate(payload)
    assert not unavailable
    assert draw.views["lab_loss::lab"].mean_exists
    assert not draw.views["lab_loss::lab"].variance_exists


def _projected(laws, view):
    projection = mark_projection(laws, (), (), (view,), scenario_id="loss")
    assert projection is not None
    return sum(
        law.amount * weight
        for law, weight in zip(laws, projection.weights[view], strict=True)
    )


def test_lab_cost_is_its_allocation_whoever_bears_and_pays_it():
    cost = ComponentAmountLaw(
        component_id="cost",
        amount=100,
        bearer_id="lab",
        nature="resource_cost",
        lab_loss_weights={"lab": 0.6},
    )
    assert [_projected((cost,), view) for view in ("gross", "welfare", LAB)] == [
        100,
        100,
        60,
    ]


def test_fine_transfer_and_defense_are_separate_from_societal_harm():
    laws = (
        ComponentAmountLaw(
            component_id="injury",
            amount=100,
            nature="injury_valuation",
            lab_loss_weights={"lab": 0.5},
        ),
        ComponentAmountLaw(
            component_id="fine",
            amount=20,
            nature="transfer",
            lab_loss_weights={"lab": 1},
        ),
        ComponentAmountLaw(
            component_id="defense",
            amount=5,
            nature="resource_cost",
            lab_loss_weights={"lab": 1},
        ),
    )
    assert _projected(laws, "welfare") == 105
    assert _projected(laws, LAB) == 75


def test_incomplete_component_composition_cannot_be_complete_lab_loss():
    shape = runtime._MarkShape(
        components=(
            (
                ComponentAmountLaw(component_id="known", amount=1, lab_loss_weights={}),
                ("loss",),
            ),
        ),
        identified=True,
        complete=False,
        lab_ids=frozenset(),
        mapping_complete=False,
    )
    values, unavailable = shape.evaluation(financial=False)
    assert "gross" in values
    assert "lab_loss" in unavailable and "welfare" in unavailable
