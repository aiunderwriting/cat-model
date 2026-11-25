"""Independent oracles for the meaning and native binding of conditional ECI."""

from copy import deepcopy

import pytest

from test_support.run_setup import setup_from_payload
from test_support.study import study_payload


def _payload():
    payload = study_payload()
    payload["entities"].extend(
        [
            {
                "entity_id": "lab",
                "kind": "lab",
                "related_entity_ids": ["model"],
                "definition": "Hypothetical laboratory owning the fixture model.",
                "display_name": "Fixture lab",
                "status": "hypothetical",
            },
            {
                "entity_id": "deployment",
                "kind": "deployment",
                "definition": "Hypothetical public deployment of the fixture model.",
                "display_name": "Fixture deployment",
                "status": "hypothetical",
            },
        ]
    )
    payload["deployments"] = [
        {
            "deployment_id": "deployment",
            "model_id": "model",
            "use_population": "Fixture users",
            "tools_and_permissions": "No tools",
            "human_oversight": "None",
        }
    ]
    payload["populations"][0]["deployment_ids"] = ["deployment"]
    prototype = payload["joint_states"][0]
    states = []
    for eci, mass in ((40, 0.5), (41, 0), (42, 0.5)):
        state = deepcopy(prototype)
        state["state_id"] = f"eci-{eci}"
        state["weight"] = mass
        for knot in state["entity_paths"][0]["knots"]:
            knot["value"] = eci
        state["laws"][0]["response"]["points"] = [
            {"value": 40, "multiplier": 1},
            {"value": 42, "multiplier": 2},
        ]
        states.append(state)
    payload["joint_states"] = states
    payload["conditional_contexts"] = [
        {
            "context_id": "public-model",
            "lab_id": "lab",
            "deployment_id": "deployment",
            "entity_id": "model",
            "scale_id": "own_score",
            "coordinate_kind": "constant_path",
            "integer_min": 40,
            "integer_max": 42,
            "required_arms": ["reference"],
            "kernels": [
                {
                    "eci": eci,
                    "arm_id": "reference",
                    "components": [{"state_id": f"eci-{eci}", "weight": 1}],
                }
                for eci in range(40, 43)
            ],
        }
    ]
    return payload


def test_zero_forecast_mass_still_has_a_bound_kernel():
    study = setup_from_payload(_payload())
    assert study.joint_states[1].weight == 0
    kernel = study.conditional_contexts[0].kernel_at(41, "reference")
    assert kernel.components[0].state_id == "eci-41"
    assert kernel.components[0].weight == 1


def test_multimodel_deployment_binding_order_cannot_change_the_context():
    payload = _payload()
    payload["entities"].append({**payload["entities"][0], "entity_id": "other-model"})
    payload["deployments"].append(
        {**payload["deployments"][0], "model_id": "other-model"}
    )
    for bindings in (payload["deployments"], list(reversed(payload["deployments"]))):
        candidate = {**payload, "deployments": bindings}
        context = setup_from_payload(candidate).conditional_contexts[0]
        assert context.entity_id == "model"
        assert context.kernel_at(41, "reference").components[0].state_id == "eci-41"


def test_lab_path_rejects_ambiguous_deployment_models_in_either_order():
    payload = _payload()
    payload["entities"].append({**payload["entities"][0], "entity_id": "other-model"})
    payload["deployments"].append(
        {**payload["deployments"][0], "model_id": "other-model"}
    )
    payload["conditional_contexts"][0]["entity_id"] = "lab"
    for state in payload["joint_states"]:
        state["entity_paths"][0]["entity_id"] = "lab"
        state["laws"][0]["entity_id"] = "lab"
    assert (
        setup_from_payload({**payload, "deployments": payload["deployments"][:1]})
        .conditional_contexts[0]
        .entity_id
        == "lab"
    )
    for bindings in (payload["deployments"], list(reversed(payload["deployments"]))):
        with pytest.raises(ValueError, match="unambiguous deployment/model binding"):
            setup_from_payload({**payload, "deployments": bindings})


@pytest.mark.parametrize("value", [True, 41.0, 40.9, "41"])
def test_lookup_never_silently_rounds_or_coerces(value):
    context = setup_from_payload(_payload()).conditional_contexts[0]
    with pytest.raises(ValueError, match="integer_required"):
        context.kernel_at(value, "reference")


@pytest.mark.parametrize("value", [39, 43])
def test_boundary_lookup_does_not_extrapolate(value):
    context = setup_from_payload(_payload()).conditional_contexts[0]
    with pytest.raises(ValueError, match="out_of_modeled_range"):
        context.kernel_at(value, "reference")


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("lab_id", "model", "declared lab"),
        ("deployment_id", "missing", "declared lab and deployment"),
        ("scale_id", "missing", "source entity path"),
        ("entity_id", "deployment", "conditioned entity"),
    ],
)
def test_context_cannot_invent_source_bindings(field, value, match):
    payload = _payload()
    payload["conditional_contexts"][0][field] = value
    with pytest.raises(ValueError, match=match):
        setup_from_payload(payload)


def test_relabelled_source_coordinate_is_rejected():
    payload = _payload()
    payload["conditional_contexts"][0]["kernels"][1]["components"][0]["state_id"] = (
        "eci-40"
    )
    with pytest.raises(ValueError, match="source entity path"):
        setup_from_payload(payload)


def test_current_coordinate_keeps_conditional_future_paths_and_weights():
    payload = _payload()
    states = payload["joint_states"]
    for state, terminal in zip(states, [41, 50, 60], strict=True):
        state["entity_paths"][0]["knots"][0]["value"] = 41
        state["entity_paths"][0]["knots"][1]["value"] = terminal
    context = payload["conditional_contexts"][0]
    context.update(
        coordinate_kind="at_time",
        at_time=0,
        integer_min=41,
        integer_max=41,
        kernels=[
            {
                "eci": 41,
                "arm_id": "reference",
                "components": [
                    {"state_id": "eci-40", "weight": 0.2},
                    {"state_id": "eci-41", "weight": 0.7},
                    {"state_id": "eci-42", "weight": 0.1},
                ],
            }
        ],
    )
    study = setup_from_payload(payload)
    kernel = study.conditional_contexts[0].kernel_at(41, "reference")
    assert [row.weight for row in kernel.components] == [0.2, 0.7, 0.1]
    assert [row.entity_paths[0].knots[-1].value for row in study.joint_states] == [
        41,
        50,
        60,
    ]
    assert study.joint_states[1].weight == 0


def test_annual_average_integrates_the_path_instead_of_averaging_knots():
    payload = _payload()
    for state, eci in zip(payload["joint_states"], range(40, 43), strict=True):
        state["entity_paths"][0]["knots"] = [
            {"time": 0, "value": eci - 1, "model_revision": "r1"},
            {"time": 0.75, "value": eci + 3, "model_revision": "r1"},
            {"time": 1, "value": eci + 10, "model_revision": "r1"},
        ]
    payload["conditional_contexts"][0]["coordinate_kind"] = "annual_average"
    study = setup_from_payload(payload)
    assert study.conditional_contexts[0].kernel_at(42, "reference").eci == 42
    payload["conditional_contexts"][0]["coordinate_kind"] = "constant_path"
    with pytest.raises(ValueError, match="constant path"):
        setup_from_payload(payload)


def test_missing_zero_prior_interior_kernel_fails_completeness():
    payload = _payload()
    del payload["conditional_contexts"][0]["kernels"][1]
    with pytest.raises(ValueError, match="every integer"):
        setup_from_payload(payload)
