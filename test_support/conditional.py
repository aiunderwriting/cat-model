"""Stipulated lab and deployment context for conditional-kernel tests."""

import math
from collections.abc import Mapping
from dataclasses import replace

from catmodel.contracts.conditional import (
    ConditionalComponent,
    ConditionalContext,
    ConditionalKernel,
)
from catmodel.contracts.incident import ExecutionPlan
from catmodel.contracts.law_overrides import HeldOverrides
from catmodel.contracts.run_setup import RunSetup
from catmodel.setup.store import required_state_ids
from test_support.run_setup import setup_from_payload, with_execution
from test_support.study import study_payload


def _forecast_weights(
    context: ConditionalContext, weights: Mapping[int, float]
) -> None:
    if any(isinstance(key, bool) or not isinstance(key, int) for key in weights):
        raise ValueError("forecast grid coordinates must be integers")
    if set(weights) != set(range(context.integer_min, context.integer_max + 1)):
        raise ValueError(
            "forecast weights must cover exactly the modeled integer range"
        )
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight < 0
        for weight in weights.values()
    ) or not math.isclose(math.fsum(weights.values()), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("forecast weights must be finite probabilities summing to one")


def materialize_constant_eci_grid(
    setup: RunSetup,
    *,
    context_id: str,
    lab_id: str,
    deployment_id: str,
    entity_id: str,
    scale_id: str,
    integer_min: int,
    integer_max: int,
    forecast_weights: Mapping[int, float],
) -> RunSetup:
    """Evaluate an authored constant-path response at each integer before mixing."""
    if (
        isinstance(integer_min, bool)
        or isinstance(integer_max, bool)
        or not isinstance(integer_min, int)
        or not isinstance(integer_max, int)
        or integer_min > integer_max
    ):
        raise ValueError(
            "constant-path materialization requires an inclusive integer range"
        )
    if setup.conditional_contexts:
        raise ValueError(
            "constant-path materialization requires an unbound authored setup"
        )
    for state in setup.joint_states:
        path = next(
            (row for row in state.entity_paths if row.entity_id == entity_id), None
        )
        if (
            path is None
            or path.scale_id != scale_id
            or path.history_status != "stipulated"
        ):
            raise ValueError(
                "constant-path materialization requires a stipulated entity path"
            )
    arms = (
        "reference",
        *(row.alternative_id for row in setup.alternatives),
    )
    base_weights = {row.state_id: row.weight for row in setup.joint_states}
    weights_by_arm = {
        "reference": base_weights,
        **{
            row.alternative_id: row.state_weights or base_weights
            for row in setup.alternatives
        },
    }
    states, kernels = [], []
    state_ids = {}
    for eci in range(integer_min, integer_max + 1):
        for source in setup.joint_states:
            state_id = f"{context_id}:eci:{eci}:{source.state_id}"
            state_ids[eci, source.state_id] = state_id
            states.append(
                source.model_copy(
                    update={
                        "state_id": state_id,
                        "weight": forecast_weights.get(eci, 0) * source.weight,
                        "entity_paths": tuple(
                            path.model_copy(
                                update={
                                    "knots": tuple(
                                        knot.model_copy(update={"value": eci})
                                        for knot in path.knots
                                    )
                                }
                            )
                            if path.entity_id == entity_id
                            else path
                            for path in source.entity_paths
                        ),
                    }
                )
            )
        for arm in arms:
            kernels.append(
                ConditionalKernel(
                    eci=eci,
                    arm_id=arm,
                    components=tuple(
                        ConditionalComponent(
                            state_id=state_ids[eci, state], weight=weight
                        )
                        for state, weight in weights_by_arm[arm].items()
                        if weight > 0
                    ),
                )
            )
    context = ConditionalContext(
        context_id=context_id,
        lab_id=lab_id,
        deployment_id=deployment_id,
        entity_id=entity_id,
        scale_id=scale_id,
        coordinate_kind="constant_path",
        integer_min=integer_min,
        integer_max=integer_max,
        required_arms=arms,
        kernels=tuple(kernels),
    )
    _forecast_weights(context, forecast_weights)
    alternatives = tuple(
        alternative.model_copy(
            update={
                "state_weights": {
                    state_ids[eci, state]: forecast_weights[eci] * weight
                    for eci in forecast_weights
                    for state, weight in weights_by_arm[
                        alternative.alternative_id
                    ].items()
                },
            }
        )
        for alternative in setup.alternatives
    )
    overrides = HeldOverrides(
        {
            alternative_id: tuple(
                override.model_copy(
                    update={"state_id": state_ids[eci, override.state_id]}
                )
                for eci in forecast_weights
                for override in setup.overrides.of(alternative_id)
            )
            for alternative_id in setup.overrides.arms()
        }
    )
    execution = setup.execution
    if execution.required_state_ids_by_arm:
        execution = ExecutionPlan.model_validate(
            {
                **execution.model_dump(mode="python"),
                "required_state_ids_by_arm": {
                    arm: tuple(
                        state_ids[eci, state]
                        for eci in range(integer_min, integer_max + 1)
                        for state in required_state_ids(execution, arm, base_weights)
                    )
                    for arm in execution.required_state_ids_by_arm
                },
            }
        )
    # RunSetup revalidates the copied states, alternatives and context.
    return replace(
        setup,
        joint_states=tuple(states),
        conditional_contexts=(context,),
        alternatives=alternatives,
        overrides=overrides,
        execution=execution,
    )


def conditional_study(*, lab_loss=True, two_templates=False):
    payload = study_payload()
    payload["entities"][0]["related_entity_ids"] = ["lab"]
    payload["entities"].extend(
        [
            {
                "entity_id": "lab",
                "kind": "lab",
                "definition": "Stipulated lab",
                "display_name": "Stipulated lab",
                "status": "hypothetical",
            },
            {
                "entity_id": "deployment",
                "kind": "deployment",
                "definition": "Internal stipulated use",
                "display_name": "Internal use",
                "status": "hypothetical",
            },
        ]
    )
    payload["deployments"] = [
        {
            "deployment_id": "deployment",
            "model_id": "model",
            "use_population": "Synthetic internal use",
            "tools_and_permissions": "None",
            "human_oversight": "Stipulated",
        }
    ]
    payload["populations"][0]["deployment_ids"] = ["deployment"]
    component = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
        "member_outcomes"
    ][0]["components"][0]
    if lab_loss:
        component["lab_loss_weights"] = {"lab": 0.4}
    if two_templates:
        import copy

        payload["joint_states"][0]["weight"] = 0.25
        second = copy.deepcopy(payload["joint_states"][0])
        second["state_id"] = "second"
        second["epistemic_group_id"] = "alternative"
        second["weight"] = 0.75
        second["laws"][0]["reference_annual_rate"] = 2
        payload["joint_states"].append(second)
    setup = with_execution(
        setup_from_payload(payload),
        ExecutionPlan(
            plan_id="conditional-fixture",
            required_arms=("reference",),
            simulations=1,
            seed=1,
            requested_views=("gross", "welfare", "lab_loss::lab"),
        ),
        setup_id="conditional-fixture",
    )
    expanded = materialize_constant_eci_grid(
        setup,
        context_id="internal",
        lab_id="lab",
        deployment_id="deployment",
        entity_id="model",
        scale_id="own_score",
        integer_min=0,
        integer_max=2,
        forecast_weights={0: 0.5, 1: 0, 2: 0.5},
    )

    return expanded
