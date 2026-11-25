"""Authored finite conditional kernels, independent of forecast probabilities."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from catmodel.contracts.primitives import RequiredIdentifier

if TYPE_CHECKING:
    from catmodel.contracts.incident import EntityPath
    from catmodel.contracts.run_setup import RunSetup


def _number(value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("conditional weights must be finite JSON numbers")
    return value


type Weight = Annotated[
    float, Field(ge=0, le=1, allow_inf_nan=False), BeforeValidator(_number)
]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConditionalComponent(_Record):
    state_id: RequiredIdentifier
    weight: Annotated[Weight, Field(gt=0)]


class ConditionalKernel(_Record):
    eci: int = Field(strict=True)
    arm_id: RequiredIdentifier
    components: tuple[ConditionalComponent, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def probability_vector(self) -> Self:
        if len({row.state_id for row in self.components}) != len(self.components):
            raise ValueError("conditional components must name distinct states")
        if not math.isclose(
            math.fsum(row.weight for row in self.components),
            1,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("conditional component weights must sum to one")
        return self


class ConditionalContext(_Record):
    context_id: RequiredIdentifier
    lab_id: RequiredIdentifier
    deployment_id: RequiredIdentifier
    entity_id: RequiredIdentifier
    scale_id: RequiredIdentifier
    coordinate_kind: Literal["at_time", "annual_average", "constant_path"]
    at_time: Weight | None = None
    integer_min: int = Field(strict=True)
    integer_max: int = Field(strict=True)
    required_arms: tuple[RequiredIdentifier, ...] = Field(min_length=1)
    kernels: tuple[ConditionalKernel, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def complete_integer_grid(self) -> Self:
        if self.integer_min > self.integer_max:
            raise ValueError("conditional integer range must be nonempty")
        if (self.coordinate_kind == "at_time") != (self.at_time is not None):
            raise ValueError("only at_time coordinates require an explicit time")
        if len(set(self.required_arms)) != len(self.required_arms):
            raise ValueError("conditional required arms must be unique")
        actual = {(row.eci, row.arm_id) for row in self.kernels}
        expected = {
            (eci, arm)
            for eci in range(self.integer_min, self.integer_max + 1)
            for arm in self.required_arms
        }
        if len(actual) != len(self.kernels) or actual != expected:
            raise ValueError(
                "conditional kernels must cover every integer and required arm exactly once"
            )
        return self

    def kernel_at(self, eci: int, arm_id: str) -> ConditionalKernel:
        if isinstance(eci, bool) or not isinstance(eci, int):
            raise ValueError("integer_required: ECI lookup requires an integer")
        if not self.integer_min <= eci <= self.integer_max:
            raise ValueError("out_of_modeled_range: ECI lies outside this context")
        if arm_id not in self.required_arms:
            raise ValueError("unsupported_arm: arm is outside this context")
        return next(
            row for row in self.kernels if row.eci == eci and row.arm_id == arm_id
        )


def _coordinate(path: EntityPath, context: ConditionalContext) -> float:
    if context.coordinate_kind == "annual_average":
        return math.fsum(
            (right.time - left.time)
            * (
                left.value
                if path.interpolation == "step"
                else (left.value / 2 + right.value / 2)
            )
            for left, right in zip(path.knots, path.knots[1:])
        )
    if context.coordinate_kind == "constant_path":
        if len({knot.value for knot in path.knots}) != 1:
            raise ValueError("constant_path conditioning requires a constant path")
        return path.knots[0].value
    time = context.at_time
    assert time is not None
    if time == 1:
        return path.knots[-1].value
    for left, right in zip(path.knots, path.knots[1:]):
        if left.time <= time < right.time:
            if path.interpolation == "step":
                return left.value
            fraction = (time - left.time) / (right.time - left.time)
            return (1 - fraction) * left.value + fraction * right.value
    raise ValueError("conditional coordinate has no path support")


def validate_conditional_contexts(study: RunSetup) -> None:
    """Refuse conditional contexts that do not fit the setup they query: a
    repeated ID, an undeclared lab, deployment, model or arm, or a component
    whose state, response law or entity path does not support the context.
    """
    contexts = study.conditional_contexts
    queried_entities = {row.entity_id for row in contexts}
    if len({row.context_id for row in contexts}) != len(contexts):
        raise ValueError("conditional context identifiers must be unique")
    entities = {row.entity_id: row for row in study.scope.entities}
    states = {row.state_id: row for row in study.joint_states}
    alternatives = {row.alternative_id: row for row in study.alternatives}
    arms = {"reference", *alternatives}
    # The states each arm's kernels read, checked arm by arm below so that a
    # setup that stores its overrides reads each arm once.
    read: dict[str, dict[str, None]] = {}
    for context in contexts:
        lab = entities.get(context.lab_id)
        bindings = [
            row
            for row in study.scope.deployments
            if row.deployment_id == context.deployment_id
        ]
        if lab is None or lab.kind != "lab" or not bindings:
            raise ValueError(
                "conditional context requires a declared lab and deployment"
            )
        matches = (
            bindings
            if context.entity_id == context.lab_id
            else [row for row in bindings if row.model_id == context.entity_id]
        )
        if not matches:
            raise ValueError("conditioned entity must be this lab or deployed model")
        if len(matches) != 1:
            raise ValueError(
                "conditional context requires an unambiguous deployment/model binding"
            )
        model = entities[matches[0].model_id]
        if not (
            model.entity_id in lab.related_entity_ids
            or lab.entity_id in model.related_entity_ids
        ):
            raise ValueError("conditional model must be explicitly related to its lab")
        if not set(context.required_arms) <= arms:
            raise ValueError("conditional context refers to an undeclared arm")
        for kernel in context.kernels:
            for component in kernel.components:
                state = states.get(component.state_id)
                if state is None:
                    raise ValueError("conditional component refers to an absent state")
                read.setdefault(kernel.arm_id, {})[state.state_id] = None
                path = next(
                    (
                        row
                        for row in state.entity_paths
                        if row.entity_id == context.entity_id
                        and row.scale_id == context.scale_id
                    ),
                    None,
                )
                if path is None or not math.isclose(
                    _coordinate(path, context), kernel.eci, rel_tol=0, abs_tol=1e-10
                ):
                    raise ValueError(
                        "conditional ECI coordinate differs from the source entity path"
                    )
    for arm_id, state_ids in read.items():
        for state_id in state_ids:
            effective = {law.origin_id: law for law in states[state_id].laws}
            if arm_id in alternatives:
                effective.update(
                    {
                        law.origin_id: law
                        for law in study.overrides.laws(arm_id, state_id)
                    }
                )
            if any(
                law.entity_id in queried_entities
                and law.response.domain != "local_entity_state"
                for law in effective.values()
            ):
                raise ValueError(
                    "conditional context needs a local response law for every queried entity"
                )
