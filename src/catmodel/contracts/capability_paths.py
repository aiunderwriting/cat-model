"""Capability-path primitives: finite values, lab coordinates and weighted path draws."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from catmodel.contracts.primitives import RequiredIdentifier, RequiredString
from catmodel.hashing import canonical_json_sha256


def _number(value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("capability coordinates require finite numeric values")
    return value


type Finite = Annotated[float, BeforeValidator(_number), Field(allow_inf_nan=False)]
type Probability = Annotated[Finite, Field(ge=0, le=1)]
type Vector = tuple[Finite, ...]
type Matrix = tuple[Vector, ...]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def digest(self) -> str:
        return canonical_json_sha256(self.model_dump(mode="json"))


class JointCapabilityCoordinate(_Record):
    coordinate_id: RequiredIdentifier
    lab_id: RequiredIdentifier
    domain_id: RequiredIdentifier
    access: Literal["public", "internal"]
    scale_id: RequiredIdentifier
    definition: RequiredString


class JointPathDraw(_Record):
    draw_id: RequiredIdentifier
    weight: Probability
    values: Matrix
