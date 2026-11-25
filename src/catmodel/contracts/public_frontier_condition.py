"""General-ECI lab trajectories supplied as public risk conditions."""

from fractions import Fraction
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from catmodel.contracts.capability_evidence import CapabilityEvidenceRef, UTCInstant
from catmodel.contracts.capability_paths import (
    Finite,
    JointCapabilityCoordinate,
    Matrix,
    Vector,
)
from catmodel.contracts.governance import (
    SourceScaleAnchor,
    SourceScaleCorrespondence,
)
from catmodel.contracts.primitives import (
    RequiredIdentifier,
    RequiredString,
    Sha256Digest,
)

SourceModelStatus = Literal["public", "not_yet_public"]
"""Source-model public-deployment status when an incident originates."""


class ObservedFrontier(BaseModel):
    """The highest observed score among the roster's labs on one earlier date."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    at: UTCInstant
    value: Finite


class PublicTrajectoryCondition(BaseModel):
    """One trajectory node as a risk condition, rendered into every question.

    ``trajectory_sha256`` identifies the node's trajectory, not its quadrature
    rule (``PublicFrontierQuadrature.trajectory_sha256``), so a node shared by
    nested rules renders identical prompts in each.

    The path covers the policy period. ``observed_values`` are each lab's
    observed score at ``observed_at``, the forecast's as-of, from which every
    path starts, and ``earlier_frontier`` the roster's highest observed score
    one year before it. With ``scale_anchors``, the models whose fixed scores
    define the scale, they are the reference a question gives for it, and
    they are the same in every node.

    ``scale_correspondence`` is what the score's publisher reports a gain on
    the scale to have corresponded to in another measure; a question that has
    one says what a point means and how many doublings of that measure the
    target's gain along the path would be. A condition without one leaves the
    field out of its record, so an instrument frozen before the field existed
    keeps its digest and its questions' text.

    ``target_lab`` names what the lab's public capability follows: the public
    frontier, the labs' highest value at each moment (``target_path``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    trajectory_sha256: Sha256Digest
    draw_id: RequiredIdentifier
    observed_at: UTCInstant
    observed_values: Vector = Field(min_length=1)
    earlier_frontier: ObservedFrontier | None
    scale_anchors: tuple[SourceScaleAnchor, ...] = ()
    scale_correspondence: SourceScaleCorrespondence | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    times: tuple[UTCInstant, ...] = Field(min_length=2)
    coordinates: tuple[JointCapabilityCoordinate, ...] = Field(min_length=1)
    values: Matrix
    target_lab: Literal["public_frontier"]
    interpolation: Literal["linear"]
    source_model_status: SourceModelStatus
    source_references: tuple[CapabilityEvidenceRef, ...] = Field(min_length=1)
    rationale: RequiredString
    limitations: tuple[RequiredString, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def complete_public_path(self) -> Self:
        if tuple(sorted(set(self.times))) != self.times:
            raise ValueError("public trajectory dates must be unique and ordered")
        if any(row.domain_id != "general" for row in self.coordinates):
            raise ValueError("public trajectories require general ECI lab coordinates")
        labs = {row.lab_id for row in self.coordinates}
        if (
            len(labs) != len(self.coordinates)
            or "public_frontier" in labs
            or len({row.coordinate_id for row in self.coordinates})
            != len(self.coordinates)
            or any(row.access != "public" for row in self.coordinates)
        ):
            raise ValueError(
                "public trajectory coordinates must name distinct public labs"
            )
        if len({row.scale_id for row in self.coordinates}) != 1:
            raise ValueError("public lab trajectories must share one general-ECI scale")
        if len(self.values) != len(self.times) or any(
            len(row) != len(self.coordinates) for row in self.values
        ):
            raise ValueError("public trajectory values must cover every date and lab")
        if any(
            value < previous
            for left, right in zip(self.values, self.values[1:])
            for previous, value in zip(left, right, strict=True)
        ):
            raise ValueError("public capability paths must be nondecreasing")
        if self.observed_at > self.times[0]:
            raise ValueError("the observed scores cannot follow the path's start")
        if len(self.observed_values) != len(self.coordinates) or any(
            value < observed
            for observed, value in zip(
                self.observed_values, self.values[0], strict=False
            )
        ):
            raise ValueError(
                "every lab's path must start at or above its observed score"
            )
        if self.observed_at == self.times[0] and self.observed_values != self.values[0]:
            raise ValueError(
                "a path that starts on the observed date must start at the "
                "observed scores"
            )
        earlier = self.earlier_frontier
        if earlier is not None and (
            earlier.at >= self.observed_at or earlier.value > max(self.observed_values)
        ):
            raise ValueError(
                "the earlier frontier must precede the observed scores and not "
                "exceed them"
            )
        return self

    def target_path(self) -> tuple[tuple[UTCInstant, ...], tuple[float, ...]]:
        """The labs' exact piecewise-linear maximum, including lead switches."""
        points = {at: max(row) for at, row in zip(self.times, self.values, strict=True)}
        for left_at, right_at, left, right in zip(
            self.times, self.times[1:], self.values, self.values[1:]
        ):
            starts = tuple(Fraction(value) for value in left)
            changes = tuple(Fraction(b) - a for a, b in zip(starts, right, strict=True))
            for i in range(len(starts)):
                for j in range(i + 1, len(starts)):
                    difference = changes[i] - changes[j]
                    if difference == 0:
                        continue
                    fraction = (starts[j] - starts[i]) / difference
                    if not 0 < fraction < 1:
                        continue
                    heights = tuple(
                        a + fraction * b for a, b in zip(starts, changes, strict=True)
                    )
                    if heights[i] != max(heights):
                        continue
                    at = left_at + (right_at - left_at) * float(fraction)
                    actual = (at - left_at) / (right_at - left_at)
                    points[at] = max(
                        (1 - actual) * a + actual * b
                        for a, b in zip(left, right, strict=True)
                    )
        times = tuple(sorted(points))
        return times, tuple(points[at] for at in times)
