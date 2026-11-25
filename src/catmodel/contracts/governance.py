"""Source-bundle governance records: external sources, checksums and assumptions."""

from collections.abc import Sequence
from datetime import date
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from catmodel.assumption_vocabulary import (
    AssumptionMateriality,
    AssumptionStatus,
    SourceClassification,
)
from catmodel.contracts.primitives import (
    OptionalString,
    RequiredIdentifier,
    RequiredString,
    Sha256Digest,
    UniqueRequiredStringList,
)

AssumptionUncertaintyType = Literal["epistemic", "aleatory", "mixed"]


class FileChecksum(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sha256: Sha256Digest
    role: OptionalString = None


class SourceScaleAnchor(BaseModel):
    """A record to which the source assigns a fixed value to define its scale.

    The value is a convention, not a measurement, so it has no measurement
    error. Matching requires the record's name, organization and value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: RequiredString
    organization: RequiredString
    value: float = Field(allow_inf_nan=False)


class SourceScaleCorrespondence(BaseModel):
    """What the source's publisher reports a gain on its scale to have
    corresponded to in another measure.

    It is the publisher's observation about its scale, not a property the
    scale guarantees: ``observed`` says when or on what it was made, and
    ``measure`` names the other measure and says what it is. A question
    quotes it so that a reader can tell what a difference in the score means.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    measure: RequiredString
    points_per_doubling: float = Field(gt=0, allow_inf_nan=False)
    observed: RequiredString


class ExternalSourceSnapshot(BaseModel):
    """Frozen identity and use provenance for one external source.

    An empty ``transformations`` list records that none were applied; ``None``
    that the transformation history was not recorded.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: RequiredIdentifier
    provider: OptionalString = None
    uri: RequiredString
    # Deliberately opaque: upstream sources can use qualified or non-SHA hashes.
    content_hash: RequiredString
    pulled_at: OptionalString = None
    source_vintage: OptionalString = None
    availability_date: OptionalString = None
    record_count: int | None = Field(default=None, ge=0)
    date_range: OptionalString = None
    filters: OptionalString = None
    transformations: UniqueRequiredStringList | None = None
    loader: OptionalString = None
    loader_code_version: OptionalString = None
    qa_status: OptionalString = None
    license_access_basis: OptionalString = None
    redistribution_constraints: OptionalString = None
    captured_paths: UniqueRequiredStringList = Field(default_factory=list)
    # Given together with scale_anchor_basis, the citation for the convention.
    scale_anchors: list[SourceScaleAnchor] | None = Field(default=None, min_length=1)
    scale_anchor_basis: OptionalString = None
    # Given together with scale_correspondence_basis, its citation. Both are
    # left out of a record that has neither, so a manifest written without
    # them is the one written before they existed.
    scale_correspondence: SourceScaleCorrespondence | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scale_correspondence_basis: OptionalString = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_scale_anchor_basis(self) -> Self:
        if (self.scale_anchors is None) != (self.scale_anchor_basis is None):
            raise ValueError("scale_anchors and scale_anchor_basis are given together")
        if (self.scale_correspondence is None) != (
            self.scale_correspondence_basis is None
        ):
            raise ValueError(
                "scale_correspondence and scale_correspondence_basis are given together"
            )
        return self

    @field_validator("availability_date")
    @classmethod
    def validate_availability_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                "availability_date must be an ISO calendar date (YYYY-MM-DD)"
            ) from exc
        if parsed.isoformat() != value:
            raise ValueError(
                "availability_date must be an ISO calendar date (YYYY-MM-DD)"
            )
        return value

    @field_validator("captured_paths")
    @classmethod
    def validate_captured_paths(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            value = value.replace("\\", "/")
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("captured_paths must be relative paths without '..'")
            if path == PurePosixPath(".") or value.endswith("/"):
                raise ValueError("captured_paths must identify files")
            normalized.append(str(path))
        if len(normalized) != len(set(normalized)):
            raise ValueError("captured_paths must be unique after normalization")
        return normalized


class AssumptionRecord(BaseModel):
    """One source-bundle assumption with its rationale, grounding and range.

    ``sensitivity_tested`` claims that the stated range was executed; the claim
    must cite that execution evidence in ``sensitivity_evidence_refs``.
    """

    model_config = ConfigDict(extra="forbid")

    assumption_id: RequiredIdentifier
    title: RequiredString
    applies_to: list[RequiredString] = Field(default_factory=list)
    source_classification: SourceClassification
    status: AssumptionStatus
    uncertainty_type: AssumptionUncertaintyType
    materiality: AssumptionMateriality
    rationale: RequiredString
    empirical_grounding: RequiredString
    sensitivity_range: RequiredString
    sensitivity_tested: bool
    evidence_refs: list[RequiredString] = Field(default_factory=list)
    sensitivity_evidence_refs: list[RequiredString] = Field(default_factory=list)

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("uncertainty_type", "materiality", mode="before")
    @classmethod
    def normalize_lowercase_enums(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_sensitivity_evidence(self) -> Self:
        if self.sensitivity_tested and not self.sensitivity_evidence_refs:
            raise ValueError(
                f"assumption {self.assumption_id!r}: sensitivity_tested=true "
                "requires sensitivity_evidence_refs citing the executed "
                "sensitivity evidence"
            )
        return self


def validate_unique_assumption_ids(
    assumptions: Sequence[AssumptionRecord],
) -> Sequence[AssumptionRecord]:
    ids = [record.assumption_id for record in assumptions]
    if len(set(ids)) != len(ids):
        raise ValueError("assumption register contains duplicate assumption_id values")
    return assumptions


class AssumptionRegisterBase(BaseModel):
    """A register of assumptions; an empty list states that there are none."""

    model_config = ConfigDict(extra="forbid")

    assumptions: list[AssumptionRecord]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Self:
        validate_unique_assumption_ids(self.assumptions)
        return self
