"""Shared capability-evidence contracts: references, instants and periods.

Evidence references name captured bytes by path and digest, or cite a URL or an
explicit fixture.
"""

from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Annotated, Self
from urllib.parse import urlparse

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator

from catmodel.contracts.primitives import (
    RequiredString,
)
from catmodel.dateutil import add_one_year


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("capability timestamps must include a timezone")
    return value.astimezone(UTC)


type UTCInstant = Annotated[datetime, AfterValidator(_utc)]


def _captured_evidence_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or any(character in value for character in "\\#?:")
        or str(path) != value
        or value == "."
    ):
        raise ValueError("captured evidence requires a canonical relative POSIX path")
    return value


def local_capability_evidence(reference: str) -> tuple[str, str] | None:
    """Return ``(path, sha256)`` for an ``evidence_file:`` reference, else None."""
    if not reference.startswith("evidence_file:"):
        return None
    path, separator, digest = reference.removeprefix("evidence_file:").rpartition(
        "#sha256="
    )
    if (
        not separator
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("local evidence reference requires #sha256=<digest>")
    return _captured_evidence_path(path), digest


def _capability_evidence_ref(value: str) -> str:
    if local_capability_evidence(value) is not None:
        return value
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return value
    if value.startswith("urn:fixture:") and value.removeprefix("urn:fixture:"):
        return value
    raise ValueError(
        "evidence reference must be an HTTP(S) URL, a hash-pinned evidence_file, "
        "or an explicit urn:fixture reference"
    )


type CapabilityEvidenceRef = Annotated[
    RequiredString, AfterValidator(_capability_evidence_ref)
]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CapabilityPeriod(_Contract):
    """Five-date annual contract plus a separately identified forecast instant."""

    information_date_utc: UTCInstant
    data_cutoff_utc: UTCInstant
    availability_cutoff_utc: UTCInstant
    policy_period_start_utc: UTCInstant
    policy_period_end_utc: UTCInstant
    forecast_as_of: UTCInstant

    @model_validator(mode="after")
    def coherent_dates(self) -> Self:
        if not (
            self.forecast_as_of
            <= self.data_cutoff_utc
            <= self.availability_cutoff_utc
            <= self.information_date_utc
            <= self.policy_period_start_utc
        ):
            raise ValueError(
                "require as-of <= data <= availability <= information <= start"
            )
        if self.policy_period_end_utc != add_one_year(self.policy_period_start_utc):
            raise ValueError(
                "capability studies require an exact one-year policy window"
            )
        return self
