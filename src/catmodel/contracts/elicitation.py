"""Shared contracts for elicited judgments: respondents and quantiles."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from catmodel.contracts.primitives import (
    RequiredIdentifier,
    RequiredString,
)
from catmodel.probability.distributions import ADMITTED_AMOUNT_RANGE
from catmodel.validation_errors import EvidenceValidationError

QUANTILES = ("p5", "p25", "p50", "p75", "p95")
"""The five quantile levels in increasing order."""

ELICITATION_ORDER = ("p50", "p25", "p75", "p5", "p95")
"""The order in which questions ask the five quantiles and ``Quantiles``
declares them: the median first, then outward."""


def _number(value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceValidationError(
            "finite_json_number_required",
            "quantities must be finite JSON numbers, not strings or booleans",
        )
    return value


# ``Field`` precedes ``BeforeValidator`` so the answer schema shown to
# respondents carries standard JSON Schema ``minimum``/``maximum`` keywords.
type Nonnegative = Annotated[
    float, Field(ge=0, allow_inf_nan=False), BeforeValidator(_number)
]
type Finite = Annotated[float, Field(allow_inf_nan=False), BeforeValidator(_number)]
type Probability = Annotated[
    float, Field(ge=0, le=1, allow_inf_nan=False), BeforeValidator(_number)
]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Quantiles(Contract):
    """Five quantiles, declared in ``ELICITATION_ORDER``, the order the answer
    schema shows respondents; ``QUANTILES`` lists them in increasing order."""

    p50: Nonnegative
    p25: Nonnegative
    p75: Nonnegative
    p5: Nonnegative
    p95: Nonnegative

    @model_validator(mode="after")
    def ordered(self) -> Self:
        values = [getattr(self, key) for key in QUANTILES]
        if values != sorted(values):
            raise EvidenceValidationError(
                "quantiles_must_be_nondecreasing",
                "quantiles must be ordered p5 <= p25 <= p50 <= p75 <= p95",
            )
        return self


class SeverityQuantiles(Quantiles):
    """Uncapped occurrence-loss quantiles.

    p95 and p99 anchor the governed GPD tail. p999 is a deep-tail diagnostic:
    compilation ignores it, so it never changes the governed shape or the law.
    """

    p99: Nonnegative
    p999: Nonnegative

    @model_validator(mode="after")
    def tail(self) -> Self:
        # These are unconditional occurrence quantiles. Repeated zero knots
        # retain their zero-loss atom in the native quantile body; users of a
        # conditional-positive law enforce positivity at that boundary.
        if self.p99 <= self.p95 or self.p5 >= self.p95:
            raise EvidenceValidationError(
                "severity_requires_p5_below_p95_and_p99_above_p95",
                "severity requires p5 < p95 and p99 strictly above p95",
            )
        if self.p999 <= self.p99:
            raise EvidenceValidationError(
                "severity_p999_must_exceed_p99",
                "severity requires p999 strictly above p99",
            )
        lower, upper = ADMITTED_AMOUNT_RANGE
        if any(
            not (value == 0 or lower <= value <= upper)
            for value in (self.p5, self.p25, self.p50, self.p75, self.p95, self.p99)
        ) or not (lower <= self.p999 <= upper):
            raise EvidenceValidationError(
                "severity_quantiles_outside_admitted_monetary_domain",
                f"severity quantiles must lie in the admitted monetary domain: "
                f"zero or between {lower:g} and {upper:g} currency units",
            )
        return self


class Reasoning(Contract):
    median_reason: RequiredString
    low_case: RequiredString
    high_case: RequiredString
    method: Literal["construction", "direct_judgment"]
    evidence_used: tuple[RequiredString, ...]

    @model_validator(mode="after")
    def evidence(self) -> Self:
        if len(self.evidence_used) != len(set(self.evidence_used)):
            raise EvidenceValidationError(
                "evidence_ids_must_be_unique",
                "evidence IDs must be distinct",
                location=("evidence_used",),
            )
        return self


class Respondent(Contract):
    respondent_id: RequiredIdentifier
    model_id: RequiredIdentifier
    sample_index: int = Field(ge=0, strict=True)

    def key(self) -> tuple[str, str, int]:
        return self.respondent_id, self.model_id, self.sample_index
