"""The results export index over a run's setup, capture and results artifacts."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from catmodel.contracts.primitives import Sha256Digest
from catmodel.contracts.results import ContextualChainSummary
from catmodel.hashing import canonical_json_sha256


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactReference(_Record):
    path: str
    sha256: Sha256Digest

    @model_validator(mode="after")
    def relative_path(self):
        if (
            not self.path
            or self.path.startswith("/")
            or "\\" in self.path
            or any(part in {"", ".", ".."} for part in self.path.split("/"))
            or ":" in self.path
        ):
            raise ValueError("export artifact paths must be relative portable paths")
        return self


class ExceedancePoint(_Record):
    loss: float = Field(ge=0, allow_inf_nan=False)
    probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    useful_exceedances: int = Field(strict=True, ge=0)
    tail_weight_ess: float = Field(ge=0, allow_inf_nan=False)


class OutputSeries(_Record):
    id: str
    arm: str
    view: str
    measure: Literal["societal_harm", "lab_economic_loss", "supporting_diagnostic"]
    lab_id: str | None = None
    quantity: Literal["annual_aggregate", "annual_occurrence_maximum"]
    status: Literal["available", "unavailable"]
    reason: str | None = None
    basis: dict[str, str] | None = None
    metrics_pointer: str | None = None
    curve: tuple[ExceedancePoint, ...] = ()

    @model_validator(mode="after")
    def availability(self):
        if self.status == "unavailable":
            if not self.reason or self.curve or self.metrics_pointer is not None:
                raise ValueError("unavailable series need a reason and no values")
        elif self.reason is not None or self.metrics_pointer is None or not self.curve:
            raise ValueError("available series need native metrics and a curve")
        if (
            self.measure == "lab_economic_loss"
            and self.status == "available"
            and self.lab_id is None
        ):
            raise ValueError("available lab loss requires its named lab boundary")
        if self.measure != "lab_economic_loss" and self.lab_id is not None:
            raise ValueError("only lab loss series have a lab boundary")
        if any(
            a.loss >= b.loss or a.probability < b.probability
            for a, b in zip(self.curve, self.curve[1:])
        ):
            raise ValueError(
                "EP points require increasing losses and nonincreasing probabilities"
            )
        return self


CAPABILITY_SCENARIO_OUTPUTS = frozenset(
    {
        "capability_scenarios.json",
        "capability_scenarios.csv",
        "capability_nodes.csv",
    }
)


class ResultsExport(_Record):
    format: Literal["cat.results/1"] = "cat.results/1"
    # The setup's manifest, the capture's manifest, results.json and the
    # diagnostics file beside it.
    source: dict[
        Literal["setup", "capture", "results", "diagnostics"], ArtifactReference
    ]
    outputs: dict[str, ArtifactReference]
    scope: dict[str, Any]
    series: tuple[OutputSeries, ...]
    sample_sets: dict[str, Any]
    conditioning: dict[str, Any]
    chain_summary: tuple[ContextualChainSummary, ...]
    conventions: dict[str, str]

    @model_validator(mode="after")
    def complete_index(self):
        if not {"setup", "capture", "results", "diagnostics"} <= set(self.source):
            raise ValueError(
                "export requires setup, capture, results and diagnostics references"
            )
        if set(self.outputs) != CAPABILITY_SCENARIO_OUTPUTS or any(
            self.outputs[name].path != name for name in CAPABILITY_SCENARIO_OUTPUTS
        ):
            raise ValueError(
                "export requires the three capability-scenario artifacts at their declared paths"
            )
        if len({row.id for row in self.series}) != len(self.series):
            raise ValueError("export series identifiers must be unique")
        canonical_json_sha256(self.model_dump(mode="json"))
        return self
