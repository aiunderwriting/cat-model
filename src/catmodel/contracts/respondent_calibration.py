"""Frozen contracts for respondent calibration instruments and run artifacts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from catmodel.contracts.contextual_risk import (
    CONTEXTUAL_SCHEMA_VERSION,
    UNAVAILABLE_REASON_DESCRIPTION,
    StatusFields,
)
from catmodel.contracts.elicitation import (
    Nonnegative,
    Probability,
    Quantiles,
    Reasoning,
    Respondent,
)
from catmodel.contracts.primitives import (
    Frozen,
    FrozenDict,
    RequiredIdentifier,
    RequiredString,
    Sha256Digest,
)
from catmodel.hashing import canonical_json_sha256, sha256_file
from catmodel.yaml_io import safe_load


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CalibrationSeedSource(Contract):
    name: RequiredString
    url: RequiredString


class CalibrationSeedHistory(Contract):
    period: RequiredString
    period_end: date = Field(description="Last day covered by this history row.")
    value: Nonnegative


class CalibrationSeedEvidence(Contract):
    text: RequiredString
    history: tuple[CalibrationSeedHistory, ...]


class CalibrationSeed(Contract):
    """A known-answer question definition; it deliberately has no outcome field."""

    seed_id: RequiredIdentifier
    role: Literal["primary", "control", "secondary"]
    block_id: RequiredIdentifier
    quantity: RequiredString
    unit: Literal["count", "usd"]
    window_start: date
    window_end: date
    information_date: date
    source: CalibrationSeedSource
    evidence: CalibrationSeedEvidence
    resolution_script: RequiredString = Field(
        description="Repository path of the script that resolves this seed; not rendered in the question."
    )
    can_resolve_to_zero: StrictBool

    @model_validator(mode="after")
    def dates_and_units(self) -> Self:
        if self.window_start > self.window_end:
            raise ValueError("seed window_start must be on or before window_end")
        if self.information_date > self.window_start:
            raise ValueError("seed information_date must be on or before window_start")
        for row in self.evidence.history:
            if row.period_end > self.information_date:
                raise ValueError(
                    "seed history period_end must be on or before information_date"
                )
            if row.period_end >= self.window_start:
                raise ValueError("seed history period_end must be before window_start")
        _repository_script_path(self.resolution_script)
        if self.role in ("primary", "control") and self.unit != "count":
            raise ValueError("primary and control seeds must use unit 'count'")
        return self


def _repository_script_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or "\\" in value
        or path.as_posix() != value
        or any(part in (".", "..") for part in path.parts)
    ):
        raise ValueError("resolution script paths must be relative repository paths")


class ResolutionProtocol(Contract):
    """Resolution scripts and rules committed to before collection."""

    scripts: Frozen[dict[RequiredString, Sha256Digest]] = Field(min_length=1)
    snapshot_rule: RequiredString
    replacement_rule: RequiredString

    @model_validator(mode="after")
    def repository_script_paths(self) -> Self:
        for path in self.scripts:
            _repository_script_path(path)
        return self


def _validate_robustness_configuration(
    seeds: tuple[CalibrationSeed, ...],
    super_blocks: Mapping[str, tuple[str, ...]],
    curation_sensitive_seed_ids: tuple[str, ...],
) -> None:
    primary_seeds = tuple(seed for seed in seeds if seed.role == "primary")
    primary_seed_ids = {seed.seed_id for seed in primary_seeds}
    primary_block_ids = {seed.block_id for seed in primary_seeds}

    if len(set(curation_sensitive_seed_ids)) != len(curation_sensitive_seed_ids):
        raise ValueError("curation-sensitive seed ids must be unique")
    invalid_seed_ids = set(curation_sensitive_seed_ids) - primary_seed_ids
    if invalid_seed_ids:
        raise ValueError(
            "curation_sensitive_seed_ids must name primary seeds; invalid ids: "
            + ", ".join(sorted(invalid_seed_ids))
        )

    listed_blocks = [
        block_id for block_ids in super_blocks.values() for block_id in block_ids
    ]
    repeated_blocks = sorted(
        block_id for block_id, count in Counter(listed_blocks).items() if count > 1
    )
    if repeated_blocks:
        raise ValueError(
            "each primary block may appear in at most one super-block: "
            + ", ".join(repeated_blocks)
        )
    unknown_blocks = set(listed_blocks) - primary_block_ids
    if unknown_blocks:
        raise ValueError(
            "super-blocks contain unknown primary blocks: "
            + ", ".join(sorted(unknown_blocks))
        )
    if any(not block_ids for block_ids in super_blocks.values()):
        raise ValueError("super-blocks must contain at least one primary block")


def _validate_seed_roster(
    seeds: tuple[CalibrationSeed, ...],
    latest_training_cutoff: date,
    resolution_protocol: ResolutionProtocol,
) -> None:
    script_paths = set(resolution_protocol.scripts)
    for seed in seeds:
        if seed.resolution_script not in script_paths:
            raise ValueError(
                f"seed {seed.seed_id!r} resolution_script is not in the frozen protocol"
            )
        if seed.role == "primary" and seed.window_start <= latest_training_cutoff:
            raise ValueError(
                "primary seed window_start must be after latest_training_cutoff"
            )
        if seed.role == "control" and seed.window_end > latest_training_cutoff:
            raise ValueError(
                "control seed window_end must be on or before latest_training_cutoff"
            )


class CalibrationSeedSet(Contract):
    """Seed roster with its respondent training cutoff and frozen resolution rules."""

    seeds: tuple[CalibrationSeed, ...] = Field(min_length=1)
    latest_training_cutoff: date = Field(
        description="The latest training-data cutoff among the roster's respondent models."
    )
    resolution_protocol: ResolutionProtocol
    super_blocks: Frozen[dict[RequiredIdentifier, tuple[RequiredIdentifier, ...]]] = (
        Field(default_factory=FrozenDict)
    )
    curation_sensitive_seed_ids: tuple[RequiredIdentifier, ...] = ()

    @model_validator(mode="after")
    def unique_and_blocked(self) -> Self:
        identities = [seed.seed_id for seed in self.seeds]
        if len(set(identities)) != len(identities):
            raise ValueError("calibration seed ids must be unique")
        primary_per_block = Counter(
            seed.block_id for seed in self.seeds if seed.role == "primary"
        )
        if any(count > 3 for count in primary_per_block.values()):
            raise ValueError("a block may contain at most three primary seeds")
        _validate_seed_roster(
            self.seeds, self.latest_training_cutoff, self.resolution_protocol
        )
        _validate_robustness_configuration(
            self.seeds, self.super_blocks, self.curation_sensitive_seed_ids
        )
        return self


class SeedResolution(Contract):
    """A seed's resolved value and where it came from; no seed is unresolved."""

    seed_id: RequiredIdentifier
    value: Nonnegative
    retrieved_at: datetime | date
    snapshot_url: RequiredString
    snapshot_sha256: Sha256Digest
    script: RequiredString
    notes: str


class SeedResolutionSet(Contract):
    resolutions: tuple[SeedResolution, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        identities = [row.seed_id for row in self.resolutions]
        if len(set(identities)) != len(identities):
            raise ValueError("seed resolution ids must be unique")
        return self


def _yaml_mapping(path: Path | str, key: str) -> dict[str, object]:
    loaded = safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or set(loaded) != {key}:
        raise ValueError(f"YAML file must contain only a top-level {key!r} field")
    if not isinstance(loaded[key], list):
        raise ValueError(f"top-level {key!r} must be a list")
    return loaded


def load_calibration_seeds(path: Path | str) -> CalibrationSeedSet:
    """Load a roster, cutoff and resolution protocol from a YAML seed file."""
    loaded = safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"seeds", "latest_training_cutoff", "resolution_protocol"}
    allowed = required | {"super_blocks", "curation_sensitive_seed_ids"}
    if (
        not isinstance(loaded, dict)
        or not required <= set(loaded)
        or not set(loaded) <= allowed
    ):
        raise ValueError(
            "seed YAML must contain seeds, latest_training_cutoff and "
            "resolution_protocol, with optional super_blocks and "
            "curation_sensitive_seed_ids"
        )
    if not isinstance(loaded["seeds"], list):
        raise ValueError("top-level 'seeds' must be a list")
    return CalibrationSeedSet.model_validate(loaded)


def load_seed_resolutions(path: Path | str) -> SeedResolutionSet:
    """Load resolved values from a YAML file containing only ``resolutions:``."""
    return SeedResolutionSet.model_validate(_yaml_mapping(path, "resolutions"))


def resolution_script_digests(
    seed_set: CalibrationSeedSet, root: Path
) -> dict[str, str | None]:
    """Return each frozen script path whose digest under ``root`` differs.

    Missing, non-file and symlinked scripts have an actual digest of ``None``.
    """
    resolved_root = root.expanduser().resolve()
    mismatches: dict[str, str | None] = {}
    for relative_path, frozen_digest in seed_set.resolution_protocol.scripts.items():
        candidate = resolved_root.joinpath(*PurePosixPath(relative_path).parts)
        current = resolved_root
        has_symlink = False
        for part in PurePosixPath(relative_path).parts:
            current = current / part
            if current.is_symlink():
                has_symlink = True
                break
        resolved_candidate = candidate.resolve()
        if (
            has_symlink
            or not resolved_candidate.is_relative_to(resolved_root)
            or not resolved_candidate.is_file()
        ):
            actual_digest = None
        else:
            actual_digest = sha256_file(resolved_candidate)
        if actual_digest != frozen_digest:
            mismatches[relative_path] = actual_digest
    return mismatches


def resolved_outcomes(
    seeds: CalibrationSeedSet | tuple[CalibrationSeed, ...],
    resolutions: SeedResolutionSet | tuple[SeedResolution, ...],
) -> dict[str, float]:
    """Validate a complete resolution roster and return values by seed id."""
    seed_rows = seeds.seeds if isinstance(seeds, CalibrationSeedSet) else tuple(seeds)
    resolution_rows = (
        resolutions.resolutions
        if isinstance(resolutions, SeedResolutionSet)
        else tuple(resolutions)
    )
    seed_by_id = {row.seed_id: row for row in seed_rows}
    resolution_by_id = {row.seed_id: row for row in resolution_rows}
    if len(seed_by_id) != len(seed_rows):
        raise ValueError("calibration seed ids must be unique")
    if len(resolution_by_id) != len(resolution_rows):
        raise ValueError("seed resolution ids must be unique")
    if set(seed_by_id) != set(resolution_by_id):
        missing = sorted(set(seed_by_id) - set(resolution_by_id))
        extra = sorted(set(resolution_by_id) - set(seed_by_id))
        raise ValueError(
            f"resolutions must exactly match seeds: missing={missing}, extra={extra}"
        )

    outcomes: dict[str, float] = {}
    for seed_id, seed in seed_by_id.items():
        resolution = resolution_by_id[seed_id]
        if resolution.script != seed.resolution_script:
            raise ValueError(
                f"resolution for seed {seed_id!r} uses a different script than its frozen resolution_script"
            )
        value = resolution.value
        if value == 0 and not seed.can_resolve_to_zero:
            raise ValueError(f"seed {seed_id!r} cannot resolve to zero")
        if seed.unit == "count" and not value.is_integer():
            raise ValueError(f"count seed {seed_id!r} must resolve to an integer")
        outcomes[seed_id] = value
    return outcomes


_CALIBRATION_STATUS = StatusFields(
    "available",
    "unavailable",
    "unavailable_reason",
    required=("rate",),
    excluded=("rate",),
)


class CalibrationAnswer(Contract):
    """The direct expected-count answer contract without live-study extras."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, json_schema_extra=_CALIBRATION_STATUS
    )

    reasoning: Reasoning
    status: Literal["available", "unavailable"]
    unavailable_reason: RequiredString | None = Field(
        default=None, description=UNAVAILABLE_REASON_DESCRIPTION
    )
    rate: Quantiles | None = None

    @model_validator(mode="after")
    def explicit_availability(self) -> Self:
        _CALIBRATION_STATUS.validate(self)
        return self


class CalibrationInstrument(Contract):
    schema_version: Literal[1] = 1
    seeds: tuple[CalibrationSeed, ...] = Field(min_length=1)
    respondents: tuple[Respondent, ...] = Field(min_length=1)
    latest_training_cutoff: date = Field(
        description="The latest training-data cutoff among the roster's respondent models."
    )
    resolution_protocol: ResolutionProtocol
    super_blocks: Frozen[dict[RequiredIdentifier, tuple[RequiredIdentifier, ...]]] = (
        Field(default_factory=FrozenDict)
    )
    curation_sensitive_seed_ids: tuple[RequiredIdentifier, ...] = ()
    risk_contract_version: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def unique_identities_and_version(self) -> Self:
        seed_ids = [seed.seed_id for seed in self.seeds]
        if len(set(seed_ids)) != len(seed_ids):
            raise ValueError("calibration seed ids must be unique")
        primary_per_block = Counter(
            seed.block_id for seed in self.seeds if seed.role == "primary"
        )
        if any(count > 3 for count in primary_per_block.values()):
            raise ValueError("a block may contain at most three primary seeds")
        _validate_seed_roster(
            self.seeds, self.latest_training_cutoff, self.resolution_protocol
        )
        _validate_robustness_configuration(
            self.seeds, self.super_blocks, self.curation_sensitive_seed_ids
        )
        respondent_keys = [row.key() for row in self.respondents]
        if len(set(respondent_keys)) != len(respondent_keys):
            raise ValueError("instrument respondents must be unique")
        if self.risk_contract_version != CONTEXTUAL_SCHEMA_VERSION:
            raise ValueError(
                "calibration instrument uses a stale risk contract version"
            )
        return self

    def digest(self) -> str:
        return canonical_json_sha256(self.model_dump(mode="json"))


class CalibrationCase(Contract):
    seed_id: RequiredIdentifier
    respondent: Respondent
    answer: CalibrationAnswer | None


class CalibrationArtifact(Contract):
    schema_version: Literal[1] = 1
    instrument_sha256: Sha256Digest
    collection_plan_sha256: Sha256Digest
    cases: tuple[CalibrationCase, ...]

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        keys = [(row.seed_id, row.respondent.key()) for row in self.cases]
        if len(set(keys)) != len(keys):
            raise ValueError("calibration cases must be unique")
        return self


class CalibrationModelTreatment(Contract):
    provider: Literal["openai", "anthropic", "google", "xai", "meta", "openrouter"]
    model_id: RequiredIdentifier
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"]
    max_output_tokens: int = Field(gt=0, strict=True)


type SampleCount = Annotated[int, Field(ge=1, strict=True)]


class CalibrationTreatments(Contract):
    models: tuple[CalibrationModelTreatment, ...] = Field(min_length=1)
    samples_per_model: Frozen[dict[RequiredIdentifier, SampleCount]]
    parse_attempts: int = Field(gt=0, strict=True)
    frozen_call_schema_version: int = Field(gt=0, strict=True)
    risk_contract_version: int = Field(gt=0, strict=True)
    system_prompt_sha256: Sha256Digest

    @model_validator(mode="after")
    def model_roster(self) -> Self:
        model_ids = [row.model_id for row in self.models]
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("calibration treatment models must be unique")
        if set(self.samples_per_model) != set(model_ids):
            raise ValueError("sample counts must match the treatment model roster")
        return self


class CalibrationCaptureIdentity(Contract):
    instrument_sha256: Sha256Digest
    collection_plan_sha256: Sha256Digest
    artifact_sha256: Sha256Digest
    resolutions_sha256: Sha256Digest


class SpreadRange(Contract):
    low: float = Field(ge=0.5, le=4, allow_inf_nan=False)
    high: float = Field(ge=0.5, le=4, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.low > self.high:
            raise ValueError("spread range low must not exceed high")
        return self


class CommonBiasRange(Contract):
    low: float = Field(ge=0.1, le=10, allow_inf_nan=False)
    high: float = Field(ge=0.1, le=10, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.low > self.high:
            raise ValueError("common bias range low must not exceed high")
        return self


class JointGrid(Contract):
    """Fine log-grid dimensions and their coarse-search strides."""

    k_points: int = Field(ge=3, strict=True)
    b_points: int = Field(ge=3, strict=True)
    k_coarse_stride: int = Field(ge=1, strict=True)
    b_coarse_stride: int = Field(ge=1, strict=True)

    @model_validator(mode="after")
    def compatible_axes(self) -> Self:
        if (self.k_points - 1) % 3 != 0:
            raise ValueError("joint k grid points must place k = 1 on the grid")
        if (self.k_points - 1) % self.k_coarse_stride != 0:
            raise ValueError("joint k grid must include both coarse-grid bounds")
        if (self.b_points - 1) % 2 != 0:
            raise ValueError("joint b grid points must place b = 1 on the grid")
        if (self.b_points - 1) % self.b_coarse_stride != 0:
            raise ValueError("joint b grid must include both coarse-grid bounds")
        return self


class JointRange(Contract):
    k: SpreadRange
    b: CommonBiasRange
    correlation: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)


class MarginalSpreadFit(Contract):
    k_hat: float = Field(ge=0.5, le=4, allow_inf_nan=False)
    range: SpreadRange


class MarginalBiasFit(Contract):
    reference_spread: float = Field(ge=0.5, le=4, allow_inf_nan=False)
    reference_spread_basis: Literal["accepted_k_hat", "face_value"]
    b_hat: float = Field(ge=0.1, le=10, allow_inf_nan=False)
    range: CommonBiasRange


class MarginalFits(Contract):
    k_at_b_1: MarginalSpreadFit
    b_at_reference_k: MarginalBiasFit


class RespondentSpreadObservation(Contract):
    model_name: Literal["negative_binomial_2"]
    pooled_phi: float = Field(ge=0, allow_inf_nan=False)
    minimum_periods: Literal[6]
    fallback_seed_ids: tuple[RequiredIdentifier, ...]

    @model_validator(mode="after")
    def unique_fallback_seeds(self) -> Self:
        if len(set(self.fallback_seed_ids)) != len(self.fallback_seed_ids):
            raise ValueError("dispersion fallback seed ids must be unique")
        return self


class PoissonSpreadAlternative(Contract):
    k_hat: float = Field(ge=0.5, le=4, allow_inf_nan=False)
    range: SpreadRange
    b_hat: float = Field(ge=0.1, le=10, allow_inf_nan=False)
    b_range: CommonBiasRange


class PooledPhiSpreadAlternative(Contract):
    phi: float = Field(ge=0, allow_inf_nan=False)
    k_hat: float = Field(ge=0.5, le=4, allow_inf_nan=False)
    range: SpreadRange
    b_hat: float = Field(ge=0.1, le=10, allow_inf_nan=False)
    b_range: CommonBiasRange


class SpreadObservationAlternatives(Contract):
    poisson: PoissonSpreadAlternative
    pooled_phi: PooledPhiSpreadAlternative


class BootstrapSummary(Contract):
    replicates: int = Field(ge=1, strict=True)
    seed: int = Field(ge=0, strict=True)
    blocks: int = Field(ge=1, strict=True)


class SuperBlockBootstrapDiagnostics(Contract):
    range: JointRange
    super_blocks: int = Field(ge=1, strict=True)


class CurationSensitivityDiagnostics(Contract):
    seed_ids: tuple[RequiredIdentifier, ...]
    k_hat: float | None = Field(default=None, ge=0.5, le=4, allow_inf_nan=False)
    b_hat: float | None = Field(default=None, ge=0.1, le=10, allow_inf_nan=False)
    range: JointRange | None = None
    blocks: int = Field(default=0, ge=0, strict=True)
    unavailable_reason: RequiredString | None = None

    @model_validator(mode="after")
    def fit_fields_agree(self) -> Self:
        if self.k_hat is None or self.b_hat is None:
            if (
                self.k_hat is not None
                or self.b_hat is not None
                or self.range is not None
                or self.blocks != 0
                or self.unavailable_reason is None
            ):
                raise ValueError(
                    "an unavailable curation-sensitivity fit needs a reason and no range"
                )
        elif (
            self.range is None or self.blocks < 1 or self.unavailable_reason is not None
        ):
            raise ValueError(
                "a curation-sensitivity fit needs its range and bootstrap blocks"
            )
        return self


class SeedPIT(Contract):
    seed_id: RequiredIdentifier
    block_id: RequiredIdentifier
    pit: Probability


class PITAtSpread(Contract):
    median: Probability
    central_50_coverage: Probability
    central_90_coverage: Probability
    histogram_10: tuple[int, int, int, int, int, int, int, int, int, int]
    seeds: tuple[SeedPIT, ...]


class PrimaryPITDiagnostics(Contract):
    at_k_1_b_1: PITAtSpread
    at_k_hat_b_1: PITAtSpread
    at_k_1_b_hat: PITAtSpread
    at_k_hat_b_hat: PITAtSpread


class LeaveOneBlockOutRow(Contract):
    block_id: RequiredIdentifier
    k_hat: float = Field(ge=0.5, le=4, allow_inf_nan=False)
    b_hat: float = Field(ge=0.1, le=10, allow_inf_nan=False)
    log_score_at_k_hat_b_hat: float | None = Field(allow_inf_nan=False)
    log_score_at_k_1_b_hat: float | None = Field(allow_inf_nan=False)
    log_score_at_k_1_b_1: float | None = Field(allow_inf_nan=False)
    joint_gain: float | None = Field(allow_inf_nan=False)
    conditional_k_gain: float | None = Field(allow_inf_nan=False)


class LeaveOneBlockOutDiagnostics(Contract):
    joint_total_gain: float | None = Field(allow_inf_nan=False)
    conditional_k_total_gain: float | None = Field(allow_inf_nan=False)
    blocks: tuple[LeaveOneBlockOutRow, ...]


class DiagnosticReferencePercentiles(Contract):
    observed: float | None = Field(allow_inf_nan=False)
    p05: float = Field(allow_inf_nan=False)
    p95: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.p05 > self.p95:
            raise ValueError("diagnostic reference p05 must not exceed p95")
        return self


class PITReferenceDiagnostics(Contract):
    median: DiagnosticReferencePercentiles
    central_50_coverage: DiagnosticReferencePercentiles
    central_90_coverage: DiagnosticReferencePercentiles


class SimulatedReferenceDiagnostics(Contract):
    replicates: int = Field(ge=1, strict=True)
    seed: int = Field(ge=0, strict=True)
    b_hat: float = Field(ge=0.1, le=10, allow_inf_nan=False)
    primary_pit_at_k_1_b_hat: PITReferenceDiagnostics
    primary_pit_at_k_refit_b_hat: PITReferenceDiagnostics
    cooke_score_at_k_1_b_hat: DiagnosticReferencePercentiles
    cooke_score_at_k_refit_b_hat: DiagnosticReferencePercentiles
    conditional_k_leave_one_block_out_gain: DiagnosticReferencePercentiles


class CookeScore(Contract):
    observed_proportions: tuple[float, float, float, float, float, float]
    expected_proportions: tuple[float, float, float, float, float, float]
    divergence: float = Field(ge=0, allow_inf_nan=False)
    statistic: float = Field(ge=0, allow_inf_nan=False)
    degrees_of_freedom: Literal[5] = 5
    score: Probability


class CookeDiagnostics(Contract):
    at_k_1_b_1: CookeScore
    at_k_hat_b_1: CookeScore
    at_k_1_b_hat: CookeScore
    at_k_hat_b_hat: CookeScore


class ModelJointFit(Contract):
    model_id: RequiredIdentifier
    k_hat: float = Field(ge=0.5, le=4, allow_inf_nan=False)
    b_hat: float = Field(ge=0.1, le=10, allow_inf_nan=False)
    scored_seeds: int = Field(ge=1, strict=True)


class AuxiliarySeedPIT(Contract):
    seed_id: RequiredIdentifier
    role: Literal["control", "secondary"]
    pit_at_k_1_b_1: Probability | None
    pit_at_k_hat_b_hat: Probability | None


class MemorizationRow(Contract):
    seed_id: RequiredIdentifier
    role: Literal["primary", "control"]
    model_id: RequiredIdentifier
    kept_samples: int = Field(ge=0, strict=True)
    near_exact_medians: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def count_is_bounded(self) -> Self:
        if self.near_exact_medians > self.kept_samples:
            raise ValueError("near-exact medians cannot exceed kept samples")
        return self


class RespondentSpreadDiagnostics(Contract):
    primary_pit: PrimaryPITDiagnostics
    leave_one_block_out: LeaveOneBlockOutDiagnostics
    cooke: CookeDiagnostics
    super_block_bootstrap: SuperBlockBootstrapDiagnostics
    without_curation_sensitive_seeds: CurationSensitivityDiagnostics
    per_model: tuple[ModelJointFit, ...]
    controls_and_secondary_pit: tuple[AuxiliarySeedPIT, ...]
    memorization: tuple[MemorizationRow, ...]
    unscored_primary_seeds: tuple[RequiredIdentifier, ...] = ()
    """Primary seeds without a kept available answer, left out of the fit."""
    zero_probability_primary_seeds: tuple[RequiredIdentifier, ...] = ()
    """Primary seeds with zero pooled outcome probability at every coarse cell."""
    reference_replicates: int = Field(default=0, ge=0, strict=True)
    simulated_reference: SimulatedReferenceDiagnostics | None = None

    @model_validator(mode="after")
    def reference_count_matches(self) -> Self:
        if (self.reference_replicates == 0) != (self.simulated_reference is None):
            raise ValueError("simulated reference must match reference_replicates")
        if (
            self.simulated_reference is not None
            and self.simulated_reference.replicates != self.reference_replicates
        ):
            raise ValueError("simulated reference replicate count differs")
        return self


class RespondentSeedFitRow(Contract):
    seed_id: RequiredIdentifier
    block_id: RequiredIdentifier
    role: Literal["primary", "control", "secondary"]
    value: Nonnegative
    phi: float | None = Field(ge=0, allow_inf_nan=False)
    phi_periods: int | None = Field(ge=0, strict=True)
    phi_source: Literal["history", "pooled"] | None
    phi_fallback_reason: RequiredString | None
    pit_at_k_1_b_1: Probability | None
    pit_at_k_hat_b_hat: Probability | None

    @model_validator(mode="after")
    def dispersion_fields_agree(self) -> Self:
        if self.phi is None:
            if any(
                value is not None
                for value in (
                    self.phi_periods,
                    self.phi_source,
                    self.phi_fallback_reason,
                )
            ):
                raise ValueError("unmeasured seed dispersion fields must be null")
            return self
        if self.phi_source == "history":
            if self.phi_periods is None or self.phi_periods < 6:
                raise ValueError("history dispersion requires at least 6 periods")
            if self.phi_fallback_reason is not None:
                raise ValueError("history dispersion cannot have a fallback reason")
        elif self.phi_source == "pooled":
            if self.phi_fallback_reason is None:
                raise ValueError("pooled dispersion requires a fallback reason")
        else:
            raise ValueError("measured seed dispersion needs a source")
        return self


class RespondentSpreadFit(Contract):
    """Downstream fit artifact; it is not a source-bundle input.

    The fit measures how wide and how biased the respondents' one-shot answers
    to known-count seeds were in one period. ``status`` records the fit's own
    acceptance rule on those seeds; it validates no risk judgment. The fit is a
    diagnostic: no setup reads it (``POOL-SPREAD``).
    """

    schema_version: Literal[1] = 1
    status: Literal["accepted", "rejected"]
    rejection_reason: RequiredString | None = None
    capture: CalibrationCaptureIdentity
    treatments: CalibrationTreatments
    tail_form: Literal["pure_stretch", "calibrated_p95_peak"] = "calibrated_p95_peak"
    k_bounds: tuple[float, float] = (0.5, 4.0)
    b_bounds: tuple[float, float] = (0.1, 10.0)
    grid: JointGrid
    observation: RespondentSpreadObservation
    k_hat: float = Field(ge=0.5, le=4, allow_inf_nan=False)
    range: SpreadRange
    b_hat: float = Field(ge=0.1, le=10, allow_inf_nan=False)
    b_range: CommonBiasRange
    k_b_correlation: float | None = Field(
        default=None, ge=-1, le=1, allow_inf_nan=False
    )
    marginal: MarginalFits
    bootstrap: BootstrapSummary
    alternatives: SpreadObservationAlternatives
    diagnostics: RespondentSpreadDiagnostics
    seeds: tuple[RespondentSeedFitRow, ...]

    @model_validator(mode="after")
    def consistent_status_and_bounds(self) -> Self:
        if self.k_bounds != (0.5, 4.0):
            raise ValueError("respondent spread bounds are fixed at (0.5, 4.0)")
        if self.b_bounds != (0.1, 10.0):
            raise ValueError("respondent bias bounds are fixed at (0.1, 10.0)")
        if self.alternatives.pooled_phi.phi != self.observation.pooled_phi:
            raise ValueError("pooled-phi alternative must use the reference pooled phi")
        expected_basis = "accepted_k_hat" if self.status == "accepted" else "face_value"
        expected_reference_spread = self.k_hat if self.status == "accepted" else 1.0
        if (
            self.marginal.b_at_reference_k.reference_spread_basis != expected_basis
            or self.marginal.b_at_reference_k.reference_spread
            != expected_reference_spread
        ):
            raise ValueError(
                "marginal bias reference spread must match fit status and k_hat"
            )
        fallback_seed_ids = {
            row.seed_id for row in self.seeds if row.phi_source == "pooled"
        }
        if fallback_seed_ids != set(self.observation.fallback_seed_ids):
            raise ValueError("dispersion fallback ids differ from the seed rows")
        if (self.status == "rejected") != (self.rejection_reason is not None):
            raise ValueError("rejection_reason is required iff status is rejected")
        return self


__all__ = [
    "AuxiliarySeedPIT",
    "BootstrapSummary",
    "CalibrationAnswer",
    "CalibrationArtifact",
    "CalibrationCaptureIdentity",
    "CalibrationCase",
    "CalibrationInstrument",
    "CalibrationModelTreatment",
    "CalibrationSeed",
    "CalibrationSeedEvidence",
    "CalibrationSeedHistory",
    "CalibrationSeedSet",
    "CalibrationSeedSource",
    "CalibrationTreatments",
    "CommonBiasRange",
    "CookeDiagnostics",
    "CookeScore",
    "CurationSensitivityDiagnostics",
    "DiagnosticReferencePercentiles",
    "JointGrid",
    "JointRange",
    "LeaveOneBlockOutDiagnostics",
    "LeaveOneBlockOutRow",
    "MarginalBiasFit",
    "MarginalFits",
    "MarginalSpreadFit",
    "MemorizationRow",
    "ModelJointFit",
    "PITAtSpread",
    "PITReferenceDiagnostics",
    "PoissonSpreadAlternative",
    "PooledPhiSpreadAlternative",
    "PrimaryPITDiagnostics",
    "ResolutionProtocol",
    "RespondentSeedFitRow",
    "RespondentSpreadDiagnostics",
    "RespondentSpreadFit",
    "RespondentSpreadObservation",
    "SeedPIT",
    "SeedResolution",
    "SeedResolutionSet",
    "SimulatedReferenceDiagnostics",
    "SpreadObservationAlternatives",
    "SpreadRange",
    "SuperBlockBootstrapDiagnostics",
    "load_calibration_seeds",
    "load_seed_resolutions",
    "resolution_script_digests",
    "resolved_outcomes",
]
