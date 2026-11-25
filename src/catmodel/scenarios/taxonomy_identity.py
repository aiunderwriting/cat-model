"""Scenario-roster identity and population lineage.

The hierarchy owns placement. ``registries/scenario_lineage.yaml`` names the
current roster generation and records each generation's
``scenario_roster_sha256``, the digest of its sorted scenario ids. Validation
refuses a roster whose digest differs from the current generation's, so a
roster change registers a new generation. A ``transitions`` entry records how
the change moved populations and how much earlier evidence it stales, and a
``retired_scenarios`` entry records a scenario that left the roster; an edit
within an unchanged roster needs neither. Every entry must name a registered
generation and scenario ids that are live or registered as retired.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import Field, model_validator

from catmodel.hashing import canonical_json_sha256
from catmodel.resources import resource_path
from catmodel.scenarios.loader import ScenarioLineageDef
from catmodel.scenarios.schema import Prose, StrictModel
from catmodel.yaml_io import safe_load

DEFAULT_SCENARIO_LINEAGE_REGISTER_RELATIVE_PATH = Path(
    "registries/scenario_lineage.yaml"
)
RetiredScenarioDisposition = Literal["renamed", "split", "merged", "withdrawn"]
type Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class TaxonomyGenerationDef(StrictModel):
    version: Prose
    established: Prose
    scenario_roster_sha256: Sha256Digest
    description: Prose


class RetiredScenarioDef(StrictModel):
    id: Prose
    retired_in: Prose
    disposition: RetiredScenarioDisposition
    successor_ids: list[Prose] = Field(default_factory=list)
    rationale: Prose

    @model_validator(mode="after")
    def validate_successors(self) -> "RetiredScenarioDef":
        cleaned = self.successor_ids
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"retired scenario {self.id}: duplicate successor ids")
        if self.id in cleaned:
            raise ValueError(f"retired scenario {self.id}: successor references itself")
        if self.disposition == "withdrawn":
            if cleaned:
                raise ValueError(
                    f"retired scenario {self.id}: disposition 'withdrawn' must "
                    "not declare successor_ids"
                )
        elif not cleaned:
            raise ValueError(
                f"retired scenario {self.id}: disposition {self.disposition!r} "
                "requires at least one successor id"
            )
        return self


class PopulationTransitionDef(StrictModel):
    """How one roster change moved scenario populations.

    ``kind`` names the change, ``executed_in`` the generation that made it and
    ``evidence_staling`` how much of the evidence collected before it the
    change stales. A ``scope_expansion`` adds a population that no scenario
    estimated, so it names no predecessor; every other kind names at least one.
    """

    transition_id: Prose
    kind: Literal["split", "merge", "carve_out", "refinement", "scope_expansion"]
    executed_in: Prose
    predecessor_ids: list[Prose] = Field(default_factory=list)
    successor_ids: list[Prose]
    evidence_staling: Literal["none", "denominator", "all"]
    rationale: Prose

    @model_validator(mode="after")
    def transition_shape_is_coherent(self) -> "PopulationTransitionDef":
        if not self.successor_ids:
            raise ValueError("population transition must name a successor")
        for field_name, ids in (
            ("predecessor_ids", self.predecessor_ids),
            ("successor_ids", self.successor_ids),
        ):
            if len(ids) != len(set(ids)):
                raise ValueError(f"{field_name} cannot contain duplicate IDs")
        if self.kind == "scope_expansion":
            if self.predecessor_ids:
                raise ValueError("scope expansion cannot name a predecessor")
        elif not self.predecessor_ids:
            raise ValueError(f"{self.kind} transition requires a predecessor")
        return self


class ScenarioLineageRegister(StrictModel):
    schema_version: Literal[1]
    current_taxonomy_version: str
    generations: list[TaxonomyGenerationDef]
    transitions: list[PopulationTransitionDef] = Field(default_factory=list)
    retired_scenarios: list[RetiredScenarioDef] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_register(self) -> "ScenarioLineageRegister":
        versions = [generation.version for generation in self.generations]
        if not versions:
            raise ValueError("lineage register must declare at least one generation")
        duplicates = sorted({v for v in versions if versions.count(v) > 1})
        if duplicates:
            raise ValueError(f"duplicate generation versions: {duplicates}")
        if self.current_taxonomy_version not in versions:
            raise ValueError(
                f"current_taxonomy_version {self.current_taxonomy_version!r} is "
                "not a registered generation"
            )
        retired_ids = [row.id for row in self.retired_scenarios]
        duplicate_retired = sorted(
            {retired for retired in retired_ids if retired_ids.count(retired) > 1}
        )
        if duplicate_retired:
            raise ValueError(f"duplicate retired scenario ids: {duplicate_retired}")
        transition_ids = [row.transition_id for row in self.transitions]
        duplicate_transitions = sorted(
            {value for value in transition_ids if transition_ids.count(value) > 1}
        )
        if duplicate_transitions:
            raise ValueError(f"duplicate transition ids: {duplicate_transitions}")
        return self

    @property
    def generation_versions(self) -> set[str]:
        return {generation.version for generation in self.generations}

    @property
    def current_generation(self) -> TaxonomyGenerationDef:
        for generation in self.generations:
            if generation.version == self.current_taxonomy_version:
                return generation
        raise ValueError(  # pragma: no cover - rejected by validate_register
            f"current generation {self.current_taxonomy_version!r} not registered"
        )

    @property
    def retired_ids(self) -> set[str]:
        return {row.id for row in self.retired_scenarios}


def load_scenario_lineage_register(path: Path | None = None) -> ScenarioLineageRegister:
    """Load the lineage register, by default the packaged repository one."""
    resolved = path or resource_path(
        *DEFAULT_SCENARIO_LINEAGE_REGISTER_RELATIVE_PATH.parts
    )
    try:
        data = safe_load(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"scenario lineage register not found: {resolved}"
        ) from None
    except Exception as exc:
        raise ValueError(
            f"failed to parse scenario lineage register {resolved}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"scenario lineage register {resolved} must be a mapping")
    try:
        return ScenarioLineageRegister.model_validate(data)
    except Exception as exc:
        raise ValueError(
            f"invalid scenario lineage register {resolved}: {exc}"
        ) from exc


def scenario_roster_sha256(scenario_ids: list[str] | tuple[str, ...]) -> str:
    """Canonical digest of the scenario roster (sorted unique ids)."""
    return canonical_json_sha256(sorted(set(scenario_ids)))


class ScenarioLineageView(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def lineage(self) -> "ScenarioLineageDef": ...


def validate_scenario_lineage(
    scenarios: Sequence[ScenarioLineageView],
    register: ScenarioLineageRegister,
) -> list[str]:
    """Validate roster identity and lineage references; empty list means success."""
    errors: list[str] = []
    live_ids = {scenario.id for scenario in scenarios}
    retired_ids = register.retired_ids
    known_ids = live_ids | retired_ids
    generations = register.generation_versions

    recorded = register.current_generation.scenario_roster_sha256
    live_digest = scenario_roster_sha256(sorted(live_ids))
    if recorded != live_digest:
        errors.append(
            "ERROR scenario roster drifted from taxonomy generation "
            f"{register.current_taxonomy_version!r}: live roster digest "
            f"{live_digest} != registered {recorded}. A roster change "
            "(add/retire/rename/split/merge) requires a new taxonomy "
            "generation with lineage entries in "
            "registries/scenario_lineage.yaml, not a silent edit."
        )

    stale_retired = sorted(retired_ids & live_ids)
    if stale_retired:
        errors.append(
            f"ERROR retired scenario ids still present in the live roster: "
            f"{stale_retired}"
        )

    for row in register.retired_scenarios:
        if row.retired_in not in generations:
            errors.append(
                f"ERROR retired scenario {row.id!r}: retired_in "
                f"{row.retired_in!r} is not a registered generation"
            )
        unknown_successors = sorted(set(row.successor_ids) - known_ids)
        if unknown_successors:
            errors.append(
                f"ERROR retired scenario {row.id!r}: successor ids "
                f"{unknown_successors} are neither live nor registered retired"
            )

    for transition in register.transitions:
        if transition.executed_in not in generations:
            errors.append(
                f"ERROR transition {transition.transition_id!r}: executed_in "
                f"{transition.executed_in!r} is not a registered generation"
            )
        unknown_ids = sorted(
            (set(transition.predecessor_ids) | set(transition.successor_ids))
            - known_ids
        )
        if unknown_ids:
            errors.append(
                f"ERROR transition {transition.transition_id!r}: scenario ids "
                f"{unknown_ids} are neither live nor registered retired"
            )

    for scenario in scenarios:
        lineage = scenario.lineage
        if lineage.introduced_in not in generations:
            errors.append(
                f"ERROR {scenario.id}: lineage.introduced_in "
                f"{lineage.introduced_in!r} is not a registered generation"
            )
    return errors
