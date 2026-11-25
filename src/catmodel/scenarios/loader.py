"""Load and validate the YAML scenario library.

Each file under ``scenarios/<origin>/`` authors one forecast population: its
counted terminal unit, severity scope, event chain, composition, severity
governance and empirical grounding. Content shared across scenarios is authored
once and expanded before validation: the admission threshold comes from the
hierarchy contract, where a terminal unit writes ``{{admission}}``; composition
rosters from ``composition_rosters.yaml``; severity policies and tail classes
from ``severity_governance_policies.yaml``. ``docs/adding-a-scenario.md`` is
the authoring guide.
"""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from catmodel.assumption_vocabulary import (
    AssumptionMateriality,
    AssumptionStatus,
    SourceClassification,
)
from catmodel.economic_loss import EconomicNature
from catmodel.identifiers import AtomicIdentifier
from catmodel.scenarios.composition_text import expand_accounting_text
from catmodel.scenarios.facets import ScenarioFacetsDef, scenario_facet_violations
from catmodel.scenarios.hierarchy_contract import (
    MONETARY_AMOUNT,
    admission_clause,
    admission_statement,
    admission_threshold,
    cached_hierarchy_contract,
    scenario_placement_coordinates,
)
from catmodel.scenarios.schema import Identifier, Prose, StrictModel, Text
from catmodel.scenarios.sensitivity_coverage import (
    CAPPED_TAIL_SHAPE_SANITY_BOUND,
    TAIL_SHAPE_SENSITIVITY_STEP,
    UNCAPPED_TAIL_SHAPE_BOUND,
    shifted_tail_shape,
)
from catmodel.scenarios.taxonomy_registry import DOMAIN_SORT_RANK
from catmodel.yaml_io import safe_load

SEVERITY_GOVERNANCE_POLICIES_FILENAME = "severity_governance_policies.yaml"
COMPOSITION_ROSTERS_FILENAME = "composition_rosters.yaml"
SHARED_LIBRARY_FILENAMES = frozenset(
    {SEVERITY_GOVERNANCE_POLICIES_FILENAME, COMPOSITION_ROSTERS_FILENAME}
)

# Typed out-of-estimand classes have no quantifiable economic measure;
# severity_occurrence_scope remains the authoritative exclusion prose.
SeverityScopeExclusionClass = Literal["existential_value"]

type PositiveFinite = Annotated[float, Field(gt=0, allow_inf_nan=False)]

ADMISSION_MARKER = "{{admission}}"
"""Where a terminal unit states the hierarchy contract's admission threshold."""


def _word_set(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", text.casefold()))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


# The rendered final stage of every chain: one qualified terminal unit.
EVENT_CHAIN_TERMINAL_STAGE_ID = "qualified_terminal"
_RESERVED_STAGE_IDS = frozenset({EVENT_CHAIN_TERMINAL_STAGE_ID})
# A stage whose distinct words reach this Jaccard similarity with the terminal
# unit's restates it rather than adding a transition.
_RESTATEMENT_JACCARD = 0.8


class EventChainOpportunityDef(StrictModel):
    """The annual opportunity that an event chain starts from.

    ``unit`` states what one opportunity is for the named lab or deployment,
    ``counting_rule`` how opportunities are counted so that each yields at most
    one qualified terminal unit, and ``cohort`` which opportunities enter the
    study year's risk set under the scenario's year rule. Membership never
    depends on later success.
    """

    unit: Prose
    counting_rule: Prose
    cohort: Prose


CapabilityAccess = Literal["public", "internal"]
"""Which deployment a context-bound instrument asks about: the lab's public
models in its public deployment, or its not-yet-public models in internal and
pre-release use."""

EXPOSURE_SCALES: Mapping[str, str] = MappingProxyType(
    {
        "consumer_weekly_users": "weekly active users of the consumer assistant",
        "paying_business_customers": "paying business customers",
        "lab_staff": "lab staff",
    }
)
"""The stipulated deployment scales a derived exposure base may read, each with
the unit it counts. A study that selects a scenario stipulates a level for every
scale the scenario's chains read."""

ExposureBasis = Literal["data", "judgment", "normalisation"]
"""What an exposure base's count rests on.

``data``: the base unit is the population from which opportunities arise, and
a cited source measures its count or its share of a stipulated scale.
``judgment``: the base unit is that population, and its count is an authored
estimate, such as a share of a stipulated scale, a fixed number or a
stipulated scale taken as the population. ``normalisation``: the base unit only
measures the size of the lab or deployment, because no stipulated scale counts
the population from which opportunities arise; the opportunity rate carries the
whole exposure judgment, and the scenario has no governed exposure denominator,
which its coverage-gap row records.
"""


class DerivedBaseCount(StrictModel):
    """A base count derived from one of the lab's stipulated deployment scales.

    The count is the scale's stipulated level on the instrument, which on a
    public trajectory moves with the trajectory, times ``factor``
    (``FREQ-CHAIN-EXPOSURE``). ``rationale`` says why the factor holds and
    ``source`` what supports it.
    """

    kind: Literal["derived"]
    scale_id: Identifier
    factor: PositiveFinite
    rationale: Prose
    source: Prose


class FixedBaseCount(StrictModel):
    """A base count authored in the scenario library as one number at every
    anchor and trajectory."""

    kind: Literal["fixed"]
    count: PositiveFinite
    rationale: Prose
    source: Prose


class ExposureBaseDef(StrictModel):
    """The natural unit a chain's opportunities are counted against.

    ``unit`` names one base unit in the singular (for example ``customer
    organization shipping AI-written code to production``). The respondent
    sees the stipulated number of base units, computed by code from
    ``count``, beside the one-sentence ``derivation``, and states only the
    expected number of opportunities per base unit; it never converts tokens
    or users into opportunities itself. ``basis`` says what the count rests on;
    the respondent never sees it.
    """

    unit: Prose
    basis: ExposureBasis
    count: Annotated[DerivedBaseCount | FixedBaseCount, Field(discriminator="kind")]
    derivation: Prose


class EventChainExposureDef(StrictModel):
    """A chain's exposure base for each capability access.

    One scenario is asked of the lab's public deployment and of its internal
    and pre-release use, whose populations differ, so each access names its
    own base. The opportunity rate covers every opportunity of the named lab
    and deployment whatever its channel; the base fixes the unit in which it
    is stated and, through its stipulated count, the expected opportunity
    count.
    """

    public: ExposureBaseDef
    internal: ExposureBaseDef

    def base(self, access: CapabilityAccess) -> ExposureBaseDef:
        return self.public if access == "public" else self.internal


ChainAuthoringStatus = Literal["authored"]
"""That a chain's exposure bases, opportunity and stages are authored content;
reported with chain results, never shown to respondents."""


class EventChainStageDef(StrictModel):
    """One conditional event, given the opportunity and every earlier stage."""

    stage_id: Identifier
    event: Prose


class EventChainDef(StrictModel):
    """Bottom-up event chain ending at the scenario's qualified terminal unit.

    The expected annual qualified count equals the expected opportunity count
    times the product of the stages' conditional probabilities, because each
    opportunity yields at most one terminal unit and every stage is necessary
    for it; no stage independence is assumed. The expected opportunity count
    is the stipulated count of the access's exposure base times the elicited
    opportunity rate per base unit (``exposure``). The authored stages precede
    the final transition, whose event is the scenario's own terminal unit
    (``event_chain_terminal_stage``) rather than a restatement of it.
    """

    authoring_status: ChainAuthoringStatus = "authored"
    opportunity: EventChainOpportunityDef
    exposure: EventChainExposureDef
    stages: tuple[EventChainStageDef, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_stages(self) -> Self:
        stage_ids = [stage.stage_id for stage in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError(f"event chain stage_id values repeat: {stage_ids}")
        reserved = sorted(set(stage_ids) & _RESERVED_STAGE_IDS)
        if reserved:
            raise ValueError(f"event chain stage_id values are reserved: {reserved}")
        events = [stage.event.casefold() for stage in self.stages]
        if len(events) != len(set(events)):
            raise ValueError("event chain stages must state distinct events")
        opportunity = self.opportunity
        monetary = [
            name
            for name, text in (
                ("opportunity", opportunity.unit),
                ("counting_rule", opportunity.counting_rule),
                ("cohort", opportunity.cohort),
                *((stage.stage_id, stage.event) for stage in self.stages),
            )
            if MONETARY_AMOUNT.search(text)
        ]
        if monetary:
            raise ValueError(
                f"event chain parts {monetary} name a monetary amount; the "
                "admission threshold belongs to the terminal definition"
            )
        return self


class AttributionNexusDef(StrictModel):
    """Provider attribution nexus for one scenario.

    Context-bound questions render ``kind`` and ``terminal_qualification`` as
    data in the scenario's terminal-loss population, beside the code-owned gross
    loss assignment from ``canonical_loss_assignment``.
    """

    kind: Literal["model_contribution", "regulatory_respondent"]
    terminal_qualification: Prose


class FrequencySemanticsDef(StrictModel):
    """The counted population's identity; the event chain owns the opportunity.

    ``counting_identity`` states how terminal units are counted (grouping and
    repeats). ``terminal_event_unit`` defines one qualified unit and states the
    hierarchy contract's admission threshold once, as ``{{admission}}``, which
    loading replaces with ``admission_clause``. ``year_rule`` assigns each
    counted unit to one year and is rendered once into every question about the
    population.
    """

    counting_identity: Prose
    terminal_event_unit: Prose
    year_rule: Prose

    @field_validator("terminal_event_unit")
    @classmethod
    def state_admission_threshold(cls, value: str) -> str:
        # A validated unit holds the expanded clause, so it validates again
        # unchanged; a clause for another threshold no longer matches.
        clause = admission_clause()
        if value.count(ADMISSION_MARKER) + value.count(clause) != 1:
            raise ValueError(
                "terminal_event_unit must state the admission threshold exactly "
                f"once, as {ADMISSION_MARKER}"
            )
        return value.replace(ADMISSION_MARKER, clause)


CompositionSubject = Literal["third_party", "provider", "other"]

AttachmentBasis = Literal["incident_consequence", "regulatory_scenario"]
"""Why a tracked composition cell's loss is its scenario's.

``regulatory_scenario``: the scenario is a government matter against the
provider, and the cell is one of that matter's fines, penalties or costs.
``incident_consequence``: the scenario is an incident, and the cell is a loss
the provider bears because of it, in this library a civil fine, which the
hierarchy contract's ownership convention keeps with the incident.
"""

type AccountingProse = Annotated[Prose, AfterValidator(expand_accounting_text)]
"""Prose that may instead be one ``{{accounting:<name>}}`` reference."""


class CompositionCellDef(StrictModel):
    """One loss-form and bearer cell of the severity estimand (``ACC-COMPOSITION``).

    Respondents estimate only each cell's share of one incident's gross loss;
    ``nature`` and ``subject`` are authored (``ACC-WELFARE``). A cell with a
    ``loss_type`` is tracked: ``attachment_basis`` records why its loss is the
    scenario's (``AttachmentBasis``), and ``{{loss_component:<cell_id>}}`` in
    the severity scope renders its loss form. ``rationale`` records why a cell
    is classified as it is where its loss form does not say.
    ``{{accounting:<name>}}`` as a whole field takes shared text from
    ``catmodel.scenarios.composition_text``.
    """

    cell_id: Identifier
    loss_form: AccountingProse
    nature: EconomicNature
    subject: CompositionSubject
    rationale: AccountingProse | None = None
    loss_type: Identifier | None = None
    attachment_basis: AttachmentBasis | None = None

    @model_validator(mode="after")
    def validate_tracking(self) -> Self:
        if (self.loss_type is None) != (self.attachment_basis is None):
            raise ValueError(
                "a tracked cell names both its loss_type and attachment_basis"
            )
        if self.loss_type == "civil_fines" and self.nature != "transfer":
            raise ValueError(
                "civil fines must be transfers with zero aggregate welfare weight"
            )
        if (
            self.attachment_basis == "incident_consequence"
            and self.subject != "provider"
        ):
            raise ValueError("incident_consequence cells must be provider-borne")
        return self

    @property
    def tracked(self) -> bool:
        return self.loss_type is not None


class EstimandCompositionDef(StrictModel):
    """Economic classification of the severity estimand (``ACC-COMPOSITION``).

    The cells are a roster whose shares the contextual instrument elicits; no
    share is authored. A scenario either names a roster of
    ``composition_rosters.yaml``, optionally restating some cells' loss forms in
    ``loss_forms``, or authors ``status``, ``rationale``, ``evidence`` and
    ``cells`` itself. Loading expands a named roster, so ``roster`` records
    which one a composition came from.
    """

    roster: Identifier | None = None
    status: AssumptionStatus
    rationale: Prose
    evidence: Prose | None = None
    cells: tuple[CompositionCellDef, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_cells(self) -> Self:
        cell_ids = [cell.cell_id for cell in self.cells]
        duplicates = sorted({cell for cell in cell_ids if cell_ids.count(cell) > 1})
        if duplicates:
            raise ValueError(f"composition cell_id values repeat: {duplicates}")
        labels = [(cell.loss_form, cell.subject, cell.loss_type) for cell in self.cells]
        if len(labels) != len(set(labels)):
            raise ValueError("composition loss-form and bearer cells must be unique")
        return self

    @property
    def tracked_cells(self) -> tuple[CompositionCellDef, ...]:
        return tuple(cell for cell in self.cells if cell.tracked)


class ScenarioLineageDef(StrictModel):
    """Taxonomy lineage for one scenario cell.

    ``introduced_in`` records the taxonomy generation in which the cell joined
    the scenario roster; predecessor/successor relationships belong to
    ``transitions`` and ``retired_scenarios`` in
    ``registries/scenario_lineage.yaml``.
    """

    introduced_in: Prose


class SeverityGovernanceDef(StrictModel):
    """Executable policy for constructing one scenario's severity tail.

    Severity quantiles come from context-bound elicitation answers; this block
    owns the governed tail shape and the cap posture applied to them.

    ``severity_scope_exclusions`` encodes out-of-estimand loss classes for audit;
    it does not replace ``severity_occurrence_scope``.

    ``severity_cap_uncapped_stress_alternative`` requests the uncapped stress
    alternative for a finite-capped scenario: the authored cap is removed at the
    governed shape without introducing a numerical bound, and infinite-variance
    laws report conventional precision as unavailable.
    ``uncapped_endpoint_stress`` is the authored finite endpoint for the
    finite-endpoint stress of an uncapped scenario. ``severity_cap_basis``
    records why a cap or endpoint has its value.

    ``tail_class`` names the shared tail class whose shape a repository scenario
    adopts (``severity_governance_policies.yaml``), and ``tail_class_rationale``
    states in one line why the scenario belongs to it.
    ``endpoint_from_tail_class`` marks a cap that is the class's endpoint
    convention for the whole event, a valuation and truncation convention
    rather than a physical bound, which also bounds the contextual lab law; a
    scenario-authored cap such as a statutory fine maximum bounds only the
    scenario's own severity population.

    A governed ``gpd_shape`` must admit both required tail-shape sensitivity
    alternatives (``TAIL_SHAPE_SENSITIVITY_STEP`` lower and higher) inside the
    runtime shape bounds, so the shape bounds never make a required alternative
    unavailable, and the uncapped stress alternative must keep a finite mean at
    the governed shape.
    """

    gpd_shape: float = Field(allow_inf_nan=False)
    tail_class: Identifier | None = None
    tail_class_rationale: Prose | None = None
    cap_posture: Literal["finite", "uncapped"]
    severity_cap: PositiveFinite | None = None
    severity_cap_sensitivity_values: tuple[float, float] | None = None
    severity_cap_uncapped_stress_alternative: bool = False
    uncapped_endpoint_stress: PositiveFinite | None = None
    severity_cap_basis: Prose | None = None
    endpoint_from_tail_class: bool = False
    severity_scope_exclusions: tuple[SeverityScopeExclusionClass, ...] = ()
    source_classification: SourceClassification
    status: AssumptionStatus
    materiality: AssumptionMateriality
    rationale: Prose
    empirical_grounding: Prose
    sensitivity_range: Prose
    evidence_refs: tuple[Text, ...] = ()
    sensitivity_evidence_refs: tuple[Text, ...] = ()

    @field_validator("severity_scope_exclusions")
    @classmethod
    def validate_severity_scope_exclusions(
        cls, value: tuple[SeverityScopeExclusionClass, ...]
    ) -> tuple[SeverityScopeExclusionClass, ...]:
        if len(value) != len(set(value)):
            raise ValueError("severity_scope_exclusions must be unique")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def validate_tail_class(self) -> Self:
        if (self.tail_class is None) != (self.tail_class_rationale is None):
            raise ValueError(
                "severity_governance.tail_class and tail_class_rationale must be "
                "authored together"
            )
        if self.endpoint_from_tail_class and (
            self.tail_class is None or self.cap_posture != "finite"
        ):
            raise ValueError(
                "severity_governance.endpoint_from_tail_class requires a tail class "
                "and its finite cap"
            )
        if self.severity_cap_basis is not None and (
            self.severity_cap is None and self.uncapped_endpoint_stress is None
        ):
            raise ValueError(
                "severity_governance.severity_cap_basis requires a finite cap or "
                "an uncapped endpoint stress"
            )
        return self

    @model_validator(mode="after")
    def validate_cap_posture(self) -> Self:
        if self.cap_posture == "finite":
            if self.severity_cap is None:
                raise ValueError(
                    "severity_governance.cap_posture='finite' requires a "
                    "positive finite severity_cap"
                )
            if self.severity_cap_sensitivity_values is not None:
                low, high = self.severity_cap_sensitivity_values
                if not (
                    math.isfinite(low)
                    and math.isfinite(high)
                    and 0.0 < low < self.severity_cap < high
                ):
                    raise ValueError(
                        "severity_cap_sensitivity_values must be positive, finite, "
                        "and straddle severity_cap"
                    )
            if self.uncapped_endpoint_stress is not None:
                raise ValueError(
                    "severity_governance.uncapped_endpoint_stress requires "
                    "cap_posture='uncapped'; a finite posture tests its cap "
                    "through severity_cap_sensitivity_values"
                )
        elif self.severity_cap is not None:
            raise ValueError(
                "severity_governance.cap_posture='uncapped' forbids severity_cap"
            )
        elif self.severity_cap_sensitivity_values is not None:
            raise ValueError(
                "severity_governance.cap_posture='uncapped' forbids "
                "severity_cap_sensitivity_values"
            )
        elif self.severity_cap_uncapped_stress_alternative:
            raise ValueError(
                "severity_governance.severity_cap_uncapped_stress_alternative requires "
                "cap_posture='finite'; an uncapped posture already is the "
                "uncapped form"
            )
        return self

    @model_validator(mode="after")
    def validate_tail_shape_alternatives_representable(self) -> Self:
        step = TAIL_SHAPE_SENSITIVITY_STEP
        lower = shifted_tail_shape(self.gpd_shape, -step)
        higher = shifted_tail_shape(self.gpd_shape, step)
        if lower < -UNCAPPED_TAIL_SHAPE_BOUND:
            raise ValueError(
                f"severity_governance.gpd_shape={self.gpd_shape:g}: the lower "
                f"tail-shape sensitivity alternative (gpd_shape - {step:g} = {lower:g}) "
                f"is below the runtime shape bound {-UNCAPPED_TAIL_SHAPE_BOUND:g}"
            )
        # An uncapped law's shape is below the bound, so an alternative
        # exactly at it is not representable either.
        if self.cap_posture == "uncapped" and higher >= UNCAPPED_TAIL_SHAPE_BOUND:
            raise ValueError(
                f"severity_governance.gpd_shape={self.gpd_shape:g}: the higher "
                f"tail-shape sensitivity alternative (gpd_shape + {step:g} = {higher:g}) "
                "is not below the uncapped finite-mean operating bound "
                f"{UNCAPPED_TAIL_SHAPE_BOUND:g}"
            )
        if self.cap_posture == "finite":
            if higher > CAPPED_TAIL_SHAPE_SANITY_BOUND:
                raise ValueError(
                    f"severity_governance.gpd_shape={self.gpd_shape:g}: the "
                    f"higher tail-shape sensitivity alternative (gpd_shape + {step:g} = "
                    f"{higher:g}) exceeds the capped sanity bound "
                    f"{CAPPED_TAIL_SHAPE_SANITY_BOUND:g}"
                )
            if (
                self.severity_cap_uncapped_stress_alternative
                and self.gpd_shape >= UNCAPPED_TAIL_SHAPE_BOUND
            ):
                raise ValueError(
                    f"severity_governance.gpd_shape={self.gpd_shape:g}: the "
                    "uncapped stress alternative needs a governed shape below the "
                    f"uncapped finite-mean operating bound {UNCAPPED_TAIL_SHAPE_BOUND:g}"
                )
        return self

    @property
    def effective_severity_cap_sensitivity_values(self) -> tuple[float, float] | None:
        """Return finite-posture cap alternatives, defaulting to 75%/125%; ``None`` if uncapped."""

        if self.severity_cap is None:
            return None
        if self.severity_cap_sensitivity_values is not None:
            return self.severity_cap_sensitivity_values
        return (0.75 * self.severity_cap, 1.25 * self.severity_cap)


class ScenarioDefinition(StrictModel):
    id: AtomicIdentifier
    name: Prose
    # One forecast owns the disjoint union of these terminal populations.
    # Reporting coordinates are common to every member and derive from ancestry.
    node_ids: tuple[Annotated[str, Field(pattern=r"^n_[a-z0-9_]+$")], ...] = Field(
        min_length=1
    )
    lineage: ScenarioLineageDef
    description: Text
    # Bottom-up chain from an annual opportunity to the qualified terminal unit;
    # a separate frequency question beside the direct qualified count.
    event_chain: EventChainDef
    estimand_composition: EstimandCompositionDef
    severity_occurrence_scope: Prose
    severity_governance: SeverityGovernanceDef
    frequency_semantics: FrequencySemanticsDef
    attribution_nexus: AttributionNexusDef
    # Reporting facets; neither rendered into questions nor read by any law.
    facets: ScenarioFacetsDef
    # The occurrence unit, which is both the counted terminal unit and the unit
    # severity measures; reported in the catalog, not rendered.
    occurrence_unit_id: Identifier
    # What evidence exists for this population and what it does not establish;
    # reported in the catalog, not rendered.
    empirical_grounding: Prose

    @field_validator("node_ids")
    @classmethod
    def unique_node_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("scenario node_ids must be unique")
        return value

    @field_validator("severity_occurrence_scope")
    @classmethod
    def validate_severity_occurrence_scope(cls, value: str) -> str:
        if "consequential civil fines" in value.lower():
            raise ValueError(
                "severity_occurrence_scope contains renderer-reserved civil-fine "
                "prose; use the tracked-component marker instead"
            )
        return value

    @property
    def risk_domain(self) -> str:
        return scenario_placement_coordinates(
            cached_hierarchy_contract(), self.node_ids
        )[2]

    @property
    def normalisation_bases(self) -> tuple[CapabilityAccess, ...]:
        """The capability accesses whose exposure base is only a normalisation."""
        exposure = self.event_chain.exposure
        accesses: tuple[CapabilityAccess, ...] = ("public", "internal")
        return tuple(
            access
            for access in accesses
            if exposure.base(access).basis == "normalisation"
        )

    @model_validator(mode="after")
    def validate_facet_coherence(self) -> Self:
        violations = scenario_facet_violations(self)
        if violations:
            raise ValueError(
                f"{self.id}: facet coherence violations: " + "; ".join(violations)
            )
        return self

    @model_validator(mode="after")
    def validate_event_chain(self) -> Self:
        # The admission clause is the contract's, so a stage restates the
        # terminal unit when it restates the rest of its predicate.
        terminal = _word_set(
            qualified_terminal_unit(self).replace(admission_clause(), "")
        )
        restated = [
            stage.stage_id
            for stage in self.event_chain.stages
            if _jaccard(_word_set(stage.event), terminal) >= _RESTATEMENT_JACCARD
        ]
        if restated:
            raise ValueError(
                f"{self.id}: event chain stages {restated} restate the terminal "
                "unit; the final stage is rendered from it"
            )
        exposure = self.event_chain.exposure
        unknown = sorted(
            {
                base.count.scale_id
                for base in (exposure.public, exposure.internal)
                if isinstance(base.count, DerivedBaseCount)
            }
            - set(EXPOSURE_SCALES)
        )
        if unknown:
            raise ValueError(
                f"{self.id}: exposure bases read undeclared scales {unknown}; "
                f"declared scales are {sorted(EXPOSURE_SCALES)}"
            )
        return self

    @model_validator(mode="after")
    def validate_one_admission_threshold(self) -> Self:
        """The contract's admission threshold is the only amount that defines an
        incident: no other monetary amount may appear in the terminal unit,
        counting identity, year rule or attribution nexus."""
        semantics = self.frequency_semantics
        texts = {
            "terminal_event_unit": semantics.terminal_event_unit.replace(
                admission_clause(), ""
            ),
            "counting_identity": semantics.counting_identity,
            "year_rule": semantics.year_rule,
            "attribution_nexus": self.attribution_nexus.terminal_qualification,
        }
        monetary = [
            name for name, text in texts.items() if MONETARY_AMOUNT.search(text)
        ]
        if monetary:
            raise ValueError(
                f"{self.id}: {monetary} name a monetary amount other than the "
                "contract's admission threshold"
            )
        return self

    @model_validator(mode="after")
    def validate_severity_scope_markers(self) -> Self:
        """The severity scope marks each tracked composition cell exactly once
        and no other cell."""
        tracked = self.estimand_composition.tracked_cells
        markers = set(_SCOPE_MARKER.findall(self.severity_occurrence_scope))
        unknown = sorted(markers - {cell.cell_id for cell in tracked})
        if unknown:
            raise ValueError(
                f"severity_occurrence_scope marks {unknown}, which are not tracked "
                "composition cells"
            )
        for cell in tracked:
            count = self.severity_occurrence_scope.count(
                loss_component_scope_marker(cell.cell_id)
            )
            if count != 1:
                raise ValueError(
                    "every tracked cell appears once as a marker in "
                    f"severity_occurrence_scope; cell={cell.cell_id!r}, count={count}"
                )
        return self


_SCOPE_MARKER = re.compile(r"\{\{loss_component:([a-z][a-z0-9_]*)\}\}")


def loss_component_scope_marker(component_id: str) -> str:
    return f"{{{{loss_component:{component_id}}}}}"


def rendered_severity_occurrence_scope(scenario: ScenarioDefinition) -> str:
    """The severity scope with each tracked cell's marker replaced by its loss form."""
    rendered = scenario.severity_occurrence_scope
    for cell in scenario.estimand_composition.tracked_cells:
        rendered = rendered.replace(
            loss_component_scope_marker(cell.cell_id), cell.loss_form
        )
    return rendered


def qualified_terminal_unit(scenario: ScenarioDefinition) -> str:
    """The counted terminal unit."""
    return scenario.frequency_semantics.terminal_event_unit


def event_chain_terminal_stage(scenario: ScenarioDefinition) -> EventChainStageDef:
    """The final chain transition, whose event is the qualified terminal unit."""
    return EventChainStageDef(
        stage_id=EVENT_CHAIN_TERMINAL_STAGE_ID,
        event=qualified_terminal_unit(scenario),
    )


def canonical_loss_assignment() -> str:
    """Return the code-owned gross-loss assignment every population uses.

    Every population counts gross harm: the whole incident's loss before claim
    assertion, payment, liability allocation or coverage.
    """
    return (
        "Once the terminal event qualifies, attach every distinct quantifiable gross "
        "loss component within the authored severity scope once, before claim "
        "assertion, liability allocation, coverage, exclusions, retentions, limits, "
        "or reinsurance. The whole incident and all its consequences, including "
        "claims, fines, legal, recovery and remediation costs, belong to this one "
        "terminal; do not apportion the loss to an incremental AI-caused or "
        "provider-caused share, and do not count any part of it in another "
        "scenario. The authored year rule assigns each qualifying unit to one year, "
        "and the unit carries its ultimate causal loss even if recognized later; do "
        "not move it to a cash-flow, payment or harm-manifestation year that the "
        "rule does not name."
    )


def _derived_materiality_predicate() -> str:
    return (
        f"{admission_statement()} The authored terminal-event predicate states "
        "this one admission threshold; no other reporting, complaint, claim, "
        "insurance, payment, materiality or minimum-dollar threshold applies."
    )


def _derived_below_threshold_treatment() -> str:
    return (
        "An incident below the admission threshold is a disclosed, unestimated "
        "leftover: not zero and not another scenario's incident. Within an "
        "admitted incident, every in-scope loss counts, including the small losses "
        "of individual parties, without creating more incidents."
    )


def effective_loss_population_contract(
    scenario: ScenarioDefinition,
) -> dict[str, Any]:
    """Return the canonical terminal-loss population shared by all instruments.

    It binds the terminal unit, its counting and year rules, attribution, the
    severity scope, the admission threshold and below-threshold treatment.
    """
    semantics = scenario.frequency_semantics
    return {
        "scenario_id": scenario.id,
        "counting_identity": semantics.counting_identity,
        "terminal_event_unit": semantics.terminal_event_unit,
        "year_rule": semantics.year_rule,
        "loss_definition": canonical_loss_assignment(),
        "attribution_nexus": scenario.attribution_nexus.model_dump(mode="json"),
        "severity_occurrence_scope": rendered_severity_occurrence_scope(scenario),
        "severity_minimum": admission_threshold(),
        "materiality_predicate": _derived_materiality_predicate(),
        "below_threshold_treatment": _derived_below_threshold_treatment(),
    }


class TailClassEndpointDef(StrictModel):
    """One finite endpoint convention shared by every scenario in a tail class."""

    severity_cap: PositiveFinite
    severity_cap_sensitivity_values: tuple[float, float]
    basis: Prose

    @model_validator(mode="after")
    def straddles_cap(self) -> Self:
        low, high = self.severity_cap_sensitivity_values
        if not (math.isfinite(low) and math.isfinite(high)) or not (
            0 < low < self.severity_cap < high
        ):
            raise ValueError("endpoint stresses must be finite and straddle the cap")
        return self


class TailClassDef(StrictModel):
    """A named tail class: one governed GPD shape and its evidence.

    A class with an ``endpoint`` also fixes its scenarios' finite cap, cap
    stresses and uncapped stress alternative, so every member shares one convention.
    """

    gpd_shape: float = Field(allow_inf_nan=False)
    rationale: Prose
    empirical_grounding: Prose
    endpoint: TailClassEndpointDef | None = None


class SensitivityRangeByCapPosture(StrictModel):
    finite: Prose
    uncapped: Prose


class SeverityGovernancePolicyDef(StrictModel):
    """Shared severity-governance prose and classification for one policy."""

    rationale: Prose
    empirical_grounding: Prose
    source_classification: SourceClassification
    status: AssumptionStatus
    materiality: AssumptionMateriality
    sensitivity_evidence_refs: tuple[Text, ...]
    sensitivity_range_by_cap_posture: SensitivityRangeByCapPosture


class SeverityGovernancePolicies(StrictModel):
    """Shared severity-governance policies and tail classes."""

    policies: dict[Identifier, SeverityGovernancePolicyDef] = Field(min_length=1)
    tail_classes: dict[Identifier, TailClassDef] = Field(min_length=1)


class CompositionRosterDef(StrictModel):
    """A named composition roster that several scenarios share."""

    status: AssumptionStatus
    rationale: Prose
    evidence: Prose | None = None
    cells: tuple[CompositionCellDef, ...] = Field(min_length=2)


class CompositionRosters(StrictModel):
    rosters: dict[Identifier, CompositionRosterDef] = Field(min_length=1)


@dataclass(frozen=True)
class SharedLibrary:
    """The shared files a scenario directory's scenarios may reference."""

    policies: SeverityGovernancePolicies | None
    rosters: CompositionRosters | None


def _load_shared_file[T: StrictModel](path: Path, model: type[T]) -> T | None:
    if not path.exists():
        return None
    try:
        return model.model_validate(safe_load(path.read_text(encoding="utf-8")))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_severity_governance_policies(
    scenarios_dir: Path,
) -> SeverityGovernancePolicies | None:
    return _load_shared_file(
        scenarios_dir / SEVERITY_GOVERNANCE_POLICIES_FILENAME,
        SeverityGovernancePolicies,
    )


def load_shared_library(scenarios_dir: Path) -> SharedLibrary:
    return SharedLibrary(
        policies=load_severity_governance_policies(scenarios_dir),
        rosters=_load_shared_file(
            scenarios_dir / COMPOSITION_ROSTERS_FILENAME, CompositionRosters
        ),
    )


_POLICY_OWNED_KEYS = frozenset(
    {
        "rationale",
        "empirical_grounding",
        "source_classification",
        "status",
        "materiality",
        "sensitivity_evidence_refs",
        "sensitivity_range",
    }
)
_ENDPOINT_KEYS = frozenset(
    {
        "cap_posture",
        "severity_cap",
        "severity_cap_sensitivity_values",
        "severity_cap_uncapped_stress_alternative",
        "uncapped_endpoint_stress",
        "severity_cap_basis",
        "endpoint_from_tail_class",
    }
)


def _resolved_severity_sensitivity_range(
    block: Mapping[str, Any],
    policy: SeverityGovernancePolicyDef,
    *,
    cap_posture: str,
) -> str:
    """Return the policy's posture range, specialized to authored cap alternatives.

    Authored finite-cap endpoints replace the policy's default cap sentence; the
    tail-shape operation re-derives the GPD scale from the unchanged p95/p99
    anchor.
    """

    cap_values = block.get("severity_cap_sensitivity_values")
    if cap_posture == "finite" and isinstance(cap_values, (list, tuple)):
        low, high = cap_values
        step = f"{TAIL_SHAPE_SENSITIVITY_STEP:.2f}"
        text = (
            "Test the scenario-authored absolute severity_cap endpoints "
            f"{float(low):g} and {float(high):g}, and re-derive the GPD scale from "
            f"the unchanged p95/p99 anchor at gpd_shape {step} lower and higher."
        )
    elif cap_posture == "finite":
        text = policy.sensitivity_range_by_cap_posture.finite
    else:
        text = policy.sensitivity_range_by_cap_posture.uncapped
    if (
        cap_posture == "finite"
        and block.get("severity_cap_uncapped_stress_alternative") is True
    ):
        text += (
            " The uncapped stress alternative removes the authored cap at the governed "
            "shape without introducing a numerical bound; infinite-variance "
            "losses report conventional precision as unavailable."
        )
    return text


def resolve_severity_governance_block(
    block: Mapping[str, Any],
    policies: SeverityGovernancePolicies | None,
) -> dict[str, Any]:
    """Expand a policy reference and tail class into a complete governance block.

    A scenario names its policy; a block without ``policy`` is refused. The
    ``governed_tail_shape`` policy requires a tail class, which owns the
    governed shape; a class with an endpoint convention also owns the cap
    posture.
    """
    if "policy" not in block:
        raise ValueError(
            "severity_governance must reference a shared policy: author "
            "'policy: governed_tail_shape' and a 'tail_class' from "
            f"{SEVERITY_GOVERNANCE_POLICIES_FILENAME} instead of a fully "
            "authored block"
        )
    if policies is None:
        raise ValueError(
            "severity_governance references a policy but "
            f"{SEVERITY_GOVERNANCE_POLICIES_FILENAME} was not found"
        )
    resolved = dict(block)
    policy_id = resolved.pop("policy")
    policy = policies.policies.get(policy_id) if isinstance(policy_id, str) else None
    if policy is None:
        raise ValueError(
            f"unknown severity_governance policy id {policy_id!r}; known ids: "
            f"{sorted(policies.policies)}"
        )
    collisions = sorted(_POLICY_OWNED_KEYS & set(resolved))
    if collisions:
        raise ValueError(
            f"severity_governance policy {policy_id!r} owns {collisions}; do not "
            "author them"
        )
    class_id = resolved.get("tail_class")
    if policy_id == "governed_tail_shape":
        tail_class = (
            policies.tail_classes.get(class_id) if isinstance(class_id, str) else None
        )
        if tail_class is None:
            raise ValueError(
                f"severity_governance policy {policy_id!r} requires a tail_class "
                f"from {sorted(policies.tail_classes)}, got {class_id!r}"
            )
        if "gpd_shape" in resolved:
            raise ValueError(
                f"tail class {class_id!r} owns gpd_shape; do not author it"
            )
        resolved["gpd_shape"] = tail_class.gpd_shape
        if tail_class.endpoint is not None:
            authored = sorted(_ENDPOINT_KEYS & set(resolved))
            if authored:
                raise ValueError(
                    f"tail class {class_id!r} owns its endpoint convention; do not "
                    f"author {authored}"
                )
            endpoint = tail_class.endpoint
            resolved.update(
                cap_posture="finite",
                severity_cap=endpoint.severity_cap,
                severity_cap_sensitivity_values=list(
                    endpoint.severity_cap_sensitivity_values
                ),
                severity_cap_uncapped_stress_alternative=True,
                severity_cap_basis=endpoint.basis,
                endpoint_from_tail_class=True,
            )
    elif class_id is not None:
        raise ValueError(
            f"severity_governance policy {policy_id!r} takes no tail_class"
        )
    if (
        resolved.get("severity_cap") is not None
        or resolved.get("uncapped_endpoint_stress") is not None
    ) and not resolved.get("severity_cap_basis"):
        raise ValueError(
            "a finite severity_cap or uncapped_endpoint_stress requires "
            "severity_cap_basis"
        )
    cap_posture = resolved.get("cap_posture")
    if cap_posture not in {"finite", "uncapped"}:
        raise ValueError(
            f"severity_governance policy {policy_id!r} requires cap_posture to be "
            "'finite' or 'uncapped'"
        )
    resolved.update(
        source_classification=policy.source_classification,
        status=policy.status,
        materiality=policy.materiality,
        rationale=policy.rationale,
        empirical_grounding=policy.empirical_grounding,
        sensitivity_range=_resolved_severity_sensitivity_range(
            resolved, policy, cap_posture=cap_posture
        ),
        sensitivity_evidence_refs=policy.sensitivity_evidence_refs,
    )
    return resolved


def resolve_estimand_composition(
    block: Mapping[str, Any],
    rosters: CompositionRosters | None,
) -> dict[str, Any]:
    """Expand a named roster, applying the scenario's restated loss forms.

    A block without ``roster`` is complete as authored. A named roster fixes
    its cells' identities, natures, bearers and tracking; a scenario may
    restate a cell's loss form in ``loss_forms``.
    """
    if "roster" not in block:
        return dict(block)
    name = block["roster"]
    known = {} if rosters is None else rosters.rosters
    roster = known.get(name) if isinstance(name, str) else None
    if roster is None:
        raise ValueError(
            f"unknown composition roster {name!r}; known rosters: {sorted(known)}"
        )
    unexpected = sorted(set(block) - {"roster", "loss_forms"})
    if unexpected:
        raise ValueError(
            f"a composition that names roster {name!r} may add only loss_forms; "
            f"got {unexpected}"
        )
    loss_forms = block.get("loss_forms") or {}
    if not isinstance(loss_forms, Mapping):
        raise ValueError("estimand_composition.loss_forms must map cell_id to text")
    unknown = sorted(set(loss_forms) - {cell.cell_id for cell in roster.cells})
    if unknown:
        raise ValueError(
            f"loss_forms restate cells {unknown} absent from roster {name!r}"
        )
    return {
        "roster": name,
        **roster.model_dump(mode="json", exclude={"cells"}, exclude_none=True),
        "cells": [
            {
                **cell.model_dump(mode="json", exclude_none=True),
                **(
                    {"loss_form": loss_forms[cell.cell_id]}
                    if cell.cell_id in loss_forms
                    else {}
                ),
            }
            for cell in roster.cells
        ],
    }


def expand_shared_content(
    data: Mapping[str, Any], library: SharedLibrary
) -> dict[str, Any]:
    """Expand a scenario's severity policy and composition roster references."""
    expanded = dict(data)
    governance = data.get("severity_governance")
    if isinstance(governance, Mapping):
        expanded["severity_governance"] = resolve_severity_governance_block(
            governance, library.policies
        )
    composition = data.get("estimand_composition")
    if isinstance(composition, Mapping):
        expanded["estimand_composition"] = resolve_estimand_composition(
            composition, library.rosters
        )
    return expanded


class ScenarioLibraryError(ValueError):
    """Every invalid scenario file of a library, each with its path."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


def _validation_errors(path: Path, exc: Exception) -> list[str]:
    if isinstance(exc, ValidationError):
        return [
            f"{path}: {'.'.join(str(part) for part in error['loc']) or '<root>'}: "
            f"{error['msg']}"
            for error in exc.errors()
        ]
    return [f"{path}: {exc}"]


def scenario_files(scenarios_dir: Path) -> list[Path]:
    """Every scenario YAML file under a library directory."""
    return [
        path
        for path in sorted(scenarios_dir.rglob("*.yaml"))
        if path.name not in SHARED_LIBRARY_FILENAMES
    ]


def load_scenario_files(scenarios_dir: Path) -> dict[Path, ScenarioDefinition]:
    """Load every scenario file, collecting all errors before refusing.

    Raises ``ScenarioLibraryError`` listing each invalid file's errors with its
    path, and each scenario ID that more than one file defines.
    """
    if not scenarios_dir.is_dir():
        raise FileNotFoundError(f"Scenarios directory not found: {scenarios_dir}")
    library = load_shared_library(scenarios_dir)
    loaded: dict[Path, ScenarioDefinition] = {}
    errors: list[str] = []
    for path in scenario_files(scenarios_dir):
        try:
            data = safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(data, Mapping):
                raise ValueError("a scenario file must contain a YAML mapping")
            loaded[path] = ScenarioDefinition.model_validate(
                expand_shared_content(data, library)
            )
        except Exception as exc:
            errors.extend(_validation_errors(path, exc))
    paths_by_id: dict[str, list[Path]] = {}
    for path, scenario in loaded.items():
        paths_by_id.setdefault(scenario.id, []).append(path)
    errors.extend(
        f"scenario id {scenario_id!r} is defined by {[str(path) for path in paths]}"
        for scenario_id, paths in sorted(paths_by_id.items())
        if len(paths) > 1
    )
    if errors:
        raise ScenarioLibraryError(errors)
    return loaded


def load_scenarios(scenarios_dir: Path) -> list[ScenarioDefinition]:
    """Load and validate scenarios in reporting-domain rank and ID order."""
    return sorted(
        load_scenario_files(scenarios_dir).values(),
        key=lambda s: (DOMAIN_SORT_RANK.get(s.risk_domain, 99), s.id),
    )
