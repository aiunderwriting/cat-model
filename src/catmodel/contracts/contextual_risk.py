"""Frozen questions and matched loss judgments for one lab/deployment context."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Protocol, Self, cast, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from catmodel.contracts.capability_evidence import CapabilityPeriod
from catmodel.contracts.components import Fraction
from catmodel.contracts.elicitation import (
    QUANTILES,
    Finite,
    Nonnegative,
    Probability,
    Quantiles,
    Reasoning,
    Respondent,
    SeverityQuantiles,
)
from catmodel.contracts.incident import (
    CapabilityScale,
    DeploymentBinding,
    Entity,
    Population,
    Positive,
)
from catmodel.contracts.primitives import (
    RequiredIdentifier,
    RequiredString,
    Sha256Digest,
)
from catmodel.contracts.public_frontier_condition import PublicTrajectoryCondition
from catmodel.economic_loss import EconomicNature
from catmodel.hashing import canonical_json_sha256
from catmodel.json_io import require_marker
from catmodel.probability.distributions import (
    SeverityDomainError,
    validate_admitted_amount,
)
from catmodel.scenarios.base_rates import DatedEvidenceText
from catmodel.scenarios.loader import (
    EVENT_CHAIN_TERMINAL_STAGE_ID,
    ChainAuthoringStatus,
    DerivedBaseCount,
    EventChainOpportunityDef,
    EventChainStageDef,
    ExposureBaseDef,
    FixedBaseCount,
)
from catmodel.scenarios.sensitivity_coverage import (
    CAPPED_TAIL_SHAPE_SANITY_BOUND,
    UNCAPPED_TAIL_SHAPE_BOUND,
)
from catmodel.scenarios.threat_models import ThreatModel
from catmodel.validation_errors import EvidenceValidationError


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


CONTEXTUAL_SCHEMA_VERSION = 1
"""Version of the contextual instrument and artifact contracts.

A saved instrument or capture of another version is refused before its fields
are read: its questions or its record format differ from the current ones, so
it is recollected. A saved file states its version (``from_saved``); an
authored context and a record built in memory may leave it out, and get the
current one.
"""

_RECOLLECT = "recollect it with a current instrument"


def _current_schema(data: object, label: str) -> object:
    if isinstance(data, Mapping):
        version = cast("Mapping[str, object]", data).get(
            "schema_version", CONTEXTUAL_SCHEMA_VERSION
        )
        if version != CONTEXTUAL_SCHEMA_VERSION:
            raise ValueError(
                f"{label} schema {version!r} is not the current "
                f"{CONTEXTUAL_SCHEMA_VERSION}; {_RECOLLECT}"
            )
    return data


def _require_saved_schema(payload: object, label: str) -> None:
    require_marker(
        payload,
        "schema_version",
        CONTEXTUAL_SCHEMA_VERSION,
        f"{label} schema",
        _RECOLLECT,
    )


class ContextualBranch(_Record):
    branch_id: RequiredIdentifier
    definition: RequiredString


class ContextualComponent(_Record):
    component_id: RequiredIdentifier
    definition: RequiredString
    nature: EconomicNature
    bearer: Literal["provider", "third_party", "other"]
    """The scenario's authored subject or bearer of this cost.

    A ``provider`` component falls on the named lab: its amount from the same
    draw is part of the lab's episode loss, so the lab view never falls
    below the lab-borne slice of the event except where the per-episode cap
    binds. Every other component's amount from the same draw counts in the
    lab's loss at the answer's lab compensation share (``ACC-LAB-BEARER``).
    """
    rank_group_id: RequiredIdentifier | None = None
    role: Literal["independent", "partition"] = Field(
        default="independent", exclude_if=lambda value: value == "independent"
    )
    gpd_shape: Finite | None = None
    severity_cap: Nonnegative | None = None

    @model_validator(mode="after")
    def partition_inherits_total(self) -> Self:
        if self.role == "partition" and any(
            value is not None
            for value in (self.rank_group_id, self.gpd_shape, self.severity_cap)
        ):
            raise ValueError("partition components inherit their total law and rank")
        return self


class ContextualFrequencyChain(_Record):
    """The scenario's authored event chain, frozen into the question.

    ``exposure`` is the scenario's exposure base for the instrument's
    capability access: the natural unit whose stipulated count multiplies the
    elicited opportunity rate per base unit into the expected opportunity
    count (``FREQ-CHAIN-EXPOSURE``). ``stages`` are the authored stages followed by
    the rendered terminal stage (``qualified_terminal``), whose event is the
    node's qualified terminal unit, so the expected opportunity count times
    every stage's conditional probability estimates the expected annual count
    the direct question asks for (``FREQ-EVENT-CHAIN``).
    """

    authoring_status: ChainAuthoringStatus
    """The scenario's chain authoring status, as frozen; reported, never shown
    to the respondent."""
    opportunity: EventChainOpportunityDef
    exposure: ExposureBaseDef
    stages: tuple[EventChainStageDef, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def terminal_last(self) -> Self:
        identities = [stage.stage_id for stage in self.stages]
        if (
            len(identities) != len(set(identities))
            or identities[-1] != EVENT_CHAIN_TERMINAL_STAGE_ID
        ):
            raise ValueError(
                "a frozen chain has unique authored stages and ends at the rendered terminal stage"
            )
        return self

    def asked(self) -> dict[str, object]:
        """What a question shows of the chain or code computes from it.

        The opportunity, the exposure base's unit, count rule and derivation,
        and the stages. The base's basis, rationale and source and the chain's
        authoring status are audit context: revising them changes no question
        and no law, so it leaves earlier answers valid.
        """
        return self.model_dump(
            mode="json",
            exclude={
                "authoring_status": True,
                "exposure": {"basis": True, "count": {"rationale", "source"}},
            },
        )


class SharedTotal(_Record):
    """One native severity source, partitioned after its floor and cap."""

    floor: Nonnegative = 0
    strict: bool = False
    cap: Nonnegative | None = None
    gpd_shape: Finite
    """The scenario's governed GPD shape above p95."""
    large_loss_level: Nonnegative | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """Amount at which the direct question asks the expected annual number of
    admitted incidents (``SEV-LARGE-LOSS-ANCHOR``); its share anchors the tail
    above the respondent's p99."""
    very_large_loss_level: Nonnegative | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """A higher amount whose expected annual count enters no reference law:
    results report it beside the model's own count, and the two-anchor
    comparison takes its share as the tail's second anchor."""
    large_loss_level_exclusions: tuple[RequiredString, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    """Why an authored large-loss level is not asked of this total, such as a
    level at or above its cap, where no loss can reach it."""

    @model_validator(mode="after")
    def admitted_source(self) -> Self:
        if self.cap is not None and (
            self.cap < self.floor or (self.strict and self.cap == self.floor)
        ):
            raise ValueError("shared total cap must preserve its qualifying floor")
        large, very = self.large_loss_level, self.very_large_loss_level
        if very is not None and large is None:
            raise ValueError("the very-large-loss level needs the large-loss level")
        if large is not None:
            if not self.floor < large or (very is not None and not large < very):
                raise ValueError(
                    "large-loss levels must lie above the floor in increasing order"
                )
            if self.cap is not None and not max(large, very or large) < self.cap:
                raise ValueError("large-loss levels must lie below the cap")
        return self


class ContextualRiskNode(_Record):
    node_id: RequiredIdentifier
    population: Population
    qualified_count_definition: RequiredString
    dependence_assumption: RequiredString = "Within each joint branch, components sharing a rank group use one severity rank; distinct rank groups are conditionally independent."
    total: SharedTotal | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    count_family: Literal["poisson", "negative_binomial"] | None = None
    count_variance_to_mean: Nonnegative = 1
    branches: tuple[ContextualBranch, ...] = ()
    components: tuple[ContextualComponent, ...] = ()
    lab_loss_gpd_shape: Finite | None = None
    """Governed GPD shape above p95 of a stochastic lab business-loss law.

    Respondents supply conditional-positive quantiles; code binds this shape,
    which the scenario's severity governance owns. ``None`` when the node
    has no governed shape, as for a scenario whose severity is unavailable: a
    stochastic lab law is then unavailable, while exact zero and fixed amounts
    remain usable. An uncapped lab law needs a shape with a finite mean, as in
    the ``lab_loss_cap`` sensitivity alternative that removes the frontier-lab value
    from a scenario without a class endpoint. A law capped at its class
    endpoint has a finite mean at any shape inside the capped sanity bound,
    such as the catastrophe class's higher tail-shape alternative at 1.0.
    """
    lab_loss_severity_cap: Nonnegative | None = None
    """Endpoint convention applied to one episode's complete lab loss.

    It is the scenario tail class's endpoint for the whole event, so one
    episode's lab loss, provider-borne components plus compensation plus
    business loss, cannot exceed the bound on that event's economic loss.
    Compilation also caps the episode at the stipulated frontier-lab value
    where that is lower (``SEV-LAB-CAP``), which is not part of the question.
    ``None`` means no class endpoint; a scenario-specific cap such as a
    statutory fine maximum bounds a fine, not the lab's other costs, and is not
    copied here.
    """
    frequency_chain: ContextualFrequencyChain | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """Bottom-up route to the qualified count, asked in its own separate call.

    Every collectible node has one. Each opportunity yields at most one
    qualified terminal unit.
    """
    source_cards: tuple[DatedEvidenceText, ...] = ()
    threat_model: ThreatModel | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The scenario's threat model, which the node's questions show
    after its source cards and whose ID an answer may cite as evidence. None
    where the source bundle holds none for the scenario."""
    unavailable_reason: RequiredString | None = None

    @model_validator(mode="after")
    def complete_roster(self) -> Self:
        if self.lab_loss_severity_cap is not None and not (
            0 < self.lab_loss_severity_cap < math.inf
        ):
            raise ValueError("the lab-loss endpoint must be positive and finite")
        if self.lab_loss_gpd_shape is not None:
            shape = self.lab_loss_gpd_shape
            if self.lab_loss_severity_cap is None and not (
                -UNCAPPED_TAIL_SHAPE_BOUND <= shape < UNCAPPED_TAIL_SHAPE_BOUND
            ):
                raise ValueError(
                    "the uncapped lab-loss tail shape must lie inside the finite-mean "
                    f"operating bounds [-{UNCAPPED_TAIL_SHAPE_BOUND:g}, "
                    f"{UNCAPPED_TAIL_SHAPE_BOUND:g})"
                )
            if self.lab_loss_severity_cap is not None and not (
                -UNCAPPED_TAIL_SHAPE_BOUND <= shape <= CAPPED_TAIL_SHAPE_SANITY_BOUND
            ):
                raise ValueError(
                    "the capped lab-loss tail shape must lie inside the operating "
                    f"bounds [-{UNCAPPED_TAIL_SHAPE_BOUND:g}, "
                    f"{CAPPED_TAIL_SHAPE_SANITY_BOUND:g}]"
                )
        if self.unavailable_reason is None and (
            self.count_family is None or not self.branches or not self.components
        ):
            raise ValueError(
                "a collectible node needs count, branch and component definitions"
            )
        if self.unavailable_reason is None and self.frequency_chain is None:
            raise ValueError("a collectible node needs its frequency chain")
        if self.count_family == "poisson" and self.count_variance_to_mean != 1:
            raise ValueError("Poisson counts require variance-to-mean one")
        if (
            self.count_family == "negative_binomial"
            and self.count_variance_to_mean <= 1
        ):
            raise ValueError(
                "negative-binomial counts require variance-to-mean above one"
            )
        for rows, field in (
            (self.branches, "branch_id"),
            (self.components, "component_id"),
            (self.source_cards, "card_id"),
        ):
            keys = [getattr(row, field) for row in rows]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{field} values must be unique within a node")
        if self.threat_model is not None:
            threat_model = self.threat_model
            if threat_model.scenario_id != self.population.scenario_id:
                raise ValueError("a node shows its own scenario's threat model")
            if threat_model.threat_model_id in {
                card.card_id for card in self.source_cards
            }:
                raise ValueError(
                    "a threat model's ID must differ from the node's source card IDs"
                )
        partitions = [row for row in self.components if row.role == "partition"]
        if bool(partitions) != (self.total is not None):
            raise ValueError(
                "a shared total and its partition components must be supplied together"
            )
        if self.total is not None and (
            self.total.strict or len(partitions) != len(self.components)
        ):
            raise ValueError(
                "a shared total is the whole estimand: its floor is inclusive and "
                "every component is a partition share of it"
            )
        if (
            self.total is not None
            and self.population.threshold is not None
            and (
                self.total.floor != self.population.threshold
                or self.total.strict != (self.population.threshold_comparator == "gt")
            )
        ):
            raise ValueError(
                "shared total floor must preserve the population qualification"
            )
        return self


class ContextualParentNode(_Record):
    """A taxonomy parent asked for its whole-category expected annual societal harm.

    ``children`` lists its modeled children: collectible node IDs and the
    parent IDs of declared parents below it. ``unmodeled_children`` labels
    each category directly beneath it in the taxonomy that no modeled child
    covers, whose loss its remainder estimates; when it is empty, the
    remainder covers only losses the children's definitions miss. ``level``
    labels the root and the parents beneath it; the pool does not depend on
    it. ``threshold`` states the one admission threshold that every leaf
    shares (``HIER-THRESHOLD``).
    """

    parent_id: RequiredIdentifier
    level: Literal["root", "branch"]
    label: RequiredString
    definition: RequiredString
    threshold: RequiredString
    children: tuple[RequiredIdentifier, ...]
    unmodeled_children: tuple[RequiredString, ...]


class ExposureScale(_Record):
    """One stipulated deployment scale: what one unit of it counts, its level
    at the forecast as-of date and its study-year average on the forecast's
    median path."""

    scale_id: RequiredIdentifier
    unit: RequiredString
    """What one unit of the scale counts, as a frequency chain's question
    states it, worded as ``use_population`` words it: for example ``tokens
    per day processed through the API``."""
    as_of: Positive
    study_year_median: Positive

    @model_validator(mode="after")
    def growth_only(self) -> Self:
        # Fast trajectories render up to M times the median-path change, so a
        # falling scale would fall up to M times as far and could render non-positive.
        if self.study_year_median < self.as_of:
            raise ValueError(
                "a path-linked scale's study-year level cannot be below its as-of level"
            )
        return self


class ExposureVariant(_Record):
    """An exposure-scale sensitivity variant: every rendered scale times ``factor``."""

    variant_id: RequiredIdentifier
    factor: Positive


SAFEGUARD_LEVEL_FIELDS: Mapping[str, tuple[str, ...]] = {
    "public": (
        "fixed_controls",
        "deployment_binding.tools_and_permissions",
        "deployment_binding.human_oversight",
    ),
    "not_yet_public": (
        "fixed_controls",
        "deployment.definition",
        "deployment_binding.use_population",
        "deployment_binding.tools_and_permissions",
        "deployment_binding.human_oversight",
    ),
}
"""The stipulated safeguard text a ``safeguard_level`` variant replaces, by
source status (``POP-SAFEGUARD-LEVEL``).

Scope, permissions and levers: the shared controls; each status's tools and
permissions and human oversight; and, for internal and pre-release use, the
deployment definition and use population that bound what internal agents are
used for. A not-yet-public ``use_population`` keeps its scale placeholders.
"""


class PathLinkedExposure(_Record):
    """Deployment scales that move with the public trajectory.

    Each ``{scale_id}`` in a deployment binding's ``use_population`` is a
    placeholder. On a trajectory node with progress ratio q, its frontier
    progress over the base forecast's median progress, both measured at the
    middle of the policy period, ``public-study`` renders the scale as
    ``as_of + (study_year_median - as_of) * path_growth_multiple(q, M)`` to
    two significant figures, where M is ``max_growth_multiple``: growth is
    proportional to q up to the median path and saturates above it, never
    exceeding M times the median-path growth before rounding. It records q
    and keeps the unrendered text as ``use_population_template``; the
    no-progress node has q = 0. An ``exposure_scale`` variant multiplies that level
    by its factor before rounding. An instrument without a public trajectory keeps its placeholders
    and cannot be asked.
    """

    scales: tuple[ExposureScale, ...] = Field(min_length=1)
    max_growth_multiple: Finite = Field(gt=1)
    variant: ExposureVariant | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    progress_ratio: Nonnegative | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    use_population_template: RequiredString | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @property
    def placeholders(self) -> tuple[str, ...]:
        return tuple("{" + row.scale_id + "}" for row in self.scales)

    def value(self, scale: ExposureScale) -> float:
        """A scale's rendered value at this instrument's progress ratio."""
        if self.progress_ratio is None:
            raise ValueError("path-linked exposure has no progress ratio yet")
        exact = scale.as_of + (scale.study_year_median - scale.as_of) * (
            path_growth_multiple(self.progress_ratio, self.max_growth_multiple)
        )
        factor = 1.0 if self.variant is None else self.variant.factor
        return float(f"{exact * factor:.2g}")


class FixedExposureScale(_Record):
    """One stipulated deployment scale held at one level at every anchor or
    trajectory; ``use_population`` states that level."""

    scale_id: RequiredIdentifier
    unit: RequiredString
    level: Positive


BASE_COUNT_SIGNIFICANT_FIGURES = 3
"""Significant figures of a stipulated base count, which the chain question
shows as a number and code multiplies by the elicited rate."""


@dataclass(frozen=True)
class ChainExposure:
    """The stipulated exposure base one frequency chain's opportunity rate is
    stated against (``FREQ-CHAIN-EXPOSURE``).

    ``count`` is the stipulated number of base units, each one ``unit``: a
    fixed count as authored, or a derived count, the instrument's level of
    the scale ``scale_id`` (on a public trajectory, the scale rendered at the
    node's progress ratio, as ``use_population`` shows it) times ``factor``.
    Either is rounded to ``BASE_COUNT_SIGNIFICANT_FIGURES``, and the question
    shows exactly the number code multiplies. The expected opportunity count
    is ``count`` times the elicited rate per base unit.
    """

    unit: str
    count: float
    derivation: str
    scale_id: str | None = None
    scale_level: float | None = None
    factor: float | None = None

    @classmethod
    def of(cls, base: ExposureBaseDef, levels: Mapping[str, float]) -> ChainExposure:
        """``base`` read against the instrument's stipulated scale ``levels``."""
        rule = base.count
        if isinstance(rule, FixedBaseCount):
            return cls(base.unit, _significant_count(rule.count), base.derivation)
        level = levels[rule.scale_id]
        return cls(
            base.unit,
            _significant_count(level * rule.factor),
            base.derivation,
            rule.scale_id,
            level,
            rule.factor,
        )

    def view(self) -> dict[str, object]:
        """What the chain question shows of it, beside the node's chain."""
        return {
            "unit": self.unit,
            "count": self.count,
            "derivation": self.derivation,
        }


def _significant_count(value: float) -> float:
    return float(f"{value:.{BASE_COUNT_SIGNIFICANT_FIGURES}g}")


def scale_text(value: float) -> str:
    """A rendered scale in words: 1.4 billion, 2.5 million, 6,500 or 1.2."""
    for size, word in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if value >= size:
            return f"{value / size:g} {word}"
    return f"{value:,.0f}" if value >= 100 else f"{value:g}"


def path_growth_multiple(progress_ratio: float, max_growth_multiple: float) -> float:
    """A path-linked scale's growth, as a multiple of its median-path growth,
    on a trajectory whose frontier progress is ``progress_ratio`` times the
    base forecast's median.

    f(x) = x up to the median path (x <= 1), so the no-progress node has 0 and
    the median path 1, and f(x) = 1 + (M - 1) tanh((x - 1) / (M - 1)) above
    it, with M = ``max_growth_multiple`` > 1. f is increasing, has slope 1 and
    no curvature at x = 1, is concave above it and stays below M: before
    rounding, no trajectory's scale grows more than M times its median-path
    growth. Its limit as M grows is the linear link f(x) = x
    (``POP-HYPOTHETICAL-LAB``).
    """
    if progress_ratio <= 1:
        return progress_ratio
    span = max_growth_multiple - 1
    return 1 + span * math.tanh((progress_ratio - 1) / span)


def render_exposure(text: str, exposure: PathLinkedExposure) -> str:
    """Replace every scale placeholder in ``text`` with its rendered value."""
    for scale in exposure.scales:
        text = text.replace(
            "{" + scale.scale_id + "}", scale_text(exposure.value(scale))
        )
    return text


class ContextualRiskInstrument(_Record):
    schema_version: Literal[1] = 1

    @model_validator(mode="before")
    @classmethod
    def current_schema(cls, data: object) -> object:
        return _current_schema(data, "contextual instrument")

    @classmethod
    def from_saved(cls, payload: object) -> Self:
        """The instrument a saved file's decoded JSON holds, which states the
        current schema version."""
        _require_saved_schema(payload, "contextual instrument")
        return cls.model_validate(payload)

    context_id: RequiredIdentifier
    source_bundle_manifest_sha256: Sha256Digest
    period: CapabilityPeriod
    lab: Entity
    model: Entity
    deployment: Entity
    deployment_binding: DeploymentBinding
    capability_scale: CapabilityScale
    capability_domain_id: RequiredIdentifier
    capability_access: Literal["public", "internal"]
    reference_eci: Finite
    integer_min: int = Field(strict=True)
    integer_max: int = Field(strict=True)
    anchor_scores: tuple[Finite, ...] = Field(min_length=1)
    exposure: RequiredString
    fixed_controls: RequiredString
    fixed_adoption: RequiredString
    locality_assumption: RequiredString
    lab_compensation_definition: RequiredString = (
        "Compensation, settlements, refunds and service credits that the named "
        "consolidated lab/group itself pays to others for the same qualifying "
        "initiating incident and all its consequences, before insurance and other "
        "recoveries, as a share of the amount of the node's components whose "
        "bearer is not provider. Those components already count the other "
        "parties' loss in gross loss; the share counts only the part of it that "
        "the lab pays. Amounts in components whose bearer is provider, such as a "
        "fine the lab pays, are counted there and not again here."
    )
    lab_annual_business_loss_definition: RequiredString = (
        "The named consolidated lab/group's own total business loss in one "
        "study year from all of the node's qualifying initiating incidents in "
        "that year and all their consequences, as one combined annual amount, "
        "such as lost revenue and customer churn, net of the costs the lab "
        "avoids, and impairment of its assets, before insurance and other "
        "recoveries, on the incident population's currency and monetary basis. "
        "It is stated for a year in which at least one such incident occurs, "
        "and is zero in a year with none. It excludes compensation the lab pays "
        "to others and every amount within the definition of a node component "
        "whose bearer is provider, even where that amount is lost business or "
        "impairment; those are counted there. Avoid duplicate valuations of the "
        "same economic loss."
    )
    lab_loss_scope: Literal["episode_shares_and_annual_business_loss"] = (
        "episode_shares_and_annual_business_loss"
    )
    """Provider-borne components and compensation count per initiating
    episode; business loss counts once per study year with an episode."""
    lab_annual_business_loss_dependence: Literal["conditional_independence"] = (
        "conditional_independence"
    )
    independent_node_origins_rationale: RequiredString | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    path_linked_exposure: PathLinkedExposure | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    safeguard_level: RequiredIdentifier | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The ``safeguard_level`` variant whose authored safeguard texts replace the
    study's own (``SAFEGUARD_LEVEL_FIELDS``); ``None`` for the study's own
    stipulation. Questions never show it."""
    fixed_exposure: tuple[FixedExposureScale, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    """Stipulated scales held at one level at every anchor or trajectory,
    which the frequency chains read, for a context whose scales are not
    path-linked."""
    coordinate_kind: Literal["constant_path", "public_trajectory"] = "constant_path"
    public_trajectory: PublicTrajectoryCondition | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    interpolation: Literal["linear"] = "linear"
    nodes: tuple[ContextualRiskNode, ...] = Field(min_length=1)
    parents: tuple[ContextualParentNode, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    """Every taxonomy split whose subtree holds an estimated scenario, when
    the instrument covers the complete catalog; each is asked in its own
    call."""
    respondents: tuple[Respondent, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if (self.coordinate_kind == "public_trajectory") != (
            self.public_trajectory is not None
        ):
            raise ValueError("public trajectory mode requires its complete condition")
        if self.public_trajectory is not None:
            condition = self.public_trajectory
            if (
                self.reference_eci != 0
                or self.integer_min != 0
                or self.integer_max != 0
                or self.anchor_scores != (0,)
            ):
                raise ValueError(
                    "public trajectory uses a single zero storage selector, not capability anchors"
                )
            if (condition.times[0], condition.times[-1]) != (
                self.period.policy_period_start_utc,
                self.period.policy_period_end_utc,
            ):
                raise ValueError(
                    "public trajectory must cover exactly the study period"
                )
            general = next(
                row for row in condition.coordinates if row.domain_id == "general"
            )
            if self.capability_scale.scale_id != general.scale_id:
                raise ValueError(
                    "public trajectory context scale must identify public general ECI"
                )
        if (self.lab.kind, self.model.kind, self.deployment.kind) != (
            "lab",
            "model",
            "deployment",
        ):
            raise ValueError(
                "context entities must identify a lab, model and deployment"
            )
        identities = {
            self.lab.entity_id,
            self.model.entity_id,
            self.deployment.entity_id,
        }
        if len(identities) != 3:
            raise ValueError("lab, model and deployment identifiers must differ")
        if (
            self.deployment_binding.deployment_id != self.deployment.entity_id
            or self.deployment_binding.model_id != self.model.entity_id
        ):
            raise ValueError("deployment binding differs from the context entities")
        if not (
            self.model.entity_id in self.lab.related_entity_ids
            or self.lab.entity_id in self.model.related_entity_ids
        ):
            raise ValueError("the model must be explicitly related to the named lab")
        if any(
            not set(entity.related_entity_ids) <= identities
            for entity in (self.lab, self.model, self.deployment)
        ):
            raise ValueError(
                "context entity relations must remain within the declared roster"
            )
        exposure = self.path_linked_exposure
        if (
            self.safeguard_level is not None
            and exposure is not None
            and exposure.variant is not None
        ):
            raise ValueError("an instrument carries at most one context variant")
        if exposure is not None:
            ids = [row.scale_id for row in exposure.scales]
            if len(ids) != len(set(ids)):
                raise ValueError("path-linked exposure scales must be unique")
            if (exposure.progress_ratio is None) != (self.public_trajectory is None):
                raise ValueError(
                    "path-linked exposure has a progress ratio exactly on a public trajectory"
                )
            if (exposure.use_population_template is None) != (
                exposure.progress_ratio is None
            ):
                raise ValueError(
                    "rendered path-linked exposure keeps its use_population template"
                )
            text = self.deployment_binding.use_population
            template = exposure.use_population_template or text
            if not any(token in template for token in exposure.placeholders):
                raise ValueError(
                    "path-linked exposure needs a scale placeholder in use_population"
                )
            if exposure.progress_ratio is not None:
                rendered = render_exposure(template, exposure)
                if "{" in rendered or "}" in rendered:
                    raise ValueError(
                        "rendered path-linked exposure keeps no placeholder"
                    )
                if text != rendered:
                    raise ValueError(
                        "use_population must be its template rendered at the progress ratio"
                    )
        if self.fixed_exposure:
            ids = [row.scale_id for row in self.fixed_exposure]
            if exposure is not None:
                raise ValueError(
                    "an instrument's scales are fixed or path-linked, not both"
                )
            if len(ids) != len(set(ids)):
                raise ValueError("fixed exposure scales must be unique")
            if any(
                scale_text(row.level) not in self.deployment_binding.use_population
                for row in self.fixed_exposure
            ):
                raise ValueError(
                    "use_population must state every fixed exposure level as "
                    "scale_text renders it"
                )
        shown = self._shown_scales()
        for node in self.collectible_nodes:
            chain = node.frequency_chain
            rule = None if chain is None else chain.exposure.count
            if isinstance(rule, DerivedBaseCount) and rule.scale_id not in shown:
                raise ValueError(
                    f"node {node.node_id!r}: its frequency chain's base count "
                    f"derives from {rule.scale_id!r}, which must be a stipulated "
                    "scale that use_population shows"
                )
        if self.integer_min > self.integer_max:
            raise ValueError("modeled integer range must be nonempty")
        if list(self.anchor_scores) != sorted(set(self.anchor_scores)):
            raise ValueError("anchor scores must be distinct and increasing")
        if not (
            self.anchor_scores[0]
            <= self.integer_min
            <= self.integer_max
            <= self.anchor_scores[-1]
            and self.anchor_scores[0] <= self.reference_eci <= self.anchor_scores[-1]
        ):
            raise ValueError(
                "anchor hull must cover the integer range and exact reference ECI"
            )
        if self.reference_eci not in self.anchor_scores:
            raise ValueError("the exact reference ECI must be an anchor")
        for values in (
            [node.node_id for node in self.nodes],
            [node.population.scenario_id for node in self.nodes],
            [respondent.key() for respondent in self.respondents],
            [
                (respondent.model_id, respondent.sample_index)
                for respondent in self.respondents
            ],
        ):
            if len(values) != len(set(values)):
                raise ValueError(
                    "node, population and respondent identities must be unique"
                )
        if self.parents:
            identities = [row.parent_id for row in self.parents]
            owned = [child for row in self.parents for child in row.children]
            collectible = {node.node_id for node in self.collectible_nodes}
            roots = set(identities) - set(owned)
            children = {row.parent_id: row.children for row in self.parents}
            reached, frontier = set(roots), list(roots)
            while frontier:
                for child in children.get(frontier.pop(), ()):
                    if child in children and child not in reached:
                        reached.add(child)
                        frontier.append(child)
            if (
                len(identities) != len(set(identities))
                or len(owned) != len(set(owned))
                or len(roots) != 1
                or set(owned) != (set(identities) - roots) | collectible
                or reached != set(identities)
                or any(
                    (row.level == "root") != (row.parent_id in roots)
                    for row in self.parents
                )
            ):
                raise ValueError(
                    "taxonomy parents need to form one tree whose root, the only parent at level root, is no parent's child, with every other parent and every collectible node the child of exactly one parent"
                )
        for node in self.nodes:
            if node.population.deployment_ids != (self.deployment.entity_id,):
                raise ValueError(
                    "each node population must name this deployment exclusively"
                )
            if any(
                not card.eligible(self.period.model_dump(mode="json"))
                for card in node.source_cards
            ):
                raise ValueError("source cards must satisfy the frozen period cutoffs")
            if node.threat_model is not None and not node.threat_model.eligible(
                self.period.model_dump(mode="json")
            ):
                raise ValueError(
                    "a threat model must satisfy the frozen period cutoffs"
                )
        return self

    def _shown_scales(self) -> frozenset[str]:
        """Every stipulated scale that use_population shows: a path-linked
        scale whose placeholder its template holds, or a fixed scale, whose
        level validation finds in it."""
        exposure = self.path_linked_exposure
        if exposure is None:
            return frozenset(row.scale_id for row in self.fixed_exposure)
        template = (
            exposure.use_population_template or self.deployment_binding.use_population
        )
        return frozenset(
            row.scale_id
            for row in exposure.scales
            if "{" + row.scale_id + "}" in template
        )

    def chain_exposure(self, chain: ContextualFrequencyChain) -> ChainExposure:
        """The stipulated exposure base of ``chain`` on this instrument: a
        derived count reads the scale level ``use_population`` states, which
        on a public trajectory is the scale rendered at this node's progress
        ratio."""
        return ChainExposure.of(chain.exposure, self.scale_levels())

    def scale_levels(self) -> dict[str, float]:
        """Every stipulated scale's level on this instrument."""
        exposure = self.path_linked_exposure
        if exposure is not None:
            return {row.scale_id: exposure.value(row) for row in exposure.scales}
        return {row.scale_id: row.level for row in self.fixed_exposure}

    @property
    def collectible_nodes(self) -> tuple[ContextualRiskNode, ...]:
        return tuple(node for node in self.nodes if node.unavailable_reason is None)

    def digest(self) -> str:
        """The instrument's identity, computed once per instrument object: a
        frozen instrument cannot change, so its object fixes its digest."""
        cached = _INSTRUMENT_DIGESTS.get(id(self))
        if cached is not None:
            return cached[1]
        value = canonical_json_sha256(self.model_dump(mode="json"))
        if len(_INSTRUMENT_DIGESTS) >= _INSTRUMENT_DIGEST_SLOTS:
            _INSTRUMENT_DIGESTS.pop(next(iter(_INSTRUMENT_DIGESTS)))
        _INSTRUMENT_DIGESTS[id(self)] = (self, value)
        return value


_INSTRUMENT_DIGEST_SLOTS = 32
_INSTRUMENT_DIGESTS: dict[int, tuple[ContextualRiskInstrument, str]] = {}
"""Digests of recently used instruments, keyed by object identity; each entry
keeps its instrument alive, so an identity is never reused while cached.
Thirty-two covers every instrument of a default public study (16) with room
for its exposure variants. Only one thread uses it: collection calls ``digest``
on its event-loop thread, and its worker threads (chain numerics) never do,
so the evict-then-insert pair needs no lock."""


@lru_cache(maxsize=64)
def _model_schema(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema())


def json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A fresh copy of ``model``'s JSON schema, generated once per model."""
    return json.loads(_model_schema(model))


class _StatusAnswer(Protocol):
    @property
    def status(self) -> str: ...


@dataclass(frozen=True)
class StatusFields:
    """Which fields each status of an answer owns.

    An answer's ``reason`` field explains its ``reason_status`` (an
    unavailable answer, or a cross-check that keeps every answer) and is
    given exactly under that status. Under ``answered_status`` the reason is
    null or absent, every field in ``required`` is supplied and, where
    ``alternatives`` names fields, exactly one of them is supplied and the
    others are null; under ``reason_status`` every field in ``excluded`` is
    null or empty.

    The same rule is stated twice from this one definition: as the answer
    schema's ``anyOf`` of status variants, which the question shows (pass
    the instance as the model's ``json_schema_extra``), and by ``validate``,
    whose errors name the field and status so that a validation retry says
    which field to change. Each variant names its status as a constant and
    lists its own required and empty fields, so it reads the same in any key
    order. ``require`` adds the fields one question's answer must supply
    beyond these, as ``validate_against`` checks them.
    """

    answered_status: str
    reason_status: str
    reason: str
    required: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()

    def __call__(self, schema: dict[str, Any]) -> None:
        properties = schema["properties"]

        def answered(chosen: Mapping[str, Any], selected: tuple[str, ...]) -> dict:
            return {
                "properties": {
                    "status": {"const": self.answered_status},
                    self.reason: {"type": "null"},
                    **{
                        name: _supplied_schema(properties[name])
                        for name in self.required
                    },
                    **chosen,
                },
                "required": ["status", *self.required, *selected],
            }

        variants = (
            [
                answered(
                    {
                        name: _supplied_schema(properties[name])
                        if name == selected
                        else {"type": "null"}
                        for name in self.alternatives
                    },
                    (selected,),
                )
                for selected in self.alternatives
            ]
            if self.alternatives
            else [answered({}, ())]
        )
        variants.append(
            {
                "properties": {
                    "status": {"const": self.reason_status},
                    self.reason: _supplied_schema(properties[self.reason]),
                    **{name: _empty_schema(properties[name]) for name in self.excluded},
                },
                "required": ["status", self.reason],
            }
        )
        schema["anyOf"] = variants

    def require(
        self,
        schema: dict[str, Any],
        supplied: tuple[str, ...] = (),
        empty: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """``schema`` whose answered variants also supply ``supplied`` and
        leave ``empty`` null: the fields one question's answer owns beyond
        this rule, as its ``validate_against`` checks them."""
        properties = schema["properties"]
        for variant in schema["anyOf"]:
            if variant["properties"]["status"]["const"] == self.answered_status:
                for name in supplied:
                    variant["properties"][name] = _supplied_schema(properties[name])
                    variant["required"].append(name)
                for name in empty:
                    variant["properties"][name] = _empty_schema(properties[name])
        return schema

    def validate(self, answer: _StatusAnswer) -> None:
        status = answer.status
        reason_given = getattr(answer, self.reason) is not None
        if status == self.reason_status:
            fields = [(self.reason, "required", not reason_given)] + [
                (name, "not_allowed", _supplied(getattr(answer, name)))
                for name in self.excluded
            ]
        else:
            fields = [(self.reason, "not_allowed", reason_given)] + [
                (name, "required", not _supplied(getattr(answer, name)))
                for name in self.required
            ]
        for name, rule, broken in fields:
            if broken:
                raise EvidenceValidationError(
                    f"{name}_{rule}_when_{status}",
                    f"{name} is {rule.replace('_', ' ')} when status is {status}",
                    location=(name,),
                )
        if (
            self.alternatives
            and status == self.answered_status
            and sum(_supplied(getattr(answer, name)) for name in self.alternatives) != 1
        ):
            raise EvidenceValidationError(
                f"exactly_one_of_{'_'.join(self.alternatives)}_required_when_{status}",
                f"exactly one of {', '.join(self.alternatives)} is required when "
                f"status is {status}",
            )


def _supplied_schema(field: Mapping[str, Any]) -> dict[str, Any]:
    """The schema of a status field's supplied value: its non-null form, or
    a nonempty list."""
    if field.get("type") == "array":
        return {"type": "array", "minItems": 1}
    options = [
        option for option in field.get("anyOf", ()) if option != {"type": "null"}
    ]
    if not options:
        raise ValueError("a status field must be nullable or a list")
    return options[0] if len(options) == 1 else {"anyOf": options}


def _empty_schema(field: Mapping[str, Any]) -> dict[str, Any]:
    if field.get("type") == "array":
        return {"type": "array", "maxItems": 0}
    return {"type": "null"}


def _supplied(value: object) -> bool:
    return value is not None and value != ()


def _first_repeat(values: Iterable[object]) -> int | None:
    """The index of the first value equal to an earlier one; None when all
    differ."""
    seen = set()
    for index, value in enumerate(values):
        if value in seen:
            return index
        seen.add(value)
    return None


def _validate_answer_amount(name: str, amount: float) -> None:
    """An answer's fixed ``amount`` lies in the admitted monetary domain
    (``validate_admitted_amount``); the error locates the field."""
    try:
        validate_admitted_amount(name, amount)
    except SeverityDomainError as error:
        raise EvidenceValidationError(
            "amount_outside_admitted_monetary_domain",
            str(error),
            location=("amount",),
        ) from error


UNAVAILABLE_REASON_DESCRIPTION = (
    "Why the answer is unavailable. Given only when status is unavailable; "
    "null or omitted when status is available."
)

_COMPONENT_AMOUNT_FIELDS = ("amount", "loss", "share")
_COMPONENT_STATUS = StatusFields(
    "available",
    "unavailable",
    "unavailable_reason",
    excluded=_COMPONENT_AMOUNT_FIELDS,
    alternatives=_COMPONENT_AMOUNT_FIELDS,
)


class ContextualComponentAnswer(_Record):
    model_config = ConfigDict(json_schema_extra=_COMPONENT_STATUS)

    component_id: RequiredIdentifier
    status: Literal["available", "unavailable"]
    unavailable_reason: RequiredString | None = Field(
        default=None, description=UNAVAILABLE_REASON_DESCRIPTION
    )
    share: Fraction | None = Field(default=None, exclude_if=lambda value: value is None)
    amount: Nonnegative | None = None
    loss: SeverityQuantiles | None = None

    @model_validator(mode="after")
    def explicit_availability(self) -> Self:
        _COMPONENT_STATUS.validate(self)
        if self.amount is not None:
            _validate_answer_amount("component amount", self.amount)
        return self


class ContextualBranchAnswer(_Record):
    branch_id: RequiredIdentifier
    probability: Probability
    components: tuple[ContextualComponentAnswer, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_components(self) -> Self:
        repeat = _first_repeat(row.component_id for row in self.components)
        if repeat is not None:
            raise EvidenceValidationError(
                "component_ids_must_be_unique_within_a_branch",
                "component answers must be unique within a branch",
                location=("components", repeat, "component_id"),
            )
        return self


_LAB_COMPENSATION_STATUS = StatusFields(
    "available",
    "unavailable",
    "unavailable_reason",
    required=("share",),
    excluded=("share",),
)


class LabCompensationAnswer(_Record):
    """The share of an episode's non-provider-borne amount that the lab pays.

    One point share per answer, like a partition component's share: code
    multiplies the same draw's non-provider-borne components by it, so the
    payment is bounded by and moves with the harm it compensates
    (``ACC-LAB-BEARER``).
    """

    model_config = ConfigDict(json_schema_extra=_LAB_COMPENSATION_STATUS)

    status: Literal["available", "unavailable"]
    unavailable_reason: RequiredString | None = Field(
        default=None, description=UNAVAILABLE_REASON_DESCRIPTION
    )
    share: Fraction | None = None

    @model_validator(mode="after")
    def explicit_share(self) -> Self:
        _LAB_COMPENSATION_STATUS.validate(self)
        return self


_LAB_ANNUAL_BUSINESS_LOSS_STATUS = StatusFields(
    "available",
    "unavailable",
    "unavailable_reason",
    required=("positive_probability",),
    excluded=("positive_probability", "amount", "loss"),
)


class LabAnnualBusinessLossAnswer(_Record):
    """The lab's total business loss from a node in a year with an incident.

    One combined amount for all the node's qualifying incidents in a study
    year in which at least one occurs, drawn once for each such simulated
    year: variation between such years, not a component allocation, a share
    of harm or an amount per incident. A stochastic answer supplies
    conditional-positive quantiles only. The node's governed shape sets the
    tail above p95; p999 is a diagnostic.
    """

    model_config = ConfigDict(json_schema_extra=_LAB_ANNUAL_BUSINESS_LOSS_STATUS)

    status: Literal["available", "unavailable"]
    unavailable_reason: RequiredString | None = Field(
        default=None, description=UNAVAILABLE_REASON_DESCRIPTION
    )
    positive_probability: Probability | None = None
    amount: Nonnegative | None = None
    loss: SeverityQuantiles | None = None

    @model_validator(mode="after")
    def explicit_aggregate_law(self) -> Self:
        _LAB_ANNUAL_BUSINESS_LOSS_STATUS.validate(self)
        if self.status == "unavailable":
            return self
        if self.positive_probability == 0:
            for name in ("amount", "loss"):
                if getattr(self, name) is not None:
                    raise EvidenceValidationError(
                        f"{name}_not_allowed_when_positive_probability_is_zero",
                        "exact-zero lab business loss must omit its unused "
                        "positive law",
                        location=(name,),
                    )
        elif (self.amount is None) == (self.loss is None):
            raise EvidenceValidationError(
                "exactly_one_of_amount_loss_required_when_positive_probability_is_positive",
                "a positive lab business-loss probability requires exactly one of "
                "a fixed positive amount and conditional-positive quantiles",
            )
        elif self.amount is not None and self.amount <= 0:
            raise EvidenceValidationError(
                "fixed_lab_business_loss_amount_must_be_positive",
                "fixed positive lab business loss requires a positive amount",
                location=("amount",),
            )
        elif self.loss is not None and self.loss.p5 <= 0:
            raise EvidenceValidationError(
                "lab_business_loss_quantiles_must_be_positive",
                "stochastic lab business loss requires conditional-positive quantiles",
                location=("loss", "p5"),
            )
        if self.amount is not None:
            _validate_answer_amount("lab business loss amount", self.amount)
        return self


def validate_large_loss_counts(
    rate: Quantiles,
    large: Quantiles | None,
    very_large: Quantiles | None,
    total: SharedTotal | None,
) -> None:
    """Large-loss counts are asked exactly where the total has levels, and nest.

    Incidents at or above the higher level are a subset of those at or above
    the lower level, which are a subset of every admitted incident, so in
    every epistemic state the expected counts are ordered and hence so is
    each matching quantile (``SEV-LARGE-LOSS-ANCHOR``). A zero count
    therefore forces zero large-loss counts.
    """
    levels = (
        (None, None)
        if total is None
        else (total.large_loss_level, total.very_large_loss_level)
    )
    for name, level, count in (
        ("large_loss_count", levels[0], large),
        ("very_large_loss_count", levels[1], very_large),
    ):
        if (level is not None) != (count is not None):
            raise EvidenceValidationError(
                "large_loss_counts_required_exactly_with_levels",
                f"{name} is required exactly when the node's shared total states "
                "its level",
                location=(name,),
            )
    pairs = [] if large is None else [("large_loss_count", large, rate)]
    if very_large is not None and large is not None:
        pairs.append(("very_large_loss_count", very_large, large))
    for name, lower, upper in pairs:
        for key in QUANTILES:
            if getattr(lower, key) > getattr(upper, key):
                raise EvidenceValidationError(
                    f"{name}_exceeds_its_superset",
                    f"each {name} quantile must be at most the matching quantile "
                    "of the larger population it belongs to",
                    location=(name, key),
                )


def _asked_levels(total: SharedTotal | None) -> tuple[str, ...]:
    """The large-loss count fields a node with ``total`` asks."""
    if total is None or total.large_loss_level is None:
        return ()
    if total.very_large_loss_level is None:
        return ("large_loss_count",)
    return ("large_loss_count", "very_large_loss_count")


def without_unasked_levels(
    schema: dict[str, Any], total: SharedTotal | None, *extra: str
) -> dict[str, Any]:
    """``schema`` without the large-loss fields whose level the node does not
    ask: a question never offers a field its answer must leave empty.
    ``extra`` names fields that go with ``large_loss_count``."""
    large = None if total is None else total.large_loss_level
    very = None if total is None else total.very_large_loss_level
    unasked = (("large_loss_count", *extra) if large is None else ()) + (
        ("very_large_loss_count",) if very is None else ()
    )
    for name in unasked:
        for part in (schema, *schema.get("anyOf", ())):
            part["properties"].pop(name, None)
            if name in part.get("required", ()):
                part["required"].remove(name)
    return schema


_RISK_STATUS = StatusFields(
    "available",
    "unavailable",
    "unavailable_reason",
    required=("rate", "branches", "direct_eal"),
    excluded=(
        "rate",
        "large_loss_count",
        "very_large_loss_count",
        "total_loss",
        "lab_compensation",
        "lab_annual_business_loss",
        "branches",
        "direct_eal",
    ),
)


class ContextualRiskAnswer(_Record):
    """A leaf's direct answer, declared in the order the question asks it:
    reasoning first, then the count, the loss of one qualified unit, its
    large-loss counts, its components, the lab's own loss and the direct
    EAL."""

    model_config = ConfigDict(json_schema_extra=_RISK_STATUS)

    reasoning: Reasoning
    status: Literal["available", "unavailable"]
    unavailable_reason: RequiredString | None = Field(
        default=None, description=UNAVAILABLE_REASON_DESCRIPTION
    )
    rate: Quantiles | None = None
    total_loss: SeverityQuantiles | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    large_loss_count: Quantiles | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """Epistemic quantiles of the expected annual number of admitted incidents
    at or above the total's ``large_loss_level``: the large-loss anchor."""
    very_large_loss_count: Quantiles | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The same at ``very_large_loss_level``. No reference law reads it; it
    is the second anchor of the two-anchor comparison."""
    branches: tuple[ContextualBranchAnswer, ...] = ()
    lab_compensation: LabCompensationAnswer | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    lab_annual_business_loss: LabAnnualBusinessLossAnswer | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    direct_eal: Quantiles | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """Epistemic quantiles of the node's expected annual societal harm, judged
    directly rather than built from the count and marks: the leaf EAL
    cross-check's comparison (``FREQ-EAL-CROSSCHECK``). It is a check and is
    never simulated."""

    @classmethod
    def json_schema_for(cls, node: ContextualRiskNode) -> dict[str, Any]:
        total = node.total
        return _RISK_STATUS.require(
            without_unasked_levels(json_schema(cls), total),
            supplied=(
                "lab_compensation",
                "lab_annual_business_loss",
                *(("total_loss",) if total is not None else ()),
                *_asked_levels(total),
            ),
        )

    @model_validator(mode="after")
    def explicit_availability(self) -> Self:
        _RISK_STATUS.validate(self)
        if self.status != "available":
            return self
        repeat = _first_repeat(row.branch_id for row in self.branches)
        if repeat is not None:
            raise EvidenceValidationError(
                "branch_ids_must_be_unique",
                "joint branches must be unique",
                location=("branches", repeat, "branch_id"),
            )
        if not math.isclose(
            math.fsum(row.probability for row in self.branches),
            1,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise EvidenceValidationError(
                "branch_probabilities_must_sum_to_one",
                "joint branch probabilities must sum to one",
                location=("branches",),
            )
        return self

    def validate_against(self, node: ContextualRiskNode) -> None:
        # Marks are required at any count: severity is conditional on a
        # qualifying event, and the EAL cross-check may revise the count.
        if self.status == "unavailable":
            return
        business = self.lab_annual_business_loss
        if self.lab_compensation is None or business is None:
            name = (
                "lab_compensation"
                if self.lab_compensation is None
                else "lab_annual_business_loss"
            )
            raise EvidenceValidationError(
                f"{name}_required_when_available",
                "available answers require lab compensation and lab business-loss "
                "answers, including explicit unavailability",
                location=(name,),
            )
        if business.loss is not None and node.lab_loss_gpd_shape is None:
            raise EvidenceValidationError(
                "lab_business_loss_quantiles_require_governed_tail_shape",
                "this node has no governed lab business-loss tail shape; supply "
                "exact zero, a fixed positive amount or an unavailable reason",
                location=("lab_annual_business_loss", "loss"),
            )
        elicited = node.total is not None
        if elicited != (self.total_loss is not None):
            raise EvidenceValidationError(
                "total_loss_required_for_this_node"
                if elicited
                else "total_loss_not_asked_for_this_node",
                "only elicited shared totals require total-loss quantiles",
                location=("total_loss",),
            )
        assert self.rate is not None
        validate_large_loss_counts(
            self.rate, self.large_loss_count, self.very_large_loss_count, node.total
        )
        if (
            self.total_loss is not None
            and node.total is not None
            and (
                self.total_loss.p5 < node.total.floor
                or (node.total.strict and self.total_loss.p5 == node.total.floor)
            )
        ):
            raise EvidenceValidationError(
                "total_loss_must_preserve_qualifying_floor",
                "shared total quantiles must preserve the qualifying floor",
                location=("total_loss", "p5"),
            )
        authored = {row.branch_id for row in node.branches}
        for index, branch in enumerate(self.branches):
            if branch.branch_id not in authored:
                raise EvidenceValidationError(
                    "branch_id_not_in_authored_roster",
                    "answer must cover the exact authored joint branch roster",
                    location=("branches", index, "branch_id"),
                )
        if len(self.branches) != len(authored):
            raise EvidenceValidationError(
                "branches_must_cover_every_authored_branch",
                "answer must cover the exact authored joint branch roster",
                location=("branches",),
            )
        components = {row.component_id: row for row in node.components}
        for index, branch in enumerate(self.branches):
            location = ("branches", index, "components")
            for position, answer in enumerate(branch.components):
                if answer.component_id not in components:
                    raise EvidenceValidationError(
                        "component_id_not_in_authored_roster",
                        "every branch must cover the complete authored component "
                        "roster",
                        location=(*location, position, "component_id"),
                    )
            if len(branch.components) != len(components):
                raise EvidenceValidationError(
                    "components_must_cover_every_authored_component",
                    "every branch must cover the complete authored component roster",
                    location=location,
                )
            for position, answer in enumerate(branch.components):
                definition = components[answer.component_id]
                partition = definition.role == "partition"
                if answer.status == "available" and partition != (
                    answer.share is not None
                ):
                    raise EvidenceValidationError(
                        "share_required_for_partition_component"
                        if partition
                        else "share_not_allowed_for_independent_component",
                        "partition components require shares; independent "
                        "components require amount laws",
                        location=(*location, position, "share"),
                    )
                if answer.loss is not None and (
                    definition.gpd_shape is None or definition.rank_group_id is None
                ):
                    raise EvidenceValidationError(
                        "component_loss_requires_authored_tail_shape",
                        "elicited severity quantiles require an authored tail shape",
                        location=(*location, position, "loss"),
                    )
            shares = [
                row.share
                for row in branch.components
                if components[row.component_id].role == "partition"
            ]
            if (
                shares
                and all(share is not None for share in shares)
                and math.fsum(share for share in shares if share is not None) != 1
            ):
                raise EvidenceValidationError(
                    "partition_shares_must_sum_to_one",
                    "shared-total partition shares must sum to one without "
                    "renormalization",
                    location=location,
                )


class ProbabilityQuantiles(Quantiles):
    """Epistemic quantiles of one conditional probability."""

    @model_validator(mode="after")
    def probability(self) -> Self:
        if self.p95 > 1:
            raise EvidenceValidationError(
                "probability_quantiles_cannot_exceed_one",
                "probability quantiles cannot exceed one",
                location=("p95",),
            )
        return self


class ContextualStageAnswer(_Record):
    stage_id: RequiredIdentifier
    probability: ProbabilityQuantiles


_CHAIN_STATUS = StatusFields(
    "available",
    "unavailable",
    "unavailable_reason",
    required=("opportunity_rate", "stages"),
    excluded=("opportunity_rate", "stages"),
)


class ContextualChainAnswer(_Record):
    """The opportunity rate and every stage probability of one chain.

    ``opportunity_rate`` is the expected number of opportunities in the study
    year's risk set per base unit of the chain's stipulated exposure; code
    multiplies it by the stipulated number of base units
    (``ChainExposure.count``), so the respondent never chooses the
    opportunity count's scale. Quantiles express uncertainty about that
    expected rate and about each conditional probability, not realized
    variation.
    """

    model_config = ConfigDict(json_schema_extra=_CHAIN_STATUS)

    reasoning: Reasoning
    status: Literal["available", "unavailable"]
    unavailable_reason: RequiredString | None = Field(
        default=None, description=UNAVAILABLE_REASON_DESCRIPTION
    )
    opportunity_rate: Quantiles | None = None
    stages: tuple[ContextualStageAnswer, ...] = ()

    @model_validator(mode="after")
    def explicit_availability(self) -> Self:
        _CHAIN_STATUS.validate(self)
        return self

    def validate_against(self, chain: ContextualFrequencyChain) -> None:
        if self.status != "available":
            return
        authored = [stage.stage_id for stage in chain.stages]
        for index, row in enumerate(self.stages):
            if index >= len(authored) or row.stage_id != authored[index]:
                raise EvidenceValidationError(
                    "stage_must_follow_authored_order",
                    "an available chain answer must cover every authored stage in "
                    "order",
                    location=("stages", index, "stage_id"),
                )
        if len(self.stages) != len(authored):
            raise EvidenceValidationError(
                "stages_must_cover_every_authored_stage",
                "an available chain answer must cover every authored stage in order",
                location=("stages",),
            )


class RevisedJudgment(_Record):
    judgment_id: RequiredIdentifier
    explanation: RequiredString


class DirectSeverityRetry(_Record):
    """The one retry of a direct answer whose large-loss count contradicts its
    loss quantiles (``SEV-LARGE-LOSS-ANCHOR``).

    ``contradiction`` is the neutral statement the retry question showed,
    rendered from the case's stored share. ``answer`` is the retry's complete
    answer, or None when the retry exhausted its validation attempts
    (``ExhaustedCall``). ``large_loss_share`` is the retry answer's stored share.
    """

    prompt_sha256: Sha256Digest
    contradiction: RequiredString
    answer: ContextualRiskAnswer | None
    large_loss_share: Nonnegative | None = None
    resolved: bool
    """Whether the retry answer is available and coherent: the run then uses
    it; otherwise the original answer stands as ``severity_unresolved``."""

    @model_validator(mode="after")
    def resolution_needs_an_answer(self) -> Self:
        if self.resolved and self.answer is None:
            raise ValueError("a resolved retry needs its answer")
        return self


class ChainGuardrailRetry(_Record):
    """The single retry of an available chain answer that trips a guardrail."""

    prompt_sha256: Sha256Digest
    findings: tuple[RequiredString, ...] = Field(min_length=1)
    answer: ContextualChainAnswer | None
    resolved: bool

    @model_validator(mode="after")
    def resolution_needs_an_available_answer(self) -> Self:
        if self.resolved and (self.answer is None or self.answer.status != "available"):
            raise ValueError(
                "a resolved chain guardrail retry needs an available answer"
            )
        return self


EalCrosscheckJudgment = Literal["count", "severity", "large_loss_count", "direct_eal"]
"""A judgment a leaf EAL cross-check may revise (``FREQ-EAL-CROSSCHECK``)."""

EAL_CROSSCHECK_FIELDS: Mapping[EalCrosscheckJudgment, tuple[str, ...]] = {
    "count": ("rate",),
    "severity": ("total_loss", "branches"),
    "large_loss_count": ("large_loss_count", "very_large_loss_count"),
    "direct_eal": ("direct_eal",),
}
"""The answer fields each revisable judgment owns. A revised severity supplies
``total_loss``, the complete ``branches`` or both; a revised large-loss count
supplies ``large_loss_count`` and, where the node asks it,
``very_large_loss_count``."""


def eal_crosscheck_judgments(node: ContextualRiskNode) -> tuple[str, ...]:
    """The judgments a node's EAL cross-check may revise: its large-loss count
    only where its shared total states large-loss levels."""
    levels = node.total is not None and node.total.large_loss_level is not None
    return tuple(
        judgment
        for judgment in EAL_CROSSCHECK_FIELDS
        if judgment != "large_loss_count" or levels
    )


_CROSSCHECK_STATUS = StatusFields(
    "revised",
    "unchanged",
    "unchanged_reason",
    required=("revised",),
    excluded=(
        "revised",
        *(name for fields in EAL_CROSSCHECK_FIELDS.values() for name in fields),
    ),
)


class EalCrosscheckAnswer(_Record):
    """A leaf's answer after seeing its direct EAL beside the EAL its count
    and marks imply.

    Two replies are equally acceptable. ``unchanged``: the respondent keeps
    every answer, with its ``unchanged_reason``; the case is recorded as
    declined. ``revised``: ``revised`` names each judgment the respondent
    changes, with why, and the answer supplies exactly those judgments' new
    values (``EAL_CROSSCHECK_FIELDS``); every other judgment stands as stated.
    """

    model_config = ConfigDict(json_schema_extra=_CROSSCHECK_STATUS)

    reasoning: Reasoning
    status: Literal["revised", "unchanged"]
    unchanged_reason: RequiredString | None = Field(
        default=None,
        description="Why every answer stands. Given only when status is "
        "unchanged; null or omitted when status is revised.",
    )
    revised: tuple[RevisedJudgment, ...] = ()
    rate: Quantiles | None = Field(default=None, exclude_if=lambda value: value is None)
    total_loss: SeverityQuantiles | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    large_loss_count: Quantiles | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    very_large_loss_count: Quantiles | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    branches: tuple[ContextualBranchAnswer, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    direct_eal: Quantiles | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @classmethod
    def json_schema_for(cls, node: ContextualRiskNode) -> dict[str, Any]:
        schema = json_schema(cls)
        schema["$defs"]["RevisedJudgment"]["properties"]["judgment_id"] = {
            "enum": list(eal_crosscheck_judgments(node))
        }
        return without_unasked_levels(schema, node.total)

    @property
    def revises(self) -> bool:
        return self.status == "revised"

    def supplied(self, judgment: EalCrosscheckJudgment) -> bool:
        """Whether the answer supplies any field of ``judgment``."""
        return any(
            getattr(self, field) not in (None, ())
            for field in EAL_CROSSCHECK_FIELDS[judgment]
        )

    @model_validator(mode="after")
    def revised_judgments_supplied(self) -> Self:
        _CROSSCHECK_STATUS.validate(self)
        named = [row.judgment_id for row in self.revised]
        repeat = _first_repeat(named)
        if repeat is not None:
            raise EvidenceValidationError(
                "revised_judgments_must_be_unique",
                "each revised judgment is named once",
                location=("revised", repeat, "judgment_id"),
            )
        if self.status == "unchanged":
            return self
        for index, judgment in enumerate(named):
            if judgment not in EAL_CROSSCHECK_FIELDS:
                raise EvidenceValidationError(
                    "unknown_revised_judgment",
                    "revised judgments are count, severity, large_loss_count or "
                    "direct_eal",
                    location=("revised", index, "judgment_id"),
                )
        for judgment, fields in EAL_CROSSCHECK_FIELDS.items():
            given = [field for field in fields if _supplied(getattr(self, field))]
            if judgment in named:
                # A revised large-loss count always supplies large_loss_count.
                needed = fields[:1] if judgment == "large_loss_count" else fields
                if not set(needed) & set(given):
                    raise EvidenceValidationError(
                        f"{'_or_'.join(needed)}_required_when_{judgment}_revised",
                        "a revised answer supplies exactly the judgments it names",
                        location=needed if len(needed) == 1 else (),
                    )
            elif given:
                raise EvidenceValidationError(
                    f"{given[0]}_not_allowed_when_{judgment}_not_revised",
                    "a revised answer supplies exactly the judgments it names",
                    location=(given[0],),
                )
        return self


class ImpliedEal(_Record):
    """The expected annual societal harm a leaf's count and marks imply at one
    coordinate, and what drives it, each to four significant figures
    (``FREQ-EAL-CROSSCHECK``).

    ``eal_mean`` is ``count_mean`` times ``harm_per_incident_mean``, the
    compiled law's analytic welfare mean per expected qualified unit, as the
    taxonomy parents' pools read it; None without an admitted law or a finite
    mean. ``gross_per_incident_mean`` is the same law's gross mean per unit,
    so their ratio is the share of gross loss that is societal harm.
    ``total_mean`` is the shared total's gross mean per unit and
    ``severity_median`` the stated shared-total median, on the same gross
    basis; each is None without that total or statement.
    ``count_share_beyond_p95`` is the part of the count mean that comes from
    ranks beyond the stated p95, where code extrapolates;
    ``gross_share_at_or_above_large_loss`` is the part of the shared total's
    gross mean from units at or above the large-loss level, None without one.
    """

    eal_mean: Nonnegative | None
    count_mean: Nonnegative | None
    count_median: Nonnegative
    harm_per_incident_mean: Nonnegative | None
    gross_per_incident_mean: Nonnegative | None
    total_mean: Nonnegative | None
    severity_median: Nonnegative | None
    count_share_beyond_p95: Fraction | None
    gross_share_at_or_above_large_loss: Fraction | None


class EalCrosscheckSeverityRetry(_Record):
    """The one retry of a cross-check answer whose large-loss count
    contradicts its loss quantiles (``SEV-LARGE-LOSS-ANCHOR``)."""

    prompt_sha256: Sha256Digest
    contradiction: RequiredString
    answer: EalCrosscheckAnswer | None
    large_loss_share: Nonnegative | None = None
    resolved: bool

    @model_validator(mode="after")
    def resolution_needs_an_answer(self) -> Self:
        if self.resolved and self.answer is None:
            raise ValueError("a resolved retry needs its answer")
        return self


class EalReconciliation(_Record):
    """The cross-check call of a leaf whose stored values fire a cross-check
    trigger (``elicitation.eal_crosscheck.crosscheck_triggers``).

    ``large_loss_share`` is the revised answer's stored share, as for a
    direct answer. ``direct_eal_mean_after`` and ``implied_after`` are the
    adopted revised answer's direct EAL mean and implied EAL, stored once so
    the case's status does not depend on platform floating point; None when
    the respondent kept every answer. ``answer`` is None when the call
    exhausted its validation attempts (``ExhaustedCall``); every answer then
    stands as stated.
    """

    prompt_sha256: Sha256Digest
    presentation_order: tuple[
        Literal["direct", "implied"], Literal["direct", "implied"]
    ]
    answer: EalCrosscheckAnswer | None
    large_loss_share: Nonnegative | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    severity_retry: EalCrosscheckSeverityRetry | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    direct_eal_mean_after: Nonnegative | None = None
    implied_after: ImpliedEal | None = None

    @model_validator(mode="after")
    def after_values(self) -> Self:
        if set(self.presentation_order) != {"direct", "implied"}:
            raise ValueError("the cross-check presents both EALs once")
        revises = self.answer is not None and self.answer.revises
        if revises != (self.implied_after is not None):
            raise ValueError(
                "a revised cross-check answer, and only one, records its implied EAL"
            )
        if not revises and (
            self.large_loss_share is not None
            or self.severity_retry is not None
            or self.direct_eal_mean_after is not None
        ):
            raise ValueError(
                "an unchanged cross-check answer records no revised values"
            )
        return self


class EalCrosscheck(_Record):
    """A leaf case's direct EAL beside the EAL its final count and marks imply.

    Collection stores both once after the direct answer's coherence retry and
    before the count is pooled with the chain. The EAL ratio determines
    whether a cross-check question is asked, and replay checks the stored
    values within ``CHAIN_COUNT_TOLERANCE``.
    """

    direct_eal_mean: Nonnegative | None
    """The direct EAL law's epistemic mean at four significant figures; None
    without a finite mean."""
    implied: ImpliedEal
    reconciliation: EalReconciliation | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


_PARENT_STATUS = StatusFields(
    "available",
    "unavailable",
    "unavailable_reason",
    required=("category_eal",),
    excluded=("category_eal", "remainder_eal"),
)


class ParentAnswer(_Record):
    """A parent's whole-category EAL and the part its modeled children miss.

    Both are epistemic quantiles of an expected annual societal harm in the
    population's monetary units. A parent with no modeled child supplies no
    remainder: its whole category is unmodeled.
    """

    model_config = ConfigDict(json_schema_extra=_PARENT_STATUS)

    reasoning: Reasoning
    status: Literal["available", "unavailable"]
    unavailable_reason: RequiredString | None = Field(
        default=None, description=UNAVAILABLE_REASON_DESCRIPTION
    )
    category_eal: Quantiles | None = None
    remainder_eal: Quantiles | None = None

    @classmethod
    def json_schema_for(cls, parent: ContextualParentNode) -> dict[str, Any]:
        """A parent with modeled children supplies its remainder; one
        without them leaves it null."""
        remainder = ("remainder_eal",)
        if parent.children:
            return _PARENT_STATUS.require(json_schema(cls), supplied=remainder)
        return _PARENT_STATUS.require(json_schema(cls), empty=remainder)

    @model_validator(mode="after")
    def explicit_availability(self) -> Self:
        _PARENT_STATUS.validate(self)
        return self

    def validate_against(self, parent: ContextualParentNode) -> None:
        if self.status == "available" and (self.remainder_eal is None) == bool(
            parent.children
        ):
            raise EvidenceValidationError(
                "remainder_eal_required_with_modeled_children"
                if parent.children
                else "remainder_eal_not_allowed_without_modeled_children",
                "a parent with modeled children supplies its remainder EAL, "
                "and one without them supplies none",
                location=("remainder_eal",),
            )


ParentPoolStatus = Literal[
    "pooled", "not_pooled", "parent_exhausted", "comparison_unavailable"
]
"""How a parent's pool ended (``HIER-CATEGORY-POOL``).

``pooled``: its category answer and what is beneath it are pooled.
``not_pooled``: one of the two is zero, so the other stands.
``parent_exhausted``: the parent question exhausted its validation attempts.
``comparison_unavailable``: the category answer or a mean beneath it is not
identified. Only a pooled parent carries a factor other than one.
"""


class ParentPool(_Record):
    """One parent's pool of its category answer with what is beneath it
    (``HIER-CATEGORY-POOL``), as a parent case stores it.

    ``direct_eal_mean`` is the mean of the parent's category law and
    ``modeled_eal_mean`` the sum of its children's mean EALs going in plus the
    mean of its remainder law, each at four significant figures and None where
    it is not identified. ``weight`` is the category answer's weight at the
    parent's ``height``, and ``factor`` is the direct mean over the modeled
    mean, raised to the weight, at four significant figures: the frequency
    factor on every leaf beneath the parent and the scale on every remainder
    at or beneath it. ``reason`` says why a parent is not pooled.
    """

    status: ParentPoolStatus
    reason: RequiredString | None = None
    height: int = Field(ge=1, strict=True)
    weight: Fraction
    direct_eal_mean: Nonnegative | None
    modeled_eal_mean: Nonnegative | None
    factor: Positive

    @model_validator(mode="after")
    def coherent_status(self) -> Self:
        if (self.status == "pooled") == (self.reason is not None):
            raise ValueError(
                "a parent that is not pooled, and only one, states its reason"
            )
        if self.status == "pooled" and not (
            self.direct_eal_mean and self.modeled_eal_mean
        ):
            raise ValueError(
                "a pooled parent records its positive direct and modeled mean EALs"
            )
        if self.status != "pooled" and self.factor != 1:
            raise ValueError("a parent that is not pooled has factor one")
        return self


class ContextualParentCase(_Record):
    """One parent, coordinate and respondent: its direct answer, the child
    EALs going into its pool and the pool.

    ``answer`` is None when the parent question exhausted its validation
    attempts (``ExhaustedCall``): the parent is then not pooled, and what is
    beneath it stands. A parent of an incomplete group, one with an
    unanswered leaf (``UnansweredCase``), is not pooled at collection: its
    child EALs and its pool are None. Import pools it only where the group's
    states read another model's answers to that leaf
    (``POOL-UNANSWERED-SCENARIOS``).

    The child EALs and the pool are an audit record. Every reader replays the
    pool from the answers, and replay checks the stored values against it
    within ``HIERARCHY_EAL_TOLERANCE``.
    """

    parent_id: RequiredIdentifier
    anchor_score: Finite
    respondent: Respondent
    answer: ParentAnswer | None
    prompt_sha256: Sha256Digest
    children_median_eal: dict[RequiredIdentifier, Nonnegative | None] | None
    """Each leaf child's median EAL going into the pool, rounded to four
    significant figures; None where unidentified and for a parent child, whose
    EAL is a sum of means."""
    children_mean_eal: dict[RequiredIdentifier, Nonnegative | None] | None
    """Each child's mean EAL going into the pool, rounded to four significant
    figures: a leaf's own, and a parent child's pooled value. None where no
    finite mean is identified."""
    pool: ParentPool | None
    """The parent's pool with what is beneath it (``HIER-CATEGORY-POOL``)."""

    @model_validator(mode="after")
    def pooled_when_complete(self) -> Self:
        if not (
            (self.children_median_eal is None)
            == (self.children_mean_eal is None)
            == (self.pool is None)
        ):
            raise ValueError(
                "a parent case stores its child medians, child means and pool together"
            )
        if self.pool is not None and (
            (self.pool.status == "parent_exhausted") != (self.answer is None)
        ):
            raise ValueError(
                "a parent case's pool is parent_exhausted exactly when its "
                "question has no answer"
            )
        return self

    def key(self) -> tuple[str, float, tuple[str, str, int]]:
        return self.parent_id, self.anchor_score, self.respondent.key()


class ContextualRiskCase(_Record):
    """One node, coordinate and respondent: direct, chain and cross-check answers."""

    node_id: RequiredIdentifier
    anchor_score: Finite
    respondent: Respondent
    answer: ContextualRiskAnswer
    prompt_sha256: Sha256Digest
    chain_answer: ContextualChainAnswer | None
    """The original chain answer; None when its question exhausted."""
    chain_prompt_sha256: Sha256Digest
    chain_guardrail: ChainGuardrailRetry | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The one retry asked when the original available chain answer trips a
    guardrail. An available retry answer supplies the stored chain count even
    when its findings remain; otherwise the original answer supplies it."""
    chain_count: Quantiles | None
    """The five quantiles of the driving chain answer at the stipulated base
    count, recomputed within ``CHAIN_COUNT_TOLERANCE`` during replay."""
    large_loss_share: Nonnegative | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The direct answer's large-loss share, its large-loss count mean divided
    by its count mean, unrounded; None where no share
    exists (no level, an unavailable answer, or a zero count or anchor).

    The coherence check and its retry question read this stored value, and
    replay checks it within ``CHAIN_COUNT_TOLERANCE``."""
    severity_retry: DirectSeverityRetry | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    eal_crosscheck: EalCrosscheck | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The leaf EAL cross-check (``FREQ-EAL-CROSSCHECK``); every case with an
    available direct answer records one, and only those."""

    @model_validator(mode="after")
    def paired_chain(self) -> Self:
        retry_answer = (
            None if self.chain_guardrail is None else self.chain_guardrail.answer
        )
        chain_available = (
            retry_answer is not None and retry_answer.status == "available"
        ) or (
            (retry_answer is None or retry_answer.status != "available")
            and self.chain_answer is not None
            and self.chain_answer.status == "available"
        )
        if (self.chain_count is None) == chain_available:
            raise ValueError(
                "an available driving chain answer, and only one, records its count law"
            )
        if self.chain_guardrail is not None and (
            self.chain_answer is None or self.chain_answer.status != "available"
        ):
            raise ValueError(
                "a chain guardrail retry needs an available original answer"
            )
        return self

    def key(self) -> tuple[str, float, tuple[str, str, int]]:
        return self.node_id, self.anchor_score, self.respondent.key()


class UnansweredCase(_Record):
    """A leaf, coordinate and respondent whose direct question exhausted its
    validation attempts (``ExhaustedCall``).

    Without a direct answer the leaf has no law, so its (coordinate,
    respondent) group is incomplete: its other leaves complete as usual and
    its parents are answered but not pooled at collection. Import excludes
    every joint state the group belongs to (``POOL-EXCLUDED-SAMPLES``),
    unless the respondent's model does not answer the leaf over the study,
    whose states then read another model's answers to it
    (``POOL-UNANSWERED-SCENARIOS``). The original chain question, asked
    independently, is recorded; None when it exhausted its attempts too.
    """

    node_id: RequiredIdentifier
    anchor_score: Finite
    respondent: Respondent
    prompt_sha256: Sha256Digest
    chain_answer: ContextualChainAnswer | None
    chain_prompt_sha256: Sha256Digest

    def key(self) -> tuple[str, float, tuple[str, str, int]]:
        return self.node_id, self.anchor_score, self.respondent.key()


ExhaustedCallKind = Literal[
    "direct",
    "chain",
    "direct_severity_retry",
    "chain_guardrail_retry",
    "eal_crosscheck",
    "eal_crosscheck_severity_retry",
    "parent",
]
EXHAUSTED_CALL_KINDS: tuple[ExhaustedCallKind, ...] = get_args(ExhaustedCallKind)


@dataclass(frozen=True)
class ExhaustedCall:
    """One exhausted call: its kind, the leaf or parent it asks about, its
    coordinate, its respondent's key and its question's digest, which with
    the respondent identifies the call in its capture.

    A call is exhausted when every validation attempt returned a response
    that failed validation, a provider's refusal or output-limit stop
    included. Collection records it with answer None and goes on
    (``POOL-EXCLUDED-SAMPLES``). A provider failure, an error rather than a
    response, is never exhausted: a transient one fails only its call's
    (coordinate, respondent) group and a systematic one stops the study, and
    a rerun sends the failed call again (``NUM-PROVIDER-FAILURES``).
    """

    kind: ExhaustedCallKind
    subject_id: str
    anchor_score: float
    respondent: tuple[str, str, int]
    prompt_sha256: str


type GroupKey = tuple[float, tuple[str, str, int]]
"""An (anchor score, respondent key) group: one respondent's hierarchy at one
coordinate."""


class ContextualRiskArtifact(_Record):
    schema_version: Literal[1] = 1

    @model_validator(mode="before")
    @classmethod
    def current_schema(cls, data: object) -> object:
        return _current_schema(data, "contextual capture")

    @classmethod
    def from_saved(cls, payload: object) -> Self:
        """The capture a saved file's decoded JSON holds, which states the
        current schema version."""
        _require_saved_schema(payload, "contextual capture")
        return cls.model_validate(payload)

    instrument_sha256: Sha256Digest
    cases: tuple[ContextualRiskCase, ...] = ()
    unanswered_cases: tuple[UnansweredCase, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    parent_cases: tuple[ContextualParentCase, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    evidence_status: Literal["fixture", "preliminary", "reviewed"] = "preliminary"
    reviewed_by: RequiredString | None = None
    review_rationale: RequiredString | None = None

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        keys = [case.key() for case in (*self.cases, *self.unanswered_cases)]
        if len(set(keys)) != len(keys):
            raise ValueError("contextual cases must be unique")
        if (
            self.evidence_status == "reviewed"
            and (self.reviewed_by is None or self.review_rationale is None)
        ) or (
            self.evidence_status != "reviewed"
            and (self.reviewed_by is not None or self.review_rationale is not None)
        ):
            raise ValueError(
                "reviewed artifacts require reviewer and rationale together"
            )
        return self

    def incomplete_groups(self) -> frozenset[GroupKey]:
        """The groups with an unanswered leaf."""
        return frozenset(
            (row.anchor_score, row.respondent.key()) for row in self.unanswered_cases
        )

    def exhausted_calls(self) -> tuple[ExhaustedCall, ...]:
        """Every exhausted call the artifact records, in case order."""
        rows: list[ExhaustedCall] = []

        def add(kind: ExhaustedCallKind, row, subject: str, digest: str) -> None:
            rows.append(
                ExhaustedCall(
                    kind, subject, row.anchor_score, row.respondent.key(), digest
                )
            )

        for row in self.unanswered_cases:
            add("direct", row, row.node_id, row.prompt_sha256)
            if row.chain_answer is None:
                add("chain", row, row.node_id, row.chain_prompt_sha256)
        for case in self.cases:
            node = case.node_id
            if case.chain_answer is None:
                add("chain", case, node, case.chain_prompt_sha256)
            retry = case.severity_retry
            if retry is not None and retry.answer is None:
                add("direct_severity_retry", case, node, retry.prompt_sha256)
            guardrail = case.chain_guardrail
            if guardrail is not None and guardrail.answer is None:
                add("chain_guardrail_retry", case, node, guardrail.prompt_sha256)
            check = case.eal_crosscheck
            crosscheck = None if check is None else check.reconciliation
            if crosscheck is not None:
                if crosscheck.answer is None:
                    add("eal_crosscheck", case, node, crosscheck.prompt_sha256)
                retry_check = crosscheck.severity_retry
                if retry_check is not None and retry_check.answer is None:
                    add(
                        "eal_crosscheck_severity_retry",
                        case,
                        node,
                        retry_check.prompt_sha256,
                    )
        for parent in self.parent_cases:
            if parent.answer is None:
                add("parent", parent, parent.parent_id, parent.prompt_sha256)
        return tuple(rows)

    def validate_against(self, instrument: ContextualRiskInstrument) -> None:
        if self.instrument_sha256 != instrument.digest():
            raise ValueError("contextual artifact instrument identity differs")
        expected = {
            (node.node_id, score, respondent.key())
            for node in instrument.collectible_nodes
            for score in instrument.anchor_scores
            for respondent in instrument.respondents
        }
        if {case.key() for case in (*self.cases, *self.unanswered_cases)} != expected:
            raise ValueError(
                "artifact must cover every collectible node, anchor and respondent"
            )
        parents = {row.parent_id: row for row in instrument.parents}
        keys = [case.key() for case in self.parent_cases]
        if len(keys) != len(set(keys)) or set(keys) != {
            (parent_id, score, respondent.key())
            for parent_id in parents
            for score in instrument.anchor_scores
            for respondent in instrument.respondents
        }:
            raise ValueError(
                "artifact must cover every taxonomy parent, anchor and respondent once"
            )
        incomplete = self.incomplete_groups()
        for case in self.parent_cases:
            parent = parents[case.parent_id]
            if case.answer is not None:
                case.answer.validate_against(parent)
            if (case.children_mean_eal is None) != (
                (case.anchor_score, case.respondent.key()) in incomplete
            ):
                raise ValueError(
                    "a parent case stores its child EALs exactly when its group has "
                    "every leaf answered"
                )
            if case.children_median_eal is not None and not (
                set(case.children_median_eal)
                == set(case.children_mean_eal or ())
                == set(parent.children)
            ):
                raise ValueError(
                    "a parent case stores one median and one mean EAL for each of its children"
                )
        nodes = {node.node_id: node for node in instrument.collectible_nodes}
        for row in self.unanswered_cases:
            chain = nodes[row.node_id].frequency_chain
            assert chain is not None
            if row.chain_answer is not None:
                row.chain_answer.validate_against(chain)
        for case in self.cases:
            node = nodes[case.node_id]
            case.answer.validate_against(node)
            if (
                case.severity_retry is not None
                and case.severity_retry.answer is not None
            ):
                case.severity_retry.answer.validate_against(node)
            chain = node.frequency_chain
            assert chain is not None
            if case.chain_answer is not None:
                case.chain_answer.validate_against(chain)
            guardrail = case.chain_guardrail
            if guardrail is not None and guardrail.answer is not None:
                guardrail.answer.validate_against(chain)
            check = case.eal_crosscheck
            if check is not None and check.reconciliation is not None:
                allowed = set(eal_crosscheck_judgments(node))
                retry_check = check.reconciliation.severity_retry
                for answer in (
                    check.reconciliation.answer,
                    None if retry_check is None else retry_check.answer,
                ):
                    if (
                        answer is not None
                        and not {row.judgment_id for row in answer.revised} <= allowed
                    ):
                        raise ValueError(
                            "a cross-check revises only the judgments its node asks"
                        )
