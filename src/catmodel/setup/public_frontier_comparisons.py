"""Comparison arms of a context-bound public-trajectory setup.

Reweighting arms keep every conditional law and change only joint-state
weights, so execution reuses the reference draws. Simulated arms recompile the
captured answers under one declared change and override only the laws whose
numbers change. Every catalog family is frozen as a comparison requirement:
executed when selected, otherwise unavailable with the reason. The register
challenges no context-bound setup executes are frozen unavailable beside them.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import (
    dataclass,
    field,
    replace,
)
from functools import partial
from types import MappingProxyType
from typing import Literal

from catmodel.contracts.contextual_risk import ContextualRiskNode
from catmodel.contracts.incident import (
    Alternative,
    ComparisonRequirement,
    EntityPath,
    IncidentLaw,
    JointState,
    LawOverride,
    PopulationAbsence,
)
from catmodel.contracts.law_overrides import (
    HeldOverrides,
    LawOverrides,
    OverrideWriter,
)
from catmodel.elicitation.epistemic import epistemic_knots, tail_widths
from catmodel.scenarios.sensitivity_coverage import (
    CAPPED_TAIL_SHAPE_SANITY_BOUND,
    TAIL_SHAPE_SENSITIVITY_STEP,
    UNCAPPED_TAIL_SHAPE_BOUND,
    shifted_tail_shape,
)
from catmodel.setup.contextual_compilation import ContextualRespondentSelection
from catmodel.setup.contextual_node import CompileOptions
from catmodel.setup.rank_spread import REFERENCE_SPREAD_TAIL_FORM
from catmodel.setup.sensitivity_catalog import (
    COMMON_FREQUENCY_BIAS_FACTORS,
    COUNT_FAMILY_VARIANCE_TO_MEAN,
    DEFAULT_CONTEXT_BOUND_FAMILIES,
    EPISTEMIC_SPREAD_FACTORS,
    EPISTEMIC_TAIL_FACTORS,
    EXPOSURE_SCALE_VARIANTS,
    FRONTIER_LAB_VALUE_SENSITIVITY_USD,
    FRONTIER_LAB_VALUE_USD,
    LARGE_LOSS_ANCHOR_FACTORS,
    PUBLIC_STUDY_UNAVAILABLE_CHALLENGES,
    SAFEGUARD_LEVELS,
    SENSITIVITY_FAMILIES,
    unavailable_challenge_requirements,
    usd_text,
)

type Status = Literal["unavailable", "inapplicable"]

NO_MAPPING: Mapping = MappingProxyType({})


@dataclass(frozen=True)
class PublicForecastAlternative:
    """One alternative forecast's exact trajectory-node weights, or why none exist."""

    path_weights: Mapping[str, float] | None
    provenance: Mapping[str, object]
    reason: str | None = None


@dataclass(frozen=True)
class AuthoredCapStresses:
    """Severity-governance stresses authored for one node's frozen cap posture.

    ``endpoint_from_tail_class`` marks a cap that is the tail class's endpoint
    for the whole event. Its stresses move the aggregate lab-loss cap only
    while that cap equals the same endpoint; a lab cap set on another basis is
    left to its own comparison family.
    """

    cap_values: tuple[float, float] | None = None
    uncapped_alternative: bool = False
    endpoint_stress: float | None = None
    endpoint_from_tail_class: bool = False


@dataclass(frozen=True)
class StateCell:
    """One joint state's compiled outcome by scenario: its laws or an absence."""

    state_id: str
    draw_id: str
    path_weight: float
    selection: ContextualRespondentSelection
    paths: tuple[EntityPath, ...]
    outcomes: Mapping[str, tuple[IncidentLaw, ...] | PopulationAbsence]


@dataclass(frozen=True)
class PublicFrontierComparisons:
    """The study's arms, the frozen requirement of every catalog family, and
    the laws each simulated arm puts in place of reference laws."""

    alternatives: tuple[Alternative, ...]
    requirements: tuple[ComparisonRequirement, ...]
    overrides: LawOverrides = field(default_factory=HeldOverrides)


type CompileCells = Callable[..., list[StateCell]]
"""Recompiles the reference cells under a comparison arm's options, with
keyword ``edits`` (node changes by scenario)."""

GENERATED_FAMILIES = (
    "public_forecast",
    "exposure_scale",
    "safeguard_level",
    "lab_loss_cap",
    "lab_loss_provider_only",
    "model_pooling_leave_one_out",
    "model_pooling",
    "tail_shape",
    "uncapped_severity",
    "severity_cap",
    "severity_endpoint_stress",
    "count_family",
    "epistemic_spread_calibration",
    "epistemic_tail_width",
    "common_frequency_bias",
    "large_loss_anchor",
    "large_loss_anchor_ignored",
    "very_large_loss_anchor",
)
HELD_FIXED = (
    "Original captured questions, answers and public trajectory nodes.",
    "All scientific parameters outside the named challenge.",
)


@dataclass(frozen=True)
class ContextVariantFamily:
    """A family of context variants: separately collected public studies of the
    same forecast on the headline's embedded trajectory rule, each varying one
    stipulation of the study context, measured against one matched control.

    ``variants`` holds each catalog variant's arm description and provenance.
    """

    family_id: str
    noun: str
    """The variant kind as text, with its article: ``an exposure-scale``."""
    study_option: str
    state_prefix: str
    variants: Mapping[str, tuple[str, Mapping[str, object]]]
    control_description: str
    held_fixed: tuple[str, ...]

    @property
    def control(self) -> str:
        return f"{self.family_id}::embedded_control"

    def state_id(self, variant_id: str, state_id: str) -> str:
        """A variant state's ID: the variant, then its own study's node and selection."""
        return f"{self.state_prefix}-{variant_id}-{state_id}"


_EMBEDDED = "at each node of the embedded trajectory rule, from a separately collected study; measured against the embedded-rule control."
CONTEXT_VARIANT_FAMILIES: Mapping[str, ContextVariantFamily] = MappingProxyType(
    {
        "exposure_scale": ContextVariantFamily(
            family_id="exposure_scale",
            noun="an exposure-scale",
            study_option="--variant-study",
            state_prefix="exposure",
            variants={
                variant_id: (
                    f"Every path-linked deployment scale multiplied by {factor:g} {_EMBEDDED}",
                    {"exposure_variant": variant_id, "scale_factor": factor},
                )
                for variant_id, factor in EXPOSURE_SCALE_VARIANTS.items()
            },
            control_description="Matched control of the exposure-scale variants: the headline captures reweighted to the embedded trajectory rule, at the stipulated scales.",
            held_fixed=(
                "Incident definitions, lab, period, source bundle, respondent models, controls and adoption.",
                "The embedded rule's trajectory nodes and weights.",
            ),
        ),
        "safeguard_level": ContextVariantFamily(
            family_id="safeguard_level",
            noun="a safeguard-level",
            study_option="--variant-study",
            state_prefix="safeguard",
            variants={
                level: (
                    f"The study's stipulated safeguards (shared controls, each source status's tools and permissions and human oversight, and internal use's scope) replaced by the study config's {level} level {_EMBEDDED}",
                    {"safeguard_level": level},
                )
                for level in SAFEGUARD_LEVELS
            },
            control_description="Matched control of the safeguard-level variants: the headline captures reweighted to the embedded trajectory rule, at the stipulated safeguards.",
            held_fixed=(
                "Incident definitions, lab, period, source bundle, respondent models, deployment scales and adoption.",
                "The embedded rule's trajectory nodes and weights.",
            ),
        ),
    }
)
EXPOSURE_CONTROL = CONTEXT_VARIANT_FAMILIES["exposure_scale"].control
SAFEGUARD_CONTROL = CONTEXT_VARIANT_FAMILIES["safeguard_level"].control


class _Excluded(Exception):
    def __init__(self, status: Status, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _shifted(shape: float, step: float, cap: float | None, part: str) -> float:
    """The shape one step away, or its exclusion where a law cannot represent it.

    The shifted shape is the decimal sum (``shifted_tail_shape``), so each
    bound is compared as a stated decimal: a shape exactly at the lower or
    the capped bound stands, and one exactly at the uncapped bound is
    excluded.
    """
    value = shifted_tail_shape(shape, step)
    upper = UNCAPPED_TAIL_SHAPE_BOUND if cap is None else CAPPED_TAIL_SHAPE_SANITY_BOUND
    if (
        value < -UNCAPPED_TAIL_SHAPE_BOUND
        or value > upper
        or (cap is None and value == upper)
    ):
        raise _Excluded(
            "unavailable",
            f"A {step:+g} step puts the {part} tail shape at {value:g}, outside its "
            "operating bound; a shift a law cannot represent is not clamped.",
        )
    return value


def _shift_tail(
    node: ContextualRiskNode, stresses: object, *, step: float
) -> ContextualRiskNode:
    """Move every governed shape, societal and lab, by one step."""
    original = node.model_dump(mode="json")
    payload = node.model_dump(mode="json")
    if node.total is not None:
        payload["total"]["gpd_shape"] = _shifted(
            node.total.gpd_shape, step, node.total.cap, "shared-total"
        )
    for component in payload["components"]:
        if component.get("gpd_shape") is not None:
            component["gpd_shape"] = _shifted(
                component["gpd_shape"],
                step,
                component.get("severity_cap"),
                f"component {component['component_id']!r}",
            )
    if node.lab_loss_gpd_shape is not None:
        payload["lab_loss_gpd_shape"] = _shifted(
            node.lab_loss_gpd_shape,
            step,
            node.lab_loss_severity_cap,
            "aggregate lab-loss",
        )
    if payload == original:
        raise _Excluded("inapplicable", "The node has no governed GPD shape.")
    return ContextualRiskNode.model_validate(payload)


def _authored(stresses: AuthoredCapStresses | str | None) -> AuthoredCapStresses:
    if stresses is None:
        raise _Excluded(
            "unavailable", "No authored severity governance was supplied for this node."
        )
    if isinstance(stresses, str):
        raise _Excluded("unavailable", stresses)
    return stresses


def _with_cap(
    node: ContextualRiskNode, cap: float | None, *, lab: bool
) -> ContextualRiskNode:
    """The node with its shared total capped at ``cap`` (``SEV-CAP``).

    A total's large-loss levels lie below its cap, since no capped loss
    exceeds a level at or above it (``SEV-LARGE-LOSS-ANCHOR``). The
    very-large-loss count enters no reference law, so its level is dropped
    with its reason. The anchor can shape the law below the cap, so dropping
    it would change more than the cap, and such an alternative is
    unavailable. A lab law whose endpoint the alternative removes must keep a
    finite mean.
    """
    if node.total is None:
        raise _Excluded("inapplicable", "The node has no shared total to cap.")
    if cap is not None and cap < node.total.floor:
        raise _Excluded(
            "unavailable",
            "The authored cap alternative contradicts the retained qualified severity floor; a changed qualification/count model is required.",
        )
    payload = node.model_dump(mode="json")
    if cap is not None:
        level = node.total.large_loss_level
        if level is not None and level >= cap:
            raise _Excluded(
                "unavailable",
                f"The cap {usd_text(cap)} lies at or below the large-loss level "
                f"{usd_text(level)}, whose stated count can shape the law below "
                "that cap; dropping the anchor would change more than the cap, "
                "so this alternative is not generated.",
            )
        very = node.total.very_large_loss_level
        if very is not None and very >= cap:
            reason = (
                f"The very-large-loss level {usd_text(very)} is at or above the "
                f"alternative's cap {usd_text(cap)}, so no capped loss exceeds it and its "
                "diagnostic count does not apply under this alternative."
            )
            payload["total"]["very_large_loss_level"] = None
            payload["total"]["large_loss_level_exclusions"] = (
                *node.total.large_loss_level_exclusions,
                reason,
            )
    payload["total"]["cap"] = cap
    if lab and node.lab_loss_severity_cap == node.total.cap:
        shape = node.lab_loss_gpd_shape
        if cap is None and shape is not None and shape >= UNCAPPED_TAIL_SHAPE_BOUND:
            raise _Excluded(
                "unavailable",
                f"Removing the lab-loss endpoint leaves its tail shape {shape:g} at "
                f"or above the finite-mean bound {UNCAPPED_TAIL_SHAPE_BOUND:g}; a "
                "law without a finite mean is not generated.",
            )
        payload["lab_loss_severity_cap"] = cap
    return ContextualRiskNode.model_validate(payload)


def _uncapped(node: ContextualRiskNode, stresses) -> ContextualRiskNode:
    authored = _authored(stresses)
    if not authored.uncapped_alternative:
        raise _Excluded(
            "inapplicable",
            "No uncapped outer-cap removal stress is authored for this source law.",
        )
    return _with_cap(node, None, lab=authored.endpoint_from_tail_class)


def _cap_alternative(
    node: ContextualRiskNode, stresses, *, index: int
) -> ContextualRiskNode:
    authored = _authored(stresses)
    if authored.cap_values is None:
        raise _Excluded(
            "inapplicable", "The source severity has no authored outer cap."
        )
    return _with_cap(
        node, authored.cap_values[index], lab=authored.endpoint_from_tail_class
    )


def _endpoint(node: ContextualRiskNode, stresses) -> ContextualRiskNode:
    authored = _authored(stresses)
    if authored.endpoint_stress is None:
        raise _Excluded(
            "unavailable"
            if node.total is not None and node.total.cap is None
            else "inapplicable",
            "No finite endpoint stress is authored for this source law.",
        )
    # Only uncapped scenario postures author this endpoint (today one fine
    # matter); like a statutory cap it bounds the scenario's severity population,
    # not the lab's other costs, so the lab law keeps its own posture.
    return _with_cap(node, authored.endpoint_stress, lab=False)


def _overdispersed(node: ContextualRiskNode, stresses) -> ContextualRiskNode:
    if node.count_family is None:
        raise _Excluded("inapplicable", "The node has no annual count law.")
    payload = node.model_dump(mode="json")
    payload["count_family"] = "negative_binomial"
    payload["count_variance_to_mean"] = COUNT_FAMILY_VARIANCE_TO_MEAN
    return ContextualRiskNode.model_validate(payload)


def _count_law_reading(options: CompileOptions) -> str:
    """How an arm's description states the reading of its count laws: the
    spread and tail form, and any factor on the stated counts."""
    reading = (
        f"Count laws use k = {options.rank_spread:g} with "
        f"tail_form = {options.spread_tail_form}."
    )
    if options.count_factor != 1:
        reading += f" Count factor = {options.count_factor:g}."
    return reading


def _numbers(law: IncidentLaw) -> dict:
    numbers = law.model_dump(mode="json", exclude={"conditioning_context"})
    rank = law.parameter_rank
    if rank is not None:
        numbers["effective_parameter_rank_tail_widths"] = tail_widths(
            epistemic_knots(rank.count, rank.epistemic_tail_scale),
            spread=rank.rank_spread,
            epistemic_tail_scale=rank.epistemic_tail_scale,
            tail_form=rank.tail_form,
        )
    return numbers


@dataclass(frozen=True)
class _SimulatedArm:
    """One simulated comparison arm: the change it applies to every node in
    scope, and how it describes it.

    ``edit`` changes each scenario's node, or excludes the scenario with its
    reason (``_Excluded``); the other fields change the reference compile
    options (``options``). ``count_factor`` multiplies the reference count
    laws. ``unchanged`` is why a scenario whose laws the change leaves
    unchanged is inapplicable.
    """

    arm: str
    family: str
    description: str
    unchanged: str
    edit: Callable[[ContextualRiskNode, object], ContextualRiskNode] | None = None
    rank_spread: float | None = None
    tail_factor: float = 1.0
    count_factor: float = 1.0
    frontier_lab_value: float | None = FRONTIER_LAB_VALUE_USD
    lab_loss_provider_only: bool = False
    large_loss_factor: float | None = 1.0
    very_large_loss_anchor: bool = False

    def options(self, reference: CompileOptions) -> CompileOptions:
        """The arm's compile options: the reference options with its change."""
        return replace(
            reference,
            rank_spread=reference.rank_spread
            if self.rank_spread is None
            else self.rank_spread,
            epistemic_tail_scale=reference.epistemic_tail_scale * self.tail_factor,
            count_factor=reference.count_factor * self.count_factor,
            frontier_lab_value=self.frontier_lab_value,
            lab_loss_provider_only=self.lab_loss_provider_only,
            large_loss_factor=self.large_loss_factor,
            very_large_loss_anchor=self.very_large_loss_anchor,
        )


def _simulated_arms() -> tuple[_SimulatedArm, ...]:
    """Every arm the catalog's simulated families generate, in arm order."""
    low_anchor, high_anchor = LARGE_LOSS_ANCHOR_FACTORS
    lower_value, higher_value = FRONTIER_LAB_VALUE_SENSITIVITY_USD
    return (
        *(
            _SimulatedArm(
                f"tail_shape_{side}",
                "tail_shape",
                f"Every governed societal and lab-loss GPD shape {step:+g}; elicited knots and p99 are kept and the GPD scale re-derived.",
                "The source law has no GPD component.",
                edit=partial(_shift_tail, step=step),
            )
            for side, step in (
                ("lower", -TAIL_SHAPE_SENSITIVITY_STEP),
                ("higher", TAIL_SHAPE_SENSITIVITY_STEP),
            )
        ),
        *(
            _SimulatedArm(
                arm,
                family,
                description,
                "No large-loss anchor shapes this scenario's law at any state.",
                large_loss_factor=factor,
            )
            for arm, family, factor, description in (
                (
                    "large_loss_anchor_ignored",
                    "large_loss_anchor_ignored",
                    None,
                    "Every law compiled with the p99-only tail, the large-loss anchor ignored: a design comparison, not an assumption challenge or a lower bound.",
                ),
                (
                    "large_loss_anchor_lower",
                    "large_loss_anchor",
                    low_anchor,
                    f"Every large-loss share that shapes a law has its conditional odds above p99 multiplied by {low_anchor:.4g}, keeping which cases the anchor shapes (a judgmental size).",
                ),
                (
                    "large_loss_anchor_higher",
                    "large_loss_anchor",
                    high_anchor,
                    f"Every large-loss share that shapes a law has its conditional odds above p99 multiplied by {high_anchor:g}, keeping which cases the anchor shapes (a judgmental size).",
                ),
            )
        ),
        _SimulatedArm(
            "very_large_loss_anchor",
            "very_large_loss_anchor",
            "Every law whose stated very-large-loss share is positive and coherent also passes its tail through that share (the two-anchor tail): above a shaping large-loss anchor, or above the law's own p99 where the large-loss level lies within the stated quantiles. No reference law reads that share.",
            "No very-large-loss share shapes this scenario's two-anchor law at any state.",
            very_large_loss_anchor=True,
        ),
        *(
            _SimulatedArm(
                arm,
                "lab_loss_cap",
                f"Each episode's lab loss, and each node's annual business loss, capped at a frontier-lab value of {usd_text(value)} instead of {usd_text(FRONTIER_LAB_VALUE_USD)}, or at the tail-class endpoint where that is lower."
                if value is not None
                else "The frontier-lab value cap removed; the lab law keeps only the tail-class endpoint where one exists.",
                "The lab-loss law is unchanged: no stochastic or positive lab loss reaches the cap.",
                frontier_lab_value=value,
            )
            for arm, value in (
                ("lab_loss_cap_lower", lower_value),
                ("lab_loss_cap_higher", higher_value),
                ("lab_loss_cap_removed", None),
            )
        ),
        _SimulatedArm(
            "lab_loss_provider_only",
            "lab_loss_provider_only",
            "Each episode's lab loss is only its provider-borne components, with the lab compensation share and annual business loss set to zero: the lower bound of the lab-loss rule.",
            "The node's lab compensation share and annual business loss are already exactly zero.",
            lab_loss_provider_only=True,
        ),
        _SimulatedArm(
            "uncapped_severity",
            "uncapped_severity",
            "Every cap whose severity governance requests the uncapped alternative removed at the governed shape, with the lab-loss cap where it is the tail-class endpoint.",
            "Removing the cap leaves every compiled law unchanged.",
            edit=_uncapped,
        ),
        *(
            _SimulatedArm(
                f"severity_cap_{side}",
                "severity_cap",
                f"Every authored finite cap at its {side} authored value, with the lab-loss cap where it is the tail-class endpoint; a very-large-loss diagnostic at or above the new cap does not apply and is listed in the arm's provenance.",
                "The cap alternative leaves every compiled law unchanged.",
                edit=partial(_cap_alternative, index=index),
            )
            for index, side in enumerate(("lower", "higher"))
        ),
        _SimulatedArm(
            "severity_endpoint",
            "severity_endpoint_stress",
            "Every uncapped posture with an authored finite endpoint capped at it; a very-large-loss diagnostic at or above the new cap does not apply and is listed in the arm's provenance.",
            "The endpoint leaves every compiled law unchanged.",
            edit=_endpoint,
        ),
        _SimulatedArm(
            "fixed_vmr_nb_2",
            "count_family",
            f"Negative-binomial annual counts with variance-to-mean {COUNT_FAMILY_VARIANCE_TO_MEAN:g} at fixed mean.",
            "The count law is unchanged.",
            edit=_overdispersed,
        ),
        *(
            _SimulatedArm(
                f"epistemic_spread_{side}",
                "epistemic_spread_calibration",
                f"Uncalibrated alternative: every parameter rank u is read at Phi({factor:g} Phi^-1(u)); medians and the five stated count quantiles are fixed, with tail_form = {REFERENCE_SPREAD_TAIL_FORM}.",
                "Degenerate stated parameter quantiles leave the law unchanged.",
                rank_spread=factor,
            )
            for side, factor in zip(
                ("widened", "narrowed"), EPISTEMIC_SPREAD_FACTORS, strict=True
            )
        ),
        *(
            _SimulatedArm(
                f"epistemic_tail_{side}",
                "epistemic_tail_width",
                f"At the reference k, multiply extrapolated parameter tail widths by {factor:g} with tail_form = {REFERENCE_SPREAD_TAIL_FORM}; all five count quantiles and non-count inputs are fixed.",
                "Degenerate stated parameter quantiles leave the law unchanged.",
                tail_factor=factor,
            )
            for side, factor in zip(
                ("narrowed", "widened"), EPISTEMIC_TAIL_FACTORS, strict=True
            )
        ),
        *(
            _SimulatedArm(
                f"common_frequency_bias_{side}",
                "common_frequency_bias",
                f"Multiply every node's five expected-count quantiles by {factor:g} at the reference k and tail form; severity, branch probabilities and shares stay fixed.",
                "A zero count law leaves the common frequency-bias arm unchanged.",
                count_factor=factor,
            )
            for side, factor in zip(
                ("lower", "higher"), COMMON_FREQUENCY_BIAS_FACTORS, strict=True
            )
        ),
    )


class _Comparisons:
    """The comparison arms and frozen requirements of one study, as
    ``public_frontier_comparisons`` builds them: every arm in order, and the
    notes of every family, scenario and status that has no arm."""

    def __init__(
        self,
        nodes: Sequence[ContextualRiskNode],
        states: Sequence[JointState],
        comparison_families: tuple[str, ...] | None,
        reference: CompileOptions,
        writer: OverrideWriter | None = None,
    ) -> None:
        self.writer = OverrideWriter() if writer is None else writer
        catalog = {family.family_id for family in SENSITIVITY_FAMILIES}
        if set(GENERATED_FAMILIES) != catalog:
            raise ValueError(
                "context-bound comparisons must generate every catalog family"
            )
        selected = set(
            DEFAULT_CONTEXT_BOUND_FAMILIES
            if comparison_families is None
            else comparison_families
        )
        if not selected <= catalog:
            raise ValueError(
                f"unknown comparison families: {sorted(selected - catalog)}"
            )
        self.catalog = catalog
        self.selected = selected
        self.reference = reference
        self.ids = tuple(node.population.scenario_id for node in nodes)
        self.by_scenario = {node.population.scenario_id: node for node in nodes}
        self.weights = {state.state_id: state.weight for state in states}
        self.notes: dict[tuple[str, Status, str], list[str]] = defaultdict(list)
        self.alternatives: list[Alternative] = []
        self.required: list[ComparisonRequirement] = []
        selection = (
            "the context-bound compact default comparison set"
            if comparison_families is None
            else "this import's comparison-family list"
        )
        for family in sorted(catalog):
            if family not in selected:
                self.note(
                    family,
                    self.ids,
                    "unavailable",
                    f"Not selected by {selection}; request it with --comparison-family {family} or --comparison-family all.",
                )

    def note(
        self, family: str, scope: Sequence[str], status: Status, reason: str
    ) -> None:
        """Record that ``family`` has no arm for the scenarios in ``scope``."""
        if scope:
            self.notes[family, status, reason].extend(scope)

    def reweight(
        self,
        arm,
        family,
        state_weights,
        description,
        provenance,
        *,
        control: str | None = None,
        held_fixed: tuple[str, ...] = HELD_FIXED,
    ) -> None:
        """Add a reweighting arm: the reference laws at other state weights."""
        weights = self.weights
        if set(state_weights) - set(weights):
            raise ValueError("reweighting arms must weight declared joint states")
        description = f"{description} {_count_law_reading(self.reference)}"
        self.alternatives.append(
            Alternative(
                alternative_id=arm,
                purpose="scientific_alternative",
                description=description,
                state_weights={key: state_weights.get(key, 0.0) for key in weights},
                matched_control_id=control,
                selection_provenance={"family_id": family, **provenance},
            )
        )
        self.required.append(
            ComparisonRequirement(
                family_id=family,
                arm_id=arm,
                scenario_ids=self.ids,
                held_fixed=held_fixed,
            )
        )

    def forecast_arms(
        self,
        cells: Sequence[StateCell],
        forecast_alternatives: Mapping[str, PublicForecastAlternative],
    ) -> None:
        """Each alternative forecast's trajectory nodes reweighted."""
        if "public_forecast" not in self.selected:
            return
        ids = self.ids
        if not forecast_alternatives:
            self.note(
                "public_forecast",
                ids,
                "unavailable",
                "No public-forecast alternatives were supplied with the trajectory nodes.",
            )
        draws = {cell.draw_id for cell in cells}
        for label, alternative in forecast_alternatives.items():
            if alternative.path_weights is None:
                self.note(
                    "public_forecast",
                    ids,
                    "unavailable",
                    f"{label}: {alternative.reason}",
                )
                continue
            if set(alternative.path_weights) != draws:
                raise ValueError(
                    "forecast alternative weights must cover exactly the trajectory nodes"
                )
            self.reweight(
                f"public_forecast::{label}",
                "public_forecast",
                {
                    cell.state_id: alternative.path_weights[cell.draw_id]
                    * cell.selection.weight
                    for cell in cells
                },
                "The study's trajectory nodes reweighted by the alternative forecast's exact quadrature weights; every conditional law, respondent and rank is unchanged.",
                {"base_support_only": True, **alternative.provenance},
            )

    def pooling_arms(self, cells: Sequence[StateCell]) -> None:
        """The respondents restricted to each model, and to all models but one."""
        weights = self.weights
        models = sorted({cell.selection.respondent.model_id for cell in cells})
        for family, minimum, subsets in (
            ("model_pooling", 2, [(model, {model}) for model in models]),
            (
                "model_pooling_leave_one_out",
                3,
                [(model, set(models) - {model}) for model in models],
            ),
        ):
            if family not in self.selected:
                continue
            if len(models) < minimum:
                self.note(
                    family,
                    self.ids,
                    "inapplicable",
                    f"This recipe requires at least {minimum} respondent models; the setup has {len(models)}.",
                )
                continue
            for label, retained in subsets:
                kept = {
                    cell.state_id
                    for cell in cells
                    if cell.selection.respondent.model_id in retained
                }
                mass = math.fsum(weights[key] for key in kept)
                self.reweight(
                    f"{family}::{label}",
                    family,
                    {
                        key: weights[key] / mass if key in kept else 0.0
                        for key in weights
                    },
                    f"Respondents restricted to models {sorted(retained)}, keeping their base relative weights; trajectories, ranks and laws are unchanged.",
                    {"base_support_only": True, "retained_models": sorted(retained)},
                )

    def context_variant_arms(
        self,
        cells: Sequence[StateCell],
        context_variants: Mapping[str, Mapping[str, Sequence[StateCell]]],
        embedded_path_weights: Mapping[str, float],
        *,
        path_linked: bool,
    ) -> None:
        """Each context-variant family's matched control and variant arms."""
        ids = self.ids
        for kind in CONTEXT_VARIANT_FAMILIES.values():
            family = kind.family_id
            variants = context_variants.get(family, NO_MAPPING)
            if set(variants) - set(kind.variants):
                raise ValueError(
                    f"{kind.noun} variant must be one of {sorted(kind.variants)}"
                )
            if variants and family not in self.selected:
                raise ValueError(
                    f"{kind.noun.split(' ', 1)[1]} variant studies require the {family} "
                    "comparison family"
                )
            if family not in self.selected:
                continue
            if family == "exposure_scale" and not path_linked:
                self.note(
                    family,
                    ids,
                    "unavailable",
                    "The study's deployment scales are authored text, not path-linked values, so no variant can rescale them.",
                )
                continue
            if not variants:
                self.note(
                    family,
                    ids,
                    "unavailable",
                    f"No {kind.noun.split(' ', 1)[1]} variant studies were imported; collect each variant as a public study on the embedded rule's nodes and the no-progress node and import it with {kind.study_option}.",
                )
                continue
            self.reweight(
                kind.control,
                family,
                {
                    cell.state_id: embedded_path_weights[cell.draw_id]
                    * cell.selection.weight
                    for cell in cells
                },
                kind.control_description,
                {"base_support_only": True, "trajectory_rule": "embedded"},
                held_fixed=kind.held_fixed,
            )
            for variant_id, (description, provenance) in kind.variants.items():
                variant_cells = variants.get(variant_id)
                if variant_cells is None:
                    self.note(
                        family,
                        ids,
                        "unavailable",
                        f"{variant_id}: no variant study was imported.",
                    )
                    continue
                self.reweight(
                    f"{family}::{variant_id}",
                    family,
                    {
                        cell.state_id: cell.path_weight * cell.selection.weight
                        for cell in variant_cells
                    },
                    description,
                    {"base_support_only": False, **provenance},
                    control=kind.control,
                    held_fixed=kind.held_fixed,
                )

    def simulate(
        self,
        arm: _SimulatedArm,
        cells: Sequence[StateCell],
        compile_cells: CompileCells,
        cap_stresses: Mapping[str, AuthoredCapStresses | str],
    ) -> None:
        """Add a simulated arm: the laws its change alters, recompiled in
        every scenario in its scope, with notes for the scenarios it
        excludes, fails or leaves unchanged."""
        family, edit = arm.family, arm.edit
        if family not in self.selected:
            return
        ids = self.ids
        edits = {}
        level_exclusions: dict[str, tuple[str, ...]] = {}
        if edit is not None:
            for scenario_id, node in self.by_scenario.items():
                stresses = cap_stresses.get(scenario_id)
                try:
                    edited = edit(node, stresses)
                except _Excluded as exc:
                    self.note(family, (scenario_id,), exc.status, exc.reason)
                    continue
                added = _level_exclusions(edited)[len(_level_exclusions(node)) :]
                if added:
                    level_exclusions[scenario_id] = added
                edits[scenario_id] = partial(_apply, edit, stresses)
        scope = set(edits) if edit is not None else set(ids)
        if not scope:
            return
        options = arm.options(self.reference)
        description = f"{arm.description} {_count_law_reading(options)}"
        variant = compile_cells(options, edits=edits)
        pending: dict[str, list[LawOverride]] = defaultdict(list)
        failed: dict[str, str] = {}
        for base, other in zip(cells, variant, strict=True):
            for scenario_id in scope - set(failed):
                before, after = base.outcomes[scenario_id], other.outcomes[scenario_id]
                if (
                    isinstance(after, PopulationAbsence)
                    and after.disposition == "unavailable"
                ):
                    failed[scenario_id] = after.reason
                elif isinstance(before, tuple) != isinstance(after, tuple) or (
                    isinstance(before, tuple)
                    and isinstance(after, tuple)
                    and [law.origin_id for law in before]
                    != [law.origin_id for law in after]
                ):
                    failed[scenario_id] = (
                        "The challenge changes which source-status populations have a "
                        "positive expected count, which needs a new joint-state support."
                    )
                elif isinstance(before, tuple) and isinstance(after, tuple):
                    pending[scenario_id].extend(
                        LawOverride(state_id=base.state_id, law=new)
                        for old, new in zip(before, after, strict=True)
                        if _numbers(old) != _numbers(new)
                    )
        for scenario_id, reason in failed.items():
            self.note(family, (scenario_id,), "unavailable", reason)
        changed = tuple(
            scenario_id
            for scenario_id in ids
            if scenario_id in scope
            and scenario_id not in failed
            and pending[scenario_id]
        )
        self.note(
            family,
            [
                scenario_id
                for scenario_id in ids
                if scenario_id in scope
                and scenario_id not in failed
                and scenario_id not in changed
            ],
            "inapplicable",
            arm.unchanged,
        )
        if not changed:
            return
        selection_provenance: dict[str, object] = {
            "family_id": family,
            "scenario_ids": changed,
            "rank_spread": options.rank_spread,
            "rank_spread_tail_form": options.spread_tail_form,
            "count_factor": options.count_factor,
        }
        excluded_levels = {
            scenario_id: level_exclusions[scenario_id]
            for scenario_id in changed
            if scenario_id in level_exclusions
        }
        if excluded_levels:
            selection_provenance["large_loss_level_exclusions"] = excluded_levels
        # Written, and released, before the next arm compiles.
        self.writer.add(
            arm.arm,
            (override for scenario_id in changed for override in pending[scenario_id]),
        )
        self.alternatives.append(
            Alternative(
                alternative_id=arm.arm,
                purpose="scientific_alternative",
                description=description,
                selection_provenance=selection_provenance,
            )
        )
        self.required.append(
            ComparisonRequirement(
                family_id=family,
                arm_id=arm.arm,
                scenario_ids=changed,
                held_fixed=HELD_FIXED,
            )
        )

    def result(self) -> PublicFrontierComparisons:
        """Every arm, then a requirement for every arm, every note and every
        register challenge no context-bound setup executes."""
        notes = [
            ComparisonRequirement(
                family_id=family,
                scenario_ids=tuple(dict.fromkeys(scope)),
                status=status,
                reason=reason,
                held_fixed=HELD_FIXED,
            )
            for (family, status, reason), scope in sorted(self.notes.items())
        ]
        rows = (
            *self.required,
            *notes,
            *unavailable_challenge_requirements(
                PUBLIC_STUDY_UNAVAILABLE_CHALLENGES, self.ids
            ),
        )
        missing = self.catalog - {row.family_id for row in rows}
        if missing:
            raise ValueError(
                f"context-bound comparisons omitted catalog families: {sorted(missing)}"
            )
        return PublicFrontierComparisons(
            tuple(self.alternatives), rows, self.writer.overrides()
        )


def public_frontier_comparisons(
    cells: Sequence[StateCell],
    states: Sequence[JointState],
    nodes: Sequence[ContextualRiskNode],
    compile_cells: CompileCells,
    *,
    reference: CompileOptions,
    comparison_families: tuple[str, ...] | None,
    forecast_alternatives: Mapping[str, PublicForecastAlternative],
    cap_stresses: Mapping[str, AuthoredCapStresses | str],
    path_linked: bool = False,
    context_variants: Mapping[str, Mapping[str, Sequence[StateCell]]] = NO_MAPPING,
    embedded_path_weights: Mapping[str, float] = NO_MAPPING,
    writer: OverrideWriter | None = None,
) -> PublicFrontierComparisons:
    """Freeze every catalog family and build the selected arms this route generates.

    ``compile_cells`` recompiles the reference cells under an arm's
    ``CompileOptions``, changed from the ``reference`` options, and keyword
    ``edits`` (node changes by scenario). Each simulated arm
    (``_simulated_arms``) applies its change to every node in scope at once,
    so it costs one reference-equivalent run however many scenarios it
    touches.
    ``context_variants`` holds each context-variant family's zero-weight state cells,
    by family and variant ID (``CONTEXT_VARIANT_FAMILIES``), which ``states``
    includes; their arms reweight to them and are measured against the
    family's control, the headline cells reweighted to
    ``embedded_path_weights``. ``writer`` takes each simulated arm's law
    overrides as the arm compiles; one that writes them to a directory lets
    the study hold one arm at a time (``OverrideWriter``).
    """
    comparisons = _Comparisons(nodes, states, comparison_families, reference, writer)
    comparisons.forecast_arms(cells, forecast_alternatives)
    comparisons.pooling_arms(cells)
    comparisons.context_variant_arms(
        cells, context_variants, embedded_path_weights, path_linked=path_linked
    )
    for arm in _simulated_arms():
        comparisons.simulate(arm, cells, compile_cells, cap_stresses)
    return comparisons.result()


def _apply(edit, stresses, node: ContextualRiskNode) -> ContextualRiskNode:
    return edit(node, stresses)


def _level_exclusions(node: ContextualRiskNode) -> tuple[str, ...]:
    return () if node.total is None else node.total.large_loss_level_exclusions
