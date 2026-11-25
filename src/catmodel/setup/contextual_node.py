"""Compile one contextual node's answers into its admitted annual law.

A node's law at a coordinate interpolates its bracketing anchors' answers
(``compile_contextual_node``) and is admitted as a run admits it
(``admit_node``). A leaf's expected annual welfare is its expected count times
its law's analytic mean per unit count (``leaf_eal``), which the taxonomy
parents' pools and the EAL cross-checks read. The setup compilers, the
exhibits, the hierarchy and the cross-checks all build on these.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal

from pydantic import TypeAdapter

from catmodel.contracts.components import (
    AggregateLabLossLaw,
    AnnualLabLossLaw,
    ComponentAmountLaw,
)
from catmodel.contracts.contextual_risk import (
    ContextualComponent,
    ContextualComponentAnswer,
    ContextualRiskAnswer,
    ContextualRiskInstrument,
    ContextualRiskNode,
    SharedTotal,
)
from catmodel.contracts.elicitation import (
    Quantiles,
    SeverityQuantiles,
)
from catmodel.contracts.incident import (
    SPREAD_TAIL_FORMS,
    ClusterCount,
    DeploymentBinding,
    Entity,
    EntityPath,
    ExecutionPlan,
    IncidentLaw,
    IncidentSegment,
    JointMarkOutcome,
    JointState,
    LocalResponse,
    MarkKernel,
    ParameterRankLaw,
    PathKnot,
    Population,
    PopulationAbsence,
    ResponsePoint,
    SpreadTailForm,
    UnitCountMass,
)
from catmodel.contracts.native_severity import (
    NativeSeverityComponent,
    NativeSeverityLaw,
    native_severity_from_sampler,
    native_severity_sampler,
)
from catmodel.contracts.run_setup import (
    AssumptionSet,
    EvidenceSet,
    RunScope,
    RunSetup,
)
from catmodel.elicitation.epistemic import epistemic_mean
from catmodel.elicitation.large_loss import (
    contradiction,
    large_loss_pin,
    quantile_large_loss_share,
    very_large_loss_pin,
)
from catmodel.probability.distributions import (
    LARGE_LOSS_BREAK_EXCEEDANCE,
    Distribution,
    p999_implied_gpd_shape,
    severity_lower_support,
)
from catmodel.probability.numerics import TIE_RELATIVE_TOLERANCE
from catmodel.setup.rank_spread import REFERENCE_SPREAD_TAIL_FORM
from catmodel.setup.sensitivity_catalog import (
    FRONTIER_LAB_VALUE_USD,
)
from catmodel.simulation.incident import check_incident_state
from catmodel.simulation.incident_moments import single_period_law_means

_QUANTILES = ("p5", "p25", "p50", "p75", "p95")


LAB_LOSS_TAIL_LIMITATION = (
    "A year's lab loss from a node is, for each episode, the amounts of its "
    "provider-borne components plus an elicited compensation share of its other "
    "components, both from the same draw as the societal marks and capped once "
    "per episode, plus the lab's annual business loss, elicited directly as one "
    "combined amount for all the node's episodes in a year with at least one "
    "and drawn once in each such simulated year (ACC-LAB-BEARER, "
    "ACC-LAB-ANNUAL-BUSINESS-LOSS). The compensation share is one point share "
    "per respondent and anchor, so it moves with the harm it compensates and "
    "never exceeds it; its variation between episodes is not represented. The "
    "annual business loss does not depend on the year's count beyond at least "
    "one: a respondent's single annual judgment does not vary with the "
    "parameter rank of their own count, so years at high count ranks may "
    "understate business loss. Its conditional-positive GPD tail above p95 uses "
    "the scenario's governed tail shape, the same judgment as the societal "
    "law, with scale from the respondent's p95 and p99. Each episode's sum and "
    "the annual business loss are each capped at the stipulated frontier-lab "
    "value, or at the tail class's endpoint convention where that is lower; no "
    "lab-specific shape is applied. Where the episode cap can bind, the capped "
    "episode mean is integrated exactly, since every contextual episode has one "
    "member; the annual term's mean is its law's mean times the probability of "
    "at least one episode in a year, integrated over the count's parameter "
    "rank. The shape implied by each respondent's business-loss p999 is "
    "reported as a diagnostic and never enters the law. Business loss is "
    "conditionally independent of societal severity; that dependence "
    "sensitivity is unassessed, and paired draws do not identify an empirical "
    "cross-measure copula. Business loss is counted in the gross view, so the "
    "lab-loss view never exceeds gross loss."
)


LAB_LOSS_DEPENDENCE = (
    "Provider-borne components, and the lab compensation share of every other "
    "component, count as the lab's loss from the same draw. Lab annual business "
    "loss is drawn once per year with at least one episode, conditionally "
    "independent of that year's count beyond at least one, of societal severity "
    "and of joint branches given the context and respondent. Counts and "
    "epistemic selections remain shared."
)


def hypothetical_entity_limitations(entities: Sequence[Entity]) -> tuple[str, ...]:
    """Name every hypothetical entity a setup describes; nothing for actual ones.

    Setups list it first, so every result's limitations open with it.
    """
    named = [
        f"{entity.display_name} ({entity.kind} `{entity.entity_id}`)"
        for entity in entities
        if entity.status == "hypothetical"
    ]
    if not named:
        return ()
    return (
        "Hypothetical entities, stipulated rather than observed: "
        + "; ".join(named)
        + ". Results describe these stipulated entities at their stated exposure, "
        "not any actual company, model or deployment.",
    )


def threat_model_limitations(instrument: ContextualRiskInstrument) -> tuple[str, ...]:
    """What a setup says about the threat models its questions showed: how
    many scenarios had one, and that an analysis every respondent reads can
    anchor the panel. Nothing where no question showed one."""
    shown = sum(node.threat_model is not None for node in instrument.nodes)
    if not shown:
        return ()
    return (
        f"Respondents read a threat model for {shown} of "
        f"{len(instrument.nodes)} scenarios before answering: research on how "
        "the scenario's incidents arise, written before any quantity was "
        "asked. An analysis that every respondent reads can anchor the panel "
        "on one framing, and no collection without it measures that.",
    )


class ContextualLawUnavailable(ValueError):
    """One node at one coordinate lacks an identified executable conditional law."""


@dataclass(frozen=True)
class CompileOptions:
    """How a node's answers compile into its law (``compile_contextual_node``).

    The defaults are the reference law's; each simulated comparison arm
    changes one option (``public_frontier_comparisons``), and the exhibits
    compile laws at unit expected count.

    ``epistemic_tail_scale`` scales the count law's extrapolated tails and
    ``rank_spread`` the parameter score it is read at, with
    ``spread_tail_form`` its post-stretch tail rule; ``count_factor`` scales
    every count quantile and the integrated rate, a common-frequency-bias
    arm's factor. ``frontier_lab_value`` caps
    each episode's lab loss and the annual business loss (``SEV-LAB-CAP``;
    None removes the cap), ``lab_loss_provider_only`` keeps only the
    provider-borne lab loss and ``large_loss_factor`` scales a shaping
    large-loss share's conditional odds (None ignores the anchor).
    ``very_large_loss_anchor`` also passes the tail through the stated
    very-large-loss share wherever it is coherent, the two-anchor comparison
    (``_very_large_loss_anchor``); no reference law does.
    ``unit_count`` compiles the law at unit expected count.
    """

    epistemic_tail_scale: float = 1.0
    rank_spread: float = 1.0
    spread_tail_form: SpreadTailForm = REFERENCE_SPREAD_TAIL_FORM
    count_factor: float = 1.0
    frontier_lab_value: float | None = FRONTIER_LAB_VALUE_USD
    lab_loss_provider_only: bool = False
    large_loss_factor: float | None = 1.0
    very_large_loss_anchor: bool = False
    unit_count: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.count_factor) or self.count_factor <= 0:
            raise ValueError("count factor must be positive and finite")
        if self.spread_tail_form not in SPREAD_TAIL_FORMS:
            raise ValueError(f"unknown spread tail form: {self.spread_tail_form!r}")


def interpolate(left: float, right: float, fraction: float) -> float:
    # This form does not overflow on two large, finite positive endpoints.
    return (1 - fraction) * left + fraction * right


def bracket_anchors(anchors, score: float) -> tuple[float, float, float]:
    if not math.isfinite(score):
        raise ValueError("contextual coordinate must be finite")
    anchors = sorted(anchors)
    if not anchors or score < anchors[0] or score > anchors[-1]:
        raise ContextualLawUnavailable(
            "Coordinate is outside the elicited anchor hull."
        )
    if score in anchors:
        return score, score, 0.0
    for lower, upper in pairwise(anchors):
        if lower < score < upper:
            return lower, upper, (score - lower) / (upper - lower)
    raise ContextualLawUnavailable("Coordinate has no enclosing elicited anchors.")


def interpolated_rate(left: Quantiles, right: Quantiles, fraction: float) -> Quantiles:
    return Quantiles(
        **{
            key: interpolate(getattr(left, key), getattr(right, key), fraction)
            for key in _QUANTILES
        }
    )


def scaled_rate(rate: Quantiles, count_factor: float) -> Quantiles:
    """A count law with each of its five quantiles times ``count_factor``,
    which scales the whole law, and its mean at any spread, by that factor."""
    if count_factor == 1:
        return rate
    return Quantiles(**{key: getattr(rate, key) * count_factor for key in _QUANTILES})


def integrated_count(
    left: Quantiles,
    right: Quantiles,
    fraction: float,
    epistemic_tail_scale: float,
    rank_spread: float = 1.0,
    spread_tail_form: SpreadTailForm = REFERENCE_SPREAD_TAIL_FORM,
    count_factor: float = 1.0,
) -> float | None:
    """The interpolated count law's mean over the parameter score, read at
    ``rank_spread * z`` and multiplied by ``count_factor``; None when the mean
    is not finite."""
    try:
        return float(
            count_factor
            * epistemic_mean(
                interpolated_rate(left, right, fraction),
                epistemic_tail_scale=epistemic_tail_scale,
                spread=rank_spread,
                tail_form=spread_tail_form,
            )
        )
    except ValueError:
        return None


def _component(
    definition: ContextualComponent,
    left: ContextualComponentAnswer,
    right: ContextualComponentAnswer,
    fraction: float,
    *,
    scenario_id: str,
    lab_id: str,
    total: NativeSeverityLaw | None = None,
    total_rank: str = "shared_total",
    partition_share: float | None = None,
    compensation_share: float = 0.0,
) -> ComponentAmountLaw:
    # A provider-borne component is the named lab's own loss from this draw;
    # the lab pays ``compensation_share`` of every other component's amount
    # from the same draw (``ACC-LAB-BEARER``).
    weight = 1.0 if definition.bearer == "provider" else compensation_share
    lab_loss_weights = {lab_id: weight} if weight > 0 else {}
    for answer in (left, right):
        if answer.status != "available":
            raise ContextualLawUnavailable(
                f"Component {definition.component_id!r} is unavailable: {answer.unavailable_reason}"
            )
    if (left.amount is None) != (right.amount is None):
        raise ContextualLawUnavailable(
            f"Component {definition.component_id!r} changes amount-law family between anchors."
        )
    if definition.role == "partition":
        if total is None or left.share is None or right.share is None:
            raise ContextualLawUnavailable(
                "A partition requires its identified shared total and shares."
            )
        return ComponentAmountLaw(
            component_id=definition.component_id,
            nature=definition.nature,
            lab_loss_weights=lab_loss_weights,
            scenario_ids=(scenario_id,),
            severity=total,
            rank_group_id=total_rank,
            rank_scope="campaign",
            post_multiplier=partition_share
            if partition_share is not None
            else interpolate(left.share, right.share, fraction),
        )
    if left.amount is not None and right.amount is not None:
        return ComponentAmountLaw(
            component_id=definition.component_id,
            nature=definition.nature,
            lab_loss_weights=lab_loss_weights,
            scenario_ids=(scenario_id,),
            amount=interpolate(left.amount, right.amount, fraction),
        )
    if left.loss is None or right.loss is None or definition.gpd_shape is None:
        raise ContextualLawUnavailable(
            f"Component {definition.component_id!r} has no complete governed severity law."
        )
    try:
        quantiles = SeverityQuantiles(
            **{
                key: interpolate(
                    getattr(left.loss, key), getattr(right.loss, key), fraction
                )
                for key in (*_QUANTILES, "p99", "p999")
            }
        )
        if quantiles.p95 <= 0:
            raise ContextualLawUnavailable("A GPD splice requires a positive p95.")
        severity = NativeSeverityLaw(
            representation="distribution",
            components=(
                NativeSeverityComponent(
                    **{key: getattr(quantiles, key) for key in _QUANTILES},
                    gpd_shape=definition.gpd_shape,
                    gpd_tail_ratio=quantiles.p99 / quantiles.p95,
                    severity_minimum=0,
                    severity_cap=definition.severity_cap,
                ),
            ),
            weights=(1.0,),
        )
    except ValueError as exc:
        raise ContextualLawUnavailable(
            f"Component {definition.component_id!r} has unsupported native severity: {exc}"
        ) from exc
    return ComponentAmountLaw(
        component_id=definition.component_id,
        nature=definition.nature,
        lab_loss_weights=lab_loss_weights,
        scenario_ids=(scenario_id,),
        severity=severity,
        rank_group_id=definition.rank_group_id,
        rank_scope="campaign",
    )


LargeLossRole = Literal[
    "shapes_tail_above_p99",
    "diagnostic_within_stated_quantiles",
    "no_anchor",
    "severity_unresolved_fallback",
    "interpolated_band_conflict",
    "skipped_at_or_below_computed_p99",
]
"""How a coordinate's large-loss anchor enters its law (``SEV-LARGE-LOSS-ANCHOR``).

``shapes_tail_above_p99``: the tail above the law's own p99 passes the
anchor (``Distribution.large_loss_amount``).
``diagnostic_within_stated_quantiles``: the level lies within the stated
quantiles, which govern. ``no_anchor``: an exact-zero anchor or count.
``severity_unresolved_fallback``: an elicited answer's share still
contradicts its own quantiles, so the law is the provisional p99-only tail.
``interpolated_band_conflict``: both bracketing answers are coherent but the
interpolated share falls outside the interpolated quantiles' band; the
stated quantiles govern. ``skipped_at_or_below_computed_p99``: the share
would shape the tail, but the law's own computed p99 is at or above the
level, so the stated quantiles govern.
"""


@dataclass(frozen=True)
class LargeLossAnchor:
    """The large-loss anchor at one coordinate and how it enters the law.

    ``share`` is the interpolated large-loss count law's mean over the
    interpolated count law's mean (``SEV-LARGE-LOSS-ANCHOR``), so under the
    rank-integrated count the compiled model's expected annual number of
    incidents at or above ``level`` is exactly that large-loss mean whenever
    the anchor shapes the law.
    """

    level: float
    share: float | None
    stated_p99: float
    role: LargeLossRole
    computed_p99: float | None = None

    @property
    def shapes_law(self) -> bool:
        return self.role == "shapes_tail_above_p99"

    def record(self) -> dict[str, object]:
        return {
            "large_loss_level": self.level,
            "large_loss_share": self.share,
            "stated_p99": self.stated_p99,
            "computed_p99": self.computed_p99,
            "large_loss_role": self.role,
        }


def _answer_share(answer: ContextualRiskAnswer) -> float | None:
    if answer.rate is None or answer.large_loss_count is None:
        return None
    return quantile_large_loss_share(answer.rate, answer.large_loss_count)


def odds_scaled_large_loss_share(share: float, factor: float) -> float:
    """A large-loss share with its conditional odds above p99 times ``factor``.

    The ``large_loss_anchor`` alternatives (``SEV-LARGE-LOSS-ANCHOR``). An incident
    above the law's own p99, whose exceedance probability is B =
    ``LARGE_LOSS_BREAK_EXCEEDANCE``, reaches the level with probability
    q = share / B, odds o = q / (1 - q). Multiplying the odds by k gives
    k * share / (1 + (k - 1) q): it stays strictly below B, in the band
    where a share shapes the law, for every share below B and every k > 0,
    moves the log-odds by exactly ln k, so reciprocal factors are symmetric
    alternatives, and approaches k * share for a small share. A zero share stays
    zero. Where rounding would reach 0 or B (a share within an ulp of B, or a
    subnormal one), the result is the nearest float strictly inside them.
    """
    bound = LARGE_LOSS_BREAK_EXCEEDANCE
    if not math.isfinite(share) or not 0.0 <= share < bound:
        raise ValueError(
            f"a large-loss share must lie in [0, {bound:g}), got {share!r}"
        )
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"an odds factor must be finite and positive, got {factor!r}")
    if share == 0.0:
        return 0.0
    scaled = factor * share / (1.0 + (factor - 1.0) * (share / bound))
    return min(max(scaled, math.ulp(0.0)), math.nextafter(bound, 0.0))


def _large_loss_anchor(
    specification: SharedTotal,
    left: ContextualRiskAnswer,
    right: ContextualRiskAnswer,
    fraction: float,
    quantiles: SeverityQuantiles,
    computed_p99: float | None,
    factor: float | None = 1.0,
) -> LargeLossAnchor | None:
    """The coordinate's large-loss anchor and its role, or None where none is
    asked.

    An elicited answer whose share contradicts its own quantiles can only be
    an unresolved case (collection retries every other), so its law falls
    back to the p99-only tail. Whether the anchor shapes the law is decided
    against the law's own computed p99, not the stated one.

    ``factor`` is the ``large_loss_anchor`` challenge: None ignores every
    anchor (the p99-only tail), and a number scales the conditional odds of a
    share that shapes the law via ``odds_scaled_large_loss_share``, keeping
    which cases it shapes.
    """
    level = specification.large_loss_level
    if factor is None:
        return None
    counts = (left.rate, right.rate, left.large_loss_count, right.large_loss_count)
    if level is None or any(value is None for value in counts):
        return None
    assert left.rate is not None and right.rate is not None
    assert left.large_loss_count is not None and right.large_loss_count is not None
    share = quantile_large_loss_share(
        interpolated_rate(left.rate, right.rate, fraction),
        interpolated_rate(left.large_loss_count, right.large_loss_count, fraction),
    )

    def contradicts(answer_share, loss) -> bool:
        return (
            answer_share is not None
            and loss is not None
            and contradiction(level, answer_share, loss.p95, loss.p99) is not None
        )

    role: LargeLossRole
    if any(
        contradicts(_answer_share(answer), answer.total_loss)
        for answer in (left, right)
    ):
        role = "severity_unresolved_fallback"
    elif share is None:
        role = "no_anchor"
    elif 0 < fraction < 1 and contradicts(share, quantiles):
        role = "interpolated_band_conflict"
    elif large_loss_pin(level, share, computed_p99) is not None:
        role = "shapes_tail_above_p99"
    elif large_loss_pin(level, share, quantiles.p99) is not None:
        role = "skipped_at_or_below_computed_p99"
    else:
        role = "diagnostic_within_stated_quantiles"
    if factor != 1 and role == "shapes_tail_above_p99" and share is not None:
        share = odds_scaled_large_loss_share(share, factor)
    return LargeLossAnchor(
        level=level,
        share=share,
        stated_p99=quantiles.p99,
        role=role,
        computed_p99=computed_p99,
    )


VeryLargeLossRole = Literal[
    "shapes_tail_above_large_loss_level",
    "shapes_tail_above_p99",
    "no_anchor",
    "not_below_large_loss_share",
    "contradicts_stated_quantiles",
    "large_loss_anchor_unresolved",
    "diagnostic_within_stated_quantiles",
    "skipped_at_or_below_computed_p99",
]
"""How a coordinate's very-large-loss share enters its two-anchor law
(``SEV-LARGE-LOSS-ANCHOR``); it enters no reference law.

``shapes_tail_above_large_loss_level``: the first anchor shapes the tail
above p99, and the tail above its level also passes this second point
(``Distribution.very_large_loss_amount``). ``shapes_tail_above_p99``: the
first level lies within the stated quantiles, which govern up to p99, and
the tail above the law's own p99 passes this point alone. ``no_anchor``: an
exact-zero share. ``not_below_large_loss_share``: the first anchor shapes the
tail and this share is not below its share beyond a tie, which would leave
no incident between the two levels. ``contradicts_stated_quantiles``: the
first level lies within the stated quantiles and this share implies a
different tail band from the one its level occupies. No retry asks about
either contradiction, so the law keeps its reference tail.
``large_loss_anchor_unresolved``: the first anchor is unresolved or in an
interpolated band conflict, so no anchor shapes the law.
``diagnostic_within_stated_quantiles`` and
``skipped_at_or_below_computed_p99``: as for the first anchor
(``LargeLossRole``), with the first level within the stated quantiles too.
"""


@dataclass(frozen=True)
class VeryLargeLossAnchor:
    """The very-large-loss share at one coordinate and how it enters the
    two-anchor law (``CompileOptions.very_large_loss_anchor``).

    ``share`` is the interpolated very-large-loss count law's mean over the
    interpolated count law's mean, so where it shapes the law the compiled
    model's expected annual number of incidents at or above ``level`` is that
    very-large-loss mean under the rank-integrated count.
    """

    level: float
    share: float | None
    role: VeryLargeLossRole

    @property
    def shapes_law(self) -> bool:
        return self.role in (
            "shapes_tail_above_large_loss_level",
            "shapes_tail_above_p99",
        )

    def record(self) -> dict[str, object]:
        return {
            "very_large_loss_level": self.level,
            "very_large_loss_share": self.share,
            "very_large_loss_role": self.role,
        }


def _very_large_loss_anchor(
    specification: SharedTotal,
    left: ContextualRiskAnswer,
    right: ContextualRiskAnswer,
    fraction: float,
    quantiles: SeverityQuantiles,
    computed_p99: float | None,
    first: LargeLossAnchor | None,
) -> VeryLargeLossAnchor | None:
    """The coordinate's very-large-loss share and its role in the two-anchor
    law, or None where the level is not asked or every anchor is ignored.

    The second point continues the tail from the last point the law already
    passes: the first anchor where it shapes the law, otherwise the law's own
    p99 where the first level lies within the stated quantiles, under the
    first anchor's own rule. Its coherence is judged at the coordinate, on
    the interpolated shares and quantiles. A share the law cannot take leaves
    the reference tail: collection retries only the first anchor's
    contradictions.
    """
    level = specification.very_large_loss_level
    counts = (
        left.rate,
        right.rate,
        left.very_large_loss_count,
        right.very_large_loss_count,
    )
    if first is None or level is None or any(value is None for value in counts):
        return None
    assert left.rate is not None and right.rate is not None
    assert left.very_large_loss_count is not None
    assert right.very_large_loss_count is not None
    share = quantile_large_loss_share(
        interpolated_rate(left.rate, right.rate, fraction),
        interpolated_rate(
            left.very_large_loss_count, right.very_large_loss_count, fraction
        ),
    )
    role: VeryLargeLossRole
    if share is None:
        role = "no_anchor"
    elif first.shapes_law:
        assert first.share is not None
        role = (
            "shapes_tail_above_large_loss_level"
            if very_large_loss_pin(level, share, (first.level, first.share)) is not None
            else "not_below_large_loss_share"
        )
    elif first.role in (
        "diagnostic_within_stated_quantiles",
        "skipped_at_or_below_computed_p99",
    ):
        if contradiction(level, share, quantiles.p95, quantiles.p99) is not None:
            role = "contradicts_stated_quantiles"
        elif large_loss_pin(level, share, computed_p99) is not None:
            role = "shapes_tail_above_p99"
        elif large_loss_pin(level, share, quantiles.p99) is not None:
            role = "skipped_at_or_below_computed_p99"
        else:
            role = "diagnostic_within_stated_quantiles"
    else:
        role = "large_loss_anchor_unresolved"
    return VeryLargeLossAnchor(level=level, share=share, role=role)


def _shared_total(
    specification: SharedTotal,
    left: ContextualRiskAnswer,
    right: ContextualRiskAnswer,
    fraction: float,
    large_loss_factor: float | None = 1.0,
    very_large_loss_anchor: bool = False,
) -> tuple[NativeSeverityLaw, LargeLossAnchor | None, VeryLargeLossAnchor | None]:
    """Retain one complete total; partition only after its floor and cap.

    The floor is inclusive: a shared total is the node's whole estimand
    (``ContextualRiskNode``).

    Where the large-loss anchor shapes the law, the tail above the law's own
    p99 passes through it (``Distribution.large_loss_amount``). With
    ``very_large_loss_anchor`` the tail also passes through the very-large-loss
    share where that is coherent (``_very_large_loss_anchor``): above the
    first anchor, or above the law's own p99 where the first level lies
    within the stated quantiles."""
    anchor, second = None, None
    try:
        if left.total_loss is None or right.total_loss is None:
            raise ContextualLawUnavailable(
                "The shared total lacks required anchor quantiles."
            )
        quantiles = SeverityQuantiles(
            **{
                key: interpolate(
                    getattr(left.total_loss, key),
                    getattr(right.total_loss, key),
                    fraction,
                )
                for key in (*_QUANTILES, "p99", "p999")
            }
        )
        base = NativeSeverityComponent(
            **{key: getattr(quantiles, key) for key in _QUANTILES},
            gpd_shape=specification.gpd_shape,
            gpd_tail_ratio=quantiles.p99 / quantiles.p95,
            severity_minimum=specification.floor,
            severity_cap=specification.cap,
        )
        computed_p99 = Distribution(**base.model_dump()).large_loss_break
        anchor = _large_loss_anchor(
            specification,
            left,
            right,
            fraction,
            quantiles,
            computed_p99,
            large_loss_factor,
        )
        component = base
        if anchor is not None and anchor.shapes_law:
            component = base.model_copy(
                update={
                    "large_loss_amount": anchor.level,
                    "large_loss_exceedance": anchor.share,
                }
            )
        if very_large_loss_anchor:
            second = _very_large_loss_anchor(
                specification, left, right, fraction, quantiles, computed_p99, anchor
            )
        if second is not None and second.shapes_law:
            # Above a shaping first anchor the second is the law's second
            # point; where the first level lies within the stated quantiles it
            # is the only point above the law's own p99.
            names = (
                ("very_large_loss_amount", "very_large_loss_exceedance")
                if second.role == "shapes_tail_above_large_loss_level"
                else ("large_loss_amount", "large_loss_exceedance")
            )
            component = component.model_copy(
                update=dict(zip(names, (second.level, second.share), strict=True))
            )
        law = NativeSeverityLaw(
            representation="distribution",
            components=(component,),
            weights=(1.0,),
        )
        lower, _ = severity_lower_support(
            native_severity_sampler(law), apply_severity_cap=True
        )
        if lower < specification.floor:
            raise ContextualLawUnavailable(
                "The shared total does not preserve its qualifying floor."
            )
        return law, anchor, second
    except ValueError as exc:
        raise ContextualLawUnavailable(
            f"The shared total has no admitted native law: {exc}"
        ) from exc


def _lower_cap(*caps: float | None) -> float | None:
    finite = [cap for cap in caps if cap is not None]
    return min(finite) if finite else None


def _lab_compensation_share(
    left: ContextualRiskAnswer, right: ContextualRiskAnswer, fraction: float
) -> float:
    """The interpolated share of the non-provider-borne amount the lab pays.

    A convex combination of two shares in [0, 1] stays in [0, 1], so the
    lab's compensation never exceeds the harm it compensates.
    """
    first, second = left.lab_compensation, right.lab_compensation
    if first is None or second is None:
        raise ContextualLawUnavailable(
            "A lab compensation answer is missing at an anchor."
        )
    for answer in (first, second):
        if answer.status != "available":
            raise ContextualLawUnavailable(
                answer.unavailable_reason or "Lab compensation is unavailable."
            )
    assert first.share is not None and second.share is not None
    return interpolate(first.share, second.share, fraction)


def _lab_annual_business_loss(
    instrument: ContextualRiskInstrument,
    node: ContextualRiskNode,
    left: ContextualRiskAnswer,
    right: ContextualRiskAnswer,
    fraction: float,
    frontier_lab_value: float | None,
) -> tuple[tuple[AnnualLabLossLaw, ...], dict[str, str], dict[str, object]]:
    """The lab's business loss in a year with at least one episode, capped.

    One combined amount for all the node's episodes in a year
    (``ACC-LAB-ANNUAL-BUSINESS-LOSS``), drawn once per simulated year whose
    count is at least one. The class endpoint bounds a whole event and the
    frontier-lab value what the lab can bear (``SEV-LAB-CAP``); the lower
    caps this annual amount, separately from each episode's provider-borne
    components plus compensation. An exact-zero law returns no annual law.
    """
    lab_id = instrument.lab.entity_id
    cap = _lower_cap(node.lab_loss_severity_cap, frontier_lab_value)
    try:
        first, second = left.lab_annual_business_loss, right.lab_annual_business_loss
        if first is None or second is None:
            raise ContextualLawUnavailable(
                "A lab business-loss answer is missing at an anchor."
            )
        for answer in (first, second):
            if answer.status != "available":
                raise ContextualLawUnavailable(
                    answer.unavailable_reason or "Lab business loss is unavailable."
                )
        assert (
            first.positive_probability is not None
            and second.positive_probability is not None
        )
        probability = interpolate(
            first.positive_probability, second.positive_probability, fraction
        )
        if probability == 0:
            return (), {}, {}
        if first.positive_probability == 0 or second.positive_probability == 0:
            raise ContextualLawUnavailable(
                "Positive interpolated lab business loss needs its conditional-positive law at both bounding anchors; a zero-loss anchor supplies no unused severity."
            )
        if (first.amount is None) != (second.amount is None):
            raise ContextualLawUnavailable(
                "Lab business loss changes amount-law family between anchors."
            )
        diagnostic: dict[str, object] = {}
        if first.amount is not None and second.amount is not None:
            amount = interpolate(first.amount, second.amount, fraction)
            if cap is not None:
                amount = min(amount, cap)
            if probability == 1:
                return (AnnualLabLossLaw(lab_id=lab_id, amount=amount),), {}, {}
            positive = NativeSeverityComponent(
                p5=amount, p25=amount, p50=amount, p75=amount, p95=amount
            )
        else:
            assert first.loss is not None and second.loss is not None
            shape = node.lab_loss_gpd_shape
            if shape is None:
                raise ContextualLawUnavailable(
                    "The node has no governed lab business-loss tail shape for conditional-positive quantiles."
                )
            quantiles = SeverityQuantiles(
                **{
                    key: interpolate(
                        getattr(first.loss, key), getattr(second.loss, key), fraction
                    )
                    for key in (*_QUANTILES, "p99", "p999")
                }
            )
            try:
                positive = NativeSeverityComponent(
                    **{key: getattr(quantiles, key) for key in _QUANTILES},
                    gpd_shape=shape,
                    gpd_tail_ratio=quantiles.p99 / quantiles.p95,
                    severity_minimum=0,
                    severity_cap=cap,
                )
                conditional = NativeSeverityLaw(
                    representation="distribution",
                    components=(positive,),
                    weights=(1.0,),
                )
                diagnostic = {
                    "governed_gpd_shape": shape,
                    "governed_severity_cap": cap,
                    "class_endpoint_cap": node.lab_loss_severity_cap,
                    "frontier_lab_value_cap": frontier_lab_value,
                    "interpolated_p999_implied_gpd_shape": p999_implied_gpd_shape(
                        quantiles.p95, quantiles.p99, quantiles.p999
                    ),
                    "elicited_conditional_positive_p999": quantiles.p999,
                    "native_conditional_positive_p999": native_severity_sampler(
                        conditional
                    ).quantile(0.999),
                    "p999_role": "diagnostic_only_not_a_tail_fit_target",
                    "tail_shape_origin": "governed_scenario_tail_shape",
                }
            except ValueError as exc:
                raise ContextualLawUnavailable(
                    f"The elicited lab business-loss conditional-positive law has no supported native representation: {exc}"
                ) from exc
        if probability == 1:
            law = NativeSeverityLaw(
                representation="distribution", components=(positive,), weights=(1.0,)
            )
        else:
            # The zero atom shares the positive law's qualifying minimum, as
            # every active mixture component must.
            zero = NativeSeverityComponent(
                p5=0,
                p25=0,
                p50=0,
                p75=0,
                p95=0,
                severity_minimum=positive.severity_minimum,
                severity_minimum_strict=positive.severity_minimum_strict,
            )
            law = NativeSeverityLaw(
                representation="mixture",
                components=(zero, positive),
                weights=(1 - probability, probability),
            )
        return ((AnnualLabLossLaw(lab_id=lab_id, severity=law),), {}, diagnostic)
    except ContextualLawUnavailable as exc:
        return (), {lab_id: f"Lab loss is unavailable: {exc}"}, {}


def _severity_scaled(
    components: tuple[ComponentAmountLaw, ...], factor: float
) -> tuple[ComponentAmountLaw, ...]:
    """Apply a severity factor to every component's raw loss.

    A severity law's raw amounts scale above its qualifying floor, ``L +
    factor*(X - L)`` (``with_excess_scale``), so the node's admission screen
    still holds and partition components keep one shared law; without a floor,
    and for a fixed amount, the scaling is proportional. Inside a mixture
    each active component's own cap moves with its amounts to ``L +
    factor*(cap - L)``; the outer cap stays nominal. A result outside the
    representable amounts leaves the law unavailable.
    """
    laws: dict[str, NativeSeverityLaw] = {}
    scaled = []
    try:
        for component in components:
            payload = component.model_dump(mode="python")
            if component.amount is not None:
                payload["amount"] = component.amount * factor
            else:
                assert component.severity is not None
                key = component.severity.model_dump_json()
                if key not in laws:
                    laws[key] = native_severity_from_sampler(
                        native_severity_sampler(component.severity).with_excess_scale(
                            factor
                        )
                    )
                payload["severity"] = laws[key]
            scaled.append(ComponentAmountLaw.model_validate(payload))
    except ValueError as exc:
        raise ContextualLawUnavailable(
            f"The severity factor {factor:g} leaves a loss law outside the supported range: {exc}"
        ) from exc
    return tuple(scaled)


def compile_contextual_node(
    instrument: ContextualRiskInstrument,
    node: ContextualRiskNode,
    case_answers: Mapping[float, ContextualRiskAnswer],
    score: float,
    *,
    rank_group: str | None = None,
    severity_factors: Mapping[float, float] | None = None,
    options: CompileOptions = CompileOptions(),
) -> IncidentLaw | PopulationAbsence:
    """Compile one fixed-score node, reusable by explicit path assemblers.

    Rate quantiles describe uncertainty in an expected count. Severity quantiles
    describe between-incident variation and become one native severity law.
    Interpolation acts on those parameters, never on annual draws or EP curves.
    The caller owns respondent selection and cross-node dependence.
    ``options`` (``CompileOptions``) holds the reference law's options or a
    comparison arm's change to them. ``options.frontier_lab_value`` caps each
    episode's lab loss and the annual business loss (``SEV-LAB-CAP``); only
    the ``lab_loss_cap`` sensitivity alternatives change it, and None removes
    it. Each episode's lab loss is its provider-borne components and the lab
    compensation share of its other components, capped once
    (``ACC-LAB-BEARER``); the law's annual lab loss is the lab's business
    loss, drawn once per year whose count is at least one
    (``ACC-LAB-ANNUAL-BUSINESS-LOSS``). ``options.lab_loss_provider_only``
    keeps only the provider-borne components, a lower bound on the lab's loss.
    ``severity_factors`` maps anchors to severity factors, interpolated like
    the other parameters; each component's raw loss above its qualifying
    floor is scaled before its cap (``_severity_scaled``), so provider-borne
    components and the compensation share scale with the same draw, while the
    separately elicited annual business loss, not a share of harm, is
    unchanged. No reference law carries one: the taxonomy parents' pools scale
    counts only (``HIER-CATEGORY-POOL``).

    The law's reference rate is the count law's mean over its parameter score,
    and the law carries that count law (``ParameterRankLaw``) to be read at a
    fresh normal score each simulated year, with ``options.rank_spread``
    multiplying the score and ``rank_group`` naming the score (the node by
    default): a joint state's laws sharing a group read the same score in a
    year, and each state reads its own (``NUM-STREAMS``). A
    count law that is zero at every rank is a structural-zero absence; one
    whose mean is not finite is unavailable.

    The count law carries ``options.spread_tail_form`` as its post-stretch
    ``tail_form``. ``options.count_factor`` scales each of the five count
    quantiles and the integrated reference rate; it leaves severity laws,
    branch probabilities and shares unchanged.

    ``options.unit_count`` compiles the same law at unit expected count, for
    a mean per expected count: the large-loss share still comes from the
    answers' own counts, so the law is the one the run uses.
    ``options.large_loss_factor`` scales the conditional odds of a shaping
    share for the ``large_loss_anchor`` comparison (``_large_loss_anchor``),
    and ``options.very_large_loss_anchor`` compiles the two-anchor tail for
    the ``very_large_loss_anchor`` comparison; the law then records how the
    second share entered it.
    """
    scenario_id = node.population.scenario_id
    if node.unavailable_reason is not None or node.count_family is None:
        return PopulationAbsence(
            scenario_id=scenario_id,
            disposition="unavailable",
            reason=node.unavailable_reason
            or "No native annual count law was authored.",
        )
    try:
        bracket = _anchor_bracket(case_answers, score, severity_factors)
        count_law, integrated = _count_law(bracket, options)
        if integrated == 0:
            return PopulationAbsence(
                scenario_id=scenario_id,
                disposition="structural_zero",
                reason="The conditional expected count is exactly zero at every parameter rank.",
            )
        if not bracket.left.branches or not bracket.right.branches:
            raise ContextualLawUnavailable(
                "A positive interpolated count requires marks at both bounding anchors."
            )
        compensation_share, compensation_unavailable = 0.0, None
        if not options.lab_loss_provider_only:
            try:
                compensation_share = _lab_compensation_share(
                    bracket.left, bracket.right, bracket.fraction
                )
            except ContextualLawUnavailable as exc:
                compensation_unavailable = f"Lab loss is unavailable: {exc}"
        total, large_loss, very_large_loss = (
            (None, None, None)
            if node.total is None
            else _shared_total(
                node.total,
                bracket.left,
                bracket.right,
                bracket.fraction,
                options.large_loss_factor,
                options.very_large_loss_anchor,
            )
        )
        branches = _branch_outcomes(
            instrument, node, bracket, total, compensation_share
        )
        lab = _lab_losses(instrument, node, bracket, options, compensation_unavailable)
        return _node_law(
            instrument,
            node,
            score,
            rank_group=rank_group,
            options=options,
            bracket=bracket,
            count_law=count_law,
            integrated=integrated,
            branches=branches,
            lab=lab,
            compensation_share=None
            if compensation_unavailable is not None
            else compensation_share,
            large_loss=large_loss,
            very_large_loss=very_large_loss,
        )
    except ContextualLawUnavailable as exc:
        return PopulationAbsence(
            scenario_id=scenario_id,
            disposition="unavailable",
            reason=f"Contextual node {node.node_id!r} at ECI {score:g}: {exc}",
        )


@dataclass(frozen=True)
class _Bracket:
    """A coordinate's bracketing anchors' answers, its fraction of the way
    from the lower to the upper and its interpolated severity factor."""

    left: ContextualRiskAnswer
    right: ContextualRiskAnswer
    fraction: float
    severity: float


def _anchor_bracket(
    case_answers: Mapping[float, ContextualRiskAnswer],
    score: float,
    severity_factors: Mapping[float, float] | None,
) -> _Bracket:
    """The answers at the anchors bracketing ``score``, each with an
    expected-count judgment, and the severity factor interpolated between
    them (one without factors)."""
    lower, upper, fraction = bracket_anchors(case_answers, score)
    left, right = case_answers[lower], case_answers[upper]
    severity = (
        1.0
        if severity_factors is None
        else interpolate(
            severity_factors.get(lower, 1.0),
            severity_factors.get(upper, 1.0),
            fraction,
        )
    )
    for answer in (left, right):
        if answer.status != "available" or answer.rate is None:
            raise ContextualLawUnavailable(
                answer.unavailable_reason
                or "A required anchor has no expected-count judgment."
            )
    return _Bracket(left, right, fraction, severity)


def _count_law(bracket: _Bracket, options: CompileOptions) -> tuple[Quantiles, float]:
    """The interpolated count law, scaled by ``options.count_factor``, and its
    mean over the parameter rank, one at unit count; refuses a mean that is
    not finite."""
    left, right = bracket.left.rate, bracket.right.rate
    assert left is not None and right is not None
    count_factor = options.count_factor
    count_law = scaled_rate(
        interpolated_rate(left, right, bracket.fraction), count_factor
    )
    integrated = (
        1.0
        if options.unit_count
        else integrated_count(
            left,
            right,
            bracket.fraction,
            options.epistemic_tail_scale,
            options.rank_spread,
            options.spread_tail_form,
            count_factor,
        )
    )
    if integrated is None:
        raise ContextualLawUnavailable(
            "The count law's mean over its parameter rank is not finite."
        )
    return count_law, integrated


def _branch_outcomes(
    instrument: ContextualRiskInstrument,
    node: ContextualRiskNode,
    bracket: _Bracket,
    total: NativeSeverityLaw | None,
    compensation_share: float,
) -> list[JointMarkOutcome]:
    """Each joint branch of positive interpolated probability as one member
    outcome: its components at the coordinate, partition shares conserving
    the shared total, scaled by the severity factor."""
    left, right, fraction = bracket.left, bracket.right, bracket.fraction
    scenario_id = node.population.scenario_id
    right_branches = {branch.branch_id: branch for branch in right.branches}
    definitions = {component.component_id: component for component in node.components}
    branches = []
    for branch in left.branches:
        other = right_branches[branch.branch_id]
        probability = interpolate(branch.probability, other.probability, fraction)
        if probability == 0:
            continue
        other_components = {
            component.component_id: component for component in other.components
        }
        shares = {
            component.component_id: interpolate(share, other_share, fraction)
            for component in branch.components
            if definitions[component.component_id].role == "partition"
            and (share := component.share) is not None
            and (other_share := other_components[component.component_id].share)
            is not None
        }
        if shares and len(shares) == sum(
            row.role == "partition" for row in definitions.values()
        ):
            if fraction not in (0, 1):
                # Evaluate the simplex identity at interior points, keeping
                # the authored endpoint shares unchanged.
                pivot = max(shares, key=lambda key: shares[key])
                shares[pivot] = math.fsum(
                    (
                        1.0,
                        *(-value for key, value in shares.items() if key != pivot),
                    )
                )
            if math.fsum(shares.values()) != 1 or any(
                not 0 <= value <= 1 for value in shares.values()
            ):
                raise ContextualLawUnavailable(
                    "Interpolated partition shares do not conserve the total."
                )
        components = tuple(
            _component(
                definitions[component.component_id],
                component,
                other_components[component.component_id],
                fraction,
                scenario_id=scenario_id,
                lab_id=instrument.lab.entity_id,
                total=total,
                total_rank=f"{node.node_id}::shared_total",
                partition_share=shares.get(component.component_id),
                compensation_share=compensation_share,
            )
            for component in branch.components
        )
        if bracket.severity != 1:
            components = _severity_scaled(components, bracket.severity)
        branches.append(
            JointMarkOutcome(probability=probability, components=components)
        )
    return branches


@dataclass(frozen=True)
class _LabLosses:
    """A node law's lab-loss terms: each episode's aggregate law, the annual
    business-loss law and, where the lab's loss is unavailable, why, with
    the business loss's tail diagnostic."""

    episode: tuple[AggregateLabLossLaw, ...]
    annual: tuple[AnnualLabLossLaw, ...]
    unavailable: dict[str, str]
    diagnostic: dict[str, object]


def _lab_losses(
    instrument: ContextualRiskInstrument,
    node: ContextualRiskNode,
    bracket: _Bracket,
    options: CompileOptions,
    compensation_unavailable: str | None,
) -> _LabLosses:
    """The node's lab-loss terms; an unavailable compensation share leaves
    the lab's whole loss unavailable."""
    lab_id = instrument.lab.entity_id
    # Each episode's provider-borne components and compensation are capped
    # once; the annual business loss carries its own cap.
    lab_losses: tuple[AggregateLabLossLaw, ...] = (
        AggregateLabLossLaw(
            lab_id=lab_id,
            amount=0,
            episode_cap=_lower_cap(
                node.lab_loss_severity_cap, options.frontier_lab_value
            ),
        ),
    )
    annual_lab_losses, unavailable_lab_losses, lab_diagnostic = (
        ((), {}, {})
        if options.lab_loss_provider_only
        else _lab_annual_business_loss(
            instrument,
            node,
            bracket.left,
            bracket.right,
            bracket.fraction,
            options.frontier_lab_value,
        )
    )
    if compensation_unavailable is not None:
        annual_lab_losses, unavailable_lab_losses, lab_diagnostic = (
            (),
            {lab_id: compensation_unavailable},
            {},
        )
    if unavailable_lab_losses:
        lab_losses = ()
    return _LabLosses(
        lab_losses, annual_lab_losses, unavailable_lab_losses, lab_diagnostic
    )


def _node_law(
    instrument: ContextualRiskInstrument,
    node: ContextualRiskNode,
    score: float,
    *,
    rank_group: str | None,
    options: CompileOptions,
    bracket: _Bracket,
    count_law: Quantiles,
    integrated: float,
    branches: list[JointMarkOutcome],
    lab: _LabLosses,
    compensation_share: float | None,
    large_loss: LargeLossAnchor | None,
    very_large_loss: VeryLargeLossAnchor | None,
) -> IncidentLaw:
    """The node's law: one episode member per qualified incident, its count
    law read at a fresh rank each year, and the compilation's record in its
    conditioning context. ``compensation_share`` is None when unavailable.
    Only a law compiled for the two-anchor comparison records the
    very-large-loss share, so a reference law's context, and identity, do not
    depend on it."""
    assert node.count_family is not None
    # Every episode has one member: its components are drawn once.
    cluster = ClusterCount(
        family="finite", support=(UnitCountMass(count=1, probability=1),)
    )
    return IncidentLaw(
        origin_id=f"{instrument.context_id}::{node.node_id}",
        scenario_id=node.population.scenario_id,
        entity_id=instrument.model.entity_id,
        reference_annual_rate=integrated,
        parameter_rank=None
        if options.unit_count
        else ParameterRankLaw(
            group_id=rank_group or node.node_id,
            count=count_law,
            epistemic_tail_scale=options.epistemic_tail_scale,
            rank_spread=options.rank_spread,
            tail_form=options.spread_tail_form,
        ),
        count_family=node.count_family,
        count_variance_to_mean=node.count_variance_to_mean,
        response=LocalResponse(
            domain="local_entity_state",
            interpolation="linear",
            extrapolation="unavailable",
            points=(ResponsePoint(value=score, multiplier=1),),
            held_fixed="; ".join(
                (
                    instrument.exposure,
                    instrument.fixed_controls,
                    instrument.fixed_adoption,
                    instrument.locality_assumption,
                )
            ),
        ),
        segments=(
            IncidentSegment(
                start=0,
                end=1,
                reference_exposure_share=1,
                initiation_probability=1,
                kernel=MarkKernel(
                    cluster=cluster,
                    campaign_components=(),
                    campaign_lab_losses=lab.episode,
                    shared_probability=1,
                    unit_probability=1,
                    member_outcomes=tuple(branches),
                    gate_meaning=node.qualified_count_definition,
                    selection_assumption=instrument.exposure,
                    displacement_assumption=instrument.fixed_adoption,
                ),
            ),
        ),
        input_stage="qualified_initiator",
        qualification_assumption=node.qualified_count_definition,
        unavailable_lab_losses=lab.unavailable,
        annual_lab_losses=lab.annual,
        conditioning_context={
            "context_id": instrument.context_id,
            "node_id": node.node_id,
            "coordinate_kind": "constant_path",
            "score": score,
            "rank_spread": options.rank_spread,
            "count_factor": options.count_factor,
            "severity_factor": bracket.severity,
            "epistemic_method": "pchip_local_tails",
            "epistemic_tail_scale": options.epistemic_tail_scale,
            "parameter_interpolation": instrument.interpolation,
            "instrument_sha256": instrument.digest(),
            "lab_loss_method": "episode_provider_components_plus_compensation_share_plus_annual_business_loss",
            "provider_borne_component_ids": [
                component.component_id
                for component in node.components
                if component.bearer == "provider"
            ],
            "lab_compensation_share": compensation_share,
            "lab_loss_provider_only": options.lab_loss_provider_only,
            "lab_compensation_definition": instrument.lab_compensation_definition,
            "lab_annual_business_loss_definition": instrument.lab_annual_business_loss_definition,
            "lab_loss_scope": instrument.lab_loss_scope,
            "lab_annual_business_loss_dependence": instrument.lab_annual_business_loss_dependence,
            "lab_annual_business_loss_tail_diagnostic": lab.diagnostic,
            "lab_loss_frontier_lab_value": options.frontier_lab_value,
            "qualified_count_mean": integrated,
            "shared_total": None
            if node.total is None
            else node.total.model_dump(mode="json"),
            "shared_total_qualification": "The qualified count applies to the shared estimand total before any additive costs; additive costs do not establish eligibility.",
            "large_loss_anchor": None if large_loss is None else large_loss.record(),
            **(
                {
                    "very_large_loss_anchor": None
                    if very_large_loss is None
                    else very_large_loss.record()
                }
                if options.very_large_loss_anchor
                else {}
            ),
            "lab_loss_unavailable_reasons": lab.unavailable,
        },
    )


def compile_once(
    compile_law: Callable[..., IncidentLaw | PopulationAbsence],
) -> Callable[..., IncidentLaw | PopulationAbsence]:
    """``compile_law`` compiling each distinct request once.

    The exhibits ask one node and state for the same unit-count laws more than
    once: its run law, and its law without a severity factor. A request is the
    same when it names the same answer and severity-factor objects, which the
    state's selection holds for as long as the returned function is used, and
    the same options. A request that raises is not kept.
    """
    compiled: dict[tuple, IncidentLaw | PopulationAbsence] = {}

    def once(
        answers: Mapping[float, ContextualRiskAnswer],
        severity_factors: Mapping[float, float] | None,
        **options: object,
    ) -> IncidentLaw | PopulationAbsence:
        key = (
            tuple((anchor, id(answer)) for anchor, answer in sorted(answers.items())),
            None if severity_factors is None else id(severity_factors),
            tuple(sorted(options.items())),
        )
        if key not in compiled:
            compiled[key] = compile_law(answers, severity_factors, **options)
        return compiled[key]

    return once


def shared_total_sampler(law: IncidentLaw | PopulationAbsence):
    """The shared total's severity law of a compiled node, or None."""
    if not isinstance(law, IncidentLaw):
        return None
    for outcome in law.segments[0].kernel.member_outcomes:
        for component in outcome.components:
            if (
                component.severity is not None
                and component.rank_group_id is not None
                and component.rank_group_id.endswith("::shared_total")
            ):
                sampler = native_severity_sampler(component.severity)
                return sampler if isinstance(sampler, Distribution) else None
    return None


def tail_measures(total: Distribution | None, level: float, very: float | None):
    """Survival at both levels and the gross mean in total and above ``level``.

    The gross mean of a shared total is the total's mean, since its partition
    shares sum to one; E[X; X >= A] is E[min(X, C)] - E[min(X, A)] + A S(A).
    None marks a law without a finite mean.
    """
    if total is None:
        return None
    try:
        mean = total.analytic_mean()
        below = total.analytic_mean(severity_cap_override=level)
    except ValueError:
        return None
    survival = total.survival_probability(level)
    # The strict survival is the mass at or above a continuous level.
    return {
        "large": survival,
        "very_large": None if very is None else total.survival_probability(very),
        "gross": mean,
        "gross_large": mean - below + level * survival,
    }


def contextual_scope(
    instrument: ContextualRiskInstrument,
    *,
    scope_id: str,
    title: str,
    populations: tuple[Population, ...],
    entities: tuple[Entity, ...] | None = None,
    deployments: tuple[DeploymentBinding, ...] | None = None,
) -> RunScope:
    period = instrument.period
    return RunScope(
        scope_id=scope_id,
        title=title,
        period_start=period.policy_period_start_utc,
        period_end=period.policy_period_end_utc,
        forecast_as_of=period.forecast_as_of,
        data_cutoff=period.data_cutoff_utc,
        availability_cutoff=period.availability_cutoff_utc,
        information_date=period.information_date_utc,
        capability_scales=(instrument.capability_scale,),
        deployments=(
            (instrument.deployment_binding,) if deployments is None else deployments
        ),
        entities=(
            (instrument.lab, instrument.model, instrument.deployment)
            if entities is None
            else entities
        ),
        populations=populations,
    )


def contextual_independence(instrument: ContextualRiskInstrument) -> tuple[str, ...]:
    return (
        instrument.independent_node_origins_rationale
        or "One named node; no cross-node aggregation independence is asserted.",
        "Within each node the captured joint branch and rank-group coupling is retained.",
        LAB_LOSS_DEPENDENCE,
        "Respondent opinions receive equal weight, except that a sample excluded for an unanswered leaf leaves its weight to its model's remaining samples (POOL-EXCLUDED-SAMPLES). Rate-parameter ranks are independent between nodes and shared across all anchors within one node.",
    )


def contextual_limitations(instrument: ContextualRiskInstrument) -> tuple[str, ...]:
    return (
        *hypothetical_entity_limitations(
            (instrument.lab, instrument.model, instrument.deployment)
        ),
        instrument.locality_assumption,
        *threat_model_limitations(instrument),
        "Branch probabilities, component shares and the lab compensation share are point judgments per respondent and anchor.",
        "Constant-path conditional judgments; no future capability path or forecast prior is inferred.",
        "Linear interpolation applies to the captured expected-count quantiles, joint-branch probabilities and component-law parameters. Severity p999 remains a diagnostic; p95/p99 and the authored shape determine the GPD tail.",
        instrument.lab_compensation_definition,
        instrument.lab_annual_business_loss_definition,
        LAB_LOSS_TAIL_LIMITATION,
        "Native incident simulation retains the separately elicited lab business-loss law. Frozen work allowances per simulated year refuse a run whose individually drawn events or campaign members exceed them.",
    )


def constant_path(instrument: ContextualRiskInstrument, score: float) -> EntityPath:
    return EntityPath(
        entity_id=instrument.model.entity_id,
        scale_id=instrument.capability_scale.scale_id,
        history_status="stipulated",
        history_rationale="Captured constant-path query at the exact named coordinate.",
        interpolation="step",
        knots=(
            PathKnot(time=0, value=score, model_revision=instrument.model.revision),
            PathKnot(time=1, value=score, model_revision=instrument.model.revision),
        ),
    )


def admitted_contextual_node(
    instrument: ContextualRiskInstrument,
    node: ContextualRiskNode,
    answers: Mapping[float, ContextualRiskAnswer],
    score: float,
    *,
    severity_factors: Mapping[float, float] | None = None,
    options: CompileOptions = CompileOptions(),
) -> IncidentLaw | PopulationAbsence:
    """One node's law at a fixed coordinate after the run's own admission
    checks, so a value computed from it never comes from a law a run would
    refuse."""
    return admit_node(
        instrument,
        node,
        constant_path(instrument, score),
        compile_contextual_node(
            instrument,
            node,
            answers,
            score,
            severity_factors=severity_factors,
            options=options,
        ),
    )


_ADMISSION_PROBE_SLOTS = 256
_ADMISSION_PROBES: dict[
    tuple[int, int], tuple[ContextualRiskInstrument, ContextualRiskNode, RunSetup]
] = {}
"""Recent admission setups, by the identities of their instrument and node;
each entry keeps both alive, so an identity is never reused while it is kept."""


def _admission_probe(
    instrument: ContextualRiskInstrument,
    node: ContextualRiskNode,
    state: JointState,
) -> RunSetup:
    """The setup a law is admitted in: the node's population in the
    instrument's scope, holding ``state`` alone.

    All but the state depends only on the instrument and the node, both
    frozen. It is built and validated for the first law admitted at a node,
    and each later law's setup is that one with its own state
    (``RunSetup.with_joint_states``), which validates the state and checks
    it against the rest as building the whole setup would.
    """
    key = id(instrument), id(node)
    kept = _ADMISSION_PROBES.get(key)
    if kept is not None:
        return kept[2].with_joint_states((state,))
    probe = RunSetup(
        setup_id="contextual-admission",
        scope=contextual_scope(
            instrument,
            scope_id="contextual-admission",
            title="Contextual node support admission",
            populations=(node.population,),
        ),
        evidence=EvidenceSet(),
        joint_states=(state,),
        assumptions=AssumptionSet(independence=contextual_independence(instrument)),
        execution=ExecutionPlan(
            plan_id="contextual-admission",
            required_arms=("reference",),
            simulations=1,
            seed=0,
            requested_views=("gross",),
        ),
        limitations=contextual_limitations(instrument),
    )
    if len(_ADMISSION_PROBES) >= _ADMISSION_PROBE_SLOTS:
        _ADMISSION_PROBES.pop(next(iter(_ADMISSION_PROBES)))
    _ADMISSION_PROBES[key] = (instrument, node, probe)
    return probe


_INCIDENT_LAW = TypeAdapter(IncidentLaw)


def admit_node(
    instrument: ContextualRiskInstrument,
    node: ContextualRiskNode,
    path: EntityPath,
    law: IncidentLaw | PopulationAbsence,
) -> IncidentLaw | PopulationAbsence:
    """``law`` as a run would hold it, or the absence that takes its place.

    A node with a qualification threshold has its law checked as a run checks
    it (``check_incident_state``), in a setup holding the law alone; a law
    those checks refuse becomes an unavailable absence, not an error.

    The law returned is validated: the setup validates the law it is given
    (``RunSetup``), and a node without a threshold has its law validated
    here. So ``law`` may be an unvalidated copy, as conditioning a compiled
    law on its trajectory makes, and is validated once, not once by whoever
    built it and again by the setup.
    """
    if isinstance(law, PopulationAbsence):
        return law
    if node.population.threshold is None:
        return _INCIDENT_LAW.validate_python(
            _INCIDENT_LAW.dump_python(law, mode="python", warnings=False)
        )
    # Reuse the native support and qualification checks; a node they refuse
    # becomes an unavailable absence rather than an error.
    try:
        probe = _admission_probe(
            instrument,
            node,
            # Constructed without validation: the setup validates its states.
            JointState.model_construct(
                state_id="admission",
                epistemic_group_id="admission",
                weight=1,
                entity_paths=(path,),
                laws=(law,),
            ),
        )
        check_incident_state(probe, "admission")
    except ValueError as exc:
        if not any(
            message in str(exc)
            for message in (
                "surviving terminal-total support contradicts qualification",
                "qualifying terminal-total support is unidentified",
            )
        ):
            raise
        return PopulationAbsence(
            scenario_id=node.population.scenario_id,
            disposition="unavailable",
            reason=f"Contextual node {node.node_id!r} lacks admitted native support: {exc}",
        )
    return probe.joint_states[0].laws[0]


@dataclass(frozen=True)
class _LeafQuery:
    """A cache key for one leaf's per-count welfare mean; the instrument is
    carried by reference and identified by its digest, and the frozen answer
    is carried itself."""

    digest: str
    node_id: str
    anchor_score: float
    answer: ContextualRiskAnswer
    severity: float
    instrument: ContextualRiskInstrument = field(compare=False, hash=False)


@dataclass(frozen=True)
class UnitLaw:
    """Analytic means of a leaf's law compiled at unit expected count.

    ``welfare`` and ``gross`` are the views' means per expected unit; None
    without a finite mean. ``total_mean`` is the shared total's gross mean
    per unit, and ``gross_share_at_or_above_large_loss`` the share of it from
    units at or above the total's large-loss level; None without a shared
    total, a level or a positive finite mean.
    """

    welfare: float | None
    gross: float | None
    total_mean: float | None
    gross_share_at_or_above_large_loss: float | None


UNIT_COUNT = CompileOptions(unit_count=True)
"""The reference options at unit expected count, for a law's mean per
expected count."""


_UNIT_LAW_SLOTS = 65536
_UNIT_LAWS: dict[_LeafQuery, UnitLaw | None] = {}
"""The unit laws this process has worked out or been given, oldest first."""


def unit_law(query: _LeafQuery) -> UnitLaw | None:
    """A leaf's law compiled and admitted, as a run admits it, at one
    coordinate with a severity factor and unit expected count; None without
    an admitted law. Severity and cluster laws do not depend on the count.

    Each is worked out once and kept (``_UNIT_LAWS``): checking a collection
    asks for every case's, and so do its taxonomy-parent outcomes and
    summaries. A process that reads a collection for another hands its unit
    laws on with it (``unit_laws_of`` and ``keep_unit_laws``).
    """
    try:
        return _UNIT_LAWS[query]
    except KeyError:
        pass
    law = _unit_law(query)
    if len(_UNIT_LAWS) >= _UNIT_LAW_SLOTS:
        _UNIT_LAWS.pop(next(iter(_UNIT_LAWS)))
    _UNIT_LAWS[query] = law
    return law


def unit_laws_of(
    instrument: ContextualRiskInstrument,
) -> dict[_LeafQuery, UnitLaw | None]:
    """Every unit law this process holds for ``instrument``'s leaves."""
    digest = instrument.digest()
    return {query: law for query, law in _UNIT_LAWS.items() if query.digest == digest}


def keep_unit_laws(laws: Mapping[_LeafQuery, UnitLaw | None]) -> None:
    """Keep unit laws another process worked out, as if this one had. They
    are whatever ``unit_law`` returned there for the same queries, so a
    caller passes only laws from a process running this code."""
    for query, law in laws.items():
        if query not in _UNIT_LAWS and len(_UNIT_LAWS) >= _UNIT_LAW_SLOTS:
            _UNIT_LAWS.pop(next(iter(_UNIT_LAWS)))
        _UNIT_LAWS[query] = law


def _unit_law(query: _LeafQuery) -> UnitLaw | None:
    answer = query.answer
    node = next(
        row
        for row in query.instrument.collectible_nodes
        if row.node_id == query.node_id
    )
    law = admitted_contextual_node(
        query.instrument,
        node,
        {query.anchor_score: answer},
        query.anchor_score,
        severity_factors={query.anchor_score: query.severity},
        options=UNIT_COUNT,
    )
    if not isinstance(law, IncidentLaw):
        return None
    total = node.total
    sampler = shared_total_sampler(law)
    total_mean = share = None
    if sampler is not None:
        try:
            total_mean = sampler.analytic_mean()
        except ValueError:
            total_mean = None
    if sampler is not None and total is not None and total.large_loss_level is not None:
        measures = tail_measures(sampler, total.large_loss_level, None)
        if measures is not None and measures["gross"] > 0:
            share = measures["gross_large"] / measures["gross"]
    # Per incident: the lab's annual business loss, which gross counts once
    # per year with an incident, is not an incident's loss.
    means = single_period_law_means(law, ("welfare", "gross"), annual=False)
    return UnitLaw(
        welfare=means.get("welfare"),
        gross=means.get("gross"),
        total_mean=total_mean,
        gross_share_at_or_above_large_loss=share,
    )


def leaf_query(
    instrument: ContextualRiskInstrument,
    node_id: str,
    answer: ContextualRiskAnswer,
    anchor_score: float,
    severity: float = 1.0,
) -> _LeafQuery:
    return _LeafQuery(
        instrument.digest(),
        node_id,
        anchor_score,
        answer,
        severity,
        instrument,
    )


LEAF_FACTOR_RANGE = (1e-4, 1e4)
"""The range a leaf's cumulative factor is clamped to
(``HIER-CATEGORY-POOL``), a judgment: a leaf moved by more than four
orders of magnitude would mean that a category and what is beneath it differ
in definition, not in estimate, and the range keeps every scaled amount
representable."""


def _bounded(factor: float) -> float:
    """A cumulative factor clamped to ``LEAF_FACTOR_RANGE``. The product is
    computed, so one within ``TIE_RELATIVE_TOLERANCE`` of a bound is a tie
    and stands."""
    lower, upper = LEAF_FACTOR_RANGE
    if factor > upper * (1 + TIE_RELATIVE_TOLERANCE):
        return upper
    if factor < lower * (1 - TIE_RELATIVE_TOLERANCE):
        return lower
    return factor


@dataclass(frozen=True)
class LeafFactors:
    """A leaf's factors: a frequency factor on its expected count and a
    severity factor on its loss amounts before the outer cap. The taxonomy
    parents' pools set the frequency factor and leave the severity factor at
    one (``HIER-CATEGORY-POOL``)."""

    frequency: float = 1.0
    severity: float = 1.0

    def bounded(self) -> LeafFactors:
        """Each cumulative factor clamped to ``LEAF_FACTOR_RANGE``."""
        return LeafFactors(_bounded(self.frequency), _bounded(self.severity))


UNSCALED = LeafFactors()


def leaf_eal(
    instrument: ContextualRiskInstrument,
    node_id: str,
    answer: ContextualRiskAnswer,
    anchor_score: float,
    factors: LeafFactors = UNSCALED,
    *,
    options: CompileOptions = CompileOptions(),
) -> tuple[float | None, float | None]:
    """A leaf's (median, mean) expected annual welfare at one coordinate,
    with ``factors`` applied. None marks an unavailable answer, law or
    finite mean. The mean integrates the count over its parameter rank at
    ``options``'s spread (``epistemic_tail_scale``, ``rank_spread`` and
    ``spread_tail_form``), and ``options.count_factor`` scales the count, so
    the median and the mean alike; the per-count law is the reference law at
    unit count (``unit_law``)."""
    if answer.status != "available" or answer.rate is None:
        return None, None
    rate = answer.rate
    if rate.p95 == 0:
        return 0.0, 0.0
    law = unit_law(
        leaf_query(instrument, node_id, answer, anchor_score, factors.severity)
    )
    per_count = None if law is None else law.welfare
    if per_count is None:
        return (0.0 if rate.p50 == 0 else None), None
    try:
        mean = epistemic_mean(
            rate,
            spread=options.rank_spread,
            epistemic_tail_scale=options.epistemic_tail_scale,
            tail_form=options.spread_tail_form,
        )
    except (OverflowError, ValueError):
        mean = None
    scale = factors.frequency * per_count * options.count_factor
    return rate.p50 * scale, None if mean is None else mean * scale


def leaf_eals(
    instrument: ContextualRiskInstrument,
    answers: Mapping[str, ContextualRiskAnswer],
    anchor_score: float,
    *,
    options: CompileOptions = CompileOptions(),
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Each node's (median, mean) expected annual welfare at one coordinate,
    before any taxonomy parent's pool (``leaf_eal``)."""
    rows = {
        node_id: leaf_eal(instrument, node_id, answer, anchor_score, options=options)
        for node_id, answer in answers.items()
    }
    return (
        {node_id: row[0] for node_id, row in rows.items()},
        {node_id: row[1] for node_id, row in rows.items()},
    )
