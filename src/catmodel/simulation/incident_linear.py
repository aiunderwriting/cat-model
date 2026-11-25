"""Event-addressed native marks for single-member incidents.

A single-member kernel emits at most one member per initiating event: one
conditional outcome behind optional shared, unit and {0, 1} cluster gates, with
no campaign components. Each event's member gate, component ranks, aggregate lab
amounts and native payment uniforms are addressed by state, origin, purpose,
year and event ordinal (``event_uniforms``); the caller addresses segment choice
the same way. Native severity inverses and authored rank groups define the
joint law, and nothing is discretized.

Linear views reduce these draws as arrays, and the covered amounts of their
policy memberships enter the contract terms as arrays too
(``catmodel.simulation.policy_terms``). Requested views and batching therefore never
change an event's draw.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from catmodel.contracts.components import AggregateLabLossLaw, ComponentAmountLaw
from catmodel.contracts.incident import (
    JointMarkOutcome,
    MarkKernel,
    parse_scenario_view,
)
from catmodel.contracts.native_severity import shared_native_severity_sampler
from catmodel.probability.distributions import (
    SeveritySampler,
    sample_affine_capped_severity,
)
from catmodel.probability.keyed_streams import event_uniforms
from catmodel.probability.liability import LiabilityAttenuation
from catmodel.simulation.authored.payments import constant_native_payment_fraction
from catmodel.simulation.components import component_view_weight
from catmodel.simulation.incident import source_status_view

LINEAR_CATEGORIES = frozenset(
    {
        "gross",
        "oep_gross",
        "welfare",
        "net_resource_cost",
        "provider_contribution",
        "lab_loss",
        "bearer",
        "contributor",
        "attribution",
    }
)


@dataclass(frozen=True, slots=True)
class SingleMemberLaw:
    """Prepared native laws of a kernel with at most one member per event."""

    outcome: JointMarkOutcome
    member_probability: float
    labs: tuple[AggregateLabLossLaw, ...]
    samplers: tuple[SeveritySampler | None, ...]
    lab_samplers: tuple[SeveritySampler | None, ...]


@dataclass(frozen=True, slots=True)
class EventCell:
    """One small-event cell: its address prefix, member probability and the
    rank intervals its gross and aggregate-lab uniforms are mapped into."""

    purpose_prefix: str
    member_probability: float
    rank_interval: tuple[float, float]
    lab_interval: tuple[float, float]


@dataclass(frozen=True, slots=True)
class EventDraws:
    """Aligned per-event draws; member amounts are zero for gated-out events."""

    member: np.ndarray
    amounts: tuple[np.ndarray, ...]
    labs: dict[str, np.ndarray]
    payment_fraction: np.ndarray | None
    component_totals: np.ndarray


@dataclass(frozen=True, slots=True)
class LinearProjection:
    """Exact component coefficients for each requested linear view.

    ``direct_labs`` adds an event's aggregate lab amount to a lab view's
    component allocations and caps their sum at the episode cap, if any.
    """

    weights: dict[str, tuple[float, ...]]
    direct_labs: dict[str, tuple[str, float | None]]

    def touched(self) -> "LinearProjection":
        """This projection on the views an event can make nonzero: a nonzero
        coefficient or a direct lab amount. Every other view is zero for every
        event, so reducing it would add +0.0 to annual sums that start at +0.0
        and change none of them."""
        return LinearProjection(
            {
                view: weights
                for view, weights in self.weights.items()
                if any(weights) or view in self.direct_labs
            },
            self.direct_labs,
        )


def _without_campaign_terms(spec) -> bool:
    return not (
        spec.campaign_components
        or spec.campaign_obligations
        or spec.campaign_payments
        or spec.campaign_memberships
        or spec.campaign_payment_kernel is not None
    )


def _sampler(law) -> SeveritySampler | None:
    return (
        None if law.severity is None else shared_native_severity_sampler(law.severity)
    )


def single_member_law(kernel: MarkKernel) -> SingleMemberLaw | None:
    """Identify a kernel whose member law reduces to one Bernoulli gate.

    Shared, unit and {0, 1} cluster gates compose into one member probability.
    Kernels that can emit several members, choose among outcomes or carry
    campaign terms return ``None``; ``incident_campaigns`` draws their episodes.
    """
    if not _without_campaign_terms(kernel):
        return None
    if kernel.explicit_joint_members:
        branches = [row for row in kernel.explicit_joint_members if row.probability > 0]
        if (
            len(branches) != 1
            or len(branches[0].members) != 1
            or not _without_campaign_terms(branches[0])
        ):
            return None
        outcome, cluster_probability = branches[0].members[0], 1.0
    else:
        cluster = kernel.cluster
        outcomes = kernel.possible_member_outcomes()
        if (
            cluster is None
            or any(row.count > 1 and row.probability > 0 for row in cluster.support)
            or len(outcomes) != 1
        ):
            return None
        outcome = outcomes[0]
        cluster_probability = math.fsum(
            row.probability for row in cluster.support if row.count == 1
        )
    if (
        not outcome.overlap_resolved
        or outcome.identified_gross_total is not None
        or not outcome.composition_complete
    ):
        return None
    return SingleMemberLaw(
        outcome,
        kernel.shared_probability * kernel.unit_probability * cluster_probability,
        kernel.campaign_lab_losses,
        tuple(_sampler(law) for law in outcome.components),
        tuple(_sampler(law) for law in kernel.campaign_lab_losses),
    )


def _measure(view):
    status_scope = source_status_view(view)
    measure = view if status_scope is None else status_scope[1]
    if measure.startswith("source_status::"):
        return None
    return parse_scenario_view(measure) or (None, measure)


def linear_projection(
    law: SingleMemberLaw,
    views: Sequence[str],
    *,
    scenario_id: str | None = None,
    source_public_status: str | None = None,
) -> LinearProjection | None:
    """Return exact view coefficients, or ``None`` if a view is not linear here.

    A source-status view of another status receives zero coefficients, since
    this segment's events never enter it.
    """
    return mark_projection(
        law.outcome.components,
        law.outcome.scenario_ids,
        law.labs,
        views,
        scenario_id=scenario_id,
        source_public_status=source_public_status,
    )


def mark_projection(
    components: Sequence[ComponentAmountLaw],
    scenario_ids: Sequence[str],
    labs: Sequence[AggregateLabLossLaw],
    views: Sequence[str],
    *,
    scenario_id: str | None = None,
    source_public_status: str | None = None,
) -> LinearProjection | None:
    """Exact view coefficients of one mark's ``components``, whose scenario
    scope defaults to ``scenario_ids`` and then ``scenario_id``, with the
    episode's aggregate ``labs`` added to their own lab views; ``None`` if a
    view is not linear here (``linear_projection``)."""
    direct_labs = {lab.lab_id: lab.episode_cap for lab in labs}
    weights, direct = {}, {}
    for view in views:
        parsed = _measure(view)
        if parsed is None:
            return None
        scenario, measure = parsed
        category, _, identifier = measure.partition("::")
        if category not in LINEAR_CATEGORIES:
            return None
        status_scope = source_status_view(view)
        if status_scope is not None and status_scope[0] != source_public_status:
            weights[view] = tuple(0.0 for _ in components)
            continue
        if category == "lab_loss" and identifier in direct_labs:
            direct[view] = identifier, direct_labs[identifier]
        row = []
        for component in components:
            if scenario is not None:
                scope = component.scenario_ids or tuple(scenario_ids)
                if not scope:
                    if scenario_id is None:
                        return None
                    scope = (scenario_id,)
                # Physical scenario projection is identified only for a unique
                # allocation. Positive-support admission checks ambiguous sets.
                if scope != (scenario,):
                    row.append(0.0)
                    continue
            try:
                row.append(
                    component_view_weight(
                        component, "gross" if category == "oep_gross" else measure
                    )
                )
            except ValueError:
                return None
        weights[view] = tuple(row)
    return LinearProjection(weights, direct)


def rank_purpose(scope: str, group: str | None) -> str:
    """The event-address purpose of a rank group's uniform in one scope."""
    return "physical-rank::" + json.dumps((scope, group), separators=(",", ":"))


def draw_single_member_events(
    law: SingleMemberLaw,
    years: np.ndarray,
    event_ordinals: np.ndarray,
    *,
    root_seed: int,
    state_id: str,
    origin_id: str,
    payments: bool = False,
    cell: EventCell | None = None,
) -> EventDraws:
    """Draw one bounded batch of events from native inverse CDFs and exact ranks.

    Every uniform is addressed by the event's year and ordinal, so neither
    selection, batching nor the requested views changes a rank. Gated-out
    events never evaluate member severities. ``payments`` adds the native
    payment proxy's liability fraction for member events. ``cell`` draws the
    events of one small-event cell (``catmodel.simulation.small_events``): its own
    addresses, its member probability and its rank intervals.
    """
    years = np.asarray(years)
    ordinals = np.asarray(event_ordinals)
    if years.ndim != 1 or years.shape != ordinals.shape:
        raise ValueError("event coordinates must be aligned one-dimensional arrays")
    count = len(years)

    prefix = "" if cell is None else cell.purpose_prefix

    def addresses(selected_years, selected_ordinals):
        cache = {}

        def uniforms(purpose):
            if purpose not in cache:
                cache[purpose] = event_uniforms(
                    selected_years,
                    selected_ordinals,
                    root_seed=root_seed,
                    state_id=state_id,
                    origin_id=origin_id,
                    purpose=prefix + purpose,
                )
            return cache[purpose]

        return uniforms

    def interval(uniforms, bounds):
        low, high = bounds
        if bounds == (0.0, 1.0):
            return uniforms
        # Keep mapped ranks strictly inside the cell after rounding.
        return np.minimum(low + (high - low) * uniforms, np.nextafter(high, low))

    rank_bounds = (0.0, 1.0) if cell is None else cell.rank_interval
    lab_bounds = (0.0, 1.0) if cell is None else cell.lab_interval
    member_probability = (
        law.member_probability if cell is None else cell.member_probability
    )
    every = addresses(years, ordinals)
    if member_probability >= 1:
        member = np.ones(count, dtype=bool)
        index, members = None, every
    else:
        member = every("member-gate") < member_probability
        index = np.flatnonzero(member)
        members = addresses(years[index], ordinals[index])
    size = count if index is None else len(index)
    inverses = {}

    def inverse(sampler, selected):
        # Distinct components can share the same native inverse even when
        # their outer scaling differs.
        key = id(sampler), selected.tobytes()
        if key not in inverses:
            inverses[key] = sampler.sample_from_uniforms_uncapped(selected)
        return inverses[key]

    def native(sampler, uniforms, *, multiplier=1.0, post_multiplier=1.0):
        return sample_affine_capped_severity(
            sampler,
            uniforms,
            scale=multiplier,
            post_multiplier=post_multiplier,
            _raw_inverse=inverse,
        )

    def scatter(values):
        if index is None:
            return values
        full = np.zeros(count)
        full[index] = values
        return full

    amounts = []
    component_totals = np.zeros(count)
    for component, sampler in zip(law.outcome.components, law.samplers, strict=True):
        if component.multiplier == 0 or component.post_multiplier == 0:
            sampled = np.zeros(size)
        elif component.amount is not None:
            sampled = np.full(
                size,
                component.amount * component.multiplier * component.post_multiplier,
            )
        else:
            assert sampler is not None
            sampled = native(
                sampler,
                interval(
                    members(
                        rank_purpose(component.rank_scope, component.rank_group_id)
                    ),
                    rank_bounds,
                ),
                multiplier=component.multiplier,
                post_multiplier=component.post_multiplier,
            )
        sampled = scatter(sampled)
        amounts.append(sampled)
        component_totals += sampled
    labs = {}
    for lab, sampler in zip(law.labs, law.lab_samplers, strict=True):
        if lab.amount is not None:
            labs[lab.lab_id] = np.full(count, lab.amount)
        else:
            assert sampler is not None
            labs[lab.lab_id] = native(
                sampler,
                interval(every("aggregate-lab-mark::" + lab.lab_id), lab_bounds),
            )
    fraction = None
    kernel = law.outcome.payment_kernel
    if payments and kernel is not None:
        constant = constant_native_payment_fraction(kernel)
        if constant is not None:
            fraction = np.full(count, constant)
        else:
            assert kernel.liability is not None
            severity = native(
                shared_native_severity_sampler(kernel.source_severity),
                members(rank_purpose(kernel.rank_scope, kernel.rank_group_id)),
            )
            fraction = scatter(
                LiabilityAttenuation(
                    **kernel.liability.model_dump()
                ).sample_conditional_from_uniforms(
                    severity,
                    members("native-payment-recovery"),
                    members("native-payment-share"),
                )
            )
    return EventDraws(member, tuple(amounts), labs, fraction, component_totals)


def linear_values(
    projection: LinearProjection, draws: EventDraws
) -> dict[str, np.ndarray]:
    """Project drawn components onto each view, summing in component order."""
    values = {}
    for view, weights in projection.weights.items():
        total = np.zeros(len(draws.member))
        for amounts, weight in zip(draws.amounts, weights, strict=True):
            if weight != 0:
                total += amounts * weight
        values[view] = total
    for view, (lab, cap) in projection.direct_labs.items():
        total = values[view] + draws.labs[lab]
        values[view] = total if cap is None else np.minimum(total, cap)
    return values
