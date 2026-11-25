"""Event-addressed native marks for multi-member and campaign-term incidents.

A mark kernel that can emit several members, choose among member outcomes or
carry campaign terms emits one episode per initiating event: its aggregate lab
amounts, its campaign mark (its joint branch's campaign components and their
financial mapping) and its members, each a mark of one member outcome. Every
uniform is addressed by state, origin, purpose, year and event ordinal
(``event_uniforms``), and a member's by its index in the event too
(``member_uniforms``), so neither batching, the requested views, other origins
nor the number of years simulated changes a draw:

- a joint branch is chosen by inverting its probabilities at ``joint-branch``,
  and each aggregate lab amount is drawn at ``aggregate-lab-mark::<lab>``;
- the shared gate opens when ``shared-gate`` falls below its probability;
- a cluster kernel's potential members are its cluster count inverted at
  ``member-count``, and its realized members their binomial thinning by the
  unit probability inverted at ``member-success``; each realized member takes
  its outcome at its own ``member-outcome``. Each member of a joint branch
  passes its unit gate at its own ``member-gate``, a binomial thinning of
  exchangeable members, as a binomial count and a uniform subset are;
- a component's rank is its rank group's: a campaign-scoped group's rank is
  the event's and so shared by its campaign mark and every member; a
  member-scoped group's is the mark's own, the event's for the campaign mark
  and the member's for a member (``rank_purpose``). A native payment kernel
  reads its source severity at its own rank group, and its recovery and share
  at its mark's ``native-payment-recovery`` and ``native-payment-share``
  (suffixed ``::<k>`` for a mark's k-th kernel after the first).

Amounts come from the native severity inverses; nothing is discretized.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from catmodel.contracts.components import (
    AggregateLabLossLaw,
    ComponentAmountLaw,
    ComponentObligation,
    ComponentPayment,
    NativePaymentKernel,
    PolicyMembership,
)
from catmodel.contracts.incident import ClusterCount, JointMarkOutcome, MarkKernel
from catmodel.contracts.native_severity import shared_native_severity_sampler
from catmodel.probability.count_inversion import binomial_counts
from catmodel.probability.distributions import (
    SeveritySampler,
    sample_affine_capped_severity,
)
from catmodel.probability.keyed_streams import event_uniforms, member_uniforms
from catmodel.probability.liability import LiabilityAttenuation
from catmodel.simulation.authored.payments import constant_native_payment_fraction
from catmodel.simulation.incident_linear import rank_purpose


@dataclass(frozen=True, slots=True)
class MarkLaw:
    """One kind of mark an episode emits: a joint branch's campaign mark or a
    member outcome, with what its amounts, views and policy terms need.

    ``scenario_ids`` is the outcome's scenario scope (empty for a campaign
    mark, whose components fall back to the origin's scenario).
    """

    components: tuple[ComponentAmountLaw, ...]
    samplers: tuple[SeveritySampler | None, ...]
    scenario_ids: tuple[str, ...]
    obligations: tuple[ComponentObligation, ...]
    payments: tuple[ComponentPayment, ...]
    memberships: tuple[PolicyMembership, ...]
    financial_mapping_complete: bool
    payment_kernels: tuple[NativePaymentKernel, ...]
    identified_gross_total: float | None = None
    composition_complete: bool = True
    overlap_resolved: bool = True

    @property
    def gross_identified(self) -> bool:
        """Whether a mark's unique gross is identified: by its own total, or
        by its complete and resolved components."""
        return self.identified_gross_total is not None or (
            self.composition_complete and self.overlap_resolved
        )


def _sampler(law) -> SeveritySampler | None:
    return (
        None if law.severity is None else shared_native_severity_sampler(law.severity)
    )


def _outcome_law(outcome: JointMarkOutcome) -> MarkLaw:
    return MarkLaw(
        components=outcome.components,
        samplers=tuple(_sampler(law) for law in outcome.components),
        scenario_ids=outcome.scenario_ids,
        obligations=outcome.obligations,
        payments=outcome.payments,
        memberships=outcome.memberships,
        financial_mapping_complete=outcome.financial_mapping_complete,
        payment_kernels=()
        if outcome.payment_kernel is None
        else (outcome.payment_kernel,),
        identified_gross_total=outcome.identified_gross_total,
        composition_complete=outcome.composition_complete,
        overlap_resolved=outcome.overlap_resolved,
    )


def campaign_mark_law(kernel: MarkKernel, branch=None) -> MarkLaw:
    """The campaign mark of ``kernel`` in joint ``branch`` (if any): the
    kernel's common campaign terms followed by the branch's own. Its mapping
    is complete when each part with components declares it complete."""
    common = kernel.campaign_components
    specific = () if branch is None else branch.campaign_components
    components = common + specific
    return MarkLaw(
        components=components,
        samplers=tuple(_sampler(law) for law in components),
        scenario_ids=(),
        obligations=kernel.campaign_obligations
        + (() if branch is None else branch.campaign_obligations),
        payments=kernel.campaign_payments
        + (() if branch is None else branch.campaign_payments),
        memberships=kernel.campaign_memberships
        + (() if branch is None else branch.campaign_memberships),
        financial_mapping_complete=(
            not common or kernel.campaign_financial_mapping_complete
        )
        and (
            branch is None or not specific or branch.campaign_financial_mapping_complete
        ),
        payment_kernels=tuple(
            item
            for item in (
                kernel.campaign_payment_kernel,
                None if branch is None else branch.campaign_payment_kernel,
            )
            if item is not None
        ),
    )


@dataclass(frozen=True, slots=True)
class CampaignLaw:
    """Prepared native laws of a kernel with campaign terms or several members.

    ``marks`` holds each kind of mark: the campaign mark of each joint branch
    (or the kernel's one) and then each member outcome. ``campaigns`` gives
    each branch's campaign mark in ``marks`` (-1 for one without
    components); ``branch_members`` each branch's member marks by position;
    ``outcomes`` a cluster kernel's member outcome marks. A branch or outcome
    of probability zero is never drawn.
    """

    kernel: MarkKernel
    marks: tuple[MarkLaw, ...]
    campaigns: tuple[int, ...]
    branch_cumulative: np.ndarray | None
    branch_members: tuple[tuple[int, ...], ...]
    outcomes: tuple[int, ...]
    outcome_cumulative: np.ndarray | None
    labs: tuple[AggregateLabLossLaw, ...]
    lab_samplers: tuple[SeveritySampler | None, ...]

    @property
    def cluster(self) -> ClusterCount | None:
        return self.kernel.cluster

    @property
    def members_possible(self) -> bool:
        return self.kernel.shared_probability > 0 and self.kernel.unit_probability > 0


def _cumulative(probabilities) -> np.ndarray:
    cumulative = np.cumsum(np.asarray(probabilities, dtype=float))
    return cumulative / cumulative[-1]


def campaign_law(kernel: MarkKernel) -> CampaignLaw:
    """Prepare every mark ``kernel`` can emit (``CampaignLaw``)."""
    marks: list[MarkLaw] = []
    campaigns: list[int] = []
    branch_members: list[tuple[int, ...]] = []
    branches = kernel.explicit_joint_members or (None,)
    for branch in branches:
        law = campaign_mark_law(kernel, branch)
        campaigns.append(len(marks) if law.components else -1)
        if law.components:
            marks.append(law)
    outcomes: tuple[int, ...] = ()
    if kernel.explicit_joint_members:
        for branch in kernel.explicit_joint_members:
            start = len(marks)
            marks.extend(_outcome_law(outcome) for outcome in branch.members)
            branch_members.append(tuple(range(start, len(marks))))
    else:
        start = len(marks)
        marks.extend(_outcome_law(outcome) for outcome in kernel.member_outcomes)
        outcomes = tuple(range(start, len(marks)))
    return CampaignLaw(
        kernel=kernel,
        marks=tuple(marks),
        campaigns=tuple(campaigns),
        branch_cumulative=_cumulative(
            [branch.probability for branch in kernel.explicit_joint_members]
        )
        if kernel.explicit_joint_members
        else None,
        branch_members=tuple(branch_members),
        outcomes=outcomes,
        outcome_cumulative=_cumulative(
            [outcome.probability for outcome in kernel.member_outcomes]
        )
        if len(kernel.member_outcomes) > 1
        else None,
        labs=kernel.campaign_lab_losses,
        lab_samplers=tuple(_sampler(law) for law in kernel.campaign_lab_losses),
    )


def cluster_counts(cluster: ClusterCount, uniforms: np.ndarray) -> np.ndarray:
    """Potential members of a cluster count, inverted from ``uniforms`` by
    its support's counts in increasing order."""
    rows = sorted(cluster.support, key=lambda row: row.count)
    counts = np.array([row.count for row in rows], dtype=np.int64)
    cumulative = _cumulative([row.probability for row in rows])
    return counts[
        np.minimum(np.searchsorted(cumulative, uniforms, side="right"), len(counts) - 1)
    ]


@dataclass(frozen=True, slots=True)
class MarkDraws:
    """The marks of one kind in a batch of episodes, one row each.

    ``law`` indexes ``CampaignLaw.marks``; ``event`` gives each mark's event in
    the batch and ``member`` its member index there (-1 for a campaign mark).
    ``payments`` holds each native kernel's liability share, when drawn.
    """

    law: int
    event: np.ndarray
    member: np.ndarray
    amounts: tuple[np.ndarray, ...]
    payments: tuple[np.ndarray | None, ...]
    totals: np.ndarray


@dataclass(frozen=True, slots=True)
class EpisodeDraws:
    """A batch of episodes: their aggregate lab amounts, their potential and
    realized member counts, and their marks by kind (``MarkDraws``)."""

    labs: dict[str, np.ndarray]
    potential: np.ndarray
    realized: np.ndarray
    marks: tuple[MarkDraws, ...]


@dataclass(frozen=True, slots=True)
class _Stream:
    """An origin's event-address identity: its stream seed, state and origin."""

    root_seed: int
    state_id: str
    origin_id: str


class _Addresses:
    """Uniforms of one set of marks by purpose, cached: each mark's event's,
    or with ``members`` each mark's own as that member."""

    def __init__(self, years, ordinals, members, stream: _Stream):
        self._years, self._ordinals, self._members = years, ordinals, members
        self._stream = stream
        self._cache: dict[tuple[bool, str], np.ndarray] = {}

    def __call__(self, purpose: str, *, event: bool = False) -> np.ndarray:
        own = self._members is not None and not event
        key = (own, purpose)
        if key not in self._cache:
            self._cache[key] = (
                member_uniforms(
                    self._years,
                    self._ordinals,
                    self._members,
                    root_seed=self._stream.root_seed,
                    state_id=self._stream.state_id,
                    origin_id=self._stream.origin_id,
                    purpose=purpose,
                )
                if own
                else event_uniforms(
                    self._years,
                    self._ordinals,
                    root_seed=self._stream.root_seed,
                    state_id=self._stream.state_id,
                    origin_id=self._stream.origin_id,
                    purpose=purpose,
                )
            )
        return self._cache[key]


def episode_members(
    law: CampaignLaw,
    years: np.ndarray,
    ordinals: np.ndarray,
    *,
    root_seed: int,
    state_id: str,
    origin_id: str,
    rows: bool = False,
) -> tuple[np.ndarray, ...]:
    """Each episode's potential and realized member counts and, with
    ``rows``, the realized members' events (positions in ``years``) and
    member indices, in event and member order.

    A cluster kernel's counts are inverted without building a row per member,
    so a caller can refuse them before any member is drawn.
    """
    count = len(years)
    stream = _Stream(root_seed, state_id, origin_id)
    every = _Addresses(years, ordinals, None, stream)
    kernel = law.kernel
    gate = (
        every("shared-gate") < kernel.shared_probability
        if law.members_possible
        else np.zeros(count, dtype=bool)
    )
    if law.cluster is not None:
        potential = np.zeros(count, dtype=np.int64)
        opened = np.flatnonzero(gate)
        if opened.size:
            potential[opened] = cluster_counts(
                law.cluster, every("member-count")[opened]
            )
        realized = binomial_counts(
            every("member-success"), potential, kernel.unit_probability
        )
        if not rows:
            return potential, realized
        events, members = _rows(realized)
        return potential, realized, events, members
    branch = _branches(law, every, count)
    sizes = np.array([len(members) for members in law.branch_members], dtype=np.int64)
    potential = np.where(gate, sizes[branch], 0)
    events, members = _rows(potential)
    if kernel.unit_probability < 1 and len(events):
        gates = _Addresses(years[events], ordinals[events], members, stream)
        passed = gates("member-gate") < kernel.unit_probability
        events, members = events[passed], members[passed]
    realized = np.bincount(events, minlength=count).astype(np.int64)
    if not rows:
        return potential, realized
    return potential, realized, events, members


def _rows(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One row per member: its event and its index there."""
    events = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    members = np.arange(len(events), dtype=np.int64) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    return events, members


def _branches(law: CampaignLaw, every: _Addresses, count: int) -> np.ndarray:
    """Each episode's joint branch, zero for a cluster kernel."""
    if law.branch_cumulative is None:
        return np.zeros(count, dtype=np.int64)
    return np.minimum(
        np.searchsorted(law.branch_cumulative, every("joint-branch"), side="right"),
        len(law.branch_cumulative) - 1,
    )


def draw_episodes(
    law: CampaignLaw,
    years: np.ndarray,
    ordinals: np.ndarray,
    *,
    root_seed: int,
    state_id: str,
    origin_id: str,
    payments: bool = False,
    max_members: int,
    reserve: Callable[[int], None] | None = None,
) -> EpisodeDraws:
    """Draw one bounded batch of episodes from native inverses and exact ranks.

    Refuses an episode whose realized members exceed ``max_members``, and
    hands the batch's realized members to ``reserve``, before drawing any
    member's marks. ``payments`` adds each native payment kernel's liability
    share.
    """
    years = np.asarray(years)
    ordinals = np.asarray(ordinals)
    if years.ndim != 1 or years.shape != ordinals.shape:
        raise ValueError("event coordinates must be aligned one-dimensional arrays")
    stream = _Stream(root_seed, state_id, origin_id)
    potential, realized = episode_members(
        law,
        years,
        ordinals,
        root_seed=root_seed,
        state_id=state_id,
        origin_id=origin_id,
    )
    if len(realized) and int(realized.max()) > max_members:
        raise ValueError(
            f"a campaign's {int(realized.max()):,} realized members exceed "
            f"max_members_per_campaign ({max_members:,})"
        )
    if reserve is not None:
        reserve(int(realized.sum()))
    _, _, member_events, member_indices = episode_members(
        law,
        years,
        ordinals,
        root_seed=root_seed,
        state_id=state_id,
        origin_id=origin_id,
        rows=True,
    )
    every = _Addresses(years, ordinals, None, stream)
    inverses: dict[tuple[int, bytes], np.ndarray] = {}

    def inverse(sampler, selected):
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

    labs = {}
    for lab, sampler in zip(law.labs, law.lab_samplers, strict=True):
        if lab.amount is not None:
            labs[lab.lab_id] = np.full(len(years), lab.amount)
        else:
            assert sampler is not None
            labs[lab.lab_id] = native(
                sampler, every("aggregate-lab-mark::" + lab.lab_id)
            )
    branch = _branches(law, every, len(years))
    marks = []
    for number, campaign in enumerate(law.campaigns):
        if campaign < 0:
            continue
        events = np.flatnonzero(branch == number)
        if not events.size:
            continue
        addresses = _Addresses(years[events], ordinals[events], None, stream)
        marks.append(
            _mark_draws(
                law,
                campaign,
                events,
                np.full(len(events), -1),
                addresses,
                native,
                payments,
            )
        )
    if len(member_events):
        if law.cluster is not None:
            kinds = (
                np.asarray(law.outcomes, dtype=np.int64)[
                    np.minimum(
                        np.searchsorted(
                            law.outcome_cumulative,
                            _Addresses(
                                years[member_events],
                                ordinals[member_events],
                                member_indices,
                                stream,
                            )("member-outcome"),
                            side="right",
                        ),
                        len(law.outcomes) - 1,
                    )
                ]
                if law.outcome_cumulative is not None
                else np.full(len(member_events), law.outcomes[0])
            )
        else:
            width = max(len(members) for members in law.branch_members)
            table = np.full((len(law.branch_members), width), -1, dtype=np.int64)
            for row, members in enumerate(law.branch_members):
                table[row, : len(members)] = members
            kinds = table[branch[member_events], member_indices]
        for kind in np.unique(kinds):
            rows = np.flatnonzero(kinds == kind)
            events, members = member_events[rows], member_indices[rows]
            addresses = _Addresses(years[events], ordinals[events], members, stream)
            marks.append(
                _mark_draws(
                    law, int(kind), events, members, addresses, native, payments
                )
            )
    return EpisodeDraws(labs, potential, realized, tuple(marks))


def _mark_draws(law, kind, events, members, addresses, native, payments) -> MarkDraws:
    """One kind of mark's amounts and payment shares, its ranks read at
    ``addresses``."""
    mark = law.marks[kind]
    count = len(events)
    amounts = []
    totals = np.zeros(count)
    for component, sampler in zip(mark.components, mark.samplers, strict=True):
        if component.multiplier == 0 or component.post_multiplier == 0:
            sampled = np.zeros(count)
        elif component.amount is not None:
            sampled = np.full(
                count,
                component.amount * component.multiplier * component.post_multiplier,
            )
        else:
            assert sampler is not None
            sampled = native(
                sampler,
                addresses(
                    rank_purpose(component.rank_scope, component.rank_group_id),
                    event=component.rank_scope == "campaign",
                ),
                multiplier=component.multiplier,
                post_multiplier=component.post_multiplier,
            )
        amounts.append(sampled)
        totals += sampled
    fractions = []
    for number, kernel in enumerate(mark.payment_kernels):
        if not payments:
            fractions.append(None)
            continue
        constant = constant_native_payment_fraction(kernel)
        if constant is not None:
            fractions.append(np.full(count, constant))
            continue
        assert kernel.liability is not None
        suffix = "" if number == 0 else f"::{number}"
        severity = native(
            shared_native_severity_sampler(kernel.source_severity),
            addresses(
                rank_purpose(kernel.rank_scope, kernel.rank_group_id),
                event=kernel.rank_scope == "campaign",
            ),
        )
        fractions.append(
            LiabilityAttenuation(
                **kernel.liability.model_dump()
            ).sample_conditional_from_uniforms(
                severity,
                addresses("native-payment-recovery" + suffix),
                addresses("native-payment-share" + suffix),
            )
        )
    return MarkDraws(
        kind,
        np.asarray(events, dtype=np.int64),
        np.asarray(members, dtype=np.int64),
        tuple(amounts),
        tuple(fractions),
        totals,
    )


def gross_values(mark: MarkLaw, draws: MarkDraws) -> np.ndarray:
    """Each mark's unique gross: its identified total, which its known
    components may not exceed (nor, when complete and resolved, differ from),
    or its components' total."""
    if mark.identified_gross_total is None:
        return draws.totals
    identified = mark.identified_gross_total
    if mark.overlap_resolved:
        totals = draws.totals
        close = np.abs(totals - identified) <= 1e-12 * np.maximum(
            np.abs(totals), abs(identified)
        )
        if np.any((totals > identified) & ~close):
            raise ValueError("known components exceed the identified gross total")
        if mark.composition_complete and not np.all(close):
            raise ValueError("complete components disagree with identified gross")
    return np.full(len(draws.event), identified)
