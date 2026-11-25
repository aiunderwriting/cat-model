"""Contract terms on covered component amounts, reduced as arrays.

An eligible policy membership of a mark covers part of one physical
component: its amount times its payment fraction and the membership's covered
fraction (a *share*), plus any defense loadings on that amount, each inside or
outside the contract's limits (the share's *parts*). The terms apply in this
order:

1. A contract's parts in one occurrence add: those inside its limits to the
   occurrence's subject amount and those outside to its extra amount. A
   member- or campaign-scoped occurrence lies within one event (a member's or
   the whole episode's); a contract-year occurrence collects every event of
   the year that names it.
2. The occurrence retention and limit apply to the subject amount, and the
   extra amount adds back: the occurrence payment.
3. A contract's occurrence payments in a year add, the extra amounts among
   them only when the contract subjects outside-limit amounts to its
   aggregate; the annual retention and limit apply to that sum, and the other
   extra amounts add after them. ``financial`` sums the contracts' annual
   payments; independent contracts are not coordinated.
4. ``oep_financial`` is the year's largest occurrence payment (step 2) summed
   across the contracts sharing that occurrence; a risk unit's
   ``occurrence_net_financial`` sums its own occurrences' payments.

Every sum adds left to right from zero in one fixed order: events in the order
they are added, origin by origin and then by year and event ordinal; within an
event, its campaign mark before its members and each mark's memberships and
parts in order; a contract's payments within an event before those of later
events; contract-year occurrences after the events of their year; contracts in
the order their first events added them. A year's values therefore do not
depend on how its events are batched.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from catmodel.contracts.components import (
    ComponentAmountLaw,
    ComponentObligation,
    ComponentPayment,
    ContractTerms,
    NativePaymentKernel,
    PolicyMembership,
)
from catmodel.simulation.components import defense_loading_state

INCOMPLETE_FINANCIAL_MAPPING = (
    "Complete component, obligation and policy-occurrence mappings are required."
)
"""Why ``financial`` is unavailable for marks whose mapping is incomplete."""

type OccurrenceScope = Literal["member", "campaign", "contract_year"]


def contract_year_occurrence(contract_id: str, occurrence_id: str) -> str:
    """The resolved ID of a contract-year occurrence, shared by every event of
    the year that names it on that contract."""
    return "contract-year/" + contract_id + "/" + occurrence_id


@dataclass(frozen=True, slots=True)
class PolicyShare:
    """One eligible membership of a mark: the share of a component it covers.

    ``payment`` is the paid fraction: the authored payment, one for a native
    direct component, or ``None`` where the liability share of the mark's
    native kernel ``kernel`` pays it, drawn per mark. ``parts`` multiplies
    the covered amount: one and each defense loading, with whether the part
    lies outside the limits.
    """

    component: int
    contract_id: str
    scope: OccurrenceScope
    occurrence_id: str | None
    payment: float | None
    covered_fraction: float
    parts: tuple[tuple[bool, float], ...]
    kernel: int = 0


def policy_shares(
    components: Sequence[ComponentAmountLaw],
    obligations: Sequence[ComponentObligation],
    payments: Sequence[ComponentPayment],
    memberships: Sequence[PolicyMembership],
    payment_kernels: Sequence[NativePaymentKernel],
    contracts: Mapping[str, ContractTerms],
) -> tuple[PolicyShare, ...]:
    """The shares of one mark's complete financial mapping, in membership order.

    Ineligible and zero-covered memberships take no share. A native payment
    kernel pays its liability components their drawn liability share and its
    direct components whole, for its own entity; other coverage needs an
    authored obligation and payment. Setup validation refuses a complete
    mapping that names an undeclared contract or another entity, or covers a
    component without its obligation and payment; this refuses them too.
    Defense charges that conflict are not refused here: admission withholds
    the financial views of a law that can realize one (``view_admission``).
    """
    positions = {law.component_id: index for index, law in enumerate(components)}
    obligated = {(row.component_id, row.entity_id) for row in obligations}
    paid = {(row.component_id, row.entity_id): row.fraction for row in payments}
    shares = []
    for membership in memberships:
        if not membership.eligible or membership.covered_fraction == 0:
            continue
        contract = contracts.get(membership.contract_id)
        if contract is None:
            raise ValueError("policy membership references an undeclared contract")
        if membership.obligated_entity_id != contract.entity_id:
            raise ValueError("policy entity differs from the obligated entity")
        key = (membership.component_id, membership.obligated_entity_id)
        owner = next(
            (
                number
                for number, kernel in enumerate(payment_kernels)
                if kernel.entity_id == membership.obligated_entity_id
                and membership.component_id
                in (*kernel.liability_component_ids, *kernel.direct_component_ids)
            ),
            None,
        )
        payment: float | None
        if owner is not None:
            kernel = payment_kernels[owner]
            payment = (
                None
                if membership.component_id in kernel.liability_component_ids
                else 1.0
            )
        elif key not in obligated:
            raise ValueError("coverage cannot invent an absent legal obligation")
        elif key not in paid:
            raise ValueError("coverage requires an explicit payment outcome")
        else:
            payment = paid[key]
        contract_loading = defense_loading_state(None, True, membership, contract)[0]
        parts = [(membership.outside_limits, 1.0)]
        if membership.defense_loading > 0:
            parts.append(
                (membership.defense_outside_limits, membership.defense_loading)
            )
        if contract_loading > 0:
            parts.append((contract.defense_outside_limits, contract_loading))
        shares.append(
            PolicyShare(
                positions[membership.component_id],
                contract.contract_id,
                membership.occurrence_scope,
                membership.occurrence_id,
                payment,
                membership.covered_fraction,
                tuple(parts),
                0 if owner is None else owner,
            )
        )
    return tuple(shares)


@dataclass(frozen=True, slots=True)
class CoveredParts:
    """A batch's covered parts, one row each, in the order they add.

    ``event`` indexes the batch's events; within an event, rows run by mark
    (the campaign before members), membership and part. An event-scoped
    occurrence is ``(member, occurrence)`` within its event: ``member`` is the
    member of a member-scoped occurrence and -1 otherwise. A contract-year
    occurrence (``contract_year``) is ``occurrence`` across the year.
    ``occurrence`` holds codes from the reduction (``PolicyReduction.code``)
    and ``contract`` its contract indices.
    """

    event: np.ndarray
    member: np.ndarray
    occurrence: np.ndarray
    contract: np.ndarray
    contract_year: np.ndarray
    amount: np.ndarray
    outside: np.ndarray

    @classmethod
    def empty(cls) -> CoveredParts:
        return cls.concatenate(())

    @classmethod
    def stack(cls, batches: Sequence[CoveredParts]) -> CoveredParts:
        """Join batches' rows in the order given."""

        def column(name, dtype):
            values = [getattr(batch, name) for batch in batches]
            return np.concatenate(values) if values else np.zeros(0, dtype=dtype)

        return cls(
            column("event", np.int64),
            column("member", np.int64),
            column("occurrence", np.int64),
            column("contract", np.int64),
            column("contract_year", bool),
            column("amount", float),
            column("outside", bool),
        )

    @classmethod
    def concatenate(cls, batches: Sequence[CoveredParts]) -> CoveredParts:
        """Join batches covering disjoint events, ordered by event; each
        event's rows keep their order."""
        joined = cls.stack(batches)
        if len(batches) < 2:
            return joined
        return joined.select(np.argsort(joined.event, kind="stable"))

    def select(self, rows: np.ndarray) -> CoveredParts:
        return CoveredParts(*(getattr(self, name)[rows] for name in _COLUMNS))


_COLUMNS = (
    "event",
    "member",
    "occurrence",
    "contract",
    "contract_year",
    "amount",
    "outside",
)


def _first_ids(*columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Number each row's key (one value from each column) by its first
    appearance; returns each row's number and each number's first row."""
    count = len(columns[0])
    if not count:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    # lexsort is stable, so each run of equal keys starts at its first row.
    order = np.lexsort(columns[::-1])
    starts = np.zeros(count, dtype=bool)
    starts[0] = True
    for column in columns:
        ordered = column[order]
        starts[1:] |= ordered[1:] != ordered[:-1]
    runs = np.cumsum(starts) - 1
    firsts = order[starts]
    ranking = np.argsort(firsts, kind="stable")
    numbers = np.empty(len(firsts), dtype=np.int64)
    numbers[ranking] = np.arange(len(firsts))
    ids = np.empty(count, dtype=np.int64)
    ids[order] = numbers[runs]
    return ids, firsts[ranking]


class PolicyReduction:
    """Each year's policy views from batches of complete events added in order.

    ``financial`` keeps each contract's annual sums, ``occurrence_maximum``
    the year's largest occurrence payment (``oep_financial``) and
    ``risk_units`` the occurrence payments of each named origin, under the
    view name it maps to. A batch adds every event of one origin it covers,
    so contract-year occurrences, which can span events and origins, wait for
    ``views`` to apply their occurrence terms.
    """

    def __init__(
        self,
        contracts: Sequence[ContractTerms],
        simulations: int,
        *,
        financial: bool,
        occurrence_maximum: bool,
        risk_units: Mapping[str, str] | None = None,
    ) -> None:
        self.contracts = tuple(contracts)
        self.index = {
            contract.contract_id: number
            for number, contract in enumerate(self.contracts)
        }
        if len(self.index) != len(self.contracts):
            raise ValueError(
                "contract IDs must be unique within a contract-year reduction"
            )
        self._retention = np.array(
            [contract.occurrence_retention for contract in self.contracts], dtype=float
        )
        self._limit = np.array(
            [
                np.inf
                if contract.occurrence_limit is None
                else contract.occurrence_limit
                for contract in self.contracts
            ],
            dtype=float,
        )
        self._aggregate = np.array(
            [
                contract.outside_limits_subject_to_aggregate
                for contract in self.contracts
            ],
            dtype=bool,
        )
        self._simulations = simulations
        self._financial = financial
        self._maximum = np.zeros(simulations) if occurrence_maximum else None
        self._risk_views = dict(risk_units or {})
        self._risk = {origin: np.zeros(simulations) for origin in self._risk_views}
        # Each contract's annual (subject, outside) sums, in registration order.
        self._annual: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._codes: dict[tuple[str, str | None], int] = {}
        # Each contract-year occurrence's (covered, outside) sums, by
        # (occurrence code, contract) and, for risk units, also by origin.
        self._years: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._origin_years: dict[
            tuple[int, int, str], tuple[np.ndarray, np.ndarray]
        ] = {}

    def code(self, scope: str, occurrence_id: str | None) -> int:
        """The code of an occurrence ID within ``scope``: ``member`` or
        ``campaign`` (event-scoped) or ``contract_year`` (resolved by
        ``contract_year_occurrence``)."""
        return self._codes.setdefault((scope, occurrence_id), len(self._codes))

    def register(self, contract_ids: Iterable[str]) -> None:
        """Add contracts to the annual order, if they are not in it yet."""
        if not self._financial:
            return
        for contract_id in contract_ids:
            self._register(self.index[contract_id])

    def _register(self, contract: int) -> tuple[np.ndarray, np.ndarray]:
        sums = self._annual.get(contract)
        if sums is None:
            sums = self._annual[contract] = (
                np.zeros(self._simulations),
                np.zeros(self._simulations),
            )
        return sums

    def add(self, years: np.ndarray, parts: CoveredParts, *, origin: str) -> None:
        """Add one batch of ``origin``'s events; ``years`` holds each event's
        year, and every event of a covered occurrence is in the batch."""
        if not len(parts.amount):
            return
        within = ~parts.contract_year
        with np.errstate(over="ignore", invalid="ignore"):
            if np.all(within):
                self._add_events(years, parts, origin)
            elif np.any(within):
                self._add_events(years, parts.select(np.flatnonzero(within)), origin)
            if not np.all(within):
                self._add_contract_years(
                    years, parts.select(np.flatnonzero(~within)), origin
                )

    def _occurrence_payments(self, subject, extra, contracts):
        payout = np.minimum(
            np.maximum(subject - self._retention[contracts], 0.0),
            self._limit[contracts],
        )
        return payout, payout + extra

    def _add_events(self, years, parts: CoveredParts, origin: str) -> None:
        groups, first = _first_ids(
            parts.event, parts.member, parts.occurrence, parts.contract
        )
        subject, extra = np.zeros(len(first)), np.zeros(len(first))
        covered = ~parts.outside
        np.add.at(subject, groups[covered], parts.amount[covered])
        np.add.at(extra, groups[parts.outside], parts.amount[parts.outside])
        contracts, events = parts.contract[first], parts.event[first]
        payout, paid = self._occurrence_payments(subject, extra, contracts)
        risk = self._risk.get(origin)
        if risk is not None:
            totals = np.zeros(len(years))
            np.add.at(totals, events, paid)
            np.add.at(risk, years, totals)
        if self._maximum is not None:
            occurrences, firsts = _first_ids(
                events, parts.member[first], parts.occurrence[first]
            )
            sums = np.zeros(len(firsts))
            np.add.at(sums, occurrences, paid)
            np.maximum.at(self._maximum, years[events[firsts]], sums)
        if not self._financial:
            return
        units, firsts = _first_ids(events, contracts)
        aggregate = self._aggregate[contracts]
        subject_sums, outside_sums = np.zeros(len(firsts)), np.zeros(len(firsts))
        np.add.at(subject_sums, units, np.where(aggregate, paid, payout))
        np.add.at(outside_sums, units, np.where(aggregate, 0.0, extra))
        unit_contracts, unit_years = contracts[firsts], years[events[firsts]]
        for contract in dict.fromkeys(unit_contracts.tolist()):
            selected = unit_contracts == contract
            annual, outside = self._register(contract)
            np.add.at(annual, unit_years[selected], subject_sums[selected])
            np.add.at(outside, unit_years[selected], outside_sums[selected])

    def _add_contract_years(self, years, parts: CoveredParts, origin: str) -> None:
        keys, firsts = _first_ids(parts.occurrence, parts.contract)
        event_years = years[parts.event]
        for number, row in enumerate(firsts.tolist()):
            key = (int(parts.occurrence[row]), int(parts.contract[row]))
            selected = keys == number
            targets = [self._years.setdefault(key, self._zeros())]
            if origin in self._risk:
                targets.append(
                    self._origin_years.setdefault((*key, origin), self._zeros())
                )
            for outside, sums in ((False, 0), (True, 1)):
                rows = selected & (parts.outside == outside)
                for target in targets:
                    np.add.at(target[sums], event_years[rows], parts.amount[rows])

    def _zeros(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(self._simulations), np.zeros(self._simulations)

    def views(self) -> dict[str, np.ndarray]:
        """Apply the contract-year occurrence terms and the annual terms;
        returns ``financial``, ``oep_financial`` and each risk unit's view, as
        kept."""
        result: dict[str, np.ndarray] = {}
        with np.errstate(over="ignore", invalid="ignore"):
            for (_, contract), (covered, extra) in self._years.items():
                payout, paid = self._occurrence_payments(covered, extra, contract)
                if self._maximum is not None:
                    np.maximum(self._maximum, paid, out=self._maximum)
                if self._financial:
                    annual, outside = self._register(contract)
                    if self._aggregate[contract]:
                        annual += paid
                    else:
                        annual += payout
                        outside += extra
            for (_, contract, origin), (covered, extra) in self._origin_years.items():
                self._risk[origin] += self._occurrence_payments(
                    covered, extra, contract
                )[1]
            if self._financial:
                total = np.zeros(self._simulations)
                for contract, (annual, outside) in self._annual.items():
                    terms = self.contracts[contract]
                    payout = np.maximum(annual - terms.annual_retention, 0.0)
                    if terms.annual_limit is not None:
                        payout = np.minimum(payout, terms.annual_limit)
                    total = total + (payout + outside)
                result["financial"] = total
        if self._maximum is not None:
            result["oep_financial"] = self._maximum
        for origin, values in self._risk.items():
            result[self._risk_views[origin]] = values
        return result


def share_parts(
    reduction: PolicyReduction,
    shares: Sequence[PolicyShare],
    amounts: Sequence[np.ndarray],
    payments: Sequence[np.ndarray | None],
    events: np.ndarray,
    members: np.ndarray | None = None,
) -> CoveredParts:
    """The covered parts of one mark law's marks, one mark per row of
    ``amounts`` (its components' amounts) and of each of ``payments`` (its
    native kernels' liability shares), in mark order: ``events`` holds each
    mark's event and ``members`` its member index, -1 for a campaign mark
    (all zero by default). Each mark's parts run by share and part."""
    count = len(events)
    members = np.zeros(count, dtype=np.int64) if members is None else members
    columns, static = [], []
    for share in shares:
        fraction = payments[share.kernel] if share.payment is None else share.payment
        assert fraction is not None
        covered = amounts[share.component] * fraction * share.covered_fraction
        contract_year = share.scope == "contract_year"
        occurrence = (
            reduction.code(
                "contract_year",
                contract_year_occurrence(share.contract_id, share.occurrence_id or ""),
            )
            if contract_year
            else reduction.code(share.scope, share.occurrence_id)
        )
        for outside, multiplier in share.parts:
            columns.append(covered * multiplier)
            static.append(
                (
                    share.scope == "member",
                    occurrence,
                    reduction.index[share.contract_id],
                    contract_year,
                    outside,
                )
            )
    width = len(columns)
    if not width or not count:
        return CoveredParts.empty()
    member_scoped, occurrence, contract, contract_year, outside = (
        np.array(values) for values in zip(*static, strict=True)
    )
    return CoveredParts(
        np.repeat(np.asarray(events, dtype=np.int64), width),
        np.where(np.tile(member_scoped, count), np.repeat(members, width), -1),
        np.tile(occurrence.astype(np.int64), count),
        np.tile(contract.astype(np.int64), count),
        np.tile(contract_year.astype(bool), count),
        np.stack(columns, axis=1).ravel(),
        np.tile(outside.astype(bool), count),
    )
