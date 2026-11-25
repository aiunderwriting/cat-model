"""An exact oracle for policy terms, and the covered parts of event-addressed
draws rebuilt for it.

``exact_policy_views`` applies contract terms occurrence by occurrence with
exact sums (``math.fsum``), independently of the array reduction
(``catmodel.simulation.policy_terms``). ``event_rows`` draws an event-addressed
origin's events as the runtime draws them and derives each member's covered
parts from its outcome's memberships, so the oracle can recompute every
year's policy views from the same draws.
"""

import math
from collections import defaultdict

import numpy as np

from catmodel.simulation import incident_runtime as runtime
from catmodel.simulation.incident import CompiledIncidentState
from catmodel.simulation.incident_linear import (
    SingleMemberLaw,
    draw_single_member_events,
)

type Row = tuple[str, tuple, int, str, str | None, int, float, bool]
"""A covered part: (origin, event, mark member, occurrence scope, occurrence
ID, contract index, amount, outside the limits)."""


def exact_policy_views(rows, contracts, origins=None) -> dict[str, float]:
    """A year's ``financial``, ``oep_financial`` and summed occurrence payments
    of ``rows`` (``Row``), of ``origins`` only when given."""
    covered, extra = defaultdict(list), defaultdict(list)
    for origin, event, member, scope, occurrence, contract, amount, outside in rows:
        if origins is not None and origin not in origins:
            continue
        if scope == "contract_year":
            key = (contract, ("contract_year", occurrence))
        elif scope == "member":
            key = (contract, (event, member, occurrence))
        else:
            key = (contract, (event, "campaign", occurrence))
        (extra if outside else covered)[key].append(amount)
    paid, annual, annual_extra = {}, defaultdict(list), defaultdict(list)
    for key in set(covered) | set(extra):
        terms = contracts[key[0]]
        payout = max(math.fsum(covered[key]) - terms.occurrence_retention, 0.0)
        if terms.occurrence_limit is not None:
            payout = min(payout, terms.occurrence_limit)
        outside = math.fsum(extra[key])
        paid[key] = payout + outside
        if terms.outside_limits_subject_to_aggregate:
            annual[key[0]].append(payout + outside)
        else:
            annual[key[0]].append(payout)
            annual_extra[key[0]].append(outside)
    financial = []
    for number, terms in enumerate(contracts):
        payout = max(math.fsum(annual[number]) - terms.annual_retention, 0.0)
        if terms.annual_limit is not None:
            payout = min(payout, terms.annual_limit)
        financial.append(payout + math.fsum(annual_extra[number]))
    occurrences = defaultdict(list)
    for (contract, occurrence), value in paid.items():
        # A contract-year occurrence is each contract's own.
        shared = (
            (contract, occurrence) if occurrence[0] == "contract_year" else occurrence
        )
        occurrences[shared].append(value)
    return {
        "financial": math.fsum(financial),
        "oep_financial": max(
            (math.fsum(values) for values in occurrences.values()), default=0.0
        ),
        "occurrences": math.fsum(paid.values()),
    }


def _member_rows(origin, event, outcome, amounts, payment, contracts) -> list[Row]:
    """One member's covered parts, from its outcome's memberships."""
    index = {contract.contract_id: number for number, contract in enumerate(contracts)}
    kernel = outcome.payment_kernel
    paid = {(row.component_id, row.entity_id): row.fraction for row in outcome.payments}
    rows = []
    for membership in outcome.memberships:
        if not membership.eligible or membership.covered_fraction == 0:
            continue
        position = next(
            number
            for number, component in enumerate(outcome.components)
            if component.component_id == membership.component_id
        )
        native = (
            kernel is not None and kernel.entity_id == membership.obligated_entity_id
        )
        if native and membership.component_id in kernel.liability_component_ids:
            fraction = payment
        elif native and membership.component_id in kernel.direct_component_ids:
            fraction = 1.0
        else:
            fraction = paid[(membership.component_id, membership.obligated_entity_id)]
        amount = amounts[position] * fraction * membership.covered_fraction
        contract = contracts[index[membership.contract_id]]
        parts = [(membership.outside_limits, amount)]
        if membership.defense_loading > 0:
            parts.append(
                (membership.defense_outside_limits, amount * membership.defense_loading)
            )
        if not membership.outside_limits and contract.defense_loading > 0:
            parts.append(
                (contract.defense_outside_limits, amount * contract.defense_loading)
            )
        rows.extend(
            (
                origin,
                event,
                0,
                membership.occurrence_scope,
                membership.occurrence_id,
                index[membership.contract_id],
                value,
                outside,
            )
            for outside, value in parts
        )
    return rows


def event_rows(
    compiled: CompiledIncidentState,
    origin_id: str,
    counts: np.ndarray,
    *,
    seed: int,
) -> dict[int, list[Row]]:
    """Each year's covered parts of an event-addressed origin's events, drawn
    as the runtime draws them for ``counts`` events a year under stream seed
    ``seed`` (``runtime.state_stream_seed``)."""
    incident = next(row for row in compiled.incidents if row.origin_id == origin_id)
    laws = runtime._event_laws(compiled)[origin_id]
    years = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    ordinals = np.arange(len(years), dtype=np.int64) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    selected = runtime._segment_choice(
        incident, years, ordinals, seed=seed, state_id=compiled.state_id
    )
    rows: dict[int, list[Row]] = defaultdict(list)
    for index, law in laws.items():
        assert isinstance(law, SingleMemberLaw)
        mask = selected == index
        if not np.any(mask):
            continue
        draws = draw_single_member_events(
            law,
            years[mask],
            ordinals[mask],
            root_seed=seed,
            state_id=compiled.state_id,
            origin_id=origin_id,
            payments=True,
        )
        for row, (year, ordinal) in enumerate(
            zip(years[mask], ordinals[mask], strict=True)
        ):
            if not draws.member[row]:
                continue
            payment = (
                None if draws.payment_fraction is None else draws.payment_fraction[row]
            )
            rows[int(year)].extend(
                _member_rows(
                    origin_id,
                    (int(year), int(ordinal)),
                    law.outcome,
                    [float(amounts[row]) for amounts in draws.amounts],
                    None if payment is None else float(payment),
                    compiled.contracts,
                )
            )
    return rows
