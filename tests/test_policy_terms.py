"""The array reduction of policy terms matches an exact scalar oracle.

Seeded random years of episodes (a campaign mark and members, from two source
origins) carry covered shares of components under memberships of every
occurrence scope, with defense loadings inside and outside the limits; random
contracts carry occurrence and annual retentions and limits and either
outside-limit aggregation. ``PolicyReduction`` must give each year's
``financial``, ``oep_financial`` and risk-unit payments as
``exact_policy_views`` does: it applies the same terms occurrence by occurrence
with exact sums (``math.fsum``). With amounts on a dyadic grid every sum is
exact and the results are equal bit for bit; otherwise they may differ by
rounding alone, which ``_tolerance`` bounds.
"""

import math
from collections import defaultdict

import numpy as np
import pytest

from catmodel.contracts.components import ContractTerms
from catmodel.simulation.policy_terms import CoveredParts, PolicyReduction
from test_support.policy_oracle import exact_policy_views

ORIGINS = ("first", "second")
UNIT_ROUNDOFF = 2.0**-53


def _risk_view(origin):
    return f"risk_unit::{origin}::occurrence_net_financial"


def _value(rng, dyadic, *, scale=1.0, zero=0.1):
    """A nonnegative quantity: on a grid of eighths when ``dyadic``, else any
    float over several orders of magnitude."""
    if rng.random() < zero:
        return 0.0
    if dyadic:
        return float(rng.integers(1, 64)) / 8 * scale
    return float(rng.uniform(0.5, 2.0) * 10 ** rng.uniform(-2, 3) * scale)


def _fraction(rng, dyadic, *, zero=0.1):
    if rng.random() < zero:
        return 0.0
    return float(rng.integers(1, 9)) / 8 if dyadic else float(rng.uniform(0.05, 1))


def _contracts(rng, dyadic):
    contracts = []
    for number in range(int(rng.integers(1, 4))):
        limits = {}
        if rng.random() < 0.5:
            limits["occurrence_limit"] = _value(rng, dyadic, scale=40, zero=0.05)
        if rng.random() < 0.5:
            limits["annual_limit"] = _value(rng, dyadic, scale=120, zero=0.05)
        contracts.append(
            ContractTerms(
                contract_id=f"contract-{number}",
                contract_year="2027",
                entity_id="insured",
                occurrence_retention=_value(rng, dyadic, scale=4, zero=0.4),
                annual_retention=_value(rng, dyadic, scale=20, zero=0.4),
                defense_loading=_value(rng, dyadic, scale=0.25, zero=0.6),
                defense_outside_limits=bool(rng.random() < 0.5),
                outside_limits_subject_to_aggregate=bool(rng.random() < 0.5),
                **limits,
            )
        )
    return contracts


def _year(rng, dyadic, contracts, year):
    """One year's covered parts, as rows in the order they add.

    Each row is (origin, event, mark member, occurrence scope, occurrence ID,
    contract, amount, outside): an episode's campaign mark (member -1) comes
    before its members, and each covered share's base part before its
    membership's and its contract's defense loadings.
    """
    rows = []
    for event in range(int(rng.integers(0, 4))):
        origin = ORIGINS[int(rng.integers(len(ORIGINS)))]
        marks = ([-1] if rng.random() < 0.5 else []) + list(
            range(int(rng.integers(0, 4)))
        )
        for member in marks:
            for _ in range(int(rng.integers(1, 4))):
                component = _value(rng, dyadic, scale=10)
                for number, contract in enumerate(contracts):
                    if rng.random() >= 0.6:
                        continue
                    scopes = ["campaign", "contract_year"] + (
                        ["member"] if member >= 0 else []
                    )
                    scope = scopes[int(rng.integers(len(scopes)))]
                    occurrence = (
                        None
                        if scope == "campaign" and rng.random() < 0.5
                        else ("x", "y")[int(rng.integers(2))]
                    )
                    covered = (
                        component * _fraction(rng, dyadic) * _fraction(rng, dyadic)
                    )
                    outside = bool(rng.random() < 0.2)
                    parts = [(outside, covered)]
                    loading = _value(rng, dyadic, scale=0.5, zero=0.6)
                    if contract.defense_loading == 0 and loading > 0:
                        parts.append((bool(rng.random() < 0.5), covered * loading))
                    if contract.defense_loading > 0 and not outside:
                        parts.append(
                            (
                                contract.defense_outside_limits,
                                covered * contract.defense_loading,
                            )
                        )
                    rows.extend(
                        (
                            origin,
                            (year, event),
                            member,
                            scope,
                            occurrence,
                            number,
                            value,
                            is_outside,
                        )
                        for is_outside, value in parts
                    )
    return rows


def _reduced(year_rows, contracts):
    """Every year's views from one reduction: each origin's events added in
    order, as one batch per origin."""
    reduction = PolicyReduction(
        contracts,
        len(year_rows),
        financial=True,
        occurrence_maximum=True,
        risk_units={origin: _risk_view(origin) for origin in ORIGINS},
    )
    for origin in ORIGINS:
        events: dict[tuple, int] = {}
        columns = defaultdict(list)
        for rows in year_rows:
            for row in rows:
                source, event, member, scope, occurrence, contract, amount, outside = (
                    row
                )
                if source != origin:
                    continue
                number = events.setdefault(event, len(events))
                columns["event"].append(number)
                columns["member"].append(member if scope == "member" else -1)
                columns["occurrence"].append(reduction.code(scope, occurrence))
                columns["contract"].append(contract)
                columns["contract_year"].append(scope == "contract_year")
                columns["amount"].append(amount)
                columns["outside"].append(outside)
        if not events:
            continue
        parts = CoveredParts(
            np.array(columns["event"], dtype=np.int64),
            np.array(columns["member"], dtype=np.int64),
            np.array(columns["occurrence"], dtype=np.int64),
            np.array(columns["contract"], dtype=np.int64),
            np.array(columns["contract_year"], dtype=bool),
            np.array(columns["amount"], dtype=float),
            np.array(columns["outside"], dtype=bool),
        )
        years = np.array([event[0] for event in events], dtype=np.int64)
        reduction.add(years, parts, origin=origin)
    return reduction.views()


def _tolerance(rows, contracts):
    """A bound on the rounding difference between left-to-right and exact
    sums of a year's nonnegative parts: each of the ``count`` sums, retention
    and limit steps adds at most one rounding of a quantity no larger than
    ``magnitude``, the year's covered parts and retentions together."""
    count = 4 * (len(rows) + len(contracts) + 1)
    magnitude = math.fsum(row[6] for row in rows) + math.fsum(
        (contract.occurrence_retention + 1) * (len(rows) + 1)
        + contract.annual_retention
        for contract in contracts
    )
    return 4 * count * UNIT_ROUNDOFF * magnitude


@pytest.mark.parametrize("dyadic", [True, False], ids=["dyadic", "float"])
def test_reduction_matches_the_exact_occurrence_oracle(dyadic):
    rng = np.random.default_rng(20261105 + dyadic)
    compared = binding = 0
    for _ in range(400):
        contracts = _contracts(rng, dyadic)
        year_rows = [
            _year(rng, dyadic, contracts, year)
            for year in range(int(rng.integers(1, 4)))
        ]
        reduced = _reduced(year_rows, contracts)
        for year, rows in enumerate(year_rows):
            expected = exact_policy_views(rows, contracts)
            tolerance = 0.0 if dyadic else _tolerance(rows, contracts)
            values = {
                "financial": expected["financial"],
                "oep_financial": expected["oep_financial"],
                **{
                    _risk_view(origin): exact_policy_views(rows, contracts, {origin})[
                        "occurrences"
                    ]
                    for origin in ORIGINS
                },
            }
            for view, value in values.items():
                actual = float(reduced[view][year])
                assert abs(actual - value) <= tolerance, (view, actual, value)
                compared += 1
            binding += 0 < expected["financial"] < expected["occurrences"]
    # The cases bind annual terms in some years and leave others unpaid.
    assert compared > 3000 and binding > 50


def _single_year(parts, contracts, **views):
    """One year's views of covered parts given as (event, occurrence code,
    contract, amount, outside) rows of one origin."""
    reduction = PolicyReduction(contracts, 1, **views)
    events, codes, numbers, amounts, outside = zip(*parts, strict=True)
    reduction.add(
        np.zeros(max(events) + 1, dtype=np.int64),
        CoveredParts(
            np.array(events, dtype=np.int64),
            np.full(len(parts), -1, dtype=np.int64),
            np.array(codes, dtype=np.int64),
            np.array(numbers, dtype=np.int64),
            np.zeros(len(parts), dtype=bool),
            np.array(amounts, dtype=float),
            np.array(outside, dtype=bool),
        ),
        origin="origin",
    )
    return {view: float(values[0]) for view, values in reduction.views().items()}


def test_outside_defense_survives_the_occurrence_limit():
    """80 of indemnity under a limit of 50 and 20 of outside-limit defense
    pay 70: the defense bypasses the occurrence limit."""
    contract = ContractTerms(
        contract_id="policy",
        contract_year="2027",
        entity_id="insured",
        occurrence_limit=50,
    )
    views = _single_year(
        [(0, 0, 0, 80.0, False), (0, 0, 0, 20.0, True)],
        [contract],
        financial=True,
        occurrence_maximum=True,
    )
    assert views == {"financial": 70.0, "oep_financial": 70.0}


def test_contracts_sharing_an_occurrence_add_before_its_maximum():
    """Two independent contracts pay one occurrence's 60 and 30 after a
    retention of 10 each: oep_financial takes their sum, 70, and financial
    pays every occurrence of both, uncoordinated."""
    contracts = [
        ContractTerms(
            contract_id=name,
            contract_year="2027",
            entity_id="insured",
            occurrence_retention=10,
        )
        for name in ("primary", "excess")
    ]
    views = _single_year(
        [(0, 0, 0, 60.0, False), (0, 0, 1, 30.0, False), (1, 0, 0, 15.0, False)],
        contracts,
        financial=True,
        occurrence_maximum=True,
    )
    assert views == {"financial": 50.0 + 20.0 + 5.0, "oep_financial": 70.0}


def test_overflowing_parts_reach_infinity():
    """A sum beyond the numeric range overflows to infinity, which the
    simulation refuses as an annual loss beyond the numeric range
    (``_assemble_simulation``)."""
    contract = ContractTerms(
        contract_id="policy", contract_year="2027", entity_id="insured"
    )
    views = _single_year(
        [(0, 0, 0, 1e308, False), (0, 0, 0, 1e308, False)],
        [contract],
        financial=True,
        occurrence_maximum=True,
    )
    assert views == {"financial": math.inf, "oep_financial": math.inf}
