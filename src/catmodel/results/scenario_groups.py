"""Scenario EALs grouped by reporting domain and by facet.

The taxonomy counts each incident once, in one scenario. A reporting domain or
a facet, such as the party basis, labels whole scenarios, so a group's EAL is
the sum of its scenarios' EALs. Nothing is elicited or simulated for a group:
the exhibit only adds up ``scenario_contributions``. A group therefore cannot
show a threat that is part of a scenario or spread across several, such as
ransomware among the system-compromise scenarios.
"""

import math
from collections.abc import Iterable, Mapping
from functools import cache
from typing import Any

from catmodel.resources import scenarios_path
from catmodel.scenarios.facets import SCENARIO_GROUP_AXES, scenario_group_values
from catmodel.scenarios.loader import load_scenarios

type Classification = Mapping[str, Mapping[str, str]]
"""Each classified scenario's value on every grouping axis."""


@cache
def _library_classification() -> dict[str, dict[str, str]]:
    return {
        scenario.id: scenario_group_values(scenario)
        for scenario in load_scenarios(scenarios_path())
    }


def library_classification(scenario_ids: Iterable[str]) -> Classification:
    """The scenario library's classification of those of ``scenario_ids`` it
    holds. A setup's other scenarios, such as a stipulated example's, have no
    reporting domain or facets."""
    library = _library_classification()
    return {key: library[key] for key in scenario_ids if key in library}


def scenario_groups(
    contributions: Mapping[str, Any], classification: Classification
) -> dict[str, Any]:
    """Group ``contributions`` (the ``scenario_contributions`` section) on
    every axis of ``SCENARIO_GROUP_AXES``.

    A group's EAL is the sum of its scenarios' available EALs. It is a
    ``lower_bound`` when some of its scenarios have none, and unavailable when
    none has one. Every classified scenario is in one group of each axis, so
    with every EAL available the groups of an axis add up to the classified
    scenarios' EAL. A setup with no library scenario, such as a stipulated
    example, has no groups: the exhibit is ``inapplicable``.
    """
    if contributions.get("status") != "available":
        return {"status": "unavailable", "reason": contributions["reason"]}
    rows = {row["scenario_id"]: row for row in contributions["scenarios"]}
    classified = [key for key in rows if key in classification]
    if not classified:
        return {
            "status": "inapplicable",
            "reason": "No scenario of this setup is a scenario of the library, "
            "so none has a reporting domain or facets.",
        }
    total = contributions["expected_annual_loss"]
    axes = []
    for axis in SCENARIO_GROUP_AXES:
        members: dict[str, list[str]] = {}
        for key in classified:
            members.setdefault(classification[key][axis], []).append(key)
        groups = [_group(value, keys, rows, total) for value, keys in members.items()]
        # Decreasing EAL, then the groups without one.
        groups.sort(
            key=lambda group: (
                group["expected_annual_loss"] is None,
                -(group["expected_annual_loss"] or 0.0),
                group["value"],
            )
        )
        axes.append({"axis": axis, "groups": groups})
    return {
        "status": "available",
        "measure": contributions["measure"],
        "expected_annual_loss": total,
        "axes": axes,
        "unclassified_scenarios": sorted(set(rows) - set(classified)),
        "limitations": "A group's EAL adds up the EALs of its scenarios; no "
        "tail statistic is grouped. A group holds whole scenarios, so it "
        "cannot show a threat that is part of a scenario or spread across "
        "several. The grouping is the scenario library's at the time of the "
        "reduction.",
    }


def _group(
    value: str,
    scenario_ids: list[str],
    rows: Mapping[str, Mapping[str, Any]],
    total: float | None,
) -> dict[str, Any]:
    available = [
        rows[key]["expected_annual_loss"]
        for key in scenario_ids
        if rows[key]["status"] == "available"
    ]
    if not available:
        status, eal = "unavailable", None
        reason = "None of the group's scenarios has an EAL."
    elif len(available) < len(scenario_ids):
        status, eal = "lower_bound", math.fsum(available)
        reason = "The sum leaves out the group's scenarios without an EAL."
    else:
        status, eal, reason = "available", math.fsum(available), None
    return {
        "value": value,
        "scenario_ids": sorted(scenario_ids),
        "status": status,
        "reason": reason,
        "expected_annual_loss": eal,
        "share_of_total": eal / total if eal is not None and total else None,
    }
