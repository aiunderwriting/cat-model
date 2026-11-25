"""Report-only plausibility checks of a run's results (``NUM-PLAUSIBILITY``).

``plausibility_table`` gives each leaf's welfare EAL, and the total, as a share
of one governed reference value, world output for the study year, authored
once in ``data/reference/world_output.yaml``, and flags shares above
thresholds set by judgment. ``large_incident_table`` reads the expected annual
numbers of incidents at or above the large-loss level from the
``large_loss_anchor`` exhibit and flags counts under the run's laws that reach
``LARGE_INCIDENT_FLAGS``. Neither changes a result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from catmodel.probability.numerics import at_least
from catmodel.resources import data_path
from catmodel.yaml_io import safe_load


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorldOutput(_Record):
    measure: str
    year: int
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    value: float = Field(gt=0, allow_inf_nan=False)
    basis: str
    status: Literal["OK", "INITIAL_ESTIMATE", "PLACEHOLDER"]
    citation: str


class PlausibilityFlags(_Record):
    leaf_share: float = Field(gt=0, lt=1)
    total_share: float = Field(gt=0, lt=1)
    status: Literal["OK", "INITIAL_ESTIMATE", "PLACEHOLDER"]


class PlausibilityReference(_Record):
    world_output: WorldOutput
    flags: PlausibilityFlags


@cache
def plausibility_reference() -> PlausibilityReference:
    """The governed world output reference and flag thresholds."""
    path = data_path("reference", "world_output.yaml")
    return PlausibilityReference.model_validate(
        safe_load(path.read_text(encoding="utf-8"))
    )


def _row(metric: dict[str, Any] | None, output: float, flag: float) -> dict[str, Any]:
    if metric is None or metric.get("status") != "available":
        return {
            "status": "unavailable",
            "reason": "The run reports no available welfare EAL for it.",
        }
    eal = metric.get("expected_annual_loss")
    if not isinstance(eal, int | float):
        return {"status": "unavailable", "reason": "Its welfare EAL is unavailable."}
    share = eal / output
    return {
        "status": "available",
        "expected_annual_loss": eal,
        "eal_estimator": metric.get("eal_estimator"),
        "share_of_world_output": share,
        "flagged": share > flag,
    }


def plausibility_table(
    package, metrics: dict[str, Any], reference: PlausibilityReference | None = None
) -> dict[str, Any]:
    """The run's welfare EAL per scenario and in total as shares of world output.

    Each scenario is one leaf. A leaf share above ``flags.leaf_share`` and a
    total share above ``flags.total_share`` are flagged.
    """
    reference = reference or plausibility_reference()
    output, flags = reference.world_output, reference.flags
    currencies = {population.currency for population in package.scope.populations}
    base: dict[str, Any] = {
        "reference": output.model_dump(mode="json"),
        "flags": flags.model_dump(mode="json"),
        "view": "welfare",
        "interpretation": "Report only: each scenario's welfare EAL, one per leaf, and the total, divided by world output for the study year, with leaf shares above flags.leaf_share and a total above flags.total_share flagged for review. The reference value and both thresholds are governed judgments; a flag is a prompt to examine the judgments behind an EAL, not evidence that it is wrong, and no result changes.",
    }
    if currencies != {output.currency}:
        return base | {
            "status": "unavailable",
            "reason": f"The populations' currencies {sorted(currencies)} differ from the reference currency {output.currency}.",
        }
    year = package.scope.period_start.year
    by_scenario = {
        population.scenario_id: _row(
            metrics.get(f"scenario::{population.scenario_id}::welfare"),
            output.value,
            flags.leaf_share,
        )
        for population in package.scope.populations
    }
    total = _row(metrics.get("welfare"), output.value, flags.total_share)
    return base | {
        "status": "available",
        "study_year": year,
        "reference_year_note": None
        if year == output.year
        else f"The study period starts in {year}, but the reference is world output in {output.year}.",
        "total": total,
        "by_scenario": dict(sorted(by_scenario.items())),
        "flagged_scenarios": sorted(
            scenario_id
            for scenario_id, row in by_scenario.items()
            if row.get("flagged")
        ),
    }


class LargeIncidentFlags(_Record):
    leaf_annual_count: float = Field(gt=0, allow_inf_nan=False)
    total_annual_count: float = Field(gt=0, allow_inf_nan=False)
    status: Literal["OK", "INITIAL_ESTIMATE", "PLACEHOLDER"]


LARGE_INCIDENT_FLAGS = LargeIncidentFlags(
    leaf_annual_count=0.1, total_annual_count=1.0, status="INITIAL_ESTIMATE"
)
"""The expected annual numbers of incidents at or above the large-loss level,
under the run's laws, at which a leaf and the total are flagged for review
(``NUM-PLAUSIBILITY``).

Review triggers set by judgment, not calibrated: an expected count of one or more
such incidents in one lab-year is a strong statement that can be checked
against how many incidents of that size the public record attributes to any
one developer's models. A flag changes no result.
"""

LARGE_INCIDENT_STATISTICS = (
    "stated_large_count",
    "pooled_large_count",
    "modelled_large_count",
    "probability_at_least_one_large",
)
"""The fields ``large_incident_table`` reads from each record of the
``large_loss_anchor`` exhibit: the expected annual number of incidents at or
above the large-loss level as the respondents stated it, after pooling and
under the run's laws, and the probability of at least one such incident in
the year under the run's laws."""

_NO_VALUE = "The large-loss anchor exhibit reports no value for it."


@dataclass(frozen=True)
class LargeIncidentRow:
    """One scenario's row of the large-incident table, or the total's, whose
    ``scenario_id`` is None.

    ``values`` holds each of ``LARGE_INCIDENT_STATISTICS``, None where the
    exhibit reports no value. ``unavailable`` is the reason a row has no value
    at all, and ``missing`` the reason for each value an available row lacks.
    ``flagged`` marks a count under the run's laws that reaches its threshold.
    """

    scenario_id: str | None
    values: Mapping[str, float | None]
    flagged: bool
    unavailable: str | None
    missing: Mapping[str, str]

    def reason(self, statistic: str) -> str | None:
        """Why the row has no value for ``statistic``; None where it has one."""
        return self.unavailable or self.missing.get(statistic)


@dataclass(frozen=True)
class LargeIncidentTable:
    """The total's row and each scenario's, with the thresholds that set
    their flags. Scenarios are in decreasing order of their count under the
    run's laws, those without one last."""

    flags: LargeIncidentFlags
    total: LargeIncidentRow
    scenarios: tuple[LargeIncidentRow, ...]


def _number(value: object) -> float | None:
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return float(value)
    return None


def _large_incident_row(
    scenario_id: str | None, record: Mapping[str, Any], threshold: float
) -> LargeIncidentRow:
    """The row of one exhibit record, flagged where its count under the run's
    laws is at least ``threshold``. A record that is not available, the
    exhibit's own when it is withheld, gives a row without values."""
    status = record.get("status")
    if status != "available":
        reason = (
            "No scenario has a large-loss anchor."
            if status == "inapplicable"
            else "; ".join(record.get("reasons") or ())
            or record.get("reason")
            or _NO_VALUE
        )
        return LargeIncidentRow(
            scenario_id, dict.fromkeys(LARGE_INCIDENT_STATISTICS), False, reason, {}
        )
    values = {name: _number(record.get(name)) for name in LARGE_INCIDENT_STATISTICS}
    missing = {name: _NO_VALUE for name, value in values.items() if value is None}
    if "probability_at_least_one_large" in missing:
        withheld = "; ".join(record.get("probability_withheld_reasons") or ())
        missing["probability_at_least_one_large"] = withheld or _NO_VALUE
    count = values["modelled_large_count"]
    return LargeIncidentRow(
        scenario_id,
        values,
        count is not None and at_least(count, threshold),
        None,
        missing,
    )


def large_incident_table(exhibit: Mapping[str, Any]) -> LargeIncidentTable:
    """The ``large_loss_anchor`` exhibit's expected annual numbers of incidents
    at or above the large-loss level, for each scenario it reports and in
    total, with review flags.

    A scenario whose count under the run's laws is at least
    ``LARGE_INCIDENT_FLAGS.leaf_annual_count``, and a total of at least
    ``LARGE_INCIDENT_FLAGS.total_annual_count``, is flagged; a computed count
    within the tie tolerance below a threshold reaches it
    (``catmodel.probability.numerics.at_least``). An exhibit withheld for a
    reweighted result has only a total row, without values and with the
    exhibit's reason.
    """
    flags = LARGE_INCIDENT_FLAGS
    if "by_scenario" not in exhibit:
        return LargeIncidentTable(
            flags, _large_incident_row(None, exhibit, flags.total_annual_count), ()
        )

    def order(row: LargeIncidentRow) -> tuple[bool, float, str]:
        count = row.values["modelled_large_count"]
        return count is None, -(count or 0.0), row.scenario_id or ""

    return LargeIncidentTable(
        flags,
        _large_incident_row(None, exhibit["total"], flags.total_annual_count),
        tuple(
            sorted(
                (
                    _large_incident_row(scenario_id, record, flags.leaf_annual_count)
                    for scenario_id, record in exhibit["by_scenario"].items()
                ),
                key=order,
            )
        ),
    )
