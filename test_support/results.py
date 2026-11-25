"""Availability assertions shared by native-result tests."""

from collections.abc import Mapping
from typing import Any, cast

from catmodel.contracts.results import (
    ArmResult,
    AvailableViewResult,
    ComparedValue,
    ReducedArmResult,
    UnavailableComparison,
    UnavailableEffect,
    ViewDiagnostics,
    ViewResult,
    WithheldExhibit,
)


def view_diagnostics(report: Mapping[str, Any], arm: str, view: str) -> ViewDiagnostics:
    """One reduced view's diagnostics, which a report keeps beside its metrics."""
    return report["diagnostics"]["arms"][arm]["views"][view]


def statistic_ranking(rankings: Mapping[str, Any], view: str, statistic: str):
    """A view's ranking of one statistic, or the view's single disposition
    when every statistic shares it."""
    record = rankings[view]
    return record if "status" in record else record[statistic]


def reduced_arm(result: ArmResult) -> ReducedArmResult:
    assert result["status"] != "unavailable", result
    return result


def available_metric(result: ViewResult) -> AvailableViewResult:
    assert result["status"] == "available", result
    return result


def present[T](value: T | None) -> T:
    assert value is not None
    return value


def available_mapping[T](
    result: dict[str, T] | UnavailableComparison,
) -> dict[str, T]:
    assert result.get("status") != "unavailable", result
    return cast(dict[str, T], result)


def compared_value(result: ComparedValue | UnavailableEffect) -> ComparedValue:
    assert "reference" in result and "alternative" in result, result
    return cast(ComparedValue, result)


def reported_exhibit[T](record: T | WithheldExhibit) -> T:
    """An exhibit that rests on the setup's own weights, as a result at those
    weights reports it rather than withholds it."""
    assert cast(Mapping[str, object], record).get("status") != "unavailable", record
    return cast(T, record)
