"""Each scenario's EAL and its share of the pooled EAL.

methodology.md ("Scenario contributions") gives the method.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from catmodel.contracts.results import NativeResults
from catmodel.contracts.run_setup import RunSetup
from catmodel.results.respondent_sampling import ViewWeights
from catmodel.results.sensitivity_summary import scenario_view
from catmodel.simulation.capability_mixture import ConditionalCapabilityDraws

_MEASURES = ("welfare", "gross", "net_resource_cost")


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def view_reason(metric: Mapping[str, Any] | None, statistic: str) -> str:
    """The specific reason a view or one of its statistics is missing."""
    if metric is None:
        return "The view was not requested."
    if metric.get("status") != "available":
        states = metric.get("state_reasons") or {}
        specific = "; ".join(dict.fromkeys(states.values()))
        return specific or metric.get("reason") or "The view is unavailable."
    status = (metric.get("statistics") or {}).get(statistic) or {}
    return status.get("reason") or "No finite value is reported."


def _measure(metrics: Mapping[str, Any], scenario_ids: Sequence[str]) -> str | None:
    for measure in _MEASURES:
        if measure in metrics or any(
            scenario_view(scenario, measure) in metrics for scenario in scenario_ids
        ):
            return measure
    return None


def scenario_contributions(
    package: RunSetup,
    report: NativeResults,
    draws: Mapping[str, ConditionalCapabilityDraws],
    weights: Mapping[str, float],
    *,
    view_weights: ViewWeights | None = None,
) -> dict[str, Any]:
    """Per-scenario EAL and share of the pooled EAL in one additive measure,
    and the pooled EAL no scenario carries. ``weights`` are the reference
    arm's and ``view_weights`` its weights in each view, where some view's
    are not ``weights``: the measure's states are those it weighs, so the
    gross measure reconciles over the states that give the lab's loss
    (``ViewWeights``)."""
    metrics: Mapping[str, Any] = report["arms"][package.execution.reference_arm].get(
        "metrics", {}
    )
    scenario_ids = [population.scenario_id for population in package.scope.populations]
    measure = _measure(metrics, scenario_ids)
    if measure is None:
        return {
            "status": "unavailable",
            "reason": "No pooled or per-scenario additive view (welfare, gross or net_resource_cost) was requested.",
        }
    total_metric = metrics.get(measure)
    total = (total_metric or {}).get("expected_annual_loss")
    total = total if _finite(total) else None
    scenarios: list[dict[str, Any]] = []
    for scenario in scenario_ids:
        name = scenario_view(scenario, measure)
        metric = metrics.get(name)
        eal = (metric or {}).get("expected_annual_loss")
        if metric is None:
            status = "not_requested"
        elif metric.get("status") == "available" and _finite(eal):
            status = "available"
        else:
            status = "unavailable"
        scenarios.append(
            {
                "scenario_id": scenario,
                "view": name,
                "status": status,
                "reason": None
                if status == "available"
                else view_reason(metric, "expected_annual_loss"),
                "expected_annual_loss": eal if status == "available" else None,
                "share_of_total": eal / total
                if status == "available" and total
                else None,
            }
        )
    return {
        "status": "available",
        "measure": measure,
        "expected_annual_loss": total,
        "total_reason": None
        if total is not None
        else view_reason(total_metric, "expected_annual_loss"),
        "scenarios": scenarios,
        "unallocated": _unallocated(
            scenarios,
            total,
            metrics,
            measure,
            draws,
            weights if view_weights is None else view_weights.of(measure) or {},
        ),
        "limitations": "Shares divide a scenario's EAL by the pooled EAL when it is available. With every scenario's EAL available, the scenario EALs and the unallocated amount add up to the pooled EAL.",
    }


def _unallocated(
    scenarios: Sequence[Mapping[str, Any]],
    total: float | None,
    metrics: Mapping[str, Any],
    measure: str,
    draws: Mapping[str, ConditionalCapabilityDraws],
    weights: Mapping[str, float],
) -> dict[str, Any]:
    """Pooled EAL not carried by any scenario, and whether years reconcile."""
    if total is None or any(row["status"] != "available" for row in scenarios):
        return {
            "expected_annual_loss": None,
            "reason": "Needs the pooled EAL and every scenario's EAL.",
            "annual_losses_reconcile": None,
        }
    remainder = total - math.fsum(row["expected_annual_loss"] for row in scenarios)
    views = [row["view"] for row in scenarios]
    reconcile = all(
        np.allclose(
            sum(
                (draw.annual_losses[name] for name in views),
                start=np.zeros(draw.n_simulations),
            ),
            draw.annual_losses[measure],
            rtol=1e-9,
            atol=0,
        )
        for state, draw in draws.items()
        if weights[state] > 0
        and measure in draw.views
        and all(name in draw.views for name in views)
    )
    return {
        "expected_annual_loss": remainder,
        "share_of_total": remainder / total if total else None,
        "reason": None
        if reconcile
        else "Scenario annual losses do not sum to the pooled view in every simulated year; populations overlap or the pooled view includes losses outside them.",
        "annual_losses_reconcile": reconcile,
    }
