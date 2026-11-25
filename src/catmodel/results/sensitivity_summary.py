"""Compact EAL exhibits derived from an existing reduction, without simulation."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Mapping
from html import escape
from typing import Any

from catmodel.contracts.incident import parse_scenario_view
from catmodel.results.reduction import view_in_scope
from catmodel.setup.sensitivity_catalog import SENSITIVITY_FAMILIES

_FAMILY_KINDS = {family.family_id: family.kind for family in SENSITIVITY_FAMILIES}
_VIEW_PREFERENCE = ("welfare", "gross", "net_resource_cost")
EXTRAPOLATED_ARMS = frozenset({"epistemic_spread_widened", "epistemic_tail_widened"})
EXTRAPOLATION_NOTE = (
    "The largest effect is a widened epistemic spread or tail arm. Its EAL "
    "integrates each count law at stretched normal scores or widened tail "
    "scores. For wide count laws most of that mean is read beyond the elicited "
    "p95 (POOL-SPREAD, POOL-PCHIP): it answers how far respondent "
    "overconfidence of that size would move the EAL, dominated by judged tail "
    "extrapolation."
)


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _eal(metric: Mapping[str, Any]) -> dict[str, Any]:
    value = metric.get("expected_annual_loss")
    available = metric.get("status") == "available" and _finite(value)
    return {
        "status": "available" if available else "unavailable",
        "expected_annual_loss": value if available else None,
        "basis": metric.get("basis"),
        "reason": None
        if available
        else (metric.get("statistics") or {})
        .get("expected_annual_loss", {})
        .get("reason")
        or metric.get("reason")
        or "No finite expected annual loss is available for this view.",
    }


def _assessed_for_view(requirement: Mapping[str, Any], view: str) -> bool:
    """Mirror the reduction's per-view ranking completeness for one requirement."""
    return requirement["outcome_status"] in {"available", "inapplicable"} or (
        requirement.get("status") == "required"
        and requirement.get("view_outcomes", {}).get(view, {}).get("status")
        == "available"
    )


def _excluded_reason(
    arm: str,
    *,
    purpose: str,
    controls: set[str],
    families: set[str],
) -> str | None:
    if purpose != "scientific_alternative":
        return f"{purpose.replace('_', ' ').capitalize()} arm; not an assumption challenge."
    if arm in controls:
        return "Matched control; its named stresses are measured against it."
    kinds = sorted({_FAMILY_KINDS.get(family, "assumption") for family in families})
    if kinds != ["assumption"] and families:
        return f"Catalog {' and '.join(kinds)} family; not an assumption challenge."
    return None


def headline_view(metrics: Mapping[str, Any]) -> str | None:
    """The first available additive view, preferring societal harm.

    Selection is by availability, not by whether the view's mean exists.
    """
    aggregates = sorted(view for view in metrics if not view.startswith("scenario::"))
    return next(
        (
            view
            for view in (*_VIEW_PREFERENCE, *aggregates)
            if metrics.get(view, {}).get("status") == "available"
        ),
        None,
    )


def scenario_view(scenario_id: str, view: str) -> str:
    """The per-scenario view of an aggregate view."""
    return f"scenario::{scenario_id}::{view}"


def sensitivity_summary(
    report: Mapping[str, Any],
    *,
    reference_arm: str,
    purposes: Mapping[str, str],
) -> dict[str, Any]:
    """Show executed EAL effects of assumption challenges and what is missing.

    The view is ``headline_view``'s. Each scientific alternative is compared
    with its matched control, or with the reference when it has none; numerical
    refinements, matched controls and catalog diagnostics are listed but not
    ranked. The reduction's own ranking status and the families not assessed
    for the view are carried beside the rows, so the exhibit cannot read as
    complete when ``results.json`` is not.
    """
    arms = report["arms"]
    metrics = arms[reference_arm].get("metrics", {})
    scenarios: dict[str, dict[str, Any]] = {}
    aggregates = {}
    for view, metric in sorted(metrics.items()):
        scoped = parse_scenario_view(view)
        if scoped is not None:
            scenario, measure = scoped
            scenarios.setdefault(scenario, {})[measure] = _eal(metric)
        else:
            aggregates[view] = _eal(metric)
    selected = headline_view(metrics)
    requirements = report.get("comparison_requirements", ())
    families_by_arm: dict[str, set[str]] = {}
    for requirement in requirements:
        if requirement.get("arm_id") is not None:
            families_by_arm.setdefault(requirement["arm_id"], set()).add(
                requirement["family_id"]
            )
    control_by_arm = report.get("comparison_reference_arms", {})
    controls = set(control_by_arm.values()) - {reference_arm}
    rows, unavailable, excluded = [], {}, {}
    for arm in sorted(set(arms) - {reference_arm}):
        reason = _excluded_reason(
            arm,
            purpose=purposes.get(arm, "scientific_alternative"),
            controls=controls,
            families=families_by_arm.get(arm, set()),
        )
        if reason is not None:
            excluded[arm] = reason
            continue
        control = control_by_arm.get(arm, reference_arm)
        contrast = (
            report.get("comparisons", {}).get(arm, {}).get(selected, {})
            if selected is not None
            else {}
        )
        effect = contrast.get("expected_annual_loss", {})
        # Each EAL effect pairs both arms over their common years, the shorter
        # arm's, so its control EAL is the control's on those years where it
        # is empirical; one the tail gate withheld is not ranked.
        if effect.get("status") != "withheld" and all(
            _finite(effect.get(field))
            for field in ("reference", "alternative", "absolute_difference")
        ):
            difference = effect["absolute_difference"]
            rows.append(
                {
                    "arm": arm,
                    "control_arm": control,
                    "control_eal": effect["reference"],
                    "alternative_eal": effect["alternative"],
                    "absolute_difference": difference,
                    "relative_change": effect.get("relative_change"),
                    "mean_changing": difference != 0,
                    "half_width": effect.get("half_width"),
                }
            )
        else:
            unavailable[arm] = (
                effect.get("reason")
                or contrast.get("reason")
                or arms[arm].get("reason")
                or "No finite EAL contrast is available for the selected view."
            )
    rows.sort(key=lambda row: (-abs(row["absolute_difference"]), row["arm"]))
    leading = max(rows, key=lambda row: abs(row["absolute_difference"]), default=None)
    gaps = [
        {
            key: requirement.get(key)
            for key in (
                "family_id",
                "arm_id",
                "scenario_ids",
                "status",
                "outcome_status",
                "reason",
                "outcome_reason",
            )
        }
        for requirement in requirements
        if requirement["outcome_status"] not in {"available", "inapplicable"}
    ]
    not_assessed = (
        sorted(
            {
                requirement["family_id"]
                for requirement in requirements
                if view_in_scope(
                    selected,
                    requirement["scenario_ids"],
                    requirement.get("affected_views"),
                )
                and not _assessed_for_view(requirement, selected)
            }
        )
        if selected is not None
        else []
    )
    ranking = report.get("rankings", {}).get(selected, {}) if selected else {}
    ranking = ranking.get("expected_annual_loss", ranking)
    return {
        "analysis_sha256": report.get("analysis_sha256"),
        "reference_arm": reference_arm,
        "comparison_status": report["coverage"]["comparisons"]["status"],
        "scenario_eal": scenarios,
        "aggregate_eal": aggregates,
        "comparison_gaps": gaps,
        "tornado": {
            "view": selected,
            "basis": aggregates.get(selected, {}).get("basis"),
            "eal": aggregates.get(selected),
            "rows": rows,
            "extrapolation_note": EXTRAPOLATION_NOTE
            if leading is not None and leading["arm"] in EXTRAPOLATED_ARMS
            else None,
            "unavailable_arms": unavailable,
            "excluded_arms": excluded,
            "not_assessed_families": not_assessed,
            "results_ranking": {
                "status": ranking.get("status", "unavailable"),
                "reason": ranking.get("reason")
                or (
                    None
                    if ranking
                    else "results.json has no EAL ranking for this view."
                ),
            },
            "interpretation": "Executed EAL point effects of assumption challenges, each against its matched control or the reference, ranked by absolute size; nonzero differences change the mean. extrapolation_note flags a leading widened epistemic spread or tail arm. Numerical refinements, matched controls and catalog diagnostics are not ranked. Coverage is complete only when results_ranking is available; not_assessed_families names the catalog families and register challenges with no assessed comparison for this view. The rows are not an uncertainty interval or a precision ordering.",
        },
    }


def tornado_svg(summary: Mapping[str, Any]) -> str:
    """Render the finite contrasts and coverage already in ``sensitivity_summary``."""
    tornado = summary["tornado"]
    rows = tornado["rows"]
    width, left, center, scale_width = 1100, 24, 740, 230
    maximum = max((abs(row["absolute_difference"]) for row in rows), default=0)
    basis = tornado.get("basis") or {}
    unit = basis.get("currency", "selected monetary unit")
    title = f"EAL sensitivity — {tornado.get('view') or 'unavailable view'} ({unit})"
    ranking = tornado["results_ranking"]
    missing = tornado["not_assessed_families"]
    notes = [
        line
        for paragraph in (
            "Complete comparison coverage: results.json ranks these EAL effects."
            if ranking["status"] == "available"
            else f"Incomplete coverage; results.json gives no EAL ranking: {ranking['reason']}",
            f"Not assessed for this view ({len(missing)} families or register challenges): "
            + (", ".join(missing) or "none"),
            "Bars show executed changes in the mean against each arm's control; no uncertainty interval is implied.",
        )
        for line in textwrap.wrap(paragraph, 140)
    ]
    top = 70 + 18 * len(notes)
    height = top + 60 + 34 * max(1, len(rows))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        f'<title id="title">{escape(title)}</title>',
        '<desc id="description">Executed changes in expected annual loss, ranked by absolute point effect, with the comparison families not assessed. Negative changes extend left and positive changes extend right.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" font-size="13" fill="#182638">',
        f'<text x="{left}" y="30" font-size="20">{escape(title)}</text>',
        *(
            f'<text x="{left}" y="{55 + 18 * index}">{escape(note)}</text>'
            for index, note in enumerate(notes)
        ),
        f'<line x1="{center}" x2="{center}" y1="{top}" y2="{height - 40}" stroke="#8c98a6"/>',
    ]
    if not rows:
        svg.append(
            f'<text x="{left}" y="{top + 32}">No finite EAL contrasts are available.</text>'
        )
    for index, row in enumerate(rows):
        y = top + 18 + 34 * index
        value = row["absolute_difference"]
        length = (abs(value) / maximum) * scale_width if maximum else 0
        x = center - length if value < 0 else center
        color = "#287c8e" if value < 0 else "#bd573a"
        svg.extend(
            (
                f'<text x="{left}" y="{y + 14}">{escape(row["arm"])}</text>',
                f'<rect x="{x:.3f}" y="{y}" width="{length:.3f}" height="20" fill="{color}"/>',
                f'<text x="{width - 20}" y="{y + 14}" text-anchor="end">{value:+.5g}</text>',
            )
        )
    svg.extend(("</g>", "</svg>"))
    return "\n".join(svg) + "\n"
