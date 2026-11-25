"""Functional reduction of frozen conditional captures under explicit target weights.

An arm weighs its states alike in every view but those that count the lab's
own loss, its lab-loss views and its gross views. Those give a state left out
of them weight zero and its sampling group's weight to the group's other
states (``respondent_sampling.ViewWeights``, ``POOL-EXCLUDED-SAMPLES``). Each
view's metrics, basis, withheld statistics and epistemic-group EALs are
reduced at that view's weights.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, cast

import numpy as np
from scipy.stats import t as student_t

from catmodel.contracts.incident import ComparisonRequirement, parse_scenario_view
from catmodel.contracts.results import (
    AnalysisSpec,
    ArmDiagnostics,
    ArmResult,
    CaptureProvenance,
    CodeProvenance,
    ComparisonRequirementRecord,
    ComparisonResult,
    DiagnosticLimitations,
    FamilyCoverage,
    FamilyKind,
    ReducedArmResult,
    ReducedResults,
    ReductionSpecRecord,
    ViewCoverage,
)
from catmodel.hashing import canonical_json_sha256
from catmodel.probability.numerics import valid_probability_weights
from catmodel.results.respondent_sampling import SampleGroups, ViewWeights
from catmodel.setup.sensitivity_catalog import SENSITIVITY_FAMILIES
from catmodel.simulation.capability_mixture import (
    CapabilityMixture,
    ConditionalCapabilityDraws,
    ConditionalView,
    TailMeanEstimate,
    common_coefficient,
    design_precision_unavailable_reason,
    effective_sample_size,
    known_zero_law,
    statistic_status,
    tail_estimates,
)
from catmodel.simulation.tail_gate import (
    EAL_RECONCILIATION_LIMIT,
    MIN_TAIL_EFFECTIVE_UNITS,
    TailGateControls,
    paired_effect,
    unconverged_reason,
    view_gate,
    year_prefix,
)

# Annual non-exceedance levels reported by default: the 1-in-10, 1-in-100,
# 1-in-200, 1-in-250 and 1-in-1,000 year losses.
DEFAULT_TAIL_LEVELS = (0.9, 0.99, 0.995, 0.996, 0.999)


@dataclass(frozen=True)
class ReductionSpec:
    levels: tuple[float, ...] = DEFAULT_TAIL_LEVELS
    atom_convention: Literal["left_inverse_fractional_atom"] = (
        "left_inverse_fractional_atom"
    )
    required_grain: Literal["annual"] = "annual"
    joint_thresholds: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.levels
            or len(set(self.levels)) != len(self.levels)
            or any(not math.isfinite(p) or not 0 < p < 1 for p in self.levels)
        ):
            raise ValueError(
                "reduction tail levels must be distinct probabilities in (0,1)"
            )
        if self.atom_convention != "left_inverse_fractional_atom":
            raise ValueError("unsupported quantile/TVaR atom convention")
        if self.required_grain != "annual":
            raise ValueError("unsupported reduction capture grain")
        if any(
            not name or not math.isfinite(value)
            for name, value in self.joint_thresholds
        ):
            raise ValueError(
                "joint exceedances require named views and finite thresholds"
            )


def comparison(reference: float, alternative: float) -> ComparisonResult:
    """Absolute and relative change, each ``None`` with a reason when unavailable."""
    if not math.isfinite(reference) or not math.isfinite(alternative):
        raise ValueError("comparisons require finite reference and alternative values")
    absolute = alternative - reference
    if not math.isfinite(absolute):
        absolute_value, absolute_reason = (
            None,
            "The absolute difference exceeds finite numeric range.",
        )
    else:
        absolute_value, absolute_reason = absolute, None
    if reference == 0:
        relative, reason = None, "Relative change is undefined at a zero reference."
    else:
        relative = absolute / reference
        reason = None
        if not math.isfinite(relative):
            relative, reason = (
                None,
                "Relative change exceeds finite numeric range at this reference.",
            )
    return {
        "reference": reference,
        "alternative": alternative,
        "absolute_difference": absolute_value,
        "absolute_difference_unavailable_reason": absolute_reason,
        "relative_change": relative,
        "relative_change_unavailable_reason": reason,
    }


def _draw_digest(draw: ConditionalCapabilityDraws) -> str:
    def array_digest(values: np.ndarray) -> str:
        return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()

    return canonical_json_sha256(
        {
            "stratum": draw.stratum_sha256,
            "joint_law_semantics": draw.joint_law_semantics,
            "evaluator_provenance_by_risk_unit": None
            if draw.evaluator_provenance_by_risk_unit is None
            else {
                k: dict(v) for k, v in draw.evaluator_provenance_by_risk_unit.items()
            },
            "response": draw.response_sha256,
            "seed": draw.seed,
            "views": {
                name: {
                    "array": array_digest(view.array),
                    "mean_exists": view.mean_exists,
                    "variance_exists": view.variance_exists,
                    "native_mean": None
                    if view.native_mean is None
                    else {
                        "value": view.native_mean.value,
                        "estimator": view.native_mean.estimator,
                        "precision": None
                        if view.native_mean.precision is None
                        else dict(view.native_mean.precision),
                    },
                }
                for name, view in sorted(draw.views.items())
            },
            "n_simulations": draw.n_simulations,
        }
    )


def _scalar_metrics(record: Mapping[str, Any]) -> dict[str, float | None]:
    result = {
        "expected_annual_loss": record["expected_annual_loss"],
        "annual_probability_positive": record["annual_probability_positive"],
        "annual_loss_median": record["annual_loss_median"],
        "annual_loss_variance": record["annual_loss_variance"],
    }
    for level, tail in record["tail_metrics"].items():
        result[f"var::{level}"] = tail["var"]
        result[f"tvar::{level}"] = tail["tvar"]
    return result


_BASIS_FIELDS = ("currency", "monetary_basis", "accounting_standing")


def _support(capture: Mapping[str, Any], arm: str, state: str) -> Mapping[str, Any]:
    if "support_records" not in capture:
        return {}
    reference = capture.get("support_by_arm_state", {}).get(arm, {}).get(state)
    record = capture["support_records"].get(reference)
    if record is None:
        raise ValueError(
            "capture support reference is missing for a required conditional state"
        )
    return record


def _view_basis(
    capture: Mapping[str, Any], arm: str, weights: Mapping[str, float], view: str
) -> tuple[dict[str, str] | None, str | None]:
    if "support_records" not in capture:
        return None, None
    fields = (
        (*_BASIS_FIELDS, "severity_unit")
        if view.startswith("scenario::")
        else (*_BASIS_FIELDS, "source_public_status")
        if view.startswith("source_status::")
        else _BASIS_FIELDS
    )
    representative = None
    for state, weight in weights.items():
        if weight == 0:
            continue
        table = _support(capture, arm, state).get("view_bases", {})
        key = (
            "::".join(view.split("::")[:2])
            if view.startswith("scenario::")
            else "pooled"
        )
        basis = table.get(view, table.get(key))
        if basis is None or any(not basis.get(field) for field in fields):
            return (
                None,
                "Requested view lacks complete monetary or accounting basis metadata in a positive-weight state.",
            )
        projected = {field: basis[field] for field in fields}
        if representative is not None and representative != projected:
            return (
                None,
                "Positive-weight states use incompatible currency, monetary, accounting or severity-unit bases for this view; no numeric pooling is defined.",
            )
        representative = projected
    return representative, None


def _effect_gate(effect, measure, baseline, changed, batch_values, batches, margin=0.0):
    """Withhold an effect whose arms' statistic, or whose own paired
    batch-means half-width, has not converged.

    Both arms' records and ``batch_values``, the reference arm's and the
    alternative arm's batch statistics, are on their common years, the
    shorter arm's, so the effect's batches difference the same years. A
    statistic without batch values, such as a native EAL, has no sampling
    error to gate. ``margin`` widens the half-width of an effect on TVaR by
    both arms' integration margins (``_tvar_margin``).
    """
    withheld = [
        status["reason"]
        for record in (baseline, changed)
        if (status := record.get("statistics", {}).get(measure, {})).get("status")
        == "withheld"
    ]
    if withheld:
        effect["status"], effect["reason"] = "withheld", withheld[0]
    if batches is None or effect.get("absolute_difference") is None:
        return
    reference, alternative = batch_values
    tolerance = batches["tolerance"]
    gate = paired_effect(
        reference, alternative, effect["reference"], tolerance, margin=margin
    )
    if gate is None:
        return
    effect["half_width"] = gate["half_width"]
    if gate["status"] != "converged" and not withheld:
        relative = gate["relative_half_width"]
        width = "unbounded" if relative is None else f"±{100 * relative:.1f}%"
        effect["status"] = "withheld"
        effect["reason"] = (
            f"Unconverged at {batches['years']} years per state, the arms' common years: the effect's 95% batch-means half-width is {width} of the reference value, above the {100 * tolerance:g}% tolerance."
        )


def _view_batches(batches, arm: str, view: str) -> dict[str, Any]:
    """One arm's batch statistics and TVaR estimates for ``view`` on an
    effect's common years, empty when there are none."""
    found = (((batches or {}).get("values") or {}).get(arm) or {}).get(view) or {}
    return {"values": found.get("values") or {}, "tvar": found.get("tvar") or {}}


def _common_tvar(reference, alternative, level: str):
    """Both arms' TVaR at ``level`` and its batch values with one
    control-variate coefficient (``common_coefficient``), and that coefficient;
    None when either arm lacks its estimate or no coefficient suits both."""
    pair = (reference["tvar"].get(level), alternative["tvar"].get(level))
    if None in pair:
        return None
    (control, control_batches), (changed, changed_batches) = pair
    coefficient = common_coefficient(control, changed)
    if coefficient is None:
        return None
    values = tuple(
        None
        if parts is None
        else [estimate.tvar_at(part, coefficient) for part in parts]
        for estimate, parts in ((control, control_batches), (changed, changed_batches))
    )
    return (
        control.tvar_at(control.parts, coefficient),
        changed.tvar_at(changed.parts, coefficient),
        values,
        coefficient,
    )


def _tvar_margin(reference, alternative, level: str, coefficient: float | None):
    """Both arms' integration margins at ``level``, added: the most their
    analytic means' certified integration error bounds can move a TVaR effect
    (``TailMeanEstimate.integration_margin_at``), at the effect's shared
    ``coefficient`` or, without one, at each arm's own."""
    margin = 0.0
    for arm in (reference, alternative):
        found = arm["tvar"].get(level)
        if found is not None:
            estimate = found[0]
            margin += estimate.integration_margin_at(
                estimate.coefficient if coefficient is None else coefficient
            )
    return margin


def _compare_views(arms, alternative_arm, reference_arm, views, batches=None):
    """Each view's effects of the alternative arm against the reference arm.

    An effect on TVaR applies one control-variate coefficient to both arms
    (``common_coefficient``), so it is a difference of one estimator and keeps
    the common years' cancellation; it reports that coefficient, or null when
    none suits both arms and each keeps its own, and its half-width adds both
    arms' integration margins (``_tvar_margin``). The records do not name the
    arms, which the comparison tables key them by.
    """
    reference = arms[reference_arm]
    alternative = arms[alternative_arm]
    output: dict[str, dict[str, Any]] = {}
    for view in views:
        baseline = reference.get("metrics", {}).get(view, {})
        changed = alternative.get("metrics", {}).get(view, {})
        reason = None
        if (
            baseline.get("status") != "available"
            or changed.get("status") != "available"
        ):
            reason = "Reference or required alternative view is unavailable."
        elif baseline.get("basis") != changed.get("basis"):
            reason = "Reference and alternative use incompatible view bases; their numeric contrast is unavailable."
        if reason:
            output[view] = {"status": "unavailable", "reason": reason}
            continue
        before, after = _scalar_metrics(baseline), _scalar_metrics(changed)
        reference_batches = _view_batches(batches, reference_arm, view)
        alternative_batches = _view_batches(batches, alternative_arm, view)
        output[view] = {}
        for measure, value in before.items():
            other = after[measure]
            batch_values = (
                reference_batches["values"].get(measure),
                alternative_batches["values"].get(measure),
            )
            extra: dict[str, float | None] = {}
            margin = 0.0
            if measure.startswith("tvar::"):
                level = measure.removeprefix("tvar::")
                common = (
                    None
                    if value is None or other is None
                    else _common_tvar(reference_batches, alternative_batches, level)
                )
                shared = None
                if common is not None:
                    value, other, batch_values, shared = common
                extra["control_variate_coefficient"] = shared
                margin = _tvar_margin(
                    reference_batches, alternative_batches, level, shared
                )
            effect = (
                comparison(value, other)
                if value is not None and other is not None
                else {
                    "status": "unavailable",
                    "reason": "The requested reference or alternative functional is unavailable for this target law.",
                }
            )
            output[view][measure] = {
                **effect,
                "years_per_state": None if batches is None else batches["years"],
                **extra,
            }
            _effect_gate(
                output[view][measure],
                measure,
                baseline,
                changed,
                batch_values,
                batches,
                margin,
            )
    return output


def _view_mixture(
    draws: Mapping[str, ConditionalCapabilityDraws],
    weights: Mapping[str, float],
    view: str,
) -> CapabilityMixture:
    selected = {}
    for state, draw in draws.items():
        if view not in draw.annual_losses:
            continue
        selected[state] = replace(draw, views={view: draw.views[view]})
    return CapabilityMixture(selected, weights)


@dataclass(frozen=True)
class _ViewTask:
    """One arm's view to reduce, with that view's draws only, so a worker
    process receives just its arrays."""

    mixture: CapabilityMixture
    view: str
    levels: tuple[float, ...]
    controls: TailGateControls


def _view_task_metrics(task: _ViewTask) -> dict[str, Any]:
    return mixture_view_metrics(task.mixture, task.view, task.levels, task.controls)


def _serial_map(function, tasks):
    return [function(task) for task in tasks]


def _rank_comparisons(
    arms, comparisons, view_arms, reference_arm, views, complete, inapplicable=False
):
    """Rank each view's required alternative arms by absolute point effect.

    ``complete`` states whether every required comparison in the view's scope
    resolved. A complete view with no required alternative arm in scope has
    nothing to rank: it is ``inapplicable`` when reasoned inapplicability
    dispositions cover the view and ``not_requested`` otherwise. A view whose
    every statistic has the same disposition is that one disposition, and a
    ranked arm names its absolute effect, whose record is in the comparisons.
    """
    reference = arms[reference_arm]
    alternatives = [arm for arm in view_arms if arm != reference_arm]
    rankings: dict[str, dict[str, Any]] = {}
    for view in views:
        reference_metrics = reference.get("metrics", {}).get(view, {})
        if reference_metrics.get("status") != "available":
            rankings[view] = {
                "status": "unavailable",
                "reason": "Reference view is unavailable.",
            }
            continue
        rankings[view] = {}
        for measure in _scalar_metrics(reference_metrics):
            effects = []
            for arm in alternatives:
                effect = comparisons[arm][view].get(measure, {})
                if effect.get("absolute_difference") is not None:
                    effects.append((arm, effect))
            if not complete:
                rankings[view][measure] = {
                    "status": "unavailable",
                    "reason": "Required comparison coverage is incomplete for this view; see comparison requirements and missing arms.",
                }
            elif not alternatives:
                rankings[view][measure] = {
                    "status": "inapplicable" if inapplicable else "not_requested",
                    "reason": "Every comparison in this view's scope is reasoned inapplicable."
                    if inapplicable
                    else "No required alternative arm is in this view's scope.",
                }
            elif len(effects) != len(alternatives):
                rankings[view][measure] = {
                    "status": "unavailable",
                    "reason": "Every required alternative arm must have the requested finite comparison to the reference arm before ranking.",
                }
            elif all(effect.get("status") == "withheld" for _, effect in effects):
                # Like the tornado, rankings never order an effect the tail
                # gate withheld: a precision outcome, not a coverage gap.
                rankings[view][measure] = {
                    "status": "withheld",
                    "reason": "The tail gate withheld every effect; an unconverged effect is not ranked.",
                }
            else:
                effects.sort(
                    key=lambda row: (-abs(row[1]["absolute_difference"]), row[0])
                )
                withheld = {
                    arm: effect["reason"]
                    for arm, effect in effects
                    if effect.get("status") == "withheld"
                }
                effects = [row for row in effects if row[0] not in withheld]
                rankings[view][measure] = {
                    "status": "available",
                    "basis": "absolute_point_effect_without_precision_ordering",
                    "reference_arm": reference_arm,
                    "arms": [
                        {
                            "rank": rank,
                            "arm": arm,
                            "absolute_difference": effect["absolute_difference"],
                        }
                        for rank, (arm, effect) in enumerate(effects, start=1)
                    ],
                    "withheld_arms": withheld,
                }
        rows = list(rankings[view].values())
        if rows and all(
            row["status"] != "available" and row == rows[0] for row in rows
        ):
            rankings[view] = rows[0]
    return rankings


def view_in_scope(view, scenario_ids, affected_views=None):
    """Whether a comparison requirement's scenario and view scope covers ``view``."""
    if affected_views is not None and view not in affected_views:
        return False
    scoped = parse_scenario_view(view)
    return scoped is None or scoped[0] in scenario_ids


def _comparison_coverage(
    requirements,
    arms,
    contrasts,
    reference_contrasts,
    reference_arm,
    views,
):
    assessed = []
    families = {}
    for requirement in requirements:
        row = requirement.model_dump(mode="json")
        row["scope_basis"] = "authored_perturbation_scope"
        arm = requirement.arm_id
        relevant = tuple(
            view
            for view in views
            if view_in_scope(view, requirement.scenario_ids, requirement.affected_views)
        )
        row["requested_views"] = list(relevant)
        row["view_outcomes"] = {}
        if requirement.status != "required":
            row["outcome_status"] = requirement.status
        elif arm is None or arms[arm]["status"] == "unavailable":
            row["outcome_status"] = "missing"
            row["outcome_reason"] = "Required arm has no capture."
        elif not relevant:
            row["outcome_status"] = "unassessed"
            row["outcome_reason"] = (
                "No requested view falls within this required comparison's authored scope."
            )
        else:
            comparison_views = contrasts.get(arm)
            if comparison_views is None:
                comparison_views = _compare_views(arms, arm, reference_arm, views)
            for view in relevant:
                effect = comparison_views[view]
                row["view_outcomes"][view] = {
                    "status": "unavailable"
                    if effect.get("status") == "unavailable"
                    else "available",
                }
            row["outcome_status"] = (
                "partial"
                if any(
                    record["status"] == "unavailable"
                    for record in row["view_outcomes"].values()
                )
                else "available"
            )
        index = len(assessed)
        assessed.append(row)
        family = families.setdefault(
            requirement.family_id,
            {"requirement_indices": [], "scenario_ids": set(), "required_arms": set()},
        )
        family["requirement_indices"].append(index)
        family["scenario_ids"].update(requirement.scenario_ids)
        if requirement.status == "required":
            family["required_arms"].add(arm)
    for family in families.values():
        rows = [assessed[index] for index in family["requirement_indices"]]
        family["scenario_ids"] = sorted(family["scenario_ids"])
        family["required_arms"] = sorted(family["required_arms"])
        family["status"] = (
            "inapplicable"
            if all(row["outcome_status"] == "inapplicable" for row in rows)
            else "complete"
            if all(
                row["outcome_status"] in {"available", "inapplicable"} for row in rows
            )
            else "partial"
        )
        family["rankings"] = {}
        for view in views:
            relevant = [
                row
                for row in rows
                if view_in_scope(view, row["scenario_ids"], row["affected_views"])
            ]
            if not relevant or all(row["status"] == "inapplicable" for row in relevant):
                continue
            complete = all(
                row["outcome_status"] in {"available", "inapplicable"}
                or (
                    row["status"] == "required"
                    and row["view_outcomes"].get(view, {}).get("status") == "available"
                )
                for row in relevant
            )
            required = tuple(
                sorted(
                    {reference_arm}
                    | {row["arm_id"] for row in relevant if row["status"] == "required"}
                )
            )
            family["rankings"].update(
                _rank_comparisons(
                    arms,
                    reference_contrasts,
                    required,
                    reference_arm,
                    (view,),
                    complete,
                )
            )
    return assessed, families


# Portfolio views a reader may expect; any of them that a plan does not request
# is listed as not requested rather than silently absent.
STANDARD_VIEWS = (
    "welfare",
    "gross",
    "net_resource_cost",
    "financial",
    "oep_gross",
    "oep_financial",
)


def _view_coverage(reference: ArmResult, views: Sequence[str]) -> list[ViewCoverage]:
    """Requested views' reference-arm status, then unrequested standard views."""
    metrics: Mapping[str, Any] = reference.get("metrics", {})
    rows: list[ViewCoverage] = []
    for view in views:
        metric = metrics.get(view, {})
        if metric.get("status") == "available":
            rows.append({"view": view, "status": "available", "reason": None})
        else:
            rows.append(
                {
                    "view": view,
                    "status": "unavailable",
                    "reason": metric.get("reason") or reference.get("reason"),
                }
            )
    for view in STANDARD_VIEWS:
        if view not in views:
            rows.append(
                {
                    "view": view,
                    "status": "not_requested",
                    "reason": "The execution plan does not request this view.",
                }
            )
    return rows


def _family_coverage(
    requirements: Sequence[Mapping[str, Any]],
    families: Mapping[str, Mapping[str, Any]],
) -> list[FamilyCoverage]:
    """Each catalog family's and frozen register challenge's comparison outcome.

    Rows follow the catalog, then every other frozen requirement key, which is
    a register challenge frozen unavailable (``register_challenge``). The
    status is the reduction's own family status (``comparison_families``),
    refined only where it is silent: no frozen requirement is ``not_requested``,
    and a family whose every requirement was frozen unavailable is
    ``unavailable`` rather than ``partial``. Each row has a reason unless it
    is complete.
    """
    catalog = {family.family_id: family.kind for family in SENSITIVITY_FAMILIES}
    keys: list[tuple[str, FamilyKind]] = [
        *catalog.items(),
        *(
            (family_id, "register_challenge")
            for family_id in sorted({row["family_id"] for row in requirements})
            if family_id not in catalog
        ),
    ]
    rows: list[FamilyCoverage] = []
    for family_id, kind in keys:
        assessed = [row for row in requirements if row["family_id"] == family_id]
        status: Literal[
            "complete", "partial", "unavailable", "inapplicable", "not_requested"
        ]
        if not assessed:
            status = "not_requested"
        elif all(row["status"] == "unavailable" for row in assessed):
            status = "unavailable"
        else:
            status = families[family_id]["status"]
        reasons = [
            row.get("outcome_reason")
            or row.get("reason")
            or "Unavailable in views: "
            + ", ".join(
                view
                for view, outcome in row["view_outcomes"].items()
                if outcome["status"] == "unavailable"
            )
            for row in assessed
            if row["outcome_status"] != "available"
        ]
        rows.append(
            {
                "family_id": family_id,
                "kind": kind,
                "status": status,
                "reason": "The execution plan froze no requirement for this family."
                if not assessed
                else None
                if status == "complete"
                else "; ".join(dict.fromkeys(reasons)),
            }
        )
    return rows


def reduce_captures(
    *,
    scientific_sha256: str,
    plan_sha256: str,
    draws_by_arm: Mapping[str, Mapping[str, ConditionalCapabilityDraws]],
    state_weights: Mapping[str, float],
    required_arms: tuple[str, ...],
    reference_arm: str,
    views: tuple[str, ...],
    reducer_provenance: Mapping[str, object],
    spec: ReductionSpec | None = None,
    capture_provenance: Mapping[str, object] | None = None,
    epistemic_groups: Mapping[str, str] | None = None,
    sample_groups: SampleGroups | None = None,
    matched_controls: Mapping[str, str] | None = None,
    state_weights_by_arm: Mapping[str, Mapping[str, float]] | None = None,
    comparison_requirements: Sequence[
        ComparisonRequirement | Mapping[str, object]
    ] = (),
    view_map: Callable[[Callable[[Any], Any], Iterable[Any]], list[Any]] = _serial_map,
) -> ReducedResults:
    """Reduce each required arm without inferring completeness from recorded arms.

    ``sample_groups`` are the setup's states as respondent samples, keyed
    like the weights (``respondent_sampling.sample_groups``): they give each
    arm's weights in the views that count the lab's loss where a state is
    left out of them (``ViewWeights``). Without them every view uses the
    arm's weights.

    ``view_map(function, tasks)`` applies ``function`` to each task and returns
    the results in task order; it reduces each arm's views, one task per view,
    and a worker pool's map spreads them over processes without changing them.

    The inputs are checked first (``_validate_reduction``), each required arm
    is reduced (``_ArmReductions``), every arm is compared with its matched
    control and the reference arm (``_arm_contrasts``), and the views' arms are
    ranked by their effects (``_view_rankings``).
    """
    reduction = _validate_reduction(
        draws_by_arm=draws_by_arm,
        state_weights=state_weights,
        required_arms=required_arms,
        reference_arm=reference_arm,
        views=views,
        reducer_provenance=reducer_provenance,
        spec=spec,
        capture_provenance=capture_provenance,
        epistemic_groups=epistemic_groups,
        sample_groups=sample_groups,
        matched_controls=matched_controls,
        state_weights_by_arm=state_weights_by_arm,
        comparison_requirements=comparison_requirements,
    )
    identity = _analysis_identity(
        reduction,
        scientific_sha256=scientific_sha256,
        plan_sha256=plan_sha256,
        draws_by_arm=draws_by_arm,
        state_weights=state_weights,
        reducer_provenance=reducer_provenance,
    )
    reduced = _ArmReductions(reduction, draws_by_arm, view_map)
    arms, missing = reduced.arms, reduced.missing
    reference_arm, views = reduction.reference_arm, reduction.views
    required_alternative_arms = reduction.alternative_arms
    incomplete_views = {
        arm: row.get("missing_views", [])
        for arm, row in arms.items()
        if row.get("missing_views")
    }
    complete = not missing and not incomplete_views
    contrasted = _arm_contrasts(reduction, reduced)
    references, contrasts = contrasted.references, contrasted.by_control
    reference_contrasts = contrasted.against_reference
    unavailable_comparison_views = contrasted.unavailable_views
    complete = complete and not unavailable_comparison_views

    assessed_requirements, families = _comparison_coverage(
        reduction.requirements,
        arms,
        contrasts,
        reference_contrasts,
        reference_arm,
        views,
    )
    requirements_complete = bool(
        required_alternative_arms or assessed_requirements
    ) and all(
        row["outcome_status"] in {"available", "inapplicable"}
        for row in assessed_requirements
    )
    complete = complete and bool(required_alternative_arms) and requirements_complete
    comparison_status = "complete" if complete else "partial"
    if not required_alternative_arms and not assessed_requirements:
        comparison_status = "not_requested"
    elif not required_alternative_arms and all(
        row["outcome_status"] == "inapplicable" for row in assessed_requirements
    ):
        comparison_status = "inapplicable"
    rankings = _view_rankings(
        reduction,
        arms,
        missing,
        reference_contrasts,
        unavailable_comparison_views,
        assessed_requirements,
    )
    ranking_statuses = {
        row["status"]
        for view_rows in rankings.values()
        for row in ([view_rows] if "status" in view_rows else view_rows.values())
    }
    diagnostics = {arm: _split_diagnostics(record) for arm, record in arms.items()}
    report: ReducedResults = {
        "analysis_sha256": canonical_json_sha256(identity),
        "analysis_spec": identity,
        "status": "complete" if complete else "partial",
        "coverage": {
            "comparisons": {
                "status": comparison_status,
                "required_alternative_arms": required_alternative_arms,
                "missing_arms": missing,
                "missing_views": incomplete_views,
                "unavailable_comparison_views": unavailable_comparison_views,
                "requirements_complete": requirements_complete,
            },
            "views": _view_coverage(arms[reference_arm], views),
            "sensitivity_families": _family_coverage(assessed_requirements, families),
        },
        "arms": arms,
        "comparisons": contrasts,
        "comparison_reference_arms": references,
        # An arm compared with the reference arm in ``comparisons`` is not
        # repeated here.
        "reference_comparisons": {
            arm: rows
            for arm, rows in reference_contrasts.items()
            if references.get(arm) != reference_arm
        },
        "rankings_available": complete
        and "available" in ranking_statuses
        and "unavailable" not in ranking_statuses,
        "rankings": rankings,
        "comparison_requirements": assessed_requirements,
        "comparison_families": families,
        "diagnostics": {
            "arms": {arm: record for arm, record in diagnostics.items() if record}
        },
    }
    json.dumps(report, allow_nan=False)
    return report


@dataclass(frozen=True)
class _Reduction:
    """A reduction's checked inputs (``_validate_reduction``): its options,
    the capture's provenance, the required arms and their alternatives, the
    views, each arm's weights over the declared support and in each view,
    the frozen requirements, the matched controls and any epistemic groups."""

    spec: ReductionSpec
    capture: dict[str, Any]
    required_arms: tuple[str, ...]
    reference_arm: str
    alternative_arms: list[str]
    views: tuple[str, ...]
    weights_by_arm: dict[str, dict[str, float]]
    view_weights: dict[str, ViewWeights]
    requirements: tuple[ComparisonRequirement, ...]
    controls: dict[str, str]
    epistemic_groups: Mapping[str, str] | None

    def gate_controls(self, arm: str) -> TailGateControls:
        """The arm's tail-gate controls in the producer's frozen plan."""
        producer_plan = cast(
            dict[str, Any], self.capture.get("producer_execution_plan") or {}
        )
        return TailGateControls.from_controls(
            {
                **producer_plan.get("numerical_controls", {}),
                **producer_plan.get("numerical_controls_by_arm", {}).get(arm, {}),
            }.get("tail_gate")
        )


def _validate_reduction(
    *,
    draws_by_arm: Mapping[str, Mapping[str, ConditionalCapabilityDraws]],
    state_weights: Mapping[str, float],
    required_arms: tuple[str, ...],
    reference_arm: str,
    views: tuple[str, ...],
    reducer_provenance: Mapping[str, object],
    spec: ReductionSpec | None,
    capture_provenance: Mapping[str, object] | None,
    epistemic_groups: Mapping[str, str] | None,
    sample_groups: SampleGroups | None,
    matched_controls: Mapping[str, str] | None,
    state_weights_by_arm: Mapping[str, Mapping[str, float]] | None,
    comparison_requirements: Sequence[ComparisonRequirement | Mapping[str, object]],
) -> _Reduction:
    """Check a reduction's inputs and resolve its options.

    Refuses joint thresholds without physical joint samples, required arms
    that are not unique or lack the reference, weights that are not a
    probability over the full declared support in every arm, requirements
    naming other arms, repeated or no views, no reducer provenance, epistemic
    groups that do not cover the support, matched controls outside the
    required arms or in a cycle, and captured arms outside the requirements.
    """
    spec = spec or ReductionSpec()
    capture = dict(capture_provenance or {})
    if (
        spec.joint_thresholds
        and capture.get("physical_joint_semantics") != "physical_joint_samples"
    ):
        raise ValueError(
            "joint exceedance requires declared physical joint samples, not paired marginal ranks"
        )
    if (
        not required_arms
        or len(set(required_arms)) != len(required_arms)
        or reference_arm not in required_arms
    ):
        raise ValueError(
            "required comparison arms must be unique and include the reference"
        )
    required_alternative_arms = [arm for arm in required_arms if arm != reference_arm]

    def validate_weights(weights):
        if (
            not isinstance(weights, Mapping)
            or not weights
            or not valid_probability_weights(weights.values(), abs_tol=1e-10)
        ):
            raise ValueError("state weights must be finite, nonnegative and sum to one")

    validate_weights(state_weights)
    if state_weights_by_arm is not None and (
        not isinstance(state_weights_by_arm, Mapping)
        or set(state_weights_by_arm) != set(required_arms)
    ):
        raise ValueError("arm weights must cover exactly the frozen required arms")
    weights_by_arm = {}
    for arm in required_arms:
        weights = (
            state_weights if state_weights_by_arm is None else state_weights_by_arm[arm]
        )
        validate_weights(weights)
        if set(weights) != set(state_weights):
            raise ValueError(
                "every arm weight vector must name the full declared support"
            )
        weights_by_arm[arm] = dict(weights)
    requirements = tuple(
        ComparisonRequirement.model_validate(row) for row in comparison_requirements
    )
    if any(
        row.arm_id is not None
        and (row.arm_id not in required_arms or row.arm_id == reference_arm)
        for row in requirements
    ):
        raise ValueError(
            "comparison requirement arms must be frozen required alternative arms"
        )
    if not views or len(set(views)) != len(views):
        raise ValueError("reduction views must be distinct and nonempty")
    if not reducer_provenance:
        raise ValueError("current reducer provenance is required")
    if epistemic_groups is not None and set(epistemic_groups) != set(state_weights):
        raise ValueError("epistemic groups must cover every joint state")
    controls = dict(matched_controls or {})
    if set(controls) - set(required_arms) or set(controls.values()) - set(
        required_arms
    ):
        raise ValueError(
            "matched controls and every transitive control must be included in frozen required arms"
        )
    for arm in controls:
        visited = set()
        current = arm
        while current in controls and controls[current] != current:
            if current in visited:
                raise ValueError("matched controls contain a nontrivial cycle")
            visited.add(current)
            current = controls[current]
    unknown_arms = set(draws_by_arm) - set(required_arms)
    if unknown_arms:
        raise ValueError("capture contains arms outside the frozen requirements")
    return _Reduction(
        spec=spec,
        capture=capture,
        required_arms=required_arms,
        reference_arm=reference_arm,
        alternative_arms=required_alternative_arms,
        views=views,
        weights_by_arm=weights_by_arm,
        view_weights={
            arm: ViewWeights.same(weights)
            if sample_groups is None
            else sample_groups.view_weights(weights)
            for arm, weights in weights_by_arm.items()
        },
        requirements=requirements,
        controls=controls,
        epistemic_groups=epistemic_groups,
    )


def _analysis_identity(
    reduction: _Reduction,
    *,
    scientific_sha256: str,
    plan_sha256: str,
    draws_by_arm: Mapping[str, Mapping[str, ConditionalCapabilityDraws]],
    state_weights: Mapping[str, float],
    reducer_provenance: Mapping[str, object],
) -> AnalysisSpec:
    """What a reduction reduces: the setup, plan and capture identities,
    every captured draw's digest, the weights, arms, requirements, views,
    reduction options and reducer."""
    return {
        "scientific_sha256": scientific_sha256,
        "plan_sha256": plan_sha256,
        # A run's or replay's (``reduce_capture``); empty for a reduction
        # given none.
        "capture_provenance": cast(CaptureProvenance, reduction.capture),
        "captures": {
            arm: {key: _draw_digest(draw) for key, draw in sorted(draws.items())}
            for arm, draws in sorted(draws_by_arm.items())
        },
        "state_weights": dict(state_weights),
        "state_weights_by_arm": reduction.weights_by_arm,
        "required_arms": reduction.required_arms,
        "reference_arm": reduction.reference_arm,
        "matched_controls": reduction.controls,
        "comparison_requirements": [
            cast(ComparisonRequirementRecord, row.model_dump(mode="json"))
            for row in reduction.requirements
        ],
        "views": reduction.views,
        "reduction_spec": cast(ReductionSpecRecord, asdict(reduction.spec)),
        "reducer_provenance": cast(CodeProvenance, dict(reducer_provenance)),
        "epistemic_groups": None
        if reduction.epistemic_groups is None
        else dict(reduction.epistemic_groups),
    }


class _ArmReductions:
    """Each required arm's reduction at its own years (``_reduce_arm``), the
    arms without a capture, and, for paired effects, each arm's reduction at
    another arm's fewer years, computed once when first needed."""

    def __init__(
        self,
        reduction: _Reduction,
        draws_by_arm: Mapping[str, Mapping[str, ConditionalCapabilityDraws]],
        view_map: Callable[[Callable[[Any], Any], Iterable[Any]], list[Any]],
    ) -> None:
        self.reduction = reduction
        self.draws_by_arm = draws_by_arm
        self.view_map = view_map
        self.arms: dict[str, ArmResult] = {}
        self.missing: list[str] = []
        self.batches: dict[str, dict[str, Any]] = {}
        self.years: dict[str, int] = {}
        # An arm's statistics on another arm's fewer years, for paired effects.
        self.prefixed: dict[
            tuple[str, int], tuple[ReducedArmResult, dict[str, Any]]
        ] = {}
        for arm in reduction.required_arms:
            draws = draws_by_arm.get(arm)
            if not draws:
                self.missing.append(arm)
                self.arms[arm] = {
                    "status": "unavailable",
                    "reason": "Required arm has no capture.",
                }
                continue
            arm_years = {draw.n_simulations for draw in draws.values()}
            if len(arm_years) != 1:
                raise ValueError("an arm's captured states must share their years")
            (self.years[arm],) = arm_years
            self.arms[arm], self.batches[arm] = _reduce_arm(
                reduction, arm, draws, view_map
            )

    def at_years(self, arm: str, common: int) -> tuple[Any, dict[str, Any] | None]:
        """The arm's record and batch values at ``common`` years."""
        if arm not in self.years or self.years[arm] == common:
            return self.arms[arm], self.batches.get(arm)
        if (arm, common) not in self.prefixed:
            self.prefixed[arm, common] = _reduce_arm(
                self.reduction,
                arm,
                {
                    state: year_prefix(draw, common)
                    for state, draw in self.draws_by_arm[arm].items()
                },
                self.view_map,
            )
        return self.prefixed[arm, common]

    def contrast(self, alternative: str, control: str):
        """Each view's effects of ``alternative`` against ``control``, paired
        on their common years."""
        years, arms = self.years, self.arms
        common = min(
            (years[arm] for arm in (alternative, control) if arm in years),
            default=None,
        )
        control_record, control_batches = (
            (arms[control], None) if common is None else self.at_years(control, common)
        )
        alternative_record, alternative_batches = (
            (arms[alternative], None)
            if common is None
            else self.at_years(alternative, common)
        )
        return _compare_views(
            {control: control_record, alternative: alternative_record},
            alternative,
            control,
            self.reduction.views,
            {
                "values": {
                    control: control_batches,
                    alternative: alternative_batches,
                },
                "tolerance": self.reduction.gate_controls(control).tolerance,
                "years": common,
            },
        )


@dataclass(frozen=True)
class _Contrasts:
    """Each arm's effects against its control (``by_control``, keyed by arm,
    with each arm's control in ``references``), each alternative arm's
    against the reference arm, and the views each control comparison could
    not contrast, with why."""

    references: dict[str, str]
    by_control: dict[str, Any]
    against_reference: dict[str, Any]
    unavailable_views: dict[str, dict[str, str]]


def _arm_contrasts(reduction: _Reduction, reduced: _ArmReductions) -> _Contrasts:
    """Compare each arm with its matched control, or else the reference arm,
    and each alternative arm with the reference arm, paired on their common
    years (``_ArmReductions.contrast``)."""
    required_arms, reference_arm = reduction.required_arms, reduction.reference_arm
    controls, contrast = reduction.controls, reduced.contrast
    references = {
        arm: controls.get(arm, reference_arm)
        for arm in required_arms
        if arm != reference_arm or arm in controls
    }
    contrasts = {arm: contrast(arm, control) for arm, control in references.items()}
    reference_contrasts = {
        arm: contrasts[arm]
        if references.get(arm) == reference_arm
        else contrast(arm, reference_arm)
        for arm in required_arms
        if arm != reference_arm
    }
    unavailable_comparison_views = {
        arm: {
            view: record["reason"]
            for view, record in by_view.items()
            if record.get("status") == "unavailable"
        }
        for arm, by_view in contrasts.items()
    }
    unavailable_comparison_views = {
        arm: records for arm, records in unavailable_comparison_views.items() if records
    }
    return _Contrasts(
        references, contrasts, reference_contrasts, unavailable_comparison_views
    )


def _reduce_arm(
    reduction: _Reduction,
    arm: str,
    draws: Mapping[str, ConditionalCapabilityDraws],
    view_map: Callable[[Callable[[Any], Any], Iterable[Any]], list[Any]],
) -> tuple[ReducedArmResult, dict[str, Any]]:
    """One arm's reduced record at its draws' years, and the batch values
    of its gated statistics for paired arm effects.

    Each view's metrics (``_arm_metrics``) withhold what its states' laws do
    not identify (``_withhold_functionals``); the record adds any joint
    exceedance (``_add_joint_exceedance``) and each epistemic group's EAL
    (``_add_epistemic_group_eals``), each at its view's weights, and the
    weights of the views that count the lab's loss where they differ from the
    arm's.
    """
    spec, capture, views = reduction.spec, reduction.capture, reduction.views
    weights = reduction.view_weights[arm]
    gate_controls = reduction.gate_controls(arm)
    metrics, available_views, absent = _arm_metrics(
        spec, capture, views, arm, draws, weights, gate_controls, view_map
    )
    for view, metric in metrics.items():
        if metric["status"] != "available":
            continue
        _withhold_functionals(
            capture, arm, view, metric, draws, cast(Mapping, weights.of(view))
        )
    batch_values = {
        view: {
            "values": metric.pop("_batch_values", None),
            "tvar": metric.pop("_tvar_estimates", None),
        }
        for view, metric in metrics.items()
        if metric.get("status") == "available"
    }
    record: ReducedArmResult = {
        "status": "partial" if absent else "reduced",
        "missing_views": absent,
        "metrics": metrics,
    }
    if weights.differ and weights.counting_lab_loss is not None:
        record["lab_loss_state_weights"] = dict(weights.counting_lab_loss)
    if spec.joint_thresholds:
        _add_joint_exceedance(
            record, spec, draws, weights, available_views, gate_controls
        )
    if reduction.epistemic_groups is not None:
        _add_epistemic_group_eals(
            record,
            capture,
            arm,
            views,
            draws,
            weights,
            available_views,
            reduction.epistemic_groups,
        )
    return record, batch_values


def _arm_metrics(
    spec: ReductionSpec,
    capture: Mapping[str, Any],
    views: tuple[str, ...],
    arm: str,
    draws: Mapping[str, ConditionalCapabilityDraws],
    weights: ViewWeights,
    gate_controls: TailGateControls,
    view_map: Callable[[Callable[[Any], Any], Iterable[Any]], list[Any]],
) -> tuple[dict[str, Any], set[str], list[str]]:
    """Each view's metrics in one arm, one ``view_map`` task per available
    view, or why it is unavailable; with the available views and the absent
    ones.

    A view is available when it has weights in the arm (``ViewWeights.of``),
    every state with weight in it retains it and those states share a basis
    (``_view_basis``). Refuses states outside the support, a positive weight
    without draws and joint thresholds over states without physical joint
    provenance.
    """
    if set(draws) - set(weights.arm):
        raise ValueError("capture contains states outside the declared support")
    if any(weight > 0 and state not in draws for state, weight in weights.arm.items()):
        raise ValueError("positive reweighting requires stored conditional draws")

    def retained(of: Mapping[str, float]) -> set[str]:
        """The views every state with weight in ``of`` retains."""
        return set.intersection(
            *(
                set(draws[state].annual_losses)
                for state, weight in of.items()
                if weight > 0
            )
        )

    # Each view is available where the states with weight in it retain it.
    held = {id(weights.arm): retained(weights.arm)}
    if weights.differ and weights.counting_lab_loss is not None:
        held[id(weights.counting_lab_loss)] = retained(weights.counting_lab_loss)
    available_views = {
        view
        for view in set().union(*held.values())
        if (of := weights.of(view)) is not None and view in held[id(of)]
    }
    if spec.joint_thresholds and any(
        draws[state].joint_law_semantics != "physical_joint"
        for state, weight in weights.arm.items()
        if weight > 0
    ):
        raise ValueError(
            "joint exceedance requires captured physical joint evaluator provenance for every positive-weight state"
        )
    basis_by_view = {}
    basis_reasons = {}
    for view in views:
        of = weights.of(view)
        basis, reason = (
            (None, None) if of is None else _view_basis(capture, arm, of, view)
        )
        basis_by_view[view] = basis
        if reason is not None:
            basis_reasons[view] = reason
            available_views.discard(view)
    absent = sorted(set(views) - available_views)
    levels = spec.levels
    computed = iter(
        view_map(
            _view_task_metrics,
            (
                _ViewTask(
                    _view_mixture(draws, cast(Mapping, weights.of(view)), view),
                    view,
                    levels,
                    gate_controls,
                )
                for view in views
                if view in available_views
            ),
        )
    )
    metrics: dict[str, Any] = {}
    for view in views:
        if view not in available_views:
            metrics[view] = {
                "status": "unavailable",
                "reason": basis_reasons.get(
                    view,
                    weights.reason(view)
                    or "Requested view was not retained in every positive-weight state.",
                ),
                "state_reasons": {
                    state: _support(capture, arm, state)
                    .get("unavailable_views", {})
                    .get(view, "View was not retained.")
                    for state, weight in (weights.of(view) or {}).items()
                    if weight > 0 and view not in draws[state].annual_losses
                },
            }
            continue
        metrics[view] = {
            "status": "available",
            "basis": basis_by_view[view],
            **next(computed),
        }
    return metrics, available_views, absent


def _withhold_functionals(
    capture: Mapping[str, Any],
    arm: str,
    view: str,
    metric: dict[str, Any],
    draws: Mapping[str, ConditionalCapabilityDraws],
    weights: Mapping[str, float],
) -> None:
    """Withhold, in place, the available view's statistics its states' laws do
    not identify: a precision without a finite second moment, an EAL and
    functionals without a finite first moment, and unsupported tails
    (``withhold_unsupported_tails``)."""
    reason_lists: dict[str, list[str]] = {}
    precision_reasons = []
    other_precision_unavailable = False
    for state, weight in weights.items():
        if weight == 0:
            continue
        state_reasons = (
            _support(capture, arm, state)
            .get("unavailable_functionals", {})
            .get(view, {})
        )
        for functional, reason in state_reasons.items():
            reason_lists.setdefault(functional, []).append(reason)
        draw = draws[state]
        if not (draw.views[view].variance_exists):
            if (reason := state_reasons.get("expected_annual_loss")) and (
                draw.views[view].mean_exists
            ):
                precision_reasons.append(reason)
            else:
                other_precision_unavailable = True
    reasons = {
        functional: "; ".join(dict.fromkeys(values))
        for functional, values in reason_lists.items()
    }
    statistics = metric["statistics"]
    if precision_reasons:
        # A positive mean below numeric range leaves the variance
        # withheld, unless another state lacks a second moment.
        if other_precision_unavailable:
            precision_reasons.append(statistics["annual_loss_variance"]["reason"])
        if statistics["normalized_empirical_variance"]["reason"]:
            precision_reasons.append(
                statistics["normalized_empirical_variance"]["reason"]
            )
        precision_reason = "; ".join(dict.fromkeys(precision_reasons))
        statistics["annual_loss_variance"] = (
            statistic_status(unavailable=precision_reason)
            if other_precision_unavailable
            else statistic_status(withheld=precision_reason)
        )
        for tail in metric["tail_diagnostics"]["tails"].values():
            if tail["tvar"] is not None:
                tail["tvar_numerical_standard_error"] = None
                tail["tvar_numerical_interval_95"] = None
                tail["tvar_interval_unavailable_reason"] = precision_reason
    if "expected_annual_loss" in reasons:
        reason = reasons["expected_annual_loss"]
        metric["expected_annual_loss"] = None
        # A finite mean the output cannot represent is withheld.
        statistics["expected_annual_loss"] = (
            statistic_status(withheld=reason)
            if all(
                draws[state].views[view].mean_exists
                for state, weight in weights.items()
                if weight > 0
            )
            else statistic_status(unavailable=reason)
        )
        metric["eal_reconciliation_difference"] = None
        metric["eal_precision"] = {
            "precision_basis": "unavailable_functional",
            "precision_suppression_reason": reason,
        }
        metric["mean_diagnostics"].update(
            {
                "expected_annual_loss": None,
                "empirical_mean_standard_error": None,
                "empirical_mean_standard_error_unavailable_reason": reason,
                "empirical_mean_numerical_interval_95": None,
                "interval_unavailable_reason": reason,
            }
        )
    for functional in ("var", "tvar"):
        if functional not in reasons:
            continue
        for level, tail in metric["tail_metrics"].items():
            tail[functional] = None
            statistics[f"{functional}::{level}"] = statistic_status(
                unavailable=reasons[functional]
            )
            if functional == "tvar":
                tail["tvar_estimator"] = None
        for tail in metric["tail_diagnostics"]["tails"].values():
            tail[functional] = None
            tail[f"{functional}_numerical_interval_95"] = None
            tail[f"{functional}_interval_unavailable_reason"] = reasons[functional]
            if functional == "var":
                tail["var_interval_method"] = None
            else:
                tail["tvar_estimator"] = None
                tail["tvar_numerical_standard_error"] = None
                tail["tvar_sample_tail_mean"] = None
    withhold_unsupported_tails(metric)


def _unavailable_threshold_views(
    metrics: Mapping[str, Mapping[str, Any]],
    names: Sequence[str],
    available_views: set[str],
) -> str | None:
    """Why the joint thresholds' views are not all available in the arm, or
    None where they are.

    A view can be unavailable for an ordinary reason: a state with weight in
    it does not retain it, as where a run withholds the lab-loss views and
    its states hold no lab-loss law, or its states' bases do not pool
    (``_arm_metrics``). The reason names each such view and gives its
    metric's reason and the distinct reasons of the states that lack it.

    Refuses a threshold on a view that is unavailable and is not one of the
    reduction's views. Such a view has no metric to give a reason, and no
    other check reads the thresholds' view names, so the refusal is what
    stops a misspelt view from being reported as unavailable."""
    absent = [name for name in dict.fromkeys(names) if name not in available_views]
    unreported = [name for name in absent if name not in metrics]
    if unreported:
        raise ValueError(
            f"joint exceedance names the view {unreported[0]!r}, which is not "
            "one of the reduction's views and is not captured in the arm"
        )
    reasons = []
    for name in absent:
        metric = metrics[name]
        reason = (
            f"The joint thresholds name the view {name!r}, which is unavailable "
            f"in this arm: {metric['reason']}"
        )
        states = "; ".join(dict.fromkeys(metric["state_reasons"].values()))
        if states:
            reason += f" The states that lack it give: {states}"
        reasons.append(reason)
    return " ".join(reasons) or None


def _add_joint_exceedance(
    record: ReducedArmResult,
    spec: ReductionSpec,
    draws: Mapping[str, ConditionalCapabilityDraws],
    view_weights: ViewWeights,
    available_views: set[str],
    gate_controls: TailGateControls,
) -> None:
    """Add the probability that every joint threshold is exceeded in one
    year, with its tail-gate convergence, to the arm's record.

    The thresholds' views must be available in the arm and reported at one
    set of weights. Where a view that counts the lab's loss, a lab-loss view
    or a gross view, weighs the states otherwise than another threshold's
    view, the arm has no weights for one of them, or one of them is
    unavailable in the arm (``_unavailable_threshold_views``), the
    probability is unavailable with the reason."""
    names = [name for name, _ in spec.joint_thresholds]
    weights = view_weights.of(names[0])
    reason = next(filter(None, map(view_weights.reason, names)), None)
    if reason is None and not view_weights.shared(names):
        reason = (
            "The joint thresholds name a view that counts the lab's own loss (a "
            "lab-loss view or a gross view) and a view that does not. They weigh "
            "the arm's states differently, because a state is left out of the "
            "views that count the lab's loss (POOL-EXCLUDED-SAMPLES), so no "
            "year's losses in them are read together."
        )
    if reason is None:
        reason = _unavailable_threshold_views(
            cast(Mapping[str, Mapping[str, Any]], record["metrics"]),
            names,
            available_views,
        )
    if weights is None or reason is not None:
        record["joint_exceedance_convergence"] = None
        record["joint_exceedance_status"] = statistic_status(unavailable=reason)
        return
    probability = 0.0
    indicators = {}
    for state, mass in weights.items():
        if mass == 0:
            continue
        draw = draws[state]
        exceeds = np.ones(draw.n_simulations, dtype=bool)
        for name, threshold in spec.joint_thresholds:
            exceeds &= draw.annual_losses[name] > threshold
        probability += mass * float(draw.normalized_weights()[exceeds].sum())
        indicators[state] = replace(
            draw, views={"joint": ConditionalView(exceeds.astype(float))}
        )
    record["joint_exceedance_probability"] = probability
    # The joint exceedance is the probability that its indicator is
    # positive, gated like any annual exceedance probability.
    gate = view_gate(
        CapabilityMixture(indicators, weights),
        "joint",
        {},
        gate_controls,
        mean_exists=True,
    )
    row = gate.get("probability_positive")
    record["joint_exceedance_convergence"] = row
    reason = (
        f"The tail gate is unavailable: {gate['unavailable_reason']}"
        if row is None
        else unconverged_reason(
            {"probability": row, **row},
            "probability",
            {
                "batches": gate_controls.batches,
                "tolerance": gate_controls.tolerance,
            },
            gate["years_per_state"],
        )
    )
    record["joint_exceedance_status"] = statistic_status(withheld=reason)


def _add_epistemic_group_eals(
    record: ReducedArmResult,
    capture: Mapping[str, Any],
    arm: str,
    views: tuple[str, ...],
    draws: Mapping[str, ConditionalCapabilityDraws],
    view_weights: ViewWeights,
    available_views: set[str],
    epistemic_groups: Mapping[str, str],
) -> None:
    """Add each available view's EAL conditional on each epistemic group, its
    states mixed by their weights in that view, or why a group's is
    unavailable, to the arm's record."""
    groups: dict[str, dict[str, float | None]] = {}
    group_reasons: dict[str, dict[str, str]] = {}
    for view in views:
        if view not in available_views:
            continue
        values: dict[str, list[tuple[float, float]]] = {}
        unavailable_groups: dict[str, list[str]] = {}
        for state, weight in cast(Mapping, view_weights.of(view)).items():
            if weight == 0:
                continue
            group = epistemic_groups[state]
            draw = draws[state]
            reason = (
                _support(capture, arm, state)
                .get("unavailable_functionals", {})
                .get(view, {})
                .get("expected_annual_loss")
            )
            if not draw.views[view].mean_exists:
                reason = (
                    reason or "The transformed conditional loss law has no finite mean."
                )
            if reason is not None:
                unavailable_groups.setdefault(group, []).append(reason)
                continue
            native = draw.views[view].native_mean
            value = None if native is None else native.value
            if value is None:
                value = float(
                    np.dot(draw.annual_losses[view], draw.normalized_weights())
                )
            values.setdefault(group, []).append((weight, value))
        groups[view] = {}
        for group, parts in values.items():
            maximum = max(weight for weight, _ in parts)
            denominator = math.fsum(weight / maximum for weight, _ in parts)
            groups[view][group] = (
                math.fsum(weight / maximum * value for weight, value in parts)
                / denominator
            )
        if unavailable_groups:
            group_reasons[view] = {
                group: "; ".join(dict.fromkeys(reasons))
                for group, reasons in unavailable_groups.items()
            }
            groups[view].update(dict.fromkeys(unavailable_groups))
    record["conditional_eal_by_epistemic_group"] = groups
    if group_reasons:
        record["conditional_eal_unavailable_reasons_by_epistemic_group"] = group_reasons


def _view_rankings(
    reduction: _Reduction,
    arms: Mapping[str, ArmResult],
    missing: list[str],
    reference_contrasts: Mapping[str, Any],
    unavailable_comparison_views: Mapping[str, Mapping[str, str]],
    assessed_requirements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Each view's ranking of the arms in its scope by their effect against
    the reference arm (``_rank_comparisons``), complete only when every
    comparison in the view's scope resolved."""
    views, required_arms = reduction.views, reduction.required_arms
    reference_arm = reduction.reference_arm
    rankings = {}
    for view in views:
        in_scope = [
            row
            for row in assessed_requirements
            if view_in_scope(view, row["scenario_ids"], row["affected_views"])
        ]
        view_arms = tuple(
            arm
            for arm in required_arms
            if arm == reference_arm
            or not any(row["arm_id"] == arm for row in assessed_requirements)
            or any(
                row["arm_id"] == arm
                and view_in_scope(view, row["scenario_ids"], row["affected_views"])
                for row in assessed_requirements
            )
        )
        view_complete = (
            not set(missing).intersection(view_arms)
            and not any(
                view in records
                for arm, records in unavailable_comparison_views.items()
                if arm in view_arms
            )
            and all(
                row["outcome_status"] in {"available", "inapplicable"}
                or (
                    row["status"] == "required"
                    and row["view_outcomes"].get(view, {}).get("status") == "available"
                )
                for row in in_scope
            )
        )
        rankings.update(
            _rank_comparisons(
                arms,
                reference_contrasts,
                view_arms,
                reference_arm,
                (view,),
                view_complete,
                inapplicable=bool(in_scope)
                and all(row["outcome_status"] == "inapplicable" for row in in_scope),
            )
        )
    return rankings


# Fields of a view's reduced metrics that move to its diagnostics.
_DIAGNOSTIC_FIELDS = (
    "mean_diagnostics",
    "tail_diagnostics",
    "stratum_diagnostics",
    "unsimulated_zero_mass_strata",
)


def _split_diagnostics(record: Mapping[str, Any]) -> ArmDiagnostics | None:
    """Move each available view's diagnostics out of ``record``'s metrics.

    The headline keeps the EAL's estimator and aggregate precision, each tail
    level's batch-means intervals and the tail gate's ``eal_reconciliation``
    of the years with the analytic EAL, whose inconsistency withholds the
    view's VaR and TVaR. Each state's estimator and precision, the mean and
    tail diagnostics (without the copies of the EAL, its precision and the
    tail values) and the stratum records go to the view's diagnostics; the
    limitations of every view's mean and tail diagnostics, which depend only
    on the arm's controls, are stated once for the arm. None for an arm with
    no available view.
    """
    views: dict[str, Any] = {}
    limitations: dict[str, str] = {}
    for view, metric in record.get("metrics", {}).items():
        if metric.get("status") != "available":
            continue
        moved = {field: metric.pop(field) for field in _DIAGNOSTIC_FIELDS}
        mean, tail = moved["mean_diagnostics"], moved["tail_diagnostics"]
        for name, diagnostic in (
            ("mean_diagnostics", mean),
            ("tail_diagnostics", tail),
        ):
            text = diagnostic.pop("limitations")
            if limitations.setdefault(name, text) != text:
                raise ValueError(
                    "an arm's views state different diagnostic limitations"
                )
        mean.pop("expected_annual_loss")
        mean.pop("eal_precision")
        metric["eal_reconciliation"] = tail["convergence"].pop("eal_reconciliation")
        for level, row in metric["tail_metrics"].items():
            support = tail["tails"][level]
            for field in ("var", "tvar", "tvar_estimator"):
                support.pop(field)
            for field in ("var_numerical_interval_95", "tvar_numerical_interval_95"):
                row[field] = support.pop(field)
        estimator = metric["eal_estimator"]
        by_stratum = dict(estimator) if isinstance(estimator, Mapping) else {}
        if by_stratum:
            metric["eal_estimator"] = "+".join(sorted(set(by_stratum.values())))
        views[view] = {
            "eal_estimator_by_stratum": by_stratum,
            "eal_precision_by_stratum": metric["eal_precision"].pop(
                "stratum_estimators", {}
            ),
            **moved,
        }
    return (
        {"limitations": cast(DiagnosticLimitations, limitations), "views": views}
        if views
        else None
    )


def mixture_view_metrics(
    mixture: CapabilityMixture,
    view: str,
    levels: tuple[float, ...],
    controls: TailGateControls | None = None,
) -> dict[str, Any]:
    """Point statistics, statuses and precision diagnostics of one view.

    Tail statuses are final only after ``withhold_unsupported_tails``, which the
    reduction applies once per-state functional reasons are known.
    """
    controls = controls or TailGateControls()
    estimates = tail_estimates(mixture, view, levels)
    gate = view_gate(
        mixture,
        view,
        estimates,
        controls,
        mean_exists=all(
            draw.views[view].mean_exists for draw, _ in mixture.positive_strata()
        ),
    )
    metrics = mixture.metrics(view, levels=levels, estimates=estimates)
    batch_parts = gate.pop("tvar_parts", None) or {}
    return {
        **metrics,
        # Per-batch statistics, and each level's TVaR estimate with its batch
        # parts, for paired arm effects; the reduction removes them before
        # reporting.
        "_batch_values": gate.pop("batch_values", None),
        "_tvar_estimates": {
            str(level): (estimate, batch_parts.get(str(level)))
            for level, estimate in estimates.items()
        },
        "mean_diagnostics": mean_diagnostics(mixture, view, gate=gate, metrics=metrics),
        "tail_diagnostics": tail_diagnostics(
            mixture,
            view,
            levels=levels,
            estimates=estimates,
            controls=controls,
            gate=gate,
        ),
    }


def withhold_unsupported_tails(metric: Mapping[str, Any]) -> None:
    """Mark VaR, TVaR and the annual probability of a loss ``withheld`` where
    the tail gate did not converge.

    A statistic is withheld when the gate is unavailable for its view, when its
    batch-means half-width exceeds the tolerance, or when a batch has fewer than
    ``MIN_TAIL_EFFECTIVE_UNITS`` positive-weight years at or above the VaR (with
    a loss, for the probability). The point values stay in ``tail_metrics`` for
    reference and comparisons. Structurally zero tails keep their exact status.
    """
    statistics = metric["statistics"]
    diagnostics = metric["tail_diagnostics"]
    gate = diagnostics["convergence"]
    positive = gate.get("probability_positive")
    if statistics.get("annual_probability_positive", {}).get("status") == "available":
        reason = (
            f"The tail gate is unavailable: {gate['unavailable_reason']}"
            if positive is None
            else unconverged_reason(
                {"probability": positive, **positive},
                "probability",
                gate,
                gate["years_per_state"],
            )
        )
        if reason is not None:
            statistics["annual_probability_positive"] = statistic_status(
                withheld=reason
            )
    for level, tail in diagnostics["tails"].items():
        if tail["var_interval_method"] == "known_zero_law":
            continue
        row = tail["convergence"]
        for functional in ("var", "tvar"):
            key = f"{functional}::{level}"
            if statistics[key]["status"] != "available":
                continue
            reason = (
                f"The tail gate is unavailable: {gate['unavailable_reason']}"
                if row is None
                else unconverged_reason(row, functional, gate, gate["years_per_state"])
            )
            if reason is not None:
                statistics[key] = statistic_status(withheld=reason)


def _tail_support(
    draw: ConditionalCapabilityDraws, view: str, threshold: float
) -> dict[str, object]:
    """Descriptive support per native independent year, never a coverage test."""
    losses = draw.annual_losses[view]
    weights = draw.normalized_weights()
    strict = losses > threshold
    tail_mass = np.where(strict, weights, 0.0)
    excess = np.maximum(losses - threshold, 0.0)
    useful = (tail_mass > 0) & (excess > 0)
    tail_excess = np.zeros_like(weights)
    if np.any(useful):
        # Normalize products in log space so none overflows or underflows.
        log_excess = np.log(tail_mass[useful]) + np.log(excess[useful])
        tail_excess[useful] = np.exp(log_excess - np.max(log_excess))
    return {
        "independent_unit_count": len(tail_mass),
        "tail_bearing_independent_unit_count": int(np.count_nonzero(tail_mass)),
        "tail_mass_effective_units": effective_sample_size(tail_mass),
        "tail_excess_effective_units": effective_sample_size(tail_excess),
    }


def _tail_mass_effective_units(
    mixture: CapabilityMixture, view: str, threshold: float
) -> float:
    """Kish ESS of target tail mass over every simulated year of the mixture.

    Year ``i`` of stratum ``h`` carries ``m_h * w_hi 1{x_hi > threshold}``. With
    one stratum this is the strict exceedance count; unequal stratum masses
    reduce it.
    """
    masses = [
        mass
        * np.where(draw.annual_losses[view] > threshold, draw.normalized_weights(), 0.0)
        for draw, mass in mixture.positive_strata()
        if not known_zero_law(draw, view)
    ]
    return effective_sample_size(np.concatenate(masses)) if masses else 0.0


def tail_diagnostics(
    mixture: CapabilityMixture,
    view: str,
    *,
    levels: tuple[float, ...] = (0.99, 0.995, 0.999),
    estimates: Mapping[float, TailMeanEstimate] | None = None,
    controls: TailGateControls | None = None,
    gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Tail support and batch-means 95% VaR and TVaR intervals.

    Each interval is the point estimate plus or minus its batch-means
    half-width (``tail_gate``), projected onto nonnegative values for VaR and
    values at least VaR for TVaR. Structurally zero laws have exact zero
    intervals. TVaR is the control-variate estimate (``tail_estimates``); every
    batch uses the coefficient set on all years, and the half-width is that
    estimator's own plus the margin for a certified analytic mean's
    integration error (``tail_gate.tvar_status``), which the standard error
    leaves out.
    """
    if any(not math.isfinite(level) or not 0 < level < 1 for level in levels):
        raise ValueError("tail levels must be finite and strictly between zero and one")
    controls = controls or TailGateControls()
    losses, weights = mixture.samples(view)
    active = mixture.positive_strata()
    mean_exists = all(draw.views[view].mean_exists for draw, _ in active)
    deterministic_zero = all(known_zero_law(draw, view) for draw, _ in active)
    if estimates is None:
        estimates = tail_estimates(mixture, view, levels)
    if gate is None:
        gate = view_gate(mixture, view, estimates, controls, mean_exists=mean_exists)
    absent = (
        None
        if mean_exists
        else "TVaR does not have a finite value because the transformed conditional loss law has no finite mean."
    )
    critical = float(student_t.ppf(0.975, controls.batches - 1))
    output: dict[str, object] = {}
    for level in levels:
        estimate = estimates[level]
        var, tvar = estimate.var, estimate.tvar
        exceeds = losses > var
        tail_weights = weights[exceeds]
        row = None if deterministic_zero else gate["levels"].get(str(level))
        var_interval = tvar_interval = tvar_error = None
        method = var_reason = None
        tvar_reason = absent
        if deterministic_zero:
            method, var_interval = "known_zero_law", [0.0, 0.0]
            tvar_interval = [0.0, 0.0] if mean_exists else None
        elif row is None:
            var_reason = gate["unavailable_reason"]
            tvar_reason = absent or var_reason
        else:
            method = "batch_means"
            half = row["var"]["half_width"]
            var_interval = [max(var - half, 0.0), var + half]
            half = None if row["tvar"] is None else row["tvar"]["half_width"]
            if half is not None:
                tvar_interval = [max(var, tvar - half), tvar + half]
                # The sampling part alone: the integration margin is a bound,
                # not a standard error.
                tvar_error = max(half - estimate.integration_margin, 0.0) / critical
            elif row["tvar"] is not None:
                tvar_reason = row["tvar"]["reason"]
        output[str(level)] = {
            "var": var,
            "tvar": tvar if mean_exists else None,
            "tvar_estimator": estimate.estimator if mean_exists else None,
            # The coefficient-zero estimate, so the control term can be read.
            "tvar_sample_tail_mean": estimate.sample_tvar if mean_exists else None,
            "strict_exceedance_count": int(np.count_nonzero(exceeds)),
            "positive_year_count": int(np.count_nonzero(losses > 0)),
            "simulated_year_count": len(losses),
            "strict_exceedance_probability": float(tail_weights.sum()),
            "tail_weight_effective_sample_size": effective_sample_size(tail_weights),
            "tail_mass_effective_units": _tail_mass_effective_units(mixture, view, var)
            if design_precision_unavailable_reason(
                (draw for draw, _ in active), view=view
            )
            is None
            else None,
            "tail_support_by_stratum": {
                draw.stratum_sha256: _tail_support(draw, view, var)
                for draw, _ in active
            },
            "convergence": row,
            "var_interval_method": method,
            "var_numerical_interval_95": var_interval,
            "tvar_numerical_standard_error": tvar_error,
            "tvar_numerical_interval_95": tvar_interval,
            "var_interval_unavailable_reason": var_reason,
            "tvar_interval_unavailable_reason": None
            if tvar_interval is not None
            else tvar_reason,
        }
    return {
        "tails": output,
        "convergence": {
            key: gate[key]
            for key in ("batches", "years_per_state", "unavailable_reason")
        }
        | {
            "tolerance": controls.tolerance,
            "probability_positive": gate.get("probability_positive"),
            "eal_reconciliation": gate.get("eal_reconciliation"),
        },
        "limitations": f"VaR and TVaR intervals are pointwise batch-means intervals over {controls.batches} contiguous batches of years, not simultaneous across levels, views or arms; a TVaR's half-width adds its control-variate coefficient times the certified integration error bound of a numerically integrated EAL, over one minus the level. A statistic is reported only when its half-width is at most {100 * controls.tolerance:g}% of its value and every batch has at least {MIN_TAIL_EFFECTIVE_UNITS} years at or above the VaR, and a VaR or TVaR only when the years agree with the view's analytic EAL, if it has one, within {EAL_RECONCILIATION_LIMIT} batch-means standard errors (eal_reconciliation); the intervals are asymptotic and treat the TVaR's control-variate coefficient, estimated once on all years, as fixed. Because every year reads fresh parameter ranks, the half-widths include rank as well as count and severity sampling error. A sample that misses the far tail misses it in every batch, so neither the TVaR nor its interval covers what the years do not show. Tail support and ESS are descriptive. All numerical diagnostics condition on captured parameters and stratum masses and exclude model, forecast, elicitation and parameter uncertainty.",
    }


def mean_diagnostics(
    mixture: CapabilityMixture,
    view: str,
    *,
    controls: TailGateControls | None = None,
    gate: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Native EAL precision and the batch-means empirical-mean standard error.

    The EAL uses each stratum's native mean estimator when every positive stratum
    supplies one, so it can differ from the empirical mean. The batch-means
    standard error is an asymptotic numerical diagnostic, not a confidence
    interval: captures record whether a second moment exists but not the tail
    index, and normal intervals for finite-variance heavy tails converge slowly
    once the third moment is infinite. ``metrics``, the view's
    ``CapabilityMixture.metrics`` at any levels, saves computing them again.
    """
    if metrics is None:
        metrics = mixture.metrics(view, levels=())
    active = mixture.positive_strata()
    standard_error = None
    if not all(draw.views[view].mean_exists for draw, _ in active):
        reason = "The transformed conditional loss law has no finite mean."
    elif not all(draw.views[view].variance_exists for draw, _ in active):
        reason = "A batch-means standard error requires a finite second moment of the transformed conditional loss law."
    else:
        if gate is None:
            gate = view_gate(
                mixture, view, {}, controls or TailGateControls(), mean_exists=True
            )
        reason = gate["unavailable_reason"]
        if reason is None:
            standard_error = gate["empirical_mean_standard_error"]
    deterministic_zero = all(known_zero_law(draw, view) for draw, _ in active)
    return {
        "expected_annual_loss": metrics["expected_annual_loss"],
        "eal_precision": deepcopy(metrics["eal_precision"]),
        "empirical_mean_standard_error": standard_error,
        "empirical_mean_standard_error_unavailable_reason": reason,
        "standard_error_basis": "batch_means",
        "empirical_mean_numerical_interval_95": [0.0, 0.0]
        if deterministic_zero
        else None,
        "interval_unavailable_reason": None
        if deterministic_zero
        else reason
        or "No coverage-valid empirical-mean interval is implemented: finite variance does not establish normal-interval coverage for heavy-tailed annual losses.",
        "limitations": "The batch-means standard error is an asymptotic numerical diagnostic, not a confidence interval. Each simulated year reads fresh parameter ranks, so it covers rank, count and severity sampling; it treats the states and their masses as exact. eal_precision is the precision of the reported EAL, deterministic for an analytic mean over every rank. Neither includes model, forecast, elicitation or parameter uncertainty.",
    }
