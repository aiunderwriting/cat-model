"""Compile complete conditional paths into integrated marked incident laws."""

from __future__ import annotations

import math
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from catmodel.contracts.components import (
    AnnualLabLossLaw,
    ContractTerms,
    component_amount_lower_support,
)
from catmodel.contracts.incident import (
    LAST_SLOPE_CAP_SPACINGS,
    CountExperiment,
    EntityPath,
    IncidentLaw,
    JointState,
    LocalResponse,
    MarkKernel,
    ParameterRankLaw,
    Population,
    last_slope_cap,
)
from catmodel.contracts.run_setup import RunSetup
from catmodel.economic_loss import LAB_LOSS_BASIS
from catmodel.hashing import canonical_json_sha256
from catmodel.probability.numerics import positive_product_ratio

if TYPE_CHECKING:
    from catmodel.contracts.run_setup import RunSetup


@dataclass(frozen=True, slots=True)
class CompiledMarkSegment:
    start: float
    end: float
    integrated_rate: float
    weight: float
    kernel: MarkKernel
    source_public_status: str | None = None
    source_model_id: str | None = None
    source_model_revision: str | None = None


@dataclass(frozen=True, slots=True)
class CompiledIncident:
    origin_id: str
    scenario_id: str
    entity_id: str
    annual_rate: float
    count_family: str
    count_variance_to_mean: float
    segments: tuple[CompiledMarkSegment, ...]
    # Read at a fresh rank each simulated year; ``annual_rate`` is its mean.
    parameter_rank: ParameterRankLaw | None = None
    # Drawn once per simulated year whose count is at least one.
    annual_lab_losses: tuple[AnnualLabLossLaw, ...] = ()


@dataclass(frozen=True, slots=True)
class CompiledIncidentState:
    state_id: str
    epistemic_group_id: str
    # The address of the state's parameter-rank streams
    # (``JointState.rank_stream_id``), its own ID unless it declares another.
    rank_stream_id: str
    alternative_id: str
    conditional_key: str
    law_sha256: str
    evidence_sha256: str
    incidents: tuple[CompiledIncident, ...]
    frequency_audit: tuple[dict[str, object], ...]
    contracts: tuple[ContractTerms, ...]
    declared_view_ids: Mapping[str, frozenset[str]]
    count_experiment: CountExperiment | None = None
    view_bases: dict[str, dict[str, str]] = field(default_factory=dict)
    unavailable_views: dict[str, str] = field(default_factory=dict)
    source_status_bases: dict[str, dict[str, str]] = field(default_factory=dict)
    source_status_unavailable: dict[str, str] = field(default_factory=dict)

    def __reduce__(self):
        # Pickled for worker processes; the declared views stay read-only.
        return _compiled_incident_state, tuple(
            dict(self.declared_view_ids)
            if row.name == "declared_view_ids"
            else getattr(self, row.name)
            for row in fields(self)
        )


def _compiled_incident_state(*values) -> CompiledIncidentState:
    state = CompiledIncidentState(*values)
    object.__setattr__(
        state, "declared_view_ids", MappingProxyType(state.declared_view_ids)
    )
    return state


def source_status_view(view: str) -> tuple[str, str] | None:
    """Return source-model status at initiation and its supported linear measure."""
    if not view.startswith("source_status::"):
        return None
    status, _, measure = view.removeprefix("source_status::").partition("::")
    if status in {"public", "not_yet_public"} and (
        measure == "welfare"
        or (measure.startswith("lab_loss::") and measure.removeprefix("lab_loss::"))
    ):
        return status, measure
    return None


def observable_bases(compiled: CompiledIncidentState, views: tuple[str, ...]):
    """The compiled view bases plus those of the requested lab-loss and
    source-status views."""
    bases = dict(compiled.view_bases)
    pooled = bases.get("pooled")
    if pooled is not None:
        for view in views:
            if (
                view.startswith("lab_loss::")
                and view.removeprefix("lab_loss::")
                in compiled.declared_view_ids["lab_loss"]
            ):
                bases[view] = {**pooled, "accounting_standing": LAB_LOSS_BASIS}
    for view in views:
        scoped = source_status_view(view)
        if scoped is None:
            continue
        status, measure = scoped
        basis = compiled.source_status_bases.get(status)
        if basis is None:
            continue
        if measure.startswith("lab_loss::"):
            if (
                measure.removeprefix("lab_loss::")
                not in compiled.declared_view_ids["lab_loss"]
            ):
                continue
            basis = {**basis, "accounting_standing": LAB_LOSS_BASIS}
        bases[view] = {**basis, "source_public_status": status}
    return bases


_DECLARED_VIEW_IDENTIFIERS: dict[
    int, tuple[weakref.ref[RunSetup], Mapping[str, frozenset[str]]]
] = {}


def _setup_view_identifiers(setup: RunSetup) -> Mapping[str, frozenset[str]]:
    """``declared_view_identifiers`` once per setup object.

    Every compiled state of a setup declares the same identifiers, and a
    setup cannot change once constructed, so one read of its laws serves every
    state and arm compiled from it."""
    key = id(setup)
    cached = _DECLARED_VIEW_IDENTIFIERS.get(key)
    if cached is not None and cached[0]() is setup:
        return cached[1]
    value = MappingProxyType(declared_view_identifiers(setup))

    def forget(_: object, key: int = key) -> None:
        _DECLARED_VIEW_IDENTIFIERS.pop(key, None)

    _DECLARED_VIEW_IDENTIFIERS[key] = (weakref.ref(setup, forget), value)
    return value


def declared_view_identifiers(setup: RunSetup) -> dict[str, frozenset[str]]:
    """Bearer, contributor, attribution, lab, scenario and risk-unit IDs.

    Labs come from the scope's lab entities and scenarios from populations;
    the other categories come from laws in every joint state and alternative
    override. Laws are read as their plain encoding, so an alternative's
    stored overrides are read one arm at a time without building its models.
    """
    identifiers: dict[str, set[str]] = {
        "bearer": set(),
        "contributor": set(),
        "attribution": set(),
        "lab_loss": {
            entity.entity_id for entity in setup.scope.entities if entity.kind == "lab"
        },
        "scenario": {population.scenario_id for population in setup.scope.populations},
        "risk_unit": set(),
    }

    def declare(law: Mapping[str, Any]) -> None:
        identifiers["risk_unit"].add(law["origin_id"])
        for segment in law["segments"]:
            kernel = segment["kernel"]
            components = list(kernel.get("campaign_components") or ())
            outcomes = list(kernel.get("member_outcomes") or ())
            for branch in kernel.get("explicit_joint_members") or ():
                components.extend(branch.get("campaign_components") or ())
                outcomes.extend(branch.get("members") or ())
            components.extend(
                component
                for outcome in outcomes
                for component in outcome.get("components") or ()
            )
            for component in components:
                if component.get("bearer_id") is not None:
                    identifiers["bearer"].add(component["bearer_id"])
                identifiers["contributor"].update(
                    component.get("contributor_ids") or ()
                )
                identifiers["attribution"].update(
                    component.get("attribution_weights") or ()
                )

    for state in setup.joint_states:
        for law in state.laws:
            declare(law.model_dump(mode="json"))
    for alternative_id in setup.overrides.arms():
        for override in setup.overrides.inline(alternative_id):
            declare(override["law"])
    return {category: frozenset(values) for category, values in identifiers.items()}


def _coordinate_difference(left: float, right: float) -> tuple[float, int]:
    """Represent a finite-coordinate difference without overflowing its span."""
    exponent = max(
        (math.frexp(value)[1] for value in (left, right) if value != 0),
        default=0,
    )
    return math.ldexp(left, -exponent) - math.ldexp(right, -exponent), exponent


def _scaled_coordinate_ratio(factor, numerator, denominator) -> float:
    """Apply a coordinate ratio before rounding the resulting ordinate."""
    factor_part, factor_power = math.frexp(factor)
    numerator_part, numerator_power = math.frexp(numerator[0])
    denominator_part, denominator_power = math.frexp(denominator[0])
    try:
        return math.ldexp(
            factor_part * numerator_part / denominator_part,
            factor_power
            + numerator_power
            + numerator[1]
            - denominator_power
            - denominator[1],
        )
    except OverflowError:
        return math.copysign(math.inf, factor_part * numerator_part / denominator_part)


def _path_relative_value(path: EntityPath, time: float, anchor: float):
    if time == 1:
        return _coordinate_difference(path.knots[-1].value, anchor)
    for left, right in pairwise(path.knots):
        if left.time <= time < right.time:
            relative = _coordinate_difference(left.value, anchor)
            if path.interpolation == "step" or time == left.time:
                return relative
            end_relative = _coordinate_difference(right.value, anchor)
            exponent = max(
                (power for part, power in (relative, end_relative) if part != 0),
                default=0,
            )
            fraction = (time - left.time) / (right.time - left.time)
            return (
                math.fsum(
                    (
                        (1 - fraction)
                        * math.ldexp(relative[0], relative[1] - exponent),
                        fraction
                        * math.ldexp(end_relative[0], end_relative[1] - exponent),
                    )
                ),
                exponent,
            )
    raise ValueError("path evaluation lies outside the study period")


def _response_coefficients(response: LocalResponse, path: EntityPath, time: float):
    points = response.points
    if response.extrapolation == "unavailable" and (
        _path_relative_value(path, time, points[0].value)[0] < 0
        or _path_relative_value(path, time, points[-1].value)[0] > 0
    ):
        raise ValueError("local response is outside its authored coordinate support")
    ordinates = [point.multiplier for point in points]
    if response.interpolation == "log":
        ordinates = [math.log(y) for y in ordinates]
    if len(points) == 1:
        return points[0].value, ordinates[0], None
    if _path_relative_value(path, time, points[0].value)[0] < 0:
        # Below the lowest point both continuations are flat, as for elicited
        # responses; last_slope differs from flat only above the highest point.
        return points[0].value, ordinates[0], None
    elif _path_relative_value(path, time, points[-1].value)[0] >= 0:
        if response.extrapolation == "flat":
            return points[-1].value, ordinates[-1], None
        cap = last_slope_cap([point.value for point in points])
        if _path_relative_value(path, time, cap)[0] >= 0:
            # Capped like elicited responses: flat at the continued value.
            return (
                cap,
                ordinates[-1]
                + LAST_SLOPE_CAP_SPACINGS * (ordinates[-1] - ordinates[-2]),
                None,
            )
        index = len(points) - 2
    else:
        index = next(
            i
            for i in range(len(points) - 1)
            if _path_relative_value(path, time, points[i].value)[0] >= 0
            and _path_relative_value(path, time, points[i + 1].value)[0] < 0
        )
    difference = ordinates[index + 1] - ordinates[index]
    span = _coordinate_difference(points[index + 1].value, points[index].value)
    return points[index].value, ordinates[index], (difference, span)


def _response_breakpoints(response: LocalResponse) -> tuple[float, ...]:
    """Coordinates where the response's affine piece changes, including its cap."""
    values = tuple(point.value for point in response.points)
    if response.extrapolation == "last_slope" and len(values) > 1:
        values = (*values, last_slope_cap(values))
    return values


def _response_level(path, time, anchor, ordinate, change):
    if change is None or change[0] == 0:
        return ordinate
    relative = _path_relative_value(path, time, anchor)
    adjustment = _scaled_coordinate_ratio(change[0], relative, change[1])
    if not math.isfinite(adjustment):
        raise ValueError(
            "local response ordinate is outside the numerical floating-point range"
        )
    return ordinate + adjustment


def _log_product(numerators, denominators, exponential) -> float:
    """Natural log of a product of positive numerators over positive
    denominators times ``exp(exponential)``."""
    return math.fsum(
        (
            *(math.log(value) for value in numerators),
            *(-math.log(value) for value in denominators),
            exponential,
        )
    )


def _integrate_local_response(
    path,
    response,
    start,
    end,
    *,
    factors=(),
    divisors=(),
    quantity="integrated local response",
):
    if not 0 <= start < end <= 1:
        raise ValueError("response integration interval must lie in the study period")
    boundaries = {start, end}
    boundaries.update(knot.time for knot in path.knots if start < knot.time < end)
    if path.interpolation == "linear":
        for left, right in pairwise(path.knots):
            if left.value == right.value:
                continue
            for breakpoint in _response_breakpoints(response):
                fraction = _scaled_coordinate_ratio(
                    1.0,
                    _coordinate_difference(breakpoint, left.value),
                    _coordinate_difference(right.value, left.value),
                )
                crossing = left.time + fraction * (right.time - left.time)
                if 0 < fraction < 1 and start < crossing < end:
                    boundaries.add(crossing)
    pieces = []
    # Whether a positive piece was too small to represent.
    underflow = False
    for left, right in pairwise(sorted(boundaries)):
        if response.extrapolation == "unavailable":
            for time in (
                (left, right)
                if path.interpolation == "linear"
                else ((left + right) / 2,)
            ):
                if (
                    _path_relative_value(path, time, response.points[0].value)[0] < 0
                    or _path_relative_value(path, time, response.points[-1].value)[0]
                    > 0
                ):
                    raise ValueError(
                        "local response is outside its authored coordinate support"
                    )
        anchor, ordinate, change = _response_coefficients(
            response, path, (left + right) / 2
        )
        level0 = _response_level(path, left, anchor, ordinate, change)
        level1 = (
            level0
            if path.interpolation == "step"
            else _response_level(path, right, anchor, ordinate, change)
        )
        if response.interpolation == "linear":
            if min(level0, level1) < 0:
                raise ValueError(
                    "local response is negative on an authored path interval"
                )
            largest = max(level0, level1)
            if largest == 0:
                continue
            numerators, denominators, exponential = (
                (right - left, largest, level0 / largest + level1 / largest, *factors),
                (2.0, *divisors),
                0.0,
            )
        else:
            delta = level1 - level0
            if delta == 0:
                numerators, denominators, exponential = (
                    (right - left, *factors),
                    divisors,
                    level0,
                )
            else:
                numerators, denominators, exponential = (
                    (right - left, -math.expm1(-abs(delta)), *factors),
                    (abs(delta), *divisors),
                    max(level0, level1),
                )
        try:
            pieces.append(
                positive_product_ratio(
                    numerators,
                    denominators,
                    quantity=quantity,
                    exponential=exponential,
                )
            )
        except ValueError:
            # A positive piece too small to represent changes the integral by
            # less than the smallest subnormal; one too large to represent
            # makes the integral unrepresentable.
            if _log_product(numerators, denominators, exponential) > 0:
                raise
            underflow = True
    try:
        total = math.fsum(pieces)
    except OverflowError as exc:
        raise ValueError(
            "integrated local response is outside the numerical floating-point range"
        ) from exc
    if not math.isfinite(total):
        raise ValueError(
            "integrated local response is outside the numerical floating-point range"
        )
    if total == 0 and underflow:
        raise ValueError(
            f"positive {quantity} is outside the numerical floating-point range"
        )
    return total


def _compile_incident(law: IncidentLaw, path: EntityPath) -> CompiledIncident:
    rates = []
    for segment in law.segments:
        if (
            law.reference_annual_rate == 0
            or segment.initiation_probability == 0
            or segment.reference_exposure_share == 0
        ):
            rates.append(0.0)
            continue
        rate = _integrate_local_response(
            path,
            law.response,
            segment.start,
            segment.end,
            factors=(
                law.reference_annual_rate,
                segment.initiation_probability,
                segment.reference_exposure_share,
            ),
            divisors=(segment.end - segment.start,),
            quantity="integrated initiating intensity",
        )
        rates.append(rate)
    try:
        rate = math.fsum(rates)
    except OverflowError as exc:
        raise ValueError(
            "integrated annual initiating intensity is outside the numerical floating-point range"
        ) from exc
    if not math.isfinite(rate):
        raise ValueError(
            "integrated annual initiating intensity is outside the numerical floating-point range"
        )
    segments = tuple(
        CompiledMarkSegment(
            start=segment.start,
            end=segment.end,
            integrated_rate=segment_rate,
            weight=segment_rate / rate if rate > 0 else 0,
            kernel=segment.kernel,
            source_public_status=segment.source_public_status,
            source_model_id=segment.source_model_id,
            source_model_revision=segment.source_model_revision,
        )
        for segment, segment_rate in zip(law.segments, rates)
    )
    return CompiledIncident(
        origin_id=law.origin_id,
        scenario_id=law.scenario_id,
        entity_id=law.entity_id,
        annual_rate=rate,
        count_family=law.count_family,
        count_variance_to_mean=law.count_variance_to_mean,
        segments=segments,
        parameter_rank=law.parameter_rank,
        annual_lab_losses=law.annual_lab_losses,
    )


def _component_lower_bounds(components):
    if not components:
        return []
    grouped = {}
    independent = []
    for component in components:
        if component.severity is None or component.rank_group_id is None:
            independent.append(component)
            continue
        key = (
            component.severity.model_dump_json(),
            component.rank_scope,
            component.rank_group_id,
            component.multiplier,
        )
        grouped.setdefault(key, []).append(component)
    # Sum shares before scaling one shared lower bound; separate products can
    # round a conserving partition below its exact qualifying floor.
    independent.extend(
        values[0].model_copy(
            update={
                "post_multiplier": math.fsum(value.post_multiplier for value in values)
            }
        )
        for values in grouped.values()
    )
    return [component_amount_lower_support(component) for component in independent]


def _terminal_member_support(member, origin_id):
    bounds = _component_lower_bounds(member.components)
    if not member.composition_complete or not member.overlap_resolved:
        if member.identified_gross_total is None:
            raise ValueError(
                f"{origin_id}: qualifying terminal-total support is unidentified because member composition or overlap is unresolved"
            )
        return member.identified_gross_total, True
    return math.fsum(lower for lower, _ in bounds), all(atom for _, atom in bounds)


def _assert_terminal_threshold(population, support, origin_id):
    if population.threshold is None:
        return
    lower, atom = support
    threshold = population.threshold
    if lower < threshold or (
        lower == threshold and population.threshold_comparator == "gt" and atom
    ):
        comparator = ">" if population.threshold_comparator == "gt" else ">="
        raise ValueError(
            f"{origin_id}: surviving terminal-total support contradicts qualification {comparator} {threshold:g}; lower support is {lower:g}"
            + (" with an atom" if atom else " without a lower atom")
        )


def _validate_terminal_threshold(
    law: IncidentLaw, compiled: CompiledIncident, population: Population
) -> None:
    for segment in compiled.segments:
        kernel = segment.kernel
        if segment.integrated_rate == 0:
            continue
        _component_lower_bounds(kernel.campaign_components)
        for branch in kernel.explicit_joint_members:
            if branch.probability > 0:
                _component_lower_bounds(branch.campaign_components)
        if (
            kernel.shared_probability == 0
            or kernel.unit_probability == 0
            or kernel.expected_member_count() == 0
        ):
            continue
        for member in kernel.possible_member_outcomes():
            support = (
                _terminal_member_support(member, law.origin_id)
                if population.threshold is not None
                else None
            )
            if support is None:
                _component_lower_bounds(member.components)
            else:
                _assert_terminal_threshold(population, support, law.origin_id)


def _validate_report_populations(
    setup: RunSetup, state, law, compiled, unavailable_views
):
    populations = {row.scenario_id: row for row in setup.scope.populations}
    source = populations[law.scenario_id]
    bindings = {
        (row.source_scenario_id, row.target_scenario_id): row
        for row in setup.scope.report_population_bindings
    }
    structural_zero = {
        row.scenario_id
        for row in state.absent_populations
        if row.disposition == "structural_zero"
    }
    consumed = {}
    unresolved_candidates: set[str] = set()
    asserted_targets: set[str] = set()

    def bind(target_id):
        asserted_targets.add(target_id)
        target = populations[target_id]
        if target_id == law.scenario_id:
            return target, True
        if target_id in structural_zero:
            raise ValueError(
                "a positive-support report tag contradicts a structural-zero population"
            )
        for basis_field in ("currency", "monetary_basis", "accounting_standing"):
            if getattr(source, basis_field) != getattr(target, basis_field):
                raise ValueError(
                    f"report population {target_id!r} has incompatible {basis_field}; a report label cannot convert a scientific basis"
                )
        alias = source.model_dump(exclude={"scenario_id"}) == target.model_dump(
            exclude={"scenario_id"}
        )
        if alias:
            return target, True
        binding = bindings.get((law.scenario_id, target_id))
        if binding is None:
            raise ValueError(
                f"report population {target_id!r} requires an explicit same-terminal-members scientific binding"
            )
        if any(
            getattr(source, field) != getattr(target, field)
            for field in ("initiating_unit", "terminal_unit", "severity_unit")
        ):
            raise ValueError(
                "same-terminal-members report bindings cannot convert initiating, terminal or severity units"
            )
        consumed[(law.scenario_id, target_id)] = binding
        return target, False

    for segment in compiled.segments:
        if segment.integrated_rate == 0:
            continue
        kernel = segment.kernel
        branches = [
            branch for branch in kernel.explicit_joint_members if branch.probability > 0
        ] or [None]
        for branch in branches:
            members = (
                branch.members
                if branch is not None
                else kernel.possible_member_outcomes()
            )
            campaign = kernel.campaign_components + (
                () if branch is None else branch.campaign_components
            )
            campaign_targets = {
                tag
                for component in campaign
                for tag in (component.scenario_ids or (law.scenario_id,))
            }
            for target_id in campaign_targets:
                target, alias = bind(target_id)
                if (
                    alias
                    or target.threshold is None
                    or (target.threshold, target.threshold_comparator)
                    == (source.threshold, source.threshold_comparator)
                ):
                    continue
                count_positive = (
                    bool(branch.members)
                    if branch is not None
                    else (
                        kernel.cluster is not None
                        and min(
                            row.count
                            for row in kernel.cluster.support
                            if row.probability > 0
                        )
                        >= 1
                    )
                )
                if (
                    not members
                    or not count_positive
                    or kernel.shared_probability != 1
                    or kernel.unit_probability != 1
                ):
                    raise ValueError(
                        "campaign report binding requires an identified qualifying terminal-member scope; campaign response costs cannot establish terminal qualification"
                    )
                for member in members:
                    _assert_terminal_threshold(
                        target,
                        _terminal_member_support(member, law.origin_id),
                        law.origin_id,
                    )
            if (
                kernel.shared_probability == 0
                or kernel.unit_probability == 0
                or not members
                or kernel.expected_member_count() == 0
            ):
                continue
            for member in members:
                candidates = set(member.scenario_ids)
                ambiguous_fallback = (
                    not member.composition_complete
                    and member.identified_gross_total is not None
                    and len(candidates) > 1
                )
                if ambiguous_fallback:
                    unresolved_candidates.update(candidates)
                    # Unidentified allocation does not make the unique gross unknown.
                    for candidate in candidates:
                        unavailable_views.setdefault(
                            f"scenario::{candidate}",
                            "Identified unique gross does not identify its allocation among the declared candidate populations.",
                        )
                targets = set() if ambiguous_fallback else candidates
                for component in member.components:
                    if component.scenario_ids:
                        targets.update(component.scenario_ids)
                    elif not ambiguous_fallback:
                        targets.update(member.scenario_ids or (law.scenario_id,))
                for target_id in targets:
                    target, _ = bind(target_id)
                    if target.threshold is not None:
                        _assert_terminal_threshold(
                            target,
                            _terminal_member_support(member, law.origin_id),
                            law.origin_id,
                        )
    return consumed, unresolved_candidates, asserted_targets


def _state(setup: RunSetup, state_id: str) -> JointState:
    try:
        return next(row for row in setup.joint_states if row.state_id == state_id)
    except StopIteration as exc:
        raise ValueError(f"unknown joint state {state_id!r}") from exc


def _law_projection(law: IncidentLaw) -> dict[str, object]:
    payload = law.model_dump(mode="json", exclude={"evidence_ids"})
    for segment in payload["segments"]:
        kernel = segment["kernel"]
        component_sets = [kernel["campaign_components"]]
        component_sets.extend(
            outcome["components"] for outcome in kernel["member_outcomes"]
        )
        for branch in kernel["explicit_joint_members"]:
            component_sets.append(branch["campaign_components"])
            component_sets.extend(member["components"] for member in branch["members"])
        for components in component_sets:
            for component in components:
                component.pop("report_labels", None)
    return payload


@dataclass(frozen=True, slots=True)
class _CheckedState:
    """One joint state's laws compiled and checked under one arm, before its
    identity digests and view bases are worked out."""

    state: JointState
    experiment: CountExperiment | None
    laws: Mapping[str, IncidentLaw]
    paths: Mapping[str, EntityPath]
    populations: Mapping[str, Population]
    selected: list[Population]
    unavailable_views: dict[str, str]
    pooled_reasons: list[str]
    view_bases: dict[str, dict[str, str]]
    incidents: tuple[CompiledIncident, ...]
    consumed_bindings: dict
    unresolved_candidates: set[str]
    asserted_targets: set[str]


def check_incident_state(
    setup: RunSetup,
    state_id: str,
    alternative_id: str = "reference",
) -> None:
    """Refuse a joint state ``compile_incident_state`` would refuse, without
    compiling the rest of it.

    Every refusal comes from compiling each law along its entity's path and
    checking its qualifying support and report populations. The identity
    digests and view bases that follow refuse nothing, and a caller that only
    asks whether a law is admitted (``contextual_node.admit_node``) has no
    use for them.
    """
    _checked_state(setup, state_id, alternative_id)


def _checked_state(
    setup: RunSetup, state_id: str, alternative_id: str
) -> _CheckedState:
    state = _state(setup, state_id)
    alternatives = {row.alternative_id: row for row in setup.alternatives}
    if alternative_id != "reference" and alternative_id not in alternatives:
        raise ValueError(f"unknown study alternative {alternative_id!r}")
    laws = {row.origin_id: row for row in state.laws}
    experiment = None
    if alternative_id != "reference":
        alternative = alternatives[alternative_id]
        experiment = alternative.count_experiment
        base_weights = {row.state_id: row.weight for row in setup.joint_states}
        changed_weights = alternative.state_weights is not None and any(
            alternative.state_weights.get(key, 0) != weight
            for key, weight in base_weights.items()
        )
        if alternative.purpose == "numerical_refinement" and (
            alternative_id in setup.overrides.arms()
            or alternative.count_experiment is not None
            or changed_weights
        ):
            raise ValueError(
                "numerical refinement cannot mutate a scientific incident law or mixture weights"
            )
        laws.update(
            {
                law.origin_id: law
                for law in setup.overrides.laws(alternative_id, state_id)
            }
        )
    paths = {row.entity_id: row for row in state.entity_paths}
    populations = {row.scenario_id: row for row in setup.scope.populations}
    selected = [populations[row.scenario_id] for row in laws.values()]
    unavailable_views = {
        f"scenario::{row.scenario_id}": row.reason
        for row in state.absent_populations
        if row.disposition == "unavailable"
    }
    pooled_reasons = []
    if unavailable_views:
        pooled_reasons.append(
            "Pooled views include unquantified populations: "
            + "; ".join(
                f"{key}: {reason}" for key, reason in sorted(unavailable_views.items())
            )
        )
    view_bases = {
        f"scenario::{population.scenario_id}": {
            "currency": population.currency,
            "monetary_basis": population.monetary_basis,
            "accounting_standing": population.accounting_standing,
            "severity_unit": population.severity_unit,
        }
        for population in setup.scope.populations
    }
    incidents = tuple(
        _compile_incident(laws[key], paths[laws[key].entity_id]) for key in sorted(laws)
    )
    unavailable_lab_losses: dict[str, list[str]] = {}
    for incident in incidents:
        if incident.annual_rate > 0:
            for lab_id, reason in laws[
                incident.origin_id
            ].unavailable_lab_losses.items():
                unavailable_lab_losses.setdefault(lab_id, []).append(
                    f"{incident.scenario_id}: {reason}"
                )
    unavailable_views.update(
        {
            f"lab_loss::{lab_id}": "; ".join(dict.fromkeys(reasons))
            for lab_id, reasons in unavailable_lab_losses.items()
        }
    )
    consumed_bindings = {}
    unresolved_candidates: set[str] = set()
    asserted_targets: set[str] = set()
    for incident in incidents:
        _validate_terminal_threshold(
            laws[incident.origin_id], incident, populations[incident.scenario_id]
        )
        bindings, candidates, asserted = _validate_report_populations(
            setup, state, laws[incident.origin_id], incident, unavailable_views
        )
        consumed_bindings.update(bindings)
        unresolved_candidates.update(candidates)
        asserted_targets.update(asserted)
    return _CheckedState(
        state=state,
        experiment=experiment,
        laws=laws,
        paths=paths,
        populations=populations,
        selected=selected,
        unavailable_views=unavailable_views,
        pooled_reasons=pooled_reasons,
        view_bases=view_bases,
        incidents=incidents,
        consumed_bindings=consumed_bindings,
        unresolved_candidates=unresolved_candidates,
        asserted_targets=asserted_targets,
    )


def compile_incident_state(
    setup: RunSetup,
    state_id: str,
    alternative_id: str = "reference",
) -> CompiledIncidentState:
    """Compile one joint state under the reference law or one alternative.

    The state is one point of sparse joint support, compiled as authored; no
    product (independence) weights are imposed. Reduction applies the
    joint-state weights.
    """
    checked = _checked_state(setup, state_id, alternative_id)
    state, experiment, laws, paths = (
        checked.state,
        checked.experiment,
        checked.laws,
        checked.paths,
    )
    populations, selected = checked.populations, checked.selected
    unavailable_views, pooled_reasons = (
        checked.unavailable_views,
        checked.pooled_reasons,
    )
    view_bases, incidents = checked.view_bases, checked.incidents
    consumed_bindings = checked.consumed_bindings
    unresolved_candidates = checked.unresolved_candidates
    asserted_targets = checked.asserted_targets
    positive_origins = {
        incident.scenario_id for incident in incidents if incident.annual_rate > 0
    }
    pure_candidates = unresolved_candidates - positive_origins - asserted_targets
    basis_populations = [
        population
        for population in setup.scope.populations
        if population.scenario_id not in pure_candidates
    ]
    pooled_basis_unavailable_reasons = []
    if (
        len({row.currency for row in basis_populations}) != 1
        or len({row.monetary_basis for row in basis_populations}) != 1
    ):
        pooled_basis_unavailable_reasons.append(
            "Pooled views require one explicit currency and monetary basis; only separately scoped scenario views are available."
        )
    if len({row.accounting_standing for row in basis_populations}) != 1:
        pooled_basis_unavailable_reasons.append(
            "Pooled views require the same accounting standing; only separately scoped scenario views are available."
        )
    if not pooled_basis_unavailable_reasons:
        representative = view_bases[f"scenario::{basis_populations[0].scenario_id}"]
        view_bases["pooled"] = {
            key: value
            for key, value in representative.items()
            if key != "severity_unit"
        }
    else:
        pooled_reasons = pooled_basis_unavailable_reasons + pooled_reasons
    if pooled_reasons:
        unavailable_views["pooled"] = " ".join(pooled_reasons)
    source_status_bases = {}
    source_status_unavailable = {}
    lab_ids = [
        entity.entity_id for entity in setup.scope.entities if entity.kind == "lab"
    ]
    unknown_status = [
        f"{incident.origin_id}/segment/{index}"
        for incident in incidents
        for index, segment in enumerate(incident.segments)
        if segment.integrated_rate > 0 and segment.source_public_status is None
    ]
    for status in ("public", "not_yet_public"):
        scoped_scenarios = {
            incident.scenario_id
            for incident in incidents
            if any(
                segment.integrated_rate > 0 and segment.source_public_status == status
                for segment in incident.segments
            )
        }
        candidates = [
            population
            for population in basis_populations
            if population.scenario_id in scoped_scenarios
        ] or basis_populations
        scoped_bases = [
            {
                key: value
                for key, value in view_bases[
                    f"scenario::{population.scenario_id}"
                ].items()
                if key != "severity_unit"
            }
            for population in candidates
        ]
        reasons = []
        if scoped_bases and all(basis == scoped_bases[0] for basis in scoped_bases):
            source_status_bases[status] = scoped_bases[0]
        else:
            reasons.append(
                "Source-status views require a common currency, monetary basis and accounting standing."
            )
        if unknown_status:
            reasons.append(
                "Positive-rate incident segments have unknown source-model public status: "
                + ", ".join(unknown_status)
            )
        for absence in state.absent_populations:
            if absence.disposition == "unavailable":
                reasons.append(f"{absence.scenario_id}: {absence.reason}")
        if reasons:
            for measure in ("welfare", *(f"lab_loss::{lab}" for lab in lab_ids)):
                source_status_unavailable[f"source_status::{status}::{measure}"] = (
                    "; ".join(reasons)
                )
        for incident in incidents:
            if any(
                segment.integrated_rate > 0 and segment.source_public_status == status
                for segment in incident.segments
            ):
                for lab, reason in laws[
                    incident.origin_id
                ].unavailable_lab_losses.items():
                    source_status_unavailable[
                        f"source_status::{status}::lab_loss::{lab}"
                    ] = f"{incident.scenario_id}: {reason}"
    consumed_laws = [_law_projection(laws[key]) for key in sorted(laws)]
    mathematical = {
        "laws": consumed_laws,
        "entity_paths": [
            paths[key].model_dump(mode="json")
            for key in sorted({law.entity_id for law in laws.values()})
        ],
        "contracts": [row.model_dump(mode="json") for row in setup.contracts],
        "physical_factor_state": state.physical_factor_state,
        "population_absences": [
            row.model_dump(mode="json") for row in state.absent_populations
        ],
        "count_experiment": None
        if experiment is None
        else experiment.model_dump(mode="json"),
    }
    law_sha256 = canonical_json_sha256(mathematical)
    roster = {entity.entity_id: entity for entity in setup.scope.entities}
    selected_entities = {law.entity_id for law in laws.values()} | {
        law.deployment_id for law in laws.values() if law.deployment_id is not None
    }
    while True:
        closure = selected_entities | {
            related
            for key in selected_entities
            for related in roster[key].related_entity_ids
        }
        if closure == selected_entities:
            break
        selected_entities = closure
    conditional_key = canonical_json_sha256(
        {
            "law_sha256": law_sha256,
            "report_population_bindings": [
                row.model_dump(mode="json")
                for _, row in sorted(consumed_bindings.items())
            ],
            "populations": [
                populations[key].model_dump(mode="json") for key in sorted(populations)
            ],
            "entities": [
                entity.scientific_projection()
                for entity in setup.scope.entities
                if entity.entity_id in selected_entities
            ],
            "period_start": setup.scope.period_start.isoformat(),
            "period_end": setup.scope.period_end.isoformat(),
            "capability_scales": [
                row.model_dump(mode="json")
                for row in setup.scope.capability_scales
                if row.scale_id in {path.scale_id for path in state.entity_paths}
            ],
            "deployments": [
                row.model_dump(mode="json")
                for row in setup.scope.deployments
                if row.deployment_id
                in {
                    deployment
                    for population in selected
                    for deployment in population.deployment_ids
                }
                | {
                    law.deployment_id
                    for law in laws.values()
                    if law.deployment_id is not None
                }
            ],
            "independence_assumptions": setup.assumptions.independence,
        }
    )
    selected_evidence = {key for law in laws.values() for key in law.evidence_ids}
    evidence_sha256 = canonical_json_sha256(
        [
            row.model_dump(mode="json")
            for row in setup.evidence.records
            if row.evidence_id in selected_evidence
        ]
    )
    audit: tuple[dict[str, object], ...] = tuple(
        dict[str, object](
            {
                "origin_id": incident.origin_id,
                "count_authority": "initiating_origin",
                "integrated_annual_rate": incident.annual_rate,
                "segment_mark_weights": [
                    segment.weight for segment in incident.segments
                ],
                "segment_integrated_rates": [
                    segment.integrated_rate for segment in incident.segments
                ],
                "input_stage": laws[incident.origin_id].input_stage,
            }
        )
        for incident in incidents
    )
    return CompiledIncidentState(
        state_id,
        state.epistemic_group_id,
        state.rank_stream_id or state_id,
        alternative_id,
        conditional_key,
        law_sha256,
        evidence_sha256,
        incidents,
        audit,
        setup.contracts,
        _setup_view_identifiers(setup),
        experiment,
        view_bases,
        unavailable_views,
        source_status_bases,
        source_status_unavailable,
    )
