"""Event-addressed incident simulation under frozen numerical controls."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any, NamedTuple

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import gamma

from catmodel.contracts.components import ComponentObligation, ComponentPayment
from catmodel.contracts.incident import (
    WORK_ALLOWANCES,
    ExecutionPlan,
    ParameterRankLaw,
    parse_scenario_view,
)
from catmodel.contracts.native_severity import native_severity_sampler
from catmodel.elicitation.epistemic import (
    EPISTEMIC_KNOT_SCORES,
    epistemic_mean,
    stretched_score_quantiles,
)
from catmodel.hashing import canonical_json_sha256
from catmodel.probability.count_inversion import (
    binomial_counts,
    negative_binomial_counts,
    poisson_counts,
)
from catmodel.probability.distributions import (
    sample_affine_capped_severity,
    uncapped_infinite_variance_gpd_shapes,
)
from catmodel.probability.keyed_streams import (
    INCIDENT_STREAM_CONTRACT,
    VECTOR_EVENT_STREAM_CONTRACT,
    event_uniforms,
    incident_stream_generator,
    keyed_stream_generator,
)
from catmodel.simulation.authored.payments import (
    native_payment_positive_mean_lower_bound,
)
from catmodel.simulation.capability_mixture import (
    DETERMINISTIC_EAL_PRECISION_BASIS,
    ConditionalCapabilityDraws,
    ConditionalView,
    NativeMean,
)
from catmodel.simulation.components import (
    aggregate_lab_loss_has_positive_support,
    component_has_positive_support,
    component_view_weight,
    defense_loading_scope_reason,
    defense_loading_state,
    episode_lab_loss_caps,
)
from catmodel.simulation.execution_budget import (
    WorkLimitExceeded,
    work_allowances,
    work_limits,
)
from catmodel.simulation.incident import (
    CompiledIncidentState,
    observable_bases,
    source_status_view,
)
from catmodel.simulation.incident_campaigns import (
    CampaignLaw,
    EpisodeDraws,
    campaign_law,
    draw_episodes,
    episode_members,
    gross_values,
)
from catmodel.simulation.incident_linear import (
    EventCell,
    LinearProjection,
    SingleMemberLaw,
    draw_single_member_events,
    linear_projection,
    linear_values,
    mark_projection,
    single_member_law,
)
from catmodel.simulation.incident_moments import (
    analytic_known_zero_views,
    analytic_linear_eals,
    annual_views,
)
from catmodel.simulation.policy_terms import (
    INCOMPLETE_FINANCIAL_MAPPING,
    CoveredParts,
    PolicyReduction,
    PolicyShare,
    policy_shares,
    share_parts,
)
from catmodel.simulation.small_events import (
    SIMULATED_RANK_RANGE,
    SmallEventBuckets,
    SmallEventControls,
    SmallEventLaw,
    body_variance,
    fixed_rate_buckets,
    ranked_buckets,
)


def state_key(state_id: str) -> str:
    """A joint state's stratum key, the digest of its ID, which keys its
    draws in a capture and its target weight."""
    return canonical_json_sha256({"joint_state_id": state_id})


def state_stream_seed(seed: int, state_id: str) -> int:
    """The root seed of a joint state's random streams, from the plan seed
    and the state's ID."""
    return int(canonical_json_sha256({"seed": seed, "state_id": state_id})[:16], 16)


def _campaign_spec(kernel, branch=None):
    common = kernel.campaign_components
    specific = () if branch is None else branch.campaign_components
    return (
        common + specific,
        kernel.campaign_obligations
        + (() if branch is None else branch.campaign_obligations),
        kernel.campaign_memberships
        + (() if branch is None else branch.campaign_memberships),
        kernel.campaign_payments + (() if branch is None else branch.campaign_payments),
        (not common or kernel.campaign_financial_mapping_complete)
        and (
            branch is None or not specific or branch.campaign_financial_mapping_complete
        ),
        tuple(
            item
            for item in (
                kernel.campaign_payment_kernel,
                None if branch is None else branch.campaign_payment_kernel,
            )
            if item is not None
        ),
    )


def _risk_unit_view(view):
    prefix, suffix = "risk_unit::", "::occurrence_net_financial"
    return (
        view[len(prefix) : -len(suffix)]
        if view.startswith(prefix) and view.endswith(suffix)
        else None
    )


def _financial_view(view):
    return view in {"financial", "oep_financial"} or _risk_unit_view(view) is not None


def _needs_financial(views, origin_id=None):
    return any(
        view in {"financial", "oep_financial"}
        or (origin_id is not None and _risk_unit_view(view) == origin_id)
        for view in views
    )


def _affects_view(
    law,
    view,
    scenario_ids,
    origin_id,
    source_public_status=None,
    *,
    capped_lab_ids=frozenset(),
):
    status_scope = source_status_view(view)
    if status_scope is not None:
        status, view = status_scope
        if status != source_public_status:
            return False
    scoped = parse_scenario_view(view)
    if scoped is not None:
        scenario, view = scoped
        if scenario not in (law.scenario_ids or scenario_ids):
            return False
    risk_unit = _risk_unit_view(view)
    if risk_unit is not None:
        return risk_unit == origin_id
    if (
        view.startswith("lab_loss::")
        and view.removeprefix("lab_loss::") in capped_lab_ids
    ):
        # A finite episode cap bounds every moment of this lab view.
        return False
    # component_view_weight rejects an unknown nature; admit the law here, as
    # its mark withholds the welfare view (``_MarkShape``).
    if view.partition("::")[0] == "lab_loss" and law.lab_loss_weights is None:
        return True
    if view.partition("::")[0] == "welfare" and law.nature is None:
        return True
    return component_view_weight(law, view, default=1.0) > 0


def _financial_bounded(
    component_id, obligations, memberships, payments, contracts, *, annual
):
    obligations_by_entity = {
        (item.component_id, item.entity_id): item for item in obligations
    }
    contracts_by_id = {item.contract_id: item for item in contracts}
    payments_by_entity = {
        (item.component_id, item.entity_id): item for item in payments
    }
    for membership in memberships:
        if (
            membership.component_id != component_id
            or not membership.eligible
            or membership.covered_fraction == 0
        ):
            continue
        obligation = obligations_by_entity.get(
            (component_id, membership.obligated_entity_id)
        )
        if obligation is None:
            return False
        payment = payments_by_entity.get((component_id, membership.obligated_entity_id))
        if obligation.fraction == 0 or (payment is not None and payment.fraction == 0):
            continue
        contract = contracts_by_id[membership.contract_id]
        if (
            annual
            and contract.annual_limit is not None
            and (
                contract.outside_limits_subject_to_aggregate
                or (
                    not membership.outside_limits
                    and not (
                        membership.defense_outside_limits
                        and membership.defense_loading > 0
                    )
                    and not (
                        contract.defense_outside_limits and contract.defense_loading > 0
                    )
                )
            )
        ):
            continue
        if (
            not membership.outside_limits
            and contract.occurrence_limit is not None
            and not (
                membership.defense_outside_limits and membership.defense_loading > 0
            )
            and not (contract.defense_outside_limits and contract.defense_loading > 0)
        ):
            continue
        return False
    return True


def _native_payment_support(payment_kernels):
    obligations, payments = [], []
    for kernel in payment_kernels:
        for identifier in (
            *kernel.liability_component_ids,
            *kernel.direct_component_ids,
        ):
            fraction = float(
                native_payment_positive_mean_lower_bound(kernel, identifier) > 0
            )
            obligations.append(
                ComponentObligation(
                    component_id=identifier,
                    entity_id=kernel.entity_id,
                    fraction=fraction,
                )
            )
            payments.append(
                ComponentPayment(
                    component_id=identifier,
                    entity_id=kernel.entity_id,
                    fraction=fraction,
                )
            )
    return tuple(obligations), tuple(payments)


def _outcome_financial_spec(outcome):
    return (
        outcome.components,
        outcome.obligations,
        outcome.memberships,
        outcome.payments,
        outcome.financial_mapping_complete,
        () if outcome.payment_kernel is None else (outcome.payment_kernel,),
    )


def _cooccurring_financial_specs(kernel):
    active_members = kernel.shared_probability > 0 and kernel.unit_probability > 0
    if kernel.explicit_joint_members:
        for branch in kernel.explicit_joint_members:
            if branch.probability > 0:
                yield (
                    (
                        _campaign_spec(kernel, branch),
                        *(
                            _outcome_financial_spec(outcome)
                            for outcome in branch.members
                        ),
                    )
                    if active_members
                    else (_campaign_spec(kernel, branch),)
                )
        return
    campaign = _campaign_spec(kernel)
    if not active_members or kernel.expected_member_count() == 0:
        yield (campaign,)
        return
    assert kernel.cluster is not None
    multiple = any(
        row.probability > 0 and row.count > 1 for row in kernel.cluster.support
    )
    outcomes = tuple(
        _outcome_financial_spec(outcome)
        for outcome in kernel.possible_member_outcomes()
    )
    if multiple:
        yield (campaign, *outcomes)
    else:
        for outcome in outcomes:
            yield (campaign, outcome)


def _financial_support_admission(compiled, views):
    contracts = {contract.contract_id: contract for contract in compiled.contracts}
    occurrences, invalid_sources = {}, {}
    if not contracts or not any(
        _needs_financial(views, incident.origin_id) for incident in compiled.incidents
    ):
        return {}
    for incident in compiled.incidents:
        if incident.annual_rate == 0:
            continue
        for segment in incident.segments:
            if segment.weight == 0:
                continue
            for specs in _cooccurring_financial_specs(segment.kernel):
                explicit, loaded = set(), set()
                for (
                    laws,
                    _obligations,
                    memberships,
                    payments,
                    complete,
                    kernels,
                ) in specs:
                    if not complete:
                        continue
                    _native_obligations, native_payments = _native_payment_support(
                        kernels
                    )
                    positive_payments = {
                        (row.component_id, row.entity_id): row.fraction
                        for row in (*payments, *native_payments)
                    }
                    components = {law.component_id: law for law in laws}
                    for membership in memberships:
                        if not membership.eligible or membership.covered_fraction == 0:
                            continue
                        law = components[membership.component_id]
                        if (
                            not component_has_positive_support(law)
                            or positive_payments.get(
                                (law.component_id, membership.obligated_entity_id), 0
                            )
                            == 0
                        ):
                            continue
                        contract = contracts[membership.contract_id]
                        _, has_defense, has_loading, conflict = defense_loading_state(
                            law.loss_type, True, membership, contract
                        )
                        if conflict is not None:
                            invalid_sources[incident.origin_id] = conflict
                        if has_defense:
                            explicit.add(contract.contract_id)
                        if has_loading:
                            loaded.add(contract.contract_id)
                        if membership.occurrence_scope == "contract_year":
                            occurrences.setdefault(
                                (contract.contract_id, membership.occurrence_id), set()
                            ).add(incident.origin_id)
                conflict = defense_loading_scope_reason(explicit, loaded)
                if conflict is not None:
                    invalid_sources[incident.origin_id] = conflict
    reasons = {
        view: "A policy occurrence spans multiple source origins; its payment allocation is unidentified."
        for view in views
        if (origin := _risk_unit_view(view)) is not None
        and any(
            origin in sources and len(sources) > 1 for sources in occurrences.values()
        )
    }

    for view in views:
        origin = _risk_unit_view(view)
        invalid = (
            invalid_sources
            if view in {"financial", "oep_financial"}
            else (
                {origin: invalid_sources[origin]} if origin in invalid_sources else {}
            )
        )
        if invalid:
            reasons[view] = "; ".join(
                f"Financial transform for source {key!r} is unavailable: {reason}"
                for key, reason in sorted(invalid.items())
            )
    return reasons


_UNIDENTIFIED_VIEW = (
    "The declared marks and optional transforms do not identify this observable."
)
_NO_PAYMENT_ESTIMAND = "No policy-occurrence payment estimand has been authored."
_GROSS_UNIDENTIFIED = (
    "Complete unique gross is unidentified: composition or overlap is unresolved."
)
_NATURES_UNAVAILABLE = "Complete component natures are unavailable."
_RELATIONS_UNAVAILABLE = "Complete and unambiguous component relations are unavailable."
_RELATION_FIELDS = (
    ("bearer", "bearer_id"),
    ("contributor", "contributor_ids"),
    ("attribution", "attribution_weights"),
    ("scenario", "scenario_ids"),
)


def _empty_year_reason(view: str, *, contracts: bool) -> str | None:
    """Why a year without events leaves ``view`` unidentified, if it does: a
    policy view without contracts, or a view no mark identifies."""
    if view == "financial":
        return None if contracts else _UNIDENTIFIED_VIEW
    if view == "oep_financial" or _risk_unit_view(view) is not None:
        return None if contracts else _NO_PAYMENT_ESTIMAND
    if (
        source_status_view(view) is not None
        or parse_scenario_view(view) is not None
        or view
        in {
            "gross",
            "welfare",
            "net_resource_cost",
            "provider_contribution",
            "oep_gross",
        }
    ):
        return None
    category, separator, _ = view.partition("::")
    if separator and category in {
        "bearer",
        "contributor",
        "attribution",
        "lab_loss",
        "scenario",
    }:
        return None
    return _UNIDENTIFIED_VIEW


@dataclass(frozen=True, slots=True)
class _MarkShape:
    """What availability reads of one mark a segment can emit: its component
    laws with their resolved scenario scope, whether its gross is identified
    by a total and whether its composition is complete and its overlap
    resolved, the labs its episode names, and whether its financial mapping
    is complete (false when no requested view needs it)."""

    components: tuple[tuple[Any, tuple[str, ...]], ...]
    identified: bool
    complete: bool
    lab_ids: frozenset[str]
    mapping_complete: bool

    def evaluation(self, *, financial: bool) -> tuple[set[str], dict[str, str]]:
        """The observables one such mark identifies, and why it leaves the
        others unidentified: a total or a complete, resolved composition
        identifies gross; welfare, net resource cost, lab losses and the
        bearer, contributor, attribution and scenario relations need a complete
        composition with known natures, allocations and relations (one
        scenario each); ``financial`` (with contracts) a complete mapping."""
        laws = [law for law, _ in self.components]
        values: set[str] = set()
        unavailable: dict[str, str] = {}
        if self.identified or self.complete:
            values.add("gross")
        else:
            unavailable["gross"] = _GROSS_UNIDENTIFIED
        for name in ("welfare", "net_resource_cost"):
            if self.complete and all(law.nature is not None for law in laws):
                values.add(name)
            else:
                unavailable[name] = _NATURES_UNAVAILABLE
        allocated = self.complete and all(
            law.lab_loss_weights is not None for law in laws
        )
        if not allocated:
            unavailable["lab_loss"] = (
                "Complete component composition and lab loss allocations are unavailable."
            )
        for lab in self.lab_ids:
            if allocated:
                values.add("lab_loss::" + lab)
            else:
                unavailable["lab_loss::" + lab] = (
                    "Complete incident-linked lab loss is unavailable for a parent episode."
                )
        for view, field in _RELATION_FIELDS:
            relations = [
                scope if view == "scenario" else getattr(law, field)
                for law, scope in self.components
            ]
            if not self.complete or any(
                relation is None or (view == "scenario" and len(relation) != 1)
                for relation in relations
            ):
                unavailable[view] = _RELATIONS_UNAVAILABLE
                continue
            for relation in relations:
                identifiers = (relation,) if view == "bearer" else relation
                values.update(f"{view}::{identifier}" for identifier in identifiers)
        if financial:
            if self.complete and self.mapping_complete:
                values.add("financial")
            else:
                unavailable["financial"] = INCOMPLETE_FINANCIAL_MAPPING
        return values, unavailable


def _value_reason(values, unavailable, view: str) -> str | None:
    """Why an evaluation leaves ``view`` unidentified, if it does: its own
    reason, its category's, none for a relation or lab it does not name (a
    zero), or that nothing identifies it."""
    if view in values:
        return None
    if view in unavailable:
        return unavailable[view]
    category = view.split("::", 1)[0]
    if category in unavailable:
        return unavailable[category]
    if "::" in view and category in {
        "bearer",
        "contributor",
        "attribution",
        "lab_loss",
        "scenario",
    }:
        return None
    return _UNIDENTIFIED_VIEW


def _mark_reasons(
    shape: _MarkShape, views, *, origin: str, status, contracts: bool
) -> dict[str, str]:
    """Why a year holding one mark of ``shape``, from ``origin`` at source
    status ``status``, leaves each of ``views`` unidentified: the mark's own
    reasons (``_MarkShape.evaluation``), carried to the provider, source-status,
    scenario, risk-unit and occurrence-maximum views that read them. An
    occurrence of unknown source status withholds every source-status view; a
    scenario view reads only the scenario's own components, without the mark's
    total; a risk unit needs its own marks' complete mappings."""
    financial = contracts and _needs_financial(views)
    base_values, base_unavailable = shape.evaluation(financial=financial)
    values, unavailable = set(base_values), dict(base_unavailable)
    if "provider_contribution" in views:
        if "contributor" in unavailable:
            unavailable["provider_contribution"] = unavailable["contributor"]
        else:
            values.add("provider_contribution")
    unknown_status = any(
        source_status_view(view) is not None for view in views
    ) and status not in {"public", "not_yet_public"}
    for view in views:
        status_scope = source_status_view(view)
        if status_scope is not None:
            if unknown_status:
                unavailable[view] = (
                    "An incident occurrence has unknown source-model public status."
                )
                continue
            scope, measure = status_scope
            reason = (
                _value_reason(*shape.evaluation(financial=False), measure)
                if scope == status
                else None
            )
            if reason is not None:
                unavailable[view] = reason
            else:
                values.add(view)
        scoped = parse_scenario_view(view)
        if scoped is not None:
            scenario, measure = scoped
            if "scenario" in base_unavailable:
                unavailable[view] = base_unavailable["scenario"]
                continue
            # The scenario's own components, without the mark's identified
            # total: none is a year without events.
            laws = [law for law, scope in shape.components if scope == (scenario,)]
            if not laws:
                reason = None
            elif measure == "gross":
                reason = None if shape.complete else _GROSS_UNIDENTIFIED
            elif shape.complete and all(law.nature is not None for law in laws):
                reason = None
            else:
                reason = _NATURES_UNAVAILABLE
            if reason is not None:
                unavailable[view] = reason
            else:
                values.add(view)
        risk_unit = _risk_unit_view(view)
        if risk_unit is not None:
            if not contracts:
                unavailable[view] = _NO_PAYMENT_ESTIMAND
            elif risk_unit != origin or (shape.complete and shape.mapping_complete):
                values.add(view)
            else:
                unavailable[view] = INCOMPLETE_FINANCIAL_MAPPING
    if "oep_gross" in views:
        if "gross" in base_unavailable:
            unavailable["oep_gross"] = base_unavailable["gross"]
        else:
            values.add("oep_gross")
    if "oep_financial" in views:
        if not contracts:
            unavailable["oep_financial"] = _NO_PAYMENT_ESTIMAND
        elif "financial" in unavailable:
            unavailable["oep_financial"] = unavailable["financial"]
        else:
            values.add("oep_financial")
    return {
        view: reason
        for view in views
        if (reason := _value_reason(values, unavailable, view)) is not None
    }


def view_admission(compiled: CompiledIncidentState, views):
    """Find withheld views and moment existence before simulation.

    Returns ``(reasons, finite_variance, functional)``: the reason each withheld
    view is unavailable, whether each view has a finite second moment, and the
    functionals withheld per view for an infinite first moment. Everything
    follows from the compiled laws' structure, never from a draw: a view is
    withheld when a year without events, or a year with one mark of any law a
    segment can emit (``_mark_reasons``), leaves it unidentified, or when a
    financial transform it needs is invalid (``_financial_support_admission``).
    Moments come from the uncapped GPD tail shapes of positive-support
    component laws.
    """
    support_reasons = _financial_support_admission(compiled, views)
    active_views = tuple(view for view in views if view not in support_reasons)
    if not active_views:
        return support_reasons, dict.fromkeys(views, True), {}
    contracts = bool(compiled.contracts)
    reasons = {
        view: reason
        for view in views
        if (reason := _empty_year_reason(view, contracts=contracts)) is not None
    }
    reasons.update(support_reasons)
    active_views = tuple(view for view in views if view not in reasons)
    finite_variance = dict.fromkeys(views, True)
    functional = {}
    for incident in compiled.incidents:
        if incident.annual_rate == 0:
            continue
        for view, laws in annual_views(incident).items():
            if view not in active_views:
                continue
            shapes = tuple(
                shape
                for annual in laws
                if annual.severity is not None
                and aggregate_lab_loss_has_positive_support(annual)
                for shape in uncapped_infinite_variance_gpd_shapes(
                    native_severity_sampler(annual.severity)
                )
            )
            if shapes:
                finite_variance[view] = False
            if any(shape >= 1 for shape in shapes):
                reason = (
                    "The positive-support annual lab law has an infinite first moment."
                )
                functional[view] = {"expected_annual_loss": reason, "tvar": reason}
        for segment in incident.segments:
            if segment.weight == 0:
                continue
            kernel = segment.kernel
            capped_lab_ids = set(episode_lab_loss_caps(kernel.campaign_lab_losses))
            lab_ids = frozenset(law.lab_id for law in kernel.campaign_lab_losses)
            for aggregate in kernel.campaign_lab_losses:
                if (
                    aggregate.severity is None
                    or aggregate.episode_cap is not None
                    or not aggregate_lab_loss_has_positive_support(aggregate)
                ):
                    continue
                shapes = uncapped_infinite_variance_gpd_shapes(
                    native_severity_sampler(aggregate.severity)
                )
                for view in active_views:
                    scope = source_status_view(view)
                    measure = view if scope is None else scope[1]
                    if measure != f"lab_loss::{aggregate.lab_id}" or (
                        scope is not None and scope[0] != segment.source_public_status
                    ):
                        continue
                    if shapes:
                        finite_variance[view] = False
                    if any(shape >= 1 for shape in shapes):
                        reason = "The positive-support aggregate lab law has an infinite first moment."
                        functional[view] = {
                            "expected_annual_loss": reason,
                            "tvar": reason,
                        }
            branches = [
                branch
                for branch in kernel.explicit_joint_members
                if branch.probability > 0
            ] or [None]
            possible = []
            for branch in branches:
                spec = _campaign_spec(kernel, branch)
                if spec[0] or kernel.campaign_lab_losses:
                    possible.append((spec[0], None, spec))
            if (
                kernel.shared_probability > 0
                and kernel.unit_probability > 0
                and kernel.expected_member_count() > 0
            ):
                for outcome in kernel.possible_member_outcomes():
                    spec = (
                        outcome.components,
                        outcome.obligations,
                        outcome.memberships,
                        outcome.payments,
                        outcome.financial_mapping_complete,
                        ()
                        if outcome.payment_kernel is None
                        else (outcome.payment_kernel,),
                    )
                    possible.append((outcome.components, outcome, spec))
            for laws, outcome, spec in possible:
                scenario_ids = (
                    (incident.scenario_id,)
                    if outcome is None
                    else outcome.scenario_ids or (incident.scenario_id,)
                )
                (
                    _,
                    obligations,
                    memberships,
                    payments,
                    mapping_complete,
                    payment_kernels,
                ) = spec
                financial = _needs_financial(active_views, incident.origin_id)
                if not financial:
                    (
                        obligations,
                        memberships,
                        payments,
                        mapping_complete,
                        payment_kernels,
                    ) = (), (), (), False, ()
                positive_obligations, positive_payments = _native_payment_support(
                    payment_kernels
                )
                for law in laws:
                    if (
                        law.multiplier == 0
                        or law.post_multiplier == 0
                        or law.severity is None
                    ):
                        continue
                    sampler = native_severity_sampler(law.severity)
                    shapes = uncapped_infinite_variance_gpd_shapes(sampler)
                    for view in active_views:
                        if not _affects_view(
                            law,
                            view,
                            scenario_ids,
                            incident.origin_id,
                            segment.source_public_status,
                            capped_lab_ids=capped_lab_ids,
                        ):
                            continue
                        if (
                            (
                                view in {"financial", "oep_financial"}
                                or _risk_unit_view(view) is not None
                            )
                            and mapping_complete
                            and _financial_bounded(
                                law.component_id,
                                obligations + positive_obligations,
                                memberships,
                                payments + positive_payments,
                                compiled.contracts,
                                annual=view == "financial",
                            )
                        ):
                            continue
                        if shapes:
                            finite_variance[view] = False
                        if any(shape >= 1 for shape in shapes):
                            reason = "The positive-support transformed law has an infinite first moment."
                            functional[view] = {
                                "expected_annual_loss": reason,
                                "tvar": reason,
                            }
                shape = _MarkShape(
                    components=tuple(
                        (law, law.scenario_ids or scenario_ids) for law in laws
                    ),
                    identified=outcome is not None
                    and outcome.identified_gross_total is not None,
                    complete=outcome is None
                    or (outcome.composition_complete and outcome.overlap_resolved),
                    lab_ids=lab_ids.union(
                        *(law.lab_loss_weights or () for law in laws)
                    ),
                    mapping_complete=mapping_complete,
                )
                reasons.update(
                    _mark_reasons(
                        shape,
                        active_views,
                        origin=incident.origin_id,
                        status=segment.source_public_status,
                        contracts=contracts,
                    )
                )
    if not compiled.contracts and "financial" in views:
        reasons["financial"] = (
            "No financial contracts or payment estimand have been authored."
        )
    identifiers = compiled.declared_view_ids
    declared_scenarios = identifiers["scenario"]
    for view in views:
        category, separator, identifier = view.partition("::")
        if (
            separator
            and category in {"bearer", "contributor", "attribution", "lab_loss"}
            and identifier not in identifiers[category]
        ):
            reasons[view] = (
                "Requested component relation identifier is outside the declared study."
            )
    for view in views:
        status_scope = source_status_view(view)
        if status_scope is not None:
            _status, measure = status_scope
            if (
                measure.startswith("lab_loss::")
                and measure.removeprefix("lab_loss::") not in identifiers["lab_loss"]
            ):
                reasons[view] = "Requested lab is outside the declared study."
            elif view not in observable_bases(compiled, views):
                reasons[view] = (
                    "The source status has no identified basis for this observable."
                )
        elif view.startswith("source_status::"):
            reasons[view] = "Source-status view has an invalid status or measure."
        if view in compiled.unavailable_views:
            reasons[view] = compiled.unavailable_views[view]
        if view in compiled.source_status_unavailable:
            reasons[view] = compiled.source_status_unavailable[view]
        if (
            view.startswith("lab_loss::")
            and view not in reasons
            and view not in observable_bases(compiled, views)
        ):
            reasons[view] = (
                "Lab economic loss requires one currency, monetary basis and accounting standing across the study's populations."
            )
        scoped = parse_scenario_view(view)
        if scoped is not None and scoped[0] not in declared_scenarios:
            reasons[view] = (
                "Requested scenario is outside the declared study population."
            )
        if (
            scoped is not None
            and (reason := compiled.unavailable_views.get("scenario::" + scoped[0]))
            is not None
        ):
            reasons[view] = reason
        risk_unit = _risk_unit_view(view)
        if risk_unit is not None and risk_unit not in {
            incident.origin_id for incident in compiled.incidents
        }:
            reasons[view] = (
                "Requested source origin is outside this conditional incident law."
            )

    if (pooled_reason := compiled.unavailable_views.get("pooled")) is not None:
        for view in views:
            if not view.startswith("scenario::") and source_status_view(view) is None:
                reasons[view] = pooled_reason
    return reasons, finite_variance, functional


PARAMETER_RANK_PURPOSE = "parameter-rank"
"""The event-address purpose of each year's parameter uniforms, which no other
stream uses: they are addressed by the plan seed itself, not the state's stream
seed, so states declaring one rank stream read the same uniforms."""


def score_multiplier(law: ParameterRankLaw):
    """A ranked law's count multiplier as a function of the year's score.

    The law is read at the stretched score ``rank_spread * z`` and normalized
    by its exact mean over the year score. Callers clip the source uniform
    before converting it to a score; this function does not clip scores.
    """
    mean = epistemic_mean(
        law.count,
        epistemic_tail_scale=law.epistemic_tail_scale,
        spread=law.rank_spread,
        tail_form=law.tail_form,
    )

    def multiplier(scores: np.ndarray) -> np.ndarray:
        if mean == 0:
            return np.zeros(np.shape(scores))
        return (
            stretched_score_quantiles(
                law.count,
                scores,
                spread=law.rank_spread,
                epistemic_tail_scale=law.epistemic_tail_scale,
                tail_form=law.tail_form,
            )
            / mean
        )

    return multiplier


def parameter_scores(
    incidents, seed: int, simulations: int, rank_stream_id: str
) -> dict[str, np.ndarray]:
    """Each ranked law's normal score in every simulated year.

    Year ``y`` reads one clipped uniform per rank group from the plan seed, the
    state's rank stream (``CompiledIncidentState.rank_stream_id``), the group
    and the year alone, then converts it once with ``ndtri``. A state's laws
    sharing a group (one scenario at one source status) read the same score in
    a year and its groups are independent. States read independent scores, so
    the mixture's tail statistics have the precision of all its state-years,
    unless they declare one stream, as a variant state does with the state it
    is measured against. A state's stream does not depend on the arm, so every
    arm of a state reads the same scores, and batches leave them unchanged.
    """
    years = np.arange(simulations, dtype=np.int64)
    ordinals = np.zeros(simulations, dtype=np.int64)
    scores: dict[str, np.ndarray] = {}
    result = {}
    for incident in incidents:
        law = incident.parameter_rank
        if law is None:
            continue
        if law.group_id not in scores:
            uniforms = np.clip(
                event_uniforms(
                    years,
                    ordinals,
                    root_seed=seed,
                    state_id=rank_stream_id,
                    origin_id=law.group_id,
                    purpose=PARAMETER_RANK_PURPOSE,
                ),
                *SIMULATED_RANK_RANGE,
            )
            scores[law.group_id] = ndtri(uniforms)
        result[incident.origin_id] = scores[law.group_id]
    return result


def parameter_rank_multipliers(
    incidents, scores: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Each ranked law's per-year count multiplier at its year scores.

    A law's count is its reference rate times the multiplier
    (``score_multiplier``).
    """
    by_origin = {incident.origin_id: incident for incident in incidents}
    result = {}
    for origin, values in scores.items():
        law = by_origin[origin].parameter_rank
        assert law is not None
        result[origin] = score_multiplier(law)(values)
    return result


def _add_annual_lab_losses(compiled, origin_counts, arrays, *, seed: int):
    """Add each origin's annual lab losses to its lab and gross views, in place
    (``annual_views``).

    Each annual law is drawn once in every year in which the origin's count is
    at least one, from one uniform addressed by the state, origin, lab and
    year alone (``event_uniforms`` at ordinal zero), so the draw is
    independent of the year's count, its episodes' marks and the number of
    years simulated.
    """
    for incident in compiled.incidents:
        if not incident.annual_lab_losses:
            continue
        years = np.flatnonzero(origin_counts[incident.origin_id] > 0)
        views = {
            view: [law.lab_id for law in laws]
            for view, laws in annual_views(incident).items()
            if view in arrays
        }
        if not len(years) or not views:
            continue
        draws = {}
        for law in incident.annual_lab_losses:
            if law.amount is not None:
                draws[law.lab_id] = np.full(len(years), law.amount)
                continue
            assert law.severity is not None
            draws[law.lab_id] = sample_affine_capped_severity(
                native_severity_sampler(law.severity),
                event_uniforms(
                    years,
                    np.zeros(len(years), dtype=np.int64),
                    root_seed=seed,
                    state_id=compiled.state_id,
                    origin_id=incident.origin_id,
                    purpose="annual-lab-mark::" + law.lab_id,
                ),
            )
        for view, lab_ids in views.items():
            for lab_id in lab_ids:
                arrays[view][years] += draws[lab_id]


def _origin_buckets(incident, prepared: SmallEventLaw, segment: int):
    """An eligible origin's score buckets, or why none has a split.

    A law without a parameter rank has one bucket at its rate. A ranked law's
    buckets are cut by rate ratio from its count multiplier, whose kinks lie
    only at the law's knot scores divided by the spread
    (``ranked_buckets``).
    """
    law = incident.parameter_rank
    if law is None:
        return fixed_rate_buckets(prepared, segment, incident.annual_rate)
    breaks = tuple(float(score) / law.rank_spread for score in EPISTEMIC_KNOT_SCORES)
    return ranked_buckets(
        prepared,
        segment,
        incident.annual_rate,
        score_multiplier(law),
        breaks,
    )


def _origin_counts(
    compiled: CompiledIncidentState,
    plan: ExecutionPlan,
    scores: Mapping[str, np.ndarray] | None = None,
    simulations: int | None = None,
    first: int = 0,
):
    """Each origin's count in every simulated year from ``first`` (the plan's
    years by default); earlier years, which an extended simulation already
    holds, are zero.

    A year's count is the least whose CDF under the year's count law reaches
    one uniform (``count_inversion``). Without a count experiment that uniform
    is addressed by the state, origin and year at ordinal zero under the
    ``origin-counts`` purpose, so it depends on neither the count law nor how
    many years are drawn, and only the years from ``first`` are drawn: an arm
    that changes a year's rate or count family reads the same uniform and
    draws a comonotone count. A count experiment inverts its own uniforms, one
    per year in year order from its group and origin streams, so it draws
    every year and zeroes those before ``first``.
    """
    experiment = compiled.count_experiment
    simulations = plan.simulations if simulations is None else simulations
    if scores is None:
        scores = parameter_scores(
            compiled.incidents, plan.seed, simulations, compiled.rank_stream_id
        )
    multipliers = parameter_rank_multipliers(compiled.incidents, scores)
    result = {}

    def rates(incident):
        multiplier = multipliers.get(incident.origin_id)
        return (
            np.full(simulations, incident.annual_rate)
            if multiplier is None
            else incident.annual_rate * multiplier
        )

    def counts(incident, uniforms, rate):
        if incident.count_family == "poisson":
            return poisson_counts(uniforms, rate)
        return negative_binomial_counts(uniforms, rate, incident.count_variance_to_mean)

    if experiment is None:
        years = np.arange(first, simulations, dtype=np.int64)
        for incident in compiled.incidents:
            drawn = np.zeros(simulations, dtype=np.int64)
            if incident.annual_rate > 0:
                drawn[first:] = counts(
                    incident,
                    event_uniforms(
                        years,
                        np.zeros(len(years), dtype=np.int64),
                        root_seed=state_stream_seed(plan.seed, compiled.state_id),
                        state_id=compiled.state_id,
                        origin_id=incident.origin_id,
                        purpose="origin-counts",
                    ),
                    rates(incident)[first:],
                )
            result[incident.origin_id] = drawn
        return result
    shared = {}
    for incident in compiled.incidents:
        group = experiment.origin_groups[incident.origin_id]
        if group not in shared:
            group_rng = keyed_stream_generator(
                root_seed=state_stream_seed(plan.seed, compiled.state_id),
                risk_unit_id=f"count-experiment/group/{group}",
                purpose=experiment.kind,
            )
            shared[group] = group_rng.random(simulations)
        origin_rng = keyed_stream_generator(
            root_seed=state_stream_seed(plan.seed, compiled.state_id),
            risk_unit_id=f"count-experiment/origin/{incident.origin_id}",
            purpose=experiment.kind,
        )
        rate = rates(incident)
        if experiment.kind == "fixed_marginal_gaussian":
            assert experiment.correlation is not None
            rho = experiment.correlation
            independent = origin_rng.random(simulations)
            uniforms = ndtr(
                math.sqrt(rho)
                * ndtri(
                    np.clip(
                        shared[group], np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)
                    )
                )
                + math.sqrt(1 - rho)
                * ndtri(
                    np.clip(independent, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
                )
            )
            uniforms = np.clip(uniforms, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
            result[incident.origin_id] = counts(incident, uniforms, rate)
        else:
            assert experiment.coefficient_of_variation is not None
            variance = experiment.coefficient_of_variation**2
            factors = gamma.ppf(
                np.clip(shared[group], np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)),
                a=1 / variance,
                scale=variance,
            )
            result[incident.origin_id] = counts(
                incident, origin_rng.random(simulations), rate * factors
            )
    for values in result.values():
        values[:first] = 0
    return result


POSITIVE_MEAN_UNDERFLOW_REASON = "The positive target mean is below the numeric output range; a rounded analytic zero is unavailable."


def _event_batches(counts: np.ndarray, batch_size: int):
    """Address complete native events with bounded temporary storage."""
    total = sum(map(int, counts))
    if total > np.iinfo(np.int64).max:
        raise ValueError("realized event count exceeds the supported ordinal range")
    stops = np.cumsum(counts, dtype=np.int64)
    starts = stops - counts
    for start in range(0, total, batch_size):
        positions = np.arange(start, min(start + batch_size, total), dtype=np.int64)
        years = np.searchsorted(stops, positions, side="right")
        yield years, positions - starts[years]


def _member_chunks(members: np.ndarray, batch_size: int):
    """Split a batch's events, in order, into runs holding at most
    ``batch_size`` realized campaign members, or one event that alone holds
    more, so a batch's member marks stay bounded."""
    stops = np.cumsum(members, dtype=np.int64)
    start, count = 0, len(members)
    while start < count:
        before = 0 if start == 0 else int(stops[start - 1])
        stop = int(np.searchsorted(stops, before + batch_size, side="right"))
        stop = max(stop, start + 1)
        yield start, stop
        start = stop


def _segment_choice(incident, years, ordinals, *, seed, state_id):
    cumulative = np.cumsum(
        [segment.weight for segment in incident.segments], dtype=float
    )
    cumulative /= cumulative[-1]
    ranks = event_uniforms(
        years,
        ordinals,
        root_seed=seed,
        state_id=state_id,
        origin_id=incident.origin_id,
        purpose="segment-choice",
    )
    return np.searchsorted(cumulative, ranks, side="right")


type EventLaw = SingleMemberLaw | CampaignLaw


def _event_laws(compiled) -> dict[str, dict[int, EventLaw]]:
    """Each origin's weighted segments' event laws: the single-member law of a
    kernel with at most one member and no campaign terms
    (``single_member_law``), the campaign law of every other
    (``campaign_law``).

    The laws depend only on the kernels, never on the requested views, so an
    origin's draws do not change with the views.
    """
    return {
        incident.origin_id: {
            index: single_member_law(segment.kernel) or campaign_law(segment.kernel)
            for index, segment in enumerate(incident.segments)
            if segment.weight > 0
        }
        for incident in compiled.incidents
    }


def _mark_shares(mark, contracts):
    """A mark law's policy shares, or ``None`` for an incomplete mapping."""
    if not mark.financial_mapping_complete:
        return None
    return policy_shares(
        mark.components,
        mark.obligations,
        mark.payments,
        mark.memberships,
        mark.payment_kernels,
        contracts,
    )


def _policy_laws(compiled, laws, views):
    """The policy shares of each origin whose events enter a requested policy
    view, by segment (``policy_shares``), and for a campaign law by kind of
    mark.

    A mark whose financial mapping is incomplete has no shares (``None``);
    admission withholds the policy views whenever such a mark can be emitted,
    so its events never reach the policy terms.
    """
    contracts = {contract.contract_id: contract for contract in compiled.contracts}
    rates = {
        incident.origin_id: incident.annual_rate for incident in compiled.incidents
    }
    result = {}
    for origin, segments in laws.items():
        if not _needs_financial(views, origin) or rates[origin] == 0:
            continue
        shares = {}
        for index, law in segments.items():
            if isinstance(law, SingleMemberLaw):
                outcome = law.outcome
                shares[index] = (
                    policy_shares(
                        outcome.components,
                        outcome.obligations,
                        outcome.payments,
                        outcome.memberships,
                        ()
                        if outcome.payment_kernel is None
                        else (outcome.payment_kernel,),
                        contracts,
                    )
                    if outcome.financial_mapping_complete
                    else None
                )
            else:
                shares[index] = {
                    kind: _mark_shares(mark, contracts)
                    for kind, mark in enumerate(law.marks)
                }
        result[origin] = shares
    return result


def _policy_reduction(compiled, views, simulations) -> PolicyReduction | None:
    """The reduction of the requested policy views, if any is requested."""
    risk_units = {
        origin: view for view in views if (origin := _risk_unit_view(view)) is not None
    }
    if not risk_units and not {"financial", "oep_financial"} & set(views):
        return None
    return PolicyReduction(
        compiled.contracts,
        simulations,
        financial="financial" in views,
        occurrence_maximum="oep_financial" in views,
        risk_units=risk_units,
    )


@dataclass(frozen=True, slots=True)
class _CampaignProjection:
    """A campaign law's linear view coefficients, by kind of mark, and its
    aggregate lab amounts' views with their episode caps."""

    marks: dict[int, LinearProjection]
    direct_labs: dict[str, tuple[str, float | None]]


def _projection(law: EventLaw, linear, *, scenario_id, source_public_status):
    """A segment's linear projection: a single-member law's
    (``linear_projection``), restricted to the views its events can make
    nonzero, or a campaign law's for every mark (``_CampaignProjection``)."""
    if isinstance(law, SingleMemberLaw):
        projection = linear_projection(
            law,
            linear,
            scenario_id=scenario_id,
            source_public_status=source_public_status,
        )
        return None if projection is None else projection.touched()
    marks = {}
    for kind, mark in enumerate(law.marks):
        projection = mark_projection(
            mark.components,
            mark.scenario_ids,
            (),
            linear,
            scenario_id=scenario_id,
            source_public_status=source_public_status,
        )
        if projection is None:
            return None
        marks[kind] = projection
    labs = mark_projection(
        (),
        (),
        law.labs,
        linear,
        scenario_id=scenario_id,
        source_public_status=source_public_status,
    )
    assert labs is not None
    return _CampaignProjection(marks, labs.direct_labs)


@dataclass(frozen=True, slots=True)
class _EpisodeValues:
    """A batch of one campaign segment's episodes reduced per episode: each
    linear view's value, its unique gross, whether that gross is identified,
    and its covered parts in episode order."""

    values: dict[str, np.ndarray]
    gross: np.ndarray
    identified: np.ndarray
    covered: CoveredParts | None


def _episode_values(
    law: CampaignLaw,
    projection: _CampaignProjection,
    episodes: EpisodeDraws,
    views,
    count: int,
    *,
    policy: PolicyReduction | None,
    shares,
    positions: np.ndarray,
) -> _EpisodeValues:
    """Reduce one segment's episodes (``draw_episodes``) per episode.

    Each episode's marks add in order, its campaign mark before its members
    and members by index: a view's coefficients times their amounts, except
    ``gross`` and ``oep_gross``, which take each mark's unique gross
    (``gross_values``), its identified total where it has one. A lab view
    adds the episode's aggregate lab amount and caps the sum at the episode
    cap. ``positions`` gives each episode's position in the reduced batch, at
    which its covered parts are placed.
    """
    marks = episodes.marks
    if marks:
        event = np.concatenate([draws.event for draws in marks])
        member = np.concatenate([draws.member for draws in marks])
        order = np.lexsort((member, event))
        event = event[order]
    else:
        order = event = np.zeros(0, dtype=np.int64)

    def per_episode(parts):
        total = np.zeros(count)
        if marks:
            np.add.at(total, event, np.concatenate(parts)[order])
        return total

    gross_parts = [gross_values(law.marks[draws.law], draws) for draws in marks]
    gross = per_episode(gross_parts)
    identified = np.ones(count, dtype=bool)
    for draws in marks:
        if not law.marks[draws.law].gross_identified:
            identified[draws.event] = False
    values = {}
    with np.errstate(over="ignore", invalid="ignore"):
        for view in views:
            if view in {"gross", "oep_gross"}:
                values[view] = gross
                continue
            parts = []
            for draws in marks:
                weights = projection.marks[draws.law].weights[view]
                total = np.zeros(len(draws.event))
                for amounts, weight in zip(draws.amounts, weights, strict=True):
                    if weight != 0:
                        total += amounts * weight
                parts.append(total)
            value = per_episode(parts)
            if view in projection.direct_labs:
                lab, cap = projection.direct_labs[view]
                value = value + episodes.labs[lab]
                if cap is not None:
                    value = np.minimum(value, cap)
            values[view] = value
    covered = None
    if policy is not None and shares is not None:
        policy.register(
            share.contract_id
            for kind in sorted(shares)
            if shares[kind] is not None
            for share in shares[kind]
        )
        batches, keys = [], []
        for draws in marks:
            kind_shares = shares[draws.law]
            if kind_shares is None:
                raise ValueError(
                    "realized component admission differs from complete positive-support admission"
                )
            parts = share_parts(
                policy,
                kind_shares,
                draws.amounts,
                draws.payments,
                positions[draws.event],
                draws.member,
            )
            if len(parts.amount):
                width = len(parts.amount) // len(draws.event)
                batches.append(parts)
                keys.append(np.repeat(draws.member, width))
        if batches:
            # Each episode's parts run by mark: its campaign mark, then its
            # members by index.
            joined = CoveredParts.stack(batches)
            covered = joined.select(np.lexsort((np.concatenate(keys), joined.event)))
    return _EpisodeValues(values, gross, identified, covered)


@dataclass
class _CampaignWork:
    """A simulation's campaign member counts over every year, and the limit
    each batch of episodes reserves its realized members against."""

    potential: int
    realized: int
    limit: int
    allowance: int
    state_id: str
    simulations: int

    def reserve(self, count: int) -> None:
        if count > self.limit - self.realized:
            raise WorkLimitExceeded(
                "max_campaign_members_per_year",
                self.allowance,
                self.state_id,
                self.realized + count,
                self.simulations,
            )
        self.realized += count


def _vectorized_incidents(
    compiled,
    origin_counts,
    laws,
    policies,
    views,
    arrays,
    *,
    seed,
    batch_size,
    max_members,
    work: _CampaignWork,
    small=None,
    policy: PolicyReduction | None = None,
):
    """Reduce every origin's events as arrays, one origin stream at a time.

    Each origin reduces only the linear views and source-status scopes its
    events can make nonzero, since the others would add +0.0 to sums that
    start at +0.0. An origin with a small event plan (``_small_event_plans``)
    draws its large cells and any individually drawn small events here and
    adds its summed small events. A campaign segment's episodes
    (``draw_episodes``) are drawn in runs of at most ``batch_size`` realized
    members, each run's members reserved against ``work`` before any of their
    marks is drawn. An origin with policy shares (``policies``) adds each
    batch's covered parts to ``policy``. Returns hidden gross sums by
    source-status scope, whether each year's are complete, and the realized
    members of single-member events.
    """
    linear = tuple(view for view in views if not _financial_view(view))
    scopes = {None} | {
        scope[0] for view in views if (scope := source_status_view(view)) is not None
    }
    simulations = len(next(iter(arrays.values())))
    # Complete gross sums are checked even when no requested view reads them
    # (a projection that gives transfers zero welfare weight, say).
    gross = {scope: np.zeros(simulations) for scope in scopes}
    complete = {scope: np.ones(simulations, dtype=bool) for scope in scopes}
    members = 0
    small = small or {}
    for incident in compiled.incidents:
        segments = laws.get(incident.origin_id)
        if not segments or not np.any(origin_counts[incident.origin_id]):
            continue
        # Each segment reduces only the views its events can make nonzero
        # (``LinearProjection.touched``): a source-status view of another
        # status, another scenario's view or a lab the segment never allocates
        # to would add +0.0 to every year and change no sum. A campaign
        # segment reduces every view.
        projections = {}
        for index, law in segments.items():
            projection = _projection(
                law,
                linear,
                scenario_id=incident.scenario_id,
                source_public_status=incident.segments[index].source_public_status,
            )
            if projection is None:
                raise ValueError(
                    "realized component admission differs from complete positive-support admission"
                )
            projections[index] = projection
        touched = tuple(
            view
            for view in linear
            if any(
                isinstance(projection, _CampaignProjection)
                or view in projection.weights
                for projection in projections.values()
            )
        )
        origin_scopes = tuple(
            scope
            for scope in scopes
            if scope is None
            or any(
                incident.segments[index].source_public_status == scope
                for index in segments
            )
        )
        shares = policies.get(incident.origin_id)

        def reduce_batch(
            years,
            parts,
            incident=incident,
            segments=segments,
            shares=shares,
            projections=projections,
            touched=touched,
            origin_scopes=origin_scopes,
        ):
            """Add one batch of individually drawn events: ``parts`` holds
            (mask, segment index, draws) for each segment's events."""
            nonlocal members
            values = {view: np.zeros(len(years)) for view in touched}
            totals = np.empty(len(years))
            identified = np.ones(len(years), dtype=bool)
            statuses = np.empty(len(years), dtype=object)
            covered = []
            for mask, index, draws in parts:
                statuses[mask] = incident.segments[index].source_public_status
                if isinstance(draws, EpisodeDraws):
                    episode = _episode_values(
                        segments[index],
                        projections[index],
                        draws,
                        touched,
                        int(np.count_nonzero(mask)),
                        policy=policy,
                        shares=None if shares is None else shares[index],
                        positions=np.flatnonzero(mask),
                    )
                    totals[mask] = episode.gross
                    identified[mask] = episode.identified
                    for view, amounts in episode.values.items():
                        values[view][mask] = amounts
                    if episode.covered is not None:
                        covered.append(episode.covered)
                    continue
                members += int(np.count_nonzero(draws.member))
                totals[mask] = draws.component_totals
                for view, amounts in linear_values(projections[index], draws).items():
                    values[view][mask] = amounts
                if shares is not None:
                    covered.append(_covered_parts(policy, shares[index], mask, draws))
            # add.at preserves event order across batch boundaries; ordinary
            # chunk sums would change rounding when batch_size changes.
            with np.errstate(over="ignore", invalid="ignore"):
                for view, amounts in values.items():
                    if view == "oep_gross":
                        np.maximum.at(arrays[view], years, amounts)
                    else:
                        np.add.at(arrays[view], years, amounts)
                for scope in origin_scopes:
                    mask = (
                        np.ones(len(years), dtype=bool)
                        if scope is None
                        else statuses == scope
                    )
                    np.add.at(gross[scope], years[mask], totals[mask])
                    complete[scope][years[mask & ~identified]] = False
            if covered:
                assert policy is not None
                policy.add(
                    years,
                    CoveredParts.concatenate(covered),
                    origin=incident.origin_id,
                )

        stream = {
            "root_seed": seed,
            "state_id": compiled.state_id,
            "origin_id": incident.origin_id,
        }
        plan = small.get(incident.origin_id)
        if plan is None:
            for batch_years, batch_ordinals in _event_batches(
                origin_counts[incident.origin_id], batch_size
            ):
                batch_selected = _segment_choice(
                    incident,
                    batch_years,
                    batch_ordinals,
                    seed=seed,
                    state_id=compiled.state_id,
                )
                # Realized campaign members per event, which bound each run.
                realized = np.zeros(len(batch_years), dtype=np.int64)
                for index, law in segments.items():
                    mask = batch_selected == index
                    if isinstance(law, CampaignLaw) and np.any(mask):
                        counted = episode_members(
                            law, batch_years[mask], batch_ordinals[mask], **stream
                        )
                        if len(counted[1]) and int(counted[1].max()) > max_members:
                            raise ValueError(
                                f"a campaign's {int(counted[1].max()):,} realized "
                                f"members exceed max_members_per_campaign "
                                f"({max_members:,})"
                            )
                        realized[mask] = counted[1]
                        work.potential += int(counted[0].sum())
                for first, stop in _member_chunks(realized, batch_size):
                    years = batch_years[first:stop]
                    ordinals = batch_ordinals[first:stop]
                    selected = batch_selected[first:stop]
                    work.reserve(int(realized[first:stop].sum()))
                    parts = []
                    for index, law in segments.items():
                        mask = selected == index
                        if not np.any(mask):
                            continue
                        if isinstance(law, SingleMemberLaw):
                            draws = draw_single_member_events(
                                law,
                                years[mask],
                                ordinals[mask],
                                payments=shares is not None,
                                **stream,
                            )
                        else:
                            draws = draw_episodes(
                                law,
                                years[mask],
                                ordinals[mask],
                                payments=shares is not None,
                                max_members=max_members,
                                **stream,
                            )
                        parts.append((mask, index, draws))
                    reduce_batch(years, parts)
            continue
        index = plan.buckets.segment
        law = segments[index]
        assert isinstance(law, SingleMemberLaw)
        for cell, counts in plan.drawn:
            for years, ordinals in _event_batches(counts, batch_size):
                draws = draw_single_member_events(
                    law,
                    years,
                    ordinals,
                    payments=shares is not None,
                    cell=cell,
                    **stream,
                )
                reduce_batch(years, [(np.ones(len(years), dtype=bool), index, draws)])
        projection = projections[index]
        assert isinstance(projection, LinearProjection)
        # Summed events change only touched views: an untouched view's
        # coefficients are zero, and any component touches ``oep_gross``
        # (every gross weight is one).
        with np.errstate(over="ignore", invalid="ignore"):
            for view in touched:
                if view == "oep_gross":
                    np.maximum(arrays[view], plan.body_maximum, out=arrays[view])
                    continue
                coefficient, lab = _small_event_coefficients(
                    plan.buckets.gross_per_raw, projection, law, view
                )
                arrays[view] += coefficient * plan.body_gross + lab * plan.body_lab
            for scope in origin_scopes:
                gross[scope] += plan.body_gross
    return gross, complete, members


def _covered_parts(
    policy: PolicyReduction | None,
    shares: tuple[PolicyShare, ...] | None,
    mask: np.ndarray,
    draws,
) -> CoveredParts:
    """The covered parts of one segment's events in a batch (``mask``), each
    member event's parts at its batch position."""
    assert policy is not None
    events = np.flatnonzero(draws.member)
    if shares is None:
        if len(events):
            raise ValueError(
                "realized component admission differs from complete positive-support admission"
            )
        return CoveredParts.empty()
    # A segment's contracts join the annual order when a batch first adds
    # its events, whether or not those events have a member.
    policy.register(share.contract_id for share in shares)
    payment = draws.payment_fraction
    return share_parts(
        policy,
        shares,
        [amounts[events] for amounts in draws.amounts],
        [None if payment is None else payment[events]],
        np.flatnonzero(mask)[events],
    )


def _small_event_coefficients(
    gross_per_raw, projection, law, view
) -> tuple[float, float]:
    """A view's multiples of the small-event gross and aggregate-lab sums.

    Each stochastic component carries ``post_multiplier`` times the shared
    draw, so a view takes ``sum_k w_k * post_k / sum_k post_k`` of the gross;
    the lab's own view adds the aggregate lab loss (its cap cannot bind there).
    """
    weights = projection.weights[view]
    shares = math.fsum(
        weight * component.post_multiplier
        for weight, component in zip(weights, law.outcome.components, strict=True)
        if component.multiplier != 0 and component.post_multiplier != 0
    )
    return shares / gross_per_raw, 1.0 if view in projection.direct_labs else 0.0


@dataclass(frozen=True, slots=True)
class _SmallEventPlan:
    """One origin's realized cells and summed small events.

    ``drawn`` holds each cell's per-year counts of individually drawn events:
    the large and too-few-to-sum small cells of every bucket with a split, and,
    with no cell, every event of a year whose bucket has none.
    """

    buckets: SmallEventBuckets
    drawn: tuple[tuple[EventCell | None, np.ndarray], ...]
    body_gross: np.ndarray
    body_lab: np.ndarray
    body_maximum: np.ndarray


CAMPAIGN_REASON = "Its episodes can have several members or campaign terms."


def _small_event_splits(compiled, addressed, controls):
    """Each event-addressed origin's bucket splits, or why it is drawn event by event.

    Deterministic in the compiled law and the frozen controls, so replay
    re-derives it. Each parameter-rank bucket's split uses the lowest expected
    count a year in it can have (``_origin_buckets``); an origin whose every
    bucket lacks a split reports its highest bucket's reason.
    """
    splits, records = {}, {}
    for incident in compiled.incidents:
        origin = incident.origin_id
        laws = addressed.get(origin)
        if laws is None:
            continue
        weighted = [segment for segment in incident.segments if segment.weight > 0]
        if len(laws) != 1 or len(weighted) != 1:
            records[origin] = {
                "status": "event_by_event",
                "reason": "It has several weighted segments.",
            }
            continue
        ((index, law),) = laws.items()
        if not isinstance(law, SingleMemberLaw):
            records[origin] = {"status": "event_by_event", "reason": CAMPAIGN_REASON}
            continue
        prepared = SmallEventLaw(law, controls)
        if prepared.refusal is not None:
            records[origin] = {"status": "event_by_event", "reason": prepared.refusal}
            continue
        buckets = _origin_buckets(incident, prepared, index)
        if isinstance(buckets, str):
            records[origin] = {"status": "event_by_event", "reason": buckets}
            continue
        splits[origin] = buckets
        records[origin] = {
            "status": "summed_small_events",
            "scenario_id": incident.scenario_id,
            "source_public_status": incident.segments[index].source_public_status,
            "rank_buckets": buckets.record(),
        }
    return splits, records


def small_event_audit(compiled, views, controls, *, cache: dict | None = None):
    """The deterministic part of a state's ``small_events`` record.

    Splits, their moments, each view's small-event variance and the envelope's
    applicability follow from the compiled law and the frozen controls alone;
    simulation adds only the realized event counts. ``cache``, which a caller
    keeps for one compiled state, holds its splits by controls, so a later
    year count reuses them unchanged.
    """
    addressed = _event_laws(compiled)
    if cache is not None and controls in cache:
        splits, records = cache[controls]
    else:
        splits, records = _small_event_splits(compiled, addressed, controls)
        if cache is not None:
            cache[controls] = (splits, records)
    return splits, {
        "controls": controls.record(),
        # Realized counts are added to each origin's row, so rows are copied.
        "by_origin": {origin: dict(row) for origin, row in records.items()},
        "body_variance_by_view": _small_event_variances(
            compiled, splits, addressed, views
        ),
        # A Cauchy-Schwarz bound needs only the body's exact conditional mean
        # and variance, so it holds for Poisson and negative-binomial counts;
        # a count experiment couples origins.
        "envelope_reason": None
        if compiled.count_experiment is None
        else "The TVaR envelope assumes independent origin counts without a count experiment.",
    }


def _small_event_plans(
    compiled,
    origin_counts,
    splits,
    controls,
    *,
    scores,
    seed,
    simulations,
):
    """Split each eligible origin-year's events into cells and sum the small ones.

    Each year takes its bucket's split from its parameter score alone, before
    any event is drawn; a year in a bucket without a split draws every event
    as an origin without one would. Each cell's count is binomial given the
    year's count and the cells before it, inverted from one uniform addressed
    by the origin, the cell and the year (``count_inversion``), so a year's
    cells do not decrease when its count rises; the normal sums and the small
    events' largest gross loss come from three uniforms per year of its
    ``small-event-body`` stream, drawn whatever the requested views. A year
    with no count, such as one an extended simulation already holds, has no
    cells and no sums. Returns plans and each summed origin's realized event
    counts.
    """
    plans, records = {}, {}
    for origin, buckets in splits.items():
        counts = origin_counts[origin]
        year_bucket = buckets.assign(scores.get(origin), simulations)
        found = [split for split in buckets.splits if not isinstance(split, str)]
        # Each bucket's index among the buckets with a split, -1 without one.
        position = (
            np.cumsum([not isinstance(split, str) for split in buckets.splits]) - 1
        )
        position[[isinstance(split, str) for split in buckets.splits]] = -1
        position = position[year_bucket]
        summing = position >= 0
        selected = np.maximum(position, 0)

        def per_year(values, selected=selected):
            return np.asarray(values, dtype=float)[selected]

        def stream(purpose, origin=origin):
            return incident_stream_generator(
                root_seed=seed,
                state_id=compiled.state_id,
                origin_id=origin,
                purpose=purpose,
            )

        def cell(purpose, trials, probabilities, origin=origin):
            """Each year's binomial cell count, inverted from the year's
            uniform under ``purpose``; a year without trials draws none."""
            years = np.flatnonzero(trials > 0)
            counts = np.zeros(simulations, dtype=np.int64)
            counts[years] = binomial_counts(
                event_uniforms(
                    years,
                    np.zeros(len(years), dtype=np.int64),
                    root_seed=seed,
                    state_id=compiled.state_id,
                    origin_id=origin,
                    purpose=purpose,
                ),
                trials[years],
                probabilities[years],
            )
            return counts

        exact = np.where(summing, 0, counts)
        split_counts = counts - exact
        large_gross = cell(
            "small-event-cells::large-gross",
            split_counts,
            per_year([split.large_gross_probability for split in found]),
        )
        rest = split_counts - large_gross
        large_lab = cell(
            "small-event-cells::large-lab",
            rest,
            1.0 - per_year([split.rank_lab for split in found]),
        )
        small = rest - large_lab
        below = small < per_year([split.minimum_count for split in found])
        drawn_small = (
            small if controls.body == "individual" else np.where(below, small, 0)
        )
        summed = (small - drawn_small).astype(float)
        body = stream("small-event-body").random((simulations, 3))
        normals = ndtri(
            np.clip(body[:, :2].T, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
        )
        uniforms = body[:, 2]
        g1 = per_year([split.gross_moments[0] for split in found])
        g2 = per_year([split.gross_moments[1] for split in found])
        l1 = per_year([split.lab_moments[0] for split in found])
        l2 = per_year([split.lab_moments[1] for split in found])
        # A sum of at least ``minimum_count`` events this little skewed lies
        # many standard deviations above zero; the floor never binds in practice.
        body_gross = np.maximum(
            summed * g1 + np.sqrt(summed * np.maximum(g2 - g1 * g1, 0.0)) * normals[0],
            0.0,
        )
        body_lab = np.maximum(
            summed * l1 + np.sqrt(summed * np.maximum(l2 - l1 * l1, 0.0)) * normals[1],
            0.0,
        )
        body_maximum = np.zeros(simulations)
        drawn: list[tuple[EventCell | None, np.ndarray]] = []
        for number, split in enumerate(found):
            years = summing & (position == number) & (split_counts > 0)
            if not years.any():
                continue
            body_maximum[years] = split.body_maximum(
                summed[years].astype(np.int64), uniforms[years]
            )
            named = split.cells()
            for cell, cell_counts in (
                (named["large_gross"], large_gross),
                (named["large_lab"], large_lab),
                (named["small"], drawn_small),
            ):
                drawn.append((cell, np.where(years, cell_counts, 0)))
        if exact.any():
            drawn.append((None, exact))
        plans[origin] = _SmallEventPlan(
            buckets, tuple(drawn), body_gross, body_lab, body_maximum
        )
        records[origin] = {
            "large_events": int(large_gross.sum() + large_lab.sum()),
            "individually_drawn_events": int(
                large_gross.sum() + large_lab.sum() + drawn_small.sum() + exact.sum()
            ),
            "summed_events": int(summed.sum()),
            "summed_years": int(np.count_nonzero(summed)),
            "fallback_years": int(np.count_nonzero((small > 0) & below)),
            "event_by_event_years": int(np.count_nonzero(exact)),
        }
    return plans, records


def _small_event_variances(compiled, splits, addressed, views) -> dict[str, float]:
    """Each linear view's variance of the state's small-event sums (analytic).

    A year's small-event variance is linear in its expected count, so each
    bucket's variance at its share of the mean count, summed over buckets,
    is the variance mixed over the parameter rank.
    """
    linear = tuple(
        view for view in views if not _financial_view(view) and view != "oep_gross"
    )
    variances = dict.fromkeys(linear, 0.0)
    for incident in compiled.incidents:
        buckets = splits.get(incident.origin_id)
        if buckets is None:
            continue
        index = buckets.segment
        law = addressed[incident.origin_id][index]
        projection = linear_projection(
            law,
            linear,
            scenario_id=incident.scenario_id,
            source_public_status=incident.segments[index].source_public_status,
        )
        assert projection is not None
        dispersion = (
            1.0
            if incident.count_family == "poisson"
            else incident.count_variance_to_mean
        )
        for view in linear:
            coefficient, lab = _small_event_coefficients(
                buckets.gross_per_raw, projection, law, view
            )
            if not (coefficient or lab):
                continue
            for split, share in zip(buckets.splits, buckets.mean_shares, strict=True):
                if not isinstance(split, str):
                    variances[view] += body_variance(
                        split,
                        coefficient,
                        lab,
                        incident.annual_rate * share,
                        dispersion,
                    )
    return variances


@dataclass(frozen=True)
class SimulationAudit:
    """What a state's simulation records beside its draws.

    Its withheld functionals, view bases, work, small-event and stream
    audits. A run capture keeps the withheld functionals and view bases in the
    state's support, the stream seed as its seed and every other field in its
    ``execution`` record (``execution_record``).
    """

    functional_unavailable_by_view: dict[str, dict[str, str]]
    view_bases: dict[str, dict[str, str]]
    compiled_frequency_audit: tuple[dict[str, object], ...]
    compiled_law_sha256: str
    compiled_evidence_sha256: str
    count_experiment: dict[str, Any] | None
    initiating_counts: dict[str, int]
    total_initiators: int
    sampled_potential_members: int
    realized_member_work: int
    vectorized_initiators: int
    summed_small_initiators: int
    campaign_members: int
    work_limits: dict[str, int]
    small_events: dict[str, Any]
    resolved_numerical_controls: dict[str, Any]
    available_views: list[str]
    stream_seed: int
    stream_design: dict[str, object]
    joint_construction: str

    def execution_record(self) -> dict[str, object]:
        """The capture's ``execution`` record: every field by name, except
        those the capture keeps in the support and as the seed."""
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in _KEPT_OUTSIDE_EXECUTION
        }


_KEPT_OUTSIDE_EXECUTION = frozenset(
    {"functional_unavailable_by_view", "view_bases", "stream_seed"}
)


def stream_design(compiled: CompiledIncidentState, seed: int) -> dict[str, object]:
    """How a state's draws are addressed under plan seed ``seed``: the record
    every simulation of the state captures and replay re-derives, so a capture
    drawn under another addressing does not replay."""
    return {
        "contract": INCIDENT_STREAM_CONTRACT,
        "event_uniform_contract": VECTOR_EVENT_STREAM_CONTRACT,
        "event_uniform_contract_by_origin": {
            incident.origin_id: VECTOR_EVENT_STREAM_CONTRACT
            for incident in compiled.incidents
        },
        "root_seed": state_stream_seed(seed, compiled.state_id),
        "state_id": compiled.state_id,
        "rank_stream_id": compiled.rank_stream_id,
        "parameter_rank_scope": "Per rank group: each year's parameter uniform addressed by the plan seed, the state's rank stream, the group and the year at ordinal zero under the parameter-rank purpose, clipped and converted once to a normal score; states read independent ranks unless they declare one rank stream, and every arm of a state reads the same.",
        "count_scope": "Per origin: each year's count inverted through its count law from one uniform addressed by the year at ordinal zero under the origin-counts purpose; explicit count experiments invert one uniform per year from their group and origin streams.",
        "mark_scope": "Named origin and physical campaign; payment randomness is separate from physical marks.",
        "member_scope": "Per campaign episode: a member's uniforms are addressed by its event's year and ordinal times 2**32 plus its index in the event, under member:: purposes; campaign-scoped ranks are the event's.",
        "annual_lab_scope": "Per origin and lab: one uniform in each year whose count is at least one, addressed by the year at ordinal zero under the annual-lab-mark::<lab> purpose.",
        "small_event_scope": "Per origin: each year's rank bucket from its parameter rank alone; binomial cell counts inverted from one uniform per year under each of its two small-event-cells purposes, addressed by the year at ordinal zero; large-cell and individually drawn small events event-addressed under small-events::<cell>:: purposes, and every event of a year whose bucket has no split under the origin's own purposes; summed small events and their largest loss from three uniforms per year of its small-event-body stream.",
    }


class SimulatedState(NamedTuple):
    """A state's simulation, as ``simulate_incident_state`` returns it."""

    draws: ConditionalCapabilityDraws
    unavailable: dict[str, str]
    """The reason each withheld requested view is unavailable."""
    audit: SimulationAudit


_NOT_CONTINUED = (
    "an extended simulation must continue the same compiled state, plan and streams"
)
# Audit fields that identify what a simulation draws; an extension must
# reproduce them (``simulate_incident_state``). The seed, state, response,
# law and controls are checked before any year is simulated.
_EXTENSION_IDENTITY = (
    "compiled_law_sha256",
    "compiled_evidence_sha256",
    "count_experiment",
    "resolved_numerical_controls",
    "available_views",
    "view_bases",
    "stream_design",
)


def simulate_incident_state(
    plan: ExecutionPlan,
    compiled: CompiledIncidentState,
    *,
    response_sha256: str,
    years: int | None = None,
    previous: SimulatedState | None = None,
    split_cache: dict | None = None,
) -> SimulatedState:
    """Simulate aligned annual losses for one compiled state.

    Origin counts are drawn once, and every origin's events from their own
    event addresses. Returns the conditional draws, the reasons for withheld
    requested views and the simulation's audit. ``years`` defaults to the
    plan's simulations.

    A year's draws depend only on the plan seed, the state and the year, never
    on how many years are simulated: per-year streams are drawn in year order
    or keyed by year, every event's draws are keyed by its year and ordinal,
    and each year's losses are summed in the same order. ``previous``, this
    function's result for the same compiled state and plan at fewer years, is
    therefore extended rather than redrawn: its years are kept and only the
    later ones are simulated, which gives the same draws and audit as
    simulating every year at once. Year-addressed counts are drawn only for
    the later years, and the count and small-event audits add the previous
    result's; the admission and analytic means are the previous result's.
    Each work count over every year is checked against its limit, the plan's
    allowance per year times the years, before any new event is drawn
    (campaign members as each batch of episodes reserves them), so every tail
    gate round, extensions included, refuses a law whose mean work per year
    exceeds an allowance. ``split_cache``, kept by the caller for this
    compiled state, reuses its small-event splits across year counts
    (``small_event_audit``).

    It runs in three steps: ``_plan_simulation`` fixes everything that needs
    no draw, ``_draw_years`` draws the new years and ``_assemble_simulation``
    builds the result.
    """
    run = _plan_simulation(
        plan,
        compiled,
        response_sha256=response_sha256,
        years=years,
        previous=previous,
        split_cache=split_cache,
    )
    drawn = _draw_years(plan, compiled, run, previous)
    return _assemble_simulation(
        plan, compiled, run, drawn, response_sha256=response_sha256, previous=previous
    )


@dataclass(frozen=True)
class _SimulationPlan:
    """What a simulation fixes before it draws a year (``_plan_simulation``).

    ``finite_variance`` and ``functional`` are the admission's, None when the
    simulation extends a previous one, whose results it keeps. The draw step
    adds the realized small-event counts to ``small_audit``'s rows, which
    ``small_event_audit`` copies for that.
    """

    simulations: int
    first: int
    state_seed: int
    reasons: dict[str, str]
    finite_variance: dict[str, bool] | None
    functional: dict[str, dict[str, str]] | None
    available: tuple[str, ...]
    allowances: dict[str, int]
    limits: dict[str, int]
    max_members: int
    batch_size: int
    small_controls: SmallEventControls
    resolved_controls: dict[str, Any]
    laws: dict[str, dict[int, EventLaw]]
    policies: dict[str, dict[int, Any]]
    small_splits: dict[str, SmallEventBuckets]
    small_audit: dict[str, Any]


def _plan_simulation(
    plan: ExecutionPlan,
    compiled: CompiledIncidentState,
    *,
    response_sha256: str,
    years: int | None,
    previous: SimulatedState | None,
    split_cache: dict | None,
) -> _SimulationPlan:
    """The deterministic step of ``simulate_incident_state``: its years, the
    admitted views, the resolved controls and work limits, each origin's
    event laws, their policy shares and the small-event splits, none of which
    needs a draw.

    Refuses an unsupported control, years that do not extend ``previous``, a
    law that identifies no requested view and a ``previous`` from another
    state, plan or response.
    """
    controls = plan.numerical_controls
    # Refuses an unsupported control or a count control that is not a
    # positive integer.
    allowances = work_allowances(controls)
    max_members = controls.get("max_members_per_campaign", 2_000_000)
    batch_size = controls.get("event_batch_size", 65_536)
    assert isinstance(max_members, int) and isinstance(batch_size, int)
    small_controls = SmallEventControls.from_controls(controls.get("small_events"))
    simulations = plan.simulations if years is None else years
    first = 0 if previous is None else previous.draws.n_simulations
    if (
        isinstance(simulations, bool)
        or not isinstance(simulations, int)
        or simulations <= first
    ):
        raise ValueError(
            "simulated years must be a positive integer above the extended simulation's years"
        )
    finite_variance = functional = None
    if previous is None:
        reasons, finite_variance, functional = view_admission(
            compiled, plan.requested_views
        )
    else:
        reasons = previous.unavailable
    available = tuple(view for view in plan.requested_views if view not in reasons)
    if not available:
        raise ValueError(
            "no requested observable is identified by the authored conditional law: "
            + "; ".join(f"{view}: {reasons[view]}" for view in plan.requested_views)
        )
    resolved_controls = {
        **allowances,
        "max_members_per_campaign": max_members,
        "event_batch_size": batch_size,
        "small_events": small_controls.record(),
    }
    state_seed = state_stream_seed(plan.seed, compiled.state_id)
    if previous is not None and (
        (
            previous.draws.stratum_sha256,
            previous.draws.response_sha256,
            previous.draws.seed,
        )
        != (state_key(compiled.state_id), response_sha256, state_seed)
        or (
            previous.audit.compiled_law_sha256,
            previous.audit.compiled_evidence_sha256,
            previous.audit.resolved_numerical_controls,
        )
        != (compiled.law_sha256, compiled.evidence_sha256, resolved_controls)
    ):
        raise ValueError(_NOT_CONTINUED)
    laws = _event_laws(compiled)
    small_splits, small_audit = small_event_audit(
        compiled, available, small_controls, cache=split_cache
    )
    return _SimulationPlan(
        simulations=simulations,
        first=first,
        state_seed=state_seed,
        reasons=reasons,
        finite_variance=finite_variance,
        functional=functional,
        available=available,
        allowances=allowances,
        limits=work_limits(allowances, simulations),
        max_members=max_members,
        batch_size=batch_size,
        small_controls=small_controls,
        resolved_controls=resolved_controls,
        laws=laws,
        policies=_policy_laws(compiled, laws, available),
        small_splits=small_splits,
        small_audit=small_audit,
    )


@dataclass(frozen=True)
class _DrawnYears:
    """A simulation's drawn years and their audits (``_draw_years``).

    ``arrays`` holds each available view's annual losses, the years an
    extended simulation keeps still zero; ``gross`` the hidden complete gross
    sums by source-status scope, and ``complete`` whether each year's is
    complete. Counts cover every year.
    """

    arrays: dict[str, np.ndarray]
    gross: dict[str | None, np.ndarray]
    complete: dict[str | None, np.ndarray]
    initiating_counts: dict[str, int]
    total_events: int
    vectorized_events: int
    summed_events: int
    small_plans: dict[str, _SmallEventPlan]
    sampled_members: int
    realized_members: int
    campaign_members: int


def _draw_years(
    plan: ExecutionPlan,
    compiled: CompiledIncidentState,
    run: _SimulationPlan,
    previous: SimulatedState | None,
) -> _DrawnYears:
    """The draw step of ``simulate_incident_state``: every origin's events in
    every year from ``run.first`` as arrays, then the annual lab losses and the
    policy views, whose terms one reduction applies to every origin's covered
    parts.

    Counts and events are drawn only for the years ``previous`` lacks, but
    every count audit and work limit covers all years: the audits add the
    previous result's, which are those of the same year-addressed counts.
    Refuses an event count over its limit before any event is drawn, and
    campaign members as each batch of episodes reserves them.
    """
    simulations, first = run.simulations, run.first
    scores = parameter_scores(
        compiled.incidents, plan.seed, simulations, compiled.rank_stream_id
    )
    arrays = {view: np.zeros(simulations) for view in run.available}
    origin_counts = _origin_counts(compiled, plan, scores, simulations, first)
    held = None if previous is None else previous.audit
    held_counts = {} if held is None else held.initiating_counts
    counts_audit = {
        incident.origin_id: held_counts.get(incident.origin_id, 0)
        + sum(int(count) for count in origin_counts[incident.origin_id])
        for incident in compiled.incidents
    }
    total_events = sum(counts_audit.values())
    small_plans, realized = _small_event_plans(
        compiled,
        origin_counts,
        run.small_splits,
        run.small_controls,
        scores=scores,
        seed=run.state_seed,
        simulations=simulations,
    )
    held_small = {} if held is None else held.small_events["by_origin"]
    small_rows = run.small_audit["by_origin"]
    for origin, counts in realized.items():
        earlier = held_small.get(origin, {})
        small_rows[origin].update(
            {key: value + earlier.get(key, 0) for key, value in counts.items()}
        )
    # The vectorized limit counts individually drawn events in every year;
    # summed small events cost one draw per origin-year.
    vectorized_events = sum(
        small_rows[incident.origin_id]["individually_drawn_events"]
        if incident.origin_id in small_plans
        else counts_audit[incident.origin_id]
        for incident in compiled.incidents
    )
    summed_events = sum(int(small_rows[origin]["summed_events"]) for origin in realized)
    allowance = "max_vectorized_events_per_year"
    if vectorized_events > run.limits[WORK_ALLOWANCES[allowance]]:
        raise WorkLimitExceeded(
            allowance,
            run.allowances[allowance],
            compiled.state_id,
            vectorized_events,
            simulations,
        )
    work = _CampaignWork(
        potential=0 if held is None else held.sampled_potential_members,
        realized=0 if held is None else held.campaign_members,
        limit=run.limits["campaign_members"],
        allowance=run.allowances["max_campaign_members_per_year"],
        state_id=compiled.state_id,
        simulations=simulations,
    )
    held_campaign = work.realized
    policy = _policy_reduction(compiled, run.available, simulations)
    gross, complete, single_members = _vectorized_incidents(
        compiled,
        origin_counts,
        run.laws,
        run.policies,
        run.available,
        arrays,
        seed=run.state_seed,
        batch_size=run.batch_size,
        max_members=run.max_members,
        work=work,
        small=small_plans,
        policy=policy,
    )
    _add_annual_lab_losses(compiled, origin_counts, arrays, seed=run.state_seed)
    if policy is not None:
        # Only the policy reduction adds to policy views, which start at zero.
        for view, values in policy.views().items():
            arrays[view] += values
    held_realized = 0 if held is None else held.realized_member_work
    return _DrawnYears(
        arrays=arrays,
        gross=gross,
        complete=complete,
        initiating_counts=counts_audit,
        total_events=total_events,
        vectorized_events=vectorized_events,
        summed_events=summed_events,
        small_plans=small_plans,
        sampled_members=work.potential + single_members,
        realized_members=held_realized
        + single_members
        + (work.realized - held_campaign),
        campaign_members=work.realized,
    )


def _assemble_simulation(
    plan: ExecutionPlan,
    compiled: CompiledIncidentState,
    run: _SimulationPlan,
    drawn: _DrawnYears,
    *,
    response_sha256: str,
    previous: SimulatedState | None,
) -> SimulatedState:
    """The assemble step of ``simulate_incident_state``: each view's draws,
    with ``previous``'s years and moment claims or the law's native means,
    the risk units' evaluator provenance and the audit.

    Refuses losses beyond the numeric output range and an extension that does
    not reproduce ``previous``'s identity (``_EXTENSION_IDENTITY``).
    """
    arrays, gross, complete = drawn.arrays, drawn.gross, drawn.complete
    if any(
        np.any(~np.isfinite(values[complete[scope]])) for scope, values in gross.items()
    ):
        raise OverflowError("complete annual gross exceeds the numeric output range")
    if any(np.any(~np.isfinite(values)) for values in arrays.values()):
        raise OverflowError("annual loss exceeds the numeric output range")
    if previous is not None:
        # The earlier years, which the arrays left at zero, are the previous
        # simulation's; their moment claims are a law's and do not change.
        for view, values in arrays.items():
            values[: run.first] = previous.draws.views[view].array
        views = {
            view: replace(row, array=arrays[view])
            for view, row in previous.draws.views.items()
        }
        functional = previous.audit.functional_unavailable_by_view
    else:
        assert run.finite_variance is not None and run.functional is not None
        functional = run.functional
        views = _native_views(
            compiled, run.available, arrays, run.finite_variance, functional
        )
    draw = ConditionalCapabilityDraws(
        views=views,
        joint_law_semantics="physical_joint",
        evaluator_provenance_by_risk_unit={
            incident.origin_id: {
                "method": "vectorized_native_incident_summed_small_events"
                if incident.origin_id in drawn.small_plans
                else "vectorized_native_incident",
                "joint_law_semantics": "physical_joint",
                "count_family": incident.count_family,
                "compiled_conditional_key": compiled.conditional_key,
                "oep_gross_scope": "Maximum unique gross per initiating physical origin; empty-year maximum is zero.",
                "oep_financial_scope": "Largest single policy-occurrence payment in the year, after occurrence retentions and limits and before annual retentions and limits, summed across contracts sharing occurrence ID; empty-year maximum is zero. With annual terms it can exceed the year's financial payment.",
            }
            for incident in compiled.incidents
        }
        or {
            "exact_zero_origin_support": {
                "method": "degenerate_zero_portfolio",
                "joint_law_semantics": "physical_joint",
                "compiled_conditional_key": compiled.conditional_key,
            }
        },
        stratum_sha256=state_key(compiled.state_id),
        response_sha256=response_sha256,
        seed=run.state_seed,
    )
    audit = SimulationAudit(
        functional_unavailable_by_view=functional,
        view_bases=observable_bases(compiled, plan.requested_views),
        compiled_frequency_audit=compiled.frequency_audit,
        compiled_law_sha256=compiled.law_sha256,
        compiled_evidence_sha256=compiled.evidence_sha256,
        count_experiment=None
        if compiled.count_experiment is None
        else compiled.count_experiment.model_dump(mode="json"),
        initiating_counts=drawn.initiating_counts,
        total_initiators=drawn.total_events,
        sampled_potential_members=drawn.sampled_members,
        realized_member_work=drawn.realized_members,
        vectorized_initiators=drawn.vectorized_events,
        summed_small_initiators=drawn.summed_events,
        campaign_members=drawn.campaign_members,
        work_limits=run.limits,
        small_events=run.small_audit,
        resolved_numerical_controls=run.resolved_controls,
        available_views=list(run.available),
        stream_seed=draw.seed,
        stream_design=stream_design(compiled, plan.seed),
        joint_construction="One count per origin; conditional segment and complete component marks; annual policy terms after each year's occurrences.",
    )
    if previous is not None and (
        (
            previous.draws.stratum_sha256,
            previous.draws.response_sha256,
            previous.draws.seed,
        )
        != (draw.stratum_sha256, draw.response_sha256, draw.seed)
        or previous.draws.evaluator_provenance_by_risk_unit
        != draw.evaluator_provenance_by_risk_unit
        or any(
            getattr(previous.audit, name) != getattr(audit, name)
            for name in _EXTENSION_IDENTITY
        )
    ):
        raise ValueError(_NOT_CONTINUED)
    return SimulatedState(draw, run.reasons, audit)


def _native_views(
    compiled, available, arrays, finite_variance, functional
) -> dict[str, ConditionalView]:
    """Each view's draws with the law's moment claims and native analytic mean.

    A positive mean that rounds to an analytic zero has no native mean: its
    EAL is withheld (``POSITIVE_MEAN_UNDERFLOW_REASON``, added to
    ``functional`` in place) along with its precision.
    """
    precision = {view: finite_variance[view] for view in available}
    mean_exists = {view: view not in functional for view in available}
    known_zero = analytic_known_zero_views(compiled, available)
    analytic = analytic_linear_eals(compiled, available)
    for view, value in analytic.values.items():
        if (
            value == 0
            and view not in known_zero
            and view in available
            and view not in functional
        ):
            functional.setdefault(view, {})["expected_annual_loss"] = (
                POSITIVE_MEAN_UNDERFLOW_REASON
            )
            precision[view] = False
    analytic_means = {
        view: value
        for view, value in analytic.values.items()
        if view in available and view not in functional
    }
    return {
        view: ConditionalView(
            array=values,
            mean_exists=mean_exists[view],
            variance_exists=precision[view],
            native_mean=NativeMean(
                value=analytic_means[view],
                estimator="native_analytic_mean",
                precision={
                    "numerical_standard_error": 0.0,
                    "standard_error_upper_bound": 0.0,
                    "precision_basis": DETERMINISTIC_EAL_PRECISION_BASIS,
                    "integration_error_bound": analytic.error_bounds[view],
                    "known_zero_law": view in known_zero,
                },
            )
            if view in analytic_means
            else None,
        )
        for view, values in arrays.items()
    }
