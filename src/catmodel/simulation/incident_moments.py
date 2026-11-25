"""Analytic incident means and support-based zero proofs.

Linear views integrate in closed form. A lab view whose episode cap can bind
is integrated numerically with a certified error bound.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import cast

import numpy as np
from scipy.special import ndtr, ndtri

from catmodel.contracts.components import AnnualLabLossLaw, ComponentAmountLaw
from catmodel.contracts.incident import (
    CountExperiment,
    IncidentLaw,
    JointMarkOutcome,
    parse_scenario_view,
)
from catmodel.contracts.native_severity import (
    NativeSeverityLaw,
    native_severity_sampler,
)
from catmodel.elicitation.epistemic import (
    EPISTEMIC_KNOT_SCORES,
    epistemic_mean,
    stretched_score_quantiles,
)
from catmodel.probability.distributions import (
    Distribution,
    SeveritySampler,
    severity_lower_support,
    uncapped_infinite_variance_gpd_shapes,
)
from catmodel.probability.numerics import product_at_least, scaled_product
from catmodel.simulation.components import (
    aggregate_lab_loss_has_positive_support,
    component_has_positive_support,
    component_view_weight,
)
from catmodel.simulation.incident import (
    CompiledIncident,
    CompiledIncidentState,
    CompiledMarkSegment,
    source_status_view,
)
from catmodel.simulation.small_events import SIMULATED_RANK_RANGE


def _coefficient(
    view, components, coefficients, scenario_id, outcome, source_public_status=None
):
    scoped = source_status_view(view)
    if scoped is not None:
        status, view = scoped
        if status != source_public_status:
            return 0.0
    scenario_scope = parse_scenario_view(view)
    if scenario_scope is not None:
        scope, measure = scenario_scope
        return sum(
            coefficient
            * _coefficient(measure, (component,), (1,), scenario_id, outcome)
            for component, coefficient in zip(components, coefficients, strict=True)
            if (
                component.scenario_ids
                or (() if outcome is None else outcome.scenario_ids)
                or (scenario_id,)
            )
            == (scope,)
        )
    return sum(
        coefficient * component_view_weight(component, view)
        for component, coefficient in zip(components, coefficients, strict=True)
    )


def _aggregate_lab_views(kernel, source_public_status):
    """Project a whole initiating episode into its original source-status bucket.

    The aggregate amount adds to the episode's component lab allocations.
    """
    result = {f"lab_loss::{law.lab_id}": law for law in kernel.campaign_lab_losses}
    if source_public_status in {"public", "not_yet_public"}:
        result.update(
            {
                f"source_status::{source_public_status}::lab_loss::{law.lab_id}": law
                for law in kernel.campaign_lab_losses
            }
        )
    return result


def annual_views(
    incident: CompiledIncident,
) -> dict[str, tuple[AnnualLabLossLaw, ...]]:
    """Project an origin's annual lab laws onto the views they enter.

    Each lab's annual loss enters its own lab view and every gross view of
    the origin's scenario (``gross`` and ``scenario::<id>::gross``): the lab
    is a party, and gross counts every party's loss. It enters no welfare or
    net-resource view, nor ``oep_gross``, a maximum over occurrences. A law
    with annual lab losses has one source status across its segments
    (``IncidentLaw``), so the year's draw enters that status's lab bucket.
    """
    laws = incident.annual_lab_losses
    if not laws:
        return {}
    result: dict[str, tuple[AnnualLabLossLaw, ...]] = {
        f"lab_loss::{law.lab_id}": (law,) for law in laws
    }
    scenario = incident.scenario_id
    for view in ("gross", f"scenario::{scenario}::gross"):
        result[view] = laws
    statuses = {segment.source_public_status for segment in incident.segments}
    if len(statuses) == 1 and (status := next(iter(statuses))) in {
        "public",
        "not_yet_public",
    }:
        result.update(
            {f"source_status::{status}::lab_loss::{law.lab_id}": (law,) for law in laws}
        )
    return result


def at_least_one_probability(
    incident: CompiledIncident, count_experiment: CountExperiment | None = None
) -> tuple[float, float] | None:
    """``E_Z[P(N >= 1 | lambda(Z))]`` with its certified error bound.

    A Poisson count has ``P(N = 0 | lambda) = exp(-lambda)``; a negative
    binomial count of fixed variance-to-mean ``v`` has ``v**(-lambda/(v-1))``,
    so both are ``exp(-c lambda)`` with ``c`` one or ``ln(v)/(v-1)``. Without a
    parameter rank ``lambda`` is the annual rate. With one, ``lambda`` is the
    rate times the count multiplier at the year's score, as simulation draws
    it (``score_multiplier``), and the probability is integrated over the
    rank's uniform, clipped to ``SIMULATED_RANK_RANGE`` as simulation clips
    it. The integrand rises with the rank, so ``_monotone_integral``
    certifies it within ``CERTIFIED_INTEGRATION_RELATIVE_ERROR``; it is split
    at the count law's knot scores divided by the spread, where it can kink. A
    ``fixed_marginal_gaussian`` count experiment keeps each count's marginal
    law; a ``shared_gamma_intensity`` one changes it, and then, or when the
    integral cannot be certified, None leaves the mean to simulation.
    """
    if count_experiment is not None and count_experiment.kind != (
        "fixed_marginal_gaussian"
    ):
        return None
    rate = incident.annual_rate
    if rate == 0:
        return 0.0, 0.0
    v = incident.count_variance_to_mean
    decay = 1.0 if incident.count_family == "poisson" else math.log(v) / (v - 1)
    rank = incident.parameter_rank
    if rank is None:
        return -math.expm1(-decay * rate), 0.0
    mean = epistemic_mean(
        rank.count,
        epistemic_tail_scale=rank.epistemic_tail_scale,
        spread=rank.rank_spread,
        tail_form=rank.tail_form,
    )
    if mean == 0:
        return 0.0, 0.0
    low, high = SIMULATED_RANK_RANGE

    def descending(reflected: np.ndarray) -> np.ndarray:
        # The integrand at rank 1 - t, non-increasing in t.
        ranks = np.clip(1.0 - np.asarray(reflected, dtype=float), low, high)
        counts = stretched_score_quantiles(
            rank.count,
            ndtri(ranks),
            spread=rank.rank_spread,
            epistemic_tail_scale=rank.epistemic_tail_scale,
            tail_form=rank.tail_form,
        )
        with np.errstate(over="ignore", under="ignore"):
            return -np.expm1(-decay * rate * (counts / mean))

    knots = 1.0 - ndtr(np.asarray(EPISTEMIC_KNOT_SCORES) / rank.rank_spread)
    edges = np.unique(np.concatenate(([0.0, 1.0], knots[(knots > 0) & (knots < 1)])))
    return _monotone_integral(descending, edges, 0.0)


def analytic_known_zero_views(
    compiled: CompiledIncidentState, views: Sequence[str]
) -> frozenset[str]:
    """Prove zero linear loss from support, without multiplying small factors."""
    linear = {
        view
        for view in views
        if view not in {"financial", "oep_gross", "oep_financial"}
        and not view.startswith("risk_unit::")
    }
    positive = set()

    def inspect(components, scenario_id, source_public_status, outcome=None):
        if (
            outcome is not None
            and outcome.identified_gross_total is not None
            and outcome.identified_gross_total > 0
        ):
            positive.add("gross")
            if not outcome.composition_complete or not outcome.overlap_resolved:
                positive.update(
                    view
                    for view in linear - {"gross"}
                    if (scope := source_status_view(view)) is None
                    or scope[0] == source_public_status
                )
        for component in components:
            if component_has_positive_support(component):
                positive.update(
                    view
                    for view in linear
                    if _coefficient(
                        view,
                        (component,),
                        (1,),
                        scenario_id,
                        outcome,
                        source_public_status,
                    )
                    > 0
                )

    for incident in compiled.incidents:
        if incident.annual_rate == 0:
            continue
        positive.update(
            view
            for view, laws in annual_views(incident).items()
            if any(aggregate_lab_loss_has_positive_support(law) for law in laws)
        )
        for segment in incident.segments:
            if segment.weight == 0:
                continue
            kernel = segment.kernel
            positive.update(
                view
                for view, law in _aggregate_lab_views(
                    kernel, segment.source_public_status
                ).items()
                if aggregate_lab_loss_has_positive_support(law)
            )
            inspect(
                kernel.campaign_components,
                incident.scenario_id,
                segment.source_public_status,
            )
            members_possible = (
                kernel.shared_probability > 0 and kernel.unit_probability > 0
            )
            if kernel.explicit_joint_members:
                for branch in kernel.explicit_joint_members:
                    if branch.probability == 0:
                        continue
                    inspect(
                        branch.campaign_components,
                        incident.scenario_id,
                        segment.source_public_status,
                    )
                    if members_possible:
                        for outcome in branch.members:
                            inspect(
                                outcome.components,
                                incident.scenario_id,
                                segment.source_public_status,
                                outcome,
                            )
            elif members_possible and kernel.expected_member_count() > 0:
                for outcome in kernel.member_outcomes:
                    if outcome.probability > 0:
                        inspect(
                            outcome.components,
                            incident.scenario_id,
                            segment.source_public_status,
                            outcome,
                        )
    return frozenset(linear - positive)


def _component_upper_support(component: ComponentAmountLaw) -> float:
    pre, post = component.multiplier, component.post_multiplier
    if pre == 0 or post == 0:
        return 0.0
    if component.amount is not None:
        return component.amount * pre * post
    assert component.severity is not None
    sampler = native_severity_sampler(component.severity)
    top = replace(sampler, severity_cap=None).quantile(1) * pre
    cap = sampler.severity_cap
    return (top if cap is None else min(top, cap)) * post


def _max_member_count(kernel) -> float:
    cluster = kernel.cluster
    if cluster is None:
        return 0.0
    return max(row.count for row in cluster.support if row.probability > 0)


def _aggregate_upper_support(law) -> float:
    assert law.severity is not None
    return native_severity_sampler(law.severity).quantile(1)


def _episode_lab_upper_support(kernel, lab_id: str) -> float:
    """The largest uncapped episode lab loss the kernel can produce."""

    def allocated(components) -> float:
        return math.fsum(
            _component_upper_support(component) * weight
            for component in components
            if (weight := (component.lab_loss_weights or {}).get(lab_id, 0.0)) > 0
        )

    def members(outcomes, count) -> float:
        tops = [allocated(outcome.components) for outcome in outcomes]
        top = max(tops, default=0.0)
        return 0.0 if top == 0 or count == 0 else top * count

    members_possible = kernel.shared_probability > 0 and kernel.unit_probability > 0
    if kernel.explicit_joint_members:
        episode = max(
            (
                allocated(branch.campaign_components)
                + (
                    math.fsum(
                        allocated(outcome.components) for outcome in branch.members
                    )
                    if members_possible
                    else 0.0
                )
                for branch in kernel.explicit_joint_members
                if branch.probability > 0
            ),
            default=0.0,
        )
    else:
        episode = (
            members(kernel.possible_member_outcomes(), _max_member_count(kernel))
            if members_possible
            else 0.0
        )
    episode += allocated(kernel.campaign_components)
    for law in kernel.campaign_lab_losses:
        if law.lab_id == lab_id:
            episode += (
                law.amount if law.amount is not None else _aggregate_upper_support(law)
            )
    return episode


def _lab_parts(components, lab_id: str):
    """The fixed lab amount and the scale of one shared stochastic total.

    Returns ``(fixed, scale, total)``: the lab's episode allocation is
    ``fixed + scale * total`` when every stochastic lab-allocated component
    shares one severity law and rank group with no pre-cap multiplier, as a
    shared total partitioned by fixed shares is. Otherwise returns None.
    """
    fixed, scale, total, rank = 0.0, 0.0, None, None
    for component in components:
        weight = (component.lab_loss_weights or {}).get(lab_id, 0.0)
        if weight == 0 or component.post_multiplier == 0 or component.multiplier == 0:
            continue
        if component.amount is not None:
            fixed += (
                component.amount
                * component.multiplier
                * component.post_multiplier
                * weight
            )
            continue
        assert component.severity is not None
        law = component.severity.model_dump_json()
        if component.multiplier != 1 or (
            total is not None and (law, component.rank_group_id) != (total, rank)
        ):
            return None
        total, rank = law, component.rank_group_id
        scale += component.post_multiplier * weight
    return fixed, scale, total


def capped_episode_lab_mean(kernel, aggregate) -> tuple[float, float] | None:
    """Mean of one episode's capped lab loss, ``E[min(s*T + k + A, c)]``.

    Returns ``(mean, error_bound)``, where ``error_bound`` certifies the
    numerical integration error (``_capped_branch_mean``). ``T`` is the shared
    total that every stochastic lab-allocated component of a branch partitions
    (scale ``s``), ``k`` the fixed lab amounts and ``A`` the independent
    aggregate amount, with cap ``c``. Branches mix by their probabilities, and
    a gated-out member leaves ``min(k + A, c)``. Conditional on ``A = a`` the
    mean is ``k + a + s*LEV_T((c - k - a)/s)`` below the cap and ``c`` at or
    above it, where ``LEV_T`` is the total's limited expected value; a fixed
    ``A`` needs no further integration and a stochastic one is integrated
    numerically. Kernels with clustered members, explicit joint members or
    differently ranked stochastic allocations, and integrals that cannot be
    certified within ``CERTIFIED_INTEGRATION_RELATIVE_ERROR``, return None and
    keep a simulated lab mean.
    """
    lab_id, cap = aggregate.lab_id, aggregate.episode_cap
    cluster = kernel.cluster
    if (
        cap is None
        or kernel.explicit_joint_members
        or cluster is None
        or any(row.count > 1 and row.probability > 0 for row in cluster.support)
    ):
        return None
    campaign = _lab_parts(kernel.campaign_components, lab_id)
    if campaign is None or campaign[1] > 0:
        return None
    base = campaign[0]
    member = (
        kernel.shared_probability
        * kernel.unit_probability
        * math.fsum(row.probability for row in cluster.support if row.count == 1)
    )
    other = (
        repr(aggregate.amount)
        if aggregate.amount is not None
        else aggregate.severity.model_dump_json()
    )
    branches = [(1 - member, _capped_branch_mean(other, None, 0.0, base, cap))]
    for outcome in kernel.member_outcomes:
        if outcome.probability == 0 or member == 0:
            continue
        parts = _lab_parts(outcome.components, lab_id)
        if parts is None:
            return None
        fixed, scale, law = parts
        branches.append(
            (
                member * outcome.probability,
                _capped_branch_mean(other, law, scale, base + fixed, cap),
            )
        )
    results = [(weight, result) for weight, result in branches if result is not None]
    if len(results) < len(branches):
        return None
    return (
        math.fsum(weight * mean for weight, (mean, _) in results),
        math.fsum(weight * bound for weight, (_, bound) in results),
    )


CERTIFIED_INTEGRATION_RELATIVE_ERROR = 1e-3
"""Certified relative error of a numerically integrated capped lab mean or
probability of at least one episode.

Judgment (``NUM-PRECISION``, range 1e-4/1e-3/1e-2): refinement stops once the
monotone bounds differ by at most this share of the value's lower bound, so the
reported value is within this relative distance of the exact one.
"""
QUADRATURE_MAX_INTERVALS = 2**17
"""Refinement cap: a capped mean whose bounds are still wider after this many
intervals is not reported as an analytic mean."""
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(4)


def _knots(law: NativeSeverityLaw) -> set[float]:
    """Values where a native law's distribution function can kink or jump.

    A component with a large-loss anchor adds its p99 break and each anchor
    amount, where the density of its anchored tail steps.
    """
    return {
        float(value)
        for component in law.components
        for value in (
            component.p5,
            component.p25,
            component.p50,
            component.p75,
            component.p95,
            component.severity_minimum,
            component.severity_cap,
            *(
                ()
                if component.large_loss_amount is None
                else (
                    Distribution(**component.model_dump()).large_loss_break,
                    component.large_loss_amount,
                    component.very_large_loss_amount,
                )
            ),
        )
        if value is not None
    }


def _monotone_integral(
    integrand, edges: np.ndarray, base: float
) -> tuple[float, float] | None:
    """Integral of a non-increasing integrand over ``edges``, with its error bound.

    On an interval ``[lo, hi]`` the integral of a non-increasing function lies
    between ``(hi - lo) f(hi)`` and ``(hi - lo) f(lo)``. Intervals whose bound
    gap exceeds a quarter of the average are split each round, at the
    geometric midpoint when the interval spans more than a factor of four
    (so mass near a wide interval's lower end is resolved on a log scale) and
    at the arithmetic midpoint otherwise. Refinement stops when the total gap
    is at most ``CERTIFIED_INTEGRATION_RELATIVE_ERROR * (base + lower bound)``.
    Gauss-Legendre with positive weights on the final intervals gives the
    value, which lies inside the bounds, so the gap certifies its error.
    Returns None when the refinement cap is reached first.
    """
    lo, hi = edges[:-1], edges[1:]
    f_lo, f_hi = integrand(lo), integrand(hi)
    while True:
        width = hi - lo
        gaps = width * (f_lo - f_hi)
        lower = math.fsum(width * f_hi)
        gap = math.fsum(gaps)
        if gap <= CERTIFIED_INTEGRATION_RELATIVE_ERROR * (base + lower):
            break
        split = gaps > gap / (4 * len(lo))
        if len(lo) + np.count_nonzero(split) > QUADRATURE_MAX_INTERVALS:
            return None
        left, right = lo[split], hi[split]
        wide = (left > 0) & (right > 4 * left)
        middle = np.where(
            wide, np.sqrt(np.where(wide, left, 1.0) * right), (left + right) / 2
        )
        moved = (middle > left) & (middle < right)
        if not np.any(moved):
            return None
        split[np.flatnonzero(split)[~moved]] = False
        left, right, middle = left[moved], right[moved], middle[moved]
        f_middle = integrand(middle)
        f_left, f_right = f_lo[split], f_hi[split]
        keep = ~split
        lo = np.concatenate((lo[keep], left, middle))
        hi = np.concatenate((hi[keep], middle, right))
        f_lo = np.concatenate((f_lo[keep], f_left, f_middle))
        f_hi = np.concatenate((f_hi[keep], f_middle, f_right))
    half = (hi - lo) / 2
    amounts = ((lo + hi) / 2)[:, None] + half[:, None] * _GAUSS_NODES[None, :]
    value = math.fsum(
        (half[:, None] * _GAUSS_WEIGHTS[None, :] * integrand(amounts)).ravel()
    )
    upper = math.fsum((hi - lo) * f_lo)
    lower = math.fsum((hi - lo) * f_hi)
    value = min(max(value, lower), upper)
    return value, max(upper - value, value - lower)


@lru_cache(maxsize=65536)
def _capped_branch_mean(
    other: str, total: str | None, scale: float, fixed: float, cap: float
) -> tuple[float, float] | None:
    """``E[min(scale*T + fixed + A, cap)]`` for independent ``T`` and ``A``.

    Returns ``(mean, error_bound)``. With ``L = cap - fixed`` and
    ``g(a) = E[min(scale*T + fixed + a, cap)]``,
    ``g(a) = fixed + a + scale*LEV_T((L - a)/scale)`` below ``L``, so
    ``E[g(A)] = g(0) + integral over (0, L) of F_T((L - a)/scale) S_A(a) da``.
    ``g(0)`` is the total's closed-form limited expected value, and a fixed
    amount or a degenerate total needs no integral (bound zero). The
    integrand is a product of two non-increasing functions of ``a``, so
    ``_monotone_integral`` integrates it with a certified bound, starting from
    segments split at every point where either distribution function can kink
    or jump. None means the bound could not be certified.
    """
    limit = cap - fixed
    shared = (
        None
        if total is None or scale == 0
        else native_severity_sampler(NativeSeverityLaw.model_validate_json(total))
    )

    def conditional(amount: float) -> float:
        if amount >= limit:
            return cap
        if shared is None:
            return fixed + amount
        return (
            fixed
            + amount
            + scale
            * shared.analytic_mean(severity_cap_override=(limit - amount) / scale)
        )

    if not other.startswith("{"):
        return conditional(float(other)), 0.0
    if limit <= 0:
        return cap, 0.0
    law = NativeSeverityLaw.model_validate_json(other)
    aggregate = native_severity_sampler(law)
    if shared is None:
        return fixed + aggregate.analytic_mean(severity_cap_override=limit), 0.0
    assert total is not None
    points = (
        {0.0, limit}
        | {value for value in _knots(law) if 0 < value < limit}
        | {
            limit - scale * value
            for value in _knots(NativeSeverityLaw.model_validate_json(total))
            if 0 < limit - scale * value < limit
        }
    )

    def integrand(amounts: np.ndarray) -> np.ndarray:
        return shared.cdf((limit - amounts) / scale) * (1.0 - aggregate.cdf(amounts))

    base = conditional(0.0)
    integral = _monotone_integral(integrand, np.array(sorted(points)), base)
    if integral is None:
        return None
    return base + integral[0], integral[1]


def _finite_mean(sampler: SeveritySampler) -> float | None:
    if any(shape >= 1 for shape in uncapped_infinite_variance_gpd_shapes(sampler)):
        return None
    return sampler.analytic_mean()


def component_amount_mean(component: ComponentAmountLaw) -> float | None:
    """Integrate pre-scale, outer cap and post-scale in their declared order."""
    pre, post = component.multiplier, component.post_multiplier
    if pre == 0 or post == 0:
        return 0.0
    if component.amount is not None:
        return component.amount * pre * post
    assert component.severity is not None
    sampler = native_severity_sampler(component.severity)
    cap = sampler.severity_cap
    if pre != 1 or post != 1:
        lower, _ = severity_lower_support(sampler, apply_severity_cap=False)
        if cap is not None and product_at_least(lower, pre, cap):
            return cap * post
        sampler = replace(
            replace(sampler, severity_cap=None).with_currency_scale(pre, post),
            severity_cap=None if cap is None else cap * post,
        )
    return _finite_mean(sampler)


@dataclass(frozen=True)
class AnalyticEals:
    """Analytic annual means by view and their certified numerical error bounds.

    ``error_bounds`` holds, for every view in ``values``, an upper bound on the
    absolute difference between the reported and exact mean: zero for views
    integrated in closed form, positive where a capped lab mean or an annual lab
    term's probability of at least one episode was integrated numerically
    (``CERTIFIED_INTEGRATION_RELATIVE_ERROR``).
    """

    values: dict[str, float]
    error_bounds: dict[str, float]


def analytic_linear_eals(
    compiled: CompiledIncidentState,
    views: Sequence[str],
    *,
    annual: bool = True,
    episodes: bool = True,
) -> AnalyticEals:
    """Integrate complete native marks without using realized count means.

    Episode terms are the annual rate times an episode's mean. An origin's
    annual lab law adds its mean times the probability of at least one
    episode in a year (``at_least_one_probability``), whose certified
    integration error enters the view's bound; ``annual=False`` leaves it
    out, for a mean that is linear in the expected count, and
    ``episodes=False`` keeps only it.
    """
    bounds: dict[str, float] = {}
    estimates = {
        view: 0.0
        for view in views
        if view not in {"financial", "oep_gross", "oep_financial"}
        and not view.startswith("risk_unit::")
        and view not in compiled.unavailable_views
        and view not in compiled.source_status_unavailable
    }
    unavailable = set()
    for incident in compiled.incidents:
        if incident.annual_rate == 0:
            continue
        annual_laws = {
            view: laws
            for view, laws in annual_views(incident).items()
            if view in estimates
        }
        if annual and annual_laws:
            occurrence = at_least_one_probability(incident, compiled.count_experiment)
            for view, laws in annual_laws.items():
                means = [
                    law.amount
                    if law.severity is None
                    else _finite_mean(native_severity_sampler(law.severity))
                    for law in laws
                ]
                if occurrence is None or any(
                    mean is None or not math.isfinite(mean) for mean in means
                ):
                    unavailable.add(view)
                    continue
                mean = math.fsum(cast(list[float], means))
                value = estimates[view] + occurrence[0] * mean
                bound = bounds.get(view, 0.0) + occurrence[1] * mean
                if math.isfinite(value) and math.isfinite(bound):
                    estimates[view], bounds[view] = value, bound
                else:
                    unavailable.add(view)
        if not episodes:
            continue
        for segment in incident.segments:
            if segment.weight == 0:
                continue
            kernel = segment.kernel
            aggregate_laws = _aggregate_lab_views(kernel, segment.source_public_status)
            # Lab views whose capped episode total is integrated as a whole.
            capped_views: set[str] = set()
            for view, aggregate in aggregate_laws.items():
                if view not in estimates or view in unavailable:
                    continue
                if aggregate.episode_cap is not None and (
                    _episode_lab_upper_support(kernel, aggregate.lab_id)
                    > aggregate.episode_cap
                ):
                    # A cap that can bind makes the episode lab loss nonlinear,
                    # so the capped episode total is integrated exactly where
                    # its structure allows and is otherwise left to simulation.
                    capped = capped_episode_lab_mean(kernel, aggregate)
                    if capped is None:
                        unavailable.add(view)
                        continue
                    capped_views.add(view)
                    value = estimates[view] + scaled_product(
                        incident.annual_rate, segment.weight, capped[0]
                    )
                    bound = bounds.get(view, 0.0) + scaled_product(
                        incident.annual_rate, segment.weight, capped[1]
                    )
                    if math.isfinite(value) and math.isfinite(bound):
                        estimates[view], bounds[view] = value, bound
                    else:
                        unavailable.add(view)
                    continue
                mean = aggregate.amount
                if aggregate.severity is not None:
                    mean = _finite_mean(native_severity_sampler(aggregate.severity))
                if mean is None or not math.isfinite(mean):
                    unavailable.add(view)
                    continue
                value = estimates[view] + scaled_product(
                    incident.annual_rate, segment.weight, mean
                )
                if math.isfinite(value):
                    estimates[view] = value
                else:
                    unavailable.add(view)
            terms: list[
                tuple[ComponentAmountLaw | None, float, JointMarkOutcome | None]
            ] = [(component, 1.0, None) for component in kernel.campaign_components]

            def add_outcome(
                outcome: JointMarkOutcome,
                probability: float,
                *,
                target=terms,
                source_public_status=segment.source_public_status,
            ):
                if probability == 0:
                    return
                if outcome.identified_gross_total is not None:
                    target.append((None, probability, outcome))
                if not outcome.composition_complete or not outcome.overlap_resolved:
                    unavailable.update(
                        view
                        for view in estimates
                        if (view != "gross" or outcome.identified_gross_total is None)
                        and (
                            (scope := source_status_view(view)) is None
                            or scope[0] == source_public_status
                        )
                    )
                target.extend(
                    (component, probability, outcome)
                    for component in outcome.components
                )

            member_probability = kernel.shared_probability * kernel.unit_probability
            if kernel.explicit_joint_members:
                for branch in kernel.explicit_joint_members:
                    terms.extend(
                        (component, branch.probability, None)
                        for component in branch.campaign_components
                    )
                    for outcome in branch.members:
                        add_outcome(outcome, branch.probability * member_probability)
            else:
                for outcome in kernel.member_outcomes:
                    add_outcome(
                        outcome,
                        outcome.probability
                        * member_probability
                        * kernel.expected_member_count(),
                    )
            for component, probability, outcome in terms:
                if probability == 0:
                    continue
                if component is None:
                    assert (
                        outcome is not None
                        and outcome.identified_gross_total is not None
                    )
                    if "gross" in estimates and "gross" not in unavailable:
                        value = estimates["gross"] + scaled_product(
                            incident.annual_rate,
                            segment.weight,
                            probability,
                            outcome.identified_gross_total,
                        )
                        if math.isfinite(value):
                            estimates["gross"] = value
                        else:
                            unavailable.add("gross")
                    continue
                if component.multiplier == 0 or component.post_multiplier == 0:
                    continue
                weights = {
                    view: 0.0
                    if view in capped_views
                    or (
                        view == "gross"
                        and outcome is not None
                        and outcome.identified_gross_total is not None
                    )
                    else _coefficient(
                        view,
                        (component,),
                        (1,),
                        incident.scenario_id,
                        outcome,
                        segment.source_public_status,
                    )
                    for view in estimates
                    if view not in unavailable
                }
                if not any(weights.values()):
                    continue
                mean = component_amount_mean(component)
                for view, weight in weights.items():
                    if weight == 0:
                        continue
                    if mean is None:
                        unavailable.add(view)
                    else:
                        value = estimates[view] + scaled_product(
                            incident.annual_rate,
                            segment.weight,
                            probability,
                            mean,
                            weight,
                        )
                        if math.isfinite(value):
                            estimates[view] = value
                        else:
                            unavailable.add(view)
    available = {
        view: value for view, value in estimates.items() if view not in unavailable
    }
    return AnalyticEals(available, {view: bounds.get(view, 0.0) for view in available})


def single_period_law_means(
    law: IncidentLaw,
    views: Sequence[str],
    *,
    annual: bool = True,
    episodes: bool = True,
) -> dict[str, float]:
    """Analytic annual means for one law at its reference rate.

    ``annual=False`` omits the law's annual lab losses, leaving the episode
    terms, which are linear in the expected count; ``episodes=False`` keeps
    only the annual terms (``analytic_linear_eals``).

    A capped lab mean, and an annual term under a count with a parameter rank,
    carry a certified relative error of at most
    ``CERTIFIED_INTEGRATION_RELATIVE_ERROR``; the other terms are closed-form.

    Only a law with one whole-period segment of full exposure and initiation,
    as the contextual compilers produce, has its reference annual rate as its
    integrated rate; any other law is refused rather than integrated here.
    Views without a finite analytic mean are omitted.
    """
    (segment,) = law.segments
    if (
        (segment.start, segment.end) != (0, 1)
        or segment.reference_exposure_share != 1
        or segment.initiation_probability != 1
    ):
        raise ValueError("only a single whole-period segment has its reference rate")
    incident = CompiledIncident(
        origin_id=law.origin_id,
        scenario_id=law.scenario_id,
        entity_id=law.entity_id,
        annual_rate=law.reference_annual_rate,
        count_family=law.count_family,
        count_variance_to_mean=law.count_variance_to_mean,
        segments=(
            CompiledMarkSegment(
                start=0,
                end=1,
                integrated_rate=law.reference_annual_rate,
                weight=1,
                kernel=segment.kernel,
                source_public_status=segment.source_public_status,
            ),
        ),
        parameter_rank=law.parameter_rank,
        annual_lab_losses=law.annual_lab_losses,
    )
    state = CompiledIncidentState(
        state_id=law.origin_id,
        epistemic_group_id=law.origin_id,
        rank_stream_id=law.origin_id,
        alternative_id="reference",
        conditional_key=law.origin_id,
        law_sha256="0" * 64,
        evidence_sha256="0" * 64,
        incidents=(incident,),
        frequency_audit=(),
        contracts=(),
        declared_view_ids={},
    )
    return analytic_linear_eals(state, views, annual=annual, episodes=episodes).values
