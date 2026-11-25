"""The exhibits of a contextual setup, and the respondent selection they read.

A kept respondent's final answers, factors and exhibit inputs are one
``ContextualRespondentSelection``. Every node at a reference state adds a
``frequency_estimate`` row (``frequency_contribution``) and a
``large_loss_anchor`` row (``large_loss_contribution``), which the exhibit
records and evidence files summarize.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import numpy as np
from scipy.special import ndtri

from catmodel.contracts.contextual_risk import (
    ContextualRiskAnswer,
    ContextualRiskArtifact,
    ContextualRiskInstrument,
    ContextualRiskNode,
)
from catmodel.contracts.elicitation import (
    Quantiles,
    Respondent,
)
from catmodel.contracts.incident import (
    Evidence,
    IncidentLaw,
    ParameterRankLaw,
    PopulationAbsence,
    counts_lab_loss,
    gross_view,
)
from catmodel.contracts.run_setup import (
    EvidenceSet,
)
from catmodel.elicitation.chain_law import CHAIN_POOL_WEIGHT
from catmodel.elicitation.eal_crosscheck import (
    StatedLargeLoss,
    large_loss_status,
)
from catmodel.elicitation.epistemic import epistemic_mean, stretched_score_quantiles
from catmodel.scenarios.hierarchy_contract import large_loss_levels
from catmodel.scenarios.sensitivity_coverage import (
    TAIL_SHAPE_SENSITIVITY_STEP,
    shifted_tail_shape,
)
from catmodel.setup.contextual_node import (
    CompileOptions,
    ContextualLawUnavailable,
    bracket_anchors,
    integrated_count,
    interpolate,
    interpolated_rate,
    scaled_rate,
    shared_total_sampler,
    tail_measures,
)
from catmodel.setup.sensitivity_catalog import (
    FRONTIER_LAB_VALUE_SENSITIVITY_USD,
    FRONTIER_LAB_VALUE_USD,
    LARGE_LOSS_ANCHOR_FACTORS,
    usd_text,
)
from catmodel.simulation.incident_moments import single_period_law_means


def large_loss_limitation(artifacts: Sequence[ContextualRiskArtifact]) -> str:
    """How many cases' large-loss anchors remain unresolved, and that no
    reference law reads the very-large-loss count (``SEV-LARGE-LOSS-ANCHOR``)."""
    statuses = Counter(
        large_loss_status(case) for artifact in artifacts for case in artifact.cases
    )
    unresolved = statuses.get("severity_unresolved", 0)
    asked = sum(count for status, count in statuses.items() if status != "not_asked")
    return (
        f"Large-loss anchor: {unresolved} of {asked} answered cases are "
        "severity_unresolved, their large-loss count still contradicting their loss "
        "quantiles after one retry, so their laws use the provisional p99-only tail; "
        "the large_loss_anchor exhibit gives their gross EAL share. "
        f"{statuses.get('no_anchor', 0)} cases stated an exact-zero anchor and "
        f"{statuses.get('resolved', 0)} resolved a contradiction on retry. "
        "The count at the very-large-loss level enters no reference law: beyond "
        "the large-loss level the governed tail shape alone sets how many "
        "incidents reach it, and the exhibit's two_anchor alternative gives each "
        "view's EAL with the tail passed through the stated count instead."
    )


def lab_loss_cap_limitation(*, alternatives_executed: bool) -> str:
    """Disclose the frontier-lab value used and whether its sensitivity alternatives ran."""
    value = usd_text(FRONTIER_LAB_VALUE_USD)
    lower, higher = map(usd_text, FRONTIER_LAB_VALUE_SENSITIVITY_USD)
    executed = "executed" if alternatives_executed else "not executed"
    return (
        "Each episode's lab loss from provider-borne components plus compensation, "
        f"and separately each node's annual business loss, is capped at {value}, the stipulated "
        "enterprise value of a frontier lab during the study period, the same for "
        "every lab, or at the scenario's tail-class endpoint where that is lower "
        "(SEV-LAB-CAP, an INITIAL_ESTIMATE judgment). Limited liability bounds what "
        "a lab can bear; any excess falls on creditors, counterparties or victims, "
        "and the welfare view, which counts the complete resource loss whoever "
        "bears it, is unchanged. The cap bounds one episode or one node's annual "
        "business loss: a year's total lab loss can exceed the value when several "
        "large episodes, or several nodes' business losses, occur in one year. "
        f"The lab_loss_cap sensitivity alternatives ({lower}, {higher} and no lab cap) "
        f"are {executed} in this setup."
    )


@dataclass(frozen=True)
class ContextualRespondentSelection:
    """One respondent's final answers, factors and exhibit inputs.

    Parameter ranks are not selected here: each law carries its count's
    epistemic law (``ParameterRankLaw``), read at a fresh rank each simulated
    year.
    """

    selection_id: str
    respondent: Respondent
    weight: float
    answers_by_node: Mapping[str, Mapping[float, ContextualRiskAnswer]]
    """Each case's pooled answer carrying its final count (``pooled_answer``)
    times the factor of the taxonomy parents' pools above it."""
    frequency_answers_by_node: Mapping[
        str, Mapping[float, Mapping[str, ContextualRiskAnswer | None]]
    ]
    """Each frequency exhibit arm's answer at every node and anchor."""
    severity_factors_by_node: Mapping[str, Mapping[float, float]]
    """Each case's severity factor on each event's loss above the admission
    threshold, before the outer cap. The taxonomy parents' pools scale counts
    only (``HIER-CATEGORY-POOL``), so each is one."""
    frequency_alternatives_by_node: Mapping[
        str, Mapping[float, Mapping[str, Quantiles | None]]
    ]
    """Each case's alternative counts for the ``frequency_estimate`` exhibit;
    None where the estimate is unavailable."""
    large_loss_by_node: Mapping[str, Mapping[float, StatedLargeLoss]]
    """Each case's stated large-loss counts and anchor status for the
    ``large_loss_anchor`` exhibit."""
    donors_by_node: Mapping[str, tuple[str, str, int]] = MappingProxyType({})
    """For each leaf the respondent's model does not answer, the key of the
    respondent whose stored cases the node's answers are
    (``POOL-UNANSWERED-SCENARIOS``); every other node's are the respondent's
    own."""

    def answered_by(self, node_id: str) -> tuple[str, str, int]:
        """The key of the respondent that gave the node's answers."""
        return self.donors_by_node.get(node_id, self.respondent.key())


FREQUENCY_ESTIMATES = (
    "direct",
    "chain",
    "before_severity_retry",
    "before_chain_guardrail",
    "before_eal_crosscheck",
    "before_hierarchy",
    "hierarchy_parents_only",
)
"""Alternative expected counts of the ``frequency_estimate`` exhibit, each
integrated over the parameter rank.

The cross-checked direct count (pool weight zero); the chain count law
(pool weight one, with direct where no chain is available); the pool with
each direct answer as first given in place of its resolving large-loss
coherence retry, and the pool with each chain answer as first given in place
of its guardrail retry, each keeping every other stored answer; the pool of
the direct count before its EAL cross-check; the pooled count before the
taxonomy parents' pools, their weight zero; and that count with weight one at
every parent, each category's own answer standing for its category and
allocated to what is beneath it in proportion (``HIER-CATEGORY-POOL``), with
factors derived at ``k = 1`` and ``b = 1``. Every alternative compiles its own
answers, so each is an EAL-only comparison.
"""
HIERARCHY_ESTIMATES = ("before_hierarchy", "hierarchy_parents_only")
"""The two ends of the taxonomy parents' pool weight, zero and one; both are
inapplicable where the instrument asks no parents."""
FREQUENCY_ESTIMATE_ARMS = ("final", *FREQUENCY_ESTIMATES)
"""Every arm of the ``frequency_estimate`` exhibit.

``final`` is the run's own law, each node's expected count its mean over the
parameter rank, ``E_U[Q(U)]``: the run's reference EAL (``NUM-BUDGETS``).
Every alternative is integrated over the rank too.
"""
FOLLOW_UP_ARMS = {
    "severity_retry": "before_severity_retry",
    "chain_guardrail": "before_chain_guardrail",
    "eal_crosscheck": "before_eal_crosscheck",
}
"""The exhibit arm that undoes each follow-up call of the collection, in the
order collection asks them: the large-loss coherence retry of a direct answer
(``SEV-LARGE-LOSS-ANCHOR``), the chain guardrail retry
(``FREQ-CHAIN-GUARDRAIL``) and the leaf EAL cross-check
(``FREQ-EAL-CROSSCHECK``)."""
_NOT_REPLACED = {
    "severity_retry": "No kept case's direct answer was replaced by a large-loss coherence retry.",
    "chain_guardrail": "No kept case's chain answer was replaced by a guardrail retry.",
    "eal_crosscheck": "No kept case's EAL cross-check revised a judgment the run reads.",
}
_NO_PARENTS = "The instrument asks no taxonomy-parent questions."
_RETRY_ARMS = ("before_severity_retry", "before_chain_guardrail")
"""The arms that are inapplicable where their retry replaced no answer."""


def follow_up_effects(
    by_arm: Mapping[str, Mapping[str, object]],
    not_replaced: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    """Each follow-up call's EAL effect in one view of the exhibit.

    A follow-up's EAL before it is that of the arm that undoes it
    (``FOLLOW_UP_ARMS``). Its EAL after it is that of the pooled, cross-checked
    count before the taxonomy parents' pools (``before_hierarchy``), which
    every follow-up precedes; where the instrument asks no parents it is the
    run's own (``final``). The relative change is after over before, less
    one. ``not_replaced`` gives the reason for each follow-up that replaced
    no kept answer, whose effect is inapplicable; an effect is unavailable,
    with the reason, where either arm is.
    """
    pooled = (
        "final"
        if by_arm["before_hierarchy"]["status"] == "inapplicable"
        else "before_hierarchy"
    )
    effects: dict[str, dict[str, object]] = {}
    for kind, before_arm in FOLLOW_UP_ARMS.items():
        before, after = by_arm[before_arm], by_arm[pooled]
        row: dict[str, object] = {"before_arm": before_arm, "after_arm": pooled}
        missing = next(
            (arm for arm in (before, after) if arm["status"] != "available"), None
        )
        if kind in not_replaced:
            row |= {"status": "inapplicable", "reason": not_replaced[kind]}
        elif missing is not None:
            row |= {"status": missing["status"], "reason": missing["reason"]}
        else:
            first = cast(float, before["expected_annual_loss"])
            second = cast(float, after["expected_annual_loss"])
            row |= {
                "status": "available",
                "expected_annual_loss_before": first,
                "expected_annual_loss_after": second,
                "relative_change": (second - first) / first if first else None,
            }
        effects[kind] = row
    return effects


def frequency_estimate_views(instrument: ContextualRiskInstrument) -> tuple[str, ...]:
    """The exhibit's views: welfare, gross and the named lab's loss."""
    return ("welfare", "gross", f"lab_loss::{instrument.lab.entity_id}")


def frequency_contribution(
    instrument: ContextualRiskInstrument,
    node: ContextualRiskNode,
    selection: ContextualRespondentSelection,
    score: float,
    law: IncidentLaw | PopulationAbsence,
    *,
    compile_law: Callable[..., IncidentLaw | PopulationAbsence],
    options: CompileOptions = CompileOptions(),
) -> dict[str, dict[str, float | None]]:
    """One node's annual mean at one state under each count of the exhibit.

    Each count is integrated over its parameter rank at ``options``'s spread
    (``epistemic_tail_scale``, ``rank_spread`` and ``spread_tail_form``) and
    scaled by its ``count_factor``, the final count and every alternative
    alike, so the arms compare count answers under one reading.

    The node's episode terms are linear in its expected count in every view
    here, because severity laws do not depend on the count and the episode
    lab-loss cap applies per episode. Each arm therefore rescales an analytic
    episode mean per expected count by its count law's epistemic mean,
    integrated over its parameter rank. The gross and lab views add the annual
    business-loss term, the annual law's mean times the probability of at
    least one incident under the arm's own count law (``annual_terms``). The
    final count is the run's pooled law, with the factor of the taxonomy
    parents' pools on its count; every alternative compiles its own answers.
    Each per-count mean is the analytic mean of a compiled law, never a
    rescaled one. The per-count mean of a node compiled only for that purpose
    comes from ``compile_law``, the caller's own compilation of ``(answers,
    severity_factors)``, applied to the same answers with ``unit_count=True``.
    None marks a value the exhibit cannot identify: an unavailable estimate,
    law or analytic mean.
    """
    arms = FREQUENCY_ESTIMATE_ARMS
    views = frequency_estimate_views(instrument)
    result: dict[str, dict[str, float | None]] = {
        view: dict.fromkeys(arms) for view in views
    }
    if node.unavailable_reason is not None or (
        isinstance(law, PopulationAbsence) and law.disposition == "unavailable"
    ):
        return result
    answers = selection.answers_by_node[node.node_id]
    alternative_answers = selection.frequency_answers_by_node[node.node_id]
    alternatives = selection.frequency_alternatives_by_node[node.node_id]
    try:
        lower, upper, fraction = bracket_anchors(answers, score)
    except ContextualLawUnavailable:
        return result

    def final_rate(anchor: float) -> Quantiles | None:
        answer = answers[anchor]
        return answer.rate if answer.status == "available" else None

    def integrated(left: Quantiles | None, right: Quantiles | None) -> float | None:
        if left is None or right is None:
            return None
        return integrated_count(
            left,
            right,
            fraction,
            options.epistemic_tail_scale,
            options.rank_spread,
            options.spread_tail_form,
            options.count_factor,
        )

    estimates = {
        "final": integrated(final_rate(lower), final_rate(upper)),
        **{
            arm: integrated(alternatives[lower][arm], alternatives[upper][arm])
            for arm in FREQUENCY_ESTIMATES
        },
    }
    severity = selection.severity_factors_by_node[node.node_id]
    bracket = {anchor: answers[anchor] for anchor in (lower, upper)}

    PerCount = tuple[dict[str, float], IncidentLaw | None]

    def per_count(
        source: IncidentLaw | PopulationAbsence, basis: float | None
    ) -> PerCount:
        """Episode means per expected count, and the law for its annual term."""
        if not isinstance(source, IncidentLaw) or not basis:
            return {}, None
        means = single_period_law_means(source, views, annual=False)
        return {view: value / basis for view, value in means.items()}, source

    basis = estimates["final"]
    final = (
        per_count(law, basis)
        if isinstance(law, IncidentLaw) and basis
        else per_count(compile_law(bracket, severity, unit_count=True), 1.0)
        if any((estimates[arm] or 0) > 0 for arm in arms)
        else ({}, None)
    )
    laws: dict[tuple[int, ...], PerCount] = {}
    annual_means: dict[tuple[int, ...], dict[str, float | None]] = {}

    def law_of(arm: str) -> PerCount:
        """Per-count means, and the annual-term law, of an exhibit arm's own
        answers. Arms that hold the same answer objects, as a retry's arm
        does for a case the retry did not replace, share one law and its
        means."""
        if not (estimates[arm] or 0) > 0:
            return {}, None
        selected_answers = {}
        for anchor in (lower, upper):
            answer = alternative_answers[anchor][arm]
            if answer is None:
                raise ContextualLawUnavailable(
                    f"The positive {arm} frequency estimate has no answer."
                )
            selected_answers[anchor] = answer
        key = tuple(id(answer) for answer in selected_answers.values())
        if key not in laws:
            laws[key] = per_count(
                compile_law(selected_answers, None, unit_count=True), 1.0
            )
        return laws[key]

    def annual_terms(source: IncidentLaw | None, arm: str) -> dict[str, float | None]:
        """Each view's annual lab term under the arm's own count: the annual
        law's mean times the probability of at least one incident in a year,
        integrated over that count's rank. The term enters the gross and lab
        views. A chain count has no five-quantile law to integrate, so its term
        is unidentified wherever it is used. Arms that share a law and their
        count objects share the term."""
        if source is None or not source.annual_lab_losses:
            return dict.fromkeys(views, 0.0)
        pair = (
            (final_rate(lower), final_rate(upper))
            if arm == "final"
            else (alternatives[lower][arm], alternatives[upper][arm])
        )
        key = (id(source), id(pair[0]), id(pair[1]))
        if key not in annual_means:
            annual_means[key] = _annual_terms(source, pair)
        return annual_means[key]

    def _annual_terms(
        source: IncidentLaw, pair: tuple[Quantiles | None, Quantiles | None]
    ) -> dict[str, float | None]:
        means: dict[str, float] = {}
        if isinstance(pair[0], Quantiles) and isinstance(pair[1], Quantiles):
            count = scaled_rate(
                interpolated_rate(pair[0], pair[1], fraction), options.count_factor
            )
            mean = integrated_count(
                pair[0],
                pair[1],
                fraction,
                options.epistemic_tail_scale,
                options.rank_spread,
                options.spread_tail_form,
                options.count_factor,
            )
            if mean is not None:
                means = single_period_law_means(
                    source.model_copy(
                        update={
                            "reference_annual_rate": mean,
                            "parameter_rank": ParameterRankLaw(
                                group_id=node.node_id,
                                count=count,
                                epistemic_tail_scale=options.epistemic_tail_scale,
                                rank_spread=options.rank_spread,
                                tail_form=options.spread_tail_form,
                            ),
                        }
                    ),
                    views,
                    episodes=False,
                )
        # The annual loss enters the gross and lab views, not welfare.
        return {view: 0.0 if view == "welfare" else means.get(view) for view in views}

    for arm in arms:
        value = estimates[arm]
        if value is None or value == 0:
            for view in views:
                result[view][arm] = None if value is None else 0.0
            continue
        episode, source = final if arm == "final" else law_of(arm)
        annual = annual_terms(source, arm)
        for view in views:
            mean, extra = episode.get(view), annual[view]
            result[view][arm] = (
                None if mean is None or extra is None else value * mean + extra
            )
    return result


_RANK_POINTS = (np.arange(4096) + 0.5) / 4096
_RANK_SCORES = ndtri(_RANK_POINTS)


def _log_no_incident(law: IncidentLaw, survival: float) -> float:
    """``log E_Z[exp(-lambda(Z) S)]``: a year without an incident at or above a
    level of exceedance probability ``S``, averaged over the count's normal
    scores of 4,096 midpoint ranks, ample for this bounded smooth integrand.

    The mean is taken about the largest exponent, so a count law with
    hundreds of expected incidents at the level at every rank, where each
    ``exp`` underflows, keeps its finite log."""
    rank = law.parameter_rank
    if rank is None or survival == 0:
        return -law.reference_annual_rate * survival
    counts = stretched_score_quantiles(
        rank.count,
        _RANK_SCORES,
        spread=rank.rank_spread,
        epistemic_tail_scale=rank.epistemic_tail_scale,
        tail_form=rank.tail_form,
    )
    exponents = -survival * counts
    peak = float(np.max(exponents))
    return peak + math.log(float(np.mean(np.exp(exponents - peak))))


LARGE_LOSS_EAL_ALTERNATIVES = (
    "anchor_ignored",
    "anchor_lower",
    "anchor_higher",
    "two_anchor",
)
"""The exhibit's analytic EAL alternatives: the anchor ignored (the p99-only tail, a
design comparison), every shaping share's conditional odds above p99
scaled by each of ``LARGE_LOSS_ANCHOR_FACTORS``, and the two-anchor tail,
which also passes every law through its stated very-large-loss share where
that share is coherent (``CompileOptions.very_large_loss_anchor``)."""


def large_loss_contribution(
    instrument: ContextualRiskInstrument,
    node: ContextualRiskNode,
    selection: ContextualRespondentSelection,
    score: float,
    law: IncidentLaw | PopulationAbsence,
    *,
    compile_law: Callable[..., IncidentLaw | PopulationAbsence],
    options: CompileOptions = CompileOptions(),
) -> dict[str, object] | None:
    """One node's ``large_loss_anchor`` exhibit row at one state, or None.

    Stated counts are integrated over their parameter rank at ``options``'s
    spread (``epistemic_tail_scale``, ``rank_spread`` and
    ``spread_tail_form``) and scaled by its ``count_factor``, as the run
    law's count is.

    Counts are per state before weighting: the stated large-loss count means
    (direct and pooled), and the count law's mean over the rank
    times each law's survival at the levels: the run's law (modelled), the
    p99-only law (extrapolated, the anchor ignored) and the run's law at the
    governed shape 0.2 lower and higher. ``eal`` holds each view's EAL under
    the run's law and under each of ``LARGE_LOSS_EAL_ALTERNATIVES``, each the count
    times the analytic mean per expected count of the law it uses.
    ``two_anchor`` holds the tail measures of the two-anchor law, the run's
    own where the very-large-loss share does not shape it, and
    ``very_large_role`` how that share enters the two-anchor law.
    ``log_none`` is the log probability of no such incident in a year,
    averaged over the count's parameter rank (``_log_no_incident``); it
    assumes Poisson counts, so a node with another count family has none,
    with its reason. A row's status combines the bracketing anchors' cases:
    ``severity_unresolved`` if either is, otherwise the lower anchor's.
    ``case`` names the stored cases the row's answers are, by context and
    the respondent that gave them (``ContextualRespondentSelection.answered_by``).
    """
    total_spec = node.total
    if (
        node.unavailable_reason is not None
        or total_spec is None
        or total_spec.large_loss_level is None
        or total_spec.gpd_shape is None
    ):
        return None
    level, very = total_spec.large_loss_level, total_spec.very_large_loss_level
    answers = selection.answers_by_node[node.node_id]
    stated_cases = selection.large_loss_by_node[node.node_id]
    try:
        lower, upper, fraction = bracket_anchors(answers, score)
    except ContextualLawUnavailable:
        return None
    left, right = stated_cases[lower], stated_cases[upper]

    def mean_of(first: Quantiles | None, second: Quantiles | None) -> float | None:
        if first is None or second is None:
            return None
        return options.count_factor * epistemic_mean(
            interpolated_rate(first, second, fraction),
            spread=options.rank_spread,
            epistemic_tail_scale=options.epistemic_tail_scale,
            tail_form=options.spread_tail_form,
        )

    unresolved = "severity_unresolved" in (left.status, right.status)
    row: dict[str, object] = {
        # The stored case the row's answers are: a donor's for a borrowed leaf,
        # so the exhibit's counts of cases count it once.
        "case": "/".join(
            (instrument.context_id, *map(str, selection.answered_by(node.node_id)))
        ),
        "status": "severity_unresolved" if unresolved else left.status,
        "stated": mean_of(left.stated, right.stated),
        "stated_very": mean_of(left.stated_very, right.stated_very),
        "pooled": mean_of(left.final, right.final),
        "pooled_very": mean_of(left.final_very, right.final_very),
        "probability_withheld": None
        if node.count_family == "poisson"
        else f"The node's {node.count_family} count is not Poisson, so 1 - exp(-count x survival) is not its probability of at least one incident.",
    }
    if isinstance(law, PopulationAbsence):
        if law.disposition == "unavailable":
            return row | {"unavailable": law.reason}
        return row | {
            "count": 0.0,
            "log_none": 0.0,
            "role": "no_anchor",
            "very_large_role": None if very is None else "no_anchor",
        }
    count = law.reference_annual_rate
    bracket = {anchor: answers[anchor] for anchor in (lower, upper)}
    factors = selection.severity_factors_by_node[node.node_id]
    severity = interpolate(factors.get(lower, 1.0), factors.get(upper, 1.0), fraction)
    record = cast(
        dict[str, object],
        law.conditioning_context.get("large_loss_anchor") or {},
    )
    modelled = shared_total_sampler(law)
    views = frequency_estimate_views(instrument)

    def per_count(
        source: IncidentLaw | PopulationAbsence,
        basis: float = 1.0,
    ) -> dict[str, float | None] | str:
        if isinstance(source, PopulationAbsence):
            return "unavailable"
        means = single_period_law_means(source, views, annual=False)
        return {
            view: None
            if (mean := means.get(view)) is None or (extra := annual.get(view)) is None
            else mean / basis + extra
            for view in views
        }

    # Episode means are linear in the expected count. The run law's annual lab
    # term, per expected count, is the same under every alternative, since an alternative
    # changes only the shared total; the row's count times it is the term.
    annual: dict[str, float] = (
        {
            view: value / law.reference_annual_rate
            for view, value in single_period_law_means(
                law, views, episodes=False
            ).items()
        }
        if law.reference_annual_rate > 0
        else dict.fromkeys(views, 0.0)
    )
    run: dict[str, float | None] | str = (
        per_count(law, law.reference_annual_rate)
        if law.reference_annual_rate > 0
        else per_count(compile_law(bracket, factors, unit_count=True))
    )
    # An alternative changes only a shaping anchor, so every other law is the run law.
    shapes = record.get("large_loss_role") == "shapes_tail_above_p99"
    ignored = (
        compile_law(bracket, factors, unit_count=True, large_loss_factor=None)
        if shapes
        else law
    )
    # The two-anchor law differs from the run's only where the stated
    # very-large-loss share is positive and coherent; an exact-zero share is
    # no second anchor and needs no compilation.
    two_anchor = (
        compile_law(bracket, factors, unit_count=True, very_large_loss_anchor=True)
        if very is not None and row["pooled_very"]
        else None
    )
    second = cast(
        dict[str, object],
        (two_anchor.conditioning_context.get("very_large_loss_anchor") or {})
        if isinstance(two_anchor, IncidentLaw)
        else {},
    )
    # The law the very-large-loss share shapes, where it does.
    shaped = (
        two_anchor
        if isinstance(two_anchor, IncidentLaw)
        and second.get("very_large_loss_role")
        in ("shapes_tail_above_large_loss_level", "shapes_tail_above_p99")
        else None
    )
    eal_per_count = {
        "run": run,
        "anchor_ignored": per_count(ignored) if shapes else run,
        # A two-anchor law that does not compile is unavailable, never the
        # run's law.
        "two_anchor": per_count(shaped)
        if shaped is not None
        else per_count(two_anchor)
        if isinstance(two_anchor, PopulationAbsence)
        else run,
        **{
            alternative: per_count(
                compile_law(bracket, factors, unit_count=True, large_loss_factor=factor)
            )
            if shapes
            else run
            for alternative, factor in zip(
                ("anchor_lower", "anchor_higher"),
                LARGE_LOSS_ANCHOR_FACTORS,
                strict=True,
            )
        },
    }
    revised = any(factors.get(anchor, 1.0) != 1 for anchor in (lower, upper))
    unscaled = shared_total_sampler(
        compile_law(bracket, None, unit_count=True) if revised else law
    )
    shape = total_spec.gpd_shape
    alternatives = {}
    for alternative, step in (
        ("shape_lower", -TAIL_SHAPE_SENSITIVITY_STEP),
        ("shape_higher", TAIL_SHAPE_SENSITIVITY_STEP),
    ):
        stepped = shifted_tail_shape(shape, step)
        try:
            shifted = (
                None
                if unscaled is None
                else unscaled.with_gpd_shape(stepped).with_excess_scale(severity)
            )
        except ValueError:
            shifted = None
        alternatives[alternative] = tail_measures(shifted, level, very)
    measured = tail_measures(modelled, level, very)
    return row | {
        "count": count,
        "role": record.get("large_loss_role"),
        "log_none": None
        if measured is None or row["probability_withheld"] is not None
        else _log_no_incident(law, measured["large"]),
        "modelled": measured,
        "extrapolated": tail_measures(shared_total_sampler(ignored), level, very)
        if shapes
        else measured,
        "eal_per_count": eal_per_count,
        "two_anchor": measured
        if shaped is None
        else tail_measures(shared_total_sampler(shaped), level, very),
        "very_large_role": None
        if very is None or isinstance(two_anchor, PopulationAbsence)
        else second.get("very_large_loss_role", "no_anchor"),
        **alternatives,
    }


def _weighted(rows, key: str, inner: str | None = None) -> float | None:
    """Sum of weight x count x measure (or weight x value) over rows."""
    total = 0.0
    for weight, row in rows:
        if inner is None:
            value = row.get(key)
            if value is None:
                return None
            total += weight * value
            continue
        measures = row.get(key)
        if measures is None or row.get("count") is None:
            return None
        value = measures[inner]
        if value is None:
            return None
        total += weight * row["count"] * value
    return total


def _eal_alternatives(
    selected,
    views,
    lab_loss_weights: Mapping[str, float] | None = None,
    gross_withheld: str | None = None,
) -> dict[str, object]:
    """Each view's run EAL and each analytic alternative's EAL and relative
    effect. A view that counts the lab's loss weighs each row's state by
    ``lab_loss_weights``, where given, and passes over a state it weighs
    zero; a gross view has no value where ``gross_withheld`` says why."""

    def total(rows, alternative: str, view: str) -> float | None:
        value = 0.0
        lab = lab_loss_weights is not None and counts_lab_loss(view)
        for weight, row in rows:
            if lab:
                weight = lab_loss_weights[row["state"]]
            means = row.get("eal_per_count", {}).get(alternative)
            count = row.get("count")
            if count == 0 or weight == 0:
                continue
            if not isinstance(means, dict) or count is None or means.get(view) is None:
                return None
            value += weight * count * means[view]
        return value

    result: dict[str, object] = {}
    for view in views:
        withheld = gross_withheld is not None and gross_view(view)
        base = None if withheld else total(selected, "run", view)
        alternatives: dict[str, object] = {}
        for alternative in LARGE_LOSS_EAL_ALTERNATIVES:
            value = None if withheld else total(selected, alternative, view)
            if value is None or not base:
                alternatives[alternative] = {"status": "unavailable"}
                continue
            alternatives[alternative] = {
                "status": "available",
                "expected_annual_loss": value,
                "relative_effect": (value - base) / base,
            }
        result[view] = {"expected_annual_loss": base, **alternatives}
    return result


def large_loss_record(
    rows: Sequence[tuple[str, float, str, Mapping[str, object] | None]],
    levels: tuple[float, float | None] | None,
    views: Sequence[str] = ("welfare", "gross"),
    *,
    lab_loss_weights: Mapping[str, float] | None = None,
    gross_withheld: str | None = None,
) -> dict[str, object]:
    """Aggregate state rows into the ``large_loss_anchor`` exhibit.

    ``rows`` holds (state ID, reference weight, scenario ID, row) at
    positive weight. Expected annual numbers and EALs are weighted sums; the
    probability of at least one incident at or above the large-loss level is
    computed within each joint state, from every scenario's probability of no
    such incident averaged over its parameter rank, and then mixed, so it
    carries the model's conditional independence across scenarios given the
    state. Statuses and roles count each case once: a case that several
    respondents' states read, a borrowed one, counts as its own respondent's.

    ``lab_loss_weights`` gives each state's weight in the views that count
    the lab's loss, the lab-loss views and the gross views, where a state is
    left out of them, and ``gross_withheld`` why the gross views have no
    value, where they have none (``excluded_samples.lab_loss_state_weights``).
    Those views' EAL alternatives follow both. Everything else uses the rows'
    weights: the counts and the gross shares measure each incident's shared
    total, which holds no annual business loss.
    """
    by_scenario: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    for state_id, weight, scenario_id, row in rows:
        if row is None or weight == 0:
            continue
        by_scenario[scenario_id].append(
            (weight, dict(row) | {"state": state_id, "scenario": scenario_id})
        )

    def at_least_one(selected) -> float | None:
        """Mix 1 - prod of no-incident probabilities over states, each weighted
        once; ranks and counts are independent across scenarios in a state."""
        states: dict[str, tuple[float, list[float]]] = {}
        for weight, row in selected:
            value = row.get("log_none")
            if value is None:
                return None
            states.setdefault(row["state"], (weight, []))[1].append(value)
        return math.fsum(
            weight * -math.expm1(math.fsum(values))
            for weight, values in states.values()
        )

    def share(numerator, denominator):
        return None if numerator is None or not denominator else numerator / denominator

    def reasons(selected, key) -> list[str]:
        return sorted({str(row[key]) for _, row in selected if row.get(key)})

    def summary(selected) -> dict[str, object]:
        unavailable = sorted(
            {str(row["unavailable"]) for _, row in selected if "unavailable" in row}
        )
        if unavailable:
            return {"status": "unavailable", "reasons": unavailable}
        withheld_probability = reasons(selected, "probability_withheld")
        gross = _weighted(selected, "modelled", "gross")
        large = _weighted(selected, "modelled", "gross_large")
        unresolved = _weighted(
            [
                (weight, row)
                for weight, row in selected
                if row["status"] == "severity_unresolved"
            ],
            "modelled",
            "gross",
        )
        alternatives = {}
        for alternative in ("shape_lower", "shape_higher"):
            alternatives[alternative] = {
                "very_large_count": _weighted(selected, alternative, "very_large"),
                "large_count": _weighted(selected, alternative, "large"),
                "gross_eal_share_at_or_above_large": share(
                    _weighted(selected, alternative, "gross_large"),
                    _weighted(selected, alternative, "gross"),
                ),
            }
        return {
            "status": "available",
            "stated_large_count": _weighted(selected, "stated"),
            "pooled_large_count": _weighted(selected, "pooled"),
            "extrapolated_large_count": _weighted(selected, "extrapolated", "large"),
            "modelled_large_count": _weighted(selected, "modelled", "large"),
            "stated_very_large_count": _weighted(selected, "stated_very"),
            "pooled_very_large_count": _weighted(selected, "pooled_very"),
            "modelled_very_large_count": _weighted(selected, "modelled", "very_large"),
            "probability_at_least_one_large": None
            if withheld_probability
            else at_least_one(selected),
            "probability_withheld_reasons": withheld_probability,
            "gross_eal_share_at_or_above_large": share(large, gross),
            # A shared total is its node's whole estimand (ContextualRiskNode),
            # so no node has gross loss outside it and no share is withheld.
            "gross_eal_share_withheld_reasons": [],
            "severity_unresolved_gross_eal_share": share(unresolved, gross),
            "tail_shape_alternatives": alternatives,
            "two_anchor_tail": {
                "very_large_count": _weighted(selected, "two_anchor", "very_large"),
                "large_count": _weighted(selected, "two_anchor", "large"),
                "gross_eal_share_at_or_above_large": share(
                    _weighted(selected, "two_anchor", "gross_large"),
                    _weighted(selected, "two_anchor", "gross"),
                ),
            },
            "eal_alternatives": _eal_alternatives(
                selected, views, lab_loss_weights, gross_withheld
            ),
        }

    def counts(rows_, key) -> dict[str, int]:
        cases: dict[str, dict] = {}
        for _, row in rows_:
            cases.setdefault(str(row["case"]), row)
        return dict(
            sorted(
                Counter(
                    str(row[key]) for row in cases.values() if row.get(key) is not None
                ).items()
            )
        )

    scenarios = {}
    for scenario_id, selected in sorted(by_scenario.items()):
        scenarios[scenario_id] = summary(selected) | {
            "case_statuses": counts(selected, "status"),
            "law_roles": counts(selected, "role"),
            "very_large_law_roles": counts(selected, "very_large_role"),
        }
    flat = [pair for selected in by_scenario.values() for pair in selected]
    total: dict[str, object] = summary(flat) if flat else {"status": "inapplicable"}
    if flat and total["status"] == "available":
        for key in ("case_statuses", "law_roles", "very_large_law_roles"):
            merged: Counter[str] = Counter()
            for row in scenarios.values():
                merged.update(cast(dict, row.get(key, {})))
            total[key] = dict(sorted(merged.items()))
        total["severity_unresolved_cases"] = cast(dict, total["case_statuses"]).get(
            "severity_unresolved", 0
        )
    return {
        "method": "analytic_tail_measures",
        "levels": None
        if levels is None
        else {"large_loss": levels[0], "very_large_loss": levels[1]},
        "interpretation": "Expected annual numbers of admitted incidents at or above the large-loss levels at the reference arm, with every count integrated over its parameter rank: as stated by the direct answers and after EAL cross-checking and fixed pooling, and as modelled by the run's laws, by the laws with the large-loss anchor ignored (extrapolated: the p99-only tail) and by the run's laws at the governed shape 0.2 lower and higher. eal_alternatives gives each view's EAL, over only the scenarios this exhibit reports, under the run's laws and the analytic anchor alternatives: the anchor ignored (a design comparison with the p99-only tail, not a lower bound) and every shaping share's conditional odds of reaching the level, given an incident above the law's own p99, multiplied by one third and by three (a judgmental size, keeping which cases the anchor shapes; scaling the odds rather than the share keeps every alternative's share below 1 in 100, where a share shapes the tail above p99, and for a small share it approaches scaling the share by one third and three). two_anchor compiles every law with the two-anchor tail: where the stated very-large-loss share is positive and coherent, the tail also passes through it, as a second power law from the large-loss level to the very-large-loss level and the governed tail conditioned on reaching that level, or, where the large-loss level lies within the stated quantiles, as the only anchor above the law's own p99; every other law keeps the run's tail. No reference law reads the very-large-loss count, so this alternative shows what the stated count would do to EAL, and two_anchor_tail gives the counts at both levels and the gross EAL share under it; very_large_law_roles counts how each case's share enters it. These alternatives are EAL effects only: no VaR or TVaR; the opt-in simulated large_loss_anchor and very_large_loss_anchor families give their tail effects. The probability of at least one incident at or above the large-loss level in the year is computed within each joint state from its sampled counts and then mixed; the total assumes, as the model does, that scenarios are independent given the state, and it is withheld for a node whose count is not Poisson. The gross EAL shares, including the severity-unresolved share, are the share of expected gross loss from incidents at or above the large-loss level, withheld for a node with additive components outside its shared total. law_roles counts how each case's anchor enters its law, including the severity_unresolved_fallback p99-only tail and interpolated band conflicts. A row's status combines its bracketing anchors: severity_unresolved if either is, otherwise the lower anchor's.",
        "by_scenario": scenarios,
        "total": total,
    }


def frequency_estimate_record(
    contributions: Sequence[
        tuple[str, float, str, Mapping[str, Mapping[str, float | None]]]
    ],
    *,
    frequency_collection: Mapping[str, object],
    eal_crosscheck: Mapping[str, object],
    follow_ups: Mapping[str, Mapping[str, int]],
    hierarchy: Mapping[str, object] | None = None,
    lab_loss_weights: Mapping[str, float] | None = None,
    gross_withheld: str | None = None,
) -> dict[str, object]:
    """Aggregate state contributions into the ``frequency_estimate`` exhibit.

    ``contributions`` holds (state ID, reference weight, scenario ID,
    contribution) rows. ``lab_loss_weights`` gives each state's weight in the
    views that count the lab's loss, the lab-loss views and the gross views,
    where a state is left out of them
    (``excluded_samples.lab_loss_state_weights``); such a view weighs the
    rows by it, and every other view by the rows' own weights. A gross view
    is unavailable in every arm where ``gross_withheld`` says why. An arm is
    available for a view only when every contribution with weight in the
    view identifies it. ``follow_ups`` counts the
    kept leaf cases each follow-up call was asked of and those whose answer
    it replaced (``elicitation.eal_crosscheck.follow_up_counts``). An arm
    that undoes a retry is inapplicable where the retry replaced no kept
    case's answer, and so is the effect of any follow-up that replaced none.
    ``hierarchy`` is the taxonomy parents' summary
    (``contextual_hierarchy.hierarchy_summary``), empty where the instrument
    asks none.
    """
    arms = FREQUENCY_ESTIMATE_ARMS
    counts = {kind: dict(row) for kind, row in follow_ups.items()}
    if set(counts) != set(FOLLOW_UP_ARMS):
        raise ValueError("follow-up counts must cover each follow-up call")
    not_replaced = {
        kind: _NOT_REPLACED[kind]
        for kind in FOLLOW_UP_ARMS
        if counts[kind]["answers_replaced"] == 0
    }
    inapplicable = {
        arm: not_replaced[kind]
        for kind, arm in FOLLOW_UP_ARMS.items()
        if arm in _RETRY_ARMS and kind in not_replaced
    }
    views: dict[str, object] = {}
    for view in contributions[0][3] if contributions else ():
        totals = dict.fromkeys(arms, 0.0)
        by_scenario: dict[str, dict[str, float | None]] = {}
        gaps: dict[str, set[str]] = {arm: set() for arm in arms}
        lab = lab_loss_weights is not None and counts_lab_loss(view)
        for state_id, weight, scenario_id, contribution in contributions:
            if lab:
                weight = lab_loss_weights[state_id]
            if weight == 0:
                continue
            row = by_scenario.setdefault(scenario_id, dict.fromkeys(arms, 0.0))
            for arm in arms:
                value = contribution[view][arm]
                if value is None:
                    gaps[arm].add(scenario_id)
                    row[arm] = None
                    continue
                totals[arm] += weight * value
                current = row[arm]
                if current is not None:
                    row[arm] = current + weight * value
        withheld = gross_withheld if gross_view(view) else None
        if withheld is not None:
            by_scenario = {scenario: dict.fromkeys(arms) for scenario in by_scenario}
        by_arm: dict[str, dict[str, object]] = {
            arm: {"status": "inapplicable", "reason": _NO_PARENTS}
            if arm in HIERARCHY_ESTIMATES and not hierarchy
            else {"status": "inapplicable", "reason": inapplicable[arm]}
            if arm in inapplicable
            else {"status": "unavailable", "reason": withheld}
            if withheld is not None
            else {
                "status": "available",
                "expected_annual_loss": totals[arm],
            }
            if not gaps[arm]
            else {
                "status": "unavailable",
                "reason": "No rescaled mean for scenarios "
                + ", ".join(sorted(gaps[arm]))
                + ": an estimate, law or analytic mean is unavailable at a positive-weight state.",
            }
            for arm in arms
        }
        views[view] = {
            "by_arm": by_arm,
            "by_scenario": dict(sorted(by_scenario.items())),
            "follow_up_effects": follow_up_effects(by_arm, not_replaced),
        }
    return {
        "method": "analytic_frequency_rescaling",
        "interpretation": "Expected annual loss under each count arm, with every five-quantile count law integrated over its parameter rank. final is the pooled count after the EAL cross-check, scaled by the taxonomy parents' pools: each parent's own answer and what is beneath it are pooled by a fixed rule, bottom-up, the answer weighing one over the parent's height plus one, and a pool's factor scales the count of every leaf beneath the parent. direct is the cross-checked direct count at chain pool weight zero; chain is the chain count law at chain pool weight one, using direct where no chain is available. These two arms bracket the fixed chain pool weight 0.5. before_severity_retry uses each direct answer as first given in place of its resolving large-loss coherence retry, so its contradicting large-loss share compiles the provisional p99-only tail; before_chain_guardrail pools each chain answer as first given in place of its guardrail retry. Each keeps every other stored answer, the EAL cross-check's revisions included, and is inapplicable where the retry replaced no kept case's answer. before_eal_crosscheck pools the direct count before its EAL check with the chain law and uses the severity answers available then, so its difference from before_hierarchy isolates the cross-check's EAL effect. before_hierarchy is the pooled count after the cross-check and before the taxonomy parents' pools, their weight zero. hierarchy_parents_only sets that weight to one at every parent, so that each category's own answer stands for its category and is allocated to what is beneath it in proportion. The two are the ends of the category answer's weight; for one parent the pooled value lies between them, and across levels final need not. The chain pool and the EAL cross-check were decided at collection, and the taxonomy parents' factors are derived from the stated count laws, all at k = 1 and b = 1. Each arm's expected annual loss is its count law's epistemic mean times the analytic mean per expected count of the law used. follow_up_effects sets the arm that undoes each follow-up call beside before_hierarchy (final where no taxonomy parent is asked); a follow-up that replaced no kept answer has no effect to report. follow_ups counts the kept cases each follow-up was asked of and those whose answer it replaced. No alternative arm is simulated.",
        "tail_statistics": "unavailable: rescaled counts change annual loss distributions nonlinearly, and no alternative arm is simulated.",
        "chain_pool_weight": CHAIN_POOL_WEIGHT,
        "views": views,
        "follow_ups": {kind: counts[kind] for kind in FOLLOW_UP_ARMS},
        "frequency_collection": dict(frequency_collection),
        "eal_crosscheck": dict(eal_crosscheck),
        "hierarchy": dict(hierarchy or {}),
    }


FREQUENCY_ESTIMATE_PATH = "derived/frequency-estimate.json"


LARGE_LOSS_ANCHOR_PATH = "derived/large-loss-anchor.json"


def large_loss_levels_of(
    nodes: Sequence[ContextualRiskNode],
) -> tuple[float, float | None] | None:
    """The authored large-loss levels, from the hierarchy contract."""
    if not any(
        node.total is not None and node.total.large_loss_level is not None
        for node in nodes
    ):
        return None
    levels = large_loss_levels()
    return levels.large_loss, levels.very_large_loss


def frequency_estimate_evidence(
    record: Mapping[str, object], large_loss: Mapping[str, object] | None = None
) -> EvidenceSet:
    """Carry the exhibits to results with the setup's other evidence."""
    data = json.dumps(dict(record), sort_keys=True, allow_nan=False).encode()
    digest = hashlib.sha256(data).hexdigest()
    evidence = Evidence(
        evidence_id="frequency_estimate",
        origin="elicitation",
        original_sha256=digest,
        selected_sha256=digest,
        target="Reference expected annual loss under each node's pooled expected count, direct-only and chain-only count laws, and dual-collection validation metrics.",
        conditioning="The captured direct and chain answers, EAL cross-checks and taxonomy-parent answers at the reference arm's ranks and weights.",
        units="Annual loss in each view's monetary units; log ratios of expected counts.",
        transport="Rescaling of each state's analytic mean by its node's count ratio; no simulation.",
        acquisition={"frequency_estimate": dict(record)},
        review="Derived from captured answers; the counts remain unvalidated judgments.",
        reliance="An exhibit of the frequency judgments' disagreement, not a calibrated frequency.",
        rights="Derived from the captured answers; their rights are retained.",
    )
    if large_loss is None:
        return EvidenceSet(records=(evidence,), files={FREQUENCY_ESTIMATE_PATH: data})
    tail = json.dumps(dict(large_loss), sort_keys=True, allow_nan=False).encode()
    tail_digest = hashlib.sha256(tail).hexdigest()
    anchor = Evidence(
        evidence_id="large_loss_anchor",
        origin="elicitation",
        original_sha256=tail_digest,
        selected_sha256=tail_digest,
        target="Expected annual numbers of incidents at or above the large-loss levels as stated, extrapolated and modelled, the probability of at least one, the gross EAL share from them and each case's anchor status.",
        conditioning="The captured direct and chain answers, EAL cross-checks and taxonomy-parent answers at the reference arm's ranks and weights.",
        units="Expected annual incident counts, probabilities and shares of gross annual loss.",
        transport="Analytic tail measures of each state's compiled laws; no simulation.",
        acquisition={"large_loss_anchor": dict(large_loss)},
        review="Derived from captured answers; the large-loss counts remain unvalidated judgments.",
        reliance="An exhibit of the tail judgments, not a calibrated frequency.",
        rights="Derived from the captured answers; their rights are retained.",
    )
    return EvidenceSet(
        records=(evidence, anchor),
        files={FREQUENCY_ESTIMATE_PATH: data, LARGE_LOSS_ANCHOR_PATH: tail},
    )


def contextual_nodes(
    instrument: ContextualRiskInstrument,
) -> tuple[ContextualRiskNode, ...]:
    """The instrument's nodes in ID order, once their joint use is admitted."""
    selected = tuple(sorted(instrument.nodes, key=lambda node: node.node_id))
    if len(selected) > 1 and instrument.independent_node_origins_rationale is None:
        raise ContextualLawUnavailable(
            "Multiple nodes require an explicit independent_node_origins_rationale declaring disjoint initiating incidents, unique cross-node amounts and independent node counts."
        )
    return selected
