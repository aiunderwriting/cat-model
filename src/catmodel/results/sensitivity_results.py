"""Small result exhibits over the existing reduction and reference capture."""

import math
from dataclasses import replace

import numpy as np

from catmodel.results.dependence_bracket import dependence_bracket
from catmodel.results.plausibility import plausibility_table
from catmodel.results.respondent_sampling import (
    NOT_ANALYTIC_REASON,
    ViewWeights,
    borrowed_reason,
    quadrature_sampling,
    row_sampling,
    sample_groups,
    sample_variance,
)
from catmodel.results.sensitivity_summary import scenario_view, sensitivity_summary
from catmodel.simulation.capability_mixture import (
    CapabilityMixture,
)
from catmodel.simulation.incident_runtime import state_key
from catmodel.simulation.tail_gate import TailGateControls

CONCENTRATED_NODE_SHARE = 0.95
"""Embedded-rule node share above which the indicator is uninformative.

The coarser rule then evaluates the arm at essentially one node, so it agrees
with the full rule whatever the error. A judgment threshold.
"""

NO_NODE_SD = (
    "the node's samples are not independent and no standard deviation is reported."
)
"""What a borrowing state does to the standard deviation of its trajectory
node's state EALs (``respondent_sampling.borrowed_reason``)."""

ONE_SAMPLE_REASON = (
    "The node holds one sample; a standard deviation needs at least two."
)

REWEIGHTED_EXHIBIT_REASON = (
    "These results reweight the captured states to target weights other than the "
    "setup's own. The exhibit was computed from the setup's weights (the frequency "
    "rescaling and hierarchy summary are frozen in the setup's evidence; the "
    "embedded-rule comparison needs each arm's own trajectory quadrature weights), "
    "so it does not describe the reweighted target and is not reported for it."
)


def _trajectory_quadrature(package, draws, view):
    """Compare each integrated arm's EAL under the full and embedded node rules.

    Only the states that some listed arm weights enter either rule, and each
    one's conditional EAL comes from ``draws``, the states' draws under the
    reference arm's law (``_reference_law_draws``), so a node no listed arm
    weights, such as the no-progress node without the no-progress arm, needs no
    draw. Each arm also reports the difference's respondent-sampling standard
    error on the same states and its breakdown by node. A node's standard
    deviation of state EALs counts its samples as independent, so it is
    withheld with its reason where one of them reads a donor's answers
    (``SampleGroups.borrower``), as the standard error is. Each arm's weights are
    its weights in ``view`` (``SampleGroups.view_weights``), so a view that
    counts the lab's loss is integrated over the states that are not left out
    of it. None without a public-trajectory record.
    """
    record = next(
        (
            row.acquisition["trajectory_quadrature"]
            for row in package.evidence.records
            if row.acquisition and "trajectory_quadrature" in row.acquisition
        ),
        None,
    )
    if record is None:
        return None
    result = {
        "rule": record["rule"],
        "interior_nodes": record["interior_nodes"],
        "embedded_interior_nodes": record["embedded_interior_nodes"],
        "view": view,
        "interpretation": "EAL under the full trajectory rule and under its embedded lower-order rule on the same collected nodes and conditional laws. Where the indicator is informative, the difference indicates the quadrature error of the coarser rule, and the full rule's error is usually smaller; neither is bounded. It is uninformative when one node carries more than 95% of the embedded rule's weight, and exact for the no-progress point mass. Losses that change sharply between nodes, such as threshold-like steps, can err in either direction. It is not a forecast or model uncertainty. The two rules integrate the same respondent samples, so the difference also carries respondent-sampling noise: sampling_standard_error is its standard error (NUM-RESPONDENT-SAMPLING) and sampling_z the difference over it, and where |z| is below 2 the indicator does not distinguish integration error from respondent-sampling noise. by_node gives each node with drawn states: its frontier progress, its weights under both rules, its kept samples, the mean of their EALs at their weights within the node and their unbiased standard deviation, and its EAL contributions under both rules and to the difference, which sum to the arm's. The standard deviation counts the node's samples as independent: where one of them reads another model's answers to a scenario its own model does not answer (POOL-UNANSWERED-SCENARIOS), state_eal_sd is not reported and state_eal_sd_reason says why.",
    }
    if view is None:
        return {
            **result,
            "status": "unavailable",
            "reason": "No available view was selected.",
        }
    groups = sample_groups(package.joint_states)
    weights_by_arm = {}
    for arm, weights in {
        package.execution.reference_arm: {
            state.state_id: state.weight for state in package.joint_states
        },
        **{
            arm.alternative_id: arm.state_weights
            for arm in package.alternatives
            if arm.state_weights is not None
        },
    }.items():
        by_view = groups.view_weights(weights)
        found = by_view.of(view)
        if found is None:
            if arm in record["draw_weights"]:
                return {
                    **result,
                    "status": "unavailable",
                    "reason": by_view.reason(view),
                }
            continue
        weights_by_arm[arm] = found
    # A state enters the full rule with its arm weight and the embedded rule
    # with that weight scaled by its node's embedded share, so only states
    # some listed arm weights matter.
    weighted = {
        state
        for arm in record["draw_weights"]
        for state, weight in weights_by_arm[arm].items()
        if weight > 0
    }
    conditional = {}
    for state in package.joint_states:
        if (
            state.state_id not in record["state_draws"]
            or state.state_id not in weighted
        ):
            continue
        draw = draws.get(state_key(state.state_id))
        found = None if draw is None else draw.views.get(view)
        if found is None or not found.mean_exists:
            return {
                **result,
                "status": "unavailable",
                "reason": f"State {state.state_id!r} has no finite conditional mean in the selected view.",
            }
        conditional[state.state_id] = (draw, found)
    native = all(found.native_mean is not None for _, found in conditional.values())

    def state_eal(draw, found) -> float:
        if native and found.native_mean is not None:
            return found.native_mean.value
        return float(np.mean(found.array))

    eal = {
        state_id: state_eal(draw, found)
        for state_id, (draw, found) in conditional.items()
    }
    node_of = record["state_draws"]
    members = {
        node: sorted(state for state in eal if node_of[state] == node)
        for node in record["node_progress"]
    }
    # Each node's samples at their shares of the node, from the first listed
    # arm that weights it: every arm weights a node's samples in proportion.
    shares = {}
    for node, states in members.items():
        for arm in record["draw_weights"]:
            masses = [weights_by_arm[arm][state] for state in states]
            total = math.fsum(masses)
            if total > 0:
                shares[node] = [mass / total for mass in masses]
                break
    by_arm = {}
    for arm, nodes in record["draw_weights"].items():
        weights = weights_by_arm[arm]
        # Each state's weight under the full rule and under the embedded rule.
        full_terms = {
            state: (weights[state], value)
            for state, value in eal.items()
            if weights[state] > 0
        }
        embedded_terms = {
            state: (
                weight
                * nodes["embedded"][node_of[state]]
                / nodes["full"][node_of[state]],
                value,
            )
            for state, (weight, value) in full_terms.items()
            if nodes["full"][node_of[state]] > 0
            and nodes["embedded"][node_of[state]] > 0
        }
        full = math.fsum(weight * value for weight, value in full_terms.values())
        embedded = math.fsum(
            weight * value for weight, value in embedded_terms.values()
        )
        concentrated = max(nodes["embedded"].values()) > CONCENTRATED_NODE_SHARE
        by_node = {}
        for node, states in members.items():
            if node not in shares:
                continue
            values = [eal[state] for state in states]
            borrowing = groups.borrower(states)
            spread_reason = (
                borrowed_reason(*borrowing, NO_NODE_SD)
                if borrowing is not None
                else ONE_SAMPLE_REASON
                if len(values) < 2
                else None
            )
            contributions = [
                math.fsum(
                    terms[state][0] * terms[state][1]
                    for state in states
                    if state in terms
                )
                for terms in (full_terms, embedded_terms)
            ]
            by_node[node] = {
                "progress": record["node_progress"][node],
                "full_weight": nodes["full"][node],
                "embedded_weight": nodes["embedded"][node],
                "samples": len(states),
                "state_eal_mean": math.fsum(
                    share * value
                    for share, value in zip(shares[node], values, strict=True)
                ),
                "state_eal_sd": math.sqrt(sample_variance(values))
                if spread_reason is None
                else None,
                "state_eal_sd_reason": spread_reason,
                "full_contribution": contributions[0],
                "embedded_contribution": contributions[1],
                "difference_contribution": contributions[0] - contributions[1],
            }
        by_arm[arm] = {
            "expected_annual_loss": full,
            "embedded_rule_expected_annual_loss": embedded,
            "absolute_difference": full - embedded,
            "relative_difference": (full - embedded) / full if full else None,
            **(
                quadrature_sampling(groups, full_terms, embedded_terms, full - embedded)
                if native
                else {
                    "sampling_standard_error": None,
                    "sampling_z": None,
                    "sampling_reason": NOT_ANALYTIC_REASON,
                }
            ),
            "by_node": by_node,
            "indicator": "exact"
            if nodes["variant"] == "no_progress"
            else "uninformative"
            if concentrated
            else "informative",
            "indicator_reason": "The no-progress law is a point mass on the no-progress node, so neither rule has quadrature error."
            if nodes["variant"] == "no_progress"
            else f"One node carries more than {CONCENTRATED_NODE_SHARE:.0%} of the embedded rule's weight, so that rule evaluates the arm at one node and cannot reveal the full rule's error."
            if concentrated
            else None,
        }
    return {
        **result,
        "status": "available",
        "estimator": "native_analytic_mean" if native else "empirical_mean",
        "by_arm": by_arm,
    }


def _frequency_estimate(package, metrics):
    """The ``frequency_estimate`` exhibit and dual-collection metrics.

    The setup carries each view's reference EAL under the final count and
    under each alternative count, each integrated over the rank from every
    state's analytic means. This adds each arm's effect relative to the final
    count and the run's own reference EAL, which equals it at the setup's
    weights, as a cross-check.
    """
    record = next(
        (
            row.acquisition["frequency_estimate"]
            for row in package.evidence.records
            if row.acquisition and "frequency_estimate" in row.acquisition
        ),
        None,
    )
    if record is None:
        return None
    views = {}
    for view, values in record["views"].items():
        by_arm = values["by_arm"]
        final = by_arm["final"].get("expected_annual_loss")
        run = metrics.get(view, {})
        views[view] = {
            **values,
            "run_expected_annual_loss": run.get("expected_annual_loss")
            if run.get("status") == "available"
            else None,
            "relative_effect": {
                arm: (row["expected_annual_loss"] - final) / final
                if row["status"] == "available" and final
                else None
                for arm, row in by_arm.items()
                if arm != "final"
            },
        }
    return {**record, "views": views}


def _large_loss_anchor(package):
    return next(
        (
            row.acquisition["large_loss_anchor"]
            for row in package.evidence.records
            if row.acquisition and "large_loss_anchor" in row.acquisition
        ),
        None,
    )


def _excluded_samples(package):
    return next(
        (
            row.acquisition["excluded_samples"]
            for row in package.evidence.records
            if row.acquisition and "excluded_samples" in row.acquisition
        ),
        None,
    )


def sensitivity_results(
    package, report, draws, weights, *, levels, reweighted, sampling, view_weights=None
):
    """Summarize EAL effects and the reference capture's dependence references.

    The dependence references use the summary's selected view whenever that
    view is available, including when its mean does not exist; then only VaR
    is reported. ``reweighted`` marks target weights that differ from the
    setup's own; the trajectory quadrature, frequency-estimate, hierarchy and
    large-loss anchor exhibits, which rest on the setup's weights, are then
    unavailable.
    ``sampling`` is the ``respondent_sampling`` exhibit, whose standard
    errors each tornado row carries. ``view_weights`` gives the reference
    arm's weights in each view where some view's are not ``weights``
    (``ViewWeights``): the dependence references mix the states at their
    weights in the selected view.
    """
    summary = sensitivity_summary(
        report,
        reference_arm=package.execution.reference_arm,
        purposes={row.alternative_id: row.purpose for row in package.alternatives},
    )
    view = summary["tornado"]["view"]
    summary["respondent_sampling"] = sampling
    for row in summary["tornado"]["rows"]:
        row.update(
            row_sampling(
                sampling,
                row["arm"],
                row["control_arm"],
                view,
                row["absolute_difference"],
            )
        )
    metrics = report["arms"][package.execution.reference_arm].get("metrics", {})
    scenario_views = {
        population.scenario_id: scenario_view(population.scenario_id, view)
        for population in package.scope.populations
    }
    reason = None
    by_view = ViewWeights.same(weights) if view_weights is None else view_weights
    mixed = None if view is None else by_view.of(view)
    if view not in {"gross", "welfare", "net_resource_cost"}:
        reason = "No available additive economic view was selected."
    elif mixed is None:
        reason = by_view.reason(view)
    elif not scenario_views or any(
        metrics.get(name, {}).get("status") != "available"
        for name in scenario_views.values()
    ):
        reason = "The reference capture lacks an available view for every scenario; no partial portfolio dependence reference is substituted."
    else:
        selected_views = (view, *scenario_views.values())
        mixture = CapabilityMixture(
            {
                state: replace(
                    draw, views={name: draw.views[name] for name in selected_views}
                )
                for state, draw in draws.items()
                if mixed[state] > 0
            },
            mixed,
        )
        for draw, _ in mixture.positive_strata():
            total = sum(
                (draw.annual_losses[name] for name in scenario_views.values()),
                start=np.zeros(draw.n_simulations),
            )
            if not np.allclose(total, draw.annual_losses[view], rtol=1e-9, atol=0):
                reason = (
                    "Scenario annual losses do not sum to the selected portfolio view."
                )
                break
    quadrature = _trajectory_quadrature(package, draws, summary["tornado"]["view"])
    if quadrature is not None:
        summary["trajectory_quadrature"] = (
            {
                key: quadrature[key]
                for key in ("rule", "interior_nodes", "embedded_interior_nodes")
            }
            | {"status": "unavailable", "reason": REWEIGHTED_EXHIBIT_REASON}
            if reweighted
            else quadrature
        )
    frequency = _frequency_estimate(package, metrics)
    if frequency is not None and reweighted:
        summary["frequency_estimate"] = {
            "status": "unavailable",
            "reason": REWEIGHTED_EXHIBIT_REASON,
        }
        if frequency.get("hierarchy"):
            summary["hierarchy_pool"] = dict(summary["frequency_estimate"])
    elif frequency is not None:
        summary["frequency_estimate"] = frequency
        if frequency.get("hierarchy"):
            summary["hierarchy_pool"] = frequency["hierarchy"]
    anchor = _large_loss_anchor(package)
    if anchor is not None:
        summary["large_loss_anchor"] = (
            {"status": "unavailable", "reason": REWEIGHTED_EXHIBIT_REASON}
            if reweighted
            else anchor
        )
    excluded = _excluded_samples(package)
    if excluded is not None:
        # Which samples a setup holds no reweighting can change; the weights
        # it reports are the setup's own.
        summary["excluded_samples"] = excluded
    summary["plausibility"] = plausibility_table(package, metrics)
    summary["dependence_bracket"] = (
        {"status": "unavailable", "reason": reason}
        if reason is not None
        else {
            "status": "available",
            "view": view,
            **dependence_bracket(
                mixture,
                scenario_views=scenario_views,
                levels=levels,
                controls=TailGateControls.from_controls(
                    package.execution.for_arm(
                        package.execution.reference_arm
                    ).numerical_controls.get("tail_gate")
                ),
            ),
        }
    )
    return summary
