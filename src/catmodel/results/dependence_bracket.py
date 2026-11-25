"""Comonotone dependence reference from stored scenario annual-loss marginals.

Within each stored capability and respondent state the comonotone reference
gives every scenario the same annual-loss rank, zeros included, so occurrence,
severity and the per-year parameter ranks move together, while each scenario
keeps its weighted empirical law, and so its EAL. For those marginals it is the
convex-order (TVaR) upper bound, not a VaR bound (methodology.md). Portfolio
quantiles are taken after summing aligned losses.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from catmodel.simulation.capability_mixture import (
    CapabilityMixture,
    summed_view,
    tail_estimates,
    tail_parts,
)
from catmodel.simulation.tail_gate import (
    TailGateControls,
    batch_support_reason,
    gate_status,
    minimum_batch_support,
    tvar_status,
)


def dependence_bracket(
    mixture: CapabilityMixture,
    *,
    scenario_views: Mapping[str, str],
    levels: tuple[float, ...] = (0.99, 0.995),
    controls: TailGateControls | None = None,
) -> dict[str, object]:
    """Compare the executed and within-state comonotone scenario loss sums.

    No simulation or elicitation is rerun. The executed reference is the
    engine's own coupling of the stored draws. The comonotone reference pairs
    each scenario's k-th smallest year within each state, exactly, since a
    state's years carry equal mass. Both keep the original state masses and
    the reference's native EAL where available.
    """
    if not scenario_views or len(set(scenario_views.values())) != len(scenario_views):
        raise ValueError("scenario views must be nonempty and distinct")
    if (
        not levels
        or len(set(levels)) != len(levels)
        or any(
            isinstance(level, bool) or not math.isfinite(level) or not 0 < level < 1
            for level in levels
        )
    ):
        raise ValueError("tail levels must be distinct and strictly inside (0, 1)")
    views = tuple(scenario_views[key] for key in sorted(scenario_views))
    strata = mixture.positive_strata()
    for draw, _ in strata:
        if not set(views) <= draw.views.keys():
            raise ValueError("every state must expose each selected scenario view")

    mean_exists = all(
        draw.views[view].mean_exists for draw, _ in strata for view in views
    )
    native_available = all(
        draw.views[view].native_mean is not None for draw, _ in strata for view in views
    )
    executed_losses = np.concatenate(
        [
            sum(
                (draw.annual_losses[view] for view in views),
                start=np.zeros(draw.n_simulations),
            )
            for draw, _ in strata
        ]
    )
    executed_weights = np.concatenate(
        [mass * draw.normalized_weights() for draw, mass in strata]
    )
    empirical_eal = float(np.dot(executed_losses, executed_weights))
    eal = (
        (
            math.fsum(
                mass * native.value
                for draw, mass in strata
                for view in views
                if (native := draw.views[view].native_mean) is not None
            )
            if native_available
            else empirical_eal
        )
        if mean_exists
        else None
    )

    # Recoupling keeps every marginal, so the analytic mean of the sum and
    # every coupling's empirical mean are common to all references. Each reference
    # takes the executed sum's control-variate TVaR coefficient, so the control
    # term is the same in each and never changes their gaps.
    estimates = None
    if mean_exists and np.all(np.isfinite(executed_losses)):
        executed = CapabilityMixture(
            {
                draw.stratum_sha256: replace(
                    draw, views={"sum": summed_view(draw, views)}
                )
                for draw, _ in strata
            },
            {draw.stratum_sha256: mass for draw, mass in strata},
        )
        estimates = tail_estimates(executed, "sum", levels)

    def tvar(level, parts):
        if estimates is None:
            return None
        estimate = estimates[level]
        return estimate.tvar_at(parts, estimate.coefficient)

    def summarize(losses, weights):
        if not np.all(np.isfinite(losses)):
            return {
                "status": "unavailable",
                "reason": "The recoupled annual-loss sum exceeds finite numeric range.",
            }
        tails = {}
        for level in levels:
            parts = tail_parts(losses, weights, level)
            tails[str(level)] = {"var": parts.var, "tvar": tvar(level, parts)}
        return {
            "status": "available",
            "expected_annual_loss": eal,
            "normalized_empirical_mean": float(np.dot(losses, weights)),
            "tail": tails,
            "mean_changing": False,
        }

    controls = controls or TailGateControls()
    enough_years = all(draw.n_simulations >= controls.batches for draw, _ in strata)

    def batch(k, comonotone):
        """Batch ``k``'s years in every state, summed as executed or sorted."""
        losses, weights = [], []
        for draw, mass in strata:
            n = draw.n_simulations
            selected = np.arange(n) * controls.batches // n == k
            parts = [draw.annual_losses[view][selected] for view in views]
            if comonotone:
                parts = [np.sort(part) for part in parts]
            total = np.sum(parts, axis=0)
            losses.append(total)
            weights.append(np.full(len(total), mass / len(total)))
        return np.concatenate(losses), np.concatenate(weights)

    def gate(reference, comonotone):
        """The tail gate's half-widths of a reference's VaR and TVaR; an
        unconverged one is withheld with its half-width."""
        if reference["status"] != "available":
            return reference
        if not enough_years:
            reason = f"The tail gate needs at least {controls.batches} years per state."
            for tail in reference["tail"].values():
                tail.update(status="withheld", reason=reason)
            return reference
        samples = [batch(k, comonotone) for k in range(controls.batches)]
        for level in levels:
            tail = reference["tail"][str(level)]
            parts = [tail_parts(x, w, level) for x, w in samples]
            rows = {
                "var": gate_status(
                    tail["var"], [part.var for part in parts], controls.tolerance
                )
            }
            if estimates is not None:
                # As for any view, the reported estimator alone, its
                # coefficient fixed on all years, with the executed sum's
                # integration margin.
                rows["tvar"] = tvar_status(
                    estimates[level],
                    tail["tvar"],
                    [tvar(level, part) for part in parts],
                    controls.tolerance,
                )
            tail["half_width"] = {key: row["half_width"] for key, row in rows.items()}
            # The same floor as for any view: enough years at or above the VaR
            # in every batch.
            tail["minimum_batch_exceedances"] = minimum = minimum_batch_support(
                samples, tail["var"], "var"
            )
            support = batch_support_reason(minimum, controls.batches, "var")
            unconverged = [
                key for key, row in rows.items() if row["status"] != "converged"
            ]
            tail["status"] = "withheld" if unconverged or support else "available"
            tail["reason"] = (
                f"Unconverged: the 95% half-width of its {' and '.join(unconverged)} exceeds {100 * controls.tolerance:g}% of the value."
                if unconverged
                else f"Unconverged: {support}."
                if support
                else None
            )
        return reference

    references = {
        "executed": {
            **summarize(executed_losses, executed_weights),
            "coupling": "As simulated: origins independent within each state apart from any explicitly shared origin.",
        }
    }
    comonotone_losses, comonotone_weights = [], []
    for draw, state_mass in strata:
        # Equal year weights: the comonotone sum pairs each view's k-th
        # smallest year, exactly and in O(Y log Y). A sum beyond the numeric
        # range makes the reference unavailable (``summarize``).
        with np.errstate(over="ignore"):
            comonotone_losses.append(
                np.sum([np.sort(draw.annual_losses[view]) for view in views], axis=0)
            )
        comonotone_weights.append(state_mass * draw.normalized_weights())
    gate(references["executed"], comonotone=False)
    references["comonotone"] = {
        **summarize(
            np.concatenate(comonotone_losses), np.concatenate(comonotone_weights)
        ),
        "coupling": "One common annual-loss rank across scenarios within each state; the convex-order upper bound for the stored conditional marginals.",
    }
    gate(references["comonotone"], comonotone=True)
    effective_size = float(
        np.sum(executed_weights) ** 2 / np.sum(np.square(executed_weights))
    )
    return {
        "conditioning": "Couplings change within each stored capability/respondent state; original state masses are retained.",
        "scenario_views": dict(scenario_views),
        "references": references,
        "eal_estimator": "reference_native_mean"
        if native_available
        else "reference_weighted_empirical_mean",
        "tvar_estimators": None
        if estimates is None
        else {str(level): estimate.estimator for level, estimate in estimates.items()},
        "mean_unavailable_reason": None
        if mean_exists
        else "At least one selected scenario conditional law has no finite mean; EAL and TVaR are withheld.",
        "precision": {
            "effective_sample_size": effective_size,
            "effective_tail_observations": {
                str(level): effective_size * (1 - level) for level in levels
            },
            "interpretation": "Each reference's VaR and TVaR carry the tail gate's half-width over the same batches of years, and an unconverged one is withheld. The Kish effective sample size of the weights times one minus the level approximates the effective draws beyond each level; values beyond the captured support are not estimated.",
        },
        "limitations": "The comonotone reference is the convex-order (TVaR) upper bound within each state for the stored marginals, not a VaR bound or an empirical correlation. Tail results describe stored empirical support and exclude forecast or elicitation accuracy.",
    }
