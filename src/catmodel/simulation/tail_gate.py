"""Tail convergence gate: batch means over the simulated years.

Each positive stratum's independent years are split into ``K`` contiguous
batches, the same fraction of every stratum in each. A year's
streams depend only on the seed, the state and the year, so batch ``k`` holds
the same years in every state, and the batches are independent replicates of
the whole mixture estimator whether or not states share year-level draws. A
statistic's 95% half-width is ``t_{K-1, 0.975} sd(theta_k) / sqrt(K)``
(sectioning; Asmussen and Glynn, 2007, *Stochastic Simulation*, III.5);
``NUM-TAIL-GATE`` states when it converges and how execution doubles years.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
from scipy.stats import t as student_t

from catmodel.contracts.incident import ExecutionPlan, check_tail_gate_controls
from catmodel.probability.numerics import TIE_RELATIVE_TOLERANCE
from catmodel.probability.percentiles import SortedSample
from catmodel.simulation.capability_mixture import (
    NO_EXACT_MEAN_COEFFICIENT,
    AnalyticMean,
    CapabilityMixture,
    ConditionalCapabilityDraws,
    TailMeanEstimate,
    TailParts,
    analytic_mixture_mean,
    design_precision_unavailable_reason,
    eal_tail_coherence,
    known_zero_law,
    native_mixture_mean,
    tail_estimates,
    tail_parts,
    useful_support_unavailable_reason,
    weighted_var_tvar,
)

# A gated statistic needs this many positive-weight supporting years in every
# batch (``minimum_batch_support``): a judgment (range 5/10/20) matching the
# n*p >= 10 rule for normal approximations to a binomial count.
MIN_TAIL_EFFECTIVE_UNITS = 10
# The most batch-means standard errors by which the simulated years may
# differ from a view's analytic EAL before its VaR and TVaR are withheld
# (``eal_reconciliation``): a judgment (range 3/5/8).
EAL_RECONCILIATION_LIMIT = 5
HEADLINE_VIEWS = ("welfare", "gross")
INCOHERENT_REASON = "Unconverged: the analytic EAL exceeds the largest simulated annual loss or a reported TVaR, so the simulated years miss upper parameter ranks or severities that carry the mean."
# Why a statistic is unconverged at any number of years. It is withheld, but
# it never makes its view a failing driver (``failing_views``): doubling an
# arm's years cannot converge it.
SAMPLE_TVAR_MOMENT_REASON = "A sample-tail-mean TVaR has a batch-means half-width only with a finite second moment of the transformed conditional loss law."
INTEGRATION_MARGIN_REASON = "The certified integration error bound of the analytic EAL alone can move this control-variate TVaR by more than the tail gate's tolerance, and more simulated years do not reduce it."
YEAR_INDEPENDENT_REASONS = frozenset(
    {SAMPLE_TVAR_MOMENT_REASON, INTEGRATION_MARGIN_REASON}
)


@dataclass(frozen=True)
class TailGateControls:
    """The frozen gate: ``tolerance`` and ``batches`` are judgments.

    ``max_reference_state_years`` caps the reference arm's simulated
    state-years, its states times its years per state, and
    ``max_comparison_state_years`` each comparison arm's; ``None`` leaves an
    arm at its planned years. ``views`` at ``levels`` drive the doubling, and
    a plan names only views it requests; by default the welfare, gross and
    lab-loss views at every reported level. Every reported view and level is
    gated.
    """

    tolerance: float = 0.1
    batches: int = 16
    max_reference_state_years: int | None = None
    max_comparison_state_years: int | None = None
    views: tuple[str, ...] | None = None
    levels: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        check_tail_gate_controls(vars(self))

    @classmethod
    def from_controls(cls, value: object) -> TailGateControls:
        check_tail_gate_controls(value)
        if value is None:
            return cls()
        settings = cast(Mapping[str, Any], value)
        views, levels = settings.get("views"), settings.get("levels")
        return cls(
            tolerance=settings.get("tolerance", cls.tolerance),
            batches=settings.get("batches", cls.batches),
            max_reference_state_years=settings.get("max_reference_state_years"),
            max_comparison_state_years=settings.get("max_comparison_state_years"),
            views=None if views is None else tuple(views),
            levels=None if levels is None else tuple(levels),
        )

    def drivers(self, requested: Sequence[str]) -> tuple[str, ...]:
        if self.views is not None:
            return tuple(view for view in self.views if view in requested)
        return tuple(
            view
            for view in requested
            if view in HEADLINE_VIEWS or view.startswith("lab_loss::")
        )

    def max_doublings(self, *, reference: bool, states: int, years: int) -> int:
        """How many times an arm may double its ``years`` per state.

        The arm's cap bounds its simulated state-years, ``states`` times its
        years per state, so its largest years per state is the most of
        ``years`` times a power of two that keeps them within the cap. Raises
        when its planned years alone exceed the cap.
        """
        cap = (
            self.max_reference_state_years
            if reference
            else self.max_comparison_state_years
        )
        if cap is None:
            return 0
        if states * years > cap:
            role = "reference" if reference else "comparison"
            raise ValueError(
                f"a {role} arm's {states} states at {years} planned years per state exceed its tail gate cap of {cap} state-years"
            )
        return (cap // (states * years)).bit_length() - 1

    def record_controls(self) -> dict[str, Any]:
        """The frozen plan control, which ``from_controls`` reads back."""
        controls: dict[str, Any] = {
            "tolerance": self.tolerance,
            "batches": self.batches,
            "max_reference_state_years": self.max_reference_state_years,
            "max_comparison_state_years": self.max_comparison_state_years,
        }
        if self.views is not None:
            controls["views"] = list(self.views)
        if self.levels is not None:
            controls["levels"] = list(self.levels)
        return controls

    def record(self) -> dict[str, Any]:
        return {
            "tolerance": self.tolerance,
            "batches": self.batches,
            "max_reference_state_years": self.max_reference_state_years,
            "max_comparison_state_years": self.max_comparison_state_years,
            "views": None if self.views is None else list(self.views),
            "levels": None if self.levels is None else list(self.levels),
            "minimum_batch_exceedances": MIN_TAIL_EFFECTIVE_UNITS,
        }


def tail_gate_controls(
    max_reference_state_years: int | None, max_comparison_state_years: int | None
) -> dict[str, Any]:
    """The default gate an import freezes, with its arms' state-year caps."""
    return TailGateControls(
        max_reference_state_years=max_reference_state_years,
        max_comparison_state_years=max_comparison_state_years,
    ).record_controls()


def batch_layout(
    strata: Sequence[tuple[ConditionalCapabilityDraws, float]], batches: int
) -> tuple[np.ndarray, np.ndarray] | str:
    """Batch and stratum ordinal of every year of ``strata``, concatenated.

    Returns a reason when a stratum has too few years.
    """
    batch, stratum = [], []
    for ordinal, (draw, _) in enumerate(strata):
        count = draw.n_simulations
        if count < batches:
            return f"A stratum has {count} independent years; the {batches}-batch tail gate needs at least {batches}."
        batch.append(np.arange(count) * batches // count)
        stratum.append(np.full(count, ordinal))
    if not batch:
        return "No stratum with possible loss has positive weight."
    return np.concatenate(batch), np.concatenate(stratum)


def batch_masses(
    masses: np.ndarray, layout: tuple[np.ndarray, np.ndarray], batches: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each batch's sample selector and masses, renormalized within each stratum."""
    batch, stratum = layout
    strata = int(stratum.max()) + 1
    target = np.bincount(stratum, weights=masses, minlength=strata)
    result = []
    for k in range(batches):
        selected = batch == k
        within = masses[selected]
        totals = np.bincount(stratum[selected], weights=within, minlength=strata)
        scale = np.divide(target, totals, out=np.zeros(strata), where=totals > 0)
        result.append((selected, within * scale[stratum[selected]]))
    return result


def _batched(
    mixture: CapabilityMixture, view: str, batches: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]] | str:
    """The strata of ``view`` that can carry a loss, batched: their years and
    masses concatenated and each batch's selector and masses
    (``batch_masses``), or why they cannot be batched. Computed once per
    mixture, view and batch count, so every gated statistic shares it."""

    def compute():
        active = mixture.positive_strata()
        strata = [
            (draw, mass) for draw, mass in active if not known_zero_law(draw, view)
        ]
        layout = design_precision_unavailable_reason(
            (draw for draw, _ in active), view=view
        ) or useful_support_unavailable_reason(mixture, view)
        if layout is None:
            layout = batch_layout(strata, batches)
        if isinstance(layout, str):
            return layout
        losses = np.concatenate([draw.annual_losses[view] for draw, _ in strata])
        masses = np.concatenate(
            [mass * draw.normalized_weights() for draw, mass in strata]
        )
        parts = batch_masses(masses, layout, batches)
        # Read-only, so the shared batching cannot be changed under its users.
        for array in (losses, masses, *(value for part in parts for value in part)):
            array.flags.writeable = False
        return losses, masses, parts

    return mixture.derived(("batched", view, batches), compute)


def minimum_batch_support(
    batches: Iterable[tuple[np.ndarray, np.ndarray]],
    threshold: float,
    functional: str,
) -> int:
    """The sparsest batch's positive-weight years that support a statistic.

    ``batches`` holds each batch's simulated years and masses; an exact atom,
    such as the structurally zero states' mass, carries no sampling error and
    is left out. A VaR, and the TVaR beyond it, rest on the years at or above
    the VaR ``threshold``: TVaR averages the worst ``1 - level`` mass, every
    year above the VaR and, with the fractional atom, a share of the years at
    it, so a year at an atom VaR is a tail year, and a VaR at the top of the
    support, such as a policy limit or the largest possible occurrence, has
    support. An exceedance probability (``functional == "probability"``) rests
    on the years in its event, those strictly above ``threshold``.
    """
    counts = []
    for losses, masses in batches:
        tail = (
            losses > threshold if functional == "probability" else losses >= threshold
        )
        counts.append(int(np.count_nonzero(tail & (masses > 0))))
    return min(counts)


def batch_support_reason(
    minimum: int | None, batches: int, functional: str
) -> str | None:
    """Why the sparsest batch's support (``minimum_batch_support``) is below
    ``MIN_TAIL_EFFECTIVE_UNITS``, or None when it meets it or none is needed."""
    if minimum is None or minimum >= MIN_TAIL_EFFECTIVE_UNITS:
        return None
    years = "with a loss" if functional == "probability" else "at or above the VaR"
    return f"the sparsest of {batches} batches has {minimum} years {years}, and a converged statistic needs at least {MIN_TAIL_EFFECTIVE_UNITS} in each"


def batch_half_width(values: Sequence[float]) -> float:
    """The half-width of the 95% Student-t interval for the mean of batch
    values."""
    count = len(values)
    # Scale before squaring so large dollar statistics cannot overflow.
    scale = float(np.max(np.abs(values)))
    spread = (
        0.0 if scale == 0 else scale * float(np.std(np.divide(values, scale), ddof=1))
    )
    return float(student_t.ppf(0.975, count - 1)) * spread / math.sqrt(count)


def _status(value: float, half_width: float, tolerance: float) -> dict[str, Any]:
    relative = (
        half_width / abs(value) if value != 0 else (0.0 if half_width == 0 else None)
    )
    return {
        "half_width": half_width,
        "relative_half_width": relative,
        "status": "converged"
        if relative is not None and relative <= tolerance
        else "unconverged",
    }


def tvar_status(
    estimate: TailMeanEstimate,
    value: float,
    batch_values: Sequence[float],
    tolerance: float,
) -> dict[str, Any]:
    """Half-width and convergence of a control-variate TVaR ``value`` from
    its batch values: ``estimate``'s estimator on each batch.

    The half-width is the batch-means half-width plus the estimate's
    ``integration_margin``, the most its analytic mean's certified integration
    error bound can move the TVaR (zero for a closed-form mean), so the
    interval allows for both errors. More years shrink only the batch-means
    part: a TVaR whose margin alone exceeds the tolerance is unconverged at
    any number of years (``INTEGRATION_MARGIN_REASON``).
    """
    margin = estimate.integration_margin
    status = _status(value, batch_half_width(batch_values) + margin, tolerance)
    if margin > tolerance * abs(value):
        status["reason"] = INTEGRATION_MARGIN_REASON
    return status


def eal_reconciliation(
    analytic: AnalyticMean,
    simulated: float,
    batch_values: Sequence[float],
    *,
    limit: tuple[str, float] | None = None,
) -> dict[str, Any]:
    """Whether the simulated years agree with a view's analytic EAL.

    ``simulated`` is the years' mean, or with ``limit``, a level and its VaR,
    their mean loss limited at that VaR, and ``batch_values`` the same mean
    on each batch, whose spread gives its standard error. ``z`` is
    ``simulated`` less the EAL over that standard error, null when the batch
    values do not vary. ``analytic`` is the EAL and its certified integration
    error bound (``analytic_mixture_mean``): a mean is inconsistent when it
    lies more than ``EAL_RECONCILIATION_LIMIT`` standard errors outside the
    EAL widened by that bound and by a computed tie
    (``TIE_RELATIVE_TOLERANCE``), so ``|z|`` above the limit for a closed-form
    EAL. A limited mean is at most the mean for every law, so it is
    inconsistent only above that interval.
    """
    eal, bound = analytic.value, analytic.error_bound
    error = batch_half_width(batch_values) / float(
        student_t.ppf(0.975, len(batch_values) - 1)
    )
    difference = simulated - eal
    outside = (
        abs(difference) - bound - TIE_RELATIVE_TOLERANCE * max(eal, abs(simulated))
    )
    inconsistent = outside > EAL_RECONCILIATION_LIMIT * error and (
        limit is None or difference > 0
    )
    return {
        "statistic": "mean" if limit is None else "limited_mean",
        "limit_level": None if limit is None else limit[0],
        "limit": None if limit is None else limit[1],
        "simulated_mean": simulated,
        "simulated_mean_standard_error": error,
        "expected_annual_loss": eal,
        "integration_error_bound": bound,
        "z": difference / error if error > 0 else None,
        "status": "inconsistent" if inconsistent else "consistent",
    }


def reconciliation_reason(record: Mapping[str, Any]) -> str:
    """Why an inconsistent ``eal_reconciliation`` withholds a view's VaR and
    TVaR."""
    z = record["z"]
    limited = record["statistic"] == "limited_mean"
    beyond = ("+" if limited else "±") + str(EAL_RECONCILIATION_LIMIT)
    if record["integration_error_bound"]:
        beyond += " and the EAL's certified integration error bound"
    gap = (
        "although its batch means do not vary"
        if z is None
        else f"by z = {z:+.1f} batch-means standard errors, beyond {beyond}"
    )
    if limited:
        return f"Unconverged: the simulated years' mean loss limited at their VaR at level {record['limit_level']} exceeds the analytic EAL {gap}, yet no law's limited mean exceeds its mean: the simulation and the analytic EAL disagree on the law."
    cause = (
        "the years miss upper parameter ranks or severities that carry the mean, or the simulation and the analytic EAL disagree on the law"
        if z is not None and z < 0
        else "the simulation and the analytic EAL disagree on the law"
    )
    return f"Unconverged: the simulated years' mean differs from the analytic EAL {gap}: {cause}."


def view_gate(
    mixture: CapabilityMixture,
    view: str,
    estimates: Mapping[float, TailMeanEstimate],
    controls: TailGateControls,
    *,
    mean_exists: bool,
) -> dict[str, Any]:
    """Batch-means half-widths and convergence of one view's tail statistics.

    Structurally zero strata are an exact zero atom in every batch. Each
    batch's TVaR is the reported control-variate estimate with the coefficient
    set on all years, and its ``tail_parts`` are kept under ``tvar_parts`` so
    that an arm effect can apply one coefficient to both arms; its half-width
    adds the margin for a certified analytic mean's integration error
    (``tvar_status``). A view with an analytic EAL
    (``analytic_mixture_mean``) reconciles the years with it
    (``eal_reconciliation``): its mean with a finite second moment, and
    otherwise its mean limited at the VaR of the highest level; no tail
    statistic has converged while they are inconsistent. The batch means of
    the EAL are kept only where the reported EAL is the years' mean, so an
    arm effect on a native EAL, which has no sampling error, is not gated on
    them.
    """
    active = mixture.positive_strata()
    strata = [(draw, mass) for draw, mass in active if not known_zero_law(draw, view)]
    years = max(draw.n_simulations for draw, _ in active)
    base: dict[str, Any] = {"batches": controls.batches, "years_per_state": years}
    if not strata:
        exact_zero = _status(0.0, 0.0, controls.tolerance)
        return {
            **base,
            "unavailable_reason": None,
            "empirical_mean_standard_error": 0.0,
            "eal_reconciliation": None,
            "levels": {
                str(level): {
                    "var": dict(exact_zero),
                    "tvar": dict(exact_zero) if mean_exists else None,
                    "minimum_batch_exceedances": None,
                }
                for level in estimates
            },
            "probability_positive": {
                **exact_zero,
                "minimum_batch_exceedances": None,
            },
            "batch_values": {
                key: [0.0] * controls.batches
                for key in (
                    "annual_probability_positive",
                    "annual_loss_median",
                    *(
                        f"{name}::{level}"
                        for level in estimates
                        for name in ("var", "tvar")
                    ),
                )
            },
            "tvar_parts": {
                str(level): (TailParts(0.0, 0.0, 0.0),) * controls.batches
                for level in estimates
            }
            if mean_exists
            else {},
        }
    batched = _batched(mixture, view, controls.batches)
    if isinstance(batched, str):
        return {**base, "unavailable_reason": batched, "levels": {}}
    losses, masses, batches = batched
    analytic = analytic_mixture_mean(active, (view,)) if mean_exists else None
    finite_variance = all(draw.views[view].variance_exists for draw, _ in active)
    # A level inside the exact zero atom of structurally zero strata has VaR
    # exactly zero whatever the sample.
    positive_mass = math.fsum(mass for _, mass in strata)
    # The structurally zero strata's own masses, never a difference from one
    # that rounding could make negative.
    zero_mass = math.fsum(mass for draw, mass in active if known_zero_law(draw, view))
    # Each batch's simulated years, which alone support a statistic, and its
    # law, those years with the exact zero atom.
    simulated = [(losses[sel], m) for sel, m in batches]
    zero = np.array([0.0])
    samples = [
        (np.concatenate([values, zero]), np.append(m, zero_mass))
        for values, m in simulated
    ]
    # Each batch is sorted once for its median and every level's VaR.
    ordered = [SortedSample(values, m) for values, m in samples]
    means = [float(np.dot(values, m)) for values, m in samples]
    positive_k = [float(np.dot(values > 0, m)) for values, m in samples]
    median_k = [
        weighted_var_tvar(values, m, 0.5, ordered=order)[0]
        for (values, m), order in zip(samples, ordered, strict=True)
    ]
    probability = float(np.dot(losses > 0, masses))
    probability_row = _status(
        probability, batch_half_width(positive_k), controls.tolerance
    )
    positive_support = minimum_batch_support(simulated, 0.0, "probability")
    if (
        probability > 0
        and batch_support_reason(positive_support, controls.batches, "probability")
        is not None
    ):
        probability_row["status"] = "unconverged"
    probability_row["minimum_batch_exceedances"] = positive_support
    batch_values: dict[str, list[float]] = {
        "annual_probability_positive": positive_k,
        "annual_loss_median": median_k,
    }
    # The reported EAL is the years' mean only without a native mean in every
    # state (``CapabilityMixture.metrics``); only then do its batch means
    # measure the error of an arm effect on it.
    if mean_exists and native_mixture_mean(active, (view,)) is None:
        batch_values["expected_annual_loss"] = means
    levels: dict[str, Any] = {}
    tvar_parts: dict[str, tuple[TailParts, ...]] = {}
    for level, estimate in estimates.items():
        parts_k = tuple(
            tail_parts(values, masses, level, ordered=order)
            for (values, masses), order in zip(samples, ordered, strict=True)
        )
        var_k = [parts.var for parts in parts_k]
        batch_values[f"var::{level}"] = var_k
        # A level inside the exact zero atom of structurally zero strata has
        # VaR exactly zero and needs no support.
        minimum = (
            None
            if positive_mass <= 1 - level
            else minimum_batch_support(simulated, estimate.var, "var")
        )
        row: dict[str, Any] = {
            "var": _status(estimate.var, batch_half_width(var_k), controls.tolerance),
            "tvar": None,
            "minimum_batch_exceedances": minimum,
        }
        if (
            mean_exists
            and estimate.coefficient_basis == NO_EXACT_MEAN_COEFFICIENT
            and not finite_variance
        ):
            # Without an analytic mean to control it, the excess of a law
            # with no finite second moment has no half-width at any years.
            row["tvar"] = {
                "half_width": None,
                "relative_half_width": None,
                "status": "unconverged",
                "reason": SAMPLE_TVAR_MOMENT_REASON,
            }
        elif mean_exists:
            # The reported estimator alone, its coefficient fixed on all years.
            tvar_k = [
                estimate.tvar_at(parts, estimate.coefficient) for parts in parts_k
            ]
            row["tvar"] = tvar_status(
                estimate, estimate.tvar, tvar_k, controls.tolerance
            )
            batch_values[f"tvar::{level}"] = tvar_k
            tvar_parts[str(level)] = parts_k
        if batch_support_reason(minimum, controls.batches, "var") is not None:
            for key in ("var", "tvar"):
                if row[key] is not None:
                    row[key]["status"] = "unconverged"
        levels[str(level)] = row
    # The years must agree with an analytic EAL within sampling error, or the
    # tail statistics, and a control-variate TVaR's correction by the EAL,
    # rest on years unlike the law the EAL integrates. Without a finite second
    # moment their mean has no standard error, but their mean loss limited at
    # the highest level's VaR has one and is at most the EAL.
    reconciliation = None
    if analytic is not None and (finite_variance or estimates):
        if finite_variance:
            reconciliation = eal_reconciliation(
                analytic, float(np.dot(losses, masses)), means
            )
        else:
            top = max(estimates)
            limit = estimates[top].var
            reconciliation = eal_reconciliation(
                analytic,
                float(np.dot(np.minimum(losses, limit), masses)),
                [float(np.dot(np.minimum(values, limit), m)) for values, m in samples],
                limit=(str(top), limit),
            )
        if reconciliation["status"] == "inconsistent":
            reason = reconciliation_reason(reconciliation)
            for row in levels.values():
                for key in ("var", "tvar"):
                    if row[key] is not None:
                        row[key] = {
                            **row[key],
                            "status": "unconverged",
                            "reason": reason,
                        }
    # No distribution's TVaR is below its mean and no mean exceeds its largest
    # value: an analytic EAL above either, by more than its certified bound,
    # means the years miss the upper ranks or severities that carry the mean,
    # so no tail statistic has converged.
    if (
        analytic is not None
        and eal_tail_coherence(
            analytic.value,
            losses,
            {
                str(level): {"tvar": estimate.tvar}
                for level, estimate in estimates.items()
            },
            error_bound=analytic.error_bound,
        )["status"]
        == "incoherent"
    ):
        for row in levels.values():
            for key in ("var", "tvar"):
                if row[key] is not None:
                    row[key] = {
                        **row[key],
                        "status": "unconverged",
                        "reason": INCOHERENT_REASON,
                    }
    return {
        **base,
        "unavailable_reason": None,
        "empirical_mean_standard_error": batch_half_width(means)
        / float(student_t.ppf(0.975, controls.batches - 1)),
        "eal_reconciliation": reconciliation,
        "probability_positive": probability_row,
        "levels": levels,
        "batch_values": batch_values,
        "tvar_parts": tvar_parts,
    }


def unconverged_reason(
    row: Mapping[str, Any], functional: str, controls: Mapping[str, Any], years: int
) -> str | None:
    """Why a gated statistic is withheld, or None when it converged."""
    statistic = row[functional]
    if statistic is None or statistic["status"] == "converged":
        return None
    if statistic.get("reason"):
        return statistic["reason"]
    support = batch_support_reason(
        row["minimum_batch_exceedances"], controls["batches"], functional
    )
    if support is not None:
        return f"Unconverged at {years} years per state: {support}."
    relative = statistic["relative_half_width"]
    width = "unbounded" if relative is None else f"±{100 * relative:.1f}%"
    return f"Unconverged at {years} years per state: its 95% batch-means half-width is {width} of the estimate, above the {100 * controls['tolerance']:g}% tolerance."


def failing_views(
    gates: Mapping[str, Mapping[str, Any]], views: Iterable[str]
) -> list[str]:
    """Driver views with an unavailable gate or a statistic that more years
    could converge.

    A statistic that is unconverged at any number of years
    (``YEAR_INDEPENDENT_REASONS``) is withheld, but it does not make its view
    fail, so it never drives doubling.
    """
    failing = []
    for view in views:
        gate = gates.get(view)
        if gate is None:
            continue
        if gate["unavailable_reason"] is not None or any(
            statistic is not None
            and statistic["status"] != "converged"
            and statistic.get("reason") not in YEAR_INDEPENDENT_REASONS
            for row in gate["levels"].values()
            for statistic in (row["var"], row["tvar"])
        ):
            failing.append(view)
    return failing


def unconverged_drivers(
    draws: Mapping[str, ConditionalCapabilityDraws],
    weights: Mapping[str, float],
    controls: TailGateControls,
    requested: Sequence[str],
    levels: Sequence[float],
    *,
    view_weights: Callable[[str], Mapping[str, float] | None] | None = None,
) -> list[str]:
    """Driver views of one arm whose tail statistics have not converged.

    A view missing from a positive-weight state is unavailable, not
    unconverged, a structurally zero view is exact, and a statistic that no
    number of years can converge does not count (``failing_views``).
    ``view_weights`` gives a view's weights in the arm where some view's are
    not ``weights``, as a lab-loss or gross view's are where a state is left
    out of the views that count the lab's loss, and None for a view the arm
    has no weights for, which is unavailable too. A driver is gated at its
    own weights.
    """
    failing = []
    for view in controls.drivers(requested):
        of = weights if view_weights is None else view_weights(view)
        if of is None or any(
            of.get(state, 0) > 0 and view not in draw.annual_losses
            for state, draw in draws.items()
        ):
            continue
        mixture = CapabilityMixture(
            {
                state: replace(draw, views={view: draw.views[view]})
                for state, draw in draws.items()
                if view in draw.annual_losses
            },
            of,
        )
        active = mixture.positive_strata()
        if all(known_zero_law(draw, view) for draw, _ in active):
            continue
        mean_exists = all(draw.views[view].mean_exists for draw, _ in active)
        gate = view_gate(
            mixture,
            view,
            tail_estimates(mixture, view, controls.levels or levels),
            controls,
            mean_exists=mean_exists,
        )
        failing.extend(failing_views({view: gate}, (view,)))
    return failing


def doubling_limits(
    plan: ExecutionPlan, states: Mapping[str, int], followers: Iterable[str] = ()
) -> dict[str, int]:
    """Each required arm's most doublings within its state-year cap.

    ``states`` counts each arm's simulated states. An arm in ``followers``
    takes the reference arm's years, so it has the reference arm's limit and
    no cap of its own.
    """
    followers = set(followers)
    limits = {}
    for arm in plan.required_arms:
        if arm in followers:
            continue
        resolved = plan.for_arm(arm)
        limits[arm] = TailGateControls.from_controls(
            resolved.numerical_controls.get("tail_gate")
        ).max_doublings(
            reference=arm == plan.reference_arm,
            states=states[arm],
            years=resolved.simulations,
        )
    for arm in followers:
        limits[arm] = limits[plan.reference_arm]
    return limits


def round_arms(rounds: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Each deciding arm's last round: an arm is evaluated in every round from
    the first until it stops, and doubles once after every round but its last.

    Raises when the rounds are not consecutive or an arm skips a round.
    """
    if not rounds or [row.get("doubling") for row in rounds] != list(
        range(len(rounds))
    ):
        raise ValueError(
            "capture tail gate rounds differ from the frozen numerical controls"
        )
    last: dict[str, int] = {}
    for index, row in enumerate(rounds):
        arms = set(cast(Mapping[str, Any], row["unconverged_views"]))
        if index and not arms:
            raise ValueError("capture tail gate round evaluates no arm")
        if index and not arms <= {arm for arm, end in last.items() if end == index - 1}:
            raise ValueError(
                "capture tail gate rounds evaluate an arm after it stopped"
            )
        for arm in arms:
            last[arm] = index
    return last


def gated_years(
    plan: ExecutionPlan,
    rounds: Sequence[Mapping[str, Any]],
    states: Mapping[str, int],
    followers: Iterable[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Each required arm's final and largest years per state.

    A deciding arm runs its planned years doubled once after every round but
    its last; a follower takes the reference arm's years. Raises when the
    rounds do not evaluate exactly the deciding arms first, or run an arm past
    its cap.
    """
    followers = set(followers)
    limits = doubling_limits(plan, states, followers)
    last = round_arms(rounds)
    deciding = set(plan.required_arms) - followers
    if set(rounds[0]["unconverged_views"]) != deciding or any(
        last[arm] > limits[arm] for arm in deciding
    ):
        raise ValueError(
            "capture tail gate rounds differ from the frozen numerical controls"
        )
    reference = plan.reference_arm
    final, largest = {}, {}
    for arm in plan.required_arms:
        source = reference if arm in followers else arm
        planned = plan.for_arm(source).simulations
        final[arm] = planned * 2 ** last[source]
        largest[arm] = planned * 2 ** limits[source]
    return final, largest


def year_prefix(
    draw: ConditionalCapabilityDraws, years: int
) -> ConditionalCapabilityDraws:
    """A draw's first ``years`` years, as a shorter simulation drew them."""
    if years > draw.n_simulations:
        raise ValueError("a year prefix cannot exceed the simulated years")
    if years == draw.n_simulations:
        return draw
    return replace(
        draw,
        views={
            name: replace(view, array=view.array[:years])
            for name, view in draw.views.items()
        },
    )


def tail_gate_results(
    plan: ExecutionPlan,
    rounds: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    states: Mapping[str, int],
    followers: Iterable[str],
) -> dict[str, Any]:
    """The run's doubling rounds and each arm's controls, years and withheld
    statistics.

    ``states`` counts each arm's simulated states, which set its cap on years
    per state (``doubling_limits``); a follower takes the reference arm's
    years and does not decide doubling.
    """
    followers = set(followers)
    final, largest = gated_years(plan, rounds, states, followers)
    last = round_arms(rounds)
    by_arm = {}
    for arm in plan.required_arms:
        resolved = plan.for_arm(arm)
        controls = TailGateControls.from_controls(
            resolved.numerical_controls.get("tail_gate")
        )
        metrics = report["arms"].get(arm, {}).get("metrics", {})
        withheld = sorted(
            f"{view}::{key}"
            for view, metric in metrics.items()
            if metric.get("status") == "available"
            for key, status in metric["statistics"].items()
            if status["status"] == "withheld"
            and str(status["reason"]).startswith(
                ("Unconverged", "The tail gate is unavailable")
            )
        )
        follows = arm in followers
        by_arm[arm] = {
            "controls": controls.record(),
            "driver_views": list(controls.drivers(resolved.requested_views)),
            "follows_arm": plan.reference_arm if follows else None,
            "states": states[arm],
            "years_per_state": final[arm],
            "max_years_per_state": largest[arm],
            "drivers_converged": None
            if follows
            else not rounds[last[arm]]["unconverged_views"][arm],
            "withheld_statistics": withheld,
        }
    return {
        "method": "batch_means_sectioning",
        "doublings": len(rounds) - 1,
        "rounds": [dict(row) for row in rounds],
        "by_arm": by_arm,
    }


def paired_effect(
    reference: Sequence[float] | None,
    alternative: Sequence[float] | None,
    reference_value: float,
    tolerance: float,
    *,
    margin: float = 0.0,
) -> dict[str, Any] | None:
    """The batch-means half-width of an arm effect and whether it converged.

    Both arms' statistics come from the same years, the shorter arm's, so
    batch ``k`` holds the same years in each and the effect differences them
    batch by batch: shared noise cancels and independent noise adds. ``margin``
    widens the half-width by what the batches cannot show: for an effect on
    TVaR, both arms' integration margins (``tvar_status``). An effect
    converges when its half-width is at most ``tolerance`` times the reference
    value.
    """
    if reference is None or alternative is None or len(reference) != len(alternative):
        return None
    width = batch_half_width(
        [b - a for a, b in zip(reference, alternative, strict=True)]
    )
    return _status(reference_value, width + margin, tolerance)


def gate_status(value: float, batch_values: Sequence[float], tolerance: float):
    """Half-width and convergence of a statistic from its batch values."""
    return _status(value, batch_half_width(batch_values), tolerance)
