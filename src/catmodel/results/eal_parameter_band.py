"""How uncertain the EAL is because the elicited count laws are uncertain.

A report-only exhibit (``NUM-EAL-PARAMETER-BAND``; methodology.md defines it):
the distribution of ``A = sum_g EAL_{R,g}(Z_g)``, with one normal score ``Z_g``
per rank group (one scenario at one source status) held for every year and the
epistemic group ``R`` (in a context-bound setup, the respondent) drawn by its
weight. Each ranked count law's expected count at score ``z`` is the reference
rate times ``score_multiplier(law)(z)``, the function the simulation uses.
Every linear view's episode terms are linear in the expected count, and an
annual lab term is its law's mean times the probability of at least one
incident at that count, so a state's EAL at fixed scores is exact without
simulation: each incident's analytic episode mean (``analytic_linear_eals``)
times its multiplier, plus its annual term's mean times the probability at the
score over its mean over the rank. The band is computed when a run executes and
captured beside its draws; replay recomputes it from the laws, and the
reduction reads the captured record.

Where a state is left out of the views that count the lab's own loss, the
lab-loss views and gross (``POOL-EXCLUDED-SAMPLES``), the band of such a view
is computed at that view's weights (``respondent_sampling.ViewWeights``), from
the states it weighs, and its row records how many they are and the epistemic
groups that then hold fewer of them than the others.

The band mixes the respondents as separate sources of the count laws. A state
whose model does not answer a leaf reads a donor's answers to it
(``POOL-UNANSWERED-SCENARIOS``; ``JointState.borrowed_leaves``), so its
respondent repeats another's laws of that leaf and the mixture counts them as
a second source's: it does not show the spread that separate answers would
give. The category pools make a borrowing state's laws depend on its own
answers and on the donor's, so no regrouping of the states gives that spread
either. Results
report no band for a view in which a state with weight reads a donor's answers
(``borrowed_band_reason``), on the test that withholds the
respondent-sampling standard error (``SampleGroups.borrower``). The capture
still records the band of every view, which is a function of the laws and the
weights alone, so replay compares it as before.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import ndtr, ndtri

from catmodel.contracts.incident import counts_lab_loss
from catmodel.probability.keyed_streams import incident_stream_generator
from catmodel.probability.percentiles import SortedSample
from catmodel.results.respondent_sampling import borrowed_reason
from catmodel.simulation.incident_moments import (
    CERTIFIED_INTEGRATION_RELATIVE_ERROR,
    analytic_linear_eals,
    at_least_one_probability,
)
from catmodel.simulation.incident_runtime import score_multiplier
from catmodel.simulation.small_events import SIMULATED_RANK_RANGE
from catmodel.simulation.tail_gate import gate_status

if TYPE_CHECKING:
    from collections.abc import Callable

    from catmodel.results.respondent_sampling import SampleGroups, ViewWeights
    from catmodel.simulation.incident import CompiledIncidentState

BAND_PROBABILITIES = (0.05, 0.5, 0.95)
"""The reported band quantiles."""
RANK_GRID_POINTS = 2**17 + 1
"""Comonotone normal scores: grid points equally spaced on the score scale.

An INITIAL_ESTIMATE numerical judgment (``NUM-EAL-PARAMETER-BAND``): each
quantile is bracketed between grid values, so the grid sets the certified
error bound, which shrinks in proportion to the spacing.
"""
RANK_GRID_SCORE = 8.0
"""Half-range of the comonotone grid in the normal score; ranks beyond it
carry 1.2e-15 of the probability and only widen the bracket by that."""
INDEPENDENT_SAMPLES = 2**17
"""Independent ranks: Monte Carlo rank draws per epistemic group and rank
group, an INITIAL_ESTIMATE numerical judgment (``NUM-EAL-PARAMETER-BAND``)."""
STREAM_PURPOSE = "eal-parameter-band"
REPLAY_RELATIVE_TOLERANCE = 1e-9
"""Relative tolerance of a replayed band in a view whose means are closed form:
recomputation on another platform changes the multipliers and sums in their
last bits only, and the Monte Carlo draws not at all."""
INTEGRATED_REPLAY_RELATIVE_TOLERANCE = 2 * CERTIFIED_INTEGRATION_RELATIVE_ERROR
"""Relative tolerance in a view with a numerically integrated capped lab mean.

Each such mean is certified within ``CERTIFIED_INTEGRATION_RELATIVE_ERROR`` of
the exact one, and a last-bit change can move its refinement anywhere inside
that, so two platforms' means differ by at most twice it. Every band number is a
nonnegative combination of such means, read at the same ranks, so it moves by
at most the same share."""
BAND_MISMATCH_REASON = (
    "The captured parameter band does not match this code's recomputation from "
    "the laws; recapture to report it."
)
BAND_INTERPRETATION = (
    "Count-law and respondent uncertainty, with the capability forecast "
    "integrated: the EAL with each rank group's count law (one scenario at one "
    "source status) held at one normal score for every year instead of integrated "
    "over it, each respondent's trajectory nodes integrated by their weights at "
    "those scores, and the respondents mixed by weight. Its mean is the reported "
    "EAL. The base forecast's rate is integrated; the forecast arms show how "
    "the EAL moves under alternative forecast laws. "
    "Independent ranks are the model's own assumption; comonotone ranks give "
    "every rank group the same normal score. The comonotone case has the larger mean of "
    "the worst 1-p share (TVaR_p) at every p and the smaller mean of the best "
    "share; a single quantile under an intermediate dependence need not lie "
    "between the two cases, and negative dependence between ranks is not "
    "covered. The band is not a range of annual losses and is neither a "
    "confidence nor a predictive interval; it excludes model and elicitation "
    "error beyond the stated laws. VaR, TVaR and 1-in-N losses are statistics "
    "of a single predictive year, whose ranks are redrawn every year. Where a "
    "state is left out of the views that count the lab's own loss, the "
    "lab-loss views and gross (POOL-EXCLUDED-SAMPLES), the band of such a view "
    "is computed at that view's weights: its states counts the states it "
    "weighs, and its incomplete_epistemic_groups names the respondents that "
    "hold fewer of them than the others, each of which integrates only its "
    "own states there. Where a state with weight in a view reads another "
    "model's answers to a scenario its own model does not answer "
    "(POOL-UNANSWERED-SCENARIOS), its respondent repeats a donor's count laws, "
    "the respondents are not independent and no band is reported for the view."
)
INCOMPLETE_GROUPS_NOTE = (
    "These epistemic groups hold fewer positive-weight states than the others: "
    "in a context-bound setup, respondents excluded at some trajectory node "
    "because a leaf's direct question exhausted its validation attempts or "
    "its direct answer declared the lab's loss unavailable "
    "(POOL-EXCLUDED-SAMPLES). Each integrates only its own states, so its weight "
    "is spread over its remaining nodes, and its term mixes some variation "
    "between trajectory nodes into the variation between respondents; "
    "conditional_eal_by_epistemic_group does the same."
)

NO_BAND = "the respondents the band mixes are not independent and no band is reported."
"""What a borrowing state with weight does to a view's band
(``respondent_sampling.borrowed_reason``)."""

REWEIGHTED_BAND_REASON = (
    "These results reweight the captured states to target weights other than "
    "the run's own; the band is captured at the run's weights, so it does not "
    "describe the reweighted target. A run at the target weights reports it."
)


def band_view(view: str) -> bool:
    """The views with a band: welfare, gross and every lab-loss view."""
    return view in {"welfare", "gross"} or view.startswith("lab_loss::")


@dataclass(frozen=True)
class _Term:
    """One ranked incident's weighted analytic means by view and its multiplier."""

    scenario_id: str
    coefficients: np.ndarray
    multiplier: Callable[[np.ndarray], np.ndarray]
    # A law's episode and annual terms share its origin.
    origin_id: str


@dataclass(frozen=True)
class _Respondent:
    """An epistemic group's EAL by view, its states integrated by their
    conditional weights: the fixed-rate part, the ranked incidents by rank
    group, and the certified integration bound of the whole mean."""

    fixed: np.ndarray
    bounds: np.ndarray
    groups: dict[str, list[_Term]]

    def mean(self, index: int, group: str | None = None) -> float:
        """The analytic mean in one view, or one rank group's."""
        rows = self.groups.values() if group is None else [self.groups.get(group, [])]
        return math.fsum(
            (
                float(self.fixed[index]) if group is None else 0.0,
                *(float(term.coefficients[index]) for terms in rows for term in terms),
            )
        )

    def curve(self, group: str, scores: np.ndarray) -> np.ndarray:
        """One rank group's means by view, every node read at ``scores``."""
        result = np.zeros((len(self.fixed), len(scores)))
        for term in self.groups[group]:
            result += term.coefficients[:, None] * term.multiplier(scores)[None, :]
        return result


def _occurrence_multiplier(
    incident, multiplier: Callable[[np.ndarray], np.ndarray], probability: float
) -> Callable[[np.ndarray], np.ndarray]:
    """An annual lab term's multiplier at the year's score: the probability of
    at least one incident at that score's count, over its mean over the rank
    (``at_least_one_probability``), so the term's coefficient is its mean."""
    v = incident.count_variance_to_mean
    decay = 1.0 if incident.count_family == "poisson" else math.log(v) / (v - 1)

    def occurrence(scores: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", under="ignore"):
            counts = incident.annual_rate * multiplier(scores)
            return -np.expm1(-decay * counts) / probability

    return occurrence


def _respondents(
    compiled: Mapping[str, CompiledIncidentState],
    weights: Mapping[str, float],
    epistemic_groups: Mapping[str, str],
    views: Sequence[str],
) -> tuple[list[str], np.ndarray, list[_Respondent], dict[str, str]]:
    """Each epistemic group's weight and EAL terms, and why a view has none.

    A state's incidents contribute their analytic means times the state's
    share of its group's weight. A view has no band where some incident has
    no analytic mean in it or a negative one, since the curves must be
    nonnegative and non-decreasing.
    """
    keys = [key for key, weight in weights.items() if weight > 0]
    names = sorted({epistemic_groups[key] for key in keys})
    totals = {
        name: math.fsum(weights[key] for key in keys if epistemic_groups[key] == name)
        for name in names
    }
    rows = {
        name: _Respondent(np.zeros(len(views)), np.zeros(len(views)), {})
        for name in names
    }
    reasons: dict[str, str] = {}
    for key in keys:
        respondent = rows[epistemic_groups[key]]
        share = weights[key] / totals[epistemic_groups[key]]
        state = compiled[key]
        for incident in state.incidents:
            if incident.annual_rate == 0:
                continue
            single = replace(state, incidents=(incident,))
            analytic = analytic_linear_eals(single, views, annual=False)
            annual = (
                analytic_linear_eals(single, views, episodes=False)
                if incident.annual_lab_losses
                else None
            )
            for view in views:
                if view not in analytic.values or (
                    annual is not None and view not in annual.values
                ):
                    reasons.setdefault(
                        view,
                        "An incident has no analytic mean in this view, so its count law cannot be read at fixed ranks.",
                    )
                elif analytic.values[view] < 0:
                    reasons.setdefault(
                        view,
                        "An incident has a negative mean in this view; the band needs nonnegative curves.",
                    )
            parts = [(analytic, None)]
            if annual is not None:
                occurrence = at_least_one_probability(incident, state.count_experiment)
                if occurrence is not None and occurrence[0] > 0:
                    parts.append((annual, occurrence[0]))
            for part, probability in parts:
                coefficients = share * np.asarray(
                    [part.values.get(view, 0.0) for view in views], dtype=float
                )
                respondent.bounds[:] += share * np.asarray(
                    [part.error_bounds.get(view, 0.0) for view in views], dtype=float
                )
                law = incident.parameter_rank
                if law is None:
                    respondent.fixed[:] += coefficients
                elif np.any(coefficients):
                    multiplier = score_multiplier(law)
                    respondent.groups.setdefault(law.group_id, []).append(
                        _Term(
                            incident.scenario_id,
                            coefficients,
                            multiplier
                            if probability is None
                            else _occurrence_multiplier(
                                incident, multiplier, probability
                            ),
                            incident.origin_id,
                        )
                    )
    mass = np.asarray([totals[name] for name in names], dtype=float)
    return names, mass, [rows[name] for name in names], reasons


def _quantile_brackets(
    arrays: Sequence[np.ndarray], weights: np.ndarray, ranks: np.ndarray
) -> dict[str, dict[str, float]]:
    """Band quantiles of a mixture of monotone curves, each with its bracket.

    ``arrays[r]`` holds part ``r``'s nonnegative, non-decreasing EAL at the
    grid ``ranks``. Between two grid ranks the curve lies between their
    values, so ``P(A_r <= x)`` lies between the last grid rank whose value is
    at most ``x`` and the next one (0 and 1 beyond the grid), and each
    quantile lies between the quantiles of the mixtures of those two bounds.
    The reported value interpolates each curve's rank linearly between grid
    values, so it lies inside that bracket, and the error bound is its larger
    distance to either end. Every quantile is the least float at which its
    distribution function reaches the probability, found by bisection over
    the bit patterns of nonnegative floats, which order like the floats; all
    of them bisect together.
    """
    below = np.concatenate(([0.0], ranks))
    above = np.concatenate((ranks, [1.0]))
    targets = np.tile(np.asarray(BAND_PROBABILITIES), 3)
    count = len(BAND_PROBABILITIES)
    top = np.asarray(max(float(values[-1]) for values in arrays)).view(np.int64)
    # Below ``low`` (-1 stands for every negative value) each distribution
    # function is short of its probability; at ``high`` it reaches it.
    low = np.full(len(targets), -1, dtype=np.int64)
    high = np.full(len(targets), int(top), dtype=np.int64)
    while np.any(high - low > 1):
        middle = np.maximum(low + (high - low) // 2, 0)
        x = middle.view(np.float64)
        total = np.zeros(len(targets))
        for values, weight in zip(arrays, weights, strict=True):
            index = np.searchsorted(values, x, side="right")
            inside = np.clip(index, 1, len(values) - 1)
            left, right = values[inside - 1], values[inside]
            share = np.divide(
                x - left, right - left, out=np.zeros(len(x)), where=right > left
            )
            linear = np.where(
                (index > 0) & (index < len(values)),
                ranks[inside - 1] + share * (ranks[inside] - ranks[inside - 1]),
                below[index],
            )
            total += weight * np.concatenate(
                (
                    above[index[:count]],
                    below[index[count : 2 * count]],
                    linear[2 * count :],
                )
            )
        reached = total >= targets
        active = high - low > 1
        high = np.where(active & reached, middle, high)
        low = np.where(active & ~reached, middle, low)
    values = high.view(np.float64)
    quantiles, bounds = {}, {}
    for index, probability in enumerate(BAND_PROBABILITIES):
        first, last, value = (
            float(values[index + offset]) for offset in (0, count, 2 * count)
        )
        quantiles[str(probability)] = value
        bounds[str(probability)] = max(value - first, last - value)
    return {"quantiles": quantiles, "error_bounds": bounds}


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Left inverse of the weighted distribution function at every band
    probability, from one sort (``SortedSample``)."""
    ordered = SortedSample(values, weights)
    return np.asarray([ordered.quantile(p) for p in BAND_PROBABILITIES])


def _independent(
    samples: Sequence[np.ndarray], weights: np.ndarray, batches: int, tolerance: float
) -> dict[str, Any]:
    """Band quantiles of stratified rank samples, with batch-means half-widths.

    Every part's samples have equal weight within the part, so the mixture
    weights each sample by its part's weight over the sample count. Each of
    ``batches`` contiguous batches holds the same share of every part's
    samples and gives one estimate of each quantile (sectioning). The
    samples' mean is a check on the sampling, without a half-width: where
    rarely reached ranks carry a count law's mean it falls short, and batch
    means would not show that.
    """
    count = len(samples[0])
    estimates = _weighted_quantiles(
        np.concatenate(samples), np.repeat(weights / count, count)
    )
    edges = [index * count // batches for index in range(batches + 1)]
    by_batch = np.asarray(
        [
            _weighted_quantiles(
                np.concatenate([sample[start:stop] for sample in samples]),
                np.repeat(weights / (stop - start), stop - start),
            )
            for start, stop in pairwise(edges)
        ]
    )
    return {
        "quantiles": {
            str(probability): float(value)
            for probability, value in zip(BAND_PROBABILITIES, estimates, strict=True)
        },
        "convergence": {
            str(probability): gate_status(
                float(estimates[index]), list(by_batch[:, index]), tolerance
            )
            for index, probability in enumerate(BAND_PROBABILITIES)
        },
        "sample_mean": math.fsum(
            weight * float(np.mean(sample))
            for sample, weight in zip(samples, weights, strict=True)
        ),
    }


def _comonotone_band(
    respondents: Sequence[_Respondent],
    mass: np.ndarray,
    views: Sequence[str],
    groups: Sequence[str],
    grid_points: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Each view's band with one rank for every group, and each group's band.

    Every curve is read at the same grid ranks; a group's curve is added to
    its respondent's total and its own band computed before the next group's.
    A respondent without the group adds an atom at zero.
    """
    scores = np.linspace(-RANK_GRID_SCORE, RANK_GRID_SCORE, grid_points)
    ranks = np.asarray(ndtr(scores), dtype=float)
    zeros = np.zeros(grid_points)
    totals = [np.repeat(row.fixed[:, None], grid_points, axis=1) for row in respondents]
    by_group: list[dict[str, Any]] = [{} for _ in views]
    for group in groups:
        curves = [
            row.curve(group, scores) if group in row.groups else None
            for row in respondents
        ]
        scenarios = sorted(
            {
                term.scenario_id
                for row in respondents
                for term in row.groups.get(group, ())
            }
        )
        for total, curve in zip(totals, curves, strict=True):
            if curve is not None:
                total += curve
        for index, rows in enumerate(by_group):
            rows[group] = {
                "scenario_ids": scenarios,
                "expected_annual_loss": math.fsum(
                    weight * row.mean(index, group)
                    for row, weight in zip(respondents, mass, strict=True)
                ),
                **_quantile_brackets(
                    [
                        zeros if curve is None else np.maximum.accumulate(curve[index])
                        for curve in curves
                    ],
                    mass,
                    ranks,
                ),
            }
    # Rounding can make a sum of non-decreasing curves dip by an ulp.
    bands = [
        _quantile_brackets(
            [np.maximum.accumulate(total[index]) for total in totals], mass, ranks
        )
        for index in range(len(views))
    ]
    return bands, by_group


def _independent_band(
    respondents: Sequence[_Respondent],
    names: Sequence[str],
    mass: np.ndarray,
    views: Sequence[str],
    *,
    seed: int,
    samples: int,
    batches: int,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Each view's band with independent ranks, by Monte Carlo over ranks.

    Each epistemic group and rank group reads its ranks from its own stream,
    keyed by the seed and both groups, so every node of a respondent reads
    the same ranks and the draws do not depend on other groups.
    """
    draws = []
    for name, row in zip(names, respondents, strict=True):
        total = np.repeat(row.fixed[:, None], samples, axis=1)
        for group in sorted(row.groups):
            uniforms = incident_stream_generator(
                root_seed=seed, state_id=name, origin_id=group, purpose=STREAM_PURPOSE
            ).random(samples)
            scores = ndtri(np.clip(uniforms, *SIMULATED_RANK_RANGE))
            total += row.curve(group, scores)
        draws.append(total)
    return [
        _independent([draw[index] for draw in draws], mass, batches, tolerance)
        for index in range(len(views))
    ]


def parameter_band_method(*, seed: int, batches: int, tolerance: float) -> dict:
    """The method record this code computes a band with, which a captured band
    must match before its values are compared."""
    return {
        "probabilities": list(BAND_PROBABILITIES),
        "mixture": "epistemic groups by weight; each integrates its states by their conditional weights at shared ranks",
        "comonotone": {
            "rank_grid_points": RANK_GRID_POINTS,
            "normal_score_range": [-RANK_GRID_SCORE, RANK_GRID_SCORE],
        },
        "independent": {
            "seed": seed,
            "samples_per_epistemic_group": INDEPENDENT_SAMPLES,
            "batches": batches,
            "tolerance": tolerance,
        },
    }


def compute_parameter_band(
    compiled: Mapping[str, CompiledIncidentState],
    weights: Mapping[str, float],
    epistemic_groups: Mapping[str, str],
    views: Sequence[str],
    *,
    seed: int,
    batches: int,
    tolerance: float,
    samples: int | None = None,
    grid_points: int | None = None,
    lab_loss_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """The captured EAL parameter band of the welfare, gross and lab-loss views.

    ``compiled`` holds every positive-weight state, keyed like ``weights`` and
    ``epistemic_groups``. Each available view records the band's mean and its
    certified integration bound, which the reduction compares with the
    reported EAL. ``seed`` keys the independent ranks' streams; ``batches``
    and ``tolerance`` are the tail gate's.

    ``lab_loss_weights`` are the weights of the views that count the lab's
    loss, the lab-loss views and gross, where a state is left out of them
    (``ViewWeights.counting_lab_loss``). Those views' bands are then computed
    at them, and each of their rows records its ``states`` and its own
    ``incomplete_epistemic_groups``; the record's describe ``weights``. Each
    epistemic group reads the same ranks in both.
    """
    samples = INDEPENDENT_SAMPLES if samples is None else samples
    grid_points = RANK_GRID_POINTS if grid_points is None else grid_points
    method = parameter_band_method(seed=seed, batches=batches, tolerance=tolerance)
    method["comonotone"]["rank_grid_points"] = grid_points
    method["independent"]["samples_per_epistemic_group"] = samples
    candidates = sorted(view for view in views if band_view(view))
    if (
        lab_loss_weights is not None
        and candidates
        and all(map(counts_lab_loss, candidates))
    ):
        # A band of such views alone is the band at their weights.
        weights, lab_loss_weights = lab_loss_weights, None
    record: dict[str, Any] = {"method": method}
    incomplete = _incomplete_groups(weights, epistemic_groups)
    if incomplete:
        record["incomplete_epistemic_groups"] = incomplete
        record["incomplete_epistemic_groups_note"] = INCOMPLETE_GROUPS_NOTE
    if not candidates:
        return {**record, "views": {}}
    apart = (
        []
        if lab_loss_weights is None
        else [view for view in candidates if counts_lab_loss(view)]
    )
    options = {
        "seed": seed,
        "samples": samples,
        "batches": batches,
        "tolerance": tolerance,
        "grid_points": grid_points,
    }
    described, rows = _view_bands(
        compiled,
        weights,
        epistemic_groups,
        [view for view in candidates if view not in apart],
        **options,
    )
    record.update(described)
    if apart:
        assert lab_loss_weights is not None
        described, lab_loss_rows = _view_bands(
            compiled, lab_loss_weights, epistemic_groups, apart, **options
        )
        incomplete = _incomplete_groups(lab_loss_weights, epistemic_groups)
        for view, row in lab_loss_rows.items():
            rows[view] = {
                **row,
                "states": described["states"],
                **({"incomplete_epistemic_groups": incomplete} if incomplete else {}),
            }
    record["views"] = {view: rows[view] for view in candidates}
    return record


def _incomplete_groups(
    weights: Mapping[str, float], epistemic_groups: Mapping[str, str]
) -> list[str]:
    """The epistemic groups with fewer positive-weight states than the most
    any group has."""
    states = Counter(
        epistemic_groups[key] for key, weight in weights.items() if weight > 0
    )
    return sorted(
        name for name, count in states.items() if count < max(states.values())
    )


def _view_bands(
    compiled: Mapping[str, CompiledIncidentState],
    weights: Mapping[str, float],
    epistemic_groups: Mapping[str, str],
    candidates: Sequence[str],
    *,
    seed: int,
    samples: int,
    batches: int,
    tolerance: float,
    grid_points: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The bands of ``candidates`` at ``weights``: what they were computed
    from (the epistemic groups, states, rank groups and ranked laws) and each
    view's row."""
    names, mass, respondents, reasons = _respondents(
        compiled, weights, epistemic_groups, candidates
    )
    available = [view for view in candidates if view not in reasons]
    indices = [candidates.index(view) for view in available]
    respondents = [
        _Respondent(
            row.fixed[indices],
            row.bounds[indices],
            {
                group: [
                    _Term(
                        term.scenario_id,
                        term.coefficients[indices],
                        term.multiplier,
                        term.origin_id,
                    )
                    for term in terms
                ]
                for group, terms in row.groups.items()
            },
        )
        for row in respondents
    ]
    groups = sorted({group for row in respondents for group in row.groups})
    described = {
        "epistemic_groups": len(names),
        "states": sum(1 for weight in weights.values() if weight > 0),
        "rank_groups": len(groups),
        "ranked_laws": sum(
            len({term.origin_id for term in terms})
            for row in respondents
            for terms in row.groups.values()
        ),
    }
    rows: dict[str, dict[str, Any]] = {
        view: {"status": "unavailable", "reason": reasons[view]}
        for view in candidates
        if view in reasons
    }
    if available:
        comonotone, by_group = _comonotone_band(
            respondents, mass, available, groups, grid_points
        )
        independent = _independent_band(
            respondents,
            names,
            mass,
            available,
            seed=seed,
            samples=samples,
            batches=batches,
            tolerance=tolerance,
        )
        for index, view in enumerate(available):
            rows[view] = {
                "status": "available",
                "mean": math.fsum(
                    weight * row.mean(index)
                    for row, weight in zip(respondents, mass, strict=True)
                ),
                "integration_error_bound": math.fsum(
                    weight * float(row.bounds[index])
                    for row, weight in zip(respondents, mass, strict=True)
                ),
                "fixed_rate_expected_annual_loss": math.fsum(
                    weight * float(row.fixed[index])
                    for row, weight in zip(respondents, mass, strict=True)
                ),
                "independent": independent[index],
                "comonotone": comonotone[index],
                "by_rank_group": by_group[index],
            }
    return described, rows


def _close(
    captured: float,
    recomputed: float,
    slack: float = 0.0,
    relative: float = REPLAY_RELATIVE_TOLERANCE,
) -> bool:
    scale = max(abs(captured), abs(recomputed))
    return abs(captured - recomputed) <= slack + relative * scale


def _brackets_replay(captured: Mapping, recomputed: Mapping, relative: float) -> bool:
    """Comonotone quantiles replay when their certified brackets, widened by
    the view's relative tolerance, overlap."""
    return captured["quantiles"].keys() == recomputed["quantiles"].keys() and all(
        _close(
            value,
            recomputed["quantiles"][level],
            captured["error_bounds"][level] + recomputed["error_bounds"][level],
            relative,
        )
        for level, value in captured["quantiles"].items()
    )


def _view_replays(row: Mapping, other: Mapping) -> bool:
    """One available view's numbers agree within the view's tolerance.

    A view whose band carries a positive integration bound has a numerically
    integrated capped lab mean, so its numbers get
    ``INTEGRATED_REPLAY_RELATIVE_TOLERANCE``; a half-width is compared
    relative to its quantile, since the means' shift moves every batch value
    by that share of its size.
    """
    relative = (
        INTEGRATED_REPLAY_RELATIVE_TOLERANCE
        if row["integration_error_bound"] > 0 or other["integration_error_bound"] > 0
        else REPLAY_RELATIVE_TOLERANCE
    )
    bounds = row["integration_error_bound"] + other["integration_error_bound"]
    independent, expected = row["independent"], other["independent"]
    return (
        all(
            row.get(key) == other.get(key)
            for key in ("states", "incomplete_epistemic_groups")
        )
        and _close(row["mean"], other["mean"], bounds, relative)
        and _close(
            row["fixed_rate_expected_annual_loss"],
            other["fixed_rate_expected_annual_loss"],
            bounds,
            relative,
        )
        and _close(independent["sample_mean"], expected["sample_mean"], 0.0, relative)
        and independent["quantiles"].keys() == expected["quantiles"].keys()
        and all(
            _close(value, expected["quantiles"][level], 0.0, relative)
            and _close(
                independent["convergence"][level]["half_width"],
                expected["convergence"][level]["half_width"],
                relative * max(abs(value), abs(expected["quantiles"][level])),
                relative,
            )
            for level, value in independent["quantiles"].items()
        )
        and _brackets_replay(row["comonotone"], other["comonotone"], relative)
        and row["by_rank_group"].keys() == other["by_rank_group"].keys()
        and all(
            group["scenario_ids"] == other["by_rank_group"][name]["scenario_ids"]
            and _close(
                group["expected_annual_loss"],
                other["by_rank_group"][name]["expected_annual_loss"],
                bounds,
                relative,
            )
            and _brackets_replay(group, other["by_rank_group"][name], relative)
            for name, group in row["by_rank_group"].items()
        )
    )


def band_replays(captured: Mapping[str, Any], recomputed: Mapping[str, Any]) -> bool:
    """Whether a captured band agrees with its recomputation from the laws.

    Everything but the numbers must match exactly, the method record
    included. Comonotone quantiles replay when their certified brackets
    overlap, within the view's relative tolerance; every other number within
    that tolerance, the band's means within both integration bounds as well.
    A malformed record does not replay.
    """
    try:
        if {key: value for key, value in captured.items() if key != "views"} != {
            key: value for key, value in recomputed.items() if key != "views"
        } or captured["views"].keys() != recomputed["views"].keys():
            return False
        for view, row in captured["views"].items():
            other = recomputed["views"][view]
            if row["status"] != other["status"]:
                return False
            if row["status"] != "available":
                if row != other:
                    return False
            elif not _view_replays(row, other):
                return False
    except (KeyError, TypeError, AttributeError, ValueError):
        return False
    return True


def withheld_parameter_band(reason: str) -> dict[str, Any]:
    """The band as results report it when no captured band can be shown."""
    return {
        "status": "unavailable",
        "reason": reason,
        "interpretation": BAND_INTERPRETATION,
    }


def _eal_reason(metric: Mapping[str, Any] | None) -> str | None:
    """Why the reported EAL cannot carry a band: the headline's own rules."""
    if metric is None:
        return "The view was not reduced."
    if metric.get("status") != "available":
        return metric.get("reason") or "The view is unavailable."
    status = metric.get("statistics", {}).get("expected_annual_loss", {})
    if status.get("status") != "available":
        return status.get("reason") or "The view's EAL is not reported."
    if metric.get("eal_estimator") != "native_analytic_mean":
        return "The view's EAL is not the analytic mean of every state, so its count laws cannot be read at fixed ranks."
    return None


def borrowed_band_reason(
    groups: SampleGroups, weights: ViewWeights, view: str
) -> str | None:
    """Why ``view`` has no band in an arm that weighs the states by
    ``weights``: a state with weight in the view reads a donor's answers
    (``SampleGroups.borrower``). None where none does."""
    of = weights.of(view)
    if of is None:
        return None
    found = groups.borrower(state for state, weight in of.items() if weight > 0)
    return None if found is None else borrowed_reason(*found, NO_BAND)


def reported_parameter_band(
    captured: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    reweighted: bool,
    borrowed: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """The band as results report it, from a captured record that replays.

    A view keeps its band where its EAL is reported as the analytic mean of
    every state, the band's mean matches that EAL within both certified
    integration bounds, as replay compares captured and recomputed means,
    and no state with weight in it reads a donor's answers; otherwise it is
    unavailable with the reason. ``borrowed`` gives a view's reason of the
    last kind (``borrowed_band_reason`` at the reference arm's weights), and
    None says that no state borrows. Where no view has a band and borrowing
    alone withholds some view's, the record's reason is the first such
    view's. Nothing here raises.
    """
    if reweighted:
        return withheld_parameter_band(REWEIGHTED_BAND_REASON)
    base = {
        "interpretation": BAND_INTERPRETATION,
        **{key: value for key, value in captured.items() if key != "views"},
    }
    views: dict[str, dict[str, Any]] = {}
    borrowing = None
    for view, row in captured["views"].items():
        metric = metrics.get(view)
        reason = row.get("reason") if row["status"] != "available" else None
        reason = reason or _eal_reason(metric)
        if reason is None:
            assert metric is not None
            reported = float(metric["expected_annual_loss"])
            bound = (metric.get("eal_precision") or {}).get("integration_error_bound")
            if not _close(
                row["mean"],
                reported,
                float(bound or 0.0) + row["integration_error_bound"],
            ):
                reason = "The band's mean differs from the reported EAL by more than their certified integration bounds."
        if reason is None and borrowed is not None:
            reason = borrowed(view)
            borrowing = borrowing or reason
        if reason is not None:
            views[view] = {"status": "unavailable", "reason": reason}
            continue
        assert metric is not None
        views[view] = {
            "status": "available",
            "expected_annual_loss": float(metric["expected_annual_loss"]),
            **{
                key: value
                for key, value in row.items()
                if key not in {"status", "mean", "integration_error_bound"}
            },
        }
    if not any(row["status"] == "available" for row in views.values()):
        return {
            "status": "unavailable",
            "reason": borrowing
            or "No welfare, gross or lab-loss view reports an analytic EAL with a band.",
            **base,
            "views": views,
        }
    return {"status": "available", **base, "views": views}
