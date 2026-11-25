"""Deterministic trajectory quadrature of the public-frontier forecast.

Every lab path is ``F0_l + B·M_l·(t − as_of)``: ``B`` is the positive-truncated
Gaussian trend-rate kernel, drawn once per path and shared by the labs, and
``M_l`` each lab's independent pace, a median-one log-logistic law truncated at
the forecast's pace cap, so labs progress at different rates, none faster than
the cap times ``B``, and the frontier leader can change. The study integrates the
forecast over the frontier progress ``P``, the public frontier's gain over its
as-of level at the middle of the policy period, with a fixed rule instead of
sampling paths. Given ``B``, ``P`` is at most ``p`` exactly when every lab's
pace is at most its threshold, so every law's distribution of ``P`` is a
product of the labs' pace distributions integrated over ``B``, and exact. Each
node carries one representative joint path drawn from the base law given its
progress: the rule integrates the frontier's progress up to quadrature error,
and every other feature of a path, such as which lab leads, by one stratified
draw per node. methodology.md ("Trajectory quadrature") gives the rule: the
law of ``P``, the covering law, the stretched Fejér nodes, the representative
paths and the embedded rule.
"""

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import cache, cached_property

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr, ndtri
from scipy.stats import norm

from catmodel.contracts.public_frontier import (
    YEAR_SECONDS,
    PublicFrontierForecast,
    PublicFrontierQuadrature,
    PublicQuadratureLaw,
    PublicQuadratureNode,
    PublicTrendPosterior,
    QuadratureVariant,
    forecast_times,
    frontier_progress,
    progress_elapsed_years,
    quadrature_node_ids,
)
from catmodel.forecast.public_frontier import (
    PACE_CAP_ALTERNATIVES,
    PACE_SHAPE_ALTERNATIVES,
    fit_public_trend,
    trend_discrepancy_alternatives,
)

DEFAULT_TRAJECTORY_NODES = 7
"""Interior trajectory nodes; with the no-progress node, eight collected paths.

A collection-budget judgment, not a convergence result: the embedded three-node
rule reports how much the integrated losses move between the two nested rules.
"""
RATE_POINTS = 64
"""Gauss–Legendre points on each interval of the trend-rate integral (numerical)."""
RATE_SPAN = 10.0
"""Standard deviations of the rate kernel that the trend-rate integral covers on
each side of the kernel's location (numerical): the kernel's mass beyond them
is below 1e-22."""
PROGRESS_CHUNK = 256
"""Progress values a law evaluates at once (numerical): it bounds each
intermediate array of progress values by rate points by labs to a few MB."""
TAIL_STRETCH = 1.5
"""Normal-score stretch of the node scale (numerical design choice).

Seven Fejér nodes on the covering law's own probability scale end at its 3.8%
and 96.2% points, which under-integrates losses that grow quickly with the
progress. Stretched by 1.5 they end at its 0.4% and 99.6% points, and the
weights carry the scale's Jacobian. methodology.md gives the accuracy this
buys.
"""


def fejer_second_kind(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Fejér's second rule on (0, 1): ascending nodes and weights summing to one."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("a quadrature rule needs at least one node")
    theta = np.arange(1, count + 1) * np.pi / (count + 1)
    odd = np.arange(1, (count + 1) // 2 + 1) * 2 - 1
    weights = (
        4
        * np.sin(theta)
        / (count + 1)
        * (np.sin(np.outer(theta, odd)) / odd).sum(axis=1)
    )
    return (1 - np.cos(theta)) / 2, weights / 2


@cache
def _legendre() -> tuple[np.ndarray, np.ndarray]:
    points, weights = np.polynomial.legendre.leggauss(RATE_POINTS)
    return (points + 1) / 2, weights / 2


def _log_logistic_cdf(pace: np.ndarray | float, shape: float) -> np.ndarray:
    """Log CDF of the median-one log-logistic law (``-inf`` at zero)."""
    with np.errstate(divide="ignore"):
        return -np.logaddexp(0.0, -shape * np.log(pace))


def _log_pace_cdf(pace: np.ndarray, shape: float, cap: float) -> np.ndarray:
    """Log CDF of the pace: the median-one log-logistic law truncated above at
    ``cap`` and renormalized (``-inf`` at zero, zero at and beyond the cap)."""
    return np.minimum(
        _log_logistic_cdf(pace, shape) - _log_logistic_cdf(cap, shape), 0.0
    )


def _log_pace_density(pace: np.ndarray, shape: float, cap: float) -> np.ndarray:
    """Log density of the capped pace (``-inf`` beyond the cap)."""
    log_pace = np.log(pace)
    inside = (
        np.log(shape)
        + (shape - 1) * log_pace
        - 2 * np.logaddexp(0.0, shape * log_pace)
        - _log_logistic_cdf(cap, shape)
    )
    return np.where(pace <= cap, inside, -np.inf)


def node_uniform(draw_id: str, component: str) -> float:
    """A fixed uniform in (0, 1) for one component of one node's draw.

    It is the first 56 bits of SHA-256 of ``"{draw_id}:{component}"``, so a
    node reads the same uniforms, and so keeps its representative path, in
    every nested rule and on every platform.
    """
    digest = hashlib.sha256(f"{draw_id}:{component}".encode()).digest()
    return (int.from_bytes(digest[:7], "big") + 0.5) / 2**56


def _pick(weights: np.ndarray, uniform: float) -> int:
    cumulative = np.cumsum(weights)
    index = int(np.searchsorted(cumulative, uniform * cumulative[-1], side="left"))
    return min(index, len(weights) - 1)


@dataclass(frozen=True)
class _Law:
    """One forecast law of the frontier progress ``P`` at ``elapsed`` years."""

    label: str
    variant: QuadratureVariant
    pace_shape: float
    pace_cap: float
    posterior: PublicTrendPosterior
    elapsed: float
    lab_ids: tuple[str, ...]

    @cached_property
    def _gaps(self) -> np.ndarray:
        """Each lab's distance below the as-of frontier, in ``lab_ids`` order."""
        anchors = np.asarray(
            [self.posterior.labs[lab].as_of_value for lab in self.lab_ids]
        )
        return anchors.max() - anchors

    def _reach_rates(self, progress: np.ndarray) -> np.ndarray:
        """Each lab's lowest trend rate at which it can bring the frontier to
        ``progress``: at a lower rate the lab's threshold exceeds the pace cap.

        Axes: progress values, labs.
        """
        return (progress[:, None] + self._gaps[None, :]) / (
            self.pace_cap * self.elapsed
        )

    def _rate_span(self) -> tuple[float, float]:
        """The trend rates the rate integral covers: ``RATE_SPAN`` kernel
        standard deviations on each side of the positive kernel's bulk."""
        location = self.posterior.rate_location
        reach = RATE_SPAN * float(np.sqrt(self.posterior.rate_variance))
        return max(location - reach, 0.0), max(location, 0.0) + reach

    def _rate_rule(
        self, progress: np.ndarray, top: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trend-rate points and their kernel weights for each progress value,
        and the kernel's mass at rates too low for any lab to reach it.

        Axes of the points and weights: progress values, rate points. At a
        lab's reach rate its capped pace CDF has a kink and its leader term a
        jump, so the rule is Gauss–Legendre on each interval between
        consecutive reach rates, where the integrand is smooth, from the
        lowest reach rate to ``top`` (the end of the rate span when omitted).
        A kernel without variance is its one rate.
        """
        location, variance = self.posterior.rate_location, self.posterior.rate_variance
        count = len(progress)
        if variance == 0:
            return (
                np.full((count, 1), location),
                np.ones((count, 1)),
                np.zeros(count),
            )
        sd = float(np.sqrt(variance))
        low, high = self._rate_span()
        if top is not None:
            high = top
        edges = np.concatenate(
            [
                np.sort(np.clip(self._reach_rates(progress), low, high), axis=1),
                np.full((count, 1), high),
            ],
            axis=1,
        )
        points, weights = _legendre()
        width = np.diff(edges, axis=1)[:, :, None]
        rates = (edges[:, :-1, None] + width * points).reshape(count, -1)
        # The positive-truncated Gaussian kernel's density and distribution.
        log_mass = log_ndtr(location / sd)
        density = np.exp(-0.5 * ((rates - location) / sd) ** 2 - log_mass) / (
            sd * np.sqrt(2 * np.pi)
        )
        unreached = -np.expm1(log_ndtr((location - edges[:, 0]) / sd) - log_mass)
        return rates, (width * weights).reshape(count, -1) * density, unreached

    def _thresholds(self, progress: np.ndarray, rates: np.ndarray) -> np.ndarray:
        """Each lab's largest pace consistent with frontier progress ``progress``.

        Axes: progress values, rate points, labs.
        """
        return (progress[:, None, None] + self._gaps[None, None, :]) / (
            rates[:, :, None] * self.elapsed
        )

    def _leader_terms(self, progress: np.ndarray, rates: np.ndarray) -> np.ndarray:
        """Density of ``P`` at ``progress`` with each lab leading, per rate point.

        ``P`` has density ``Σ_k f(m_k)/(B·elapsed) · Π_{l≠k} F(m_l)`` given ``B``,
        where ``m_l`` is lab ``l``'s threshold and ``F``, ``f`` the capped pace
        law's CDF and density; term ``k`` is the density with lab ``k`` at the
        frontier, and is zero below that lab's reach rate.
        """
        thresholds = self._thresholds(progress, rates)
        log_cdf = _log_pace_cdf(thresholds, self.pace_shape, self.pace_cap)
        log_density = _log_pace_density(
            thresholds, self.pace_shape, self.pace_cap
        ) - np.log(rates[:, :, None] * self.elapsed)
        return np.exp(log_density + log_cdf.sum(axis=2, keepdims=True) - log_cdf)

    def cdf(self, progress: np.ndarray) -> np.ndarray:
        return _in_chunks(self._cdf, progress)

    def density(self, progress: np.ndarray) -> np.ndarray:
        progress = np.asarray(progress, dtype=float)
        if np.any(progress <= 0):
            raise ValueError("frontier progress densities need positive progress")
        return _in_chunks(self._density, progress)

    def _cdf(self, progress: np.ndarray) -> np.ndarray:
        rates, weights, unreached = self._rate_rule(progress)
        log_cdf = _log_pace_cdf(
            self._thresholds(progress, rates), self.pace_shape, self.pace_cap
        )
        within = (np.exp(log_cdf.sum(axis=2)) * weights).sum(axis=1)
        return np.minimum(unreached + within, 1.0)

    def _density(self, progress: np.ndarray, top: float | None = None) -> np.ndarray:
        """Density of ``P``, or its part from trend rates up to ``top``."""
        rates, weights, _ = self._rate_rule(progress, top)
        return (self._leader_terms(progress, rates).sum(axis=2) * weights).sum(axis=1)

    def _rate_given_progress(self, point: np.ndarray, uniform: float) -> float:
        """The ``uniform`` quantile of the trend rate given frontier progress
        ``point``: the rate below which that share of the density of ``P``
        there accumulates."""
        if self.posterior.rate_variance == 0:
            return self.posterior.rate_location
        total = self._density(point)[0]
        if not total > 0:
            raise ValueError(
                "frontier progress lies beyond what the pace cap allows at any "
                "integrated trend rate"
            )
        low, high = self._rate_span()
        return float(
            brentq(
                lambda top: self._density(point, top)[0] - uniform * total,
                min(max(low, float(self._reach_rates(point).min())), high),
                high,
                xtol=1e-12,
                rtol=1e-12,
            )
        )

    def representative(self, progress: float, draw_id: str) -> tuple[float, np.ndarray]:
        """One joint draw from this law given frontier progress ``progress``.

        The trend rate is drawn from its law given that progress, then the
        leading lab by its share of the density of ``P`` at that rate; the
        leader's pace sits at its threshold, so the frontier gains exactly
        ``progress`` by the period's middle, and every other lab's pace is
        drawn from its law below its own threshold and the cap. A rate at
        which the leader would need a pace above the cap has no density, so
        no pace exceeds the cap. ``node_uniform`` supplies each choice.
        """
        point = np.asarray([progress], dtype=float)
        rate = self._rate_given_progress(point, node_uniform(draw_id, "trend_rate"))
        at_rate = np.asarray([[rate]])
        thresholds = self._thresholds(point, at_rate)[0, 0]
        terms = self._leader_terms(point, at_rate)[0, 0]
        if not terms.sum() > 0:
            raise ValueError(
                "frontier progress lies beyond what the pace cap allows at the "
                "trend rate"
            )
        leader = _pick(terms, node_uniform(draw_id, "leader"))
        # Below a threshold within the cap, the capped law and the log-logistic
        # law it truncates have the same conditional law.
        log_cdf = _log_logistic_cdf(
            np.minimum(thresholds, self.pace_cap), self.pace_shape
        )
        pace = np.empty(len(self.lab_ids))
        for position, lab in enumerate(self.lab_ids):
            if position == leader:
                pace[position] = thresholds[position]
                continue
            below = node_uniform(draw_id, lab) * np.exp(log_cdf[position])
            pace[position] = np.exp(
                (np.log(below) - np.log1p(-below)) / self.pace_shape
            )
        return rate, pace


def _in_chunks(
    evaluate: Callable[[np.ndarray], np.ndarray], progress: np.ndarray
) -> np.ndarray:
    """``evaluate`` over ``progress`` in ``PROGRESS_CHUNK`` values at a time,
    keeping the input's shape."""
    values = np.asarray(progress, dtype=float)
    flat = values.reshape(-1)
    result = np.empty_like(flat)
    for start in range(0, flat.size, PROGRESS_CHUNK):
        stop = start + PROGRESS_CHUNK
        result[start:stop] = evaluate(flat[start:stop])
    return result.reshape(values.shape)


def _inverse(cdf: Callable[[np.ndarray], np.ndarray], target: float, guess: float):
    low, high = guess, guess
    while cdf(np.asarray([low]))[0] > target:
        low /= 2
    while cdf(np.asarray([high]))[0] < target:
        high *= 2
    return float(
        np.exp(
            brentq(
                lambda value: cdf(np.asarray([np.exp(value)]))[0] - target,
                np.log(low),
                np.log(high),
                xtol=1e-12,
                rtol=1e-12,
            )
        )
    )


def _base_law(
    pace_shape: float,
    pace_cap: float,
    posterior: PublicTrendPosterior | Mapping[str, object],
    elapsed: float,
) -> _Law:
    """Rebuild the base progress law from its retained setup parameters."""
    selected = (
        posterior
        if isinstance(posterior, PublicTrendPosterior)
        else PublicTrendPosterior.model_validate(posterior)
    )
    if not elapsed > 0:
        raise ValueError("frontier progress needs a positive elapsed time")
    if not pace_cap > 1:
        raise ValueError("the pace cap must exceed the median pace of one")
    return _Law(
        "baseline",
        "baseline",
        pace_shape,
        pace_cap,
        selected,
        elapsed,
        tuple(selected.labs),
    )


def _guess(law: _Law) -> float:
    return max(law.posterior.rate_location * law.elapsed, 1e-6)


def base_progress_cdf(
    pace_shape: float,
    posterior: PublicTrendPosterior | Mapping[str, object],
    elapsed: float,
    progress: float | Sequence[float] | np.ndarray,
    *,
    pace_cap: float,
) -> float | np.ndarray:
    """Evaluate the base forecast CDF of frontier progress ``P``.

    ``elapsed`` is the years from as-of to the policy period's middle. Scalar
    input returns a float; an array-like input keeps its shape. Progress at or
    below zero has CDF zero because the base law has positive support, and the
    CDF is one beyond the progress the pace cap allows at the largest rate
    point.
    """
    values = np.asarray(progress, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("frontier-progress CDF inputs must be finite")
    scalar = values.ndim == 0
    flat = values.reshape(-1)
    result = np.zeros_like(flat)
    positive = flat > 0
    if np.any(positive):
        result[positive] = _base_law(pace_shape, pace_cap, posterior, elapsed).cdf(
            flat[positive]
        )
    shaped = result.reshape(values.shape)
    return float(shaped) if scalar else shaped


def base_progress_quantile(
    pace_shape: float,
    posterior: PublicTrendPosterior | Mapping[str, object],
    elapsed: float,
    probabilities: float | Sequence[float] | np.ndarray,
    *,
    pace_cap: float,
) -> float | np.ndarray:
    """Invert the base forecast CDF of frontier progress ``P``.

    Probabilities must be strictly between zero and one. Scalar input returns
    a float; an array-like input keeps its shape. Inversion uses the same
    log-progress root solver as the trajectory quadrature.
    """
    values = np.asarray(probabilities, dtype=float)
    if not np.all(np.isfinite(values)) or np.any((values <= 0) | (values >= 1)):
        raise ValueError("frontier-progress quantile probabilities must lie in (0, 1)")
    scalar = values.ndim == 0
    law = _base_law(pace_shape, pace_cap, posterior, elapsed)
    flat = values.reshape(-1)
    result = np.asarray(
        [_inverse(law.cdf, float(target), _guess(law)) for target in flat],
        dtype=float,
    ).reshape(values.shape)
    return float(result) if scalar else result


def _laws(
    forecast: PublicFrontierForecast,
    posterior: PublicTrendPosterior,
    elapsed: float,
) -> tuple[_Law, list[_Law], dict[str, str]]:
    labs = tuple(row.lab_id for row in forecast.coordinates)
    cap = forecast.pace_cap
    base = _Law(
        "baseline", "baseline", forecast.pace_shape, cap, posterior, elapsed, labs
    )
    alternatives = [
        _Law(label, "baseline", shape, cap, posterior, elapsed, labs)
        for label, shape in PACE_SHAPE_ALTERNATIVES.items()
    ]
    # The nodes lie under the forecast's cap, so only a lower cap can be
    # integrated on them.
    unavailable: dict[str, str] = {
        "higher_pace_cap": (
            "The trajectory nodes are placed under the forecast's pace cap, so a "
            "higher cap, or none, has paths beyond every node; it needs a study "
            "planned and collected under that cap."
        )
    }
    for label, lower in PACE_CAP_ALTERNATIVES.items():
        if lower < cap:
            alternatives.append(
                _Law(
                    label,
                    "baseline",
                    forecast.pace_shape,
                    lower,
                    posterior,
                    elapsed,
                    labs,
                )
            )
        else:
            unavailable[label] = (
                "The forecast's pace cap is not above this alternative's, so the "
                "alternative is not a lower cap."
            )
    discrepancies = trend_discrepancy_alternatives(forecast, posterior)
    if not discrepancies:
        unavailable["trend_discrepancy"] = (
            "The trend discrepancy is authored and no pooled estimate is identified, "
            "so the forecast has no discrepancy alternative."
        )
    alternatives.extend(
        _Law(
            label,
            "baseline",
            forecast.pace_shape,
            cap,
            fit_public_trend(forecast.model_copy(update={"trend_discrepancy_sd": sd})),
            elapsed,
            labs,
        )
        for label, sd in discrepancies.items()
    )
    return base, alternatives, unavailable


def require_trajectory_nodes(nodes: object) -> int:
    """``nodes`` as a trajectory rule's interior node count, an odd integer
    of at least 3."""
    if (
        isinstance(nodes, bool)
        or not isinstance(nodes, int)
        or nodes < 3
        or not nodes % 2
    ):
        raise ValueError("the trajectory rule needs an odd number of at least 3 nodes")
    return nodes


def public_frontier_quadrature(
    forecast: PublicFrontierForecast,
    *,
    period_start: datetime,
    period_end: datetime,
    nodes: int = DEFAULT_TRAJECTORY_NODES,
) -> PublicFrontierQuadrature:
    """Trajectory nodes and exact node weights for the forecast and its alternatives."""
    forecast = PublicFrontierForecast.model_validate(forecast.model_dump(mode="json"))
    nodes = require_trajectory_nodes(nodes)
    times = forecast_times(period_start, period_end)
    if forecast.as_of > times[0]:
        raise ValueError("public forecast as-of cannot follow the period start")
    posterior = fit_public_trend(forecast)
    middle = progress_elapsed_years(forecast.as_of, times)
    base, alternatives, unavailable = _laws(forecast, posterior, middle)
    # The covering law's members share the forecast's cap, so every node lies
    # inside the capped pace's support.
    wider = [
        law
        for law in alternatives
        if law.pace_shape < forecast.pace_shape and law.pace_cap == forecast.pace_cap
    ]
    cover = [base, min(wider, key=lambda law: law.pace_shape)] if wider else [base]

    def cover_cdf(progress):
        return np.mean([law.cdf(progress) for law in cover], axis=0)

    scale, fejer = fejer_second_kind(nodes)
    scores = ndtri(scale)
    probabilities = ndtr(TAIL_STRETCH * scores)
    jacobian = TAIL_STRETCH * norm.pdf(TAIL_STRETCH * scores) / norm.pdf(scores)
    rule = fejer * jacobian
    embedded = np.zeros(nodes)
    embedded[1::2] = fejer_second_kind((nodes - 1) // 2)[1] * jacobian[1::2]
    progress = np.asarray(
        [_inverse(cover_cdf, float(target), _guess(base)) for target in probabilities]
    )
    mixture = np.mean([law.density(progress) for law in cover], axis=0)

    def weights(law: _Law, rule_weights: np.ndarray) -> tuple[float, ...] | None:
        ratio = rule_weights * law.density(progress) / mixture
        total = ratio.sum()
        return (0.0, *map(float, ratio / total)) if total > 0 else None

    elapsed = np.asarray(
        [(at - forecast.as_of).total_seconds() / YEAR_SECONDS for at in times]
    )
    anchors = np.asarray(
        [posterior.labs[row.lab_id].as_of_value for row in forecast.coordinates]
    )

    def quadrature_law(law: _Law) -> PublicQuadratureLaw | None:
        full, nested = weights(law, rule), weights(law, embedded)
        if full is None or nested is None:
            unavailable[law.label] = (
                "No trajectory node of one of the nested rules lies where this "
                "law has numerically positive density."
            )
            return None
        return PublicQuadratureLaw(
            label=law.label,
            variant=law.variant,
            pace_shape=law.pace_shape,
            pace_cap=law.pace_cap,
            trend_discrepancy_sd=law.posterior.trend_discrepancy_sd,
            weights=full,
            embedded_weights=nested,
        )

    base_law = quadrature_law(base)
    if base_law is None:
        raise ValueError("the trajectory nodes do not cover the base forecast")
    integrated = [
        row for row in (quadrature_law(law) for law in alternatives) if row is not None
    ]

    def node(draw_id: str, target: float | None) -> PublicQuadratureNode:
        if target is None:
            baseline, pace = None, np.zeros_like(anchors)
            rates = np.zeros_like(anchors)
        else:
            baseline, pace = base.representative(target, draw_id)
            rates = baseline * pace
        return PublicQuadratureNode(
            draw_id=draw_id,
            progress=float(frontier_progress(anchors, rates, middle)[0]),
            baseline_rate=baseline,
            pace_multipliers=tuple(map(float, pace)),
            rates=tuple(map(float, rates)),
            values=tuple(
                tuple(map(float, row))
                for row in anchors[None, :] + elapsed[:, None] * rates[None, :]
            ),
        )

    no_progress = (1.0, *([0.0] * nodes))
    return PublicFrontierQuadrature(
        forecast_sha256=forecast.digest(),
        as_of=forecast.as_of,
        coordinates=forecast.coordinates,
        times=times,
        posterior=posterior,
        tail_stretch=TAIL_STRETCH,
        covering_pace_shapes=tuple(law.pace_shape for law in cover),
        median_progress=_inverse(base.cdf, 0.5, _guess(base)),
        interior_nodes=nodes,
        nodes=tuple(
            node(draw_id, target)
            for draw_id, target in zip(
                quadrature_node_ids(nodes), (None, *progress), strict=True
            )
        ),
        base=base_law,
        alternatives=(
            *integrated,
            PublicQuadratureLaw(
                label="no_progress",
                variant="no_progress",
                pace_shape=None,
                pace_cap=None,
                trend_discrepancy_sd=posterior.trend_discrepancy_sd,
                weights=no_progress,
                embedded_weights=no_progress,
            ),
        ),
        unavailable_alternatives=unavailable,
        source_references=tuple(
            dict.fromkeys(
                [ref for row in forecast.coordinates for ref in row.evidence_refs]
                + [ref for row in forecast.releases for ref in row.evidence_refs]
            )
        ),
        assumptions=(
            "Each lab path starts at the lab's highest published score available at as-of, so no path starts below what was already public. Paths are linear and nondecreasing.",
            f"One positive trend rate, fitted to all labs' frontier-raising releases with lab-specific levels, is drawn once per path and shared by the labs; each lab draws its own independent pace multiplier, a median-one log-logistic law truncated at the pace cap of {forecast.pace_cap:g}, so labs progress at different rates, none faster than the cap times the drawn rate, and the frontier leader can change. The no-progress node is the no-progress counterfactual, every lab held at its as-of frontier, and carries weight only under that alternative.",
            "The study integrates the forecast over the frontier progress, the public frontier's gain over its as-of level at the middle of the period, by a fixed rule: Fejér nodes on a normal-score scale stretched by 1.5 over an equal mixture of the base law and its widest supported pace alternative, each law weighted by its exact density ratio to that mixture. Each node carries one representative joint path drawn from the base law given its progress, so the rule integrates the frontier's progress and every other feature of a path, such as which lab leads, by one draw per node. The integrated loss carries quadrature and sampling error, which the embedded lower-order rule indicates unless one node carries nearly all of a law's weight.",
            "Record scores have independent Gaussian error equal to the shared trend discrepancy plus the release's benchmark error; records are upward-selected releases and serial dependence of record jumps is not modeled.",
            "The frozen, retrospectively refitted ECI scale does not reconstruct historical score vintages.",
            "The public frontier is the maximum of the named lab frontiers. Its coverage depends on the explicitly declared lab roster.",
            f"Elapsed time is measured in 365.25-day years from as-of. Paths are evaluated from {times[0].isoformat()} to {times[-1].isoformat()} at each month start between.",
            "Domain benchmark results, such as cyber or biology evaluations, remain contextual evidence; none is forecast, converted to ECI, or used to relabel an existing domain response curve.",
            "Shared public trajectories preserve this source of dependence only; residual risk dependence is separate.",
            forecast.frontier_definition,
            forecast.pooling_rationale,
            forecast.likelihood_rationale,
            forecast.pace_rationale,
            forecast.rate_prior.rationale,
            *forecast.limitations,
        ),
    )
