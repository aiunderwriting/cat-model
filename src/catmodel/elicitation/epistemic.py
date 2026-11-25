"""The epistemic count law read from five elicited quantiles (``pchip_local_tails``).

methodology.md, "Pooling and parameter uncertainty", defines it: a five-knot
PCHIP through p5-p95 of the expected count, with capped local normal-score log
tails beyond them, read at the stretched normal score ``k z`` with a
post-stretch tail form (``POOL-SPREAD``). Ranks and scores are two coordinates
of one law: ``epistemic_quantiles`` reads it at ranks and
``stretched_score_quantiles`` at normal scores, through the same evaluation.
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr, ndtri

from catmodel.contracts.elicitation import QUANTILES, Quantiles

if TYPE_CHECKING:
    from catmodel.contracts.incident import SpreadTailForm

_PROBABILITIES = np.asarray([0.05, 0.25, 0.5, 0.75, 0.95])
_SCORES = ndtri(_PROBABILITIES)
EPISTEMIC_KNOT_SCORES = tuple(float(score) for score in _SCORES)
"""Normal scores of the five elicited quantiles, where the law has its knots."""
EPISTEMIC_TAIL_WIDTH_CAP = float(_SCORES[4])
"""Largest base normal-score tail width, Phi^-1(0.95), before the tail-width factor.

At this width the upper tail's mean integrand ``exp(width*z)*phi(z)`` peaks at
p95's own score, so no unelicited rank beyond p95 dominates a parameter mean.
"""
_OVERFLOW = "epistemic law exceeds finite floating-point representation"


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(_OVERFLOW)
    return value


def _log_ratio(left: float, right: float) -> float:
    relative = (left - right) / right
    return (
        math.log1p(relative)
        if -0.5 < relative < 1
        else math.log(left) - math.log(right)
    )


def _validate_spread(spread: float) -> None:
    if not math.isfinite(spread) or spread <= 0:
        raise ValueError("rank spread factor must be positive and finite")


def _validate_tail_form(tail_form: SpreadTailForm) -> None:
    if tail_form not in ("pure_stretch", "calibrated_p95_peak"):
        raise ValueError(f"unknown spread tail form: {tail_form!r}")


def epistemic_knots(
    estimate: Quantiles, epistemic_tail_scale: float = 1.0
) -> np.ndarray:
    """The five elicited quantiles as the law's knots, after validating them
    and the tail-width factor."""
    if not math.isfinite(epistemic_tail_scale) or epistemic_tail_scale <= 0:
        raise ValueError("epistemic tail scale must be positive and finite")
    values = np.asarray([getattr(estimate, name) for name in QUANTILES], dtype=float)
    if (
        not np.all(np.isfinite(values))
        or np.any(np.diff(values) < 0)
        or np.any(values < 0)
    ):
        raise ValueError("epistemic quantiles must be finite, nonnegative and ordered")
    return values


@lru_cache(maxsize=4096)
def _interpolant(knots: tuple[float, ...]) -> PchipInterpolator:
    return PchipInterpolator(_PROBABILITIES, np.asarray(knots))


def knot_interpolant(values: np.ndarray) -> PchipInterpolator:
    """The central PCHIP through five knots at the elicited ranks, built once
    per ladder; its slopes depend only on the knots."""
    return _interpolant(tuple(float(value) for value in values))


def _tail_width(values: np.ndarray, upper: bool, scale: float) -> float:
    """Capped normal-score secant of the edge pair p75/p95 or p5/p25.

    Zero width is a flat tail: a tied pair continues the flat quantile
    function, and an anchor at zero is an atom there. An inner knot at zero has
    an infinite secant and takes the cap, so the width is continuous as that
    knot approaches zero.
    """
    edge, inner = (4, 3) if upper else (0, 1)
    anchor, neighbour = float(values[edge]), float(values[inner])
    if anchor in (neighbour, 0.0):
        return 0.0
    secant = (
        math.inf
        if neighbour == 0
        else _log_ratio(anchor, neighbour) / float(_SCORES[edge] - _SCORES[inner])
    )
    return scale * min(secant, EPISTEMIC_TAIL_WIDTH_CAP)


def tail_widths(
    values: np.ndarray,
    *,
    epistemic_tail_scale: float = 1.0,
    spread: float = 1.0,
    tail_form: SpreadTailForm = "pure_stretch",
) -> tuple[float, float]:
    """The lower and upper normal-score tail widths of validated knots after
    the stretch.

    With ``f`` the epistemic tail scale and ``a = Phi^-1(0.95)``, each edge
    pair's secant is capped at ``a`` and multiplied by ``f``. The calibrated
    form caps it again at ``f * a / k``, with no additional cap for ``k <= 1``.
    """
    _validate_spread(spread)
    _validate_tail_form(tail_form)
    lower = _tail_width(values, False, epistemic_tail_scale)
    upper = _tail_width(values, True, epistemic_tail_scale)
    if tail_form == "pure_stretch":
        return lower, upper
    cap = epistemic_tail_scale * EPISTEMIC_TAIL_WIDTH_CAP * min(1.0, spread ** (-1))
    return min(lower, cap), min(upper, cap)


def _read(
    values: np.ndarray,
    widths: tuple[float, float],
    scores: np.ndarray,
    ranks: np.ndarray | None = None,
) -> np.ndarray:
    """The law of nonconstant knots at the normal scores ``scores``, a vector.

    Every knot is exact. Beyond p5's or p95's score the value is
    ``q exp(w (score - z_edge))`` in the score itself, combining logs where
    ``exp`` alone would leave the floating-point range, so a representable
    value is neither refused nor flushed to zero; a nondegenerate tail stays
    positive. Between them the central PCHIP reads the rank: ``ranks``, the
    scores' exact ranks, when given, and otherwise ``Phi(score)``, which is
    representable without saturation there.
    """
    if ranks is None:
        lower_tail, upper_tail = scores < _SCORES[0], scores > _SCORES[4]
        knots = [scores == score for score in _SCORES]
    else:
        lower_tail = ranks < _PROBABILITIES[0]
        upper_tail = ranks > _PROBABILITIES[4]
        knots = [ranks == rank for rank in _PROBABILITIES]
    result = np.empty_like(scores)
    central = ~(lower_tail | upper_tail)
    for value, knot in zip(values, knots, strict=True):
        result[knot] = value
        central &= ~knot
    if np.any(central):
        result[central] = knot_interpolant(values)(
            ndtr(scores[central]) if ranks is None else ranks[central]
        )
    for upper, mask in ((False, lower_tail), (True, upper_tail)):
        if not np.any(mask):
            continue
        edge = 4 if upper else 0
        anchor = float(values[edge])
        width = widths[1 if upper else 0]
        if width == 0:
            result[mask] = anchor
            continue
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            excess = width * (scores[mask] - _SCORES[edge])
            tail = anchor * np.exp(excess)
            lost = ~np.isfinite(tail) | (tail == 0)
            tail[lost] = np.exp(math.log(anchor) + excess[lost])
        if not np.all(np.isfinite(tail)):
            raise ValueError(_OVERFLOW)
        result[mask] = np.maximum(tail, np.nextafter(0.0, 1.0))
    return result


def epistemic_quantiles(
    estimate: Quantiles, ranks: np.ndarray, *, epistemic_tail_scale: float = 1.0
) -> np.ndarray:
    """The law at ranks strictly inside (0, 1): five-knot PCHIP with capped
    local normal-score log tails.

    All five elicited quantiles are exact. Central PCHIP uses only those knots;
    derivatives at p5/p95 need not match the tails. Beyond p5 or p95 the log
    value moves linearly in the normal score with the edge pair's secant width,
    capped at ``EPISTEMIC_TAIL_WIDTH_CAP``. A tied edge pair gives a flat tail,
    a zero anchor an atom and a constant ladder a constant. The width is
    continuous in every quantile. These are judged extrapolations, not
    empirical calibration or a claim of unbiasedness.

    ``epistemic_tail_scale`` multiplies only the capped extrapolated widths.
    """
    ranks = np.asarray(ranks, dtype=float)
    if not np.all((ranks > 0) & (ranks < 1)):
        raise ValueError("epistemic ranks must lie strictly inside (0, 1)")
    values = epistemic_knots(estimate, epistemic_tail_scale)
    if np.all(values == values[0]):
        return np.full(ranks.shape, values[0])
    flat = ranks.ravel()
    widths = tail_widths(values, epistemic_tail_scale=epistemic_tail_scale)
    return _read(values, widths, ndtri(flat), flat).reshape(ranks.shape)


def stretched_score_quantiles(
    estimate: Quantiles,
    scores: np.ndarray,
    *,
    spread: float = 1.0,
    epistemic_tail_scale: float = 1.0,
    tail_form: SpreadTailForm = "pure_stretch",
) -> np.ndarray:
    """The law at the stretched scores ``spread * scores``, never through a
    stretched rank.

    The central PCHIP is read through ``Phi`` only between p5 and p95, where
    that probability is representable without saturation. Outside those knots
    the law's local log-linear tails, with the widths of ``tail_form``, are
    evaluated directly in normal-score space.
    """
    _validate_spread(spread)
    _validate_tail_form(tail_form)
    scores = np.asarray(scores, dtype=float)
    if not np.all(np.isfinite(scores)):
        raise ValueError("normal scores must be finite")
    values = epistemic_knots(estimate, epistemic_tail_scale)
    if np.all(values == values[0]):
        return np.full(scores.shape, values[0])
    with np.errstate(over="ignore", invalid="ignore"):
        stretched = np.asarray(scores * spread, dtype=float)
    if not np.all(np.isfinite(stretched)):
        raise ValueError(_OVERFLOW)
    widths = tail_widths(
        values,
        epistemic_tail_scale=epistemic_tail_scale,
        spread=spread,
        tail_form=tail_form,
    )
    return _read(values, widths, stretched.ravel()).reshape(stretched.shape)


def epistemic_upper_tail_mean(estimate: Quantiles, rank: float) -> float:
    """Mean of ``Q(u)`` over ``u`` in ``[rank, 1)``, for ``rank >= 0.95``.

    Beyond p95 the law is ``p95 * exp(w * (z - z95))`` in the normal score
    ``z``, so the integral is closed-form: with ``k = p95 * exp(w^2/2 - w*z95)``
    it is ``k * Phi(w - z)``. A flat tail is its anchor.
    """
    if not 0.95 <= rank < 1:
        raise ValueError("the closed-form tail mean starts at or beyond p95")
    values = epistemic_knots(estimate)
    anchor = float(values[4])
    width = 0.0 if np.all(values == values[0]) else _tail_width(values, True, 1.0)
    if width == 0:
        return anchor
    start = float(ndtri(rank))
    edge = float(_SCORES[4])
    scale = anchor * math.exp(width * width / 2 - width * edge)
    # The upper-tail normal probability in its accurate complementary form.
    integral = scale * float(ndtr(width - start))
    return _finite(integral / (1 - rank))


_UNIT_NODES, _UNIT_WEIGHTS = np.polynomial.legendre.leggauss(4)
_SPREAD_NODES, _SPREAD_WEIGHTS = np.polynomial.legendre.leggauss(16)


def epistemic_mean(
    estimate: Quantiles,
    *,
    epistemic_tail_scale: float = 1.0,
    spread: float = 1.0,
    tail_form: SpreadTailForm = "pure_stretch",
) -> float:
    """Exact mean of the law over its stretched rank, ``E[Q(Phi(k Z))]``.

    ``spread`` reads every rank ``u`` at ``Phi(spread * Phi^-1(u))`` for a
    standard normal ``Z``, with the tail widths of ``tail_form`` (all forms
    coincide at ``k <= 1``). Central PCHIP is cubic on each knot interval, so
    at ``k == 1`` four-point Gauss-Legendre integrates it exactly. Otherwise
    the rank ``p`` has density ``exp((k^2 - 1) Phi^-1(p)^2 / (2 k^2)) / k`` on
    each central knot interval, a smooth weight on a bounded interval, which
    16-point Gauss-Legendre integrates with the cubic to within rounding. Each
    tail is closed form: ``E[q exp(w (k Z - z_edge)); k Z beyond z_edge]`` is
    ``q exp(w^2 k^2 / 2 - w z_edge) Phi(w k - z_edge / k)`` above p95 and the
    mirror image below p5; with ``w = 0`` it is the anchor times the tail
    probability.
    """
    _validate_spread(spread)
    _validate_tail_form(tail_form)
    values = epistemic_knots(estimate, epistemic_tail_scale)
    if np.all(values == values[0]):
        return float(values[0])
    lower_width, upper_width = tail_widths(
        values,
        epistemic_tail_scale=epistemic_tail_scale,
        spread=spread,
        tail_form=tail_form,
    )
    interpolant = knot_interpolant(values)
    if spread == 1:
        nodes, weights = _UNIT_NODES, _UNIT_WEIGHTS
    else:
        nodes, weights = _SPREAD_NODES, _SPREAD_WEIGHTS
    central = 0.0
    for left, right in pairwise(_PROBABILITIES):
        half = (right - left) / 2
        ranks = left + half + half * nodes
        density = (
            1.0
            if spread == 1
            else np.exp((spread * spread - 1) * ndtri(ranks) ** 2 / (2 * spread**2))
            / spread
        )
        central += half * float(weights @ (interpolant(ranks) * density))
    upper_edge, lower_edge = float(_SCORES[4]), float(_SCORES[0])
    upper = float(values[4]) * (
        float(ndtr(-upper_edge / spread))
        if upper_width == 0
        else math.exp(upper_width**2 * spread**2 / 2 - upper_width * upper_edge)
        * float(ndtr(upper_width * spread - upper_edge / spread))
    )
    lower = float(values[0]) * (
        float(ndtr(lower_edge / spread))
        if lower_width == 0
        else math.exp(lower_width**2 * spread**2 / 2 - lower_width * lower_edge)
        * float(ndtr(lower_edge / spread - lower_width * spread))
    )
    return _finite(central + upper + lower)
