"""An independent SciPy oracle of the five-knot epistemic count law.

It follows methodology.md, "Pooling and parameter uncertainty", and uses no
catmodel code: a PCHIP on the original scale through p5, p25, p50, p75 and p95
of the expected count; beyond p95 ``q95 exp(w [Phi^-1(u) - Phi^-1(.95)])`` and
below p5 the same form from q5, with ``w = f min(s, Phi^-1(.95))``, where ``s``
is the edge pair's normal-score secant and ``f`` the tail-width factor. A tied
edge pair or a zero anchor gives ``w = 0``; an inner knot at zero has an
infinite secant. The law is read at the stretched score ``k z``; for ``k > 1``
the calibrated and stated p95-peak tail forms cap the width again at
``f Phi^-1(.95) / k`` and ``f Phi^-1(.95) / k**2``.

Means integrate the reading against the standard normal density with adaptive
quadrature, never with the engine's closed forms.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from scipy import integrate
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

RANKS = (0.05, 0.25, 0.5, 0.75, 0.95)
SCORES = tuple(float(norm.ppf(rank)) for rank in RANKS)
WIDTH_CAP = float(norm.ppf(0.95))
_TAIL_FORM_POWERS = {"pure_stretch": 0, "calibrated_p95_peak": 1}


def _values(quantiles) -> np.ndarray:
    if isinstance(quantiles, Sequence | np.ndarray):
        return np.asarray(quantiles, dtype=float)
    return np.asarray(
        [getattr(quantiles, name) for name in ("p5", "p25", "p50", "p75", "p95")],
        dtype=float,
    )


class CountLawOracle:
    """One count law read at ``spread`` with ``tail_scale`` and ``tail_form``.

    ``quantiles`` is a ``Quantiles`` record or the five values p5-p95.
    """

    def __init__(
        self,
        quantiles,
        *,
        tail_scale: float = 1.0,
        spread: float = 1.0,
        tail_form: str = "pure_stretch",
    ) -> None:
        self.knots = _values(quantiles)
        self.spread = float(spread)
        self.central = PchipInterpolator(RANKS, self.knots)
        power = _TAIL_FORM_POWERS[tail_form]
        cap = math.inf if self.spread <= 1 else tail_scale * WIDTH_CAP / spread**power
        self.lower_width = min(self._width(0, 1, tail_scale), cap)
        self.upper_width = min(self._width(4, 3, tail_scale), cap)

    def _width(self, edge: int, inner: int, tail_scale: float) -> float:
        anchor, neighbour = self.knots[edge], self.knots[inner]
        if anchor in (neighbour, 0.0):
            return 0.0
        secant = (
            math.inf
            if neighbour == 0
            else math.log(anchor / neighbour) / (SCORES[edge] - SCORES[inner])
        )
        return tail_scale * min(secant, WIDTH_CAP)

    def at_stretched(self, x: float) -> float:
        """The law at the already stretched normal score ``x``."""
        if x > SCORES[4]:
            return float(self.knots[4] * math.exp(self.upper_width * (x - SCORES[4])))
        if x < SCORES[0]:
            return float(self.knots[0] * math.exp(self.lower_width * (x - SCORES[0])))
        return float(self.central(norm.cdf(x)))

    def at_score(self, score: float) -> float:
        """The law read at a year's normal score ``score``: ``Q(Phi(k z))``."""
        return self.at_stretched(self.spread * score)

    def quantile(self, rank: float) -> float:
        """``Q(Phi(k Phi^-1(rank)))``; at ``k = 1`` the central PCHIP reads
        ``rank`` itself."""
        if self.spread == 1 and RANKS[0] <= rank <= RANKS[4]:
            return float(self.central(rank))
        return self.at_score(float(norm.ppf(rank)))

    def kinks(self) -> list[float]:
        """Year scores where the reading kinks, and where each tail's mean
        integrand ``exp(w k z) phi(z)`` peaks, ascending. A peak can coincide
        with a kink up to rounding; such points are merged, since a sliver
        between them would only trouble a quadrature."""
        points = sorted(
            [
                *(score / self.spread for score in SCORES),
                self.upper_width * self.spread,
                -self.lower_width * self.spread,
            ]
        )
        merged = [points[0]]
        for point in points[1:]:
            if point - merged[-1] > 1e-9:
                merged.append(point)
        return merged

    def mean(self, low: float = -math.inf, high: float = math.inf) -> float:
        """``E[Q(Phi(k Z)); low <= Z <= high]`` by adaptive quadrature."""
        reach = max(abs(point) for point in self.kinks()) + 40
        low, high = max(low, -reach), min(high, reach)
        if low >= high:
            return 0.0
        inner = [x for x in self.kinks() if low + 1e-9 < x < high - 1e-9]
        points = [low, *inner, high]

        def integrand(score: float) -> float:
            return self.at_score(score) * float(norm.pdf(score))

        return math.fsum(
            integrate.quad(integrand, left, right, limit=400, epsabs=0, epsrel=1e-13)[0]
            for left, right in pairwise(points)
        )
