"""Independent SciPy oracles for the native quantile-spline severity body.

The body interpolates the log excess over the qualifying floor against the
normal score of the probability with SciPy's PCHIP rule; a pinned GPD tail
scale replaces the p95 end slope, limited to three times the last secant. These
oracles cover ladders whose excess over the floor is positive at every knot.
"""

from decimal import Decimal, localcontext

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from scipy.special import ndtr, ndtri

PROBABILITIES = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
SCORES = ndtri(PROBABILITIES)


class SplineOracle:
    def __init__(self, knots, *, floor: float = 0.0, end_slope: float | None = None):
        self.knots = np.asarray(knots, dtype=float)
        self.floor = float(floor)
        excess = self.knots - self.floor
        if np.any(excess <= 0):
            raise ValueError("the oracle covers positive excess knots only")
        log_excess = np.log(excess)
        pchip = PchipInterpolator(SCORES, log_excess)
        if end_slope is None:
            self.spline = pchip
        else:
            slopes = np.r_[pchip.c[2], pchip.derivative()(SCORES[-1])]
            density = np.exp(-0.5 * SCORES[-1] ** 2) / np.sqrt(2 * np.pi)
            secant = (log_excess[-1] - log_excess[-2]) / (SCORES[-1] - SCORES[-2])
            slopes[-1] = min(end_slope * density / excess[-1], 3 * secant)
            self.spline = CubicHermiteSpline(SCORES, log_excess, slopes)

    def quantile(self, u):
        """Body quantile on [0.05, 0.95]."""
        return self.floor + np.exp(self.spline(ndtri(np.asarray(u, dtype=float))))

    def integral(self, lower: float, upper: float) -> float:
        points = [p for p in PROBABILITIES[1:-1] if lower < p < upper]
        return quad(
            lambda u: float(self.quantile(u)),
            lower,
            upper,
            points=points or None,
            limit=400,
            epsabs=0.0,
            epsrel=1e-13,
        )[0]

    def cdf(self, amount: float) -> float:
        """Generalized inverse sup{u: q(u) <= amount} for an amount in the body.

        Inverts the segment's normalized Hermite exactly in 65-digit decimal.
        """
        index = int(np.searchsorted(self.knots, amount, side="right")) - 1
        index = min(max(index, 0), 3)
        step = SCORES[index + 1] - SCORES[index]
        slopes = (
            self.spline.derivative()(SCORES[index]),
            self.spline.derivative()(SCORES[index + 1]),
        )
        with localcontext() as context:
            context.prec = 65
            floor = Decimal.from_float(self.floor)
            low, high, x = (
                Decimal.from_float(float(value)) - floor
                for value in (self.knots[index], self.knots[index + 1], amount)
            )
            width = high.ln() - low.ln()
            target = (x.ln() - low.ln()) / width
            left, right = (
                Decimal.from_float(float(slope * step)) / width for slope in slopes
            )
            lower, upper = Decimal(0), Decimal(1)
            for _ in range(200):
                t = (lower + upper) / 2
                value = t * (left + t * (3 - 2 * left - right + t * (left + right - 2)))
                if value <= target:
                    lower = t
                else:
                    upper = t
            return float(ndtr(SCORES[index] + float(lower) * step))
