import math
import struct
from collections.abc import Callable
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import pairwise
from typing import Any

import numpy as np
import structlog
from scipy.optimize import elementwise
from scipy.special import ndtr, ndtri

from catmodel.scenarios.sensitivity_coverage import (
    UNCAPPED_TAIL_SHAPE_BOUND,
)

logger = structlog.get_logger(__name__)

_EPS = 1e-6


def _combine_severity_caps(*caps: float | None) -> float | None:
    present = [cap for cap in caps if cap is not None]
    return min(present) if present else None


# Two domains bound severity amounts: a wide numerical domain checked by every
# law and amount component, and a narrower admission domain checked where
# evidence enters. ``validate_monetary_amount`` and ``validate_admitted_amount``
# state each bound and its rationale (assumptions.md `NUM-SEVERITY-DOMAIN`).
NUMERIC_AMOUNT_RANGE = (1e-100, 1e100)
ADMITTED_AMOUNT_RANGE = (1e-9, 1e20)
SCALE_FACTOR_RANGE = (1e-12, 1e12)
GPD_SHAPE_RANGE = (-1.0, 5.0)


class SeverityDomainError(ValueError):
    """An amount, factor or parameter lies outside the numerical domain."""


def _check_range(name: str, value: float, bounds: tuple[float, float], domain: str):
    lower, upper = bounds
    if not math.isfinite(value) or not (value == 0.0 or lower <= value <= upper):
        raise SeverityDomainError(
            f"{name}={value} is outside the {domain}: zero or between "
            f"{lower:g} and {upper:g}"
        )


def validate_monetary_amount(name: str, value: float) -> None:
    """Reject a monetary amount outside the numerical domain.

    Every severity law and amount component, including laws rescaled
    internally, checks its monetary amounts (knots, floor, cap, GPD scale and
    fixed amounts, in the law's own currency unit): each is zero or within
    [1e-100, 1e100]. With scale factors zero or within [1e-12, 1e12]
    (``validate_scale_factor``) and the GPD shape in [-1, 5], every product of
    an amount, two factors and a tail quantile at a representable rank (at most
    about 1e80 times the scale for shape 5) stays inside binary64's normal
    range, so no severity evaluator needs underflow, overflow or
    extended-precision handling.
    """
    _check_range(name, value, NUMERIC_AMOUNT_RANGE, "numerical monetary domain")


def validate_admitted_amount(name: str, value: float) -> None:
    """Reject an evidence amount outside the admission domain.

    Respondent loss answers enter with amounts zero or within [1e-9, 1e20]
    currency units. The upper bound leaves ample headroom over the governed
    catastrophe endpoint: it is a million times the USD 1e14 cap and 200,000
    times the largest cap stress (USD 5e14), and lies above
    value-of-statistical-life arithmetic on the whole world population (about
    1e17). The uncapped stress has no endpoint, but its draws are not admitted
    amounts: from an admitted p95 and p99, a GPD of shape at most 1 (0.8 plus
    the 0.2 step) reaches about 1e34 at the largest representable rank. Scaling
    an admitted amount by two factors and any sensitivity multiplier stays at
    least 1e40 below the numerical bound, so internally rescaled laws are not
    re-admitted; a transform that would still leave the numerical domain is
    reported unavailable.
    """
    _check_range(name, value, ADMITTED_AMOUNT_RANGE, "admitted monetary domain")


def validate_scale_factor(name: str, value: float) -> None:
    """Reject a currency, unit or composition factor outside the domain."""
    _check_range(name, value, SCALE_FACTOR_RANGE, "admitted scale-factor domain")


# The rank of the GPD splice, the authored p95: a quantile-spline law cannot
# splice lower without discarding its p95.
_GPD_THRESHOLD_PCT = 0.95
# The rank of the tail-ratio anchor's upper quantile and of the large-loss break.
_GPD_TAIL_RATIO_UPPER_PCT = 0.99


def gpd_scale_from_tail_ratio(
    *, u_thr: float, xi: float, tail_ratio: float, p_thr: float
) -> float:
    """Solve σ so the uncapped q(p99)/q(p_thr) ratio equals ``tail_ratio``.

    For ``v=(p99-p_thr)/(1-p_thr)``, use the closed-form xi≠0 formula and its
    logarithmic xi→0 limit. This anchors near-tail scale; xi still governs the
    deep tail, and a severity cap may clip the realized p99.
    """
    upper = _GPD_TAIL_RATIO_UPPER_PCT
    if not (p_thr < upper < 1.0):
        raise ValueError(
            f"tail-ratio upper percentile {upper} must lie in (p_thr={p_thr}, 1)"
        )
    if not np.isfinite(tail_ratio) or tail_ratio <= 1.0:
        raise ValueError(f"gpd_tail_ratio must be > 1, got {tail_ratio}")
    if not np.isfinite(u_thr) or u_thr <= 0.0:
        raise ValueError(
            f"tail-ratio anchor requires a positive splice value, got u_thr={u_thr}"
        )
    v = (upper - p_thr) / (1.0 - p_thr)
    if abs(xi) < 1e-8:
        sigma = float(u_thr * (tail_ratio - 1.0) / -np.log1p(-v))
    else:
        denom = (1.0 - v) ** (-xi) - 1.0
        sigma = float(xi * u_thr * (tail_ratio - 1.0) / denom)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(
            f"tail-ratio anchor produced a non-positive σ ({sigma}); check "
            f"gpd_tail_ratio={tail_ratio}, gpd_shape={xi}"
        )
    return sigma


LARGE_LOSS_BREAK_EXCEEDANCE = 0.01
"""Exceedance probability of the break above which a large-loss anchor acts.

The break is the law's own p99, where the tail-scale pin sets the survival to
0.01; a large-loss anchor (``SEV-LARGE-LOSS-ANCHOR``) changes the law only
above it, so its exceedance can be at most 0.01: at 0.01 no incident lies
between the break and the anchor amount.
"""


@dataclass(frozen=True)
class _PowerSegment:
    """One power-law piece of an anchored tail (``SEV-LARGE-LOSS-ANCHOR``).

    On ``[start, stop]`` the survival function is ``mass * ((x - floor) /
    (start - floor)) ** -index``, a power law of the excess over the floor
    from ``mass`` at ``start`` to ``mass * exp(-log_ratio)`` at ``stop``, with
    ``log_span = ln((stop - floor) / (start - floor))``. ``offset`` is
    ``ln(0.01 / mass)``: how far the log survival has fallen from the p99
    break's by ``start``. A zero ``log_ratio`` is a flat segment: no incident
    lies inside it. The segment is bounded, so its mean is finite at every
    index, including one at or below one.
    """

    start: float
    stop: float
    mass: float
    log_span: float
    log_ratio: float
    offset: float

    @property
    def index(self) -> float:
        return self.log_ratio / self.log_span


def _power_segment_integral(log_span: float, log_ratio: float) -> float:
    """Integral of ``(e/e0)^(-alpha)`` over ``e`` from ``e0`` to
    ``e0 * exp(log_span)``, in units of ``e0``, with ``alpha * log_span =
    log_ratio``.

    It is ``(exp(log_span - log_ratio) - 1) / (1 - alpha)``, written as
    ``log_span * expm1(z) / z`` with ``z = log_span - log_ratio``, so it stays
    exact through ``alpha = 1`` (``z = 0``, the logarithmic case).
    """
    z = log_span - log_ratio
    if z == 0.0:
        return log_span
    return log_span * math.expm1(z) / z


def p999_implied_gpd_shape(p95: float, p99: float, p999: float) -> float:
    """Return the GPD shape a p95-threshold tail needs to pass p99 and p999.

    Above p95 the exceedance probabilities of p99 and p999 are 1/5 and 1/50 of
    the threshold's, so ``(p999 - p95) / (p99 - p95) = (50**xi - 1) / (5**xi -
    1)``. The ratio is strictly increasing in ``xi`` with limit
    ``log(50) / log(5)`` at zero, so every ratio above one has exactly one
    solution. The result is a model-dependent diagnostic of stated extreme
    quantiles, never authority for an executable tail.
    """
    values = (p95, p99, p999)
    if not all(math.isfinite(value) for value in values) or not (0 < p95 < p99 < p999):
        raise ValueError("implied tail shape requires finite 0 < p95 < p99 < p999")
    target = math.log(p999 - p95) - math.log(p99 - p95)
    log_five, log_fifty = math.log(5.0), math.log(50.0)

    def log_abs_expm1(value: float) -> float:
        if value < 0:
            return math.log(-math.expm1(value))
        if value < 50:
            return math.log(math.expm1(value))
        return value + math.log1p(-math.exp(-value))

    def log_ratio(xi: float) -> float:
        if xi == 0:
            return math.log(log_fifty / log_five)
        # Both expm1 terms share the sign of xi, so the ratio stays positive.
        return log_abs_expm1(xi * log_fifty) - log_abs_expm1(xi * log_five)

    low, high = -1.0, 1.0
    while log_ratio(low) > target:
        low *= 2
    while log_ratio(high) < target:
        high *= 2
    for _ in range(200):
        middle = 0.5 * (low + high)
        if log_ratio(middle) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


# The five authored knots sit at these probabilities. The body spline's
# abscissa is their normal score z = Phi^-1(u).
_KNOT_PROBABILITIES = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
_KNOT_SCORES = ndtri(_KNOT_PROBABILITIES)
_KNOT_SCORE_STEPS = np.diff(_KNOT_SCORES)
_KNOT_SCORE_DENSITIES = np.exp(-0.5 * _KNOT_SCORES**2) / math.sqrt(2.0 * math.pi)
# Gauss-Legendre nodes for the body integral. The integrand is analytic on each
# segment; 48 nodes agree with adaptive quadrature to about 1e-13 relative even
# for ladders spanning fifteen decades.
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(48)
_FLAT, _LOG, _ORIGIN = 0, 1, 2


def _monotone_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Butland PCHIP slopes (SciPy's rule) for nondecreasing ``y``."""
    h = np.diff(x)
    m = np.diff(y) / h
    slopes = np.zeros(len(x))
    if len(x) == 2:
        slopes[:] = m[0]
        return slopes
    for k in range(1, len(x) - 1):
        if m[k - 1] > 0.0 and m[k] > 0.0:
            w1, w2 = 2.0 * h[k] + h[k - 1], h[k] + 2.0 * h[k - 1]
            slopes[k] = (w1 + w2) / (w1 / m[k - 1] + w2 / m[k])
    for index, h0, h1, m0, m1 in (
        (0, h[0], h[1], m[0], m[1]),
        (-1, h[-1], h[-2], m[-1], m[-2]),
    ):
        edge = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
        slopes[index] = edge if m0 > 0.0 and edge > 0.0 else 0.0
    return slopes


def _hermite_shape(t, left, right):
    """Normalized cubic Hermite from (0, 0) to (1, 1) with endpoint slopes."""
    return t * (left + t * (3.0 - 2.0 * left - right + t * (left + right - 2.0)))


class _QuantileSplineBody:
    """Monotone quantile spline through the five knots on [p5, p95].

    The abscissa is the normal score of the probability. Between positive
    excesses over the qualifying floor the ordinate is the log excess, so a
    (shifted) lognormal ladder is reproduced exactly and a floor-preserving
    affine change of the knots changes the law by the same affine map. A
    segment that starts at a zero excess (a zero p5, or a p5 at an inclusive
    floor) interpolates the excess itself from a zero slope; equal knots give a
    flat segment and hence an atom.

    Slopes follow SciPy's PCHIP rule on the log excess. ``end_slope`` (dq/du
    at p95), when given, replaces the PCHIP end slope so the density continues
    into a pinned GPD tail. Every normalized end slope is limited to three
    times the segment's secant, a sufficient (not necessary) condition for a
    monotone cubic Hermite (Fritsch and Carlson, 1980), so a tail scale beyond
    that bound leaves a density step at p95.

    The continuations outside [p5, p95], and a GPD scale set by continuity,
    use the body's end slope, but never less than the smaller of the end
    segment's probability secant and the end slope of the lognormal through
    its two knots. PCHIP's end slope vanishes on a rising segment beside a
    much steeper one; the floor keeps a continuation from nearly holding flat
    there, an atom the knots do not imply, and changes continuously as that
    slope vanishes.
    Above p95 the secant is always the smaller. Below p5 the lognormal slope
    is smaller once p25 exceeds about 3.5 times p5, where the secant would
    raise the power continuation's exponent and pile the lowest ranks at zero.
    """

    def __init__(
        self,
        knots: np.ndarray,
        *,
        floor: float,
        zero_excess: np.ndarray,
        end_slope: float | None = None,
    ) -> None:
        self.knots = np.asarray(knots, dtype=float)
        self.floor = floor
        self.excess = self.knots - floor
        kinds = np.full(4, _LOG)
        kinds[zero_excess[:-1]] = _ORIGIN
        kinds[self.knots[1:] == self.knots[:-1]] = _FLAT
        self.kinds = kinds
        widths = np.zeros(4)
        log = kinds == _LOG
        widths[log] = np.log(self.excess[1:][log] / self.excess[:-1][log])
        origin = kinds == _ORIGIN
        widths[origin] = (self.knots[1:] - self.knots[:-1])[origin]
        self.widths = widths
        positive = np.flatnonzero(~zero_excess)
        log_slopes = np.zeros(5)
        if len(positive) >= 2:
            first = positive[0]
            log_slopes[first:] = _monotone_slopes(
                _KNOT_SCORES[first:], np.log(self.excess[first:])
            )
        left, right = np.zeros(4), np.zeros(4)
        with np.errstate(divide="ignore", invalid="ignore"):
            left[log] = (log_slopes[:-1] * _KNOT_SCORE_STEPS / widths)[log]
            right[log] = (log_slopes[1:] * _KNOT_SCORE_STEPS / widths)[log]
            for index in np.flatnonzero(origin):
                # Continue the following log segment's slope in excess units;
                # with no positive segment beyond, use the secant.
                right[index] = (
                    self.excess[index + 1]
                    * log_slopes[index + 1]
                    * _KNOT_SCORE_STEPS[index]
                    / widths[index]
                    if index + 1 < 4 and len(positive) >= 2
                    else 1.0
                )
            if end_slope is not None and kinds[3] != _FLAT:
                score_slope = end_slope * _KNOT_SCORE_DENSITIES[4]
                if kinds[3] == _LOG:
                    score_slope /= self.excess[4]
                right[3] = score_slope * _KNOT_SCORE_STEPS[3] / widths[3]
        self.left = np.clip(left, 0.0, 3.0)
        self.right = np.clip(right, 0.0, 3.0)
        # Laws with the same fields share one body, so nothing writes to it.
        for array in (
            self.knots,
            self.excess,
            self.kinds,
            self.widths,
            self.left,
            self.right,
        ):
            array.flags.writeable = False

    def _segment_slope(self, index: int, normalized: float, knot: int) -> float:
        """Return dq/du at ``knot`` from a segment's normalized endpoint slope."""
        if self.kinds[index] == _FLAT:
            return 0.0
        slope = normalized * self.widths[index] / _KNOT_SCORE_STEPS[index]
        if self.kinds[index] == _LOG:
            slope *= self.excess[knot]
        return float(slope / _KNOT_SCORE_DENSITIES[knot])

    def _end_slope(self, index: int, normalized: float, knot: int) -> float:
        """dq/du at an end knot, never below the smaller of two chord slopes.

        One is the end segment's probability secant; the other is the slope at
        the knot along its normal-score secant (normalized slope one: the
        lognormal through its two knots on a log-excess segment).
        """
        if self.kinds[index] == _FLAT:
            return 0.0
        step = _KNOT_PROBABILITIES[index + 1] - _KNOT_PROBABILITIES[index]
        secant = float((self.knots[index + 1] - self.knots[index]) / step)
        least = min(secant, self._segment_slope(index, 1.0, knot))
        return max(self._segment_slope(index, normalized, knot), least)

    @property
    def lower_slope(self) -> float:
        return self._end_slope(0, self.left[0], 0)

    @property
    def upper_slope(self) -> float:
        return self._end_slope(3, self.right[3], 4)

    def _rise(self, index: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Return q - knots[index] at fraction t of each segment's score step."""
        shape = _hermite_shape(t, self.left[index], self.right[index])
        widths = self.widths[index]
        with np.errstate(over="ignore", invalid="ignore"):
            return np.where(
                self.kinds[index] == _LOG,
                self.excess[index] * np.expm1(widths * shape),
                np.where(self.kinds[index] == _ORIGIN, widths * shape, 0.0),
            )

    def quantiles(self, u: np.ndarray) -> np.ndarray:
        """Evaluate the body at probabilities in [0.05, 0.95]."""
        z = ndtri(u)
        index = np.clip(np.searchsorted(_KNOT_SCORES, z, side="right") - 1, 0, 3)
        t = (z - _KNOT_SCORES[index]) / _KNOT_SCORE_STEPS[index]
        low, high = self.knots[index], self.knots[index + 1]
        value = np.clip(low + self._rise(index, t), low, high)
        return np.where(t >= 1.0, high, value)

    def integral(self, lower: float, upper: float) -> float:
        """Integrate the body quantile over [lower, upper] within [0.05, 0.95]."""
        terms = []
        for index in range(4):
            start = max(lower, float(_KNOT_PROBABILITIES[index]))
            stop = min(upper, float(_KNOT_PROBABILITIES[index + 1]))
            if start >= stop:
                continue
            terms.append(self.knots[index] * (stop - start))
            if self.kinds[index] == _FLAT:
                continue
            # Knot endpoints reuse the exact knot scores.
            low = _KNOT_SCORES[index] if start > lower else float(ndtri(start))
            high = _KNOT_SCORES[index + 1] if stop < upper else float(ndtri(stop))
            half, middle = (high - low) / 2.0, (high + low) / 2.0
            z = middle + half * _GAUSS_NODES
            t = (z - _KNOT_SCORES[index]) / _KNOT_SCORE_STEPS[index]
            rise = self._rise(np.full(len(z), index), t)
            density = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
            terms.append(half * float(np.dot(_GAUSS_WEIGHTS, rise * density)))
        return math.fsum(terms)

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        """Invert the body for amounts in [knots[0], knots[4])."""
        index = np.clip(np.searchsorted(self.knots, x, side="right") - 1, 0, 3)
        low, high = self.knots[index], self.knots[index + 1]
        log = self.kinds[index] == _LOG
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # log1p of exact knot differences keeps amounts beside a knot precise.
            excess = x - self.floor
            target = np.where(
                log,
                np.log1p((x - low) / self.excess[index]) / self.widths[index],
                (x - low) / (high - low),
            )
            complement = np.where(
                log,
                np.log1p((high - x) / excess) / self.widths[index],
                (high - x) / (high - low),
            )

        def residual(fraction, target, complement, left, right):
            # Equivalent forms retain both endpoints and avoid subtracting
            # almost equal amounts beside a knot.
            return np.where(
                target <= 0.5,
                _hermite_shape(fraction, left, right) - target,
                complement - _hermite_shape(1.0 - fraction, right, left),
            )

        root = elementwise.find_root(
            residual,
            (np.zeros_like(x), np.ones_like(x)),
            args=(target, complement, self.left[index], self.right[index]),
            tolerances={
                "xatol": 0.0,
                "xrtol": 4 * np.finfo(float).eps,
                "fatol": 0.0,
                "frtol": 0.0,
            },
            maxiter=128,
        )
        if not np.all(root.success):
            raise ValueError("severity body CDF inversion did not converge")
        t = root.x
        z = _KNOT_SCORES[index] + t * _KNOT_SCORE_STEPS[index]
        start, stop = _KNOT_PROBABILITIES[index], _KNOT_PROBABILITIES[index + 1]
        return np.where(
            t <= 0.5,
            start + (ndtr(z) - ndtr(_KNOT_SCORES[index])),
            stop - (ndtr(_KNOT_SCORES[index + 1]) - ndtr(z)),
        )


def _gpd_survival(excess: np.ndarray, xi: float) -> np.ndarray:
    """GPD survival of nonnegative excesses in units of the GPD scale."""
    excess = np.maximum(excess, 0.0)
    if abs(xi) < 1e-8:
        return np.exp(-excess)
    base = 1.0 + xi * excess
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.where(base > 0.0, np.exp(-np.log(base) / xi), 0.0)


def _limited_gpd_excess_mean(limit: float | None, xi: float, sigma: float) -> float:
    """Integrate the GPD survival function without subtracting tail mass.

    For negative shapes the support endpoint makes larger limits nonbinding.
    Integrating in loss space also avoids a probability floor inventing mass
    at a remote cap when its exceedance probability rounds to zero.
    """
    if limit is None:
        if xi >= 1.0:
            raise SeverityMomentUnavailable(
                "uncapped GPD tail mean requires gpd_shape < 1.0"
            )
        return sigma / (1.0 - xi)
    if limit <= 0.0:
        return 0.0
    if abs(xi) < 1e-8:
        return -sigma * math.expm1(-limit / sigma)
    if xi < 0.0 and limit >= -sigma / xi:
        return sigma / (1.0 - xi)
    log_base = math.log1p(xi * (limit / sigma))
    if abs(xi - 1.0) < 1e-12:
        return sigma * log_base
    return -sigma / (1.0 - xi) * math.expm1((xi - 1.0) / xi * log_base)


# The two concrete severity laws: a single native law and a linear pool of them.
type SeveritySampler = Distribution | MixtureSeverityDistribution


def severity_minimum(sampler: SeveritySampler) -> float:
    """Return the authored qualifying floor; natural fit support is not a screen."""
    return min(
        0.0 if law.severity_minimum is None else float(law.severity_minimum)
        for law, _ in sampler.active_laws()
    )


def uncapped_infinite_variance_gpd_shapes(
    sampler: SeveritySampler,
) -> tuple[float, ...]:
    """The GPD shapes with infinite variance (at least 0.5) of a law's active
    components that no cap bounds, intrinsic or outer.

    A law's shape is a stated decimal: an authored one, or one moved by the
    tail-shape sensitivity step as a decimal
    (``sensitivity_coverage.shifted_tail_shape``). So it is compared exactly,
    without a tie tolerance, here and where a caller tests these shapes for
    a mean (below 1), and a shape of exactly 0.5 has infinite variance.
    """
    if sampler.severity_cap is not None:
        return ()
    return tuple(
        float(law.gpd_shape)
        for law, _ in sampler.active_laws()
        if law.gpd_shape is not None
        and law.gpd_shape >= 0.5
        and law.severity_cap is None
    )


def severity_lower_support(
    sampler: SeveritySampler, *, apply_severity_cap: bool = True
) -> tuple[float, bool]:
    """Return the exact lower infimum and whether it carries positive mass."""
    if isinstance(sampler, MixtureSeverityDistribution):
        supports = [
            severity_lower_support(component)
            for component, weight in zip(
                sampler.components, sampler.weights, strict=True
            )
            if weight > 0.0
        ]
        lower = min(value for value, _ in supports)
        atom = any(value == lower and has_atom for value, has_atom in supports)
    elif isinstance(sampler, Distribution):
        lower = float(sampler._lower_body_value(np.array([0.0]))[0])
        if sampler.severity_minimum is not None:
            atom = sampler.p5 == sampler.severity_minimum
        elif sampler.p5 == 0.0 and sampler._lower_body_knot > 0.0:
            atom = False
        else:
            atom = sampler._lower_slope == 0.0
    else:
        raise TypeError(f"unsupported severity law {type(sampler).__name__}")
    cap = sampler.severity_cap if apply_severity_cap else None
    if cap is not None and cap <= lower:
        return float(cap), True
    return float(lower), bool(atom)


def _rebuilt(law_type, *args, **kwargs):
    """Build a transformed law; leaving the numerical domain is unavailable."""
    try:
        return law_type(*args, **kwargs)
    except SeverityDomainError as exc:
        raise ValueError(
            f"the transformed severity law leaves the numerical domain: {exc}"
        ) from exc


class SeverityMomentUnavailable(ValueError):
    """The native law is valid but its requested moment is infinite."""


def apply_severity_excess_scale(
    sampler: SeveritySampler, values: np.ndarray, scale: float | np.ndarray
) -> np.ndarray:
    """Evaluate L+k(S-L) without cancellation at the qualifying boundary."""
    minimum = severity_minimum(sampler)
    raw = np.asarray(values, dtype=float)
    excess = np.asarray(scale) * (raw - minimum)
    result = minimum + excess
    return np.where(
        (raw > minimum) & (np.asarray(scale) > 0.0) & (result == minimum),
        np.nextafter(minimum, np.inf),
        result,
    )


def _cap_probabilities(
    sampler: SeveritySampler, threshold: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``P(raw <= threshold)`` and ``P(raw > threshold)`` before the outer cap."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return (
            sampler.cdf(threshold, apply_cap=False),
            sampler.survival(threshold, apply_cap=False),
        )


def sample_affine_capped_severity(
    sampler: SeveritySampler,
    uniforms: np.ndarray,
    *,
    scale: float | np.ndarray = 1.0,
    offset: float | np.ndarray = 0.0,
    cap: float | None = None,
    post_multiplier: float = 1.0,
    _raw_inverse: Callable[[SeveritySampler, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Evaluate min(offset + scale*raw severity, cap) without unused tail draws.

    Component-intrinsic caps remain in the raw law. The sampler's outer cap is
    applied after the affine transform; an explicit cap combines by minimum.
    A final post-multiplier scales the capped amount, while cap-resolved ranks
    never evaluate the raw tail. ``_raw_inverse`` lets a caller reuse cached
    raw inverses of the same law.
    """
    if not math.isfinite(post_multiplier) or post_multiplier < 0:
        raise ValueError("severity post-multiplier must be finite and nonnegative")
    u = np.asarray(uniforms, dtype=float)
    slope = np.broadcast_to(np.asarray(scale, dtype=float), u.shape)
    intercept = np.broadcast_to(np.asarray(offset, dtype=float), u.shape)
    if np.any(~np.isfinite(u)) or np.any((u < 0.0) | (u > 1.0)):
        raise ValueError("severity inverse probabilities must be in [0, 1]")
    if (
        np.any(~np.isfinite(slope))
        or np.any(slope < 0.0)
        or np.any(~np.isfinite(intercept))
    ):
        raise ValueError(
            "severity affine coefficients must be finite with nonnegative scale"
        )
    effective_cap = _combine_severity_caps(sampler.severity_cap, cap)
    if effective_cap is not None and (
        not math.isfinite(effective_cap) or effective_cap < 0.0
    ):
        raise ValueError("severity evaluation cap must be finite and nonnegative")
    if post_multiplier == 0:
        return np.zeros_like(u)
    result = np.array(intercept, copy=True)
    active = slope > 0.0
    if effective_cap is not None:
        with np.errstate(divide="ignore"):
            threshold = (effective_cap - intercept[active]) / slope[active]
        cdf, survival = _cap_probabilities(sampler, threshold)
        ranks = u[active]
        above_cap = (
            (ranks > cdf)
            | ((ranks > 0.999) & (ranks != cdf) & ((1.0 - ranks) < survival))
            | ((ranks == 1.0) & (survival > 0.0))
        )
        capped = np.zeros_like(active)
        capped[active] = above_cap
        result[capped] = effective_cap
        active &= ~capped
    if np.any(active):
        evaluation_scale = slope[active]
        raw = (
            sampler.sample_from_uniforms_uncapped(u[active])
            if _raw_inverse is None
            else _raw_inverse(sampler, u[active])
        )
        minimum = severity_minimum(sampler)
        floor_preserving = intercept[active] == minimum * (1.0 - evaluation_scale)
        result[active] = np.where(
            floor_preserving,
            apply_severity_excess_scale(sampler, raw, evaluation_scale),
            intercept[active] + evaluation_scale * raw,
        )
    if post_multiplier != 1:
        result = result * post_multiplier
        if np.any(active):
            # A zero offset composes the raw draw and both factors directly.
            zero_offset = intercept[active] == 0
            completed = raw * evaluation_scale * post_multiplier
            result[active] = np.where(zero_offset, completed, result[active])
    if effective_cap is not None:
        result = np.minimum(result, effective_cap * post_multiplier)
    return result


@dataclass
class Distribution:
    """Five-quantile severity law.

    The body through p5/p25/p50/p75/p95 is a monotone spline of the log excess
    over the qualifying floor against the normal score (``_QuantileSplineBody``).
    An optional GPD tail begins at the authored p95. An explicit or anchored
    tail scale also sets the body's p95 end slope, so the density is continuous
    at the splice where a monotone body allows and the body between p75 and p95
    moves with the tail scale; without one, the tail scale continues the body's
    end slope. A large-loss anchor (an amount above the law's p99 with its
    exceedance probability) replaces the tail above p99: a power law of the
    excess over the floor up to the anchor amount, then the GPD tail
    conditioned on exceeding that amount and scaled to the anchor's mass
    (``_initialize_large_loss_anchor``); at and below p99 the law is
    unchanged. A second anchor at a higher amount adds a second power law
    between the two amounts and moves the conditioned GPD tail to the second
    (the two-anchor tail); at and below the first anchor the law is unchanged.
    Severity caps censor draws.
    Sampling uses monotone continuations outside [p5, p95], and p5=0 remains a
    support endpoint without adding a 5% atom.
    """

    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    gpd_shape: float | None = None
    severity_cap: float | None = None  # max loss per event (None = uncapped)
    severity_minimum: float | None = None  # qualification floor in severity units
    severity_minimum_strict: bool = False
    gpd_scale: float | None = (
        None  # σ: explicit GPD tail scale (None = body-slope continuity)
    )
    gpd_tail_ratio: float | None = (
        None  # R = q(p99)/q(p95) tail-scale anchor; pins σ (None = no anchor)
    )
    # Large-loss anchor: an amount above the law's p99 and its exceedance
    # probability, at most 0.01. Both or neither; with them the law passes
    # through this point above p99 (``_initialize_large_loss_anchor``).
    large_loss_amount: float | None = None
    large_loss_exceedance: float | None = None
    # Second anchor of the two-anchor tail: a higher amount and its exceedance
    # probability, at most the first anchor's. Both or neither, and only with
    # the first anchor; the law then passes through this point too.
    very_large_loss_amount: float | None = None
    very_large_loss_exceedance: float | None = None

    def __post_init__(self) -> None:
        # A law is built far more often than it differs: every validation of
        # a contract that holds one builds it, and so does every evaluator.
        # What the fields imply is therefore worked out once for each distinct
        # set of fields and shared (``_SHARED_STATE``).
        key = _state_key(self)
        state = None if key is None else _SHARED_STATE.get(key)
        if state is not None:
            vars(self).update(state)
            self._warn_of_uncapped_tail()
            return
        self._initialize()
        if key is not None:
            if len(_SHARED_STATE) >= _SHARED_STATE_LIMIT:
                _SHARED_STATE.clear()
            _SHARED_STATE[key] = {
                name: value
                for name, value in vars(self).items()
                if name not in _FIELD_NAMES
            }

    def _warn_of_uncapped_tail(self) -> None:
        if self.gpd_shape is not None and self.gpd_shape >= UNCAPPED_TAIL_SHAPE_BOUND:
            logger.warning(
                "severity component constructed at or above the uncapped xi "
                "guardrail; compare Monte Carlo and analytic means because "
                "finite-sample recovery depends on tail shape and cap binding",
                gpd_shape=float(self.gpd_shape),
                finite_mean_boundary=1.0,
                severity_cap=(
                    None if self.severity_cap is None else float(self.severity_cap)
                ),
                mean_recovery_context=(
                    "weakly or non-binding caps can converge slowly; a "
                    "body-binding cap can approach 100% recovery"
                ),
            )

    def _initialize(self) -> None:
        """Check the fields and work out what they imply: the body, the tail
        scale and the large-loss tail. Nothing here is changed afterwards, so
        laws with the same fields share it."""
        # The load-time schema delegates native law validation here.
        vals = (self.p5, self.p25, self.p50, self.p75, self.p95)
        labels = ("p5", "p25", "p50", "p75", "p95")
        for name, value in zip(labels, vals):
            if not np.isfinite(value):
                raise ValueError(
                    f"Distribution knot {name} must be finite, got {value}"
                )
        for lo, hi, lo_name, hi_name in zip(vals, vals[1:], labels, labels[1:]):
            if lo > hi:
                raise ValueError(
                    f"Distribution knots must be non-decreasing: "
                    f"{lo_name}={lo} > {hi_name}={hi}"
                )
        if self.p5 < 0.0:
            raise ValueError(f"severity knots must be nonnegative, got p5={self.p5}")
        for name, value in zip(labels, vals):
            validate_monetary_amount(name, value)
        for name in ("severity_cap", "severity_minimum", "gpd_scale"):
            value = getattr(self, name)
            if value is not None and math.isfinite(value) and value > 0.0:
                validate_monetary_amount(name, value)
        if self.gpd_shape is not None and math.isfinite(self.gpd_shape):
            lower, upper = GPD_SHAPE_RANGE
            if not lower <= self.gpd_shape <= upper:
                raise SeverityDomainError(
                    f"gpd_shape={self.gpd_shape} is outside the numerical domain "
                    f"[{lower:g}, {upper:g}]"
                )
        minimum = self.severity_minimum
        if minimum is not None:
            if not np.isfinite(minimum) or minimum < 0.0:
                raise ValueError("severity_minimum must be finite and nonnegative")
            if self.p5 < minimum or (
                self.severity_minimum_strict and self.p5 <= minimum
            ):
                comparison = (
                    "strictly above" if self.severity_minimum_strict else "at least"
                )
                raise ValueError(
                    f"severity p5 must be {comparison} severity_minimum={minimum}"
                )
            if self.severity_cap is not None and self.severity_cap <= minimum:
                raise ValueError("severity_cap must exceed severity_minimum")
        elif self.severity_minimum_strict:
            raise ValueError("severity_minimum_strict requires severity_minimum")
        # Lift a zero p5 knot only internally so zero remains the endpoint
        # without creating a 5% atom.
        lower_body_knot = self.p5
        if self.p5 == 0.0 and self.p25 > 0.0:
            lower_body_knot = max(np.nextafter(0.0, 1.0), self.p25 * _EPS)
        self._lower_body_knot: float = lower_body_knot
        floor = 0.0 if minimum is None else float(minimum)
        authored = np.array(vals, dtype=float)

        def body(end_slope: float | None) -> _QuantileSplineBody:
            return _QuantileSplineBody(
                np.array([lower_body_knot, *vals[1:]], dtype=float),
                floor=floor,
                zero_excess=authored == floor,
                end_slope=end_slope,
            )

        self._body: _QuantileSplineBody = body(None)

        if self.severity_cap is not None:
            if not np.isfinite(self.severity_cap):
                raise ValueError(
                    f"severity_cap must be finite, got {self.severity_cap}"
                )
            if self.severity_cap <= 0.0:
                raise ValueError(
                    f"severity_cap must be strictly positive, got {self.severity_cap}"
                )
            # A governed cap may censor inside the elicited body, even at or
            # below its median. No output records binding depth and nothing
            # blocks such a cap; the assumptions.md `SEV-CAP` alternatives challenge it.

        if self.gpd_shape is not None:
            if not np.isfinite(self.gpd_shape):
                raise ValueError(f"gpd_shape must be finite, got {self.gpd_shape}")
            # A valid raw law need not have a finite mean. Evaluators and
            # requested functionals impose their own moment requirements.
            self._warn_of_uncapped_tail()
            self._gpd_threshold: float = float(self.p95)
            # σ = 0.05 × the body's end dq/du continues its density, except
            # where the end-slope floor binds (``_end_slope``).
            continuity_scale = (1.0 - _GPD_THRESHOLD_PCT) * self._body.upper_slope
            if self.gpd_scale is not None and self.gpd_tail_ratio is not None:
                raise ValueError(
                    "set at most one of gpd_scale (explicit σ) and gpd_tail_ratio "
                    "(tail-scale anchor); they are alternative ways to pin σ"
                )
            if self.gpd_scale is not None:
                if not np.isfinite(self.gpd_scale) or self.gpd_scale <= 0.0:
                    raise ValueError(
                        f"explicit gpd_scale must be finite and positive, got "
                        f"{self.gpd_scale}"
                    )
                self._gpd_scale: float = float(self.gpd_scale)
                self._gpd_scale_source: str = "explicit"
            elif self.gpd_tail_ratio is not None:
                self._gpd_scale = gpd_scale_from_tail_ratio(
                    u_thr=self._gpd_threshold,
                    xi=self.gpd_shape,
                    tail_ratio=self.gpd_tail_ratio,
                    p_thr=_GPD_THRESHOLD_PCT,
                )
                self._gpd_scale_source = "tail_ratio"
            else:
                self._gpd_scale = continuity_scale
                self._gpd_scale_source = "continuity"
            if np.isfinite(self._gpd_scale) and self._gpd_scale > 0.0:
                validate_monetary_amount("GPD scale", self._gpd_scale)
            else:
                if self._gpd_scale_source == "continuity":
                    raise ValueError(
                        "GPD scale must be finite and positive: the body is flat "
                        "at p95, so set gpd_tail_ratio or gpd_scale"
                    )
                raise ValueError("GPD scale must be finite and positive")
            if self._gpd_scale_source != "continuity":
                # A pinned tail scale sets the body's p95 slope instead.
                self._body = body(self._gpd_scale / (1.0 - _GPD_THRESHOLD_PCT))
        else:
            if self.gpd_scale is not None or self.gpd_tail_ratio is not None:
                raise ValueError(
                    "gpd_scale / gpd_tail_ratio require gpd_shape to be set; the "
                    "explicit tail scale only applies when a GPD tail is configured"
                )
            self._gpd_threshold = float(self.p95)
            self._gpd_scale = 0.0  # unused
            self._gpd_scale_source = "none"
        self._lower_slope: float = max(self._body.lower_slope, 0.0)
        self._upper_slope: float = max(self._body.upper_slope, 0.0)
        self._initialize_large_loss_anchor()

    def _initialize_large_loss_anchor(self) -> None:
        """Set the tail above the law's p99 that passes the anchor.

        Without an anchor the break is infinite and the law is unchanged. The
        break b is the first piece's own p99, computed through the same
        inverse as every rank at or below 0.99, so the law there is
        bit-identical to the law without the anchor. With floor m, anchor
        amount A and exceedance pi, the survival function is

        - ``0.01 * ((x - m)/(b - m))^(-alpha)`` on [b, A], a power law of the
          excess over the floor through (b, 0.01) and (A, pi), with
          ``alpha = ln(0.01/pi) / ln((A - m)/(b - m))``; at pi = 0.01 it is
          flat, so no incident lies between b and A;
        - ``pi * S1(x) / S1(A)`` above A, where S1 is the law without the
          anchor: a GPD of the governed shape whose scale is the first
          piece's own scale at A, ``s_A = s1 + xi*(A - p95)``.

        The distribution function is continuous everywhere, and the law is
        continuous in pi, in b and in A: as b approaches A with pi
        approaching 0.01 it approaches the law without the anchor, and with
        pi equal to that law's S1(A) it is that law above A. The density steps
        at b and at A.

        A second anchor (A2, pi2), with A2 > A and 0 < pi2 <= pi, gives the
        two-anchor tail. At and below A the law is the one-anchor law, bit
        for bit; above A the survival function is

        - ``pi * ((x - m)/(A - m))^(-alpha2)`` on [A, A2], a second power law
          of the excess over the floor through (A, pi) and (A2, pi2), with
          ``alpha2 = ln(pi/pi2) / ln((A2 - m)/(A - m))``; at pi2 = pi it is
          flat, so every incident that reaches A reaches A2;
        - ``pi2 * S1(x) / S1(A2)`` above A2: the same GPD shape, with the
          first piece's own scale at A2, ``s_A2 = s1 + xi*(A2 - p95)``.

        Given X > A2 the one-anchor and two-anchor laws are the same GPD, so
        the second anchor changes how many incidents reach A2 and how losses
        between A and A2 are distributed, not how losses beyond A2 are. The
        law is continuous in pi and pi2 and rises with each at every amount
        above b. The segment between the anchors is bounded, so its mean is
        finite at every alpha2, including alpha2 <= 1; the mean beyond A2 is
        ``pi2 * (A2 + s_A2/(1 - xi))`` uncapped, finite exactly when the
        one-anchor law's is. The density also steps at A2.
        """
        self._large_loss_break: float = math.inf
        self._large_loss_floor: float = 0.0
        self._large_loss_segments: tuple[_PowerSegment, ...] = ()
        # The GPD tail beyond the last anchor: where it starts, its mass
        # there, its scale and the log of 0.01 over that mass.
        self._large_loss_tail_amount: float = math.inf
        self._large_loss_tail_mass: float = 0.0
        self._large_loss_tail_scale: float = 0.0
        self._large_loss_tail_offset: float = 0.0
        amount, exceedance = self.large_loss_amount, self.large_loss_exceedance
        higher, share = self.very_large_loss_amount, self.very_large_loss_exceedance
        if amount is None and exceedance is None:
            if higher is not None or share is not None:
                raise ValueError("a second large-loss anchor needs the first anchor")
            return
        if amount is None or exceedance is None:
            raise ValueError(
                "a large-loss anchor needs both its amount and its exceedance"
            )
        if self.gpd_shape is None:
            raise ValueError(
                "a large-loss anchor applies only above an elicited law's GPD tail"
            )
        validate_monetary_amount("large_loss_amount", amount)
        if not (
            math.isfinite(exceedance)
            and 0.0 < exceedance <= LARGE_LOSS_BREAK_EXCEEDANCE
        ):
            raise ValueError(
                "a large-loss anchor's exceedance probability must be positive "
                f"and at most {LARGE_LOSS_BREAK_EXCEEDANCE:g}, got {exceedance}"
            )
        anchors = [(float(amount), float(exceedance))]
        if higher is not None or share is not None:
            if higher is None or share is None:
                raise ValueError(
                    "a second large-loss anchor needs both its amount and its "
                    "exceedance"
                )
            validate_monetary_amount("very_large_loss_amount", higher)
            if not higher > amount:
                raise ValueError(
                    f"a second large-loss anchor amount ({higher:g}) must exceed "
                    f"the first anchor amount ({amount:g})"
                )
            if not (math.isfinite(share) and 0.0 < share <= exceedance):
                raise ValueError(
                    "a second large-loss anchor's exceedance probability must be "
                    f"positive and at most the first anchor's ({exceedance:g}), "
                    f"got {share}"
                )
            anchors.append((float(higher), float(share)))
        break_amount = float(
            self._apply_inverse_cdf(
                np.array([_GPD_TAIL_RATIO_UPPER_PCT]), apply_severity_cap=False
            )[0]
        )
        if not amount > break_amount:
            raise ValueError(
                f"a large-loss anchor amount ({amount:g}) must exceed the law's p99 "
                f"break ({break_amount:g})"
            )
        floor = 0.0 if self.severity_minimum is None else float(self.severity_minimum)
        if not break_amount > floor:
            raise ValueError("a large-loss anchor needs a p99 above the floor")
        # The GPD's threshold stability: given X > A, the first piece is a
        # GPD of the same shape with this scale. For a negative shape it is
        # positive only inside the first piece's support, and falls with A,
        # so the last anchor decides.
        last_amount, last_mass = anchors[-1]
        tail_scale = float(self._gpd_scale) + float(self.gpd_shape) * (
            last_amount - self._gpd_threshold
        )
        if not tail_scale > 0.0:
            raise ValueError(
                f"a large-loss anchor amount ({last_amount:g}) must lie inside the "
                "support of the law's GPD tail"
            )
        validate_monetary_amount("large-loss tail scale", tail_scale)
        segments = []
        start, mass, offset = break_amount, LARGE_LOSS_BREAK_EXCEEDANCE, 0.0
        for stop, stop_mass in anchors:
            log_span = math.log((stop - floor) / (start - floor))
            log_ratio = math.log(mass / stop_mass)
            segments.append(
                _PowerSegment(start, stop, mass, log_span, log_ratio, offset)
            )
            start, mass, offset = stop, stop_mass, offset + log_ratio
        self._large_loss_break = break_amount
        self._large_loss_floor = floor
        self._large_loss_segments = tuple(segments)
        self._large_loss_tail_amount = last_amount
        self._large_loss_tail_mass = last_mass
        self._large_loss_tail_scale = tail_scale
        self._large_loss_tail_offset = offset

    @cached_property
    def large_loss_break(self) -> float | None:
        """The law's own p99, Q(0.99) of its first tail piece, uncapped.

        A large-loss anchor acts only above it, so an anchor amount must exceed
        this computed value, not merely a stated p99: a clipped tail-ratio pin
        or another scale pin can place it elsewhere. None for a law without a
        GPD tail.
        """
        if not math.isinf(self._large_loss_break):
            return self._large_loss_break
        if self.gpd_shape is None:
            return None
        return float(
            self._apply_inverse_cdf(
                np.array([_GPD_TAIL_RATIO_UPPER_PCT]), apply_severity_cap=False
            )[0]
        )

    def _large_loss_kwargs(self, amount_map: Callable[[float], float]) -> dict:
        """The anchors of a transformed copy, their amounts mapped like the
        knots."""
        if self.large_loss_amount is None:
            return {}
        first = {
            "large_loss_amount": amount_map(self.large_loss_amount),
            "large_loss_exceedance": self.large_loss_exceedance,
        }
        if self.very_large_loss_amount is None:
            return first
        return first | {
            "very_large_loss_amount": amount_map(self.very_large_loss_amount),
            "very_large_loss_exceedance": self.very_large_loss_exceedance,
        }

    @property
    def effective_gpd_scale(self) -> float | None:
        return None if self.gpd_shape is None else self._gpd_scale

    @property
    def _carryforward_scale(self) -> float | None:
        """Return a pinned scale for perturbed copies, or None for continuity."""
        if self.gpd_shape is None or self._gpd_scale_source == "continuity":
            return None
        return self._gpd_scale

    def _lower_body_value(self, u: np.ndarray) -> np.ndarray:
        """Continue below p5 on the authored support when supplied.

        An inclusive p5 at its minimum retains an explicit boundary atom;
        otherwise the positive-width lower segment creates no boundary atom.
        A zero p5 is treated as the support endpoint with an internal lift. For
        nonnegative laws, use a power continuation if a linear extension crosses
        zero; negative p5 values remain unfloored to preserve monotonicity.
        """
        if self.severity_minimum is not None:
            minimum = self.severity_minimum
            width = self.p5 - minimum
            if width == 0.0:
                # An inclusive p5 at the boundary explicitly carries an atom.
                return np.full_like(u, minimum)
            fraction = np.clip(u / 0.05, 0.0, 1.0)
            value = minimum + width * fraction
            # Preserve the open interior support when floating-point addition
            # rounds a positive excess to zero. This does not alter any knot.
            return np.where(
                (fraction > 0.0) & (value == minimum),
                np.nextafter(minimum, np.inf),
                value,
            )
        lower_knot = self._lower_body_knot
        if self.p5 == 0.0 and lower_knot > 0.0:
            return lower_knot * np.clip(u / 0.05, 0.0, 1.0)

        value = lower_knot - self._lower_slope * (0.05 - u)
        if lower_knot > 0.0 and self._lower_slope * 0.05 > lower_knot:
            exponent = max(self._lower_slope * 0.05 / lower_knot, 1.0)
            return lower_knot * np.power(np.clip(u / 0.05, 0.0, 1.0), exponent)
        return value

    def _upper_body_value(self, u: np.ndarray) -> np.ndarray:
        """Extend above p95 linearly at the body's end slope when no GPD exists.

        Severity caps censor this law afterward.
        """
        return self.p95 + self._upper_slope * (u - 0.95)

    def _apply_inverse_cdf(
        self,
        u: np.ndarray,
        *,
        apply_severity_cap: bool = True,
    ) -> np.ndarray:
        """Map uniforms through the body, optional GPD tail, and cap.

        The spline body covers [0.05, 0.95]; outer regions use monotone
        continuations, and the severity cap optionally censors results.
        """
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)

        if self.gpd_shape is None:
            result = np.empty(len(u))
            below = u < 0.05
            above = u > 0.95
            body = ~(below | above)
            result[body] = self._body.quantiles(u[body])
            result[below] = self._lower_body_value(u[below])
            result[above] = self._upper_body_value(u[above])
            if apply_severity_cap and self.severity_cap is not None:
                np.minimum(result, self.severity_cap, out=result)
            return result

        result = np.empty(len(u))
        below_mask = u < 0.05
        tail_mask = u >= _GPD_THRESHOLD_PCT
        body_mask = ~(below_mask | tail_mask)

        result[body_mask] = self._body.quantiles(u[body_mask])
        result[below_mask] = self._lower_body_value(u[below_mask])

        large_mask = tail_mask & (u > _GPD_TAIL_RATIO_UPPER_PCT)
        if math.isinf(self._large_loss_break):
            large_mask[:] = False
        tail_mask &= ~large_mask
        if np.any(tail_mask):
            tail_survival = (1.0 - u[tail_mask]) / (1.0 - _GPD_THRESHOLD_PCT)
            with np.errstate(divide="ignore"):
                result[tail_mask] = self._gpd_quantile_from_log_survival(
                    np.log(tail_survival)
                )
        if np.any(large_mask):
            large_survival = (1.0 - u[large_mask]) / LARGE_LOSS_BREAK_EXCEEDANCE
            with np.errstate(divide="ignore"):
                result[large_mask] = self._large_loss_quantile_from_log_survival(
                    np.log(large_survival)
                )

        if apply_severity_cap and self.severity_cap is not None:
            np.clip(result, None, self.severity_cap, out=result)

        return result

    def _gpd_quantile_from_log_survival(self, log_survival: np.ndarray) -> np.ndarray:
        """Evaluate the native GPD tail without rounding a near-one rank."""
        assert self.gpd_shape is not None
        xi, sigma = self.gpd_shape, self._gpd_scale
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            excess = (
                -sigma * log_survival
                if abs(xi) < 1e-8
                else (sigma / xi) * np.expm1(-xi * log_survival)
            )
        return self._gpd_threshold + excess

    def _large_loss_quantile_from_log_survival(
        self, log_survival: np.ndarray
    ) -> np.ndarray:
        """Evaluate the anchored tail at the log of S(x)/0.01, at most zero.

        Within each power-law segment the log excess over the floor is linear
        in the log survival; beyond the last anchor the GPD tail is read at
        the conditional survival, S(x) over the anchor's mass.
        """
        assert self.gpd_shape is not None and self.large_loss_amount is not None
        floor = self._large_loss_floor
        result = np.empty_like(log_survival)
        remaining = np.ones(log_survival.shape, dtype=bool)
        for segment in self._large_loss_segments:
            within = remaining & ~(log_survival < -(segment.offset + segment.log_ratio))
            if np.any(within):
                fraction = (
                    np.clip(
                        (-log_survival[within] - segment.offset) / segment.log_ratio,
                        0.0,
                        1.0,
                    )
                    if segment.log_ratio > 0.0
                    else np.zeros(np.count_nonzero(within))
                )
                values = floor + (segment.start - floor) * np.exp(
                    fraction * segment.log_span
                )
                result[within] = np.clip(values, segment.start, segment.stop)
            remaining &= ~within
        if np.any(remaining):
            conditional = log_survival[remaining] + self._large_loss_tail_offset
            xi, sigma = self.gpd_shape, self._large_loss_tail_scale
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                excess = (
                    -sigma * conditional
                    if abs(xi) < 1e-8
                    else (sigma / xi) * np.expm1(-xi * conditional)
                )
            result[remaining] = self._large_loss_tail_amount + excess
        return result

    def _large_loss_survival(self, amounts: np.ndarray) -> np.ndarray:
        """S(x) of the anchored tail at amounts above the p99 break."""
        assert self.gpd_shape is not None and self.large_loss_amount is not None
        floor = self._large_loss_floor
        result = np.empty_like(amounts)
        remaining = np.ones(amounts.shape, dtype=bool)
        for segment in self._large_loss_segments:
            within = remaining & (amounts < segment.stop)
            if np.any(within):
                log_excess = np.log1p(
                    (amounts[within] - segment.start) / (segment.start - floor)
                )
                result[within] = segment.mass * np.exp(-segment.index * log_excess)
            remaining &= ~within
        if np.any(remaining):
            result[remaining] = self._large_loss_tail_mass * _gpd_survival(
                (amounts[remaining] - self._large_loss_tail_amount)
                / self._large_loss_tail_scale,
                self.gpd_shape,
            )
        return result

    def _large_loss_integral(self, cap: float | None) -> float:
        """Integral of the anchored survival function from the p99 break to
        the cap (unbounded without one), in closed form: each power-law
        segment up to the cap, then the GPD tail beyond the last anchor."""
        assert self.gpd_shape is not None and self.large_loss_amount is not None
        floor = self._large_loss_floor
        pieces = []
        for segment in self._large_loss_segments:
            scale = segment.mass * (segment.start - floor)
            if cap is not None and cap < segment.stop:
                log_span = math.log1p((cap - segment.start) / (segment.start - floor))
                pieces.append(
                    scale * _power_segment_integral(log_span, segment.index * log_span)
                )
                return math.fsum(pieces)
            pieces.append(
                scale * _power_segment_integral(segment.log_span, segment.log_ratio)
            )
        beyond = _limited_gpd_excess_mean(
            None if cap is None else cap - self._large_loss_tail_amount,
            float(self.gpd_shape),
            self._large_loss_tail_scale,
        )
        pieces.append(self._large_loss_tail_mass * beyond)
        return math.fsum(pieces)

    def sample_from_uniforms(self, u: np.ndarray) -> np.ndarray:
        result = self._apply_inverse_cdf(u)
        self._check_inverse_range(u, result)
        return result

    def sample_from_uniforms_uncapped(self, u: np.ndarray) -> np.ndarray:
        """Transform uniforms through the law before the authored cap."""
        result = self._apply_inverse_cdf(u, apply_severity_cap=False)
        self._check_inverse_range(u, result)
        return result

    @staticmethod
    def _check_inverse_range(u: np.ndarray, result: np.ndarray) -> None:
        probabilities = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        if np.any(~np.isfinite(probabilities)):
            raise ValueError("severity inverse probabilities must be finite")
        if np.any(~np.isfinite(result) & (probabilities < 1.0)):
            raise FloatingPointError(
                "severity inverse exceeds floating-point range at an interior probability"
            )

    def quantile(self, p: float) -> float:
        if not np.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"quantile probability must be in [0, 1], got {p}")
        return float(self.sample_from_uniforms(np.array([p], dtype=float))[0])

    def active_laws(self) -> list[tuple["Distribution", float]]:
        """The law as a pool of one component of weight one."""
        return [(self, 1.0)]

    def cdf(self, thresholds: np.ndarray, *, apply_cap: bool = True) -> np.ndarray:
        """``P(X <= x)``, including quantile plateaus and censored atoms;
        ``apply_cap=False`` reads the law before its cap."""
        return self._probabilities(thresholds, survival=False, apply_cap=apply_cap)

    def survival(self, thresholds: np.ndarray, *, apply_cap: bool = True) -> np.ndarray:
        """``P(X > x)``, evaluated directly in the tails for precision;
        ``apply_cap=False`` reads the law before its cap."""
        return self._probabilities(thresholds, survival=True, apply_cap=apply_cap)

    def _probabilities(
        self, thresholds: np.ndarray, *, survival: bool, apply_cap: bool
    ) -> np.ndarray:
        x = np.asarray(thresholds, dtype=float)
        knots = np.array(
            [self._lower_body_knot, self.p25, self.p50, self.p75, self.p95]
        )
        result = np.zeros_like(x)
        below = x < knots[0]
        if self.severity_minimum is not None:
            minimum = self.severity_minimum
            width = self.p5 - minimum
            if width > 0.0:
                result[below] = 0.05 * np.clip((x[below] - minimum) / width, 0.0, 1.0)
            else:
                result[below] = np.where(x[below] >= minimum, 0.05, 0.0)
        elif self.p5 == 0.0 and self._lower_body_knot > 0.0:
            result[below] = 0.05 * np.clip(x[below] / self._lower_body_knot, 0.0, 1.0)
        elif self._lower_slope > 0.0:
            if knots[0] > 0.0 and self._lower_slope * 0.05 > knots[0]:
                exponent = self._lower_slope * 0.05 / knots[0]
                result[below] = 0.05 * np.power(
                    np.clip(x[below] / knots[0], 0.0, 1.0), 1.0 / exponent
                )
            else:
                result[below] = np.clip(
                    0.05 - (knots[0] - x[below]) / self._lower_slope, 0.0, 0.05
                )
        top_x = self._gpd_threshold if self.gpd_shape is not None else self.p95
        body = (x >= knots[0]) & (x < top_x)
        if np.any(body):
            result[body] = self._body.probabilities(x[body])
        above = x >= top_x
        if self.gpd_shape is not None:
            result[above] = 1.0 - (1.0 - _GPD_THRESHOLD_PCT) * _gpd_survival(
                (x[above] - top_x) / self._gpd_scale, self.gpd_shape
            )
        elif self._upper_slope == 0.0:
            result[above] = 1.0
        else:
            result[above] = np.minimum(
                1.0, 0.95 + (x[above] - top_x) / self._upper_slope
            )
        large = x > self._large_loss_break
        if np.any(large):
            large_survival = self._large_loss_survival(x[large])
            result[large] = 1.0 - large_survival
        if survival:
            result = 1.0 - result
            if self.gpd_shape is not None:
                result[above] = (1.0 - _GPD_THRESHOLD_PCT) * _gpd_survival(
                    (x[above] - top_x) / self._gpd_scale, self.gpd_shape
                )
            if np.any(large):
                result[large] = large_survival
        result = np.clip(result, 0.0, 1.0)
        if apply_cap and self.severity_cap is not None:
            result = np.where(x >= self.severity_cap, 0.0 if survival else 1.0, result)
        return result

    def survival_probability(self, threshold: float) -> float:
        """Return strict exceedance, retaining atoms and analytic tail precision."""
        if not math.isfinite(threshold):
            raise ValueError(f"survival threshold must be finite, got {threshold}")
        return float(self.survival(np.array([threshold]))[0])

    def _body_quantile_integral(self, probability: float) -> float:
        """Integrate the spline body and its authored continuations."""
        lower_p = min(probability, 0.05)
        lower = self._lower_body_knot
        if self.severity_minimum is not None:
            minimum = self.severity_minimum
            integral = minimum * lower_p + (self.p5 - minimum) * lower_p**2 / 0.1
        elif self.p5 == 0.0 and lower > 0.0:
            integral = lower * lower_p**2 / 0.1
        elif lower > 0.0 and self._lower_slope * 0.05 > lower:
            exponent = self._lower_slope * 0.05 / lower
            integral = (
                lower * 0.05 / (exponent + 1.0) * (lower_p / 0.05) ** (exponent + 1.0)
            )
        else:
            integral = (
                lower - 0.05 * self._lower_slope
            ) * lower_p + 0.5 * self._lower_slope * lower_p**2
        if probability <= 0.05:
            return integral
        integral += self._body.integral(0.05, min(probability, 0.95))
        if probability <= 0.95:
            return integral
        width = probability - 0.95
        return integral + self.p95 * width + 0.5 * self._upper_slope * width**2

    def analytic_mean(
        self,
        *,
        severity_cap_override: float | None = None,
        apply_severity_cap: bool = True,
    ) -> float:
        """Return the law-specific mean of the capped modeled distribution.

        GPD tails use exact integrals; the spline body uses Gauss-Legendre
        quadrature at double precision.
        Own and override caps combine by minimum. To integrate a cap applied
        after a runtime multiplier a, use apply_severity_cap=False and
        severity_cap_override=cap/a, then multiply the result by a. This
        preserves the multiply-then-cap order even for multipliers below one.
        """
        cap = _combine_severity_caps(
            self.severity_cap if apply_severity_cap else None,
            severity_cap_override,
        )
        if cap is not None and (self.gpd_shape is None or cap <= self._gpd_threshold):
            # A cap inside the body: integrate up to its rank and hold the
            # remaining mass at the cap.
            cap_probability = float(self.cdf(np.array([cap]), apply_cap=False)[0])
            return (
                self._body_quantile_integral(cap_probability)
                + (1.0 - cap_probability) * cap
            )
        if self.gpd_shape is None:
            return self._body_quantile_integral(1.0)
        body_integral = self._body_quantile_integral(_GPD_THRESHOLD_PCT)

        u_thr = float(self._gpd_threshold)
        tail_width = 1.0 - _GPD_THRESHOLD_PCT
        xi = float(self.gpd_shape)
        large_break = self._large_loss_break
        if (cap is not None and cap <= large_break) or math.isinf(large_break):
            excess_limit = None if cap is None else cap - u_thr
            tail_mean = u_thr + _limited_gpd_excess_mean(
                excess_limit, xi, float(self._gpd_scale)
            )
        else:
            # Integrate the survival function piecewise: the first piece up to
            # the p99 break, then the anchored tail above it.
            first = u_thr + _limited_gpd_excess_mean(
                large_break - u_thr, xi, float(self._gpd_scale)
            )
            return body_integral + tail_width * first + self._large_loss_integral(cap)
        return body_integral + tail_width * tail_mean

    def with_gpd_shape(self, shape: float) -> "Distribution":
        """Set the raw tail shape, retaining the declared scale or tail-ratio anchor."""
        if self.gpd_shape is None:
            raise ValueError("an absolute GPD shape requires an existing GPD tail")
        kwargs: dict[str, Any] = {}
        if self._gpd_scale_source == "tail_ratio":
            # Preserve the p99/p95 anchor while varying xi to isolate deep-tail
            # behavior.
            assert self.gpd_tail_ratio is not None
            kwargs["gpd_tail_ratio"] = self.gpd_tail_ratio
        elif self._gpd_scale_source == "explicit":
            # Hold explicit sigma fixed; continuity sigma is re-derived from the
            # unchanged body slope.
            kwargs["gpd_scale"] = self._gpd_scale
        return _rebuilt(
            Distribution,
            p5=self.p5,
            p25=self.p25,
            p50=self.p50,
            p75=self.p75,
            p95=self.p95,
            gpd_shape=shape,
            severity_cap=self.severity_cap,
            severity_minimum=self.severity_minimum,
            severity_minimum_strict=self.severity_minimum_strict,
            **kwargs,
            **self._large_loss_kwargs(float),
        )

    def _with_affine_severity(
        self, offset: float, scale: float, *, intrinsic_cap: bool = False
    ) -> "Distribution":
        """Apply one monotone affine change, retaining the nominal floor.

        An intrinsic cap moves with the law; otherwise the cap stays nominal.
        """
        if scale <= 0.0:
            raise ValueError("severity scaling would collapse the law to the minimum")
        knots = apply_severity_excess_scale(
            self, np.array([self.p5, self.p25, self.p50, self.p75, self.p95]), scale
        )
        return _rebuilt(
            Distribution,
            p5=knots[0],
            p25=knots[1],
            p50=knots[2],
            p75=knots[3],
            p95=knots[4],
            gpd_shape=self.gpd_shape,
            severity_cap=None
            if self.severity_cap is None
            else offset + scale * self.severity_cap
            if intrinsic_cap
            else self.severity_cap,
            severity_minimum=self.severity_minimum,
            severity_minimum_strict=self.severity_minimum_strict,
            gpd_scale=None
            if self._carryforward_scale is None
            else self._carryforward_scale * scale,
            **self._large_loss_kwargs(
                lambda amount: float(
                    apply_severity_excess_scale(self, np.array([amount]), scale)[0]
                )
            ),
        )

    def with_currency_scale(self, factor: float, *factors: float) -> "Distribution":
        """Multiply every monetary input by the currency factors in order."""
        for value in (factor, *factors):
            if value <= 0.0:
                raise ValueError("currency scale factor must be finite and positive")
            validate_scale_factor("currency scale factor", value)

        def scaled(value: float) -> float:
            return math.prod((value, factor, *factors))

        return _rebuilt(
            Distribution,
            *(
                scaled(value)
                for value in (self.p5, self.p25, self.p50, self.p75, self.p95)
            ),
            gpd_shape=self.gpd_shape,
            severity_cap=None
            if self.severity_cap is None
            else scaled(self.severity_cap),
            severity_minimum=None
            if self.severity_minimum is None
            else scaled(self.severity_minimum),
            severity_minimum_strict=self.severity_minimum_strict,
            gpd_scale=None
            if self._carryforward_scale is None
            else scaled(self._carryforward_scale),
            **self._large_loss_kwargs(scaled),
        )

    def with_excess_scale(self, scale: float) -> "Distribution":
        """Scale every raw amount's excess over the qualifying floor, ``L +
        scale*(X - L)``, keeping the floor and this law's own cap, its outer
        cap, nominal; without a floor this is proportional scaling. As a
        component of a mixture its cap moves with its amounts
        (``MixtureSeverityDistribution.with_excess_scale``)."""
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("severity excess scale must be finite and positive")
        if scale == 1.0:
            return self
        return self._with_affine_severity(severity_minimum(self) * (1.0 - scale), scale)

    def component_gpd_shapes(self) -> list[float]:
        return [] if self.gpd_shape is None else [float(self.gpd_shape)]

    def rank_breaks(self) -> tuple[float, ...]:
        """Ranks that split a quadrature of the quantile function: the five
        knots, the p99 large-loss break and each anchor's rank."""
        anchors = (self.large_loss_exceedance, self.very_large_loss_exceedance)
        return (
            0.05,
            0.25,
            0.5,
            0.75,
            0.95,
            0.99,
            *(1.0 - anchor for anchor in anchors if anchor is not None),
        )


_FIELD_NAMES = frozenset(row.name for row in fields(Distribution))
_NUMBER_FIELDS = tuple(
    row.name for row in fields(Distribution) if row.name != "severity_minimum_strict"
)
_NUMBERS = struct.Struct(f"{len(_NUMBER_FIELDS)}d")
_SHARED_STATE: dict[tuple, dict[str, Any]] = {}
"""What ``Distribution._initialize`` worked out, by the law's fields
(``_state_key``). The checks and the body's splines depend on the fields
alone and cost far more than the lookup, and a study builds the same law
many thousands of times."""
_SHARED_STATE_LIMIT = 1 << 16
"""Laws kept in ``_SHARED_STATE``, about a kilobyte and a half each; when it
is full it is emptied, and laws are worked out again as they recur."""


def _state_key(law: Distribution) -> tuple | None:
    """``law``'s fields, each number to the bit, so that two laws share state
    only when every field is the same value. None when a number field holds
    anything but a float or nothing: such a law shares nothing, and is worked
    out, or refused, by itself."""
    numbers = []
    absent = 0
    for index, name in enumerate(_NUMBER_FIELDS):
        value = getattr(law, name)
        if value is None:
            absent |= 1 << index
            numbers.append(0.0)
        elif isinstance(value, float):
            numbers.append(value)
        else:
            return None
    return _NUMBERS.pack(*numbers), absent, bool(law.severity_minimum_strict)


@dataclass
class MixtureSeverityDistribution:
    """Linear-opinion-pool severity sampler with per-component GPD tails."""

    components: list[Distribution]
    weights: np.ndarray
    severity_cap: float | None = None

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError(
                "MixtureSeverityDistribution requires at least one component"
            )
        weights = np.asarray(self.weights, dtype=float)
        if weights.shape != (len(self.components),):
            raise ValueError(
                "mixture weights length must match components length: "
                f"{len(weights)} != {len(self.components)}"
            )
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("mixture weights must be finite and non-negative")
        total = math.fsum(weights)
        if total <= 0.0:
            raise ValueError("mixture weights must sum to a positive value")
        self.weights = weights / total
        active_populations = {
            (component.severity_minimum, component.severity_minimum_strict)
            for component, weight in zip(self.components, self.weights, strict=True)
            if weight > 0.0
        }
        if len(active_populations) != 1:
            raise ValueError(
                "active severity mixture components must share one authored qualifying minimum and comparison"
            )
        # As for a single law, a cap may censor inside the elicited body.
        if self.severity_cap is not None:
            if not np.isfinite(self.severity_cap) or self.severity_cap <= 0.0:
                raise ValueError(
                    f"severity_cap must be finite and positive, got {self.severity_cap}"
                )
            validate_monetary_amount("severity_cap", self.severity_cap)

    @property
    def severity_minimum_strict(self) -> bool:
        return any(
            component.severity_minimum_strict
            for component, weight in zip(self.components, self.weights, strict=True)
            if weight > 0.0
        )

    def active_laws(self) -> list[tuple[Distribution, float]]:
        """The positive-weight components and their weights, equal components
        merged."""
        laws: list[tuple[Distribution, float]] = []
        for component, weight in zip(self.components, self.weights, strict=True):
            if weight == 0.0:
                continue
            for index, (law, mass) in enumerate(laws):
                if component == law:
                    laws[index] = law, math.fsum((mass, float(weight)))
                    break
            else:
                laws.append((component, float(weight)))
        return laws

    @staticmethod
    def _mix(
        laws: list[tuple[Distribution, float]], x: np.ndarray, *, survival: bool
    ) -> np.ndarray:
        """The laws' distribution or survival functions at ``x``, mixed by
        weight; each law keeps its own intrinsic cap."""
        total = np.zeros(x.shape)
        for law, weight in laws:
            total = total + weight * (law.survival(x) if survival else law.cdf(x))
        return total

    def cdf(self, thresholds: np.ndarray, *, apply_cap: bool = True) -> np.ndarray:
        """``P(X <= x)``: the active components' distribution functions, each
        with its own intrinsic cap, mixed by weight. The outer cap puts every
        amount at or above it at the cap; ``apply_cap=False`` reads the pool
        before it."""
        x = np.asarray(thresholds, dtype=float)
        result = self._mix(self.active_laws(), x, survival=False)
        if apply_cap and self.severity_cap is not None:
            result = np.where(x >= self.severity_cap, 1.0, result)
        return result

    def survival(self, thresholds: np.ndarray, *, apply_cap: bool = True) -> np.ndarray:
        """``P(X > x)`` from the components' own survival functions, which keep
        their tail precision; the outer cap as in ``cdf``."""
        x = np.asarray(thresholds, dtype=float)
        result = self._mix(self.active_laws(), x, survival=True)
        if apply_cap and self.severity_cap is not None:
            result = np.where(x >= self.severity_cap, 0.0, result)
        return result

    def rank_breaks(self) -> tuple[float, ...]:
        """Each active component's ``rank_breaks`` at its share of the pooled
        rank, and the shares' boundaries."""
        breaks: set[float] = set()
        start = 0.0
        for component, weight in self.active_laws():
            breaks.add(start)
            breaks.update(start + weight * rank for rank in component.rank_breaks())
            start += weight
        return tuple(sorted(breaks))

    def sample_from_uniforms(self, u: np.ndarray) -> np.ndarray:
        return sample_affine_capped_severity(self, u)

    def sample_from_uniforms_uncapped(
        self, u: np.ndarray, *, _allow_zero_atom: bool = True
    ) -> np.ndarray:
        """Invert the component CDF mixture; retain intrinsic component caps."""
        probabilities = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        laws = self.active_laws()
        if _allow_zero_atom:
            shortcut = self._zero_atom_inverse(probabilities, laws)
            if shortcut is not None:
                return shortcut
        if len(laws) == 1:
            return laws[0][0].sample_from_uniforms(probabilities)
        supports = sorted(
            [
                (law, mass, *law.sample_from_uniforms(np.array([0.0, 1.0])))
                for law, mass in laws
            ],
            key=lambda row: (row[2], row[3]),
        )
        if all(
            math.isfinite(low) and math.isfinite(high) for _, _, low, high in supports
        ) and all(first[3] <= second[2] for first, second in pairwise(supports)):
            # Ordered disjoint intrinsic supports identify the same pooled rank
            # exactly; the mixture's outer cap does not establish eligibility.
            weights = [row[1] for row in supports]
            cumulative = np.array(
                [math.fsum(weights[: index + 1]) for index in range(len(weights))]
            )
            indices = np.minimum(
                np.searchsorted(cumulative, probabilities, side="left"),
                len(supports) - 1,
            )
            indices = np.where(probabilities == 1, len(supports) - 1, indices)
            output = np.empty_like(probabilities)
            for index, (law, weight, _low, _high) in enumerate(supports):
                selected = indices == index
                previous = 0.0 if index == 0 else cumulative[index - 1]
                ranks = (probabilities[selected] - previous) / weight
                ranks = np.where(probabilities[selected] == 1, 1.0, ranks)
                output[selected] = law.sample_from_uniforms(ranks)
            return output
        bounds = np.asarray(
            [component.sample_from_uniforms(probabilities) for component, _ in laws]
        )
        lo, hi = np.min(bounds, axis=0), np.max(bounds, axis=0)
        endpoint = probabilities == 1.0
        finite = (lo < hi) & ~endpoint
        output = hi.copy()
        if not np.any(finite):
            return output

        def cdf_difference(values, targets):
            upper = targets > 0.5
            result = np.empty_like(values)
            if np.any(upper):
                result[upper] = (1.0 - targets[upper]) - self._mix(
                    laws, values[upper], survival=True
                )
            if np.any(~upper):
                result[~upper] = (
                    self._mix(laws, values[~upper], survival=False) - targets[~upper]
                )
            return result

        # The pool before its outer cap, as ``cdf(..., apply_cap=False)``.
        lower_atom = np.zeros_like(finite)
        lower_cdf = self._mix(laws, lo[finite], survival=False)
        preceding_cdf = self._mix(
            laws, np.nextafter(lo[finite], -np.inf), survival=False
        )
        lower_atom[finite] = (
            (lower_cdf >= probabilities[finite])
            & (preceding_cdf < probabilities[finite])
        ) | (probabilities[finite] == 0.0)
        output[lower_atom] = lo[lower_atom]
        finite &= ~lower_atom
        if not np.any(finite):
            return output
        p = probabilities[finite]
        root = elementwise.find_root(
            cdf_difference,
            (lo[finite], hi[finite]),
            args=(p,),
            tolerances={
                "xatol": 0.0,
                "xrtol": 4 * np.finfo(float).eps,
                "fatol": 0.0,
                "frtol": 0.0,
            },
            maxiter=128,
        )
        left, right = root.bracket
        fleft, fright = root.f_bracket
        # At a CDF plateau an arbitrary root is not the generalized inverse.
        # Resolve the lower boundary using the monotone CDF predicate.
        plateau = (fleft == 0.0) | (fright == 0.0)
        if np.any(plateau):
            a = lo[finite][plateau].copy()
            b = np.where(fleft[plateau] == 0.0, left[plateau], right[plateau])
            target = p[plateau]
            for _ in range(64):
                mid = a + (b - a) / 2
                below = cdf_difference(mid, target) < 0.0
                a = np.where(below, mid, a)
                b = np.where(below, b, mid)
            right[plateau] = b
        failed = ~root.success & ~plateau
        if np.any(failed):
            raise ValueError("severity mixture inverse did not converge")
        output[finite] = right
        return output

    def _zero_atom_inverse(self, probabilities, laws):
        """Invert a zero atom plus one strictly increasing native GPD law.

        This is the same pooled rank, not a component-selection draw. In the
        tail, its conditional survival is (1-u)/positive_mass. Keeping that
        survival directly avoids catastrophic cancellation near u=1. The tail
        here is one GPD piece, so a law whose large-loss anchor replaces the
        tail above p99 keeps the general inverse.
        """
        if len(laws) != 2:
            return None
        for index, (zero, _zero_mass) in enumerate(laws):
            if (
                zero.gpd_shape is not None
                or not zero.p5 == zero.p25 == zero.p50 == zero.p75 == zero.p95 == 0
                or zero.severity_minimum not in (None, 0)
            ):
                continue
            positive, mass = laws[1 - index]
            if (
                positive.gpd_shape is None
                or positive.gpd_shape <= -1e-8
                or positive.large_loss_amount is not None
                or positive.severity_cap is not None
                or not positive.p5
                < positive.p25
                < positive.p50
                < positive.p75
                < positive.p95
                or positive.quantile(0) < 0
            ):
                return None
            break
        else:
            return None
        zero_mass = laws[index][1]

        # The general solver uses these unconditional component quantiles as
        # its bracket. Preserve its interior floating-point range rejection.
        positive.sample_from_uniforms(probabilities)
        output = np.zeros_like(probabilities)
        active = (probabilities > zero_mass) | (probabilities == 1)
        ranks = probabilities[active]
        survival = (1 - ranks) / mass
        conditional = np.where(ranks > 0.5, 1 - survival, (ranks - zero_mass) / mass)
        tail = survival <= 1 - _GPD_THRESHOLD_PCT
        values = np.empty_like(ranks)
        if np.any(~tail):
            values[~tail] = positive.sample_from_uniforms(conditional[~tail])
        if np.any(tail):
            with np.errstate(divide="ignore"):
                log_survival = (
                    np.log1p(-ranks[tail])
                    - math.log(mass)
                    - math.log1p(-_GPD_THRESHOLD_PCT)
                )
            values[tail] = positive._gpd_quantile_from_log_survival(log_survival)
        output[active] = values

        # Near an atom or splice, preserve the existing rounded-CDF boundary
        # convention by using the general inverse on only that narrow band.
        boundary = (
            active
            & (probabilities < 1)
            & (probabilities - zero_mass <= 64 * np.spacing(zero_mass))
        )
        for knot in (0.05, 0.25, 0.5, 0.75, 0.95):
            pooled = zero_mass + mass * knot
            boundary |= (probabilities < 1) & (
                np.abs(probabilities - pooled) <= 64 * np.spacing(pooled)
            )
        if np.any(boundary):
            output[boundary] = self.sample_from_uniforms_uncapped(
                probabilities[boundary], _allow_zero_atom=False
            )
        Distribution._check_inverse_range(probabilities, output)
        return output

    def analytic_mean(
        self,
        *,
        severity_cap_override: float | None = None,
        apply_severity_cap: bool = True,
    ) -> float:
        """Linear-pool mean with intrinsic component and optional outer caps.

        Ignoring the outer cap supports integration before runtime scaling;
        component-intrinsic caps remain in the mixture's conditional law.
        """
        cap = _combine_severity_caps(
            self.severity_cap if apply_severity_cap else None, severity_cap_override
        )
        return float(
            sum(
                weight * component.analytic_mean(severity_cap_override=cap)
                for weight, component in zip(self.weights, self.components, strict=True)
                if weight > 0.0
            )
        )

    def quantile(self, p: float) -> float:
        if not np.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"quantile probability must be in [0, 1], got {p}")
        return float(self.sample_from_uniforms(np.array([p], dtype=float))[0])

    def survival_probability(self, threshold: float) -> float:
        """Return exact linear-pool P(X > threshold).

        Mix component survival probabilities; the mixture cap censors excess mass
        at the cap.
        """
        if not np.isfinite(threshold):
            raise ValueError(f"survival threshold must be finite, got {threshold}")
        probability = float(self.survival(np.array([float(threshold)]))[0])
        return float(np.clip(probability, 0.0, 1.0))

    @property
    def p5(self) -> float:
        return self.quantile(0.05)

    @property
    def p25(self) -> float:
        return self.quantile(0.25)

    @property
    def p50(self) -> float:
        return self.quantile(0.50)

    @property
    def p75(self) -> float:
        return self.quantile(0.75)

    @property
    def p95(self) -> float:
        return self.quantile(0.95)

    @property
    def gpd_shape(self) -> float | None:
        shapes = self.component_gpd_shapes()
        return max(shapes) if shapes else None

    def with_gpd_shape(self, shape: float) -> "MixtureSeverityDistribution":
        """Set the absolute shape of each active GPD component; retain other laws."""
        return MixtureSeverityDistribution(
            components=[
                component.with_gpd_shape(shape)
                if weight > 0 and component.gpd_shape is not None
                else component
                for component, weight in zip(self.components, self.weights, strict=True)
            ],
            weights=self.weights.copy(),
            severity_cap=self.severity_cap,
        )

    def with_currency_scale(
        self, factor: float, *factors: float
    ) -> "MixtureSeverityDistribution":
        return _rebuilt(
            MixtureSeverityDistribution,
            components=[
                component.with_currency_scale(factor, *factors)
                for component in self.components
            ],
            weights=self.weights.copy(),
            severity_cap=None
            if self.severity_cap is None
            else math.prod((self.severity_cap, factor, *factors)),
        )

    def with_excess_scale(self, scale: float) -> "MixtureSeverityDistribution":
        """``L + scale*(X - L)`` for every active component, around the
        mixture's shared qualifying floor. Each active component's own cap is
        one of its amounts and moves to ``L + scale*(cap - L)``; the mixture's
        outer cap stays nominal."""
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("severity excess scale must be finite and positive")
        if scale == 1.0:
            return self
        offset = severity_minimum(self) * (1.0 - scale)
        return MixtureSeverityDistribution(
            components=[
                component._with_affine_severity(offset, scale, intrinsic_cap=True)
                if weight > 0.0
                else component
                for component, weight in zip(self.components, self.weights, strict=True)
            ],
            weights=self.weights.copy(),
            severity_cap=self.severity_cap,
        )

    def component_gpd_shapes(self) -> list[float]:
        shapes: list[float] = []
        for component, weight in zip(self.components, self.weights, strict=True):
            if weight > 0.0:
                shapes.extend(component.component_gpd_shapes())
        return shapes
