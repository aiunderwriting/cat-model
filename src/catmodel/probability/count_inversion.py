"""Counts drawn by inversion: the least count whose CDF reaches a uniform.

The count ``min{k >= 0 : F(k) >= u}`` is nondecreasing in ``u`` for every law,
and for a fixed ``u`` it does not decrease when a Poisson rate or a
negative-binomial mean rises, or when a binomial's trials or probability rise.
Laws read at the same uniform therefore draw comonotone counts, which keeps a
year paired across arms that change its count law (``NUM-STREAMS``).

Each count is exact for its law's CDF as SciPy evaluates it,
``F(k - 1) < u <= F(k)``. A uniform up to one half is compared with the lower
tail and a larger one with the upper tail, where the complement ``1 - u`` of a
53-bit uniform is exact, so both tails keep their relative accuracy at any
rate. A second-order Cornish-Fisher guess starts the search, which doubles its
step until it brackets the count and then bisects; the guess is almost always
the count, which then costs two CDF evaluations. A uniform of zero gives the
least count, zero. Each count's probability is resolved on the uniforms' grid
of 2**-53, so a count less likely than that may never be drawn.
"""

from collections.abc import Callable

import numpy as np
from scipy.special import bdtr, bdtrc, betainc, betaincc, ndtri, pdtr, pdtrc

_LARGEST_COUNT = 2**53
"""Counts from here on are not all exactly representable as floats."""

_Tail = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _uniforms(values) -> np.ndarray:
    uniforms = np.asarray(values, dtype=float)
    if uniforms.ndim != 1 or not np.all((uniforms >= 0.0) & (uniforms < 1.0)):
        raise ValueError("count uniforms must be a one-dimensional array in [0, 1)")
    return uniforms


def _parameter(values, uniforms: np.ndarray, name: str) -> np.ndarray:
    parameter = np.broadcast_to(np.asarray(values, dtype=float), uniforms.shape)
    if not np.all(np.isfinite(parameter) & (parameter >= 0.0)):
        raise ValueError(f"count {name} must be finite and nonnegative")
    return parameter


def _guess(uniforms, mean, variance, skewness, excess_kurtosis) -> np.ndarray:
    """Second-order Cornish-Fisher quantile, continuity-corrected and rounded up.

    A zero uniform reads a score of -9, below every positive 53-bit uniform's.
    """
    z = np.clip(ndtri(uniforms), -9.0, 9.0)
    w = (
        z
        + skewness * (z * z - 1.0) / 6.0
        + excess_kurtosis * (z**3 - 3.0 * z) / 24.0
        - skewness**2 * (2.0 * z**3 - 5.0 * z) / 36.0
    )
    return np.ceil(mean + np.sqrt(variance) * w - 0.5)


def _least_count(
    uniforms: np.ndarray,
    guess: np.ndarray,
    cdf: _Tail,
    survival: _Tail,
    upper: np.ndarray | None = None,
) -> np.ndarray:
    """The least count whose CDF reaches each uniform, searched from ``guess``.

    ``cdf(index, counts)`` and ``survival(index, counts)`` evaluate the law of
    the elements ``index`` at float ``counts``. ``upper``, where given, is a
    count every uniform's quantile is known not to exceed (a binomial's
    trials). From the guess the step doubles outward until the count is
    bracketed, then the bracket is bisected.
    """
    lower_tail = uniforms <= 0.5
    complement = 1.0 - uniforms

    def reaches(index: np.ndarray, counts: np.ndarray) -> np.ndarray:
        if np.any(counts >= _LARGEST_COUNT):
            raise ValueError("a count exceeds the exactly representable range")
        values = counts.astype(float)
        result = np.empty(index.size, dtype=bool)
        lower = lower_tail[index]
        low = index[lower]
        result[lower] = cdf(low, values[lower]) >= uniforms[low]
        high = index[~lower]
        result[~lower] = survival(high, values[~lower]) <= complement[high]
        return result

    count = np.maximum(np.nan_to_num(guess, nan=0.0, posinf=0.0), 0.0)
    if upper is not None:
        count = np.minimum(count, upper)
    count = count.astype(np.int64)
    every = np.arange(count.size)
    above = reaches(every, count)
    # The guess is almost always the count or one short of it, which the
    # neighbor's verdict settles: -1 never reaches a quantile, ``upper`` does.
    neighbor = np.where(above, count - 1, count + 1)
    neighbor_reaches = ~above
    if upper is None:
        unknown_neighbor = np.where(above, neighbor >= 0, True)
    else:
        unknown_neighbor = np.where(above, neighbor >= 0, neighbor < upper)
    index = np.flatnonzero(unknown_neighbor)
    neighbor_reaches[index] = reaches(index, neighbor[index])
    least = np.where(above, count, neighbor)
    unsettled = np.flatnonzero(above == neighbor_reaches)
    if not unsettled.size:
        return least
    # ``below`` is the largest count known to fall short of the quantile and
    # ``reached`` the least known to reach it; -1 always falls short, and
    # ``unknown`` marks a side the search has not bracketed yet. The step
    # doubles while a side is open.
    unknown = np.int64(-2)
    settled_above = above[unsettled]
    below = np.full(count.size, unknown)
    reached = np.full(count.size, unknown)
    below[unsettled[~settled_above]] = neighbor[unsettled[~settled_above]]
    reached[unsettled[settled_above]] = neighbor[unsettled[settled_above]]
    step = np.full(count.size, 2, dtype=np.int64)
    active = unsettled
    while active.size:
        low, high = below[active], reached[active]
        candidate = np.where(
            low == unknown,
            high - step[active],
            np.where(high == unknown, low + step[active], (low + high) // 2),
        )
        evaluate = np.ones(active.size, dtype=bool)
        under = (low == unknown) & (candidate < 0)
        below[active[under]] = -1
        evaluate &= ~under
        if upper is not None:
            over = (high == unknown) & (candidate >= upper[active])
            reached[active[over]] = upper[active[over]]
            evaluate &= ~over
        index = active[evaluate]
        if index.size:
            result = reaches(index, candidate[evaluate])
            galloping = (below[index] == unknown) | (reached[index] == unknown)
            reached[index[result]] = candidate[evaluate][result]
            below[index[~result]] = candidate[evaluate][~result]
            step[index[galloping]] *= 2
        open_ = (below[active] == unknown) | (reached[active] == unknown)
        active = active[open_ | (reached[active] - below[active] > 1)]
    least[unsettled] = reached[unsettled]
    return least


def poisson_counts(uniforms, rates) -> np.ndarray:
    """Poisson counts at ``rates``, inverted from ``uniforms``."""
    uniforms = _uniforms(uniforms)
    rates = _parameter(rates, uniforms, "rates")
    counts = np.zeros(uniforms.shape, dtype=np.int64)
    positive = np.flatnonzero(rates > 0.0)
    if not positive.size:
        return counts
    u, lam = uniforms[positive], rates[positive]
    # Below a mean of one the expansion misleads, and counts are small.
    guess = np.zeros(lam.shape)
    wide = np.flatnonzero(lam >= 1.0)
    guess[wide] = _guess(
        u[wide], lam[wide], lam[wide], lam[wide] ** -0.5, 1.0 / lam[wide]
    )
    counts[positive] = _least_count(
        u,
        guess,
        lambda index, k: pdtr(k, lam[index]),
        lambda index, k: pdtrc(k, lam[index]),
    )
    return counts


def negative_binomial_counts(uniforms, means, variance_to_mean) -> np.ndarray:
    """Negative-binomial counts of ``means`` and a variance-to-mean ratio above
    one, inverted from ``uniforms``: failures before ``mean / (v - 1)``
    successes of probability ``1 / v``."""
    uniforms = _uniforms(uniforms)
    means = _parameter(means, uniforms, "means")
    ratio = _parameter(variance_to_mean, uniforms, "variance-to-mean ratios")
    if np.any(ratio <= 1.0):
        raise ValueError("a negative-binomial variance-to-mean ratio must exceed one")
    counts = np.zeros(uniforms.shape, dtype=np.int64)
    positive = np.flatnonzero(means > 0.0)
    if not positive.size:
        return counts
    u, mean, v = uniforms[positive], means[positive], ratio[positive]
    successes, probability = mean / (v - 1.0), 1.0 / v
    guess = np.zeros(mean.shape)
    wide = np.flatnonzero(mean >= 1.0)
    m, r = mean[wide], v[wide]
    guess[wide] = _guess(
        u[wide],
        m,
        m * r,
        (2.0 * r - 1.0) / np.sqrt(m * r),
        6.0 * (r - 1.0) / m + 1.0 / (m * r),
    )
    counts[positive] = _least_count(
        u,
        guess,
        lambda index, k: betainc(successes[index], k + 1.0, probability[index]),
        lambda index, k: betaincc(successes[index], k + 1.0, probability[index]),
    )
    return counts


def binomial_counts(uniforms, trials, probabilities) -> np.ndarray:
    """Binomial counts of ``trials`` with success ``probabilities``, inverted
    from ``uniforms``."""
    uniforms = _uniforms(uniforms)
    trials = np.broadcast_to(np.asarray(trials), uniforms.shape)
    if trials.dtype.kind not in "iu" or np.any(trials < 0):
        raise ValueError("binomial trials must be nonnegative integers")
    probabilities = _parameter(probabilities, uniforms, "probabilities")
    if np.any(probabilities > 1.0):
        raise ValueError("binomial probabilities must be at most one")
    counts = np.zeros(uniforms.shape, dtype=np.int64)
    positive = np.flatnonzero((trials > 0) & (probabilities > 0.0))
    if not positive.size:
        return counts
    u = uniforms[positive]
    n = trials[positive].astype(np.int64)
    p = probabilities[positive]
    mean = n * p
    variance = mean * (1.0 - p)
    # A nearly degenerate count sits at its mean.
    guess = np.ceil(mean - 0.5)
    wide = np.flatnonzero(variance >= 1.0)
    q, var = p[wide], variance[wide]
    guess[wide] = _guess(
        u[wide],
        mean[wide],
        var,
        (1.0 - 2.0 * q) / np.sqrt(var),
        (1.0 - 6.0 * q * (1.0 - q)) / var,
    )
    counts[positive] = _least_count(
        u,
        guess,
        lambda index, k: bdtr(k, n[index], p[index]),
        lambda index, k: bdtrc(k, n[index], p[index]),
        upper=n,
    )
    return counts
