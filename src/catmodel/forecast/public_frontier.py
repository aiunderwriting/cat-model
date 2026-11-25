"""Observed-anchored lab trends with a pooled rate and independent lab paces.

All labs' frontier-raising releases inform one trend rate, with lab-specific
levels integrated out; each release enters once. Paths start at each lab's
observed published frontier at as-of, so a forecast cannot fall below what is
already public. Each lab draws its own authored pace multiplier, a median-one
log-logistic law truncated at the forecast's pace cap, so labs progress at
different rates and the frontier leader can change; future dependence between
labs comes from the shared trend-rate draw alone, not an estimated cross-lab
correlation matrix.
"""

from datetime import datetime
from typing import Any, Literal

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import truncnorm

from catmodel.contracts.public_frontier import (
    VARIANTS,
    YEAR_SECONDS,
    PublicFrontierEnsemble,
    PublicFrontierForecast,
    PublicFrontierRelease,
    PublicFrontierVariant,
    PublicLabRecords,
    PublicPathDraw,
    PublicTrendPosterior,
    forecast_times,
)

# The two judgmental pace-shape alternatives on either side of the default 5
# (docs/capability-frontiers.md): 3 is wider and 8, the backtest's lowest-loss
# shape, is narrower. They do not move with an authored shape.
PACE_SHAPE_ALTERNATIVES = {"lower_pace_shape": 3.0, "higher_pace_shape": 8.0}
# The judgmental alternative to the default pace cap of 4: a cap of 2, about
# the fastest twelve months of any roster lab in the captured record. It does
# not move with an authored cap; a forecast whose cap is not above it has no
# such alternative. A higher cap has paths beyond every node of a study planned
# under the forecast's cap, so it is never one.
PACE_CAP_ALTERNATIVES = {"lower_pace_cap": 2.0}


def lab_frontier_records(
    forecast: PublicFrontierForecast,
) -> dict[str, tuple[PublicFrontierRelease, ...]]:
    """Each lab's frontier-raising releases in availability order.

    A release is a record only when it strictly exceeds every earlier release
    of the same lab. An equal or lower score, including a re-listed score, is
    not a new frontier observation. At one instant only the highest score can
    be a record.
    """
    records: dict[str, list[PublicFrontierRelease]] = {
        row.lab_id: [] for row in forecast.coordinates
    }
    for release in sorted(
        forecast.releases,
        key=lambda row: (row.available_at, -row.value, row.release_id),
    ):
        current = records[release.lab_id]
        if not current or release.value > current[-1].value:
            current.append(release)
    return {lab: tuple(rows) for lab, rows in records.items()}


def observed_frontier(forecast: PublicFrontierForecast, at: datetime) -> float | None:
    """The highest score among the roster labs' releases available at ``at``,
    on the forecast's frozen scale, or None when none was available yet."""
    values = [row.value for row in forecast.releases if row.available_at <= at]
    return max(values, default=None)


def one_year_before(at: datetime) -> datetime:
    """The same calendar instant one year earlier; 28 February for a leap day."""
    try:
        return at.replace(year=at.year - 1)
    except ValueError:
        return at.replace(year=at.year - 1, day=28)


def window_records(
    forecast: PublicFrontierForecast, discrepancy_sd: float
) -> dict[str, list[tuple[float, float, float, str]]]:
    """Each lab's in-window records as (years from as-of, score, weight, ID)."""
    variance = discrepancy_sd**2
    return {
        lab: [
            (
                (row.available_at - forecast.as_of).total_seconds() / YEAR_SECONDS,
                row.value,
                1 / (variance + row.benchmark_error_sd**2),
                row.release_id,
            )
            for row in rows
            if row.available_at >= forecast.trend_window_start
        ]
        for lab, rows in lab_frontier_records(forecast).items()
    }


# Search range for the estimated discrepancy, in ECI points. The lower end
# stands in for zero when benchmark error alone explains the records.
DISCREPANCY_SEARCH_RANGE = (0.01, 50.0)
# Half the 90% chi-square(1) quantile: the profile restricted log-likelihood
# drop that bounds a 90% interval.
_PROFILE_DROP_90 = 1.3527717679


def _restricted_negative_log_likelihood(
    log_sd: float, design: np.ndarray, values: np.ndarray, benchmark: np.ndarray
) -> float:
    variance = np.exp(2 * log_sd) + benchmark
    inverse = 1 / variance
    information = design.T @ (inverse[:, None] * design)
    weighted = inverse * values
    fitted = design @ np.linalg.solve(information, design.T @ weighted)
    residual = values - fitted
    return 0.5 * float(
        np.sum(np.log(variance))
        + np.linalg.slogdet(information)[1]
        + residual @ (inverse * residual)
    )


def _restricted_score(
    log_sd: float, design: np.ndarray, values: np.ndarray, benchmark: np.ndarray
) -> float:
    """Derivative of ``_restricted_negative_log_likelihood`` in ``log_sd``.

    With ``V`` the diagonal of record variances and
    ``P = V⁻¹ - V⁻¹X(XᵀV⁻¹X)⁻¹XᵀV⁻¹``, it is ``τ²(tr P - yᵀP²y)``, because
    ``dV/d log τ = 2τ²I``.
    """
    discrepancy_variance = np.exp(2 * log_sd)
    inverse = 1 / (discrepancy_variance + benchmark)
    information = design.T @ (inverse[:, None] * design)
    fitted = design @ np.linalg.solve(information, design.T @ (inverse * values))
    projected = inverse * (values - fitted)
    trace = np.sum(inverse) - np.trace(
        np.linalg.solve(information, design.T @ ((inverse * inverse)[:, None] * design))
    )
    return float(discrepancy_variance * (trace - projected @ projected))


def _score_root(score, start: float, low: float, high: float) -> float:
    """The root of the restricted score next to a bounded-Brent minimum.

    Brent's method stops within about the square root of machine epsilon of a
    minimum, where objective values no longer separate points, so where it
    stops depends on its search path, such as the bounds or the SciPy version.
    The score crosses zero there with a nonzero slope, so a bracketing root
    finder fixes the estimate to about 1e-14 in ``log τ``, and import's
    recomputation of a frozen forecast reproduces it far inside its roundoff
    tolerance. A minimum at a search bound stays at the bound.
    """
    step = 1e-6
    while True:
        left, right = max(low, start - step), min(high, start + step)
        below, above = score(left), score(right)
        if left == low and below >= 0:
            return low
        if right == high and above <= 0:
            return high
        if below < 0 < above:
            return float(brentq(score, left, right, xtol=1e-14))
        step *= 8


def estimate_trend_discrepancy(
    forecast: PublicFrontierForecast,
) -> tuple[float, tuple[float, float]]:
    """Pooled restricted-maximum-likelihood discrepancy and its 90% interval.

    The model is the pooled trend: lab levels and the shared rate are fixed
    effects, each window record has variance discrepancy² + benchmark². The
    rate prior is not used. The interval is the profile-likelihood interval; a
    lower end at the search floor means benchmark error alone fits the records.
    """
    forecast = PublicFrontierForecast.model_validate(forecast.model_dump(mode="json"))
    return _reml_discrepancy(
        [
            (
                lab,
                (row.available_at - forecast.as_of).total_seconds() / YEAR_SECONDS,
                row.value,
                row.benchmark_error_sd**2,
            )
            for lab, records in lab_frontier_records(forecast).items()
            for row in records
            if row.available_at >= forecast.trend_window_start
        ]
    )


def _reml_discrepancy(
    rows: list[tuple[str, float, float, float]],
) -> tuple[float, tuple[float, float]]:
    """REML discrepancy from (lab, years from as-of, score, benchmark variance)."""
    labs = sorted({row[0] for row in rows})
    if len(rows) - len(labs) - 1 < 2:
        raise ValueError(
            "estimating the trend discrepancy needs at least two window records beyond the lab levels and rate; author trend_discrepancy_sd"
        )
    design = np.array(
        [[*(float(lab == other) for other in labs), time] for lab, time, _, _ in rows]
    )
    values = np.array([row[2] for row in rows])
    benchmark = np.array([row[3] for row in rows])
    low, high = np.log(DISCREPANCY_SEARCH_RANGE[0]), np.log(DISCREPANCY_SEARCH_RANGE[1])

    def objective(log_sd: float) -> float:
        return _restricted_negative_log_likelihood(log_sd, design, values, benchmark)

    def score(log_sd: float) -> float:
        return _restricted_score(log_sd, design, values, benchmark)

    best = minimize_scalar(
        objective, bounds=(low, high), method="bounded", options={"xatol": 1e-8}
    )
    location = _score_root(score, float(best.x), low, high)
    minimum = objective(location)

    def crossing(bound: float) -> float:
        if objective(bound) - minimum < _PROFILE_DROP_90:
            return bound
        return brentq(
            lambda value: objective(value) - minimum - _PROFILE_DROP_90,
            *sorted((location, bound)),
            xtol=1e-14,
        )

    estimate = float(np.exp(location))
    lower = crossing(low)
    upper = crossing(high)
    return estimate, (
        0.0 if lower == low else float(np.exp(lower)),
        float(np.exp(upper)),
    )


def resolve_trend_discrepancy(
    forecast: PublicFrontierForecast,
) -> tuple[
    float, Literal["pooled_reml_estimate", "authored"], tuple[float, float] | None
]:
    """The authored discrepancy, or the pooled estimate with its interval."""
    if forecast.trend_discrepancy_sd is not None:
        return forecast.trend_discrepancy_sd, "authored", None
    estimate, interval = estimate_trend_discrepancy(forecast)
    return estimate, "pooled_reml_estimate", interval


def _pooled_rate(
    prior_mean: float,
    prior_sd: float,
    records: dict[str, list[tuple[float, float, float, str]]],
) -> tuple[float, float]:
    """Gaussian rate posterior with flat lab levels integrated out.

    Each lab contributes its weighted within-lab deviations from its own mean
    time and score, so a lab with one record carries no rate information.
    """
    if prior_sd == 0:
        return prior_mean, 0.0
    precision = 1 / prior_sd**2
    weighted = prior_mean * precision
    for rows in records.values():
        if not rows:
            continue
        times = np.asarray([row[0] for row in rows])
        values = np.asarray([row[1] for row in rows])
        weights = np.asarray([row[2] for row in rows])
        total = weights.sum()
        time_mean = weights @ times / total
        value_mean = weights @ values / total
        precision += weights @ (times - time_mean) ** 2
        weighted += weights @ ((times - time_mean) * (values - value_mean))
    return float(weighted / precision), float(1 / precision)


def fit_public_trend(forecast: PublicFrontierForecast) -> PublicTrendPosterior:
    """Fit one positive trend rate to all labs' in-window records.

    Each record enters once with independent variance discrepancy² +
    benchmark²; carrying a frontier through a plateau adds no observation. Lab
    levels have flat priors. The saved location and variance describe the
    Gaussian kernel before positive-rate conditioning. The discrepancy is the
    authored value or the pooled REML estimate; benchmark errors are supplied
    with the releases.
    """
    forecast = PublicFrontierForecast.model_validate(forecast.model_dump(mode="json"))
    records = lab_frontier_records(forecast)
    discrepancy, basis, interval = resolve_trend_discrepancy(forecast)
    window = window_records(forecast, discrepancy)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            location, variance = _pooled_rate(
                forecast.rate_prior.mean, forecast.rate_prior.sd, window
            )
        except FloatingPointError as exc:
            raise ValueError(
                "Gaussian trend update exceeds finite numerical support"
            ) from exc
    return PublicTrendPosterior(
        rate_location=float(location),
        rate_variance=float(variance),
        trend_discrepancy_sd=float(discrepancy),
        trend_discrepancy_basis=basis,
        trend_discrepancy_interval=interval,
        labs={
            lab: PublicLabRecords(
                likelihood_release_ids=tuple(row[3] for row in window[lab]),
                as_of_release_id=rows[-1].release_id,
                as_of_value=rows[-1].value,
            )
            for lab, rows in records.items()
        },
    )


def sample_positive_rate(
    posterior: PublicTrendPosterior, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample the pooled rate's Gaussian kernel conditioned to be positive."""
    rate_mean, rate_variance = posterior.rate_location, posterior.rate_variance
    if rate_variance == 0:
        if rate_mean <= 0:
            raise ValueError("fixed baseline rate has no positive support")
        return np.full(n_draws, rate_mean)
    rate_sd = np.sqrt(rate_variance)
    # SciPy may underflow an irrelevant tail mass for a tightly positive rate.
    # Check the returned draws below; retain strict arithmetic elsewhere.
    with np.errstate(under="ignore"):
        rates = truncnorm.rvs(
            -rate_mean / rate_sd,
            np.inf,
            loc=rate_mean,
            scale=rate_sd,
            size=n_draws,
            random_state=rng,
        )
    if not np.isfinite(rates).all() or np.any(rates <= 0):
        raise ValueError(
            "positive baseline draw exceeds finite numerical support; no draw was replaced"
        )
    return rates


def pace_quantile(shape: float, probability: np.ndarray, cap: float) -> np.ndarray:
    """Pace multipliers at the given probabilities.

    The pace law is the median-one log-logistic law of the given shape,
    truncated above at ``cap`` and renormalized: its CDF is ``F(m) / F(cap)``
    for ``m <= cap``, with ``F(m) = m^shape / (1 + m^shape)``, and it has no
    atom at the cap.
    """
    if np.any(probability <= 0) or np.any(probability >= 1):
        raise ValueError(
            "pace quantile is outside finite numerical support; no draw was replaced"
        )
    if not cap > 0:
        raise ValueError("the pace cap must be positive")
    # log of probability * F(cap) and of its complement, without forming
    # F(cap); a cap far beyond the law's body leaves log F(cap) at zero.
    with np.errstate(under="ignore"):
        log_below = np.log(probability) - np.logaddexp(0.0, -shape * np.log(cap))
    return np.exp((log_below - np.log1p(-np.exp(log_below))) / shape)


def sample_public_frontier(
    forecast: PublicFrontierForecast,
    *,
    period_start: datetime,
    period_end: datetime,
    n_draws: int,
    seed: int,
    variant: PublicFrontierVariant = "baseline",
) -> PublicFrontierEnsemble:
    """Draw the pooled positive trend rate and each lab's persistent median-one pace.

    Every lab path starts at the lab's observed as-of frontier and is evaluated
    at the period bounds and each month start between them. The no-progress
    variant holds every lab at its as-of frontier.
    """
    forecast = PublicFrontierForecast.model_validate(forecast.model_dump(mode="json"))
    if (
        isinstance(n_draws, bool)
        or not isinstance(n_draws, int)
        or not 1 <= n_draws <= 100000
    ):
        raise ValueError("public frontier requires 1 to 100000 draws")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("public frontier seed must be a nonnegative integer")
    if variant not in VARIANTS:
        raise ValueError("unknown public frontier sensitivity variant")
    times = forecast_times(period_start, period_end)
    if forecast.as_of > times[0]:
        raise ValueError("public forecast as-of cannot follow the period start")
    posterior = fit_public_trend(forecast)
    rng = np.random.default_rng(seed)
    lab_count = len(forecast.coordinates)
    anchors = np.asarray(
        [posterior.labs[row.lab_id].as_of_value for row in forecast.coordinates]
    )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            baseline = sample_positive_rate(posterior, n_draws, rng)
            if variant == "no_progress":
                pace = np.zeros((n_draws, lab_count))
            else:
                pace = pace_quantile(
                    forecast.pace_shape,
                    rng.random((n_draws, lab_count)),
                    forecast.pace_cap,
                )
            rates = baseline[:, None] * pace
            elapsed = np.asarray(
                [(at - forecast.as_of).total_seconds() / YEAR_SECONDS for at in times]
            )
            values = anchors[None, None, :] + rates[:, None, :] * elapsed[None, :, None]
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(
            "public trend or pace draw exceeds finite numerical support; no draw was replaced"
        ) from exc
    if not all(np.isfinite(array).all() for array in (baseline, pace, rates, values)):
        raise ValueError(
            "public trend or pace draw exceeds finite numerical support; no draw was replaced"
        )
    return PublicFrontierEnsemble(
        as_of=forecast.as_of,
        forecast_sha256=forecast.digest(),
        coordinates=forecast.coordinates,
        times=times,
        draws=tuple(
            PublicPathDraw(
                draw_id=f"public-{index:06d}",
                weight=1 / n_draws,
                values=tuple(tuple(map(float, row)) for row in path),
                baseline_rate=float(baseline[index]),
                pace_multipliers=tuple(map(float, pace[index])),
                rates=tuple(map(float, rates[index])),
            )
            for index, path in enumerate(values)
        ),
        posterior=posterior,
        seed=seed,
        variant=variant,
        interpolation="linear",
        source_references=tuple(
            dict.fromkeys(
                [
                    ref
                    for coordinate in forecast.coordinates
                    for ref in coordinate.evidence_refs
                ]
                + [
                    ref
                    for release in forecast.releases
                    for ref in release.evidence_refs
                ]
            )
        ),
        assumptions=(
            "Each lab path starts at the lab's highest published score available at as-of, so no path or frontier draw starts below what was already public. Paths are linear and nondecreasing.",
            "One positive trend rate, fitted to all labs' frontier-raising releases with lab-specific levels, is drawn once per path and shared by every lab; each lab's own pace multiplier sets its relative progress, so the frontier leader can change.",
            "Only frontier-raising releases in the trend window enter the likelihood, each once; carrying a frontier through a plateau adds no observation. Waiting time since a lab's last record is not used, and the anchored path assumes no catch-up for a lab whose last record is old.",
            "Record scores have independent Gaussian error equal to the shared trend discrepancy plus the release's benchmark error. The discrepancy is the pooled restricted-maximum-likelihood estimate from the window records unless authored; its estimation uncertainty is reported as a sensitivity, not propagated. Records are upward-selected releases; serial dependence of record jumps is not modeled.",
            "The frozen, retrospectively refitted ECI scale does not reconstruct historical score vintages. Joint calibration error across different releases or labs is not represented.",
            "No-progress paths fix every lab's pace multiplier to zero and stay at the observed as-of frontiers, so their frontier leader cannot change."
            if variant == "no_progress"
            else f"Each lab draws an independent pace multiplier that persists from as-of through the complete forecast horizon, so labs progress at different rates and the frontier leader can change. The pace law is a median-one log-logistic law truncated at the pace cap of {forecast.pace_cap:g}, so no lab progresses faster than that multiple of the drawn trend rate. Its shape is a structural dispersion judgment unless a rolling-origin backtest on authored releases supports another value, and its cap a judgment about the fastest sustained progress; lab-specific structural shocks are independent, not estimated.",
            "The public frontier is the maximum of the named lab frontiers within each joint draw. Its coverage depends on the explicitly declared lab roster.",
            f"Elapsed time is measured in 365.25-day years from as-of. Paths are evaluated from {times[0].isoformat()} to {times[-1].isoformat()} at each month start between; each lab path is linear and the frontier is their pointwise maximum.",
            "Domain benchmark results, such as cyber or biology evaluations, remain contextual evidence; none is forecast, converted to ECI, or used to relabel an existing domain response curve.",
            "Shared public trajectories preserve this source of dependence only; residual risk dependence is separate.",
            "The pace is bounded by its cap, so every pace moment is finite, and frontier progress is bounded by the cap times the drawn trend rate. Finite progress moments do not establish finite loss moments; nonlinear responses and severity tails require separate assessment.",
            forecast.frontier_definition,
            forecast.pooling_rationale,
            forecast.likelihood_rationale,
            forecast.pace_rationale,
            forecast.rate_prior.rationale,
            *forecast.limitations,
        ),
    )


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return dict(
        zip(
            ("p05", "p50", "p95"),
            map(float, np.quantile(values, (0.05, 0.5, 0.95))),
            strict=True,
        )
    )


def _summary(ensemble: PublicFrontierEnsemble) -> dict[str, Any]:
    values = np.asarray([draw.values for draw in ensemble.draws])
    rates = np.asarray([draw.rates for draw in ensemble.draws])
    last = len(ensemble.times) - 1
    return {
        "checkpoints": {
            ensemble.times[index].isoformat(): {
                "labs": {
                    coordinate.lab_id: _quantiles(values[:, index, j])
                    for j, coordinate in enumerate(ensemble.coordinates)
                },
                "public_frontier": _quantiles(values[:, index, :].max(axis=1)),
            }
            for index in (0, last // 2, last)
        },
        "annual_rates": {
            coordinate.lab_id: _quantiles(rates[:, j])
            for j, coordinate in enumerate(ensemble.coordinates)
        },
        "zero_progress_probability": {
            coordinate.lab_id: float(np.mean(rates[:, j] == 0))
            for j, coordinate in enumerate(ensemble.coordinates)
        },
        "rates_all_above_own_p90_probability": float(
            np.mean(np.all(rates > np.quantile(rates, 0.9, axis=0), axis=1))
        ),
    }


def record_residuals(forecast: PublicFrontierForecast) -> dict[str, dict[str, Any]]:
    """Leave-one-out standardized residuals of each lab's in-window records.

    Each record is predicted from all other records: an estimated discrepancy
    is re-estimated without it, the pooled rate refitted without it, and the
    lab's level taken from its other records, with predictive variance
    ``1/W + v·(t - t̄)² + τ² + σ²``. A record whose lab has no other window
    record has no level and is not scored, nor is one whose removal leaves too
    few records to estimate the discrepancy. With the discrepancy right, the
    root mean square is near one. It is a diagnostic, not an estimate.
    """
    forecast = PublicFrontierForecast.model_validate(forecast.model_dump(mode="json"))
    estimated = forecast.trend_discrepancy_sd is None
    records = {
        lab: [
            (
                (row.available_at - forecast.as_of).total_seconds() / YEAR_SECONDS,
                row.value,
                row.benchmark_error_sd**2,
                row.release_id,
            )
            for row in rows
            if row.available_at >= forecast.trend_window_start
        ]
        for lab, rows in lab_frontier_records(forecast).items()
    }
    result = {}
    for lab, rows in records.items():
        residuals = []
        for index, (time, value, benchmark, _) in enumerate(rows):
            remaining = {**records, lab: rows[:index] + rows[index + 1 :]}
            if not remaining[lab]:
                continue
            if estimated:
                try:
                    discrepancy = _reml_discrepancy(
                        [
                            (other, *row[:3])
                            for other, values in remaining.items()
                            for row in values
                        ]
                    )[0]
                except ValueError:
                    continue
            else:
                assert forecast.trend_discrepancy_sd is not None
                discrepancy = forecast.trend_discrepancy_sd
            variance_floor = discrepancy**2
            weighted = {
                other: [
                    (row[0], row[1], 1 / (variance_floor + row[2]), row[3])
                    for row in values
                ]
                for other, values in remaining.items()
            }
            location, variance = _pooled_rate(
                forecast.rate_prior.mean, forecast.rate_prior.sd, weighted
            )
            others = weighted[lab]
            weights = np.asarray([row[2] for row in others])
            total = weights.sum()
            time_mean = weights @ np.asarray([row[0] for row in others]) / total
            value_mean = weights @ np.asarray([row[1] for row in others]) / total
            prediction = value_mean + location * (time - time_mean)
            spread = (
                1 / total
                + variance * (time - time_mean) ** 2
                + variance_floor
                + benchmark
            )
            residuals.append((value - prediction) / np.sqrt(spread))
        result[lab] = {
            "count": len(residuals),
            "leave_one_out_rms": float(np.sqrt(np.mean(np.square(residuals))))
            if residuals
            else None,
        }
    return result


def staleness(forecast: PublicFrontierForecast) -> dict[str, dict[str, Any]]:
    """Age of each lab's as-of record and its fitted trend level's lead over it.

    The fitted level uses the posterior rate kernel location and the lab's
    window records; a lab without window records has none. A large positive
    lead marks a lab whose anchored path assumes no catch-up to its trend.
    """
    posterior = fit_public_trend(forecast)
    records = lab_frontier_records(forecast)
    window = window_records(forecast, posterior.trend_discrepancy_sd)
    result = {}
    for lab, rows in records.items():
        level = None
        if window[lab]:
            weights = np.asarray([row[2] for row in window[lab]])
            total = weights.sum()
            time_mean = weights @ np.asarray([row[0] for row in window[lab]]) / total
            value_mean = weights @ np.asarray([row[1] for row in window[lab]]) / total
            level = float(value_mean - posterior.rate_location * time_mean)
        result[lab] = {
            "years_since_as_of_record": (
                forecast.as_of - rows[-1].available_at
            ).total_seconds()
            / YEAR_SECONDS,
            "trend_level_minus_as_of_value": None
            if level is None
            else level - posterior.labs[lab].as_of_value,
        }
    return result


def trend_discrepancy_alternatives(
    forecast: PublicFrontierForecast, posterior: PublicTrendPosterior
) -> dict[str, float]:
    """An estimated discrepancy's 90% interval ends, or an override's estimate.

    An authored override is compared with the pooled estimate when one is
    identified; otherwise there is no alternative.
    """
    if posterior.trend_discrepancy_interval is not None:
        lower, upper = posterior.trend_discrepancy_interval
        return {
            "lower_trend_discrepancy": max(lower, DISCREPANCY_SEARCH_RANGE[0]),
            "upper_trend_discrepancy": upper,
        }
    try:
        return {"estimated_trend_discrepancy": estimate_trend_discrepancy(forecast)[0]}
    except ValueError:
        return {}


def compare_public_frontier(
    forecast: PublicFrontierForecast,
    *,
    period_start: datetime,
    period_end: datetime,
    n_draws: int,
    seed: int,
) -> dict[str, Any]:
    """Compare whole lab/frontier distributions without fitting correlations.

    Besides the baseline and the no-progress counterfactual, the baseline is
    repeated at the two judgmental pace-shape alternatives, at the lower pace
    cap when the forecast's own cap is above it, and at the
    ``trend_discrepancy_alternatives``.
    """
    posterior = fit_public_trend(forecast)
    discrepancy_alternatives = trend_discrepancy_alternatives(forecast, posterior)

    def summary(source, variant):
        return _summary(
            sample_public_frontier(
                source,
                period_start=period_start,
                period_end=period_end,
                n_draws=n_draws,
                seed=seed,
                variant=variant,
            )
        )

    return {
        "forecast_sha256": forecast.digest(),
        "n_draws": n_draws,
        "seed": seed,
        "pace_shape": forecast.pace_shape,
        "pace_cap": forecast.pace_cap,
        "as_of_frontier": {
            "labs": {lab: row.as_of_value for lab, row in posterior.labs.items()},
            "public_frontier": max(row.as_of_value for row in posterior.labs.values()),
        },
        "posterior_observation_counts": {
            lab: row.observation_count for lab, row in posterior.labs.items()
        },
        "pooled_rate_kernel": {
            "location": posterior.rate_location,
            "sd": float(np.sqrt(posterior.rate_variance)),
        },
        "trend_discrepancy": {
            "sd": posterior.trend_discrepancy_sd,
            "basis": posterior.trend_discrepancy_basis,
            "interval_90": posterior.trend_discrepancy_interval,
        },
        "record_residuals": record_residuals(forecast),
        "staleness": staleness(forecast),
        "variants": {variant: summary(forecast, variant) for variant in VARIANTS},
        "trend_discrepancy_sensitivity": {
            name: {
                "trend_discrepancy_sd": value,
                **summary(
                    forecast.model_copy(update={"trend_discrepancy_sd": value}),
                    "baseline",
                ),
            }
            for name, value in discrepancy_alternatives.items()
        },
        "pace_shape_sensitivity": {
            name: {
                "pace_shape": shape,
                **summary(
                    forecast.model_copy(update={"pace_shape": shape}), "baseline"
                ),
            }
            for name, shape in PACE_SHAPE_ALTERNATIVES.items()
        },
        "pace_cap_sensitivity": {
            name: {
                "pace_cap": cap,
                **summary(forecast.model_copy(update={"pace_cap": cap}), "baseline"),
            }
            for name, cap in PACE_CAP_ALTERNATIVES.items()
            if cap < forecast.pace_cap
        },
        "moment_conditions": {
            "pace_moments": "finite_at_every_order_under_the_pace_cap",
            "continuous_forecast_loss_moments": "not_established",
            "reported_quantiles": "retained_finite_ensemble",
        },
        "limitations": [
            "Pace shape, the pace cap and the independence of lab paces are authored assumptions, not empirical validation or an estimated correlation matrix.",
            "A baseline rate is the conditional median rate given the pooled rate draw, not necessarily the marginal median after mixing posterior uncertainty.",
            "Positive-rate conditioning does not create a zero atom. The no-progress comparison is a separate diagnostic.",
            "Finite rate moments do not establish finite loss moments. In particular, bounded conditional loss means do not establish finite loss variance.",
        ],
    }
