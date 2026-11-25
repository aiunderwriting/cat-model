"""Rolling-origin pinball backtest of the anchored public-frontier forecast.

At each past month-start origin the forecast is refitted from the releases
available then, with the same trend-window length, and its public-frontier
quantiles at fixed horizons are scored against the frontier the authored
releases later reached. Each candidate gives every lab its own pace, as the
forecast does, truncated at the forecast's pace cap. Candidates differ only in
pace dispersion; they share the refitted rate draws and each lab's pace
probabilities, so score differences are not Monte Carlo noise between
candidates.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

import numpy as np

from catmodel.contracts.public_frontier import (
    YEAR_SECONDS,
    PublicBacktestCandidate,
    PublicFrontierBacktest,
    PublicFrontierForecast,
)
from catmodel.forecast.public_frontier import (
    fit_public_trend,
    pace_quantile,
    sample_positive_rate,
)

# Candidate grid: the default 5, its sensitivity alternatives 3 and 8, and the wider
# 1.5 and 2 and narrower 13. Trend-only is the infinite-shape limit and
# no-progress the persistence benchmark.
DEFAULT_BACKTEST_PACE_SHAPES = (1.5, 2.0, 3.0, 5.0, 8.0, 13.0)
DEFAULT_BACKTEST_HORIZONS_MONTHS = (3, 6, 12)
ECI_QUANTILE_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
UPPER_TAIL_LEVELS = (0.9, 0.95, 0.99)
BOOTSTRAP_REPLICATES = 400
type CandidateVariant = Literal["baseline", "trend_only", "no_progress"]


def pinball_losses(
    realized: float, predicted: np.ndarray, levels: Sequence[float]
) -> np.ndarray:
    """Pinball (quantile) loss of each predicted quantile against one outcome."""
    probabilities = np.asarray(levels, dtype=float)
    residuals = realized - np.asarray(predicted, dtype=float)
    return np.maximum(probabilities * residuals, (probabilities - 1) * residuals)


def _add_months(at: datetime, months: int) -> datetime:
    year, month = divmod(at.month - 1 + months, 12)
    return at.replace(year=at.year + year, month=month + 1)


def _month_start(at: datetime) -> datetime:
    return at.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def origin_forecast(
    forecast: PublicFrontierForecast, origin: datetime
) -> PublicFrontierForecast:
    """The forecast an author with the same settings would have made at ``origin``.

    Only releases available at the origin are retained, labs without one are
    dropped, and the trend window keeps its length. The rate prior, an authored
    discrepancy and the pace shape are reused unchanged; an estimated
    discrepancy stays unset, so fitting re-estimates it from the origin's
    records.
    """
    releases = [row for row in forecast.releases if row.available_at <= origin]
    labs = {row.lab_id for row in releases}
    return PublicFrontierForecast.model_validate(
        {
            **forecast.model_dump(mode="json"),
            "as_of": origin.isoformat(),
            "trend_window_start": (
                origin - (forecast.as_of - forecast.trend_window_start)
            ).isoformat(),
            "coordinates": [
                row.model_dump(mode="json")
                for row in forecast.coordinates
                if row.lab_id in labs
            ],
            "releases": [row.model_dump(mode="json") for row in releases],
        }
    )


def backtest_public_frontier(
    forecast: PublicFrontierForecast,
    *,
    horizons_months: tuple[int, ...] = DEFAULT_BACKTEST_HORIZONS_MONTHS,
    pace_shapes: tuple[float, ...] = DEFAULT_BACKTEST_PACE_SHAPES,
    n_draws: int = 2000,
    seed: int = 0,
) -> PublicFrontierBacktest:
    """Score pace-dispersion candidates by rolling-origin pinball loss.

    Origins are month starts from the first one with a full trend window of
    release history through the last with a resolved horizon. Each scored case
    is one origin and horizon whose target date is on or before the forecast
    as-of; the realized value is the public frontier over the whole declared
    roster at that date.
    """
    forecast = PublicFrontierForecast.model_validate(forecast.model_dump(mode="json"))
    if (
        not horizons_months
        or any(isinstance(h, bool) or not isinstance(h, int) for h in horizons_months)
        or any(h < 1 for h in horizons_months)
        or len(set(horizons_months)) != len(horizons_months)
    ):
        raise ValueError("backtest horizons must be distinct positive month counts")
    if not pace_shapes or any(
        not np.isfinite(shape) or shape <= 0 for shape in pace_shapes
    ):
        raise ValueError("backtest pace shapes must be positive and finite")
    if len(set(pace_shapes)) != len(pace_shapes):
        raise ValueError("backtest pace shapes must be distinct")
    if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws < 100:
        raise ValueError("backtest requires at least 100 draws per origin")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("backtest seed must be a nonnegative integer")
    horizons = tuple(sorted(horizons_months))
    window = forecast.as_of - forecast.trend_window_start
    first_release = min(row.available_at for row in forecast.releases)
    origin = _month_start(first_release + window)
    if origin < first_release + window:
        origin = _add_months(origin, 1)
    candidates: list[tuple[str, CandidateVariant, float | None]] = [
        (f"pace_shape_{shape:g}", "baseline", float(shape))
        for shape in sorted(pace_shapes)
    ] + [("trend_only", "trend_only", None), ("no_progress", "no_progress", None)]
    levels = np.asarray(ECI_QUANTILE_LEVELS)
    losses: dict[str, dict[int, list[float]]] = {
        name: {h: [] for h in horizons} for name, _, _ in candidates
    }
    case_losses: dict[str, list[float]] = {name: [] for name, _, _ in candidates}
    case_origins: list[int] = []
    tail_losses: dict[str, list[float]] = {name: [] for name, _, _ in candidates}
    exceedances: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in candidates}
    case_horizons: list[int] = []
    central_50: dict[str, list[bool]] = {name: [] for name, _, _ in candidates}
    central_90: dict[str, list[bool]] = {name: [] for name, _, _ in candidates}
    rng = np.random.default_rng(seed)
    origins = []
    while _add_months(origin, horizons[0]) <= forecast.as_of:
        historical = origin_forecast(forecast, origin)
        posterior = fit_public_trend(historical)
        labs = [row.lab_id for row in historical.coordinates]
        anchors = np.asarray([posterior.labs[lab].as_of_value for lab in labs])
        rates = sample_positive_rate(posterior, n_draws, rng)[:, None]
        probability = rng.random((n_draws, len(labs)))
        origins.append(origin)
        for horizon in horizons:
            target = _add_months(origin, horizon)
            if target > forecast.as_of:
                continue
            realized = max(
                row.value for row in forecast.releases if row.available_at <= target
            )
            elapsed = (target - origin).total_seconds() / YEAR_SECONDS
            case_origins.append(len(origins) - 1)
            case_horizons.append(horizon)
            for name, variant, shape in candidates:
                if variant == "no_progress":
                    pace = np.zeros((n_draws, 1))
                elif variant == "trend_only":
                    pace = np.ones((n_draws, 1))
                else:
                    assert shape is not None
                    pace = pace_quantile(shape, probability, forecast.pace_cap)
                frontier = (anchors + rates * pace * elapsed).max(axis=1)
                if not np.isfinite(frontier).all():
                    raise ValueError(
                        "backtest frontier draw exceeds finite numerical support"
                    )
                quantiles = np.quantile(frontier, levels, method="inverted_cdf")
                loss = float(
                    np.mean(pinball_losses(realized, quantiles, ECI_QUANTILE_LEVELS))
                )
                losses[name][horizon].append(loss)
                case_losses[name].append(loss)
                upper = np.quantile(frontier, UPPER_TAIL_LEVELS, method="inverted_cdf")
                tail_losses[name].append(
                    float(np.mean(pinball_losses(realized, upper, UPPER_TAIL_LEVELS)))
                )
                exceedances[name].append(upper < realized)
                central_50[name].append(bool(quantiles[1] <= realized <= quantiles[3]))
                central_90[name].append(bool(quantiles[0] <= realized <= quantiles[4]))
        origin = _add_months(origin, 1)
    scored = len(central_50[candidates[0][0]])
    if not scored:
        raise ValueError(
            "backtest needs an origin with a full trend window of releases and a horizon resolved by as-of"
        )
    means = {name: float(np.mean(case_losses[name])) for name, _, _ in candidates}
    by_horizon = np.asarray(case_horizons)
    masks = {h: by_horizon == h for h in horizons if np.any(by_horizon == h)}
    lowest = min(means, key=lambda name: means[name])
    # Paired moving-block bootstrap over origins: blocks as long as the longest
    # horizon keep overlapping cases together; every candidate uses the same
    # resampled cases, so the SE is of the paired loss difference.
    block = min(horizons[-1], len(origins))
    starts = np.arange(len(origins) - block + 1)
    by_origin = np.asarray(case_origins)
    differences = {
        name: np.asarray(case_losses[name]) - np.asarray(case_losses[lowest])
        for name, _, _ in candidates
    }
    boot = np.random.default_rng([seed, 1])
    replicate_means = {name: [] for name, _, _ in candidates}
    for _ in range(BOOTSTRAP_REPLICATES):
        chosen = np.concatenate(
            [
                np.arange(start, start + block)
                for start in boot.choice(starts, size=-(-len(origins) // block))
            ]
        )[: len(origins)]
        cases = np.concatenate([np.flatnonzero(by_origin == index) for index in chosen])
        for name in replicate_means:
            replicate_means[name].append(float(np.mean(differences[name][cases])))
    rows = tuple(
        PublicBacktestCandidate(
            candidate_id=name,
            variant=variant,
            pace_shape=shape,
            mean_pinball_loss=float(
                np.mean([value for h in horizons for value in losses[name][h]])
            ),
            mean_pinball_loss_by_horizon={
                str(h): float(np.mean(losses[name][h]))
                for h in horizons
                if losses[name][h]
            },
            central_50_coverage=float(np.mean(central_50[name])),
            central_90_coverage=float(np.mean(central_90[name])),
            mean_upper_tail_pinball_loss=float(np.mean(tail_losses[name])),
            upper_exceedance_rates={
                f"p{round(level * 100)}": float(rate)
                for level, rate in zip(
                    UPPER_TAIL_LEVELS,
                    np.mean(np.asarray(exceedances[name]), axis=0),
                    strict=True,
                )
            },
            mean_upper_tail_pinball_loss_by_horizon={
                str(h): float(np.mean(np.asarray(tail_losses[name])[mask]))
                for h, mask in masks.items()
            },
            central_90_coverage_by_horizon={
                str(h): float(np.mean(np.asarray(central_90[name])[mask]))
                for h, mask in masks.items()
            },
            upper_exceedance_rates_by_horizon={
                str(h): {
                    f"p{round(level * 100)}": float(rate)
                    for level, rate in zip(
                        UPPER_TAIL_LEVELS,
                        np.mean(np.asarray(exceedances[name])[mask], axis=0),
                        strict=True,
                    )
                }
                for h, mask in masks.items()
            },
            loss_difference_vs_lowest=means[name] - means[lowest],
            loss_difference_se=0.0
            if name == lowest
            else float(np.std(replicate_means[name], ddof=1)),
        )
        for name, variant, shape in candidates
    )
    return PublicFrontierBacktest(
        forecast_sha256=forecast.digest(),
        quantile_levels=tuple(ECI_QUANTILE_LEVELS),
        upper_tail_levels=UPPER_TAIL_LEVELS,
        horizons_months=horizons,
        origins=tuple(origins),
        scored_cases=scored,
        pace_cap=forecast.pace_cap,
        n_draws=n_draws,
        seed=seed,
        bootstrap_block_origins=block,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        candidates=rows,
        lowest_loss_candidate=lowest,
        limitations=(
            "Scores use the authored retrospective current-scale releases; refitted scales and release-date proxies are not historical score vintages.",
            "The rate prior, the window length and the benchmark errors are reused at every origin, and an authored discrepancy is reused too (an estimated one is re-estimated from each origin's records); values set with knowledge of later releases leak information into earlier origins.",
            "Monthly origins and overlapping horizons make scored cases serially dependent; their count overstates independent evidence.",
            "A lab without a release at an origin is absent from that forecast but counts in the realized frontier once it releases.",
            "The target is the public frontier at each horizon, not a policy-period average or a loss; the lowest score selects among the listed candidates only.",
            "Pinball loss at p5-p95 says little about the right tail the pace shape controls; read the upper-tail loss and exceedance rates, and treat a loss difference within about two bootstrap standard errors as no evidence.",
            "Every pace-shape candidate is truncated at the forecast's pace cap, which the backtest does not vary; the scored quantiles reach at most p99 of the frontier, so they say little about the cap.",
        ),
    )
