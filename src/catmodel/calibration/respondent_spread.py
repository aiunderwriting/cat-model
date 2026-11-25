"""Fit the POOL-SPREAD and POOL-COMMON-BIAS factors from known-answer seeds.

The fit measures how wide and how biased the respondents' one-shot answers to
known-count questions were in one period. It is a diagnostic: it validates
no risk judgment, and no setup reads it.
"""

from __future__ import annotations

import calendar
import math
import multiprocessing
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal, cast

import numpy as np
from pydantic import ValidationError
from scipy.optimize import brentq
from scipy.special import (
    betainc,
    betaincc,
    gammainc,
    gammaincc,
    gammaln,
    logsumexp,
    ndtr,
    ndtri,
)

from catmodel.contracts.elicitation import Quantiles
from catmodel.contracts.incident import SpreadTailForm
from catmodel.contracts.respondent_calibration import (
    AuxiliarySeedPIT,
    BootstrapSummary,
    CalibrationArtifact,
    CalibrationCaptureIdentity,
    CalibrationInstrument,
    CalibrationModelTreatment,
    CalibrationSeed,
    CalibrationTreatments,
    CommonBiasRange,
    CookeDiagnostics,
    CookeScore,
    CurationSensitivityDiagnostics,
    DiagnosticReferencePercentiles,
    JointGrid,
    JointRange,
    LeaveOneBlockOutDiagnostics,
    LeaveOneBlockOutRow,
    MarginalBiasFit,
    MarginalFits,
    MarginalSpreadFit,
    MemorizationRow,
    ModelJointFit,
    PITAtSpread,
    PITReferenceDiagnostics,
    PoissonSpreadAlternative,
    PooledPhiSpreadAlternative,
    PrimaryPITDiagnostics,
    RespondentSeedFitRow,
    RespondentSpreadDiagnostics,
    RespondentSpreadFit,
    RespondentSpreadObservation,
    SeedPIT,
    SeedResolution,
    SeedResolutionSet,
    SimulatedReferenceDiagnostics,
    SpreadObservationAlternatives,
    SpreadRange,
    SuperBlockBootstrapDiagnostics,
    resolved_outcomes,
)
from catmodel.elicitation.contextual_risk import CONTEXTUAL_SYSTEM_PROMPT
from catmodel.elicitation.epistemic import (
    EPISTEMIC_KNOT_SCORES,
    epistemic_knots,
    knot_interpolant,
    stretched_score_quantiles,
    tail_widths,
)
from catmodel.frozen_calls import FROZEN_CALL_SCHEMA_VERSION, AcquisitionTreatment
from catmodel.hashing import canonical_json_sha256

K_BOUNDS = (0.5, 4.0)
B_BOUNDS = (0.1, 10.0)
K_GRID_POINTS = 64
B_GRID_POINTS = 101
K_COARSE_STRIDE = 3
B_COARSE_STRIDE = 5
_REFERENCE_GRID_POINTS = 55
_BOOTSTRAP_PERCENTILES = (0.05, 0.95)
_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(16)
_SCORE_LIMIT = 40.0
_TARGET_SIGMAS = np.arange(-40, 41, dtype=float)
_NORMAL_SCORE_SPLITS = np.asarray(
    [
        -39.75,
        -39.5,
        -39.0,
        -38.0,
        -36.0,
        -32.0,
        -16.0,
        -8.0,
        -4.0,
        -2.0,
        -1.0,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        36.0,
        38.0,
        39.0,
        39.5,
        39.75,
    ]
)
_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
_SQRT_2 = math.sqrt(2.0)
_LOG_FLOAT_MAX = math.log(float(np.finfo(float).max))
_MONTH_LABEL = re.compile(r"(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])")
_QUARTER_LABEL = re.compile(r"(?P<year>\d{4})-Q(?P<quarter>[1-4])")


def _validate_phi(phi: float) -> None:
    if not math.isfinite(phi) or phi < 0:
        raise ValueError("NB2 phi must be nonnegative and finite")


def seed_dispersion(
    seed: CalibrationSeed,
) -> tuple[float | None, int | None, str | None]:
    """Estimate a seed's NB2 phi from its pre-window count history.

    Monthly histories are aligned into non-overlapping window-sized sums; a
    quarterly history is usable only for a three-month window. Under independent
    periods the NB2 extra-variance contribution scales with period length, so
    ``phi`` is measured at the seed's window granularity. The log-count delta
    method estimates ``phi`` from the residual variance after subtracting the
    Poisson term. Its mean floor is 0.5: at very low counts the delta method
    overstates the Poisson contribution, which pushes the resulting phi toward
    zero rather than treating that approximation error as process dispersion.
    """
    if seed.unit != "count":
        return None, None, "seed unit is not count"
    if (
        seed.window_start.day != 1
        or seed.window_end.day
        != calendar.monthrange(seed.window_end.year, seed.window_end.month)[1]
    ):
        return None, None, "window is not month-aligned"

    window_months = (
        (seed.window_end.year - seed.window_start.year) * 12
        + seed.window_end.month
        - seed.window_start.month
        + 1
    )
    if not seed.evidence.history:
        return None, None, "history has no period labels"

    monthly_rows: list[tuple[int, float]] = []
    quarterly_rows: list[tuple[int, float]] = []
    granularity: str | None = None
    for row in seed.evidence.history:
        month_match = _MONTH_LABEL.fullmatch(row.period)
        quarter_match = _QUARTER_LABEL.fullmatch(row.period)
        if month_match is not None:
            row_granularity = "monthly"
            year = int(month_match.group("year"))
            month = int(month_match.group("month"))
            if year == 0:
                return (
                    None,
                    None,
                    "history period labels are not valid YYYY-MM or YYYY-Qn labels",
                )
            monthly_rows.append((year * 12 + month - 1, float(row.value)))
        elif quarter_match is not None:
            row_granularity = "quarterly"
            year = int(quarter_match.group("year"))
            quarter = int(quarter_match.group("quarter"))
            if year == 0:
                return (
                    None,
                    None,
                    "history period labels are not valid YYYY-MM or YYYY-Qn labels",
                )
            quarterly_rows.append((year * 4 + quarter - 1, float(row.value)))
        else:
            return (
                None,
                None,
                "history period labels are not valid YYYY-MM or YYYY-Qn labels",
            )
        if granularity is not None and row_granularity != granularity:
            return (
                None,
                None,
                "history period labels mix monthly and quarterly granularities",
            )
        granularity = row_granularity

    if granularity == "monthly":
        monthly_rows.sort()
        month_indices = [period for period, _ in monthly_rows]
        if len(set(month_indices)) != len(month_indices):
            return None, None, "monthly history contains duplicate periods"
        if any(
            current != previous + 1 for previous, current in pairwise(month_indices)
        ):
            return None, None, "monthly history is not contiguous"
        month_values = [value for _, value in monthly_rows]
        leading_partial = len(month_values) % window_months
        aligned_values = month_values[leading_partial:]
        period_values = [
            math.fsum(aligned_values[start : start + window_months])
            for start in range(0, len(aligned_values), window_months)
        ]
    else:
        if window_months != 3:
            return (
                None,
                None,
                f"quarterly history cannot be aligned to a {window_months}-month window",
            )
        quarterly_rows.sort()
        quarter_indices = [period for period, _ in quarterly_rows]
        if len(set(quarter_indices)) != len(quarter_indices):
            return None, None, "quarterly history contains duplicate periods"
        if any(
            current != previous + 1 for previous, current in pairwise(quarter_indices)
        ):
            return None, None, "quarterly history is not contiguous"
        period_values = [value for _, value in quarterly_rows]

    periods = len(period_values)
    if periods < 6:
        return None, periods, "fewer than 6 periods at the window's granularity"

    times = np.arange(periods, dtype=float)
    log_counts = np.log(np.asarray(period_values, dtype=float) + 0.5)
    design = np.column_stack((np.ones(periods), times))
    intercept, slope = np.linalg.lstsq(design, log_counts, rcond=None)[0]
    fitted_log_counts = intercept + slope * times
    residuals = log_counts - fitted_log_counts
    residual_variance = float(np.dot(residuals, residuals) / (periods - 2))
    with np.errstate(over="ignore", invalid="ignore"):
        fitted_means = np.exp(fitted_log_counts) - 0.5
    fitted_means = np.maximum(fitted_means, 0.5)
    poisson_log_variance = float(np.mean(1.0 / fitted_means))
    phi = max(0.0, residual_variance - poisson_log_variance)
    if not math.isfinite(phi):
        return (
            None,
            periods,
            "history trend fit produced a non-finite dispersion estimate",
        )
    return phi, periods, None


def seed_dispersions(
    seeds: Sequence[CalibrationSeed],
) -> tuple[dict[str, tuple[float, int | None, str | None]], float, tuple[str, ...]]:
    """Return final per-seed phi rows, the primary-history median and fallbacks.

    Only usable primary histories determine the pooled phi. Every other
    count-valued seed with an unusable history receives that same pooled value.
    """
    raw = {
        seed.seed_id: seed_dispersion(seed) for seed in seeds if seed.unit == "count"
    }
    primary_phi = [
        estimate[0]
        for seed in seeds
        if seed.role == "primary"
        and seed.unit == "count"
        and (estimate := raw[seed.seed_id])[0] is not None
    ]
    if not primary_phi:
        raise ValueError(
            "respondent spread cannot run: no primary seed has enough history "
            "at the window's granularity"
        )
    pooled_phi = float(np.median(np.asarray(primary_phi, dtype=float)))
    fallback_seed_ids = tuple(
        seed.seed_id
        for seed in seeds
        if seed.seed_id in raw and raw[seed.seed_id][0] is None
    )
    final = {
        seed_id: (
            pooled_phi if phi is None else phi,
            periods,
            reason,
        )
        for seed_id, (phi, periods, reason) in raw.items()
    }
    return final, pooled_phi, fallback_seed_ids


def calibration_treatments(
    instrument: CalibrationInstrument,
    plan_payload: Mapping[str, object],
    *,
    acquisition_treatments: Sequence[AcquisitionTreatment] | None = None,
) -> CalibrationTreatments:
    """Derive the calibration run's effective model and parser treatment."""
    try:
        execution_value = plan_payload["execution"]
        if not isinstance(execution_value, Mapping):
            raise ValueError
        execution = cast(Mapping[str, Any], execution_value)
        model_rows = execution["models"]
        settings = execution["settings"]
        if not isinstance(model_rows, list) or not isinstance(settings, Mapping):
            raise ValueError
        model_rows = cast(list[Any], model_rows)
        settings = cast(Mapping[str, Any], settings)
        output_budgets = settings["max_output_tokens_by_model"]
        parse_attempts = plan_payload["parse_attempts"]
        if not isinstance(output_budgets, Mapping):
            raise ValueError
        output_budgets = cast(Mapping[str, Any], output_budgets)
        if isinstance(parse_attempts, bool) or not isinstance(parse_attempts, int):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "collection plan lacks its frozen execution treatments"
        ) from exc

    sample_counts = Counter(row.model_id for row in instrument.respondents)
    expected_models = set(sample_counts)
    model_treatments = []
    seen_models: set[str] = set()
    for row in model_rows:
        if not isinstance(row, Mapping):
            raise ValueError("collection plan model rows must be objects")
        row = cast(Mapping[str, Any], row)
        try:
            model_id = row["model_id"]
            if not isinstance(model_id, str):
                raise ValueError
            budget = output_budgets[model_id]
            treatment = CalibrationModelTreatment(
                provider=row["provider"],
                model_id=model_id,
                reasoning_effort=row["reasoning_effort"],
                max_output_tokens=budget,
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise ValueError("collection plan model treatment is incomplete") from exc
        if model_id in seen_models:
            raise ValueError("collection plan repeats a model treatment")
        seen_models.add(model_id)
        model_treatments.append(treatment)
    if seen_models != expected_models:
        raise ValueError("collection plan models differ from the respondent roster")
    if set(output_budgets) != expected_models:
        raise ValueError("output-token budgets differ from the respondent roster")

    if acquisition_treatments is not None:
        by_model: dict[str, list[AcquisitionTreatment]] = defaultdict(list)
        for treatment in acquisition_treatments:
            by_model[treatment.model_id].append(treatment)
        if set(by_model) != expected_models:
            raise ValueError("acquisition evidence differs from the treatment roster")
        evidence_models = []
        for planned_treatment in model_treatments:
            model_id = planned_treatment.model_id
            rows = by_model[model_id]
            for row in rows:
                differences = tuple(
                    name
                    for name, observed, planned in (
                        ("provider", row.provider_name, planned_treatment.provider),
                        (
                            "reasoning_effort",
                            row.reasoning_effort,
                            planned_treatment.reasoning_effort,
                        ),
                        (
                            "max_output_tokens",
                            row.max_output_tokens,
                            planned_treatment.max_output_tokens,
                        ),
                    )
                    if observed != planned
                )
                if differences:
                    raise ValueError(
                        f"acquisition treatment differs from the collection plan for "
                        f"model {model_id!r}: {', '.join(differences)}"
                    )
            first = rows[0]
            if any(
                row.parse_attempts != first.parse_attempts
                or row.retry_protocol != first.retry_protocol
                for row in rows
            ):
                raise ValueError("acquisition records disagree within a model")
            if first.parse_attempts != parse_attempts:
                raise ValueError("acquisition parser budget differs from the plan")
            evidence_model = CalibrationModelTreatment(
                provider=cast(Any, first.provider_name),
                model_id=model_id,
                reasoning_effort=cast(Any, first.reasoning_effort),
                max_output_tokens=first.max_output_tokens,
            )
            evidence_models.append(evidence_model)
        model_treatments = evidence_models

    try:
        return CalibrationTreatments(
            models=tuple(model_treatments),
            samples_per_model=dict(sample_counts),
            parse_attempts=parse_attempts,
            frozen_call_schema_version=FROZEN_CALL_SCHEMA_VERSION,
            risk_contract_version=instrument.risk_contract_version,
            system_prompt_sha256=canonical_json_sha256(CONTEXTUAL_SYSTEM_PROMPT),
        )
    except (TypeError, ValidationError) as exc:
        raise ValueError("collection plan treatment fields are invalid") from exc


def _tail_widths_for_grid(
    values: np.ndarray,
    spreads: np.ndarray,
    tail_form: SpreadTailForm,
) -> tuple[np.ndarray, np.ndarray]:
    widths = np.asarray(
        [
            tail_widths(values, spread=float(spread), tail_form=tail_form)
            for spread in spreads
        ]
    )
    return widths[:, 0], widths[:, 1]


def _log_quantiles_at_scores(
    values: np.ndarray,
    scores: np.ndarray,
    spreads: np.ndarray,
    tail_form: SpreadTailForm = "calibrated_p95_peak",
) -> np.ndarray:
    """Log Q(Phi(s)) at quadrature scores, preserving zero atoms and overflow."""
    if np.all(values == values[0]):
        if values[0] == 0:
            return np.full(scores.shape, -math.inf)
        return np.full(scores.shape, math.log(float(values[0])))

    lower_width, upper_width = _tail_widths_for_grid(values, spreads, tail_form)
    lower_edge = EPISTEMIC_KNOT_SCORES[0]
    upper_edge = EPISTEMIC_KNOT_SCORES[4]
    log_values = np.full(scores.shape, -math.inf)
    lower = scores < lower_edge
    upper = scores > upper_edge
    central = ~(lower | upper)

    lower_anchor = float(values[0])
    if lower_anchor > 0 and np.any(lower):
        log_values[lower] = math.log(lower_anchor) + np.broadcast_to(
            lower_width[:, None], scores.shape
        )[lower] * (scores[lower] - lower_edge)
    upper_anchor = float(values[4])
    if upper_anchor > 0 and np.any(upper):
        log_values[upper] = math.log(upper_anchor) + np.broadcast_to(
            upper_width[:, None], scores.shape
        )[upper] * (scores[upper] - upper_edge)
    if np.any(central):
        interpolant = knot_interpolant(values)
        ranks = ndtr(scores[central])
        with np.errstate(divide="ignore"):
            log_values[central] = np.log(interpolant(ranks))
    return log_values


def _central_inverse_scores(values: np.ndarray, log_targets: np.ndarray) -> np.ndarray:
    """Invert the monotone central PCHIP at the requested log-quantiles."""
    lower_log = -math.inf if values[0] == 0 else math.log(float(values[0]))
    upper_log = -math.inf if values[4] == 0 else math.log(float(values[4]))
    roots = np.full(log_targets.shape, np.nan)
    lower_edge = EPISTEMIC_KNOT_SCORES[0]
    upper_edge = EPISTEMIC_KNOT_SCORES[4]
    at_lower = log_targets == lower_log
    at_upper = log_targets == upper_log
    roots[at_lower] = lower_edge
    roots[at_upper] = upper_edge
    central = (log_targets > lower_log) & (log_targets < upper_log)
    if np.any(central):
        target_values = np.exp(log_targets[central])
        lo = np.full(target_values.shape, 0.05)
        hi = np.full(target_values.shape, 0.95)
        interpolant = knot_interpolant(values)
        for _ in range(52):
            mid = (lo + hi) / 2
            crossed = interpolant(mid) >= target_values
            hi = np.where(crossed, mid, hi)
            lo = np.where(crossed, lo, mid)
        roots[central] = ndtri(hi)
    return roots


def _logsumexp_rows(values: np.ndarray) -> np.ndarray:
    """Row-wise log-sum-exp of a 2-D array; an all ``-inf`` row gives ``-inf``."""
    maxima = np.max(values, axis=1, keepdims=True)
    shift = np.where(np.isfinite(maxima), maxima, 0.0)
    with np.errstate(divide="ignore"):
        return np.log(np.sum(np.exp(values - shift), axis=1)) + shift[:, 0]


def _scalar_piecewise_polynomial(
    interpolant: Any,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Breakpoints and per-interval coefficients for fast scalar evaluation."""
    breaks = tuple(float(value) for value in interpolant.x)
    coefficients = tuple(
        tuple(float(value) for value in interpolant.c[:, index])
        for index in range(interpolant.c.shape[1])
    )
    return breaks, coefficients


def _evaluate_scalar_polynomial(
    polynomial: tuple[tuple[float, ...], tuple[tuple[float, ...], ...]],
    point: float,
) -> float:
    """Evaluate a piecewise polynomial, extrapolating with its end pieces."""
    breaks, coefficients = polynomial
    index = min(max(bisect_right(breaks, point) - 1, 0), len(coefficients) - 1)
    offset = point - breaks[index]
    value = 0.0
    for coefficient in coefficients[index]:
        value = value * offset + coefficient
    return value


def _likelihood_peak_score(
    values: np.ndarray,
    outcome: int,
    spread: float,
    lower_width: float,
    upper_width: float,
    limit: float,
    interpolant: Any,
    derivative_interpolant: Any,
    phi: float,
) -> tuple[float, float]:
    """Locate the score-space mode to split a panel around narrow likelihoods."""
    lower_edge = EPISTEMIC_KNOT_SCORES[0]
    upper_edge = EPISTEMIC_KNOT_SCORES[4]
    normalizer = _LOG_SQRT_2PI

    def derivative(score: float) -> float:
        if score < lower_edge:
            anchor = float(values[0])
            width = lower_width
            if anchor == 0 or width == 0:
                rate = anchor
                rate_slope = 0.0
            else:
                log_rate = math.log(anchor) + width * (score - lower_edge)
                rate = math.exp(log_rate) if log_rate <= _LOG_FLOAT_MAX else math.inf
                rate_slope = width
        elif score > upper_edge:
            anchor = float(values[4])
            width = upper_width
            if anchor == 0 or width == 0:
                rate = anchor
                rate_slope = 0.0
            else:
                log_rate = math.log(anchor) + width * (score - upper_edge)
                rate = math.exp(log_rate) if log_rate <= _LOG_FLOAT_MAX else math.inf
                rate_slope = width
        else:
            probability = 0.5 * math.erfc(-score / _SQRT_2)
            rate = _evaluate_scalar_polynomial(interpolant, probability)
            rate_derivative = _evaluate_scalar_polynomial(
                derivative_interpolant, probability
            ) * math.exp(-0.5 * score * score - normalizer)
            rate_slope = 0.0 if rate <= 0 else rate_derivative / rate

        if rate == 0:
            likelihood_derivative = outcome * rate_slope
        elif math.isinf(rate):
            likelihood_derivative = -math.inf if phi == 0 else -rate_slope / phi
        elif phi == 0:
            likelihood_derivative = (outcome - rate) * rate_slope
        elif phi * rate > 1:
            likelihood_derivative = (outcome / rate - 1) / (phi + 1 / rate) * rate_slope
        else:
            likelihood_derivative = (outcome - rate) / (1 + phi * rate) * rate_slope
        return likelihood_derivative - score / (spread * spread)

    at_zero = derivative(0.0)
    if at_zero == 0 or math.isnan(at_zero):
        return 0.0, spread
    if at_zero > 0:
        left, right = 0.0, min(limit, max(1.0, spread))
        while right < limit and derivative(right) > 0:
            right = min(limit, right * 2)
        if derivative(right) > 0:
            return right, spread
    else:
        left, right = -min(limit, max(1.0, spread)), 0.0
        while left > -limit and derivative(left) < 0:
            left = max(-limit, left * 2)
        if derivative(left) < 0:
            return left, spread
    peak = float(brentq(derivative, left, right))
    step = max(1e-5, spread * 1e-3)
    curvature = -(derivative(peak + step) - derivative(peak - step)) / (2 * step)
    width = (
        1 / math.sqrt(curvature)
        if math.isfinite(curvature) and curvature > 0
        else spread
    )
    return peak, width


def _quadrature_grid(
    values: np.ndarray,
    outcome: int,
    spreads: np.ndarray,
    *,
    phi: float = 0.0,
    tail_form: SpreadTailForm = "calibrated_p95_peak",
) -> tuple[np.ndarray, np.ndarray]:
    """Build 16-point score-space panels around knots, tails and likelihood mass."""
    _validate_phi(phi)
    lower_edge = EPISTEMIC_KNOT_SCORES[0]
    upper_edge = EPISTEMIC_KNOT_SCORES[4]
    relative_variance = 1 / (outcome + 1) + phi
    sigma = math.sqrt(relative_variance)
    log_targets = math.log(outcome + 1) + _TARGET_SIGMAS * sigma
    lower_log = -math.inf if values[0] == 0 else math.log(float(values[0]))
    upper_log = -math.inf if values[4] == 0 else math.log(float(values[4]))
    target_roots = _central_inverse_scores(values, log_targets)
    lower_tail = log_targets < lower_log
    upper_tail = log_targets > upper_log
    at_edges = log_targets == lower_log
    at_edges_upper = log_targets == upper_log
    target_roots[lower_tail | upper_tail] = np.nan
    lower_width, upper_width = _tail_widths_for_grid(values, spreads, tail_form)
    roots_by_spread = np.broadcast_to(
        target_roots, (len(spreads), len(log_targets))
    ).copy()
    roots_by_spread[:, lower_tail] = np.where(
        lower_width[:, None] > 0,
        lower_edge
        + (log_targets[lower_tail][None, :] - lower_log)
        / np.where(lower_width[:, None] > 0, lower_width[:, None], 1.0),
        np.nan,
    )
    roots_by_spread[:, upper_tail] = np.where(
        upper_width[:, None] > 0,
        upper_edge
        + (log_targets[upper_tail][None, :] - upper_log)
        / np.where(upper_width[:, None] > 0, upper_width[:, None], 1.0),
        np.nan,
    )
    roots_by_spread[:, at_edges] = lower_edge
    roots_by_spread[:, at_edges_upper] = upper_edge
    finite_root_extent = np.max(
        np.where(np.isfinite(roots_by_spread), np.abs(roots_by_spread), 0.0), axis=1
    )
    limits = np.maximum(_SCORE_LIMIT * spreads, finite_root_extent + 8 * spreads)
    roots_by_spread[:, lower_tail] = np.where(
        np.isfinite(roots_by_spread[:, lower_tail]),
        roots_by_spread[:, lower_tail],
        -limits[:, None],
    )
    roots_by_spread[:, upper_tail] = np.where(
        np.isfinite(roots_by_spread[:, upper_tail]),
        roots_by_spread[:, upper_tail],
        limits[:, None],
    )
    roots_by_spread = np.where(
        np.isnan(roots_by_spread), -limits[:, None], roots_by_spread
    )

    pchip = knot_interpolant(values)
    peak_interpolant = _scalar_piecewise_polynomial(pchip)
    peak_derivative_interpolant = _scalar_piecewise_polynomial(pchip.derivative())
    peak_splits = []
    for spread, lower, upper, limit in zip(
        spreads, lower_width, upper_width, limits, strict=True
    ):
        peak, width = _likelihood_peak_score(
            values,
            outcome,
            float(spread),
            float(lower),
            float(upper),
            float(limit),
            peak_interpolant,
            peak_derivative_interpolant,
            phi,
        )
        peak_splits.append(
            peak + width * np.asarray([-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0])
        )
    peak_scores = np.asarray(peak_splits)

    knot_scores = np.broadcast_to(EPISTEMIC_KNOT_SCORES, (len(spreads), 5))
    normal_splits = spreads[:, None] * _NORMAL_SCORE_SPLITS[None, :]
    endpoints = np.column_stack((-limits, limits))
    breakpoints = np.concatenate(
        (endpoints, knot_scores, normal_splits, roots_by_spread, peak_scores), axis=1
    )
    breakpoints = np.clip(breakpoints, -limits[:, None], limits[:, None])
    breakpoints.sort(axis=1)

    left, right = breakpoints[:, :-1], breakpoints[:, 1:]
    half = (right - left) / 2
    center = (right + left) / 2
    scores = center[..., None] + half[..., None] * _QUADRATURE_NODES
    scores = scores.reshape(len(spreads), -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_weights = (
            np.log(half[..., None])
            + np.log(_QUADRATURE_WEIGHTS)[None, None, :]
            - _LOG_SQRT_2PI
            - (scores.reshape(len(spreads), -1, 16) / spreads[:, None, None]) ** 2 / 2
            - np.log(spreads[:, None, None])
        )
    # The mapped GL weights include one half-width; the density adds none.
    log_weights = log_weights.reshape(len(spreads), -1)
    return scores, log_weights


def _log_observation_likelihood(
    log_lambda: np.ndarray,
    outcome: int,
    *,
    phi: float = 0.0,
) -> np.ndarray:
    """NB2 log mass, using its Poisson limit when ``phi`` is zero."""
    _validate_phi(phi)
    if phi > 0:
        negative_binomial_shape = 1 / phi
        result = np.full(log_lambda.shape, -math.inf)
        zero_rate = np.isneginf(log_lambda)
        finite = np.isfinite(log_lambda)
        if outcome == 0:
            result[zero_rate] = 0.0
        if np.any(finite):
            log_shape = math.log(negative_binomial_shape)
            log_denominator = np.logaddexp(log_shape, log_lambda[finite])
            result[finite] = (
                gammaln(outcome + negative_binomial_shape)
                - gammaln(negative_binomial_shape)
                - gammaln(outcome + 1)
                + negative_binomial_shape * (log_shape - log_denominator)
                + outcome * (log_lambda[finite] - log_denominator)
            )
        return result

    result = np.full(log_lambda.shape, -math.inf)
    zero_rate = np.isneginf(log_lambda)
    finite = np.isfinite(log_lambda) & (log_lambda <= _LOG_FLOAT_MAX)
    if outcome == 0:
        result[zero_rate] = 0.0
        result[finite] = -np.exp(log_lambda[finite])
    else:
        result[finite] = (
            outcome * log_lambda[finite]
            - np.exp(log_lambda[finite])
            - float(gammaln(outcome + 1))
        )
    return result


def _log_probability_values_grid(
    values: np.ndarray,
    outcome: int,
    spreads: np.ndarray,
    *,
    phi: float = 0.0,
    tail_form: SpreadTailForm = "calibrated_p95_peak",
) -> np.ndarray:
    """Integrate the NB2 count likelihood at each spread in log space."""
    spreads = np.asarray(spreads, dtype=float)
    if not np.all(np.isfinite(spreads)) or np.any(spreads <= 0):
        raise ValueError("rank spread factors must be positive and finite")
    if np.all(values == values[0]):
        log_lambda = (
            np.full(len(spreads), -math.inf)
            if values[0] == 0
            else np.full(len(spreads), math.log(float(values[0])))
        )
        return _log_observation_likelihood(log_lambda, outcome, phi=phi)

    scores, log_weights = _quadrature_grid(
        values, outcome, spreads, phi=phi, tail_form=tail_form
    )
    log_lambda = _log_quantiles_at_scores(values, scores, spreads, tail_form)
    log_likelihood = _log_observation_likelihood(
        log_lambda.reshape(-1), outcome, phi=phi
    ).reshape(log_lambda.shape)
    return _logsumexp_rows(log_weights + log_likelihood)


def _fit_scores(scores: np.ndarray, log_grid: np.ndarray) -> tuple[float, int]:
    if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise ValueError("spread fit scores cannot contain NaN or positive infinity")
    if np.all(np.isneginf(scores)):
        reference_index = int(np.argmin(np.abs(log_grid)))
        return float(math.exp(log_grid[reference_index])), reference_index
    index = int(np.argmax(scores))
    if index == 0 or index == len(log_grid) - 1:
        return float(math.exp(log_grid[index])), index
    left, center, right = scores[index - 1 : index + 2]
    if not np.all(np.isfinite((left, center, right))):
        offset = 0.0
    else:
        denominator = left - 2 * center + right
        if denominator >= 0 or denominator == 0:
            offset = 0.0
        else:
            offset = 0.5 * (left - right) / denominator
            offset = float(np.clip(offset, -1.0, 1.0))
    step = float(log_grid[1] - log_grid[0])
    return float(math.exp(log_grid[index] + offset * step)), index


def _parabola_offset(scores: np.ndarray, index: int) -> float:
    if index == 0 or index == len(scores) - 1:
        return 0.0
    left, center, right = scores[index - 1 : index + 2]
    if not np.all(np.isfinite((left, center, right))):
        return 0.0
    denominator = left - 2 * center + right
    if denominator >= 0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))


def _refine_joint_surface(
    surface: np.ndarray,
    k_indices: np.ndarray,
    b_indices: np.ndarray,
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    index: tuple[int, int],
) -> tuple[float, float]:
    """Refine a fine-grid maximum in log k and log b coordinates."""
    k_index, b_index = index
    if np.all(np.isneginf(surface)):
        return 1.0, 1.0

    local_k = int(np.flatnonzero(k_indices == k_index)[0])
    local_b = int(np.flatnonzero(b_indices == b_index)[0])
    k_bound = k_index in (0, len(k_grid) - 1)
    b_bound = b_index in (0, len(b_grid) - 1)
    k_offset = 0.0
    b_offset = 0.0

    if k_bound or b_bound:
        if not k_bound:
            k_offset = _parabola_offset(surface[:, local_b], local_k)
        if not b_bound:
            b_offset = _parabola_offset(surface[local_k, :], local_b)
    else:
        stencil = surface[local_k - 1 : local_k + 2, local_b - 1 : local_b + 2]
        if np.all(np.isfinite(stencil)):
            design = []
            response = []
            for u in (-1.0, 0.0, 1.0):
                for v in (-1.0, 0.0, 1.0):
                    design.append((1.0, u, v, u * u, u * v, v * v))
                    response.append(stencil[int(u + 1), int(v + 1)])
            coefficients = np.linalg.lstsq(
                np.asarray(design), np.asarray(response), rcond=None
            )[0]
            hessian = np.asarray(
                [
                    [2 * coefficients[3], coefficients[4]],
                    [coefficients[4], 2 * coefficients[5]],
                ]
            )
            if np.all(np.linalg.eigvalsh(hessian) < 0):
                gradient = coefficients[1:3]
                offsets = -np.linalg.solve(hessian, gradient)
                k_offset, b_offset = np.clip(offsets, -1.0, 1.0)
            else:
                k_offset = _parabola_offset(surface[:, local_b], local_k)
                b_offset = _parabola_offset(surface[local_k, :], local_b)
        else:
            k_offset = _parabola_offset(surface[:, local_b], local_k)
            b_offset = _parabola_offset(surface[local_k, :], local_b)

    k_step = float(np.log(k_grid[1]) - np.log(k_grid[0]))
    b_step = float(np.log(b_grid[1]) - np.log(b_grid[0]))
    return (
        float(math.exp(math.log(float(k_grid[k_index])) + k_offset * k_step)),
        float(math.exp(math.log(float(b_grid[b_index])) + b_offset * b_step)),
    )


def _fit_matrix(matrix: np.ndarray, log_grid: np.ndarray) -> tuple[float, int]:
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return 1.0, len(log_grid) // 2
    return _fit_scores(np.sum(matrix, axis=0), log_grid)


def _interpolate_log_score(
    row: np.ndarray, log_grid: np.ndarray, log_spread: float
) -> float:
    if log_spread <= log_grid[0]:
        return float(row[0])
    if log_spread >= log_grid[-1]:
        return float(row[-1])
    right = int(np.searchsorted(log_grid, log_spread))
    left = right - 1
    if log_grid[right] == log_spread:
        return float(row[right])
    if not np.all(np.isfinite((row[left], row[right]))):
        return -math.inf
    if right + 1 >= len(log_grid):
        return float(np.interp(log_spread, log_grid, row))
    # A local quadratic keeps leave-out scoring consistent with the refined fit.
    center = (
        right
        if log_spread - log_grid[right] < log_grid[right] - log_grid[left]
        else left
    )
    if center == 0 or center == len(log_grid) - 1:
        return float(np.interp(log_spread, log_grid, row))
    x0 = log_grid[center - 1 : center + 2]
    y0 = row[center - 1 : center + 2]
    if not np.all(np.isfinite(y0)):
        return float(np.interp(log_spread, log_grid[[left, right]], row[[left, right]]))
    coefficients = np.polyfit(x0, y0, 2)
    return float(np.polyval(coefficients, log_spread))


def _conditional_k_leave_one_block_out_gain(
    matrix: np.ndarray, block_ids: Sequence[str], log_grid: np.ndarray
) -> float | None:
    """Total held-out log-score gain of the spread refitted without each block
    over k = 1, or None when a held-out score is not finite."""
    total_gain = 0.0
    total_gain_available = True
    unique_blocks = tuple(dict.fromkeys(block_ids))
    for block in unique_blocks:
        held = np.asarray([value == block for value in block_ids], dtype=bool)
        training = ~held
        if np.any(training):
            fitted, _ = _fit_matrix(matrix[training], log_grid)
        else:
            fitted = 1.0
        fitted_log = math.log(fitted)
        score_fit = math.fsum(
            _interpolate_log_score(row, log_grid, fitted_log) for row in matrix[held]
        )
        score_one = math.fsum(
            _interpolate_log_score(row, log_grid, 0.0) for row in matrix[held]
        )
        if math.isfinite(score_fit) and math.isfinite(score_one):
            total_gain += score_fit - score_one
        else:
            total_gain_available = False
    return total_gain if total_gain_available else None


def cooke_calibration_score(pits: Sequence[float]) -> CookeScore:
    """Cooke's calibration score on the fixed six-bin 5/25/50/75/95 PIT bins."""
    values = np.asarray(pits, dtype=float)
    if (
        values.ndim != 1
        or len(values) == 0
        or not np.all(np.isfinite(values) & (values >= 0) & (values <= 1))
    ):
        raise ValueError("Cooke calibration requires one or more PIT values in [0, 1]")
    edges = np.asarray([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    expected = np.asarray([0.05, 0.20, 0.25, 0.25, 0.20, 0.05])
    counts, _ = np.histogram(values, bins=edges)
    observed = counts / len(values)
    nonempty = observed > 0
    divergence = float(
        np.sum(observed[nonempty] * np.log(observed[nonempty] / expected[nonempty]))
    )
    statistic = 2 * len(values) * divergence
    score = float(gammaincc(5 / 2, statistic / 2))
    return CookeScore(
        observed_proportions=cast(
            tuple[float, float, float, float, float, float],
            tuple(float(x) for x in observed),
        ),
        expected_proportions=cast(
            tuple[float, float, float, float, float, float],
            tuple(float(x) for x in expected),
        ),
        divergence=divergence,
        statistic=statistic,
        degrees_of_freedom=5,
        score=score,
    )


def _cdf_sf_at_rate(
    log_lambda: np.ndarray,
    outcome: int,
    *,
    phi: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if outcome < 0:
        return np.zeros(log_lambda.shape), np.ones(log_lambda.shape)
    _validate_phi(phi)
    rates = np.zeros(log_lambda.shape)
    finite = np.isfinite(log_lambda) & (log_lambda <= _LOG_FLOAT_MAX)
    rates[finite] = np.exp(log_lambda[finite])
    rates[np.isposinf(log_lambda) | (log_lambda > _LOG_FLOAT_MAX)] = math.inf
    if phi > 0:
        negative_binomial_shape = 1 / phi
        probabilities = np.ones(log_lambda.shape)
        positive_rate = rates > 0
        probabilities[positive_rate] = np.exp(
            -np.logaddexp(0, math.log(phi) + np.log(rates[positive_rate]))
        )
        return (
            betainc(negative_binomial_shape, outcome + 1, probabilities),
            betaincc(negative_binomial_shape, outcome + 1, probabilities),
        )
    return gammaincc(outcome + 1, rates), gammainc(outcome + 1, rates)


def _count_cdf_pairs(
    values: np.ndarray,
    outcome: int,
    spreads: Sequence[float],
    *,
    phi: float = 0.0,
) -> np.ndarray:
    """Integrated ``F(x-1)``, ``F(x)``, ``P(X > x-1)`` and ``P(X > x)``.

    One score-space grid per spread row serves all four integrals: the
    incomplete beta or gamma function gives ``F(x)`` and ``P(X > x)`` at each
    node, and the count mass at ``x`` gives the ``x - 1`` pair as
    ``F(x) - p(x)`` and ``P(X > x) + p(x)``. Rows are independent, so evaluating
    several spreads together equals evaluating them one at a time.
    """
    spread_array = np.asarray(spreads, dtype=float)
    scores, log_weights = _quadrature_grid(values, outcome, spread_array, phi=phi)
    log_lambda = _log_quantiles_at_scores(values, scores, spread_array).reshape(-1)
    upper_cdf, upper_sf = _cdf_sf_at_rate(log_lambda, outcome, phi=phi)
    if outcome == 0:
        lower_cdf = np.zeros(log_lambda.shape)
        lower_sf = np.ones(log_lambda.shape)
    else:
        mass = np.exp(_log_observation_likelihood(log_lambda, outcome, phi=phi))
        lower_cdf = np.clip(upper_cdf - mass, 0.0, 1.0)
        lower_sf = np.clip(upper_sf + mass, 0.0, 1.0)

    def integrate(probability: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            log_probability = np.log(probability).reshape(scores.shape)
        return np.clip(np.exp(_logsumexp_rows(log_weights + log_probability)), 0, 1)

    return np.column_stack(
        (
            integrate(lower_cdf),
            integrate(upper_cdf),
            integrate(lower_sf),
            integrate(upper_sf),
        )
    )


def _pooled_count_cdf_pairs(
    laws: Sequence[tuple[str, Quantiles]],
    outcome: int,
    spreads: Sequence[float],
    *,
    phi: float = 0.0,
    scale: float = 1.0,
) -> list[tuple[float, float, float, float]]:
    """Equally model- then sample-weighted CDF pairs, one per spread."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("count-law scale must be positive and finite")
    by_model: dict[str, list[Quantiles]] = defaultdict(list)
    for model_id, rate in laws:
        by_model[model_id].append(rate)
    if not by_model:
        raise ValueError("a pooled count PIT needs at least one law")
    mixed = np.zeros((len(spreads), 4))
    for samples in by_model.values():
        model_value = np.zeros((len(spreads), 4))
        for rate in samples:
            model_value += _count_cdf_pairs(
                epistemic_knots(rate, 1.0) * scale, outcome, spreads, phi=phi
            ) / len(samples)
        mixed += model_value / len(by_model)
    clipped = np.clip(mixed, 0, 1)
    return [
        (float(row[0]), float(row[1]), float(row[2]), float(row[3])) for row in clipped
    ]


def _randomized_pit(
    cdf_pair: tuple[float, float, float, float], uniform: float
) -> float:
    lower_cdf, upper_cdf, lower_sf, upper_sf = cdf_pair
    if upper_cdf <= 0.5:
        value = lower_cdf + uniform * (upper_cdf - lower_cdf)
    else:
        value = 1 - ((1 - uniform) * lower_sf + uniform * upper_sf)
    return float(np.clip(value, 0, 1))


def _quantile_cdf_pair(
    values: np.ndarray, value: float, spread: float, tail_form: SpreadTailForm
) -> tuple[float, float]:
    """Return ``F(value-)`` and ``F(value)`` for one stretched quantile law."""
    if np.all(values == values[0]):
        if value < values[0]:
            return 0.0, 0.0
        if value == values[0]:
            return 0.0, 1.0
        return 1.0, 1.0
    widths = tail_widths(values, spread=spread, tail_form=tail_form)
    lower_width, upper_width = widths
    lower_edge = EPISTEMIC_KNOT_SCORES[0]
    upper_edge = EPISTEMIC_KNOT_SCORES[4]
    lower_value, upper_value = float(values[0]), float(values[4])

    if value < lower_value:
        if value <= 0 or lower_value == 0 or lower_width == 0:
            return 0.0, 0.0
        score = lower_edge + (math.log(value) - math.log(lower_value)) / lower_width
        cdf = float(ndtr(score / spread))
        return cdf, cdf

    if value > upper_value:
        if upper_width == 0 or upper_value == 0:
            return 1.0, 1.0
        score = upper_edge + (math.log(value) - math.log(upper_value)) / upper_width
        cdf = float(ndtr(score / spread))
        return cdf, cdf

    interpolant = knot_interpolant(values)
    if value == lower_value and (lower_width == 0 or lower_value == 0):
        lower_score = -math.inf
    elif value == lower_value:
        lower_score = lower_edge
    elif value > lower_value:
        lo, hi = 0.05, 0.95
        for _ in range(56):
            mid = (lo + hi) / 2
            if interpolant(mid) >= value:
                hi = mid
            else:
                lo = mid
        lower_score = float(ndtri(hi))
    else:
        lower_score = -math.inf

    if value == upper_value and (upper_width == 0 or upper_value == 0):
        upper_score = math.inf
    elif value == upper_value:
        upper_score = upper_edge
    elif value < upper_value and value >= lower_value:
        lo, hi = 0.05, 0.95
        for _ in range(56):
            mid = (lo + hi) / 2
            if interpolant(mid) <= value:
                lo = mid
            else:
                hi = mid
        upper_score = float(ndtri(lo))
    else:
        upper_score = -math.inf
    return (
        float(np.clip(ndtr(lower_score / spread), 0, 1)),
        float(np.clip(ndtr(upper_score / spread), 0, 1)),
    )


def _pooled_quantile_cdf_pair(
    laws: Sequence[tuple[str, Quantiles]],
    value: float,
    spread: float,
    tail_form: SpreadTailForm,
) -> tuple[float, float] | None:
    """Return the equally model and sample weighted ``F(x-)`` and ``F(x)``."""
    by_model: dict[str, list[Quantiles]] = defaultdict(list)
    for model_id, rate in laws:
        by_model[model_id].append(rate)
    if not by_model:
        return None
    mixture_lower = math.fsum(
        math.fsum(
            _quantile_cdf_pair(epistemic_knots(rate, 1.0), value, spread, tail_form)[0]
            for rate in samples
        )
        / len(samples)
        for samples in by_model.values()
    ) / len(by_model)
    mixture_upper = math.fsum(
        math.fsum(
            _quantile_cdf_pair(epistemic_knots(rate, 1.0), value, spread, tail_form)[1]
            for rate in samples
        )
        / len(samples)
        for samples in by_model.values()
    ) / len(by_model)
    return (
        float(np.clip(mixture_lower, 0, 1)),
        float(np.clip(mixture_upper, 0, 1)),
    )


def _randomized_quantile_pit(cdf_pair: tuple[float, float], uniform: float) -> float:
    return float(np.clip(cdf_pair[0] + uniform * (cdf_pair[1] - cdf_pair[0]), 0, 1))


def _pit_summary(
    seeds: Sequence[tuple[str, str, float]],
) -> PITAtSpread:
    pits = np.asarray([row[2] for row in seeds], dtype=float)
    histogram_bins = np.minimum((pits * 10).astype(int), 9)
    counts = np.bincount(histogram_bins, minlength=10)
    return PITAtSpread(
        median=float(np.median(pits)),
        central_50_coverage=float(np.mean((pits >= 0.25) & (pits <= 0.75))),
        central_90_coverage=float(np.mean((pits >= 0.05) & (pits <= 0.95))),
        histogram_10=cast(
            tuple[int, int, int, int, int, int, int, int, int, int],
            tuple(int(count) for count in counts),
        ),
        seeds=tuple(
            SeedPIT(seed_id=seed_id, block_id=block_id, pit=pit)
            for seed_id, block_id, pit in seeds
        ),
    )


def _aggregate_law_scores(
    rows: Sequence[tuple[str, np.ndarray]],
) -> np.ndarray:
    by_model: dict[str, list[np.ndarray]] = defaultdict(list)
    for model_id, scores in rows:
        by_model[model_id].append(scores)
    if not by_model:
        raise ValueError("a primary seed has no kept available respondent answers")
    model_log_means = [
        logsumexp(np.stack(samples), axis=0) - math.log(len(samples))
        for samples in by_model.values()
    ]
    return logsumexp(np.stack(model_log_means), axis=0) - math.log(len(model_log_means))


@dataclass(frozen=True)
class _GridFillTask:
    law_values: tuple[tuple[float, ...], ...]
    outcome: int
    phi: float
    groups: tuple[tuple[int, float, tuple[tuple[int, float], ...]], ...]
    tail_form: SpreadTailForm


def _fill_seed_grid_task(
    task: _GridFillTask,
) -> tuple[tuple[tuple[int, tuple[int, ...], tuple[float, ...]], ...], ...]:
    """Evaluate one seed's missing cells without consulting fit-grid globals."""
    law_results = []
    for law_values in task.law_values:
        values = np.asarray(law_values, dtype=float)
        group_results = []
        for b_index, bias, spread_rows in task.groups:
            k_indices = tuple(k_index for k_index, _ in spread_rows)
            spreads = np.asarray([spread for _, spread in spread_rows], dtype=float)
            scores = _log_probability_values_grid(
                values * bias,
                task.outcome,
                spreads,
                phi=task.phi,
                tail_form=task.tail_form,
            )
            group_results.append(
                (b_index, k_indices, tuple(float(score) for score in scores))
            )
        law_results.append(tuple(group_results))
    return tuple(law_results)


@dataclass
class _SeedJointSurface:
    seed_id: str
    block_id: str
    outcome: int
    model_ids: tuple[str, ...]
    law_values: tuple[tuple[float, ...], ...]
    law_scores: np.ndarray
    pooled_scores: np.ndarray
    per_model_scores: dict[str, np.ndarray]


class _JointLikelihoodCache:
    """Per-observation-model lazy likelihood surfaces for primary seeds."""

    def __init__(
        self,
        seed_rows: Sequence[CalibrationSeed],
        laws: Mapping[str, Sequence[tuple[str, Quantiles]]],
        outcomes: Mapping[str, float],
        phi_by_seed: Mapping[str, float],
        k_grid: np.ndarray,
        b_grid: np.ndarray,
        *,
        tail_form: SpreadTailForm,
    ) -> None:
        self.k_grid = k_grid
        self.b_grid = b_grid
        self.tail_form = tail_form
        self.seed_rows: list[_SeedJointSurface] = []
        for row in seed_rows:
            outcome = outcomes[row.seed_id]
            law_rows = laws[row.seed_id]
            law_models = tuple(model_id for model_id, _ in law_rows)
            law_values = tuple(
                tuple(float(value) for value in epistemic_knots(rate, 1.0))
                for _, rate in law_rows
            )
            shape = (len(law_rows), len(k_grid), len(b_grid))
            model_ids = tuple(dict.fromkeys(law_models))
            self.seed_rows.append(
                _SeedJointSurface(
                    seed_id=row.seed_id,
                    block_id=row.block_id,
                    outcome=int(outcome),
                    model_ids=law_models,
                    law_values=law_values,
                    law_scores=np.full(shape, np.nan),
                    pooled_scores=np.full(shape[1:], np.nan),
                    per_model_scores={
                        model_id: np.full(shape[1:], np.nan) for model_id in model_ids
                    },
                )
            )
        self.phi_by_seed = phi_by_seed

    def _task(self, seed_index: int, requested: np.ndarray) -> _GridFillTask | None:
        seed = self.seed_rows[seed_index]
        missing = np.isnan(seed.law_scores[0]) & requested
        groups = []
        for b_index in range(len(self.b_grid)):
            k_indices = np.flatnonzero(missing[:, b_index])
            if len(k_indices) == 0:
                continue
            groups.append(
                (
                    b_index,
                    float(self.b_grid[b_index]),
                    tuple(
                        (int(k_index), float(self.k_grid[k_index]))
                        for k_index in k_indices
                    ),
                )
            )
        if not groups:
            return None
        return _GridFillTask(
            law_values=seed.law_values,
            outcome=seed.outcome,
            phi=self.phi_by_seed[seed.seed_id],
            groups=tuple(groups),
            tail_form=self.tail_form,
        )

    def _apply(
        self,
        seed_index: int,
        result: tuple[tuple[tuple[int, tuple[int, ...], tuple[float, ...]], ...], ...],
    ) -> None:
        seed = self.seed_rows[seed_index]
        touched_k: list[np.ndarray] = []
        touched_b: list[np.ndarray] = []
        for law_index, law_result in enumerate(result):
            for b_index, k_indices, scores in law_result:
                k_array = np.asarray(k_indices, dtype=int)
                seed.law_scores[law_index, k_array, b_index] = scores
                if law_index == 0:
                    touched_k.append(k_array)
                    touched_b.append(np.full(len(k_array), b_index, dtype=int))
        if not touched_k:
            return
        k_cells = np.concatenate(touched_k)
        b_cells = np.concatenate(touched_b)
        law_scores = seed.law_scores[:, k_cells, b_cells]
        complete = ~np.any(np.isnan(law_scores), axis=0)
        k_cells, b_cells = k_cells[complete], b_cells[complete]
        law_scores = law_scores[:, complete]
        if len(k_cells) == 0:
            return
        seed.pooled_scores[k_cells, b_cells] = _aggregate_law_scores(
            [
                (model_id, law_scores[law_index])
                for law_index, model_id in enumerate(seed.model_ids)
            ]
        )
        for model_id in seed.per_model_scores:
            sample_indices = [
                law_index
                for law_index, law_model in enumerate(seed.model_ids)
                if law_model == model_id
            ]
            seed.per_model_scores[model_id][k_cells, b_cells] = logsumexp(
                law_scores[sample_indices], axis=0
            ) - math.log(len(sample_indices))


def _ensure_joint_caches(
    caches: Sequence[_JointLikelihoodCache],
    requests: Mapping[_JointLikelihoodCache, Mapping[int, np.ndarray]],
    pool: ProcessPoolExecutor | None,
) -> None:
    task_refs: list[tuple[_JointLikelihoodCache, int]] = []
    tasks: list[_GridFillTask] = []
    for cache in caches:
        for seed_index in sorted(requests.get(cache, {})):
            task = cache._task(seed_index, requests[cache][seed_index])
            if task is not None:
                task_refs.append((cache, seed_index))
                tasks.append(task)
    if not tasks:
        return
    results = (
        list(pool.map(_fill_seed_grid_task, tasks))
        if pool is not None
        else [_fill_seed_grid_task(task) for task in tasks]
    )
    for (cache, seed_index), result in zip(task_refs, results, strict=True):
        cache._apply(seed_index, result)


@dataclass(frozen=True)
class _JointSurfaceFamily:
    cache: _JointLikelihoodCache
    groups: tuple[tuple[int, ...], ...]
    model_id: str | None = None


@dataclass(frozen=True)
class _JointFitVariant:
    family: _JointSurfaceFamily
    group_draws: tuple[int, ...]

    def seed_indices(self) -> tuple[int, ...]:
        included: list[int] = []
        seen: set[int] = set()
        for group_index in self.group_draws:
            for seed_index in self.family.groups[group_index]:
                if seed_index not in seen:
                    included.append(seed_index)
                    seen.add(seed_index)
        return tuple(included)


@dataclass(frozen=True)
class _JointEstimate:
    k_hat: float
    b_hat: float
    index: tuple[int, int]


def _variant_surface(
    variant: _JointFitVariant, k_indices: np.ndarray, b_indices: np.ndarray
) -> np.ndarray:
    family = variant.family
    surface = np.zeros((len(k_indices), len(b_indices)), dtype=float)
    for group_index in variant.group_draws:
        group_surface = np.zeros_like(surface)
        for seed_index in family.groups[group_index]:
            seed = family.cache.seed_rows[seed_index]
            seed_surface = (
                seed.pooled_scores
                if family.model_id is None
                else seed.per_model_scores[family.model_id]
            )
            group_surface += seed_surface[np.ix_(k_indices, b_indices)]
        surface += group_surface
    return surface


def _coarse_to_fine_joint_fits(
    variants: Sequence[_JointFitVariant],
    caches: Sequence[_JointLikelihoodCache],
    coarse_k_indices: np.ndarray,
    coarse_b_indices: np.ndarray,
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    *,
    pool: ProcessPoolExecutor | None,
) -> list[_JointEstimate]:
    """Fit many variants while batching each round's lazy cache requests."""
    one_k = int(np.argmin(np.abs(np.log(k_grid))))
    one_b = int(np.argmin(np.abs(np.log(b_grid))))
    centers: list[tuple[int, int] | None] = []
    estimates: list[_JointEstimate | None] = []
    for variant in variants:
        coarse_surface = _variant_surface(variant, coarse_k_indices, coarse_b_indices)
        if np.all(np.isneginf(coarse_surface)):
            centers.append(None)
            estimates.append(_JointEstimate(1.0, 1.0, (one_k, one_b)))
            continue
        coarse_k, coarse_b = np.unravel_index(
            int(np.argmax(coarse_surface)), coarse_surface.shape
        )
        centers.append(
            (int(coarse_k_indices[coarse_k]), int(coarse_b_indices[coarse_b]))
        )
        estimates.append(None)

    for _round in range(50):
        active = [index for index, estimate in enumerate(estimates) if estimate is None]
        if not active:
            break
        requests: dict[_JointLikelihoodCache, dict[int, np.ndarray]] = defaultdict(dict)
        windows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for variant_index in active:
            center = centers[variant_index]
            assert center is not None
            k_start = max(0, center[0] - K_COARSE_STRIDE)
            k_stop = min(len(k_grid) - 1, center[0] + K_COARSE_STRIDE)
            b_start = max(0, center[1] - B_COARSE_STRIDE)
            b_stop = min(len(b_grid) - 1, center[1] + B_COARSE_STRIDE)
            k_indices = np.arange(k_start, k_stop + 1, dtype=int)
            b_indices = np.arange(b_start, b_stop + 1, dtype=int)
            windows[variant_index] = (k_indices, b_indices)
            cache_requests = requests[variants[variant_index].family.cache]
            for seed_index in variants[variant_index].seed_indices():
                mask = cache_requests.get(seed_index)
                if mask is None:
                    mask = np.zeros((len(k_grid), len(b_grid)), dtype=bool)
                    cache_requests[seed_index] = mask
                mask[np.ix_(k_indices, b_indices)] = True
        _ensure_joint_caches(caches, requests, pool)

        for variant_index in active:
            k_indices, b_indices = windows[variant_index]
            surface = _variant_surface(variants[variant_index], k_indices, b_indices)
            local_k, local_b = np.unravel_index(int(np.argmax(surface)), surface.shape)
            best_index = (
                int(k_indices[local_k]),
                int(b_indices[local_b]),
            )
            k_edge = (local_k == 0 and best_index[0] > 0) or (
                local_k == len(k_indices) - 1 and best_index[0] < len(k_grid) - 1
            )
            b_edge = (local_b == 0 and best_index[1] > 0) or (
                local_b == len(b_indices) - 1 and best_index[1] < len(b_grid) - 1
            )
            if k_edge or b_edge:
                centers[variant_index] = best_index
                continue
            k_hat, b_hat = _refine_joint_surface(
                surface,
                k_indices,
                b_indices,
                k_grid,
                b_grid,
                best_index,
            )
            estimates[variant_index] = _JointEstimate(k_hat, b_hat, best_index)
    if any(estimate is None for estimate in estimates):
        raise RuntimeError("joint spread search did not converge in 50 rounds")
    return [estimate for estimate in estimates if estimate is not None]


def _bias_likelihood_matrix(
    cache: _JointLikelihoodCache,
    seed_indices: Sequence[int],
    b_grid: np.ndarray,
    spread: float,
    pool: ProcessPoolExecutor | None,
) -> np.ndarray:
    """Pooled log likelihood of each listed seed over the b axis at one spread."""
    groups = tuple(
        (b_index, float(bias), ((0, float(spread)),))
        for b_index, bias in enumerate(b_grid)
    )
    tasks = [
        _GridFillTask(
            law_values=cache.seed_rows[seed_index].law_values,
            outcome=cache.seed_rows[seed_index].outcome,
            phi=cache.phi_by_seed[cache.seed_rows[seed_index].seed_id],
            groups=groups,
            tail_form=cache.tail_form,
        )
        for seed_index in seed_indices
    ]
    results = (
        list(pool.map(_fill_seed_grid_task, tasks))
        if pool is not None
        else [_fill_seed_grid_task(task) for task in tasks]
    )
    rows = []
    for seed_index, result in zip(seed_indices, results, strict=True):
        law_rows = []
        for model_id, law_result in zip(
            cache.seed_rows[seed_index].model_ids, result, strict=True
        ):
            scores = np.empty(len(b_grid))
            for b_index, _, values in law_result:
                scores[b_index] = values[0]
            law_rows.append((model_id, scores))
        rows.append(_aggregate_law_scores(law_rows))
    return np.stack(rows)


def _fit_range(estimates: np.ndarray) -> SpreadRange:
    low, high = np.quantile(estimates, _BOOTSTRAP_PERCENTILES)
    return SpreadRange(low=float(low), high=float(high))


def _fit_common_bias_range(estimates: np.ndarray) -> CommonBiasRange:
    low, high = np.quantile(estimates, _BOOTSTRAP_PERCENTILES)
    return CommonBiasRange(low=float(low), high=float(high))


def _correlation(k_estimates: np.ndarray, b_estimates: np.ndarray) -> float | None:
    log_k = np.log(np.asarray(k_estimates, dtype=float))
    log_b = np.log(np.asarray(b_estimates, dtype=float))
    if np.all(log_k == log_k[0]) or np.all(log_b == log_b[0]):
        return None
    value = float(np.corrcoef(log_k, log_b)[0, 1])
    return float(np.clip(value, -1.0, 1.0))


def _joint_range(k_estimates: np.ndarray, b_estimates: np.ndarray) -> JointRange:
    return JointRange(
        k=_fit_range(k_estimates),
        b=_fit_common_bias_range(b_estimates),
        correlation=_correlation(k_estimates, b_estimates),
    )


def _group_seed_rows(labels: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    ordered = tuple(dict.fromkeys(labels))
    return tuple(
        tuple(index for index, label in enumerate(labels) if label == group)
        for group in ordered
    )


def _bootstrap_joint_range(
    estimates: Sequence[_JointEstimate],
) -> JointRange:
    return _joint_range(
        np.asarray([row.k_hat for row in estimates]),
        np.asarray([row.b_hat for row in estimates]),
    )


def _adoption_reasons(
    k_at_bound: bool,
    b_at_bound: bool,
    conditional_k_gain: float | None,
    block_count: int,
    zero_probability_seed_ids: Sequence[str] = (),
) -> list[str]:
    reasons = []
    if k_at_bound:
        reasons.append("joint optimum is at the k bound")
    if b_at_bound:
        reasons.append("joint optimum is at the b bound")
    if conditional_k_gain is None or conditional_k_gain <= 0:
        reasons.append(
            "leave-one-block-out gain of k_hat over k = 1, both at each fold's "
            "b_hat, is not positive"
        )
    if block_count < 2:
        reasons.append("fewer than two primary blocks are available")
    if zero_probability_seed_ids:
        reasons.append(
            "primary seeds have zero pooled likelihood at every coarse cell: "
            + ", ".join(zero_probability_seed_ids)
        )
    return reasons


def _fit_bias_variants(
    per_seed_scores: np.ndarray,
    groups: tuple[tuple[int, ...], ...],
    draws: Sequence[tuple[int, ...]],
    log_bias_grid: np.ndarray,
) -> np.ndarray:
    fitted = []
    for group_draws in draws:
        surface = np.zeros(per_seed_scores.shape[1], dtype=float)
        for group_index in group_draws:
            group_surface = np.zeros(per_seed_scores.shape[1], dtype=float)
            for seed_index in groups[group_index]:
                group_surface += per_seed_scores[seed_index]
            surface += group_surface
        fitted.append(_fit_scores(surface, log_bias_grid)[0])
    return np.asarray(fitted, dtype=float)


def _direct_seed_log_probability(
    cache: _JointLikelihoodCache,
    seed_index: int,
    spread: float,
    bias: float,
) -> float:
    seed = cache.seed_rows[seed_index]
    rows = []
    for model_id, values in zip(seed.model_ids, seed.law_values, strict=True):
        scores = _log_probability_values_grid(
            np.asarray(values, dtype=float) * bias,
            seed.outcome,
            np.asarray([spread], dtype=float),
            phi=cache.phi_by_seed[seed.seed_id],
            tail_form=cache.tail_form,
        )
        rows.append((model_id, scores))
    return float(_aggregate_law_scores(rows)[0])


def _log_score_sum(values: Sequence[float]) -> float:
    if any(np.isneginf(value) for value in values):
        return -math.inf
    return math.fsum(values)


def _leave_one_block_out_diagnostics(
    cache: _JointLikelihoodCache,
    block_groups: tuple[tuple[int, ...], ...],
    fold_estimates: Sequence[_JointEstimate],
) -> LeaveOneBlockOutDiagnostics:
    rows = []
    joint_total = 0.0
    conditional_total = 0.0
    joint_total_available = True
    conditional_total_available = True
    unique_blocks = tuple(cache.seed_rows[group[0]].block_id for group in block_groups)
    for block_id, held_indices, estimate in zip(
        unique_blocks, block_groups, fold_estimates, strict=True
    ):
        fit_scores = [
            _direct_seed_log_probability(
                cache, seed_index, estimate.k_hat, estimate.b_hat
            )
            for seed_index in held_indices
        ]
        k_one_bias_scores = [
            _direct_seed_log_probability(cache, seed_index, 1.0, estimate.b_hat)
            for seed_index in held_indices
        ]
        one_one_scores = [
            _direct_seed_log_probability(cache, seed_index, 1.0, 1.0)
            for seed_index in held_indices
        ]
        score_fit = _log_score_sum(fit_scores)
        score_k_one_bias = _log_score_sum(k_one_bias_scores)
        score_one_one = _log_score_sum(one_one_scores)
        joint_gain = (
            score_fit - score_one_one
            if math.isfinite(score_fit) and math.isfinite(score_one_one)
            else None
        )
        conditional_gain = (
            score_fit - score_k_one_bias
            if math.isfinite(score_fit) and math.isfinite(score_k_one_bias)
            else None
        )
        if joint_gain is None:
            joint_total_available = False
        else:
            joint_total += joint_gain
        if conditional_gain is None:
            conditional_total_available = False
        else:
            conditional_total += conditional_gain
        rows.append(
            LeaveOneBlockOutRow(
                block_id=block_id,
                k_hat=estimate.k_hat,
                b_hat=estimate.b_hat,
                log_score_at_k_hat_b_hat=(
                    score_fit if math.isfinite(score_fit) else None
                ),
                log_score_at_k_1_b_hat=(
                    score_k_one_bias if math.isfinite(score_k_one_bias) else None
                ),
                log_score_at_k_1_b_1=(
                    score_one_one if math.isfinite(score_one_one) else None
                ),
                joint_gain=joint_gain,
                conditional_k_gain=conditional_gain,
            )
        )
    return LeaveOneBlockOutDiagnostics(
        joint_total_gain=joint_total if joint_total_available else None,
        conditional_k_total_gain=(
            conditional_total if conditional_total_available else None
        ),
        blocks=tuple(rows),
    )


def _reference_percentiles(
    observed: float | None, values: Sequence[float]
) -> DiagnosticReferencePercentiles:
    low, high = np.quantile(np.asarray(values, dtype=float), _BOOTSTRAP_PERCENTILES)
    return DiagnosticReferencePercentiles(
        observed=observed, p05=float(low), p95=float(high)
    )


def _pit_reference(
    observed: PITAtSpread, simulated: Sequence[PITAtSpread]
) -> PITReferenceDiagnostics:
    return PITReferenceDiagnostics(
        median=_reference_percentiles(
            observed.median, [row.median for row in simulated]
        ),
        central_50_coverage=_reference_percentiles(
            observed.central_50_coverage,
            [row.central_50_coverage for row in simulated],
        ),
        central_90_coverage=_reference_percentiles(
            observed.central_90_coverage,
            [row.central_90_coverage for row in simulated],
        ),
    )


def _reference_spread_grid() -> np.ndarray:
    """Return a coarse log-spaced fit grid that contains the reference k=1."""
    return np.geomspace(*K_BOUNDS, _REFERENCE_GRID_POINTS)


@dataclass(frozen=True)
class _ReferenceSeedInput:
    seed_id: str
    block_id: str
    phi: float
    laws_by_model: tuple[tuple[str, tuple[tuple[float, ...], ...]], ...]


@dataclass(frozen=True)
class _ReferenceReplicateTask:
    child_seed: np.random.SeedSequence
    seed_inputs: tuple[_ReferenceSeedInput, ...]
    k_hat: float
    b_hat: float
    k_grid: tuple[float, ...]
    tail_form: SpreadTailForm


@dataclass(frozen=True)
class _ReferenceReplicateResult:
    pit_at_k_1_b_hat: PITAtSpread
    pit_at_k_refit_b_hat: PITAtSpread
    cooke_at_k_1_b_hat: float
    cooke_at_k_refit_b_hat: float
    conditional_k_gain: float


def _quantiles_from_values(values: tuple[float, ...]) -> Quantiles:
    return Quantiles(
        p5=values[0], p25=values[1], p50=values[2], p75=values[3], p95=values[4]
    )


def _simulate_reference_replicate(
    task: _ReferenceReplicateTask,
) -> _ReferenceReplicateResult:
    """Simulate and refit one reference replicate from its own child seed."""
    generator = np.random.default_rng(task.child_seed)
    k_grid = np.asarray(task.k_grid, dtype=float)
    log_k_grid = np.log(k_grid)
    simulated_rows: list[tuple[_ReferenceSeedInput, int, float]] = []
    score_rows = []
    pooled_laws: dict[str, tuple[tuple[str, Quantiles], ...]] = {}

    for seed_input in task.seed_inputs:
        model_ids = tuple(model_id for model_id, _ in seed_input.laws_by_model)
        model_index = int(generator.integers(0, len(model_ids)))
        _, model_laws = seed_input.laws_by_model[model_index]
        law_values = model_laws[int(generator.integers(0, len(model_laws)))]
        law = _quantiles_from_values(law_values)
        normal_score = float(generator.normal())
        rate = float(
            task.b_hat
            * stretched_score_quantiles(
                law,
                np.asarray([normal_score]),
                spread=task.k_hat,
                tail_form=task.tail_form,
            )[0]
        )
        if seed_input.phi == 0:
            outcome = int(generator.poisson(rate))
        else:
            outcome = int(
                generator.negative_binomial(
                    1 / seed_input.phi,
                    1 / (1 + seed_input.phi * rate),
                )
            )
        uniform = float(generator.random())
        simulated_rows.append((seed_input, outcome, uniform))

        sample_scores = []
        for model, samples in seed_input.laws_by_model:
            for sample in samples:
                law_values_array = np.asarray(sample, dtype=float)
                scores = _log_probability_values_grid(
                    law_values_array * task.b_hat,
                    outcome,
                    k_grid,
                    phi=seed_input.phi,
                    tail_form=task.tail_form,
                )
                sample_scores.append((model, scores))
        score_rows.append(_aggregate_law_scores(sample_scores))
        pooled_laws[seed_input.seed_id] = tuple(
            (model, _quantiles_from_values(sample))
            for model, samples in seed_input.laws_by_model
            for sample in samples
        )

    matrix = np.stack(score_rows)
    refit_k, _ = _fit_matrix(matrix, log_k_grid)
    block_ids = [row.block_id for row in task.seed_inputs]
    conditional_k_gain = _conditional_k_leave_one_block_out_gain(
        matrix, block_ids, log_k_grid
    )
    if conditional_k_gain is None:
        raise ValueError("simulated reference has an undefined conditional-k gain")

    pit_rows_at_one = []
    pit_rows_at_refit = []
    for seed_input, outcome, uniform in simulated_rows:
        laws = pooled_laws[seed_input.seed_id]
        pair_at_one, pair_at_refit = _pooled_count_cdf_pairs(
            laws, outcome, (1.0, refit_k), phi=seed_input.phi, scale=task.b_hat
        )
        pit_rows_at_one.append(
            (
                seed_input.seed_id,
                seed_input.block_id,
                _randomized_pit(pair_at_one, uniform),
            )
        )
        pit_rows_at_refit.append(
            (
                seed_input.seed_id,
                seed_input.block_id,
                _randomized_pit(pair_at_refit, uniform),
            )
        )
    summary_at_one = _pit_summary(pit_rows_at_one)
    summary_at_refit = _pit_summary(pit_rows_at_refit)
    return _ReferenceReplicateResult(
        pit_at_k_1_b_hat=summary_at_one,
        pit_at_k_refit_b_hat=summary_at_refit,
        cooke_at_k_1_b_hat=cooke_calibration_score(
            [row[2] for row in pit_rows_at_one]
        ).score,
        cooke_at_k_refit_b_hat=cooke_calibration_score(
            [row[2] for row in pit_rows_at_refit]
        ).score,
        conditional_k_gain=conditional_k_gain,
    )


def _simulated_reference_diagnostics(
    available_primary_seeds: Sequence[CalibrationSeed],
    positive_primary_seeds: Sequence[CalibrationSeed],
    laws: Mapping[str, Sequence[tuple[str, Quantiles]]],
    outcomes: Mapping[str, float],
    *,
    phi_by_seed: Mapping[str, float],
    k_hat: float,
    b_hat: float,
    primary_uniforms: Mapping[str, float],
    zero_probability_seed_ids: Sequence[str],
    replicates: int,
    seed: int,
    pool: ProcessPoolExecutor | None,
) -> SimulatedReferenceDiagnostics:
    """Compare observed diagnostics with refits at the joint level estimate."""
    reference_spreads = _reference_spread_grid()
    reference_log_grid = np.log(reference_spreads)
    positive_block_ids = [row.block_id for row in positive_primary_seeds]
    observed_matrix = []
    for row in positive_primary_seeds:
        outcome = outcomes[row.seed_id]
        sample_scores = [
            (
                model_id,
                _log_probability_values_grid(
                    epistemic_knots(rate, 1.0) * b_hat,
                    int(outcome),
                    reference_spreads,
                    phi=phi_by_seed[row.seed_id],
                    tail_form="calibrated_p95_peak",
                ),
            )
            for model_id, rate in laws[row.seed_id]
        ]
        observed_matrix.append(_aggregate_law_scores(sample_scores))
    observed_matrix_array = np.stack(observed_matrix)
    observed_refit_k, _ = _fit_matrix(observed_matrix_array, reference_log_grid)
    observed_gain = _conditional_k_leave_one_block_out_gain(
        observed_matrix_array, positive_block_ids, reference_log_grid
    )

    zero_probability = set(zero_probability_seed_ids)
    observed_pit_at_one = []
    observed_pit_at_refit = []
    for row in available_primary_seeds:
        outcome = outcomes[row.seed_id]
        if row.seed_id in zero_probability:
            pit_at_one = pit_at_refit = 1.0
        else:
            pair_at_one, pair_at_refit = _pooled_count_cdf_pairs(
                laws[row.seed_id],
                int(outcome),
                (1.0, observed_refit_k),
                phi=phi_by_seed[row.seed_id],
                scale=b_hat,
            )
            uniform = primary_uniforms[row.seed_id]
            pit_at_one = _randomized_pit(pair_at_one, uniform)
            pit_at_refit = _randomized_pit(pair_at_refit, uniform)
        observed_pit_at_one.append((row.seed_id, row.block_id, pit_at_one))
        observed_pit_at_refit.append((row.seed_id, row.block_id, pit_at_refit))
    observed_summary_at_one = _pit_summary(observed_pit_at_one)
    observed_summary_at_refit = _pit_summary(observed_pit_at_refit)
    observed_cooke_at_one = cooke_calibration_score(
        [row[2] for row in observed_pit_at_one]
    )
    observed_cooke_at_refit = cooke_calibration_score(
        [row[2] for row in observed_pit_at_refit]
    )

    seed_inputs = []
    for row in available_primary_seeds:
        by_model: dict[str, list[tuple[float, ...]]] = defaultdict(list)
        for model_id, rate in laws[row.seed_id]:
            by_model[model_id].append(
                tuple(float(value) for value in epistemic_knots(rate, 1.0))
            )
        seed_inputs.append(
            _ReferenceSeedInput(
                seed_id=row.seed_id,
                block_id=row.block_id,
                phi=phi_by_seed[row.seed_id],
                laws_by_model=tuple(
                    (model_id, tuple(by_model[model_id]))
                    for model_id in sorted(by_model)
                ),
            )
        )
    child_seeds = np.random.SeedSequence(seed).spawn(replicates)
    tasks = [
        _ReferenceReplicateTask(
            child_seed=child_seed,
            seed_inputs=tuple(seed_inputs),
            k_hat=k_hat,
            b_hat=b_hat,
            k_grid=tuple(float(value) for value in reference_spreads),
            tail_form="calibrated_p95_peak",
        )
        for child_seed in child_seeds
    ]
    simulated = (
        list(pool.map(_simulate_reference_replicate, tasks))
        if pool is not None
        else [_simulate_reference_replicate(task) for task in tasks]
    )
    return SimulatedReferenceDiagnostics(
        replicates=replicates,
        seed=seed,
        b_hat=b_hat,
        primary_pit_at_k_1_b_hat=_pit_reference(
            observed_summary_at_one, [row.pit_at_k_1_b_hat for row in simulated]
        ),
        primary_pit_at_k_refit_b_hat=_pit_reference(
            observed_summary_at_refit,
            [row.pit_at_k_refit_b_hat for row in simulated],
        ),
        cooke_score_at_k_1_b_hat=_reference_percentiles(
            observed_cooke_at_one.score,
            [row.cooke_at_k_1_b_hat for row in simulated],
        ),
        cooke_score_at_k_refit_b_hat=_reference_percentiles(
            observed_cooke_at_refit.score,
            [row.cooke_at_k_refit_b_hat for row in simulated],
        ),
        conditional_k_leave_one_block_out_gain=_reference_percentiles(
            observed_gain,
            [row.conditional_k_gain for row in simulated],
        ),
    )


def _validated_joint_grids() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    integer_values = (
        K_GRID_POINTS,
        B_GRID_POINTS,
        K_COARSE_STRIDE,
        B_COARSE_STRIDE,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in integer_values
    ):
        raise ValueError("joint grid dimensions and strides must be integers")
    if K_GRID_POINTS < 3 or B_GRID_POINTS < 3:
        raise ValueError("joint spread grids must each contain at least three points")
    if K_COARSE_STRIDE < 1 or B_COARSE_STRIDE < 1:
        raise ValueError("joint coarse-grid strides must be positive")
    if (K_GRID_POINTS - 1) % 3 != 0:
        raise ValueError("joint k grid must place k = 1 on the grid")
    if (K_GRID_POINTS - 1) % K_COARSE_STRIDE != 0:
        raise ValueError("joint k grid must include both coarse-grid bounds")
    if (B_GRID_POINTS - 1) % 2 != 0:
        raise ValueError("joint b grid must place b = 1 on the grid")
    if (B_GRID_POINTS - 1) % B_COARSE_STRIDE != 0:
        raise ValueError("joint b grid must include both coarse-grid bounds")

    k_grid = np.geomspace(*K_BOUNDS, K_GRID_POINTS)
    b_grid = np.geomspace(*B_BOUNDS, B_GRID_POINTS)
    coarse_k_indices = np.arange(0, K_GRID_POINTS, K_COARSE_STRIDE, dtype=int)
    coarse_b_indices = np.arange(0, B_GRID_POINTS, B_COARSE_STRIDE, dtype=int)
    if coarse_k_indices[-1] != K_GRID_POINTS - 1:
        raise ValueError("joint k coarse grid must include its upper bound")
    if coarse_b_indices[-1] != B_GRID_POINTS - 1:
        raise ValueError("joint b coarse grid must include its upper bound")
    return k_grid, b_grid, coarse_k_indices, coarse_b_indices


def _draw_rows(draws: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(index) for index in row) for row in draws)


def _fit_respondent_spread_impl(
    instrument: CalibrationInstrument,
    artifact: CalibrationArtifact,
    plan_payload: Mapping[str, object],
    resolutions: SeedResolutionSet | tuple[SeedResolution, ...],
    *,
    instrument_sha256: str | None,
    collection_plan_sha256: str | None,
    artifact_sha256: str | None,
    resolutions_sha256: str | None,
    treatments: CalibrationTreatments | None,
    bootstrap: int,
    seed: int,
    reference_replicates: int,
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    coarse_k_indices: np.ndarray,
    coarse_b_indices: np.ndarray,
    pool: ProcessPoolExecutor | None,
) -> RespondentSpreadFit:
    outcomes = resolved_outcomes(instrument.seeds, resolutions)
    dispersion_by_seed, pooled_phi, fallback_seed_ids = seed_dispersions(
        instrument.seeds
    )
    phi_by_seed = {
        seed_id: dispersion[0] for seed_id, dispersion in dispersion_by_seed.items()
    }
    instrument_digest = instrument.digest()
    plan_digest = canonical_json_sha256(dict(plan_payload))
    artifact_digest = canonical_json_sha256(artifact.model_dump(mode="json"))
    resolutions_digest = resolutions_sha256 or canonical_json_sha256(
        {
            "resolutions": [
                row.model_dump(mode="json")
                for row in (
                    resolutions.resolutions
                    if isinstance(resolutions, SeedResolutionSet)
                    else resolutions
                )
            ]
        }
    )
    for label, given, actual in (
        ("instrument", instrument_sha256, instrument_digest),
        ("collection plan", collection_plan_sha256, plan_digest),
        ("artifact", artifact_sha256, artifact_digest),
    ):
        if given is not None and given != actual:
            raise ValueError(f"{label} digest differs from the validated capture")
    if artifact.instrument_sha256 != instrument_digest:
        raise ValueError("calibration artifact names a different instrument")
    if artifact.collection_plan_sha256 != plan_digest:
        raise ValueError("calibration artifact names a different collection plan")

    plan_treatments = calibration_treatments(instrument, plan_payload)
    if treatments is not None and treatments != plan_treatments:
        raise ValueError("validated acquisition treatments differ from the plan")
    treatments = treatments or plan_treatments
    questions = {
        (seed_row.seed_id, respondent.key())
        for seed_row in instrument.seeds
        for respondent in instrument.respondents
    }
    cases = {(row.seed_id, row.respondent.key()): row for row in artifact.cases}
    if set(cases) != questions:
        raise ValueError(
            "calibration artifact cases differ from the full instrument roster"
        )

    laws: dict[str, list[tuple[str, Quantiles]]] = defaultdict(list)
    medians: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for case in artifact.cases:
        answer = case.answer
        if answer is None or answer.status == "unavailable":
            continue
        if answer.rate is None:
            raise ValueError("available calibration answer has no count law")
        laws[case.seed_id].append((case.respondent.model_id, answer.rate))
        medians[case.seed_id][case.respondent.model_id].append(answer.rate.p50)

    all_primary_seeds = [row for row in instrument.seeds if row.role == "primary"]
    unscored = tuple(row.seed_id for row in all_primary_seeds if not laws[row.seed_id])
    available_primary_seeds = [row for row in all_primary_seeds if laws[row.seed_id]]
    if not available_primary_seeds:
        raise ValueError("respondent spread needs a primary seed with a kept answer")

    reference_cache = _JointLikelihoodCache(
        available_primary_seeds,
        laws,
        outcomes,
        phi_by_seed,
        k_grid,
        b_grid,
        tail_form="calibrated_p95_peak",
    )
    poisson_phi = dict.fromkeys((row.seed_id for row in available_primary_seeds), 0.0)
    poisson_cache = _JointLikelihoodCache(
        available_primary_seeds,
        laws,
        outcomes,
        poisson_phi,
        k_grid,
        b_grid,
        tail_form="calibrated_p95_peak",
    )
    pooled_phi_by_seed = dict.fromkeys(
        (row.seed_id for row in available_primary_seeds), pooled_phi
    )
    pooled_phi_cache = _JointLikelihoodCache(
        available_primary_seeds,
        laws,
        outcomes,
        pooled_phi_by_seed,
        k_grid,
        b_grid,
        tail_form="calibrated_p95_peak",
    )
    caches = (reference_cache, poisson_cache, pooled_phi_cache)

    coarse_mask = np.zeros((len(k_grid), len(b_grid)), dtype=bool)
    coarse_mask[np.ix_(coarse_k_indices, coarse_b_indices)] = True
    coarse_requests = {
        cache: {
            seed_index: coarse_mask
            for seed_index in range(len(available_primary_seeds))
        }
        for cache in caches
    }
    _ensure_joint_caches(caches, coarse_requests, pool)

    positive_indices = []
    primary_seeds = []
    zero_probability: list[str] = []
    for seed_index, row in enumerate(available_primary_seeds):
        coarse_scores = reference_cache.seed_rows[seed_index].pooled_scores[
            np.ix_(coarse_k_indices, coarse_b_indices)
        ]
        if np.all(np.isneginf(coarse_scores)):
            zero_probability.append(row.seed_id)
        else:
            positive_indices.append(seed_index)
            primary_seeds.append(row)
    if not primary_seeds:
        raise ValueError("respondent spread needs a positive-probability primary seed")

    block_ids = [row.block_id for row in primary_seeds]
    block_groups_local = _group_seed_rows(block_ids)
    block_groups = tuple(
        tuple(positive_indices[index] for index in group)
        for group in block_groups_local
    )
    unique_blocks = tuple(block_ids[group[0]] for group in block_groups_local)
    generator = np.random.default_rng(seed)
    block_draws = generator.integers(
        0, len(unique_blocks), size=(bootstrap, len(unique_blocks))
    )
    block_draw_rows = _draw_rows(block_draws)

    variants: list[_JointFitVariant] = []
    spans: dict[str, tuple[int, int]] = {}

    def add_variants(
        name: str,
        family: _JointSurfaceFamily,
        group_draw_rows: Sequence[tuple[int, ...]],
    ) -> tuple[int, int]:
        start_index = len(variants)
        variants.extend(
            _JointFitVariant(family=family, group_draws=draws)
            for draws in group_draw_rows
        )
        span = (start_index, len(variants))
        spans[name] = span
        return span

    point_family = _JointSurfaceFamily(reference_cache, block_groups)
    point_draws = (tuple(range(len(block_groups))),)
    add_variants("point", point_family, point_draws)
    add_variants("whole_block", point_family, block_draw_rows)

    super_block_for_block = {
        block_id: super_block_id
        for super_block_id, block_members in instrument.super_blocks.items()
        for block_id in block_members
    }
    super_block_ids = [
        f"super-block:{super_block_for_block[block_id]}"
        if block_id in super_block_for_block
        else f"primary-block:{block_id}"
        for block_id in block_ids
    ]
    super_groups_local = _group_seed_rows(super_block_ids)
    super_groups = tuple(
        tuple(positive_indices[index] for index in group)
        for group in super_groups_local
    )
    super_rng = np.random.default_rng(seed)
    super_draws = super_rng.integers(
        0, len(super_groups), size=(bootstrap, len(super_groups))
    )
    super_family = _JointSurfaceFamily(reference_cache, super_groups)
    add_variants("super_block", super_family, _draw_rows(super_draws))

    curation_seed_ids = instrument.curation_sensitive_seed_ids
    curation_sensitive = set(curation_seed_ids)
    curation_cache_indices = [
        seed_index
        for seed_index in positive_indices
        if reference_cache.seed_rows[seed_index].seed_id not in curation_sensitive
    ]
    curation_groups_local: tuple[tuple[int, ...], ...] = ()
    if curation_cache_indices:
        curation_block_ids = [
            reference_cache.seed_rows[index].block_id
            for index in curation_cache_indices
        ]
        curation_groups_local = _group_seed_rows(curation_block_ids)
        curation_groups = tuple(
            tuple(curation_cache_indices[index] for index in group)
            for group in curation_groups_local
        )
        curation_rng = np.random.default_rng(seed)
        curation_draws = curation_rng.integers(
            0,
            len(curation_groups),
            size=(bootstrap, len(curation_groups)),
        )
        curation_family = _JointSurfaceFamily(reference_cache, curation_groups)
        add_variants(
            "curation_point",
            curation_family,
            (tuple(range(len(curation_groups))),),
        )
        add_variants("curation_bootstrap", curation_family, _draw_rows(curation_draws))

    lobo_variant_indices: list[int | None] = []
    if len(block_groups) > 1:
        for held_index in range(len(block_groups)):
            training_groups = tuple(
                group
                for group_index, group in enumerate(block_groups)
                if group_index != held_index
            )
            family = _JointSurfaceFamily(reference_cache, training_groups)
            start, _ = add_variants(
                f"lobo_{held_index}", family, (tuple(range(len(training_groups))),)
            )
            lobo_variant_indices.append(start)
    else:
        lobo_variant_indices.append(None)

    per_model_span: dict[str, int] = {}
    model_ids = sorted(
        {
            model_id
            for seed_index in positive_indices
            for model_id in reference_cache.seed_rows[seed_index].per_model_scores
        }
    )
    for model_id in model_ids:
        model_seed_indices = tuple(
            seed_index
            for seed_index in positive_indices
            if model_id in reference_cache.seed_rows[seed_index].per_model_scores
        )
        family = _JointSurfaceFamily(
            reference_cache,
            tuple((seed_index,) for seed_index in model_seed_indices),
            model_id=model_id,
        )
        start, _ = add_variants(
            f"per_model_{model_id}",
            family,
            (tuple(range(len(model_seed_indices))),),
        )
        per_model_span[model_id] = start

    alternative_caches = {
        "poisson": poisson_cache,
        "pooled_phi": pooled_phi_cache,
    }
    alternative_spans: dict[str, tuple[int, int]] = {}
    for name, cache in alternative_caches.items():
        family = _JointSurfaceFamily(cache, block_groups)
        alternative_spans[name] = add_variants(
            f"alternative_{name}", family, (*point_draws, *block_draw_rows)
        )

    joint_estimates = _coarse_to_fine_joint_fits(
        variants,
        caches,
        coarse_k_indices,
        coarse_b_indices,
        k_grid,
        b_grid,
        pool=pool,
    )

    def estimates_for(name: str) -> list[_JointEstimate]:
        start_index, stop_index = spans[name]
        return joint_estimates[start_index:stop_index]

    point_estimate = estimates_for("point")[0]
    block_estimates = estimates_for("whole_block")
    fitted_range = _bootstrap_joint_range(block_estimates)
    super_block_range = _bootstrap_joint_range(estimates_for("super_block"))
    if curation_cache_indices:
        curation_point = estimates_for("curation_point")[0]
        curation_range = _bootstrap_joint_range(estimates_for("curation_bootstrap"))
        curation_diagnostics = CurationSensitivityDiagnostics(
            seed_ids=curation_seed_ids,
            k_hat=curation_point.k_hat,
            b_hat=curation_point.b_hat,
            range=curation_range,
            blocks=len(curation_groups_local),
        )
    else:
        curation_diagnostics = CurationSensitivityDiagnostics(
            seed_ids=curation_seed_ids,
            unavailable_reason=(
                "excluding curation-sensitive seeds leaves no positive-probability "
                "primary seeds"
            ),
        )

    fold_estimates = []
    for variant_index in lobo_variant_indices:
        fold_estimates.append(
            _JointEstimate(1.0, 1.0, (0, 0))
            if variant_index is None
            else joint_estimates[variant_index]
        )
    lobo = _leave_one_block_out_diagnostics(
        reference_cache, block_groups, fold_estimates
    )
    reasons = _adoption_reasons(
        point_estimate.index[0] in (0, len(k_grid) - 1),
        point_estimate.index[1] in (0, len(b_grid) - 1),
        lobo.conditional_k_total_gain,
        len(unique_blocks),
        zero_probability,
    )

    alternative_fits: dict[str, tuple[_JointEstimate, JointRange]] = {}
    for name in alternative_caches:
        start_index, stop_index = alternative_spans[name]
        point = joint_estimates[start_index]
        alternative_range = _bootstrap_joint_range(
            joint_estimates[start_index + 1 : stop_index]
        )
        alternative_fits[name] = point, alternative_range

    one_b = int(np.argmin(np.abs(np.log(b_grid))))
    k_column_mask = np.zeros((len(k_grid), len(b_grid)), dtype=bool)
    k_column_mask[:, one_b] = True
    _ensure_joint_caches(
        (reference_cache,),
        {
            reference_cache: {
                seed_index: k_column_mask for seed_index in positive_indices
            }
        },
        pool,
    )
    all_k_indices = np.arange(len(k_grid), dtype=int)
    one_b_index = np.asarray([one_b], dtype=int)
    point_variant = variants[spans["point"][0]]
    marginal_k_hat, _ = _fit_scores(
        _variant_surface(point_variant, all_k_indices, one_b_index)[:, 0],
        np.log(k_grid),
    )
    marginal_k_bootstrap = np.asarray(
        [
            _fit_scores(
                _variant_surface(variant, all_k_indices, one_b_index)[:, 0],
                np.log(k_grid),
            )[0]
            for variant in variants[slice(*spans["whole_block"])]
        ],
        dtype=float,
    )
    marginal_k_range = _fit_range(marginal_k_bootstrap)

    marginal_bias_basis: Literal["accepted_k_hat", "face_value"] = (
        "accepted_k_hat" if not reasons else "face_value"
    )
    marginal_bias_reference = point_estimate.k_hat if not reasons else 1.0
    bias_matrix = _bias_likelihood_matrix(
        reference_cache,
        positive_indices,
        b_grid,
        marginal_bias_reference,
        pool,
    )
    marginal_b_hat, _ = _fit_scores(np.sum(bias_matrix, axis=0), np.log(b_grid))
    marginal_b_bootstrap = _fit_bias_variants(
        bias_matrix, block_groups_local, block_draw_rows, np.log(b_grid)
    )
    marginal_b_range = _fit_common_bias_range(marginal_b_bootstrap)

    model_fits = []
    for model_id, variant_index in per_model_span.items():
        model_estimate = joint_estimates[variant_index]
        model_seed_count = len(variants[variant_index].family.groups)
        model_fits.append(
            ModelJointFit(
                model_id=model_id,
                k_hat=model_estimate.k_hat,
                b_hat=model_estimate.b_hat,
                scored_seeds=model_seed_count,
            )
        )

    primary_uniforms: dict[str, float] = {}
    for row in available_primary_seeds:
        primary_uniforms[row.seed_id] = float(generator.random())
    zero_probability_set = set(zero_probability)
    pit_rows: dict[str, list[tuple[str, str, float]]] = {
        "at_k_1_b_1": [],
        "at_k_hat_b_1": [],
        "at_k_1_b_hat": [],
        "at_k_hat_b_hat": [],
    }
    seed_pits: dict[str, tuple[float | None, float | None]] = {}
    for row in available_primary_seeds:
        if row.seed_id in zero_probability_set:
            values = (1.0, 1.0, 1.0, 1.0)
        else:
            uniform = primary_uniforms[row.seed_id]
            # Order: (1, 1), (k_hat, 1), (1, b_hat), (k_hat, b_hat).
            pairs = [
                pair
                for bias in (1.0, point_estimate.b_hat)
                for pair in _pooled_count_cdf_pairs(
                    laws[row.seed_id],
                    int(outcomes[row.seed_id]),
                    (1.0, point_estimate.k_hat),
                    phi=phi_by_seed[row.seed_id],
                    scale=bias,
                )
            ]
            values = tuple(_randomized_pit(pair, uniform) for pair in pairs)
        for name, pit in zip(pit_rows, values, strict=True):
            pit_rows[name].append((row.seed_id, row.block_id, pit))
        seed_pits[row.seed_id] = (values[0], values[3])
    primary_pit = PrimaryPITDiagnostics(
        at_k_1_b_1=_pit_summary(pit_rows["at_k_1_b_1"]),
        at_k_hat_b_1=_pit_summary(pit_rows["at_k_hat_b_1"]),
        at_k_1_b_hat=_pit_summary(pit_rows["at_k_1_b_hat"]),
        at_k_hat_b_hat=_pit_summary(pit_rows["at_k_hat_b_hat"]),
    )
    cooke = CookeDiagnostics(
        at_k_1_b_1=cooke_calibration_score([row[2] for row in pit_rows["at_k_1_b_1"]]),
        at_k_hat_b_1=cooke_calibration_score(
            [row[2] for row in pit_rows["at_k_hat_b_1"]]
        ),
        at_k_1_b_hat=cooke_calibration_score(
            [row[2] for row in pit_rows["at_k_1_b_hat"]]
        ),
        at_k_hat_b_hat=cooke_calibration_score(
            [row[2] for row in pit_rows["at_k_hat_b_hat"]]
        ),
    )
    simulated_reference = (
        _simulated_reference_diagnostics(
            available_primary_seeds,
            primary_seeds,
            laws,
            outcomes,
            phi_by_seed=phi_by_seed,
            k_hat=point_estimate.k_hat,
            b_hat=point_estimate.b_hat,
            primary_uniforms=primary_uniforms,
            zero_probability_seed_ids=zero_probability,
            replicates=reference_replicates,
            seed=seed,
            pool=pool,
        )
        if reference_replicates
        else None
    )

    auxiliary_pits: list[AuxiliarySeedPIT] = []
    seed_rows: list[RespondentSeedFitRow] = []
    for row in instrument.seeds:
        value = outcomes[row.seed_id]
        pit_at_one: float | None = None
        pit_at_joint: float | None = None
        if row.role == "primary":
            pit_at_one, pit_at_joint = seed_pits.get(row.seed_id, (None, None))
        elif (
            row.role in ("control", "secondary")
            and laws[row.seed_id]
            and row.unit == "count"
        ):
            uniform = float(generator.random())
            count = int(value)
            phi = phi_by_seed[row.seed_id]
            (pair_one,) = _pooled_count_cdf_pairs(
                laws[row.seed_id], count, (1.0,), phi=phi, scale=1.0
            )
            (pair_joint,) = _pooled_count_cdf_pairs(
                laws[row.seed_id],
                count,
                (point_estimate.k_hat,),
                phi=phi,
                scale=point_estimate.b_hat,
            )
            pit_at_one = _randomized_pit(pair_one, uniform)
            pit_at_joint = _randomized_pit(pair_joint, uniform)
        elif row.role == "secondary" and laws[row.seed_id] and row.unit == "usd":
            uniform = float(generator.random())
            pair_one = _pooled_quantile_cdf_pair(
                laws[row.seed_id], value, 1.0, "calibrated_p95_peak"
            )
            pair_joint = _pooled_quantile_cdf_pair(
                laws[row.seed_id],
                value,
                point_estimate.k_hat,
                "calibrated_p95_peak",
            )
            assert pair_one is not None and pair_joint is not None
            pit_at_one = _randomized_quantile_pit(pair_one, uniform)
            pit_at_joint = _randomized_quantile_pit(pair_joint, uniform)
        if row.role in ("control", "secondary"):
            auxiliary_pits.append(
                AuxiliarySeedPIT(
                    seed_id=row.seed_id,
                    role=row.role,
                    pit_at_k_1_b_1=pit_at_one,
                    pit_at_k_hat_b_hat=pit_at_joint,
                )
            )
        phi_row = dispersion_by_seed.get(row.seed_id)
        seed_rows.append(
            RespondentSeedFitRow(
                seed_id=row.seed_id,
                block_id=row.block_id,
                role=row.role,
                value=value,
                phi=phi_row[0] if phi_row is not None else None,
                phi_periods=phi_row[1] if phi_row is not None else None,
                phi_source=(
                    "pooled"
                    if phi_row is not None and phi_row[2] is not None
                    else "history"
                    if phi_row is not None
                    else None
                ),
                phi_fallback_reason=phi_row[2] if phi_row is not None else None,
                pit_at_k_1_b_1=pit_at_one,
                pit_at_k_hat_b_hat=pit_at_joint,
            )
        )

    memorization = []
    for row in instrument.seeds:
        if row.role not in ("primary", "control"):
            continue
        outcome = outcomes[row.seed_id]
        for model_id in sorted({item.model_id for item in instrument.respondents}):
            samples = medians[row.seed_id].get(model_id, [])
            near = sum(abs(value - outcome) <= 0.01 * abs(outcome) for value in samples)
            memorization.append(
                MemorizationRow(
                    seed_id=row.seed_id,
                    role=row.role,
                    model_id=model_id,
                    kept_samples=len(samples),
                    near_exact_medians=near,
                )
            )

    diagnostics = RespondentSpreadDiagnostics(
        primary_pit=primary_pit,
        leave_one_block_out=lobo,
        cooke=cooke,
        super_block_bootstrap=SuperBlockBootstrapDiagnostics(
            range=super_block_range,
            super_blocks=len(super_groups),
        ),
        without_curation_sensitive_seeds=curation_diagnostics,
        per_model=tuple(model_fits),
        controls_and_secondary_pit=tuple(auxiliary_pits),
        memorization=tuple(memorization),
        unscored_primary_seeds=unscored,
        zero_probability_primary_seeds=tuple(zero_probability),
        reference_replicates=reference_replicates,
        simulated_reference=simulated_reference,
    )

    capture = CalibrationCaptureIdentity(
        instrument_sha256=instrument_digest,
        collection_plan_sha256=plan_digest,
        artifact_sha256=artifact_digest,
        resolutions_sha256=resolutions_digest,
    )
    poisson_point, poisson_range = alternative_fits["poisson"]
    pooled_phi_point, pooled_phi_range = alternative_fits["pooled_phi"]
    return RespondentSpreadFit(
        status="rejected" if reasons else "accepted",
        rejection_reason="; ".join(reasons) if reasons else None,
        capture=capture,
        treatments=treatments,
        tail_form="calibrated_p95_peak",
        k_bounds=K_BOUNDS,
        b_bounds=B_BOUNDS,
        grid=JointGrid(
            k_points=K_GRID_POINTS,
            b_points=B_GRID_POINTS,
            k_coarse_stride=K_COARSE_STRIDE,
            b_coarse_stride=B_COARSE_STRIDE,
        ),
        observation=RespondentSpreadObservation(
            model_name="negative_binomial_2",
            pooled_phi=pooled_phi,
            minimum_periods=6,
            fallback_seed_ids=fallback_seed_ids,
        ),
        k_hat=point_estimate.k_hat,
        range=fitted_range.k,
        b_hat=point_estimate.b_hat,
        b_range=fitted_range.b,
        k_b_correlation=fitted_range.correlation,
        marginal=MarginalFits(
            k_at_b_1=MarginalSpreadFit(
                k_hat=marginal_k_hat,
                range=marginal_k_range,
            ),
            b_at_reference_k=MarginalBiasFit(
                reference_spread=marginal_bias_reference,
                reference_spread_basis=marginal_bias_basis,
                b_hat=marginal_b_hat,
                range=marginal_b_range,
            ),
        ),
        bootstrap=BootstrapSummary(
            replicates=bootstrap, seed=seed, blocks=len(unique_blocks)
        ),
        alternatives=SpreadObservationAlternatives(
            poisson=PoissonSpreadAlternative(
                k_hat=poisson_point.k_hat,
                range=poisson_range.k,
                b_hat=poisson_point.b_hat,
                b_range=poisson_range.b,
            ),
            pooled_phi=PooledPhiSpreadAlternative(
                phi=pooled_phi,
                k_hat=pooled_phi_point.k_hat,
                range=pooled_phi_range.k,
                b_hat=pooled_phi_point.b_hat,
                b_range=pooled_phi_range.b,
            ),
        ),
        diagnostics=diagnostics,
        seeds=tuple(seed_rows),
    )


def fit_respondent_spread(
    instrument: CalibrationInstrument,
    artifact: CalibrationArtifact,
    plan_payload: Mapping[str, object],
    resolutions: SeedResolutionSet | tuple[SeedResolution, ...],
    *,
    instrument_sha256: str | None = None,
    collection_plan_sha256: str | None = None,
    artifact_sha256: str | None = None,
    resolutions_sha256: str | None = None,
    treatments: CalibrationTreatments | None = None,
    bootstrap: int = 2000,
    seed: int = 42,
    reference_replicates: int = 200,
    workers: int = 1,
) -> RespondentSpreadFit:
    """Fit joint spread and level factors and produce their diagnostics."""
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 1:
        raise ValueError("bootstrap replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("random seed must be a nonnegative integer")
    if (
        isinstance(reference_replicates, bool)
        or not isinstance(reference_replicates, int)
        or reference_replicates < 0
    ):
        raise ValueError("reference replicates must be a nonnegative integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    k_grid, b_grid, coarse_k_indices, coarse_b_indices = _validated_joint_grids()

    def run(pool: ProcessPoolExecutor | None) -> RespondentSpreadFit:
        return _fit_respondent_spread_impl(
            instrument,
            artifact,
            plan_payload,
            resolutions,
            instrument_sha256=instrument_sha256,
            collection_plan_sha256=collection_plan_sha256,
            artifact_sha256=artifact_sha256,
            resolutions_sha256=resolutions_sha256,
            treatments=treatments,
            bootstrap=bootstrap,
            seed=seed,
            reference_replicates=reference_replicates,
            k_grid=k_grid,
            b_grid=b_grid,
            coarse_k_indices=coarse_k_indices,
            coarse_b_indices=coarse_b_indices,
            pool=pool,
        )

    if workers == 1:
        return run(None)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        return run(pool)


__all__ = [
    "B_BOUNDS",
    "B_COARSE_STRIDE",
    "B_GRID_POINTS",
    "K_BOUNDS",
    "K_COARSE_STRIDE",
    "K_GRID_POINTS",
    "calibration_treatments",
    "cooke_calibration_score",
    "fit_respondent_spread",
    "seed_dispersion",
    "seed_dispersions",
]
