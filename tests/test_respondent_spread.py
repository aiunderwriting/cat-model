"""Numerics and diagnostics for respondent spread calibration."""

from __future__ import annotations

import calendar
import math
from datetime import date
from typing import Literal, cast

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp, ndtri
from scipy.stats import chi2, nbinom
from scipy.stats import poisson as poisson_distribution

import catmodel.calibration.respondent_spread as respondent_spread
from catmodel.calibration.respondent_spread import (
    _adoption_reasons,
    _aggregate_law_scores,
    _cdf_sf_at_rate,
    _log_probability_values_grid,
    _pit_summary,
    cooke_calibration_score,
    fit_respondent_spread,
    seed_dispersion,
    seed_dispersions,
)
from catmodel.contracts.contextual_risk import CONTEXTUAL_SCHEMA_VERSION
from catmodel.contracts.elicitation import Quantiles, Reasoning, Respondent
from catmodel.contracts.incident import SpreadTailForm
from catmodel.contracts.respondent_calibration import (
    CalibrationAnswer,
    CalibrationArtifact,
    CalibrationCase,
    CalibrationInstrument,
    CalibrationModelTreatment,
    CalibrationSeed,
    CalibrationSeedEvidence,
    CalibrationSeedHistory,
    CalibrationSeedSource,
    CalibrationTreatments,
    CommonBiasRange,
    CurationSensitivityDiagnostics,
    JointGrid,
    JointRange,
    ResolutionProtocol,
    RespondentSpreadFit,
    SeedResolution,
    SeedResolutionSet,
    SpreadRange,
)
from catmodel.elicitation.epistemic import (
    epistemic_knots,
    epistemic_mean,
    stretched_score_quantiles,
)
from catmodel.hashing import canonical_json_sha256
from test_support.count_law_oracle import CountLawOracle

_PROBABILITIES = np.asarray([0.05, 0.25, 0.5, 0.75, 0.95])
_KNOT_SCORES = ndtri(_PROBABILITIES)
_FORMS: tuple[SpreadTailForm, ...] = ("pure_stretch", "calibrated_p95_peak")


def _law(p5, p25, p50, p75, p95) -> Quantiles:
    return Quantiles(p5=p5, p25=p25, p50=p50, p75=p75, p95=p95)


def _log_symmetric(ratio: float, median: float = 1.0) -> Quantiles:
    sigma = math.log(ratio) / (2 * float(ndtri(0.95)))
    values = median * np.exp(sigma * _KNOT_SCORES)
    return _law(*values)


def _log_observation(
    outcome: int,
    rate: float,
    phi: float,
) -> float:
    if rate == 0:
        return 0.0 if outcome == 0 else -math.inf
    if phi > 0:
        shape = 1 / phi
        return (
            gammaln(outcome + shape)
            - gammaln(shape)
            - gammaln(outcome + 1)
            + shape * math.log(shape / (shape + rate))
            + outcome * math.log(rate / (shape + rate))
        )
    return outcome * math.log(rate) - rate - gammaln(outcome + 1)


def _quad_log_probability(
    estimate: Quantiles,
    outcome: int,
    spread: float,
    phi: float,
    tail_form: SpreadTailForm = "calibrated_p95_peak",
) -> float:
    def log_quantile(score: float) -> float:
        value = float(
            stretched_score_quantiles(
                estimate,
                np.asarray([score / spread]),
                spread=spread,
                epistemic_tail_scale=1.0,
                tail_form=tail_form,
            )[0]
        )
        return -math.inf if value == 0 else math.log(value)

    def log_integrand(score: float) -> float:
        log_rate = log_quantile(score)
        rate = 0.0 if log_rate == -math.inf else math.exp(log_rate)
        z = score / spread
        return (
            _log_observation(outcome, rate, phi)
            - 0.5 * z * z
            - 0.5 * math.log(2 * math.pi)
            - math.log(spread)
        )

    limit = 40 * spread
    relative_variance = 1 / (outcome + 1) + phi
    log_targets = math.log(outcome + 1) + np.arange(-40, 41) * math.sqrt(
        relative_variance
    )
    points = [
        -limit,
        limit,
        *[float(score) for score in _KNOT_SCORES],
        *[spread * sign * 2**power for sign in (-1, 1) for power in range(6)],
        *[
            spread * sign * value
            for sign in (-1, 1)
            for value in (36, 38, 39, 39.5, 39.75)
        ],
    ]
    for target in log_targets:
        low_value = log_quantile(-limit) - target
        high_value = log_quantile(limit) - target
        if low_value == 0:
            points.append(-limit)
        elif high_value == 0:
            points.append(limit)
        elif low_value < 0 < high_value:
            points.append(
                brentq(
                    lambda score, target=target: log_quantile(score) - target,
                    -limit,
                    limit,
                )
            )
    points = sorted({max(-limit, min(limit, float(point))) for point in points})

    sample_points = np.linspace(-limit, limit, 8001)
    sample_rates = stretched_score_quantiles(
        estimate,
        sample_points / spread,
        spread=spread,
        epistemic_tail_scale=1.0,
        tail_form=tail_form,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        sample_log_rates = np.log(sample_rates)
    sample_likelihood = np.full(sample_rates.shape, -math.inf)
    positive = sample_rates > 0
    if phi > 0:
        shape = 1 / phi
        rates = sample_rates[positive]
        with np.errstate(divide="ignore", invalid="ignore"):
            sample_likelihood[positive] = (
                gammaln(outcome + shape)
                - gammaln(shape)
                - gammaln(outcome + 1)
                + shape * (math.log(shape) - np.log(shape + rates))
                + outcome * (sample_log_rates[positive] - np.log(shape + rates))
            )
        if outcome == 0:
            sample_likelihood[~positive] = 0
    else:
        sample_likelihood[positive] = (
            outcome * sample_log_rates[positive]
            - sample_rates[positive]
            - gammaln(outcome + 1)
        )
        if outcome == 0:
            sample_likelihood[~positive] = 0
    z = sample_points / spread
    sampled_logs = (
        sample_likelihood - 0.5 * z * z - 0.5 * math.log(2 * math.pi) - math.log(spread)
    )
    maximum = float(np.max(sampled_logs))
    integral, _ = quad(
        lambda score: math.exp(log_integrand(score) - maximum),
        -limit,
        limit,
        points=points[1:-1],
        epsabs=1e-12,
        epsrel=2e-11,
        limit=5000,
    )
    return maximum + math.log(integral)


def _whole_line_narrow_law_reference(outcome: int, spread: float) -> float:
    """Independent whole-line quad for Q(Phi(s)) = exp(0.1 s)."""
    slope = 0.1 * spread

    def derivative(score):
        return outcome * slope - slope * math.exp(slope * score) - score

    lower, upper = -1.0, 1.0
    while derivative(lower) < 0:
        lower *= 2
    while derivative(upper) > 0:
        upper *= 2
    peak = brentq(derivative, lower, upper)

    def log_integrand(score):
        log_rate = slope * score
        if log_rate > math.log(float(np.finfo(float).max)):
            return -math.inf
        rate = math.exp(log_rate)
        return (
            outcome * log_rate
            - rate
            - gammaln(outcome + 1)
            - 0.5 * score * score
            - 0.5 * math.log(2 * math.pi)
        )

    maximum = log_integrand(peak)

    def shifted(score):
        return math.exp(log_integrand(score) - maximum)

    left, _ = quad(shifted, -math.inf, peak, epsabs=1e-12, epsrel=2e-11, limit=2000)
    right, _ = quad(shifted, peak, math.inf, epsabs=1e-12, epsrel=2e-11, limit=2000)
    return maximum + math.log(left + right)


def _seed(
    seed_id: str,
    block_id: str,
    role: Literal["primary", "control", "secondary"] = "primary",
    unit: Literal["count", "usd"] = "count",
) -> CalibrationSeed:
    if role == "control":
        window_start = date(2026, 3, 1)
        window_end = date(2026, 3, 31)
        information_date = date(2026, 2, 28)
    else:
        window_start = date(2026, 7, 1)
        window_end = date(2026, 7, 31)
        information_date = date(2026, 6, 30)
    latest_month = information_date.year * 12 + information_date.month - 1
    history = []
    for month_index in range(latest_month - 11, latest_month + 1):
        year, month_offset = divmod(month_index, 12)
        month = month_offset + 1
        history.append(
            CalibrationSeedHistory(
                period=f"{year:04d}-{month:02d}",
                period_end=date(year, month, calendar.monthrange(year, month)[1]),
                value=100,
            )
        )
    return CalibrationSeed(
        seed_id=seed_id,
        role=role,
        block_id=block_id,
        quantity=f"quantity {seed_id}",
        unit=unit,
        window_start=window_start,
        window_end=window_end,
        information_date=information_date,
        source=CalibrationSeedSource(
            name="Example source", url="https://example.test/source"
        ),
        evidence=CalibrationSeedEvidence(
            text="Dated synthetic evidence.", history=tuple(history)
        ),
        resolution_script="scripts/resolve.py",
        can_resolve_to_zero=True,
    )


def _seed_for_history(
    seed_id: str,
    block_id: str,
    *,
    window_start: date,
    window_end: date,
    information_date: date,
    history: tuple[CalibrationSeedHistory, ...],
    role: Literal["primary", "control", "secondary"] = "primary",
) -> CalibrationSeed:
    return CalibrationSeed(
        seed_id=seed_id,
        role=role,
        block_id=block_id,
        quantity=f"quantity {seed_id}",
        unit="count",
        window_start=window_start,
        window_end=window_end,
        information_date=information_date,
        source=CalibrationSeedSource(
            name="Example source", url="https://example.test/source"
        ),
        evidence=CalibrationSeedEvidence(
            text="Dated synthetic evidence.", history=history
        ),
        resolution_script="scripts/resolve.py",
        can_resolve_to_zero=True,
    )


def _nb2_sample(random: np.random.Generator, mean: float, phi: float) -> int:
    if phi == 0:
        return int(random.poisson(mean))
    return int(random.negative_binomial(1 / phi, 1 / (1 + phi * mean)))


def _monthly_history(
    random: np.random.Generator,
    *,
    periods: int,
    phi: float,
    mean: float,
    trend_per_month: float = 0.0003,
    start_year: int = 1976,
    start_month: int = 1,
) -> tuple[CalibrationSeedHistory, ...]:
    start_index = start_year * 12 + start_month - 1
    rows = []
    for offset in range(periods):
        year, month_offset = divmod(start_index + offset, 12)
        month = month_offset + 1
        period_mean = mean * math.exp(trend_per_month * offset)
        rows.append(
            CalibrationSeedHistory(
                period=f"{year:04d}-{month:02d}",
                period_end=date(year, month, calendar.monthrange(year, month)[1]),
                value=_nb2_sample(random, period_mean, phi),
            )
        )
    return tuple(rows)


def _quarterly_history(
    random: np.random.Generator,
    *,
    periods: int,
    phi: float,
    mean: float,
    trend_per_quarter: float = 0.001,
    start_year: int = 1951,
    start_quarter: int = 1,
) -> tuple[CalibrationSeedHistory, ...]:
    start_index = start_year * 4 + start_quarter - 1
    rows = []
    for offset in range(periods):
        year, quarter_offset = divmod(start_index + offset, 4)
        quarter = quarter_offset + 1
        month = 3 * quarter
        period_mean = mean * math.exp(trend_per_quarter * offset)
        rows.append(
            CalibrationSeedHistory(
                period=f"{year:04d}-Q{quarter}",
                period_end=date(year, month, calendar.monthrange(year, month)[1]),
                value=_nb2_sample(random, period_mean, phi),
            )
        )
    return tuple(rows)


def _resolution_protocol() -> ResolutionProtocol:
    return ResolutionProtocol(
        scripts={"scripts/resolve.py": "a" * 64},
        snapshot_rule="Use the dated source snapshot after the window closes.",
        replacement_rule="Replace only when the source republishes its frozen snapshot.",
    )


def _reasoning(seed_id: str) -> Reasoning:
    return Reasoning(
        median_reason="The supplied record supports this central estimate.",
        low_case="The source can omit some records.",
        high_case="Later qualifying records can appear.",
        method="direct_judgment",
        evidence_used=(seed_id,),
    )


def _plan(instrument: CalibrationInstrument) -> dict[str, object]:
    models = sorted({row.model_id for row in instrument.respondents})
    counts = {
        model: sum(row.model_id == model for row in instrument.respondents)
        for model in models
    }
    specs = [
        {
            "provider": "openai",
            "model_id": model,
            "reasoning_effort": "low",
            "max_output_tokens": 8192,
        }
        for model in models
    ]
    assert len(set(counts.values())) == 1
    return {
        "instrument_sha256": instrument.digest(),
        "parse_attempts": 3,
        "execution": {
            "models": specs,
            "settings": {
                "samples_per_question": next(iter(counts.values())),
                "max_output_tokens_by_model": dict.fromkeys(models, 8192),
                "request_timeout_seconds": 600.0,
                "provider_message_layout": "system_then_user",
            },
        },
    }


def _simulate_capture(
    k_true: float,
    *,
    rng_seed: int,
    blocks: int = 15,
    samples: int = 2,
    super_blocks=None,
    curation_sensitive_seed_ids: tuple[str, ...] = (),
):
    model_ids = ("model-a", "model-b")
    respondents = tuple(
        Respondent(
            respondent_id=f"{model}:{sample}", model_id=model, sample_index=sample
        )
        for model in model_ids
        for sample in range(samples)
    )
    count = blocks * 3
    seed_rows = tuple(
        _seed(f"primary-{index:02}", f"block-{index // 3:02}") for index in range(count)
    )
    instrument = CalibrationInstrument(
        seeds=seed_rows,
        respondents=respondents,
        latest_training_cutoff=date(2026, 5, 31),
        resolution_protocol=_resolution_protocol(),
        risk_contract_version=CONTEXTUAL_SCHEMA_VERSION,
        super_blocks=super_blocks or {},
        curation_sensitive_seed_ids=curation_sensitive_seed_ids,
    )
    random = np.random.default_rng(rng_seed)
    block_scores = random.normal(size=blocks)
    independent_scores = random.normal(size=count)
    scores = (
        math.sqrt(0.5) * block_scores[np.arange(count) // 3]
        + math.sqrt(0.5) * independent_scores
    )
    cases = []
    resolutions = []
    for index, (seed_row, score) in enumerate(zip(seed_rows, scores, strict=True)):
        ratio = float(np.exp(random.uniform(math.log(10), math.log(400))))
        median = float(np.exp(random.uniform(math.log(5), math.log(20_000))))
        law = _log_symmetric(ratio, median)
        true_rate = float(
            stretched_score_quantiles(
                law,
                np.asarray([score]),
                spread=k_true,
                tail_form="calibrated_p95_peak",
            )[0]
        )
        outcome = int(random.poisson(true_rate))
        resolutions.append(
            SeedResolution(
                seed_id=seed_row.seed_id,
                value=outcome,
                retrieved_at=date(2026, 11, 1),
                snapshot_url="https://example.test/resolved",
                snapshot_sha256=f"{index + 1:064x}",
                script="scripts/resolve.py",
                notes="",
            )
        )
        for respondent in respondents:
            cases.append(
                CalibrationCase(
                    seed_id=seed_row.seed_id,
                    respondent=respondent,
                    answer=CalibrationAnswer(
                        status="available",
                        rate=law,
                        reasoning=_reasoning(seed_row.seed_id),
                    ),
                )
            )
    plan = _plan(instrument)
    artifact = CalibrationArtifact(
        instrument_sha256=instrument.digest(),
        collection_plan_sha256=canonical_json_sha256(plan),
        cases=tuple(cases),
    )
    return instrument, artifact, plan, SeedResolutionSet(resolutions=tuple(resolutions))


def _simulate_nb2_reference_capture(
    rng_seed: int = 20261103,
    *,
    k_true: float = 1.0,
    b_true: float = 1.0,
    blocks: int = 15,
):
    respondent = Respondent(
        respondent_id="model-a:0", model_id="model-a", sample_index=0
    )
    seed_rows = [
        _seed(f"nb2-{index:02}", f"nb2-block-{index // 3:02}")
        for index in range(blocks * 3)
    ]
    random = np.random.default_rng(rng_seed)
    cases = []
    resolutions = []
    phis = random.uniform(0.01, 0.3, size=len(seed_rows))
    medians = np.exp(
        random.uniform(math.log(50), math.log(20_000), size=len(seed_rows))
    )
    for index, (seed_row, median, phi) in enumerate(
        zip(seed_rows, medians, phis, strict=True)
    ):
        law = _log_symmetric(float(random.uniform(2, 8)), float(median))
        history = _monthly_history(
            random,
            periods=360,
            phi=float(phi),
            mean=max(float(median) * 10, 500),
            start_year=1996,
        )
        seed_rows[index] = seed_row.model_copy(
            update={
                "evidence": CalibrationSeedEvidence(
                    text="Synthetic history from the same NB2 process.",
                    history=history,
                )
            }
        )
        normal_score = float(random.normal())
        expected_rate = float(
            stretched_score_quantiles(
                law,
                np.asarray([normal_score]),
                spread=k_true,
                tail_form="calibrated_p95_peak",
            )[0]
        )
        outcome = _nb2_sample(random, b_true * expected_rate, float(phi))
        resolutions.append(
            SeedResolution(
                seed_id=seed_row.seed_id,
                value=outcome,
                retrieved_at=date(2026, 11, 1),
                snapshot_url="https://example.test/nb2",
                snapshot_sha256=f"{index + 1:064x}",
                script="scripts/resolve.py",
                notes="",
            )
        )
        cases.append(
            CalibrationCase(
                seed_id=seed_row.seed_id,
                respondent=respondent,
                answer=CalibrationAnswer(
                    status="available",
                    rate=law,
                    reasoning=_reasoning(seed_row.seed_id),
                ),
            )
        )
    instrument = CalibrationInstrument(
        seeds=tuple(seed_rows),
        respondents=(respondent,),
        latest_training_cutoff=date(2026, 5, 31),
        resolution_protocol=_resolution_protocol(),
        risk_contract_version=CONTEXTUAL_SCHEMA_VERSION,
    )
    plan = _plan(instrument)
    artifact = CalibrationArtifact(
        instrument_sha256=instrument.digest(),
        collection_plan_sha256=canonical_json_sha256(plan),
        cases=tuple(cases),
    )
    return (
        instrument,
        artifact,
        plan,
        SeedResolutionSet(resolutions=tuple(resolutions)),
    )


@pytest.mark.parametrize("spread", [0.5, 0.75, 1.0])
@pytest.mark.parametrize("tail_scale", [0.5, 0.75, 1.0, 1.5])
def test_all_tail_forms_coincide_at_and_below_one(spread, tail_scale):
    law = _law(0.2, 2, 10, 80, 1000)
    values = [
        stretched_score_quantiles(
            law,
            np.asarray([-8.0, -1.6448536269514729, 0.0, 1.6448536269514729, 8.0]),
            spread=spread,
            epistemic_tail_scale=tail_scale,
            tail_form=form,
        )
        for form in _FORMS
    ]
    assert np.allclose(values[0], values[1], rtol=0, atol=0)
    means = [
        epistemic_mean(
            law,
            spread=spread,
            epistemic_tail_scale=tail_scale,
            tail_form=form,
        )
        for form in _FORMS
    ]
    assert means[0] == pytest.approx(means[1], rel=0, abs=0)


@pytest.mark.parametrize("tail_scale", [0.5, 0.75, 1.0, 1.5])
@pytest.mark.parametrize("tail_form", _FORMS)
def test_spread_mean_multiplier_is_one_at_reference(tail_scale, tail_form):
    law = _law(0.2, 2, 10, 80, 1000)
    assert epistemic_mean(
        law,
        spread=1.0,
        epistemic_tail_scale=tail_scale,
        tail_form=tail_form,
    ) == pytest.approx(epistemic_mean(law, epistemic_tail_scale=tail_scale), rel=2e-14)


def test_stretched_quantiles_reject_unknown_tail_form_even_for_constant_laws():
    with pytest.raises(ValueError, match="unknown spread tail form"):
        stretched_score_quantiles(
            _law(2, 2, 2, 2, 2),
            np.asarray([0.0]),
            spread=1.5,
            tail_form=cast(SpreadTailForm, "unknown"),
        )


@pytest.mark.parametrize("spread", [0.5, 1.0, 1.5, 2.0, 4.0])
@pytest.mark.parametrize("tail_scale", [0.5, 0.75, 1.0, 1.5])
@pytest.mark.parametrize("tail_form", _FORMS)
def test_stretched_mean_matches_normal_score_quadrature(spread, tail_scale, tail_form):
    law = _log_symmetric(400, median=12)
    exact = epistemic_mean(
        law,
        spread=spread,
        epistemic_tail_scale=tail_scale,
        tail_form=tail_form,
    )
    oracle = CountLawOracle(
        law, tail_scale=tail_scale, spread=spread, tail_form=tail_form
    )
    assert exact == pytest.approx(oracle.mean(), rel=1e-9)


def test_log_symmetric_count_multiplier_table():
    measured = {}
    for ratio in (20, 160, 400):
        law = _log_symmetric(ratio)
        base = epistemic_mean(law)
        measured[ratio] = {
            form: {
                spread: epistemic_mean(law, spread=spread, tail_form=form) / base
                for spread in (1.5, 2.0)
            }
            for form in ("pure_stretch", "calibrated_p95_peak")
        }
    assert measured[20]["pure_stretch"][1.5] == pytest.approx(1.7, abs=0.06)
    assert measured[20]["pure_stretch"][2.0] == pytest.approx(3.4, abs=0.08)
    assert measured[20]["calibrated_p95_peak"][1.5] == pytest.approx(1.7, abs=0.06)
    assert measured[20]["calibrated_p95_peak"][2.0] == pytest.approx(2.9, abs=0.08)
    assert 4.2 <= measured[160]["pure_stretch"][1.5] <= 5.4
    assert 33 <= measured[160]["pure_stretch"][2.0] <= 57
    assert measured[160]["calibrated_p95_peak"][1.5] == pytest.approx(2.2, abs=0.08)
    assert measured[160]["calibrated_p95_peak"][2.0] == pytest.approx(3.3, abs=0.08)
    assert 4.2 <= measured[400]["pure_stretch"][1.5] <= 5.4
    assert 33 <= measured[400]["pure_stretch"][2.0] <= 57
    assert measured[400]["calibrated_p95_peak"][1.5] == pytest.approx(2.2, abs=0.08)
    assert measured[400]["calibrated_p95_peak"][2.0] == pytest.approx(3.4, abs=0.08)


@pytest.mark.parametrize(
    ("estimate", "outcome"),
    [
        (_law(1, 2, 4, 8, 16), 0),
        (_law(1, 2, 4, 8, 16), 1),
        (_law(1, 2, 4, 8, 16), 3),
        (_law(0.5, 2, 10, 10, 10), 40),
        (_law(3, 60, 1000, 10000, 100000), 17000),
        (_law(1, 1, 1, 1, 1), 230),
        (_law(0, 0.1, 1, 3, 8), 17000),
        (_law(0, 0, 0, 1, 8), 0),
    ],
    ids=[
        "zero-count",
        "narrow-one",
        "narrow-three",
        "flat-upper",
        "wide-17000",
        "far-poisson",
        "zero-p5",
        "zero-p5-zero",
    ],
)
@pytest.mark.parametrize("phi", [0.0, 0.04, 0.5])
@pytest.mark.parametrize("spread", [0.5, 1.0, 1.5, 4.0])
def test_count_likelihood_integral_matches_adaptive_quad(
    estimate, outcome, phi, spread
):
    actual = float(
        _log_probability_values_grid(
            epistemic_knots(estimate, 1.0),
            outcome,
            np.asarray([spread]),
            phi=phi,
        )[0]
    )
    expected = _quad_log_probability(estimate, outcome, spread, phi)
    assert math.isfinite(actual)
    assert math.expm1(abs(actual - expected)) <= 1e-8


@pytest.mark.parametrize("scale", [1.0, 2.3])
@pytest.mark.parametrize("phi", [0.0, 0.17])
def test_scaling_the_knots_scales_the_count_law(scale, phi):
    estimate = _law(0, 1.5, 8, 70, 900)
    spread = np.asarray([1.6])
    scaled_estimate = _law(
        estimate.p5 * scale,
        estimate.p25 * scale,
        estimate.p50 * scale,
        estimate.p75 * scale,
        estimate.p95 * scale,
    )
    actual = respondent_spread._log_probability_values_grid(
        epistemic_knots(estimate, 1.0) * scale, 42, spread, phi=phi
    )
    expected = _log_probability_values_grid(
        epistemic_knots(scaled_estimate, 1.0), 42, spread, phi=phi
    )
    assert np.array_equal(actual, expected)


def test_scaled_zero_constant_law_keeps_its_zero_atom():
    zeros = np.zeros(5)
    spread = np.asarray([1.4])
    for scale in (0.1, 1.0, 10.0):
        assert respondent_spread._log_probability_values_grid(
            zeros * scale, 0, spread, phi=0.2
        ) == pytest.approx(np.zeros(1))
        assert np.isneginf(
            respondent_spread._log_probability_values_grid(
                zeros * scale, 1, spread, phi=0.2
            )[0]
        )


@pytest.mark.parametrize("spread", [0.5, 1.0, 4.0])
def test_quadrature_reaches_a_likelihood_peak_far_out_on_the_score_scale(spread):
    estimate = _law(*np.exp(0.1 * _KNOT_SCORES))
    actual = float(
        _log_probability_values_grid(
            epistemic_knots(estimate, 1.0), 1000, np.asarray([spread])
        )[0]
    )
    expected = _whole_line_narrow_law_reference(1000, spread)
    assert actual == pytest.approx(expected, abs=2e-7)
    if spread == 1:
        assert expected == pytest.approx(-2116.4, abs=0.5)


def test_log_likelihood_stays_finite_for_probabilities_below_e_minus_500():
    constant = epistemic_knots(_law(1, 1, 1, 1, 1), 1.0)
    poisson = float(_log_probability_values_grid(constant, 230, np.asarray([1.0]))[0])
    negative_binomial = float(
        _log_probability_values_grid(constant, 800, np.asarray([1.0]), phi=0.5)[0]
    )
    assert -1100 < poisson < -900
    assert -1_000 < negative_binomial < -500


@pytest.mark.parametrize("phi", [0.0, 0.04, 0.5])
def test_zero_rate_atoms_have_the_correct_count_likelihood(phi):
    all_zero = epistemic_knots(_law(0, 0, 0, 0, 0), 1.0)
    spreads = np.asarray([0.5, 1.0, 4.0])
    assert _log_probability_values_grid(all_zero, 0, spreads, phi=phi) == pytest.approx(
        np.zeros(3)
    )
    assert np.all(
        np.isneginf(_log_probability_values_grid(all_zero, 1, spreads, phi=phi))
    )


@pytest.mark.parametrize("phi", [-0.1, math.inf, math.nan])
@pytest.mark.parametrize(
    "estimate", [_law(2, 2, 2, 2, 2), _law(1, 2, 4, 8, 16)], ids=["constant", "wide"]
)
def test_count_likelihood_refuses_an_invalid_dispersion(estimate, phi):
    with pytest.raises(ValueError, match="NB2 phi must be nonnegative and finite"):
        _log_probability_values_grid(
            epistemic_knots(estimate, 1.0), 3, np.asarray([1.0]), phi=phi
        )


@pytest.mark.parametrize("phi", [0.0, 0.04, 0.5])
def test_nb2_cdf_and_survival_match_count_distributions(phi):
    log_rates = np.log(np.asarray([1.0, 10.0, 100.0]))
    actual_cdf, actual_sf = _cdf_sf_at_rate(log_rates, 13, phi=phi)
    rates = np.exp(log_rates)
    if phi == 0:
        assert actual_cdf == pytest.approx(poisson_distribution.cdf(13, rates))
        assert actual_sf == pytest.approx(poisson_distribution.sf(13, rates))
    else:
        probability = 1 / (1 + phi * rates)
        assert actual_cdf == pytest.approx(nbinom.cdf(13, 1 / phi, probability))
        assert actual_sf == pytest.approx(nbinom.sf(13, 1 / phi, probability))


@pytest.mark.parametrize(
    ("granularity", "window_months", "monthly_phi", "expected_periods"),
    [
        ("quarterly", 3, 0.12, 300),
        ("monthly", 1, 0.12, 600),
        ("monthly", 2, 0.24, 300),
        ("monthly", 3, 0.36, 201),
    ],
)
def test_seed_dispersion_recovers_nb2_phi_at_window_granularity(
    granularity, window_months, monthly_phi, expected_periods
):
    random = np.random.default_rng(20261104 + window_months)
    if window_months == 1:
        window_start, window_end, information_date = (
            date(2026, 1, 1),
            date(2026, 1, 31),
            date(2025, 12, 31),
        )
    elif window_months == 2:
        window_start, window_end, information_date = (
            date(2026, 2, 1),
            date(2026, 3, 31),
            date(2026, 1, 31),
        )
    else:
        window_start, window_end, information_date = (
            date(2026, 5, 1),
            date(2026, 7, 31),
            date(2026, 4, 30),
        )

    if granularity == "quarterly":
        history = _quarterly_history(
            random,
            periods=300,
            phi=monthly_phi,
            mean=4_000,
        )
        expected_phi = monthly_phi
    else:
        end_index = information_date.year * 12 + information_date.month - 1
        periods = end_index - (1976 * 12) + 1
        history = _monthly_history(
            random,
            periods=periods,
            phi=monthly_phi,
            mean=4_000,
            start_year=1976,
        )
        expected_phi = monthly_phi / window_months

    seed = _seed_for_history(
        "phi-recovery",
        "phi-block",
        window_start=window_start,
        window_end=window_end,
        information_date=information_date,
        history=history,
    )
    estimated_phi, periods, reason = seed_dispersion(seed)

    assert reason is None
    assert periods == expected_periods
    assert estimated_phi == pytest.approx(expected_phi, abs=0.025)


def test_poisson_history_estimates_zero_extra_dispersion():
    random = np.random.default_rng(6103)
    seed = _seed_for_history(
        "poisson-history",
        "poisson-block",
        window_start=date(2026, 1, 1),
        window_end=date(2026, 1, 31),
        information_date=date(2025, 12, 31),
        history=_monthly_history(
            random,
            periods=600,
            phi=0,
            mean=0.5,
            trend_per_month=0.0001,
        ),
    )
    estimated_phi, periods, reason = seed_dispersion(seed)
    assert reason is None
    assert periods == 600
    assert estimated_phi == pytest.approx(0, abs=0.005)


def test_seed_dispersion_fallback_reasons_and_pooled_median():
    random = np.random.default_rng(882)
    monthly = _monthly_history(
        random,
        periods=12,
        phi=0.1,
        mean=500,
        start_year=2025,
        start_month=7,
    )
    five_months = monthly[:5]
    five_quarters = _quarterly_history(
        random,
        periods=5,
        phi=0.1,
        mean=500,
        start_year=2024,
    )
    fewer = _seed_for_history(
        "fewer",
        "block-fewer",
        window_start=date(2026, 7, 1),
        window_end=date(2026, 7, 31),
        information_date=date(2026, 6, 30),
        history=five_months,
    )
    quarterly_month = _seed_for_history(
        "quarterly-month",
        "block-quarterly",
        window_start=date(2026, 7, 1),
        window_end=date(2026, 7, 31),
        information_date=date(2026, 6, 30),
        history=five_quarters,
    )
    unparseable = _seed_for_history(
        "unparseable",
        "block-unparseable",
        window_start=date(2026, 7, 1),
        window_end=date(2026, 7, 31),
        information_date=date(2026, 6, 30),
        history=(
            CalibrationSeedHistory(
                period="2025 March", period_end=date(2025, 3, 31), value=500
            ),
        ),
    )
    gap = _seed_for_history(
        "gap",
        "block-gap",
        window_start=date(2026, 7, 1),
        window_end=date(2026, 7, 31),
        information_date=date(2026, 6, 30),
        history=tuple(row for index, row in enumerate(monthly) if index != 4),
    )
    non_aligned = _seed("non-aligned", "block-non-aligned").model_copy(
        update={"window_start": date(2026, 7, 2)}
    )

    assert seed_dispersion(fewer) == (
        None,
        5,
        "fewer than 6 periods at the window's granularity",
    )
    assert seed_dispersion(non_aligned) == (
        None,
        None,
        "window is not month-aligned",
    )
    assert seed_dispersion(quarterly_month) == (
        None,
        None,
        "quarterly history cannot be aligned to a 1-month window",
    )
    unparseable_reason = seed_dispersion(unparseable)[2]
    assert unparseable_reason is not None
    assert "not valid YYYY-MM or YYYY-Qn" in unparseable_reason
    assert seed_dispersion(gap) == (None, None, "monthly history is not contiguous")
    quarterly = _quarterly_history(
        random, periods=8, phi=0.1, mean=500, start_year=2024
    )
    quarter_gap = _seed_for_history(
        "quarter-gap",
        "block-quarter-gap",
        window_start=date(2026, 7, 1),
        window_end=date(2026, 9, 30),
        information_date=date(2026, 6, 30),
        history=tuple(row for index, row in enumerate(quarterly) if index != 3),
    )
    assert seed_dispersion(quarter_gap) == (
        None,
        None,
        "quarterly history is not contiguous",
    )

    valid_a = _seed_for_history(
        "valid-a",
        "block-a",
        window_start=date(2026, 7, 1),
        window_end=date(2026, 7, 31),
        information_date=date(2026, 6, 30),
        history=_monthly_history(
            np.random.default_rng(710),
            periods=360,
            phi=0.05,
            mean=2_000,
            start_year=1996,
        ),
    )
    valid_b = _seed_for_history(
        "valid-b",
        "block-b",
        window_start=date(2026, 7, 1),
        window_end=date(2026, 7, 31),
        information_date=date(2026, 6, 30),
        history=_monthly_history(
            np.random.default_rng(711),
            periods=360,
            phi=0.25,
            mean=2_000,
            start_year=1996,
        ),
    )
    fallback_primary = _seed_for_history(
        "fallback-primary",
        "block-fallback-primary",
        window_start=date(2026, 7, 1),
        window_end=date(2026, 7, 31),
        information_date=date(2026, 6, 30),
        history=five_months,
    )
    fallback_control = _seed(
        "fallback-control", "block-control", role="control"
    ).model_copy(
        update={
            "evidence": CalibrationSeedEvidence(
                text="Short monthly history.", history=five_months
            )
        }
    )
    mapping, pooled, fallback_ids = seed_dispersions(
        (valid_a, valid_b, fallback_primary, fallback_control)
    )
    raw_a = seed_dispersion(valid_a)[0]
    raw_b = seed_dispersion(valid_b)[0]
    assert pooled == pytest.approx(float(np.median([raw_a, raw_b])))
    assert fallback_ids == ("fallback-primary", "fallback-control")
    assert mapping["fallback-primary"] == (
        pooled,
        5,
        "fewer than 6 periods at the window's granularity",
    )
    assert mapping["fallback-control"][0] == pooled


def test_seed_dispersions_require_at_least_one_usable_primary_history():
    no_history = _seed("no-history", "block").model_copy(
        update={
            "evidence": CalibrationSeedEvidence(text="No period history.", history=())
        }
    )
    with pytest.raises(ValueError, match="no primary seed has enough history"):
        seed_dispersions((no_history,))


def test_pit_histogram_coverage_and_median_on_known_values():
    summary = _pit_summary(
        tuple(
            (f"seed-{index}", f"block-{index}", pit)
            for index, pit in enumerate([0.02, 0.20, 0.40, 0.60, 0.80, 0.98])
        )
    )
    assert summary.median == pytest.approx(0.5)
    assert summary.central_50_coverage == pytest.approx(2 / 6)
    assert summary.central_90_coverage == pytest.approx(4 / 6)
    assert summary.histogram_10 == (1, 0, 1, 0, 1, 0, 1, 0, 1, 1)
    assert [row.pit for row in summary.seeds] == [0.02, 0.2, 0.4, 0.6, 0.8, 0.98]


def test_cooke_score_matches_the_six_bin_formula_and_five_degrees_of_freedom():
    pits = [0.01, 0.12, 0.30, 0.45, 0.60, 0.80, 0.98]
    result = cooke_calibration_score(pits)
    expected_bins = np.asarray([0.05, 0.20, 0.25, 0.25, 0.20, 0.05])
    observed = np.asarray([1, 1, 2, 1, 1, 1]) / len(pits)
    nonempty = observed > 0
    divergence = float(
        np.sum(
            observed[nonempty] * np.log(observed[nonempty] / expected_bins[nonempty])
        )
    )
    assert result.observed_proportions == pytest.approx(observed)
    assert result.expected_proportions == pytest.approx(expected_bins)
    assert result.divergence == pytest.approx(divergence)
    assert result.statistic == pytest.approx(2 * len(pits) * divergence)
    assert result.degrees_of_freedom == 5
    assert result.score == pytest.approx(chi2.sf(2 * len(pits) * divergence, df=5))


def test_model_equal_then_sample_equal_weighting():
    sample_rows = [
        ("model-a", np.log(np.asarray([0.2, 0.8]))),
        ("model-a", np.log(np.asarray([0.4, 0.6]))),
        ("model-b", np.log(np.asarray([0.01, 0.99]))),
    ]
    actual = _aggregate_law_scores(sample_rows)
    model_a = np.logaddexp(sample_rows[0][1], sample_rows[1][1]) - math.log(2)
    model_b = sample_rows[2][1]
    expected = np.logaddexp(model_a, model_b) - math.log(2)
    pooled_sample_mean = np.logaddexp.reduce(
        [row[1] for row in sample_rows], axis=0
    ) - math.log(len(sample_rows))
    assert actual == pytest.approx(expected)
    assert not np.allclose(actual, pooled_sample_mean)


def test_adoption_gates_cover_bounds_and_conditional_gain_in_order():
    assert _adoption_reasons(False, False, 0.25, 2) == []
    reasons = _adoption_reasons(True, True, 0.0, 1, ("seed-z",))
    assert reasons == [
        "joint optimum is at the k bound",
        "joint optimum is at the b bound",
        "leave-one-block-out gain of k_hat over k = 1, both at each fold's b_hat, is not positive",
        "fewer than two primary blocks are available",
        "primary seeds have zero pooled likelihood at every coarse cell: seed-z",
    ]
    # A positive joint gain cannot rescue a nonpositive conditional k gain.
    assert _adoption_reasons(False, False, -0.01, 3) == [
        "leave-one-block-out gain of k_hat over k = 1, both at each fold's b_hat, is not positive"
    ]
    assert _adoption_reasons(False, False, None, 3)


def test_joint_grid_defaults_and_validation(monkeypatch):
    assert respondent_spread.K_BOUNDS == (0.5, 4.0)
    assert respondent_spread.B_BOUNDS == (0.1, 10.0)
    assert respondent_spread.K_GRID_POINTS == 64
    assert respondent_spread.B_GRID_POINTS == 101
    assert respondent_spread.K_COARSE_STRIDE == 3
    assert respondent_spread.B_COARSE_STRIDE == 5
    k_grid = np.geomspace(*respondent_spread.K_BOUNDS, 64)
    b_grid = np.geomspace(*respondent_spread.B_BOUNDS, 101)
    assert k_grid[21] == 1.0
    assert b_grid[50] == 1.0

    monkeypatch.setattr(respondent_spread, "K_GRID_POINTS", 19)
    monkeypatch.setattr(respondent_spread, "B_GRID_POINTS", 21)
    k, b, coarse_k, coarse_b = respondent_spread._validated_joint_grids()
    assert len(k) == 19 and len(coarse_k) == 7
    assert len(b) == 21 and len(coarse_b) == 5
    assert coarse_k[-1] == len(k) - 1
    assert coarse_b[-1] == len(b) - 1

    monkeypatch.setattr(respondent_spread, "K_GRID_POINTS", 20)
    with pytest.raises(ValueError, match="place k = 1"):
        respondent_spread._validated_joint_grids()
    monkeypatch.setattr(respondent_spread, "K_GRID_POINTS", 19)
    monkeypatch.setattr(respondent_spread, "B_GRID_POINTS", 20)
    with pytest.raises(ValueError, match="place b = 1"):
        respondent_spread._validated_joint_grids()


def test_joint_refinement_recovers_a_correlated_off_grid_quadratic():
    k_grid = np.geomspace(0.5, 4.0, 64)
    b_grid = np.geomspace(0.1, 10.0, 101)
    k_index, b_index = 31, 47
    k_offset, b_offset = 0.23, -0.31
    u = np.arange(len(k_grid), dtype=float)[:, None] - k_index
    v = np.arange(len(b_grid), dtype=float)[None, :] - b_index
    surface = (
        8.0
        - 2.0 * (u - k_offset) ** 2
        + 0.7 * (u - k_offset) * (v - b_offset)
        - 3.0 * (v - b_offset) ** 2
    )
    estimate = respondent_spread._refine_joint_surface(
        surface,
        np.arange(len(k_grid)),
        np.arange(len(b_grid)),
        k_grid,
        b_grid,
        (k_index, b_index),
    )
    expected_k = math.exp(
        math.log(k_grid[k_index]) + k_offset * np.diff(np.log(k_grid))[0]
    )
    expected_b = math.exp(
        math.log(b_grid[b_index]) + b_offset * np.diff(np.log(b_grid))[0]
    )
    assert estimate[0] == pytest.approx(expected_k, rel=1e-9, abs=1e-12)
    assert estimate[1] == pytest.approx(expected_b, rel=1e-9, abs=1e-12)


def test_joint_refinement_uses_edge_and_nonconcave_axis_fallbacks():
    k_grid = np.geomspace(0.5, 4.0, 64)
    b_grid = np.geomspace(0.1, 10.0, 101)
    b_index = 48
    u = np.arange(len(k_grid), dtype=float)[:, None]
    v = np.arange(len(b_grid), dtype=float)[None, :] - (b_index + 0.22)
    edge_surface = -4.0 * u**2 - 2.0 * v**2
    edge = respondent_spread._refine_joint_surface(
        edge_surface,
        np.arange(len(k_grid)),
        np.arange(len(b_grid)),
        k_grid,
        b_grid,
        (0, b_index),
    )
    expected_b = math.exp(math.log(b_grid[b_index]) + 0.22 * np.diff(np.log(b_grid))[0])
    assert edge[0] == k_grid[0]
    assert edge[1] == pytest.approx(expected_b, rel=1e-12)

    k_index, b_index = 30, 50
    stencil = np.asarray(
        [
            [-0.26114185, -0.48962513, -1.53869271],
            [-2.18565755, 0.0, -0.42593936],
            [-0.63036599, -0.11361674, -1.06108412],
        ]
    )
    u = np.arange(len(k_grid), dtype=float)[:, None] - k_index
    v = np.arange(len(b_grid), dtype=float)[None, :] - b_index
    nonconcave = -100.0 - u**2 - v**2
    nonconcave[k_index - 1 : k_index + 2, b_index - 1 : b_index + 2] = stencil
    expected_k = respondent_spread._parabola_offset(stencil[:, 1], 1)
    expected_b_offset = respondent_spread._parabola_offset(stencil[1, :], 1)
    refined = respondent_spread._refine_joint_surface(
        nonconcave,
        np.arange(len(k_grid)),
        np.arange(len(b_grid)),
        k_grid,
        b_grid,
        (k_index, b_index),
    )
    assert refined[0] == pytest.approx(
        math.exp(math.log(k_grid[k_index]) + expected_k * np.diff(np.log(k_grid))[0])
    )
    assert refined[1] == pytest.approx(
        math.exp(
            math.log(b_grid[b_index]) + expected_b_offset * np.diff(np.log(b_grid))[0]
        )
    )


def test_lazy_joint_search_matches_full_grid_and_recenters(monkeypatch):
    k_grid = np.geomspace(0.5, 4.0, 19)
    b_grid = np.geomspace(0.1, 10.0, 21)
    coarse_k = np.arange(0, len(k_grid), 3)
    coarse_b = np.arange(0, len(b_grid), 5)
    k_index = np.arange(len(k_grid), dtype=float)[:, None]
    b_index = np.arange(len(b_grid), dtype=float)[None, :]
    surface = 10.0 * np.exp(
        -0.5 * (((k_index - 6.0) / 2.3) ** 2 + ((b_index - 10.0) / 2.3) ** 2)
    ) + 65.0 * np.exp(
        -0.5 * (((k_index - 10.2) / 0.8) ** 2 + ((b_index - 12.5) / 0.55) ** 2)
    )

    class SeedSurface:
        def __init__(self):
            self.block_id = "block"
            self.pooled_scores = np.full(surface.shape, np.nan)
            self.per_model_scores = {}

    class SyntheticCache:
        def __init__(self):
            self.seed_rows = [SeedSurface()]

    cache = cast(respondent_spread._JointLikelihoodCache, SyntheticCache())
    cache.seed_rows[0].pooled_scores[np.ix_(coarse_k, coarse_b)] = surface[
        np.ix_(coarse_k, coarse_b)
    ]
    family = respondent_spread._JointSurfaceFamily(cache, ((0,),))
    variant = respondent_spread._JointFitVariant(family, (0,))
    request_shapes = []

    def ensure(caches, requests, _pool):
        for cache_key, seed_masks in requests.items():
            for seed_index, mask in seed_masks.items():
                row = cache_key.seed_rows[seed_index]
                row.pooled_scores[mask] = surface[mask]
            mask = next(iter(seed_masks.values()))
            request_shapes.append(
                (
                    tuple(np.flatnonzero(mask.any(axis=1))),
                    tuple(np.flatnonzero(mask.any(axis=0))),
                )
            )

    monkeypatch.setattr(respondent_spread, "_ensure_joint_caches", ensure)
    result = respondent_spread._coarse_to_fine_joint_fits(
        (variant,),
        (cache,),
        coarse_k,
        coarse_b,
        k_grid,
        b_grid,
        pool=None,
    )[0]
    full_index = np.unravel_index(int(np.argmax(surface)), surface.shape)
    full_refined = respondent_spread._refine_joint_surface(
        surface,
        np.arange(len(k_grid)),
        np.arange(len(b_grid)),
        k_grid,
        b_grid,
        (int(full_index[0]), int(full_index[1])),
    )
    assert result.k_hat == pytest.approx(full_refined[0], rel=1e-12)
    assert result.b_hat == pytest.approx(full_refined[1], rel=1e-12)
    assert len(request_shapes) >= 2
    assert 9 in request_shapes[0][0]
    assert 12 in request_shapes[1][0]


def test_log_probability_values_are_bit_identical_for_spread_subsets():
    values = epistemic_knots(_log_symmetric(6.0, 120), 1.0)
    spreads = np.geomspace(0.5, 4.0, 19)
    complete = respondent_spread._log_probability_values_grid(
        values, 143, spreads, phi=0.07
    )
    selection = np.asarray([0, 3, 7, 12, 18], dtype=np.intp)
    subset = respondent_spread._log_probability_values_grid(
        values, 143, spreads[selection], phi=0.07
    )
    assert np.array_equal(complete[selection], subset)


@pytest.mark.parametrize(
    "knots",
    [
        (1.0, 3.0, 5.0, 9.0, 30.0),
        (0.0, 0.0, 2.0, 2.0, 700.0),
        (4.0, 4.0, 4.0, 5.0, 5.0),
    ],
)
def test_scalar_peak_polynomials_match_the_pchip_law_and_its_derivative(knots):
    pchip = PchipInterpolator(_PROBABILITIES, knots)
    value = respondent_spread._scalar_piecewise_polynomial(pchip)
    slope = respondent_spread._scalar_piecewise_polynomial(pchip.derivative())
    probabilities = np.concatenate((np.linspace(0.04, 0.96, 461), _PROBABILITIES))
    for probability in probabilities:
        expected_value = float(pchip(probability))
        expected_slope = float(pchip.derivative()(probability))
        assert respondent_spread._evaluate_scalar_polynomial(
            value, float(probability)
        ) == pytest.approx(expected_value, rel=1e-13, abs=1e-12)
        assert respondent_spread._evaluate_scalar_polynomial(
            slope, float(probability)
        ) == pytest.approx(expected_slope, rel=1e-13, abs=1e-10)


def _integrated_count_probability(values, outcome, spread, phi, *, survival):
    """Reference integral of one CDF side on a grid built for ``outcome``."""
    spreads = np.asarray([spread])
    scores, log_weights = respondent_spread._quadrature_grid(
        values, outcome, spreads, phi=phi
    )
    log_lambda = respondent_spread._log_quantiles_at_scores(values, scores, spreads)
    cdf, sf = respondent_spread._cdf_sf_at_rate(
        log_lambda.reshape(-1), outcome, phi=phi
    )
    with np.errstate(divide="ignore"):
        log_probability = np.log(sf if survival else cdf).reshape(1, -1)
    return float(np.exp(logsumexp(log_weights + log_probability, axis=1)[0]))


@pytest.mark.parametrize(
    ("outcome", "phi"), [(27, 0.08), (27, 0.0), (1, 0.6), (0, 0.04), (4_000, 0.0)]
)
def test_count_cdf_pairs_use_one_grid_per_outcome_and_match_direct_integrals(
    monkeypatch, outcome, phi
):
    values = epistemic_knots(_log_symmetric(4.0, 35), 1.0)
    spreads = (1.0, 1.4)
    original = respondent_spread._quadrature_grid
    calls = []

    def counted(*args, **kwargs):
        calls.append((args[1], tuple(args[2])))
        return original(*args, **kwargs)

    monkeypatch.setattr(respondent_spread, "_quadrature_grid", counted)
    together = respondent_spread._count_cdf_pairs(values, outcome, spreads, phi=phi)
    assert calls == [(outcome, spreads)]
    monkeypatch.setattr(respondent_spread, "_quadrature_grid", original)

    for row, spread in zip(together, spreads, strict=True):
        alone = respondent_spread._count_cdf_pairs(values, outcome, (spread,), phi=phi)
        assert np.array_equal(alone[0], row)
        lower_cdf, upper_cdf, lower_sf, upper_sf = row
        assert upper_cdf == pytest.approx(
            _integrated_count_probability(values, outcome, spread, phi, survival=False),
            abs=1e-13,
        )
        assert upper_sf == pytest.approx(
            _integrated_count_probability(values, outcome, spread, phi, survival=True),
            abs=1e-13,
        )
        if outcome == 0:
            assert (lower_cdf, lower_sf) == (0.0, 1.0)
        else:
            # The x - 1 pair comes from the x grid; a grid built for x - 1 agrees.
            assert lower_cdf == pytest.approx(
                _integrated_count_probability(
                    values, outcome - 1, spread, phi, survival=False
                ),
                abs=1e-10,
            )
            assert lower_sf == pytest.approx(
                _integrated_count_probability(
                    values, outcome - 1, spread, phi, survival=True
                ),
                abs=1e-10,
            )


def _fit_grid_small(monkeypatch):
    monkeypatch.setattr(respondent_spread, "K_GRID_POINTS", 19)
    monkeypatch.setattr(respondent_spread, "B_GRID_POINTS", 21)
    monkeypatch.setattr(respondent_spread, "K_COARSE_STRIDE", 3)
    monkeypatch.setattr(respondent_spread, "B_COARSE_STRIDE", 5)


def test_joint_fit_separates_level_bias_from_spread(monkeypatch):
    _fit_grid_small(monkeypatch)
    capture = _simulate_nb2_reference_capture(
        rng_seed=20261105, k_true=1.0, b_true=2.5, blocks=10
    )
    fit = fit_respondent_spread(*capture, bootstrap=8, seed=29, reference_replicates=0)
    assert abs(math.log(fit.k_hat)) < 0.15
    assert abs(math.log(fit.b_hat / 2.5)) < 0.15
    # With b fixed at 1 the level shift can only be absorbed by widening.
    assert fit.marginal.k_at_b_1.k_hat > 1.5
    assert fit.range.low <= fit.k_hat <= fit.range.high
    assert fit.b_range.low <= fit.b_hat <= fit.b_range.high
    assert fit.k_b_correlation is None or -1 <= fit.k_b_correlation <= 1
    # The level correction raises the left-out blocks' scores, but k adds
    # nothing given b.
    lobo = fit.diagnostics.leave_one_block_out
    assert lobo.joint_total_gain is not None and lobo.joint_total_gain > 0
    assert lobo.conditional_k_total_gain is not None
    assert lobo.conditional_k_total_gain <= 0
    assert fit.status == "rejected"
    assert fit.rejection_reason == (
        "leave-one-block-out gain of k_hat over k = 1, both at each fold's "
        "b_hat, is not positive"
    )
    assert fit.marginal.b_at_reference_k.reference_spread == 1.0
    assert fit.marginal.b_at_reference_k.reference_spread_basis == "face_value"
    assert fit.alternatives.poisson.b_range.low <= fit.alternatives.poisson.b_range.high
    assert (
        fit.alternatives.pooled_phi.b_range.low
        <= fit.alternatives.pooled_phi.b_range.high
    )


def test_joint_fit_recovers_nonunit_spread_and_level_bias(monkeypatch):
    _fit_grid_small(monkeypatch)
    capture = _simulate_nb2_reference_capture(
        rng_seed=20261106, k_true=1.6, b_true=2.0, blocks=10
    )
    fit = fit_respondent_spread(*capture, bootstrap=8, seed=31, reference_replicates=0)
    assert fit.status == "accepted", fit.rejection_reason
    # The finite NB2 sample is noisy in k; its joint estimate remains close.
    assert abs(math.log(fit.k_hat / 1.6)) < 0.15
    assert abs(math.log(fit.b_hat / 2.0)) < 0.15
    assert fit.range.low <= fit.k_hat <= fit.range.high
    assert fit.b_range.low <= fit.b_hat <= fit.b_range.high


def test_joint_fit_recovers_reference_spread_and_level(monkeypatch):
    _fit_grid_small(monkeypatch)
    capture = _simulate_nb2_reference_capture(
        rng_seed=20261103, k_true=1.0, b_true=1.0, blocks=6
    )
    fit = fit_respondent_spread(*capture, bootstrap=8, seed=37, reference_replicates=0)
    assert abs(math.log(fit.k_hat)) < 0.15
    assert abs(math.log(fit.b_hat)) < 0.2


def test_joint_fit_block_lobo_scores_are_evaluated_off_grid(monkeypatch):
    _fit_grid_small(monkeypatch)
    capture = _simulate_capture(1.25, rng_seed=4401, blocks=3, samples=1)
    instrument, artifact, _plan_payload, resolutions = capture
    fit = fit_respondent_spread(*capture, bootstrap=4, seed=3, reference_replicates=0)
    outcome_by_seed = {row.seed_id: row.value for row in resolutions.resolutions}
    phi_rows, _, _ = seed_dispersions(instrument.seeds)
    phi_by_seed = {seed_id: values[0] for seed_id, values in phi_rows.items()}
    laws_by_seed = {}
    for case in artifact.cases:
        assert case.answer is not None and case.answer.rate is not None
        laws_by_seed.setdefault(case.seed_id, []).append(
            (case.respondent.model_id, case.answer.rate)
        )

    def exact_score(seed_rows, k_value, b_value):
        seed_scores = []
        for seed_row in seed_rows:
            samples = [
                (
                    model_id,
                    respondent_spread._log_probability_values_grid(
                        epistemic_knots(law, 1.0) * b_value,
                        int(outcome_by_seed[seed_row.seed_id]),
                        np.asarray([k_value]),
                        phi=phi_by_seed[seed_row.seed_id],
                    ),
                )
                for model_id, law in laws_by_seed[seed_row.seed_id]
            ]
            seed_scores.append(_aggregate_law_scores(samples)[0])
        return math.fsum(seed_scores)

    for row in fit.diagnostics.leave_one_block_out.blocks:
        held = [
            seed_row
            for seed_row in instrument.seeds
            if seed_row.role == "primary" and seed_row.block_id == row.block_id
        ]
        assert row.log_score_at_k_hat_b_hat == pytest.approx(
            exact_score(held, row.k_hat, row.b_hat), abs=1e-12
        )
        assert row.log_score_at_k_1_b_hat == pytest.approx(
            exact_score(held, 1.0, row.b_hat), abs=1e-12
        )
        assert row.log_score_at_k_1_b_1 == pytest.approx(
            exact_score(held, 1.0, 1.0), abs=1e-12
        )


def _contract_treatments():
    digest = "a" * 64
    return CalibrationTreatments(
        models=(
            CalibrationModelTreatment(
                provider="openai",
                model_id="model",
                reasoning_effort="low",
                max_output_tokens=2048,
            ),
        ),
        samples_per_model={"model": 1},
        parse_attempts=3,
        frozen_call_schema_version=1,
        risk_contract_version=CONTEXTUAL_SCHEMA_VERSION,
        system_prompt_sha256=digest,
    )


def test_joint_contract_schema_ranges_and_status_validators():
    from test_support.respondent_calibration import respondent_spread_fit

    fit = respondent_spread_fit(_contract_treatments())
    assert fit.schema_version == 1
    assert RespondentSpreadFit.model_validate(fit.model_dump(mode="python")) == fit
    with pytest.raises(ValidationError, match="schema_version"):
        RespondentSpreadFit.model_validate(
            fit.model_dump(mode="python") | {"schema_version": 2}
        )
    with pytest.raises(ValidationError, match="low must not exceed high"):
        CommonBiasRange(low=2.0, high=1.0)
    with pytest.raises(ValidationError):
        CommonBiasRange(low=0.09, high=1.0)
    with pytest.raises(ValidationError):
        SpreadRange(low=1.2, high=1.1)
    with pytest.raises(ValidationError):
        JointGrid(k_points=20, b_points=21, k_coarse_stride=3, b_coarse_stride=5)
    with pytest.raises(ValidationError):
        JointGrid(k_points=19, b_points=20, k_coarse_stride=3, b_coarse_stride=5)
    with pytest.raises(ValidationError):
        JointRange(
            k=SpreadRange(low=1.0, high=2.0),
            b=CommonBiasRange(low=0.5, high=2.0),
            correlation=1.01,
        )

    payload = fit.model_dump(mode="python")
    # A fit records no review: none would make its use less exploratory.
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RespondentSpreadFit.model_validate(
            payload | {"reviewed_by": "A reviewer", "review_rationale": "Read the fit."}
        )
    for tail_form in _FORMS:
        accepted = RespondentSpreadFit.model_validate(
            payload | {"tail_form": tail_form}
        )
        assert accepted.tail_form == tail_form
    with pytest.raises(ValidationError, match="tail_form"):
        RespondentSpreadFit.model_validate(payload | {"tail_form": "stated_p95_peak"})
    bad_basis = payload.copy()
    bad_basis["marginal"] = payload["marginal"] | {
        "b_at_reference_k": payload["marginal"]["b_at_reference_k"]
        | {"reference_spread_basis": "face_value"}
    }
    with pytest.raises(ValidationError, match="reference spread"):
        RespondentSpreadFit.model_validate(bad_basis)

    rejected = payload | {
        "status": "rejected",
        "rejection_reason": "conditional k gain is not positive",
        "marginal": payload["marginal"]
        | {
            "b_at_reference_k": payload["marginal"]["b_at_reference_k"]
            | {"reference_spread": 1.0, "reference_spread_basis": "face_value"}
        },
    }
    assert RespondentSpreadFit.model_validate(rejected).status == "rejected"

    bad_phi = payload | {
        "alternatives": payload["alternatives"]
        | {"pooled_phi": payload["alternatives"]["pooled_phi"] | {"phi": 0.2}}
    }
    with pytest.raises(ValidationError, match="pooled-phi alternative"):
        RespondentSpreadFit.model_validate(bad_phi)

    bad_fallback = payload | {
        "seeds": (
            payload["seeds"][0]
            | {"phi_source": "pooled", "phi_fallback_reason": "short history"},
        )
    }
    with pytest.raises(ValidationError, match="fallback ids"):
        RespondentSpreadFit.model_validate(bad_fallback)

    with pytest.raises(ValidationError, match="unavailable curation-sensitivity"):
        CurationSensitivityDiagnostics(
            seed_ids=(), k_hat=1.0, unavailable_reason="no rows remain"
        )


def test_fit_records_joint_superblock_curation_and_per_model_ranges(monkeypatch):
    _fit_grid_small(monkeypatch)
    capture = _simulate_capture(
        1.0,
        rng_seed=20261103,
        blocks=4,
        samples=1,
        super_blocks={
            "region-a": ("block-00", "block-01"),
            "region-b": ("block-02", "block-03"),
        },
        curation_sensitive_seed_ids=("primary-00", "primary-04"),
    )
    fit = fit_respondent_spread(*capture, bootstrap=8, seed=31, reference_replicates=0)
    assert fit.diagnostics.super_block_bootstrap.super_blocks == 2
    assert (
        fit.diagnostics.super_block_bootstrap.range.k.low
        <= fit.diagnostics.super_block_bootstrap.range.k.high
    )
    assert (
        fit.diagnostics.super_block_bootstrap.range.b.low
        <= fit.diagnostics.super_block_bootstrap.range.b.high
    )
    curation = fit.diagnostics.without_curation_sensitive_seeds
    assert curation.seed_ids == ("primary-00", "primary-04")
    assert curation.k_hat is not None and curation.b_hat is not None
    assert curation.range is not None and curation.blocks == 4
    assert fit.diagnostics.per_model
    assert all(
        row.k_hat >= 0.5 and row.b_hat >= 0.1 for row in fit.diagnostics.per_model
    )


def test_zero_probability_primary_is_reported_excluded_and_has_pit_one(monkeypatch):
    _fit_grid_small(monkeypatch)
    respondent = Respondent(respondent_id="model:0", model_id="model", sample_index=0)
    seeds = (
        _seed("zero", "zero-block"),
        _seed("live-a", "block-a"),
        _seed("live-b", "block-b"),
    )
    instrument = CalibrationInstrument(
        seeds=seeds,
        respondents=(respondent,),
        latest_training_cutoff=date(2026, 5, 31),
        resolution_protocol=_resolution_protocol(),
        risk_contract_version=CONTEXTUAL_SCHEMA_VERSION,
    )
    laws = {
        "zero": _law(0, 0, 0, 0, 0),
        "live-a": _law(1, 2, 4, 8, 16),
        "live-b": _law(1, 2, 4, 8, 16),
    }
    plan = _plan(instrument)
    artifact = CalibrationArtifact(
        instrument_sha256=instrument.digest(),
        collection_plan_sha256=canonical_json_sha256(plan),
        cases=tuple(
            CalibrationCase(
                seed_id=row.seed_id,
                respondent=respondent,
                answer=CalibrationAnswer(
                    status="available",
                    rate=laws[row.seed_id],
                    reasoning=_reasoning(row.seed_id),
                ),
            )
            for row in seeds
        ),
    )
    resolutions = SeedResolutionSet(
        resolutions=tuple(
            SeedResolution(
                seed_id=row.seed_id,
                value=4,
                retrieved_at=date(2026, 11, 1),
                snapshot_url="https://example.test/final",
                snapshot_sha256="d" * 64,
                script="scripts/resolve.py",
                notes="",
            )
            for row in seeds
        )
    )
    fit = fit_respondent_spread(
        instrument, artifact, plan, resolutions, bootstrap=4, reference_replicates=0
    )
    assert fit.diagnostics.zero_probability_primary_seeds == ("zero",)
    assert fit.status == "rejected"
    assert fit.rejection_reason is not None and "zero" in fit.rejection_reason
    assert fit.marginal.b_at_reference_k.reference_spread_basis == "face_value"
    for summary in (
        fit.diagnostics.primary_pit.at_k_1_b_1,
        fit.diagnostics.primary_pit.at_k_hat_b_1,
        fit.diagnostics.primary_pit.at_k_1_b_hat,
        fit.diagnostics.primary_pit.at_k_hat_b_hat,
    ):
        zero_pit = next(row for row in summary.seeds if row.seed_id == "zero")
        assert zero_pit.pit == 1.0


def test_auxiliary_count_and_usd_pits_use_the_joint_fit(monkeypatch):
    _fit_grid_small(monkeypatch)
    respondent = Respondent(respondent_id="model:0", model_id="model", sample_index=0)
    seeds = (
        _seed("primary-a", "block-a"),
        _seed("primary-b", "block-b"),
        _seed("control", "control", role="control"),
        _seed("money", "secondary", role="secondary", unit="usd"),
        _seed("secondary-count", "secondary-count", role="secondary"),
    )
    instrument = CalibrationInstrument(
        seeds=seeds,
        respondents=(respondent,),
        latest_training_cutoff=date(2026, 5, 31),
        resolution_protocol=_resolution_protocol(),
        risk_contract_version=CONTEXTUAL_SCHEMA_VERSION,
    )
    law = _law(1, 2, 4, 8, 16)
    rates = {row.seed_id: law for row in seeds}
    rates["money"] = _law(100, 100, 100, 100, 100)
    plan = _plan(instrument)
    artifact = CalibrationArtifact(
        instrument_sha256=instrument.digest(),
        collection_plan_sha256=canonical_json_sha256(plan),
        cases=tuple(
            CalibrationCase(
                seed_id=row.seed_id,
                respondent=respondent,
                answer=CalibrationAnswer(
                    status="available",
                    rate=rates[row.seed_id],
                    reasoning=_reasoning(row.seed_id),
                ),
            )
            for row in seeds
        ),
    )
    values = {
        "primary-a": 3,
        "primary-b": 5,
        "control": 4,
        "money": 100,
        "secondary-count": 3,
    }
    resolutions = SeedResolutionSet(
        resolutions=tuple(
            SeedResolution(
                seed_id=row.seed_id,
                value=values[row.seed_id],
                retrieved_at=date(2026, 11, 1),
                snapshot_url="https://example.test/final",
                snapshot_sha256="e" * 64,
                script="scripts/resolve.py",
                notes="",
            )
            for row in seeds
        )
    )
    fit = fit_respondent_spread(
        instrument,
        artifact,
        plan,
        resolutions,
        bootstrap=4,
        seed=7,
        reference_replicates=0,
    )
    rows = {row.seed_id: row for row in fit.diagnostics.controls_and_secondary_pit}
    assert set(rows) == {"control", "money", "secondary-count"}
    assert rows["control"].pit_at_k_1_b_1 is not None
    assert rows["control"].pit_at_k_hat_b_hat is not None
    assert rows["secondary-count"].pit_at_k_1_b_1 is not None
    assert rows["secondary-count"].pit_at_k_hat_b_hat is not None
    assert rows["money"].pit_at_k_1_b_1 == rows["money"].pit_at_k_hat_b_hat
    # Every seed's row carries its resolved value, whatever its role.
    assert {row.seed_id: row.value for row in fit.seeds} == values


def test_simulated_reference_is_seeded_and_reports_fixed_bias_diagnostics(monkeypatch):
    _fit_grid_small(monkeypatch)
    capture = _simulate_capture(1.25, rng_seed=20261102, blocks=2, samples=1)
    fit = fit_respondent_spread(*capture, bootstrap=4, seed=19, reference_replicates=2)
    reference = fit.diagnostics.simulated_reference
    assert reference is not None
    assert reference.replicates == 2 and reference.seed == 19
    assert reference.b_hat == fit.b_hat
    assert reference.primary_pit_at_k_1_b_hat.median.observed is not None
    assert reference.primary_pit_at_k_refit_b_hat.median.observed is not None
    assert reference.cooke_score_at_k_1_b_hat.observed is not None
    assert reference.cooke_score_at_k_refit_b_hat.observed is not None
    assert (
        reference.conditional_k_leave_one_block_out_gain.p05
        <= reference.conditional_k_leave_one_block_out_gain.p95
    )
    assert fit.diagnostics.reference_replicates == 2


def test_parallel_and_serial_fits_are_bit_identical(monkeypatch):
    _fit_grid_small(monkeypatch)
    capture = _simulate_capture(1.25, rng_seed=20261201, blocks=2, samples=1)
    serial = fit_respondent_spread(
        *capture,
        bootstrap=20,
        seed=71,
        reference_replicates=2,
        workers=1,
    )
    parallel = fit_respondent_spread(
        *capture,
        bootstrap=20,
        seed=71,
        reference_replicates=2,
        workers=2,
    )
    assert parallel.model_dump(mode="json") == serial.model_dump(mode="json")


def test_resolution_script_mismatch_still_refuses_a_fit():
    instrument, artifact, plan, resolutions = _simulate_capture(
        1.0, rng_seed=17, blocks=2, samples=1
    )
    changed = resolutions.resolutions[0].model_copy(
        update={"script": "scripts/other.py"}
    )
    modified = SeedResolutionSet(resolutions=(changed, *resolutions.resolutions[1:]))
    with pytest.raises(ValueError, match="different script"):
        fit_respondent_spread(
            instrument, artifact, plan, modified, bootstrap=2, reference_replicates=0
        )


def test_fit_preserves_pooled_phi_history_fallback_fields(monkeypatch):
    _fit_grid_small(monkeypatch)
    instrument, artifact, _plan_payload, resolutions = _simulate_capture(
        1.0, rng_seed=203, blocks=2, samples=1
    )
    first_seed = instrument.seeds[0]
    first_seed = first_seed.model_copy(
        update={
            "evidence": CalibrationSeedEvidence(
                text="A deliberately short monthly history.",
                history=first_seed.evidence.history[:5],
            )
        }
    )
    instrument = instrument.model_copy(
        update={"seeds": (first_seed, *instrument.seeds[1:])}
    )
    plan = _plan(instrument)
    artifact = CalibrationArtifact(
        instrument_sha256=instrument.digest(),
        collection_plan_sha256=canonical_json_sha256(plan),
        cases=artifact.cases,
    )
    fit = fit_respondent_spread(
        instrument,
        artifact,
        plan,
        resolutions,
        bootstrap=4,
        seed=8,
        reference_replicates=0,
    )
    fallback = next(row for row in fit.seeds if row.seed_id == first_seed.seed_id)
    assert fit.observation.fallback_seed_ids == (first_seed.seed_id,)
    assert fallback.phi == fit.observation.pooled_phi
    assert fallback.phi_periods == 5
    assert fallback.phi_source == "pooled"
    assert (
        fallback.phi_fallback_reason
        == "fewer than 6 periods at the window's granularity"
    )


def test_unscored_primary_seed_is_recorded_without_a_pit(monkeypatch):
    _fit_grid_small(monkeypatch)
    instrument, artifact, plan, resolutions = _simulate_capture(
        1.0, rng_seed=2201, blocks=2, samples=1
    )
    unscored_seed_id = instrument.seeds[-1].seed_id
    artifact = artifact.model_copy(
        update={
            "cases": tuple(
                row.model_copy(update={"answer": None})
                if row.seed_id == unscored_seed_id
                else row
                for row in artifact.cases
            )
        }
    )
    fit = fit_respondent_spread(
        instrument,
        artifact,
        plan,
        resolutions,
        bootstrap=4,
        seed=1,
        reference_replicates=0,
    )
    assert fit.diagnostics.unscored_primary_seeds == (unscored_seed_id,)
    row = next(row for row in fit.seeds if row.seed_id == unscored_seed_id)
    assert row.pit_at_k_1_b_1 is None
    assert row.pit_at_k_hat_b_hat is None
