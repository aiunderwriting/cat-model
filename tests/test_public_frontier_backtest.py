"""Rolling-origin pinball backtest of synthetic anchored frontier forecasts."""

import json
from datetime import UTC, datetime

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.stats import fisk

from catmodel.contracts.public_frontier import (
    YEAR_SECONDS,
    PublicFrontierBacktest,
    PublicFrontierForecast,
)
from catmodel.forecast.public_frontier_backtest import (
    backtest_public_frontier,
    origin_forecast,
)
from scripts.backtest_public_frontier import main
from test_support.public_frontier import synthetic_forecast, synthetic_release

AS_OF = datetime(2026, 10, 1, tzinfo=UTC)
REFERENCE = datetime(2024, 1, 1, tzinfo=UTC)


def _month_starts(first, last):
    at = first
    while at <= last:
        yield at
        at = at.replace(year=at.year + at.month // 12, month=at.month % 12 + 1)


def _line(at):
    return 100 + 12 * (at - REFERENCE).total_seconds() / YEAR_SECONDS


def _linear_forecast(extra=()):
    """One lab releasing exactly on a 12-point-per-year line every month."""
    payload = synthetic_forecast().model_dump(mode="json")
    payload["coordinates"] = payload["coordinates"][:1]
    payload["rate_prior"] = {
        "mean": 12,
        "sd": 0,
        "rationale": "Synthetic fixed rate equal to the release line's.",
    }
    payload["trend_window_start"] = "2025-10-01T00:00:00Z"
    payload["releases"] = [
        synthetic_release("lab_a", f"a-{at:%Y-%m}", at.isoformat(), _line(at), sd=0)
        for at in _month_starts(REFERENCE, AS_OF)
    ] + list(extra)
    return PublicFrontierForecast.model_validate(payload)


def test_exact_trend_wins_and_persistence_has_its_closed_form_loss():
    result = backtest_public_frontier(_linear_forecast(), n_draws=500, seed=1)
    # The first origin has a full 365-day window of releases behind it; the
    # last has its shortest horizon resolved by as-of.
    assert result.origins[0] == datetime(2025, 1, 1, tzinfo=UTC)
    assert result.origins[-1] == datetime(2026, 7, 1, tzinfo=UTC)
    assert result.scored_cases == 19 + 16 + 10
    rows = {row.candidate_id: row for row in result.candidates}
    assert result.lowest_loss_candidate == "trend_only"
    assert rows["trend_only"].mean_pinball_loss == pytest.approx(0, abs=1e-9)
    assert all(
        rows[f"pace_shape_{shape:g}"].mean_pinball_loss > 0
        for shape in (1.5, 2, 3, 5, 8, 13)
    )
    # A point forecast at the observed frontier has mean pinball loss equal to
    # half of the realized gain, which is 12 points per year of horizon.
    gains = [
        12 * (horizon_end - origin).total_seconds() / YEAR_SECONDS
        for months in (3, 6, 12)
        for origin in result.origins
        if (horizon_end := _shift(origin, months)) <= AS_OF
    ]
    assert rows["no_progress"].mean_pinball_loss == pytest.approx(0.5 * np.mean(gains))
    assert rows["no_progress"].central_90_coverage == 0
    assert set(rows["trend_only"].mean_pinball_loss_by_horizon) == {"3", "6", "12"}
    # Persistence is always exceeded; the exact trend has no upper-tail loss.
    assert rows["no_progress"].upper_exceedance_rates == {"p90": 1, "p95": 1, "p99": 1}
    assert rows["trend_only"].mean_upper_tail_pinball_loss == pytest.approx(0, abs=1e-9)
    # The winner differs from itself by zero; others carry a paired bootstrap SE.
    assert (
        rows["trend_only"].loss_difference_vs_lowest,
        rows["trend_only"].loss_difference_se,
    ) == (0, 0)
    assert rows["no_progress"].loss_difference_vs_lowest == pytest.approx(
        rows["no_progress"].mean_pinball_loss
    )
    assert rows["no_progress"].loss_difference_se > 0
    assert result.bootstrap_block_origins == 12


def test_every_pace_candidate_is_truncated_at_the_forecasts_cap():
    """One lab on an exact 12-point line with a fixed rate: a forecast quantile
    of the gain is the realized gain times the pace's quantile, so each
    candidate's mean loss has a closed form in the capped pace's quantiles,
    here taken from SciPy's log-logistic law."""
    cap = 1.5
    forecast = _linear_forecast().model_copy(update={"pace_cap": cap})
    result = backtest_public_frontier(forecast, n_draws=4000, seed=2)
    assert result.pace_cap == cap
    assert any(
        "truncated at the forecast's pace cap" in row for row in result.limitations
    )
    gains = np.asarray(
        [
            12 * (horizon_end - origin).total_seconds() / YEAR_SECONDS
            for months in (3, 6, 12)
            for origin in result.origins
            if (horizon_end := _shift(origin, months)) <= AS_OF
        ]
    )
    levels = np.asarray(result.quantile_levels)

    def expected_loss(shape, limit):
        law = fisk(c=shape)
        pace = law.ppf(levels * law.cdf(limit))
        # Pinball loss of a quantile forecast at `pace` times the realized gain.
        per_gain = np.where(pace < 1, levels * (1 - pace), (1 - levels) * (pace - 1))
        return gains.mean() * per_gain.mean()

    rows = {row.candidate_id: row for row in result.candidates}
    for shape in (1.5, 3, 8):
        loss = rows[f"pace_shape_{shape:g}"].mean_pinball_loss
        assert loss == pytest.approx(expected_loss(shape, cap), rel=0.04)
        # The untruncated law would cost the wide shapes clearly more.
        if shape < 8:
            assert expected_loss(shape, np.inf) > 1.15 * loss
    # The cap does not touch the two benchmarks.
    uncapped = {
        row.candidate_id: row
        for row in backtest_public_frontier(
            _linear_forecast(), n_draws=4000, seed=2
        ).candidates
    }
    for benchmark in ("trend_only", "no_progress"):
        assert rows[benchmark] == uncapped[benchmark]


def _shift(at, months):
    year, month = divmod(at.month - 1 + months, 12)
    return at.replace(year=at.year + year, month=month + 1)


def test_relisted_scores_do_not_change_the_backtest():
    relisted = [
        synthetic_release(
            "lab_a", f"a-{at:%Y-%m}-relisted", at.isoformat(), _line(at) - 1, sd=0
        )
        for at in _month_starts(datetime(2024, 2, 1, tzinfo=UTC), AS_OF)
    ]
    base = backtest_public_frontier(_linear_forecast(), n_draws=200, seed=3)
    duplicated = backtest_public_frontier(
        _linear_forecast(relisted), n_draws=200, seed=3
    )
    assert duplicated.candidates == base.candidates
    assert duplicated.origins == base.origins


def test_origin_forecast_uses_only_available_releases_and_the_same_settings():
    source = synthetic_forecast()
    origin = datetime(2026, 4, 1, tzinfo=UTC)
    historical = origin_forecast(source, origin)
    assert historical.as_of == origin
    # The window keeps its length (273 days) rather than its calendar months.
    assert historical.trend_window_start == datetime(2025, 7, 2, tzinfo=UTC)
    assert all(row.available_at <= origin for row in historical.releases)
    assert historical.rate_prior == source.rate_prior
    assert historical.trend_discrepancy_sd == source.trend_discrepancy_sd
    early = origin_forecast(source, datetime(2026, 2, 15, tzinfo=UTC))
    assert [row.lab_id for row in early.coordinates] == ["lab_a"]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"horizons_months": ()}, "horizons"),
        ({"horizons_months": (3, 3)}, "horizons"),
        ({"horizons_months": (0,)}, "horizons"),
        ({"pace_shapes": (0.0,)}, "pace shapes"),
        ({"pace_shapes": (2.0, 2.0)}, "distinct"),
        ({"n_draws": 10}, "100 draws"),
        ({"seed": -1}, "seed"),
        ({"horizons_months": (48,)}, "full trend window"),
    ],
)
def test_invalid_backtests_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        backtest_public_frontier(_linear_forecast(), **kwargs)


def test_backtest_contract_requires_its_lowest_loss_candidate():
    payload = backtest_public_frontier(
        _linear_forecast(), n_draws=200, seed=1
    ).model_dump(mode="json")
    PublicFrontierBacktest.model_validate(payload)
    lowest = next(
        index
        for index, row in enumerate(payload["candidates"])
        if row["candidate_id"] == payload["lowest_loss_candidate"]
    )
    nonzero = [dict(row) for row in payload["candidates"]]
    nonzero[lowest]["loss_difference_se"] = 1.0
    for change in (
        {"lowest_loss_candidate": "no_progress"},
        {"candidates": nonzero},
        {
            "candidates": [
                {**payload["candidates"][0], "pace_shape": None},
                *payload["candidates"][1:],
            ]
        },
    ):
        with pytest.raises(ValidationError):
            PublicFrontierBacktest.model_validate({**payload, **change})


def test_the_script_writes_the_backtest_and_refuses_to_overwrite(tmp_path, capsys):
    source = tmp_path / "forecast.json"
    output = tmp_path / "backtest.json"
    source.write_text(_linear_forecast().model_dump_json())
    args = [str(source), str(output)]
    assert main(args) == 0
    saved = PublicFrontierBacktest.model_validate_json(output.read_text())
    # The script prints where it wrote and each candidate's mean loss.
    assert json.loads(capsys.readouterr().out) == {
        "output": str(output),
        "scored_cases": saved.scored_cases,
        "lowest_loss_candidate": "trend_only",
        "mean_pinball_loss": {
            row.candidate_id: row.mean_pinball_loss for row in saved.candidates
        },
    }
    # It scores the documented horizons and pace-shape candidates.
    assert saved.horizons_months == (3, 6, 12)
    assert [row.pace_shape for row in saved.candidates] == [
        1.5,
        2,
        3,
        5,
        8,
        13,
        None,
        None,
    ]
    assert [row.candidate_id for row in saved.candidates][-2:] == [
        "trend_only",
        "no_progress",
    ]
    written = output.read_bytes()
    assert main(args) == 2
    captured = capsys.readouterr()
    assert f"{output} already exists" in captured.err and not captured.out
    assert output.read_bytes() == written
    # A forecast without releases, such as the committed template, is refused.
    template = tmp_path / "template.json"
    payload = _linear_forecast().model_dump(mode="json")
    del payload["releases"]
    template.write_text(json.dumps(payload))
    assert main([str(template), str(tmp_path / "other.json")]) == 2
    assert "releases" in capsys.readouterr().err
    assert not (tmp_path / "other.json").exists()
