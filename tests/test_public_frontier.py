"""Synthetic lab trends, observed anchors, independent lab paces and provenance checks."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.integrate import quad
from scipy.stats import fisk, rankdata

from catmodel.contracts.public_frontier import (
    VARIANTS,
    YEAR_SECONDS,
    PublicFrontierEnsemble,
    PublicFrontierForecast,
    PublicRatePrior,
    forecast_times,
)
from catmodel.forecast.public_frontier import (
    compare_public_frontier,
    fit_public_trend,
    lab_frontier_records,
    pace_quantile,
    record_residuals,
    sample_public_frontier,
    staleness,
)
from test_support.public_frontier import (
    SYNTHETIC_RELEASES,
    synthetic_forecast,
    synthetic_release,
)

START = datetime(2027, 1, 1, tzinfo=UTC)
END = datetime(2028, 1, 1, tzinfo=UTC)


def sample(source, **kwargs):
    return sample_public_frontier(source, period_start=START, period_end=END, **kwargs)


def values(ensemble):
    return np.asarray([draw.values for draw in ensemble.draws])


def changed(source=None, **updates):
    return PublicFrontierForecast.model_validate(
        {**(source or synthetic_forecast()).model_dump(mode="json"), **updates}
    )


def changed_lab(source, lab_id, **updates):
    payload = source.model_dump(mode="json")
    for coordinate in payload["coordinates"]:
        if coordinate["lab_id"] == lab_id:
            coordinate.update(updates)
    return PublicFrontierForecast.model_validate(payload)


def years(source, at):
    return (datetime.fromisoformat(at) - source.as_of).total_seconds() / YEAR_SECONDS


def test_only_frontier_raising_releases_are_records_and_last_record_anchors():
    source = synthetic_forecast()
    records = lab_frontier_records(source)
    assert [row.release_id for row in records["lab_a"]] == [
        "a-prewindow",
        "a-1",
        "a-2",
    ]
    posterior = fit_public_trend(source)
    # The pre-window leader is outside the trend window; the lower release is
    # not a frontier observation.
    assert posterior.labs["lab_a"].likelihood_release_ids == ("a-1", "a-2")
    assert posterior.labs["lab_a"].as_of_release_id == "a-2"
    assert posterior.labs["lab_a"].as_of_value == 100
    assert posterior.labs["lab_b"].observation_count == 2
    assert posterior.labs["lab_b"].as_of_value == 90


def test_pooled_rate_matches_generalized_least_squares_with_lab_levels():
    source = synthetic_forecast()
    posterior = fit_public_trend(source)
    # Window records: lab_a (93, 100), lab_b (85, 90); each has trend
    # discrepancy 2 and benchmark error 1, so variance 5.
    rows = [
        (0, "2026-02-01T00:00:00+00:00", 93),
        (0, "2026-08-15T00:00:00+00:00", 100),
        (1, "2026-03-01T00:00:00+00:00", 85),
        (1, "2026-09-01T00:00:00+00:00", 90),
    ]
    design = np.array(
        [[lab == 0, lab == 1, years(source, at)] for lab, at, _ in rows], float
    )
    values = np.array([value for *_, value in rows], float)
    # Nearly flat lab levels and the N(10, 2²) rate prior.
    precision = np.diag([1e-12, 1e-12, 1 / 4]) + design.T @ design / 5
    covariance = np.linalg.inv(precision)
    mean = covariance @ (design.T @ values / 5 + np.array([0, 0, 10 / 4]))
    assert posterior.rate_location == pytest.approx(mean[2], rel=1e-9)
    assert posterior.rate_variance == pytest.approx(covariance[2, 2], rel=1e-9)


def test_lab_with_one_window_record_carries_no_rate_information():
    source = synthetic_forecast()
    payload = source.model_dump(mode="json")
    payload["coordinates"].append({**payload["coordinates"][0], "lab_id": "lab_c"})
    payload["releases"].append(
        synthetic_release("lab_c", "c-1", "2026-06-01T00:00:00Z", 70)
    )
    a = fit_public_trend(source)
    b = fit_public_trend(PublicFrontierForecast.model_validate(payload))
    assert (b.rate_location, b.rate_variance) == pytest.approx(
        (a.rate_location, a.rate_variance)
    )
    assert b.labs["lab_c"].observation_count == 1


def test_carried_forward_and_relisted_scores_do_not_shrink_the_posterior():
    source = synthetic_forecast()
    repeated = list(SYNTHETIC_RELEASES)
    # Re-list every record's score at each later month start, as a monthly
    # carried-forward panel would, and add lower releases. None is a new
    # frontier observation.
    for release in SYNTHETIC_RELEASES:
        carried = datetime.fromisoformat(release["available_at"]).replace(day=1)
        month = 0
        while True:
            month += 1
            carried = (carried + timedelta(days=32)).replace(day=1)
            if carried > source.as_of:
                break
            repeated.append(
                synthetic_release(
                    release["lab_id"],
                    f"{release['release_id']}-carried-{month}",
                    carried.isoformat(),
                    release["value"],
                    sd=0,
                )
            )
            repeated.append(
                synthetic_release(
                    release["lab_id"],
                    f"{release['release_id']}-lower-{month}",
                    carried.isoformat(),
                    release["value"] - 5,
                )
            )
    duplicated = synthetic_forecast(releases=repeated)
    assert len(duplicated.releases) > 5 * len(source.releases)
    assert fit_public_trend(duplicated) == fit_public_trend(source)
    for variant in VARIANTS:
        np.testing.assert_array_equal(
            values(sample(duplicated, n_draws=50, seed=3, variant=variant)),
            values(sample(source, n_draws=50, seed=3, variant=variant)),
        )


def test_same_instant_lower_release_is_not_a_record_whatever_its_identity():
    releases = [
        synthetic_release("lab_a", "a-high", "2026-08-15T00:00:00Z", 100),
        synthetic_release("lab_a", "a-0-low", "2026-08-15T00:00:00Z", 95),
        synthetic_release("lab_b", "b-1", "2026-09-01T00:00:00Z", 90),
    ]
    forward = synthetic_forecast(releases=releases)
    backward = synthetic_forecast(releases=list(reversed(releases)))
    assert fit_public_trend(forward) == fit_public_trend(backward)
    assert fit_public_trend(forward).labs["lab_a"].likelihood_release_ids == ("a-high",)


def test_without_window_records_the_rate_keeps_its_prior_and_labs_their_anchor():
    source = changed(
        synthetic_forecast(), trend_window_start=synthetic_forecast().as_of
    )
    posterior = fit_public_trend(source)
    assert all(lab.observation_count == 0 for lab in posterior.labs.values())
    assert (posterior.rate_location, posterior.rate_variance) == (10, 4)
    assert posterior.labs["lab_a"].as_of_value == 100


@pytest.mark.parametrize("variant", VARIANTS)
def test_paths_start_at_published_frontier_and_frontier_never_falls(variant):
    source = synthetic_forecast()
    anchors = np.array([100, 90])
    at_as_of = sample_public_frontier(
        source,
        period_start=source.as_of,
        period_end=END,
        n_draws=400,
        seed=11,
        variant=variant,
    )
    paths = values(at_as_of)
    np.testing.assert_array_equal(paths[:, 0, :], np.broadcast_to(anchors, (400, 2)))
    assert np.all(np.diff(paths, axis=1) >= 0)
    frontier = paths.max(axis=2)
    assert np.all(frontier >= 100)
    assert np.all(np.diff(frontier, axis=1) >= 0)
    later = values(sample(source, n_draws=400, seed=11, variant=variant))
    assert np.all(later >= anchors)


def test_forecast_period_is_a_parameter_with_month_start_knots():
    source = synthetic_forecast()
    start = datetime(2026, 10, 15, tzinfo=UTC)
    end = datetime(2027, 4, 1, tzinfo=UTC)
    expected = (
        start,
        *(datetime(2026, month, 1, tzinfo=UTC) for month in (11, 12)),
        *(datetime(2027, month, 1, tzinfo=UTC) for month in (1, 2, 3)),
        end,
    )
    assert forecast_times(start, end) == expected
    ensemble = sample_public_frontier(
        source, period_start=start, period_end=end, n_draws=3, seed=1
    )
    assert ensemble.times == expected
    later = sample_public_frontier(
        source,
        period_start=datetime(2029, 1, 1, tzinfo=UTC),
        period_end=datetime(2030, 1, 1, tzinfo=UTC),
        n_draws=3,
        seed=1,
    )
    assert len(later.times) == 13
    assert later.times[-1].year == 2030


@pytest.mark.parametrize(
    "start,end,match",
    [
        (datetime(2026, 9, 1, tzinfo=UTC), END, "as-of cannot follow"),
        (END, START, "end after it starts"),
        (START, START, "end after it starts"),
        (datetime(2027, 1, 1), END, "timezone"),
    ],
)
def test_invalid_forecast_periods_are_rejected(start, end, match):
    with pytest.raises(ValueError, match=match):
        sample_public_frontier(
            synthetic_forecast(), period_start=start, period_end=end, n_draws=2, seed=1
        )


def test_ensemble_rejects_altered_times_or_anchor():
    payload = sample(synthetic_forecast(), n_draws=2, seed=1).model_dump(mode="json")
    altered = {**payload, "times": payload["times"][:-2] + payload["times"][-1:]}
    with pytest.raises(ValidationError, match="month start"):
        PublicFrontierEnsemble.model_validate(altered)
    payload["posterior"]["labs"]["lab_a"]["as_of_value"] = 101
    with pytest.raises(ValidationError, match="observed as-of frontier"):
        PublicFrontierEnsemble.model_validate(payload)


def test_fixed_prior_paths_grow_each_lab_at_the_rate_times_its_own_pace():
    source = synthetic_forecast(uncertain=False)
    result = sample(source, n_draws=2, seed=3)
    assert result.times[0] == START and result.times[-1] == END
    elapsed = np.asarray(
        [(at - source.as_of).total_seconds() / YEAR_SECONDS for at in result.times]
    )
    assert result.posterior.rate_location == 10
    for draw in result.draws:
        assert draw.baseline_rate == 10
        rates = 10 * np.asarray(draw.pace_multipliers)
        np.testing.assert_allclose(draw.rates, rates, rtol=1e-12)
        np.testing.assert_allclose(
            draw.values, [100, 90] + elapsed[:, None] * rates, rtol=1e-12
        )


def test_positive_truncation_samples_the_rate_kernel():
    source = changed(
        synthetic_forecast(),
        trend_window_start=synthetic_forecast().as_of,
        rate_prior={"mean": 0, "sd": 1, "rationale": "Synthetic half-normal rate."},
    )
    ensemble = sample(source, n_draws=5000, seed=5)
    rates = np.array([row.baseline_rate for row in ensemble.draws])
    assert np.all(rates > 0)
    assert np.mean(rates) == pytest.approx(np.sqrt(2 / np.pi), abs=0.025)
    assert np.all(np.diff(values(ensemble), axis=1) >= 0)


def _capped_cdf(pace, shape, cap):
    """The capped pace law's CDF from its definition: the median-one
    log-logistic CDF divided by its value at the cap."""
    return fisk(c=shape).cdf(np.minimum(pace, cap)) / fisk(c=shape).cdf(cap)


@pytest.mark.parametrize("shape,cap", [(5, 4), (3, 4), (8, 4), (5, 2), (0.8, 1.5)])
def test_pace_quantiles_invert_the_log_logistic_law_truncated_at_the_cap(shape, cap):
    probabilities = np.array([1e-9, 0.01, 0.25, 0.5, 0.75, 0.99, 1 - 1e-9])
    pace = pace_quantile(shape, probabilities, cap)
    assert np.all(np.diff(pace) > 0) and pace[0] > 0 and pace[-1] < cap
    np.testing.assert_allclose(_capped_cdf(pace, shape, cap), probabilities, rtol=1e-9)
    # The cap removes the share of the log-logistic law beyond it, so the
    # capped median lies below one, the median of the law it truncates.
    below_cap = cap**shape / (1 + cap**shape)
    assert pace[3] == pytest.approx(
        (0.5 * below_cap / (1 - 0.5 * below_cap)) ** (1 / shape), rel=1e-12
    )
    assert pace[3] < 1


def test_a_distant_cap_leaves_the_log_logistic_pace_and_bad_inputs_fail():
    probabilities = np.array([0.001, 0.3, 0.5, 0.9, 0.999])
    np.testing.assert_allclose(
        pace_quantile(5, probabilities, 1e6), fisk(c=5).ppf(probabilities), rtol=1e-12
    )
    # The sampler raises on underflow; a negligible share beyond the cap is
    # not one.
    with np.errstate(under="raise"):
        np.testing.assert_allclose(
            pace_quantile(60, probabilities, 1e9),
            fisk(c=60).ppf(probabilities),
            rtol=1e-12,
        )
    for probability in (0.0, 1.0):
        with pytest.raises(ValueError, match="no draw was replaced"):
            pace_quantile(5, np.array([0.5, probability]), 4)
    with pytest.raises(ValueError, match="cap must be positive"):
        pace_quantile(5, probabilities, 0)


def test_each_lab_draws_an_independent_capped_loglogistic_pace():
    source = synthetic_forecast(uncertain=False)
    ensemble = sample(source, n_draws=6000, seed=7)
    pace = np.array([row.pace_multipliers for row in ensemble.draws])
    assert np.all(pace[:, 0] != pace[:, 1])
    assert abs(np.corrcoef(rankdata(pace[:, 0]), rankdata(pace[:, 1]))[0, 1]) < 0.05
    # No pace exceeds the cap, and the law has no atom there: the untruncated
    # law would put 1.5% of 12000 draws beyond it.
    assert (source.pace_shape, source.pace_cap) == (3, 4)
    assert 3.5 < pace.max() < 4
    assert fisk(c=3).sf(4) * pace.size > 180
    probabilities = np.array([0.05, 0.5, 0.95])
    below = probabilities * fisk(c=3).cdf(4)
    expected = (below / (1 - below)) ** (1 / 3)
    for lab in range(2):
        np.testing.assert_allclose(
            np.quantile(pace[:, lab], probabilities), expected, rtol=0.05
        )
    # The comparison separates the capped law from the one it truncates.
    grid = np.sort(pace.ravel())
    empirical = np.arange(1, grid.size + 1) / grid.size
    assert np.abs(empirical - _capped_cdf(grid, 3, 4)).max() < 0.015
    assert np.abs(empirical - fisk(c=3).cdf(grid)).max() > 0.015


def test_the_frontier_leader_changes_as_often_as_independent_paces_imply():
    """The trailing lab leads at the period end with the probability its pace
    must exceed the leader's by the 10-point gap, from an independent integral."""
    source = synthetic_forecast(uncertain=False)
    n = 20000
    ensemble = sample(source, n_draws=n, seed=12)
    final = values(ensemble)[:, -1, :]
    share = float(np.mean(final[:, 1] > final[:, 0]))
    elapsed = years(source, "2028-01-01T00:00:00+00:00")
    excess = (100 - 90) / (10 * elapsed)
    law, cap = fisk(c=source.pace_shape), source.pace_cap
    # The leader's pace has the capped density, and the trailing lab's pace
    # must exceed it by the gap without passing the cap.
    expected = quad(
        lambda m: law.pdf(m) / law.cdf(cap) * (1 - law.cdf(m + excess) / law.cdf(cap)),
        0,
        cap - excess,
    )[0]
    assert 0 < expected < 0.5
    assert share == pytest.approx(
        expected, abs=5 * np.sqrt(expected * (1 - expected) / n)
    )


def test_no_progress_stays_at_observed_frontier_with_positive_baseline_rates():
    flat = sample(synthetic_forecast(), n_draws=20, seed=3, variant="no_progress")
    assert all(row.baseline_rate > 0 for row in flat.draws)
    assert all(
        row.rates == (0, 0) and row.pace_multipliers == (0, 0) for row in flat.draws
    )
    np.testing.assert_array_equal(values(flat), np.full((20, 13, 2), [100, 90]))


@pytest.mark.parametrize(
    "domain,scale,units", [("cyber", "eci", "ECI points"), ("bio", "bci", "BCI points")]
)
def test_domain_indices_cannot_be_relabelled_as_lab_general_eci(domain, scale, units):
    payload = synthetic_forecast().model_dump(mode="json")
    payload["coordinates"][0].update(domain_id=domain, scale_id=scale, units=units)
    with pytest.raises(ValidationError):
        PublicFrontierForecast.model_validate(payload)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p["coordinates"][1].update(lab_id="lab_a"), "unique"),
        (lambda p: p["coordinates"][1].update(scale_id="other_eci"), "one frozen"),
        (lambda p: p.update(pace_shape=0), "greater than 0"),
        (lambda p: p.update(trend_discrepancy_sd=0), "greater"),
        (lambda p: p["releases"][0].update(lab_id="unknown_lab"), "forecast labs"),
        (
            lambda p: p["releases"][0].update(available_at="2026-10-02T00:00:00Z"),
            "cannot follow",
        ),
        (
            lambda p: p["releases"].append(dict(p["releases"][0])),
            "release ID must be unique",
        ),
        (
            lambda p: p.update(
                releases=[row for row in p["releases"] if row["lab_id"] == "lab_a"]
            ),
            "anchor its path",
        ),
        (
            lambda p: p.update(trend_window_start="2026-10-02T00:00:00Z"),
            "window cannot start",
        ),
    ],
)
def test_rejects_inconsistent_forecast_inputs(mutation, match):
    payload = synthetic_forecast().model_dump(mode="json")
    mutation(payload)
    with pytest.raises(ValidationError, match=match):
        PublicFrontierForecast.model_validate(payload)


def test_fixed_rate_prior_must_be_positive_and_diffuse_prior_is_the_default():
    with pytest.raises(ValidationError, match="positive"):
        PublicRatePrior(mean=0, sd=0, rationale="Synthetic invalid fixed rate.")
    payload = synthetic_forecast().model_dump(mode="json")
    del payload["rate_prior"]
    forecast = PublicFrontierForecast.model_validate(payload)
    assert (forecast.rate_prior.mean, forecast.rate_prior.sd) == (0, 100)


def test_seed_replay_and_serialized_rate_identity_are_checked():
    source = synthetic_forecast()
    a = sample(source, n_draws=2, seed=1)
    assert a == sample(source, n_draws=2, seed=1)
    assert a != sample(source, n_draws=2, seed=2)
    assert a == PublicFrontierEnsemble.model_validate_json(a.model_dump_json())
    payload = a.model_dump(mode="json")
    payload["draws"][0]["values"][1][1] += 1
    with pytest.raises(ValidationError):
        PublicFrontierEnsemble.model_validate(payload)


def test_unsupported_loglogistic_tail_fails_without_clipping_or_replacement():
    source = changed(pace_shape=1e-300)
    with pytest.raises(ValueError, match=r"finite|support|overflow"):
        sample(source, n_draws=20, seed=3)
    flat = sample(source, n_draws=20, seed=3, variant="no_progress")
    assert all(row.rates == (0, 0) for row in flat.draws)


def test_comparison_reports_anchor_and_moment_conditions_without_mean_claims():
    result = compare_public_frontier(
        changed(pace_shape=0.8),
        period_start=START,
        period_end=END,
        n_draws=30,
        seed=9,
    )
    assert set(result["variants"]) == set(VARIANTS)
    assert result["as_of_frontier"] == {
        "labs": {"lab_a": 100, "lab_b": 90},
        "public_frontier": 100,
    }
    assert list(result["variants"]["baseline"]["checkpoints"]) == [
        "2027-01-01T00:00:00+00:00",
        "2027-07-01T00:00:00+00:00",
        "2028-01-01T00:00:00+00:00",
    ]
    assert result["moment_conditions"] == {
        "pace_moments": "finite_at_every_order_under_the_pace_cap",
        "continuous_forecast_loss_moments": "not_established",
        "reported_quantiles": "retained_finite_ensemble",
    }
    assert "mean" not in result["variants"]["baseline"]["annual_rates"]["lab_a"]


@pytest.mark.parametrize("lab_count", [1, 3])
def test_generator_admits_variable_lab_rosters(lab_count):
    payload = synthetic_forecast(uncertain=False).model_dump(mode="json")
    if lab_count == 1:
        payload["coordinates"] = payload["coordinates"][:1]
        payload["releases"] = [
            row for row in payload["releases"] if row["lab_id"] == "lab_a"
        ]
    else:
        payload["coordinates"].append({**payload["coordinates"][1], "lab_id": "lab_c"})
        payload["releases"].append(
            synthetic_release("lab_c", "c-1", "2026-06-01T00:00:00Z", 80)
        )
    source = PublicFrontierForecast.model_validate(payload)
    ensemble = sample(source, n_draws=2, seed=9)
    assert len(ensemble.coordinates) == lab_count
    assert all(len(row.rates) == lab_count for row in ensemble.draws)
    assert all(
        len(value) == lab_count for row in ensemble.draws for value in row.values
    )


def test_labs_share_one_baseline_rate_and_a_fixed_prior_stays_fixed():
    result = sample(synthetic_forecast(), n_draws=200, seed=71)
    for row in result.draws:
        np.testing.assert_allclose(
            np.asarray(row.rates) / np.asarray(row.pace_multipliers),
            row.baseline_rate,
            rtol=1e-12,
        )
    posterior = fit_public_trend(synthetic_forecast(uncertain=False))
    assert (posterior.rate_location, posterior.rate_variance) == (10, 0)
    assert posterior.labs["lab_a"].observation_count == 2


def test_backtested_pace_shape_default_is_frozen_in_forecast():
    payload = synthetic_forecast().model_dump(mode="json")
    del payload["pace_shape"]
    forecast = PublicFrontierForecast.model_validate(payload)
    assert forecast.pace_shape == 5
    assert forecast.model_dump(mode="json")["pace_shape"] == 5


def test_default_pace_cap_is_frozen_in_forecast_and_must_exceed_the_median_pace():
    payload = synthetic_forecast().model_dump(mode="json")
    assert "pace_cap" not in synthetic_forecast().model_fields_set
    assert payload["pace_cap"] == 4
    assert changed(pace_cap=2.5).digest() != synthetic_forecast().digest()
    for cap in (1, 0.5, float("inf")):
        with pytest.raises(ValidationError):
            changed(pace_cap=cap)


@pytest.mark.parametrize("sd", (0.3, 1e-12))
def test_tightly_positive_rate_samples_despite_negligible_gaussian_tail(sd):
    payload = synthetic_forecast().model_dump(mode="json")
    payload["rate_prior"] = {"mean": 12, "sd": sd, "rationale": "Synthetic tight rate."}
    payload["trend_window_start"] = payload["as_of"]
    source = PublicFrontierForecast.model_validate(payload)
    result = sample(source, n_draws=20, seed=5)
    rates = np.array([draw.baseline_rate for draw in result.draws])
    assert np.all(np.isfinite(rates))
    assert np.all(rates > 0)
    assert np.mean(rates) == pytest.approx(12, abs=0.3)


@pytest.mark.parametrize("variant", VARIANTS)
def test_assumptions_identify_the_executed_pace(variant):
    result = sample(synthetic_forecast(), n_draws=2, seed=5, variant=variant)
    assert not any("shared across all labs" in item for item in result.assumptions)
    assert any("historical score vintages" in item for item in result.assumptions)
    assert any("highest published score" in item for item in result.assumptions)
    if variant == "no_progress":
        assert any("leader cannot change" in item for item in result.assumptions)
    else:
        assert any(
            "independent pace multiplier" in item
            and "leader can change" in item
            and "median-one log-logistic law truncated at the pace cap of 4," in item
            for item in result.assumptions
        )
        assert any("every pace moment is finite" in item for item in result.assumptions)


def test_comparison_brackets_the_default_pace_shape_with_round_alternatives():
    result = compare_public_frontier(
        synthetic_forecast().model_copy(update={"pace_shape": 5}),
        period_start=START,
        period_end=END,
        n_draws=2000,
        seed=4,
    )
    sensitivity = result["pace_shape_sensitivity"]
    assert set(sensitivity) == {"lower_pace_shape", "higher_pace_shape"}
    assert sensitivity["lower_pace_shape"]["pace_shape"] == 3
    assert sensitivity["higher_pace_shape"]["pace_shape"] == 8
    end = "2028-01-01T00:00:00+00:00"

    def width(row):
        quantiles = row["checkpoints"][end]["public_frontier"]
        return quantiles["p95"] - quantiles["p05"]

    assert (
        width(sensitivity["lower_pace_shape"])
        > width(result["variants"]["baseline"])
        > width(sensitivity["higher_pace_shape"])
        > width(result["variants"]["no_progress"])
        == 0
    )
    # The lower cap of 2 is the one cap alternative, and it narrows the upper
    # tail; a forecast whose own cap is not above it has none.
    assert result["pace_cap"] == 4
    capped = result["pace_cap_sensitivity"]
    assert set(capped) == {"lower_pace_cap"}
    assert capped["lower_pace_cap"]["pace_cap"] == 2
    baseline = result["variants"]["baseline"]["checkpoints"][end]["public_frontier"]
    lower = capped["lower_pace_cap"]["checkpoints"][end]["public_frontier"]
    assert lower["p95"] < baseline["p95"]
    assert width(result["variants"]["baseline"]) > width(capped["lower_pace_cap"])
    assert (
        compare_public_frontier(
            synthetic_forecast().model_copy(update={"pace_cap": 2.0}),
            period_start=START,
            period_end=END,
            n_draws=20,
            seed=4,
        )["pace_cap_sensitivity"]
        == {}
    )
    residuals = result["record_residuals"]
    assert residuals["lab_a"]["count"] == 2
    assert residuals["lab_a"]["leave_one_out_rms"] > 0
    assert result["staleness"]["lab_b"]["years_since_as_of_record"] == pytest.approx(
        30 / 365.25
    )
    assert result["pooled_rate_kernel"]["sd"] > 0


def _calibration_forecast(discrepancy_sd):
    rng = np.random.default_rng(7)
    payload = synthetic_forecast().model_dump(mode="json")
    template = payload["coordinates"][0]
    payload.update(
        coordinates=[{**template, "lab_id": f"lab_{lab}"} for lab in range(60)],
        trend_window_start="2024-01-01T00:00:00Z",
        trend_discrepancy_sd=discrepancy_sd,
        rate_prior={"mean": 0, "sd": 100, "rationale": "Synthetic diffuse prior."},
        releases=[],
    )
    as_of = datetime(2026, 10, 1, tzinfo=UTC)
    for lab in range(60):
        for step in range(5):
            elapsed = -2.5 + 0.5 * step
            at = (as_of + timedelta(seconds=elapsed * YEAR_SECONDS)).isoformat()
            # Steps of 6 points against unit noise: every release is a record.
            value = 100 + lab + 12 * elapsed + rng.normal()
            payload["releases"].append(
                synthetic_release(f"lab_{lab}", f"r{step}", at, value, sd=0)
            )
    return PublicFrontierForecast.model_validate(payload)


def test_leave_one_out_residuals_are_calibrated_only_with_the_right_discrepancy():
    def pooled_rms(source):
        rows = record_residuals(source).values()
        return np.sqrt(
            sum(row["count"] * row["leave_one_out_rms"] ** 2 for row in rows)
            / sum(row["count"] for row in rows)
        )

    assert pooled_rms(_calibration_forecast(1)) == pytest.approx(1, abs=0.1)
    # An understated discrepancy is flagged even with five records per lab.
    assert pooled_rms(_calibration_forecast(0.6)) > 1.4


def test_staleness_reports_a_lab_left_behind_its_trend():
    releases = [
        synthetic_release("lab_a", "a-1", "2026-02-01T00:00:00Z", 94),
        synthetic_release("lab_a", "a-2", "2026-09-01T00:00:00Z", 101),
        synthetic_release("lab_b", "b-1", "2026-01-01T00:00:00Z", 85),
        synthetic_release("lab_b", "b-2", "2026-02-01T00:00:00Z", 86),
    ]
    result = staleness(synthetic_forecast(uncertain=False, releases=releases))
    assert result["lab_b"]["years_since_as_of_record"] == pytest.approx(242 / 365.25)
    # Fixed rate 10/yr: lab_b's level is its mean score carried forward at that
    # rate from its mean record time, 273 and 242 days before as-of.
    assert result["lab_b"]["trend_level_minus_as_of_value"] == pytest.approx(
        85.5 + 10 * (273 + 242) / 2 / 365.25 - 86
    )
    assert result["lab_a"]["trend_level_minus_as_of_value"] < 1
