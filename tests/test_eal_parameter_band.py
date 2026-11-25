"""The report-only EAL parameter band (``NUM-EAL-PARAMETER-BAND``).

A state's EAL with its count laws held at fixed ranks is each incident's
analytic mean times its count multiplier at that rank. The band integrates
each respondent's states by their weights at shared ranks and mixes the
respondents, with comonotone or independent ranks across rank groups; its
mean is the reported EAL.
"""

import json
import math
from dataclasses import replace
from functools import partial
from typing import Any

import pytest
from scipy.optimize import brentq
from scipy.special import ndtri

import catmodel.results.eal_parameter_band as band_module
from catmodel.contracts.contextual_risk import (
    ContextualRiskAnswer,
    ContextualRiskArtifact,
)
from catmodel.contracts.elicitation import Quantiles
from catmodel.elicitation.contextual_risk import contextual_prompts
from catmodel.hashing import canonical_json_sha256
from catmodel.results.eal_parameter_band import (
    BAND_MISMATCH_REASON,
    BAND_PROBABILITIES,
    NO_BAND,
    REWEIGHTED_BAND_REASON,
    band_replays,
    borrowed_band_reason,
    compute_parameter_band,
    reported_parameter_band,
)
from catmodel.results.execution import replay_capture, run_setup
from catmodel.results.respondent_sampling import SampleGroups, ViewWeights
from catmodel.results.results_summary import results_summary
from catmodel.setup.contextual_setup import compile_contextual_setup
from catmodel.setup.store import build_setup
from catmodel.simulation.incident import CompiledIncidentState, compile_incident_state
from catmodel.simulation.incident_moments import AnalyticEals, analytic_linear_eals
from test_support.contextual_risk import (
    contextual_answer,
    contextual_case,
    contextual_instrument,
    matched_chain_answer,
)
from test_support.count_law_oracle import CountLawOracle
from test_support.results import available_metric, present, reduced_arm
from test_support.study import _package, _plan, _study

# Each view's loss per qualified incident in the fixture answer.
PER_COUNT = {"welfare": 100.0, "gross": 120.0, "lab_loss::lab": 20.0}
ANNUAL_BUSINESS_LOSS = 70.0


def _quantile(law: Quantiles, probability: float) -> float:
    """The count law's quantile from the independent oracle."""
    return CountLawOracle(law).quantile(probability)


def _curve(view: str, count: float) -> float:
    """A view's EAL at a fixed expected count: the episode terms per count,
    plus the lab's fixture business loss once in a year with an incident."""
    # Gross and the lab view count the business loss; welfare does not.
    annual = ANNUAL_BUSINESS_LOSS * -math.expm1(-count) if view != "welfare" else 0
    return PER_COUNT[view] * count + annual


VIEWS = tuple(PER_COUNT)
# Right-skewed: ln 10 from p5 to the median, ln 16 from the median to p95.
WIDE = Quantiles(p5=0.05, p25=0.2, p50=0.5, p75=1.5, p95=8)
LEVELS = tuple(str(p) for p in BAND_PROBABILITIES)
# Small budgets keep the tests fast; the half-widths scale with them.
SAMPLES = 2**15
GRID = 2**14 + 1


def _artifact(instrument, rate):
    cases = []
    for prompt in contextual_prompts(instrument):
        payload = contextual_answer().model_dump(mode="json")
        payload["rate"] = rate.model_dump()
        answer = ContextualRiskAnswer.model_validate(payload)
        # The fixture chain's exposure is one unit, so its rate is its count.
        chain = matched_chain_answer(prompt.node, answer, 1).model_copy(
            update={"opportunity_rate": rate}
        )
        cases.append(contextual_case(instrument, prompt, answer, chain=chain))
    return ContextualRiskArtifact(
        instrument_sha256=instrument.digest(),
        evidence_status="fixture",
        cases=tuple(cases),
    )


def _setup(rate=WIDE, seed=7, years=512):
    instrument = contextual_instrument()
    return compile_contextual_setup(
        instrument,
        _artifact(instrument, rate),
        setup_id="band",
        years_per_state=years,
        seed=seed,
    )


def _state(rate=WIDE) -> CompiledIncidentState:
    setup = _setup(rate)
    state = next(state for state in setup.joint_states if state.weight > 0)
    return compile_incident_state(setup, state.state_id)


def _metrics(states: dict[str, CompiledIncidentState], weights: dict[str, float]):
    """Reported metrics as the reduction states them for analytic EALs."""
    eals = {key: analytic_linear_eals(state, VIEWS) for key, state in states.items()}
    return {
        view: {
            "status": "available",
            "expected_annual_loss": math.fsum(
                weights[key] * eal.values[view] for key, eal in eals.items()
            ),
            "eal_estimator": "native_analytic_mean",
            "eal_precision": {
                "integration_error_bound": math.fsum(
                    weights[key] * eal.error_bounds[view] for key, eal in eals.items()
                )
            },
            "statistics": {
                "expected_annual_loss": {"status": "available", "reason": None}
            },
        }
        for view in VIEWS
    }


def _captured(states, weights, groups=None, *, seed=11) -> dict[str, Any]:
    return compute_parameter_band(
        states,
        weights,
        {key: key for key in states} if groups is None else groups,
        VIEWS,
        seed=seed,
        batches=16,
        tolerance=0.05,
        samples=SAMPLES,
        grid_points=GRID,
    )


def _band(states, weights, groups=None, *, seed=11, metrics=None) -> dict[str, Any]:
    return reported_parameter_band(
        _captured(states, weights, groups, seed=seed),
        _metrics(states, weights) if metrics is None else metrics,
        reweighted=False,
    )


def _ranked(state, group, origin, factor=1.0):
    (incident,) = state.incidents
    rank = present(incident.parameter_rank)
    return replace(
        incident,
        origin_id=origin,
        annual_rate=incident.annual_rate * factor,
        parameter_rank=rank.model_copy(update={"group_id": group}),
    )


def _fixed(state, origin, factor):
    (incident,) = state.incidents
    return replace(
        incident,
        origin_id=origin,
        annual_rate=incident.annual_rate * factor,
        parameter_rank=None,
    )


def _within(row, case, level, value, factor=1.0):
    """``value`` lies within the case's error bound or half-width."""
    estimate = row[case]["quantiles"][level]
    if case == "comonotone":
        error = row[case]["error_bounds"][level]
    else:
        error = row[case]["convergence"][level]["half_width"]
    return abs(estimate - value) <= factor * error + 1e-12 * abs(value)


def test_a_run_captures_the_band_and_its_replay_reads_it(tmp_path):
    setup = _setup(years=256)
    package = build_setup(destination=tmp_path / "setup", setup=setup, code_commit="x")
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    band = report["sensitivity_summary"]["eal_parameter_band"]
    assert band["status"] == "available"
    independent = band["method"]["independent"]
    assert independent["seed"] == setup.execution.seed
    assert independent["samples_per_epistemic_group"] == 2**17
    assert band["epistemic_groups"] == band["states"] == 1
    state = next(state for state in setup.joint_states if state.weight > 0)
    (incident_law,) = state.laws
    assert incident_law.parameter_rank is not None
    count_law = incident_law.parameter_rank.count
    metrics = reduced_arm(report["arms"]["reference"])["metrics"]
    for view in PER_COUNT:
        row = band["views"][view]
        assert row["status"] == "available"
        eal = available_metric(metrics[view])["expected_annual_loss"]
        assert row["expected_annual_loss"] == eal
        assert "mean" not in row
        # One state and one rank group: both cases are the compiled pooled
        # count law itself, read through the view's ``_curve``.
        for probability, level in zip(BAND_PROBABILITIES, LEVELS, strict=True):
            exact = _curve(view, _quantile(count_law, probability))
            assert _within(row, "comonotone", level, exact)
            assert _within(row, "independent", level, exact, factor=2)
        (only,) = row["by_rank_group"].values()
        assert only["quantiles"] == row["comonotone"]["quantiles"]
        assert only["expected_annual_loss"] == pytest.approx(eal, rel=1e-12)
        assert row["independent"]["sample_mean"] == pytest.approx(eal, rel=0.05)
    # Replay rederives the captured band from the laws and reports it unchanged.
    replay = replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    assert replay["sensitivity_summary"]["eal_parameter_band"] == band
    markdown, rows = results_summary(report)
    assert "## EAL parameter band" in markdown
    assert markdown.index("## EAL parameter band") < markdown.index(
        "## Tail value at risk"
    )
    band_rows = [row for row in rows if row["section"] == "eal_parameter_band"]
    assert {
        (row["subject"], float(str(row["level"])))
        for row in band_rows
        if row["statistic"] == "eal_band_quantile"
    } == {
        (case, p) for case in ("independent", "comonotone") for p in BAND_PROBABILITIES
    }
    # Each quantile's precision is its own row: the independent ranks'
    # Monte Carlo half-width and the comonotone ranks' error bound.
    assert {row["statistic"] for row in band_rows} == {
        "eal_band_quantile",
        "eal_band_half_width",
        "eal_band_error_bound",
    }


def test_a_fixed_rate_collapses_the_band_to_its_eal():
    state = _state()
    (incident,) = state.incidents
    constant = replace(
        incident,
        parameter_rank=present(incident.parameter_rank).model_copy(
            update={"count": Quantiles(p5=2, p25=2, p50=2, p75=2, p95=2)}
        ),
    )
    fixed = replace(state, incidents=(_fixed(state, "fixed", 1.0),))
    flat = replace(state, incidents=(constant,))
    for compiled, ranked in ((fixed, 0), (flat, 1)):
        band = _band({"z": compiled}, {"z": 1.0})
        assert band["ranked_laws"] == ranked
        for view in VIEWS:
            row = band["views"][view]
            eal = row["expected_annual_loss"]
            for case in ("comonotone", "independent"):
                assert row[case]["quantiles"] == dict.fromkeys(LEVELS, eal)
            assert row["comonotone"]["error_bounds"] == dict.fromkeys(LEVELS, 0.0)
            assert all(
                cell["half_width"] == 0
                for cell in row["independent"]["convergence"].values()
            )
            assert row["fixed_rate_expected_annual_loss"] == (eal if not ranked else 0)


def test_comonotone_ranks_spread_the_band_more_than_independent_ones():
    """Two rank groups with the same wide law: comonotone ranks double one
    draw, 2 c Q(p) exactly, while independent ranks average two draws. Here
    the quantiles order; convex order itself only orders the tail means."""
    state = _state()
    two = replace(
        state,
        incidents=(_ranked(state, "first", "a"), _ranked(state, "second", "b")),
    )
    band = _band({"z": two}, {"z": 1.0})
    assert band["rank_groups"] == 2
    for view in PER_COUNT:
        row = band["views"][view]
        for probability, level in zip(BAND_PROBABILITIES, LEVELS, strict=True):
            exact = 2 * _curve(view, _quantile(WIDE, probability))
            assert _within(row, "comonotone", level, exact)
        comonotone = row["comonotone"]["quantiles"]
        independent = row["independent"]["quantiles"]
        error = {
            level: 2 * row["independent"]["convergence"][level]["half_width"]
            for level in LEVELS
        }
        assert comonotone["0.95"] > independent["0.95"] + error["0.95"]
        assert comonotone["0.05"] < independent["0.05"] - error["0.05"]
        # Each group alone is one rank's law.
        for group in ("first", "second"):
            for probability, level in zip(BAND_PROBABILITIES, LEVELS, strict=True):
                assert _within(
                    {"comonotone": row["by_rank_group"][group]},
                    "comonotone",
                    level,
                    _curve(view, _quantile(WIDE, probability)),
                )
        assert row["expected_annual_loss"] == pytest.approx(
            math.fsum(
                group["expected_annual_loss"] for group in row["by_rank_group"].values()
            )
        )


@pytest.mark.parametrize(("spread", "tail_scale"), [(1.5, 1.0), (1.0, 1.5), (1.5, 1.5)])
def test_a_spread_or_tail_width_law_reads_its_rank_through_its_own_quantiles(
    spread, tail_scale
):
    state = _state()
    (incident,) = state.incidents
    law = present(incident.parameter_rank).model_copy(
        update={"rank_spread": spread, "epistemic_tail_scale": tail_scale}
    )
    compiled = replace(state, incidents=(replace(incident, parameter_rank=law),))
    band = _band({"z": compiled}, {"z": 1.0})
    oracle = CountLawOracle(
        law.count, tail_scale=tail_scale, spread=spread, tail_form=law.tail_form
    )
    for view in VIEWS:
        row = band["views"][view]
        for probability, level in zip(BAND_PROBABILITIES, LEVELS, strict=True):
            score = float(ndtri(probability))
            count = incident.annual_rate * oracle.at_score(score) / oracle.mean()
            exact = _curve(view, count)
            assert _within(row, "comonotone", level, exact)
            assert _within(row, "independent", level, exact, factor=2)


def test_the_band_mixes_states_by_their_weights():
    """Three fixed-rate states with EALs in the ratio 1 : 10 : 100 at weights
    0.04, 0.5 and 0.46: p5 and the median are the middle state's EAL and p95
    the largest, exactly; equal weights would put p5 at the smallest."""
    state = _state()
    factors = {"low": 0.01, "middle": 0.1, "high": 1.0}
    states = {
        key: replace(state, state_id=key, incidents=(_fixed(state, "fixed", factor),))
        for key, factor in factors.items()
    }
    weights = {"low": 0.04, "middle": 0.5, "high": 0.46}
    band = _band(states, weights)
    for view in VIEWS:
        row = band["views"][view]
        eal = {
            key: analytic_linear_eals(compiled, (view,)).values[view]
            for key, compiled in states.items()
        }
        assert row["expected_annual_loss"] == pytest.approx(
            math.fsum(weights[key] * value for key, value in eal.items())
        )
        expected = {"0.05": eal["middle"], "0.5": eal["middle"], "0.95": eal["high"]}
        for case in ("comonotone", "independent"):
            assert row[case]["quantiles"] == pytest.approx(expected)
        assert row["comonotone"]["error_bounds"] == dict.fromkeys(LEVELS, 0.0)


def test_each_respondent_integrates_its_nodes_at_shared_ranks():
    """Respondent A (weight 0.55, two nodes) has small fixed rates; B (0.25,
    two nodes at weights 0.15 and 0.10) a ranked law in group ``shared`` at
    two and four times the base rate; C (0.20, one node) ``shared`` at three
    times and ``only``, a group no other state has, at five times. Each node
    of B reads the same rank, so B's EAL is 2.8 times the base curve."""
    state = _state()
    states = {
        "a1": replace(state, state_id="a1", incidents=(_fixed(state, "f1", 0.01),)),
        "a2": replace(state, state_id="a2", incidents=(_fixed(state, "f2", 0.01),)),
        "b1": replace(
            state, state_id="b1", incidents=(_ranked(state, "shared", "s1", 2),)
        ),
        "b2": replace(
            state, state_id="b2", incidents=(_ranked(state, "shared", "s2", 4),)
        ),
        "c1": replace(
            state,
            state_id="c1",
            incidents=(
                _ranked(state, "shared", "s3", 3),
                _ranked(state, "only", "o1", 5),
            ),
        ),
    }
    weights = {"a1": 0.33, "a2": 0.22, "b1": 0.15, "b2": 0.10, "c1": 0.20}
    groups = {"a1": "A", "a2": "A", "b1": "B", "b2": "B", "c1": "C"}
    band = _band(states, weights, groups)
    assert (band["epistemic_groups"], band["states"]) == (3, 5)
    row = band["views"]["welfare"]
    base = analytic_linear_eals(
        replace(state, incidents=(_ranked(state, "shared", "x"),)), ("welfare",)
    ).values["welfare"]
    fixed = 0.01 * base
    # Group means weight every state, and ``only`` sits in C alone.
    shared, only = row["by_rank_group"]["shared"], row["by_rank_group"]["only"]
    assert shared["expected_annual_loss"] == pytest.approx(
        (0.15 * 2 + 0.10 * 4 + 0.20 * 3) * base
    )
    assert only["expected_annual_loss"] == pytest.approx(0.20 * 5 * base)
    # Respondents without ``only`` are an atom at zero of weight 0.8.
    assert only["quantiles"]["0.05"] == only["quantiles"]["0.5"] == 0
    curve = base / CountLawOracle(WIDE).mean()
    assert (
        abs(only["quantiles"]["0.95"] - 5 * curve * _quantile(WIDE, 0.75))
        <= only["error_bounds"]["0.95"]
    )
    # A's constant holds the lower half of the mixture in every batch, so
    # those half-widths are zero; equal respondent weights would not.
    for case in ("comonotone", "independent"):
        assert row[case]["quantiles"]["0.05"] == pytest.approx(fixed)
        assert row[case]["quantiles"]["0.5"] == pytest.approx(fixed)
    for level in ("0.05", "0.5"):
        assert row["independent"]["convergence"][level]["half_width"] == 0

    # With one rank for every group and node, B is 2.8 and C 8 times the
    # base curve, so the mixture's distribution function at its p95 is 0.95.
    def rank(value, scale):
        return brentq(
            lambda u: curve * scale * _quantile(WIDE, u) - value,
            1e-12,
            1 - 1e-12,
            xtol=1e-14,
        )

    p95 = row["comonotone"]["quantiles"]["0.95"]
    level = 0.55 + 0.25 * rank(p95, 2.8) + 0.20 * rank(p95, 8.0)
    assert level == pytest.approx(0.95, abs=1e-4)


def test_changing_the_seed_moves_only_the_independent_band():
    state = _state()
    two = replace(
        state,
        incidents=(_ranked(state, "first", "a"), _ranked(state, "second", "b")),
    )
    first, second = (_band({"z": two}, {"z": 1.0}, seed=seed) for seed in (1, 2))
    assert first["method"]["independent"]["seed"] == 1
    assert second["method"]["independent"]["seed"] == 2
    for view in VIEWS:
        one, other = first["views"][view], second["views"][view]
        assert one["comonotone"] == other["comonotone"]
        assert one["by_rank_group"] == other["by_rank_group"]
        assert one["independent"]["quantiles"] != other["independent"]["quantiles"]
        for level in LEVELS:
            difference = abs(
                one["independent"]["quantiles"][level]
                - other["independent"]["quantiles"][level]
            )
            assert difference <= (
                one["independent"]["convergence"][level]["half_width"]
                + other["independent"]["convergence"][level]["half_width"]
            )


def test_the_band_follows_the_eal_availability_and_never_raises():
    state = _state()
    states, weights = {"z": state}, {"z": 1.0}
    captured = _captured(states, weights)
    metrics = _metrics(states, weights)
    metrics["gross"]["statistics"]["expected_annual_loss"] = {
        "status": "withheld",
        "reason": "The positive target mean is below the numeric output range.",
    }
    metrics["lab_loss::lab"]["eal_estimator"] = "normalized_empirical_mean"
    band = reported_parameter_band(captured, metrics, reweighted=False)
    assert band["status"] == "available"
    assert band["views"]["gross"] == {
        "status": "unavailable",
        "reason": "The positive target mean is below the numeric output range.",
    }
    assert band["views"]["lab_loss::lab"]["status"] == "unavailable"
    assert band["views"]["welfare"]["status"] == "available"
    only = {"gross": metrics["gross"]}
    assert reported_parameter_band(captured, only, reweighted=False)["status"] == (
        "unavailable"
    )
    reweighted = reported_parameter_band(captured, metrics, reweighted=True)
    assert reweighted["status"] == "unavailable"
    assert reweighted["reason"] == REWEIGHTED_BAND_REASON
    # The mean ties as replay ties a numerically integrated mean: within the
    # two certified bounds. A reported EAL 1.5 bounds away, each mean with
    # bound b, still ties; one 3 bounds away leaves the view unavailable.
    welfare = captured["views"]["welfare"]
    bound = 1e-4 * welfare["mean"]
    welfare["integration_error_bound"] = bound
    for offset, status in ((1.5, "available"), (3.0, "unavailable")):
        metrics = _metrics(states, weights)
        metrics["welfare"]["expected_annual_loss"] += offset * bound
        metrics["welfare"]["eal_precision"]["integration_error_bound"] = bound
        row = reported_parameter_band(captured, metrics, reweighted=False)["views"][
            "welfare"
        ]
        assert row["status"] == status
        if status == "unavailable":
            assert "certified integration bounds" in row["reason"]


def test_no_band_is_reported_for_a_view_that_weights_a_borrowing_state():
    """Two respondents at a half each, the second reading the first's answers
    to the only leaf, so both hold one set of laws. Their band at one rank
    for every group is exactly the band of the first respondent alone: the
    mixture shows no spread between them, whatever the second model would
    have answered. Results report no band
    for a view in which the borrowing state has weight, and the same band as
    before where no state borrows or the borrowing state has no weight."""
    state = _state()
    states = {"own": state, "borrowing": state}
    weights = {"own": 0.5, "borrowing": 0.5}
    captured, metrics = _captured(states, weights), _metrics(states, weights)
    plain = reported_parameter_band(captured, metrics, reweighted=False)
    alone = _band({"own": state}, {"own": 1.0})
    assert plain["status"] == alone["status"] == "available"
    for view in VIEWS:
        assert plain["views"][view]["comonotone"]["quantiles"] == pytest.approx(
            alone["views"][view]["comonotone"]["quantiles"], rel=1e-12
        )

    sampling = {"own": "node:donor", "borrowing": "node:borrower"}
    borrowers = {"borrowing": ("state-b", ("leaf",))}
    reason = (
        "State 'state-b' reads a donor's answers to leaf 'leaf' "
        "(POOL-UNANSWERED-SCENARIOS), so the respondents the band mixes are not "
        "independent and no band is reported."
    )
    assert reason.endswith(NO_BAND)

    def reported(groups, by_view, band=captured, values=metrics):
        return reported_parameter_band(
            band,
            values,
            reweighted=False,
            borrowed=partial(borrowed_band_reason, groups, by_view),
        )

    # No state borrows: the record is the one reported before.
    assert reported(SampleGroups(sampling), ViewWeights.same(weights)) == plain
    # The borrowing state has weight in every view: none has a band, and
    # the record's reason is the views'. It keeps what it was computed from.
    withheld = reported(SampleGroups(sampling, borrowers), ViewWeights.same(weights))
    assert withheld["status"] == "unavailable"
    assert withheld["reason"] == reason
    assert withheld["views"] == {
        view: {"status": "unavailable", "reason": reason} for view in VIEWS
    }
    kept = withheld.keys() - {"status", "reason", "views"}
    assert kept == plain.keys() - {"status", "views"}
    assert all(withheld[key] == plain[key] for key in kept)
    # A view that is unavailable for its own reason keeps that reason.
    uneven = _metrics(states, weights)
    uneven["gross"]["eal_estimator"] = "normalized_empirical_mean"
    own = reported(
        SampleGroups(sampling, borrowers), ViewWeights.same(weights), values=uneven
    )
    assert own["reason"] == reason
    assert "not the analytic mean" in own["views"]["gross"]["reason"]
    # At zero weight the borrowing state withholds nothing.
    donor = {"own": 1.0, "borrowing": 0.0}
    at_donor = _captured(states, donor), _metrics(states, donor)
    assert reported(
        SampleGroups(sampling, borrowers), ViewWeights.same(donor), *at_donor
    ) == reported_parameter_band(*at_donor, reweighted=False)
    # The test is each view's own: where the borrowing state is left out of
    # the views that count the lab's loss, those views weigh the donor alone
    # and keep their bands, and welfare, which weighs both, has none.
    by_view = ViewWeights(weights, donor)
    apart = compute_parameter_band(
        states,
        weights,
        {key: key for key in states},
        VIEWS,
        seed=11,
        batches=16,
        tolerance=0.05,
        samples=SAMPLES,
        grid_points=GRID,
        lab_loss_weights={"own": 1.0},
    )
    mixed = {**_metrics(states, donor), "welfare": metrics["welfare"]}
    partly = reported(SampleGroups(sampling, borrowers), by_view, apart, mixed)
    assert partly["status"] == "available"
    assert partly["views"]["welfare"] == {"status": "unavailable", "reason": reason}
    for view in VIEWS:
        if view != "welfare":
            assert partly["views"][view]["status"] == "available"
            assert partly["views"][view]["states"] == 1


def test_a_negative_or_missing_mean_leaves_its_view_without_a_band(monkeypatch):
    original = band_module.analytic_linear_eals

    def shifted(compiled, views, **options):
        result = original(compiled, views, **options)
        values = {view: value for view, value in result.values.items()}
        values["gross"] = -1.0
        values.pop("lab_loss::lab")
        return AnalyticEals(values, {view: 0.0 for view in values})

    monkeypatch.setattr(band_module, "analytic_linear_eals", shifted)
    state = _state()
    captured = _captured({"z": state}, {"z": 1.0})
    assert captured["views"]["gross"]["status"] == "unavailable"
    assert "negative" in captured["views"]["gross"]["reason"]
    assert captured["views"]["lab_loss::lab"]["status"] == "unavailable"
    assert captured["views"]["welfare"]["status"] == "available"


def test_a_replayed_band_agrees_within_its_bounds_and_tolerance():
    state = _state()
    two = replace(
        state,
        incidents=(_ranked(state, "first", "a"), _ranked(state, "second", "b")),
    )
    captured = _captured({"z": two}, {"z": 1.0})
    assert band_replays(captured, _captured({"z": two}, {"z": 1.0}))

    def changed(path, factor):
        copy = _captured({"z": two}, {"z": 1.0})
        row = copy["views"]["welfare"]
        for key in path[:-1]:
            row = row[key]
        row[path[-1]] *= factor
        return copy

    comonotone = captured["views"]["welfare"]["comonotone"]
    within = 1 + comonotone["error_bounds"]["0.95"] / comonotone["quantiles"]["0.95"]
    assert band_replays(changed(("comonotone", "quantiles", "0.95"), within), captured)
    assert not band_replays(
        changed(("comonotone", "quantiles", "0.95"), 1 + 3 * (within - 1)), captured
    )
    assert not band_replays(
        changed(("independent", "quantiles", "0.5"), 1 + 1e-6), captured
    )
    assert not band_replays(_captured({"z": two}, {"z": 1.0}, seed=12), captured)


def test_a_capped_lab_mean_replays_within_twice_the_quadrature_tolerance(
    monkeypatch,
):
    """A capped lab mean is certified within 0.1% and another platform's
    refinement can land anywhere inside it, as the per-state replay accepts.
    A shift of 5e-5 with a bound of 1e-4 replays; one of 5e-3 does not, and
    a closed-form view keeps its 1e-9 tolerance."""
    original = band_module.analytic_linear_eals

    def platform(lab_factor, welfare_factor=1.0):
        def evaluate(compiled, views, **options):
            result = original(compiled, views, **options)
            values, bounds = dict(result.values), dict(result.error_bounds)
            if "lab_loss::lab" in values:
                value = values["lab_loss::lab"]
                bounds["lab_loss::lab"] = 1e-4 * value
                values["lab_loss::lab"] = value * lab_factor
            if "welfare" in values:
                values["welfare"] *= welfare_factor
            return AnalyticEals(values, bounds)

        return evaluate

    state = _state()
    two = replace(
        state,
        incidents=(_ranked(state, "first", "a"), _ranked(state, "second", "b")),
    )

    def captured(*factors):
        monkeypatch.setattr(band_module, "analytic_linear_eals", platform(*factors))
        return _captured({"z": two}, {"z": 1.0})

    reference = captured(1.0)
    assert reference["views"]["lab_loss::lab"]["integration_error_bound"] > 0
    assert reference["views"]["welfare"]["integration_error_bound"] == 0
    assert band_replays(reference, captured(1 + 5e-5))
    assert not band_replays(reference, captured(1 + 5e-3))
    assert not band_replays(reference, captured(1.0, 1 + 5e-5))
    # A half-width moves with the batch quantiles, so its slack is a share of
    # the quantile, not of the half-width itself.
    for share, replays in ((1e-3, True), (3e-3, False)):
        moved = captured(1.0)
        row = moved["views"]["lab_loss::lab"]["independent"]
        row["convergence"]["0.95"]["half_width"] += share * row["quantiles"]["0.95"]
        assert band_replays(reference, moved) is replays


def _stipulated_run(tmp_path):
    package = _package(tmp_path, _study(rate=171.6 / 350), _plan(50))
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    assert report["sensitivity_summary"]["eal_parameter_band"]["status"] == (
        "available"
    )
    return package, report


def _replay(package, tmp_path, name, **options):
    return replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / f"{name}.json",
        code_commit="fixture",
        **options,
    )


def test_a_mismatched_band_is_withheld_and_the_capture_still_replays(
    tmp_path, monkeypatch
):
    package, report = _stipulated_run(tmp_path)
    baseline = _replay(package, tmp_path, "baseline")
    assert (
        baseline["sensitivity_summary"]["eal_parameter_band"]
        == (report["sensitivity_summary"]["eal_parameter_band"])
    )
    # Code with another band constant cannot reproduce the captured method.
    monkeypatch.setattr(band_module, "INDEPENDENT_SAMPLES", 2**10)
    changed = _replay(package, tmp_path, "changed")
    monkeypatch.undo()
    # A captured value moved outside its tolerance is withheld, never shown.
    path = tmp_path / "run" / "capture" / "run-capture.json"
    record = json.loads(path.read_text())
    record["eal_parameter_band"]["views"]["gross"]["comonotone"]["quantiles"][
        "0.95"
    ] *= 1.5
    record["content_sha256"] = canonical_json_sha256(
        {key: value for key, value in record.items() if key != "content_sha256"}
    )
    path.write_text(json.dumps(record))
    tampered = _replay(package, tmp_path, "tampered")
    for result in (changed, tampered):
        assert result["sensitivity_summary"]["eal_parameter_band"] == {
            "status": "unavailable",
            "reason": BAND_MISMATCH_REASON,
            "interpretation": band_module.BAND_INTERPRETATION,
        }
        # Everything the capture's draws support is reported unchanged.
        assert result["arms"] == baseline["arms"]
        assert result["tail_gate"] == baseline["tail_gate"]


def test_a_reweighted_replay_withholds_the_band(tmp_path):
    package, _ = _stipulated_run(tmp_path)
    reweighted = _replay(
        package, tmp_path, "reweighted", state_weights={"low": 0.5, "high": 0.5}
    )
    band = reweighted["sensitivity_summary"]["eal_parameter_band"]
    assert band["status"] == "unavailable"
    assert band["reason"] == REWEIGHTED_BAND_REASON
    assert "views" not in band
