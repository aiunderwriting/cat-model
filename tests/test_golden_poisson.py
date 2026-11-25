"""The golden Poisson setup reproduces its hand derivation through the public CLI.

``config/setups/golden-poisson/README.md`` derives every expected value by hand:
Poisson ``lambda = 0.5`` incidents, ground-up loss 10 or 100 with probabilities
0.8 and 0.2, and a policy with per-occurrence retention 20 and limit 50. This
test recomputes those values independently of the engine (closed forms and a
Panjer recursion), then runs ``setup build``, ``run``, ``results replay`` and
``results export`` and compares. ``summary.md`` of the replayed results, the
README quickstart's summary, is compared with its reviewed snapshot,
``tests/fixtures/summary/golden-poisson-summary.md``, and the run's and the
replay's results files with their JSON Schema documents.

The analytic gross EAL is deterministic. Empirical quantities use five standard
errors of their estimator under the exact law at the planned simulation count.
Empirical VaR must equal the hand atom: the test first checks that each stated
level lies at least five standard errors inside its exact CDF jump.

Each statistic's status is predicted from the same hand laws: the tail gate
(``NUM-TAIL-GATE``) makes a statistic available when every batch has at least
the floor's years at or above the VaR (with a loss, for the probability) and
its batch-means half-width is within the tolerance, so the prediction uses the
batches' expected support and the half-width the exact law implies. A VaR or
TVaR of gross, whose EAL is analytic, also needs the years' mean within five
batch-means standard errors of 14; under the exact law that z is about
t-distributed with 15 degrees of freedom, so it fails with probability 1.6e-4.
"""

import copy
import json
import math
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from click.testing import CliRunner
from scipy.stats import t as student_t

from catmodel.cli import cli
from catmodel.contracts.results import RESULTS_SCHEMA_VERSION, reason_code
from catmodel.contracts.results_schema import DIAGNOSTICS_SCHEMA_FILE
from catmodel.resources import config_path
from catmodel.results.results_file import diagnostics_path, load_results
from test_support.results_schema import payload_violations, schema_violations

RATE = 0.5
SEVERITY = {10.0: 0.8, 100.0: 0.2}
RETENTION, LIMIT = 20.0, 50.0
LEVELS = (0.95, 0.98)
LATTICE = 10.0
LATTICE_POINTS = 200  # Annual mass above 1,990 is below 1e-38.
SUMMARY_SNAPSHOT = (
    Path(__file__).parent / "fixtures" / "summary" / "golden-poisson-summary.md"
)


def _panjer(severity: dict[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Compound Poisson annual pmf on the loss lattice (Panjer, a = 0, b = RATE)."""
    jump = np.zeros(LATTICE_POINTS)
    for loss, probability in severity.items():
        jump[round(loss / LATTICE)] += probability
    mass = np.zeros(LATTICE_POINTS)
    mass[0] = math.exp(-RATE * (1 - jump[0]))
    for k in range(1, LATTICE_POINTS):
        j = np.arange(1, k + 1)
        mass[k] = RATE / k * np.sum(j * jump[j] * mass[k - j])
    return LATTICE * np.arange(LATTICE_POINTS), mass


def _occurrence_maximum(severity: dict[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Annual maximum: P(M <= x) = exp(-RATE P(X > x)); an empty year has M = 0."""
    support = np.array(sorted({0.0, *severity}))
    cdf = np.array(
        [
            math.exp(-RATE * sum(p for loss, p in severity.items() if loss > x))
            for x in support
        ]
    )
    return support, np.diff(cdf, prepend=0.0)


def _payout(loss: float) -> float:
    return min(max(loss - RETENTION, 0.0), LIMIT)


def _hand_laws() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    payout: dict[float, float] = {}
    for loss, probability in SEVERITY.items():
        payout[_payout(loss)] = payout.get(_payout(loss), 0.0) + probability
    return {
        "gross": _panjer(SEVERITY),
        "oep_gross": _occurrence_maximum(SEVERITY),
        "financial": _panjer(payout),
        "oep_financial": _occurrence_maximum(payout),
    }


def _var_tvar(support, mass, level):
    """Left-inverse VaR and expected shortfall, including the VaR atom's share."""
    cdf = np.cumsum(mass)
    index = int(np.searchsorted(cdf, level))
    var = float(support[index])
    above = support > var
    tvar = (np.dot(support[above], mass[above]) + (cdf[index] - level) * var) / (
        1 - level
    )
    below = float(cdf[index - 1]) if index else 0.0
    return var, float(tvar), below, float(cdf[index])


def _predicted_status(
    support_mass: float, standard_deviation: float, value: float, controls, n: int
) -> str | None:
    """The tail gate's verdict predicted from the exact law, or None when too
    close to call.

    ``support_mass`` is the probability of a supporting year and
    ``standard_deviation`` the exact standard deviation of the statistic's
    estimator from one year, so a batch of ``n / batches`` years has expected
    support ``support_mass * n / batches`` and the half-width is about
    ``t * standard_deviation / sqrt(n)``. The sparsest batch must clear the
    floor by three binomial standard deviations. The half-width is itself
    estimated from the batches, with a relative standard error of about 18%
    at 16 batches, so a predicted relative half-width within 25% of the
    tolerance is not called.
    """
    batches, tolerance = controls["batches"], controls["tolerance"]
    floor = controls["minimum_batch_exceedances"]
    support = support_mass * n / batches
    if support - 3 * math.sqrt(support) < floor:
        return None
    critical = float(student_t.ppf(0.975, batches - 1))
    relative = (
        0.0
        if standard_deviation == 0
        else critical * standard_deviation / (math.sqrt(n) * value)
    )
    if relative < tolerance / 1.25:
        return "available"
    if relative > 1.25 * tolerance:
        return "withheld"
    return None


def _deviation(values, mass):
    return math.sqrt(
        max(float(np.dot(values**2, mass) - np.dot(values, mass) ** 2), 0.0)
    )


def _control_variate(support, mass, var, exact):
    """The exact law's control-variate coefficient and the standard deviation
    of the TVaR estimator's one-year term, before dividing by 1 - level.

    Without an exact mean the coefficient is zero and the term is the excess
    over the VaR, ``(L - VaR)+``. With one (and a finite variance, as here) it
    is ``Cov((L - VaR)+, L) / Var(L)`` clipped to [0, 1], and the term is the
    excess less the coefficient times ``L - E[L]``.
    """
    excess = np.maximum(support - var, 0.0)
    if not exact:
        return 0.0, _deviation(excess, mass)
    mean = float(np.dot(support, mass))
    centered = support - mean
    covariance = float(np.dot((excess - np.dot(excess, mass)) * centered, mass))
    beta = min(max(covariance / float(np.dot(centered**2, mass)), 0.0), 1.0)
    return beta, _deviation(excess - beta * centered, mass)


def _coefficient_error(support, mass, var, beta, n):
    """The standard error of the control-variate coefficient estimated from
    ``n`` years.

    The regression residual, the excess less its mean and ``beta`` times
    ``L - E[L]``, is a function of ``L`` and largest in the tail, where
    ``L - E[L]`` is too, so the slope's error is the heteroscedasticity-
    consistent (sandwich) one, ``sqrt(E[residual² (L - E[L])²] / n) / Var(L)``;
    on this law it is about six times the error a constant residual variance
    would give.
    """
    mean = float(np.dot(support, mass))
    centered = support - mean
    excess = np.maximum(support - var, 0.0)
    residual = excess - float(np.dot(excess, mass)) - beta * centered
    variance = float(np.dot(centered**2, mass))
    return math.sqrt(float(np.dot(residual**2 * centered**2, mass)) / n) / variance


def _predicted_tail_statuses(support, mass, level, var, tvar, controls, n, exact):
    """The predicted VaR and TVaR statuses at ``level`` (``_predicted_status``).

    The VaR is the hand atom in every batch when the level lies at least two
    batch standard errors inside its CDF jump, so its half-width is about
    zero. The control-variate TVaR's error is that of its one-year term
    (``_control_variate``) over ``1 - level``.
    """
    batches = controls["batches"]
    cdf = np.cumsum(mass)
    index = int(np.searchsorted(cdf, level))
    below = float(cdf[index - 1]) if index else 0.0
    margin = min(level - below, float(cdf[index]) - level)
    support_mass = float(np.sum(mass[support >= var]))
    pinned = margin >= 2 * math.sqrt(level * (1 - level) * batches / n)
    statuses = {
        "var": _predicted_status(support_mass, 0.0, var, controls, n)
        if pinned
        else None
    }

    _, spread = _control_variate(support, mass, var, exact)
    statuses["tvar"] = _predicted_status(
        support_mass, spread / (1 - level), tvar, controls, n
    )
    return statuses


class GoldenOutputs(NamedTuple):
    results: dict
    export: dict
    simulations: int
    summary: Path
    # The run's and the replay's results files.
    results_files: tuple[Path, Path]


@pytest.fixture(scope="module")
def golden_outputs(tmp_path_factory) -> GoldenOutputs:
    root = tmp_path_factory.mktemp("golden-poisson")
    source = config_path("setups", "golden-poisson")
    runner = CliRunner()

    def invoke(*arguments):
        # Outside a git checkout, such as from an sdist, the commands record
        # the installed version instead of a commit.
        result = runner.invoke(cli, [str(argument) for argument in arguments])
        assert result.exit_code == 0, result.output

    invoke(
        "setup",
        "build",
        "--spec",
        source / "setup.json",
        "--plan",
        source / "plan.json",
        "--setup-id",
        "golden-poisson",
        "--output",
        root / "setup",
    )
    invoke("run", "--setup", root / "setup", "--output", root / "run")
    levels = [item for level in LEVELS for item in ("--level", level)]
    invoke(
        "results",
        "replay",
        "--setup",
        root / "setup",
        "--capture",
        root / "run" / "capture",
        *levels,
        "--output",
        root / "replay",
    )
    invoke(
        "results",
        "export",
        "--setup",
        root / "setup",
        "--capture",
        root / "run" / "capture",
        "--output",
        root / "export",
    )
    # The README quickstart's summary: of the replay, at the default levels
    # and the two the hand derivation checks.
    invoke(
        "results",
        "summary",
        "--results",
        root / "replay" / "results.json",
        "--output",
        root / "summary",
    )
    return GoldenOutputs(
        results=dict(load_results(root / "replay" / "results.json", diagnostics=True)),
        export=json.loads((root / "export" / "export.json").read_bytes()),
        simulations=json.loads((source / "plan.json").read_bytes())["simulations"],
        summary=root / "summary" / "summary.md",
        results_files=(root / "run" / "results.json", root / "replay" / "results.json"),
    )


def test_hand_derivation_matches_closed_forms():
    """Guard the oracle itself against the README's hand arithmetic."""
    laws = _hand_laws()
    support, mass = laws["gross"]
    assert np.dot(support, mass) == pytest.approx(14.0, rel=1e-12)
    assert np.dot(support**2, mass) - 14.0**2 == pytest.approx(1040.0, rel=1e-12)
    assert 1 - mass[0] == pytest.approx(1 - math.exp(-0.5), rel=1e-12)
    # Poisson thinning: L = 10 A + 100 B, A ~ Poisson(0.4), B ~ Poisson(0.1).
    assert mass[11] == pytest.approx(math.exp(-0.5) * 0.4 * 0.1, rel=1e-12)
    payout_support, payout_mass = laws["financial"]
    assert np.dot(payout_support, payout_mass) == pytest.approx(5.0, rel=1e-12)
    expected_tail = {
        ("gross", 0.95): (100.0, 117.287843),
        ("gross", 0.98): (110.0, 135.964849),
        ("financial", 0.95): (50.0, 54.837418),
        ("financial", 0.98): (50.0, 62.093545),
    }
    for (view, level), (var, tvar) in expected_tail.items():
        computed = _var_tvar(*laws[view], level)
        assert computed[0] == var
        assert computed[1] == pytest.approx(tvar, abs=1e-6)


@pytest.mark.stochastic
@pytest.mark.parametrize("view", ["gross", "oep_gross", "financial", "oep_financial"])
def test_golden_poisson_cli_outputs_match_the_hand_derivation(golden_outputs, view):
    results, export, n = (
        golden_outputs.results,
        golden_outputs.export,
        golden_outputs.simulations,
    )
    support, mass = _hand_laws()[view]
    metrics = results["arms"]["reference"]["metrics"][view]
    assert metrics["status"] == "available"
    mean = float(np.dot(support, mass))
    variance = float(np.dot(support**2, mass)) - mean**2

    reconciliation = metrics["eal_reconciliation"]
    diagnostics = results["diagnostics"]["arms"]["reference"]["views"][view]
    if view == "gross":
        # Native analytic EAL is exact integration of the authored law.
        assert metrics["eal_estimator"] == "native_analytic_mean"
        assert metrics["expected_annual_loss"] == pytest.approx(mean, rel=1e-12)
        # Fourth cumulant of compound Poisson: RATE E[X^4].
        fourth = RATE * sum(x**4 * p for x, p in SEVERITY.items())
        assert abs(metrics["annual_loss_variance"] - variance) <= 5 * math.sqrt(
            (fourth + 2 * variance**2) / n
        )
        # The years reconcile with the analytic EAL, so it withholds nothing.
        error = diagnostics["mean_diagnostics"]["empirical_mean_standard_error"]
        assert reconciliation["statistic"] == "mean"
        assert reconciliation["z"] == pytest.approx(
            (metrics["normalized_empirical_mean"] - mean) / error, rel=1e-9
        )
        assert reconciliation["status"] == "consistent"
    else:
        # Payout and occurrence-maximum views have no analytic EAL.
        assert reconciliation is None
    assert abs(metrics["normalized_empirical_mean"] - mean) <= 5 * math.sqrt(
        variance / n
    )
    positive = float(np.sum(mass[support > 0]))
    assert abs(metrics["annual_probability_positive"] - positive) <= 5 * math.sqrt(
        positive * (1 - positive) / n
    )
    statistics = metrics["statistics"]
    controls = results["tail_gate"]["by_arm"]["reference"]["controls"]
    assert statistics["expected_annual_loss"]["status"] == "available"
    assert statistics["annual_probability_positive"]["status"] == _predicted_status(
        positive, math.sqrt(positive * (1 - positive)), positive, controls, n
    )

    for level in LEVELS:
        var, tvar, below, at = _var_tvar(support, mass, level)
        level_error = math.sqrt(level * (1 - level) / n)
        assert min(level - below, at - level) >= 5 * level_error
        # Every statistic here is clear of the gate's thresholds, so each is
        # called; only gross has an exact mean.
        predicted = _predicted_tail_statuses(
            support, mass, level, var, tvar, controls, n, exact=view == "gross"
        )
        for functional, status in predicted.items():
            record = statistics[f"{functional}::{level}"]
            assert status is not None
            assert record["status"] == status, record["reason"]
            if status == "withheld":
                assert "half-width" in record["reason"]
        tail = metrics["tail_metrics"][str(level)]
        assert tail["var"] == var
        # Only gross has an exact mean; its coefficient is estimated from the
        # years, within five standard errors of the exact law's.
        beta, spread = _control_variate(support, mass, var, view == "gross")
        estimator = tail["tvar_estimator"]
        assert estimator["method"] == "control_variate"
        if view == "gross":
            assert estimator["coefficient_basis"] == "regression"
            assert abs(estimator["coefficient"] - beta) <= 5 * _coefficient_error(
                support, mass, var, beta, n
            )
        else:
            assert estimator["coefficient_basis"] == "no_exact_mean"
            assert estimator["coefficient"] == 0
        beyond = float(np.sum(mass[support > var]))
        assert abs(tail["exceedance_count"] / n - beyond) <= 5 * math.sqrt(
            beyond * (1 - beyond) / n
        )
        # A degenerate excess has no sampling error; allow only summation rounding.
        assert (
            abs(tail["tvar"] - tvar)
            <= 5 * spread / ((1 - level) * math.sqrt(n)) + 1e-9 * tvar
        )

    # The export's curve gives P(loss > x) at 0 and at each percentile of
    # the simulated losses, the largest of which no year exceeds.
    (series,) = [row for row in export["series"] if row["id"] == f"reference/{view}"]
    curve = {point["loss"]: point["probability"] for point in series["curve"]}
    assert 0.0 in curve and len(curve) <= 102
    assert curve[max(curve)] == 0
    for threshold, empirical in curve.items():
        exceedance = float(np.sum(mass[support > threshold]))
        assert abs(empirical - exceedance) <= 5 * math.sqrt(
            exceedance * (1 - exceedance) / n
        )


def _without_analysis_digest(markdown: str) -> str:
    """The summary with its analysis digest replaced by ``<analysis>``: the
    digest covers the reducer's provenance, whose Python, platform and package
    versions differ between CI runners."""
    return re.sub(r"(\*\*Analysis:\*\* `)[0-9a-f]{12}(`)", r"\1<analysis>\2", markdown)


@pytest.mark.stochastic
def test_golden_poisson_summary_matches_its_reviewed_snapshot(golden_outputs):
    """The readable summary of the seeded golden run: its study header,
    headline, tables, statuses and reasons. Figures are rounded for display,
    so a last-bit difference between platforms does not change them.
    ``tests/fixtures/summary/README.md`` says how to regenerate the snapshot."""
    rendered = _without_analysis_digest(
        golden_outputs.summary.read_text(encoding="utf-8")
    )
    assert "<analysis>" in rendered
    assert rendered == SUMMARY_SNAPSHOT.read_text(encoding="utf-8"), (
        f"{golden_outputs.summary} differs from {SUMMARY_SNAPSHOT}"
    )


def test_golden_poisson_results_files_match_their_schemas(golden_outputs):
    """The run's and the replay's results and diagnostics files conform to
    the committed JSON Schema documents (``contracts/results/``)."""
    for path in golden_outputs.results_files:
        assert schema_violations(path) == [], path


def test_a_misspelled_field_or_a_wrong_type_fails_the_schema(golden_outputs):
    """The documents close every record and type every value, so a misspelt
    or retyped field, or a reason stored as text, fails validation."""
    path = golden_outputs.results_files[1]
    stored = json.loads(path.read_bytes())
    assert payload_violations(stored) == []

    def violations(change) -> list[str]:
        payload = copy.deepcopy(stored)
        change(payload)
        return payload_violations(payload)

    def gross(payload):
        return payload["arms"]["reference"]["metrics"]["gross"]

    def misspell(record, key, wrong):
        record[wrong] = record.pop(key)

    assert any(
        "'rankings_availabel' was unexpected" in row
        for row in violations(
            lambda payload: misspell(
                payload, "rankings_available", "rankings_availabel"
            )
        )
    )
    assert any(
        row.startswith("/arms/reference/metrics/gross")
        for row in violations(
            lambda payload: misspell(
                gross(payload), "annual_loss_median", "annual_los_median"
            )
        )
    )
    assert any(
        "'one' is not of type 'number'" in row
        for row in violations(
            lambda payload: gross(payload).update(expected_annual_loss="one")
        )
    )
    assert any(
        "'14' is not of type 'number'" in row
        for row in violations(
            lambda payload: gross(payload)["tail_metrics"]["0.95"].update(var="14")
        )
    )
    assert violations(
        lambda payload: payload.update(schema_version=RESULTS_SCHEMA_VERSION + 1)
    )
    assert violations(lambda payload: payload.update(status="done"))
    # A reason must be stored as its code.
    text = "Unconverged at 20000 years per state."
    assert reason_code(text) != text
    assert violations(
        lambda payload: gross(payload)["statistics"]["var::0.9"].update(reason=text)
    )
    diagnostics = json.loads(diagnostics_path(path).read_bytes())
    assert payload_violations(diagnostics, DIAGNOSTICS_SCHEMA_FILE) == []
    views = diagnostics["arms"]["reference"]["views"]
    misspell(views["gross"], "mean_diagnostics", "mean_diagnostic")
    assert payload_violations(diagnostics, DIAGNOSTICS_SCHEMA_FILE)
