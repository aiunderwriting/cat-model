"""Scenario contributions: each scenario's EAL and share of the pooled EAL,
the unallocated remainder, and the summary's table."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from catmodel.cli import cli
from catmodel.results.execution import run_setup
from catmodel.results.scenario_contributions import scenario_contributions
from catmodel.simulation.capability_mixture import (
    ConditionalCapabilityDraws,
    ConditionalView,
)
from test_support.results import available_metric, reduced_arm
from test_support.study import _package, _plan


def _available(eal):
    return {
        "status": "available",
        "expected_annual_loss": eal,
        "statistics": {"expected_annual_loss": {"status": "available", "reason": None}},
    }


def _fixture(*, drop=None, pooled=True, extra=0.0):
    rng = np.random.default_rng(3)
    years = {
        "a": np.where(rng.random(400) < 0.05, 100.0, 0.0),
        "b": rng.poisson(1.0, 400).astype(float),
        "c": rng.poisson(2.0, 400).astype(float),
    }
    views = {
        f"scenario::{key}::welfare": ConditionalView(values)
        for key, values in years.items()
    }
    views["welfare"] = ConditionalView(years["a"] + years["b"] + years["c"] + extra)
    draw = ConditionalCapabilityDraws(
        stratum_sha256="state", response_sha256="response", seed=1, views=views
    )
    metrics = {
        name: _available(float(view.array.mean())) for name, view in views.items()
    }
    if drop == "unavailable":
        metrics["scenario::b::welfare"] = {
            "status": "unavailable",
            "reason": "Requested view was not retained in every positive-weight state.",
            "state_reasons": {"state": "No composition was elicited."},
        }
    elif drop == "not_requested":
        del metrics["scenario::b::welfare"]
    if not pooled:
        metrics["welfare"] = {
            "status": "unavailable",
            "reason": "Pooled views include unquantified populations: scenario::b: no law.",
            "state_reasons": {},
        }
    package = SimpleNamespace(
        execution=SimpleNamespace(reference_arm="reference"),
        scope=SimpleNamespace(
            populations=tuple(SimpleNamespace(scenario_id=key) for key in years)
        ),
    )
    report = {"arms": {"reference": {"metrics": metrics}}}
    return package, report, {"state": draw}, years


def _result(package, report, draws):
    return scenario_contributions(package, report, draws, {"state": 1.0})


def test_each_scenario_carries_its_mean_and_its_share_of_the_pooled_mean():
    package, report, draws, years = _fixture()
    result = _result(package, report, draws)
    assert result["status"] == "available"
    assert result["measure"] == "welfare"
    total = sum(values.mean() for values in years.values())
    assert result["expected_annual_loss"] == pytest.approx(total)
    rows = {row["scenario_id"]: row for row in result["scenarios"]}
    assert list(rows) == ["a", "b", "c"]
    for key, values in years.items():
        assert rows[key]["status"] == "available"
        assert rows[key]["view"] == f"scenario::{key}::welfare"
        assert rows[key]["expected_annual_loss"] == pytest.approx(values.mean())
        assert rows[key]["share_of_total"] == pytest.approx(values.mean() / total)
    assert result["unallocated"]["expected_annual_loss"] == pytest.approx(0, abs=1e-9)
    assert result["unallocated"]["annual_losses_reconcile"] is True


def test_a_missing_scenario_names_its_reason_and_leaves_no_remainder():
    package, report, draws, years = _fixture(drop="unavailable", pooled=False)
    result = _result(package, report, draws)
    assert result["expected_annual_loss"] is None
    assert "unquantified" in result["total_reason"]
    rows = {row["scenario_id"]: row for row in result["scenarios"]}
    assert rows["b"]["status"] == "unavailable"
    assert rows["b"]["reason"] == "No composition was elicited."
    assert rows["b"]["expected_annual_loss"] is None
    # Without the pooled EAL no scenario has a share.
    assert rows["c"]["expected_annual_loss"] == pytest.approx(years["c"].mean())
    assert rows["c"]["share_of_total"] is None
    assert result["unallocated"] == {
        "expected_annual_loss": None,
        "reason": "Needs the pooled EAL and every scenario's EAL.",
        "annual_losses_reconcile": None,
    }


def test_an_unrequested_view_and_loss_outside_every_scenario_are_reported():
    package, report, draws, _ = _fixture(drop="not_requested", extra=1.0)
    result = _result(package, report, draws)
    row = next(row for row in result["scenarios"] if row["scenario_id"] == "b")
    assert row["status"] == "not_requested"
    assert row["reason"] == "The view was not requested."
    # One unit a year outside the three scenarios, in every year.
    package, report, draws, _ = _fixture(extra=1.0)
    result = _result(package, report, draws)
    assert result["unallocated"]["expected_annual_loss"] == pytest.approx(1.0)
    assert result["unallocated"]["share_of_total"] == pytest.approx(
        1.0 / result["expected_annual_loss"]
    )
    assert result["unallocated"]["annual_losses_reconcile"] is False


def test_no_additive_view_is_unavailable_with_its_reason():
    package, report, draws, _ = _fixture()
    report["arms"]["reference"]["metrics"] = {}
    assert _result(package, report, draws) == {
        "status": "unavailable",
        "reason": "No pooled or per-scenario additive view (welfare, gross or net_resource_cost) was requested.",
    }


def test_run_reports_contributions_and_summary_writes_markdown_and_csv(tmp_path):
    package = _package(
        tmp_path,
        plan=_plan(64).model_copy(
            update={"requested_views": ("welfare", "scenario::loss::welfare")}
        ),
    )
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    contributions = report["scenario_contributions"]
    assert contributions["status"] == "available"
    welfare = available_metric(
        reduced_arm(report["arms"]["reference"])["metrics"]["welfare"]
    )
    (scenario,) = contributions["scenarios"]
    assert scenario["scenario_id"] == "loss"
    assert scenario["expected_annual_loss"] == pytest.approx(
        welfare["expected_annual_loss"]
    )
    assert scenario["share_of_total"] == pytest.approx(1)
    assert contributions["unallocated"]["annual_losses_reconcile"] is True

    def summarize(output):
        return CliRunner().invoke(
            cli,
            [
                "results",
                "summary",
                "--results",
                str(tmp_path / "run" / "results.json"),
                "--output",
                str(output),
            ],
        )

    result = summarize(tmp_path / "summary")
    assert result.exit_code == 0, result.output
    markdown = (tmp_path / "summary" / "summary.md").read_text()
    for heading in (
        "## Expected annual loss by scenario",
        # The tail statistics' tables are labelled illustrative.
        "## Annual aggregate loss (AEP), illustrative\n",
        "## Largest single occurrence in a year (OEP), illustrative\n",
        "## Tail value at risk, illustrative\n",
        "## Sensitivity",
        "## Unavailable, withheld and not requested",
        "## Limitations",
    ):
        assert heading in markdown
    section = markdown.split("## Expected annual loss by scenario")[1].split("\n## ")[0]
    # An authored setup has no large-loss anchor, so the table has no such column.
    assert "| Scenario | EAL | Share of EAL | Note |" in section.splitlines()
    assert "| `loss` |" in section and "| 100.0% |" in section
    assert "| total (pooled view) |" in section
    assert "1-in-200" in markdown
    rows = list(csv.DictReader((tmp_path / "summary" / "summary.csv").open()))
    eal = next(
        row
        for row in rows
        if row["section"] == "exceedance"
        and row["view"] == "welfare"
        and row["statistic"] == "expected_annual_loss"
    )
    assert float(eal["value"]) == pytest.approx(welfare["expected_annual_loss"])
    assert eal["unit"] == "USD"
    tvar = [
        row
        for row in rows
        if row["section"] == "tvar"
        and row["view"] == "welfare"
        and row["statistic"] == "tvar"
    ]
    assert {row["level"] for row in tvar} == {"0.9", "0.99", "0.995", "0.996", "0.999"}
    # 64 years cannot support the deep tail: those values are withheld.
    assert {row["status"] for row in tvar if row["level"] == "0.999"} == {"withheld"}
    share = next(
        row
        for row in rows
        if row["section"] == "scenario"
        and row["subject"] == "loss"
        and row["statistic"] == "share_of_total"
    )
    assert (share["unit"], share["status"]) == ("share", "available")
    assert float(share["value"]) == pytest.approx(1)
    assert not any(
        row["statistic"] == "gross_eal_share_at_or_above_large" for row in rows
    )
    # Every status is a statistic's; a coverage outcome is a row's label.
    assert {row["status"] for row in rows} <= {
        "available",
        "withheld",
        "unavailable",
        "not_requested",
    }
    assert any(
        row["section"] == "coverage" and row["label"] == "not_requested" for row in rows
    )
    assert summarize(tmp_path / "summary").exit_code != 0
