"""Run reports give each number a status, one coverage table and one limitations list."""

import json
from typing import Any, cast

from catmodel.results.execution import run_setup
from catmodel.results.reduction import STANDARD_VIEWS
from catmodel.results.result_disclosures import MODEL_LIMITATIONS
from catmodel.setup.sensitivity_catalog import SENSITIVITY_FAMILIES
from catmodel.setup.store import load_setup
from test_support.results import reduced_arm
from test_support.study import _package, _plan


def test_report_states_status_coverage_and_limitations(tmp_path):
    package = _package(tmp_path, plan=_plan(32))
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="producer"
    )
    assert "axes" not in report
    provenance = report["analysis_spec"]["capture_provenance"]
    assert provenance["producer_provenance"]["commit"] == "producer"
    assert not {"law_validity", "evaluator_feasibility", "evidence_reliance"} & set(
        provenance
    )
    # An unchanged producer assessment is recorded only by its digest.
    assert "producer_assessment" not in provenance
    assert len(provenance["producer_assessment_sha256"]) == 64

    setup = load_setup(package.root)
    requested = list(setup.execution.requested_views)
    reference = reduced_arm(report["arms"][setup.execution.reference_arm])
    for row in reference["metrics"].values():
        metric = cast(dict[str, Any], row)
        if metric["status"] != "available":
            assert metric["reason"]
            continue
        statistics = metric["statistics"]
        levels = metric["tail_metrics"]
        assert set(statistics) == {
            "expected_annual_loss",
            "annual_probability_positive",
            "annual_loss_median",
            "annual_loss_variance",
            "normalized_empirical_variance",
            *(f"{name}::{level}" for level in levels for name in ("var", "tvar")),
        }
        for name, row in statistics.items():
            assert row["status"] in {"available", "unavailable", "withheld"}
            assert (row["reason"] is None) == (row["status"] == "available")
            value = (
                levels[name.split("::")[1]][name.split("::")[0]]
                if "::" in name
                else metric[name]
            )
            if row["status"] == "available":
                assert value is not None
            elif row["status"] == "unavailable":
                assert value is None
        # 32 simulated years cannot support any requested tail level.
        for level in levels:
            for name in ("var", "tvar"):
                assert statistics[f"{name}::{level}"]["status"] == "withheld"

    coverage = report["coverage"]
    assert set(coverage) == {
        "comparisons",
        "views",
        "sensitivity_families",
        "scenarios",
    }
    views = [row["view"] for row in coverage["views"]]
    assert views[: len(requested)] == requested
    assert all(
        row["status"] == "not_requested" and row["reason"]
        for row in coverage["views"][len(requested) :]
    )
    assert set(views[len(requested) :]) == set(STANDARD_VIEWS) - set(requested)
    assert [row["family_id"] for row in coverage["sensitivity_families"]] == [
        family.family_id for family in SENSITIVITY_FAMILIES
    ]
    assert all(
        (row["reason"] is None) == (row["status"] == "complete")
        for row in coverage["sensitivity_families"]
    )
    assert [row["scenario_id"] for row in coverage["scenarios"]] == [
        row.scenario_id for row in setup.scope.populations
    ]
    assert all(row["status"] == "included" for row in coverage["scenarios"])

    sources = [row["source"] for row in report["limitations"]]
    assert sources == sorted(sources, key=["setup", "run", "model"].index)
    assert [
        row["text"] for row in report["limitations"] if row["source"] == "model"
    ] == [text for _, text in MODEL_LIMITATIONS]
    json.dumps(report, allow_nan=False)
