"""Executed EAL effects remain useful without hiding missing comparisons."""

import json
import xml.etree.ElementTree as ET

import pytest

from catmodel.results.sensitivity_summary import (
    _FAMILY_KINDS,
    sensitivity_summary,
    tornado_svg,
)


def _report():
    basis = {"currency": "USD", "monetary_basis": "2027 USD"}

    def metric(value):
        return {"status": "available", "expected_annual_loss": value, "basis": basis}

    return {
        "analysis_sha256": "a" * 64,
        "status": "partial",
        "coverage": {"comparisons": {"status": "partial"}},
        "arms": {
            "reference": {
                "metrics": {
                    "welfare": metric(100),
                    "gross": metric(110),
                    "scenario::a::welfare": metric(40),
                    "scenario::b::welfare": {
                        **metric(None),
                        "statistics": {
                            "expected_annual_loss": {
                                "status": "unavailable",
                                "reason": "The mean does not exist.",
                            }
                        },
                    },
                    "scenario::b::gross": {
                        "status": "unavailable",
                        "reason": "No gross ledger was elicited.",
                    },
                }
            },
            "higher": {},
            "lower": {},
            "unchanged": {},
            "missing": {"status": "unavailable", "reason": "No captured law."},
        },
        "comparisons": {
            arm: {
                "welfare": {
                    "expected_annual_loss": {
                        "reference": 100,
                        "alternative": 100 + difference,
                        "absolute_difference": difference,
                        "relative_change": difference / 100,
                    }
                }
            }
            for arm, difference in (("higher", 30), ("lower", -50), ("unchanged", 0))
        },
        "comparison_reference_arms": {
            arm: "reference" for arm in ("higher", "lower", "unchanged", "missing")
        },
        "rankings": {
            "welfare": {
                "expected_annual_loss": {
                    "status": "unavailable",
                    "reason": "Required comparison coverage is incomplete for this view.",
                }
            }
        },
        "comparison_requirements": [
            {
                "family_id": "OBS-CAUSAL",
                "scenario_ids": ["a", "b"],
                "status": "unavailable",
                "outcome_status": "unavailable",
                "reason": "Excluded from this import.",
            },
            {
                "family_id": "tail_shape",
                "arm_id": "higher",
                "scenario_ids": ["a"],
                "status": "required",
                "outcome_status": "available",
            },
        ],
    }


def _summary(report):
    return sensitivity_summary(report, reference_arm="reference", purposes={})


def test_summary_preserves_scenario_mean_and_view_unavailability():
    summary = _summary(_report())
    assert summary["scenario_eal"]["a"]["welfare"] == {
        "status": "available",
        "expected_annual_loss": 40,
        "basis": {"currency": "USD", "monetary_basis": "2027 USD"},
        "reason": None,
    }
    missing = summary["scenario_eal"]["b"]
    assert missing["welfare"]["status"] == "unavailable"
    assert missing["welfare"]["expected_annual_loss"] is None
    assert missing["welfare"]["reason"] == "The mean does not exist."
    assert missing["gross"]["reason"] == "No gross ledger was elicited."
    assert summary["comparison_gaps"][0]["reason"] == "Excluded from this import."
    assert len(summary["comparison_gaps"]) == 1
    json.dumps(summary, allow_nan=False)


def test_partial_coverage_keeps_finite_negative_positive_and_zero_effects():
    summary = _summary(_report())
    tornado = summary["tornado"]
    assert summary["comparison_status"] == "partial"
    assert tornado["view"] == "welfare"
    assert [row["arm"] for row in tornado["rows"]] == [
        "lower",
        "higher",
        "unchanged",
    ]
    assert [row["absolute_difference"] for row in tornado["rows"]] == [-50, 30, 0]
    assert [row["mean_changing"] for row in tornado["rows"]] == [True, True, False]
    assert tornado["unavailable_arms"] == {"missing": "No captured law."}


def test_empty_and_zero_effect_charts_are_finite_and_escape_authored_text():
    report = _report()
    report["comparisons"] = {"<zero>&": report["comparisons"]["unchanged"]}
    report["arms"] = {
        "reference": report["arms"]["reference"],
        "<zero>&": {},
    }
    summary = _summary(report)
    chart = tornado_svg(summary)
    assert "&lt;zero&gt;&amp;" in chart
    assert 'width="0.000"' in chart
    root = ET.fromstring(chart)
    assert root.attrib["role"] == "img"
    summary["tornado"]["rows"] = []
    assert "No finite EAL contrasts" in tornado_svg(summary)


def test_unavailable_reference_does_not_choose_or_invent_a_numeric_view():
    report = _report()
    report["arms"]["reference"] = {"status": "unavailable", "reason": "No capture."}
    summary = _summary(report)
    assert summary["tornado"]["view"] is None
    assert summary["tornado"]["rows"] == []
    assert summary["scenario_eal"] == {}


def test_svg_keeps_finite_bar_coordinates_for_tiny_eal_effects():
    summary = _summary(_report())
    summary["tornado"]["rows"] = [{"arm": "tiny", "absolute_difference": 1e-310}]
    chart = tornado_svg(summary)
    assert 'width="230.000"' in chart
    assert 'width="inf"' not in chart


def test_scenario_view_measure_follows_a_scenario_id_holding_the_separator():
    report = _report()
    metric = report["arms"]["reference"]["metrics"].pop("scenario::a::welfare")
    report["arms"]["reference"]["metrics"]["scenario::a::subscenario::gross"] = metric
    summary = _summary(report)
    assert set(summary["scenario_eal"]["a::subscenario"]) == {"gross"}
    assert (
        summary["scenario_eal"]["a::subscenario"]["gross"]["expected_annual_loss"] == 40
    )


def test_summary_carries_results_ranking_status_and_unassessed_families():
    summary = _summary(_report())
    tornado = summary["tornado"]
    assert tornado["results_ranking"] == {
        "status": "unavailable",
        "reason": "Required comparison coverage is incomplete for this view.",
    }
    assert tornado["not_assessed_families"] == ["OBS-CAUSAL"]
    chart = tornado_svg(summary)
    assert "Incomplete coverage" in chart
    assert (
        "Not assessed for this view (1 families or register challenges): OBS-CAUSAL"
        in chart
    )


@pytest.mark.parametrize("status", ("not_requested", "inapplicable"))
def test_summary_preserves_comparison_disposition_when_overall_result_is_partial(
    status,
):
    report = _report()
    report["coverage"]["comparisons"]["status"] = status
    report["arms"] = {"reference": report["arms"]["reference"]}
    report["comparisons"] = {}
    report["comparison_reference_arms"] = {}
    report["comparison_requirements"] = []
    if status == "inapplicable":
        report["comparison_requirements"] = [
            {
                "family_id": "tail_shape",
                "scenario_ids": ["a", "b"],
                "status": "inapplicable",
                "outcome_status": "inapplicable",
                "reason": "The selected law has no tail-shape challenge.",
            }
        ]
    report["rankings"]["welfare"]["expected_annual_loss"] = {
        "status": status,
        "reason": "No numerical comparison was executed.",
    }

    summary = _summary(report)

    assert report["status"] == "partial"
    assert summary["comparison_status"] == status
    assert summary["tornado"]["results_ranking"]["status"] == status
    assert "Incomplete coverage" in tornado_svg(summary)


def test_only_assumption_challenges_are_ranked_against_their_controls(monkeypatch):
    # No current catalog family is a diagnostic; a synthetic one checks the rule.
    monkeypatch.setitem(_FAMILY_KINDS, "synthetic_diagnostic", "diagnostic")
    report = _report()
    effect = report["comparisons"]["higher"]
    for arm in ("refined", "control", "stress", "diagnostic"):
        report["arms"][arm] = {}
        report["comparisons"][arm] = effect
    report["comparison_reference_arms"].update(
        refined="reference",
        control="control",
        stress="control",
        diagnostic="reference",
    )
    report["comparison_requirements"].append(
        {
            "family_id": "synthetic_diagnostic",
            "arm_id": "diagnostic",
            "scenario_ids": ["a"],
            "status": "required",
            "outcome_status": "available",
        }
    )
    summary = sensitivity_summary(
        report,
        reference_arm="reference",
        purposes={"refined": "numerical_refinement"},
    )
    tornado = summary["tornado"]
    assert set(tornado["excluded_arms"]) == {"refined", "control", "diagnostic"}
    (stress,) = (row for row in tornado["rows"] if row["arm"] == "stress")
    assert stress["control_arm"] == "control"
    assert {row["arm"] for row in tornado["rows"]} == {
        "higher",
        "lower",
        "unchanged",
        "stress",
    }


def test_view_selection_does_not_require_a_finite_mean():
    report = _report()
    report["arms"]["reference"]["metrics"]["welfare"] = {
        "status": "available",
        "expected_annual_loss": None,
        "statistics": {
            "expected_annual_loss": {
                "status": "unavailable",
                "reason": "The mean does not exist.",
            }
        },
    }
    tornado = _summary(report)["tornado"]
    assert tornado["view"] == "welfare"
    assert tornado["eal"]["reason"] == "The mean does not exist."
