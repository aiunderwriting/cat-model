"""Comparison completeness follows frozen expectations, including an empty inventory."""

import pytest
from pydantic import ValidationError

from catmodel.contracts.incident import ExecutionPlan
from test_support.results import statistic_ranking
from tests.test_reduction import draw, reduce, requirement


def test_reference_only_capture_does_not_claim_comparison_completion():
    report = reduce()
    assert report["arms"]["base"]["metrics"]["gross"]["expected_annual_loss"] == 40
    assert report["status"] == "partial"
    coverage = report["coverage"]["comparisons"]
    assert coverage["status"] == "not_requested"
    assert coverage["required_alternative_arms"] == []
    assert not coverage["requirements_complete"]
    assert report["comparisons"] == {}
    assert report["comparison_requirements"] == []
    assert not report["rankings_available"]
    # Every statistic of the view has the one disposition.
    ranking = report["rankings"]["gross"]
    assert ranking["status"] == "not_requested"
    assert "No required alternative arm" in ranking["reason"]


@pytest.mark.parametrize("capture_status", ["complete", "missing"])
def test_expected_inventory_is_independent_of_available_captures(capture_status):
    captures = {
        "base": {"s": draw(values=(10, 10))},
        "lower": {"s": draw(values=(8, 8))},
    }
    if capture_status == "complete":
        captures["higher"] = {"s": draw(values=(20, 20))}
    report = reduce(
        captures,
        required_arms=("base", "lower", "higher"),
        comparison_requirements=(
            requirement("severity", "lower"),
            requirement("severity", "higher"),
        ),
    )
    complete = capture_status == "complete"
    coverage = report["coverage"]["comparisons"]
    assert coverage["required_alternative_arms"] == ["lower", "higher"]
    assert coverage["status"] == ("complete" if complete else "partial")
    assert coverage["requirements_complete"] is complete
    assert report["status"] == ("complete" if complete else "partial")
    assert report["rankings_available"] is complete
    assert report["comparison_requirements"][1]["outcome_status"] == (
        "available" if complete else capture_status
    )
    assert (
        report["comparisons"]["lower"]["gross"]["expected_annual_loss"][
            "absolute_difference"
        ]
        == -2
    )


def test_explicit_arm_inventory_can_complete_without_family_metadata():
    report = reduce(
        {"base": {"s": draw(values=(10, 10))}, "higher": {"s": draw(values=(20, 20))}},
        required_arms=("base", "higher"),
    )
    assert report["coverage"]["comparisons"]["required_alternative_arms"] == ["higher"]
    assert report["status"] == "complete"
    assert report["rankings_available"]


@pytest.mark.parametrize("status", ["inapplicable", "unavailable"])
def test_reasoned_disposition_does_not_claim_completed_numerical_comparisons(status):
    reason = "The selected cohort does not support this comparison."
    report = reduce(
        comparison_requirements=(
            requirement("calendar", status=status, reason=reason),
        ),
    )
    assert report["status"] == "partial"
    assert report["coverage"]["comparisons"]["status"] == (
        "inapplicable" if status == "inapplicable" else "partial"
    )
    assert report["comparison_requirements"][0]["outcome_status"] == status
    assert report["comparison_requirements"][0]["reason"] == reason
    assert not report["rankings_available"]
    ranking = statistic_ranking(report["rankings"], "gross", "expected_annual_loss")
    assert ranking["status"] == status
    assert ("inapplicable" if status == "inapplicable" else "incomplete") in ranking[
        "reason"
    ]


@pytest.mark.parametrize("status", ["required", "unavailable", "inapplicable"])
def test_comparison_requirement_cannot_name_the_reference_arm(status):
    row = requirement(
        "severity",
        "reference",
        status=status,
        reason=None if status == "required" else "The reference is not an alternative.",
    )
    with pytest.raises(ValidationError, match="required alternative arms"):
        ExecutionPlan.model_validate(
            {
                "plan_id": "plan",
                "required_arms": ["reference", "higher"],
                "simulations": 1,
                "seed": 0,
                "requested_views": ["gross"],
                "comparison_requirements": [row],
            }
        )
    with pytest.raises(ValueError, match="required alternative arms"):
        reduce(
            {"base": {"s": draw()}, "higher": {"s": draw()}},
            required_arms=("base", "higher"),
            comparison_requirements=({**row, "arm_id": "base"},),
        )
