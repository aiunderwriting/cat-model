"""Comparison scope follows declared output views independently of family names."""

from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError

from catmodel.contracts.incident import ComparisonRequirement
from catmodel.results.execution import replay_capture, run_setup
from catmodel.simulation.capability_mixture import ConditionalView
from test_support.results import statistic_ranking
from test_support.study import _financial_study, _package, _plan
from tests.test_reduction import draw, reduce, requirement


def test_required_scope_without_requested_views_is_unassessed():
    result = reduce(
        {"base": {"s": draw(values=(10, 10))}, "payment": {"s": draw(values=(20, 20))}},
        required_arms=("base", "payment"),
        views=("gross",),
        comparison_requirements=(scoped_requirement("payment"),),
    )
    row = result["comparison_requirements"][0]
    assert row["requested_views"] == []
    assert row["view_outcomes"] == {}
    assert row["outcome_status"] == "unassessed"
    assert "No requested view" in row["outcome_reason"]
    assert result["comparison_families"]["arbitrary_family"]["status"] == "partial"
    assert result["status"] == "partial"
    assert not result["coverage"]["comparisons"]["requirements_complete"]
    assert result["rankings"]["gross"] == {
        "status": "not_requested",
        "reason": "No required alternative arm is in this view's scope.",
    }
    assert not result["rankings_available"]


def test_empty_requested_scope_preserves_missing_arm_reason():
    result = reduce(
        {"base": {"s": draw(values=(10, 10))}},
        required_arms=("base", "payment"),
        views=("gross",),
        comparison_requirements=(scoped_requirement("payment"),),
    )
    row = result["comparison_requirements"][0]
    assert row["requested_views"] == []
    assert row["outcome_status"] == "missing"
    assert row["outcome_reason"]


@pytest.mark.parametrize(
    "affected_views,expected_status",
    [
        (("financial",), "unassessed"),
        (("gross", "financial"), "available"),
        (None, "available"),
    ],
)
def test_public_partial_view_plan_assesses_only_requested_scoped_views(
    tmp_path, affected_views, expected_status
):
    plan = _plan(4).model_copy(
        update={
            "requested_views": ("gross",),
            "comparison_requirements": (
                ComparisonRequirement(
                    family_id="payment",
                    arm_id="contained",
                    scenario_ids=("loss",),
                    affected_views=affected_views,
                    held_fixed=("Original conditional laws and target state weights.",),
                ),
            ),
        }
    )
    package = _package(tmp_path, study=_financial_study(), plan=plan)
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    row = report["comparison_requirements"][0]
    assert row["outcome_status"] == expected_status
    assert row["requested_views"] == (
        [] if expected_status == "unassessed" else ["gross"]
    )
    complete = expected_status == "available"
    assert report["status"] == ("complete" if complete else "partial")
    assert report["coverage"]["comparisons"]["requirements_complete"] is complete
    assert report["comparison_families"]["payment"]["status"] == (
        "complete" if complete else "partial"
    )
    ranking = statistic_ranking(report["rankings"], "gross", "expected_annual_loss")
    assert ranking["status"] == ("available" if complete else "not_requested")
    assert report["rankings_available"] is complete
    replay = replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run/capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    assert replay["comparison_requirements"] == report["comparison_requirements"]
    assert replay["comparison_families"] == report["comparison_families"]
    assert replay["coverage"]["comparisons"] == report["coverage"]["comparisons"]


def scoped_requirement(arm=None, *, status="required", views=("financial",)):
    return {
        **requirement(
            "arbitrary_family",
            arm,
            status=status,
            reason=None
            if status == "required"
            else "Required financial evidence is unavailable.",
        ),
        "affected_views": views,
    }


def test_unavailable_financial_requirement_does_not_hide_gross_rankings():
    result = reduce(
        {"base": {"s": draw(values=(10, 10))}, "higher": {"s": draw(values=(20, 20))}},
        required_arms=("base", "higher"),
        views=("gross", "financial"),
        comparison_requirements=(scoped_requirement(status="unavailable"),),
    )
    assert result["status"] == "partial"
    ranking = result["rankings"]["gross"]["expected_annual_loss"]
    assert ranking["status"] == "available"
    assert [row["arm"] for row in ranking["arms"]] == ["higher"]
    family = result["comparison_families"]["arbitrary_family"]
    assert family["status"] == "partial"
    assert "gross" not in family["rankings"]
    assert result["comparison_requirements"][0]["requested_views"] == ["financial"]


@pytest.mark.parametrize("affected_views", [("financial",), None])
def test_missing_arm_only_blocks_views_inside_its_declared_scope(affected_views):
    result = reduce(
        {"base": {"s": draw(values=(10, 10))}, "higher": {"s": draw(values=(20, 20))}},
        required_arms=("base", "higher", "missing"),
        views=("gross", "financial"),
        comparison_requirements=(scoped_requirement("missing", views=affected_views),),
    )
    assert result["status"] == "partial"
    ranking = statistic_ranking(result["rankings"], "gross", "expected_annual_loss")
    assert ranking["status"] == (
        "available" if affected_views is not None else "unavailable"
    )


def two_view_captures():
    return {
        arm: {
            "s": replace(
                draw(),
                views={
                    "gross": ConditionalView(np.array([value, value])),
                    "financial": ConditionalView(np.array([value / 2, value / 2])),
                },
            )
        }
        for arm, value in (("base", 10.0), ("payment", 20.0))
    }


def test_family_effects_and_rankings_do_not_publish_unaffected_views():
    result = reduce(
        two_view_captures(),
        required_arms=("base", "payment"),
        views=("gross", "financial"),
        comparison_requirements=(scoped_requirement("payment"),),
    )
    family = result["comparison_families"]["arbitrary_family"]
    assert set(family["rankings"]) == {"financial"}
    assert result["comparison_requirements"][0]["view_outcomes"] == {
        "financial": {"status": "available"}
    }
    assert result["status"] == "complete"
    assert result["rankings_available"]
    ranking = statistic_ranking(result["rankings"], "gross", "expected_annual_loss")
    assert ranking["status"] == "not_requested"
    assert "No required alternative arm" in ranking["reason"]
    financial = result["rankings"]["financial"]["expected_annual_loss"]
    assert financial["status"] == "available"
    assert [row["arm"] for row in financial["arms"]] == ["payment"]


def test_unavailable_disposition_ranks_its_view_as_incomplete_coverage():
    result = reduce(
        two_view_captures(),
        required_arms=("base", "payment"),
        views=("gross", "financial"),
        comparison_requirements=(
            scoped_requirement("payment"),
            scoped_requirement(status="unavailable", views=("gross",)),
        ),
    )
    family = result["comparison_families"]["arbitrary_family"]
    assert family["status"] == "partial"
    assert result["status"] == "partial"
    assert not result["rankings_available"]
    for rankings in (result["rankings"], family["rankings"]):
        gross = statistic_ranking(rankings, "gross", "expected_annual_loss")
        assert gross["status"] == "unavailable"
        assert "coverage is incomplete" in gross["reason"]
        assert rankings["financial"]["expected_annual_loss"]["status"] == "available"


@pytest.mark.parametrize("views", [(), ("gross", "gross")])
def test_explicit_view_scope_is_nonempty_and_unique(views):
    with pytest.raises(ValidationError):
        ComparisonRequirement.model_validate(scoped_requirement("payment", views=views))
