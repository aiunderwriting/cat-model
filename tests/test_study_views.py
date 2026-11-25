"""Scoped output identifiers follow explicit relations throughout the frozen study."""

import numpy as np
import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.contracts.incident import ExecutionPlan
from catmodel.results.execution import load_run_capture, replay_capture, run_setup
from catmodel.setup.store import build_setup
from catmodel.simulation.incident import (
    compile_incident_state,
    declared_view_identifiers,
)
from catmodel.simulation.incident_runtime import simulate_incident_state
from test_support.results import available_metric, reduced_arm
from test_support.run_setup import setup_from_payload, with_execution
from test_support.study import _plan, _study, study_payload


def _roles(component, identifier):
    return {
        **component,
        "component_id": identifier,
        "bearer_id": identifier + "-bearer",
        "contributor_ids": [identifier + "-contributor"],
        "attribution_weights": {identifier + "-attribution": 1},
    }


def test_declared_roles_include_zero_support_joint_branches_and_alternative_overrides():
    payload = encode_scientific(_study())
    kernel = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"]
    member = kernel["member_outcomes"][0]
    prototype = member["components"][0]
    kernel.update(
        cluster=None,
        member_outcomes=[],
        explicit_joint_members=[
            {
                "probability": 1,
                "campaign_components": [_roles(prototype, "campaign")],
                "members": [{**member, "components": [_roles(prototype, "member")]}],
            },
            {
                "probability": 0,
                "campaign_components": [_roles(prototype, "zero-campaign")],
                "members": [
                    {**member, "components": [_roles(prototype, "zero-member")]}
                ],
            },
        ],
    )
    override = payload["alternatives"][0]["overrides"][0]["law"]
    override["segments"][0]["kernel"]["campaign_components"] = [
        _roles(prototype, "alternative")
    ]
    identifiers = declared_view_identifiers(setup_from_payload(payload))
    for role in ("bearer", "contributor", "attribution"):
        assert {
            identifier + "-" + role
            for identifier in (
                "campaign",
                "member",
                "zero-campaign",
                "zero-member",
                "alternative",
            )
        } <= identifiers[role]
    assert identifiers["scenario"] == frozenset({"loss"})
    assert identifiers["risk_unit"] == frozenset({"campaign"})
    assert "model" not in identifiers["bearer"]


@pytest.mark.parametrize("category", ["bearer", "contributor", "attribution"])
def test_role_declared_in_another_state_remains_a_known_zero_here(category):
    payload = encode_scientific(_study())
    component = payload["joint_states"][1]["laws"][0]["segments"][0]["kernel"][
        "member_outcomes"
    ][0]["components"][0]
    component["bearer_id"] = "high-only"
    component["contributor_ids"] = ["high-only"]
    component["attribution_weights"] = {"high-only": 1}
    study = setup_from_payload(payload)
    view = f"{category}::high-only"
    plan = _plan(16).model_copy(update={"requested_views": ("gross", view)})
    source, unavailable, _ = simulate_incident_state(
        plan, compile_incident_state(study, "low"), response_sha256="a" * 64
    )
    assert view not in unavailable
    np.testing.assert_array_equal(
        source.annual_losses[view], np.zeros(plan.simulations)
    )
    assert any(view.native_mean is not None for view in source.views.values())
    native = source.views[view].native_mean
    assert native is not None
    assert native.value == 0


@pytest.mark.parametrize(
    "category,typo",
    [("bearer", "victm"), ("contributor", "modle"), ("attribution", "modle")],
)
def test_direct_runtime_withholds_literal_typo_in_scoped_output(category, typo):
    unknown = f"{category}::{typo}"
    study = _study()
    plan = _plan(16).model_copy(update={"requested_views": ("gross", unknown)})
    draw, unavailable, _ = simulate_incident_state(
        plan, compile_incident_state(study, "low"), response_sha256="a" * 64
    )
    assert "gross" in draw.annual_losses
    assert unknown not in draw.annual_losses
    assert unknown in unavailable


@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize("measure", ["welfare", "net_resource_cost"])
def test_missing_scoped_nature_withholds_only_that_view(tmp_path, aggregate, measure):
    payload = study_payload()
    law = payload["joint_states"][0]["laws"][0]
    law["reference_annual_rate"] = 100
    law["segments"][0]["kernel"]["member_outcomes"][0]["components"][0]["nature"] = None
    study = setup_from_payload(payload)
    scoped = "scenario::scenario::" + measure
    captured = []
    for label, views in (("gross", ("gross",)), ("scoped", ("gross", scoped))):
        plan = ExecutionPlan(
            plan_id="scoped-nature",
            required_arms=("reference",),
            simulations=8,
            seed=19,
            requested_views=views,
        )
        package = build_setup(
            destination=tmp_path / label,
            setup=with_execution(study, plan, setup_id="scoped-nature"),
            code_commit="fixture",
        )
        run = tmp_path / (label + "-run")
        report = run_setup(setup_path=package.root, output=run, code_commit="fixture")
        replay = replay_capture(
            setup_path=package.root,
            capture_directory=run / "capture",
            output=tmp_path / (label + "-replay.json"),
            code_commit="fixture",
        )
        assert replay["arms"] == report["arms"]
        metrics = reduced_arm(report["arms"]["reference"])["metrics"]
        assert available_metric(metrics["gross"])[
            "expected_annual_loss"
        ] == pytest.approx(1000)
        if label == "scoped":
            assert metrics[scoped]["status"] == "unavailable"
        draws, capture = load_run_capture(run / "capture")
        if label == "scoped":
            assert all(
                "natures" in support.unavailable_views[scoped]
                for support in capture.supports.values()
            )
        captured.append(next(iter(draws["reference"].values())).views["gross"].array)
    np.testing.assert_array_equal(*captured)
