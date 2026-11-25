"""Source-status subtotals filter shared incident samples without allocating them twice."""

from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from catmodel.results.conditional import ConditionalCapture
from catmodel.results.execution import replay_capture, run_setup
from catmodel.results.results_export import export_results, load_results_export
from catmodel.setup.store import build_setup
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_runtime import simulate_incident_state
from test_support.conditional import materialize_constant_eci_grid
from test_support.run_setup import setup_from_payload, with_execution
from test_support.study import _plan, study_payload

PUBLIC = "source_status::public::"
NOT_PUBLIC = "source_status::not_yet_public::"
VIEWS = (
    "welfare",
    "lab_loss::lab",
    PUBLIC + "welfare",
    PUBLIC + "lab_loss::lab",
    NOT_PUBLIC + "welfare",
    NOT_PUBLIC + "lab_loss::lab",
)


def _payload():
    payload = study_payload()
    payload["entities"][0]["related_entity_ids"] = ["lab"]
    payload["entities"].append(
        {
            "entity_id": "lab",
            "kind": "lab",
            "definition": "Hypothetical lab before recoveries",
            "display_name": "Lab",
            "status": "hypothetical",
        }
    )
    payload["deployments"] = []
    for deployment in ("public", "internal"):
        payload["entities"].append(
            {
                "entity_id": deployment,
                "kind": "deployment",
                "definition": deployment + " use",
                "display_name": deployment,
                "status": "hypothetical",
            }
        )
        payload["deployments"].append(
            {
                "deployment_id": deployment,
                "model_id": "model",
                "use_population": deployment + " users",
                "tools_and_permissions": "Fixed",
                "human_oversight": "Fixed",
            }
        )
    payload["populations"][0]["deployment_ids"] = ["public", "internal"]
    original = payload["joint_states"][0]["laws"][0]
    laws = []
    for deployment, rate, amount, fraction, fine in (
        ("public", 2, 10, 0.5, 4),
        ("internal", 3, 20, 0.25, 2),
    ):
        law = deepcopy(original)
        law.update(
            origin_id=deployment,
            deployment_id="internal" if deployment == "public" else "public",
            reference_annual_rate=rate,
        )
        law["segments"][0].update(
            source_model_id="source-" + deployment,
            source_model_revision="r1",
            source_public_status="public"
            if deployment == "public"
            else "not_yet_public",
        )
        components = law["segments"][0]["kernel"]["member_outcomes"][0]["components"]
        components[0].update(amount=amount, lab_loss_weights={"lab": fraction})
        components.append(
            {
                "component_id": "fine",
                "amount": fine,
                "nature": "transfer",
                "lab_loss_weights": {"lab": 1},
            }
        )
        laws.append(law)
    payload["joint_states"][0]["laws"] = laws
    return payload


def _plan_for_views():
    return _plan(48).model_copy(
        update={
            "required_arms": ("reference",),
            "requested_views": VIEWS,
        }
    )


def _simulate(payload):
    return simulate_incident_state(
        _plan_for_views(),
        compile_incident_state(setup_from_payload(payload), "state"),
        response_sha256="a" * 64,
    )


def test_subtotals_share_rows_and_exact_native_means():
    payload = _payload()
    assert payload["joint_states"][0]["laws"][0]["deployment_id"] == "internal"
    draw, unavailable, execution = _simulate(payload)
    assert not unavailable
    for measure in ("welfare", "lab_loss::lab"):
        np.testing.assert_allclose(
            draw.annual_losses[measure],
            draw.annual_losses[PUBLIC + measure]
            + draw.annual_losses[NOT_PUBLIC + measure],
        )
    for view, mean in (
        (PUBLIC + "welfare", 20),
        (NOT_PUBLIC + "welfare", 60),
        (PUBLIC + "lab_loss::lab", 18),
        (NOT_PUBLIC + "lab_loss::lab", 21),
    ):
        assert draw.views[view].native_mean.value == pytest.approx(mean)
        assert execution.view_bases[view]["source_public_status"] in {
            "public",
            "not_yet_public",
        }


@pytest.mark.parametrize("campaign_cost", [False, True])
def test_release_midyear_retains_initiating_status_for_entire_occurrence(campaign_cost):
    payload = _payload()
    law = payload["joint_states"][0]["laws"][0]
    payload["joint_states"][0]["laws"] = [law]
    law["reference_annual_rate"] = 4
    before = law["segments"][0]
    before.update(
        start=0,
        end=0.5,
        reference_exposure_share=0.5,
        source_public_status="not_yet_public",
    )
    defense = {
        "component_id": "ultimate_defense_after_release",
        "amount": 6,
        "nature": "resource_cost",
        "lab_loss_weights": {"lab": 1},
    }
    if campaign_cost:
        before["kernel"]["campaign_components"] = [defense]
        before["kernel"]["cluster"]["support"][0]["count"] = 2
    else:
        before["kernel"]["member_outcomes"][0]["components"].append(defense)
    after = deepcopy(before)
    after.update(start=0.5, end=1, source_public_status="public")
    law["segments"] = [before, after]
    draw, unavailable, _execution = _simulate(payload)
    assert not unavailable
    event_welfare, event_lab = (26, 24) if campaign_cost else (16, 15)
    for status in (PUBLIC, NOT_PUBLIC):
        assert draw.views[status + "welfare"].native_mean.value == pytest.approx(
            2 * event_welfare
        )
        assert draw.views[status + "lab_loss::lab"].native_mean.value == pytest.approx(
            2 * event_lab
        )
        # Every campaign and member amount keeps the initiating segment's bucket.
        np.testing.assert_allclose(
            draw.annual_losses[status + "lab_loss::lab"],
            draw.annual_losses[status + "welfare"] * event_lab / event_welfare,
        )
    for measure in ("welfare", "lab_loss::lab"):
        np.testing.assert_allclose(
            draw.annual_losses[measure],
            draw.annual_losses[PUBLIC + measure]
            + draw.annual_losses[NOT_PUBLIC + measure],
        )


def test_unassigned_positive_origin_withholds_breakdown_even_if_unsampled():
    payload = _payload()
    unassigned = payload["joint_states"][0]["laws"][0]
    segment = unassigned["segments"][0]
    for field in ("source_public_status", "source_model_id", "source_model_revision"):
        segment.pop(field)
    unassigned["reference_annual_rate"] = 1e-20
    draw, unavailable, _ = _simulate(payload)
    assert {"welfare", "lab_loss::lab"} <= draw.views.keys()
    assert set(VIEWS[2:]) <= unavailable.keys()
    assert all(
        "unknown source-model public status" in unavailable[view] for view in VIEWS[2:]
    )
    unassigned["reference_annual_rate"] = 0
    draw, unavailable, _ = _simulate(payload)
    assert not unavailable
    np.testing.assert_array_equal(draw.annual_losses[PUBLIC + "welfare"], 0)


def test_missing_lab_mapping_is_local_and_empty_status_is_known_zero():
    payload = _payload()
    laws = payload["joint_states"][0]["laws"]
    laws.pop()
    laws[0]["segments"][0]["kernel"]["member_outcomes"][0]["components"][0][
        "lab_loss_weights"
    ] = None
    draw, unavailable, _ = _simulate(payload)
    assert PUBLIC + "lab_loss::lab" in unavailable
    assert "lab_loss::lab" in unavailable
    assert PUBLIC + "welfare" in draw.views
    for measure in ("welfare", "lab_loss::lab"):
        view = NOT_PUBLIC + measure
        assert view not in unavailable
        assert draw.views[view].native_mean.value == 0
        assert draw.views[view].native_mean.precision["known_zero_law"]


def test_unmodeled_population_without_source_status_withholds_breakdowns():
    payload = _payload()
    absent = deepcopy(payload["populations"][0])
    absent.update(scenario_id="unmodeled")
    payload["populations"].append(absent)
    payload["joint_states"][0]["absent_populations"] = [
        {
            "scenario_id": "unmodeled",
            "disposition": "unavailable",
            "reason": "Incident source status and response law have not been authored.",
        }
    ]
    plan = _plan_for_views().model_copy(
        update={"requested_views": (*VIEWS, "scenario::scenario::gross")}
    )
    draw, unavailable, _ = simulate_incident_state(
        plan,
        compile_incident_state(setup_from_payload(payload), "state"),
        response_sha256="a" * 64,
    )
    assert set(VIEWS) <= unavailable.keys()
    assert "scenario::scenario::gross" in draw.views


def test_heavy_tail_moment_scope_does_not_spread_between_statuses():
    payload = _payload()
    component = payload["joint_states"][0]["laws"][1]["segments"][0]["kernel"][
        "member_outcomes"
    ][0]["components"][0]
    component.pop("amount")
    component["rank_group_id"] = "internal_tail"
    component["severity"] = {
        "representation": "distribution",
        "components": [
            {
                "p5": 1,
                "p25": 2,
                "p50": 3,
                "p75": 4,
                "p95": 5,
                "gpd_shape": 0.75,
                "gpd_scale": 1,
            }
        ],
        "weights": [1],
    }
    draw, unavailable, _ = _simulate(payload)
    assert not unavailable
    for measure in ("welfare", "lab_loss::lab"):
        assert draw.views[PUBLIC + measure].variance_exists
        assert not draw.views[NOT_PUBLIC + measure].variance_exists


def test_deployment_binding_is_exact_and_absent_annotation_is_not_serialized():
    old = setup_from_payload(study_payload())
    assert "deployment_id" not in old.joint_states[0].laws[0].model_dump(mode="json")
    assert not any(
        name in old.joint_states[0].laws[0].segments[0].model_dump(mode="json")
        for name in ("source_public_status", "source_model_id", "source_model_revision")
    )
    incomplete = _payload()
    incomplete["joint_states"][0]["laws"][0]["segments"][0].pop("source_model_revision")
    with pytest.raises(ValidationError, match="supplied together"):
        setup_from_payload(incomplete)
    payload = _payload()
    payload["deployments"][0]["model_id"] = "other-model"
    payload["entities"].append({**payload["entities"][0], "entity_id": "other-model"})
    with pytest.raises(ValidationError, match="originating model"):
        setup_from_payload(payload)
    payload = _payload()
    payload["populations"][0]["deployment_ids"] = ["public"]
    with pytest.raises(ValidationError, match="population scope"):
        setup_from_payload(payload)


def test_embedded_lab_identifier_and_invalid_status():
    payload = _payload()
    payload["entities"][0]["related_entity_ids"] = ["lab::group"]
    payload["entities"][1]["entity_id"] = "lab::group"
    for law in payload["joint_states"][0]["laws"]:
        for component in law["segments"][0]["kernel"]["member_outcomes"][0][
            "components"
        ]:
            component["lab_loss_weights"] = {
                "lab::group": component["lab_loss_weights"]["lab"]
            }
    scoped = "source_status::public::lab_loss::lab::group"
    invalid = "source_status::shared::welfare"
    plan = _plan_for_views().model_copy(
        update={"requested_views": ("welfare", scoped, invalid)}
    )
    draw, unavailable, execution = simulate_incident_state(
        plan,
        compile_incident_state(setup_from_payload(payload), "state"),
        response_sha256="a" * 64,
    )
    mean = draw.views[scoped].native_mean
    assert mean is not None and mean.value == pytest.approx(18)
    bases = execution.view_bases
    assert bases[scoped]["source_public_status"] == "public"
    assert "invalid status or measure" in unavailable[invalid]
    assert invalid not in draw.views


def test_export_and_replay_retain_source_status_peer_measures(tmp_path):
    study = materialize_constant_eci_grid(
        with_execution(
            setup_from_payload(_payload()),
            _plan_for_views(),
            setup_id="source-status-example",
        ),
        context_id="public-context",
        lab_id="lab",
        deployment_id="public",
        entity_id="model",
        scale_id="own_score",
        integer_min=0,
        integer_max=2,
        forecast_weights={0: 0.5, 1: 0, 2: 0.5},
    )
    package = build_setup(
        destination=tmp_path / "package",
        setup=study,
        code_commit="fixture",
    )
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")
    captured = ConditionalCapture(package, tmp_path / "run/capture")
    for conditional in (
        captured.kernel_at("public-context", eci, "reference") for eci in (0, 1, 2)
    ):
        for measure in ("welfare", "lab_loss::lab"):
            total, weights = conditional.samples(measure)
            public, public_weights = conditional.samples(PUBLIC + measure)
            internal, internal_weights = conditional.samples(NOT_PUBLIC + measure)
            np.testing.assert_allclose(total, public + internal)
            np.testing.assert_array_equal(weights, public_weights)
            np.testing.assert_array_equal(weights, internal_weights)
    original = {
        path: path.read_bytes()
        for path in (tmp_path / "run/capture").rglob("*")
        if path.is_file()
    }
    replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run/capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    assert all(path.read_bytes() == value for path, value in original.items())
    output = export_results(
        setup_path=package.root,
        capture_directory=tmp_path / "run/capture",
        results_path=tmp_path / "replay.json",
        output=tmp_path / "export",
    )
    assert load_results_export(tmp_path / "export") == output
    for row in output.series:
        if row.view.startswith("source_status::"):
            assert row.status == "available"
            assert row.basis is not None
            assert row.basis["source_public_status"] in {"public", "not_yet_public"}
            assert row.measure == (
                "lab_economic_loss" if "lab_loss" in row.view else "societal_harm"
            )
            assert row.lab_id == ("lab" if "lab_loss" in row.view else None)
