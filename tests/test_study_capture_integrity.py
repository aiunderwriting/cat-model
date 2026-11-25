"""Producer controls, exact benchmarks and arm targets retain separate authority."""

import hashlib
import json

import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.hashing import canonical_json_sha256
from catmodel.results.execution import load_run_capture, replay_capture, run_setup
from catmodel.setup.store import load_setup
from test_support.results import available_metric, reduced_arm
from test_support.run_setup import setup_from_payload
from test_support.study import (
    _financial_study,
    _package,
    _plan,
    _study,
    partial_financial_payload,
    study_payload,
)


def restamp_capture(path, change):
    record = json.loads(path.read_text())
    change(record)
    record["content_sha256"] = canonical_json_sha256(
        {key: value for key, value in record.items() if key != "content_sha256"}
    )
    path.write_text(json.dumps(record))


@pytest.mark.parametrize(
    "field,shape", [("mean_exists", 1.2), ("variance_exists", 0.6)]
)
def test_rehashed_finite_moment_claim_is_refused(tmp_path, field, shape):
    package = _package(tmp_path, _financial_study(shape=shape), _plan(16))
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")

    def change(record):
        for row in record["arms"]["reference"].values():
            assert row["draw"]["views"]["gross"][field] is False
            row["draw"]["views"]["gross"][field] = True

    capture = tmp_path / "run/capture"
    restamp_capture(capture / "run-capture.json", change)
    with pytest.raises(ValueError, match="moment availability"):
        replay_capture(
            setup_path=package.root,
            capture_directory=capture,
            output=tmp_path / "replay.json",
            code_commit="fixture",
        )


@pytest.mark.parametrize("change_kind", ["missing", "extra"])
def test_rehashed_observable_roster_is_refused(tmp_path, change_kind):
    package = _package(tmp_path, plan=_plan(16))
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")

    def change(record):
        for row in record["arms"]["reference"].values():
            views = row["draw"]["views"]
            if change_kind == "missing":
                del views["welfare"]
            else:
                views["scenario::invented::gross"] = views["gross"]

    capture = tmp_path / "run/capture"
    restamp_capture(capture / "run-capture.json", change)
    with pytest.raises(ValueError, match="observable admission"):
        replay_capture(
            setup_path=package.root,
            capture_directory=capture,
            output=tmp_path / "replay.json",
            code_commit="fixture",
        )


@pytest.mark.parametrize(
    "claim", ["unavailable_views", "unavailable_functionals", "view_bases"]
)
def test_rehashed_support_claim_is_refused(tmp_path, claim):
    plan = _plan(16).model_copy(
        update={
            "required_arms": ("reference",),
            "requested_views": ("gross", "financial"),
        }
    )
    package = _package(tmp_path, setup_from_payload(partial_financial_payload()), plan)
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")

    def change(record):
        old_key, support = next(iter(record["supports"].items()))
        if claim == "unavailable_views":
            support[claim] = {}
        elif claim == "unavailable_functionals":
            support[claim] = {"gross": {"expected_annual_loss": "invented limitation"}}
        else:
            support[claim]["pooled"]["currency"] = "EUR"
        new_key = canonical_json_sha256(support)
        record["supports"] = {new_key: support}
        for rows in record["arms"].values():
            for row in rows.values():
                assert row["support_sha256"] == old_key
                row["support_sha256"] = new_key

    capture = tmp_path / "run/capture"
    restamp_capture(capture / "run-capture.json", change)
    with pytest.raises(ValueError, match=r"(captured support|functional support)"):
        replay_capture(
            setup_path=package.root,
            capture_directory=capture,
            output=tmp_path / "replay.json",
            code_commit="fixture",
        )


@pytest.mark.parametrize("case", ["underflow", "exact_zero", "bounded_financial"])
def test_replay_preserves_law_and_numeric_moment_distinctions(tmp_path, case):
    views = ("gross",)
    if case == "bounded_financial":
        payload = encode_scientific(_financial_study(shape=1.2))
        payload["contracts"][0]["annual_limit"] = 5
        views = ("gross", "financial")
    else:
        payload = study_payload()
        law = payload["joint_states"][0]["laws"][0]
        law["reference_annual_rate"] = 1e-316 if case == "underflow" else 0
        law["segments"][0]["kernel"]["member_outcomes"][0]["components"][0][
            "amount"
        ] = 1e-9
    plan = _plan(16).model_copy(
        update={"required_arms": ("reference",), "requested_views": views}
    )
    package = _package(tmp_path, setup_from_payload(payload), plan)
    original = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    draws, capture = load_run_capture(tmp_path / "run/capture")
    for draw in draws["reference"].values():
        assert draw.views["gross"].mean_exists is (case != "bounded_financial")
        if case == "underflow":
            assert not any(view.native_mean is not None for view in draw.views.values())
            assert not draw.views["gross"].variance_exists
            assert all(
                "numeric output range"
                in support.unavailable_functionals["gross"]["expected_annual_loss"]
                for support in capture.supports.values()
            )
        elif case == "exact_zero":
            assert draw.views["gross"].native_mean.precision["known_zero_law"] is True
        else:
            assert draw.views["financial"].mean_exists
            assert draw.views["financial"].variance_exists
    replay = replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run/capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    assert replay["arms"] == original["arms"]


def test_replay_reuses_actual_producer_controls_and_identifies_current_request(
    tmp_path,
):
    original = _package(
        tmp_path, _study(), _plan(200).model_copy(update={"seed": 1}), name="producer"
    )
    run_setup(setup_path=original.root, output=tmp_path / "run", code_commit="fixture")
    selected = _package(
        tmp_path,
        original,
        original.execution.model_copy(update={"simulations": 20000, "seed": 999}),
        name="selected",
    )
    replay = replay_capture(
        setup_path=selected.root,
        capture_directory=tmp_path / "run/capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    provenance = replay["analysis_spec"]["capture_provenance"]
    assert provenance["producer_execution_plan"]["simulations"] == 200
    assert provenance["producer_execution_plan"]["seed"] == 1
    assert replay["analysis_spec"]["plan_sha256"] == original.execution_plan_sha256
    assert provenance["producer_execution_plan"] == original.execution.model_dump(
        mode="json"
    )
    assert provenance["selected_setup_plan_sha256"] == selected.execution_plan_sha256
    assert "seed" not in replay["analysis_spec"]["reduction_spec"]
    draws, _ = load_run_capture(tmp_path / "run/capture")
    assert {
        draw.n_simulations for rows in draws.values() for draw in rows.values()
    } == {200}


@pytest.mark.parametrize(
    "mutation", ["value", "estimator", "precision", "known_zero", "missing"]
)
def test_rehashed_false_native_analytic_benchmark_is_refused(tmp_path, mutation):
    package = _package(tmp_path, _study(rate=171.6 / 350), _plan(50))
    original = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    assert available_metric(
        reduced_arm(original["arms"]["reference"])["metrics"]["gross"]
    )["expected_annual_loss"] == pytest.approx(171.6)

    def change(record):
        for rows in record["arms"].values():
            for row in rows.values():
                for view in row["draw"]["views"].values():
                    estimate = view["native_mean"]
                    if estimate is None:
                        continue
                    if mutation == "value":
                        estimate["value"] /= 10
                    elif mutation == "estimator":
                        estimate["estimator"] = "unsupported_analytic_estimator"
                    elif mutation == "precision":
                        estimate["precision"]["precision_basis"] = "unverified"
                    elif mutation == "known_zero":
                        estimate["precision"]["known_zero_law"] = True
                    else:
                        view["native_mean"] = None

    capture = tmp_path / "run/capture"
    restamp_capture(capture / "run-capture.json", change)
    for label, weights in (("replay", None), ("reweight", {"low": 0.5, "high": 0.5})):
        with pytest.raises(ValueError, match="native EAL"):
            replay_capture(
                setup_path=package.root,
                capture_directory=capture,
                output=tmp_path / (label + ".json"),
                state_weights=weights,
                code_commit="fixture",
            )


def test_raw_authored_plan_identity_survives_new_model_defaults(tmp_path):
    package = _package(tmp_path, plan=_plan(20))
    path = package.root / "execution-plan.json"
    authored = json.loads(path.read_bytes())
    for key in (
        "simulations_by_arm",
        "numerical_controls_by_arm",
        "comparison_requirements",
    ):
        authored.pop(key)
    path.write_text(json.dumps(authored))
    manifest_path = package.root / "setup-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"]["execution-plan.json"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    selected = load_setup(package.root)
    expected = canonical_json_sha256(authored)
    assert selected.execution_plan_sha256 == expected
    assert selected.execution.digest() != expected
    run = run_setup(
        setup_path=selected.root, output=tmp_path / "run", code_commit="fixture"
    )
    replay = replay_capture(
        setup_path=selected.root,
        capture_directory=tmp_path / "run/capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    assert (
        run["analysis_spec"]["plan_sha256"]
        == replay["analysis_spec"]["plan_sha256"]
        == expected
    )
    provenance = replay["analysis_spec"]["capture_provenance"]
    assert provenance["producer_execution_plan"] == authored


def test_arm_specific_targets_and_effective_sample_controls_round_trip(tmp_path):
    payload = encode_scientific(_study())
    payload["alternatives"] = [
        {
            "alternative_id": "selection",
            "purpose": "scientific_alternative",
            "description": "Scientific opinion-pool alternative with every conditional law fixed.",
            "state_weights": {"low": 1, "high": 0},
        }
    ]
    plan = _plan(200).model_copy(
        update={
            "required_arms": ("reference", "selection"),
            "simulations_by_arm": {"selection": 80},
            "required_state_ids_by_arm": {"selection": ("low",)},
            "numerical_controls_by_arm": {
                "selection": {"max_campaign_members_per_year": 125}
            },
        }
    )
    package = _package(tmp_path, setup_from_payload(payload), plan)
    run = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    assert (
        available_metric(reduced_arm(run["arms"]["reference"])["metrics"]["gross"])[
            "expected_annual_loss"
        ]
        == 350
    )
    assert (
        available_metric(reduced_arm(run["arms"]["selection"])["metrics"]["gross"])[
            "expected_annual_loss"
        ]
        == 200
    )
    draws, capture = load_run_capture(tmp_path / "run/capture")
    assert {draw.n_simulations for draw in draws["reference"].values()} == {200}
    assert {draw.n_simulations for draw in draws["selection"].values()} == {80}
    assert sum(capture.state_weights_by_arm["selection"].values()) == 1
    assert len(draws["selection"]) == 1
    assert len(draws["reference"]) == 2
    replay = replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run/capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    assert replay["arms"] == run["arms"]
    with pytest.raises(ValueError, match=r"selection.*keyed by arm ID"):
        replay_capture(
            setup_path=package.root,
            capture_directory=tmp_path / "run/capture",
            output=tmp_path / "ambiguous-reweight.json",
            state_weights={"low": 0.5, "high": 0.5},
            code_commit="fixture",
        )
    weighted = replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run/capture",
        output=tmp_path / "explicit-reweight.json",
        state_weights={"low": 0.5, "high": 0.5},
        state_weights_by_arm={
            "reference": {"low": 0.5, "high": 0.5},
            "selection": {"low": 1, "high": 0},
        },
        code_commit="fixture",
    )
    assert (
        available_metric(
            reduced_arm(weighted["arms"]["reference"])["metrics"]["gross"]
        )["expected_annual_loss"]
        == 300
    )
    assert (
        available_metric(
            reduced_arm(weighted["arms"]["selection"])["metrics"]["gross"]
        )["expected_annual_loss"]
        == 200
    )


def test_frozen_arm_support_must_cover_positive_target_states_before_publication(
    tmp_path,
):
    plan = _plan(20).model_copy(
        update={"required_state_ids_by_arm": {"reference": ("low",)}}
    )
    with pytest.raises(ValueError, match="positive arm target weights"):
        _package(tmp_path, plan=plan)
    assert not (tmp_path / "package").exists()


def test_new_positive_arm_state_requires_a_retained_capture(tmp_path):
    payload = encode_scientific(_study())
    payload["alternatives"] = [
        {
            "alternative_id": "selection",
            "purpose": "scientific_alternative",
            "description": "Alternative selection keeps complete kernels fixed.",
            "state_weights": {"low": 1, "high": 0},
        }
    ]
    plan = _plan(20).model_copy(
        update={
            "required_arms": ("reference", "selection"),
            "required_state_ids_by_arm": {"selection": ("low",)},
        }
    )
    package = _package(tmp_path, setup_from_payload(payload), plan)
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")
    with pytest.raises(ValueError, match="newly positive"):
        replay_capture(
            setup_path=package.root,
            capture_directory=tmp_path / "run/capture",
            output=tmp_path / "reweight.json",
            state_weights_by_arm={
                "reference": {"low": 0.25, "high": 0.75},
                "selection": {"low": 0.5, "high": 0.5},
            },
            code_commit="fixture",
        )


def test_uncoordinated_contract_exposures_can_exceed_unique_gross_and_are_labeled(
    tmp_path,
):
    import numpy as np

    from test_support.study import _financial_study

    payload = encode_scientific(_financial_study())
    payload["contracts"] = [
        {"contract_id": identifier, "contract_year": "2027", "entity_id": "model"}
        for identifier in ("first", "second")
    ]
    for state in payload["joint_states"]:
        outcome = state["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0]
        outcome["memberships"] = [
            {
                "component_id": "harm",
                "contract_id": identifier,
                "obligated_entity_id": "model",
                "eligible": True,
            }
            for identifier in ("first", "second")
        ]
    plan = _plan(64).model_copy(update={"requested_views": ("gross", "financial")})
    package = _package(tmp_path, setup_from_payload(payload), plan)
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    draws, _ = load_run_capture(tmp_path / "run/capture")
    for draw in draws["reference"].values():
        np.testing.assert_array_equal(
            draw.annual_losses["financial"], 2 * draw.annual_losses["gross"]
        )
    semantics = report["analysis_spec"]["capture_provenance"][
        "financial_view_semantics"
    ]
    assert semantics["aggregation"] == "SUM_UNCOORDINATED_CONTRACT_EXPOSURE"
    assert "bypass" in semantics["outside_limit_expenses"]
