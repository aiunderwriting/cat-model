"""End-to-end frozen incident execution and capture sufficiency oracles."""

import hashlib
import json
import socket

import numpy as np
import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.results.execution import (
    load_run_capture,
    replay_capture,
    run_setup,
)
from catmodel.results.reduction import ReductionSpec
from catmodel.results.results_file import load_results
from catmodel.setup.store import load_setup
from test_support.results import (
    available_mapping,
    available_metric,
    compared_value,
    present,
    reduced_arm,
)
from test_support.run_setup import setup_from_payload
from test_support.study import (
    _count_control_study,
    _financial_study,
    _package,
    _plan,
    _study,
)


def test_frozen_run_replay_reweight_and_current_reducer_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda *_: (_ for _ in ()).throw(AssertionError("network is forbidden")),
    )
    package = _package(tmp_path, study=_study(response_cost=10), plan=_plan(3000))
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="producer"
    )
    precision = available_metric(
        reduced_arm(report["arms"]["reference"])["metrics"]["gross"]
    )["eal_precision"]
    assert precision["precision_basis"] == "deterministic_integration"
    capture = tmp_path / "run" / "capture"
    draws, metadata = load_run_capture(capture)
    assert set(draws) == {"reference", "contained"}
    assert all(len(arm) == 2 for arm in draws.values())
    assert metadata.physical_joint_semantics == "physical_joint_samples"
    assert metadata.grain == "annual"
    original = {
        path: path.read_bytes() for path in capture.rglob("*") if path.is_file()
    }
    replay = replay_capture(
        setup_path=package.root,
        capture_directory=capture,
        output=tmp_path / "replay.json",
        code_commit="current-reducer",
    )
    assert replay["arms"] == report["arms"]
    assert replay["analysis_sha256"] != report["analysis_sha256"]
    assert replay["producer_provenance"]["commit"] == "producer"
    weighted = replay_capture(
        setup_path=package.root,
        capture_directory=capture,
        output=tmp_path / "weighted.json",
        state_weights={"low": 1, "high": 0},
        code_commit="current-reducer",
        reduction_spec=ReductionSpec(levels=(0.95,)),
    )
    assert weighted["analysis_sha256"] != replay["analysis_sha256"]
    assert present(
        available_metric(
            reduced_arm(weighted["arms"]["reference"])["metrics"]["gross"]
        )["expected_annual_loss"]
    ) < present(
        available_metric(reduced_arm(report["arms"]["reference"])["metrics"]["gross"])[
            "expected_annual_loss"
        ]
    )
    assert all(path.read_bytes() == data for path, data in original.items())
    json.dumps(weighted, allow_nan=False)


def test_weight_only_new_package_reuses_but_changed_kernel_refuses(tmp_path):
    package = _package(tmp_path)
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="producer")
    changed = encode_scientific(package)
    changed["title"] = "A presentation-only edit"
    changed["entities"][0]["status"] = "actual"
    changed["joint_states"][0]["weight"] = 0.6
    changed["joint_states"][1]["weight"] = 0.4
    new = _package(tmp_path, setup_from_payload(changed), name="reweighted-package")
    replay_capture(
        setup_path=new.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "reweighted.json",
        code_commit="reducer",
    )
    changed["joint_states"][0]["laws"][0]["segments"][0]["kernel"]["member_outcomes"][
        0
    ]["components"][0]["amount"] = 101
    changed_package = _package(
        tmp_path, setup_from_payload(changed), name="changed-package"
    )
    with pytest.raises(ValueError, match="conditional law"):
        replay_capture(
            setup_path=changed_package.root,
            capture_directory=tmp_path / "run" / "capture",
            output=tmp_path / "changed.json",
            code_commit="reducer",
        )


def test_new_positive_uncaptured_state_refused(tmp_path):
    package = _package(tmp_path)
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="producer")
    payload = encode_scientific(package)
    payload["joint_states"].append(
        {**payload["joint_states"][0], "state_id": "new", "weight": 0}
    )
    new = _package(tmp_path, setup_from_payload(payload), name="new-support")
    with pytest.raises(ValueError, match="newly positive"):
        replay_capture(
            setup_path=new.root,
            capture_directory=tmp_path / "run" / "capture",
            output=tmp_path / "missing.json",
            state_weights={"low": 0.2, "high": 0.3, "new": 0.5},
            code_commit="reducer",
        )


def test_original_evidence_bytes_required_and_exact_package_closure(tmp_path):
    data = b"Explicitly authored source evidence; no invented observation.\n"
    digest = hashlib.sha256(data).hexdigest()
    payload = encode_scientific(_study())
    payload["evidence"] = [
        {
            "evidence_id": "source",
            "origin": "stipulation",
            "original_sha256": digest,
            "selected_sha256": digest,
            "target": "Attempted campaign rate.",
            "conditioning": "Selected origin population.",
            "units": "per study year",
            "transport": "None; authored for this target.",
            "review": "Software fixture",
            "reliance": "None",
            "rights": "Authored fixture",
        }
    ]
    study = setup_from_payload(payload)
    with pytest.raises(ValueError, match="original bytes"):
        _package(tmp_path, study)
    package = _package(tmp_path, study, evidence={"source.txt": data})
    assert package.files["evidence/source.txt"] == data
    (package.root / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="untracked"):
        load_setup(package.root)


def test_view_specific_moments_and_bounded_financial_transform_survive_storage(
    tmp_path,
):
    study = _financial_study(shape=1.2, nature="transfer")
    plan = _plan(200).model_copy(
        update={
            "required_arms": ("reference",),
            "requested_views": ("gross", "welfare", "net_resource_cost", "financial"),
        }
    )
    package = _package(tmp_path, study, plan)
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="producer"
    )
    metrics = reduced_arm(report["arms"]["reference"])["metrics"]
    assert available_metric(metrics["gross"])["expected_annual_loss"] is None
    assert available_metric(metrics["gross"])["tail_metrics"]["0.99"]["tvar"] is None
    assert available_metric(metrics["gross"])["tail_metrics"]["0.99"]["var"] is not None
    assert available_metric(metrics["welfare"])["expected_annual_loss"] == 0
    assert available_metric(metrics["net_resource_cost"])["expected_annual_loss"] == 0
    draws, _ = load_run_capture(tmp_path / "run" / "capture")
    for draw in draws["reference"].values():
        assert {name: view.mean_exists for name, view in draw.views.items()} == {
            "financial": True,
            "gross": False,
            "net_resource_cost": True,
            "welfare": True,
        }
        assert {name: view.variance_exists for name, view in draw.views.items()} == {
            "financial": True,
            "gross": False,
            "net_resource_cost": True,
            "welfare": True,
        }
    replay = replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "replay.json",
        code_commit="reducer",
    )
    assert reduced_arm(replay["arms"]["reference"])["metrics"] == metrics


def test_common_outputs_cannot_be_nested_inside_packages_or_captures(tmp_path):
    package = _package(tmp_path)
    with pytest.raises(ValueError, match="immutable evidence"):
        run_setup(
            setup_path=package.root,
            output=package.root / "nested",
            code_commit="producer",
        )
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="producer")
    with pytest.raises(ValueError, match="immutable evidence"):
        replay_capture(
            setup_path=package.root,
            capture_directory=tmp_path / "run" / "capture",
            output=tmp_path / "run" / "capture" / "nested.json",
            code_commit="reducer",
        )


def test_reduction_retains_original_and_current_evidence_assessments(tmp_path):
    source = b"An openly authored stipulation.\n"
    digest = hashlib.sha256(source).hexdigest()
    payload = encode_scientific(_study())
    payload["evidence"] = [
        {
            "evidence_id": "source",
            "origin": "stipulation",
            "original_sha256": digest,
            "selected_sha256": digest,
            "target": "Attempted campaigns.",
            "conditioning": "Complete conditional state.",
            "units": "per year",
            "transport": "Authored for this target.",
            "review": "Producer assessment",
            "reliance": "Internal fixture only",
            "rights": "Authored fixture",
        }
    ]
    package = _package(
        tmp_path, setup_from_payload(payload), evidence={"source.txt": source}
    )
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="producer")
    payload["evidence"][0]["review"] = "Current assessment"
    current = _package(
        tmp_path,
        setup_from_payload(payload),
        name="current",
        evidence={"source.txt": source},
    )
    report = replay_capture(
        setup_path=current.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "current.json",
        code_commit="current-reducer",
    )
    provenance = report["analysis_spec"]["capture_provenance"]
    assert (
        provenance["producer_assessment"]["evidence"][0]["review"]
        == "Producer assessment"
    )
    assert (
        provenance["current_assessment"]["evidence"][0]["review"]
        == "Current assessment"
    )
    assert provenance["producer_provenance"]["commit"] == "producer"
    assert (
        provenance["current_assessment"]["assessor_provenance"]["commit"]
        == "current-reducer"
    )


def test_frozen_plan_requires_transitive_named_control_closure(tmp_path):
    study = _count_control_study()
    plan = _plan(200).model_copy(update={"required_arms": ("reference", "stress")})
    with pytest.raises(ValueError, match=r"matched controls.*control-a.*control-b"):
        _package(tmp_path, study, plan, name="missing-both")
    assert not (tmp_path / "missing-both").exists()
    plan = plan.model_copy(
        update={"required_arms": ("reference", "stress", "control-a")}
    )
    with pytest.raises(ValueError, match=r"matched controls.*control-b"):
        _package(tmp_path, study, plan, name="missing-transitive")
    assert not (tmp_path / "missing-transitive").exists()


def test_run_captures_controls_and_reports_designated_control_contrasts(tmp_path):
    study = _count_control_study()
    plan = _plan(300).model_copy(
        update={"required_arms": ("reference", "stress", "control-a", "control-b")}
    )
    package = _package(tmp_path, study, plan)
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="producer"
    )
    assert report["status"] == "complete"
    assert report["comparison_reference_arms"]["stress"] == "control-a"
    assert report["comparison_reference_arms"]["control-a"] == "control-b"
    metrics = report["arms"]

    contrasts = available_mapping(report["comparisons"]["stress"]["gross"])
    contrast = compared_value(contrasts["expected_annual_loss"])
    assert (
        contrast["reference"]
        == available_metric(reduced_arm(metrics["control-a"])["metrics"]["gross"])[
            "expected_annual_loss"
        ]
    )
    assert (
        contrast["alternative"]
        == available_metric(reduced_arm(metrics["stress"])["metrics"]["gross"])[
            "expected_annual_loss"
        ]
    )
    assert "stress" in report["reference_comparisons"]
    captures, _ = load_run_capture(tmp_path / "run" / "capture")
    assert set(captures) == set(plan.required_arms)


def test_unavailable_population_is_not_zero_and_reweight_rechecks_scope(tmp_path):
    payload = encode_scientific(_study())
    payload["populations"].append(
        {**payload["populations"][0], "scenario_id": "unestimated"}
    )
    for state in payload["joint_states"]:
        state["weight"] = 1 if state["state_id"] == "low" else 0
        state["absent_populations"] = [
            {
                "scenario_id": "unestimated",
                "disposition": "structural_zero"
                if state["state_id"] == "low"
                else "unavailable",
                "reason": "No selected activity in this state."
                if state["state_id"] == "low"
                else "No incident law has been estimated.",
            }
        ]
    study = setup_from_payload(payload)
    plan = _plan(200).model_copy(
        update={
            "requested_views": (
                "gross",
                "scenario::loss::gross",
                "scenario::unestimated::gross",
            )
        }
    )
    package = _package(tmp_path, study, plan)
    initial = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="producer"
    )
    first = reduced_arm(initial["arms"]["reference"])["metrics"]
    assert first["gross"]["status"] == "available"
    assert (
        available_metric(first["scenario::unestimated::gross"])["expected_annual_loss"]
        == 0
    )
    weighted = replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "weighted.json",
        state_weights={"low": 0.5, "high": 0.5},
        code_commit="reducer",
    )
    metrics = reduced_arm(weighted["arms"]["reference"])["metrics"]
    assert metrics["scenario::loss::gross"]["status"] == "available"
    assert metrics["scenario::unestimated::gross"]["status"] == "unavailable"
    assert metrics["gross"]["status"] == "unavailable"
    assert any(
        "No incident law" in reason
        for reason in metrics["scenario::unestimated::gross"]["state_reasons"].values()
    )


@pytest.mark.parametrize("levels", [(), (0.95,)])
@pytest.mark.parametrize("reweighted", [False, True], ids=["replay", "reweight"])
def test_cli_reduction_matches_the_deterministic_service_reduction(
    tmp_path, reweighted, levels
):
    from click.testing import CliRunner

    from catmodel.cli import cli
    from catmodel.results.reduction import DEFAULT_TAIL_LEVELS

    package = _package(tmp_path, plan=_plan(64))
    produced = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="same"
    )
    capture = tmp_path / "run" / "capture"
    output = tmp_path / "reduced"
    arguments = [
        "results",
        "replay",
        "--setup",
        str(package.root),
        "--capture",
        str(capture),
        "--output",
        str(output),
    ]
    for level in levels:
        arguments.extend(("--level", str(level)))
    if reweighted:
        weights = tmp_path / "weights.json"
        weights.write_text(
            json.dumps({state.state_id: state.weight for state in package.joint_states})
        )
        arguments.extend(("--weights", str(weights)))
    result = CliRunner().invoke(cli, arguments)
    assert result.exit_code == 0, result.output
    report = load_results(output / "results.json")
    # A requested level is reduced beside the default levels.
    expected = replay_capture(
        setup_path=package.root,
        capture_directory=capture,
        output=tmp_path / "service.json",
        code_commit="same",
        reduction_spec=ReductionSpec(
            levels=tuple(sorted({*DEFAULT_TAIL_LEVELS, *levels}))
        )
        if levels
        else None,
    )
    assert report["arms"] == expected["arms"]
    # Design-based precision draws no random numbers, so no reduction seed.
    assert "seed" not in report["analysis_spec"]["reduction_spec"]
    if not levels:
        assert report["arms"] == produced["arms"]


def test_capture_has_one_manifest_and_shared_support_with_derived_masses(tmp_path):
    package = _package(tmp_path)
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="producer")
    directory = tmp_path / "run" / "capture"
    draws, capture = load_run_capture(directory)
    assert {path.name for path in directory.rglob("*.json")} == {"run-capture.json"}
    assert len(capture.supports) == 1
    payload = json.loads((directory / "run-capture.json").read_bytes())
    assert set(next(iter(payload["states"].values()))) == {
        "state_id",
        "epistemic_group_id",
    }
    for arm, rows in capture.arms.items():
        for state, row in rows.items():
            assert set(row.execution).isdisjoint(
                {"view_bases", "functional_unavailable_by_view", "stream_seed"}
            )
            assert set(row.draw.model_dump()).isdisjoint(
                {"stratum_sha256", "response_sha256", "seed", "normalized_masses"}
            )
            np.testing.assert_array_equal(
                draws[arm][state].normalized_weights(),
                np.full(
                    package.execution.simulations, 1 / package.execution.simulations
                ),
            )
    payload["states"][next(iter(payload["states"]))]["epistemic_group_id"] = "changed"
    (directory / "run-capture.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="metadata checksum"):
        load_run_capture(directory)


@pytest.mark.parametrize("change", ["untracked", "missing", "numeric", "symlink"])
def test_single_capture_enforces_numeric_closure_and_immutable_paths(tmp_path, change):
    package = _package(tmp_path)
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="producer")
    directory = tmp_path / "run" / "capture"
    array = next(directory.rglob("*.npz"))
    if change == "untracked":
        (directory / "extra").write_text("untracked")
    elif change == "missing":
        array.unlink()
    elif change == "numeric":
        array.write_bytes(array.read_bytes() + b"changed")
    else:
        original = tmp_path / "original.npz"
        array.rename(original)
        array.symlink_to(original)
    with pytest.raises(ValueError, match=r"untracked|checksum|symlink"):
        load_run_capture(directory)


def test_same_capture_spec_and_reducer_have_identical_analysis_identity(tmp_path):
    package = _package(tmp_path)
    initial = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="same-reducer"
    )
    replay = replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "replay.json",
        code_commit="same-reducer",
    )
    assert initial["analysis_sha256"] == replay["analysis_sha256"]
    assert initial["arms"] == replay["arms"]
