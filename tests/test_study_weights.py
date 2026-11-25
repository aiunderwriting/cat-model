"""Public target-weight admission: one vector for every arm or one for each
arm, with malformed JSON kept distinct from either."""

import json
from typing import Any, cast

import pytest
from click.testing import CliRunner

from catmodel.cli import cli
from catmodel.contracts._setup_codec import encode_scientific
from catmodel.results.execution import replay_capture, run_setup
from catmodel.results.results_file import load_results
from catmodel.simulation.incident_runtime import state_key
from test_support.results import available_metric, reduced_arm
from test_support.run_setup import setup_from_payload
from test_support.study import _package, _plan, _study


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    root = tmp_path_factory.mktemp("state-weights")
    package = _package(root, plan=_plan(32))
    run_setup(setup_path=package.root, output=root / "run", code_commit="fixture")
    return root, package


@pytest.mark.parametrize("payload", [[["low", 0.25], ["high", 0.75]], 1, True, "low"])
def test_service_rejects_non_mapping_weights_before_writing_results(
    captured, tmp_path, payload
):
    root, package = captured
    output = tmp_path / "result.json"
    with pytest.raises(ValueError, match="JSON object keyed by state ID"):
        replay_capture(
            setup_path=package.root,
            capture_directory=root / "run" / "capture",
            output=output,
            state_weights=cast(Any, payload),
            code_commit="fixture",
        )
    assert not output.exists()


@pytest.mark.parametrize("payload", [[["low", 0.25], ["high", 0.75]], 1, None])
def test_cli_explains_invalid_json_weight_shape(captured, tmp_path, payload):
    root, package = captured
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps(payload))
    output = tmp_path / "replay"
    result = CliRunner().invoke(
        cli,
        [
            "results",
            "replay",
            "--setup",
            str(package.root),
            "--capture",
            str(root / "run" / "capture"),
            "--output",
            str(output),
            "--weights",
            str(weights),
        ],
    )
    assert result.exit_code == 1
    assert "JSON object in one of two shapes" in result.output
    assert not output.exists()


def test_json_object_weights_reuse_capture_without_resimulation(
    captured, tmp_path, monkeypatch
):
    import catmodel.results.execution as service

    root, package = captured
    monkeypatch.setattr(
        service,
        "simulate_incident_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected resimulation")
        ),
    )
    result = replay_capture(
        setup_path=package.root,
        capture_directory=root / "run" / "capture",
        output=tmp_path / "result.json",
        state_weights={"low": 1.0, "high": 0.0},
        code_commit="fixture",
    )
    assert (
        available_metric(reduced_arm(result["arms"]["reference"])["metrics"]["gross"])[
            "expected_annual_loss"
        ]
        == 200
    )


@pytest.fixture(scope="module")
def captured_selection(tmp_path_factory):
    """A capture of a setup whose ``selection`` arm authors its own state
    weights, all on ``low``, and simulates that state alone. Each campaign
    harms two members for 100 each, at an annual rate of 1 in ``low`` and 2 in
    ``high``, so the states' gross EALs are 200 and 400."""
    root = tmp_path_factory.mktemp("arm-weights")
    payload = encode_scientific(_study())
    payload["alternatives"] = [
        {
            "alternative_id": "selection",
            "purpose": "scientific_alternative",
            "description": "Alternative selection keeps complete kernels fixed.",
            "state_weights": {"low": 1, "high": 0},
        }
    ]
    plan = _plan(32).model_copy(
        update={
            "required_arms": ("reference", "selection"),
            "required_state_ids_by_arm": {"selection": ("low",)},
        }
    )
    package = _package(root, setup_from_payload(payload), plan)
    run_setup(setup_path=package.root, output=root / "run", code_commit="fixture")
    return root, package


def _cli_replay(captured, tmp_path, payload):
    root, package = captured
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps(payload))
    output = tmp_path / "replay"
    result = CliRunner().invoke(
        cli,
        [
            "--workers",
            "1",
            "results",
            "replay",
            "--setup",
            str(package.root),
            "--capture",
            str(root / "run" / "capture"),
            "--weights",
            str(weights),
            "--output",
            str(output),
        ],
    )
    return result, output


def _gross_eal(report, arm):
    return available_metric(reduced_arm(report["arms"][arm])["metrics"]["gross"])[
        "expected_annual_loss"
    ]


def test_cli_refuses_one_changed_vector_where_an_arm_authors_its_weights(
    captured_selection, tmp_path
):
    result, output = _cli_replay(
        captured_selection, tmp_path, {"low": 0.5, "high": 0.5}
    )
    assert result.exit_code == 1
    # The refusal names the arm one vector cannot set and the shape that can.
    assert "selection" in result.output
    assert "keyed by arm ID" in result.output
    assert not output.exists()


def test_cli_accepts_the_setup_vector_where_an_arm_authors_its_weights(
    captured_selection, tmp_path
):
    root, _ = captured_selection
    result, output = _cli_replay(
        captured_selection, tmp_path, {"low": 0.25, "high": 0.75}
    )
    assert result.exit_code == 0, result.output
    report = load_results(output / "results.json")
    assert report["arms"] == load_results(root / "run" / "results.json")["arms"]


def test_cli_per_arm_weights_move_each_arm_to_its_own_target(
    captured_selection, tmp_path
):
    root, _ = captured_selection
    initial = load_results(root / "run" / "results.json")
    assert _gross_eal(initial, "reference") == pytest.approx(0.25 * 200 + 0.75 * 400)
    targets = {
        "reference": {"low": 0.5, "high": 0.5},
        "selection": {"low": 1, "high": 0},
    }
    result, output = _cli_replay(captured_selection, tmp_path, targets)
    assert result.exit_code == 0, result.output
    report = load_results(output / "results.json")
    # Weight moved from the high state to the low one lowers the reference
    # EAL, to the states' equal-weight mean; the selection arm stays on low.
    assert _gross_eal(report, "reference") == pytest.approx((200 + 400) / 2)
    assert _gross_eal(report, "selection") == pytest.approx(200)
    assert report["analysis_spec"]["state_weights_by_arm"] == {
        arm: {state_key(state): weight for state, weight in vector.items()}
        for arm, vector in targets.items()
    }
    assert report["analysis_sha256"] != initial["analysis_sha256"]


_EVEN = {"low": 0.5, "high": 0.5}
_LOW = {"low": 1, "high": 0}


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({}, "one of two shapes"),
        # One arm's vector beside a state's probability mixes the two shapes.
        ({"reference": _EVEN, "low": 1.0}, "one of two shapes"),
        ({"reference": _EVEN, "selection": _LOW, "high": 0.0}, "one of two shapes"),
        ({"reference": _EVEN}, "exactly the required arms"),
        (
            {"reference": _EVEN, "selection": _LOW, "other": _LOW},
            "exactly the required arms",
        ),
        (
            {"reference": {"low": 0.5}, "selection": _LOW},
            "exactly the declared joint states",
        ),
        (
            {"reference": {"low": 0.5, "high": 0.6}, "selection": _LOW},
            "sum to one",
        ),
        (
            {"reference": {"low": {"weight": 0.5}, "high": 0.5}, "selection": _LOW},
            "must be finite",
        ),
        # The selection arm captured the low state alone.
        ({"reference": _EVEN, "selection": _EVEN}, "newly positive"),
    ],
)
def test_cli_refuses_malformed_or_unsupported_arm_weights(
    captured_selection, tmp_path, payload, reason
):
    result, output = _cli_replay(captured_selection, tmp_path, payload)
    assert result.exit_code == 1
    assert reason in result.output
    assert not output.exists()
