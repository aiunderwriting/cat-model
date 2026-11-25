"""Public reweighting validates target weights before writing any result."""

import json
from pathlib import Path
from typing import Any, cast

import pytest
from click.testing import CliRunner

from catmodel.cli import cli
from catmodel.contracts.run_setup import RunSetup
from catmodel.results.execution import replay_capture, run_setup
from test_support.study import _package, _plan


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> tuple[RunSetup, Path]:
    root = tmp_path_factory.mktemp("weight-validation")
    package = _package(root, plan=_plan(simulations=4))
    run_setup(setup_path=package.root, output=root / "run", code_commit="fixture")
    return package, root / "run/capture"


def _weights(package: RunSetup) -> dict[str, float]:
    return {state.state_id: state.weight for state in package.joint_states}


def test_cli_rejects_oversized_json_weight_cleanly(package, tmp_path):
    package, capture = package
    weights = _weights(package)
    weights[next(iter(weights))] = 10**400
    weights_file = tmp_path / "weights.json"
    weights_file.write_text(json.dumps(weights))
    output = tmp_path / "uncreated" / "replay"
    result = CliRunner().invoke(
        cli,
        [
            "results",
            "replay",
            "--setup",
            str(package.root),
            "--capture",
            str(capture),
            "--weights",
            str(weights_file),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 1
    assert "Error: target state weights must be finite" in result.output
    assert not output.parent.exists()


@pytest.mark.parametrize(
    "invalid",
    [
        "unknown",
        "unknown_string",
        "string",
        "bool",
        "huge",
        "negative",
        "nonsumming",
        "not_mapping",
    ],
)
def test_public_weights_fail_cleanly_without_creating_files(package, tmp_path, invalid):
    package, capture = package
    # Invalid callers deliberately violate the public mapping/value annotations.
    weights: Any = _weights(package)
    state = next(iter(weights))
    if invalid == "unknown":
        weights["unknown"] = weights.pop(state)
    elif invalid == "unknown_string":
        weights.pop(state)
        weights["unknown"] = "invalid"
    elif invalid == "string":
        weights[state] = "invalid"
    elif invalid == "bool":
        weights[state] = True
    elif invalid == "huge":
        weights[state] = 10**400
    elif invalid == "negative":
        weights[state] = -1
    elif invalid == "nonsumming":
        weights[state] = 0
    else:
        weights = cast(Any, [])
    output = tmp_path / "uncreated" / "result.json"
    with pytest.raises(ValueError, match=r"(target state weights|target weights)"):
        replay_capture(
            setup_path=package.root,
            capture_directory=capture,
            output=output,
            code_commit="fixture",
            state_weights=weights,
        )
    assert not output.parent.exists()
