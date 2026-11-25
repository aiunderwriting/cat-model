"""Frozen study captures bind alternative laws and complete required arms."""

import json

import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.hashing import canonical_json_sha256
from catmodel.results.execution import (
    CAPTURE_MANIFEST,
    load_run_capture,
    replay_capture,
    run_setup,
)
from test_support.run_setup import setup_from_payload
from test_support.study import _package, _plan, _study


def test_same_alternative_name_cannot_reuse_a_different_conditional_kernel(tmp_path):
    package = _package(tmp_path, plan=_plan(32))
    run = tmp_path / "run"
    run_setup(setup_path=package.root, output=run, code_commit="producer")
    payload = encode_scientific(_study())
    payload["alternatives"][0]["overrides"][0]["law"]["reference_annual_rate"] = 2
    changed = _package(
        tmp_path, setup_from_payload(payload), plan=_plan(32), name="changed"
    )
    with pytest.raises(ValueError, match="conditional law"):
        replay_capture(
            setup_path=changed.root,
            capture_directory=run / "capture",
            output=tmp_path / "uncreated.json",
            code_commit="reducer",
        )
    assert not (tmp_path / "uncreated.json").exists()


def test_restamping_cannot_hide_a_missing_required_arm(tmp_path):
    package = _package(tmp_path, plan=_plan(32))
    run = tmp_path / "run"
    run_setup(setup_path=package.root, output=run, code_commit="producer")
    capture = run / "capture"
    manifest = capture / CAPTURE_MANIFEST
    payload = json.loads(manifest.read_bytes())
    payload["arms"].pop("contained")
    payload["content_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=r"comparison arms.*frozen requirements"):
        load_run_capture(capture)


@pytest.mark.parametrize(
    "change,reason",
    [
        ("seed", "stream seed"),
        ("simulations", "frozen numerical controls"),
        ("state_id", "state identity"),
        ("conditional_key", "conditional law"),
    ],
)
def test_restamping_cannot_override_frozen_capture_controls(tmp_path, change, reason):
    package = _package(tmp_path, plan=_plan(32).model_copy(update={"seed": 19}))
    run = tmp_path / "run"
    run_setup(setup_path=package.root, output=run, code_commit="producer")
    capture = run / "capture"
    manifest = capture / CAPTURE_MANIFEST
    payload = json.loads(manifest.read_bytes())
    arm = package.execution.reference_arm
    state = next(iter(payload["arms"][arm]))
    row = payload["arms"][arm][state]
    if change == "seed":
        row["seed"] += 1
    elif change == "simulations":
        payload["execution_plan"]["simulations"] += 1
    elif change == "state_id":
        payload["states"][state]["state_id"] += "-changed"
    else:
        row["conditional_key"] = "0" * 64
    payload["content_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    manifest.write_text(json.dumps(payload))
    output = tmp_path / "uncreated.json"
    with pytest.raises(ValueError, match=reason):
        replay_capture(
            setup_path=package.root,
            capture_directory=capture,
            output=output,
            code_commit="reducer",
        )
    assert not output.exists()
