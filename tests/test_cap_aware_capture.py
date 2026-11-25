"""Completed capped amounts survive public capture and replay."""

import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.contracts.native_severity import native_severity_from_sampler
from catmodel.probability.distributions import Distribution
from catmodel.results.execution import load_run_capture, replay_capture, run_setup
from catmodel.setup.store import build_setup
from test_support.run_setup import setup_from_payload, with_execution
from test_support.study import _plan, _study


def _source():
    # An infinite-mean raw tail whose cap lies below the whole body.
    return Distribution(1, 2, 3, 4, 5, gpd_shape=1.5, gpd_scale=1, severity_cap=0.5)


def _case():
    payload = encode_scientific(_study(rate=5))
    payload["alternatives"] = []
    payload["joint_states"] = payload["joint_states"][:1]
    payload["joint_states"][0]["weight"] = 1
    for state in payload["joint_states"]:
        kernel = state["laws"][0]["segments"][0]["kernel"]
        kernel["cluster"]["support"] = [{"count": 1, "probability": 1}]
        kernel["member_outcomes"][0]["components"][0].update(
            amount=None,
            severity=native_severity_from_sampler(_source()).model_dump(mode="json"),
            rank_group_id="severity",
            post_multiplier=1e3,
        )
    return setup_from_payload(payload)


def _execution_plan():
    return _plan(8).model_copy(
        update={
            "required_arms": ("reference",),
            "requested_views": ("gross",),
        }
    )


def test_capped_heavy_tail_run_and_replay_preserve_finite_mean(tmp_path):
    package = build_setup(
        destination=tmp_path / "package",
        setup=with_execution(_case(), _execution_plan(), setup_id="fixture"),
        code_commit="fixture",
    )
    original = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    draws, _ = load_run_capture(tmp_path / "run/capture")
    for draw in draws["reference"].values():
        assert any(view.native_mean is not None for view in draw.views.values())
        assert draw.views["gross"].native_mean.value == pytest.approx(
            2500, rel=5e-14, abs=0
        )

        assert draw.views["gross"].mean_exists
    replay = replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run/capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    assert replay["arms"] == original["arms"]
