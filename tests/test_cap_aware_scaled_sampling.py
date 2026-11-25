"""A cap-resolved rank keeps its completed amount without evaluating the raw tail."""

import numpy as np
import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.contracts.components import ComponentAmountLaw
from catmodel.contracts.native_severity import native_severity_from_sampler
from catmodel.probability.distributions import Distribution
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_runtime import simulate_incident_state
from test_support.components import sample_component_amounts
from test_support.run_setup import setup_from_payload
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


def test_cap_resolved_rank_preserves_completed_amount_without_uncapped_inverse(
    monkeypatch,
):
    law = ComponentAmountLaw(
        component_id="loss",
        severity=native_severity_from_sampler(_source()),
        rank_group_id="severity",
        post_multiplier=1e3,
    )

    def unused(*args, **kwargs):
        raise AssertionError("a cap-resolved rank evaluated the raw tail")

    monkeypatch.setattr(Distribution, "sample_from_uniforms_uncapped", unused)
    sampled = sample_component_amounts(
        (law,), np.random.default_rng(0), rank_uniforms={"severity": 0.99}
    )
    assert sampled[0].amount == 500


def test_capped_heavy_tail_native_execution_preserves_counts_and_amounts():
    compiled = compile_incident_state(_case(), "low")
    direct, _, direct_audit = simulate_incident_state(
        _execution_plan(), compiled, response_sha256="a" * 64
    )
    total = direct_audit.total_initiators
    assert isinstance(total, int) and total > 0
    assert np.sum(direct.annual_losses["gross"]) == pytest.approx(
        total * 500, rel=5e-14, abs=0
    )
    assert direct.views["gross"].native_mean is not None
    assert direct.views["gross"].native_mean.value == pytest.approx(
        2500, rel=5e-14, abs=0
    )
