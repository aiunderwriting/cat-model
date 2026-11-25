"""Rare incident draws preserve optional native severity moments."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.contracts.native_severity import (
    NativeSeverityLaw,
    native_severity_sampler,
)
from catmodel.probability.distributions import Distribution
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_runtime import simulate_incident_state
from test_support.run_setup import setup_from_payload
from test_support.study import _financial_study, _plan


def _study(*, finite_mean=False, shape=1.2):
    payload = encode_scientific(_financial_study(shape=shape))
    payload["alternatives"] = []
    for state in payload["joint_states"]:
        state["laws"][0]["reference_annual_rate"] = 1e-6
        if finite_mean:
            component = state["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0][
                "components"
            ][0]
            severity = NativeSeverityLaw.model_validate(component["severity"])
            component["severity"] = severity.model_copy(
                update={
                    "severity_cap": 100,
                    "representation": "mixture",
                }
            ).model_dump(mode="json")
    return setup_from_payload(payload)


@pytest.mark.parametrize("finite_mean", [False, True])
def test_rare_native_draws_preserve_optional_mean_status(finite_mean):
    study = _study(finite_mean=finite_mean)
    compiled = compile_incident_state(study, "low")
    plan = _plan(1000).model_copy(update={"requested_views": ("gross",), "seed": 7})
    draw, unavailable, allocation = simulate_incident_state(
        plan, compiled, response_sha256=study.scientific_sha256
    )
    assert not unavailable
    assert np.isfinite(draw.annual_losses["gross"]).all()
    assert draw.views["gross"].mean_exists is finite_mean
    if finite_mean:
        severity = (
            study.joint_states[0]
            .laws[0]
            .segments[0]
            .kernel.member_outcomes[0]
            .components[0]
            .severity
        )
        sampler = native_severity_sampler(severity)
        assert any(view.native_mean is not None for view in draw.views.values())
        assert draw.views["gross"].native_mean is not None
        assert draw.views["gross"].native_mean.value == pytest.approx(
            2e-6 * sampler.analytic_mean()
        )
    else:
        assert draw.views["gross"].native_mean is None
    json.dumps(asdict(allocation), allow_nan=False)


def test_unexpected_rare_analytic_mean_errors_remain_fatal(monkeypatch):
    study = _study(shape=0.5)

    def broken_mean(self, **kwargs):
        raise ValueError("unexpected native mean defect")

    monkeypatch.setattr(Distribution, "analytic_mean", broken_mean)
    with pytest.raises(ValueError, match="unexpected native mean defect"):
        simulate_incident_state(
            _plan(1000).model_copy(update={"requested_views": ("gross",)}),
            compile_incident_state(study, "low"),
            response_sha256=study.scientific_sha256,
        )
