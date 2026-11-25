"""Positive scientific intensity must not become an invented exact-zero law."""

import math
from copy import deepcopy
from decimal import Decimal

import pytest

import catmodel.simulation.incident as compiler
from catmodel.contracts.incident import (
    EntityPath,
    LocalResponse,
)
from test_support.run_setup import setup_from_payload
from test_support.study import study_payload


def _constant_response(payload, multiplier):
    response = payload["joint_states"][0]["laws"][0]["response"]
    response.update(
        interpolation="linear", points=[{"value": 0, "multiplier": multiplier}]
    )


def test_positive_intensity_below_float_range_is_not_a_zero_frequency_audit():
    payload = study_payload()
    law = payload["joint_states"][0]["laws"][0]
    law["reference_annual_rate"] = 1e-200
    law["segments"][0]["reference_exposure_share"] = 1e-200
    assert Decimal("1e-200") * Decimal("1e-200") > 0
    with pytest.raises(
        ValueError, match=r"positive integrated initiating intensity.*numerical.*range"
    ):
        compiler.compile_incident_state(setup_from_payload(payload), "state")


def test_representable_intensity_recovers_from_intermediate_underflow():
    payload = study_payload()
    law = payload["joint_states"][0]["laws"][0]
    law["reference_annual_rate"] = 1e-200
    law["segments"][0]["reference_exposure_share"] = 1e-200
    _constant_response(payload, 1e300)
    compiled = compiler.compile_incident_state(setup_from_payload(payload), "state")
    assert compiled.incidents[0].annual_rate == pytest.approx(1e-100, rel=2e-15, abs=0)
    assert (
        compiled.frequency_audit[0]["integrated_annual_rate"]
        == compiled.incidents[0].annual_rate
    )


def test_representable_intensity_recovers_from_intermediate_division_overflow():
    payload = study_payload()
    law = payload["joint_states"][0]["laws"][0]
    law["reference_annual_rate"] = 1e300
    first = law["segments"][0]
    first["end"] = 1e-200
    second = deepcopy(first)
    second.update(start=1e-200, end=1, reference_exposure_share=0)
    law["segments"].append(second)
    _constant_response(payload, 1)
    compiled = compiler.compile_incident_state(setup_from_payload(payload), "state")
    assert compiled.incidents[0].annual_rate == pytest.approx(1e300, rel=2e-15)


@pytest.mark.parametrize(
    "zero_field",
    ["reference_annual_rate", "reference_exposure_share", "initiation_probability"],
)
def test_authored_zero_skips_unused_response_evaluation(monkeypatch, zero_field):
    payload = study_payload()
    law = payload["joint_states"][0]["laws"][0]
    if zero_field == "reference_annual_rate":
        law[zero_field] = 0
    else:
        law["segments"][0][zero_field] = 0
    if zero_field == "initiation_probability":
        law["input_stage"] = "attempted_initiator"
    monkeypatch.setattr(
        compiler,
        "_integrate_local_response",
        lambda *_: pytest.fail("unused response evaluated"),
    )
    compiled = compiler.compile_incident_state(setup_from_payload(payload), "state")
    assert compiled.incidents[0].annual_rate == 0
    assert compiled.frequency_audit[0]["integrated_annual_rate"] == 0


def test_authored_zero_response_stays_zero_with_other_extreme_factors():
    payload = study_payload()
    payload["joint_states"][0]["laws"][0]["reference_annual_rate"] = 1e300
    law = payload["joint_states"][0]["laws"][0]
    # The capped last-slope continuation is exactly zero from x = 2 upward.
    law["response"].update(
        interpolation="linear",
        points=[{"value": 0, "multiplier": 2}, {"value": 1, "multiplier": 1}],
    )
    for knot in payload["joint_states"][0]["entity_paths"][0]["knots"]:
        knot["value"] = 5
    compiled = compiler.compile_incident_state(setup_from_payload(payload), "state")
    assert compiled.incidents[0].annual_rate == 0


def _log_response_case(*, linear=False):
    # The capped last-slope continuation reaches log multiplier -1000 at x = 2
    # and holds it beyond, below the floating-point range of the multiplier.
    payload = study_payload()
    path = payload["joint_states"][0]["entity_paths"][0]
    path["interpolation"] = "linear" if linear else "step"
    path["knots"][0]["value"] = 1000
    path["knots"][1]["value"] = 0 if linear else 1000
    response = payload["joint_states"][0]["laws"][0]["response"]
    response["points"] = [
        {"value": 0, "multiplier": 1},
        {"value": 1, "multiplier": math.exp(-500)},
    ]
    return EntityPath.model_validate(path), LocalResponse.model_validate(response)


def test_positive_log_response_underflow_is_numerical_unavailability():
    path, response = _log_response_case()
    with pytest.raises(
        ValueError, match=r"positive integrated local response.*numerical.*range"
    ):
        compiler._integrate_local_response(path, response, 0, 1)


def test_log_path_integral_recovers_finite_exponential_product():
    path, response = _log_response_case(linear=True)
    # x = 1000(1 - t): exp(-500000(1-t)) over the last 0.002 of the year gives
    # (1-exp(-1000))/500000; the capped remainder adds 0.998 exp(-1000).
    assert compiler._integrate_local_response(path, response, 0, 1) == pytest.approx(
        2e-6, rel=2e-14
    )


def _uneven_response_payload(kind):
    payload = study_payload()
    state = payload["joint_states"][0]
    # Log uses the capped last-slope continuation to reach exp(-1000).
    high = 2000 if kind == "log" else 1
    state["entity_paths"][0].update(
        interpolation="step",
        knots=[
            {"time": time, "value": value, "model_revision": "r1"}
            for time, value in ((0, 0), (0.5, high), (1, high))
        ],
    )
    state["laws"][0].update(
        reference_annual_rate=1 if kind == "log" else 1e-30,
        response={
            **state["laws"][0]["response"],
            "interpolation": kind,
            "extrapolation": "last_slope" if kind == "log" else "flat",
            "points": [
                {"value": 0, "multiplier": 1},
                {"value": 1, "multiplier": math.exp(-500) if kind == "log" else 1e-300},
            ],
        },
    )
    return payload


@pytest.mark.parametrize("kind,expected", [("log", 0.5), ("linear", 5e-31)])
def test_negligible_internal_piece_does_not_reject_representable_rate(kind, expected):
    compiled = compiler.compile_incident_state(
        setup_from_payload(_uneven_response_payload(kind)), "state"
    )
    assert compiled.incidents[0].annual_rate == pytest.approx(
        expected, rel=2e-15, abs=0
    )
    assert (
        compiled.frequency_audit[0]["integrated_annual_rate"]
        == compiled.incidents[0].annual_rate
    )


@pytest.mark.parametrize("kind,expected", [("log", 0.5), ("linear", 5e-31)])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("multiplier_scale", [1e-20, 1.0, 1e20])
def test_internal_piece_reversal_and_compensating_units_preserve_total(
    kind, expected, reverse, multiplier_scale
):
    payload = _uneven_response_payload(kind)
    state = payload["joint_states"][0]
    path = state["entity_paths"][0]
    if reverse:
        high = path["knots"][1]["value"]
        for knot, value in zip(path["knots"], (high, 0, 0), strict=True):
            knot["value"] = value
    law = state["laws"][0]
    law["reference_annual_rate"] /= multiplier_scale
    for point in law["response"]["points"]:
        point["multiplier"] *= multiplier_scale
        assert point["multiplier"] > 0
    compiled = compiler.compile_incident_state(setup_from_payload(payload), "state")
    assert compiled.incidents[0].annual_rate == pytest.approx(
        expected, rel=2e-15, abs=0
    )
    assert (
        compiled.frequency_audit[0]["integrated_annual_rate"]
        == compiled.incidents[0].annual_rate
    )


def test_positive_completed_integral_below_float_range_is_still_refused():
    payload = study_payload()
    path = EntityPath.model_validate(payload["joint_states"][0]["entity_paths"][0])
    response = LocalResponse.model_validate(
        {
            **payload["joint_states"][0]["laws"][0]["response"],
            "interpolation": "linear",
            "points": [{"value": 0, "multiplier": math.ulp(0.0)}],
        }
    )
    with pytest.raises(
        ValueError, match=r"positive integrated local response.*numerical.*range"
    ):
        compiler._integrate_local_response(path, response, 0, 0.5)


def test_unrepresentable_distinct_mark_segment_is_not_erased_by_dominant_segment():
    payload = _uneven_response_payload("linear")
    law = payload["joint_states"][0]["laws"][0]
    first = law["segments"][0]
    first.update(end=0.5, reference_exposure_share=0.5)
    second = deepcopy(first)
    second.update(start=0.5, end=1)
    second["kernel"]["member_outcomes"][0]["components"][0]["amount"] = 20
    law["segments"].append(second)
    with pytest.raises(
        ValueError, match=r"positive integrated initiating intensity.*numerical.*range"
    ):
        compiler.compile_incident_state(setup_from_payload(payload), "state")
