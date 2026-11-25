"""Coordinate translations and scaled products retain the authored law."""

from decimal import Decimal, localcontext

import pytest

from catmodel.simulation.incident import compile_incident_state
from test_support.run_setup import setup_from_payload
from test_support.study import study_payload


def _translated_path_response(offset, interpolation, path_interpolation, extrapolation):
    payload = study_payload()
    law = payload["joint_states"][0]["laws"][0]
    law["response"].update(
        interpolation=interpolation,
        extrapolation=extrapolation,
        points=[
            {"value": offset + 2, "multiplier": 1},
            {"value": offset + 6, "multiplier": 9 if interpolation == "log" else 3},
        ],
    )
    path = payload["joint_states"][0]["entity_paths"][0]
    path["interpolation"] = path_interpolation
    values = (
        [(0, 0), (0.5, 4), (1, 8)]
        if path_interpolation == "linear"
        else [(0, 0), (0.25, 4), (0.75, 8), (1, 8)]
    )
    path["knots"] = [
        {"time": time, "value": offset + value, "model_revision": "r1"}
        for time, value in values
    ]
    return (
        compile_incident_state(setup_from_payload(payload), "state")
        .incidents[0]
        .annual_rate
    )


@pytest.mark.parametrize("interpolation", ["linear", "log"])
@pytest.mark.parametrize("path_interpolation", ["step", "linear"])
@pytest.mark.parametrize("extrapolation", ["flat", "last_slope"])
def test_path_response_translation_preserves_integrals_across_nodes_and_extrapolation(
    interpolation, path_interpolation, extrapolation
):
    with localcontext() as context:
        context.prec = 70
        # Below the lowest point (2) both continuations are flat; last_slope
        # continues above 6 toward its cap at 10, which the path never reaches.
        if interpolation == "linear":
            if extrapolation == "flat":
                expected = Decimal(2)
            else:
                step = path_interpolation == "step"
                expected = Decimal("2.25") if step else Decimal("2.125")
        elif path_interpolation == "step":
            expected = Decimal(4) if extrapolation == "flat" else Decimal(17) / 2
        elif extrapolation == "flat":
            expected = Decimal("2.5") + Decimal(4) / Decimal(9).ln()
        else:
            expected = Decimal("0.25") + Decimal(26) / (2 * Decimal(9).ln())
    baseline = _translated_path_response(
        0, interpolation, path_interpolation, extrapolation
    )
    translated = _translated_path_response(
        1e16, interpolation, path_interpolation, extrapolation
    )
    assert baseline == pytest.approx(float(expected), rel=2e-14)
    assert translated == pytest.approx(float(expected), rel=2e-14)


@pytest.mark.parametrize("path_interpolation,expected", [("step", 1), ("linear", 2)])
def test_large_anchor_does_not_cancel_constant_or_linear_response(
    path_interpolation, expected
):
    payload = study_payload()
    response = payload["joint_states"][0]["laws"][0]["response"]
    response.update(
        interpolation="linear",
        points=[{"value": 1e16, "multiplier": 1}, {"value": 1e16 + 2, "multiplier": 3}],
    )
    path = payload["joint_states"][0]["entity_paths"][0]
    path["interpolation"] = path_interpolation
    path["knots"][0]["value"] = 1e16
    path["knots"][1]["value"] = 1e16 + 2
    assert (
        compile_incident_state(setup_from_payload(payload), "state")
        .incidents[0]
        .annual_rate
        == expected
    )


def test_nonzero_subnormal_intermediate_does_not_corrupt_composed_intensity():
    payload = study_payload()
    law = payload["joint_states"][0]["laws"][0]
    law["reference_annual_rate"] = 1e-160
    law["segments"][0]["reference_exposure_share"] = 1e-160
    law["response"].update(
        interpolation="linear", points=[{"value": 0, "multiplier": 1e100}]
    )
    with localcontext() as context:
        context.prec = 80
        expected = Decimal.from_float(1e-160) ** 2 * Decimal.from_float(1e100)
    assert compile_incident_state(setup_from_payload(payload), "state").incidents[
        0
    ].annual_rate == pytest.approx(float(expected), rel=2e-15, abs=0)


@pytest.mark.parametrize(
    "coordinates,multipliers,reference_rate",
    [
        ((0.0, 1e16), (1e-308, 3e-308), 1e308),
        ((0.0, 1e-308), (1.0, 3.0), 1.0),
        ((-1e308, 1e308), (1.0, 3.0), 1.0),
        ((0.0, 5e-324), (1.0, 3.0), 1.0),
    ],
)
def test_scaled_response_coordinates_preserve_finite_intensity(
    coordinates, multipliers, reference_rate
):
    payload = study_payload()
    state = payload["joint_states"][0]
    path = state["entity_paths"][0]
    path["interpolation"] = "linear"
    for knot, value in zip(path["knots"], coordinates, strict=True):
        knot["value"] = value
    law = state["laws"][0]
    law["reference_annual_rate"] = reference_rate
    law["response"].update(
        interpolation="linear",
        points=[
            {"value": value, "multiplier": multiplier}
            for value, multiplier in zip(coordinates, multipliers, strict=True)
        ],
    )
    with localcontext() as context:
        context.prec = 80
        # A linear path between response endpoints has their arithmetic mean,
        # independently of coordinate units, spacing, or translation.
        expected = (
            Decimal.from_float(reference_rate)
            * sum(Decimal.from_float(value) for value in multipliers)
            / 2
        )
    actual = (
        compile_incident_state(setup_from_payload(payload), "state")
        .incidents[0]
        .annual_rate
    )
    assert actual == pytest.approx(float(expected), rel=2e-15, abs=0)


@pytest.mark.parametrize("path_interpolation", ["step", "linear"])
def test_authored_last_slope_is_capped_one_spacing_above_the_points(path_interpolation):
    import math

    from catmodel.contracts import incident
    from catmodel.contracts.incident import EntityPath, LocalResponse
    from catmodel.simulation.incident import _integrate_local_response

    response = LocalResponse.model_validate(
        {
            "interpolation": "log",
            "extrapolation": "last_slope",
            "points": [
                {"value": 0, "multiplier": 1},
                {"value": 1, "multiplier": 4},
            ],
            "held_fixed": "Synthetic authored response.",
        }
    )
    # The cap is one top-anchor spacing above the highest point: x = 2, where
    # the continued multiplier is 16; it stays 16 for every higher coordinate.
    assert incident.last_slope_cap((0, 1)) == 2
    knots = ((0, 4), (1, 4)) if path_interpolation == "step" else ((0, 0), (1, 4))
    path = EntityPath.model_validate(
        {
            "entity_id": "model",
            "scale_id": "test-scale",
            "history_status": "stipulated",
            "history_rationale": "Synthetic path crossing the cap.",
            "interpolation": path_interpolation,
            "knots": [
                {"time": time, "value": value, "model_revision": "r1"}
                for time, value in knots
            ],
        }
    )
    expected = (
        16.0
        if path_interpolation == "step"
        else 15 / (4 * math.log(4)) + 8  # 4^(4t) to t=0.5, then flat at 16
    )
    assert _integrate_local_response(path, response, 0, 1) == pytest.approx(
        expected, rel=1e-13
    )
