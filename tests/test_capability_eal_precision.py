"""Keep conditional headline-EAL uncertainty attached to the estimator it describes."""

from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from catmodel.results.capability_storage import (
    decode_conditional_draw,
    encode_conditional_draw,
)
from catmodel.simulation.capability_mixture import (
    DETERMINISTIC_EAL_PRECISION_BASIS,
    CapabilityMixture,
    ConditionalCapabilityDraws,
    ConditionalView,
    NativeMean,
)
from catmodel.simulation.incident_runtime import simulate_incident_state
from tests.test_capability_runtime import setup_runtime

pytestmark = pytest.mark.filterwarnings("ignore:n_simulations=.*:UserWarning")


def _precision(standard_error=2.0, *, suppression=None):
    return {
        "numerical_standard_error": standard_error,
        "precision_basis": "conditional_estimator_finite_variance",
        "precision_suppression_reason": suppression,
        "conditional_draws": 512,
        "seed": 42,
        "scope": "Synthetic native benchmark population.",
        "native_future_flag": {"retained": True},
    }


def _draw(path="a" * 64, standard_error=2.0):
    return ConditionalCapabilityDraws(
        stratum_sha256=path,
        response_sha256="d" * 64,
        seed=42,
        views={
            "gross": ConditionalView(
                np.array([0.0, 1000, 0, 3000]),
                native_mean=NativeMean(
                    10,
                    "separate_native_conditional_benchmark",
                    _precision(standard_error),
                ),
            )
        },
    )


def _mixture(first, second, first_weight=0.25):
    return CapabilityMixture(
        {first.stratum_sha256: first, second.stratum_sha256: second},
        {first.stratum_sha256: first_weight, second.stratum_sha256: 1 - first_weight},
    )


def test_headline_precision_uses_conditional_estimators_and_never_assumes_path_independence():
    first, second = _draw(), _draw("e" * 64, 3)
    metric = _mixture(first, second).metrics("gross")
    assert metric["expected_annual_loss"] == 10
    assert metric["normalized_empirical_mean"] == 1000
    precision = metric["eal_precision"]
    assert precision["numerical_standard_error"] is None
    assert precision["standard_error_upper_bound"] == 0.25 * 2 + 0.75 * 3
    assert (
        precision["precision_basis"]
        == "sum_of_weighted_standard_error_bounds_no_independence_assumed"
    )
    assert precision["stratum_estimators"][first.stratum_sha256] == _precision(2)
    assert (
        _mixture(first, second, 0.5).metrics("gross")["eal_precision"][
            "standard_error_upper_bound"
        ]
        == 2.5
    )


def test_empirical_mean_without_native_benchmark_claims_no_standard_error():
    empirical = replace(
        _draw(), views={"gross": ConditionalView(np.array([0.0, 1000, 0, 3000]))}
    )
    mixture = CapabilityMixture(
        {empirical.stratum_sha256: empirical}, {empirical.stratum_sha256: 1}
    )
    precision = cast(dict[str, object], mixture.metrics("gross")["eal_precision"])
    assert precision["numerical_standard_error"] is None
    assert precision["standard_error_upper_bound"] is None


@pytest.mark.parametrize("kind", ["missing", "suppressed"])
def test_missing_or_suppressed_positive_stratum_precision_is_not_zero(kind):
    first, second = _draw(), _draw("e" * 64)
    second = replace(
        second,
        views={
            "gross": replace(
                second.views["gross"],
                native_mean=replace(
                    second.views["gross"].native_mean,
                    precision=None
                    if kind == "missing"
                    else _precision(
                        None, suppression="Infinite-variance mark estimator."
                    ),
                ),
            )
        },
    )
    precision = _mixture(first, second).metrics("gross")["eal_precision"]
    assert precision["standard_error_upper_bound"] is None
    assert second.stratum_sha256 in precision["precision_suppression_reason"]
    # A zero-mass unresolved stratum does not contaminate the current forecast.
    assert (
        _mixture(first, second, 1).metrics("gross")["eal_precision"][
            "standard_error_upper_bound"
        ]
        == 2
    )


@pytest.mark.parametrize("shared", [False, True])
def test_native_runtime_preserves_exact_full_law_precision_and_provenance(shared):
    _unused_study, plan, compiled = setup_runtime(shared=shared, simulations=128)
    draw, reasons, allocation = simulate_incident_state(
        plan, compiled["low"], response_sha256="a" * 64
    )
    assert not reasons
    assert isinstance(allocation.total_initiators, int)
    assert allocation.total_initiators >= 0
    precision = {
        name: view.native_mean.precision
        for name, view in draw.views.items()
        if view.native_mean is not None
    }
    assert any(view.native_mean is not None for view in draw.views.values())
    for name in ("gross", "welfare", "net_resource_cost", "provider_contribution"):
        expected = -np.log(0.9) * (1 if shared else 2)
        if name == "provider_contribution" and shared:
            expected *= 2
        native = draw.views[name].native_mean
        assert native is not None
        assert native.value == pytest.approx(expected)
        assert precision[name] == {
            "numerical_standard_error": 0,
            "standard_error_upper_bound": 0,
            "precision_basis": DETERMINISTIC_EAL_PRECISION_BASIS,
            "integration_error_bound": 0,
            "known_zero_law": False,
        }
    assert draw.joint_law_semantics == "physical_joint"
    assert draw.evaluator_provenance_by_risk_unit


def test_numeric_codec_preserves_complete_precision_metadata_and_detects_tampering():
    draw = _draw()
    record, data = encode_conditional_draw(draw)
    arguments = dict(
        stratum_sha256=draw.stratum_sha256,
        response_sha256=draw.response_sha256,
        seed=draw.seed,
    )
    loaded = decode_conditional_draw(record, data, **arguments)
    assert loaded.views["gross"].native_mean == draw.views["gross"].native_mean
    assert loaded.seed == draw.seed
    assert not loaded.annual_losses["gross"].flags.writeable
    corrupted = bytes([data[0] ^ 1]) + data[1:]
    with pytest.raises(ValueError, match="numeric array checksum"):
        decode_conditional_draw(record, corrupted, **arguments)


@pytest.mark.parametrize("standard_error", [float("nan"), float("inf"), -1, True])
def test_precision_cannot_smuggle_nonfinite_or_negative_error(standard_error):
    with pytest.raises(ValueError):
        _draw(standard_error=standard_error)


def test_missing_precision_remains_unavailable_on_its_own_native_mean():
    source = _draw()
    source = replace(
        source,
        views={
            "gross": replace(
                source.views["gross"],
                native_mean=replace(source.views["gross"].native_mean, precision={}),
            )
        },
    )
    assert (
        _mixture(source, _draw("e" * 64)).metrics("gross")["eal_precision"][
            "standard_error_upper_bound"
        ]
        is None
    )
