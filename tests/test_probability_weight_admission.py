"""Direct Python weighting APIs reject unrepresentable or malformed probabilities."""

from typing import Any

import pytest

from catmodel.results.execution import _weights
from catmodel.results.reduction import reduce_captures
from test_support.study import _package


def _reduce(weights: Any, *, per_arm: bool = False):
    return reduce_captures(
        scientific_sha256="study",
        plan_sha256="plan",
        draws_by_arm={},
        state_weights={"first": 0.5, "second": 0.5} if per_arm else weights,
        state_weights_by_arm={"base": weights} if per_arm else None,
        required_arms=("base",),
        reference_arm="base",
        views=("gross",),
        reducer_provenance={"commit": "fixture"},
    )


@pytest.mark.parametrize("per_arm", [False, True])
@pytest.mark.parametrize(
    "bad", [10**400, "0.5", True, None, float("nan"), float("inf"), -0.5]
)
def test_direct_apis_reject_invalid_probability_values(per_arm, bad):
    with pytest.raises(ValueError, match="finite, nonnegative and sum to one"):
        _reduce({"first": bad, "second": 0.5}, per_arm=per_arm)


@pytest.mark.parametrize("per_arm", [False, True])
def test_direct_apis_reject_overflowing_sum_of_finite_weights(per_arm):
    with pytest.raises(ValueError, match="finite, nonnegative and sum to one"):
        _reduce({"first": 1e308, "second": 1e308}, per_arm=per_arm)


@pytest.mark.parametrize("malformed", [[], 1, True, "weights"])
def test_direct_apis_require_weight_mappings(malformed: Any):
    with pytest.raises(ValueError, match="state weights"):
        _reduce(malformed)


@pytest.mark.parametrize("malformed", [[], 1, True, "weights"])
def test_reduction_requires_per_arm_weight_mapping(malformed: Any):
    with pytest.raises(ValueError, match="arm weights must cover exactly"):
        reduce_captures(
            scientific_sha256="study",
            plan_sha256="plan",
            draws_by_arm={},
            state_weights={"first": 0.5, "second": 0.5},
            state_weights_by_arm=malformed,
            required_arms=("base",),
            reference_arm="base",
            views=("gross",),
            reducer_provenance={},
        )


def test_reduction_keeps_its_sum_tolerance():
    report = _reduce({"first": 0.5 + 5e-11, "second": 0.5})
    assert report["analysis_spec"]["state_weights"]["first"] == 0.5 + 5e-11
    with pytest.raises(ValueError, match="sum to one"):
        _reduce({"first": 0.5 + 5e-10, "second": 0.5})


def test_service_preserves_its_existing_relative_sum_tolerance(tmp_path):
    package = _package(tmp_path)
    weights = _weights(package, {"low": 0.25 + 5e-10, "high": 0.75})
    assert sum(weights.values()) == 1 + 5e-10
    with pytest.raises(ValueError, match="sum to one"):
        _weights(package, {"low": 0.25 + 2e-9, "high": 0.75})


def test_direct_apis_keep_their_roster_checks():
    with pytest.raises(ValueError, match="full declared support"):
        _reduce({"other": 1.0}, per_arm=True)
