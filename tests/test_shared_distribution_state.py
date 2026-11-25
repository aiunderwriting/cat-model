"""Severity laws with the same fields share what their fields imply."""

from typing import Any

import numpy as np
import pytest
import structlog.testing

from catmodel.probability import distributions
from catmodel.probability.distributions import Distribution
from catmodel.scenarios.sensitivity_coverage import UNCAPPED_TAIL_SHAPE_BOUND

RANKS = np.linspace(0.001, 0.9999, 41)


def _law(**changes):
    fields: dict[str, Any] = {
        "p5": 10.0,
        "p25": 20.0,
        "p50": 40.0,
        "p75": 80.0,
        "p95": 300.0,
        "gpd_shape": 0.3,
        "severity_minimum": 5.0,
        **changes,
    }
    return Distribution(**fields)


@pytest.fixture(autouse=True)
def empty_store(monkeypatch):
    monkeypatch.setattr(distributions, "_SHARED_STATE", {})


def test_laws_with_the_same_fields_share_their_state_and_evaluate_alike(monkeypatch):
    first = _law()
    worked_out = []
    initialize = Distribution._initialize

    def counted(self):
        worked_out.append(self)
        initialize(self)

    monkeypatch.setattr(Distribution, "_initialize", counted)
    second = _law()
    assert worked_out == []
    assert second is not first and second._body is first._body
    assert vars(second).keys() == vars(first).keys()
    assert np.array_equal(
        second.sample_from_uniforms(RANKS), first.sample_from_uniforms(RANKS)
    )
    assert second.analytic_mean() == first.analytic_mean()
    # A law that shares gives what a law worked out by itself gives.
    monkeypatch.setattr(distributions, "_SHARED_STATE", {})
    alone = _law()
    assert worked_out == [alone] and alone._body is not first._body
    assert np.array_equal(
        alone.sample_from_uniforms(RANKS), second.sample_from_uniforms(RANKS)
    )
    assert alone.analytic_mean() == second.analytic_mean()
    assert alone.quantile(0.999) == second.quantile(0.999)


@pytest.mark.parametrize(
    "change",
    [
        {"p95": 300.00000000001},
        {"gpd_shape": 0.30000001},
        {"gpd_shape": None},
        {"severity_cap": 1e6},
        {"severity_minimum": None},
        {"severity_minimum_strict": True},
        {"gpd_tail_ratio": 2.0},
        {"gpd_scale": 100.0},
        {"large_loss_amount": 1e6, "large_loss_exceedance": 0.001},
        {
            "large_loss_amount": 1e6,
            "large_loss_exceedance": 0.001,
            "very_large_loss_amount": 1e7,
            "very_large_loss_exceedance": 0.0001,
        },
    ],
)
def test_a_law_that_differs_in_any_field_shares_nothing(change):
    reference, changed = _law(), _law(**change)
    assert changed._body is not reference._body
    assert len(distributions._SHARED_STATE) == 2


def test_zero_and_negative_zero_are_different_fields():
    zero = Distribution(0.0, 1.0, 2.0, 3.0, 4.0)
    assert Distribution(0.0, 1.0, 2.0, 3.0, 4.0)._body is zero._body
    assert Distribution(-0.0, 1.0, 2.0, 3.0, 4.0)._body is not zero._body


def test_only_laws_whose_numbers_are_floats_share():
    """A float cannot hold every integer, so a law given integers, or any
    other kind of number, is worked out by itself as it always was."""
    floats = Distribution(10.0, 20.0, 40.0, 80.0, 300.0)
    assert Distribution(10.0, 20.0, 40.0, 80.0, 300.0)._body is floats._body
    assert Distribution(10, 20, 40, 80, 300)._body is not floats._body
    assert Distribution(10, 20, 40, 80, 300)._body is not (
        Distribution(10, 20, 40, 80, 300)._body
    )
    # NumPy's float64 is a float.
    assert Distribution(np.float64(10.0), 20.0, 40.0, 80.0, 300.0)._body is (
        floats._body
    )
    assert len(distributions._SHARED_STATE) == 1


def test_a_refused_law_is_refused_every_time_and_nothing_is_kept():
    for _ in range(2):
        with pytest.raises(ValueError, match="non-decreasing"):
            Distribution(5, 4, 3, 2, 1)
        with pytest.raises(ValueError, match="must be finite"):
            _law(severity_cap=float("nan"))
        with pytest.raises(TypeError):
            Distribution(1, 2, "three", 4, 5)  # ty: ignore[invalid-argument-type]
    assert distributions._SHARED_STATE == {}


def test_shared_state_cannot_be_written_and_a_law_keeps_its_own_fields():
    first, second = _law(), _law()
    with pytest.raises(ValueError, match="read-only"):
        first._body.knots[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        first._body.left[0] = 0.0
    before = second.sample_from_uniforms(RANKS)
    # A caller may change an evaluator it owns; no other law sees it.
    first.severity_cap = 50.0
    assert second.severity_cap is None and _law().severity_cap is None
    assert np.array_equal(second.sample_from_uniforms(RANKS), before)


def test_the_uncapped_tail_warning_is_logged_for_every_law_built():
    with structlog.testing.capture_logs() as logs:
        for _ in range(3):
            _law(gpd_shape=UNCAPPED_TAIL_SHAPE_BOUND)
        _law()
    assert len(logs) == 3
    assert all("uncapped xi guardrail" in row["event"] for row in logs)


def test_the_store_is_emptied_when_it_is_full(monkeypatch):
    monkeypatch.setattr(distributions, "_SHARED_STATE_LIMIT", 3)
    laws = [_law(p95=300.0 + index) for index in range(5)]
    # Three laws fill it; the fourth empties it and starts again.
    assert len(distributions._SHARED_STATE) == 2
    assert _law(p95=304.0)._body is laws[4]._body
    assert _law(p95=300.0)._body is not laws[0]._body
