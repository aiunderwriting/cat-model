"""Known physical harm survives an explicitly incomplete financial mapping."""

import numpy as np
import pytest
from pydantic import ValidationError

from catmodel.contracts.incident import ExecutionPlan
from catmodel.results.execution import load_run_capture, replay_capture, run_setup
from catmodel.setup.store import build_setup
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_runtime import simulate_incident_state
from test_support.run_setup import setup_from_payload, with_execution
from test_support.study import partial_financial_payload


def _plan(views):
    return ExecutionPlan(
        plan_id="partial-mapping",
        required_arms=("reference",),
        simulations=512,
        seed=42,
        requested_views=tuple(views),
    )


def test_public_runtime_keeps_known_gross_and_withholds_incomplete_finance():
    study = setup_from_payload(partial_financial_payload())
    compiled = compile_incident_state(study, "state")
    gross, unavailable, _ = simulate_incident_state(
        _plan(("gross",)), compiled, response_sha256="a" * 64
    )
    assert not unavailable
    partial, unavailable, _ = simulate_incident_state(
        _plan(("gross", "financial")), compiled, response_sha256="a" * 64
    )
    np.testing.assert_array_equal(
        gross.annual_losses["gross"], partial.annual_losses["gross"]
    )
    assert partial.views["gross"].native_mean is not None
    assert (
        any(view.native_mean is not None for view in partial.views.values())
        and partial.views["gross"].native_mean.value == 10
    )
    assert np.any(partial.annual_losses["gross"] > 0)
    assert "financial" not in partial.annual_losses
    assert "financial" in unavailable


def test_financial_only_incomplete_request_refuses_before_count_sampling(monkeypatch):
    import catmodel.simulation.incident_runtime as runtime

    study = setup_from_payload(partial_financial_payload())
    monkeypatch.setattr(
        runtime,
        "_origin_counts",
        lambda *_: pytest.fail("counts drawn for an unidentified financial observable"),
    )
    with pytest.raises(ValueError, match="no requested observable is identified"):
        simulate_incident_state(
            _plan(("financial",)),
            compile_incident_state(study, "state"),
            response_sha256="a" * 64,
        )


def test_complete_payment_outcome_restores_finance_without_changing_physical_draws():
    partial_study = setup_from_payload(partial_financial_payload())
    complete_study = setup_from_payload(
        partial_financial_payload(complete=True, payment=1)
    )
    plan = _plan(("gross", "financial"))
    partial, _, _ = simulate_incident_state(
        plan, compile_incident_state(partial_study, "state"), response_sha256="a" * 64
    )
    complete, unavailable, _ = simulate_incident_state(
        plan, compile_incident_state(complete_study, "state"), response_sha256="a" * 64
    )
    assert not unavailable
    np.testing.assert_array_equal(
        partial.annual_losses["gross"], complete.annual_losses["gross"]
    )
    np.testing.assert_array_equal(
        complete.annual_losses["gross"], complete.annual_losses["financial"]
    )
    with pytest.raises(
        ValidationError, match="separate explicit obligation and payment"
    ):
        setup_from_payload(partial_financial_payload(complete=True))


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("contract_id", "unknown", "undeclared contract"),
        ("obligated_entity_id", "other", "contract entity"),
    ],
)
def test_incomplete_mapping_keeps_entity_and_reference_guards(field, value, reason):
    payload = partial_financial_payload()
    membership = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
        "member_outcomes"
    ][0]["memberships"][0]
    membership[field] = value
    with pytest.raises(ValidationError, match=reason):
        setup_from_payload(payload)


@pytest.mark.parametrize("complete", [False, True])
def test_partial_finance_admission_survives_capture_and_replay(
    tmp_path, monkeypatch, complete
):
    import catmodel.simulation.incident_runtime as runtime

    package = build_setup(
        destination=tmp_path / "package",
        setup=with_execution(
            setup_from_payload(
                partial_financial_payload(
                    complete=complete, payment=1 if complete else None
                )
            ),
            _plan(("gross", "financial")),
            setup_id="partial-finance",
        ),
        code_commit="fixture",
    )
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    draws, capture = load_run_capture(tmp_path / "run/capture")
    draw = next(iter(draws["reference"].values()))
    support = next(iter(capture.supports.values()))
    assert draw.views["gross"].native_mean.value == 10
    assert ("financial" in support.unavailable_views) is not complete
    assert ("financial" in draw.annual_losses) is complete
    monkeypatch.setattr(
        runtime,
        "_origin_counts",
        lambda *_: pytest.fail("replay sampled new physical counts"),
    )
    replay = replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run/capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    assert replay["arms"] == report["arms"]
