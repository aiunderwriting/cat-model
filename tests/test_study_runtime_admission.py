"""Positive-support finance admission and purpose-specific physical streams."""

import copy

import numpy as np
import pytest

import catmodel.simulation.incident_runtime as runtime
from catmodel.contracts._setup_codec import encode_scientific
from catmodel.probability.distributions import Distribution
from catmodel.probability.liability import LiabilityAttenuation
from catmodel.simulation.incident import compile_incident_state
from test_support.incident import common_case as _case
from test_support.results import available_metric, present, reduced_arm
from test_support.run_setup import setup_from_payload
from test_support.study import _financial_study, _native_payment_study, _plan


def _run(study, views, *, seed=42, simulations=32):
    plan = _plan(simulations).model_copy(
        update={
            "requested_views": tuple(views),
            "seed": seed,
            "numerical_controls": {
                "max_campaign_members_per_year": 25_000,
            },
        }
    )
    return runtime.simulate_incident_state(
        plan, compile_incident_state(study, "low"), response_sha256="a" * 64
    )


def _defense_case(*, members=2, exclusive=False, zero_loading_payment=False):
    payload = encode_scientific(_financial_study())
    payload["contracts"][0].update(occurrence_retention=0, occurrence_limit=None)
    for state in payload["joint_states"]:
        kernel = state["laws"][0]["segments"][0]["kernel"]
        kernel["cluster"]["support"] = [{"count": members, "probability": 1}]
        first = kernel["member_outcomes"][0]
        first["probability"] = 0.5
        first["components"][0]["loss_type"] = "indemnity"
        first["memberships"][0]["defense_loading"] = 0.2
        if zero_loading_payment:
            first["payments"][0]["fraction"] = 0
        second = copy.deepcopy(first)
        second["components"][0]["loss_type"] = "defense"
        second["memberships"][0]["defense_loading"] = 0
        second["payments"][0]["fraction"] = 1
        kernel["member_outcomes"].append(second)
        if exclusive:
            kernel.update(
                cluster=None,
                member_outcomes=[],
                explicit_joint_members=[
                    {"probability": 0.5, "members": [{**outcome, "probability": 1}]}
                    for outcome in (first, second)
                ],
            )
    return setup_from_payload(payload)


def test_jointly_possible_defense_loading_conflict_fails_before_random_draws(
    monkeypatch,
):
    study = _defense_case()
    compiled = compile_incident_state(study, "low")
    monkeypatch.setattr(
        np.random,
        "default_rng",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("randomness before support admission")
        ),
    )
    reasons, _, _ = runtime.view_admission(compiled, ("financial",))
    assert "same physical scope" in reasons["financial"]


@pytest.mark.parametrize("mode", ["one_member", "exclusive_branch", "zero_payment"])
def test_mutually_exclusive_or_zero_payment_routes_do_not_invent_loading_conflict(mode):
    study = _defense_case(
        members=1 if mode == "one_member" else 2,
        exclusive=mode == "exclusive_branch",
        zero_loading_payment=mode == "zero_payment",
    )
    draw, reasons, _ = _run(study, ("gross", "financial"))
    assert not reasons
    assert np.all(draw.annual_losses["financial"] >= 0)


def test_shared_contract_year_occurrence_withholds_only_source_allocation():
    payload = encode_scientific(_financial_study("contract_year"))
    for state in payload["joint_states"]:
        law = state["laws"][0]
        law["segments"][0]["kernel"]["cluster"]["support"] = [
            {"count": 1, "probability": 1}
        ]
        other = copy.deepcopy(law)
        other["origin_id"] = "other"
        state["laws"].append(other)
    study = setup_from_payload(payload)
    view = "risk_unit::campaign::occurrence_net_financial"
    reasons, _, _ = runtime.view_admission(
        compile_incident_state(study, "low"), ("gross", "financial", view)
    )
    assert view in reasons
    for seed in (0, 1, 2):
        draw, unavailable, _ = _run(
            study, ("gross", "financial", view), seed=seed, simulations=8
        )
        assert view in unavailable
        assert set(draw.annual_losses) == {"gross", "financial"}


@pytest.mark.parametrize("outside", ["none", "component", "loading"])
def test_annual_limit_bounds_only_the_actual_in_limit_payment_paths(outside):
    payload = encode_scientific(_financial_study(shape=1.2))
    payload["contracts"][0].update(
        occurrence_retention=0,
        occurrence_limit=None,
        annual_limit=5,
        outside_limits_subject_to_aggregate=False,
    )
    for state in payload["joint_states"]:
        membership = state["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0][
            "memberships"
        ][0]
        membership["outside_limits"] = outside == "component"
        if outside == "loading":
            membership.update(defense_loading=0.2, defense_outside_limits=True)
    study = setup_from_payload(payload)
    draw, _, _ = _run(study, ("gross", "financial", "oep_financial"))
    assert draw.views["financial"].mean_exists == (outside == "none")
    assert not draw.views["oep_financial"].mean_exists
    if outside == "none":
        assert np.max(draw.annual_losses["financial"]) <= 5
        assert draw.views["financial"].variance_exists


def _add_origin(study, identifier):
    payload = encode_scientific(study)
    payload["alternatives"] = []
    population = copy.deepcopy(payload["populations"][0])
    population["scenario_id"] = "other-loss"
    payload["populations"].append(population)
    for state in payload["joint_states"]:
        law = copy.deepcopy(state["laws"][0])
        law.update(origin_id=identifier, scenario_id="other-loss")
        for segment in law["segments"]:
            for outcome in segment["kernel"]["member_outcomes"]:
                for component in outcome["components"]:
                    component["scenario_ids"] = ["other-loss"]
        state["laws"].append(law)
    return setup_from_payload(payload)


def test_adding_earlier_or_later_origins_preserves_original_counts_and_random_marks():
    study, _, _, _ = _case(sampler=Distribution(1, 2, 4, 8, 16), rate=2, simulations=32)
    base, _, base_audit = _run(study, ("scenario::loss::gross",))
    for identifier in ("aaa-new", "zzz-new"):
        extended, _, audit = _run(
            _add_origin(study, identifier), ("scenario::loss::gross",)
        )
        assert (
            audit.initiating_counts["campaign"]
            == base_audit.initiating_counts["campaign"]
        )
        np.testing.assert_array_equal(
            extended.annual_losses["scenario::loss::gross"],
            base.annual_losses["scenario::loss::gross"],
        )


def test_native_payment_endpoint_changes_preserve_the_same_physical_marks():
    outcomes = []
    for mean in (1, 0.3):
        payload = encode_scientific(_native_payment_study(mean=mean, recovery_rate=1))
        for state in payload["joint_states"]:
            state["laws"][0]["reference_annual_rate"] = 2
            state["laws"][0]["segments"][0]["kernel"]["cluster"]["support"] = [
                {"count": 1, "probability": 1}
            ]
        outcomes.append(
            _run(
                setup_from_payload(payload),
                ("gross", "financial", "oep_gross"),
            )
        )
    for view in ("gross", "oep_gross"):
        np.testing.assert_array_equal(
            outcomes[0][0].annual_losses[view], outcomes[1][0].annual_losses[view]
        )
    assert outcomes[0].audit.initiating_counts == outcomes[1].audit.initiating_counts


def test_bounded_annual_payment_mean_and_tvar_survive_public_run_and_replay(tmp_path):
    from catmodel.results.execution import replay_capture, run_setup
    from test_support.study import _package

    payload = encode_scientific(_financial_study(shape=1.2))
    payload["contracts"][0].update(
        occurrence_retention=0,
        occurrence_limit=None,
        annual_limit=5,
        outside_limits_subject_to_aggregate=False,
    )
    study = setup_from_payload(payload)
    plan = _plan(64).model_copy(
        update={
            "required_arms": ("reference",),
            "requested_views": ("gross", "financial"),
        }
    )
    package = _package(tmp_path, study, plan)
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    metrics = reduced_arm(report["arms"]["reference"])["metrics"]
    assert available_metric(metrics["gross"])["expected_annual_loss"] is None
    assert (
        0
        <= present(available_metric(metrics["financial"])["expected_annual_loss"])
        <= 5
    )
    assert (
        0
        <= present(
            available_metric(metrics["financial"])["tail_metrics"]["0.99"]["tvar"]
        )
        <= 5
    )
    replay = replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    assert reduced_arm(replay["arms"]["reference"])["metrics"] == metrics


@pytest.mark.parametrize("scope", ["campaign", "member"])
def test_financial_only_rank_groups_do_not_consume_physical_randomness(scope):
    from catmodel.contracts.native_severity import native_severity_from_sampler
    from test_support.study import _study

    study = _study(rate=1, response_cost=10, unit_probability=0.5)
    baseline, _, _ = _run(study, ("gross",), simulations=100)
    payload = encode_scientific(study)
    for state in payload["joint_states"]:
        kernel = state["laws"][0]["segments"][0]["kernel"]
        native = {
            "source_severity": native_severity_from_sampler(
                Distribution(1, 1, 1, 1, 1)
            ).model_dump(mode="json"),
            "rank_group_id": "payment-only",
            "rank_scope": scope,
            "entity_id": "model",
            "direct_component_ids": ["response" if scope == "campaign" else "harm"],
        }
        if scope == "campaign":
            kernel["campaign_payment_kernel"] = native
        else:
            kernel["member_outcomes"][0]["payment_kernel"] = native
    changed, _, _ = _run(setup_from_payload(payload), ("gross",), simulations=100)
    np.testing.assert_array_equal(
        changed.annual_losses["gross"], baseline.annual_losses["gross"]
    )


def test_native_zero_requires_support_proof_and_positive_underflow_is_unavailable():
    from catmodel.simulation.incident_moments import analytic_known_zero_views
    from test_support.study import _study

    payload = encode_scientific(_study(rate=1e-316))
    for state in payload["joint_states"]:
        state["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0]["components"][
            0
        ]["amount"] = 1e-9
    positive = setup_from_payload(payload)
    compiled = compile_incident_state(positive, "low")
    assert "gross" not in analytic_known_zero_views(compiled, ("gross",))
    draw, _, execution = _run(positive, ("gross",))
    assert not any(view.native_mean is not None for view in draw.views.values())
    assert draw.views["gross"].mean_exists
    assert not draw.views["gross"].variance_exists
    assert (
        "numeric output range"
        in execution.functional_unavailable_by_view["gross"]["expected_annual_loss"]
    )

    zero = _study(rate=0)
    draw, _, _ = _run(zero, ("gross",))
    assert draw.views["gross"].native_mean.value == 0
    assert draw.views["gross"].native_mean.precision["known_zero_law"] is True


@pytest.mark.parametrize("views", [("gross",), ("gross", "financial")])
def test_invalid_optional_finance_preserves_complete_physical_gross(monkeypatch, views):
    study = _defense_case()
    counts = np.array([0, 1, 2, 3], dtype=np.int64)
    monkeypatch.setattr(runtime, "_origin_counts", lambda *_: {"campaign": counts})
    monkeypatch.setattr(
        LiabilityAttenuation,
        "sample_conditional_from_uniforms",
        lambda *_: (_ for _ in ()).throw(AssertionError("unneeded payment draw")),
    )
    draw, unavailable, execution = _run(study, views, simulations=4)
    np.testing.assert_array_equal(draw.annual_losses["gross"], 100 * counts)
    assert set(draw.annual_losses) == {"gross"}
    assert ("financial" in unavailable) == ("financial" in views)
    assert execution.available_views == ["gross"]


def test_invalid_financial_only_request_refuses_before_counts(monkeypatch):
    monkeypatch.setattr(
        runtime,
        "_origin_counts",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("counts drawn before admission")
        ),
    )
    with pytest.raises(ValueError, match="same physical scope"):
        _run(_defense_case(), ("financial",), simulations=4)


def test_independent_occurrence_payment_survives_another_sources_invalid_transform(
    monkeypatch,
):
    payload = encode_scientific(_defense_case())
    for state in payload["joint_states"]:
        valid = copy.deepcopy(state["laws"][0])
        valid["origin_id"] = "independent"
        kernel = valid["segments"][0]["kernel"]
        kernel["cluster"]["support"] = [{"count": 1, "probability": 1}]
        member = kernel["member_outcomes"][0]
        member["probability"] = 1
        member["memberships"][0]["defense_loading"] = 0
        kernel["member_outcomes"] = [member]
        state["laws"].append(valid)
    study = setup_from_payload(payload)
    counts = np.array([0, 1, 2, 3], dtype=np.int64)
    monkeypatch.setattr(
        runtime,
        "_origin_counts",
        lambda *_: {"campaign": counts, "independent": counts},
    )
    view = "risk_unit::independent::occurrence_net_financial"
    draw, unavailable, _ = _run(study, ("gross", "financial", view), simulations=4)
    assert "financial" in unavailable and view not in unavailable
    assert "financial" not in draw.annual_losses
    np.testing.assert_array_equal(draw.annual_losses[view], 50 * counts)
    np.testing.assert_array_equal(draw.annual_losses["gross"], 150 * counts)


def test_omitting_native_finance_preserves_shared_physical_ranks(monkeypatch):
    study = _native_payment_study(mean=0.3, recovery_rate=1)
    complete, _, _ = _run(study, ("gross", "financial"), simulations=64)
    monkeypatch.setattr(
        LiabilityAttenuation,
        "sample_conditional_from_uniforms",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("unneeded native finance sampled")
        ),
    )
    physical, unavailable, _ = _run(study, ("gross",), simulations=64)
    assert not unavailable
    np.testing.assert_array_equal(
        physical.annual_losses["gross"], complete.annual_losses["gross"]
    )
