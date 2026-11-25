"""Exact incident means retain support, composition and scaling semantics."""

import numpy as np
import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_moments import analytic_linear_eals
from test_support.run_setup import setup_from_payload
from test_support.study import _study


def test_identified_full_gross_does_not_become_a_native_zero_from_missing_components():
    payload = encode_scientific(_study(rate=1))
    payload["alternatives"] = []
    for state in payload["joint_states"]:
        outcome = state["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0]
        outcome.update(
            components=[], identified_gross_total=100, composition_complete=False
        )
    study = setup_from_payload(payload)
    means = analytic_linear_eals(
        compile_incident_state(study, "low"), ("gross", "welfare")
    ).values
    assert means == {"gross": 200}


def test_unresolved_overlap_has_no_analytic_unique_gross_authority():
    payload = encode_scientific(_study(rate=1))
    payload["alternatives"] = []
    for state in payload["joint_states"]:
        state["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0][
            "overlap_resolved"
        ] = False
    study = setup_from_payload(payload)
    assert (
        analytic_linear_eals(
            compile_incident_state(study, "low"), ("gross", "welfare")
        ).values
        == {}
    )


def test_fixed_counts_keep_annual_totals_maxima_and_native_mean_distinct(monkeypatch):
    import catmodel.simulation.incident_runtime as runtime
    from test_support.incident import common_case

    _study, plan, compiled, counts = common_case(rate=20)
    counts["campaign"][::2] = 0
    counts["campaign"][1::2] *= 2
    monkeypatch.setattr(runtime, "_origin_counts", lambda *_: counts)
    views = (
        "gross",
        "oep_gross",
        "bearer::victim",
        "contributor::model",
        "attribution::model",
    )
    draw, unavailable, audit = runtime.simulate_incident_state(
        plan.model_copy(update={"requested_views": views}),
        compiled,
        response_sha256="a" * 64,
    )
    assert not unavailable
    for view in views:
        expected = (
            100 * (counts["campaign"] > 0)
            if view == "oep_gross"
            else 100 * counts["campaign"]
        )
        np.testing.assert_array_equal(draw.annual_losses[view], expected)
    assert audit.total_initiators == sum(counts["campaign"])
    assert draw.views["gross"].native_mean is not None
    assert draw.views["gross"].native_mean.value == 2000
    counts["campaign"][:] = 0
    draw, _, _ = runtime.simulate_incident_state(
        plan.model_copy(update={"requested_views": ("gross",)}),
        compiled,
        response_sha256="a" * 64,
    )
    assert not np.any(draw.annual_losses["gross"])
    assert draw.views["gross"].native_mean is not None
    assert draw.views["gross"].native_mean.value == 2000


def test_financial_occurrence_and_shared_annual_terms_use_common_marks(monkeypatch):
    import catmodel.simulation.incident_runtime as runtime
    from test_support.incident import common_case

    study, plan, _, counts = common_case(financial=True, rate=20)
    payload = encode_scientific(study)
    for state in payload["joint_states"]:
        state["laws"].append({**state["laws"][0], "origin_id": "other"})
    counts["other"] = counts["campaign"] // 2
    monkeypatch.setattr(runtime, "_origin_counts", lambda *_: counts)
    views = (
        "gross",
        "financial",
        "oep_gross",
        "oep_financial",
        "risk_unit::campaign::occurrence_net_financial",
    )
    draw, unavailable, _ = runtime.simulate_incident_state(
        plan.model_copy(update={"requested_views": views}),
        compile_incident_state(setup_from_payload(payload), "low"),
        response_sha256="a" * 64,
    )
    assert not unavailable
    total = counts["campaign"] + counts["other"]
    np.testing.assert_array_equal(draw.annual_losses["gross"], 100 * total)
    np.testing.assert_array_equal(draw.annual_losses["financial"], 40 * total - 500)
    np.testing.assert_array_equal(
        draw.annual_losses["risk_unit::campaign::occurrence_net_financial"],
        40 * counts["campaign"],
    )
    np.testing.assert_array_equal(draw.annual_losses["oep_gross"], 100)
    np.testing.assert_array_equal(draw.annual_losses["oep_financial"], 40)


def test_outside_defense_is_added_after_occurrence_limit(monkeypatch):
    import catmodel.simulation.incident_runtime as runtime
    from test_support.incident import common_case

    study, plan, _, counts = common_case(financial=True)
    payload = encode_scientific(study)
    payload["contracts"][0].update(
        occurrence_retention=0, occurrence_limit=50, annual_retention=0
    )
    for state in payload["joint_states"]:
        outcome = state["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0]
        first = outcome["components"][0]
        outcome["components"] = [
            {**first, "amount": 80},
            {**first, "component_id": "cost", "amount": 20, "loss_type": "defense"},
        ]
        outcome["obligations"].append(
            {**outcome["obligations"][0], "component_id": "cost"}
        )
        outcome["payments"].append({**outcome["payments"][0], "component_id": "cost"})
        outcome["memberships"].append(
            {
                **outcome["memberships"][0],
                "component_id": "cost",
                "outside_limits": True,
            }
        )
    monkeypatch.setattr(runtime, "_origin_counts", lambda *_: counts)
    draw, unavailable, _ = runtime.simulate_incident_state(
        plan.model_copy(
            update={"requested_views": ("gross", "financial", "oep_financial")}
        ),
        compile_incident_state(setup_from_payload(payload), "low"),
        response_sha256="a" * 64,
    )
    assert not unavailable
    np.testing.assert_array_equal(draw.annual_losses["gross"], 100 * counts["campaign"])
    np.testing.assert_array_equal(
        draw.annual_losses["financial"], 70 * counts["campaign"]
    )
    np.testing.assert_array_equal(draw.annual_losses["oep_financial"], 70)


def test_native_payment_beta_and_limits_remain_a_joint_mark(monkeypatch):
    from scipy.integrate import quad
    from scipy.stats import beta

    import catmodel.simulation.incident_runtime as runtime
    from catmodel.probability.distributions import Distribution
    from test_support.incident import common_case

    study, plan, _, counts = common_case(
        sampler=Distribution(100, 100, 100, 100, 100),
        financial=True,
        rate=100,
        simulations=64,
    )
    payload = encode_scientific(study)
    payload["contracts"][0].update(annual_retention=0, occurrence_limit=20)
    for state in payload["joint_states"]:
        outcome = state["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0]
        outcome["obligations"], outcome["payments"] = [], []
        outcome["payment_kernel"] = {
            "source_severity": outcome["components"][0]["severity"],
            "rank_group_id": "severity",
            "entity_id": "model",
            "liability_component_ids": ["harm"],
            "liability": {
                "mean": 0.5,
                "concentration": 10,
                "recovery_rate": 1,
                "severity_slope": 0,
                "regulatory_recovery_threshold": 1e9,
                "regulatory_recovery_uplift": 0,
            },
        }
    monkeypatch.setattr(runtime, "_origin_counts", lambda *_: counts)
    draw, unavailable, _ = runtime.simulate_incident_state(
        plan.model_copy(
            update={"requested_views": ("gross", "financial", "oep_financial")}
        ),
        compile_incident_state(setup_from_payload(payload), "low"),
        response_sha256="a" * 64,
    )
    assert not unavailable

    def payment(share):
        return min(max(100 * share - 60, 0), 20)

    expected, second = (
        quad(
            lambda share, power=power: payment(share) ** power * beta.pdf(share, 5, 5),
            0,
            1,
            points=(0.6, 0.8),
            epsabs=0,
            epsrel=1e-12,
        )[0]
        for power in (1, 2)
    )
    total_count = sum(counts["campaign"])
    standard_error = np.sqrt((second - expected**2) / total_count)
    assert (
        abs(draw.annual_losses["financial"].sum() / total_count - expected)
        <= 6 * standard_error
    )
    np.testing.assert_array_equal(draw.annual_losses["gross"], counts["campaign"] * 100)
    assert np.all(draw.annual_losses["oep_financial"] <= 20)
    cap_probability = 1 - beta.cdf(0.8, 5, 5) ** 100
    observed_cap = np.mean(draw.annual_losses["oep_financial"] == 20)
    assert observed_cap == pytest.approx(
        cap_probability, abs=6 * np.sqrt(cap_probability * (1 - cap_probability) / 64)
    )


def test_branch_campaign_cost_is_not_erased_by_zero_member_gates(monkeypatch):
    import catmodel.simulation.incident_runtime as runtime
    from test_support.incident import common_case

    def compiled(rate):
        study, plan, _, counts = common_case(rate=rate)
        payload = encode_scientific(study)
        for state in payload["joint_states"]:
            kernel = state["laws"][0]["segments"][0]["kernel"]
            component = {
                **kernel["member_outcomes"][0]["components"][0],
                "component_id": "response",
                "amount": 20,
            }
            kernel.update(
                cluster=None,
                member_outcomes=[],
                unit_probability=0,
                explicit_joint_members=[
                    {
                        "probability": 1,
                        "members": [],
                        "campaign_components": [component],
                    }
                ],
            )
        return (
            compile_incident_state(setup_from_payload(payload), "low"),
            plan,
            counts,
        )

    assert analytic_linear_eals(compiled(10_000)[0], ("gross",)).values == {
        "gross": 200_000
    }
    state, plan, counts = compiled(10)
    counts["campaign"][::3] = 0
    monkeypatch.setattr(runtime, "_origin_counts", lambda *_: counts)
    draw, unavailable, audit = runtime.simulate_incident_state(
        plan.model_copy(update={"requested_views": ("gross",)}),
        state,
        response_sha256="a" * 64,
    )
    assert not unavailable
    assert audit.campaign_members == audit.realized_member_work == 0
    np.testing.assert_array_equal(draw.annual_losses["gross"], 20 * counts["campaign"])
    native = draw.views["gross"].native_mean
    assert native is not None and native.value == 200
