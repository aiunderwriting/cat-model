"""Direct execution invariants for frozen complete incident laws."""

import numpy as np
import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_runtime import simulate_incident_state
from test_support.run_setup import setup_from_payload
from test_support.study import (
    _financial_study,
    _native_payment_study,
    _plan,
    _study,
)


def test_response_cost_once_per_campaign_even_perfect_prevention():
    study = _study(rate=2, response_cost=10)
    plan = _plan(30000)
    compiled = compile_incident_state(study, "low", "contained")
    draw, unavailable, allocation = simulate_incident_state(
        plan, compiled, response_sha256="a" * 64
    )
    assert not unavailable
    assert isinstance(allocation.total_initiators, int)
    assert draw.annual_losses["gross"].sum() == allocation.total_initiators * 10
    assert np.mean(draw.annual_losses["gross"]) == pytest.approx(20, abs=0.4)
    assert np.all(
        draw.annual_losses["gross"] == draw.annual_losses["scenario::loss::gross"]
    )


def test_missing_composition_keeps_unique_gross_and_withholds_dependent_views():
    payload = encode_scientific(_study())
    for state in payload["joint_states"]:
        outcome = state["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0]
        outcome.update(
            components=[], identified_gross_total=100, composition_complete=False
        )
    study = setup_from_payload(payload)
    compiled = compile_incident_state(study, "low")
    draw, unavailable, _ = simulate_incident_state(
        _plan(100), compiled, response_sha256="a" * 64
    )
    assert set(draw.annual_losses) == {"gross"}
    assert "welfare" in unavailable
    assert "scenario::loss::gross" in unavailable


def test_initiation_gate_and_cluster_single_count_authority():
    payload = encode_scientific(_study(rate=1))
    state = payload["joint_states"][0]
    state["laws"][0]["segments"][0]["initiation_probability"] = 0.2
    study = setup_from_payload(payload)
    compiled = compile_incident_state(study, "low")
    draw, _, allocation = simulate_incident_state(
        _plan(30000), compiled, response_sha256="a" * 64
    )
    assert isinstance(allocation.sampled_potential_members, int)
    assert allocation.total_initiators == allocation.sampled_potential_members / 2
    assert np.mean(draw.annual_losses["gross"]) == pytest.approx(40, abs=2)
    assert np.all(draw.annual_losses["gross"] % 200 == 0)


def test_authored_campaign_and_member_policy_occurrences_reach_financial_transform():
    plan = _plan(1000).model_copy(update={"requested_views": ("gross", "financial")})
    common = _financial_study("campaign")
    split = _financial_study("member")
    common_draw, _, allocation = simulate_incident_state(
        plan, compile_incident_state(common, "low"), response_sha256="a" * 64
    )
    split_draw, _, _ = simulate_incident_state(
        plan, compile_incident_state(split, "low"), response_sha256="a" * 64
    )
    assert isinstance(allocation.total_initiators, int)
    # Two physical losses of 50: one policy occurrence pays 100 - 60; separate
    # occurrences pay zero.
    assert (
        common_draw.annual_losses["financial"].sum() == allocation.total_initiators * 40
    )
    assert np.all(split_draw.annual_losses["financial"] == 0)
    np.testing.assert_array_equal(
        common_draw.annual_losses["gross"], split_draw.annual_losses["gross"]
    )


def test_finite_mean_infinite_variance_remains_point_available():
    study = _financial_study(shape=0.6)
    plan = _plan(200).model_copy(update={"requested_views": ("gross", "financial")})
    draw, _, _ = simulate_incident_state(
        plan, compile_incident_state(study, "low"), response_sha256="a" * 64
    )
    assert {name: view.mean_exists for name, view in draw.views.items()} == {
        "gross": True,
        "financial": True,
    }
    assert {name: view.variance_exists for name, view in draw.views.items()} == {
        "gross": False,
        "financial": True,
    }


def test_gaussian_count_stress_and_gamma_intensity_retain_distinct_marginals():
    from catmodel.simulation.incident_runtime import _origin_counts

    payload = encode_scientific(_study(rate=4))
    for state in payload["joint_states"]:
        state["laws"].append({**state["laws"][0], "origin_id": "other"})
    controls = {
        "gaussian-control": {
            "kind": "fixed_marginal_gaussian",
            "origin_groups": {"campaign": "all", "other": "all"},
            "correlation": 0,
        },
        "gaussian-stress": {
            "kind": "fixed_marginal_gaussian",
            "origin_groups": {"campaign": "all", "other": "all"},
            "correlation": 0.8,
        },
        "gamma-control": {
            "kind": "shared_gamma_intensity",
            "origin_groups": {"campaign": "campaign", "other": "other"},
            "coefficient_of_variation": 1,
        },
        "gamma-stress": {
            "kind": "shared_gamma_intensity",
            "origin_groups": {"campaign": "all", "other": "all"},
            "coefficient_of_variation": 1,
        },
    }
    payload["alternatives"] = [
        {
            "alternative_id": arm,
            "purpose": "scientific_alternative",
            "description": "Declared count experiment with matched control.",
            "count_experiment": experiment,
            "matched_control_id": arm.replace("stress", "control"),
        }
        for arm, experiment in controls.items()
    ]
    study = setup_from_payload(payload)
    plan = _plan(40000)
    outcomes = {
        arm: _origin_counts(compile_incident_state(study, "low", arm), plan)
        for arm in controls
    }
    for arm, counts in outcomes.items():
        assert np.mean(counts["campaign"]) == pytest.approx(4, abs=0.08)
        expected_variance = 20 if arm.startswith("gamma") else 4
        assert np.var(counts["campaign"]) == pytest.approx(expected_variance, rel=0.08)
    assert (
        np.corrcoef(
            outcomes["gaussian-stress"]["campaign"],
            outcomes["gaussian-stress"]["other"],
        )[0, 1]
        > 0.7
    )
    assert (
        abs(
            np.corrcoef(
                outcomes["gaussian-control"]["campaign"],
                outcomes["gaussian-control"]["other"],
            )[0, 1]
        )
        < 0.03
    )
    assert (
        np.corrcoef(
            outcomes["gamma-stress"]["campaign"], outcomes["gamma-stress"]["other"]
        )[0, 1]
        > 0.7
    )
    assert (
        abs(
            np.corrcoef(
                outcomes["gamma-control"]["campaign"],
                outcomes["gamma-control"]["other"],
            )[0, 1]
        )
        < 0.03
    )


def test_common_oep_retains_empty_year_atom_and_physical_campaign_scope():
    study = _study(rate=1)
    plan = _plan(10000).model_copy(update={"requested_views": ("gross", "oep_gross")})
    draw, unavailable, _ = simulate_incident_state(
        plan, compile_incident_state(study, "low"), response_sha256="a" * 64
    )
    assert not unavailable
    gross, oep = draw.annual_losses["gross"], draw.annual_losses["oep_gross"]
    np.testing.assert_array_equal(oep, np.where(gross > 0, 200, 0))
    assert np.mean(oep == 0) == pytest.approx(np.exp(-1), abs=0.015)
    assert np.all(oep <= gross)


def test_financial_oep_precedes_annual_terms_and_retains_its_own_moment_status():
    payload = encode_scientific(_financial_study())
    payload["contracts"][0]["annual_limit"] = 5
    study = setup_from_payload(payload)
    plan = _plan(1000).model_copy(
        update={"requested_views": ("financial", "oep_financial")}
    )
    draw, unavailable, _ = simulate_incident_state(
        plan, compile_incident_state(study, "low"), response_sha256="a" * 64
    )
    assert not unavailable
    financial = draw.annual_losses["financial"]
    oep = draw.annual_losses["oep_financial"]
    np.testing.assert_array_equal(oep, np.where(financial > 0, 40, 0))
    assert np.all(financial <= 5)
    payload = encode_scientific(_financial_study(shape=1.2))
    payload["contracts"][0].update(occurrence_limit=None, annual_limit=5)
    heavy = setup_from_payload(payload)
    draw, _, _ = simulate_incident_state(
        plan, compile_incident_state(heavy, "low"), response_sha256="a" * 64
    )
    assert {name: view.mean_exists for name, view in draw.views.items()} == {
        "financial": True,
        "oep_financial": False,
    }
    assert {name: view.variance_exists for name, view in draw.views.items()} == {
        "financial": True,
        "oep_financial": False,
    }


def test_common_zero_rate_skips_unused_heavy_law_evaluation(monkeypatch):
    import catmodel.simulation.incident_runtime as runtime

    payload = encode_scientific(_financial_study(shape=1.2))
    for state in payload["joint_states"]:
        state["laws"][0]["reference_annual_rate"] = 0
    study = setup_from_payload(payload)
    monkeypatch.setattr(
        runtime,
        "native_severity_sampler",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("unused severity law evaluated")
        ),
    )
    plan = _plan(1).model_copy(
        update={"requested_views": ("gross", "oep_gross", "financial", "oep_financial")}
    )
    draw, unavailable, allocation = simulate_incident_state(
        plan, compile_incident_state(study, "low"), response_sha256="a" * 64
    )
    assert not unavailable
    assert allocation.total_initiators == 0
    assert all(np.array_equal(values, [0]) for values in draw.annual_losses.values())

    assert all({name: view.mean_exists for name, view in draw.views.items()}.values())


def test_explicit_structural_zero_portfolio_has_true_degenerate_joint_capture():
    payload = encode_scientific(_study())
    payload["alternatives"] = []
    for state in payload["joint_states"]:
        state["laws"] = []
        state["absent_populations"] = [
            {
                "scenario_id": "loss",
                "disposition": "structural_zero",
                "reason": "No selected activity in this conditional state.",
            }
        ]
    study = setup_from_payload(payload)
    plan = _plan(1).model_copy(update={"required_arms": ("reference",)})
    draw, unavailable, _ = simulate_incident_state(
        plan, compile_incident_state(study, "low"), response_sha256="a" * 64
    )
    assert not unavailable
    assert draw.joint_law_semantics == "physical_joint"
    assert all(np.array_equal(values, [0]) for values in draw.annual_losses.values())


def test_collected_view_names_reduce_physical_scopes_before_annual_terms():
    payload = encode_scientific(_financial_study())
    payload["contracts"][0]["annual_limit"] = 5
    for state in payload["joint_states"]:
        outcome = state["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0]
        outcome["components"][0]["contributor_ids"] = ["provider-a", "provider-b"]
    study = setup_from_payload(payload)
    plan = _plan(300).model_copy(
        update={
            "requested_views": (
                "gross",
                "scenario::loss::gross",
                "scenario::loss::welfare",
                "scenario::loss::net_resource_cost",
                "provider_contribution",
                "risk_unit::campaign::occurrence_net_financial",
                "financial",
            )
        }
    )
    draw, unavailable, allocations = simulate_incident_state(
        plan, compile_incident_state(study, "low"), response_sha256="a" * 64
    )
    assert not unavailable
    gross = draw.annual_losses["gross"]
    view_bases = allocations.view_bases
    for view in (
        "scenario::loss::gross",
        "scenario::loss::welfare",
        "scenario::loss::net_resource_cost",
    ):
        np.testing.assert_array_equal(draw.annual_losses[view], gross)
        assert view_bases["scenario::loss"]["currency"] == "USD"
        assert view not in view_bases
    np.testing.assert_array_equal(
        draw.annual_losses["provider_contribution"], 2 * gross
    )
    np.testing.assert_array_equal(
        draw.annual_losses["risk_unit::campaign::occurrence_net_financial"], gross * 0.4
    )
    assert np.all(draw.annual_losses["financial"] <= 5)


@pytest.mark.parametrize(("mean", "recovery_rate"), [(0, 0), (1, 1)])
def test_native_payment_endpoints_are_exact_complete_joint_marks(mean, recovery_rate):
    study = _native_payment_study(mean=mean, recovery_rate=recovery_rate)
    plan = _plan(200).model_copy(
        update={
            "requested_views": (
                "gross",
                "financial",
                "risk_unit::campaign::occurrence_net_financial",
            )
        }
    )
    draw, unavailable, _ = simulate_incident_state(
        plan, compile_incident_state(study, "low"), response_sha256="a" * 64
    )
    assert not unavailable
    np.testing.assert_allclose(
        draw.annual_losses["financial"],
        mean * draw.annual_losses["gross"],
        rtol=1e-14,
        atol=0,
    )
    np.testing.assert_allclose(
        draw.annual_losses["risk_unit::campaign::occurrence_net_financial"],
        mean * draw.annual_losses["gross"],
        rtol=1e-14,
        atol=0,
    )


def test_native_random_payment_has_uniform_tail_support_and_zero_postscale_restores_moments():
    study = _native_payment_study(mean=0.2, recovery_rate=0.4, shape=1.2)
    plan = _plan(200).model_copy(update={"requested_views": ("gross", "financial")})
    draw, unavailable, _ = simulate_incident_state(
        plan, compile_incident_state(study, "low"), response_sha256="a" * 64
    )
    assert not unavailable
    assert {name: view.mean_exists for name, view in draw.views.items()} == {
        "gross": False,
        "financial": False,
    }
    assert np.all(draw.annual_losses["financial"] <= draw.annual_losses["gross"])
    zero = _native_payment_study(mean=1, recovery_rate=1, shape=1.2, post_multiplier=0)
    draw, unavailable, _ = simulate_incident_state(
        plan, compile_incident_state(zero, "low"), response_sha256="a" * 64
    )
    assert not unavailable
    assert {name: view.mean_exists for name, view in draw.views.items()} == {
        "gross": True,
        "financial": True,
    }
    assert np.all(draw.annual_losses["gross"] == 0)
    assert np.all(draw.annual_losses["financial"] == 0)


def test_finite_large_population_is_thinned_before_member_work_admission():
    payload = encode_scientific(
        _study(rate=2, response_cost=10, unit_probability=1e-08)
    )
    for state in payload["joint_states"]:
        state["laws"][0]["segments"][0]["kernel"]["cluster"]["support"][0]["count"] = (
            100_000_000
        )
    study = setup_from_payload(payload)
    plan = _plan(1000).model_copy(
        update={"numerical_controls": {"max_campaign_members_per_year": 10}}
    )
    draw, unavailable, allocations = simulate_incident_state(
        plan, compile_incident_state(study, "low"), response_sha256="a" * 64
    )
    assert not unavailable
    assert np.mean(draw.annual_losses["gross"]) == pytest.approx(220, abs=25)
    assert allocations.sampled_potential_members > 100_000_000
    assert allocations.realized_member_work < 10000


def test_exact_prevention_skips_unused_population_and_positive_work_is_bounded():
    payload = encode_scientific(_study(rate=100, response_cost=10, unit_probability=0))
    for state in payload["joint_states"]:
        state["laws"][0]["segments"][0]["kernel"]["cluster"]["support"][0]["count"] = (
            100_000_000
        )
    study = setup_from_payload(payload)
    plan = _plan(1).model_copy(
        update={
            "numerical_controls": {
                "max_campaign_members_per_year": 20,
                # So the member allowance, not one campaign's, refuses.
                "max_members_per_campaign": 10**9,
            }
        }
    )
    draw, unavailable, allocations = simulate_incident_state(
        plan, compile_incident_state(study, "low"), response_sha256="a" * 64
    )
    assert not unavailable
    assert allocations.realized_member_work == 0
    assert draw.annual_losses["gross"].sum() == allocations.total_initiators * 10
    payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
        "unit_probability"
    ] = 1
    positive = setup_from_payload(payload)
    with pytest.raises(ValueError, match="max_campaign_members_per_year"):
        simulate_incident_state(
            plan, compile_incident_state(positive, "low"), response_sha256="a" * 64
        )
    one_campaign = plan.model_copy(
        update={
            "numerical_controls": {
                **plan.numerical_controls,
                "max_members_per_campaign": 1000,
            }
        }
    )
    with pytest.raises(ValueError, match="members exceed max_members_per_campaign"):
        simulate_incident_state(
            one_campaign,
            compile_incident_state(positive, "low"),
            response_sha256="a" * 64,
        )


@pytest.mark.parametrize("suffix", ["::gross", "::welfare", "::net_resource_cost"])
def test_undeclared_scenario_is_unavailable_even_when_no_losses_are_sampled(suffix):
    study = _study(rate=0)
    view = "scenario::undeclared" + suffix
    plan = _plan(10).model_copy(update={"requested_views": ("gross", view)})
    draw, reasons, _ = simulate_incident_state(
        plan, compile_incident_state(study, "low"), response_sha256="a" * 64
    )
    assert set(draw.annual_losses) == {"gross"}
    assert "outside the declared study population" in reasons[view]


def test_shared_gamma_and_conditional_fixed_vmr_nb_have_the_declared_joint_pgf():
    from catmodel.simulation.incident_runtime import _origin_counts
    from test_support.study import study_payload

    payload = study_payload()
    first = payload["joint_states"][0]["laws"][0]
    first.update(
        reference_annual_rate=2,
        count_family="negative_binomial",
        count_variance_to_mean=2,
    )
    second = {**first, "origin_id": "other", "reference_annual_rate": 3}
    payload["joint_states"][0]["laws"].append(second)
    payload["alternatives"] = [
        {
            "alternative_id": arm,
            "purpose": "scientific_alternative",
            "description": "Gamma intensity with conditional fixed-VMR NB counts and an independent-factor control.",
            "count_experiment": {
                "kind": "shared_gamma_intensity",
                "coefficient_of_variation": 0.5,
                "origin_groups": {"origin": "a", "other": "b"}
                if arm == "control"
                else {"origin": "shared", "other": "shared"},
            },
            "matched_control_id": "control",
        }
        for arm in ("control", "shared")
    ]
    study = setup_from_payload(payload)
    plan = _plan(120000).model_copy(
        update={"required_arms": ("reference", "control", "shared")}
    )
    variance = 0.25
    for arm in ("control", "shared"):
        counts = _origin_counts(compile_incident_state(study, "state", arm), plan)
        for origin, rate in (("origin", 2), ("other", 3)):
            assert np.mean(counts[origin]) == pytest.approx(rate, abs=0.04)
            # E[Var(N|F)] + Var(E[N|F]) = d*lambda + lambda²*CV².
            assert np.var(counts[origin]) == pytest.approx(
                2 * rate + rate**2 * variance, rel=0.04
            )

        def pgf(z, w, *, shared=arm == "shared"):
            # Conditional NB PGF is exp[-F*lambda*log(2-z)] at d=2.
            a, b = 2 * np.log(2 - z), 3 * np.log(2 - w)
            return (
                (1 + variance * (a + b)) ** (-1 / variance)
                if shared
                else (1 + variance * a) ** (-1 / variance)
                * (1 + variance * b) ** (-1 / variance)
            )

        for z, w in ((0.0, 0.0), (0.3, 0.6), (0.8, 0.4)):
            values = z ** counts["origin"] * w ** counts["other"]
            target = pgf(z, w)
            standard_error = np.sqrt((pgf(z * z, w * w) - target**2) / plan.simulations)
            assert abs(np.mean(values) - target) < 5 * standard_error
