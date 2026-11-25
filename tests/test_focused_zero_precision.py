"""Incident precision uses declared zero support, never a sampled zero."""

from copy import deepcopy

import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.results.reduction import mean_diagnostics, tail_diagnostics
from catmodel.simulation.capability_mixture import CapabilityMixture
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_runtime import simulate_incident_state
from test_support.run_setup import setup_from_payload
from test_support.study import _financial_study, _plan


def _study(*, zero=None, shape=0.6):
    payload = encode_scientific(_financial_study(shape=shape))
    payload["alternatives"] = []
    for state in payload["joint_states"]:
        law = state["laws"][0]
        component = law["segments"][0]["kernel"]["member_outcomes"][0]["components"][0]
        if zero == "rate":
            law["reference_annual_rate"] = 0
        elif zero == "coefficient":
            component["multiplier"] = 0
        elif zero in {"underflow", "response_underflow"}:
            law["reference_annual_rate"] = 1e-200
            if zero == "underflow":
                law["segments"][0]["initiation_probability"] = 1e-200
            else:
                law["response"]["points"][0]["multiplier"] = 1e-200
    return setup_from_payload(payload)


def _run(study, *, years=16, views=("gross",)):
    plan = _plan(years).model_copy(update={"requested_views": views})
    draw, unavailable, _ = simulate_incident_state(
        plan,
        compile_incident_state(study, "low"),
        response_sha256=study.scientific_sha256,
    )
    assert not unavailable
    return draw, CapabilityMixture(
        {draw.stratum_sha256: draw}, {draw.stratum_sha256: 1}
    )


@pytest.mark.parametrize("zero", ["rate", "coefficient"])
@pytest.mark.parametrize("shape", [0.6, 1.2])
@pytest.mark.parametrize("years", [1, 16])
def test_nested_exact_zero_support_restores_zero_se_and_intervals(zero, shape, years):
    draw, mixture = _run(_study(zero=zero, shape=shape), years=years)
    assert any(
        view.native_mean is not None and view.native_mean.precision is not None
        for view in draw.views.values()
    )

    for view in draw.annual_losses:
        assert draw.views[view].variance_exists
        assert draw.views[view].native_mean.precision["known_zero_law"] is True
        assert draw.views[view].native_mean.precision["numerical_standard_error"] == 0
        metric = mixture.metrics(view)
        assert metric["expected_annual_loss"] == 0
        assert metric["eal_precision"]["standard_error_upper_bound"] == 0
        mean = mean_diagnostics(mixture, view)
        tail = tail_diagnostics(mixture, view, levels=(0.99,))["tails"]["0.99"]
        assert mean["empirical_mean_standard_error"] == 0
        assert mean["empirical_mean_numerical_interval_95"] == [0, 0]
        assert mean["interval_unavailable_reason"] is None
        assert tail["var_numerical_interval_95"] == [0, 0]
        assert tail["var_interval_method"] == "known_zero_law"
        assert tail["tvar_numerical_interval_95"] == [0, 0]
        assert tail["var_interval_unavailable_reason"] is None
        assert tail["tvar_interval_unavailable_reason"] is None


def test_positive_heavy_support_withholds_mean_precision_and_gates_tvar_interval():
    draw, mixture = _run(_study(), years=128)

    assert not draw.views["gross"].variance_exists
    native = draw.views["gross"].native_mean
    assert native is not None
    assert not (native.precision or {}).get("known_zero_law", False)
    mean = mean_diagnostics(mixture, "gross")
    tail = tail_diagnostics(mixture, "gross", levels=(0.99,))["tails"]["0.99"]
    assert mean["empirical_mean_standard_error"] is None
    assert "second moment" in mean["empirical_mean_standard_error_unavailable_reason"]
    assert mean["interval_unavailable_reason"] is not None
    # Infinite variance fixes the control-variate coefficient at one, so the
    # TVaR's error is a bounded mean's and its batch half-width is defined;
    # 128 years leave too few exceedances to converge.
    assert tail["tvar_estimator"]["coefficient"] == 1.0
    assert tail["tvar_numerical_standard_error"] >= 0
    assert tail["convergence"]["tvar"]["status"] == "unconverged"
    assert tail["convergence"]["minimum_batch_exceedances"] < 10


def test_scoped_zero_view_keeps_proof_beside_positive_heavy_population():
    payload = encode_scientific(_study(zero="coefficient"))
    payload["populations"].append(
        {**payload["populations"][0], "scenario_id": "positive"}
    )
    for state in payload["joint_states"]:
        other = deepcopy(state["laws"][0])
        other.update(origin_id="positive", scenario_id="positive")
        other["segments"][0]["kernel"]["member_outcomes"][0]["components"][0][
            "multiplier"
        ] = 1
        state["laws"].append(other)
    draw, mixture = _run(
        setup_from_payload(payload), years=128, views=("gross", "scenario::loss::gross")
    )

    assert not draw.views["gross"].variance_exists
    view = "scenario::loss::gross"
    assert draw.views[view].variance_exists
    tail = tail_diagnostics(mixture, view, levels=(0.99,))["tails"]["0.99"]
    assert tail["tvar_numerical_interval_95"] == [0, 0]


@pytest.mark.parametrize("zero", ["underflow", "response_underflow"])
def test_positive_intensity_below_float_range_is_not_a_structural_zero(zero):
    with pytest.raises(
        ValueError, match=r"positive integrated initiating intensity.*range"
    ):
        compile_incident_state(_study(zero=zero), "low")
