"""Independent finite laws for capture reduction, completeness and precision."""

import json
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.hashing import canonical_json_sha256
from catmodel.results.reduction import (
    ReductionSpec,
    comparison,
    reduce_captures,
    tail_diagnostics,
)
from catmodel.results.respondent_sampling import SampleGroups
from catmodel.simulation.capability_mixture import (
    CapabilityMixture,
    ConditionalCapabilityDraws,
    ConditionalView,
    NativeMean,
    effective_sample_size,
)
from test_support.results import present, statistic_ranking, view_diagnostics
from test_support.run_setup import setup_from_payload


def draw(state="s", values=(0, 80, 40, 40), **kwargs):
    view_options = {
        key: kwargs.pop(key)
        for key in ("mean_exists", "variance_exists", "native_mean")
        if key in kwargs
    }
    return ConditionalCapabilityDraws(
        stratum_sha256=state,
        response_sha256="response",
        seed=1,
        views={
            "gross": ConditionalView(np.asarray(values, dtype=float), **view_options)
        },
        **kwargs,
    )


def reduce(draws=None, **kwargs):
    return reduce_captures(
        scientific_sha256="study",
        plan_sha256="plan",
        draws_by_arm=draws or {"base": {"s": draw()}},
        state_weights=kwargs.pop("state_weights", {"s": 1.0}),
        required_arms=kwargs.pop("required_arms", ("base",)),
        reference_arm="base",
        views=kwargs.pop("views", ("gross",)),
        reducer_provenance=kwargs.pop("reducer_provenance", {"commit": "current"}),
        spec=kwargs.pop("spec", ReductionSpec()),
        **kwargs,
    )


def support_metadata(records):
    shared, bindings = {}, {}
    for arm, states in records.items():
        bindings[arm] = {}
        for state, record in states.items():
            key = canonical_json_sha256(record)
            shared[key] = record
            bindings[arm][state] = key
    return {"support_records": shared, "support_by_arm_state": bindings}


def test_physical_joint_round_trip_oracle():
    # Ten equally likely years: five without loss, one at 80 and four at 40.
    source = draw(
        values=(0,) * 5 + (80,) + (40,) * 4,
        joint_law_semantics="physical_joint",
        evaluator_provenance_by_risk_unit={
            "fixture": {
                "joint_law_semantics": "physical_joint",
                "method": "authored_exact_joint_support",
            }
        },
    )
    source = replace(
        source,
        views={
            "gross": ConditionalView(source.annual_losses["gross"]),
            "resource": ConditionalView(
                np.array((0.0,) * 5 + (40.0,) * 3 + (0.0,) * 2)
            ),
        },
    )
    result = reduce(
        {"base": {"s": source}},
        views=("gross", "resource"),
        spec=ReductionSpec(joint_thresholds=(("gross", 60), ("resource", 20))),
        capture_provenance={
            "grain": "annual",
            "physical_joint_semantics": "physical_joint_samples",
        },
    )
    assert result["arms"]["base"]["joint_exceedance_probability"] == pytest.approx(0.1)
    metrics = result["arms"]["base"]["metrics"]["gross"]
    assert metrics["expected_annual_loss"] == pytest.approx(24)
    assert metrics["annual_probability_positive"] == pytest.approx(0.5)
    assert set(result["coverage"]) == {"comparisons", "views", "sensitivity_families"}
    json.dumps(result, allow_nan=False)


def test_annual_capture_rejects_false_joint():
    with pytest.raises(ValueError, match="physical joint"):
        reduce(spec=ReductionSpec(joint_thresholds=(("gross", 1),)))


def test_missing_required_arm_withholds_complete_comparison():
    result = reduce(required_arms=("base", "higher"))
    assert result["status"] == "partial"
    assert result["coverage"]["comparisons"]["missing_arms"] == ["higher"]
    assert not result["rankings_available"]
    assert result["comparisons"]["higher"]["gross"]["status"] == "unavailable"


def test_absolute_effect_survives_zero_reference_and_tiny_reference_is_finite():
    result = reduce(
        {"base": {"s": draw(values=(0, 0))}, "higher": {"s": draw(values=(10, 10))}},
        required_arms=("base", "higher"),
    )
    effect = result["comparisons"]["higher"]["gross"]["expected_annual_loss"]
    assert effect["absolute_difference"] == 10
    assert effect["alternative"] == 10
    assert effect["relative_change"] is None
    tiny = comparison(1e-310, 1)
    assert tiny["absolute_difference"] == 1
    assert tiny["relative_change"] is None
    assert "numeric range" in present(tiny["relative_change_unavailable_reason"])
    json.dumps(tiny, allow_nan=False)


def test_rankings_follow_each_functional_and_ignore_capture_dictionary_order():
    captures = {
        "base": {"s": draw(values=(0, 10, 0, 10))},
        "location": {"s": draw(values=(11, 11, 11, 11))},
        "tail": {"s": draw(values=(0, 0, 0, 40))},
    }
    result = reduce(captures, required_arms=("base", "location", "tail"))
    reversed_result = reduce(
        dict(reversed(list(captures.items()))),
        required_arms=("base", "location", "tail"),
    )
    ranking = result["rankings"]["gross"]
    assert [item["arm"] for item in ranking["expected_annual_loss"]["arms"]] == [
        "location",
        "tail",
    ]
    assert [item["arm"] for item in ranking["annual_loss_variance"]["arms"]] == [
        "tail",
        "location",
    ]
    # Four years cannot converge a tail, so its effects are not ranked.
    assert ranking["tvar::0.99"]["status"] == "withheld"
    assert result["rankings"] == reversed_result["rankings"]
    assert (
        ranking["expected_annual_loss"]["basis"]
        == "absolute_point_effect_without_precision_ordering"
    )


def test_rankings_use_one_reference_while_scientific_stresses_keep_matched_controls():
    captures = {
        arm: {"s": draw(values=(value,) * 4)}
        for arm, value in {
            "base": 100,
            "control": 1000,
            "stress": 1010,
            "other": 200,
        }.items()
    }
    result = reduce(
        captures,
        required_arms=("base", "control", "stress", "other"),
        matched_controls={"control": "control", "stress": "control"},
    )
    ranking = result["rankings"]["gross"]["expected_annual_loss"]
    assert ranking["reference_arm"] == "base"
    assert [row["arm"] for row in ranking["arms"]] == ["stress", "control", "other"]
    assert [row["absolute_difference"] for row in ranking["arms"]] == [910, 900, 100]
    comparisons = result["comparisons"]
    assert (
        comparisons["control"]["gross"]["expected_annual_loss"]["absolute_difference"]
        == 0
    )
    assert (
        comparisons["stress"]["gross"]["expected_annual_loss"]["absolute_difference"]
        == 10
    )
    assert result["comparison_reference_arms"]["stress"] == "control"
    # Only arms whose matched control is not the reference arm are compared
    # with it again; "other" has no control, so its comparison is in
    # ``comparisons``.
    assert set(result["reference_comparisons"]) == {"control", "stress"}
    stress = result["reference_comparisons"]["stress"]["gross"]["expected_annual_loss"]
    assert stress["absolute_difference"] == 910
    assert comparisons["other"]["gross"]["expected_annual_loss"]["reference"] == 100


def test_weighted_mixture_tail_preserves_fractional_quantile_atom():
    result = reduce(
        {
            "base": {
                "low": draw("low", (0, 0, 0, 10)),
                "high": draw("high", (0, 10, 20, 30)),
            }
        },
        state_weights={"low": 0.75, "high": 0.25},
        spec=ReductionSpec(levels=(0.8,)),
    )
    metric = result["arms"]["base"]["metrics"]["gross"]
    assert metric["expected_annual_loss"] == pytest.approx(0.75 * 2.5 + 0.25 * 15)
    point = metric["tail_metrics"]["0.8"]
    tail = view_diagnostics(result, "base", "gross")["tail_diagnostics"]["tails"]["0.8"]
    assert point["var"] == 10
    assert tail["strict_exceedance_count"] == 2
    assert tail["strict_exceedance_probability"] == 0.125
    assert point["tvar"] == pytest.approx(
        (0.0625 * 20 + 0.0625 * 30 + 0.075 * 10) / 0.2
    )
    assert point["var_numerical_interval_95"] is None
    assert "needs at least 16" in present(tail["var_interval_unavailable_reason"])


def test_tail_support_counts_independent_years():
    source = draw(values=(0, 2, 10, 12))
    mixture = CapabilityMixture({"s": source}, {"s": 1.0})
    tail = tail_diagnostics(mixture, "gross", levels=(0.5,))["tails"]["0.5"]
    support = tail["tail_support_by_stratum"]["s"]
    assert tail["var"] == 2
    assert tail["tail_weight_effective_sample_size"] == 2
    assert support["independent_unit_count"] == 4
    assert support["tail_bearing_independent_unit_count"] == 2
    assert support["tail_mass_effective_units"] == 2
    # Excesses 8 and 10 over the VaR: Kish ESS (8 + 10)^2 / (8^2 + 10^2).
    assert support["tail_excess_effective_units"] == pytest.approx(324 / 164)
    assert tail["tail_mass_effective_units"] == 2
    assert tail["var_numerical_interval_95"] is None
    assert "needs at least 16" in present(tail["var_interval_unavailable_reason"])


def test_native_eal_and_empirical_mean_keep_separate_estimators_and_precision():
    source = draw(
        values=(0, 80, 40, 40) * 4,
        native_mean=NativeMean(
            7,
            "analytic_full_conditional_law",
            precision={
                "numerical_standard_error": 0,
                "standard_error_upper_bound": 0,
                "precision_basis": "analytic_full_conditional_law",
            },
        ),
    )
    result = reduce({"base": {"s": source}})
    metric = result["arms"]["base"]["metrics"]["gross"]
    diagnostics = view_diagnostics(result, "base", "gross")
    assert metric["expected_annual_loss"] == 7
    assert metric["eal_estimator"] == "analytic_full_conditional_law"
    assert diagnostics["eal_estimator_by_stratum"] == {
        "s": "analytic_full_conditional_law"
    }
    assert metric["normalized_empirical_mean"] == 40
    assert metric["eal_precision"]["standard_error_upper_bound"] == 0
    assert "stratum_estimators" not in metric["eal_precision"]
    assert diagnostics["eal_precision_by_stratum"]["s"]["numerical_standard_error"] == 0
    mean = diagnostics["mean_diagnostics"]
    # Hand oracle: 16 batches of one year each.
    assert mean["empirical_mean_standard_error"] == pytest.approx(
        np.std((0, 80, 40, 40) * 4, ddof=1) / 4
    )
    assert mean["empirical_mean_numerical_interval_95"] is None
    assert metric["tail_metrics"]["0.99"]["var"] == 80


@pytest.mark.parametrize("outside", [False, True])
def test_common_explicit_defense_component_keeps_transformed_moment_admission(outside):

    from catmodel.simulation.incident import compile_incident_state
    from catmodel.simulation.incident_runtime import simulate_incident_state
    from test_support.study import _financial_study, _plan

    payload = encode_scientific(_financial_study(shape=0.6))
    payload["contracts"][0]["occurrence_retention"] = 0
    for state in payload["joint_states"]:
        outcome = state["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0]
        outcome["components"][0]["loss_type"] = "defense"
        outcome["memberships"][0]["outside_limits"] = outside
    study = setup_from_payload(payload)
    plan = _plan(128).model_copy(
        update={"required_arms": ("reference",), "requested_views": ("financial",)}
    )
    source, unavailable, _ = simulate_incident_state(
        plan, compile_incident_state(study, "low"), response_sha256="a" * 64
    )
    assert not unavailable
    result = reduce(
        {"base": {source.stratum_sha256: source}},
        state_weights={source.stratum_sha256: 1.0},
        views=("financial",),
    )
    metric = result["arms"]["base"]["metrics"]["financial"]
    assert metric["expected_annual_loss"] is not None
    assert source.views["financial"].variance_exists is (not outside)
    diagnostics = view_diagnostics(result, "base", "financial")
    tail = diagnostics["tail_diagnostics"]["tails"]["0.99"]
    interval = metric["tail_metrics"]["0.99"]["tvar_numerical_interval_95"]
    if outside:
        assert interval is None
        assert "second moment" in present(tail["tvar_interval_unavailable_reason"])
    else:
        convergence = present(tail["convergence"])
        assert present(convergence["tvar"])["half_width"] is not None
        assert interval is not None


def test_current_analysis_identity_changes_without_rewriting_capture_identity():
    first = reduce()
    assert first["analysis_sha256"] == reduce()["analysis_sha256"]
    changed = reduce(spec=ReductionSpec(levels=(0.95,)))
    code = reduce(reducer_provenance={"commit": "changed"})
    assert (
        first["analysis_sha256"]
        != changed["analysis_sha256"]
        != code["analysis_sha256"]
    )
    assert first["analysis_spec"]["captures"] == changed["analysis_spec"]["captures"]


def test_sparse_joint_state_groups_are_preserved_before_mixing():
    result = reduce(
        {
            "base": {
                "a-high": draw("a-high", (10, 10)),
                "b-low": draw("b-low", (10, 10)),
            }
        },
        state_weights={"a-high": 0.5, "b-low": 0.5},
        epistemic_groups={"a-high": "a", "b-low": "b"},
    )
    arm = result["arms"]["base"]
    assert arm["metrics"]["gross"]["expected_annual_loss"] == 10
    assert arm["conditional_eal_by_epistemic_group"] == {"gross": {"a": 10, "b": 10}}
    with pytest.raises(ValueError, match="positive reweighting"):
        reduce(state_weights={"s": 0.5, "absent": 0.5})


def test_identifiable_gross_survives_unavailable_composition_across_states():
    result = reduce(
        {
            "base": {
                "a": draw("a", (10, 10)),
                "b": replace(
                    draw("b", (20, 20)),
                    views={
                        "gross": ConditionalView(np.array([20.0, 20.0])),
                        "resource": ConditionalView(np.array([10.0, 10.0])),
                    },
                ),
            }
        },
        state_weights={"a": 0.5, "b": 0.5},
        views=("gross", "resource"),
        capture_provenance=support_metadata(
            {
                "base": {
                    "a": {
                        "view_bases": {"pooled": basis()},
                        "unavailable_views": {
                            "resource": "Composition is not identified."
                        },
                    },
                    "b": {"view_bases": {"pooled": basis()}},
                }
            }
        ),
    )
    metrics = result["arms"]["base"]["metrics"]
    assert metrics["gross"]["expected_annual_loss"] == 15
    assert metrics["resource"]["status"] == "unavailable"
    assert metrics["resource"]["state_reasons"]["a"] == "Composition is not identified."
    assert result["status"] == "partial"


def test_infinite_variance_withholds_tvar_interval_but_retains_var_and_points():
    source = draw(values=(0, 80, 40, 40) * 1000, variance_exists=False)
    result = reduce({"base": {"s": source}})
    metrics = result["arms"]["base"]["metrics"]["gross"]
    point = metrics["tail_metrics"]["0.99"]
    tail = view_diagnostics(result, "base", "gross")["tail_diagnostics"]["tails"]
    assert metrics["expected_annual_loss"] == pytest.approx(40)
    assert point["tvar"] == pytest.approx(80)
    assert point["var_numerical_interval_95"] == [80, 80]
    assert point["tvar_numerical_interval_95"] is None
    assert "second moment" in present(tail["0.99"]["tvar_interval_unavailable_reason"])


def test_zero_weight_heavy_state_precision_is_reassessed_when_positive():
    finite = draw("finite", (0, 80, 40, 40) * 4)
    heavy = draw("heavy", (0, 80, 40, 40) * 4, variance_exists=False)
    draws = {"base": {"finite": finite, "heavy": heavy}}
    first = reduce(draws, state_weights={"finite": 1.0, "heavy": 0.0})
    second = reduce(draws, state_weights={"finite": 0.5, "heavy": 0.5})

    def standard_error(result):
        return view_diagnostics(result, "base", "gross")["mean_diagnostics"][
            "empirical_mean_standard_error"
        ]

    assert standard_error(first) is not None
    assert standard_error(second) is None


def test_a_positive_law_without_sampled_losses_has_no_tail_precision():
    source = draw(values=(0,) * 32, native_mean=NativeMean(0.0003, "analytic"))
    mixture = CapabilityMixture({"s": source}, {"s": 1.0})
    tail = tail_diagnostics(mixture, "gross")["tails"]["0.99"]
    assert tail["positive_year_count"] == 0
    assert tail["tail_weight_effective_sample_size"] == 0
    assert tail["var_numerical_interval_95"] is None
    assert tail["tvar_numerical_interval_95"] is None
    assert "No positive loss samples" in tail["var_interval_unavailable_reason"]
    json.dumps(tail, allow_nan=False)


def test_public_zero_rate_capture_is_known_zero_and_keeps_mixture_precision():
    from catmodel.simulation.incident import compile_incident_state
    from catmodel.simulation.incident_runtime import simulate_incident_state
    from test_support.study import _plan, _study

    study = _study(rate=0)
    plan = _plan(64).model_copy(update={"requested_views": ("gross",)})
    zero, unavailable, _ = simulate_incident_state(
        plan, compile_incident_state(study, "low"), response_sha256="a" * 64
    )
    assert not unavailable
    assert zero.views["gross"].native_mean is not None
    assert zero.views["gross"].native_mean.value == 0
    zero_result = reduce(
        {"base": {zero.stratum_sha256: zero}},
        state_weights={zero.stratum_sha256: 1.0},
    )
    metrics = zero_result["arms"]["base"]["metrics"]["gross"]
    mean = view_diagnostics(zero_result, "base", "gross")["mean_diagnostics"]
    assert metrics["eal_precision"]["numerical_standard_error"] == 0
    assert mean["empirical_mean_standard_error"] == 0
    assert mean["empirical_mean_numerical_interval_95"] == [0, 0]
    tail = metrics["tail_metrics"]["0.99"]
    assert tail["var_numerical_interval_95"] == [0, 0]
    assert tail["tvar_numerical_interval_95"] == [0, 0]
    mixed_result = reduce(
        {
            "base": {
                zero.stratum_sha256: zero,
                # Sixty-four years, as the no-progress state has; sorted, so the
                # batches differ.
                "positive": draw("positive", sorted((1, 2, 3, 4) * 16)),
            }
        },
        state_weights={zero.stratum_sha256: 0.5, "positive": 0.5},
    )
    mixed = view_diagnostics(mixed_result, "base", "gross")
    assert (
        mixed_result["arms"]["base"]["metrics"]["gross"]["expected_annual_loss"] == 1.25
    )
    assert present(mixed["mean_diagnostics"]["empirical_mean_standard_error"]) > 0
    # No positive year lies above the 99% VaR, the largest loss.
    row = present(mixed["tail_diagnostics"]["tails"]["0.99"]["convergence"])
    assert row["var"]["status"] == "unconverged"


def test_positive_mean_below_numeric_range_cannot_certify_an_exact_zero_law():
    from decimal import Decimal

    from catmodel.simulation.incident import compile_incident_state
    from catmodel.simulation.incident_runtime import simulate_incident_state
    from test_support.study import _plan, _study

    payload = encode_scientific(_study(rate=1e-316))
    for state in payload["joint_states"]:
        state["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0]["components"][
            0
        ]["amount"] = 1e-9
    study = setup_from_payload(payload)
    plan = _plan(64).model_copy(update={"requested_views": ("gross",)})
    compiled = compile_incident_state(study, "low")
    source, unavailable, allocation = simulate_incident_state(
        plan, compiled, response_sha256="a" * 64
    )
    assert 2 * Decimal("1e-316") * Decimal("1e-9") > 0
    assert not unavailable
    assert {name: view.mean_exists for name, view in source.views.items()} == {
        "gross": True
    }
    assert source.views["gross"].native_mean is None
    functional = allocation.functional_unavailable_by_view
    reason = functional["gross"]["expected_annual_loss"]
    assert "range" in reason
    key = source.stratum_sha256
    result = reduce(
        {"base": {key: source}},
        state_weights={key: 1.0},
        capture_provenance=support_metadata(
            {
                "base": {
                    key: {
                        "view_bases": compiled.view_bases,
                        "unavailable_functionals": {
                            "gross": {"expected_annual_loss": reason}
                        },
                    }
                }
            }
        ),
    )
    metrics = result["arms"]["base"]["metrics"]["gross"]
    assert metrics["expected_annual_loss"] is None
    assert metrics["normalized_empirical_mean"] == 0
    mean = view_diagnostics(result, "base", "gross")["mean_diagnostics"]
    assert mean["empirical_mean_standard_error"] is None
    tail = metrics["tail_metrics"]["0.99"]
    assert tail["var_numerical_interval_95"] is None
    assert tail["tvar_numerical_interval_95"] is None


def test_rounded_native_mean_zero_needs_an_explicit_zero_law_proof():
    from catmodel.simulation.capability_mixture import DETERMINISTIC_EAL_PRECISION_BASIS

    source = draw(
        values=(0, 0, 0, 0),
        native_mean=NativeMean(
            0,
            "native_analytic_mean",
            precision={
                "precision_basis": DETERMINISTIC_EAL_PRECISION_BASIS,
                "numerical_standard_error": 0,
                "standard_error_upper_bound": 0,
                "integration_error_bound": 0,
            },
        ),
    )
    result = reduce({"base": {"s": source}})
    metric = result["arms"]["base"]["metrics"]["gross"]
    mean = view_diagnostics(result, "base", "gross")["mean_diagnostics"]
    assert mean["empirical_mean_standard_error"] is None
    assert metric["tail_metrics"]["0.99"]["var_numerical_interval_95"] is None


def test_tiny_lawful_tail_weights_have_stable_ess_and_finite_json():
    assert effective_sample_size(np.full(100, 1e-200)) == pytest.approx(100)
    assert effective_sample_size(np.array([1.0, 1.0, 2.0])) == pytest.approx(16 / 6)
    mixture = CapabilityMixture(
        {key: draw(key, (value,)) for key, value in (("a", 0), ("b", 1), ("c", 2))},
        {"a": 1.0, "b": 1e-200, "c": 1e-200},
    )
    tail = tail_diagnostics(mixture, "gross")["tails"]["0.99"]
    assert tail["tail_weight_effective_sample_size"] == pytest.approx(2)
    json.dumps(tail, allow_nan=False)


def test_target_functional_existence_is_independent_of_finite_sample_values():
    result = reduce(
        capture_provenance=support_metadata(
            {
                "base": {
                    "s": {
                        "view_bases": {"pooled": basis()},
                        "unavailable_functionals": {
                            "gross": {
                                "expected_annual_loss": "Target has infinite mean.",
                                "tvar": "Target has infinite mean.",
                            }
                        },
                    }
                }
            }
        )
    )
    metrics = result["arms"]["base"]["metrics"]["gross"]
    assert metrics["expected_annual_loss"] is None
    assert metrics["normalized_empirical_mean"] == 40
    assert metrics["tail_metrics"]["0.99"]["var"] == 80
    assert metrics["tail_metrics"]["0.99"]["tvar"] is None
    mean = view_diagnostics(result, "base", "gross")["mean_diagnostics"]
    assert mean["empirical_mean_standard_error"] is None
    json.dumps(result, allow_nan=False)


def test_draw_mean_admission_withholds_infinite_target_eal_and_tvar():
    source = draw(
        values=(0, 80, 40, 40) * 1000, mean_exists=False, variance_exists=False
    )
    mixture = CapabilityMixture({"s": source}, {"s": 1.0})
    metrics = mixture.metrics("gross")
    assert metrics["expected_annual_loss"] is None
    assert metrics["normalized_empirical_mean"] == pytest.approx(40)
    tails = cast(dict[str, Any], metrics["tail_metrics"])
    assert tails["0.99"]["var"] == 80
    assert tails["0.99"]["tvar"] is None
    statistics = cast(dict[str, Any], metrics["statistics"])
    assert statistics["tvar::0.99"]["status"] == "unavailable"
    assert statistics["tvar::0.99"]["reason"]
    tail = tail_diagnostics(mixture, "gross")["tails"]["0.99"]
    assert tail["var_numerical_interval_95"] == [80, 80]
    assert tail["tvar"] is None
    assert tail["tvar_numerical_interval_95"] is None
    reduced = reduce({"base": {"s": source}})
    reported = reduced["arms"]["base"]["metrics"]["gross"]
    assert reported["expected_annual_loss"] is None
    assert reported["tail_metrics"]["0.99"]["var_numerical_interval_95"] == [80, 80]
    json.dumps(reduced, allow_nan=False)


def test_mean_existence_survives_capture_and_changed_positive_state_weights(tmp_path):
    from catmodel.results.capability_storage import (
        decode_conditional_draw,
        encode_conditional_draw,
    )

    first = replace(draw("a" * 64), response_sha256="d" * 64)
    second = replace(
        draw("b" * 64, mean_exists=False, variance_exists=False),
        response_sha256="d" * 64,
    )
    loaded = {}
    for source in (first, second):
        record, data = encode_conditional_draw(source)
        loaded[source.stratum_sha256] = decode_conditional_draw(
            record,
            data,
            stratum_sha256=source.stratum_sha256,
            response_sha256=source.response_sha256,
            seed=source.seed,
        )
    assert not loaded[second.stratum_sha256].views["gross"].mean_exists
    finite = CapabilityMixture(
        loaded, {first.stratum_sha256: 1, second.stratum_sha256: 0}
    )
    heavy = CapabilityMixture(
        loaded, {first.stratum_sha256: 0.5, second.stratum_sha256: 0.5}
    )
    assert finite.metrics("gross")["expected_annual_loss"] == 40
    assert heavy.metrics("gross")["expected_annual_loss"] is None
    tail = tail_diagnostics(heavy, "gross")["tails"]["0.99"]
    assert tail["var"] == 80
    assert tail["tvar"] is None
    json.dumps(tail, allow_nan=False)


def test_aligned_arrays_and_broad_metadata_do_not_establish_physical_joint_law():
    with pytest.raises(ValueError, match="evaluator provenance"):
        reduce(
            spec=ReductionSpec(joint_thresholds=(("gross", 1),)),
            capture_provenance={"physical_joint_semantics": "physical_joint_samples"},
        )
    with pytest.raises(ValueError, match="constituent evaluator"):
        draw(
            joint_law_semantics="physical_joint",
            evaluator_provenance_by_risk_unit={
                "row": {"joint_law_semantics": "marginal_only"}
            },
        )


def test_conditional_epistemic_mean_does_not_underflow_with_tiny_group_mass():
    result = reduce(
        {
            "base": {
                "rare": draw("rare", (1e-100, 1e-100)),
                "usual": draw("usual", (10, 10)),
            }
        },
        state_weights={"rare": 1e-300, "usual": 1.0},
        epistemic_groups={"rare": "rare-group", "usual": "usual-group"},
    )
    assert result["arms"]["base"]["conditional_eal_by_epistemic_group"]["gross"][
        "rare-group"
    ] == pytest.approx(1e-100, rel=1e-12, abs=0)


def test_one_independent_year_retains_points_without_precision():
    result = reduce({"base": {"s": draw(values=(30,))}})
    metrics = result["arms"]["base"]["metrics"]["gross"]
    diagnostics = view_diagnostics(result, "base", "gross")
    assert metrics["expected_annual_loss"] == 30
    mean = diagnostics["mean_diagnostics"]
    assert mean["empirical_mean_standard_error"] is None
    assert "fewer than two independent" in present(mean["interval_unavailable_reason"])
    for level, tail in diagnostics["tail_diagnostics"]["tails"].items():
        point = metrics["tail_metrics"][level]
        assert point["var"] == point["tvar"] == 30
        assert point["var_numerical_interval_95"] is None
        assert point["tvar_numerical_interval_95"] is None
        for reason in (
            tail["var_interval_unavailable_reason"],
            tail["tvar_interval_unavailable_reason"],
        ):
            assert "fewer than two independent" in present(reason)


def test_matched_control_contrast_retains_variance_change_when_eal_matches():
    result = reduce(
        {
            # Thirty years each, since effects compare the arms' common years.
            "base": {"s": draw(values=(0, 30, 30) * 10)},
            "independent": {"s": draw(values=(0, 40) * 15)},
            "shared": {"s": draw(values=(0, 0, 0, 50, 50) * 6)},
        },
        required_arms=("base", "independent", "shared"),
        matched_controls={"independent": "independent", "shared": "independent"},
    )
    assert [
        result["arms"][arm]["metrics"]["gross"]["annual_loss_variance"]
        for arm in ("base", "independent", "shared")
    ] == pytest.approx([200, 400, 600])
    contrast = result["comparisons"]["shared"]["gross"]
    assert contrast["expected_annual_loss"]["absolute_difference"] == pytest.approx(
        0, abs=1e-12
    )
    assert contrast["annual_loss_variance"]["reference"] == pytest.approx(400)
    assert contrast["annual_loss_variance"]["absolute_difference"] == pytest.approx(200)
    assert result["comparison_reference_arms"]["shared"] == "independent"
    assert result["reference_comparisons"]["shared"]["gross"]["annual_loss_variance"][
        "absolute_difference"
    ] == pytest.approx(400)
    with pytest.raises(ValueError, match="transitive control"):
        reduce(
            required_arms=("base", "shared"), matched_controls={"shared": "independent"}
        )


def basis(currency="USD", **changes):
    return {
        "currency": currency,
        "monetary_basis": "nominal_2025",
        "accounting_standing": "unique_harm",
        "severity_unit": "victim",
        **changes,
    }


@pytest.mark.parametrize(
    "changed",
    [
        {"currency": "EUR"},
        {"monetary_basis": "nominal_2024"},
        {"accounting_standing": "contributor_view"},
    ],
)
def test_reweight_withholds_incompatible_pooled_bases_and_retains_scoped_view(changed):
    scoped = "scenario::one::gross"
    first = replace(
        draw("a", (10, 20)),
        views={
            "gross": ConditionalView(np.array([10.0, 20.0])),
            scoped: ConditionalView(np.array([10.0, 20.0])),
        },
    )
    second = replace(
        draw("b", (30, 40)),
        views={
            "gross": ConditionalView(np.array([30.0, 40.0])),
            scoped: ConditionalView(np.array([30.0, 40.0])),
        },
    )
    captures = {"base": {"a": first, "b": second}}
    metadata = support_metadata(
        {
            "base": {
                "a": {"view_bases": {"pooled": basis(), "scenario::one": basis()}},
                "b": {
                    "view_bases": {"pooled": basis(**changed), "scenario::one": basis()}
                },
            }
        }
    )
    original = reduce(
        captures,
        state_weights={"a": 1.0, "b": 0.0},
        views=("gross", scoped),
        capture_provenance=metadata,
    )
    assert original["arms"]["base"]["metrics"]["gross"]["expected_annual_loss"] == 15
    changed_result = reduce(
        captures,
        state_weights={"a": 0.5, "b": 0.5},
        views=("gross", scoped),
        capture_provenance=metadata,
    )
    metrics = changed_result["arms"]["base"]["metrics"]
    assert metrics["gross"]["status"] == "unavailable"
    assert "incompatible" in metrics["gross"]["reason"]
    assert "expected_annual_loss" not in metrics["gross"]
    assert metrics[scoped]["expected_annual_loss"] == 25
    assert changed_result["status"] == "partial"


def test_scenario_severity_units_and_matched_control_bases_must_agree():
    scoped = "scenario::one::gross"
    source = replace(
        draw(),
        views={scoped: ConditionalView(np.array([10.0, 20.0]))},
    )
    other = replace(
        draw("other"),
        views={scoped: ConditionalView(np.array([30.0, 40.0]))},
    )
    result = reduce(
        {"base": {"s": source}, "other": {"s": source}},
        required_arms=("base", "other"),
        views=(scoped,),
        capture_provenance=support_metadata(
            {
                "base": {"s": {"view_bases": {"scenario::one": basis()}}},
                "other": {
                    "s": {"view_bases": {"scenario::one": basis(currency="EUR")}}
                },
            }
        ),
    )
    assert result["arms"]["other"]["metrics"][scoped]["status"] == "available"
    assert result["comparisons"]["other"][scoped]["status"] == "unavailable"
    assert result["status"] == "partial"
    units = reduce(
        {"base": {"s": source, "other": other}},
        state_weights={"s": 0.5, "other": 0.5},
        views=(scoped,),
        capture_provenance=support_metadata(
            {
                "base": {
                    "s": {"view_bases": {"scenario::one": basis()}},
                    "other": {
                        "view_bases": {"scenario::one": basis(severity_unit="campaign")}
                    },
                }
            }
        ),
    )
    assert units["arms"]["base"]["metrics"][scoped]["status"] == "unavailable"


def test_variance_of_equal_years_and_constant_atoms_is_exact():
    source = draw(values=(1, 2))
    result = reduce({"base": {"s": source}})
    assert result["arms"]["base"]["metrics"]["gross"]["annual_loss_variance"] == 0.25
    constant = reduce({"base": {"s": draw(values=(30,) * 100)}})
    assert constant["arms"]["base"]["metrics"]["gross"]["annual_loss_variance"] == 0


def requirement(
    family, arm=None, *, status="required", scenarios=("one",), reason=None
):
    return {
        "family_id": family,
        "arm_id": arm,
        "scenario_ids": scenarios,
        "status": status,
        "reason": reason,
        "held_fixed": ("All quantities outside the declared perturbation scope.",),
    }


def test_arm_specific_weights_reduce_sparse_joint_support_without_rebalancing():
    captures = {
        "base": {"a": draw("a", (10, 10))},
        "alternative": {"b": draw("b", (100, 100))},
    }
    weights = {"base": {"a": 1.0, "b": 0.0}, "alternative": {"a": 0.0, "b": 1.0}}
    result = reduce(
        captures,
        required_arms=("base", "alternative"),
        state_weights=weights["base"],
        state_weights_by_arm=weights,
        epistemic_groups={"a": "group-a", "b": "group-b"},
    )
    assert result["arms"]["base"]["metrics"]["gross"]["expected_annual_loss"] == 10
    assert (
        result["arms"]["alternative"]["metrics"]["gross"]["expected_annual_loss"] == 100
    )
    assert result["arms"]["alternative"]["conditional_eal_by_epistemic_group"] == {
        "gross": {"group-b": 100}
    }
    assert (
        result["comparisons"]["alternative"]["gross"]["expected_annual_loss"][
            "absolute_difference"
        ]
        == 90
    )
    assert result["analysis_spec"]["state_weights_by_arm"] == weights
    with pytest.raises(ValueError, match="positive reweighting"):
        reduce(
            captures,
            required_arms=("base", "alternative"),
            state_weights=weights["base"],
            state_weights_by_arm={**weights, "alternative": {"a": 0.5, "b": 0.5}},
        )


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"base": {"s": 1}, "extra": {"s": 1}},
        {"base": {"other": 1}},
        {"base": {"s": 0.9}},
        {"base": {"s": True}},
    ],
)
def test_arm_weights_require_the_frozen_full_support_and_total_mass(weights):
    with pytest.raises(
        ValueError, match=r"arm weights|full declared support|state weights"
    ):
        reduce(state_weights_by_arm=weights)


def test_arm_weights_reassess_precision_only_on_each_arm_target_support():
    captures = {
        arm: {
            "finite": draw("finite", (0, 80, 40, 40) * 4),
            "heavy": draw("heavy", (0, 80, 40, 40) * 4, variance_exists=False),
        }
        for arm in ("base", "stress")
    }
    result = reduce(
        captures,
        required_arms=("base", "stress"),
        state_weights={"finite": 1.0, "heavy": 0.0},
        state_weights_by_arm={
            "base": {"finite": 1.0, "heavy": 0.0},
            "stress": {"finite": 0.0, "heavy": 1.0},
        },
    )
    base = result["arms"]["base"]["metrics"]["gross"]["tail_metrics"]["0.99"]
    stress = result["arms"]["stress"]["metrics"]["gross"]["tail_metrics"]["0.99"]
    assert base["tvar_numerical_interval_95"] is not None
    assert stress["tvar_numerical_interval_95"] is None
    tails = view_diagnostics(result, "stress", "gross")["tail_diagnostics"]["tails"]
    assert "second moment" in present(tails["0.99"]["tvar_interval_unavailable_reason"])


def test_unavailable_family_preserves_complete_family_effects_and_rankings():
    requirements = (
        requirement("severity", "lower"),
        requirement("severity", "higher"),
        requirement(
            "liability",
            status="unavailable",
            scenarios=("two",),
            reason="No joint legal outcome is identified.",
        ),
        requirement(
            "calendar",
            status="inapplicable",
            reason="This is an origin-year ultimate study.",
        ),
    )
    result = reduce(
        {
            arm: {"s": draw(values=(value,) * 4)}
            for arm, value in {"base": 10, "lower": 8, "higher": 20}.items()
        },
        required_arms=("base", "lower", "higher"),
        comparison_requirements=requirements,
    )
    assert result["status"] == "partial"
    assert not result["rankings_available"]
    families = result["comparison_families"]
    assert families["severity"]["status"] == "complete"
    ranks = families["severity"]["rankings"]["gross"]["expected_annual_loss"]
    assert [row["arm"] for row in ranks["arms"]] == ["higher", "lower"]
    assert ranks["reference_arm"] == "base"
    assert families["liability"]["status"] == "partial"
    assert families["calendar"]["status"] == "inapplicable"
    unavailable = result["comparison_requirements"][2]
    assert unavailable["scenario_ids"] == ["two"]
    assert unavailable["reason"] == "No joint legal outcome is identified."
    assert unavailable["scope_basis"] == "authored_perturbation_scope"
    assert (
        result["comparisons"]["lower"]["gross"]["expected_annual_loss"][
            "absolute_difference"
        ]
        == -2
    )


def test_missing_family_arm_preserves_the_available_absolute_effect():
    result = reduce(
        {"base": {"s": draw(values=(10, 10))}, "lower": {"s": draw(values=(8, 8))}},
        required_arms=("base", "lower", "higher"),
        comparison_requirements=(
            requirement("severity", "lower"),
            requirement("severity", "higher"),
        ),
    )
    family = result["comparison_families"]["severity"]
    assert family["status"] == "partial"
    assert result["comparison_requirements"][0]["view_outcomes"] == {
        "gross": {"status": "available"}
    }
    assert result["comparison_requirements"][1]["outcome_status"] == "missing"
    assert (
        result["comparisons"]["lower"]["gross"]["expected_annual_loss"][
            "absolute_difference"
        ]
        == -2
    )
    # Incomplete coverage leaves every statistic of the view unranked, for
    # one reason.
    assert family["rankings"]["gross"]["status"] == "unavailable"
    assert "incomplete" in family["rankings"]["gross"]["reason"]


def test_unrelated_unavailable_scenario_does_not_hide_complete_scoped_family():
    scoped, unrelated = "scenario::one::gross", "scenario::two::gross"
    captures = {
        arm: {
            "s": replace(
                draw(values=(value, value)),
                views={
                    "gross": ConditionalView(np.array([value, value], dtype=float)),
                    scoped: ConditionalView(np.array([value, value], dtype=float)),
                },
            )
        }
        for arm, value in {"base": 10, "higher": 20}.items()
    }
    result = reduce(
        captures,
        required_arms=("base", "higher"),
        views=("gross", scoped, unrelated),
        comparison_requirements=(requirement("one-only", "higher"),),
    )
    family = result["comparison_families"]["one-only"]
    assert result["status"] == "partial"
    assert family["status"] == "complete"
    assert unrelated not in family["rankings"]
    assert family["rankings"][scoped]["expected_annual_loss"]["status"] == "available"


def test_family_capture_completeness_does_not_assert_functional_existence():
    result = reduce(
        {
            "base": {"s": draw()},
            "heavy": {"s": draw(mean_exists=False, variance_exists=False)},
        },
        required_arms=("base", "heavy"),
        comparison_requirements=(requirement("tail", "heavy"),),
    )
    family = result["comparison_families"]["tail"]
    assert family["status"] == "complete"
    rankings = family["rankings"]
    assert (
        statistic_ranking(rankings, "gross", "expected_annual_loss")["status"]
        == "unavailable"
    )
    # The tail gate withholds four-year tails, and so their effects' ranking.
    assert statistic_ranking(rankings, "gross", "var::0.99")["status"] == "withheld"
    assert "tail gate" in family["rankings"]["gross"]["var::0.99"]["reason"]


def _analytic(value, state="s"):
    return draw(
        state,
        native_mean=NativeMean(
            value,
            "native_analytic_mean",
            precision={
                "numerical_standard_error": 0,
                "standard_error_upper_bound": 0,
                "precision_basis": "deterministic_integration",
                "integration_error_bound": 0,
            },
        ),
    )


# The views that count the lab's own loss, the lab-loss views and the gross
# views, with a state left out of them (POOL-EXCLUDED-SAMPLES): one sampling
# group of three states at 0.2 each and another of two.
LAB = "lab_loss::lab"
LAB_WEIGHTS = {"a0": 0.2, "a1": 0.2, "a2": 0.2, "b0": 0.2, "b1": 0.2}
LAB_GROUPS = SampleGroups(
    {"a0": "n:a", "a1": "n:a", "a2": "n:a", "b0": "n:b", "b1": "n:b"},
    left_out=frozenset({"a0"}),
)
WELFARE_YEARS = {"a0": 100, "a1": 1, "a2": 3, "b0": 5, "b1": 7}
LAB_YEARS = {"a1": 10, "a2": 30, "b0": 50, "b1": 70}
VIEWS = ("welfare", "gross", "scenario::x::gross", LAB)
NO_STATE_LEFT = (
    "Every state with weight in sampling group 'n:a' is left out of the views "
    "that count the lab's own loss, because a case it reads declares that loss "
    "unavailable (POOL-EXCLUDED-SAMPLES), so no state carries the group's weight "
    "in them."
)


def _lab_draws(lab_years=LAB_YEARS, **options):
    """Each state's two equal years in every view. Gross is welfare plus the
    lab's loss; ``a0``, left out of the views that count the lab's loss, has
    no lab-loss view, and its gross is its welfare alone."""
    draws = {}
    for state, welfare in WELFARE_YEARS.items():
        gross = np.full(2, float(welfare + lab_years.get(state, 0)))
        draws[state] = ConditionalCapabilityDraws(
            stratum_sha256=state,
            response_sha256="response",
            seed=1,
            views={
                "welfare": ConditionalView(np.full(2, float(welfare))),
                "gross": ConditionalView(gross),
                "scenario::x::gross": ConditionalView(gross),
                **(
                    {LAB: ConditionalView(np.full(2, float(lab_years[state])))}
                    if state in lab_years
                    else {}
                ),
            },
            **options,
        )
    return draws


def _lab_reduce(**kwargs):
    return reduce(
        {"base": _lab_draws()},
        state_weights=LAB_WEIGHTS,
        views=VIEWS,
        epistemic_groups={state: state[0] for state in LAB_WEIGHTS},
        **kwargs,
    )


def test_the_views_that_count_the_lab_loss_are_reduced_at_their_own_weights():
    # Without the sampling groups a state that lacks the lab-loss view
    # withholds it, and every state weighs alike in gross.
    plain = _lab_reduce()["arms"]["base"]
    assert plain["metrics"][LAB] == {
        "status": "unavailable",
        "reason": "Requested view was not retained in every positive-weight state.",
        "state_reasons": {"a0": "View was not retained."},
    }
    assert plain["metrics"]["gross"]["expected_annual_loss"] == pytest.approx(55.2)
    assert "lab_loss_state_weights" not in plain
    arm = _lab_reduce(sample_groups=LAB_GROUPS)["arms"]["base"]
    # The welfare view keeps a0 at its weight: 0.2 * (100 + 1 + 3 + 5 + 7).
    assert arm["metrics"]["welfare"] == plain["metrics"]["welfare"]
    assert arm["metrics"]["welfare"]["expected_annual_loss"] == pytest.approx(23.2)
    # The lab-loss view gives a0's 0.2 to a1 and a2, 0.3 each:
    # 0.3 * (10 + 30) + 0.2 * (50 + 70) = 36.
    assert arm["status"] == "reduced" and arm["missing_views"] == []
    assert arm["metrics"][LAB]["expected_annual_loss"] == pytest.approx(36.0)
    # So do the gross views, which add the lab's loss:
    # 0.3 * (11 + 33) + 0.2 * (55 + 77) = 39.6, the scenario's the same.
    for view in ("gross", "scenario::x::gross"):
        assert arm["metrics"][view]["expected_annual_loss"] == pytest.approx(39.6)
    assert arm["lab_loss_state_weights"] == pytest.approx(
        {"a0": 0.0, "a1": 0.3, "a2": 0.3, "b0": 0.2, "b1": 0.2}
    )
    # Each epistemic group's EAL mixes its states at the view's weights.
    assert arm["conditional_eal_by_epistemic_group"] == {
        "welfare": {"a": pytest.approx(104 / 3), "b": pytest.approx(6.0)},
        "gross": {"a": pytest.approx(22.0), "b": pytest.approx(66.0)},
        "scenario::x::gross": {"a": pytest.approx(22.0), "b": pytest.approx(66.0)},
        LAB: {"a": pytest.approx(20.0), "b": pytest.approx(60.0)},
    }
    # A left-out state that holds the lab-loss view all the same enters no
    # number of these views: the weights decide, not what the state retained.
    held = reduce(
        {"base": _lab_draws(LAB_YEARS | {"a0": 1e9})},
        state_weights=LAB_WEIGHTS,
        views=VIEWS,
        sample_groups=LAB_GROUPS,
    )["arms"]["base"]
    assert held["metrics"][LAB]["expected_annual_loss"] == pytest.approx(36.0)
    assert held["metrics"]["gross"]["expected_annual_loss"] == pytest.approx(39.6)
    assert held["metrics"]["welfare"] == plain["metrics"]["welfare"]


def test_an_arm_whose_group_weighs_only_left_out_states_has_no_view_of_the_lab_loss():
    alone = {"a0": 0.6, "a1": 0.0, "a2": 0.0, "b0": 0.2, "b1": 0.2}
    draws = _lab_draws()
    result = reduce(
        {"base": draws, "alone": draws},
        required_arms=("base", "alone"),
        state_weights=LAB_WEIGHTS,
        state_weights_by_arm={"base": LAB_WEIGHTS, "alone": alone},
        views=VIEWS,
        sample_groups=LAB_GROUPS,
    )
    arm = result["arms"]["alone"]
    for view in ("gross", "scenario::x::gross", LAB):
        assert arm["metrics"][view] == {
            "status": "unavailable",
            "reason": NO_STATE_LEFT,
            "state_reasons": {},
        }
        assert result["comparisons"]["alone"][view]["status"] == "unavailable"
    assert arm["status"] == "partial"
    assert arm["missing_views"] == sorted(("gross", "scenario::x::gross", LAB))
    assert "lab_loss_state_weights" not in arm
    # Its welfare view is reported at its own weights: 0.6*100 + 0.2*(5+7).
    assert arm["metrics"]["welfare"]["expected_annual_loss"] == pytest.approx(62.4)
    assert result["comparisons"]["alone"]["welfare"]["expected_annual_loss"][
        "absolute_difference"
    ] == pytest.approx(62.4 - 23.2)
    # The arm that weighs the group's other states reports them.
    assert result["arms"]["base"]["metrics"][LAB]["expected_annual_loss"] == (
        pytest.approx(36.0)
    )


def test_gross_is_withheld_where_a_state_without_the_lab_loss_is_not_left_out():
    """A state that keeps its weight without a law of the lab's loss would
    omit the lab's annual business loss from gross, so the arm reports no
    gross view; its lab-loss view is unavailable because the state does not
    hold it, and its welfare view is unchanged."""
    plain = _lab_reduce()["arms"]["base"]
    arm = _lab_reduce(
        sample_groups=SampleGroups(
            LAB_GROUPS.group, without_lab_loss={"a0": "state-a0"}
        )
    )["arms"]["base"]
    for view in ("gross", "scenario::x::gross"):
        assert arm["metrics"][view] == {
            "status": "unavailable",
            "reason": "State 'state-a0' has weight and holds no law of the lab's "
            "own loss, so its gross loss would omit the lab's annual business "
            "loss, and it is not left out of the views that count that loss "
            "(POOL-EXCLUDED-SAMPLES).",
            "state_reasons": {},
        }
    assert arm["metrics"][LAB] == plain["metrics"][LAB]
    assert arm["metrics"]["welfare"] == plain["metrics"]["welfare"]
    assert "lab_loss_state_weights" not in arm


def _lab_joint(
    *thresholds,
    groups: SampleGroups | None = LAB_GROUPS,
    views=VIEWS,
    support: dict[str, Any] | None = None,
):
    """The arm of ``_lab_draws`` as physical joint samples, reduced with
    joint ``thresholds``; ``support`` is its capture's support metadata."""
    physical = {
        "joint_law_semantics": "physical_joint",
        "evaluator_provenance_by_risk_unit": {
            "fixture": {
                "joint_law_semantics": "physical_joint",
                "method": "authored_exact_joint_support",
            }
        },
    }
    return reduce(
        {"base": _lab_draws(**physical)},
        state_weights=LAB_WEIGHTS,
        views=views,
        sample_groups=groups,
        spec=ReductionSpec(joint_thresholds=thresholds),
        capture_provenance={
            "grain": "annual",
            "physical_joint_semantics": "physical_joint_samples",
            **(support or {}),
        },
    )["arms"]["base"]


def test_joint_exceedance_is_withheld_across_views_with_different_weights():
    # The welfare view and a view that counts the lab's loss weigh the states
    # differently here, so no year's losses in the two are read together.
    for other in ("gross", LAB):
        mixed = _lab_joint(("welfare", 4), (other, 40))
        assert "joint_exceedance_probability" not in mixed
        assert mixed["joint_exceedance_convergence"] is None
        assert mixed["joint_exceedance_status"] == {
            "status": "unavailable",
            "reason": "The joint thresholds name a view that counts the lab's own "
            "loss (a lab-loss view or a gross view) and a view that does not. "
            "They weigh the arm's states differently, because a state is left "
            "out of the views that count the lab's loss (POOL-EXCLUDED-SAMPLES), "
            "so no year's losses in them are read together.",
        }
        assert mixed["metrics"]["welfare"]["status"] == "available"
    # Where no state is left out, welfare and gross share the arm's weights:
    # a0 (100, 100), b0 (5, 55) and b1 (7, 77) exceed both, at 0.2 each.
    assert _lab_joint(("welfare", 4), ("gross", 40), groups=None)[
        "joint_exceedance_probability"
    ] == pytest.approx(0.6)
    # Thresholds on views that count the lab's loss use their weights: b0 and
    # b1 exceed 40 in the lab-loss view and 50 in gross, at 0.2 each. On the
    # welfare view alone, the arm's: a0, b0 and b1.
    assert _lab_joint((LAB, 40), ("gross", 50))["joint_exceedance_probability"] == (
        pytest.approx(0.4)
    )
    assert _lab_joint(("welfare", 4))["joint_exceedance_probability"] == pytest.approx(
        0.6
    )


def test_joint_exceedance_is_unavailable_where_a_threshold_view_is_not_retained():
    """A run that withholds the lab-loss views leaves no sample out: ``a0``
    keeps its weight and holds no lab-loss law, so the arm does not retain
    the lab-loss view and reports no joint exceedance over it."""
    withheld = SampleGroups(LAB_GROUPS.group, without_lab_loss={"a0": "state-a0"})
    no_law = "The state holds no law of the lab's own loss."
    records: dict[str, dict[str, Any]] = {
        state: {"view_bases": {"pooled": basis()}} for state in LAB_WEIGHTS
    }
    records["a0"]["unavailable_views"] = {LAB: no_law}

    def joint(*thresholds, views=("welfare", LAB)):
        return _lab_joint(
            *thresholds,
            groups=withheld,
            views=views,
            support=support_metadata({"base": records}),
        )

    arm = joint(("welfare", 4), (LAB, 40))
    assert "joint_exceedance_probability" not in arm
    assert arm["joint_exceedance_convergence"] is None
    assert arm["joint_exceedance_status"] == {
        "status": "unavailable",
        "reason": f"The joint thresholds name the view {LAB!r}, which is "
        "unavailable in this arm: Requested view was not retained in every "
        f"positive-weight state. The states that lack it give: {no_law}",
    }
    # The view's own metric carries the same reasons, and every view's metrics
    # are those of a reduction whose thresholds do not name it: on the welfare
    # view alone a0, b0 and b1 exceed 4, at 0.2 each.
    assert arm["metrics"][LAB] == {
        "status": "unavailable",
        "reason": "Requested view was not retained in every positive-weight state.",
        "state_reasons": {"a0": no_law},
    }
    alone = joint(("welfare", 4))
    assert alone["joint_exceedance_probability"] == pytest.approx(0.6)
    assert arm["metrics"] == alone["metrics"]
    assert arm["metrics"]["welfare"]["expected_annual_loss"] == pytest.approx(23.2)
    assert arm["status"] == alone["status"] == "partial"
    assert arm["missing_views"] == alone["missing_views"] == [LAB]
    # A threshold on a view that the reduction does not report and the arm
    # does not capture has no metric to give a reason, and is refused.
    with pytest.raises(ValueError, match="not one of the reduction's views"):
        joint(("welfare", 4), (LAB, 40), views=("welfare",))
    with pytest.raises(ValueError, match="'lab_los::lab'"):
        joint(("welfare", 4), ("lab_los::lab", 40))
