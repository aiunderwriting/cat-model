import math
import xml.etree.ElementTree as ET
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from catmodel.contracts.law_overrides import HeldOverrides
from catmodel.results.execution import replay_capture, run_setup
from catmodel.results.respondent_sampling import borrowed_reason
from catmodel.results.sensitivity_results import (
    NO_NODE_SD,
    ONE_SAMPLE_REASON,
    REWEIGHTED_EXHIBIT_REASON,
    sensitivity_results,
)
from catmodel.simulation.capability_mixture import (
    ConditionalCapabilityDraws,
    ConditionalView,
    NativeMean,
)
from test_support.study import _package, _plan, _study

NO_SAMPLING = {"status": "unavailable", "reason": "No state is a respondent sample."}


def test_dependence_bracket_reads_the_gross_scenario_view(tmp_path):
    study = _study()
    renamed = study.alternatives[0]
    study = replace(
        study,
        alternatives=(renamed.model_copy(update={"alternative_id": "严重"}),),
        overrides=HeldOverrides({"严重": study.overrides.of(renamed.alternative_id)}),
    )
    package = _package(
        tmp_path,
        study=study,
        plan=_plan(16).model_copy(
            update={
                "requested_views": ("gross", "scenario::loss::gross"),
                "required_arms": ("reference", "严重"),
            }
        ),
    )
    report = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    bracket = report["sensitivity_summary"]["dependence_bracket"]
    assert bracket["status"] == "available"
    assert bracket["scenario_views"] == {"loss": "scenario::loss::gross"}
    assert bracket["references"]["comonotone"]["status"] == "available"
    chart = (tmp_path / "run" / "tornado.svg").read_bytes()
    assert "严重".encode() in chart
    assert ET.fromstring(chart).attrib["role"] == "img"


def test_replay_workload_retains_producer_states_when_setup_adds_unused_support(
    tmp_path,
):
    plan = _plan(2)
    package = _package(tmp_path, plan=plan)
    original = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="fixture"
    )
    study = _study()
    extra = study.joint_states[0].model_copy(
        update={"state_id": "unused", "weight": 0.0}
    )
    expanded = _package(
        tmp_path,
        study=replace(study, joint_states=(*study.joint_states, extra)),
        plan=plan,
        name="expanded",
    )
    replay = replay_capture(
        setup_path=expanded.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    assert original["workload"]["simulated_conditional_years"] == 8
    assert replay["workload"] == original["workload"]


def test_dependence_references_run_when_the_portfolio_mean_is_infinite():
    losses = {"a": [0.0, 5.0, 1.0], "b": [3.0, 0.0, 2.0]}
    views = {
        f"scenario::{key}::gross": ConditionalView(
            np.asarray(values), mean_exists=False
        )
        for key, values in losses.items()
    }
    views["gross"] = ConditionalView(
        np.asarray(losses["a"]) + np.asarray(losses["b"]), mean_exists=False
    )
    draw = ConditionalCapabilityDraws(
        stratum_sha256="state", response_sha256="response", seed=1, views=views
    )
    infinite = {
        "status": "available",
        "expected_annual_loss": None,
        "statistics": {
            "expected_annual_loss": {
                "status": "unavailable",
                "reason": "The conditional law has no finite mean.",
            }
        },
    }
    package = SimpleNamespace(
        execution=SimpleNamespace(
            reference_arm="reference",
            for_arm=lambda arm: SimpleNamespace(numerical_controls={}),
        ),
        scope=SimpleNamespace(
            period_start=SimpleNamespace(year=2027),
            populations=(
                SimpleNamespace(scenario_id="a", currency="USD"),
                SimpleNamespace(scenario_id="b", currency="USD"),
            ),
        ),
        alternatives=(),
        evidence=SimpleNamespace(records=()),
    )
    report = {
        "status": "partial",
        "coverage": {"comparisons": {"status": "not_requested"}},
        "arms": {"reference": {"metrics": dict.fromkeys(views, infinite)}},
    }
    summary = sensitivity_results(
        package,
        report,
        {"state": draw},
        {"state": 1.0},
        levels=(0.7,),
        reweighted=False,
        sampling=NO_SAMPLING,
    )
    assert summary["comparison_status"] == "not_requested"
    assert summary["tornado"]["view"] == "gross"
    bracket = summary["dependence_bracket"]
    assert bracket["status"] == "available"
    assert bracket["mean_unavailable_reason"] is not None
    for name, var in (("executed", 5), ("comonotone", 8)):
        reference = bracket["references"][name]
        assert reference["expected_annual_loss"] is None
        assert {key: reference["tail"]["0.7"][key] for key in ("var", "tvar")} == {
            "var": var,
            "tvar": None,
        }
    # The plausibility table reports what the run reports: no finite welfare
    # EAL, so no share.
    plausibility = summary["plausibility"]
    assert plausibility["status"] == "available"
    assert plausibility["total"]["status"] == "unavailable"
    assert plausibility["flagged_scenarios"] == []


def test_reweighted_targets_do_not_inherit_setup_weight_exhibits():
    """The trajectory quadrature exhibit's reference EAL is the setup-weighted
    sum of the states' conditional means, and the frequency exhibit is frozen
    at setup weights; neither describes a reweighted target."""
    from catmodel.simulation.incident_runtime import state_key

    means = {"a": 10.0, "b": 30.0}
    draws = {
        state_key(key): ConditionalCapabilityDraws(
            stratum_sha256=key,
            response_sha256="response",
            seed=1,
            views={
                "gross": ConditionalView(
                    np.asarray([0.0, 2 * mean]),
                    native_mean=NativeMean(
                        mean,
                        "native_analytic_mean",
                        {
                            "numerical_standard_error": 0.0,
                            "standard_error_upper_bound": 0.0,
                            "precision_basis": "deterministic_integration",
                            "integration_error_bound": 0.0,
                        },
                    ),
                )
            },
        )
        for key, mean in means.items()
    }
    record = {
        "rule": "clenshaw_curtis",
        "interior_nodes": 3,
        "embedded_interior_nodes": 1,
        "draw_weights": {
            "reference": {
                "variant": "base",
                "full": {"n1": 0.5, "n2": 0.5},
                "embedded": {"n1": 0.4, "n2": 0.6},
            }
        },
        "node_progress": {"n1": 1.0, "n2": 2.0},
        "state_draws": {"a": "n1", "b": "n2"},
    }
    package = SimpleNamespace(
        execution=SimpleNamespace(
            reference_arm="reference",
            for_arm=lambda arm: SimpleNamespace(numerical_controls={}),
        ),
        scope=SimpleNamespace(
            period_start=SimpleNamespace(year=2027),
            populations=(SimpleNamespace(scenario_id="a", currency="USD"),),
        ),
        alternatives=(),
        joint_states=(
            SimpleNamespace(
                state_id="a",
                weight=0.5,
                sampling_group_id="n1:m",
                borrowed_leaves=(),
                lab_loss_left_out=False,
                laws=(),
            ),
            SimpleNamespace(
                state_id="b",
                weight=0.5,
                sampling_group_id="n2:m",
                borrowed_leaves=(),
                lab_loss_left_out=False,
                laws=(),
            ),
        ),
        evidence=SimpleNamespace(
            records=(
                SimpleNamespace(acquisition={"trajectory_quadrature": record}),
                SimpleNamespace(
                    acquisition={
                        "frequency_estimate": {
                            "views": {},
                            "hierarchy": {"omega": {}},
                        }
                    }
                ),
            )
        ),
    )
    report = {
        "status": "reduced",
        "coverage": {"comparisons": {"status": "not_requested"}},
        "arms": {
            "reference": {
                "metrics": {
                    "gross": {
                        "status": "available",
                        "expected_annual_loss": 20.0,
                        "statistics": {
                            "expected_annual_loss": {
                                "status": "available",
                                "reason": None,
                            }
                        },
                    }
                }
            }
        },
    }

    def summary(weights, reweighted):
        return sensitivity_results(
            package,
            report,
            draws,
            weights,
            levels=(0.5,),
            reweighted=reweighted,
            sampling=NO_SAMPLING,
        )

    own = summary({state_key("a"): 0.5, state_key("b"): 0.5}, False)
    reference = own["trajectory_quadrature"]["by_arm"]["reference"]
    assert reference["expected_annual_loss"] == 0.5 * 10 + 0.5 * 30
    # Each node holds one sample, so neither has a standard deviation.
    assert {
        (row["state_eal_sd"], row["state_eal_sd_reason"])
        for row in reference["by_node"].values()
    } == {(None, ONE_SAMPLE_REASON)}
    assert own["frequency_estimate"]["views"] == {}
    target = {state_key("a"): 0.9, state_key("b"): 0.1}
    reweighted = summary(target, True)
    for key in (
        "trajectory_quadrature",
        "frequency_estimate",
        "hierarchy_pool",
    ):
        assert reweighted[key]["status"] == "unavailable"
        assert reweighted[key]["reason"] == REWEIGHTED_EXHIBIT_REASON
    assert "by_arm" not in reweighted["trajectory_quadrature"]


def _two_node_quadrature(borrowed_leaves=None):
    """The trajectory quadrature exhibit of two nodes, one model and three
    samples each: n1 with state EALs 2, 4, 9 and n2 with 5, 30, 55, at a full
    rule of 0.5/0.5 and an embedded rule of 0.4/0.6. ``borrowed_leaves`` gives
    the leaves a state reads from a donor, by state."""
    from catmodel.simulation.incident_runtime import state_key

    borrowed_leaves = borrowed_leaves or {}
    values = {"a0": 2.0, "a1": 4.0, "a2": 9.0, "b0": 5.0, "b1": 30.0, "b2": 55.0}
    node = {state: "n1" if state.startswith("a") else "n2" for state in values}
    draws = {
        state_key(state): ConditionalCapabilityDraws(
            stratum_sha256=state,
            response_sha256="response",
            seed=1,
            views={
                "gross": ConditionalView(
                    np.asarray([value]), native_mean=NativeMean(value, "exact")
                )
            },
        )
        for state, value in values.items()
    }
    record = {
        "rule": "fejer_second",
        "interior_nodes": 3,
        "embedded_interior_nodes": 1,
        "draw_weights": {
            "reference": {
                "variant": "base",
                "full": {"n1": 0.5, "n2": 0.5},
                "embedded": {"n1": 0.4, "n2": 0.6},
            }
        },
        "node_progress": {"n1": 0.8, "n2": 1.6},
        "state_draws": node,
    }
    package = SimpleNamespace(
        execution=SimpleNamespace(
            reference_arm="reference",
            for_arm=lambda arm: SimpleNamespace(numerical_controls={}),
        ),
        scope=SimpleNamespace(
            period_start=SimpleNamespace(year=2027),
            populations=(SimpleNamespace(scenario_id="a", currency="USD"),),
        ),
        alternatives=(),
        joint_states=tuple(
            SimpleNamespace(
                state_id=state,
                weight=0.5 / 3,
                sampling_group_id=f"{node[state]}:m",
                borrowed_leaves=borrowed_leaves.get(state, ()),
                lab_loss_left_out=False,
                laws=(),
            )
            for state in values
        ),
        evidence=SimpleNamespace(
            records=(SimpleNamespace(acquisition={"trajectory_quadrature": record}),)
        ),
    )
    report = {
        "status": "reduced",
        "coverage": {"comparisons": {"status": "not_requested"}},
        "arms": {
            "reference": {
                "metrics": {
                    "gross": {
                        "status": "available",
                        "expected_annual_loss": 17.5,
                        "statistics": {
                            "expected_annual_loss": {
                                "status": "available",
                                "reason": None,
                            }
                        },
                    }
                }
            }
        },
    }
    summary = sensitivity_results(
        package,
        report,
        draws,
        {state_key(state): 0.5 / 3 for state in values},
        levels=(0.5,),
        reweighted=False,
        sampling={"status": "available", "noise_z": 2.0, "views": {}},
    )
    return summary["trajectory_quadrature"]


def test_quadrature_indicator_reports_its_sampling_error_and_nodes():
    """Two nodes, one model, three samples each, by hand.

    n1: EALs 2, 4, 9 (mean 5, s^2 13); n2: 5, 30, 55 (mean 30, s^2 625).
    Full rule 0.5/0.5 gives 17.5, embedded 0.4/0.6 gives 20; the difference
    -2.5 has SE^2 = (0.1^2 * 13 + 0.1^2 * 625) / 3, so z = -1.714: within
    sampling noise.
    """
    from catmodel.results.results_summary import _Summary

    quadrature = _two_node_quadrature()
    row = quadrature["by_arm"]["reference"]
    assert row["expected_annual_loss"] == pytest.approx(17.5, rel=1e-12)
    assert row["embedded_rule_expected_annual_loss"] == pytest.approx(20, rel=1e-12)
    error = math.sqrt(0.01 * (13 + 625) / 3)
    assert row["sampling_standard_error"] == pytest.approx(error, rel=1e-12)
    assert row["sampling_z"] == pytest.approx(-2.5 / error, rel=1e-12)
    assert row["sampling_reason"] is None
    expected = {
        "n1": (0.8, 0.5, 0.4, 5.0, math.sqrt(13), 2.5, 2.0),
        "n2": (1.6, 0.5, 0.6, 30.0, 25.0, 15.0, 18.0),
    }
    assert set(row["by_node"]) == set(expected)
    for name, (progress, full, embedded, mean, sd, inside, outside) in expected.items():
        found = row["by_node"][name]
        assert (found["progress"], found["full_weight"], found["embedded_weight"]) == (
            progress,
            full,
            embedded,
        )
        assert found["samples"] == 3
        assert found["state_eal_mean"] == pytest.approx(mean, rel=1e-12)
        assert found["state_eal_sd"] == pytest.approx(sd, rel=1e-12)
        assert found["state_eal_sd_reason"] is None
        assert found["full_contribution"] == pytest.approx(inside, rel=1e-12)
        assert found["embedded_contribution"] == pytest.approx(outside, rel=1e-12)
        assert found["difference_contribution"] == pytest.approx(
            inside - outside, rel=1e-12
        )
    assert math.fsum(
        found["difference_contribution"] for found in row["by_node"].values()
    ) == pytest.approx(row["absolute_difference"], rel=1e-12)
    # The results summary reads the indicator against its sampling error.
    rendered = _Summary(
        {
            "analysis_spec": {"reference_arm": "reference"},
            "arms": {"reference": {"metrics": {}}},
            "sensitivity_summary": {
                "respondent_sampling": {
                    "status": "available",
                    "samples": 6,
                    "groups": 2,
                    "smallest_group": 3,
                    "noise_z": 2.0,
                    "views": {},
                },
                "trajectory_quadrature": quadrature,
            },
        }
    )
    rendered.respondent_sampling()
    text = "\n".join(rendered.lines)
    assert "does not distinguish integration error from respondent-sampling noise" in (
        text
    )
    assert (
        "| `reference` | informative | 17.5 | 20 | -2.5 | 1.46 | -1.71 | within sampling noise |"
        in text
    )


def test_a_node_with_a_borrowing_sample_reports_no_standard_deviation():
    """The same two nodes, with the third sample of n1 reading a donor's
    answers to leaf x. The standard deviation of n1's state EALs would count
    three independent samples, so it is withheld with its reason, as the
    arm's standard error is. n2 holds no borrowing sample and keeps its own,
    and every weight, mean and contribution is as without the borrowing."""
    independent = _two_node_quadrature()["by_arm"]["reference"]
    row = _two_node_quadrature({"a2": ("x",)})["by_arm"]["reference"]
    reason = (
        "State 'a2' reads a donor's answers to leaf 'x' (POOL-UNANSWERED-SCENARIOS), "
        "so the node's samples are not independent and no standard deviation is "
        "reported."
    )
    assert reason.endswith(NO_NODE_SD)
    assert reason == borrowed_reason("a2", ("x",), NO_NODE_SD)
    assert row["by_node"]["n1"] == {
        **independent["by_node"]["n1"],
        "state_eal_sd": None,
        "state_eal_sd_reason": reason,
    }
    assert row["by_node"]["n2"] == independent["by_node"]["n2"]
    assert row["by_node"]["n2"]["state_eal_sd"] == pytest.approx(25.0, rel=1e-12)
    assert (row["sampling_standard_error"], row["sampling_z"]) == (None, None)
    assert row["sampling_reason"] == borrowed_reason("a2", ("x",))
    for key in ("expected_annual_loss", "absolute_difference", "indicator"):
        assert row[key] == independent[key]
