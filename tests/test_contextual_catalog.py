"""Synthetic numeric oracles for catalog totals."""

import math

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.special import ndtri

from catmodel.contracts.contextual_risk import (
    ContextualRiskAnswer,
    ContextualRiskInstrument,
)
from catmodel.contracts.incident import IncidentLaw
from catmodel.contracts.native_severity import (
    NativeSeverityLaw,
    native_severity_sampler,
)
from catmodel.elicitation.contextual_risk import (
    contextual_node_from_scenario,
)
from catmodel.elicitation.epistemic import epistemic_quantiles
from catmodel.resources import scenarios_path
from catmodel.results.conditional import ConditionalCapture
from catmodel.results.execution import run_setup
from catmodel.results.results_export import export_results
from catmodel.scenarios.loader import (
    load_scenarios,
)
from catmodel.setup.contextual_node import CompileOptions, compile_contextual_node
from catmodel.setup.contextual_setup import compile_contextual_setup
from catmodel.setup.sensitivity_catalog import FRONTIER_LAB_VALUE_USD
from catmodel.setup.store import build_setup, load_setup
from catmodel.simulation.incident import compile_incident_state
from test_support.components import sample_component_amounts
from test_support.contextual_risk import (
    contextual_instrument,
    fixture_artifact,
    scenario_node_answer,
)
from test_support.results import reduced_arm

_IDS = (
    "developer_control_loss",
    "agent_actions_for_deployer",
    "enforcement_matter",
    "government_designation",
)


def _catalog(*, all_nodes=False):
    return {
        row.id: contextual_node_from_scenario(row, "deployment", "public")
        for row in load_scenarios(scenarios_path())
        if all_nodes or row.id in _IDS
    }


def _instrument(node):
    payload = contextual_instrument().model_dump(mode="json")
    payload["nodes"] = [node.model_dump(mode="json")]
    return ContextualRiskInstrument.model_validate(payload)


def _artifact(instrument, answer):
    return fixture_artifact(instrument, lambda _prompt: answer)


def test_all_catalog_routes_compile_from_authored_definitions(tmp_path):
    nodes = _catalog(all_nodes=True)
    assert set(nodes) == {row.id for row in load_scenarios(scenarios_path())}
    for node in nodes.values():
        assert node.unavailable_reason is None
        instrument = _instrument(node)
        composed_setup = compile_contextual_setup(
            instrument,
            _artifact(instrument, scenario_node_answer(node)),
            setup_id=node.node_id,
            years_per_state=4,
        )
        study = composed_setup
        for state in study.joint_states:
            compiled = compile_incident_state(study, state.state_id)
            assert compiled.incidents
        if node.node_id not in _IDS:
            continue
        root = tmp_path / node.node_id
        package = build_setup(
            destination=root / "package",
            setup=composed_setup,
            code_commit="synthetic-fixture",
        )
        report = run_setup(
            setup_path=package.root,
            output=root / "run",
            code_commit="synthetic-fixture",
        )
        capture = ConditionalCapture(package, root / "run/capture")
        kernel = capture.kernel_at(instrument.context_id, 1, "reference")
        welfare, weights = kernel.samples("welfare")
        lab, lab_weights = kernel.samples("lab_loss::lab")
        assert len(welfare) == len(lab) == 4
        np.testing.assert_array_equal(weights, lab_weights)
        for view in ("welfare", "lab_loss::lab"):
            assert (
                reduced_arm(report["arms"]["reference"])["metrics"][view]["status"]
                == "available"
            )
        index = export_results(
            setup_path=package.root,
            capture_directory=root / "run/capture",
            results_path=root / "run/results.json",
            output=root / "export",
            thresholds=(0.0, 1e6, 1e9),
            code_commit="synthetic-fixture",
        )
        assert index.conditioning["status"] == "available"
        assert all(
            row.status == "available"
            for row in index.series
            if row.view in ("welfare", "lab_loss::lab")
        )
        for row in index.series:
            if row.view in ("welfare", "lab_loss::lab"):
                assert row.basis is not None
                assert row.basis["monetary_basis"] == node.population.monetary_basis
        exported_package = load_setup(
            (root / "export" / index.source["setup"].path).parent
        )
        exported_kernel = ConditionalCapture(
            exported_package,
            (root / "export" / index.source["capture"].path).parent,
        ).kernel_at(instrument.context_id, 1, "reference")
        for view in ("welfare", "lab_loss::lab"):
            actual, actual_weights = exported_kernel.samples(view)
            expected, expected_weights = kernel.samples(view)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual_weights, expected_weights)


def test_admission_floor_and_catastrophe_cap_are_shared_before_partition(
    tmp_path,
):
    nodes = _catalog()
    for key in ("agent_actions_for_deployer", "developer_control_loss"):
        node = nodes[key]
        instrument = _instrument(node)
        answer = scenario_node_answer(node)
        law = compile_contextual_node(
            instrument,
            node,
            {0: answer},
            0,
        )
        assert isinstance(law, IncidentLaw)
        components = law.segments[0].kernel.member_outcomes[0].components
        rank_group = components[0].rank_group_id
        assert rank_group is not None
        assert (
            len(
                {
                    row.severity.model_dump_json()
                    for row in components
                    if row.severity is not None
                }
            )
            == 1
        )
        for rank in (0.001, 0.5, 0.999999):
            sampled = sample_component_amounts(
                components,
                np.random.default_rng(1),
                rank_uniforms={rank_group: rank},
            )
            total = math.fsum(row.amount for row in sampled)
            assert total >= node.total.floor
            if node.total.cap is not None:
                assert total <= node.total.cap
        if key == "agent_actions_for_deployer":
            small = sample_component_amounts(
                components,
                np.random.default_rng(1),
                rank_uniforms={rank_group: 0.001},
            )
            assert all(row.amount < node.total.floor for row in small)

            # These shares sum to exactly one, although summing their rounded
            # individual lower bounds yields 99999.99999999999 at a $100k floor.
            payload = answer.model_dump(mode="json")
            shares = (0.6819821366349667, 0.29542129527597577, 0.022596568089057578)
            assert math.fsum(shares) == 1
            for index, component in enumerate(payload["branches"][0]["components"]):
                component["share"] = shares[index] if index < len(shares) else 0
            rounding_answer = ContextualRiskAnswer.model_validate(payload)
            rounding_setup = compile_contextual_setup(
                instrument,
                _artifact(instrument, rounding_answer),
                setup_id="conserving-floor",
                years_per_state=4,
            )
            rounding_study = rounding_setup
            for state in rounding_study.joint_states:
                assert compile_incident_state(rounding_study, state.state_id).incidents


def test_partition_roundoff_does_not_create_interior_grid_holes(tmp_path):
    nodes = _catalog()
    node = nodes["agent_actions_for_deployer"]
    payload = _instrument(node).model_dump(mode="json")
    payload.update(reference_eci=0, anchor_scores=[0, 3], integer_max=3)
    instrument = ContextualRiskInstrument.model_validate(payload)
    endpoints = (
        (0.048404420830601036, 0.590238181385196, 0.36135739778420295),
        (0.014234031173422257, 0.8799337427116336, 0.10583222611494414),
    )
    answers = []
    for shares in endpoints:
        data = scenario_node_answer(node).model_dump(mode="json")
        for index, component in enumerate(data["branches"][0]["components"]):
            component["share"] = shares[index] if index < 3 else 0
        answer = ContextualRiskAnswer.model_validate(data)
        answer.validate_against(node)
        answers.append(answer)
    artifact = fixture_artifact(
        instrument,
        lambda prompt: answers[0 if prompt.anchor_score == 0 else 1],
    )
    composed_setup = compile_contextual_setup(
        instrument,
        artifact,
        setup_id="roundoff",
        years_per_state=1,
    )
    study = composed_setup
    assert len(study.joint_states) == 4
    assert all(
        state.laws and (not state.absent_populations) for state in study.joint_states
    )
    bad = answers[0].model_dump(mode="json")
    for index, component in enumerate(bad["branches"][0]["components"]):
        component["share"] = (0.333333333333, 0.666666666666)[index] if index < 2 else 0
    with pytest.raises(ValueError, match="must sum to one"):
        ContextualRiskAnswer.model_validate(bad).validate_against(node)


def test_zero_rate_keeps_its_marks_and_compiles_to_a_structural_zero(
    tmp_path,
):
    nodes = _catalog()
    node = nodes["agent_actions_for_deployer"]
    instrument = _instrument(node)
    data = scenario_node_answer(node).model_dump(mode="json")
    data["rate"] = dict.fromkeys(("p5", "p25", "p50", "p75", "p95"), 0)
    zero = ContextualRiskAnswer.model_validate(data)
    # Severity is conditional on a qualifying event, so it is required at a zero count.
    zero.validate_against(node)
    with pytest.raises(
        ValidationError, match="branches is required when status is available"
    ):
        ContextualRiskAnswer.model_validate({**data, "branches": []})
    study = compile_contextual_setup(
        instrument,
        _artifact(instrument, zero),
        setup_id="zero",
        years_per_state=1,
    )
    assert all(not state.laws for state in study.joint_states)
    halfway = compile_contextual_node(
        instrument,
        node,
        {0: zero, 2: scenario_node_answer(node)},
        1,
    )
    assert isinstance(halfway, IncidentLaw)
    assert halfway.reference_annual_rate == pytest.approx(6)


def test_aggregate_lab_tail_uses_governed_shape_without_societal_floor_or_cap(
    tmp_path,
):
    nodes = _catalog()
    node = nodes["agent_actions_for_deployer"]
    governed = next(
        row
        for row in load_scenarios(scenarios_path())
        if row.id == "agent_actions_for_deployer"
    ).severity_governance
    assert governed is not None and governed.gpd_shape is not None
    assert node.lab_loss_gpd_shape == governed.gpd_shape
    instrument = _instrument(node)
    payload = scenario_node_answer(node).model_dump(mode="json")
    payload["lab_annual_business_loss"] = {
        "status": "available",
        "positive_probability": 1,
        "loss": dict(
            zip(
                ("p5", "p25", "p50", "p75", "p95", "p99", "p999"),
                (1, 2, 3, 4, 5, 10, 20),
                strict=True,
            )
        ),
    }
    first = compile_contextual_node(
        instrument,
        node,
        {0: ContextualRiskAnswer.model_validate(payload)},
        0,
    )
    assert isinstance(first, IncidentLaw)
    (aggregate,) = first.annual_lab_losses
    assert isinstance(aggregate.severity, NativeSeverityLaw)
    assert node.total is not None
    assert native_severity_sampler(aggregate.severity).quantile(0.05) < node.total.floor
    assert aggregate.severity.components[0].gpd_shape == governed.gpd_shape
    # Only the stipulated frontier-lab value caps the lab law, not the societal cap.
    assert aggregate.severity.components[0].severity_cap == FRONTIER_LAB_VALUE_USD
    payload["lab_annual_business_loss"]["loss"]["p999"] = 40
    second = compile_contextual_node(
        instrument,
        node,
        {0: ContextualRiskAnswer.model_validate(payload)},
        0,
    )
    assert isinstance(second, IncidentLaw)
    assert second.annual_lab_losses == first.annual_lab_losses


def test_catastrophe_endpoint_caps_the_lab_law_but_statutory_caps_do_not(tmp_path):
    nodes = _catalog()
    assert nodes["enforcement_matter"].lab_loss_severity_cap is None
    assert nodes["agent_actions_for_deployer"].lab_loss_severity_cap is None
    node = nodes["developer_control_loss"]
    assert node.lab_loss_severity_cap == 1e14
    instrument = _instrument(node)
    payload = scenario_node_answer(node).model_dump(mode="json")
    payload["lab_annual_business_loss"] = {
        "status": "available",
        "positive_probability": 1,
        "loss": dict(
            zip(
                ("p5", "p25", "p50", "p75", "p95", "p99", "p999"),
                (1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13),
                strict=True,
            )
        ),
    }
    # Without the frontier-lab value, as in its removal alternative, the endpoint binds.
    law = compile_contextual_node(
        instrument,
        node,
        {0: ContextualRiskAnswer.model_validate(payload)},
        0,
        options=CompileOptions(frontier_lab_value=None),
    )
    assert isinstance(law, IncidentLaw)
    # The endpoint caps each episode and, separately, the annual business loss.
    assert law.segments[0].kernel.campaign_lab_losses[0].episode_cap == 1e14
    (aggregate,) = law.annual_lab_losses
    assert isinstance(aggregate.severity, NativeSeverityLaw)
    assert aggregate.severity.components[0].severity_cap == 1e14
    sampler = native_severity_sampler(aggregate.severity)
    assert sampler.quantile(1 - 1e-9) <= 1e14
    payload["lab_annual_business_loss"] = {
        "status": "available",
        "positive_probability": 1,
        "amount": 2e14,
    }
    for value, cap in ((None, 1e14), (FRONTIER_LAB_VALUE_USD, FRONTIER_LAB_VALUE_USD)):
        fixed = compile_contextual_node(
            instrument,
            node,
            {0: ContextualRiskAnswer.model_validate(payload)},
            0,
            options=CompileOptions(frontier_lab_value=value),
        )
        assert isinstance(fixed, IncidentLaw)
        assert fixed.annual_lab_losses[0].amount == cap


def test_contextual_count_tail_scale_and_method_are_persisted(tmp_path):
    from catmodel.contracts.elicitation import Quantiles

    nodes = _catalog()
    node = nodes["agent_actions_for_deployer"]
    instrument = _instrument(node)
    answer = scenario_node_answer(node).model_copy(
        update={"rate": Quantiles(p5=0.1, p25=1, p50=5, p75=20, p95=100)}
    )
    base = compile_contextual_node(
        instrument,
        node,
        {0: answer},
        0,
    )
    widened = compile_contextual_node(
        instrument,
        node,
        {0: answer},
        0,
        options=CompileOptions(epistemic_tail_scale=1.5),
    )
    assert isinstance(base, IncidentLaw) and isinstance(widened, IncidentLaw)
    # The p75/p95 secant log(5)/(z95 - z75) exceeds the cap z95, so the tail
    # width at rank 0.99 is the cap times the tail-width factor.
    z95, z99 = ndtri([0.95, 0.99])
    for law, factor in ((base, 1), (widened, 1.5)):
        rank = law.parameter_rank
        assert rank is not None and rank.epistemic_tail_scale == factor
        (value,) = epistemic_quantiles(
            rank.count, np.array([0.99]), epistemic_tail_scale=factor
        )
        assert value == pytest.approx(100 * math.exp(factor * z95 * (z99 - z95)))
    assert widened.conditioning_context["epistemic_method"] == "pchip_local_tails"
    assert widened.conditioning_context["epistemic_tail_scale"] == 1.5
