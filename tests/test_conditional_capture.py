"""Conditional kernels retain native state weights and paired loss rows."""

from itertools import groupby

import numpy as np
import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.contracts.conditional import validate_conditional_contexts
from catmodel.contracts.incident import ExecutionPlan
from catmodel.contracts.law_overrides import HeldOverrides
from catmodel.results.conditional import ConditionalCapture
from catmodel.results.execution import replay_capture, run_setup
from catmodel.results.results_export import export_results, load_results_export
from catmodel.setup.store import build_setup, load_setup
from test_support.conditional import conditional_study, materialize_constant_eci_grid
from test_support.run_setup import setup_from_payload, with_execution


def _capture(
    tmp_path, *, lab_loss=True, two_templates=False, independent_annual_kernel=False
):
    study = conditional_study(lab_loss=lab_loss, two_templates=two_templates)
    if independent_annual_kernel:
        payload = encode_scientific(study)
        payload["entities"].append(
            {**payload["entities"][0], "entity_id": "other-model"}
        )
        for state in payload["joint_states"]:
            path = state["entity_paths"][0]
            state["entity_paths"].append(
                {
                    **path,
                    "entity_id": "other-model",
                    "knots": [{**knot, "value": 0} for knot in path["knots"]],
                }
            )
            law = state["laws"][0]
            state["laws"].append(
                {
                    **law,
                    "entity_id": "other-model",
                    "origin_id": "independent-annual-origin",
                    "reference_annual_rate": 3,
                    "response": {
                        **law["response"],
                        "domain": "conditional_annual_kernel",
                        "points": [{"value": 0, "multiplier": 1}],
                        "extrapolation": "flat",
                        "held_fixed": "Independent other-model annual kernel held fixed across the queried model scores.",
                    },
                }
            )
        study = setup_from_payload(payload)
    plan = ExecutionPlan(
        plan_id="conditional",
        required_arms=("reference",),
        simulations=48,
        seed=31,
        requested_views=("gross", "welfare", "lab_loss::lab"),
    )
    package = build_setup(
        destination=tmp_path / "package",
        setup=with_execution(study, plan, setup_id="conditional"),
        code_commit="fixture",
    )
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")
    return ConditionalCapture(load_setup(package.root), tmp_path / "run" / "capture")


def test_zero_prior_kernel_is_executed_and_independent_of_forecast(tmp_path):
    bound = _capture(tmp_path)
    kernel = bound.kernel_at("internal", 1, "reference")
    assert kernel.metrics("welfare")["expected_annual_loss"] == pytest.approx(40)
    assert kernel.metrics("lab_loss::lab")["expected_annual_loss"] == pytest.approx(16)
    society, weights = kernel.samples("welfare")
    lab, same_weights = kernel.samples("lab_loss::lab")
    np.testing.assert_allclose(lab, society * 0.4)
    np.testing.assert_array_equal(weights, same_weights)
    assert bound.kernel_at("internal", 0, "reference").metrics("welfare")[
        "expected_annual_loss"
    ] == pytest.approx(10)


def test_conditional_nuisance_weights_survive_grid_expansion(tmp_path):
    bound = _capture(tmp_path, two_templates=True)
    kernel = bound.kernel_at("internal", 1, "reference")
    assert sorted(kernel.stratum_weights.values()) == [0.25, 0.75]
    assert kernel.metrics("welfare")["expected_annual_loss"] == pytest.approx(70)


def test_missing_peer_is_unavailable_not_zero(tmp_path):
    bound = _capture(tmp_path, lab_loss=False)
    with pytest.raises(ValueError, match="societal or lab loss is missing"):
        bound.kernel_at("internal", 1, "reference")


@pytest.mark.parametrize("independent_annual_kernel", [False, True])
def test_conditional_source_reuses_capture_for_replay(
    tmp_path, independent_annual_kernel
):
    bound = _capture(tmp_path, independent_annual_kernel=independent_annual_kernel)
    before = {
        p: p.read_bytes()
        for p in (tmp_path / "run" / "capture").rglob("*")
        if p.is_file()
    }
    replay_capture(
        setup_path=bound.setup.root,
        capture_directory=tmp_path / "run" / "capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    assert bound.kernel_at("internal", 1, "reference").metrics("welfare")[
        "expected_annual_loss"
    ] == pytest.approx(70 if independent_annual_kernel else 40)
    index = export_results(
        setup_path=bound.setup.root,
        capture_directory=tmp_path / "run" / "capture",
        results_path=tmp_path / "replay.json",
        output=tmp_path / "export",
    )
    assert load_results_export(tmp_path / "export") == index
    assert index.conditioning["status"] == "available"
    assert all(path.read_bytes() == data for path, data in before.items())


def test_omitted_zero_prior_state_is_rejected_by_execution_plan(tmp_path):
    study = conditional_study()
    plan = ExecutionPlan(
        plan_id="hole",
        required_arms=("reference",),
        simulations=8,
        seed=1,
        requested_views=("gross", "welfare", "lab_loss::lab"),
        required_state_ids_by_arm={
            "reference": tuple(s.state_id for s in study.joint_states if s.weight > 0)
        },
    )
    with pytest.raises(ValueError, match="omits a conditional kernel component"):
        build_setup(
            destination=tmp_path / "hole",
            setup=with_execution(study, plan, setup_id="hole"),
            code_commit="fixture",
        )


def test_conditional_plan_requires_both_peer_observables(tmp_path):
    study = conditional_study()
    plan = ExecutionPlan(
        plan_id="missing-peer",
        required_arms=("reference",),
        simulations=8,
        seed=1,
        requested_views=("welfare",),
    )
    with pytest.raises(ValueError, match="conditional peer observable"):
        build_setup(
            destination=tmp_path / "missing-peer",
            setup=with_execution(study, plan, setup_id="missing-peer"),
            code_commit="fixture",
        )


def test_true_zero_kernel_is_available(tmp_path):
    study = conditional_study()
    payload = encode_scientific(study)
    payload["joint_states"][1]["laws"][0]["reference_annual_rate"] = 0
    study = setup_from_payload(payload)
    plan = ExecutionPlan(
        plan_id="zero",
        required_arms=("reference",),
        simulations=8,
        seed=2,
        requested_views=("gross", "welfare", "lab_loss::lab"),
    )
    package = build_setup(
        destination=tmp_path / "zero",
        setup=with_execution(study, plan, setup_id="zero"),
        code_commit="fixture",
    )
    run_setup(
        setup_path=package.root, output=tmp_path / "zero-run", code_commit="fixture"
    )
    kernel = ConditionalCapture(package, tmp_path / "zero-run" / "capture").kernel_at(
        "internal", 1, "reference"
    )
    assert kernel.metrics("welfare")["expected_annual_loss"] == 0
    np.testing.assert_array_equal(kernel.samples("lab_loss::lab")[0], 0)


def test_conditional_constructor_rejects_modified_numeric_capture(tmp_path):
    bound = _capture(tmp_path)
    path = next((tmp_path / "run" / "capture" / "arrays").glob("*.npz"))
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        ConditionalCapture(bound.setup, tmp_path / "run" / "capture")


def _alternative_grid(stresses=("stress",)):
    """A constant-path grid over an authored setup whose ``stresses`` arms
    each reweight the two states and override one law."""
    base = encode_scientific(conditional_study(two_templates=True))
    # Reuse an unexpanded template to exercise the same helper with an authored arm.
    source = base["joint_states"][:2]
    for row, weight in zip(source, (0.25, 0.75)):
        row["state_id"] = row["state_id"].split(":")[-1]
        row["weight"] = weight
    base["joint_states"] = source
    base["conditional_contexts"] = []
    override = source[0]["laws"][0]
    override = {**override, "reference_annual_rate": 3}
    base["alternatives"] = [
        {
            "alternative_id": stress,
            "purpose": "scientific_alternative",
            "description": "Stipulated different conditional mixture",
            "state_weights": {"state": 0.8, "second": 0.2},
            "overrides": [{"state_id": "state", "law": override}],
        }
        for stress in stresses
    ]
    arms = ("reference", *stresses)
    return materialize_constant_eci_grid(
        with_execution(
            setup_from_payload(base),
            ExecutionPlan(
                plan_id="conditional-grid",
                required_arms=arms,
                simulations=7,
                seed=31,
                requested_views=("welfare", "lab_loss::lab"),
                numerical_controls={"max_campaign_members_per_year": 300},
                required_state_ids_by_arm={arm: ("state", "second") for arm in arms},
            ),
            setup_id="conditional-grid",
        ),
        context_id="stress-grid",
        lab_id="lab",
        deployment_id="deployment",
        entity_id="model",
        scale_id="own_score",
        integer_min=0,
        integer_max=2,
        forecast_weights={0: 0.5, 1: 0, 2: 0.5},
    )


def test_conditional_contexts_read_each_arm_s_overrides_in_one_pass(monkeypatch):
    """A context's kernels list every arm at each score; a setup that stores
    its overrides holds few arms at a time, so the check reads arm by arm."""
    stresses = ("stress", "second-stress", "third-stress")
    expanded = _alternative_grid(stresses)
    read = []
    laws = HeldOverrides.laws

    def recorded(self, alternative_id, state_id):
        read.append(alternative_id)
        return laws(self, alternative_id, state_id)

    monkeypatch.setattr(HeldOverrides, "laws", recorded)
    validate_conditional_contexts(expanded)
    assert [arm for arm, _ in groupby(read)] == list(stresses)
    assert len(read) == 3 * len(expanded.joint_states)


def test_alternative_grid_keeps_explicit_state_support_weights_and_laws(tmp_path):
    expanded = _alternative_grid()
    middle = expanded.conditional_contexts[0].kernel_at(1, "stress")
    assert [row.weight for row in middle.components] == [0.8, 0.2]
    arm = expanded.alternatives[0]
    overrides = expanded.overrides.of(arm.alternative_id)
    assert len(overrides) == 3
    assert all(override.law.reference_annual_rate == 3 for override in overrides)
    weights = arm.state_weights
    assert weights is not None
    assert all(weights[row.state_id] == 0 for row in middle.components)

    assert expanded.execution.required_arms == ("reference", "stress")
    assert expanded.execution.requested_views == ("welfare", "lab_loss::lab")
    assert expanded.execution.simulations == 7
    assert expanded.execution.seed == 31
    assert expanded.execution.numerical_controls == {
        "max_campaign_members_per_year": 300
    }
    support = {row.state_id for row in expanded.joint_states}
    assert all(
        set(expanded.execution.required_state_ids_by_arm[arm]) == support
        for arm in expanded.execution.required_arms
    )
    saved = build_setup(
        destination=tmp_path / "expanded", setup=expanded, code_commit="fixture"
    )
    assert saved.execution == expanded.execution
