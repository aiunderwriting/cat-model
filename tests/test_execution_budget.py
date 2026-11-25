import pytest

from catmodel.contracts.incident import ExecutionPlan
from catmodel.simulation.execution_budget import (
    execution_budget,
    work_allowances,
    work_limits,
)
from catmodel.simulation.tail_gate import TailGateControls
from test_support.study import _plan


def test_budget_uses_each_arms_frozen_support_and_simulation_count():
    plan = _plan(20).model_copy(
        update={
            "required_arms": ("reference", "contained", "reweighted"),
            "required_state_ids_by_arm": {"contained": ("a",)},
            "simulations_by_arm": {"contained": 30},
        }
    )
    budget = execution_budget(
        plan,
        {
            "reference": {"a": "law-a", "b": "law-b"},
            "contained": {"a": "law-a"},
            "reweighted": {"a": "law-a", "b": "changed-b"},
        },
    )
    assert budget["arms"] == 3
    assert budget["by_arm"]["reference"]["simulated_conditional_years"] == 40
    # A different simulation count is a different simulation of the same law.
    assert budget["by_arm"]["contained"]["simulated_conditional_years"] == 30
    # Only the state whose law changed is simulated again.
    reweighted = budget["by_arm"]["reweighted"]
    assert reweighted["conditional_years"] == 40
    assert reweighted["simulated_conditional_years"] == 20
    assert budget["simulated_conditional_years"] == 90
    assert budget["reused_conditional_years"] == 20
    assert budget["reference_work_multiple"] == 2.25
    assert "new_llm_calls" not in budget


def test_each_work_limit_is_its_allowance_per_year_times_the_years():
    allowances = work_allowances({"max_campaign_members_per_year": 7})
    # Every allowance the plan omits takes its own default.
    assert allowances == {
        "max_vectorized_events_per_year": 50_000,
        "max_campaign_members_per_year": 7,
    }
    assert work_limits(allowances, 3) == {
        "vectorized_initiators": 150_000,
        "campaign_members": 21,
    }
    for value in (0, 1.5, True, None):
        with pytest.raises(ValueError, match="must be a positive integer"):
            work_allowances({"max_campaign_members_per_year": value})


@pytest.mark.parametrize("arm", [False, True], ids=["plan", "arm"])
@pytest.mark.parametrize("control", ["max_events", "max_event_per_year"])
def test_a_plan_refuses_an_unsupported_numerical_control(control, arm):
    """An unknown or misspelled control fails where the plan is built or loaded,
    plan-wide or for an arm."""
    plan = _plan(20)
    controls = {control: 100}
    with pytest.raises(ValueError, match=f"unsupported numerical controls .*{control}"):
        ExecutionPlan.model_validate(
            {
                **plan.model_dump(mode="json"),
                "numerical_controls": {} if arm else controls,
                "numerical_controls_by_arm": {"reference": controls} if arm else {},
            }
        )


def _with_controls(controls, arm, **fields):
    """The fixture plan's JSON with ``controls`` plan-wide or for one arm."""
    return {
        **_plan(20).model_dump(mode="json"),
        "numerical_controls": {} if arm else controls,
        "numerical_controls_by_arm": {"contained": controls} if arm else {},
        **fields,
    }


@pytest.mark.parametrize("arm", [False, True], ids=["plan", "arm"])
@pytest.mark.parametrize(
    "control",
    [
        "max_vectorized_events_per_year",
        "max_campaign_members_per_year",
        "max_members_per_campaign",
        "event_batch_size",
    ],
)
def test_a_plan_refuses_a_count_control_that_is_present_without_a_count(control, arm):
    """Null is a value, not an omission: only an absent control takes its
    default."""
    for value in (None, 0, 2.0, "4096", True):
        with pytest.raises(ValueError, match=f"{control} must be a positive integer"):
            ExecutionPlan.model_validate(_with_controls({control: value}, arm))
    plan = ExecutionPlan.model_validate(_with_controls({control: 4096}, arm))
    resolved = plan.for_arm("contained" if arm else "reference")
    assert resolved.numerical_controls[control] == 4096


@pytest.mark.parametrize("arm", [False, True], ids=["plan", "arm"])
@pytest.mark.parametrize(
    ("gate", "refusal"),
    [
        ({"views": "gross"}, "tail gate views must be a nonempty list"),
        ({"views": []}, "tail gate views must be a nonempty list"),
        ({"views": ["gros"]}, "'gros'.* not requested views"),
        ({"views": ["gross", "financial"]}, "'financial'.* not requested views"),
        ({"levels": 0.99}, "tail gate levels"),
        ({"levels": ["0.99"]}, "tail gate levels"),
        ({"tolerance": None}, "tail gate tolerance"),
        ({"batches": None}, "tail gate batches"),
        ({"max_reference_state_years": "4096"}, "max_reference_state_years"),
        ({"max_years": 4096}, "tail_gate controls take only"),
        ("gross", "tail_gate controls take only"),
    ],
)
def test_a_plan_refuses_a_tail_gate_that_is_mistyped_or_drives_no_view(
    gate, refusal, arm
):
    """A gate whose authored driver views select nothing would report its arm
    converged without checking a statistic."""
    with pytest.raises(ValueError, match=refusal):
        ExecutionPlan.model_validate(_with_controls({"tail_gate": gate}, arm))


@pytest.mark.parametrize("arm", [False, True], ids=["plan", "arm"])
def test_a_plan_keeps_a_well_formed_tail_gate_and_the_default_drivers(arm):
    gate = {
        "tolerance": 0.05,
        "batches": 8,
        "max_reference_state_years": None,
        "max_comparison_state_years": 4096,
        "views": ["scenario::loss::gross", "gross"],
        "levels": [0.9, 0.99],
    }
    plan = ExecutionPlan.model_validate(_with_controls({"tail_gate": gate}, arm))
    resolved = plan.for_arm("contained" if arm else "reference")
    controls = TailGateControls.from_controls(resolved.numerical_controls["tail_gate"])
    assert controls.record_controls() == gate
    assert controls.drivers(plan.requested_views) == ("scenario::loss::gross", "gross")
    # A null gate is the default gate, and a plan requesting no headline view
    # has no driver view without authoring one.
    quiet = ExecutionPlan.model_validate(
        _with_controls({"tail_gate": None}, arm, requested_views=["net_resource_cost"])
    )
    assert TailGateControls().drivers(quiet.requested_views) == ()


def test_each_arm_reports_its_states_work_limits_at_its_largest_years():
    plan = _plan(20).model_copy(
        update={
            "required_arms": ("reference", "contained"),
            "required_state_ids_by_arm": {"contained": ("a",)},
            "numerical_controls": {
                "max_campaign_members_per_year": 3,
                "tail_gate": {
                    "max_reference_state_years": 80,
                    "max_comparison_state_years": 80,
                },
            },
            "numerical_controls_by_arm": {
                "contained": {"max_campaign_members_per_year": 5}
            },
        }
    )
    budget = execution_budget(
        plan,
        {"reference": {"a": "law-a", "b": "law-b"}, "contained": {"a": "law-a"}},
    )
    # Two states double once within 80 state-years, one state twice.
    assert [
        budget["by_arm"][arm]["max_work_per_state"]
        for arm in ("reference", "contained")
    ] == [
        {"vectorized_initiators": 2_000_000, "campaign_members": 120},
        {"vectorized_initiators": 4_000_000, "campaign_members": 400},
    ]
