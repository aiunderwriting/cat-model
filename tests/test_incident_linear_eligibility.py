"""Only single-member laws and proven linear projections use array marks."""

from copy import deepcopy

import numpy as np
import pytest

from catmodel.contracts.incident import ExecutionPlan, MarkKernel
from catmodel.simulation import incident_runtime as runtime
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_linear import (
    linear_projection,
    single_member_law,
)
from test_support.run_setup import setup_from_payload
from test_support.study import study_payload

VIEWS = ("welfare", "lab_loss::lab")


def _projection(payload, views, **kwargs):
    law = single_member_law(MarkKernel.model_validate(payload))
    return None if law is None else linear_projection(law, views, **kwargs)


def _payload(*, with_lab=True):
    # The harm component allocates nothing to any lab.
    return {
        "cluster": {
            "family": "finite",
            "support": [{"count": 1, "probability": 1}],
        },
        "shared_probability": 1,
        "unit_probability": 1,
        "campaign_lab_losses": [{"lab_id": "lab", "amount": 3}] if with_lab else [],
        "member_outcomes": [
            {
                "probability": 1,
                "components": [
                    {
                        "component_id": "harm",
                        "nature": "resource_cost",
                        "amount": 10,
                        "lab_loss_weights": {},
                    }
                ],
            }
        ],
        "gate_meaning": "All units pass the gate.",
        "selection_assumption": "The selected unit is fixed.",
        "displacement_assumption": "No displacement.",
    }


@pytest.mark.parametrize(
    "views",
    [
        ("welfare",),
        ("lab_loss::lab",),
        ("welfare", "lab_loss::lab"),
        (
            "source_status::public::welfare",
            "source_status::not_yet_public::lab_loss::lab",
        ),
    ],
)
def test_identified_linear_views_are_admitted(views):
    for status in ("public", "not_yet_public"):
        assert (
            _projection(
                _payload(with_lab=views != ("welfare",)),
                views,
                source_public_status=status,
            )
            is not None
        )


@pytest.mark.parametrize(
    "view",
    [
        "financial",
        "oep_financial",
        "risk_unit::origin::occurrence_net_financial",
        "source_status::unknown::welfare",
        "lab_loss::missing",
        "source_status::public::lab_loss::missing",
    ],
)
def test_nonlinear_views_and_unknown_lab_allocations_decline(view):
    payload = _payload()
    payload["member_outcomes"][0]["components"][0].pop("lab_loss_weights")
    assert _projection(payload, (view,), source_public_status="public") is None


def test_other_status_views_never_read_this_segments_allocations():
    payload = _payload(with_lab=False)
    payload["member_outcomes"][0]["components"][0].pop("lab_loss_weights")
    view = "source_status::not_yet_public::lab_loss::lab"
    assert _projection(payload, (view,), source_public_status="public") is not None
    assert _projection(payload, (view,), source_public_status="not_yet_public") is None


@pytest.mark.parametrize(
    "update,probability",
    [
        ({"shared_probability": 0.75}, 0.75),
        ({"unit_probability": 0.5, "shared_probability": 0.5}, 0.25),
        (
            {
                "cluster": {
                    "family": "finite",
                    "support": [
                        {"count": 0, "probability": 0.4},
                        {"count": 1, "probability": 0.6},
                    ],
                }
            },
            0.6,
        ),
    ],
)
def test_bernoulli_member_gates_compose_into_one_member_probability(
    update, probability
):
    payload = _payload()
    payload.update(update)
    law = single_member_law(MarkKernel.model_validate(payload))
    assert law is not None
    assert law.member_probability == pytest.approx(probability, rel=1e-15)
    assert _projection(payload, VIEWS) is not None


def test_one_explicit_single_member_branch_is_a_single_member_law():
    payload = _payload()
    outcome = payload.pop("member_outcomes")[0]
    payload.pop("cluster")
    payload["unit_probability"] = 0.5
    payload["explicit_joint_members"] = [
        {"probability": 1, "members": [outcome]},
    ]
    law = single_member_law(MarkKernel.model_validate(payload))
    assert law is not None and law.member_probability == 0.5
    payload["explicit_joint_members"] = [
        {"probability": 0.5, "members": [outcome]},
        {"probability": 0.5, "members": [outcome, outcome]},
    ]
    assert single_member_law(MarkKernel.model_validate(payload)) is None


@pytest.mark.parametrize(
    "cluster",
    [
        {"family": "finite", "support": [{"count": 2, "probability": 1}]},
        {
            "family": "finite",
            "support": [
                {"count": 1, "probability": 0.5},
                {"count": 2, "probability": 0.5},
            ],
        },
    ],
)
def test_laws_with_several_possible_members_decline(cluster):
    payload = _payload()
    payload["cluster"] = cluster
    assert single_member_law(MarkKernel.model_validate(payload)) is None


def test_campaign_components_and_alternative_outcomes_decline():
    payload = _payload()
    payload["campaign_components"] = [
        {"component_id": "campaign", "nature": "resource_cost", "amount": 0}
    ]
    assert single_member_law(MarkKernel.model_validate(payload)) is None
    payload = _payload()
    outcome = payload["member_outcomes"][0]
    outcome["probability"] = 0.5
    payload["member_outcomes"].append(dict(outcome))
    assert single_member_law(MarkKernel.model_validate(payload)) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("identified_gross_total", 10),
        ("overlap_resolved", False),
        ("composition_complete", False),
    ],
)
def test_gross_consistency_checks_and_unresolved_overlap_stay_on_physical_path(
    field, value
):
    payload = _payload()
    payload["member_outcomes"][0][field] = value
    if field == "composition_complete":
        payload["member_outcomes"][0]["identified_gross_total"] = 10
    assert single_member_law(MarkKernel.model_validate(payload)) is None


@pytest.mark.parametrize("amount", [0, 10])
def test_unknown_identified_nature_is_not_zero_welfare_weight(amount):
    payload = _payload()
    payload["member_outcomes"][0]["components"][0].update(nature=None, amount=amount)
    assert _projection(payload, VIEWS) is None


def test_extra_independent_lab_totals_do_not_prevent_linear_evaluation():
    assert _projection(_payload(), ("welfare",)) is not None
    payload = _payload()
    payload["campaign_lab_losses"].append({"lab_id": "other", "amount": 0})
    assert _projection(payload, VIEWS) is not None


def test_unrequested_component_allocation_keys_do_not_change_identified_views():
    payload = _payload()
    component = payload["member_outcomes"][0]["components"][0]
    component["lab_loss_weights"] = {"lab": 0.5, "other": 0.5}
    assert _projection(payload, VIEWS) is not None
    component["lab_loss_weights"] = {"lab": 0.5}
    assert _projection(payload, VIEWS) is not None


def _plan(views):
    return ExecutionPlan(
        plan_id="hybrid-evaluation",
        required_arms=("reference",),
        simulations=1,
        seed=42,
        requested_views=views,
    )


def _hybrid_state(*, physical_grouping=False):
    payload = study_payload()
    payload["entities"].append(
        {
            "entity_id": "lab",
            "kind": "lab",
            "definition": "Synthetic lab",
            "display_name": "Lab",
            "status": "hypothetical",
        }
    )
    state = payload["joint_states"][0]
    first = state["laws"][0]
    kernel = _payload()
    component = kernel["member_outcomes"][0]["components"][0]
    component.update(
        nature="transfer",
        amount=0 if physical_grouping else 1e15,
        bearer_id="victim",
        contributor_ids=["model"],
        attribution_weights={"model": 1},
    )
    first["segments"][0].update(
        kernel=kernel,
        source_public_status="not_yet_public" if physical_grouping else "public",
        source_model_id="model",
        source_model_revision="r1",
    )
    second = deepcopy(first)
    second["origin_id"] = "rich-origin"
    rich = second["segments"][0]["kernel"]
    rich["campaign_components"] = [
        {
            "component_id": "campaign",
            "nature": "transfer",
            "amount": 0,
            "bearer_id": "victim",
            "contributor_ids": ["model"],
            "attribution_weights": {"model": 1},
            "lab_loss_weights": {},
        }
    ]
    rich["member_outcomes"][0]["components"][0]["amount"] = (
        1e15 if physical_grouping else 0
    )
    second["segments"][0]["source_public_status"] = (
        "public" if physical_grouping else "not_yet_public"
    )
    state["laws"].append(second)
    if physical_grouping:
        # Campaign components require physical grouping. A third origin
        # exercises their combination with an independently vectorized native
        # mark. The lab view needs every episode's composition identified.
        vector = deepcopy(first)
        vector["origin_id"] = "vector-origin"
        vector_kernel = vector["segments"][0]["kernel"]
        vector_kernel["campaign_lab_losses"][0]["amount"] = 0
        state["laws"].append(vector)
    return compile_incident_state(setup_from_payload(payload), "state")


@pytest.mark.parametrize("physical_grouping", [False, True])
@pytest.mark.parametrize("scope", [False, True])
def test_hybrid_hidden_gross_uses_full_cohort_and_source_status_availability(
    monkeypatch, physical_grouping, scope
):
    compiled = _hybrid_state(physical_grouping=physical_grouping)
    counts = {
        "origin": np.array([1 if physical_grouping else 2]),
        "rich-origin": np.array([2 if physical_grouping else 1]),
    }
    if physical_grouping:
        counts["vector-origin"] = np.array([1])
    monkeypatch.setattr(runtime, "_origin_counts", lambda *_: counts)
    views = (
        ("lab_loss::lab", "source_status::public::lab_loss::lab")
        if scope
        else ("lab_loss::lab",)
    )
    plan = _plan(views)
    expected = runtime.simulate_incident_state(plan, compiled, response_sha256="a" * 64)
    assert expected[0].views["lab_loss::lab"].array.tolist() == [9]
    assert expected.audit.vectorized_initiators == expected.audit.total_initiators
    with monkeypatch.context() as physical:
        physical.setattr(runtime, "single_member_law", lambda _kernel: None)
        reference = runtime.simulate_incident_state(
            plan, compiled, response_sha256="a" * 64
        )
    np.testing.assert_array_equal(
        expected[0].views["lab_loss::lab"].array,
        reference[0].views["lab_loss::lab"].array,
    )
