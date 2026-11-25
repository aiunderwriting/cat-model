from copy import deepcopy

import pytest
from pydantic import ValidationError

from catmodel.contracts.incident import (
    ClusterCount,
    ExecutionPlan,
    parse_scenario_view,
)
from test_support.run_setup import setup_from_payload
from test_support.study import study_payload


@pytest.mark.parametrize(
    "field,value", [("weight", True), ("weight", "1"), ("weight", float("nan"))]
)
def test_joint_mass_is_a_finite_number(field, value) -> None:
    payload = study_payload()
    payload["joint_states"][0][field] = value
    with pytest.raises(ValidationError):
        setup_from_payload(payload)


def test_initiating_origin_has_one_count_authority() -> None:
    payload = study_payload()
    payload["joint_states"][0]["laws"] *= 2
    with pytest.raises(ValidationError, match="initiating origin IDs must be unique"):
        setup_from_payload(payload)


def test_qualified_initiation_does_not_repeat_the_gate() -> None:
    payload = study_payload()
    payload["joint_states"][0]["laws"][0]["segments"][0]["initiation_probability"] = 0.5
    with pytest.raises(ValidationError, match="repeat initiation qualification"):
        setup_from_payload(payload)


def test_finite_cluster_retains_real_support() -> None:
    cluster = ClusterCount.model_validate(
        {
            "family": "finite",
            "support": [
                {"count": 0, "probability": 0.25},
                {"count": 4, "probability": 0.75},
            ],
        }
    )
    assert cluster.expected_count() == 3
    with pytest.raises(ValidationError, match="sum to one"):
        ClusterCount.model_validate(
            {"family": "finite", "support": [{"count": 2, "probability": 0.5}]}
        )
    with pytest.raises(ValidationError, match="complete support"):
        ClusterCount.model_validate({"family": "finite"})


@pytest.mark.parametrize(
    "cluster",
    [
        {"family": "geometric", "mean": 2},
        {"family": "finite", "support": [{"count": 1, "probability": 1}], "mean": 1},
    ],
)
def test_cluster_counts_are_finite_supports_only(cluster) -> None:
    with pytest.raises(ValidationError):
        ClusterCount.model_validate(cluster)


def test_plan_freezes_all_required_comparison_arms() -> None:
    plan = ExecutionPlan(
        plan_id="plan",
        required_arms=("reference", "containment"),
        simulations=100,
        seed=4,
        requested_views=("gross",),
    )
    assert plan.required_arms == ("reference", "containment")
    with pytest.raises(ValidationError, match="reference arm must be required"):
        ExecutionPlan(
            plan_id="plan",
            required_arms=("containment",),
            simulations=100,
            seed=4,
            requested_views=("gross",),
        )


def test_scenario_view_names_its_scenario_and_measure() -> None:
    assert parse_scenario_view("scenario::loss::gross") == ("loss", "gross")
    assert parse_scenario_view("scenario::loss::welfare") == ("loss", "welfare")
    assert parse_scenario_view("scenario::loss::net_resource_cost") == (
        "loss",
        "net_resource_cost",
    )
    # The measure is the last part, so a scenario ID may itself hold "::".
    assert parse_scenario_view("scenario::a::b::gross") == ("a::b", "gross")
    for other in ("gross", "lab_loss::lab", "source_status::public::welfare"):
        assert parse_scenario_view(other) is None
    for view in ("scenario::loss", "scenario::loss::financial", "scenario::gross"):
        with pytest.raises(ValueError, match="names no measure"):
            parse_scenario_view(view)


@pytest.mark.parametrize(
    "place", ["requested view", "comparison view scope", "tail gate", "arm tail gate"]
)
def test_plan_refuses_a_scenario_view_without_its_measure(place) -> None:
    """``scenario::<id>`` alone is refused wherever a plan names a view, with
    the spelling of the scenario's gross loss."""

    def plan(*, explicit: bool) -> dict:
        view = "scenario::loss::gross" if explicit else "scenario::loss"
        gate = {"tail_gate": {"views": [view]}}
        return {
            "plan_id": "plan",
            "required_arms": ["reference", "containment"],
            "simulations": 100,
            "seed": 4,
            "requested_views": ["gross", "scenario::loss::gross"]
            + (
                ["scenario::loss"] if place == "requested view" and not explicit else []
            ),
            "comparison_requirements": [
                {
                    "family_id": "family",
                    "arm_id": "containment",
                    "scenario_ids": ["loss"],
                    "affected_views": [view],
                    "held_fixed": ["counts"],
                }
            ]
            if place == "comparison view scope"
            else [],
            "numerical_controls": gate if place == "tail gate" else {},
            "numerical_controls_by_arm": {"containment": gate}
            if place == "arm tail gate"
            else {},
        }

    ExecutionPlan.model_validate(plan(explicit=True))
    with pytest.raises(ValidationError, match="names no measure") as refusal:
        ExecutionPlan.model_validate(plan(explicit=False))
    assert "'scenario::loss::gross'" in str(refusal.value)


@pytest.mark.parametrize("mutation", ["naive", "short", "reversed"])
def test_period_is_an_exact_timezone_aware_year(mutation) -> None:
    payload = study_payload()
    if mutation == "naive":
        payload["period_start"] = "2027-01-01T00:00:00"
    elif mutation == "short":
        payload["period_end"] = "2027-12-31T00:00:00Z"
    else:
        payload["data_cutoff"] = "2026-12-31T00:00:00Z"
    with pytest.raises(ValidationError):
        setup_from_payload(payload)


def test_fixed_model_revision_cannot_change_with_path() -> None:
    payload = study_payload()
    payload["joint_states"][0]["entity_paths"][0]["knots"][-1]["model_revision"] = "r2"
    with pytest.raises(ValidationError, match="fixed-model paths"):
        setup_from_payload(payload)


def test_alternative_needs_its_own_entity_path() -> None:
    payload = study_payload()
    second = deepcopy(payload["entities"][0])
    second["entity_id"] = "other"
    payload["entities"].append(second)
    law = deepcopy(payload["joint_states"][0]["laws"][0])
    law["entity_id"] = "other"
    payload["alternatives"] = [
        {
            "alternative_id": "other_model",
            "purpose": "scientific_alternative",
            "description": "A different entity requires its own path.",
            "overrides": [{"state_id": "state", "law": law}],
        }
    ]
    with pytest.raises(ValidationError, match="own declared entity path"):
        setup_from_payload(payload)


def test_optional_count_experiments_require_truthful_matched_controls() -> None:
    payload = study_payload()
    gaussian: dict = {
        "kind": "fixed_marginal_gaussian",
        "origin_groups": {"origin": "portfolio"},
        "correlation": 0,
    }
    payload["alternatives"] = [
        {
            "alternative_id": "copula_control",
            "purpose": "scientific_alternative",
            "description": "Fixed count marginals, independent ranks.",
            "count_experiment": gaussian,
            "matched_control_id": "copula_control",
        },
        {
            "alternative_id": "copula_stress",
            "purpose": "scientific_alternative",
            "description": "Fixed count marginals, dependent ranks.",
            "count_experiment": {**gaussian, "correlation": 0.7},
            "matched_control_id": "copula_control",
        },
    ]
    experiment = setup_from_payload(payload).alternatives[1].count_experiment
    assert experiment is not None
    assert experiment.kind == "fixed_marginal_gaussian"
    payload["alternatives"][0]["count_experiment"]["correlation"] = 0.1
    with pytest.raises(ValidationError, match="zero correlation"):
        setup_from_payload(payload)
    payload["alternatives"][0]["count_experiment"]["correlation"] = 0
    payload["alternatives"][1]["matched_control_id"] = "undeclared"
    with pytest.raises(ValidationError, match="declared alternative"):
        setup_from_payload(payload)
    del payload["alternatives"][1]["matched_control_id"]
    with pytest.raises(ValidationError, match="must name its matched control"):
        setup_from_payload(payload)


def test_deployment_cannot_silently_bind_to_lab_frontier() -> None:
    payload = study_payload()
    payload["entities"].extend(
        [
            {
                "entity_id": "lab",
                "kind": "lab",
                "definition": "Lab frontier population",
                "display_name": "Lab",
                "status": "actual",
            },
            {
                "entity_id": "use",
                "kind": "deployment",
                "definition": "Selected use",
                "display_name": "Use",
                "status": "actual",
            },
        ]
    )
    payload["deployments"] = [
        {
            "deployment_id": "use",
            "model_id": "lab",
            "use_population": "Selected customers",
            "tools_and_permissions": "No tools",
            "human_oversight": "Reviewed",
        }
    ]
    with pytest.raises(ValidationError, match="not a lab frontier"):
        setup_from_payload(payload)


def test_fixed_vmr_is_not_silently_used_as_literal_thinning() -> None:
    payload = study_payload()
    law = payload["joint_states"][0]["laws"][0]
    law.update(
        count_family="negative_binomial",
        count_variance_to_mean=2,
        input_stage="attempted_initiator",
    )
    law["segments"][0]["initiation_probability"] = 0.5
    with pytest.raises(ValidationError, match="final qualified annual count"):
        setup_from_payload(payload)


def test_native_severity_and_stipulated_frequency_share_the_neutral_law() -> None:
    from scipy.stats import lognorm

    payload = study_payload()
    component = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
        "member_outcomes"
    ][0]["components"][0]
    component.pop("amount")
    knots = dict(
        zip(
            ("p5", "p25", "p50", "p75", "p95"),
            map(float, lognorm.ppf([0.05, 0.25, 0.5, 0.75, 0.95], s=0.5, scale=10)),
        )
    )
    from catmodel.contracts.native_severity import native_severity_from_sampler
    from catmodel.probability.distributions import Distribution

    component.update(
        severity=native_severity_from_sampler(
            Distribution(
                knots["p5"],
                knots["p25"],
                knots["p50"],
                knots["p75"],
                knots["p95"],
            )
        ).model_dump(mode="json"),
        rank_group_id="victim_amount",
    )
    study = setup_from_payload(payload)
    assert (
        study.joint_states[0]
        .laws[0]
        .segments[0]
        .kernel.member_outcomes[0]
        .components[0]
        .severity
        is not None
    )


def test_frozen_staged_rules_only_admit_executable_rules() -> None:
    plan = ExecutionPlan(
        plan_id="plan",
        required_arms=("reference",),
        simulations=100,
        seed=4,
        requested_views=("gross",),
    )
    assert plan.staged_rules.work_limit_exceedance == "abort_without_capture"
    with pytest.raises(ValidationError, match="Extra inputs"):
        ExecutionPlan.model_validate(
            {
                **plan.model_dump(mode="json"),
                "staged_rules": {"refine_until_good": "ignored"},
            }
        )


@pytest.mark.parametrize(
    "scope", ["member", "campaign", "joint_member", "joint_campaign"]
)
def test_partial_financial_mapping_keeps_physical_law_and_complete_claim_is_strict(
    scope,
):
    payload = study_payload()
    payload["contracts"] = [
        {"contract_id": "policy", "contract_year": "2027", "entity_id": "model"}
    ]
    kernel = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"]
    member = kernel["member_outcomes"][0]
    if scope.startswith("joint_"):
        kernel.update(
            cluster=None,
            member_outcomes=[],
            explicit_joint_members=[{"probability": 1, "members": [member]}],
        )
    campaign = scope.endswith("campaign")
    target = (
        kernel["explicit_joint_members"][0]
        if scope == "joint_campaign"
        else kernel
        if campaign
        else member
    )
    prefix = "campaign_" if campaign else ""
    target.update(
        {
            prefix + "components": deepcopy(member["components"]),
            prefix + "obligations": [
                {"component_id": "victim_harm", "entity_id": "model", "fraction": 1}
            ],
            prefix + "memberships": [
                {
                    "component_id": "victim_harm",
                    "contract_id": "policy",
                    "obligated_entity_id": "model",
                    "eligible": True,
                }
            ],
            prefix + "financial_mapping_complete": False,
        }
    )
    setup_from_payload(payload)
    target[prefix + "financial_mapping_complete"] = True
    with pytest.raises(
        ValidationError, match="separate explicit obligation and payment"
    ):
        setup_from_payload(payload)
