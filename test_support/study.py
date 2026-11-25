"""Stipulated incident-study inputs for compiler tests."""

from typing import Any

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.contracts.incident import ExecutionPlan
from catmodel.contracts.native_severity import native_severity_from_sampler
from catmodel.contracts.run_setup import RunSetup
from catmodel.probability.distributions import Distribution
from catmodel.setup.store import build_setup
from test_support.run_setup import setup_from_payload, with_execution


def study_payload() -> dict:
    component = {
        "component_id": "victim_harm",
        "amount": 10,
        "bearer_id": "victim",
        "nature": "resource_cost",
        "contributor_ids": ["model"],
        "scenario_ids": ["scenario"],
    }
    kernel = {
        "cluster": {"family": "finite", "support": [{"count": 1, "probability": 1}]},
        "shared_probability": 1,
        "unit_probability": 1,
        "member_outcomes": [{"probability": 1, "components": [component]}],
        "gate_meaning": "No extra qualification in this fully scoped fixture.",
        "selection_assumption": "Complete fixed selected population.",
        "displacement_assumption": "No displacement in this declared fixture.",
    }
    law = {
        "origin_id": "origin",
        "scenario_id": "scenario",
        "entity_id": "model",
        "reference_annual_rate": 1,
        "count_family": "poisson",
        "response": {
            "interpolation": "log",
            "extrapolation": "last_slope",
            "points": [{"value": 0, "multiplier": 1}, {"value": 1, "multiplier": 4}],
            "held_fixed": "Exposure, qualification and complete mark law held fixed.",
        },
        "segments": [
            {
                "start": 0,
                "end": 1,
                "reference_exposure_share": 1,
                "initiation_probability": 1,
                "kernel": kernel,
            }
        ],
        "input_stage": "qualified_initiator",
        "qualification_assumption": "Already qualifying origins; no second narrative gate.",
    }
    return {
        "scope_id": "stipulated_fixture",
        "title": "Openly stipulated exact-law fixture",
        "period_start": "2027-01-01T00:00:00Z",
        "period_end": "2028-01-01T00:00:00Z",
        "forecast_as_of": "2027-01-01T00:00:00Z",
        "data_cutoff": "2027-01-01T00:00:00Z",
        "availability_cutoff": "2027-01-01T00:00:00Z",
        "information_date": "2027-01-01T00:00:00Z",
        "capability_scales": [
            {
                "scale_id": "own_score",
                "quantity_definition": "Own entity score",
                "unit": "stipulated index points",
                "reference_definition": "Zero is the rate reference",
                "measurement_protocol": "Openly stipulated fixture, not measurement.",
            }
        ],
        "entities": [
            {
                "entity_id": "model",
                "kind": "model",
                "revision": "r1",
                "definition": "One fixed model revision",
                "display_name": "Fixture model",
                "status": "hypothetical",
            }
        ],
        "populations": [
            {
                "scenario_id": "scenario",
                "definition": "Unique test victims",
                "causal_qualification": "Complete origin-owned qualifying harm",
                "initiating_unit": "campaign",
                "terminal_unit": "victim",
                "severity_unit": "victim ultimate loss",
                "monetary_basis": "USD at study start before recovery",
                "currency": "USD",
                "accounting_standing": "Victim harm without offsetting perpetrator benefit",
            }
        ],
        "joint_states": [
            {
                "state_id": "state",
                "epistemic_group_id": "fixed",
                "weight": 1,
                "entity_paths": [
                    {
                        "entity_id": "model",
                        "scale_id": "own_score",
                        "history_status": "stipulated",
                        "history_rationale": "Whole path openly stipulated.",
                        "interpolation": "step",
                        "knots": [
                            {"time": 0, "value": 0, "model_revision": "r1"},
                            {"time": 1, "value": 0, "model_revision": "r1"},
                        ],
                    }
                ],
                "laws": [law],
            }
        ],
        "independence_assumptions": [
            "Distinct initiating origins are conditionally independent absent an authored experiment."
        ],
        "limitations": [
            "Synthetic numerical fixture; no empirical calibration or reliance claim."
        ],
    }


def partial_financial_payload(*, complete=False, payment=None):
    payload = study_payload()
    payload["contracts"] = [
        {"contract_id": "policy", "contract_year": "2027", "entity_id": "model"}
    ]
    member = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
        "member_outcomes"
    ][0]
    member.update(
        financial_mapping_complete=complete,
        obligations=[
            {"component_id": "victim_harm", "entity_id": "model", "fraction": 1}
        ],
        payments=[]
        if payment is None
        else [
            {"component_id": "victim_harm", "entity_id": "model", "fraction": payment}
        ],
        memberships=[
            {
                "component_id": "victim_harm",
                "contract_id": "policy",
                "obligated_entity_id": "model",
                "eligible": True,
            }
        ],
    )
    return payload


def stipulated_study() -> RunSetup:
    return setup_from_payload(study_payload())


def _study(*, rate=1.0, response_cost=0.0, unit_probability=1.0):
    component: dict[str, Any] = {
        "component_id": "harm",
        "amount": 100,
        "bearer_id": "victim",
        "nature": "resource_cost",
        "contributor_ids": ["model"],
        "attribution_weights": {"model": 1},
    }
    kernel: dict[str, Any] = {
        "cluster": {"family": "finite", "support": [{"count": 2, "probability": 1}]},
        "shared_probability": 1,
        "unit_probability": unit_probability,
        "campaign_components": []
        if not response_cost
        else [{**component, "component_id": "response", "amount": response_cost}],
        "member_outcomes": [{"probability": 1, "components": [component]}],
        "gate_meaning": "Independent prevention conditional on this complete state.",
        "selection_assumption": "Potential loss and prevention are independent.",
        "displacement_assumption": "Zero displacement is an explicit stipulation.",
    }
    law: dict[str, Any] = {
        "origin_id": "campaign",
        "scenario_id": "loss",
        "entity_id": "model",
        "reference_annual_rate": rate,
        "count_family": "poisson",
        "response": {
            "interpolation": "linear",
            "extrapolation": "flat",
            "points": [{"value": 0, "multiplier": 1}],
            "held_fixed": "Reference exposure and marks.",
        },
        "segments": [
            {
                "start": 0,
                "end": 1,
                "reference_exposure_share": 1,
                "initiation_probability": 1,
                "kernel": kernel,
            }
        ],
        "input_stage": "attempted_initiator",
        "qualification_assumption": "Authored attempted campaigns, not transported annual losses.",
    }
    state: dict[str, Any] = {
        "state_id": "low",
        "epistemic_group_id": "judgment-a",
        "weight": 0.25,
        "entity_paths": [
            {
                "entity_id": "model",
                "interpolation": "step",
                "scale_id": "test-scale",
                "history_status": "stipulated",
                "history_rationale": "Authored as of period start.",
                "knots": [
                    {"time": 0, "value": 0, "model_revision": "one"},
                    {"time": 1, "value": 0, "model_revision": "one"},
                ],
            }
        ],
        "laws": [law],
    }
    high: dict[str, Any] = {
        **state,
        "state_id": "high",
        "epistemic_group_id": "judgment-b",
        "weight": 0.75,
        "laws": [{**law, "reference_annual_rate": rate * 2}],
    }
    return setup_from_payload(
        {
            "scope_id": "software-oracle",
            "title": "Explicitly stipulated software oracle",
            "period_start": "2027-01-01T00:00:00Z",
            "period_end": "2028-01-01T00:00:00Z",
            "forecast_as_of": "2027-01-01T00:00:00Z",
            "data_cutoff": "2027-01-01T00:00:00Z",
            "availability_cutoff": "2027-01-01T00:00:00Z",
            "information_date": "2027-01-01T00:00:00Z",
            "capability_scales": [
                {
                    "scale_id": "test-scale",
                    "quantity_definition": "Authored zero coordinate.",
                    "unit": "dimensionless",
                    "reference_definition": "Zero reference.",
                    "measurement_protocol": "Stipulated fixture.",
                }
            ],
            "entities": [
                {
                    "entity_id": "model",
                    "kind": "model",
                    "revision": "one",
                    "definition": "One stipulated deployed revision.",
                    "display_name": "Model",
                    "status": "hypothetical",
                }
            ],
            "populations": [
                {
                    "scenario_id": "loss",
                    "definition": "All harm from these campaign origins.",
                    "causal_qualification": "Model contribution.",
                    "initiating_unit": "campaign",
                    "terminal_unit": "victim",
                    "severity_unit": "victim gross",
                    "monetary_basis": "2027 USD before recovery",
                    "currency": "USD",
                    "accounting_standing": "All scoped victims.",
                }
            ],
            "joint_states": [state, high],
            "alternatives": [
                {
                    "alternative_id": "contained",
                    "purpose": "scientific_alternative",
                    "description": "Perfect containment retains campaign response cost.",
                    "overrides": [
                        {
                            "state_id": item["state_id"],
                            "law": {
                                **item["laws"][0],
                                "segments": [
                                    {
                                        **law["segments"][0],
                                        "kernel": {**kernel, "unit_probability": 0},
                                    }
                                ],
                            },
                        }
                        for item in (state, high)
                    ],
                }
            ],
            "independence_assumptions": [
                "Independent origin arrivals conditional on the complete joint state."
            ],
            "limitations": ["Software validation fixture; no empirical claim."],
        }
    )


def _plan(simulations=1000):
    return ExecutionPlan.model_validate(
        {
            "plan_id": "frozen-oracle",
            "required_arms": ["reference", "contained"],
            "simulations": simulations,
            "seed": 492,
            "requested_views": [
                "gross",
                "welfare",
                "net_resource_cost",
                "scenario::loss::gross",
            ],
        }
    )


def _financial_study(scope="campaign", *, shape=None, cap=None, nature="resource_cost"):
    payload = encode_scientific(_study(rate=1))
    payload["contracts"] = [
        {
            "contract_id": "policy",
            "contract_year": "2027",
            "entity_id": "model",
            "occurrence_retention": 60,
            "occurrence_limit": 50,
        }
    ]
    for state in payload["joint_states"]:
        outcome = state["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0]
        component = outcome["components"][0]
        component["amount"] = 50
        component["nature"] = nature
        if shape is not None:
            component.update(
                amount=None,
                rank_group_id="loss-rank",
                severity=native_severity_from_sampler(
                    Distribution(
                        1, 2, 3, 4, 5, gpd_shape=shape, gpd_scale=1, severity_cap=cap
                    )
                ).model_dump(mode="json"),
            )
        outcome.update(
            financial_mapping_complete=True,
            obligations=[{"component_id": "harm", "entity_id": "model", "fraction": 1}],
            payments=[{"component_id": "harm", "entity_id": "model", "fraction": 1}],
            memberships=[
                {
                    "component_id": "harm",
                    "contract_id": "policy",
                    "obligated_entity_id": "model",
                    "eligible": True,
                    "occurrence_id": "shared",
                    "occurrence_scope": scope,
                }
            ],
        )
    return setup_from_payload(payload)


def _native_payment_study(*, mean, recovery_rate, shape=0.3, post_multiplier=1):
    payload = encode_scientific(_financial_study(shape=shape))
    payload["contracts"][0].update(occurrence_retention=0, occurrence_limit=None)
    for state in payload["joint_states"]:
        outcome = state["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0]
        component = outcome["components"][0]
        component["post_multiplier"] = post_multiplier
        outcome["obligations"], outcome["payments"] = [], []
        outcome["payment_kernel"] = {
            "source_severity": component["severity"],
            "rank_group_id": "loss-rank",
            "rank_scope": "member",
            "entity_id": "model",
            "liability_component_ids": ["harm"],
            "liability": {
                "mean": mean,
                "concentration": 10,
                "recovery_rate": recovery_rate,
                "severity_slope": 0,
                "median_severity": 3,
                "regulatory_recovery_threshold": 100,
                "regulatory_recovery_uplift": 0,
            },
        }
    return setup_from_payload(payload)


def _package(tmp_path, study=None, plan=None, name="package", *, evidence=None):
    return build_setup(
        destination=tmp_path / name,
        setup=with_execution(
            study or _study(), plan or _plan(), setup_id=name, evidence_files=evidence
        ),
        code_commit="software-fixture",
    )


def _count_control_study():
    payload = encode_scientific(_study())
    for state in payload["joint_states"]:
        state["laws"].append({**state["laws"][0], "origin_id": "other-campaign"})
    experiment = {
        "kind": "fixed_marginal_gaussian",
        "origin_groups": {"campaign": "all", "other-campaign": "all"},
        "correlation": 0,
    }
    payload["alternatives"] = [
        {
            "alternative_id": "control-b",
            "purpose": "scientific_alternative",
            "description": "Independent count-rank control.",
            "count_experiment": experiment,
            "matched_control_id": "control-b",
        },
        {
            "alternative_id": "control-a",
            "purpose": "scientific_alternative",
            "description": "Another independent control.",
            "count_experiment": experiment,
            "matched_control_id": "control-b",
        },
        {
            "alternative_id": "stress",
            "purpose": "scientific_alternative",
            "description": "Coupled count ranks.",
            "count_experiment": {**experiment, "correlation": 0.8},
            "matched_control_id": "control-a",
        },
    ]
    return setup_from_payload(payload)
