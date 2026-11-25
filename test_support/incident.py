"""Small common physical incident laws for independent execution oracles."""

import numpy as np

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.contracts.native_severity import native_severity_from_sampler
from catmodel.simulation.incident import compile_incident_state
from test_support.run_setup import setup_from_payload
from test_support.study import _plan, _study


def common_case(
    *, sampler=None, multiplier=1, financial=False, simulations=32, rate=10
):
    payload = encode_scientific(_study(rate=rate))
    payload["alternatives"] = []
    for state in payload["joint_states"]:
        kernel = state["laws"][0]["segments"][0]["kernel"]
        kernel["cluster"]["support"] = [{"count": 1, "probability": 1}]
        outcome = kernel["member_outcomes"][0]
        component = outcome["components"][0]
        component["scenario_ids"] = ["loss"]
        component["loss_type"] = "indemnity"
        if sampler is not None:
            component["amount"] = None
            component["severity"] = native_severity_from_sampler(sampler).model_dump(
                mode="json"
            )
            component["rank_group_id"] = "severity"
            component["multiplier"] = multiplier
        if financial:
            outcome["financial_mapping_complete"] = True
            outcome["obligations"] = [
                {"component_id": "harm", "entity_id": "model", "fraction": 1}
            ]
            outcome["payments"] = [
                {"component_id": "harm", "entity_id": "model", "fraction": 1}
            ]
            outcome["memberships"] = [
                {
                    "component_id": "harm",
                    "contract_id": "policy",
                    "obligated_entity_id": "model",
                    "eligible": True,
                    "occurrence_id": "incident",
                    "occurrence_scope": "member",
                }
            ]
    if financial:
        payload["contracts"] = [
            {
                "contract_id": "policy",
                "contract_year": "2027",
                "entity_id": "model",
                "occurrence_retention": 60,
                "annual_retention": 500,
            }
        ]
    study = setup_from_payload(payload)
    compiled = compile_incident_state(study, "low")
    plan = _plan(simulations).model_copy(
        update={
            "required_arms": ("reference",),
        }
    )
    return (
        study,
        plan,
        compiled,
        {"campaign": np.full(simulations, rate, dtype=np.int64)},
    )
