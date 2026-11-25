"""A synthetic threat model for software tests.

Its text is a fixture about service outages, not research: every source is a
fixture citation and no claim describes the world.
"""

from datetime import date
from typing import Any

from catmodel.resources import scenarios_path
from catmodel.scenarios.loader import ScenarioDefinition, load_scenarios
from catmodel.scenarios.threat_models import (
    ThreatModel,
    brief_sha256,
    research_brief,
    scenario_definition_sha256,
)

SCENARIO_ID = "service_outage_degradation"
RESEARCH_THROUGH = date(2026, 6, 30)
REVIEW = {
    "reviewed_by": "Fixture reviewer",
    "reviewer_kind": "ai_agent",
    "reviewed_on": "2026-07-15",
    "accuracy_review": "Fixture: every cited finding was opened and read.",
    "hazard_review": "Fixture: no step says how it is done.",
}
"""The ``review`` record of ``payload_with_records``."""


def library_scenario(scenario_id: str = SCENARIO_ID) -> ScenarioDefinition:
    return next(
        row for row in load_scenarios(scenarios_path()) if row.id == scenario_id
    )


def threat_model_payload(scenario_id: str = SCENARIO_ID) -> dict[str, Any]:
    """A complete synthetic threat model for ``scenario_id``, written against
    its current definition, as a register entry holds it: without the
    optional ``drafting`` and ``review`` records."""
    scenario = library_scenario(scenario_id)
    return {
        "threat_model_id": f"tm-{scenario_id.replace('_', '-')}",
        "scenario_id": scenario_id,
        "scenario_definition_sha256": scenario_definition_sha256(scenario),
        "research_through_utc": RESEARCH_THROUGH.isoformat(),
        "summary": "Fixture summary: a production fault becomes a counted "
        "incident when failover does not mask it and customers depend on the "
        "service at that moment.",
        "actors": [
            {
                "actor_id": "service-operators",
                "description": "Fixture: the lab's own engineers, who change "
                "production systems many times a day.",
            }
        ],
        "pathway": [
            {
                "step_id": "production-fault",
                "description": "Fixture: a change or dependency failure "
                "introduces a fault in the serving path.",
                "ai_contribution": "Fixture: automated change tooling raises the "
                "rate of production changes.",
                "bottleneck": "Fixture: staged rollouts catch most faults "
                "before they reach every region.",
                "defender_response": "Fixture: canary analysis and automatic rollback.",
                "sources": ["fixture-postmortem"],
            },
            {
                "step_id": "customer-dependence",
                "description": "Fixture: customers have no fallback and their "
                "own services stop.",
                "ai_contribution": "Fixture: deeper integration leaves fewer "
                "manual fallbacks.",
                "bottleneck": "Fixture: large customers keep a second provider.",
                "defender_response": "Fixture: customers fail over or degrade "
                "gracefully.",
            },
            {
                "step_id": "shared-dependency",
                "kind": "escalation",
                "description": "Fixture: the fault sits in a dependency that "
                "other providers share.",
                "ai_contribution": "Fixture: none of its own.",
                "bottleneck": "Fixture: few dependencies are shared so widely.",
                "defender_response": "Fixture: providers isolate the dependency.",
            },
        ],
        "severity_modes": [
            {
                "mode_id": "contained",
                "description": "Fixture: one region fails for a short time.",
                "loss_drivers": "Fixture: duration and the share of customers "
                "in the region.",
            },
            {
                "mode_id": "propagating",
                "description": "Fixture: every region fails and recovery "
                "needs manual work.",
                "loss_drivers": "Fixture: time to restore and customers' dependence.",
            },
        ],
        "analogues": [
            {
                "analogue_id": "fixture-outage-2024",
                "description": "Fixture: a software update stopped many "
                "customers' systems on one day in 2024.",
                "outcome": "Fixture: a reported direct loss of USD 5 billion, "
                "of which about 15% was insured.",
                "sources": ["fixture-loss-estimate"],
            }
        ],
        "key_uncertainties": [
            "Fixture: how many customers keep a working fallback.",
        ],
        "liability": {
            "theories": "Fixture: breach of a service-level agreement.",
            "defences": "Fixture: liability caps in the service terms.",
            "contract_terms": "Fixture: service credits as the sole remedy.",
        },
        "sources": [
            {
                "source_id": "fixture-postmortem",
                "citation": "Fixture postmortem, 2025",
                "published_utc": "2025-03-01",
                "accessed_utc": "2026-06-01",
                "finding": "Fixture: the outage began with a configuration change.",
            },
            {
                "source_id": "fixture-loss-estimate",
                "citation": "Fixture loss estimate, 2024",
                "published_utc": "2024-08-01",
                "accessed_utc": "2026-07-02",
                "finding": "Fixture: direct loss of USD 5 billion, about 15% insured.",
            },
        ],
    }


def draft_payload(scenario_id: str = SCENARIO_ID) -> dict[str, Any]:
    """The synthetic threat model as a research stage returns it: with the
    working notes of a draft that answers the brief for a scenario without
    source cards."""
    return {
        **threat_model_payload(scenario_id),
        "drafting": {
            "drafted_by": "Fixture research agent with retrieval",
            "brief_sha256": brief_sha256(
                research_brief(
                    library_scenario(scenario_id), (), research_through=RESEARCH_THROUGH
                )
            ),
            "left_out": ["Fixture report: not opened."],
        },
    }


def payload_with_records(scenario_id: str = SCENARIO_ID) -> dict[str, Any]:
    """The synthetic threat model with both optional records, as an entry of
    a register that kept them holds it."""
    return {**draft_payload(scenario_id), "review": dict(REVIEW)}


def synthetic_threat_model(scenario_id: str = SCENARIO_ID) -> ThreatModel:
    return ThreatModel.model_validate(threat_model_payload(scenario_id))
