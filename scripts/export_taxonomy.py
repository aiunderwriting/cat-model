"""Render the canonical taxonomy as a standalone, nonquantitative HTML explorer.

Navigation tags are illustrative, overlapping views, never accounting assignments.
Sector tags name an explicit setting where available; broad populations stay broad.
This module has no command of its own:
``scripts/validate_scenario_library.py --fix`` regenerates the explorer with the
other derived artifacts.
"""

import json
from collections.abc import Sequence
from pathlib import Path

from catmodel.scenarios.hierarchy_contract import (
    HierarchyContract,
    children_by_id,
    hierarchy_authority,
    nodes_by_id,
    validate_hierarchy_contract,
)
from catmodel.scenarios.loader import (
    ScenarioDefinition,
    event_chain_terminal_stage,
    rendered_severity_occurrence_scope,
)

DEFAULT_EXPLORER_RELATIVE_PATH = Path("docs/taxonomy-explorer.html")
SECTORS = {
    "n_compromise_critical_infrastructure": "Critical infrastructure",
    "n_failure_developer_control": "AI providers",
    "n_misuse_asset": "AI providers",
    "n_conduct": "AI providers",
    "n_failure_physical_control": "Physical systems",
    "n_reliance_health": "Health care",
    "n_reliance_legal": "Legal services",
    "n_reliance_financial": "Financial services",
    "n_reliance_software": "Software",
}
INTERVENTIONS = {
    "n_misuse": "Misuse prevention",
    "n_fraud": "Human verification",
    "n_compromise": "Cyber defense",
    "n_cbrn": "Hazardous-use safeguards",
    "n_component_tampering": "Model supply-chain and configuration security",
    "n_misuse_asset": "Model and service security",
    "n_input_hijack": "Input isolation and agent permissions",
    "n_failure_developer_control": "Internal autonomy controls",
    "n_failure_autonomous_action": "Deployment permissions and containment",
    "n_failure_physical_control": "Physical safety assurance",
    "n_failure_output_reliance": "Output verification",
    "n_failure_bounded_decision": "Decision-system assurance",
    "n_failure_service": "Service resilience",
    "n_conduct": "Provider governance",
}
HARM_LABELS = {
    "economic_loss": "Economic loss",
    "financial_theft": "Theft",
    "bodily_injury": "Bodily injury",
    "property_damage": "Property damage",
    "discrimination_harm": "Discrimination harm",
    "ip_loss": "Intellectual-property loss",
    "regulatory_sanction": "Regulatory sanction",
}


def taxonomy_data(
    contract: HierarchyContract, scenarios: Sequence[ScenarioDefinition]
) -> dict:
    errors = validate_hierarchy_contract(contract=contract, scenarios=scenarios).errors
    if errors:
        raise ValueError("; ".join(errors))
    nodes = nodes_by_id(contract)
    children = children_by_id(contract)
    owners = {node: scenario for scenario in scenarios for node in scenario.node_ids}
    rows = []
    for node in contract.nodes:
        path = [node.node_id]
        while (parent := nodes[path[-1]].parent_id) is not None:
            path.append(parent)
        scenario = owners.get(node.node_id)
        disposition = node.unestimated_disposition
        status = "branch"
        if scenario:
            status = "forecast"
        elif disposition:
            status = disposition.disposition
        rows.append(
            {
                "id": node.node_id,
                "parent": node.parent_id,
                "label": node.display_label,
                "summary": node.display_summary,
                "definition": node.display_prose,
                "question": None
                if node.classification_question is None
                else node.classification_question.text,
                "split_rationale": node.split_rationale,
                "children": list(children[node.node_id]),
                "status": status,
                "forecast": scenario.id if scenario else None,
                "shared": bool(scenario and len(scenario.node_ids) > 1),
                "tags": {
                    "sector": [
                        next(
                            (SECTORS[p] for p in path if p in SECTORS),
                            "Cross-sector / unspecified",
                        )
                    ],
                    "harm": [HARM_LABELS[h] for h in scenario.facets.harm_types]
                    if scenario
                    else ["Not separately characterized"],
                    "intervention": sorted(
                        {INTERVENTIONS[p] for p in path if p in INTERVENTIONS}
                    )
                    or ["Not specified"],
                },
            }
        )
    return {
        **hierarchy_authority(contract).model_dump(),
        "rule": " ".join(contract.rule.split()),
        "nodes": rows,
        "forecasts": {
            scenario.id: {
                "label": scenario.name,
                "nodes": list(scenario.node_ids),
                "severity_scope": rendered_severity_occurrence_scope(scenario),
                "chain": [
                    ["Opportunity", scenario.event_chain.opportunity.unit],
                    ["Counting", scenario.event_chain.opportunity.counting_rule],
                    ["Cohort", scenario.event_chain.opportunity.cohort],
                    [
                        "Exposure",
                        " ".join(
                            f"{access.capitalize()}: rate per {base.unit}; "
                            f"{base.derivation}"
                            for access, base in (
                                ("public", scenario.event_chain.exposure.public),
                                ("internal", scenario.event_chain.exposure.internal),
                            )
                        ),
                    ],
                    ["Year rule", scenario.frequency_semantics.year_rule],
                    *(
                        [f"Stage {index}", stage.event]
                        for index, stage in enumerate(
                            (
                                *scenario.event_chain.stages,
                                event_chain_terminal_stage(scenario),
                            ),
                            1,
                        )
                    ),
                ],
            }
            for scenario in scenarios
        },
    }


def render_html(data: dict) -> str:
    # Escape HTML delimiters even in application/json: a literal closing script
    # tag otherwise ends the element before JSON parsing. DOM uses textContent.
    embedded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    for character, escaped in (("&", r"\u0026"), ("<", r"\u003c"), (">", r"\u003e")):
        embedded = embedded.replace(character, escaped)
    template = Path(__file__).with_name("templates") / "taxonomy.html"
    return template.read_text(encoding="utf-8").replace("__TAXONOMY_JSON__", embedded)
