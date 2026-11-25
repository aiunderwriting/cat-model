"""The public explorer preserves accounting identity and numerical ownership."""

import json
import re
from pathlib import Path

import pytest

from catmodel.resources import scenarios_path
from catmodel.scenarios.hierarchy_contract import (
    hierarchy_authority,
    load_hierarchy_contract,
    terminal_node_ids,
)
from catmodel.scenarios.loader import load_scenarios, rendered_severity_occurrence_scope
from scripts.export_taxonomy import render_html, taxonomy_data


@pytest.fixture
def exported():
    return taxonomy_data(load_hierarchy_contract(), load_scenarios(scenarios_path()))


def test_committed_explorer_matches_current_sources(exported):
    artifact = Path(__file__).resolve().parents[1] / "docs/taxonomy-explorer.html"
    assert artifact.read_text(encoding="utf-8") == render_html(exported), (
        "Regenerate it with uv run python scripts/validate_scenario_library.py --fix"
    )


def test_export_preserves_nodes_paths_and_forecast_unions(exported):
    contract = load_hierarchy_contract()
    source = {node.node_id: node for node in contract.nodes}
    assert set(source) == {node["id"] for node in exported["nodes"]}
    for node in exported["nodes"]:
        assert node["parent"] == source[node["id"]].parent_id
        assert node["label"] == source[node["id"]].display_label
        assert node["summary"] == source[node["id"]].display_summary
    assert {
        node["id"] for node in exported["nodes"] if not node["children"]
    } == terminal_node_ids(contract)
    for scenario in load_scenarios(scenarios_path()):
        assert exported["forecasts"][scenario.id]["nodes"] == list(scenario.node_ids)
        assert exported["forecasts"][scenario.id][
            "severity_scope"
        ] == rendered_severity_occurrence_scope(scenario)
        assert "{{" not in exported["forecasts"][scenario.id]["severity_scope"]
        assert {
            node["id"] for node in exported["nodes"] if node["forecast"] == scenario.id
        } == set(scenario.node_ids)
    authority = hierarchy_authority(contract).model_dump()
    assert all(exported[key] == value for key, value in authority.items())


def test_forecasts_are_distinct_from_unestimated_and_unsupported_cells(exported):
    nodes = {node["id"]: node for node in exported["nodes"]}
    statuses = {node["status"] for node in nodes.values()}
    assert statuses == {"branch", "forecast", "unestimated", "unsupported"}
    assert nodes["n_reliance_other"]["status"] == "forecast"
    assert not nodes["n_reliance_other"]["shared"]
    assert nodes["n_service_other"]["status"] == "unestimated"
    assert nodes["n_service_other"]["forecast"] is None
    assert nodes["n_other_origin"]["status"] == "unsupported"


def test_navigation_tags_preserve_broad_scopes_and_scenario_harms(exported):
    nodes = {node["id"]: node for node in exported["nodes"]}
    assert nodes["n_reliance_health"]["tags"]["sector"] == ["Health care"]
    assert nodes["n_compromise_organizations"]["tags"]["sector"] == [
        "Cross-sector / unspecified"
    ]
    assert "Bodily injury" in nodes["n_reliance_health"]["tags"]["harm"]
    assert set(nodes["n_compromise_organizations"]["tags"]["intervention"]) == {
        "Misuse prevention",
        "Cyber defense",
    }
    assert all(all(node["tags"].values()) for node in nodes.values())


def test_every_split_shows_its_question_and_rationale(exported):
    for node in exported["nodes"]:
        assert bool(node["question"]) == bool(node["children"]), node["id"]
        assert bool(node["split_rationale"]) == bool(node["children"]), node["id"]
    assert exported["rule"].startswith("The tree splits losses by how they arise")


def test_embedded_text_cannot_end_json_script_or_execute_markup(exported):
    hostile = '</script><script>alert("test")</script> & <b>markup</b>\u2028'
    exported["nodes"][0]["summary"] = hostile
    rendered = render_html(exported)
    match = re.search(
        r'<script type="application/json" id="taxonomy-data">(.*?)</script>',
        rendered,
        re.DOTALL,
    )
    assert match is not None
    assert "<" not in match[1]
    assert json.loads(match[1]) == exported
    assert hostile not in rendered
    assert "innerHTML" not in rendered
