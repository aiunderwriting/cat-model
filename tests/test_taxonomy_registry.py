from pathlib import Path

import pytest
from pydantic import ValidationError

from catmodel.scenarios.catalog import DOCUMENT_COUNT_MARKERS
from catmodel.scenarios.hierarchy_contract import HierarchyNode
from catmodel.scenarios.loader import load_scenarios
from catmodel.scenarios.taxonomy_registry import RISK_DOMAIN_SET

_SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def test_hierarchy_reporting_projections_use_registry_vocabulary() -> None:
    projected = {scenario.risk_domain for scenario in load_scenarios(_SCENARIOS_DIR)}

    assert projected == RISK_DOMAIN_SET


def test_hierarchy_nodes_assign_only_registry_domains() -> None:
    node = {
        "node_id": "n_test",
        "path_label": "test",
        "display_label": "Test",
        "display_summary": "Test.",
        "display_prose": "Test.",
    }
    assert HierarchyNode.model_validate({**node, "reporting_domain": "cyber"})
    with pytest.raises(ValidationError, match="reporting_domain"):
        HierarchyNode.model_validate({**node, "reporting_domain": "space"})


def test_domain_keyed_tables_cover_the_registry() -> None:
    assert "risk_domains" in DOCUMENT_COUNT_MARKERS[Path("README.md")]
