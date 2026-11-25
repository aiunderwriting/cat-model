from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from catmodel.identifiers import AtomicIdentifier
from catmodel.scenarios.loader import ScenarioDefinition

ROOT = Path(__file__).parent.parent


def test_atomic_identifier_reserves_colon_for_composite_grammars() -> None:
    adapter = TypeAdapter(AtomicIdentifier)

    assert adapter.validate_python("scenario_a") == "scenario_a"
    with pytest.raises(ValidationError, match="surrounding whitespace"):
        adapter.validate_python("  scenario_a ")
    with pytest.raises(ValidationError, match="must not contain ':'"):
        adapter.validate_python("scenario:a")


def test_scenario_authority_rejects_colon_bearing_ids() -> None:
    scenario_path = next((ROOT / "scenarios" / "misuse").glob("*.yaml"))
    scenario_raw = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    scenario_raw["id"] = "scenario:forged"
    with pytest.raises(ValidationError, match="must not contain ':'"):
        ScenarioDefinition.model_validate(scenario_raw)
