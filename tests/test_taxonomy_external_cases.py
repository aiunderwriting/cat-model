"""Replay explicitly separated interpretations of primary-source narratives.

The classifier is given the author's interpretation; it does not read prose.
Missing incident evidence must not silently become a residual classification.
"""

from pathlib import Path

import pytest
import yaml

from catmodel.resources import scenarios_path
from catmodel.scenarios.hierarchy_contract import load_hierarchy_contract
from catmodel.scenarios.loader import load_scenarios
from test_support.hierarchy import scenario_owner_for_history, terminal_for_history

CASES = yaml.safe_load(
    (Path(__file__).parent / "fixtures/taxonomy_external_cases.yaml").read_text()
)["cases"]


@pytest.fixture(scope="module")
def contract():
    return load_hierarchy_contract()


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios(scenarios_path())


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(answer, id=f"{case['id']}-{i}")
        for case in CASES
        for i, answer in enumerate(case["interpretation"]["replay"])
    ],
)
def test_external_case_interpretation_preserves_known_and_missing_facts(
    answer, contract, scenarios
):
    facts = answer["facts"]
    if "missing_fact" in answer:
        with pytest.raises(ValueError, match=answer["missing_fact"]):
            terminal_for_history(facts, contract=contract)
        return
    assert terminal_for_history(facts, contract=contract) == answer["terminal_id"]
    assert (
        scenario_owner_for_history(facts, contract=contract, scenarios=scenarios)
        == answer["scenario_id"]
    )


def test_accidental_disclosure_is_not_in_the_outage_forecast(contract, scenarios):
    facts = {
        "origin_regime": "ai_failure",
        "failure_role": "service_failure",
        "service_failure_kind": "other",
    }
    assert terminal_for_history(facts, contract=contract) == "n_service_other"
    assert (
        scenario_owner_for_history(facts, contract=contract, scenarios=scenarios)
        is None
    )
    # Unknown cause is not evidence that the cause is a known other mechanism.
    facts.pop("service_failure_kind")
    with pytest.raises(ValueError, match="service_failure_kind"):
        terminal_for_history(facts, contract=contract)
