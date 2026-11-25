"""The agent harmful-action chain counts root causes, not deployment-years."""

from pathlib import Path

import pytest

from catmodel.scenarios.loader import load_scenarios

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ID = "agent_actions_for_deployer"


@pytest.fixture(scope="module")
def scenario():
    return next(s for s in load_scenarios(ROOT / "scenarios") if s.id == SCENARIO_ID)


def test_chain_counts_root_causes_with_one_incident_each(scenario):
    chain = scenario.event_chain
    assert chain.opportunity.unit.startswith("One root cause")
    rule = chain.opportunity.counting_rule
    assert "a defect reaching many deployments or decisions is one" in rule
    assert "at most one qualified incident" in rule
    (stage,) = chain.stages
    assert stage.stage_id == "action_executed"
    assert "blocked before execution does not" in stage.event
