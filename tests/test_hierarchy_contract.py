from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from catmodel.hashing import canonical_json_sha256
from catmodel.resources import scenarios_path
from catmodel.scenarios.hierarchy_contract import (
    HierarchyContract,
    admission_clause,
    admission_statement,
    children_by_id,
    hierarchy_authority,
    load_hierarchy_contract,
    node_path,
    reporting_domain_of,
    terminal_node_ids,
    validate_hierarchy_contract,
)
from catmodel.scenarios.loader import ScenarioDefinition, load_scenarios
from test_support.hierarchy import scenario_owner_for_history, terminal_for_history

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

RULE = (
    "The tree splits losses by how they arise: what started it, what the AI did, and "
    "what process caused the loss. A leaf is one population of attempts taking one "
    "path to loss, so a node is split only where its incidents cannot share an "
    "exposure unit, an event chain or severity evidence. A split asks about a fact "
    "fixed when the incident begins: never one settled as it unfolds, such as its "
    "size, its reach or who bore it, never a condition of the whole study, such as "
    "capability or safeguards, and never a named scenario's definition. Each "
    "incident is counted once, with all its losses, under the process that caused "
    "most of them."
)


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_the_rule_is_quoted_verbatim_from_the_methodology() -> None:
    contract = load_hierarchy_contract()
    assert _flat(contract.rule) == RULE
    root = _PROJECT_ROOT
    methodology = _flat((root / "methodology.md").read_text(encoding="utf-8"))
    assert f"## Loss taxonomy {RULE}" in methodology
    header = " ".join(
        line.removeprefix("#").strip()
        for line in (root / "contracts/scenarios/hierarchy-contract.yaml")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("#")
    )
    assert RULE in _flat(header)
    # The README quotes the rule's first and last sentences.
    readme = _flat((root / "README.md").read_text(encoding="utf-8"))
    first, *_, last = (sentence.rstrip(".") + "." for sentence in RULE.split(". "))
    assert f"{first} {last}" in readme


def test_every_leaf_is_estimated_by_one_scenario_or_disposed() -> None:
    contract = load_hierarchy_contract()
    children = children_by_id(contract)
    terminals = terminal_node_ids(contract)
    scenarios = load_scenarios(scenarios_path())
    estimated = {node for scenario in scenarios for node in scenario.node_ids}
    dispositions = {
        node.node_id: node.unestimated_disposition.disposition
        for node in contract.nodes
        if node.unestimated_disposition is not None
    }
    assert len(estimated) == len(scenarios)
    assert estimated | set(dispositions) == terminals
    assert not estimated & set(dispositions)
    assert set(dispositions.values()) == {"unestimated", "unsupported"}
    assert all(len(kids) != 1 for kids in children.values())
    assert children["omega"] == ("n_misuse", "n_failure", "n_conduct", "n_other_origin")


def test_structure_and_projection_are_derived_from_parent_links() -> None:
    contract = load_hierarchy_contract()

    assert node_path(contract, "n_fraud_organizations") == (
        "deliberate_misuse/assisted_attacker/fraud/organizations"
    )
    assert reporting_domain_of(contract, "n_fraud_organizations") == "cyber"
    assert reporting_domain_of(contract, "n_reliance_health") == "eo"
    assert reporting_domain_of(contract, "n_influence") == "societal"


def _revalidated(scenario, **updates):
    payload = scenario.model_dump(mode="json")
    for path, value in updates.items():
        block, field = path.split(".")
        payload[block][field] = value
    return ScenarioDefinition.model_validate(payload)


def test_every_scenario_takes_the_one_admission_threshold_from_the_contract() -> None:
    contract = load_hierarchy_contract()
    admission = contract.admission
    assert (admission.comparator, admission.threshold, admission.currency) == (
        "ge",
        100_000,
        "USD",
    )
    assert "is at least USD 100,000; exactly USD 100,000 is admitted" in (
        admission_statement(contract)
    )
    target = load_scenarios(scenarios_path())[0]
    assert admission_clause(contract) in target.frequency_semantics.terminal_event_unit
    with pytest.raises(ValidationError, match="must state the admission threshold"):
        _revalidated(
            target,
            **{
                "frequency_semantics.terminal_event_unit": (
                    "One incident with at least USD 1 of loss."
                )
            },
        )


def test_a_scenario_cannot_add_a_second_threshold() -> None:
    target = load_scenarios(scenarios_path())[0]
    semantics = target.frequency_semantics
    for field, text in (
        ("terminal_event_unit", semantics.terminal_event_unit),
        ("counting_identity", semantics.counting_identity),
        ("year_rule", semantics.year_rule),
    ):
        appended = f"{text} Each victim must also lose at least USD 10,000."
        with pytest.raises(
            ValidationError, match=rf"{field}.*other than the contract's admission"
        ):
            _revalidated(target, **{f"frequency_semantics.{field}": appended})
    with pytest.raises(ValidationError, match="attribution_nexus"):
        _revalidated(
            target,
            **{
                "attribution_nexus.terminal_qualification": (
                    target.attribution_nexus.terminal_qualification
                    + " Only losses above USD 10,000 count."
                )
            },
        )


def test_authority_hashes_semantics_not_yaml_formatting() -> None:
    contract = load_hierarchy_contract()
    authority = hierarchy_authority(contract)

    assert authority.hierarchy_sha256 == canonical_json_sha256(
        contract.model_dump(mode="json", exclude_none=True)
    )
    payload = contract.model_dump(mode="json")
    mutated = deepcopy(payload)
    mutated["nodes"][1]["display_prose"] += " Material semantic change."
    changed = hierarchy_authority(HierarchyContract.model_validate(mutated))
    assert changed.hierarchy_sha256 != authority.hierarchy_sha256


def test_nonterminal_assignment_and_undisposed_terminal_fail() -> None:
    contract = load_hierarchy_contract()
    scenarios = load_scenarios(scenarios_path())
    changed = scenarios[0].model_copy(update={"node_ids": ("n_misuse",)})

    result = validate_hierarchy_contract(
        contract=contract, scenarios=[changed, *scenarios[1:]]
    )

    assert any("must assign terminal" in error for error in result.errors)
    assert any("needs an explicit disposition" in error for error in result.errors)


def test_residual_complements_must_name_siblings() -> None:
    contract = load_hierarchy_contract()
    payload = contract.model_dump(mode="json")
    residual = next(
        node for node in payload["nodes"] if node["node_id"] == "n_reliance_other"
    )
    residual["predicate"][0]["complement_of"] = ["n_conduct_designation"]

    result = validate_hierarchy_contract(
        contract=HierarchyContract.model_validate(payload)
    )

    assert any("complement references non-siblings" in error for error in result.errors)
    assert any("must name every other sibling" in error for error in result.errors)


def _replace_scenario(scenarios, changed):
    return [
        changed if scenario.id == changed.id else scenario for scenario in scenarios
    ]


def _developer_control(scenarios):
    return next(
        scenario for scenario in scenarios if scenario.id == "developer_control_loss"
    )


def test_ai_rd_uncapped_base_with_typed_exclusion_fails() -> None:
    contract = load_hierarchy_contract()
    scenarios = load_scenarios(scenarios_path())
    target = _developer_control(scenarios)
    assert target.severity_governance is not None
    governance = target.severity_governance.model_copy(
        update={
            "cap_posture": "uncapped",
            "severity_cap": None,
            "severity_cap_sensitivity_values": None,
            "severity_cap_uncapped_stress_alternative": False,
        }
    )
    changed = target.model_copy(update={"severity_governance": governance})

    result = validate_hierarchy_contract(
        contract=contract,
        scenarios=_replace_scenario(scenarios, changed),
    )

    assert any(
        "combines typed out-of-estimand exclusions with cap_posture='uncapped'" in error
        for error in result.errors
    )


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("health", "n_reliance_health"),
        ("legal", "n_reliance_legal"),
        ("financial", "n_reliance_financial"),
        ("software", "n_reliance_software"),
        ("other", "n_reliance_other"),
    ],
)
def test_each_reliance_field_has_exactly_one_owner(field, expected):
    contract = load_hierarchy_contract()
    facts = {"reliance_field": field}
    assert (
        terminal_for_history(
            facts, start_node_id="n_failure_output_reliance", contract=contract
        )
        == expected
    )
    scenario = next(
        s for s in load_scenarios(scenarios_path()) if expected in s.node_ids
    )
    assert (
        scenario_owner_for_history(
            facts, start_node_id="n_failure_output_reliance", contract=contract
        )
        == scenario.id
    )


def test_known_incomparable_origins_are_unsupported_without_a_joint_branch():
    contract = load_hierarchy_contract()
    nodes = {node.node_id: node for node in contract.nodes}
    facts = {"origin_regime": "other"}
    assert "losses are unsupported, never assigned to a named leaf" in _flat(
        contract.conventions
    )
    terminal = terminal_for_history(facts, contract=contract)
    assert terminal == "n_other_origin"
    assert scenario_owner_for_history(facts, contract=contract) is None
    disposition = nodes[terminal].unestimated_disposition
    assert disposition is not None and disposition.disposition == "unsupported"


@pytest.mark.parametrize("facts", [{}, {"origin_regime": None}])
def test_unresolved_origin_does_not_become_known_other(facts):
    with pytest.raises(ValueError, match=r"history fact.*origin_regime"):
        terminal_for_history(facts)


def test_known_multiple_causes_require_a_resolved_origin_classification():
    # A list of known causes is neither one axis answer nor a probability mixture.
    with pytest.raises(
        ValueError, match=r"invalid resolved history fact.*origin_regime"
    ):
        terminal_for_history({"origin_regime": {"deliberate_misuse", "ai_failure"}})


def test_an_unknown_fact_is_uncertainty_not_other():
    with pytest.raises(ValueError, match=r"history fact.*compromise_target"):
        terminal_for_history(
            {
                "origin_regime": "deliberate_misuse",
                "misuse_role": "assisted_attacker",
                "loss_process": "system_compromise",
            }
        )
