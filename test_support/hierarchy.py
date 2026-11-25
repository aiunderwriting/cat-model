"""Map explicit history facts to a taxonomy terminal in boundary tests.

The caller supplies every fact a traversed split reads; a missing or invalid fact
raises instead of being inferred.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from catmodel.resources import scenarios_path
from catmodel.scenarios.hierarchy_contract import (
    Clause,
    HierarchyContract,
    ScenarioPlacement,
    children_by_id,
    load_hierarchy_contract,
    nodes_by_id,
)
from catmodel.scenarios.loader import load_scenarios


def terminal_for_history(
    facts: Mapping[str, Any],
    *,
    start_node_id: str = "omega",
    contract: HierarchyContract | None = None,
) -> str:
    """Traverse every selected split and require exactly one matching child."""
    contract = contract or load_hierarchy_contract()
    nodes = nodes_by_id(contract)
    children = children_by_id(contract)

    def required(key: str) -> Any:
        if key not in facts:
            raise ValueError(f"missing declared history fact {key!r}")
        return facts[key]

    def clause_matches(clause: Clause) -> bool:
        if clause.complement_of is not None:
            matches = [node_matches(node_id) for node_id in clause.complement_of]
            return not any(matches)
        if clause.any_of is not None:
            return any([clause_matches(child) for child in clause.any_of])
        if clause.axis is not None:
            value = required(clause.axis)
            axis = contract.axes[clause.axis]
            expected = clause.value_in or (clause.value,)
            if axis.set_valued:
                if not isinstance(value, (set, frozenset, tuple, list)) or not set(
                    value
                ) <= set(axis.values):
                    raise ValueError(f"invalid set-valued history fact {clause.axis!r}")
                result = bool(set(value) & set(expected))
            else:
                if not isinstance(value, str) or value not in axis.values:
                    raise ValueError(f"invalid resolved history fact {clause.axis!r}")
                result = value in expected
            return not result if clause.negate else result
        raise AssertionError(f"unsupported test clause: {clause}")

    def node_matches(node_id: str) -> bool:
        return all([clause_matches(clause) for clause in nodes[node_id].predicate])

    current = start_node_id
    visited: set[str] = set()
    while children[current]:
        assert current not in visited, f"cyclic history path at {current!r}"
        visited.add(current)
        matches = [node_id for node_id in children[current] if node_matches(node_id)]
        assert len(matches) == 1, f"history at {current!r} has owners {matches}"
        current = matches[0]
    return current


def scenario_owner_for_history(
    facts: Mapping[str, Any],
    *,
    start_node_id: str = "omega",
    contract: HierarchyContract | None = None,
    scenarios: Sequence[ScenarioPlacement] | None = None,
) -> str | None:
    """Resolve a terminal to its sole forecast, including multi-terminal unions."""
    terminal = terminal_for_history(
        facts, start_node_id=start_node_id, contract=contract
    )
    scenarios = load_scenarios(scenarios_path()) if scenarios is None else scenarios
    owners = [scenario.id for scenario in scenarios if terminal in scenario.node_ids]
    assert len(owners) <= 1, (
        f"terminal {terminal!r} has duplicate scenario owners {owners}"
    )
    return owners[0] if owners else None
