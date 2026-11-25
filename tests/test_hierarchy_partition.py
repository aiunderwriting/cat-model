"""Each authored local split sends every declared answer to exactly one child."""

from itertools import combinations

import pytest

from catmodel.scenarios.hierarchy_contract import (
    Clause,
    HierarchyContract,
    children_by_id,
    load_hierarchy_contract,
    nodes_by_id,
    validate_hierarchy_contract,
)
from test_support.hierarchy import terminal_for_history


def _answers(contract: HierarchyContract, dimension: str) -> tuple[object, ...]:
    axis = contract.axes[dimension.removeprefix("axis:")]
    if axis.set_valued:
        return tuple(
            frozenset(subset)
            for size in range(len(axis.values) + 1)
            for subset in combinations(axis.values, size)
        )
    return axis.values


def _assert_local_partition(contract: HierarchyContract, parent_id: str) -> None:
    nodes = nodes_by_id(contract)
    children = children_by_id(contract)[parent_id]
    question = nodes[parent_id].classification_question
    assert question is not None
    # Stop the existing evaluator after one split. Ancestor and descendant facts
    # are intentionally absent; immediate children must use only this dimension.
    local = contract.model_copy(
        update={"nodes": tuple(nodes[node_id] for node_id in (parent_id, *children))}
    )
    fact_name = question.dimension.split(":", 1)[1]
    reached = set()
    for answer in _answers(contract, question.dimension):
        try:
            reached.add(
                terminal_for_history(
                    {fact_name: answer}, start_node_id=parent_id, contract=local
                )
            )
        except AssertionError as exc:
            raise AssertionError(f"{fact_name}={answer!r}: {exc}") from exc
    assert reached == set(children), f"unreachable children: {set(children) - reached}"


@pytest.mark.parametrize(
    "parent_id",
    [
        node.node_id
        for node in load_hierarchy_contract().nodes
        if node.classification_question is not None
    ],
)
def test_every_local_split_partitions_all_declared_answers(parent_id: str) -> None:
    _assert_local_partition(load_hierarchy_contract(), parent_id)


@pytest.mark.parametrize(
    ("node_id", "predicate", "parent_id", "message"),
    [
        (
            "n_reliance_legal",
            (Clause(axis="reliance_field", value="financial"),),
            "n_failure_output_reliance",
            r"reliance_field='financial': .*owners \['n_reliance_legal', 'n_reliance_financial'\]",
        ),
        (
            "n_cbrn_bio_expert",
            (
                Clause(axis="program_expertise", value="expert"),
                Clause(axis="program_expertise", value="novice"),
            ),
            "n_cbrn_biological",
            r"program_expertise='expert': .*owners \[\]",
        ),
    ],
    ids=["reliance-field-overlap", "expertise-gap"],
)
def test_regression_detects_semantic_defects_structural_validation_accepts(
    node_id: str, predicate: tuple[Clause, ...], parent_id: str, message: str
) -> None:
    contract = load_hierarchy_contract()
    changed = contract.model_copy(
        update={
            "nodes": tuple(
                node.model_copy(update={"predicate": predicate})
                if node.node_id == node_id
                else node
                for node in contract.nodes
            )
        }
    )
    assert validate_hierarchy_contract(contract=changed).errors == ()
    with pytest.raises(AssertionError, match=message):
        _assert_local_partition(changed, parent_id)
