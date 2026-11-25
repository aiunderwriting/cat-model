"""Typed single-fact classification and full-path native prompt boundaries."""

import pytest
import yaml
from pydantic import ValidationError

from catmodel.scenarios.hierarchy_contract import (
    ClassificationQuestion,
    Clause,
    HierarchyContract,
    load_hierarchy_contract,
    validate_hierarchy_contract,
)


def changed_node(node_id, **updates) -> HierarchyContract:
    payload = load_hierarchy_contract().model_dump(mode="json")
    next(node for node in payload["nodes"] if node["node_id"] == node_id).update(
        updates
    )
    return HierarchyContract.model_validate(payload)


def errors(contract: HierarchyContract) -> str:
    return "\n".join(validate_hierarchy_contract(contract=contract).errors)


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
def test_classification_question_rejects_blank_text(text) -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        ClassificationQuestion(text=text, dimension="axis:cbrn_hazard")


@pytest.mark.parametrize(
    "dimension", ["", "cbrn_hazard", "prose:anything", "axis:", "qualifier:any"]
)
def test_classification_question_rejects_unknown_reference_form(dimension) -> None:
    with pytest.raises(ValidationError, match="must name a declared axis"):
        ClassificationQuestion(text="Which hazard?", dimension=dimension)


def test_a_split_never_classifies_a_loss_threshold() -> None:
    with pytest.raises(ValidationError, match="is a loss threshold"):
        ClassificationQuestion(
            text="Is the loss above USD 1 million?", dimension="threshold:loss_1m"
        )


def test_classification_requires_a_declared_axis() -> None:
    contract = changed_node(
        "n_compromise",
        classification_question={"text": "Which answer?", "dimension": "axis:missing"},
    )
    assert "unknown classification dimension axis:missing" in errors(contract)


def test_internal_node_cannot_omit_question() -> None:
    assert "internal node needs a classification question" in errors(
        changed_node("n_compromise", classification_question=None)
    )


def test_every_split_names_the_estimate_that_differs() -> None:
    contract = load_hierarchy_contract()
    splits = [node for node in contract.nodes if node.classification_question]
    assert len(splits) == 14
    assert all(node.split_rationale for node in splits)
    assert "needs a split_rationale" in errors(
        changed_node("n_compromise", split_rationale=None)
    )


def test_terminal_cannot_carry_a_split_rationale() -> None:
    assert "cannot carry a classification question or split_rationale" in errors(
        changed_node("n_reliance_legal", split_rationale="An estimate differs.")
    )


@pytest.mark.parametrize("defect", ["missing_question", "mixed_dimension"])
def test_native_load_rejects_structural_defect_before_prompt_generation(
    tmp_path, defect
) -> None:
    payload = load_hierarchy_contract().model_dump(mode="json")
    if defect == "missing_question":
        node = next(n for n in payload["nodes"] if n["node_id"] == "n_compromise")
        node["classification_question"] = None
        message = "internal node needs a classification question"
    else:
        node = next(n for n in payload["nodes"] if n["node_id"] == "n_cbrn_biological")
        node["predicate"].append({"axis": "program_expertise", "value": "expert"})
        message = "child predicate uses axis:program_expertise"
    path = tmp_path / "hierarchy.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match=message):
        load_hierarchy_contract(path)


def test_native_load_accepts_a_self_contained_smaller_tree_without_scenario_roster(
    tmp_path,
) -> None:
    payload = load_hierarchy_contract().model_dump(mode="json")
    retained = {"n_cbrn_biological", "n_cbrn_bio_expert", "n_cbrn_bio_novice"}
    payload["nodes"] = [
        node for node in payload["nodes"] if node["node_id"] in retained
    ]
    root = next(
        node for node in payload["nodes"] if node["node_id"] == "n_cbrn_biological"
    )
    root.update(parent_id=None, predicate=[])
    path = tmp_path / "scoped-hierarchy.yaml"
    path.write_text(yaml.safe_dump(payload))
    assert len(load_hierarchy_contract(path).nodes) == 3


def test_terminal_cannot_ask_an_unanswered_classification_question() -> None:
    assert "cannot carry a classification question" in errors(
        changed_node(
            "n_reliance_legal",
            classification_question={
                "text": "Which hazard?",
                "dimension": "axis:cbrn_hazard",
            },
        )
    )


def test_nonroot_internal_node_requires_its_parent_answer() -> None:
    assert "non-root nodes need a local predicate" in errors(
        changed_node("n_compromise", predicate=[])
    )


def test_unary_node_is_not_a_classification_split() -> None:
    payload = load_hierarchy_contract().model_dump(mode="json")
    payload["nodes"] = [
        node for node in payload["nodes"] if node["node_id"] != "n_cbrn_bio_novice"
    ]
    assert "n_cbrn_biological: classification requires at least two children" in (
        errors(HierarchyContract.model_validate(payload))
    )


def test_child_cannot_smuggle_an_additional_dimension_into_parent_answer() -> None:
    contract = changed_node(
        "n_cbrn_biological",
        predicate=[
            {"axis": "cbrn_hazard", "value": "biological"},
            {"axis": "program_expertise", "value": "expert"},
        ],
    )
    assert (
        "child predicate uses axis:program_expertise, but parent classifies "
        "axis:cbrn_hazard"
    ) in errors(contract)


def test_any_of_accepts_alternative_values_of_the_same_fact() -> None:
    contract = changed_node(
        "n_cbrn_chemical",
        predicate=[
            {
                "any_of": [
                    {"axis": "cbrn_hazard", "value": "chemical"},
                    {"axis": "cbrn_hazard", "value": "radiological_nuclear"},
                ]
            }
        ],
    )
    assert errors(contract) == ""


def test_any_of_cannot_hide_a_second_fact() -> None:
    contract = changed_node(
        "n_cbrn_chemical",
        predicate=[
            {
                "any_of": [
                    {"axis": "cbrn_hazard", "value": "chemical"},
                    {"axis": "program_expertise", "value": "novice"},
                ]
            }
        ],
    )
    assert "child predicate uses axis:program_expertise" in errors(contract)


def test_nested_complement_must_refer_to_another_answer_at_same_split() -> None:
    contract = changed_node(
        "n_cbrn_chemical",
        predicate=[
            {
                "any_of": [
                    {"axis": "cbrn_hazard", "value": "chemical"},
                    {"complement_of": ["n_reliance_legal"]},
                ]
            }
        ],
    )
    assert "complement references non-siblings ['n_reliance_legal']" in errors(contract)


@pytest.mark.parametrize("payload", [{"any_of": []}, {"complement_of": []}])
def test_empty_logical_clause_cannot_answer_a_question(payload) -> None:
    with pytest.raises(ValidationError):
        Clause.model_validate(payload)


def test_a_clause_form_the_contract_does_not_define_is_not_accepted() -> None:
    with pytest.raises(ValidationError):
        Clause.model_validate({"unknown_form": True})
