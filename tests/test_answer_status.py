"""Each answer status owns its fields, in the emitted schema and at validation.

An answer with an availability status gives its reason field only under the
status that reason explains and leaves it null or absent otherwise. The JSON
schema a question shows and the runtime validator must agree on every such
rule, and a validation retry must name the broken rule and field.
"""

import json
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError

from catmodel.contracts.contextual_risk import (
    ContextualChainAnswer,
    ContextualComponentAnswer,
    ContextualRiskAnswer,
    ContextualRiskInstrument,
    EalCrosscheckAnswer,
    LabAnnualBusinessLossAnswer,
    LabCompensationAnswer,
    ParentAnswer,
    StatusFields,
    json_schema,
)
from catmodel.elicitation.contextual_risk import (
    contextual_prompts,
    parse_contextual_answer,
)
from catmodel.elicitation.question_text import ANSWERING_RULES
from catmodel.frozen_calls import validation_feedback
from test_support.contextual_risk import (
    chain_answer,
    contextual_answer,
    contextual_instrument,
    crosscheck_answer,
    declined_crosscheck,
    governed_contextual_answer,
    governed_contextual_instrument,
    hierarchy_instrument,
    parent_answer,
)

_REASONING = contextual_answer().reasoning.model_dump(mode="json")
_COUNT = dict.fromkeys(("p5", "p25", "p50", "p75", "p95"), 1.0)
_LOSS = {
    key: value
    for value, key in enumerate(("p5", "p25", "p50", "p75", "p95", "p99", "p999"), 1)
}


_SUPPLEMENT = {"loss": _LOSS, "share": 0.5}
"""Values for excluded fields that an available payload cannot hold at once
(a component's alternative amount laws, business loss's stochastic law)."""


def _value(base: dict[str, Any], name: str) -> Any:
    value = base.get(name)
    return _SUPPLEMENT.get(name) if value in (None, []) else value


def _dump(answer: BaseModel) -> dict[str, Any]:
    return answer.model_dump(mode="json")


def _branches() -> list[dict[str, Any]]:
    return _dump(contextual_answer())["branches"]


# Each answer kind: its model, an available payload supplying every field its
# status rule names, (answered status, reason status, reason field), the
# fields the answered status requires, the fields the reason's status
# excludes, and the fields of which an answered answer supplies exactly one.
ANSWERS: dict[str, tuple[Any, ...]] = {
    "direct": (
        ContextualRiskAnswer,
        lambda: {
            **_dump(contextual_answer()),
            "total_loss": _LOSS,
            "large_loss_count": _COUNT,
            "very_large_loss_count": _COUNT,
        },
        ("available", "unavailable", "unavailable_reason"),
        ("rate", "branches", "direct_eal"),
        (
            "rate",
            "large_loss_count",
            "very_large_loss_count",
            "total_loss",
            "lab_compensation",
            "lab_annual_business_loss",
            "branches",
            "direct_eal",
        ),
        (),
    ),
    "component": (
        ContextualComponentAnswer,
        lambda: {"component_id": "victim", "status": "available", "amount": 0},
        ("available", "unavailable", "unavailable_reason"),
        (),
        ("amount", "loss", "share"),
        ("amount", "loss", "share"),
    ),
    "lab_compensation": (
        LabCompensationAnswer,
        lambda: _dump(contextual_answer())["lab_compensation"],
        ("available", "unavailable", "unavailable_reason"),
        ("share",),
        ("share",),
        (),
    ),
    "lab_annual_business_loss": (
        LabAnnualBusinessLossAnswer,
        lambda: _dump(contextual_answer())["lab_annual_business_loss"],
        ("available", "unavailable", "unavailable_reason"),
        ("positive_probability",),
        ("positive_probability", "amount", "loss"),
        (),
    ),
    "chain": (
        ContextualChainAnswer,
        lambda: _dump(chain_answer()),
        ("available", "unavailable", "unavailable_reason"),
        ("opportunity_rate", "stages"),
        ("opportunity_rate", "stages"),
        (),
    ),
    "crosscheck": (
        EalCrosscheckAnswer,
        lambda: _dump(
            crosscheck_answer(
                {
                    "rate": _COUNT,
                    "total_loss": _LOSS,
                    "branches": _branches(),
                    "large_loss_count": _COUNT,
                    "very_large_loss_count": _COUNT,
                    "direct_eal": _COUNT,
                }
            )
        ),
        ("revised", "unchanged", "unchanged_reason"),
        ("revised",),
        (
            "revised",
            "rate",
            "total_loss",
            "branches",
            "large_loss_count",
            "very_large_loss_count",
            "direct_eal",
        ),
        (),
    ),
    "parent": (
        ParentAnswer,
        lambda: _dump(parent_answer(1e9, 1e6)),
        ("available", "unavailable", "unavailable_reason"),
        ("category_eal",),
        ("category_eal", "remainder_eal"),
        (),
    ),
}


def _rule(model) -> StatusFields:
    rule = model.model_config["json_schema_extra"]
    assert isinstance(rule, StatusFields)
    return rule


def _schema(model) -> dict[str, Any]:
    """The model's schema as nested answers see it, through ``$defs``."""
    schema = json_schema(ContextualRiskAnswer)
    if model in (
        ContextualComponentAnswer,
        LabCompensationAnswer,
        LabAnnualBusinessLossAnswer,
    ):
        return {"$ref": f"#/$defs/{model.__name__}", "$defs": schema["$defs"]}
    return json_schema(model)


def _cases(kind):
    model, available, statuses, required, excluded, alternatives = ANSWERS[kind]
    answered, reason_status, reason = statuses
    base = available()
    stated = {
        **{key: base[key] for key in ("component_id",) if key in base},
        "status": reason_status,
        reason: "No defensible judgment",
        **({"reasoning": _REASONING} if "reasoning" in base else {}),
    }
    lists = {
        name
        for name in (*required, *excluded)
        if json_schema(model)["properties"][name].get("type") == "array"
    }
    yield "available", base, None
    yield "available_reason_null", {**base, reason: None}, None
    yield (
        "available_reason_given",
        {**base, reason: "An explanation"},
        f"{reason}_not_allowed_when_{answered}",
    )
    yield "reason_status", stated, None
    missing = {key: value for key, value in stated.items() if key != reason}
    yield "reason_missing", missing, f"{reason}_required_when_{reason_status}"
    yield (
        "reason_null",
        {**stated, reason: None},
        f"{reason}_required_when_{reason_status}",
    )
    for name in required:
        omitted = {key: value for key, value in base.items() if key != name}
        yield f"{name}_omitted", omitted, f"{name}_required_when_{answered}"
        if name in lists:
            yield (
                f"{name}_empty",
                {**base, name: []},
                f"{name}_required_when_{answered}",
            )
    for name in excluded:
        yield (
            f"{name}_with_{reason_status}",
            {**stated, name: _value(base, name)},
            f"{name}_not_allowed_when_{reason_status}",
        )
        if name in lists:
            yield f"{name}_empty_with_{reason_status}", {**stated, name: []}, None
    if alternatives:
        exactly_one = (
            f"exactly_one_of_{'_'.join(alternatives)}_required_when_{answered}"
        )
        none = {key: value for key, value in base.items() if key not in alternatives}
        yield "no_alternative", none, exactly_one
        yield "two_alternatives", {**base, "share": 0.5}, exactly_one
        for name, value in (("amount", 0), ("loss", _LOSS), ("share", 0.5)):
            yield f"only_{name}", {**none, name: value}, None


def test_each_row_states_its_model_rule_completely():
    for (
        model,
        available,
        statuses,
        required,
        excluded,
        alternatives,
    ) in ANSWERS.values():
        rule = _rule(model)
        assert (rule.answered_status, rule.reason_status, rule.reason) == statuses
        assert (rule.required, rule.excluded, rule.alternatives) == (
            required,
            excluded,
            alternatives,
        )
        base = available()
        assert all(_value(base, name) is not None for name in excluded), model


@pytest.mark.parametrize(
    ("kind", "case"),
    [(kind, case) for kind in ANSWERS for case, _, _ in _cases(kind)],
)
def test_schema_and_validator_agree_and_name_the_broken_rule(kind, case):
    model = ANSWERS[kind][0]
    payload, code = next((row[1], row[2]) for row in _cases(kind) if row[0] == case)
    schema = _schema(model)
    Draft202012Validator.check_schema(schema)
    assert Draft202012Validator(schema).is_valid(payload) == (code is None)
    if code is None:
        model.model_validate(payload)
        return
    with pytest.raises(ValidationError) as caught:
        model.model_validate(payload)
    field = code.split("_required_when_")[0].split("_not_allowed_when_")[0]
    location = [] if code.startswith("exactly_one_of_") else [field]
    assert json.loads(validation_feedback(caught.value))["validation_errors"] == [
        {"type": code, "location": location}
    ]


def test_status_variants_name_their_status_and_use_no_negation():
    for kind, (model, *_rest) in ANSWERS.items():
        text = json.dumps(_schema(model))
        assert '"not"' not in text and '"if"' not in text, kind
        rule = _rule(model)
        variants = json_schema(model)["anyOf"]
        assert {row["properties"]["status"]["const"] for row in variants} == {
            rule.answered_status,
            rule.reason_status,
        }
        assert all(row["required"][0] == "status" for row in variants)


def test_node_specific_fields_are_required_where_the_question_asks_them():
    instrument = governed_contextual_instrument(("output_reliance_other",))
    node = instrument.nodes[0]
    assert node.total is not None and node.frequency_chain is not None
    assert node.total.very_large_loss_level is not None
    prompt = contextual_prompts(instrument)[0]
    direct = _dump(governed_contextual_answer(node))
    schema = ContextualRiskAnswer.json_schema_for(node)
    assert Draft202012Validator(schema).is_valid(direct)
    parse_contextual_answer(json.dumps(direct), prompt)
    for name in (
        "lab_compensation",
        "lab_annual_business_loss",
        "total_loss",
        "large_loss_count",
        "very_large_loss_count",
    ):
        omitted = {key: value for key, value in direct.items() if key != name}
        assert not Draft202012Validator(schema).is_valid(omitted), name
        with pytest.raises(ValueError):
            parse_contextual_answer(json.dumps(omitted), prompt)
    crosscheck_schema = EalCrosscheckAnswer.json_schema_for(node)
    assert "direct_eal" in crosscheck_schema["properties"]
    assert "rate" in crosscheck_schema["properties"]


def test_a_parent_supplies_its_remainder_exactly_when_it_has_modeled_children():
    parents = hierarchy_instrument(("output_reliance_other",)).parents
    with_children = next(row for row in parents if row.children)
    without = next(row for row in parents if not row.children)
    for parent, remainder in ((with_children, 1e6), (without, None)):
        schema = ParentAnswer.json_schema_for(parent)
        right = _dump(parent_answer(1e9, remainder))
        wrong = _dump(parent_answer(1e9, 1e6 if remainder is None else None))
        assert Draft202012Validator(schema).is_valid(right)
        ParentAnswer.model_validate(right).validate_against(parent)
        assert not Draft202012Validator(schema).is_valid(wrong)
        with pytest.raises(ValueError):
            ParentAnswer.model_validate(wrong).validate_against(parent)


def test_a_nested_reason_is_located_at_its_component():
    payload = _dump(contextual_answer())
    payload["branches"][0]["components"][0]["unavailable_reason"] = "Explained"
    with pytest.raises(ValidationError) as caught:
        ContextualRiskAnswer.model_validate(payload)
    assert json.loads(validation_feedback(caught.value))["validation_errors"] == [
        {
            "type": "unavailable_reason_not_allowed_when_available",
            "location": ["branches", 0, "components", 0, "unavailable_reason"],
        }
    ]


def test_every_reason_field_is_described_by_its_status():
    for model, _, (_, reason_status, reason), *_rest in ANSWERS.values():
        description = json_schema(model)["properties"][reason]["description"]
        assert f"only when status is {reason_status}" in description, model
    assert declined_crosscheck().unchanged_reason


def test_direct_question_states_the_rule_and_names_citable_cards():
    instrument = contextual_instrument()
    prompt = contextual_prompts(instrument)[0]
    text = prompt.render()
    assert (
        "If the scenario cannot be estimated at all, return status unavailable "
        "with only unavailable_reason and reasoning." in text
    )
    assert "With status available, set unavailable_reason to null or omit it" in text
    assert ANSWERING_RULES in text
    assert (
        "No source excerpts are supplied for this question, so evidence_used is "
        "empty." in text
    )
    payload = _dump(contextual_answer())
    payload["reasoning"]["evidence_used"] = ["invented"]
    with pytest.raises(ValueError) as caught:
        parse_contextual_answer(json.dumps(payload), prompt)
    assert json.loads(validation_feedback(caught.value))["validation_errors"] == [
        {
            "type": "evidence_used_must_cite_supplied_cards",
            "location": ["reasoning", "evidence_used", 0],
        }
    ]


def test_supplied_cards_are_named_and_only_they_are_citable():
    payload = contextual_instrument().model_dump(mode="json")
    payload["nodes"][0]["source_cards"] = [
        {
            "card_id": card,
            "text": "Synthetic excerpt",
            "observation_through_utc": "2025-01-01",
            "availability_date_utc": "2025-01-01",
            "availability_basis": "synthetic",
            "sources": [f"https://example.org/{card}"],
        }
        for card in ("card-a", "card-b")
    ]
    prompt = contextual_prompts(ContextualRiskInstrument.model_validate(payload))[0]
    text = prompt.render()
    assert "evidence_used lists only IDs of excerpts supplied in this question" in text
    for card in ("card-a", "card-b"):
        assert (
            f'<excerpt id="{card}">\n'
            "Observations through 2025-01-01; available by 2025-01-01.\n"
            "Synthetic excerpt\n"
            f"Sources: https://example.org/{card}\n"
            "</excerpt>"
        ) in text
    answer = _dump(contextual_answer())
    answer["reasoning"]["evidence_used"] = ["card-b"]
    parse_contextual_answer(json.dumps(answer), prompt)
    answer["reasoning"]["evidence_used"] = ["card-b", "card-c"]
    with pytest.raises(ValueError) as caught:
        parse_contextual_answer(json.dumps(answer), prompt)
    assert json.loads(validation_feedback(caught.value))["validation_errors"] == [
        {
            "type": "evidence_used_must_cite_supplied_cards",
            "location": ["reasoning", "evidence_used", 1],
        }
    ]


def test_an_unasked_large_loss_field_leaves_every_status_branch():
    node = contextual_instrument().nodes[0]
    assert node.frequency_chain is not None and node.total is None
    for schema in (ContextualRiskAnswer.json_schema_for(node),):
        offered = set(schema["properties"])
        assert "large_loss_count" not in offered
        for variant in schema["anyOf"]:
            assert set(variant["properties"]) <= offered
            assert set(variant["required"]) <= offered
