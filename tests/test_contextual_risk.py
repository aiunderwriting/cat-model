"""Context-bound questions preserve complete matched economic judgments."""

import copy
import json

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from catmodel.contracts.contextual_risk import (
    ContextualComponentAnswer,
    ContextualRiskAnswer,
    ContextualRiskArtifact,
    ContextualRiskInstrument,
    LabAnnualBusinessLossAnswer,
)
from catmodel.elicitation.contextual_frequency import validate_contextual_artifact
from catmodel.elicitation.contextual_risk import (
    ContextualPrompt,
    contextual_prompts,
    parse_contextual_answer,
)
from test_support.contextual_risk import (
    contextual_answer,
    contextual_artifact,
    contextual_instrument,
    governed_contextual_instrument,
)
from test_support.question_text import answer_schema_of
from test_support.version_markers import unstated_markers


def test_complete_synthetic_artifact_is_bound_to_exact_context_and_anchors():
    instrument = contextual_instrument()
    artifact = contextual_artifact(instrument)
    validate_contextual_artifact(artifact, instrument)
    assert [case.anchor_score for case in artifact.cases] == [0, 0.5, 2]
    assert all(not case.answer.reasoning.evidence_used for case in artifact.cases)


def test_prompt_explains_constant_path_and_two_types_of_uncertainty():
    prompt = contextual_prompts(contextual_instrument())[1]
    rendered = prompt.render()
    assert (
        "at exactly 0.5 stipulated index points (Zero is the rate reference) "
        "throughout the study year" in rendered
    )
    assert "not the maximum over a forecast's labs" in rendered
    assert "rate: uncertainty quantiles of the expected number of qualified units" in (
        rendered
    )
    # The two kinds of quantiles are defined once and named by every quantity.
    assert rendered.count("Uncertainty quantiles express") == 1
    assert rendered.count("Variation quantiles describe") == 1
    assert "loss: variation quantiles" in rendered
    assert "five-provider" not in rendered
    assert "before insurance and other recoveries" in rendered
    assert "a point value" in rendered


@pytest.mark.parametrize(
    ("fields", "valid"),
    [
        ({"status": "available", "amount": 0, "unavailable_reason": None}, True),
        (
            {
                "status": "available",
                "loss": {
                    key: value
                    for value, key in enumerate(
                        ("p5", "p25", "p50", "p75", "p95", "p99", "p999"), 1
                    )
                },
            },
            True,
        ),
        ({"status": "available", "share": 0.5}, True),
        ({"status": "unavailable", "unavailable_reason": "Unknown"}, True),
        (
            {
                "status": "unavailable",
                "unavailable_reason": "Unknown",
                "amount": None,
                "loss": None,
                "share": None,
            },
            True,
        ),
        (
            {"status": "available", "amount": 10, "unavailable_reason": "Explained"},
            False,
        ),
        ({"status": "available", "amount": None}, False),
        ({"status": "available", "amount": 10, "share": 0.5}, False),
        (
            {"status": "unavailable", "unavailable_reason": "Unknown", "amount": 0},
            False,
        ),
        ({"status": "unavailable", "unavailable_reason": None}, False),
    ],
)
def test_emitted_component_schema_matches_runtime_availability(fields, valid):
    rendered = contextual_prompts(contextual_instrument())[0].render()
    schema = answer_schema_of(rendered)
    schema["$ref"] = "#/$defs/ContextualComponentAnswer"
    schema = {key: schema[key] for key in ("$ref", "$defs")}
    Draft202012Validator.check_schema(schema)
    payload = {"component_id": "victim", **fields}
    assert Draft202012Validator(schema).is_valid(payload) == valid
    if valid:
        ContextualComponentAnswer.model_validate(payload)
    else:
        with pytest.raises(ValidationError):
            ContextualComponentAnswer.model_validate(payload)


@pytest.mark.parametrize(
    "change",
    ["round_reference", "wrong_deployment", "duplicate_anchor", "insufficient_hull"],
)
def test_context_rejects_identity_and_score_substitution(change):
    payload = contextual_instrument().model_dump(mode="json")
    if change == "round_reference":
        payload["reference_eci"] = 0.6
    elif change == "wrong_deployment":
        payload["deployment_binding"]["model_id"] = "other"
    elif change == "duplicate_anchor":
        payload["anchor_scores"] = [0, 0.5, 0.5, 2]
    else:
        payload["integer_max"] = 3
    with pytest.raises(ValidationError):
        ContextualRiskInstrument.model_validate(payload)


def test_unsupported_authored_node_is_retained_without_a_collection_call():
    payload = contextual_instrument().model_dump(mode="json")
    missing = copy.deepcopy(payload["nodes"][0])
    missing.update(
        node_id="missing",
        branches=[],
        components=[],
        count_family=None,
        unavailable_reason="Severity inputs have not been authored",
    )
    missing["population"].update(scenario_id="missing")
    payload["nodes"].append(missing)
    instrument = ContextualRiskInstrument.model_validate(payload)
    artifact = contextual_artifact(instrument)
    validate_contextual_artifact(artifact, instrument)
    assert len(instrument.nodes) == 2
    assert len(artifact.cases) == 3


def test_explicit_zero_and_unavailable_business_loss_are_distinct():
    prompt = contextual_prompts(contextual_instrument())[0]
    payload = contextual_answer().model_dump(mode="json")
    payload["lab_annual_business_loss"] = {
        "status": "available",
        "positive_probability": 0,
    }
    parsed = parse_contextual_answer(
        ContextualRiskAnswer.model_validate(payload).model_dump_json(), prompt
    )
    assert (
        parsed.lab_annual_business_loss is not None
        and parsed.lab_annual_business_loss.positive_probability == 0
    )
    payload["lab_annual_business_loss"] = {
        "status": "unavailable",
        "unavailable_reason": "Aggregate unquantified",
    }
    parsed = parse_contextual_answer(
        ContextualRiskAnswer.model_validate(payload).model_dump_json(), prompt
    )
    assert parsed.status == "available"
    assert parsed.lab_annual_business_loss is not None
    assert (
        parsed.lab_annual_business_loss.unavailable_reason == "Aggregate unquantified"
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing_component",
        "wrong_branch",
        "bad_probability",
        "fabricated_citation",
        "no_amount",
    ],
)
def test_parser_rejects_incomplete_or_unbound_judgments(change):
    import json

    prompt = contextual_prompts(contextual_instrument())[0]
    payload = contextual_answer().model_dump(mode="json")
    if change == "missing_component":
        payload["branches"][0]["components"].pop()
    elif change == "wrong_branch":
        payload["branches"][0]["branch_id"] = "new-story"
    elif change == "bad_probability":
        payload["branches"][0]["probability"] = 0.8
    elif change == "fabricated_citation":
        payload["reasoning"]["evidence_used"] = ["invented"]
    else:
        payload["branches"][0]["components"][0]["amount"] = None
    with pytest.raises(ValueError):
        parse_contextual_answer(json.dumps(payload), prompt)


def test_explicitly_unavailable_answer_is_retained():
    payload = contextual_answer().model_dump(mode="json")
    payload.update(
        status="unavailable",
        rate=None,
        lab_compensation=None,
        lab_annual_business_loss=None,
        branches=[],
        direct_eal=None,
        unavailable_reason="No defensible count estimate",
    )
    prompt = contextual_prompts(contextual_instrument())[0]
    parsed = parse_contextual_answer(
        ContextualRiskAnswer.model_validate(payload).model_dump_json(), prompt
    )
    assert parsed.unavailable_reason == "No defensible count estimate"


def test_prompt_hash_covers_context_and_artifact_requires_every_anchor():
    instrument = contextual_instrument()
    artifact = contextual_artifact(instrument)
    with pytest.raises(ValueError, match="every collectible"):
        validate_contextual_artifact(
            artifact.model_copy(update={"cases": artifact.cases[:-1]}), instrument
        )
    changed = instrument.model_copy(update={"fixed_adoption": "Changed adoption"})
    with pytest.raises(ValueError, match="instrument identity"):
        validate_contextual_artifact(artifact, changed)
    case = artifact.cases[0].model_copy(update={"prompt_sha256": "0" * 64})
    with pytest.raises(ValueError, match="frozen prompt"):
        validate_contextual_artifact(
            artifact.model_copy(update={"cases": (case, *artifact.cases[1:])}),
            instrument,
        )


def test_prompt_rejects_a_boolean_or_unplanned_coordinate():
    instrument = contextual_instrument()
    for score in (True, 1):
        with pytest.raises(ValueError, match="outside"):
            ContextualPrompt(
                instrument, instrument.nodes[0], instrument.respondents[0], score
            )


def test_scenario_adapter_builds_uncollected_composition_from_the_roster():
    from catmodel.elicitation.contextual_risk import contextual_node_from_scenario
    from catmodel.resources import scenarios_path
    from catmodel.scenarios.loader import load_scenarios

    scenario = next(
        row
        for row in load_scenarios(scenarios_path())
        if row.id == "influence_manipulation"
    )
    node = contextual_node_from_scenario(scenario, "deployment", "public")
    assert node.unavailable_reason is None
    assert node.population.deployment_ids == ("deployment",)
    assert [row.cell_id for row in scenario.estimand_composition.cells] == [
        row.component_id for row in node.components
    ]
    assert "collected composition" in node.dependence_assumption
    assert node.total is not None
    assert all(
        row.role == "partition" and row.rank_group_id is None for row in node.components
    )


def test_close_noninteger_anchors_have_distinct_call_identities():
    payload = contextual_instrument().model_dump(mode="json")
    payload["anchor_scores"] = [0, 0.5, 0.5000001, 0.5000002, 2]
    prompts = contextual_prompts(ContextualRiskInstrument.model_validate(payload))
    assert len({prompt.call_id for prompt in prompts}) == len(prompts)


@pytest.mark.parametrize("field", ["capability_domain_id", "capability_access"])
def test_domain_and_access_must_be_frozen_before_collection(field):
    payload = contextual_instrument().model_dump(mode="json")
    payload.pop(field)
    with pytest.raises(ValidationError):
        ContextualRiskInstrument.model_validate(payload)


def test_domain_and_access_change_the_exact_prompt():
    public = contextual_instrument()
    internal = public.model_copy(update={"capability_access": "internal"})
    changed_domain = public.model_copy(update={"capability_domain_id": "cyber"})
    first = contextual_prompts(public)[0]
    assert "Hold the lab's public capability (general domain)" in first.render()
    assert first.prompt_sha256() != contextual_prompts(internal)[0].prompt_sha256()
    assert (
        first.prompt_sha256() != contextual_prompts(changed_domain)[0].prompt_sha256()
    )


def test_aggregate_scope_is_independent_of_the_societal_component_roster():
    from catmodel.elicitation.contextual_risk import contextual_node_from_scenario
    from catmodel.resources import scenarios_path
    from catmodel.scenarios.loader import load_scenarios

    scenario = next(
        row
        for row in load_scenarios(scenarios_path())
        if row.id == "influence_manipulation"
    )
    node = contextual_node_from_scenario(scenario, "deployment", "public")
    assert "lab_costs_complete" not in node.model_dump()
    explicit = contextual_instrument()
    rendered = contextual_prompts(explicit)[0].render()
    assert "without a breakdown by cost" in rendered
    assert "Do not multiply a per-unit loss by the number of units" in rendered
    assert "however many units each has" in rendered
    assert (
        "the admission threshold decides which incidents count, not how small "
        "this amount may be, and the amount is not the societal loss times a "
        "conversion ratio" in rendered
    )
    assert "fitted" not in rendered
    assert explicit.lab_annual_business_loss_definition in rendered
    assert (
        "Broader franchise or enterprise-value effects remain unquantified"
        not in rendered
    )
    assert "lab_loss_fraction" not in rendered
    assert "lab_costs_complete" not in rendered
    assert explicit.lab_loss_scope == "episode_shares_and_annual_business_loss"
    assert explicit.lab_annual_business_loss_dependence == "conditional_independence"


def _positive_lab_quantiles():
    return dict(
        zip(
            ("p5", "p25", "p50", "p75", "p95", "p99", "p999"),
            (20, 40, 80, 120, 200, 600, 1000),
            strict=True,
        )
    )


def test_stochastic_aggregate_supplies_quantiles_and_code_binds_governed_shape():
    payload = contextual_answer().model_dump(mode="json")
    payload["lab_annual_business_loss"] = {
        "status": "available",
        "positive_probability": 0.4,
        "loss": _positive_lab_quantiles(),
    }
    instrument = contextual_instrument()
    prompt = contextual_prompts(instrument)[0]
    parsed = parse_contextual_answer(
        ContextualRiskAnswer.model_validate(payload).model_dump_json(), prompt
    )
    assert parsed.lab_annual_business_loss is not None
    assert (
        parsed.lab_annual_business_loss.loss is not None
        and parsed.lab_annual_business_loss.loss.p999 == 1000
    )
    assert parsed.branches == contextual_answer().branches
    rendered = prompt.render()
    # The governed shape is code's: the question neither shows nor asks it.
    assert "tail_shape" not in json.dumps(ContextualRiskAnswer.model_json_schema())
    assert "gpd_shape" not in rendered
    assert "shape" not in rendered
    assert instrument.nodes[0].lab_loss_gpd_shape == 0.25
    elicited_shape = {**payload["lab_annual_business_loss"], "tail_shape": 0.2}
    with pytest.raises(ValueError, match="tail_shape"):
        LabAnnualBusinessLossAnswer.model_validate(elicited_shape)


def test_node_without_governed_lab_shape_refuses_stochastic_lab_quantiles():
    payload = contextual_instrument().model_dump(mode="json")
    payload["nodes"][0]["lab_loss_gpd_shape"] = None
    instrument = ContextualRiskInstrument.model_validate(payload)
    prompt = contextual_prompts(instrument)[0]
    assert "Loss quantiles are not accepted for this scenario" in prompt.render()
    answer = contextual_answer().model_dump(mode="json")
    answer["lab_annual_business_loss"] = {
        "status": "available",
        "positive_probability": 1,
        "loss": _positive_lab_quantiles(),
    }
    with pytest.raises(ValueError, match="no governed lab business-loss tail shape"):
        parse_contextual_answer(
            ContextualRiskAnswer.model_validate(answer).model_dump_json(), prompt
        )


@pytest.mark.parametrize("shape", [0.95, 1.2, -0.96])
def test_governed_lab_shape_keeps_an_uncapped_finite_mean(shape):
    payload = contextual_instrument().model_dump(mode="json")
    payload["nodes"][0]["lab_loss_gpd_shape"] = shape
    with pytest.raises(ValueError, match="finite-mean"):
        ContextualRiskInstrument.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "available"},
        {"status": "available", "positive_probability": 0, "amount": 1},
        {"status": "available", "positive_probability": 0.5},
        {"status": "available", "positive_probability": 1, "amount": 0},
        {
            "status": "available",
            "positive_probability": 1,
            "amount": 1,
            "loss": _positive_lab_quantiles(),
        },
        {
            "status": "available",
            "positive_probability": 1,
            "loss": {**_positive_lab_quantiles(), "p5": 0},
        },
        {"status": "unavailable"},
        {
            "status": "unavailable",
            "unavailable_reason": "Unknown",
            "positive_probability": 0,
        },
    ],
)
def test_business_loss_requires_explicit_zero_positive_or_unavailable_law(payload):
    with pytest.raises(ValueError):
        LabAnnualBusinessLossAnswer.model_validate(payload)


def test_available_answers_require_their_marks_at_any_count():
    prompt = contextual_prompts(contextual_instrument())[0]
    zero = contextual_answer().model_dump(mode="json")
    zero["rate"] = dict.fromkeys(("p5", "p25", "p50", "p75", "p95"), 0)
    # Severity is conditional on a qualifying event, so a zero count keeps it.
    assert parse_contextual_answer(json.dumps(zero), prompt).branches
    for rate in (contextual_answer().model_dump(mode="json")["rate"], zero["rate"]):
        for field in ("lab_compensation", "lab_annual_business_loss"):
            missing = {**contextual_answer().model_dump(mode="json"), "rate": rate}
            missing.pop(field)
            with pytest.raises(ValueError, match="lab business-loss answers"):
                parse_contextual_answer(json.dumps(missing), prompt)
        payload = {**contextual_answer().model_dump(mode="json"), "rate": rate}
        with pytest.raises(
            ValidationError, match="branches is required when status is available"
        ):
            ContextualRiskAnswer.model_validate({**payload, "branches": []})
        # The direct EAL is asked at any count, beside the marks.
        with pytest.raises(
            ValidationError, match="direct_eal is required when status is available"
        ):
            ContextualRiskAnswer.model_validate({**payload, "direct_eal": None})


def test_unknown_component_and_node_fields_are_rejected():
    for field, value in (("lab_loss_fraction", 0.5), ("lab_loss_reason", "Unknown")):
        payload = contextual_answer().model_dump(mode="json")
        payload["branches"][0]["components"][0][field] = value
        with pytest.raises(ValueError, match="Extra inputs"):
            ContextualRiskAnswer.model_validate(payload)
    payload = contextual_instrument().model_dump(mode="json")
    payload["nodes"][0]["lab_costs_complete"] = True
    with pytest.raises(ValueError, match="Extra inputs"):
        ContextualRiskInstrument.model_validate(payload)
    payload = contextual_instrument().model_dump(mode="json")
    payload["schema_version"] = 2
    with pytest.raises(ValueError):
        ContextualRiskInstrument.model_validate(payload)


def test_high_level_definition_is_frozen_and_changes_prompt_identity():
    instrument = contextual_instrument()
    changed = instrument.model_copy(
        update={"lab_compensation_definition": "A different economic target"}
    )
    assert (
        contextual_prompts(instrument)[0].prompt_sha256()
        != contextual_prompts(changed)[0].prompt_sha256()
    )
    changed = instrument.model_copy(
        update={"lab_annual_business_loss_definition": "A different economic target"}
    )
    assert (
        contextual_prompts(instrument)[0].prompt_sha256()
        != contextual_prompts(changed)[0].prompt_sha256()
    )
    payload = instrument.model_dump(mode="json")
    payload["lab_annual_business_loss_dependence"] = "shared_severity_rank"
    with pytest.raises(ValueError):
        ContextualRiskInstrument.model_validate(payload)


@pytest.mark.parametrize("version", [0, 2])
def test_instruments_and_captures_of_another_schema_are_refused(version):
    instrument = contextual_instrument()
    stale = {**instrument.model_dump(mode="json"), "schema_version": version}
    stale["extrapolation"] = "unavailable"
    with pytest.raises(ValidationError, match=f"schema {version} is not the current 1"):
        ContextualRiskInstrument.model_validate(stale)
    capture = {**contextual_artifact(instrument).model_dump(mode="json")}
    capture["schema_version"] = version
    with pytest.raises(ValidationError, match=f"schema {version} is not the current 1"):
        ContextualRiskArtifact.model_validate(capture)


def test_a_saved_instrument_or_capture_must_state_its_schema():
    instrument = contextual_instrument()
    for contract, record in (
        (ContextualRiskInstrument, instrument),
        (ContextualRiskArtifact, contextual_artifact(instrument)),
    ):
        saved = json.loads(record.model_dump_json())
        assert saved["schema_version"] == 1
        assert contract.from_saved(saved) == record
        for payload in unstated_markers(saved).values():
            with pytest.raises(ValueError, match="schema"):
                contract.from_saved(payload)
    # An instrument built in memory, as from an authored context, may leave
    # the version out: it gets the current one and the same identity.
    authored = json.loads(instrument.model_dump_json())
    del authored["schema_version"]
    built = ContextualRiskInstrument.model_validate(authored)
    assert built == instrument and built.digest() == instrument.digest()


def test_respondents_asked_one_question_share_its_text():
    # A collection holds every respondent's question while it waits; the
    # text, the same for each of them, is kept once.
    instrument = governed_contextual_instrument(model_ids=("model-a", "model-b"))
    first, second = (
        prompt
        for prompt in contextual_prompts(instrument)
        if prompt.node == instrument.collectible_nodes[0]
        and prompt.anchor_score == instrument.anchor_scores[0]
    )
    assert first.respondent != second.respondent
    assert first.render() is second.render()
    assert first.prompt_sha256() == second.prompt_sha256()
