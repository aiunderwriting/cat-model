"""What respondents read: ordered schemas, each definition once, no internals.

Questions are rendered from a public-trajectory instrument built like the
hypothetical-lab study's: repository scenario nodes and every taxonomy parent
over them, a trajectory node carrying the forecast's method notes, digests
and weighting rationale, and evidence cards. Every kind of question is
rendered: the direct, chain and parent questions and each follow-up.
"""

import asyncio
import json
import re
from typing import cast

import pytest

from catmodel.collection.contextual_collection import (
    collect_contextual_risk,
    run_contextual_collection,
)
from catmodel.config import ExecutionSettings, ModelProfile
from catmodel.contracts.contextual_risk import (
    ContextualChainAnswer,
    ContextualRiskInstrument,
)
from catmodel.contracts.elicitation import ELICITATION_ORDER
from catmodel.contracts.public_frontier_condition import PublicTrajectoryCondition
from catmodel.contracts.respondent_calibration import CalibrationAnswer
from catmodel.elicitation.contextual_frequency import (
    ChainGuardrailRetryPrompt,
    ContextualChainPrompt,
    chain_guardrail_retry_prompt,
)
from catmodel.elicitation.contextual_hierarchy import ContextualParentPrompt
from catmodel.elicitation.contextual_risk import (
    ContextualPrompt,
    ContextualSeverityRetryPrompt,
    direct_severity_retry_prompt,
)
from catmodel.elicitation.eal_crosscheck import (
    EalCrosscheckPrompt,
    case_crosscheck_prompt,
)
from catmodel.elicitation.question_text import (
    ANSWER_SCHEMA_HEADING,
    ANSWERING_RULES,
    SOCIETAL_HARM_BASIS,
    entity_notice,
    qualification,
    respondent_schema,
    trajectory_text,
)
from catmodel.evidence.source_bundle import capture_source_bundle_snapshot
from catmodel.llm.models import ModelSpec
from catmodel.llm.transport import LLMProvider, ProviderResponse
from catmodel.scenarios.hierarchy_contract import admission_statement
from catmodel.setup.contextual_inputs import load_contextual_collection
from test_support.capability_sources import write_source_bundle
from test_support.collection_trace import StudyResponder, TriggerRates
from test_support.contextual_risk import (
    contextual_case,
    governed_contextual_answer,
    hierarchy_instrument,
    matched_chain_answer,
)
from test_support.question_text import answer_schema_of

LEAVES = ("output_reliance_health", "compromise_organizations")
LIMITATION = (
    "The study integrates the forecast over R by a fixed rule: Fejér nodes on "
    "a normal-score scale stretched by 1.5 over an equal mixture of the base law."
)
RATIONALE = (
    "The no-progress counterfactual. This node carries weight only under the "
    "no-progress forecast alternative."
)
CARD = {
    "card_id": "synthetic-source-1",
    "observation_through_utc": "2026-09-07",
    "availability_date_utc": "2026-09-07",
    "availability_basis": "Conservative availability bound for this curated "
    "synthetic excerpt; this is not a claimed original publication date.",
    "text": "Synthetic evidence excerpt for software tests only.",
    "sources": ["https://example.org/synthetic-source"],
}


def _public(instrument: ContextualRiskInstrument) -> ContextualRiskInstrument:
    """``instrument`` on one public trajectory node, as a public study builds it."""
    payload = instrument.model_dump(mode="json")
    payload.update(
        context_id="node-1-2-not_yet_public",
        coordinate_kind="public_trajectory",
        reference_eci=0,
        anchor_scores=[0],
        integer_min=0,
        integer_max=0,
        capability_access="internal",
    )
    payload["capability_scale"]["scale_id"] = "epoch-eci-current-fit"
    times = [f"2027-{month:02d}-01T00:00:00Z" for month in range(1, 13)]
    payload["public_trajectory"] = {
        "trajectory_sha256": "3e" * 32,
        "draw_id": "node-1-2",
        "observed_at": "2026-10-29T00:00:00Z",
        "observed_values": [168.9, 158.3],
        "earlier_frontier": {"at": "2025-10-29T00:00:00Z", "value": 150.7},
        "scale_anchors": [
            {"model": "Model One", "organization": "Lab A", "value": 130.0},
            {"model": "Model Two", "organization": "Lab B", "value": 150.0},
        ],
        "times": [*times, "2028-01-01T00:00:00Z"],
        "coordinates": [
            {
                "coordinate_id": f"public_{lab}",
                "lab_id": lab,
                "domain_id": "general",
                "access": "public",
                "scale_id": "epoch-eci-current-fit",
                "definition": f"Publicly available general ECI frontier for {lab}.",
            }
            for lab in ("lab_a", "lab_b")
        ],
        "values": [[170.25 + month, 160.5 + month] for month in range(13)],
        "interpolation": "linear",
        "target_lab": "public_frontier",
        "source_model_status": "not_yet_public",
        "source_references": [
            "https://epoch.ai/benchmarks/eci",
            "evidence_file:evidence/epoch_eci_scores.csv#sha256=" + "1d" * 32,
        ],
        "rationale": RATIONALE,
        "limitations": [LIMITATION, "Elapsed time is measured in 365.25-day years."],
    }
    for node in payload["nodes"]:
        node["source_cards"] = [CARD]
    return ContextualRiskInstrument.model_validate(payload)


@pytest.fixture(scope="module")
def instrument() -> ContextualRiskInstrument:
    return _public(hierarchy_instrument(LEAVES))


def _questions(instrument: ContextualRiskInstrument) -> dict[str, str]:
    """One rendered question of every kind."""
    node = next(row for row in instrument.nodes if row.node_id == LEAVES[1])
    respondent, score = instrument.respondents[0], instrument.anchor_scores[0]
    direct = ContextualPrompt(instrument, node, respondent, score)
    contradicting = governed_contextual_answer(node).model_copy(
        update={
            "large_loss_count": governed_contextual_answer(node).rate,
            "very_large_loss_count": None,
        }
    )
    # One of these shares contradicts the stated loss quantiles' tail band.
    retry = direct_severity_retry_prompt(
        direct, contradicting, 1e-6
    ) or direct_severity_retry_prompt(direct, contradicting, 0.5)
    assert isinstance(retry, ContextualSeverityRetryPrompt)
    chain = ContextualChainPrompt(instrument, node, respondent, score)
    answer = governed_contextual_answer(node)
    assert answer.rate is not None
    rare = matched_chain_answer(node, answer, chain.exposure.count).model_dump()
    rare["stages"][-1]["probability"] = dict.fromkeys(ELICITATION_ORDER, 1e-4)
    guardrail = chain_guardrail_retry_prompt(
        chain, ContextualChainAnswer.model_validate(rare)
    )
    assert isinstance(guardrail, ChainGuardrailRetryPrompt)
    distant = answer.model_copy(update={"direct_eal": answer.rate})
    case = contextual_case(instrument, direct, distant)
    crosscheck = case_crosscheck_prompt(instrument, case)
    assert isinstance(crosscheck, EalCrosscheckPrompt)
    parent = next(row for row in instrument.parents if LEAVES[1] in row.children)
    return {
        "direct": direct.render(),
        "direct_retry": retry.render(),
        "chain": chain.render(),
        "chain_retry": guardrail.render(),
        "crosscheck": crosscheck.render(),
        "parent": ContextualParentPrompt(
            instrument, parent, respondent, score
        ).render(),
    }


@pytest.fixture(scope="module")
def questions(instrument) -> dict[str, str]:
    return _questions(instrument)


def _ordered_after(names, declared):
    """Whether ``names`` keep their order in ``declared``."""
    positions = [declared.index(name) for name in names]
    return positions == sorted(positions)


def test_answer_schemas_list_reasoning_first_and_quantiles_median_first(questions):
    for kind, text in questions.items():
        schema = answer_schema_of(text)
        assert next(iter(schema["properties"])) == "reasoning", kind
        definitions = schema["$defs"]
        assert list(definitions["Reasoning"]["properties"])[:3] == [
            "median_reason",
            "low_case",
            "high_case",
        ]
        for name, levels in (
            ("Quantiles", ELICITATION_ORDER),
            ("SeverityQuantiles", (*ELICITATION_ORDER, "p99", "p999")),
            ("ProbabilityQuantiles", ELICITATION_ORDER),
        ):
            if name in definitions:
                assert tuple(definitions[name]["properties"]) == levels, (kind, name)
                assert tuple(definitions[name]["required"]) == levels, (kind, name)
        # Every object's status variants list their fields in its own order.
        for part in (schema, *definitions.values()):
            declared = list(part.get("properties", ()))
            for variant in part.get("anyOf", ()):
                assert _ordered_after(list(variant["properties"]), declared), kind
                assert _ordered_after(variant["required"], declared), kind
        # The schema is the last section and shows no annotations.
        assert text.rindex(ANSWER_SCHEMA_HEADING) > text.rindex("## Questions")
        shown = json.dumps(schema)
        for annotation in ('"title"', '"description"', '"default"'):
            assert annotation not in shown, (kind, annotation)
    # The direct question asks the count, then the loss of one unit, the
    # large-loss counts, the components, the lab's loss and the direct EAL,
    # in its text as in its schema.
    direct = questions["direct"]
    order = [
        "status",
        "rate",
        "total_loss",
        "large_loss_count",
        "branches",
        "lab_compensation",
        "lab_annual_business_loss",
        "direct_eal",
    ]
    assert _ordered_after(order, list(answer_schema_of(direct)["properties"]))
    asked = direct.split("## Questions\n", 1)[1]
    positions = [asked.index(f"\n\n{name}") for name in order[1:]]
    assert positions == sorted(positions)


def _sentences(text: str) -> list[str]:
    """The prose a respondent reads, without the evidence excerpts (authored
    data that may quote a source twice) and the answer schema, as sentences."""
    text = text.split(ANSWER_SCHEMA_HEADING, 1)[0]
    text = re.sub(r"## Evidence\n.*?(?=\n## )", "", text, flags=re.DOTALL)
    return [
        part.strip()
        for part in re.split(r"(?<=[.:;])\s+|\n", text)
        if len(part.strip()) >= 40
    ]


JSON_SCHEMA_KEYWORDS = frozenset(
    {
        # Core
        "$schema",
        "$id",
        "$ref",
        "$defs",
        "$anchor",
        "$dynamicRef",
        "$dynamicAnchor",
        "$vocabulary",
        "$comment",
        # Applicators
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "dependentSchemas",
        "prefixItems",
        "items",
        "contains",
        "properties",
        "patternProperties",
        "additionalProperties",
        "propertyNames",
        "unevaluatedItems",
        "unevaluatedProperties",
        # Validation
        "type",
        "enum",
        "const",
        "multipleOf",
        "maximum",
        "exclusiveMaximum",
        "minimum",
        "exclusiveMinimum",
        "maxLength",
        "minLength",
        "pattern",
        "maxItems",
        "minItems",
        "uniqueItems",
        "maxContains",
        "minContains",
        "maxProperties",
        "minProperties",
        "required",
        "dependentRequired",
        # Metadata, format and content
        "title",
        "description",
        "default",
        "deprecated",
        "readOnly",
        "writeOnly",
        "examples",
        "format",
        "contentEncoding",
        "contentMediaType",
        "contentSchema",
    }
)
"""The keywords of JSON Schema draft 2020-12."""

_SCHEMA_MAPS = ("properties", "$defs", "patternProperties", "dependentSchemas")
_SCHEMA_LISTS = ("allOf", "anyOf", "oneOf", "prefixItems")
_SCHEMA_VALUES = (
    "items",
    "not",
    "if",
    "then",
    "else",
    "contains",
    "additionalProperties",
    "propertyNames",
    "unevaluatedItems",
    "unevaluatedProperties",
    "contentSchema",
)


def _keywords(schema: object) -> set[str]:
    """The keywords ``schema`` and its subschemas use: the values of
    ``properties`` and ``$defs`` are schemas, not keywords."""
    if not isinstance(schema, dict):
        return set()
    found = set(schema)
    for key, value in schema.items():
        if key in _SCHEMA_MAPS:
            for subschema in value.values():
                found |= _keywords(subschema)
        elif key in _SCHEMA_LISTS:
            for subschema in value:
                found |= _keywords(subschema)
        elif key in _SCHEMA_VALUES:
            found |= _keywords(value)
    return found


def test_answer_schemas_use_only_standard_json_schema_keywords(questions):
    shown = {kind: answer_schema_of(text) for kind, text in questions.items()} | {
        "calibration": respondent_schema(CalibrationAnswer.model_json_schema())
    }
    for kind, schema in shown.items():
        unknown = _keywords(schema) - JSON_SCHEMA_KEYWORDS
        assert not unknown, (kind, unknown)


def test_no_definition_is_repeated_within_a_question(instrument, questions):
    node = next(row for row in instrument.nodes if row.node_id == LEAVES[1])
    terms = qualification(node)
    for kind, text in questions.items():
        sentences = _sentences(text)
        repeated = {part for part in sentences if sentences.count(part) > 1}
        assert not repeated, (kind, repeated)
        # Shared definitions appear once, and no JSON is nested in a string.
        for shared in (ANSWERING_RULES, entity_notice(instrument)):
            assert text.count(shared) == 1, kind
        assert '\\"' not in text.split(ANSWER_SCHEMA_HEADING, 1)[0], kind
        assert "{" not in text.split(ANSWER_SCHEMA_HEADING, 1)[0], kind
    for kind in ("direct", "chain", "crosscheck"):
        text = questions[kind]
        for field in ("terminal_event_unit", "counting_identity", "year_rule"):
            assert text.count(terms[field]) == 1, (kind, field)
        assert text.count(terms["loss_definition"]) == 1, kind
        # The chain's last stage is the qualified unit, not its restatement.
        assert text.count(node.population.terminal_unit) == 1, kind
    for kind in ("direct", "crosscheck", "parent"):
        assert questions[kind].count(SOCIETAL_HARM_BASIS) == 1, kind
    direct = questions["direct"]
    for definition in (
        instrument.lab_compensation_definition,
        instrument.lab_annual_business_loss_definition,
        instrument.exposure,
        instrument.lab.definition,
    ):
        assert direct.count(definition) == 1
    # Sentences that several components state verbatim are stated once.
    for component in node.components:
        assert direct.count(f"- {component.component_id} (") == 1
    assert questions["parent"].count(admission_statement()) == 1


INTERNAL_TERMS = (
    "sha256",
    "Fej",
    "not shown",
    "quadrature",
    "carries weight",
    "GPD",
    "gpd",
    "governed",
    "rank group",
    "rank_group",
    "joint mark",
    "kernel",
    "conditional_independence",
    "diagnostic",
    "manifest_sha256",
    "source bundle",
    "draw_id",
    "code applies",
    "code sets",
    "Code multiplies",
    "is simulated",
    "pooled",
    "frozen",
)


def test_questions_show_no_internals(instrument, questions):
    condition = instrument.public_trajectory
    assert condition is not None
    for kind, text in questions.items():
        prose = text.split(ANSWER_SCHEMA_HEADING, 1)[0]
        assert not re.search(r"\b[0-9a-f]{64}\b", text), kind
        assert "3e3e3e3e" not in text and "1d1d1d1d" not in text, kind
        for term in INTERNAL_TERMS:
            assert term not in prose, (kind, term)
        for hidden in (
            instrument.context_id,
            instrument.source_bundle_manifest_sha256,
            condition.rationale,
            *condition.limitations,
            CARD["availability_basis"],
        ):
            assert hidden not in text, (kind, hidden)
        # The trajectory is a dated table; its source is cited by URL only.
        assert "date        lab_a  lab_b  frontier" in text, kind
        assert "2027-01-01  170.2  160.5  170.2" in text, kind
        # The scale has a reference: the text names the models whose fixed
        # scores define it, the observed scores open the table, and the text
        # gives the frontier a year before them.
        assert (
            "Its publisher fixes the scale by placing Model One (Lab A) at 130.0 "
            "and Model Two (Lab B) at 150.0."
        ) in text, kind
        assert (
            "date        lab_a  lab_b  frontier\n"
            "2026-10-29  168.9  158.3  168.9\n"
            "2027-01-01  170.2  160.5  170.2\n"
        ) in text, kind
        assert (
            "The first row, 2026-10-29, is observed, not assumed: each lab's "
            "score on that date. On 2025-10-29, one year before it, the highest "
            "of these labs' scores was 150.7."
        ) in text, kind
        assert "https://epoch.ai/benchmarks/eci" in text, kind
        assert "evidence_file" not in text, kind


def _trajectory(instrument, **changes) -> str:
    condition = instrument.public_trajectory
    assert condition is not None
    return trajectory_text(
        PublicTrajectoryCondition.model_validate(
            condition.model_dump(mode="json") | changes
        )
    )


def test_the_trajectory_gives_only_the_scale_references_it_has(instrument):
    """Each reference is optional. A forecast may record no anchor models,
    its labs may have had no release a year earlier, and a path that starts
    on the as-of date already opens with the observed scores."""
    bare = _trajectory(instrument, scale_anchors=[], earlier_frontier=None)
    assert "fixes the scale" not in bare and "one year before" not in bare
    assert (
        "The first row, 2026-10-29, is observed, not assumed: each lab's score "
        "on that date. Each lab's value changes linearly"
    ) in bare
    same_day = _trajectory(
        instrument,
        observed_at="2027-01-01T00:00:00Z",
        observed_values=[170.25, 160.5],
    )
    assert "The first row, 2027-01-01, is observed, not assumed" in same_day
    assert (
        "date        lab_a  lab_b  frontier\n2027-01-01  170.2  160.5  170.2\n"
    ) in same_day
    assert "2026-10-29" not in same_day


CORRESPONDENCE = {
    "measure": "the other measure, which a gloss describes",
    "points_per_doubling": 5.0,
    "observed": "when the index was introduced",
}


def test_the_trajectory_says_what_a_point_means_where_its_source_does(instrument):
    """With the publisher's correspondence, the text quotes it and gives the
    doublings the target's rise would be, from the scores the table shows."""
    text = _trajectory(instrument, scale_correspondence=CORRESPONDENCE)
    # The frontier rises from 168.9 in the first row to 182.2 in the last.
    assert "2026-10-29  168.9  158.3  168.9\n" in text
    assert "2028-01-01  182.2  172.5  182.2\n" in text
    assert (
        "(column frontier). The index's publisher reports that, when the index "
        "was introduced, a gain of 5 points appeared to correspond roughly to a "
        "doubling of the other measure, which a gloss describes. If that held "
        "along this path, the frontier's rise from the first row to the last, "
        "13.3 points, would be 2.7 such doublings. Whether it holds that far is "
        "uncertain: include that uncertainty in your answers.\ndate "
    ) in text
    # A path that does not rise states the correspondence and no doublings.
    flat = _trajectory(
        instrument,
        scale_correspondence=CORRESPONDENCE,
        times=["2027-01-01T00:00:00Z", "2028-01-01T00:00:00Z"],
        values=[[168.9, 158.3], [168.9, 158.3]],
    )
    assert "a doubling of the other measure, which a gloss describes.\ndate " in flat
    assert "such doublings" not in flat and "If that held" not in flat


def test_a_trajectory_without_a_correspondence_is_the_one_frozen_before_it(
    instrument,
):
    """A condition that records no correspondence leaves the field out of its
    record and its text, so an instrument frozen before the field existed
    keeps its digest and its questions' prompts."""
    condition = instrument.public_trajectory
    assert condition is not None and condition.scale_correspondence is None
    record = condition.model_dump(mode="json")
    assert "scale_correspondence" not in record
    assert "scale_correspondence" not in json.dumps(instrument.model_dump(mode="json"))
    assert PublicTrajectoryCondition.model_validate(record) == condition
    text = trajectory_text(condition)
    assert "publisher reports" not in text and "doubling" not in text
    stated = PublicTrajectoryCondition.model_validate(
        record | {"scale_correspondence": CORRESPONDENCE}
    )
    assert stated.model_dump(mode="json")["scale_correspondence"] == CORRESPONDENCE
    changed = instrument.model_copy(update={"public_trajectory": stated})
    assert changed.digest() != instrument.digest()
    # Only the two sentences differ.
    with_it = trajectory_text(stated)
    start = with_it.index(" The index's publisher reports that")
    end = with_it.index("\ndate ")
    assert with_it[:start] + with_it[end:] == text


def test_a_lead_change_within_the_year_is_given_as_a_date(instrument):
    """Two labs' straight paths cross at an instant between the dates shown;
    the text names its day and the frontier's score there."""
    text = _trajectory(
        instrument,
        times=["2027-01-01T00:00:00Z", "2028-01-01T00:00:00Z"],
        values=[[170.0, 169.0], [171.0, 173.0]],
        observed_values=[168.9, 158.3],
    )
    # The paths meet a third of the way through the year, at 170 1/3.
    assert (
        "The lead changes between the dates shown, on 2027-05-02 (frontier 170.3)."
    ) in text
    # One lab holds while the other rises, then the reverse: two changes.
    twice = _trajectory(
        instrument,
        times=[
            "2027-01-01T00:00:00Z",
            "2027-07-01T00:00:00Z",
            "2028-01-01T00:00:00Z",
        ],
        values=[[170.0, 169.0], [170.0, 171.0], [174.0, 171.0]],
        observed_values=[168.9, 158.3],
    )
    assert (
        "The lead changes between the dates shown, on 2027-04-01 (frontier 170.0) "
        "and 2027-08-16 (frontier 171.0)."
    ) in twice


def test_question_lengths_stay_within_budget(questions):
    """A regression budget on one representative node, a few percent above its
    rendered lengths."""
    budget = {
        "direct": 30_900,
        "direct_retry": 31_800,
        "chain": 19_000,
        "chain_retry": 19_300,
        "crosscheck": 27_600,
        "parent": 14_100,
    }
    lengths = {kind: len(text) for kind, text in questions.items()}
    over = {kind: n for kind, n in lengths.items() if n > budget[kind]}
    assert not over, (over, lengths)


class _Provider:
    """Answers every question of the instrument through the study responder,
    taking every conditional turn: each direct answer contradicts its
    large-loss count, each chain trips the guardrail and each direct EAL
    triggers the cross-check. Each parent's answer lies ten times above what
    its leaves imply."""

    model_id = "fixture-model"
    provider_name = "openai"
    reasoning_effort = "low"

    def __init__(self, instrument):
        self.responder = StudyResponder(
            [instrument],
            TriggerRates(
                direct_retry=1.0,
                chain_guardrail_retry=1.0,
                crosscheck=1.0,
                crosscheck_retry=0.0,
                parent_gap=1.0,
            ),
        )

    async def complete(self, *, system_prompt, user_prompt, **_kwargs):
        return ProviderResponse(
            text=self.responder(system_prompt, user_prompt),
            model_id=self.model_id,
            provider_name=self.provider_name,
        )


def test_a_fake_collection_with_every_follow_up_replays_offline(tmp_path, monkeypatch):
    """At three fixed capability anchors, whose source bundle holds every
    input the questions show."""
    import catmodel.config

    source = write_source_bundle(tmp_path / "source")
    payload = hierarchy_instrument(LEAVES).model_dump(mode="json")
    payload["source_bundle_manifest_sha256"] = capture_source_bundle_snapshot(
        source
    ).manifest_sha256
    bound = ContextualRiskInstrument.model_validate(payload)
    monkeypatch.setattr(
        catmodel.config,
        "load_model_profile",
        lambda **kwargs: ModelProfile(
            models=(
                ModelSpec(
                    provider="openai",
                    model_id="fixture-model",
                    reasoning_effort="low",
                    max_output_tokens=100_000,
                ),
            )
        ),
    )
    run = tmp_path / "run"
    run_contextual_collection(
        instrument=bound,
        source_bundle=source,
        run_directory=run,
        profile="test",
        dry_run=True,
    )
    provider = _Provider(bound)
    artifact = asyncio.run(
        collect_contextual_risk(
            instrument=bound,
            run_directory=run,
            providers=[cast(LLMProvider, provider)],
            config=ExecutionSettings(
                samples_per_question=1,
                max_output_tokens_by_model={provider.model_id: 100},
                request_timeout_seconds=60,
                concurrency_by_model={provider.model_id: 4},
            ),
        )
    )
    # Every kind of follow-up was asked and answered.
    assert all(case.severity_retry is not None for case in artifact.cases)
    assert all(case.chain_guardrail is not None for case in artifact.cases)
    assert all(
        case.eal_crosscheck is not None
        and case.eal_crosscheck.reconciliation is not None
        for case in artifact.cases
    )
    # Every parent was asked once, and its pool recorded with no further call.
    assert all(row.pool is not None for row in artifact.parent_cases)
    assert any(
        row.pool is not None and row.pool.factor > 1.5 for row in artifact.parent_cases
    )
    # Raw replay re-renders every question from the instrument and stored
    # answers and re-parses every stored response.
    (run / "contextual-risk.json").write_text(artifact.model_dump_json())
    loaded, replayed, _ = load_contextual_collection(run, source)
    assert loaded == bound and replayed == artifact
