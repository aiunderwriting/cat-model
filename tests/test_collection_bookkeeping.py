"""Collection's bookkeeping keeps every frozen byte while doing less work.

Questions render and digest once, captures are indexed once, parser
provenance is computed once per parser and model, and every chain count law
replays from the answer that drives it. None of these may change a rendered
question, a stored request, a replayed answer or a capture.
"""

from functools import partial
from typing import cast

import pytest

from catmodel.calibration.respondent_questions import CalibrationQuestion
from catmodel.collection.contextual_collection import collect_contextual_risk
from catmodel.config import ExecutionSettings
from catmodel.contracts.contextual_risk import (
    ContextualRiskAnswer,
    ContextualRiskInstrument,
    ParentAnswer,
    json_schema,
)
from catmodel.elicitation.contextual_frequency import (
    ChainGuardrailRetryPrompt,
    ContextualChainPrompt,
    chain_count_law_for_answer,
    chain_guardrail_retry_prompt,
    driving_chain_answer,
)
from catmodel.elicitation.contextual_hierarchy import ContextualParentPrompt
from catmodel.elicitation.contextual_risk import (
    CONTEXTUAL_SYSTEM_PROMPT,
    ContextualPrompt,
    ContextualSeverityRetryPrompt,
    RenderedOnce,
    parse_contextual_answer,
)
from catmodel.elicitation.eal_crosscheck import (
    EalCrosscheckPrompt,
    EalCrosscheckRetryPrompt,
    case_crosscheck_prompt,
)
from catmodel.frozen_calls import (
    CaptureFiles,
    _parser_protocol,
    call_attempt_paths,
    frozen_call_key,
    parser_protocol,
    replay_call,
)
from catmodel.hashing import canonical_json_sha256
from catmodel.json_io import strict_json_loads
from catmodel.llm.transport import LLMProvider
from catmodel.setup.contextual_inputs import (
    contextual_request,
    read_contextual_capture_files,
    validate_contextual_capture,
)
from test_support.collection_trace import (
    CollectionTrace,
    LatencyProvider,
    ModelLatency,
    StudyResponder,
    TriggerRates,
    run_warped,
)
from test_support.contextual_risk import hierarchy_instrument

SCENARIOS = ("output_reliance_legal", "fraud_organizations", "enforcement_matter")
EVERY_TURN = TriggerRates(
    direct_retry=1.0,
    chain_guardrail_retry=1.0,
    crosscheck=1.0,
    crosscheck_retry=1.0,
    parent_gap=1.0,
)


def _collect(tmp_path, rates=EVERY_TURN):
    instrument = hierarchy_instrument(SCENARIOS)
    responder = StudyResponder([instrument], rates)
    provider = LatencyProvider(
        model_id="fixture-model",
        provider_name="openai",
        latency=ModelLatency(output_tokens=1000, tokens_per_second=100),
        responder=responder,
        trace=CollectionTrace(),
        max_concurrent=4,
    )
    run = tmp_path / "run"
    artifact, _ = run_warped(
        collect_contextual_risk(
            instrument=instrument,
            run_directory=run,
            providers=[cast(LLMProvider, provider)],
            config=ExecutionSettings(
                samples_per_question=1,
                max_output_tokens_by_model={provider.model_id: 100},
                request_timeout_seconds=7200,
                concurrency_by_model={provider.model_id: 4},
            ),
        )
    )
    return instrument, artifact, run


def test_questions_render_and_digest_once_without_changing_a_byte(tmp_path):
    instrument, artifact, run = _collect(tmp_path)
    files = read_contextual_capture_files(run)
    questions = []
    for case in artifact.cases:
        node = next(row for row in instrument.nodes if row.node_id == case.node_id)
        questions.append(
            ContextualPrompt(instrument, node, case.respondent, case.anchor_score)
        )
        questions.append(
            ContextualChainPrompt(instrument, node, case.respondent, case.anchor_score)
        )
        guardrail_question = chain_guardrail_retry_prompt(
            ContextualChainPrompt(instrument, node, case.respondent, case.anchor_score),
            case.chain_answer,
        )
        if guardrail_question is not None:
            questions.append(guardrail_question)
        for question in (case_crosscheck_prompt(instrument, case),):
            if question is not None:
                questions.append(question)
    questions.extend(
        ContextualParentPrompt(instrument, parent, respondent, score)
        for parent in instrument.parents
        for respondent in instrument.respondents
        for score in instrument.anchor_scores
    )
    kinds = {type(question).__name__ for question in questions}
    assert {
        "ContextualPrompt",
        "ContextualChainPrompt",
        "ChainGuardrailRetryPrompt",
        "EalCrosscheckPrompt",
        "ContextualParentPrompt",
    } <= kinds
    for question in questions:
        text = question.render()
        # The kept text is the text a fresh render gives, and is kept.
        assert text == question._render()
        assert question.render() is text
        assert question.prompt_sha256() == canonical_json_sha256(
            {"system_prompt": question.system_prompt, "user_prompt": text}
        )
        # The stored request is that text, byte for byte.
        request = contextual_request(question)
        stored = files[f"calls/{frozen_call_key(request)}/request.json"]
        assert strict_json_loads(stored) == request
    assert questions[0].system_prompt == CONTEXTUAL_SYSTEM_PROMPT
    # Keeping the text never changes a question's identity or equality.
    first = questions[0]
    assert isinstance(first, ContextualPrompt)
    twin = type(first)(
        first.instrument, first.node, first.respondent, first.anchor_score
    )
    assert twin == first
    validate_contextual_capture(instrument, artifact, files)


def test_an_indexed_capture_replays_exactly_as_a_plain_one(tmp_path):
    instrument, artifact, run = _collect(tmp_path)
    indexed = read_contextual_capture_files(run)
    assert isinstance(indexed, CaptureFiles)
    plain = dict(indexed)
    keys = {name.split("/")[1] for name in plain if name.startswith("calls/")}
    assert keys
    for key in keys:
        assert call_attempt_paths(indexed, key) == call_attempt_paths(plain, key)
    # Paths that only share a prefix are never another call's records.
    tricky = CaptureFiles(
        {
            "calls/ab/attempt-001.json": b"",
            "calls/abc/attempt-001.json": b"",
            "calls/ab/attempt-002.txt": b"",
            "calls/ab": b"",
        }
    )
    assert call_attempt_paths(tricky, "ab") == (
        "calls/ab/attempt-001.json",
        "calls/ab/attempt-002.txt",
    )
    assert call_attempt_paths(tricky, "ab") == call_attempt_paths(dict(tricky), "ab")
    case = artifact.cases[0]
    node = next(row for row in instrument.nodes if row.node_id == case.node_id)
    question = ContextualPrompt(instrument, node, case.respondent, case.anchor_score)
    request = contextual_request(question)
    parse = partial(parse_contextual_answer, prompt=question)
    assert (
        replay_call(request=request, files=indexed, parse_answer=parse).answer
        == replay_call(request=request, files=plain, parse_answer=parse).answer
    )
    validate_contextual_capture(instrument, artifact, plain)
    validate_contextual_capture(instrument, artifact, indexed)
    # A call outside the plan, or a stray file beside the calls, is refused.
    known = sorted(keys)[0]
    for stray in (f"calls/{known}x/attempt-001.json", "calls/stray"):
        with pytest.raises(ValueError, match="unplanned call"):
            validate_contextual_capture(
                instrument, artifact, CaptureFiles({**plain, stray: b"{}"})
            )


def test_parser_provenance_is_kept_per_parser_and_model():
    # Partials that bind different arguments to one parser.
    first = parser_protocol(
        partial(parse_contextual_answer, "{}"), ContextualRiskAnswer
    )
    second = parser_protocol(
        partial(parse_contextual_answer, "[]"), ContextualRiskAnswer
    )
    assert first is second
    fresh = _parser_protocol.__wrapped__(parse_contextual_answer, ContextualRiskAnswer)
    assert fresh == first


def test_chain_laws_replay_from_each_driving_answer(tmp_path):
    instrument, artifact, _ = _collect(tmp_path, TriggerRates())
    nodes = {node.node_id: node for node in instrument.collectible_nodes}
    for case in artifact.cases:
        assert case.chain_answer is not None
        chain = nodes[case.node_id].frequency_chain
        assert chain is not None
        law = chain_count_law_for_answer(
            driving_chain_answer(case), instrument.chain_exposure(chain).count
        )
        assert case.chain_count == law


def test_kept_digests_and_schemas_are_each_instruments_and_never_shared():
    instrument = hierarchy_instrument(SCENARIOS)
    payload = instrument.model_dump(mode="json")
    twin = ContextualRiskInstrument.model_validate(payload)
    payload["context_id"] = "another-lab"
    other = ContextualRiskInstrument.model_validate(payload)
    fresh = canonical_json_sha256(instrument.model_dump(mode="json"))
    assert instrument.digest() == instrument.digest() == twin.digest() == fresh
    assert other.digest() == canonical_json_sha256(other.model_dump(mode="json"))
    assert other.digest() != fresh
    first = next(row for row in instrument.parents if row.children)
    second = next(row for row in instrument.parents if not row.children)
    schemas = [
        ParentAnswer.json_schema_for(parent) for parent in (first, second, first)
    ]
    assert schemas[0] == schemas[2] != schemas[1]
    # Each schema is a copy: requiring one parent's remainder never reaches
    # the kept schema or another parent's.
    assert json_schema(ParentAnswer) == ParentAnswer.model_json_schema()
    assert json_schema(ParentAnswer) is not json_schema(ParentAnswer)


def test_every_question_class_keeps_its_docstring():
    """A class variable above a dataclass's docstring turns the docstring
    into a bare expression and its ``__doc__`` into the generated
    signature."""
    classes = (
        ContextualPrompt,
        ContextualSeverityRetryPrompt,
        ContextualChainPrompt,
        ChainGuardrailRetryPrompt,
        ContextualParentPrompt,
        EalCrosscheckPrompt,
        EalCrosscheckRetryPrompt,
        CalibrationQuestion,
    )
    assert len(classes) == 8
    assert all(issubclass(cls, RenderedOnce) for cls in classes)
    for cls in classes:
        doc = cls.__doc__ or ""
        assert doc.strip() and not doc.startswith(f"{cls.__name__}("), cls.__name__
