"""Collection and capture replay ask the same questions, each read one way.

``catmodel.elicitation.question_flow`` lists every kind of question and holds
each leaf path's sequence; collection and capture validation both drive it.
"""

from collections import Counter
from types import FunctionType
from typing import cast

import pytest

from catmodel.collection import contextual_collection
from catmodel.collection.contextual_collection import collect_contextual_risk
from catmodel.config import ExecutionSettings
from catmodel.contracts.contextual_risk import EXHAUSTED_CALL_KINDS
from catmodel.elicitation.contextual_frequency import ContextualChainPrompt
from catmodel.elicitation.contextual_risk import ContextualPrompt
from catmodel.elicitation.eal_crosscheck import case_crosscheck_prompt
from catmodel.elicitation.question_flow import (
    DIRECT_ANSWERED,
    QUESTION_KINDS,
    chain_path,
    crosscheck_path,
    direct_path,
)
from catmodel.llm.transport import LLMProvider
from catmodel.setup import contextual_inputs
from catmodel.setup.contextual_inputs import (
    REPLAY_ERRORS,
    read_contextual_capture_files,
    recorded_answers,
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

QUESTIONS = {
    # kind: question class, parser, response model, turns left on its path
    "direct": (
        "ContextualPrompt",
        "parse_contextual_answer",
        "ContextualRiskAnswer",
        4,
    ),
    "direct_severity_retry": (
        "ContextualSeverityRetryPrompt",
        "parse_severity_retry_answer",
        "ContextualRiskAnswer",
        3,
    ),
    "eal_crosscheck": (
        "EalCrosscheckPrompt",
        "parse_eal_crosscheck_answer",
        "EalCrosscheckAnswer",
        2,
    ),
    "eal_crosscheck_severity_retry": (
        "EalCrosscheckRetryPrompt",
        "parse_eal_crosscheck_retry_answer",
        "EalCrosscheckAnswer",
        1,
    ),
    "chain": (
        "ContextualChainPrompt",
        "parse_chain_answer",
        "ContextualChainAnswer",
        2,
    ),
    "chain_guardrail_retry": (
        "ChainGuardrailRetryPrompt",
        "parse_chain_guardrail_retry_answer",
        "ContextualChainAnswer",
        1,
    ),
    "parent": ("ContextualParentPrompt", "parse_parent_answer", "ParentAnswer", 1),
}
"""Every kind of question, written out here and not read from the code."""

LEAF_KINDS = {name for name in QUESTIONS if not name.startswith("parent")}
SCENARIOS = ("output_reliance_legal", "fraud_organizations")
EVERY_TURN = TriggerRates(
    direct_retry=1.0,
    chain_guardrail_retry=1.0,
    crosscheck=1.0,
    crosscheck_retry=1.0,
    parent_gap=1.0,
)


def test_the_roster_lists_each_kind_once_with_its_parser_model_and_turns():
    assert {
        kind.name: (
            cast(FunctionType, kind.parser).__name__,
            kind.response_model.__name__,
            kind.turns,
        )
        for kind in QUESTION_KINDS
    } == {name: row[1:] for name, row in QUESTIONS.items()}
    assert len(QUESTION_KINDS) == len(QUESTIONS)
    # An artifact names an exhausted call by the same kinds.
    assert sorted(EXHAUSTED_CALL_KINDS) == sorted(QUESTIONS)
    # Replay has a finding for every kind, and for nothing else.
    assert set(REPLAY_ERRORS) == set(QUESTION_KINDS)
    assert len(set(REPLAY_ERRORS.values())) == len(QUESTIONS)


def test_a_taxonomy_parent_has_one_question_and_no_follow_up(collection):
    instrument, artifact, _, collected, _ = collection
    asked = [row for row in collected if row[0] == "ContextualParentPrompt"]
    # One call per parent, coordinate and respondent, although every parent's
    # answer lies ten times above what its leaves imply.
    assert len(asked) == len(artifact.parent_cases)
    assert len(asked) == len(instrument.parents) * len(instrument.anchor_scores)
    # Each parent case stores the pool the fixed rule gives.
    pools = [case.pool for case in artifact.parent_cases]
    assert all(pool is not None for pool in pools)
    assert any(pool is not None and pool.factor > 1.5 for pool in pools)


@pytest.fixture(scope="module")
def collection(tmp_path_factory):
    """One collection in which every question takes every follow-up, with
    each call collection made and each call replay read."""
    instrument = hierarchy_instrument(SCENARIOS)
    provider = LatencyProvider(
        model_id="fixture-model",
        provider_name="openai",
        latency=ModelLatency(output_tokens=1000, tokens_per_second=100),
        responder=StudyResponder([instrument], EVERY_TURN),
        trace=CollectionTrace(),
        max_concurrent=4,
    )
    run = tmp_path_factory.mktemp("question-flow") / "run"
    collected: list[tuple[str, str, str]] = []
    replayed: list[tuple[str, str]] = []
    collect, replay = (
        contextual_collection.collect_frozen_call,
        contextual_inputs.replay_call,
    )

    async def collecting(**kwargs):
        parse = kwargs["parse_answer"]
        collected.append(
            (
                type(parse.keywords["prompt"]).__name__,
                parse.func.__name__,
                kwargs["response_model"].__name__,
            )
        )
        return await collect(**kwargs)

    def replaying(**kwargs):
        parse = kwargs["parse_answer"]
        replayed.append((type(parse.keywords["prompt"]).__name__, parse.func.__name__))
        return replay(**kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(contextual_collection, "collect_frozen_call", collecting)
        patch.setattr(contextual_inputs, "replay_call", replaying)
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
        validate_contextual_capture(
            instrument, artifact, read_contextual_capture_files(run)
        )
    return instrument, artifact, run, collected, replayed


def test_collection_asks_and_replay_reads_every_kind_the_same_way(collection):
    _, _, _, collected, replayed = collection
    # Every kind is asked with its parser and response model, and no other.
    assert set(collected) == {row[:3] for row in QUESTIONS.values()}
    # Replay reads each collected call once, with the parser that collected it.
    assert Counter(replayed) == Counter(row[:2] for row in collected)


def test_replay_fails_a_capture_whose_rows_miss_a_kind_collection_asked(
    collection, monkeypatch
):
    instrument, artifact, run, _, _ = collection
    files = read_contextual_capture_files(run)
    by_name = {kind.name: kind for kind in QUESTION_KINDS}
    assert all(
        {kind.name for kind in recorded_answers(case)} == LEAF_KINDS
        for case in artifact.cases
    )
    for name in ("direct_severity_retry", "chain_guardrail_retry"):
        dropped = by_name[name]
        monkeypatch.setattr(
            contextual_inputs,
            "recorded_answers",
            lambda row, dropped=dropped: {
                kind: answer
                for kind, answer in recorded_answers(row).items()
                if kind is not dropped
            },
        )
        with pytest.raises(ValueError, match=REPLAY_ERRORS[dropped]):
            validate_contextual_capture(instrument, artifact, files)


def _kinds(path, replies):
    """The questions a path asks when sent ``replies`` in turn, and what it
    returns."""
    asked, reply = [], None
    replies = iter(replies)
    while True:
        try:
            step = path.send(reply)
        except StopIteration as finished:
            return asked, finished.value
        asked.append("direct_answered" if step is DIRECT_ANSWERED else step.kind.name)
        reply = next(replies)


def test_each_leaf_path_asks_its_follow_up_only_when_its_answer_calls_for_it(
    collection,
):
    instrument, artifact, _, _, _ = collection
    case = artifact.cases[0]
    node = next(row for row in instrument.nodes if row.node_id == case.node_id)
    place = (instrument, node, case.respondent, case.anchor_score)
    direct, chain = ContextualPrompt(*place), ContextualChainPrompt(*place)
    assert case.severity_retry is not None and case.chain_guardrail is not None

    # The direct answer contradicts itself, so its retry follows; what the
    # path returns is what the case stores.
    asked, stored = _kinds(
        direct_path(direct), [case.answer, case.severity_retry.answer]
    )
    assert asked == ["direct", "direct_severity_retry"]
    assert stored == (case.answer, case.large_loss_share, case.severity_retry)
    # An exhausted direct question leaves the leaf unanswered and asks no more.
    assert _kinds(direct_path(direct), [None]) == (["direct"], None)
    # Replay words the retry from the stored share: a case that stores none
    # was not retried, whatever its answer's own share.
    shareless = case.model_copy(update={"large_loss_share": None})
    asked, stored = _kinds(direct_path(direct, shareless), [case.answer])
    assert asked == ["direct"] and stored == (case.answer, None, None)

    # The chain answer trips a guardrail: the retry waits for the direct
    # question and is asked only when that has an answer.
    retry = case.chain_guardrail
    assert _kinds(chain_path(chain), [case.chain_answer, True, retry.answer]) == (
        ["chain", "direct_answered", "chain_guardrail_retry"],
        (case.chain_answer, retry),
    )
    assert _kinds(chain_path(chain), [case.chain_answer, False]) == (
        ["chain", "direct_answered"],
        (case.chain_answer, None),
    )
    assert _kinds(chain_path(chain), [None, True]) == (
        ["chain", "direct_answered"],
        (None, None),
    )

    # The cross-check revises into a contradiction, so its retry follows; an
    # exhausted cross-check asks none.
    question = case_crosscheck_prompt(instrument, case)
    assert question is not None and case.eal_crosscheck is not None
    reconciliation = case.eal_crosscheck.reconciliation
    assert reconciliation is not None and reconciliation.severity_retry is not None
    second = reconciliation.severity_retry
    assert _kinds(
        crosscheck_path(question), [reconciliation.answer, second.answer]
    ) == (
        ["eal_crosscheck", "eal_crosscheck_severity_retry"],
        (
            reconciliation.answer,
            (second.prompt_sha256, second.contradiction, second.answer),
        ),
    )
    assert _kinds(crosscheck_path(question), [None]) == (
        ["eal_crosscheck"],
        (None, None),
    )
