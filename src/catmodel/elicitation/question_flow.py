"""The question flow of a contextual collection.

Every kind of question a collection asks is listed once (``QUESTION_KINDS``),
with its parser, its response model and the turns its path can still take.
Each path of a leaf is one generator: it yields a question (``Ask``), is sent
the answer, None when the call exhausted its validation attempts, and decides
which question follows. Collection drives a path by sending each question to
its provider (``catmodel.collection.contextual_collection``); capture validation
drives the same path by replaying each question's stored attempts
(``catmodel.setup.contextual_inputs``).

A leaf has two paths, asked side by side. The direct path (``direct_path``)
asks the direct question and one coherence retry when the answer's large-loss
count contradicts its loss quantiles; the EAL cross-check and its own
coherence retry (``crosscheck_path``) follow where the case's stored EALs call
for a cross-check. The chain path (``chain_path``) asks the event-chain
question and one guardrail retry, which waits for the leaf's direct question
because an unanswered leaf asks none.

A taxonomy parent has one question, asked beside its group's leaves and
followed by none. It is a kind here but not a path.

What decides a follow-up is stored with the case: the large-loss share of a
direct or revised answer and the cross-check EALs. Collection computes each
value once; replay reads it from the case, so a capture replays on a platform
whose floating point differs in the last digits, and the artifact's own
validation holds each stored value to its answers.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any, Final, NamedTuple

from pydantic import BaseModel

from catmodel.contracts.contextual_risk import (
    ChainGuardrailRetry,
    ContextualChainAnswer,
    ContextualRiskAnswer,
    ContextualRiskCase,
    DirectSeverityRetry,
    EalCrosscheckAnswer,
    EalReconciliation,
    ExhaustedCallKind,
    ParentAnswer,
)
from catmodel.elicitation.contextual_frequency import (
    ChainGuardrailRetryPrompt,
    ContextualChainPrompt,
    adopt_retry,
    chain_guardrail_retry_prompt,
    parse_chain_answer,
    parse_chain_guardrail_retry_answer,
)
from catmodel.elicitation.contextual_hierarchy import (
    ContextualParentPrompt,
    parse_parent_answer,
)
from catmodel.elicitation.contextual_risk import (
    ContextualPrompt,
    ContextualSeverityRetryPrompt,
    direct_severity_retry_prompt,
    large_loss_contradiction,
    parse_contextual_answer,
    parse_severity_retry_answer,
    stored_large_loss_share,
)
from catmodel.elicitation.eal_crosscheck import (
    EalCrosscheckPrompt,
    EalCrosscheckRetryPrompt,
    crosscheck_retry_prompt,
    parse_eal_crosscheck_answer,
    parse_eal_crosscheck_retry_answer,
    revised_share,
)

type Question = (
    ContextualPrompt
    | ContextualSeverityRetryPrompt
    | ContextualChainPrompt
    | ChainGuardrailRetryPrompt
    | EalCrosscheckPrompt
    | EalCrosscheckRetryPrompt
    | ContextualParentPrompt
)
"""A rendered question of any kind."""


@dataclass(frozen=True)
class QuestionKind:
    """One kind of question: how its response is read and how long its path
    still is."""

    name: ExhaustedCallKind
    """The name an artifact gives the kind when it lists an exhausted call."""
    parser: Callable[..., Any]
    """Reads a response into the kind's answer, given its question as
    ``prompt``, or raises for a response that fails validation."""
    response_model: type[BaseModel]
    turns: int
    """The most turns its path can still take when it is asked, itself
    included. Collection sends the request with the most turns left first."""


DIRECT: Final = QuestionKind("direct", parse_contextual_answer, ContextualRiskAnswer, 4)
DIRECT_SEVERITY_RETRY: Final = QuestionKind(
    "direct_severity_retry", parse_severity_retry_answer, ContextualRiskAnswer, 3
)
EAL_CROSSCHECK: Final = QuestionKind(
    "eal_crosscheck", parse_eal_crosscheck_answer, EalCrosscheckAnswer, 2
)
EAL_CROSSCHECK_SEVERITY_RETRY: Final = QuestionKind(
    "eal_crosscheck_severity_retry",
    parse_eal_crosscheck_retry_answer,
    EalCrosscheckAnswer,
    1,
)
CHAIN: Final = QuestionKind("chain", parse_chain_answer, ContextualChainAnswer, 2)
CHAIN_GUARDRAIL_RETRY: Final = QuestionKind(
    "chain_guardrail_retry",
    parse_chain_guardrail_retry_answer,
    ContextualChainAnswer,
    1,
)
PARENT: Final = QuestionKind("parent", parse_parent_answer, ParentAnswer, 1)

QUESTION_KINDS: Final = (
    DIRECT,
    DIRECT_SEVERITY_RETRY,
    EAL_CROSSCHECK,
    EAL_CROSSCHECK_SEVERITY_RETRY,
    CHAIN,
    CHAIN_GUARDRAIL_RETRY,
    PARENT,
)
"""Every kind of question a contextual collection asks."""


@dataclass(frozen=True)
class Ask:
    """One question to ask now."""

    kind: QuestionKind
    prompt: Question


class _DirectAnswered:
    def __repr__(self) -> str:
        return "DIRECT_ANSWERED"


DIRECT_ANSWERED: Final = _DirectAnswered()
"""What a chain path yields to wait for its leaf's direct question. It is sent
whether that question has an answer: False when its call is exhausted or
failed."""


class DirectPath(NamedTuple):
    """What a leaf's direct question and its coherence retry leave for the
    case to store."""

    answer: ContextualRiskAnswer
    large_loss_share: float | None
    severity_retry: DirectSeverityRetry | None

    @property
    def stated(self) -> ContextualRiskAnswer:
        """The answer the leaf goes on with: the direct answer, with a
        resolving retry's counts and loss quantiles."""
        retry = self.severity_retry
        if retry is not None and retry.resolved and retry.answer is not None:
            return adopt_retry(self.answer, retry.answer)
        return self.answer


def _share(answer: ContextualRiskAnswer | None) -> float | None:
    """The large-loss share collection stores with an available answer."""
    if answer is None or answer.status != "available":
        return None
    return stored_large_loss_share(answer.rate, answer.large_loss_count)


def direct_path(
    prompt: ContextualPrompt, stored: ContextualRiskCase | None = None
) -> Generator[Ask, ContextualRiskAnswer | None, DirectPath | None]:
    """A leaf's direct question, then one coherence retry when the answer's
    large-loss count contradicts its loss quantiles.

    Returns None when the direct question is exhausted, which leaves the leaf
    unanswered. A retry resolves when its answer is available and coherent;
    otherwise the original answer stands.

    Replay passes the case the capture ``stored``: its stored share then
    decides the retry and words its question, as it did in collection.
    """
    answer = yield Ask(DIRECT, prompt)
    if answer is None:
        return None
    share = _share(answer) if stored is None else stored.large_loss_share
    coherence = direct_severity_retry_prompt(prompt, answer, share)
    if coherence is None:
        return DirectPath(answer, share, None)
    second = yield Ask(DIRECT_SEVERITY_RETRY, coherence)
    second_share = _share(second)
    return DirectPath(
        answer,
        share,
        DirectSeverityRetry(
            prompt_sha256=coherence.prompt_sha256(),
            contradiction=coherence.contradiction,
            answer=second,
            large_loss_share=second_share,
            resolved=second is not None
            and second.status == "available"
            and large_loss_contradiction(
                prompt.node.total, second.total_loss, second_share
            )
            is None,
        ),
    )


def crosscheck_path(
    question: EalCrosscheckPrompt, stored: EalReconciliation | None = None
) -> Generator[
    Ask,
    EalCrosscheckAnswer | None,
    tuple[
        EalCrosscheckAnswer | None, tuple[str, str, EalCrosscheckAnswer | None] | None
    ],
]:
    """A leaf's EAL cross-check, then one coherence retry when the answer
    revises the count, severity or large-loss count into a contradiction.

    ``question`` is the cross-check the case's stored EALs call for
    (``crosscheck_prompt``). Returns the answer and, when the retry was asked,
    its question's digest, its contradiction and its answer: what
    ``catmodel.setup.implied_eal.crosscheck_reconciliation`` records.

    Replay passes the record the capture ``stored``: its stored share of the
    revised answer then decides the retry and words its question.
    """
    reply = yield Ask(EAL_CROSSCHECK, question)
    share = (
        revised_share(question, reply) if stored is None else stored.large_loss_share
    )
    coherence = (
        None if reply is None else crosscheck_retry_prompt(question, reply, share)
    )
    if coherence is None:
        return reply, None
    second = yield Ask(EAL_CROSSCHECK_SEVERITY_RETRY, coherence)
    return reply, (coherence.prompt_sha256(), coherence.contradiction, second)


def chain_path(
    prompt: ContextualChainPrompt,
) -> Generator[
    Ask | _DirectAnswered,
    Any,
    tuple[ContextualChainAnswer | None, ChainGuardrailRetry | None],
]:
    """A leaf's event-chain question, then one guardrail retry when the leaf's
    direct question has an answer and the available chain answer trips a
    guardrail.

    The chain question is asked without waiting for the direct question; only
    the retry waits (``DIRECT_ANSWERED``). A retry resolves when its answer is
    available and trips no guardrail.
    """
    chain = yield Ask(CHAIN, prompt)
    if not (yield DIRECT_ANSWERED):
        return chain, None
    retry_prompt = chain_guardrail_retry_prompt(prompt, chain)
    if retry_prompt is None:
        return chain, None
    second = yield Ask(CHAIN_GUARDRAIL_RETRY, retry_prompt)
    return chain, ChainGuardrailRetry(
        prompt_sha256=retry_prompt.prompt_sha256(),
        findings=retry_prompt.findings,
        answer=second,
        resolved=second is not None
        and second.status == "available"
        and chain_guardrail_retry_prompt(prompt, second) is None,
    )
