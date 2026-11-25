"""The chain guardrail's one retry."""

from __future__ import annotations

import pytest

from catmodel.contracts.contextual_risk import (
    ChainGuardrailRetry,
    ContextualRiskCase,
)
from catmodel.contracts.elicitation import QUANTILES, Quantiles
from catmodel.elicitation.contextual_frequency import (
    ContextualChainPrompt,
    chain_count_law_for_answer,
    chain_guardrail_retry_prompt,
    chain_prompts,
    driving_chain_answer,
)
from test_support.contextual_risk import (
    chain_answer,
    contextual_answer,
    contextual_instrument,
)


def _prompt() -> ContextualChainPrompt:
    return chain_prompts(contextual_instrument())[0]


def _wide(answer):
    """``answer`` with an opportunity rate spanning four decades."""
    rate = Quantiles(**dict(zip(QUANTILES, (0.01, 0.1, 1.0, 10.0, 100.0), strict=True)))
    return answer.model_copy(update={"opportunity_rate": rate})


def _case(chain, retry: ChainGuardrailRetry | None) -> ContextualRiskCase:
    prompt = _prompt()
    driving = chain if retry is None or retry.answer is None else retry.answer
    return ContextualRiskCase(
        node_id=prompt.node.node_id,
        anchor_score=prompt.anchor_score,
        respondent=prompt.respondent,
        answer=contextual_answer(),
        prompt_sha256="a" * 64,
        chain_answer=chain,
        chain_prompt_sha256=prompt.prompt_sha256(),
        chain_guardrail=retry,
        chain_count=chain_count_law_for_answer(driving, prompt.exposure.count),
    )


def test_only_a_flagged_chain_gets_a_retry_that_names_its_findings():
    prompt = _prompt()
    assert chain_guardrail_retry_prompt(prompt, chain_answer()) is None
    rare = chain_answer(8, (0.5, 1e-4))
    retry = chain_guardrail_retry_prompt(prompt, rare)
    assert retry is not None and len(retry.findings) == 1
    assert "terminal stage" in retry.findings[0]
    both = chain_guardrail_retry_prompt(prompt, _wide(rare))
    assert both is not None and len(both.findings) == 2
    text = both.render()
    assert text.startswith(prompt.render())
    check = text.removeprefix(prompt.render())
    # The retry states the values that tripped the guardrail and asks to confirm
    # or revise, naming no threshold and no direction of change.
    assert (
        "your opportunity_rate p95 is 10,000 times its p5 (100 and 0.01) and your "
        "qualified_terminal stage's p50 is 0.0001"
    ) in check
    assert "Confirm or revise it" in check
    assert "Keeping your answer is as acceptable as changing it" in check
    for word in ("1,000", "0.001", "grain", "too ", "narrow", "wide", "fine"):
        assert word not in check, word
    # The retry is its own frozen call, distinct from the chain question.
    assert both.call_id != prompt.call_id


def test_an_available_retry_drives_the_chain_even_when_it_still_violates():
    first = _wide(chain_answer())
    second = _wide(chain_answer(4))
    retry = ChainGuardrailRetry(
        prompt_sha256="c" * 64,
        findings=("opportunity_rate p95/p5 exceeds 1,000",),
        answer=second,
        resolved=False,
    )
    assert driving_chain_answer(_case(first, retry)) == second


def test_an_exhausted_retry_leaves_the_first_answer_standing():
    first = _wide(chain_answer())
    retry = ChainGuardrailRetry(
        prompt_sha256="c" * 64,
        findings=("opportunity_rate p95/p5 exceeds 1,000",),
        answer=None,
        resolved=False,
    )
    assert driving_chain_answer(_case(first, retry)) == first


def test_a_resolved_retry_needs_an_available_answer():
    with pytest.raises(ValueError):
        ChainGuardrailRetry(
            prompt_sha256="c" * 64,
            findings=("opportunity_rate p95/p5 exceeds 1,000",),
            answer=None,
            resolved=True,
        )
