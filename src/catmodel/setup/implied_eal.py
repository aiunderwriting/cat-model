"""The EAL a leaf's answers imply, and the leaf EAL cross-check's stored records.

A leaf's implied EAL at a coordinate is the taxonomy parents' leaf mean EAL
(``contextual_node.leaf_eal``): its count law's epistemic mean times the
analytic welfare mean per expected unit of its law, compiled and admitted as a
run admits it, large-loss anchor included. Collection stores it, its drivers
and the direct EAL's mean to four significant figures
(``elicitation.eal_crosscheck``); replay recomputes them here within
``CHAIN_COUNT_TOLERANCE``.
"""

from __future__ import annotations

import math

from catmodel.contracts.contextual_risk import (
    ContextualRiskAnswer,
    ContextualRiskArtifact,
    ContextualRiskInstrument,
    ContextualRiskNode,
    EalCrosscheck,
    EalCrosscheckAnswer,
    EalCrosscheckSeverityRetry,
    EalReconciliation,
    ImpliedEal,
)
from catmodel.contracts.elicitation import Quantiles
from catmodel.elicitation.contextual_frequency import (
    CHAIN_COUNT_TOLERANCE,
    direct_answer,
)
from catmodel.elicitation.contextual_hierarchy import law_mean
from catmodel.elicitation.contextual_risk import large_loss_contradiction
from catmodel.elicitation.eal_crosscheck import (
    EalCrosscheckPrompt,
    adopted_answer,
    answer_before_crosscheck,
    case_crosscheck_prompt,
    crosscheck_retry_prompt,
    revised_answer,
    revised_share,
    validate_crosscheck_answer,
)
from catmodel.elicitation.epistemic import epistemic_upper_tail_mean
from catmodel.probability.numerics import significant
from catmodel.setup.contextual_node import leaf_eal, leaf_query, unit_law


def _significant(value: float | None) -> float | None:
    return None if value is None else significant(value)


def _share(value: float | None) -> float | None:
    """A share at four significant figures, kept inside [0, 1]."""
    return None if value is None else min(1.0, max(0.0, significant(value)))


def _tail_share(rate: Quantiles, mean: float | None) -> float | None:
    """The part of a count law's mean from ranks beyond its p95."""
    if not mean:
        return None
    return 0.05 * epistemic_upper_tail_mean(rate, 0.95) / mean


def implied_eal(
    instrument: ContextualRiskInstrument,
    node: ContextualRiskNode,
    answer: ContextualRiskAnswer,
    anchor_score: float,
) -> ImpliedEal | None:
    """The EAL an answer implies at one coordinate, with its drivers; None
    for an unavailable answer."""
    if answer.status != "available" or answer.rate is None:
        return None
    rate = answer.rate
    _, mean = leaf_eal(instrument, node.node_id, answer, anchor_score)
    count = law_mean(rate)
    law = unit_law(leaf_query(instrument, node.node_id, answer, anchor_score))
    return ImpliedEal(
        eal_mean=_significant(mean),
        count_mean=_significant(count),
        count_median=float(f"{rate.p50:.4g}"),
        harm_per_incident_mean=None if law is None else _significant(law.welfare),
        gross_per_incident_mean=None if law is None else _significant(law.gross),
        total_mean=None if law is None else _significant(law.total_mean),
        severity_median=None
        if answer.total_loss is None
        else float(f"{answer.total_loss.p50:.4g}"),
        count_share_beyond_p95=_share(_tail_share(rate, count)),
        gross_share_at_or_above_large_loss=None
        if law is None
        else _share(law.gross_share_at_or_above_large_loss),
    )


def direct_eal_mean(answer: ContextualRiskAnswer) -> float | None:
    """The direct EAL law's epistemic mean at four significant figures."""
    return _significant(law_mean(answer.direct_eal))


def recorded_crosscheck(
    instrument: ContextualRiskInstrument,
    node: ContextualRiskNode,
    before: ContextualRiskAnswer,
    anchor_score: float,
) -> EalCrosscheck | None:
    """The values collection stores for a case's cross-check, from the
    answer that would drive the run without it; None for an unavailable
    answer."""
    implied = implied_eal(instrument, node, before, anchor_score)
    if implied is None:
        return None
    return EalCrosscheck(direct_eal_mean=direct_eal_mean(before), implied=implied)


def crosscheck_reconciliation(
    prompt: EalCrosscheckPrompt,
    answer: EalCrosscheckAnswer | None,
    retry: tuple[str, str, EalCrosscheckAnswer | None] | None,
) -> EalReconciliation:
    """The stored record of a cross-check call and its coherence retry.

    ``answer`` is None when the call exhausted its validation attempts; every
    answer then stands and nothing else is recorded. ``retry`` holds the
    retry question's digest, its contradiction and its answer (None when it
    exhausted its validation attempts), or is None when the answer needed no
    retry. The after values are those of the answer that then drives the run.
    """
    if answer is None:
        return EalReconciliation(
            prompt_sha256=prompt.prompt_sha256(),
            presentation_order=prompt.presentation_order,
            answer=None,
        )
    share = revised_share(prompt, answer)
    severity_retry = None
    if retry is not None:
        digest, contradiction, second = retry
        second_share = revised_share(prompt, second)
        severity_retry = EalCrosscheckSeverityRetry(
            prompt_sha256=digest,
            contradiction=contradiction,
            answer=second,
            large_loss_share=second_share,
            resolved=second is not None
            and second.revises
            and large_loss_contradiction(
                prompt.node.total,
                revised_answer(prompt.stated, second).total_loss,
                second_share,
            )
            is None,
        )
    driving = (
        revised_answer(prompt.stated, adopted_answer(answer, severity_retry))
        if answer.revises
        else None
    )
    return EalReconciliation(
        prompt_sha256=prompt.prompt_sha256(),
        presentation_order=prompt.presentation_order,
        answer=answer,
        large_loss_share=share,
        severity_retry=severity_retry,
        direct_eal_mean_after=None if driving is None else direct_eal_mean(driving),
        implied_after=None
        if driving is None
        else implied_eal(prompt.instrument, prompt.node, driving, prompt.anchor_score),
    )


def _close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=CHAIN_COUNT_TOLERANCE, abs_tol=1e-12)


def _implied_matches(stored: ImpliedEal | None, expected: ImpliedEal | None) -> bool:
    if stored is None or expected is None:
        return stored is expected
    return all(
        _close(getattr(stored, field), getattr(expected, field))
        for field in ImpliedEal.model_fields
    )


def validate_eal_crosschecks(
    artifact: ContextualRiskArtifact, instrument: ContextualRiskInstrument
) -> None:
    """Each case's stored cross-check values are this code's reading of its
    answers, and exactly the triggered cases carry the cross-check their
    stored values call for, answered validly and resolved as recorded."""
    nodes = {node.node_id: node for node in instrument.collectible_nodes}
    for case in artifact.cases:
        node = nodes[case.node_id]
        check = case.eal_crosscheck
        before = answer_before_crosscheck(case)
        if (check is None) != (direct_answer(case).status != "available"):
            raise ValueError(
                "a case records an EAL cross-check exactly when its direct answer is available"
            )
        if check is None:
            continue
        expected = recorded_crosscheck(instrument, node, before, case.anchor_score)
        assert expected is not None
        if not (
            _close(check.direct_eal_mean, expected.direct_eal_mean)
            and _implied_matches(check.implied, expected.implied)
        ):
            raise ValueError(
                "stored cross-check EALs differ from those the case's answers imply"
            )
        prompt = case_crosscheck_prompt(instrument, case)
        reconciliation = check.reconciliation
        if (prompt is None) != (reconciliation is None):
            raise ValueError(
                "a case carries an EAL cross-check call exactly when its stored "
                "direct and implied EAL means differ by more than the cross-check factor"
            )
        if prompt is None or reconciliation is None:
            continue
        if (
            reconciliation.prompt_sha256 != prompt.prompt_sha256()
            or reconciliation.presentation_order != prompt.presentation_order
        ):
            raise ValueError("EAL cross-check differs from its frozen prompt")
        answer = reconciliation.answer
        if answer is None:
            continue
        validate_crosscheck_answer(answer, prompt)
        if not _close(reconciliation.large_loss_share, revised_share(prompt, answer)):
            raise ValueError("stored large-loss share differs from the cross-check's")
        retry_prompt = crosscheck_retry_prompt(
            prompt, answer, reconciliation.large_loss_share
        )
        retry = reconciliation.severity_retry
        if (retry_prompt is None) != (retry is None):
            raise ValueError(
                "a cross-check carries a coherence retry exactly when its revised "
                "large-loss count contradicts its loss quantiles"
            )
        if retry_prompt is not None and retry is not None:
            if retry.answer is not None:
                validate_crosscheck_answer(retry.answer, prompt)
            if (
                retry.prompt_sha256 != retry_prompt.prompt_sha256()
                or retry.contradiction != retry_prompt.contradiction
            ):
                raise ValueError(
                    "cross-check coherence retry differs from its frozen question"
                )
        recomputed = crosscheck_reconciliation(
            prompt,
            answer,
            None
            if retry is None
            else (retry.prompt_sha256, retry.contradiction, retry.answer),
        )
        if retry is not None and (
            recomputed.severity_retry is None
            or retry.resolved != recomputed.severity_retry.resolved
            or not _close(
                retry.large_loss_share, recomputed.severity_retry.large_loss_share
            )
        ):
            raise ValueError(
                "cross-check coherence retry resolution differs from its answer"
            )
        if not (
            _close(
                reconciliation.direct_eal_mean_after, recomputed.direct_eal_mean_after
            )
            and _implied_matches(reconciliation.implied_after, recomputed.implied_after)
        ):
            raise ValueError(
                "stored cross-check EALs after revision differ from those its answers imply"
            )
