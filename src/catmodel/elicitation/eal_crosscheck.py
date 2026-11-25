"""The leaf EAL cross-check, and the pooled answer that drives each leaf's run.

Every leaf's direct question also asks for its direct EAL: the respondent's
uncertainty about the node's expected annual societal harm, judged directly
rather than built from its count and marks. After any large-loss coherence
retry, collection compares the direct EAL's epistemic mean with the implied
EAL: the direct count law's epistemic mean times the compiled law's analytic
welfare mean per expected unit (``setup.implied_eal``). When the two differ
by more than ``EAL_CROSSCHECK_RATIO``, one more call shows the respondent both
and asks which of its judgments, if any, it revises: the count, the severity,
the large-loss count or the direct EAL. The check reads only the direct
answer, so it runs before the chain enters (``FREQ-EAL-CROSSCHECK``).

``final_answer`` is the cross-checked direct answer. ``pooled_answer`` pools
its count with the case's chain count law at ``CHAIN_POOL_WEIGHT``
(``FREQ-CHAIN-POOL``) and keeps its large-loss share; it drives the leaf's
run, and the taxonomy parents' pools read it.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from catmodel.contracts.contextual_risk import (
    EAL_CROSSCHECK_FIELDS,
    ContextualComponentAnswer,
    ContextualRiskAnswer,
    ContextualRiskArtifact,
    ContextualRiskCase,
    ContextualRiskInstrument,
    ContextualRiskNode,
    EalCrosscheck,
    EalCrosscheckAnswer,
    EalCrosscheckSeverityRetry,
    EalReconciliation,
    eal_crosscheck_judgments,
)
from catmodel.contracts.elicitation import QUANTILES, Quantiles, Respondent
from catmodel.elicitation.chain_law import CHAIN_POOL_WEIGHT, pooled_count
from catmodel.elicitation.contextual_frequency import (
    CHAIN_SYSTEM_PROMPT,
    LargeLossStatus,
    answer_large_loss_status,
    chain_guardrail_replaces,
    direct_answer,
    severity_retry_replaces,
)
from catmodel.elicitation.contextual_risk import (
    ContextualPrompt,
    RenderedOnce,
    check_cited_cards,
    components_text,
    large_loss_contradiction,
    stored_large_loss_share,
)
from catmodel.elicitation.epistemic import epistemic_mean
from catmodel.elicitation.loss_level_counts import (
    FIXED_LOSS_LEVELS,
    LOSS_LEVEL_COUNTS_NOTE,
    REPORTED_QUANTILES,
    LossLevelCount,
    loss_level_counts,
)
from catmodel.elicitation.question_text import (
    SOCIETAL_HARM_BASIS,
    answer_schema_text,
    number,
    quantile_text,
    reasoning_text,
    usd,
)
from catmodel.hashing import canonical_json_sha256
from catmodel.json_io import strict_json_loads
from catmodel.probability.numerics import differs_by_more_than
from catmodel.validation_errors import EvidenceValidationError

EAL_CROSSCHECK_RATIO = 3.0
"""INITIAL_ESTIMATE factor between a leaf's direct and implied EAL means above
which the respondent is asked to reconcile them (``FREQ-EAL-CROSSCHECK``).

An operational convention, a coarse checkpoint for a disagreement of about
half an order of magnitude that decides which calls collection makes.
A judgment, not calibrated: no data identify how far a direct EAL
judgment and the EAL its own count and marks imply may differ before one of
them is probably wrong. Compared exactly as the stored decimals
(``differs_by_more_than``); equality does not trigger, two zeros never do and
exactly one zero always does.
"""

type EalCrosscheckStatus = Literal[
    "agreed",
    "reconciled",
    "declined",
    "unresolved",
    "exhausted",
    "unavailable",
]
"""How a leaf case's EAL cross-check stands.

``agreed``: the EAL means are within the factor. ``reconciled``: a trigger
fired, the respondent revised, and the revised means differ by at most the
factor. ``declined``: the EAL means differ by more than the factor and the
respondent kept every answer (``unchanged``), with its reason.
``unresolved``: the revised means differ by more than the factor, or one of
them is unidentified; the revised answers drive the run. ``exhausted``: a
trigger fired and the cross-check call exhausted its validation attempts, so
every answer stands as stated. ``unavailable``: no direct answer, direct EAL
mean or implied EAL to compare. ``declined`` and ``unresolved`` cases are
``eal_crosscheck_unresolved``; ``exhausted`` ones are counted apart, since no
respondent judgment kept their answers.
"""

EAL_CROSSCHECK_STATUSES: tuple[EalCrosscheckStatus, ...] = (
    "agreed",
    "reconciled",
    "declined",
    "unresolved",
    "exhausted",
    "unavailable",
)


type CrosscheckTrigger = Literal["eal_gap"]
"""Why the direct EAL cross-check is asked."""

CROSSCHECK_TRIGGERS: tuple[CrosscheckTrigger, ...] = ("eal_gap",)


def crosscheck_triggers(check: EalCrosscheck) -> tuple[CrosscheckTrigger, ...]:
    """The EAL-ratio trigger stored values fire."""
    direct, implied = check.direct_eal_mean, check.implied.eal_mean
    if direct is None or implied is None:
        return ()
    if differs_by_more_than(direct, implied, EAL_CROSSCHECK_RATIO):
        return ("eal_gap",)
    return ()


def case_triggers(case: ContextualRiskCase) -> tuple[CrosscheckTrigger, ...]:
    """The cross-check triggers a stored case fires; none without its
    cross-check record."""
    check = case.eal_crosscheck
    return () if check is None else crosscheck_triggers(check)


def adopted_answer(
    answer: EalCrosscheckAnswer, retry: EalCrosscheckSeverityRetry | None
) -> EalCrosscheckAnswer:
    """The cross-check answer the run uses: its coherence retry where that
    resolved a contradiction, otherwise the first answer."""
    if retry is not None and retry.resolved and retry.answer is not None:
        return retry.answer
    return answer


def adopted_crosscheck(reconciliation: EalReconciliation) -> EalCrosscheckAnswer:
    """The adopted answer of an answered cross-check call."""
    assert reconciliation.answer is not None
    return adopted_answer(reconciliation.answer, reconciliation.severity_retry)


def eal_crosscheck_status(case: ContextualRiskCase) -> EalCrosscheckStatus:
    check = case.eal_crosscheck
    if check is None or check.direct_eal_mean is None or check.implied.eal_mean is None:
        return "unavailable"
    triggers = case_triggers(case)
    if not triggers:
        return "agreed"
    reconciliation = check.reconciliation
    if reconciliation is None:
        raise ValueError("a triggered EAL cross-check carries its reconciliation")
    if reconciliation.answer is None:
        return "exhausted"
    if not reconciliation.answer.revises:
        return "declined"
    direct = reconciliation.direct_eal_mean_after
    implied = (
        None
        if reconciliation.implied_after is None
        else reconciliation.implied_after.eal_mean
    )
    if (
        direct is None
        or implied is None
        or differs_by_more_than(direct, implied, EAL_CROSSCHECK_RATIO)
    ):
        return "unresolved"
    return "reconciled"


def eal_ratio(direct: float | None, implied: float | None) -> float | None:
    """The implied EAL mean divided by the direct one; None where either is
    unidentified or the direct mean is zero."""
    if direct is None or implied is None or direct == 0:
        return None
    return implied / direct


def revised_answer(
    before: ContextualRiskAnswer, answer: EalCrosscheckAnswer
) -> ContextualRiskAnswer:
    """``before`` with the judgments ``answer`` revises replaced, validated as
    a direct answer; an unchanged answer leaves it as it was."""
    if not answer.revises:
        return before
    supplied = answer.model_dump(mode="json")
    data = before.model_dump(mode="json")
    named = {row.judgment_id for row in answer.revised}
    for judgment, fields in EAL_CROSSCHECK_FIELDS.items():
        for field in fields:
            if judgment in named and field in supplied:
                data[field] = supplied[field]
    return ContextualRiskAnswer.model_validate(data)


def answer_before_crosscheck(case: ContextualRiskCase) -> ContextualRiskAnswer:
    """The direct answer after its large-loss coherence retry."""
    return direct_answer(case)


LOSS_JUDGMENTS = ("count", "severity", "large_loss_count")
"""The judgments whose revision can change the large-loss share or the loss
quantiles it is checked against; only these call for a coherence retry."""


def revises_loss(answer: EalCrosscheckAnswer) -> bool:
    """Whether a cross-check answer revises the count, severity or large-loss
    count; a revision of the direct EAL alone leaves the anchor as it was."""
    return answer.revises and any(
        row.judgment_id in LOSS_JUDGMENTS for row in answer.revised
    )


def crosscheck_large_loss_status(
    revised: ContextualRiskAnswer,
    share: float | None,
    retry: EalCrosscheckSeverityRetry | None,
) -> LargeLossStatus:
    """The large-loss status of a cross-check: from its coherence retry where
    one ran, otherwise from the first revised answer and its stored share."""
    return answer_large_loss_status(
        revised, share, retry is not None, retry is not None and retry.resolved
    )


def _driving_crosscheck(case: ContextualRiskCase) -> EalReconciliation | None:
    check = case.eal_crosscheck
    if (
        check is None
        or check.reconciliation is None
        or check.reconciliation.answer is None
        or not check.reconciliation.answer.revises
    ):
        return None
    return check.reconciliation


def large_loss_status(case: ContextualRiskCase) -> LargeLossStatus:
    """The status of the large-loss anchor of the answer that drives the run:
    the EAL cross-check's revised answer where the respondent revised,
    otherwise the direct answer and its coherence retry."""
    crosscheck = _driving_crosscheck(case)
    first = None if crosscheck is None else crosscheck.answer
    if crosscheck is not None and first is not None and revises_loss(first):
        revised = revised_answer(direct_answer(case), adopted_crosscheck(crosscheck))
        return crosscheck_large_loss_status(
            revised, crosscheck.large_loss_share, crosscheck.severity_retry
        )
    retry = case.severity_retry
    return answer_large_loss_status(
        case.answer,
        case.large_loss_share,
        retry is not None,
        retry is not None and retry.resolved,
    )


def _crosschecked(
    case: ContextualRiskCase, before: ContextualRiskAnswer
) -> ContextualRiskAnswer:
    """``before`` with the judgments the case's EAL cross-check revises."""
    crosscheck = _driving_crosscheck(case)
    if crosscheck is None:
        return before
    return revised_answer(before, adopted_crosscheck(crosscheck))


def final_answer(case: ContextualRiskCase) -> ContextualRiskAnswer:
    """The cross-checked direct answer that supplies the severity judgments."""
    return _crosschecked(case, answer_before_crosscheck(case))


def _pool_direct_answer(
    direct: ContextualRiskAnswer,
    chain_count: Quantiles | None,
    weight: float,
) -> ContextualRiskAnswer:
    """Pool one direct answer and preserve its large-loss count share."""
    if direct.status != "available" or direct.rate is None:
        return direct
    pooled = pooled_count(direct.rate, chain_count, weight)
    update: dict[str, Any] = {"rate": pooled}
    try:
        direct_mean = epistemic_mean(direct.rate)
        pooled_mean = epistemic_mean(pooled)
    except ValueError:
        direct_mean = pooled_mean = math.nan
    if direct_mean > 0 and math.isfinite(direct_mean) and math.isfinite(pooled_mean):
        factor = pooled_mean / direct_mean
        for field in ("large_loss_count", "very_large_loss_count"):
            law = getattr(direct, field)
            if law is not None:
                update[field] = Quantiles(
                    **{key: getattr(law, key) * factor for key in QUANTILES}
                )
    return direct.model_copy(update=update)


def pooled_answer(
    case: ContextualRiskCase, weight: float = CHAIN_POOL_WEIGHT
) -> ContextualRiskAnswer:
    """The cross-checked direct answer with its count pooled with the chain law.

    Large-loss counts keep the cross-checked direct answer's share by scaling
    each stated quantile by the pooled-to-direct count mean ratio.
    """
    return _pool_direct_answer(final_answer(case), case.chain_count, weight)


def pooled_answer_before_crosscheck(case: ContextualRiskCase) -> ContextualRiskAnswer:
    """The pooled answer built from the direct answer before its EAL check."""
    return _pool_direct_answer(
        answer_before_crosscheck(case), case.chain_count, CHAIN_POOL_WEIGHT
    )


def pooled_answer_before_severity_retry(
    case: ContextualRiskCase,
) -> ContextualRiskAnswer:
    """The pooled answer built from the direct answer as first given, in place
    of its resolving large-loss coherence retry.

    The original answer's large-loss share contradicts its loss quantiles, so
    its law is the provisional p99-only tail of an unresolved case. A later
    EAL cross-check was asked about the retried answer; the judgments it
    revises still replace the original's, and every other stored answer stands.
    """
    return _pool_direct_answer(
        _crosschecked(case, case.answer), case.chain_count, CHAIN_POOL_WEIGHT
    )


def pooled_answer_before_chain_guardrail(
    case: ContextualRiskCase, original_chain_count: Quantiles | None
) -> ContextualRiskAnswer:
    """The pooled answer with ``original_chain_count``, the count law of the
    chain answer as first given, in place of its guardrail retry's."""
    return _pool_direct_answer(
        final_answer(case), original_chain_count, CHAIN_POOL_WEIGHT
    )


def _final_count_mean(case: ContextualRiskCase) -> float | None:
    """The stored direct count mean after the EAL cross-check, when revised."""
    check = case.eal_crosscheck
    if check is None:
        return None
    crosscheck = _driving_crosscheck(case)
    if crosscheck is None:
        return check.implied.count_mean
    assert crosscheck.implied_after is not None
    return crosscheck.implied_after.count_mean


def case_loss_level_counts(case: ContextualRiskCase) -> dict[str, Any]:
    """Loss-level counts before and after the EAL cross-check."""

    def counts(
        mean: float | None, answer: ContextualRiskAnswer
    ) -> list[dict[str, Any]] | None:
        if mean is None or answer.total_loss is None:
            return None
        return [row.record() for row in loss_level_counts(mean, answer.total_loss)]

    check = case.eal_crosscheck
    before_mean = None if check is None else check.implied.count_mean
    final_mean = _final_count_mean(case)
    return {
        "before_crosscheck_count_mean": before_mean,
        "final_count_mean": final_mean,
        "before_crosscheck": counts(before_mean, answer_before_crosscheck(case)),
        "final": counts(final_mean, final_answer(case)),
    }


@dataclass(frozen=True)
class StatedLargeLoss:
    """A case's stated large-loss counts, for reporting.

    ``stated`` are the adopted direct answer's counts; ``final`` those of the
    pooled answer that drives the run, as stated even when unresolved.
    """

    stated: Quantiles | None
    stated_very: Quantiles | None
    final: Quantiles | None
    final_very: Quantiles | None
    status: LargeLossStatus


def stated_large_loss(case: ContextualRiskCase) -> StatedLargeLoss:
    direct = direct_answer(case)
    pooled = pooled_answer(case)
    return StatedLargeLoss(
        stated=direct.large_loss_count,
        stated_very=direct.very_large_loss_count,
        final=pooled.large_loss_count,
        final_very=pooled.very_large_loss_count,
        status=large_loss_status(case),
    )


def _number(value: float) -> str:
    """A stored count or amount as the question shows it: whole numbers with
    thousands separators from ten thousand, four significant figures below,
    so every figure in one question reads alike."""
    return f"{value:,.0f}" if abs(value) >= 1e4 else f"{value:,.4g}"


def _value(value: float | None) -> str:
    return "not identified" if value is None else _number(value)


def _usd(value: float | None) -> str:
    return "not identified" if value is None else f"USD {_number(value)}"


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def _times(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator is None or denominator == 0:
        return ""
    return f", {numerator / denominator:,.3g} times "


def _level_cell(count: LossLevelCount, label: str) -> str:
    value = (
        _number(count.annual_count_low)
        if count.single
        else f"{_number(count.annual_count_low)} to {_number(count.annual_count_high)}"
    )
    return f"{value} a year {label}"


def _basis_text(basis: str) -> str:
    return "a reporting level" if basis == "fixed" else f"your total_loss {basis}"


def _gap_text(direct: float, implied: float) -> str:
    low, high = sorted((direct, implied))
    if high == 0:
        return "both are exactly zero"
    if low == 0:
        return "one of them is exactly zero and the other is not"
    return f"they differ by a factor of {high / low:,.3g}"


def _component_value(row: ContextualComponentAnswer) -> str:
    if row.status != "available":
        return f"{row.component_id} unavailable"
    if row.share is not None:
        return f"{row.component_id} share {number(row.share)}"
    if row.amount is not None:
        return f"{row.component_id} amount {number(row.amount)}"
    assert row.loss is not None
    return f"{row.component_id} loss " + quantile_text(row.loss)


def _severity_text(stated: ContextualRiskAnswer) -> str:
    """A stated answer's severity judgments: its total loss and every branch's
    component values, in declaration order."""
    parts = (
        []
        if stated.total_loss is None
        else ["total_loss " + quantile_text(stated.total_loss)]
    )
    parts.extend(
        f"branch {branch.branch_id}, probability {number(branch.probability)}: "
        + ", ".join(_component_value(row) for row in branch.components)
        for branch in stated.branches
    )
    return "; ".join(parts)


@dataclass(frozen=True)
class EalCrosscheckPrompt(RenderedOnce):
    """The cross-check call: a leaf's direct EAL beside the EAL its count and
    marks imply, in a neutral order.

    ``stated`` is the direct answer after its coherence retry. ``check`` holds
    the stored EAL means and drivers used by the trigger.
    """

    system_prompt: ClassVar[str] = CHAIN_SYSTEM_PROMPT

    instrument: ContextualRiskInstrument
    node: ContextualRiskNode
    respondent: Respondent
    anchor_score: float
    stated: ContextualRiskAnswer
    check: EalCrosscheck

    def __post_init__(self) -> None:
        ContextualPrompt(self.instrument, self.node, self.respondent, self.anchor_score)
        if self.stated.status != "available" or not self.triggers:
            raise ValueError("only a triggered available case is cross-checked")

    @property
    def triggers(self) -> tuple[CrosscheckTrigger, ...]:
        return crosscheck_triggers(self.check)

    @property
    def call_id(self) -> str:
        return canonical_json_sha256(
            [
                self.instrument.context_id,
                self.node.node_id,
                self.anchor_score,
                "eal_crosscheck",
            ]
        )

    @property
    def presentation_order(
        self,
    ) -> tuple[Literal["direct", "implied"], Literal["direct", "implied"]]:
        """A neutral order, fixed by the call and respondent, never by the answers."""
        digest = canonical_json_sha256([self.call_id, self.respondent.key()])
        return (
            ("direct", "implied") if int(digest[-1], 16) % 2 else ("implied", "direct")
        )

    def _render(self) -> str:
        stated, check, implied = self.stated, self.check, self.check.implied
        triggers = self.triggers
        assert stated.rate is not None and stated.direct_eal is not None
        assert check.direct_eal_mean is not None and implied.eal_mean is not None
        total = self.node.total
        level = (
            None
            if total is None or total.large_loss_level is None
            else usd(total.large_loss_level)
        )
        count_text = (
            "Count (judgment_id count): rate "
            + quantile_text(stated.rate)
            + f". Its mean is {_value(implied.count_mean)}"
            + (
                _times(implied.count_mean, implied.count_median)
                + f"its median {_value(implied.count_median)}"
                if implied.count_median
                else f" and its median {_value(implied.count_median)}"
            )
            + (
                ""
                if implied.count_share_beyond_p95 is None
                else f"; {implied.count_share_beyond_p95:.0%} of that mean comes "
                "from the part of your rate distribution beyond your p95, which is "
                "extrapolated"
            )
            + "."
        )
        welfare_share = _ratio(
            implied.harm_per_incident_mean, implied.gross_per_incident_mean
        )
        severity_text = (
            "Severity (judgment_id severity): "
            + _severity_text(stated)
            + "."
            + (
                " From these answers, the mean gross loss per qualified unit of "
                f"total_loss is {_usd(implied.total_mean)}"
                + (
                    _times(implied.total_mean, implied.severity_median)
                    + f"your total_loss median {_usd(implied.severity_median)}"
                    if implied.severity_median
                    else ""
                )
                + (
                    ""
                    if implied.gross_share_at_or_above_large_loss is None
                    or level is None
                    else f"; {implied.gross_share_at_or_above_large_loss:.0%} of "
                    f"that mean comes from units at or above {level}"
                )
                + "."
                if implied.total_mean is not None
                else ""
            )
            + " The mean societal harm per qualified unit is "
            f"{_usd(implied.harm_per_incident_mean)}"
            + (
                f", {welfare_share:.0%} of the mean gross loss per unit "
                f"({_usd(implied.gross_per_incident_mean)}): the rest is ordinary "
                "transfers, which carry no societal weight"
                if welfare_share is not None
                else ""
            )
            + "."
        )
        large_text: tuple[str, ...] = ()
        if stated.large_loss_count is not None:
            counts = (
                "large_loss_count "
                + quantile_text(stated.large_loss_count)
                + (
                    ""
                    if stated.very_large_loss_count is None
                    else " and very_large_loss_count "
                    + quantile_text(stated.very_large_loss_count)
                )
            )
            large_text = (
                (
                    "Your stated "
                    + counts
                    + " (judgment_id large_loss_count) contradict your total_loss "
                    "quantiles, so they do not shape this implied EAL; your "
                    "total_loss quantiles do.",
                )
                if self._unresolved()
                else (
                    "Large-loss count (judgment_id large_loss_count): " + counts + ".",
                )
            )
        blocks = {
            "direct": (
                "Your direct EAL (judgment_id direct_eal): "
                + quantile_text(stated.direct_eal)
                + f"; its mean is {_usd(check.direct_eal_mean)} (four significant "
                "figures). You judged it directly, not from your other answers. "
                "Your reasoning then: " + reasoning_text(stated.reasoning)
            ),
            "implied": "\n".join(
                (
                    "The EAL your count and severity answers imply: "
                    f"{_usd(implied.eal_mean)}, the mean of your rate distribution "
                    "times the mean societal harm per qualified unit (four "
                    "significant figures). What drives it:",
                    count_text,
                    severity_text,
                    *large_text,
                )
            ),
        }
        very = total is not None and total.very_large_loss_level is not None
        large_values = (
            (
                "; large_loss_count in both large_loss_count and very_large_loss_count"
                if very
                else "; large_loss_count in large_loss_count"
            )
            if large_text
            else ""
        )
        return "\n\n".join(
            (
                self.direct.scenario(),
                "## Your answers to compare\n"
                "You answered this scenario's question with a count, severity "
                "answers and, separately, a direct judgment of its expected annual "
                "societal harm (EAL). The EAL your count and severity answers imply "
                "and your direct EAL are compared as means, because the EAL is a "
                "mean: "
                + _gap_text(check.direct_eal_mean, implied.eal_mean)
                + (
                    f", more than {EAL_CROSSCHECK_RATIO:g} times"
                    if "eal_gap" in triggers
                    else f", within {EAL_CROSSCHECK_RATIO:g} times"
                )
                + ". Both follow in an order fixed by this question's identity, "
                "not by the answers; the order carries no information and neither "
                f"is presumed more reliable. {SOCIETAL_HARM_BASIS}",
                *(
                    f"Item {index}. {blocks[name]}"
                    for index, name in enumerate(self.presentation_order, start=1)
                ),
                *self._loss_level_text(),
                "## Questions\n"
                + "\n\n".join(
                    (
                        "Say which of your judgments, if any, you now revise: the "
                        "count, the severity, "
                        + ("the large-loss count " if large_text else "")
                        + "or the direct EAL. Two replies are equally acceptable. "
                        "If, having seen both figures, you still hold every answer, "
                        "return status unchanged with your unchanged_reason: your "
                        "answers stand. Otherwise return status revised, add one "
                        "entry to revised for each judgment you change, naming it by "
                        "judgment_id with why you changed it, and supply exactly that "
                        "judgment's new values: count in rate; severity in "
                        "total_loss, the complete branches, or both"
                        + large_values
                        + "; direct_eal in direct_eal. Omit every other field: a "
                        "judgment you do not name stands as stated. Change a "
                        "judgment only if you now hold a different one, not to close "
                        "a gap, and do not split the difference by default.",
                        "Each field keeps its meaning in your original answer. rate "
                        "holds uncertainty quantiles of the expected annual number of "
                        "qualified units that the year rule assigns to the study "
                        "year. total_loss holds variation quantiles of the gross loss "
                        "of one qualified unit, from one unit to the next, with no "
                        "cap. branches hold each branch's probability and every "
                        "component's value, as in your original answer. direct_eal "
                        "holds uncertainty quantiles of the expected annual societal "
                        "harm, judged directly.\n" + components_text(self.node),
                        *self._large_loss_task(level),
                        "Status: supply unchanged_reason only with status unchanged; "
                        "with status revised, set unchanged_reason to null or omit "
                        "it, and give your explanation in revised[].explanation and "
                        "reasoning.",
                    )
                ),
                answer_schema_text(EalCrosscheckAnswer.json_schema_for(self.node)),
            )
        )

    @property
    def direct(self) -> ContextualPrompt:
        """The direct question this cross-check follows up."""
        return ContextualPrompt(
            self.instrument, self.node, self.respondent, self.anchor_score
        )

    def _loss_level_text(self) -> tuple[str, ...]:
        """The loss-level counts at the stored count means; empty without a
        stated total_loss or an identified count mean."""
        total, current = self.stated.total_loss, self.check.implied.count_mean
        if total is None or current is None:
            return ()
        counts = loss_level_counts(current, total)
        rows = [
            f"At or above {_usd(count.level)} ({_basis_text(count.basis)}): "
            + _level_cell(count, "at the count shown here")
            + "."
            for count in counts
        ]
        return (
            "The expected annual number of qualified units with gross loss at or "
            "above each amount that follows from these answers: the count mean "
            "(four significant figures) times the share of units your total_loss "
            "quantiles put there. At one of your quantiles that share is one minus "
            "its probability; at a reporting level between them, and where "
            "quantiles are tied, only bounds follow, rounded outward; above your "
            "p99 the share is at most one in a hundred.\n" + "\n".join(rows),
        )

    def _unresolved(self) -> bool:
        """Whether the stated large-loss count contradicts the stated loss
        quantiles: after collection's retries only an unresolved one does,
        and compilation then uses the p99-only tail."""
        stated = self.stated
        return (
            large_loss_contradiction(
                self.node.total,
                stated.total_loss,
                stored_large_loss_share(stated.rate, stated.large_loss_count),
            )
            is not None
        )

    def _large_loss_task(self, level: str | None) -> tuple[str, ...]:
        if level is None or self.stated.large_loss_count is None:
            return ()
        return (
            "After any revision, your large-loss counts must still nest within your "
            "count, quantile by quantile. large_loss_count's mean divided by your "
            f"rate's mean is the share of qualified units that reach {level}: it "
            f"must be below 1 in 100 if your total_loss p99 is below {level}; at "
            f"least 1 in 100 and below 1 in 20 if your p95 is below {level} and your "
            f"p99 at or above it; and at least 1 in 20 if your p95 is at or above "
            f"{level}.",
        )


@dataclass(frozen=True)
class EalCrosscheckRetryPrompt(RenderedOnce):
    """The one retry of a cross-check answer whose revised large-loss count
    contradicts its loss quantiles (``SEV-LARGE-LOSS-ANCHOR``)."""

    system_prompt: ClassVar[str] = CHAIN_SYSTEM_PROMPT

    crosscheck: EalCrosscheckPrompt
    revised: ContextualRiskAnswer
    contradiction: str

    @property
    def respondent(self) -> Respondent:
        return self.crosscheck.respondent

    @property
    def node(self) -> ContextualRiskNode:
        return self.crosscheck.node

    @property
    def call_id(self) -> str:
        return canonical_json_sha256([self.crosscheck.call_id, "large_loss_coherence"])

    def _render(self) -> str:
        revised = self.revised
        assert revised.rate is not None and revised.large_loss_count is not None
        assert revised.total_loss is not None
        return "\n\n".join(
            (
                self.crosscheck.render(),
                "## Check your answer\nYour answer to this question contradicted "
                "itself. With the judgments it kept, it states rate "
                + quantile_text(revised.rate)
                + "; large_loss_count "
                + quantile_text(revised.large_loss_count)
                + f"; and total_loss p95 {number(revised.total_loss.p95)} and p99 "
                f"{number(revised.total_loss.p99)}. "
                + self.contradiction
                + " The share is your large_loss_count mean divided by your rate "
                "mean, each the mean of the distribution its five quantiles "
                "describe.\nAnswer this question again. Any of the count, the "
                "large-loss count and the severity may be the judgment you now hold "
                "differently: revise whichever it is, naming every judgment you "
                "change. Do not change a judgment you still hold only to remove the "
                "contradiction.",
            )
        )


def crosscheck_prompt(
    instrument: ContextualRiskInstrument,
    node: ContextualRiskNode,
    respondent: Respondent,
    anchor_score: float,
    stated: ContextualRiskAnswer,
    check: EalCrosscheck,
) -> EalCrosscheckPrompt | None:
    """The cross-check a case needs, or None when its EALs are within the factor."""
    if stated.status != "available" or not crosscheck_triggers(check):
        return None
    return EalCrosscheckPrompt(
        instrument,
        node,
        respondent,
        anchor_score,
        stated,
        check,
    )


def case_crosscheck_prompt(
    instrument: ContextualRiskInstrument, case: ContextualRiskCase
) -> EalCrosscheckPrompt | None:
    """A stored case's cross-check question, from its stored values."""
    check = case.eal_crosscheck
    if check is None:
        return None
    node = next(
        row for row in instrument.collectible_nodes if row.node_id == case.node_id
    )
    return crosscheck_prompt(
        instrument,
        node,
        case.respondent,
        case.anchor_score,
        direct_answer(case),
        check,
    )


def crosscheck_retry_prompt(
    prompt: EalCrosscheckPrompt,
    answer: EalCrosscheckAnswer,
    share: float | None,
) -> EalCrosscheckRetryPrompt | None:
    """The coherence retry a cross-check answer needs, or None: only a
    revision of the count, severity or large-loss count is checked, so a
    revision of the direct EAL alone never re-asks a contradiction the
    answers already carried."""
    if not revises_loss(answer):
        return None
    revised = revised_answer(prompt.stated, answer)
    reason = large_loss_contradiction(prompt.node.total, revised.total_loss, share)
    if reason is None:
        return None
    return EalCrosscheckRetryPrompt(prompt, revised, reason)


def revised_share(
    prompt: EalCrosscheckPrompt, answer: EalCrosscheckAnswer | None
) -> float | None:
    """The stored large-loss share of a cross-check answer's revised answer;
    None for a declined or missing answer."""
    if answer is None or not answer.revises:
        return None
    revised = revised_answer(prompt.stated, answer)
    return stored_large_loss_share(revised.rate, revised.large_loss_count)


def validate_crosscheck_answer(
    answer: EalCrosscheckAnswer, prompt: EalCrosscheckPrompt
) -> None:
    """A cross-check answer names only its node's judgments, cites only its
    cards, and leaves a revised answer that passes every direct-answer check."""
    asked = eal_crosscheck_judgments(prompt.node)
    for index, row in enumerate(answer.revised):
        if row.judgment_id not in asked:
            raise EvidenceValidationError(
                "revised_judgment_not_asked_for_this_node",
                "a cross-check revises only the judgments its node asks",
                location=("revised", index, "judgment_id"),
            )
    check_cited_cards(
        answer.reasoning, prompt.node.source_cards, prompt.node.threat_model
    )
    revised_answer(prompt.stated, answer).validate_against(prompt.node)


def parse_eal_crosscheck_answer(
    text: str, prompt: EalCrosscheckPrompt
) -> EalCrosscheckAnswer:
    answer = EalCrosscheckAnswer.model_validate(strict_json_loads(text))
    validate_crosscheck_answer(answer, prompt)
    return answer


def parse_eal_crosscheck_retry_answer(
    text: str, prompt: EalCrosscheckRetryPrompt
) -> EalCrosscheckAnswer:
    return parse_eal_crosscheck_answer(text, prompt.crosscheck)


def crosscheck_revises(case: ContextualRiskCase) -> bool:
    """Whether the cross-check changes the answer that drives the run beyond
    its direct EAL: a revised count, severity or large-loss count, or a
    changed large-loss status."""
    if _driving_crosscheck(case) is None:
        return False
    before = answer_before_crosscheck(case)
    final = final_answer(case)
    return final.model_copy(update={"direct_eal": before.direct_eal}) != before


def eal_crosscheck_limitation(artifacts: Sequence[ContextualRiskArtifact]) -> str:
    """Summarize direct-versus-implied EAL cross-check outcomes."""
    cases = [case for artifact in artifacts for case in artifact.cases]
    statuses = Counter(eal_crosscheck_status(case) for case in cases)
    compared = sum(
        count for status, count in statuses.items() if status != "unavailable"
    )
    return (
        f"Leaf EAL cross-check: of {compared} cases whose direct and implied EAL "
        f"means were compared, {statuses['agreed']} agreed within "
        f"{EAL_CROSSCHECK_RATIO:g} times, {statuses['reconciled']} were revised "
        f"and brought within the factor, {statuses['declined']} declined and "
        f"{statuses['unresolved']} remain more than {EAL_CROSSCHECK_RATIO:g} "
        "times apart after revision. Declined and unresolved cases keep the "
        "answers that drive them. "
        f"{statuses['exhausted']} cross-check calls exhausted their validation "
        "attempts; those cases keep every answer as stated. The direct EAL is a "
        "check and is never simulated; the frequency_estimate exhibit reports "
        "the EAL before and after the cross-check."
    )


LEAF_FOLLOW_UPS = ("severity_retry", "chain_guardrail", "eal_crosscheck")
"""The follow-up calls a leaf case can receive, in the order collection asks
them: the retry of a direct answer whose large-loss count contradicts its loss
quantiles (``SEV-LARGE-LOSS-ANCHOR``), the retry of a chain answer that trips
a guardrail (``FREQ-CHAIN-GUARDRAIL``) and the leaf EAL cross-check
(``FREQ-EAL-CROSSCHECK``)."""


def follow_up_counts(
    captures: Iterable[tuple[ContextualRiskArtifact, Iterable[str]]],
) -> dict[str, dict[str, int]]:
    """How many leaf cases each follow-up call was asked of, and how many of
    them it gave an answer that replaced judgments the run reads.

    A coherence retry replaces the direct answer where it resolves the
    contradiction. A guardrail retry replaces the chain answer where its
    available answer states another opportunity rate or stage probability;
    one that restates the original values replaces nothing. A cross-check
    replaces the direct answer where it changes a judgment other than the
    direct EAL, which is never simulated (``crosscheck_revises``).
    """
    counts = {
        kind: {"cases_asked": 0, "answers_replaced": 0} for kind in LEAF_FOLLOW_UPS
    }
    for artifact, node_ids in captures:
        selected = set(node_ids)
        for case in artifact.cases:
            if case.node_id not in selected:
                continue
            check = case.eal_crosscheck
            for kind, asked, replaced in (
                (
                    "severity_retry",
                    case.severity_retry is not None,
                    severity_retry_replaces(case),
                ),
                (
                    "chain_guardrail",
                    case.chain_guardrail is not None,
                    chain_guardrail_replaces(case),
                ),
                (
                    "eal_crosscheck",
                    check is not None and check.reconciliation is not None,
                    crosscheck_revises(case),
                ),
            ):
                counts[kind]["cases_asked"] += asked
                counts[kind]["answers_replaced"] += replaced
    return counts


def _revised(case: ContextualRiskCase) -> list[str]:
    crosscheck = _driving_crosscheck(case)
    if crosscheck is None:
        return []
    return [row.judgment_id for row in adopted_crosscheck(crosscheck).revised]


def eal_crosscheck_summary(
    captures: Iterable[
        tuple[
            ContextualRiskArtifact,
            Mapping[str, str],
            Mapping[tuple[str, float, tuple[str, str, int]], float],
        ]
    ],
) -> dict[str, Any]:
    """Every leaf case's EAL cross-check, by leaf and status.

    Each capture carries its leaves' scenario IDs, by node ID, and the
    reference weight of the states that read each case the run weights, by
    the case's key (node ID, anchor score and respondent key); a case that
    several respondents' states read weighs as all of them. A leaf's
    ``weighted_direct_eal`` and ``weighted_implied_eal`` are the sums over its
    cases, in every capture, of that weight times the case's direct and
    implied EAL means after any revision: the respondent's direct judgment
    and what its own count and severity answers imply, weighted as the run
    weights its states, so they stand beside the leaf's EAL in the run. One
    is None where a weighted case has no such mean.

    Case values are stored at four significant figures. Loss-level counts are
    shown before and after the direct-answer cross-check.
    """
    statuses: Counter[str] = Counter(dict.fromkeys(EAL_CROSSCHECK_STATUSES, 0))
    revised: Counter[str] = Counter()
    leaves: dict[str, list[dict[str, Any]]] = {}
    scenarios: dict[str, str] = {}
    weighted: dict[str, dict[str, float | None]] = {}
    triggered: Counter[str] = Counter(dict.fromkeys(CROSSCHECK_TRIGGERS, 0))
    for artifact, node_ids, weights in captures:
        for case in sorted(artifact.cases, key=lambda row: row.key()):
            if case.node_id not in node_ids:
                continue
            scenario_id = node_ids[case.node_id]
            if scenarios.setdefault(case.node_id, scenario_id) != scenario_id:
                raise ValueError("a leaf names one scenario in every capture")
            status = eal_crosscheck_status(case)
            statuses[status] += 1
            check = case.eal_crosscheck
            crosscheck = None if check is None else check.reconciliation
            direct = None if check is None else check.direct_eal_mean
            before = None if check is None else check.implied.eal_mean
            if (
                crosscheck is not None
                and crosscheck.answer is not None
                and crosscheck.answer.revises
            ):
                direct_after = crosscheck.direct_eal_mean_after
                after = (
                    None
                    if crosscheck.implied_after is None
                    else crosscheck.implied_after.eal_mean
                )
            else:
                direct_after, after = direct, before
            judgments = _revised(case)
            revised.update(judgments)
            triggers = list(case_triggers(case))
            triggered.update(triggers)
            totals = weighted.setdefault(case.node_id, {"direct": 0.0, "implied": 0.0})
            weight = weights.get(case.key(), 0.0)
            if weight > 0:
                for name, value in (("direct", direct_after), ("implied", after)):
                    total = totals[name]
                    totals[name] = (
                        None
                        if total is None or value is None
                        else total + weight * value
                    )
            leaves.setdefault(case.node_id, []).append(
                {
                    "anchor_score": case.anchor_score,
                    "respondent": list(case.respondent.key()),
                    "status": status,
                    "triggers": triggers,
                    "eal_crosscheck_unresolved": status in ("declined", "unresolved"),
                    "direct_eal": direct,
                    "implied_eal_before": before,
                    "implied_to_direct_before": eal_ratio(direct, before),
                    "direct_eal_after": direct_after,
                    "implied_eal_after": after,
                    "implied_to_direct_after": eal_ratio(direct_after, after),
                    "revised": judgments,
                    "loss_level_counts": case_loss_level_counts(case),
                }
            )
    cases = sum(statuses.values())
    return {
        "crosscheck_ratio": EAL_CROSSCHECK_RATIO,
        "cases": cases,
        "cases_by_status": dict(statuses),
        "cases_by_trigger": dict(triggered),
        "eal_crosscheck_unresolved": statuses["declined"] + statuses["unresolved"],
        "eal_crosscheck_exhausted": statuses["exhausted"],
        "revised_by_judgment": dict(sorted(revised.items())),
        "loss_level_counts": {
            "fixed_levels": list(FIXED_LOSS_LEVELS),
            "reported_quantiles": list(REPORTED_QUANTILES),
            "note": LOSS_LEVEL_COUNTS_NOTE,
        },
        "by_leaf": {
            node_id: {
                "scenario_id": scenarios[node_id],
                "cases_by_status": dict(
                    sorted(Counter(row["status"] for row in rows).items())
                ),
                "weighted_direct_eal": weighted[node_id]["direct"],
                "weighted_implied_eal": weighted[node_id]["implied"],
                "cases": rows,
            }
            for node_id, rows in sorted(leaves.items())
        },
    }
