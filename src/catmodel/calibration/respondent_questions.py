"""Independent known-answer questions for calibrating respondent spread."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar

from catmodel.contracts.elicitation import Respondent
from catmodel.contracts.respondent_calibration import (
    CalibrationAnswer,
    CalibrationInstrument,
    CalibrationSeed,
)
from catmodel.elicitation.contextual_risk import (
    CONTEXTUAL_SYSTEM_PROMPT,
    RenderedOnce,
    check_cited_cards,
)
from catmodel.elicitation.question_text import answer_schema_text
from catmodel.hashing import canonical_json_sha256
from catmodel.json_io import strict_json_loads
from catmodel.scenarios.base_rates import DatedEvidenceText


@dataclass(frozen=True)
class CalibrationQuestion(RenderedOnce):
    """One seed question for one respondent, rendered and hashed once."""

    system_prompt: ClassVar[str] = CONTEXTUAL_SYSTEM_PROMPT

    instrument: CalibrationInstrument
    seed: CalibrationSeed
    respondent: Respondent

    def __post_init__(self) -> None:
        if self.seed not in self.instrument.seeds:
            raise ValueError("calibration question seed is outside its instrument")
        if self.respondent not in self.instrument.respondents:
            raise ValueError(
                "calibration question respondent is outside its instrument"
            )

    @property
    def call_id(self) -> str:
        return canonical_json_sha256(["respondent-calibration", self.seed.seed_id])

    def card(self) -> DatedEvidenceText:
        history = "\n".join(
            f"{row.period}: {row.value:g}" for row in self.seed.evidence.history
        )
        text = self.seed.evidence.text
        if history:
            text += "\n" + history
        return DatedEvidenceText(
            card_id=self.seed.seed_id,
            observation_through_utc=max(
                (row.period_end for row in self.seed.evidence.history),
                default=self.seed.information_date,
            ),
            availability_date_utc=self.seed.information_date,
            availability_basis=(
                "The seed's information date bounds its evidence; every history "
                "row ends on or before it."
            ),
            text=text,
            sources=(self.seed.source.url,),
        )

    def _render(self) -> str:
        seed = self.seed
        # The quantity is a complete definition, often a sentence; state it on
        # its own line rather than inside another sentence.
        condition = (
            f"Quantity: {seed.quantity.rstrip().rstrip('.')}.\n"
            f"Window: {seed.window_start.isoformat()} through "
            f"{seed.window_end.isoformat()}, inclusive, as {seed.source.name} "
            "records it.\n"
            f"Your information date is {seed.information_date.isoformat()}.\n"
            "Evidence history counts records as the source records them at retrieval, "
            "for periods ending before the information date; later revisions to those "
            "periods may be included. Judge as of the information date."
        )
        if seed.unit == "count":
            estimate_task = (
                "Supply rate p5/p25/p50/p75/p95 for epistemic uncertainty about the "
                "EXPECTED value of this count in the window. These are not "
                "quantiles of the realized count: the realized count adds chance variation "
                "around its expected value. Count every unit the source records in the "
                "window under its own definition and dating rule."
            )
        else:
            estimate_task = (
                "Supply rate p5/p25/p50/p75/p95 for your uncertainty about this "
                "amount, in USD; for this question rate holds that amount, not a count."
            )
        card = self.card()
        context = {
            "seed_id": seed.seed_id,
            "quantity": seed.quantity,
            "unit": seed.unit,
            "window_start": seed.window_start.isoformat(),
            "window_end": seed.window_end.isoformat(),
            "information_date": seed.information_date.isoformat(),
            "source": seed.source.model_dump(mode="json"),
        }
        return "\n\n".join(
            (
                condition,
                estimate_task,
                "If the quantity cannot be estimated, return status=unavailable with "
                "only unavailable_reason and reasoning. Supply unavailable_reason "
                "only with status=unavailable; with status=available, set "
                "unavailable_reason to null or omit it, and give your explanation in "
                "reasoning.",
                "Describe the judgment and uncertainty in reasoning. evidence_used may "
                "be empty; if used, it lists only these supplied card IDs: "
                f"{card.card_id}. A judgment without empirical evidence remains a "
                "judgment, not an observation.",
                "Frozen question context:\n"
                + json.dumps(context, ensure_ascii=False, indent=2),
                "Supplied evidence:\n" + card.render(),
                answer_schema_text(CalibrationAnswer.model_json_schema()),
            )
        )


def calibration_questions(
    instrument: CalibrationInstrument,
) -> tuple[CalibrationQuestion, ...]:
    """Return every seed/respondent pair: primary, control and secondary seeds."""
    return tuple(
        CalibrationQuestion(instrument, seed, respondent)
        for seed in instrument.seeds
        for respondent in instrument.respondents
    )


def parse_calibration_answer(
    text: str, question: CalibrationQuestion
) -> CalibrationAnswer:
    answer = CalibrationAnswer.model_validate(strict_json_loads(text))
    check_cited_cards(answer.reasoning, [question.card()])
    return answer


def calibration_request(question: CalibrationQuestion) -> dict[str, object]:
    """The stateless frozen-call request shape used by contextual collection."""
    return {
        "call_id": question.call_id,
        "respondent": question.respondent.model_dump(mode="json"),
        "prompt_sha256": question.prompt_sha256(),
        "system_prompt": question.system_prompt,
        "user_prompt": question.render(),
    }


__all__ = [
    "CalibrationQuestion",
    "calibration_questions",
    "calibration_request",
    "parse_calibration_answer",
]
