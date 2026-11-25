"""Read back what a rendered question shows a respondent."""

import json
from typing import Any

from catmodel.elicitation.question_text import (
    ANSWER_SCHEMA_HEADING,
    ANSWER_SCHEMA_INTRO,
)


def answer_schema_of(text: str) -> dict[str, Any]:
    """The answer schema a rendered question shows, which a follow-up's own
    section may follow."""
    shown = text.split(ANSWER_SCHEMA_HEADING + ANSWER_SCHEMA_INTRO, 1)[1]
    return json.JSONDecoder().raw_decode(shown)[0]
