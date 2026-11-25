"""Shared vocabulary of assumption governance.

The source-bundle assumption register and each scenario's severity governance
and composition classify an assumption by its status, materiality and source
with these terms.
"""

from enum import StrEnum
from typing import Literal

AssumptionStatus = Literal["OK", "INITIAL_ESTIMATE", "PLACEHOLDER", "GAP"]
AssumptionMateriality = Literal["low", "medium", "high", "critical"]


class SourceClassification(StrEnum):
    """Where an assumption's value comes from.

    ``human_expert`` requires documented external expert or panel input;
    AIUC-authored values remain ``internal_judgment`` even when informed by
    published or LLM-assisted analysis.
    """

    EMPIRICAL_FIT = "empirical_fit"
    EMPIRICAL_WITH_HUMAN_OVERLAY = "empirical_with_human_overlay"
    HUMAN_EXPERT = "human_expert"
    INTERNAL_JUDGMENT = "internal_judgment"
    LLM_PRELIMINARY = "llm_preliminary"
