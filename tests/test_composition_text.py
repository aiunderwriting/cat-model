"""Shared authoring text expands before instruments and snapshots are built."""

import pytest
from pydantic import ValidationError

from catmodel.scenarios.composition_text import ACCOUNTING_TEXT
from catmodel.scenarios.loader import CompositionCellDef


def _cell(**fields) -> CompositionCellDef:
    return CompositionCellDef.model_validate(
        {
            "cell_id": "provider_resource_cost",
            "loss_form": "Provider resources.",
            "nature": "resource_cost",
            "subject": "provider",
            **fields,
        }
    )


@pytest.mark.parametrize(
    "text",
    [
        "{{accounting:unknown}}",
        "{{accounting:}}",
        "{{accounting:provider_resources}",
        "Additional scope: {{accounting:provider_resources}}",
        "{{accounting:provider_resources}} Additional scope.",
    ],
)
@pytest.mark.parametrize("field", ["loss_form", "rationale"])
def test_invalid_references_cannot_reach_loss_forms_or_rationales(text, field):
    with pytest.raises(ValidationError, match="accounting"):
        _cell(**{field: text})


def test_whole_field_references_expand_to_the_shared_text():
    cell = _cell(
        loss_form="{{accounting:provider_resources}}",
        rationale="{{accounting:appropriation_rationale}}",
    )
    assert cell.loss_form == ACCOUNTING_TEXT["provider_resources"]
    assert cell.rationale == ACCOUNTING_TEXT["appropriation_rationale"]


def test_literal_scope_is_not_inferred_from_classification():
    text = "Customer restoration resources only; exclude the provider's own costs."
    cell = _cell(cell_id="customer_restoration", loss_form=text, subject="third_party")
    assert cell.loss_form == text
    assert cell.subject == "third_party"
