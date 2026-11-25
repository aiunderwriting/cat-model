"""A chain's exposure base: an authored unit with a stipulated count."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from catmodel.contracts.contextual_risk import ChainExposure, ContextualRiskInstrument
from catmodel.elicitation.contextual_frequency import chain_prompts
from catmodel.elicitation.contextual_risk import contextual_prompts
from catmodel.scenarios.loader import ExposureBaseDef
from test_support.contextual_risk import contextual_instrument


def _base(count: dict) -> ExposureBaseDef:
    return ExposureBaseDef.model_validate(
        {
            "unit": "customer organization shipping AI-written code to production",
            "basis": "judgment",
            "count": {"rationale": "Fixture.", "source": "Fixture.", **count},
            "derivation": "About 0.15 times the paying business customers.",
        }
    )


def test_a_derived_count_is_the_scale_level_times_the_factor_at_three_figures():
    base = _base({"kind": "derived", "scale_id": "customers", "factor": 0.15})
    exposure = ChainExposure.of(base, {"customers": 1_234_567.0})
    assert exposure.count == 185_000
    assert (exposure.scale_id, exposure.scale_level, exposure.factor) == (
        "customers",
        1_234_567.0,
        0.15,
    )
    # A faster trajectory's larger scale moves the count with it.
    assert ChainExposure.of(base, {"customers": 2_000_000.0}).count == 300_000
    assert exposure.view() == {
        "unit": base.unit,
        "count": 185_000,
        "derivation": base.derivation,
    }


def test_a_fixed_count_ignores_the_scales():
    base = _base({"kind": "fixed", "count": 12.345})
    exposure = ChainExposure.of(base, {})
    assert exposure.count == 12.3
    assert exposure.scale_id is None


@pytest.mark.parametrize(
    "count",
    [
        {"kind": "derived", "scale_id": "customers", "factor": 0},
        {"kind": "derived", "scale_id": "customers", "factor": float("inf")},
        {"kind": "fixed", "count": -1},
        {"kind": "fixed", "count": 3, "scale_id": "customers"},
        {"kind": "derived", "scale_id": "customers", "factor": 1, "source": " "},
    ],
)
def test_invalid_counts_are_refused(count):
    with pytest.raises(ValidationError):
        _base(count)


def test_a_derived_count_must_read_a_scale_use_population_shows():
    payload = contextual_instrument().model_dump(mode="json")
    chain = payload["nodes"][0]["frequency_chain"]
    chain["exposure"]["count"]["scale_id"] = "unstipulated_scale"
    with pytest.raises(ValidationError, match="unstipulated_scale"):
        ContextualRiskInstrument.model_validate(payload)
    chain["exposure"]["count"] = {
        "kind": "fixed",
        "count": 40,
        "rationale": "Fixture.",
        "source": "Fixture.",
    }
    instrument = ContextualRiskInstrument.model_validate(payload)
    node = instrument.nodes[0]
    assert node.frequency_chain is not None
    assert instrument.chain_exposure(node.frequency_chain).count == 40


def test_the_chain_question_shows_the_base_as_a_number_with_its_derivation():
    instrument = contextual_instrument()
    prompt = chain_prompts(instrument)[0]
    text = prompt.render()
    assert (
        'Exposure base: one base unit is "block of 1,000 synthetic users"; the '
        "stipulated count is 1 base units. The stipulated synthetic users divided "
        "by 1,000."
    ) in text
    assert "opportunities in the study year's risk set per base unit" in text
    assert "It is multiplied by the stipulated count, 1, to give the expected" in text
    # The factor, scale, rationale, source and authoring status are authoring
    # content, not shown.
    for hidden in (
        "Synthetic base; software fixture only.",
        "Software fixture only.",
        "users",
        "authored",
    ):
        assert hidden not in text.split("Exposure base", 1)[1].split("\n", 1)[1]
    # The direct question shows no chain or base.
    direct = contextual_prompts(instrument)[0].render()
    assert "Exposure base" not in direct and "opportunit" not in direct
