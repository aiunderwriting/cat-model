"""Context-bound nodes follow their repository scenario definitions."""

from typing import cast

import pytest

from catmodel.collection.contextual_collection import run_contextual_collection
from catmodel.config import ModelProfile
from catmodel.elicitation.contextual_frequency import chain_prompts
from catmodel.elicitation.contextual_risk import contextual_prompts
from catmodel.evidence.source_bundle import capture_source_bundle_snapshot
from catmodel.llm.models import ModelSpec
from catmodel.llm.transport import LLMProvider, ProviderResponse
from catmodel.resources import scenarios_path
from catmodel.setup.contextual_inputs import (
    load_contextual_collection,
    validate_contextual_sources,
)
from test_support.capability_sources import write_source_bundle
from test_support.contextual_risk import (
    contextual_instrument,
    fixture_response,
    scenario_node_answer,
)


class AnswerProvider:
    provider_name = "offline-fixture"
    reasoning_effort = "low"
    model_id = "fixture-model"

    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    async def complete(self, **kwargs):
        self.calls += 1
        return ProviderResponse(
            text=fixture_response(kwargs["user_prompt"], self.answer),
            model_id=self.model_id,
            provider_name=self.provider_name,
        )


def use_provider(monkeypatch, provider):
    import catmodel.config
    import catmodel.llm.factory

    monkeypatch.setattr(
        catmodel.config,
        "load_model_profile",
        lambda **kwargs: ModelProfile(
            models=(
                ModelSpec(
                    provider="openai",
                    model_id=provider.model_id,
                    reasoning_effort="low",
                    max_output_tokens=100_000,
                ),
            )
        ),
    )
    monkeypatch.setattr(catmodel.config.Settings, "provider_api_keys", lambda self: {})
    monkeypatch.setattr(
        catmodel.llm.factory,
        "build_providers",
        lambda *args, **kwargs: [cast(LLMProvider, provider)],
    )


def collect(instrument, source, run, **kwargs):
    return run_contextual_collection(
        instrument=instrument,
        source_bundle=source,
        run_directory=run,
        profile="test",
        **kwargs,
    )


_SCENARIO = "service_outage_degradation"


def _outage_instrument(tmp_path, monkeypatch):
    from catmodel.setup.contextual_inputs import build_contextual_instrument
    from test_support.capability_sources import authored_source_files

    source = write_source_bundle(tmp_path / "source", files=authored_source_files())
    base = contextual_instrument()
    payload = base.model_dump(
        mode="json", exclude={"nodes", "respondents", "source_bundle_manifest_sha256"}
    )
    instrument = build_contextual_instrument(
        payload,
        source_bundle=source,
        respondents=base.respondents,
        scenario_ids=(_SCENARIO,),
    )
    return source, instrument


def test_explicit_node_must_match_its_scenario_definition(tmp_path, monkeypatch):
    source, instrument = _outage_instrument(tmp_path, monkeypatch)
    snapshot = capture_source_bundle_snapshot(source)
    validate_contextual_sources(instrument, snapshot)
    node = instrument.nodes[0]
    for field, value in (
        (
            "qualified_count_definition",
            "Expected annual count of customer-hours of unavailability.",
        ),
        (
            "population",
            node.population.model_copy(
                update={"initiating_unit": "One customer-hour."}
            ),
        ),
    ):
        explicit = instrument.model_copy(
            update={"nodes": (node.model_copy(update={field: value}),)}
        )
        message = (
            f"contextual node for scenario '{_SCENARIO}' differs from its authored "
            f"scenario definition in {field}"
        )
        with pytest.raises(ValueError, match=message):
            validate_contextual_sources(explicit, snapshot)
        with pytest.raises(ValueError, match=message):
            collect(explicit, source, tmp_path / f"refused-{field}", dry_run=True)
        assert not (tmp_path / f"refused-{field}").exists()


def test_capture_under_an_earlier_definition_cannot_be_imported(tmp_path, monkeypatch):
    source, instrument = _outage_instrument(tmp_path, monkeypatch)
    provider = AnswerProvider(scenario_node_answer(instrument.nodes[0]))
    use_provider(monkeypatch, provider)
    run = tmp_path / "run"
    assert collect(instrument, source, run) is not None
    # One direct and one chain question per node, coordinate and respondent.
    assert provider.calls == len(contextual_prompts(instrument)) + len(
        chain_prompts(instrument)
    )
    assert load_contextual_collection(run, source)[0] == instrument
    # The scenario's authored counting identity changes after collection.
    import catmodel.setup.contextual_inputs as contextual_inputs

    current = contextual_inputs.load_scenarios(scenarios_path())
    revised = []
    for row in current:
        if row.id == _SCENARIO:
            row = row.model_copy(
                update={
                    "frequency_semantics": row.frequency_semantics.model_copy(
                        update={"counting_identity": "A revised counting identity."}
                    )
                }
            )
        revised.append(row)
    monkeypatch.setattr(
        "catmodel.setup.contextual_inputs.load_scenarios",
        lambda *_args, **_kwargs: revised,
    )
    with pytest.raises(
        ValueError,
        match=f"scenario '{_SCENARIO}' differs from its authored scenario "
        "definition in population",
    ):
        load_contextual_collection(run, source)
    # The refused import sends no request.
    assert provider.calls == len(contextual_prompts(instrument)) + len(
        chain_prompts(instrument)
    )


def test_audit_text_revisions_leave_earlier_nodes_valid(tmp_path, monkeypatch):
    """Only what a question shows or code computes binds a node to its scenario:
    the exposure base's basis, rationale and source are audit context."""
    from catmodel.scenarios.loader import load_scenarios
    from catmodel.setup.contextual_inputs import require_scenario_nodes

    _, instrument = _outage_instrument(tmp_path, monkeypatch)
    assert instrument.capability_access == "public"
    scenario = next(
        row for row in load_scenarios(scenarios_path()) if row.id == _SCENARIO
    )
    exposure = scenario.event_chain.exposure

    def with_public_base(**updates):
        base = exposure.public.model_copy(update=updates)
        chain = scenario.event_chain.model_copy(
            update={"exposure": exposure.model_copy(update={"public": base})}
        )
        return {_SCENARIO: scenario.model_copy(update={"event_chain": chain})}

    audited = with_public_base(
        basis="normalisation",
        count=exposure.public.count.model_copy(
            update={
                "rationale": "Revised audit rationale.",
                "source": "Revised source.",
            }
        ),
    )
    require_scenario_nodes(instrument, audited)
    for asked in (
        with_public_base(derivation="A revised derivation the question shows."),
        with_public_base(unit="revised base unit"),
    ):
        with pytest.raises(ValueError, match="in frequency_chain"):
            require_scenario_nodes(instrument, asked)
