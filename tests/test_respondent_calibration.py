"""Contracts, prompts and frozen fake-provider collection for calibration."""

from __future__ import annotations

import asyncio
import calendar
import hashlib
import json
import re
from datetime import date
from functools import partial
from typing import Literal

import pytest
from pydantic import ValidationError

from catmodel.calibration.respondent_questions import (
    CalibrationQuestion,
    calibration_questions,
    calibration_request,
    parse_calibration_answer,
)
from catmodel.calibration.respondent_spread import (
    calibration_treatments,
)
from catmodel.collection.frozen_call_collection import collect_frozen_call
from catmodel.collection.respondent_calibration import (
    build_calibration_instrument,
    fit_calibration,
    run_calibration_collection,
    validate_calibration_capture,
)
from catmodel.config import ModelProfile, ProfileSettings
from catmodel.contracts.contextual_risk import CONTEXTUAL_SCHEMA_VERSION
from catmodel.contracts.elicitation import (
    ELICITATION_ORDER,
    Quantiles,
    Reasoning,
    Respondent,
)
from catmodel.contracts.respondent_calibration import (
    CalibrationAnswer,
    CalibrationArtifact,
    CalibrationInstrument,
    CalibrationSeed,
    CalibrationSeedEvidence,
    CalibrationSeedHistory,
    CalibrationSeedSet,
    CalibrationSeedSource,
    ResolutionProtocol,
    SeedResolution,
    load_calibration_seeds,
    load_seed_resolutions,
    resolution_script_digests,
    resolved_outcomes,
)
from catmodel.elicitation.question_text import ANSWER_SCHEMA_HEADING
from catmodel.frozen_calls import PARSE_ATTEMPTS, frozen_call_key
from catmodel.hashing import canonical_json_sha256
from catmodel.llm.models import ModelSpec, ReasoningEffort
from catmodel.llm.transport import ProviderError, ProviderResponse
from scripts import respondent_calibration as calibration_script
from test_support.question_text import answer_schema_of
from test_support.version_markers import unstated_markers


def _seed(
    seed_id: str = "seed-primary",
    *,
    role: Literal["primary", "control", "secondary"] = "primary",
    block_id: str = "block-1",
    unit: Literal["count", "usd"] = "count",
    can_zero: bool = True,
) -> CalibrationSeed:
    if role == "control":
        window_start = date(2026, 3, 1)
        window_end = date(2026, 3, 31)
        information_date = date(2026, 2, 28)
        history_end = date(2025, 12, 31)
    else:
        window_start = date(2026, 7, 1)
        window_end = date(2026, 7, 31)
        information_date = date(2026, 6, 30)
        history_end = date(2026, 3, 31)
    return CalibrationSeed(
        seed_id=seed_id,
        role=role,
        block_id=block_id,
        quantity="records matching the source's rule",
        unit=unit,
        window_start=window_start,
        window_end=window_end,
        information_date=information_date,
        source=CalibrationSeedSource(
            name="Example registry", url="https://example.test/records"
        ),
        evidence=CalibrationSeedEvidence(
            text="The supplied snapshot ends before the window begins.",
            history=(
                CalibrationSeedHistory(
                    period="2026-Q1", period_end=history_end, value=12
                ),
            ),
        ),
        resolution_script="scripts/resolve.py",
        can_resolve_to_zero=can_zero,
    )


def _fit_seed(seed_id: str = "seed-primary", *, block_id: str = "block-1"):
    seed = _seed(seed_id, block_id=block_id)
    latest_month = seed.information_date.year * 12 + seed.information_date.month - 1
    history = []
    for month_index in range(latest_month - 11, latest_month + 1):
        year, month_offset = divmod(month_index, 12)
        month = month_offset + 1
        history.append(
            CalibrationSeedHistory(
                period=f"{year:04d}-{month:02d}",
                period_end=date(year, month, calendar.monthrange(year, month)[1]),
                value=100,
            )
        )
    return seed.model_copy(
        update={
            "evidence": CalibrationSeedEvidence(
                text="The supplied snapshot ends before the window begins.",
                history=tuple(history),
            )
        }
    )


def _resolution_protocol(
    digest: str = "a" * 64,
) -> ResolutionProtocol:
    return ResolutionProtocol(
        scripts={"scripts/resolve.py": digest},
        snapshot_rule="Use the dated source snapshot after the window closes.",
        replacement_rule="Replace only when the source republishes its frozen snapshot.",
    )


def _seed_set(seeds: tuple[CalibrationSeed, ...]) -> CalibrationSeedSet:
    return CalibrationSeedSet(
        seeds=seeds,
        latest_training_cutoff=date(2026, 5, 31),
        resolution_protocol=_resolution_protocol(),
    )


def _respondents(
    models: tuple[str, ...] = ("fixture-model",), samples: int = 1
) -> tuple[Respondent, ...]:
    return tuple(
        Respondent(
            respondent_id=f"{model}:{sample}", model_id=model, sample_index=sample
        )
        for model in models
        for sample in range(samples)
    )


def _instrument(
    seeds: tuple[CalibrationSeed, ...] | None = None,
    respondents: tuple[Respondent, ...] | None = None,
    *,
    resolution_digest: str = "a" * 64,
) -> CalibrationInstrument:
    return CalibrationInstrument(
        seeds=seeds or (_seed(),),
        respondents=respondents or _respondents(),
        latest_training_cutoff=date(2026, 5, 31),
        resolution_protocol=_resolution_protocol(resolution_digest),
        risk_contract_version=CONTEXTUAL_SCHEMA_VERSION,
    )


def _reasoning(seed_id: str) -> Reasoning:
    return Reasoning(
        median_reason="The median follows the supplied dated record.",
        low_case="A lower result follows the source's inclusion rule.",
        high_case="Late records can move the final count upward.",
        method="direct_judgment",
        evidence_used=(seed_id,),
    )


def _answer_payload(seed_id: str) -> dict[str, object]:
    return {
        "status": "available",
        "rate": {"p5": 1, "p25": 2, "p50": 4, "p75": 8, "p95": 16},
        "reasoning": _reasoning(seed_id).model_dump(mode="json"),
    }


def _seeds_yaml(*, count: int = 1, digest: str = "a" * 64) -> str:
    rows = []
    for index in range(count):
        seed = _fit_seed(f"seed-{index}", block_id=f"block-{index}")
        rows.append(seed.model_dump(mode="json"))
    return json.dumps(
        {
            "seeds": rows,
            "latest_training_cutoff": "2026-05-31",
            "resolution_protocol": _resolution_protocol(digest).model_dump(mode="json"),
        },
        indent=2,
    )


class _Provider:
    provider_name = "openai"
    reasoning_effort: ReasoningEffort = "low"

    def __init__(
        self, model_id: str, *, invalid_seed_ids: frozenset[str] = frozenset()
    ):
        self.model_id = model_id
        self.invalid_seed_ids = invalid_seed_ids
        self.calls = 0
        self.prompts: list[str] = []

    async def complete(self, **request):
        self.calls += 1
        prompt = request["user_prompt"]
        self.prompts.append(prompt)
        match = re.search(r'"seed_id":\s*"([^"]+)"', prompt)
        assert match is not None
        seed_id = match.group(1)
        text = (
            "not a JSON answer"
            if seed_id in self.invalid_seed_ids
            else json.dumps(_answer_payload(seed_id))
        )
        return ProviderResponse(
            text=text,
            model_id=self.model_id,
            provider_name=self.provider_name,
            input_tokens=50,
            output_tokens=20,
        )


def _profile(models: tuple[str, ...] = ("fixture-model",), samples: int = 1):
    return ModelProfile(
        models=tuple(
            ModelSpec(
                provider="openai",
                model_id=model,
                reasoning_effort="low",
                max_output_tokens=2048,
            )
            for model in models
        ),
        settings=ProfileSettings(samples_per_question=samples),
    )


def _patch_collection(monkeypatch, profile, providers):
    monkeypatch.setattr("catmodel.config.load_model_profile", lambda **_kwargs: profile)
    monkeypatch.setattr(
        "catmodel.llm.factory.build_providers", lambda *_args, **_kwargs: providers
    )


def _collect_fake_run(tmp_path, monkeypatch, name="calibration-run", instrument=None):
    instrument = instrument or _instrument()
    profile = _profile()
    _patch_collection(monkeypatch, profile, [_Provider("fixture-model")])
    run = tmp_path / name
    run_calibration_collection(
        instrument=instrument,
        run_directory=run,
        profile="test",
    )
    plan = json.loads((run / "collection-plan.json").read_text())
    return run, plan


def test_seed_contract_loaders_and_resolutions_are_separate(tmp_path):
    seeds_path = tmp_path / "seeds.yaml"
    seeds_path.write_text(_seeds_yaml(count=2), encoding="utf-8")
    loaded = load_calibration_seeds(seeds_path)
    assert [row.seed_id for row in loaded.seeds] == ["seed-0", "seed-1"]
    assert "value" not in CalibrationSeed.model_fields

    resolutions_path = tmp_path / "resolutions.yaml"
    resolutions_path.write_text(
        json.dumps(
            {
                "resolutions": [
                    {
                        "seed_id": "seed-0",
                        "value": 3,
                        "retrieved_at": "2026-11-01",
                        "snapshot_url": "https://example.test/snapshot",
                        "snapshot_sha256": "a" * 64,
                        "script": "scripts/resolve.py",
                        "notes": "",
                    },
                    {
                        "seed_id": "seed-1",
                        "value": 5,
                        "retrieved_at": "2026-11-01T12:00:00Z",
                        "snapshot_url": "https://example.test/snapshot",
                        "snapshot_sha256": "b" * 64,
                        "script": "scripts/resolve.py",
                        "notes": "",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    resolved = load_seed_resolutions(resolutions_path)
    assert resolved_outcomes(loaded, resolved) == {"seed-0": 3.0, "seed-1": 5.0}


def test_seed_validation_enforces_dates_units_blocks_and_extra_forbid():
    invalid_dates = _seed().model_dump(mode="json")
    invalid_dates["window_start"] = "2026-08-01"
    with pytest.raises(ValidationError, match="window_start"):
        CalibrationSeed.model_validate(invalid_dates)
    with pytest.raises(ValidationError, match="primary and control"):
        _seed(unit="usd")
    with pytest.raises(ValidationError, match="at most three"):
        _seed_set(tuple(_seed(f"p{i}", block_id="same") for i in range(4)))
    leaked_history = _seed().model_dump(mode="json")
    leaked_history["evidence"]["history"][0]["period_end"] = "2026-07-01"
    with pytest.raises(ValidationError, match="on or before information_date"):
        CalibrationSeed.model_validate(leaked_history)
    history_in_window = _seed().model_dump(mode="json")
    history_in_window["information_date"] = "2026-07-01"
    history_in_window["evidence"]["history"][0]["period_end"] = "2026-07-01"
    with pytest.raises(ValidationError, match="before window_start"):
        CalibrationSeed.model_validate(history_in_window)
    with pytest.raises(ValidationError, match="primary seed window_start"):
        CalibrationSeedSet(
            seeds=(_seed(),),
            latest_training_cutoff=date(2026, 7, 1),
            resolution_protocol=_resolution_protocol(),
        )
    with pytest.raises(ValidationError, match="control seed window_end"):
        CalibrationSeedSet(
            seeds=(_seed("control", role="control"),),
            latest_training_cutoff=date(2026, 2, 28),
            resolution_protocol=_resolution_protocol(),
        )
    payload = _seed().model_dump(mode="json") | {"value": 9}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CalibrationSeed.model_validate(payload)
    assert CalibrationSeedEvidence.model_fields["history"].is_required()
    assert CalibrationSeedHistory.model_fields["period_end"].is_required()
    assert SeedResolution.model_fields["notes"].is_required()
    assert CalibrationArtifact.model_fields["cases"].is_required()


def test_a_holdout_role_and_an_unresolved_seed_are_refused(tmp_path):
    """Every seed is primary, control or secondary and every one is resolved:
    a seed file that names a holdout, a resolution without a value and one
    that defers its value to a later date are refused where they are read."""
    seed_file = json.loads(_seeds_yaml(count=2))
    seed_file["seeds"][1]["role"] = "holdout"
    seeds_path = tmp_path / "seeds.yaml"
    seeds_path.write_text(json.dumps(seed_file), encoding="utf-8")
    with pytest.raises(ValidationError) as holdout:
        load_calibration_seeds(seeds_path)
    assert [(error["loc"], error["type"]) for error in holdout.value.errors()] == [
        (("seeds", 1, "role"), "literal_error")
    ]
    instrument = _instrument().model_dump(mode="json")
    instrument["seeds"][0]["role"] = "holdout"
    with pytest.raises(ValidationError, match="literal_error"):
        CalibrationInstrument.model_validate(instrument)

    resolved: dict[str, object] = {
        "seed_id": "seed-0",
        "value": 3,
        "retrieved_at": "2026-11-01",
        "snapshot_url": "https://example.test/snapshot",
        "snapshot_sha256": "a" * 64,
        "script": "scripts/resolve.py",
        "notes": "",
    }
    resolutions_path = tmp_path / "resolutions.yaml"

    def refused(row: dict[str, object]) -> list[tuple[object, ...]]:
        """Where loading refuses ``row``, listed after a resolved seed."""
        resolutions_path.write_text(
            json.dumps({"resolutions": [resolved, row | {"seed_id": "seed-1"}]}),
            encoding="utf-8",
        )
        with pytest.raises(ValidationError) as error:
            load_seed_resolutions(resolutions_path)
        return [issue["loc"] for issue in error.value.errors()]

    assert refused(resolved | {"value": None}) == [("resolutions", 1, "value")]
    without_value = {key: item for key, item in resolved.items() if key != "value"}
    assert refused(without_value) == [("resolutions", 1, "value")]
    assert refused(resolved | {"resolve_after": "2027-01-01"}) == [
        ("resolutions", 1, "resolve_after")
    ]
    resolutions_path.write_text(
        json.dumps({"resolutions": [resolved]}), encoding="utf-8"
    )
    assert load_seed_resolutions(resolutions_path).resolutions[0].value == 3


def test_resolution_requirements_reject_missing_fractional_and_forbidden_zero():
    primary = _seed("primary", can_zero=False)
    control = _seed("control", role="control", can_zero=True)
    seeds = _seed_set((primary, control))

    def resolution(seed_id, value):
        return SeedResolution(
            seed_id=seed_id,
            value=value,
            retrieved_at=date(2026, 11, 1),
            snapshot_url="https://example.test/snapshot",
            snapshot_sha256="a" * 64,
            script="scripts/resolve.py",
            notes="",
        )

    with pytest.raises(ValueError, match="integer"):
        resolved_outcomes(seeds, (resolution("primary", 1.5), resolution("control", 0)))
    with pytest.raises(ValueError, match="cannot resolve to zero"):
        resolved_outcomes(seeds, (resolution("primary", 0), resolution("control", 0)))
    with pytest.raises(ValueError, match="exactly match"):
        resolved_outcomes(seeds, (resolution("primary", 1),))
    # A seed that can resolve to zero may.
    assert resolved_outcomes(
        seeds, (resolution("primary", 1), resolution("control", 0))
    ) == {"primary": 1.0, "control": 0.0}
    with pytest.raises(ValueError, match="different script"):
        resolved_outcomes(
            seeds,
            (
                resolution("primary", 1).model_copy(
                    update={"script": "scripts/other.py"}
                ),
                resolution("control", 0),
            ),
        )


def test_resolution_script_digests_report_missing_and_changed_files(tmp_path):
    script = tmp_path / "scripts" / "resolve.py"
    script.parent.mkdir()
    script.write_bytes(b"frozen resolver\n")
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    seed_set = CalibrationSeedSet(
        seeds=(_seed(),),
        latest_training_cutoff=date(2026, 5, 31),
        resolution_protocol=_resolution_protocol(digest),
    )
    assert resolution_script_digests(seed_set, tmp_path) == {}

    script.write_bytes(b"changed resolver\n")
    changed = hashlib.sha256(script.read_bytes()).hexdigest()
    assert resolution_script_digests(seed_set, tmp_path) == {
        "scripts/resolve.py": changed
    }
    script.unlink()
    assert resolution_script_digests(seed_set, tmp_path) == {"scripts/resolve.py": None}


def test_super_blocks_and_curation_sensitive_seeds_are_validated_and_carried(
    tmp_path,
):
    seed_rows = (
        _seed("primary-a", block_id="block-a"),
        _seed("primary-b", block_id="block-b"),
    )
    config = {
        "seeds": [row.model_dump(mode="json") for row in seed_rows],
        "latest_training_cutoff": "2026-05-31",
        "resolution_protocol": _resolution_protocol().model_dump(mode="json"),
        "super_blocks": {"same-source": ["block-a", "block-b"]},
        "curation_sensitive_seed_ids": ["primary-b"],
    }
    seeds_path = tmp_path / "seeds.json"
    seeds_path.write_text(json.dumps(config), encoding="utf-8")

    seed_set = load_calibration_seeds(seeds_path)
    assert seed_set.super_blocks == {"same-source": ("block-a", "block-b")}
    assert seed_set.curation_sensitive_seed_ids == ("primary-b",)
    instrument = build_calibration_instrument(seeds_path, _respondents())
    assert instrument.super_blocks == seed_set.super_blocks
    assert (
        instrument.curation_sensitive_seed_ids == seed_set.curation_sensitive_seed_ids
    )

    duplicate_block = config | {
        "super_blocks": {
            "same-source": ["block-a"],
            "other-source": ["block-a"],
        }
    }
    with pytest.raises(ValidationError, match="at most one super-block"):
        CalibrationSeedSet.model_validate(duplicate_block)

    nonprimary_seed = config | {"curation_sensitive_seed_ids": ["missing-seed"]}
    with pytest.raises(ValidationError, match="must name primary seeds"):
        CalibrationSeedSet.model_validate(nonprimary_seed)

    unknown_block = config | {"super_blocks": {"same-source": ["missing-block"]}}
    with pytest.raises(ValidationError, match="unknown primary blocks"):
        CalibrationSeedSet.model_validate(unknown_block)


def test_calibration_answer_status_rule_and_schema_are_a_direct_count_subset():
    available = CalibrationAnswer(
        status="available",
        rate=Quantiles(p5=1, p25=2, p50=3, p75=4, p95=5),
        reasoning=_reasoning("seed-primary"),
    )
    assert available.unavailable_reason is None
    with pytest.raises(ValidationError, match="rate is required"):
        CalibrationAnswer(status="available", reasoning=_reasoning("seed-primary"))
    with pytest.raises(ValidationError, match="rate is not allowed"):
        CalibrationAnswer(
            status="unavailable",
            rate=Quantiles(p5=1, p25=2, p50=3, p75=4, p95=5),
            unavailable_reason="No estimate is possible.",
            reasoning=_reasoning("seed-primary"),
        )
    schema = CalibrationAnswer.model_json_schema()
    assert set(schema["properties"]) == {
        "status",
        "rate",
        "unavailable_reason",
        "reasoning",
    }
    assert len(schema["anyOf"]) == 2


def test_calibration_questions_render_seed_evidence_and_parse_only_its_card():
    seed = _seed()
    instrument = _instrument((seed, _seed("control", role="control")))
    questions = calibration_questions(instrument)
    assert len(questions) == 2
    question = questions[0]
    card = question.card()
    assert card.card_id == seed.seed_id
    assert card.observation_through_utc == date(2026, 3, 31)
    assert card.availability_date_utc == seed.information_date
    assert card.sources == (seed.source.url,)
    assert "2026-Q1: 12" in card.text
    prompt = question.render()
    for required in (
        "Quantity: records matching the source's rule",
        "Window: 2026-07-01 through 2026-07-31, inclusive",
        "as Example registry records it",
        "Your information date is 2026-06-30",
        "history counts records as the source records them at retrieval",
        "later revisions to those periods may be included",
        "Judge as of the information date.",
        "EXPECTED value of this count in the window",
        "not quantiles of the realized count",
        "evidence_used may be empty",
        '"seed_id": "seed-primary"',
        ANSWER_SCHEMA_HEADING,
    ):
        assert required in prompt
    # The answer schema lists reasoning first and the quantiles median first.
    schema = answer_schema_of(prompt)
    assert list(schema["properties"]) == [
        "reasoning",
        "status",
        "unavailable_reason",
        "rate",
    ]
    assert list(schema["$defs"]["Quantiles"]["properties"]) == list(ELICITATION_ORDER)
    assert question.call_id == canonical_json_sha256(
        ["respondent-calibration", seed.seed_id]
    )
    assert "the supplied evidence is what is known as of that date" not in prompt
    assert "scripts/resolve.py" not in prompt
    request = calibration_request(question)
    assert set(request) == {
        "call_id",
        "respondent",
        "prompt_sha256",
        "system_prompt",
        "user_prompt",
    }
    assert request["prompt_sha256"] == question.prompt_sha256()
    assert (
        parse_calibration_answer(
            json.dumps(_answer_payload(seed.seed_id)), question
        ).status
        == "available"
    )
    invalid = _answer_payload(seed.seed_id)
    invalid["reasoning"] = _reasoning("not-the-supplied-card").model_dump(mode="json")
    with pytest.raises(ValueError, match="does not supply"):
        parse_calibration_answer(json.dumps(invalid), question)


def test_usd_seed_has_money_wording_and_history_still_uses_source_card():
    seed = _seed("usd", role="secondary", unit="usd")
    question = CalibrationQuestion(_instrument((seed,)), seed, _respondents()[0])
    prompt = question.render()
    assert "uncertainty about this amount, in USD" in prompt
    assert "rate holds that amount, not a count" in prompt
    assert "2026-Q1: 12" in question.card().render()


def test_empty_history_uses_information_date_as_observation_through():
    payload = _seed().model_dump(mode="json")
    payload["evidence"]["history"] = []
    seed = CalibrationSeed.model_validate(payload)
    question = CalibrationQuestion(_instrument((seed,)), seed, _respondents()[0])
    assert question.card().observation_through_utc == seed.information_date
    assert question.card().availability_date_utc == seed.information_date


def test_treatments_are_the_frozen_plan_s_models_and_parser_treatment():
    spec = ModelSpec(
        provider="openai",
        model_id="fixture-model",
        reasoning_effort="low",
        max_output_tokens=2048,
    )
    profile = ModelProfile(
        models=(spec,), settings=ProfileSettings(samples_per_question=1)
    )
    selected = [spec]
    execution = profile.execution_settings(selected, max_output_tokens=4096)
    plan = {
        "parse_attempts": PARSE_ATTEMPTS,
        "execution": {
            "models": [spec.model_dump(mode="json")],
            "settings": execution.model_dump(
                mode="json", exclude={"concurrency_by_model"}
            ),
        },
    }
    calibration = _instrument()
    treatments = calibration_treatments(calibration, plan)
    # The model's own output budget, not the collection's ceiling.
    assert [row.model_dump() for row in treatments.models] == [
        {
            "provider": "openai",
            "model_id": "fixture-model",
            "reasoning_effort": "low",
            "max_output_tokens": 2048,
        }
    ]
    assert treatments.samples_per_model == {
        "fixture-model": len(calibration.respondents)
    }
    assert treatments.parse_attempts == PARSE_ATTEMPTS
    assert treatments.risk_contract_version == calibration.risk_contract_version
    # A plan whose models differ from the instrument's respondents is refused.
    other = {
        **plan,
        "execution": {
            **plan["execution"],
            "models": [spec.model_dump(mode="json") | {"model_id": "other-model"}],
        },
    }
    with pytest.raises(ValueError, match="collection plan"):
        calibration_treatments(calibration, other)


def test_dry_run_freezes_requests_without_building_a_provider(tmp_path, monkeypatch):
    seeds_path = tmp_path / "seeds.yaml"
    seeds_path.write_text(_seeds_yaml(count=2), encoding="utf-8")
    profile = _profile()
    monkeypatch.setattr("catmodel.config.load_model_profile", lambda **_kwargs: profile)
    monkeypatch.setattr(
        "catmodel.llm.factory.build_providers",
        lambda *_args, **_kwargs: pytest.fail("dry run built providers"),
    )
    instrument = build_calibration_instrument(seeds_path, _respondents())
    run = tmp_path / "calibration-run"
    assert (
        run_calibration_collection(
            instrument=instrument,
            run_directory=run,
            profile="test",
            dry_run=True,
        )
        is None
    )
    plan = json.loads((run / "collection-plan.json").read_text())
    assert len(plan["requests"]) == 2
    assert not (run / "calibration-answers.json").exists()
    for request in plan["requests"]:
        assert (run / "calls" / request["key"] / "request.json").is_file()
    assert not (run / "source-bundle").exists()


def test_collection_resumes_stored_calls_and_replays_without_provider_calls(
    tmp_path, monkeypatch
):
    instrument = _instrument(respondents=_respondents(samples=2))
    run = tmp_path / "resume-run"
    profile = _profile(samples=2)
    monkeypatch.setattr("catmodel.config.load_model_profile", lambda **_kwargs: profile)
    run_calibration_collection(
        instrument=instrument,
        run_directory=run,
        profile="test",
        dry_run=True,
    )
    questions = calibration_questions(instrument)
    selected = [profile.models[0]]
    execution = profile.execution_settings(selected, max_output_tokens=100_000)
    first = questions[0]
    provider = _Provider("fixture-model")
    asyncio.run(
        collect_frozen_call(
            run_directory=run,
            request=calibration_request(first),
            provider=provider,
            config=execution,
            parse_answer=partial(parse_calibration_answer, question=first),
            response_model=CalibrationAnswer,
        )
    )
    remaining_provider = _Provider("fixture-model")
    _patch_collection(monkeypatch, profile, [remaining_provider])
    artifact = run_calibration_collection(
        instrument=instrument,
        run_directory=run,
        profile="test",
    )
    assert artifact is not None
    assert len(artifact.cases) == 2
    assert provider.calls == 1
    assert remaining_provider.calls == 1
    _, replayed, _ = validate_calibration_capture(run)
    assert replayed == artifact
    assert remaining_provider.calls == 1


def test_exhausted_answer_is_recorded_as_none_and_replayable(tmp_path, monkeypatch):
    instrument = _instrument((_seed(), _seed("seed-b", block_id="block-2")))
    provider = _Provider("fixture-model", invalid_seed_ids=frozenset({"seed-b"}))
    profile = _profile()
    _patch_collection(monkeypatch, profile, [provider])
    run = tmp_path / "exhausted-run"
    artifact = run_calibration_collection(
        instrument=instrument,
        run_directory=run,
        profile="test",
    )
    assert artifact is not None
    assert [row.answer is None for row in artifact.cases] == [False, True]
    assert provider.calls == 1 + PARSE_ATTEMPTS
    _, replayed, _ = validate_calibration_capture(run)
    assert replayed == artifact


def test_capture_validation_rejects_a_changed_frozen_request(tmp_path, monkeypatch):
    instrument = _instrument()
    profile = _profile()
    _patch_collection(monkeypatch, profile, [_Provider("fixture-model")])
    run = tmp_path / "changed-request"
    run_calibration_collection(
        instrument=instrument,
        run_directory=run,
        profile="test",
    )
    plan = json.loads((run / "collection-plan.json").read_text())
    request_path = run / "calls" / plan["requests"][0]["key"] / "request.json"
    request = json.loads(request_path.read_text())
    request["user_prompt"] += " changed"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="captured frozen request differs"):
        validate_calibration_capture(run)


@pytest.mark.parametrize(
    ("directory", "message"),
    [
        ("acquisition", "acquisition files differ"),
        ("parser-protocols", "parser protocol definition"),
    ],
)
def test_capture_validation_requires_acquisition_and_parser_protocol_files(
    tmp_path, monkeypatch, directory, message
):
    run, _ = _collect_fake_run(tmp_path, monkeypatch, directory)
    captured = next((run / directory).iterdir())
    captured.unlink()
    with pytest.raises(ValueError, match=message):
        validate_calibration_capture(run)


def test_capture_validation_requires_each_file_to_state_its_schema(
    tmp_path, monkeypatch
):
    run, _ = _collect_fake_run(tmp_path, monkeypatch)
    _, artifact, _ = validate_calibration_capture(run)
    for name in ("calibration-instrument.json", "calibration-answers.json"):
        written = (run / name).read_bytes()
        for payload in unstated_markers(json.loads(written)).values():
            (run / name).write_text(json.dumps(payload))
            with pytest.raises(ValueError, match="schema"):
                validate_calibration_capture(run)
        (run / name).write_bytes(written)
    assert validate_calibration_capture(run)[1] == artifact


def test_capture_validation_rejects_acquisition_effort_that_differs_from_plan(
    tmp_path, monkeypatch
):
    run, plan = _collect_fake_run(tmp_path, monkeypatch)
    key = plan["requests"][0]["key"]
    acquisition_path = run / "acquisition" / f"{key}.json"
    acquisition = json.loads(acquisition_path.read_text())
    acquisition["reasoning_effort"] = "high"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    with pytest.raises(ValueError, match="acquisition treatment differs"):
        validate_calibration_capture(run)


def test_a_second_calibration_collection_is_answered_from_the_table(
    tmp_path, monkeypatch
):
    """An answer table serves a calibration collection as it does a contextual
    one: a second run of the same instrument sends nothing, and a larger
    roster sends only its new samples."""
    table = tmp_path / "answers"
    profile = _profile(samples=2)

    def collect(name: str, samples: int) -> int:
        provider = _Provider("fixture-model")
        _patch_collection(monkeypatch, profile, [provider])
        artifact = run_calibration_collection(
            instrument=_instrument(respondents=_respondents(samples=samples)),
            run_directory=tmp_path / name,
            profile="test",
            answers=table,
        )
        assert artifact is not None and len(artifact.cases) == samples
        return provider.calls

    assert collect("first", 1) == 1
    assert collect("again", 1) == 0
    assert collect("larger", 2) == 1
    first = json.loads((tmp_path / "first" / "calibration-answers.json").read_text())
    again = json.loads((tmp_path / "again" / "calibration-answers.json").read_text())
    assert first["cases"] == again["cases"]


def test_a_systematic_provider_failure_stops_collection_without_answer_artifact(
    tmp_path, monkeypatch
):
    class FailingProvider(_Provider):
        async def complete(self, **_request):
            self.calls += 1
            raise RuntimeError("fake provider failure")

    instrument = _instrument()
    profile = _profile()
    provider = FailingProvider("fixture-model")
    _patch_collection(monkeypatch, profile, [provider])
    run = tmp_path / "provider-failure"
    with pytest.raises(
        ValueError, match=r"Collection incomplete.*fake provider failure"
    ):
        run_calibration_collection(
            instrument=instrument,
            run_directory=run,
            profile="test",
        )
    assert provider.calls == 1
    assert not (run / "calibration-answers.json").exists()


def test_a_transient_provider_failure_fails_only_its_call_and_resumes(
    tmp_path, monkeypatch
):
    class FlakyProvider(_Provider):
        """Fails seed-b's first request transiently, after the transport's
        own retries ran out."""

        async def complete(self, **request):
            if '"seed_id": "seed-b"' in request["user_prompt"] and not self.failed:
                self.calls += 1
                self.failed = True
                raise ProviderError("submission failed 5 times: 502", transient=True)
            return await super().complete(**request)

    instrument = _instrument((_seed(), _seed("seed-b", block_id="block-2")))
    provider = FlakyProvider("fixture-model")
    provider.failed = False
    _patch_collection(monkeypatch, _profile(), [provider])
    run = tmp_path / "flaky"
    with pytest.raises(ValueError, match=r"1 calls failed at the provider") as caught:
        run_calibration_collection(
            instrument=instrument, run_directory=run, profile="test"
        )
    plan = json.loads((run / "collection-plan.json").read_text())
    failed = [
        row["key"]
        for row in plan["requests"]
        if (run / "calls" / row["key"] / "attempt-001.json").exists()
        and json.loads((run / "calls" / row["key"] / "attempt-001.json").read_text())[
            "raw_response"
        ]
        is None
    ]
    assert len(failed) == 1 and failed[0] in str(caught.value)
    # The other seed was answered; a rerun sends only the failed call again.
    assert provider.calls == 2
    artifact = run_calibration_collection(
        instrument=instrument, run_directory=run, profile="test"
    )
    assert artifact is not None and all(row.answer for row in artifact.cases)
    assert provider.calls == 3
    assert (run / "calls" / failed[0] / "attempt-002.json").exists()
    _, replayed, _ = validate_calibration_capture(run)
    assert replayed == artifact


def test_resume_removes_the_temporary_files_an_interrupted_write_left(
    tmp_path, monkeypatch
):
    instrument = _instrument()
    profile = _profile()
    monkeypatch.setattr("catmodel.config.load_model_profile", lambda **_kwargs: profile)
    run = tmp_path / "interrupted"
    run_calibration_collection(
        instrument=instrument, run_directory=run, profile="test", dry_run=True
    )
    key = json.loads((run / "collection-plan.json").read_text())["requests"][0]["key"]
    (run / "acquisition").mkdir()
    # A crash while freezing the call's acquisition record, or its first
    # attempt, leaves a truncated temporary file and no record.
    leftovers = [
        run / "acquisition" / f".{key}.json.0123456789abcdef.tmp",
        run / "calls" / key / ".attempt-001.json.0123456789abcdef.tmp",
    ]
    for path in leftovers:
        path.write_bytes(b'{"truncat')
    provider = _Provider("fixture-model")
    _patch_collection(monkeypatch, profile, [provider])
    artifact = run_calibration_collection(
        instrument=instrument, run_directory=run, profile="test"
    )
    assert artifact is not None and provider.calls == 1
    assert not any(path.exists() for path in leftovers)
    _, replayed, _ = validate_calibration_capture(run)
    assert replayed == artifact


def test_fit_refuses_an_output_inside_the_capture_calls_tree(tmp_path):
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="inside the capture's calls tree"):
        fit_calibration(
            run,
            tmp_path / "resolutions.yaml",
            run / "calls" / "fit.json",
            bootstrap=1,
        )


def test_fit_refuses_an_output_inside_any_source_bundle_ancestor(tmp_path):
    bundle = tmp_path / "source-bundle"
    nested = bundle / "runs" / "calibration"
    nested.mkdir(parents=True)
    (bundle / "manifest.yaml").write_text("bundle: frozen\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inside a source bundle"):
        fit_calibration(
            tmp_path / "missing-run",
            tmp_path / "resolutions.yaml",
            nested / "fit.json",
            bootstrap=1,
            reference_replicates=0,
        )


def test_fit_verifies_resolution_script_digests_under_the_selected_root(
    tmp_path, monkeypatch
):
    script_root = tmp_path / "script-root"
    script = script_root / "scripts" / "resolve.py"
    script.parent.mkdir(parents=True)
    script.write_bytes(b"frozen resolver bytes\n")
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    instrument = _instrument((_fit_seed(),), resolution_digest=digest)
    run, _ = _collect_fake_run(tmp_path, monkeypatch, instrument=instrument)
    resolutions_path = tmp_path / "resolutions.json"
    resolutions_path.write_text(
        json.dumps(
            {
                "resolutions": [
                    SeedResolution(
                        seed_id="seed-primary",
                        value=4,
                        retrieved_at=date(2026, 11, 1),
                        snapshot_url="https://example.test/final",
                        snapshot_sha256="b" * 64,
                        script="scripts/resolve.py",
                        notes="",
                    ).model_dump(mode="json")
                ]
            }
        ),
        encoding="utf-8",
    )
    import catmodel.calibration.respondent_spread as respondent_spread

    monkeypatch.setattr(respondent_spread, "K_GRID_POINTS", 19)
    monkeypatch.setattr(respondent_spread, "B_GRID_POINTS", 21)
    output = tmp_path / "fit.json"
    fit = fit_calibration(
        run,
        resolutions_path,
        output,
        scripts_root=script_root,
        bootstrap=2,
        reference_replicates=0,
    )
    assert output.is_file()
    assert fit.capture.instrument_sha256 == instrument.digest()

    script.write_bytes(b"changed resolver bytes\n")
    with pytest.raises(ValueError, match=r"scripts/resolve\.py"):
        fit_calibration(
            run,
            resolutions_path,
            tmp_path / "changed-fit.json",
            scripts_root=script_root,
            bootstrap=2,
            reference_replicates=0,
        )

    monkeypatch.setattr("catmodel.resources.source_checkout_root", lambda: None)
    with pytest.raises(
        ValueError,
        match="resolution scripts cannot be verified outside a source checkout; "
        "fit from the checkout that holds them, or pass their root as scripts_root",
    ):
        fit_calibration(
            run,
            resolutions_path,
            tmp_path / "unverified-fit.json",
            bootstrap=2,
            reference_replicates=0,
        )


def _profiles_yaml(samples: int = 1) -> str:
    """A profiles file whose profile ``mine`` is ``_profile(samples=samples)``."""
    return (
        "profiles:\n"
        "  mine:\n"
        "    models:\n"
        "      - provider: openai\n"
        "        model_id: fixture-model\n"
        "        reasoning_effort: low\n"
        "        max_output_tokens: 2048\n"
        "    settings:\n"
        f"      samples_per_question: {samples}\n"
    )


def _script_json(capsys, *arguments: object) -> dict:
    """What a successful run of the script with ``arguments`` prints."""
    capsys.readouterr()
    assert calibration_script.main([str(argument) for argument in arguments]) == 0
    return json.loads(capsys.readouterr().out)


def test_script_reads_a_supplied_profiles_file_and_the_plan_records_it(
    tmp_path, monkeypatch, capsys
):
    seeds = tmp_path / "seeds.yaml"
    seeds.write_text(_seeds_yaml(), encoding="utf-8")
    profiles = tmp_path / "my-profiles.yaml"
    profiles.write_text(_profiles_yaml(), encoding="utf-8")
    monkeypatch.setattr(
        "catmodel.llm.factory.build_providers",
        lambda *_args, **_kwargs: pytest.fail("dry run built providers"),
    )
    run = tmp_path / "run"
    arguments = ("collect", "--seeds", seeds, "--run-directory", run, "--dry-run")
    assert _script_json(
        capsys, *arguments, "--profile", "mine", "--profiles", profiles
    ) == {"frozen_requests": 1}
    # The instrument's respondents are the profile's roster.
    instrument = CalibrationInstrument.model_validate_json(
        (run / "calibration-instrument.json").read_text()
    )
    assert [row.model_id for row in instrument.respondents] == ["fixture-model"]
    plan = json.loads((run / "collection-plan.json").read_text())
    assert plan["profile"] == "mine"
    assert plan["profiles_file"] == {
        "path": str(profiles),
        "sha256": hashlib.sha256(profiles.read_bytes()).hexdigest(),
    }
    assert [row["model_id"] for row in plan["execution"]["models"]] == ["fixture-model"]


def test_script_collect_requires_exactly_one_of_dry_run_and_live(tmp_path, capsys):
    seeds = tmp_path / "seeds.yaml"
    seeds.write_text(_seeds_yaml(), encoding="utf-8")
    run = tmp_path / "run"
    collect = ["collect", "--seeds", str(seeds), "--run-directory", str(run)]
    for flags in ([], ["--dry-run", "--live"]):
        assert calibration_script.main([*collect, *flags]) == 2
        assert "exactly one of --dry-run" in capsys.readouterr().err
    assert not run.exists()


def test_script_dry_run_reports_what_the_answer_table_holds(
    tmp_path, monkeypatch, capsys
):
    """A dry run with an answer table says how many of its frozen requests
    the table already answers; one without a table counts nothing."""
    seeds = tmp_path / "seeds.yaml"
    seeds.write_text(_seeds_yaml(count=2), encoding="utf-8")
    profiles = tmp_path / "profiles.yaml"
    table = tmp_path / "answers"

    def collect(name: str, samples: int, *flags: object) -> dict:
        profiles.write_text(_profiles_yaml(samples), encoding="utf-8")
        return _script_json(
            capsys,
            *("collect", "--seeds", seeds, "--run-directory", tmp_path / name),
            *("--profile", "mine", "--profiles", profiles),
            *flags,
        )

    assert collect("bare", 1, "--dry-run") == {"frozen_requests": 2}
    assert collect("empty", 1, "--dry-run", "--answers", table) == {
        "frozen_requests": 2,
        "held_by_answer_table": 0,
    }
    provider = _Provider("fixture-model")
    monkeypatch.setattr(
        "catmodel.llm.factory.build_providers", lambda *_args, **_kwargs: [provider]
    )
    assert collect("live", 1, "--live", "--answers", table)["answered"] == 2
    assert provider.calls == 2
    # A second sample per model doubles the requests; the table answers the
    # first sample's.
    assert collect("larger", 2, "--dry-run", "--answers", table) == {
        "frozen_requests": 4,
        "held_by_answer_table": 2,
    }
    assert provider.calls == 2


def test_script_collect_and_fit_use_the_calibration_service(
    tmp_path, monkeypatch, capsys
):
    script_root = tmp_path / "script-root"
    script = script_root / "scripts" / "resolve.py"
    script.parent.mkdir(parents=True)
    script.write_bytes(b"frozen cli resolver\n")
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    seeds = tmp_path / "seeds.yaml"
    seeds.write_text(_seeds_yaml(digest=digest), encoding="utf-8")
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text(_profiles_yaml(), encoding="utf-8")
    monkeypatch.setattr(
        "catmodel.llm.factory.build_providers",
        lambda *_args, **_kwargs: pytest.fail("dry run built providers"),
    )
    run = tmp_path / "script-run"
    collect = (
        *("collect", "--seeds", seeds, "--run-directory", run),
        *("--profile", "mine", "--profiles", profiles),
    )
    assert _script_json(capsys, *collect, "--dry-run") == {"frozen_requests": 1}
    instrument = CalibrationInstrument.model_validate_json(
        (run / "calibration-instrument.json").read_text()
    )
    assert len(instrument.respondents) == 1

    provider = _Provider("fixture-model")
    monkeypatch.setattr(
        "catmodel.llm.factory.build_providers",
        lambda *_args, **_kwargs: [provider],
    )
    # Collecting in the same directory resumes the frozen collection.
    assert _script_json(capsys, *collect, "--live") == {
        "frozen_requests": 1,
        "cases": 1,
        "answered": 1,
        "exhausted": 0,
        "unavailable": 0,
    }

    resolution_path = tmp_path / "resolutions.yaml"
    resolution_path.write_text(
        json.dumps(
            {
                "resolutions": [
                    {
                        "seed_id": "seed-0",
                        "value": 4,
                        "retrieved_at": "2026-11-01",
                        "snapshot_url": "https://example.test/final",
                        "snapshot_sha256": "c" * 64,
                        "script": "scripts/resolve.py",
                        "notes": "",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    import catmodel.calibration.respondent_spread as respondent_spread

    monkeypatch.setattr(respondent_spread, "K_GRID_POINTS", 19)
    monkeypatch.setattr(respondent_spread, "B_GRID_POINTS", 21)
    # The script's bootstrap and reference sizes are fixed; a small budget
    # keeps the fit fast without changing what the script writes. It verifies
    # the resolvers in the source checkout; this test's resolver has its own
    # root.
    monkeypatch.setattr(
        calibration_script,
        "fit_calibration",
        partial(
            fit_calibration,
            scripts_root=script_root,
            bootstrap=4,
            reference_replicates=1,
            seed=8,
        ),
    )
    fit_path = tmp_path / "fit.json"
    fit = (
        *("fit", "--run-directory", run, "--resolutions", resolution_path),
        *("--output", fit_path, "--workers", "1"),
    )
    summary = _script_json(capsys, *fit)
    assert summary["status"] in {"accepted", "rejected"}
    assert summary["observation"]["model_name"] == "negative_binomial_2"
    assert set(summary["alternatives"]) == {"poisson", "pooled_phi"}
    assert set(summary) == {
        "status",
        "k_hat",
        "range",
        "b_hat",
        "b_range",
        "k_b_correlation",
        "marginal",
        "observation",
        "alternatives",
        "leave_one_block_out",
        "primary_pit_medians",
    }
    assert set(summary["marginal"]) == {"k_at_b_1", "b_at_reference_k"}
    assert set(summary["leave_one_block_out"]) == {
        "joint_total_gain",
        "conditional_k_total_gain",
    }
    assert set(summary["primary_pit_medians"]) == {
        "at_k_1_b_1",
        "at_k_hat_b_1",
        "at_k_1_b_hat",
        "at_k_hat_b_hat",
    }
    assert summary["marginal"]["b_at_reference_k"]["reference_spread_basis"] in {
        "accepted_k_hat",
        "face_value",
    }
    assert fit_path.is_file()
    # An output is never overwritten.
    assert calibration_script.main([str(argument) for argument in fit]) == 2
    assert "already exists" in capsys.readouterr().err


def test_request_key_is_the_frozen_call_key():
    instrument = _instrument()
    question = calibration_questions(instrument)[0]
    request = calibration_request(question)
    assert frozen_call_key(request) == canonical_json_sha256(
        [question.call_id, question.respondent.key()]
    )
