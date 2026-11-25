"""Offline contextual collection, immutable source binding and raw replay."""

import asyncio
import hashlib
import json
import shutil
from typing import cast

import pytest

from catmodel.collection.contextual_collection import (
    collect_contextual_risk,
    run_contextual_collection,
)
from catmodel.config import PROFILES_ENV, ExecutionSettings, ModelProfile, settings
from catmodel.contracts.contextual_risk import ContextualRiskInstrument
from catmodel.evidence.source_bundle import (
    MANIFEST_PATH,
    PROVENANCE_PATH,
    capture_source_bundle_snapshot,
)
from catmodel.llm.models import ModelSpec
from catmodel.llm.transport import LLMProvider, ProviderResponse
from catmodel.setup.contextual_inputs import (
    load_contextual_collection,
    validate_contextual_sources,
)
from catmodel.yaml_io import safe_dump, safe_load
from test_support.capability_sources import authored_source_files, write_source_bundle
from test_support.contextual_risk import (
    contextual_answer,
    contextual_instrument,
    fixture_response,
)
from test_support.version_markers import unstated_markers


class Provider:
    provider_name = "offline-fixture"
    model_id = "fixture-model"
    reasoning_effort = "low"

    def __init__(self):
        self.calls = 0

    async def complete(self, **kwargs):
        self.calls += 1
        return ProviderResponse(
            text=fixture_response(kwargs["user_prompt"], contextual_answer()),
            model_id=self.model_id,
            provider_name=self.provider_name,
        )


def _prepared(tmp_path, monkeypatch):
    import catmodel.config

    source = write_source_bundle(tmp_path / "source", files=authored_source_files())
    snapshot = capture_source_bundle_snapshot(source)
    data = contextual_instrument().model_dump(mode="json")
    data["source_bundle_manifest_sha256"] = snapshot.manifest_sha256
    instrument = ContextualRiskInstrument.model_validate(data)
    monkeypatch.setattr(
        catmodel.config,
        "load_model_profile",
        lambda **kwargs: ModelProfile(
            models=(
                ModelSpec(
                    provider="openai",
                    model_id="fixture-model",
                    reasoning_effort="low",
                    max_output_tokens=100_000,
                ),
            )
        ),
    )
    monkeypatch.setattr(
        catmodel.config.Settings,
        "provider_api_keys",
        lambda self: pytest.fail("offline path requested credentials"),
    )
    run = tmp_path / "run"
    run_contextual_collection(
        instrument=instrument,
        source_bundle=source,
        run_directory=run,
        profile="test",
        dry_run=True,
    )
    return source, snapshot, instrument, run


FIXTURE_PROFILES = """\
profiles:
  mine:
    models:
      - provider: openai
        model_id: fixture-model
        reasoning_effort: low
        max_output_tokens: 100000
"""


def test_plan_records_a_supplied_profiles_file_beside_the_profile(
    tmp_path, monkeypatch
):
    import catmodel.config

    source = write_source_bundle(tmp_path / "source", files=authored_source_files())
    data = contextual_instrument().model_dump(mode="json")
    data["source_bundle_manifest_sha256"] = capture_source_bundle_snapshot(
        source
    ).manifest_sha256
    instrument = ContextualRiskInstrument.model_validate(data)
    profiles = tmp_path / "my-profiles.yaml"
    profiles.write_text(FIXTURE_PROFILES, encoding="utf-8")

    def plan(name: str, **options) -> dict:
        run = tmp_path / name
        run_contextual_collection(
            instrument=instrument,
            source_bundle=source,
            run_directory=run,
            profile="mine",
            dry_run=True,
            **options,
        )
        return json.loads((run / "collection-plan.json").read_text())

    # Read as the packaged example, the same roster leaves the plan with only
    # the fields every plan has.
    monkeypatch.setattr(catmodel.config, "data_path", lambda *_parts: profiles)
    default = plan("default")
    assert set(default) == {
        "instrument_sha256",
        "profile",
        "parse_attempts",
        "execution",
        "requests",
    }
    record = {
        "path": str(profiles),
        "sha256": hashlib.sha256(profiles.read_bytes()).hexdigest(),
    }
    supplied = plan("supplied", profiles=profiles)
    assert supplied == {**default, "profiles_file": record}
    monkeypatch.setenv(PROFILES_ENV, str(profiles))
    assert plan("environment") == supplied

    # A run directory resumes only with the same file: an edited one, even
    # by a comment, is another origin of the roster.
    profiles.write_text("# edited\n" + FIXTURE_PROFILES, encoding="utf-8")
    with pytest.raises(ValueError, match="frozen artifact differs: collection-plan"):
        plan("supplied", profiles=profiles)


def test_dry_run_raw_replay_source_binding_and_resumption(tmp_path, monkeypatch):
    source, snapshot, instrument, run = _prepared(tmp_path, monkeypatch)
    plan = json.loads((run / "collection-plan.json").read_text())
    # The plan freezes each model's ceiling and the budget it asks for.
    assert plan["execution"]["models"][0]["max_output_tokens"] == 100_000
    assert plan["execution"]["settings"]["max_output_tokens_by_model"] == {
        "fixture-model": min(100_000, settings.ELICITATION_MAX_TOKENS)
    }
    provider = Provider()
    config = ExecutionSettings(
        samples_per_question=1,
        max_output_tokens_by_model={provider.model_id: 100},
        request_timeout_seconds=60,
        concurrency_by_model={provider.model_id: 1},
    )
    artifact = asyncio.run(
        collect_contextual_risk(
            instrument=instrument,
            run_directory=run,
            providers=[cast(LLMProvider, provider)],
            config=config,
        )
    )
    (run / "contextual-risk.json").write_text(artifact.model_dump_json())
    (run / "private-notes.txt").write_text("Not collection evidence")
    loaded, result, files = load_contextual_collection(run, source)
    assert loaded == instrument and result == artifact
    assert "private-notes.txt" not in files
    assert all(
        files["source-bundle/" + name] == data for name, data in snapshot.files.items()
    )
    # One direct and one chain call per case: no answer triggers a retry or
    # cross-check.
    assert provider.calls == 2 * len(artifact.cases) == 6
    assert (
        run_contextual_collection(
            instrument=instrument,
            source_bundle=source,
            run_directory=run,
            profile="test",
        )
        == artifact
    )
    assert provider.calls == 6
    changed = json.loads((run / "contextual-risk.json").read_text())
    changed["cases"][0]["answer"]["reasoning"]["median_reason"] = "Tampered."
    (run / "contextual-risk.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="raw response"):
        load_contextual_collection(run)


def test_a_capture_whose_instrument_or_answers_do_not_state_their_schema_is_refused(
    tmp_path, monkeypatch
):
    source, _, instrument, run = _prepared(tmp_path, monkeypatch)
    provider = Provider()
    artifact = asyncio.run(
        collect_contextual_risk(
            instrument=instrument,
            run_directory=run,
            providers=[cast(LLMProvider, provider)],
            config=ExecutionSettings(
                samples_per_question=1,
                max_output_tokens_by_model={provider.model_id: 100},
                request_timeout_seconds=60,
                concurrency_by_model={provider.model_id: 1},
            ),
        )
    )
    (run / "contextual-risk.json").write_text(artifact.model_dump_json())
    for name in ("contextual-instrument.json", "contextual-risk.json"):
        written = (run / name).read_bytes()
        for payload in unstated_markers(json.loads(written)).values():
            (run / name).write_text(json.dumps(payload))
            with pytest.raises(ValueError, match="schema"):
                load_contextual_collection(run, source)
        (run / name).write_bytes(written)
    loaded, replayed, _ = load_contextual_collection(run, source)
    assert loaded.digest() == instrument.digest() and replayed == artifact


def test_resume_removes_the_temporary_files_an_interrupted_write_left(
    tmp_path, monkeypatch
):
    source, _, instrument, run = _prepared(tmp_path, monkeypatch)
    plan = json.loads((run / "collection-plan.json").read_text())
    key = plan["requests"][0]["key"]
    (run / "acquisition").mkdir()
    # Truncated temporary files, as a crash while freezing each file leaves
    # them; the source bundle's own check refuses any file it does not track.
    leftovers = [
        run / "source-bundle" / f".{MANIFEST_PATH}.0123456789abcdef.tmp",
        run / "calls" / key / ".attempt-001.json.0123456789abcdef.tmp",
        run / "acquisition" / f".{key}.json.0123456789abcdef.tmp",
        run / ".contextual-risk.json.0123456789abcdef.tmp",
    ]
    for path in leftovers:
        path.write_bytes(b'{"truncat')
    provider = Provider()
    monkeypatch.setattr(
        "catmodel.llm.factory.build_providers", lambda *_args, **_kwargs: [provider]
    )
    monkeypatch.setattr("catmodel.config.Settings.provider_api_keys", lambda self: {})
    artifact = run_contextual_collection(
        instrument=instrument,
        source_bundle=source,
        run_directory=run,
        profile="test",
    )
    assert artifact is not None and provider.calls == 6
    assert not any(path.exists() for path in leftovers)
    assert list(run.rglob("*.tmp")) == []
    assert load_contextual_collection(run, source)[1] == artifact


def test_source_and_plan_tamper_fail_before_use(tmp_path, monkeypatch):
    _, snapshot, instrument, run = _prepared(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="stale"):
        validate_contextual_sources(contextual_instrument(), snapshot)
    data = instrument.model_dump(mode="json")
    data["nodes"][0]["source_cards"] = [
        {
            "card_id": "invented",
            "text": "Not captured",
            "observation_through_utc": "2025-01-01",
            "availability_date_utc": "2025-01-01",
            "availability_basis": "synthetic",
            "sources": ["https://example.org/invented"],
        }
    ]
    with pytest.raises(ValueError, match="differs"):
        validate_contextual_sources(
            ContextualRiskInstrument.model_validate(data), snapshot
        )
    provider = Provider()
    artifact = asyncio.run(
        collect_contextual_risk(
            instrument=instrument,
            run_directory=run,
            providers=[cast(LLMProvider, provider)],
            config=ExecutionSettings(
                samples_per_question=1,
                max_output_tokens_by_model={provider.model_id: 100},
                request_timeout_seconds=60,
                concurrency_by_model={provider.model_id: 1},
            ),
        )
    )
    (run / "contextual-risk.json").write_text(artifact.model_dump_json())
    plan = json.loads((run / "collection-plan.json").read_text())
    plan["requests"] = []
    (run / "collection-plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="plan differs"):
        load_contextual_collection(run)


def _with_provenance(bundle, destination, commit):
    shutil.copytree(bundle, destination)
    manifest = (destination / MANIFEST_PATH).read_bytes()
    (destination / PROVENANCE_PATH).write_text(
        safe_dump(
            {
                "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                "repo_git_sha": commit * 40,
                "assumptions_md_sha256": commit * 64,
                "hierarchy_version": "v1",
                "hierarchy_sha256": commit * 64,
                "source_input_recipe_sha256": "c" * 64,
            }
        )
    )
    return destination


def test_bundle_rebuilt_with_other_provenance_keeps_its_identity(tmp_path, monkeypatch):
    source, snapshot, instrument, run = _prepared(tmp_path, monkeypatch)
    original = _with_provenance(source, tmp_path / "built-at-a", "a")
    rebuilt = _with_provenance(source, tmp_path / "built-at-b", "b")
    assert (
        capture_source_bundle_snapshot(original).manifest_sha256
        == capture_source_bundle_snapshot(rebuilt).manifest_sha256
        == snapshot.manifest_sha256
    )
    for bundle in (original, rebuilt):
        run_contextual_collection(
            instrument=instrument,
            source_bundle=bundle,
            run_directory=run,
            profile="test",
            dry_run=True,
        )
    assert (run / "source-bundle" / PROVENANCE_PATH).read_bytes() == (
        original / PROVENANCE_PATH
    ).read_bytes()
    provider = Provider()
    artifact = asyncio.run(
        collect_contextual_risk(
            instrument=instrument,
            run_directory=run,
            providers=[cast(LLMProvider, provider)],
            config=ExecutionSettings(
                samples_per_question=1,
                max_output_tokens_by_model={provider.model_id: 100},
                request_timeout_seconds=60,
                concurrency_by_model={provider.model_id: 1},
            ),
        )
    )
    (run / "contextual-risk.json").write_text(artifact.model_dump_json())
    _, loaded, retained = load_contextual_collection(run, rebuilt)
    assert loaded == artifact
    assert (
        retained["source-bundle/" + PROVENANCE_PATH]
        == (original / PROVENANCE_PATH).read_bytes()
    )
    different = tmp_path / "different"
    shutil.copytree(source, different)
    manifest = safe_load((different / MANIFEST_PATH).read_text())
    manifest["created_at"] = "2026-09-02T00:00:00Z"
    (different / MANIFEST_PATH).write_text(safe_dump(manifest, sort_keys=False))
    with pytest.raises(ValueError, match="supplied source bundle differs"):
        load_contextual_collection(run, different)
