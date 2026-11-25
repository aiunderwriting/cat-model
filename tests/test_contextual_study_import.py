"""Offline collection-to-native-package checks with original replayed responses."""

import json

import pytest
from click.testing import CliRunner

from catmodel.cli import cli
from catmodel.collection.contextual_collection import run_contextual_collection
from catmodel.config import ModelProfile
from catmodel.contracts.contextual_risk import ContextualRiskInstrument
from catmodel.contracts.incident import DEFAULT_MAX_REFERENCE_STATE_YEARS
from catmodel.evidence.source_bundle import capture_source_bundle_snapshot
from catmodel.llm.models import ModelSpec
from catmodel.llm.transport import ProviderResponse
from catmodel.results.conditional import ConditionalCapture
from catmodel.results.execution import run_setup
from catmodel.results.results_export import export_results
from catmodel.setup.contextual_import import import_contextual_setup
from catmodel.setup.contextual_inputs import load_contextual_collection
from test_support.capability_sources import write_source_bundle
from test_support.contextual_risk import (
    contextual_answer,
    contextual_instrument,
    fixture_response,
)


class _Provider:
    model_id = "fixture-model"
    provider_name = "openai"
    reasoning_effort = "low"

    def __init__(self):
        self.calls = 0

    async def complete(self, **kwargs):
        self.calls += 1
        return ProviderResponse(
            text=fixture_response(kwargs["user_prompt"], contextual_answer()),
            model_id=self.model_id,
            provider_name=self.provider_name,
            provider_request_id=f"offline-{self.calls}",
            input_tokens=5,
            output_tokens=10,
        )


class _OverDeepFirst(_Provider):
    """Answers its first request with JSON nested deeper than any
    interpreter's decoder recurses through, then as ``_Provider``."""

    async def complete(self, **kwargs):
        if self.calls:
            return await super().complete(**kwargs)
        self.calls += 1
        return ProviderResponse(
            text="[" * 200_000 + "]" * 200_000,
            model_id=self.model_id,
            provider_name=self.provider_name,
        )


def _collection(tmp_path, monkeypatch, provider=None):
    source_bundle = write_source_bundle(tmp_path / "source")
    snapshot = capture_source_bundle_snapshot(source_bundle)
    payload = contextual_instrument().model_dump(mode="json")
    payload["source_bundle_manifest_sha256"] = snapshot.manifest_sha256
    instrument = ContextualRiskInstrument.model_validate(payload)
    provider = provider or _Provider()
    profile = ModelProfile(
        models=(
            ModelSpec(
                provider="openai",
                model_id=provider.model_id,
                reasoning_effort="low",
                max_output_tokens=100_000,
            ),
        )
    )
    monkeypatch.setattr("catmodel.config.load_model_profile", lambda **_kwargs: profile)
    monkeypatch.setattr(
        "catmodel.llm.factory.build_providers", lambda *_args, **_kwargs: [provider]
    )
    directory = tmp_path / "collected"
    run_contextual_collection(
        instrument=instrument,
        source_bundle=source_bundle,
        run_directory=directory,
        profile="test",
    )
    return directory, source_bundle, provider


def test_frozen_collection_generates_native_package_run_and_conditional_export(
    tmp_path, monkeypatch
):
    directory, source_bundle, provider = _collection(tmp_path, monkeypatch)
    instrument, _, original_files = load_contextual_collection(directory, source_bundle)
    assert provider.calls == 2 * len(instrument.anchor_scores)
    source_before = {
        path: path.read_bytes() for path in source_bundle.rglob("*") if path.is_file()
    }

    def no_collection(*_args, **_kwargs):
        raise AssertionError("native import must not collect new answers")

    monkeypatch.setattr("catmodel.llm.factory.build_providers", no_collection)
    package = import_contextual_setup(
        source=directory,
        source_bundle=source_bundle,
        destination=tmp_path / "package",
        setup_id="offline-contextual",
        years_per_state=48,
        # 4,096 years for each of the 4 states.
        max_reference_state_years=4 * 4096,
        seed=19,
        code_commit="fixture",
    )
    assert any(
        "Respondent spread is uncalibrated" in limitation
        and "face value (k = 1)" in limitation
        for limitation in package.limitations
    )
    assert all(
        package.files["evidence/captured/" + name] == data
        for name, data in original_files.items()
    )
    assert all(path.read_bytes() == data for path, data in source_before.items())
    assert package.evidence.records[0].origin == "elicitation"
    assert package.evidence.records[0].raw_attempt_sha256
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")
    captured = ConditionalCapture(package, tmp_path / "run/capture")
    assert captured.kernel_at(instrument.context_id, 1, "reference").metrics("welfare")[
        "expected_annual_loss"
    ] == pytest.approx(100)
    export_results(
        setup_path=package.root,
        capture_directory=tmp_path / "run/capture",
        output=tmp_path / "export",
        code_commit="fixture",
    )
    index = json.loads((tmp_path / "export/export.json").read_text())
    assert index["conditioning"]["status"] == "available"


def test_import_replays_raw_attempts_instead_of_trusting_artifact(
    tmp_path, monkeypatch
):
    directory, source_bundle, _ = _collection(tmp_path, monkeypatch)
    attempt = next((directory / "calls").rglob("attempt-*.json"))
    content = json.loads(attempt.read_text())
    content["raw_response"] = (
        contextual_answer().model_copy(update={"rate": None}).model_dump_json()
    )
    attempt.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        import_contextual_setup(
            source=directory,
            source_bundle=source_bundle,
            destination=tmp_path / "bad-package",
            setup_id="tampered",
            code_commit="fixture",
        )
    assert not (tmp_path / "bad-package").exists()


def test_a_capture_holding_an_over_deep_response_validates_imports_and_resumes(
    tmp_path, monkeypatch
):
    directory, source_bundle, provider = _collection(
        tmp_path, monkeypatch, _OverDeepFirst()
    )
    attempts = [
        json.loads(path.read_bytes())
        for path in (directory / "calls").rglob("attempt-*.json")
    ]
    [failed] = [row for row in attempts if not row["parse_success"]]
    assert failed["raw_response"].startswith("[[[") and failed["attempt"] == 1
    assert provider.calls == len(attempts)
    validated = CliRunner().invoke(
        cli, ["capability-risk", "validate", "--run-directory", str(directory)]
    )
    assert validated.exit_code == 0, validated.output
    usage = json.loads(validated.output)["call_limits"][provider.model_id]
    assert usage["responses"] == len(attempts)
    imported = CliRunner().invoke(
        cli,
        [
            "setup",
            "import",
            "--run-directory",
            str(directory),
            "--setup-id",
            "over-deep",
            "--output",
            str(tmp_path / "package"),
        ],
    )
    assert imported.exit_code == 0, imported.output
    assert (tmp_path / "package/setup-manifest.json").is_file()
    # Resume replays every stored call and sends nothing.
    instrument, artifact, _ = load_contextual_collection(directory, source_bundle)
    resumed = run_contextual_collection(
        instrument=instrument,
        source_bundle=source_bundle,
        run_directory=directory,
        profile="test",
    )
    assert resumed == artifact and provider.calls == len(attempts)


def test_import_cli_imports_a_fixed_anchor_capture_and_refuses_study_options(
    tmp_path, monkeypatch
):
    directory, _, _ = _collection(tmp_path, monkeypatch)
    arguments = [
        "setup",
        "import",
        "--run-directory",
        str(directory),
        "--setup-id",
        "cli-context",
    ]
    rejected = CliRunner().invoke(
        cli,
        [
            *arguments,
            "--output",
            str(tmp_path / "rejected"),
            "--comparison-family",
            "tail_shape",
        ],
    )
    assert rejected.exit_code == 2
    assert "--comparison-family applies to public studies (--study) only" in (
        rejected.output
    )
    assert not (tmp_path / "rejected").exists()
    accepted = CliRunner().invoke(
        cli, [*arguments, "--output", str(tmp_path / "cli-package")]
    )
    assert accepted.exit_code == 0, accepted.output
    assert (tmp_path / "cli-package/setup-manifest.json").is_file()
    plan = json.loads((tmp_path / "cli-package/execution-plan.json").read_text())
    common_bias = next(
        row
        for row in plan["comparison_requirements"]
        if row["family_id"] == "POOL-COMMON-BIAS"
    )
    assert common_bias["status"] == "unavailable"
    assert "only their reference arm" in common_bias["reason"]
    # Its conditional kernels read every modeled integer's state, so the
    # reference arm simulates the zero-mass states too.
    setup = json.loads((tmp_path / "cli-package/setup.json").read_text())
    states = [row["state_id"] for row in setup["joint_states"]]
    assert any(row["weight"] == 0 for row in setup["joint_states"])
    assert plan["required_state_ids_by_arm"] == {"reference": states}
    # A fixed-anchor setup runs only its reference arm, under the fixed cap.
    assert plan["numerical_controls"]["tail_gate"] == {
        "tolerance": 0.1,
        "batches": 16,
        "max_reference_state_years": DEFAULT_MAX_REFERENCE_STATE_YEARS,
        "max_comparison_state_years": None,
    }
    # The work allowances and starting years are fixed defaults (NUM-BUDGETS),
    # as is the plan seed.
    assert {
        key: value
        for key, value in plan["numerical_controls"].items()
        if key.endswith("_per_year")
    } == {
        "max_vectorized_events_per_year": 50_000,
        "max_campaign_members_per_year": 50_000,
    }
    assert (plan["simulations"], plan["seed"]) == (1_024, 42)
