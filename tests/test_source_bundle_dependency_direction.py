from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from catmodel.contracts.governance import ExternalSourceSnapshot, FileChecksum
from catmodel.evidence import source_bundle
from catmodel.evidence.source_bundle import (
    SourceBundleManifest,
    SourceBundleSnapshot,
    validate_source_bundle_dependency_direction,
    validate_source_bundle_schema_dependency_direction,
)
from catmodel.yaml_io import safe_dump


def _manifest(*relative_paths: str) -> SourceBundleManifest:
    return SourceBundleManifest(
        bundle_schema_version=source_bundle.SOURCE_BUNDLE_SCHEMA_VERSION,
        bundle_id="dependency-direction-test",
        bundle_version="2026-08-01",
        created_at="2026-08-01T00:00:00+00:00",
        files={
            relative_path: FileChecksum(sha256="a" * 64)
            for relative_path in relative_paths
        },
    )


def _scan(root: Path, manifest: SourceBundleManifest) -> None:
    """Scan the manifest's files as an intake would have captured them from
    ``root``."""
    validate_source_bundle_dependency_direction(
        SourceBundleSnapshot(
            root,
            manifest,
            {relative: (root / relative).read_bytes() for relative in manifest.files},
        )
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"outer": [{"run_id": None}]}, r"\$\.outer\[0\]\.run_id"),
        (
            {"outer": {"model_id": None}},
            r"\$\.outer\.model_id.*model identity",
        ),
        (
            {"outer": {"elicitation_run_id": None}},
            r"\$\.outer\.elicitation_run_id.*elicitation identity",
        ),
        (
            {"outer": {"question_authority_sha256": "a" * 64}},
            r"\$\.outer\.question_authority_sha256.*downstream authority identity",
        ),
        (
            {"outer": {"model_authority_sha256": "a" * 64}},
            r"\$\.outer\.model_authority_sha256.*downstream authority identity",
        ),
        ({"basis": "elicited"}, "basis='elicited'"),
        (
            {"source_classification": "llm_preliminary"},
            "source_classification='llm_preliminary'",
        ),
        (
            {"notes": ["promoted from run://elicitation-example"]},
            "elicitation-run URI",
        ),
        (
            {"outer": {"elicitation_sha256": "a" * 64}},
            r"\$\.outer\.elicitation_sha256.*elicitation identity",
        ),
        # Structure written inside a string: a key at the start of a line or
        # a quoted key anywhere.
        ({"notes": "model_id: respondent"}, "model identity assignment"),
        (
            {"notes": 'Copied from {"run_id": "downstream"} by hand.'},
            "run identity assignment",
        ),
    ],
)
def test_content_guard_rejects_nested_downstream_identity(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    relative_path = "newly_tracked.yaml"
    (tmp_path / relative_path).write_text(safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _scan(tmp_path, _manifest(relative_path))


def test_content_guard_walks_exactly_manifest_files(tmp_path: Path) -> None:
    relative_path = "untracked.yaml"
    (tmp_path / relative_path).write_text(
        safe_dump({"nested": {"elicitation_run_id": None}}),
        encoding="utf-8",
    )

    _scan(tmp_path, _manifest())

    with pytest.raises(ValueError, match=r"untracked\.yaml.*elicitation_run_id"):
        _scan(tmp_path, _manifest(relative_path))


def test_content_guard_allows_insured_model_identifiers(tmp_path: Path) -> None:
    relative_path = "entities.yaml"
    (tmp_path / relative_path).write_text(
        safe_dump(
            {
                "entities": [
                    {
                        "source_model_id": "insured-source-model",
                        "coi_exclusion_model_id": "insured-coi-model",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    _scan(tmp_path, _manifest(relative_path))


def test_authored_question_provenance_does_not_authorize_downstream_content(
    tmp_path: Path,
) -> None:
    relative_path = "source-provenance.yaml"
    path = tmp_path / relative_path
    payload = {"question_semantics_version": "v1", "loss_population_sha256": "a" * 64}
    path.write_text(safe_dump(payload), encoding="utf-8")
    _scan(tmp_path, _manifest(relative_path))

    payload["elicitation_run_id"] = "a-downstream-run"
    path.write_text(safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="elicitation identity"):
        _scan(tmp_path, _manifest(relative_path))


def test_content_guard_allows_non_assignment_prose_and_normal_run_url(
    tmp_path: Path,
) -> None:
    relative_path = "legitimate-prose.yaml"
    (tmp_path / relative_path).write_text(
        safe_dump(
            {
                "notes": [
                    (
                        "Neither model_id nor run_id belongs in source-bundle "
                        "data; this sentence records no identity value."
                    ),
                    (
                        "A question_authority_sha256 reference describes the "
                        "downstream authority concept only."
                    ),
                    "See https://example.test/run/model-governance for background.",
                    "Structured expert elicitation: Cooke (1991), Experts in "
                    "Uncertainty.",
                    "The evaluation used the vendor's model (model_id: GPT-4).",
                    "Logs: https://github.com/example/repo/actions/runs/123/job/4",
                    "Captured from /private/tmp/runs/example-run/output.yaml.",
                ],
                "elicitation_method": "structured expert judgment",
                "expert_elicitation": {"protocol": "classical model"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "excerpt.md").write_text(
        "# Excerpt\n\nStructured expert elicitation: Cooke (1991).\n"
        "The system card evaluates GPT-4 (model_id: GPT-4) at "
        "https://example.test/runs/2024-03/report.html.\n"
    )
    manifest = _manifest(relative_path, "excerpt.md").model_copy(
        update={
            "external_sources": [
                ExternalSourceSnapshot(
                    source_id="ci-artifact",
                    uri="https://github.com/example/repo/actions/runs/123",
                    content_hash="sha256:fixture",
                )
            ]
        }
    )

    _scan(tmp_path, manifest)


@pytest.mark.parametrize(
    "field_name",
    [
        "elicitation_run_id",
        "run_id",
        "provider_model_id",
        "question_authority_sha256",
    ],
)
def test_schema_guard_rejects_new_downstream_identity_field(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
) -> None:
    adversarial_schema = type(
        "AdversarialSourceBundleSchema",
        (BaseModel,),
        {
            "__module__": source_bundle.__name__,
            "__annotations__": {field_name: str | None},
            field_name: None,
        },
    )
    monkeypatch.setattr(
        source_bundle,
        "AdversarialSourceBundleSchema",
        adversarial_schema,
        raising=False,
    )

    with pytest.raises(ValueError, match=field_name):
        validate_source_bundle_schema_dependency_direction()


def test_guard_rejects_renamed_key_carrying_run_locator(tmp_path: Path) -> None:
    relative_path = "renamed.yaml"
    (tmp_path / relative_path).write_text(
        safe_dump({"sampling_context": "elicitation-run://prod-2026-08"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="elicitation-run URI"):
        _scan(tmp_path, _manifest(relative_path))


def test_guard_rejects_encoded_payload_instead_of_skipping(tmp_path: Path) -> None:
    relative_path = "encoded.yaml"
    payload = safe_dump({"nested": {"elicitation_run_id": "prod"}})
    (tmp_path / relative_path).write_bytes(payload.encode("utf-16"))

    with pytest.raises(ValueError, match=r"encoded\.yaml carries an opaque payload"):
        _scan(tmp_path, _manifest(relative_path))


def test_manifest_rejects_paths_without_typed_codec() -> None:
    with pytest.raises(ValueError, match="no registered text codec"):
        _manifest("payload.bin")


def test_guard_scans_manifest_metadata(tmp_path: Path) -> None:
    manifest = _manifest()
    tampered = manifest.model_copy(
        update={
            "external_sources": [
                ExternalSourceSnapshot(
                    source_id="laundered",
                    uri="run://laundered/evidence",
                    content_hash="sha256:fixture",
                )
            ],
        }
    )
    with pytest.raises(ValueError, match=r"manifest\.yaml.*elicitation-run URI"):
        _scan(tmp_path, tampered)


@pytest.mark.parametrize(
    ("suffix", "payload"),
    [
        (".txt", '{"run_id":"downstream","model_id":"respondent"}'),
        (".txt", r'{"run\u005fid":"downstream"}'),
        (".md", 'Captured output: {"model_id": "respondent"}'),
        (".html", '<div data-output=\'{"run_id":"downstream"}\'></div>'),
        (".txt", "'provider_model_id' = 'respondent'"),
        (".txt", '{"basis":"elicited"}'),
        (".txt", '{"source_classification":"llm_preliminary"}'),
        (".yaml", 'notes: \'{"run_id":"downstream"}\''),
        (".csv", "run_id,model_id\ndownstream,respondent\n"),
        (".csv", '\ufeff"run_id","model_id"\n'),
        (".csv", "source_classification,value\nllm_preliminary,42\n"),
        (".csv", "basis,basis\nelicited,empirical\n"),
        (".csv", 'notes\n"{""model_id"":""respondent""}"\n'),
    ],
)
def test_all_text_codecs_reject_downstream_content(tmp_path, suffix, payload):
    relative = f"evidence{suffix}"
    (tmp_path / relative).write_text(payload)
    with pytest.raises(ValueError, match="downstream-only elicitation dependency"):
        _scan(tmp_path, _manifest(relative))


@pytest.mark.parametrize(
    ("suffix", "payload"),
    [
        (".json", '{"basis": "elicited", "basis": "empirical"}'),
        (
            ".json",
            '{"outer": [{"source_classification": "llm_preliminary",'
            ' "source_classification": "empirical"}]}',
        ),
        (".json", '{"score": NaN}'),
        (".yaml", "basis: elicited\nbasis: empirical\n"),
        (".yml", "outer:\n  - {basis: elicited, basis: empirical}\n"),
    ],
)
def test_structured_evidence_that_repeats_a_key_or_is_not_finite_is_malformed(
    tmp_path, suffix, payload
):
    relative = f"evidence{suffix}"
    (tmp_path / relative).write_text(payload)
    with pytest.raises(ValueError, match=r"structured source-bundle file.*malformed"):
        _scan(tmp_path, _manifest(relative))


def test_csv_source_model_ids_are_upstream_evidence(tmp_path):
    relative = "evidence.csv"
    (tmp_path / relative).write_text(
        "source_model_id,coi_exclusion_model_id,score\nobserved,excluded,150\n"
    )
    _scan(tmp_path, _manifest(relative))


def test_malformed_csv_is_rejected_instead_of_skipping(tmp_path):
    relative = "evidence.csv"
    (tmp_path / relative).write_text('notes\n"unterminated')
    with pytest.raises(ValueError, match=r"CSV source-bundle file.*malformed"):
        _scan(tmp_path, _manifest(relative))
