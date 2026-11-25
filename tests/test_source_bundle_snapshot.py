"""Captured capability evidence is portable, complete, and immutable during use."""

import hashlib
import shutil
from pathlib import Path
from typing import cast

import pytest
import yaml

from catmodel.evidence import source_bundle as bundles
from test_support.capability_sources import CSV_PATH, copy_source_bundle


def _restamp(root: Path, relative: str) -> None:
    manifest_path = root / bundles.MANIFEST_PATH
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["files"][relative] = {
        "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest()
    }
    manifest_path.write_text(yaml.safe_dump(manifest))


def test_snapshot_preserves_manifest_and_payload_bytes(tmp_path, monkeypatch):
    root = copy_source_bundle(tmp_path)
    expected_manifest = (root / bundles.MANIFEST_PATH).read_bytes()
    original = bundles.validate_source_bundle_checksums

    def validate_then_replace(root, manifest):
        files = original(root, manifest)
        (root / bundles.MANIFEST_PATH).write_text("changed: true")
        (root / CSV_PATH).write_text("changed")
        return files

    monkeypatch.setattr(
        bundles, "validate_source_bundle_checksums", validate_then_replace
    )
    snapshot = bundles.capture_source_bundle_snapshot(root)
    assert snapshot.read_bytes(bundles.MANIFEST_PATH) == expected_manifest
    assert snapshot.manifest_sha256 == hashlib.sha256(expected_manifest).hexdigest()
    assert snapshot.read_bytes(CSV_PATH) != b"changed"


def test_snapshot_validates_without_original_files(tmp_path):
    root = copy_source_bundle(tmp_path)
    snapshot = bundles.capture_source_bundle_snapshot(root)
    cards = snapshot.read_bytes(bundles.SOURCE_CARDS_PATH)
    shutil.rmtree(root)
    bundles.validate_source_bundle_snapshot(snapshot)
    assert snapshot.read_bytes(bundles.SOURCE_CARDS_PATH) == cards
    with pytest.raises(TypeError):
        cast(dict[str, bytes], snapshot.files)[CSV_PATH] = b"changed"


@pytest.mark.parametrize("relative", sorted(bundles.REQUIRED_SOURCE_BUNDLE_FILES))
def test_all_required_evidence_is_required_even_in_captured_packages(
    tmp_path, relative
):
    snapshot = bundles.capture_source_bundle_snapshot(copy_source_bundle(tmp_path))
    manifest = snapshot.manifest.model_copy(
        update={
            "files": {
                key: value
                for key, value in snapshot.manifest.files.items()
                if key != relative
            }
        }
    )
    files = {key: value for key, value in snapshot.files.items() if key != relative}
    incomplete = bundles.SourceBundleSnapshot(snapshot.root, manifest, files)
    with pytest.raises(ValueError, match="missing required files"):
        bundles.validate_source_bundle_snapshot(incomplete)


def test_untracked_files_and_captured_tampering_are_rejected(tmp_path):
    root = copy_source_bundle(tmp_path)
    snapshot = bundles.capture_source_bundle_snapshot(root)
    (root / "untracked.txt").write_text("unselected evidence")
    with pytest.raises(ValueError, match="manifest does not track"):
        bundles.capture_source_bundle_snapshot(root)
    forged = bundles.SourceBundleSnapshot(
        snapshot.root, snapshot.manifest, {**snapshot.files, CSV_PATH: b"changed"}
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        bundles.validate_source_bundle_snapshot(forged)


@pytest.mark.parametrize("schema", [None, 0, 2, True, 1.0, "1"])
def test_other_schemas_are_rejected_without_rewriting(tmp_path, schema):
    root = copy_source_bundle(tmp_path)
    path = root / bundles.MANIFEST_PATH
    manifest = yaml.safe_load(path.read_text())
    manifest["bundle_schema_version"] = schema
    path.write_text(yaml.safe_dump(manifest))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Unsupported source-bundle schema"):
        bundles.capture_source_bundle_snapshot(root)
    assert path.read_bytes() == original


def test_a_manifest_repeating_a_key_is_rejected(tmp_path):
    root = copy_source_bundle(tmp_path)
    bundles.capture_source_bundle_snapshot(root)
    with open(root / bundles.MANIFEST_PATH, "a") as manifest:
        manifest.write("created_at: '2026-09-02T00:00:00Z'\n")
    with pytest.raises(ValueError, match="duplicate key 'created_at'"):
        bundles.capture_source_bundle_snapshot(root)


def test_assumption_register_is_bound_to_bundle_identity(tmp_path):
    root = copy_source_bundle(tmp_path)
    path = root / bundles.ASSUMPTION_REGISTER_PATH
    payload = yaml.safe_load(path.read_text())
    payload["source_bundle_id"] = "different-study"
    path.write_text(yaml.safe_dump(payload))
    _restamp(root, bundles.ASSUMPTION_REGISTER_PATH)
    with pytest.raises(ValueError, match="does not match manifest"):
        bundles.capture_source_bundle_snapshot(root)


def test_source_cards_are_typed_and_required_for_diagnostics(tmp_path):
    root = copy_source_bundle(tmp_path)
    (root / bundles.SOURCE_CARDS_PATH).write_text("cards: []\nscenarios: {}")
    _restamp(root, bundles.SOURCE_CARDS_PATH)
    with pytest.raises(ValueError, match="must list every library scenario"):
        bundles.capture_source_bundle_snapshot(root)


def test_source_symlink_cannot_escape_bundle(tmp_path):
    root = copy_source_bundle(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_bytes((root / CSV_PATH).read_bytes())
    (root / CSV_PATH).unlink()
    (root / CSV_PATH).symlink_to(outside)
    with pytest.raises(ValueError, match="must remain inside its root"):
        bundles.capture_source_bundle_snapshot(root)


@pytest.mark.parametrize(
    ("relative", "payload"),
    [
        ("evidence/downstream.txt", '{"run_id":"downstream"}'),
        ("evidence/downstream.csv", "model_id,value\nrespondent,42\n"),
    ],
)
def test_tracked_downstream_text_is_rejected_on_disk_and_in_snapshots(
    tmp_path, relative, payload
):
    root = copy_source_bundle(tmp_path)
    (root / relative).write_text(payload)
    _restamp(root, relative)
    with pytest.raises(ValueError, match="downstream-only elicitation dependency"):
        bundles.capture_source_bundle_snapshot(root)

    manifest = bundles.SourceBundleManifest.model_validate(
        yaml.safe_load((root / bundles.MANIFEST_PATH).read_text())
    )
    snapshot = bundles.SourceBundleSnapshot(
        root,
        manifest,
        {
            name: (root / name).read_bytes()
            for name in [bundles.MANIFEST_PATH, *manifest.files]
        },
    )
    shutil.rmtree(root)
    with pytest.raises(ValueError, match="downstream-only elicitation dependency"):
        bundles.validate_source_bundle_snapshot(snapshot)
