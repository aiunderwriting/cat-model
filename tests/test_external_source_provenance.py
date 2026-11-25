import pytest

from catmodel.contracts.governance import (
    ExternalSourceSnapshot,
)
from catmodel.evidence.source_bundle import (
    SOURCE_BUNDLE_SCHEMA_VERSION,
    SourceBundleManifest,
)


def _source(
    *,
    captured_paths: list[str] | None = None,
) -> ExternalSourceSnapshot:
    return ExternalSourceSnapshot(
        source_id="source-a",
        uri="https://example.test/source-a",
        content_hash="sha256:test",
        captured_paths=captured_paths or [],
    )


def _manifest(*, sources: list[ExternalSourceSnapshot]) -> SourceBundleManifest:
    return SourceBundleManifest(
        bundle_schema_version=SOURCE_BUNDLE_SCHEMA_VERSION,
        bundle_id="test-bundle",
        bundle_version="2026-01-01",
        created_at="2026-01-01T00:00:00+00:00",
        files={},
        external_sources=sources,
    )


def test_source_bundle_rejects_untracked_external_source_captured_path() -> None:
    with pytest.raises(ValueError, match="paths not tracked by manifest files"):
        _manifest(sources=[_source(captured_paths=["evidence/observations.csv"])])


def test_source_bundle_rejects_duplicate_external_source_ids() -> None:
    with pytest.raises(ValueError, match="unique source_id"):
        _manifest(sources=[_source(), _source()])
