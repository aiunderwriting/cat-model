"""Capture explicitly selected upstream evidence in an immutable source bundle.

The recipe selects source files and records their provenance. Every source hash
is derived from captured bytes.
"""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from catmodel.contracts.governance import ExternalSourceSnapshot
from catmodel.contracts.primitives import RequiredString
from catmodel.evidence.source_bundle import (
    MANIFEST_PATH,
    PROVENANCE_PATH,
    SOURCE_BUNDLE_SCHEMA_VERSION,
    SourceBundleManifest,
    SourceBundleProvenance,
    capture_source_bundle_snapshot,
    normalize_source_bundle_path,
)
from catmodel.hashing import sha256_file
from catmodel.path_safety import contained_local_path
from catmodel.resources import source_checkout_root
from catmodel.scenarios.hierarchy_contract import hierarchy_authority
from catmodel.worktree import resolve_code_provenance
from catmodel.yaml_io import safe_dump, safe_load


class SourceInputReference(BaseModel):
    """One captured file and the authored YAML file, beside the recipe, that
    records its provenance."""

    model_config = ConfigDict(extra="forbid")

    snapshot_path: str
    metadata_file: str


class SourceInputRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle_id: RequiredString
    bundle_version: RequiredString
    created_at: RequiredString
    files: dict[str, str] = Field(min_length=1)
    sources: list[SourceInputReference] = Field(default_factory=list)


def _yaml_bytes(value: Any) -> bytes:
    return safe_dump(value, sort_keys=False).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_inventory(
    recipe: SourceInputRecipe, files: dict[str, bytes], root: Path
) -> list[ExternalSourceSnapshot]:
    sources = []
    for source in recipe.sources:
        path = normalize_source_bundle_path(source.snapshot_path)
        if path not in files:
            raise ValueError(f"Source snapshot is not an exported file: {path}")
        metadata = safe_load(
            contained_local_path(
                root, source.metadata_file, artifact_label="source metadata"
            ).read_bytes()
        )
        if not isinstance(metadata, dict):
            raise ValueError("source metadata file must contain a mapping")
        if "content_hash" in metadata:
            raise ValueError("Source hashes are generated from captured bytes")
        record = ExternalSourceSnapshot.model_validate(
            {**metadata, "content_hash": _digest(files[path])}
        )
        if path not in record.captured_paths:
            record.captured_paths.insert(0, path)
        sources.append(record)
    return sources


def build_source_bundle(
    recipe_path: Path,
    output_path: Path,
) -> SourceBundleManifest:
    """Validate captured evidence before publishing a new immutable directory.

    Recipe paths resolve beside the recipe. The selected bytes, not later
    reads of mutable authoring files, determine every exported checksum. The
    method commit, ``assumptions.md`` digest, hierarchy contract and recipe
    digest are recorded in ``provenance.yaml``, outside the manifest whose
    digest is the bundle's identity, so the same evidence rebuilt elsewhere
    keeps that identity.
    """
    recipe_path = recipe_path.expanduser().resolve()
    recipe_bytes = recipe_path.read_bytes()
    recipe = SourceInputRecipe.model_validate(safe_load(recipe_bytes))
    output_path = output_path.expanduser().absolute()
    if output_path.exists() or output_path.is_symlink():
        raise ValueError(f"Source-bundle output must be new: {output_path}")
    if any(
        (parent / MANIFEST_PATH).is_file()
        for parent in (
            output_path.parent.resolve(),
            *output_path.parent.resolve().parents,
        )
    ):
        raise ValueError("Source output must remain outside every existing bundle")
    files: dict[str, bytes] = {}
    for destination, source in recipe.files.items():
        destination = normalize_source_bundle_path(destination)
        if destination in (MANIFEST_PATH, PROVENANCE_PATH):
            raise ValueError(
                "Generated manifest and provenance cannot be authored inputs"
            )
        if destination in files:
            raise ValueError(f"Duplicate normalized destination: {destination}")
        path = contained_local_path(
            recipe_path.parent, source, artifact_label="source input"
        )
        files[destination] = path.read_bytes()
    sources = _source_inventory(recipe, files, recipe_path.parent)

    code = resolve_code_provenance(None)
    commit = code.commit
    if code.tree_state != "clean" or commit is None:
        raise ValueError("Source authoring requires a clean committed method checkout")
    root = source_checkout_root()
    if root is None:
        raise ValueError("Source authoring requires the model's source checkout")
    hierarchy = hierarchy_authority()
    manifest = SourceBundleManifest.model_validate(
        {
            "bundle_schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
            "bundle_id": recipe.bundle_id,
            "bundle_version": recipe.bundle_version,
            "created_at": recipe.created_at,
            "files": {
                path: {"sha256": _digest(content)}
                for path, content in sorted(files.items())
            },
            "external_sources": [
                source.model_dump(mode="json", exclude_none=True) for source in sources
            ],
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".source-build-", dir=output_path.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        for path, content in files.items():
            destination = stage / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        manifest_bytes = _yaml_bytes(
            manifest.model_dump(mode="json", exclude_none=True)
        )
        (stage / MANIFEST_PATH).write_bytes(manifest_bytes)
        provenance = SourceBundleProvenance(
            manifest_sha256=_digest(manifest_bytes),
            repo_git_sha=commit,
            assumptions_md_sha256=sha256_file(root / "assumptions.md"),
            hierarchy_version=hierarchy.hierarchy_version,
            hierarchy_sha256=hierarchy.hierarchy_sha256,
            source_input_recipe_sha256=_digest(recipe_bytes),
        )
        (stage / PROVENANCE_PATH).write_bytes(
            _yaml_bytes(provenance.model_dump(mode="json"))
        )
        capture_source_bundle_snapshot(stage)
        current_code = resolve_code_provenance(None)
        if current_code.tree_state != "clean" or current_code.commit != commit:
            raise ValueError(
                "Method checkout changed during source build; rerun from a clean fixed commit"
            )
        stage.rename(output_path)
    return manifest
