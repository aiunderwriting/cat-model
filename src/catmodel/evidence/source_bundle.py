"""Frozen upstream evidence for context-bound instruments and their setups.

A bundle captures selected observations, source excerpts and study assumptions.
Model structure remains in the repository; conditional judgments remain downstream.
Every read uses one immutable, checksum-verified set of bytes. The bundle's
identity is the SHA-256 of its manifest; how it was built is recorded beside the
manifest, outside that identity.
"""

import csv
import hashlib
import io
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from catmodel.contracts.governance import (
    AssumptionRegisterBase,
    ExternalSourceSnapshot,
    FileChecksum,
)
from catmodel.contracts.primitives import (
    RequiredIdentifier,
    RequiredString,
    Sha256Digest,
)
from catmodel.json_io import exact_marker, strict_json_loads
from catmodel.path_safety import contained_local_path
from catmodel.resources import scenarios_path
from catmodel.scenarios.base_rates import SourceCardsFile
from catmodel.scenarios.loader import load_scenarios
from catmodel.scenarios.threat_models import ThreatModelsFile, threat_model_errors
from catmodel.yaml_io import safe_load

SOURCE_BUNDLE_SCHEMA_VERSION = 1
MANIFEST_PATH = "manifest.yaml"
PROVENANCE_PATH = "provenance.yaml"
ASSUMPTION_REGISTER_PATH = "assumptions/assumption_register.yaml"
SOURCE_CARDS_PATH = "evidence/source_cards.yaml"
THREAT_MODELS_PATH = "evidence/threat_models.yaml"
REQUIRED_SOURCE_BUNDLE_FILES = frozenset({ASSUMPTION_REGISTER_PATH, SOURCE_CARDS_PATH})
OPTIONAL_SOURCE_BUNDLE_FILES = frozenset({THREAT_MODELS_PATH})
_SOURCE_BUNDLE_STRUCTURED_SUFFIXES = frozenset({".yaml", ".yml", ".json"})
_SOURCE_BUNDLE_TEXT_SUFFIXES = frozenset({".md", ".txt", ".csv", ".html"})
# ``threat_model_id`` names a threat model, not a language model.
_SOURCE_BUNDLE_MODEL_ID_ALLOWLIST = frozenset(
    {"source_model_id", "coi_exclusion_model_id", "threat_model_id"}
)
_SOURCE_BUNDLE_RUN_URI_RE = re.compile(
    r"\b(?:run|elicitation-run)://\S+", re.IGNORECASE
)
# An assignment written as structure inside text: a quoted key anywhere, as in
# JSON or a Python mapping, or a bare key at the start of a line, as in YAML.
# A bare key inside a sentence, such as "(model_id: GPT-4)", is prose.
_SOURCE_BUNDLE_ASSIGNMENT_RE = re.compile(
    r"(?im)(?=(?:(?P<quote>[\"'])(?P<quoted>[a-z0-9_-]+)(?P=quote)"
    r"|^[ \t]*(?P<bare>[a-z0-9_-]+))[ \t]*[:=][ \t]*"
    r"(?P<value>\"[^\"\r\n]*\"|'[^'\r\n]*'|[^,\r\n}\]]*))"
)
# A key names an elicitation identity when it mentions elicitation and ends in
# one of these; a key such as ``elicitation_method`` describes a cited method.
_SOURCE_BUNDLE_IDENTITY_SUFFIXES = ("_id", "_ids", "_run", "_sha256", "_digest", "_uri")
_SOURCE_BUNDLE_DOWNSTREAM_AUTHORITY_KEYS = frozenset(
    {
        "model_authority_sha256",
        "package_authority_sha256",
        "question_authority_sha256",
        "stipulated_risk_inputs_version",
        "stipulated_subject_eci_version",
    }
)


class SourceBundleAssumptionRegister(AssumptionRegisterBase):
    source_bundle_id: RequiredIdentifier
    source_bundle_version: RequiredString
    generated_at: RequiredString


def normalize_source_bundle_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    if not normalized:
        raise ValueError("path must not be empty")
    path = PurePosixPath(normalized)
    if path.is_absolute():
        raise ValueError(f"path must be bundle-relative, got absolute path {value!r}")
    if any(part == ".." for part in path.parts):
        raise ValueError(f"path must not traverse outside the bundle, got {value!r}")
    if normalized.endswith("/"):
        raise ValueError(f"path must reference a file, got directory path {value!r}")
    return str(path)


def _validate_relative_bundle_path(value: str) -> str:
    return normalize_source_bundle_path(value)


def _source_bundle_path(root: Path, relative_path: str | Path) -> Path:
    """Return one real, bundle-contained path for a read of the bundle directory."""
    return contained_local_path(
        root,
        normalize_source_bundle_path(str(relative_path)),
        artifact_label="source-bundle",
    )


def _source_bundle_file_suffix(relative_path: str) -> str:
    normalized = normalize_source_bundle_path(relative_path)
    suffix = PurePosixPath(normalized).suffix.lower()
    if suffix not in _SOURCE_BUNDLE_STRUCTURED_SUFFIXES | _SOURCE_BUNDLE_TEXT_SUFFIXES:
        raise ValueError(
            f"source-bundle path {relative_path!r} has no registered text codec "
            f"for suffix {suffix!r}; opaque/binary files are not admissible "
            "bundle surfaces"
        )
    return suffix


class SourceBundleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle_schema_version: Literal[1]
    bundle_id: RequiredIdentifier
    bundle_version: RequiredString
    created_at: RequiredString
    files: dict[str, FileChecksum]
    external_sources: list[ExternalSourceSnapshot] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_file_paths(self) -> Self:
        normalized_files: dict[str, FileChecksum] = {}
        raw_by_normalized: dict[str, str] = {}
        for raw_path, ref in self.files.items():
            normalized_path = _validate_relative_bundle_path(raw_path)
            if normalized_path in normalized_files:
                raise ValueError(
                    "manifest files mapping contains paths that collide after "
                    f"normalization: {raw_by_normalized[normalized_path]!r} "
                    f"and {raw_path!r} both normalize to {normalized_path!r}"
                )
            _source_bundle_file_suffix(normalized_path)
            normalized_files[normalized_path] = ref
            raw_by_normalized[normalized_path] = raw_path
        self.files = normalized_files
        if self.files.keys() & {MANIFEST_PATH, PROVENANCE_PATH}:
            raise ValueError(
                "manifest files mapping must not include manifest.yaml or provenance.yaml"
            )
        source_ids = [source.source_id for source in self.external_sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("external_sources must use unique source_id values")
        for source in self.external_sources:
            untracked_links = sorted(set(source.captured_paths) - set(self.files))
            if untracked_links:
                raise ValueError(
                    f"external source {source.source_id!r} links captured "
                    f"paths not tracked by manifest files: {untracked_links}"
                )
        return self


class SourceBundleProvenance(BaseModel):
    """How a bundle was built, recorded beside its manifest outside its identity.

    It names the manifest it describes, the clean method commit, the model-wide
    ``assumptions.md`` and the hierarchy contract at that commit, and the
    source-input recipe. Rebuilding the same evidence at another commit, or
    after an ``assumptions.md`` or hierarchy edit, display text included,
    changes this record, not the bundle identity that instruments bind.
    Readers check only that it names their manifest and carries no downstream
    content; instruments check their scenario nodes and any taxonomy parents
    against the current hierarchy directly.
    """

    model_config = ConfigDict(extra="forbid")

    manifest_sha256: Sha256Digest
    repo_git_sha: str = Field(pattern=r"^[0-9a-f]{40}([0-9a-f]{24})?$")
    assumptions_md_sha256: Sha256Digest
    hierarchy_version: RequiredString
    hierarchy_sha256: Sha256Digest
    source_input_recipe_sha256: Sha256Digest


@dataclass(frozen=True, slots=True)
class SourceBundleSnapshot:
    """Source identity and immutable file bytes captured by one bundle intake.

    Paths locate original evidence for reporting only. Consumers read this mapping,
    so edits to the source directory cannot change an in-progress operation. The
    mapping holds the manifest, every tracked file and, when present, the
    build provenance record.
    """

    root: Path
    manifest: SourceBundleManifest
    files: Mapping[str, bytes]

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))

    def _relative_path(self, path: Path | str) -> str:
        candidate = Path(path)
        if candidate.is_absolute():
            candidate = candidate.relative_to(self.root)
        return normalize_source_bundle_path(candidate.as_posix())

    def read_bytes(self, path: Path | str) -> bytes:
        relative = self._relative_path(path)
        try:
            return self.files[relative]
        except KeyError:
            raise FileNotFoundError(self.root / relative) from None

    def read_text(self, path: Path | str) -> str:
        return self.read_bytes(path).decode("utf-8")

    def exists(self, path: Path | str) -> bool:
        return self._relative_path(path) in self.files

    def sha256(self, path: Path | str) -> str:
        return hashlib.sha256(self.read_bytes(path)).hexdigest()

    @property
    def manifest_sha256(self) -> str:
        """The bundle's identity: the SHA-256 of its manifest bytes."""
        return self.sha256(MANIFEST_PATH)


def _source_bundle_forbidden_identity_key(field_name: str) -> str | None:
    normalized = field_name.strip().lower().replace("-", "_")
    if "elicitation" in normalized and normalized.endswith(
        _SOURCE_BUNDLE_IDENTITY_SUFFIXES
    ):
        return "elicitation identity"
    if normalized == "run_id" or normalized.endswith("_run_id"):
        return "run identity"
    if normalized in _SOURCE_BUNDLE_DOWNSTREAM_AUTHORITY_KEYS:
        return "downstream authority identity"
    if (
        normalized == "model_id" or normalized.endswith("_model_id")
    ) and normalized not in _SOURCE_BUNDLE_MODEL_ID_ALLOWLIST:
        return "model identity"
    return None


def validate_source_bundle_schema_dependency_direction() -> None:
    """Reject this module's schema fields that name downstream or model identities."""

    violations: list[str] = []
    for name, candidate in sorted(globals().items()):
        if not isinstance(candidate, type) or candidate.__module__ != __name__:
            continue
        field_names: tuple[str, ...]
        try:
            is_pydantic_model = issubclass(candidate, BaseModel)
        except TypeError:  # pragma: no cover - guarded by isinstance(type)
            is_pydantic_model = False
        if is_pydantic_model:
            field_names = tuple(candidate.model_fields)
        elif is_dataclass(candidate):
            field_names = tuple(field.name for field in dataclass_fields(candidate))
        else:
            continue
        for field_name in field_names:
            reason = _source_bundle_forbidden_identity_key(field_name)
            if reason is not None:
                violations.append(f"{name}.{field_name} exposes {reason}")
    if violations:
        raise ValueError(
            "Source-bundle schema violates the downstream-only elicitation "
            "dependency: " + "; ".join(violations)
        )


def _source_bundle_json_path(parent: str, key: object) -> str:
    if isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key):
        return f"{parent}.{key}"
    return f"{parent}[{key!r}]"


def _source_bundle_content_violations(
    value: object,
    *,
    relative_path: str,
    json_path: str = "$",
) -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = _source_bundle_json_path(json_path, raw_key)
            reason = _source_bundle_forbidden_identity_key(key)
            if reason is not None:
                violations.append(
                    f"{relative_path}:{child_path} carries forbidden {reason} key"
                )
            normalized_key = key.strip().lower().replace("-", "_")
            if (
                normalized_key == "evidence_status"
                and child == "stipulated_unvalidated"
            ):
                violations.append(
                    f"{relative_path}:{child_path} carries downstream stipulated risk inputs"
                )
            if (
                normalized_key == "basis"
                and isinstance(child, str)
                and child.strip().lower() == "elicited"
            ):
                violations.append(
                    f"{relative_path}:{child_path} carries basis='elicited'"
                )
            if (
                normalized_key == "source_classification"
                and isinstance(child, str)
                and child.strip().lower() == "llm_preliminary"
            ):
                violations.append(
                    f"{relative_path}:{child_path} carries "
                    "source_classification='llm_preliminary'"
                )
            violations.extend(
                _source_bundle_content_violations(
                    child,
                    relative_path=relative_path,
                    json_path=child_path,
                )
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(
                _source_bundle_content_violations(
                    child,
                    relative_path=relative_path,
                    json_path=f"{json_path}[{index}]",
                )
            )
    elif isinstance(value, str):
        violations.extend(
            _source_bundle_text_violations(
                value, relative_path=f"{relative_path}:{json_path}"
            )
        )
    return violations


def _source_bundle_text_violations(text: str, *, relative_path: str) -> list[str]:
    violations: list[str] = []
    # A text attachment or quoted YAML value may itself carry a JSON object.
    # Parse it so escaped key spellings cannot bypass the identity check.
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, (dict, list)):
        violations.extend(
            _source_bundle_content_violations(parsed, relative_path=relative_path)
        )
    elif isinstance(parsed, str) and parsed != text:
        violations.extend(
            _source_bundle_text_violations(parsed, relative_path=relative_path)
        )
    for match in _SOURCE_BUNDLE_ASSIGNMENT_RE.finditer(text):
        key = match.group("quoted") or match.group("bare")
        value = match.group("value").strip().strip("'\"")
        reason = _source_bundle_forbidden_identity_key(key)
        if reason is not None:
            violations.append(
                f"{relative_path}:line {text.count(chr(10), 0, match.start()) + 1} "
                f"carries forbidden {reason} assignment"
            )
        normalized_key = key.strip().lower().replace("-", "_")
        if (
            normalized_key == "evidence_status"
            and value.lower() == "stipulated_unvalidated"
        ):
            violations.append(
                f"{relative_path}:line {text.count(chr(10), 0, match.start()) + 1} "
                "carries downstream stipulated risk inputs"
            )
        if normalized_key == "basis" and value.lower() == "elicited":
            violations.append(
                f"{relative_path}:line {text.count(chr(10), 0, match.start()) + 1} "
                "carries basis='elicited'"
            )
        if (
            normalized_key == "source_classification"
            and value.lower() == "llm_preliminary"
        ):
            violations.append(
                f"{relative_path}:line {text.count(chr(10), 0, match.start()) + 1} "
                "carries source_classification='llm_preliminary'"
            )
    for line_number, line in enumerate(text.splitlines(), start=1):
        if _SOURCE_BUNDLE_RUN_URI_RE.search(line):
            violations.append(
                f"{relative_path}:line {line_number} carries an elicitation-run URI"
            )
    return violations


def _source_bundle_csv_violations(text: str, *, relative_path: str) -> list[str]:
    """Treat CSV headers as schema keys and inspect every cell's text content."""
    reader = csv.reader(io.StringIO(text.lstrip("\ufeff")), strict=True)
    try:
        headers = next(reader, [])
        violations = _source_bundle_content_violations(
            dict.fromkeys(headers), relative_path=relative_path
        )
        for row_number, row in enumerate(reader, start=2):
            # Keep duplicate columns rather than allowing a later cell to hide
            # an earlier downstream classification with the same header.
            for column, value in enumerate(row):
                key = headers[column] if column < len(headers) else str(column)
                violations.extend(
                    _source_bundle_content_violations(
                        {key: value},
                        relative_path=relative_path,
                        json_path=f"$[row {row_number}, column {column + 1}]",
                    )
                )
    except csv.Error as exc:
        raise ValueError(
            f"Manifest-tracked CSV source-bundle file {relative_path!r} is malformed: {exc}"
        ) from exc
    return violations


def validate_source_bundle_dependency_direction(snapshot: SourceBundleSnapshot) -> None:
    """Enforce the downstream-only elicitation dependency for the manifest's
    closed world, reading the captured bytes only.

    Tracked files are decoded by canonical suffix and manifest metadata is
    scanned; undecodable payloads are violations, and untracked files are
    rejected by checksum validation. Structured content (YAML, JSON, CSV
    headers and cells, and any text that is itself JSON) is checked key by
    key: no run, model, elicitation or downstream-authority identity key, and
    no elicited basis, preliminary LLM classification or stipulated risk
    input. Other text is prose: only an assignment written as structure (a
    quoted key, or a bare key at the start of a line) and a ``run://`` or
    ``elicitation-run://`` URI are refused there.
    """

    validate_source_bundle_schema_dependency_direction()
    manifest = snapshot.manifest
    violations: list[str] = []
    violations.extend(
        _source_bundle_content_violations(
            manifest.model_dump(mode="json"),
            relative_path=MANIFEST_PATH,
        )
    )
    for relative_path in sorted(manifest.files):
        suffix = _source_bundle_file_suffix(relative_path)
        try:
            text = snapshot.read_text(relative_path)
        except UnicodeDecodeError:
            violations.append(
                f"{relative_path} carries an opaque payload its {suffix} text "
                "codec cannot decode"
            )
            continue
        if suffix in _SOURCE_BUNDLE_STRUCTURED_SUFFIXES:
            try:
                parsed = (
                    strict_json_loads(text) if suffix == ".json" else safe_load(text)
                )
            except Exception as exc:
                raise ValueError(
                    f"Manifest-tracked structured source-bundle file "
                    f"{relative_path!r} is malformed: {exc}"
                ) from exc
            violations.extend(
                _source_bundle_content_violations(
                    parsed,
                    relative_path=relative_path,
                )
            )
        elif suffix == ".csv":
            violations.extend(
                _source_bundle_csv_violations(text, relative_path=relative_path)
            )
        else:
            violations.extend(
                _source_bundle_text_violations(text, relative_path=relative_path)
            )
    if violations:
        raise ValueError(
            "Source bundle violates the downstream-only elicitation dependency: "
            + "; ".join(violations)
        )


def require_current_source_bundle_schema(
    raw_schema_version: object,
    *,
    manifest_path: Path,
) -> None:
    if exact_marker(raw_schema_version, SOURCE_BUNDLE_SCHEMA_VERSION):
        return
    raise ValueError(
        "Unsupported source-bundle schema version "
        f"{raw_schema_version!r} in {manifest_path}; expected "
        f"{SOURCE_BUNDLE_SCHEMA_VERSION}. Re-author or rebuild the bundle at "
        "the current schema into a new directory; frozen evidence is never rewritten "
        "and other schema versions are not read."
    )


def parse_source_bundle_manifest(
    data: dict[str, Any], *, manifest_path: Path
) -> SourceBundleManifest:
    require_current_source_bundle_schema(
        data.get("bundle_schema_version"), manifest_path=manifest_path
    )
    try:
        manifest = SourceBundleManifest.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"Invalid source-bundle manifest {manifest_path}: {exc}"
        ) from exc
    return manifest


def capture_source_bundle_snapshot(root: Path) -> SourceBundleSnapshot:
    """Read once, validate those bytes and retain them for the whole operation."""
    root = root.expanduser().resolve()
    manifest_path = _source_bundle_path(root, MANIFEST_PATH)
    manifest_bytes = manifest_path.read_bytes()
    try:
        data = safe_load(manifest_bytes.decode("utf-8")) or {}
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Failed to parse source-bundle manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"source-bundle manifest {manifest_path} must be a YAML mapping"
        )
    manifest = parse_source_bundle_manifest(data, manifest_path=manifest_path)
    files = validate_source_bundle_checksums(root, manifest)
    provenance_path = _source_bundle_path(root, PROVENANCE_PATH)
    if provenance_path.is_file():
        files[PROVENANCE_PATH] = provenance_path.read_bytes()
    snapshot = SourceBundleSnapshot(
        root=root,
        manifest=manifest,
        files={MANIFEST_PATH: manifest_bytes, **files},
    )
    validate_source_bundle_snapshot(snapshot)
    return snapshot


def validate_source_bundle_snapshot(snapshot: SourceBundleSnapshot) -> None:
    """Validate a captured bundle without reading its original directory."""
    manifest = snapshot.manifest
    missing_required = sorted(REQUIRED_SOURCE_BUNDLE_FILES - set(manifest.files))
    if missing_required:
        raise ValueError(f"manifest is missing required files {missing_required}")
    if set(snapshot.files) - {PROVENANCE_PATH} != {MANIFEST_PATH, *manifest.files}:
        raise ValueError(
            "captured source files differ from the manifest-tracked closure"
        )
    if snapshot.exists(PROVENANCE_PATH):
        _validate_source_bundle_provenance(snapshot)
    for relative, checksum in manifest.files.items():
        if snapshot.sha256(relative) != checksum.sha256:
            raise ValueError(f"captured source checksum mismatch: {relative}")
    validate_source_bundle_dependency_direction(snapshot)
    _load_model(
        snapshot, SOURCE_CARDS_PATH, model=SourceCardsFile, label="source cards"
    )
    _load_source_bundle_assumption_register(snapshot)
    load_source_bundle_threat_models(snapshot)


def load_source_bundle_threat_models(
    snapshot: SourceBundleSnapshot,
) -> ThreatModelsFile:
    """The bundle's threat models; an empty register when it captures none.

    Each is for a library scenario and was written against that scenario's
    current definition, so a bundle captured before a scenario changed is
    refused until its threat model is revised for the new definition.
    """
    if not snapshot.exists(THREAT_MODELS_PATH):
        return ThreatModelsFile()
    register = _load_model(
        snapshot, THREAT_MODELS_PATH, model=ThreatModelsFile, label="threat models"
    )
    if not register.threat_models:
        return register
    errors = threat_model_errors(
        register.threat_models, load_scenarios(scenarios_path())
    )
    if errors:
        raise ValueError("; ".join(errors))
    return register


def _validate_source_bundle_provenance(snapshot: SourceBundleSnapshot) -> None:
    """The build provenance names this manifest and carries no downstream content."""
    path = snapshot.root / PROVENANCE_PATH
    data = _load_yaml_mapping(
        snapshot, PROVENANCE_PATH, label="source-bundle provenance"
    )
    violations = _source_bundle_content_violations(data, relative_path=PROVENANCE_PATH)
    if violations:
        raise ValueError(
            "Source bundle violates the downstream-only elicitation dependency: "
            + "; ".join(violations)
        )
    try:
        provenance = SourceBundleProvenance.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid source-bundle provenance {path}: {exc}") from exc
    if provenance.manifest_sha256 != snapshot.manifest_sha256:
        raise ValueError("source-bundle provenance describes a different manifest")


def list_source_bundle_files(root: Path) -> set[str]:
    resolved_root = root.expanduser().resolve()
    files: set[str] = set()
    for candidate in resolved_root.rglob("*"):
        if not candidate.is_file():
            continue
        relative_path = candidate.relative_to(resolved_root).as_posix()
        _source_bundle_path(resolved_root, relative_path)
        files.add(relative_path)
    return files


def validate_source_bundle_checksums(
    root: Path, manifest: SourceBundleManifest
) -> dict[str, bytes]:
    """Validate the manifest checksum closure and the bundle's closed world.

    Untracked physical files are rejected because the dependency-direction guard
    scans only manifest-tracked content.
    """
    missing_manifest_entries = sorted(
        REQUIRED_SOURCE_BUNDLE_FILES - set(manifest.files)
    )
    untracked_optional_entries = sorted(
        relative_path
        for relative_path in OPTIONAL_SOURCE_BUNDLE_FILES
        if _source_bundle_path(root, relative_path).exists()
        and relative_path not in manifest.files
    )
    untracked_files = sorted(
        list_source_bundle_files(root)
        - set(manifest.files)
        - set(untracked_optional_entries)
        - {MANIFEST_PATH, PROVENANCE_PATH}
    )
    files: dict[str, bytes] = {}
    missing: list[str] = []
    mismatched: list[str] = []
    for relative_path, ref in sorted(manifest.files.items()):
        path = _source_bundle_path(root, relative_path)
        if not path.exists():
            missing.append(relative_path)
            continue
        raw = path.read_bytes()
        files[relative_path] = raw
        actual = hashlib.sha256(raw).hexdigest()
        if actual != ref.sha256:
            mismatched.append(f"{relative_path}: expected {ref.sha256}, got {actual}")
    messages = []
    if missing_manifest_entries:
        messages.append(
            f"manifest is missing required files {missing_manifest_entries}"
        )
    if untracked_optional_entries:
        messages.append(
            f"manifest is missing present optional files {untracked_optional_entries}"
        )
    if untracked_files:
        messages.append(
            f"bundle contains files the manifest does not track {untracked_files}"
        )
    if missing:
        messages.append(f"missing files {missing}")
    if mismatched:
        messages.append(f"checksum mismatches {mismatched}")
    if messages:
        raise ValueError(
            "Source-bundle manifest validation failed: " + "; ".join(messages)
        )

    return files


def _load_source_bundle_assumption_register(
    snapshot: SourceBundleSnapshot,
) -> SourceBundleAssumptionRegister:
    manifest = snapshot.manifest
    register = _load_model(
        snapshot,
        ASSUMPTION_REGISTER_PATH,
        label="source-bundle assumption register",
        model=SourceBundleAssumptionRegister,
    )
    if register.source_bundle_id != manifest.bundle_id:
        raise ValueError(
            "Source-bundle assumption register source_bundle_id "
            f"{register.source_bundle_id!r} does not match manifest bundle_id "
            f"{manifest.bundle_id!r}"
        )
    if register.source_bundle_version != manifest.bundle_version:
        raise ValueError(
            "Source-bundle assumption register source_bundle_version "
            f"{register.source_bundle_version!r} does not match manifest "
            f"bundle_version {manifest.bundle_version!r}"
        )
    return register


def _load_yaml_mapping(
    snapshot: SourceBundleSnapshot, relative_path: str, *, label: str
) -> dict[str, Any]:
    """One captured YAML mapping; errors name where its original was read."""
    path = snapshot.root / relative_path
    try:
        data = safe_load(snapshot.read_text(relative_path)) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} not found: {path}") from None
    except Exception as exc:
        raise ValueError(f"Failed to parse {label} {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} {path} must be a YAML mapping")
    return data


def _load_model[T: BaseModel](
    snapshot: SourceBundleSnapshot, relative_path: str, *, label: str, model: type[T]
) -> T:
    data = _load_yaml_mapping(snapshot, relative_path, label=label)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"Invalid {label} {snapshot.root / relative_path}: {exc}"
        ) from exc
