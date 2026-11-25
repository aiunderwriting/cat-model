"""Synthetic upstream evidence fixtures for source-bundle tests.

No calibration, study package or elicitation output is used to build them.
"""

import hashlib
from pathlib import Path

from catmodel.evidence.source_bundle import (
    ASSUMPTION_REGISTER_PATH,
    MANIFEST_PATH,
    SOURCE_BUNDLE_SCHEMA_VERSION,
    SOURCE_CARDS_PATH,
)
from catmodel.resources import scenarios_path
from catmodel.scenarios.loader import load_scenarios
from catmodel.yaml_io import safe_dump

CSV_PATH = "evidence/official-eci.csv"
CSV_CONTENT = b"Model,ECI\nsynthetic-release,150.0\n"
AS_OF = "2026-09-01T00:00:00Z"
BUNDLE_ID = "synthetic-capability-evidence"
BUNDLE_VERSION = "2026-09-01"


def authored_source_files() -> dict[str, bytes]:
    """Return a complete upstream fixture without generated numeric parameters."""
    assumptions = {
        "source_bundle_id": BUNDLE_ID,
        "source_bundle_version": BUNDLE_VERSION,
        "generated_at": AS_OF,
        "assumptions": [
            {
                "assumption_id": "synthetic-availability-capture",
                "title": "Synthetic public availability evidence",
                "applies_to": ["source_evidence"],
                "source_classification": "internal_judgment",
                "status": "PLACEHOLDER",
                "uncertainty_type": "epistemic",
                "materiality": "high",
                "rationale": "A single synthetic observation isolates evidence-binding software behavior.",
                "empirical_grounding": "Synthetic software fixture; no empirical risk claim.",
                "sensitivity_range": "Tests perturb observed scores and evidence bytes.",
                "sensitivity_tested": False,
                "evidence_refs": ["urn:fixture:synthetic-boundary-capture"],
            }
        ],
    }
    cards = {
        "cards": [],
        "scenarios": {
            scenario_id: []
            for scenario_id in sorted(
                row.id for row in load_scenarios(scenarios_path())
            )
        },
    }
    return {
        ASSUMPTION_REGISTER_PATH: safe_dump(assumptions, sort_keys=False).encode(),
        SOURCE_CARDS_PATH: safe_dump(cards, sort_keys=False).encode(),
        CSV_PATH: CSV_CONTENT,
    }


def write_source_bundle(
    root: Path,
    *,
    files: dict[str, bytes] | None = None,
    extra_sources: list[dict] | None = None,
) -> Path:
    """Write ``files`` (the authored fixture by default) and their manifest."""
    files = authored_source_files() if files is None else files
    root.mkdir(parents=True)
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = {
        "bundle_schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
        "bundle_id": BUNDLE_ID,
        "bundle_version": BUNDLE_VERSION,
        "created_at": AS_OF,
        "files": {
            relative: {"sha256": hashlib.sha256(content).hexdigest()}
            for relative, content in files.items()
        },
        "external_sources": [
            {
                "source_id": "synthetic-official-snapshot",
                "uri": "urn:fixture:synthetic-official-snapshot",
                "content_hash": hashlib.sha256(files[CSV_PATH]).hexdigest(),
                "qa_status": "fixture",
            },
            *(extra_sources or ()),
        ],
    }
    (root / MANIFEST_PATH).write_text(safe_dump(manifest, sort_keys=False))
    return root


def copy_source_bundle(tmp_path: Path, name: str | None = None) -> Path:
    """Write a fresh synthetic bundle to a new ``tmp_path`` subdirectory."""
    if name is None:
        index = len(list(tmp_path.glob("source_bundle*")))
        name = f"source_bundle_{index}"
    return write_source_bundle(tmp_path / name)
