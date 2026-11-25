"""Preflight new outputs without writing into any immutable evidence ancestor."""

from pathlib import Path

_EVIDENCE_MARKERS = (
    "setup-manifest.json",
    "run-capture.json",
    "manifest.yaml",
    "results.json",
)


def require_new_evidence_output(output: Path) -> None:
    """Check every evidence ancestor after resolving parent symlinks."""
    if output.exists() or output.is_symlink():
        raise ValueError(f"evidence output must be new: {output} already exists")
    resolved = output.expanduser().resolve()
    for ancestor in (resolved, *resolved.parents):
        if any((ancestor / marker).exists() for marker in _EVIDENCE_MARKERS):
            raise ValueError(f"output cannot modify immutable evidence: {ancestor}")
