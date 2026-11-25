"""Fail-closed path containment for files read from local artifact roots."""

from pathlib import Path


def contained_local_path(
    root: Path,
    relative_path: str | Path,
    *,
    artifact_label: str,
) -> Path:
    """Resolve ``relative_path`` and require its real path to remain under ``root``.

    Both paths are resolved to reject traversal and symlink escapes; callers
    must use the returned path.
    """
    resolved_root = root.expanduser().resolve()
    candidate = (resolved_root / relative_path).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(
            f"{artifact_label} path must remain inside its root; "
            f"got {str(relative_path)!r}"
        )
    return candidate
