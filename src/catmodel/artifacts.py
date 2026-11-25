"""Write-once artifacts, published whole or not at all."""

import json
import os
import re
import secrets
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

_TEMPORARY = re.compile(r"\..+\.[0-9a-f]{16}\.tmp")


def frozen_json(payload: object) -> bytes:
    """The bytes a frozen JSON artifact holds for ``payload``: UTF-8 JSON
    with sorted keys, two-space indentation and a final newline. A NaN or an
    infinity is refused."""
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    return text.encode()


def freeze_json(path: Path, payload: object) -> None:
    """Freeze ``payload`` at ``path`` as ``frozen_json`` (``freeze_bytes``)."""
    freeze_bytes(path, frozen_json(payload))


def freeze_bytes(path: Path, data: bytes) -> None:
    """Publish ``data`` at ``path`` once; an existing artifact must hold
    exactly ``data`` and is never replaced.

    The bytes go to a temporary file beside ``path`` (``.<name>.<random>.tmp``),
    which is flushed and fsynced, then linked to ``path``; the link fails if
    ``path`` exists. The directory is fsynced after the link, so a crash
    leaves the artifact whole or absent and no later write can survive
    without it. A crash can leave the temporary file behind; it is never
    evidence, and ``remove_temporary_files`` deletes it.
    """
    try:
        existing = path.read_bytes()
    except FileNotFoundError:
        pass
    else:
        if existing != data:
            raise ValueError(f"frozen artifact differs: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError(f"frozen artifact differs: {path.name}") from None
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def remove_temporary_files(root: Path) -> None:
    """Delete every temporary file an interrupted ``freeze_bytes`` left under
    ``root``."""
    for path in sorted(root.rglob(".*.tmp")):
        if _TEMPORARY.fullmatch(path.name) and path.is_file():
            path.unlink()
            logger.info(
                "removed a temporary file an interrupted write left", path=str(path)
            )
