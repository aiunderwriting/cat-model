"""Canonical JSON and SHA-256 kernel for governed identity digests."""

import hashlib
import json
from pathlib import Path


def canonical_json(payload: object) -> str:
    """Serialize a payload using the repository's canonical JSON convention.

    Keys are sorted, separators compact, non-ASCII text preserved, and
    non-finite floats rejected.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_json_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
