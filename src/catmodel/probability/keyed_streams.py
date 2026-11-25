"""Named deterministic random streams keyed by seed, identity and purpose."""

import hashlib
import json
import operator
from functools import lru_cache
from typing import SupportsIndex, cast

import numpy as np

# Preserve the recorded stream identity independently of evaluator selection.
EVENT_UNIFORM_METHOD_ID = "philox4x32-10-simulation-event-ordinal-v1"


def _integer_scalar(value: object, *, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field_name} must be an integer, not a boolean")
    try:
        return int(operator.index(cast(SupportsIndex, value)))
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an integer") from exc


def _stream_key(
    *,
    root_seed: int,
    risk_unit_id: str,
    purpose: str,
) -> int:
    payload = json.dumps(
        {
            "contract": EVENT_UNIFORM_METHOD_ID,
            "root_seed": str(root_seed),
            "risk_unit_id": risk_unit_id,
            "purpose": purpose,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _stream_identity_seed(
    *,
    root_seed: object,
    risk_unit_id: str,
    purpose: str,
) -> int:
    normalized_seed = _integer_scalar(root_seed, field_name="root_seed")
    if normalized_seed < 0:
        raise ValueError("root_seed must be nonnegative")
    for field_name, value in (
        ("risk_unit_id", risk_unit_id),
        ("purpose", purpose),
    ):
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")
        if not value.strip():
            raise ValueError(f"{field_name} must not be blank")
    return normalized_seed


def keyed_stream_generator(
    *,
    root_seed: object,
    risk_unit_id: str,
    purpose: str,
) -> np.random.Generator:
    """Return a Philox ``Generator`` keyed by seed, row identity, and purpose.

    The stream is independent of roster membership and position, so paired
    draws are unchanged when rows are added, removed, or reordered.
    """

    normalized_seed = _stream_identity_seed(
        root_seed=root_seed,
        risk_unit_id=risk_unit_id,
        purpose=purpose,
    )
    return np.random.Generator(
        np.random.Philox(
            key=_stream_key(
                root_seed=normalized_seed,
                risk_unit_id=risk_unit_id,
                purpose=purpose,
            )
        )
    )


INCIDENT_STREAM_CONTRACT = "state-origin-purpose-scope-v1"

# Every event's marks are addressed by the event directly, and a campaign
# member's also by its index (``member_uniforms``), rather than allocating a
# Generator per event; a different evaluator needs a new contract string. Origin
# and small-event cell counts read one uniform per year, at event ordinal zero.
VECTOR_EVENT_STREAM_CONTRACT = "philox4x32-10-state-origin-purpose-year-event"


@lru_cache(maxsize=4096)
def _event_key(root_seed: int, state_id: str, origin_id: str, purpose: str):
    normalized = _stream_identity_seed(
        root_seed=root_seed, risk_unit_id=origin_id, purpose=purpose
    )
    payload = json.dumps(
        {
            "contract": VECTOR_EVENT_STREAM_CONTRACT,
            "root_seed": str(normalized),
            "state_id": state_id,
            "origin_id": origin_id,
            "purpose": purpose,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    key = hashlib.sha256(payload).digest()
    return int.from_bytes(key[:4], "little"), int.from_bytes(key[4:8], "little")


def _philox4x32(counter, key):
    """Random123 Philox4x32-10 rounds, elementwise on unsigned counters."""
    mask = np.uint64(0xFFFFFFFF)
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for _ in range(10):
        left = np.uint64(0xD2511F53) * c0
        right = np.uint64(0xCD9E8D57) * c2
        c0, c1, c2, c3 = (
            (right >> 32) ^ c1 ^ np.uint64(k0),
            right & mask,
            (left >> 32) ^ c3 ^ np.uint64(k1),
            left & mask,
        )
        k0 = (k0 + 0x9E3779B9) & 0xFFFFFFFF
        k1 = (k1 + 0xBB67AE85) & 0xFFFFFFFF
    return c0, c1, c2, c3


def event_uniforms(
    years: np.ndarray,
    event_ordinals: np.ndarray,
    *,
    root_seed: int,
    state_id: str,
    origin_id: str,
    purpose: str,
) -> np.ndarray:
    """Return event-addressed ranks, invariant to batch size and row ordering.

    The counter contains the full unsigned 64-bit event ordinal and year.
    Rank-group and numerical-stage identities belong in ``purpose``. Separate
    keys keep segment choices, physical ranks and direct lab amounts independent;
    reusing a physical rank-group key implements the authored quantile coupling.
    """
    normalized_seed = _stream_identity_seed(
        root_seed=root_seed, risk_unit_id=origin_id, purpose=purpose
    )
    if not isinstance(state_id, str) or not state_id.strip():
        raise ValueError("event state_id must be a nonempty string")
    year = np.asarray(years)
    ordinal = np.asarray(event_ordinals)
    if (
        year.shape != ordinal.shape
        or year.dtype.kind not in {"i", "u"}
        or ordinal.dtype.kind not in {"i", "u"}
        or np.any(year < 0)
        or np.any(ordinal < 0)
    ):
        raise ValueError(
            "event years and ordinals must be aligned nonnegative integers"
        )
    year = year.astype(np.uint64, copy=False)
    ordinal = ordinal.astype(np.uint64, copy=False)
    mask = np.uint64(0xFFFFFFFF)
    c0, c1, _, _ = _philox4x32(
        (ordinal & mask, ordinal >> 32, year & mask, year >> 32),
        _event_key(normalized_seed, state_id, origin_id, purpose),
    )
    return ((c0 << 21) | (c1 >> 11)).astype(float) * (1.0 / 9007199254740992.0)


MEMBER_ADDRESS_BITS = 32
"""Bits of a member's address within its event: ``member_uniforms`` reads the
event ordinal shifted left by this many bits, plus the member index."""


def member_uniforms(
    years: np.ndarray,
    event_ordinals: np.ndarray,
    members: np.ndarray,
    *,
    root_seed: int,
    state_id: str,
    origin_id: str,
    purpose: str,
) -> np.ndarray:
    """Return ranks addressed by an event's members, invariant like
    ``event_uniforms``.

    A member's rank is ``event_uniforms`` under the purpose
    ``member::<purpose>`` at the ordinal ``event_ordinal * 2**32 + member``,
    so a member's ranks depend only on its event and its index there, and
    never coincide with an event-level purpose's. Event ordinals and member
    indices must lie below ``2**32``.
    """
    ordinals = np.asarray(event_ordinals)
    indices = np.asarray(members)
    limit = 1 << MEMBER_ADDRESS_BITS
    if (
        ordinals.shape != indices.shape
        or ordinals.dtype.kind not in {"i", "u"}
        or indices.dtype.kind not in {"i", "u"}
        or np.any(ordinals < 0)
        or np.any(indices < 0)
        or np.any(ordinals >= limit)
        or np.any(indices >= limit)
    ):
        raise ValueError(
            "member addresses need aligned event ordinals and member indices in [0, 2**32)"
        )
    combined = (ordinals.astype(np.uint64) << np.uint64(MEMBER_ADDRESS_BITS)) | (
        indices.astype(np.uint64)
    )
    return event_uniforms(
        years,
        combined,
        root_seed=root_seed,
        state_id=state_id,
        origin_id=origin_id,
        purpose="member::" + purpose,
    )


def incident_stream_generator(
    *, root_seed: int, state_id: str, origin_id: str, purpose: str, scope_id: str = ""
) -> np.random.Generator:
    """Address one numerical stage without depending on unrelated origins."""
    identity = json.dumps(
        {
            "contract": INCIDENT_STREAM_CONTRACT,
            "state_id": state_id,
            "origin_id": origin_id,
            "scope_id": scope_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return keyed_stream_generator(
        root_seed=root_seed, risk_unit_id=identity, purpose=purpose
    )
