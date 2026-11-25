"""Strict JSON for numerical evidence and immutable execution artifacts."""

import json
import math
import re
from collections.abc import Mapping
from itertools import accumulate
from typing import Any

from catmodel.validation_errors import EvidenceValidationError

MAX_JSON_NESTING = 200
"""The most arrays and objects ``strict_json_loads`` accepts open at once.

The decoder recurses once for each and raises ``RecursionError`` at a depth
that the Python build and the caller's own stack decide: thousands of levels,
several times more under CPython 3.14 than 3.13, and fewer inside an event
loop than outside one. Text nested near that depth would decode in one process
and fail in another, so this limit, far below any interpreter's, decides
instead, alike everywhere. It is far above what any artifact or answer this
package reads nests, which is under twenty levels."""

_TOO_DEEP = f"JSON nests arrays and objects more than {MAX_JSON_NESTING} deep"
_CONTAINERS = frozenset((dict, list))
_OTHER_BYTES = bytes(set(range(256)) - set(b'"[]{}'))
_STRING = re.compile(rb'"[^"]*(?:"|\Z)')
_NESTING_STEP = {ord("["): 1, ord("{"): 1, ord("]"): -1, ord("}"): -1}


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceValidationError(
                "json_keys_must_be_unique",
                f"duplicate JSON key: {key}",
                location=(key,),
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise EvidenceValidationError(
        "json_numbers_must_be_finite",
        f"nonfinite JSON number is not admissible: {value}",
    )


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise EvidenceValidationError(
            "json_numbers_must_be_finite",
            "JSON exponent exceeds the finite numerical range",
        )
    return number


def _value_nests_beyond_limit(value: Any) -> bool:
    """Whether a decoded value holds arrays and objects more than
    ``MAX_JSON_NESTING`` deep."""
    level = [value] if type(value) in _CONTAINERS else []
    for _ in range(MAX_JSON_NESTING):
        if not level:
            return False
        level = [
            child
            for node in level
            for child in (node.values() if type(node) is dict else node)
            if type(child) in _CONTAINERS
        ]
    return bool(level)


def _text_nests_beyond_limit(value: str | bytes | bytearray) -> bool:
    """Whether text opens more than ``MAX_JSON_NESTING`` arrays and objects at
    once, whether or not it is JSON.

    A bracket inside a string opens and closes nothing; a string runs from a
    quote to the next quote that no backslash escapes, or to the end of the
    text.
    """
    if isinstance(value, str):
        text = value.encode("utf-8", "surrogatepass")
    elif (encoding := json.detect_encoding(value)) == "utf-8":
        text = bytes(value)
    else:
        text = value.decode(encoding, "surrogatepass").encode("utf-8", "surrogatepass")
    # Without its escaped backslashes, and then its escaped quotes, every
    # quote left opens or closes a string.
    text = text.replace(b"\\\\", b"").replace(b'\\"', b"")
    brackets = _STRING.sub(b"", text.translate(None, _OTHER_BYTES))
    deepest = max(accumulate(map(_NESTING_STEP.__getitem__, brackets), initial=0))
    return deepest > MAX_JSON_NESTING


def strict_json_loads(value: str | bytes) -> Any:
    """Accept JSON exponent notation; reject duplicate keys, NaN/Infinity and
    nesting deeper than ``MAX_JSON_NESTING``.

    Over-deep text raises a plain ``ValueError`` in every process that reads
    it: never the decoder's ``RecursionError``, and never whatever else is
    wrong with the text, which the decoder may or may not reach first.
    """
    try:
        decoded = json.loads(
            value,
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (RecursionError, ValueError):
        if _text_nests_beyond_limit(value):
            raise ValueError(_TOO_DEEP) from None
        raise
    if _value_nests_beyond_limit(decoded):
        raise ValueError(_TOO_DEEP)
    return decoded


def exact_marker(found: object, expected: int | str) -> bool:
    """Whether ``found`` is the version marker ``expected``: the same value of
    the same type.

    ``True`` and ``1.0`` equal ``1`` and are not it, so a marker a file writes
    as ``true`` or ``1.0`` is another marker.
    """
    return type(found) is type(expected) and found == expected


def require_marker(
    payload: object, key: str, expected: int | str, what: str, remedy: str = ""
) -> None:
    """Refuse the decoded content of a file that does not state the current
    version marker.

    ``payload`` is what a saved file's JSON decoded to. It must be an object
    whose ``key`` is present and is ``expected`` (``exact_marker``); a file
    without the marker is refused, not read as current. ``what`` names the
    marker for the message, which ends with ``remedy`` when one is given.
    """
    if not isinstance(payload, Mapping):
        problem = f"{what} is not stated: the file does not hold a JSON object"
    elif key not in payload:
        problem = f"{what} is not stated ({key} is absent); the current is {expected!r}"
    elif not exact_marker(payload[key], expected):
        problem = f"{what} {payload[key]!r} is not the current {expected!r}"
    else:
        return
    raise ValueError(f"{problem}; {remedy}" if remedy else problem)
