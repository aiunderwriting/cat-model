"""Saved files that do not state their current version marker."""

from typing import Any


def unstated_markers(
    payload: dict[str, Any], key: str = "schema_version"
) -> dict[str, dict[str, Any]]:
    """Copies of a saved file's decoded ``payload``, by label, none of which
    states the marker the file was written with: the marker absent, a value
    of another type that compares equal to it or reads like it, and another
    version."""
    current = payload[key]
    if isinstance(current, int):
        others = {
            "true": True,
            "float": float(current),
            "text": str(current),
            "another version": current + 1,
        }
    else:
        others = {"number": 1, "another version": current + "0"}
    rest = {name: value for name, value in payload.items() if name != key}
    return {
        "absent": rest,
        **{label: {**payload, key: value} for label, value in others.items()},
    }
