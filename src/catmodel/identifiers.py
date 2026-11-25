"""Injective grammar for atomic model identifiers."""

from typing import Annotated

from pydantic import BeforeValidator


def validate_atomic_identifier(value: object) -> str:
    """Require a trimmed nonblank atom and reserve ``:`` for composition."""

    if not isinstance(value, str):
        raise TypeError("atomic identifiers must be strings")
    if value != value.strip():
        raise ValueError("atomic identifiers must not contain surrounding whitespace")
    if not value:
        raise ValueError("atomic identifiers must be nonblank")
    if ":" in value:
        raise ValueError("atomic identifiers must not contain ':'")
    return value


type AtomicIdentifier = Annotated[str, BeforeValidator(validate_atomic_identifier)]
