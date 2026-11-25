"""Dependency-light primitive contracts shared across persisted model surfaces."""

from typing import Annotated, NoReturn

from pydantic import AfterValidator, BeforeValidator


def normalize_required_string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("value must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError("value must not be blank")
    return normalized


def normalize_optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("value must be a string or None")
    return value.strip() or None


def validate_sha256_digest(value: object) -> str:
    normalized = normalize_required_string(value)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("value must be a lowercase 64-character SHA-256 hex digest")
    return normalized


def validate_unique_strings(values: list[str]) -> list[str]:
    if len(set(values)) != len(values):
        raise ValueError("values must be unique")
    return values


type RequiredString = Annotated[str, BeforeValidator(normalize_required_string)]
type OptionalString = Annotated[str | None, BeforeValidator(normalize_optional_string)]
type UniqueRequiredStringList = Annotated[
    list[RequiredString], AfterValidator(validate_unique_strings)
]

# Identical to RequiredString; the alias marks identifier fields.
type RequiredIdentifier = RequiredString
type Sha256Digest = Annotated[str, BeforeValidator(validate_sha256_digest)]


class FrozenDict(dict):
    """A dict that refuses every change after construction.

    It is still a dict: pydantic and ``json`` serialize it as one, and it equals
    a dict with equal items. ``dict(value)`` or ``{**value}`` give a mutable
    copy. ``yaml.safe_dump`` cannot represent it.
    """

    __slots__ = ()

    def _refuse(self, *args: object, **kwargs: object) -> NoReturn:
        raise TypeError("a frozen mapping cannot be changed")

    __setitem__ = __delitem__ = __ior__ = _refuse
    clear = pop = popitem = setdefault = update = _refuse

    def __reduce__(self):
        return (type(self), (dict(self),))


def frozen(value: object) -> object:
    """``value`` with every dict and list in it, at any depth, replaced by a
    ``FrozenDict`` or tuple; scalars are returned unchanged.

    Its JSON is unchanged, but a value that held a list no longer equals its
    source (a tuple never equals a list), and Python-mode dumps carry tuples.
    """
    if isinstance(value, dict):
        return FrozenDict({key: frozen(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(frozen(item) for item in value)
    return value


# A mapping field of a frozen model: validation freezes the mapping and every
# dict or list within it, so no field of the model can change after validation.
# Give such a field ``default_factory=FrozenDict``; defaults are not validated.
type Frozen[T] = Annotated[T, AfterValidator(frozen)]
