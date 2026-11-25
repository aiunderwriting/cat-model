"""JSON Schema (2020-12) documents of ``results.json`` and its diagnostics file.

``results_schemas`` generates both documents from the TypedDicts of
``catmodel.contracts.results``: ``results.json``'s from ``HeadlineResults``
and the diagnostics file's from ``ResultsExhibits`` and ``ResultsDiagnostics``.
A TypedDict is a closed object, a ``dict`` a map with typed values, a
``Literal`` an enumeration and a type alias a definition. Each file stores
every reason as a code (``intern_reasons``), so a
string the reason rule selects is a ``ReasonCode`` in the documents, and each
file's ``reasons`` table maps those codes to their texts. ``schema_version`` is
pinned to ``RESULTS_SCHEMA_VERSION``.

The documents are committed under ``contracts/results/``;
``scripts/generate_results_schemas.py`` writes them again after a results
contract changes. ``schema_errors`` checks a file's content against its
committed document.
"""

import copy
import inspect
import json
import re
import types
import typing
from collections.abc import Iterator, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NotRequired,
    Required,
    TypeAliasType,
    cast,
)

from catmodel.contracts.results import (
    REASON_CODE_LENGTH,
    RESULTS_SCHEMA_VERSION,
    HeadlineResults,
    ResultsDiagnostics,
    ResultsExhibits,
    holds_reasons,
    is_reason,
)
from catmodel.resources import resource_path

if TYPE_CHECKING:
    from jsonschema import ValidationError
    from jsonschema.protocols import Validator

DIALECT = "https://json-schema.org/draft/2020-12/schema"
RESULTS_SCHEMA_FILE = "results.schema.json"
DIAGNOSTICS_SCHEMA_FILE = "results.diagnostics.schema.json"
REGENERATE_COMMAND = "uv run python scripts/generate_results_schemas.py"

_REASON_CODE = "ReasonCode"
_REASON_CODE_SCHEMA = {
    "description": (
        f"A reason's code: the first {REASON_CODE_LENGTH} hexadecimal digits of "
        "the SHA-256 of its UTF-8 text. A reason is a string under a key named "
        "`reason` or ending in `_reason`, or any string in the value of a key "
        "ending in `_reasons` or containing `_reasons_`; the file stores each as "
        "its code, and its `reasons` table gives the text."
    ),
    "type": "string",
    "pattern": f"^[0-9a-f]{{{REASON_CODE_LENGTH}}}$",
}
_REASONS_SCHEMA = {
    "description": (
        "Each reason code in the file and its text. Every code has an entry, "
        "every entry is used, and each entry's code is its text's code."
    ),
    "type": "object",
    "propertyNames": {"$ref": f"#/$defs/{_REASON_CODE}"},
    "additionalProperties": {"type": "string"},
}
_SCALARS: Mapping[object, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    types.NoneType: {"type": "null"},
    object: {},
}
_JSON_TYPES = {
    "string": "a string",
    "integer": "an integer",
    "number": "a number",
    "boolean": "a boolean",
    "null": "null",
    "object": "an object",
    "array": "an array",
}
_PLAIN_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class _Generator:
    """Translates annotations into schemas, collecting each TypedDict and type
    alias once under ``defs``."""

    def __init__(self) -> None:
        self.defs: dict[str, dict[str, Any]] = {
            _REASON_CODE: copy.deepcopy(_REASON_CODE_SCHEMA)
        }
        self._sources: dict[str, object] = {}

    def schema(self, annotation: object, *, coded: bool = False) -> dict[str, Any]:
        """The schema of ``annotation``; with ``coded``, the value of a key that
        holds reasons, whose every string is a reason code."""
        if coded and annotation is str:
            return self._reason_code()
        if annotation in _SCALARS:
            return dict(_SCALARS[annotation])
        if annotation is Any:
            raise TypeError("results contracts declare every value's type, not Any")
        if isinstance(annotation, TypeAliasType):
            return self._alias(annotation, coded)
        if typing.is_typeddict(annotation):
            return self._typed_dict(annotation, coded)
        origin, arguments = typing.get_origin(annotation), typing.get_args(annotation)
        if origin is Literal:
            if coded and all(isinstance(value, str) for value in arguments):
                return self._reason_code()
            return (
                {"const": arguments[0]}
                if len(arguments) == 1
                else {"enum": list(arguments)}
            )
        if origin in (typing.Union, types.UnionType):
            return _union([self.schema(item, coded=coded) for item in arguments])
        if origin in (list, Sequence):
            (item,) = arguments
            return {"type": "array", "items": self.schema(item, coded=coded)}
        if origin is tuple:
            if len(arguments) == 2 and arguments[1] is Ellipsis:
                return {
                    "type": "array",
                    "items": self.schema(arguments[0], coded=coded),
                }
            return {
                "type": "array",
                "prefixItems": [self.schema(item, coded=coded) for item in arguments],
                "items": False,
                "minItems": len(arguments),
            }
        if origin in (dict, Mapping):
            key, value = arguments
            schema: dict[str, Any] = {
                "type": "object",
                "additionalProperties": self.schema(value, coded=coded),
            }
            if key is not str:
                schema["propertyNames"] = self.schema(key)
            return schema
        raise TypeError(f"no JSON Schema for the results annotation {annotation!r}")

    def _reason_code(self) -> dict[str, Any]:
        return {"$ref": f"#/$defs/{_REASON_CODE}"}

    def _define(self, name: str, source: object, build) -> dict[str, Any]:
        if self._sources.setdefault(name, source) is not source:
            raise TypeError(f"two results contracts are named {name}")
        if name not in self.defs:
            # Registered before it is built, so a recursive reference ends.
            self.defs[name] = {}
            self.defs[name] = build()
        return {"$ref": f"#/$defs/{name}"}

    def _alias(self, alias: TypeAliasType, coded: bool) -> dict[str, Any]:
        if coded:
            return self.schema(alias.__value__, coded=True)
        return self._define(alias.__name__, alias, lambda: self.schema(alias.__value__))

    def _typed_dict(self, cls: Any, coded: bool) -> dict[str, Any]:
        if coded:
            return self.object_schema(cls, coded=True)
        return self._define(cls.__name__, cls, lambda: self.object_schema(cls))

    def object_schema(self, cls: Any, *, coded: bool = False) -> dict[str, Any]:
        """A TypedDict's closed object schema; a reason key's string value is a
        reason code, and so is every string in a key that holds reasons."""
        schema: dict[str, Any] = {}
        if cls.__doc__:
            # Docstrings mark code with double backquotes; Markdown with one.
            schema["description"] = inspect.cleandoc(cls.__doc__).replace("``", "`")
        properties = {
            key: self.property_schema(key, annotation, coded=coded)
            for key, annotation in typing.get_type_hints(
                cls, include_extras=True
            ).items()
        }
        required = [key for key in properties if key in cls.__required_keys__]
        schema.update(type="object", properties=properties)
        if required:
            schema["required"] = required
        schema["additionalProperties"] = False
        return schema

    def property_schema(
        self, key: str, annotation: object, *, coded: bool = False
    ) -> dict[str, Any]:
        """The schema of one field, as ``intern_reasons`` stores it."""
        if typing.get_origin(annotation) in (NotRequired, Required):
            (annotation,) = typing.get_args(annotation)
        if coded or holds_reasons(key):
            return self.schema(annotation, coded=True)
        if is_reason(key):
            return self._reason_value(annotation)
        return self.schema(annotation)

    def _reason_value(self, annotation: object) -> dict[str, Any]:
        """A reason key's value: a string is a reason code; anything else is
        stored as it is."""
        if isinstance(annotation, TypeAliasType):
            return self._reason_value(annotation.__value__)
        if typing.get_origin(annotation) in (typing.Union, types.UnionType):
            return _union(
                [self._reason_value(item) for item in typing.get_args(annotation)]
            )
        if annotation is str or typing.get_origin(annotation) is Literal:
            return self.schema(annotation, coded=True)
        return self.schema(annotation)


def _union(options: list[dict[str, Any]]) -> dict[str, Any]:
    unique = [
        option for index, option in enumerate(options) if option not in options[:index]
    ]
    if all(set(option) == {"type"} for option in unique):
        kinds = [option["type"] for option in unique]
        return {"type": kinds[0] if len(kinds) == 1 else kinds}
    return unique[0] if len(unique) == 1 else {"anyOf": unique}


def _references(node: object) -> list[str]:
    """The definitions ``node`` refers to directly or in its parts."""
    if isinstance(node, dict):
        found = [node["$ref"].removeprefix("#/$defs/")] if "$ref" in node else []
        return found + [name for value in node.values() for name in _references(value)]
    if isinstance(node, list):
        return [name for value in node for name in _references(value)]
    return []


def _document(
    generator: _Generator,
    *,
    title: str,
    description: str,
    properties: Mapping[str, dict[str, Any]],
    required: Sequence[str],
) -> dict[str, Any]:
    """A schema document of ``properties`` with the definitions they reach."""
    reached: set[str] = set()
    pending = _references(dict(properties))
    while pending:
        name = pending.pop()
        if name not in reached:
            reached.add(name)
            pending.extend(_references(generator.defs[name]))
    return {
        "$schema": DIALECT,
        "title": title,
        **({"description": description} if description else {}),
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
        "$defs": {name: generator.defs[name] for name in sorted(reached)},
    }


def _version_schema() -> dict[str, Any]:
    return {
        "description": (
            "The layout of results.json and of its diagnostics file; readers "
            "accept only the current version."
        ),
        "const": RESULTS_SCHEMA_VERSION,
    }


def results_schemas() -> dict[str, dict[str, Any]]:
    """The JSON Schema documents of ``results.json`` and of its diagnostics
    file, by file name."""
    generator = _Generator()
    headline = generator.object_schema(HeadlineResults)
    properties = {
        key: _version_schema() if key == "schema_version" else value
        for key, value in headline["properties"].items()
    }
    properties["reasons"] = copy.deepcopy(_REASONS_SCHEMA)
    results = _document(
        generator,
        title="results.json",
        description=(
            "The headline of a native results report "
            "(catmodel.contracts.results.HeadlineResults): what the result "
            "describes, its identity, coverage and limitations, each arm's "
            "statistics, the comparisons and each scenario's contribution. The "
            "diagnostics file beside it holds the report's exhibits and view "
            "diagnostics. Every reason is stored as a code that the `reasons` "
            "table resolves."
        ),
        properties=properties,
        required=[*headline["required"], "reasons"],
    )
    generator = _Generator()
    exhibits = generator.object_schema(ResultsExhibits)
    arms = generator.object_schema(ResultsDiagnostics)
    shared = set(exhibits["properties"]) & set(arms["properties"])
    if shared:
        raise TypeError(f"an exhibit and the view diagnostics are both named {shared}")
    diagnostics = _document(
        generator,
        title="results.diagnostics.json",
        description=(
            "The rest of the report whose headline is in the results file "
            "beside it, tied to it by their analysis identity: the exhibits "
            "(catmodel.contracts.results.ResultsExhibits) and, under `arms`, "
            "each arm's view diagnostics "
            "(catmodel.contracts.results.ResultsDiagnostics). Every reason is "
            "stored as a code that the `reasons` table resolves."
        ),
        properties={
            "schema_version": _version_schema(),
            "analysis_sha256": {
                "description": "The analysis_sha256 of the results file beside it.",
                "type": "string",
            },
            **exhibits["properties"],
            **arms["properties"],
            "reasons": copy.deepcopy(_REASONS_SCHEMA),
        },
        required=[
            "schema_version",
            "analysis_sha256",
            *exhibits["required"],
            *arms["required"],
            "reasons",
        ],
    )
    return {RESULTS_SCHEMA_FILE: results, DIAGNOSTICS_SCHEMA_FILE: diagnostics}


def schema_path(name: str) -> Path:
    """The committed document ``name`` (``RESULTS_SCHEMA_FILE`` or
    ``DIAGNOSTICS_SCHEMA_FILE``), in the source checkout or among the
    installed package's resources."""
    return resource_path("contracts", "results", name)


def schema_text(schema: Mapping[str, Any]) -> str:
    """A schema document as committed: indented JSON with a final newline."""
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def write_results_schemas(directory: Path) -> list[Path]:
    """Write both documents into ``directory``, replacing earlier ones."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, schema in results_schemas().items():
        path = directory / name
        path.write_text(schema_text(schema), encoding="utf-8")
        paths.append(path)
    return paths


@cache
def _validator(name: str) -> "Validator":
    # Imported here so that only reading a results file loads jsonschema.
    from jsonschema import Draft202012Validator

    return Draft202012Validator(json.loads(schema_path(name).read_bytes()))


def schema_errors(payload: object, name: str) -> list[str]:
    """Each way ``payload`` departs from the committed document ``name``
    (``RESULTS_SCHEMA_FILE`` or ``DIAGNOSTICS_SCHEMA_FILE``), as the path of
    the value in the payload and what is wrong with it; empty when the
    payload conforms.

    A path joins keys with dots and puts an index, or a key that is not a
    plain name, in brackets, as jq does:
    ``arms.reference.metrics.gross.tail_metrics["0.95"].var``.
    """
    rows = [
        row
        for error in _validator(name).iter_errors(payload)
        for departure in _departures(error)
        for row in _rows(departure)
    ]
    return list(dict.fromkeys(rows))


def _departures(error: "ValidationError") -> "Iterator[ValidationError]":
    """The errors to report for ``error``. A value that matches no member of
    a union fails each of them, so its errors are those of the member it
    comes closest to: of the members of the value's own JSON type, the one it
    fails in the fewest places. A value of no member's type is the error."""
    if not error.context:
        yield error
        return
    members: dict[object, list[ValidationError]] = {}
    for nested in error.context:
        members.setdefault(nested.relative_schema_path[0], []).append(nested)
    typed = [errors for errors in members.values() if not any(map(_wrong_type, errors))]
    if not typed:
        yield error
        return
    for nested in min(typed, key=len):
        yield from _departures(nested)


def _wrong_type(error: "ValidationError") -> bool:
    """Whether a union member's ``error`` is that the value itself is not of
    the member's JSON type."""
    return error.validator == "type" and not error.relative_path


def _rows(error: "ValidationError") -> Iterator[str]:
    """``error`` in terms of the payload: each missing or unknown field by
    its own path, and any other value by its path, what it is and what the
    schema asks for."""
    path, value = list(error.absolute_path), error.instance
    expected: Any = error.validator_value
    # The object schema that holds the keyword, never a boolean schema.
    schema = cast(Mapping[str, Any], error.schema)
    match error.validator:
        case "required":
            for key in expected:
                if key not in value:
                    yield f"{_path([*path, key])} is missing"
        case "additionalProperties" if expected is False:
            for key in value:
                if key not in schema.get("properties", {}):
                    yield f"{_path([*path, key])} is not a field of the schema"
        case "type":
            yield _mismatch(error, f"not {_types([expected])}")
        case "anyOf":
            # A value of none of the union's types (``_departures``).
            wanted = [
                nested.validator_value
                for nested in error.context or ()
                if _wrong_type(nested)
            ]
            yield _mismatch(error, f"not {_types(wanted)}")
        case "const":
            yield _mismatch(error, f"not {_show(expected)}")
        case "enum":
            yield _mismatch(error, f"not one of {', '.join(map(_show, expected))}")
        case "pattern":
            yield _mismatch(error, f"which does not match {expected}")
        case "minItems":
            yield f"{_path(path)} is an array of length {len(value)}, not {expected}"
        case "items" if expected is False:
            items = len(schema["prefixItems"])
            yield f"{_path(path)} is an array of length {len(value)}, not {items}"
        case _:
            yield f"{_path(path)}: {error.message[:200]}"


def _mismatch(error: "ValidationError", expected: str) -> str:
    """A value, or a key of a map whose keys are typed, that is not what the
    schema asks for."""
    path = _path(list(error.absolute_path))
    if "propertyNames" in error.absolute_schema_path:
        return f"{path} has the key {_show(error.instance)}, {expected}"
    return f"{path} is {_show(error.instance)}, {expected}"


def _types(wanted: Sequence[str | Sequence[str]]) -> str:
    """The JSON types of ``type`` keywords' values, in words."""
    kinds = [
        kind for item in wanted for kind in ([item] if isinstance(item, str) else item)
    ]
    return " or ".join(_JSON_TYPES[kind] for kind in dict.fromkeys(kinds))


def _path(parts: Sequence[str | int]) -> str:
    path = ""
    for part in parts:
        if isinstance(part, str) and _PLAIN_KEY.fullmatch(part):
            path += f".{part}" if path else part
        else:
            path += f"[{json.dumps(part, ensure_ascii=False)}]"
    return path or "the document"


def _show(value: object) -> str:
    if isinstance(value, dict):
        return "an object"
    if isinstance(value, list):
        return "an array"
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 60 else f"{text[:59]}…"
