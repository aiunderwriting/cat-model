"""Encode and decode a setup's scientific inputs, inline and as ``setup.json``.

The inline encoding is one flat JSON object whose keys are the setup's fields
(``scope_id``, ``joint_states`` and so on) with every value written in place:
the layout of authored inputs (``setup build --spec``). ``INLINE_FIELDS`` maps
each key to the ``RunSetup`` field it holds, or to a field of the scope or
assumption section; encoding and decoding both use it. Import fails unless it
holds every ``RunSetup`` field except the setup ID and execution plan exactly
once, and decoding rejects unknown or missing required keys.
``AssumptionSet.independence`` is held as ``independence_assumptions`` and
``EvidenceSet.records`` as ``evidence``. An empty ``conditional_contexts`` is
omitted, so a setup without conditional contexts has no such key. The scientific
digest is the canonical sorted-key JSON digest of the inline encoding without
its disclosure prose (``disclosure_prose``): the scope's ``title``, the
``limitations`` and each alternative's ``description``, which the setup keeps
to report and no law reads. Rewording them, key order, formatting and the
saved layout do not affect it.

A saved setup (schema ``SCHEMA_VERSION``) is ``setup.json`` and one file for
each alternative that overrides laws (``saved_setup``). ``setup.json`` is the
inline encoding with two changes. The records many laws repeat are stored
once: every native severity law, mark kernel, public trajectory and repeated
prose field of a law (``SHARED_RECORDS``) sits in its table keyed by the
canonical JSON SHA-256 of its value, and each occurrence holds that key. And
each alternative's ``overrides`` holds the name of the alternative's file
(``law_overrides.arm_file``), or null, instead of the overrides, so that a
setup with many arms is read one arm at a time. The files are written as
canonical JSON (sorted keys, no whitespace), so object key order is not
preserved and carries no meaning. Decoding refuses any other schema version, a
malformed structure, a key that is not its entry's digest, a reference without
an entry, an entry no law uses and a shared position that holds a value
instead of a key.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import fields
from types import MappingProxyType
from typing import Any

from pydantic import TypeAdapter

from catmodel.contracts._shared_records import (
    SEVERITY_PATHS,
    SHARED_RECORDS,
    SavedTables,
    SavingTables,
    an_object,
    objects,
)
from catmodel.contracts.incident import Alternative, Evidence, ExecutionPlan
from catmodel.contracts.law_overrides import (
    HeldOverrides,
    LawOverrides,
    StoredOverrides,
    arm_file,
    arm_file_name,
)
from catmodel.contracts.run_setup import (
    AssumptionSet,
    EvidenceSet,
    RunScope,
    RunSetup,
    element_adapters,
    field_adapters,
)
from catmodel.hashing import canonical_json
from catmodel.json_io import require_marker
from catmodel.stored_files import StoredFiles

__all__ = [
    "INLINE_FIELDS",
    "SCHEMA_VERSION",
    "SEVERITY_PATHS",
    "SHARED_RECORDS",
    "decode_saved_setup",
    "decode_scientific",
    "disclosure_prose",
    "encode_scientific",
    "saved_setup",
    "scientific_sha256",
]

SCHEMA_VERSION = 1
_RECORDS = TypeAdapter(tuple[Evidence, ...])
_SECTIONS = {"scope": RunScope, "assumptions": AssumptionSet}
# Not keys of the inline encoding: the package's own identity and plan, and the
# overrides, which the inline encoding lists inside their alternative.
_NOT_INLINE = frozenset({"setup_id", "execution", "overrides"})
# Inline key order: (RunSetup field, field of a section or None for the field itself).
_LAYOUT: tuple[tuple[str, str | None], ...] = (
    *(("scope", name) for name in RunScope.model_fields),
    ("evidence", None),
    ("contracts", None),
    ("joint_states", None),
    ("conditional_contexts", None),
    ("alternatives", None),
    *(("assumptions", name) for name in AssumptionSet.model_fields),
    ("limitations", None),
)
_INLINE_NAMES: dict[tuple[str, str | None], str] = {
    ("assumptions", "independence"): "independence_assumptions"
}
_REQUIRED = frozenset({"joint_states", "independence_assumptions", "limitations"})
_OMITTED_WHEN_EMPTY = frozenset({"conditional_contexts"})
# Disclosure prose, which the scientific digest leaves out: inline keys, and
# fields of the models a RunSetup tuple-of-models field holds, by that field.
_PROSE_KEYS = frozenset({"title", "limitations"})
_PROSE_FIELDS: Mapping[str, set[str]] = MappingProxyType(
    {"alternatives": {"description"}}
)


def _inline_fields() -> Mapping[str, tuple[str, str | None]]:
    held = {row.name for row in fields(RunSetup) if row.init} - _NOT_INLINE
    if {attribute for attribute, _ in _LAYOUT} != held:
        raise RuntimeError("the inline encoding must hold every RunSetup input field")
    table: dict[str, tuple[str, str | None]] = {}
    for attribute, name in _LAYOUT:
        key = _INLINE_NAMES.get((attribute, name), name or attribute)
        if key in table or key == "schema_version":
            raise RuntimeError(f"inline setup key {key!r} is not unique")
        table[key] = (attribute, name)
    return MappingProxyType(table)


INLINE_FIELDS = _inline_fields()
"""Each key of the inline (authored) encoding and the setup field it holds."""
if not _PROSE_KEYS.issubset(INLINE_FIELDS) or not set(_PROSE_FIELDS).issubset(
    element_adapters()
):
    raise RuntimeError("disclosure prose must name inline keys and model fields")


def _alternative(row: Alternative, overrides: object) -> dict[str, Any]:
    """An alternative as an encoding holds it: its fields, with ``overrides``
    after its description, where the inline encoding lists its overrides."""
    result: dict[str, Any] = {}
    for key, value in (
        element_adapters()["alternatives"].dump_python(row, mode="json").items()
    ):
        result[key] = value
        if key == "description":
            result["overrides"] = overrides
    return result


def _encode(setup: RunSetup, overrides) -> dict[str, Any]:
    """The inline encoding with ``overrides(alternative)`` as each
    alternative's ``overrides``."""
    sections = {
        attribute: getattr(setup, attribute).model_dump(mode="json", warnings=False)
        for attribute in _SECTIONS
    }
    adapters = field_adapters()
    values: dict[str, Any] = {}
    for key, (attribute, name) in INLINE_FIELDS.items():
        if name is not None:
            value = sections[attribute][name]
        elif attribute == "evidence":
            value = _RECORDS.dump_python(setup.evidence.records, mode="json")
        elif attribute == "alternatives":
            value = [_alternative(row, overrides(row)) for row in setup.alternatives]
        else:
            raw = getattr(setup, attribute)
            value = adapters[attribute].dump_python(
                dict(raw) if isinstance(raw, Mapping) else raw, mode="json"
            )
        if key in _OMITTED_WHEN_EMPTY and not value:
            continue
        values[key] = value
    return values


def encode_scientific(setup: RunSetup) -> dict[str, Any]:
    """The setup's inline encoding, as JSON values: every key, every value in
    place, each alternative's overrides included. It holds every arm at once;
    ``scientific_sha256`` and ``saved_setup`` read a stored setup arm by arm."""
    return _encode(setup, lambda row: setup.overrides.inline(row.alternative_id))


def disclosure_prose(setup: RunSetup) -> tuple[object, ...]:
    """The text the scientific digest leaves out: the scope's title, the
    limitations and each alternative's description."""
    return (
        setup.scope.title,
        setup.limitations,
        tuple((row.alternative_id, row.description) for row in setup.alternatives),
    )


def _digest_alternative(
    digest,
    row: Alternative,
    overrides: LawOverrides,
    arm_files: Mapping[str, str] | None = None,
) -> None:
    """Feed ``digest`` the canonical JSON of one alternative as the inline
    encoding holds it, less its description, one override at a time; with
    ``arm_files``, the name of the alternative's file in place of its
    overrides."""
    fields = element_adapters()["alternatives"].dump_python(
        row, mode="json", exclude=_PROSE_FIELDS["alternatives"]
    )
    digest.update(b"{")
    for index, key in enumerate(sorted({*fields, "overrides"})):
        if index:
            digest.update(b",")
        digest.update(canonical_json(key).encode("utf-8") + b":")
        if key != "overrides":
            digest.update(canonical_json(fields[key]).encode("utf-8"))
            continue
        if arm_files is not None:
            name = arm_files.get(row.alternative_id)
            digest.update(canonical_json(name).encode("utf-8"))
            continue
        digest.update(b"[")
        for position, override in enumerate(overrides.inline(row.alternative_id)):
            if position:
                digest.update(b",")
            digest.update(canonical_json(override).encode("utf-8"))
        digest.update(b"]")
    digest.update(b"}")


def scientific_sha256(setup: RunSetup) -> str:
    """The canonical JSON digest of ``encode_scientific(setup)`` without its
    disclosure prose (``disclosure_prose``), hashed one field at a time; in a
    tuple of models, one model at a time; and in an alternative, one override
    at a time, so a large setup's inline encoding is never held whole and a
    stored setup is read one arm at a time."""
    return _scientific_sha256(setup, None)


def stored_inputs_sha256(setup: RunSetup) -> str:
    """The scientific digest of a setup whose overrides are stored, with each
    alternative's overrides entered as the name of its file.

    A file's name is the SHA-256 of its bytes, so two setups with this digest
    in common hold the same arm files and have the scientific digest in
    common too. Setups that store the same arm files, as a composed setup
    and its saved form do, differ in it exactly when they differ in the
    scientific digest, and it reads no arm. It is for that comparison
    (``store.build_setup``); it is not a setup's identity.
    """
    if not isinstance(setup.overrides, StoredOverrides):
        raise ValueError("only a setup with stored overrides has arm files to name")
    return _scientific_sha256(setup, setup.overrides.file_names())


def _scientific_sha256(setup: RunSetup, arm_files: Mapping[str, str] | None) -> str:
    digest = hashlib.sha256()
    adapters, elements = field_adapters(), element_adapters()
    sections: dict[str, dict[str, Any]] = {}
    written = 0
    digest.update(b"{")
    for key in sorted(set(INLINE_FIELDS) - _PROSE_KEYS):
        attribute, name = INLINE_FIELDS[key]
        raw = getattr(setup, attribute)
        if key in _OMITTED_WHEN_EMPTY and not raw:
            continue
        if written:
            digest.update(b",")
        written += 1
        digest.update(canonical_json(key).encode("utf-8") + b":")
        if name is not None:
            if attribute not in sections:
                sections[attribute] = raw.model_dump(mode="json", warnings=False)
            digest.update(canonical_json(sections[attribute][name]).encode("utf-8"))
        elif attribute == "evidence":
            value = _RECORDS.dump_python(setup.evidence.records, mode="json")
            digest.update(canonical_json(value).encode("utf-8"))
        elif attribute in elements:
            digest.update(b"[")
            for index, row in enumerate(raw):
                if index:
                    digest.update(b",")
                if attribute == "alternatives":
                    _digest_alternative(digest, row, setup.overrides, arm_files)
                    continue
                value = elements[attribute].dump_python(
                    row, mode="json", exclude=_PROSE_FIELDS.get(attribute)
                )
                digest.update(canonical_json(value).encode("utf-8"))
            digest.update(b"]")
        else:
            value = adapters[attribute].dump_python(
                dict(raw) if isinstance(raw, Mapping) else raw, mode="json"
            )
            digest.update(canonical_json(value).encode("utf-8"))
    digest.update(b"}")
    return digest.hexdigest()


def _without_overrides(
    alternatives: object,
) -> tuple[object, dict[str, Any]]:
    """Alternatives as the setup holds them, and what each listed as its
    ``overrides``, by alternative. A value that is not a list of objects is
    returned as given, for validation to refuse."""
    if not isinstance(alternatives, list) or not all(
        isinstance(row, dict) for row in alternatives
    ):
        return alternatives, {}
    rows, listed = [], {}
    for row in alternatives:
        row = dict(row)
        if "overrides" in row:
            listed[str(row.get("alternative_id"))] = row.pop("overrides")
        rows.append(row)
    return rows, listed


def decode_scientific(
    payload: object,
    execution: ExecutionPlan,
    *,
    setup_id: str,
    evidence_files: Mapping[str, bytes] | None = None,
) -> RunSetup:
    """Decode the inline encoding (``encode_scientific``)."""
    if not isinstance(payload, dict):
        raise ValueError("scientific setup payload must be an object")
    values = dict(payload)
    if "schema_version" in values or set(SHARED_RECORDS) & set(values):
        raise ValueError(
            f"this is a saved setup.json (schema {values.get('schema_version')!r}); "
            "authored inputs (setup build --spec) take the inline layout, every "
            "value written in place, with no schema version or shared tables"
        )
    if "alternatives" in values:
        values["alternatives"], listed = _without_overrides(values["alternatives"])
    else:
        listed = {}
    if any(not isinstance(rows, list) for rows in listed.values()):
        raise ValueError("an alternative's overrides must be a list of objects")
    return _decode(
        values,
        execution,
        setup_id=setup_id,
        evidence_files=evidence_files,
        overrides=HeldOverrides.validated(listed),
    )


def _decode(
    values: dict[str, Any],
    execution: ExecutionPlan,
    *,
    setup_id: str,
    evidence_files: Mapping[str, bytes] | None,
    overrides: LawOverrides,
) -> RunSetup:
    """A setup from inline values whose alternatives list no overrides."""
    if unexpected := set(values) - set(INLINE_FIELDS):
        raise ValueError(f"unexpected scientific setup fields: {sorted(unexpected)}")
    if missing := _REQUIRED - set(values):
        raise ValueError(f"scientific setup requires fields: {sorted(missing)}")
    sections: dict[str, dict[str, object]] = {name: {} for name in _SECTIONS}
    inputs: dict[str, Any] = {}
    for key, value in values.items():
        attribute, name = INLINE_FIELDS[key]
        if name is not None:
            sections[attribute][name] = value
        elif attribute != "evidence":
            # Held as decoded: RunSetup validates each input field on construction.
            inputs[attribute] = value
    return RunSetup(
        setup_id=setup_id,
        scope=RunScope.model_validate(sections["scope"]),
        evidence=EvidenceSet(
            records=values.get("evidence", ()), files=evidence_files or {}
        ),
        assumptions=AssumptionSet.model_validate(sections["assumptions"]),
        execution=execution,
        overrides=overrides,
        **inputs,
    )


def _state_laws(payload: dict[str, Any], replace: Any) -> dict[str, Any]:
    """``payload`` with ``replace`` applied to every law of a joint state."""
    result = dict(payload)
    result["joint_states"] = [
        {
            **state,
            "laws": [
                replace(an_object(law, "law", "setup.json"))
                for law in objects(state.get("laws"), "joint state laws", "setup.json")
            ],
        }
        for state in objects(payload.get("joint_states"), "joint states", "setup.json")
    ]
    return result


def saved_setup(setup: RunSetup) -> tuple[str, StoredFiles]:
    """The setup as it is saved: ``setup.json``'s text, and the file of each
    alternative that overrides laws, by name.

    ``setup.json`` is the inline encoding with every shared record stored once
    in its table and each alternative's overrides replaced by the name of its
    file, as canonical JSON. A setup that already stores its overrides keeps
    its files; one that holds them writes each arm's in turn.
    """
    overrides = setup.overrides
    if isinstance(overrides, StoredOverrides):
        names, arms = overrides.file_names(), overrides.stored()
    else:
        names, written = {}, {}
        for alternative_id in overrides.arms():
            data = arm_file(alternative_id, overrides.inline(alternative_id))
            names[alternative_id] = arm_file_name(data)
            written[names[alternative_id]] = data
        arms = StoredFiles(written)
    tables = SavingTables()
    saved = _state_laws(
        _encode(setup, lambda row: names.get(row.alternative_id)), tables.law
    )
    text = canonical_json(
        {"schema_version": SCHEMA_VERSION, **saved, **tables.sorted()}
    )
    return text, arms


def decode_saved_setup(
    payload: object,
    execution: ExecutionPlan,
    *,
    setup_id: str,
    evidence_files: Mapping[str, bytes] | None = None,
    files: Mapping[str, bytes] | None = None,
) -> RunSetup:
    """Decode a saved ``setup.json`` (``saved_setup``); ``files`` holds the
    alternatives' files it names, which are read when an arm is."""
    if not isinstance(payload, dict):
        raise ValueError("setup.json must be an object")
    values = dict(payload)
    require_marker(
        values,
        "schema_version",
        SCHEMA_VERSION,
        "setup.json schema version",
        "rebuild the setup with the current software or read it with its "
        "pinned producer software",
    )
    del values["schema_version"]
    tables = SavedTables(values)
    inline = _state_laws(values, tables.law)
    tables.require_all_used()
    names: dict[str, str] = {}
    if "alternatives" in inline:
        inline["alternatives"], listed = _without_overrides(
            objects(inline["alternatives"], "alternatives", "setup.json")
        )
        for alternative_id, name in listed.items():
            if name is None:
                continue
            if not isinstance(name, str):
                raise ValueError(
                    "setup.json alternative overrides must name the alternative's file"
                )
            names[alternative_id] = name
    return _decode(
        inline,
        execution,
        setup_id=setup_id,
        evidence_files=evidence_files,
        overrides=StoredOverrides(files or {}, names),
    )
