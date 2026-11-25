"""The laws each alternative puts in place of reference laws.

A setup holds its alternatives' settings directly and their law overrides
here, by alternative. A study with many arms has one changed copy of nearly
every law per arm, so a saved setup stores each arm's overrides in the arm's
own file and reads one arm at a time (``StoredOverrides``); an authored setup
holds its few overrides in memory (``HeldOverrides``). The setup checks each
arm against its own inputs (``checked_by``): held overrides when the setup is
made, stored ones whenever an arm is read.

An alternative's file (``arm_file``) is canonical JSON: the alternative's ID,
its overrides in their inline order, each law with its shared records in the
file's tables (``_shared_records``). It is named ``alternatives/`` and the
SHA-256 of its bytes, so the name ``setup.json`` records also fixes its
content.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from catmodel.contracts._shared_records import (
    SavedTables,
    SavingTables,
    an_object,
    objects,
)
from catmodel.contracts.incident import IncidentLaw, LawOverride
from catmodel.hashing import canonical_json
from catmodel.json_io import require_marker, strict_json_loads
from catmodel.stored_files import DiskFile, StoredFiles

ARM_FILE_SCHEMA_VERSION = 1
ARM_DIRECTORY = "alternatives/"

_OVERRIDE = TypeAdapter(LawOverride)

type ArmCheck = Callable[[str, tuple[LawOverride, ...]], None]
"""A setup's check of one alternative's overrides, by alternative ID, against
its own inputs; it raises ``ValueError`` for overrides the setup refuses."""


def arm_file(alternative_id: str, overrides: Iterable[Mapping[str, Any]]) -> bytes:
    """An alternative's file, from its overrides as the inline encoding holds
    them (``LawOverrides.inline``)."""
    tables = SavingTables(file="an alternative's file")
    rows = [
        {**row, "law": tables.law(an_object(row.get("law"), "override law", "it"))}
        for row in overrides
    ]
    text = canonical_json(
        {
            "schema_version": ARM_FILE_SCHEMA_VERSION,
            "alternative_id": alternative_id,
            "overrides": rows,
            **tables.sorted(),
        }
    )
    return (text + "\n").encode("utf-8")


def arm_file_name(data: bytes) -> str:
    return f"{ARM_DIRECTORY}{hashlib.sha256(data).hexdigest()}.json"


def read_arm_file(data: bytes, alternative_id: str, name: str) -> list[dict[str, Any]]:
    """The overrides an alternative's file holds, as the inline encoding holds
    them. Refuses a file not named by its digest, of another schema version or
    alternative, with a malformed structure, or whose tables fail
    ``SavedTables``' checks."""
    if name != arm_file_name(data):
        raise ValueError(f"{name} is not named by the SHA-256 of its bytes")
    values = an_object(strict_json_loads(data), "content", name)
    values = dict(values)
    require_marker(
        values,
        "schema_version",
        ARM_FILE_SCHEMA_VERSION,
        f"{name} schema version",
        "rebuild the setup with the current software or read it with its "
        "pinned producer software",
    )
    del values["schema_version"]
    if values.pop("alternative_id", None) != alternative_id:
        raise ValueError(f"{name} does not hold the overrides of {alternative_id!r}")
    tables = SavedTables(values, file=name)
    rows = objects(values.pop("overrides", None), "overrides", name)
    if values:
        raise ValueError(f"{name} has unexpected fields: {sorted(values)}")
    inline = [
        {**row, "law": tables.law(an_object(row.get("law"), "override law", name))}
        for row in rows
    ]
    tables.require_all_used()
    return inline


def _validated(rows: list[dict[str, Any]], where: str) -> tuple[LawOverride, ...]:
    """Validate overrides read as plain values, and require that they were
    stored exactly as their validated form is written: the canonical JSON the
    scientific digest reads from the file is then that of the laws a run
    compiles. Values are compared as written, so an integer is not the float
    it validates to."""
    result = []
    for row in rows:
        override = _OVERRIDE.validate_python(row)
        if canonical_json(_OVERRIDE.dump_python(override, mode="json")) != (
            canonical_json(row)
        ):
            raise ValueError(f"{where} holds an override that is not as it is written")
        result.append(override)
    return tuple(result)


class LawOverrides(ABC):
    """Each alternative's law overrides, in the order its inline encoding
    lists them. An alternative without overrides is not listed."""

    @abstractmethod
    def arms(self) -> tuple[str, ...]:
        """The alternatives that have overrides."""

    @abstractmethod
    def inline(self, alternative_id: str) -> list[dict[str, Any]]:
        """The alternative's overrides as plain JSON values, every value in
        place: what the inline encoding and the scientific digest hold. Not
        validated models, so it reads one arm without building them. Each
        call gives values of its own, which the caller may change."""

    @abstractmethod
    def of(self, alternative_id: str) -> tuple[LawOverride, ...]:
        """The alternative's overrides, validated."""

    def keys(self, alternative_id: str) -> tuple[tuple[str, str], ...]:
        """The state and origin of each override, in order."""
        return tuple(
            (row["state_id"], row["law"]["origin_id"])
            for row in self.inline(alternative_id)
        )

    def laws(self, alternative_id: str, state_id: str) -> tuple[IncidentLaw, ...]:
        """The laws the alternative puts in place in one state."""
        return tuple(
            row.law for row in self.of(alternative_id) if row.state_id == state_id
        )

    @abstractmethod
    def checked_by(self, check: ArmCheck) -> LawOverrides:
        """These overrides with ``check`` applied to each arm: at once where
        they are held, and whenever an arm is read where they are stored."""

    @abstractmethod
    def read_every_arm(self) -> None:
        """Read each arm not read so far, so that every arm has been
        validated and checked, and keep none."""

    @abstractmethod
    def release(self) -> None:
        """Let go of any arm kept from an earlier read. A caller that has
        walked the arms calls it so that the last one is not held on."""

    def __iter__(self) -> Iterator[str]:
        return iter(self.arms())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LawOverrides):
            return NotImplemented
        return self.arms() == other.arms() and all(
            self.inline(arm) == other.inline(arm) for arm in self.arms()
        )

    __hash__ = None


class HeldOverrides(LawOverrides):
    """Overrides held in memory, as an authored setup gives them."""

    def __init__(
        self, overrides: Mapping[str, Iterable[LawOverride]] | None = None
    ) -> None:
        held = {
            alternative_id: tuple(rows)
            for alternative_id, rows in (overrides or {}).items()
        }
        for alternative_id, rows in held.items():
            if not isinstance(alternative_id, str) or not all(
                isinstance(row, LawOverride) for row in rows
            ):
                raise TypeError("overrides must be LawOverride models by alternative")
        self._held = {key: rows for key, rows in held.items() if rows}

    @classmethod
    def validated(cls, plain: Mapping[str, Iterable[object]]) -> HeldOverrides:
        """Overrides validated from plain values or models, one at a time, so
        that a constructed or copied model cannot skip validation."""
        return cls(
            {
                alternative_id: tuple(
                    _OVERRIDE.validate_python(
                        _OVERRIDE.dump_python(row, mode="python", warnings=False)
                        if isinstance(row, LawOverride)
                        else row
                    )
                    for row in rows
                )
                for alternative_id, rows in plain.items()
            }
        )

    def arms(self) -> tuple[str, ...]:
        return tuple(self._held)

    def inline(self, alternative_id: str) -> list[dict[str, Any]]:
        return [
            _OVERRIDE.dump_python(row, mode="json")
            for row in self._held.get(alternative_id, ())
        ]

    def of(self, alternative_id: str) -> tuple[LawOverride, ...]:
        return self._held.get(alternative_id, ())

    def checked_by(self, check: ArmCheck) -> HeldOverrides:
        for alternative_id, rows in self._held.items():
            check(alternative_id, rows)
        return self

    def read_every_arm(self) -> None:
        return None  # Held overrides are validated and checked when taken.

    def release(self) -> None:
        return None  # Nothing is read, so nothing is kept from a read.


class StoredOverrides(LawOverrides):
    """Overrides in each alternative's file, read one arm at a time.

    ``names`` gives each alternative's file in ``files``. ``inline`` reads and
    checks the file each time it is asked and keeps nothing; laws that repeat
    a record share it within one result. ``of`` and ``laws`` validate the
    file's overrides, apply the setup's ``check`` to them and keep the last
    arm read until another is read or ``release`` is called, so compiling an
    arm's states in turn reads its file once. Callers that walk every arm
    should finish one before starting the next.

    Nothing is read when the store is made. An arm is therefore validated and
    checked when something first reads it; ``read_every_arm`` reads the rest,
    and loading or building a setup calls it, so every arm of a saved setup
    has passed.
    """

    def __init__(
        self,
        files: Mapping[str, bytes],
        names: Mapping[str, str],
        check: ArmCheck | None = None,
    ) -> None:
        self._files = StoredFiles(files)
        self._names = dict(names)
        missing = sorted(set(self._names.values()) - set(self._files))
        if missing:
            raise ValueError(f"a setup's alternative files are missing: {missing}")
        self._check = check
        self._read: set[str] = set()
        # The last arm validated: its ID, its overrides and its laws by state.
        self._kept: (
            tuple[str, tuple[LawOverride, ...], dict[str, tuple[IncidentLaw, ...]]]
            | None
        ) = None

    def arms(self) -> tuple[str, ...]:
        return tuple(self._names)

    def file_names(self) -> dict[str, str]:
        """Each alternative's file."""
        return dict(self._names)

    def stored(self) -> StoredFiles:
        """The alternatives' files, as held: bytes or files on disk."""
        names = set(self._names.values())
        return self._files.selected(names.__contains__)

    def checked_by(self, check: ArmCheck) -> StoredOverrides:
        return StoredOverrides(self._files, self._names, check)

    def reading(self, files: Mapping[str, bytes]) -> StoredOverrides:
        """The same overrides and check, read from ``files``: the same files,
        held elsewhere. Arms already read are not read again by
        ``read_every_arm``."""
        moved = StoredOverrides(files, self._names, self._check)
        moved._read = set(self._read)
        return moved

    def inline(self, alternative_id: str) -> list[dict[str, Any]]:
        name = self._names.get(alternative_id)
        if name is None:
            return []
        return read_arm_file(self._files[name], alternative_id, name)

    def _arm(
        self, alternative_id: str
    ) -> tuple[tuple[LawOverride, ...], dict[str, tuple[IncidentLaw, ...]]]:
        if self._kept is None or self._kept[0] != alternative_id:
            # Released first, so two validated arms are never kept.
            self._kept = None
            rows = _validated(self.inline(alternative_id), self._names[alternative_id])
            if self._check is not None:
                self._check(alternative_id, rows)
            by_state: dict[str, list[IncidentLaw]] = {}
            for row in rows:
                by_state.setdefault(row.state_id, []).append(row.law)
            self._kept = (
                alternative_id,
                rows,
                {state_id: tuple(laws) for state_id, laws in by_state.items()},
            )
            self._read.add(alternative_id)
        return self._kept[1], self._kept[2]

    def keys(self, alternative_id: str) -> tuple[tuple[str, str], ...]:
        if self._kept is not None and self._kept[0] == alternative_id:
            return tuple((row.state_id, row.law.origin_id) for row in self._kept[1])
        return super().keys(alternative_id)

    def of(self, alternative_id: str) -> tuple[LawOverride, ...]:
        if alternative_id not in self._names:
            return ()
        return self._arm(alternative_id)[0]

    def laws(self, alternative_id: str, state_id: str) -> tuple[IncidentLaw, ...]:
        if alternative_id not in self._names:
            return ()
        return self._arm(alternative_id)[1].get(state_id, ())

    def read_every_arm(self) -> None:
        for alternative_id in self._names:
            if alternative_id not in self._read:
                self._arm(alternative_id)
        self.release()

    def release(self) -> None:
        self._kept = None


class OverrideWriter:
    """Takes each alternative's overrides as an import compiles them.

    With a ``directory``, an arm's file is written there as soon as the arm is
    given and its laws are released, so an import of many arms holds one at a
    time; the result reads them from there (``StoredOverrides``). Without one,
    the overrides are held (``HeldOverrides``).

    The writer validates nothing, because a setup validates every override it
    is given before a run can read it: held ones when the setup is made, and
    stored ones whenever an arm is read, which also requires that the file
    hold each override as its validated form is written. Building or loading
    a setup reads every arm (``store.validate_setup``).
    """

    def __init__(self, directory: Path | None = None) -> None:
        self._directory = directory
        self._held: dict[str, tuple[LawOverride, ...]] = {}
        self._names: dict[str, str] = {}
        self._files: dict[str, DiskFile] = {}

    def add(self, alternative_id: str, overrides: Iterable[LawOverride]) -> None:
        if alternative_id in self._held or alternative_id in self._names:
            raise ValueError(f"alternative {alternative_id!r} is written twice")
        rows = tuple(overrides)
        if any(not isinstance(row, LawOverride) for row in rows):
            raise TypeError("an alternative's overrides must be LawOverride records")
        if not rows:
            return
        if self._directory is None:
            self._held[alternative_id] = rows
            return
        data = arm_file(
            alternative_id, [_OVERRIDE.dump_python(row, mode="json") for row in rows]
        )
        del rows
        name = arm_file_name(data)
        path = self._directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        self._names[alternative_id] = name
        self._files[name] = DiskFile(path, hashlib.sha256(data).hexdigest())

    def overrides(self) -> LawOverrides:
        """Every arm written so far."""
        if self._directory is None:
            return HeldOverrides(self._held)
        return StoredOverrides(StoredFiles(self._files), self._names)
