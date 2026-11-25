"""The records many laws repeat, stored once in a saved setup's files.

``setup.json`` and each alternative's file hold their laws with every native
severity law, mark kernel, public trajectory and repeated prose field
(``SHARED_RECORDS``) moved to a table keyed by the canonical JSON SHA-256 of its
value; each occurrence holds that key. ``SavingTables`` moves them out as a file
is written and ``SavedTables`` puts them back as it is read, refusing a key that
is not its entry's digest, a reference without an entry and an entry no law
uses.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from catmodel.hashing import canonical_json_sha256

SHARED_RECORDS = ("severity_laws", "mark_kernels", "public_trajectories", "texts")
"""The tables of a saved file, each holding records many laws repeat."""

_LAW_TEXTS = (("qualification_assumption",), ("response", "held_fixed"))
_KERNEL_TEXTS = (
    ("gate_meaning",),
    ("selection_assumption",),
    ("displacement_assumption",),
)
# Every position of a native severity law within a mark kernel; ``*`` walks a
# list. tests/test_setup_storage.py checks them against the contracts.
SEVERITY_PATHS = (
    ("campaign_components", "*", "severity"),
    ("campaign_lab_losses", "*", "severity"),
    ("campaign_payment_kernel", "source_severity"),
    ("explicit_joint_members", "*", "campaign_components", "*", "severity"),
    ("explicit_joint_members", "*", "campaign_payment_kernel", "source_severity"),
    ("explicit_joint_members", "*", "members", "*", "components", "*", "severity"),
    (
        "explicit_joint_members",
        "*",
        "members",
        "*",
        "payment_kernel",
        "source_severity",
    ),
    ("member_outcomes", "*", "components", "*", "severity"),
    ("member_outcomes", "*", "payment_kernel", "source_severity"),
)
_TRAJECTORY = (("conditioning_context", "public_trajectory"),)
_KERNEL = (("segments", "*", "kernel"),)


def _at(value: Any, path: tuple[str, ...], replace: Any, what: str) -> Any:
    """``value`` with ``replace`` applied to each non-null value at ``path``."""
    if value is None:
        return None
    if not path:
        return replace(value)
    head, *rest = path
    if head == "*":
        if not isinstance(value, list):
            raise ValueError(
                f"{what} holds {type(value).__name__} where a list is required"
            )
        return [_at(item, tuple(rest), replace, what) for item in value]
    if not isinstance(value, dict) or head not in value:
        return value
    return {**value, head: _at(value[head], tuple(rest), replace, what)}


def _each(value: Any, paths, replace: Any, what: str) -> Any:
    for path in paths:
        value = _at(value, path, replace, what)
    return value


def objects(value: Any, what: str, file: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{file} {what} must be a list of objects")
    return value


def an_object(value: Any, what: str, file: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{file} {what} must be an object")
    return value


class SavingTables:
    """The tables of a file being written: ``law`` gives a law with each
    shared record replaced by its key, and keeps the record."""

    def __init__(self, file: str = "setup.json") -> None:
        self.file = file
        self.rows: dict[str, dict[str, object]] = {name: {} for name in SHARED_RECORDS}

    def _keep(self, table: str) -> Callable[[object], str]:
        def key(value: object) -> str:
            digest = canonical_json_sha256(value)
            self.rows[table].setdefault(digest, value)
            return digest

        return key

    def _kernel(self, value: dict[str, Any]) -> str:
        value = _each(value, _KERNEL_TEXTS, self._keep("texts"), self.file)
        return self._keep("mark_kernels")(
            _each(value, SEVERITY_PATHS, self._keep("severity_laws"), self.file)
        )

    def law(self, value: dict[str, Any]) -> dict[str, Any]:
        value = _each(value, _LAW_TEXTS, self._keep("texts"), self.file)
        value = _each(value, _TRAJECTORY, self._keep("public_trajectories"), self.file)
        return _each(value, _KERNEL, self._kernel, self.file)

    def sorted(self) -> dict[str, dict[str, object]]:
        """Every table, its rows in key order."""
        return {name: dict(sorted(rows.items())) for name, rows in self.rows.items()}


class SavedTables:
    """The tables of a file being read, taken out of its ``values``: ``law``
    gives a law with each key replaced by its record."""

    def __init__(self, values: dict[str, Any], file: str = "setup.json") -> None:
        self.file = file
        self.rows: dict[str, dict[str, Any]] = {}
        for name in SHARED_RECORDS:
            rows = values.pop(name, None)
            if not isinstance(rows, dict):
                raise ValueError(f"{file} lacks its {name} table")
            if any(canonical_json_sha256(value) != key for key, value in rows.items()):
                raise ValueError(f"{file} {name} keys differ from their digests")
            self.rows[name] = rows
        self.used: dict[str, set[str]] = {name: set() for name in SHARED_RECORDS}
        self._kernels: dict[str, Any] = {}

    def _take(self, table: str) -> Callable[[object], Any]:
        def entry(key: object) -> Any:
            if not isinstance(key, str) or key not in self.rows[table]:
                raise ValueError(
                    f"{self.file} refers to no {table} entry: {str(key)[:80]!r}"
                )
            self.used[table].add(key)
            return self.rows[table][key]

        return entry

    def _kernel(self, key: object) -> Any:
        value = self._take("mark_kernels")(key)
        assert isinstance(key, str)
        if key not in self._kernels:
            value = _each(value, _KERNEL_TEXTS, self._take("texts"), self.file)
            self._kernels[key] = _each(
                value, SEVERITY_PATHS, self._take("severity_laws"), self.file
            )
        return self._kernels[key]

    def law(self, value: dict[str, Any]) -> dict[str, Any]:
        value = _each(value, _LAW_TEXTS, self._take("texts"), self.file)
        value = _each(value, _TRAJECTORY, self._take("public_trajectories"), self.file)
        return _each(value, _KERNEL, self._kernel, self.file)

    def require_all_used(self) -> None:
        if unused := {
            name: len(set(self.rows[name]) - self.used[name])
            for name in SHARED_RECORDS
            if set(self.rows[name]) - self.used[name]
        }:
            raise ValueError(f"{self.file} table entries no law uses: {unused}")
