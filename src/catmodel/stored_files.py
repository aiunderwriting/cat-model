"""Files by name, each held as bytes or left on disk and read when asked.

A capture or a saved setup is tens of thousands of files and can run to
gigabytes. Code that copies them, hashes them or reads a few needs none of them
held in memory, so a mapping here can hold a file as its path and the SHA-256
its bytes must have. A read returns the bytes only while they have that digest,
so a file that changes after it was indexed is refused, not silently used.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path


def _sha256_file(path: Path) -> str:
    """The SHA-256 of a file's bytes, read a megabyte at a time."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DiskFile:
    """One file left on disk, with the SHA-256 its bytes must have.

    Without a digest, the first read, copy or ``sha256`` call fixes it.
    """

    __slots__ = ("_sha256", "path")

    def __init__(self, path: Path, sha256: str | None = None) -> None:
        self.path = path
        self._sha256 = sha256

    def _refused(self) -> ValueError:
        return ValueError(f"file differs from its recorded SHA-256: {self.path}")

    def read(self) -> bytes:
        data = self.path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if self._sha256 is None:
            self._sha256 = digest
        elif digest != self._sha256:
            raise self._refused()
        return data

    def sha256(self) -> str:
        """The recorded digest, fixed from the file's bytes if none was given."""
        if self._sha256 is None:
            self._sha256 = _sha256_file(self.path)
        return self._sha256

    def check(self) -> None:
        """Refuse a file whose bytes no longer have the recorded digest,
        without holding them."""
        if _sha256_file(self.path) != self.sha256():
            raise self._refused()

    def copy_to(self, target: Path) -> None:
        """Copy the file to ``target``, a new file, checking the digest of the
        bytes copied."""
        digest = hashlib.sha256()
        with self.path.open("rb") as source, target.open("xb") as sink:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                sink.write(chunk)
        if self._sha256 is None:
            self._sha256 = digest.hexdigest()
        elif digest.hexdigest() != self._sha256:
            target.unlink()
            raise self._refused()


type Stored = bytes | DiskFile


class StoredFiles(Mapping[str, bytes]):
    """A read-only mapping of names to bytes whose larger members stay on disk.

    Looking up a name returns its bytes, reading and checking a ``DiskFile``
    each time, so iterating ``items()`` holds one file at a time. ``sha256``,
    ``selected``, ``prefixed``, ``unprefixed`` and ``merged`` work on names and
    digests without reading a file that has its digest.
    """

    def __init__(
        self, files: Mapping[str, Stored] | Iterable[tuple[str, Stored]] = ()
    ) -> None:
        entries = files._entries if isinstance(files, StoredFiles) else files
        self._entries: dict[str, Stored] = dict(entries)
        for name, entry in self._entries.items():
            if not isinstance(name, str) or not isinstance(entry, bytes | DiskFile):
                raise ValueError("stored files need string names and bytes or files")

    @classmethod
    def under(
        cls,
        root: Path,
        names: Iterable[str] | None = None,
        digests: Mapping[str, str] | None = None,
    ) -> StoredFiles:
        """The files under ``root`` by their POSIX path relative to it, left on
        disk: all of them, or ``names``. ``digests`` gives the SHA-256 each
        must have; a file without one fixes it when first read."""
        if names is None:
            names = sorted(
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
            )
        return cls(
            (name, DiskFile(root / name, None if digests is None else digests[name]))
            for name in names
        )

    def __getitem__(self, name: str) -> bytes:
        entry = self._entries[name]
        return entry.read() if isinstance(entry, DiskFile) else entry

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def sha256(self, name: str) -> str:
        """The file's digest, without holding a file that is on disk."""
        entry = self._entries[name]
        if isinstance(entry, DiskFile):
            return entry.sha256()
        return hashlib.sha256(entry).hexdigest()

    def check(self) -> None:
        """Refuse any file on disk whose bytes no longer have its digest."""
        for entry in self._entries.values():
            if isinstance(entry, DiskFile):
                entry.check()

    def selected(self, keep: Callable[[str], bool]) -> StoredFiles:
        """The files whose name ``keep`` accepts."""
        return StoredFiles(
            (name, entry) for name, entry in self._entries.items() if keep(name)
        )

    def prefixed(self, prefix: str) -> StoredFiles:
        """Every file, named ``prefix`` and then its name."""
        return StoredFiles(
            (prefix + name, entry) for name, entry in self._entries.items()
        )

    def unprefixed(self, prefix: str) -> StoredFiles:
        """The files whose name starts with ``prefix``, named without it."""
        return StoredFiles(
            (name.removeprefix(prefix), entry)
            for name, entry in self._entries.items()
            if name.startswith(prefix)
        )

    @classmethod
    def merged(cls, *parts: Mapping[str, bytes]) -> StoredFiles:
        """Every part's files; a later part's file replaces an earlier one of
        the same name. A part that is not a ``StoredFiles`` gives its bytes."""
        entries: dict[str, Stored] = {}
        for part in parts:
            entries.update(part._entries if isinstance(part, StoredFiles) else part)
        return cls(entries)

    def write(self, name: str, target: Path) -> None:
        """Write one file to ``target``, a new file whose parent exists,
        checking a copy from disk against its digest."""
        entry = self._entries[name]
        if isinstance(entry, DiskFile):
            entry.copy_to(target)
        else:
            with target.open("xb") as sink:
                sink.write(entry)
