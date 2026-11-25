"""Frozen artifacts are published whole, once, and never replaced."""

import errno
import os
import stat

import pytest

from catmodel.artifacts import (
    freeze_bytes,
    freeze_json,
    frozen_json,
    remove_temporary_files,
)


def _names(directory):
    return sorted(path.name for path in directory.iterdir())


def test_an_artifact_is_published_once_and_never_replaced(tmp_path):
    path = tmp_path / "attempt-001.json"
    freeze_json(path, {"b": 1, "a": [2]})
    assert path.read_bytes() == b'{\n  "a": [\n    2\n  ],\n  "b": 1\n}\n'
    inode = path.stat().st_ino
    freeze_json(path, {"a": [2], "b": 1})
    with pytest.raises(ValueError, match=r"frozen artifact differs: attempt-001\.json"):
        freeze_json(path, {"a": [3], "b": 1})
    assert path.stat().st_ino == inode
    assert path.read_bytes() == b'{\n  "a": [\n    2\n  ],\n  "b": 1\n}\n'
    assert _names(tmp_path) == ["attempt-001.json"]
    # The process umask sets its mode, as for any file the process creates.
    reference = tmp_path / "reference"
    reference.write_bytes(b"")
    assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)


def test_frozen_json_is_the_bytes_a_json_artifact_holds(tmp_path):
    payload = {"b": -0.0, "a": ["\u00e9", 10**30, 5e-324], "\U0001f600": None}
    expected = (
        b'{\n  "a": [\n    "\\u00e9",\n    1000000000000000000000000000000,\n'
        b'    5e-324\n  ],\n  "b": -0.0,\n  "\\ud83d\\ude00": null\n}\n'
    )
    assert frozen_json(payload) == expected
    freeze_json(tmp_path / "artifact.json", payload)
    assert (tmp_path / "artifact.json").read_bytes() == expected
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            frozen_json({"a": value})
        with pytest.raises(ValueError):
            freeze_json(tmp_path / "refused.json", {"a": value})
    assert _names(tmp_path) == ["artifact.json"]


def test_an_interrupted_publish_leaves_neither_the_artifact_nor_its_temporary_file(
    tmp_path, monkeypatch
):
    def fail(_descriptor):
        raise OSError(errno.EIO, "device failed")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="device failed"):
        freeze_bytes(tmp_path / "request.json", b"{}\n")
    assert _names(tmp_path) == []


@pytest.mark.parametrize(("winner", "refused"), [(b"same", False), (b"other", True)])
def test_a_writer_that_loses_the_race_holds_the_winner_to_its_bytes(
    tmp_path, monkeypatch, winner, refused
):
    """Another process publishes between the existence check and the link."""
    link = os.link

    def racing(source, target):
        (tmp_path / "plan.json").write_bytes(winner)
        link(source, target)

    monkeypatch.setattr(os, "link", racing)
    if refused:
        with pytest.raises(ValueError, match="frozen artifact differs"):
            freeze_bytes(tmp_path / "plan.json", b"same")
    else:
        freeze_bytes(tmp_path / "plan.json", b"same")
    assert (tmp_path / "plan.json").read_bytes() == winner
    assert _names(tmp_path) == ["plan.json"]


def test_only_temporary_files_are_removed(tmp_path):
    calls = tmp_path / "calls" / "key"
    calls.mkdir(parents=True)
    temporary = [
        calls / ".attempt-002.json.0123456789abcdef.tmp",
        tmp_path / ".contextual-risk.json.fedcba9876543210.tmp",
    ]
    kept = [
        calls / "attempt-001.json",
        calls / "attempt-001.json.0123456789abcdef.tmp",
        tmp_path / ".notes.tmp",
        tmp_path / ".plan.json.0123456789ABCDEF.tmp",
    ]
    for path in (*temporary, *kept):
        path.write_bytes(b'{"truncat')
    remove_temporary_files(tmp_path)
    assert [path.exists() for path in temporary] == [False, False]
    assert all(path.exists() for path in kept)
