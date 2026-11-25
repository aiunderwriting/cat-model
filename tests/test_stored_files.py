"""Files held by name stay on disk until read, and a changed file is refused."""

import hashlib
import shutil

import pytest

from catmodel.contracts.incident import ExecutionPlan
from catmodel.contracts.run_setup import EvidenceSet
from catmodel.frozen_calls import CaptureFiles
from catmodel.json_io import strict_json_loads
from catmodel.resources import config_path
from catmodel.setup.store import build_setup, load_setup
from catmodel.stored_files import DiskFile, StoredFiles
from test_support.run_setup import setup_from_json, with_execution


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "tree"
    (root / "calls" / "a").mkdir(parents=True)
    (root / "plan.json").write_bytes(b"plan")
    (root / "calls" / "a" / "request.json").write_bytes(b"request")
    (root / "calls" / "a" / "attempt-1.json").write_bytes(b"attempt")
    return root


def test_a_tree_is_indexed_by_relative_name_without_reading_it(tree):
    files = StoredFiles.under(tree)
    assert list(files) == [
        "calls/a/attempt-1.json",
        "calls/a/request.json",
        "plan.json",
    ]
    assert len(files) == 3
    assert "plan.json" in files and "absent.json" not in files
    assert files["plan.json"] == b"plan"
    assert files.get("absent.json") is None
    assert dict(files.items()) == {
        "calls/a/attempt-1.json": b"attempt",
        "calls/a/request.json": b"request",
        "plan.json": b"plan",
    }
    assert files == dict(files.items())


def test_a_name_is_looked_up_without_reading_its_file(tmp_path):
    files = StoredFiles({"gone.json": DiskFile(tmp_path / "gone.json")})
    assert "gone.json" in files and list(files) == ["gone.json"]
    with pytest.raises(FileNotFoundError):
        files["gone.json"]


def test_bytes_and_files_on_disk_give_the_same_digest(tree):
    files = StoredFiles.merged(StoredFiles.under(tree), {"held.json": b"held"})
    assert files.sha256("plan.json") == _sha(b"plan")
    assert files.sha256("held.json") == _sha(b"held")
    assert files["held.json"] == b"held"


def test_a_file_that_changes_after_its_first_read_is_refused(tree):
    files = StoredFiles.under(tree)
    assert files["plan.json"] == b"plan"
    (tree / "plan.json").write_bytes(b"other")
    with pytest.raises(ValueError, match="differs from its recorded SHA-256"):
        files["plan.json"]
    with pytest.raises(ValueError, match="differs from its recorded SHA-256"):
        files.check()


def test_a_file_must_have_the_digest_it_was_indexed_with(tree):
    files = StoredFiles.under(
        tree, ["plan.json"], {"plan.json": _sha(b"something else")}
    )
    assert files.sha256("plan.json") == _sha(b"something else")
    with pytest.raises(ValueError, match="differs from its recorded SHA-256"):
        files["plan.json"]
    good = StoredFiles.under(tree, ["plan.json"], {"plan.json": _sha(b"plan")})
    good.check()
    assert good["plan.json"] == b"plan"


def test_views_share_one_record_of_each_file(tree):
    """Selecting, renaming and merging never read a file, and a view refuses
    a file another view read before it changed."""
    files = StoredFiles.under(tree)
    calls = files.selected(lambda name: name.startswith("calls/"))
    assert list(calls) == ["calls/a/attempt-1.json", "calls/a/request.json"]
    captured = files.prefixed("captured/")
    assert list(captured.unprefixed("captured/calls/")) == [
        "a/attempt-1.json",
        "a/request.json",
    ]
    merged = StoredFiles.merged(captured, {"captured/plan.json": b"replaced"})
    assert merged["captured/plan.json"] == b"replaced"
    assert merged["captured/calls/a/request.json"] == b"request"
    assert files["calls/a/request.json"] == b"request"
    (tree / "calls" / "a" / "request.json").write_bytes(b"edited")
    with pytest.raises(ValueError, match="differs from its recorded SHA-256"):
        merged["captured/calls/a/request.json"]


def test_writing_copies_a_file_and_checks_the_copy(tree, tmp_path):
    files = StoredFiles.merged(StoredFiles.under(tree), {"held.json": b"held"})
    files.write("plan.json", tmp_path / "plan-copy.json")
    files.write("held.json", tmp_path / "held-copy.json")
    assert (tmp_path / "plan-copy.json").read_bytes() == b"plan"
    assert (tmp_path / "held-copy.json").read_bytes() == b"held"
    with pytest.raises(FileExistsError):
        files.write("plan.json", tmp_path / "plan-copy.json")
    (tree / "plan.json").write_bytes(b"other")
    with pytest.raises(ValueError, match="differs from its recorded SHA-256"):
        files.write("plan.json", tmp_path / "second-copy.json")
    assert not (tmp_path / "second-copy.json").exists()


def test_only_names_with_bytes_or_files_are_accepted():
    with pytest.raises(ValueError, match="string names and bytes or files"):
        StoredFiles({"text.json": "not bytes"})  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="string names and bytes or files"):
        StoredFiles({1: b"bytes"})  # ty: ignore[invalid-argument-type]


def test_a_capture_indexes_its_calls_over_files_on_disk(tree):
    capture = CaptureFiles(StoredFiles.under(tree))
    assert capture.call_records("a") == (
        "calls/a/attempt-1.json",
        "calls/a/request.json",
    )
    assert capture.call_records("b") == ()
    assert capture["calls/a/request.json"] == b"request"


def test_an_evidence_set_leaves_its_files_on_disk(tmp_path):
    held = EvidenceSet(files={"note.txt": b"note"})
    assert held.files == {"note.txt": b"note"}
    absent = EvidenceSet(
        files=StoredFiles({"gone.txt": DiskFile(tmp_path / "gone.txt")})
    )
    assert list(absent.files) == ["gone.txt"]
    with pytest.raises(FileNotFoundError):
        absent.files["gone.txt"]
    with pytest.raises(ValueError, match="string paths and original bytes"):
        EvidenceSet(files={"text.txt": "text"})  # ty: ignore[invalid-argument-type]


SOURCE = config_path("setups", "stipulated-campaign")


def _setup(evidence: StoredFiles):
    """The packaged campaign example, which cites one evidence file."""
    return with_execution(
        setup_from_json((SOURCE / "setup.json").read_bytes()),
        ExecutionPlan.model_validate(
            strict_json_loads((SOURCE / "plan.json").read_bytes())
        ),
        setup_id="stored-files",
        evidence_files=evidence,
    )


@pytest.fixture
def source(tmp_path):
    """A copy of the example's evidence that a test may edit or delete."""
    root = tmp_path / "source"
    shutil.copytree(SOURCE / "evidence", root)
    return root


def test_a_setup_is_built_from_evidence_on_disk_and_then_reads_its_own_copy(
    source, tmp_path
):
    expected = dict(StoredFiles.under(source).items())
    assert expected
    package = build_setup(
        destination=tmp_path / "package", setup=_setup(StoredFiles.under(source))
    )
    # Nothing the package reads is left where its evidence came from.
    shutil.rmtree(source)
    assert dict(package.evidence.files.items()) == expected
    assert {
        name.removeprefix("evidence/")
        for name in package.files
        if name.startswith("evidence/")
    } == set(expected)
    assert package.files.sha256("setup.json") == _sha(
        (tmp_path / "package" / "setup.json").read_bytes()
    )
    loaded = load_setup(tmp_path / "package")
    assert loaded == package
    assert loaded.scientific_sha256 == package.scientific_sha256
    assert loaded.manifest_sha256 == package.manifest_sha256


def test_evidence_that_changes_while_a_setup_is_built_is_refused(source, tmp_path):
    evidence = StoredFiles.under(source)
    setup = _setup(evidence)
    name = next(iter(evidence))
    data = evidence[name]
    (source / name).write_bytes(data + b" edited")
    with pytest.raises(ValueError, match="differs from its recorded SHA-256"):
        build_setup(destination=tmp_path / "package", setup=setup)
    assert not (tmp_path / "package").exists()


def test_a_loaded_setup_refuses_a_file_edited_after_it_was_loaded(source, tmp_path):
    package = build_setup(
        destination=tmp_path / "package", setup=_setup(StoredFiles.under(source))
    )
    loaded = load_setup(package.root)
    name = next(iter(loaded.evidence.files))
    path = package.root / "evidence" / name
    path.write_bytes(path.read_bytes() + b" edited")
    with pytest.raises(ValueError, match="differs from its recorded SHA-256"):
        loaded.evidence.files[name]
    with pytest.raises(ValueError, match="setup checksum mismatch"):
        load_setup(package.root)
