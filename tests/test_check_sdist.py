"""The source-distribution check compares an archive with the tracked tree."""

import io
import subprocess
import tarfile
from pathlib import Path

from scripts.check_sdist import (
    content_errors,
    export_ignored_files,
    link_errors,
    sdist_errors,
    tracked_files,
)

_FILES = {
    "README.md": "[guide](docs/guide.md#setup) [data](data/table.csv) [top](#readme)\n\n# Readme\n",
    "docs/guide.md": "# Guide\n\n## Setup\n\n[back](../README.md) [tests](../tests/)\n",
    "data/table.csv": "a,b\n",
    "tests/test_table.py": "",
}


def _archive(tmp_path: Path, files: dict[str, str]) -> Path:
    tree = tmp_path / "tree" / "catmodel-1.2.3"
    for name, text in {**files, "PKG-INFO": "Metadata-Version: 2.4\n"}.items():
        path = tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    archive = tmp_path / "catmodel-1.2.3.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(tree, arcname=tree.name)
    return archive


def test_an_archive_of_the_tracked_tree_with_resolving_links_passes(tmp_path):
    assert sdist_errors(_archive(tmp_path, _FILES), set(_FILES)) == []


def test_an_export_ignored_file_is_the_one_tracked_file_left_out(tmp_path):
    tracked = {*_FILES, "NOTES.md"}
    archive = _archive(tmp_path, _FILES)
    assert sdist_errors(archive, tracked, {"NOTES.md"}) == []
    # Unmarked, the same file is missing from the archive.
    (missing,) = sdist_errors(archive, tracked)
    assert missing.startswith("NOTES.md:")
    assert "not in the source distribution" in missing

    shipped = {**_FILES, "NOTES.md": "# Notes\n"}
    (error,) = sdist_errors(_archive(tmp_path, shipped), tracked, {"NOTES.md"})
    assert error.startswith("NOTES.md:")
    assert "export-ignore" in error


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True
    ).stdout


def test_export_ignored_files_are_those_git_archive_leaves_out(tmp_path):
    """The left-out files are read from .gitattributes as Git reads it: a
    file that has the attribute, at any depth for a pattern without a slash,
    and every file under a directory that has it."""
    files = {
        ".gitattributes": "NOTES.md export-ignore\nscratch/ export-ignore\n*.draft export-ignore\n",
        "README.md": "# Readme\n",
        "NOTES.md": "# Notes\n",
        "docs/NOTES.md": "# Notes\n",
        "docs/guide.md": "# Guide\n",
        "docs/plan.draft": "plan\n",
        "scratch/a.txt": "a\n",
        "scratch/deep/b.txt": "b\n",
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(tmp_path, "init")
    _git(tmp_path, "add", "--all")
    # A fixture commit must not depend on the developer's identity or signing.
    _git(
        tmp_path,
        *("-c", "user.name=Test", "-c", "user.email=test@example.com"),
        *("-c", "commit.gpgsign=false", "commit", "--message", "fixture"),
    )
    tracked = tracked_files(tmp_path)
    assert tracked == set(files)

    ignored = export_ignored_files(tmp_path, tracked)

    assert ignored == {
        "NOTES.md",
        "docs/NOTES.md",
        "docs/plan.draft",
        "scratch/a.txt",
        "scratch/deep/b.txt",
    }
    with tarfile.open(fileobj=io.BytesIO(_git(tmp_path, "archive", "HEAD"))) as tar:
        exported = {member.name for member in tar.getmembers() if member.isfile()}
    assert ignored == tracked - exported
    assert export_ignored_files(tmp_path, set()) == set()


def test_a_file_git_does_not_track_is_reported(tmp_path):
    shipped = {**_FILES, "data/external/download.csv": "restricted\n"}
    (error,) = sdist_errors(_archive(tmp_path, shipped), set(_FILES))
    assert error.startswith("data/external/download.csv:")
    assert "not tracked" in error


def test_a_tracked_file_the_archive_leaves_out_is_reported(tmp_path):
    tracked = {*_FILES, ".github/workflows/ci.yml", "data/cards.yaml"}
    errors = sdist_errors(_archive(tmp_path, _FILES), tracked)
    assert [error.partition(":")[0] for error in errors] == [
        ".github/workflows/ci.yml",
        "data/cards.yaml",
    ]


def test_the_generated_metadata_file_is_the_only_untracked_name_allowed():
    assert content_errors({"PKG-INFO", "a.py"}, {"a.py"}) == []
    assert len(content_errors({"PKG-INFO", "a.py", "b.py"}, {"a.py"})) == 1


def test_broken_links_are_reported_in_every_page_of_the_archive(tmp_path):
    files = {
        **_FILES,
        "README.md": "[gone](data/missing.yaml) [guide](docs/guide.md#no-such-heading)\n",
        "config/example/README.md": "[plan](plan.json) [setup](setup.json)\n",
        "config/example/setup.json": "{}\n",
    }
    errors = sdist_errors(_archive(tmp_path, files), set(files))
    assert [error.partition(":")[0] for error in errors] == [
        "README.md",
        "README.md",
        "config/example/README.md",
    ]
    assert "data/missing.yaml" in errors[0]
    assert "no-such-heading" in errors[1]
    assert "plan.json" in errors[2]


def test_a_link_to_a_file_outside_the_tree_is_broken(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (tmp_path / "beside.md").write_text("# Beside\n", encoding="utf-8")
    (root / "inside.md").write_text("# Inside\n", encoding="utf-8")
    page = root / "page.md"
    page.write_text("[in](inside.md) [out](../beside.md)\n", encoding="utf-8")

    (error,) = link_errors(page, root)
    assert "../beside.md" in error


def test_links_in_code_and_to_other_sites_are_not_followed(tmp_path):
    files = {
        **_FILES,
        "README.md": (
            "`[example](not-a-file.md)` [site](https://example.org/page.md)\n\n"
            "```md\n[fenced](also-not-a-file.md)\n```\n"
        ),
    }
    assert sdist_errors(_archive(tmp_path, files), set(files)) == []
