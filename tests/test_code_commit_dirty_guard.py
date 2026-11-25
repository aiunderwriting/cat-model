import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import pytest

from catmodel import worktree
from catmodel.setup.store import setup_code_provenance
from catmodel.worktree import (
    CodeProvenance,
    resolve_code_provenance,
    working_tree_state,
)
from test_support.study import _package

INSTALLED = CodeProvenance(
    commit=None,
    auto_resolved=True,
    tree_state="installed",
    diff_sha256=None,
    version=version("catmodel"),
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    # Synthetic fixture commits must not depend on the developer's signing agent.
    _git(root, "config", "commit.gpgsign", "false")
    (root / "tracked.txt").write_text("v1\n")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-m", "init")
    return root


def test_explicit_code_commit_is_recorded_as_asserted_not_clean() -> None:
    # An asserted commit is never inspected, so it must not claim a clean tree.
    prov = resolve_code_provenance("  abc123  ")
    assert prov == CodeProvenance(
        commit="abc123",
        auto_resolved=False,
        tree_state="asserted",
        diff_sha256=None,
        version=version("catmodel"),
    )


def test_a_setup_records_the_commit_its_builder_is_given_as_asserted(
    tmp_path: Path,
) -> None:
    # No catmodel command takes a commit; a caller of the Python API gives one.
    code = _package(tmp_path).manifest.code_provenance
    assert (code["commit"], code["tree_state"]) == ("software-fixture", "asserted")


def test_clean_tree_reports_clean(temp_git_repo: Path) -> None:
    state, diff_sha = working_tree_state(temp_git_repo)
    assert state == "clean"
    assert diff_sha is None


def test_tracked_modification_is_dirty_with_stable_digest(temp_git_repo: Path) -> None:
    (temp_git_repo / "tracked.txt").write_text("v2\n")
    state, diff_sha = working_tree_state(temp_git_repo)
    assert state == "dirty"
    assert diff_sha is not None
    assert len(diff_sha) == 64
    assert working_tree_state(temp_git_repo) == ("dirty", diff_sha)
    (temp_git_repo / "tracked.txt").write_text("v3\n")
    _, diff_sha_v3 = working_tree_state(temp_git_repo)
    assert diff_sha_v3 != diff_sha


def test_staged_only_change_is_dirty(temp_git_repo: Path) -> None:
    (temp_git_repo / "tracked.txt").write_text("staged\n")
    _git(temp_git_repo, "add", "tracked.txt")
    state, diff_sha = working_tree_state(temp_git_repo)
    assert state == "dirty"
    assert diff_sha is not None


def test_staged_then_reverted_worktree_is_fingerprinted(temp_git_repo: Path) -> None:
    # A restored worktree still needs the staged index in its digest.
    tracked = temp_git_repo / "tracked.txt"
    tracked.write_text("staged-A\n")
    _git(temp_git_repo, "add", "tracked.txt")
    tracked.write_text("v1\n")
    state_a, digest_a = working_tree_state(temp_git_repo)
    assert state_a == "dirty"
    assert digest_a is not None

    tracked.write_text("staged-B\n")
    _git(temp_git_repo, "add", "tracked.txt")
    tracked.write_text("v1\n")
    state_b, digest_b = working_tree_state(temp_git_repo)
    assert state_b == "dirty"
    assert digest_b is not None
    assert digest_b != digest_a


def test_diff_failure_on_dirty_tree_is_unknown(
    monkeypatch: pytest.MonkeyPatch, temp_git_repo: Path
) -> None:
    # A failed diff cannot produce a complete fingerprint; return "unknown".
    (temp_git_repo / "tracked.txt").write_text("v2\n")
    real_run = subprocess.run

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        cmd = args[0]
        if "diff" in cmd:
            return subprocess.CompletedProcess(cmd, 128, stdout=b"", stderr=b"boom")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(worktree.subprocess, "run", fake_run)
    state, digest = working_tree_state(temp_git_repo)
    assert state == "unknown"
    assert digest is None


def test_untracked_file_is_dirty(temp_git_repo: Path) -> None:
    (temp_git_repo / "new_source.py").write_text("x = 1\n")
    state, diff_sha = working_tree_state(temp_git_repo)
    assert state == "dirty"
    assert diff_sha is not None


def test_untracked_file_detected_despite_showuntrackedfiles_no(
    temp_git_repo: Path,
) -> None:
    _git(temp_git_repo, "config", "status.showUntrackedFiles", "no")
    (temp_git_repo / "sneaky.py").write_text("x = 1\n")
    state, _ = working_tree_state(temp_git_repo)
    assert state == "dirty"


def test_untracked_directory_digest_is_file_granular(temp_git_repo: Path) -> None:
    # File-granular status avoids collapsing an untracked directory.
    nested = temp_git_repo / "pkg"
    nested.mkdir()
    (nested / "a.py").write_text("a = 1\n")
    _, digest_one = working_tree_state(temp_git_repo)
    (nested / "b.py").write_text("b = 2\n")
    _, digest_two = working_tree_state(temp_git_repo)
    assert digest_one is not None
    assert digest_two is not None
    assert digest_one != digest_two


def test_gitignored_untracked_file_is_clean(temp_git_repo: Path) -> None:
    (temp_git_repo / ".gitignore").write_text("build_artifacts/\n")
    _git(temp_git_repo, "add", ".gitignore")
    _git(temp_git_repo, "commit", "-m", "ignore")
    artifacts = temp_git_repo / "build_artifacts"
    artifacts.mkdir()
    (artifacts / "output.bin").write_text("noise")
    state, diff_sha = working_tree_state(temp_git_repo)
    assert state == "clean"
    assert diff_sha is None


def test_non_git_directory_is_unknown(tmp_path: Path) -> None:
    plain = tmp_path / "not_a_repo"
    plain.mkdir()
    state, diff_sha = working_tree_state(plain)
    assert state == "unknown"
    assert diff_sha is None


def test_non_utf8_content_does_not_crash(temp_git_repo: Path) -> None:
    # Hash raw diff bytes so Latin-1 text cannot trigger UnicodeDecodeError.
    (temp_git_repo / "tracked.txt").write_bytes("café\n".encode("latin-1"))
    state, diff_sha = working_tree_state(temp_git_repo)
    assert state == "dirty"
    assert diff_sha is not None
    assert len(diff_sha) == 64


def test_digest_is_independent_of_gitconfig(temp_git_repo: Path) -> None:
    (temp_git_repo / "tracked.txt").write_text("a\nb\nc\nmodified\nd\ne\nf\n")
    _, baseline = working_tree_state(temp_git_repo)
    _git(temp_git_repo, "config", "diff.context", "7")
    _git(temp_git_repo, "config", "color.ui", "always")
    _git(temp_git_repo, "config", "diff.mnemonicPrefix", "true")
    _git(temp_git_repo, "config", "diff.noprefix", "true")
    _git(temp_git_repo, "config", "diff.indentHeuristic", "false")
    _git(temp_git_repo, "config", "diff.algorithm", "histogram")
    _, after = working_tree_state(temp_git_repo)
    assert baseline == after


def test_auto_resolve_uses_source_checkout_not_cwd(
    monkeypatch: pytest.MonkeyPatch, temp_git_repo: Path
) -> None:
    monkeypatch.setattr(worktree, "source_checkout_root", lambda: temp_git_repo)
    prov = resolve_code_provenance(None)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=temp_git_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert prov == CodeProvenance(
        commit=head,
        auto_resolved=True,
        tree_state="clean",
        diff_sha256=None,
        version=version("catmodel"),
    )


def test_installed_code_records_its_version_without_a_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An installed wheel has no source checkout; even run from inside another
    # repository, it records its distribution version, not that repository's
    # commit, and does not fail.
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init")
    monkeypatch.chdir(other)
    monkeypatch.setattr(worktree, "source_checkout_root", lambda: None)
    assert resolve_code_provenance(None) == INSTALLED


def test_vendored_copy_records_its_version_not_the_parent_commit(
    monkeypatch: pytest.MonkeyPatch, temp_git_repo: Path
) -> None:
    # A vendored checkout's parent HEAD does not identify this code.
    vendored = temp_git_repo / "vendor" / "cat-model"
    vendored.mkdir(parents=True)
    monkeypatch.setattr(worktree, "source_checkout_root", lambda: vendored)
    assert resolve_code_provenance(None) == INSTALLED


def test_source_archive_without_git_records_its_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A "Download ZIP" source tree has no .git directory.
    archive = tmp_path / "plain"
    archive.mkdir()
    monkeypatch.setattr(worktree, "source_checkout_root", lambda: archive)
    assert resolve_code_provenance(None) == INSTALLED


def test_missing_git_executable_records_the_version(
    monkeypatch: pytest.MonkeyPatch, temp_git_repo: Path
) -> None:
    def no_git(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("git")

    monkeypatch.setattr(worktree, "source_checkout_root", lambda: temp_git_repo)
    monkeypatch.setattr(worktree.subprocess, "run", no_git)
    assert resolve_code_provenance(None) == INSTALLED


def test_code_without_checkout_or_distribution_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worktree, "source_checkout_root", lambda: None)
    monkeypatch.setattr(worktree, "_installed_version", lambda: None)
    with pytest.raises(ValueError, match="cannot identify the running code"):
        resolve_code_provenance(None)


def test_the_recorded_environment_names_only_what_can_change_a_number():
    """Setups, captures and results carry this record inside their identities,
    so like machines record the same identity and a kernel update or another
    build of the same interpreter changes none."""
    environment = cast(
        dict[str, Any], setup_code_provenance("fixture")["numerical_environment"]
    )
    assert environment["python"] == (
        f"{platform.python_implementation()} {platform.python_version()}"
    )
    assert environment["platform"].startswith(
        f"{platform.system()}-{platform.machine()}"
    )
    recorded = environment["python"] + " " + environment["platform"]
    for excluded in (
        platform.release(),
        platform.python_compiler(),
        platform.python_build()[1],
    ):
        assert excluded not in recorded
    assert set(environment["packages"]) == {"catmodel", "numpy", "scipy", "pydantic"}
