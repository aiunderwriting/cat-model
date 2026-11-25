"""Resolve the running code's identity: its Git commit and working-tree state,
or the installed distribution's version."""

import hashlib
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

import structlog

from catmodel.resources import source_checkout_root

CodeTreeState = Literal["clean", "dirty", "unknown", "installed", "asserted"]

logger = structlog.get_logger(__name__)


def working_tree_state(root: Path) -> tuple[CodeTreeState, str | None]:
    """Return ``(state, diff_sha256)`` for the Git working tree at ``root``.

    ``state`` is ``"unknown"`` when ``git status`` or ``git diff`` fails. A dirty
    tree's digest hashes the raw bytes of the porcelain status and of the
    working-tree and staged diffs, with flags that override the repository's
    Git config. It identifies untracked files by path only and may differ
    between Git versions, so it is an audit aid, not a complete content hash.
    """

    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if status.returncode != 0:
        logger.warning(
            "code working-tree status check failed; cleanliness unknown",
            stderr=status.stderr.decode("utf-8", "replace").strip(),
        )
        return "unknown", None
    porcelain = status.stdout
    if not porcelain.strip():
        return "clean", None

    diff_base = [
        "git",
        "-c",
        "color.ui=false",
        "-c",
        "core.autocrlf=false",
        "diff",
        "--no-color",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--no-indent-heuristic",
        "--full-index",
        "--unified=3",
        "--diff-algorithm=myers",
        "--src-prefix=a/",
        "--dst-prefix=b/",
    ]
    # Hash both: ``git diff HEAD`` misses staged content when the working file
    # has been restored to HEAD.
    worktree_diff = subprocess.run(
        [*diff_base, "HEAD", "--"], cwd=root, check=False, capture_output=True
    )
    staged_diff = subprocess.run(
        [*diff_base, "--cached", "HEAD", "--"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if worktree_diff.returncode != 0 or staged_diff.returncode != 0:
        logger.warning(
            "code working-tree diff failed; cleanliness unknown",
            worktree_diff_returncode=worktree_diff.returncode,
            staged_diff_returncode=staged_diff.returncode,
        )
        return "unknown", None
    payload = (
        porcelain
        + b"\n\x00\n"
        + worktree_diff.stdout
        + b"\n\x00\n"
        + staged_diff.stdout
    )
    return "dirty", hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CodeProvenance:
    """The running code's identity, as a setup, run or reduction records it.

    ``tree_state`` says how ``commit`` was established. ``clean``, ``dirty``
    and ``unknown`` describe a source checkout that is the root of its own
    Git repository, whose HEAD is ``commit``; a dirty tree also records its
    diff digest. ``installed`` is code outside such a checkout, such as an
    installed wheel or a source archive without ``.git``: no commit
    identifies it, so ``commit`` is None and ``version`` alone identifies it.
    ``asserted`` is a commit a caller of the Python API gives as
    ``code_commit``, recorded without inspection; no ``catmodel`` command
    asserts one. ``version`` is the installed ``catmodel`` distribution's
    version, None without distribution metadata.
    """

    commit: str | None
    auto_resolved: bool
    tree_state: "CodeTreeState"
    diff_sha256: str | None
    version: str | None = None


def _installed_version() -> str | None:
    try:
        return version("catmodel")
    except PackageNotFoundError:
        return None


def _checkout_commit(root: Path) -> str | None:
    """HEAD of the Git repository whose top level is ``root``, else None.

    None when Git is missing, ``root`` holds no repository, its HEAD has no
    commit, or ``root`` is vendored inside another repository, whose commit
    does not identify this code.
    """
    try:
        toplevel = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if toplevel.returncode != 0 or not toplevel.stdout.strip():
        return None
    if Path(toplevel.stdout.strip()).resolve() != root.resolve():
        return None
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = head.stdout.strip()
    return commit if head.returncode == 0 and commit else None


def resolve_code_provenance(code_commit: str | None) -> CodeProvenance:
    """Identify the running code (``CodeProvenance`` describes the states).

    A nonblank ``code_commit`` is recorded as ``asserted``. Otherwise the
    source checkout holding this module, never the working directory, gives
    the commit and tree state, so installed code cannot record an unrelated
    repository's commit; code outside such a checkout is ``installed``.
    """
    installed = _installed_version()
    if code_commit is not None and code_commit.strip():
        return CodeProvenance(
            commit=code_commit.strip(),
            auto_resolved=False,
            tree_state="asserted",
            diff_sha256=None,
            version=installed,
        )
    root = source_checkout_root()
    commit = None if root is None else _checkout_commit(root)
    if root is None or commit is None:
        if installed is None:
            raise ValueError(
                "cannot identify the running code: it is not a Git checkout and "
                "has no installed catmodel distribution; install it (uv sync), or "
                "give its commit as code_commit through the Python API"
            )
        return CodeProvenance(
            commit=None,
            auto_resolved=True,
            tree_state="installed",
            diff_sha256=None,
            version=installed,
        )
    state, diff_sha = working_tree_state(root)
    return CodeProvenance(
        commit=commit,
        auto_resolved=True,
        tree_state=state,
        diff_sha256=diff_sha,
        version=installed,
    )
