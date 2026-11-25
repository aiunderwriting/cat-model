"""Enforce exact-version pins for first-party dependencies.

Requirements in the enforced pyproject.toml sections must use exact versions:
the project's dependencies and optional dependencies, the dependency groups
and the build-system requirements. Direct references and bare requirements
backed by tag or full-SHA uv git sources are exempt; other bare or ranged
requirements fail. Exact pins prevent silent upstream changes to model outputs
and security posture.

``[tool.uv] constraint-dependencies`` and ``[tool.uv.exclude-newer-package]``
are not checked. Transitive security floors in ``constraint-dependencies`` may
use ``>=`` because uv.lock fixes resolved versions and exact floors can block
later fixes.

With no arguments every pyproject.toml found in the repository is checked. A
manifest named on the command line must exist, and a run that checks no
manifest fails.
"""

import argparse
import re
import sys
import tomllib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Exclude VCS, environments, caches, and build/vendor output explicitly so dotted
# directories such as ``.github`` remain searchable.
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        ".tox",
        ".nox",
        ".eggs",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".cache",
        "build",
        "dist",
        "site-packages",
        "__pycache__",
    }
)


def _discover_manifests(root: Path = _REPO_ROOT) -> list[Path]:
    manifests: list[Path] = []
    for path in sorted(root.rglob("pyproject.toml")):
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        manifests.append(path)
    return manifests


@dataclass(frozen=True, slots=True)
class Violation:
    manifest: Path
    section: str
    requirement: str
    reason: str


def _iter_requirements(data: dict[str, Any]) -> Iterator[tuple[str, str]]:
    build = data.get("build-system")
    if isinstance(build, dict):
        requires = build.get("requires")
        if isinstance(requires, list):
            for entry in requires:
                if isinstance(entry, str):
                    yield "build-system.requires", entry

    project = data.get("project")
    if isinstance(project, dict):
        deps = project.get("dependencies")
        if isinstance(deps, list):
            for entry in deps:
                if isinstance(entry, str):
                    yield "project.dependencies", entry

        optional = project.get("optional-dependencies")
        if isinstance(optional, dict):
            for extra, entries in optional.items():
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, str):
                            yield f"project.optional-dependencies.{extra}", entry

    groups = data.get("dependency-groups")
    if isinstance(groups, dict):
        for group, entries in groups.items():
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, str):
                        yield f"dependency-groups.{group}", entry


def _pin_problem(req: Requirement) -> str | None:
    if req.url is not None:
        return None  # Direct references are accepted as pins.

    specifiers = list(req.specifier)
    if not specifiers:
        return "no version specifier (must be an exact '==' pin)"
    if len(specifiers) > 1:
        return f"multiple version specifiers '{req.specifier}' (use a single '==' pin)"

    specifier = specifiers[0]
    if specifier.operator != "==":
        return f"non-exact operator in '{req.specifier}' (use '==')"
    if "*" in specifier.version:
        return f"wildcard version in '{req.specifier}' (use a fully-resolved '==' pin)"
    return None


#: Only a full 40-hex commit SHA is immutable; branches and abbreviated SHAs are not.
_FULL_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


def _git_ref_pinned_sources(data: dict[str, Any]) -> set[str]:
    """Return names backed by tag or full-SHA git sources accepted as pins.

    Branches, abbreviated SHAs, and non-git sources are not accepted.
    """
    tool = data.get("tool")
    uv = tool.get("uv") if isinstance(tool, dict) else None
    sources = uv.get("sources") if isinstance(uv, dict) else None
    if not isinstance(sources, dict):
        return set()
    pinned: set[str] = set()
    for name, spec in sources.items():
        if not (isinstance(spec, dict) and "git" in spec):
            continue
        rev = spec.get("rev")
        if spec.get("tag") or (
            isinstance(rev, str) and _FULL_COMMIT_SHA.fullmatch(rev)
        ):
            pinned.add(canonicalize_name(name))
    return pinned


def _check_manifest(manifest: Path) -> list[Violation]:
    data = tomllib.loads(manifest.read_text())
    git_ref_pinned = _git_ref_pinned_sources(data)
    violations: list[Violation] = []
    for section, requirement in _iter_requirements(data):
        try:
            req = Requirement(requirement)
        except InvalidRequirement as exc:
            violations.append(
                Violation(
                    manifest=manifest,
                    section=section,
                    requirement=requirement,
                    reason=f"could not parse requirement ({exc})",
                )
            )
            continue
        reason = _pin_problem(req)
        if reason is None:
            continue
        # Only a bare requirement may rely on its pinned git source.
        if not list(req.specifier) and canonicalize_name(req.name) in git_ref_pinned:
            continue
        violations.append(
            Violation(
                manifest=manifest,
                section=section,
                requirement=requirement,
                reason=reason,
            )
        )
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifests",
        nargs="*",
        type=Path,
        help="pyproject.toml files to check (default: all discovered in the repo)",
    )
    args = parser.parse_args(argv)

    manifests: list[Path] = args.manifests or _discover_manifests()
    missing = [manifest for manifest in manifests if not manifest.is_file()]
    for manifest in missing:
        print(f"error: {manifest} is not a file", file=sys.stderr)
    if not manifests:
        print("error: no pyproject.toml found to check", file=sys.stderr)
    if missing or not manifests:
        return 1

    violations: list[Violation] = []
    for manifest in manifests:
        violations.extend(_check_manifest(manifest))

    if violations:
        print("Dependency pin check found non-exact first-party pins:", file=sys.stderr)
        for v in violations:
            print(
                f"  {v.manifest} [{v.section}] '{v.requirement}': {v.reason}",
                file=sys.stderr,
            )
        print(
            "\nFirst-party dependencies must use exact '==' pins. "
            "See scripts/check_dependency_pins.py for the policy and the "
            "constraint-dependencies exception.",
            file=sys.stderr,
        )
        return 1

    print(f"Dependency pin check passed for {len(manifests)} manifest(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
